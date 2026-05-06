"""Static validation for synthesized PostGIS SQL queries."""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

from src.synthesis.database.models import SynthesizedSpatialDatabase
from src.synthesis.database.utils import to_text

from .function_library import PostGISFunctionLibrary, fixed_spatial_join_function_names
from .models import SQLValidationResult

import sqlglot
from sqlglot import exp


DANGEROUS_SQL_PATTERN = re.compile(
    r"\b(drop|delete|update|insert|alter|truncate|create|grant|revoke|comment|copy|vacuum|analyze|refresh)\b",
    re.I,
)
SQL_KEYWORDS = {
    "select", "from", "where", "join", "left", "right", "inner", "outer", "on", "and", "or", "not",
    "group", "by", "order", "limit", "with", "as", "distinct", "count", "sum", "avg", "min", "max",
    "case", "when", "then", "else", "end", "exists", "in", "union", "intersect", "except", "having",
    "asc", "desc", "is", "null", "like", "between", "true", "false",
}


def contains_dangerous_sql(sql: str) -> bool:
    return DANGEROUS_SQL_PATTERN.search(sql or "") is not None


def _strip_string_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'", "''", sql)


def _split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_string = False
    for char in sql:
        if char == "'":
            in_string = not in_string
        if char == ";" and not in_string:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            continue
        current.append(char)
    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    return statements


def _build_allowed_schema(
    database: SynthesizedSpatialDatabase,
    runtime_metadata: Mapping[str, object] | None = None,
) -> tuple[set[str], dict[str, set[str]]]:
    if isinstance(runtime_metadata, Mapping):
        tables_payload = runtime_metadata.get("tables")
        if isinstance(tables_payload, Sequence) and not isinstance(tables_payload, (str, bytes)):
            allowed_tables: set[str] = set()
            allowed_columns: dict[str, set[str]] = {}
            union_columns: set[str] = set()
            for table_meta in tables_payload:
                if not isinstance(table_meta, Mapping):
                    continue
                table_name = to_text(table_meta.get("table_name"))
                if not table_name:
                    continue
                allowed_tables.add(table_name)
                columns = {
                    to_text(column.get("column_name"))
                    for column in table_meta.get("columns", [])
                    if isinstance(column, Mapping)
                }
                columns = {column for column in columns if column}
                allowed_columns[table_name] = columns
                union_columns.update(columns)
            if allowed_tables:
                allowed_columns["*"] = union_columns
                return allowed_tables, allowed_columns

    allowed_tables: set[str] = set()
    allowed_columns: dict[str, set[str]] = {}
    union_columns: set[str] = set()
    for table in database.selected_tables:
        table_name = to_text(table.table_name)
        allowed_tables.add(table_name)
        columns = {
            to_text(column.get("canonical_name") or column.get("name"))
            for column in table.normalized_schema
            if isinstance(column, Mapping)
        }
        columns = {column for column in columns if column}
        allowed_columns[table_name] = columns
        union_columns.update(columns)
    allowed_columns["*"] = union_columns
    return allowed_tables, allowed_columns


def _detect_tables_regex(sql: str) -> tuple[list[str], dict[str, str]]:
    pattern = re.compile(
        r"\b(?:from|join)\s+([a-zA-Z_][\w\.]*)(?:\s+(?:as\s+)?([a-zA-Z_][\w]*))?",
        re.I,
    )
    tables: list[str] = []
    aliases: dict[str, str] = {}
    for match in pattern.finditer(sql):
        raw_table = match.group(1).split(".")[-1]
        alias = to_text(match.group(2))
        if raw_table.lower() in {"select"}:
            continue
        tables.append(raw_table)
        if alias and alias.lower() not in SQL_KEYWORDS:
            aliases[alias] = raw_table
    return tables, aliases


def _detect_columns_regex(sql: str, aliases: Mapping[str, str]) -> list[str]:
    columns: list[str] = []
    for alias, column in re.findall(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b", sql):
        if alias.lower() in SQL_KEYWORDS:
            continue
        if alias in aliases or alias.lower().startswith("st_"):
            columns.append(column)
    return columns


def _detect_functions(sql: str) -> list[str]:
    return sorted(set(match.group(1) for match in re.finditer(r"\b(ST_[A-Za-z0-9_]+)\s*\(", sql, re.I)))


def _function_call_arg_counts(sql: str) -> dict[str, list[int]]:
    arg_counts: dict[str, list[int]] = {}
    cleaned = _strip_string_literals(sql)
    for match in re.finditer(r"\b(ST_[A-Za-z0-9_]+)\s*\(", cleaned, re.I):
        name = match.group(1)
        start = match.end()
        depth = 1
        index = start
        while index < len(cleaned) and depth > 0:
            char = cleaned[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            index += 1
        if depth != 0:
            continue
        inner = cleaned[start : index - 1].strip()
        if not inner:
            count = 0
        else:
            count = 1
            level = 0
            for char in inner:
                if char == "(":
                    level += 1
                elif char == ")":
                    level -= 1
                elif char == "," and level == 0:
                    count += 1
        arg_counts.setdefault(name.lower(), []).append(count)
    return arg_counts


def _iter_spatial_function_calls(sql: str) -> Iterable[tuple[str, str]]:
    cleaned = _strip_string_literals(sql)
    for match in re.finditer(r"\b(ST_[A-Za-z0-9_]+)\s*\(", cleaned, re.I):
        name = match.group(1)
        start = match.end()
        depth = 1
        index = start
        while index < len(cleaned) and depth > 0:
            char = cleaned[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            index += 1
        if depth != 0:
            continue
        yield name, cleaned[start : index - 1]


def _extract_condition_segments(sql: str) -> list[str]:
    cleaned = _strip_string_literals(sql)
    segments: list[str] = []
    join_pattern = re.compile(
        r"\bjoin\b.*?\bon\b(?P<condition>.*?)(?=\bjoin\b|\bwhere\b|\bgroup\s+by\b|\bhaving\b|\border\s+by\b|\blimit\b|\bunion\b|\bintersect\b|\bexcept\b|$)",
        re.I | re.S,
    )
    where_pattern = re.compile(
        r"\bwhere\b(?P<condition>.*?)(?=\bgroup\s+by\b|\bhaving\b|\border\s+by\b|\blimit\b|\bunion\b|\bintersect\b|\bexcept\b|$)",
        re.I | re.S,
    )
    for pattern in (join_pattern, where_pattern):
        for match in pattern.finditer(cleaned):
            condition = match.group("condition").strip()
            if condition:
                segments.append(condition)
    return segments


def _referenced_tables_in_expression(expression: str, aliases: Mapping[str, str]) -> set[str]:
    referenced: set[str] = set()
    for qualifier, _column in re.findall(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b", expression):
        table_name = aliases.get(qualifier, qualifier)
        referenced.add(table_name)
    return referenced


def _detect_spatial_join_count(
    sql: str,
    aliases: Mapping[str, str],
) -> int:
    count = 0
    for segment in _extract_condition_segments(sql):
        for _name, args in _iter_spatial_function_calls(segment):
            if len(_referenced_tables_in_expression(args, aliases)) >= 2:
                count += 1
    return count


def _detect_difficulty_features(
    sql: str,
    detected_tables: Sequence[str],
    aliases: Mapping[str, str],
) -> dict[str, object]:
    lowered = sql.lower()
    subquery_count = len(re.findall(r"\(\s*select\b", lowered))
    cte_count = 1 if lowered.lstrip().startswith("with ") else 0
    select_count = len(re.findall(r"\bselect\b", lowered))
    return {
        "table_count": len(set(detected_tables)),
        "join_count": len(re.findall(r"\bjoin\b", lowered)),
        "spatial_join_count": _detect_spatial_join_count(sql, aliases),
        "has_group_by": " group by " in f" {lowered} ",
        "has_order_by": " order by " in f" {lowered} ",
        "has_limit": " limit " in f" {lowered} ",
        "has_cte": cte_count > 0,
        "cte_count": cte_count,
        "has_subquery": subquery_count > 0,
        "subquery_count": subquery_count,
        "has_exists": " exists " in f" {lowered} ",
        "has_set_operation": bool(re.search(r"\b(union|intersect|except)\b", lowered)),
        "select_count": select_count,
    }


def _difficulty_matches(target: str, features: Mapping[str, object]) -> tuple[bool, str]:
    table_count = int(features.get("table_count", 0))
    join_count = int(features.get("join_count", 0))
    spatial_join_count = int(features.get("spatial_join_count", 0))
    has_group_by = bool(features.get("has_group_by"))
    has_cte = bool(features.get("has_cte"))
    cte_count = int(features.get("cte_count", 0))
    has_subquery = bool(features.get("has_subquery"))
    subquery_count = int(features.get("subquery_count", 0))
    has_set_operation = bool(features.get("has_set_operation"))
    select_count = int(features.get("select_count", 0))
    if target == "easy":
        if table_count != 1:
            return False, "Easy queries must use exactly one table."
        if join_count > 0 or spatial_join_count > 0 or has_group_by or has_cte or has_subquery:
            return False, "Easy queries must stay a single-table spatial filter or lookup without joins, GROUP BY, CTEs, or subqueries."
        return True, ""
    if target == "medium":
        if table_count != 2:
            return False, "Medium queries must use exactly two tables."
        if spatial_join_count != 1:
            return False, "Medium queries must contain exactly one spatial join between the two tables."
        if join_count > 1:
            return False, "Medium queries should not contain more than one join."
        if has_cte or has_subquery or has_set_operation:
            return False, "Medium queries should stay flat and avoid subqueries, CTEs, and set operations."
        return True, ""
    if target == "hard":
        if table_count != 3:
            return False, "Hard queries must use exactly three tables."
        if spatial_join_count != 2:
            return False, "Hard queries must contain exactly two spatial joins across the three tables."
        if has_cte or has_subquery or has_set_operation:
            return False, "Hard queries should stay flat and avoid nested subqueries, CTEs, and set operations."
        return True, ""
    if target == "extra-hard":
        if table_count < 3 or table_count > 4:
            return False, "Extra-hard queries must use between three and four tables."
        if spatial_join_count < 1:
            return False, "Extra-hard queries must include at least one spatial join."
        nested_query_count = cte_count + subquery_count
        advanced_op_count = spatial_join_count + nested_query_count
        if advanced_op_count < 2 or advanced_op_count > 4:
            return False, "Extra-hard queries must keep spatial joins plus nested queries between two and four operations in total."
        if has_set_operation:
            return False, "Extra-hard queries should avoid UNION/INTERSECT/EXCEPT so the SQL stays executable."
        if select_count > 5:
            return False, "Extra-hard queries should keep the structure bounded to avoid over-complex SQL."
        return True, ""
    return True, ""


class SQLValidator:
    def __init__(self, function_library: PostGISFunctionLibrary):
        self.function_library = function_library

    def validate(
        self,
        *,
        sql: str,
        database: SynthesizedSpatialDatabase,
        sampled_functions: Sequence[str],
        difficulty_level: str,
        database_runtime_metadata: Mapping[str, object] | None = None,
    ) -> SQLValidationResult:
        sql_text = to_text(sql)
        errors: list[str] = []
        warnings: list[str] = []
        if not sql_text:
            errors.append("SQL is empty.")
            return SQLValidationResult(is_valid=False, errors=errors)

        statements = _split_sql_statements(sql_text)
        if len(statements) != 1:
            errors.append("SQL must contain exactly one statement.")
        if contains_dangerous_sql(sql_text):
            errors.append("SQL contains dangerous or non-read-only operations.")
        if not re.match(r"^\s*(select|with)\b", sql_text, re.I):
            errors.append("SQL must be a SELECT or WITH query.")

        detected_tables: list[str]
        detected_columns: list[str]
        aliases: dict[str, str]
        if sqlglot is not None:
            detected_tables, detected_columns, aliases = self._validate_with_sqlglot(sql_text, warnings)
        else:
            detected_tables, aliases = _detect_tables_regex(sql_text)
            detected_columns = _detect_columns_regex(sql_text, aliases)

        allowed_tables, allowed_columns = _build_allowed_schema(database, database_runtime_metadata)
        unknown_tables = [table for table in detected_tables if table not in allowed_tables]
        if unknown_tables:
            errors.append(f"Unknown tables referenced: {', '.join(sorted(set(unknown_tables)))}")

        unknown_columns = [column for column in detected_columns if column not in allowed_columns["*"]]
        if unknown_columns:
            errors.append(f"Unknown columns referenced: {', '.join(sorted(set(unknown_columns)))}")

        detected_functions = _detect_functions(sql_text)
        if not detected_functions:
            errors.append("SQL does not use any PostGIS ST_* function.")
        sampled_lower = {name.lower() for name in sampled_functions}
        if difficulty_level in {"medium", "hard", "extra-hard"}:
            sampled_lower.update(name.lower() for name in fixed_spatial_join_function_names())
        if sampled_lower and not any(func.lower() in sampled_lower for func in detected_functions):
            errors.append("SQL does not use any of the sampled required spatial functions.")
        if sampled_lower:
            unexpected_functions = sorted(
                {
                    func
                    for func in detected_functions
                    if func.lower() not in sampled_lower
                }
            )
            if unexpected_functions:
                errors.append(
                    "SQL uses PostGIS functions outside the externally provided candidate set: "
                    + ", ".join(unexpected_functions)
                )

        raster_topology = [
            func for func in detected_functions
            if any(token in func.lower() for token in ("raster", "topology"))
        ]
        if raster_topology:
            errors.append(f"Raster/topology functions are not allowed: {', '.join(raster_topology)}")

        arg_counts = _function_call_arg_counts(sql_text)
        for function_name, observed_counts in arg_counts.items():
            signatures = self.function_library.get_function_signatures(function_name)
            if not signatures:
                warnings.append(f"Function {function_name} is not present in the filtered PostGIS library.")
                continue
            allowed_counts = {
                len(item.input_args)
                for item in signatures
                if item.input_args
            }
            if allowed_counts and any(count not in allowed_counts for count in observed_counts):
                errors.append(
                    f"Function {function_name} appears to use an incompatible number of arguments."
                )

        difficulty_features = _detect_difficulty_features(sql_text, detected_tables, aliases)
        difficulty_ok, difficulty_message = _difficulty_matches(difficulty_level, difficulty_features)
        if not difficulty_ok:
            errors.append(difficulty_message)

        return SQLValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            detected_tables=sorted(set(detected_tables)),
            detected_columns=sorted(set(detected_columns)),
            detected_spatial_functions=sorted(set(detected_functions)),
            detected_difficulty_features=difficulty_features,
        )

    @staticmethod
    def _validate_with_sqlglot(sql_text: str, warnings: list[str]) -> tuple[list[str], list[str], dict[str, str]]:
        tables: list[str] = []
        columns: list[str] = []
        aliases: dict[str, str] = {}
        try:  # pragma: no cover - optional path
            expression = sqlglot.parse_one(sql_text, read="postgres")
            for table in expression.find_all(exp.Table):
                table_name = to_text(table.name)
                if table_name:
                    tables.append(table_name)
                    alias = to_text(table.alias)
                    if alias:
                        aliases[alias] = table_name
            for column in expression.find_all(exp.Column):
                column_name = to_text(column.name)
                if column_name:
                    columns.append(column_name)
        except Exception as exc:  # pragma: no cover
            warnings.append(f"sqlglot parsing failed; falling back to regex validation: {exc}")
            tables, aliases = _detect_tables_regex(sql_text)
            columns = _detect_columns_regex(sql_text, aliases)
        return tables, columns, aliases
