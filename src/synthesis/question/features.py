"""SQL feature extraction for diversity-aware question generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.synthesis.database.utils import to_text

from .models import SQLFeatureSummary

import sqlglot
from sqlglot import exp


AGGREGATE_FUNCTIONS = {"count", "sum", "avg", "min", "max"}
SPATIAL_PREDICATE_FUNCTIONS = {
    "st_contains",
    "st_within",
    "st_intersects",
    "st_dwithin",
    "st_touches",
    "st_crosses",
    "st_overlaps",
    "st_covers",
    "st_coveredby",
    "st_disjoint",
}


def _strip_string_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'", "''", sql)


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            value = "".join(current).strip()
            if value:
                parts.append(value)
            current = []
            continue
        current.append(char)
    trailing = "".join(current).strip()
    if trailing:
        parts.append(trailing)
    return parts


def _extract_function_calls(sql: str) -> dict[str, list[str]]:
    cleaned = _strip_string_literals(sql)
    calls: dict[str, list[str]] = {}
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
        calls.setdefault(name.upper(), []).append(inner)
    return calls


def _extract_numeric_literals(text: str) -> list[str]:
    return [match.group(0) for match in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?", text)]


@dataclass
class SQLFeatureExtractor:
    def extract(self, sql: str) -> SQLFeatureSummary:
        sql_text = to_text(sql)
        if not sql_text:
            return SQLFeatureSummary()

        tables, columns, group_by_columns, order_by, limit = self._extract_schema_features(sql_text)
        functions = sorted(set(match.group(1).upper() for match in re.finditer(r"\b(ST_[A-Za-z0-9_]+)\s*\(", sql_text, re.I)))
        aggregates = self._extract_aggregates(sql_text)
        spatial_predicates = [name for name in functions if name.lower() in SPATIAL_PREDICATE_FUNCTIONS]
        distance_thresholds = self._extract_distance_thresholds(sql_text)
        filters = self._extract_filters(sql_text)
        lowered = sql_text.lower()
        return SQLFeatureSummary(
            tables=tables,
            columns=columns,
            postgis_functions=functions,
            aggregates=aggregates,
            group_by_columns=group_by_columns,
            order_by=order_by,
            limit=limit,
            spatial_predicates=spatial_predicates,
            distance_thresholds=distance_thresholds,
            filters=filters,
            has_cte=lowered.lstrip().startswith("with "),
            has_subquery=bool(re.search(r"\(\s*select\b", lowered)),
        )

    def _extract_schema_features(
        self,
        sql_text: str,
    ) -> tuple[list[str], list[str], list[str], list[dict[str, str]], int | None]:
        try:  # pragma: no cover - optional dependency path
            expression = sqlglot.parse_one(sql_text, read="postgres")
            tables = []
            columns = []
            group_by_columns = []
            order_by: list[dict[str, str]] = []
            limit = None
            for table in expression.find_all(exp.Table):
                table_name = to_text(table.name)
                if table_name:
                    tables.append(table_name)
            for column in expression.find_all(exp.Column):
                column_name = to_text(column.name)
                if column_name:
                    columns.append(column_name)
            group = expression.args.get("group")
            if group is not None:
                for item in group.expressions:
                    column_name = to_text(getattr(item, "name", None) or item.sql())
                    if column_name:
                        group_by_columns.append(column_name)
            order = expression.args.get("order")
            if order is not None:
                for item in order.expressions:
                    direction = "desc" if getattr(item, "desc", False) else "asc"
                    column_name = to_text(getattr(item.this, "name", None) or item.this.sql())
                    if column_name:
                        order_by.append({"column": column_name, "direction": direction})
            limit_exp = expression.args.get("limit")
            if limit_exp is not None:
                limit_text = to_text(getattr(limit_exp, "expression", None) or getattr(limit_exp, "this", None) or limit_exp.sql())
                numeric = _extract_numeric_literals(limit_text)
                if numeric:
                    limit = int(float(numeric[0]))
            return (
                sorted(set(tables)),
                sorted(set(columns)),
                sorted(set(group_by_columns)),
                order_by,
                limit,
            )
        except Exception:
            return self._extract_schema_features_regex(sql_text)

    def _extract_schema_features_regex(
        self,
        sql_text: str,
    ) -> tuple[list[str], list[str], list[str], list[dict[str, str]], int | None]:
        tables = []
        for match in re.finditer(r"\b(?:from|join)\s+([a-zA-Z_][\w\.]*)", sql_text, re.I):
            tables.append(match.group(1).split(".")[-1])
        columns = []
        for match in re.finditer(r"\b[a-zA-Z_][\w]*\.([a-zA-Z_][\w]*)\b", sql_text):
            columns.append(match.group(1))
        group_by_columns: list[str] = []
        group_match = re.search(r"\bgroup\s+by\s+(.+?)(?:\border\b|\blimit\b|$)", sql_text, re.I | re.S)
        if group_match:
            for item in _split_top_level_commas(group_match.group(1)):
                column_name = item.split(".")[-1].strip().strip('"')
                if column_name:
                    group_by_columns.append(column_name)
        order_by: list[dict[str, str]] = []
        order_match = re.search(r"\border\s+by\s+(.+?)(?:\blimit\b|$)", sql_text, re.I | re.S)
        if order_match:
            for item in _split_top_level_commas(order_match.group(1)):
                lowered = item.lower()
                direction = "desc" if lowered.endswith(" desc") else "asc"
                cleaned = re.sub(r"\s+(asc|desc)\s*$", "", item, flags=re.I)
                column_name = cleaned.split(".")[-1].strip().strip('"')
                if column_name:
                    order_by.append({"column": column_name, "direction": direction})
        limit = None
        limit_match = re.search(r"\blimit\s+(\d+)", sql_text, re.I)
        if limit_match:
            limit = int(limit_match.group(1))
        return (
            sorted(set(tables)),
            sorted(set(columns)),
            sorted(set(group_by_columns)),
            order_by,
            limit,
        )

    @staticmethod
    def _extract_aggregates(sql_text: str) -> list[str]:
        cleaned = _strip_string_literals(sql_text)
        aggregates: list[str] = []
        for function_name in AGGREGATE_FUNCTIONS:
            if re.search(rf"\b{function_name}\s*\(", cleaned, re.I):
                aggregates.append(function_name.upper())
        return sorted(set(aggregates))

    @staticmethod
    def _extract_distance_thresholds(sql_text: str) -> list[str]:
        thresholds: list[str] = []
        for function_name, argument_texts in _extract_function_calls(sql_text).items():
            if function_name.lower() == "st_dwithin":
                for args_text in argument_texts:
                    args = _split_top_level_commas(args_text)
                    if len(args) >= 3:
                        numeric = _extract_numeric_literals(args[2])
                        if numeric:
                            thresholds.append(numeric[0])
            elif function_name.lower() == "st_buffer":
                for args_text in argument_texts:
                    args = _split_top_level_commas(args_text)
                    if len(args) >= 2:
                        numeric = _extract_numeric_literals(args[1])
                        if numeric:
                            thresholds.append(numeric[0])
        return thresholds

    @staticmethod
    def _extract_filters(sql_text: str) -> list[str]:
        where_match = re.search(
            r"\bwhere\b(.+?)(?:\bgroup\b|\border\b|\blimit\b|$)",
            sql_text,
            re.I | re.S,
        )
        if not where_match:
            return []
        where_text = " ".join(where_match.group(1).strip().split())
        if not where_text:
            return []
        parts = re.split(r"\b(?:and|or)\b", where_text, flags=re.I)
        return [part.strip() for part in parts if part.strip()]
