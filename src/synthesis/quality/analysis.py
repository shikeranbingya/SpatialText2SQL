"""Pluggable SQL analysis helpers for quality control."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from src.synthesis.database.utils import to_text
from src.synthesis.question.features import SQLFeatureExtractor

from .models import ParsedSQL


DISALLOWED_SQL_PATTERN = re.compile(
    r"\b(drop|delete|update|insert|alter|truncate|create|grant|revoke|comment|copy|vacuum|analyze|refresh|"
    r"begin|commit|rollback|savepoint|release|explain|call|do|set|reset|prepare|deallocate|listen|notify)\b",
    re.I,
)


def contains_disallowed_sql(sql: str) -> bool:
    return DISALLOWED_SQL_PATTERN.search(sql or "") is not None


def split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_string = False
    escape = False
    for char in sql:
        if char == "\\" and not escape:
            escape = True
            current.append(char)
            continue
        if char == "'" and not escape:
            in_string = not in_string
        if char == ";" and not in_string:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            continue
        current.append(char)
        escape = False
    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    return statements


def strip_string_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'", "''", sql)


def split_top_level_commas(text: str) -> list[str]:
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


def extract_function_calls(sql: str) -> dict[str, list[str]]:
    cleaned = strip_string_literals(sql)
    calls: dict[str, list[str]] = {}
    for match in re.finditer(r"\b(ST_[A-Za-z0-9_]+)\s*\(", cleaned, re.I):
        name = match.group(1).upper()
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
        calls.setdefault(name, []).append(cleaned[start : index - 1].strip())
    return calls


def detect_tables_and_aliases(sql: str) -> tuple[list[str], dict[str, str]]:
    pattern = re.compile(
        r"\b(?:from|join)\s+([a-zA-Z_][\w\.]*)(?:\s+(?:as\s+)?([a-zA-Z_][\w]*))?",
        re.I,
    )
    tables: list[str] = []
    aliases: dict[str, str] = {}
    for match in pattern.finditer(sql):
        table_name = match.group(1).split(".")[-1]
        alias = to_text(match.group(2))
        if table_name.lower() == "select":
            continue
        tables.append(table_name)
        if alias:
            aliases[alias] = table_name
    return sorted(set(tables)), aliases


def detect_columns(sql: str, aliases: dict[str, str]) -> list[str]:
    columns: list[str] = []
    for alias, column in re.findall(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b", sql):
        if alias.lower().startswith("st_"):
            continue
        if alias in aliases or alias in aliases.values():
            columns.append(column)
    return sorted(set(columns))


class SQLAnalyzer(Protocol):
    def analyze(self, sql: str) -> ParsedSQL:
        ...


@dataclass
class DefaultSQLAnalyzer:
    feature_extractor: SQLFeatureExtractor = field(default_factory=SQLFeatureExtractor)

    def analyze(self, sql: str) -> ParsedSQL:
        sql_text = to_text(sql)
        statements = split_sql_statements(sql_text)
        features = self.feature_extractor.extract(sql_text)
        tables, aliases = detect_tables_and_aliases(sql_text)
        columns = detect_columns(sql_text, aliases)
        string_literals = [match.group(1) for match in re.finditer(r"'((?:''|[^'])*)'", sql_text)]
        comparison_operators = re.findall(r"(<=|>=|<>|!=|=|<|>)", strip_string_literals(sql_text))
        return ParsedSQL(
            sql=sql_text,
            statement_count=len(statements),
            tables=tables or features.tables,
            columns=columns or features.columns,
            aliases=aliases,
            postgis_functions=list(features.postgis_functions),
            aggregates=list(features.aggregates),
            group_by_columns=list(features.group_by_columns),
            order_by=list(features.order_by),
            limit=features.limit,
            spatial_predicates=list(features.spatial_predicates),
            distance_thresholds=list(features.distance_thresholds),
            filters=list(features.filters),
            has_cte=features.has_cte,
            has_subquery=features.has_subquery,
            string_literals=string_literals,
            comparison_operators=comparison_operators,
            function_calls=extract_function_calls(sql_text),
        )
