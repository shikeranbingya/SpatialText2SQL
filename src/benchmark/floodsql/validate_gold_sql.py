#!/usr/bin/env python3
"""
Validate all FloodSQL gold SQL queries on PostgreSQL/PostGIS.

Usage:
    python -m src.benchmark.floodsql.validate_gold_sql [--utils-first] [--connect-timeout SEC]

Outputs:
    - Per-query validation progress on stdout
    - scripts/benchmark/floodsql/gold_validation_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)


def _connection_timeout_sec(db_config: dict, override: int | None) -> int:
    if override is not None and override > 0:
        return override
    timeout_cfg = db_config.get("timeout")
    if isinstance(timeout_cfg, dict):
        try:
            return max(1, int(timeout_cfg.get("connection_timeout", 10)))
        except (TypeError, ValueError):
            pass
    return 10


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def _values_equal(left: Any, right: Any, tolerance: float) -> bool:
    left = _normalize_value(left)
    right = _normalize_value(right)
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_equal(l_item, r_item, tolerance)
            for l_item, r_item in zip(left, right)
        )
    return left == right


def _stable_sort_key(value: Any) -> str:
    normalized = _normalize_value(value)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _rows_equal_ignoring_order(left: Any, right: Any, tolerance: float) -> bool:
    left = _normalize_value(left)
    right = _normalize_value(right)
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    if len(left) != len(right):
        return False

    left_sorted = sorted(left, key=_stable_sort_key)
    right_sorted = sorted(right, key=_stable_sort_key)
    return all(
        _values_equal(left_item, right_item, tolerance)
        for left_item, right_item in zip(left_sorted, right_sorted)
    )


def _split_top_level_expressions(expr_text: str) -> list[str]:
    args: list[str] = []
    buf: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    for ch in expr_text:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                args.append("".join(buf).strip())
                buf = []
                continue
        buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    return [arg for arg in args if arg]


def _single_column_rows(rows: Any) -> bool:
    return (
        isinstance(rows, list)
        and all(isinstance(row, list) and len(row) == 1 for row in rows)
    )


def _is_probably_nondeterministic_topk_query(sql: str) -> bool:
    normalized = " ".join((sql or "").strip().split())
    lowered = normalized.lower()
    order_pos = lowered.rfind(" order by ")
    if order_pos < 0 or " limit " not in lowered[order_pos:]:
        return False

    limit_pos = lowered.find(" limit ", order_pos)
    if limit_pos < 0:
        return False

    order_clause = normalized[order_pos + len(" order by ") : limit_pos].strip()
    if not order_clause:
        return False

    order_items = _split_top_level_expressions(order_clause)
    if len(order_items) != 1:
        return False

    order_expr = order_items[0]
    if re.search(r"\bnulls\s+(first|last)\b", order_expr, re.I):
        order_expr = re.sub(r"\bnulls\s+(first|last)\b", "", order_expr, flags=re.I).strip()
    order_expr = re.sub(r"\basc\b|\bdesc\b", "", order_expr, flags=re.I).strip()

    if not order_expr:
        return False

    has_grouping = " group by " in lowered or "select distinct " in lowered
    aggregate_order = bool(re.search(r"\b(count|sum|avg|min|max)\s*\(", order_expr, re.I))
    computed_order = any(token in order_expr for token in ("+", "-", "*", "/"))
    return has_grouping and (aggregate_order or computed_order or order_expr.startswith("("))


def load_preprocessed_items() -> list[dict]:
    """Load all preprocessed floodsql_pg items."""
    prep_dir = REPO_ROOT / "data" / "preprocessed" / "floodsql_pg"
    items: list[dict] = []
    for json_file in sorted(prep_dir.glob("*.json")):
        if json_file.name != "samples.json" and not json_file.name.endswith("_samples.json"):
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            items.extend(data)
        elif isinstance(data, dict) and "items" in data:
            items.extend(data["items"])
    return items


def validate_all(
    items: list[dict],
    db_config: dict,
    *,
    timeout_ms: int = 60000,
    connect_timeout_sec: int = 10,
    float_tolerance: float = 1e-6,
) -> dict:
    """Execute all FloodSQL gold SQL queries on PostgreSQL and validate the results."""
    results = {
        "dataset": "floodsql_pg",
        "database": db_config["database"],
        "total": 0,
        "gold_success": 0,
        "gold_failed": 0,
        "exact_match": 0,
        "order_insensitive_match": 0,
        "nondeterministic_topk_match": 0,
        "row_count_mismatch": 0,
        "value_mismatch": 0,
        "execution_error": 0,
        "details": [],
    }

    from src.sql.sql_dialect_adapter import convert_duckdb_to_postgis

    conn = psycopg2.connect(
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["database"],
        user=db_config["user"],
        password=db_config["password"],
        connect_timeout=connect_timeout_sec,
        options=f"-c statement_timeout={timeout_ms}",
    )

    for item in items:
        item_id = item.get("id", "?")
        metadata = item.get("metadata", {})
        source_id = metadata.get("source_id", item_id)
        official_row_count = metadata.get("official_row_count")
        official_result = metadata.get("official_result")
        translated_sql, issues = convert_duckdb_to_postgis(item.get("gold_sql", ""))

        entry = {
            "id": item_id,
            "source_id": source_id,
            "level": metadata.get("level", ""),
            "family": metadata.get("family", ""),
            "gold_sql": item.get("gold_sql", ""),
            "translated_sql": translated_sql,
            "issues": issues,
            "status": "unknown",
            "row_count": None,
            "result_preview": [],
            "error": None,
        }
        results["total"] += 1

        try:
            cur = conn.cursor()
            cur.execute(translated_sql)
            rows = cur.fetchall()
            cur.close()

            normalized_rows = [_normalize_value(list(row)) for row in rows]
            entry["row_count"] = len(rows)
            entry["result_preview"] = normalized_rows[:5]

            if official_row_count is not None and len(rows) != official_row_count:
                entry["status"] = "row_count_mismatch"
                results["row_count_mismatch"] += 1
                results["gold_failed"] += 1
            elif official_result is not None and not _values_equal(
                normalized_rows,
                official_result,
                float_tolerance,
            ):
                if _rows_equal_ignoring_order(normalized_rows, official_result, float_tolerance):
                    entry["status"] = "order_insensitive_match"
                    results["order_insensitive_match"] += 1
                    results["gold_success"] += 1
                elif (
                    _single_column_rows(normalized_rows)
                    and _single_column_rows(official_result)
                    and len(normalized_rows) == len(official_result)
                    and _is_probably_nondeterministic_topk_query(translated_sql)
                ):
                    entry["status"] = "nondeterministic_topk_match"
                    results["nondeterministic_topk_match"] += 1
                    results["gold_success"] += 1
                else:
                    entry["status"] = "value_mismatch"
                    results["value_mismatch"] += 1
                    results["gold_failed"] += 1
            else:
                entry["status"] = "ok"
                results["exact_match"] += 1
                results["gold_success"] += 1
        except Exception as exc:
            entry["status"] = "execution_error"
            entry["error"] = str(exc)
            results["execution_error"] += 1
            results["gold_failed"] += 1
            conn.rollback()

        results["details"].append(entry)

        status_char = "." if entry["status"] in {"ok", "order_insensitive_match", "nondeterministic_topk_match"} else "X"
        print(status_char, end="", flush=True)
        if results["total"] % 50 == 0:
            print(f"  [{results['total']}]")

    conn.close()
    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate FloodSQL gold SQL on PostgreSQL")
    parser.add_argument(
        "--utils-first",
        action="store_true",
        help="Re-run preprocessing before validation",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=None,
        metavar="SEC",
        help="TCP connect timeout in seconds. Defaults to databases.floodsql.timeout.connection_timeout or 10.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=60000,
        help="PostgreSQL statement_timeout in milliseconds",
    )
    parser.add_argument(
        "--float-tolerance",
        type=float,
        default=1e-6,
        help="Floating-point comparison tolerance",
    )
    parser.add_argument(
        "--report",
        default="scripts/benchmark/floodsql/gold_validation_report.json",
        help="Report output path. Relative paths are resolved from the repository root.",
    )
    args = parser.parse_args()

    if args.preprocess_first:
        print("=== Re-running preprocessing ===")
        from src.datasets.processing import DataPreprocessor

        cfg_dir = REPO_ROOT / "config"
        preprocessor = DataPreprocessor(
            dataset_config_path=str(cfg_dir / "dataset_config.yaml"),
            db_config_path=str(cfg_dir / "db_config.yaml"),
        )
        preprocessor.preprocess("floodsql_pg")
        print()

    with open(REPO_ROOT / "config" / "db_config.yaml", "r", encoding="utf-8") as f:
        db_cfg_all = yaml.safe_load(f)
    db_config = db_cfg_all.get("databases", {}).get("floodsql", db_cfg_all.get("database", {}))

    items = load_preprocessed_items()
    if not items:
        print("No preprocessed items found. Run with --utils-first or ensure data exists.")
        return

    print(f"Loaded {len(items)} gold SQL items")
    ct = _connection_timeout_sec(db_config, args.connect_timeout)
    print(
        f"Connecting to {db_config['host']}:{db_config['port']}/{db_config['database']} "
        f"(connect_timeout={ct}s)"
    )
    print()

    t0 = time.time()
    try:
        results = validate_all(
            items,
            db_config,
            timeout_ms=args.timeout_ms,
            connect_timeout_sec=ct,
            float_tolerance=args.float_tolerance,
        )
    except Exception as exc:
        err = str(exc).lower()
        if any(
            token in err
            for token in (
                "timeout",
                "timed out",
                "could not connect",
                "connection refused",
                "no route to host",
            )
        ):
            print(
                "\nFailed to connect to PostgreSQL (timeout or unreachable). Common causes:\n"
                "  - The host is only reachable on VPN or an internal network, or the server is down.\n"
                "  - For local debugging, update databases.floodsql.host in config/db_config.yaml.\n"
                "  - You can also use an SSH tunnel: ssh -L 5432:127.0.0.1:5432 user@jump-host and then set host to 127.0.0.1.\n"
                f"\nOriginal error: {exc}\n",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print("Gold SQL Validation Results")
    print(f"{'=' * 60}")
    print(f"Total:   {results['total']}")
    print(f"Success: {results['gold_success']} ({results['gold_success'] / max(results['total'], 1) * 100:.1f}%)")
    print(f"Failed:  {results['gold_failed']} ({results['gold_failed'] / max(results['total'], 1) * 100:.1f}%)")
    print(f"Time:    {elapsed:.1f}s")
    print(f"Exact:   {results['exact_match']}")
    print(f"OrderOK: {results['order_insensitive_match']}")
    print(f"TieOK:   {results['nondeterministic_topk_match']}")
    print(f"RowCnt:  {results['row_count_mismatch']}")
    print(f"Value:   {results['value_mismatch']}")
    print(f"ExecErr: {results['execution_error']}")

    if results["gold_failed"] > 0:
        print("\nFailed items:")
        for detail in results["details"]:
            if detail["status"] in {"ok", "order_insensitive_match", "nondeterministic_topk_match"}:
                continue
            if detail["status"] == "execution_error":
                reason = (detail.get("error") or "?")[:80]
            else:
                reason = detail["status"]
            print(f"  [{detail['source_id']}] {detail['translated_sql'][:60]}... -> {reason}")

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report: {report_path}")


if __name__ == "__main__":
    main()
