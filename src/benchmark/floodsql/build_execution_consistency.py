#!/usr/bin/env python3
"""
Build a FloodSQL source-vs-target execution consistency report.

Outputs:
    - scripts/benchmark/floodsql/execution_consistency_report.json
    - scripts/benchmark/floodsql/consistency_clusters.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import psycopg2
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)

from src.datasets.loaders.floodsql_loader import FloodSQLLoader, _resolve_benchmark_root
from src.sql.floodsql_migration import EXPECTED_TABLES, _load_metadata, discover_floodsql_data_layout
from src.sql.sql_dialect_adapter import convert_duckdb_to_postgis


SPATIAL_FUNCTION_PATTERN = re.compile(r"\bST_[A-Za-z0-9_]+\s*\(", re.I)


def _log(message: str) -> None:
    print(message, flush=True)


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[no-work]"
    ratio = max(0.0, min(1.0, current / total))
    filled = int(width * ratio)
    return f"[{'#' * filled}{'.' * (width - filled)}] {current}/{total} ({ratio * 100:.1f}%)"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_db_config(db_config_path: Path) -> Dict[str, Any]:
    with open(db_config_path, "r", encoding="utf-8") as f:
        db_cfg = yaml.safe_load(f) or {}
    return db_cfg.get("databases", {}).get("floodsql", db_cfg.get("database", {}))


def _load_dataset_cfg(dataset_config_path: Path) -> Dict[str, Any]:
    with open(dataset_config_path, "r", encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f) or {}
    return dataset_cfg.get("datasets", {}).get("floodsql_pg", {})


def load_floodsql_items(dataset_config_path: Path) -> List[Dict[str, Any]]:
    dataset_cfg = _load_dataset_cfg(dataset_config_path)
    loader = FloodSQLLoader(dataset_cfg)
    raw_data = loader.load_raw_data(dataset_cfg.get("data_path", "../FloodSQL-Bench"))
    return loader.extract_questions_and_sqls(raw_data)


def resolve_floodsql_paths(
    dataset_config_path: Path,
    *,
    benchmark_root: Optional[str] = None,
    data_root: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    dataset_cfg = _load_dataset_cfg(dataset_config_path)
    benchmark_root_path = (
        Path(benchmark_root).expanduser().resolve()
        if benchmark_root
        else _resolve_benchmark_root(dataset_cfg.get("data_path", "../FloodSQL-Bench"))
    )
    data_root_path = (
        Path(data_root).expanduser().resolve()
        if data_root
        else (benchmark_root_path / "data").resolve()
    )
    metadata_file = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path
        else (data_root_path / "metadata_parquet.json").resolve()
    )
    return benchmark_root_path, data_root_path, metadata_file


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, Decimal):
        return round(float(value), 6)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str):
        return value.strip()
    return value


def _normalize_rows(rows: Iterable[tuple]) -> List[Tuple[Any, ...]]:
    return [tuple(_normalize_scalar(value) for value in row) for row in rows]


def compare_sql_results(source_rows: Iterable[tuple], target_rows: Iterable[tuple]) -> Tuple[str, Dict[str, Any]]:
    normalized_source = _normalize_rows(source_rows)
    normalized_target = _normalize_rows(target_rows)
    if normalized_source == normalized_target:
        return "exact_match", {}

    if sorted(normalized_source, key=repr) == sorted(normalized_target, key=repr):
        return "format_difference", {
            "source_count": len(normalized_source),
            "target_count": len(normalized_target),
        }

    return "semantic_mismatch", {
        "source_count": len(normalized_source),
        "target_count": len(normalized_target),
        "only_in_source": [repr(row) for row in sorted(set(normalized_source) - set(normalized_target), key=repr)[:10]],
        "only_in_target": [repr(row) for row in sorted(set(normalized_target) - set(normalized_source), key=repr)[:10]],
    }


def _looks_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal))


def _split_top_level_expressions(expr_text: str) -> List[str]:
    args: List[str] = []
    buf: List[str] = []
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
    return isinstance(rows, list) and all(isinstance(row, tuple) and len(row) == 1 for row in rows)


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


def classify_floodsql_failure(
    error_message: Optional[str] = None,
    issues: Optional[Iterable[str]] = None,
) -> str:
    text = (error_message or "").lower()
    issue_text = " ".join(issues or []).lower()
    merged = f"{text} {issue_text}".strip()

    if not merged:
        return "execution_error"
    if "spatial" in merged and ("extension" in merged or "install" in merged or "load" in merged):
        return "source_runtime_setup_error"
    if "catalog error" in merged or "does not exist" in merged or "no such" in merged or "not found" in merged:
        return "sql_mapping_error"
    if "must appear in the group by" in merged or "aggregate" in merged:
        return "sql_aggregate_error"
    if "syntax error" in merged or "parser error" in merged or "possible unconverted function" in merged:
        return "sql_rule_gap"
    if "st_" in merged and "function" in merged:
        return "spatial_function_setup_error"
    return "execution_error"


def classify_floodsql_mismatch(
    detail: Dict[str, Any],
    comparison_details: Optional[Dict[str, Any]] = None,
) -> str:
    comparison = comparison_details or detail.get("comparison_details") or {}
    source_count = comparison.get("source_count")
    target_count = comparison.get("target_count")
    only_in_source = comparison.get("only_in_source") or []
    only_in_target = comparison.get("only_in_target") or []
    source_sql = detail.get("source_sql") or ""
    target_sql = detail.get("target_sql") or ""

    if (
        source_count == target_count
        and _is_probably_nondeterministic_topk_query(source_sql)
        and len(only_in_source) == len(only_in_target)
    ):
        parsed_source: List[Any] = []
        parsed_target: List[Any] = []
        for values in only_in_source[:5]:
            try:
                parsed_source.append(eval(values, {"__builtins__": {}}))
            except Exception:
                parsed_source.append(values)
        for values in only_in_target[:5]:
            try:
                parsed_target.append(eval(values, {"__builtins__": {}}))
            except Exception:
                parsed_target.append(values)
        if _single_column_rows(parsed_source) and _single_column_rows(parsed_target):
            return "nondeterministic_topk_difference"

    if source_count != target_count:
        return "result_scope_difference"

    parsed_rows = []
    for values in (only_in_source[:3] + only_in_target[:3]):
        try:
            parsed_rows.append(eval(values, {"__builtins__": {}}))
        except Exception:
            parsed_rows.append(values)

    source_parsed = []
    for values in only_in_source[:3]:
        try:
            source_parsed.append(eval(values, {"__builtins__": {}}))
        except Exception:
            source_parsed.append(values)
    target_parsed = []
    for values in only_in_target[:3]:
        try:
            target_parsed.append(eval(values, {"__builtins__": {}}))
        except Exception:
            target_parsed.append(values)

    tuple_widths = {len(row) for row in parsed_rows if isinstance(row, tuple)}
    if len(tuple_widths) > 1:
        return "result_structure_difference"

    if tuple_widths and next(iter(tuple_widths), 0) > 1:
        for left, right in zip(source_parsed, target_parsed):
            if not (isinstance(left, tuple) and isinstance(right, tuple)):
                continue
            if len(left) != len(right):
                return "result_structure_difference"
            differing_positions = [idx for idx, (lval, rval) in enumerate(zip(left, right)) if lval != rval]
            if not differing_positions:
                continue
            if all(_looks_numeric(left[idx]) and _looks_numeric(right[idx]) for idx in differing_positions):
                return "numeric_measurement_difference"
            if len(differing_positions) == 1:
                last_idx = len(left) - 1
                if differing_positions[0] == last_idx and _looks_numeric(left[last_idx]) and _looks_numeric(right[last_idx]):
                    return "numeric_measurement_difference"
        return "result_structure_difference"

    scalar_values = []
    for row in parsed_rows:
        if isinstance(row, tuple) and len(row) == 1:
            scalar_values.append(row[0])
        elif not isinstance(row, tuple):
            scalar_values.append(row)
    if scalar_values and all(_looks_numeric(value) for value in scalar_values):
        return "numeric_measurement_difference"

    joined_rows = " ".join(map(str, only_in_source[:5] + only_in_target[:5]))
    if re.search(r"[\u4e00-\u9fff]", joined_rows):
        return "label_value_difference"
    if SPATIAL_FUNCTION_PATTERN.search(source_sql) or SPATIAL_FUNCTION_PATTERN.search(target_sql):
        return "spatial_relation_difference"
    return "result_value_difference"


def summarize_consistency_report(report: Dict[str, Any]) -> Dict[str, Any]:
    by_status: Counter = Counter()
    by_classification: Counter = Counter()
    by_level: Dict[str, Counter] = defaultdict(Counter)
    by_family: Dict[str, Counter] = defaultdict(Counter)
    by_mismatch_subtype: Counter = Counter()

    for detail in report.get("details", []):
        status = detail.get("status", "unknown")
        by_status[status] += 1
        level = detail.get("level") or "unknown"
        family = detail.get("family") or "unknown"
        by_level[level][status] += 1
        by_family[family][status] += 1
        classification = detail.get("classification")
        if classification:
            by_classification[classification] += 1
        if status == "semantic_mismatch":
            subtype = detail.get("mismatch_subtype") or "unknown"
            by_mismatch_subtype[subtype] += 1

    summary = dict(report.get("summary", {}))
    summary["by_status"] = dict(sorted(by_status.items()))
    summary["by_classification"] = dict(sorted(by_classification.items()))
    summary["by_level"] = {
        level: dict(sorted(counter.items()))
        for level, counter in sorted(by_level.items())
    }
    summary["by_family"] = {
        family: dict(sorted(counter.items()))
        for family, counter in sorted(by_family.items())
    }
    summary["by_mismatch_subtype"] = dict(sorted(by_mismatch_subtype.items()))
    return summary


def build_consistency_cluster_report(consistency_report: Dict[str, Any]) -> Dict[str, Any]:
    clusters: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    for detail in consistency_report.get("details", []):
        status = detail.get("status")
        if status in {"exact_match", "skipped"}:
            continue

        classification = detail.get("classification") or "unknown"
        mismatch_subtype = detail.get("mismatch_subtype")
        if status == "target_error":
            raw_message = detail.get("target_error") or ""
        elif status == "source_error":
            raw_message = detail.get("source_error") or ""
        else:
            raw_message = ""
        first_line = raw_message.splitlines()[0].strip()[:160] if raw_message else ""
        if status == "semantic_mismatch":
            comparison = detail.get("comparison_details", {})
            fingerprint = (
                f"source={comparison.get('source_count', '?')};"
                f"target={comparison.get('target_count', '?')};"
                f"only_source={len(comparison.get('only_in_source', []) or [])};"
                f"only_target={len(comparison.get('only_in_target', []) or [])}"
            )
        elif status == "format_difference":
            comparison = detail.get("comparison_details", {})
            fingerprint = (
                f"source={comparison.get('source_count', '?')};"
                f"target={comparison.get('target_count', '?')}"
            )
        else:
            fingerprint = first_line or classification

        key = (status, classification, mismatch_subtype or "", fingerprint)
        if key not in clusters:
            clusters[key] = {
                "status": status,
                "classification": classification,
                "mismatch_subtype": mismatch_subtype,
                "fingerprint": fingerprint,
                "count": 0,
                "levels": Counter(),
                "families": Counter(),
                "sample_cases": [],
            }
        cluster = clusters[key]
        cluster["count"] += 1
        cluster["levels"][detail.get("level") or "unknown"] += 1
        cluster["families"][detail.get("family") or "unknown"] += 1
        if len(cluster["sample_cases"]) < 5:
            cluster["sample_cases"].append(
                {
                    "source_id": detail.get("source_id"),
                    "level": detail.get("level"),
                    "family": detail.get("family"),
                    "source_sql": detail.get("source_sql"),
                    "target_sql": detail.get("target_sql"),
                    "source_error": detail.get("source_error"),
                    "target_error": detail.get("target_error"),
                    "comparison_details": detail.get("comparison_details"),
                }
            )

    cluster_rows = []
    for cluster in sorted(clusters.values(), key=lambda item: (-item["count"], item["status"], item["classification"])):
        cluster_rows.append(
            {
                **cluster,
                "levels": dict(sorted(cluster["levels"].items())),
                "families": dict(sorted(cluster["families"].items())),
            }
        )

    return {
        "generated_at_epoch": time.time(),
        "summary": {"cluster_count": len(cluster_rows)},
        "clusters": cluster_rows,
    }


class DuckDBFloodExecutor:
    def __init__(self, data_root: Path, metadata_path: Path):
        self.metadata = _load_metadata(metadata_path)
        self.layout = discover_floodsql_data_layout(data_root, self.metadata)
        self.conn = duckdb.connect(database=":memory:")
        self._load_spatial_extension()
        self._register_tables()

    def _load_spatial_extension(self) -> None:
        last_error: Optional[Exception] = None
        for statement in ("LOAD spatial", "INSTALL spatial", "LOAD spatial"):
            try:
                self.conn.execute(statement)
            except Exception as exc:  # pragma: no cover - runtime dependency
                last_error = exc
                if statement == "INSTALL spatial":
                    continue
                if statement == "LOAD spatial":
                    continue
            else:
                if statement == "LOAD spatial":
                    return
        raise RuntimeError(f"DuckDB spatial extension is unavailable: {last_error}")

    def _register_tables(self) -> None:
        for table_name in EXPECTED_TABLES:
            metadata_info = self.metadata.get(table_name) or {}
            parquet_path = (self.layout.parquet_root / str(metadata_info.get("file"))).resolve()
            parquet_sql = str(parquet_path).replace("\\", "\\\\").replace("'", "''")
            self.conn.execute(
                f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{parquet_sql}')"
            )

    def execute(self, sql: str) -> Dict[str, Any]:
        try:
            rows = self.conn.execute(sql).fetchall()
            return {"status": "ok", "rows": rows}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def close(self) -> None:
        self.conn.close()


class PostgresExecutor:
    def __init__(self, db_cfg: Dict[str, Any], *, timeout_ms: int, connect_timeout_sec: int):
        self.conn = psycopg2.connect(
            host=db_cfg["host"],
            port=db_cfg["port"],
            database=db_cfg["database"],
            user=db_cfg["user"],
            password=db_cfg["password"],
            connect_timeout=connect_timeout_sec,
            options=f"-c statement_timeout={timeout_ms}",
        )

    def execute(self, sql: str) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            return {"status": "ok", "rows": rows}
        except Exception as exc:
            self.conn.rollback()
            return {"status": "error", "error": str(exc)}
        finally:
            cursor.close()

    def close(self) -> None:
        self.conn.close()


def _official_rows_from_item(item: Dict[str, Any]) -> Tuple[str, Optional[List[Tuple[Any, ...]]], Optional[str]]:
    metadata = item.get("metadata", {})
    official_result = metadata.get("official_result")
    official_row_count = metadata.get("official_row_count")
    if official_result is None:
        return "error", None, "missing_official_result"
    if not isinstance(official_result, list):
        return "error", None, "invalid_official_result_type"

    normalized_rows: List[Tuple[Any, ...]] = []
    for row in official_result:
        if isinstance(row, (list, tuple)):
            normalized_rows.append(tuple(row))
        else:
            normalized_rows.append((row,))

    if official_row_count is not None and len(normalized_rows) != official_row_count:
        return "error", None, "official_row_count_mismatch"
    return "ok", normalized_rows, None


def build_execution_consistency_report(
    items: List[Dict[str, Any]],
    target_executor: Any,
    *,
    source_mode: str = "official_results",
    source_executor: Any = None,
) -> Dict[str, Any]:
    report = {
        "generated_at_epoch": time.time(),
        "summary": {
            "total": 0,
            "validated": 0,
            "format_difference": 0,
            "semantic_mismatch": 0,
            "source_errors": 0,
            "target_errors": 0,
            "skipped": 0,
        },
        "details": [],
    }

    _log("=" * 70)
    _log(f"[FloodSQL Consistency] start total_items={len(items)} source_mode={source_mode}")
    if source_mode == "duckdb":
        _log("[FloodSQL Consistency] executes source DuckDB SQL and translated PostgreSQL SQL pairwise")
    else:
        _log("[FloodSQL Consistency] uses official benchmark results as source truth and executes translated PostgreSQL SQL")
    _log("=" * 70)

    for index, item in enumerate(items, start=1):
        report["summary"]["total"] += 1
        metadata = item.get("metadata", {})
        source_sql = item.get("source_sql") or item.get("gold_sql", "")
        target_sql, issues = convert_duckdb_to_postgis(source_sql)
        source_id = metadata.get("source_id") or item.get("id") or "?"
        level = metadata.get("level") or "unknown"
        family = metadata.get("family") or "unknown"

        if index == 1 or index % 25 == 0 or index == len(items):
            _log(
                f"[FloodSQL Consistency] {_progress_bar(index, len(items))} "
                f"current_level={level} current_source_id={source_id} "
                f"validated={report['summary']['validated']} "
                f"source_errors={report['summary']['source_errors']} "
                f"target_errors={report['summary']['target_errors']} "
                f"semantic_mismatch={report['summary']['semantic_mismatch']}"
            )

        if source_mode == "duckdb":
            if source_executor is None:
                raise ValueError("source_executor is required when source_mode=duckdb")
            source_result = source_executor.execute(source_sql)
        else:
            source_status, source_rows, source_error = _official_rows_from_item(item)
            if source_status == "ok":
                source_result = {"status": "ok", "rows": source_rows}
            else:
                source_result = {"status": "error", "error": source_error}
        target_result = target_executor.execute(target_sql)
        detail = {
            "id": item.get("id"),
            "source_id": source_id,
            "level": level,
            "family": family,
            "source_sql": source_sql,
            "target_sql": target_sql,
            "translation_issues": issues,
            "source_mode": source_mode,
        }

        if source_result["status"] != "ok":
            report["summary"]["source_errors"] += 1
            detail["status"] = "source_error"
            if source_mode == "official_results":
                detail["classification"] = "official_result_missing"
            else:
                detail["classification"] = classify_floodsql_failure(error_message=source_result["error"], issues=issues)
            detail["source_error"] = source_result["error"]
            report["details"].append(detail)
            continue
        if target_result["status"] != "ok":
            report["summary"]["target_errors"] += 1
            detail["status"] = "target_error"
            detail["classification"] = classify_floodsql_failure(error_message=target_result["error"], issues=issues)
            detail["target_error"] = target_result["error"]
            report["details"].append(detail)
            continue

        comparison, comparison_details = compare_sql_results(source_result["rows"], target_result["rows"])
        detail["status"] = comparison
        detail["comparison_details"] = comparison_details
        if comparison == "exact_match":
            detail["classification"] = None
            report["summary"]["validated"] += 1
        elif comparison == "format_difference":
            detail["classification"] = "semantic_mismatch"
            detail["mismatch_subtype"] = "format_difference"
            report["summary"]["format_difference"] += 1
        else:
            detail["classification"] = "semantic_mismatch"
            detail["mismatch_subtype"] = classify_floodsql_mismatch(detail, comparison_details)
            report["summary"]["semantic_mismatch"] += 1
        report["details"].append(detail)

    report["summary"] = summarize_consistency_report(report)
    _log(
        f"[FloodSQL Consistency] done total={report['summary']['total']} "
        f"validated={report['summary']['validated']} "
        f"format_difference={report['summary']['format_difference']} "
        f"semantic_mismatch={report['summary']['semantic_mismatch']} "
        f"source_errors={report['summary']['source_errors']} "
        f"target_errors={report['summary']['target_errors']}"
    )
    _log("=" * 70)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FloodSQL execution consistency report")
    parser.add_argument(
        "--dataset-config",
        default=str(REPO_ROOT / "config" / "dataset_config.yaml"),
        help="Path to dataset_config.yaml",
    )
    parser.add_argument(
        "--db-config",
        default=str(REPO_ROOT / "config" / "db_config.yaml"),
        help="Path to db_config.yaml",
    )
    parser.add_argument("--benchmark-root", default=None, help="FloodSQL-Bench root")
    parser.add_argument("--data-root", default=None, help="FloodSQL parquet data root")
    parser.add_argument("--metadata", default=None, help="Path to metadata_parquet.json")
    parser.add_argument(
        "--report",
        default=str(REPO_ROOT / "scripts" / "benchmark" / "floodsql" / "execution_consistency_report.json"),
        help="Consistency report output path",
    )
    parser.add_argument(
        "--clusters",
        default=str(REPO_ROOT / "scripts" / "benchmark" / "floodsql" / "consistency_clusters.json"),
        help="Consistency cluster report output path",
    )
    parser.add_argument(
        "--source-mode",
        choices=["official_results", "duckdb"],
        default="official_results",
        help="Source side comparison mode; official_results is recommended for full FloodSQL runs",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        metavar="SEC",
        help="PostgreSQL connect timeout in seconds",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=60000,
        help="PostgreSQL statement_timeout in milliseconds",
    )
    args = parser.parse_args()

    dataset_config_path = Path(args.dataset_config).expanduser().resolve()
    db_config_path = Path(args.db_config).expanduser().resolve()
    db_cfg = _load_db_config(db_config_path)
    _benchmark_root, data_root, metadata_path = resolve_floodsql_paths(
        dataset_config_path,
        benchmark_root=args.benchmark_root,
        data_root=args.data_root,
        metadata_path=args.metadata,
    )
    items = load_floodsql_items(dataset_config_path)
    if not items:
        raise SystemExit("No FloodSQL samples were loaded. Check the benchmark path first.")

    source_executor = None
    if args.source_mode == "duckdb":
        try:
            source_executor = DuckDBFloodExecutor(data_root, metadata_path)
        except Exception as exc:
            report = {
                "generated_at_epoch": time.time(),
                "summary": {
                    "total": len(items),
                    "validated": 0,
                    "format_difference": 0,
                    "semantic_mismatch": 0,
                    "source_errors": 0,
                    "target_errors": 0,
                    "skipped": len(items),
                    "source_setup_error": str(exc),
                },
                "details": [],
            }
            _write_json(Path(args.report).expanduser().resolve(), report)
            raise SystemExit(f"Failed to initialize the DuckDB source executor: {exc}")

    try:
        target_executor = PostgresExecutor(
            db_cfg,
            timeout_ms=args.timeout_ms,
            connect_timeout_sec=args.connect_timeout,
        )
    except Exception as exc:
        if source_executor is not None:
            source_executor.close()
        report = {
            "generated_at_epoch": time.time(),
            "summary": {
                "total": len(items),
                "validated": 0,
                "format_difference": 0,
                "semantic_mismatch": 0,
                "source_errors": 0,
                "target_errors": 0,
                "skipped": len(items),
                "target_connection_error": str(exc),
            },
            "details": [],
        }
        _write_json(Path(args.report).expanduser().resolve(), report)
        raise SystemExit(f"Failed to connect to PostgreSQL: {exc}")

    try:
        report = build_execution_consistency_report(
            items,
            target_executor,
            source_mode=args.source_mode,
            source_executor=source_executor,
        )
    finally:
        if source_executor is not None:
            source_executor.close()
        target_executor.close()

    cluster_report = build_consistency_cluster_report(report)

    report_path = Path(args.report).expanduser().resolve()
    clusters_path = Path(args.clusters).expanduser().resolve()

    _write_json(report_path, report)
    _write_json(clusters_path, cluster_report)

    _log(f"Detailed report: {report_path}")
    _log(f"Cluster report:  {clusters_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
