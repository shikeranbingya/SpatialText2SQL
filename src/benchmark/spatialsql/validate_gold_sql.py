#!/usr/bin/env python3
"""
Validate all spatialsql_pg gold SQL queries on PostgreSQL.

Usage:
    python -m src.benchmark.spatialsql.validate_gold_sql [--utils-first] [--connect-timeout SEC]

Outputs:
    - Per-query execution status on stdout
    - scripts/benchmark/spatialsql/gold_validation_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psycopg2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)


def _connection_timeout_sec(db_config: dict, override: int | None) -> int:
    if override is not None and override > 0:
        return override
    t = db_config.get("timeout")
    if isinstance(t, dict):
        try:
            return max(1, int(t.get("connection_timeout", 10)))
        except (TypeError, ValueError):
            pass
    return 10


def load_preprocessed_items() -> list[dict]:
    """Load all preprocessed spatialsql_pg items."""
    prep_dir = REPO_ROOT / "data" / "preprocessed" / "spatialsql_pg"
    items = []
    for json_file in sorted(prep_dir.rglob("*.json")):
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
    timeout_ms: int = 30000,
    connect_timeout_sec: int = 10,
):
    """Execute all gold SQL queries on PostgreSQL and summarize the results."""
    results = {
        "total": 0,
        "gold_success": 0,
        "gold_failed": 0,
        "details": [],
    }

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
        source_id = item.get("metadata", {}).get("source_id", "?")
        gold_sql = item.get("gold_sql", "")
        candidates = item.get("gold_sql_candidates", [])

        all_sqls = [gold_sql] + [c for c in candidates if c != gold_sql]
        results["total"] += 1
        entry = {
            "id": item_id,
            "source_id": source_id,
            "split": item.get("metadata", {}).get("split", ""),
            "gold_sql": gold_sql,
            "status": "unknown",
            "errors": [],
        }

        any_success = False
        for sql in all_sqls:
            if not sql or not sql.strip():
                continue
            try:
                cur = conn.cursor()
                cur.execute(sql)
                cur.fetchall()
                cur.close()
                any_success = True
                break
            except Exception as e:
                entry["errors"].append({"sql": sql[:300], "error": str(e)[:200]})
                conn.rollback()

        if any_success:
            results["gold_success"] += 1
            entry["status"] = "ok"
        else:
            results["gold_failed"] += 1
            entry["status"] = "failed"

        results["details"].append(entry)

        status_char = "." if any_success else "X"
        print(status_char, end="", flush=True)
        if results["total"] % 50 == 0:
            print(f"  [{results['total']}]")

    conn.close()
    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate gold SQL on PostgreSQL")
    parser.add_argument("--utils-first", action="store_true",
                        help="Re-run preprocessing before validation")
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=None,
        metavar="SEC",
        help="TCP connect timeout in seconds. Defaults to databases.spatial_sql.timeout.connection_timeout or 10.",
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
        preprocessor.preprocess("spatialsql_pg")
        print()

    with open(REPO_ROOT / "config" / "db_config.yaml", "r", encoding="utf-8") as f:
        db_cfg_all = yaml.safe_load(f)
    db_config = db_cfg_all.get("databases", {}).get("spatial_sql", db_cfg_all.get("database", {}))

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
        results = validate_all(items, db_config, connect_timeout_sec=ct)
    except Exception as e:
        err = str(e).lower()
        if any(
            x in err
            for x in (
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
                "  - For local debugging, update databases.spatial_sql.host in config/db_config.yaml.\n"
                "  - You can also use an SSH tunnel: ssh -L 5432:127.0.0.1:5432 user@jump-host and then set host to 127.0.0.1.\n"
                f"\nOriginal error: {e}\n",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Gold SQL Validation Results")
    print(f"{'='*60}")
    print(f"Total:   {results['total']}")
    print(f"Success: {results['gold_success']} ({results['gold_success']/max(results['total'],1)*100:.1f}%)")
    print(f"Failed:  {results['gold_failed']} ({results['gold_failed']/max(results['total'],1)*100:.1f}%)")
    print(f"Time:    {elapsed:.1f}s")

    if results["gold_failed"] > 0:
        print(f"\nFailed items:")
        for d in results["details"]:
            if d["status"] == "failed":
                err_msg = d["errors"][0]["error"][:80] if d["errors"] else "?"
                print(f"  [{d['source_id']}] {d['gold_sql'][:60]}... -> {err_msg}")

    report_path = REPO_ROOT / "scripts" / "benchmark" / "spatialsql" / "gold_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report: {report_path}")


if __name__ == "__main__":
    main()
