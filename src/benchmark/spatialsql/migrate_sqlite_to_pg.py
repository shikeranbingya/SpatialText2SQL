#!/usr/bin/env python3
"""
Migrate SpatialSQL SQLite/SpatiaLite databases to PostgreSQL/PostGIS.

Scans sdbdatasets/dataset1|2/<domain>/<domain>.sqlite and imports each database
into a PostgreSQL schema named spatialsql_<version>_<domain>.

Requires GDAL/ogr2ogr for geometry import. Without GDAL, the script only creates
schemas and writes the report files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import psycopg2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VERSIONS = ["dataset1", "dataset2"]
DOMAINS = ["ada", "edu", "tourism", "traffic"]
# Skip SpatiaLite internal tables and views.
SQLITE_SKIP_PATTERN = re.compile(
    r"^(sqlite_|spatial_ref_sys|geometry_columns|spatialite_|SpatialIndex|"
    r"ElementaryGeometries|KNN|virtuoso|views_geometry_columns|"
    r"virts_geometry_columns|geom_cols_ref_sys|geometry_columns_auth|"
    r"spatial_ref_sys_all|sqlite_sequence)$",
    re.I,
)


def load_db_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("database", {})


def pg_connection_string(cfg: dict) -> str:
    return (
        f"PG:host={cfg['host']} port={cfg['port']} dbname={cfg['database']} "
        f"user={cfg['user']} password={cfg['password']}"
    )


def list_sqlite_tables(sqlite_path: str) -> list[str]:
    conn = sqlite3.connect(sqlite_path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = [row[0] for row in cur.fetchall() if row[0]]
    finally:
        conn.close()
    return [n for n in names if not SQLITE_SKIP_PATTERN.match(n)]


def create_pg_schema(cursor, schema_name: str):
    cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')


def run_ogr2ogr(
    sqlite_path: str,
    table: str,
    pg_conn_str: str,
    schema: str,
) -> tuple[bool, str]:
    """Return (success, error_message)."""
    try:
        cmd = [
            "ogr2ogr",
            "-f", "PostgreSQL",
            pg_conn_str,
            sqlite_path,
            table,
            "-nln", f"{schema}.{table}",
            "-lco", "GEOMETRY_NAME=shape",
            "-overwrite",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return False, (r.stderr or r.stdout or "unknown error").strip()
        return True, ""
    except FileNotFoundError:
        return False, "ogr2ogr not found (install GDAL)"
    except subprocess.TimeoutExpired:
        return False, "ogr2ogr timeout"
    except Exception as e:
        return False, str(e)


def migrate_one_db(
    sqlite_path: str,
    schema_name: str,
    pg_conn_str: str,
    db_config: dict,
    report: dict,
) -> None:
    """Migrate one SQLite file into the target PostgreSQL schema."""
    report["tables"] = {}
    report["row_counts"] = {}
    report["errors"] = []
    report["geometry_status"] = {}

    if not os.path.isfile(sqlite_path):
        report["errors"].append(f"File not found: {sqlite_path}")
        return

    tables = list_sqlite_tables(sqlite_path)
    report["table_count"] = len(tables)

    for table in tables:
        ok, err = run_ogr2ogr(sqlite_path, table, pg_conn_str, schema_name)
        report["tables"][table] = "ok" if ok else "failed"
        report["geometry_status"][table] = "migrated (ogr2ogr)" if ok else ("failed: " + err)
        if not ok:
            report["errors"].append(f"{table}: {err}")

    # Optionally collect row counts from PostgreSQL when psycopg2 is available.
    try:
        conn = psycopg2.connect(
            host=db_config.get("host"),
            port=db_config.get("port"),
            database=db_config.get("database"),
            user=db_config.get("user"),
            password=db_config.get("password"),
        )
        cur = conn.cursor()
        for table in tables:
            if report["tables"].get(table) != "ok":
                continue
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{schema_name}"."{table}"')
                report["row_counts"][table] = cur.fetchone()[0]
            except Exception as e:
                report["row_counts"][table] = None
                report["errors"].append(f"count {table}: {e}")
        cur.close()
        conn.close()
    except Exception as e:
        report["row_counts_summary"] = f"Could not get row counts: {e}"


def main():
    parser = argparse.ArgumentParser(description="SpatialSQL SQLite -> PostgreSQL/PostGIS migration")
    parser.add_argument("sdbdatasets", type=str, nargs="?", default="sdbdatasets",
                        help="Path to sdbdatasets directory (default: sdbdatasets)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to db_config.yaml (default: config/db_config.yaml)")
    parser.add_argument("--report-dir", type=str, default=None,
                        help="Directory for migration_report.* (default: scripts/benchmark/spatialsql)")
    args = parser.parse_args()

    sdb_root = Path(args.sdbdatasets).resolve()
    if not sdb_root.is_dir():
        print(f"Error: not a directory: {sdb_root}")
        sys.exit(1)

    config_path = Path(args.config) if args.config else REPO_ROOT / "config" / "db_config.yaml"
    if not config_path.is_file():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    db_config = load_db_config(config_path)
    pg_conn_str = pg_connection_string(db_config)
    report_dir = Path(args.report_dir) if args.report_dir else (REPO_ROOT / "scripts" / "benchmark" / "spatialsql")
    report_dir.mkdir(parents=True, exist_ok=True)

    overall = {
        "sdbdatasets_path": str(sdb_root),
        "pg_connection": {"host": db_config["host"], "database": db_config["database"]},
        "databases": {},
        "ogr2ogr_available": shutil.which("ogr2ogr") is not None,
    }

    for version in VERSIONS:
        for domain in DOMAINS:
            schema_name = f"spatialsql_{version}_{domain}"
            sqlite_path = sdb_root / version / domain / f"{domain}.sqlite"
            key = f"{version}_{domain}"
            report = {
                "sqlite_path": str(sqlite_path),
                "schema_name": schema_name,
            }
            schema_err = None
            try:
                conn = psycopg2.connect(
                    host=db_config["host"],
                    port=db_config["port"],
                    database=db_config["database"],
                    user=db_config["user"],
                    password=db_config["password"],
                )
                cur = conn.cursor()
                create_pg_schema(cur, schema_name)
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                schema_err = f"Create schema: {e}"
            migrate_one_db(str(sqlite_path), schema_name, pg_conn_str, db_config, report)
            if schema_err:
                report["errors"].append(schema_err)

            overall["databases"][key] = report

    report_json = report_dir / "migration_report.json"
    report_txt = report_dir / "migration_report.txt"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(_format_report_txt(overall))
    print(f"Migration report: {report_json}")
    print(f"Summary: {report_txt}")


def _format_report_txt(overall: dict) -> str:
    lines = [
        "SpatialSQL SQLite -> PostgreSQL/PostGIS Migration Report",
        "=" * 60,
        f"sdbdatasets: {overall['sdbdatasets_path']}",
        f"ogr2ogr available: {overall['ogr2ogr_available']}",
        "",
    ]
    for key, r in overall.get("databases", {}).items():
        lines.append(f"[{key}] schema={r.get('schema_name')} path={r.get('sqlite_path')}")
        lines.append(f"  tables: {r.get('table_count', 0)}")
        for t, status in r.get("tables", {}).items():
            rc = r.get("row_counts", {}).get(t)
            rc_str = str(rc) if rc is not None else "?"
            lines.append(f"    - {t}: {status} (rows: {rc_str})")
        for err in r.get("errors", []):
            lines.append(f"  error: {err}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
