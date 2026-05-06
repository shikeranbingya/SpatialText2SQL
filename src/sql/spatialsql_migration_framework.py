"""SpatialSQL PG 半自动迭代迁移框架。"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psycopg2
import yaml

from src.inference.sql_utils import normalize_spatialsql_predicted_sql
from src.sql.sql_dialect_adapter import classify_spatialsql_failure, convert_spatialite_to_postgis


DEFAULT_VERSIONS = ["dataset1", "dataset2"]
DEFAULT_DOMAINS = ["ada", "edu", "tourism", "traffic"]
GEOMETRY_TYPES = {
    "POINT",
    "LINESTRING",
    "POLYGON",
    "MULTIPOINT",
    "MULTILINESTRING",
    "MULTIPOLYGON",
    "GEOMETRY",
}
SPATIAL_FUNCTION_PATTERN = re.compile(
    r"\b("
    r"Intersects|Intersection|GLength|Area|Distance|Buffer|ConvexHull|Centroid|"
    r"Contains|Within|Touches|Overlaps|Crosses|Disjoint|AsText|GeomFromText|"
    r"ST_[A-Za-z0-9_]+"
    r")\s*\(",
    re.I,
)
SPATIALITE_EXTENSION_CANDIDATES = [
    "mod_spatialite",
    "mod_spatialite.so",
    "libspatialite",
    "libspatialite.so",
]


def _log(message: str) -> None:
    """输出迁移进度日志。"""
    print(message, flush=True)


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    """构造轻量文本进度条。"""
    if total <= 0:
        return "[no-work]"
    ratio = max(0.0, min(1.0, current / total))
    filled = int(width * ratio)
    return f"[{'#' * filled}{'.' * (width - filled)}] {current}/{total} ({ratio * 100:.1f}%)"


@dataclass(frozen=True)
class SplitSpec:
    """描述一个 SpatialSQL split。"""

    version: str
    domain: str
    sdbdatasets_path: Path

    @property
    def split(self) -> str:
        return f"{self.version}_{self.domain}"

    @property
    def domain_dir(self) -> Path:
        return self.sdbdatasets_path / self.version / self.domain

    @property
    def sqlite_path(self) -> Path:
        return self.domain_dir / f"{self.domain}.sqlite"

    @property
    def schema_path(self) -> Path:
        return self.domain_dir / f"{self.domain}.schema"

    @property
    def table_catalog_path(self) -> Path:
        return self.domain_dir / f"{self.domain}.table.csv"


def load_spatialsql_db_config(config_path: Path) -> Dict[str, Any]:
    """加载 `spatial_sql` 目标数据库配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    databases = data.get("databases", {})
    return databases.get("spatial_sql", data.get("database", {}))


def pg_connection_string(cfg: Dict[str, Any]) -> str:
    """构造 ogr2ogr 使用的 PostgreSQL 连接串。"""
    return (
        f"PG:host={cfg['host']} port={cfg['port']} dbname={cfg['database']} "
        f"user={cfg['user']} password={cfg['password']}"
    )


def iter_split_specs(
    sdbdatasets_path: Path,
    versions: Optional[Iterable[str]] = None,
    domains: Optional[Iterable[str]] = None,
) -> List[SplitSpec]:
    """生成所有 split 定义。"""
    split_specs: List[SplitSpec] = []
    for version in versions or DEFAULT_VERSIONS:
        for domain in domains or DEFAULT_DOMAINS:
            split_specs.append(SplitSpec(version=version, domain=domain, sdbdatasets_path=sdbdatasets_path))
    return split_specs


def _sqlite_query(sqlite_path: Path, sql: str, params: Tuple[Any, ...] = ()) -> List[tuple]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.execute(sql, params)
        return cur.fetchall()
    finally:
        conn.close()


def list_sqlite_tables(sqlite_path: Path) -> List[str]:
    """列出 SQLite 中全部 table。"""
    rows = _sqlite_query(
        sqlite_path,
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
    )
    return [row[0] for row in rows if row and row[0]]


def load_business_tables(table_catalog_path: Path) -> List[str]:
    """从 `<domain>.table.csv` 读取业务表白名单。"""
    if not table_catalog_path.exists():
        return []

    business_tables: List[str] = []
    with open(table_catalog_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table_name = (row.get("name") or "").strip()
            if table_name:
                business_tables.append(table_name)
    return business_tables


def parse_schema_file(schema_path: Path) -> Dict[str, Dict[str, Any]]:
    """解析 SpatialSQL 自带 schema 文件。"""
    if not schema_path.exists():
        return {}

    text = schema_path.read_text(encoding="utf-8")
    table_blocks = re.finditer(
        r'CREATE TABLE\s+"?([^"\s]+)"?\s*\((.*?)\)\s*(?=CREATE TABLE|\Z)',
        text,
        re.S | re.I,
    )
    parsed: Dict[str, Dict[str, Any]] = {}
    for match in table_blocks:
        table_name = match.group(1)
        body = match.group(2)
        info = {
            "columns": [],
            "primary_keys": [],
            "foreign_keys": [],
            "geometry_columns": [],
        }
        for raw_line in body.splitlines():
            line = raw_line.strip().rstrip(",")
            if not line:
                continue
            if line.upper().startswith("FOREIGN KEY"):
                info["foreign_keys"].append(line)
                continue

            parts = line.split()
            column_name = parts[0].strip('"')
            declared_type = parts[1].upper() if len(parts) > 1 else ""
            column_info = {
                "name": column_name,
                "declared_type": declared_type,
                "raw_definition": line,
            }
            info["columns"].append(column_info)
            if "PRIMARY KEY" in line.upper():
                info["primary_keys"].append(column_name)
            if "REFERENCES" in line.upper():
                info["foreign_keys"].append(line)
            if declared_type in GEOMETRY_TYPES:
                info["geometry_columns"].append(column_name)
        parsed[table_name] = info
    return parsed


def load_geometry_metadata(sqlite_path: Path) -> List[Dict[str, Any]]:
    """读取 SpatiaLite `geometry_columns` 元数据。"""
    tables = list_sqlite_tables(sqlite_path)
    if "geometry_columns" not in tables:
        return []

    rows = _sqlite_query(
        sqlite_path,
        (
            "SELECT f_table_name, f_geometry_column, geometry_type, "
            "coord_dimension, srid, spatial_index_enabled "
            "FROM geometry_columns"
        ),
    )
    return [
        {
            "table": row[0],
            "column": row[1],
            "geometry_type": row[2],
            "coord_dimension": row[3],
            "srid": row[4],
            "spatial_index_enabled": row[5],
        }
        for row in rows
    ]


def load_table_columns(sqlite_path: Path, table_name: str) -> List[Dict[str, Any]]:
    """读取 SQLite 表结构。"""
    rows = _sqlite_query(sqlite_path, f"PRAGMA table_info('{table_name}')")
    return [
        {
            "cid": row[0],
            "name": row[1],
            "declared_type": row[2],
            "notnull": bool(row[3]),
            "default": row[4],
            "pk": bool(row[5]),
        }
        for row in rows
    ]


def sqlite_row_count(sqlite_path: Path, table_name: str) -> Optional[int]:
    """读取 SQLite 行数。"""
    try:
        rows = _sqlite_query(sqlite_path, f'SELECT COUNT(*) FROM "{table_name}"')
        return int(rows[0][0]) if rows else 0
    except Exception:
        return None


def _index_geometry_metadata(metadata_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in metadata_rows:
        grouped[row["table"].lower()].append(row)
    return grouped


def scan_source_split(split_spec: SplitSpec) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """扫描单个 split 的源库与元数据。"""
    business_tables = load_business_tables(split_spec.table_catalog_path)
    schema_tables = parse_schema_file(split_spec.schema_path)
    actual_tables = list_sqlite_tables(split_spec.sqlite_path) if split_spec.sqlite_path.exists() else []
    geometry_metadata = load_geometry_metadata(split_spec.sqlite_path) if split_spec.sqlite_path.exists() else []
    geometry_index = _index_geometry_metadata(geometry_metadata)

    inventory = {
        "split": split_spec.split,
        "version": split_spec.version,
        "domain": split_spec.domain,
        "sqlite_path": str(split_spec.sqlite_path),
        "schema_path": str(split_spec.schema_path),
        "table_catalog_path": str(split_spec.table_catalog_path),
        "business_tables": business_tables,
        "schema_tables": sorted(schema_tables.keys()),
        "actual_tables": actual_tables,
        "geometry_metadata": geometry_metadata,
        "tables": {},
    }
    anomalies: List[Dict[str, Any]] = []

    actual_table_set = {table.lower() for table in actual_tables}
    business_set = {table.lower() for table in business_tables}
    schema_set = {table.lower() for table in schema_tables}

    for missing in sorted(business_set - actual_table_set):
        anomalies.append(
            {
                "split": split_spec.split,
                "classification": "data_table_filter_error",
                "table": missing,
                "message": "业务表在 SQLite 中不存在",
            }
        )
    for extra in sorted(actual_table_set - business_set):
        anomalies.append(
            {
                "split": split_spec.split,
                "classification": "data_table_filter_error",
                "table": extra,
                "message": "非业务表出现在 SQLite 用户表集合中，迁移时必须过滤",
            }
        )
    for table_name in business_tables:
        table_columns = load_table_columns(split_spec.sqlite_path, table_name) if split_spec.sqlite_path.exists() else []
        schema_info = schema_tables.get(table_name, {})
        row_count = sqlite_row_count(split_spec.sqlite_path, table_name) if split_spec.sqlite_path.exists() else None
        expected_geometry_columns = schema_info.get("geometry_columns", [])
        metadata_matches = geometry_index.get(table_name.lower(), [])
        inventory["tables"][table_name] = {
            "columns": table_columns,
            "schema_columns": schema_info.get("columns", []),
            "primary_keys": schema_info.get("primary_keys", []),
            "foreign_keys": schema_info.get("foreign_keys", []),
            "expected_geometry_columns": expected_geometry_columns,
            "geometry_metadata": metadata_matches,
            "row_count": row_count,
        }
        if expected_geometry_columns and not metadata_matches:
            anomalies.append(
                {
                    "split": split_spec.split,
                    "classification": "data_geometry_error",
                    "table": table_name,
                    "message": "schema 声明了几何列，但 geometry_columns 中没有同名业务表记录",
                    "expected_geometry_columns": expected_geometry_columns,
                }
            )
        if metadata_matches and expected_geometry_columns:
            metadata_columns = {row["column"].lower() for row in metadata_matches}
            missing_columns = [
                column for column in expected_geometry_columns if column.lower() not in metadata_columns
            ]
            if missing_columns:
                anomalies.append(
                    {
                        "split": split_spec.split,
                        "classification": "data_geometry_error",
                        "table": table_name,
                        "message": "geometry_columns 存在，但几何列名与 schema 不一致",
                        "expected_geometry_columns": expected_geometry_columns,
                        "metadata_columns": sorted(metadata_columns),
                    }
                )

    for metadata_row in geometry_metadata:
        if metadata_row["table"].lower() not in business_set:
            anomalies.append(
                {
                    "split": split_spec.split,
                    "classification": "data_geometry_error",
                    "table": metadata_row["table"],
                    "message": "geometry_columns 指向了非业务表，疑似元数据异常",
                    "metadata": metadata_row,
                }
            )
    for schema_table in sorted(schema_set - business_set):
        anomalies.append(
            {
                "split": split_spec.split,
                "classification": "data_table_filter_error",
                "table": schema_table,
                "message": "schema 文件中声明的表不在业务表白名单内",
            }
        )

    return inventory, anomalies


def build_source_inventory(
    sdbdatasets_path: Path,
    versions: Optional[Iterable[str]] = None,
    domains: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """构建全量源库 inventory 与 anomaly 报告。"""
    inventory = {
        "sdbdatasets_path": str(sdbdatasets_path),
        "generated_at_epoch": time.time(),
        "splits": {},
    }
    anomaly_report = {
        "sdbdatasets_path": str(sdbdatasets_path),
        "generated_at_epoch": time.time(),
        "summary": {},
        "details": [],
    }
    anomaly_counter: Counter = Counter()

    for split_spec in iter_split_specs(sdbdatasets_path, versions=versions, domains=domains):
        split_inventory, anomalies = scan_source_split(split_spec)
        inventory["splits"][split_spec.split] = split_inventory
        anomaly_report["details"].extend(anomalies)
        for anomaly in anomalies:
            anomaly_counter[anomaly["classification"]] += 1

    anomaly_report["summary"] = {
        "total_anomalies": sum(anomaly_counter.values()),
        "by_classification": dict(anomaly_counter),
    }
    return inventory, anomaly_report


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 2, "completed": {}}


def _write_checkpoint(checkpoint_path: Path, checkpoint: Dict[str, Any]) -> None:
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def _safe_pg_connect(db_cfg: Dict[str, Any]):
    timeout_cfg = db_cfg.get("timeout", {}) if isinstance(db_cfg.get("timeout"), dict) else {}
    return psycopg2.connect(
        host=db_cfg["host"],
        port=db_cfg["port"],
        database=db_cfg["database"],
        user=db_cfg["user"],
        password=db_cfg["password"],
        connect_timeout=int(timeout_cfg.get("connection_timeout", 10)),
    )


def _pg_row_count(db_cfg: Dict[str, Any], table_name: str) -> Optional[int]:
    try:
        conn = _safe_pg_connect(db_cfg)
        cur = conn.cursor()
        cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        row_count = int(cur.fetchone()[0])
        cur.close()
        conn.close()
        return row_count
    except Exception:
        return None


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _target_pg_table_name(split: str, table_name: str) -> str:
    """统一 PG 物理表名为小写，避免未加引号查询命中另一张旧表。"""
    return f"{split}_{table_name}".lower()


def _map_sqlite_declared_type_to_pg(declared_type: str, *, is_geometry: bool = False, geometry_srid: Optional[int] = None) -> str:
    normalized = (declared_type or "").strip().upper()
    if is_geometry:
        if normalized == "POINT" and geometry_srid:
            return f"geometry(Point,{geometry_srid})"
        if normalized == "POINT":
            return "geometry(Point)"
        return "geometry"
    if "INT" in normalized:
        return "BIGINT"
    if any(token in normalized for token in {"REAL", "FLOA", "DOUB"}):
        return "DOUBLE PRECISION"
    if any(token in normalized for token in {"NUM", "DEC"}):
        return "NUMERIC"
    if "BLOB" in normalized:
        return "BYTEA"
    return "TEXT"


def _decode_spatialite_point_blob(blob_value: Any) -> Tuple[float, float, int]:
    if blob_value is None:
        raise ValueError("geometry blob is null")
    if isinstance(blob_value, memoryview):
        blob_value = blob_value.tobytes()
    if not isinstance(blob_value, (bytes, bytearray)):
        raise TypeError(f"unsupported geometry blob type: {type(blob_value)!r}")
    blob = bytes(blob_value)
    if len(blob) != 60:
        raise ValueError(f"unexpected point blob length: {len(blob)}")

    endian_flag = blob[1]
    if endian_flag == 0:
        endian = ">"
    elif endian_flag == 1:
        endian = "<"
    else:
        raise ValueError(f"unexpected endian flag: {endian_flag}")

    srid = struct.unpack(f"{endian}I", blob[2:6])[0]
    marker = blob[38]
    end_marker = blob[59]
    geom_type = struct.unpack(f"{endian}I", blob[39:43])[0]
    if marker != 0x7C or end_marker != 0xFE or geom_type != 1:
        raise ValueError(
            f"unexpected spatialite point blob markers marker={marker:#x} geom_type={geom_type} end={end_marker:#x}"
        )
    x = struct.unpack(f"{endian}d", blob[43:51])[0]
    y = struct.unpack(f"{endian}d", blob[51:59])[0]
    return x, y, srid


def _supports_manual_blob_geometry_fallback(table_info: Dict[str, Any]) -> bool:
    expected_geometry_columns = table_info.get("expected_geometry_columns", [])
    if len(expected_geometry_columns) != 1:
        return False
    if table_info.get("geometry_metadata"):
        return False
    geometry_column = expected_geometry_columns[0].lower()
    for column in table_info.get("columns", []):
        if column.get("name", "").lower() == geometry_column:
            return (column.get("declared_type") or "").strip().upper() == "POINT"
    return False


def _manual_migrate_blob_geometry_table(
    sqlite_path: Path,
    table_name: str,
    target_table_name: str,
    table_info: Dict[str, Any],
    db_cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    geometry_column = table_info.get("expected_geometry_columns", [None])[0]
    if not geometry_column:
        return False, "missing geometry column for manual blob geometry fallback"

    columns = table_info.get("columns", [])
    if not columns:
        columns = load_table_columns(sqlite_path, table_name)
    if not columns:
        return False, "source table columns unavailable"

    sqlite_conn = sqlite3.connect(str(sqlite_path))
    pg_conn = None
    try:
        quoted_source_columns = ", ".join(_quote_ident(column["name"]) for column in columns)
        sqlite_cur = sqlite_conn.execute(f'SELECT {quoted_source_columns} FROM {_quote_ident(table_name)}')
        rows = sqlite_cur.fetchall()

        geometry_srid: Optional[int] = None
        geometry_name_lower = geometry_column.lower()
        for row in rows:
            for column, value in zip(columns, row):
                if column["name"].lower() == geometry_name_lower and value is not None:
                    _, _, geometry_srid = _decode_spatialite_point_blob(value)
                    break
            if geometry_srid is not None:
                break

        pg_conn = _safe_pg_connect(db_cfg)
        pg_cur = pg_conn.cursor()
        pg_cur.execute(f'DROP TABLE IF EXISTS {_quote_ident(target_table_name)}')

        create_defs = []
        for column in columns:
            is_geometry = column["name"].lower() == geometry_name_lower
            pg_type = _map_sqlite_declared_type_to_pg(
                column.get("declared_type", ""),
                is_geometry=is_geometry,
                geometry_srid=geometry_srid,
            )
            create_defs.append(f'{_quote_ident(column["name"].lower())} {pg_type}')
        pg_cur.execute(f'CREATE TABLE {_quote_ident(target_table_name)} ({", ".join(create_defs)})')

        insert_columns = [_quote_ident(column["name"].lower()) for column in columns]
        insert_exprs = []
        for column in columns:
            if column["name"].lower() == geometry_name_lower:
                insert_exprs.append("ST_SetSRID(ST_MakePoint(%s, %s), %s)")
            else:
                insert_exprs.append("%s")
        insert_sql = (
            f'INSERT INTO {_quote_ident(target_table_name)} ({", ".join(insert_columns)}) '
            f'VALUES ({", ".join(insert_exprs)})'
        )

        batch: List[Tuple[Any, ...]] = []
        for row in rows:
            params: List[Any] = []
            for column, value in zip(columns, row):
                if column["name"].lower() == geometry_name_lower:
                    if value is None:
                        params.extend([None, None, geometry_srid or 4326])
                    else:
                        x, y, srid = _decode_spatialite_point_blob(value)
                        params.extend([x, y, srid])
                else:
                    params.append(value)
            batch.append(tuple(params))
        if batch:
            pg_cur.executemany(insert_sql, batch)
        pg_conn.commit()
        pg_cur.close()
        return True, ""
    except Exception as exc:
        if pg_conn is not None:
            pg_conn.rollback()
        return False, str(exc)
    finally:
        sqlite_conn.close()
        if pg_conn is not None:
            pg_conn.close()


def run_ogr2ogr(
    sqlite_path: Path,
    table_name: str,
    pg_conn_str: str,
    target_table_name: str,
    geometry_name: Optional[str] = None,
) -> Tuple[bool, str]:
    """调用 ogr2ogr 迁移单表。"""
    cmd = [
        "ogr2ogr",
        "-f",
        "PostgreSQL",
        pg_conn_str,
        str(sqlite_path),
        table_name,
        "-nln",
        target_table_name,
        "-lco",
        "OVERWRITE=YES",
    ]
    if geometry_name:
        cmd.extend(["-lco", f"GEOMETRY_NAME={geometry_name.lower()}"])
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError:
        return False, "ogr2ogr not found (GDAL not installed)"
    except subprocess.TimeoutExpired:
        return False, "ogr2ogr timeout after 600 seconds"
    except Exception as exc:  # pragma: no cover - 防御式返回
        return False, str(exc)

    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "unknown error").strip()
    return True, ""


def migrate_inventory_to_pg(
    inventory: Dict[str, Any],
    db_cfg: Dict[str, Any],
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """按 inventory 的业务表白名单迁移到 PostgreSQL。"""
    checkpoint = _load_checkpoint(checkpoint_path)
    pg_conn_str = pg_connection_string(db_cfg)
    report = {
        "target_database": db_cfg.get("database"),
        "target_host": db_cfg.get("host"),
        "generated_at_epoch": time.time(),
        "ogr2ogr_available": shutil.which("ogr2ogr") is not None,
        "splits": {},
        "summary": {
            "tables_total": 0,
            "tables_ok": 0,
            "tables_failed": 0,
            "tables_skipped_from_checkpoint": 0,
        },
    }
    _log("=" * 70)
    _log(
        f"[Migration] target={db_cfg.get('host')}:{db_cfg.get('port')}/{db_cfg.get('database')} "
        f"ogr2ogr_available={report['ogr2ogr_available']}"
    )
    _log("=" * 70)

    for split, split_info in sorted(inventory.get("splits", {}).items()):
        business_tables = split_info.get("business_tables", [])
        _log(f"[Migration][{split}] start tables={len(business_tables)} sqlite={split_info.get('sqlite_path')}")
        split_report = {
            "sqlite_path": split_info.get("sqlite_path"),
            "tables": {},
            "status": "completed",
        }
        sqlite_path = Path(split_info["sqlite_path"])
        for index, table_name in enumerate(business_tables, start=1):
            report["summary"]["tables_total"] += 1
            table_key = f"{split}:{table_name}"
            target_table_name = _target_pg_table_name(split, table_name)
            table_info = split_info.get("tables", {}).get(table_name, {})
            geometry_columns = table_info.get("expected_geometry_columns", [])
            geometry_name = geometry_columns[0].lower() if geometry_columns else None
            row_count_source = table_info.get("row_count")
            _log(
                f"[Migration][{split}][{index}/{len(business_tables)}] "
                f"{table_name} -> {target_table_name} geometry={geometry_name or '-'} "
                f"source_rows={row_count_source if row_count_source is not None else '?'}"
            )

            if checkpoint.get("completed", {}).get(table_key, {}).get("status") == "ok":
                split_report["tables"][table_name] = {
                    "status": "skipped",
                    "reason": "checkpoint",
                    "target_name": target_table_name,
                    "row_count_source": row_count_source,
                    "row_count_target": checkpoint["completed"][table_key].get("rows"),
                    "geometry_column": geometry_name,
                    "classification": "checkpoint",
                }
                report["summary"]["tables_skipped_from_checkpoint"] += 1
                _log(f"[Migration][{split}][{table_name}] skipped from checkpoint")
                continue

            if _supports_manual_blob_geometry_fallback(table_info):
                _log(
                    f"[Migration][{split}][{table_name}] using manual blob geometry fallback "
                    f"geometry={geometry_name or '-'}"
                )
                ok, error = _manual_migrate_blob_geometry_table(
                    sqlite_path=sqlite_path,
                    table_name=table_name,
                    target_table_name=target_table_name,
                    table_info=table_info,
                    db_cfg=db_cfg,
                )
            else:
                ok, error = run_ogr2ogr(
                    sqlite_path=sqlite_path,
                    table_name=table_name,
                    pg_conn_str=pg_conn_str,
                    target_table_name=target_table_name,
                    geometry_name=geometry_name,
                )
            row_count_target = _pg_row_count(db_cfg, target_table_name) if ok else None
            classification = classify_spatialsql_failure(error_message=error) if error else None
            split_report["tables"][table_name] = {
                "status": "ok" if ok else "failed",
                "error": error or None,
                "target_name": target_table_name,
                "row_count_source": row_count_source,
                "row_count_target": row_count_target,
                "geometry_column": geometry_name,
                "classification": classification,
            }
            if ok:
                checkpoint.setdefault("completed", {})[table_key] = {
                    "target_table": target_table_name,
                    "rows": row_count_target,
                    "status": "ok",
                }
                report["summary"]["tables_ok"] += 1
                _log(
                    f"[Migration][{split}][{table_name}] ok "
                    f"target_rows={row_count_target if row_count_target is not None else '?'}"
                )
            else:
                report["summary"]["tables_failed"] += 1
                _log(
                    f"[Migration][{split}][{table_name}] failed "
                    f"classification={classification or 'unknown'} error={error[:240] if error else 'unknown'}"
                )
            _write_checkpoint(checkpoint_path, checkpoint)

        report["splits"][split] = split_report
        _log(
            f"[Migration][{split}] done "
            f"ok={sum(1 for t in split_report['tables'].values() if t.get('status') == 'ok')} "
            f"failed={sum(1 for t in split_report['tables'].values() if t.get('status') == 'failed')} "
            f"skipped={sum(1 for t in split_report['tables'].values() if t.get('status') == 'skipped')}"
        )

    _log(
        f"[Migration] summary total={report['summary']['tables_total']} ok={report['summary']['tables_ok']} "
        f"failed={report['summary']['tables_failed']} "
        f"checkpoint_skipped={report['summary']['tables_skipped_from_checkpoint']}"
    )
    return report


def repair_bytea_geometry_columns(
    inventory: Dict[str, Any],
    db_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """修复被迁移成 bytea 的 Point 几何列。"""
    report = {
        "generated_at_epoch": time.time(),
        "summary": {
            "tables_checked": 0,
            "tables_repaired": 0,
            "tables_skipped": 0,
            "tables_failed": 0,
        },
        "details": [],
    }

    for split, split_info in sorted(inventory.get("splits", {}).items()):
        sqlite_path = Path(split_info.get("sqlite_path", ""))
        for table_name, table_info in sorted(split_info.get("tables", {}).items()):
            if not _supports_manual_blob_geometry_fallback(table_info):
                report["summary"]["tables_skipped"] += 1
                continue

            geometry_column = table_info.get("expected_geometry_columns", [None])[0]
            if not geometry_column:
                report["summary"]["tables_skipped"] += 1
                continue

            report["summary"]["tables_checked"] += 1
            target_table_name = _target_pg_table_name(split, table_name)
            detail = {
                "split": split,
                "table": table_name,
                "target_table": target_table_name,
                "geometry_column": geometry_column.lower(),
                "status": "skipped",
            }

            try:
                conn = _safe_pg_connect(db_cfg)
                cur = conn.cursor()
                cur.execute(
                    f'''
                    select data_type, udt_name
                    from information_schema.columns
                    where table_schema='public' and table_name=%s and column_name=%s
                    ''',
                    (target_table_name, geometry_column.lower()),
                )
                schema_row = cur.fetchone()
                cur.execute(
                    f'SELECT pg_typeof({_quote_ident(geometry_column.lower())})::text, COUNT(*) '
                    f'FROM {_quote_ident(target_table_name)} GROUP BY 1'
                )
                runtime_rows = cur.fetchall()
                cur.close()
                conn.close()
            except Exception as exc:
                detail["status"] = "failed"
                detail["error"] = str(exc)
                report["summary"]["tables_failed"] += 1
                report["details"].append(detail)
                _log(
                    f"[Bytea Repair][{split}][{table_name}] failed to inspect target column "
                    f"error={str(exc)[:240]}"
                )
                continue

            if not schema_row:
                detail["reason"] = "target column missing"
                report["summary"]["tables_skipped"] += 1
                report["details"].append(detail)
                _log(f"[Bytea Repair][{split}][{table_name}] skipped target column missing")
                continue

            data_type, udt_name = schema_row
            runtime_types = sorted({row[0] for row in runtime_rows if row and row[0]})
            runtime_type = runtime_types[0] if len(runtime_types) == 1 else ",".join(runtime_types)
            detail["schema_type"] = f"{data_type}/{udt_name}"
            detail["runtime_type"] = runtime_type

            if runtime_type.lower() != "bytea":
                detail["reason"] = (
                    f"target column runtime type already {runtime_type} "
                    f"(schema={data_type}/{udt_name})"
                )
                report["summary"]["tables_skipped"] += 1
                report["details"].append(detail)
                _log(
                    f"[Bytea Repair][{split}][{table_name}] skipped target already runtime={runtime_type} "
                    f"schema={data_type}/{udt_name}"
                )
                continue

            _log(
                f"[Bytea Repair][{split}][{table_name}] repairing bytea -> geometry "
                f"column={geometry_column.lower()} schema={data_type}/{udt_name} runtime={runtime_type}"
            )
            ok, error = _manual_migrate_blob_geometry_table(
                sqlite_path=sqlite_path,
                table_name=table_name,
                target_table_name=target_table_name,
                table_info=table_info,
                db_cfg=db_cfg,
            )
            if ok:
                detail["status"] = "repaired"
                report["summary"]["tables_repaired"] += 1
                _log(f"[Bytea Repair][{split}][{table_name}] repaired")
            else:
                detail["status"] = "failed"
                detail["error"] = error
                report["summary"]["tables_failed"] += 1
                _log(
                    f"[Bytea Repair][{split}][{table_name}] failed "
                    f"error={error[:240] if error else 'unknown'}"
                )
            report["details"].append(detail)

    _log(
        f"[Bytea Repair] summary checked={report['summary']['tables_checked']} "
        f"repaired={report['summary']['tables_repaired']} "
        f"failed={report['summary']['tables_failed']} skipped={report['summary']['tables_skipped']}"
    )
    return report


def validate_geometry_columns(
    inventory: Dict[str, Any],
    db_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """对迁移后的几何列做健康检查。"""
    report = {
        "generated_at_epoch": time.time(),
        "summary": {
            "tables_checked": 0,
            "tables_ok": 0,
            "tables_failed": 0,
            "tables_skipped": 0,
        },
        "splits": {},
    }
    try:
        conn = _safe_pg_connect(db_cfg)
    except Exception as exc:
        report["summary"]["connection_error"] = str(exc)
        _log(f"[Geometry] connection failed: {exc}")
        return report

    _log("=" * 70)
    _log("[Geometry] start validation")
    _log("=" * 70)
    cur = conn.cursor()
    for split, split_info in sorted(inventory.get("splits", {}).items()):
        business_tables = split_info.get("business_tables", [])
        _log(f"[Geometry][{split}] start tables={len(business_tables)}")
        split_report = {"tables": {}}
        for index, table_name in enumerate(business_tables, start=1):
            target_table = _target_pg_table_name(split, table_name)
            table_info = split_info.get("tables", {}).get(table_name, {})
            geometry_columns = table_info.get("expected_geometry_columns", [])
            geometry_name = geometry_columns[0].lower() if geometry_columns else None
            report["summary"]["tables_checked"] += 1
            _log(
                f"[Geometry][{split}][{index}/{len(business_tables)}] "
                f"{target_table} geometry={geometry_name or '-'}"
            )

            if not geometry_name:
                split_report["tables"][table_name] = {
                    "status": "skipped",
                    "reason": "non_spatial_table",
                    "target_table": target_table,
                }
                report["summary"]["tables_skipped"] += 1
                _log(f"[Geometry][{split}][{table_name}] skipped non_spatial_table")
                continue

            try:
                cur.execute(
                    (
                        f'SELECT COUNT(*) FILTER (WHERE "{geometry_name}" IS NULL), '
                        f'COUNT(*) FILTER (WHERE NOT ST_IsValid("{geometry_name}")) '
                        f'FROM "{target_table}"'
                    )
                )
                null_count, invalid_count = cur.fetchone()
                cur.execute(
                    (
                        f'SELECT GeometryType("{geometry_name}"), ST_SRID("{geometry_name}") '
                        f'FROM "{target_table}" WHERE "{geometry_name}" IS NOT NULL LIMIT 5'
                    )
                )
                samples = cur.fetchall()
                status = "ok"
                if invalid_count:
                    status = "failed"
                split_report["tables"][table_name] = {
                    "status": status,
                    "target_table": target_table,
                    "geometry_column": geometry_name,
                    "null_geometry_count": null_count,
                    "invalid_geometry_count": invalid_count,
                    "sample_types": [row[0] for row in samples],
                    "sample_srids": [row[1] for row in samples],
                    "classification": None if status == "ok" else "data_geometry_error",
                }
                if status == "ok":
                    report["summary"]["tables_ok"] += 1
                    _log(
                        f"[Geometry][{split}][{table_name}] ok "
                        f"null={null_count} invalid={invalid_count}"
                    )
                else:
                    report["summary"]["tables_failed"] += 1
                    _log(
                        f"[Geometry][{split}][{table_name}] failed "
                        f"classification=data_geometry_error null={null_count} invalid={invalid_count}"
                    )
            except Exception as exc:
                split_report["tables"][table_name] = {
                    "status": "failed",
                    "target_table": target_table,
                    "geometry_column": geometry_name,
                    "error": str(exc),
                    "classification": "data_geometry_error",
                }
                report["summary"]["tables_failed"] += 1
                conn.rollback()
                _log(f"[Geometry][{split}][{table_name}] failed error={exc}")
        report["splits"][split] = split_report
        _log(
            f"[Geometry][{split}] done "
            f"ok={sum(1 for t in split_report['tables'].values() if t.get('status') == 'ok')} "
            f"failed={sum(1 for t in split_report['tables'].values() if t.get('status') == 'failed')} "
            f"skipped={sum(1 for t in split_report['tables'].values() if t.get('status') == 'skipped')}"
        )

    cur.close()
    conn.close()
    _log(
        f"[Geometry] summary checked={report['summary']['tables_checked']} "
        f"ok={report['summary']['tables_ok']} failed={report['summary']['tables_failed']} "
        f"skipped={report['summary']['tables_skipped']}"
    )
    return report


def load_spatialsql_items(dataset_config_path: Path) -> List[Dict[str, Any]]:
    """加载 SpatialSQL QA 样本。"""
    from src.datasets.loaders.spatial_sql_loader import SpatialSQLLoader

    with open(dataset_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_info = cfg["datasets"]["spatialsql_pg"]
    loader = SpatialSQLLoader(dataset_info)
    raw_data = loader.load_raw_data(dataset_info.get("data_path", "sdbdatasets"))
    return loader.extract_questions_and_sqls(raw_data)


def build_sql_conversion_report(
    dataset_config_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """生成规则转换报告与初始修复候选。"""
    items = load_spatialsql_items(dataset_config_path)
    split_counter: Dict[str, Counter] = defaultdict(Counter)
    details: List[Dict[str, Any]] = []
    repair_candidates: List[Dict[str, Any]] = []

    for item in items:
        split = item.get("metadata", {}).get("split", "")
        table_prefix = f"{split}_" if split else None
        source_sql = item.get("source_sql") or item.get("gold_sql", "")
        converted_sql, issues = convert_spatialite_to_postgis(source_sql, table_prefix=table_prefix)
        classification = classify_spatialsql_failure(issues=issues) if issues else None
        detail = {
            "id": item.get("id"),
            "source_id": item.get("metadata", {}).get("source_id"),
            "split": split,
            "source_sql": source_sql,
            "converted_sql": converted_sql,
            "issues": issues,
            "classification": classification,
            "repair_source": "rule",
            "repair_status": "rule_validated" if not issues else "rule_needs_review",
        }
        details.append(detail)
        if classification:
            split_counter[split][classification] += 1
            repair_candidates.append(
                {
                    "id": item.get("id"),
                    "source_id": item.get("metadata", {}).get("source_id"),
                    "split": split,
                    "source_sql": source_sql,
                    "candidate_sql": converted_sql,
                    "classification": classification,
                    "error_message": None,
                    "repair_source": "rule",
                    "repair_status": "pending_llm" if classification != "data_geometry_error" else "manual_review",
                    "target_table_prefix": f"{split}_" if split else "",
                    "source_backend": "sqlite",
                    "target_backend": "postgres",
                }
            )

    report = {
        "generated_at_epoch": time.time(),
        "total_items": len(items),
        "issue_count": len(repair_candidates),
        "issues_by_split": {split: dict(counter) for split, counter in sorted(split_counter.items())},
        "details": details,
    }
    return report, repair_candidates


class SQLiteSpatialExecutor:
    """源 SQLite/SpatiaLite 执行器。"""

    def __init__(self, sqlite_path: Path):
        self.sqlite_path = sqlite_path
        self.conn = sqlite3.connect(str(sqlite_path))
        self.extension_loaded = self._load_spatialite_extension()

    def _load_spatialite_extension(self) -> bool:
        try:
            self.conn.enable_load_extension(True)
        except Exception:
            return False

        for candidate in SPATIALITE_EXTENSION_CANDIDATES:
            try:
                self.conn.load_extension(candidate)
                return True
            except Exception:
                continue
        return False

    def execute(self, sql: str) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            return {"status": "ok", "rows": rows, "extension_loaded": self.extension_loaded}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "extension_loaded": self.extension_loaded}
        finally:
            cursor.close()

    def close(self) -> None:
        self.conn.close()


class PostgresExecutor:
    """目标 PostgreSQL 执行器。"""

    def __init__(self, db_cfg: Dict[str, Any]):
        self.db_cfg = db_cfg
        self.conn = _safe_pg_connect(db_cfg)

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


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, Decimal):
        return round(float(value), 6)
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str):
        return value.strip()
    return value


def _normalize_rows(rows: Iterable[tuple]) -> List[Tuple[Any, ...]]:
    normalized = []
    for row in rows:
        normalized.append(tuple(_normalize_scalar(value) for value in row))
    return normalized


def compare_sql_results(source_rows: Iterable[tuple], target_rows: Iterable[tuple]) -> Tuple[str, Dict[str, Any]]:
    """比较源库与目标库结果。"""
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


def classify_semantic_mismatch(
    detail: Dict[str, Any],
    comparison_details: Optional[Dict[str, Any]] = None,
) -> str:
    """为 semantic mismatch 提供更稳定的子类型。"""
    comparison = comparison_details or detail.get("comparison_details") or {}
    source_count = comparison.get("source_count")
    target_count = comparison.get("target_count")
    only_in_source = comparison.get("only_in_source") or []
    only_in_target = comparison.get("only_in_target") or []
    source_sql = (detail.get("source_sql") or "").lower()
    target_sql = (detail.get("target_sql") or "").lower()

    if source_count != target_count:
        return "result_scope_difference"

    source_row_widths = {len(eval_row) for eval_row in []}
    del source_row_widths
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

    tuple_widths = {
        len(row)
        for row in parsed_rows
        if isinstance(row, tuple)
    }
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
    """为一致性报告生成更稳定的聚合摘要。"""
    by_status: Counter = Counter()
    by_classification: Counter = Counter()
    by_split: Dict[str, Counter] = defaultdict(Counter)
    by_mismatch_subtype: Counter = Counter()

    for detail in report.get("details", []):
        status = detail.get("status", "unknown")
        by_status[status] += 1
        split = detail.get("split") or "unknown"
        by_split[split][status] += 1
        classification = detail.get("classification")
        if classification:
            by_classification[classification] += 1
        if status == "semantic_mismatch":
            subtype = detail.get("mismatch_subtype") or "unknown"
            by_mismatch_subtype[subtype] += 1

    summary = dict(report.get("summary", {}))
    summary["by_status"] = dict(sorted(by_status.items()))
    summary["by_classification"] = dict(sorted(by_classification.items()))
    summary["by_split"] = {
        split: dict(sorted(counter.items()))
        for split, counter in sorted(by_split.items())
    }
    summary["by_mismatch_subtype"] = dict(sorted(by_mismatch_subtype.items()))
    return summary


def _load_schema_text(project_root: Path) -> str:
    candidates = [
        project_root / "data" / "schemas" / "spatial_sql_schema.txt",
        project_root / "data" / "schemas" / "database_schema_spatial_sql.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return "-- Schema unavailable"


def _build_repair_prompt(candidate: Dict[str, Any], schema_text: str) -> str:
    return "\n".join(
        [
            "你是 PostgreSQL + PostGIS SQL 修复助手。",
            "只允许在不改变问题语义的前提下修复方言、聚合、表列映射和空间函数写法。",
            "",
            "## 当前 split",
            candidate.get("split", ""),
            "",
            "## 目标 Schema",
            schema_text,
            "",
            "## 原始 SQL",
            candidate.get("source_sql", ""),
            "",
            "## 当前候选 SQL",
            candidate.get("candidate_sql", ""),
            "",
            "## 报错信息",
            candidate.get("error_message", "") or "无",
            "",
            "## 要求",
            "1. 只输出一条可执行 SQL",
            "2. 只做最小必要修改",
            "3. 保持 split 前缀正确",
            "4. 不要输出解释",
            "",
            "SQL:",
        ]
    )


def attempt_llm_repairs(
    candidates: List[Dict[str, Any]],
    project_root: Path,
    model_config_path: Optional[Path],
    eval_config_path: Optional[Path],
    repair_model: Optional[str],
    repair_backend: Optional[str] = None,
    repair_limit: int = 50,
) -> List[Dict[str, Any]]:
    """可选地对规则失败候选尝试 LLM 修复。"""
    if not repair_model or not model_config_path or not eval_config_path:
        _log("[LLM Repair] skipped: no repair model configured")
        return candidates

    from src.inference.model_inference import ModelInference, ModelLoaderFactory

    inference = ModelInference(
        model_config_path=str(model_config_path),
        eval_config_path=str(eval_config_path),
    )
    resolved_model_cfg, resolved_backend = inference.resolve_model_config(repair_model, repair_backend)
    loader = ModelLoaderFactory.create(resolved_model_cfg["loader_class"], resolved_model_cfg)
    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("classification") != "data_geometry_error"
        and candidate.get("repair_status") in {"pending_llm", "rule_needs_review"}
    ]
    target_count = min(len(eligible_candidates), max(0, repair_limit))
    _log("=" * 70)
    _log(
        f"[LLM Repair] start model={repair_model} backend={resolved_backend} "
        f"eligible={len(eligible_candidates)} limit={repair_limit} will_process={target_count}"
    )
    if target_count == 0:
        _log("[LLM Repair] nothing to process")
        return candidates

    loader.load_model()
    schema_text = _load_schema_text(project_root)

    processed = 0
    llm_ok = 0
    llm_failed = 0
    skipped = 0
    for candidate in candidates:
        if candidate.get("classification") == "data_geometry_error":
            skipped += 1
            continue
        if candidate.get("repair_status") not in {"pending_llm", "rule_needs_review"}:
            skipped += 1
            continue
        if processed >= repair_limit:
            break
        current_index = processed + 1
        short_source = candidate.get("source_id") or candidate.get("id") or "?"
        _log(
            f"[LLM Repair] {_progress_bar(current_index, target_count)} "
            f"split={candidate.get('split', '')} source_id={short_source} "
            f"classification={candidate.get('classification', 'unknown')}"
        )
        prompt = _build_repair_prompt(candidate, schema_text)
        try:
            repaired_sql = loader.generate_sql(prompt)
            repaired_sql = normalize_spatialsql_predicted_sql(
                repaired_sql,
                {"split": candidate.get("split", "")},
            )
            candidate["llm_candidate_sql"] = repaired_sql
            candidate["repair_source"] = "llm"
            candidate["repair_status"] = "llm_candidate_generated"
            llm_ok += 1
            _log(
                f"[LLM Repair] success split={candidate.get('split', '')} source_id={short_source} "
                f"sql_len={len(repaired_sql)}"
            )
            processed += 1
        except Exception as exc:
            candidate["repair_status"] = "manual_review"
            candidate["llm_error"] = str(exc)
            llm_failed += 1
            _log(
                f"[LLM Repair] failed split={candidate.get('split', '')} source_id={short_source} "
                f"error={str(exc)[:240]}"
            )
            processed += 1

    if hasattr(loader, "unload"):
        loader.unload()
    _log(
        f"[LLM Repair] done processed={processed} success={llm_ok} failed={llm_failed} "
        f"pre_skipped={skipped} remaining={max(0, len(eligible_candidates) - processed)}"
    )
    _log("=" * 70)
    return candidates


def build_execution_consistency_report(
    inventory: Dict[str, Any],
    dataset_config_path: Path,
    db_cfg: Dict[str, Any],
    candidate_overrides: Optional[Dict[Tuple[str, str], str]] = None,
) -> Dict[str, Any]:
    """执行源/目标一致性校验。"""
    items = load_spatialsql_items(dataset_config_path)
    split_to_sqlite = {
        split: Path(split_info["sqlite_path"])
        for split, split_info in inventory.get("splits", {}).items()
    }
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

    try:
        pg_executor = PostgresExecutor(db_cfg)
    except Exception as exc:
        report["summary"]["connection_error"] = str(exc)
        _log(f"[Consistency] postgres connection failed: {exc}")
        return report

    _log("=" * 70)
    _log(
        f"[Consistency] start total_items={len(items)} "
        f"candidate_overrides={len(candidate_overrides or {})}"
    )
    _log("[Consistency] this phase may take a while because it executes source SQLite and target PostgreSQL SQL pairwise")
    _log("=" * 70)
    executors: Dict[str, SQLiteSpatialExecutor] = {}
    try:
        for index, item in enumerate(items, start=1):
            report["summary"]["total"] += 1
            split = item.get("metadata", {}).get("split", "")
            sqlite_path = split_to_sqlite.get(split)
            source_sql = item.get("source_sql") or item.get("gold_sql", "")
            target_sql, _ = convert_spatialite_to_postgis(source_sql, table_prefix=f"{split}_" if split else None)
            override_key = (split, item.get("metadata", {}).get("source_id", ""))
            if candidate_overrides and override_key in candidate_overrides:
                target_sql = candidate_overrides[override_key]
            source_id = item.get("metadata", {}).get("source_id", "?")

            if index == 1 or index % 25 == 0 or index == len(items):
                _log(
                    f"[Consistency] {_progress_bar(index, len(items))} "
                    f"current_split={split} current_source_id={source_id} "
                    f"validated={report['summary']['validated']} "
                    f"source_errors={report['summary']['source_errors']} "
                    f"target_errors={report['summary']['target_errors']} "
                    f"semantic_mismatch={report['summary']['semantic_mismatch']}"
                )

            if sqlite_path is None or not sqlite_path.exists():
                report["summary"]["skipped"] += 1
                report["details"].append(
                    {
                        "split": split,
                        "source_id": item.get("metadata", {}).get("source_id"),
                        "status": "skipped",
                        "classification": "data_table_filter_error",
                        "message": "缺少源 SQLite 路径",
                    }
                )
                _log(f"[Consistency][{split}][{source_id}] skipped: missing source sqlite path")
                continue

            executor = executors.get(split)
            if executor is None:
                executor = SQLiteSpatialExecutor(sqlite_path)
                executors[split] = executor

            source_result = executor.execute(source_sql)
            target_result = pg_executor.execute(target_sql)
            detail = {
                "split": split,
                "id": item.get("id"),
                "source_id": item.get("metadata", {}).get("source_id"),
                "source_sql": source_sql,
                "target_sql": target_sql,
            }

            if source_result["status"] != "ok":
                report["summary"]["source_errors"] += 1
                detail["status"] = "source_error"
                detail["classification"] = classify_spatialsql_failure(error_message=source_result["error"])
                detail["source_error"] = source_result["error"]
                report["details"].append(detail)
                _log(
                    f"[Consistency][{split}][{source_id}] source_error "
                    f"classification={detail['classification'] or 'unknown'} "
                    f"error={source_result['error'][:200]}"
                )
                continue
            if target_result["status"] != "ok":
                report["summary"]["target_errors"] += 1
                detail["status"] = "target_error"
                detail["classification"] = classify_spatialsql_failure(error_message=target_result["error"])
                detail["target_error"] = target_result["error"]
                report["details"].append(detail)
                _log(
                    f"[Consistency][{split}][{source_id}] target_error "
                    f"classification={detail['classification'] or 'unknown'} "
                    f"error={target_result['error'][:200]}"
                )
                continue

            comparison, comparison_details = compare_sql_results(source_result["rows"], target_result["rows"])
            detail["status"] = comparison
            detail["comparison_details"] = comparison_details
            if comparison == "exact_match":
                detail["classification"] = None
                report["summary"]["validated"] += 1
                if index <= 5 or index % 50 == 0:
                    _log(f"[Consistency][{split}][{source_id}] exact_match")
            elif comparison == "format_difference":
                detail["classification"] = "semantic_mismatch"
                detail["mismatch_subtype"] = "format_difference"
                report["summary"]["format_difference"] += 1
                _log(
                    f"[Consistency][{split}][{source_id}] format_difference "
                    f"source_count={comparison_details.get('source_count')} "
                    f"target_count={comparison_details.get('target_count')}"
                )
            else:
                detail["classification"] = "semantic_mismatch"
                detail["mismatch_subtype"] = classify_semantic_mismatch(detail, comparison_details)
                report["summary"]["semantic_mismatch"] += 1
                _log(
                    f"[Consistency][{split}][{source_id}] semantic_mismatch "
                    f"source_count={comparison_details.get('source_count')} "
                    f"target_count={comparison_details.get('target_count')}"
                )
            report["details"].append(detail)
    finally:
        for executor in executors.values():
            executor.close()
        pg_executor.close()

    report["summary"] = summarize_consistency_report(report)
    _log(
        f"[Consistency] done total={report['summary']['total']} validated={report['summary']['validated']} "
        f"format_difference={report['summary']['format_difference']} "
        f"semantic_mismatch={report['summary']['semantic_mismatch']} "
        f"source_errors={report['summary']['source_errors']} "
        f"target_errors={report['summary']['target_errors']} skipped={report['summary']['skipped']}"
    )
    _log("=" * 70)
    return report


def merge_repair_candidates(
    initial_candidates: List[Dict[str, Any]],
    consistency_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """合并规则期告警与一致性失败，生成最终修复候选。"""
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def ensure_candidate(split: str, source_id: str) -> Dict[str, Any]:
        key = (split, source_id)
        if key not in merged:
            merged[key] = {
                "split": split,
                "source_id": source_id,
                "repair_source": "rule",
                "repair_status": "pending_llm",
            }
        return merged[key]

    for candidate in initial_candidates:
        key = (candidate.get("split", ""), candidate.get("source_id", ""))
        merged[key] = dict(candidate)

    for detail in consistency_report.get("details", []):
        if detail.get("status") in {"exact_match", "skipped"}:
            continue
        candidate = ensure_candidate(detail.get("split", ""), detail.get("source_id", ""))
        candidate.setdefault("source_sql", detail.get("source_sql"))
        candidate["candidate_sql"] = detail.get("target_sql")
        candidate["classification"] = detail.get("classification")
        candidate["error_message"] = detail.get("target_error") or detail.get("source_error")
        candidate["repair_status"] = (
            "manual_review"
            if detail.get("classification") == "data_geometry_error"
            else "pending_llm"
        )
        candidate["target_table_prefix"] = f"{detail.get('split', '')}_"
        candidate["source_backend"] = "sqlite"
        candidate["target_backend"] = "postgres"
    return list(merged.values())


def build_manual_review_items(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """挑选需要人工确认的候选。"""
    manual_review = []
    for candidate in candidates:
        if candidate.get("repair_status") in {"manual_review", "pending_llm", "llm_candidate_generated"}:
            manual_review.append(candidate)
    return manual_review


def build_regression_cases(
    consistency_report: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """从成功样本中提取回归集。"""
    candidate_map = {
        (candidate.get("split", ""), candidate.get("source_id", "")): candidate
        for candidate in candidates
    }
    regression_cases = []
    for detail in consistency_report.get("details", []):
        if detail.get("status") != "exact_match":
            continue
        candidate = candidate_map.get((detail.get("split", ""), detail.get("source_id", "")), {})
        regression_cases.append(
            {
                "split": detail.get("split"),
                "source_id": detail.get("source_id"),
                "source_sql": detail.get("source_sql"),
                "validated_sql": detail.get("target_sql"),
                "repair_source": candidate.get("repair_source", "rule"),
                "repair_status": candidate.get("repair_status", "rule_validated"),
            }
        )
    return regression_cases


def build_consistency_cluster_report(consistency_report: Dict[str, Any]) -> Dict[str, Any]:
    """按失败模式聚合一致性结果，便于迭代优化。"""
    clusters: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for detail in consistency_report.get("details", []):
        status = detail.get("status")
        if status in {"exact_match", "skipped"}:
            continue

        classification = detail.get("classification") or "unknown"
        split = detail.get("split") or "unknown"
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
                "splits": Counter(),
                "sample_cases": [],
            }
        cluster = clusters[key]
        cluster["count"] += 1
        cluster["splits"][split] += 1
        if len(cluster["sample_cases"]) < 5:
            cluster["sample_cases"].append(
                {
                    "split": split,
                    "source_id": detail.get("source_id"),
                    "source_sql": detail.get("source_sql"),
                    "target_sql": detail.get("target_sql"),
                    "message": raw_message or detail.get("comparison_details"),
                }
            )

    sorted_clusters = sorted(
        clusters.values(),
        key=lambda item: (-item["count"], item["status"], item["classification"], item["fingerprint"]),
    )
    for cluster in sorted_clusters:
        cluster["splits"] = dict(sorted(cluster["splits"].items()))

    return {
        "generated_at_epoch": time.time(),
        "summary": {
            "cluster_count": len(sorted_clusters),
            "non_exact_total": sum(cluster["count"] for cluster in sorted_clusters),
        },
        "clusters": sorted_clusters,
    }


def build_final_migration_report(
    migration_report: Dict[str, Any],
    bytea_repair_report: Dict[str, Any],
    geometry_report: Dict[str, Any],
) -> Dict[str, Any]:
    """将导入、修复、几何核验合并成最终可用状态。"""
    report = json.loads(json.dumps(migration_report))
    repair_index = {
        (item.get("split"), item.get("table")): item
        for item in bytea_repair_report.get("details", [])
    }
    geometry_index = {}
    for split, split_report in geometry_report.get("splits", {}).items():
        for table_name, table_info in split_report.get("tables", {}).items():
            geometry_index[(split, table_name)] = table_info

    final_summary: Counter = Counter()
    for split, split_report in report.get("splits", {}).items():
        for table_name, table_report in split_report.get("tables", {}).items():
            repair_detail = repair_index.get((split, table_name), {})
            geometry_detail = geometry_index.get((split, table_name), {})

            repaired_by = []
            if repair_detail.get("status") == "repaired":
                repaired_by.append("bytea_repair")
            if "manual blob geometry fallback" in str(table_report.get("error") or ""):
                repaired_by.append("manual_blob_geometry_fallback")

            if table_report.get("status") == "failed":
                final_status = "failed"
            elif geometry_detail.get("status") == "failed":
                final_status = "failed"
            elif geometry_detail.get("status") == "ok":
                final_status = "ok"
            elif geometry_detail.get("status") == "skipped":
                final_status = "ok"
            elif table_report.get("status") == "ok":
                final_status = "ok"
            else:
                final_status = table_report.get("status", "unknown")

            table_report["final_status"] = final_status
            table_report["repaired_by"] = repaired_by or None
            table_report["final_geometry_status"] = geometry_detail.get("status")
            table_report["final_geometry_error"] = geometry_detail.get("error")
            if geometry_detail.get("sample_types") is not None:
                table_report["sample_geometry_types"] = geometry_detail.get("sample_types")
            if geometry_detail.get("sample_srids") is not None:
                table_report["sample_srids"] = geometry_detail.get("sample_srids")
            final_summary[final_status] += 1

    report["summary"]["final_status_counts"] = dict(sorted(final_summary.items()))
    return report


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_iterative_spatialsql_migration(
    project_root: Path,
    db_config_path: Path,
    dataset_config_path: Path,
    model_config_path: Optional[Path],
    eval_config_path: Optional[Path],
    report_dir: Path,
    repair_model: Optional[str] = None,
    repair_backend: Optional[str] = None,
    repair_limit: int = 50,
) -> Dict[str, Any]:
    """执行半自动闭环迁移。"""
    report_dir.mkdir(parents=True, exist_ok=True)
    sdbdatasets_path = project_root / "sdbdatasets"
    db_cfg = load_spatialsql_db_config(db_config_path)
    _log("=" * 70)
    _log("[Framework] SpatialSQL iterative migration started")
    _log(f"[Framework] project_root={project_root}")
    _log(f"[Framework] report_dir={report_dir}")
    _log(f"[Framework] sdbdatasets_path={sdbdatasets_path}")
    _log("=" * 70)

    _log("[Framework] phase 1/6 build source inventory")
    inventory, anomaly_report = build_source_inventory(sdbdatasets_path)
    _write_json(report_dir / "source_inventory.json", inventory)
    _write_json(report_dir / "source_anomalies.json", anomaly_report)
    _log(
        f"[Framework] source inventory done splits={len(inventory.get('splits', {}))} "
        f"anomalies={anomaly_report.get('summary', {}).get('total_anomalies', 0)}"
    )

    checkpoint_path = report_dir / "migration_checkpoint.json"
    _log("[Framework] phase 2/6 migrate inventory to postgres")
    migration_report = migrate_inventory_to_pg(inventory, db_cfg, checkpoint_path)
    _write_json(report_dir / "migration_report.json", migration_report)
    _write_json(report_dir / "migration_report_separate_db.json", migration_report)

    _log("[Framework] phase 3/7 repair bytea geometry columns")
    bytea_repair_report = repair_bytea_geometry_columns(inventory, db_cfg)
    _write_json(report_dir / "bytea_geometry_repair_report.json", bytea_repair_report)

    _log("[Framework] phase 4/7 validate geometry columns")
    geometry_report = validate_geometry_columns(inventory, db_cfg)
    _write_json(report_dir / "geometry_validation_report.json", geometry_report)
    migration_report = build_final_migration_report(migration_report, bytea_repair_report, geometry_report)
    _write_json(report_dir / "migration_report.json", migration_report)
    _write_json(report_dir / "migration_report_separate_db.json", migration_report)

    _log("[Framework] phase 5/7 build SQL conversion report")
    conversion_report, initial_candidates = build_sql_conversion_report(dataset_config_path)
    _write_json(report_dir / "sql_conversion_report.json", conversion_report)
    _log(
        f"[Framework] sql conversion done total_items={conversion_report.get('total_items', 0)} "
        f"issue_count={conversion_report.get('issue_count', 0)}"
    )

    candidates = merge_repair_candidates(initial_candidates, {"details": []})
    _log("[Framework] phase 6/7 attempt LLM repairs")
    candidates = attempt_llm_repairs(
        candidates=candidates,
        project_root=project_root,
        model_config_path=model_config_path,
        eval_config_path=eval_config_path,
        repair_model=repair_model,
        repair_backend=repair_backend,
        repair_limit=repair_limit,
    )
    llm_overrides = {
        (candidate.get("split", ""), candidate.get("source_id", "")): candidate["llm_candidate_sql"]
        for candidate in candidates
        if candidate.get("llm_candidate_sql")
    }
    consistency_report = build_execution_consistency_report(
        inventory=inventory,
        dataset_config_path=dataset_config_path,
        db_cfg=db_cfg,
        candidate_overrides=llm_overrides or None,
    )
    _write_json(report_dir / "execution_consistency_report.json", consistency_report)
    consistency_cluster_report = build_consistency_cluster_report(consistency_report)
    _write_json(report_dir / "consistency_clusters.json", consistency_cluster_report)
    _log(
        f"[Framework] consistency done validated={consistency_report.get('summary', {}).get('validated', 0)} "
        f"format_difference={consistency_report.get('summary', {}).get('format_difference', 0)} "
        f"semantic_mismatch={consistency_report.get('summary', {}).get('semantic_mismatch', 0)} "
        f"source_errors={consistency_report.get('summary', {}).get('source_errors', 0)} "
        f"target_errors={consistency_report.get('summary', {}).get('target_errors', 0)}"
    )

    _log("[Framework] phase 7/7 write repair outputs")
    candidates = merge_repair_candidates(candidates, consistency_report)
    _write_jsonl(report_dir / "sql_repair_candidates.jsonl", candidates)

    manual_review = build_manual_review_items(candidates)
    regression_cases = build_regression_cases(consistency_report, candidates)
    _write_jsonl(report_dir / "manual_review.jsonl", manual_review)
    _write_jsonl(report_dir / "regression_cases.jsonl", regression_cases)
    _log(
        f"[Framework] outputs done repair_candidates={len(candidates)} "
        f"manual_review={len(manual_review)} regression_cases={len(regression_cases)} "
        f"clusters={consistency_cluster_report.get('summary', {}).get('cluster_count', 0)}"
    )

    summary = {
        "report_dir": str(report_dir),
        "source_inventory": str(report_dir / "source_inventory.json"),
        "source_anomalies": str(report_dir / "source_anomalies.json"),
        "migration_report": str(report_dir / "migration_report.json"),
        "bytea_geometry_repair_report": str(report_dir / "bytea_geometry_repair_report.json"),
        "geometry_validation_report": str(report_dir / "geometry_validation_report.json"),
        "sql_conversion_report": str(report_dir / "sql_conversion_report.json"),
        "sql_repair_candidates": str(report_dir / "sql_repair_candidates.jsonl"),
        "execution_consistency_report": str(report_dir / "execution_consistency_report.json"),
        "consistency_clusters": str(report_dir / "consistency_clusters.json"),
        "manual_review": str(report_dir / "manual_review.jsonl"),
        "regression_cases": str(report_dir / "regression_cases.jsonl"),
    }
    _log("=" * 70)
    _log("[Framework] SpatialSQL iterative migration completed")
    _log("=" * 70)
    return summary
