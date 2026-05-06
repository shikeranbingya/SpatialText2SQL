"""FloodSQL Parquet -> PostGIS streaming migration utilities."""
from __future__ import annotations

import binascii
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2
import pyarrow.parquet as pq
import pyarrow.types as patypes
from psycopg2 import extras

EXPECTED_TABLES: Tuple[str, ...] = (
    "county",
    "floodplain",
    "census_tracts",
    "hospitals",
    "zcta",
    "schools",
    "svi",
    "cre",
    "claims",
    "nri",
)

BINARY_GEOMETRY_TABLES = {"county", "census_tracts", "floodplain", "zcta", "claims"}
POINT_TABLES = {"hospitals", "schools"}
ATTRIBUTE_ONLY_TABLES = {"svi", "cre", "nri"}
TEXT_IDENTIFIER_COLUMNS = {
    "COUNTY",
    "COUNTYFP",
    "COUNTYFIPS",
    "FIPS",
    "GEOID",
    "GEO_ID",
    "GFID",
    "HOSPITAL_ID",
    "SCHOOL_ID",
    "ST",
    "STATE",
    "STATEFP",
    "STCNTY",
    "TRACT",
    "UNIQUE_ID",
    "ZIP",
}

INDEX_COLUMNS = {
    "claims": ["GEOID", "STATEFP"],
    "county": ["GEOID", "STATEFP", "COUNTYFP"],
    "census_tracts": ["GEOID", "STATEFP", "COUNTYFP"],
    "floodplain": ["GFID", "STATEFP", "FLD_ZONE"],
    "hospitals": ["COUNTYFIPS", "ZIP", "STATEFP", "UNIQUE_ID"],
    "schools": ["ZIP", "STATEFP", "UNIQUE_ID"],
    "svi": ["GEOID", "STATE"],
    "cre": ["GEOID", "STATE", "COUNTY", "TRACT"],
    "nri": ["GEOID", "STATE"],
    "zcta": ["GEOID", "STATEFP"],
}


@dataclass(frozen=True)
class SourceLayout:
    requested_root: Path
    parquet_root: Path
    layout_name: str


@dataclass(frozen=True)
class TableStrategy:
    name: str
    source_geometry_column: Optional[str] = None
    use_coordinate_fallback: bool = False

    @property
    def has_materialized_geometry(self) -> bool:
        return self.name in {"binary_geometry", "point"}


@dataclass(frozen=True)
class ColumnPlan:
    name: str
    pg_type: str
    source_name: Optional[str] = None
    json_encoded: bool = False


@dataclass(frozen=True)
class TablePlan:
    table_name: str
    parquet_path: Path
    strategy: TableStrategy
    columns: Tuple[ColumnPlan, ...]
    read_columns: Tuple[str, ...]
    total_rows: int
    num_row_groups: int

    @property
    def has_geometry(self) -> bool:
        return self.strategy.has_materialized_geometry

    @property
    def geometry_column_name(self) -> Optional[str]:
        return "geometry" if self.has_geometry else None


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


def _load_pyarrow():
    return pq, patypes


def _load_pg_dependencies():
    return psycopg2, extras


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _normalize_identifier(name: str) -> str:
    return str(name).strip().lower()


def _is_text_identifier_column(name: Optional[str]) -> bool:
    if not name:
        return False
    return str(name).strip().upper() in TEXT_IDENTIFIER_COLUMNS


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _is_numeric_pg_type(pg_type: Optional[str]) -> bool:
    normalized = str(pg_type or "").strip().upper()
    return normalized in {
        "INTEGER",
        "BIGINT",
        "REAL",
        "DOUBLE PRECISION",
        "DECIMAL",
    } or normalized.startswith("DECIMAL(")


def _normalize_nonfinite_numeric_value(value: Any, pg_type: Optional[str]) -> Any:
    if value is None or not _is_numeric_pg_type(pg_type):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {
            "nan",
            "+nan",
            "-nan",
            "inf",
            "+inf",
            "-inf",
            "infinity",
            "+infinity",
            "-infinity",
        }:
            return None
    return value


def _normalize_scalar(value: Any, *, json_encoded: bool = False) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if json_encoded:
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def _is_hex_text(value: str) -> bool:
    if not value:
        return False
    return all(ch in "0123456789abcdefABCDEF" for ch in value)


def _strip_geopackage_header(blob: bytes) -> bytes:
    if len(blob) < 8 or blob[:2] != b"GP":
        return blob
    flags = blob[3]
    envelope_indicator = (flags >> 1) & 0b111
    envelope_size = {
        0: 0,
        1: 32,
        2: 48,
        3: 48,
        4: 64,
    }.get(envelope_indicator)
    if envelope_size is None:
        return blob
    header_size = 8 + envelope_size
    if header_size >= len(blob):
        return blob
    return blob[header_size:]


def _normalize_geometry_blob(blob: bytes) -> Optional[bytes]:
    blob = bytes(blob).strip()
    if not blob:
        return None

    decoded_hex: Optional[bytes] = None
    prefixed = blob.startswith((b"\\x", b"\\X", b"0x", b"0X"))
    candidate = blob[2:] if prefixed else blob
    if candidate and len(candidate) % 2 == 0:
        try:
            text = candidate.decode("ascii")
        except UnicodeDecodeError:
            text = ""
        if text and _is_hex_text(text):
            try:
                decoded_hex = binascii.unhexlify(text)
            except (binascii.Error, ValueError):
                decoded_hex = None
    blob = decoded_hex if decoded_hex is not None else blob
    blob = _strip_geopackage_header(blob)
    return blob or None


def _coerce_binary_geometry_value(value: Any) -> Optional[bytes]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return _normalize_geometry_blob(value)
    if isinstance(value, bytearray):
        return _normalize_geometry_blob(bytes(value))
    if isinstance(value, memoryview):
        return _normalize_geometry_blob(value.tobytes())
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return _normalize_geometry_blob(stripped.encode("ascii"))
        except UnicodeEncodeError:
            return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _normalize_geometry_blob(bytes(value))
        except ValueError:
            return None
    if hasattr(value, "tobytes"):
        try:
            return _normalize_geometry_blob(value.tobytes())
        except Exception:
            return None
    return None


def _geometry_metadata_prefers_coordinate_fallback(metadata_info: Optional[Dict[str, Any]]) -> bool:
    if not metadata_info:
        return False

    for column in metadata_info.get("schema", []):
        if column.get("column_name") != "geometry":
            continue
        description = str(column.get("description") or "").lower()
        if "construct via st_point" in description or "geometry not stored directly" in description:
            return True

    for sample in metadata_info.get("sample_rows", []):
        geometry_value = str(sample.get("geometry") or "").strip().lower()
        if geometry_value == "blob(0 bytes)":
            return True

    return False


def _load_metadata(metadata_path: Path) -> Dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata 文件不存在: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    missing = [table for table in EXPECTED_TABLES if table not in metadata]
    if missing:
        raise ValueError(f"metadata 缺少 FloodSQL 表定义: {', '.join(missing)}")
    return metadata


def _expected_file_map(metadata: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for table_name in EXPECTED_TABLES:
        file_name = (metadata.get(table_name) or {}).get("file")
        if not file_name:
            raise ValueError(f"metadata 缺少 {table_name}.file")
        mapping[table_name] = str(file_name)
    return mapping


def discover_floodsql_data_layout(data_root: str | Path, metadata: Dict[str, Any]) -> SourceLayout:
    requested_root = Path(data_root).expanduser().resolve()
    if not requested_root.exists():
        raise FileNotFoundError(f"data-root 不存在: {requested_root}")

    expected_files = _expected_file_map(metadata)
    candidates = [
        (requested_root, "flat"),
        (requested_root / "data", "nested"),
    ]
    missing_by_layout: Dict[str, List[str]] = {}
    for candidate_root, layout_name in candidates:
        missing = [
            table_name
            for table_name, file_name in expected_files.items()
            if not (candidate_root / file_name).exists()
        ]
        if not missing:
            return SourceLayout(
                requested_root=requested_root,
                parquet_root=candidate_root,
                layout_name=layout_name,
            )
        missing_by_layout[layout_name] = missing

    detail = "; ".join(
        f"{layout} 缺少 {', '.join(missing)}"
        for layout, missing in missing_by_layout.items()
    )
    raise FileNotFoundError(
        f"在 {requested_root} 下找不到完整 FloodSQL parquet 布局。"
        f" 支持 `<root>/*.parquet` 或 `<root>/data/*.parquet`；{detail}"
    )


def _map_scalar_type_name_to_pg(
    type_name: str,
    *,
    precision: Optional[int] = None,
    scale: Optional[int] = None,
    timezone: Optional[str] = None,
) -> str:
    normalized = type_name.lower()
    if normalized in {"string", "large_string", "utf8"}:
        return "TEXT"
    if normalized in {"bool", "boolean"}:
        return "BOOLEAN"
    if normalized in {"int8", "int16", "int32", "uint8", "uint16"}:
        return "INTEGER"
    if normalized in {"int64", "uint32", "uint64"}:
        return "BIGINT"
    if normalized in {"float", "float16", "float32"}:
        return "REAL"
    if normalized in {"double", "float64"}:
        return "DOUBLE PRECISION"
    if normalized == "decimal":
        if precision is not None and scale is not None:
            return f"DECIMAL({precision},{scale})"
        return "DECIMAL"
    if normalized in {"date32", "date64", "date"}:
        return "DATE"
    if normalized == "timestamp":
        return "TIMESTAMPTZ" if timezone else "TIMESTAMP"
    if normalized in {"binary", "large_binary", "fixed_size_binary"}:
        return "BYTEA"
    if normalized in {"list", "large_list", "struct", "map"}:
        return "JSONB"
    return "TEXT"


def _arrow_field_to_column_plan(field: Any) -> ColumnPlan:
    _pq, patypes = _load_pyarrow()
    arrow_type = field.type
    pg_type = "TEXT"
    json_encoded = False
    is_fixed_size_binary = getattr(patypes, "is_fixed_size_binary", lambda value: False)
    is_large_list = getattr(patypes, "is_large_list", lambda value: False)
    is_map = getattr(patypes, "is_map", lambda value: False)

    if _is_text_identifier_column(field.name):
        pg_type = _map_scalar_type_name_to_pg("string")
    elif patypes.is_string(arrow_type) or patypes.is_large_string(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("string")
    elif patypes.is_boolean(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("boolean")
    elif patypes.is_int8(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("int8")
    elif patypes.is_int16(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("int16")
    elif patypes.is_int32(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("int32")
    elif patypes.is_int64(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("int64")
    elif patypes.is_uint8(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("uint8")
    elif patypes.is_uint16(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("uint16")
    elif patypes.is_uint32(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("uint32")
    elif patypes.is_uint64(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("uint64")
    elif patypes.is_float32(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("float32")
    elif patypes.is_float64(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("float64")
    elif patypes.is_decimal(arrow_type):
        pg_type = _map_scalar_type_name_to_pg(
            "decimal",
            precision=getattr(arrow_type, "precision", None),
            scale=getattr(arrow_type, "scale", None),
        )
    elif patypes.is_timestamp(arrow_type):
        pg_type = _map_scalar_type_name_to_pg(
            "timestamp",
            timezone=getattr(arrow_type, "tz", None),
        )
    elif patypes.is_date32(arrow_type) or patypes.is_date64(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("date")
    elif patypes.is_binary(arrow_type) or patypes.is_large_binary(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("binary")
    elif is_fixed_size_binary(arrow_type):
        pg_type = _map_scalar_type_name_to_pg("fixed_size_binary")
    elif (
        patypes.is_list(arrow_type)
        or is_large_list(arrow_type)
        or patypes.is_struct(arrow_type)
        or is_map(arrow_type)
    ):
        pg_type = _map_scalar_type_name_to_pg("list")
        json_encoded = True
    return ColumnPlan(
        name=_normalize_identifier(field.name),
        pg_type=pg_type,
        source_name=field.name,
        json_encoded=json_encoded,
    )


def determine_table_strategy(
    table_name: str,
    field_names: Sequence[str],
    metadata_info: Optional[Dict[str, Any]] = None,
) -> TableStrategy:
    field_name_set = set(field_names)
    if table_name in BINARY_GEOMETRY_TABLES:
        if "geometry" not in field_name_set:
            raise ValueError(f"{table_name} 缺少 geometry 列，无法按二进制几何表迁移")
        return TableStrategy(name="binary_geometry", source_geometry_column="geometry")

    if table_name in POINT_TABLES:
        has_source_geometry = "geometry" in field_name_set
        has_latlon = "LAT" in field_name_set and "LON" in field_name_set
        if not has_source_geometry and not has_latlon:
            raise ValueError(f"{table_name} 既没有 geometry，也没有 LAT/LON，无法迁移")
        prefers_coordinate_fallback = _geometry_metadata_prefers_coordinate_fallback(metadata_info)
        return TableStrategy(
            name="point",
            source_geometry_column=(
                "geometry"
                if has_source_geometry and not prefers_coordinate_fallback
                else None
            ),
            use_coordinate_fallback=has_latlon,
        )

    if table_name in ATTRIBUTE_ONLY_TABLES:
        return TableStrategy(name="attribute")

    raise ValueError(f"未知 FloodSQL 表策略: {table_name}")


def _build_table_plan(table_name: str, metadata_info: Dict[str, Any], parquet_root: Path) -> TablePlan:
    pq, _patypes = _load_pyarrow()
    parquet_path = parquet_root / str(metadata_info.get("file"))
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet 文件不存在: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow
    field_names = [field.name for field in schema]
    strategy = determine_table_strategy(table_name, field_names, metadata_info)

    column_plans: List[ColumnPlan] = []
    read_columns: List[str] = []
    for field in schema:
        if field.name == "geometry" and strategy.name in {"binary_geometry", "point"}:
            if field.name == strategy.source_geometry_column:
                read_columns.append(field.name)
            continue
        column_plans.append(_arrow_field_to_column_plan(field))
        read_columns.append(field.name)

    if strategy.name == "binary_geometry":
        column_plans.append(ColumnPlan(name="geometry", pg_type="geometry(Geometry,4326)"))
    elif strategy.name == "point":
        column_plans.append(ColumnPlan(name="geometry", pg_type="geometry(Point,4326)"))

    return TablePlan(
        table_name=table_name,
        parquet_path=parquet_path,
        strategy=strategy,
        columns=tuple(column_plans),
        read_columns=tuple(read_columns),
        total_rows=parquet_file.metadata.num_rows,
        num_row_groups=parquet_file.num_row_groups,
    )


def _build_create_table_sql(plan: TablePlan) -> str:
    column_defs = [
        f"{_quote_ident(column.name)} {column.pg_type}"
        for column in plan.columns
    ]
    return f"CREATE TABLE {_quote_ident(plan.table_name)} ({', '.join(column_defs)})"


def _build_insert_template(plan: TablePlan) -> Tuple[List[str], str]:
    target_columns = [column.name for column in plan.columns]
    non_geometry_columns = [column for column in plan.columns if column.name != "geometry"]
    value_parts = ["%s" for _ in non_geometry_columns]

    if plan.strategy.name == "binary_geometry":
        value_parts.append("ST_SetSRID(ST_GeomFromWKB(%s), 4326)")
    elif plan.strategy.name == "point":
        value_parts.append(
            "CASE "
            "WHEN %s IS NOT NULL THEN ST_SetSRID(ST_GeomFromWKB(%s), 4326) "
            "WHEN %s IS NOT NULL AND %s IS NOT NULL THEN ST_SetSRID(ST_Point(%s, %s), 4326) "
            "ELSE NULL END"
        )
    template = f"({', '.join(value_parts)})"
    return target_columns, template


def _row_to_params(plan: TablePlan, row: Dict[str, Any]) -> Tuple[Any, ...]:
    params: List[Any] = []
    for column in plan.columns:
        if column.name == "geometry":
            continue
        source_name = column.source_name or column.name
        value = _normalize_scalar(row.get(source_name), json_encoded=column.json_encoded)
        value = _normalize_nonfinite_numeric_value(value, column.pg_type)
        if value is not None and _is_text_identifier_column(source_name):
            value = str(value)
        params.append(value)

    if plan.strategy.name == "binary_geometry":
        params.append(_coerce_binary_geometry_value(row.get(plan.strategy.source_geometry_column or "geometry")))
    elif plan.strategy.name == "point":
        raw_geometry = _coerce_binary_geometry_value(
            row.get(plan.strategy.source_geometry_column or "")
        ) if plan.strategy.source_geometry_column else None
        lon = _normalize_scalar(row.get("LON"))
        lat = _normalize_scalar(row.get("LAT"))
        params.extend([raw_geometry, raw_geometry, lon, lat, lon, lat])
    return tuple(params)


def _iter_table_batches(plan: TablePlan, row_group_index: int, batch_size: int) -> Iterable[List[Tuple[Any, ...]]]:
    pq, _patypes = _load_pyarrow()
    parquet_file = pq.ParquetFile(plan.parquet_path)
    for batch in parquet_file.iter_batches(
        row_groups=[row_group_index],
        columns=list(plan.read_columns),
        batch_size=batch_size,
    ):
        rows = batch.to_pylist()
        yield [_row_to_params(plan, row) for row in rows]


def _geometry_param_index(plan: TablePlan) -> Optional[int]:
    if not plan.has_geometry:
        return None
    return len([column for column in plan.columns if column.name != "geometry"])


def _summarize_geometry_param(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return f"bytes(len={len(value)}, prefix={value[:16].hex()})"
    summary = repr(value)
    if len(summary) > 80:
        summary = summary[:77] + "..."
    return f"{type(value).__name__}({summary})"


def _summarize_batch_geometry(plan: TablePlan, batch_params: Sequence[Tuple[Any, ...]]) -> str:
    geometry_index = _geometry_param_index(plan)
    if geometry_index is None:
        return "geometry=not_applicable"
    geometry_values = [
        row[geometry_index]
        for row in batch_params
        if len(row) > geometry_index
    ]
    non_null = [value for value in geometry_values if value is not None]
    samples = ", ".join(_summarize_geometry_param(value) for value in non_null[:3])
    return (
        f"geometry_non_null_rows={len(non_null)}/{len(batch_params)} "
        f"samples=[{samples or 'none'}]"
    )


def _raise_batch_import_error(
    plan: TablePlan,
    row_group_index: int,
    batch_index: int,
    batch_params: Sequence[Tuple[Any, ...]],
    exc: Exception,
) -> None:
    raise RuntimeError(
        f"[FloodSQL][{plan.table_name}] batch import failed "
        f"row_group={row_group_index + 1}/{plan.num_row_groups} "
        f"batch_index={batch_index} batch_rows={len(batch_params)} "
        f"strategy={plan.strategy.name} {_summarize_batch_geometry(plan, batch_params)} "
        f"error={exc}"
    ) from exc


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_tables": {}, "table_progress": {}}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _mark_row_group_completed(
    checkpoint: Dict[str, Any],
    checkpoint_path: Path,
    plan: TablePlan,
    row_group_index: int,
) -> None:
    checkpoint.setdefault("table_progress", {})[plan.table_name] = {
        "last_completed_row_group": row_group_index,
        "row_groups_total": plan.num_row_groups,
    }
    _write_json(checkpoint_path, checkpoint)


def _mark_table_completed(
    checkpoint: Dict[str, Any],
    checkpoint_path: Path,
    plan: TablePlan,
    validation: Dict[str, Any],
) -> None:
    checkpoint.setdefault("completed_tables", {})[plan.table_name] = {
        "status": "ok",
        "rows": validation.get("target_row_count"),
        "source_rows": validation.get("source_row_count"),
        "num_row_groups": plan.num_row_groups,
    }
    checkpoint.setdefault("table_progress", {}).pop(plan.table_name, None)
    _write_json(checkpoint_path, checkpoint)


def _ensure_database(psycopg2, args: Dict[str, Any]) -> None:
    conn = psycopg2.connect(
        host=args["host"],
        port=args["port"],
        database=args["maintenance_db"],
        user=args["user"],
        password=args["password"],
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (args["database"],))
    if cur.fetchone() is None:
        cur.execute(f'CREATE DATABASE "{args["database"]}"')
    cur.close()
    conn.close()


def _connect_target(psycopg2, args: Dict[str, Any]):
    conn = psycopg2.connect(
        host=args["host"],
        port=args["port"],
        database=args["database"],
        user=args["user"],
        password=args["password"],
    )
    conn.autocommit = False
    return conn


def _table_exists(cursor, table_name: str) -> bool:
    cursor.execute("SELECT to_regclass(%s)", (table_name,))
    return cursor.fetchone()[0] is not None


def _recreate_table(cursor, plan: TablePlan) -> None:
    cursor.execute(f"DROP TABLE IF EXISTS {_quote_ident(plan.table_name)} CASCADE")
    cursor.execute(_build_create_table_sql(plan))


def _create_indexes(cursor, plan: TablePlan) -> None:
    for column_name in INDEX_COLUMNS.get(plan.table_name, []):
        normalized_column_name = _normalize_identifier(column_name)
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{plan.table_name}_{column_name.lower()} "
            f"ON {_quote_ident(plan.table_name)} ({_quote_ident(normalized_column_name)})"
        )
    if plan.has_geometry:
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{plan.table_name}_geometry "
            f"ON {_quote_ident(plan.table_name)} USING GIST (geometry)"
        )


def _collect_validation(cursor, plan: TablePlan) -> Dict[str, Any]:
    cursor.execute(f"SELECT COUNT(*) FROM {_quote_ident(plan.table_name)}")
    target_row_count = cursor.fetchone()[0]

    geometry_non_null_count = None
    invalid_geometry_count = None
    if plan.has_geometry:
        cursor.execute(
            f"SELECT COUNT(*) FROM {_quote_ident(plan.table_name)} WHERE geometry IS NOT NULL"
        )
        geometry_non_null_count = cursor.fetchone()[0]
        cursor.execute(
            f"SELECT COUNT(*) FROM {_quote_ident(plan.table_name)} "
            "WHERE geometry IS NOT NULL AND NOT ST_IsValid(geometry)"
        )
        invalid_geometry_count = cursor.fetchone()[0]

    return {
        "source_row_count": plan.total_rows,
        "target_row_count": target_row_count,
        "geometry_non_null_count": geometry_non_null_count,
        "invalid_geometry_count": invalid_geometry_count,
    }


def _validate_existing_table(cursor, plan: TablePlan) -> Dict[str, Any]:
    if not _table_exists(cursor, plan.table_name):
        return {
            "status": "missing_target_table",
            "source_row_count": plan.total_rows,
            "target_row_count": 0,
            "geometry_non_null_count": None,
            "invalid_geometry_count": None,
        }
    result = _collect_validation(cursor, plan)
    result["status"] = "validated"
    return result


def _import_single_table(
    conn,
    extras,
    checkpoint: Dict[str, Any],
    checkpoint_path: Path,
    plan: TablePlan,
    *,
    batch_size: int,
    resume: bool,
) -> Dict[str, Any]:
    cur = conn.cursor()
    try:
        table_started_at = time.monotonic()
        progress = checkpoint.get("table_progress", {}).get(plan.table_name, {})
        last_completed_row_group = int(progress.get("last_completed_row_group", -1))
        next_row_group = last_completed_row_group + 2 if plan.num_row_groups > 0 else 0

        _log(
            f"[FloodSQL][{plan.table_name}] start "
            f"strategy={plan.strategy.name} "
            f"source_rows={plan.total_rows} row_groups={plan.num_row_groups} "
            f"resume={resume} next_row_group={next_row_group}"
        )

        if not resume or last_completed_row_group < 0:
            _log(f"[FloodSQL][{plan.table_name}] recreating target table")
            _recreate_table(cur, plan)
            conn.commit()
            last_completed_row_group = -1
        elif not _table_exists(cur, plan.table_name):
            _log(f"[FloodSQL][{plan.table_name}] checkpoint found but target table missing; recreating")
            _recreate_table(cur, plan)
            conn.commit()
            last_completed_row_group = -1
        else:
            _log(
                f"[FloodSQL][{plan.table_name}] resuming from checkpoint "
                f"completed_row_groups={last_completed_row_group + 1}/{plan.num_row_groups}"
            )

        target_columns, template = _build_insert_template(plan)
        insert_sql = (
            f"INSERT INTO {_quote_ident(plan.table_name)} "
            f"({', '.join(_quote_ident(column) for column in target_columns)}) VALUES %s"
        )

        for row_group_index in range(last_completed_row_group + 1, plan.num_row_groups):
            row_group_started_at = time.monotonic()
            row_group_rows = 0
            for batch_index, batch_params in enumerate(_iter_table_batches(plan, row_group_index, batch_size), start=1):
                if not batch_params:
                    continue
                row_group_rows += len(batch_params)
                try:
                    extras.execute_values(
                        cur,
                        insert_sql,
                        batch_params,
                        template=template,
                        page_size=batch_size,
                    )
                except Exception as exc:
                    _raise_batch_import_error(plan, row_group_index, batch_index, batch_params, exc)
            conn.commit()
            _mark_row_group_completed(checkpoint, checkpoint_path, plan, row_group_index)
            _log(
                f"[FloodSQL][{plan.table_name}] row-group "
                f"{_progress_bar(row_group_index + 1, plan.num_row_groups)} "
                f"rows={row_group_rows} elapsed={time.monotonic() - row_group_started_at:.1f}s"
            )

        _log(f"[FloodSQL][{plan.table_name}] creating indexes")
        _create_indexes(cur, plan)
        conn.commit()
        validation = _collect_validation(cur, plan)
        validation["status"] = "imported"
        _mark_table_completed(checkpoint, checkpoint_path, plan, validation)
        _log(
            f"[FloodSQL][{plan.table_name}] done "
            f"target_rows={validation.get('target_row_count')} "
            f"geometry_non_null={validation.get('geometry_non_null_count')} "
            f"invalid_geometry={validation.get('invalid_geometry_count')} "
            f"elapsed={time.monotonic() - table_started_at:.1f}s"
        )
        return validation
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def _resolve_checkpoint_path(report_path: Path, checkpoint_path: Optional[str | Path]) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path).expanduser().resolve()
    return report_path.with_name("migration_checkpoint.json")


def run_floodsql_migration(
    *,
    data_root: str | Path,
    metadata_path: str | Path,
    report_path: str | Path,
    checkpoint_path: Optional[str | Path] = None,
    host: str,
    port: int,
    database: str,
    maintenance_db: str,
    user: str,
    password: str,
    batch_size: int = 5000,
    resume: bool = False,
    validate_only: bool = False,
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    started_at = time.monotonic()
    metadata_file = Path(metadata_path).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    checkpoint_file = _resolve_checkpoint_path(report_file, checkpoint_path)
    metadata = _load_metadata(metadata_file)
    layout = discover_floodsql_data_layout(data_root, metadata)
    table_plans = {
        table_name: _build_table_plan(table_name, metadata[table_name], layout.parquet_root)
        for table_name in EXPECTED_TABLES
    }

    report: Dict[str, Any] = {
        "database": database,
        "requested_data_root": str(Path(data_root).expanduser()),
        "parquet_root": str(layout.parquet_root),
        "layout": layout.layout_name,
        "metadata": str(metadata_file),
        "batch_size": batch_size,
        "resume": bool(resume),
        "validate_only": bool(validate_only),
        "tables": {},
        "summary": {
            "tables_total": len(EXPECTED_TABLES),
            "tables_imported": 0,
            "tables_validated": 0,
            "tables_skipped_from_checkpoint": 0,
        },
    }

    _log("=" * 70)
    _log("[FloodSQL] migration started")
    _log(
        f"[FloodSQL] target={host}:{port}/{database} "
        f"validate_only={validate_only} resume={resume} batch_size={batch_size}"
    )
    _log(f"[FloodSQL] requested_data_root={Path(data_root).expanduser().resolve()}")
    _log(f"[FloodSQL] parquet_root={layout.parquet_root} layout={layout.layout_name}")
    _log(f"[FloodSQL] metadata={metadata_file}")
    _log(f"[FloodSQL] report={report_file}")
    _log(f"[FloodSQL] checkpoint={checkpoint_file}")
    _log("=" * 70)

    if validate_only:
        psycopg2, _extras = _load_pg_dependencies()
        conn = _connect_target(
            psycopg2,
            {
                "host": host,
                "port": port,
                "database": database,
                "maintenance_db": maintenance_db,
                "user": user,
                "password": password,
            },
        )
        cur = conn.cursor()
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            conn.commit()
            _log("[FloodSQL] phase 1/1 validate existing tables")
            for index, table_name in enumerate(EXPECTED_TABLES, start=1):
                _log(
                    f"[FloodSQL][validate][{index}/{len(EXPECTED_TABLES)}] "
                    f"{table_name} source_rows={table_plans[table_name].total_rows}"
                )
                result = _validate_existing_table(cur, table_plans[table_name])
                report["tables"][table_name] = result
                report["summary"]["tables_validated"] += 1
                _log(
                    f"[FloodSQL][validate][{table_name}] status={result.get('status')} "
                    f"target_rows={result.get('target_row_count')} "
                    f"invalid_geometry={result.get('invalid_geometry_count')}"
                )
        finally:
            cur.close()
            conn.close()
        _write_json(report_file, report)
        _log(
            f"[FloodSQL] validation completed tables={report['summary']['tables_validated']} "
            f"elapsed={time.monotonic() - started_at:.1f}s"
        )
        _log("=" * 70)
        return report

    psycopg2, extras = _load_pg_dependencies()
    _log("[FloodSQL] phase 1/3 ensure database")
    _ensure_database(
        psycopg2,
        {
            "host": host,
            "port": port,
            "database": database,
            "maintenance_db": maintenance_db,
            "user": user,
            "password": password,
        },
    )
    conn = _connect_target(
        psycopg2,
        {
            "host": host,
            "port": port,
            "database": database,
            "maintenance_db": maintenance_db,
            "user": user,
            "password": password,
        },
    )
    checkpoint = _load_checkpoint(checkpoint_file if resume else Path("__nonexistent__"))
    if not resume:
        checkpoint = {"completed_tables": {}, "table_progress": {}}
        _write_json(checkpoint_file, checkpoint)

    try:
        _log("[FloodSQL] phase 2/3 import tables")
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        conn.commit()
        cur.close()

        for index, table_name in enumerate(EXPECTED_TABLES, start=1):
            plan = table_plans[table_name]
            completed = checkpoint.get("completed_tables", {}).get(table_name, {})
            _log(
                f"[FloodSQL][table {index}/{len(EXPECTED_TABLES)}] "
                f"{table_name} strategy={plan.strategy.name} source_rows={plan.total_rows}"
            )
            if resume and completed.get("status") == "ok":
                cur = conn.cursor()
                try:
                    validated = _validate_existing_table(cur, plan)
                    if validated.get("status") != "missing_target_table":
                        validated["status"] = "skipped_from_checkpoint"
                        report["tables"][table_name] = validated
                        report["summary"]["tables_skipped_from_checkpoint"] += 1
                        _log(
                            f"[FloodSQL][{table_name}] skipped from checkpoint "
                            f"target_rows={validated.get('target_row_count')}"
                        )
                        continue
                finally:
                    cur.close()
                checkpoint.setdefault("completed_tables", {}).pop(table_name, None)
                _write_json(checkpoint_file, checkpoint)

            report["tables"][table_name] = _import_single_table(
                conn,
                extras,
                checkpoint,
                checkpoint_file,
                plan,
                batch_size=batch_size,
                resume=resume,
            )
            report["summary"]["tables_imported"] += 1
    except Exception as exc:
        report["error"] = str(exc)
        _write_json(report_file, report)
        _log(f"[FloodSQL] failed error={exc}")
        _log("=" * 70)
        raise
    finally:
        conn.close()

    _write_json(report_file, report)
    _log("[FloodSQL] phase 3/3 write report")
    _log(
        f"[FloodSQL] completed imported={report['summary']['tables_imported']} "
        f"checkpoint_skipped={report['summary']['tables_skipped_from_checkpoint']} "
        f"elapsed={time.monotonic() - started_at:.1f}s"
    )
    _log("=" * 70)
    return report


__all__ = [
    "ATTRIBUTE_ONLY_TABLES",
    "BINARY_GEOMETRY_TABLES",
    "EXPECTED_TABLES",
    "INDEX_COLUMNS",
    "POINT_TABLES",
    "ColumnPlan",
    "SourceLayout",
    "TablePlan",
    "TableStrategy",
    "determine_table_strategy",
    "discover_floodsql_data_layout",
    "run_floodsql_migration",
]
