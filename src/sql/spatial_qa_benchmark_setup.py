"""Benchmark setup helpers for the Spatial QA PostgreSQL database."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import psycopg2


PROFILE_NAME = "spatial_qa_geography_v1"
PROFILE_DESCRIPTION = (
    "Geography expression indexes and lookup indexes required for Spatial QA benchmark queries."
)

SPATIAL_QA_BENCHMARK_INDEX_SPECS: List[Dict[str, str]] = [
    {
        "name": "idx_counties_geom_geog",
        "table": "counties",
        "kind": "gist_geography_expr",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_counties_geom_geog "
        "ON public.counties USING GIST ((geom::geography))",
    },
    {
        "name": "idx_poi_geom_geog",
        "table": "poi",
        "kind": "gist_geography_expr",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_poi_geom_geog "
        "ON public.poi USING GIST ((geom::geography))",
    },
    {
        "name": "idx_ghcn_geom_geog",
        "table": "ghcn",
        "kind": "gist_geography_expr",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ghcn_geom_geog "
        "ON public.ghcn USING GIST ((geom::geography))",
    },
    {
        "name": "idx_ne_protected_areas_geom_geog",
        "table": "ne_protected_areas",
        "kind": "gist_geography_expr",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ne_protected_areas_geom_geog "
        "ON public.ne_protected_areas USING GIST ((geom::geography))",
    },
    {
        "name": "idx_ne_time_zones_geom_geog",
        "table": "ne_time_zones",
        "kind": "gist_geography_expr",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ne_time_zones_geom_geog "
        "ON public.ne_time_zones USING GIST ((geom::geography))",
    },
    {
        "name": "idx_roads_geom_geog",
        "table": "roads",
        "kind": "gist_geography_expr",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_roads_geom_geog "
        "ON public.roads USING GIST ((geom::geography))",
    },
    {
        "name": "idx_blockgroups_geom_geog",
        "table": "blockgroups",
        "kind": "gist_geography_expr",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_blockgroups_geom_geog "
        "ON public.blockgroups USING GIST ((geom::geography))",
    },
    {
        "name": "idx_tracts_geom_geog",
        "table": "tracts",
        "kind": "gist_geography_expr",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tracts_geom_geog "
        "ON public.tracts USING GIST ((geom::geography))",
    },
    {
        "name": "idx_poi_name",
        "table": "poi",
        "kind": "btree_lookup",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_poi_name ON public.poi (name)",
    },
    {
        "name": "idx_ne_protected_areas_nps_region",
        "table": "ne_protected_areas",
        "kind": "btree_lookup",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ne_protected_areas_nps_region "
        "ON public.ne_protected_areas (nps_region)",
    },
    {
        "name": "idx_ne_time_zones_name",
        "table": "ne_time_zones",
        "kind": "btree_lookup",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ne_time_zones_name "
        "ON public.ne_time_zones (name)",
    },
    {
        "name": "idx_ghcn_elev",
        "table": "ghcn",
        "kind": "btree_lookup",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ghcn_elev ON public.ghcn (elev)",
    },
    {
        "name": "idx_ghcn_element_value",
        "table": "ghcn",
        "kind": "btree_lookup",
        "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ghcn_element_value "
        "ON public.ghcn (element, value)",
    },
]

SPATIAL_QA_ANALYZE_TABLES = [
    "counties",
    "poi",
    "ghcn",
    "ne_protected_areas",
    "ne_time_zones",
    "roads",
    "blockgroups",
    "tracts",
]


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _build_base_metadata(db_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": "spatial_qa",
        "status": "unknown",
        "checked_at": _timestamp(),
        "index_profile": PROFILE_NAME,
        "description": PROFILE_DESCRIPTION,
        "database_host": db_config.get("host"),
        "database_port": db_config.get("port"),
        "database_name": db_config.get("database"),
        "required_indexes": [
            {
                "name": spec["name"],
                "table": spec["table"],
                "kind": spec["kind"],
            }
            for spec in SPATIAL_QA_BENCHMARK_INDEX_SPECS
        ],
        "present_indexes": [],
        "missing_indexes": [],
        "analyze_tables": list(SPATIAL_QA_ANALYZE_TABLES),
    }


def _connect(db_config: Dict[str, Any]):
    return psycopg2.connect(
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["database"],
        user=db_config["user"],
        password=db_config["password"],
    )


def inspect_spatial_qa_benchmark_setup(db_config: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _build_base_metadata(db_config)
    conn = None
    cur = None
    try:
        conn = _connect(db_config)
        cur = conn.cursor()
        expected_names = [spec["name"] for spec in SPATIAL_QA_BENCHMARK_INDEX_SPECS]
        cur.execute(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND indexname = ANY(%s)
            """,
            (expected_names,),
        )
        existing_names = {row[0] for row in cur.fetchall()}
    except Exception as exc:  # pragma: no cover - exercised via pipeline integration
        metadata["status"] = "check_failed"
        metadata["error"] = str(exc)
        return metadata
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    metadata["present_indexes"] = sorted(existing_names)
    metadata["missing_indexes"] = [
        spec["name"]
        for spec in SPATIAL_QA_BENCHMARK_INDEX_SPECS
        if spec["name"] not in existing_names
    ]
    metadata["status"] = "ready" if not metadata["missing_indexes"] else "missing"
    return metadata


def apply_spatial_qa_benchmark_setup(
    db_config: Dict[str, Any],
    *,
    concurrently: bool = True,
    analyze: bool = True,
    create_missing_only: bool = True,
) -> Dict[str, Any]:
    before = inspect_spatial_qa_benchmark_setup(db_config)
    if before.get("status") == "check_failed":
        raise RuntimeError(before.get("error") or "Failed to inspect current index status")

    missing_names = set(before.get("missing_indexes") or [])
    planned_specs = SPATIAL_QA_BENCHMARK_INDEX_SPECS
    if create_missing_only:
        planned_specs = [
            spec for spec in SPATIAL_QA_BENCHMARK_INDEX_SPECS if spec["name"] in missing_names
        ]

    conn = _connect(db_config)
    conn.autocommit = True
    cur = conn.cursor()
    executed_indexes: List[str] = []
    analyzed_tables: List[str] = []
    try:
        for spec in planned_specs:
            sql = spec["sql"]
            if not concurrently:
                sql = sql.replace("CREATE INDEX CONCURRENTLY", "CREATE INDEX")
            cur.execute(sql)
            executed_indexes.append(spec["name"])

        if analyze:
            for table_name in SPATIAL_QA_ANALYZE_TABLES:
                cur.execute(f"ANALYZE public.{table_name}")
                analyzed_tables.append(table_name)
    finally:
        try:
            cur.close()
        finally:
            conn.close()

    after = inspect_spatial_qa_benchmark_setup(db_config)
    after["applied_indexes"] = executed_indexes
    after["analyzed_tables"] = analyzed_tables
    after["checked_before_apply"] = before
    return after
