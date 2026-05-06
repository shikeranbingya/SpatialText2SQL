from __future__ import annotations

import os
from typing import Optional

import geopandas as gpd

from .logging_config import spatial_logger as logger
from .shp2db import (
    IfExistsMode,
    SpatialDBImporter,
    collect_input_files,
    get_importer,
    iter_with_progress,
    normalize_geodataframe,
)


def read_geojson(geojson_path: str) -> gpd.GeoDataFrame:
    """Read a GeoJSON file into a GeoDataFrame."""

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as exc:
        logger.error("Failed to read GeoJSON file %s: %s", geojson_path, exc)
        raise

    logger.info("Successfully read %s features from %s", len(gdf), geojson_path)
    return gdf


def _process_and_import(
    geojson_path: str,
    importer: SpatialDBImporter,
    table_name: str,
    schema: Optional[str],
    if_exists: IfExistsMode,
) -> None:
    gdf = read_geojson(geojson_path)
    if gdf.empty:
        logger.warning("No features found in GeoJSON file: %s", geojson_path)
        return
    importer.write(normalize_geodataframe(gdf), table_name, schema, if_exists)


def geojson2db(
    input_path: str,
    db_url: str,
    table_name: Optional[str] = None,
    schema: Optional[str] = None,
    if_exists: IfExistsMode = "replace",
) -> None:
    """Import a GeoJSON file or a directory of GeoJSON files into a spatial database."""

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Path not found: {input_path}")

    importer = get_importer(db_url)

    if os.path.isdir(input_path):
        logger.info("Scanning directory recursively for GeoJSON files: %s", input_path)
        geojson_files = collect_input_files(input_path, (".geojson", ".json"))
        if not geojson_files:
            logger.warning("No GeoJSON files found in directory tree: %s", input_path)
            return

        for geojson_file in iter_with_progress(
            geojson_files,
            desc="GeoJSON Import Progress",
            unit="file",
        ):
            derived_table_name = os.path.splitext(os.path.basename(geojson_file))[0]
            try:
                _process_and_import(
                    geojson_file,
                    importer,
                    derived_table_name,
                    schema,
                    if_exists,
                )
            except Exception as exc:
                logger.error("Failed to import %s: %s", geojson_file, exc)
                continue
        return

    if not input_path.lower().endswith((".geojson", ".json")):
        logger.warning(
            "Input file extension is not .geojson or .json: %s",
            input_path,
        )

    target_table = table_name or os.path.splitext(os.path.basename(input_path))[0]
    _process_and_import(input_path, importer, target_table, schema, if_exists)


__all__ = ["geojson2db", "read_geojson"]
