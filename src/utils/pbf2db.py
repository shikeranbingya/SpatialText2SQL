from __future__ import annotations

import os
from typing import Optional

import geopandas as gpd

from .logging_config import init_pbf_logging, pbf_logger
from .shp2db import (
    IfExistsMode,
    collect_input_files,
    get_importer,
    iter_with_progress,
    normalize_geodataframe,
)


def read_pbf_layers(pbf_path: str) -> list[tuple[str, object]]:
    """Read standard OSM layers from a PBF file."""

    osm_layers = [
        "points",
        "lines",
        "multilinestrings",
        "multipolygons",
        "other_relations",
    ]
    valid_layers: list[tuple[str, object]] = []

    pbf_logger.info("Reading PBF file: %s", pbf_path)

    for layer in osm_layers:
        try:
            gdf = gpd.read_file(pbf_path, layer=layer, engine="pyogrio")
            if not gdf.empty:
                pbf_logger.info("Layer '%s' found with %s features.", layer, len(gdf))
                valid_layers.append((layer, gdf))
        except Exception:
            continue

    return valid_layers


def pbf2db(
    input_path: str,
    db_url: str,
    schema: Optional[str] = None,
    if_exists: IfExistsMode = "replace",
) -> None:
    """Import OSM PBF files into PostGIS or SpatiaLite."""

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Path not found: {input_path}")

    importer = get_importer(db_url)

    if os.path.isdir(input_path):
        pbf_logger.info("Scanning directory recursively for PBF files: %s", input_path)
        files = collect_input_files(input_path, (".pbf",))
        if not files:
            pbf_logger.warning("No PBF files found in directory tree: %s", input_path)
            return
    else:
        if not input_path.lower().endswith(".pbf"):
            pbf_logger.warning("Input file extension is not .pbf: %s", input_path)
        files = [input_path]

    for pbf_file in iter_with_progress(files, desc="PBF Import Progress", unit="file"):
        file_base_name = os.path.splitext(os.path.basename(pbf_file))[0]
        try:
            layers = read_pbf_layers(pbf_file)
            for layer_name, gdf in layers:
                target_table = f"{file_base_name}_{layer_name}".lower()
                importer.write(
                    normalize_geodataframe(gdf),
                    target_table,
                    schema,
                    if_exists,
                )
        except Exception as exc:
            pbf_logger.error("Failed to process %s: %s", pbf_file, exc)


__all__ = ["init_pbf_logging", "pbf2db", "read_pbf_layers"]
