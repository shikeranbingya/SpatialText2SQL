from __future__ import annotations

import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Iterable, Literal, Optional

import chardet
import fiona
import geopandas as gpd
from osgeo import gdal
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from tqdm import tqdm

from .logging_config import spatial_logger as logger


IfExistsMode = Literal["fail", "replace", "append"]

gdal.SetConfigOption("SHAPE_RESTORE_SHX", "YES")
gdal.SetConfigOption("SHAPE_ENCODING", "")


class SpatialDBImporter(ABC):
    """Abstract base class for spatial database writers."""

    def __init__(self, db_url: str):
        self.db_url = db_url

    @abstractmethod
    def write(
        self,
        gdf: gpd.GeoDataFrame,
        table_name: str,
        schema: Optional[str] = None,
        if_exists: IfExistsMode = "replace",
    ) -> None:
        """Write a GeoDataFrame into the target database."""


class PostGISImporter(SpatialDBImporter):
    """Importer for PostgreSQL/PostGIS targets."""

    def write(
        self,
        gdf: gpd.GeoDataFrame,
        table_name: str,
        schema: Optional[str] = None,
        if_exists: IfExistsMode = "replace",
    ) -> None:
        try:
            engine: Engine = create_engine(self.db_url)
            gdf.to_postgis(
                name=table_name,
                con=engine,
                schema=schema,
                if_exists=if_exists,
                index=False,
            )
            target = f"{schema}.{table_name}" if schema else table_name
            logger.info("Data successfully written to PostGIS table: %s", target)
        except Exception as exc:
            logger.error("Failed to write to PostGIS: %s", exc)
            raise


class SpatiaLiteImporter(SpatialDBImporter):
    """Importer for SQLite/SpatiaLite targets."""

    def write(
        self,
        gdf: gpd.GeoDataFrame,
        table_name: str,
        schema: Optional[str] = None,
        if_exists: IfExistsMode = "replace",
    ) -> None:
        del schema  # SpatiaLite does not use PostgreSQL schemas.
        db_path = self.db_url.split(":///")[-1]
        existing_layers: list[str] = []
        if os.path.exists(db_path):
            try:
                existing_layers = fiona.listlayers(db_path)
            except Exception as exc:
                logger.debug("Could not list layers in %s: %s", db_path, exc)

        mode = "w" if not os.path.exists(db_path) else "a"

        if table_name in existing_layers:
            if if_exists == "fail":
                raise FileExistsError(f"Layer '{table_name}' already exists in {db_path}")
            if if_exists == "replace":
                try:
                    with sqlite3.connect(db_path) as conn:
                        conn.enable_load_extension(True)
                        try:
                            conn.load_extension("mod_spatialite")
                        except Exception:
                            pass
                        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                        try:
                            conn.execute(
                                "DELETE FROM geometry_columns WHERE f_table_name = ?",
                                (table_name,),
                            )
                            conn.execute(
                                "DELETE FROM spatial_ref_sys WHERE srid NOT IN "
                                "(SELECT srid FROM geometry_columns)"
                            )
                        except sqlite3.OperationalError:
                            pass
                        conn.commit()
                    mode = "a"
                    logger.info("Dropped existing layer '%s' for replacement.", table_name)
                except Exception as exc:
                    logger.warning(
                        "Could not drop existing layer '%s' via sqlite3: %s",
                        table_name,
                        exc,
                    )

        try:
            gdf.to_file(
                db_path,
                layer=table_name,
                driver="SQLite",
                engine="fiona",
                spatialite=True,
                mode=mode,
            )
            logger.info(
                "Data successfully written to SpatiaLite: %s (Layer: %s)",
                db_path,
                table_name,
            )
        except Exception as exc:
            logger.error("Failed to write to SpatiaLite: %s", exc)
            raise


def get_importer(db_url: str) -> SpatialDBImporter:
    """Return the matching database importer for a SQLAlchemy URL."""

    if db_url.startswith("postgresql"):
        return PostGISImporter(db_url)
    if db_url.startswith("sqlite"):
        return SpatiaLiteImporter(db_url)
    raise ValueError(f"Unsupported database type: {db_url}")


def detect_shp_encoding(shp_path: str) -> str:
    """Best-effort DBF encoding detection for shapefiles."""

    dbf_path = os.path.splitext(shp_path)[0] + ".dbf"
    if not os.path.exists(dbf_path):
        return "utf-8"

    try:
        with open(dbf_path, "rb") as handle:
            raw_data = handle.read(10000)
        result = chardet.detect(raw_data)
        return result.get("encoding") or "utf-8"
    except Exception:
        return "utf-8"


def read_shp_with_fallback_encoding(
    shp_path: str,
    layer_name: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Read a shapefile with multiple encoding fallbacks."""

    detected_encoding = detect_shp_encoding(shp_path)
    candidate_encodings = [
        "utf-8",
        "gb18030",
        "gbk",
        detected_encoding,
        "cp936",
        "MacRoman",
        "latin-1",
    ]
    candidate_encodings = list(dict.fromkeys(candidate_encodings))

    for encoding in candidate_encodings:
        try:
            if layer_name:
                gdf = gpd.read_file(shp_path, layer=layer_name, encoding=encoding)
            else:
                gdf = gpd.read_file(shp_path, encoding=encoding)
            logger.info("Successfully read data with encoding %s", encoding)
            return gdf
        except Exception:
            continue

    logger.warning("All tested encodings failed. Falling back to tolerant reading mode.")
    with fiona.open(
        shp_path,
        layer=layer_name,
        encoding="utf-8",
        errors="replace",
    ) as src:
        features = []
        for feat in src:
            clean_properties = {}
            for key, value in feat["properties"].items():
                if isinstance(value, str):
                    try:
                        clean_properties[key] = value.encode(
                            "utf-8",
                            errors="replace",
                        ).decode("utf-8")
                    except Exception:
                        clean_properties[key] = str(value)
                else:
                    clean_properties[key] = value
            features.append(
                {"geometry": feat["geometry"], "properties": clean_properties}
            )
        return gpd.GeoDataFrame.from_features(features, crs=src.crs)


def normalize_geodataframe(
    gdf: gpd.GeoDataFrame,
    default_epsg: int = 4326,
) -> gpd.GeoDataFrame:
    """Normalize CRS, repair geometries, and clean object columns."""

    if gdf.crs is None:
        logger.warning("Input data missing CRS, assuming EPSG:%s", default_epsg)
        gdf = gdf.set_crs(epsg=default_epsg)
    else:
        source_epsg = gdf.crs.to_epsg()
        if source_epsg != default_epsg:
            logger.info("Converting CRS from %s to EPSG:%s", source_epsg, default_epsg)
            gdf = gdf.to_crs(epsg=default_epsg)

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception as exc:
        logger.warning("Failed to repair geometries with make_valid(): %s", exc)

    for column in gdf.columns:
        if gdf[column].dtype == "object":
            gdf[column] = gdf[column].apply(
                lambda value: value.replace("\x00", "") if isinstance(value, str) else value
            )

    return gdf


def collect_input_files(input_path: str, extensions: tuple[str, ...]) -> list[str]:
    """Collect matching files from a file path or directory tree."""

    normalized_exts = tuple(ext.lower() for ext in extensions)

    if os.path.isdir(input_path):
        matches: list[str] = []
        for root, _dirs, files in os.walk(input_path):
            for name in files:
                if name.lower().endswith(normalized_exts):
                    matches.append(os.path.join(root, name))
        return sorted(matches)

    return [input_path]


def iter_with_progress(
    items: Iterable[str],
    desc: str,
    unit: str,
) -> Iterable[str]:
    """Wrap an iterable with tqdm when available."""
    return tqdm(list(items), desc=desc, unit=unit)


def _process_and_import(
    shp_path: str,
    importer: SpatialDBImporter,
    table_name: str,
    schema: Optional[str],
    if_exists: IfExistsMode,
) -> None:
    try:
        layers = fiona.listlayers(shp_path)
    except Exception:
        layers = []

    if len(layers) > 1:
        logger.info("Detected %s layers in %s. Importing them one by one.", len(layers), shp_path)
        for layer_name in layers:
            try:
                logger.info("Processing layer: %s", layer_name)
                gdf = read_shp_with_fallback_encoding(shp_path, layer_name=layer_name)
                if gdf.empty:
                    continue
                importer.write(
                    normalize_geodataframe(gdf),
                    layer_name,
                    schema,
                    if_exists,
                )
            except Exception as exc:
                logger.error("Failed to import layer %s: %s", layer_name, exc)
                continue
        return

    logger.info("Processing standard SHP file: %s", table_name)
    try:
        gdf = read_shp_with_fallback_encoding(shp_path)
    except Exception as exc:
        logger.error("Failed to read SHP file: %s", exc)
        raise

    if gdf.empty:
        logger.warning("No features found in SHP file: %s", shp_path)
        return

    importer.write(normalize_geodataframe(gdf), table_name, schema, if_exists)


def shp2db(
    input_path: str,
    db_url: str,
    table_name: Optional[str] = None,
    schema: Optional[str] = None,
    if_exists: IfExistsMode = "replace",
) -> None:
    """Import a SHP file or a directory of SHP files into PostGIS or SpatiaLite."""

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Path not found: {input_path}")

    importer = get_importer(db_url)

    if os.path.isdir(input_path):
        logger.info("Scanning directory recursively for SHP files: %s", input_path)
        shp_files = collect_input_files(input_path, (".shp",))
        if not shp_files:
            logger.warning("No SHP files found in directory tree: %s", input_path)
            return

        for shp_file in iter_with_progress(shp_files, desc="SHP Import Progress", unit="file"):
            derived_table_name = os.path.splitext(os.path.basename(shp_file))[0]
            try:
                _process_and_import(
                    shp_file,
                    importer,
                    derived_table_name,
                    schema,
                    if_exists,
                )
            except Exception as exc:
                logger.error("Failed to import %s: %s", shp_file, exc)
                continue
        return

    if not input_path.lower().endswith(".shp"):
        logger.warning("Input file extension is not .shp: %s", input_path)

    target_table = table_name or os.path.splitext(os.path.basename(input_path))[0]
    _process_and_import(input_path, importer, target_table, schema, if_exists)


__all__ = [
    "IfExistsMode",
    "PostGISImporter",
    "SpatialDBImporter",
    "SpatiaLiteImporter",
    "collect_input_files",
    "detect_shp_encoding",
    "get_importer",
    "iter_with_progress",
    "normalize_geodataframe",
    "read_shp_with_fallback_encoding",
    "shp2db",
]
