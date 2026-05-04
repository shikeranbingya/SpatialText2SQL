"""PostGIS function library loading and sampling for SQL synthesis."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.synthesis.database.models import SynthesizedSpatialDatabase
from src.synthesis.database.utils import stable_jsonify, to_text, unique_preserve_order

from .models import DIFFICULTY_LEVELS, PostGISFunction

LOGGER = logging.getLogger(__name__)

SPATIAL_FUNCTION_BLACKLIST = {
    "st_estimatedextent",
    "st_memsize",
    "st_summary",
    "st_asgml",
    "st_askml",
    "st_asgeojson",
    "st_asbinary",
    "st_asewkb",
    "st_asewkt",
    "st_box2dfromgeohash",
}

CATEGORY_BY_NAME_PATTERN: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^st_(contains|within|intersects|touches|crosses|overlaps|covers|coveredby|disjoint)$", re.I), "spatial_predicate"),
    (re.compile(r"^st_(dwithin|intersects|contains|within|touches|crosses|overlaps)$", re.I), "spatial_join"),
    (re.compile(r"^st_(distance|area|length|perimeter|hausdorffdistance|maxdistance)$", re.I), "spatial_measurement"),
    (re.compile(r"^st_(buffer|intersection|difference|symdifference|transform|simplify|snapto(grid)?|makevalid|centroid|convexhull|envelope|unaryunion|union)$", re.I), "geometry_transformation"),
    (re.compile(r"^st_(makepoint|point|collect|setsrid|geomfromtext|geomfromgeojson|makeline|makepolygon)$", re.I), "geometry_constructor"),
    (re.compile(r"^st_(collect|union|unaryunion|extent|envelope)$", re.I), "spatial_aggregation"),
    (re.compile(r"^st_(transform|setsrid|srid)$", re.I), "coordinate_operation"),
    (re.compile(r"^st_(geometrytype|x|max|xmin|xmax|y|min|ymin|ymax|npoints|startpoint|endpoint|srid|isempty|isvalid)$", re.I), "geometry_accessor"),
    (re.compile(r"^st_(lineinterpolatepoint|linesubstring|locatealong|locatebetween)$", re.I), "linear_referencing"),
]


def infer_function_categories(function_name: str, description: str = "", chapter_info: str = "") -> list[str]:
    text = " ".join([function_name, description, chapter_info]).lower()
    categories: list[str] = []
    for pattern, category in CATEGORY_BY_NAME_PATTERN:
        if pattern.search(function_name):
            categories.append(category)
    if "bounding box" in text or "bbox" in text or function_name.lower() in {"st_xmax", "st_xmin", "st_ymax", "st_ymin"}:
        categories.append("bbox_operation")
    if "output" in text and "geometry_output" not in categories:
        categories.append("geometry_output")
    if not categories:
        categories.append("geometry_transformation")
    return unique_preserve_order(categories)


def infer_compatible_difficulties(categories: Sequence[str]) -> list[str]:
    category_set = set(categories)
    if {"spatial_aggregation", "linear_referencing"} & category_set:
        return list(DIFFICULTY_LEVELS[1:])
    if {"geometry_constructor"} & category_set:
        return ["medium", "hard", "extra-hard"]
    if {"spatial_join"} & category_set:
        return ["medium", "hard", "extra-hard"]
    if {"spatial_measurement", "spatial_predicate", "geometry_transformation", "geometry_accessor", "bbox_operation"} & category_set:
        return list(DIFFICULTY_LEVELS)
    return list(DIFFICULTY_LEVELS)


def _truncate_text(value: str, max_chars: int) -> str:
    text = to_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_signature(function_name: str, signature: str, input_args: Sequence[str]) -> str:
    text = to_text(signature)
    if text:
        return re.sub(r"\s+", " ", text).strip()
    args = ", ".join(arg.strip() for arg in input_args if to_text(arg))
    return f"{function_name}({args})" if args else f"{function_name}(...)"


def _should_exclude_function(
    function_name: str,
    description: str,
    chapter_info: str,
    categories: Sequence[str],
    exclude_categories: Sequence[str],
) -> bool:
    lowered_name = function_name.lower()
    lowered_text = f"{description} {chapter_info} {' '.join(categories)}".lower()
    if not lowered_name.startswith("st_"):
        return True
    if lowered_name in SPATIAL_FUNCTION_BLACKLIST:
        return True
    if any(token in lowered_text or token in lowered_name for token in exclude_categories):
        return True
    if "raster" in lowered_text or "topology" in lowered_text:
        return True
    if any(token in lowered_text for token in ("version", "management", "metadata", "debug", "extension", "address_standardizer")):
        return True
    return False


def _extract_example_sqls(record: Mapping[str, Any], max_examples: int = 2) -> list[str]:
    examples: list[str] = []
    for example in stable_jsonify(record.get("examples")) or []:
        if not isinstance(example, Mapping):
            continue
        for step in example.get("steps", []):
            if not isinstance(step, Mapping):
                continue
            sql = to_text(step.get("sql"))
            if sql:
                examples.append(_truncate_text(sql, 240))
                if len(examples) >= max_examples:
                    return unique_preserve_order(examples)
    return unique_preserve_order(examples)


class PostGISFunctionLibrary:
    def __init__(self, functions: Sequence[PostGISFunction]):
        ordered = sorted(functions, key=lambda item: (item.function_name.lower(), item.signature.lower()))
        self.functions = list(ordered)
        self.by_name: dict[str, list[PostGISFunction]] = {}
        for item in self.functions:
            self.by_name.setdefault(item.function_name.lower(), []).append(item)

    @classmethod
    def load(
        cls,
        json_path: str | Path,
        markdown_path: str | Path,
        exclude_categories: Sequence[str] | None = None,
    ) -> "PostGISFunctionLibrary":
        exclude_categories = [to_text(item).lower() for item in (exclude_categories or ["raster", "topology"]) if to_text(item)]
        merged: dict[tuple[str, str], PostGISFunction] = {}

        for item in cls._load_from_json(json_path, exclude_categories):
            merged[(item.function_name.lower(), item.signature.lower())] = item

        markdown_functions = cls._load_from_markdown(markdown_path)
        for item in markdown_functions:
            matched = [key for key in merged if key[0] == item.function_name.lower()]
            if matched:
                for key in matched:
                    existing = merged[key]
                    existing.source = unique_preserve_order(existing.source + item.source)
                    existing.metadata = {
                        **stable_jsonify(existing.metadata),
                        **stable_jsonify(item.metadata),
                    }
            else:
                merged[(item.function_name.lower(), item.signature.lower())] = item

        functions = list(merged.values())
        if not functions:
            raise ValueError("PostGIS function library is empty after loading and filtering.")
        return cls(functions)

    @staticmethod
    def _load_from_json(
        json_path: str | Path,
        exclude_categories: Sequence[str],
    ) -> list[PostGISFunction]:
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"PostGIS function JSON not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Invalid PostGIS function JSON in {path}: expected a list.")

        items: list[PostGISFunction] = []
        for record in payload:
            if not isinstance(record, Mapping):
                continue
            description = _truncate_text(to_text(record.get("description")), 600)
            chapter_info = to_text(record.get("chapter_info"))
            source_file = to_text(record.get("source_file"))
            example_usages = _extract_example_sqls(record)
            definitions = record.get("function_definitions") or []
            if not isinstance(definitions, list):
                continue
            for definition in definitions:
                if not isinstance(definition, Mapping):
                    continue
                function_name = to_text(definition.get("function_name"))
                input_args = [to_text(arg) for arg in stable_jsonify(definition.get("arguments")) or [] if to_text(arg)]
                signature = _normalize_signature(
                    function_name,
                    definition.get("signature_str"),
                    input_args,
                )
                categories = infer_function_categories(function_name, description, chapter_info)
                if _should_exclude_function(function_name, description, chapter_info, categories, exclude_categories):
                    continue
                items.append(
                    PostGISFunction(
                        function_name=function_name,
                        signature=signature,
                        input_args=input_args,
                        return_type=to_text(definition.get("return_type")),
                        description=description,
                        example_usages=example_usages,
                        categories=categories,
                        compatible_difficulties=infer_compatible_difficulties(categories),
                        source=["postgis_extracted.json"],
                        metadata={
                            "chapter_info": chapter_info,
                            "function_id": record.get("function_id"),
                            "source_file": source_file,
                        },
                    )
                )
        return items

    @staticmethod
    def _load_from_markdown(markdown_path: str | Path) -> list[PostGISFunction]:
        path = Path(markdown_path)
        if not path.is_file():
            raise FileNotFoundError(f"ST_Function markdown not found: {path}")
        items: dict[str, set[str]] = {}
        current_dataset = "unknown"
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("## "):
                current_dataset = line[3:].strip()
                continue
            if not line.lower().startswith("st_"):
                continue
            items.setdefault(line, set()).add(current_dataset)

        functions: list[PostGISFunction] = []
        for function_name, datasets in sorted(items.items(), key=lambda item: item[0].lower()):
            categories = infer_function_categories(function_name)
            functions.append(
                PostGISFunction(
                    function_name=function_name,
                    signature=f"{function_name}(...)",
                    input_args=[],
                    return_type="",
                    description="Function referenced in ST_Function.md.",
                    example_usages=[],
                    categories=categories,
                    compatible_difficulties=infer_compatible_difficulties(categories),
                    source=["ST_Function.md"],
                    metadata={"datasets": sorted(datasets)},
                )
            )
        return functions

    def sample_functions(
        self,
        database: SynthesizedSpatialDatabase,
        difficulty_level: str,
        rng: np.random.Generator,
    ) -> list[PostGISFunction]:
        if difficulty_level not in DIFFICULTY_LEVELS:
            raise ValueError(f"Unsupported difficulty level: {difficulty_level}")

        spatial_table_count = sum(1 for table in database.selected_tables if table.spatial_fields)
        if spatial_table_count <= 0:
            return []

        preferred_categories = {
            "easy": ["spatial_predicate", "spatial_measurement", "geometry_transformation", "geometry_accessor", "bbox_operation"],
            "medium": ["spatial_predicate", "spatial_join", "spatial_measurement", "geometry_transformation", "geometry_accessor"],
            "hard": ["spatial_predicate", "spatial_join", "spatial_measurement", "geometry_transformation", "geometry_constructor", "spatial_aggregation", "coordinate_operation"],
            "extra-hard": ["spatial_predicate", "spatial_join", "spatial_measurement", "geometry_transformation", "geometry_constructor", "spatial_aggregation", "coordinate_operation", "linear_referencing"],
        }[difficulty_level]

        candidates = [
            item
            for item in self.functions
            if difficulty_level in item.compatible_difficulties
            and set(item.categories) & set(preferred_categories)
            and self._function_is_schema_compatible(item, spatial_table_count, len(database.selected_tables))
        ]
        if not candidates:
            candidates = [
                item
                for item in self.functions
                if difficulty_level in item.compatible_difficulties
                and self._function_is_schema_compatible(item, spatial_table_count, len(database.selected_tables))
            ]
        if not candidates:
            return []

        desired_count = {
            "easy": 1,
            "medium": 2,
            "hard": min(3, max(2, len(database.selected_tables))),
            "extra-hard": min(4, max(2, len(database.selected_tables))),
        }[difficulty_level]
        desired_count = min(desired_count, len(candidates))
        preferred_candidates = [item for item in candidates if self._prefer_st_function_source(item)]
        fallback_candidates = [item for item in candidates if not self._prefer_st_function_source(item)]
        selected: list[PostGISFunction] = []
        remaining = desired_count
        for pool in (preferred_candidates, fallback_candidates):
            if remaining <= 0 or not pool:
                continue
            sampled = self._weighted_sample_without_replacement(pool, remaining, rng)
            selected.extend(sampled)
            remaining = desired_count - len(selected)
        return selected

    @staticmethod
    def _sampling_weight(item: PostGISFunction) -> float:
        score = 1.0
        score += min(len(item.description), 400) / 400.0
        score += min(len(item.example_usages), 2) * 0.5
        if PostGISFunctionLibrary._prefer_st_function_source(item):
            score += 2.5
        if "spatial_predicate" in item.categories:
            score += 0.4
        if "spatial_join" in item.categories:
            score += 0.4
        return score

    @staticmethod
    def _prefer_st_function_source(item: PostGISFunction) -> bool:
        return any(to_text(source).strip() == "ST_Function.md" for source in item.source)

    @classmethod
    def _weighted_sample_without_replacement(
        cls,
        candidates: Sequence[PostGISFunction],
        desired_count: int,
        rng: np.random.Generator,
    ) -> list[PostGISFunction]:
        sample_size = min(int(desired_count), len(candidates))
        if sample_size <= 0:
            return []
        weights = np.array([cls._sampling_weight(item) for item in candidates], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(candidates), dtype=float)
        probs = weights / weights.sum()
        indices = rng.choice(len(candidates), size=sample_size, replace=False, p=probs)
        return [candidates[int(index)] for index in indices]

    @staticmethod
    def _function_is_schema_compatible(
        item: PostGISFunction,
        spatial_table_count: int,
        total_table_count: int,
    ) -> bool:
        categories = set(item.categories)
        if "spatial_join" in categories and spatial_table_count < 2:
            return False
        if item.function_name.lower() == "st_union" and total_table_count < 1:
            return False
        return spatial_table_count >= 1

    def get_function_signatures(self, function_name: str) -> list[PostGISFunction]:
        return list(self.by_name.get(function_name.lower(), []))
