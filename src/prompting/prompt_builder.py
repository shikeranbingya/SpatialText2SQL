"""Prompt builder backed by a standalone template file."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .prompt_enhancements.registry import PromptEnhancementRegistry
from .schema_compactor import DEFAULT_PROJECT_ROOT, SchemaCompactor
from .sample_data_provider import PostgresSampleDataProvider


class PromptBuilder:
    """Build prompts from a shared text template plus structured context."""

    def __init__(self, config: Dict):
        self.config = config
        project_root = config.get("project_root")
        self.project_root = Path(project_root).resolve() if project_root else DEFAULT_PROJECT_ROOT
        template_path = config.get("prompt_template_path")
        self.template_path = (
            Path(template_path).resolve()
            if template_path
            else self.project_root / "prompts" / "text2sql_prompt.txt"
        )
        self.ablation_configs = config.get("ablation_configs", {})
        self.prompt_styles = config.get("prompt_styles", {})
        self._template_cache: Dict[str, str] = {}
        self.schema_compactor = SchemaCompactor(project_root=self.project_root)
        self.sample_data_provider = config.get("sample_data_provider") or PostgresSampleDataProvider(
            project_root=self.project_root,
            db_config_path=config.get("db_config_path"),
        )
        self.prompt_enhancement_registry = (
            config.get("prompt_enhancement_registry")
            or PromptEnhancementRegistry(self.project_root)
        )
        self.named_prompt_templates = {
            "sql_synthesis": self.project_root / "prompts" / "sql_synthesis_prompt.txt",
            "sql_feedback": self.project_root / "prompts" / "sql_feedback_prompt.txt",
            "question_generation": self.project_root / "prompts" / "question_generation_prompt.txt",
            "question_feedback": self.project_root / "prompts" / "question_feedback_prompt.txt",
            "quality_control": self.project_root / "prompts" / "quality_control_prompt.txt",
        }
    
    def build_prompt(
        self,
        question: str,
        schema: str,
        config_type: str = 'base',
        rag_context: Optional[str] = None,
        keyword_context: Optional[str] = None,
        dataset_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        metadata = metadata or {}
        config_spec = self._resolve_ablation_config(config_type)
        prompt_style = str(config_spec.get("prompt_style") or "default")
        style_spec = self._resolve_prompt_style(prompt_style, dataset_name)
        template_text = self._load_template_text(prompt_style, style_spec)
        compact_schema = self.schema_compactor.compact_schema(
            schema=schema,
            question=question,
            dataset_name=dataset_name,
            metadata=metadata,
        )
        sample_data_block = ""
        if style_spec.get("include_sample_data", True):
            sample_data_block = self.sample_data_provider.build_sample_data(
                dataset_name=dataset_name or "",
                metadata=metadata,
                compact_schema=compact_schema,
            )
        grounding_block = self._build_grounding_block(
            dataset_name=dataset_name,
            metadata=metadata,
            style_spec=style_spec,
        )
        schema_semantics_block = self._build_schema_semantics_block(
            dataset_name=dataset_name,
            metadata=metadata,
            style_spec=style_spec,
            compact_schema=compact_schema,
        )
        placeholders = {
            "schema_block": compact_schema.strip(),
            "schema_semantics_block": schema_semantics_block.strip(),
            "sample_data_block": sample_data_block.strip(),
            "content_information_block": sample_data_block.strip(),
            "rag_block": self._stringify_value(
                rag_context if config_spec.get("use_rag") else None,
            ),
            "keyword_block": self._stringify_value(
                keyword_context if config_spec.get("use_keyword") else None,
            ),
            "grounding_block": grounding_block.strip(),
            "question_block": (question or "").strip(),
        }
        return self._render_template(template_text, placeholders)

    def build_sql_synthesis_prompt(
        self,
        *,
        database: Any,
        difficulty_level: str,
        structural_constraints: Dict[str, Any],
        sampled_functions: List[Dict[str, Any]],
        database_runtime_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        schema_lines, spatial_lines, representative_values = self._build_sql_prompt_context(
            database,
            database_runtime_metadata=database_runtime_metadata,
        )

        functions_payload = []
        for item in sampled_functions:
            functions_payload.append(
                {
                    "function_name": item.get("function_name"),
                    "signature": item.get("signature"),
                    "categories": item.get("categories"),
                    "description": item.get("description"),
                    "example_usages": item.get("example_usages"),
                }
            )

        template_text = self._load_named_template_text("sql_synthesis")
        return self._render_template(
            template_text,
            {
                "database_id": self._stringify_value(getattr(database, "database_id", "")),
                "city": self._stringify_value(getattr(database, "city", "")),
                "selected_tables": ", ".join(getattr(database, "selected_table_names", []) or []),
                "schema_block": chr(10).join(schema_lines) if schema_lines else "No schema available.",
                "spatial_field_block": chr(10).join(spatial_lines) if spatial_lines else "No spatial fields listed.",
                "representative_values_block": self._stable_json_text(representative_values),
                "difficulty_level": self._stringify_value(difficulty_level),
                "difficulty_constraint_block": self._stable_json_text(structural_constraints),
                "required_function_block": self._stable_json_text(functions_payload),
            },
        )

    @staticmethod
    def _detect_geometry_columns(table: Any) -> set[str]:
        names: set[str] = set()
        for column in getattr(table, "normalized_schema", []):
            if not isinstance(column, dict):
                continue
            canonical_type = str(column.get("canonical_type") or column.get("type") or "").strip().lower()
            if canonical_type != "spatial":
                continue
            for key in ("canonical_name", "name"):
                value = str(column.get(key) or "").strip().lower()
                if value:
                    names.add(value)
        for field in getattr(table, "spatial_fields", []):
            if not isinstance(field, dict):
                continue
            value = str(field.get("canonical_name") or "").strip().lower()
            if value:
                names.add(value)
        return names

    def _prepare_representative_rows(
        self,
        representative_values: Any,
        *,
        geometry_columns: set[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        normalized = representative_values
        if isinstance(normalized, dict) and isinstance(normalized.get("rows"), list):
            normalized = normalized.get("rows")

        rows: List[Dict[str, Any]] = []
        if isinstance(normalized, list):
            for item in normalized:
                if isinstance(item, dict):
                    rows.append(dict(item))
                else:
                    rows.append({"value": item})
        elif isinstance(normalized, dict):
            if self._looks_column_oriented_samples(normalized):
                rows = self._transpose_column_oriented_samples(normalized)
            else:
                rows = [dict(normalized)]

        prepared: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            normalized_row: Dict[str, Any] = {}
            for key, raw_value in row.items():
                column_name = str(key).strip()
                if not column_name:
                    continue
                normalized_row[column_name] = self._normalize_representative_value(
                    column_name,
                    raw_value,
                    geometry_columns=geometry_columns,
                )
            signature = self._stable_json_text(normalized_row)
            if signature in seen:
                continue
            seen.add(signature)
            prepared.append(normalized_row)
            if len(prepared) >= limit:
                break
        return prepared

    @staticmethod
    def _looks_column_oriented_samples(values: Dict[str, Any]) -> bool:
        if not values:
            return False
        if any(isinstance(item, dict) for item in values.values()):
            return False
        return any(isinstance(item, (list, tuple)) for item in values.values())

    @staticmethod
    def _transpose_column_oriented_samples(values: Dict[str, Any]) -> List[Dict[str, Any]]:
        normalized_columns: Dict[str, List[Any]] = {}
        max_len = 0
        for key, raw_value in values.items():
            column_name = str(key).strip()
            if not column_name:
                continue
            if isinstance(raw_value, (list, tuple)):
                items = list(raw_value)
            else:
                items = [raw_value]
            normalized_columns[column_name] = items
            max_len = max(max_len, len(items))

        rows: List[Dict[str, Any]] = []
        for index in range(max_len):
            row: Dict[str, Any] = {}
            for column_name, items in normalized_columns.items():
                row[column_name] = items[index] if index < len(items) else None
            rows.append(row)
        return rows

    def _normalize_representative_value(
        self,
        column_name: str,
        value: Any,
        *,
        geometry_columns: set[str],
    ) -> Any:
        if column_name.lower() in geometry_columns:
            if value in (None, ""):
                return None
            return self._geometry_preview(value)
        return value

    @staticmethod
    def _geometry_preview(value: Any) -> str:
        if isinstance(value, dict):
            geometry_type = str(value.get("type") or "").strip()
            return geometry_type.upper() if geometry_type else "GEOMETRY"
        text = str(value or "").strip()
        if not text:
            return "GEOMETRY"
        upper = text.upper()
        if upper.startswith("SRID=") and ";" in text:
            text = text.split(";", 1)[1].strip()
        match = re.match(r"^([A-Za-z]+)", text)
        if match:
            return match.group(1).upper()
        return "GEOMETRY"

    def build_sql_feedback_prompt(
        self,
        *,
        database: Any,
        difficulty_level: str,
        structural_constraints: Dict[str, Any],
        sampled_functions: List[Dict[str, Any]],
        original_candidate: Dict[str, Any],
        validation_errors: List[str],
        execution_error: str = "",
        empty_result: bool = False,
        database_runtime_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        schema_lines, spatial_lines, representative_values = self._build_sql_prompt_context(
            database,
            database_runtime_metadata=database_runtime_metadata,
        )

        template_text = self._load_named_template_text("sql_feedback")
        return self._render_template(
            template_text,
            {
                "original_candidate_block": self._stable_json_text(original_candidate),
                "database_id": self._stringify_value(getattr(database, "database_id", "")),
                "city": self._stringify_value(getattr(database, "city", "")),
                "schema_block": chr(10).join(schema_lines) if schema_lines else "No schema available.",
                "spatial_field_block": chr(10).join(spatial_lines) if spatial_lines else "No spatial fields listed.",
                "representative_values_block": self._stable_json_text(representative_values),
                "difficulty_level": self._stringify_value(difficulty_level),
                "difficulty_constraint_block": self._stable_json_text(structural_constraints),
                "required_function_block": self._stable_json_text(sampled_functions),
                "validation_errors_block": self._stable_json_text(validation_errors),
                "execution_error": execution_error or "none",
                "empty_result": str(bool(empty_result)).lower(),
            },
        )

    def _build_sql_prompt_context(
        self,
        database: Any,
        *,
        database_runtime_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        if isinstance(database_runtime_metadata, dict) and database_runtime_metadata.get("tables"):
            schema_lines: List[str] = []
            representative_values: Dict[str, Any] = {}
            spatial_lines: List[str] = []
            for table_meta in database_runtime_metadata.get("tables", []):
                if not isinstance(table_meta, dict):
                    continue
                table_name = str(table_meta.get("table_name") or "").strip()
                columns = []
                for column in table_meta.get("columns", []):
                    if not isinstance(column, dict):
                        continue
                    column_name = str(column.get("column_name") or "").strip()
                    column_type = str(column.get("column_type") or column.get("data_type") or "text").strip()
                    if column_name:
                        columns.append(f"{column_name} {column_type}")
                if table_name:
                    schema_lines.append(f"- {table_name}({', '.join(columns)})")
                table_rep_values = table_meta.get("representative_values") or {}
                geometry_columns = {
                    str(field.get("column_name") or field.get("canonical_name") or "").strip().lower()
                    for field in table_meta.get("spatial_fields", [])
                    if isinstance(field, dict)
                }
                if table_name and table_rep_values:
                    representative_values[table_name] = self._prepare_representative_rows(
                        table_rep_values,
                        geometry_columns=geometry_columns,
                        limit=3,
                    )
                for field in table_meta.get("spatial_fields", []):
                    if not isinstance(field, dict):
                        continue
                    spatial_name = str(field.get("column_name") or field.get("canonical_name") or "").strip()
                    column_type = str(field.get("column_type") or "").strip()
                    spatial_type = str(field.get("spatial_type") or "").strip() or "spatial"
                    geometry_type = str(field.get("geometry_type") or "").strip() or "GEOMETRY"
                    srid = field.get("srid")
                    if spatial_name and table_name:
                        spatial_lines.append(
                            f"- {table_name}.{spatial_name} "
                            f"(type={column_type or spatial_type}, family={spatial_type}, geometry_type={geometry_type}, srid={srid if srid not in (None, '') else 'unknown'})"
                        )
            return schema_lines, spatial_lines, representative_values

        schema_lines = []
        representative_values: Dict[str, Any] = {}
        spatial_lines: List[str] = []
        for table in getattr(database, "selected_tables", []):
            table_name = str(getattr(table, "table_name", "")).strip()
            geometry_columns = self._detect_geometry_columns(table)
            columns = []
            for column in getattr(table, "normalized_schema", []):
                if not isinstance(column, dict):
                    continue
                column_name = str(column.get("canonical_name") or column.get("name") or "").strip()
                column_type = str(column.get("canonical_type") or column.get("type") or "text").strip()
                if column_name:
                    columns.append(f"{column_name} {column_type}")
            schema_lines.append(f"- {table_name}({', '.join(columns)})")
            table_rep_values = getattr(table, "representative_values", None) or {}
            if table_rep_values:
                representative_values[table_name] = self._prepare_representative_rows(
                    table_rep_values,
                    geometry_columns=geometry_columns,
                    limit=3,
                )
            for field in getattr(table, "spatial_fields", []):
                if not isinstance(field, dict):
                    continue
                spatial_name = str(field.get("canonical_name") or "").strip()
                crs = str(field.get("crs") or "null").strip()
                if spatial_name:
                    spatial_lines.append(f"- {table_name}.{spatial_name} (crs={crs})")
        return schema_lines, spatial_lines, representative_values

    def build_question_generation_prompt(
        self,
        *,
        sql_query: Any,
        database_context: Dict[str, Any],
        sql_features: Dict[str, Any],
        style_constraint: Dict[str, Any],
        spatial_relation_constraints: List[Dict[str, Any]],
    ) -> str:
        schema_lines = self._build_question_schema_lines(database_context)
        representative_values = self._build_question_representative_values(database_context)
        spatial_lines = self._build_question_spatial_lines(database_context)
        template_text = self._load_named_template_text("question_generation")
        return self._render_template(
            template_text,
            {
                "sql_query": self._stringify_value(getattr(sql_query, "sql", "")),
                "database_id": self._stringify_value(database_context.get("database_id")),
                "city": self._stringify_value(database_context.get("city")),
                "selected_tables": ", ".join(database_context.get("selected_table_names", []) or []),
                "schema_block": chr(10).join(schema_lines) if schema_lines else "No schema available.",
                "spatial_field_block": chr(10).join(spatial_lines) if spatial_lines else "No spatial fields listed.",
                "representative_values_block": self._stable_json_text(representative_values),
                "sql_feature_block": self._stable_json_text(sql_features),
                "style_constraint_block": self._stable_json_text(style_constraint),
                "spatial_relation_block": self._stable_json_text(spatial_relation_constraints),
                "style_name": self._stringify_value(style_constraint.get("style")),
            },
        )

    def build_question_feedback_prompt(
        self,
        *,
        sql_query: Any,
        database_context: Dict[str, Any],
        sql_features: Dict[str, Any],
        style_constraint: Dict[str, Any],
        spatial_relation_constraints: List[Dict[str, Any]],
        original_candidate: Dict[str, Any],
        validation_errors: List[str],
    ) -> str:
        schema_lines = self._build_question_schema_lines(database_context)
        representative_values = self._build_question_representative_values(database_context)
        spatial_lines = self._build_question_spatial_lines(database_context)
        template_text = self._load_named_template_text("question_feedback")
        return self._render_template(
            template_text,
            {
                "sql_query": self._stringify_value(getattr(sql_query, "sql", "")),
                "original_candidate_block": self._stable_json_text(original_candidate),
                "validation_errors_block": self._stable_json_text(validation_errors),
                "database_id": self._stringify_value(database_context.get("database_id")),
                "city": self._stringify_value(database_context.get("city")),
                "selected_tables": ", ".join(database_context.get("selected_table_names", []) or []),
                "schema_block": chr(10).join(schema_lines) if schema_lines else "No schema available.",
                "spatial_field_block": chr(10).join(spatial_lines) if spatial_lines else "No spatial fields listed.",
                "representative_values_block": self._stable_json_text(representative_values),
                "sql_feature_block": self._stable_json_text(sql_features),
                "style_constraint_block": self._stable_json_text(style_constraint),
                "spatial_relation_block": self._stable_json_text(spatial_relation_constraints),
                "style_name": self._stringify_value(style_constraint.get("style")),
            },
        )

    def build_quality_control_prompt(
        self,
        *,
        sample: Dict[str, Any],
        schema_lines: List[str],
        sql_feature_summary: Dict[str, Any],
        execution_summary: Dict[str, Any],
        representative_values: Dict[str, Any],
        judge_rules: Dict[str, Any],
    ) -> str:
        template_text = self._load_named_template_text("quality_control")
        return self._render_template(
            template_text,
            {
                "sample_block": self._stable_json_text(sample),
                "schema_block": chr(10).join(schema_lines) if schema_lines else "No schema available.",
                "sql_feature_block": self._stable_json_text(sql_feature_summary),
                "execution_summary_block": self._stable_json_text(execution_summary),
                "representative_values_block": self._stable_json_text(representative_values),
                "judge_rules_block": self._stable_json_text(judge_rules),
            },
        )

    @staticmethod
    def _build_question_schema_lines(database_context: Dict[str, Any]) -> List[str]:
        schema_lines: List[str] = []
        for table_item in database_context.get("schema", []) or []:
            if not isinstance(table_item, dict):
                continue
            table_name = str(table_item.get("table_name") or "").strip()
            columns = []
            for column in table_item.get("normalized_schema", []) or []:
                if not isinstance(column, dict):
                    continue
                column_name = str(column.get("canonical_name") or column.get("name") or "").strip()
                column_type = str(column.get("canonical_type") or column.get("type") or "text").strip()
                if column_name:
                    columns.append(f"{column_name} {column_type}")
            if table_name:
                schema_lines.append(f"- {table_name}({', '.join(columns)})")
        return schema_lines

    def _build_question_representative_values(self, database_context: Dict[str, Any]) -> Dict[str, Any]:
        representative_values: Dict[str, Any] = {}
        geometry_columns_by_table: Dict[str, set[str]] = {}

        for table_item in database_context.get("table_contexts", []) or []:
            if not isinstance(table_item, dict):
                continue
            table_name = str(table_item.get("table_name") or "").strip()
            if not table_name:
                continue
            geometry_columns_by_table[table_name] = {
                str(field.get("canonical_name") or field.get("column_name") or "").strip().lower()
                for field in table_item.get("spatial_fields", []) or []
                if isinstance(field, dict)
            }
            table_values = table_item.get("representative_values")
            if table_values:
                representative_values[table_name] = self._prepare_representative_rows(
                    table_values,
                    geometry_columns=geometry_columns_by_table[table_name],
                    limit=3,
                )

        if representative_values:
            return representative_values

        values = database_context.get("representative_values")
        if isinstance(values, dict):
            normalized: Dict[str, Any] = {}
            for table_name, table_values in values.items():
                geometry_columns = geometry_columns_by_table.get(str(table_name).strip(), set())
                normalized[str(table_name)] = self._prepare_representative_rows(
                    table_values,
                    geometry_columns=geometry_columns,
                    limit=3,
                )
            return normalized
        return {}

    @staticmethod
    def _build_question_spatial_lines(database_context: Dict[str, Any]) -> List[str]:
        spatial_lines: List[str] = []
        for field in database_context.get("spatial_fields", []) or []:
            if not isinstance(field, dict):
                continue
            table_name = str(field.get("table_name") or "").strip()
            column_name = str(
                field.get("column_name")
                or field.get("canonical_name")
                or field.get("field_name")
                or ""
            ).strip()
            column_type = str(field.get("column_type") or field.get("family") or "spatial").strip()
            spatial_type = str(field.get("spatial_type") or "").strip()
            geometry_type = str(field.get("geometry_type") or "").strip()
            srid = field.get("srid")
            if table_name and column_name:
                details = []
                if column_type:
                    details.append(f"type={column_type}")
                if spatial_type:
                    details.append(f"family={spatial_type}")
                if geometry_type:
                    details.append(f"geometry_type={geometry_type}")
                if srid not in (None, ""):
                    details.append(f"srid={srid}")
                spatial_lines.append(f"- {table_name}.{column_name} ({', '.join(details)})")
        return spatial_lines

    @staticmethod
    def _stringify_value(value: Any) -> str:
        return str(value).strip() if value not in (None, "") else ""

    @staticmethod
    def _stable_json_text(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)

    def _render_template(self, template_text: str, placeholders: Dict[str, str]) -> str:
        rendered = template_text
        for key, value in placeholders.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", value.strip())
        rendered = re.sub(r"\{\{[^{}]+\}\}", "", rendered)
        return self._cleanup_rendered_prompt(rendered)

    def _resolve_ablation_config(self, config_type: str) -> Dict[str, Any]:
        config = self.ablation_configs.get(config_type)
        if config is not None:
            return config
        return {
            "use_rag": config_type in ["rag", "full"],
            "use_keyword": config_type in ["keyword", "full"],
            "prompt_style": "default",
        }

    def _resolve_prompt_style(
        self,
        prompt_style: str,
        dataset_name: Optional[str],
    ) -> Dict[str, Any]:
        default_style = self.prompt_styles.get("default", {})
        style_config = self.prompt_styles.get(prompt_style, {})
        merged = dict(default_style)
        merged.update(style_config)
        if merged.get("dataset_specific") and dataset_name:
            merged.update(
                self.prompt_enhancement_registry.resolve_dataset_override(dataset_name)
            )
        return merged

    def _load_template_text(self, prompt_style: str, style_spec: Dict[str, Any]) -> str:
        template_path = style_spec.get("template_path")
        if template_path:
            path = Path(template_path)
            if not path.is_absolute():
                path = (self.project_root / template_path).resolve()
        else:
            path = self.template_path

        cache_key = f"{prompt_style}:{path}"
        if cache_key not in self._template_cache:
            self._template_cache[cache_key] = path.read_text(encoding="utf-8")
        return self._template_cache[cache_key]

    def _load_named_template_text(self, template_name: str) -> str:
        path = self.named_prompt_templates[template_name]
        cache_key = f"named:{template_name}:{path}"
        if cache_key not in self._template_cache:
            self._template_cache[cache_key] = path.read_text(encoding="utf-8")
        return self._template_cache[cache_key]

    def _build_grounding_block(
        self,
        dataset_name: Optional[str],
        metadata: Dict[str, Any],
        style_spec: Dict[str, Any],
    ) -> str:
        if not style_spec.get("use_dataset_context"):
            return ""
        return self.prompt_enhancement_registry.build_grounding_block(
            dataset_name or "",
            metadata,
        )

    def _build_schema_semantics_block(
        self,
        dataset_name: Optional[str],
        metadata: Dict[str, Any],
        style_spec: Dict[str, Any],
        compact_schema: str,
    ) -> str:
        if not style_spec.get("use_dataset_context"):
            return ""
        build_fn = getattr(
            self.prompt_enhancement_registry,
            "build_schema_semantics_block",
            None,
        )
        if build_fn is None:
            return ""
        return build_fn(
            dataset_name or "",
            metadata,
            compact_schema,
        )

    @staticmethod
    def _cleanup_rendered_prompt(rendered: str) -> str:
        preamble, sections = PromptBuilder._split_prompt_sections(rendered)
        cleaned_sections: List[Tuple[str, List[str]]] = []

        for header, body_lines in sections:
            cleaned_body = [
                line.rstrip()
                for line in body_lines
                if not PromptBuilder._is_empty_metadata_line(line)
            ]
            cleaned_body = PromptBuilder._trim_blank_lines(cleaned_body)
            if cleaned_body:
                cleaned_sections.append((header, cleaned_body))

        prompt_lines = PromptBuilder._trim_blank_lines([line.rstrip() for line in preamble])
        for header, body in cleaned_sections:
            if prompt_lines:
                prompt_lines.append("")
            prompt_lines.append(header)
            prompt_lines.extend(body)

        return "\n".join(PromptBuilder._trim_blank_lines(prompt_lines))

    @staticmethod
    def _split_prompt_sections(rendered: str) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
        preamble: List[str] = []
        sections: List[Tuple[str, List[str]]] = []
        current_header: Optional[str] = None
        current_body: List[str] = []

        for line in rendered.splitlines():
            if line.startswith("## "):
                if current_header is None:
                    pass
                else:
                    sections.append((current_header, current_body))
                current_header = line.rstrip()
                current_body = []
                continue

            if current_header is None:
                preamble.append(line.rstrip())
            else:
                current_body.append(line.rstrip())

        if current_header is not None:
            sections.append((current_header, current_body))

        return preamble, sections

    @staticmethod
    def _is_empty_metadata_line(line: str) -> bool:
        return bool(re.match(r"^\s*-\s+[A-Za-z0-9_ ]+:\s*$", line))

    @staticmethod
    def _trim_blank_lines(lines: List[str]) -> List[str]:
        start = 0
        end = len(lines)
        while start < end and not lines[start].strip():
            start += 1
        while end > start and not lines[end - 1].strip():
            end -= 1
        trimmed = lines[start:end]
        compacted: List[str] = []
        previous_blank = False
        for line in trimmed:
            is_blank = not line.strip()
            if is_blank and previous_blank:
                continue
            compacted.append(line)
            previous_blank = is_blank
        return compacted
    
    def build_batch_prompts(
        self,
        questions: list,
        schema: str,
        config_type: str = 'base',
        rag_contexts: Optional[list] = None,
        keyword_contexts: Optional[list] = None,
        dataset_name: Optional[str] = None,
        metadatas: Optional[list] = None,
    ) -> list:
        """
        批量构建prompts
        
        Args:
            questions: 问题列表
            schema: 数据库Schema
            config_type: 配置类型
            rag_contexts: RAG context列表（可选）
            keyword_contexts: Keyword context列表（可选）
            
        Returns:
            prompt列表
        """
        prompts = []
        
        for i, question in enumerate(questions):
            rag_ctx = rag_contexts[i] if rag_contexts and i < len(rag_contexts) else None
            kw_ctx = keyword_contexts[i] if keyword_contexts and i < len(keyword_contexts) else None
            metadata = metadatas[i] if metadatas and i < len(metadatas) else None
            
            prompt = self.build_prompt(
                question=question,
                schema=schema,
                config_type=config_type,
                rag_context=rag_ctx,
                keyword_context=kw_ctx,
                dataset_name=dataset_name,
                metadata=metadata,
            )
            prompts.append(prompt)
        
        return prompts
    
    @staticmethod
    def get_config_description(config_type: str) -> str:
        """
        获取配置类型的描述
        
        Args:
            config_type: 配置类型
            
        Returns:
            配置描述字符串
        """
        descriptions = {
            'base': 'Question + Schema',
            'rag': 'Question + Schema + Retrieved Context',
            'keyword': 'Question + Schema + Keyword Context',
            'full': 'Question + Schema + Retrieved Context + Keyword Context'
        }
        return descriptions.get(config_type, 'Unknown')
