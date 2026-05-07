import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.prompting.prompt_builder import PromptBuilder
from src.synthesis.database.models import CanonicalSpatialTable, SynthesizedSpatialDatabase
from src.synthesis.question import (
    DiversityAwareQuestionSynthesizer,
    MockQuestionLLM,
    OllamaQuestionLLM,
    QuestionGenerationConfig,
    QuestionGenerationContext,
    QuestionGenerationLLMConfig,
    QuestionGenerationLoggingConfig,
    QuestionGenerationRunConfig,
    QuestionValidator,
    SQLFeatureExtractor,
    SQLQuestionSource,
    SpatialPhraseSelector,
    StyleSelector,
    append_synthesized_question,
    build_question_generation_contexts_from_sql_sources,
    build_question_llm,
    ensure_question_output,
    load_existing_question_id_offsets,
    load_question_generation_config,
    load_question_generation_contexts,
    load_sql_question_sources,
    override_question_generation_config,
    parse_question_generation_response,
    write_synthesized_questions,
)
from src.synthesis.question.models import QuestionGenerationCandidate


def _make_table(table_id: str, table_name: str, *, city: str = "nyc") -> CanonicalSpatialTable:
    return CanonicalSpatialTable.from_dict(
        {
            "table_id": table_id,
            "city": city,
            "table_name": table_name,
            "semantic_summary": f"{table_name} summary",
            "normalized_schema": [
                {"name": "id", "canonical_name": "id", "canonical_type": "integer"},
                {"name": "name", "canonical_name": "name", "canonical_type": "text"},
                {"name": "geom", "canonical_name": "geom", "canonical_type": "spatial"},
            ],
            "representative_values": {"name": [f"{table_name}_sample"]},
            "themes": ["parks"],
            "spatial_fields": [{"canonical_name": "geom", "crs": "EPSG:4326"}],
            "path": f"/tmp/{table_name}.geojson",
        }
    )


def _make_database() -> SynthesizedSpatialDatabase:
    tables = [_make_table("t1", "parks"), _make_table("t2", "neighborhoods")]
    return SynthesizedSpatialDatabase.from_selected_tables(
        database_id="nyc_0001",
        city="nyc",
        selected_tables=tables,
        sampling_trace=[],
        graph_stats={},
        synthesize_config={},
    )


def _make_sql_source(sql: str) -> SQLQuestionSource:
    return SQLQuestionSource.from_dict(
        {
            "sql_id": "sql_001",
            "database_id": "nyc_0001",
            "city": "nyc",
            "difficulty_level": "medium",
            "sql": sql,
            "reasoning_summary": "Join parks to neighborhoods with the distance filter preserved.",
            "used_tables": ["parks", "neighborhoods"],
            "used_columns": ["name", "geom"],
            "used_spatial_functions": ["ST_DWithin"],
            "spatial_function_constraints": [
                {
                    "function_name": "ST_DWithin",
                    "signature": "ST_DWithin(geometry geom1, geometry geom2, double precision distance)",
                    "description": "True when two geometries are within a given distance.",
                    "example_usages": ["SELECT ST_DWithin(a.geom, b.geom, 100)"],
                }
            ],
            "metadata": {
                "database_context": {
                    "database_id": "nyc_0001",
                    "city": "nyc",
                    "selected_table_names": ["parks", "neighborhoods"],
                    "schema_ddls": [
                        "CREATE TABLE parks (\n    id integer,\n    name text,\n    geom geometry(GEOMETRY,4326)\n);",
                        "CREATE TABLE neighborhoods (\n    id integer,\n    name text,\n    geom geometry(GEOMETRY,4326)\n);",
                    ],
                    "tables": [
                        {
                            "table_name": "parks",
                            "create_table_ddl": "CREATE TABLE parks (\n    id integer,\n    name text,\n    geom geometry(GEOMETRY,4326)\n);",
                            "columns": [
                                {"column_name": "id", "column_type": "integer"},
                                {"column_name": "name", "column_type": "text"},
                                {"column_name": "geom", "column_type": "geometry(GEOMETRY,4326)"},
                            ],
                            "spatial_fields": [
                                {
                                    "column_name": "geom",
                                    "column_type": "geometry(GEOMETRY,4326)",
                                    "spatial_type": "geometry",
                                    "geometry_type": "GEOMETRY",
                                    "srid": 4326,
                                }
                            ],
                            "representative_values": {"name": ["central park"]},
                        },
                        {
                            "table_name": "neighborhoods",
                            "create_table_ddl": "CREATE TABLE neighborhoods (\n    id integer,\n    name text,\n    geom geometry(GEOMETRY,4326)\n);",
                            "columns": [
                                {"column_name": "id", "column_type": "integer"},
                                {"column_name": "name", "column_type": "text"},
                                {"column_name": "geom", "column_type": "geometry(GEOMETRY,4326)"},
                            ],
                            "spatial_fields": [
                                {
                                    "column_name": "geom",
                                    "column_type": "geometry(GEOMETRY,4326)",
                                    "spatial_type": "geometry",
                                    "geometry_type": "GEOMETRY",
                                    "srid": 4326,
                                }
                            ],
                            "representative_values": {"name": ["harlem"]},
                        },
                    ],
                }
            },
            "execution_result": {
                "status": "success",
                "row_count": 2,
                "columns": ["name"],
                "rows": [{"name": "central park"}, {"name": "prospect park"}],
            },
        }
    )


def _make_config(**generation_overrides) -> QuestionGenerationConfig:
    base = QuestionGenerationConfig(
        llm=QuestionGenerationLLMConfig(
            provider="mock",
            model="mock-model",
            base_url="http://mock",
            api_key_env="OPENAI_API_KEY",
        ),
        generation=QuestionGenerationRunConfig(
            num_questions_per_sql=1,
            fixed_style="direct",
        ),
        logging=QuestionGenerationLoggingConfig(log_level="INFO"),
    )
    return override_question_generation_config(base, generation=generation_overrides or None)


def _make_context(database: SynthesizedSpatialDatabase) -> QuestionGenerationContext:
    return QuestionGenerationContext.from_database(database)


class QuestionGenerationTests(unittest.TestCase):
    def test_config_loading_and_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "question_synthesis.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "llm:",
                        "  provider: ollama",
                        "  model: qwen2.5:7b",
                        "generation:",
                        "  sql_input_path: data/in.jsonl",
                        "  output_path: data/out.jsonl",
                        "  num_questions_per_sql: 2",
                        "  style_weights:",
                        "    conversational: 2",
                        "    formal: 1",
                        "    direct: 0",
                        "    concise: 0",
                        "    polite: 0",
                        "    analytical: 0",
                    ]
                ),
                encoding="utf-8",
            )
            loaded = load_question_generation_config(config_path)
            overridden = override_question_generation_config(
                loaded,
                generation={"output_path": str(root / "override.jsonl"), "style": "formal"},
            )
        self.assertEqual(loaded.llm.provider, "ollama")
        self.assertEqual(loaded.llm.model, "qwen2.5:7b")
        self.assertEqual(loaded.generation.num_questions_per_sql, 2)
        self.assertTrue(overridden.generation.output_path.endswith("override.jsonl"))
        self.assertEqual(overridden.generation.fixed_style, "formal")

    def test_build_question_llm_supports_config_object(self):
        llm = build_question_llm(
            config=QuestionGenerationLLMConfig(
                provider="ollama",
                model="qwen2.5:7b",
                base_url="http://localhost:11434/v1",
                api_key_env="IGNORED_FOR_OLLAMA",
                temperature=0.1,
                max_tokens=256,
                timeout=30,
                max_retries=1,
            )
        )
        self.assertIsInstance(llm, OllamaQuestionLLM)

    def test_feature_extractor_parses_spatial_sql(self):
        sql = """
        SELECT n.name, COUNT(*) AS park_count
        FROM neighborhoods n
        JOIN parks p ON ST_DWithin(p.geom, n.geom, 100)
        WHERE ST_Contains(n.geom, p.geom)
        GROUP BY n.name
        ORDER BY park_count DESC
        LIMIT 5
        """
        features = SQLFeatureExtractor().extract(sql)
        self.assertIn("neighborhoods", features.tables)
        self.assertIn("parks", features.tables)
        self.assertIn("ST_DWITHIN", features.postgis_functions)
        self.assertIn("ST_CONTAINS", features.postgis_functions)
        self.assertIn("COUNT", features.aggregates)
        self.assertEqual(features.limit, 5)
        self.assertIn("100", features.distance_thresholds)
        self.assertIn("name", features.group_by_columns)

    def test_style_and_spatial_phrase_selection_are_deterministic(self):
        sql = "SELECT p.name FROM parks p WHERE ST_DWithin(p.geom, p.geom, 100) LIMIT 3"
        features = SQLFeatureExtractor().extract(sql)

        selector = StyleSelector()
        rng_one = np.random.default_rng(42)
        rng_two = np.random.default_rng(42)
        plan_one = selector.build_style_plan(features=features, total_questions=5, rng=rng_one)
        plan_two = selector.build_style_plan(features=features, total_questions=5, rng=rng_two)
        self.assertEqual(plan_one, plan_two)

        spatial_selector = SpatialPhraseSelector()
        constraints_one = spatial_selector.build_constraints(features=features, rng=np.random.default_rng(7))
        constraints_two = spatial_selector.build_constraints(features=features, rng=np.random.default_rng(7))
        self.assertEqual([item.to_dict() for item in constraints_one], [item.to_dict() for item in constraints_two])
        self.assertTrue(constraints_one)
        self.assertIn("100", constraints_one[0].preferred_phrase)

    def test_prompt_builder_contains_required_sections(self):
        sql = _make_sql_source("SELECT p.name FROM parks p WHERE ST_DWithin(p.geom, p.geom, 100)")
        context = build_question_generation_contexts_from_sql_sources([sql])["nyc_0001"]
        features = SQLFeatureExtractor().extract(sql.sql)
        constraints = SpatialPhraseSelector().build_constraints(features=features, rng=np.random.default_rng(11))
        prompt_builder = PromptBuilder({"project_root": Path(__file__).resolve().parents[2]})
        prompt = prompt_builder.build_question_generation_prompt(
            sql_query=sql,
            database_context=context.to_prompt_payload(),
            sql_features=features.to_dict(),
            style_constraint={"style": "direct", "description": "Use a direct and minimal request."},
            spatial_relation_constraints=[item.to_dict() for item in constraints],
        )
        self.assertIn("## SQL Query", prompt)
        self.assertIn("## Database Context", prompt)
        self.assertIn("## Execution Results", prompt)
        self.assertNotIn("## Representative Values", prompt)
        self.assertNotIn("## Spatial Field Metadata", prompt)
        self.assertNotIn("## SQL Feature Summary", prompt)
        self.assertIn("## Style Constraint", prompt)
        self.assertIn("## Spatial Relation Constraint", prompt)
        self.assertIn('"question"', prompt)
        self.assertIn('"style"', prompt)
        self.assertIn('"row_count": 2', prompt)
        self.assertIn('"name": "central park"', prompt)
        self.assertIn('"function_name": "ST_DWithin"', prompt)
        self.assertIn('"description": "True when two geometries are within a given distance."', prompt)
        self.assertIn("CREATE TABLE parks", prompt)
        self.assertIn("geometry(GEOMETRY,4326)", prompt)

    def test_build_question_generation_contexts_from_sql_metadata(self):
        sql = _make_sql_source("SELECT p.name FROM parks p WHERE ST_DWithin(p.geom, p.geom, 100)")
        contexts = build_question_generation_contexts_from_sql_sources([sql])
        context = contexts["nyc_0001"]
        self.assertEqual(context.selected_table_names, ["parks", "neighborhoods"])
        self.assertEqual(len(context.schema_ddls), 2)
        self.assertIn("CREATE TABLE parks", context.schema_ddls[0])

    def test_response_parser_handles_json_and_markdown(self):
        parsed = parse_question_generation_response(
            '{"question":"Which parks are within 100 units of the neighborhood boundary?","style":"direct","reasoning_summary":"Preserved the threshold.","spatial_phrases":["within 100 units of"]}'
        )
        self.assertEqual(parsed.question, "Which parks are within 100 units of the neighborhood boundary?")
        self.assertEqual(parsed.style, "direct")

        fenced = parse_question_generation_response(
            """```json
            {"question":"Please list how many parks are within 100 units of each neighborhood.","style":"polite","reasoning_summary":"Kept the aggregation.","spatial_phrases":["within 100 units of"]}
            ```"""
        )
        self.assertEqual(fenced.style, "polite")
        self.assertFalse(fenced.parse_error)

    def test_validator_rejects_raw_postgis_names_and_missing_threshold(self):
        sql = "SELECT p.name FROM parks p WHERE ST_DWithin(p.geom, p.geom, 100)"
        features = SQLFeatureExtractor().extract(sql)
        constraints = SpatialPhraseSelector().build_constraints(features=features, rng=np.random.default_rng(5))
        validator = QuestionValidator()

        raw_function = validator.validate(
            candidate=QuestionGenerationCandidate(question="Which parks use ST_DWithin within 100 units?", style="direct"),
            requested_style="direct",
            sql_features=features,
            spatial_constraints=constraints,
        )
        self.assertFalse(raw_function.is_valid)
        self.assertTrue(any("raw PostGIS function names" in item for item in raw_function.errors))

        missing_threshold = validator.validate(
            candidate=QuestionGenerationCandidate(question="Find parks near the neighborhood boundary.", style="direct"),
            requested_style="direct",
            sql_features=features,
            spatial_constraints=constraints,
        )
        self.assertFalse(missing_threshold.is_valid)
        self.assertTrue(any("100" in item for item in missing_threshold.errors))

    def test_end_to_end_generation_keeps_invalid_candidate(self):
        sql = _make_sql_source(
            "SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100) LIMIT 5"
        )
        database = _make_database()
        context = _make_context(database)
        llm = MockQuestionLLM(
            responses=[
                json.dumps(
                    {
                        "question": "Which parks are near neighborhoods?",
                        "style": "direct",
                        "reasoning_summary": "Initial attempt.",
                        "spatial_phrases": ["near"],
                    }
                ),
            ]
        )
        generator = DiversityAwareQuestionSynthesizer(
            config=_make_config(),
            llm_client=llm,
            prompt_builder=PromptBuilder({"project_root": Path(__file__).resolve().parents[2]}),
        )
        rows = generator.generate_for_sql(sql, context)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0].validation_result["is_valid"])
        self.assertEqual(rows[0].question_id, "nyc_0001_0001")
        self.assertEqual(len(llm.prompts), 1)

    def test_synthesizer_continues_question_id_from_existing_offsets(self):
        sql = _make_sql_source("SELECT p.name FROM parks p WHERE ST_DWithin(p.geom, p.geom, 100)")
        database = _make_database()
        context = _make_context(database)
        llm = MockQuestionLLM(
            responses=[
                json.dumps(
                    {
                        "question": "Find parks within 100 units of themselves.",
                        "style": "direct",
                        "reasoning_summary": "Preserved the threshold.",
                        "spatial_phrases": ["within 100 units of"],
                    }
                )
            ]
        )
        generator = DiversityAwareQuestionSynthesizer(
            config=_make_config(),
            llm_client=llm,
            prompt_builder=PromptBuilder({"project_root": Path(__file__).resolve().parents[2]}),
            existing_question_id_offsets={"nyc_0001": 7},
        )
        rows = generator.generate_for_sql(sql, context)
        self.assertEqual(rows[0].question_id, "nyc_0001_0008")

    def test_io_round_trip_and_append(self):
        sql = _make_sql_source("SELECT p.name FROM parks p WHERE ST_DWithin(p.geom, p.geom, 100)")
        database = _make_database()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sql_path = root / "sql.jsonl"
            db_path = root / "db.jsonl"
            out_path = root / "questions.jsonl"
            sql_path.write_text(json.dumps(sql.__dict__, ensure_ascii=False) + "\n", encoding="utf-8")
            db_path.write_text(json.dumps(database.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")

            loaded_sql = load_sql_question_sources(str(sql_path))
            loaded_contexts = load_question_generation_contexts(str(db_path))

            generator = DiversityAwareQuestionSynthesizer(
                config=_make_config(),
                llm_client=MockQuestionLLM(
                    responses=[
                        json.dumps(
                            {
                                "question": "Find parks within 100 units of themselves.",
                                "style": "direct",
                                "reasoning_summary": "Preserved the threshold.",
                                "spatial_phrases": ["within 100 units of"],
                            }
                        )
                    ]
                ),
                prompt_builder=PromptBuilder({"project_root": Path(__file__).resolve().parents[2]}),
            )
            ensure_question_output(str(out_path))
            rows = generator.generate_all(
                loaded_sql,
                loaded_contexts,
                on_row_generated=lambda row: append_synthesized_question(str(out_path), row),
            )
            offsets = load_existing_question_id_offsets(str(out_path))

            written = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(loaded_sql), 1)
        self.assertIn("nyc_0001", loaded_contexts)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["sql_id"], "sql_001")
        self.assertEqual(written[0]["question_id"], "nyc_0001_0001")
        self.assertEqual(
            written[0]["sql_reasoning_summary"],
            "Join parks to neighborhoods with the distance filter preserved.",
        )
        self.assertEqual(offsets, {"nyc_0001": 1})

    def test_write_synthesized_questions_overwrites_existing_file(self):
        sql = _make_sql_source("SELECT p.name FROM parks p WHERE ST_DWithin(p.geom, p.geom, 100)")
        database = _make_database()
        context = _make_context(database)
        generator = DiversityAwareQuestionSynthesizer(
            config=_make_config(),
            llm_client=MockQuestionLLM(
                responses=[
                    json.dumps(
                        {
                            "question": "Find parks within 100 units of themselves.",
                            "style": "direct",
                            "reasoning_summary": "Preserved the threshold.",
                            "spatial_phrases": ["within 100 units of"],
                        }
                    )
                ]
            ),
            prompt_builder=PromptBuilder({"project_root": Path(__file__).resolve().parents[2]}),
        )
        rows = generator.generate_for_sql(sql, context)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "questions.jsonl"
            path.write_text('{"question_id":"nyc_0001_9999"}\n', encoding="utf-8")
            write_synthesized_questions(str(path), rows)
            written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["question_id"], "nyc_0001_0001")


if __name__ == "__main__":
    unittest.main()
