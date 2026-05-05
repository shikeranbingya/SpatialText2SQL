import json
import tempfile
import unittest
from pathlib import Path

from src.prompting.prompt_builder import PromptBuilder
from src.synthesis.database.models import CanonicalSpatialTable, SynthesizedSpatialDatabase
from src.synthesis.sql import (
    ConstraintGuidedSQLSynthesizer,
    MockSQLGenerator,
    OllamaSQLGenerator,
    PostGISFunctionLibrary,
    PostGISPromptMetadataProvider,
    SQLExecutionChecker,
    SQLExecutionCheckConfig,
    SQLSynthesisConfig,
    SQLSynthesisDBConfig,
    SQLSynthesisFunctionConfig,
    SQLSynthesisLLMConfig,
    SQLSynthesisLoggingConfig,
    SQLSynthesisRunConfig,
    SQLValidator,
    build_sql_generator,
    contains_dangerous_sql,
    load_sql_synthesis_config,
    override_sql_synthesis_config,
    parse_sql_generation_response,
)


def _make_table(table_id: str, table_name: str, *, city: str = "nyc", spatial_name: str = "geom"):
    return CanonicalSpatialTable.from_dict(
        {
            "table_id": table_id,
            "city": city,
            "table_name": table_name,
            "semantic_summary": f"{table_name} summary",
            "normalized_schema": [
                {"name": "id", "canonical_name": "id", "canonical_type": "integer"},
                {"name": "name", "canonical_name": "name", "canonical_type": "text"},
                {"name": spatial_name, "canonical_name": spatial_name, "canonical_type": "spatial"},
            ],
            "representative_values": {"name": [f"{table_name}_sample"]},
            "themes": ["utilities"],
            "spatial_fields": [{"canonical_name": spatial_name, "crs": "EPSG:4326"}],
            "path": f"/tmp/{table_name}.geojson",
            "description": f"{table_name} description",
        }
    )


def _make_database(table_count: int = 2, *, database_id: str = "nyc_0001") -> SynthesizedSpatialDatabase:
    city = database_id.split("_", 1)[0]
    tables = [_make_table(f"t{i+1}", f"table_{i+1}", city=city) for i in range(table_count)]
    return SynthesizedSpatialDatabase.from_selected_tables(
        database_id=database_id,
        city=city,
        selected_tables=tables,
        sampling_trace=[],
        graph_stats={},
        synthesize_config={},
    )


def _sample_function_json_payload():
    return [
        {
            "function_id": "st_dwithin",
            "chapter_info": "reference_relationship",
            "source_file": "reference_relationship.xml",
            "function_definitions": [
                {
                    "function_name": "ST_DWithin",
                    "return_type": "boolean",
                    "arguments": ["geometry geom1", "geometry geom2", "double precision distance"],
                    "signature_str": "ST_DWithin(geometry geom1, geometry geom2, double precision distance)",
                }
            ],
            "description": "Returns true if two geometries are within a given distance.",
            "examples": [{"steps": [{"sql": "SELECT ST_DWithin(a.geom, b.geom, 100);"}]}],
        },
        {
            "function_id": "st_buffer",
            "chapter_info": "reference_processing",
            "source_file": "reference_processing.xml",
            "function_definitions": [
                {
                    "function_name": "ST_Buffer",
                    "return_type": "geometry",
                    "arguments": ["geometry geom", "double precision radius"],
                    "signature_str": "ST_Buffer(geometry geom, double precision radius)",
                }
            ],
            "description": "Returns a geometry covering points within the given radius.",
            "examples": [{"steps": [{"sql": "SELECT ST_Buffer(geom, 10) FROM parcels;"}]}],
        },
        {
            "function_id": "st_union",
            "chapter_info": "reference_processing",
            "source_file": "reference_processing.xml",
            "function_definitions": [
                {
                    "function_name": "ST_Union",
                    "return_type": "geometry",
                    "arguments": ["geometry geom1", "geometry geom2"],
                    "signature_str": "ST_Union(geometry geom1, geometry geom2)",
                }
            ],
            "description": "Returns the union of two geometries.",
            "examples": [{"steps": [{"sql": "SELECT ST_Union(a.geom, b.geom) FROM a JOIN b ON true;"}]}],
        },
        {
            "function_id": "st_asraster",
            "chapter_info": "reference_raster",
            "source_file": "reference_raster.xml",
            "function_definitions": [
                {
                    "function_name": "ST_AsRaster",
                    "return_type": "raster",
                    "arguments": ["geometry geom"],
                    "signature_str": "ST_AsRaster(geometry geom)",
                }
            ],
            "description": "Raster output function.",
            "examples": [],
        },
        {
            "function_id": "st_topology",
            "chapter_info": "reference_topology",
            "source_file": "reference_topology.xml",
            "function_definitions": [
                {
                    "function_name": "ST_TopologyThing",
                    "return_type": "geometry",
                    "arguments": ["geometry geom"],
                    "signature_str": "ST_TopologyThing(geometry geom)",
                }
            ],
            "description": "Topology function.",
            "examples": [],
        },
    ]


def _sample_function_markdown():
    return "\n".join(
        [
            "## spatialsql_pg",
            "ST_DWithin",
            "ST_Buffer",
            "ST_Contains",
        ]
    )


def _load_library(payload, markdown_text):
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "postgis.json"
        md_path = Path(tmpdir) / "ST_Function.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(markdown_text, encoding="utf-8")
        return PostGISFunctionLibrary.load(json_path, md_path, ["raster", "topology"])


def _make_config(**kwargs) -> SQLSynthesisConfig:
    base = SQLSynthesisConfig(
        database=SQLSynthesisDBConfig(),
        llm=SQLSynthesisLLMConfig(provider="mock", model="mock-model", base_url="http://mock", api_key_env="OPENAI_API_KEY"),
        synthesis=SQLSynthesisRunConfig(
            num_sql_per_database={"default": 1},
            max_revision_rounds=1,
            keep_invalid=False,
            keep_failed_execution=False,
        ),
        functions=SQLSynthesisFunctionConfig(postgis_function_json_path="", st_function_markdown_path=""),
        execution=SQLExecutionCheckConfig(enable_execution_check=True, require_non_empty_result=True),
        logging=SQLSynthesisLoggingConfig(),
    )
    return override_sql_synthesis_config(
        base,
        synthesis=kwargs.get("synthesis"),
        execution=kwargs.get("execution"),
        llm=kwargs.get("llm"),
    )


class FakeExecutionChecker:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def check(self, sql, database):
        self.calls.append((sql, database.database_id))
        if not self.results:
            raise RuntimeError("No queued execution result.")
        return self.results.pop(0)


class FakePromptMetadataProvider:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def load_database_metadata(self, database):
        self.calls.append(database.database_id)
        return self.payload


class SQLSynthesisTests(unittest.TestCase):
    def test_config_loading_and_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "sql_synthesis.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "database:",
                        "  host: db.local",
                        "  port: 6543",
                        "llm:",
                        "  model: demo-model",
                        "synthesis:",
                        "  input_path: data/in.jsonl",
                        "  output_path: data/out.jsonl",
                        "  num_sql_per_database:",
                        "    nyc: 8",
                        "    sf: 4",
                        "  difficulty_weights:",
                        "    easy: 2",
                        "    medium: 1",
                        "    hard: 0",
                        "    extra-hard: 0",
                    ]
                ),
                encoding="utf-8",
            )
            loaded = load_sql_synthesis_config(config_path)
            overridden = override_sql_synthesis_config(
                loaded,
                synthesis={"output_path": str(root / "override.jsonl")},
                llm={"model": "override-model"},
            )
        self.assertEqual(loaded.database.host, "db.local")
        self.assertEqual(loaded.database.port, 6543)
        self.assertEqual(overridden.llm.model, "override-model")
        self.assertTrue(overridden.synthesis.output_path.endswith("override.jsonl"))
        self.assertEqual(loaded.synthesis.num_sql_per_database, {"nyc": 8, "sf": 4})

    def test_build_sql_generator_supports_ollama_provider(self):
        generator = build_sql_generator(
            provider="ollama",
            model="qwen2.5:14b",
            base_url="http://localhost:11434",
            api_key_env="IGNORED_FOR_OLLAMA",
            temperature=0.1,
            max_tokens=512,
            timeout=30,
            max_retries=1,
        )
        self.assertIsInstance(generator, OllamaSQLGenerator)

    def test_sql_synthesis_prompt_limits_representative_values_and_masks_geometry(self):
        table = CanonicalSpatialTable.from_dict(
            {
                "table_id": "t1",
                "city": "nyc",
                "table_name": "table_1",
                "normalized_schema": [
                    {"name": "name", "canonical_name": "name", "canonical_type": "text"},
                    {"name": "geometry", "canonical_name": "geometry", "canonical_type": "spatial"},
                ],
                "spatial_fields": [{"canonical_name": "geometry", "crs": "EPSG:4326"}],
                "representative_values": {
                    "name": ["a", "b", "c", "d"],
                    "geometry": [
                        "POINT (1 2)",
                        "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                        "LINESTRING (0 0, 1 1)",
                        "POINT (9 9)",
                    ],
                },
            }
        )
        database = SynthesizedSpatialDatabase.from_selected_tables(
            database_id="nyc_0001",
            city="nyc",
            selected_tables=[table],
            sampling_trace=[],
            graph_stats={},
            synthesize_config={},
        )
        prompt_builder = PromptBuilder({"project_root": Path.cwd()})
        prompt = prompt_builder.build_sql_synthesis_prompt(
            database=database,
            difficulty_level="easy",
            structural_constraints={"difficulty_level": "easy"},
            sampled_functions=[],
        )
        representative_section = prompt.split("## Representative Values", 1)[1].split("## Difficulty Constraint", 1)[0].strip()
        representative_values = json.loads(representative_section)
        self.assertEqual(
            representative_values["table_1"],
            [
                {"geometry": "POINT", "name": "a"},
                {"geometry": "POLYGON", "name": "b"},
                {"geometry": "LINESTRING", "name": "c"},
            ],
        )

    def test_ollama_generator_uses_openai_style_client(self):
        generator = OllamaSQLGenerator(
            model="qwen2.5:14b",
            base_url="http://localhost:11434",
            api_key_env="IGNORED_FOR_OLLAMA",
            temperature=0.1,
            max_tokens=512,
            timeout=30,
            max_retries=1,
        )
        from unittest import mock

        class FakeUsage:
            prompt_tokens = 12
            completion_tokens = 34
            total_tokens = 46

        class FakeMessage:
            content = '{"sql":"SELECT 1","used_tables":[],"used_columns":[],"used_spatial_functions":[],"reasoning_summary":"ok"}'

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]
            usage = FakeUsage()

            def model_dump(self):
                return {"id": "fake"}

        with mock.patch.object(
            generator.client.chat.completions,
            "create",
            return_value=FakeResponse(),
        ) as patched:
            response = generator.generate("Return SQL JSON")
        patched.assert_called_once()
        self.assertIn('"sql":"SELECT 1"', response.text)
        self.assertEqual(response.usage["prompt_tokens"], 12)
        self.assertEqual(response.usage["completion_tokens"], 34)
        self.assertEqual(response.usage["total_tokens"], 46)
        self.assertEqual(response.attempts, 1)

    def test_function_library_loads_json_and_markdown_and_excludes_raster_topology(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "postgis.json"
            md_path = Path(tmpdir) / "ST_Function.md"
            json_path.write_text(json.dumps(_sample_function_json_payload(), ensure_ascii=False), encoding="utf-8")
            md_path.write_text(_sample_function_markdown(), encoding="utf-8")
            library = PostGISFunctionLibrary.load(json_path, md_path, ["raster", "topology"])
        names = {item.function_name for item in library.functions}
        self.assertIn("ST_DWithin", names)
        self.assertIn("ST_Buffer", names)
        self.assertIn("ST_Contains", names)
        self.assertNotIn("ST_AsRaster", names)
        self.assertNotIn("ST_TopologyThing", names)

    def test_function_sampling_and_seed_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "postgis.json"
            md_path = Path(tmpdir) / "ST_Function.md"
            json_path.write_text(json.dumps(_sample_function_json_payload(), ensure_ascii=False), encoding="utf-8")
            md_path.write_text(_sample_function_markdown(), encoding="utf-8")
            library = PostGISFunctionLibrary.load(json_path, md_path, ["raster", "topology"])
        import numpy as np

        db = _make_database(table_count=2)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        sample1 = [item.function_name for item in library.sample_functions(db, "medium", rng1)]
        sample2 = [item.function_name for item in library.sample_functions(db, "medium", rng2)]
        self.assertEqual(sample1, sample2)
        self.assertTrue(sample1)

    def test_function_sampling_prefers_st_function_markdown_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "postgis.json"
            md_path = Path(tmpdir) / "ST_Function.md"
            json_path.write_text(
                json.dumps(_sample_function_json_payload()[:3], ensure_ascii=False),
                encoding="utf-8",
            )
            md_path.write_text("## spatialsql_pg\nST_Buffer\n", encoding="utf-8")
            library = PostGISFunctionLibrary.load(json_path, md_path, ["raster", "topology"])
        import numpy as np

        db = _make_database(table_count=1)
        sampled = library.sample_functions(db, "easy", np.random.default_rng(11))
        self.assertEqual(len(sampled), 1)
        self.assertEqual(sampled[0].function_name, "ST_Buffer")
        self.assertIn("ST_Function.md", sampled[0].source)

    def test_prompt_builder_contains_required_sql_context(self):
        builder = PromptBuilder({"project_root": Path.cwd()})
        database = _make_database(table_count=2)
        prompt = builder.build_sql_synthesis_prompt(
            database=database,
            difficulty_level="medium",
            structural_constraints={"min_tables": 2, "require_join": True},
            sampled_functions=[
                {
                    "function_name": "ST_DWithin",
                    "signature": "ST_DWithin(geometry, geometry, double precision)",
                    "categories": ["spatial_join"],
                    "description": "Returns true if geometries are within distance.",
                    "example_usages": ["SELECT ST_DWithin(a.geom, b.geom, 100)"],
                }
            ],
        )
        self.assertIn("Representative Values", prompt)
        self.assertIn("Difficulty Constraint", prompt)
        self.assertIn("ST_DWithin", prompt)
        self.assertIn("used_spatial_functions", prompt)
        self.assertIn("Return a JSON object only", prompt)

    def test_prompt_builder_prefers_live_postgis_metadata(self):
        builder = PromptBuilder({"project_root": Path.cwd()})
        database = _make_database(table_count=1)
        runtime_metadata = {
            "tables": [
                {
                    "table_name": "table_1",
                    "columns": [
                        {"column_name": "id", "column_type": "integer"},
                        {"column_name": "name", "column_type": "text"},
                        {"column_name": "shape", "column_type": "geography(Point,4326)"},
                    ],
                    "spatial_fields": [
                        {
                            "column_name": "shape",
                            "column_type": "geography(Point,4326)",
                            "spatial_type": "geography",
                            "geometry_type": "POINT",
                            "srid": 4326,
                        }
                    ],
                    "representative_values": {
                        "name": ["hydrant", "valve"],
                        "shape": ["POINT (SRID=4326)"],
                    },
                }
            ]
        }
        prompt = builder.build_sql_synthesis_prompt(
            database=database,
            difficulty_level="easy",
            structural_constraints={"min_tables": 1},
            sampled_functions=[{"function_name": "ST_DWithin"}],
            database_runtime_metadata=runtime_metadata,
        )
        self.assertIn("table_1(id integer, name text, shape geography(Point,4326))", prompt)
        self.assertIn("family=geography", prompt)
        self.assertIn('"hydrant"', prompt)
        self.assertIn('"shape": "POINT"', prompt)
        self.assertIn("Geometry and geography are different types", prompt)
        self.assertNotIn("geom spatial", prompt)

    def test_feedback_prompt_contains_errors_and_empty_result(self):
        builder = PromptBuilder({"project_root": Path.cwd()})
        database = _make_database(table_count=1)
        prompt = builder.build_sql_feedback_prompt(
            database=database,
            difficulty_level="easy",
            structural_constraints={"min_tables": 1},
            sampled_functions=[{"function_name": "ST_Buffer"}],
            original_candidate={"sql": "SELECT * FROM missing_table"},
            validation_errors=["Unknown tables referenced: missing_table"],
            execution_error="relation does not exist",
            empty_result=True,
        )
        self.assertIn("Unknown tables referenced", prompt)
        self.assertIn("relation does not exist", prompt)
        self.assertIn("empty_result: true", prompt)

    def test_feedback_prompt_uses_live_postgis_metadata(self):
        builder = PromptBuilder({"project_root": Path.cwd()})
        database = _make_database(table_count=1)
        runtime_metadata = {
            "tables": [
                {
                    "table_name": "table_1",
                    "columns": [
                        {"column_name": "id", "column_type": "integer"},
                        {"column_name": "footprint", "column_type": "geometry(MultiPolygon,3857)"},
                    ],
                    "spatial_fields": [
                        {
                            "column_name": "footprint",
                            "column_type": "geometry(MultiPolygon,3857)",
                            "spatial_type": "geometry",
                            "geometry_type": "MULTIPOLYGON",
                            "srid": 3857,
                        }
                    ],
                    "representative_values": {
                        "id": [1, 2, 3],
                    },
                }
            ]
        }
        prompt = builder.build_sql_feedback_prompt(
            database=database,
            difficulty_level="easy",
            structural_constraints={"min_tables": 1},
            sampled_functions=[{"function_name": "ST_Buffer"}],
            original_candidate={"sql": "SELECT * FROM t"},
            validation_errors=["x"],
            execution_error="boom",
            empty_result=False,
            database_runtime_metadata=runtime_metadata,
        )
        self.assertIn("footprint geometry(MultiPolygon,3857)", prompt)
        self.assertIn("geometry_type=MULTIPOLYGON", prompt)
        self.assertIn('"id": 1', prompt)
        self.assertIn("geometry/geography signature mismatches", prompt)

    def test_response_parser_supports_json_and_markdown_json(self):
        plain = parse_sql_generation_response(
            '{"sql":"SELECT ST_Buffer(geom, 10) FROM table_1","used_tables":["table_1"],"used_columns":["geom"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"ok"}'
        )
        fenced = parse_sql_generation_response(
            "```json\n{\"sql\":\"SELECT ST_DWithin(a.geom,b.geom,10) FROM table_1 a JOIN table_2 b ON true\",\"used_tables\":[\"table_1\",\"table_2\"],\"used_columns\":[\"geom\"],\"used_spatial_functions\":[\"ST_DWithin\"],\"reasoning_summary\":\"ok\"}\n```"
        )
        self.assertEqual(plain.used_spatial_functions, ["ST_Buffer"])
        self.assertEqual(fenced.used_tables, ["table_1", "table_2"])

    def test_validator_detects_dangerous_sql_and_hallucinated_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "postgis.json"
            md_path = Path(tmpdir) / "ST_Function.md"
            json_path.write_text(json.dumps(_sample_function_json_payload(), ensure_ascii=False), encoding="utf-8")
            md_path.write_text(_sample_function_markdown(), encoding="utf-8")
            library = PostGISFunctionLibrary.load(json_path, md_path, ["raster", "topology"])
        validator = SQLValidator(library)
        database = _make_database(table_count=1)
        self.assertTrue(contains_dangerous_sql("DROP TABLE x"))
        invalid = validator.validate(
            sql="SELECT ST_DWithin(x.geom, y.geom, 10) FROM missing_table x JOIN table_1 y ON true",
            database=database,
            sampled_functions=["ST_DWithin"],
            difficulty_level="medium",
        )
        self.assertFalse(invalid.is_valid)
        self.assertTrue(any("Unknown tables" in item for item in invalid.errors))
        valid = validator.validate(
            sql="SELECT ST_Buffer(t.geom, 10) FROM table_1 t LIMIT 5",
            database=database,
            sampled_functions=["ST_Buffer"],
            difficulty_level="easy",
        )
        self.assertTrue(valid.is_valid)
        self.assertIn("ST_Buffer", valid.detected_spatial_functions)

    def test_validator_prefers_live_postgis_schema_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "postgis.json"
            md_path = Path(tmpdir) / "ST_Function.md"
            json_path.write_text(json.dumps(_sample_function_json_payload(), ensure_ascii=False), encoding="utf-8")
            md_path.write_text(_sample_function_markdown(), encoding="utf-8")
            library = PostGISFunctionLibrary.load(json_path, md_path, ["raster", "topology"])
        validator = SQLValidator(library)
        database = _make_database(table_count=1)
        runtime_metadata = {
            "tables": [
                {
                    "table_name": "table_1",
                    "columns": [
                        {"column_name": "id", "column_type": "integer"},
                        {"column_name": "shape", "column_type": "geography(Point,4326)"},
                    ],
                }
            ]
        }
        result = validator.validate(
            sql="SELECT ST_Buffer(t.shape::geometry, 10) FROM table_1 t LIMIT 5",
            database=database,
            sampled_functions=["ST_Buffer"],
            difficulty_level="easy",
            database_runtime_metadata=runtime_metadata,
        )
        self.assertTrue(result.is_valid)
        self.assertIn("shape", result.detected_columns)

    def test_execution_checker_rejects_write_operations(self):
        checker = SQLExecutionChecker(
            SQLSynthesisDBConfig(),
            SQLExecutionCheckConfig(enable_execution_check=True, require_non_empty_result=False),
        )
        result = checker.check("INSERT INTO x VALUES (1)", _make_database(table_count=1))
        self.assertFalse(result.success)
        self.assertFalse(result.executed)

    def test_end_to_end_synthesizer_revises_after_execution_error(self):
        library = _load_library([_sample_function_json_payload()[0]], "## spatialsql_pg\nST_DWithin\n")
        database = _make_database(table_count=2)
        config = _make_config()
        generator = MockSQLGenerator(
            responses=[
                json.dumps(
                    {
                        "sql": "SELECT ST_DWithin(a.geom, b.geom, 10) FROM table_1 a JOIN table_2 b ON true WHERE a.name = 'x'",
                        "used_tables": ["table_1", "table_2"],
                        "used_columns": ["geom", "name"],
                        "used_spatial_functions": ["ST_DWithin"],
                        "reasoning_summary": "first try",
                    }
                ),
                json.dumps(
                    {
                        "sql": "SELECT ST_DWithin(a.geom, b.geom, 1000) FROM table_1 a JOIN table_2 b ON true LIMIT 5",
                        "used_tables": ["table_1", "table_2"],
                        "used_columns": ["geom"],
                        "used_spatial_functions": ["ST_DWithin"],
                        "reasoning_summary": "second try",
                    }
                ),
            ]
        )
        from src.synthesis.sql.models import SQLExecutionResult

        execution_checker = FakeExecutionChecker(
            [
                SQLExecutionResult(executed=True, success=False, error_message="operator does not exist"),
                SQLExecutionResult(executed=True, success=True, row_count=1, sample_rows=[{"st_dwithin": True}], execution_time_ms=10.0),
            ]
        )
        synthesizer = ConstraintGuidedSQLSynthesizer(
            config=config,
            function_library=library,
            sql_generator=generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=execution_checker,
        )
        rows = synthesizer.synthesize_for_database(database)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].revision_rounds, 1)
        self.assertIn("operator does not exist", rows[0].feedback_prompts[0])
        self.assertIn("1000", rows[0].sql)

    def test_synthesizer_logs_city_schema_function_and_progress(self):
        library = _load_library([_sample_function_json_payload()[0]], "## spatialsql_pg\nST_DWithin\n")
        database = _make_database(table_count=2)
        config = _make_config()
        generator = MockSQLGenerator(
            responses=[
                json.dumps(
                    {
                        "sql": "SELECT ST_DWithin(a.geom, b.geom, 10) FROM table_1 a JOIN table_2 b ON true",
                        "used_tables": ["table_1", "table_2"],
                        "used_columns": ["geom"],
                        "used_spatial_functions": ["ST_DWithin"],
                        "reasoning_summary": "ok",
                    }
                )
            ]
        )
        from src.synthesis.sql.models import SQLExecutionResult

        synthesizer = ConstraintGuidedSQLSynthesizer(
            config=config,
            function_library=library,
            sql_generator=generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker([SQLExecutionResult(executed=True, success=True, row_count=1)]),
        )
        with self.assertLogs("src.synthesis.sql.synthesizer", level="INFO") as captured:
            rows = synthesizer.synthesize_for_database(database)
        self.assertEqual(len(rows), 1)
        log_text = "\n".join(captured.output)
        self.assertIn("LLM prompt | sample=nyc/nyc_0001/sql_0001 | round=1/2", log_text)
        self.assertIn("## Task Goal", log_text)
        self.assertIn("SQL synthesis progress 1/1", log_text)
        self.assertIn("city=nyc", log_text)
        self.assertIn("schema_id=nyc_0001", log_text)
        self.assertIn("spatial_functions=ST_DWithin", log_text)

    def test_empty_result_triggers_feedback_and_keep_controls(self):
        library = _load_library([_sample_function_json_payload()[1]], "## spatialsql_pg\nST_Buffer\n")
        database = _make_database(table_count=1)
        from src.synthesis.sql.models import SQLExecutionResult

        generator = MockSQLGenerator(
            responses=[
                json.dumps(
                    {
                        "sql": "SELECT ST_Buffer(t.geom, 10) FROM table_1 t LIMIT 5",
                        "used_tables": ["table_1"],
                        "used_columns": ["geom"],
                        "used_spatial_functions": ["ST_Buffer"],
                        "reasoning_summary": "first try",
                    }
                ),
                json.dumps(
                    {
                        "sql": "SELECT ST_Buffer(t.geom, 100) FROM table_1 t LIMIT 5",
                        "used_tables": ["table_1"],
                        "used_columns": ["geom"],
                        "used_spatial_functions": ["ST_Buffer"],
                        "reasoning_summary": "second try",
                    }
                ),
            ]
        )
        execution_checker = FakeExecutionChecker(
            [
                SQLExecutionResult(executed=True, success=False, empty_result=True, error_message="SQL executed successfully but returned no rows."),
                SQLExecutionResult(executed=True, success=True, row_count=1, sample_rows=[{"geom": "x"}]),
            ]
        )
        synthesizer = ConstraintGuidedSQLSynthesizer(
            config=_make_config(),
            function_library=library,
            sql_generator=generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=execution_checker,
        )
        rows = synthesizer.synthesize_for_database(database)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].revision_rounds, 1)
        self.assertIn("empty_result", rows[0].feedback_prompts[0])

    def test_synthesizer_uses_live_postgis_metadata_in_prompt(self):
        library = _load_library([_sample_function_json_payload()[1]], "## spatialsql_pg\nST_Buffer\n")
        database = _make_database(table_count=1)
        runtime_metadata = {
            "tables": [
                {
                    "table_name": "table_1",
                    "columns": [
                        {"column_name": "id", "column_type": "integer"},
                        {"column_name": "name", "column_type": "text"},
                        {"column_name": "shape", "column_type": "geography(Point,4326)"},
                    ],
                    "spatial_fields": [
                        {
                            "column_name": "shape",
                            "column_type": "geography(Point,4326)",
                            "spatial_type": "geography",
                            "geometry_type": "POINT",
                            "srid": 4326,
                        }
                    ],
                    "representative_values": {"name": ["hydrant", "valve"]},
                }
            ]
        }
        generator = MockSQLGenerator(
            responses=['{"sql":"SELECT ST_Buffer(t.shape::geometry, 10) FROM table_1 t LIMIT 5","used_tables":["table_1"],"used_columns":["shape"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"ok"}']
        )
        from src.synthesis.sql.models import SQLExecutionResult

        provider = FakePromptMetadataProvider(runtime_metadata)
        synthesizer = ConstraintGuidedSQLSynthesizer(
            config=_make_config(synthesis={"max_revision_rounds": 0}),
            function_library=library,
            sql_generator=generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker([SQLExecutionResult(executed=True, success=True, row_count=1)]),
            prompt_metadata_provider=provider,
        )
        rows = synthesizer.synthesize_for_database(database)
        self.assertEqual(len(rows), 1)
        self.assertEqual(provider.calls, ["nyc_0001"])
        self.assertIn("shape geography(Point,4326)", generator.prompts[0])
        self.assertIn('"hydrant"', generator.prompts[0])
        self.assertNotIn("geom spatial", generator.prompts[0])

    def test_random_seed_keeps_sampling_prompt_reproducible(self):
        library = _load_library([_sample_function_json_payload()[0]], "## spatialsql_pg\nST_DWithin\n")
        database = _make_database(table_count=2)
        prompt_builder_1 = PromptBuilder({"project_root": Path.cwd()})
        prompt_builder_2 = PromptBuilder({"project_root": Path.cwd()})
        generator_1 = MockSQLGenerator(responses=['{"sql":"SELECT ST_DWithin(a.geom,b.geom,10) FROM table_1 a JOIN table_2 b ON true","used_tables":["table_1","table_2"],"used_columns":["geom"],"used_spatial_functions":["ST_DWithin"],"reasoning_summary":"ok"}'])
        generator_2 = MockSQLGenerator(responses=['{"sql":"SELECT ST_DWithin(a.geom,b.geom,10) FROM table_1 a JOIN table_2 b ON true","used_tables":["table_1","table_2"],"used_columns":["geom"],"used_spatial_functions":["ST_DWithin"],"reasoning_summary":"ok"}'])
        config = _make_config()
        from src.synthesis.sql.models import SQLExecutionResult

        exec_checker = FakeExecutionChecker([SQLExecutionResult(executed=True, success=True, row_count=1)])
        exec_checker_2 = FakeExecutionChecker([SQLExecutionResult(executed=True, success=True, row_count=1)])
        synth1 = ConstraintGuidedSQLSynthesizer(config=config, function_library=library, sql_generator=generator_1, prompt_builder=prompt_builder_1, validator=SQLValidator(library), execution_checker=exec_checker)
        synth2 = ConstraintGuidedSQLSynthesizer(config=config, function_library=library, sql_generator=generator_2, prompt_builder=prompt_builder_2, validator=SQLValidator(library), execution_checker=exec_checker_2)
        rows1 = synth1.synthesize_for_database(database)
        rows2 = synth2.synthesize_for_database(database)
        self.assertEqual(rows1[0].difficulty_level, rows2[0].difficulty_level)
        self.assertEqual(rows1[0].spatial_function_constraints, rows2[0].spatial_function_constraints)
        self.assertEqual(generator_1.prompts[0], generator_2.prompts[0])

    def test_keep_invalid_and_keep_failed_execution(self):
        library = _load_library([_sample_function_json_payload()[1]], "## spatialsql_pg\nST_Buffer\n")
        database = _make_database(table_count=1)
        invalid_generator = MockSQLGenerator(
            responses=['{"sql":"SELECT * FROM missing_table","used_tables":["missing_table"],"used_columns":[],"used_spatial_functions":[],"reasoning_summary":"bad"}']
        )
        synth_invalid = ConstraintGuidedSQLSynthesizer(
            config=_make_config(synthesis={"keep_invalid": True, "max_revision_rounds": 0}),
            function_library=library,
            sql_generator=invalid_generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker([]),
        )
        kept_invalid = synth_invalid.synthesize_for_database(database)
        self.assertEqual(len(kept_invalid), 1)

        failed_exec_generator = MockSQLGenerator(
            responses=['{"sql":"SELECT ST_Buffer(t.geom, 10) FROM table_1 t","used_tables":["table_1"],"used_columns":["geom"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"ok"}']
        )
        from src.synthesis.sql.models import SQLExecutionResult
        synth_failed_exec = ConstraintGuidedSQLSynthesizer(
            config=_make_config(synthesis={"keep_failed_execution": True, "max_revision_rounds": 0}),
            function_library=library,
            sql_generator=failed_exec_generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker([SQLExecutionResult(executed=True, success=False, error_message="execution failed")]),
        )
        kept_failed = synth_failed_exec.synthesize_for_database(database)
        self.assertEqual(len(kept_failed), 1)

    def test_num_sql_per_database_uses_city_mapping(self):
        library = _load_library([_sample_function_json_payload()[1]], "## spatialsql_pg\nST_Buffer\n")
        config = _make_config(synthesis={"num_sql_per_database": {"nyc": 2}})
        generator = MockSQLGenerator(
            responses=[
                '{"sql":"SELECT ST_Buffer(t.geom, 10) FROM table_1 t","used_tables":["table_1"],"used_columns":["geom"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"ok"}',
                '{"sql":"SELECT ST_Buffer(t.geom, 20) FROM table_1 t","used_tables":["table_1"],"used_columns":["geom"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"ok"}',
            ]
        )
        from src.synthesis.sql.models import SQLExecutionResult

        synthesizer = ConstraintGuidedSQLSynthesizer(
            config=config,
            function_library=library,
            sql_generator=generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker(
                [
                    SQLExecutionResult(executed=True, success=True, row_count=1),
                    SQLExecutionResult(executed=True, success=True, row_count=1),
                ]
            ),
        )

        nyc_rows = synthesizer.synthesize_for_database(_make_database(table_count=1, database_id="nyc_0001"))
        sf_rows = synthesizer.synthesize_for_database(_make_database(table_count=1, database_id="sf_0001"))
        self.assertEqual(len(nyc_rows), 2)
        self.assertEqual(len(sf_rows), 0)

    def test_difficulty_plan_follows_weights_in_easy_to_extra_hard_order(self):
        library = _load_library(_sample_function_json_payload()[:3], _sample_function_markdown())
        config = _make_config(
            synthesis={
                "num_sql_per_database": {"nyc": 8},
                "difficulty_weights": {
                    "easy": 1,
                    "medium": 1,
                    "hard": 1,
                    "extra-hard": 1,
                },
                "max_revision_rounds": 0,
            }
        )
        synthesizer = ConstraintGuidedSQLSynthesizer(
            config=config,
            function_library=library,
            sql_generator=MockSQLGenerator(responses=[]),
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker([]),
        )
        rows = synthesizer._build_difficulty_plan(_make_database(table_count=3, database_id="nyc_0001"), 8)
        self.assertEqual(
            rows,
            ["easy", "easy", "medium", "medium", "hard", "hard", "extra-hard", "extra-hard"],
        )


if __name__ == "__main__":
    unittest.main()
