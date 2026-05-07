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
    build_create_table_ddl_query,
    contains_dangerous_sql,
    append_sql_query,
    ensure_sql_output,
    initialize_sql_output,
    load_existing_sql_id_offsets,
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
            "function_id": "st_contains",
            "chapter_info": "reference_relationship",
            "source_file": "reference_relationship.xml",
            "function_definitions": [
                {
                    "function_name": "ST_Contains",
                    "return_type": "boolean",
                    "arguments": ["geometry geomA", "geometry geomB"],
                    "signature_str": "ST_Contains(geometry geomA, geometry geomB)",
                }
            ],
            "description": "Returns true if geometry A contains geometry B.",
            "examples": [{"steps": [{"sql": "SELECT ST_Contains(a.geom, b.geom) FROM a JOIN b ON true;"}]}],
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
        execution=SQLExecutionCheckConfig(enable_execution_check=True, require_non_empty_result=False),
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
        self.calls.append((database.database_id, list(database.selected_table_names)))
        if callable(self.payload):
            return self.payload(database)
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

    def test_sql_output_is_appended_incrementally(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sql.jsonl"
            initialize_sql_output(str(output_path))
            row1 = type(
                "Row",
                (),
                {
                    "to_dict": lambda self: {"sql_id": "a", "sql": "SELECT 1"},
                },
            )()
            row2 = type(
                "Row",
                (),
                {
                    "to_dict": lambda self: {"sql_id": "b", "sql": "SELECT 2"},
                },
            )()
            append_sql_query(str(output_path), row1)
            lines_after_first = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines_after_first), 1)
            append_sql_query(str(output_path), row2)
            lines_after_second = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines_after_second), 2)
            self.assertIn('"sql_id": "a"', lines_after_second[0])
            self.assertIn('"sql_id": "b"', lines_after_second[1])

    def test_ensure_sql_output_preserves_existing_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sql.jsonl"
            output_path.write_text('{"sql_id":"existing","sql":"SELECT 0"}\n', encoding="utf-8")
            row = type(
                "Row",
                (),
                {
                    "to_dict": lambda self: {"sql_id": "new", "sql": "SELECT 1"},
                },
            )()

            ensure_sql_output(str(output_path))
            append_sql_query(str(output_path), row)

            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('"sql_id":"existing"', lines[0])
            self.assertIn('"sql_id": "new"', lines[1])

    def test_load_existing_sql_id_offsets_tracks_max_suffix_per_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sql.jsonl"
            output_path.write_text(
                "\n".join(
                    [
                        '{"sql_id":"nyc_0001_0002","database_id":"nyc_0001","sql":"SELECT 1"}',
                        '{"sql_id":"nyc_0001_0005","database_id":"nyc_0001","sql":"SELECT 2"}',
                        '{"sql_id":"sf_0001_0003","sql":"SELECT 3"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            offsets = load_existing_sql_id_offsets(str(output_path))

            self.assertEqual(offsets, {"nyc_0001": 5, "sf_0001": 3})

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

    def test_build_sql_generator_supports_config_object_and_mock_provider(self):
        generator = build_sql_generator(
            config=SQLSynthesisLLMConfig(
                provider="mock",
                model="mock-model",
                base_url="http://mock",
                api_key_env="OPENAI_API_KEY",
            )
        )
        self.assertIsInstance(generator, MockSQLGenerator)

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

        buffer_items = [item for item in library.functions if item.function_name == "ST_Buffer"]
        self.assertTrue(buffer_items)
        self.assertIn("ST_Function.md", buffer_items[0].source)
        self.assertEqual(
            buffer_items[0].description,
            "Returns a geometry covering points within the given radius.",
        )
        contains_items = [item for item in library.functions if item.function_name == "ST_Contains"]
        self.assertTrue(contains_items)
        self.assertIn("ST_Function.md", contains_items[0].source)
        self.assertEqual(
            contains_items[0].description,
            "Returns true if geometry A contains geometry B.",
        )

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

    def test_function_library_matches_markdown_entries_by_function_id(self):
        payload = [
            {
                "function_id": "st_intersects",
                "chapter_info": "reference_relationship",
                "source_file": "reference_relationship.xml",
                "function_definitions": [
                    {
                        "function_name": "ST_IntersectsGeometry",
                        "return_type": "boolean",
                        "arguments": ["geometry geom1", "geometry geom2"],
                        "signature_str": "ST_IntersectsGeometry(geometry geom1, geometry geom2)",
                    },
                    {
                        "function_name": "ST_IntersectsGeography",
                        "return_type": "boolean",
                        "arguments": ["geography geog1", "geography geog2"],
                        "signature_str": "ST_IntersectsGeography(geography geog1, geography geog2)",
                    },
                ],
                "description": "Returns true if two spatial objects intersect.",
                "examples": [],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "postgis.json"
            md_path = Path(tmpdir) / "ST_Function.md"
            json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            md_path.write_text("## spatialsql_pg\nST_Intersects\n", encoding="utf-8")
            library = PostGISFunctionLibrary.load(json_path, md_path, ["raster", "topology"])

        intersect_items = [item for item in library.functions if item.metadata.get("function_id") == "st_intersects"]
        self.assertEqual(len(intersect_items), 2)
        for item in intersect_items:
            self.assertIn("ST_Function.md", item.source)

    def test_function_library_does_not_exclude_valid_functions_with_version_text_in_description(self):
        payload = [
            {
                "function_id": "ST_IsValid",
                "chapter_info": "reference_validation",
                "source_file": "reference_validation.xml",
                "function_definitions": [
                    {
                        "function_name": "ST_IsValid",
                        "return_type": "boolean",
                        "arguments": ["geometry g"],
                        "signature_str": "ST_IsValid(geometry g)",
                    }
                ],
                "description": "Tests geometry validity. The flag is a PostGIS extension. The version accepting flags is available starting with 2.0.0.",
                "examples": [],
            },
            {
                "function_id": "PostGIS_Extensions_Upgrade",
                "chapter_info": "reference_version",
                "source_file": "reference_version.xml",
                "function_definitions": [
                    {
                        "function_name": "PostGIS_Extensions_Upgrade",
                        "return_type": "text",
                        "arguments": [],
                        "signature_str": "PostGIS_Extensions_Upgrade()",
                    }
                ],
                "description": "Upgrade helper.",
                "examples": [],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "postgis.json"
            md_path = Path(tmpdir) / "ST_Function.md"
            json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            md_path.write_text("## spatialsql_pg\nST_IsValid\n", encoding="utf-8")
            library = PostGISFunctionLibrary.load(json_path, md_path, ["raster", "topology"])

        names = {item.function_name for item in library.functions}
        self.assertIn("ST_IsValid", names)
        self.assertNotIn("PostGIS_Extensions_Upgrade", names)
        is_valid_items = [item for item in library.functions if item.function_name == "ST_IsValid"]
        self.assertTrue(is_valid_items)
        self.assertIn("ST_Function.md", is_valid_items[0].source)

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
        self.assertNotIn("Spatial Field Metadata", prompt)
        self.assertIn("Difficulty Constraint", prompt)
        self.assertIn("ST_DWithin", prompt)
        self.assertIn("used_spatial_functions", prompt)
        self.assertIn("Return a JSON object only", prompt)
        self.assertIn("do not use `SELECT *`", prompt)

    def test_minor_revision_prompt_contains_error_and_involved_table_metadata(self):
        builder = PromptBuilder({"project_root": Path.cwd()})
        database = _make_database(table_count=2)
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
                },
                {
                    "table_name": "table_2",
                    "columns": [
                        {"column_name": "id", "column_type": "integer"},
                    ],
                    "spatial_fields": [],
                    "representative_values": {"id": [1]},
                },
            ]
        }
        prompt = builder.build_sql_revision_prompt(
            database=database,
            original_sql="SELECT a.name FROM table_1 a JOIN table_2 b ON ST_DWithin(a.shape, b.geom, 10)",
            execution_error="operator does not exist",
            used_tables=["table_1"],
            database_runtime_metadata=runtime_metadata,
        )
        self.assertIn("operator does not exist", prompt)
        self.assertIn("table_1(id integer, name text, shape geography(Point,4326))", prompt)
        self.assertIn('"hydrant"', prompt)
        self.assertNotIn("table_2(id integer)", prompt)

    def test_prompt_builder_renders_explicit_difficulty_tier_guidance(self):
        builder = PromptBuilder({"project_root": Path.cwd()})
        database = _make_database(table_count=5)
        prompt = builder.build_sql_synthesis_prompt(
            database=database,
            difficulty_level="extra-hard",
            structural_constraints={
                "difficulty_level": "extra-hard",
                "difficulty_summary": "Three-to-four-table query with bounded advanced structure.",
                "min_tables": 3,
                "max_tables": 4,
                "min_spatial_joins": 1,
                "min_advanced_ops": 2,
                "max_advanced_ops": 4,
            },
            sampled_functions=[],
        )
        self.assertIn("Use between 3 and 4 tables.", prompt)
        self.assertIn("Include at least 1 spatial join.", prompt)
        self.assertIn("Count each spatial join and each nested query", prompt)
        self.assertIn("between 2 and 4", prompt)
        self.assertIn("Prefer the simplest executable SQL that satisfies the tier", prompt)

    def test_synthesizer_prompt_includes_fixed_spatial_join_functions_for_medium(self):
        library = _load_library(_sample_function_json_payload()[:4], _sample_function_markdown())
        database = _make_database(table_count=2)
        generator = MockSQLGenerator(
            responses=[
                '{"sql":"SELECT a.name FROM table_1 a JOIN table_2 b ON ST_Intersects(a.geom, b.geom) LIMIT 5","used_tables":["table_1","table_2"],"used_columns":["name","geom"],"used_spatial_functions":["ST_Intersects"],"reasoning_summary":"ok"}'
            ]
        )
        from src.synthesis.sql.models import SQLExecutionResult

        synthesizer = ConstraintGuidedSQLSynthesizer(
            config=_make_config(synthesis={"difficulty": "medium"}),
            function_library=library,
            sql_generator=generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker([SQLExecutionResult(executed=True, success=True, row_count=1)]),
        )
        rows = synthesizer.synthesize_for_database(database)
        self.assertEqual(len(rows), 1)
        prompt = generator.prompts[0]
        self.assertIn("ST_DWithin", prompt)
        self.assertIn("ST_Intersects", prompt)
        self.assertIn("ST_Contains", prompt)

    def test_prompt_builder_prefers_live_postgis_metadata(self):
        builder = PromptBuilder({"project_root": Path.cwd()})
        database = _make_database(table_count=1)
        runtime_metadata = {
            "schema_name": "nyc_0001",
            "tables": [
                {
                    "table_name": "table_1",
                    "create_table_ddl": "CREATE TABLE table_1 (\n    id integer,\n    name text,\n    shape geography(Point,4326)\n);",
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
        self.assertIn("CREATE TABLE table_1", prompt)
        self.assertIn("shape geography(Point,4326)", prompt)
        self.assertNotIn("Spatial Field Metadata", prompt)
        self.assertIn('"hydrant"', prompt)
        self.assertIn('"shape": "POINT"', prompt)
        self.assertNotIn("geom spatial", prompt)

    def test_build_create_table_ddl_query_uses_regclass_placeholders(self):
        query, params = build_create_table_ddl_query("nyc_0001", "biannual_pedestrian_counts_map")
        self.assertIn("c.oid = %s::regclass", query)
        self.assertIn("conrelid = %s::regclass", query)
        self.assertIn("'    %%I %%s%%s%%s'", query)
        self.assertIn("'    CONSTRAINT %%I %%s'", query)
        self.assertIn("'CREATE TABLE %%I (%%s%%s%%s);'", query)
        self.assertEqual(
            params,
            (
                "nyc_0001.biannual_pedestrian_counts_map",
                "nyc_0001.biannual_pedestrian_counts_map",
            ),
        )

    def test_synthesized_sql_includes_metadata_for_question_step(self):
        library = _load_library(_sample_function_json_payload()[:4], _sample_function_markdown())
        database = _make_database(table_count=1)
        generator = MockSQLGenerator(
            responses=[
                '{"sql":"SELECT t.name FROM table_1 t LIMIT 5","used_tables":["table_1"],"used_columns":["name"],"used_spatial_functions":[],"reasoning_summary":"ok"}'
            ]
        )
        from src.synthesis.sql.models import SQLExecutionResult

        prompt_metadata = {
            "database_id": "nyc_0001",
            "city": "nyc",
            "schema_name": "nyc_0001",
            "schema_ddls": ["CREATE TABLE table_1 (\n    id integer,\n    name text,\n    geom geometry(GEOMETRY,4326)\n);"],
            "tables": [
                {
                    "table_name": "table_1",
                    "create_table_ddl": "CREATE TABLE table_1 (\n    id integer,\n    name text,\n    geom geometry(GEOMETRY,4326)\n);",
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
                    "representative_values": {"name": ["hydrant"]},
                }
            ],
            "representative_values": {"table_1": {"name": ["hydrant"]}},
        }
        synthesizer = ConstraintGuidedSQLSynthesizer(
            config=_make_config(synthesis={"difficulty": "easy"}),
            function_library=library,
            sql_generator=generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker([SQLExecutionResult(executed=True, success=True, row_count=1)]),
            prompt_metadata_provider=FakePromptMetadataProvider(prompt_metadata),
        )
        rows = synthesizer.synthesize_for_database(database)
        self.assertEqual(len(rows), 1)
        self.assertIn("database_context", rows[0].metadata)
        self.assertIn("schema_ddls", rows[0].metadata["database_context"])
        self.assertIn("CREATE TABLE table_1", rows[0].metadata["database_context"]["schema_ddls"][0])

    def test_response_parser_supports_json_and_markdown_json(self):
        plain = parse_sql_generation_response(
            '{"sql":"SELECT ST_Buffer(geom, 10) FROM table_1","used_tables":["table_1"],"used_columns":["geom"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"ok"}'
        )
        fenced = parse_sql_generation_response(
            "```json\n{\"sql\":\"SELECT a.name FROM table_1 a JOIN table_2 b ON ST_DWithin(a.geom,b.geom,10) LIMIT 5\",\"used_tables\":[\"table_1\",\"table_2\"],\"used_columns\":[\"name\",\"geom\"],\"used_spatial_functions\":[\"ST_DWithin\"],\"reasoning_summary\":\"ok\"}\n```"
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
        extra_function = validator.validate(
            sql="SELECT ST_Buffer(t.geom, 10), ST_Union(t.geom, t.geom) FROM table_1 t LIMIT 5",
            database=database,
            sampled_functions=["ST_Buffer"],
            difficulty_level="easy",
        )
        self.assertFalse(extra_function.is_valid)
        self.assertTrue(
            any("outside the externally provided candidate set" in item for item in extra_function.errors)
        )

    def test_validator_rejects_medium_query_without_spatial_join(self):
        library = _load_library(_sample_function_json_payload()[:4], _sample_function_markdown())
        validator = SQLValidator(library)
        database = _make_database(table_count=2)
        result = validator.validate(
            sql="SELECT ST_Buffer(a.geom, 10) FROM table_1 a JOIN table_2 b ON a.id = b.id LIMIT 5",
            database=database,
            sampled_functions=["ST_Buffer"],
            difficulty_level="medium",
        )
        self.assertFalse(result.is_valid)
        self.assertTrue(any("exactly one spatial join" in item for item in result.errors))

    def test_validator_allows_fixed_spatial_join_functions_for_medium(self):
        library = _load_library(_sample_function_json_payload()[:4], _sample_function_markdown())
        validator = SQLValidator(library)
        database = _make_database(table_count=2)
        result = validator.validate(
            sql="SELECT a.name FROM table_1 a JOIN table_2 b ON ST_Intersects(a.geom, b.geom) LIMIT 5",
            database=database,
            sampled_functions=["ST_DWithin"],
            difficulty_level="medium",
        )
        self.assertTrue(result.is_valid)
        self.assertFalse(
            any("outside the externally provided candidate set" in item for item in result.errors)
        )

    def test_validator_rejects_hard_query_without_two_spatial_joins(self):
        library = _load_library(_sample_function_json_payload()[:4], _sample_function_markdown())
        validator = SQLValidator(library)
        database = _make_database(table_count=3)
        result = validator.validate(
            sql=(
                "SELECT a.name FROM table_1 a "
                "JOIN table_2 b ON ST_Contains(a.geom, b.geom) "
                "JOIN table_3 c ON b.id = c.id "
                "LIMIT 5"
            ),
            database=database,
            sampled_functions=["ST_Contains"],
            difficulty_level="hard",
        )
        self.assertFalse(result.is_valid)
        self.assertTrue(any("exactly two spatial joins" in item for item in result.errors))

    def test_validator_rejects_extra_hard_query_below_minimum_advanced_ops(self):
        library = _load_library(_sample_function_json_payload()[:4], _sample_function_markdown())
        validator = SQLValidator(library)
        database = _make_database(table_count=3)
        result = validator.validate(
            sql=(
                "SELECT a.name FROM table_1 a "
                "JOIN table_2 b ON ST_Contains(a.geom, b.geom) "
                "JOIN table_3 c ON b.id = c.id "
                "LIMIT 5"
            ),
            database=database,
            sampled_functions=["ST_Contains"],
            difficulty_level="extra-hard",
        )
        self.assertFalse(result.is_valid)
        self.assertTrue(any("between two and four operations in total" in item for item in result.errors))

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

    def test_end_to_end_synthesizer_repairs_non_timeout_execution_error_once(self):
        library = _load_library([_sample_function_json_payload()[0]], "## spatialsql_pg\nST_DWithin\n")
        database = _make_database(table_count=2)
        config = _make_config()
        generator = MockSQLGenerator(
            responses=[
                json.dumps(
                    {
                        "sql": "SELECT a.name FROM table_1 a JOIN table_2 b ON ST_DWithin(a.geom, b.geom, 10) WHERE a.name = 'x' LIMIT 5",
                        "used_tables": ["table_1", "table_2"],
                        "used_columns": ["geom", "name"],
                        "used_spatial_functions": ["ST_DWithin"],
                        "reasoning_summary": "first try",
                    }
                ),
                json.dumps(
                    {
                        "sql": "SELECT a.name FROM table_1 a JOIN table_2 b ON ST_DWithin(a.geom, b.geom, 1000) LIMIT 5",
                        "used_tables": ["table_1", "table_2"],
                        "used_columns": ["name", "geom"],
                        "used_spatial_functions": ["ST_DWithin"],
                        "reasoning_summary": "minor fix",
                    }
                ),
            ]
        )
        from src.synthesis.sql.models import SQLExecutionResult

        execution_checker = FakeExecutionChecker(
            [
                SQLExecutionResult(executed=True, success=False, error_message="operator does not exist"),
                SQLExecutionResult(executed=True, success=True, row_count=1, sample_rows=[{"name": "ok"}], execution_time_ms=5.0),
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
        self.assertEqual(rows[0].feedback_prompts, [])
        self.assertEqual(len(rows[0].minor_revision_prompts), 1)
        self.assertIn("operator does not exist", rows[0].minor_revision_prompts[0])
        self.assertIn("table_1_sample", rows[0].minor_revision_prompts[0])
        self.assertIn("1000", rows[0].sql)
        self.assertEqual(rows[0].reasoning_summary, "minor fix")
        self.assertTrue(rows[0].execution_result["success"])
        self.assertEqual(len(generator.prompts), 2)

    def test_synthesizer_continues_sql_id_from_existing_offsets(self):
        library = _load_library([_sample_function_json_payload()[1]], "## spatialsql_pg\nST_Buffer\n")
        database = _make_database(table_count=1)
        config = _make_config(synthesis={"num_sql_per_database": {"default": 2}, "difficulty": "easy"})
        generator = MockSQLGenerator(
            responses=[
                json.dumps(
                    {
                        "sql": "SELECT ST_Buffer(t.geom, 10) FROM table_1 t LIMIT 5",
                        "used_tables": ["table_1"],
                        "used_columns": ["geom"],
                        "used_spatial_functions": ["ST_Buffer"],
                        "reasoning_summary": "first",
                    }
                ),
                json.dumps(
                    {
                        "sql": "SELECT ST_Buffer(t.geom, 20) FROM table_1 t LIMIT 5",
                        "used_tables": ["table_1"],
                        "used_columns": ["geom"],
                        "used_spatial_functions": ["ST_Buffer"],
                        "reasoning_summary": "second",
                    }
                ),
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
            existing_sql_id_offsets={"nyc_0001": 5},
        )

        rows = synthesizer.synthesize_for_database(database)

        self.assertEqual([row.sql_id for row in rows], ["nyc_0001_0006", "nyc_0001_0007"])

    def test_synthesizer_logs_city_schema_function_and_progress(self):
        library = _load_library([_sample_function_json_payload()[0]], "## spatialsql_pg\nST_DWithin\n")
        database = _make_database(table_count=2)
        config = _make_config()
        generator = MockSQLGenerator(
            responses=[
                json.dumps(
                    {
                        "sql": "SELECT a.name FROM table_1 a JOIN table_2 b ON ST_DWithin(a.geom, b.geom, 10) LIMIT 5",
                        "used_tables": ["table_1", "table_2"],
                        "used_columns": ["name", "geom"],
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
        self.assertIn("LLM prompt | sample=nyc/nyc_0001/sql_0001 | round=1/1", log_text)
        self.assertIn("## Database Context", log_text)
        self.assertIn("SQL synthesis progress 1/1", log_text)
        self.assertIn("city=nyc", log_text)
        self.assertIn("schema_id=nyc_0001", log_text)
        self.assertIn("spatial_functions=ST_DWithin", log_text)

    def test_empty_result_is_kept_without_feedback_retry(self):
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
            ]
        )
        execution_checker = FakeExecutionChecker(
            [
                SQLExecutionResult(executed=True, success=True, empty_result=True, row_count=0, sample_rows=[]),
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
        self.assertEqual(rows[0].revision_rounds, 0)
        self.assertEqual(rows[0].feedback_prompts, [])
        self.assertEqual(rows[0].minor_revision_prompts, [])
        self.assertTrue(rows[0].execution_result["empty_result"])

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
        self.assertEqual(provider.calls, [("nyc_0001", ["table_1"])])
        self.assertIn("shape geography(Point,4326)", generator.prompts[0])
        self.assertIn('"hydrant"', generator.prompts[0])
        self.assertNotIn("geom spatial", generator.prompts[0])

    def test_prompt_uses_difficulty_selected_table_subset(self):
        library = _load_library([_sample_function_json_payload()[1]], "## spatialsql_pg\nST_Buffer\n")
        database = _make_database(table_count=4, database_id="nyc_0001")
        generator = MockSQLGenerator(
            responses=[
                '{"sql":"SELECT ST_Buffer(t.geom, 10) FROM table_1 t LIMIT 5","used_tables":["table_1"],"used_columns":["geom"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"ok"}'
            ]
        )
        from src.synthesis.sql.models import SQLExecutionResult

        provider = FakePromptMetadataProvider(
            lambda db: {
                "tables": [
                    {
                        "table_name": table.table_name,
                        "columns": [
                            {"column_name": "id", "column_type": "integer"},
                            {"column_name": "name", "column_type": "text"},
                            {"column_name": "geom", "column_type": "geometry(Point,4326)"},
                        ],
                        "spatial_fields": [
                            {
                                "column_name": "geom",
                                "column_type": "geometry(Point,4326)",
                                "spatial_type": "geometry",
                                "geometry_type": "POINT",
                                "srid": 4326,
                            }
                        ],
                        "representative_values": [{"name": f"{table.table_name}_sample", "geom": "POINT"}],
                    }
                    for table in db.selected_tables
                ]
            }
        )
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
        self.assertEqual(provider.calls, [("nyc_0001", ["table_1"])])
        self.assertEqual(rows[0].generation_metadata["database_table_count"], 4)
        self.assertEqual(rows[0].generation_metadata["prompt_table_count"], 1)
        self.assertEqual(rows[0].generation_metadata["prompt_table_names"], ["table_1"])
        self.assertIn("selected_tables: table_1", generator.prompts[0])
        self.assertNotIn("table_2(id integer", generator.prompts[0])
        self.assertNotIn("table_3(id integer", generator.prompts[0])
        self.assertNotIn("table_4(id integer", generator.prompts[0])

    def test_extra_hard_prompt_subset_allows_up_to_four_tables(self):
        self.assertEqual(
            ConstraintGuidedSQLSynthesizer._resolve_prompt_table_count("extra-hard", 7),
            4,
        )

    def test_random_seed_keeps_sampling_prompt_reproducible(self):
        library = _load_library([_sample_function_json_payload()[0]], "## spatialsql_pg\nST_DWithin\n")
        database = _make_database(table_count=2)
        prompt_builder_1 = PromptBuilder({"project_root": Path.cwd()})
        prompt_builder_2 = PromptBuilder({"project_root": Path.cwd()})
        generator_1 = MockSQLGenerator(responses=['{"sql":"SELECT a.name FROM table_1 a JOIN table_2 b ON ST_DWithin(a.geom,b.geom,10) LIMIT 5","used_tables":["table_1","table_2"],"used_columns":["name","geom"],"used_spatial_functions":["ST_DWithin"],"reasoning_summary":"ok"}'])
        generator_2 = MockSQLGenerator(responses=['{"sql":"SELECT a.name FROM table_1 a JOIN table_2 b ON ST_DWithin(a.geom,b.geom,10) LIMIT 5","used_tables":["table_1","table_2"],"used_columns":["name","geom"],"used_spatial_functions":["ST_DWithin"],"reasoning_summary":"ok"}'])
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

    def test_validation_mismatches_are_kept_but_unfixed_execution_errors_are_discarded(self):
        library = _load_library(_sample_function_json_payload()[:3], "## spatialsql_pg\nST_Buffer\n")
        database = _make_database(table_count=1)
        invalid_generator = MockSQLGenerator(
            responses=['{"sql":"SELECT ST_Union(t.geom, t.geom) FROM table_1 t LIMIT 5","used_tables":["table_1"],"used_columns":["geom"],"used_spatial_functions":["ST_Union"],"reasoning_summary":"off-constraint but executable"}']
        )
        from src.synthesis.sql.models import SQLExecutionResult
        synth_warning = ConstraintGuidedSQLSynthesizer(
            config=_make_config(),
            function_library=library,
            sql_generator=invalid_generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker([SQLExecutionResult(executed=True, success=True, row_count=1)]),
        )
        kept_warning = synth_warning.synthesize_for_database(database)
        self.assertEqual(len(kept_warning), 1)
        self.assertFalse(kept_warning[0].validation_result["is_valid"])
        self.assertTrue(kept_warning[0].generation_metadata["retained_with_warning"])

        failed_exec_generator = MockSQLGenerator(
            responses=[
                '{"sql":"SELECT ST_Buffer(t.geom, 10) FROM table_1 t","used_tables":["table_1"],"used_columns":["geom"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"ok"}',
                '{"sql":"SELECT ST_Buffer(t.bad_geom, 10) FROM table_1 t","used_tables":["table_1"],"used_columns":["bad_geom"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"broken fix"}',
            ]
        )
        synth_failed_exec = ConstraintGuidedSQLSynthesizer(
            config=_make_config(synthesis={"max_revision_rounds": 0}),
            function_library=library,
            sql_generator=failed_exec_generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker(
                [
                    SQLExecutionResult(executed=True, success=False, error_message="execution failed"),
                    SQLExecutionResult(executed=True, success=False, error_message="execution failed again"),
                ]
            ),
        )
        kept_failed = synth_failed_exec.synthesize_for_database(database)
        self.assertEqual(len(kept_failed), 0)
        self.assertEqual(len(failed_exec_generator.prompts), 2)

    def test_execution_timeout_is_discarded_without_minor_revision(self):
        library = _load_library([_sample_function_json_payload()[1]], "## spatialsql_pg\nST_Buffer\n")
        database = _make_database(table_count=1)
        generator = MockSQLGenerator(
            responses=['{"sql":"SELECT ST_Buffer(t.geom, 10) FROM table_1 t","used_tables":["table_1"],"used_columns":["geom"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"ok"}']
        )
        from src.synthesis.sql.models import SQLExecutionResult

        synthesizer = ConstraintGuidedSQLSynthesizer(
            config=_make_config(),
            function_library=library,
            sql_generator=generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker(
                [SQLExecutionResult(executed=True, success=False, error_message="canceling statement due to statement timeout")]
            ),
        )
        rows = synthesizer.synthesize_for_database(database)
        self.assertEqual(len(rows), 0)
        self.assertEqual(len(generator.prompts), 1)

    def test_obvious_write_operations_are_discarded(self):
        library = _load_library([_sample_function_json_payload()[1]], "## spatialsql_pg\nST_Buffer\n")
        database = _make_database(table_count=1)
        generator = MockSQLGenerator(
            responses=['{"sql":"DELETE FROM table_1 WHERE id = 1","used_tables":["table_1"],"used_columns":["id"],"used_spatial_functions":[],"reasoning_summary":"bad write"}']
        )
        from src.synthesis.sql.models import SQLExecutionResult
        synthesizer = ConstraintGuidedSQLSynthesizer(
            config=_make_config(),
            function_library=library,
            sql_generator=generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker(
                [
                    SQLExecutionResult(
                        executed=False,
                        success=False,
                        error_message="Refused to execute non-read-only SQL.",
                    )
                ]
            ),
        )
        rows = synthesizer.synthesize_for_database(database)
        self.assertEqual(len(rows), 0)
        self.assertEqual(len(generator.prompts), 1)

    def test_synthesizer_run_stats_track_generated_and_retained_counts(self):
        library = _load_library([_sample_function_json_payload()[1]], "## spatialsql_pg\nST_Buffer\n")
        database_1 = _make_database(table_count=1, database_id="nyc_0001")
        database_2 = _make_database(table_count=1, database_id="nyc_0002")
        generator = MockSQLGenerator(
            responses=[
                '{"sql":"SELECT ST_Buffer(t.geom, 10) FROM table_1 t LIMIT 5","used_tables":["table_1"],"used_columns":["geom"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"ok"}',
                '{"sql":"SELECT ST_Buffer(t.geom, 20) FROM table_1 t LIMIT 5","used_tables":["table_1"],"used_columns":["geom"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"fails execution"}',
                '{"sql":"SELECT ST_Buffer(t.bad_geom, 20) FROM table_1 t LIMIT 5","used_tables":["table_1"],"used_columns":["bad_geom"],"used_spatial_functions":["ST_Buffer"],"reasoning_summary":"still broken"}',
            ]
        )
        from src.synthesis.sql.models import SQLExecutionResult

        synthesizer = ConstraintGuidedSQLSynthesizer(
            config=_make_config(),
            function_library=library,
            sql_generator=generator,
            prompt_builder=PromptBuilder({"project_root": Path.cwd()}),
            validator=SQLValidator(library),
            execution_checker=FakeExecutionChecker(
                [
                    SQLExecutionResult(executed=True, success=True, row_count=1),
                    SQLExecutionResult(executed=True, success=False, error_message="execution failed"),
                    SQLExecutionResult(executed=True, success=False, error_message="execution failed again"),
                ]
            ),
        )
        rows = synthesizer.synthesize_all([database_1, database_2])
        stats = synthesizer.get_run_stats()
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats["generated_total"], 2)
        self.assertEqual(stats["retained_total"], 1)
        self.assertEqual(stats["generated_by_difficulty"]["easy"], 2)
        self.assertEqual(stats["retained_by_difficulty"]["easy"], 1)

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
