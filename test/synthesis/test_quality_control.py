import json
import tempfile
import unittest
from pathlib import Path

from src.synthesis.quality import (
    InMemorySchemaRegistry,
    NLSQLSample,
    QualityControlConfig,
    QualityControlDatabaseConfig,
    QualityControlFunctionConfig,
    QualityControlLoggingConfig,
    QualityControlPipeline,
    QualityControlRunConfig,
    SemanticCheckConfig,
    DuplicateDetectionConfig,
    DiversityBalancingConfig,
    BalanceDimensionConfig,
    DatabaseSchema,
    TableSchema,
    StaticDatabaseRegistry,
    load_quality_control_config,
    load_nl_sql_samples,
    write_nl_sql_samples,
)
from src.synthesis.quality.models import ColumnSchema
from src.synthesis.sql.function_library import PostGISFunctionLibrary


class _RejectingJudgeResult:
    passed = False
    warnings = ["judge_warning"]
    reason_codes = ["wrong_entity"]

    def to_dict(self):
        return {
            "passed": False,
            "pass_votes": 0,
            "fail_votes": 1,
            "reason_codes": list(self.reason_codes),
            "warnings": list(self.warnings),
            "rounds": [],
        }


class _RejectingJudge:
    def judge(self, **_kwargs):
        return _RejectingJudgeResult()


def _build_function_library():
    payload = [
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
            "function_id": "st_contains",
            "chapter_info": "reference_relationship",
            "source_file": "reference_relationship.xml",
            "function_definitions": [
                {
                    "function_name": "ST_Contains",
                    "return_type": "boolean",
                    "arguments": ["geometry geom1", "geometry geom2"],
                    "signature_str": "ST_Contains(geometry geom1, geometry geom2)",
                }
            ],
            "description": "Returns true if one geometry contains another.",
            "examples": [{"steps": [{"sql": "SELECT ST_Contains(a.geom, b.geom);"}]}],
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
            "description": "Raster function.",
            "examples": [],
        },
    ]
    markdown = "\n".join(["## spatialsql_pg", "ST_DWithin", "ST_Contains"])
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        json_path = root / "postgis.json"
        md_path = root / "ST_Function.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
        return PostGISFunctionLibrary.load(json_path, md_path, ["raster", "topology"])


def _build_schema(database_id: str = "nyc_0001") -> DatabaseSchema:
    return DatabaseSchema(
        database_id=database_id,
        tables={
            "parks": TableSchema(
                table_name="parks",
                columns={
                    "id": ColumnSchema(column_name="id", column_type="integer"),
                    "name": ColumnSchema(column_name="name", column_type="text"),
                    "geom": ColumnSchema(column_name="geom", column_type="geometry(Point,4326)", spatial_type="geometry", geometry_type="POINT", srid=4326),
                },
            ),
            "neighborhoods": TableSchema(
                table_name="neighborhoods",
                columns={
                    "id": ColumnSchema(column_name="id", column_type="integer"),
                    "name": ColumnSchema(column_name="name", column_type="text"),
                    "geom": ColumnSchema(column_name="geom", column_type="geometry(Polygon,4326)", spatial_type="geometry", geometry_type="POLYGON", srid=4326),
                },
            ),
        },
    )


class MockDatabaseClient:
    def __init__(self, schema: DatabaseSchema, *, row_count: int = 1, preview=None, error: Exception | None = None):
        self.schema = schema
        self.row_count = row_count
        self.preview = preview if preview is not None else [{"name": "central park"}]
        self.error = error
        self.executed_sql: list[str] = []

    def inspect_schema(self) -> DatabaseSchema:
        return self.schema

    def execute_read_only(self, sql: str, *, max_preview_rows: int):
        self.executed_sql.append(sql)
        if self.error is not None:
            raise self.error
        return self.row_count, self.preview[:max_preview_rows]


def _sample(
    sample_id: str,
    *,
    question: str,
    sql: str,
    difficulty: str = "easy",
    used_tables=None,
    used_columns=None,
    used_spatial_functions=None,
    style: str = "factual_lookup",
) -> NLSQLSample:
    return NLSQLSample(
        sample_id=sample_id,
        database_id="nyc_0001",
        question=question,
        sql=sql,
        difficulty_level=difficulty,
        used_tables=list(used_tables or ["parks"]),
        used_columns=list(used_columns or ["name", "geom"]),
        used_spatial_functions=list(used_spatial_functions or ["ST_DWithin"]),
        linguistic_style=style,
        metadata={},
    )


def _config(
    *,
    allow_empty_result: bool = False,
    semantic_mode: str = "strict",
    duplicates: DuplicateDetectionConfig | None = None,
    balancing: DiversityBalancingConfig | None = None,
) -> QualityControlConfig:
    return QualityControlConfig(
        database=QualityControlDatabaseConfig(),
        functions=QualityControlFunctionConfig(postgis_function_json_path="", st_function_markdown_path=""),
        run=QualityControlRunConfig(
            allow_empty_result=allow_empty_result,
            max_result_rows=5,
            prefer_live_schema=True,
        ),
        semantic=SemanticCheckConfig(mode=semantic_mode, debug_mode=False),
        duplicates=duplicates or DuplicateDetectionConfig(),
        balancing=balancing or DiversityBalancingConfig(enabled=False),
        logging=QualityControlLoggingConfig(log_level="INFO"),
    )


class QualityControlTests(unittest.TestCase):
    def setUp(self):
        self.function_library = _build_function_library()
        self.pipeline = QualityControlPipeline(function_library=self.function_library)
        self.schema = _build_schema()
        self.schema_registry = InMemorySchemaRegistry({"nyc_0001": self.schema})

    def test_non_read_only_sql_is_downgraded_to_warning_and_retained(self):
        sample = _sample(
            "s1",
            question="Which parks are within 100 units of the neighborhood?",
            sql="DELETE FROM parks WHERE id = 1",
            used_spatial_functions=[],
            used_columns=["id"],
        )
        registry = StaticDatabaseRegistry({"nyc_0001": MockDatabaseClient(self.schema)})
        retained, report = self.pipeline.run([sample], registry, self.schema_registry, _config())
        self.assertEqual(len(retained), 1)
        self.assertEqual(report.passed_samples, 1)
        warnings = retained[0].metadata["quality_control"]["warnings"]
        self.assertTrue(any("non-read-only" in item for item in warnings))

    def test_unknown_tables_or_columns_are_downgraded_to_warning_and_retained(self):
        sample = _sample(
            "s1",
            question="Which parks are within 100 units of the neighborhood?",
            sql="SELECT x.fake FROM unknown_table x WHERE ST_DWithin(x.fake_geom, x.fake_geom, 100)",
            used_tables=["unknown_table"],
            used_columns=["fake", "fake_geom"],
        )
        registry = StaticDatabaseRegistry({"nyc_0001": MockDatabaseClient(self.schema)})
        retained, _report = self.pipeline.run([sample], registry, self.schema_registry, _config())
        self.assertEqual(len(retained), 1)
        warnings = retained[0].metadata["quality_control"]["warnings"]
        self.assertTrue(any("Unknown tables referenced" in item for item in warnings))
        self.assertTrue(any("Unknown columns referenced" in item for item in warnings))

    def test_disallowed_postgis_function_is_downgraded_to_warning_and_retained(self):
        sample = _sample(
            "s1",
            question="Which parks match the raster operation?",
            sql="SELECT ST_AsRaster(p.geom) FROM parks p",
            used_spatial_functions=["ST_AsRaster"],
        )
        registry = StaticDatabaseRegistry({"nyc_0001": MockDatabaseClient(self.schema)})
        retained, _report = self.pipeline.run([sample], registry, self.schema_registry, _config())
        self.assertEqual(len(retained), 1)
        warnings = retained[0].metadata["quality_control"]["warnings"]
        self.assertTrue(any("Disallowed or unknown PostGIS function" in item for item in warnings))

    def test_pass_executable_select_with_non_empty_results(self):
        sample = _sample(
            "s1",
            question="Which parks are within 100 units of neighborhoods?",
            sql="SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100)",
            used_tables=["parks", "neighborhoods"],
        )
        registry = StaticDatabaseRegistry({"nyc_0001": MockDatabaseClient(self.schema, row_count=2)})
        retained, report = self.pipeline.run([sample], registry, self.schema_registry, _config())
        self.assertEqual(len(retained), 1)
        self.assertEqual(report.passed_samples, 1)

    def test_empty_result_is_downgraded_to_warning_and_retained(self):
        sample = _sample(
            "s1",
            question="Which parks are within 100 units of neighborhoods?",
            sql="SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100)",
            used_tables=["parks", "neighborhoods"],
        )
        registry = StaticDatabaseRegistry({"nyc_0001": MockDatabaseClient(self.schema, row_count=0, preview=[])})
        retained, report = self.pipeline.run([sample], registry, self.schema_registry, _config(allow_empty_result=False))
        self.assertEqual(len(retained), 1)
        self.assertEqual(report.passed_samples, 1)
        quality_control = retained[0].metadata["quality_control"]
        self.assertEqual(quality_control["execution_status"], "empty_result")
        self.assertIn("SQL executed successfully but returned no rows.", quality_control["warnings"])

    def test_detect_exact_and_normalized_duplicate_sql(self):
        samples = [
            _sample("s1", question="Which parks are within 100 units of neighborhoods?", sql="SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100)", used_tables=["parks", "neighborhoods"]),
            _sample("s2", question="Which parks are within 100 units of neighborhoods?", sql="SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100)", used_tables=["parks", "neighborhoods"]),
            _sample("s3", question="Which parks are within 100 units of neighborhoods?", sql="  select  p.name  from parks p join neighborhoods n on st_dwithin(p.geom,n.geom,100)  ", used_tables=["parks", "neighborhoods"]),
        ]
        registry = StaticDatabaseRegistry({"nyc_0001": MockDatabaseClient(self.schema, row_count=1)})
        retained, report = self.pipeline.run([samples[0]], registry, self.schema_registry, _config())
        self.assertEqual(len(retained), 1)
        retained, report = self.pipeline.run(samples, registry, self.schema_registry, _config())
        self.assertEqual(len(retained), 1)
        self.assertEqual(report.duplicate_count, 2)

    def test_detect_near_duplicate_questions(self):
        sql = "SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100)"
        samples = [
            _sample("s1", question="Which parks are within 100 units of neighborhoods?", sql=sql, used_tables=["parks", "neighborhoods"]),
            _sample("s2", question="Which parks are within 100 units from neighborhoods?", sql=sql, used_tables=["parks", "neighborhoods"]),
        ]
        registry = StaticDatabaseRegistry({"nyc_0001": MockDatabaseClient(self.schema, row_count=1)})
        retained, report = self.pipeline.run(samples, registry, self.schema_registry, _config())
        self.assertEqual(len(retained), 1)
        self.assertEqual(report.duplicate_count, 1)

    def test_warn_when_question_misses_aggregation_semantics(self):
        sample = _sample(
            "s1",
            question="Which parks contain themselves?",
            sql="SELECT COUNT(*) FROM parks p WHERE ST_Contains(p.geom, p.geom)",
            used_spatial_functions=["ST_Contains"],
            used_columns=["geom"],
        )
        registry = StaticDatabaseRegistry({"nyc_0001": MockDatabaseClient(self.schema, row_count=1, preview=[{"count": 3}])})
        retained, _report = self.pipeline.run([sample], registry, self.schema_registry, _config(semantic_mode="warning_only"))
        self.assertEqual(len(retained), 1)
        warnings = retained[0].metadata["quality_control"]["warnings"]
        self.assertTrue(any("aggregation semantics" in item for item in warnings))

    def test_warn_when_question_misses_ranking_semantics(self):
        sample = _sample(
            "s1",
            question="Which parks are within 100 units of neighborhoods?",
            sql="SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100) ORDER BY p.name DESC LIMIT 3",
            used_tables=["parks", "neighborhoods"],
        )
        registry = StaticDatabaseRegistry({"nyc_0001": MockDatabaseClient(self.schema, row_count=3)})
        retained, _report = self.pipeline.run([sample], registry, self.schema_registry, _config(semantic_mode="warning_only"))
        self.assertEqual(len(retained), 1)
        warnings = retained[0].metadata["quality_control"]["warnings"]
        self.assertTrue(any("ranking or top-k semantics" in item for item in warnings))

    def test_warn_when_question_misses_distance_threshold(self):
        sample = _sample(
            "s1",
            question="Which parks are near neighborhoods?",
            sql="SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100)",
            used_tables=["parks", "neighborhoods"],
        )
        registry = StaticDatabaseRegistry({"nyc_0001": MockDatabaseClient(self.schema, row_count=1)})
        retained, _report = self.pipeline.run([sample], registry, self.schema_registry, _config(semantic_mode="warning_only"))
        self.assertEqual(len(retained), 1)
        warnings = retained[0].metadata["quality_control"]["warnings"]
        self.assertTrue(any("distance threshold 100" in item for item in warnings))

    def test_self_consistency_judge_rejection_is_recorded_but_not_filtered(self):
        sample = _sample(
            "s1",
            question="Which parks are within 100 units of neighborhoods?",
            sql="SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100)",
            used_tables=["parks", "neighborhoods"],
        )
        pipeline = QualityControlPipeline(
            function_library=self.function_library,
            self_consistency_judge=_RejectingJudge(),
        )
        registry = StaticDatabaseRegistry({"nyc_0001": MockDatabaseClient(self.schema, row_count=1)})
        retained, report = pipeline.run([sample], registry, self.schema_registry, _config())
        self.assertEqual(len(retained), 1)
        self.assertEqual(report.passed_samples, 1)
        self.assertIn("Self-consistency judge rejected the NL-SQL pair.", retained[0].metadata["quality_control"]["errors"])
        self.assertIn("judge:wrong_entity", report.failure_reasons)

    def test_balance_retained_samples(self):
        samples = [
            _sample("s1", question="Which parks are within 100 units of neighborhoods?", sql="SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100)", difficulty="easy", used_tables=["parks", "neighborhoods"], style="factual_lookup"),
            _sample("s2", question="Which parks are within 100 units of neighborhoods?", sql="SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100) LIMIT 2", difficulty="medium", used_tables=["parks", "neighborhoods"], style="ranking_inquiry"),
            _sample("s3", question="Which parks contain themselves?", sql="SELECT COUNT(*) FROM parks p WHERE ST_Contains(p.geom, p.geom)", difficulty="hard", used_spatial_functions=["ST_Contains"], used_columns=["geom"], style="aggregation_inquiry"),
        ]
        registry = StaticDatabaseRegistry({"nyc_0001": MockDatabaseClient(self.schema, row_count=1)})
        balancing = DiversityBalancingConfig(
            enabled=True,
            difficulty=BalanceDimensionConfig(max_per_bucket=1),
            spatial_function=BalanceDimensionConfig(max_per_bucket=1),
            linguistic_style=BalanceDimensionConfig(max_per_bucket=1),
        )
        retained, report = self.pipeline.run(samples, registry, self.schema_registry, _config(semantic_mode="warning_only", balancing=balancing))
        self.assertEqual(len(retained), 2)
        self.assertLessEqual(max(report.distribution_by_difficulty.values()), 1)

    def test_config_and_io_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "quality_control.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        "  input_path: data/in.jsonl",
                        "  output_path: data/out.jsonl",
                        "semantic:",
                        "  mode: warning_only",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_quality_control_config(config_path)
            self.assertEqual(config.semantic.mode, "warning_only")
            self.assertTrue(config.run.input_path.endswith("data/in.jsonl"))

            sample_path = root / "samples.jsonl"
            out_path = root / "out.jsonl"
            sample = _sample(
                "s1",
                question="Which parks are within 100 units of neighborhoods?",
                sql="SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100)",
                used_tables=["parks", "neighborhoods"],
            )
            write_nl_sql_samples(str(sample_path), [sample])
            loaded = load_nl_sql_samples(str(sample_path))
            write_nl_sql_samples(str(out_path), loaded)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].sample_id, "s1")
            self.assertIn("s1", out_path.read_text(encoding="utf-8"))

    def test_writer_preserves_original_synthesized_question_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "synthesized_questions.jsonl"
            output_path = root / "nl2sql.jsonl"
            payload = {
                "question_id": "nyc_0001_0001",
                "sql_id": "nyc_0001_0001",
                "database_id": "nyc_0001",
                "city": "new york",
                "style": "formal",
                "question": "Which parks are within 100 units of neighborhoods?",
                "sql": "SELECT p.name FROM parks p JOIN neighborhoods n ON ST_DWithin(p.geom, n.geom, 100)",
                "reasoning_summary": "Use the spatial join and return the park names.",
                "sql_reasoning_summary": "Use ST_DWithin to relate parks and neighborhoods within 100 units.",
                "spatial_phrases": ["within 100 units of"],
                "source_difficulty_level": "medium",
                "used_tables": ["parks", "neighborhoods"],
                "used_columns": ["name", "geom"],
                "used_spatial_functions": ["ST_DWithin"],
                "spatial_relation_constraints": [{"function_name": "ST_DWithin"}],
                "sql_features": {"tables": ["parks", "neighborhoods"]},
                "prompt": "prompt text",
                "feedback_prompts": [],
                "validation_result": {"is_valid": True, "warnings": []},
                "generation_metadata": {"generator": "question"},
            }
            input_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            samples = load_nl_sql_samples(str(input_path))
            self.assertEqual(len(samples), 1)
            samples[0].metadata = {"quality_control": {"passed": True, "warnings": []}}
            write_nl_sql_samples(str(output_path), samples)
            row = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["question_id"], payload["question_id"])
            self.assertEqual(row["prompt"], payload["prompt"])
            self.assertEqual(row["generation_metadata"], payload["generation_metadata"])
            self.assertEqual(row["sql_features"], payload["sql_features"])
            self.assertEqual(row["sql_reasoning_summary"], payload["sql_reasoning_summary"])
            self.assertEqual(row["metadata"]["quality_control"]["passed"], True)


if __name__ == "__main__":
    unittest.main()
