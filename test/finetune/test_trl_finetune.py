import json
import tempfile
import unittest
from pathlib import Path

from src.finetune.config import DEFAULT_TRL_FINETUNE_CONFIG_PATH
from src.finetune.config import FinetuneDBConfig, FinetuneDataConfig, FinetuneModelConfig
from src.finetune.dataset import SpatialText2SQLDatasetBuilder
from src.finetune.models import RawFinetuneSample
from src.finetune.prompting import FinetunePromptRenderer


class FakeMetadataProvider:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def load_database_metadata(self, request):
        self.calls.append((request.database_id, list(request.selected_table_names)))
        return self.payload


class TRLFinetuneTests(unittest.TestCase):
    def test_default_finetune_config_path_matches_repo_config(self):
        self.assertTrue(str(DEFAULT_TRL_FINETUNE_CONFIG_PATH).endswith("config/finetune.yaml"))

    def test_default_finetune_model_matches_repo_default(self):
        self.assertEqual(FinetuneModelConfig().model_name_or_path, "Qwen/Qwen2.5-Coder-7B-Instruct")

    def test_raw_finetune_sample_accepts_nl2sql_metadata(self):
        row = RawFinetuneSample.from_dict(
            {
                "question_id": "nyc_0001_0001",
                "database_id": "nyc_0001",
                "city": "new york",
                "question": "Which parks intersect schools?",
                "sql": "SELECT p.name FROM parks p JOIN schools s ON ST_Intersects(p.geom, s.geom)",
                "source_difficulty_level": "medium",
                "used_tables": ["parks", "schools"],
                "used_columns": ["name", "geom"],
                "used_spatial_functions": ["ST_Intersects"],
                "sql_features": {"tables": ["parks", "schools"]},
                "metadata": {
                    "quality_control": {"passed": True},
                    "database_context": {"tables": []},
                },
            }
        )
        self.assertEqual(row.difficulty, "medium")
        self.assertIn("database_context", row.metadata)

    def test_prompt_renderer_includes_required_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "prompt.txt"
            template_path.write_text(
                "\n".join(
                    [
                        "## Task Description",
                        "{{task_description}}",
                        "## Schema",
                        "{{schema_block}}",
                        "## Spatial Field Metadata",
                        "{{spatial_field_block}}",
                        "## Representative Values",
                        "{{representative_values_block}}",
                        "## Question",
                        "{{question_block}}",
                    ]
                ),
                encoding="utf-8",
            )
            renderer = FinetunePromptRenderer(
                template_path=template_path,
                task_description="Translate the question to SQL.",
                max_representative_rows=3,
            )
            prompt = renderer.render_prompt(
                question="Which parks intersect schools?",
                schema_lines=["- parks(id integer, geom geometry(Point,4326))"],
                spatial_lines=["- parks.geom (type=geometry(Point,4326), family=geometry, geometry_type=POINT, srid=4326)"],
                representative_values={"parks": [{"id": 1, "geom": "POINT"}]},
            )
        self.assertIn("## Task Description", prompt)
        self.assertIn("## Schema", prompt)
        self.assertIn("## Spatial Field Metadata", prompt)
        self.assertIn("## Representative Values", prompt)
        self.assertIn("## Question", prompt)
        self.assertIn("Which parks intersect schools?", prompt)

    def test_dataset_builder_normalizes_question_id_and_difficulty(self):
        runtime_metadata = {
            "tables": [
                {
                    "table_name": "table_1",
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
                    "representative_values": {"name": ["alpha"], "geom": ["POINT (0 0)"]},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "prompt.txt"
            template_path.write_text(
                "## Task Description\n{{task_description}}\n## Schema\n{{schema_block}}\n## Spatial Field Metadata\n{{spatial_field_block}}\n## Representative Values\n{{representative_values_block}}\n## Question\n{{question_block}}\n",
                encoding="utf-8",
            )
            builder = SpatialText2SQLDatasetBuilder(
                db_config=FinetuneDBConfig(),
                data_config=FinetuneDataConfig(
                    input_path="",
                    prepared_output_path="",
                    prompt_template_path=str(template_path),
                    task_description="Translate to SQL.",
                    question_id_start=0,
                    max_representative_rows=3,
                ),
                metadata_provider=FakeMetadataProvider(runtime_metadata),
            )
            raw = RawFinetuneSample(
                question_id="nyc_0001_q_001",
                database_id="nyc_0001",
                city="nyc",
                sql="SELECT name FROM table_1 LIMIT 5",
                question="Which names should be returned?",
                difficulty="medium",
                used_tables=["table_1"],
                used_columns=["name"],
                used_spatial_functions=["ST_Buffer"],
                sql_features={"limit": 5},
            )
            prepared = builder.prepare_samples([raw])
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0].question_id, 0)
        self.assertEqual(prepared[0].difficulty, "medium")
        self.assertEqual(prepared[0].completion, "SELECT name FROM table_1 LIMIT 5")
        self.assertIn("## Schema", prepared[0].prompt)
        self.assertIn('"geom": "POINT"', prepared[0].prompt)

    def test_dataset_builder_prefers_embedded_nl2sql_metadata_over_live_db(self):
        runtime_metadata = {
            "tables": [
                {
                    "table_name": "parks",
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
                    "representative_values": [{"id": 1, "name": "alpha", "geom": "POINT (0 0)"}],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "prompt.txt"
            template_path.write_text(
                "## Task Description\n{{task_description}}\n## Schema\n{{schema_block}}\n## Spatial Field Metadata\n{{spatial_field_block}}\n## Representative Values\n{{representative_values_block}}\n## Question\n{{question_block}}\n",
                encoding="utf-8",
            )
            provider = FakeMetadataProvider({"tables": []})
            builder = SpatialText2SQLDatasetBuilder(
                db_config=FinetuneDBConfig(),
                data_config=FinetuneDataConfig(
                    input_path="",
                    prepared_output_path="",
                    prompt_template_path=str(template_path),
                    task_description="Translate to SQL.",
                    question_id_start=0,
                    max_representative_rows=3,
                ),
                metadata_provider=provider,
            )
            raw = RawFinetuneSample.from_dict(
                {
                    "question_id": "nyc_0001_0001",
                    "database_id": "nyc_0001",
                    "city": "new york",
                    "question": "Which park names should be returned?",
                    "sql": "SELECT name FROM parks LIMIT 5",
                    "source_difficulty_level": "easy",
                    "used_tables": ["parks"],
                    "used_columns": ["name"],
                    "metadata": {
                        "database_context": runtime_metadata,
                        "quality_control": {"passed": True},
                    },
                }
            )
            prepared = builder.prepare_samples([raw])
        self.assertEqual(len(prepared), 1)
        self.assertEqual(provider.calls, [])
        self.assertIn("- parks(id integer, name text, geom geometry(Point,4326))", prepared[0].prompt)


if __name__ == "__main__":
    unittest.main()
