import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.finetune.config import DEFAULT_TRL_FINETUNE_CONFIG_PATH
from src.finetune.config import (
    FinetuneDataConfig,
    FinetuneModelConfig,
    FinetuneRuntimeConfig,
    load_trl_finetune_config,
    override_trl_finetune_config,
)
from src.finetune.cli import _apply_runtime_environment, _build_accelerate_command, _effective_num_processes
from src.finetune.dataset import SpatialText2SQLDatasetBuilder
from src.finetune.formatter import NL2SQLAlpacaFormatter
from src.finetune.models import RawFinetuneSample
from src.finetune.prompting import FinetunePromptRenderer


class TRLFinetuneTests(unittest.TestCase):
    def test_default_finetune_config_path_matches_repo_config(self):
        self.assertTrue(str(DEFAULT_TRL_FINETUNE_CONFIG_PATH).endswith("config/finetune.yaml"))

    def test_default_finetune_model_matches_repo_default(self):
        self.assertEqual(FinetuneModelConfig().model_name_or_path, "Qwen/Qwen2.5-Coder-7B-Instruct")

    def test_runtime_gpu_indices_override_sets_visible_devices(self):
        config = override_trl_finetune_config(
            load_trl_finetune_config(),
            runtime={"nvidia_gpu_indices": [2, 5]},
        )
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        original_nvidia = os.environ.get("NVIDIA_VISIBLE_DEVICES")
        try:
            _apply_runtime_environment(config)
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "2,5")
            self.assertEqual(os.environ.get("NVIDIA_VISIBLE_DEVICES"), "2,5")
        finally:
            if original_cuda is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda
            if original_nvidia is None:
                os.environ.pop("NVIDIA_VISIBLE_DEVICES", None)
            else:
                os.environ["NVIDIA_VISIBLE_DEVICES"] = original_nvidia

    def test_default_runtime_uses_all_eight_gpus_with_accelerate(self):
        runtime = FinetuneRuntimeConfig()
        self.assertEqual(runtime.nvidia_gpu_indices, list(range(8)))
        self.assertEqual(runtime.distributed_backend, "accelerate")

    def test_accelerate_command_uses_alpaca_input_and_num_processes(self):
        config = override_trl_finetune_config(
            load_trl_finetune_config(),
            runtime={"nvidia_gpu_indices": [0, 1], "num_processes": 2},
            training={"deepspeed_config_path": "configs/ds_zero2.json"},
        )
        args = SimpleNamespace(config="config/finetune.yaml")
        command = _build_accelerate_command(config, args)
        self.assertIn("accelerate.commands.launch", command)
        self.assertIn("--alpaca-input", command)
        self.assertIn(config.data.alpaca_output_path, command)
        self.assertIn("--num_processes", command)
        self.assertIn("2", command)
        self.assertIn("--nvidia-gpu-indices", command)
        self.assertIn("0,1", command)
        self.assertIn("--deepspeed-config-path", command)

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
        renderer = FinetunePromptRenderer(
            task_description="Translate the question to SQL.",
            max_representative_rows=3,
        )
        instruction = renderer.render_instruction()
        input_text = renderer.render_input(
            question="Which parks intersect schools?",
            schema_lines=["- parks(id integer, geom geometry(Point,4326))"],
            representative_values={"parks": [{"id": 1, "geom": "POINT"}]},
        )
        prompt = renderer.compose_prompt(instruction, input_text)
        self.assertIn("## Task Description", instruction)
        self.assertIn("## Response Requirements", instruction)
        self.assertIn("## Schema", input_text)
        self.assertIn("## Representative Values", input_text)
        self.assertIn("## Question", input_text)
        self.assertNotIn("## Spatial Field Metadata", prompt)
        self.assertIn("Which parks intersect schools?", prompt)

    def test_alpaca_formatter_splits_instruction_input_and_output(self):
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
        formatter = NL2SQLAlpacaFormatter(
            data_config=FinetuneDataConfig(
                input_path="",
                alpaca_output_path="",
                task_description="Translate to SQL.",
                question_id_start=0,
                max_representative_rows=3,
            ),
        )
        raw = RawFinetuneSample.from_dict(
            {
                "question_id": "nyc_0001_0001",
                "database_id": "nyc_0001",
                "city": "new york",
                "question": "Which park names should be returned?",
                "sql": "SELECT name FROM parks LIMIT 5",
                "source_difficulty_level": "easy",
                "sql_reasoning_summary": "Use the parks table and return the name column.",
                "used_tables": ["parks"],
                "used_columns": ["name"],
                "metadata": {
                    "database_context": runtime_metadata,
                    "quality_control": {"passed": True},
                },
            }
        )
        rows = formatter.format_samples([raw])
        self.assertEqual(len(rows), 1)
        self.assertIn("## Task Description", rows[0].instruction)
        self.assertIn("## Schema", rows[0].input_text)
        self.assertNotIn("## Spatial Field Metadata", rows[0].input_text)
        self.assertIn("Use the parks table and return the name column.", rows[0].output_text)
        self.assertIn("```sql", rows[0].output_text)
        self.assertIn("SELECT name FROM parks LIMIT 5", rows[0].output_text)

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
        builder = SpatialText2SQLDatasetBuilder(
            data_config=FinetuneDataConfig(
                input_path="",
                alpaca_output_path="",
                task_description="Translate to SQL.",
                question_id_start=0,
                max_representative_rows=3,
            ),
        )
        raw = RawFinetuneSample(
            question_id="nyc_0001_q_001",
            database_id="nyc_0001",
            city="nyc",
            sql="SELECT name FROM table_1 LIMIT 5",
            question="Which names should be returned?",
            difficulty="medium",
            instruction="Do the task.",
            input_text="## Schema\n- table_1(id integer, name text, geom geometry(Point,4326))\n\n## Representative Values\n{}\n\n## Question\nWhich names should be returned?",
            output_text="Return the names.\n\n```sql\nSELECT name FROM table_1 LIMIT 5\n```",
            sql_reasoning_summary="Return the names.",
            used_tables=["table_1"],
            used_columns=["name"],
            used_spatial_functions=["ST_Buffer"],
            sql_features={"limit": 5},
            metadata={"database_context": runtime_metadata},
        )
        prepared = builder.prepare_samples([raw])
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0].question_id, 0)
        self.assertEqual(prepared[0].difficulty, "medium")
        self.assertIn("```sql", prepared[0].completion)
        self.assertEqual(prepared[0].sql_reasoning_summary, "Return the names.")
        self.assertEqual(prepared[0].instruction, "Do the task.")
        self.assertIn("## Schema", prepared[0].input_text)
        self.assertIn("## Representative Values", prepared[0].prompt)
        self.assertNotIn("## Spatial Field Metadata", prepared[0].prompt)

    def test_dataset_builder_uses_embedded_nl2sql_metadata_only(self):
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
        builder = SpatialText2SQLDatasetBuilder(
            data_config=FinetuneDataConfig(
                input_path="",
                alpaca_output_path="",
                task_description="Translate to SQL.",
                question_id_start=0,
                max_representative_rows=3,
            ),
        )
        raw = RawFinetuneSample.from_dict(
            {
                "question_id": "nyc_0001_0001",
                "database_id": "nyc_0001",
                "city": "new york",
                "question": "Which park names should be returned?",
                "sql": "SELECT name FROM parks LIMIT 5",
                "source_difficulty_level": "easy",
                "sql_reasoning_summary": "Use the parks table and return the name column.",
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
        self.assertIn("- parks(id integer, name text, geom geometry(Point,4326))", prepared[0].prompt)
        self.assertNotIn("## Spatial Field Metadata", prepared[0].prompt)
        self.assertIn("```sql", prepared[0].completion)

    def test_dataset_builder_does_not_fallback_to_database_when_metadata_missing(self):
        builder = SpatialText2SQLDatasetBuilder(
            data_config=FinetuneDataConfig(
                input_path="",
                alpaca_output_path="",
                task_description="Translate to SQL.",
                question_id_start=0,
                max_representative_rows=3,
            ),
        )
        raw = RawFinetuneSample.from_dict(
            {
                "question_id": "nyc_0001_0002",
                "database_id": "nyc_0001",
                "question": "Which park names should be returned?",
                "sql": "SELECT name FROM parks LIMIT 5",
                "source_difficulty_level": "easy",
                "sql_reasoning_summary": "Use the parks table and return the name column.",
                "used_tables": ["parks"],
                "used_columns": ["name"],
                "metadata": {"quality_control": {"passed": True}},
            }
        )
        prepared = builder.prepare_samples([raw])
        self.assertEqual(len(prepared), 1)
        self.assertIn("No schema available.", prepared[0].prompt)
        self.assertIn("{}", prepared[0].prompt)
        self.assertIn("```sql", prepared[0].completion)


if __name__ == "__main__":
    unittest.main()
