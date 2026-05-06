#!/usr/bin/env python3
"""
Validate the SpatialSQL integration without requiring live databases.

1) Regression-check the original spatial_qa flow.
2) Verify the SpatialSQL loader and SQL dialect adapter.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)


def test_original_flow():
    """Check that the original spatial_qa flow is still wired correctly."""
    from src.datasets import DataLoaderFactory

    config_path = os.path.join(REPO_ROOT, "config", "dataset_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    default_name = config.get("default_dataset", "spatial_qa")
    assert default_name == "spatial_qa", "default_dataset must remain spatial_qa"
    dataset_info = config["datasets"].get(default_name)
    assert dataset_info is not None
    loader = DataLoaderFactory.create(dataset_info["loader_class"], dataset_info)
    info = loader.get_dataset_info()
    assert info["name"] == "spatial_qa"
    assert "grouping_fields" in info and "level" in info["grouping_fields"]
    print("[OK] Original spatial_qa configuration is intact and SpatialQALoader is available.")
    return True


def test_spatialsql_loader_and_adapter():
    """Check SpatialSQLLoader parsing and the SQL dialect adapter."""
    from src.datasets.loaders import SpatialSQLLoader
    from src.sql import convert_spatialite_to_postgis

    config = {
        "data_path": "",
        "dataset_versions": ["dataset1"],
        "domains": ["ada"],
    }
    loader = SpatialSQLLoader(config)
    raw = loader.load_raw_data(tempfile.mkdtemp())
    assert raw == [], "An empty directory should return an empty list."
    extracted = loader.extract_questions_and_sqls([])
    assert extracted == []

    # Stub QA block
    stub_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(stub_dir, "dataset1", "ada"), exist_ok=True)
    with open(os.path.join(stub_dir, "dataset1", "ada", "QA-ada-stub.txt"), "w", encoding="utf-8") as f:
        f.write("label:S\nquestion: What is the border length?\nSQL: Select GLength(Intersection(a.Shape,b.Shape),1);\nEval: Select GLength(Intersection(a.Shape,b.Shape),1);\nid: stub1\n\n")
    raw = loader.load_raw_data(stub_dir)
    assert len(raw) >= 1
    extracted = loader.extract_questions_and_sqls(raw)
    assert len(extracted) >= 1
    assert "gold_sql" in extracted[0] and "gold_sql_candidates" in extracted[0]
    assert "metadata" in extracted[0] and "split" in extracted[0]["metadata"]

    # Dialect conversion
    sql = "Select GLength(Intersection(a.Shape, b.Shape),1) from t a, t b Where Intersects(a.Shape, b.Shape)=1;"
    converted, issues = convert_spatialite_to_postgis(sql)
    assert "ST_Length" in converted and "ST_Intersection" in converted and "ST_Intersects" in converted
    assert "shape" in converted.lower()
    print("[OK] SpatialSQLLoader and sql_dialect_adapter behave as expected.")
    return True


def test_evaluator_multigold():
    """Check that the evaluator accepts multi-gold candidates."""
    from src.evaluation import Evaluator

    eval_config = {"evaluation": {"timeout": 60}}
    # No live DB is required here; we only verify the interface behavior.
    evaluator = Evaluator(db_config={"host": "127.0.0.1", "port": 5432, "database": "test", "user": "u", "password": "p"}, eval_config=eval_config)
    pred_sql = "SELECT 1;"
    gold_sql = "SELECT 1;"
    info = evaluator._execution_accuracy(pred_sql, gold_sql, gold_sql_candidates=None)
    assert "correct" in info and "error_type" in info
    info2 = evaluator._execution_accuracy(pred_sql, gold_sql, gold_sql_candidates=["SELECT 1;"])
    assert "correct" in info2
    print("[OK] Evaluator multi-gold candidate support is wired correctly.")
    return True


def main():
    print("SpatialSQL adaptation validation (no sdbdatasets or migrated PostgreSQL required)\n")
    try:
        test_original_flow()
        test_spatialsql_loader_and_adapter()
        test_evaluator_multigold()
        print("\nAll checks passed. The original flow is intact and the spatialsql_pg extension is available.")
        return 0
    except Exception as e:
        print(f"\nValidation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
