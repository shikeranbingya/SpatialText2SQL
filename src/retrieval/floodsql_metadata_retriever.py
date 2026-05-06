"""FloodSQL 专属 metadata 检索器。"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sentence_transformers import SentenceTransformer


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_TABLE_TOP_K = {
    "L0": 3,
    "L1": 4,
    "L2": 4,
    "L3": 5,
    "L4": 5,
    "L5": 5,
}


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class _FloodSQLMetadataBase:
    def __init__(self, config: Dict[str, Any], mode: str):
        self.config = config
        self.mode = mode
        self.doc_source = config.get("doc_source")
        self.vector_db_path = config.get("vector_db_path", "data/indexes/vector/floodsql")
        self.embedding_model_name = config.get(
            "embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.table_top_k_by_level = {
            **DEFAULT_TABLE_TOP_K,
            **config.get("table_top_k_by_level", {}),
        }
        self.column_top_k = int(config.get("column_top_k", 5))
        self.metadata: Dict[str, Any] = {}
        self.table_docs: Dict[str, str] = {}
        self.table_tokens: Dict[str, set[str]] = {}
        self.column_docs: Dict[str, List[Dict[str, Any]]] = {}
        self.embedding_model = None
        self.table_embeddings: Dict[str, Sequence[float]] = {}
        self.column_embeddings: Dict[str, Dict[str, Sequence[float]]] = {}

    def build_index(self):
        self._ensure_loaded()
        os.makedirs(self.vector_db_path, exist_ok=True)
        manifest_path = os.path.join(self.vector_db_path, "floodsql_metadata_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": self.mode,
                    "doc_source": self.doc_source,
                    "table_count": len(self.table_docs),
                    "tables": sorted(self.table_docs.keys()),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load_documents(self):
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self.metadata:
            return
        if not self.doc_source or not os.path.exists(self.doc_source):
            raise FileNotFoundError(f"FloodSQL metadata 文件不存在: {self.doc_source}")

        with open(self.doc_source, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        for table_name, info in self.metadata.items():
            if table_name == "_global" or not isinstance(info, dict):
                continue
            table_text = self._build_table_text(table_name, info)
            self.table_docs[table_name] = table_text
            self.table_tokens[table_name] = set(_tokenize(table_text))

            column_entries: List[Dict[str, Any]] = []
            for column in info.get("schema", []):
                column_name = column.get("column_name", "")
                description = column.get("description", "")
                column_text = f"{table_name}.{column_name}: {description}"
                column_entries.append(
                    {
                        "column_name": column_name,
                        "description": description,
                        "text": column_text,
                        "tokens": set(_tokenize(column_text)),
                    }
                )
            self.column_docs[table_name] = column_entries

        if self.mode == "semantic":
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.table_embeddings = {
                table_name: self.embedding_model.encode(text)
                for table_name, text in self.table_docs.items()
            }
            self.column_embeddings = {
                table_name: {
                    entry["column_name"]: self.embedding_model.encode(entry["text"])
                    for entry in entries
                }
                for table_name, entries in self.column_docs.items()
            }

    @staticmethod
    def _build_table_text(table_name: str, info: Dict[str, Any]) -> str:
        parts = [table_name]
        meta_text = info.get("_meta")
        if meta_text:
            parts.append(meta_text)
        for column in info.get("schema", []):
            column_name = column.get("column_name", "")
            description = column.get("description", "")
            if column_name or description:
                parts.append(f"{column_name}: {description}")
        return "\n".join(parts)

    def _get_level(self, item: Optional[Dict[str, Any]]) -> str:
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        return metadata.get("level") or item.get("level") or "L5"

    def _get_table_top_k(self, item: Optional[Dict[str, Any]]) -> int:
        return int(self.table_top_k_by_level.get(self._get_level(item), 5))

    def _score_text(
        self,
        question: str,
        tokens: set[str],
        embedding: Optional[Sequence[float]] = None,
        question_embedding: Optional[Sequence[float]] = None,
    ) -> float:
        if (
            self.mode == "semantic"
            and self.embedding_model is not None
            and embedding is not None
            and question_embedding is not None
        ):
            return _cosine_similarity(question_embedding, embedding)

        question_tokens = set(_tokenize(question))
        if not question_tokens:
            return 0.0
        overlap = len(question_tokens & tokens)
        coverage = overlap / len(question_tokens)
        return coverage + overlap * 0.001

    def retrieve(self, question: str, item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._ensure_loaded()
        question_embedding = None
        if self.mode == "semantic" and self.embedding_model is not None:
            question_embedding = self.embedding_model.encode(question)
        table_scores: List[Tuple[str, float]] = []
        for table_name, text in self.table_docs.items():
            del text
            score = self._score_text(
                question,
                self.table_tokens[table_name],
                self.table_embeddings.get(table_name),
                question_embedding=question_embedding,
            )
            table_scores.append((table_name, score))

        table_scores.sort(key=lambda pair: pair[1], reverse=True)
        chosen_tables = [table for table, _score in table_scores[: self._get_table_top_k(item)]]

        chosen_columns: Dict[str, List[Tuple[str, float, str]]] = {}
        for table_name in chosen_tables:
            column_scores: List[Tuple[str, float, str]] = []
            for entry in self.column_docs.get(table_name, []):
                embedding = self.column_embeddings.get(table_name, {}).get(entry["column_name"])
                score = self._score_text(
                    question,
                    entry["tokens"],
                    embedding,
                    question_embedding=question_embedding,
                )
                column_scores.append((entry["column_name"], score, entry["description"]))
            column_scores.sort(key=lambda pair: pair[1], reverse=True)
            chosen_columns[table_name] = column_scores[: self.column_top_k]

        return {
            "mode": self.mode,
            "selected_tables": chosen_tables,
            "selected_columns": chosen_columns,
            "join_rules": self.metadata.get("_global", {}).get("join_rules", {}),
            "rules": self.metadata.get("_global", {}).get("rules", {}),
            "notes": self.metadata.get("_global", {}).get("notes", []),
            "triple_table_notes": self.metadata.get("_global", {}).get("triple_table_notes", []),
            "spatial_function_notes": self.metadata.get("_global", {}).get("spatial_function_notes", []),
            "basic_function_notes": self.metadata.get("_global", {}).get("basic_function_notes", []),
        }

    def format_context(self, retrieved_docs: Dict[str, Any]) -> str:
        if not retrieved_docs:
            return ""

        lines = [f"## FloodSQL Metadata ({retrieved_docs.get('mode', self.mode)})"]
        selected_tables = retrieved_docs.get("selected_tables", [])
        if selected_tables:
            lines.append("[TABLES SELECTED]")
            for table_name in selected_tables:
                lines.append(f"- {table_name}")

        selected_columns = retrieved_docs.get("selected_columns", {})
        if selected_columns:
            lines.append("\n[COLUMNS SELECTED]")
            for table_name, columns in selected_columns.items():
                for column_name, _score, description in columns:
                    lines.append(f"- {table_name}.{column_name}: {description}")

        join_rules = retrieved_docs.get("join_rules", {})
        for title, rules in (
            ("[JOIN RULES: KEY-BASED DIRECT]", join_rules.get("key_based", {}).get("direct", [])),
            ("[JOIN RULES: KEY-BASED CONCAT]", join_rules.get("key_based", {}).get("concat", [])),
            ("[JOIN RULES: SPATIAL POINT-POLYGON]", join_rules.get("spatial", {}).get("point_polygon", [])),
            ("[JOIN RULES: SPATIAL POLYGON-POLYGON]", join_rules.get("spatial", {}).get("polygon_polygon", [])),
        ):
            if not rules:
                continue
            lines.append(f"\n{title}")
            for item in rules:
                pair = item.get("pair", [])
                if len(pair) == 2:
                    lines.append(f"- {pair[0]} <-> {pair[1]}")

        for title, items in (
            ("[RULES]", retrieved_docs.get("rules", {}).items()),
            ("[NOTES]", retrieved_docs.get("notes", [])),
            ("[TRIPLE-TABLE-NOTES]", retrieved_docs.get("triple_table_notes", [])),
            ("[SPATIAL-NOTES]", retrieved_docs.get("spatial_function_notes", [])),
            ("[BASIC-FUNCTION-NOTES]", retrieved_docs.get("basic_function_notes", [])),
        ):
            items = list(items)
            if not items:
                continue
            lines.append(f"\n{title}")
            if title == "[RULES]":
                for key, value in items:
                    lines.append(f"- {key}: {value}")
            else:
                for value in items:
                    lines.append(f"- {value}")

        return "\n".join(lines)


class FloodSQLMetadataRAGRetriever(_FloodSQLMetadataBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, mode="semantic")


class FloodSQLMetadataKeywordSearcher(_FloodSQLMetadataBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, mode="keyword")

    def search(self, question: str, item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.retrieve(question, item=item)
