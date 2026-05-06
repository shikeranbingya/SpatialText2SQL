"""Relation graph construction for canonical spatial tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np

from .embeddings import EmbeddingProvider
from .models import CanonicalSpatialTable
from .text import build_table_text

LOGGER = logging.getLogger(__name__)


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=float)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    return normalized @ normalized.T


@dataclass
class RelationGraphBuilder:
    """Build a weighted table relation graph for one city."""

    embedding_provider: EmbeddingProvider
    target_avg_degree: float = 4.0

    def build_city_graph(
        self,
        tables: Sequence[CanonicalSpatialTable],
    ) -> tuple[Any, dict[str, Any]]:
        ordered_tables = sorted(tables, key=lambda table: table.table_id)
        graph = nx.Graph()
        for table in ordered_tables:
            graph.add_node(
                table.table_id,
                table=table,
                city=table.city,
                table_name=table.table_name,
            )

        num_nodes = len(ordered_tables)
        effective_target = min(max(float(self.target_avg_degree), 0.0), max(num_nodes - 1, 0))
        if num_nodes < 2:
            stats = {
                "num_nodes": num_nodes,
                "num_edges": 0,
                "avg_degree": 0.0,
                "effective_target_avg_degree": effective_target,
            }
            return graph, stats

        texts = [build_table_text(table) for table in ordered_tables]
        embeddings = self.embedding_provider.encode(texts, normalize_embeddings=True)
        if len(embeddings) != num_nodes:
            raise ValueError("Embedding provider returned an unexpected number of vectors.")
        similarities = _cosine_similarity_matrix(embeddings)

        candidate_edges: list[tuple[float, str, str]] = []
        for index, left_table in enumerate(ordered_tables):
            for jndex in range(index + 1, num_nodes):
                right_table = ordered_tables[jndex]
                similarity = float(similarities[index, jndex])
                candidate_edges.append((similarity, left_table.table_id, right_table.table_id))

        candidate_edges.sort(key=lambda item: (-item[0], item[1], item[2]))
        if effective_target > 0:
            for similarity, left_id, right_id in candidate_edges:
                graph.add_edge(left_id, right_id, weight=similarity)
                avg_degree = (2.0 * graph.number_of_edges()) / num_nodes
                if avg_degree >= effective_target:
                    break
        stats = {
            "num_nodes": num_nodes,
            "num_edges": graph.number_of_edges(),
            "avg_degree": (2.0 * graph.number_of_edges() / num_nodes) if num_nodes else 0.0,
            "effective_target_avg_degree": effective_target,
        }
        LOGGER.info(
            "Built relation graph for city=%s with nodes=%s edges=%s avg_degree=%.3f target=%.3f",
            ordered_tables[0].city if ordered_tables else "",
            stats["num_nodes"],
            stats["num_edges"],
            stats["avg_degree"],
            stats["effective_target_avg_degree"],
        )
        return graph, stats
