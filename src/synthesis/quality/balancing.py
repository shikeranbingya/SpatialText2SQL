"""Diversity balancing for retained NL-SQL samples."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from .config import BalanceDimensionConfig, DiversityBalancingConfig
from .models import NLSQLSample


def _non_empty_buckets(values: Sequence[str]) -> list[str]:
    return [value for value in values if value]


def _build_caps(
    total_samples: int,
    buckets: Sequence[str],
    config: BalanceDimensionConfig,
) -> dict[str, int]:
    unique_buckets = sorted({bucket for bucket in buckets if bucket})
    caps: dict[str, int] = {}
    if config.max_per_bucket > 0:
        for bucket in unique_buckets:
            caps[bucket] = config.max_per_bucket
    if config.target_distribution:
        total_weight = sum(weight for weight in config.target_distribution.values() if weight > 0)
        if total_weight > 0:
            for bucket, weight in config.target_distribution.items():
                if weight <= 0:
                    continue
                quota = max(1, int(math.ceil(total_samples * (weight / total_weight))))
                caps[bucket] = min(caps.get(bucket, quota), quota) if bucket in caps else quota
    return caps


def _apply_dimension_balance(
    samples: Sequence[NLSQLSample],
    *,
    getter: Callable[[NLSQLSample], list[str]],
    config: BalanceDimensionConfig,
) -> tuple[list[NLSQLSample], list[str]]:
    if config.max_per_bucket <= 0 and not config.target_distribution:
        return list(samples), []
    all_buckets = [bucket for sample in samples for bucket in getter(sample)]
    caps = _build_caps(len(samples), all_buckets, config)
    if not caps:
        return list(samples), []
    counts: Counter[str] = Counter()
    kept: list[NLSQLSample] = []
    dropped: list[str] = []
    for sample in samples:
        buckets = _non_empty_buckets(getter(sample))
        if not buckets:
            kept.append(sample)
            continue
        if any(counts[bucket] >= caps.get(bucket, float("inf")) for bucket in buckets):
            dropped.append(sample.sample_id)
            continue
        kept.append(sample)
        for bucket in buckets:
            counts[bucket] += 1
    return kept, dropped


@dataclass
class DiversityBalancer:
    config: DiversityBalancingConfig

    def run(self, samples: Sequence[NLSQLSample]) -> tuple[list[NLSQLSample], list[str]]:
        if not self.config.enabled:
            return list(samples), []
        current = list(samples)
        dropped_ids: list[str] = []
        current, dropped = _apply_dimension_balance(
            current,
            getter=lambda sample: [sample.difficulty_level],
            config=self.config.difficulty,
        )
        dropped_ids.extend(dropped)
        current, dropped = _apply_dimension_balance(
            current,
            getter=lambda sample: sample.used_spatial_functions,
            config=self.config.spatial_function,
        )
        dropped_ids.extend(dropped)
        current, dropped = _apply_dimension_balance(
            current,
            getter=lambda sample: [sample.linguistic_style],
            config=self.config.linguistic_style,
        )
        dropped_ids.extend(dropped)
        return current, dropped_ids

