"""Sample-weighted loss aggregation for batches and streams.

Interview prompt: aggregate means correctly when batch sizes differ, and support
merging independent accumulators without storing every observation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamingWeightedMean:
    weighted_sum: float = 0.0
    total_weight: float = 0.0

    def update(self, value: float, *, weight: float = 1.0) -> None:
        """Add one value or a batch mean with its sample count as the weight."""
        if weight < 0:
            raise ValueError("weight must be non-negative")
        self.weighted_sum += float(value) * float(weight)
        self.total_weight += float(weight)

    def merge(self, other: "StreamingWeightedMean") -> None:
        """Combine state from another shard or worker."""
        self.weighted_sum += other.weighted_sum
        self.total_weight += other.total_weight

    @property
    def value(self) -> float:
        if self.total_weight == 0:
            raise ValueError("mean is undefined before a positive-weight update")
        return self.weighted_sum / self.total_weight

