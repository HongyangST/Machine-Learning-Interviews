"""Fit/transform preprocessing for missing numeric and unseen categorical data.

Interview prompt: learn imputation and category mappings only from training data;
map later unseen categories to an explicit unknown bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class TabularPreprocessor:
    numeric_medians: np.ndarray
    category_to_index: dict[str, int]

    @classmethod
    def fit(
        cls, numeric_features: np.ndarray, categorical_values: Sequence[str | None]
    ) -> "TabularPreprocessor":
        numeric = np.asarray(numeric_features, dtype=np.float64)
        if numeric.ndim != 2 or numeric.shape[0] != len(categorical_values):
            raise ValueError("numeric and categorical rows must match")
        medians = np.nanmedian(numeric, axis=0)
        if np.any(np.isnan(medians)):
            raise ValueError("every numeric feature needs at least one observed value")
        vocabulary = sorted({value for value in categorical_values if value is not None})
        mapping = {value: index + 1 for index, value in enumerate(vocabulary)}
        return cls(medians, mapping)

    def transform(
        self, numeric_features: np.ndarray, categorical_values: Sequence[str | None]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Impute numeric NaNs and encode missing/unseen categories as zero."""
        numeric = np.asarray(numeric_features, dtype=np.float64).copy()
        if numeric.ndim != 2 or numeric.shape != (len(categorical_values), len(self.numeric_medians)):
            raise ValueError("input shape does not match fitted preprocessing state")
        rows, columns = np.where(np.isnan(numeric))
        numeric[rows, columns] = self.numeric_medians[columns]
        categorical = np.array(
            [self.category_to_index.get(value, 0) for value in categorical_values],
            dtype=np.int64,
        )
        return numeric, categorical

