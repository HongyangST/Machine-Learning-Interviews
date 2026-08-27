"""Tests for additional classic ML and data-processing answers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from problems.classic_ml.categorical_preprocessing import TabularPreprocessor  # noqa: E402
from problems.classic_ml.dataset_split import (  # noqa: E402
    stratified_train_validation_test_indices,
    train_validation_test_indices,
)
from problems.classic_ml.feature_standardization import (  # noqa: E402
    fit_standardizer,
    transform_standardized,
)
from problems.classic_ml.gradient_boosting_step import (  # noqa: E402
    apply_stump,
    fit_residual_stump,
)
from problems.classic_ml.matrix_factorization import matrix_factorization_sgd  # noqa: E402
from problems.classic_ml.multiclass_metrics import (  # noqa: E402
    confusion_matrix,
    macro_precision_recall_f1,
)
from problems.classic_ml.multinomial_naive_bayes import (  # noqa: E402
    fit_multinomial_naive_bayes,
    predict_multinomial_naive_bayes,
)
from problems.classic_ml.sampling_strategies import (  # noqa: E402
    sample_indices,
    stratified_sample_indices,
)
from problems.classic_ml.streaming_weighted_mean import StreamingWeightedMean  # noqa: E402


class AdditionalClassicAnswersTest(unittest.TestCase):
    def test_multiclass_metrics(self):
        matrix = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1])
        np.testing.assert_array_equal(matrix, [[1, 1], [0, 2]])
        metrics = macro_precision_recall_f1(matrix)
        self.assertAlmostEqual(metrics["precision"], (1.0 + 2 / 3) / 2)

    def test_multinomial_naive_bayes(self):
        model = fit_multinomial_naive_bayes(
            ["good bright", "good fun", "bad dark", "bad dull"], np.array([1, 1, 0, 0])
        )
        predictions = predict_multinomial_naive_bayes(["bright good", "dark bad"], *model)
        np.testing.assert_array_equal(predictions, [1, 0])

    def test_matrix_factorization_reconstructs_observations(self):
        ratings = np.array([[5.0, 3.0], [4.0, np.nan], [1.0, 1.0]])
        users, items = matrix_factorization_sgd(ratings, 2, epochs=1_500, seed=2)
        reconstructed = users @ items.T
        observed = np.isfinite(ratings)
        self.assertLess(float(np.sqrt(np.mean((ratings[observed] - reconstructed[observed]) ** 2))), 0.25)

    def test_gradient_boosting_residual_stump(self):
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        stump = fit_residual_stump(x, [0.0, 0.0, 2.0, 2.0], np.zeros(4))
        np.testing.assert_allclose(apply_stump(x, stump), [0.0, 0.0, 2.0, 2.0])

    def test_dataset_splits_are_disjoint_and_complete(self):
        train, validation, test = train_validation_test_indices(20, seed=4)
        combined = np.concatenate([train, validation, test])
        self.assertEqual(len(np.unique(combined)), 20)
        stratified = stratified_train_validation_test_indices(np.repeat([0, 1], 10), seed=4)
        self.assertEqual(sum(map(len, stratified)), 20)

    def test_standardizer_reuses_training_statistics(self):
        mean, scale = fit_standardizer([[1.0, 2.0], [3.0, 2.0]])
        transformed = transform_standardized([[5.0, 2.0]], mean, scale)
        np.testing.assert_allclose(transformed, [[3.0, 0.0]])

    def test_missing_and_unseen_values(self):
        preprocessor = TabularPreprocessor.fit([[1.0], [np.nan], [3.0]], ["a", "b", "a"])
        numeric, categorical = preprocessor.transform([[np.nan], [5.0]], ["new", None])
        np.testing.assert_allclose(numeric, [[2.0], [5.0]])
        np.testing.assert_array_equal(categorical, [0, 0])

    def test_sampling_strategies(self):
        weighted = sample_indices(3, 4, weights=[0.0, 0.0, 1.0], replace=True, seed=1)
        np.testing.assert_array_equal(weighted, [2, 2, 2, 2])
        stratified = stratified_sample_indices([0, 0, 1, 1], 1, seed=1)
        self.assertEqual(set(np.asarray([0, 0, 1, 1])[stratified]), {0, 1})

    def test_streaming_weighted_mean(self):
        left = StreamingWeightedMean()
        left.update(2.0, weight=2)
        right = StreamingWeightedMean()
        right.update(5.0, weight=1)
        left.merge(right)
        self.assertEqual(left.value, 3.0)


if __name__ == "__main__":
    unittest.main()

