"""Tests for language-model and generative-AI answers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from problems.genai.classifier_free_guidance import classifier_free_guidance  # noqa: E402
from problems.genai.contrastive_loss import symmetric_contrastive_loss  # noqa: E402
from problems.genai.dpo_loss import dpo_loss  # noqa: E402
from problems.genai.lora_linear import lora_linear  # noqa: E402
from problems.genai.rag_retrieval import cosine_top_k  # noqa: E402
from problems.language_models.byte_pair_encoding import encode_bpe, learn_bpe  # noqa: E402
from problems.language_models.causal_attention_mask import causal_attention_mask  # noqa: E402
from problems.language_models.kv_cache import KVCache  # noqa: E402
from problems.language_models.padded_batches import pad_sequences  # noqa: E402
from problems.language_models.sinusoidal_position_encoding import (  # noqa: E402
    sinusoidal_position_encoding,
)
from problems.language_models.top_k_top_p_sampling import (  # noqa: E402
    filtered_probabilities,
)


class LanguageAndGenAIAnswersTest(unittest.TestCase):
    def test_bpe(self):
        merges = learn_bpe(["low", "low", "lower"], 3)
        encoded = encode_bpe("low", merges)
        self.assertLess(len(encoded), 4)

    def test_causal_attention_mask(self):
        mask = causal_attention_mask(3, key_is_valid=[True, True, False])
        np.testing.assert_array_equal(mask, [[True, False, False], [True, True, False], [True, True, False]])

    def test_sinusoidal_position_encoding(self):
        encoding = sinusoidal_position_encoding(3, 5)
        self.assertEqual(encoding.shape, (3, 5))
        np.testing.assert_allclose(encoding[0, 0::2], 0.0)
        np.testing.assert_allclose(encoding[0, 1::2], 1.0)

    def test_top_k_top_p_filtering(self):
        probabilities = filtered_probabilities([0.0, 1.0, 5.0], top_k=1)
        np.testing.assert_allclose(probabilities, [0.0, 0.0, 1.0])

    def test_kv_cache(self):
        cache = KVCache(max_length=2)
        cache.append([[1.0, 2.0]], [[3.0, 4.0]])
        cache.append([[5.0, 6.0]], [[7.0, 8.0]])
        keys, values = cache.arrays()
        self.assertEqual(keys.shape, (1, 2, 2))
        self.assertEqual(values[0, 1, 1], 8.0)

    def test_padding(self):
        ids, valid = pad_sequences([[1, 2], [3]], padding_id=-1)
        np.testing.assert_array_equal(ids, [[1, 2], [3, -1]])
        np.testing.assert_array_equal(valid, [[True, True], [True, False]])

    def test_lora_linear(self):
        output = lora_linear([[1.0, 2.0]], np.eye(2), [[1.0, 0.0]], [[2.0], [3.0]], alpha=1.0)
        np.testing.assert_allclose(output, [[3.0, 5.0]])

    def test_contrastive_loss_prefers_matching_pairs(self):
        matched = symmetric_contrastive_loss(np.eye(3), np.eye(3), temperature=0.2)
        mismatched = symmetric_contrastive_loss(np.eye(3), np.roll(np.eye(3), 1, axis=0), temperature=0.2)
        self.assertLess(matched, mismatched)

    def test_rag_top_k(self):
        indices, scores = cosine_top_k([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], 2)
        np.testing.assert_array_equal(indices, [0, 2])
        self.assertGreater(scores[0], scores[1])

    def test_dpo_loss_rewards_policy_preference(self):
        preferred = dpo_loss([0.0], [-2.0], [0.0], [-0.5], beta=1.0)
        reversed_preference = dpo_loss([-2.0], [0.0], [0.0], [-0.5], beta=1.0)
        self.assertLess(preferred, reversed_preference)

    def test_classifier_free_guidance(self):
        result = classifier_free_guidance([1.0, 2.0], [2.0, 4.0], guidance_scale=2.0)
        np.testing.assert_allclose(result, [3.0, 6.0])


if __name__ == "__main__":
    unittest.main()

