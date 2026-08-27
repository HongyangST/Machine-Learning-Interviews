"""Direct Preference Optimization loss from sequence log probabilities.

Interview prompt: compare chosen versus rejected policy likelihood gaps against a
reference model and implement the stable negative log-sigmoid objective.
"""

from __future__ import annotations

import numpy as np


def dpo_loss(
    policy_chosen_logp: np.ndarray,
    policy_rejected_logp: np.ndarray,
    reference_chosen_logp: np.ndarray,
    reference_rejected_logp: np.ndarray,
    *,
    beta: float = 0.1,
) -> float:
    """Return mean DPO loss for one chosen/rejected pair per batch row."""
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            policy_chosen_logp,
            policy_rejected_logp,
            reference_chosen_logp,
            reference_rejected_logp,
        )
    ]
    if arrays[0].ndim != 1 or any(value.shape != arrays[0].shape for value in arrays) or beta <= 0:
        raise ValueError("all log-probability vectors must match and beta be positive")
    policy_gap = arrays[0] - arrays[1]
    reference_gap = arrays[2] - arrays[3]
    margin = beta * (policy_gap - reference_gap)
    return float(np.logaddexp(0.0, -margin).mean())

