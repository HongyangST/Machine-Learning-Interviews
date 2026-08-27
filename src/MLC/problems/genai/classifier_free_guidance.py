"""Classifier-free guidance combination for diffusion predictions.

Interview prompt: combine conditional and unconditional noise predictions and
explain the fidelity/diversity trade-off as guidance scale increases.
"""

from __future__ import annotations

import numpy as np


def classifier_free_guidance(
    unconditional_prediction: np.ndarray,
    conditional_prediction: np.ndarray,
    *,
    guidance_scale: float = 7.5,
) -> np.ndarray:
    """Return uncond + scale * (cond - uncond), preserving tensor shape."""
    unconditional = np.asarray(unconditional_prediction, dtype=np.float64)
    conditional = np.asarray(conditional_prediction, dtype=np.float64)
    if unconditional.shape != conditional.shape or not np.isfinite(guidance_scale):
        raise ValueError("prediction shapes must match and guidance_scale be finite")
    return unconditional + guidance_scale * (conditional - unconditional)

