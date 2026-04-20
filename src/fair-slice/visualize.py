"""
Module: Visualization (Overlay & Output)
Owner: Quirce

Responsible for:
  - Taking the original photo + partition masks
  - Rendering colored overlays with cut lines
  - Generating the final output image + fairness stats

See README.md Section 4 for the full contract specification.
"""

import numpy as np
from PIL import Image


def render_partition(
    image_path: str,
    masks: list[np.ndarray],
    scores: np.ndarray,
    ingredient_labels: dict[int, str],
    fairness: float
) -> Image.Image:
    """
    Renders the partition overlay on the original dish photo.

    Args:
        image_path: Path to the original photo.
        masks: List of N boolean masks (H, W) from compute_partition().
        scores: (N, K) array of ingredient proportions per person.
        ingredient_labels: dict mapping index to ingredient name.
        fairness: Overall fairness score.

    Returns:
        PIL Image with:
          - Semi-transparent colored regions (one color per person)
          - White cut lines at region boundaries
          - Legend with per-person ingredient breakdown
    """
    raise NotImplementedError("TODO: Implement visualization")