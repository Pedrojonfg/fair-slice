"""
Module: Partition (Fair Division Algorithm)
Owner: Pedro

Responsible for:
  - Receiving the ingredient density map from vision.py
  - Computing the optimal partition into N connected regions
  - Returning per-person masks and fairness metrics

See README.md Section 3 for the full contract specification.
"""

import numpy as np


def compute_partition(
    ingredient_map: np.ndarray,
    n_people: int,
    mode: str = "free"
) -> dict:
    """
    Computes the fairest partition of a dish into n_people regions.

    Args:
        ingredient_map: np.ndarray of shape (H, W, K), dtype float32.
                        Output of segment_dish().
        n_people: Number of people to divide the dish between (2-8).
        mode: "radial" for radial cuts from optimal center,
              "free" for Voronoi-based connected partition (default).

    Returns:
        dict with keys:
            "masks": list of N np.ndarrays of shape (H, W), dtype bool.
                     Each mask is True for pixels belonging to that person.
                     Masks are non-overlapping and cover the full dish.
            "scores": np.ndarray of shape (N, K), dtype float32.
                      scores[i, k] = proportion of ingredient k assigned
                      to person i. Perfect fairness = 1/N for all entries.
            "fairness": float in [0, 1]. 1.0 = perfectly fair.
                        Computed as 1 - normalized max deviation.
            "seeds": np.ndarray of shape (N, 2). Voronoi seed positions.
                     Only relevant for mode="free".

    Raises:
        ValueError: If n_people < 2 or > 8.
        ValueError: If ingredient_map has wrong shape/dtype.
    """
    raise NotImplementedError("TODO: Implement Voronoi partition")