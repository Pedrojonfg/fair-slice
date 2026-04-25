from __future__ import annotations

import numpy as np


def build_preference_matrix(
    n_people: int,
    ingredient_labels: dict[int, str],
    user_inputs: dict[int, list[float]],
) -> np.ndarray:
    """
    Build a (N, K) float32 preference matrix from UI slider inputs.

    Args:
        n_people: number of people (N).
        ingredient_labels: dict mapping ingredient index -> label. K = len(ingredient_labels).
        user_inputs: dict mapping person index (0..N-1) -> list of K floats.

    Returns:
        np.ndarray of shape (N, K), dtype float32.

    Raises:
        ValueError: on any validation error (shape mismatch, missing people, negatives).
    """
    if not isinstance(n_people, int):
        raise ValueError(f"n_people must be int, got {type(n_people).__name__}")
    if n_people < 1:
        raise ValueError(f"n_people must be >= 1, got {n_people}")
    if not isinstance(ingredient_labels, dict):
        raise ValueError(
            f"ingredient_labels must be dict[int, str], got {type(ingredient_labels).__name__}"
        )
    if not isinstance(user_inputs, dict):
        raise ValueError(
            f"user_inputs must be dict[int, list[float]], got {type(user_inputs).__name__}"
        )

    K = len(ingredient_labels)
    if K < 1:
        raise ValueError(f"ingredient_labels must define at least 1 ingredient, got K={K}")

    if len(user_inputs) != n_people:
        raise ValueError(
            f"user_inputs must have exactly n_people={n_people} entries, got {len(user_inputs)}"
        )

    P = np.empty((n_people, K), dtype=np.float32)
    for person_idx in range(n_people):
        if person_idx not in user_inputs:
            raise ValueError(
                f"user_inputs is missing preferences for person index {person_idx} "
                f"(expected keys 0..{n_people - 1})"
            )
        values = user_inputs[person_idx]
        if not isinstance(values, list):
            raise ValueError(
                f"user_inputs[{person_idx}] must be a list of floats, got {type(values).__name__}"
            )
        if len(values) != K:
            raise ValueError(
                f"user_inputs[{person_idx}] must have length K={K}, got {len(values)}"
            )
        try:
            row = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise ValueError(f"user_inputs[{person_idx}] contains non-numeric values") from e

        if (row < 0).any():
            bad = np.where(row < 0)[0][0]
            raise ValueError(
                f"user_inputs[{person_idx}][{bad}] must be >= 0, got {float(row[bad])}"
            )
        P[person_idx, :] = row

    return P


def uniform_preferences(n_people: int, K: int) -> np.ndarray:
    """
    Uniform preferences for the case where the user does not specify any.
    Returns a (N, K) float32 matrix of 1.0.
    """
    if not isinstance(n_people, int):
        raise ValueError(f"n_people must be int, got {type(n_people).__name__}")
    if not isinstance(K, int):
        raise ValueError(f"K must be int, got {type(K).__name__}")
    if n_people < 1:
        raise ValueError(f"n_people must be >= 1, got {n_people}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    return np.ones((n_people, K), dtype=np.float32)

