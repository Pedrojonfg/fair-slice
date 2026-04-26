"""
Module: Partition (Fair Division Algorithm)
Owner: Pedro

Responsible for:
  - Receiving the ingredient density map from vision.py
  - Computing the optimal partition into N connected regions
  - Returning per-person masks and fairness metrics

Algorithm: weighted power diagram optimized via three-phase descent,
with optional preference matrix for non-uniform fair division.

See README.md Section 3 for the full contract specification.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import (
    label as cc_label,
    binary_dilation,
    distance_transform_edt,
)
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Hyperparameters (tunable; defaults chosen for N in [2,8], M in [2k, 25k])
# ---------------------------------------------------------------------------

_EPS_DISH = 1e-3                 # threshold: dish vs background in density sum
_MAX_OPT_PIXELS = 20_000         # subsample domain if it exceeds this count
_EPS_FD_W = 0.5                  # finite-difference step for weights

# Phase budgets
_LLOYD_MAX_ITERS = 20
_WEIGHT_MAX_ITERS = 60
_JOINT_MAX_ITERS = 60

# Adaptive learning rates
_ETA_W_INIT = 5.0
_ETA_P_INIT = 3.0
_LR_DECAY = 0.5
_LR_GROWTH = 1.15
_ETA_FLOOR = 1e-8
_PATIENCE_GROW = 4

# Finite-difference step sizes — larger steps capture concentrated-ingredient
# transitions (e.g. pepperoni blobs) that small epsilons miss entirely
_EPS_FD_P_INIT = 2.0
_EPS_FD_P_MIN = 0.5

# Multi-start: Lloyd settles into a centroidal Voronoi (near-uniform cells),
# which is a terrible local minimum when ingredients are concentrated.
# We run several optimizations from different initializations and keep the best.
_MULTI_START_RUNS = 4

# Regularization & convergence
_REG_W = 1e-4                    # L2 on weights for numerical stability
_REL_TOL = 1e-5                  # relative L change below this => early stop
_BOUNDARY_PENALTY = 0.05         # soft projection strength
_LLOYD_SHIFT_TOL = 0.5           # Lloyd early stop in pixels
_RESEED_ALPHA_MOVE = 0.6         # active reseeding interpolation factor

_MIN_CHANNEL_TOTAL_FRACTION = 1e-3  # canales con total < esto * max_total se ignoran en fairness
_FAIRNESS_EARLY_EXIT = 0.92


# ===========================================================================
# Public API
# ===========================================================================

def compute_partition(
    ingredient_map: np.ndarray,
    n_people: int,
    mode: str = "free",
    preferences: np.ndarray | None = None,
) -> dict:
    """
    Computes the fairest partition of a dish into n_people regions.

    Args:
        ingredient_map: shape (H, W, K), dtype float32. Output of segment_dish().
        n_people: number of people (2-8).
        mode: "free" (power diagram, default) or "radial".
        preferences: optional shape (N, K) float array of per-person ingredient
                     preferences. preferences[i, j] = how much person i values
                     ingredient j (relative weights). Will be normalized so
                     that each ingredient sums to 1 across people. Default:
                     equal split (preferences[i, j] = 1/N for all).

    Returns:
        dict with keys "masks", "scores", "fairness", "seeds".
        See README.md Section 3 for full contract.

    Raises:
        ValueError on invalid inputs.
    """
    _validate_inputs(ingredient_map, n_people, mode, preferences)
    # Degenerate dish: a single dish pixel cannot form meaningful connected regions.
    dish_pixels = int((ingredient_map.sum(axis=-1) > _EPS_DISH).sum())
    if dish_pixels == 0:
        raise ValueError("ingredient_map has no dish pixels (all-zero).")
    if dish_pixels <= 1:
        raise ValueError("ingredient_map dish has too few pixels")
    P_norm = _normalize_preferences(preferences, n_people, ingredient_map.shape[-1])

    if mode == "radial":
        return _solve_radial(ingredient_map, n_people, P_norm)
    elif mode == "free":
        result_free = _solve_power_diagram(ingredient_map, n_people, P_norm)
        result_radial = _solve_radial(ingredient_map, n_people, P_norm)

        if result_radial["fairness"] > result_free["fairness"]:
            print(
                f"[partition] Radial wins: "
                f"{result_radial['fairness']:.3f} > {result_free['fairness']:.3f}"
            )
            result_free = result_radial
            result_free["mode_used"] = "radial"
        else:
            print(
                f"[partition] Free wins: "
                f"{result_free['fairness']:.3f} >= {result_radial['fairness']:.3f}"
            )
            result_free["mode_used"] = "free"

        return result_free
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'free' or 'radial'.")


# ===========================================================================
# Input validation & preference normalization
# ===========================================================================

def _validate_inputs(
    ingredient_map: np.ndarray,
    n_people: int,
    mode: str,
    preferences: np.ndarray | None,
) -> None:
    if not isinstance(n_people, (int, np.integer)):
        raise ValueError(f"n_people must be int, got {type(n_people).__name__}")
    if n_people < 2 or n_people > 8:
        raise ValueError(f"n_people must be in [2, 8], got {n_people}")
    if not isinstance(ingredient_map, np.ndarray):
        raise ValueError("ingredient_map must be numpy.ndarray")
    if ingredient_map.ndim != 3:
        raise ValueError(f"ingredient_map must be 3D (H,W,K); got ndim={ingredient_map.ndim}")
    if ingredient_map.dtype != np.float32:
        raise ValueError(f"ingredient_map must be float32, got {ingredient_map.dtype}")
    H, W, K = ingredient_map.shape
    if K < 1:
        raise ValueError(f"ingredient_map must have at least 1 channel, got K={K}")
    if H < 4 or W < 4:
        raise ValueError(f"ingredient_map too small: {H}x{W}")
    if not np.isfinite(ingredient_map).all():
        raise ValueError("ingredient_map contains NaN or Inf")
    if mode not in ("free", "radial"):
        raise ValueError(f"mode must be 'free' or 'radial', got {mode!r}")

    if preferences is not None:
        if not isinstance(preferences, np.ndarray):
            raise ValueError("preferences must be numpy.ndarray or None")
        if preferences.ndim != 2:
            raise ValueError(
                f"preferences must be 2D (N, K); got ndim={preferences.ndim}"
            )
        if preferences.shape != (n_people, K):
            raise ValueError(
                f"preferences shape {preferences.shape} does not match "
                f"(n_people={n_people}, K={K})"
            )
        if (preferences < 0).any():
            raise ValueError("preferences must be non-negative")


def _normalize_preferences(
    preferences: np.ndarray | None,
    n_people: int,
    K: int,
) -> np.ndarray:
    """
    Normalize the preference matrix so that each column (ingredient) sums to 1.
    This guarantees mass conservation: for every ingredient j,
        sum_i targets[i, j] == T[j].

    If preferences is None: return uniform (1/N for every entry).
    If a column sums to zero (no one wants ingredient j): split it uniformly.
    """
    if preferences is None:
        return np.full((n_people, K), 1.0 / n_people, dtype=np.float64)

    P = np.asarray(preferences, dtype=np.float64).copy()
    col_sums = P.sum(axis=0)                                # (K,)
    zero_cols = col_sums <= 1e-12
    if zero_cols.any():
        P[:, zero_cols] = 1.0 / n_people
        col_sums = P.sum(axis=0)
    P /= col_sums[None, :]
    return P


# ===========================================================================
# Power diagram solver (mode="free")
# ===========================================================================

def _solve_power_diagram(
    ingredient_map: np.ndarray,
    n_people: int,
    P_norm: np.ndarray,                 # (N, K) column-normalized preferences
) -> dict:
    """
    Multi-start three-phase weighted power diagram optimization with
    optional per-person preferences and Hungarian person<->cell matching.

    The target of person i for ingredient j is T[j] * P_norm[i, j]
    (uniform 1/N when no preferences). Hungarian matching is invoked between
    phases to keep generators aligned with the persons they best serve.
    """
    H, W, K = ingredient_map.shape
    T_total = ingredient_map.reshape(-1, K).sum(axis=0)
    active_channels = T_total > _MIN_CHANNEL_TOTAL_FRACTION * T_total.max()

    # --- Precompute things shared across restarts ---
    full_density = ingredient_map.sum(axis=-1)
    domain_full = full_density > _EPS_DISH
    if not domain_full.any():
        raise ValueError("ingredient_map has no dish pixels (all-zero).")

    dt_outside = distance_transform_edt(~domain_full).astype(np.float64)
    coords_opt, dens_opt, stride = _build_optimization_grid(ingredient_map, domain_full)
    T_opt = dens_opt.sum(axis=0)                                  # (K,)
    targets_opt = T_opt[None, :] * P_norm                         # (N, K) per-person targets
    T_safe = np.where(T_opt > 1e-12, T_opt, 1.0)
    alpha = 1.0 / (T_safe ** 2)                                   # (K,)
    alpha[T_opt <= 1e-12] = 0.0
    total_dens_opt = dens_opt.sum(axis=1)

    # --- Multi-start loop ---
    best_L = np.inf
    best_p = None
    best_w = None
    init_strategies = _build_init_strategies(
        coords_opt=coords_opt,
        dens_opt=dens_opt,
        total_dens_opt=total_dens_opt,
        n_people=n_people,
    )
    for run in range(_MULTI_START_RUNS):
        rng = np.random.default_rng(seed=42 + run * 97)
        p_init = init_strategies[run](rng)

        # Phase 1: Lloyd (geometric init, no preferences yet)
        p = _phase_lloyd(
            coords_opt, total_dens_opt, p_init,
            max_iters=_LLOYD_MAX_ITERS,
            domain_full=domain_full,
        )
        w = np.zeros(n_people, dtype=np.float64)

        # Hungarian matching #1: align persons with cells
        # Critical when preferences are non-uniform (otherwise no-op).
        p, w = _match_persons_to_cells(
            coords_opt, dens_opt, p, w, targets_opt, alpha
        )

        # Phase 2: weights (positions frozen)
        w = _phase_weights(
            coords_opt, dens_opt, p, w,
            targets_opt, alpha,
            dt_outside, domain_full.shape,
            max_iters=_WEIGHT_MAX_ITERS // 2,
        )

        # Hungarian matching #2: re-match after weight optimization
        p, w = _match_persons_to_cells(
            coords_opt, dens_opt, p, w, targets_opt, alpha
        )

        # Active reseeding: kick a deficit cell towards where it needs mass.
        p, w = _active_reseed(
            coords_opt, dens_opt, p, w,
            targets_opt, alpha,
            domain_full=domain_full,
        )

        # Phase 3: joint refinement of (p, w)
        p, w = _phase_joint(
            coords_opt, dens_opt, p, w,
            targets_opt, alpha,
            dt_outside, domain_full,
            max_iters=_JOINT_MAX_ITERS // 2,
        )

        # Final matching after joint phase
        p, w = _match_persons_to_cells(
            coords_opt, dens_opt, p, w, targets_opt, alpha
        )

        L_final, _ = _eval_loss(
            coords_opt, dens_opt, p, w,
            targets_opt, alpha, dt_outside, domain_full.shape
        )
        if L_final < best_L:
            best_L, best_p, best_w = L_final, p, w

        # Quick fairness estimate on the optimization grid (early exit).
        assignment_temp = _assign(coords_opt, p, w)
        temp_masks = _build_masks(coords_opt, assignment_temp, n_people, H, W)
        temp_scores = _compute_scores(temp_masks, ingredient_map)
        temp_fairness = _compute_fairness(
            temp_scores,
            P_norm,
            active_channels,
            ingredient_map,
            domain_full,
            n_ingredients=int(active_channels.sum()) if active_channels is not None else K,
        )
        if temp_fairness >= _FAIRNESS_EARLY_EXIT:
            break

    p, w = best_p, best_w

    # Refinamiento final solo con el mejor init
    w = _phase_weights(
        coords_opt, dens_opt, p, w,
        targets_opt, alpha,
        dt_outside, domain_full.shape,
        max_iters=_WEIGHT_MAX_ITERS,
    )
    p, w = _match_persons_to_cells(
        coords_opt, dens_opt, p, w, targets_opt, alpha
    )
    p, w = _phase_joint(
        coords_opt, dens_opt, p, w,
        targets_opt, alpha,
        dt_outside, domain_full,
        max_iters=_JOINT_MAX_ITERS,
    )
    p, w = _match_persons_to_cells(
        coords_opt, dens_opt, p, w, targets_opt, alpha
    )

    # --- Final assignment at FULL resolution ---
    coords_full, _ = _extract_domain(ingredient_map, domain_full)
    assignment_full = _assign(coords_full, p, w)

    masks = _build_masks(coords_full, assignment_full, n_people, H, W)
    masks = _enforce_connectivity(masks, domain_full)
    scores = _compute_scores(masks, ingredient_map)
    fairness = _compute_fairness(
        scores,
        P_norm,
        active_channels,
        ingredient_map,
        domain_full,
        n_ingredients=int(active_channels.sum()) if active_channels is not None else K,
    )

    return {
        "masks": masks,
        "scores": scores,
        "fairness": fairness,
        "seeds": p.astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Hungarian matching: persons <-> cells
# ---------------------------------------------------------------------------

def _match_persons_to_cells(
    coords: np.ndarray,
    densities: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    targets: np.ndarray,         # (N, K)
    alpha: np.ndarray,           # (K,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute current cell integrals, then find the optimal bijection
    person <-> cell that minimizes the total loss
        cost[i, c] = sum_k alpha_k * (I[c, k] - targets[i, k])^2

    Reorders p and w so that generator i serves person i.

    For uniform preferences, every row of `targets` is identical, so every
    permutation yields the same cost, and this is effectively a no-op.
    """
    N = p.shape[0]
    assignment = _assign(coords, p, w)
    I = _compute_integrals(densities, assignment, N)              # (N, K)

    # cost[i_person, c_cell]
    diff = I[None, :, :] - targets[:, None, :]                    # (N, N, K)
    cost = (alpha[None, None, :] * diff ** 2).sum(axis=-1)        # (N, N)

    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind == [0, 1, ..., N-1]; col_ind[i] = cell assigned to person i.
    p_new = p[col_ind].copy()
    w_new = w[col_ind].copy()
    return p_new, w_new


# ---------------------------------------------------------------------------
# Active reseeding (one-shot kick between weights and joint)
# ---------------------------------------------------------------------------

def _active_reseed(
    coords: np.ndarray,
    densities: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    targets: np.ndarray,         # (N, K)
    alpha: np.ndarray,           # (K,)
    domain_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    After optimizing weights with fixed positions, use the current integrals to
    move the most-deficit generator towards the centroid of the ingredient it
    lacks, within the cell that has the greatest excess of that ingredient.

    This is a one-shot, problem-informed kick to escape local minima.
    """
    N = p.shape[0]
    if N == 0:
        return p, w

    assignment = _assign(coords, p, w)
    I = _compute_integrals(densities, assignment, N)              # (N, K)

    # deficits/excesses (N, K)
    excess = np.maximum(0.0, I - targets)
    deficit = np.maximum(0.0, targets - I)

    # imbalance per person: sum_j alpha[j] * deficit[i,j]^2
    imbalance = (alpha[None, :] * (deficit ** 2)).sum(axis=1)     # (N,)
    if float(imbalance.max()) <= 1e-18:
        return p, w

    i_poor = int(np.argmax(imbalance))

    # ingredient most lacking for that person (weighted)
    lack = alpha * deficit[i_poor]
    j_star = int(np.argmax(lack))
    if float(lack[j_star]) <= 1e-18:
        return p, w

    # cell richest in that ingredient
    i_rich = int(np.argmax(excess[:, j_star]))
    if i_rich == i_poor:
        return p, w

    sel_rich = assignment == i_rich
    if not sel_rich.any():
        return p, w

    weights_rich = densities[sel_rich, j_star]
    wsum = float(weights_rich.sum())
    if wsum <= 1e-12:
        return p, w

    pixels_rich = coords[sel_rich]
    centroid_target = (pixels_rich * weights_rich[:, None]).sum(axis=0) / wsum

    p_new = p.copy()
    p_new[i_poor] = (1.0 - _RESEED_ALPHA_MOVE) * p[i_poor] + _RESEED_ALPHA_MOVE * centroid_target
    p_new = _hard_project_to_domain(p_new, domain_full)

    w_new = w.copy()
    w_new[i_poor] = 0.0
    return p_new, w_new


# ---------------------------------------------------------------------------
# Domain extraction & subsampling
# ---------------------------------------------------------------------------

def _extract_domain(
    ingredient_map: np.ndarray,
    domain_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords (M,2) as (y,x), densities (M,K)) for pixels in domain."""
    ys, xs = np.where(domain_mask)
    coords = np.stack([ys, xs], axis=1).astype(np.float64)
    densities = ingredient_map[ys, xs, :].astype(np.float64)
    return coords, densities


def _build_optimization_grid(
    ingredient_map: np.ndarray,
    domain_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Returns coords, densities, stride. If the domain is small enough,
    stride=1. Otherwise we subsample by an integer stride, aggregating
    pixel blocks of area stride^2 (block-sum preserves T_j exactly up
    to edge cropping).
    """
    n_full = int(domain_full.sum())
    if n_full <= _MAX_OPT_PIXELS:
        coords, dens = _extract_domain(ingredient_map, domain_full)
        return coords, dens, 1

    stride = int(np.ceil(np.sqrt(n_full / _MAX_OPT_PIXELS)))
    H, W, K = ingredient_map.shape
    Hc = (H // stride) * stride
    Wc = (W // stride) * stride
    cropped = ingredient_map[:Hc, :Wc, :]
    s = stride
    coarse = cropped.reshape(Hc // s, s, Wc // s, s, K).sum(axis=(1, 3))
    dens_total_coarse = coarse.sum(axis=-1)
    coarse_domain = dens_total_coarse > _EPS_DISH * (s * s)
    ys, xs = np.where(coarse_domain)
    coords = np.stack([ys * s + s / 2.0, xs * s + s / 2.0], axis=1)
    densities = coarse[ys, xs, :].astype(np.float64)
    return coords, densities, stride


# ---------------------------------------------------------------------------
# k-means++ initialization
# ---------------------------------------------------------------------------

def _build_init_strategies(
    coords_opt: np.ndarray,
    dens_opt: np.ndarray,
    total_dens_opt: np.ndarray,
    n_people: int,
) -> list:
    """
    Multi-start initializations designed to be structurally diverse.

    Returns a list of callables: init(rng) -> p_init (N, 2).
    The first four are:
      A) k-means++ weighted by total density (baseline)
      B) k-means++ weighted by the most concentrated ingredient channel
      C) stripe initialization along the dominant spatial axis
      D) random uniform sampling from the domain
    """
    A = lambda rng: _kmeans_pp_init(coords_opt, total_dens_opt, n_people, rng)

    j_star = _most_concentrated_channel(dens_opt)
    if j_star is None:
        B = A
        stripe_axis = None
    else:
        B = lambda rng: _kmeans_pp_init(coords_opt, dens_opt[:, j_star], n_people, rng)
        stripe_axis = _dominant_axis_for_channel(coords_opt, dens_opt[:, j_star])

    C = lambda rng: _stripe_init(
        coords_opt, total_dens_opt, n_people, rng,
        axis=stripe_axis,
    )
    D = lambda rng: _random_uniform_init(coords_opt, n_people, rng)
    return [A, B, C, D]


def _most_concentrated_channel(dens_opt: np.ndarray) -> int | None:
    """
    Pick the ingredient channel whose mass is most spatially concentrated.

    Heuristic: maximize (std / mean) over domain points.
    This is less sensitive to single-pixel outliers than max/mean and matches
    the "spatial concentration normalized by volume" intuition.
    Returns None if all channels are (near-)zero.
    """
    if dens_opt.ndim != 2 or dens_opt.shape[1] == 0:
        return None
    K = dens_opt.shape[1]
    scores = np.full(K, -np.inf, dtype=np.float64)
    for j in range(K):
        col = dens_opt[:, j]
        m = float(col.mean())
        if m <= 1e-12:
            continue
        scores[j] = float(col.std()) / (m + 1e-12)
    j_star = int(np.argmax(scores))
    if not np.isfinite(scores[j_star]):
        return None
    return j_star


def _dominant_axis_for_channel(
    coords: np.ndarray,
    channel_density: np.ndarray,
) -> int:
    """
    Choose which axis the *distribution* varies most along for a given channel.

    Returns 0 for y-dominant variation, 1 for x-dominant variation.
    We measure weighted spatial variance of coords along each axis.
    """
    w = np.asarray(channel_density, dtype=np.float64).clip(min=0)
    w_sum = float(w.sum())
    if w_sum <= 1e-12:
        # Fallback: unweighted variance
        vy = float(np.var(coords[:, 0]))
        vx = float(np.var(coords[:, 1]))
        return 0 if vy >= vx else 1

    my = float((coords[:, 0] * w).sum() / w_sum)
    mx = float((coords[:, 1] * w).sum() / w_sum)
    vy = float(((coords[:, 0] - my) ** 2 * w).sum() / w_sum)
    vx = float(((coords[:, 1] - mx) ** 2 * w).sum() / w_sum)
    return 0 if vy >= vx else 1


def _stripe_init(
    coords: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
    axis: int | None = None,
) -> np.ndarray:
    """
    Seeds aligned along stripes on the dominant spatial axis (x or y).
    Good when the main asymmetry is roughly 1D.
    """
    if axis is None:
        # Fallback when we don't have a meaningful "most concentrated" channel.
        # Use total domain spread as a weak proxy.
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        y_range = float(y_max - y_min)
        x_range = float(x_max - x_min)
        axis = 0 if y_range >= x_range else 1

    w = np.asarray(weights, dtype=np.float64).clip(min=0)
    w_sum = float(w.sum())
    if w_sum <= 1e-12:
        w = np.ones(len(coords), dtype=np.float64)
        w_sum = float(len(coords))

    other_axis = 1 - axis
    other_center = float((coords[:, other_axis] * w).sum() / w_sum)

    # Evenly spaced stripe positions across the domain bounding box.
    lo = float(coords[:, axis].min())
    hi = float(coords[:, axis].max())
    if k == 1:
        stripe_pos = np.array([(lo + hi) * 0.5], dtype=np.float64)
    else:
        stripe_pos = np.linspace(lo, hi, k, dtype=np.float64)

    # Small jitter to avoid identical outcomes under Lloyd.
    jitter_scale = 0.15 * (hi - lo) / max(k, 2)
    stripe_pos = stripe_pos + rng.normal(0.0, jitter_scale, size=k)

    p = np.empty((k, 2), dtype=np.float64)
    for i in range(k):
        if axis == 0:
            cand = np.array([stripe_pos[i], other_center], dtype=np.float64)
        else:
            cand = np.array([other_center, stripe_pos[i]], dtype=np.float64)
        # Snap candidate to nearest admissible point in the optimization grid.
        d2 = ((coords - cand[None, :]) ** 2).sum(axis=1)
        p[i] = coords[int(np.argmin(d2))]
    return p


def _random_uniform_init(
    coords: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample k seeds from the domain points."""
    M = coords.shape[0]
    replace = M < k
    idx = rng.choice(M, size=k, replace=replace)
    return coords[idx].astype(np.float64, copy=True)


def _kmeans_pp_init(
    coords: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns k seed positions (k, 2). Each new seed is sampled with
    probability proportional to d(x)^2 * weight(x), where d(x) is the
    distance to the nearest already-chosen seed.
    """
    M = coords.shape[0]
    w = np.asarray(weights, dtype=np.float64).clip(min=0)
    w_sum = w.sum()
    if w_sum <= 0:
        w = np.ones(M, dtype=np.float64)
        w_sum = float(M)

    chosen = np.empty((k, 2), dtype=np.float64)

    probs = w / w_sum
    idx = rng.choice(M, p=probs)
    chosen[0] = coords[idx]

    diff = coords - chosen[0]
    d2_min = (diff ** 2).sum(axis=1)

    for i in range(1, k):
        combined = d2_min * w
        total = combined.sum()
        if total <= 0:
            idx = rng.integers(0, M)
        else:
            probs = combined / total
            idx = rng.choice(M, p=probs)
        chosen[i] = coords[idx]
        diff = coords - chosen[i]
        d2_new = (diff ** 2).sum(axis=1)
        d2_min = np.minimum(d2_min, d2_new)

    return chosen


# ---------------------------------------------------------------------------
# Assignment, integrals, loss
# ---------------------------------------------------------------------------

def _assign(
    coords: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Power-diagram assignment: argmin over i of (||x-p_i||^2 - w_i)."""
    diff = coords[:, None, :] - p[None, :, :]
    dist_sq = (diff ** 2).sum(axis=-1)
    pow_dist = dist_sq - w[None, :]
    return np.argmin(pow_dist, axis=1)


def _compute_integrals(
    densities: np.ndarray,
    assignment: np.ndarray,
    k: int,
) -> np.ndarray:
    """Returns I of shape (k, K): per-cell totals of each ingredient."""
    K = densities.shape[1]
    I = np.empty((k, K), dtype=np.float64)
    for j in range(K):
        I[:, j] = np.bincount(assignment, weights=densities[:, j], minlength=k)
    return I


def _soft_projection_penalty(
    p: np.ndarray,
    dt_outside: np.ndarray,
    shape_hw: tuple[int, int],
) -> float:
    """Bilinear-sampled penalty: lambda * sum(d_outside(p_i)^2)."""
    H, W = shape_hw
    y = np.clip(p[:, 0], 0.0, H - 1.001)
    x = np.clip(p[:, 1], 0.0, W - 1.001)
    y0 = y.astype(int); x0 = x.astype(int)
    dy = y - y0; dx = x - x0
    d = (dt_outside[y0, x0]     * (1 - dy) * (1 - dx) +
         dt_outside[y0 + 1, x0] * dy       * (1 - dx) +
         dt_outside[y0, x0 + 1] * (1 - dy) * dx       +
         dt_outside[y0 + 1, x0 + 1] * dy   * dx)
    return _BOUNDARY_PENALTY * float((d ** 2).sum())


def _eval_loss(
    coords: np.ndarray, densities: np.ndarray,
    p: np.ndarray, w: np.ndarray,
    targets: np.ndarray,         # (N, K) — per-person targets
    alpha: np.ndarray,           # (K,)
    dt_outside: np.ndarray, shape_hw: tuple[int, int],
) -> tuple[float, np.ndarray]:
    """
    L = sum_{i,j} alpha_j * (I[i,j] - targets[i,j])^2 + reg + boundary_penalty

    Assumes generator i serves person i — Hungarian matching aligns them.
    """
    k = len(w)
    assignment = _assign(coords, p, w)
    I = _compute_integrals(densities, assignment, k)              # (N, K)
    residual = I - targets                                        # (N, K)
    data_loss = float((alpha[None, :] * residual ** 2).sum())
    reg_loss = _REG_W * float((w ** 2).sum())
    pen_loss = _soft_projection_penalty(p, dt_outside, shape_hw)
    return data_loss + reg_loss + pen_loss, I


# ---------------------------------------------------------------------------
# Phase 1: Lloyd
# ---------------------------------------------------------------------------

def _phase_lloyd(
    coords: np.ndarray,
    total_density: np.ndarray,
    p_init: np.ndarray,
    max_iters: int,
    domain_full: np.ndarray,
) -> np.ndarray:
    p = p_init.copy()
    k = p.shape[0]
    zero_w = np.zeros(k, dtype=np.float64)

    for _ in range(max_iters):
        assignment = _assign(coords, p, zero_w)
        p_new = p.copy()
        for i in range(k):
            sel = assignment == i
            if not sel.any():
                # Empty cell: reseed at the point farthest from any generator
                diff = coords[:, None, :] - p[None, :, :]
                d2 = (diff ** 2).sum(axis=-1).min(axis=1)
                far_idx = int(np.argmax(d2))
                p_new[i] = coords[far_idx]
                continue
            w_sel = total_density[sel]
            w_sum = w_sel.sum()
            if w_sum < 1e-12:
                continue
            cy = float((coords[sel, 0] * w_sel).sum() / w_sum)
            cx = float((coords[sel, 1] * w_sel).sum() / w_sum)
            p_new[i] = [cy, cx]
        p_new = _hard_project_to_domain(p_new, domain_full)
        shift = float(np.linalg.norm(p_new - p, axis=1).max())
        p = p_new
        if shift < _LLOYD_SHIFT_TOL:
            break
    return p


def _hard_project_to_domain(
    p: np.ndarray,
    domain_full: np.ndarray,
) -> np.ndarray:
    """Snap any generator outside the dish to the nearest dish pixel (safety net)."""
    H, W = domain_full.shape
    p_out = p.copy()
    for i in range(len(p)):
        y, x = p[i]
        yi = int(np.clip(round(y), 0, H - 1))
        xi = int(np.clip(round(x), 0, W - 1))
        if domain_full[yi, xi]:
            continue
        ys, xs = np.where(domain_full)
        d2 = (ys - y) ** 2 + (xs - x) ** 2
        j = int(np.argmin(d2))
        p_out[i] = [float(ys[j]), float(xs[j])]
    return p_out


# ---------------------------------------------------------------------------
# Phase 2: weights only
# ---------------------------------------------------------------------------

def _phase_weights(
    coords, densities, p, w_init,
    targets, alpha,
    dt_outside, shape_hw,
    max_iters,
) -> np.ndarray:
    w = w_init.copy()
    eta = _ETA_W_INIT
    L, _ = _eval_loss(coords, densities, p, w, targets, alpha, dt_outside, shape_hw)
    best_L, best_w = L, w.copy()
    consecutive_improvements = 0

    for it in range(max_iters):
        grad = _grad_w_fd(coords, densities, p, w, targets, alpha, dt_outside, shape_hw)
        w_new = w - eta * grad
        L_new, _ = _eval_loss(coords, densities, p, w_new, targets, alpha, dt_outside, shape_hw)

        if L_new < L - 1e-12:
            rel_change = (L - L_new) / max(L, 1e-12)
            w, L = w_new, L_new
            if L < best_L:
                best_L, best_w = L, w.copy()
            consecutive_improvements += 1
            if consecutive_improvements >= _PATIENCE_GROW:
                eta *= _LR_GROWTH
                consecutive_improvements = 0
            if rel_change < _REL_TOL:
                break
        else:
            eta *= _LR_DECAY
            consecutive_improvements = 0
            if eta < _ETA_FLOOR:
                break
    return best_w


def _grad_w_fd(
    coords, densities, p, w,
    targets, alpha, dt_outside, shape_hw,
) -> np.ndarray:
    """Centered finite differences on weights."""
    k = len(w)
    eps = _EPS_FD_W
    grad = np.empty(k, dtype=np.float64)
    for i in range(k):
        w_p = w.copy(); w_p[i] += eps
        w_m = w.copy(); w_m[i] -= eps
        L_p, _ = _eval_loss(coords, densities, p, w_p, targets, alpha, dt_outside, shape_hw)
        L_m, _ = _eval_loss(coords, densities, p, w_m, targets, alpha, dt_outside, shape_hw)
        grad[i] = (L_p - L_m) / (2 * eps)
    return grad


# ---------------------------------------------------------------------------
# Phase 3: joint (positions + weights) with adaptive eps_p
# ---------------------------------------------------------------------------

def _phase_joint(
    coords, densities, p_init, w_init,
    targets, alpha,
    dt_outside, domain_full,
    max_iters,
) -> tuple[np.ndarray, np.ndarray]:
    p = p_init.copy()
    w = w_init.copy()
    shape_hw = domain_full.shape
    eta_p, eta_w = _ETA_P_INIT, _ETA_W_INIT
    eps_p = _EPS_FD_P_INIT
    L, _ = _eval_loss(coords, densities, p, w, targets, alpha, dt_outside, shape_hw)
    best_L, best_p, best_w = L, p.copy(), w.copy()
    consecutive_improvements = 0

    for it in range(max_iters):
        grad_p, grad_w = _grad_joint_fd(
            coords, densities, p, w, targets, alpha, dt_outside, shape_hw,
            eps_p=eps_p,
        )
        p_new = p - eta_p * grad_p
        w_new = w - eta_w * grad_w
        p_new = _hard_project_to_domain(p_new, domain_full)
        L_new, _ = _eval_loss(coords, densities, p_new, w_new, targets, alpha, dt_outside, shape_hw)

        if L_new < L - 1e-12:
            rel_change = (L - L_new) / max(L, 1e-12)
            p, w, L = p_new, w_new, L_new
            if L < best_L:
                best_L, best_p, best_w = L, p.copy(), w.copy()
            consecutive_improvements += 1
            if consecutive_improvements >= _PATIENCE_GROW:
                eta_p *= _LR_GROWTH
                eta_w *= _LR_GROWTH
                consecutive_improvements = 0
            if rel_change < _REL_TOL:
                break
        else:
            eta_p *= _LR_DECAY
            eta_w *= _LR_DECAY
            eps_p = max(_EPS_FD_P_MIN, eps_p * 0.7)
            consecutive_improvements = 0
            if max(eta_p, eta_w) < _ETA_FLOOR:
                break
    return best_p, best_w


def _grad_joint_fd(
    coords, densities, p, w,
    targets, alpha, dt_outside, shape_hw,
    eps_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Centered FD over 3k parameters."""
    k = p.shape[0]
    grad_p = np.zeros_like(p)
    grad_w = np.zeros_like(w)

    for i in range(k):
        for d in range(2):
            p_p = p.copy(); p_p[i, d] += eps_p
            p_m = p.copy(); p_m[i, d] -= eps_p
            Lp, _ = _eval_loss(coords, densities, p_p, w, targets, alpha, dt_outside, shape_hw)
            Lm, _ = _eval_loss(coords, densities, p_m, w, targets, alpha, dt_outside, shape_hw)
            grad_p[i, d] = (Lp - Lm) / (2 * eps_p)

    ew = _EPS_FD_W
    for i in range(k):
        w_p = w.copy(); w_p[i] += ew
        w_m = w.copy(); w_m[i] -= ew
        Lp, _ = _eval_loss(coords, densities, p, w_p, targets, alpha, dt_outside, shape_hw)
        Lm, _ = _eval_loss(coords, densities, p, w_m, targets, alpha, dt_outside, shape_hw)
        grad_w[i] = (Lp - Lm) / (2 * ew)

    return grad_p, grad_w


# ---------------------------------------------------------------------------
# Masks: build from assignment and enforce connectivity
# ---------------------------------------------------------------------------

def _build_masks(
    coords_full: np.ndarray,
    assignment_full: np.ndarray,
    n_people: int,
    H: int, W: int,
) -> list[np.ndarray]:
    ys = coords_full[:, 0].astype(int)
    xs = coords_full[:, 1].astype(int)
    masks = []
    for i in range(n_people):
        m = np.zeros((H, W), dtype=bool)
        sel = assignment_full == i
        m[ys[sel], xs[sel]] = True
        masks.append(m)
    return masks


def _enforce_connectivity(
    masks: list[np.ndarray],
    domain_full: np.ndarray,
) -> list[np.ndarray]:
    """Keep largest CC of each mask; reabsorb orphans via iterative dilation."""
    H, W = domain_full.shape
    k = len(masks)
    orphans = np.zeros((H, W), dtype=bool)

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    for i in range(k):
        m = masks[i]
        if not m.any():
            continue
        lbl, num = cc_label(m, structure=structure)
        if num <= 1:
            continue
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        biggest = int(np.argmax(sizes))
        keep = lbl == biggest
        orphans |= (m & ~keep)
        masks[i] = keep

    if not orphans.any():
        return masks

    max_rounds = max(H, W)
    for _ in range(max_rounds):
        claimed_this_round = np.zeros((H, W), dtype=bool)
        for i in range(k):
            if not orphans.any():
                break
            dilated = binary_dilation(masks[i], structure=structure)
            capture = orphans & dilated & ~claimed_this_round
            if capture.any():
                masks[i] = masks[i] | capture
                claimed_this_round |= capture
        orphans &= ~claimed_this_round
        if not orphans.any():
            break

    if orphans.any():
        ys_o, xs_o = np.where(orphans)
        for y, x in zip(ys_o, xs_o):
            best_i, best_d = 0, np.inf
            for i in range(k):
                if not masks[i].any():
                    continue
                ys_m, xs_m = np.where(masks[i])
                d = float(np.min((ys_m - y) ** 2 + (xs_m - x) ** 2))
                if d < best_d:
                    best_d, best_i = d, i
            masks[best_i][y, x] = True

    return masks


# ---------------------------------------------------------------------------
# Scores & fairness (preference-aware)
# ---------------------------------------------------------------------------

def _compute_scores(
    masks: list[np.ndarray],
    ingredient_map: np.ndarray,
) -> np.ndarray:
    n = len(masks)
    K = ingredient_map.shape[-1]
    totals = ingredient_map.reshape(-1, K).sum(axis=0)
    active_channels = totals > _MIN_CHANNEL_TOTAL_FRACTION * totals.max()
    totals_safe = np.where(active_channels & (totals > 1e-12), totals, 1.0)
    scores = np.zeros((n, K), dtype=np.float32)
    for i, m in enumerate(masks):
        if not m.any():
            continue
        per_channel = ingredient_map[m].sum(axis=0)
        s = (per_channel / totals_safe).astype(np.float32)
        s[~active_channels] = 0.0
        scores[i, :] = s
    return scores


def _compute_fairness(
    scores: np.ndarray,          # (N, K) float32
    P_norm: np.ndarray,          # (N, K) float64
    active_channels: np.ndarray | None = None,  # (K,) bool
    ingredient_map: np.ndarray | None = None,   # (H, W, K) float32
    dish_mask: np.ndarray | None = None,        # (H, W) bool
    n_ingredients: int | None = None,           # complexity adjustment
) -> float:
    """
    Fairness metric for a partition.

    Backward-compatible mode (default):
      If `ingredient_map` or `dish_mask` is not provided, compute the original
      preference-aware metric:
        fairness = 1 - max_{i,j} |scores[i,j] - P_norm[i,j]| / max(P_norm[i,j], 1-P_norm[i,j]).

    Relative-to-achievable mode:
      If `ingredient_map` and `dish_mask` are provided, compute fairness
      relative to the *best distribution that is physically achievable* given
      the input geometry.

      Intuition: some ingredients occupy only a small fraction of the dish area,
      making an equal split impossible even with perfect cuts. In this mode,
      `fairness=1.0` means “you achieved the best possible split given the
      ingredient layout”, not “a perfect equal split exists”.
    """
    s = scores.astype(np.float64)

    if active_channels is not None:
        active = np.asarray(active_channels, dtype=bool)
        if active.ndim != 1:
            raise ValueError("active_channels must be a 1D boolean array (K,)")
        if active.size != s.shape[1]:
            raise ValueError("active_channels has wrong shape")
        if not active.any():
            return 1.0
    else:
        active = np.ones(s.shape[1], dtype=bool)

    # Relative-to-achievable metric (uniform ideal split, adjusted by coverage constraints)
    if ingredient_map is not None and dish_mask is not None:
        imap = np.asarray(ingredient_map)
        dmask = np.asarray(dish_mask, dtype=bool)
        if imap.ndim != 3 or dmask.ndim != 2:
            raise ValueError("ingredient_map must be (H,W,K) and dish_mask must be (H,W)")
        if imap.shape[:2] != dmask.shape:
            raise ValueError("ingredient_map and dish_mask shape mismatch")
        if imap.shape[2] != s.shape[1]:
            raise ValueError("ingredient_map K mismatch with scores")

        dish_area = float(dmask.sum())
        if dish_area <= 0:
            # Degenerate: fall back to the original metric
            ingredient_map = None
        else:
            n_people = int(s.shape[0])
            ideal = 1.0 / float(n_people)

            # Channel importance weights (normalized over active channels).
            # We use preference importance per ingredient:
            #   w_j ∝ sum_i P_norm[i, j]
            # With column-normalized preferences this is uniform (=1) per channel,
            # so the result matches equal weighting when preferences are uniform.
            P_full = P_norm.astype(np.float64)
            importance = P_full.sum(axis=0)  # (K,)
            weights = importance[active]
            w_sum = float(weights.sum())
            if w_sum <= 1e-18:
                weights = np.full(active.sum(), 1.0 / float(active.sum()), dtype=np.float64)
            else:
                weights = weights / w_sum

            rel_fairness_vals: list[float] = []
            active_idx = np.where(active)[0]
            for j in active_idx:
                presence_j = imap[:, :, j] > 0.05
                dish_area_i = float(dish_area)
                # Fracción del plato con presencia del ingrediente j
                f_j = float(presence_j.sum()) / float(dish_area_i)

                # Concentración: cuánto del ingrediente está fuera del alcance de N celdas iguales.
                # Si f_j < 1/N, al menos una celda no puede tener acceso → min_dev proporcional al déficit
                # Si f_j >= 1/N, geométricamente es posible repartir bien → min_dev = 0
                # Esto escala la desviación inevitable al rango [0, 1] relativo a la concentración,
                # no al ideal absoluto, eliminando la dependencia espuria de N.
                if f_j >= ideal:
                    min_dev_j = 0.0
                else:
                    # Desviación inevitable normalizada por el rango posible de desviación
                    min_dev_j = (ideal - f_j) / (1.0 - f_j + 1e-9)
                    min_dev_j = float(np.clip(min_dev_j, 0.0, 1.0 - ideal))

                # Desviación sobre el mínimo inevitable, normalizada por el máximo posible sobre ese mínimo
                actual_dev_j = float(np.max(np.abs(scores[:, j] - ideal)))
                achievable_dev_j = max(0.0, actual_dev_j - min_dev_j * ideal)
                worst_possible_j = max(ideal - min_dev_j * ideal, 1e-9)
                rel_fairness_j = float(np.clip(1.0 - achievable_dev_j / worst_possible_j, 0.0, 1.0))
                rel_fairness_vals.append(rel_fairness_j)

            rel_fairness_arr = np.asarray(rel_fairness_vals, dtype=np.float64)
            raw_fairness = float(np.clip(np.sum(rel_fairness_arr * weights), 0.0, 1.0))
            if n_ingredients is not None and n_ingredients >= 2:
                complexity_factor = 1.0 / math.log2(n_ingredients + 1)
                raw_fairness = float(np.clip(1.0 - (1.0 - raw_fairness) * complexity_factor, 0.0, 1.0))
            return raw_fairness

    # Original metric (preference-aware, worst-case normalized deviation)
    P = P_norm.astype(np.float64)
    s0 = s[:, active]
    P0 = P[:, active]
    dev = np.abs(s0 - P0)
    worst = np.maximum(P0, 1.0 - P0)
    worst = np.where(worst > 1e-12, worst, 1.0)
    norm_dev = dev / worst
    raw_fairness = float(np.clip(1.0 - norm_dev.max(), 0.0, 1.0))
    if n_ingredients is not None and n_ingredients >= 2:
        complexity_factor = 1.0 / math.log2(n_ingredients + 1)
        raw_fairness = float(np.clip(1.0 - (1.0 - raw_fairness) * complexity_factor, 0.0, 1.0))
    result = raw_fairness
    # Rawlsian component: penaliza si alguien lo pasa muy mal
    # Fairness rawlsiana = fairness de la persona que peor está
    rawls_fairness = float(np.min([
        np.clip(1.0 - np.max(np.abs(scores[i, active] - P_norm[i, active])) /
                np.maximum(P_norm[i, active], 1e-9).max(), 0.0, 1.0)
        for i in range(scores.shape[0])
    ]))

    # Blend: 70% utilitarista (lo que ya tienes), 30% rawlsiano
    RAWLS_WEIGHT = 0.30
    result = (1 - RAWLS_WEIGHT) * result + RAWLS_WEIGHT * rawls_fairness
    result = float(np.clip(result, 0.0, 1.0))
    return result


# ===========================================================================
# Radial solver (mode="radial")
# ===========================================================================

def _solve_radial(
    ingredient_map: np.ndarray,
    n_people: int,
    P_norm: np.ndarray,             # (N, K)
) -> dict:
    """
    Radial partition around the density centroid. Cuts equalize TOTAL
    density across N sectors. With preferences, sectors are assigned to
    persons via Hungarian matching (optimal bijection person <-> sector).

    Limitation: radial cuts cannot adapt asymmetrically per ingredient, so
    this mode is materially worse than free mode when preferences are
    non-uniform AND ingredients are angularly heterogeneous.
    """
    H, W, K = ingredient_map.shape
    T_total = ingredient_map.reshape(-1, K).sum(axis=0)
    active_channels = T_total > _MIN_CHANNEL_TOTAL_FRACTION * T_total.max()
    total_density = ingredient_map.sum(axis=-1)
    domain = total_density > _EPS_DISH

    if not domain.any():
        raise ValueError("ingredient_map has no dish pixels (all-zero).")

    ys, xs = np.where(domain)
    w = total_density[ys, xs]
    w_sum = w.sum()
    cy = float((ys * w).sum() / w_sum)
    cx = float((xs * w).sum() / w_sum)

    dy = ys - cy
    dx = xs - cx
    theta = np.arctan2(dy, dx)
    theta_pos = np.where(theta < 0, theta + 2 * np.pi, theta).astype(np.float64, copy=False)

    # Hungarian assignment cost weights (fixed across rotations)
    T_safe = np.where(T_total > 1e-12, T_total, 1.0)
    alpha = 1.0 / (T_safe ** 2)
    alpha[T_total <= 1e-12] = 0.0
    targets = T_total[None, :] * P_norm                          # (N, K)

    def _radial_for_offset(offset_rad: float) -> tuple[list[np.ndarray], np.ndarray, float]:
        """
        Build radial sectors starting at `offset_rad` (in radians), assign persons to sectors,
        and return (masks, seeds, fairness).
        """
        theta_shift = (theta_pos - offset_rad) % (2 * np.pi)
        order = np.argsort(theta_shift)
        theta_sorted = theta_shift[order]
        w_sorted = w[order]
        cum = np.cumsum(w_sorted)
        total = float(cum[-1])

        cut_angles = np.empty(n_people - 1, dtype=np.float64)
        for i in range(1, n_people):
            target = total * i / n_people
            idx = int(np.searchsorted(cum, target))
            idx = min(idx, len(theta_sorted) - 1)
            cut_angles[i - 1] = theta_sorted[idx]
        bounds = np.concatenate([[0.0], cut_angles, [2 * np.pi]])

        # Sector id per domain pixel (vectorized, no HxW theta field needed)
        sector_id = np.searchsorted(bounds, theta_shift, side="right") - 1
        sector_id = np.clip(sector_id, 0, n_people - 1).astype(np.int32, copy=False)

        raw_masks: list[np.ndarray] = []
        raw_seeds = np.zeros((n_people, 2), dtype=np.float64)
        for c in range(n_people):
            m = np.zeros((H, W), dtype=bool)
            sel = sector_id == c
            if sel.any():
                m[ys[sel], xs[sel]] = True
            raw_masks.append(m)

            if sel.any():
                lo, hi = float(bounds[c]), float(bounds[c + 1])
                mid_angle = offset_rad + 0.5 * (lo + hi)
                r_mean = float(np.mean(np.sqrt((ys[sel] - cy) ** 2 + (xs[sel] - cx) ** 2)))
                raw_seeds[c] = [cy + r_mean * np.sin(mid_angle), cx + r_mean * np.cos(mid_angle)]
            else:
                raw_seeds[c] = [cy, cx]

        # Hungarian assignment of persons to sectors (depends on integrals)
        sector_integrals = np.zeros((n_people, K), dtype=np.float64)
        for c in range(n_people):
            if raw_masks[c].any():
                sector_integrals[c] = ingredient_map[raw_masks[c]].sum(axis=0)

        diff = sector_integrals[None, :, :] - targets[:, None, :]    # (N, N, K)
        cost = (alpha[None, None, :] * diff ** 2).sum(axis=-1)       # (N, N)
        _, col_ind = linear_sum_assignment(cost)

        masks = [raw_masks[col_ind[i]] for i in range(n_people)]
        seeds = np.stack([raw_seeds[col_ind[i]] for i in range(n_people)])

        scores = _compute_scores(masks, ingredient_map)
        fairness = _compute_fairness(
            scores,
            P_norm,
            active_channels,
            ingredient_map,
            domain,
            n_ingredients=int(active_channels.sum()) if active_channels is not None else K,
        )
        return masks, seeds, float(fairness)

    best: dict | None = None
    best_f = -np.inf
    best_deg = 0

    # Evaluate every degree (0..359). This is usually fast because most work is vectorized.
    for deg in range(360):
        offset = (deg * np.pi) / 180.0
        masks, seeds, f = _radial_for_offset(offset)
        if f > best_f:
            best_f = f
            best_deg = deg
            scores = _compute_scores(masks, ingredient_map)
            best = {
                "masks": masks,
                "scores": scores,
                "fairness": float(f),
                "seeds": seeds,
                "radial_rotation_deg": int(deg),
            }
            if best_f >= 0.999:
                break

    assert best is not None
    return best