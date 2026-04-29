import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, label as cc_label

from partition import compute_partition
import partition as partition_mod


@dataclass(frozen=True)
class CaseSpec:
    name: str
    seed: int
    dish_kind: str
    ingredient_kind: str
    k_ingredients: int
    n_people: int


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    # Keep largest connected component so the dish stays meaningful.
    lbl, num = cc_label(mask)
    if num <= 1:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    biggest = int(np.argmax(sizes))
    return lbl == biggest


def _make_dish_mask(rng: np.random.Generator, h: int, w: int, kind: str) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy = float(rng.uniform(h * 0.25, h * 0.75))
    cx = float(rng.uniform(w * 0.25, w * 0.75))

    if kind == "circle":
        r = float(rng.uniform(min(h, w) * 0.18, min(h, w) * 0.32))
        dist = (yy - cy) ** 2 + (xx - cx) ** 2
        return dist <= r**2

    if kind == "ellipse":
        ry = float(rng.uniform(h * 0.16, h * 0.30))
        rx = float(rng.uniform(w * 0.16, w * 0.30))
        dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        return dist <= 1.0

    if kind == "donut":
        r_out = float(rng.uniform(min(h, w) * 0.24, min(h, w) * 0.36))
        r_in = float(rng.uniform(r_out * 0.35, r_out * 0.65))
        dist = (yy - cy) ** 2 + (xx - cx) ** 2
        return (dist <= r_out**2) & (dist >= r_in**2)

    if kind == "crescent":
        r1 = float(rng.uniform(min(h, w) * 0.22, min(h, w) * 0.35))
        # Shift the subtracting circle to create a crescent.
        shift = float(rng.uniform(min(h, w) * 0.08, min(h, w) * 0.18))
        cy2 = cy + shift * (1.0 if rng.random() < 0.5 else -1.0)
        cx2 = cx + shift * (1.0 if rng.random() < 0.5 else -1.0)
        dist1 = (yy - cy) ** 2 + (xx - cx) ** 2
        dist2 = (yy - cy2) ** 2 + (xx - cx2) ** 2
        return (dist1 <= r1**2) & (dist2 >= (r1 * 0.75) ** 2)

    if kind == "bridge":
        r1 = float(rng.uniform(min(h, w) * 0.20, min(h, w) * 0.32))
        r2 = float(rng.uniform(min(h, w) * 0.20, min(h, w) * 0.32))
        sep = float(rng.uniform(min(h, w) * 0.10, min(h, w) * 0.22))
        # Two circles plus a connecting corridor.
        cy1, cx1 = cy, cx - sep / 2.0
        cy2, cx2 = cy, cx + sep / 2.0
        dist1 = (yy - cy1) ** 2 + (xx - cx1) ** 2
        dist2 = (yy - cy2) ** 2 + (xx - cx2) ** 2
        corridor = (yy > cy - r1 * 0.35) & (yy < cy + r1 * 0.35)
        corridor &= (xx > cx1) & (xx < cx2)
        m = ((dist1 <= r1**2) | (dist2 <= r2**2)) & corridor
        # Make sure we don't end up with something too small:
        if m.sum() < (h * w) * 0.10:
            m = (dist1 <= r1**2) | (dist2 <= r2**2)
        return m

    if kind == "irregular":
        # Smoothed random field threshold -> irregular but connected-ish shapes.
        field = rng.normal(size=(h, w)).astype(np.float64)
        field = gaussian_filter(field, sigma=float(rng.uniform(3.0, 7.0)))
        # Add a gentle center bias so the dish is usually around the middle.
        field += -(((yy - cy) ** 2) / (h * h * 0.20) + ((xx - cx) ** 2) / (w * w * 0.20))
        thresh = float(np.quantile(field, rng.uniform(0.35, 0.55)))
        mask = field >= thresh
        mask = _largest_cc(mask)
        return mask

    raise ValueError(f"Unknown dish kind: {kind}")


def _sample_points_from_mask(rng: np.random.Generator, mask: np.ndarray, n: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise ValueError("mask has no points")
    replace = len(ys) < n
    idx = rng.choice(len(ys), size=n, replace=replace)
    return np.stack([ys[idx].astype(np.float64), xs[idx].astype(np.float64)], axis=1)


def _gaussian_blob(yy: np.ndarray, xx: np.ndarray, cy: float, cx: float, sigma: float) -> np.ndarray:
    # Stable positive gaussian (no need to normalize since we control amplitude).
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-d2 / (2.0 * sigma**2))


def _make_ingredient_map(
    rng: np.random.Generator,
    h: int,
    w: int,
    dish_mask: np.ndarray,
    k: int,
    ingredient_kind: str,
) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    dish_mask_f = dish_mask.astype(np.float64)

    # Centroid for angular patterns.
    ys, xs = np.where(dish_mask)
    if ys.size == 0:
        raise ValueError("dish mask empty")
    cy = float(ys.mean())
    cx = float(xs.mean())
    dy = yy - cy
    dx = xx - cx
    theta = np.arctan2(dy, dx)
    theta_pos = np.where(theta < 0, theta + 2.0 * np.pi, theta)

    imap = np.zeros((h, w, k), dtype=np.float64)

    # Make all dishes "physically valid" by keeping a small baseline density.
    # Dish pixels are computed by sum across channels, so patterns can be sparse
    # in some channels while others keep the dish well-defined.
    base_all_channels = float(rng.uniform(0.006, 0.012))

    # Decide which ingredient (channel index) becomes sparse (inactive in fairness).
    sparse_channel = int(rng.integers(0, k))

    for ch in range(k):
        if ingredient_kind in ("uniform", "uniform_noisy"):
            m = base_all_channels
            if ingredient_kind == "uniform_noisy":
                m = m * (1.0 + 0.05 * rng.normal(size=(h, w)))
            dens = np.clip(m, 0.0, None)

        elif ingredient_kind in ("pepperoni_like", "pepperoni_like_sparse"):
            n_blobs = int(rng.integers(1, 5))
            pts = _sample_points_from_mask(rng, dish_mask, n_blobs)
            dens = np.zeros((h, w), dtype=np.float64) + base_all_channels
            for (bcy, bcx) in pts:
                sigma = float(rng.uniform(1.5, 7.0))
                amp = float(rng.uniform(0.15, 0.9))
                dens += amp * _gaussian_blob(yy, xx, float(bcy), float(bcx), sigma)
            dens = dens.astype(np.float64)

            if ingredient_kind == "pepperoni_like_sparse" and ch == sparse_channel:
                # Turn one channel into a "rare ingredient".
                dens = dens * float(rng.uniform(1e-5, 2e-4))

        elif ingredient_kind == "angular_heterogeneous":
            # Mix blobs + one angular sector channel (radial should do well).
            if ch == 0:
                n_sectors = int(rng.integers(3, 7))
                sector_edges = np.linspace(0.0, 2.0 * np.pi, n_sectors + 1)
                sector_id = np.searchsorted(sector_edges, theta_pos, side="right") - 1
                sector_id = np.clip(sector_id, 0, n_sectors - 1)
                # Create alternating-ish importance with some noise.
                weights = rng.uniform(0.2, 1.2, size=n_sectors)
                dens = base_all_channels + float(rng.uniform(0.25, 0.75)) * weights[sector_id]
            else:
                n_blobs = int(rng.integers(1, 4))
                pts = _sample_points_from_mask(rng, dish_mask, n_blobs)
                dens = np.zeros((h, w), dtype=np.float64) + base_all_channels
                for (bcy, bcx) in pts:
                    sigma = float(rng.uniform(2.0, 8.0))
                    amp = float(rng.uniform(0.1, 0.7))
                    dens += amp * _gaussian_blob(yy, xx, float(bcy), float(bcx), sigma)

            if ch == sparse_channel:
                dens = dens * float(rng.uniform(1e-5, 5e-4))

        elif ingredient_kind == "stripes_and_gradient":
            # One channel has stripes, another has a radial-ish gradient, others blobs.
            if ch % 3 == 0:
                axis = int(rng.integers(0, 2))
                stripes_n = int(rng.integers(3, 9))
                coord = yy if axis == 0 else xx
                # Normalize coord to [0,1] on the dish.
                coord_dish = coord[dish_mask]
                lo = float(coord_dish.min())
                hi = float(coord_dish.max())
                t = (coord - lo) / max(hi - lo, 1e-9)
                stripe_field = 0.5 + 0.5 * np.sin(2.0 * np.pi * stripes_n * t)
                dens = base_all_channels + float(rng.uniform(0.15, 0.8)) * stripe_field
            elif ch % 3 == 1:
                # Distance gradient (center high, edges low) with dish bias.
                dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                dist_dish = dist[dish_mask]
                dist_lo = float(dist_dish.min())
                dist_hi = float(dist_dish.max())
                t = (dist - dist_lo) / max(dist_hi - dist_lo, 1e-9)
                dens = base_all_channels + float(rng.uniform(0.10, 0.70)) * (1.0 - t)
            else:
                n_blobs = int(rng.integers(1, 4))
                pts = _sample_points_from_mask(rng, dish_mask, n_blobs)
                dens = np.zeros((h, w), dtype=np.float64) + base_all_channels
                for (bcy, bcx) in pts:
                    sigma = float(rng.uniform(1.5, 6.0))
                    amp = float(rng.uniform(0.15, 0.9))
                    dens += amp * _gaussian_blob(yy, xx, float(bcy), float(bcx), sigma)

            if ch == sparse_channel:
                dens = dens * float(rng.uniform(1e-5, 3e-4))

        else:
            raise ValueError(f"Unknown ingredient kind: {ingredient_kind}")

        # Apply dish mask.
        dens = dens * dish_mask_f

        # Ensure non-negativity and float32 compatibility.
        dens = np.clip(dens, 0.0, None)
        imap[:, :, ch] = dens

    return imap.astype(np.float32, copy=False)


def _make_case(case_spec: CaseSpec, h: int = 48, w: int = 48) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(case_spec.seed)
    # Ensure dish is not tiny.
    for _ in range(10):
        dish_mask = _make_dish_mask(rng, h, w, case_spec.dish_kind)
        if dish_mask.sum() >= (h * w) * 0.12:
            break
    else:
        dish_mask = np.ones((h, w), dtype=bool)

    imap = _make_ingredient_map(
        rng=rng,
        h=h,
        w=w,
        dish_mask=dish_mask,
        k=case_spec.k_ingredients,
        ingredient_kind=case_spec.ingredient_kind,
    )
    return imap, case_spec.n_people


def _sweep_convex_hyperparams(
    monkeypatch: pytest.MonkeyPatch,
    imap: np.ndarray,
    n_people: int,
    base_eta_w2_init: float,
    base_eta_w3_init: float,
    base_eta_p_init: float,
    eta_w2_mults: list[float],
    eta_w3_mults: list[float],
    eta_p_mults: list[float],
) -> tuple[float, dict, list[dict]]:
    # We sweep decoupled learning rates:
    # - phase2 weights LR uses `_ETA_W2_INIT`
    # - phase3 weights LR uses `_ETA_W3_INIT`
    # - phase3 position LR uses `_ETA_P_INIT`

    best_fairness = -np.inf
    best_cfg: dict = {}
    per_cfg_rows: list[dict] = []

    # Ensure each sweep starts from the same baseline constants, so we can
    # aggregate results consistently across cases.
    # When weights are coupled (legacy behavior), we set both phase2 and phase3
    # weight LRs to the same value.
    monkeypatch.setattr(partition_mod, "_ETA_W2_INIT", float(base_eta_w2_init))
    monkeypatch.setattr(partition_mod, "_ETA_W3_INIT", float(base_eta_w3_init))
    monkeypatch.setattr(partition_mod, "_ETA_P_INIT", float(base_eta_p_init))

    for eta_w2_mult in eta_w2_mults:
        for eta_w3_mult in eta_w3_mults:
            for eta_p_mult in eta_p_mults:
                eta_w2 = float(base_eta_w2_init) * float(eta_w2_mult)
                eta_w3 = float(base_eta_w3_init) * float(eta_w3_mult)
                eta_p = float(base_eta_p_init) * float(eta_p_mult)
                monkeypatch.setattr(partition_mod, "_ETA_W2_INIT", eta_w2)
                monkeypatch.setattr(partition_mod, "_ETA_W3_INIT", eta_w3)
                monkeypatch.setattr(partition_mod, "_ETA_P_INIT", eta_p)

                t0 = time.perf_counter()
                res = compute_partition(imap, n_people, mode="convex")
                dt = time.perf_counter() - t0
                f = float(res["fairness"])
                assert 0.0 <= f <= 1.0
                per_cfg_rows.append(
                    {
                        "eta_w2_init": eta_w2,
                        "eta_w3_init": eta_w3,
                        "eta_p_init": eta_p,
                        "fairness": f,
                        "runtime_s": dt,
                    }
                )
                if f > best_fairness:
                    best_fairness = f
                    best_cfg = {
                        "eta_w2_init": eta_w2,
                        "eta_w3_init": eta_w3,
                        "eta_p_init": eta_p,
                    }

    # Reset to baseline constants so the next case baseline isn't affected.
    monkeypatch.setattr(partition_mod, "_ETA_W2_INIT", float(base_eta_w2_init))
    monkeypatch.setattr(partition_mod, "_ETA_W3_INIT", float(base_eta_w3_init))
    monkeypatch.setattr(partition_mod, "_ETA_P_INIT", float(base_eta_p_init))

    return best_fairness, best_cfg, per_cfg_rows


CASE_SPECS: list[CaseSpec] = [
    CaseSpec("c_circle_uniform", seed=1, dish_kind="circle", ingredient_kind="uniform_noisy", k_ingredients=4, n_people=3),
    CaseSpec("c_ellipse_pepperoni", seed=2, dish_kind="ellipse", ingredient_kind="pepperoni_like", k_ingredients=5, n_people=4),
    CaseSpec("c_donut_angular", seed=3, dish_kind="donut", ingredient_kind="angular_heterogeneous", k_ingredients=4, n_people=4),
    CaseSpec("c_crescent_pepperoni_sparse", seed=4, dish_kind="crescent", ingredient_kind="pepperoni_like_sparse", k_ingredients=6, n_people=4),
    CaseSpec("c_bridge_stripes", seed=5, dish_kind="bridge", ingredient_kind="stripes_and_gradient", k_ingredients=4, n_people=3),
    CaseSpec("c_irregular_angular", seed=6, dish_kind="irregular", ingredient_kind="angular_heterogeneous", k_ingredients=5, n_people=4),
    CaseSpec("c_circle_pepperoni_sparse", seed=7, dish_kind="circle", ingredient_kind="pepperoni_like_sparse", k_ingredients=4, n_people=2),
    CaseSpec("c_ellipse_stripes_gradient", seed=8, dish_kind="ellipse", ingredient_kind="stripes_and_gradient", k_ingredients=6, n_people=3),
    CaseSpec("u_circle_simple2", seed=9, dish_kind="circle", ingredient_kind="uniform_noisy", k_ingredients=2, n_people=2),
    CaseSpec("u_ellipse_uniform5", seed=10, dish_kind="ellipse", ingredient_kind="uniform_noisy", k_ingredients=5, n_people=5),
    CaseSpec("i_irregular_pepperoni", seed=11, dish_kind="irregular", ingredient_kind="pepperoni_like", k_ingredients=6, n_people=3),
    CaseSpec("d_donut_pepperoni_sparse", seed=12, dish_kind="donut", ingredient_kind="pepperoni_like_sparse", k_ingredients=5, n_people=4),
    CaseSpec("c_circle_stripes_grad", seed=13, dish_kind="circle", ingredient_kind="stripes_and_gradient", k_ingredients=6, n_people=4),
    CaseSpec("e_ellipse_angular7", seed=14, dish_kind="ellipse", ingredient_kind="angular_heterogeneous", k_ingredients=7, n_people=3),
    CaseSpec("b_bridge_pepperoni", seed=15, dish_kind="bridge", ingredient_kind="pepperoni_like", k_ingredients=5, n_people=2),
    CaseSpec("irregular_uniform6", seed=16, dish_kind="irregular", ingredient_kind="uniform_noisy", k_ingredients=6, n_people=6),
]


@pytest.mark.slow
def test_convex_hyperparam_sweep_exhaustive(monkeypatch: pytest.MonkeyPatch):
    """
    Barrido exhaustivo (diseñado para que luego puedas elegir la mejor LR de fase 2/3).

    - No tiene asserts fuertes de "mejora", porque el fairness depende del caso.
    - Asegura invariantes: fairness en [0,1].
    - Calcula baseline vs mejor fairness dentro de una grilla y guarda una mini-reporte en stdout.

    Nota importante sobre el código actual:
    - `_ETA_W_INIT` controla tanto fase2 (weights) como fase3 (eta_w para weights), así que están acopladas.
    - `_ETA_P_INIT` controla fase3 para las posiciones (eta_p).
    """
    run_mode = os.getenv("FAIRS_SLICE_RUN_HYPER_SWEEP", "0") == "1"
    if not run_mode:
        pytest.skip("set FAIRS_SLICE_RUN_HYPER_SWEEP=1 to run the exhaustive sweep")

    # Default grids can be heavy; tweak env vars to make it manageable locally.
    # We intentionally keep this moderate since we now sweep 3 dimensions:
    # eta_w2 (phase2 weights LR), eta_w3 (phase3 weights LR), eta_p (phase3 positions LR)
    eta_w2_mults = [0.5, 1.0, 1.5, 2.0]
    eta_w3_mults = [0.5, 1.0, 1.5, 2.0]
    eta_p_mults = [0.5, 1.0, 1.5]

    grid_w = os.getenv("FAIRS_SLICE_ETA_W_MULTS")  # legacy name: applies to both eta_w2/eta_w3
    grid_w2 = os.getenv("FAIRS_SLICE_ETA_W2_MULTS")
    grid_w3 = os.getenv("FAIRS_SLICE_ETA_W3_MULTS")
    grid_p = os.getenv("FAIRS_SLICE_ETA_P_MULTS")

    def _parse_csv_floats(v: str) -> list[float]:
        return [float(x.strip()) for x in v.split(",") if x.strip()]

    if grid_w2:
        eta_w2_mults = _parse_csv_floats(grid_w2)
    elif grid_w:
        eta_w2_mults = _parse_csv_floats(grid_w)

    if grid_w3:
        eta_w3_mults = _parse_csv_floats(grid_w3)
    elif grid_w:
        eta_w3_mults = _parse_csv_floats(grid_w)

    if grid_p:
        eta_p_mults = _parse_csv_floats(grid_p)

    h = int(os.getenv("FAIRS_SLICE_CASE_H", "48"))
    w = int(os.getenv("FAIRS_SLICE_CASE_W", "48"))

    case_specs = CASE_SPECS
    max_cases = int(os.getenv("FAIRS_SLICE_MAX_CASES", str(len(case_specs))))
    case_specs = case_specs[:max_cases]

    report = {
        "eta_w2_mults": eta_w2_mults,
        "eta_w3_mults": eta_w3_mults,
        "eta_p_mults": eta_p_mults,
        "cases": [],
        "global_cfg_agg": {},
    }

    base_eta_w2 = float(partition_mod._ETA_W2_INIT)
    base_eta_w3 = float(partition_mod._ETA_W3_INIT)
    base_eta_p = float(partition_mod._ETA_P_INIT)

    for cs in case_specs:
        imap, n_people = _make_case(cs, h=h, w=w)

        # Baseline with default hyperparams:
        monkeypatch.setattr(partition_mod, "_ETA_W2_INIT", float(base_eta_w2))
        monkeypatch.setattr(partition_mod, "_ETA_W3_INIT", float(base_eta_w3))
        monkeypatch.setattr(partition_mod, "_ETA_P_INIT", float(base_eta_p))
        t0 = time.perf_counter()
        base_res = compute_partition(imap, n_people, mode="convex")
        base_dt = time.perf_counter() - t0
        base_f = float(base_res["fairness"])
        assert 0.0 <= base_f <= 1.0

        best_f, best_cfg, per_cfg_rows = _sweep_convex_hyperparams(
            monkeypatch=monkeypatch,
            imap=imap,
            n_people=n_people,
            base_eta_w2_init=float(base_eta_w2),
            base_eta_w3_init=float(base_eta_w3),
            base_eta_p_init=float(base_eta_p),
            eta_w2_mults=eta_w2_mults,
            eta_w3_mults=eta_w3_mults,
            eta_p_mults=eta_p_mults,
        )
        assert best_f >= base_f - 1e-12  # grid includes (1.0,1.0) so best can't be worse

        # Aggregate all configurations across cases for an overall recommendation.
        for row in per_cfg_rows:
            key = f"{row['eta_w2_init']:.6g},{row['eta_w3_init']:.6g},{row['eta_p_init']:.6g}"
            agg = report["global_cfg_agg"].get(
                key,
                {
                    "eta_w2_init": row["eta_w2_init"],
                    "eta_w3_init": row["eta_w3_init"],
                    "eta_p_init": row["eta_p_init"],
                    "n": 0,
                    "fairness_sum": 0.0,
                    "runtime_sum": 0.0,
                },
            )
            agg["n"] += 1
            agg["fairness_sum"] += float(row["fairness"])
            agg["runtime_sum"] += float(row["runtime_s"])
            report["global_cfg_agg"][key] = agg

        report["cases"].append(
            {
                "name": cs.name,
                "dish_kind": cs.dish_kind,
                "ingredient_kind": cs.ingredient_kind,
                "seed": cs.seed,
                "k_ingredients": cs.k_ingredients,
                "n_people": cs.n_people,
                "baseline": {
                    "fairness": base_f,
                    "runtime_s": base_dt,
                    "eta_w2_init": base_eta_w2,
                    "eta_w3_init": base_eta_w3,
                    "eta_p_init": base_eta_p,
                },
                "best": {"fairness": best_f, **best_cfg},
            }
        )

    # Print a compact summary.
    avg_base = float(np.mean([c["baseline"]["fairness"] for c in report["cases"]]))
    avg_best = float(np.mean([c["best"]["fairness"] for c in report["cases"]]))
    best_overall = max(c["best"]["fairness"] for c in report["cases"])

    # Rank configs by average fairness; use average runtime as tie-breaker.
    global_cfg_list = list(report["global_cfg_agg"].values())
    for cfg in global_cfg_list:
        cfg["fairness_avg"] = cfg["fairness_sum"] / max(1, cfg["n"])
        cfg["runtime_avg_s"] = cfg["runtime_sum"] / max(1, cfg["n"])
    global_cfg_list.sort(key=lambda x: (x["fairness_avg"], -x["runtime_avg_s"]), reverse=True)
    report["top_configs"] = global_cfg_list[:10]

    report["summary"] = {
        "avg_baseline_fairness": avg_base,
        "avg_best_fairness": avg_best,
        "best_overall_fairness": float(best_overall),
        "n_cases": len(report["cases"]),
    }

    print("[hyperparam-sweep] summary:", json.dumps(report["summary"], indent=2))
    print("[hyperparam-sweep] top_configs:", json.dumps(report["top_configs"], indent=2))
    # Save full report for later analysis (optional).
    out_path = os.getenv("FAIRS_SLICE_SWEEP_OUT", "")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f)

