# Run with: RUN_REAL_IMAGES=1 pytest -m slow -s tests/integration/test_real_images_smoke.py
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.slow
def test_real_images_pipeline_smoke():
    """
    Smoke test for real images.

    Disabled by default because vision.segment_dish depends on Vertex AI creds.
    To enable:
      RUN_REAL_IMAGES=1 pytest -m slow -s tests/integration/test_real_images_smoke.py -q
    """
    if os.environ.get("RUN_REAL_IMAGES") != "1":
        pytest.skip("Set RUN_REAL_IMAGES=1 to enable real-images smoke test.")

    root = Path(__file__).resolve().parents[2]
    real_dir = root / "tests" / "fixtures" / "real_images"
    if not real_dir.exists():
        pytest.skip("tests/fixtures/real_images not found")

    imgs = [p for p in real_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not imgs:
        pytest.skip("No real images found")

    # Import inside test for clearer skip/failure behavior.
    from vision import segment_dish
    from partition import compute_partition

    # Process a small sample to keep CI sane.
    sample = imgs[: min(3, len(imgs))]
    out_dir = root / "tests" / "fixtures" / "real_images_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "smoke_report.json"

    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    ingredient_deviation_accum: dict[str, list[float]] = {}

    print("\n==============================")
    print("REAL IMAGES SMOKE REPORT")
    print("==============================\n")

    for img in sample:
        img_path = str(img)
        filename = img.name

        print("\n------------------------------")
        print(f"Image: {filename}")
        print("------------------------------")

        t0 = time.perf_counter()
        imap, labels = segment_dish(img_path)
        t_vision = time.perf_counter() - t0

        # Required assertions (do not change)
        assert imap.ndim == 3 and imap.dtype.name == "float32"
        assert len(labels) == imap.shape[-1]
        # New required assertion
        assert float(imap.sum(axis=-1).max()) > 0

        H, W, K = imap.shape
        labels_dict = {int(k): str(v) for k, v in labels.items()}

        dish_mask = imap.sum(axis=-1) > 1e-6
        dish_pixels = int(dish_mask.sum())
        if dish_pixels == 0:
            pytest.fail("Dish mask has 0 pixels (unexpected with imap.sum(axis=-1).max() > 0).")

        # Active channels: ignore near-empty channels in fairness reporting.
        import partition as partition_mod
        T_total = imap.reshape(-1, K).sum(axis=0)
        active_channels = T_total > partition_mod._MIN_CHANNEL_TOTAL_FRACTION * T_total.max()

        channel_stats: list[dict] = []
        low_coverage_channels: list[int] = []
        for k in range(K):
            channel = imap[:, :, k]
            cov = float(((channel > 0.1) & dish_mask).sum()) / float(dish_pixels)
            mean_val = float(channel[dish_mask].mean())
            name = labels_dict.get(k, f"channel_{k}")
            flag = " [LOW COVERAGE]" if cov < 0.05 else ""
            if cov < 0.05:
                low_coverage_channels.append(k)
            active = bool(active_channels[k])
            active_flag = "active" if active else "inactive"
            print(
                f"Channel {k} ({name}): coverage={cov*100:.1f}%, mean={mean_val:.3f}, {active_flag}{flag}"
            )
            channel_stats.append(
                {
                    "k": k,
                    "name": name,
                    "coverage": cov,
                    "mean": mean_val,
                    "low_coverage": cov < 0.05,
                    "active": active,
                }
            )

        print(f"\nRuntime segment_dish: {t_vision:.3f}s")
        print(f"imap shape: ({H}, {W}, {K})")
        print(f"labels: {labels_dict}")

        runs: dict[str, dict] = {}
        # Run a mix of scenarios:
        # - Uniform preferences at N=3 and N=4 (baseline)
        # - A larger partition N>4
        # - A non-uniform preference matrix (complex preferences)
        rng = np.random.default_rng(abs(hash(filename)) % (2**32))
        run_specs: list[dict] = [
            {"N": 3, "tag": "uniform", "preferences": None},
            {"N": 4, "tag": "uniform", "preferences": None},
            {"N": 6, "tag": "uniform", "preferences": None},
        ]
        if K >= 2:
            N_pref = 5
            P = rng.lognormal(mean=0.0, sigma=0.9, size=(N_pref, K)).astype(np.float32)
            # Make preferences structured and challenging (deterministic):
            # - One person strongly values the most concentrated non-base channel (if any)
            # - Another person strongly dislikes it (near-zero but non-negative)
            if K > 1:
                P[0, 1] *= 12.0
                P[1, 1] *= 0.05
            # - Encourage a couple of "specialists"
            if K > 2:
                P[2, 2] *= 8.0
            if K > 3:
                P[3, 3] *= 6.0
            run_specs.append({"N": N_pref, "tag": "complex_prefs", "preferences": P})

        for spec in run_specs:
            N = int(spec["N"])
            tag = str(spec["tag"])
            preferences = spec["preferences"]
            t1 = time.perf_counter()
            r = compute_partition(imap, N, mode="free", preferences=preferences)
            t_part = time.perf_counter() - t1

            fairness_relative = float(r["fairness"])
            scores = np.asarray(r["scores"], dtype=float)
            if preferences is None:
                P_norm = np.full((N, K), 1.0 / float(N), dtype=np.float64)
            else:
                P_norm = partition_mod._normalize_preferences(np.asarray(preferences), N, K)
            n_ing = int(active_channels.sum()) if bool(active_channels.any()) else int(K)
            fairness = float(partition_mod._compute_fairness(scores, P_norm, active_channels))
            fairness_complexity_adjusted = float(
                partition_mod._compute_fairness(scores, P_norm, active_channels, n_ingredients=n_ing)
            )
            fairness_relative_active = float(
                partition_mod._compute_fairness(scores, P_norm, active_channels, imap, dish_mask)
            )

            # Required assertion (do not change)
            assert 0.0 <= float(r["fairness"]) <= 1.0

            fairness_flag = " [LOW FAIRNESS]" if fairness_relative < 0.6 else ""
            print(
                f"\ncompute_partition N={N} (free, {tag}): runtime={t_part:.3f}s, "
                f"fairness={fairness:.3f}, fairness_complexity_adjusted={fairness_complexity_adjusted:.3f}, "
                f"fairness_relative={fairness_relative:.3f}, "
                f"fairness_relative_active={fairness_relative_active:.3f}{fairness_flag}"
            )

            # Scores matrix (N,K) with ingredient names in columns
            col_names = [labels_dict.get(k, f"channel_{k}") for k in range(K)]
            header = "person\\ingr | " + " | ".join(col_names)
            print("\nScores (N,K):")
            print(header)
            print("-" * len(header))
            for i in range(scores.shape[0]):
                row = " | ".join(f"{scores[i, k]:.3f}" for k in range(K))
                print(f"{i:>10} | {row}")

            # Per-ingredient deviations (uniform ideal and preference targets)
            ideal = 1.0 / float(N)
            per_ing = {}
            print("\nPer-ingredient deviation:")
            # Importance weights used by the relative fairness aggregation:
            # w_j ∝ sum_i P_norm[i, j], normalized over active channels.
            importance = P_norm.sum(axis=0).astype(float)  # (K,)
            importance_active_sum = float(importance[active_channels].sum())
            if importance_active_sum <= 1e-18:
                ingredient_importance = np.zeros(K, dtype=float)
                if bool(active_channels.any()):
                    ingredient_importance[active_channels] = 1.0 / float(active_channels.sum())
            else:
                ingredient_importance = np.zeros(K, dtype=float)
                ingredient_importance[active_channels] = importance[active_channels] / importance_active_sum
            for k in range(K):
                name = labels_dict.get(k, f"channel_{k}")
                dev_uniform = float(np.max(np.abs(scores[:, k] - ideal)))
                dev_target = float(np.max(np.abs(scores[:, k] - P_norm[:, k])))
                per_ing[k] = {
                    "name": name,
                    "max_abs_dev_uniform": dev_uniform,
                    "max_abs_dev_target": dev_target,
                    "ideal": ideal,
                    "active": bool(active_channels[k]),
                    "ingredient_importance": float(ingredient_importance[k]),
                }
                active_tag = "active" if active_channels[k] else "inactive"
                print(
                    f"- {name} ({active_tag}): "
                    f"dev_uniform={dev_uniform:.3f}, dev_target={dev_target:.3f} "
                    f"(importance={ingredient_importance[k]:.3f})"
                )
                ingredient_deviation_accum.setdefault(name, []).append(dev_uniform)

            run_key = f"N={N}_{tag}"
            runs[run_key] = {
                "N": N,
                "tag": tag,
                "runtime_s": t_part,
                "fairness": fairness,
                "fairness_complexity_adjusted": fairness_complexity_adjusted,
                "fairness_relative": fairness_relative,
                "fairness_relative_active": fairness_relative_active,
                "scores": scores.tolist(),
                "ingredient_columns": col_names,
                "ingredient_fairness_deviation": per_ing,
                "preferences": None if preferences is None else np.asarray(preferences).tolist(),
                "flags": {
                    "low_fairness": fairness_relative < 0.6,
                },
            }

        row = {
            "image": {"filename": filename, "path": img_path},
            "vision": {
                "runtime_s": t_vision,
                "imap_shape": [H, W, K],
                "labels": labels_dict,
            },
            "dish": {
                "dish_pixels": dish_pixels,
                "dish_pixel_fraction": float(dish_pixels) / float(H * W),
            },
            "channels": channel_stats,
            "runs": runs,
        }
        all_rows.append(row)

        summary_rows.append(
            {
                "image": filename,
                "K": K,
                "fairness_N3": float(runs["N=3_uniform"]["fairness"]),
                "fairness_N4": float(runs["N=4_uniform"]["fairness"]),
                "fairness_N3_complexity_adjusted": float(runs["N=3_uniform"]["fairness_complexity_adjusted"]),
                "fairness_N4_complexity_adjusted": float(runs["N=4_uniform"]["fairness_complexity_adjusted"]),
                "fairness_N3_relative": float(runs["N=3_uniform"]["fairness_relative"]),
                "fairness_N4_relative": float(runs["N=4_uniform"]["fairness_relative"]),
                "fairness_N3_relative_active": float(runs["N=3_uniform"]["fairness_relative_active"]),
                "fairness_N4_relative_active": float(runs["N=4_uniform"]["fairness_relative_active"]),
                "runtime_vision_s": float(t_vision),
                "runtime_part_N3_s": float(runs["N=3_uniform"]["runtime_s"]),
                "runtime_part_N4_s": float(runs["N=4_uniform"]["runtime_s"]),
            }
        )

    # ---- Resumen agregado ----
    print("\n\n==============================")
    print("AGGREGATE SUMMARY")
    print("==============================\n")

    # Table: image | K | N=3 fairness | N=4 fairness | N=3 fairness (active) | N=4 fairness (active) | runtimes
    cols = [
        ("image", 24),
        ("K", 3),
        ("N=3 fairness", 12),
        ("N=4 fairness", 12),
        ("N=3 fair adj", 12),
        ("N=4 fair adj", 12),
        ("N=3 fair rel", 12),
        ("N=4 fair rel", 12),
        ("N=3 rel act", 12),
        ("N=4 rel act", 12),
        ("rt vision", 10),
        ("rt part N=3", 12),
        ("rt part N=4", 12),
    ]
    header = " | ".join(f"{name:<{w}}" for name, w in cols)
    print(header)
    print("-" * len(header))
    for s in summary_rows:
        line = " | ".join(
            [
                f"{s['image']:<24}"[:24],
                f"{s['K']:<3}",
                f"{s['fairness_N3']:<12.3f}",
                f"{s['fairness_N4']:<12.3f}",
                f"{s['fairness_N3_complexity_adjusted']:<12.3f}",
                f"{s['fairness_N4_complexity_adjusted']:<12.3f}",
                f"{s['fairness_N3_relative']:<12.3f}",
                f"{s['fairness_N4_relative']:<12.3f}",
                f"{s['fairness_N3_relative_active']:<12.3f}",
                f"{s['fairness_N4_relative_active']:<12.3f}",
                f"{s['runtime_vision_s']:<10.3f}",
                f"{s['runtime_part_N3_s']:<12.3f}",
                f"{s['runtime_part_N4_s']:<12.3f}",
            ]
        )
        print(line)

    fairness_mean_n3 = float(np.mean([s["fairness_N3"] for s in summary_rows])) if summary_rows else float("nan")
    fairness_mean_n4 = float(np.mean([s["fairness_N4"] for s in summary_rows])) if summary_rows else float("nan")
    fairness_mean_n3_complexity_adjusted = float(np.mean([s["fairness_N3_complexity_adjusted"] for s in summary_rows])) if summary_rows else float("nan")
    fairness_mean_n4_complexity_adjusted = float(np.mean([s["fairness_N4_complexity_adjusted"] for s in summary_rows])) if summary_rows else float("nan")
    fairness_mean_n3_relative = float(np.mean([s["fairness_N3_relative"] for s in summary_rows])) if summary_rows else float("nan")
    fairness_mean_n4_relative = float(np.mean([s["fairness_N4_relative"] for s in summary_rows])) if summary_rows else float("nan")
    fairness_mean_n3_relative_active = float(np.mean([s["fairness_N3_relative_active"] for s in summary_rows])) if summary_rows else float("nan")
    fairness_mean_n4_relative_active = float(np.mean([s["fairness_N4_relative_active"] for s in summary_rows])) if summary_rows else float("nan")
    rt_vision_mean = float(np.mean([s["runtime_vision_s"] for s in summary_rows])) if summary_rows else float("nan")
    rt_part_n3_mean = float(np.mean([s["runtime_part_N3_s"] for s in summary_rows])) if summary_rows else float("nan")
    rt_part_n4_mean = float(np.mean([s["runtime_part_N4_s"] for s in summary_rows])) if summary_rows else float("nan")

    hardest_name = None
    hardest_avg = None
    if ingredient_deviation_accum:
        avg_dev = {name: float(np.mean(vals)) for name, vals in ingredient_deviation_accum.items() if vals}
        hardest_name, hardest_avg = max(avg_dev.items(), key=lambda kv: kv[1])

    print(
        f"\nFairness mean: N=3 -> {fairness_mean_n3:.3f}, N=4 -> {fairness_mean_n4:.3f} | "
        f"fairness_mean_complexity_adjusted: N=3 -> {fairness_mean_n3_complexity_adjusted:.3f}, N=4 -> {fairness_mean_n4_complexity_adjusted:.3f} | "
        f"fairness_relative: N=3 -> {fairness_mean_n3_relative:.3f}, N=4 -> {fairness_mean_n4_relative:.3f} | "
        f"fairness_relative_active: N=3 -> {fairness_mean_n3_relative_active:.3f}, N=4 -> {fairness_mean_n4_relative_active:.3f}"
    )
    print(
        "Runtime mean: "
        f"vision -> {rt_vision_mean:.3f}s, "
        f"partition N=3 -> {rt_part_n3_mean:.3f}s, "
        f"partition N=4 -> {rt_part_n4_mean:.3f}s"
    )
    if hardest_name is not None:
        print(f"Hardest ingredient (avg deviation): {hardest_name} (avg={hardest_avg:.3f})")

    report = {
        "meta": {
            "run_required": {"RUN_REAL_IMAGES": "1", "pytest_marker": "slow", "note": "Run with -s to see full report."},
            "n_images_processed": len(all_rows),
            "images_dir": str(real_dir),
            "output_path": str(report_path),
        },
        "images": all_rows,
        "aggregate": {
            "table": summary_rows,
            "fairness_mean": {"N=3": fairness_mean_n3, "N=4": fairness_mean_n4},
            "fairness_mean_complexity_adjusted": {"N=3": fairness_mean_n3_complexity_adjusted, "N=4": fairness_mean_n4_complexity_adjusted},
            "fairness_mean_relative": {"N=3": fairness_mean_n3_relative, "N=4": fairness_mean_n4_relative},
            "fairness_mean_relative_active": {"N=3": fairness_mean_n3_relative_active, "N=4": fairness_mean_n4_relative_active},
            "runtime_mean_s": {
                "vision": rt_vision_mean,
                "partition_N3": rt_part_n3_mean,
                "partition_N4": rt_part_n4_mean,
            },
            "hardest_ingredient_avg_deviation": (
                {"name": hardest_name, "avg_max_abs_dev": hardest_avg} if hardest_name is not None else None
            ),
        },
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved at: {report_path}")

