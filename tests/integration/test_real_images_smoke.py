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
        print(f"Imagen: {filename}")
        print("------------------------------")

        t0 = time.perf_counter()
        imap, labels = segment_dish(img_path)
        t_vision = time.perf_counter() - t0

        # Aserciones requeridas (no tocar)
        assert imap.ndim == 3 and imap.dtype.name == "float32"
        assert len(labels) == imap.shape[-1]
        # Aserción nueva requerida
        assert float(imap.sum(axis=-1).max()) > 0

        H, W, K = imap.shape
        labels_dict = {int(k): str(v) for k, v in labels.items()}

        dish_mask = imap.sum(axis=-1) > 1e-6
        dish_pixels = int(dish_mask.sum())
        if dish_pixels == 0:
            pytest.fail("Dish mask has 0 pixels (unexpected with imap.sum(axis=-1).max() > 0).")

        channel_stats: list[dict] = []
        low_coverage_channels: list[int] = []
        for k in range(K):
            channel = imap[:, :, k]
            cov = float(((channel > 0.1) & dish_mask).sum()) / float(dish_pixels)
            mean_val = float(channel[dish_mask].mean())
            name = labels_dict.get(k, f"channel_{k}")
            flag = " ⚠️ BAJA COBERTURA" if cov < 0.05 else ""
            if cov < 0.05:
                low_coverage_channels.append(k)
            print(f"Canal {k} ({name}): cobertura={cov*100:.1f}%, media={mean_val:.3f}{flag}")
            channel_stats.append(
                {
                    "k": k,
                    "name": name,
                    "coverage": cov,
                    "mean": mean_val,
                    "low_coverage": cov < 0.05,
                }
            )

        print(f"\nRuntime segment_dish: {t_vision:.3f}s")
        print(f"imap shape: ({H}, {W}, {K})")
        print(f"labels: {labels_dict}")

        runs: dict[str, dict] = {}
        for N in (3, 4):
            t1 = time.perf_counter()
            r = compute_partition(imap, N, mode="free")
            t_part = time.perf_counter() - t1

            fairness = float(r["fairness"])
            scores = np.asarray(r["scores"], dtype=float)

            # Aserción requerida (no tocar)
            assert 0.0 <= float(r["fairness"]) <= 1.0

            fairness_flag = " ⚠️ FAIRNESS BAJA" if fairness < 0.6 else ""
            print(f"\ncompute_partition N={N} (free): runtime={t_part:.3f}s, fairness={fairness:.3f}{fairness_flag}")

            # Scores matrix (N,K) con nombres de ingredientes en columnas
            col_names = [labels_dict.get(k, f"channel_{k}") for k in range(K)]
            header = "person\\ingr | " + " | ".join(col_names)
            print("\nScores (N,K):")
            print(header)
            print("-" * len(header))
            for i in range(scores.shape[0]):
                row = " | ".join(f"{scores[i, k]:.3f}" for k in range(K))
                print(f"{i:>10} | {row}")

            # Fairness por ingrediente (desviación del ideal 1/N)
            ideal = 1.0 / float(N)
            per_ing = {}
            print("\nDesviación por ingrediente (max(|scores[:,k] - 1/N|)):")
            for k in range(K):
                name = labels_dict.get(k, f"channel_{k}")
                dev = float(np.max(np.abs(scores[:, k] - ideal)))
                per_ing[k] = {"name": name, "max_abs_dev": dev, "ideal": ideal}
                print(f"- {name}: {dev:.3f}")
                ingredient_deviation_accum.setdefault(name, []).append(dev)

            runs[str(N)] = {
                "N": N,
                "runtime_s": t_part,
                "fairness": fairness,
                "scores": scores.tolist(),
                "ingredient_columns": col_names,
                "ingredient_fairness_deviation": per_ing,
                "flags": {
                    "low_fairness": fairness < 0.6,
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
                "fairness_N3": float(runs["3"]["fairness"]),
                "fairness_N4": float(runs["4"]["fairness"]),
                "runtime_vision_s": float(t_vision),
                "runtime_part_N3_s": float(runs["3"]["runtime_s"]),
                "runtime_part_N4_s": float(runs["4"]["runtime_s"]),
            }
        )

    # ---- Resumen agregado ----
    print("\n\n==============================")
    print("RESUMEN AGREGADO")
    print("==============================\n")

    # Tabla: imagen | K | N=3 fairness | N=4 fairness | runtime vision | runtime partition N=3 | runtime partition N=4
    cols = [
        ("imagen", 24),
        ("K", 3),
        ("N=3 fairness", 12),
        ("N=4 fairness", 12),
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
                f"{s['runtime_vision_s']:<10.3f}",
                f"{s['runtime_part_N3_s']:<12.3f}",
                f"{s['runtime_part_N4_s']:<12.3f}",
            ]
        )
        print(line)

    fairness_mean_n3 = float(np.mean([s["fairness_N3"] for s in summary_rows])) if summary_rows else float("nan")
    fairness_mean_n4 = float(np.mean([s["fairness_N4"] for s in summary_rows])) if summary_rows else float("nan")
    rt_vision_mean = float(np.mean([s["runtime_vision_s"] for s in summary_rows])) if summary_rows else float("nan")
    rt_part_n3_mean = float(np.mean([s["runtime_part_N3_s"] for s in summary_rows])) if summary_rows else float("nan")
    rt_part_n4_mean = float(np.mean([s["runtime_part_N4_s"] for s in summary_rows])) if summary_rows else float("nan")

    hardest_name = None
    hardest_avg = None
    if ingredient_deviation_accum:
        avg_dev = {name: float(np.mean(vals)) for name, vals in ingredient_deviation_accum.items() if vals}
        hardest_name, hardest_avg = max(avg_dev.items(), key=lambda kv: kv[1])

    print(f"\nFairness media: N=3 -> {fairness_mean_n3:.3f}, N=4 -> {fairness_mean_n4:.3f}")
    print(
        "Runtime medio: "
        f"vision -> {rt_vision_mean:.3f}s, "
        f"partition N=3 -> {rt_part_n3_mean:.3f}s, "
        f"partition N=4 -> {rt_part_n4_mean:.3f}s"
    )
    if hardest_name is not None:
        print(f"Ingrediente más difícil (promedio desviación): {hardest_name} (avg={hardest_avg:.3f})")

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
    print(f"\nInforme guardado en: {report_path}")

