import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.mark.slow
def test_vision_runtime_real_images_report():
    """
    Runtime breakdown report for real images.

    Disabled by default because vision depends on Vertex AI / Gemini creds.
    Enable with:
      RUN_REAL_IMAGES=1 pytest -m slow -s tests/integration/test_vision_runtime.py -q
    """
    if os.environ.get("RUN_REAL_IMAGES") != "1":
        pytest.skip("Set RUN_REAL_IMAGES=1 to enable real-images runtime report.")

    root = Path(__file__).resolve().parents[2]
    real_dir = root / "tests" / "fixtures" / "real_images"
    if not real_dir.exists():
        pytest.skip("tests/fixtures/real_images not found")

    imgs = [
        p
        for p in real_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    if not imgs:
        pytest.skip("No real images found")

    out_dir = root / "tests" / "fixtures" / "real_images_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "vision_runtime_report.json"

    # Import inside the test so skips happen cleanly.
    import vision as vision_mod

    rows: list[dict] = []

    print("\n==============================")
    print("VISION RUNTIME REPORT (REAL IMAGES)")
    print("==============================\n")

    for img in imgs:
        image_path = str(img)
        filename = img.name

        t_total0 = time.perf_counter()

        # --- t_load: cv2.imread + resize (same logic as segment_dish) ---
        t0 = time.perf_counter()
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            pytest.fail(f"Cannot load image from '{image_path}'")

        H_orig, W_orig = img_bgr.shape[:2]
        scale = vision_mod.TARGET_SIZE / max(H_orig, W_orig)
        if scale < 1.0:
            new_w = int(W_orig * scale)
            new_h = int(H_orig * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        t_load = time.perf_counter() - t0

        # --- t_perspective: correct_perspective ---
        t0 = time.perf_counter()
        img_bgr = vision_mod.correct_perspective(img_bgr)
        t_perspective = time.perf_counter() - t0
        H, W = img_bgr.shape[:2]

        # Model is created before dish mask + ingredients (as in segment_dish).
        model = vision_mod._init_model()

        # --- t_dish_mask: Gemini polygon dish detection ---
        t0 = time.perf_counter()
        dish_mask = vision_mod._detect_dish_mask(img_bgr, model, image_path)
        t_dish_mask = time.perf_counter() - t0

        # --- t_gemini_ingredients: ingredient identification ---
        t0 = time.perf_counter()
        ingredients = vision_mod._call_gemini(model, image_path)
        t_gemini_ingredients = time.perf_counter() - t0

        K = len(ingredients)

        # --- t_segmentation: loop over _segment_ingredient (all channels) ---
        t0 = time.perf_counter()
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        ingredient_map = np.zeros((H, W, K), dtype=np.float32)
        for k, ing in enumerate(ingredients):
            density = vision_mod._segment_ingredient(
                hsv,
                dish_mask,
                ingredient_name=ing["name"],
                color_description=ing.get("color_description", ""),
            )
            ingredient_map[:, :, k] = density
        t_segmentation = time.perf_counter() - t0

        # --- t_normalize: _normalize_map + channel filtering (same as segment_dish) ---
        t0 = time.perf_counter()
        pixel_total = ingredient_map.sum(axis=2)
        unclaimed = (pixel_total < 0.05) & dish_mask
        if K > 0:
            ingredient_map[unclaimed, 0] = 0.3

        # Base ingredient cumulative over whole dish before normalization.
        if K > 0:
            ingredient_map[:, :, 0][dish_mask] = 1.0
        ingredient_map = vision_mod._normalize_map(ingredient_map, dish_mask)

        dish_pixels = int(dish_mask.sum())
        if dish_pixels > 0 and K > 1:
            to_drop: list[int] = []
            for k in range(K):
                mean_k = float(ingredient_map[:, :, k][dish_mask].mean())
                if mean_k < vision_mod.MIN_CHANNEL_MEAN_COVERAGE:
                    if k == 0:
                        continue
                    to_drop.append(k)

            if to_drop:
                keep = [k for k in range(K) if k not in set(to_drop)]
                ingredient_map = ingredient_map[:, :, keep]
                ingredient_map = vision_mod._normalize_map(ingredient_map, dish_mask)
                K = ingredient_map.shape[-1]
        t_normalize = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total0

        rows.append(
            {
                "image": {"filename": filename, "path": image_path},
                "times_s": {
                    "t_load": float(t_load),
                    "t_perspective": float(t_perspective),
                    "t_dish_mask": float(t_dish_mask),
                    "t_gemini_ingredients": float(t_gemini_ingredients),
                    "t_segmentation": float(t_segmentation),
                    "t_normalize": float(t_normalize),
                    "t_total": float(t_total),
                },
            }
        )

    # ---- Print required table ----
    cols = [("image", 28), ("t_dish_mask", 12), ("t_gemini_ing", 16), ("t_segmentation", 14), ("t_total", 10)]
    header = " | ".join(f"{name:<{w}}" for name, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        t = r["times_s"]
        line = " | ".join(
            [
                f"{r['image']['filename']:<28}"[:28],
                f"{t['t_dish_mask']:<12.3f}",
                f"{t['t_gemini_ingredients']:<16.3f}",
                f"{t['t_segmentation']:<14.3f}",
                f"{t['t_total']:<10.3f}",
            ]
        )
        print(line)

    # ---- Bottleneck summary ----
    stage_keys = [
        "t_load",
        "t_perspective",
        "t_dish_mask",
        "t_gemini_ingredients",
        "t_segmentation",
        "t_normalize",
    ]
    means = {k: float(np.mean([row["times_s"][k] for row in rows])) for k in stage_keys}
    bottleneck_stage, bottleneck_mean = max(means.items(), key=lambda kv: kv[1])
    print(f"\nBottleneck: {bottleneck_stage} ({bottleneck_mean:.3f}s de media)")

    report = {
        "meta": {
            "run_required": {"RUN_REAL_IMAGES": "1", "pytest_marker": "slow"},
            "images_dir": str(real_dir),
            "output_path": str(report_path),
            "n_images": len(rows),
        },
        "images": rows,
        "aggregate": {
            "mean_times_s": means,
            "bottleneck": {"stage": bottleneck_stage, "mean_s": bottleneck_mean},
        },
        "table_columns": ["image", "t_dish_mask", "t_gemini_ingredients", "t_segmentation", "t_total"],
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved at: {report_path}")

