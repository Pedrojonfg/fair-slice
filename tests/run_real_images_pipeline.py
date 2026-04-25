"""
Run a detailed end-to-end pipeline check on real pizza photos.

It processes every image under:
    tests/fixtures/real_images/

For each image it can run:
  1) vision.segment_dish()      -> ingredient_map + labels
  2) partition.compute_partition() for multiple N / modes
  3) visualize.render_partition()  -> overlay PNG

It writes a per-image report + intermediate artifacts under:
    tests/fixtures/real_images_outputs/<image_stem>/

This is intentionally a *script* (not a unit test) because Vision depends on
Vertex AI credentials and network access.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "tests" / "fixtures" / "real_images"
OUT_DIR = ROOT / "tests" / "fixtures" / "real_images_outputs"


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class StageResult:
    ok: bool
    runtime_s: float
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    extra: dict[str, Any] | None = None


def _run_stage(fn, *args, **kwargs) -> tuple[StageResult, Any]:
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
        return StageResult(ok=True, runtime_s=time.perf_counter() - t0, extra={}), out
    except Exception as e:  # noqa: BLE001 (script wants full diagnostics)
        return (
            StageResult(
                ok=False,
                runtime_s=time.perf_counter() - t0,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                extra={},
            ),
            None,
        )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _summarize_imap(ingredient_map: np.ndarray, labels: dict[int, str]) -> dict[str, Any]:
    H, W, K = ingredient_map.shape
    total = ingredient_map.sum(axis=-1)
    dish = total > 1e-3
    dish_pixels = int(dish.sum())
    channel_sums = ingredient_map[dish].sum(axis=1) if dish_pixels > 0 else np.array([], dtype=np.float32)
    return {
        "shape": [H, W, K],
        "dtype": str(ingredient_map.dtype),
        "min": float(ingredient_map.min(initial=0.0)),
        "max": float(ingredient_map.max(initial=0.0)),
        "dish_pixels": dish_pixels,
        "channel_sum_mean": float(channel_sums.mean()) if dish_pixels > 0 else None,
        "channel_sum_std": float(channel_sums.std()) if dish_pixels > 0 else None,
        "labels": {str(k): v for k, v in labels.items()},
    }


def run_one_image(
    image_path: Path,
    *,
    people: list[int],
    modes: list[str],
    do_visualize: bool,
    use_cache: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """
    Returns a rich dict suitable for JSON.
    """
    # Lazy imports so the script can still print a clean error if deps missing.
    from vision import segment_dish
    from partition import compute_partition
    from visualize import render_partition

    img_stem = image_path.stem
    out_dir = OUT_DIR / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_map = out_dir / "ingredient_map.npy"
    cache_labels = out_dir / "labels.json"

    report: dict[str, Any] = {
        "image": str(image_path),
        "started_at_ms": _now_ms(),
        "vision": None,
        "partition": [],
        "visualize": [],
        "artifacts_dir": str(out_dir),
    }

    # -----------------------
    # Stage 1: Vision
    # -----------------------
    ingredient_map = None
    labels = None

    if use_cache and cache_map.exists() and cache_labels.exists() and not overwrite:
        t0 = time.perf_counter()
        ingredient_map = np.load(cache_map)
        labels = json.loads(cache_labels.read_text(encoding="utf-8"))
        labels = {int(k): str(v) for k, v in labels.items()}
        report["vision"] = StageResult(ok=True, runtime_s=time.perf_counter() - t0, extra={"cached": True}).__dict__
    else:
        vision_res, out = _run_stage(segment_dish, str(image_path))
        if vision_res.ok:
            ingredient_map, labels = out
            vision_res.extra = {"cached": False, **_summarize_imap(ingredient_map, labels)}
            np.save(cache_map, ingredient_map)
            _write_json(cache_labels, labels)
        report["vision"] = vision_res.__dict__

    if ingredient_map is None or labels is None:
        report["ended_at_ms"] = _now_ms()
        return report

    # -----------------------
    # Stage 2: Partition
    # -----------------------
    for N in people:
        for mode in modes:
            part_res, result = _run_stage(compute_partition, ingredient_map, N, mode=mode)
            entry: dict[str, Any] = {
                "N": N,
                "mode": mode,
                **part_res.__dict__,
            }
            if part_res.ok:
                # save artifacts
                entry["fairness"] = float(result["fairness"])
                entry["scores_shape"] = list(result["scores"].shape)
                np.save(out_dir / f"scores_N{N}_{mode}.npy", result["scores"])
                np.save(out_dir / f"seeds_N{N}_{mode}.npy", result["seeds"])
                _write_json(out_dir / f"partition_N{N}_{mode}.json", {
                    "fairness": float(result["fairness"]),
                    "scores": result["scores"].tolist(),
                    "labels": {str(k): v for k, v in labels.items()},
                })
            report["partition"].append(entry)

            # -----------------------
            # Stage 3: Visualize (optional)
            # -----------------------
            if do_visualize and part_res.ok:
                vis_res, img = _run_stage(
                    render_partition,
                    image_path=str(image_path),
                    masks=result["masks"],
                    scores=result["scores"],
                    ingredient_labels=labels,
                    fairness=float(result["fairness"]),
                )
                vis_entry = {
                    "N": N,
                    "mode": mode,
                    **vis_res.__dict__,
                }
                if vis_res.ok:
                    png_path = out_dir / f"overlay_N{N}_{mode}.png"
                    img.save(png_path)
                    vis_entry["overlay_path"] = str(png_path)
                report["visualize"].append(vis_entry)

    report["ended_at_ms"] = _now_ms()
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", default="3,4", help="Comma-separated N values (e.g. 3,4,6,8)")
    ap.add_argument("--modes", default="free,radial", help="Comma-separated modes (free,radial)")
    ap.add_argument("--no-visualize", action="store_true", help="Skip visualize stage")
    ap.add_argument("--use-cache", action="store_true", help="Reuse cached ingredient_map.npy if present")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite cached artifacts")
    ap.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all)")
    args = ap.parse_args()

    if not REAL_DIR.exists():
        print(f"[real_images] Directory not found: {REAL_DIR}")
        print("[real_images] Create it and add images to run this script.")
        return 2

    images = sorted([p for p in REAL_DIR.iterdir() if p.is_file() and _is_image(p)])
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    if not images:
        print(f"[real_images] No images found in {REAL_DIR}")
        return 2

    # This stage is the common failure point; surface it early.
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("[real_images] Missing GOOGLE_CLOUD_PROJECT; vision.segment_dish will fail.")
        print("[real_images] Either set credentials, or run with --use-cache if you already generated .npy files.")

    # Ensure imports work (prints clearer error than stacktrace later).
    try:
        import vision  # noqa: F401
        import partition  # noqa: F401
        import visualize  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[real_images] Import error: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return 3

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_reports = []
    for img in images:
        print(f"[real_images] Processing {img.name} …")
        rep = run_one_image(
            img,
            people=[int(x) for x in args.people.split(",") if x.strip()],
            modes=[m.strip() for m in args.modes.split(",") if m.strip()],
            do_visualize=not args.no_visualize,
            use_cache=bool(args.use_cache),
            overwrite=bool(args.overwrite),
        )
        all_reports.append(rep)
        _write_json(OUT_DIR / f"{img.stem}.report.json", rep)
        print(f"[real_images] Wrote {OUT_DIR / f'{img.stem}.report.json'}")

    summary_path = OUT_DIR / "_summary.json"
    _write_json(summary_path, {"reports": all_reports})
    print(f"[real_images] Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

