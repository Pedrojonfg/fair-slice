import os
from pathlib import Path

import pytest


@pytest.mark.slow
def test_real_images_pipeline_smoke():
    """
    Smoke test for real images.

    Disabled by default because vision.segment_dish depends on Vertex AI creds.
    To enable:
      RUN_REAL_IMAGES=1 pytest -m slow tests/integration/test_real_images_smoke.py -q
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
    for img in sample:
        imap, labels = segment_dish(str(img))
        assert imap.ndim == 3 and imap.dtype.name == "float32"
        assert len(labels) == imap.shape[-1]
        r = compute_partition(imap, 4, mode="free")
        assert 0.0 <= float(r["fairness"]) <= 1.0

