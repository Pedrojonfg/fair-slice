"""
generate_mock_data.py
Generates synthetic test fixtures so the team can develop and test
without needing the real Vertex AI API.

Outputs:
  tests/fixtures/mock_data/pizza_mock_500.npy    — (500,500,5) ingredient map
  tests/fixtures/mock_data/pizza_labels.npy      — ingredient label dict
  tests/fixtures/sample_images/pizza_test.jpg    — synthetic pizza photo (500x500)

Usage:
  python tests/generate_mock_data.py
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_DIR   = os.path.join(ROOT, "tests", "fixtures", "mock_data")
IMAGES_DIR = os.path.join(ROOT, "tests", "fixtures", "sample_images")
os.makedirs(MOCK_DIR,   exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
SIZE = 500
RNG  = np.random.default_rng(42)


# ── Build dish mask (circular) ────────────────────────────────────────────────
def _circle_mask(size: int, cx: int, cy: int, r: int) -> np.ndarray:
    ys, xs = np.ogrid[:size, :size]
    return ((xs - cx) ** 2 + (ys - cy) ** 2) <= r ** 2


# ── Soft blob at position ─────────────────────────────────────────────────────
def _gaussian_blob(size: int, cx: int, cy: int, sigma: float) -> np.ndarray:
    ys, xs = np.mgrid[:size, :size]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)).astype(
        np.float32
    )


# ── Generate ingredient map ───────────────────────────────────────────────────
def make_ingredient_map(size: int = SIZE) -> tuple[np.ndarray, dict[int, str]]:
    """
    Creates a synthetic (size, size, 5) ingredient map for a pizza-like dish.

    Channels:
      0 — dough      (whole disc, dominant at crust)
      1 — sauce      (inner 75 % of disc, patchy)
      2 — mozzarella (random blobs over sauce area)
      3 — pepperoni  (concentrated on ONE side — makes fairness interesting)
      4 — basil      (tiny scattered spots)
    """
    H = W = size
    cx = cy = size // 2
    r_dish  = int(size * 0.46)
    r_inner = int(size * 0.38)

    dish_mask = _circle_mask(size, cx, cy, r_dish)

    K = 5
    imap = np.zeros((H, W, K), dtype=np.float32)

    # Channel 0 — dough (base, uniform across whole disc)
    imap[:, :, 0][dish_mask] = 0.55

    # Channel 1 — sauce (inner disc, add some noise)
    sauce_mask = _circle_mask(size, cx, cy, r_inner)
    sauce_base = sauce_mask.astype(np.float32) * 0.6
    sauce_noise = RNG.uniform(0, 0.3, (H, W)).astype(np.float32)
    sauce_base += sauce_noise * sauce_mask
    imap[:, :, 1] = sauce_base

    # Channel 2 — mozzarella blobs
    moz = np.zeros((H, W), dtype=np.float32)
    for _ in range(18):
        bx = int(cx + RNG.integers(-r_inner + 30, r_inner - 30))
        by = int(cy + RNG.integers(-r_inner + 30, r_inner - 30))
        sigma = float(RNG.integers(18, 42))
        moz += _gaussian_blob(size, bx, by, sigma) * 0.9
    imap[:, :, 2] = moz * sauce_mask

    # Channel 3 — pepperoni (concentrated on the left half — uneven distribution)
    pep = np.zeros((H, W), dtype=np.float32)
    for _ in range(14):
        # left side heavy (x < cx+50)
        bx = int(cx + RNG.integers(-r_inner + 20, 50))
        by = int(cy + RNG.integers(-r_inner + 20, r_inner - 20))
        sigma = float(RNG.integers(12, 22))
        pep += _gaussian_blob(size, bx, by, sigma) * 1.2
    imap[:, :, 3] = pep * sauce_mask

    # Channel 4 — basil (tiny random spots all over)
    basil = np.zeros((H, W), dtype=np.float32)
    for _ in range(8):
        bx = int(cx + RNG.integers(-r_inner + 10, r_inner - 10))
        by = int(cy + RNG.integers(-r_inner + 10, r_inner - 10))
        basil += _gaussian_blob(size, bx, by, 10.0) * 0.5
    imap[:, :, 4] = basil * sauce_mask

    # Zero outside dish
    imap[~dish_mask] = 0.0

    # Normalise: channels sum to 1.0 at each dish pixel
    total = imap.sum(axis=2, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    imap /= total
    imap[~dish_mask] = 0.0

    labels = {
        0: "dough",
        1: "tomato_sauce",
        2: "mozzarella",
        3: "pepperoni",
        4: "basil",
    }
    return imap.astype(np.float32), labels


# ── Generate synthetic pizza image ───────────────────────────────────────────
def make_pizza_image(imap: np.ndarray, labels: dict[int, str], size: int = SIZE) -> Image.Image:
    """Render a plausible-looking pizza from the ingredient map."""
    img = Image.new("RGB", (size, size), (185, 160, 120))  # wooden table
    draw = ImageDraw.Draw(img)

    cx = cy = size // 2
    r_dish  = int(size * 0.46)

    # ── Dough base ──────────────────────────────────────────────────────
    draw.ellipse(
        [cx - r_dish, cy - r_dish, cx + r_dish, cy + r_dish],
        fill=(215, 175, 85),
    )

    # ── Sauce ────────────────────────────────────────────────────────────
    sauce_density = imap[:, :, 1]
    sauce_arr = (sauce_density * 255).astype(np.uint8)
    sauce_layer = Image.fromarray(sauce_arr)
    sauce_color = Image.new("RGB", (size, size), (185, 50, 30))
    img.paste(sauce_color, mask=sauce_layer)

    # ── Mozzarella ────────────────────────────────────────────────────────
    moz_density = imap[:, :, 2]
    moz_arr = (np.clip(moz_density * 1.4, 0, 1) * 255).astype(np.uint8)
    moz_layer = Image.fromarray(moz_arr)
    moz_color = Image.new("RGB", (size, size), (240, 228, 195))
    img.paste(moz_color, mask=moz_layer)

    # ── Pepperoni (draw circles where density > threshold) ──────────────
    pep_density = imap[:, :, 3]
    pep_img = Image.fromarray((np.clip(pep_density, 0, 1) * 255).astype(np.uint8))
    pep_color = Image.new("RGB", (size, size), (145, 30, 20))
    img.paste(pep_color, mask=pep_img)

    # ── Basil spots ───────────────────────────────────────────────────────
    basil_density = imap[:, :, 4]
    basil_arr = (np.clip(basil_density * 2, 0, 1) * 255).astype(np.uint8)
    basil_layer = Image.fromarray(basil_arr)
    basil_color = Image.new("RGB", (size, size), (40, 160, 50))
    img.paste(basil_color, mask=basil_layer)

    # Light blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    return img


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[generate_mock_data] Building synthetic pizza dataset …")

    imap, labels = make_ingredient_map(SIZE)

    # Validate
    assert imap.dtype == np.float32, "dtype should be float32"
    assert imap.shape == (SIZE, SIZE, 5), f"expected (500,500,5), got {imap.shape}"
    assert imap.min() >= -1e-5,          "values should be >= 0"
    assert imap.max() <= 1.0 + 1e-5,    "values should be <= 1"

    # Save ingredient map
    map_path = os.path.join(MOCK_DIR, "pizza_mock_500.npy")
    np.save(map_path, imap)
    print(f"  ✓ Ingredient map  → {map_path}  shape={imap.shape}")

    # Save labels
    labels_path = os.path.join(MOCK_DIR, "pizza_labels.npy")
    np.save(labels_path, labels)
    print(f"  ✓ Labels          → {labels_path}  {labels}")

    # Save synthetic image
    img = make_pizza_image(imap, labels, SIZE)
    img_path = os.path.join(IMAGES_DIR, "pizza_test.jpg")
    img.save(img_path, format="JPEG", quality=92)
    print(f"  ✓ Pizza image     → {img_path}  {img.size}")

    print("\n[generate_mock_data] Done! You can now run the full pipeline:")
    print(f"  python src/fair-slice/main.py {img_path} 4")
    print("  (partition.py must be implemented first)")


if __name__ == "__main__":
    main()
