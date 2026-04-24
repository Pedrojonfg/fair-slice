"""
Genera un mockup completo del resultado de FairSlice usando datos sintéticos.
Produce: tests/fixtures/mockup_result.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'fair-slice'))

import numpy as np
from PIL import Image
from visualize import render_partition

SIZE = 500
RNG = np.random.default_rng(7)

img_path = "Neapolitan_Pizza_-_Web_1c60fb1e-76dd-42a8-bdda-d3ab84d0231f.webp"

imap = np.load("tests/fixtures/mock_data/pizza_mock_500.npy")
labels = {0: "dough", 1: "sauce", 2: "mozzarella", 3: "pepperoni", 4: "basil"}

H, W, K = imap.shape
cx, cy = W // 2, H // 2
r_dish = int(SIZE * 0.46)

# ── Dish mask ─────────────────────────────────────────────────────────
ys, xs = np.ogrid[:H, :W]
dish_mask = ((xs - cx)**2 + (ys - cy)**2) <= r_dish**2

# ── Partition: Voronoi-style (4 seeds slightly offset) ────────────────
n_people = 4
# Seeds placed roughly at cardinal positions, shifted inward
seeds = np.array([
    [cx - 110, cy - 80],   # person 1 — top-left
    [cx + 120, cy - 70],   # person 2 — top-right
    [cx + 100, cy + 110],  # person 3 — bottom-right
    [cx - 90,  cy + 120],  # person 4 — bottom-left
])

# Assign each dish pixel to nearest seed
label_map = np.full((H, W), -1, dtype=np.int16)
for y in range(H):
    for x in range(W):
        if not dish_mask[y, x]:
            continue
        dists = ((seeds[:, 0] - x)**2 + (seeds[:, 1] - y)**2)
        label_map[y, x] = int(np.argmin(dists))

masks = [label_map == i for i in range(n_people)]

# ── Compute realistic scores from the ingredient map ─────────────────
scores = np.zeros((n_people, K), dtype=np.float32)
for i in range(n_people):
    m = masks[i]
    total_per_ingredient = imap[dish_mask].sum(axis=0)
    person_total = imap[m].sum(axis=0)
    scores[i] = person_total / np.where(total_per_ingredient > 0, total_per_ingredient, 1)

# ── Fairness score (simulated post-algorithm result) ──────────────────
# In the real app, Pedro's Voronoi algorithm optimises this to ~0.87+
fairness = 0.87

print(f"Fairness: {fairness:.3f}")
for i in range(n_people):
    parts = [f"{labels[k]}={scores[i,k]:.1%}" for k in range(K)]
    print(f"  Person {i+1}: {', '.join(parts)}")

# ── Render ────────────────────────────────────────────────────────────
out = render_partition(img_path, masks, scores, labels, fairness)
out_path = "tests/fixtures/mockup_result.png"
out.save(out_path)
print(f"\nSaved → {out_path}  ({out.width}x{out.height})")
