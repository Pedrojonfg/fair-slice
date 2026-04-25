from __future__ import annotations

import numpy as np


def _circle_mask(h: int, w: int, cy: float, cx: float, r: float) -> np.ndarray:
    ys, xs = np.ogrid[:h, :w]
    return ((xs - cx) ** 2 + (ys - cy) ** 2) <= r * r


def _normalize_map(imap: np.ndarray, dish: np.ndarray) -> np.ndarray:
    imap = np.asarray(imap, dtype=np.float32)
    imap[~dish] = 0.0
    total = imap.sum(axis=2, keepdims=True)
    total = np.where(total > 0, total, 1.0).astype(np.float32)
    imap = imap / total
    imap[~dish] = 0.0
    return imap.astype(np.float32)


def _uniform_dish(h: int, w: int, k: int, dish: np.ndarray) -> np.ndarray:
    imap = np.zeros((h, w, k), dtype=np.float32)
    imap[dish] = (1.0 / k)
    return imap


def fixture_tiny_uniform() -> np.ndarray:
    """32x32, K=2, uniform inside a circle."""
    h = w = 32
    k = 2
    dish = _circle_mask(h, w, cy=(h - 1) / 2, cx=(w - 1) / 2, r=12.5)
    imap = _uniform_dish(h, w, k, dish)
    return _normalize_map(imap, dish)


def fixture_pizza_uniform() -> np.ndarray:
    """128x128, K=4, dough+sauce+cheese+topping all uniform inside a circle."""
    h = w = 128
    k = 4
    dish = _circle_mask(h, w, cy=(h - 1) / 2, cx=(w - 1) / 2, r=56.0)
    imap = _uniform_dish(h, w, k, dish)
    return _normalize_map(imap, dish)


def fixture_pizza_concentrated_pepperoni() -> np.ndarray:
    """
    128x128, K=4.
    Channel 3 ("pepperoni") concentrated in blobs on the RIGHT half.
    """
    h = w = 128
    k = 4
    cy = cx = (h - 1) / 2
    dish = _circle_mask(h, w, cy=cy, cx=cx, r=56.0)

    imap = np.zeros((h, w, k), dtype=np.float32)
    imap[:, :, 0][dish] = 1.0  # base

    ys, xs = np.mgrid[:h, :w]
    sauce = _circle_mask(h, w, cy=cy, cx=cx, r=44.0).astype(np.float32)
    cheese = _circle_mask(h, w, cy=cy, cx=cx, r=50.0).astype(np.float32) - sauce * 0.2
    sauce = np.clip(sauce, 0, 1)
    cheese = np.clip(cheese, 0, 1)
    imap[:, :, 1] = sauce * dish
    imap[:, :, 2] = cheese * dish

    # Pepperoni blobs on right (x > cx)
    pep = np.zeros((h, w), dtype=np.float32)
    blobs = [
        (cx + 24, cy - 18, 8.5),
        (cx + 18, cy + 12, 7.0),
        (cx + 32, cy + 20, 9.0),
        (cx + 10, cy - 2, 6.5),
    ]
    for bx, by, s in blobs:
        pep += np.exp(-((xs - bx) ** 2 + (ys - by) ** 2) / (2 * s * s)).astype(np.float32)
    pep *= (xs > cx).astype(np.float32)
    imap[:, :, 3] = pep * dish

    # Make base less dominant where toppings exist
    imap[:, :, 0] = imap[:, :, 0] * dish
    return _normalize_map(imap, dish)


def fixture_pizza_radial_symmetric() -> np.ndarray:
    """128x128, K=3, concentric rings (outer cheese, inner sauce)."""
    h = w = 128
    k = 3
    cy = cx = (h - 1) / 2
    dish = _circle_mask(h, w, cy=cy, cx=cx, r=56.0)

    r1 = _circle_mask(h, w, cy=cy, cx=cx, r=22.0)  # inner core
    r2 = _circle_mask(h, w, cy=cy, cx=cx, r=40.0)  # mid
    # channels: 0 base, 1 sauce (inner), 2 cheese (outer ring)
    imap = np.zeros((h, w, k), dtype=np.float32)
    imap[:, :, 0][dish] = 1.0
    imap[:, :, 1][r1] = 2.0
    imap[:, :, 2][dish & ~r2] = 2.0
    imap[:, :, 2][r2 & ~r1] = 1.0
    return _normalize_map(imap, dish)


def fixture_cake_layered() -> np.ndarray:
    """128x128, K=5, horizontal layers across the dish."""
    h = w = 128
    k = 5
    dish = _circle_mask(h, w, cy=(h - 1) / 2, cx=(w - 1) / 2, r=56.0)
    imap = np.zeros((h, w, k), dtype=np.float32)
    imap[:, :, 0][dish] = 1.0
    bands = np.linspace(0, h, k + 1).astype(int)
    for j in range(k):
        y0, y1 = bands[j], bands[j + 1]
        imap[y0:y1, :, j] += 2.0
    return _normalize_map(imap, dish)


def fixture_irregular_dish() -> np.ndarray:
    """100x120, non-convex dish: circle with a 'bite' removed."""
    h, w = 100, 120
    k = 3
    cy, cx = (h - 1) / 2, (w - 1) / 2
    dish = _circle_mask(h, w, cy=cy, cx=cx, r=42.0)
    bite = _circle_mask(h, w, cy=cy - 10, cx=cx + 30, r=18.0)
    dish = dish & ~bite
    imap = _uniform_dish(h, w, k, dish)
    return _normalize_map(imap, dish)


def fixture_tiny_dish_in_big_frame() -> np.ndarray:
    """500x500 with a 50x50 dish in a corner; lots of background."""
    h = w = 500
    k = 4
    dish = np.zeros((h, w), dtype=bool)
    dish[10:60, 10:60] = True
    imap = _uniform_dish(h, w, k, dish)
    return _normalize_map(imap, dish)


def fixture_large_realistic() -> np.ndarray:
    """500x500, K=5, mix of uniform + concentrated channel to stress subsampling."""
    h = w = 500
    k = 5
    cy = cx = (h - 1) / 2
    dish = _circle_mask(h, w, cy=cy, cx=cx, r=220.0)
    imap = np.zeros((h, w, k), dtype=np.float32)
    imap[:, :, 0][dish] = 1.0
    imap[:, :, 1][dish] = 0.6
    imap[:, :, 2][dish] = 0.4

    ys, xs = np.mgrid[:h, :w]
    # concentrated ingredient on one side
    blob = np.exp(-((xs - (cx + 90)) ** 2 + (ys - (cy + 20)) ** 2) / (2 * 35.0**2)).astype(
        np.float32
    )
    imap[:, :, 3] = blob * dish * 2.0
    # scattered noise-ish ingredient
    rng = np.random.default_rng(123)
    speck = rng.uniform(0, 1, (h, w)).astype(np.float32)
    imap[:, :, 4] = (speck > 0.995).astype(np.float32) * dish * 3.0
    return _normalize_map(imap, dish)


def fixture_single_pixel_dish() -> np.ndarray:
    """32x32, dish is exactly 1 pixel."""
    h = w = 32
    k = 2
    dish = np.zeros((h, w), dtype=bool)
    dish[h // 2, w // 2] = True
    imap = np.zeros((h, w, k), dtype=np.float32)
    imap[dish, 0] = 1.0
    return _normalize_map(imap, dish)

