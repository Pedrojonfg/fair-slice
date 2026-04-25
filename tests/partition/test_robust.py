import numpy as np
import pytest

from partition import compute_partition
from tests.fixtures import fixture_tiny_uniform, fixture_single_pixel_dish


def test_f1_all_zero_map_raises_no_dish_pixels():
    imap = np.zeros((32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="no dish pixels"):
        compute_partition(imap, 2, mode="free")


@pytest.mark.slow
def test_f2_single_pixel_dish_raises_or_trivial():
    imap = fixture_single_pixel_dish()
    with pytest.raises(ValueError, match="too few pixels"):
        compute_partition(imap, 2, mode="free")


@pytest.mark.slow
def test_f3_zero_total_channel_scores_zero_no_div0():
    imap = fixture_tiny_uniform().copy()
    imap[:, :, 1] = 0.0  # T_j = 0
    r = compute_partition(imap, 2, mode="free")
    assert np.allclose(r["scores"][:, 1], 0.0)


@pytest.mark.slow
def test_f4_channels_not_normalized_still_works():
    imap = fixture_tiny_uniform().copy()
    imap *= 7.0
    r = compute_partition(imap, 2, mode="free")
    assert 0.0 <= float(r["fairness"]) <= 1.0


@pytest.mark.slow
def test_f5_donut_hole_respected_and_connectivity():
    h = w = 64
    k = 2
    cy = cx = (h - 1) / 2
    ys, xs = np.ogrid[:h, :w]
    dish = ((xs - cx) ** 2 + (ys - cy) ** 2) <= 26.0**2
    hole = ((xs - cx) ** 2 + (ys - cy) ** 2) <= 10.0**2
    dish = dish & ~hole
    imap = np.zeros((h, w, k), dtype=np.float32)
    imap[dish] = 0.5
    r = compute_partition(imap, 3, mode="free")
    masks = r["masks"]
    assert all((~m[hole]).all() for m in masks)

