import numpy as np
import pytest

from partition import compute_partition
from tests.fixtures import fixture_tiny_uniform


def _base_map():
    return fixture_tiny_uniform()


def test_a1_ingredient_map_not_ndarray():
    with pytest.raises(ValueError, match="must be numpy\\.ndarray"):
        compute_partition(list(_base_map()), 2)


def test_a2_ingredient_map_not_3d():
    imap = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 3D"):
        compute_partition(imap, 2)


def test_a3_ingredient_map_wrong_dtype():
    imap = _base_map().astype(np.float64)
    with pytest.raises(ValueError, match="must be float32"):
        compute_partition(imap, 2)


def test_a4_too_small_spatial():
    imap = np.zeros((3, 3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="too small"):
        compute_partition(imap, 2)


def test_a5_zero_channels():
    imap = np.zeros((8, 8, 0), dtype=np.float32)
    with pytest.raises(ValueError, match="at least 1 channel"):
        compute_partition(imap, 2)


@pytest.mark.parametrize("n_people", [1, 9])
def test_a6_a7_n_people_range(n_people):
    with pytest.raises(ValueError, match=r"must be in \[2, 8\]"):
        compute_partition(_base_map(), n_people)


def test_a8_n_people_not_int():
    with pytest.raises(ValueError, match="must be int"):
        compute_partition(_base_map(), 4.0)


def test_a9_mode_invalid():
    # Current implementation supports:
    #   'free' (legacy alias of 'convex'), 'convex', 'radial', 'auto'
    with pytest.raises(ValueError, match=r"mode must be .*free.*radial"):
        compute_partition(_base_map(), 2, mode="weird")


def test_a10_preferences_shape_mismatch():
    imap = _base_map()
    n_people = 3
    k = imap.shape[-1]
    prefs = np.ones((n_people, k - 1), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        compute_partition(imap, n_people, preferences=prefs)


def test_a11_preferences_ndim_not_2d():
    imap = _base_map()
    prefs = np.ones((imap.shape[-1],), dtype=np.float32)
    with pytest.raises(ValueError, match="preferences must be 2D"):
        compute_partition(imap, 2, preferences=prefs)


def test_a12_preferences_negative_values():
    imap = _base_map()
    prefs = np.ones((2, imap.shape[-1]), dtype=np.float32)
    prefs[0, 0] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        compute_partition(imap, 2, preferences=prefs)


def test_a13_preferences_not_ndarray():
    imap = _base_map()
    with pytest.raises(ValueError, match="numpy\\.ndarray or None"):
        compute_partition(imap, 2, preferences=[[1, 1], [1, 1]])


def test_f6_nan_or_inf_in_input():
    imap = _base_map().copy()
    imap[10, 10, 0] = np.nan
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        compute_partition(imap, 2)

