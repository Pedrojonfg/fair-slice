import numpy as np
import pytest
from scipy.ndimage import label as cc_label

from partition import compute_partition
from tests.fixtures import (
    fixture_tiny_uniform,
    fixture_pizza_uniform,
    fixture_pizza_concentrated_pepperoni,
    fixture_pizza_radial_symmetric,
    fixture_cake_layered,
    fixture_irregular_dish,
)


def _dish_mask(imap: np.ndarray) -> np.ndarray:
    return imap.sum(axis=-1) > 1e-3


FIXTURES = {
    "tiny_uniform": fixture_tiny_uniform,
    "pizza_uniform": fixture_pizza_uniform,
    "pizza_concentrated_pepperoni": fixture_pizza_concentrated_pepperoni,
    "pizza_radial_symmetric": fixture_pizza_radial_symmetric,
    "cake_layered": fixture_cake_layered,
    "irregular_dish": fixture_irregular_dish,
}


def _assert_contract(imap: np.ndarray, n_people: int, mode: str) -> None:
    H, W, K = imap.shape
    dish = _dish_mask(imap)

    result = compute_partition(imap, n_people=n_people, mode=mode)

    masks = result["masks"]
    scores = result["scores"]

    # B1
    assert len(masks) == n_people
    # B2/B3
    assert all(m.shape == (H, W) for m in masks)
    assert all(m.dtype == np.bool_ for m in masks)

    stack = np.stack(masks).astype(np.int32)
    s = stack.sum(axis=0)

    # B4/B5/B6
    assert np.all(s[dish] == 1)
    assert np.all(s[~dish] == 0)

    # B7/B8
    assert scores.shape == (n_people, K)
    assert scores.dtype == np.float32

    # B9: for channels with T_j > 0, column sums ≈ 1
    totals = imap.reshape(-1, K).sum(axis=0)
    for j in range(K):
        if totals[j] > 1e-9:
            assert np.isclose(float(scores[:, j].sum()), 1.0, atol=2e-3)

    # B10
    assert 0.0 <= float(result["fairness"]) <= 1.0

    # B11
    seeds = result["seeds"]
    assert isinstance(seeds, np.ndarray)
    assert seeds.shape == (n_people, 2)

    # B12 connectivity: exactly 1 connected component per mask (4-neighborhood)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    for m in masks:
        if m.any():
            _, num = cc_label(m, structure=structure)
            assert num == 1

    # B13 determinism
    result2 = compute_partition(imap, n_people=n_people, mode=mode)
    assert np.allclose(result2["seeds"], result["seeds"])
    assert np.allclose(result2["scores"], result["scores"])
    assert np.isclose(float(result2["fairness"]), float(result["fairness"]))
    assert all(np.array_equal(a, b) for a, b in zip(result2["masks"], result["masks"]))


@pytest.mark.parametrize(
    ("fixture_name", "n_people", "mode"),
    [
        ("tiny_uniform", 2, "radial"),
    ],
)
def test_contract_fast_smoke(fixture_name: str, n_people: int, mode: str):
    _assert_contract(FIXTURES[fixture_name](), n_people=n_people, mode=mode)


@pytest.mark.slow
@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("n_people", [2, 3, 4])
@pytest.mark.parametrize("mode", ["radial"])
def test_contract_full_cartesian_slow(fixture_name: str, n_people: int, mode: str):
    _assert_contract(FIXTURES[fixture_name](), n_people=n_people, mode=mode)


@pytest.mark.slow
@pytest.mark.parametrize("fixture_name", ["pizza_uniform", "pizza_concentrated_pepperoni"])
@pytest.mark.parametrize("n_people", [3, 4])
def test_contract_free_subset_slow(fixture_name: str, n_people: int):
    _assert_contract(FIXTURES[fixture_name](), n_people=n_people, mode="free")

