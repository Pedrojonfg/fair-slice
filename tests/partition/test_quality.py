import numpy as np
import pytest

from partition import compute_partition
from tests.fixtures import (
    fixture_pizza_uniform,
    fixture_pizza_concentrated_pepperoni,
    fixture_pizza_radial_symmetric,
    fixture_cake_layered,
    fixture_irregular_dish,
    fixture_large_realistic,
)


@pytest.mark.slow
def test_c1_pizza_uniform_free_fairness():
    imap = fixture_pizza_uniform()
    r = compute_partition(imap, 4, mode="free")
    assert r["fairness"] >= 0.85


@pytest.mark.slow
def test_c2_pizza_uniform_radial_fairness():
    imap = fixture_pizza_uniform()
    r = compute_partition(imap, 4, mode="radial")
    assert r["fairness"] >= 0.80


@pytest.mark.slow
def test_c3_pepperoni_concentrated_free_threshold():
    imap = fixture_pizza_concentrated_pepperoni()
    r = compute_partition(imap, 3, mode="free")
    assert r["fairness"] >= 0.50


@pytest.mark.slow
def test_c4_pepperoni_concentrated_with_pep_lover():
    imap = fixture_pizza_concentrated_pepperoni()
    P = np.ones((3, imap.shape[-1]), dtype=np.float32)
    P[0, 3] = 10.0
    r = compute_partition(imap, 3, mode="free", preferences=P)
    assert r["fairness"] >= 0.80


@pytest.mark.slow
def test_c5_radial_symmetric_radial_mode_shines():
    imap = fixture_pizza_radial_symmetric()
    r = compute_partition(imap, 4, mode="radial")
    assert r["fairness"] >= 0.85


@pytest.mark.slow
def test_c6_cake_layered_free():
    imap = fixture_cake_layered()
    r = compute_partition(imap, 4, mode="free")
    assert r["fairness"] >= 0.55


@pytest.mark.slow
def test_c7_irregular_dish_free():
    imap = fixture_irregular_dish()
    r = compute_partition(imap, 3, mode="free")
    assert r["fairness"] >= 0.65


@pytest.mark.slow
def test_c8_large_realistic_free():
    imap = fixture_large_realistic()
    r = compute_partition(imap, 4, mode="free")
    assert r["fairness"] >= 0.62


@pytest.mark.slow
def test_free_vs_radial_on_concentrated_pepperoni():
    imap = fixture_pizza_concentrated_pepperoni()
    free = compute_partition(imap, 4, mode="free")["fairness"]
    radial = compute_partition(imap, 4, mode="radial")["fairness"]
    assert free >= radial

