import numpy as np
import pytest

from partition import compute_partition
from tests.fixtures import fixture_pizza_concentrated_pepperoni


pytestmark = pytest.mark.slow


def test_d1_preferences_none_vs_ones_similar_scores():
    imap = fixture_pizza_concentrated_pepperoni()
    N = 4
    r_none = compute_partition(imap, N, mode="free", preferences=None)

    P_ones = np.ones((N, imap.shape[-1]), dtype=np.float32)
    r_ones = compute_partition(imap, N, mode="free", preferences=P_ones)

    assert np.allclose(r_none["scores"], r_ones["scores"], atol=0.05)


def test_d2_person0_vegan_avoids_pepperoni():
    imap = fixture_pizza_concentrated_pepperoni()
    N = 4
    P = np.ones((N, imap.shape[-1]), dtype=np.float32)
    pep_idx = 3
    P[0, pep_idx] = 0.0
    r = compute_partition(imap, N, mode="free", preferences=P)
    # With the solver and fairness constraints, perfect zeros may not be
    # achievable; we just need a clear reduction.
    assert float(r["scores"][0, pep_idx]) < 0.03


def test_d3_person0_pepperoni_hoarder():
    imap = fixture_pizza_concentrated_pepperoni()
    N = 4
    P = np.ones((N, imap.shape[-1]), dtype=np.float32)
    pep_idx = 3
    P[0, pep_idx] = 10.0
    r = compute_partition(imap, N, mode="free", preferences=P)
    # Conservative threshold: strong preference should noticeably shift allocation.
    assert float(r["scores"][0, pep_idx]) > 0.60


def test_d4_zero_column_no_one_wants_does_not_crash():
    imap = fixture_pizza_concentrated_pepperoni()
    N = 4
    P = np.ones((N, imap.shape[-1]), dtype=np.float32)
    P[:, 2] = 0.0
    r = compute_partition(imap, N, mode="free", preferences=P)
    assert 0.0 <= float(r["fairness"]) <= 1.0


def test_d5_all_zero_preferences_graceful_equivalent_to_uniform():
    imap = fixture_pizza_concentrated_pepperoni()
    N = 4
    P = np.zeros((N, imap.shape[-1]), dtype=np.float32)
    r = compute_partition(imap, N, mode="free", preferences=P)
    r2 = compute_partition(imap, N, mode="free", preferences=None)
    assert np.allclose(r["scores"], r2["scores"], atol=0.05)


def test_d6_one_huge_value_normalizes():
    imap = fixture_pizza_concentrated_pepperoni()
    N = 4
    P = np.ones((N, imap.shape[-1]), dtype=np.float32)
    P[0, 0] = 1e10
    r = compute_partition(imap, N, mode="free", preferences=P)
    assert np.isfinite(r["scores"]).all()
    assert 0.0 <= float(r["fairness"]) <= 1.0


def test_d7_determinism_with_preferences():
    imap = fixture_pizza_concentrated_pepperoni()
    N = 4
    P = np.ones((N, imap.shape[-1]), dtype=np.float32)
    P[0, 3] = 10.0
    r1 = compute_partition(imap, N, mode="free", preferences=P)
    r2 = compute_partition(imap, N, mode="free", preferences=P)
    assert np.allclose(r1["scores"], r2["scores"])
    assert np.allclose(r1["seeds"], r2["seeds"])
    assert all(np.array_equal(a, b) for a, b in zip(r1["masks"], r2["masks"]))

