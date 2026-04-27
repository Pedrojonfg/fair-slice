import time

import numpy as np
import pytest

from partition import compute_partition
from tests.fixtures import (
    fixture_pizza_uniform,
    fixture_pizza_concentrated_pepperoni,
    fixture_pizza_radial_symmetric,
    fixture_cake_layered,
    fixture_irregular_dish,
)


def _run_case(imap: np.ndarray, n_people: int, seed: int) -> tuple[float, float]:
    t0 = time.perf_counter()
    res = compute_partition(imap, n_people, mode="free")
    dt = time.perf_counter() - t0
    fairness = float(res["fairness"])
    return dt, fairness


@pytest.mark.slow
def test_joint_phase_exploration_smoke_local():
    """
    Local, deterministic smoke test to compare joint-phase behavior.

    It intentionally includes "hard" concentrated-ingredient cases where the joint
    phase tends to get stuck if positions don't explore enough.
    """
    cases: list[tuple[str, np.ndarray, int]] = [
        ("uniform", fixture_pizza_uniform(), 4),
        ("pepperoni_right_blobs", fixture_pizza_concentrated_pepperoni(), 4),
        ("radial_symmetric", fixture_pizza_radial_symmetric(), 4),
        ("cake_layered", fixture_cake_layered(), 4),
        ("irregular_dish", fixture_irregular_dish(), 4),
    ]

    dts: list[float] = []
    fairnesses: list[float] = []

    # Repeat each case a few times to reduce noise while staying "smoke".
    repeats = 3
    for rep in range(repeats):
        for name, imap, n_people in cases:
            dt, f = _run_case(imap, n_people=n_people, seed=1234 + rep)
            dts.append(dt)
            fairnesses.append(f)
            print(f"[local-smoke] case={name} rep={rep} dt_s={dt:.3f} fairness={f:.4f}")

    dt_mean = float(np.mean(dts))
    dt_p95 = float(np.percentile(dts, 95))
    f_mean = float(np.mean(fairnesses))
    f_min = float(np.min(fairnesses))

    print(
        "[local-smoke] summary "
        f"n={len(dts)} dt_mean_s={dt_mean:.3f} dt_p95_s={dt_p95:.3f} "
        f"fairness_mean={f_mean:.4f} fairness_min={f_min:.4f}"
    )

    # Loose sanity bounds (this is not a strict perf test).
    assert 0.0 <= f_mean <= 1.0
    assert dt_mean < 60.0
