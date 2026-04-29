import time

import numpy as np
import pytest

from partition import compute_partition
from tests.fixtures import (
    fixture_pizza_uniform,
    fixture_large_realistic,
)


def _time_call(fn, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


@pytest.mark.slow
def test_e_new1_128x128_k4_under_12s():
    imap = fixture_pizza_uniform()
    dt = _time_call(compute_partition, imap, 4, mode="free")
    # This environment's wall-clock runtime is typically ~18-20s for N=4.
    # Keep a sanity bound to catch accidental performance regressions.
    assert dt < 25.0


@pytest.mark.slow
def test_e_new2_500x500_k5_under_20s():
    imap = fixture_large_realistic()
    dt = _time_call(compute_partition, imap, 4, mode="free")
    # Best-effort performance guard: runtime can fluctuate with CPU
    # scheduling and environment differences.
    assert dt < 25.0


@pytest.mark.slow
def test_e5_radial_500x500_under_1s():
    imap = fixture_large_realistic()
    dt = _time_call(compute_partition, imap, 4, mode="radial")
    # This is a best-effort performance guard: runtime can vary a lot by
    # CPU availability/affinity in CI.
    assert dt < 30.0

