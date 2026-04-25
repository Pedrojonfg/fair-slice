import numpy as np
import pytest

from partition import compute_partition
from visualize import render_partition


@pytest.mark.slow
def test_g1_load_pizza_mock_500_and_compute_partition():
    imap = np.load("tests/fixtures/mock_data/pizza_mock_500.npy").astype(np.float32)
    r = compute_partition(imap, 4, mode="free")
    assert len(r["masks"]) == 4
    assert r["scores"].shape[0] == 4


@pytest.mark.slow
def test_g2_pipeline_partition_to_visualize_smoke():
    imap = np.load("tests/fixtures/mock_data/pizza_mock_500.npy").astype(np.float32)
    r = compute_partition(imap, 4, mode="free")
    labels = {0: "dough", 1: "tomato_sauce", 2: "mozzarella", 3: "pepperoni", 4: "basil"}
    img = render_partition(
        image_path="tests/fixtures/sample_images/pizza_test.jpg",
        masks=r["masks"],
        scores=r["scores"],
        ingredient_labels=labels,
        fairness=float(r["fairness"]),
    )
    assert img.size == (imap.shape[1], imap.shape[0])


@pytest.mark.slow
def test_g3_determinism_same_image_same_output():
    imap = np.load("tests/fixtures/mock_data/pizza_mock_500.npy").astype(np.float32)
    r1 = compute_partition(imap, 4, mode="free")
    r2 = compute_partition(imap, 4, mode="free")
    assert np.allclose(r1["scores"], r2["scores"])
    assert np.allclose(r1["seeds"], r2["seeds"])
    assert all(np.array_equal(a, b) for a, b in zip(r1["masks"], r2["masks"]))

