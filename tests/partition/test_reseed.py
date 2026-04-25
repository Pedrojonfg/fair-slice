import numpy as np
import pytest

import partition as partition_mod


def _grid_coords(h: int, w: int) -> np.ndarray:
    ys, xs = np.mgrid[:h, :w]
    return np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float64)


def _coords_to_domain(coords: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    domain = np.zeros((h, w), dtype=bool)
    ys = coords[:, 0].astype(int)
    xs = coords[:, 1].astype(int)
    domain[ys, xs] = True
    return domain


def test_r1_balanced_distribution_skips_reseed():
    # Two points, two people, one ingredient: perfectly balanced by construction.
    coords = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    densities = np.array([[1.0], [1.0]], dtype=np.float64)  # (M=2, K=1)
    p = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    w = np.zeros(2, dtype=np.float64)
    alpha = np.array([1.0], dtype=np.float64)
    domain_full = _coords_to_domain(coords, (1, 2))

    assignment = partition_mod._assign(coords, p, w)
    I = partition_mod._compute_integrals(densities, assignment, 2)
    targets = I.copy()  # exact match => all imbalances are zero

    p2, w2 = partition_mod._active_reseed(
        coords, densities, p, w, targets, alpha, domain_full
    )
    assert np.allclose(p2, p)
    assert np.allclose(w2, w)


def test_r2_moves_deficit_seed_towards_rich_ingredient_centroid():
    # 2 clusters: left has no pepperoni, right has all pepperoni.
    # Person 0 currently owns left cluster (deficit), person 1 owns right (excess).
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 10.0],
            [1.0, 10.0],
        ],
        dtype=np.float64,
    )
    # K=2: [base, pepperoni]
    densities = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 5.0],
            [1.0, 5.0],
        ],
        dtype=np.float64,
    )
    p = np.array([[0.5, 0.0], [0.5, 10.0]], dtype=np.float64)
    w = np.zeros(2, dtype=np.float64)
    alpha = np.array([0.0, 1.0], dtype=np.float64)  # only care about pepperoni
    domain_full = _coords_to_domain(coords, (2, 11))

    # Targets: person 0 wants most pepperoni.
    I = partition_mod._compute_integrals(densities, partition_mod._assign(coords, p, w), 2)
    targets = I.copy()
    targets[0, 1] = 8.0  # wants pepperoni but currently gets ~0
    targets[1, 1] = 2.0  # rich cell has excess

    p2, _ = partition_mod._active_reseed(
        coords, densities, p, w, targets, alpha, domain_full
    )

    # Pepperoni centroid in rich cell is around x=10, y=0.5
    centroid = np.array([0.5, 10.0], dtype=np.float64)
    # Must move person0 seed towards centroid (increase x).
    assert p2[0, 1] > p[0, 1]
    assert np.linalg.norm(p2[0] - centroid) < np.linalg.norm(p[0] - centroid)


def test_r3_rich_cell_empty_skips_without_crashing(monkeypatch):
    # Force i_rich to refer to an empty cell by making all excess ties at 0,
    # and having cell 0 be empty under the current assignment.
    coords = np.array([[0.0, 10.0], [1.0, 10.0]], dtype=np.float64)
    densities = np.array([[0.0], [0.0]], dtype=np.float64)  # ingredient absent everywhere
    p = np.array([[0.5, 0.0], [0.5, 10.0], [0.5, 20.0]], dtype=np.float64)  # 3 cells
    w = np.zeros(3, dtype=np.float64)
    alpha = np.array([1.0], dtype=np.float64)
    domain_full = _coords_to_domain(coords, (2, 21))

    # Create artificial targets so person 1 is "deficit" (even though mass is 0).
    targets = np.zeros((3, 1), dtype=np.float64)
    targets[1, 0] = 1.0  # impossible target => deficit, but excess stays 0 for all

    p2, w2 = partition_mod._active_reseed(
        coords, densities, p, w, targets, alpha, domain_full
    )
    assert np.allclose(p2, p)
    assert np.allclose(w2, w)


def test_r4_rich_weights_all_zero_skips_without_crashing():
    # Rich cell has pixels assigned but ingredient density is zero there.
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float64)
    densities = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    p = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    w = np.zeros(2, dtype=np.float64)
    alpha = np.array([1.0], dtype=np.float64)
    domain_full = _coords_to_domain(coords, (1, 3))

    targets = np.array([[1.0], [0.0]], dtype=np.float64)  # deficit for cell0
    p2, w2 = partition_mod._active_reseed(
        coords, densities, p, w, targets, alpha, domain_full
    )
    assert np.allclose(p2, p)
    assert np.allclose(w2, w)


def test_r5_alpha_move_zero_is_identity(monkeypatch):
    monkeypatch.setattr(partition_mod, "_RESEED_ALPHA_MOVE", 0.0)

    coords = np.array([[0.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    densities = np.array([[0.0, 0.0], [0.0, 5.0]], dtype=np.float64)
    p = np.array([[0.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    w = np.array([1.0, 1.0], dtype=np.float64)
    alpha = np.array([0.0, 1.0], dtype=np.float64)
    domain_full = _coords_to_domain(coords, (1, 11))
    targets = np.array([[0.0, 5.0], [0.0, 0.0]], dtype=np.float64)

    p2, w2 = partition_mod._active_reseed(
        coords, densities, p, w, targets, alpha, domain_full
    )
    assert np.allclose(p2, p)
    # Even with alpha_move=0, the reseed "kick" resets the poor cell weight.
    assert w2[0] == 0.0


def test_r6_alpha_move_one_teleports_to_centroid(monkeypatch):
    monkeypatch.setattr(partition_mod, "_RESEED_ALPHA_MOVE", 1.0)

    coords = np.array([[0.0, 0.0], [0.0, 10.0], [1.0, 10.0]], dtype=np.float64)
    densities = np.array([[1.0], [2.0], [2.0]], dtype=np.float64)  # K=1 ingredient
    p = np.array([[0.0, 0.0], [0.5, 10.0]], dtype=np.float64)
    w = np.zeros(2, dtype=np.float64)
    alpha = np.array([1.0], dtype=np.float64)
    domain_full = _coords_to_domain(coords, (2, 11))

    # Make person 0 deficit, person 1 rich.
    targets = np.array([[5.0], [0.0]], dtype=np.float64)

    p2, _ = partition_mod._active_reseed(
        coords, densities, p, w, targets, alpha, domain_full
    )

    # Rich cell consists of the two right points with equal weight => centroid at (0.5, 10)
    centroid = np.array([0.5, 10.0], dtype=np.float64)
    assert np.allclose(p2[0], centroid)


def test_r7_irregular_domain_projection_donut_stays_inside_domain(monkeypatch):
    monkeypatch.setattr(partition_mod, "_RESEED_ALPHA_MOVE", 1.0)

    # Donut domain in 7x7: ring where 1 <= r <= 2 (approx using euclidean)
    h = 7
    width = 7
    cy = cx = 3.0
    ys, xs = np.mgrid[:h, :width]
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    donut = (r >= 1.5) & (r <= 2.5)

    coords = np.stack(np.where(donut), axis=1).astype(np.float64)  # (y,x)
    domain_full = donut

    # Two cells, single ingredient. Put all points in rich cell (cell 1).
    densities = np.ones((coords.shape[0], 1), dtype=np.float64)
    p = np.array([[0.0, 0.0], [cy, cx]], dtype=np.float64)
    wts = np.array([0.0, 1e9], dtype=np.float64)  # huge weight => assign everything to cell 1
    alpha = np.array([1.0], dtype=np.float64)

    # Force person 0 to be the most deficit, ingredient 0 the focus.
    targets = np.array([[10.0], [0.0]], dtype=np.float64)

    p2, _ = partition_mod._active_reseed(
        coords, densities, p, wts, targets, alpha, domain_full
    )

    y = float(p2[0, 0])
    x = float(p2[0, 1])
    yi = max(0, min(h - 1, int(round(y))))
    xi = max(0, min(width - 1, int(round(x))))
    assert domain_full[yi, xi]


def test_r8_resets_poor_weight_to_zero():
    coords = np.array([[0.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    densities = np.array([[0.0, 0.0], [0.0, 5.0]], dtype=np.float64)
    p = np.array([[0.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    w = np.array([3.0, 7.0], dtype=np.float64)
    alpha = np.array([0.0, 1.0], dtype=np.float64)
    domain_full = _coords_to_domain(coords, (1, 11))
    targets = np.array([[0.0, 5.0], [0.0, 0.0]], dtype=np.float64)

    p2, w2 = partition_mod._active_reseed(
        coords, densities, p, w, targets, alpha, domain_full
    )

    # Seed should have moved and the poor weight reset.
    assert p2[0, 1] >= p[0, 1]
    assert w2[0] == 0.0
