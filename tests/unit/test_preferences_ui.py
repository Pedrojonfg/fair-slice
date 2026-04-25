import numpy as np
import pytest

from preferences_ui import build_preference_matrix, uniform_preferences


def test_uniform_valid_input_shape_and_equal_values():
    n_people = 3
    ingredient_labels = {0: "a", 1: "b", 2: "c"}
    K = len(ingredient_labels)
    user_inputs = {i: [1.0] * K for i in range(n_people)}

    P = build_preference_matrix(n_people, ingredient_labels, user_inputs)
    assert P.shape == (n_people, K)
    assert P.dtype == np.float32
    assert np.all(P == P[0, 0])


def test_distinct_values_no_normalization_in_this_module():
    n_people = 2
    ingredient_labels = {0: "base", 1: "topping"}
    user_inputs = {
        0: [1.0, 10.0],
        1: [2.0, 3.0],
    }

    P = build_preference_matrix(n_people, ingredient_labels, user_inputs)
    assert P.shape == (2, 2)
    assert np.allclose(P[0], np.array([1.0, 10.0], dtype=np.float32))
    assert np.allclose(P[1], np.array([2.0, 3.0], dtype=np.float32))


def test_n_people_mismatch_raises_value_error():
    n_people = 3
    ingredient_labels = {0: "a", 1: "b"}
    user_inputs = {0: [1.0, 1.0], 1: [1.0, 1.0]}  # only 2 people

    with pytest.raises(ValueError, match="n_people"):
        build_preference_matrix(n_people, ingredient_labels, user_inputs)


def test_wrong_list_length_raises_value_error():
    n_people = 2
    ingredient_labels = {0: "a", 1: "b", 2: "c"}
    user_inputs = {0: [1.0, 1.0, 1.0], 1: [1.0, 1.0]}  # wrong length

    with pytest.raises(ValueError, match="length K="):
        build_preference_matrix(n_people, ingredient_labels, user_inputs)


def test_negative_value_raises_value_error():
    n_people = 2
    ingredient_labels = {0: "a", 1: "b"}
    user_inputs = {0: [1.0, -0.1], 1: [1.0, 1.0]}

    with pytest.raises(ValueError, match="must be >= 0"):
        build_preference_matrix(n_people, ingredient_labels, user_inputs)


def test_k_equals_1_works():
    n_people = 4
    ingredient_labels = {0: "only"}
    user_inputs = {i: [float(i + 1)] for i in range(n_people)}

    P = build_preference_matrix(n_people, ingredient_labels, user_inputs)
    assert P.shape == (n_people, 1)
    assert np.all(P[:, 0] == np.array([1, 2, 3, 4], dtype=np.float32))


def test_uniform_preferences_helper():
    P = uniform_preferences(5, 1)
    assert P.shape == (5, 1)
    assert P.dtype == np.float32
    assert np.allclose(P, 1.0)

