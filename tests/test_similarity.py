import numpy as np
import pytest
from src.similarity import cosine_similarity, batch_cosine_similarity, dot_product, l2_distance


def test_dot_product_known_values():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert dot_product(a, b) == pytest.approx(32.0)


def test_dot_product_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert dot_product(a, b) == pytest.approx(0.0)


def test_cosine_similarity_identical_vectors():
    a = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_perpendicular_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    a = np.array([1.0, 2.0, 3.0])
    zero = np.array([0.0, 0.0, 0.0])
    assert cosine_similarity(a, zero) == 0.0


def test_cosine_similarity_direction_not_magnitude():
    # Scaling a vector shouldn't change similarity
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])  # same direction, double the magnitude
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_l2_distance_known_values():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert l2_distance(a, b) == pytest.approx(5.0)  # 3-4-5 triangle


def test_l2_distance_same_point():
    a = np.array([1.0, 2.0, 3.0])
    assert l2_distance(a, a) == pytest.approx(0.0)


def test_batch_cosine_similarity_shape():
    query = np.array([1.0, 0.0, 0.0])
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    result = batch_cosine_similarity(query, matrix)
    assert result.shape == (3,)


def test_batch_cosine_similarity_values():
    query = np.array([1.0, 0.0, 0.0])
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = batch_cosine_similarity(query, matrix)
    assert result[0] == pytest.approx(1.0)  # identical direction
    assert result[1] == pytest.approx(0.0)  # perpendicular
