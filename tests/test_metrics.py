"""Unit tests for utility metrics."""

import pytest

from core.entities import Customer, Depot
from utils.metrics import euclidean_distance, build_distance_matrix


class TestEuclideanDistance:
    def test_known_345_triangle(self):
        assert euclidean_distance(0, 0, 3, 4) == pytest.approx(5.0)

    def test_same_point(self):
        assert euclidean_distance(3, 7, 3, 7) == pytest.approx(0.0)

    def test_negative_coordinates(self):
        assert euclidean_distance(-3, 0, 0, 4) == pytest.approx(5.0)

    def test_symmetry(self):
        assert euclidean_distance(1, 2, 5, 6) == pytest.approx(
            euclidean_distance(5, 6, 1, 2)
        )


class TestBuildDistanceMatrix:
    @pytest.fixture
    def depots(self):
        return [Depot(index=1, x=0.0, y=0.0, max_duration=100.0, max_capacity=50)]

    @pytest.fixture
    def customers(self):
        return [
            Customer(index=9,  x=3.0, y=4.0, demand=10, service_time=0),
            Customer(index=10, x=6.0, y=8.0, demand=10, service_time=0),
        ]

    def test_matrix_size(self, depots, customers):
        matrix = build_distance_matrix(depots, customers)
        n = len(depots) + len(customers)  # 3
        assert len(matrix) == n
        assert all(len(row) == n for row in matrix)

    def test_diagonal_is_zero(self, depots, customers):
        matrix = build_distance_matrix(depots, customers)
        for i in range(len(matrix)):
            assert matrix[i][i] == pytest.approx(0.0)

    def test_symmetric(self, depots, customers):
        matrix = build_distance_matrix(depots, customers)
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                assert matrix[i][j] == pytest.approx(matrix[j][i])

    def test_depot_to_first_customer(self, depots, customers):
        # depot(0,0) → customer(3,4) = 5.0
        matrix = build_distance_matrix(depots, customers)
        assert matrix[0][1] == pytest.approx(5.0)

    def test_customer_to_customer(self, depots, customers):
        # (3,4) → (6,8) = sqrt(9+16) = 5.0
        matrix = build_distance_matrix(depots, customers)
        assert matrix[1][2] == pytest.approx(5.0)
