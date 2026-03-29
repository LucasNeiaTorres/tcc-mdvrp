"""Unit tests for Solution evaluation."""

import pytest

from core.entities import Customer, Depot, Route
from core.solution import Solution


@pytest.fixture
def depot_a():
    return Depot(index=1, x=0.0, y=0.0, max_duration=200.0, max_capacity=100)


@pytest.fixture
def depot_b():
    return Depot(index=2, x=10.0, y=10.0, max_duration=200.0, max_capacity=100)


@pytest.fixture
def feasible_route(depot_a):
    customers = [
        Customer(index=1, x=1.0, y=0.0, demand=10, service_time=0),
        Customer(index=2, x=2.0, y=0.0, demand=10, service_time=0),
    ]
    return Route(depot=depot_a, customers=customers)


@pytest.fixture
def infeasible_route(depot_b):
    # combined demand (120) exceeds max_capacity (100)
    customers = [
        Customer(index=3, x=11.0, y=10.0, demand=60, service_time=0),
        Customer(index=4, x=12.0, y=10.0, demand=60, service_time=0),
    ]
    return Route(depot=depot_b, customers=customers)


class TestSolution:
    def test_total_cost_empty(self):
        assert Solution().total_cost() == 0.0

    def test_total_cost_single_route(self, feasible_route):
        sol = Solution(routes=[feasible_route])
        assert sol.total_cost() == pytest.approx(feasible_route.total_distance())

    def test_total_cost_multiple_routes(self, feasible_route, infeasible_route):
        sol = Solution(routes=[feasible_route, infeasible_route])
        expected = feasible_route.total_distance() + infeasible_route.total_distance()
        assert sol.total_cost() == pytest.approx(expected)

    def test_is_feasible_all_ok(self, feasible_route):
        assert Solution(routes=[feasible_route]).is_feasible()

    def test_is_feasible_empty(self):
        assert Solution().is_feasible()

    def test_is_infeasible_with_bad_route(self, feasible_route, infeasible_route):
        sol = Solution(routes=[feasible_route, infeasible_route])
        assert not sol.is_feasible()

    def test_feasibility_report_keys(self, feasible_route):
        report = Solution(routes=[feasible_route]).feasibility_report()
        entry = report[0]
        assert "depot" in entry
        assert "demand" in entry
        assert "max_capacity" in entry
        assert "capacity_ok" in entry
        assert "duration" in entry
        assert "max_duration" in entry
        assert "duration_ok" in entry

    def test_feasibility_report_feasible_route(self, feasible_route):
        report = Solution(routes=[feasible_route]).feasibility_report()
        assert report[0]["capacity_ok"] is True
        assert report[0]["duration_ok"] is True

    def test_feasibility_report_infeasible_route(self, infeasible_route):
        report = Solution(routes=[infeasible_route]).feasibility_report()
        assert report[0]["capacity_ok"] is False

    def test_visualizable_routes(self, feasible_route, infeasible_route):
        sol = Solution(routes=[feasible_route, infeasible_route])
        assert sol.visualizable_routes == [feasible_route, infeasible_route]
