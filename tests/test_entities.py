"""Unit tests for core domain entities."""

import math
import pytest

from core.entities import Customer, Depot, Route


@pytest.fixture
def depot() -> Depot:
    return Depot(index=1, x=0.0, y=0.0, max_duration=100.0, max_capacity=50)


@pytest.fixture
def customers() -> list:
    return [
        Customer(index=1, x=3.0,  y=4.0,  demand=10, service_time=0),
        Customer(index=2, x=6.0,  y=8.0,  demand=15, service_time=0),
        Customer(index=3, x=9.0,  y=12.0, demand=20, service_time=0),
    ]


@pytest.fixture
def route(depot, customers) -> Route:
    return Route(depot=depot, customers=list(customers))


class TestCustomer:
    def test_fields(self):
        c = Customer(index=1, x=3.0, y=4.0, demand=10, service_time=5)
        assert c.index == 1
        assert c.x == 3.0
        assert c.y == 4.0
        assert c.demand == 10
        assert c.service_time == 5


class TestDepot:
    def test_fields(self):
        d = Depot(index=1, x=0.0, y=0.0, max_duration=100.0, max_capacity=50)
        assert d.index == 1
        assert d.max_duration == 100.0
        assert d.max_capacity == 50


class TestRoute:
    def test_total_demand(self, route):
        assert route.total_demand() == 45  # 10 + 15 + 20

    def test_total_demand_empty(self, depot):
        assert Route(depot=depot).total_demand() == 0

    def test_total_distance(self, route):
        # depot(0,0) → C1(3,4) → C2(6,8) → C3(9,12) → depot(0,0)
        d01 = math.sqrt(3 ** 2 + 4 ** 2)   # 5.0
        d12 = math.sqrt(3 ** 2 + 4 ** 2)   # 5.0
        d23 = math.sqrt(3 ** 2 + 4 ** 2)   # 5.0
        d30 = math.sqrt(9 ** 2 + 12 ** 2)  # 15.0
        assert route.total_distance() == pytest.approx(d01 + d12 + d23 + d30)

    def test_total_distance_empty(self, depot):
        assert Route(depot=depot).total_distance() == 0.0

    def test_total_duration_no_service(self, route):
        assert route.total_duration() == pytest.approx(route.total_distance())

    def test_total_duration_with_service(self, depot):
        customers = [
            Customer(index=1, x=3.0, y=4.0, demand=5, service_time=10),
            Customer(index=2, x=6.0, y=8.0, demand=5, service_time=20),
        ]
        r = Route(depot=depot, customers=customers)
        assert r.total_duration() == pytest.approx(r.total_distance() + 30)

    def test_total_distance_with_wasted_history(self, route):
        wasted = 12.5
        r = Route(
            depot=route.depot,
            customers=route.customers,
            wasted_distance=wasted,
        )
        assert r.total_distance() == pytest.approx(route.total_distance() + wasted)

    def test_total_duration_with_wasted_history(self, route):
        wasted_time = 9.0
        r = Route(
            depot=route.depot,
            customers=route.customers,
            wasted_duration=wasted_time,
        )
        assert r.total_duration() == pytest.approx(route.total_duration() + wasted_time)

    def test_depot_index_property(self, route):
        assert route.depot_index == 1

    def test_customer_indices_property(self, route):
        assert route.customer_indices == [1, 2, 3]

    def test_is_feasible(self, route):
        # demand=45 <= 50, distance well within 100
        assert route.is_feasible()

    def test_infeasible_capacity(self, depot):
        # 3 customers × demand=20 = 60 > max_capacity=50
        customers = [
            Customer(index=i, x=0.0, y=float(i), demand=20, service_time=0)
            for i in range(1, 4)
        ]
        assert not Route(depot=depot, customers=customers).is_feasible()

    def test_infeasible_duration(self):
        strict = Depot(index=1, x=0.0, y=0.0, max_duration=1.0, max_capacity=9999)
        customers = [Customer(index=1, x=50.0, y=50.0, demand=1, service_time=0)]
        assert not Route(depot=strict, customers=customers).is_feasible()

    def test_unlimited_duration(self, customers):
        # max_duration=0 means no duration limit.
        unlimited = Depot(index=1, x=0.0, y=0.0, max_duration=0.0, max_capacity=9999)
        assert Route(depot=unlimited, customers=customers).is_feasible()
