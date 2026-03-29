"""Unit tests for the Cordeau → domain entity converter."""

import pytest

from utils.data_loader import CordeauInstance, Node
from utils.converter import build_customers, build_depots


@pytest.fixture
def instance() -> CordeauInstance:
    customers = [
        Node(index=1, x=10.0, y=20.0, service_time=0, demand=5,
             frequency=1, combination_count=1, combinations=[]),
        Node(index=2, x=30.0, y=40.0, service_time=5, demand=10,
             frequency=1, combination_count=1, combinations=[]),
    ]
    depots = [
        Node(index=3, x=0.0, y=0.0, service_time=0, demand=0,
             frequency=0, combination_count=0, combinations=[]),
        Node(index=4, x=5.0, y=5.0, service_time=0, demand=0,
             frequency=0, combination_count=0, combinations=[]),
    ]
    return CordeauInstance(
        problem_type=2,
        vehicle_count=4,
        customer_count=2,
        depot_count=2,
        duration_limits=[100.0, 80.0],
        capacity_limits=[50, 40],
        customers=customers,
        depots=depots,
    )


class TestBuildCustomers:
    def test_count(self, instance):
        assert len(build_customers(instance)) == 2

    def test_fields_first(self, instance):
        c = build_customers(instance)[0]
        assert c.index == 1
        assert c.x == 10.0
        assert c.y == 20.0
        assert c.demand == 5
        assert c.service_time == 0

    def test_service_time_second(self, instance):
        assert build_customers(instance)[1].service_time == 5

    def test_demand_second(self, instance):
        assert build_customers(instance)[1].demand == 10


class TestBuildDepots:
    def test_count(self, instance):
        assert len(build_depots(instance)) == 2

    def test_index_is_one_based_position(self, instance):
        depots = build_depots(instance)
        assert depots[0].index == 1
        assert depots[1].index == 2

    def test_coordinates(self, instance):
        depots = build_depots(instance)
        assert depots[0].x == 0.0
        assert depots[0].y == 0.0
        assert depots[1].x == 5.0
        assert depots[1].y == 5.0

    def test_limits_come_from_instance(self, instance):
        depots = build_depots(instance)
        assert depots[0].max_duration == 100.0
        assert depots[0].max_capacity == 50
        assert depots[1].max_duration == 80.0
        assert depots[1].max_capacity == 40
