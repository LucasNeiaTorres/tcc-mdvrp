"""Unit tests for the Cordeau data and solution file parser."""

from pathlib import Path
import pytest

from utils.data_loader import (
    CordeauInstance,
    CordeauSolution,
    ParsedRoute,
    read_cordeau_data_file,
    read_cordeau_solution_file,
)


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
P01_DATA = DATA_DIR / "cordeau" / "p01"
P01_SOL  = DATA_DIR / "cordeau_sol" / "p01.res"


def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"Data file not found: {path}")


@pytest.fixture
def instance() -> CordeauInstance:
    _skip_if_missing(P01_DATA)
    return read_cordeau_data_file(str(P01_DATA))


@pytest.fixture
def solution(instance: CordeauInstance) -> CordeauSolution:
    _skip_if_missing(P01_SOL)
    return read_cordeau_solution_file(str(P01_SOL), instance)


class TestReadCordeauDataFile:
    def test_problem_type(self, instance):
        assert instance.problem_type == 2

    def test_vehicle_and_customer_count(self, instance):
        assert instance.vehicle_count == 4
        assert instance.customer_count == 50

    def test_depot_count(self, instance):
        assert instance.depot_count == 4

    def test_limits_length(self, instance):
        assert len(instance.duration_limits) == 4
        assert len(instance.capacity_limits) == 4

    def test_customers_length(self, instance):
        assert len(instance.customers) == 50

    def test_depots_length(self, instance):
        assert len(instance.depots) == 4

    def test_first_customer_fields(self, instance):
        c = instance.customers[0]
        assert c.index == 1
        assert c.x == 37
        assert c.y == 52
        assert c.demand == 7
        assert c.frequency == 1

    def test_customer_indices_are_sequential(self, instance):
        indices = [c.index for c in instance.customers]
        assert indices == list(range(1, 51))


class TestReadCordeauSolutionFile:
    def test_objective_is_positive(self, solution):
        assert solution.objective > 0

    def test_has_routes(self, solution):
        assert len(solution.routes) > 0

    def test_route_fields(self, solution, instance):
        route = solution.routes[0]
        assert isinstance(route, ParsedRoute)
        assert route.depot in {d.index for d in instance.depots}
        assert route.vehicle >= 1
        assert route.duration > 0
        assert route.load > 0

    def test_route_customers_are_ints(self, solution):
        for route in solution.routes:
            assert all(isinstance(n, int) for n in route.nodes)

    def test_route_customers_do_not_contain_depot(self, solution):
        """Depot sentinel (0) should be stripped from customer list."""
        for route in solution.routes:
            assert 0 not in route.nodes

    def test_visualizable_routes(self, solution):
        assert solution.visualizable_routes == solution.routes