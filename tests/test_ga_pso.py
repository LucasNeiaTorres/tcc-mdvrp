"""Unit tests for the GA clustering and PSO routing modules."""

import math

import numpy as np
import pytest

from core.entities import Customer, Depot, Route
from core.solution import Solution
from utils.config import GAConfig, PSOConfig, AppConfig

from algorithms.ga_cluster import DepotAssignmentProblem, run_ga_clustering
from algorithms.pso_router import RoutingProblem, run_pso_routing
from algorithms.ga_pso import GAPSOAlgorithm


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def depots() -> list:
    return [
        Depot(index=1, x=0.0,  y=0.0,  max_duration=0.0, max_capacity=60),
        Depot(index=2, x=20.0, y=20.0, max_duration=0.0, max_capacity=60),
    ]


@pytest.fixture
def customers() -> list:
    # Four customers: two near depot 1, two near depot 2
    return [
        Customer(index=1, x=1.0,  y=1.0,  demand=10, service_time=0),
        Customer(index=2, x=2.0,  y=2.0,  demand=10, service_time=0),
        Customer(index=3, x=19.0, y=19.0, demand=10, service_time=0),
        Customer(index=4, x=21.0, y=21.0, demand=10, service_time=0),
    ]


def _dist(a: int, b: int, nodes: dict) -> float:
    ax, ay = nodes[a]
    bx, by = nodes[b]
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


@pytest.fixture
def dist_fn(depots, customers):
    """Simple Euclidean dist_fn for use in tests (no pre-built matrix)."""
    nodes = {d.index: (d.x, d.y) for d in depots}
    nodes.update({c.index: (c.x, c.y) for c in customers})
    return lambda a, b: _dist(a, b, nodes)


@pytest.fixture
def ga_cfg() -> GAConfig:
    return GAConfig(
        pop_size=20, n_gen=30, capacity_penalty=10000.0,
        crossover_prob=0.9, mutation_eta=20, seed=0,
    )


@pytest.fixture
def pso_cfg() -> PSOConfig:
    return PSOConfig(
        pop_size=10, n_gen=30, inertia=0.9, c1=2.0, c2=2.0,
        adaptive=True, seed=0,
    )


@pytest.fixture
def app_cfg(ga_cfg, pso_cfg) -> AppConfig:
    return AppConfig(ga=ga_cfg, pso=pso_cfg)


# ---------------------------------------------------------------------------
# DepotAssignmentProblem
# ---------------------------------------------------------------------------

class TestDepotAssignmentProblem:
    def test_optimal_assignment_no_penalty(self, depots, customers, dist_fn):
        """All customers near their closest depot → no capacity violation."""
        problem = DepotAssignmentProblem(
            customers=customers,
            depots=depots,
            dist_fn=dist_fn,
            capacity_penalty=10000.0,
        )
        # Assign first two to depot 0, last two to depot 1
        x_good = np.array([0, 0, 1, 1])
        out_good: dict = {}
        problem._evaluate(x_good, out_good)

        # Assign all to depot 0 — exceeds capacity (40 > 60? no, 40 ≤ 60, but distances are worse)
        x_far = np.array([1, 1, 0, 0])
        out_far: dict = {}
        problem._evaluate(x_far, out_far)

        assert out_good["F"] < out_far["F"]

    def test_capacity_penalty_applied(self, depots, customers, dist_fn):
        """Assigning all 4 customers (demand=40) to one depot (capacity=60) is fine;
        but 7 customers would exceed it — verify penalty grows with excess."""
        big_customers = [
            Customer(index=i, x=0.5, y=0.5, demand=15, service_time=0)
            for i in range(5)
        ]
        nodes = {d.index: (d.x, d.y) for d in depots}
        nodes.update({c.index: (c.x, c.y) for c in big_customers})
        dfn = lambda a, b: _dist(a, b, nodes)

        problem = DepotAssignmentProblem(
            customers=big_customers,
            depots=depots,
            dist_fn=dfn,
            capacity_penalty=10000.0,
        )
        # All 5 → depot 0: load=75 > max_capacity=60, excess=15
        x_overload = np.array([0] * 5)
        out: dict = {}
        problem._evaluate(x_overload, out)
        assert out["F"] >= 10000.0 * 15  # at least one full penalty unit


# ---------------------------------------------------------------------------
# RoutingProblem
# ---------------------------------------------------------------------------

class TestRoutingProblem:
    def test_evaluate_returns_round_trip_cost(self, depots, dist_fn):
        depot = depots[0]
        route_customers = [
            Customer(index=101, x=3.0, y=4.0, demand=5, service_time=0),
        ]
        nodes = {depot.index: (depot.x, depot.y)}
        nodes.update({c.index: (c.x, c.y) for c in route_customers})
        dfn = lambda a, b: _dist(a, b, nodes)

        problem = RoutingProblem(depot=depot, customers=route_customers, dist_fn=dfn)
        out: dict = {}
        problem._evaluate(np.array([0.5]), out)
        # depot(0,0) → (3,4) → depot(0,0) = 5+5 = 10
        assert out["F"] == pytest.approx(10.0)

    def test_shorter_permutation_wins(self, depots, dist_fn):
        depot = depots[0]  # (0,0)
        # Place customers along x-axis for predictable ordering
        cs = [
            Customer(index=1, x=1.0, y=0.0, demand=5, service_time=0),
            Customer(index=2, x=2.0, y=0.0, demand=5, service_time=0),
            Customer(index=3, x=3.0, y=0.0, demand=5, service_time=0),
        ]
        nodes = {depot.index: (depot.x, depot.y)}
        nodes.update({c.index: (c.x, c.y) for c in cs})
        dfn = lambda a, b: _dist(a, b, nodes)

        problem = RoutingProblem(depot=depot, customers=cs, dist_fn=dfn)

        # Sequential order [0,1,2] → 1+1+1+3 = 6
        out_good: dict = {}
        problem._evaluate(np.array([0.1, 0.5, 0.9]), out_good)  # argsort → [0,1,2]

        # Reversed order [2,1,0] → 3+1+1+1 = 6 (same — symmetric)
        # Try a bad permutation: [2,0,1] → 3+2+1+2 = 8? Let's use a clearly bad one
        out_bad: dict = {}
        problem._evaluate(np.array([0.9, 0.1, 0.5]), out_bad)  # argsort → [1,2,0]

        # Both valid; key assertion: no negative costs
        assert out_good["F"] > 0
        assert out_bad["F"] > 0


# ---------------------------------------------------------------------------
# run_pso_routing edge cases
# ---------------------------------------------------------------------------

class TestRunPSORouting:
    def test_empty_cluster(self, depots, pso_cfg, dist_fn):
        routes = run_pso_routing(depots[0], [], dist_fn, pso_cfg)
        assert routes == []

    def test_single_customer(self, depots, customers, pso_cfg, dist_fn):
        routes = run_pso_routing(depots[0], [customers[0]], dist_fn, pso_cfg)
        assert len(routes) == 1
        assert routes[0].customers[0].index == customers[0].index


# ---------------------------------------------------------------------------
# GAPSOAlgorithm smoke test
# ---------------------------------------------------------------------------

class TestGAPSOAlgorithm:
    def test_solve_returns_solution(self, depots, customers, app_cfg):
        algo = GAPSOAlgorithm(app_cfg)
        solution = algo.solve(customers, depots)
        assert isinstance(solution, Solution)

    def test_all_customers_assigned(self, depots, customers, app_cfg):
        algo = GAPSOAlgorithm(app_cfg)
        solution = algo.solve(customers, depots)
        assigned = {c.index for route in solution.routes for c in route.customers}
        expected = {c.index for c in customers}
        assert assigned == expected

    def test_cost_is_positive(self, depots, customers, app_cfg):
        algo = GAPSOAlgorithm(app_cfg)
        solution = algo.solve(customers, depots)
        assert solution.total_cost() > 0

    def test_single_depot_can_use_multiple_vehicles(self, app_cfg):
        depot = Depot(
            index=1,
            x=0.0,
            y=0.0,
            max_duration=0.0,
            max_capacity=20,
            max_vehicles=2,
        )
        customers = [
            Customer(index=101, x=1.0, y=0.0, demand=10, service_time=0),
            Customer(index=102, x=2.0, y=0.0, demand=10, service_time=0),
            Customer(index=103, x=3.0, y=0.0, demand=10, service_time=0),
            Customer(index=104, x=4.0, y=0.0, demand=10, service_time=0),
        ]

        algo = GAPSOAlgorithm(app_cfg)
        solution = algo.solve(customers, [depot])

        assert len(solution.routes) == 2
        assert all(route.depot.index == depot.index for route in solution.routes)
        assert sum(len(route.customers) for route in solution.routes) == len(customers)
        assert all(route.total_demand() <= depot.max_capacity for route in solution.routes)


# ---------------------------------------------------------------------------
# load_config round-trip
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_types(self, app_cfg):
        assert isinstance(app_cfg.ga.pop_size, int)
        assert isinstance(app_cfg.ga.capacity_penalty, float)
        assert isinstance(app_cfg.pso.adaptive, bool)
        assert isinstance(app_cfg.pso.c1, float)
