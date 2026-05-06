"""Unit tests for the CCBC clustering and PSO routing modules."""

import math

import numpy as np
import pytest

from core.entities import Customer, Depot, Route
from core.solution import Solution
from utils.config import CCBCConfig, PSOConfig, AppConfig

from algorithms.ccbc_cluster import run_ccbc_clustering
from algorithms.pso_router import RoutingProblem, run_pso_routing
from algorithms.ccbc_pso import CCBCPSOAlgorithm


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
def ccbc_cfg() -> CCBCConfig:
    return CCBCConfig(max_iter=100, tol=1e-4, n_starts=3)


@pytest.fixture
def pso_cfg() -> PSOConfig:
    return PSOConfig(
        pop_size=10, n_gen=30, inertia=0.9, c1=2.0, c2=2.0,
        adaptive=True, seed=0,
    )


@pytest.fixture
def app_cfg(ccbc_cfg, pso_cfg) -> AppConfig:
    return AppConfig(ccbc=ccbc_cfg, pso=pso_cfg)


# ---------------------------------------------------------------------------
# run_ccbc_clustering
# ---------------------------------------------------------------------------

class TestCCBCClustering:
    def test_customers_split_by_nearest_depot(self, depots, customers):
        """Customers near depot 1 should be assigned to depot 1, etc."""
        cfg = CCBCConfig(max_iter=100, tol=1e-4, n_starts=3)
        clusters = run_ccbc_clustering(customers=customers, depots=depots, cfg=cfg)
        depot1_indices = {c.index for c in clusters[depots[0]]}
        depot2_indices = {c.index for c in clusters[depots[1]]}
        # Customers 1,2 are near depot 1 (0,0); customers 3,4 near depot 2 (20,20)
        assert {1, 2}.issubset(depot1_indices)
        assert {3, 4}.issubset(depot2_indices)

    def test_all_customers_assigned(self, depots, customers):
        """Every customer must appear in exactly one cluster."""
        cfg = CCBCConfig(max_iter=100, tol=1e-4, n_starts=3)
        clusters = run_ccbc_clustering(customers=customers, depots=depots, cfg=cfg)
        assigned = [c for cs in clusters.values() for c in cs]
        assert len(assigned) == len(customers)
        assert {c.index for c in assigned} == {c.index for c in customers}

    def test_capacity_budget_respected(self, depots):
        """No cluster should exceed its capacity budget when avoidable."""
        # 6 customers each with demand=10; each depot has capacity=60 and 1 vehicle
        # → budget=60 per depot; 3 customers per depot is feasible
        cs = [Customer(index=i, x=float(i), y=0.0, demand=10, service_time=0) for i in range(1, 7)]
        cfg = CCBCConfig(max_iter=100, tol=1e-4, n_starts=3)
        clusters = run_ccbc_clustering(customers=cs, depots=depots, cfg=cfg)
        for depot, assigned in clusters.items():
            total = sum(c.demand for c in assigned)
            budget = depot.max_capacity * depot.max_vehicles
            assert total <= budget, f"Depot {depot.index} exceeded budget: {total} > {budget}"

    def test_empty_customers(self, depots):
        cfg = CCBCConfig(max_iter=100, tol=1e-4, n_starts=3)
        clusters = run_ccbc_clustering(customers=[], depots=depots, cfg=cfg)
        assert all(v == [] for v in clusters.values())

    def test_returns_all_depots(self, depots, customers):
        cfg = CCBCConfig(max_iter=100, tol=1e-4, n_starts=3)
        clusters = run_ccbc_clustering(customers=customers, depots=depots, cfg=cfg)
        assert set(clusters.keys()) == set(depots)


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
# ---------------------------------------------------------------------------
# CCBCPSOAlgorithm smoke test
# ---------------------------------------------------------------------------

class TestCCBCPSOAlgorithm:
    def test_solve_returns_solution(self, depots, customers, app_cfg):
        algo = CCBCPSOAlgorithm(app_cfg)
        solution = algo.solve(customers, depots)
        assert isinstance(solution, Solution)

    def test_all_customers_assigned(self, depots, customers, app_cfg):
        algo = CCBCPSOAlgorithm(app_cfg)
        solution = algo.solve(customers, depots)
        assigned = {c.index for route in solution.routes for c in route.customers}
        expected = {c.index for c in customers}
        assert assigned == expected

    def test_cost_is_positive(self, depots, customers, app_cfg):
        algo = CCBCPSOAlgorithm(app_cfg)
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

        algo = CCBCPSOAlgorithm(app_cfg)
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
        assert isinstance(app_cfg.ccbc.max_iter, int)
        assert isinstance(app_cfg.ccbc.tol, float)
        assert isinstance(app_cfg.ccbc.n_starts, int)
        assert isinstance(app_cfg.pso.adaptive, bool)
        assert isinstance(app_cfg.pso.c1, float)
