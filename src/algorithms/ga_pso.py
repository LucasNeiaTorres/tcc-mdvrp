"""
GA+PSO Cluster-first, Route-second algorithm for MDVRP.

Phase 1 — Clustering (GA):
    Each customer is assigned to a depot via a GA that minimises total
    customer-to-depot distance subject to a capacity penalty.

Phase 2 — Routing (PSO):
    Within each depot's cluster, a PSO with SPV encoding finds the
    optimal visiting order minimising total route distance.

Usage
-----
    cfg = load_config()
    algorithm = GAPSOAlgorithm(cfg)
    solution = algorithm.solve(customers, depots)
"""

from typing import Dict, List

from algorithms.base import ClusterFirstAlgorithm
from algorithms.ga_cluster import run_ga_clustering
from algorithms.pso_router import run_pso_routing
from core.entities import Customer, Depot
from core.solution import Solution
from utils.config import AppConfig, load_config


class GAPSOAlgorithm(ClusterFirstAlgorithm):
    """
    Cluster-first, route-second MDVRP solver using GA and PSO.

    The distance matrix is built once by ``ClusterFirstAlgorithm.solve()``
    before either phase runs, so both ``cluster()`` and ``route()`` can use
    ``self._dist()`` for O(1) lookups.

    Parameters
    ----------
    cfg:
        Full application config loaded from config.yaml.
        If omitted, config.yaml is loaded from the project root.
    """

    def __init__(self, cfg: AppConfig | None = None) -> None:
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.last_clusters: Dict[int, List[int]] = {}


    def cluster(
        self, customers: List[Customer], depots: List[Depot]
    ) -> Dict[Depot, List[Customer]]:
        """Phase 1: assign customers to depots via GA."""
        clusters = run_ga_clustering(
            customers=customers,
            depots=depots,
            dist_fn=self._dist,
            cfg=self.cfg.ga,
        )
        self.last_clusters = {
            depot.index: [customer.index for customer in assigned]
            for depot, assigned in clusters.items()
        }
        return clusters

    def route(self, clusters: Dict[Depot, List[Customer]]) -> Solution:
        """Phase 2: optimise visiting order per depot via PSO."""
        routes = []
        for depot, depot_customers in clusters.items():
            for group in self._split_customers_by_vehicle(depot, depot_customers):
                route = run_pso_routing(
                    depot=depot,
                    customers=group,
                    dist_fn=self._dist,
                    cfg=self.cfg.pso,
                )
                if route.customers:
                    routes.append(route)
        return Solution(routes=routes)

    def _split_customers_by_vehicle(
        self, depot: Depot, customers: List[Customer]
    ) -> List[List[Customer]]:
        """
        Partition a depot cluster into up to ``max_vehicles`` groups.

        Uses first-fit decreasing by demand to keep each vehicle load within
        capacity whenever possible. If strict feasibility is impossible with
        available vehicles, remaining customers are assigned to the least-loaded
        group so every customer is still routed.
        """
        if not customers:
            return []

        max_vehicles = max(1, depot.max_vehicles)
        groups: List[List[Customer]] = [[] for _ in range(max_vehicles)]
        group_loads = [0.0 for _ in range(max_vehicles)]

        for customer in sorted(customers, key=lambda c: c.demand, reverse=True):
            chosen_idx = None
            for i in range(max_vehicles):
                if group_loads[i] + customer.demand <= depot.max_capacity:
                    chosen_idx = i
                    break

            if chosen_idx is None:
                chosen_idx = min(range(max_vehicles), key=lambda i: group_loads[i])

            groups[chosen_idx].append(customer)
            group_loads[chosen_idx] += customer.demand

        return [group for group in groups if group]

    def __repr__(self) -> str:
        return (
            f"GAPSOAlgorithm("
            f"ga_pop={self.cfg.ga.pop_size}, ga_gen={self.cfg.ga.n_gen}, "
            f"pso_pop={self.cfg.pso.pop_size}, pso_gen={self.cfg.pso.n_gen})"
        )
