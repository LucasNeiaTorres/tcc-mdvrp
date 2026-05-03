"""
GA+PSO Cluster-first, Route-second algorithm for MDVRP.

Phase 1 — Clustering (GA):
    Each customer is assigned to a depot via a GA that minimises total
    customer-to-depot distance subject to a capacity penalty.

Phase 2 — Routing (PSO + Bellman split):
    Within each depot's cluster a PSO optimises the giant-tour order via SPV
    encoding.  For each candidate permutation the Bellman split finds the
    optimal vehicle partition given the depot's capacity.

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
        """Phase 2: optimise visiting order and vehicle split per depot via PSO."""
        routes = []
        for depot, depot_customers in clusters.items():
            routes.extend(
                run_pso_routing(
                    depot=depot,
                    customers=depot_customers,
                    dist_fn=self._dist,
                    cfg=self.cfg.pso,
                )
            )
        return Solution(routes=routes)

    def __repr__(self) -> str:
        return (
            f"GAPSOAlgorithm("
            f"ga_pop={self.cfg.ga.pop_size}, ga_gen={self.cfg.ga.n_gen}, "
            f"pso_pop={self.cfg.pso.pop_size}, pso_gen={self.cfg.pso.n_gen})"
        )

    def reroute_local(
        self,
        depot: Depot,
        customers: List[Customer],
    ) -> Solution:
        """Reroute only the customers for a single depot using the PSO router.

        Builds a local distance matrix for the depot+customers and runs the
        PSO-based routing on that subset. Returns a Solution.
        """
        return Solution(routes=run_pso_routing(
            depot=depot,
            customers=customers,
            dist_fn=self._dist,
            cfg=self.cfg.pso,
        ))
