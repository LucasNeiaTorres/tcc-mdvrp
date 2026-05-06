"""
CCBC+PSO Cluster-first, Route-second algorithm for MDVRP.

Phase 1 — Clustering (CCBC):
    Each customer is assigned to a depot via Constrained Centroid-Based
    Clustering (CCBC) — an augmented capacitated k-means with vehicle-level
    slots, multi-start Voronoi initialisation, and two-phase boundary
    resolution.

Phase 2 — Routing (PSO + Bellman split):
    Within each depot's cluster a PSO optimises the giant-tour order via SPV
    encoding.  For each candidate permutation the Bellman split finds the
    optimal vehicle partition given the depot's capacity.

Usage
-----
    cfg = load_config()
    algorithm = CCBCPSOAlgorithm(cfg)
    solution = algorithm.solve(customers, depots)
"""

from typing import Dict, List

from algorithms.base import ClusterFirstAlgorithm
from algorithms.ccbc_cluster import run_ccbc_clustering
from algorithms.pso_router import run_pso_routing
from core.entities import Customer, Depot
from core.solution import Solution
from utils.config import AppConfig, load_config


class CCBCPSOAlgorithm(ClusterFirstAlgorithm):
    """
    Cluster-first, route-second MDVRP solver using CCBC and PSO.

    The distance matrix is built once by ``ClusterFirstAlgorithm.solve()``
    before either phase runs, so ``route()`` can use ``self._dist()`` for
    O(1) lookups.  The CCBC phase uses raw (x, y) coordinates directly
    for centroid arithmetic.

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


    def cluster(
        self, customers: List[Customer], depots: List[Depot]
    ) -> Dict[Depot, List[Customer]]:
        """Phase 1: assign customers to depots via CCBC."""
        return run_ccbc_clustering(
            customers=customers,
            depots=depots,
            cfg=self.cfg.ccbc,
        )

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
            f"CCBCPSOAlgorithm("
            f"ccbc_iter={self.cfg.ccbc.max_iter}, ccbc_starts={self.cfg.ccbc.n_starts}, "
            f"pso_pop={self.cfg.pso.pop_size}, pso_gen={self.cfg.pso.n_gen})"
        )