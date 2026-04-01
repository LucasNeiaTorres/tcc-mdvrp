"""
PSO-based routing module for MDVRP.

Solves the route-optimisation sub-problem: given a depot and a fixed set of
customers already assigned to it, find the visiting order that minimises the
total travel distance of the round-trip.

SPV encoding (Smallest Position Value)
---------------------------------------
pymoo's built-in PSO operates on real-valued vectors.  A permutation is
obtained by ranking (argsort) the position vector:

    x = [0.72, 0.11, 0.55]  →  argsort → [1, 2, 0]  →  visit C2, C3, C1

This avoids any custom operator and is a well-established technique for
permutation problems in continuous-space swarms.

Fitness
-------
    f(x) = dist(depot, customers[perm[0]])
           + Σ dist(customers[perm[i]], customers[perm[i+1]])
           + dist(customers[perm[-1]], depot)
"""

from typing import Callable, List

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from core.entities import Customer, Depot, Route
from utils.config import PSOConfig


class RoutingProblem(ElementwiseProblem):
    """
    SPV-encoded route optimisation problem for pymoo PSO.

    Parameters
    ----------
    depot:
        The depot that starts and ends the route.
    customers:
        Ordered list of Customer entities to visit (order will be optimised).
    dist_fn:
        Callable ``(a_index, b_index) -> float`` returning pre-computed
        distance between any two node indices.
    """

    def __init__(
        self,
        depot: Depot,
        customers: List[Customer],
        dist_fn: Callable[[int, int], float],
    ) -> None:
        super().__init__(
            n_var=len(customers),
            n_obj=1,
            xl=0.0,
            xu=1.0,
        )
        self.depot = depot
        self.customers = customers
        self.dist_fn = dist_fn

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        perm = np.argsort(x)
        ordered = [self.customers[i] for i in perm]

        cost = self.dist_fn(self.depot.index, ordered[0].index)
        for i in range(len(ordered) - 1):
            cost += self.dist_fn(ordered[i].index, ordered[i + 1].index)
        cost += self.dist_fn(ordered[-1].index, self.depot.index)

        out["F"] = cost


def run_pso_routing(
    depot: Depot,
    customers: List[Customer],
    dist_fn: Callable[[int, int], float],
    cfg: PSOConfig,
) -> Route:
    """
    Run PSO to find the best visiting order for a cluster of customers.

    Parameters
    ----------
    depot:
        Depot that serves this cluster.
    customers:
        Customers assigned to this depot (to be ordered).
    dist_fn:
        Pre-computed O(1) distance callable from ``MDVRPAlgorithm._dist``.
    cfg:
        PSOConfig loaded from config.yaml.

    Returns
    -------
    A Route with customers in the optimised visiting order.
    """
    if not customers:
        return Route(depot=depot)

    if len(customers) == 1:
        return Route(depot=depot, customers=list(customers))

    problem = RoutingProblem(depot=depot, customers=customers, dist_fn=dist_fn)

    algorithm = PSO(
        pop_size=cfg.pop_size,
        w=cfg.inertia,
        c1=cfg.c1,
        c2=cfg.c2,
        adaptive=cfg.adaptive,
    )

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", cfg.n_gen),
        seed=cfg.seed,
        verbose=False,
    )

    perm = np.argsort(result.X)
    ordered_customers = [customers[i] for i in perm]
    return Route(depot=depot, customers=ordered_customers)
