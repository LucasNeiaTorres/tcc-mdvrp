"""
PSO-based routing module for MDVRP.

Solves the route-optimisation sub-problem: given a depot and a fixed set of
customers already assigned to it, find both the visiting order and the vehicle
partition that minimise total travel distance.

SPV encoding (Smallest Position Value)
---------------------------------------
pymoo's built-in PSO operates on real-valued vectors.  A permutation is
obtained by ranking (argsort) the position vector:

    x = [0.72, 0.11, 0.55]  →  argsort → [1, 2, 0]  →  visit C2, C3, C1

This avoids any custom operator and is a well-established technique for
permutation problems in continuous-space swarms.

Bellman split
-------------
For each candidate permutation the fitness is computed by the Bellman (DAG
shortest-path) split algorithm (Prins, 2004).  The PSO therefore co-optimises
both the visiting order and the vehicle boundaries.

Fitness
-------
    f(x) = min-cost partition of the giant tour into capacity-feasible routes
           = Σ_k [ dist(depot, r_k[0])
                   + Σ dist(r_k[i], r_k[i+1])
                   + dist(r_k[-1], depot) ]
"""

from typing import Callable, List

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from core.entities import Customer, Depot, Route
from utils.config import PSOConfig


def bellman_split(
    ordered: List[Customer],
    depot: Depot,
    dist_fn: Callable[[int, int], float],
) -> List[List[Customer]]:
    """
    Optimally partition an ordered customer sequence into capacity- and
    duration-feasible vehicle routes using the Bellman (DAG shortest-path)
    split algorithm.

    Each contiguous segment becomes one vehicle route
    ``depot → seg[0] → ... → seg[-1] → depot``.  The DP finds the cut points
    that minimise total travel distance subject to each segment's total demand
    not exceeding ``depot.max_capacity`` and total duration (travel + service
    times) not exceeding ``depot.max_duration`` (when non-zero).

    Customers whose individual demand or duration exceeds the limits are placed
    in their own route so that all customers are always routed.
    """
    n = len(ordered)
    INF = float("inf")
    dp = [INF] * (n + 1)
    pred = [-1] * (n + 1)
    dp[0] = 0.0

    for i in range(n):
        if dp[i] == INF:
            continue
        load = 0.0
        service = 0.0
        prev_idx = depot.index
        travel = 0.0
        for j in range(i, n):
            load += ordered[j].demand
            service += ordered[j].service_time
            # Allow singleton segments even when constraints are exceeded.
            if load > depot.max_capacity and j > i:
                break
            travel += dist_fn(prev_idx, ordered[j].index)
            prev_idx = ordered[j].index
            route_dist = travel + dist_fn(ordered[j].index, depot.index)
            if depot.max_duration > 0 and route_dist + service > depot.max_duration and j > i:
                break
            total = dp[i] + route_dist
            if total < dp[j + 1]:
                dp[j + 1] = total
                pred[j + 1] = i

    # Backtrack from n to 0 to recover segments.
    segments: List[List[Customer]] = []
    j = n
    while j > 0:
        i = pred[j]
        segments.append(list(ordered[i:j]))
        j = i
    segments.reverse()
    return segments


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
        segments = bellman_split(ordered, self.depot, self.dist_fn)

        total = 0.0
        for seg in segments:
            total += self.dist_fn(self.depot.index, seg[0].index)
            for i in range(len(seg) - 1):
                total += self.dist_fn(seg[i].index, seg[i + 1].index)
            total += self.dist_fn(seg[-1].index, self.depot.index)

        out["F"] = total


def two_opt(route: List[Customer], depot: Depot, dist_fn: Callable[[int, int], float]) -> List[Customer]:
    """
    Improve a single-vehicle route with 2-opt local search.

    Repeatedly reverses sub-sequences between indices i and k whenever doing
    so reduces the total round-trip distance.  Runs until no improving swap
    remains (first-improvement strategy, O(n²) per pass).
    """
    best = list(route)
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 2, n):
                # Edges being removed: (i-1 → i) and (k → k+1)
                a = depot.index if i == 0 else best[i - 1].index
                b = best[i].index
                c = best[k].index
                d = depot.index if k == n - 1 else best[k + 1].index

                if dist_fn(a, c) + dist_fn(b, d) < dist_fn(a, b) + dist_fn(c, d):
                    best[i : k + 1] = best[i : k + 1][::-1]
                    improved = True
    return best


def run_pso_routing(
    depot: Depot,
    customers: List[Customer],
    dist_fn: Callable[[int, int], float],
    cfg: PSOConfig,
) -> List[Route]:
    """
    Run PSO to find the best visiting order for a depot's customers, then use
    the Bellman split to partition the giant tour into capacity-feasible routes.

    Parameters
    ----------
    depot:
        Depot that serves this cluster.
    customers:
        All customers assigned to this depot.
    dist_fn:
        Pre-computed O(1) distance callable from ``MDVRPAlgorithm._dist``.
    cfg:
        PSOConfig loaded from config.yaml.

    Returns
    -------
    List of Routes covering all customers, one per vehicle.
    """
    if not customers:
        return []

    if len(customers) == 1:
        return [Route(depot=depot, customers=list(customers))]

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
    segments = bellman_split(ordered_customers, depot, dist_fn)
    # return [Route(depot=depot, customers=seg) for seg in segments]
    return [Route(depot=depot, customers=two_opt(seg, depot, dist_fn)) for seg in segments]
