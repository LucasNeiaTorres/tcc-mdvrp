"""
GA-based routing entry points for MDVRP.

Thin wrappers that wire together the problem definitions, GA operators, and
Bellman split to solve the route-optimisation sub-problem for a single depot.

Sub-modules
-----------
ga_split        — Bellman (DAG) split algorithm
ga_local_search — Prins (2004) 9-move local search + route helpers
ga_problems     — RoutingProblem / DynamicRoutingProblem (pymoo)
ga_operators    — HeuristicSampling + LSMutation (pymoo)
"""

from typing import Callable, List

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.optimize import minimize

from core.entities import Customer, Depot, Route
from utils.config import GAConfig
from algorithms.ga_split import bellman_split
from algorithms.ga_problems import RoutingProblem, DynamicRoutingProblem
from algorithms.ga_operators import HeuristicSampling, LSMutation

def run_ga_routing(
    depot: Depot,
    customers: List[Customer],
    dist_fn: Callable[[int, int], float],
    cfg: GAConfig,
) -> List[Route]:
    """
    Run GA to find the best visiting order for a depot's customers, then use
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
        GAConfig loaded from config.yaml.

    Returns
    -------
    List of Routes covering all customers, one per vehicle.
    """
    if not customers:
        return []

    if len(customers) == 1:
        return [Route(depot=depot, customers=list(customers))]

    problem = RoutingProblem(depot=depot, customers=customers, dist_fn=dist_fn)

    algorithm = GA(
        pop_size=cfg.pop_size,
        sampling=HeuristicSampling(
            customers=customers,
            start_node=depot,
            dist_fn=dist_fn,
            n_heuristic=max(1, cfg.pop_size // 5),
        ),
        crossover=OrderCrossover(),
        mutation=LSMutation(
            depot=depot,
            customers=customers,
            dist_fn=dist_fn,
            prob=cfg.mutation_prob,
        ),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", cfg.n_gen),
        seed=cfg.seed,
        verbose=False,
    )

    ordered_customers = [customers[i] for i in result.X.astype(int)]
    segments = bellman_split(ordered_customers, depot, dist_fn)
    return [Route(depot=depot, customers=seg) for seg in segments]


def run_ga_reroute(
    current_start_node: Customer | Depot,
    pending_customers: List[Customer],
    real_end_depot: Depot,
    dist_fn: Callable[[int, int], float],
    cfg: GAConfig,
) -> List[Route]:
    """
    Run GA for dynamic reroute scenario (VRP-OD: origin-destination).
    Vehicle is at current_start_node and must visit pending_customers, then return to real_end_depot.
    This reroute is for a single vehicle, so it returns one open route.

    Parameters
    ----------
    current_start_node:
        Current vehicle position (Customer or Depot, NOT necessarily the original depot).
    pending_customers:
        Customers still to be served.
    real_end_depot:
        The original depot where the route must terminate.
    dist_fn:
        Pre-computed O(1) distance callable.
    cfg:
        GAConfig loaded from config.yaml.

    Returns
    -------
    List of Routes optimized for the dynamic scenario. For reroute, this is a
    single route: current_start_node -> ordered customers -> real_end_depot.
    """
    if not pending_customers:
        return []

    if len(pending_customers) == 1:
        return [Route(depot=real_end_depot, customers=list(pending_customers))]

    problem = DynamicRoutingProblem(
        current_start_node=current_start_node,
        pending_customers=pending_customers,
        real_end_depot=real_end_depot,
        dist_fn=dist_fn,
    )

    algorithm = GA(
        pop_size=cfg.pop_size,
        sampling=HeuristicSampling(
            customers=pending_customers,
            start_node=current_start_node,
            dist_fn=dist_fn,
            n_heuristic=max(1, cfg.pop_size // 5),
        ),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", cfg.n_gen),
        seed=cfg.seed,
        verbose=False,
    )

    ordered_customers = [pending_customers[i] for i in result.X.astype(int)]
    return [Route(depot=real_end_depot, customers=ordered_customers)]
