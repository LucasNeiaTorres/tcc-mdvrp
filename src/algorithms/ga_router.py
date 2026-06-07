"""
GA-based routing entry points for MDVRP.

Thin wrappers that wire together the problem definitions, GA operators, and
Vidal (2016) split algorithm to solve the route-optimisation sub-problem for a single depot.

Sub-modules
-----------
ga_split        — Vidal (2016) split algorithm
ga_local_search — Prins (2004) 9-move local search + route helpers
ga_problems     — RoutingProblem / DynamicRoutingProblem (pymoo)
ga_operators    — HeuristicSampling + LSMutation (pymoo)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.default import DefaultSingleObjectiveTermination

from core.entities import Customer, Depot, Route
from utils.config import GAConfig
from algorithms.ga_split import linear_split
from algorithms.ga_problems import RoutingProblem, DynamicRoutingProblem
from algorithms.ga_operators import HeuristicSampling, LSMutation, WellSpacedSurvival


class _ProgressCallback(Callback):
    """Prints a compact progress line every `interval` generations."""

    def __init__(self, depot_index: int, n_gen: int, interval: int = 5) -> None:
        super().__init__()
        self.depot_index = depot_index
        self.n_gen = n_gen
        self.interval = interval

    def notify(self, algorithm) -> None:
        gen = algorithm.n_gen
        if gen % self.interval != 0 and not algorithm.termination.has_terminated():
            return
        F = algorithm.pop.get("F").flatten()
        best = F.min()
        mean = F.mean()
        print(
            f"\r  depot {self.depot_index:>3} | gen {gen:>4}/{self.n_gen}"
            f" | best {best:>10.2f} | mean {mean:>10.2f}",
            end="",
            flush=True,
        )


@dataclass
class GADepotHistory:
    depot_index: int
    best: List[float] = field(default_factory=list)
    mean: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    clones_removed: int = 0
    stopped_early: bool = False
    feasible_seen: int = 0
    total_evaluated: int = 0


def run_ga_routing(
    depot: Depot,
    customers: List[Customer],
    dist_fn: Callable[[int, int], float],
    cfg: GAConfig,
) -> Tuple[List[Route], GADepotHistory]:
    """
    Run GA to find the best visiting order for a depot's customers, then use
    the Linear split to partition the giant tour into feasible routes.

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
    Tuple of (routes, history) where routes cover all customers and history
    holds per-generation best/mean/std fitness values.
    """
    if not customers:
        return [], GADepotHistory(depot_index=depot.index)

    if len(customers) == 1:
        return [Route(depot=depot, customers=list(customers))], GADepotHistory(depot_index=depot.index)

    problem = RoutingProblem(depot=depot, customers=customers, dist_fn=dist_fn,
                             capacity_penalty=cfg.capacity_penalty, duration_penalty=cfg.duration_penalty)

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
            local_search_max_iterations=cfg.local_search_max_iterations,
        ),
        survival=WellSpacedSurvival(delta=cfg.clone_delta),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        termination=TerminationCollection(
            DefaultSingleObjectiveTermination(
                ftol=cfg.stagnation_ftol,
                period=cfg.stagnation_period,
                n_max_gen=cfg.n_gen,
            ),
            get_termination("time", cfg.time_limit),
        ),
        seed=cfg.seed,
        save_history=True,
        verbose=False,
        callback=_ProgressCallback(depot_index=depot.index, n_gen=cfg.n_gen),
    )
    print()  # newline after the last \r update

    gens = [g.pop.get("F").flatten() for g in (result.history or [])]
    history = GADepotHistory(
        depot_index=depot.index,
        best=[float(f.min()) for f in gens],
        mean=[float(f.mean()) for f in gens],
        std=[float(f.std()) for f in gens],
        clones_removed=result.algorithm.survival.eliminated_count,
        stopped_early=len(gens) < cfg.n_gen,
        feasible_seen=problem.feasible_seen,
        total_evaluated=problem.total_evaluated,
    )

    ordered_customers = [customers[i] for i in result.X.astype(int)]
    return linear_split(ordered_customers, depot, dist_fn,
                         capacity_penalty=cfg.capacity_penalty, duration_penalty=cfg.duration_penalty), history


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
        crossover=OrderCrossover(),
        mutation=LSMutation(
            depot=real_end_depot,
            customers=pending_customers,
            dist_fn=dist_fn,
            prob=cfg.mutation_prob,
            local_search_max_iterations=cfg.local_search_max_iterations,
        ),
        survival=WellSpacedSurvival(delta=cfg.clone_delta),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        termination=TerminationCollection(
            DefaultSingleObjectiveTermination(
                ftol=cfg.stagnation_ftol,
                period=cfg.stagnation_period,
                n_max_gen=cfg.n_gen,
            ),
            get_termination("time", cfg.time_limit),
        ),
        seed=cfg.seed,
        verbose=False,
    )

    ordered_customers = [pending_customers[i] for i in result.X.astype(int)]
    return [Route(depot=real_end_depot, customers=ordered_customers)]
