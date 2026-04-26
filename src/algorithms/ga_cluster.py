"""
GA-based clustering module for MDVRP.

Solves the depot-assignment sub-problem: given a set of customers and depots,
find the assignment of each customer to a depot that minimises total
customer-to-depot travel while penalising capacity violations.

Chromosome encoding
-------------------
Each individual is an integer vector of length ``n_customers``.
Gene ``i`` holds a 0-based depot slot index, i.e. the position of the
assigned depot inside the ``depots`` list.  This keeps the bounds
``[0, n_depots - 1]`` natural for pymoo's integer operators.

Fitness
-------
    f(x) = Σ dist(customer_i, depots[x[i]])
           + capacity_penalty × Σ max(0, load[depot] − depot.max_capacity × depot.max_vehicles)

The capacity penalty turns a hard constraint into a soft one: solutions that
violate capacity are still explored but are strongly discouraged.
"""

from typing import Callable, Dict, List

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from core.entities import Customer, Depot
from utils.config import GAConfig


class DepotAssignmentProblem(ElementwiseProblem):
    """
    Integer-encoded depot assignment problem for pymoo.

    Parameters
    ----------
    customers:
        Ordered list of Customer entities to assign.
    depots:
        Ordered list of Depot entities (0-based slots).
    dist_fn:
        Callable ``(a_index, b_index) -> float`` returning pre-computed
        distance between any two node indices.
    capacity_penalty:
        Multiplier applied to excess demand on each overloaded depot.
    """

    def __init__(
        self,
        customers: List[Customer],
        depots: List[Depot],
        dist_fn: Callable[[int, int], float],
        capacity_penalty: float,
    ) -> None:
        super().__init__(
            n_var=len(customers),
            n_obj=1,
            xl=0,
            xu=len(depots) - 1,
            vtype=int,
        )
        self.customers = customers
        self.depots = depots
        self.dist_fn = dist_fn
        self.capacity_penalty = capacity_penalty

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        # Travel cost: sum of customer → assigned depot distances
        travel = sum(
            self.dist_fn(self.depots[x[i]].index, self.customers[i].index)
            for i in range(len(self.customers))
        )

        # Capacity penalty: accumulate load per depot slot
        loads: Dict[int, float] = {i: 0.0 for i in range(len(self.depots))}
        service_times: Dict[int, float] = {i: 0.0 for i in range(len(self.depots))}
        for i, slot in enumerate(x):
            loads[slot] += self.customers[i].demand
            service_times[slot] += self.customers[i].service_time

        capacity_penalty = self.capacity_penalty * sum(
            max(0.0, loads[i] - (self.depots[i].max_capacity * self.depots[i].max_vehicles))
            for i in range(len(self.depots))
        )

        duration_penalty = self.capacity_penalty * sum(
            max(0.0, service_times[i] - (self.depots[i].max_duration * self.depots[i].max_vehicles))
            for i in range(len(self.depots))
            if self.depots[i].max_duration > 0
        )

        out["F"] = travel + capacity_penalty + duration_penalty


def run_ga_clustering(
    customers: List[Customer],
    depots: List[Depot],
    dist_fn: Callable[[int, int], float],
    cfg: GAConfig,
) -> Dict[Depot, List[Customer]]:
    """
    Run the GA and decode the best chromosome into a depot → customers map.

    Parameters
    ----------
    customers:
        All Customer entities to cluster.
    depots:
        All Depot entities available as cluster centres.
    dist_fn:
        Pre-computed O(1) distance callable from ``MDVRPAlgorithm._dist``.
    cfg:
        GAConfig loaded from config.yaml.

    Returns
    -------
    Dict mapping each Depot to its assigned list of Customers.
    Depots with no assigned customers are included with an empty list.
    """
    problem = DepotAssignmentProblem(
        customers=customers,
        depots=depots,
        dist_fn=dist_fn,
        capacity_penalty=cfg.capacity_penalty,
    )

    algorithm = GA(
        pop_size=cfg.pop_size,
        sampling=IntegerRandomSampling(),
        crossover=TwoPointCrossover(prob=cfg.crossover_prob),
        mutation=PM(eta=cfg.mutation_eta, repair=RoundingRepair(), vtype=float),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", cfg.n_gen),
        seed=cfg.seed,
        verbose=False,
    )

    # Decode best chromosome: gene i → 0-based depot slot
    best = result.X.astype(int)
    clusters: Dict[Depot, List[Customer]] = {depot: [] for depot in depots}
    for i, slot in enumerate(best):
        clusters[depots[slot]].append(customers[i])

    return clusters
