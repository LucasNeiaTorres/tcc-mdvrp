"""
pymoo problem definitions for GA-based VRP routing.

Provides the ElementwiseProblem subclasses that encode the VRP fitness
function for the standard depot-to-depot case and the dynamic
origin-destination reroute case.
"""

from typing import Callable, List

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from core.entities import Customer, Depot
from core.solution import Solution
from algorithms.ga_split import linear_split


class RoutingProblem(ElementwiseProblem):
    """
    Permutation-encoded route optimisation problem for pymoo GA.
    Solves standard VRP: depot -> customers -> depot.

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
        capacity_penalty: float,
        duration_penalty: float,
    ) -> None:
        super().__init__(
            n_var=len(customers),
            n_obj=1,
            xl=0,
            xu=len(customers) - 1,
            vtype=int,
        )
        self.depot = depot
        self.customers = customers
        self.dist_fn = dist_fn
        self.start_node = depot
        self.end_depot = depot
        self.capacity_penalty = capacity_penalty
        self.duration_penalty = duration_penalty
        self.feasible_seen: int = 0
        self.total_evaluated: int = 0

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        x = np.rint(x).astype(int)
        ordered = [self.customers[i] for i in x]
        segments = linear_split(ordered, self.end_depot, self.dist_fn,
                                 capacity_penalty=self.capacity_penalty, duration_penalty=self.duration_penalty)

        self.total_evaluated += 1
        sol = Solution(routes=segments)
        if sol.fully_feasible():
            self.feasible_seen += 1

        # Fitness = total travel distance + soft-constraint penalties.
        total = sum(seg.total_distance() for seg in segments)
        for seg in segments:
            cap_excess = max(0.0, seg.total_demand() - self.depot.max_capacity)
            total += self.capacity_penalty * cap_excess
            if self.depot.max_duration > 0:
                dur_excess = max(0.0, seg.total_duration() - self.depot.max_duration)
                total += self.duration_penalty * dur_excess

        excess_vehicles = max(0, len(segments) - self.depot.max_vehicles)
        out["F"] = total + excess_vehicles * total
