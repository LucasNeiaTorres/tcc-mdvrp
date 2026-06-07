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


class DynamicRoutingProblem(ElementwiseProblem):
    """
    Permutation-encoded route optimisation for VRP-OD (origin-destination).
    Solves dynamic reroute: current_node -> customers -> real_depot.

    Parameters
    ----------
    current_start_node:
        Current position of vehicle (Customer or Depot), NOT the original depot.
    pending_customers:
        Customers still to be served.
    real_end_depot:
        The original depot where the route must end.
    dist_fn:
        Callable ``(a_index, b_index) -> float`` returning pre-computed distance.
    """

    def __init__(
        self,
        current_start_node: Customer | Depot,
        pending_customers: List[Customer],
        real_end_depot: Depot,
        dist_fn: Callable[[int, int], float],
    ) -> None:
        super().__init__(
            n_var=len(pending_customers),
            n_obj=1,
            xl=0,
            xu=len(pending_customers) - 1,
            vtype=int,
        )
        self.current_start_node = current_start_node
        self.pending_customers = pending_customers
        self.real_end_depot = real_end_depot
        self.dist_fn = dist_fn

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evaluate cost of a route from current_start_node -> customers -> real_end_depot."""
        if len(self.pending_customers) == 0:
            # No customers: just return to depot
            total = self.dist_fn(self.current_start_node.index, self.real_end_depot.index)
            out["F"] = total
            return

        x = np.rint(x).astype(int)
        ordered = [self.pending_customers[i] for i in x]

        total = 0.0
        # 1. From current location to first customer
        total += self.dist_fn(self.current_start_node.index, ordered[0].index)
        # 2. Between customers
        for i in range(len(ordered) - 1):
            total += self.dist_fn(ordered[i].index, ordered[i + 1].index)
        # 3. From last customer to real depot (NOT back to current node)
        total += self.dist_fn(ordered[-1].index, self.real_end_depot.index)

        out["F"] = total
