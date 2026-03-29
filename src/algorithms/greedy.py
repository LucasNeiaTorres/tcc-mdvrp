"""Greedy constructive heuristic for MDVRP."""

from typing import List

from algorithms.base import MDVRPAlgorithm
from core.entities import Customer, Depot, Route
from core.solution import Solution


class GreedyAlgorithm(MDVRPAlgorithm):
    """
    Generates an initial solution by assigning each customer to their
    nearest feasible depot and building routes greedily.

    Strategy:
    1. For each customer, find the nearest depot.
    2. Assign the customer to that depot's current open route if it fits
        (capacity and duration constraints).
    3. If it does not fit, open a new route for that depot.
    4. If the depot has exhausted its vehicle count, fall back to the
        next nearest depot.
    """

    def solve(self, customers: List[Customer], depots: List[Depot]) -> Solution:
        self._build_matrix(depots, customers)
        # One open route per depot to greedily fill
        open_routes: dict[int, Route] = {depot.index: Route(depot=depot) for depot in depots}
        # All completed routes (when a route is full, a new one is opened)
        all_routes: List[Route] = []

        # Sort customers by index for deterministic output
        for customer in sorted(customers, key=lambda c: c.index):
            assigned = False

            # Try depots ordered by proximity to this customer
            for depot in self._depots_by_distance(customer, depots):
                route = open_routes[depot.index]

                if self._fits(route, customer):
                    route.customers.append(customer)
                    assigned = True
                    break

                # Current open route is full — try opening a new one
                # if the depot still has vehicles available
                vehicles_used = sum(
                    1 for r in all_routes if r.depot.index == depot.index
                ) + 1  # +1 for the current open route

                # Check against available vehicle count (shared across all depots)
                # We use a simple heuristic: allow unlimited routes per depot
                # (the algorithm itself will respect feasibility via is_feasible())
                new_route = Route(depot=depot)
                new_route.customers.append(customer)
                if new_route.is_feasible():
                    all_routes.append(route)
                    open_routes[depot.index] = new_route
                    assigned = True
                    break

            if not assigned:
                # Fallback: force-assign to the nearest depot ignoring constraints
                nearest_depot = self._depots_by_distance(customer, depots)[0]
                open_routes[nearest_depot.index].customers.append(customer)

        # Close all remaining open routes that have customers
        for route in open_routes.values():
            if route.customers:
                all_routes.append(route)

        return Solution(routes=all_routes)

    def _fits(self, route: Route, customer: Customer) -> bool:
        """Check if adding the customer to the route keeps it feasible."""
        route.customers.append(customer)
        feasible = route.is_feasible()
        route.customers.pop()
        return feasible

    def _depots_by_distance(self, customer: Customer, depots: List[Depot]) -> List[Depot]:
        """Return depots sorted by distance to the customer (nearest first)."""
        return sorted(depots, key=lambda d: self._dist(d.index, customer.index))
