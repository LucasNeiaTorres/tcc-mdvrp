"""
Solution representation and evaluation for the MDVRP problem.

A Solution is a collection of Routes, each assigning an ordered
sequence of customers to a vehicle departing from a depot.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from core.entities import Customer, Depot, Route


@dataclass
class Solution:
    routes: List[Route] = field(default_factory=list)

    def total_cost(self) -> float:
        """Total travel distance across all routes."""
        return sum(route.total_distance() for route in self.routes)

    def routes_are_feasible(self) -> bool:
        """True if every route satisfies capacity and duration constraints."""
        return all(route.is_feasible() for route in self.routes)

    def fleet_is_feasible(self) -> bool:
        """True if each depot's route count does not exceed its vehicle limit."""
        per_depot_count: Dict[int, int] = {}
        for route in self.routes:
            per_depot_count[route.depot.index] = per_depot_count.get(route.depot.index, 0) + 1

        return all(
            per_depot_count.get(route.depot.index, 0) <= route.depot.max_vehicles
            for route in self.routes
        )

    def is_feasible(self) -> bool:
        """True if all routes are individually feasible.

        Fleet limits are checked separately via :meth:`fleet_is_feasible`.
        """
        return self.routes_are_feasible()

    def fully_feasible(self) -> bool:
        """True if route constraints and per-depot vehicle limits are satisfied."""
        return self.routes_are_feasible() and self.fleet_is_feasible()

    @property
    def visualizable_routes(self) -> List[Route]:
        """All routes, satisfies VisualizableSolution protocol."""
        return self.routes

    def feasibility_report(self) -> Dict[int, dict]:
        """Returns a per-route feasibility breakdown useful for debugging."""
        report = {}
        for i, route in enumerate(self.routes):
            report[i] = {
                "depot": route.depot.index,
                "customers": [c.index for c in route.customers],
                "routes_for_depot": sum(1 for r in self.routes if r.depot.index == route.depot.index),
                "max_vehicles": route.depot.max_vehicles,
                "fleet_ok": sum(1 for r in self.routes if r.depot.index == route.depot.index) <= route.depot.max_vehicles,
                "demand": route.total_demand(),
                "max_capacity": route.depot.max_capacity,
                "capacity_ok": route.total_demand() <= route.depot.max_capacity,
                "duration": route.total_duration(),
                "max_duration": route.depot.max_duration,
                "duration_ok": (
                    route.depot.max_duration == 0
                    or route.total_duration() <= route.depot.max_duration
                ),
            }
        return report

    def feasibility_overview(self) -> Dict[str, bool]:
        """Return the global feasibility flags for this solution."""
        return {
            "routes_ok": self.routes_are_feasible(),
            "fleet_ok": self.fleet_is_feasible(),
            "fully_feasible": self.fully_feasible(),
        }
