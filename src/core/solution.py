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

    def is_feasible(self) -> bool:
        """True if every route in the solution satisfies its constraints."""
        return all(route.is_feasible() for route in self.routes)

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
