"""Core domain entities for the MDVRP problem."""

from dataclasses import dataclass, field
from typing import List

from utils.metrics import euclidean_distance


@dataclass(frozen=True)
class Customer:
    index: int
    x: float
    y: float
    demand: float
    service_time: float


@dataclass(frozen=True)
class Depot:
    index: int
    x: float
    y: float
    max_duration: float
    max_capacity: float


@dataclass
class Route:
    depot: Depot
    customers: List[Customer] = field(default_factory=list)

    def total_demand(self) -> float:
        """Sum of all customer demands in this route."""
        return sum(c.demand for c in self.customers)

    def total_distance(self) -> float:
        """Total travel distance of the route."""
        if not self.customers:
            return 0.0

        nodes = [self.depot] + self.customers + [self.depot]
        return sum(
            euclidean_distance(nodes[i].x, nodes[i].y, nodes[i + 1].x, nodes[i + 1].y)
            for i in range(len(nodes) - 1)
        )

    def total_duration(self) -> float:
        """
        Total route duration including travel time and service times.
        Assumes travel time equals travel distance (unit speed).
        """
        return self.total_distance() + sum(c.service_time for c in self.customers)

    def is_feasible(self) -> bool:
        """Check whether this route satisfies both capacity and duration constraints."""
        capacity_ok = self.total_demand() <= self.depot.max_capacity
        duration_ok = (
            self.depot.max_duration == 0  # 0 means unconstrained
            or self.total_duration() <= self.depot.max_duration
        )
        return capacity_ok and duration_ok

    @property
    def depot_index(self) -> int:
        """1-based depot number, satisfies VisualizableRoute protocol."""
        return self.depot.index

    @property
    def customer_indices(self) -> List[int]:
        """Ordered customer indices, satisfies VisualizableRoute protocol."""
        return [c.index for c in self.customers]
