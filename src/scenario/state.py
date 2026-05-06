from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.entities import Customer, Route


def _normalize_edge(node_a: int, node_b: int) -> tuple[int, int]:
    """Return a canonical undirected representation of an edge."""
    return (node_a, node_b) if node_a <= node_b else (node_b, node_a)


@dataclass
class VehicleState:
    """
    Mutable state of a vehicle during simulation.
    
    Tracks the current position, progress, and status of a vehicle executing
    a planned route, allowing the simulator to handle dynamic events like
    blocked edges and on-the-fly rerouting.
    
    Attributes
    ----------
    route_id:
        Unique identifier for this vehicle/route (1-indexed).
    route:
        Reference to the original planned Route (immutable).
    current_node_index:
        Index of the node where the vehicle currently is.
    next_stop_index:
        Index (1-based) of the next stop in the original route sequence.
    last_event_time_min:
        Timestamp (in minutes) of the vehicle's last event.
    visited_customer_ids:
        Set of customer indices already visited by this vehicle.
    pending_customer_ids:
        Set of customer indices still to visit in this route.
    capacity_total:
        Total vehicle capacity used by simulation checks.
    load_current:
        Current load carried by the vehicle.
    status:
        Current state: "en_route", "servicing", "at_depot", "blocked".
    """
    
    route_id: int
    route: Route
    current_node_index: int
    next_stop_index: int
    last_event_time_min: float
    visited_customer_ids: set[int] = field(default_factory=set)
    pending_customer_ids: set[int] = field(default_factory=set)
    capacity_total: float = 0.0
    load_current: float = 0.0
    customers_by_index: dict[int, Customer] = field(default_factory=dict)
    status: str = "at_depot"

    def __post_init__(self) -> None:
        """Initialize pending customers and capacity defaults."""
        if not self.pending_customer_ids:
            self.pending_customer_ids = {c.index for c in self.route.customers}
        if not self.customers_by_index:
            self.customers_by_index = {c.index: c for c in self.route.customers}
        if self.capacity_total <= 0:
            self.capacity_total = self.route.depot.max_capacity
        if self.load_current < 0:
            self.load_current = 0.0

    def mark_visited(self, customer_id: int) -> None:
        """Move a customer from pending to visited."""
        self.visited_customer_ids.add(customer_id)
        self.pending_customer_ids.discard(customer_id)

    @property
    def capacity_remaining(self) -> float:
        """Remaining capacity available for pickups."""
        return max(0.0, self.capacity_total - self.load_current)

    def can_add_load(self, demand: float) -> bool:
        """Return True if adding demand does not exceed total capacity."""
        return demand <= self.capacity_remaining

    def add_load(self, demand: float) -> None:
        """Increase carried load while respecting capacity constraints."""
        if demand < 0:
            raise ValueError("demand must be non-negative")
        if not self.can_add_load(demand):
            raise ValueError("vehicle capacity exceeded")
        self.load_current += demand

    def remove_load(self, amount: float) -> None:
        """Decrease carried load (e.g., unloading at depot)."""
        if amount < 0:
            raise ValueError("amount must be non-negative")
        self.load_current = max(0.0, self.load_current - amount)

    def is_complete(self) -> bool:
        """Check if all customers have been visited."""
        return len(self.pending_customer_ids) == 0

    def progress_pct(self) -> float:
        """Return completion percentage (0.0 to 100.0)."""
        total = len(self.route.customers)
        if total == 0:
            return 100.0
        visited = len(self.visited_customer_ids)
        return (visited / total) * 100.0

    def _planned_nodes(self) -> list[int]:
        """Return planned path node indices as depot -> customers -> depot."""
        return [
            self.route.depot.index,
            *[c.index for c in self.route.customers],
            self.route.depot.index,
        ]

    def current_leg(self) -> Optional[tuple[int, int]]:
        """Return current traversal leg based on next_stop_index."""
        nodes = self._planned_nodes()
        i = self.next_stop_index - 1
        if i < 0 or i >= len(nodes) - 1:
            return None
        return (nodes[i], nodes[i + 1])

    def is_travelling_edge(self, node_a: int, node_b: int) -> bool:
        """True if vehicle is currently traversing edge (node_a, node_b)."""
        if self.status != "en_route":
            return False

        leg = self.current_leg()
        if leg is None:
            return False

        return _normalize_edge(*leg) == _normalize_edge(node_a, node_b)

    def has_future_broken_edge(
        self,
        broken_edges: set[tuple[int, int]],
        include_current_leg: bool = True,
    ) -> bool:
        """True if any remaining planned leg is in broken_edges."""
        nodes = self._planned_nodes()
        start = self.next_stop_index - 1 if include_current_leg else self.next_stop_index

        for i in range(start, len(nodes) - 1):
            if _normalize_edge(nodes[i], nodes[i + 1]) in broken_edges:
                return True
        return False
