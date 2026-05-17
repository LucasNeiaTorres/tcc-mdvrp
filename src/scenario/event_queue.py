"""Event queue management for simulation."""

from dataclasses import dataclass
import heapq
from itertools import count
from typing import Any, List, Tuple

from core.entities import Depot, Customer, Route
from core.solution import Solution
from utils.metrics import euclidean_distance

UNIT_SPEED = 1.0
_EVENT_SEQ = count()


def euclidean_distance_fn(a: Depot | Customer, b: Depot | Customer) -> float:
    return euclidean_distance(a.x, a.y, b.x, b.y)


def travel_time(a: Depot | Customer, b: Depot | Customer) -> float:
    return euclidean_distance_fn(a, b) / UNIT_SPEED


@dataclass(frozen=True)
class SimulationEvent:
    trigger_time: float
    type: str
    payload: dict[str, Any]


QueueItem = Tuple[float, int, SimulationEvent]


def arrival_events_from_solution(solution: Solution) -> List[SimulationEvent]:
    """Generate arrival and service_end events from initial solution."""
    events: List[SimulationEvent] = []

    for route_id, route in enumerate(solution.routes, start=1):
        t = 0.0
        prev: Depot | Customer = route.depot

        for stop_idx, customer in enumerate(route.customers, start=1):
            t += travel_time(prev, customer)
            events.append(
                SimulationEvent(
                    trigger_time=t,
                    type="arrival",
                    payload={
                        "route_id": route_id,
                        "depot_index": route.depot.index,
                        "node_index": customer.index,
                        "stop_index": stop_idx,
                        "service_time": customer.service_time,
                    },
                )
            )
            t += customer.service_time
            events.append(
                SimulationEvent(
                    trigger_time=t,
                    type="service_end",
                    payload={
                        "route_id": route_id,
                        "depot_index": route.depot.index,
                        "node_index": customer.index,
                        "stop_index": stop_idx,
                    },
                )
            )
            prev = customer

        if route.customers:
            t += travel_time(prev, route.depot)
            events.append(
                SimulationEvent(
                    trigger_time=t,
                    type="arrival",
                    payload={
                        "route_id": route_id,
                        "depot_index": route.depot.index,
                        "node_index": route.depot.index,
                        "stop_index": len(route.customers) + 1,
                        "is_return_to_depot": True,
                    },
                )
            )

    return events


class EventQueue:
    """Manages the priority queue of simulation events."""

    def __init__(self):
        self.queue: List[QueueItem] = []

    def add_event(self, event: SimulationEvent) -> None:
        """Add an event to the queue."""
        heapq.heappush(self.queue, (event.trigger_time, next(_EVENT_SEQ), event))

    def add_events(self, events: List[SimulationEvent]) -> None:
        """Add multiple events to the queue."""
        for event in events:
            self.add_event(event)

    def pop_next(self) -> SimulationEvent | None:
        """Remove and return the next event from the queue."""
        if not self.queue:
            return None
        _, _, event = heapq.heappop(self.queue)
        return event

    def remove_future_events_for_route(self, route_id: int, current_time: float) -> None:
        """Remove all future events for a specific route after current_time."""
        self.queue[:] = [
            (t, seq, event)
            for t, seq, event in self.queue
            if not (
                t >= current_time
                and event.payload.get("route_id") == route_id
                and event.type in {"arrival", "service_end"}
            )
        ]
        heapq.heapify(self.queue)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.queue) == 0


def build_future_events_for_route(
    route_id: int,
    route: Route,
    start_node: Depot | Customer,
    start_time: float,
) -> List[SimulationEvent]:
    """Build a sequence of arrival/service_end events for a route starting at start_node."""
    events: List[SimulationEvent] = []
    t = start_time
    prev: Depot | Customer = start_node

    for stop_idx, customer in enumerate(route.customers, start=1):
        t += travel_time(prev, customer)
        events.append(
            SimulationEvent(
                trigger_time=t,
                type="arrival",
                payload={
                    "route_id": route_id,
                    "depot_index": route.depot.index,
                    "node_index": customer.index,
                    "stop_index": stop_idx,
                    "service_time": customer.service_time,
                },
            )
        )
        t += customer.service_time
        events.append(
            SimulationEvent(
                trigger_time=t,
                type="service_end",
                payload={
                    "route_id": route_id,
                    "depot_index": route.depot.index,
                    "node_index": customer.index,
                    "stop_index": stop_idx,
                },
            )
        )
        prev = customer

    if route.customers:
        t += travel_time(prev, route.depot)
        events.append(
            SimulationEvent(
                trigger_time=t,
                type="arrival",
                payload={
                    "route_id": route_id,
                    "depot_index": route.depot.index,
                    "node_index": route.depot.index,
                    "stop_index": len(route.customers) + 1,
                    "is_return_to_depot": True,
                },
            )
        )

    return events


def interpolate_position(
    node_a: Depot | Customer,
    node_b: Depot | Customer,
    elapsed_time: float,
) -> tuple[float, float]:
    """Interpolate vehicle position between two nodes."""
    total_time = travel_time(node_a, node_b)
    if total_time <= 0:
        return (node_a.x, node_a.y)

    ratio = max(0.0, min(1.0, elapsed_time / total_time))
    x = node_a.x + (node_b.x - node_a.x) * ratio
    y = node_a.y + (node_b.y - node_a.y) * ratio
    return (x, y)
