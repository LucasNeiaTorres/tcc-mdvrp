from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import count
from typing import Any, List, Tuple

from core.entities import Customer, Depot
from core.solution import Solution
from utils.metrics import euclidean_distance
from utils.results_io import save_history_log
from .models import FailureEvent

SPEED_KMH = 50.0
_EVENT_SEQ = count()


@dataclass(frozen=True)
class SimulationEvent:
    trigger_time: float
    type: str
    payload: dict[str, Any]


QueueItem = Tuple[float, int, SimulationEvent]


def _dist(a: Depot | Customer, b: Depot | Customer) -> float:
    return euclidean_distance(a.x, a.y, b.x, b.y)


def _travel_time(a: Depot | Customer, b: Depot | Customer) -> float:
    return (_dist(a, b) / SPEED_KMH) * 60.0


def _arrival_events_from_solution(solution: Solution) -> List[SimulationEvent]:
    events: List[SimulationEvent] = []

    for route_id, route in enumerate(solution.routes, start=1):
        t = 0.0
        prev: Depot | Customer = route.depot

        for stop_idx, customer in enumerate(route.customers, start=1):
            t += _travel_time(prev, customer)
            events.append(
                SimulationEvent(
                    trigger_time=t,
                    type="arrival",
                    payload={
                        "route_id": route_id,
                        "depot_index": route.depot.index,
                        "node_index": customer.index,
                        "stop_index": stop_idx,
                    },
                )
            )
            t += customer.service_time
            prev = customer

        if route.customers:
            t += _travel_time(prev, route.depot)
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


def generate_event_queue(
    solution: Solution,
    failures: List[FailureEvent],
) -> List[QueueItem]:
    all_events: List[QueueItem] = []

    for event in _arrival_events_from_solution(solution):
        all_events.append((event.trigger_time, next(_EVENT_SEQ), event))

    for failure in failures:
        event = SimulationEvent(
            trigger_time=failure.trigger_time,
            type=failure.type,
            payload={
                "node_a": failure.node_a,
                "node_b": failure.node_b,
            },
        )
        all_events.append((event.trigger_time, next(_EVENT_SEQ), event))

    heapq.heapify(all_events)
    return all_events


def push_event(
    queue: List[QueueItem],
    event: SimulationEvent,
    seq_gen: Any = None,
) -> None:
    if seq_gen is None:
        next_seq = next(_EVENT_SEQ)
    else:
        next_seq = next(seq_gen)

    heapq.heappush(queue, (event.trigger_time, next_seq, event))


def pop_next_event(queue: List[QueueItem]) -> SimulationEvent | None:
    if not queue:
        return None
    _, _, event = heapq.heappop(queue)
    return event

def run_simulation(initial_solution: Solution, failures: List[FailureEvent], instance_name: str):
    queue = generate_event_queue(initial_solution, failures)
    
    history_log = []
    current_time = 0.0
    
    # Itera sobre os eventos na fila e guarda no historico
    while queue:
        event = pop_next_event(queue)
        if event is None:
            break
        print(f"Processing event: time={event.trigger_time:.2f}, type={event.type}, payload={event.payload}")
        current_time = event.trigger_time
        history_log.append((current_time, event.type, event.payload))
    
    
    # Salva json do historico  
    output_path = f"data/processed/simulation_logs/{instance_name}_log.json"
    save_history_log(output_path, instance_name, history_log)
    print(f"Saved simulation log to {output_path}")
    
    return history_log
    