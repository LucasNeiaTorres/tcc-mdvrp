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
from .state import VehicleState

UNIT_SPEED = 1.0
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
    return _dist(a, b) / UNIT_SPEED


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

def _build_vehicle_states(initial_solution: Solution) -> dict[int, VehicleState]:
    """Create one mutable VehicleState per route in the initial solution."""
    vehicle_states: dict[int, VehicleState] = {}

    for route_id, route in enumerate(initial_solution.routes, start=1):
        vehicle_states[route_id] = VehicleState(
            route_id=route_id,
            route=route,
            current_node_index=route.depot.index,
            next_stop_index=1,
            last_event_time_min=0.0,
            status="at_depot",
        )

    return vehicle_states


def run_simulation(initial_solution: Solution, failures: List[FailureEvent], instance_name: str):
    queue = generate_event_queue(initial_solution, failures)
    vehicle_states = _build_vehicle_states(initial_solution)
    
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
        
        if event.type == "arrival":
            _handle_arrival(event, current_time, vehicle_states)

        elif event.type == "service_end":
            _handle_service_end(event, current_time, vehicle_states)
            
        elif event.type == "edge_block":
            _handle_disaster(event, current_time, queue, vehicle_states)
    
    
    # Salva json do historico  
    output_path = f"data/processed/simulation_logs/{instance_name}_log.json"
    save_history_log(output_path, instance_name, history_log)
    print(f"Saved simulation log to {output_path}")
    
    return history_log
    
    
def _handle_arrival(
    event: SimulationEvent,
    current_time: float,
    vehicle_states: dict[int, VehicleState],
):
    route_id = event.payload.get("route_id")
    node_index = event.payload.get("node_index")
    stop_index = event.payload.get("stop_index")

    if route_id is None or route_id not in vehicle_states:
        return

    state = vehicle_states[route_id]
    state.current_node_index = int(node_index)
    state.next_stop_index = int(stop_index) + 1
    state.last_event_time_min = current_time
    
    # print(f"Vehicle {route_id} arrived at node {node_index} at time {current_time:.2f} min (stop {stop_index}), state updated: {state}")

    if event.payload.get("is_return_to_depot"):
        state.remove_load(state.load_current)
        state.status = "at_depot"
        return

    customer = state.customers_by_index.get(int(node_index))
    if customer is None:
        return

    if not state.can_add_load(customer.demand):
        state.status = "blocked"
        print(
            f"Vehicle {route_id} cannot load demand {customer.demand} at node {node_index}; "
            f"remaining capacity={state.capacity_remaining:.2f}"
        )
        return

    state.add_load(customer.demand)
    state.mark_visited(int(node_index))
    state.status = "servicing"


def _handle_service_end(
    event: SimulationEvent,
    current_time: float,
    vehicle_states: dict[int, VehicleState],
):
    route_id = event.payload.get("route_id")

    if route_id is None or route_id not in vehicle_states:
        return

    state = vehicle_states[route_id]
    state.last_event_time_min = current_time

    if state.status == "servicing":
        state.status = "en_route"

def _handle_disaster(
    event: SimulationEvent,
    current_time: float,
    queue: List[QueueItem],
    vehicle_states: dict[int, VehicleState],
):
    # Lógica de acionar o PSO e reescrever a fila (queue)
    del current_time, queue, vehicle_states