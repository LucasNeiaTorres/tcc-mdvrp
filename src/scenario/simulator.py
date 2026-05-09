from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import count
from pathlib import Path
from typing import Any, List, Tuple

from core.entities import Customer, Depot, Route
from core.solution import Solution
from algorithms.base import MDVRPAlgorithm
from utils.metrics import euclidean_distance
from utils.results_io import save_history_log, save_reroute_result
from .models import FailureEvent
from .state import VehicleState, _normalize_edge

UNIT_SPEED = 1.0
_EVENT_SEQ = count()
SIMULATION_LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "simulation_logs"


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


def _interpolate_position(
    node_a: Depot | Customer,
    node_b: Depot | Customer,
    elapsed_time: float,
) -> tuple[float, float]:
    total_time = _travel_time(node_a, node_b)
    if total_time <= 0:
        return (node_a.x, node_a.y)

    ratio = max(0.0, min(1.0, elapsed_time / total_time))
    x = node_a.x + (node_b.x - node_a.x) * ratio
    y = node_a.y + (node_b.y - node_a.y) * ratio
    return (x, y)


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

def _remove_future_events_for_route(queue, route_id, current_time):
    queue[:] = [
        (t, seq, event)
        for t, seq, event in queue
        if not (
            t >= current_time
            and event.payload.get("route_id") == route_id
            and event.type in {"arrival", "service_end"}
        )
    ]
    heapq.heapify(queue)


def _build_future_events_for_route(
    route_id: int,
    route: Route,
    start_node: Depot | Customer,
    start_time: float,
) -> List[SimulationEvent]:
    events: List[SimulationEvent] = []
    t = start_time
    prev: Depot | Customer = start_node

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


def _insert_events(queue: List[QueueItem], events: List[SimulationEvent]) -> None:
    for event in events:
        push_event(queue, event)
        


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

def run_simulation(
    initial_solution: Solution,
    failures: List[FailureEvent],
    instance_name: str,
    algorithm: MDVRPAlgorithm
):
    current_solution = initial_solution
    expected_customer_indices = [
        customer.index
        for route in initial_solution.routes
        for customer in route.customers
    ]
    queue = generate_event_queue(current_solution, failures)
    vehicle_states = _build_vehicle_states(current_solution)
    reroute_count = 0
    history_log = []
    total_wasted_distance = 0.0
    current_time = 0.0
    
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
            reroute_inc, wasted = _handle_disaster(
                event,
                current_time,
                queue,
                vehicle_states,
                current_solution,
                algorithm,
                instance_name,
                reroute_count,
            )
            reroute_count += reroute_inc
            total_wasted_distance += wasted
    
    
    # Salva json do historico  
    output_path = SIMULATION_LOG_DIR / f"{instance_name}_log.json"
    save_history_log(
        str(output_path),
        instance_name,
        history_log,
        expected_customer_indices=expected_customer_indices,
    )
    print(f"Saved simulation log to {output_path}")

    # Compute realized cost: sum of current solution routes + wasted travel
    planned_cost = float(current_solution.total_cost())
    realized_cost = planned_cost + float(total_wasted_distance)

    # Compute visited / unserved customers from vehicle states
    visited = set()
    for vs in vehicle_states.values():
        visited |= set(vs.visited_customer_ids)
    expected_set = set(expected_customer_indices)
    unserved_customers = sorted(list(expected_set - visited))

    # Collect broken edges from history_log and check if any current route still uses them
    broken_edges: set[tuple[int, int]] = set()
    for _, etype, payload in history_log:
        if etype == "edge_block":
            broken_edges.add(_normalize_edge(payload["node_a"], payload["node_b"]))

    routes_using_broken: list[int] = []
    for i, route in enumerate(current_solution.routes, start=1):
        nodes = [route.depot.index] + [c.index for c in route.customers] + [route.depot.index]
        for a, b in zip(nodes, nodes[1:]):
            if _normalize_edge(a, b) in broken_edges:
                routes_using_broken.append(i)
                break

    feasible_now = current_solution.is_feasible()
    feasible_considering_broken = feasible_now and len(routes_using_broken) == 0

    # Print final summary
    print("--- Simulation summary ---")
    print(f"Planned total cost : {planned_cost:.2f}")
    print(f"Realized total cost: {realized_cost:.2f} (wasted: {total_wasted_distance:.2f})")
    print(f"Reroute operations  : {reroute_count}")
    if unserved_customers:
        print(f"Unserved customers  : {unserved_customers}")
    else:
        print("Unserved customers  : none")
    print(f"Feasible (routes)   : {feasible_now}")
    print(f"Feasible (w/ broken): {feasible_considering_broken}")
    if routes_using_broken:
        print(f"Routes using broken edges: {routes_using_broken}")

    # Also save a small summary file next to the history log
    try:
        summary_path = SIMULATION_LOG_DIR / f"{instance_name}_summary.json"
        import json
        summary = {
            "instance": instance_name,
            "planned_total_cost": planned_cost,
            "realized_total_cost": realized_cost,
            "wasted_travel_distance": total_wasted_distance,
            "reroute_count": reroute_count,
            "unserved_customers": unserved_customers,
            "feasible": feasible_now,
            "feasible_considering_broken": feasible_considering_broken,
            "routes_using_broken": routes_using_broken,
        }
        with summary_path.open("w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2)
        print(f"Saved simulation summary to {summary_path}")
    except Exception:
        pass

    return current_solution, history_log
    
    
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
    current_solution: Solution,
    algorithm: MDVRPAlgorithm,
    instance_name: str,
    reroute_count: int,
) -> tuple[int, float]:
    node_a = event.payload["node_a"]
    node_b = event.payload["node_b"]
    
    affected_route = _find_affected_route_by_broken_edge(node_a, node_b, vehicle_states)
    if affected_route is None:
        return 0, 0.0
    
    affected_vehicle_state = vehicle_states[affected_route]
    original_route = affected_vehicle_state.route

    # Resolve current node and determine whether we are on the broken edge.
    current_node = (
        original_route.depot
        if affected_vehicle_state.current_node_index == original_route.depot.index
        else affected_vehicle_state.customers_by_index.get(
            affected_vehicle_state.current_node_index, original_route.depot
        )
    )
    leg = affected_vehicle_state.current_leg()
    on_broken_edge = affected_vehicle_state.is_travelling_edge(node_a, node_b)

    # If the broken edge is in the future, keep the current next customer fixed.
    fixed_next_customer: Customer | None = None
    travel_to_next = 0.0
    if not on_broken_edge and leg is not None:
        _, to_idx = leg
        to_node = (
            original_route.depot
            if to_idx == original_route.depot.index
            else affected_vehicle_state.customers_by_index.get(to_idx, original_route.depot)
        )
        if isinstance(to_node, Customer):
            fixed_next_customer = to_node
            if affected_vehicle_state.status == "en_route":
                elapsed = max(0.0, current_time - affected_vehicle_state.last_event_time_min)
                travel_to_next = max(0.0, _travel_time(current_node, to_node) - elapsed)
            else:
                travel_to_next = _travel_time(current_node, to_node)

    # If the vehicle is on the broken edge, perform a U-turn and reroute from the leg origin.
    wasted_travel_time = 0.0
    wasted_travel_distance = 0.0
    routing_depot = original_route.depot
    event_start_node: Depot | Customer = current_node
    start_time = current_time
    if on_broken_edge and leg is not None:
        from_idx, to_idx = leg
        from_node = (
            original_route.depot
            if from_idx == original_route.depot.index
            else affected_vehicle_state.customers_by_index.get(from_idx, original_route.depot)
        )
        to_node = (
            original_route.depot
            if to_idx == original_route.depot.index
            else affected_vehicle_state.customers_by_index.get(to_idx, original_route.depot)
        )
        elapsed = max(0.0, current_time - affected_vehicle_state.last_event_time_min)
        wasted_travel_time = elapsed * 2.0
        wasted_travel_distance = wasted_travel_time * UNIT_SPEED
        start_time = current_time + elapsed
        event_start_node = from_node
        routing_depot = Depot(
            index=-(1000 + affected_route),
            x=from_node.x,
            y=from_node.y,
            max_duration=original_route.depot.max_duration,
            max_capacity=original_route.depot.max_capacity,
            max_vehicles=1,
        )

    # Build pending customers list from cached ids (deterministic order)
    pending_customers = [
        affected_vehicle_state.customers_by_index[cid]
        for cid in sorted(affected_vehicle_state.pending_customer_ids)
        if fixed_next_customer is None or cid != fixed_next_customer.index
    ]

    # Ask the algorithm to reroute locally for the depot
    broken_edge = _normalize_edge(node_a, node_b)
    algorithm._build_matrix([routing_depot], pending_customers)
    # When we used a virtual routing_depot (u-turn case) the local node ids
    # in the algorithm's matrix include the virtual depot index instead of
    # the original from-node index. In that case block the edge using the
    # virtual depot index so the local router cannot use the broken link.
    if routing_depot.index < 0:
        # determine which of node_a/node_b is the from-node for this leg
        if leg is not None:
            from_idx, to_idx = leg
        else:
            # fallback: assume node_a is the from-node
            from_idx, to_idx = node_a, node_b
        other = node_b if node_a == from_idx else node_a
        algorithm._set_edge_inf(routing_depot.index, other)
    else:
        algorithm._set_edge_inf(*broken_edge)
    if pending_customers:
        reroute_solution = algorithm.reroute_local(routing_depot, pending_customers)
    else:
        reroute_solution = Solution(routes=[Route(depot=routing_depot, customers=[])])
    print(
        f"Reroute local returned {len(reroute_solution.routes)} route(s) for depot {affected_vehicle_state.route.depot.index}"
    )
    if not reroute_solution.routes:
        print("Reroute local returned no routes; keeping original route.")
        return 0, 0.0

    if len(reroute_solution.routes) > 1:
        print("Reroute local returned multiple routes; using the first one for now.")

    # Restore real depot if we used a virtual one, and prepend fixed next customer if needed.
    new_route = reroute_solution.routes[0]
    if routing_depot.index < 0:
        new_route = Route(depot=original_route.depot, customers=new_route.customers)
    if fixed_next_customer is not None:
        new_route = Route(
            depot=new_route.depot,
            customers=[fixed_next_customer, *new_route.customers],
        )

    reroute_vehicle_payload = _build_reroute_vehicle_payload(
        vehicle_state=affected_vehicle_state,
        original_route=original_route,
        rerouted_route=new_route,
        wasted_travel_time=wasted_travel_time,
        wasted_travel_distance=wasted_travel_distance,
    )

    # Build a combined route that preserves already executed customers
    # (so visualization and saved solution reflect executed path + rerouted remainder)
    executed_count = max(0, affected_vehicle_state.next_stop_index - 1)
    executed_customers = original_route.customers[:executed_count]
    combined_customers = [*executed_customers, *new_route.customers]
    combined_route = Route(depot=original_route.depot, customers=combined_customers)

    current_solution.routes[affected_route - 1] = combined_route

    # Apply the combined route to the affected vehicle state
    affected_vehicle_state.route = combined_route
    affected_vehicle_state.customers_by_index = {c.index: c for c in combined_route.customers}
    affected_vehicle_state.pending_customer_ids = {
        c.index for c in combined_route.customers
    } - affected_vehicle_state.visited_customer_ids
    # Set next_stop_index to point to the next pending customer in the combined route
    affected_vehicle_state.next_stop_index = executed_count + 1

    reroute_index = reroute_count + 1
    time_tag = int(round(current_time * 100))
    output_path = (
        f"data/processed/results/{instance_name}_reroute_{reroute_index:03d}_"
        f"t{time_tag:06d}.json"
    )
    save_reroute_result(
        output_path=output_path,
        instance_name=instance_name,
        algorithm_name=f"{algorithm} (reroute {reroute_index})",
        solution=current_solution,
        vehicles=[reroute_vehicle_payload],
        current_time_minutes=current_time,
        broken_edge=broken_edge,
        reroute_index=reroute_index,
    )
    print(f"Saved reroute result to {output_path}")
    
    # Replace future events for the affected route with new ones.
    _remove_future_events_for_route(queue, affected_route, current_time)

    service_end_event: SimulationEvent | None = None
    # Schedule the fixed next customer first, then the rerouted remainder.
    if fixed_next_customer is not None:
        depart_time = current_time
        if affected_vehicle_state.status == "servicing" and isinstance(current_node, Customer):
            elapsed = current_time - affected_vehicle_state.last_event_time_min
            remaining_service = max(0.0, current_node.service_time - elapsed)
            service_end_time = current_time + remaining_service
            service_end_event = SimulationEvent(
                trigger_time=service_end_time,
                type="service_end",
                payload={
                    "route_id": affected_route,
                    "depot_index": new_route.depot.index,
                    "node_index": current_node.index,
                    "stop_index": 0,
                },
            )
            depart_time = service_end_time

        arrival_time = depart_time + travel_to_next
        arrival_event = SimulationEvent(
            trigger_time=arrival_time,
            type="arrival",
            payload={
                "route_id": affected_route,
                "depot_index": new_route.depot.index,
                "node_index": fixed_next_customer.index,
                "stop_index": 1,
                "service_time": fixed_next_customer.service_time,
            },
        )
        service_end_time = arrival_time + fixed_next_customer.service_time
        fixed_service_end = SimulationEvent(
            trigger_time=service_end_time,
            type="service_end",
            payload={
                "route_id": affected_route,
                "depot_index": new_route.depot.index,
                "node_index": fixed_next_customer.index,
                "stop_index": 1,
            },
        )
        remaining_route = Route(
            depot=new_route.depot,
            customers=new_route.customers[1:],
        )
        future_events = _build_future_events_for_route(
            route_id=affected_route,
            route=remaining_route,
            start_node=fixed_next_customer,
            start_time=service_end_time,
        )
        events_to_insert = [arrival_event, fixed_service_end] + future_events
        if service_end_event is None:
            _insert_events(queue, events_to_insert)
        else:
            _insert_events(queue, [service_end_event] + events_to_insert)
    else:
        # No fixed next customer; schedule from the current node or U-turn start.
        if affected_vehicle_state.status == "servicing" and isinstance(event_start_node, Customer):
            elapsed = current_time - affected_vehicle_state.last_event_time_min
            remaining_service = max(0.0, event_start_node.service_time - elapsed)
            service_end_time = current_time + remaining_service
            service_end_event = SimulationEvent(
                trigger_time=service_end_time,
                type="service_end",
                payload={
                    "route_id": affected_route,
                    "depot_index": new_route.depot.index,
                    "node_index": event_start_node.index,
                    "stop_index": 0,
                },
            )
            start_time = service_end_time

        future_events = _build_future_events_for_route(
            route_id=affected_route,
            route=new_route,
            start_node=event_start_node,
            start_time=start_time,
        )
        if service_end_event is None:
            _insert_events(queue, future_events)
        else:
            _insert_events(queue, [service_end_event] + future_events)
    
    # TODO: proximos passos:
    # Validar se custo total esta correto
    # Fazer caso de rua cair enquanto veiculo esta nela (copilot agr)
    # Testes com mais de um veiculo afetado (ex: bloqueio entre dois clientes que estão em rotas diferentes)
    # Testar caso de bloqueio acontecer enquanto veiculo esta parado no cliente (ex: bloqueio entre cliente e depot)
    # Testar caso de bloqueio acontecer enquanto veiculo esta durante viagem na rua quebrada e quando for futura tambem
    # Implementar logica se reroute sugerido for muito pior que original (ex: aumento de custo > 20%), ai rotear tudo do zero e talvez reclusterizar os clientes
    # Deixar codigo mais limpo e organizado, esse py esta ficando grande, talvez separar funcoes de events, handle disaster tambem esta grande
    return 1, wasted_travel_distance


def _build_reroute_vehicle_payload(
    vehicle_state: VehicleState,
    original_route: Route,
    rerouted_route: Route,
    wasted_travel_time: float = 0.0,
    wasted_travel_distance: float = 0.0,
) -> dict[str, Any]:
    def _path_payload(entities: List[Depot | Customer], include_first_customer: bool) -> dict[str, Any]:
        travel_distance = 0.0
        for i in range(len(entities) - 1):
            travel_distance += _dist(entities[i], entities[i + 1])

        customer_entities = entities if include_first_customer else entities[1:]
        customer_indices = [entity.index for entity in customer_entities if isinstance(entity, Customer)]
        service_time = sum(
            entity.service_time for entity in customer_entities if isinstance(entity, Customer)
        )

        return {
            "path_nodes": [entity.index for entity in entities],
            "customer_indices": customer_indices,
            "travel_distance": travel_distance,
            "service_time": service_time,
            "total_duration": travel_distance + service_time,
        }

    original_nodes: List[Depot | Customer] = [
        original_route.depot,
        *original_route.customers,
        original_route.depot,
    ]
    executed_nodes = original_nodes[: vehicle_state.next_stop_index]

    current_node = (
        original_route.depot
        if vehicle_state.current_node_index == original_route.depot.index
        else vehicle_state.customers_by_index[vehicle_state.current_node_index]
    )
    future_nodes: List[Depot | Customer] = [current_node, *rerouted_route.customers, rerouted_route.depot]
    combined_nodes: List[Depot | Customer] = executed_nodes + future_nodes[1:]

    executed_payload = _path_payload(executed_nodes, include_first_customer=True)
    future_payload = _path_payload(future_nodes, include_first_customer=False)
    full_payload = _path_payload(combined_nodes, include_first_customer=True)

    return {
        "route_id": vehicle_state.route_id,
        "depot_index": original_route.depot.index,
        "status": vehicle_state.status,
        "current_node_index": vehicle_state.current_node_index,
        "next_stop_index": vehicle_state.next_stop_index,
        "visited_customer_indices": sorted(vehicle_state.visited_customer_ids),
        "pending_customer_indices": sorted(vehicle_state.pending_customer_ids),
        "wasted_travel_time": wasted_travel_time,
        "wasted_travel_distance": wasted_travel_distance,
        "executed_path": executed_payload,
        "future_path": future_payload,
        "full_route": {
            **full_payload,
            "feasible": rerouted_route.is_feasible(),
        },
    }

def _find_affected_route_by_broken_edge(
    node_a: int, 
    node_b: int, 
    vehicle_states: dict[int, VehicleState]
) -> int | None:
    broken_edge = _normalize_edge(node_a, node_b)
    
    for state in vehicle_states.values():
        if state.has_future_broken_edge({broken_edge}):
            print(f"Route {state.route_id} is affected by broken edge {broken_edge}.")  
            return state.route_id

    print("Route affected by edge block not found among vehicle states.")
    return None