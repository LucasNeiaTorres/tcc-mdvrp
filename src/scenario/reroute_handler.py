"""Reroute handling and disaster response logic."""

from typing import Any, List

from core.entities import Depot, Customer, Route
from scenario.event_queue import SimulationEvent, build_future_events_for_route, EventQueue
from scenario.state import VehicleState, _normalize_edge

UNIT_SPEED = 1.0


def find_affected_route_by_broken_edge(
    node_a: int,
    node_b: int,
    vehicle_states: dict[int, VehicleState],
) -> int | None:
    """Find the route affected by a broken edge."""
    broken_edge = _normalize_edge(node_a, node_b)

    for state in vehicle_states.values():
        if state.has_future_broken_edge({broken_edge}):
            print(f"Route {state.route_id} is affected by broken edge {broken_edge}.")
            return state.route_id

    print("Route affected by edge block not found among vehicle states.")
    return None


def calculate_wasted_distance(
    affected_vehicle_state: VehicleState,
    current_node: Depot | Customer,
    on_broken_edge: bool,
    current_leg: tuple[int, int] | None,
    current_time: float,
) -> tuple[float, float, Depot | Customer, float]:
    """
    Calculate wasted distance and time for a U-turn (if on broken edge).
    
    Returns (wasted_travel_time, wasted_travel_distance, event_start_node, reroute_start_time).
    """
    if not on_broken_edge or current_leg is None:
        return 0.0, 0.0, current_node, current_time

    from_idx, to_idx = current_leg
    from_node = (
        affected_vehicle_state.route.depot
        if from_idx == affected_vehicle_state.route.depot.index
        else affected_vehicle_state.customers_by_index.get(from_idx, affected_vehicle_state.route.depot)
    )

    elapsed = max(0.0, current_time - affected_vehicle_state.last_event_time_min)
    wasted_travel_time = elapsed * 2.0
    wasted_travel_distance = wasted_travel_time * UNIT_SPEED
    reroute_start_time = current_time + elapsed

    return wasted_travel_time, wasted_travel_distance, from_node, reroute_start_time


def build_reroute_vehicle_payload(
    vehicle_state: VehicleState,
    original_route: Route,
    rerouted_route: Route,
    wasted_travel_time: float = 0.0,
    wasted_travel_distance: float = 0.0,
) -> dict[str, Any]:
    """Build the payload for a rerouted vehicle snapshot."""
    def _path_payload(
        entities: List[Depot | Customer], include_first_customer: bool
    ) -> dict[str, Any]:
        # Compute Euclidean distance between consecutive path nodes
        travel_distance = 0.0
        for i in range(len(entities) - 1):
            dx = entities[i + 1].x - entities[i].x
            dy = entities[i + 1].y - entities[i].y
            travel_distance += (dx * dx + dy * dy) ** 0.5

        # Extract customer info and aggregate service times
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

    # Separate route into already-executed and future path segments
    original_nodes: List[Depot | Customer] = [
        original_route.depot,
        *original_route.customers,
        original_route.depot,
    ]
    executed_nodes = original_nodes[: vehicle_state.next_stop_index]

    # Build future path from current position through rerouted customers
    current_node = (
        original_route.depot
        if vehicle_state.current_node_index == original_route.depot.index
        else vehicle_state.customers_by_index[vehicle_state.current_node_index]
    )
    future_nodes: List[Depot | Customer] = [current_node, *rerouted_route.customers, rerouted_route.depot]
    combined_nodes: List[Depot | Customer] = executed_nodes + future_nodes[1:]

    # Compute metrics for each path segment (executed, future, combined)
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


def schedule_rerouted_events(
    queue: EventQueue,
    affected_route: int,
    new_route: Route,
    event_start_node: Depot | Customer,
    current_node: Depot | Customer,
    start_time: float,
    affected_vehicle_state: VehicleState,
    fixed_next_customer: Customer | None,
    travel_to_next: float,
    stop_index_offset: int
) -> None:
    """Schedule future events for the rerouted vehicle."""
    service_end_event: SimulationEvent | None = None

    def _apply_stop_index_offset(events: List[SimulationEvent], offset: int) -> None:
        for event in events:
            if "stop_index" in event.payload:
                event.payload["stop_index"] += offset

    # Case 1: Next customer is fixed (broken edge in future, not current travel)
    if fixed_next_customer is not None:
        depart_time = start_time
        # If vehicle is currently servicing, schedule service_end event first
        if affected_vehicle_state.status == "servicing" and isinstance(current_node, Customer):
            elapsed = start_time - affected_vehicle_state.last_event_time_min
            remaining_service = max(0.0, current_node.service_time - elapsed)
            service_end_time = start_time + remaining_service
            service_end_event = SimulationEvent(
                trigger_time=service_end_time,
                type="service_end",
                payload={
                    "route_id": affected_route,
                    "depot_index": new_route.depot.index,
                    "node_index": current_node.index,
                    "stop_index": stop_index_offset,
                },
            )
            depart_time = service_end_time

        # Schedule arrival at fixed next customer
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
        # Schedule service_end event at fixed customer
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
        # Remaining customers in new route (excluding fixed next customer)
        remaining_route = Route(
            depot=new_route.depot,
            customers=new_route.customers[1:],
        )
        # Build events for remaining route after fixed customer
        future_events = build_future_events_for_route(
            route_id=affected_route,
            route=remaining_route,
            start_node=fixed_next_customer,
            start_time=service_end_time,
        )
        events_to_insert = [arrival_event, fixed_service_end]
        _apply_stop_index_offset(events_to_insert, stop_index_offset)
        _apply_stop_index_offset(future_events, stop_index_offset + 1)
        events_to_insert += future_events
        # Insert all events, optionally prepended with service_end if currently servicing
        if service_end_event is None:
            queue.add_events(events_to_insert)
        else:
            queue.add_events([service_end_event] + events_to_insert)
    else:
        # Case 2: No fixed next customer (on broken edge, need full reroute)
        if affected_vehicle_state.status == "servicing" and isinstance(event_start_node, Customer):
            # Calculate remaining service time at current node
            elapsed = start_time - affected_vehicle_state.last_event_time_min
            remaining_service = max(0.0, event_start_node.service_time - elapsed)
            service_end_time = start_time + remaining_service
            service_end_event = SimulationEvent(
                trigger_time=service_end_time,
                type="service_end",
                payload={
                    "route_id": affected_route,
                    "depot_index": new_route.depot.index,
                    "node_index": event_start_node.index,
                    "stop_index": stop_index_offset,
                },
            )
            start_time = service_end_time

        # Build full reroute path starting from event_start_node
        future_events = build_future_events_for_route(
            route_id=affected_route,
            route=new_route,
            start_node=event_start_node,
            start_time=start_time,
        )
        _apply_stop_index_offset(future_events, stop_index_offset)
        # Insert all events with optional service_end prepended
        if service_end_event is None:
            queue.add_events(future_events)
        else:
            queue.add_events([service_end_event] + future_events)
