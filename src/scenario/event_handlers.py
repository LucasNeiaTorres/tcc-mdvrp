"""Event handlers for simulation events."""

from typing import List

from core.entities import Customer
from scenario.event_queue import SimulationEvent, travel_time
from scenario.state import VehicleState


def handle_arrival(
    event: SimulationEvent,
    current_time: float,
    vehicle_states: dict[int, VehicleState],
) -> None:
    """Handle vehicle arrival at a node."""
    route_id = event.payload.get("route_id")
    node_index = event.payload.get("node_index")
    stop_index = event.payload.get("stop_index")

    if route_id is None or route_id not in vehicle_states:
        return

    state = vehicle_states[route_id]
    state.current_node_index = int(node_index)
    state.next_stop_index = int(stop_index) + 1
    state.last_event_time_min = current_time

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


def handle_service_end(
    event: SimulationEvent,
    current_time: float,
    vehicle_states: dict[int, VehicleState],
) -> None:
    """Handle end of service at a node."""
    route_id = event.payload.get("route_id")

    if route_id is None or route_id not in vehicle_states:
        return

    state = vehicle_states[route_id]
    state.last_event_time_min = current_time

    if state.status == "servicing":
        state.status = "en_route"


def build_pending_customers_list(
    affected_vehicle_state: VehicleState,
    fixed_next_customer: Customer | None,
) -> List[Customer]:
    """Build pending customers in route order and apply next-node commitment."""
    ordered_pending_customers = [
        customer
        for customer in affected_vehicle_state.route.customers
        if customer.index in affected_vehicle_state.pending_customer_ids
    ]

    if fixed_next_customer is None:
        return ordered_pending_customers

    if (
        ordered_pending_customers
        and ordered_pending_customers[0].index == fixed_next_customer.index
    ):
        # Commitment rule: optimize only pending_customers[1:].
        return ordered_pending_customers[1:]

    # Safety fallback if state and planned order diverge.
    return [
        customer
        for customer in ordered_pending_customers
        if customer.index != fixed_next_customer.index
    ]


def determine_fixed_next_customer(
    affected_vehicle_state: VehicleState,
    on_broken_edge: bool,
    current_time: float,
) -> tuple[Customer | None, float]:
    """
    Determine if the vehicle should proceed to the next planned customer.
    
    Returns (fixed_next_customer, travel_time_to_next).
    - If the vehicle is on the broken edge, no customer is fixed.
    - If the broken edge is in the future, the next customer is fixed.
    """
    if on_broken_edge:
        return None, 0.0

    leg = affected_vehicle_state.current_leg()
    if leg is None:
        return None, 0.0

    _, to_idx = leg
    to_node = (
        affected_vehicle_state.route.depot
        if to_idx == affected_vehicle_state.route.depot.index
        else affected_vehicle_state.customers_by_index.get(to_idx, affected_vehicle_state.route.depot)
    )

    if not isinstance(to_node, Customer):
        return None, 0.0

    current_node = (
        affected_vehicle_state.route.depot
        if affected_vehicle_state.current_node_index == affected_vehicle_state.route.depot.index
        else affected_vehicle_state.customers_by_index.get(
            affected_vehicle_state.current_node_index, affected_vehicle_state.route.depot
        )
    )

    if affected_vehicle_state.status == "en_route":
        elapsed = max(0.0, current_time - affected_vehicle_state.last_event_time_min)
        travel_to_next = max(0.0, travel_time(current_node, to_node) - elapsed)
    else:
        travel_to_next = travel_time(current_node, to_node)

    return to_node, travel_to_next
