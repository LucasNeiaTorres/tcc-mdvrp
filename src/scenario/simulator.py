"""Main simulation engine for dynamic vehicle routing with failures."""

from pathlib import Path
from typing import List, Tuple
import json

from core.entities import Depot, Customer, Route
from core.solution import Solution
from algorithms.base import MDVRPAlgorithm
from algorithms.ga_local_search import local_search_stage1_intra
from utils.results_io import save_history_log, save_reroute_result

from .event_queue import EventQueue, SimulationEvent, arrival_events_from_solution
from .event_handlers import handle_arrival, handle_service_end, determine_fixed_next_customer, build_pending_customers_list
from .reroute_handler import (
    find_affected_route_by_broken_edge,
    calculate_wasted_distance,
    build_reroute_vehicle_payload,
    schedule_rerouted_events,
)
from .simulation_metrics import (
    extract_blocked_edges,
    extract_route_stop_events,
    find_routes_using_broken_edges,
    extract_visited_customers,
    calculate_cost_metrics,
)
from .models import FailureEvent
from .state import VehicleState, _normalize_edge

SIMULATION_LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "simulation_logs"
UNIT_SPEED = 1.0
DEGRADATION_THRESHOLD = 1.20


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
    algorithm: MDVRPAlgorithm,
):
    """
    Run event-driven simulation with dynamic rerouting on edge failures.
    
    Parameters
    ----------
    initial_solution : Solution
        Initial routing solution.
    failures : List[FailureEvent]
        List of edge block events.
    instance_name : str
        Instance identifier for output files.
    algorithm : MDVRPAlgorithm
        Algorithm to use for rerouting.
        
    Returns
    -------
    Tuple[Solution, List[Tuple]]
        Final solution and history log.
    """
    current_solution = initial_solution
    original_solution_cost = float(initial_solution.total_cost())

    expected_customer_indices = [
        customer.index
        for route in initial_solution.routes
        for customer in route.customers
    ]

    # Initialize event queue with route events and edge failures
    event_queue = EventQueue()
    event_queue.add_events(arrival_events_from_solution(current_solution))
    for failure in failures:
        event = SimulationEvent(
            trigger_time=failure.trigger_time,
            type=failure.type,
            payload={
                "node_a": failure.node_a,
                "node_b": failure.node_b,
            },
        )
        event_queue.add_event(event)

    vehicle_states = _build_vehicle_states(current_solution)
    reroute_count = 0
    history_log = []
    total_wasted_distance = 0.0
    current_time = 0.0
    blocked_edges: set[tuple[int, int]] = set()

    # Main simulation loop: process events in chronological order
    while not event_queue.is_empty():
        event = event_queue.pop_next()
        if event is None:
            break

        print(f"Processing event: time={event.trigger_time:.2f}, type={event.type}, payload={event.payload}")
        current_time = event.trigger_time
        history_log.append((current_time, event.type, event.payload))

        # Dispatch event to appropriate handler
        if event.type == "arrival":
            handle_arrival(event, current_time, vehicle_states)

        elif event.type == "service_end":
            handle_service_end(event, current_time, vehicle_states)

        elif event.type == "edge_block":
            blocked_edges.add(_normalize_edge(event.payload["node_a"], event.payload["node_b"]))
            reroute_inc, wasted = _handle_disaster(
                event,
                current_time,
                event_queue,
                vehicle_states,
                current_solution,
                algorithm,
                instance_name,
                reroute_count,
                blocked_edges,
            )
            reroute_count += reroute_inc
            total_wasted_distance += wasted

    # Save temporal history log for validation (blocked-edge checks)
    output_path = SIMULATION_LOG_DIR / f"{instance_name}_log.json"
    save_history_log(
        str(output_path),
        instance_name,
        history_log,
        expected_customer_indices=expected_customer_indices,
    )
    print(f"Saved simulation log to {output_path}")

    # Compute cost metrics: original to post-reroute to realized
    post_reroute_cost = float(current_solution.total_cost())
    reroute_cost_increase, realized_cost, total_cost_impact, _ = calculate_cost_metrics(
        original_solution_cost, post_reroute_cost, float(total_wasted_distance)
    )

    # Extract feasibility metrics from vehicle states and history
    visited = extract_visited_customers(vehicle_states)
    expected_set = set(expected_customer_indices)
    unserved_customers = sorted(list(expected_set - visited))

    # Check for temporal violations: routes using blocked edges after block
    blocked_edges = extract_blocked_edges(history_log)
    route_stop_events = extract_route_stop_events(history_log)
    routes_using_broken_set = find_routes_using_broken_edges(route_stop_events, blocked_edges)
    routes_using_broken = sorted(routes_using_broken_set)

    routes_feasible_now = current_solution.is_feasible()
    fleet_feasible_now = current_solution.fleet_is_feasible()
    fully_feasible_now = current_solution.fully_feasible()
    feasible_considering_broken = routes_feasible_now and len(routes_using_broken) == 0

    # Output final simulation metrics
    print("--- Simulation summary ---")
    print(f"Original solution cost  : {original_solution_cost:.2f}")
    print(f"Post-reroute cost       : {post_reroute_cost:.2f} (change: {reroute_cost_increase:+.2f})")
    print(f"Wasted (U-turns)        : {total_wasted_distance:.2f}")
    print(f"Realized total cost     : {realized_cost:.2f} (total impact: {total_cost_impact:+.2f})")
    print(f"Reroute operations      : {reroute_count}")
    if unserved_customers:
        print(f"Unserved customers      : {unserved_customers}")
    else:
        print("Unserved customers      : none")
    print(f"Feasible (routes)       : {routes_feasible_now}")
    print(f"Feasible (fleet)        : {fleet_feasible_now}")
    print(f"Feasible (full)         : {fully_feasible_now}")
    print(f"Feasible (w/ broken)    : {feasible_considering_broken}")
    if routes_using_broken:
        print(f"Routes using broken edges: {routes_using_broken}")

    # Persist aggregated summary to JSON for analysis
    try:
        summary_path = SIMULATION_LOG_DIR / f"{instance_name}_summary.json"
        summary = {
            "instance": instance_name,
            "original_solution_cost": original_solution_cost,
            "post_reroute_cost": post_reroute_cost,
            "reroute_cost_increase": reroute_cost_increase,
            "wasted_travel_distance": total_wasted_distance,
            "realized_total_cost": realized_cost,
            "total_cost_impact": total_cost_impact,
            "reroute_count": reroute_count,
            "unserved_customers": unserved_customers,
            "feasible": routes_feasible_now,
            "fleet_feasible": fleet_feasible_now,
            "fully_feasible": fully_feasible_now,
            "feasible_considering_broken": feasible_considering_broken,
            "routes_using_broken": routes_using_broken,
        }
        with summary_path.open("w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2)
        print(f"Saved simulation summary to {summary_path}")
    except Exception:
        pass

    return current_solution, history_log


def _handle_disaster(
    event: SimulationEvent,
    current_time: float,
    event_queue: EventQueue,
    vehicle_states: dict[int, VehicleState],
    current_solution: Solution,
    algorithm: MDVRPAlgorithm,
    instance_name: str,
    reroute_count: int,
    blocked_edges: set[tuple[int, int]],
) -> Tuple[int, float]:
    """
    Handle edge block event by finding affected vehicle and rerouting.

    Returns
    -------
    Tuple[int, float]
        (number of reroutes performed, wasted distance from U-turn).
    """
    node_a = event.payload["node_a"]
    node_b = event.payload["node_b"]

    affected_route = find_affected_route_by_broken_edge(node_a, node_b, vehicle_states)
    if affected_route is None:
        return 0, 0.0

    affected_vehicle_state = vehicle_states[affected_route]
    original_route = affected_vehicle_state.route
    original_route_duration = original_route.total_duration()

    # Resolve vehicle position and check if traversing the broken edge now.
    current_node = (
        original_route.depot
        if affected_vehicle_state.current_node_index == original_route.depot.index
        else affected_vehicle_state.customers_by_index.get(
            affected_vehicle_state.current_node_index, original_route.depot
        )
    )
    leg = affected_vehicle_state.current_leg()
    on_broken_edge = affected_vehicle_state.is_travelling_edge(node_a, node_b)

    # Next-node commitment: keep immediate destination fixed unless this is the current broken leg.
    fixed_next_customer, travel_to_next = determine_fixed_next_customer(
        affected_vehicle_state, on_broken_edge, current_time
    )

    # U-turn exception: when currently on the broken edge, commitment is broken and wasted travel is accounted.
    wasted_travel_time, wasted_travel_distance, event_start_node, reroute_start_time = calculate_wasted_distance(
        affected_vehicle_state, current_node, on_broken_edge, leg, current_time
    )

    # Pending pool for optimization (ordered and commitment-aware).
    pending_customers = build_pending_customers_list(affected_vehicle_state, fixed_next_customer)

    broken_edge = _normalize_edge(node_a, node_b)
    reroute_start_node: Depot | Customer = (
        fixed_next_customer if fixed_next_customer is not None else event_start_node
    )

    # Build distance matrix for the local patch and fallback reroute.
    nodes_for_matrix: list[Depot | Customer] = [original_route.depot]
    if isinstance(reroute_start_node, Customer):
        nodes_for_matrix.append(reroute_start_node)
    if fixed_next_customer is not None and fixed_next_customer.index not in {
        node.index for node in nodes_for_matrix
    }:
        nodes_for_matrix.append(fixed_next_customer)

    algorithm._build_matrix(nodes_for_matrix, pending_customers)
    for blocked_edge in blocked_edges:
        algorithm._set_edge_inf(*blocked_edge)

    historical_wasted_duration = original_route.wasted_duration + wasted_travel_time
    historical_wasted_distance = original_route.wasted_distance + wasted_travel_distance

    executed_count = max(0, affected_vehicle_state.next_stop_index - 1)
    executed_customers = original_route.customers[:executed_count]

    # Stage 1: local containment (intra-route M1/M2/M3 only).
    stage1_pending_customers = local_search_stage1_intra(
        customers=pending_customers,
        start_node=reroute_start_node,
        end_node=original_route.depot,
        dist_fn=algorithm._dist,
    )
    stage1_customers = [
        *([fixed_next_customer] if fixed_next_customer is not None else []),
        *stage1_pending_customers,
    ]

    stage1_combined_route = Route(
        depot=original_route.depot,
        customers=[*executed_customers, *stage1_customers],
        wasted_duration=historical_wasted_duration,
        wasted_distance=historical_wasted_distance,
    )
    stage1_duration_limit = original_route_duration * DEGRADATION_THRESHOLD

    if (
        stage1_combined_route.is_feasible()
        and stage1_combined_route.total_duration() <= stage1_duration_limit
    ):
        print(
            "Stage 1 accepted "
            f"(duration={stage1_combined_route.total_duration():.2f}, "
            f"limit={stage1_duration_limit:.2f})."
        )
        new_route = Route(
            depot=original_route.depot,
            customers=stage1_customers,
            wasted_duration=historical_wasted_duration,
            wasted_distance=historical_wasted_distance,
        )
    else:
        print(
            "Stage 1 rejected "
            f"(duration={stage1_combined_route.total_duration():.2f}, "
            f"limit={stage1_duration_limit:.2f}, "
            f"feasible={stage1_combined_route.is_feasible()})."
        )
        print("Reverting local patch and proceeding to Stage 2 reroute.")

        if pending_customers:
            print(
                f"pending_customers for reroute: {[c.index for c in pending_customers]}, "
                f"fixed_next_customer: {fixed_next_customer.index if fixed_next_customer else None}, "
                f"start_node: {event_start_node.index}, "
                f"real_end_depot: {original_route.depot.index}, "
                f"broken_edge: {broken_edge}"
            )
            reroute_solution = algorithm.reroute_local(
                current_start_node=reroute_start_node,
                pending_customers=pending_customers,
                real_end_depot=original_route.depot,
            )
        else:
            reroute_solution = Solution(routes=[Route(depot=original_route.depot, customers=[])])

        print(
            f"Reroute local returned {len(reroute_solution.routes)} route(s) "
            f"for depot {affected_vehicle_state.route.depot.index}"
        )
        if not reroute_solution.routes:
            print("Reroute local returned no routes; keeping original route.")
            return 0, 0.0

        if len(reroute_solution.routes) > 1:
            print("Reroute local returned multiple routes; using the first one for now.")

        stage2_customers = list(reroute_solution.routes[0].customers)
        if fixed_next_customer is not None:
            stage2_customers = [fixed_next_customer, *stage2_customers]

        new_route = Route(
            depot=original_route.depot,
            customers=stage2_customers,
            wasted_duration=historical_wasted_duration,
            wasted_distance=historical_wasted_distance,
        )

    # Build reroute snapshot for output JSON (executed path + future path).
    reroute_vehicle_payload = build_reroute_vehicle_payload(
        vehicle_state=affected_vehicle_state,
        original_route=original_route,
        rerouted_route=new_route,
        wasted_travel_time=wasted_travel_time,
        wasted_travel_distance=wasted_travel_distance,
    )

    # Merge executed path (completed stops) with rerouted path (future).
    combined_customers = [*executed_customers, *new_route.customers]
    combined_route = Route(
        depot=original_route.depot,
        customers=combined_customers,
        wasted_duration=new_route.wasted_duration,
        wasted_distance=new_route.wasted_distance,
    )

    # Update solution with combined route.
    current_solution.routes[affected_route - 1] = combined_route

    # Sync vehicle state with new combined route (for future event processing).
    affected_vehicle_state.route = combined_route
    affected_vehicle_state.customers_by_index = {c.index: c for c in combined_route.customers}
    affected_vehicle_state.pending_customer_ids = (
        {c.index for c in combined_route.customers} - affected_vehicle_state.visited_customer_ids
    )
    affected_vehicle_state.next_stop_index = executed_count + 1

    # Save reroute result.
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

    # Remove old future events for this route and inject new ones based on reroute.
    event_queue.remove_future_events_for_route(affected_route, current_time)
    schedule_rerouted_events(
        event_queue,
        affected_route,
        new_route,
        event_start_node,
        current_node,
        reroute_start_time,
        affected_vehicle_state,
        fixed_next_customer,
        travel_to_next,
        stop_index_offset=executed_count,
    )

    return 1, wasted_travel_distance

