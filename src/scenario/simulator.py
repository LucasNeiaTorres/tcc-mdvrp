"""Main simulation engine for dynamic vehicle routing with failures."""

from copy import deepcopy

from pathlib import Path
from typing import Callable, List, Tuple
import json

from core.entities import Depot, Customer, Route
from core.solution import Solution
from algorithms.base import MDVRPAlgorithm
from algorithms.ga_local_search import local_search, local_search_stage1_intra
from utils.results_io import save_history_log, save_reroute_result

from .event_queue import EventQueue, SimulationEvent, arrival_events_from_solution, travel_time
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
from .stage3_global_repair import stage3_global_cross_depot_repair
from .state import VehicleState, _normalize_edge

SIMULATION_LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "simulation_logs"
UNIT_SPEED = 1.0


def _path_uses_blocked_edge(
    start_node: Depot | Customer,
    customers: list[Customer],
    end_node: Depot | Customer,
    blocked_edges: set[tuple[int, int]],
) -> bool:
    if not blocked_edges:
        return False

    prev_idx = start_node.index
    for customer in customers:
        if _normalize_edge(prev_idx, customer.index) in blocked_edges:
            return True
        prev_idx = customer.index

    return _normalize_edge(prev_idx, end_node.index) in blocked_edges


def _resolve_current_node(state: VehicleState) -> Depot | Customer:
    return (
        state.route.depot
        if state.current_node_index == state.route.depot.index
        else state.customers_by_index.get(state.current_node_index, state.route.depot)
    )


def _clone_route(route: Route) -> Route:
    """Return a shallow route clone preserving historical waste fields."""
    return Route(
        depot=route.depot,
        customers=list(route.customers),
        wasted_duration=route.wasted_duration,
        wasted_distance=route.wasted_distance,
    )


def _resolve_stage3_target_node(
    state: VehicleState,
    broken_edge: tuple[int, int],
) -> int | None:
    """Resolve target_node as the to_idx on the blocked leg in route orientation."""
    planned_nodes = [
        state.route.depot.index,
        *[customer.index for customer in state.route.customers],
        state.route.depot.index,
    ]
    leg_start = max(0, state.next_stop_index - 1)
    for i in range(leg_start, len(planned_nodes) - 1):
        if _normalize_edge(planned_nodes[i], planned_nodes[i + 1]) == broken_edge:
            return planned_nodes[i + 1]
    return None


def _build_stage3_distance_matrix(
    vehicle_states: dict[int, VehicleState],
    blocked_edges: set[tuple[int, int]],
) -> list[list[float]]:
    """Build node-id-indexed matrix and apply blocked edges as infinite cost."""
    active_nodes: dict[int, Depot | Customer] = {}
    for state in vehicle_states.values():
        active_nodes[state.route.depot.index] = state.route.depot
        for customer in state.route.customers:
            active_nodes[customer.index] = customer

    if not active_nodes:
        return [[0.0]]

    size = max(active_nodes) + 1
    matrix = [[float("inf")] * size for _ in range(size)]

    for node_id in range(size):
        matrix[node_id][node_id] = 0.0

    for i, node_i in active_nodes.items():
        matrix[i][i] = 0.0
        for j, node_j in active_nodes.items():
            matrix[i][j] = travel_time(node_i, node_j)

    for node_a, node_b in blocked_edges:
        if node_a < size and node_b < size:
            matrix[node_a][node_b] = float("inf")
            matrix[node_b][node_a] = float("inf")

    return matrix


def reoptimize_intra_route_stage1(
    pending_customers: list[Customer],
    fixed_next_customer: Customer | None,
    reroute_start_node: Depot | Customer,
    event_start_node: Depot | Customer,
    depot: Depot,
    dist_fn: Callable[[Depot | Customer, Depot | Customer], float],
    executed_customers: list[Customer],
    historical_wasted_duration: float,
    historical_wasted_distance: float,
    original_route_cost: float,
    reroute_degradation_threshold: float,
    blocked_edges: set[tuple[int, int]],
) -> tuple[list[Customer], Route, bool]:
    stage1_pending_customers = local_search_stage1_intra(
        customers=pending_customers,
        start_node=reroute_start_node,
        end_node=depot,
        dist_fn=dist_fn,
    )
    stage1_customers = [
        *([fixed_next_customer] if fixed_next_customer is not None else []),
        *stage1_pending_customers,
    ]

    stage1_combined_route = Route(
        depot=depot,
        customers=[*executed_customers, *stage1_customers],
        wasted_duration=historical_wasted_duration,
        wasted_distance=historical_wasted_distance,
    )
    stage1_cost = stage1_combined_route.total_distance()
    stage1_cost_limit = original_route_cost * reroute_degradation_threshold
    stage1_uses_blocked_edge = _path_uses_blocked_edge(
        event_start_node,
        stage1_customers,
        depot,
        blocked_edges,
    )
    print(
        f"Stage 1 result route: customers={[c.index for c in stage1_customers]}, "
        f"cost={stage1_cost:.2f}, "
        f"cost_limit={stage1_cost_limit:.2f}, "
        f"duration={stage1_combined_route.total_duration():.2f}, "
        f"fixed_next_customer={fixed_next_customer.index if fixed_next_customer else None}, "
        f"uses_broken_edge={stage1_uses_blocked_edge}"
    )

    accepted = (
        stage1_combined_route.is_feasible()
        and stage1_cost <= stage1_cost_limit
        and not stage1_uses_blocked_edge
    )
    if accepted:
        print(
            "Stage 1 accepted "
            f"(cost={stage1_cost:.2f}, "
            f"cost_limit={stage1_cost_limit:.2f}, "
            f"duration={stage1_combined_route.total_duration():.2f}, "
            f"hard_duration_limit={depot.max_duration:.2f})."
        )
    else:
        print(
            "Stage 1 rejected "
            f"(cost={stage1_cost:.2f}, "
            f"cost_limit={stage1_cost_limit:.2f}, "
            f"duration={stage1_combined_route.total_duration():.2f}, "
            f"hard_duration_limit={depot.max_duration:.2f}, "
            f"feasible={stage1_combined_route.is_feasible()}, "
            f"uses_broken_edge={stage1_uses_blocked_edge})."
        )

    return stage1_customers, stage1_combined_route, accepted


def reoptimize_intra_cluster(
    vehicle_states: dict[int, VehicleState],
    algorithm: MDVRPAlgorithm,
    affected_route_id: int,
    current_time: float,
    blocked_edges: set[tuple[int, int]],
    cluster_degradation_threshold: float,
    event_start_node: Depot | Customer,
    reroute_start_time: float,
    wasted_travel_time: float,
    wasted_travel_distance: float,
    fixed_next_customer: Customer | None,
    travel_to_next: float,
    leg: tuple[int, int] | None,
    on_broken_edge: bool,
) -> tuple[dict[int, dict[str, object]] | None, dict[int, dict[str, object]] | None]:
    affected_state = vehicle_states[affected_route_id]
    depot = affected_state.route.depot

    # Scope to vehicles that belong to the same depot (cluster).
    cluster_states = [
        state
        for state in vehicle_states.values()
        if state.route.depot.index == depot.index
    ]
    if not cluster_states:
        return None, None

    def _route_cost_with_return(
        route: Route,
        current_node_local: Depot | Customer,
        wasted_distance_override: float | None = None,
    ) -> float:
        if route.customers:
            if wasted_distance_override is None:
                return route.total_distance()
            temp_route = Route(
                depot=route.depot,
                customers=list(route.customers),
                wasted_duration=route.wasted_duration,
                wasted_distance=wasted_distance_override,
            )
            return temp_route.total_distance()

        base_wasted = (
            wasted_distance_override
            if wasted_distance_override is not None
            else route.wasted_distance
        )
        if current_node_local.index == route.depot.index:
            return base_wasted
        return base_wasted + travel_time(current_node_local, route.depot)

    # Baseline cost for the whole cluster (used by the gatekeeper).
    original_cluster_cost = 0.0
    for state in cluster_states:
        current_node_local = _resolve_current_node(state)
        if state.route_id == affected_route_id:
            original_cluster_cost += _route_cost_with_return(
                state.route,
                current_node_local,
                wasted_distance_override=state.route.wasted_distance + wasted_travel_distance,
            )
        else:
            original_cluster_cost += _route_cost_with_return(
                state.route,
                current_node_local,
            )

    # Prepare per-route pending sets and a dummy route to drain orphan customers.
    unassigned_customers: list[Customer] = []
    cluster_routes: list[list[Customer]] = []
    route_items: list[dict[str, object]] = []
    route_items_by_id: dict[int, dict[str, object]] = {}
    empty_route_items: list[dict[str, object]] = []

    for state in cluster_states:
        current_node_local = _resolve_current_node(state)
        executed_count = max(0, state.next_stop_index - 1)
        executed_customers = state.route.customers[:executed_count]

        if state.route_id == affected_route_id:
            fixed_next_local = fixed_next_customer
            travel_to_next_local = travel_to_next
            event_start_node_local = event_start_node
            reroute_start_time_local = reroute_start_time
            historical_wasted_duration = state.route.wasted_duration + wasted_travel_time
            historical_wasted_distance = state.route.wasted_distance + wasted_travel_distance
            on_broken_edge_local = on_broken_edge
            leg_local = leg
        else:
            fixed_next_local, travel_to_next_local = determine_fixed_next_customer(
                state, False, current_time
            )
            event_start_node_local = current_node_local
            reroute_start_time_local = current_time
            historical_wasted_duration = state.route.wasted_duration
            historical_wasted_distance = state.route.wasted_distance
            on_broken_edge_local = False
            leg_local = state.current_leg()

        pending_customers = build_pending_customers_list(state, fixed_next_local)

        if (
            state.route_id == affected_route_id
            and on_broken_edge_local
            and leg_local is not None
        ):
            _, to_idx = leg_local
            pending_customers = [c for c in pending_customers if c.index != to_idx]
            removed = state.customers_by_index.get(to_idx)
            if removed is not None:
                unassigned_customers.append(removed)

        route_customers: list[Customer] = []
        if fixed_next_local is not None:
            route_customers.append(fixed_next_local)
        route_customers.extend(pending_customers)

        if route_customers:
            route_item = {
                "route_id": state.route_id,
                "state": state,
                "current_node": current_node_local,
                "event_start_node": event_start_node_local,
                "reroute_start_time": reroute_start_time_local,
                "fixed_next_customer": fixed_next_local,
                "travel_to_next": travel_to_next_local,
                "executed_count": executed_count,
                "executed_customers": executed_customers,
                "wasted_duration": historical_wasted_duration,
                "wasted_distance": historical_wasted_distance,
                "route_customers": route_customers,
            }
            route_items.append(route_item)
            route_items_by_id[state.route_id] = route_item
            cluster_routes.append(route_customers)
        elif state.route_id == affected_route_id:
            route_item = {
                "route_id": state.route_id,
                "state": state,
                "current_node": current_node_local,
                "event_start_node": event_start_node_local,
                "reroute_start_time": reroute_start_time_local,
                "fixed_next_customer": fixed_next_local,
                "travel_to_next": travel_to_next_local,
                "executed_count": executed_count,
                "executed_customers": executed_customers,
                "wasted_duration": historical_wasted_duration,
                "wasted_distance": historical_wasted_distance,
                "route_customers": route_customers,
            }
            empty_route_items.append(route_item)
            route_items_by_id[state.route_id] = route_item

    if not route_items and not empty_route_items:
        return None, None

    if not route_items and unassigned_customers:
        print("Stage 2 rejected: no routes available to absorb unassigned customers.")
        return None, None

    # Insert unassigned customers into the first route if possible (VND will handle them).
    if unassigned_customers and cluster_routes:
        cluster_routes[0].extend(unassigned_customers)

    unique_customers: dict[int, Customer] = {}
    for route in cluster_routes:
        for customer in route:
            unique_customers[customer.index] = customer

    optimized_routes: list[list[Customer]] = []
    if route_items:
        # Build local matrix with blocked edges and run VND local search.
        algorithm._build_matrix([depot], list(unique_customers.values()))
        for edge in blocked_edges:
            algorithm._set_edge_inf(*edge)

        optimized_routes = local_search(
            deepcopy(cluster_routes),
            depot,
            algorithm._dist,
            is_stage_2=True,
            apply_frozen_prefix=True,
        )

    prefix_to_item = {
        item["route_customers"][0].index: item
        for item in route_items
    }
    assigned_routes: dict[int, list[Customer]] = {}
    leftover_routes: list[list[Customer]] = []

    for route in optimized_routes:
        if not route:
            continue
        first_idx = route[0].index
        if first_idx in prefix_to_item and first_idx not in assigned_routes:
            assigned_routes[first_idx] = route
        else:
            leftover_routes.append(route)

    if leftover_routes:
        leftover_ids = sorted({c.index for route in leftover_routes for c in route})
        print(
            "Stage 2 rejected: unmatched optimized routes or prefix mismatch. "
            f"leftover_customers={leftover_ids}"
        )
        return None, None

    for item in route_items:
        prefix_idx = item["route_customers"][0].index
        if prefix_idx not in assigned_routes:
            print("Stage 2 rejected: missing route for frozen prefix.")
            return None, None

    new_routes_by_id: dict[int, dict[str, object]] = {}
    new_cluster_cost = 0.0

    for state in cluster_states:
        item = route_items_by_id.get(state.route_id)
        if item is None:
            new_cluster_cost += _route_cost_with_return(
                state.route,
                _resolve_current_node(state),
            )
            continue

        if item["route_customers"]:
            prefix_idx = item["route_customers"][0].index
            optimized_pending = assigned_routes[prefix_idx]
        else:
            optimized_pending = []
        combined_customers = [*item["executed_customers"], *optimized_pending]
        combined_route = Route(
            depot=state.route.depot,
            customers=combined_customers,
            wasted_duration=item["wasted_duration"],
            wasted_distance=item["wasted_distance"],
        )
        if combined_route.customers:
            new_cluster_cost += combined_route.total_distance()
        else:
            new_cluster_cost += _route_cost_with_return(
                combined_route,
                item["current_node"],
                wasted_distance_override=item["wasted_distance"],
            )
        new_routes_by_id[state.route_id] = {
            "combined_route": combined_route,
            "future_route": Route(
                depot=state.route.depot,
                customers=optimized_pending,
                wasted_duration=item["wasted_duration"],
                wasted_distance=item["wasted_distance"],
            ),
            "item": item,
        }

    # Gatekeeper: feasibility, broken edge avoidance, and cluster cost threshold.
    all_routes_feasible = True
    for route_id, data in new_routes_by_id.items():
        if not data["combined_route"].is_feasible():
            all_routes_feasible = False
            break
        
        item = data["item"]
        uses_broken = _path_uses_blocked_edge(
            item["event_start_node"],
            data["future_route"].customers,
            data["combined_route"].depot,
            blocked_edges,
        )
        if uses_broken:
            print(f"Stage 2 rejected: route {route_id} attempts to cross a blocked edge.")
            all_routes_feasible = False
            break

    is_within_threshold = (
        new_cluster_cost <= original_cluster_cost * cluster_degradation_threshold
    )

    affected_data = new_routes_by_id.get(affected_route_id)
    if affected_data is None:
        print("Stage 2 rejected: affected route missing in optimized cluster.")
        return None, None

    if not all_routes_feasible:
        print(f"Stage 2 rejected (infeasible).")
        return None, None

    if not is_within_threshold:
        print(
            "Stage 2 rejected by threshold but saved as FALLBACK "
            f"(cost={new_cluster_cost:.2f}, "
            f"limit={original_cluster_cost * cluster_degradation_threshold:.2f})."
        )
        return None, new_routes_by_id

    delta_pct = 0.0
    if original_cluster_cost > 0:
        delta_pct = ((new_cluster_cost - original_cluster_cost) / original_cluster_cost) * 100.0
    print(
        "Stage 2 cluster cost change "
        f"(old={original_cluster_cost:.2f}, "
        f"new={new_cluster_cost:.2f}, "
        f"delta={delta_pct:+.2f}%)."
    )

    return new_routes_by_id, None


def _commit_stage2_updates(
    *,
    new_routes_by_id: dict[int, dict[str, object]],
    affected_route_id: int,
    current_solution: Solution,
    event_queue: EventQueue,
    current_time: float,
    instance_name: str,
    algorithm: MDVRPAlgorithm,
    reroute_index: int,
    broken_edge: tuple[int, int],
    wasted_travel_time: float,
    wasted_travel_distance: float,
) -> None:
    affected_data = new_routes_by_id.get(affected_route_id)
    if affected_data is None:
        print("Stage 2 commit aborted: affected route missing in optimized cluster.")
        return

    affected_state = affected_data["item"]["state"]
    original_route = affected_state.route

    reroute_vehicle_payload = build_reroute_vehicle_payload(
        vehicle_state=affected_state,
        original_route=original_route,
        rerouted_route=affected_data["future_route"],
        wasted_travel_time=wasted_travel_time,
        wasted_travel_distance=wasted_travel_distance,
    )

    for route_id, data in new_routes_by_id.items():
        current_solution.routes[route_id - 1] = data["combined_route"]
        state = data["item"]["state"]
        state.route = data["combined_route"]
        state.customers_by_index = {c.index: c for c in data["combined_route"].customers}
        state.pending_customer_ids = (
            {c.index for c in data["combined_route"].customers}
            - state.visited_customer_ids
        )
        state.next_stop_index = data["item"]["executed_count"] + 1

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

    for route_id, data in new_routes_by_id.items():
        item = data["item"]
        event_queue.remove_future_events_for_route(route_id, current_time)
        schedule_rerouted_events(
            event_queue,
            route_id,
            data["future_route"],
            item["event_start_node"],
            item["current_node"],
            item["reroute_start_time"],
            item["state"],
            item["fixed_next_customer"],
            item["travel_to_next"],
            stop_index_offset=item["executed_count"],
        )


def _commit_stage3_updates(
    *,
    winner_state: VehicleState,
    target_node: int,
    donor_repaired_customers: list[Customer],
    vehicle_states: dict[int, VehicleState],
    current_solution: Solution,
    event_queue: EventQueue,
    current_time: float,
    instance_name: str,
    algorithm: MDVRPAlgorithm,
    reroute_index: int,
    broken_edge: tuple[int, int],
    original_route: Route,
    routes_before_stage3: dict[int, Route],
    baseline_route_duration: float,
    event_start_node: Depot | Customer,
    current_node: Depot | Customer,
    reroute_start_time: float,
    fixed_next_customer: Customer | None,
    travel_to_next: float,
    wasted_travel_time: float,
    wasted_travel_distance: float,
    affected_route_id: int,
) -> None:
    blocked_state = vehicle_states[affected_route_id]
    affected_executed_count = max(0, blocked_state.next_stop_index - 1)
    
    blocked_updated_customers = [
        *blocked_state.route.customers[:affected_executed_count],
        *donor_repaired_customers
    ]
    
    blocked_state.route = Route(
        depot=blocked_state.route.depot,
        customers=blocked_updated_customers,
        wasted_duration=blocked_state.route.wasted_duration,
        wasted_distance=blocked_state.route.wasted_distance,
    )
    
    blocked_state.customers_by_index = {
        customer.index: customer for customer in blocked_state.route.customers
    }
    blocked_state.pending_customer_ids = (
        {customer.index for customer in blocked_state.route.customers}
        - blocked_state.visited_customer_ids
    )

    winner_live_state = vehicle_states[winner_state.route_id]
    winner_live_state.route = winner_state.route
    winner_live_state.customers_by_index = {
        customer.index: customer for customer in winner_state.route.customers
    }
    winner_live_state.pending_customer_ids = (
        {customer.index for customer in winner_state.route.customers}
        - winner_live_state.visited_customer_ids
    )

    current_solution.routes[affected_route_id - 1] = blocked_state.route
    if winner_live_state.route_id != affected_route_id:
        current_solution.routes[winner_live_state.route_id - 1] = winner_live_state.route

    affected_executed_count = max(0, blocked_state.next_stop_index - 1)
    affected_future_route = Route(
        depot=blocked_state.route.depot,
        customers=list(blocked_state.route.customers[affected_executed_count:]),
        wasted_duration=blocked_state.route.wasted_duration,
        wasted_distance=blocked_state.route.wasted_distance,
    )

    blocked_old = baseline_route_duration
    blocked_new = blocked_state.route.total_duration()
    blocked_delta_pct = 0.0
    if blocked_old > 0:
        blocked_delta_pct = ((blocked_new - blocked_old) / blocked_old) * 100.0
    print(
        "Stage 3 blocked-route change "
        f"(old={blocked_old:.2f}, "
        f"new={blocked_new:.2f}, "
        f"delta={blocked_delta_pct:+.2f}%)."
    )

    reroute_vehicles_payload = [
        build_reroute_vehicle_payload(
            vehicle_state=blocked_state,
            original_route=original_route,
            rerouted_route=affected_future_route,
            wasted_travel_time=wasted_travel_time,
            wasted_travel_distance=wasted_travel_distance,
        )
    ]

    winner_future_route: Route | None = None
    winner_old = 0.0
    winner_new = 0.0
    if winner_live_state.route_id != affected_route_id:
        winner_executed_count = max(0, winner_live_state.next_stop_index - 1)
        winner_future_route = Route(
            depot=winner_live_state.route.depot,
            customers=list(winner_live_state.route.customers[winner_executed_count:]),
            wasted_duration=winner_live_state.route.wasted_duration,
            wasted_distance=winner_live_state.route.wasted_distance,
        )
        winner_original_route = routes_before_stage3[winner_live_state.route_id]
        winner_old = winner_original_route.total_duration()
        winner_new = winner_live_state.route.total_duration()
        winner_delta_pct = 0.0
        if winner_old > 0:
            winner_delta_pct = ((winner_new - winner_old) / winner_old) * 100.0
        print(
            "Stage 3 winner-route change "
            f"(old={winner_old:.2f}, "
            f"new={winner_new:.2f}, "
            f"delta={winner_delta_pct:+.2f}%)."
        )
        reroute_vehicles_payload.append(
            build_reroute_vehicle_payload(
                vehicle_state=winner_live_state,
                original_route=winner_original_route,
                rerouted_route=winner_future_route,
                wasted_travel_time=0.0,
                wasted_travel_distance=0.0,
            )
        )
    else:
        print("Stage 3 winner-route change skipped (winner is blocked vehicle).")

    net_old = blocked_old + winner_old
    net_new = blocked_new + winner_new
    net_delta_pct = 0.0
    if net_old > 0:
        net_delta_pct = ((net_new - net_old) / net_old) * 100.0
    print(
        "Stage 3 net change (blocked+winner only) "
        f"(old={net_old:.2f}, "
        f"new={net_new:.2f}, "
        f"delta={net_delta_pct:+.2f}%)."
    )

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
        vehicles=reroute_vehicles_payload,
        current_time_minutes=current_time,
        broken_edge=broken_edge,
        reroute_index=reroute_index,
    )
    print(f"Saved reroute result to {output_path}")

    event_queue.remove_future_events_for_route(affected_route_id, current_time)
    schedule_rerouted_events(
        event_queue,
        affected_route_id,
        affected_future_route,
        event_start_node,
        current_node,
        reroute_start_time,
        blocked_state,
        fixed_next_customer,
        travel_to_next,
        stop_index_offset=affected_executed_count,
    )

    if winner_live_state.route_id != affected_route_id and winner_future_route is not None:
        winner_current_node = _resolve_current_node(winner_live_state)
        winner_fixed_next, winner_travel_to_next = determine_fixed_next_customer(
            winner_live_state,
            on_broken_edge=False,
            current_time=current_time,
        )

        event_queue.remove_future_events_for_route(winner_live_state.route_id, current_time)
        schedule_rerouted_events(
            event_queue,
            winner_live_state.route_id,
            winner_future_route,
            winner_current_node,
            winner_current_node,
            current_time,
            winner_live_state,
            winner_fixed_next,
            winner_travel_to_next,
            stop_index_offset=max(0, winner_live_state.next_stop_index - 1),
        )


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
    reroute_degradation_threshold: float = 1.20,
    cluster_degradation_threshold: float = 1.05,
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
    cluster_degradation_threshold : float
        Stage-2 cluster acceptance limit for total cost.
        
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
    reroute_by_stage = {"stage1": 0, "stage2": 0, "stage3": 0}
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
            reroute_inc, wasted, accepted_stage = _handle_disaster(
                event,
                current_time,
                event_queue,
                vehicle_states,
                current_solution,
                algorithm,
                instance_name,
                reroute_count,
                blocked_edges,
                reroute_degradation_threshold,
                cluster_degradation_threshold,
            )
            reroute_count += reroute_inc
            if reroute_inc > 0 and accepted_stage in reroute_by_stage:
                reroute_by_stage[accepted_stage] += reroute_inc
            total_wasted_distance += wasted

    depot_arrival_times = [
        event_time
        for event_time, event_type, payload in history_log
        if event_type == "arrival"
        and (
            payload.get("is_return_to_depot", False)
            or payload.get("node_index") == payload.get("depot_index")
        )
    ]
    total_execution_time = max(depot_arrival_times) if depot_arrival_times else current_time

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
    post_reroute_cost_without_wasted = post_reroute_cost - float(total_wasted_distance)

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
    print(
        "Post-reroute (sem U-turn embutido): "
        f"{post_reroute_cost_without_wasted:.2f} "
        f"(change: {post_reroute_cost_without_wasted - original_solution_cost:+.2f})"
    )
    print(f"Wasted (U-turns)        : {total_wasted_distance:.2f}")
    print(
        f"Realized total cost     : {realized_cost:.2f} "
        f"(U-turns ja embutidos, total impact: {total_cost_impact:+.2f})"
    )
    print(f"Reroute operations      : {reroute_count}")
    print(
        "Reroutes by stage      : "
        f"S1={reroute_by_stage['stage1']} | "
        f"S2={reroute_by_stage['stage2']} | "
        f"S3={reroute_by_stage['stage3']}"
    )
    print(
        f"Total execution time   : {total_execution_time:.2f} min "
        "(last arrival at depot)"
    )
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
            "post_reroute_cost_without_wasted": post_reroute_cost_without_wasted,
            "reroute_cost_increase": reroute_cost_increase,
            "wasted_travel_distance": total_wasted_distance,
            "realized_total_cost": realized_cost,
            "total_cost_impact": total_cost_impact,
            "total_execution_time_minutes": total_execution_time,
            "reroute_count": reroute_count,
            "reroute_by_stage": {
                "stage1": reroute_by_stage["stage1"],
                "stage2": reroute_by_stage["stage2"],
                "stage3": reroute_by_stage["stage3"],
            },
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
    reroute_degradation_threshold: float,
    cluster_degradation_threshold: float,
) -> Tuple[int, float, str | None]:
    """
    Handle edge block event by finding affected vehicle and rerouting.

    Returns
    -------
    Tuple[int, float, str | None]
        (number of reroutes performed, wasted distance from U-turn, accepted stage key).
    """
    node_a = event.payload["node_a"]
    node_b = event.payload["node_b"]

    affected_route = find_affected_route_by_broken_edge(node_a, node_b, vehicle_states)
    if affected_route is None:
        return 0, 0.0, None

    affected_vehicle_state = vehicle_states[affected_route]
    original_route = affected_vehicle_state.route
    original_route_cost = original_route.total_distance()

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

    baseline_route_duration = Route(
        depot=original_route.depot,
        customers=list(original_route.customers),
        wasted_duration=historical_wasted_duration,
        wasted_distance=historical_wasted_distance,
    ).total_duration()
    accepted_stage: str | None = None
    accepted_stage_key: str | None = None

    executed_count = max(0, affected_vehicle_state.next_stop_index - 1)
    executed_customers = original_route.customers[:executed_count]

    # Stage 1: local containment (intra-route M1/M2/M3 only).
    stage1_customers, stage1_combined_route, stage1_accepted = reoptimize_intra_route_stage1(
        pending_customers=pending_customers,
        fixed_next_customer=fixed_next_customer,
        reroute_start_node=reroute_start_node,
        event_start_node=event_start_node,
        depot=original_route.depot,
        dist_fn=algorithm._dist,
        executed_customers=executed_customers,
        historical_wasted_duration=historical_wasted_duration,
        historical_wasted_distance=historical_wasted_distance,
        original_route_cost=original_route_cost,
        reroute_degradation_threshold=reroute_degradation_threshold,
        blocked_edges=blocked_edges,
    )

    def _stage1_fallback_is_valid() -> bool:
        uses_broken_edge = _path_uses_blocked_edge(
            event_start_node,
            stage1_customers,
            original_route.depot,
            blocked_edges,
        )
        is_feasible = stage1_combined_route.is_feasible()
        if not is_feasible or uses_broken_edge:
            print(
                "Stage 1 fallback rejected "
                f"(feasible={is_feasible}, uses_broken_edge={uses_broken_edge})."
            )
            return False
        return True

    def _stage2_fallback_is_valid(
        routes_by_id: dict[int, dict[str, object]],
    ) -> bool:
        for route_id, data in routes_by_id.items():
            combined_route: Route = data["combined_route"]
            if not combined_route.is_feasible():
                print(
                    "Stage 2 fallback rejected "
                    f"(route={route_id}, feasible=False)."
                )
                return False

            item = data["item"]
            future_route: Route = data["future_route"]
            uses_broken_edge = _path_uses_blocked_edge(
                item["event_start_node"],
                future_route.customers,
                combined_route.depot,
                blocked_edges,
            )
            if uses_broken_edge:
                print(
                    "Stage 2 fallback rejected "
                    f"(route={route_id}, uses_broken_edge=True)."
                )
                return False
        return True

    if stage1_accepted:
        accepted_stage = "Stage 1"
        accepted_stage_key = "stage1"
        new_route = Route(
            depot=original_route.depot,
            customers=stage1_customers,
            wasted_duration=historical_wasted_duration,
            wasted_distance=historical_wasted_distance,
        )
    else:
        print("Reverting local patch and attempting Stage 2 intra-cluster reoptimization.")

        stage2_accepted, stage2_fallback = reoptimize_intra_cluster(
            vehicle_states=vehicle_states,
            algorithm=algorithm,
            affected_route_id=affected_route,
            current_time=current_time,
            blocked_edges=blocked_edges,
            cluster_degradation_threshold=cluster_degradation_threshold,
            event_start_node=event_start_node,
            reroute_start_time=reroute_start_time,
            wasted_travel_time=wasted_travel_time,
            wasted_travel_distance=wasted_travel_distance,
            fixed_next_customer=fixed_next_customer,
            travel_to_next=travel_to_next,
            leg=leg,
            on_broken_edge=on_broken_edge,
        )
        if stage2_accepted is not None:
            _commit_stage2_updates(
                new_routes_by_id=stage2_accepted,
                affected_route_id=affected_route,
                current_solution=current_solution,
                event_queue=event_queue,
                current_time=current_time,
                instance_name=instance_name,
                algorithm=algorithm,
                reroute_index=reroute_count + 1,
                broken_edge=broken_edge,
                wasted_travel_time=wasted_travel_time,
                wasted_travel_distance=wasted_travel_distance,
            )
            return 1, wasted_travel_distance, "stage2"

        print("Stage 2 rejected; proceeding to Stage 3 global cross-depot repair.")

        # Align affected vehicle historical waste before Stage 3 transfer updates.
        affected_vehicle_state.route = Route(
            depot=affected_vehicle_state.route.depot,
            customers=list(affected_vehicle_state.route.customers),
            wasted_duration=historical_wasted_duration,
            wasted_distance=historical_wasted_distance,
        )

        target_node = _resolve_stage3_target_node(affected_vehicle_state, broken_edge)
        if target_node is None:
            print(
                "Stage 3 aborted: could not resolve target_node from blocked edge; "
                "falling back to Stage 1 contingency."
            )
            if _stage1_fallback_is_valid():
                print("Using Stage 1 feasible route as contingency after Stage 3 failure.")
                accepted_stage = "Stage 3 fallback"
                accepted_stage_key = "stage1"
                new_route = Route(
                    depot=original_route.depot,
                    customers=stage1_customers,
                    wasted_duration=historical_wasted_duration,
                    wasted_distance=historical_wasted_distance,
                )
            else:
                print("Stage 3 failed and Stage 1 is infeasible; keeping original route.")
                return 0, 0.0, None
        else:
            routes_before_stage3 = {
                route_id: _clone_route(state.route)
                for route_id, state in vehicle_states.items()
            }
            stage3_distance_matrix = _build_stage3_distance_matrix(vehicle_states, blocked_edges)
            rescued_state = stage3_global_cross_depot_repair(
                target_node=target_node,
                vehicle_states=vehicle_states,
                distance_matrix=stage3_distance_matrix,
                blocked_vehicle_id=affected_route,
            )

            if rescued_state is None:
                if _stage1_fallback_is_valid():
                    print("Using Stage 1 feasible route as contingency after Stage 3 failure.")
                    accepted_stage = "Stage 3 fallback"
                    accepted_stage_key = "stage1"
                    new_route = Route(
                        depot=original_route.depot,
                        customers=stage1_customers,
                        wasted_duration=historical_wasted_duration,
                        wasted_distance=historical_wasted_distance,
                    )

                elif stage2_fallback is not None and _stage2_fallback_is_valid(stage2_fallback):
                    print("Using Stage 2 feasible cluster as contingency after Stage 3 failure.")
                    _commit_stage2_updates(
                        new_routes_by_id=stage2_fallback,
                        affected_route_id=affected_route,
                        current_solution=current_solution,
                        event_queue=event_queue,
                        current_time=current_time,
                        instance_name=instance_name,
                        algorithm=algorithm,
                        reroute_index=reroute_count + 1,
                        broken_edge=broken_edge,
                        wasted_travel_time=wasted_travel_time,
                        wasted_travel_distance=wasted_travel_distance,
                    )
                    return 1, wasted_travel_distance, "stage2"
                else:
                    print("Stage 3 failed and Fallbacks are infeasible; keeping original route.")
                    return 0, 0.0, None
            else:
                donor_pending = [c for c in pending_customers if c.index != target_node]
                
                algorithm._build_matrix(nodes_for_matrix, pending_customers)
                for b_edge in blocked_edges:
                    algorithm._set_edge_inf(*b_edge)
                
                # Reoptimize only the tail of the donor route (after the fixed_next_customer), since the prefix is commitment-bound and the target_node is now removed from the donor's pending pool.
                donor_optimized_tail = local_search_stage1_intra(
                    customers=donor_pending,
                    start_node=reroute_start_node,
                    end_node=original_route.depot,
                    dist_fn=algorithm._dist,
                )
                
                # Rebuild the donor's future route with the optimized tail and check for broken edge usage, applying cascading failure cleanup if necessary.
                donor_future_customers = [
                    *([fixed_next_customer] if fixed_next_customer is not None else []),
                    *donor_optimized_tail,
                ]
                
                donor_uses_broken = _path_uses_blocked_edge(
                    event_start_node,
                    donor_future_customers,
                    original_route.depot,
                    blocked_edges,
                )
                
                cascading_victims = []
                while donor_uses_broken and donor_optimized_tail:
                    # Iteratively remove the first customer from the optimized tail (the one closest to the fixed_next_customer) until the broken edge is no longer used or we run out of customers. This simulates a cascading failure effect where customers that cannot be served without crossing the broken edge are dropped.
                    dropped_customer = donor_optimized_tail.pop(0)
                    cascading_victims.append(dropped_customer)
                    
                    donor_future_customers = [
                        *([fixed_next_customer] if fixed_next_customer is not None else []),
                        *donor_optimized_tail,
                    ]
                    
                    donor_uses_broken = _path_uses_blocked_edge(
                        event_start_node,
                        donor_future_customers,
                        original_route.depot,
                        blocked_edges,
                    )
                
                if cascading_victims:
                    dropped_ids = [c.index for c in cascading_victims]
                    print(f"Stage 3 Domino Effect: the following customers had to be dropped from the donor route to eliminate broken edge usage: {dropped_ids}")   
                
                if donor_uses_broken and not donor_optimized_tail:
                    print(f"Stage 3 Warning: Vehicle {rescued_state.route_id} is left with only the fixed next customer and still uses the broken edge. This customer may be effectively unserviceable in the short term.")
                
                # If the donor route still uses the broken edge after cleanup, we consider Stage 3 a failure for this iteration, as it indicates that the fixed_next_customer is effectively trapped and cannot be served without crossing the broken edge. In a real-world scenario, this might trigger an alert for manual intervention or a more drastic contingency plan.
                if donor_uses_broken:
                    print("Stage 3 aborted: unresolvable broken edge on donor route (likely fixed_next_customer is trapped).")
                    if _stage1_fallback_is_valid():
                        print("Using Stage 1 feasible route as fallback.")
                        accepted_stage = "Stage 3 fallback"
                        accepted_stage_key = "stage1"
                        new_route = Route(
                            depot=original_route.depot,
                            customers=stage1_customers,
                            wasted_duration=historical_wasted_duration,
                            wasted_distance=historical_wasted_distance,
                        )
                    else:
                        return 0, 0.0, None
                else:
                    _commit_stage3_updates(
                        winner_state=rescued_state,
                        target_node=target_node,
                        donor_repaired_customers=donor_future_customers,
                        vehicle_states=vehicle_states,
                        current_solution=current_solution,
                        event_queue=event_queue,
                        current_time=current_time,
                        instance_name=instance_name,
                        algorithm=algorithm,
                        reroute_index=reroute_count + 1,
                        broken_edge=broken_edge,
                        original_route=original_route,
                        routes_before_stage3=routes_before_stage3,
                        baseline_route_duration=baseline_route_duration,
                        event_start_node=event_start_node,
                        current_node=current_node,
                        reroute_start_time=reroute_start_time,
                        fixed_next_customer=fixed_next_customer,
                        travel_to_next=travel_to_next,
                        wasted_travel_time=wasted_travel_time,
                        wasted_travel_distance=wasted_travel_distance,
                        affected_route_id=affected_route,
                    )

                    for orphan in cascading_victims:
                        orphan_id = orphan.index
                        print(f"Stage 3 Rescue Operation: Attempting to find a new route for orphaned customer {orphan_id} dropped from donor route {rescued_state.route_id}.")
                        
                        # Temporarily inject the orphan customer into the affected vehicle's state to evaluate if any other vehicle can pick it up without crossing the broken edge. This simulates a rescue operation for customers that were collateral damage in the Stage 3 repair.
                        vehicle_states[affected_route].customers_by_index[orphan_id] = orphan
                        
                        orphan_winner_state = stage3_global_cross_depot_repair(
                            target_node=orphan_id,
                            vehicle_states=vehicle_states,
                            distance_matrix=stage3_distance_matrix,
                            blocked_vehicle_id=affected_route,
                        )
                        
                        # Remove the temporary injection to keep memory clean
                        del vehicle_states[affected_route].customers_by_index[orphan_id]
                        
                        if orphan_winner_state:
                            w_live = vehicle_states[orphan_winner_state.route_id]
                            w_live.route = orphan_winner_state.route
                            w_live.customers_by_index = {c.index: c for c in w_live.route.customers}
                            w_live.pending_customer_ids = {c.index for c in w_live.route.customers} - w_live.visited_customer_ids
                            current_solution.routes[w_live.route_id - 1] = w_live.route
                            
                            event_queue.remove_future_events_for_route(w_live.route_id, current_time)
                            
                            o_exec_count = max(0, w_live.next_stop_index - 1)
                            o_future_route = Route(
                                depot=w_live.route.depot,
                                customers=list(w_live.route.customers[o_exec_count:]),
                                wasted_duration=w_live.route.wasted_duration,
                                wasted_distance=w_live.route.wasted_distance,
                            )
                            
                            o_curr_node = _resolve_current_node(w_live)
                            o_fixed_next, o_travel_to = determine_fixed_next_customer(w_live, False, current_time)
                            
                            schedule_rerouted_events(
                                event_queue,
                                w_live.route_id,
                                o_future_route,
                                o_curr_node,
                                o_curr_node,
                                current_time,
                                w_live,
                                o_fixed_next,
                                o_travel_to,
                                stop_index_offset=o_exec_count,
                            )
                            print(f"Stage 3 Rescue Operation: SUCCESS. Orphan customer {orphan_id} rescued by vehicle {w_live.route_id} in a secondary Stage 3 operation.")
                        else:
                            print(f"Stage 3 Rescue Operation: FAILED. No vehicle in the network has the capacity/time to save customer {orphan_id}.")

                    return 1, wasted_travel_distance, "stage3"

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

    if accepted_stage is not None:
        route_delta_pct = 0.0
        if baseline_route_duration > 0:
            route_delta_pct = (
                (combined_route.total_duration() - baseline_route_duration)
                / baseline_route_duration
            ) * 100.0
        print(
            f"{accepted_stage} cost change "
            f"(old={baseline_route_duration:.2f}, "
            f"new={combined_route.total_duration():.2f}, "
            f"delta={route_delta_pct:+.2f}%)."
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

    final_stage_key = accepted_stage_key if accepted_stage_key is not None else "stage1"
    return 1, wasted_travel_distance, final_stage_key

