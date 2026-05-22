"""Post-simulation validation and metric calculations."""

from typing import List, Tuple
from scenario.state import VehicleState, _normalize_edge


def extract_blocked_edges(history_log: List[Tuple[float, str, dict]]) -> dict[tuple[int, int], float]:
    """Extract blocked edges from history log with their block times."""
    blocked_edges: dict[tuple[int, int], float] = {}
    for event_time, etype, payload in history_log:
        if etype == "edge_block":
            edge = _normalize_edge(payload["node_a"], payload["node_b"])
            blocked_edges[edge] = min(blocked_edges.get(edge, float("inf")), float(event_time))
    return blocked_edges


def extract_route_stop_events(
    history_log: List[Tuple[float, str, dict]]
) -> dict[int, list[Tuple[float, int, float | None]]]:
    """Extract arrival/service_end events per route."""
    route_stop_events: dict[int, list[Tuple[float, int, float | None]]] = {}

    for event_time, etype, payload in history_log:
        if etype == "arrival":
            route_id = payload.get("route_id")
            node_index = payload.get("node_index")
            if route_id is None or node_index is None:
                continue
            route_stop_events.setdefault(int(route_id), []).append((float(event_time), int(node_index), None))
            continue

        if etype == "service_end":
            route_id = payload.get("route_id")
            node_index = payload.get("node_index")
            if route_id is None or node_index is None:
                continue
            stops = route_stop_events.get(int(route_id), [])
            for idx in range(len(stops) - 1, -1, -1):
                arrival_time, existing_node_index, service_end_time = stops[idx]
                if existing_node_index == int(node_index) and service_end_time is None:
                    stops[idx] = (arrival_time, existing_node_index, float(event_time))
                    break

    return route_stop_events


def find_routes_using_broken_edges(
    route_stop_events: dict[int, list[Tuple[float, int, float | None]]],
    blocked_edges: dict[tuple[int, int], float],
) -> set[int]:
    """Identify routes that used blocked edges after they were blocked."""
    routes_using_broken: set[int] = set()

    for route_id, stops in route_stop_events.items():
        if len(stops) < 2:
            continue
        prev_arrival_time, prev_node, prev_service_end_time = stops[0]
        depart_time = prev_service_end_time if prev_service_end_time is not None else prev_arrival_time

        for arrival_time, node, service_end_time in stops[1:]:
            edge = _normalize_edge(prev_node, node)
            block_time = blocked_edges.get(edge)
            if block_time is not None and block_time < depart_time:
                routes_using_broken.add(route_id)
                break
            depart_time = service_end_time if service_end_time is not None else arrival_time
            prev_node = node

    return routes_using_broken


def extract_visited_customers(vehicle_states: dict[int, VehicleState]) -> set[int]:
    """Extract set of visited customer indices from vehicle states."""
    visited = set()
    for vs in vehicle_states.values():
        visited |= set(vs.visited_customer_ids)
    return visited


def calculate_cost_metrics(
    original_solution_cost: float,
    post_reroute_cost: float,
    total_wasted_distance: float,
) -> tuple[float, float, float, float]:
    """Calculate all cost-related metrics."""
    reroute_cost_increase = post_reroute_cost - original_solution_cost
    # Wasted distance is already embedded in Route.total_distance when reroutes are committed.
    _ = total_wasted_distance
    realized_cost = post_reroute_cost
    total_cost_impact = realized_cost - original_solution_cost
    return reroute_cost_increase, realized_cost, total_cost_impact, realized_cost
