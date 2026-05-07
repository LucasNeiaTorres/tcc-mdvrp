import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

# Default directory for simulation logs
LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "simulation_logs"
LOG_FILENAME = "{instance}_log.json"


def load_log(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_blocked_edges(events: Iterable[dict]) -> Dict[frozenset, float]:
    """Return map edge -> first block time (minutes).

    Edge is represented as frozenset({node_a, node_b}).
    """
    blocked: Dict[frozenset, float] = {}
    for ev in events:
        if ev.get("type") == "edge_block":
            payload = ev.get("payload", {})
            a = payload.get("node_a")
            b = payload.get("node_b")
            if a is None or b is None:
                continue
            key = frozenset({int(a), int(b)})
            t = float(ev.get("time_minutes", 0.0))
            if key not in blocked or t < blocked[key]:
                blocked[key] = t
    return blocked


def extract_routes(events: Iterable[dict]) -> Dict[int, List[Tuple[int, int, float, float | None]]]:
    """Return mapping route_id -> ordered list of stops.

    Each stop is represented as (stop_index, node_index, arrival_time, service_end_time).
    The final return-to-depot stop has no service_end_time.
    """
    routes: Dict[int, List[Tuple[int, int, float, float | None]]] = {}
    for ev in events:
        if ev.get("type") != "arrival":
            continue
        payload = ev.get("payload", {})
        route_id = payload.get("route_id")
        stop_index = payload.get("stop_index")
        node_index = payload.get("node_index")
        t = float(ev.get("time_minutes", 0.0))
        if route_id is None or node_index is None or stop_index is None:
            continue
        routes.setdefault(int(route_id), []).append((int(stop_index), int(node_index), t, None))
    for k in list(routes.keys()):
        routes[k].sort(key=lambda x: x[0])

    for ev in events:
        if ev.get("type") != "service_end":
            continue
        payload = ev.get("payload", {})
        route_id = payload.get("route_id")
        stop_index = payload.get("stop_index")
        node_index = payload.get("node_index")
        t = float(ev.get("time_minutes", 0.0))
        if route_id is None or node_index is None or stop_index is None:
            continue
        route_stops = routes.get(int(route_id), [])
        for idx, (existing_stop_index, existing_node_index, arrival_time, service_end_time) in enumerate(route_stops):
            if existing_stop_index == int(stop_index) and existing_node_index == int(node_index):
                route_stops[idx] = (existing_stop_index, existing_node_index, arrival_time, t)
                break
    return routes


def extract_expected_customers(data: dict) -> set[int] | None:
    metadata = data.get("metadata", {})
    expected = metadata.get("expected_customer_indices")
    if expected is None:
        return None
    return {int(node_index) for node_index in expected}


def find_violations(blocked: Dict[frozenset, float], routes: Dict[int, List[Tuple[int, int, float, float | None]]]):
    """Yield violations as tuples (route_id, from_node, to_node, depart_time, arrive_time, block_time)."""
    for route_id, seq in routes.items():
        if len(seq) < 2:
            continue
        prev_stop_index, prev_node, prev_arrival_time, prev_service_end_time = seq[0]
        depart_time = prev_service_end_time if prev_service_end_time is not None else prev_arrival_time
        for stop_index, node, arrival_time, service_end_time in seq[1:]:
            edge = frozenset({prev_node, node})
            block_time = blocked.get(edge)
            if block_time is not None and block_time < depart_time:
                yield (route_id, prev_node, node, depart_time, arrival_time, block_time)
            prev_stop_index = stop_index
            prev_node = node
            prev_arrival_time = arrival_time
            depart_time = service_end_time if service_end_time is not None else arrival_time


def find_unserved_customers(
    events: Iterable[dict],
    expected_customers: set[int] | None = None,
) -> list[int]:
    arrived_customers: set[int] = set()
    serviced_customers: set[int] = set()

    for ev in events:
        payload = ev.get("payload", {})
        node_index = payload.get("node_index")
        if node_index is None:
            continue

        node_index = int(node_index)
        if node_index == 0 or payload.get("is_return_to_depot", False):
            continue

        if ev.get("type") == "arrival":
            arrived_customers.add(node_index)
        elif ev.get("type") == "service_end":
            serviced_customers.add(node_index)

    unserved_customers = arrived_customers - serviced_customers
    if expected_customers is not None:
        unserved_customers |= expected_customers - arrived_customers

    return sorted(unserved_customers)


def validate_simulation_log(log_path: Path) -> dict:
    """Return blocked-edge violations and unserved customers found in a simulation log."""
    data = load_log(log_path)
    events = data.get("events", [])
    blocked = extract_blocked_edges(events)
    routes = extract_routes(events)
    blocked_edge_violations = list(find_violations(blocked, routes))
    expected_customers = extract_expected_customers(data)
    unserved_customers = find_unserved_customers(events, expected_customers)
    return {
        "blocked_edge_violations": blocked_edge_violations,
        "unserved_customers": unserved_customers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate simulation log against blocked edges.")
    parser.add_argument("--instance", help="Instance name, e.g. p01")
    parser.add_argument("--log-file", help="Path to simulation log JSON. If provided, it overrides --instance.")
    args = parser.parse_args()

    if args.log_file:
        log_path = Path(args.log_file)
    elif args.instance:
        log_path = LOG_DIR / LOG_FILENAME.format(instance=args.instance)
    else:
        parser.error("Either --instance or --log-file must be provided")

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return 2

    data = load_log(log_path)
    events = data.get("events", [])
    blocked = extract_blocked_edges(events)
    routes = extract_routes(events)
    validation_result = validate_simulation_log(log_path)
    blocked_edge_violations = validation_result["blocked_edge_violations"]
    unserved_customers = validation_result["unserved_customers"]

    print(f"Loaded log: {log_path}")
    print(f"Blocked edges: {len(blocked)}")
    print(f"Routes found: {len(routes)}")
    print(f"Blocked-edge violations: {len(blocked_edge_violations)}")
    print(f"Unserved customers: {len(unserved_customers)}")

    if blocked_edge_violations:
        print("\nBlocked-edge violations detected:\n")
        for r_id, a, b, depart, arrival, btime in blocked_edge_violations:
            print(
                f"Route {r_id}: traversed edge {a} <-> {b} between t={depart:.3f}min and t={arrival:.3f}min "
                f"but it was blocked at t={btime:.3f}min"
            )

    if unserved_customers:
        print("\nUnserved customers detected:\n")
        print(f"Customers not served: {unserved_customers}")

    if blocked_edge_violations or unserved_customers:
        return 1

    print("No violations found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
