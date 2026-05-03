"""Utilities to serialize clustering and routing outputs to JSON."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.entities import Route
from core.solution import Solution


def _build_metadata(instance_name: str, algorithm_name: str) -> dict:
    return {
        "instance": instance_name,
        "algorithm": algorithm_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _serialize_route(route: Route, route_id: int) -> dict:
    return {
        "route_id": route_id,
        "depot_index": route.depot.index,
        "customer_indices": [c.index for c in route.customers],
        "total_demand": route.total_demand(),
        "total_distance": route.total_distance(),
        "total_duration": route.total_duration(),
        "feasible": route.is_feasible(),
    }


def save_clustering_result(
    output_path: str,
    instance_name: str,
    algorithm_name: str,
    clusters: Dict[int, list[int]],
) -> Path:
    """Save clustering artifact (customer assignment by depot) to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": _build_metadata(instance_name, algorithm_name),
        "summary": {
            "cluster_count": len(clusters),
            "customer_count": sum(len(v) for v in clusters.values()),
        },
        "clusters": [
            {
                "depot_index": depot_idx,
                "customer_indices": customer_indices,
                "customer_count": len(customer_indices),
            }
            for depot_idx, customer_indices in sorted(clusters.items())
        ],
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")

    return out


def save_routing_result(
    output_path: str,
    instance_name: str,
    algorithm_name: str,
    solution: Solution,
) -> Path:
    """Save routing artifact (final routes and metrics) to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    routes_payload = [
        _serialize_route(route, route_idx)
        for route_idx, route in enumerate(solution.routes, start=1)
    ]

    payload = {
        "metadata": _build_metadata(instance_name, algorithm_name),
        "summary": {
            "route_count": len(solution.routes),
            "total_cost": solution.total_cost(),
            "feasible": solution.is_feasible(),
        },
        "routes": routes_payload,
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")

    return out


def save_reroute_result(
    output_path: str,
    instance_name: str,
    algorithm_name: str,
    solution: Solution,
    vehicles: List[dict[str, Any]],
    current_time_minutes: float,
    broken_edge: tuple[int, int],
    reroute_index: int,
) -> Path:
    """Save a reroute snapshot with executed/future/combined per-vehicle paths."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    realized_total_cost = sum(
        float(vehicle.get("full_route", {}).get("travel_distance", 0.0))
        for vehicle in vehicles
    )

    payload = {
        "metadata": {
            **_build_metadata(instance_name, algorithm_name),
            "reroute_index": reroute_index,
            "current_time_minutes": current_time_minutes,
            "broken_edge": list(broken_edge),
        },
        "summary": {
            "route_count": len(solution.routes),
            "planned_total_cost": solution.total_cost(),
            "realized_total_cost": realized_total_cost,
            "feasible": solution.is_feasible(),
            "affected_vehicle_count": len(vehicles),
        },
        "routes": [
            _serialize_route(route, route_idx)
            for route_idx, route in enumerate(solution.routes, start=1)
        ],
        "vehicles": vehicles,
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")

    return out


def save_clustering_and_routing(
    output_path: str,
    instance_name: str,
    algorithm_name: str,
    clusters: Dict[int, list[int]],
    solution: Solution,
) -> Path:
    """Backward-compatible combined export with both clusters and routes."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": _build_metadata(instance_name, algorithm_name),
        "summary": {
            "cluster_count": len(clusters),
            "route_count": len(solution.routes),
            "total_cost": solution.total_cost(),
            "feasible": solution.is_feasible(),
        },
        "clusters": [
            {
                "depot_index": depot_idx,
                "customer_indices": customer_indices,
                "customer_count": len(customer_indices),
            }
            for depot_idx, customer_indices in sorted(clusters.items())
        ],
        "routes": [
            _serialize_route(route, route_idx)
            for route_idx, route in enumerate(solution.routes, start=1)
        ],
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")

    return out

def save_history_log(
    output_path: str,
    instance_name: str,
    history_log: List[Tuple[float, str, Dict[str, Any]]],
) -> Path:
    """Save simulation event history to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Count events by type
    event_counts: Dict[str, int] = {}
    for _, event_type, _ in history_log:
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    # Serialize events
    events_payload = [
        {
            "time_minutes": time,
            "type": event_type,
            "payload": payload,
        }
        for time, event_type, payload in history_log
    ]

    # Build payload
    payload = {
        "metadata": {
            "instance": instance_name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "summary": {
            "total_events": len(history_log),
            "event_counts_by_type": event_counts,
            "total_time_minutes": history_log[-1][0] if history_log else 0.0,
        },
        "events": events_payload,
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")

    return out