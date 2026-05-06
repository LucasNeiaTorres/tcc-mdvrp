from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Ensure project `src` package is importable when running this script from repo root
repo_root = Path(__file__).resolve().parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.entities import Route
from core.solution import Solution
from utils.converter import build_customers, build_depots
from utils.data_loader import read_cordeau_data_file, read_cordeau_solution_file
from utils.visualizer import visualize_comparison


def _node_positions(instance) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for customer in instance.customers:
        positions[customer.index] = (customer.x, customer.y)
    for depot in instance.depots:
        positions[depot.index] = (depot.x, depot.y)
    return positions


def _draw_nodes(ax, instance) -> None:
    customer_x = [c.x for c in instance.customers]
    customer_y = [c.y for c in instance.customers]
    ax.scatter(customer_x, customer_y, c="steelblue", s=80, marker="o", label="Customers", zorder=3)

    depot_x = [d.x for d in instance.depots]
    depot_y = [d.y for d in instance.depots]
    ax.scatter(depot_x, depot_y, c="crimson", s=250, marker="*", label="Depots", zorder=4)

    for customer in instance.customers:
        ax.annotate(str(customer.index), xy=(customer.x, customer.y), xytext=(3, 3), textcoords="offset points", fontsize=6, alpha=0.7)
    for depot in instance.depots:
        ax.annotate(f"D{depot.index}", xy=(depot.x, depot.y), xytext=(3, 3), textcoords="offset points", fontsize=8, fontweight="bold", alpha=0.9)


def _draw_path(ax, positions: dict[int, tuple[float, float]], node_indices: list[int], *, color: str, linestyle: str, label: str | None = None, zorder: int = 2) -> None:
    if len(node_indices) < 2:
        return

    xs = [positions[idx][0] for idx in node_indices if idx in positions]
    ys = [positions[idx][1] for idx in node_indices if idx in positions]
    if len(xs) < 2:
        return

    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.85, label=label, zorder=zorder)
    for i in range(len(xs) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1], ys[i + 1]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="->", color=color, alpha=0.55, lw=1),
            zorder=zorder + 1,
        )


def _draw_vehicle_snapshots(instance, vehicles: list[dict]) -> None:
    if not vehicles:
        return

    positions = _node_positions(instance)
    fig, axes = plt.subplots(1, len(vehicles), figsize=(7 * len(vehicles), 6))
    if len(vehicles) == 1:
        axes = [axes]

    for ax, vehicle in zip(axes, vehicles):
        _draw_nodes(ax, instance)

        executed = vehicle.get("executed_path", {})
        future = vehicle.get("future_path", {})
        full_route = vehicle.get("full_route", {})

        _draw_path(
            ax,
            positions,
            executed.get("path_nodes", []),
            color="black",
            linestyle="-",
            label="Executed",
            zorder=5,
        )
        _draw_path(
            ax,
            positions,
            future.get("path_nodes", []),
            color="darkorange",
            linestyle="--",
            label="Future",
            zorder=6,
        )
        _draw_path(
            ax,
            positions,
            full_route.get("path_nodes", []),
            color="dodgerblue",
            linestyle=":",
            label="Full",
            zorder=4,
        )

        ax.set_title(
            f"Vehicle {vehicle.get('route_id')} | executed={executed.get('travel_distance', 0.0):.2f} | full={full_route.get('travel_distance', 0.0):.2f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def _build_solution_from_routing_json(instance, routing_json: dict) -> Solution:
    customers_by_index = {customer.index: customer for customer in build_customers(instance)}
    depots_by_index = {depot.index: depot for depot in build_depots(instance)}

    routes = []
    for route_payload in routing_json.get("routes", []):
        depot_index = int(route_payload["depot_index"])
        customer_indices = [int(index) for index in route_payload.get("customer_indices", [])]

        routes.append(
            Route(
                depot=depots_by_index[depot_index],
                customers=[customers_by_index[index] for index in customer_indices],
            )
        )

    return Solution(routes=routes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a saved reroute JSON against the reference solution for debugging."
    )
    parser.add_argument(
        "--data-file",
        default="data/raw/cordeau/p02",
        help="Path to the raw Cordeau instance file.",
    )
    parser.add_argument(
        "--reference-solution",
        default="data/raw/cordeau_sol/p02.res",
        help="Path to the reference solution file.",
    )
    parser.add_argument(
        "--routing-json",
        default="data/processed/results/p02_reroute_001_t001630.json",
        help="Path to the saved routing JSON generated during simulation.",
    )
    args = parser.parse_args()

    data_file = Path(args.data_file)
    reference_file = Path(args.reference_solution)
    routing_file = Path(args.routing_json)

    instance = read_cordeau_data_file(str(data_file))
    reference_solution = read_cordeau_solution_file(str(reference_file), instance)

    with routing_file.open("r", encoding="utf-8") as f:
        routing_payload = json.load(f)

    reroute_solution = _build_solution_from_routing_json(instance, routing_payload)

    comparison_title = routing_payload.get("metadata", {}).get("algorithm", "Reroute result")
    visualize_comparison(
        instance,
        [reference_solution, reroute_solution],
        titles=[
            f"Reference (obj: {reference_solution.objective:.2f})",
            f"{comparison_title} (cost: {reroute_solution.total_cost():.2f})",
        ],
    )

    _draw_vehicle_snapshots(instance, routing_payload.get("vehicles", []))


if __name__ == "__main__":
    main()