import matplotlib.pyplot as plt
from typing import List, Optional
import numpy as np

from .data_loader import CordeauInstance
from core.protocols import VisualizableSolution


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _node_positions(instance: CordeauInstance) -> dict:
    """Return {node_index: (x, y)} for every customer and depot."""
    positions: dict = {}
    for c in instance.customers:
        positions[c.index] = (c.x, c.y)
    for d in instance.depots:
        positions[d.index] = (d.x, d.y)
    return positions


def _draw_routes(ax, instance: CordeauInstance, solution: VisualizableSolution) -> None:
    """Draw all routes of *solution* onto *ax*."""
    routes = solution.visualizable_routes
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(routes), 3)))

    for route_idx, route in enumerate(routes):
        color = colors[route_idx % len(colors)]

        depot_pos = _node_positions(instance).get(route.depot_index)
        if depot_pos is None:
            continue

        xs = [depot_pos[0]]
        ys = [depot_pos[1]]

        node_pos = _node_positions(instance)
        for cidx in route.customer_indices:
            pos = node_pos.get(cidx)
            if pos:
                xs.append(pos[0])
                ys.append(pos[1])

        xs.append(depot_pos[0])
        ys.append(depot_pos[1])

        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.6, zorder=1)

        for i in range(len(xs) - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.5, lw=1),
                zorder=2,
            )


def _draw_nodes(ax, instance: CordeauInstance, show_labels: bool) -> None:
    """Scatter-plot customers and depots; optionally annotate with index."""
    customer_x = [c.x for c in instance.customers]
    customer_y = [c.y for c in instance.customers]
    ax.scatter(customer_x, customer_y, c="steelblue", s=80, marker="o",
               label="Customers", zorder=3)

    depot_x = [d.x for d in instance.depots]
    depot_y = [d.y for d in instance.depots]
    ax.scatter(depot_x, depot_y, c="crimson", s=250, marker="*",
               label="Depots", zorder=4)

    if show_labels:
        for c in instance.customers:
            ax.annotate(str(c.index), xy=(c.x, c.y), xytext=(3, 3),
                        textcoords="offset points", fontsize=6, alpha=0.7)
        for d in instance.depots:
            ax.annotate(f"D{d.index}", xy=(d.x, d.y), xytext=(3, 3),
                        textcoords="offset points", fontsize=8,
                        fontweight="bold", alpha=0.9)


def _style_ax(ax, title: str) -> None:
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_instance(
    instance: CordeauInstance,
    title: str = "MDVRP Instance",
    show_labels: bool = True,
) -> None:
    """Plot customers and depots with no routes."""
    fig, ax = plt.subplots(figsize=(10, 9))
    _draw_nodes(ax, instance, show_labels)
    ax.legend(loc="upper right", fontsize=11)
    _style_ax(ax, title)
    plt.tight_layout()
    plt.show()


def visualize_solution(
    instance: CordeauInstance,
    solution: VisualizableSolution,
    title: str = "MDVRP Solution",
    show_labels: bool = True,
) -> None:
    """Plot a single solution overlaid on the instance."""
    fig, ax = plt.subplots(figsize=(12, 10))
    _draw_routes(ax, instance, solution)
    _draw_nodes(ax, instance, show_labels)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    _style_ax(ax, title)
    plt.tight_layout()
    plt.show()


def visualize_comparison(
    instance: CordeauInstance,
    solutions: List[VisualizableSolution],
    titles: Optional[List[str]] = None,
    show_labels: bool = True,
) -> None:
    """Show multiple solutions side-by-side. """
    if not solutions:
        raise ValueError("At least one solution is required.")

    if titles is None:
        titles = [f"Solution {i + 1}" for i in range(len(solutions))]

    n = len(solutions)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, solution, title in zip(axes, solutions, titles):
        _draw_routes(ax, instance, solution)
        _draw_nodes(ax, instance, show_labels)
        _style_ax(ax, title)

    plt.tight_layout()
    plt.show()
