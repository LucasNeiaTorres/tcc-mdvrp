import matplotlib.pyplot as plt
from typing import Optional
import numpy as np

from .data_loader import CordeauInstance, CordeauSolution


def visualize_instance(
    instance: CordeauInstance,
    title: str = "MDVRP Instance",
    show_labels: bool = True
) -> None:
    """
    Visualize the MDVRP instance with customers and depots.
    
    Args:
        instance: CordeauInstance to visualize
        title: Title for the plot
        show_labels: Whether to show node index labels
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot customers
    customer_x = [c.x for c in instance.customers]
    customer_y = [c.y for c in instance.customers]
    ax.scatter(customer_x, customer_y, c="blue", s=100, marker="o", label="Customers", zorder=3)
    
    # Plot depots
    depot_x = [d.x for d in instance.depots]
    depot_y = [d.y for d in instance.depots]
    ax.scatter(depot_x, depot_y, c="red", s=300, marker="*", label="Depots", zorder=4)
    
    # Add labels
    if show_labels:
        for customer in instance.customers:
            ax.annotate(
                str(customer.index),
                xy=(customer.x, customer.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7
            )
        for depot in instance.depots:
            ax.annotate(
                f"D{depot.index}",
                xy=(depot.x, depot.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                alpha=0.8
            )
    
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    
    plt.tight_layout()
    plt.show()


def visualize_solution(
    instance: CordeauInstance,
    solution: CordeauSolution,
    title: Optional[str] = None,
    show_labels: bool = True
) -> None:
    """
    Visualize the MDVRP solution with routes overlaid on the instance.
        
    Args:
        instance: CordeauInstance
        solution: CordeauSolution with routes
        title: Title for the plot (default: includes objective value)
        show_labels: Whether to show node index labels
    """
    if title is None:
        title = f"MDVRP Solution (Objective: {solution.objective:.2f})"
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Generate distinct colors for routes
    num_routes = len(solution.routes)
    colors = plt.cm.tab20(np.linspace(0, 1, max(num_routes, 3)))
    
    # Plot routes
    for route_idx, route in enumerate(solution.routes):
        color = colors[route_idx % len(colors)]
        
        # Find depot for this route (route.depot is 1-based depot number)
        if route.depot - 1 >= len(instance.depots) or route.depot < 1:
            continue
        depot = instance.depots[route.depot - 1]
        
        # Build route sequence: depot -> customers -> depot
        route_points_x = [depot.x]
        route_points_y = [depot.y]
        
        for customer_idx in route.nodes:
            customer = next((c for c in instance.customers if c.index == customer_idx), None)
            if customer:
                route_points_x.append(customer.x)
                route_points_y.append(customer.y)
        
        route_points_x.append(depot.x)
        route_points_y.append(depot.y)
        
        # Draw route line
        ax.plot(
            route_points_x,
            route_points_y,
            color=color,
            linewidth=1.5,
            alpha=0.6,
            zorder=1,
            label=f"Route {route_idx + 1} (Depot {route.depot}, Vehicle {route.vehicle})"
        )
        
        # Mark route direction with arrows
        for i in range(len(route_points_x) - 1):
            ax.annotate(
                "",
                xy=(route_points_x[i + 1], route_points_y[i + 1]),
                xytext=(route_points_x[i], route_points_y[i]),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.5, lw=1),
                zorder=2
            )
    
    # Plot customers on top
    customer_x = [c.x for c in instance.customers]
    customer_y = [c.y for c in instance.customers]
    ax.scatter(customer_x, customer_y, c="blue", s=100, marker="o", label="Customers", zorder=3)
    
    # Plot depots on top
    depot_x = [d.x for d in instance.depots]
    depot_y = [d.y for d in instance.depots]
    ax.scatter(depot_x, depot_y, c="red", s=300, marker="*", label="Depots", zorder=4)
    
    # Add labels
    if show_labels:
        for customer in instance.customers:
            ax.annotate(
                str(customer.index),
                xy=(customer.x, customer.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                alpha=0.7
            )
        for depot in instance.depots:
            ax.annotate(
                f"D{depot.index}",
                xy=(depot.x, depot.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                alpha=0.8
            )
    
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    
    plt.tight_layout()
    plt.show()


def visualize_comparison(
    instance: CordeauInstance,
    solutions: list,
    titles: Optional[list] = None,
    show_labels: bool = True
) -> None:
    """
    Visualize multiple solutions side-by-side for comparison.
    
    Args:
        instance: CordeauInstance
        solutions: List of CordeauSolution objects
        titles: Titles for each subplot (default: includes objective values)
        show_labels: Whether to show node index labels
    """
    num_solutions = len(solutions)
    if num_solutions == 0:
        raise ValueError("At least one solution is required")
    
    if titles is None:
        titles = [f"Solution {i + 1}\n(Obj: {sol.objective:.2f})" for i, sol in enumerate(solutions)]
    
    fig, axes = plt.subplots(1, num_solutions, figsize=(6 * num_solutions, 5))
    if num_solutions == 1:
        axes = [axes]
    
    for sol_idx, (ax, solution, title) in enumerate(zip(axes, solutions, titles)):
        # Generate colors for routes
        num_routes = len(solution.routes)
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_routes, 3)))
        
        # Plot routes
        for route_idx, route in enumerate(solution.routes):
            color = colors[route_idx % len(colors)]
            
            # Find depot for this route (route.depot is 1-based depot number)
            if route.depot - 1 >= len(instance.depots) or route.depot < 1:
                continue
            depot = instance.depots[route.depot - 1]
            
            route_points_x = [depot.x]
            route_points_y = [depot.y]
            
            for customer_idx in route.nodes:
                customer = next((c for c in instance.customers if c.index == customer_idx), None)
                if customer:
                    route_points_x.append(customer.x)
                    route_points_y.append(customer.y)
            
            route_points_x.append(depot.x)
            route_points_y.append(depot.y)
            
            ax.plot(route_points_x, route_points_y, color=color, linewidth=1.5, alpha=0.6, zorder=1)
            
            for i in range(len(route_points_x) - 1):
                ax.annotate(
                    "",
                    xy=(route_points_x[i + 1], route_points_y[i + 1]),
                    xytext=(route_points_x[i], route_points_y[i]),
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.5, lw=1),
                    zorder=2
                )
        
        # Plot customers
        customer_x = [c.x for c in instance.customers]
        customer_y = [c.y for c in instance.customers]
        ax.scatter(customer_x, customer_y, c="blue", s=80, marker="o", zorder=3)
        
        # Plot depots
        depot_x = [d.x for d in instance.depots]
        depot_y = [d.y for d in instance.depots]
        ax.scatter(depot_x, depot_y, c="red", s=250, marker="*", zorder=4)
        
        # Add labels if requested
        if show_labels:
            for customer in instance.customers:
                ax.annotate(str(customer.index), xy=(customer.x, customer.y), xytext=(2, 2),
                           textcoords="offset points", fontsize=6, alpha=0.6)
        
        ax.set_xlabel("X Coordinate", fontsize=10)
        ax.set_ylabel("Y Coordinate", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    
    plt.tight_layout()
    plt.show()
