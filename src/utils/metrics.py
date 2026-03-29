"""Utility functions for metric calculations in the MDVRP problem."""

import math
from typing import List, Protocol


class _HasCoordinates(Protocol):
    x: float
    y: float


def euclidean_distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def build_distance_matrix(
    depots: List[_HasCoordinates],
    customers: List[_HasCoordinates],
) -> List[List[float]]:
    """
    Pre-compute a full distance matrix between all nodes (depots + customers).

    Nodes are indexed as follows:
      - 0 to len(depots)-1         → depots (by their position in the list)
      - len(depots) to n+d-1       → customers (by their position in the list)

    This allows O(1) distance lookups during algorithm execution instead of
    recomputing euclidean distance on every call.

    Args:
        depots: List of Depot entities
        customers: List of Customer entities

    Returns:
        A 2D matrix where matrix[i][j] is the distance between node i and node j
    """
    nodes = [*depots, *customers]
    n = len(nodes)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y)
            matrix[i][j] = dist
            matrix[j][i] = dist

    return matrix
