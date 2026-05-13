"""
Bellman (DAG shortest-path) split algorithm for VRP.

Provides the core dynamic-programming routine that optimally partitions an
ordered customer sequence (giant tour) into capacity- and duration-feasible
vehicle routes.
"""

from typing import Callable, List

from core.entities import Customer, Depot


def bellman_split(
    ordered: List[Customer],
    depot: Depot,
    dist_fn: Callable[[int, int], float],
) -> List[List[Customer]]:
    """
    Optimally partition an ordered customer sequence into capacity- and
    duration-feasible vehicle routes using the Bellman (DAG shortest-path)
    split algorithm.

    Each contiguous segment becomes one vehicle route
    ``depot → seg[0] → ... → seg[-1] → depot``.  The DP finds the cut points
    that minimise total travel distance subject to each segment's total demand
    not exceeding ``depot.max_capacity`` and total duration (travel + service
    times) not exceeding ``depot.max_duration`` (when non-zero).

    Customers whose individual demand or duration exceeds the limits are placed
    in their own route so that all customers are always routed.
    """
    n = len(ordered)
    INF = float("inf")
    dp = [INF] * (n + 1)
    pred = [-1] * (n + 1)
    dp[0] = 0.0

    for i in range(n):
        if dp[i] == INF:
            continue
        load = 0.0
        service = 0.0
        prev_idx = depot.index
        travel = 0.0
        for j in range(i, n):
            load += ordered[j].demand
            service += ordered[j].service_time
            # Allow singleton segments even when constraints are exceeded.
            if load > depot.max_capacity and j > i:
                break
            travel += dist_fn(prev_idx, ordered[j].index)
            prev_idx = ordered[j].index
            route_dist = travel + dist_fn(ordered[j].index, depot.index)
            if depot.max_duration > 0 and route_dist + service > depot.max_duration and j > i:
                break
            total = dp[i] + route_dist
            if total < dp[j + 1]:
                dp[j + 1] = total
                pred[j + 1] = i

    # Backtrack from n to 0 to recover segments.
    segments: List[List[Customer]] = []
    j = n
    while j > 0:
        i = pred[j]
        segments.append(list(ordered[i:j]))
        j = i
    segments.reverse()
    return segments
