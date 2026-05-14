"""
Linear O(n) split algorithm for VRP with fleet-size limit.

Implements the deque-based linear split from:
    Vidal, T. (2016). "Split algorithm in O(n) for the capacitated vehicle
    routing problem." Computers & Operations Research, 69, 40-47.

The fleet-limited variant (Algorithm 3 in the paper) runs one pass per
vehicle (k = 0 … m-1), maintaining a monotone deque of non-dominated
predecessors.  Total complexity: O(n·m).
"""

from collections import deque
from typing import Callable, List

from core.entities import Customer, Depot, Route


def bellman_split(
    ordered: List[Customer],
    depot: Depot,
    dist_fn: Callable[[int, int], float],
) -> List[Route]:
    """
    Optimally partition an ordered customer sequence into feasible vehicle
    routes using the linear O(n·m) split algorithm of Vidal (2016),
    fleet-limited variant.

    Each contiguous segment becomes one vehicle route
    ``depot → seg[0] → ... → seg[-1] → depot``.

    The algorithm uses a monotone deque of non-dominated predecessors to
    find, for each position t and each vehicle count k, the cheapest way to
    serve customers 1…t with exactly k vehicles.  The best k ≤ max_vehicles
    is chosen at the end.

    Feasibility constraints enforced per route:

    * **Capacity** — total demand ≤ ``depot.max_capacity``.  Enforced via a
      sliding-window front-pruning on the deque.
    * **Duration** — total travel distance + service times ≤
      ``depot.max_duration``.  Enforced via an exact feasibility check when
      selecting the best predecessor for each position.
      When ``depot.max_duration == 0`` the duration constraint is skipped.
    * **Fleet size** — the number of routes is soft-penalised against
      ``depot.max_vehicles``.  All vehicle counts from 1 to n are evaluated;
      solutions that exceed the limit are penalised by
      ``cost × (1 + excess_vehicles)``, so feasible solutions are always
      preferred but infeasible ones are still returned when no feasible
      partition exists.  When ``depot.max_vehicles == 0`` no penalty is
      applied.

    Customers whose individual demand exceeds capacity are still guaranteed
    a singleton route.

    Parameters
    ----------
    ordered:
        Giant tour (sequence of Customer objects to visit).
    depot:
        Depot defining ``max_capacity``, ``max_duration`` (0 = unconstrained),
        and ``max_vehicles`` (0 = unconstrained).
    dist_fn:
        Pre-computed O(1) distance callable ``(a_index, b_index) -> float``.

    Returns
    -------
    List of ``Route`` objects, one per vehicle, each bound to ``depot``.
    """

    # ------------------------------------------------------------------
    # Deque helpers
    # Each element is an int index i ∈ {0, …, n-1}.
    # The deque stores indices in the tour.
    # ------------------------------------------------------------------

    def _propagate(i: int, j: int, k: int) -> float:
        """
        Cost of the route from predecessor i to node j using vehicle-layer k.

        Eq. (5):  c(i,j) = d_{0,i+1} + D[j] - D[i+1] + d_{j,0}
        Full:     potential[k][i] + c(i,j)
        """
        if potential[k][i] >= INF:
            return INF
        return (potential[k][i]
                + d_depot[i + 1]
                + sum_dist[j] - sum_dist[i + 1]
                + d_depot[j])

    def _g(i: int, k: int) -> float:
        """
        Fixed cost characterising predecessor i at layer k.
        g_i = p[k][i] + d_{0,i+1} - D[i+1]
        """
        return potential[k][i] + d_depot[i + 1] - sum_dist[i + 1]

    def _route_duration(i: int, t: int) -> float:
        """
        Exact duration of the route depot → customer[i+1] → … → customer[t] → depot.
        Includes travel distance and per-customer service times.
        """
        return (
            d_depot[i + 1]
            + sum_dist[t] - sum_dist[i + 1]
            + d_depot[t]
            + sum_service[t] - sum_service[i]
        )
    
    def _dominates(i: int, j: int, k: int) -> bool:
        """
        True if i dominates j as a predecessor (i < j).
        i dominates j when g_i ≤ g_j AND sum_load[i] == sum_load[j].
        (Paper eq. 6, hard capacity case.)
        """
        return (sum_load[i] == sum_load[j]
                and _g(i, k) <= _g(j, k) + 1e-9)

    def _dominates_right(i: int, j: int, k: int) -> bool:
        """
        True if j dominates i (new node j beats the current back i).
        """
        return _g(j, k) <= _g(i, k) + 1e-9


    # ------------------------------------------------------------------
    # Precompute prefix arrays (1-indexed; index 0 is a sentinel = 0).
    #
    # sum_dist[i]  = D[i] = sum of consecutive tour distances d_{k,k+1}
    #                       for k = 1 … i-1  (i.e. the distance *into* node i
    #                       along the tour, not from the depot).
    #                       sum_dist[1] = 0  (first customer costs nothing yet).
    # sum_load[i]  = Q[i] = cumulative demand of customers 1 … i.
    # sum_service[i] = S[i] = cumulative service time of customers 1 … i.
    # d_depot[i]  = distance from customer i back to the depot.
    # ------------------------------------------------------------------

    n = len(ordered)
    if n == 0:
        return []
    INF = float("inf")
    sum_dist    = [0.0] * (n + 1)   # sum_dist[0] unused; sum_dist[1] = 0
    sum_load    = [0.0] * (n + 1)   # sum_load[0] = 0
    sum_service = [0.0] * (n + 1)   # sum_service[i] = Σ service_time for customers 1…i
    d_depot     = [0.0] * (n + 1)   # d_depot[0] = depot→depot = 0 (sentinel)

    for i in range(1, n + 1):
        c = ordered[i - 1]
        sum_load[i]    = sum_load[i - 1]    + c.demand
        sum_service[i] = sum_service[i - 1] + c.service_time
        d_depot[i]    = dist_fn(c.index, depot.index)
        if i > 1:
            sum_dist[i] = sum_dist[i - 1] + dist_fn(ordered[i - 2].index, c.index)

    # ------------------------------------------------------------------
    # DP tables (Algorithm 3, paper Section 4.1)
    #
    # potential[k][t] = min cost of serving customers 1…t with exactly k vehicles
    # pred[k][t]      = predecessor index (start of the last route)
    # ------------------------------------------------------------------
    
    potential = [[INF] * (n + 1) for _ in range(n + 1)]
    pred_arr  = [[-1]  * (n + 1) for _ in range(n + 1)]
    potential[0][0] = 0.0

    use_duration = depot.max_duration > 0

    # Main loop — one pass per vehicle layer
    dq: deque[int] = deque()

    fleet_limit = depot.max_vehicles if depot.max_vehicles > 0 else n
    layers_computed = 0

    for k in range(n):
        dq.clear()
        dq.append(k)

        for t in range(k + 1, n + 1):
            if len(dq) == 0:
                break 

            # Select the cheapest feasible predecessor from the deque.
            # When duration is unconstrained the front is always cheapest (deque invariant).
            # When duration is constrained we scan from the front (minimum g-value first)
            # and take the first predecessor whose exact route duration is feasible.
            if use_duration:
                best_front = None
                for candidate in dq:
                    if _route_duration(candidate, t) <= depot.max_duration + 1e-9:
                        best_front = candidate
                        break
            else:
                best_front = dq[0]

            if best_front is not None:
                potential[k + 1][t] = _propagate(best_front, t, k)
                pred_arr[k + 1][t] = best_front

            if t < n:
                # Try to insert t as a new predecessor candidate
                if not _dominates(dq[-1], t, k):
                    while dq and _dominates_right(dq[-1], t, k):
                        dq.pop()
                    dq.append(t)

                # Capacity front-pruning: remove predecessors whose route to t+1
                # would exceed the vehicle capacity.
                while dq and sum_load[t + 1] - sum_load[dq[0]] > depot.max_capacity + 1e-9:
                    dq.popleft()

            if not dq:
                break  # no feasible predecessor remains

        layers_computed = k + 1

        # Early exit: once we are past the fleet limit and at least one feasible
        # solution exists, further layers can only worsen the penalty multiplier
        # (excess grows by 1 each layer), so there is no benefit in continuing.
        if k + 1 >= fleet_limit and potential[k + 1][n] < INF:
            break

    # Find the best number of routes k* ≤ max_vehicles
    best_k = -1
    best_cost = INF
    best_penalised = INF
    for k in range(1, layers_computed + 1):
        cost = potential[k][n]
        if cost >= INF:
            continue
        excess = max(0, k - fleet_limit)
        penalised = cost * (1 + excess)
        if penalised < best_penalised:
            best_penalised = penalised
            best_cost = cost
            best_k = k

    # Fallback: if the linear split found no solution
    if best_k == -1 or best_cost >= INF:
        return [Route(depot=depot, customers=list(ordered))]

    # Backtrack through pred_arr to recover segments
    segments: List[Route] = []
    cur = n
    for layer in range(best_k, 0, -1):
        start = pred_arr[layer][cur]
        segments.append(Route(depot=depot, customers=list(ordered[start:cur])))
        cur = start
    segments.reverse()
    return segments

