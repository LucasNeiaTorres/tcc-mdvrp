"""
Linear O(n·m) split algorithm for VRP with soft capacity, soft duration,
and hard fleet limit.

Implements the deque-based split from:
    Vidal, T. (2016). "Split algorithm in O(n) for the capacitated vehicle
    routing problem." Computers & Operations Research, 69, 40-47.

Both capacity and forward-duration are treated as *soft* constraints
(Section 4.2): exceeding a limit incurs a linear penalty, so every arc
(i, j) is admissible.  The fleet size is a *hard* constraint: at most
max_vehicles routes are ever produced.

The fleet-limited variant (Algorithm 3) runs one pass per vehicle layer
(k = 0 … m-1), maintaining a monotone deque of non-dominated predecessors.
Total complexity: O(n·m).
"""

from collections import deque
from typing import Callable, List

from core.entities import Customer, Depot, Route


def linear_split(
    ordered: List[Customer],
    depot: Depot,
    dist_fn: Callable[[int, int], float],
    capacity_penalty: float,
    duration_penalty: float,
) -> List[Route]:
    """
    Partition an ordered customer sequence into vehicle routes using the
    linear O(n·m) split algorithm of Vidal (2016), with soft capacity
    constraints (Section 4.2) and a hard fleet-size limit.

    Each contiguous segment becomes one vehicle route
    ``depot → seg[0] → ... → seg[-1] → depot``.

    The algorithm uses a monotone deque of non-dominated predecessors to
    find, for each position t and each vehicle count k, the cheapest way to
    serve customers 1…t with exactly k vehicles.  The best k ≤ max_vehicles
    is chosen at the end.

    Constraints:

    * **Capacity** — *soft* constraint.  Exceeding ``depot.max_capacity`` by
      Δ demand units on a route adds a penalty of ``capacity_penalty × Δ`` to that
      route's cost (Paper eq. 8).  Every arc is therefore admissible and the
      algorithm always produces a complete solution.
    * **Duration** — *soft* constraint.  The forward route duration
      (depot → customers, excluding the return leg) exceeding
      ``depot.max_duration`` by Δ adds a penalty of ``duration_penalty × Δ`` to that
      route's cost.  Treated as monotone increasing in the number of
      customers served, analogous to capacity.  No penalty when
      ``depot.max_duration == 0``.
    * **Fleet size** — *hard* constraint.  The number of routes produced is
      always ≤ ``depot.max_vehicles`` (or ≤ n when max_vehicles == 0).
      The cheapest solution within that budget is returned.

    Parameters
    ----------
    ordered:
        Giant tour (sequence of Customer objects to visit).
    depot:
        Depot defining ``max_capacity``, ``max_duration`` (0 = unconstrained),
        and ``max_vehicles`` (0 = unconstrained).
    dist_fn:
        Pre-computed O(1) distance callable ``(a_index, b_index) -> float``.
    capacity_penalty:
        Penalty coefficient per unit of excess capacity demand on a route.
        Higher values push the solver toward capacity-feasible routes.
    duration_penalty:
        Penalty coefficient per unit of excess forward duration on a route.
        Higher values push the solver toward duration-feasible routes.

    Returns
    -------
    List of ``Route`` objects, one per vehicle, each bound to ``depot``.
    """

    def _propagate(i: int, j: int, k: int) -> float:
        """
        Total cost of assigning customers (i, j] to vehicle k+1.

        Returns potential[k][i] + travel_cost(i, j) + _arc_penalty(i, j),
        or INF if predecessor i is unreachable.
        """
        if potential[k][i] >= INF:
            return INF
        travel = (
            d_depot[i + 1]
            + sum_dist[j] - sum_dist[i + 1]
            + d_depot[j]
        )
        return potential[k][i] + travel + _arc_penalty(i, j)

    def _g(i: int, k: int) -> float:
        """
        Fixed cost characterising predecessor i at layer k.
        g_i = p[k][i] + d_{0,i+1} - D[i+1]
        """
        return potential[k][i] + d_depot[i + 1] - sum_dist[i + 1]

    def _dominates(i: int, j: int, k: int) -> bool:
        """
        True if i dominates j as a predecessor (i < j).

        Sufficient condition for g_i(x) ≤ g_j(x) for all x, combining both
        soft capacity and soft duration penalties:
          h_i + capacity_penalty*(Q[j]-Q[i]) + duration_penalty*max(0, fo_i-fo_j) ≤ h_j
        where fo_i = fwd_offset[i] = d_{0,i+1} - D[i+1] - S[i].
        When fo_i ≤ fo_j (i's duration-free zone is wider), the duration_penalty term
        vanishes and the condition reduces to the capacity-only form.
        """
        fo_diff = (fwd_offset[i] - fwd_offset[j]) if use_duration else 0.0
        return (
            _g(i, k)
            + capacity_penalty * (sum_load[j] - sum_load[i])
            + duration_penalty * max(0.0, fo_diff)
            <= _g(j, k) + 1e-9
        )

    def _dominates_right(i: int, j: int, k: int) -> bool:
        """
        True if new node j dominates the current back i (j > i).

        Sufficient condition for g_j(x) ≤ g_i(x) for all x:
          h_j + β*max(0, fo_j-fo_i) ≤ h_i
        The capacity term vanishes because Q[j] ≥ Q[i] (j pays at least as
        much capacity penalty as i), so only the duration offset matters.
        """
        fo_diff = (fwd_offset[j] - fwd_offset[i]) if use_duration else 0.0
        return (
            _g(j, k) + duration_penalty * max(0.0, fo_diff)
            <= _g(i, k) + 1e-9
        )
    
    def _arc_penalty(i: int, j: int) -> float:
        """Penalty-only portion of the arc (i, j) cost."""
        cap_excess = max(0.0, sum_load[j] - sum_load[i] - depot.max_capacity)
        p = capacity_penalty * cap_excess
        if use_duration:
            full_dur = fwd_offset[i] + sum_dur_fwd[j] + d_depot[j]
            dur_excess = max(0.0, full_dur - depot.max_duration)
            p += duration_penalty * dur_excess
        return p


    n = len(ordered)
    if n == 0:
        return []
    m = depot.max_vehicles if depot.max_vehicles > 0 else n
    INF = float("inf")

    # Precompute prefix arrays (1-indexed; index 0 is a sentinel = 0).
    sum_dist    = [0.0] * (n + 1)   # D[i] = sum of consecutive tour distances d_{k,k+1} for k = 1 … i-1.
    sum_load    = [0.0] * (n + 1)   # Q[i] = cumulative demand of customers 1 … i.
    sum_service = [0.0] * (n + 1)   # S[i] = cumulative service time of customers 1 … i.
    sum_dur_fwd = [0.0] * (n + 1)   # D[i] + S[i], used to compute forward duration.
    d_depot     = [0.0] * (n + 1)   # distance from customer i back to the depot.

    # fwd_offset[i] = d_depot[i+1] - D[i+1] - S[i]
    # Forward duration from predecessor i to node j: fwd_offset[i] + sum_dur_fwd[j]
    # = d_{0,i+1} + (D[j]-D[i+1]) + (S[j]-S[i])  — monotone increasing in j.
    fwd_offset  = [0.0] * (n + 1)

    for i in range(1, n + 1):
        c = ordered[i - 1]
        sum_load[i]    = sum_load[i - 1]    + c.demand
        sum_service[i] = sum_service[i - 1] + c.service_time
        d_depot[i]    = dist_fn(c.index, depot.index)
        if i > 1:
            sum_dist[i] = sum_dist[i - 1] + dist_fn(ordered[i - 2].index, c.index)
        sum_dur_fwd[i] = sum_dist[i] + sum_service[i]

    for i in range(n):  # predecessor indices 0 … n-1
        fwd_offset[i] = d_depot[i + 1] - sum_dist[i + 1] - sum_service[i]

    # DP tables
    potential = [[INF] * (n + 1) for _ in range(n + 1)] # potential[k][t] = min penalised cost of serving customers 1…t with k vehicles
    penalty   = [[INF] * (n + 1) for _ in range(n + 1)] # penalty[k][t]   = accumulated soft-constraint penalty portion of potential[k][t]
    pred_arr  = [[-1]  * (n + 1) for _ in range(n + 1)] # pred[k][t]      = predecessor index (start of the last route)
    potential[0][0] = 0.0
    penalty[0][0]   = 0.0

    use_duration = depot.max_duration > 0

    # Main loop — one pass per vehicle layer
    dq: deque[int] = deque()

    # Hard fleet limit: run at most m vehicle layers.
    for k in range(m):
        dq.clear()
        dq.append(k)

        for t in range(k + 1, n + 1):
            potential[k + 1][t] = _propagate(dq[0], t, k)
            penalty[k + 1][t]   = penalty[k][dq[0]] + _arc_penalty(dq[0], t)
            pred_arr[k + 1][t]  = dq[0]

            if t < n:
                # Try to insert t as a new predecessor candidate.
                if not _dominates(dq[-1], t, k):
                    while dq and _dominates_right(dq[-1], t, k):
                        dq.pop()
                    dq.append(t)

                # Soft capacity + soft duration front-pruning:
                # remove the front when the second element already yields a
                # lower or equal cost to t+1, keeping at least one element.
                while (
                    len(dq) > 1
                    and _propagate(dq[0], t + 1, k)
                    >= _propagate(dq[1], t + 1, k) - 1e-9
                ):
                    dq.popleft()


    # Find the best k ≤ fleet_limit:
    # This prevents the DP from choosing fewer vehicles just because soft
    # penalties are cheaper than the travel cost of an extra depot round-trip.
    best_k = -1
    best_cost = INF

    # Pass 1: feasible solutions only
    for k in range(1, m + 1):
        if penalty[k][n] < 1e-9 and potential[k][n] < best_cost:
            best_cost = potential[k][n]
            best_k = k

    # Pass 2: fallback to minimum penalised cost
    if best_k == -1:
        for k in range(1, m + 1):
            if potential[k][n] < best_cost:
                best_cost = potential[k][n]
                best_k = k

    # Backtrack through pred_arr to recover segments
    segments: List[Route] = []
    cur = n
    for layer in range(best_k, 0, -1):
        start = pred_arr[layer][cur]
        segments.append(Route(depot=depot, customers=list(ordered[start:cur])))
        cur = start
    segments.reverse()
    return segments

