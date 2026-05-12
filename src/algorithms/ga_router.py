"""
GA-based routing module for MDVRP.

Solves the route-optimisation sub-problem: given a depot and a fixed set of
customers already assigned to it, find both the visiting order and the vehicle
partition that minimise total travel distance.

SPV encoding (Smallest Position Value)
---------------------------------------
A permutation is obtained by ranking (argsort) the real-valued position vector:

    x = [0.72, 0.11, 0.55]  →  argsort → [1, 2, 0]  →  visit C2, C3, C1

Bellman split
-------------
For each candidate permutation the fitness is computed by the Bellman (DAG
shortest-path) split algorithm (Prins, 2004).  The GA therefore co-optimises
both the visiting order and the vehicle boundaries.

Fitness
-------
    f(x) = min-cost partition of the giant tour into capacity-feasible routes
           = Σ_k [ dist(depot, r_k[0])
                   + Σ dist(r_k[i], r_k[i+1])
                   + dist(r_k[-1], depot) ]
"""

from typing import Callable, List

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from core.entities import Customer, Depot, Route
from utils.config import GAConfig


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


class RoutingProblem(ElementwiseProblem):
    """
    SPV-encoded route optimisation problem for pymoo GA.
    Solves standard VRP: depot -> customers -> depot.

    Parameters
    ----------
    depot:
        The depot that starts and ends the route.
    customers:
        Ordered list of Customer entities to visit (order will be optimised).
    dist_fn:
        Callable ``(a_index, b_index) -> float`` returning pre-computed
        distance between any two node indices.
    """

    def __init__(
        self,
        depot: Depot,
        customers: List[Customer],
        dist_fn: Callable[[int, int], float],
    ) -> None:
        super().__init__(
            n_var=len(customers),
            n_obj=1,
            xl=0.0,
            xu=1.0,
        )
        self.depot = depot
        self.customers = customers
        self.dist_fn = dist_fn
        self.start_node = depot
        self.end_depot = depot

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        perm = np.argsort(x)
        ordered = [self.customers[i] for i in perm]
        segments = bellman_split(ordered, self.end_depot, self.dist_fn)

        total = 0.0
        for seg in segments:
            total += self.dist_fn(self.start_node.index, seg[0].index)
            for i in range(len(seg) - 1):
                total += self.dist_fn(seg[i].index, seg[i + 1].index)
            total += self.dist_fn(seg[-1].index, self.end_depot.index)

        excess = max(0, len(segments) - self.depot.max_vehicles)
        out["F"] = total + excess * total


class DynamicRoutingProblem(ElementwiseProblem):
    """
    SPV-encoded route optimisation for VRP-OD (origin-destination).
    Solves dynamic reroute: current_node -> customers -> real_depot.

    Parameters
    ----------
    current_start_node:
        Current position of vehicle (Customer or Depot), NOT the original depot.
    pending_customers:
        Customers still to be served.
    real_end_depot:
        The original depot where the route must end.
    dist_fn:
        Callable ``(a_index, b_index) -> float`` returning pre-computed distance.
    """

    def __init__(
        self,
        current_start_node: Customer | Depot,
        pending_customers: List[Customer],
        real_end_depot: Depot,
        dist_fn: Callable[[int, int], float],
    ) -> None:
        super().__init__(
            n_var=len(pending_customers),
            n_obj=1,
            xl=0.0,
            xu=1.0,
        )
        self.current_start_node = current_start_node
        self.pending_customers = pending_customers
        self.real_end_depot = real_end_depot
        self.dist_fn = dist_fn

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evaluate cost of a route from current_start_node -> customers -> real_end_depot."""
        if len(self.pending_customers) == 0:
            # No customers: just return to depot
            total = self.dist_fn(self.current_start_node.index, self.real_end_depot.index)
            out["F"] = total
            return

        perm = np.argsort(x)
        ordered = [self.pending_customers[i] for i in perm]

        total = 0.0
        # 1. From current location to first customer
        total += self.dist_fn(self.current_start_node.index, ordered[0].index)
        # 2. Between customers
        for i in range(len(ordered) - 1):
            total += self.dist_fn(ordered[i].index, ordered[i + 1].index)
        # 3. From last customer to real depot (NOT back to current node)
        total += self.dist_fn(ordered[-1].index, self.real_end_depot.index)

        out["F"] = total


def _route_cost(route: List[Customer], depot: Depot, dist_fn: Callable[[int, int], float]) -> float:
    """Total round-trip cost: depot → route[0] → ... → route[-1] → depot."""
    if not route:
        return 0.0
    cost = dist_fn(depot.index, route[0].index)
    for i in range(len(route) - 1):
        cost += dist_fn(route[i].index, route[i + 1].index)
    cost += dist_fn(route[-1].index, depot.index)
    return cost


def _is_route_feasible(route: List[Customer], depot: Depot, dist_fn: Callable[[int, int], float]) -> bool:
    """Check capacity and (if constrained) duration feasibility."""
    if sum(c.demand for c in route) > depot.max_capacity:
        return False
    if depot.max_duration > 0:
        travel = _route_cost(route, depot, dist_fn)
        service = sum(c.service_time for c in route)
        if travel + service > depot.max_duration:
            return False
    return True


def local_search(
    routes: List[List[Customer]],
    depot: Depot,
    dist_fn: Callable[[int, int], float],
) -> List[List[Customer]]:
    """
    Prins (2004) 9-move local search over a multi-route VRP solution.

    Operates on mutable lists of customers (routes).  The depot is implicit
    at the start and end of every route.  Scans all O(n²) (u, v) customer
    pairs and applies the first improving move found, then restarts.  Moves
    M1–M6 are inter/intra-route relocate/swap moves; M7 is intra-route 2-opt;
    M8–M9 are inter-route 2-opt variants.

    Empty routes are removed at the end.

    Parameters
    ----------
    routes:
        Mutable list-of-lists; modified in-place but also returned for
        convenience.
    depot:
        Shared depot for all routes (capacity / duration limits).
    dist_fn:
        O(1) pre-computed distance callable.

    Returns
    -------
    Cleaned list of non-empty routes.
    """
    # Work on copies so callers keep the originals until committed.
    routes = [list(r) for r in routes if r]

    def _prev(route: List[Customer], pos: int) -> int:
        """Index of node before route[pos]; returns depot.index if pos == 0."""
        return depot.index if pos == 0 else route[pos - 1].index

    def _next(route: List[Customer], pos: int) -> int:
        """Index of node after route[pos]; returns depot.index if pos == last."""
        return depot.index if pos == len(route) - 1 else route[pos + 1].index

    improved = True
    while improved:
        improved = False
        n_routes = len(routes)
        d = dist_fn  # alias for brevity

        for ru_idx in range(n_routes):
            ru = routes[ru_idx]
            if improved:
                break
            for rv_idx in range(n_routes):
                rv = routes[rv_idx]
                if improved:
                    break
                for u_idx in range(len(ru)):
                    if improved:
                        break
                    u = ru[u_idx]
                    x = ru[u_idx + 1] if u_idx + 1 < len(ru) else depot

                    for v_idx in range(len(rv)):
                        if improved:
                            break
                        if ru_idx == rv_idx and u_idx == v_idx:
                            continue
                        v = rv[v_idx]
                        y = rv[v_idx + 1] if v_idx + 1 < len(rv) else depot

                        # M1: relocate u after v
                        if not (ru_idx == rv_idx and v_idx == u_idx - 1):
                            u_prev = _prev(ru, u_idx)
                            u_next = _next(ru, u_idx)
                            v_next = _next(rv, v_idx)

                            gain = (
                                d(u_prev, u.index) + d(u.index, u_next) + d(v.index, v_next)
                                - d(u_prev, u_next) - d(v.index, u.index) - d(u.index, v_next)
                            )
                            if gain > 1e-9:
                                if ru_idx == rv_idx:
                                    # Same route: remove then re-insert
                                    new_r = [c for k, c in enumerate(ru) if c is not u]
                                    insert_pos = next(k for k, c in enumerate(new_r) if c is v) + 1
                                    new_r.insert(insert_pos, u)
                                    if _is_route_feasible(new_r, depot, dist_fn):
                                        routes[ru_idx] = new_r
                                        improved = True
                                        break
                                else:
                                    new_ru = [c for k, c in enumerate(routes[ru_idx]) if k != u_idx]
                                    new_rv = list(routes[rv_idx])
                                    new_rv.insert(v_idx + 1, u)
                                    if (_is_route_feasible(new_ru, depot, dist_fn) and
                                            _is_route_feasible(new_rv, depot, dist_fn)):
                                        routes[ru_idx] = new_ru
                                        routes[rv_idx] = new_rv
                                        improved = True
                                        break

                        if improved:
                            break

                        # M2: relocate (u, x) after v
                        # Skip no-op same-route cases:
                        #   v_idx == u_idx - 1: v is directly before u, reinserting (u,x) after v is a no-op
                        #   v_idx == u_idx + 1: v IS x; after removing u and x, v is gone from new_r
                        _m2_noop = ru_idx == rv_idx and v_idx in (u_idx - 1, u_idx + 1)
                        if isinstance(x, Customer) and not _m2_noop:
                            u_prev = _prev(ru, u_idx)
                            x_next = _next(ru, u_idx + 1)
                            v_next = _next(rv, v_idx)

                            gain = (
                                d(u_prev, u.index) + d(x.index, x_next) + d(v.index, v_next)
                                - d(u_prev, x_next) - d(v.index, u.index) - d(x.index, v_next)
                            )
                            if gain > 1e-9:
                                if ru_idx == rv_idx:
                                    new_r = [c for k, c in enumerate(ru) if k not in (u_idx, u_idx + 1)]
                                    insert_pos = next(k for k, c in enumerate(new_r) if c is v) + 1
                                    new_r.insert(insert_pos, x)
                                    new_r.insert(insert_pos, u)
                                    if _is_route_feasible(new_r, depot, dist_fn):
                                        routes[ru_idx] = new_r
                                        improved = True
                                        break
                                else:
                                    new_ru = [c for k, c in enumerate(ru) if k not in (u_idx, u_idx + 1)]
                                    new_rv = list(rv)
                                    new_rv.insert(v_idx + 1, x)
                                    new_rv.insert(v_idx + 1, u)
                                    if (_is_route_feasible(new_ru, depot, dist_fn) and
                                            _is_route_feasible(new_rv, depot, dist_fn)):
                                        routes[ru_idx] = new_ru
                                        routes[rv_idx] = new_rv
                                        improved = True
                                        break

                        if improved:
                            break

                        # M3: relocate (x, u) after v
                        # Same no-op guards as M2:
                        #   v_idx == u_idx - 1: v is directly before u, reinserting (x,u) after v is a no-op
                        #   v_idx == u_idx + 1: v IS x; after removing u and x, v is gone from new_r
                        _m3_noop = ru_idx == rv_idx and v_idx in (u_idx - 1, u_idx + 1)
                        if isinstance(x, Customer) and not _m3_noop:
                            u_prev = _prev(ru, u_idx)
                            x_next = _next(ru, u_idx + 1)
                            v_next = _next(rv, v_idx)

                            gain = (
                                d(u_prev, u.index) + d(x.index, x_next) + d(v.index, v_next)
                                - d(u_prev, x_next) - d(v.index, x.index) - d(u.index, v_next)
                            )
                            if gain > 1e-9:
                                if ru_idx == rv_idx:
                                    new_r = [c for k, c in enumerate(ru) if k not in (u_idx, u_idx + 1)]
                                    insert_pos = next(k for k, c in enumerate(new_r) if c is v) + 1
                                    new_r.insert(insert_pos, u)  # insert u first (will be after x)
                                    new_r.insert(insert_pos, x)  # insert x at same pos → [v, x, u, ...]
                                    if _is_route_feasible(new_r, depot, dist_fn):
                                        routes[ru_idx] = new_r
                                        improved = True
                                        break
                                else:
                                    new_ru = [c for k, c in enumerate(ru) if k not in (u_idx, u_idx + 1)]
                                    new_rv = list(rv)
                                    new_rv.insert(v_idx + 1, u)  # insert u first (will be after x)
                                    new_rv.insert(v_idx + 1, x)  # insert x at same pos → [v, x, u, ...]
                                    if (_is_route_feasible(new_ru, depot, dist_fn) and
                                            _is_route_feasible(new_rv, depot, dist_fn)):
                                        routes[ru_idx] = new_ru
                                        routes[rv_idx] = new_rv
                                        improved = True
                                        break

                        if improved:
                            break

                        # M4: swap u and v
                        if ru_idx == rv_idx and abs(u_idx - v_idx) == 1:
                            # Adjacent same-route: the shared edge u↔v cancels (symmetric distances).
                            # gain = d(p, lo) + d(hi, q) - d(p, hi) - d(lo, q)
                            lo, hi = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)
                            lo_c, hi_c = ru[lo], ru[hi]
                            p = _prev(ru, lo)
                            q = _next(ru, hi)
                            gain = (
                                d(p, lo_c.index) + d(hi_c.index, q)
                                - d(p, hi_c.index) - d(lo_c.index, q)
                            )
                            if gain > 1e-9:
                                new_r = list(ru)
                                new_r[lo] = hi_c
                                new_r[hi] = lo_c
                                if _is_route_feasible(new_r, depot, dist_fn):
                                    routes[ru_idx] = new_r
                                    improved = True
                                    break
                        else:
                            # Non-adjacent or inter-route: general 4-edge formula.
                            u_prev = _prev(ru, u_idx)
                            u_next = _next(ru, u_idx)
                            v_prev = _prev(rv, v_idx)
                            v_next = _next(rv, v_idx)

                            gain = (
                                d(u_prev, u.index) + d(u.index, u_next)
                                + d(v_prev, v.index) + d(v.index, v_next)
                                - d(u_prev, v.index) - d(v.index, u_next)
                                - d(v_prev, u.index) - d(u.index, v_next)
                            )
                            if gain > 1e-9:
                                if ru_idx == rv_idx:
                                    new_r = list(ru)
                                    new_r[u_idx] = v
                                    new_r[v_idx] = u
                                    if _is_route_feasible(new_r, depot, dist_fn):
                                        routes[ru_idx] = new_r
                                        improved = True
                                        break
                                else:
                                    new_ru = list(ru)
                                    new_rv = list(rv)
                                    new_ru[u_idx] = v
                                    new_rv[v_idx] = u
                                    if (_is_route_feasible(new_ru, depot, dist_fn) and
                                            _is_route_feasible(new_rv, depot, dist_fn)):
                                        routes[ru_idx] = new_ru
                                        routes[rv_idx] = new_rv
                                        improved = True
                                        break

                        if improved:
                            break

                        # M5: swap (u, x) with v
                        # v IS x (v_idx == u_idx+1 in same route) → no-op, skip.
                        # Two adjacent same-route cases need corrected gain formulas because
                        # boundary nodes coincide with x or u, breaking the general 8-term formula.
                        if isinstance(x, Customer) and not (ru_idx == rv_idx and v_idx == u_idx + 1):
                            u_prev = _prev(ru, u_idx)
                            x_next = _next(ru, u_idx + 1)
                            v_prev = _prev(rv, v_idx)
                            v_next = _next(rv, v_idx)

                            if ru_idx == rv_idx and v_idx == u_idx + 2:
                                # Route: [..., p, u, x, v, q, ...] → [..., p, v, u, x, q, ...]
                                # d(u,x) cancels; v_prev==x so general formula double-counts d(x,v).
                                p, q = u_prev, v_next
                                gain = (
                                    d(p, u.index) + d(x.index, v.index) + d(v.index, q)
                                    - d(p, v.index) - d(v.index, u.index) - d(x.index, q)
                                )
                            elif ru_idx == rv_idx and v_idx == u_idx - 1:
                                # Route: [..., p, v, u, x, q, ...] → [..., p, u, x, v, q, ...]
                                # d(u,x) cancels; v_next==u so general formula double-counts d(v,u).
                                p, q = v_prev, x_next
                                gain = (
                                    d(p, v.index) + d(v.index, u.index) + d(x.index, q)
                                    - d(p, u.index) - d(x.index, v.index) - d(v.index, q)
                                )
                            else:
                                # General case: all 6 boundary nodes are distinct.
                                gain = (
                                    d(u_prev, u.index) + d(x.index, x_next)
                                    + d(v_prev, v.index) + d(v.index, v_next)
                                    - d(u_prev, v.index) - d(v.index, x_next)
                                    - d(v_prev, u.index) - d(x.index, v_next)
                                )

                            if gain > 1e-9:
                                if ru_idx == rv_idx:
                                    if u_idx < v_idx:
                                        new_r = list(ru)
                                        new_r[u_idx] = v
                                        new_r.pop(u_idx + 1)   # remove x; v_idx shifts by -1
                                        adj_v = v_idx - 1
                                        new_r[adj_v] = u
                                        new_r.insert(adj_v + 1, x)
                                    else:
                                        new_r = list(ru)
                                        new_r[v_idx] = u
                                        new_r.insert(v_idx + 1, x)  # u_idx shifts by +1
                                        adj_u = u_idx + 1
                                        new_r[adj_u] = v
                                        new_r.pop(adj_u + 1)   # remove x
                                    if _is_route_feasible(new_r, depot, dist_fn):
                                        routes[ru_idx] = new_r
                                        improved = True
                                        break
                                else:
                                    new_ru = list(ru)
                                    new_ru[u_idx] = v
                                    new_ru.pop(u_idx + 1)
                                    new_rv = list(rv)
                                    new_rv[v_idx] = u
                                    new_rv.insert(v_idx + 1, x)
                                    if (_is_route_feasible(new_ru, depot, dist_fn) and
                                            _is_route_feasible(new_rv, depot, dist_fn)):
                                        routes[ru_idx] = new_ru
                                        routes[rv_idx] = new_rv
                                        improved = True
                                        break

                        if improved:
                            break

                        # M6: swap (u, x) with (v, y)
                        # Same-route pairs overlap when |v_idx - u_idx| <= 1:
                        #   v_idx == u_idx+1 → v IS x; v_idx == u_idx-1 → y IS u. Skip both.
                        # Adjacent non-overlapping pairs (distance == 2) share a boundary node
                        # with the other pair, breaking the general gain formula.
                        _m6_overlap = ru_idx == rv_idx and abs(v_idx - u_idx) <= 1
                        if isinstance(x, Customer) and isinstance(y, Customer) and not _m6_overlap:
                            u_prev = _prev(ru, u_idx)
                            x_next = _next(ru, u_idx + 1)
                            v_prev = _prev(rv, v_idx)
                            y_next = _next(rv, v_idx + 1)

                            if ru_idx == rv_idx and v_idx == u_idx + 2:
                                # [p, u, x, v, y, q] → [p, v, y, u, x, q]
                                # x_next==v and v_prev==x → d(x,v) double-counted
                                p, q = u_prev, y_next
                                gain = (
                                    d(p, u.index) + d(x.index, v.index) + d(y.index, q)
                                    - d(p, v.index) - d(y.index, u.index) - d(x.index, q)
                                )
                            elif ru_idx == rv_idx and v_idx == u_idx - 2:
                                # [p, v, y, u, x, q] → [p, u, x, v, y, q]
                                # u_prev==y and y_next==u → d(y,u) double-counted
                                p, q = v_prev, x_next
                                gain = (
                                    d(p, v.index) + d(y.index, u.index) + d(x.index, q)
                                    - d(p, u.index) - d(x.index, v.index) - d(y.index, q)
                                )
                            else:
                                # General case: all 8 boundary terms are distinct.
                                gain = (
                                    d(u_prev, u.index) + d(x.index, x_next)
                                    + d(v_prev, v.index) + d(y.index, y_next)
                                    - d(u_prev, v.index) - d(y.index, x_next)
                                    - d(v_prev, u.index) - d(x.index, y_next)
                                )

                            if gain > 1e-9:
                                if ru_idx == rv_idx:
                                    # Pairs don't overlap → 4 direct index assignments.
                                    new_r = list(ru)
                                    new_r[u_idx] = v
                                    new_r[u_idx + 1] = y
                                    new_r[v_idx] = u
                                    new_r[v_idx + 1] = x
                                    if _is_route_feasible(new_r, depot, dist_fn):
                                        routes[ru_idx] = new_r
                                        improved = True
                                        break
                                else:
                                    new_ru = list(ru)
                                    new_ru[u_idx] = v
                                    new_ru[u_idx + 1] = y
                                    new_rv = list(rv)
                                    new_rv[v_idx] = u
                                    new_rv[v_idx + 1] = x
                                    if (_is_route_feasible(new_ru, depot, dist_fn) and
                                            _is_route_feasible(new_rv, depot, dist_fn)):
                                        routes[ru_idx] = new_ru
                                        routes[rv_idx] = new_rv
                                        improved = True
                                        break

                        if improved:
                            break

                        # M7: 2-opt within same route
                        # Replace edges (u→x) and (v→y) with (u→v) and (x→y).
                        # Reverses segment route[u_idx+1 : v_idx+1].
                        # Guard: u_idx+1 < v_idx ensures non-degenerate (≥1 node reversed)
                        # and guarantees x = ru[u_idx+1] is a Customer (not depot).
                        if ru_idx == rv_idx and u_idx + 1 < v_idx:
                            gain = (
                                d(u.index, x.index) + d(v.index, y.index)
                                - d(u.index, v.index) - d(x.index, y.index)
                            )
                            if gain > 1e-9:
                                new_r = list(ru)
                                new_r[u_idx + 1: v_idx + 1] = new_r[u_idx + 1: v_idx + 1][::-1]
                                if _is_route_feasible(new_r, depot, dist_fn):
                                    routes[ru_idx] = new_r
                                    improved = True
                                    break

                        if improved:
                            break

                        # M8: inter-route 2-opt (reconnect)
                        # T(u) ≠ T(v): different routes only.
                        # Replace edges (u→x) and (v→y) with (u→v) and (x→y).
                        # new_ru = ru[:u+1] + reversed(rv[:v+1])
                        # new_rv = ru[u+1:] + rv[v+1:]
                        # Use actual route-cost delta to avoid gain-formula oscillation:
                        # the edge-delta formula is symmetric so scanning (v,u) after (u,v)
                        # reports the same positive gain and reverses the move endlessly.
                        if ru_idx < rv_idx:
                            prefix_u = ru[: u_idx + 1]
                            suffix_u = ru[u_idx + 1:]
                            prefix_v = rv[: v_idx + 1]
                            suffix_v = rv[v_idx + 1:]
                            new_ru = prefix_u + prefix_v[::-1]
                            new_rv = suffix_u + suffix_v
                            cost_before = _route_cost(ru, depot, dist_fn) + _route_cost(rv, depot, dist_fn)
                            cost_after = _route_cost(new_ru, depot, dist_fn) + _route_cost(new_rv, depot, dist_fn)
                            if cost_before - cost_after > 1e-9:
                                if (_is_route_feasible(new_ru, depot, dist_fn) and
                                        _is_route_feasible(new_rv, depot, dist_fn)):
                                    routes[ru_idx] = new_ru
                                    routes[rv_idx] = new_rv
                                    improved = True
                                    break

                        if improved:
                            break

                        # M9: inter-route tail-swap
                        # T(u) ≠ T(v): different routes only.
                        # Replace edges (u→x) and (v→y) with (u→y) and (v→x).
                        # new_ru = ru[:u+1] + rv[v+1:]  (u's prefix + v's suffix)
                        # new_rv = rv[:v+1] + ru[u+1:]  (v's prefix + u's suffix)
                        # Gain is exact (no reversals): d(u,x)+d(v,y)-d(u,y)-d(v,x).
                        # Reverse scan gives -gain < 0, so no oscillation; ru_idx < rv_idx
                        # just avoids computing each pair twice.
                        if ru_idx < rv_idx:
                            gain = (
                                d(u.index, x.index) + d(v.index, y.index)
                                - d(u.index, y.index) - d(v.index, x.index)
                            )
                            if gain > 1e-9:
                                new_ru = ru[: u_idx + 1] + rv[v_idx + 1:]
                                new_rv = rv[: v_idx + 1] + ru[u_idx + 1:]
                                if (_is_route_feasible(new_ru, depot, dist_fn) and
                                        _is_route_feasible(new_rv, depot, dist_fn)):
                                    routes[ru_idx] = new_ru
                                    routes[rv_idx] = new_rv
                                    improved = True
                                    break

    # Remove empty routes and return
    return [r for r in routes if r]


class LSMutation(Mutation):
    """
    Prins (2004) local-search mutation operator for pymoo GA.

    Applied with probability ``prob`` to each child chromosome after crossover.
    The chromosome (SPV real vector) is decoded to routes via bellman_split,
    improved by ``local_search``, then re-encoded back to SPV so pymoo can
    continue operating on it.

    Parameters
    ----------
    depot:
        The depot used for route evaluation.
    customers:
        All customers in the current cluster (defines the SPV index mapping).
    dist_fn:
        O(1) distance callable.
    prob:
        Per-individual mutation probability (passed to pymoo Mutation base).
    """

    def __init__(
        self,
        depot: Depot,
        customers: List[Customer],
        dist_fn: Callable[[int, int], float],
        prob: float,
    ) -> None:
        super().__init__(prob=prob)
        self.depot = depot
        self.customers = customers
        self.dist_fn = dist_fn
        # Map customer object → position in customers list for re-encoding
        self._customer_pos = {c: i for i, c in enumerate(customers)}

    def _do(self, problem, X: np.ndarray, **kwargs) -> np.ndarray:
        X = X.copy()
        n = len(self.customers)
        if n <= 1:
            return X

        rng = np.random.default_rng()
        for k in range(len(X)):
            if rng.random() >= self.prob.value:
                continue

            perm = np.argsort(X[k])
            ordered = [self.customers[i] for i in perm]
            segments = bellman_split(ordered, self.depot, self.dist_fn)
            improved_segs = local_search(segments, self.depot, self.dist_fn)

            # Re-encode: flatten improved segments → new customer order
            new_order = [c for route in improved_segs for c in route]
            if len(new_order) != n:
                # Fallback: keep original if LS dropped customers (shouldn't happen)
                continue

            # Build new SPV vector: position i in new_order → SPV rank i/n
            x_new = np.empty(n)
            for rank, customer in enumerate(new_order):
                x_new[self._customer_pos[customer]] = rank / n
            X[k] = x_new

        return X


def run_ga_routing(
    depot: Depot,
    customers: List[Customer],
    dist_fn: Callable[[int, int], float],
    cfg: GAConfig,
) -> List[Route]:
    """
    Run GA to find the best visiting order for a depot's customers, then use
    the Bellman split to partition the giant tour into capacity-feasible routes.

    Parameters
    ----------
    depot:
        Depot that serves this cluster.
    customers:
        All customers assigned to this depot.
    dist_fn:
        Pre-computed O(1) distance callable from ``MDVRPAlgorithm._dist``.
    cfg:
        GAConfig loaded from config.yaml.

    Returns
    -------
    List of Routes covering all customers, one per vehicle.
    """
    if not customers:
        return []

    if len(customers) == 1:
        return [Route(depot=depot, customers=list(customers))]

    problem = RoutingProblem(depot=depot, customers=customers, dist_fn=dist_fn)

    algorithm = GA(
        pop_size=cfg.pop_size,
        mutation=LSMutation(
            depot=depot,
            customers=customers,
            dist_fn=dist_fn,
            prob=cfg.mutation_prob,
        ),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", cfg.n_gen),
        seed=cfg.seed,
        verbose=False,
    )

    perm = np.argsort(result.X)
    ordered_customers = [customers[i] for i in perm]
    segments = bellman_split(ordered_customers, depot, dist_fn)
    return [Route(depot=depot, customers=seg) for seg in segments]


def run_ga_reroute(
    current_start_node: Customer | Depot,
    pending_customers: List[Customer],
    real_end_depot: Depot,
    dist_fn: Callable[[int, int], float],
    cfg: GAConfig,
) -> List[Route]:
    """
    Run GA for dynamic reroute scenario (VRP-OD: origin-destination).
    Vehicle is at current_start_node and must visit pending_customers, then return to real_end_depot.
    This reroute is for a single vehicle, so it returns one open route.

    Parameters
    ----------
    current_start_node:
        Current vehicle position (Customer or Depot, NOT necessarily the original depot).
    pending_customers:
        Customers still to be served.
    real_end_depot:
        The original depot where the route must terminate.
    dist_fn:
        Pre-computed O(1) distance callable.
    cfg:
        GAConfig loaded from config.yaml.

    Returns
    -------
    List of Routes optimized for the dynamic scenario. For reroute, this is a
    single route: current_start_node -> ordered customers -> real_end_depot.
    """
    if not pending_customers:
        return []

    if len(pending_customers) == 1:
        return [Route(depot=real_end_depot, customers=list(pending_customers))]

    problem = DynamicRoutingProblem(
        current_start_node=current_start_node,
        pending_customers=pending_customers,
        real_end_depot=real_end_depot,
        dist_fn=dist_fn,
    )

    algorithm = GA(
        pop_size=cfg.pop_size,
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", cfg.n_gen),
        seed=cfg.seed,
        verbose=False,
    )

    perm = np.argsort(result.X)
    ordered_customers = [pending_customers[i] for i in perm]
    return [Route(depot=real_end_depot, customers=ordered_customers)]
