"""
Prins (2004) local search for VRP.

Provides route-cost helpers and the 9-move local search (M1-M9) that
improves a multi-route VRP solution by scanning all O(n^2) customer pairs
and applying the first improving move found, then restarting.
"""

from dataclasses import dataclass
import math
from typing import Callable, List

from algorithms.ga_split import linear_split
from core.entities import Customer, Depot, Route


@dataclass(frozen=True)
class _LocalSearchContext:
    """Runtime state injected once per local-search run."""

    consumed_capacity: list[float]
    consumed_duration: list[float]
    executed_last_nodes: list[Depot | Customer]


_LOCAL_SEARCH_CONTEXT: _LocalSearchContext | None = None


def _route_cost(route: List[Customer], depot: Depot, dist_fn: Callable[[int, int], float]) -> float:
    """Total round-trip cost: depot -> route[0] -> ... -> route[-1] -> depot."""
    if not route:
        return 0.0
    cost = dist_fn(depot.index, route[0].index)
    for i in range(len(route) - 1):
        cost += dist_fn(route[i].index, route[i + 1].index)
    cost += dist_fn(route[-1].index, depot.index)
    return cost


def _open_path_cost(
    route: List[Customer],
    start_node: Depot | Customer,
    end_node: Depot | Customer,
    dist_fn: Callable[[int, int], float],
) -> float:
    """Path cost for start_node -> route -> end_node."""
    if not route:
        return dist_fn(start_node.index, end_node.index)

    cost = dist_fn(start_node.index, route[0].index)
    for i in range(len(route) - 1):
        cost += dist_fn(route[i].index, route[i + 1].index)
    cost += dist_fn(route[-1].index, end_node.index)
    return cost


def _is_route_feasible(
    route: List[Customer],
    depot: Depot,
    dist_fn: Callable[[int, int], float],
    original_route: List[Customer] | None = None,
    *,
    route_idx: int | None = None,
) -> bool:
    """Check route constraints while tolerating historical violations from original_route."""
    if route_idx is not None and _LOCAL_SEARCH_CONTEXT is not None:
        if route_idx < len(_LOCAL_SEARCH_CONTEXT.consumed_capacity):
            consumed_capacity = _LOCAL_SEARCH_CONTEXT.consumed_capacity[route_idx]
            consumed_duration = (
                _LOCAL_SEARCH_CONTEXT.consumed_duration[route_idx]
                if route_idx < len(_LOCAL_SEARCH_CONTEXT.consumed_duration)
                else 0.0
            )
            context_last_node = (
                _LOCAL_SEARCH_CONTEXT.executed_last_nodes[route_idx]
                if route_idx < len(_LOCAL_SEARCH_CONTEXT.executed_last_nodes)
                else None
            )
        else:
            consumed_capacity = 0.0
            consumed_duration = 0.0
            context_last_node = None
    else:
        # When running outside simulation, there is no per-route runtime context.
        consumed_capacity = 0.0
        consumed_duration = 0.0
        context_last_node = None

    actual_start = context_last_node if context_last_node is not None else depot

    tolerated_capacity = max(0.0, depot.max_capacity - consumed_capacity)
    tolerated_duration = depot.max_duration
    if tolerated_duration > 0:
        tolerated_duration = max(0.0, tolerated_duration - consumed_duration)

    if original_route is not None:
        tolerated_capacity = max(tolerated_capacity, sum(c.demand for c in original_route))
        if depot.max_duration > 0:
            original_travel = _open_path_cost(original_route, actual_start, depot, dist_fn)
            original_service = sum(c.service_time for c in original_route)
            tolerated_duration = max(tolerated_duration, original_travel + original_service)

    if sum(c.demand for c in route) > tolerated_capacity:
        return False

    travel = _open_path_cost(route, actual_start, depot, dist_fn)
    if math.isinf(travel):
        return False

    if tolerated_duration > 0:
        service = sum(c.service_time for c in route)
        if travel + service > tolerated_duration:
            return False
    return True


def local_search_stage1_intra(
    customers: List[Customer],
    start_node: Depot | Customer,
    end_node: Depot | Customer,
    dist_fn: Callable[[int, int], float],
) -> List[Customer]:
    """
    Stage-1 disaster containment local search.

    Uses exactly three intra-route operators:
    M1: relocate, M2: swap, M3: 2-opt.
    """
    best = list(customers)
    if len(best) <= 1:
        return best

    def _duration(route: List[Customer]) -> float:
        service = sum(c.service_time for c in route)
        return _open_path_cost(route, start_node, end_node, dist_fn) + service

    best_cost = _duration(best)
    improved = True
    while improved:
        improved = False
        n = len(best)

        # M1: relocate
        for from_idx in range(n):
            if improved:
                break
            for to_idx in range(n + 1):
                if to_idx == from_idx or to_idx == from_idx + 1:
                    continue

                candidate = list(best)
                moved = candidate.pop(from_idx)
                insert_idx = to_idx if to_idx <= from_idx else to_idx - 1
                candidate.insert(insert_idx, moved)
                candidate_cost = _duration(candidate)
                if candidate_cost + 1e-9 < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break

        if improved:
            continue

        # M2: swap
        for i in range(n - 1):
            if improved:
                break
            for j in range(i + 1, n):
                candidate = list(best)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                candidate_cost = _duration(candidate)
                if candidate_cost + 1e-9 < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break

        if improved:
            continue

        # M3: 2-opt
        for i in range(n - 1):
            if improved:
                break
            for j in range(i + 1, n):
                candidate = list(best)
                candidate[i : j + 1] = candidate[i : j + 1][::-1]
                candidate_cost = _duration(candidate)
                if candidate_cost + 1e-9 < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break

    return best


def local_search(
    routes: List[Route | List[Customer]],
    depot: Depot,
    dist_fn: Callable[[int, int], float],
    local_search_max_iterations: int,
    is_stage_2: bool = False,
    frozen_route_indices: set[int] | None = None,
    executed_capacity_by_route: List[float] | None = None,
    executed_duration_by_route: List[float] | None = None,
    executed_last_nodes: List[Depot | Customer] | None = None,
) -> List[List[Customer]]:
    """
    Prins (2004) 9-move local search over a multi-route VRP solution.

    Operates on mutable lists of customers (routes).  The depot is implicit
    at the start and end of every route.  Scans all O(n^2) (u, v) customer
    pairs and applies the first improving move found, then restarts.  Moves
    M1-M6 are inter/intra-route relocate/swap moves; M7 is intra-route 2-opt;
    M8-M9 are inter-route 2-opt variants.

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
    is_stage_2:
        Enable VND phase control (INTER -> INTRA) for Stage 2 cluster reopt.
    frozen_route_indices:
        Route indices whose first customer must remain fixed during local search.
    executed_capacity_by_route:
        Capacity already consumed by each route prefix before optimization.
    executed_duration_by_route:
        Duration already consumed by each route prefix before optimization.
    executed_last_nodes:
        Real vehicle positions at the start of each pending suffix.

    Returns
    -------
    Cleaned list of non-empty routes.
    """
    # Work on copies so callers keep the originals until committed.
    normalized_routes: List[List[Customer]] = []
    for route in routes:
        if not route:
            continue
        if isinstance(route, Route):
            normalized_routes.append(list(route.customers))
        else:
            normalized_routes.append(list(route))
    routes = normalized_routes

    local_search_max_iterations = max(1, local_search_max_iterations)

    consumed_capacity = [0.0] * len(routes)
    if executed_capacity_by_route is not None:
        for idx, value in enumerate(executed_capacity_by_route[: len(routes)]):
            consumed_capacity[idx] = float(value)

    consumed_duration = [0.0] * len(routes)
    if executed_duration_by_route is not None:
        for idx, value in enumerate(executed_duration_by_route[: len(routes)]):
            consumed_duration[idx] = float(value)

    real_start_nodes: list[Depot | Customer] = [depot] * len(routes)
    if executed_last_nodes is not None:
        for idx, node in enumerate(executed_last_nodes[: len(routes)]):
            real_start_nodes[idx] = node

    global _LOCAL_SEARCH_CONTEXT
    _LOCAL_SEARCH_CONTEXT = _LocalSearchContext(
        consumed_capacity=consumed_capacity,
        consumed_duration=consumed_duration,
        executed_last_nodes=real_start_nodes,
    )

    def _prev(route: List[Customer], pos: int) -> int:
        """Index of node before route[pos]; returns depot.index if pos == 0."""
        return depot.index if pos == 0 else route[pos - 1].index

    def _next(route: List[Customer], pos: int) -> int:
        """Index of node after route[pos]; returns depot.index if pos == last."""
        return depot.index if pos == len(route) - 1 else route[pos + 1].index

    current_phase = "INTER" if is_stage_2 else "ALL"
    improved = True
    iterations = 0
    while iterations < local_search_max_iterations:
        if not improved:
            if is_stage_2 and current_phase == "INTER":
                current_phase = "INTRA"
                improved = True
            else:
                break

        iterations += 1
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
                    if (
                        frozen_route_indices is not None
                        and ru_idx in frozen_route_indices
                        and u_idx == 0
                    ):
                        continue
                    u = ru[u_idx]
                    x = ru[u_idx + 1] if u_idx + 1 < len(ru) else depot

                    for v_idx in range(len(rv)):
                        if improved:
                            break
                        if ru_idx == rv_idx and u_idx == v_idx:
                            continue

                        is_intra = (ru_idx == rv_idx)
                        if current_phase == "INTER" and is_intra:
                            continue
                        if current_phase == "INTRA" and not is_intra:
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
                                    if _is_route_feasible(new_r, depot, dist_fn, original_route=ru, route_idx=ru_idx):
                                        routes[ru_idx] = new_r
                                        improved = True
                                        break
                                else:
                                    new_ru = [c for k, c in enumerate(routes[ru_idx]) if k != u_idx]
                                    new_rv = list(routes[rv_idx])
                                    new_rv.insert(v_idx + 1, u)
                                    if (_is_route_feasible(new_ru, depot, dist_fn, original_route=ru, route_idx=ru_idx) and
                                        _is_route_feasible(new_rv, depot, dist_fn, original_route=rv, route_idx=rv_idx)):
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
                                    if _is_route_feasible(new_r, depot, dist_fn, original_route=ru, route_idx=ru_idx):
                                        routes[ru_idx] = new_r
                                        improved = True
                                        break
                                else:
                                    new_ru = [c for k, c in enumerate(ru) if k not in (u_idx, u_idx + 1)]
                                    new_rv = list(rv)
                                    new_rv.insert(v_idx + 1, x)
                                    new_rv.insert(v_idx + 1, u)
                                    if (_is_route_feasible(new_ru, depot, dist_fn, original_route=ru, route_idx=ru_idx) and
                                        _is_route_feasible(new_rv, depot, dist_fn, original_route=rv, route_idx=rv_idx)):
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
                                    new_r.insert(insert_pos, x)  # insert x at same pos -> [v, x, u, ...]
                                    if _is_route_feasible(new_r, depot, dist_fn, original_route=ru, route_idx=ru_idx):
                                        routes[ru_idx] = new_r
                                        improved = True
                                        break
                                else:
                                    new_ru = [c for k, c in enumerate(ru) if k not in (u_idx, u_idx + 1)]
                                    new_rv = list(rv)
                                    new_rv.insert(v_idx + 1, u)  # insert u first (will be after x)
                                    new_rv.insert(v_idx + 1, x)  # insert x at same pos -> [v, x, u, ...]
                                    if (_is_route_feasible(new_ru, depot, dist_fn, original_route=ru, route_idx=ru_idx) and
                                        _is_route_feasible(new_rv, depot, dist_fn, original_route=rv, route_idx=rv_idx)):
                                        routes[ru_idx] = new_ru
                                        routes[rv_idx] = new_rv
                                        improved = True
                                        break

                        if improved:
                            break

                        skip_swap = (
                            frozen_route_indices is not None
                            and rv_idx in frozen_route_indices
                            and v_idx == 0
                        )
                        if skip_swap:
                            continue

                        # M4: swap u and v
                        if ru_idx == rv_idx and abs(u_idx - v_idx) == 1:
                            # Adjacent same-route: the shared edge u<->v cancels (symmetric distances).
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
                                if _is_route_feasible(new_r, depot, dist_fn, original_route=ru, route_idx=ru_idx):
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
                                    if _is_route_feasible(new_r, depot, dist_fn, original_route=ru, route_idx=ru_idx):
                                        routes[ru_idx] = new_r
                                        improved = True
                                        break
                                else:
                                    new_ru = list(ru)
                                    new_rv = list(rv)
                                    new_ru[u_idx] = v
                                    new_rv[v_idx] = u
                                    if (_is_route_feasible(new_ru, depot, dist_fn, original_route=ru, route_idx=ru_idx) and
                                        _is_route_feasible(new_rv, depot, dist_fn, original_route=rv, route_idx=rv_idx)):
                                        routes[ru_idx] = new_ru
                                        routes[rv_idx] = new_rv
                                        improved = True
                                        break

                        if improved:
                            break

                        # M5: swap (u, x) with v
                        # v IS x (v_idx == u_idx+1 in same route) -> no-op, skip.
                        # Two adjacent same-route cases need corrected gain formulas because
                        # boundary nodes coincide with x or u, breaking the general 8-term formula.
                        if isinstance(x, Customer) and not (ru_idx == rv_idx and v_idx == u_idx + 1):
                            if (
                                frozen_route_indices is not None
                                and rv_idx in frozen_route_indices
                                and v_idx == 0
                            ):
                                continue
                            
                            u_prev = _prev(ru, u_idx)
                            x_next = _next(ru, u_idx + 1)
                            v_prev = _prev(rv, v_idx)
                            v_next = _next(rv, v_idx)

                            if ru_idx == rv_idx and v_idx == u_idx + 2:
                                # Route: [..., p, u, x, v, q, ...] -> [..., p, v, u, x, q, ...]
                                # d(u,x) cancels; v_prev==x so general formula double-counts d(x,v).
                                p, q = u_prev, v_next
                                gain = (
                                    d(p, u.index) + d(x.index, v.index) + d(v.index, q)
                                    - d(p, v.index) - d(v.index, u.index) - d(x.index, q)
                                )
                            elif ru_idx == rv_idx and v_idx == u_idx - 1:
                                # Route: [..., p, v, u, x, q, ...] -> [..., p, u, x, v, q, ...]
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
                                    if _is_route_feasible(new_r, depot, dist_fn, original_route=ru, route_idx=ru_idx):
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
                                    if (_is_route_feasible(new_ru, depot, dist_fn, original_route=ru, route_idx=ru_idx) and
                                        _is_route_feasible(new_rv, depot, dist_fn, original_route=rv, route_idx=rv_idx)):
                                        routes[ru_idx] = new_ru
                                        routes[rv_idx] = new_rv
                                        improved = True
                                        break

                        if improved:
                            break

                        # M6: swap (u, x) with (v, y)
                        # Same-route pairs overlap when |v_idx - u_idx| <= 1:
                        #   v_idx == u_idx+1 -> v IS x; v_idx == u_idx-1 -> y IS u. Skip both.
                        # Adjacent non-overlapping pairs (distance == 2) share a boundary node
                        # with the other pair, breaking the general gain formula.
                        _m6_overlap = ru_idx == rv_idx and abs(v_idx - u_idx) <= 1
                        if isinstance(x, Customer) and isinstance(y, Customer) and not _m6_overlap:
                            if (
                                frozen_route_indices is not None
                                and rv_idx in frozen_route_indices
                                and v_idx == 0
                            ):
                                continue

                            u_prev = _prev(ru, u_idx)
                            x_next = _next(ru, u_idx + 1)
                            v_prev = _prev(rv, v_idx)
                            y_next = _next(rv, v_idx + 1)

                            if ru_idx == rv_idx and v_idx == u_idx + 2:
                                # [p, u, x, v, y, q] -> [p, v, y, u, x, q]
                                # x_next==v and v_prev==x -> d(x,v) double-counted
                                p, q = u_prev, y_next
                                gain = (
                                    d(p, u.index) + d(x.index, v.index) + d(y.index, q)
                                    - d(p, v.index) - d(y.index, u.index) - d(x.index, q)
                                )
                            elif ru_idx == rv_idx and v_idx == u_idx - 2:
                                # [p, v, y, u, x, q] -> [p, u, x, v, y, q]
                                # u_prev==y and y_next==u -> d(y,u) double-counted
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
                                    # Pairs don't overlap -> 4 direct index assignments.
                                    new_r = list(ru)
                                    new_r[u_idx] = v
                                    new_r[u_idx + 1] = y
                                    new_r[v_idx] = u
                                    new_r[v_idx + 1] = x
                                    if _is_route_feasible(new_r, depot, dist_fn, original_route=ru, route_idx=ru_idx):
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
                                    if (_is_route_feasible(new_ru, depot, dist_fn, original_route=ru, route_idx=ru_idx) and
                                        _is_route_feasible(new_rv, depot, dist_fn, original_route=rv, route_idx=rv_idx)):
                                        routes[ru_idx] = new_ru
                                        routes[rv_idx] = new_rv
                                        improved = True
                                        break

                        if improved:
                            break

                        # M7: 2-opt within same route
                        # Replace edges (u->x) and (v->y) with (u->v) and (x->y).
                        # Reverses segment route[u_idx+1 : v_idx+1].
                        # Guard: u_idx+1 < v_idx ensures non-degenerate (>=1 node reversed)
                        # and guarantees x = ru[u_idx+1] is a Customer (not depot).
                        if ru_idx == rv_idx and u_idx + 1 < v_idx:
                            gain = (
                                d(u.index, x.index) + d(v.index, y.index)
                                - d(u.index, v.index) - d(x.index, y.index)
                            )
                            if gain > 1e-9:
                                new_r = list(ru)
                                new_r[u_idx + 1: v_idx + 1] = new_r[u_idx + 1: v_idx + 1][::-1]
                                if _is_route_feasible(new_r, depot, dist_fn, original_route=ru, route_idx=ru_idx):
                                    routes[ru_idx] = new_r
                                    improved = True
                                    break

                        if improved:
                            break

                        # M8: inter-route 2-opt (reconnect)
                        # T(u) != T(v): different routes only.
                        # Replace edges (u->x) and (v->y) with (u->v) and (x->y).
                        # new_ru = ru[:u+1] + reversed(rv[:v+1])
                        # new_rv = ru[u+1:] + rv[v+1:]
                        # Use actual route-cost delta to avoid gain-formula oscillation:
                        # the edge-delta formula is symmetric so scanning (v,u) after (u,v)
                        # reports the same positive gain and reverses the move endlessly.
                        if ru_idx < rv_idx:
                            if (
                                frozen_route_indices is not None
                                and (
                                    ru_idx in frozen_route_indices
                                    or rv_idx in frozen_route_indices
                                )
                            ):
                                continue
                            
                            prefix_u = ru[: u_idx + 1]
                            suffix_u = ru[u_idx + 1:]
                            prefix_v = rv[: v_idx + 1]
                            suffix_v = rv[v_idx + 1:]
                            new_ru = prefix_u + prefix_v[::-1]
                            new_rv = suffix_u + suffix_v
                            cost_before = _route_cost(ru, depot, dist_fn) + _route_cost(rv, depot, dist_fn)
                            cost_after = _route_cost(new_ru, depot, dist_fn) + _route_cost(new_rv, depot, dist_fn)
                            if cost_before - cost_after > 1e-9:
                                if (_is_route_feasible(new_ru, depot, dist_fn, original_route=ru, route_idx=ru_idx) and
                                    _is_route_feasible(new_rv, depot, dist_fn, original_route=rv, route_idx=rv_idx)):
                                    routes[ru_idx] = new_ru
                                    routes[rv_idx] = new_rv
                                    improved = True
                                    break

                        if improved:
                            break

                        # M9: inter-route tail-swap
                        # T(u) != T(v): different routes only.
                        # Replace edges (u->x) and (v->y) with (u->y) and (v->x).
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
                                if (_is_route_feasible(new_ru, depot, dist_fn, original_route=ru, route_idx=ru_idx) and
                                    _is_route_feasible(new_rv, depot, dist_fn, original_route=rv, route_idx=rv_idx)):
                                    routes[ru_idx] = new_ru
                                    routes[rv_idx] = new_rv
                                    improved = True
                                    break

    # Remove empty routes and return (except in Stage 2, where physical
    # vehicle index mapping is strictly required).
    if is_stage_2:
        result = routes
    else:
        result = [r for r in routes if r]

    _LOCAL_SEARCH_CONTEXT = None
    return result

