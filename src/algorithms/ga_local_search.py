"""
Prins (2004) local search for VRP.

Provides route-cost helpers and the 9-move local search (M1–M9) that
improves a multi-route VRP solution by scanning all O(n²) customer pairs
and applying the first improving move found, then restarting.
"""

from typing import Callable, List

from core.entities import Customer, Depot, Route


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
    routes: List[Route],
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
    routes = [list(r.customers) for r in routes if r]

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
