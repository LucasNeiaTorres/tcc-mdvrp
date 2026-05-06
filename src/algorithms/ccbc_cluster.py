"""
Constrained Centroid-Based Clustering (CCBC) module for MDVRP.

Solves the depot-assignment sub-problem by extending capacitated k-means
with three enhancements described in the CCBC literature:

1. Vehicle-level slots
   The cluster count k equals the total number of vehicles across all depots
   (Σ max_vehicles).  Each slot has a capacity budget equal to one vehicle's
   capacity (depot.max_capacity), giving the algorithm the tightest feasible
   bound at assignment time.  After clustering the slots are mapped back to
   their originating depot.

2. Multi-start initialisation
   Start 0 — Voronoi sub-clustering:
     a. Upper-level Voronoi: assign each customer to its nearest depot.
     b. Lower-level k-means: for each depot with v vehicles, run uncapacitated
        k-means on its Voronoi members to generate v sub-centroids.  Empty
        Voronoi regions receive angularly offset positions around the depot.
   Starts 1..n_starts-1 — Gaussian perturbation of the start-0 centroids
   with σ = 10 % of the mean inter-depot distance, giving diverse coverage of
   the centroid space without losing the geometric signal.

3. Two-phase constrained assignment (run each iteration)
   Phase 1 — controversy-ordered greedy:
     Sort customers by controversy score = dist_nearest − dist_second_nearest
     (descending).  Customers with a large gap are most "decided" and are
     assigned first, preserving capacity for ambiguous boundary customers.
     Each customer is assigned to the nearest slot that still has remaining
     capacity.
   Phase 2 — weighted lower-level Voronoi repair:
     Overflow customers (no slot has capacity) are assigned to the slot that
     minimises dist × (1 + overload_ratio), where overload_ratio is the
     fraction of capacity already exceeded.  This approximates the lower-level
     Voronoi boundary adjustment described in bi-level Voronoi MDVRP approaches.

The best start (lowest total customer-to-centroid distance) is selected as
the final assignment.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from core.entities import Customer, Depot
from utils.config import CCBCConfig

_RNG_SEED = 42


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _build_slots(depots: List[Depot]) -> Tuple[List[Depot], List[float]]:
    """
    Expand depots into individual vehicle slots.

    Returns
    -------
    slot_depots:
        One Depot entry per vehicle (repeated depot.max_vehicles times).
    slot_capacities:
        Per-slot capacity budget = depot.max_capacity.
    """
    slot_depots: List[Depot] = []
    slot_capacities: List[float] = []
    for depot in depots:
        for _ in range(max(1, depot.max_vehicles)):
            slot_depots.append(depot)
            slot_capacities.append(depot.max_capacity)
    return slot_depots, slot_capacities


def _simple_kmeans_centroids(
    points: List[Tuple[float, float]],
    k: int,
    max_iter: int = 30,
    rng: Optional[random.Random] = None,
) -> List[Tuple[float, float]]:
    """
    Run uncapacitated k-means on a list of (x, y) points.

    Used during initialisation to place sub-centroids within each depot's
    Voronoi region.  Returns k centroids (fewer if |points| < k).
    """
    if not points:
        return []
    if len(points) <= k:
        return list(points)

    _rng = rng or random.Random(_RNG_SEED)
    centroids = _rng.sample(points, k)

    for _ in range(max_iter):
        clusters: List[List[Tuple[float, float]]] = [[] for _ in range(k)]
        for p in points:
            nearest = min(range(k), key=lambda s: _euclidean(p[0], p[1], centroids[s][0], centroids[s][1]))
            clusters[nearest].append(p)

        new_centroids: List[Tuple[float, float]] = []
        for s in range(k):
            if clusters[s]:
                cx = sum(p[0] for p in clusters[s]) / len(clusters[s])
                cy = sum(p[1] for p in clusters[s]) / len(clusters[s])
                new_centroids.append((cx, cy))
            else:
                new_centroids.append(centroids[s])

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return centroids


def _init_centroids(
    customers: List[Customer],
    depots: List[Depot],
    slot_depots: List[Depot],
    start_idx: int,
    base_centroids: List[Tuple[float, float]],
    rng: random.Random,
    sigma: float,
) -> List[Tuple[float, float]]:
    """
    Produce initial centroid positions for one multi-start iteration.

    Start 0
    -------
    Upper-level Voronoi: assign each customer to its nearest depot.
    Lower-level k-means: for each depot, run uncapacitated k-means on its
    Voronoi members to place max_vehicles sub-centroids within that region.
    Empty Voronoi cells fall back to angularly offset positions.

    Start i > 0
    -----------
    Add Gaussian noise (σ = sigma) to base_centroids for diversity.
    """
    k = len(slot_depots)

    if start_idx == 0:
        # Upper-level Voronoi assignment
        depot_members: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(depots))}
        for c in customers:
            nearest_depot_idx = min(
                range(len(depots)),
                key=lambda i: _euclidean(c.x, c.y, depots[i].x, depots[i].y),
            )
            depot_members[nearest_depot_idx].append((c.x, c.y))

        # Lower-level k-means per depot
        centroids: List[Tuple[float, float]] = []
        slot_offset = 0
        for di, depot in enumerate(depots):
            v = max(1, depot.max_vehicles)
            members = depot_members[di]
            if members:
                sub = _simple_kmeans_centroids(members, v, rng=rng)
                # Pad with depot position if fewer sub-centroids than vehicles
                while len(sub) < v:
                    angle = 2 * math.pi * len(sub) / v
                    sub.append((depot.x + math.cos(angle), depot.y + math.sin(angle)))
                centroids.extend(sub[:v])
            else:
                # Angular offsets around depot for empty Voronoi cells
                for vi in range(v):
                    angle = 2 * math.pi * vi / v
                    centroids.append((depot.x + math.cos(angle), depot.y + math.sin(angle)))
            slot_offset += v

        return centroids

    # Gaussian perturbation of start-0 centroids
    return [
        (cx + rng.gauss(0, sigma), cy + rng.gauss(0, sigma))
        for cx, cy in base_centroids
    ]


def _assign_customers(
    customers: List[Customer],
    slot_capacities: List[float],
    centroids: List[Tuple[float, float]],
) -> List[int]:
    """
    Two-phase constrained assignment.

    Phase 1 — controversy-ordered greedy (capacity-respecting).
    Phase 2 — weighted lower-level Voronoi repair for overflow customers.

    Returns slot index per customer (same order as customers list).
    """
    k = len(centroids)
    remaining = list(slot_capacities)

    # Pre-compute distances: customer i → slot s
    dists = [
        [_euclidean(c.x, c.y, centroids[s][0], centroids[s][1]) for s in range(k)]
        for c in customers
    ]

    # Controversy = dist_nearest - dist_second_nearest (descending = most decided first)
    def _controversy(i: int) -> float:
        sorted_d = sorted(dists[i])
        return sorted_d[1] - sorted_d[0] if k >= 2 else 0.0

    phase1_order = sorted(range(len(customers)), key=_controversy, reverse=True)
    assignment = [-1] * len(customers)
    overflow: List[int] = []

    # Phase 1: greedy with capacity check
    for i in phase1_order:
        c = customers[i]
        slots_by_dist = sorted(range(k), key=lambda s: dists[i][s])
        chosen = None
        for s in slots_by_dist:
            if remaining[s] >= c.demand:
                chosen = s
                break
        if chosen is not None:
            assignment[i] = chosen
            remaining[chosen] -= c.demand
        else:
            overflow.append(i)

    # Phase 2: weighted lower-level Voronoi repair
    for i in overflow:
        c = customers[i]
        # overload_ratio = fraction by which the slot is already over budget
        def _weight(s: int) -> float:
            overload = max(0.0, -remaining[s]) / slot_capacities[s] if slot_capacities[s] > 0 else 0.0
            return dists[i][s] * (1.0 + overload)

        chosen = min(range(k), key=_weight)
        assignment[i] = chosen
        remaining[chosen] -= c.demand

    return assignment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ccbc_clustering(
    customers: List[Customer],
    depots: List[Depot],
    cfg: CCBCConfig,
) -> Dict[Depot, List[Customer]]:
    """
    Assign customers to depots via Constrained Centroid-Based Clustering (CCBC).

    Parameters
    ----------
    customers:
        All Customer entities to cluster.
    depots:
        All Depot entities, one per logical depot.
    cfg:
        CCBCConfig loaded from config.yaml (uses max_iter, tol, n_starts).

    Returns
    -------
    Dict mapping each Depot to its assigned list of Customers.
    Depots with no assigned customers are included with an empty list.
    """
    if not customers:
        return {depot: [] for depot in depots}

    slot_depots, slot_capacities = _build_slots(depots)
    k = len(slot_depots)

    # Sigma for Gaussian perturbation = 10 % of mean inter-depot distance
    if len(depots) >= 2:
        inter_depot_distances = [
            _euclidean(depots[i].x, depots[i].y, depots[j].x, depots[j].y)
            for i in range(len(depots))
            for j in range(i + 1, len(depots))
        ]
        sigma = 0.1 * (sum(inter_depot_distances) / len(inter_depot_distances))
    else:
        sigma = 1.0

    rng = random.Random(_RNG_SEED)

    best_assignment: Optional[List[int]] = None
    best_cost = float("inf")
    base_centroids: List[Tuple[float, float]] = []

    for start in range(cfg.n_starts):
        centroids = _init_centroids(
            customers=customers,
            depots=depots,
            slot_depots=slot_depots,
            start_idx=start,
            base_centroids=base_centroids,
            rng=rng,
            sigma=sigma,
        )
        if start == 0:
            base_centroids = list(centroids)

        assignment = [0] * len(customers)

        for _ in range(cfg.max_iter):
            new_assignment = _assign_customers(customers, slot_capacities, centroids)

            # Update centroids to mean of assigned members
            new_centroids: List[Tuple[float, float]] = []
            for s in range(k):
                members = [customers[i] for i in range(len(customers)) if new_assignment[i] == s]
                if members:
                    cx = sum(c.x for c in members) / len(members)
                    cy = sum(c.y for c in members) / len(members)
                    new_centroids.append((cx, cy))
                else:
                    # Reset to originating depot position
                    new_centroids.append((slot_depots[s].x, slot_depots[s].y))

            # Convergence: max centroid shift
            max_shift = max(
                _euclidean(centroids[s][0], centroids[s][1], new_centroids[s][0], new_centroids[s][1])
                for s in range(k)
            )
            centroids = new_centroids
            assignment = new_assignment

            if max_shift < cfg.tol:
                break

        # Evaluate this start: total customer-to-centroid distance
        cost = sum(
            _euclidean(customers[i].x, customers[i].y, centroids[assignment[i]][0], centroids[assignment[i]][1])
            for i in range(len(customers))
        )
        if cost < best_cost:
            best_cost = cost
            best_assignment = list(assignment)

    # Map slot assignments back to depots
    clusters: Dict[Depot, List[Customer]] = {depot: [] for depot in depots}
    assert best_assignment is not None
    for i, slot in enumerate(best_assignment):
        clusters[slot_depots[slot]].append(customers[i])

    return clusters
