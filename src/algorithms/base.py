"""Abstract base classes for MDVRP algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, List

from core.entities import Customer, Depot, Route
from core.solution import Solution
from utils.metrics import build_distance_matrix


class MDVRPAlgorithm(ABC):
    """Base interface for all MDVRP solving algorithms."""

    # Set by _build_matrix() before solve logic runs
    _dist_matrix: List[List[float]]
    _node_offset: Dict[int, int]  # node.index → row in matrix

    def _build_matrix(self, depots: List[Depot], customers: List[Customer]) -> None:
        """Pre-compute the distance matrix and index maps for O(1) lookups."""
        self._dist_matrix = build_distance_matrix(depots, customers)
        self._node_offset = {
            **{d.index: i for i, d in enumerate(depots)},
            **{c.index: len(depots) + i for i, c in enumerate(customers)},
        }

    def _dist(self, a_index: int, b_index: int) -> float:
        """O(1) distance lookup between any two node indices."""
        return self._dist_matrix[self._node_offset[a_index]][self._node_offset[b_index]]

    def _set_edge_inf(self, node_a: int, node_b: int) -> None:
        """Set undirected edge (node_a, node_b) to infinity in the distance matrix."""
        ia = self._node_offset.get(node_a)
        ib = self._node_offset.get(node_b)
        if ia is None or ib is None:
            return
        self._dist_matrix[ia][ib] = float("inf")
        self._dist_matrix[ib][ia] = float("inf")

    @abstractmethod
    def solve(self, customers: List[Customer], depots: List[Depot]) -> Solution:
        """
        Solve the MDVRP problem and return a Solution.

        Args:
            customers: List of Customer entities to be served
            depots: List of Depot entities with capacity and duration limits

        Returns:
            A Solution containing the set of routes found
        """
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class ClusterFirstAlgorithm(MDVRPAlgorithm):
    """
    Base class for cluster-first, route-second algorithms.

    Flow: customers + depots → cluster() → route() → Solution
    """

    @abstractmethod
    def cluster(
        self, customers: List[Customer], depots: List[Depot]
    ) -> Dict[Depot, List[Customer]]:
        """Assign customers to depots. """
        ...

    @abstractmethod
    def route(self, clusters: Dict[Depot, List[Customer]]) -> Solution:
        """Build routes for each depot's cluster of customers."""
        ...

    def solve(self, customers: List[Customer], depots: List[Depot]) -> Solution:
        self._build_matrix(depots, customers)
        clusters = self.cluster(customers, depots)
        return self.route(clusters)