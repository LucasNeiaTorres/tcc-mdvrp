"""Abstract base classes for MDVRP algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, List

from core.entities import Customer, Depot
from core.solution import Solution
from utils.metrics import build_distance_matrix


class MDVRPAlgorithm(ABC):
    """Base interface for all MDVRP solving algorithms."""

    # Set by _build_matrix() before solve logic runs
    _dist_matrix: List[List[float]]
    _depot_offset: Dict[int, int]    # depot.index → row in matrix
    _customer_offset: Dict[int, int]  # customer.index → row in matrix

    def _build_matrix(self, depots: List[Depot], customers: List[Customer]) -> None:
        """Pre-compute the distance matrix and index maps for O(1) lookups."""
        self._dist_matrix = build_distance_matrix(depots, customers)
        self._depot_offset = {d.index: i for i, d in enumerate(depots)}
        self._customer_offset = {
            c.index: len(depots) + i for i, c in enumerate(customers)
        }

    def _dist(self, a_index: int, b_index: int) -> float:
        """O(1) distance lookup between any two node indices."""
        if a_index in self._depot_offset:
            a_row = self._depot_offset[a_index]
        else:
            a_row = self._customer_offset[a_index]

        if b_index in self._depot_offset:
            b_row = self._depot_offset[b_index]
        else:
            b_row = self._customer_offset[b_index]

        return self._dist_matrix[a_row][b_row]

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


class RouteFirstAlgorithm(MDVRPAlgorithm):
    """
    Base class for route-first, cluster-second algorithms.

    Flow: customers + depots → route() → cluster() → Solution

    A giant tour visiting all customers is built first, then the
    clustering step partitions and assigns segments to depots/vehicles.
    """

    @abstractmethod
    def route(self, customers: List[Customer], depots: List[Depot]) -> List[Customer]:
        """Build a giant tour: an ordered sequence of all customers."""
        ...

    @abstractmethod
    def cluster(
        self, giant_tour: List[Customer], depots: List[Depot]
    ) -> Solution:
        """Partition the giant tour into feasible routes assigned to depots."""
        ...

    def solve(self, customers: List[Customer], depots: List[Depot]) -> Solution:
        self._build_matrix(depots, customers)
        giant_tour = self.route(customers, depots)
        return self.cluster(giant_tour, depots)
