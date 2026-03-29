"""Protocols defining shared interfaces for visualization and algorithm interoperability."""

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class VisualizableRoute(Protocol):
    @property
    def depot_index(self) -> int:
        """1-based depot number."""
        ...

    @property
    def customer_indices(self) -> List[int]:
        """Ordered list of customer indices."""
        ...


@runtime_checkable
class VisualizableSolution(Protocol):
    @property
    def visualizable_routes(self) -> List[VisualizableRoute]:
        """All routes in this solution."""
        ...
