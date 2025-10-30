from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    point_id: int
    name: str
    x: float
    y: float
    location_type: str 
    demand: int
    tw_start: int
    tw_end: int
    service_time: int
    risk: float

    def distance_to(self, other: 'Point') -> float:
        """Calculates the Euclidean distance to another point."""
        return ((self.x - other.x)**2 + 
                (self.y - other.y)**2)**0.5

    def __repr__(self) -> str:
        return f"Point(id={self.point_id}, name='{self.name}')"