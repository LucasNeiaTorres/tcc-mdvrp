"""Converter functions to build domain entities from parsed Cordeau data."""

from typing import List

from core.entities import Customer, Depot
from utils.data_loader import CordeauInstance


def build_customers(instance: CordeauInstance) -> List[Customer]:
    """Build a list of Customer entities from a parsed CordeauInstance."""
    return [
        Customer(
            index=node.index,
            x=node.x,
            y=node.y,
            demand=node.demand,
            service_time=node.service_time,
        )
        for node in instance.customers
    ]


def build_depots(instance: CordeauInstance) -> List[Depot]:
    """
    Build a list of Depot entities from a parsed CordeauInstance.
    """
    return [
        Depot(
            index=node.index,
            x=node.x,
            y=node.y,
            max_duration=instance.duration_limits[depot_number - 1],
            max_capacity=instance.capacity_limits[depot_number - 1],
            max_vehicles=instance.vehicle_count,
        )
        for depot_number, node in enumerate(instance.depots, start=1)
    ]
