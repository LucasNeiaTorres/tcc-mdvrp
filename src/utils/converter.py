"""Converter functions to build domain entities from parsed Cordeau data."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.entities import Customer, Depot
from utils.data_loader import CordeauInstance, CordeauSolution, read_cordeau_data_file, read_cordeau_solution_file

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


@dataclass
class LoadedInstance:
    name: str
    raw: CordeauInstance
    customers: List[Customer]
    depots: List[Depot]
    reference: Optional[CordeauSolution]


def load_instance(name: str, data_dir: Path = _DATA_DIR) -> LoadedInstance:
    """
    Load a Cordeau benchmark instance by name (e.g. ``"p01"``).

    Looks for the data file at ``data_dir/cordeau/<name>`` and the solution
    file at ``data_dir/cordeau_sol/<name>.res``.  The reference solution is
    ``None`` when the solution file does not exist.

    Parameters
    ----------
    name:
        Instance name, e.g. ``"p01"`` or ``"pr03"``.
    data_dir:
        Base data directory.  Defaults to ``<project_root>/data/raw``.
    """
    data_file = data_dir / "cordeau" / name
    sol_file = data_dir / "cordeau_sol" / f"{name}.res"

    instance = read_cordeau_data_file(str(data_file))
    customers = build_customers(instance)
    depots = build_depots(instance)

    reference: Optional[CordeauSolution] = None
    if sol_file.exists():
        reference = read_cordeau_solution_file(str(sol_file), instance)

    return LoadedInstance(name=name, raw=instance, customers=customers, depots=depots, reference=reference)


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
