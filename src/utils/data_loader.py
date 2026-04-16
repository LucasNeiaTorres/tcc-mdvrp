from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

@dataclass
class VisitCombination:
    code: int
    days: List[int]


@dataclass
class Node:
    index: int  # i: customer/depot number (1 to n for customers, n+1 to n+t for depots)
    x: float
    y: float
    service_time: float  # d: time required to serve this customer
    demand: float  # q: demand of the customer (capacity constraint)
    frequency: int  # f: how often the customer must be visited (periodicity)
    combination_count: int  # a: number of visit combinations
    combinations: List[VisitCombination] = field(default_factory=list)  # list: visit combinations (code and corresponding days)
    time_window: Optional[Tuple[float, float]] = None  # optional time window (e, l) for service start time


@dataclass
class ParsedRoute:
    depot: int  # l: number of the depot
    vehicle: int  # k: number of the vehicle
    duration: float  # d: duration of the route
    load: float  # q: load of the vehicle
    nodes: List[int]  # list: ordered sequence of customers

    @property
    def depot_index(self) -> int:
        return self.depot

    @property
    def customer_indices(self) -> List[int]:
        return self.nodes


@dataclass
class CordeauInstance:
    problem_type: int
    vehicle_count: int
    customer_count: int
    depot_count: int
    duration_limits: List[float]
    capacity_limits: List[float]
    customers: List[Node]
    depots: List[Node]


@dataclass
class CordeauSolution:
    objective: float
    routes: List[ParsedRoute]

    @property
    def visualizable_routes(self) -> List[ParsedRoute]:
        return self.routes


def _code_to_days(code: int, period: int) -> List[int]:
    """Convert binary code to list of days (1-based indices)."""
    if period < 1:
        return []
    b = format(code, "b").zfill(period)
    return [i + 1 for i, bit in enumerate(b) if bit == "1"]


def read_cordeau_data_file(path: str) -> CordeauInstance:
    """Read a Cordeau MDVRP data file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if len(lines) < 1:
        raise ValueError("Empty data file")

    header = lines[0].split()
    if len(header) < 4:
        raise ValueError("Invalid header line in Cordeau data file")

    problem_type, m, n, t = map(int, header[:4])
    depot_count = t

    if len(lines) < 1 + depot_count:
        raise ValueError("File does not contain depot constraints")

    duration_limits = []
    capacity_limits = []
    for i in range(1, 1 + depot_count):
        parts = lines[i].split()
        if len(parts) < 2:
            raise ValueError(f"Invalid period line: {lines[i]}")
        d, q = map(float, parts[:2])
        duration_limits.append(d)
        capacity_limits.append(q)

    node_lines = lines[1 + depot_count :]

    nodes: List[Node] = []
    for line in node_lines:
        parts = line.split()
        if len(parts) < 7:
            raise ValueError(f"Expected at least 7 tokens in node line, got {len(parts)}: {line}")

        idx = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        service_time = float(parts[3])
        demand = float(parts[4])
        frequency = int(parts[5])
        combination_count = int(parts[6])

        expected_len = 7 + combination_count
        if len(parts) < expected_len:
            raise ValueError(f"Not enough visit combination tokens in line: {line}")

        combination_tokens = parts[7 : 7 + combination_count]
        combinations = [
            VisitCombination(code=int(token), days=_code_to_days(int(token), depot_count))
            for token in combination_tokens
        ]

        time_window = None
        remaining = parts[7 + combination_count :]
        if len(remaining) == 2:
            e = float(remaining[0])
            l = float(remaining[1])
            time_window = (e, l)
        elif len(remaining) != 0:
            raise ValueError(f"Invalid remaining tokens for node line: {remaining}")

        nodes.append(
            Node(
                index=idx,
                x=x,
                y=y,
                service_time=service_time,
                demand=demand,
                frequency=frequency,
                combination_count=combination_count,
                combinations=combinations,
                time_window=time_window,
            )
        )

    depots: List[Node] = []
    customers: List[Node] = []

    assert problem_type == 2, "Only MDVRP (type 2) is supported"
    if len(nodes) < n + depot_count:
        raise ValueError("Node count is smaller than n + t for MDVRP")
    customers = [node for node in nodes if node.index <= n]
    depots = [node for node in nodes if node.index > n]

    return CordeauInstance(
        problem_type=problem_type,
        vehicle_count=m,
        customer_count=n,
        depot_count=depot_count,
        duration_limits=duration_limits,
        capacity_limits=capacity_limits,
        customers=customers,
        depots=depots,
    )


def read_cordeau_solution_file(
    path: str,
    instance: Optional["CordeauInstance"] = None,
) -> CordeauSolution:
    """Read a Cordeau MDVRP solution file.

    If *instance* is provided, the 1-based depot numbers in the solution file
    are translated to the raw node indices used by the instance, so that
    ``ParsedRoute.depot_index`` is consistent with ``Depot.index``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Solution file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not lines:
        raise ValueError("Empty solution file")

    objective = float(lines[0].split()[0])
    routes: List[ParsedRoute] = []

    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid route line: {line}")

        depot_number = int(parts[0])
        vehicle = int(parts[1])
        duration = float(parts[2])
        load = float(parts[3])
        seq_items = [int(x) for x in parts[4:]]

        if seq_items and seq_items[0] == 0:
            seq_items = seq_items[1:]
        if seq_items and seq_items[-1] == 0:
            seq_items = seq_items[:-1]

        if instance is not None:
            depot_node_index = instance.depots[depot_number - 1].index
        else:
            depot_node_index = depot_number

        routes.append(
            ParsedRoute(
                depot=depot_node_index,
                vehicle=vehicle,
                duration=duration,
                load=load,
                nodes=seq_items,
            )
        )

    return CordeauSolution(objective=objective, routes=routes)
