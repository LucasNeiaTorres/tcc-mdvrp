"""
Configuration loader for the MDVRP solver.

Reads config.yaml from the project root and returns typed dataclasses
so all algorithm modules receive validated, IDE-completable parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from utils.defaults import (
    DEFAULT_GA_CLONE_DELTA,
    DEFAULT_GA_LOCAL_SEARCH_MAX_ITERATIONS,
    DEFAULT_GA_STAGNATION_FTOL,
    DEFAULT_GA_STAGNATION_PERIOD,
    DEFAULT_GA_TIME_LIMIT,
    DEFAULT_SIMULATION_CLUSTER_DEGRADATION_THRESHOLD,
    DEFAULT_SIMULATION_PENALTY_OVERCAPACITY_PER_UNIT,
    DEFAULT_SIMULATION_PENALTY_OVERTIME_PER_MINUTE,
    DEFAULT_SIMULATION_REROUTE_DEGRADATION_THRESHOLD,
)


@dataclass
class CCBCConfig:
    max_iter: int
    tol: float
    n_starts: int


@dataclass
class GAConfig:
    pop_size: int
    n_gen: int
    seed: int
    mutation_prob: float
    local_search_max_iterations: int = DEFAULT_GA_LOCAL_SEARCH_MAX_ITERATIONS
    clone_delta: float = DEFAULT_GA_CLONE_DELTA
    stagnation_period: int = DEFAULT_GA_STAGNATION_PERIOD
    stagnation_ftol: float = DEFAULT_GA_STAGNATION_FTOL
    time_limit: str = DEFAULT_GA_TIME_LIMIT


@dataclass
class SimulationConfig:
    reroute_degradation_threshold: float = DEFAULT_SIMULATION_REROUTE_DEGRADATION_THRESHOLD
    cluster_degradation_threshold: float = DEFAULT_SIMULATION_CLUSTER_DEGRADATION_THRESHOLD
    penalty_overcapacity_per_unit: float = DEFAULT_SIMULATION_PENALTY_OVERCAPACITY_PER_UNIT
    penalty_overtime_per_minute: float = DEFAULT_SIMULATION_PENALTY_OVERTIME_PER_MINUTE


@dataclass
class AppConfig:
    ccbc: CCBCConfig
    ga: GAConfig
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def load_config(path: Optional[str] = None) -> AppConfig:
    """
    Load and parse config.yaml into an AppConfig dataclass.

    Args:
        path: Path to the YAML file. Defaults to config.yaml in the
              project root (two levels above this file).

    Returns:
        AppConfig with fully typed ccbc and ga sub-configs.
    """
    if path is None:
        path = str(Path(__file__).parent.parent.parent / "config.yaml")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    ccbc_raw = raw["ccbc"]
    ga_raw = raw["ga"]
    simulation_raw = raw.get("simulation", {})

    return AppConfig(
        ccbc=CCBCConfig(
            max_iter=int(ccbc_raw["max_iter"]),
            tol=float(ccbc_raw["tol"]),
            n_starts=int(ccbc_raw["n_starts"]),
        ),
        ga=GAConfig(
            pop_size=int(ga_raw["pop_size"]),
            n_gen=int(ga_raw["n_gen"]),
            seed=int(ga_raw["seed"]),
            mutation_prob=float(ga_raw["mutation_prob"]),
            local_search_max_iterations=int(
                ga_raw.get(
                    "local_search_max_iterations",
                    DEFAULT_GA_LOCAL_SEARCH_MAX_ITERATIONS,
                )
            ),
            clone_delta=float(ga_raw.get("clone_delta", DEFAULT_GA_CLONE_DELTA)),
            stagnation_period=int(
                ga_raw.get("stagnation_period", DEFAULT_GA_STAGNATION_PERIOD)
            ),
            stagnation_ftol=float(
                ga_raw.get("stagnation_ftol", DEFAULT_GA_STAGNATION_FTOL)
            ),
            time_limit=str(ga_raw.get("time_limit", DEFAULT_GA_TIME_LIMIT)),
        ),
        simulation=SimulationConfig(
            reroute_degradation_threshold=float(
                simulation_raw.get(
                    "reroute_degradation_threshold",
                    DEFAULT_SIMULATION_REROUTE_DEGRADATION_THRESHOLD,
                )
            ),
            cluster_degradation_threshold=float(
                simulation_raw.get(
                    "cluster_degradation_threshold",
                    DEFAULT_SIMULATION_CLUSTER_DEGRADATION_THRESHOLD,
                )
            ),
            penalty_overcapacity_per_unit=float(
                simulation_raw.get(
                    "penalty_overcapacity_per_unit",
                    DEFAULT_SIMULATION_PENALTY_OVERCAPACITY_PER_UNIT,
                )
            ),
            penalty_overtime_per_minute=float(
                simulation_raw.get(
                    "penalty_overtime_per_minute",
                    DEFAULT_SIMULATION_PENALTY_OVERTIME_PER_MINUTE,
                )
            ),
        ),
    )
