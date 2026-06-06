"""
Configuration loader for the MDVRP solver.

Reads config.yaml from the project root and returns typed dataclasses
so all algorithm modules receive validated, IDE-completable parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


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


@dataclass
class SimulationConfig:
    reroute_degradation_threshold: float = 1.20
    cluster_degradation_threshold: float = 1.05
    penalty_overcapacity_per_unit: float = 100000.0
    penalty_overtime_per_minute: float = 50000.0


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
        ),
        simulation=SimulationConfig(
            reroute_degradation_threshold=float(
                simulation_raw.get("reroute_degradation_threshold", 1.20)
            ),
            cluster_degradation_threshold=float(
                simulation_raw.get("cluster_degradation_threshold", 1.05)
            ),
            penalty_overcapacity_per_unit=float(
                simulation_raw.get(
                    "penalty_overcapacity_per_unit",
                    simulation_raw.get("penalty_overcapacity_per_unit", 100000.0),
                )
            ),
            penalty_overtime_per_minute=float(
                simulation_raw.get(
                    "penalty_overtime_per_minute",
                    simulation_raw.get("penalty_overtime_per_minute", 50000.0),
                )
            ),
        ),
    )
