"""
Configuration loader for the MDVRP solver.

Reads config.yaml from the project root and returns typed dataclasses
so all algorithm modules receive validated, IDE-completable parameters.
"""

from dataclasses import dataclass
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
    local_search_max_iterations: int
    clone_delta: float
    stagnation_period: int
    stagnation_ftol: float
    time_limit: str


@dataclass
class AppConfig:
    ccbc: CCBCConfig
    ga: GAConfig


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
            local_search_max_iterations=int(ga_raw["local_search_max_iterations"]),
            clone_delta=float(ga_raw["clone_delta"]),
            stagnation_period=int(ga_raw["stagnation_period"]),
            stagnation_ftol=float(ga_raw["stagnation_ftol"]),
            time_limit=str(ga_raw["time_limit"]),
        ),
    )
