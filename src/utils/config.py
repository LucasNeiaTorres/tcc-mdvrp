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
class GAConfig:
    pop_size: int
    n_gen: int
    capacity_penalty: float
    crossover_prob: float
    mutation_eta: int
    seed: int


@dataclass
class PSOConfig:
    pop_size: int
    n_gen: int
    inertia: float
    c1: float
    c2: float
    adaptive: bool
    seed: int


@dataclass
class AppConfig:
    ga: GAConfig
    pso: PSOConfig


def load_config(path: Optional[str] = None) -> AppConfig:
    """
    Load and parse config.yaml into an AppConfig dataclass.

    Args:
        path: Path to the YAML file. Defaults to config.yaml in the
              project root (two levels above this file).

    Returns:
        AppConfig with fully typed ga and pso sub-configs.
    """
    if path is None:
        path = str(Path(__file__).parent.parent.parent / "config.yaml")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    ga_raw = raw["ga"]
    pso_raw = raw["pso"]

    return AppConfig(
        ga=GAConfig(
            pop_size=int(ga_raw["pop_size"]),
            n_gen=int(ga_raw["n_gen"]),
            capacity_penalty=float(ga_raw["capacity_penalty"]),
            crossover_prob=float(ga_raw["crossover_prob"]),
            mutation_eta=int(ga_raw["mutation_eta"]),
            seed=int(ga_raw["seed"]),
        ),
        pso=PSOConfig(
            pop_size=int(pso_raw["pop_size"]),
            n_gen=int(pso_raw["n_gen"]),
            inertia=float(pso_raw["inertia"]),
            c1=float(pso_raw["c1"]),
            c2=float(pso_raw["c2"]),
            adaptive=bool(pso_raw["adaptive"]),
            seed=int(pso_raw["seed"]),
        ),
    )
