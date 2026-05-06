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
    ccbc: CCBCConfig
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

    ccbc_raw = raw["ccbc"]
    pso_raw = raw["pso"]

    return AppConfig(
        ccbc=CCBCConfig(
            max_iter=int(ccbc_raw["max_iter"]),
            tol=float(ccbc_raw["tol"]),
            n_starts=int(ccbc_raw["n_starts"]),
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
