"""
Benchmark runner for the MDVRP solver.

Runs a chosen algorithm across all (or selected) Cordeau benchmark instances
and prints a results table comparing algorithm cost against the reference
solution.

Usage
-----
    # Run all p-instances with default config
    python src/benchmark.py

    # Run specific instances
    python src/benchmark.py --instances p01 p02 p05

    # Include pr-instances
    python src/benchmark.py --set pr

    # Run both sets
    python src/benchmark.py --set all
"""

import argparse
import time
from pathlib import Path

from algorithms.ccbc_pso import CCBCPSOAlgorithm
from utils.config import load_config
from utils.converter import load_instance

_P_INSTANCES = [f"p{i:02d}" for i in range(1, 24)]
_PR_INSTANCES = [f"pr{i:02d}" for i in range(1, 11)]

_COL_W = (8, 12, 12, 8, 10, 8)
_HEADERS = ("Instance", "Reference", "Algorithm", "Gap %", "Feasible", "Time (s)")


def _header_line() -> str:
    return "  ".join(h.ljust(w) for h, w in zip(_HEADERS, _COL_W))


def _separator() -> str:
    return "  ".join("-" * w for w in _COL_W)


def _row(name: str, ref: float | None, cost: float, feasible: bool, elapsed: float) -> str:
    ref_str = f"{ref:.2f}" if ref is not None else "N/A"
    gap_str = f"{(cost - ref) / ref * 100:.1f}" if ref is not None else "N/A"
    return "  ".join([
        name.ljust(_COL_W[0]),
        ref_str.ljust(_COL_W[1]),
        f"{cost:.2f}".ljust(_COL_W[2]),
        gap_str.ljust(_COL_W[3]),
        str(feasible).ljust(_COL_W[4]),
        f"{elapsed:.1f}".ljust(_COL_W[5]),
    ])


def run_benchmark(instance_names: list[str]) -> None:
    cfg = load_config()
    algorithm = CCBCPSOAlgorithm(cfg)

    print(f"\nAlgorithm : {algorithm}")
    print(f"Instances : {len(instance_names)}\n")
    print(_header_line())
    print(_separator())

    total_gap = 0.0
    gap_count = 0

    for name in instance_names:
        try:
            loaded = load_instance(name)
        except FileNotFoundError as exc:
            print(f"  {name:<8}  SKIPPED  ({exc})")
            continue

        t0 = time.perf_counter()
        solution = algorithm.solve(loaded.customers, loaded.depots)
        elapsed = time.perf_counter() - t0

        cost = solution.total_cost()
        feasible = solution.is_feasible()
        ref = loaded.reference.objective if loaded.reference is not None else None

        print(_row(name, ref, cost, feasible, elapsed))

        if ref is not None:
            total_gap += (cost - ref) / ref * 100
            gap_count += 1

    print(_separator())
    if gap_count:
        print(f"\nMean gap over {gap_count} instances: {total_gap / gap_count:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MDVRP solver on Cordeau instances.")
    parser.add_argument(
        "--instances",
        nargs="+",
        metavar="NAME",
        help="Specific instance names to run, e.g. p01 p02 pr03.",
    )
    parser.add_argument(
        "--set",
        choices=["p", "pr", "all"],
        default="p",
        help="Which instance set to run when --instances is not provided (default: p).",
    )
    args = parser.parse_args()

    if args.instances:
        names = args.instances
    elif args.set == "pr":
        names = _PR_INSTANCES
    elif args.set == "all":
        names = _P_INSTANCES + _PR_INSTANCES
    else:
        names = _P_INSTANCES

    run_benchmark(names)


if __name__ == "__main__":
    main()
