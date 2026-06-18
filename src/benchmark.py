"""
Benchmark runner for the MDVRP solver.

Runs a chosen algorithm across all (or selected) Cordeau benchmark instances
and prints a results table comparing algorithm cost against the reference
solution.

Usage
-----
    # Run all instances (default)
    python src/benchmark.py

    # Run specific instances
    python src/benchmark.py --instances p01 p02 p05

    # Run only p- or pr-instances
    python src/benchmark.py --set p
    python src/benchmark.py --set pr
"""

import argparse
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from algorithms.ccbc_ga import CCBCGAAlgorithm
from utils.config import load_config
from utils.converter import load_instance

_P_INSTANCES = [f"p{i:02d}" for i in range(1, 24)]
_PR_INSTANCES = [f"pr{i:02d}" for i in range(1, 11)]
_AUTHOR_ORDER = ("Christofides", "Gillett", "Chao", "Cordeau", "Unknown")

# Column widths:  Instance  Cust  Dep  Routes  Reference   Algorithm   Gap %   Feasible  Time(s)
_COL_W = (10, 6, 4, 7, 12, 12, 7, 9, 8)
_HEADERS = ("Instance", "Cust", "Dep", "Routes", "Reference", "Algorithm", "Gap %", "Feasible", "Time(s)")


@dataclass
class _InstanceResult:
    name: str
    n_customers: int
    n_depots: int
    n_routes: int
    ref: float | None
    cost: float
    feasible: bool
    elapsed: float

    @property
    def gap(self) -> float | None:
        if self.ref is None:
            return None
        return (self.cost - self.ref) / self.ref * 100.0


def _header_line() -> str:
    return "  ".join(h.ljust(w) for h, w in zip(_HEADERS, _COL_W))


def _separator() -> str:
    return "  ".join("-" * w for w in _COL_W)


def _row(r: _InstanceResult) -> str:
    ref_str = f"{r.ref:.2f}" if r.ref is not None else "N/A"
    gap_str = f"{r.gap:.1f}" if r.gap is not None else "N/A"
    return "  ".join([
        r.name.ljust(_COL_W[0]),
        str(r.n_customers).ljust(_COL_W[1]),
        str(r.n_depots).ljust(_COL_W[2]),
        str(r.n_routes).ljust(_COL_W[3]),
        ref_str.ljust(_COL_W[4]),
        f"{r.cost:.2f}".ljust(_COL_W[5]),
        gap_str.ljust(_COL_W[6]),
        str(r.feasible).ljust(_COL_W[7]),
        f"{r.elapsed:.1f}".ljust(_COL_W[8]),
    ])


def _author_from_instance(name: str) -> str:
    if name.startswith("pr"):
        return "Cordeau"
    if name.startswith("p"):
        idx = int(name[1:])
        if 1 <= idx <= 7:
            return "Christofides"
        if 8 <= idx <= 11:
            return "Gillett"
        if 12 <= idx <= 23:
            return "Chao"
    return "Unknown"


def _author_sort_key(author: str) -> int:
    try:
        return _AUTHOR_ORDER.index(author)
    except ValueError:
        return len(_AUTHOR_ORDER)


def run_benchmark(instance_names: list[str], cfg, run_label: str | None = None) -> list[_InstanceResult]:
    algorithm = CCBCGAAlgorithm(cfg, debug=False)

    if run_label is not None:
        print(f"\n{run_label}")
    print(f"\nAlgorithm : {algorithm}")
    print(f"Seeds     : ccbc={cfg.ccbc.seed}, ga={cfg.ga.seed}")
    print(f"Instances : {len(instance_names)}\n")
    print(_header_line())
    print(_separator())

    results: list[_InstanceResult] = []

    for name in instance_names:
        try:
            loaded = load_instance(name)
        except FileNotFoundError as exc:
            print(f"  {name:<10}  SKIPPED  ({exc})")
            continue

        t0 = time.perf_counter()
        solution = algorithm.solve(loaded.customers, loaded.depots)
        elapsed = time.perf_counter() - t0

        result = _InstanceResult(
            name=name,
            n_customers=len(loaded.customers),
            n_depots=len(loaded.depots),
            n_routes=len(solution.routes),
            ref=loaded.reference.objective if loaded.reference is not None else None,
            cost=solution.total_cost(),
            feasible=solution.is_feasible(),
            elapsed=elapsed,
        )
        results.append(result)
        print(_row(result))

    print(_separator())
    _print_summary(results)
    return results


def _print_summary(results: list[_InstanceResult]) -> None:
    if not results:
        return

    gaps = [r.gap for r in results if r.gap is not None]
    feasible_count = sum(1 for r in results if r.feasible)
    total_elapsed = sum(r.elapsed for r in results)

    print(f"\n{'Summary':=<50}")
    print(f"  Instances run   : {len(results)}")
    print(f"  Feasible        : {feasible_count} / {len(results)}")
    print(f"  Total time      : {total_elapsed:.1f}s")

    if gaps:
        mean_gap = sum(gaps) / len(gaps)
        best_gap = min(gaps)
        worst_gap = max(gaps)
        variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        std_gap = variance ** 0.5

        print(f"\n  Gap vs reference ({len(gaps)} instances with known optimum):")
        print(f"    Mean  : {mean_gap:+.2f}%")
        print(f"    Best  : {best_gap:+.2f}%  ({next(r.name for r in results if r.gap == best_gap)})")
        print(f"    Worst : {worst_gap:+.2f}%  ({next(r.name for r in results if r.gap == worst_gap)})")
        print(f"    Std   : {std_gap:.2f}%")

        # Per-instance gap sorted best → worst
        print(f"\n  Gap ranking (best → worst):")
        ranked = sorted((r for r in results if r.gap is not None), key=lambda r: r.gap)
        for r in ranked:
            bar_len = max(0, min(40, int(abs(r.gap) / max(abs(worst_gap), 1e-9) * 20)))
            bar = ("+" if r.gap >= 0 else "-") * bar_len
            print(f"    {r.name:<10}  {r.gap:+6.1f}%  {bar}")


def _print_multi_run_summary(all_runs: list[list[_InstanceResult]]) -> None:
    """Print per-instance aggregate metrics across multiple benchmark executions."""
    if not all_runs:
        return

    by_instance: dict[str, list[_InstanceResult]] = defaultdict(list)
    for run_results in all_runs:
        for r in run_results:
            by_instance[r.name].append(r)

    print(f"\n{'Aggregate Summary Across Runs':=<80}")
    print(f"  Runs executed: {len(all_runs)}")
    print("  Per-instance metrics across all executions:")
    print("  Instance     BKS          Best Cost   Mean Cost   Gap %      Feasible  Time (s)")
    print("  -----------  -----------  -----------  -----------  ----------  ---------  --------")
    
    for name in sorted(by_instance):
        rows = by_instance[name]
        gaps = [r.gap for r in rows if r.gap is not None]
        
        # Best cost across all runs
        best_cost = min(r.cost for r in rows)
        
        # Mean cost across all runs
        mean_cost = sum(r.cost for r in rows) / len(rows)
        
        # Mean gap across all runs
        mean_gap = sum(gaps) / len(gaps) if gaps else None
        mean_gap_str = f"{mean_gap:+.2f}%" if mean_gap is not None else "N/A"
        
        # Feasibility rate
        feasible_count = sum(1 for r in rows if r.feasible)
        feasible_pct = (100.0 * feasible_count / len(rows))
        
        # Average time
        avg_time = sum(r.elapsed for r in rows) / len(rows)
        
        # Reference (BKS)
        bks_str = f"{rows[0].ref:.2f}" if rows[0].ref is not None else "N/A"
        
        print(f"  {name:<11}  {bks_str:<11}  {best_cost:<11.2f}  {mean_cost:<11.2f}  {mean_gap_str:<10}  {feasible_pct:>6.1f}%  {avg_time:>7.2f}")

    by_author: dict[str, list[_InstanceResult]] = defaultdict(list)
    for rows in by_instance.values():
        author = _author_from_instance(rows[0].name)
        by_author[author].extend(rows)

    print("\n  Author summary (for table-resumo):")
    print("  Author         Gap Mean %   Gap Median % Feasible %   Time Mean(s) #Inst")
    print("  -------------  -----------  ------------ ----------   ------------ -----")
    for author in sorted(by_author, key=_author_sort_key):
        rows = by_author[author]
        gaps = [r.gap for r in rows if r.gap is not None]
        mean_gap = sum(gaps) / len(gaps) if gaps else None
        median_gap = median(gaps) if gaps else None
        feasible_pct = 100.0 * sum(1 for r in rows if r.feasible) / len(rows)
        mean_time = sum(r.elapsed for r in rows) / len(rows)
        n_inst = len({r.name for r in rows})

        mean_gap_str = f"{mean_gap:+.2f}" if mean_gap is not None else "N/A"
        med_gap_str = f"{median_gap:+.2f}" if median_gap is not None else "N/A"
        print(f"  {author:<13}  {mean_gap_str:>11}  {med_gap_str:>12} {feasible_pct:>9.1f}%   {mean_time:>12.2f} {n_inst:>5}")


def _save_results_csv(all_runs: list[list[_InstanceResult]], output_file: str) -> None:
    """Save aggregated benchmark results to CSV file for easy table population."""
    if not all_runs:
        return

    by_instance: dict[str, list[_InstanceResult]] = defaultdict(list)
    for run_results in all_runs:
        for r in run_results:
            by_instance[r.name].append(r)

    with open(output_file, "w") as f:
        # Header
        f.write("Instance,Author,BKS,Best,Mean,Gap(%),Feasible(%),Time(s),Customers,Depots\n")

        # Data rows
        for name in sorted(by_instance):
            rows = by_instance[name]
            gaps = [r.gap for r in rows if r.gap is not None]

            best_cost = min(r.cost for r in rows)
            mean_cost = sum(r.cost for r in rows) / len(rows)
            mean_gap = sum(gaps) / len(gaps) if gaps else None

            feasible_count = sum(1 for r in rows if r.feasible)
            feasible_pct = 100.0 * feasible_count / len(rows)

            avg_time = sum(r.elapsed for r in rows) / len(rows)

            bks_str = f"{rows[0].ref:.2f}" if rows[0].ref is not None else "N/A"
            mean_gap_val = f"{mean_gap:.2f}" if mean_gap is not None else "N/A"
            author = _author_from_instance(name)

            n_customers = rows[0].n_customers
            n_depots = rows[0].n_depots

            f.write(
                f"{name},{author},{bks_str},{best_cost:.2f},{mean_cost:.2f},{mean_gap_val},{feasible_pct:.1f},{avg_time:.2f},{n_customers},{n_depots}\n"
            )

    print(f"\n  Results saved to: {output_file}")


def _save_author_summary_csv(all_runs: list[list[_InstanceResult]], output_file: str) -> None:
    """Save author-level aggregate metrics for direct use in the summary table."""
    if not all_runs:
        return

    by_instance: dict[str, list[_InstanceResult]] = defaultdict(list)
    for run_results in all_runs:
        for r in run_results:
            by_instance[r.name].append(r)

    by_author: dict[str, list[_InstanceResult]] = defaultdict(list)
    for rows in by_instance.values():
        author = _author_from_instance(rows[0].name)
        by_author[author].extend(rows)

    with open(output_file, "w") as f:
        f.write("Author,GapMean(%),GapMedian(%),FeasibleMean(%),TimeMean(s),NumInstances\n")
        for author in sorted(by_author, key=_author_sort_key):
            rows = by_author[author]
            gaps = [r.gap for r in rows if r.gap is not None]
            mean_gap = sum(gaps) / len(gaps) if gaps else None
            median_gap = median(gaps) if gaps else None
            feasible_pct = 100.0 * sum(1 for r in rows if r.feasible) / len(rows)
            mean_time = sum(r.elapsed for r in rows) / len(rows)
            n_inst = len({r.name for r in rows})

            mean_gap_val = f"{mean_gap:.2f}" if mean_gap is not None else "N/A"
            median_gap_val = f"{median_gap:.2f}" if median_gap is not None else "N/A"
            f.write(f"{author},{mean_gap_val},{median_gap_val},{feasible_pct:.1f},{mean_time:.2f},{n_inst}\n")

        all_rows = [r for rows in by_author.values() for r in rows]
        all_gaps = [r.gap for r in all_rows if r.gap is not None]
        all_mean_gap = sum(all_gaps) / len(all_gaps) if all_gaps else None
        all_median_gap = median(all_gaps) if all_gaps else None
        all_feasible_pct = 100.0 * sum(1 for r in all_rows if r.feasible) / len(all_rows)
        all_mean_time = sum(r.elapsed for r in all_rows) / len(all_rows)
        all_n_inst = len({r.name for r in all_rows})

        all_mean_gap_val = f"{all_mean_gap:.2f}" if all_mean_gap is not None else "N/A"
        all_median_gap_val = f"{all_median_gap:.2f}" if all_median_gap is not None else "N/A"
        f.write(
            f"Global,{all_mean_gap_val},{all_median_gap_val},{all_feasible_pct:.1f},{all_mean_time:.2f},{all_n_inst}\n"
        )

    print(f"  Author summary saved to: {output_file}")


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
        default="all",
        help="Which instance set to run when --instances is not provided (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override both ccbc.seed and ga.seed for deterministic runs.",
    )
    parser.add_argument(
        "--seed-runs",
        type=int,
        default=1,
        help="Run benchmark N times with random seeds (default: 1).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        metavar="FILE",
        help="Save aggregated results to CSV file.",
    )
    parser.add_argument(
        "--output-author-csv",
        type=str,
        metavar="FILE",
        help="Save author-level summary metrics to CSV file.",
    )
    args = parser.parse_args()

    if args.instances:
        names = args.instances
    elif args.set == "pr":
        names = _PR_INSTANCES
    elif args.set == "p":
        names = _P_INSTANCES
    else:
        names = _P_INSTANCES + _PR_INSTANCES

    if args.seed_runs < 1:
        raise ValueError("--seed-runs must be >= 1")

    if args.seed_runs == 1:
        cfg = load_config()
        if args.seed is not None:
            cfg.ccbc.seed = args.seed
            cfg.ga.seed = args.seed
        run_benchmark(names, cfg)
        return

    if args.seed is not None:
        print("[warning] --seed is ignored when --seed-runs > 1")

    seed_rng = random.SystemRandom()
    all_runs: list[list[_InstanceResult]] = []
    for run_idx in range(1, args.seed_runs + 1):
        seed = seed_rng.randint(1, 1_000_000)
        cfg = load_config()
        cfg.ccbc.seed = seed
        cfg.ga.seed = seed
        run_results = run_benchmark(names, cfg, run_label=f"Run {run_idx}/{args.seed_runs} | seed={seed}")
        all_runs.append(run_results)

    _print_multi_run_summary(all_runs)
    if args.output_csv:
        _save_results_csv(all_runs, args.output_csv)
    if args.output_author_csv:
        _save_author_summary_csv(all_runs, args.output_author_csv)


if __name__ == "__main__":
    main()
