"""
dynamic_benchmark.py
────────────────────
Orquestrador de experimentos dinâmicos para o D-MDVRP.

Pipeline:
  1. Descobre routes-files pré-calculados em data/processed/results/
  2. Gera cenários de falha (Collapse Zones) variando DOD × EDOD × seed
  3. Executa run_simulation para cada combinação
  4. Lê o summary JSON gerado pelo simulador e extrai métricas
  5. Agrega tudo num DataFrame pandas e exporta tabelas para CSV + LaTeX

Uso:
    cd <repo_root>
    python src/dynamic_benchmark.py [--results-dir PATH] [--output-dir PATH]
                                     [--stages 1 2 3] [--seeds 5]
                                     [--dod 0.05 0.10 0.20 0.40]
                                     [--edod 0.25 0.5 0.75]
                                     [--dry-run]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# ── resolve import paths ──────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from algorithms.ccbc_ga import CCBCGAAlgorithm
from scenario.generate_failures import generate_collapse_zones
from scenario.models import FailureEvent
from scenario.simulator import SIMULATION_LOG_DIR, run_simulation
from utils.config import load_config
from utils.converter import load_instance
from utils.data_loader import read_cordeau_data_file, read_json_solution_file
from utils.results_io import load_routing_result_as_solution

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")

# ── constants ─────────────────────────────────────────────────────────────────
_RESULTS_DIR = _REPO_ROOT / "data" / "processed" / "results"
_OUTPUT_DIR = _REPO_ROOT / "data" / "processed" / "analysis"
_FAILURES_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "failures" / "benchmark"
_CORDEAU_DIR = _REPO_ROOT / "data" / "raw" / "cordeau"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    dod_levels: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.40])
    edod_levels: list[float] = field(default_factory=lambda: [0.25, 0.50, 0.75])
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    enabled_stages: frozenset[int] = field(default_factory=lambda: frozenset({1, 2, 3}))
    results_dir: Path = field(default_factory=lambda: _RESULTS_DIR)
    output_dir: Path = field(default_factory=lambda: _OUTPUT_DIR)
    failures_cache_dir: Path = field(default_factory=lambda: _FAILURES_CACHE_DIR)
    dry_run: bool = False
    suppress_sim_output: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Route-file discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_routes_files(results_dir: Path) -> list[Path]:
    """Return all *_routes.json files found in results_dir."""
    files = sorted(results_dir.glob("*_routes.json"))
    if not files:
        raise FileNotFoundError(f"No *_routes.json files found in {results_dir}")
    log.info("Discovered %d routes files in %s", len(files), results_dir)
    return files


def instance_name_from_routes_file(routes_file: Path) -> str:
    """Extract the instance name (e.g. 'p01') from 'p01_routes.json'."""
    return routes_file.stem.replace("_routes", "")


# ─────────────────────────────────────────────────────────────────────────────
# Failure generation
# ─────────────────────────────────────────────────────────────────────────────

def _failures_cache_path(
    cache_dir: Path,
    instance: str,
    dod: float,
    edod: float,
    seed: int,
) -> Path:
    tag = f"{instance}_dod{int(dod*100):03d}_edod{int(edod*100):03d}_s{seed}"
    return cache_dir / f"{tag}.json"


def generate_or_load_failures(
    instance: str,
    routes_file: Path,
    dod: float,
    edod: float,
    seed: int,
    cache_dir: Path,
    max_time: float = 120.0,
) -> tuple[list[FailureEvent], dict[str, Any]]:
    """
    Generate Collapse-Zones failures for (instance, dod, edod, seed).
    Results are cached to disk so repeated runs are fast.

    Returns (events, metadata).
    """
    cache_path = _failures_cache_path(cache_dir, instance, dod, edod, seed)
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        events = [
            FailureEvent(
                trigger_time=e["trigger_time"],
                type=e["type"],
                node_a=e["node_a"],
                node_b=e["node_b"],
            )
            for e in raw["events"]
        ]
        return events, raw["metadata"]

    # Load data
    data_file = _CORDEAU_DIR / instance
    raw_instance = read_cordeau_data_file(str(data_file))
    solution = read_json_solution_file(str(routes_file))
    rng = random.Random(seed)

    events = generate_collapse_zones(
        routes=solution,
        G=raw_instance,
        DOD=dod,
        EDOD=edod,
        max_time=max_time,
        rng=rng,
    )

    # Compute realised EDOD from the generated trigger times
    if events:
        mean_t = sum(e.trigger_time for e in events) / len(events)
        realised_edod = mean_t / max_time
    else:
        realised_edod = 0.0

    metadata = {
        "instance": instance,
        "seed": seed,
        "dod": dod,
        "edod_target": edod,
        "edod_realised": round(realised_edod, 4),
        "n_events": len(events),
        "max_time": max_time,
    }

    # Persist cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "events": [
            {"trigger_time": e.trigger_time, "type": e.type,
             "node_a": e.node_a, "node_b": e.node_b}
            for e in events
        ],
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    return events, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def _read_summary(run_tag: str) -> dict[str, Any]:
    """Read the summary JSON written by run_simulation for run_tag."""
    path = SIMULATION_LOG_DIR / f"{run_tag}_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def run_one_experiment(
    instance: str,
    routes_file: Path,
    failures: list[FailureEvent],
    failure_meta: dict[str, Any],
    cfg: Any,
    algorithm: Any,
    enabled_stages: frozenset[int],
    suppress_output: bool = True,
) -> dict[str, Any]:
    """
    Run a single simulation experiment and return a flat metrics dict.

    The run_tag is unique per (instance, dod, edod, seed) so summary files
    never overwrite each other across experiments.
    """
    dod = failure_meta["dod"]
    edod_t = failure_meta["edod_target"]
    edod_r = failure_meta["edod_realised"]
    seed = failure_meta["seed"]
    n_events = failure_meta["n_events"]

    run_tag = (
        f"{instance}"
        f"_dod{int(dod*100):03d}"
        f"_edod{int(edod_t*100):03d}"
        f"_s{seed}"
    )

    # Load baseline solution
    loaded = load_instance(instance)
    solution = load_routing_result_as_solution(
        routes_file, loaded.customers, loaded.depots
    )

    wall_t0 = time.perf_counter()

    sink = io.StringIO() if suppress_output else sys.stdout
    with contextlib.redirect_stdout(sink):
        sim_solution, _ = run_simulation(
            initial_solution=solution,
            failures=failures,
            instance_name=run_tag,
            algorithm=algorithm,
            cfg=cfg,
            enabled_stages=enabled_stages,
            save_reroute_files=False,
        )

    wall_elapsed = time.perf_counter() - wall_t0

    # Read metrics from summary JSON
    summary = _read_summary(run_tag)

    # ── flatten metrics ──────────────────────────────────────────────────────
    rb = summary.get("reroute_by_stage", {})
    rt = summary.get("reroute_time_by_stage_s", {})
    ra = summary.get("reroute_avg_time_by_stage_s", {})
    cap_of = summary.get("capacity_overflow", {})
    dur_of = summary.get("duration_overflow", {})
    total_c = summary.get("total_customers", 0)
    reroute_count = summary.get("reroute_count", 0)
    s1 = rb.get("stage1", 0)
    s2 = rb.get("stage2", 0)
    s3 = rb.get("stage3", 0)

    # Stage utilisation rates (% of reroute events resolved by each stage)
    s1_rate = 100.0 * s1 / reroute_count if reroute_count > 0 else 0.0
    s2_rate = 100.0 * s2 / reroute_count if reroute_count > 0 else 0.0
    s3_rate = 100.0 * s3 / reroute_count if reroute_count > 0 else 0.0

    orig_cost = summary.get("original_solution_cost", float("nan"))
    dyn_cost  = summary.get("post_reroute_cost_without_wasted",
                            summary.get("post_reroute_cost", float("nan")))
    cost_deg  = ((dyn_cost - orig_cost) / orig_cost * 100.0
                 if orig_cost and orig_cost > 0 else float("nan"))

    unserved_count   = summary.get("unserved_count", 0)
    unserved_rate    = summary.get("unserved_rate_percent", 0.0)

        # Count edge_block events that actually intercepted a vehicle
    n_triggered_edges = summary.get("n_disasters_triggered", 0)


    return {
        # ── experiment identifiers ──────────────────────────────────────────
        "instance":         instance,
        "dod":              dod,
        "edod_target":      edod_t,
        "edod_realised":    edod_r,
        "seed":             seed,
        "n_edges_blocked":  n_events,
        "n_triggered_edges":n_triggered_edges,
        "enabled_stages":   sorted(enabled_stages),
        # ── timing ─────────────────────────────────────────────────────────
        "wall_time_total_s":         round(wall_elapsed, 4),
        "reroute_time_total_s":      summary.get("reroute_time_total_s", float("nan")),
        "reroute_time_s1_s":         rt.get("stage1", float("nan")),
        "reroute_time_s2_s":         rt.get("stage2", float("nan")),
        "reroute_time_s3_s":         rt.get("stage3", float("nan")),
        "reroute_avg_s1_s":          ra.get("stage1", float("nan")),
        "reroute_avg_s2_s":          ra.get("stage2", float("nan")),
        "reroute_avg_s3_s":          ra.get("stage3", float("nan")),
        "reroute_avg_total_s":       summary.get("reroute_avg_time_total_s", float("nan")),
        # ── cost / quality ──────────────────────────────────────────────────
        "cost_static":               orig_cost,
        "cost_dynamic":              dyn_cost,
        "cost_degradation_pct":      cost_deg,
        "wasted_distance":           summary.get("wasted_travel_distance", float("nan")),
        "total_execution_time_min":  summary.get("total_execution_time_minutes", float("nan")),
        # ── reroute hierarchy ───────────────────────────────────────────────
        "reroute_count":             reroute_count,
        "reroutes_s1":               s1,
        "reroutes_s2":               s2,
        "reroutes_s3":               s3,
        "utilisation_s1_pct":        round(s1_rate, 2),
        "utilisation_s2_pct":        round(s2_rate, 2),
        "utilisation_s3_pct":        round(s3_rate, 2),
        # ── unserved / infeasibility ────────────────────────────────────────
        "unserved_count":            unserved_count,
        "unserved_rate_pct":         unserved_rate,
        "total_customers":           total_c,
        "unserved_stage3_fallback":  summary.get("unserved_stage3_fallback_count", 0),
        "capacity_violations":       cap_of.get("routes_count", 0),
        "capacity_excess_pct":       cap_of.get("excess_vs_limit_percent", 0.0),
        "duration_violations":       dur_of.get("routes_count", 0),
        "duration_excess_pct":       dur_of.get("excess_vs_limit_percent", 0.0),
        # ── feasibility flags ───────────────────────────────────────────────
        "fully_feasible":            summary.get("fully_feasible", False),
        "feasible_hard":             summary.get("feasible_hard_constraints", False),
        "feasible_soft":             summary.get("feasible_soft_constraints", False),
        "operation_verdict":         summary.get("operation_verdict", "UNKNOWN"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(cfg_bench: BenchmarkConfig) -> pd.DataFrame:
    """
    Iterate over all (routes_file × DOD × EDOD × seed) combinations,
    run the simulation, collect results and return a DataFrame.
    """
    cfg_app  = load_config()
    algorithm = CCBCGAAlgorithm(cfg_app, debug=False)

    routes_files = discover_routes_files(cfg_bench.results_dir)

    total = (
        len(routes_files)
        * len(cfg_bench.dod_levels)
        * len(cfg_bench.edod_levels)
        * len(cfg_bench.seeds)
    )
    log.info(
        "Experiments to run: %d instances × %d DOD × %d EDOD × %d seeds = %d",
        len(routes_files), len(cfg_bench.dod_levels),
        len(cfg_bench.edod_levels), len(cfg_bench.seeds), total,
    )

    records: list[dict[str, Any]] = []
    n_done = 0
    n_errors = 0

    for routes_file in routes_files:
        instance = instance_name_from_routes_file(routes_file)

        for dod in cfg_bench.dod_levels:
            for edod in cfg_bench.edod_levels:
                for seed in cfg_bench.seeds:
                    n_done += 1
                    label = (
                        f"[{n_done}/{total}] "
                        f"{instance} DOD={dod:.2f} EDOD={edod:.2f} seed={seed}"
                    )

                    try:
                        # Generate / load failure scenario
                        failures, fail_meta = generate_or_load_failures(
                            instance=instance,
                            routes_file=routes_file,
                            dod=dod,
                            edod=edod,
                            seed=seed,
                            cache_dir=cfg_bench.failures_cache_dir,
                        )

                        if cfg_bench.dry_run:
                            log.info("DRY-RUN  %s  (%d events)", label, len(failures))
                            continue

                        log.info("Running  %s  (%d events)", label, len(failures))
                        metrics = run_one_experiment(
                            instance=instance,
                            routes_file=routes_file,
                            failures=failures,
                            failure_meta=fail_meta,
                            cfg=cfg_app,
                            algorithm=algorithm,
                            enabled_stages=cfg_bench.enabled_stages,
                            suppress_output=cfg_bench.suppress_sim_output,
                        )
                        records.append(metrics)

                    except Exception as exc:
                        n_errors += 1
                        log.error("FAILED   %s: %s", label, exc)
                        log.debug(traceback.format_exc())
                        records.append({
                            "instance": instance,
                            "dod": dod,
                            "edod_target": edod,
                            "seed": seed,
                            "error": str(exc),
                        })

    log.info(
        "Batch complete: %d succeeded, %d failed.",
        len([r for r in records if "error" not in r]),
        n_errors,
    )
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables (pivot for academic output)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ms(m: Any, s: Any, digits: int = 3) -> str:
    """Format a mean±std pair as a LaTeX-ready string."""
    if pd.isna(m):
        return "--"
    fmt = f"{{:.{digits}f}}"
    m_str = fmt.format(m)
    if pd.isna(s) or float(s) == 0.0:
        return m_str
    return f"{m_str} $\\pm${fmt.format(s)}"


def _pivot_ms(
    df: pd.DataFrame,
    values: list[str],
    index: str | list[str],
    columns: str | None = None,
    digits: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute a pivot table and return (mean_df, display_df).

    mean_df   — numeric DataFrame saved to CSV.
    display_df — string DataFrame with "X.XXX $\\pm$Y.YYY" cells for LaTeX.
    """
    kw: dict[str, Any] = dict(values=values, index=index)
    if columns:
        kw["columns"] = columns
    mean_df = pd.pivot_table(df, aggfunc="mean", **kw).round(digits)
    std_df  = pd.pivot_table(df, aggfunc="std",  **kw).round(digits)
    display_df = mean_df.copy().astype(object)
    for col in mean_df.columns:
        display_df[col] = [
            _fmt_ms(m, s, digits)
            for m, s in zip(mean_df[col], std_df[col])
        ]
    return mean_df, display_df


# Type alias for the table dict returned by generate_summary_tables
SummaryTables = dict[str, tuple[pd.DataFrame, pd.DataFrame]]


def generate_summary_tables(df: pd.DataFrame) -> SummaryTables:
    """
    Generate pivot tables ready for academic publication.

    Returns a dict mapping table_name → (mean_df, display_df) where:
      mean_df    — numeric values (saved to CSV)
      display_df — "mean ±std" string cells (printed as LaTeX)
    """
    clean = df[df.get("error", pd.Series(dtype=str)).isna()].copy() if "error" in df.columns else df.copy()

    tables: SummaryTables = {}

    # ── Table 1: DOD × EDOD impact on unserved customers and cost degradation ─
    mean, disp = _pivot_ms(
        clean,
        values=["unserved_count", "cost_degradation_pct"],
        index="dod",
        columns="edod_target",
    )
    mean.index.name = disp.index.name = "DOD"
    mean.columns.names = disp.columns.names = ["Metric", "EDOD"]
    tables["t1_dod_x_edod_unserved_cost"] = (mean, disp)

    # ── Table 2: Stage utilisation frequency by DOD ────────────────────────
    mean, disp = _pivot_ms(
        clean,
        values=["reroutes_s1", "reroutes_s2", "reroutes_s3",
                "utilisation_s1_pct", "utilisation_s2_pct", "utilisation_s3_pct"],
        index="dod",
    )
    mean.index.name = disp.index.name = "DOD"
    tables["t2_stage_utilisation_by_dod"] = (mean, disp)

    # ── Table 3: Mean computational times S1 vs S3 by DOD ─────────────────
    mean, disp = _pivot_ms(
        clean,
        values=["reroute_avg_s1_s", "reroute_avg_s3_s",
                "reroute_time_total_s", "reroute_avg_total_s"],
        index="dod",
        columns="edod_target",
        digits=6,
    )
    mean.index.name = disp.index.name = "DOD"
    mean.columns.names = disp.columns.names = ["Timing metric", "EDOD"]
    tables["t3_timing_s1_vs_s3"] = (mean, disp)

    # ── Table 4: Unserved rate and feasibility by instance and DOD ─────────
    vals4 = ["unserved_rate_pct", "cost_degradation_pct", "wasted_distance"]
    mean4 = pd.pivot_table(clean, values=vals4, index=["instance", "dod"], aggfunc="mean").round(3)
    std4  = pd.pivot_table(clean, values=vals4, index=["instance", "dod"], aggfunc="std").round(3)
    disp4 = mean4.copy().astype(object)
    for col in mean4.columns:
        disp4[col] = [_fmt_ms(m, s) for m, s in zip(mean4[col], std4[col])]
    tables["t4_quality_by_instance_dod"] = (mean4, disp4)

    # ── Table 5: Triggered edges and resolution overview ──────────────────
    vals5 = ["n_edges_blocked", "n_triggered_edges", "reroute_count",
             "reroutes_s1", "reroutes_s2", "reroutes_s3", "unserved_count"]
    mean, disp = _pivot_ms(clean, values=vals5, index="dod")
    mean.index.name = disp.index.name = "DOD"
    tables["t5_event_resolution_overview"] = (mean, disp)

    # ── Table 6: Wall-clock time vs DOD (latency budget insight) ──────────
    mean6 = pd.pivot_table(clean, values="wall_time_total_s", index="dod",
                           columns="instance", aggfunc="mean").round(2)
    std6  = pd.pivot_table(clean, values="wall_time_total_s", index="dod",
                           columns="instance", aggfunc="std").round(2)
    disp6 = mean6.copy().astype(object)
    for col in mean6.columns:
        disp6[col] = [_fmt_ms(m, s, 2) for m, s in zip(mean6[col], std6[col])]
    mean6.index.name = disp6.index.name = "DOD"
    tables["t6_wall_time_by_instance_dod"] = (mean6, disp6)

    return tables


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_latex(df: pd.DataFrame, name: str) -> str:
    """
    Generate a LaTeX tabular string without requiring jinja2.
    Handles flat and MultiIndex columns with proper \\multicolumn headers.
    Cells may be pre-formatted strings (e.g. "3.142 $\\pm$0.051") or numerics.
    """
    caption = name.replace("_", " ").title()
    label = f"tab:{name}"

    def _fmt(v: Any) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, float):
            return f"{v:.3f}" if not pd.isna(v) else "--"
        return str(v)

    if isinstance(df.index, pd.MultiIndex):
        idx_labels = [" / ".join(str(i) for i in idx) for idx in df.index]
    else:
        idx_labels = [str(i) for i in df.index]

    n_idx = df.index.nlevels
    is_multi_col = isinstance(df.columns, pd.MultiIndex)

    if is_multi_col:
        top = [str(c[0]) for c in df.columns]
        bot = [str(c[1]) for c in df.columns]
        col_spec = "l" * n_idx + "r" * len(bot)

        groups: list[tuple[str, int]] = []
        for lbl in top:
            if groups and groups[-1][0] == lbl:
                groups[-1] = (lbl, groups[-1][1] + 1)
            else:
                groups.append((lbl, 1))

        mc_cells = [""] * n_idx + [
            f"\\multicolumn{{{span}}}{{c}}{{{lbl}}}" for lbl, span in groups
        ]
        header_top = " & ".join(mc_cells) + r" \\"
        header_bot = " & ".join([""] * n_idx + bot) + r" \\"
        cmidrule = f"\\cmidrule(lr){{{n_idx+1}-{n_idx+len(bot)}}}"
    else:
        col_labels = [str(c) for c in df.columns]
        col_spec = "l" * n_idx + "r" * len(col_labels)
        header_top = None
        header_bot = " & ".join([""] * n_idx + col_labels) + r" \\"
        cmidrule = None

    lines_out = [
        r"\begin{table}[ht]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]
    if header_top:
        lines_out.append(f"    {header_top}")
        lines_out.append(f"    {cmidrule}")
    lines_out.append(f"    {header_bot}")
    lines_out.append(r"    \midrule")

    for label_val, row in zip(idx_labels, df.itertuples(index=False)):
        cells = [label_val] + [_fmt(v) for v in row]
        lines_out.append("    " + " & ".join(cells) + r" \\")

    lines_out += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines_out)


def export_tables(
    tables: SummaryTables,
    output_dir: Path,
    latex: bool = True,
) -> None:
    """Save mean tables as CSV and print LaTeX (mean ± std) to stdout."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (mean_tbl, display_tbl) in tables.items():
        csv_path = output_dir / f"{name}.csv"
        mean_tbl.to_csv(csv_path)
        log.info("Saved CSV: %s", csv_path)

        if latex:
            print(f"\n{'='*70}")
            print(f"  {name}")
            print(f"{'='*70}")
            print(_df_to_latex(display_tbl, name))


def export_raw(df: pd.DataFrame, output_dir: Path) -> None:
    """Export the full raw results DataFrame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "raw_results.csv"
    df.to_csv(path, index=False)
    log.info("Raw results saved to %s (%d rows)", path, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# Quick descriptive stats
# ─────────────────────────────────────────────────────────────────────────────

def print_descriptive_stats(df: pd.DataFrame) -> None:
    clean = df[~df.get("error", pd.Series(dtype=str)).notna()].copy() if "error" in df.columns else df.copy()
    numeric_cols = [
        "cost_degradation_pct", "unserved_count", "unserved_rate_pct",
        "reroute_count", "reroutes_s1", "reroutes_s2", "reroutes_s3",
        "reroute_time_total_s", "reroute_avg_total_s",
        "reroute_avg_s1_s", "reroute_avg_s3_s",
        "wasted_distance",
    ]
    present = [c for c in numeric_cols if c in clean.columns]
    print("\n" + "="*60)
    print("  Descriptive statistics (all clean runs)")
    print("="*60)
    print(clean[present].describe().round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D-MDVRP dynamic benchmark orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir", type=Path, default=_RESULTS_DIR,
        help="Directory containing *_routes.json files",
    )
    p.add_argument(
        "--output-dir", type=Path, default=_OUTPUT_DIR,
        help="Output directory for CSV and LaTeX tables",
    )
    p.add_argument(
        "--dod", nargs="+", type=float,
        default=[0.05, 0.10, 0.20, 0.40],
        metavar="D",
        help="DOD levels to test",
    )
    p.add_argument(
        "--edod", nargs="+", type=float,
        default=[0.25, 0.50, 0.75],
        metavar="E",
        help="EDOD levels to test",
    )
    p.add_argument(
        "--seeds", type=int, default=5,
        help="Number of random seeds per combination",
    )
    p.add_argument(
        "--seed-start", type=int, default=42,
        help="First seed; subsequent seeds are seed_start + i*111",
    )
    p.add_argument(
        "--stages", nargs="+", type=int, choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Contingency stages to enable",
    )
    p.add_argument(
        "--no-latex", action="store_true",
        help="Skip printing LaTeX tables to stdout",
    )
    p.add_argument(
        "--verbose-sim", action="store_true",
        help="Do not suppress simulator output (very verbose)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Generate failure files and print plan, but do not run simulations",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    seeds = [args.seed_start + i * 111 for i in range(args.seeds)]

    cfg = BenchmarkConfig(
        dod_levels=sorted(set(args.dod)),
        edod_levels=sorted(set(args.edod)),
        seeds=seeds,
        enabled_stages=frozenset(args.stages),
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        suppress_sim_output=not args.verbose_sim,
    )

    df = run_batch(cfg)

    if cfg.dry_run:
        log.info("Dry-run complete — no simulations executed.")
        return

    export_raw(df, cfg.output_dir)
    print_descriptive_stats(df)

    tables = generate_summary_tables(df)
    export_tables(tables, cfg.output_dir, latex=not args.no_latex)


if __name__ == "__main__":
    main()
