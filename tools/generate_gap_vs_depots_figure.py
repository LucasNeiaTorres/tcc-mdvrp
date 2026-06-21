#!/usr/bin/env python3
"""Generate scatter plot of average gap (%) versus number of depots.

Expected input CSV columns (from benchmark output):
- Instance
- Gap(%)
- Depots

Usage:
    python3 tools/generate_gap_vs_depots_figure.py \
        --input resultados_instancias.csv \
        --output figura-gap-vs-depots.pdf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_points(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    depots: list[float] = []
    gaps: list[float] = []
    names: list[str] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Instance", "Gap(%)", "Depots"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            gap_raw = (row.get("Gap(%)") or "").strip()
            dep_raw = (row.get("Depots") or "").strip()
            if not gap_raw or gap_raw.upper() == "N/A" or not dep_raw:
                continue

            depots.append(float(dep_raw))
            gaps.append(float(gap_raw))
            names.append((row.get("Instance") or "").strip())

    if not depots:
        raise ValueError("No valid rows found in CSV (Gap(%) and Depots are required).")

    return np.array(depots, dtype=float), np.array(gaps, dtype=float), names


def _plot_gap_vs_depots(
    depots: np.ndarray,
    gaps: np.ndarray,
    names: list[str],
    output_path: Path,
    title: str,
    annotate: bool,
    show_global_trend: bool,
    show_error_bars: bool,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    # Raw instance points
    ax.scatter(
        depots,
        gaps,
        s=44,
        color="steelblue",
        alpha=0.45,
        edgecolor="white",
        linewidth=0.5,
        label="Instâncias",
    )

    # Mean gap per depot count and dispersion
    unique_depots = np.array(sorted(set(int(v) for v in depots.tolist())), dtype=float)
    mean_gap = np.array([gaps[depots == d].mean() for d in unique_depots], dtype=float)
    std_gap = np.array([gaps[depots == d].std(ddof=0) for d in unique_depots], dtype=float)

    if show_error_bars:
        ax.errorbar(
            unique_depots,
            mean_gap,
            yerr=std_gap,
            fmt="-o",
            color="crimson",
            ecolor="crimson",
            elinewidth=1.4,
            capsize=4,
            markersize=5,
            linewidth=2.0,
            label="Média por número de depósitos ± DP",
        )
    else:
        ax.plot(
            unique_depots,
            mean_gap,
            "-o",
            color="crimson",
            linewidth=2.0,
            markersize=5,
            label="Média por número de depósitos",
        )

    # Optional global linear trend (visual aid)
    if show_global_trend and len(depots) >= 2:
        coeffs = np.polyfit(depots, gaps, deg=1)
        x_line = np.linspace(depots.min(), depots.max(), 200)
        y_line = coeffs[0] * x_line + coeffs[1]
        ax.plot(x_line, y_line, linestyle="--", linewidth=1.7, color="darkorange", label="Tendência linear global")

    if annotate:
        for x, y, name in zip(depots, gaps, names):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Número de depósitos", fontsize=11)
    ax.set_ylabel("Gap médio (%)", fontsize=11)
    ax.grid(True, alpha=0.25)

    # Integer ticks for depot counts
    ticks = sorted(set(int(v) for v in depots.tolist()))
    ax.set_xticks(ticks)

    if len(depots) >= 2:
        ax.legend(fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gap(%) vs Depots figure from benchmark CSV.")
    parser.add_argument("--input", default="resultados_instancias.csv", help="Input CSV path")
    parser.add_argument("--output", default="figura-gap-vs-depots.pdf", help="Output image path (.pdf or .png)")
    parser.add_argument(
        "--title",
        default="Gap médio (%) em função do número de depósitos",
        help="Figure title",
    )
    parser.add_argument("--annotate", action="store_true", help="Annotate points with instance names")
    parser.add_argument(
        "--no-global-trend",
        action="store_true",
        help="Disable global linear trend line.",
    )
    parser.add_argument(
        "--no-error-bars",
        action="store_true",
        help="Disable standard deviation error bars in mean-per-depot line.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (used mainly for raster formats)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    depots, gaps, names = _read_points(input_path)
    _plot_gap_vs_depots(
        depots,
        gaps,
        names,
        output_path,
        args.title,
        args.annotate,
        show_global_trend=not args.no_global_trend,
        show_error_bars=not args.no_error_bars,
        dpi=args.dpi,
    )

    print(f"Figure saved to: {output_path}")
    print(f"Points plotted: {len(depots)}")


if __name__ == "__main__":
    main()
