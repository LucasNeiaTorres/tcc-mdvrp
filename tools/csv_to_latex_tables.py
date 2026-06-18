#!/usr/bin/env python3
"""Convert benchmark CSV results into LaTeX table rows.

Usage:
    python3 tools/csv_to_latex_tables.py \
        --input resultado_autores.csv \
        --out-results tabela-resultados-auto.tex \
        --out-summary tabela-resumo-auto.tex
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import median

AUTHOR_ORDER = ["Christofides", "Gillett", "Chao", "Cordeau", "Unknown"]
AUTHOR_LABEL = {
    "Christofides": "Christofides et al.",
    "Gillett": "Gillett \\& Miller",
    "Chao": "Chao et al.",
    "Cordeau": "Cordeau et al.",
    "Unknown": "Unknown",
}


def _author_sort_key(author: str) -> int:
    try:
        return AUTHOR_ORDER.index(author)
    except ValueError:
        return len(AUTHOR_ORDER)


def _to_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    return float(value)


def _read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_results_rows(rows: list[dict[str, str]]) -> str:
    by_author: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        author = row.get("Author", "Unknown") or "Unknown"
        by_author[author].append(row)

    out: list[str] = []
    for author in sorted(by_author, key=_author_sort_key):
        label = AUTHOR_LABEL.get(author, author)
        out.append(rf"\multicolumn{{7}}{{|c|}}{{\textbf{{{label}}}}} \\")
        out.append(r"\hline")
        author_rows = sorted(by_author[author], key=lambda r: r["Instance"])
        for r in author_rows:
            out.append(
                f"{r['Instance']} & {r['BKS']} & {r['Best']} & {r['Mean']} & {r['Gap(%)']} & {r['Feasible(%)']} & {r['Time(s)']} \\\\"
            )
        out.append(r"\hline")

    return "\n".join(out)


def build_author_summary(rows: list[dict[str, str]]) -> str:
    by_author: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        author = row.get("Author", "Unknown") or "Unknown"
        by_author[author].append(row)

    out: list[str] = []
    for author in sorted(by_author, key=_author_sort_key):
        author_rows = by_author[author]

        gaps = [_to_float(r.get("Gap(%)", "")) for r in author_rows]
        gaps = [g for g in gaps if g is not None]
        feasible = [_to_float(r.get("Feasible(%)", "")) for r in author_rows]
        feasible = [v for v in feasible if v is not None]
        times = [_to_float(r.get("Time(s)", "")) for r in author_rows]
        times = [t for t in times if t is not None]

        gap_mean = sum(gaps) / len(gaps) if gaps else None
        gap_median = median(gaps) if gaps else None
        feasible_mean = sum(feasible) / len(feasible) if feasible else None
        time_mean = sum(times) / len(times) if times else None

        gap_mean_s = f"{gap_mean:.2f}" if gap_mean is not None else "N/A"
        gap_median_s = f"{gap_median:.2f}" if gap_median is not None else "N/A"
        feasible_s = f"{feasible_mean:.1f}" if feasible_mean is not None else "N/A"
        time_s = f"{time_mean:.2f}" if time_mean is not None else "N/A"

        label = AUTHOR_LABEL.get(author, author)
        out.append(
            f"{label} & {gap_mean_s} & {gap_median_s} & {feasible_s} & {time_s} & {len(author_rows)} \\\\"
        )

    global_gaps = [_to_float(r.get("Gap(%)", "")) for r in rows]
    global_gaps = [g for g in global_gaps if g is not None]
    global_feasible = [_to_float(r.get("Feasible(%)", "")) for r in rows]
    global_feasible = [v for v in global_feasible if v is not None]
    global_times = [_to_float(r.get("Time(s)", "")) for r in rows]
    global_times = [t for t in global_times if t is not None]

    global_gap_mean = sum(global_gaps) / len(global_gaps) if global_gaps else None
    global_gap_median = median(global_gaps) if global_gaps else None
    global_feasible_mean = sum(global_feasible) / len(global_feasible) if global_feasible else None
    global_time_mean = sum(global_times) / len(global_times) if global_times else None

    global_gap_mean_s = f"{global_gap_mean:.2f}" if global_gap_mean is not None else "N/A"
    global_gap_median_s = f"{global_gap_median:.2f}" if global_gap_median is not None else "N/A"
    global_feasible_s = f"{global_feasible_mean:.1f}" if global_feasible_mean is not None else "N/A"
    global_time_s = f"{global_time_mean:.2f}" if global_time_mean is not None else "N/A"

    out.append(r"\hline")
    out.append(
        f"\\textbf{{Global}} & \\textbf{{{global_gap_mean_s}}} & \\textbf{{{global_gap_median_s}}} & \\textbf{{{global_feasible_s}}} & \\textbf{{{global_time_s}}} & \\textbf{{{len(rows)}}} \\\\"
    )

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX table rows from benchmark CSV.")
    parser.add_argument("--input", required=True, help="CSV with per-instance results")
    parser.add_argument("--out-results", help="Output .tex for detailed results rows")
    parser.add_argument("--out-summary", help="Output .tex for author summary rows")
    args = parser.parse_args()

    rows = _read_rows(args.input)
    results_tex = build_results_rows(rows)
    summary_tex = build_author_summary(rows)

    if args.out_results:
        with open(args.out_results, "w", encoding="utf-8") as f:
            f.write(results_tex + "\n")

    if args.out_summary:
        with open(args.out_summary, "w", encoding="utf-8") as f:
            f.write(summary_tex + "\n")

    if not args.out_results and not args.out_summary:
        print("% Results table rows")
        print(results_tex)
        print("\n% Summary table rows")
        print(summary_tex)


if __name__ == "__main__":
    main()
