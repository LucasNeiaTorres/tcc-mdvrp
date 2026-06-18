#!/usr/bin/env python3
"""Generate LaTeX rows for the MDVRP instance table from data/raw/cordeau files.

Expected columns:
- Instancia
- Clientes
- Depositos
- Duracao Max
- Capacidade Max

Usage:
    python3 tools/generate_instance_table.py \
        --input-dir data/raw/cordeau \
        --out tabela-instancias-auto.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

AUTHOR_ORDER = ["Christofides", "Gillett", "Chao", "Cordeau", "Unknown"]
AUTHOR_LABEL = {
    "Christofides": "Christofides et al.",
    "Gillett": "Gillett \\& Miller",
    "Chao": "Chao et al.",
    "Cordeau": "Cordeau et al.",
    "Unknown": "Unknown",
}


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
        return AUTHOR_ORDER.index(author)
    except ValueError:
        return len(AUTHOR_ORDER)


def _instance_sort_key(name: str) -> tuple[int, int]:
    if name.startswith("p") and not name.startswith("pr"):
        return (0, int(name[1:]))
    if name.startswith("pr"):
        return (1, int(name[2:]))
    return (2, 9999)


def _fmt_num(value: float) -> str:
    # Keep integer-like values without decimal places.
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def _parse_instance_file(path: Path) -> dict[str, str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    header = lines[0].split()
    if len(header) < 4:
        raise ValueError(f"Invalid header in {path.name}: expected 4 values, got {header}")

    # Standard Cordeau format: type m n t
    # m = number of vehicles, n = number of customers, t = number of depots (MDVRP)
    n = int(float(header[2]))  # number of customers
    t = int(float(header[3]))  # number of depots

    depot_lines = lines[1 : 1 + t]
    if len(depot_lines) != t:
        raise ValueError(f"Invalid depot section in {path.name}: expected {t} lines, got {len(depot_lines)}")

    durations: list[float] = []
    capacities: list[float] = []
    for depot_ln in depot_lines:
        parts = depot_ln.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid depot line in {path.name}: '{depot_ln}'")
        durations.append(float(parts[0]))
        capacities.append(float(parts[1]))

    duration_max = max(durations)
    capacity_max = max(capacities)

    return {
        "instance": path.name,
        "author": _author_from_instance(path.name),
        "customers": str(n),
        "depots": str(t),
        "duration_max": _fmt_num(duration_max),
        "capacity_max": _fmt_num(capacity_max),
    }


def build_latex_rows(input_dir: Path) -> str:
    files = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and (p.name.startswith("p") or p.name.startswith("pr")) and p.name != "README.TXT"
    ]

    instances = [_parse_instance_file(p) for p in sorted(files, key=lambda p: _instance_sort_key(p.name))]

    by_author: dict[str, list[dict[str, str]]] = {}
    for row in instances:
        by_author.setdefault(row["author"], []).append(row)

    out: list[str] = []
    for author in sorted(by_author, key=_author_sort_key):
        label = AUTHOR_LABEL.get(author, author)
        out.append(rf"\multicolumn{{5}}{{|c|}}{{\textbf{{{label}}}}} \\")
        out.append(r"\hline")
        for r in sorted(by_author[author], key=lambda x: _instance_sort_key(x["instance"])):
            out.append(
                f"{r['instance']} & {r['customers']} & {r['depots']} & {r['duration_max']} & {r['capacity_max']} \\\\"
            )
        out.append(r"\hline")

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX rows for instance table from Cordeau files.")
    parser.add_argument("--input-dir", default="data/raw/cordeau", help="Directory with Cordeau instance files")
    parser.add_argument("--out", help="Output .tex file for table rows")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    tex = build_latex_rows(input_dir)

    if args.out:
        Path(args.out).write_text(tex + "\n", encoding="utf-8")
    else:
        print(tex)


if __name__ == "__main__":
    main()
