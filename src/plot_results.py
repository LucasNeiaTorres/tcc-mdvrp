"""Generate academic-quality plots for the D-MDVRP dissertation.

Plots produced
--------------
1. stacked_bar_stage_utilisation.{pdf,png}
2. line_cost_degradation_vs_dod.{pdf,png}
3. boxplot_reroute_time_ms.{pdf,png}
4. violin_wasted_distance_by_dod.{pdf,png}
5. stacked_bar_feasibility_by_dod.{pdf,png}

Usage
-----
    python src/plot_results.py [--analysis-dir PATH] [--out-dir PATH]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

STAGE_COLORS = {
    "S1": "#2a9d5c",
    "S2": "#e07b27",
    "S3": "#9b2335",
    "Unresolved": "#b0b0b0",
}
EDOD_COLORS = {"0.25": "#1a6faf", "0.50": "#e07b27", "0.75": "#9b2335"}
EDOD_LABELS = {
    "0.25": "EDOD = 0.25 (precoce)",
    "0.50": "EDOD = 0.50 (uniforme)",
    "0.75": "EDOD = 0.75 (tardio)",
}

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
_SAVE_DPI = 300


def _save(fig, stem, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=_SAVE_DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — Stacked bar: stage utilisation vs DOD
# ---------------------------------------------------------------------------
def plot_stage_utilisation(df, out_dir):
    stages = ["utilisation_s1_pct", "utilisation_s2_pct", "utilisation_s3_pct"]
    labels = ["S1 — Contenção Local", "S2 — Reotimização Cluster", "S3 — Inserção geral"]
    colors = [STAGE_COLORS["S1"], STAGE_COLORS["S2"], STAGE_COLORS["S3"]]

    x = np.arange(len(df))
    bottoms = np.zeros(len(df))
    fig, ax = plt.subplots(figsize=(8, 5))

    for col, label, color in zip(stages, labels, colors):
        vals = df[col].values
        ax.bar(x, vals, 0.55, bottom=bottoms, label=label,
               color=color, edgecolor="white", linewidth=0.6)
        for xi, (b, v) in enumerate(zip(bottoms, vals)):
            if v >= 5:
                ax.text(xi, b + v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        bottoms += vals

    ax.bar(x, 100.0 - bottoms, 0.55, bottom=bottoms,
           label="Nao resolvido", color=STAGE_COLORS["Unresolved"],
           edgecolor="white", linewidth=0.6, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0%}" for d in df.index])
    ax.set_xlabel("Grau de Dinamismo (DOD)", fontsize=12)
    ax.set_ylabel("Proporcao de eventos (%)", fontsize=12)
    ax.set_title("Arquitetura Hierarquica: Utilizacao dos Estagios de Contingencia", fontsize=13)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(loc="upper left", frameon=True, fontsize=9.5, ncol=2)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "stacked_bar_stage_utilisation", out_dir)


# ---------------------------------------------------------------------------
# Plot 2 — Line plot: cost degradation vs DOD per EDOD
# ---------------------------------------------------------------------------
def plot_cost_degradation(df_multi, out_dir):
    cost_df = df_multi["cost_degradation_pct"]
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^"]

    for i, edod in enumerate(sorted(cost_df.columns, key=float)):
        color = EDOD_COLORS.get(str(edod), f"C{i}")
        label = EDOD_LABELS.get(str(edod), f"EDOD = {edod}")
        ax.plot([float(d) for d in cost_df.index], cost_df[edod].values,
                marker=markers[i % 3], color=color, label=label,
                linewidth=2, markersize=7, markeredgewidth=1.5, markeredgecolor="white")

    ax.set_xlabel("Grau de Dinamismo (DOD)", fontsize=12)
    ax.set_ylabel("Degradacao de custo (%)", fontsize=12)
    ax.set_title("Fisica do Desastre: Impacto de DOD x EDOD na Degradacao de Custo", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=1))
    ax.legend(frameon=True, fontsize=10)
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "line_cost_degradation_vs_dod", out_dir)


# ---------------------------------------------------------------------------
# Plot 3 — Boxplot: per-stage rerouting latency (ms, log scale)
# ---------------------------------------------------------------------------
def plot_reroute_time_boxplot(df, out_dir):
    time_cols = {
        "reroute_avg_s1_s": "S1 — Contenção Local",
        "reroute_avg_s2_s": "S2 — Reotimização Cluster",
        "reroute_avg_s3_s": "S3 — Inserção geral",
    }
    rows = []
    for col, label in time_cols.items():
        vals_ms = df[col].dropna() * 1000.0
        rows.append(pd.DataFrame({"Stage": label, "Latencia (ms)": vals_ms[vals_ms > 0]}))
    long_df = pd.concat(rows, ignore_index=True)

    stage_order = list(time_cols.values())
    stage_colors = [STAGE_COLORS["S1"], STAGE_COLORS["S2"], STAGE_COLORS["S3"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(
        data=long_df, x="Stage", y="Latencia (ms)",
        hue="Stage", order=stage_order, hue_order=stage_order,
        palette=dict(zip(stage_order, stage_colors)),
        width=0.5, linewidth=1.2, legend=False,
        flierprops={"marker": ".", "markersize": 4, "alpha": 0.4},
        ax=ax,
    )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:.3f}" if y < 1 else f"{y:.1f}")
    )
    ax.set_xlabel("")
    ax.set_ylabel("Tempo medio por rerouting (ms) -- escala log", fontsize=11)
    ax.set_title("Desempenho em Tempo Real: Latencia de Reroteamento por Estagio", fontsize=13)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

    for i, stage in enumerate(stage_order):
        vals = long_df.loc[long_df["Stage"] == stage, "Latencia (ms)"]
        if len(vals):
            med = vals.median()
            ax.text(
                i, med * 2.5, f"mediana\n{med:.3f} ms",
                ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#111111",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#aaaaaa", alpha=0.93, linewidth=0.9),
            )

    fig.tight_layout()
    _save(fig, "boxplot_reroute_time_ms", out_dir)


# ---------------------------------------------------------------------------
# Plot 4 — Violin: wasted distance vs DOD
# ---------------------------------------------------------------------------
def plot_wasted_distance(df, out_dir):
    df = df.dropna(subset=["dod"]).copy()
    df["DOD_label"] = df["dod"].apply(lambda v: f"{v:.0%}")
    dod_order = [f"{v:.0%}" for v in sorted(df["dod"].unique())]

    palette = {
        "5%":  "#a8d5ba",
        "10%": "#5aab7d",
        "20%": "#2a9d5c",
        "40%": "#1a6640",
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))

    sns.violinplot(
        data=df, x="DOD_label", y="wasted_distance",
        hue="DOD_label", order=dod_order, hue_order=dod_order,
        palette=palette, inner="quartile", cut=0,
        density_norm="width", linewidth=1.1, legend=False, ax=ax,
    )

    sample_frac = min(1.0, 400 / len(df))
    df_sample = df.sample(frac=sample_frac, random_state=0) if sample_frac < 1 else df
    sns.stripplot(
        data=df_sample, x="DOD_label", y="wasted_distance",
        order=dod_order, color="black", size=2.2, alpha=0.25, jitter=True, ax=ax,
    )

    for i, dod_lbl in enumerate(dod_order):
        vals = df.loc[df["DOD_label"] == dod_lbl, "wasted_distance"]
        med = vals.median()
        ax.text(
            i, med + vals.std() * 0.08, f"md={med:.1f}",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#111111",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#bbbbbb", alpha=0.9, linewidth=0.7),
        )

    ax.set_xlabel("Grau de Dinamismo (DOD)", fontsize=12)
    ax.set_ylabel("Distancia desperdicada (unidades de distancia)", fontsize=11)
    ax.set_title(
        "Custo Operacional Oculto: Dispersao do Deslocamento Inutilizado por DOD",
        fontsize=13,
    )
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "violin_wasted_distance_by_dod", out_dir)



# ---------------------------------------------------------------------------
# Plot 5 — Stacked bar: feasibility rate by DOD (+ EDOD breakdown)
# ---------------------------------------------------------------------------
def plot_feasibility_by_dod(df, out_dir):
    """Two-panel figure: feasibility by DOD (left) and by DOD x EDOD (right).

    Hard feasibility (all customers served) is always 100% -- the protocol
    never abandons a customer.  Soft violations (duration / capacity) grow
    with DOD and, crucially, also with EDOD: late disasters give the protocol
    less time to produce stable routes, raising violation rates by ~8pp at
    DOD=40%.
    """
    df = df.dropna(subset=["dod", "edod_target"]).copy()
    df["fully_feasible"] = (
        df["fully_feasible"].astype(str).str.strip().str.lower() == "true"
    )

    # ---- Panel A: aggregate by DOD ----------------------------------------
    grp = df.groupby("dod")["fully_feasible"].value_counts().unstack(fill_value=0)
    grp.columns = [bool(c) for c in grp.columns]
    grp["total"]    = grp.sum(axis=1)
    grp["pct_ok"]   = grp.get(True,  0) / grp["total"] * 100
    grp["pct_viol"] = grp.get(False, 0) / grp["total"] * 100
    dod_labels = [f"{d:.0%}" for d in grp.index]
    x = np.arange(len(grp))

    # ---- Panel B: pct_feasible by DOD x EDOD (line chart) ------------------
    piv = (
        df.groupby(["dod", "edod_target"])["fully_feasible"]
        .mean().mul(100).unstack("edod_target")
    )
    edod_vals = sorted(piv.columns)

    COLOR_OK   = "#2a9d5c"
    COLOR_VIOL = "#e07b27"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- ax1: stacked bar ---
    ax1.bar(x, grp["pct_ok"], 0.55,
            label="Totalmente viavel", color=COLOR_OK,
            edgecolor="white", linewidth=0.7)
    ax1.bar(x, grp["pct_viol"], 0.55, bottom=grp["pct_ok"],
            label="Violacoes flexíveis\n(duracao / capacidade)",
            color=COLOR_VIOL, edgecolor="white", linewidth=0.7)

    for xi, (ok, viol) in enumerate(zip(grp["pct_ok"], grp["pct_viol"])):
        ax1.text(xi, ok / 2,         f"{ok:.1f}%", ha="center", va="center",
                 fontsize=10, color="white", fontweight="bold")
        ax1.text(xi, ok + viol / 2,  f"{viol:.1f}%", ha="center", va="center",
                 fontsize=10, color="white", fontweight="bold")

    ax1.axhline(100, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.text(len(x) - 0.52, 101.5,
             "100% clientes atendidos (restricao forte)",
             ha="right", va="bottom", fontsize=7.5, color="#444444", style="italic")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dod_labels)
    ax1.set_xlabel("Grau de Dinamismo (DOD)", fontsize=11)
    ax1.set_ylabel("Proporcao de execucoes (%)", fontsize=11)
    ax1.set_title("(a) Viabilidade por DOD (agregado)", fontsize=12)
    ax1.set_ylim(0, 108)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax1.legend(loc="lower left", frameon=True, fontsize=9.5)
    ax1.grid(axis="y", alpha=0.4)
    ax1.set_axisbelow(True)

    # --- ax2: line chart ---
    markers = ["o", "s", "^"]
    edod_colors = [EDOD_COLORS.get(f"{e:.2f}", f"C{i}")
                   for i, e in enumerate(edod_vals)]
    edod_labels_map = {
        0.25: "EDOD = 0.25 (precoce)",
        0.50: "EDOD = 0.50 (uniforme)",
        0.75: "EDOD = 0.75 (tardio)",
    }
    x_dod = [float(d) for d in piv.index]

    for i, (edod, color) in enumerate(zip(edod_vals, edod_colors)):
        label = edod_labels_map.get(round(edod, 2), f"EDOD={edod:.2f}")
        ax2.plot(x_dod, piv[edod].values,
                 marker=markers[i % 3], color=color, label=label,
                 linewidth=2, markersize=7, markeredgewidth=1.5,
                 markeredgecolor="white")

    ax2.set_xlabel("Grau de Dinamismo (DOD)", fontsize=11)
    ax2.set_ylabel("Execucoes totalmente viaveis (%)", fontsize=11)
    ax2.set_title("(b) Efeito do EDOD na Viabilidade", fontsize=12)
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=1))
    ax2.set_ylim(50, 80)
    ax2.legend(frameon=True, fontsize=9.5)
    ax2.grid(alpha=0.4)
    ax2.set_axisbelow(True)

    fig.suptitle(
        "Robustez do Protocolo: Taxa de Viabilidade por DOD e EDOD",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "stacked_bar_feasibility_by_dod", out_dir)




# ---------------------------------------------------------------------------
# Plot 6 — Combined stacked bars (stage utilisation + feasibility)
# ---------------------------------------------------------------------------
def plot_combined_stacked_bars(t2_df, raw_df, out_dir):
    """Two stacked bar charts side-by-side, sharing the same DOD x-axis.

    axes[0] — Stage utilisation (S1/S2/S3 + unresolved)
    axes[1] — Feasibility rate (fully feasible vs soft violations)
    """
    # ---- Shared x setup ----
    dod_vals    = sorted(t2_df.index.tolist())
    dod_labels  = [f"{d:.0%}" for d in dod_vals]
    x           = np.arange(len(dod_vals))
    BAR_W       = 0.6

    # ---- Feasibility data ----
    raw = raw_df.dropna(subset=["dod"]).copy()
    raw["fully_feasible"] = (
        raw["fully_feasible"].astype(str).str.strip().str.lower() == "true"
    )
    grp = raw.groupby("dod")["fully_feasible"].value_counts().unstack(fill_value=0)
    grp.columns = [bool(c) for c in grp.columns]
    grp = grp.reindex(dod_vals)
    grp["total"]    = grp.sum(axis=1)
    grp["pct_ok"]   = grp.get(True,  0) / grp["total"] * 100
    grp["pct_viol"] = grp.get(False, 0) / grp["total"] * 100

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             sharey=False,
                             gridspec_kw={"wspace": 0.10})

    # ===== axes[0]: Stage utilisation =====
    ax = axes[0]
    stage_cols   = ["utilisation_s1_pct", "utilisation_s2_pct", "utilisation_s3_pct"]
    stage_labels = ["S1 — Contenção Local", "S2 — Reotimização Cluster", "S3 — Inserção geral"]
    stage_colors = [STAGE_COLORS["S1"], STAGE_COLORS["S2"], STAGE_COLORS["S3"]]

    bottoms = np.zeros(len(dod_vals))
    for col, label, color in zip(stage_cols, stage_labels, stage_colors):
        vals = t2_df.reindex(dod_vals)[col].values
        ax.bar(x, vals, BAR_W, bottom=bottoms, label=label,
               color=color, edgecolor="white", linewidth=0.6)
        for xi, (b, v) in enumerate(zip(bottoms, vals)):
            if v >= 5:
                ax.text(xi, b + v / 2, f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        bottoms += vals

    remainder = 100.0 - bottoms
    ax.bar(x, remainder, BAR_W, bottom=bottoms,
           label="Nao resolvido", color=STAGE_COLORS["Unresolved"],
           edgecolor="white", linewidth=0.6, alpha=0.7)

    ax.set_xticks(x); ax.set_xticklabels(dod_labels)
    ax.set_xlabel("Grau de Dinamismo (DOD)", fontsize=12)
    ax.set_ylabel("Proporcao de eventos (%)", fontsize=12)
    ax.set_title("(a) Utilizacao dos Estagios de Contingencia", fontsize=12, pad=10)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(loc="upper left", frameon=True, fontsize=9, ncol=1)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

    # ===== axes[1]: Feasibility =====
    ax = axes[1]
    COLOR_OK   = "#2a9d5c"
    COLOR_VIOL = "#e07b27"

    ax.bar(x, grp["pct_ok"],   BAR_W,
           label="Totalmente viavel", color=COLOR_OK,
           edgecolor="white", linewidth=0.7)
    ax.bar(x, grp["pct_viol"], BAR_W, bottom=grp["pct_ok"],
           label="Violacoes flexíveis (duracao/capacidade)",
           color=COLOR_VIOL, edgecolor="white", linewidth=0.7)

    for xi, (ok, viol) in enumerate(zip(grp["pct_ok"], grp["pct_viol"])):
        ax.text(xi, ok / 2,        f"{ok:.1f}%",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        ax.text(xi, ok + viol / 2, f"{viol:.1f}%",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")

    ax.axhline(100, color="#444", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.text(len(x) - 0.52, 100.8,
            "100% clientes atendidos (restricao forte)",
            ha="right", va="bottom", fontsize=7.5, color="#555", style="italic")

    ax.set_xticks(x); ax.set_xticklabels(dod_labels)
    ax.set_xlabel("Grau de Dinamismo (DOD)", fontsize=12)
    ax.set_ylabel("Proporcao de execucoes (%)", fontsize=12)
    ax.set_title("(b) Taxa de Viabilidade Operacional", fontsize=12, pad=10)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

    # ---- Final polish ----
    for spine in ("top", "right"):
        axes[0].spines[spine].set_visible(False)
        axes[1].spines[spine].set_visible(False)

    fig.tight_layout()
    _save(fig, "combined_stacked_bars", out_dir)


# ---------------------------------------------------------------------------
# Plot 7 — Line: feasibility survival rate by DOD x EDOD
# ---------------------------------------------------------------------------
def plot_feasibility_survival(raw_df, out_dir):
    """Line chart: % fully feasible vs DOD, one line per EDOD.

    Reads each (dod, edod_target) group, computes the fraction of runs where
    fully_feasible is True, and plots survival-style curves so the reader can
    see both the DOD and EDOD effects simultaneously.
    """
    df = raw_df.dropna(subset=["dod", "edod_target"]).copy()
    df["fully_feasible"] = (
        df["fully_feasible"].astype(str).str.strip().str.lower() == "true"
    )
    pct = (
        df.groupby(["dod", "edod_target"])["fully_feasible"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"fully_feasible": "pct_feasible", "edod_target": "EDOD"})
    )

    edod_order  = sorted(pct["EDOD"].unique())
    edod_labels = {e: EDOD_LABELS.get(f"{e:.2f}", f"EDOD = {e:.2f}") for e in edod_order}
    pct["EDOD_label"] = pct["EDOD"].map(edod_labels)
    label_order = [edod_labels[e] for e in edod_order]

    palette = {edod_labels[e]: EDOD_COLORS.get(f"{e:.2f}", f"C{i}")
               for i, e in enumerate(edod_order)}

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=pct, x="dod", y="pct_feasible",
        hue="EDOD_label", style="EDOD_label",
        hue_order=label_order, style_order=label_order,
        palette=palette,
        markers=["o", "s", "^"], dashes=False,
        linewidth=2, markersize=8, markeredgewidth=1.5,
        markeredgecolor="white",
        ax=ax,
    )

    ax.set_xlim(pct["dod"].min() - 0.02, pct["dod"].max() + 0.02)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=1))
    ax.set_xlabel("Grau de Dinamismo (DOD)", fontsize=12)
    ax.set_ylabel("Solucoes totalmente viaveis (%)", fontsize=12)
    ax.set_title(
        "Curva de Sobrevivencia: Viabilidade Total vs DOD e EDOD",
        fontsize=13,
    )
    ax.legend(title="Concentracao temporal", frameon=True, fontsize=10, title_fontsize=10)
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "line_feasibility_survival", out_dir)


# ---------------------------------------------------------------------------
# Plot 8 — Heatmap: feasibility rate matrix (DOD x EDOD)
# ---------------------------------------------------------------------------
def plot_feasibility_heatmap(raw_df, out_dir):
    """Academic heatmap: feasibility rate for each (DOD, EDOD) cell.

    The pivot matrix makes the interaction structure immediately visible --
    the reader can trace either axis to see independent effects and spot the
    worst-case corner (high DOD, high EDOD) at a glance.
    """
    df = raw_df.dropna(subset=["dod", "edod_target"]).copy()
    df["fully_feasible"] = (
        df["fully_feasible"].astype(str).str.strip().str.lower() == "true"
    )
    pct = (
        df.groupby(["dod", "edod_target"])["fully_feasible"]
        .mean()
        .mul(100)
        .unstack("edod_target")
    )
    # Rename axes for display
    pct.index   = [f"{d:.0%}" for d in pct.index]
    pct.columns = [f"{e:.2f}" for e in pct.columns]

    # Annotation strings: "65.5%"
    annot = pct.map(lambda v: f"{v:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    sns.heatmap(
        pct,
        annot=annot, fmt="",
        cmap="YlGnBu",
        vmin=50, vmax=80,
        linewidths=0.5, linecolor="#dddddd",
        cbar_kws={"label": "Solucoes totalmente viaveis (%)", "shrink": 0.85},
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax,
    )

    ax.set_xlabel("EDOD (Concentracao Temporal)", fontsize=12, labelpad=8)
    ax.set_ylabel("DOD (Extensao do Desastre)", fontsize=12, labelpad=8)
    ax.set_title(
        "Mapa de Viabilidade: DOD x EDOD",
        fontsize=13, pad=12,
    )
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11, rotation=0)

    # Highlight the worst cell with a red border
    worst_row = pct.values.argmin() // pct.shape[1]
    worst_col = pct.values.argmin() %  pct.shape[1]
    ax.add_patch(plt.Rectangle(
        (worst_col, worst_row), 1, 1,
        fill=False, edgecolor="#c0392b", linewidth=2.5, clip_on=False,
    ))

    fig.tight_layout()
    _save(fig, "heatmap_feasibility", out_dir)


# ---------------------------------------------------------------------------
# Plot 9 — Line: cost degradation vs effective edges blocked, by EDOD
# ---------------------------------------------------------------------------
def plot_cost_vs_effective_edges(df_results, out_dir):
    """Line chart (mean +/- SD) of cost degradation vs effective edges blocked.

    "Effective" edges are those that actually hit a vehicle in motion
    (n_triggered_edges), as opposed to all blocked edges (n_edges_blocked).
    This is the algorithmic view of disruption: the protocol only activates
    when a vehicle is en-route on a blocked edge.

    Key insight visible in this chart: EDOD=0.25 (early disasters) produces
    more triggered edges on average than EDOD=0.75 (late), because early
    events intercept vehicles that are still travelling.  The lines therefore
    cross, which is worth discussing in the dissertation.

    The x-axis is binned into intervals so that the lineplot with error bands
    remains readable (raw counts 0-104 would produce ~100 noisy points).
    """
    df = df_results.dropna(subset=["n_triggered_edges", "cost_degradation_pct",
                                   "edod_target"]).copy()

    # Bin n_triggered_edges into readable intervals
    bins   = [0, 5, 10, 20, 35, 55, 105]
    labels = ["1-5", "6-10", "11-20", "21-35", "36-55", "56+"]
    df["edges_bin"] = pd.cut(
        df["n_triggered_edges"],
        bins=bins, labels=labels, right=True, include_lowest=True,
    )
    # Use bin mid-point as numeric x for proper spacing
    midpoints = {lbl: mid for lbl, mid in zip(
        labels, [3, 8, 15.5, 28, 45.5, 70]
    )}
    df["edges_mid"] = df["edges_bin"].map(midpoints)

    # Remove zero-triggered rows (no vehicle was ever hit — degenerate case)
    df = df[df["n_triggered_edges"] > 0]

    edod_order  = sorted(df["edod_target"].unique())
    edod_labels = {e: EDOD_LABELS.get(f"{e:.2f}", f"EDOD = {e:.2f}") for e in edod_order}
    df["EDOD_label"] = df["edod_target"].map(edod_labels)
    label_order = [edod_labels[e] for e in edod_order]

    palette = {edod_labels[e]: EDOD_COLORS.get(f"{e:.2f}", f"C{i}")
               for i, e in enumerate(edod_order)}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # --- Background scatter (individual runs, very light) ---
    for i, edod in enumerate(edod_order):
        sub = df[df["edod_target"] == edod]
        color = EDOD_COLORS.get(f"{edod:.2f}", f"C{i}")
        ax.scatter(
            sub["edges_mid"].astype(float) + (i - 1) * 0.6,
            sub["cost_degradation_pct"],
            color=color, alpha=0.08, s=12, zorder=1,
        )

    # --- Main lineplot: mean +/- SD per bin ---
    sns.lineplot(
        data=df,
        x="edges_mid",
        y="cost_degradation_pct",
        hue="EDOD_label",
        style="EDOD_label",
        hue_order=label_order,
        style_order=label_order,
        palette=palette,
        markers=["o", "s", "^"],
        dashes=False,
        errorbar="sd",
        linewidth=2,
        markersize=8,
        markeredgewidth=1.5,
        markeredgecolor="white",
        err_kws={"alpha": 0.15},
        ax=ax,
        zorder=2,
    )

    # Custom x-ticks aligned to bin labels
    ax.set_xticks(list(midpoints.values()))
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_xlabel("Arestas Efetivamente Encontradas (bloqueios reais por execucao)",
                  fontsize=12)
    ax.set_ylabel("Degradacao de Custo Total (%)", fontsize=12)
    ax.set_title(
        "Otica Algoritmica: Degradacao de Custo vs Bloqueios Efetivos por EDOD",
        fontsize=13,
    )
    ax.legend(
        title="Grau Efetivo de Dinamismo (EDOD)",
        title_fontsize=10,
        fontsize=9.5,
        frameon=True,
        loc="upper left",
    )
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "line_cost_vs_effective_edges", out_dir)


# ---------------------------------------------------------------------------
# Plot 10 — Bar: marginal cost per blocked edge, by EDOD
# ---------------------------------------------------------------------------
def plot_marginal_cost_by_edod(df_results, out_dir):
    """Bar chart: mean cost degradation *per effective blocked edge*, by EDOD.

    Hypothesis: as the day progresses (higher EDOD = later disasters), the
    algorithm has fewer degrees of freedom -- depots are full, vehicles are
    committed -- so each additional blocked edge costs proportionally more.
    A rising marginal cost bar from EDOD=0.25 to EDOD=0.75 would confirm this.

    Processing:
        - Rows with n_triggered_edges == 0 are dropped (no vehicle was hit,
          division by zero).
        - custo_marginal = cost_degradation_pct / n_triggered_edges
    """
    df = df_results.dropna(subset=["n_triggered_edges", "cost_degradation_pct",
                                   "edod_target"]).copy()
    # Remove zero-triggered runs (degenerate: edge blocked but no vehicle hit)
    df = df[df["n_triggered_edges"] > 0].copy()

    df["custo_marginal"] = df["cost_degradation_pct"] / df["n_triggered_edges"]

    edod_order  = sorted(df["edod_target"].unique())
    edod_labels = {e: EDOD_LABELS.get(f"{e:.2f}", f"EDOD = {e:.2f}") for e in edod_order}
    df["EDOD_label"] = df["edod_target"].map(edod_labels)
    label_order = [edod_labels[e] for e in edod_order]

    palette = {edod_labels[e]: EDOD_COLORS.get(f"{e:.2f}", f"C{i}")
               for i, e in enumerate(edod_order)}

    # Print the key numbers so the author can cite them in text
    stats = (
        df.groupby("EDOD_label")["custo_marginal"]
        .agg(["mean", "std", "median"])
        .reindex(label_order)
    )
    print("  Marginal cost stats (cost_deg % per triggered edge):")
    print(stats.to_string())
    print()

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.barplot(
        data=df,
        x="EDOD_label",
        y="custo_marginal",
        order=label_order,
        hue="EDOD_label",
        hue_order=label_order,
        palette=palette,
        errorbar="sd",
        capsize=0.10,
        err_kws={"linewidth": 1.5},
        width=0.55,
        legend=False,
        ax=ax,
    )

    # Annotate mean value above each bar
    for i, lbl in enumerate(label_order):
        mean_val = df.loc[df["EDOD_label"] == lbl, "custo_marginal"].mean()
        ax.text(
            i, mean_val + 0.02,
            f"{mean_val:.3f}%",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#111111",
        )

    ax.set_xlabel("Grau Efetivo de Dinamismo (EDOD)", fontsize=12)
    ax.set_ylabel("Degradacao media por aresta bloqueada (%)", fontsize=12)
    ax.set_title("Custo Marginal de Contingencia por Nivel de EDOD", fontsize=13)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "bar_marginal_cost_by_edod", out_dir)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    default_analysis = repo_root / "data" / "processed" / "analysis"
    parser.add_argument("--analysis-dir", type=Path, default=default_analysis)
    parser.add_argument("--out-dir",      type=Path, default=None)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir
    out_dir = args.out_dir or (analysis_dir / "plots")

    print(f"Reading from : {analysis_dir}")
    print(f"Saving plots : {out_dir}\n")

    print("[1/5] Stage utilisation stacked bar")
    plot_stage_utilisation(
        pd.read_csv(analysis_dir / "t2_stage_utilisation_by_dod.csv", index_col=0),
        out_dir,
    )

    print("[2/5] Cost degradation line plot")
    plot_cost_degradation(
        pd.read_csv(analysis_dir / "t1_dod_x_edod_unserved_cost.csv",
                    header=[0, 1], index_col=0),
        out_dir,
    )

    raw = pd.read_csv(analysis_dir / "raw_results.csv")

    print("[3/5] Reroute latency boxplot")
    plot_reroute_time_boxplot(raw, out_dir)

    print("[4/5] Wasted distance violin")
    plot_wasted_distance(raw, out_dir)

    print("[5/5] Feasibility rate by DOD")
    plot_feasibility_by_dod(raw, out_dir)

    print("[6/8] Combined stacked bars")
    plot_combined_stacked_bars(
        pd.read_csv(analysis_dir / "t2_stage_utilisation_by_dod.csv", index_col=0),
        raw,
        out_dir,
    )

    print("[7/8] Feasibility survival line chart")
    plot_feasibility_survival(raw, out_dir)

    print("[8/9] Feasibility heatmap")
    plot_feasibility_heatmap(raw, out_dir)

    print("[9/10] Cost vs effective edges")
    plot_cost_vs_effective_edges(raw, out_dir)

    print("[10/10] Marginal cost by EDOD")
    plot_marginal_cost_by_edod(raw, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
