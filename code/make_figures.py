from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from data_loader import FEATURE_COLUMNS, TEMP_K_HI, TEMP_K_LO, TEMP_LEVELS_K


FIG_DIR = Path(__file__).resolve().parent / "figures"

# Heater-tube temperature in true Kelvin (converted from GM 4L23 deg F levels).
TEMP_UNIT = "K"
TEMP_SPAN = TEMP_K_HI - TEMP_K_LO

PLASMA_CMAP = "plasma"
PLASMA = plt.get_cmap(PLASMA_CMAP)
PLASMA_COLORS = {
    "purple": PLASMA(0.08),
    "violet": PLASMA(0.22),
    "magenta": PLASMA(0.42),
    "orange": PLASMA(0.68),
    "gold": PLASMA(0.88),
    "yellow": PLASMA(0.97),
}


def setup_ieee_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for text in fig.findobj(match=matplotlib.text.Text):
        text.set_fontfamily("Times New Roman")
        if text.get_fontsize() < 10:
            text.set_fontsize(10)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _white_label() -> dict:
    return dict(fc="white", ec="0.65", lw=0.45, alpha=0.96, boxstyle="round,pad=0.22")


def _framework_box(axes: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        0.095,
        0.075,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        fc=color,
        ec="0.25",
        lw=0.85,
    )
    axes.add_patch(box)
    axes.text(x + 0.0475, y + 0.0375, text, ha="center", va="center", fontsize=7)


def _framework_arrow(axes: plt.Axes, start_box: tuple, end_box: tuple) -> None:
    sx, sy = start_box[0] + 0.095, start_box[1] + 0.0375
    ex, ey = end_box[0], end_box[1] + 0.0375
    if end_box[0] < start_box[0]:
        sx, sy = start_box[0], start_box[1] + 0.0375
        ex, ey = end_box[0] + 0.095, end_box[1] + 0.0375
    axes.annotate(
        "",
        xy=(ex, ey),
        xytext=(sx, sy),
        arrowprops=dict(arrowstyle="->", lw=0.75, color="0.22", shrinkA=3, shrinkB=3),
    )


def plot_framework() -> None:
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def section_header(y: float, title: str) -> None:
        ax.text(0.070, y + 0.167, title, ha="left", va="center", fontsize=10.5, weight="bold")

    def process_box(x: float, y: float, w: float, h: float, text: str, color: str, icon: str = "") -> tuple:
        ax.add_patch(Rectangle((x, y), w, h, fc=color, ec="white", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, weight="bold")
        return (x, y, w, h)

    def light_plasma(value: float, white_mix: float = 0.74) -> tuple[float, float, float, float]:
        r, g, b, a = PLASMA(value)
        return (
            r * (1.0 - white_mix) + white_mix,
            g * (1.0 - white_mix) + white_mix,
            b * (1.0 - white_mix) + white_mix,
            a,
        )

    def arrow_between(left: tuple, right: tuple) -> None:
        ax.annotate(
            "",
            xy=(right[0] - 0.010, right[1] + right[3] / 2),
            xytext=(left[0] + left[2] + 0.010, left[1] + left[3] / 2),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
        )

    ax.text(
        0.5,
        0.965,
        "Multi-agent physics-guided active modelling and dispatch for WHR microgrids",
        ha="center",
        va="center",
        weight="bold",
        fontsize=12,
    )

    y1 = 0.735
    section_header(y1, "(1) Operating-evidence knowledge base")
    b11 = process_box(0.075, y1, 0.17, 0.105, "WHR operating\nmeasurements", light_plasma(0.18))
    b12 = process_box(0.285, y1, 0.17, 0.105, "Data cleaning &\nunit harmonisation", light_plasma(0.28))
    b13 = process_box(0.495, y1, 0.17, 0.105, "Device bounds &\nphysical priors", light_plasma(0.38))
    b14 = process_box(0.735, y1, 0.17, 0.105, "Operating-space\nmemory", light_plasma(0.48))
    for left, right in [(b11, b12), (b12, b13), (b13, b14)]:
        arrow_between(left, right)

    y2 = 0.545
    section_header(y2, "(2) LLM/RAG-guided active case generation")
    b21 = process_box(0.075, y2, 0.17, 0.105, "Constraint-aware\nprompt schema", PLASMA(0.38))
    b22 = process_box(0.285, y2, 0.17, 0.105, "Sparse-region\ncase proposal", PLASMA(0.45))
    b23 = process_box(0.495, y2, 0.17, 0.105, "Acquisition score\nsparsity + uncertainty\nquantification", PLASMA(0.52))
    b24 = process_box(0.735, y2, 0.17, 0.105, "Machine-readable\ncandidate set", PLASMA(0.58))
    for left, right in [(b21, b22), (b22, b23), (b23, b24)]:
        arrow_between(left, right)

    y3 = 0.335
    section_header(y3, "(3) Physics review, labelling, and surrogate learning")
    b31 = process_box(0.075, y3, 0.20, 0.115, "Feasibility filter\nbounds + monotonicity", PLASMA(0.66))
    b32 = process_box(0.315, y3, 0.20, 0.115, "Solver labelling\nfor accepted cases", PLASMA(0.72))
    b33 = process_box(0.555, y3, 0.20, 0.115, "Multi-output\nsurrogate ensemble", PLASMA(0.78))
    b34 = process_box(0.795, y3, 0.11, 0.115, "Uncertainty\nquantification\nmap", PLASMA(0.83))
    for left, right in [(b31, b32), (b32, b33), (b33, b34)]:
        arrow_between(left, right)

    y4 = 0.135
    section_header(y4, "(4) Confidence-aware WHR-PV-storage dispatch")
    b41 = process_box(0.075, y4, 0.20, 0.105, "PV-load-waste\nprofile encoder", PLASMA(0.86))
    b42 = process_box(0.315, y4, 0.20, 0.105, "Risk-adjusted\nWHR action value", PLASMA(0.90))
    b43 = process_box(0.555, y4, 0.20, 0.105, "Storage SOC\ntransition model", PLASMA(0.94))
    b44 = process_box(0.795, y4, 0.11, 0.105, "Dispatch\npolicy", PLASMA(0.98))
    for left, right in [(b41, b42), (b42, b43), (b43, b44)]:
        arrow_between(left, right)

    for x in [0.82]:
        ax.annotate("", xy=(x, y2 + 0.112), xytext=(x, y1 - 0.005), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.25"))
        ax.annotate("", xy=(x, y3 + 0.123), xytext=(x, y2 - 0.005), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.25"))
        ax.annotate("", xy=(x, y4 + 0.112), xytext=(x, y3 - 0.005), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.25"))

    ax.text(
        0.5,
        0.040,
        "Safeguard: LLM/RAG proposes candidate cases only; physics filters, solver labels, and uncertainty-aware optimisation determine numerical decisions.",
        ha="center",
        color="0.18",
        fontsize=8.4,
    )
    save_figure(fig, "fig01_multi_agent_framework")


def plot_agent_protocol() -> None:
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def mix(value: float, white: float = 0.36) -> tuple[float, float, float, float]:
        r, g, b, a = PLASMA(value)
        return (r * (1 - white) + white, g * (1 - white) + white, b * (1 - white) + white, a)

    lanes = [
        ("Operating\nmemory", 0.84, mix(0.16, 0.62)),
        ("LLM/RAG\nproposer agent", 0.68, mix(0.28, 0.38)),
        ("Critic + acquisition\nagent", 0.52, mix(0.46, 0.34)),
        ("Physics validator\n+ solver", 0.36, mix(0.68, 0.30)),
        ("Surrogate learner\n+ audit logger", 0.20, mix(0.88, 0.25)),
    ]

    for title, y, color in lanes:
        ax.add_patch(Rectangle((0.035, y - 0.055), 0.93, 0.105, fc=color, ec="white", lw=1.2))
        ax.text(0.052, y, title, ha="left", va="center", fontsize=7.7, weight="bold", color="black")

    def box(x: float, y: float, w: float, h: float, text: str, fc: tuple | str = "white") -> tuple[float, float, float, float]:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            fc=fc,
            ec="0.18",
            lw=0.8,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, color="black")
        return (x, y, w, h)

    def center_right(b: tuple[float, float, float, float]) -> tuple[float, float]:
        return (b[0] + b[2], b[1] + b[3] / 2)

    def center_left(b: tuple[float, float, float, float]) -> tuple[float, float]:
        return (b[0], b[1] + b[3] / 2)

    def center_bottom(b: tuple[float, float, float, float]) -> tuple[float, float]:
        return (b[0] + b[2] / 2, b[1])

    def center_top(b: tuple[float, float, float, float]) -> tuple[float, float]:
        return (b[0] + b[2] / 2, b[1] + b[3])

    def arrow(p1: tuple[float, float], p2: tuple[float, float], style: str = "->", color: str = "0.15", lw: float = 0.9) -> None:
        ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle=style, lw=lw, color=color, shrinkA=2, shrinkB=2))

    def elbow_arrow(points: list[tuple[float, float]], color: str = "0.15", lw: float = 0.9) -> None:
        """Draw an orthogonal arrow through whitespace corridors between boxes."""
        if len(points) < 2:
            return
        for start, end in zip(points[:-2], points[1:-1]):
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw, solid_capstyle="butt", clip_on=False)
        ax.annotate(
            "",
            xy=points[-1],
            xytext=points[-2],
            arrowprops=dict(arrowstyle="->", lw=lw, color=color, shrinkA=0, shrinkB=2),
        )

    # Operating memory lane
    m1 = box(0.235, 0.807, 0.105, 0.066, "Measured\nWHR cases")
    m2 = box(0.370, 0.807, 0.105, 0.066, "Device\nbounds")
    m3 = box(0.505, 0.807, 0.105, 0.066, "Physical\npriors")
    m4 = box(0.640, 0.807, 0.125, 0.066, "Surrogate\nuncertainty")
    mem = box(0.800, 0.807, 0.115, 0.066, "Retrieved\nevidence")

    # Proposer lane
    p1 = box(0.220, 0.647, 0.120, 0.066, "Retrieve sparse\noperating context")
    p2 = box(0.370, 0.647, 0.135, 0.066, "Generate candidate\nregion descriptions")
    p3 = box(0.545, 0.647, 0.125, 0.066, "Emit JSON\ncandidate pool")
    p4 = box(0.800, 0.647, 0.115, 0.066, "No power / efficiency\nlabels allowed", fc=mix(0.04, 0.76))

    # Critic lane
    c1 = box(0.220, 0.487, 0.120, 0.066, "Compute\nsparsity score")
    c2 = box(0.370, 0.487, 0.135, 0.066, "Compute ensemble\nuncertainty")
    c3 = box(0.545, 0.487, 0.125, 0.066, "Acquisition score\n0.45s + 0.55u")
    c4 = box(0.800, 0.487, 0.115, 0.066, "Diversity filter\n+ top-K cases")

    # Physics lane
    v1 = box(0.220, 0.327, 0.120, 0.066, "Bounds\ncheck")
    v2 = box(0.370, 0.327, 0.135, 0.066, "Monotonicity +\nfeasibility review")
    v3 = box(0.545, 0.327, 0.125, 0.066, "Solver / oracle\nlabelling")
    v4 = box(0.800, 0.327, 0.115, 0.066, "Rejected-case\nreasons")

    # Learner lane
    l1 = box(0.220, 0.167, 0.120, 0.066, "Augmented\ntraining set")
    l2 = box(0.370, 0.167, 0.135, 0.066, "Retrain multi-output\nsurrogate ensemble")
    l3 = box(0.545, 0.167, 0.125, 0.066, "Update uncertainty\nquantification map")
    l4 = box(0.800, 0.167, 0.115, 0.066, "Audit trace:\nprompts, scores,\nlabels, rejects")

    # Horizontal protocol arrows
    for left, right in [(m1, m2), (m2, m3), (m3, m4), (m4, mem), (p1, p2), (p2, p3), (p3, p4), (c1, c2), (c2, c3), (c3, c4), (v1, v2), (v2, v3), (v3, v4), (l1, l2), (l2, l3), (l3, l4)]:
        arrow(center_right(left), center_left(right))

    # Cross-lane information flow, routed as right-angle arrows through empty corridors.
    elbow_arrow([center_bottom(mem), (center_bottom(mem)[0], 0.760), (center_top(p2)[0], 0.760), center_top(p2)], color=PLASMA(0.10), lw=1.0)
    elbow_arrow([center_bottom(p3), (center_bottom(p3)[0], 0.600), (center_top(c1)[0], 0.600), center_top(c1)], color=PLASMA(0.28), lw=1.0)
    elbow_arrow([center_bottom(c4), (center_bottom(c4)[0], 0.440), (center_top(v1)[0], 0.440), center_top(v1)], color=PLASMA(0.46), lw=1.0)
    elbow_arrow([center_bottom(v3), (center_bottom(v3)[0], 0.280), (center_top(l1)[0], 0.280), center_top(l1)], color=PLASMA(0.70), lw=1.0)
    elbow_arrow([center_bottom(v4), (center_bottom(v4)[0], 0.280), (center_top(l4)[0], 0.280), center_top(l4)], color=PLASMA(0.70), lw=0.9)
    elbow_arrow([center_top(l3), (center_top(l3)[0], 0.280), (0.952, 0.280), (0.952, 0.780), (center_bottom(m4)[0], 0.780), center_bottom(m4)], color=PLASMA(0.90), lw=1.1)

    ax.text(0.50, 0.955, "Auditable multi-agent protocol for physics-guided operating-case augmentation", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(
        0.50,
        0.055,
        "Key safeguard: language agents propose and structure candidate cases only; numerical labels are produced by physics review and solver/oracle evaluation.",
        ha="center",
        va="center",
        fontsize=8.2,
        bbox=_white_label(),
    )
    save_figure(fig, "fig08_agent_protocol")


def plot_agent_decision_audit(selected: pd.DataFrame, metrics: dict, train: pd.DataFrame | None = None, name: str = "fig08_agent_protocol") -> None:
    setup_ieee_style()
    if train is None:
        train_path = FIG_DIR.parent / "results" / "train_cases.csv"
        train = pd.read_csv(train_path)

    panel_title = {"fontsize": 9, "fontname": "Times New Roman"}
    axis_label = {"fontsize": 10}
    tick_label_size = 9
    legend_size = 9
    annotation_size = 9

    fig, (ax_cov, ax_trace, ax_err) = plt.subplots(3, 1, figsize=(6.9, 10.8))

    feature_cols = ["source_temperature", "mean_pressure", "engine_speed"]
    baseline_x = train[feature_cols].to_numpy(float)
    augmented_x = pd.concat([train[feature_cols], selected[feature_cols]], ignore_index=True).to_numpy(float)

    temps = np.linspace(min(baseline_x[:, 0].min(), augmented_x[:, 0].min()), max(baseline_x[:, 0].max(), augmented_x[:, 0].max()), 17)
    pressures = np.linspace(min(baseline_x[:, 1].min(), augmented_x[:, 1].min()), max(baseline_x[:, 1].max(), augmented_x[:, 1].max()), 45)
    speeds = np.linspace(min(baseline_x[:, 2].min(), augmented_x[:, 2].min()), max(baseline_x[:, 2].max(), augmented_x[:, 2].max()), 45)
    tt, pp, ss = np.meshgrid(temps, pressures, speeds, indexing="ij")
    grid_x = np.column_stack([tt.ravel(), pp.ravel(), ss.ravel()])
    scale = np.array([TEMP_SPAN, 2000.0, 2800.0])

    def nearest_distance(reference: np.ndarray) -> np.ndarray:
        dist = np.sqrt((((grid_x[:, None, :] - reference[None, :, :]) / scale) ** 2).sum(axis=2))
        return dist.min(axis=1)

    def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    baseline_dist = nearest_distance(baseline_x)
    augmented_dist = nearest_distance(augmented_x)
    xb, yb = ecdf(baseline_dist)
    xa, ya = ecdf(augmented_dist)
    ax_cov.plot(xb, yb, color=PLASMA(0.80), lw=2.0, label=f"Sparse baseline ({len(train)} cases)")
    ax_cov.plot(xa, ya, color=PLASMA(0.12), lw=2.0, label=f"Agent augmented (+{len(selected)} cases)")
    med_b = float(np.median(baseline_dist))
    med_a = float(np.median(augmented_dist))
    ax_cov.axvline(med_b, color=PLASMA(0.80), lw=1.0, ls="--")
    ax_cov.axvline(med_a, color=PLASMA(0.12), lw=1.0, ls="--")
    ax_cov.fill_betweenx([0, 1], med_a, med_b, color=PLASMA(0.35), alpha=0.12, lw=0)
    ax_cov.set_title("(a) Operating-space coverage audit", pad=7, **panel_title)
    ax_cov.set_xlabel("Normalised distance to nearest labelled case", **axis_label)
    ax_cov.set_ylabel("Fraction of dense operating grid", **axis_label)
    ax_cov.set_xlim(0, np.percentile(baseline_dist, 99.5) * 1.03)
    ax_cov.set_ylim(0, 1.01)
    ax_cov.tick_params(axis="both", labelsize=tick_label_size)
    ax_cov.legend(loc="lower right", fontsize=legend_size, frameon=True, facecolor="white", edgecolor="0.65")
    ax_cov.grid(False)

    score = selected["agent_score"].to_numpy(float)
    sizes = 52 + 165 * (score - score.min()) / (score.max() - score.min() + 1e-12)
    sparse_med = float(selected["sparsity_score"].median())
    uncert_med = float(selected["uncertainty_score"].median())
    sparse_line = "#2f6db3"
    uncert_line = "#e3a51a"
    ax_trace.axvspan(sparse_med, 1.0, color=sparse_line, alpha=0.07, lw=0)
    ax_trace.axhspan(uncert_med, 1.0, color=uncert_line, alpha=0.08, lw=0)
    ax_trace.axvline(sparse_med, color=sparse_line, lw=1.05, ls="--")
    ax_trace.axhline(uncert_med, color=uncert_line, lw=1.05, ls="--")
    sc_trade = ax_trace.scatter(
        selected["sparsity_score"],
        selected["uncertainty_score"],
        c=selected["source_temperature"],
        s=sizes,
        cmap=PLASMA_CMAP,
        edgecolors="black",
        linewidths=0.55,
        alpha=0.92,
        zorder=3,
    )
    top = selected.nlargest(5, "agent_score")
    ax_trace.scatter(
        top["sparsity_score"],
        top["uncertainty_score"],
        s=145,
        facecolors="none",
        edgecolors=PLASMA(0.02),
        linewidths=1.15,
        zorder=4,
    )
    ax_trace.set_title("(b) Sparsity-uncertainty selection map", pad=7, **panel_title)
    ax_trace.set_xlabel("Sparse-region score", **axis_label)
    ax_trace.set_ylabel("Ensemble uncertainty score", **axis_label)
    ax_trace.set_xlim(-0.03, 1.03)
    ax_trace.set_ylim(-0.03, 1.03)
    ax_trace.tick_params(axis="both", labelsize=tick_label_size)
    ax_trace.legend(
        handles=[
            Line2D([0], [0], color=sparse_line, lw=1.05, ls="--", label="Median sparse-region score"),
            Line2D([0], [0], color=uncert_line, lw=1.05, ls="--", label="Median uncertainty score"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=PLASMA(0.02), markeredgewidth=1.0, markersize=6.0, label="Top-score cases"),
        ],
        loc="lower left",
        fontsize=legend_size,
        frameon=True,
        facecolor="white",
        edgecolor="0.65",
    )
    cbar = fig.colorbar(sc_trade, ax=ax_trace, pad=0.02)
    cbar.set_label(f"Source temperature / {TEMP_UNIT}", fontsize=10)
    cbar.ax.tick_params(labelsize=tick_label_size)
    ax_trace.grid(False)

    metric_pairs = [
        ("Power\nMAE", "power_mae", "W"),
        ("Power\nRMSE", "power_rmse", "W"),
        ("Eff.\nMAE", "efficiency_mae_pct_point", "%"),
        ("Eff.\nRMSE", "efficiency_rmse_pct_point", "%"),
    ]
    labels = [m[0] for m in metric_pairs]
    baseline_vals = np.array([float(metrics["baseline"][key]) for _, key, _ in metric_pairs])
    augmented_vals = np.array([float(metrics["agent_augmented"][key]) for _, key, _ in metric_pairs])
    relative_aug = 100 * augmented_vals / baseline_vals
    x = np.arange(len(metric_pairs))
    w = 0.34
    ax_err.bar(
        x - w / 2,
        np.full_like(relative_aug, 100.0),
        width=w,
        color=PLASMA(0.82),
        edgecolor="black",
        linewidth=0.45,
        label="Sparse baseline",
        zorder=2,
    )
    agent_bars = ax_err.bar(
        x + w / 2,
        relative_aug,
        width=w,
        color=PLASMA(0.16),
        edgecolor="black",
        linewidth=0.45,
        label="Agent-augmented",
        zorder=2,
    )
    for bar, rel in zip(agent_bars, relative_aug):
        cx = bar.get_x() + bar.get_width() / 2
        ax_err.text(cx, rel + 2.5, f"{rel:.1f}", ha="center", va="bottom", fontsize=annotation_size)
    ax_err.set_title("(c) Holdout error after augmentation", pad=7, **panel_title)
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(labels)
    ax_err.set_ylabel("Holdout error relative to sparse baseline / %", **axis_label)
    ax_err.set_ylim(0, 120)
    ax_err.set_yticks(np.arange(0, 101, 20))
    ax_err.tick_params(axis="both", labelsize=tick_label_size)
    ax_err.legend(
        loc="upper right",
        ncol=2,
        fontsize=legend_size,
        frameon=True,
        facecolor="white",
        edgecolor="0.65",
        borderaxespad=0.35,
        columnspacing=1.2,
        handletextpad=0.45,
    )
    ax_err.grid(False)

    fig.suptitle("Agent augmentation audit: coverage gain, selection behaviour, and holdout generalisation", y=0.995, weight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965], h_pad=1.8)
    save_figure(fig, name)


def plot_agent_ranktrace_audit(selected: pd.DataFrame, metrics: dict, name: str = "fig08_agent_decision_audit_ranktrace") -> None:
    setup_ieee_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2))
    ax_rank, ax_trade, ax_label, ax_impact = axes.ravel()

    ranked = selected.sort_values("agent_score", ascending=False).reset_index(drop=True)
    rank = np.arange(1, len(ranked) + 1)
    sparsity_part = 0.45 * ranked["sparsity_score"].to_numpy(float)
    uncertainty_part = 0.55 * ranked["uncertainty_score"].to_numpy(float)
    score = ranked["agent_score"].to_numpy(float)

    ax_rank.bar(rank, sparsity_part, width=0.72, color=PLASMA(0.32), edgecolor="black", linewidth=0.35, label="Sparse-region part")
    ax_rank.bar(
        rank,
        uncertainty_part,
        width=0.72,
        bottom=sparsity_part,
        color=PLASMA(0.78),
        edgecolor="black",
        linewidth=0.35,
        label="Uncertainty part",
    )
    ax_rank.plot(rank, score, color=PLASMA(0.05), marker="o", ms=3.0, lw=1.0, label="Agent score")
    ax_rank.set_title("(a) Ranked acquisition trace", pad=7)
    ax_rank.set_xlabel("Selected query rank")
    ax_rank.set_ylabel("Normalised score")
    ax_rank.set_xlim(0.25, len(ranked) + 0.75)
    ax_rank.set_ylim(0, max(0.82, float(score.max()) + 0.08))
    ax_rank.set_xticks(np.arange(1, len(ranked) + 1, 4))
    ax_rank.legend(loc="upper right", fontsize=7, frameon=True, facecolor="white", edgecolor="0.65")
    ax_rank.grid(False)

    sizes = 95 + 175 * (score - score.min()) / (score.max() - score.min() + 1e-12)
    sc = ax_trade.scatter(
        ranked["sparsity_score"],
        ranked["uncertainty_score"],
        c=ranked["source_temperature"],
        s=sizes,
        cmap=PLASMA_CMAP,
        edgecolors="black",
        linewidths=0.55,
        alpha=0.92,
    )
    ax_trade.set_title("(b) Sparsity-uncertainty trade-off", pad=7)
    ax_trade.set_xlabel("Sparse-region score")
    ax_trade.set_ylabel("Ensemble uncertainty score")
    ax_trade.set_xlim(-0.03, 1.03)
    ax_trade.set_ylim(-0.03, 1.03)
    cbar = fig.colorbar(sc, ax=ax_trade, pad=0.02)
    cbar.set_label(f"Source temperature / {TEMP_UNIT}")
    cbar.ax.tick_params(labelsize=7)
    ax_trade.grid(False)

    speed = ranked["engine_speed"].to_numpy(float)
    speed_size = 42 + 125 * (speed - speed.min()) / (speed.max() - speed.min() + 1e-12)
    sc2 = ax_label.scatter(
        ranked["power"],
        100 * ranked["efficiency"],
        c=ranked["source_temperature"],
        s=speed_size,
        cmap=PLASMA_CMAP,
        edgecolors="black",
        linewidths=0.55,
        alpha=0.92,
    )
    ax_label.set_title("(c) Solver-labelled selected cases", pad=7)
    ax_label.set_xlabel("Labelled power / W")
    ax_label.set_ylabel("Labelled efficiency / %")
    source_counts = ranked["label_source"].value_counts().to_dict() if "label_source" in ranked else {}
    label_note = ", ".join([f"{k}: {v}" for k, v in source_counts.items()]) or "labels: solver/oracle"
    ax_label.text(
        0.03,
        0.95,
        f"{len(ranked)} queried cases\n{label_note}\n0 LLM-generated labels",
        transform=ax_label.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox=_white_label(),
    )
    cbar2 = fig.colorbar(sc2, ax=ax_label, pad=0.02)
    cbar2.set_label(f"Source temperature / {TEMP_UNIT}")
    cbar2.ax.tick_params(labelsize=7)
    ax_label.grid(False)

    base_power = float(metrics["baseline"]["power_mae"])
    aug_power = float(metrics["agent_augmented"]["power_mae"])
    base_eff = float(metrics["baseline"]["efficiency_mae_pct_point"])
    aug_eff = float(metrics["agent_augmented"]["efficiency_mae_pct_point"])
    rel_aug = np.array([100 * aug_power / base_power, 100 * aug_eff / base_eff])
    x = np.arange(2)
    w = 0.34
    ax_impact.bar(x - w / 2, [100, 100], width=w, color=PLASMA(0.82), edgecolor="black", linewidth=0.45, label="Sparse baseline")
    ax_impact.bar(x + w / 2, rel_aug, width=w, color=PLASMA(0.18), edgecolor="black", linewidth=0.45, label="Agent augmented")
    ax_impact.set_xticks(x)
    ax_impact.set_xticklabels(["Power MAE", "Efficiency MAE"])
    ax_impact.set_ylabel("Error relative to sparse baseline / %")
    ax_impact.set_title("(d) Downstream validation impact", pad=7)
    ax_impact.set_ylim(0, 118)
    reductions = [100 - rel_aug[0], 100 - rel_aug[1]]
    actual_labels = [f"{base_power:.2f} -> {aug_power:.2f} W", f"{base_eff:.2f} -> {aug_eff:.2f} %-pt"]
    for i, (pct, reduction, label) in enumerate(zip(rel_aug, reductions, actual_labels)):
        ax_impact.text(i + w / 2, pct + 4, f"-{reduction:.1f}%\n{label}", ha="center", va="bottom", fontsize=7.0, bbox=_white_label())
    ax_impact.legend(loc="upper right", fontsize=7, frameon=True, facecolor="white", edgecolor="0.65")
    ax_impact.grid(False)

    fig.suptitle("Agent operating-case augmentation audit: selection scores, labels, and validation impact", y=0.992, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.958])
    save_figure(fig, name)


def plot_data_coverage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    panel_a_xlim: tuple[float, float] | None = None,
    panel_a_xticks: list[float] | None = None,
    panel_a_ylim: tuple[float, float] | None = None,
    panel_a_yticks: list[float] | None = None,
    panel_a_zlim: tuple[float, float] | None = None,
    panel_a_zticks: list[float] | None = None,
) -> None:
    setup_ieee_style()
    fig = plt.figure(figsize=(10.2, 5.05))
    ax3d = fig.add_axes([0.015, 0.300, 0.395, 0.60], projection="3d")
    axheat = fig.add_axes([0.575, 0.335, 0.305, 0.525])

    sc = ax3d.scatter(
        train["engine_speed"],
        train["mean_pressure"],
        train["source_temperature"],
        c=train["efficiency"] * 100,
        cmap=PLASMA_CMAP,
        s=13,
        alpha=0.7,
        label="Measured training cases",
    )
    ax3d.scatter(
        selected["engine_speed"],
        selected["mean_pressure"],
        selected["source_temperature"],
        c=[PLASMA(0.05)],
        marker="^",
        s=72,
        edgecolors="black",
        linewidths=0.7,
        label="Agent-selected cases",
    )
    ax3d.scatter(
        test["engine_speed"],
        test["mean_pressure"],
        test["source_temperature"],
        c="black",
        marker="x",
        s=30,
        label="Holdout cases",
    )
    ax3d.set_xlabel("Speed / rpm", labelpad=-4)
    ax3d.set_ylabel("Pressure / kPa", labelpad=-4)
    ax3d.set_zlabel(f"Temperature / {TEMP_UNIT}", labelpad=1)
    if panel_a_zlim is not None:
        ax3d.set_zlim(*panel_a_zlim)
    else:
        ax3d.set_zlim(TEMP_K_LO - 15, TEMP_K_HI + 15)
    if panel_a_zticks is not None:
        ax3d.set_zticks(panel_a_zticks)
    else:
        ax3d.set_zticks([round(t) for t in TEMP_LEVELS_K])
    if panel_a_xlim is not None:
        ax3d.set_xlim(*panel_a_xlim)
    else:
        ax3d.set_xlim(0, 3200)
    if panel_a_xticks is not None:
        ax3d.set_xticks(panel_a_xticks)
    else:
        ax3d.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
    if panel_a_ylim is not None:
        ax3d.set_ylim(*panel_a_ylim)
    if panel_a_yticks is not None:
        ax3d.set_yticks(panel_a_yticks)
    ax3d.set_box_aspect((1.0, 1.0, 1.0))
    ax3d.view_init(elev=22, azim=-52)
    ax3d.tick_params(axis="x", pad=-2, labelsize=10)
    ax3d.tick_params(axis="y", pad=-2, labelsize=10)
    ax3d.tick_params(axis="z", pad=2, labelsize=10)
    for lbl in ax3d.get_xticklabels():
        lbl.set_rotation(40)
        lbl.set_rotation_mode("anchor")
    ax3d.set_title("(a) Selected cases in operating space", pad=4, fontname="Times New Roman", fontsize=10)
    cb_ax_eff = fig.add_axes([0.435, 0.365, 0.018, 0.455])
    cb = fig.colorbar(sc, cax=cb_ax_eff, orientation="vertical")
    cb.set_label("Measured efficiency / %", fontsize=10, labelpad=7)
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_linewidth(0.8)

    all_cases = pd.concat([train, test, selected], ignore_index=True)
    spd = all_cases["engine_speed"].to_numpy(float)
    prs = all_cases["mean_pressure"].to_numpy(float)
    speed_lo = max(0.0, float(spd.min()) - 100.0)
    speed_hi = float(spd.max()) + 100.0
    pressure_lo = float(prs.min()) - 100.0
    pressure_hi = float(prs.max()) + 100.0
    speed_grid = np.linspace(speed_lo, speed_hi, 130)
    pressure_grid = np.linspace(pressure_lo, pressure_hi, 105)
    sg, pg = np.meshgrid(speed_grid, pressure_grid)
    train_sp = train[["engine_speed", "mean_pressure"]].to_numpy(float)
    scale = np.array([max(speed_hi - speed_lo, 1.0), max(pressure_hi - pressure_lo, 1.0)])
    grid_points = np.column_stack([sg.ravel(), pg.ravel()])
    distances = np.sqrt((((grid_points[:, None, :] - train_sp[None, :, :]) / scale) ** 2).sum(axis=2))
    sparse_field = distances.min(axis=1).reshape(pg.shape)
    sparse_field = (sparse_field - sparse_field.min()) / (sparse_field.max() - sparse_field.min() + 1e-12)
    im = axheat.contourf(
        sg,
        pg,
        sparse_field,
        levels=np.linspace(0, 1, 16),
        cmap=PLASMA_CMAP,
    )
    axheat.scatter(
        train["engine_speed"],
        train["mean_pressure"],
        s=9,
        marker="o",
        c="black",
        alpha=0.42,
        linewidths=0,
        clip_on=True,
    )
    axheat.scatter(
        selected["engine_speed"],
        selected["mean_pressure"],
        s=34,
        marker="^",
        c=[PLASMA(0.90)],
        edgecolors="black",
        linewidths=0.35,
        clip_on=True,
    )
    pad_x = max(40.0, (speed_hi - speed_lo) * 0.04)
    pad_y = max(40.0, (pressure_hi - pressure_lo) * 0.04)
    axheat.set_xlim(speed_lo - pad_x, speed_hi + pad_x)
    axheat.set_ylim(pressure_lo - pad_y, pressure_hi + pad_y)
    axheat.set_xlabel("Speed / rpm")
    axheat.set_ylabel("Pressure / kPa")
    axheat.set_title("(b) Sparse regions targeted by agents", fontname="Times New Roman", fontsize=10)
    cb_ax_bins = fig.add_axes([0.905, 0.365, 0.018, 0.455])
    cb2 = fig.colorbar(im, cax=cb_ax_bins, orientation="vertical")
    cb2.set_label("Sparse-region index", fontsize=10)
    cb2.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
    cb2.ax.tick_params(labelsize=10)
    cb2.outline.set_linewidth(0.8)
    axheat.tick_params(axis="both", labelsize=10)
    axheat.grid(True, ls=":", lw=0.4, color="0.75")

    handles3d = [
        Line2D([], [], marker="o", color="none", markerfacecolor="black", markeredgecolor="black", markersize=5.5, label="Measured training cases"),
        Line2D([], [], marker="^", color="none", markerfacecolor="black", markeredgecolor="black", markersize=9, label="Agent-selected cases"),
        Line2D([], [], marker="x", color="none", markeredgecolor="black", markersize=7, markeredgewidth=1.4, label="Holdout cases"),
    ]
    labels3d = ["Measured training cases", "Agent-selected cases", "Holdout cases"]
    fig.legend(
        handles3d,
        labels3d,
        title="(a)",
        loc="lower center",
        bbox_to_anchor=(0.275, -0.045),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="0.65",
        fontsize=10,
    )
    fig.text(0.5, 0.965, "Operating-case augmentation targets sparse regions of the measured space", ha="center", weight="bold", fontsize=11)
    save_figure(fig, "fig02_agent_selected_cases")


def plot_parity(
    y_true: np.ndarray,
    y_base: np.ndarray,
    y_aug: np.ndarray,
    name_suffix: str = "",
    show_mae: bool = True,
) -> None:
    setup_ieee_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.0), gridspec_kw={"height_ratios": [1.0, 0.55]})
    labels = [("Power / W", 0), ("Efficiency", 1)]
    panel_titles = ["(a) Power holdout parity", "(b) Efficiency holdout parity"]
    for ax, (label, idx), ptitle in zip(axes[0], labels, panel_titles):
        ax.scatter(y_true[:, idx], y_base[:, idx], facecolors="none", edgecolors=PLASMA(0.80), linewidths=1.1, s=44, label="Sparse baseline")
        ax.scatter(y_true[:, idx], y_aug[:, idx], c=[PLASMA(0.12)], marker="s", s=26, label="Agent-augmented")
        lo = min(y_true[:, idx].min(), y_base[:, idx].min(), y_aug[:, idx].min())
        hi = max(y_true[:, idx].max(), y_base[:, idx].max(), y_aug[:, idx].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.9)
        ax.set_xlabel(f"Measured {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(ptitle, pad=4)
        ax.grid(False)
        if show_mae:
            mae_base = np.mean(np.abs(y_base[:, idx] - y_true[:, idx]))
            mae_aug = np.mean(np.abs(y_aug[:, idx] - y_true[:, idx]))
            unit = "W" if idx == 0 else ""
            scale = 1.0 if idx == 0 else 100.0
            ax.text(
                0.62,
                0.08,
                f"MAE base: {mae_base * scale:.2f}{unit}\nMAE agent: {mae_aug * scale:.2f}{unit}",
                transform=ax.transAxes,
                va="bottom",
                fontsize=7,
                bbox=_white_label(),
            )

    case_id = np.arange(1, len(y_true) + 1)
    power_pct_base = np.abs((y_base[:, 0] - y_true[:, 0]) / y_true[:, 0]) * 100
    power_pct_aug = np.abs((y_aug[:, 0] - y_true[:, 0]) / y_true[:, 0]) * 100
    eff_pct_base = np.abs((y_base[:, 1] - y_true[:, 1]) / y_true[:, 1]) * 100
    eff_pct_aug = np.abs((y_aug[:, 1] - y_true[:, 1]) / y_true[:, 1]) * 100
    for ax, base_err, aug_err, title in [
        (axes[1, 0], power_pct_base, power_pct_aug, "Power absolute percentage error"),
        (axes[1, 1], eff_pct_base, eff_pct_aug, "Efficiency absolute percentage error"),
    ]:
        ax.plot(case_id, base_err, "o-", color=PLASMA(0.80), ms=3.2, lw=1.0, label="Sparse baseline")
        ax.plot(case_id, aug_err, "s-", color=PLASMA(0.12), ms=3.2, lw=1.0, label="Agent-augmented")
        ax.set_xlabel("Holdout case")
        ax.set_ylabel("Error / %")
        ax.set_title(title, pad=4)
        ax.grid(False)
    axes[0, 0].legend(frameon=True, facecolor="white", edgecolor="0.65", loc="upper left")
    fig.suptitle("Holdout prediction audit with pointwise error comparison", y=0.995, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, f"fig03_holdout_parity{name_suffix}")


def plot_residuals(y_true: np.ndarray, y_aug: np.ndarray, name_suffix: str = "") -> None:
    setup_ieee_style()
    residual = y_aug - y_true
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.7), gridspec_kw={"height_ratios": [1.0, 0.70]})
    case = np.arange(1, len(residual) + 1)
    axes[0, 0].bar(case, residual[:, 0], color=PLASMA(0.35), edgecolor="black", linewidth=0.4)
    axes[0, 0].axhline(0, color=PLASMA(0.08), lw=0.8)
    axes[0, 0].set_xticks(np.arange(2, len(case) + 1, 2))
    axes[0, 0].set_xlabel("Holdout case")
    axes[0, 0].set_ylabel("Power residual / W")
    axes[0, 0].set_title("(a) Power residual sequence")

    eff_res = residual[:, 1] * 100.0
    axes[0, 1].bar(case, eff_res, color=PLASMA(0.75), edgecolor="black", linewidth=0.4)
    axes[0, 1].axhline(0, color=PLASMA(0.08), lw=0.8)
    axes[0, 1].set_xticks(np.arange(2, len(case) + 1, 2))
    axes[0, 1].set_xlabel("Holdout case")
    axes[0, 1].set_ylabel("Efficiency residual / %")
    axes[0, 1].set_title("(b) Efficiency residual sequence")

    axes[1, 0].hist(residual[:, 0], bins=6, color=PLASMA(0.20), edgecolor="black", alpha=0.65)
    axes[1, 0].set_xlabel("Power residual / W")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(eff_res, bins=6, color=PLASMA(0.72), edgecolor="black", alpha=0.65)
    axes[1, 1].set_xlabel("Efficiency residual / %")
    axes[1, 1].set_ylabel("Count")
    for ax in axes.ravel():
        ax.grid(False)
    fig.suptitle("Residual diagnostics for agent-augmented surrogate", y=0.995, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, f"fig04_holdout_residuals{name_suffix}")


def _rated_mesh_index(speed: np.ndarray, pressure: np.ndarray, rpm: float, p_kpa: float) -> tuple[int, int]:
    i_rpm = int(np.argmin(np.abs(speed[0, :] - rpm)))
    i_p = int(np.argmin(np.abs(pressure[:, 0] - p_kpa)))
    return i_p, i_rpm


def _panel_z_limits(eff: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Round per-panel η limits so local relief is visible on each 3D axis."""
    local_min = float(np.nanmin(eff))
    local_max = float(np.nanmax(eff))
    span = max(local_max - local_min, 0.8)
    pad = max(0.4, 0.10 * span)
    step = 1.0 if span <= 6.0 else 2.0
    z_lo = step * np.floor((local_min - pad) / step)
    z_hi = step * np.ceil((local_max + pad) / step)
    z_ticks = np.arange(z_lo, z_hi + 0.25 * step, step)
    if z_ticks.size < 3:
        z_ticks = np.linspace(z_lo, z_hi, 3)
    elif z_ticks.size > 5:
        z_ticks = np.linspace(z_lo, z_hi, 4)
    return z_lo, z_hi, z_ticks


def _eval_efficiency_maps(
    source_temperatures: tuple[float, ...],
    pressure: np.ndarray,
    speed: np.ndarray,
    model,
    rk_oracle: Callable[[float, float, float], tuple[float, float]] | None,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    from scipy.interpolate import RegularGridInterpolator

    ss, pp = np.meshgrid(speed, pressure)
    maps: list[tuple[float, np.ndarray, np.ndarray]] = []
    for source_temperature in source_temperatures:
        t_c = float(source_temperature) - 273.15
        if rk_oracle is not None:
            try:
                from yanmar.engine_specs import PRESSURE_LEVELS_KPA, SPEED_LEVELS_RPM
            except ImportError:
                PRESSURE_LEVELS_KPA = tuple(float(p) for p in pressure)  # type: ignore[misc]
                SPEED_LEVELS_RPM = tuple(float(s) for s in speed)  # type: ignore[misc]
            p_axis = np.asarray(PRESSURE_LEVELS_KPA, dtype=float)
            s_axis = np.asarray(SPEED_LEVELS_RPM, dtype=float)
            eff_grid = np.zeros((p_axis.size, s_axis.size))
            pow_grid = np.zeros((p_axis.size, s_axis.size))
            for i, p_kpa in enumerate(p_axis):
                for j, rpm in enumerate(s_axis):
                    p_w, eff_pct = rk_oracle(t_c, float(p_kpa), float(rpm))
                    pow_grid[i, j] = p_w
                    eff_grid[i, j] = eff_pct
            eff_interp = RegularGridInterpolator(
                (p_axis, s_axis), eff_grid, bounds_error=False, fill_value=np.nan
            )
            pow_interp = RegularGridInterpolator(
                (p_axis, s_axis), pow_grid, bounds_error=False, fill_value=np.nan
            )
            pts = np.column_stack([pp.ravel(), ss.ravel()])
            eff = eff_interp(pts).reshape(pp.shape)
            power = pow_interp(pts).reshape(pp.shape)
        else:
            eff = np.empty(ss.shape)
            power = np.empty(ss.shape)
            for ip in range(pp.shape[0]):
                for ir in range(pp.shape[1]):
                    p_kpa = float(pp[ip, ir])
                    rpm = float(ss[ip, ir])
                    x = np.array([[source_temperature, p_kpa, rpm]])
                    mean = model.predict(x)
                    power[ip, ir] = mean[0, 0]
                    eff[ip, ir] = mean[0, 1] * 100.0
        maps.append((source_temperature, eff, power))
    return maps


def plot_efficiency_surface(
    model,
    source_temperatures: tuple[float, ...] = TEMP_LEVELS_K,
    name_suffix: str = "",
    display_temperatures: tuple[float, ...] | None = None,
    pressure_range: tuple[float, float] = (500.0, 2500.0),
    speed_range: tuple[float, float] = (200.0, 3000.0),
    rated_point: tuple[float, float, float] | None = None,
    rk_oracle: Callable[[float, float, float], tuple[float, float]] | None = None,
    surface_caption: str | None = None,
    map_caption: str | None = None,
    z_label: str = "Efficiency / %",
) -> None:
    # ``source_temperatures`` are fed to the model / RK oracle;
    # ``display_temperatures`` (if given) are shown in the panel titles.
    if display_temperatures is None:
        display_temperatures = source_temperatures
    setup_ieee_style()
    p_lo, p_hi = pressure_range
    s_lo, s_hi = speed_range
    pressure = np.linspace(p_lo, p_hi, 60)
    speed = np.linspace(s_lo, s_hi, 60)
    ss, pp = np.meshgrid(speed, pressure)

    raw_maps = _eval_efficiency_maps(source_temperatures, pressure, speed, model, rk_oracle)
    maps: list[tuple[float, np.ndarray, np.ndarray, tuple[int, int]]] = []
    for source_temperature, eff, power in raw_maps:
        if rated_point is not None:
            _ts, rated_p, rated_rpm = rated_point
            star = _rated_mesh_index(ss, pp, rated_rpm, rated_p)
        else:
            star = np.unravel_index(np.argmax(eff), eff.shape)
        maps.append((source_temperature, eff, power, star))

    eff_min = min(float(np.nanmin(eff)) for _, eff, _, _ in maps)
    eff_max = max(float(np.nanmax(eff)) for _, eff, _, _ in maps)
    levels = np.linspace(eff_min, eff_max, 18)
    x_ticks = [s_lo, (s_lo + s_hi) / 2, s_hi]
    y_ticks = [p_lo, (p_lo + p_hi) / 2, p_hi]

    default_surface_caption = (
        "Star: catalogue rated point (2.8 MPa, 800 rpm). "
        "Surfaces from RK η_indicated = W/Q_in."
    )
    default_map_caption = (
        "Star: catalogue rated point; isolines: RK electrical power / W."
    )
    surface_caption = surface_caption or default_surface_caption
    map_caption = map_caption or default_map_caption

    fig3d = plt.figure(figsize=(9.8, 3.65))
    norm = matplotlib.colors.Normalize(vmin=eff_min, vmax=eff_max)
    for idx, (source_temperature, eff, _power, star) in enumerate(maps, start=1):
        ax = fig3d.add_subplot(1, len(maps), idx, projection="3d")
        panel_z_lo, panel_z_hi, panel_z_ticks = _panel_z_limits(eff)
        ax.plot_surface(
            ss,
            pp,
            eff,
            cmap=PLASMA_CMAP,
            norm=norm,
            linewidth=0,
            antialiased=True,
            alpha=0.94,
            shade=True,
            rcount=60,
            ccount=60,
        )
        ax.scatter(
            ss[star],
            pp[star],
            eff[star],
            c=[PLASMA(0.05)],
            s=46,
            marker="*",
            edgecolors="white",
            linewidths=0.45,
            depthshade=False,
        )
        ax.set_title(f"({chr(96 + idx)}) $T_s$ = {display_temperatures[idx - 1]:.0f} {TEMP_UNIT}", pad=4)
        ax.set_xlabel("Speed / rpm", labelpad=-3)
        ax.set_ylabel("Pressure / kPa", labelpad=-5)
        ax.set_zlabel(z_label, labelpad=-6)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_zticks(panel_z_ticks)
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.tick_params(axis="x", pad=-2)
        ax.tick_params(axis="y", pad=-2)
        ax.tick_params(axis="z", pad=-1)
        ax.set_zlim(panel_z_lo, panel_z_hi)
        ax.view_init(elev=25, azim=-55)
        ax.set_box_aspect((1.25, 1.0, 0.85))
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.55)
            axis._axinfo["grid"]["linewidth"] = 0.45
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("0.78")
        ax.yaxis.pane.set_edgecolor("0.78")
        ax.zaxis.pane.set_edgecolor("0.78")

    cax3d = fig3d.add_axes([0.930, 0.235, 0.016, 0.560])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=PLASMA_CMAP)
    cb3d = fig3d.colorbar(sm, cax=cax3d)
    cb3d.set_label(z_label)
    fig3d.text(0.50, 0.035, surface_caption, ha="center", fontsize=7, bbox=_white_label())
    fig3d.subplots_adjust(left=0.020, right=0.890, bottom=0.155, top=0.825, wspace=0.180)
    fig3d.suptitle("Figure 5.1  Thermal-source sensitivity of 3D efficiency surfaces", y=0.995, weight="bold")
    save_figure(fig3d, f"fig05_1_efficiency_surfaces_3d{name_suffix}")

    fig, axes = plt.subplots(1, len(maps), figsize=(8.9, 3.45), sharex=True, sharey=True)
    contour = None
    for ax, (_source_temperature, eff, power, star), panel, disp_temp in zip(
        axes, maps, ["(a)", "(b)", "(c)"], display_temperatures
    ):
        contour = ax.contourf(ss, pp, eff, levels=levels, cmap=PLASMA_CMAP)
        cs = ax.contour(ss, pp, power, levels=7, colors="0.15", linewidths=0.65, alpha=0.85)
        ax.clabel(cs, inline=True, fmt="%.0f W", fontsize=6, colors="0.15")
        ax.scatter(
            ss[star],
            pp[star],
            c=[PLASMA(0.05)],
            s=62,
            marker="*",
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        ax.set_title(f"{panel} $T_s$ = {disp_temp:.0f} {TEMP_UNIT}", pad=6)
        ax.set_xlabel("Speed / rpm")
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(False)

    axes[0].set_ylabel("Pressure / kPa")
    fig.text(0.50, 0.035, map_caption, ha="center", fontsize=7, bbox=_white_label())
    fig.subplots_adjust(left=0.070, right=0.890, bottom=0.210, top=0.810, wspace=0.150)
    if contour is not None:
        cax = fig.add_axes([0.915, 0.210, 0.018, 0.610])
        cb = fig.colorbar(contour, cax=cax)
        cb.set_label(z_label)
    fig.suptitle("Figure 5.2  Thermal-source sensitivity of efficiency maps", y=0.995, weight="bold")
    save_figure(fig, f"fig05_2_efficiency_maps{name_suffix}")


def plot_uncertainty_map(
    model,
    source_temperature: float = TEMP_LEVELS_K[1],
    name_suffix: str = "",
    pressure_range: tuple[float, float] = (500.0, 2500.0),
    speed_range: tuple[float, float] = (200.0, 3000.0),
    rated_point: tuple[float, float, float] | None = None,
    caption: str | None = None,
    uncertainty_quantiles: tuple[float, ...] = (0.75, 0.85, 0.95),
) -> None:
    setup_ieee_style()
    p_lo, p_hi = pressure_range
    s_lo, s_hi = speed_range
    pressure = np.linspace(p_lo, p_hi, 60)
    speed = np.linspace(s_lo, s_hi, 60)
    ss, pp = np.meshgrid(speed, pressure)
    x = np.column_stack([np.full(ss.size, source_temperature), pp.ravel(), ss.ravel()])
    mean, std = model.predict(x, return_std=True)
    power_std = std[:, 0].reshape(pp.shape)
    power_mean = mean[:, 0].reshape(pp.shape)
    risk_adjusted = power_mean - 1.96 * power_std
    quantile_levels = tuple(sorted({float(q) for q in uncertainty_quantiles if 0.0 < q < 1.0}))
    if not quantile_levels:
        quantile_levels = (0.75, 0.85, 0.95)
    unc_thresholds = [float(np.quantile(power_std, q)) for q in quantile_levels]
    mean_contour_lw = 0.45
    percentile_contour_lw = 1.55
    x_ticks = [s_lo, (s_lo + s_hi) / 2, s_hi]
    y_ticks = [p_lo, (p_lo + p_hi) / 2, p_hi]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.45), gridspec_kw={"width_ratios": [1, 1]})
    im = axes[0].contourf(ss, pp, power_std, levels=16, cmap=PLASMA_CMAP)
    for q, thresh in zip(quantile_levels, unc_thresholds, strict=True):
        axes[0].contour(
            ss,
            pp,
            power_std,
            levels=[thresh],
            colors="white",
            linewidths=percentile_contour_lw,
            linestyles="dashed",
            zorder=3,
        )
    axes[0].set_xlabel("Speed / rpm")
    axes[0].set_ylabel("Pressure / kPa")
    axes[0].set_xticks(x_ticks)
    axes[0].set_yticks(y_ticks)
    axes[0].set_title("(a) Epistemic power uncertainty")
    cb = fig.colorbar(im, ax=axes[0], pad=0.02)
    cb.set_label("Power uncertainty / W")

    im2 = axes[1].contourf(ss, pp, risk_adjusted, levels=16, cmap=PLASMA_CMAP)
    axes[1].contour(ss, pp, power_mean, levels=8, colors="0.25", linewidths=mean_contour_lw, alpha=0.7)
    axes[1].set_xlabel("Speed / rpm")
    axes[1].set_ylabel("Pressure / kPa")
    axes[1].set_xticks(x_ticks)
    axes[1].set_yticks(y_ticks)
    axes[1].set_title("(b) Conservative power estimate")
    cb2 = fig.colorbar(im2, ax=axes[1], pad=0.02)
    cb2.set_label("Conservative power / W")

    pct_labels = ", ".join(f"{int(round(100 * q))}th" for q in quantile_levels)
    default_caption = (
        f"(a) White dashed contours: {pct_labels} percentiles of map-wide ensemble spread; "
        "(b) conservative power = mean minus 1.96 times ensemble spread."
    )
    fig.text(0.50, 0.035, caption or default_caption, ha="center", fontsize=7, bbox=_white_label())
    fig.suptitle("Uncertainty-aware operating map for confidence-aware dispatch", y=0.995, weight="bold")
    fig.tight_layout(rect=[0, 0.10, 1, 0.93])
    save_figure(fig, f"fig06_power_uncertainty_map{name_suffix}")

    if len(quantile_levels) == 1:
        legend_label = f"{int(round(100 * quantile_levels[0]))}th percentile of map-wide ensemble spread"
    else:
        legend_label = f"Percentile contours ({pct_labels}) of map-wide ensemble spread"
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="0.35",
            linewidth=percentile_contour_lw,
            linestyle="--",
            label=legend_label,
        ),
    ]
    legend_fig = plt.figure(figsize=(4.6, 0.55))
    legend = legend_fig.legend(
        handles=legend_handles,
        loc="center",
        frameon=True,
        facecolor="white",
        edgecolor="0.65",
        framealpha=0.96,
        fontsize=7,
        handlelength=1.4,
    )
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")
        text.set_fontsize(7)
    FIG_DIR.mkdir(exist_ok=True)
    legend_name = f"fig06_power_uncertainty_map_legend{name_suffix}"
    legend_fig.savefig(FIG_DIR / f"{legend_name}.png", bbox_inches="tight", transparent=True, pad_inches=0.02)
    legend_fig.savefig(FIG_DIR / f"{legend_name}.pdf", bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(legend_fig)


def plot_dispatch(dispatch: pd.DataFrame) -> None:
    setup_ieee_style()
    daily = (
        dispatch.groupby("day", as_index=False)
        .agg(
            whr_kw=("whr_kw", "mean"),
            pv_kw=("pv_kw", "mean"),
            storage_power_kw=("storage_power_kw", "mean"),
            load_kw=("load_kw", "mean"),
            whr_uncertainty_kw=("whr_uncertainty_kw", "mean"),
            served_balance_kw=("served_balance_kw", "mean"),
        )
        .sort_values("day")
    )

    fig = plt.figure(figsize=(8.6, 4.2))
    ax = fig.add_axes([0.08, 0.14, 0.58, 0.76])
    whr_color = PLASMA(0.18)
    pv_color = PLASMA(0.72)
    storage_color = PLASMA(0.48)
    ax.scatter(daily["day"], daily["whr_kw"], s=13, marker="x", c=[whr_color], lw=0.65, label="Stirling Engine")
    ax.scatter(daily["day"], daily["pv_kw"], s=13, marker="*", c=[pv_color], alpha=0.72, label="Photovoltaic")
    ax.scatter(
        daily["day"],
        daily["storage_power_kw"],
        s=12,
        facecolors="none",
        edgecolors=[storage_color],
        lw=0.65,
        label="Energy Storage",
    )
    ax.plot(daily["day"], daily["load_kw"], color="0.15", lw=0.8, alpha=0.45, label="Load")
    ax.fill_between(
        daily["day"].to_numpy(),
        (daily["whr_kw"] - daily["whr_uncertainty_kw"]).to_numpy(),
        (daily["whr_kw"] + daily["whr_uncertainty_kw"]).to_numpy(),
        color=whr_color,
        alpha=0.10,
        lw=0,
    )
    ax.axhline(0, color="0.25", lw=0.7)
    ax.set_xlim(0, 365)
    ax.set_ylim(-850, 3500)
    ax.set_xlabel("Time Series / days")
    ax.set_ylabel("Power / kW")
    ax.legend(frameon=True, edgecolor="0.25", facecolor="white", loc="upper left", fontsize=7)
    ax.grid(True, ls=":", lw=0.45, color="0.75")

    zoom_start, zoom_end = 292, 314
    rect = plt.Rectangle((zoom_start, -800), zoom_end - zoom_start, 4200, fill=False, ec=PLASMA(0.92), lw=1.1)
    ax.add_patch(rect)
    ax.annotate(
        "Energy Complementarity",
        xy=(zoom_end, 2700),
        xytext=(218, 3020),
        color=PLASMA(0.90),
        weight="bold",
        bbox=_white_label(),
        arrowprops=dict(arrowstyle="->", color=PLASMA(0.90), lw=1.0),
    )

    inset_specs = [
        ([0.70, 0.67, 0.25, 0.22], "whr_kw", "Stirling Engine Output", whr_color),
        ([0.70, 0.40, 0.25, 0.22], "pv_kw", "PV Output", pv_color),
        ([0.70, 0.13, 0.25, 0.22], "storage_power_kw", "Energy Storage", storage_color),
    ]
    zoom = daily[(daily["day"] >= zoom_start) & (daily["day"] <= zoom_end)]
    for idx, (spec, col, title, color) in enumerate(inset_specs):
        iax = fig.add_axes(spec)
        iax.plot(zoom["day"], zoom[col], color=color, lw=1.0, marker="o", ms=2.2)
        iax.axvspan(302, 307, color=PLASMA(0.92), alpha=0.12)
        iax.set_title(title, fontsize=7, pad=4, bbox=dict(fc="white", ec="none", alpha=0.95))
        iax.set_xlim(zoom_start, zoom_end)
        iax.tick_params(labelsize=6, length=2)
        iax.grid(True, ls=":", lw=0.35, color="0.75")
        for spine in iax.spines.values():
            spine.set_linestyle((0, (4, 3)))
            spine.set_linewidth(0.9)
            spine.set_edgecolor("#1f3b73")

    fig.text(0.822, 0.05, "Zoomed operating window / days", ha="center", fontsize=8, bbox=_white_label())
    save_figure(fig, "fig07_confidence_aware_dispatch")


def plot_dispatch_hourly(dispatch: pd.DataFrame, *, whr_dispatch: pd.DataFrame | None = None) -> None:
    setup_ieee_style()
    dispatch = dispatch.copy()
    if whr_dispatch is not None:
        whr_by_hour = whr_dispatch.set_index("hour")["whr_kw"]
        dispatch["whr_kw"] = dispatch["hour"].map(whr_by_hour).astype(float)
    dispatch["time_day"] = dispatch["hour"] / 24.0
    t = dispatch["time_day"].to_numpy(float)

    fig = plt.figure(figsize=(8.8, 4.35))
    ax = fig.add_axes([0.08, 0.15, 0.58, 0.75])
    load_color = "0.62"
    whr_color = "#2166AC"
    storage_color = PLASMA(0.72)
    pv_color = PLASMA(0.96)

    # Annual hourly overview as line traces.
    ax.plot(t, dispatch["load_kw"], color=load_color, lw=0.60, alpha=0.92, rasterized=True)
    ax.plot(t, dispatch["whr_kw"], color=whr_color, lw=0.60, alpha=0.92, rasterized=True)
    ax.plot(t, dispatch["pv_kw"], color=pv_color, lw=0.60, alpha=0.92, rasterized=True)
    ax.plot(t, dispatch["storage_power_kw"], color=storage_color, lw=0.55, alpha=0.92, rasterized=True)
    ax.axhline(0, color="0.25", lw=0.7)
    power_cols = ["load_kw", "whr_kw", "pv_kw", "storage_power_kw"]
    y_vals = dispatch[power_cols].to_numpy(float).ravel()
    y_pad = max(120.0, 0.08 * float(np.nanmax(np.abs(y_vals))))
    y_lo = float(np.nanmin(y_vals)) - y_pad
    y_hi = float(np.nanmax(y_vals)) + y_pad
    ax.set_xlim(0, 365)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Time Series / days (hourly samples)")
    ax.set_ylabel("Power / kW")
    ax.set_title("(a) Annual hourly campus dispatch", fontname="Times New Roman", fontsize=10, pad=7)
    ax.grid(True, ls=":", lw=0.42, color="0.78")

    zoom_start, zoom_end = 300.0, 307.0
    y_span = y_hi - y_lo
    rect = plt.Rectangle(
        (zoom_start, y_lo + 0.02 * y_span),
        zoom_end - zoom_start,
        0.96 * y_span,
        fill=False,
        ec="#8B4513",
        lw=1.1,
        zorder=100,
    )
    ax.add_patch(rect)
    rect.set_clip_on(False)

    zoom = dispatch[(dispatch["time_day"] >= zoom_start) & (dispatch["time_day"] <= zoom_end)]
    inset_specs = [
        ([0.725, 0.705, 0.25, 0.175], "load_kw", load_color),
        ([0.725, 0.505, 0.25, 0.175], "whr_kw", whr_color),
        ([0.725, 0.305, 0.25, 0.175], "pv_kw", pv_color),
        ([0.725, 0.105, 0.25, 0.175], "storage_power_kw", storage_color),
    ]
    for idx, (spec, col, color) in enumerate(inset_specs):
        iax = fig.add_axes(spec)
        series = zoom[col].to_numpy(float)
        iax.plot(zoom["time_day"], series, color=color, lw=0.95)
        iax.axvspan(302.0, 304.0, color=PLASMA(0.92), alpha=0.12)
        if col == "storage_power_kw":
            y_pad = max(40.0, 0.12 * float(np.max(np.abs(series))))
            iax.set_ylim(float(series.min()) - y_pad, float(series.max()) + y_pad)
            iax.axhline(0, color="0.35", lw=0.5)
        iax.set_xlim(zoom_start, zoom_end)
        iax.tick_params(labelsize=6, length=2)
        if idx < len(inset_specs) - 1:
            iax.set_xticklabels([])
        iax.grid(False)
        for spine in iax.spines.values():
            spine.set_linestyle((0, (4, 3)))
            spine.set_linewidth(0.9)
            spine.set_edgecolor("#8B4513")
    fig.text(0.847, 0.895, "(b) Zoomed operating window", ha="center", fontsize=10, fontname="Times New Roman")
    fig.text(0.847, 0.010, "Zoomed operating window / days", ha="center", fontsize=9)
    save_figure(fig, "fig07_hourly_confidence_aware_dispatch")

    legend_handles = [
        Line2D([0], [0], color=load_color, lw=1.2, label="Load"),
        Line2D([0], [0], color=whr_color, lw=1.2, label="Stirling Engine"),
        Line2D([0], [0], color=pv_color, lw=1.2, label="Photovoltaic"),
        Line2D([0], [0], color=storage_color, lw=1.2, label="Energy Storage"),
        Line2D([0], [0], color=load_color, lw=1.2, label="Hourly Load"),
        Line2D([0], [0], color=whr_color, lw=1.2, label="Hourly Stirling"),
        Line2D([0], [0], color=pv_color, lw=1.2, label="Hourly PV Output"),
        Line2D([0], [0], color=storage_color, lw=1.2, label="Hourly Storage"),
    ]
    legend_fig = plt.figure(figsize=(8.4, 1.35), facecolor="none")
    legend = legend_fig.legend(
        handles=legend_handles,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=2.0,
        columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")
        text.set_fontsize(9)
    FIG_DIR.mkdir(exist_ok=True)
    legend_path = FIG_DIR / "fig07_hourly_confidence_aware_dispatch_legend"
    legend_fig.savefig(f"{legend_path}.png", bbox_inches="tight", transparent=True, pad_inches=0.08)
    legend_fig.savefig(f"{legend_path}.pdf", bbox_inches="tight", transparent=True, pad_inches=0.08)
    plt.close(legend_fig)


def plot_dispatch_hourly_agent_comparison(baseline_dispatch: pd.DataFrame, augmented_dispatch: pd.DataFrame) -> None:
    setup_ieee_style()
    base = baseline_dispatch.copy()
    aug = augmented_dispatch.copy()
    base["time_day"] = base["hour"] / 24.0
    aug["time_day"] = aug["hour"] / 24.0

    zoom_start, zoom_end = 300.0, 307.0
    bz = base[(base["time_day"] >= zoom_start) & (base["time_day"] <= zoom_end)]
    az = aug[(aug["time_day"] >= zoom_start) & (aug["time_day"] <= zoom_end)]

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.6), sharex=True)
    whr_base_color = PLASMA(0.72)
    whr_aug_color = PLASMA(0.14)
    unc_base_color = "#C0392B"
    unc_aug_color = "#117A65"

    axes[0].plot(bz["time_day"], bz["whr_kw"], color=whr_base_color, lw=1.35, ls="--", label="WHR sparse baseline")
    axes[0].plot(az["time_day"], az["whr_kw"], color=whr_aug_color, lw=1.45, label="WHR case-augmented")
    axes[0].set_ylabel("Power / kW")
    axes[0].set_title("(c) Seven-day hourly WHR dispatch response", fontname="Times New Roman", fontsize=10, pad=7)

    base_unc_pct = 100.0 * bz["whr_uncertainty_kw"].to_numpy(float) / np.maximum(np.abs(bz["whr_kw"].to_numpy(float)), 1e-12)
    aug_unc_pct = 100.0 * az["whr_uncertainty_kw"].to_numpy(float) / np.maximum(np.abs(az["whr_kw"].to_numpy(float)), 1e-12)
    axes[1].plot(bz["time_day"], base_unc_pct, color=unc_base_color, lw=1.25, ls="-.", label="Uncertainty sparse baseline")
    axes[1].plot(az["time_day"], aug_unc_pct, color=unc_aug_color, lw=1.35, ls="-", label="Uncertainty case-augmented")
    axes[1].fill_between(
        az["time_day"].to_numpy(),
        aug_unc_pct,
        base_unc_pct,
        where=(base_unc_pct >= aug_unc_pct),
        color=unc_aug_color,
        alpha=0.14,
        lw=0,
    )
    axes[1].set_ylabel("Relative WHR uncertainty / %")
    axes[1].set_xlabel("Zoomed operating window / days")
    axes[1].set_title("(d) Seven-day WHR model-confidence gain", fontname="Times New Roman", fontsize=10, pad=7)

    for ax in axes:
        ax.set_xlim(zoom_start, zoom_end)
        ax.grid(True, ls=":", lw=0.42, color="0.78")
        ax.tick_params(axis="both", labelsize=10)

    handles_a, labels_a = axes[0].get_legend_handles_labels()
    handles_b, labels_b = axes[1].get_legend_handles_labels()
    fig.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        edgecolor="0.25",
        facecolor="white",
        fontsize=8,
        handlelength=2.0,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0.015, 1, 0.94], h_pad=1.8)
    save_figure(fig, "fig07_hourly_case_augmentation_dispatch")

    legend_handles = [
        Line2D([0], [0], color=whr_base_color, lw=1.45, ls="--", label="WHR sparse baseline"),
        Line2D([0], [0], color=whr_aug_color, lw=1.55, label="WHR case-augmented"),
        Line2D([0], [0], color=unc_base_color, lw=1.35, ls="-.", label="Uncertainty sparse baseline"),
        Line2D([0], [0], color=unc_aug_color, lw=1.45, label="Uncertainty case-augmented"),
    ]
    legend_fig = plt.figure(figsize=(5.8, 0.72), facecolor="none")
    legend = legend_fig.legend(
        handles=legend_handles,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=2.2,
        columnspacing=1.5,
    )
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")
        text.set_fontsize(10)
    FIG_DIR.mkdir(exist_ok=True)
    legend_fig.savefig(FIG_DIR / "fig07_hourly_case_augmentation_dispatch_legend.png", bbox_inches="tight", transparent=True, pad_inches=0.02)
    legend_fig.savefig(FIG_DIR / "fig07_hourly_case_augmentation_dispatch_legend.pdf", bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(legend_fig)


def _save_campus_load_legend(component_specs: list[tuple[str, str, tuple]], legend_name: str) -> None:
    legend_handles = [
        Line2D([0], [0], color=color, lw=1.5, label=label) for _, label, color in component_specs
    ]
    legend_fig = plt.figure(figsize=(5.6, 0.72), facecolor="none")
    legend = legend_fig.legend(
        handles=legend_handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=2.0,
        columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")
        text.set_fontsize(9)
    FIG_DIR.mkdir(exist_ok=True)
    legend_path = FIG_DIR / legend_name
    legend_fig.savefig(f"{legend_path}.png", bbox_inches="tight", transparent=True, pad_inches=0.04)
    legend_fig.savefig(f"{legend_path}.pdf", bbox_inches="tight", transparent=True, pad_inches=0.04)
    plt.close(legend_fig)


def _plot_campus_load_breakdown_figure(
    df: pd.DataFrame,
    t: np.ndarray,
    component_specs: list[tuple[str, str, tuple]],
    figure_name: str,
    legend_name: str,
) -> None:
    fig = plt.figure(figsize=(8.6, 5.8))
    gs = fig.add_gridspec(len(component_specs), 2, width_ratios=[0.24, 0.76], wspace=0.28, hspace=0.38)

    for row, (col, label, color) in enumerate(component_specs):
        series = df[col].to_numpy(float)
        daily_stats = df.groupby("day", as_index=False).agg(
            daily_kwh=(col, "sum"),
            daily_peak=(col, "max"),
        )
        mean_daily_kwh = float(daily_stats["daily_kwh"].mean())
        peak_daily_kw = float(daily_stats["daily_peak"].max())

        info_ax = fig.add_subplot(gs[row, 0])
        info_ax.set_xlim(0, 1)
        info_ax.set_ylim(0, 1)
        info_ax.axis("off")
        box_x, box_w, box_h = 0.02, 0.82, 0.78
        text_x = box_x + box_w / 2.0
        info_box = FancyBboxPatch(
            (box_x, 0.11),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            fc=(*color[:3], 0.12),
            ec=color,
            lw=1.0,
        )
        info_ax.add_patch(info_box)
        info_ax.text(
            text_x,
            0.72,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            weight="bold",
            color=color,
        )
        info_ax.text(
            text_x,
            0.46,
            f"Daily Consumption:\n{mean_daily_kwh:,.1f} kWh/day",
            ha="center",
            va="center",
            fontsize=8.5,
        )
        info_ax.text(
            text_x,
            0.22,
            f"Daily Peak Power:\n{peak_daily_kw:,.1f} kW",
            ha="center",
            va="center",
            fontsize=8.5,
        )

        ax = fig.add_subplot(gs[row, 1])
        ax.plot(t, series, color=color, lw=0.55, alpha=0.95, rasterized=True)
        y_pad = max(8.0, 0.10 * float(np.max(series)))
        ax.set_ylim(max(0.0, float(np.min(series)) - 0.05 * y_pad), float(np.max(series)) + y_pad)
        ax.set_xlim(0, 365)
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350])
        ax.set_ylabel("Power / kW", fontsize=9, labelpad=4)
        ax.grid(True, ls=":", lw=0.42, color="0.78")
        ax.tick_params(axis="both", labelsize=9)
        ax.set_title(f"Annual load curve for {label}", fontname="Times New Roman", fontsize=11.5, pad=5)
        if row == len(component_specs) - 1:
            ax.set_xlabel("Time / days")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.08)
    save_figure(fig, figure_name)
    _save_campus_load_legend(component_specs, legend_name)


def plot_campus_load_breakdown(profiles: pd.DataFrame) -> None:
    """Separate annual hourly curves for each campus-load component (two figures, three loads each)."""
    setup_ieee_style()
    df = profiles.copy()
    df["time_day"] = df["hour"] / 24.0
    t = df["time_day"].to_numpy(float)
    all_specs = [
        ("load_manufacturing_kw", "Manufacturing", PLASMA(0.92)),
        ("load_hvac_heating_kw", "HVAC heating and steam", PLASMA(0.78)),
        ("load_hvac_cooling_kw", "HVAC cooling and ventilation", PLASMA(0.64)),
        ("load_refrigeration_kw", "Refrigeration", PLASMA(0.50)),
        ("load_office_campus_kw", "Office and campus aux.", PLASMA(0.36)),
        ("load_warehouse_aux_kw", "Warehouse aux.", PLASMA(0.14)),
    ]
    _plot_campus_load_breakdown_figure(
        df,
        t,
        all_specs[:3],
        "fig07_campus_load_breakdown_1",
        "fig07_campus_load_breakdown_legend_1",
    )
    _plot_campus_load_breakdown_figure(
        df,
        t,
        all_specs[3:],
        "fig07_campus_load_breakdown_2",
        "fig07_campus_load_breakdown_legend_2",
    )

