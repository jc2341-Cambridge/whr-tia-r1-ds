"""R1 figures: Figs. 6-13 regenerated with the calibrated solver and the main-pipeline batch,
new Fig. 10 (selector benchmark) and Fig. 17 (oracle-evaluated dispatch).

    python -m yanmar.r1_make_figures      (from journal_extension/)

Writes writing/Yanmar/figures_r1/ and results/r1_figures_metrics.json.
"""
from __future__ import annotations

import matplotlib  # noqa: F401  (must precede pandas on this install)

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data_loader as dl  # noqa: E402
import make_figures as mf  # noqa: E402
from data_loader import FEATURE_COLUMNS  # noqa: E402
from physics_constraints import residual_metrics  # noqa: E402
from yanmar.active_sampling_benchmark import _fit  # noqa: E402
from yanmar.dataset import YANMAR_ROOT, feature_target_arrays, load_operating_data, sparse_active_learning_split, temp_levels_k  # noqa: E402
from yanmar.engine_specs import YANMAR  # noqa: E402
from yanmar.r1_dispatch import RESULT_DIR as DISPATCH_DIR  # noqa: E402
from yanmar.r1_pool_benchmark import PROPOSER_DIR  # noqa: E402
from yanmar.r1_pool_revised import MAIN_BATCH  # noqa: E402
from yanmar.stirling_rk_solver import rk_power_efficiency  # noqa: E402

FIG_DIR = YANMAR_ROOT / "figures_r1"


def _setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mf.FIG_DIR = FIG_DIR
    mf.TEMP_UNIT = "K"
    levels = temp_levels_k()
    t_lo, t_mid, t_hi = levels
    mf.TEMP_SPAN = t_hi - t_lo
    dl.TEMP_LEVELS_K = levels
    dl.TEMP_K_LO, dl.TEMP_K_MID, dl.TEMP_K_HI = t_lo, t_mid, t_hi
    mf.TEMP_K_LO, mf.TEMP_K_HI, mf.TEMP_LEVELS_K = t_lo, t_hi, levels


def _relabel_coverage_plot() -> None:
    import inspect

    src = inspect.getsource(mf.plot_data_coverage)
    for old, new in (("Measured training cases", "Solver-labelled seed cases"),
                     ("Measured efficiency / %", "RK4-labelled efficiency / %"),
                     ("Agent-selected cases", "Selected cases (K = 28)"),
                     ("sparse regions of the measured space", "sparse regions of the seed design")):
        src = src.replace(old, new)
    exec(src, mf.__dict__)


def _save_split(fig, name: str) -> None:
    """Save the left and right halves of a two-panel figure as separate files."""
    from matplotlib.transforms import Bbox

    for text in fig.findobj(match=matplotlib.text.Text):
        text.set_fontfamily("Times New Roman")
        if text.get_fontsize() < 10:
            text.set_fontsize(10)
    for t in list(fig.texts):
        t.set_visible(False)
    leg = fig.legends[0] if fig.legends else None
    if leg is not None:
        leg.set_title(None)
        leg.set_bbox_to_anchor((0.0, 0.86), transform=fig.transFigure)
        leg._loc = 2
        for t in leg.get_texts():
            t.set_fontsize(8.5)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    left = [a for a in fig.axes if a.get_position().x0 < 0.45]
    right = [a for a in fig.axes if a.get_position().x0 >= 0.45]
    for tag, group, extra in (("a", left, [leg] if leg is not None else []), ("b", right, [])):
        boxes = [a.get_tightbbox(r) for a in group] + [e.get_window_extent(r) for e in extra]
        bb = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(0.06)
        fig.savefig(FIG_DIR / f"{name}_{tag}.png", bbox_inches=bb, dpi=600)
    plt.close(fig)


def surrogate_figures() -> dict:
    df = load_operating_data()
    train, _oracle, test = sparse_active_learning_split(df)
    x_test, y_test = feature_target_arrays(test)
    sel = pd.read_csv(MAIN_BATCH).assign(label_source="rk4_solver")
    baseline = _fit(train, 21)
    augmented = _fit(pd.concat([train.assign(label_source="rk4_grid"),
                                sel[FEATURE_COLUMNS + ["power", "efficiency", "label_source"]]], ignore_index=True), 22)
    y_base, y_aug = baseline.predict(x_test), augmented.predict(x_test)
    metrics = {"baseline": residual_metrics(y_test, y_base), "agent_augmented": residual_metrics(y_test, y_aug)}

    _relabel_coverage_plot()
    orig_save = mf.save_figure
    mf.save_figure = _save_split
    try:
        mf.plot_data_coverage(
            train, test, sel,
            panel_a_xlim=(400.0, 1300.0), panel_a_xticks=[400, 700, 1000, 1300], panel_a_ylim=(1500.0, 3000.0),
            panel_a_yticks=[1750, 2000, 2250, 2500, 2750], panel_a_zlim=(700.0, 1100.0),
            panel_a_zticks=[700, 800, 900, 1000, 1100],
        )
    finally:
        mf.save_figure = orig_save
    def _save_clean(fig, name):
        if fig._suptitle is not None:
            fig._suptitle.set_visible(False)
        for t in fig.texts:
            t.set_visible(False)
        orig_save(fig, name)

    mf.save_figure = _save_clean
    import inspect

    exec(inspect.getsource(mf.plot_parity).replace('f"Measured {label}"', 'f"RK4-labelled {label}"'), mf.__dict__)
    mf.plot_parity(y_test, y_base, y_aug, name_suffix="_rep3", show_mae=False)
    mf.plot_residuals(y_test, y_aug, name_suffix="_rep3")
    mf.plot_agent_decision_audit(sel, metrics, train=train, name="fig08_agent_decision_audit")

    levels = temp_levels_k()
    rated = (YANMAR.temp_rated_c + 273.15, YANMAR.pressure_rated_kpa, YANMAR.speed_rated_rpm)
    mf.plot_efficiency_surface(
        augmented, source_temperatures=levels, name_suffix="_rep3", pressure_range=(1500.0, 3000.0),
        speed_range=(400.0, 1200.0), rated_point=rated, rk_oracle=rk_power_efficiency,
        surface_caption="Star: Yanmar catalogue rated point (750°C, 2.8 MPa, 800 rpm). Electrical efficiency.",
        map_caption="Star: catalogue rated point (2.8 MPa, 800 rpm); isolines: electrical power.",
        z_label="Efficiency / %",
    )
    mf.plot_uncertainty_map(
        augmented, source_temperature=levels[1], name_suffix="_rep3", pressure_range=(1500.0, 3000.0),
        speed_range=(400.0, 1200.0),
        caption="Map domain 1500–3000 kPa x 400–1200 rpm. (a) White dashed contour: 75th percentile of spread.",
        uncertainty_quantiles=(0.75,),
    )
    mf.save_figure = orig_save
    sel_out = sel[FEATURE_COLUMNS + ["power", "efficiency", "agent_score", "sparsity_score", "uncertainty_score",
                                     "candidate_source"]]
    sel_out.to_csv(YANMAR_ROOT / "results" / "r1_main_batch_with_scores.csv", index=False)
    return metrics


# label offsets in points from the marker
SELECTOR_STYLE = {
    "Proposed (12)": ("*", 0.05, 190, (-9, -9, "right")),
    "(12), R0 weights": ("P", 0.30, 70, (-8, 9, "right")),
    "Coverage-only": ("o", 0.62, 55, (9, -9, "left")),
    "Maximin": ("s", 0.45, 55, (0, -11, "center")),
    "Random (pool)": ("^", 0.75, 60, (-8, -11, "right")),
    "Latin hypercube (box)": ("v", 0.85, 60, (6, 11, "left")),
    "D-optimal": ("D", 0.20, 50, (8, 0, "left")),
    "Uncertainty-only": ("X", 0.92, 60, (10, 8, "left")),
    "BALD-like": ("h", 0.97, 60, (10, -8, "left")),
}


def selector_figure() -> None:
    mf.setup_ieee_style()
    rows = pd.read_csv(PROPOSER_DIR / "r1_selector_rows.csv")
    g = rows.groupby("selector")
    mean, sd = g[["dense_power_mae", "coverage_gain_pct"]].mean(), g[["dense_power_mae", "coverage_gain_pct"]].std()
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for name, (mk, c, s, (dx, dy, ha)) in SELECTOR_STYLE.items():
        x, y = mean.loc[name, "coverage_gain_pct"], mean.loc[name, "dense_power_mae"]
        ax.errorbar(x, y, xerr=sd.loc[name, "coverage_gain_pct"], yerr=sd.loc[name, "dense_power_mae"], fmt="none",
                    ecolor="0.55", elinewidth=0.7, capsize=2, zorder=1)
        ax.scatter(x, y, marker=mk, s=s, color=mf.PLASMA(c), edgecolors="black", linewidths=0.5, zorder=3 if mk != "*" else 4)
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, ha=ha, va="center")
    ax.set_yscale("log")
    ax.set_yticks([10, 20, 50, 100, 200, 400])
    ax.set_yticklabels(["10", "20", "50", "100", "200", "400"])
    ax.set_xlabel("Coverage gain on the dense test set / %")
    ax.set_ylabel("Dense-test power MAE / W (log scale)")
    ax.set_xlim(5, 42)
    ax.set_ylim(10, 400)
    ax.grid(False)
    fig.tight_layout()
    mf.save_figure(fig, "fig_r1_selector_benchmark")


def dispatch_figure() -> dict:
    mf.setup_ieee_style()
    sp = pd.read_csv(DISPATCH_DIR / "hourly_sparse_base.csv")
    au = pd.read_csv(DISPATCH_DIR / "hourly_augmented_base.csv")
    orc = pd.read_csv(DISPATCH_DIR / "hourly_oracle_base.csv")
    t = au["hour"].to_numpy(float) / 24.0
    fig = plt.figure(figsize=(7.2, 7.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.85, 0.85], width_ratios=[2.2, 1.0], hspace=0.62, wspace=0.28)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
    series = [("Load", "load_kw", "0.55"), ("Stirling WHR (realised)", "whr_kw", "#2166AC"),
              ("PV", "pv_kw", mf.PLASMA(0.92)), ("BESS (+ charge)", "storage_power_kw", mf.PLASMA(0.62)),
              ("Grid import", "grid_import_kw", mf.PLASMA(0.25))]
    for lab, col, c in series:
        ax_a.plot(t, au[col], color=c, lw=0.35, alpha=0.85, rasterized=True, label=lab)
    z0, z1 = 300.0, 307.0
    ax_a.axvspan(z0, z1, color="0.2", alpha=0.12, lw=0)
    ax_a.set_xlim(0, 365)
    ax_a.set_xlabel("Time / day")
    ax_a.set_ylabel("Power / kW")
    ax_a.set_title("(a) Annual hourly balance, augmented policy", fontsize=9)
    zm = (t >= z0) & (t <= z1)
    for lab, col, c in series:
        ax_b.plot(t[zm], au.loc[zm, col], color=c, lw=0.9, label=lab)
    ax_b.set_xlim(z0, z1)
    ax_b.set_xlabel("Time / day")
    ax_b.set_title("(b) Days 300–307", fontsize=9)
    from matplotlib.lines import Line2D

    ax_a.legend(handles=[Line2D([0], [0], color=c, lw=1.6, label=lab) for lab, _, c in series],
                loc="upper center", bbox_to_anchor=(0.70, -0.27), ncol=5, fontsize=7.5, frameon=False, handlelength=1.6)
    for ax in (ax_a, ax_b):
        ax.grid(False)

    ax_c.plot(t[zm], orc.loc[zm, "whr_kw"], color="black", lw=1.6, label="Oracle policy (RK4 table)")
    ax_c.plot(t[zm], sp.loc[zm, "whr_kw"], color=mf.PLASMA(0.72), lw=1.1, ls="--", label="Sparse surrogate: realised")
    ax_c.plot(t[zm], au.loc[zm, "whr_kw"], color=mf.PLASMA(0.14), lw=1.1, label="Augmented surrogate: realised")
    ax_c.set_xlim(z0, z1)
    ax_c.set_ylim(-420, 1500)
    ax_c.set_yticks([0, 500, 1000, 1500])
    ax_c.set_ylabel("WHR power / kW")
    ax_c.set_title("(c) Realised WHR output under the RK4 oracle", fontsize=9)
    ax_c.legend(loc="lower center", ncol=3, fontsize=7, frameon=True, facecolor="white", edgecolor="0.7")
    ax_c.grid(False)

    e_s = (sp["whr_scheduled_kw"] - sp["whr_kw"]).to_numpy()
    e_a = (au["whr_scheduled_kw"] - au["whr_kw"]).to_numpy()
    ax_d.axhline(0, color="0.3", lw=0.6)
    ax_d.plot(t[zm], e_s[zm], color=mf.PLASMA(0.72), lw=1.1, ls="--", label="Sparse surrogate")
    ax_d.plot(t[zm], e_a[zm], color=mf.PLASMA(0.14), lw=1.1, label="Augmented surrogate")
    ax_d.set_xlim(z0, z1)
    ax_d.set_xlabel("Time / day")
    ax_d.set_ylabel("Committed − realised / kW")
    ax_d.set_title("(d) Commitment error (positive = shortfall)", fontsize=9)
    ax_d.legend(loc="upper right", ncol=2, fontsize=7, frameon=True, facecolor="white", edgecolor="0.7")
    ax_d.grid(False)
    mf.save_figure(fig, "fig_r1_dispatch_oracle")
    return {
        "zoom_mean_abs_commit_error_kw": {"sparse": float(np.abs(e_s[zm]).mean()), "augmented": float(np.abs(e_a[zm]).mean())},
        "annual_mean_whr_kw": {"sparse": float(sp["whr_kw"].mean()), "augmented": float(au["whr_kw"].mean()),
                               "oracle": float(orc["whr_kw"].mean())},
    }


def main() -> None:
    _setup()
    if "--quick" in sys.argv:
        selector_figure()
        dispatch_figure()
        return
    out = {"holdout18": surrogate_figures()}
    selector_figure()
    out["dispatch_fig"] = dispatch_figure()
    (YANMAR_ROOT / "results" / "r1_figures_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
