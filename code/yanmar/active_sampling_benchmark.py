"""Same-budget active-sampling benchmark for the Yanmar manuscript.

Fig. 10 uses n = 20 paired repeats on a shared seed list. Two noise sources
are separated. Selection: Random and LHS are stochastic; Maximin is
deterministic given the pool and labelled set; Coverage-only, Uncertainty-only,
and BALD-like are deterministic given the pool and the current ensemble;
Agentic-batch randomness is proposer sampling from that pool. Model: the
B = 90 bootstrap (random_state = 21 + seed for acquisition, 22 + seed for
the post-label fit) is retrained on every repeat for every method. The
candidate pool is built once and held fixed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import make_proposer  # noqa: E402
from data_loader import FEATURE_COLUMNS, TARGET_COLUMNS  # noqa: E402
from physics_constraints import residual_metrics  # noqa: E402
from surrogate_models import BootstrapSurrogate, _poly_features  # noqa: E402
from yanmar.agents_yanmar import RKSolverLabelAgent, yanmar_physics  # noqa: E402
from yanmar.dataset import YANMAR_ROOT, feature_target_arrays, load_operating_data, sparse_active_learning_split  # noqa: E402


RESULT_DIR = YANMAR_ROOT / "results"
FIG_DIR = YANMAR_ROOT / "figures"
BUDGET = 28
N_REPEATS = 20
REPEAT_SEEDS = tuple(range(N_REPEATS))
RANDOM_SEEDS = REPEAT_SEEDS


@dataclass(frozen=True)
class Selection:
    name: str
    cases: pd.DataFrame
    seed: int | None = None


def _fit(train: pd.DataFrame, random_state: int) -> BootstrapSurrogate:
    x, y = feature_target_arrays(train)
    return BootstrapSurrogate(
        n_estimators=90,
        degree=3,
        alpha=0.08,
        random_state=random_state,
        physics_agent=yanmar_physics(),
    ).fit(x, y)


def _normalizer(reference: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = reference[FEATURE_COLUMNS].to_numpy(float)
    mean = x.mean(axis=0)
    std = np.where(x.std(axis=0) == 0.0, 1.0, x.std(axis=0))
    return mean, std


def _scaled_distance(a: pd.DataFrame, b: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    xa = (a[FEATURE_COLUMNS].to_numpy(float) - mean) / std
    xb = (b[FEATURE_COLUMNS].to_numpy(float) - mean) / std
    return np.sqrt(((xa[:, None, :] - xb[None, :, :]) ** 2).sum(axis=2))


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    if span < 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / span


def _lhs(bounds: dict[str, tuple[float, float]], n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for col in FEATURE_COLUMNS:
        lo, hi = bounds[col]
        edges = np.linspace(0.0, 1.0, n + 1)
        u = rng.uniform(edges[:-1], edges[1:])
        rng.shuffle(u)
        data[col] = lo + u * (hi - lo)
    return pd.DataFrame(data)


def _batch_design_features(
    batch: pd.DataFrame,
    train: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    model: BootstrapSurrogate,
) -> dict[str, float]:
    """Scores a full 28-point batch without using holdout labels or locations."""
    mean, std = _normalizer(train)
    augmented_x = pd.concat([train[FEATURE_COLUMNS], batch[FEATURE_COLUMNS]], ignore_index=True)

    # Domain-filling score: how well train+batch covers the feasible candidate pool.
    pool_dist = _scaled_distance(candidate_pool, augmented_x, mean, std).min(axis=1)
    coverage = -float(pool_dist.mean())

    # Novelty score: keep the queried batch away from the already measured sparse grid.
    novelty = float(_scaled_distance(batch, train, mean, std).min(axis=1).mean())

    # Regression-design score in the actual polynomial basis used by the surrogate.
    x_aug = augmented_x.to_numpy(float)
    xs = (x_aug - mean) / std
    phi = _poly_features(xs, degree=3)
    sign, logdet = np.linalg.slogdet(phi.T @ phi + 0.08 * np.eye(phi.shape[1]))
    design = float(logdet if sign > 0 else -1e6)

    # Epistemic score from the sparse model, averaged over the proposed batch.
    _, pred_std = model.predict(batch[FEATURE_COLUMNS].to_numpy(float), return_std=True)
    uncertainty = float(np.mean(_normalize(pred_std[:, 0]) + _normalize(pred_std[:, 1])))

    # Batch diversity prevents the critic from choosing clustered high-uncertainty points.
    d_batch = _scaled_distance(batch, batch, mean, std)
    d_batch[d_batch == 0.0] = np.inf
    diversity = float(np.min(d_batch, axis=1).mean())

    return {
        "coverage": coverage,
        "novelty": novelty,
        "design": design,
        "uncertainty": uncertainty,
        "diversity": diversity,
    }


def _critic_select_batch(
    batches: list[pd.DataFrame],
    train: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    model: BootstrapSurrogate,
    weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows = [_batch_design_features(b, train, candidate_pool, model) for b in batches]
    features = pd.DataFrame(feature_rows)
    score = np.zeros(len(features))
    for col, weight in weights.items():
        score += weight * _normalize(features[col].to_numpy(float))
    features["critic_score"] = score
    idx = int(np.argmax(score))
    return batches[idx].reset_index(drop=True), features


def _agent_batch_critic(
    train: pd.DataFrame,
    oracle_pool: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    model: BootstrapSurrogate,
    variant: str,
    n: int = BUDGET,
) -> pd.DataFrame:
    bounds = {col: (float(oracle_pool[col].min()), float(oracle_pool[col].max())) for col in FEATURE_COLUMNS}
    batches: list[pd.DataFrame] = []
    # Multiple proposer agents: LHS, random batches, and greedy maximin from a shared pool.
    for seed in range(80):
        batches.append(_lhs(bounds, n, seed))
    rng = np.random.default_rng(2027)
    for _ in range(40):
        batches.append(candidate_pool.iloc[rng.choice(len(candidate_pool), n, replace=False)][FEATURE_COLUMNS].reset_index(drop=True))
    batches.append(_maximin(candidate_pool, train, n=n)[FEATURE_COLUMNS].reset_index(drop=True))
    batches.append(_d_optimal_batch(candidate_pool, train, n=n)[FEATURE_COLUMNS].reset_index(drop=True))

    if variant == "coverage":
        weights = {"coverage": 0.55, "diversity": 0.25, "design": 0.20}
    elif variant == "balanced":
        weights = {"coverage": 0.35, "design": 0.30, "diversity": 0.20, "uncertainty": 0.15}
    elif variant == "design":
        weights = {"design": 0.45, "coverage": 0.25, "diversity": 0.20, "uncertainty": 0.10}
    else:
        raise ValueError(f"unknown batch critic variant: {variant}")
    chosen, features = _critic_select_batch(batches, train, candidate_pool, model, weights)
    chosen = chosen.copy()
    chosen["batch_critic_score"] = float(features["critic_score"].max())
    return chosen


def _agent_lhs_batch_critic(
    train: pd.DataFrame,
    oracle_pool: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    model: BootstrapSurrogate,
    n: int = BUDGET,
    proposer_seed: int = 0,
) -> pd.DataFrame:
    """Agentic LHS portfolio: proposer emits LHS batches, critic selects one batch.

    The critic is based on three quantities that are available before querying:
    polynomial design information, sparse-model uncertainty, and within-batch
    diversity. Holdout labels and holdout locations are not used.
    """
    bounds = {col: (float(oracle_pool[col].min()), float(oracle_pool[col].max())) for col in FEATURE_COLUMNS}
    batches = [_lhs(bounds, n, proposer_seed * 40 + i) for i in range(40)]
    weights = {"design": 0.4, "uncertainty": 0.3, "diversity": 0.3}
    chosen, features = _critic_select_batch(batches, train, candidate_pool, model, weights)
    chosen = chosen.copy()
    chosen["batch_critic_score"] = float(features["critic_score"].max())
    chosen["batch_seed"] = int(features["critic_score"].idxmax())
    return chosen


def _diverse_top(scored: pd.DataFrame, train: pd.DataFrame, score_col: str, n: int = BUDGET) -> pd.DataFrame:
    temp_bins = np.linspace(train["source_temperature"].min(), train["source_temperature"].max(), 4)
    pressure_bins = np.linspace(train["mean_pressure"].min(), train["mean_pressure"].max(), 6)
    speed_bins = np.linspace(train["engine_speed"].min(), train["engine_speed"].max(), 7)
    ranked = scored.sort_values(score_col, ascending=False).copy()
    ranked["temp_bin"] = np.digitize(ranked["source_temperature"], temp_bins, right=True)
    ranked["pressure_bin"] = np.digitize(ranked["mean_pressure"], pressure_bins, right=True)
    ranked["speed_bin"] = np.digitize(ranked["engine_speed"], speed_bins, right=True)

    rows = []
    bin_counts: dict[tuple[int, int, int], int] = {}
    for _, row in ranked.iterrows():
        key = (int(row["temp_bin"]), int(row["pressure_bin"]), int(row["speed_bin"]))
        if bin_counts.get(key, 0) >= 2:
            continue
        rows.append(row)
        bin_counts[key] = bin_counts.get(key, 0) + 1
        if len(rows) == n:
            break
    if len(rows) < n:
        used = {int(r.name) for r in rows}
        for idx, row in ranked.iterrows():
            if int(idx) not in used:
                rows.append(row)
            if len(rows) == n:
                break
    return pd.DataFrame(rows).drop(columns=["temp_bin", "pressure_bin", "speed_bin"]).reset_index(drop=True)


def _maximin(candidate_pool: pd.DataFrame, train: pd.DataFrame, n: int = BUDGET) -> pd.DataFrame:
    mean, std = _normalizer(train)
    selected_rows = []
    remaining = candidate_pool.copy().reset_index(drop=True)
    current = train[FEATURE_COLUMNS].copy()
    for _ in range(n):
        d = _scaled_distance(remaining, current, mean, std)
        idx = int(np.argmax(d.min(axis=1)))
        selected_rows.append(remaining.iloc[idx])
        current = pd.concat([current, remaining.iloc[[idx]][FEATURE_COLUMNS]], ignore_index=True)
        remaining = remaining.drop(index=idx).reset_index(drop=True)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _d_optimal_batch(candidate_pool: pd.DataFrame, train: pd.DataFrame, n: int = BUDGET) -> pd.DataFrame:
    """Greedy D-optimal query design for the polynomial-ridge surrogate basis."""
    x_train = train[FEATURE_COLUMNS].to_numpy(float)
    mean = x_train.mean(axis=0)
    std = np.where(x_train.std(axis=0) == 0.0, 1.0, x_train.std(axis=0))

    train_phi = _poly_features((x_train - mean) / std, degree=3)
    can = candidate_pool.reset_index(drop=True)
    can_phi = _poly_features((can[FEATURE_COLUMNS].to_numpy(float) - mean) / std, degree=3)
    info_inv = np.linalg.pinv(train_phi.T @ train_phi + 0.08 * np.eye(train_phi.shape[1]))

    selected_idx: list[int] = []
    available = np.ones(len(can), dtype=bool)
    for _ in range(n):
        phi_avail = can_phi[available]
        leverage = np.einsum("ij,jk,ik->i", phi_avail, info_inv, phi_avail)
        local_idx = int(np.argmax(leverage))
        global_idx = int(np.flatnonzero(available)[local_idx])
        selected_idx.append(global_idx)

        phi = can_phi[global_idx : global_idx + 1]
        denom = 1.0 + float((phi @ info_inv @ phi.T).item())
        info_inv = info_inv - (info_inv @ phi.T @ phi @ info_inv) / denom
        available[global_idx] = False

    return can.iloc[selected_idx].reset_index(drop=True)


def _agent_batch_design(
    candidate_pool: pd.DataFrame,
    train: pd.DataFrame,
    model: BootstrapSurrogate,
    n: int = BUDGET,
) -> pd.DataFrame:
    """Revised multi-agent selector: design leverage + coverage + epistemic uncertainty."""
    scored = _score_pool(candidate_pool, train, model)

    x_train = train[FEATURE_COLUMNS].to_numpy(float)
    mean = x_train.mean(axis=0)
    std = np.where(x_train.std(axis=0) == 0.0, 1.0, x_train.std(axis=0))
    train_phi = _poly_features((x_train - mean) / std, degree=3)
    can_phi = _poly_features((scored[FEATURE_COLUMNS].to_numpy(float) - mean) / std, degree=3)
    info_inv = np.linalg.pinv(train_phi.T @ train_phi + 0.08 * np.eye(train_phi.shape[1]))
    leverage = np.einsum("ij,jk,ik->i", can_phi, info_inv, can_phi)
    scored["design_score"] = _normalize(leverage)
    scored["multi_agent_score"] = (
        0.45 * scored["design_score"]
        + 0.35 * scored["coverage_score"]
        + 0.20 * scored["uncertainty_score"]
    )

    selected_rows = []
    current = train[FEATURE_COLUMNS].copy()
    remaining = scored.copy().reset_index(drop=True)
    for _ in range(n):
        mean_d, std_d = _normalizer(current)
        novelty = _scaled_distance(remaining, current, mean_d, std_d).min(axis=1)
        combined = 0.80 * remaining["multi_agent_score"].to_numpy(float) + 0.20 * _normalize(novelty)
        idx = int(np.argmax(combined))
        selected_rows.append(remaining.iloc[idx])
        current = pd.concat([current, remaining.iloc[[idx]][FEATURE_COLUMNS]], ignore_index=True)
        remaining = remaining.drop(index=idx).reset_index(drop=True)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _score_pool(candidate_pool: pd.DataFrame, train: pd.DataFrame, model: BootstrapSurrogate) -> pd.DataFrame:
    scored = candidate_pool.copy()
    mean, std = _normalizer(train)
    d = _scaled_distance(scored, train, mean, std)
    scored["coverage_score"] = _normalize(d.min(axis=1))

    _, pred_std = model.predict(scored[FEATURE_COLUMNS].to_numpy(float), return_std=True)
    power_u = _normalize(pred_std[:, 0])
    eff_u = _normalize(pred_std[:, 1])
    scored["uncertainty_score"] = _normalize(power_u + eff_u)
    scored["bald_like_score"] = _normalize(np.log1p(power_u * eff_u + 1e-12))
    return scored


def _label(cases: pd.DataFrame, oracle_pool: pd.DataFrame) -> pd.DataFrame:
    return RKSolverLabelAgent(physics_agent=yanmar_physics()).label(cases[FEATURE_COLUMNS].copy(), oracle_pool)


def _evaluate(
    strategy: str,
    selected: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_test: np.ndarray,
    seed: int | None,
) -> dict[str, float | int | str | None]:
    labelled = selected if set(TARGET_COLUMNS).issubset(selected.columns) else _label(selected, train)
    model_train = pd.concat([train.assign(label_source="rk4_grid"), labelled], ignore_index=True)
    model = _fit(model_train, 22 + (0 if seed is None else seed))
    y_pred = model.predict(test[FEATURE_COLUMNS].to_numpy(float))
    metrics = residual_metrics(y_test, y_pred)

    mean, std = _normalizer(train)
    d_aug = _scaled_distance(test, model_train, mean, std).min(axis=1)
    d_base = _scaled_distance(test, train, mean, std).min(axis=1)
    metrics.update(
        {
            "strategy": strategy,
            "seed": seed,
            "budget": len(labelled),
            "holdout_coverage_distance": float(d_aug.mean()),
            "coverage_gain_pct": float((d_base.mean() - d_aug.mean()) / d_base.mean() * 100.0),
        }
    )
    return metrics


def _shared_pool(oracle_pool: pd.DataFrame) -> pd.DataFrame:
    """Build the Table III feasible pool once; Fig. 10 repeats do not rebuild it."""
    physics = yanmar_physics()
    proposed = make_proposer(audit_path=RESULT_DIR / "kimi_proposer_audit.json").propose(oracle_pool)
    return physics.validate_candidates(proposed).drop_duplicates(subset=FEATURE_COLUMNS).reset_index(drop=True)


def _make_selections(
    train: pd.DataFrame,
    oracle_pool: pd.DataFrame,
    baseline: BootstrapSurrogate,
    pool: pd.DataFrame,
    seed: int,
) -> list[Selection]:
    scored = _score_pool(pool, train, baseline)
    bounds = {col: (float(oracle_pool[col].min()), float(oracle_pool[col].max())) for col in FEATURE_COLUMNS}
    rng = np.random.default_rng(seed)

    selections: list[Selection] = [
        Selection(
            "Agentic LHS batch",
            _label(_agent_lhs_batch_critic(train, oracle_pool, pool, baseline, proposer_seed=seed), oracle_pool),
            seed,
        ),
        Selection("Coverage-only", _label(_diverse_top(scored, train, "coverage_score"), oracle_pool), seed),
        Selection("Uncertainty-only", _label(_diverse_top(scored, train, "uncertainty_score"), oracle_pool), seed),
        Selection("BALD-like", _label(_diverse_top(scored, train, "bald_like_score"), oracle_pool), seed),
        Selection("Maximin", _label(_maximin(pool, train), oracle_pool), seed),
        Selection("D-optimal", _label(_d_optimal_batch(pool, train), oracle_pool), seed),
        Selection(
            "Random",
            _label(pool.iloc[rng.choice(len(pool), BUDGET, replace=False)].reset_index(drop=True), oracle_pool),
            seed,
        ),
        Selection("Latin hypercube", _label(_lhs(bounds, BUDGET, seed), oracle_pool), seed),
    ]
    return selections


def _plot_metric_benchmark(results: pd.DataFrame) -> None:
    order = [
        "Sparse baseline",
        "Random",
        "Latin hypercube",
        "Maximin",
        "Coverage-only",
        "Uncertainty-only",
        "BALD-like",
        "Agentic LHS batch",
    ]
    colors = {
        "Sparse baseline": "#8c8c8c",
        "Random": "#9ecae1",
        "Latin hypercube": "#6baed6",
        "Maximin": "#74c476",
        "Coverage-only": "#fd8d3c",
        "Uncertainty-only": "#fdae6b",
        "BALD-like": "#bcbddc",
        "Agentic LHS batch": "#3f007d",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), dpi=300)
    metrics = [("power_mae", "Power MAE / W"), ("efficiency_mae_pct_point", "Efficiency MAE / percentage point")]
    for ax, (metric, ylabel) in zip(axes, metrics):
        data = [results.loc[results["strategy"].eq(s), metric].to_numpy(float) for s in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, s in zip(bp["boxes"], order):
            patch.set_facecolor(colors[s])
            patch.set_alpha(0.65)
            patch.set_edgecolor("#333333")
        for element in ("whiskers", "caps", "medians"):
            for artist in bp[element]:
                artist.set_color("#333333")
                artist.set_linewidth(1.0)
        for i, s in enumerate(order, start=1):
            y = data[i - 1]
            if len(y) == 1:
                marker = "*" if s == "Agentic LHS batch" else "D"
                size = 120 if s == "Agentic LHS batch" else 42
                ax.scatter(i, y[0], s=size, marker=marker, color=colors[s], edgecolor="black", zorder=4)
            else:
                rng = np.random.default_rng(100 + i)
                ax.scatter(i + rng.normal(0.0, 0.035, len(y)), y, s=9, color=colors[s], alpha=0.35, lw=0)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", lw=0.7)
    fig.suptitle("Same-budget active-sampling benchmark on 18 Yanmar holdout cases", weight="bold", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_sampling_benchmark_metrics.{ext}", bbox_inches="tight")
    plt.close(fig)


def _plot_selection_tradeoff(results: pd.DataFrame) -> None:
    summary = (
        results.groupby("strategy", as_index=False)
        .agg(
            power_mae=("power_mae", "mean"),
            power_mae_std=("power_mae", "std"),
            efficiency_mae_pct_point=("efficiency_mae_pct_point", "mean"),
            coverage_gain_pct=("coverage_gain_pct", "mean"),
        )
        .fillna(0.0)
    )
    order = [
        "Sparse baseline",
        "Random",
        "Latin hypercube",
        "Maximin",
        "Coverage-only",
        "Uncertainty-only",
        "BALD-like",
        "Agentic LHS batch",
    ]
    summary = summary[summary["strategy"].isin(order)].copy()
    colors = {
        "Sparse baseline": "#8c8c8c",
        "Random": "#9ecae1",
        "Latin hypercube": "#6baed6",
        "Maximin": "#74c476",
        "Coverage-only": "#fd8d3c",
        "Uncertainty-only": "#fdae6b",
        "BALD-like": "#bcbddc",
        "Agentic LHS batch": "#3f007d",
    }

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    fig, (ax_top, ax_hi, ax) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(5.9, 4.45),
        dpi=300,
        gridspec_kw={"height_ratios": [0.62, 0.72, 2.1], "hspace": 0.045},
    )
    label_offsets = {
        "Sparse baseline": (7, 0),
        "Agentic LHS batch": (-6, 8),
        "Latin hypercube": (6, -8),
        "Maximin": (-5, 6),
        "Random": (-6, 6),
        "Coverage-only": (6, 7),
        "BALD-like": (6, -1),
        "Uncertainty-only": (6, 0),
    }
    display_names = {
        "Agentic LHS batch": "Agentic KAG batch",
    }
    for _, row in summary.iterrows():
        if row["power_mae"] > 500.0:
            target_ax = ax_top
        elif row["power_mae"] > 100.0:
            target_ax = ax_hi
        else:
            target_ax = ax
        marker = "*" if row["strategy"] == "Agentic LHS batch" else "o"
        size = 175 if row["strategy"] == "Agentic LHS batch" else 62
        target_ax.scatter(
            row["coverage_gain_pct"],
            row["power_mae"],
            s=size,
            marker=marker,
            color=colors[row["strategy"]],
            edgecolor="black",
            linewidth=0.65,
            label=row["strategy"],
            zorder=4 if row["strategy"] in {"Agentic LHS batch", "Sparse baseline"} else 3,
        )
        dx, dy = label_offsets.get(row["strategy"], (5, 0))
        target_ax.annotate(
            display_names.get(row["strategy"], row["strategy"]),
            xy=(row["coverage_gain_pct"], row["power_mae"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.6,
            va="center",
            ha="right" if dx < 0 else "left",
        )

    ax.set_xlabel("Holdout-distance reduction / %", fontsize=9)
    fig.text(0.035, 0.52, "Mean power MAE / W", rotation=90, va="center", fontsize=9)
    ax_top.set_title("Equal-budget sampling trade-off", fontsize=9.5, weight="bold", pad=5)
    ax.set_xlim(-1.2, 40.9)
    ax.set_ylim(20.0, 92.0)
    ax_hi.set_ylim(165.0, 218.0)
    ax_top.set_ylim(880.0, 930.0)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([25, 40, 55, 70, 85])
    ax_hi.set_yticks([175, 200])
    ax_top.set_yticks([900])
    for a in (ax, ax_hi, ax_top):
        a.grid(False)
        a.tick_params(axis="both", labelsize=8.2, length=3.2)
    ax_top.spines["bottom"].set_visible(False)
    ax_hi.spines["bottom"].set_visible(False)
    ax_hi.spines["top"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_hi.tick_params(labelbottom=False, bottom=False)

    # Diagonal break marks on both sides of the closed broken-y frame.
    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, linewidth=0.9)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs = dict(transform=ax_hi.transAxes, color="black", clip_on=False, linewidth=0.9)
    ax_hi.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_hi.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    ax_hi.plot((-d, +d), (-d, +d), **kwargs)
    ax_hi.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax.transAxes)
    ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig.tight_layout(rect=(0.055, 0.0, 1.0, 1.0), pad=0.45)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_sampling_benchmark_tradeoff.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_operating_data()
    train, oracle_pool, test = sparse_active_learning_split(df)
    x_test, y_test = feature_target_arrays(test)

    baseline = _fit(train, 21)
    baseline_metrics = residual_metrics(y_test, baseline.predict(x_test))
    mean, std = _normalizer(train)
    baseline_metrics.update(
        {
            "strategy": "Sparse baseline",
            "seed": 21,
            "budget": 0,
            "holdout_coverage_distance": float(_scaled_distance(test, train, mean, std).min(axis=1).mean()),
            "coverage_gain_pct": 0.0,
        }
    )

    rows = [baseline_metrics]
    pool = _shared_pool(oracle_pool)
    for seed in REPEAT_SEEDS:
        ensemble = _fit(train, 21 + seed)
        for selection in _make_selections(train, oracle_pool, ensemble, pool, seed):
            rows.append(_evaluate(selection.name, selection.cases, train, test, y_test, selection.seed))

    results = pd.DataFrame(rows)
    results.to_csv(RESULT_DIR / "sampling_strategy_benchmark.csv", index=False)

    summary = (
        results.groupby("strategy", as_index=False)
        .agg(
            runs=("power_mae", "size"),
            budget=("budget", "median"),
            power_mae_mean=("power_mae", "mean"),
            power_mae_std=("power_mae", "std"),
            power_rmse_mean=("power_rmse", "mean"),
            efficiency_mae_pp_mean=("efficiency_mae_pct_point", "mean"),
            efficiency_rmse_pp_mean=("efficiency_rmse_pct_point", "mean"),
            coverage_gain_pct_mean=("coverage_gain_pct", "mean"),
        )
        .fillna(0.0)
    )
    summary.to_csv(RESULT_DIR / "sampling_strategy_benchmark_summary.csv", index=False)

    _plot_metric_benchmark(results)
    _plot_selection_tradeoff(results)

    print(json.dumps(summary.to_dict(orient="records"), indent=2))
    print("Saved:", RESULT_DIR / "sampling_strategy_benchmark.csv")
    print("Saved:", FIG_DIR / "fig_sampling_benchmark_metrics.png")
    print("Saved:", FIG_DIR / "fig_sampling_benchmark_tradeoff.png")


if __name__ == "__main__":
    main()
