"""R1 selector benchmark (Fig. 10) and acquisition ablations on the archived proposer pools.

Part A: all selectors on the same LLM pool, 5 proposer runs x 20 shared seeds. "Proposed (12)" uses
the weights and filter chosen on the validation set (r1_acquisition_validation.py).
Part B: lambda_s sweep (lambda_u = 1 - lambda_s) and R0 diversity filter on/off for (12), test set.

    python -m yanmar.r1_selector_benchmark      (from journal_extension/)
"""
from __future__ import annotations

import matplotlib  # noqa: F401  (must precede pandas on this install)

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import CandidateCriticAgent  # noqa: E402
from data_loader import FEATURE_COLUMNS  # noqa: E402
from yanmar.active_sampling_benchmark import (  # noqa: E402
    BUDGET, _d_optimal_batch, _diverse_top, _fit, _lhs, _maximin, _normalizer, _scaled_distance, _score_pool,
)
from yanmar.r1_acquisition_validation import make_selector  # noqa: E402
from yanmar.r1_pool_benchmark import PROPOSER_DIR, _gate, _rk_label, _state  # noqa: E402

N_SEEDS = 20
RUNS = (0, 1, 2, 3, 4)
LAMBDA_S = (0.0, 0.25, 0.45, 0.65, 0.85, 1.0)


class NoDiversityCritic(CandidateCriticAgent):
    def _diverse_top_cases(self, scored, reference, n_select):
        return scored.sort_values("agent_score", ascending=False).head(n_select).reset_index(drop=True)


def _pool(run_id: int) -> pd.DataFrame:
    return _gate(pd.read_csv(PROPOSER_DIR / f"deepseek_run{run_id:02d}_records.csv"))


def _metrics(selected: pd.DataFrame, seed: int) -> dict:
    st = _state()
    train, dense, test = st["train"], st["dense"], st["test"]
    y = _rk_label(selected[FEATURE_COLUMNS].to_numpy(float))
    lab = selected[FEATURE_COLUMNS].assign(power=y[:, 0], efficiency=y[:, 1])
    model_train = pd.concat([train, lab], ignore_index=True)
    final = _fit(model_train, 22 + seed)
    mu_d, sd_d = final.predict(dense[FEATURE_COLUMNS].to_numpy(float), return_std=True)
    mu_h = final.predict(test[FEATURE_COLUMNS].to_numpy(float))
    err = np.abs(mu_d[:, 0] - dense["power"].to_numpy())
    mean, std = _normalizer(train)
    d_aug = _scaled_distance(dense, model_train, mean, std).min(axis=1)
    d_base = _scaled_distance(dense, train, mean, std).min(axis=1)
    return {
        "dense_power_mae": float(err.mean()), "dense_power_p95": float(np.percentile(err, 95)),
        "dense_eff_mae_pp": float(100 * np.abs(mu_d[:, 1] - dense["efficiency"].to_numpy()).mean()),
        "dense_cov95": float((err <= 1.96 * sd_d[:, 0]).mean()),
        "holdout_power_mae": float(np.abs(mu_h[:, 0] - st["y_test"][:, 0]).mean()),
        "holdout_eff_mae_pp": float(100 * np.abs(mu_h[:, 1] - st["y_test"][:, 1]).mean()),
        "coverage_gain_pct": float((d_base.mean() - d_aug.mean()) / d_base.mean() * 100.0),
    }


def _task_a(args):
    run_id, seed = args
    st = _state()
    train, oracle = st["train"], st["oracle"]
    pool = _pool(run_id)
    model = _fit(train, 21 + seed)
    scored = _score_pool(pool, train, model)
    rng = np.random.default_rng(seed)
    bounds = {c: (float(oracle[c].min()), float(oracle[c].max())) for c in FEATURE_COLUMNS}
    sels = {
        "Proposed (12)": make_selector("cell-cap", 0.95)(pool, train, model, BUDGET),
        "(12), R0 weights": CandidateCriticAgent(0.55, 0.45).select(pool, train, model, n_select=BUDGET),
        "Coverage-only": _diverse_top(scored, train, "coverage_score"),
        "Uncertainty-only": _diverse_top(scored, train, "uncertainty_score"),
        "BALD-like": _diverse_top(scored, train, "bald_like_score"),
        "Maximin": _maximin(pool, train),
        "D-optimal": _d_optimal_batch(pool, train),
        "Random (pool)": pool.iloc[rng.choice(len(pool), BUDGET, replace=False)].reset_index(drop=True),
        "Latin hypercube (box)": _lhs(bounds, BUDGET, seed),
    }
    return [{"part": "A", "selector": k, "run_id": run_id, "seed": seed, **_metrics(v, seed)} for k, v in sels.items()]


def _task_b(args):
    run_id, seed = args
    st = _state()
    train = st["train"]
    pool = _pool(run_id)
    model = _fit(train, 21 + seed)
    out = []
    for ls in LAMBDA_S:
        for div in (True, False):
            critic = (CandidateCriticAgent if div else NoDiversityCritic)(uncertainty_weight=1.0 - ls, sparsity_weight=ls)
            sel = critic.select(pool, train, model, n_select=BUDGET)
            out.append({"part": "B", "lambda_s": ls, "diversity_filter": div, "run_id": run_id, "seed": seed,
                        **_metrics(sel, seed)})
    return out


def main() -> None:
    tasks = [(r, s) for r in RUNS for s in range(N_SEEDS)]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("R1_WORKERS", "6"))) as ex:
        a = [row for rows in ex.map(_task_a, tasks, chunksize=2) for row in rows]
        b = [row for rows in ex.map(_task_b, tasks, chunksize=2) for row in rows]
    a, b = pd.DataFrame(a), pd.DataFrame(b)
    a.to_csv(PROPOSER_DIR / "r1_selector_rows.csv", index=False)
    b.to_csv(PROPOSER_DIR / "r1_lambda_diversity_rows.csv", index=False)
    m = ["dense_power_mae", "dense_power_p95", "dense_eff_mae_pp", "dense_cov95", "holdout_power_mae",
         "holdout_eff_mae_pp", "coverage_gain_pct"]
    sa = a.groupby("selector")[m].agg(["mean", "std"])
    sb = b.groupby(["lambda_s", "diversity_filter"])[m].agg(["mean", "std"])
    sa.to_csv(PROPOSER_DIR / "r1_selector_summary.csv")
    sb.to_csv(PROPOSER_DIR / "r1_lambda_diversity_summary.csv")

    from scipy.stats import wilcoxon

    tests = []
    prop = a[a["selector"] == "Proposed (12)"].set_index(["run_id", "seed"])
    for k in a["selector"].unique():
        if k == "Proposed (12)":
            continue
        o = a[a["selector"] == k].set_index(["run_id", "seed"]).loc[prop.index]
        for metric in ("dense_power_mae", "holdout_power_mae"):
            d = prop[metric] - o[metric]
            tests.append({"vs": k, "metric": metric, "mean_diff": float(d.mean()), "proposed_wins": int((d < 0).sum()),
                          "n": int(len(d)), "p": float(wilcoxon(d).pvalue)})
    pd.DataFrame(tests).to_csv(PROPOSER_DIR / "r1_selector_tests.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(sa.round(2))
        print(sb.round(2))
        print(pd.DataFrame(tests).round(4))


if __name__ == "__main__":
    main()
