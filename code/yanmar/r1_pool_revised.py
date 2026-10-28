"""Candidate-pool comparison under the validated acquisition rule (cell-cap filter, lambda_s = 0.95).

Same pools, seeds and dense test set as the pre-registered E5b (r1_pool_benchmark.py), K = 28,
R0-prompt proposer runs only. Also writes the main-pipeline batch (run 0, seed 0).

    python -m yanmar.r1_pool_revised      (from journal_extension/)
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

from data_loader import FEATURE_COLUMNS  # noqa: E402
from kimi_proposer import tag_query_origin  # noqa: E402
from yanmar.active_sampling_benchmark import BUDGET, _fit  # noqa: E402
from yanmar.r1_acquisition_validation import make_selector  # noqa: E402
from yanmar.r1_pool_benchmark import CONTROLS, N_SEEDS, PROPOSER_DIR, _rk_label, _state, build_pools  # noqa: E402

LAMBDA_S = 0.95
MECHANISM = "cell-cap"
MAIN_BATCH = PROPOSER_DIR / "main_pipeline_batch.csv"


def _task(args):
    name, run_id, seed, pool = args
    st = _state()
    train, dense = st["train"], st["dense"]
    model = _fit(train, 21 + seed)
    sel = tag_query_origin(make_selector(MECHANISM, LAMBDA_S)(pool, train, model, BUDGET), st["lattice"])
    y = _rk_label(sel[FEATURE_COLUMNS].to_numpy(float))
    sel = sel.assign(power=y[:, 0], efficiency=y[:, 1])
    final = _fit(pd.concat([train, sel[FEATURE_COLUMNS + ["power", "efficiency"]]], ignore_index=True), 22 + seed)
    mu_d, sd_d = final.predict(dense[FEATURE_COLUMNS].to_numpy(float), return_std=True)
    mu_h = final.predict(st["test"][FEATURE_COLUMNS].to_numpy(float))
    err = np.abs(mu_d[:, 0] - dense["power"].to_numpy())
    row = {
        "pool": name, "run_id": run_id, "seed": seed, "pool_size": len(pool),
        "dense_power_mae": float(err.mean()), "dense_power_p95": float(np.percentile(err, 95)),
        "dense_eff_mae_pp": float(100 * np.abs(mu_d[:, 1] - dense["efficiency"].to_numpy()).mean()),
        "dense_cov95": float((err <= 1.96 * sd_d[:, 0]).mean()),
        "holdout_power_mae": float(np.abs(mu_h[:, 0] - st["y_test"][:, 0]).mean()),
        "holdout_eff_mae_pp": float(100 * np.abs(mu_h[:, 1] - st["y_test"][:, 1]).mean()),
    }
    return row, sel.assign(pool=name, run_id=run_id, seed=seed)


def main() -> None:
    from scipy.stats import wilcoxon

    st = _state()
    pools = build_pools(st["oracle"], [0, 1, 2, 3, 4])
    keep = {("Lattice", "-")} | {(n, "bounds") for n in ("LLM", *CONTROLS)}
    tasks = [(n, r, s, p) for (n, v, r), p in pools.items() if (n, v) in keep for s in range(N_SEEDS)]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("R1_WORKERS", "6"))) as ex:
        out = list(ex.map(_task, tasks, chunksize=4))
    rows = pd.DataFrame([r for r, _ in out])
    sels = pd.concat([s for _, s in out], ignore_index=True)
    rows.to_csv(PROPOSER_DIR / "r1_pool_revised_rows.csv", index=False)
    sels.to_csv(PROPOSER_DIR / "r1_pool_revised_selected.csv", index=False)
    sels[(sels["pool"] == "LLM") & (sels["run_id"] == 0) & (sels["seed"] == 0)].to_csv(MAIN_BATCH, index=False)

    m = ["dense_power_mae", "dense_power_p95", "dense_eff_mae_pp", "dense_cov95", "holdout_power_mae", "holdout_eff_mae_pp"]
    summ = rows.groupby("pool")[m].agg(["mean", "std"])
    between = rows.groupby(["pool", "run_id"])["dense_power_mae"].mean().groupby("pool").std()
    summ[("dense_power_mae", "sd_between_runs")] = between
    summ.to_csv(PROPOSER_DIR / "r1_pool_revised_summary.csv")
    tests = []
    llm = rows[rows["pool"] == "LLM"].set_index(["run_id", "seed"])
    for ctl in CONTROLS:
        o = rows[rows["pool"] == ctl].set_index(["run_id", "seed"]).loc[llm.index]
        for metric in ("dense_power_mae", "dense_power_p95", "holdout_power_mae"):
            d = llm[metric] - o[metric]
            per_run = d.groupby(level=0).mean()
            tests.append({"vs": ctl, "metric": metric, "mean_diff": float(d.mean()), "llm_wins": int((d < 0).sum()),
                          "runs_lower_mean": int((per_run < 0).sum()), "p_pooled": float(wilcoxon(d).pvalue)})
    lat = rows[rows["pool"] == "Lattice"].set_index("seed")["dense_power_mae"]
    d = llm["dense_power_mae"] - lat.loc[llm.index.get_level_values(1)].to_numpy()
    tests.append({"vs": "Lattice", "metric": "dense_power_mae", "mean_diff": float(d.mean()), "llm_wins": int((d < 0).sum()),
                  "runs_lower_mean": int((d.groupby(level=0).mean() < 0).sum()), "p_pooled": float(wilcoxon(d).pvalue)})
    tests = pd.DataFrame(tests)
    tests.to_csv(PROPOSER_DIR / "r1_pool_revised_tests.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.round(3).to_string())
        print(tests.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
