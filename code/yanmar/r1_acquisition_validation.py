"""R1 acquisition-rule selection on an independent validation set (R3-2, R3-3, AE-8).

Validation set: 256-point scrambled Sobol (seed 2028) over the oracle box, RK4-labelled.
The 512-point test set (seed 2027) is not used here. Two diversity mechanisms for (12):

  bin-cap     R0 filter: rank by a(x), at most 2 picks per (T_s, p, omega) bin, 5 per speed bin,
              7 per pressure bin.
  cell-cap    rank by a(x), at most 2 picks per (T_s, p, omega) bin; no marginal caps.
  sequential  s(x) is recomputed against D u S after every pick, so a(x) is re-ranked K times.

    python -m yanmar.r1_acquisition_validation      (from journal_extension/)
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
from llm_proposer import space_filling_pool  # noqa: E402
from yanmar.active_sampling_benchmark import BUDGET, _fit  # noqa: E402
from yanmar.r1_pool_benchmark import PROPOSER_DIR, _gate, _rk_label, _state  # noqa: E402

VAL_PATH = PROPOSER_DIR / "validation_sobol256.csv"
VAL_N, VAL_SEED = 256, 2028
N_SEEDS = 20
RUNS = (0, 1, 2, 3, 4)
LAMBDA_S = (0.0, 0.25, 0.45, 0.55, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
MECHANISMS = ("bin-cap", "cell-cap", "sequential")


class CellCapCritic(CandidateCriticAgent):
    def _diverse_top_cases(self, scored, reference, n_select):
        from yanmar.active_sampling_benchmark import _diverse_top

        return _diverse_top(scored, reference, "agent_score", n=n_select)


def make_selector(mechanism: str, lambda_s: float):
    if mechanism == "sequential":
        return lambda pool, train, model, k: sequential_select(pool, train, model, k, lambda_s, 1.0 - lambda_s)
    cls = CellCapCritic if mechanism == "cell-cap" else CandidateCriticAgent
    critic = cls(uncertainty_weight=1.0 - lambda_s, sparsity_weight=lambda_s)
    return lambda pool, train, model, k: critic.select(pool, train, model, n_select=k)


def _minmax(v: np.ndarray) -> np.ndarray:
    span = v.max() - v.min()
    return np.zeros_like(v) if span <= 0 else (v - v.min()) / span


def sequential_select(pool: pd.DataFrame, reference: pd.DataFrame, model, n_select: int,
                      lambda_s: float, lambda_u: float) -> pd.DataFrame:
    """Greedy batch form of (12): s(x) is updated against D u S after each pick."""
    x_ref = reference[FEATURE_COLUMNS].to_numpy(float)
    mean = x_ref.mean(axis=0)
    std = np.where(x_ref.std(axis=0) == 0, 1.0, x_ref.std(axis=0))
    xr = (x_ref - mean) / std
    xc = (pool[FEATURE_COLUMNS].to_numpy(float) - mean) / std
    d = np.sqrt(((xc[:, None, :] - xr[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    _, sd = model.predict(pool[FEATURE_COLUMNS].to_numpy(float), return_std=True)
    u = _minmax(_minmax(sd[:, 0]) + _minmax(sd[:, 1]))
    taken = np.zeros(len(pool), dtype=bool)
    picks, scores, s_at_pick = [], [], []
    for _ in range(min(n_select, len(pool))):
        s = _minmax(np.where(taken, 0.0, d))
        a = np.where(taken, -np.inf, lambda_s * s + lambda_u * u)
        i = int(np.argmax(a))
        picks.append(i)
        scores.append(float(a[i]))
        s_at_pick.append(float(s[i]))
        taken[i] = True
        d = np.minimum(d, np.sqrt(((xc - xc[i]) ** 2).sum(axis=1)))
    out = pool.iloc[picks].copy().reset_index(drop=True)
    out["agent_score"] = scores
    out["sparsity_score"] = s_at_pick
    out["uncertainty_score"] = u[picks]
    return out


def validation_set() -> pd.DataFrame:
    if VAL_PATH.exists():
        return pd.read_csv(VAL_PATH)
    oracle = _state()["oracle"]
    pts = space_filling_pool(oracle, VAL_N, "sobol", seed=VAL_SEED)[FEATURE_COLUMNS]
    y = _rk_label(pts.to_numpy(float))
    out = pts.assign(power=y[:, 0], efficiency=y[:, 1], label_source="rk4_oracle")
    out.to_csv(VAL_PATH, index=False)
    return out


def _task(args):
    run_id, seed = args
    st = _state()
    train = st["train"]
    val = pd.read_csv(VAL_PATH)
    pool = _gate(pd.read_csv(PROPOSER_DIR / f"deepseek_run{run_id:02d}_records.csv"))
    model = _fit(train, 21 + seed)
    rows = []
    for ls in LAMBDA_S:
        for mech in MECHANISMS:
            sel = make_selector(mech, ls)(pool, train, model, BUDGET)
            y = _rk_label(sel[FEATURE_COLUMNS].to_numpy(float))
            lab = sel[FEATURE_COLUMNS].assign(power=y[:, 0], efficiency=y[:, 1])
            final = _fit(pd.concat([train, lab], ignore_index=True), 22 + seed)
            mu = final.predict(val[FEATURE_COLUMNS].to_numpy(float))
            rows.append({
                "lambda_s": ls, "mechanism": mech, "run_id": run_id, "seed": seed,
                "val_power_mae": float(np.abs(mu[:, 0] - val["power"]).mean()),
                "val_eff_mae_pp": float(100 * np.abs(mu[:, 1] - val["efficiency"]).mean()),
            })
    return rows


def main() -> None:
    validation_set()
    tasks = [(r, s) for r in RUNS for s in range(N_SEEDS)]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("R1_WORKERS", "6"))) as ex:
        rows = pd.DataFrame([x for out in ex.map(_task, tasks, chunksize=2) for x in out])
    rows.to_csv(PROPOSER_DIR / "r1_acq_validation_rows.csv", index=False)
    summ = rows.groupby(["mechanism", "lambda_s"])[["val_power_mae", "val_eff_mae_pp"]].agg(["mean", "std"])
    summ.to_csv(PROPOSER_DIR / "r1_acq_validation_summary.csv")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
