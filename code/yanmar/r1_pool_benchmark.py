"""R1 E5b benchmark: proposer variants versus density-matched pools (pre-registered).

See WHR-IEEE-TIA-R0/08_r1_experiments/E5b_preregistration.md for the protocol.

    python -m yanmar.r1_pool_benchmark          (from journal_extension/)

Writes writing/Yanmar/results/r1_proposer/e5b_{rows,summary,tests,selected}.csv
"""
from __future__ import annotations

# On this Windows install, importing pandas before matplotlib segfaults in
# matplotlib._c_internal_utils; matplotlib must be loaded first.
import matplotlib  # noqa: F401

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
from kimi_proposer import lattice_pool, tag_query_origin  # noqa: E402
from llm_proposer import space_filling_pool  # noqa: E402
from yanmar.agents_yanmar import yanmar_physics  # noqa: E402
from yanmar.dataset import YANMAR_ROOT, feature_target_arrays, load_operating_data, sparse_active_learning_split  # noqa: E402
from yanmar.active_sampling_benchmark import _fit  # noqa: E402
from yanmar.stirling_rk_solver import rk_power_efficiency  # noqa: E402
from yanmar.r1_candidate_source_ablation import JitterCritic  # noqa: E402

PROPOSER_DIR = YANMAR_ROOT / "results" / "r1_proposer"
DENSE_PATH = PROPOSER_DIR / "dense_test_sobol512.csv"
VARIANTS = {"bounds": "deepseek", "kag": "deepseek-kag"}
CONTROLS = ("Sobol", "Random", "LHS")
N_SEEDS = 20
BUDGETS = (7, 14, 21, 28)
DENSE_N, DENSE_SEED = 512, 2027

_STATE: dict = {}
_LABEL_CACHE: dict[tuple[float, float, float], tuple[float, float]] = {}


def _rk_label(x: np.ndarray) -> np.ndarray:
    phys = yanmar_physics()
    out = []
    for t, p, n in x:
        key = (round(float(t), 6), round(float(p), 6), round(float(n), 6))
        if key not in _LABEL_CACHE:
            pw, eff_pct = rk_power_efficiency(float(t) - 273.15, float(p), float(n))
            _LABEL_CACHE[key] = (pw, eff_pct / 100.0)
        out.append(_LABEL_CACHE[key])
    return phys.clip_predictions(np.asarray(out, dtype=float))


def dense_test_set(oracle: pd.DataFrame) -> pd.DataFrame:
    if DENSE_PATH.exists():
        return pd.read_csv(DENSE_PATH)
    pts = space_filling_pool(oracle, DENSE_N, "sobol", seed=DENSE_SEED)[FEATURE_COLUMNS]
    y = _rk_label(pts.to_numpy(float))
    out = pts.assign(power=y[:, 0], efficiency=y[:, 1], label_source="rk4_oracle")
    PROPOSER_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(DENSE_PATH, index=False)
    return out


def _state():
    if not _STATE:
        data = load_operating_data()
        train, oracle, test = sparse_active_learning_split(data)
        _STATE.update(
            train=train, oracle=oracle, test=test, y_test=feature_target_arrays(test)[1],
            dense=pd.read_csv(DENSE_PATH), lattice=lattice_pool(oracle),
        )
    return _STATE


def _gate(pool: pd.DataFrame) -> pd.DataFrame:
    valid = yanmar_physics().validate_candidates(pool[FEATURE_COLUMNS].copy())
    return valid.drop_duplicates(subset=FEATURE_COLUMNS).reset_index(drop=True)


def build_pools(oracle: pd.DataFrame, run_ids: list[int]) -> dict[tuple[str, str, int | None], pd.DataFrame]:
    lattice = lattice_pool(oracle)
    pools = {("Lattice", "-", None): _gate(lattice), ("Lattice-jitter", "-", None): _gate(lattice)}
    for variant, stem in VARIANTS.items():
        for r in run_ids:
            llm = _gate(pd.read_csv(PROPOSER_DIR / f"{stem}_run{r:02d}_records.csv"))
            pools[("LLM", variant, r)] = llm
            pools[("L+LLM", variant, r)] = _gate(pd.concat([lattice, llm], ignore_index=True))
            for k, kind in enumerate(CONTROLS):
                ctl = _gate(space_filling_pool(oracle, len(llm), kind.lower(), seed=1000 * r + 100 * k + len(variant)))
                pools[(kind, variant, r)] = ctl
                pools[(f"L+{kind}", variant, r)] = _gate(pd.concat([lattice, ctl], ignore_index=True))
    return pools


def _task(args):
    pool_name, variant, run_id, seed, budget, pool = args
    st = _state()
    train, dense = st["train"], st["dense"]
    model = _fit(train, 21 + seed)
    critic = JitterCritic(seed) if pool_name == "Lattice-jitter" else CandidateCriticAgent(0.55, 0.45)
    selected = tag_query_origin(critic.select(pool, train, model, n_select=budget), st["lattice"])
    y_sel = _rk_label(selected[FEATURE_COLUMNS].to_numpy(float))
    labelled = selected[FEATURE_COLUMNS].assign(power=y_sel[:, 0], efficiency=y_sel[:, 1])
    model_train = pd.concat([train, labelled], ignore_index=True)
    final = _fit(model_train, 22 + seed)

    mu_h = final.predict(st["test"][FEATURE_COLUMNS].to_numpy(float))
    mu_d, sd_d = final.predict(dense[FEATURE_COLUMNS].to_numpy(float), return_std=True)
    err = np.abs(mu_d[:, 0] - dense["power"].to_numpy())
    from scipy.stats import spearmanr

    row = {
        "pool": pool_name, "variant": variant, "run_id": run_id, "seed": seed, "budget": budget,
        "pool_size": len(pool),
        "n_off_lattice": int((selected["candidate_source"] == "off_lattice_proposal").sum()),
        "dense_power_mae": float(err.mean()),
        "dense_power_p95": float(np.percentile(err, 95)),
        "dense_power_max": float(err.max()),
        "dense_eff_mae_pp": float(100 * np.abs(mu_d[:, 1] - dense["efficiency"].to_numpy()).mean()),
        "dense_cov95": float((err <= 1.96 * sd_d[:, 0]).mean()),
        "dense_spearman_sigma_err": float(spearmanr(sd_d[:, 0], err).statistic),
        "holdout_power_mae": float(np.abs(mu_h[:, 0] - st["y_test"][:, 0]).mean()),
        "holdout_eff_mae_pp": float(100 * np.abs(mu_h[:, 1] - st["y_test"][:, 1]).mean()),
    }
    sel = selected[FEATURE_COLUMNS + ["agent_score", "sparsity_score", "uncertainty_score", "candidate_source"]].assign(
        pool=pool_name, variant=variant, run_id=run_id, seed=seed, budget=budget
    )
    return row, sel


def main() -> None:
    data = load_operating_data()
    _train, oracle, _test = sparse_active_learning_split(data)
    dense_test_set(oracle)
    run_ids = sorted(int(p.stem.split("_run")[1][:2]) for p in PROPOSER_DIR.glob("deepseek-kag_run*_records.csv"))
    pools = build_pools(oracle, run_ids)
    standalone = {"Lattice", "Lattice-jitter", "LLM", *CONTROLS}
    tasks = [
        (name, v, r, s, k, pool)
        for (name, v, r), pool in pools.items()
        for k in (BUDGETS if name in standalone else (28,))
        for s in range(N_SEEDS)
    ]
    print(f"{len(pools)} pools, {len(tasks)} tasks, runs {run_ids}", flush=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("R1_WORKERS", "6"))) as ex:
        out = list(ex.map(_task, tasks, chunksize=4))
    rows = pd.DataFrame([r for r, _ in out])
    rows.to_csv(PROPOSER_DIR / "e5b_rows.csv", index=False)
    pd.concat([s for _, s in out], ignore_index=True).to_csv(PROPOSER_DIR / "e5b_selected.csv", index=False)
    summarize(rows)


def summarize(rows: pd.DataFrame) -> None:
    from scipy.stats import wilcoxon

    metrics = ["dense_power_mae", "dense_power_p95", "dense_power_max", "dense_eff_mae_pp", "dense_cov95",
               "dense_spearman_sigma_err", "holdout_power_mae", "holdout_eff_mae_pp"]
    summ = []
    for (name, v, k), g in rows.groupby(["pool", "variant", "budget"], sort=False):
        rec = {"pool": name, "variant": v, "budget": k, "n_eval": len(g), "pool_size": g["pool_size"].mean(),
               "off_lattice": g["n_off_lattice"].mean()}
        for m in metrics:
            rec[f"{m}_mean"] = g[m].mean()
            rec[f"{m}_sd"] = g[m].std(ddof=1)
            if g["run_id"].notna().any():
                rec[f"{m}_sd_between_runs"] = g.groupby("run_id")[m].mean().std(ddof=1)
        summ.append(rec)
    summary = pd.DataFrame(summ)
    summary.to_csv(PROPOSER_DIR / "e5b_summary.csv", index=False)

    tests = []
    k28 = rows[rows["budget"] == 28]
    for v in VARIANTS:
        for r in sorted(k28.loc[k28["variant"] == v, "run_id"].dropna().unique()):
            def series(name, variant=v, run=r):
                g = k28[(k28["pool"] == name) & (k28["variant"] == variant) & (k28["run_id"] == run)]
                return g.set_index("seed")
            for a, b in [("LLM", c) for c in CONTROLS] + [("L+LLM", f"L+{c}") for c in CONTROLS]:
                for m in ("dense_power_mae", "holdout_power_mae", "dense_power_p95"):
                    xa, xb = series(a)[m], series(b)[m]
                    d = xa - xb.loc[xa.index]
                    p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
                    tests.append({"variant": v, "run_id": int(r), "a": a, "b": b, "metric": m,
                                  "mean_diff": float(d.mean()), "a_wins": int((d < 0).sum()), "p": p})
    for r in sorted(k28.loc[k28["variant"] == "kag", "run_id"].dropna().unique()):
        for m in ("dense_power_mae", "holdout_power_mae"):
            ga = k28[(k28["pool"] == "LLM") & (k28["variant"] == "kag") & (k28["run_id"] == r)].set_index("seed")[m]
            gb = k28[(k28["pool"] == "LLM") & (k28["variant"] == "bounds") & (k28["run_id"] == r)].set_index("seed")[m]
            d = ga - gb.loc[ga.index]
            tests.append({"variant": "kag-vs-bounds", "run_id": int(r), "a": "LLM-kag", "b": "LLM-bounds", "metric": m,
                          "mean_diff": float(d.mean()), "a_wins": int((d < 0).sum()), "p": float(wilcoxon(d).pvalue)})
    tests = pd.DataFrame(tests)
    tests.to_csv(PROPOSER_DIR / "e5b_tests.csv", index=False)

    cols = ["pool", "variant", "budget", "pool_size", "off_lattice", "dense_power_mae_mean", "dense_power_mae_sd",
            "dense_power_p95_mean", "dense_eff_mae_pp_mean", "dense_cov95_mean", "holdout_power_mae_mean"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(summary.loc[summary["budget"] == 28, cols].round(2).to_string(index=False))
        print(summary.loc[summary["pool"].isin(["Lattice", "LLM", *CONTROLS]), ["pool", "variant", "budget",
              "dense_power_mae_mean", "holdout_power_mae_mean"]].round(1).to_string(index=False))
        prim = tests[tests["metric"] == "dense_power_mae"]
        print(prim.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
