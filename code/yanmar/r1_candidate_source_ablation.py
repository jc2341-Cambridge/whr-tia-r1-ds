"""R1 candidate-source ablation (Table IV) over proposer runs x shared seeds.

Acquisition (12) is fixed: CandidateCriticAgent, λs=0.45, λu=0.55, K=28.
For shared seed s the acquisition ensemble uses random_state 21+s and the
evaluation ensemble 22+s. Pools:

    Lattice-only            1989-node retrieval lattice
    Lattice-jitter          lattice, N(0, JITTER_SD) added to the score before the greedy step
    LLM-only (run r)        schema-valid DeepSeek records of run r
    Lattice + LLM (run r)   merged pool after the bounds gate
    Sobol-only (run r)      deterministic heuristic proposer, count matched to run r unique records
    Lattice + Sobol (run r) merged pool after the bounds gate

    python -m yanmar.r1_candidate_source_ablation          (from journal_extension/)

Writes writing/Yanmar/results/r1_proposer/table4_{rows,summary,tests}.csv
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
from llm_proposer import heuristic_proposer  # noqa: E402
from yanmar.agents_yanmar import RKSolverLabelAgent, yanmar_physics  # noqa: E402
from yanmar.dataset import YANMAR_ROOT, feature_target_arrays, load_operating_data, sparse_active_learning_split  # noqa: E402
from yanmar.active_sampling_benchmark import BUDGET, _evaluate, _fit  # noqa: E402

PROPOSER_DIR = YANMAR_ROOT / "results" / "r1_proposer"
PROVIDER = "deepseek"
N_SEEDS = 20
JITTER_SD = 0.05


class JitterCritic(CandidateCriticAgent):
    def __init__(self, seed: int, sd: float = JITTER_SD) -> None:
        super().__init__(uncertainty_weight=0.55, sparsity_weight=0.45)
        self.rng = np.random.default_rng(10_000 + seed)
        self.sd = sd

    def _diverse_top_cases(self, scored, reference, n_select):
        scored = scored.copy()
        scored["agent_score"] = scored["agent_score"] + self.rng.normal(0.0, self.sd, len(scored))
        return super()._diverse_top_cases(scored, reference, n_select)


def _gate(pool: pd.DataFrame) -> pd.DataFrame:
    valid = yanmar_physics().validate_candidates(pool[FEATURE_COLUMNS].copy())
    return valid.drop_duplicates(subset=FEATURE_COLUMNS).reset_index(drop=True)


def _pools(oracle: pd.DataFrame, run_ids: list[int]) -> dict[tuple[str, int | None], pd.DataFrame]:
    lattice = lattice_pool(oracle)
    pools: dict[tuple[str, int | None], pd.DataFrame] = {
        ("Lattice-only", None): _gate(lattice),
        ("Lattice-jitter", None): _gate(lattice),
    }
    for r in run_ids:
        llm = pd.read_csv(PROPOSER_DIR / f"{PROVIDER}_run{r:02d}_records.csv")[FEATURE_COLUMNS]
        llm_gated = _gate(llm)
        sobol = heuristic_proposer(oracle, n_records=len(llm_gated), seed=r)[FEATURE_COLUMNS]
        pools[("LLM-only", r)] = llm_gated
        pools[("Lattice + LLM", r)] = _gate(pd.concat([lattice, llm], ignore_index=True))
        pools[("Sobol-only", r)] = _gate(sobol)
        pools[("Lattice + Sobol", r)] = _gate(pd.concat([lattice, sobol], ignore_index=True))
    return pools


def _task(args):
    name, run_id, seed, pool = args
    data = load_operating_data()
    train, oracle, test = sparse_active_learning_split(data)
    y_test = feature_target_arrays(test)[1]
    lattice = lattice_pool(oracle)
    model = _fit(train, 21 + seed)
    critic = JitterCritic(seed) if name == "Lattice-jitter" else CandidateCriticAgent(0.55, 0.45)
    selected = critic.select(pool, train, model, n_select=BUDGET)
    selected = tag_query_origin(selected, lattice)
    n_off = int((selected["candidate_source"] == "off_lattice_proposal").sum())
    labelled = RKSolverLabelAgent(physics_agent=yanmar_physics()).label(selected, oracle)
    m = _evaluate(name, labelled, train, test, y_test, seed=seed)
    m.update({"pool": name, "run_id": run_id, "pool_size": int(len(pool)), "n_off_lattice": n_off,
              "n_lattice": BUDGET - n_off})
    sel = selected[FEATURE_COLUMNS].assign(pool=name, run_id=run_id, seed=seed,
                                           candidate_source=selected["candidate_source"].to_numpy())
    return m, sel


def main() -> None:
    data = load_operating_data()
    _train, oracle, _test = sparse_active_learning_split(data)
    run_ids = sorted(
        int(p.stem.split("_run")[1].split("_")[0]) for p in PROPOSER_DIR.glob(f"{PROVIDER}_run*_records.csv")
    )
    pools = _pools(oracle, run_ids)
    tasks = [(name, r, s, pool) for (name, r), pool in pools.items() for s in range(N_SEEDS)]
    print(f"{len(tasks)} tasks, runs {run_ids}")

    with ProcessPoolExecutor(max_workers=int(os.environ.get("R1_WORKERS", "4"))) as ex:
        out = list(ex.map(_task, tasks, chunksize=2))
    rows = pd.DataFrame([m for m, _ in out])
    selections = pd.concat([s for _, s in out], ignore_index=True)

    PROPOSER_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(PROPOSER_DIR / "table4_rows.csv", index=False)
    selections.to_csv(PROPOSER_DIR / "table4_selected_batches.csv", index=False)

    metrics = ["power_mae", "efficiency_mae_pct_point"]
    summ = []
    for name, g in rows.groupby("pool", sort=False):
        rec = {"pool": name, "n_runs": g["run_id"].nunique(dropna=False), "n_eval": len(g),
               "pool_size_mean": g["pool_size"].mean(), "off_lattice_mean": g["n_off_lattice"].mean()}
        for k in metrics:
            rec[f"{k}_mean"] = g[k].mean()
            rec[f"{k}_sd_all"] = g[k].std(ddof=1)
            if g["run_id"].notna().any():
                run_means = g.groupby("run_id")[k].mean()
                rec[f"{k}_sd_between_runs"] = run_means.std(ddof=1)
                rec[f"{k}_sd_within_run"] = g.groupby("run_id")[k].std(ddof=1).mean()
            else:
                rec[f"{k}_sd_within_run"] = g[k].std(ddof=1)
        summ.append(rec)
    summary = pd.DataFrame(summ)
    summary.to_csv(PROPOSER_DIR / "table4_summary.csv", index=False)

    from scipy.stats import wilcoxon

    lat = rows[rows["pool"] == "Lattice-only"].set_index("seed")["power_mae"]
    jit = rows[rows["pool"] == "Lattice-jitter"].set_index("seed")["power_mae"]
    tests = [{"a": "Lattice-only", "b": "Lattice-jitter", "run_id": None, "median_diff_W": float((jit - lat).median()),
              "p": float(wilcoxon(lat, jit).pvalue)}]
    for r in run_ids:
        for a, b in (("Lattice + LLM", "Lattice-only"), ("Lattice + LLM", "Lattice + Sobol"), ("LLM-only", "Sobol-only")):
            xa = rows[(rows["pool"] == a) & (rows["run_id"] == r)].set_index("seed")["power_mae"]
            xb = (rows[(rows["pool"] == b) & (rows["run_id"] == r)] if b != "Lattice-only"
                  else rows[rows["pool"] == b]).set_index("seed")["power_mae"]
            tests.append({"a": a, "b": b, "run_id": r, "median_diff_W": float((xa - xb).median()),
                          "p": float(wilcoxon(xa, xb.loc[xa.index]).pvalue)})
    pd.DataFrame(tests).to_csv(PROPOSER_DIR / "table4_tests.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[["pool", "pool_size_mean", "off_lattice_mean", "power_mae_mean", "power_mae_sd_within_run",
                       "power_mae_sd_between_runs", "efficiency_mae_pct_point_mean"]].round(2))
        print(pd.DataFrame(tests).round(4))


if __name__ == "__main__":
    main()
