import matplotlib  # noqa: F401

import numpy as np
import pandas as pd

from data_loader import FEATURE_COLUMNS
from llm_proposer import space_filling_pool
from yanmar.r1_pool_benchmark import PROPOSER_DIR, _gate, _state

st = _state()
train, dense, oracle = st["train"], st["dense"], st["oracle"]
mu = train[FEATURE_COLUMNS].to_numpy(float).mean(0)
sd = train[FEATURE_COLUMNS].to_numpy(float).std(0)
z = lambda df: (df[FEATURE_COLUMNS].to_numpy(float) - mu) / sd
zt, zd = z(train), z(dense)


def nn(a, b):
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(2)).min(1)


pools = {}
for r in range(5):
    llm = _gate(pd.read_csv(PROPOSER_DIR / f"deepseek_run{r:02d}_records.csv"))
    pools[("LLM", r)] = llm
    for k, kind in enumerate(("Sobol", "Random", "LHS")):
        pools[(kind, r)] = _gate(space_filling_pool(oracle, len(llm), kind.lower(), seed=1000 * r + 100 * k + len("bounds")))
rows = []
for (name, r), p in pools.items():
    d = nn(z(p), zt)
    rows.append({"pool": name, "run": r, "n": len(p), "q90_dist_to_seed": np.quantile(d, 0.9),
                 "n_far_1.0": int((d > 1.0).sum()), "frac_far_1.0": float((d > 1.0).mean()), "max": d.max()})
print(pd.DataFrame(rows).groupby("pool").mean().round(3))

sel = pd.read_csv(PROPOSER_DIR / "r1_pool_revised_selected.csv")
print(sel.columns.tolist())
keys = [c for c in ("run_id", "seed") if c in sel.columns]
out = []
for key, g in sel.groupby(["pool"] + keys):
    lab = np.vstack([zt, z(g)])
    dd = nn(zd, lab)
    out.append({"pool": key[0], "s_sel_mean": nn(z(g), zt).mean(), "dense_nn_median": np.median(dd), "dense_nn_q95": np.quantile(dd, 0.95)})
print(pd.DataFrame(out).groupby("pool").mean().round(3))
print("sparse dense_nn median", np.median(nn(zd, zt)).round(3))
