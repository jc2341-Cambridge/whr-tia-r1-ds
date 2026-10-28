import matplotlib  # noqa: F401

import numpy as np
import pandas as pd

from data_loader import FEATURE_COLUMNS
from yanmar.r1_pool_benchmark import PROPOSER_DIR, _gate, _state
from yanmar.r1_pool_revised import MAIN_BATCH
from llm_proposer import space_filling_pool

st = _state()
oracle, lattice = st["oracle"], st["lattice"]
lo = oracle[FEATURE_COLUMNS].min().to_numpy(float)
hi = oracle[FEATURE_COLUMNS].max().to_numpy(float)


def unit(df):
    return (df[FEATURE_COLUMNS].to_numpy(float) - lo) / (hi - lo)


def stats(df):
    u = unit(df)
    near_face = ((u <= 0.05) | (u >= 0.95))
    return {"n": len(df), "face_frac": float(near_face.any(axis=1).mean()),
            "corner_frac": float((near_face.sum(axis=1) == 3).mean()),
            "edge_frac": float((near_face.sum(axis=1) >= 2).mean())}


lat_keys = set(map(tuple, np.round(lattice[FEATURE_COLUMNS].to_numpy(float), 3)))
rows = []
for r in range(5):
    llm = _gate(pd.read_csv(PROPOSER_DIR / f"deepseek_run{r:02d}_records.csv"))
    on_lat = sum(tuple(x) in lat_keys for x in np.round(llm[FEATURE_COLUMNS].to_numpy(float), 3))
    rows.append({"pool": "LLM", "run": r, **stats(llm), "on_lattice": on_lat})
    sob = _gate(space_filling_pool(oracle, len(llm), "sobol", seed=1000 * r + 0 + len("bounds")))
    rows.append({"pool": "Sobol", "run": r, **stats(sob), "on_lattice": 0})
rows.append({"pool": "Lattice", "run": -1, **stats(lattice), "on_lattice": len(lattice)})
sel = pd.read_csv(MAIN_BATCH)
on_lat = sum(tuple(x) in lat_keys for x in np.round(sel[FEATURE_COLUMNS].to_numpy(float), 3))
rows.append({"pool": "main_batch", "run": 0, **stats(sel), "on_lattice": on_lat})
sel_all = pd.read_csv(PROPOSER_DIR / "r1_pool_revised_selected.csv")
for pool, g in sel_all.groupby("pool"):
    rows.append({"pool": f"selected:{pool}", "run": -1, **stats(g), "on_lattice": -1})
dense = st["dense"]
rows.append({"pool": "dense512", "run": -1, **stats(dense), "on_lattice": -1})
out = pd.DataFrame(rows)
print(out.round(3).to_string(index=False))
print(out.groupby("pool")[["n", "face_frac", "edge_frac", "corner_frac", "on_lattice"]].mean().round(3))
print(sel["candidate_source"].value_counts() if "candidate_source" in sel else "")
