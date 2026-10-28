import matplotlib  # noqa: F401

import json

import numpy as np
import pandas as pd

from data_loader import FEATURE_COLUMNS
from yanmar.active_sampling_benchmark import _fit
from yanmar.agents_yanmar import yanmar_physics
from yanmar.r1_dispatch import RESULT_DIR
from yanmar.r1_pool_benchmark import PROPOSER_DIR, _gate, _state
from yanmar.r1_pool_revised import MAIN_BATCH

st = _state()
train, dense = st["train"], st["dense"]
sel = pd.read_csv(MAIN_BATCH)
aug = _fit(pd.concat([train, sel[FEATURE_COLUMNS + ["power", "efficiency"]]], ignore_index=True), 22)
base = _fit(train, 21)
out = {}
for name, m in (("sparse", base), ("augmented", aug)):
    mu = m.predict(dense[FEATURE_COLUMNS].to_numpy(float))
    out[f"{name}_dense_power_mape_pct"] = float(np.mean(np.abs(mu[:, 0] - dense["power"]) / dense["power"]) * 100)
    grid = pd.DataFrame([(t, p, w) for t in (773.15, 923.15, 1073.15) for w in np.linspace(400, 1300, 10)
                         for p in np.linspace(1500, 3000, 16)], columns=FEATURE_COLUMNS)
    pred = m.predict(grid.to_numpy(float))
    out[f"{name}_monotonicity_penalty_W"] = float(yanmar_physics().monotonicity_penalty(grid, pred))
    tmp = grid.assign(pw=pred[:, 0])
    viol = 0
    for _, part in tmp.groupby(["source_temperature", "engine_speed"]):
        viol += int((np.diff(part.sort_values("mean_pressure")["pw"].to_numpy()) < 0).sum())
    out[f"{name}_monotonicity_violations"] = viol
out["dense_mean_power_W"] = float(dense["power"].mean())
sizes = {}
for stem in ("deepseek", "deepseek-kag"):
    sizes[stem] = [len(_gate(pd.read_csv(PROPOSER_DIR / f"{stem}_run{r:02d}_records.csv"))) for r in range(5)]
out["gated_pool_sizes"] = sizes
runs = pd.read_csv(RESULT_DIR / "dispatch_runs.csv")
out["max_abs_residual_kw"] = float(runs["max_abs_balance_residual_kw"].max())
t = pd.read_csv(PROPOSER_DIR / "r1_pool_revised_tests.csv")
out["pool_tests"] = t.to_dict(orient="records")
s = pd.read_csv(PROPOSER_DIR / "r1_selector_tests.csv")
out["selector_tests"] = s.to_dict(orient="records")
print(json.dumps(out, indent=1, default=float))
