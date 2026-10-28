import matplotlib  # noqa: F401

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from data_loader import FEATURE_COLUMNS
from yanmar.active_sampling_benchmark import BUDGET, _fit
from yanmar.r1_acquisition_validation import VAL_PATH, make_selector
from yanmar.r1_pool_benchmark import CONTROLS, PROPOSER_DIR, _rk_label, _state, build_pools

LS = (0.45, 0.65, 0.85, 0.95, 1.0)
_POOLS = {}


def _task(args):
    name, r, seed = args
    st = _state()
    if not _POOLS:
        _POOLS.update(build_pools(st["oracle"], [0, 1, 2, 3, 4]))
    pool = _POOLS[(name, "bounds", r)]
    val = pd.read_csv(VAL_PATH)
    model = _fit(st["train"], 21 + seed)
    out = []
    for ls in LS:
        sel = make_selector("cell-cap", ls)(pool, st["train"], model, BUDGET)
        y = _rk_label(sel[FEATURE_COLUMNS].to_numpy(float))
        lab = sel[FEATURE_COLUMNS].assign(power=y[:, 0], efficiency=y[:, 1])
        final = _fit(pd.concat([st["train"], lab], ignore_index=True), 22 + seed)
        mu = final.predict(val[FEATURE_COLUMNS].to_numpy(float))
        out.append({"pool": name, "run_id": r, "seed": seed, "lambda_s": ls,
                    "val_power_mae": float(np.abs(mu[:, 0] - val["power"]).mean())})
    return out


if __name__ == "__main__":
    tasks = [(n, r, s) for n in CONTROLS for r in range(5) for s in range(10)]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("R1_WORKERS", "6"))) as ex:
        rows = pd.DataFrame([x for o in ex.map(_task, tasks, chunksize=3) for x in o])
    rows.to_csv(PROPOSER_DIR / "r1_acq_validation_controls_rows.csv", index=False)
    print(rows.groupby(["pool", "lambda_s"])["val_power_mae"].mean().unstack().round(2))
