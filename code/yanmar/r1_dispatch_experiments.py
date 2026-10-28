"""R1 dispatch experiments: E0 corrected dispatch, E4 calibration, E6 beta x kappa, E7 oracle evaluation.

    python -m yanmar.r1_dispatch_experiments      (from journal_extension/)

Writes writing/Yanmar/results/r1_dispatch/*.csv|json
"""
from __future__ import annotations

import matplotlib  # noqa: F401  (must precede pandas on this install)

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import FEATURE_COLUMNS  # noqa: E402
from yanmar.active_sampling_benchmark import _fit  # noqa: E402
from yanmar.dataset import YANMAR_ROOT, load_operating_data, sparse_active_learning_split  # noqa: E402
from yanmar.r1_dispatch import (  # noqa: E402
    ACTIONS, RESULT_DIR, TS_GRID, DispatchConfig, dispatch_profiles, oracle_table, run_dispatch, surrogate_table,
)
from yanmar.r1_acquisition_validation import VAL_PATH  # noqa: E402
from yanmar.r1_pool_benchmark import DENSE_PATH, _rk_label  # noqa: E402
from yanmar.r1_pool_revised import MAIN_BATCH  # noqa: E402
KAPPAS = (0.0, 1.0, 1.645, 1.96, 2.576, 3.0)
BETAS = (0.0, 0.5, 1.0, 2.5)


def conformal_kappa(abs_err: np.ndarray, sigma: np.ndarray, level: float = 0.95) -> float:
    r = abs_err / np.maximum(sigma, 1e-9)
    n = len(r)
    k = int(np.ceil((n + 1) * level))
    return float(np.sort(r)[min(k, n) - 1])


def calib_stats(name: str, abs_err: np.ndarray, sigma: np.ndarray, kappas: dict[str, float]) -> dict:
    rec = {"set": name, "n": int(len(abs_err)), "mae_W": float(abs_err.mean()), "mean_sigma_W": float(sigma.mean()),
           "spearman_sigma_abs_err": float(spearmanr(sigma, abs_err).statistic),
           "coverage_1.96": float((abs_err <= 1.96 * sigma).mean()),
           "kappa95_empirical": float(np.quantile(abs_err / np.maximum(sigma, 1e-9), 0.95))}
    for k, v in kappas.items():
        rec[f"coverage_{k}"] = float((abs_err <= v * sigma).mean())
    return rec


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_operating_data()
    train, _oracle_pool, _test = sparse_active_learning_split(data)

    sel = pd.read_csv(MAIN_BATCH)[FEATURE_COLUMNS]
    y_sel = _rk_label(sel.to_numpy(float))
    aug_train = pd.concat([train, sel.assign(power=y_sel[:, 0], efficiency=y_sel[:, 1])], ignore_index=True)
    models = {"sparse": _fit(train, 21), "augmented": _fit(aug_train, 22)}

    val = pd.read_csv(VAL_PATH)
    dense = pd.read_csv(DENSE_PATH)
    kap = {}
    for name, m in models.items():
        mu, sd = m.predict(val[FEATURE_COLUMNS].to_numpy(float), return_std=True)
        kap[name] = conformal_kappa(np.abs(mu[:, 0] - val["power"].to_numpy()), sd[:, 0])
    calib_rows = []
    for name, m in models.items():
        mu, sd = m.predict(dense[FEATURE_COLUMNS].to_numpy(float), return_std=True)
        e = np.abs(mu[:, 0] - dense["power"].to_numpy())
        calib_rows.append(calib_stats(f"{name}:dense512", e, sd[:, 0], {"conformal": kap[name]}))

    oracle = oracle_table()
    prof = dispatch_profiles()
    tables = {name: surrogate_table(m) for name, m in models.items()}
    tables["oracle"] = (oracle, np.zeros_like(oracle))

    runs, hourly = [], {}
    base = DispatchConfig()
    for name, (mu, sd) in tables.items():
        for tag, cfg in (("base", base), ("R0_stacked", replace(base, beta=2.5)),
                         ("conformal", replace(base, kappa=kap.get(name, 0.0)))):
            if name == "oracle" and tag != "base":
                continue
            h, met = run_dispatch(prof, mu, sd, oracle, cfg)
            runs.append({"policy": name, "setting": tag, **met})
            hourly[(name, tag)] = h
            h.to_csv(RESULT_DIR / f"hourly_{name}_{tag}.csv", index=False)

    grid = []
    for name in ("sparse", "augmented"):
        mu, sd = tables[name]
        for k in (*KAPPAS, kap[name]):
            for b in BETAS:
                _, met = run_dispatch(prof, mu, sd, oracle, replace(base, kappa=float(k), beta=b))
                grid.append({"policy": name, "kappa": float(k), "beta": b, "kappa_is_conformal": k == kap[name], **met})
        for hc in (1, 2, 4, 8, 24):
            _, met = run_dispatch(prof, mu, sd, oracle, replace(base, commitment_hours=hc))
            grid.append({"policy": name, "kappa": base.kappa, "beta": base.beta, "sweep": "H_c", **met})
    pd.DataFrame(runs).to_csv(RESULT_DIR / "dispatch_runs.csv", index=False)
    pd.DataFrame(grid).to_csv(RESULT_DIR / "dispatch_sensitivity.csv", index=False)

    diag = []
    for name in ("sparse", "augmented"):
        h = hourly[(name, "base")]
        on = h["action"].to_numpy() >= 0
        ts_idx = np.rint(h["source_temperature"].to_numpy() - TS_GRID[0]).astype(int)[on]
        a = h["action"].to_numpy()[on]
        pts = pd.DataFrame({"ts_idx": ts_idx, "a": a}).drop_duplicates()
        mu, sd = tables[name]
        p_hat = mu[pts["ts_idx"], pts["a"], 0]
        s_hat = sd[pts["ts_idx"], pts["a"], 0]
        p_or = oracle[pts["ts_idx"], pts["a"], 0]
        e = np.abs(p_hat - p_or)
        calib_rows.append(calib_stats(f"{name}:controller_selected", e, s_hat, {"conformal": kap[name]}))
        pd.DataFrame({"source_temperature": TS_GRID[pts["ts_idx"]], "mean_pressure": ACTIONS[pts["a"], 0],
                      "engine_speed": ACTIONS[pts["a"], 1], "mu_W": p_hat, "sigma_W": s_hat, "oracle_W": p_or}
                     ).to_csv(RESULT_DIR / f"controller_points_{name}.csv", index=False)
        q = pd.qcut(pd.Series(s_hat), 5, labels=False, duplicates="drop")
        for b in sorted(q.unique()):
            mask = (q == b).to_numpy()
            diag.append({"policy": name, "sigma_quintile": int(b), "n": int(mask.sum()),
                         "mean_sigma_W": float(s_hat[mask].mean()), "mae_W": float(e[mask].mean()),
                         "rmse_W": float(np.sqrt((e[mask] ** 2).mean()))})
    calib = pd.DataFrame(calib_rows)
    calib["conformal_kappa95_dense"] = calib["set"].str.split(":").str[0].map(kap)
    calib.to_csv(RESULT_DIR / "calibration.csv", index=False)
    pd.DataFrame(diag).to_csv(RESULT_DIR / "error_vs_sigma_bins.csv", index=False)

    a_s = hourly[("sparse", "base")]["action"].to_numpy()
    a_a = hourly[("augmented", "base")]["action"].to_numpy()
    summary = {"conformal_kappa95": kap, "hours_setpoint_differs": int((a_s != a_a).sum()),
               "frac_setpoint_differs": float((a_s != a_a).mean()),
               "augmented_selection": sel.round(3).to_dict(orient="records")}
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cols = ["policy", "setting", "cfg_kappa", "cfg_beta", "hours_on", "setpoint_changes", "whr_scheduled_mwh",
            "whr_realised_mwh", "uncertainty_discount_mwh", "abs_schedule_error_mwh", "shortfall_mwh",
            "mean_pred_abs_error_mwh", "heat_shortage_hours", "grid_import_mwh", "grid_export_mwh",
            "soc_violations", "max_abs_balance_residual_kw", "soc_terminal_minus_initial_kwh"]
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(pd.DataFrame(runs)[cols].round(2).to_string(index=False))
        print(calib.round(3).to_string(index=False))
        print(json.dumps({k: v for k, v in summary.items() if k != "augmented_selection"}, indent=2))


if __name__ == "__main__":
    main()
