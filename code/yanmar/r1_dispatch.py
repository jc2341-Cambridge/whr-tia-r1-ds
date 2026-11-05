"""R1 dispatch (Section VI): corrected objective, energy-feasible BESS, explicit grid balance.

Commitment rule (no horizon optimisation): at the start of each H_c-hour window the
controller picks the (p, ω) set-point that minimises

    J_q(a) = sum_{t in W_q} | P_pen,t(a) - P_res,t |  +  beta * sum_{t in W_q} U_t(a)

over the ramp-feasible, generator-feasible action set, where
P_pen,t(a) = n_t(a) [mu_P(a) - kappa sigma_P(a)] and U_t(a) = n_t(a) sigma_P(a).
Uncertainty enters once through kappa when beta = 0 (default).

n_t(a) = min(N, floor(Q_avail,t / q(a))) is the number of units the available source
heat can run at set-point a, with q(a) = mu_P / mu_eta the heat drawn per unit.

Every policy is evaluated against the same RK4 oracle: the realised unit output is the
oracle power (generator-limited), and the realised unit count is limited by the oracle
heat draw. The BESS then balances the realised net load.
"""
from __future__ import annotations

import matplotlib  # noqa: F401  (must precede pandas on this install)

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import FEATURE_COLUMNS

from .dataset import YANMAR_ROOT
from .dispatch import CASE, campus_whr_profiles
from .engine_specs import YANMAR

RESULT_DIR = YANMAR_ROOT / "results" / "r1_dispatch"
ORACLE_TABLE = RESULT_DIR / "oracle_action_table.csv"

PRESSURES = np.linspace(1500.0, 3000.0, 7)
SPEEDS = np.linspace(400.0, 1300.0, 10)
ACTIONS = np.array([(p, w) for p in PRESSURES for w in SPEEDS])
TS_GRID = 773.15 + np.arange(301, dtype=float)


@dataclass(frozen=True)
class DispatchConfig:
    kappa: float = 1.96
    beta: float = 0.0
    commitment_hours: int = 4
    ramp_dp_kpa: float = 500.0
    ramp_dw_rpm: float = 300.0
    unit_gen_limit_kw: float = YANMAR.power_rated_w / 1000.0
    n_units: int = CASE.stirling_units
    unit_design_heat_kw: float = YANMAR.power_rated_w / 1000.0 / YANMAR.efficiency_rated
    bess_energy_kwh: float = CASE.storage_capacity_kwh
    bess_power_kw: float = CASE.storage_power_limit_kw
    soc_min_frac: float = CASE.storage_soc_min_fraction
    soc_max_frac: float = CASE.storage_soc_max_fraction
    soc_init_frac: float = CASE.storage_initial_soc_fraction
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    export_limit_kw: float = 1000.0
    dt_h: float = 1.0


def _oracle_row(ts_k: float) -> list[tuple[float, float]]:
    from .stirling_rk_solver import rk_power_efficiency

    return [rk_power_efficiency(ts_k - 273.15, p, w) for p, w in ACTIONS]


def oracle_table(workers: int = 6) -> np.ndarray:
    """RK4 power [W] and efficiency [-] on TS_GRID x ACTIONS, cached; shape (n_ts, n_act, 2)."""
    if ORACLE_TABLE.exists():
        df = pd.read_csv(ORACLE_TABLE)
    else:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_oracle_row, TS_GRID, chunksize=4))
        recs = [
            (ts, p, w, pw, eff / 100.0)
            for ts, row in zip(TS_GRID, rows)
            for (p, w), (pw, eff) in zip(ACTIONS, row)
        ]
        df = pd.DataFrame(recs, columns=["source_temperature", "mean_pressure", "engine_speed", "power", "efficiency"])
        df.to_csv(ORACLE_TABLE, index=False)
    arr = df[["power", "efficiency"]].to_numpy(float).reshape(len(TS_GRID), len(ACTIONS), 2)
    return arr


def surrogate_table(model) -> tuple[np.ndarray, np.ndarray]:
    """Surrogate mean and std on TS_GRID x ACTIONS; shapes (n_ts, n_act, 2)."""
    x = np.array([(ts, p, w) for ts in TS_GRID for p, w in ACTIONS])
    mu, sd = model.predict(x, return_std=True)
    return mu.reshape(len(TS_GRID), len(ACTIONS), 2), sd.reshape(len(TS_GRID), len(ACTIONS), 2)


class OracleModel:
    """Perfect-information reference: oracle mean, zero spread."""

    def __init__(self, table: np.ndarray) -> None:
        self.table = table


def dispatch_profiles() -> pd.DataFrame:
    prof = campus_whr_profiles()
    ts = np.clip(prof["source_temperature"].to_numpy(float), TS_GRID[0], TS_GRID[-1])
    prof["ts_idx"] = np.rint(ts - TS_GRID[0]).astype(int)
    avail = prof["heat_index"].to_numpy(float) * (~prof["stirling_outage"].to_numpy(bool))
    prof["heat_avail_frac"] = np.clip(avail, 0.0, 1.0)
    return prof


def run_dispatch(
    prof: pd.DataFrame,
    mu: np.ndarray,
    sd: np.ndarray,
    oracle: np.ndarray,
    cfg: DispatchConfig = DispatchConfig(),
) -> tuple[pd.DataFrame, dict]:
    n_h = len(prof)
    ts_idx = prof["ts_idx"].to_numpy(int)
    load = prof["load_kw"].to_numpy(float)
    pv = prof["pv_kw"].to_numpy(float)
    bess_out = prof["bess_outage"].to_numpy(bool)
    q_avail = prof["heat_avail_frac"].to_numpy(float) * cfg.n_units * cfg.unit_design_heat_kw
    p_res = np.maximum(load - pv, 0.0)

    mu_p = mu[ts_idx, :, 0] / 1000.0          # kW per unit, (n_h, n_act)
    sd_p = sd[ts_idx, :, 0] / 1000.0
    mu_e = np.clip(mu[ts_idx, :, 1], 1e-3, None)
    q_unit = mu_p / mu_e
    n_hat = np.minimum(cfg.n_units, np.floor(q_avail[:, None] / np.maximum(q_unit, 1e-6)))
    gen_ok = mu_p <= cfg.unit_gen_limit_kw + 1e-9
    p_pen = n_hat * np.maximum(mu_p - cfg.kappa * sd_p, 0.0)
    u_sys = n_hat * sd_p
    cost_h = np.abs(p_pen - p_res[:, None]) + cfg.beta * u_sys

    act = np.full(n_h, -1, dtype=int)
    prev = -1
    n_changes = 0
    for t0 in range(0, n_h, cfg.commitment_hours):
        w = slice(t0, min(t0 + cfg.commitment_hours, n_h))
        feasible = gen_ok[w].all(axis=0) & (n_hat[w].sum(axis=0) > 0)
        if prev >= 0:
            dp = np.abs(ACTIONS[:, 0] - ACTIONS[prev, 0])
            dw = np.abs(ACTIONS[:, 1] - ACTIONS[prev, 1])
            feasible &= (dp <= cfg.ramp_dp_kpa + 1e-9) & (dw <= cfg.ramp_dw_rpm + 1e-9)
        if not feasible.any():
            act[w] = -1
            prev = -1
            continue
        j = np.where(feasible, cost_h[w].sum(axis=0), np.inf)
        best = int(np.argmin(j))
        if prev >= 0 and best != prev:
            n_changes += 1
        act[w] = best
        prev = best

    on = act >= 0
    a = np.where(on, act, 0)
    rows = np.arange(n_h)
    sched_units = np.where(on, n_hat[rows, a], 0.0)
    sched_kw = np.where(on, p_pen[rows, a], 0.0)
    mean_kw = np.where(on, sched_units * mu_p[rows, a], 0.0)
    discount_kw = np.where(on, sched_units * cfg.kappa * sd_p[rows, a], 0.0)
    unc_kw = np.where(on, u_sys[rows, a], 0.0)

    o_p_raw = oracle[ts_idx, a, 0] / 1000.0
    o_e = np.clip(oracle[ts_idx, a, 1], 1e-3, None)
    o_q = o_p_raw / o_e
    real_units = np.where(on, np.minimum(sched_units, np.floor(q_avail / np.maximum(o_q, 1e-6))), 0.0)
    gen_clip = on & (o_p_raw > cfg.unit_gen_limit_kw)
    real_unit_kw = np.minimum(o_p_raw, cfg.unit_gen_limit_kw)
    whr_real = np.where(on, real_units * real_unit_kw, 0.0)
    heat_short = on & (real_units < sched_units)

    e_max = cfg.bess_energy_kwh
    soc_lo, soc_hi = cfg.soc_min_frac * e_max, cfg.soc_max_frac * e_max
    soc = cfg.soc_init_frac * e_max
    p_target = float(np.mean(load - pv - sched_kw))
    ch = np.zeros(n_h)
    dis = np.zeros(n_h)
    soc_tr = np.zeros(n_h)
    imp = np.zeros(n_h)
    exp_ = np.zeros(n_h)
    curt = np.zeros(n_h)
    for t in range(n_h):
        net = load[t] - pv[t] - whr_real[t]
        if not bess_out[t]:
            if net > p_target:
                dis[t] = max(min(net - p_target, cfg.bess_power_kw, (soc - soc_lo) * cfg.eta_discharge / cfg.dt_h), 0.0)
            elif net < p_target:
                ch[t] = max(min(p_target - net, cfg.bess_power_kw, (soc_hi - soc) / (cfg.eta_charge * cfg.dt_h)), 0.0)
        soc = soc + cfg.eta_charge * ch[t] * cfg.dt_h - dis[t] * cfg.dt_h / cfg.eta_discharge
        soc_tr[t] = soc
        grid = net - dis[t] + ch[t]
        if grid >= 0:
            imp[t] = grid
        else:
            exp_[t] = min(-grid, cfg.export_limit_kw)
            curt[t] = -grid - exp_[t]
    resid = load - (pv - curt) - whr_real - dis + ch - imp + exp_

    out = pd.DataFrame({
        "hour": prof["hour"].to_numpy(), "day": prof["day"].to_numpy(), "hour_of_day": prof["hour_of_day"].to_numpy(),
        "load_kw": load, "pv_kw": pv, "source_temperature": TS_GRID[ts_idx], "heat_avail_kw": q_avail,
        "action": act, "selected_pressure": np.where(on, ACTIONS[a, 0], np.nan),
        "selected_speed": np.where(on, ACTIONS[a, 1], np.nan),
        "units_scheduled": sched_units, "units_realised": real_units,
        "whr_mean_kw": mean_kw, "whr_scheduled_kw": sched_kw, "whr_uncertainty_kw": unc_kw,
        "uncertainty_discount_kw": discount_kw, "whr_kw": whr_real,
        "bess_charge_kw": ch, "bess_discharge_kw": dis, "storage_power_kw": ch - dis, "storage_soc_kwh": soc_tr,
        "grid_import_kw": imp, "grid_export_kw": exp_, "pv_curtailed_kw": curt, "balance_residual_kw": resid,
    })
    err = sched_kw - whr_real
    m = {
        **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
        "hours_on": int(on.sum()),
        "setpoint_changes": int(n_changes),
        "whr_scheduled_mwh": sched_kw.sum() * cfg.dt_h / 1000,
        "whr_mean_pred_mwh": mean_kw.sum() * cfg.dt_h / 1000,
        "whr_realised_mwh": whr_real.sum() * cfg.dt_h / 1000,
        "uncertainty_discount_mwh": discount_kw.sum() * cfg.dt_h / 1000,
        "abs_schedule_error_mwh": np.abs(err).sum() * cfg.dt_h / 1000,
        "shortfall_mwh": np.maximum(err, 0).sum() * cfg.dt_h / 1000,
        "shortfall_hours": int((err > 1e-6).sum()),
        "mean_pred_abs_error_mwh": np.abs(mean_kw - whr_real).sum() * cfg.dt_h / 1000,
        "heat_shortage_hours": int(heat_short.sum()),
        "gen_limit_clip_hours": int(gen_clip.sum()),
        "grid_import_mwh": imp.sum() * cfg.dt_h / 1000,
        "grid_export_mwh": exp_.sum() * cfg.dt_h / 1000,
        "grid_peak_import_kw": float(imp.max()),
        "pv_curtailed_mwh": curt.sum() * cfg.dt_h / 1000,
        "bess_throughput_mwh": (ch + dis).sum() * cfg.dt_h / 1000,
        "bess_equiv_cycles": dis.sum() * cfg.dt_h / (soc_hi - soc_lo),
        "soc_min_kwh": float(soc_tr.min()), "soc_max_kwh": float(soc_tr.max()),
        "soc_terminal_minus_initial_kwh": float(soc_tr[-1] - cfg.soc_init_frac * e_max),
        "soc_violations": int(((soc_tr < soc_lo - 1e-6) | (soc_tr > soc_hi + 1e-6)).sum()),
        "bess_power_violations": int(((ch > cfg.bess_power_kw + 1e-6) | (dis > cfg.bess_power_kw + 1e-6)).sum()),
        "simultaneous_charge_discharge": int(((ch > 0) & (dis > 0)).sum()),
        "max_abs_balance_residual_kw": float(np.abs(resid).max()),
        "import_setpoint_kw": p_target,
    }
    return out, m
