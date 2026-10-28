"""R1 E2: solver-vs-measured and surrogate LOO on public measured Stirling maps (GM 4L23, GPU-3).

The oracle model form of stirling_rk_solver (RK4 cycle + windage / conduction /
heater-effectiveness channels) is re-calibrated to the GM 4L23 measured map
(heater-tube temperature 1000/1200/1400 F, mean pressure 500-2500 psi,
200-3000 rpm) and compared with the measurements:

  (a) single-point calibration, as done for the Yanmar oracle: effective swept volume
      and heater efficiency fitted at one reference point, loss coefficients fixed;
  (b) level calibration: all loss coefficients fitted on the 1200 F level,
      validated on 1000 F and 1400 F.

Also reports leave-one-out error of the bootstrap polynomial-ridge surrogate on the
measured points and on the 27 Yanmar seed cases.

    python -m yanmar.r1_measured_validation      (from journal_extension/)
"""
from __future__ import annotations

import matplotlib  # noqa: F401  (must precede pandas on this install)

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import FEATURE_COLUMNS, load_raw_operating_data  # noqa: E402
from physics_constraints import PhysicsConstraintAgent  # noqa: E402
from surrogate_models import BootstrapSurrogate  # noqa: E402
from yanmar.dataset import YANMAR_ROOT, load_operating_data, sparse_active_learning_split  # noqa: E402
from yanmar.stirling_rk_solver import (  # noqa: E402
    ETA_GEN, ETA_MECH, LossCoefficients, StirlingGeometry, _integrate_cycle,
)

MEASURED_PATH = Path(r"D:\IEEE-CONF\CODE\jordon源数据.txt")
GPU3_PATH = YANMAR_ROOT / "results" / "r1_measured" / "gpu3_helium_nasa_tm79103.csv"
OUT_DIR = YANMAR_ROOT / "results" / "r1_measured"
PSI_TO_PA = 6894.757
HP_TO_W = 745.699872
T_COLD_K = 330.0
GPU3_T_COLD_K = 286.15  # 13 C cooler inlet, NASA TM-79103
OMEGA_REF = 1800.0
P_REF_PA = 1500.0 * PSI_TO_PA
REF_POINT = (922.04, 1500.0, 1800.0)  # 1200 F, 1500 psi, 1800 rpm
GPU3_REF = (922.04, 1000.0, 3000.0)  # 1200 F, 1000 psi, 3000 rpm: NASA max He power (5.25 hp)


def _cycle_tables(t_cold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per unit (p_mean V_sweep) indicated work and heat input versus hot-gas temperature."""
    geom = StirlingGeometry(v_sweep=1.0, v_dead_hot=0.04, v_dead_cold=0.04, v_regen=0.15)
    t_grid = np.linspace(420.0, 1150.0, 74)
    w, q = [], []
    for th in t_grid:
        wc, qc = _integrate_cycle(th, t_cold, 1.0, geom, LossCoefficients(), n_theta=720)
        w.append(wc)
        q.append(qc)
    return t_grid, np.array(w), np.array(q)


_CYCLE: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def cycle_tables(t_cold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = round(float(t_cold), 2)
    if key not in _CYCLE:
        _CYCLE[key] = _cycle_tables(t_cold)
    return _CYCLE[key]


T_TAB, W_TAB, Q_TAB = cycle_tables(T_COLD_K)


def model(theta: np.ndarray, x: np.ndarray, t_cold: float = T_COLD_K) -> tuple[np.ndarray, np.ndarray]:
    """theta = [V_eff, K_w, e_w, c_hx, K_c, eta_h]; x = [T_heater K, p psi, rpm]; returns (P, eta)."""
    t_tab, w_tab, q_tab = cycle_tables(t_cold)
    v, k_w, e_w, c_hx, k_c, eta_h = theta
    t_h, p_psi, rpm = x[:, 0], x[:, 1], x[:, 2]
    p = p_psi * PSI_TO_PA
    sr = rpm / OMEGA_REF
    eff = np.clip(0.92 - c_hx * (sr**0.55 - 1.0), 0.48, 0.93)
    t_hot = t_cold + eff * (t_h - t_cold)
    w_cyc = p * v * np.interp(t_hot, t_tab, w_tab)
    q_cyc = p * v * np.interp(t_hot, t_tab, q_tab)
    windage = k_w * (p / P_REF_PA) ** e_w * P_REF_PA * v * sr**1.08
    cond = k_c * (t_hot - t_cold) * v * (1.0 + 0.22 * (sr - 1.0))
    heat_pen = 1.0 + 0.10 * np.maximum(0.0, P_REF_PA / p - 1.0) ** 0.65
    w_net = np.maximum(w_cyc - windage, 0.0)
    q_in = q_cyc * heat_pen + cond
    power = w_net * rpm / 60.0 * ETA_MECH * ETA_GEN
    carnot = 1.0 - t_cold / t_hot
    eta_ind = np.clip(w_net / np.maximum(q_in, 1e-9), 0.0, carnot)
    return power, eta_ind * ETA_MECH * ETA_GEN * eta_h


THETA0 = np.array([1e-4, 5.0e-3, 1.55, 0.085, 0.032, 0.6])


def _residuals(theta, x, y, free, t_cold, use_eff):
    th = THETA0.copy()
    th[free] = theta
    p, e = model(th, x, t_cold)
    rp = (p - y[:, 0]) / y[:, 0]
    if not use_eff:
        return rp
    return np.concatenate([rp, (e - y[:, 1]) / y[:, 1]])


def calibrate(x, y, free, t_cold=T_COLD_K, use_eff=True):
    lo = np.array([1e-7, 0.0, 0.5, 0.0, 0.0, 0.05])[free]
    hi = np.array([1e-1, 0.5, 3.0, 0.5, 5.0, 1.0])[free]
    x0 = THETA0[free]
    # Volume scale first: closed-form start so the optimiser begins near the data.
    if 0 in free:
        p0, _ = model(THETA0, x, t_cold)
        x0 = x0.copy()
        x0[list(free).index(0)] = THETA0[0] * float(np.median(y[:, 0] / np.maximum(p0, 1e-12)))
    sol = least_squares(_residuals, x0, bounds=(lo, hi), args=(x, y, free, t_cold, use_eff), x_scale="jac")
    th = THETA0.copy()
    th[free] = sol.x
    return th


def errors(theta, x, y, t_cold=T_COLD_K) -> dict:
    p, e = model(theta, x, t_cold)
    rp = (p - y[:, 0]) / y[:, 0]
    out = {"n": int(len(y)), "power_mape_pct": float(100 * np.abs(rp).mean()),
           "power_median_ape_pct": float(100 * np.median(np.abs(rp))),
           "power_bias_pct": float(100 * rp.mean())}
    if np.isfinite(y[:, 1]).all() and np.any(y[:, 1] > 0):
        re = (e - y[:, 1]) / y[:, 1]
        out["eff_mape_pct"] = float(100 * np.abs(re).mean())
        out["eff_bias_pct"] = float(100 * re.mean())
    return out


def load_gpu3() -> pd.DataFrame:
    raw = pd.read_csv(GPU3_PATH, comment="#")
    df = pd.DataFrame({
        "source_temperature": (raw["T_heater_F"] - 32.0) * 5.0 / 9.0 + 273.15,
        "mean_pressure": raw["p_psi"].astype(float),
        "engine_speed": raw["rpm"].astype(float),
        "power": raw["power_hp"].astype(float) * HP_TO_W,
        "efficiency": raw["efficiency_pct"].astype(float) / 100.0,
    })
    return df.reset_index(drop=True)


def surrogate_loo(df: pd.DataFrame, eff_col: str = "efficiency") -> dict:
    x = df[FEATURE_COLUMNS].to_numpy(float)
    y = df[["power", eff_col]].to_numpy(float)
    phys = PhysicsConstraintAgent(
        temperature_bounds=(x[:, 0].min(), x[:, 0].max()), pressure_bounds=(x[:, 1].min(), x[:, 1].max()),
        speed_bounds=(x[:, 2].min(), x[:, 2].max()), efficiency_bounds=(0.0, 1.0))
    pred = np.zeros_like(y)
    for i in range(len(df)):
        m = np.ones(len(df), dtype=bool)
        m[i] = False
        s = BootstrapSurrogate(n_estimators=90, degree=3, alpha=0.08, random_state=21, physics_agent=phys).fit(x[m], y[m])
        pred[i] = s.predict(x[i : i + 1])[0]
    ap = np.abs(pred[:, 0] - y[:, 0])
    return {"n": int(len(df)), "power_mae": float(ap.mean()), "power_mape_pct": float(100 * (ap / y[:, 0]).mean()),
            "eff_mae_pp": float(100 * np.abs(pred[:, 1] - y[:, 1]).mean())}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meas = load_raw_operating_data(MEASURED_PATH)
    x = meas[FEATURE_COLUMNS].to_numpy(float)
    y = meas[["power", "efficiency"]].to_numpy(float)
    t_levels = sorted(meas["source_temperature"].round(2).unique())

    ref = np.argmin(np.abs(x - np.array(REF_POINT)).sum(axis=1))
    th_a = calibrate(x[[ref]], y[[ref]], free=[0, 5])
    cal = np.isclose(x[:, 0], t_levels[1], atol=0.1)
    th_b = calibrate(x[cal], y[cal], free=[0, 1, 2, 3, 4, 5])

    res = {
        "data": {"source": "GM 4L23 measured performance map (Martini, Stirling Engine Design Manual, NASA CR-168088)",
                 "n_points": int(len(meas)), "heater_temperatures_K": [float(t) for t in t_levels],
                 "pressure_psi": [float(meas["mean_pressure"].min()), float(meas["mean_pressure"].max())],
                 "speed_rpm": [float(meas["engine_speed"].min()), float(meas["engine_speed"].max())],
                 "reference_point": [float(v) for v in x[ref]]},
        "single_point_calibration": {"theta": th_a.tolist(), "all_other_points": errors(th_a, np.delete(x, ref, 0), np.delete(y, ref, 0))},
        "level_calibration_1200F": {"theta": th_b.tolist(), "calibration_level": errors(th_b, x[cal], y[cal]),
                                    "validation_levels": errors(th_b, x[~cal], y[~cal])},
        "surrogate_loo_measured_gm4l23": surrogate_loo(meas),
    }
    data = load_operating_data()
    train, _, _ = sparse_active_learning_split(data)
    res["surrogate_loo_yanmar_27_seeds"] = surrogate_loo(train)

    p_b, e_b = model(th_b, x)
    pd.DataFrame({"T_heater_K": x[:, 0], "p_psi": x[:, 1], "rpm": x[:, 2], "power_meas": y[:, 0], "eff_meas": y[:, 1],
                  "power_model": p_b, "eff_model": e_b, "calibration_set": cal}).to_csv(OUT_DIR / "gm4l23_model_vs_measured.csv", index=False)

    gpu = load_gpu3()
    xg = gpu[FEATURE_COLUMNS].to_numpy(float)
    yg = np.column_stack([gpu["power"].to_numpy(float), gpu["efficiency"].to_numpy(float)])
    has_eta = np.isfinite(yg[:, 1])
    ref_g = np.argmin(np.abs(xg - np.array(GPU3_REF)).sum(axis=1))
    th_g = calibrate(xg[[ref_g]], yg[[ref_g]], free=[0, 5], t_cold=GPU3_T_COLD_K)
    other = np.ones(len(gpu), dtype=bool)
    other[ref_g] = False
    th_g6 = calibrate(xg[has_eta], yg[has_eta], free=[0, 1, 2, 3, 4, 5], t_cold=GPU3_T_COLD_K)
    pg, eg = model(th_g6, xg, GPU3_T_COLD_K)
    gpu_out = gpu.copy()
    gpu_out["power_model"] = pg
    gpu_out["eff_model"] = eg
    gpu_out.to_csv(OUT_DIR / "gpu3_model_vs_measured.csv", index=False)
    res["gpu3"] = {
        "data": {"source": "GPU-3 helium brake map, NASA Lewis (Thieme, NASA TM-79103, 1979)",
                 "n_power": int(len(gpu)), "n_efficiency": int(has_eta.sum()),
                 "heater_temperatures_K": [float(t) for t in sorted(gpu["source_temperature"].round(2).unique())],
                 "reference_point": [float(v) for v in xg[ref_g]]},
        "single_point_calibration": {
            "theta": th_g.tolist(),
            "other_points_with_efficiency": errors(th_g, xg[other & has_eta], yg[other & has_eta], GPU3_T_COLD_K),
            "other_points_power": errors(th_g, xg[other], yg[other], GPU3_T_COLD_K),
            "off_1200F_power": errors(th_g, xg[~has_eta], yg[~has_eta], GPU3_T_COLD_K),
        },
        "level_calibration_1200F": {
            "theta": th_g6.tolist(),
            "calibration_level": errors(th_g6, xg[has_eta], yg[has_eta], GPU3_T_COLD_K),
            "validation_levels_power": errors(th_g6, xg[~has_eta], yg[~has_eta], GPU3_T_COLD_K),
        },
        "surrogate_loo_measured_gpu3": surrogate_loo(gpu.loc[has_eta, FEATURE_COLUMNS + ["power", "efficiency"]]),
    }

    (OUT_DIR / "measured_validation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
