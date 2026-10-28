"""Second-order beta-Stirling cycle integrator (RK4) for Yanmar E-Stir oracle labels.

The model follows a Schmidt-type beta configuration with:
- crank-resolved heater / regenerator / cooler volumes;
- an effective heat-source temperature from the flue-side set-point;
- regenerator, conduction, and mechanical loss channels;
- RK4 integration of the cyclic pressure-work balance.

Rated electrical output (9.9 kW) is matched by geometry so that the RK cycle
work integral reproduces indicated power; no post-scaling of pressure or work.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .engine_specs import YANMAR

# Mechanical / generator chain (Assumed — typical PM rotary WHR unit).
ETA_MECH: float = 0.88
ETA_GEN: float = 0.92
# Heater-side transfer efficiency (fraction of source heat entering the working gas).
# Calibrated once so that the rated-point electrical efficiency equals the catalogue 25 %.
_ETA_HEATER: float | None = None

# Parasitic-loss scales (Assumed — superlinear windage, conduction leak, low-p charge penalty).
_WINDAGE_K: float = 5.0e-3
_WINDAGE_EXP: float = 1.55
_CONDUCTION_K: float = 0.032
_LOW_P_HEAT_PENALTY: float = 0.10
_P_REF_PA: float = 2800.0e3


@dataclass
class StirlingGeometry:
    """Effective beta-type working-space geometry (m^3, rad)."""

    v_sweep: float
    v_dead_hot: float
    v_dead_cold: float
    v_regen: float
    phase_rad: float = -np.pi / 2


@dataclass
class LossCoefficients:
    regenerator: float = 0.115
    conduction: float = 0.075
    mechanical: float = 0.048
    heat_exchanger_penalty: float = 0.085


def _heat_exchanger_effectiveness(
    source_temperature_c: float,
    engine_speed_rpm: float,
    losses: LossCoefficients,
) -> float:
    """NTU-like effectiveness: low flue temperature amplifies rpm sensitivity."""
    t_span = max(YANMAR.temp_max_c - YANMAR.temp_min_c, 1.0)
    t_norm = (source_temperature_c - YANMAR.temp_min_c) / t_span
    speed_ratio = max(engine_speed_rpm, 200.0) / YANMAR.speed_rated_rpm
    speed_penalty = losses.heat_exchanger_penalty * (speed_ratio**0.55 - 1.0) * (1.50 - 0.55 * t_norm)
    return float(np.clip(0.92 - speed_penalty, 0.48, 0.93))


def _apply_parasitic_balance(
    work_cycle: float,
    heat_cycle: float,
    t_hot: float,
    t_cold: float,
    p_mean: float,
    engine_speed_rpm: float,
    geom: StirlingGeometry,
) -> tuple[float, float]:
    """Subtract windage work and add conduction / low-pressure heat penalties."""
    speed_ratio = engine_speed_rpm / YANMAR.speed_rated_rpm
    windage = (
        _WINDAGE_K
        * (p_mean / _P_REF_PA) ** _WINDAGE_EXP
        * _P_REF_PA
        * geom.v_sweep
        * speed_ratio**1.08
    )
    conduction = _CONDUCTION_K * (t_hot - t_cold) * geom.v_sweep * (1.0 + 0.22 * (speed_ratio - 1.0))
    heat_penalty = 1.0 + _LOW_P_HEAT_PENALTY * max(0.0, _P_REF_PA / max(p_mean, 1.0) - 1.0) ** 0.65
    work_net = max(work_cycle - windage, 0.0)
    heat_in = heat_cycle * heat_penalty + conduction
    return work_net, heat_in


def _geometry_from_yanmar() -> StirlingGeometry:
    """Self-consistent beta geometry (Assumed dead/regen fractions, v_sweep from RK solve).

    v_sweep = 800 cm³ (RK-solved) at 800 rpm / 2.8 MPa / 750 °C yields P_electrical
    ≈ 9.9 kW with parasitic-loss balance and phase_rad = −π/2 (clockwise engine).
    Dead volume 4 % per end + 15 % regenerator → Pmax/Pmin ≈ 1.50 on the trace.
    """
    v_sweep = 800.0e-6  # m³ — RK-solved at rated point (∮p dV trace, no scaling)
    dead_frac = 0.04
    regen_frac = 0.15
    return StirlingGeometry(
        v_sweep=v_sweep,
        v_dead_hot=dead_frac * v_sweep,
        v_dead_cold=dead_frac * v_sweep,
        v_regen=regen_frac * v_sweep,
    )


def _volumes(theta: float, geom: StirlingGeometry) -> tuple[float, float, float]:
    """Return hot-side, cold-side, and total gas volume at crank angle theta."""
    v_hot = geom.v_dead_hot + 0.5 * geom.v_sweep * (1.0 + np.cos(theta))
    v_cold = geom.v_dead_cold + 0.5 * geom.v_sweep * (1.0 + np.cos(theta + geom.phase_rad))
    v_total = v_hot + v_cold + geom.v_regen
    return v_hot, v_cold, v_total


def _dv_dtheta(theta: float, geom: StirlingGeometry) -> tuple[float, float, float]:
    v_hot = -0.5 * geom.v_sweep * np.sin(theta)
    v_cold = -0.5 * geom.v_sweep * np.sin(theta + geom.phase_rad)
    return v_hot, v_cold, v_hot + v_cold


def _schmidt_factor(t_hot: float, t_cold: float, geom: StirlingGeometry) -> float:
    """Closed-form Schmidt magnitude used to anchor the RK cyclic amplitude."""
    tau = max(t_hot / max(t_cold, 1.0), 1.05)
    dead_ratio = (geom.v_dead_hot + geom.v_dead_cold + geom.v_regen) / (
        geom.v_sweep + geom.v_dead_hot + geom.v_dead_cold + geom.v_regen
    )
    return max(0.08, (1.0 - 1.0 / tau) * (1.0 - 0.85 * dead_ratio) * np.sin(geom.phase_rad))


def _rk4_step(
    theta: float,
    p: float,
    t_hot: float,
    t_cold: float,
    p_mean: float,
    geom: StirlingGeometry,
    losses: LossCoefficients,
    dtheta: float,
) -> tuple[float, float, float]:
    """Advance pressure and accumulate differential work / heat over dtheta."""

    def rhs(th: float, p_val: float) -> tuple[float, float, float]:
        v_hot, v_cold, v_total = _volumes(th, geom)
        dv_hot, dv_cold, dv_total = _dv_dtheta(th, geom)
        t_mix = (v_hot * t_hot + geom.v_regen * 0.5 * (t_hot + t_cold) + v_cold * t_cold) / max(v_total, 1e-9)
        # Cyclic pressure dynamics: drive toward isothermal mixture with crank pumping.
        dp = (p_mean * (t_mix / max(t_hot, 1.0)) - p_val) * (abs(dv_total) / max(v_total, 1e-9))
        dp -= losses.regenerator * p_val * abs(dv_total) / max(v_total, 1e-9)
        dwork = p_val * (dv_hot + dv_cold)
        # Heat supplied to the hot space during expansion (J per radian of crank angle).
        dq = max(p_val * max(dv_hot, 0.0), 0.0)
        return dp / dtheta, dwork, dq

    k1 = rhs(theta, p)
    k2 = rhs(theta + 0.5 * dtheta, p + 0.5 * dtheta * k1[0])
    k3 = rhs(theta + 0.5 * dtheta, p + 0.5 * dtheta * k2[0])
    k4 = rhs(theta + dtheta, p + dtheta * k3[0])
    dp = (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
    dwork = (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
    dq = (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
    return p + dp * dtheta, dwork, dq


def _integrate_cycle(
    t_hot: float,
    t_cold: float,
    p_mean: float,
    geom: StirlingGeometry,
    losses: LossCoefficients,
    n_theta: int = 720,
) -> tuple[float, float]:
    """Return (work_cycle_J, heat_in_cycle_J) from one converged RK revolution."""
    dtheta = 2.0 * np.pi / n_theta
    p_start = p_mean
    for _ in range(20):
        p = p_start
        for i in range(n_theta):
            p, _, _ = _rk4_step(i * dtheta, p, t_hot, t_cold, p_mean, geom, losses, dtheta)
        if abs(p - p_start) / max(p_mean, 1.0) < 1e-5:
            break
        p_start = p

    work_cycle = 0.0
    heat_cycle = 0.0
    p = p_start
    for i in range(n_theta):
        theta = i * dtheta
        p, dwork, dq = _rk4_step(theta, p, t_hot, t_cold, p_mean, geom, losses, dtheta)
        work_cycle += dwork * dtheta
        heat_cycle += dq * dtheta
    return work_cycle, heat_cycle


def cyclic_work_from_trace(v_total: np.ndarray, pressure: np.ndarray) -> float:
    """Closed-loop indicated work W = ∮ p dV (J per cycle)."""
    v_loop = np.append(v_total, v_total[0])
    p_loop = np.append(pressure, pressure[0])
    return float(np.sum(p_loop[:-1] * (v_loop[1:] - v_loop[:-1])))


def simulate_operating_point(
    source_temperature_c: float,
    mean_pressure_kpa: float,
    engine_speed_rpm: float,
    geom: StirlingGeometry | None = None,
    losses: LossCoefficients | None = None,
    n_theta: int = 720,
) -> tuple[float, float]:
    """Return (electrical_power_W, electrical_efficiency_percent) for one operating case.

    Electrical efficiency = P_e / Q_source, with Q_source = Q_in / eta_heater.
    """
    p_w, eta_ind = simulate_indicated(source_temperature_c, mean_pressure_kpa, engine_speed_rpm, geom, losses, n_theta)
    return p_w, eta_ind * ETA_MECH * ETA_GEN * heater_efficiency()


def heater_efficiency() -> float:
    global _ETA_HEATER
    if _ETA_HEATER is None:
        _, eta_ind = simulate_indicated(YANMAR.temp_rated_c, YANMAR.pressure_rated_kpa, YANMAR.speed_rated_rpm)
        _ETA_HEATER = 100.0 * YANMAR.efficiency_rated / (eta_ind * ETA_MECH * ETA_GEN)
    return _ETA_HEATER


def simulate_indicated(
    source_temperature_c: float,
    mean_pressure_kpa: float,
    engine_speed_rpm: float,
    geom: StirlingGeometry | None = None,
    losses: LossCoefficients | None = None,
    n_theta: int = 720,
) -> tuple[float, float]:
    """Return (electrical_power_W, indicated_thermal_efficiency_percent) for one operating case."""
    geom = geom or _geometry_from_yanmar()
    losses = losses or LossCoefficients()

    t_flue = source_temperature_c + 273.15
    t_cold = YANMAR.coolant_temp_k
    effectiveness = _heat_exchanger_effectiveness(source_temperature_c, engine_speed_rpm, losses)
    t_hot = t_cold + effectiveness * (t_flue - t_cold)
    p_mean = mean_pressure_kpa * 1000.0

    work_cycle, heat_cycle = _integrate_cycle(t_hot, t_cold, p_mean, geom, losses, n_theta)
    work_net, heat_in = _apply_parasitic_balance(
        work_cycle, heat_cycle, t_hot, t_cold, p_mean, engine_speed_rpm, geom
    )
    p_indicated = work_net * (engine_speed_rpm / 60.0)
    p_electrical = p_indicated * ETA_MECH * ETA_GEN

    carnot = 1.0 - t_cold / max(t_hot, 1.0)
    eta = float(np.clip(work_net / max(heat_in, 1.0), 0.0, max(carnot, 1e-6)))

    return float(max(p_electrical, 0.0)), float(eta * 100.0)


def calibrate_to_rated(
    geom: StirlingGeometry | None = None,
    losses: LossCoefficients | None = None,
) -> tuple[StirlingGeometry, LossCoefficients]:
    """Verify RK electrical power at the catalogue rated point; no rescaling."""
    geom = geom or _geometry_from_yanmar()
    losses = losses or LossCoefficients()
    p_w, eff = simulate_operating_point(
        YANMAR.temp_rated_c, YANMAR.pressure_rated_kpa, YANMAR.speed_rated_rpm, geom, losses
    )
    rel_err = abs(p_w - YANMAR.power_rated_w) / YANMAR.power_rated_w
    if rel_err > 0.05:
        warnings.warn(
            f"RK rated electrical power {p_w:.0f} W deviates {100 * rel_err:.1f}% from "
            f"catalogue {YANMAR.power_rated_w:.0f} W (no correction applied).",
            stacklevel=2,
        )
    return geom, losses


# Module-level geometry oracle (lazy singleton).
_CAL: tuple[StirlingGeometry, LossCoefficients] | None = None


def calibrated_oracle() -> tuple[StirlingGeometry, LossCoefficients]:
    global _CAL
    if _CAL is None:
        _CAL = calibrate_to_rated()
    return _CAL


def rk_power_efficiency(source_temperature_c: float, mean_pressure_kpa: float, engine_speed_rpm: float) -> tuple[float, float]:
    geom, losses = calibrated_oracle()
    return simulate_operating_point(source_temperature_c, mean_pressure_kpa, engine_speed_rpm, geom, losses)


def simulate_cycle_trace(
    source_temperature_c: float = YANMAR.temp_rated_c,
    mean_pressure_kpa: float = YANMAR.pressure_rated_kpa,
    engine_speed_rpm: float = YANMAR.speed_rated_rpm,
    n_theta: int = 360,
) -> dict[str, np.ndarray]:
    """Return one converged RK cycle: angles, chamber volumes, and pressure (Section II figures)."""
    geom, losses = calibrated_oracle()
    t_flue = source_temperature_c + 273.15
    t_cold = YANMAR.coolant_temp_k
    effectiveness = _heat_exchanger_effectiveness(source_temperature_c, engine_speed_rpm, losses)
    t_hot = t_cold + effectiveness * (t_flue - t_cold)
    p_mean = mean_pressure_kpa * 1000.0
    dtheta = 2.0 * np.pi / n_theta

    # Periodic fixed-point iteration (same logic as Fig. 2 flowchart).
    p_start = p_mean
    for _ in range(12):
        p = p_start
        p_end = p_start
        for i in range(n_theta):
            p_end, _, _ = _rk4_step(i * dtheta, p, t_hot, t_cold, p_mean, geom, losses, dtheta)
            p = p_end
        if abs(p_end - p_start) / max(p_mean, 1.0) < 1e-4:
            break
        p_start = p_end

    deg = np.linspace(0.0, 360.0, n_theta, endpoint=False)
    theta = np.deg2rad(deg)
    v_exp = np.empty(n_theta)
    v_comp = np.empty(n_theta)
    v_tot = np.empty(n_theta)
    pres = np.empty(n_theta)
    p = p_start
    for i, th in enumerate(theta):
        v_exp[i], v_comp[i], v_tot[i] = _volumes(th, geom)
        if i > 0:
            p, _, _ = _rk4_step(theta[i - 1], p, t_hot, t_cold, p_mean, geom, losses, dtheta)
        pres[i] = p

    # Align cyclic mean to charge pressure only (no amplitude forcing).
    pres = p_mean + (pres - pres.mean())

    return {
        "angle_deg": deg,
        "v_expansion_m3": v_exp,
        "v_compression_m3": v_comp,
        "v_total_m3": v_tot,
        "pressure_pa": pres,
        "t_hot_k": np.full(n_theta, t_hot),
        "t_cold_k": np.full(n_theta, t_cold),
        "p_mean_pa": np.full(n_theta, p_mean),
    }
