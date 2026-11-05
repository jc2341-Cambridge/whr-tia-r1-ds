"""Yanmar-specific microgrid profiles and merit-order dispatch.

Case study (Fig. 7): grid-connected industrial-campus WHR microgrid.

Heat-source narrative (500–800 C flue, Yanmar SE220-100 fleet):
  High-temperature WHR is tied to **batch / shift manufacturing** (880 kW nominal
  process hall) and associated **plant steam auxiliaries** (~35 % of HVAC load).
  Cold storage and warehouse loads do NOT provide flue
  gas in the 500–800 C envelope. The source is therefore **shift-intermittent**:
  weekday day-shift furnaces hot, nights/weekends cool down (thermal lag, not a
  flat 0.9 MW floor). WHR may approach 0 on extended shutdown; nominal cap 1.485 MW (±5 % hourly).

BESS: C&I import setpoint controller (peak-shave / valley-fill from grid).
  net = load - WHR - PV; discharge if net > P_target; charge if net < P_target
  (including night valley fill without PV). Not PV-surplus self-consumption.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .engine_specs import YANMAR, temp_c_to_k


@dataclass(frozen=True)
class MicrogridCaseStudy:
    """Documented sizing for the Yanmar WHR–PV–storage dispatch illustration."""

    stirling_units: int = 140
    unit_rated_kw: float = YANMAR.power_rated_w / 1000.0
    whr_hard_cap_kw: float = 1485.0
    whr_peak_jitter_fraction: float = 0.05
    whr_target_mean_kw: float = 1080.0

    pv_dc_capacity_kwp: float = 1600.0
    pv_ac_peak_kw: float = 1300.0
    pv_target_capacity_factor: float = 0.105

    load_office_campus_kw: float = 210.0
    load_manufacturing_kw: float = 880.0
    load_hvac_steam_kw: float = 340.0
    load_refrigeration_kw: float = 240.0
    load_warehouse_aux_kw: float = 170.0

    storage_capacity_kwh: float = 6000.0
    storage_power_limit_kw: float = 2000.0
    storage_soc_min_fraction: float = 0.10
    storage_soc_max_fraction: float = 0.95
    storage_initial_soc_fraction: float = 0.50

    grid_firm_import_limit_kw: float = 1100.0

    reference_notes: tuple[str, ...] = (
        "WHR heat source: shift-intermittent manufacturing + plant steam (not office/IT supply).",
        "140 x 9.9 kW = 1.386 MW nameplate; nominal cap 1.485 MW with ±5 % hourly peak jitter.",
        "PV: 1.6 MWp / 1.30 MW AC; UK CF ~10.5 %.",
        "BESS: 2 MW / 6 MWh; import-setpoint peak-shave / valley-fill (grid charging at night).",
    )

    @property
    def fleet_nameplate_kw(self) -> float:
        return self.stirling_units * self.unit_rated_kw

    @property
    def nominal_campus_load_kw(self) -> float:
        return (
            self.load_office_campus_kw
            + self.load_manufacturing_kw
            + self.load_hvac_steam_kw
            + self.load_refrigeration_kw
            + self.load_warehouse_aux_kw
        )


CASE = MicrogridCaseStudy()

LOAD_FLOOR_KW = 1080.0
LOAD_REF_PEAK_KW = 2150.0  # soft reference only — never a hard clip
WHR_NAMEPLATE_KW = CASE.fleet_nameplate_kw
BESS_ZOOM_CYCLING_DAYS = (300, 301, 302)  # zoom inset: daily setpoint + morning SOC headroom

LOAD_COMPONENT_COLUMNS: tuple[str, ...] = (
    "load_manufacturing_kw",
    "load_hvac_heating_kw",
    "load_hvac_cooling_kw",
    "load_refrigeration_kw",
    "load_office_campus_kw",
    "load_warehouse_aux_kw",
)


@dataclass(frozen=True)
class EquipmentMaintenanceSchedule:
    """Documented zero-output / BESS-lock maintenance windows (day-of-year, inclusive)."""

    stirling_major_overhaul_days: int = 7
    spring_trough_search: tuple[int, int] = (60, 125)  # spring shoulder band for overhaul placement
    pv_maintenance: tuple[int, int] = (150, 151)  # early-summer concentrated maintenance, 2 d
    bess_inspection: tuple[int, int] = (275, 276)  # autumn trough, 2 d battery annual check
    stirling_minor_outage_count: int = 3
    stirling_minor_outage_length_days: tuple[int, int] = (1, 2)  # each event 1–2 d


MAINTENANCE = EquipmentMaintenanceSchedule()


def _finalize_campus_load(load_kw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Soft floor + stochastic soft ceiling (ref peak × U[1.0, 1.10]); no fixed upper bound."""
    load = np.asarray(load_kw, dtype=float)
    floor_deficit = np.maximum(LOAD_FLOOR_KW - load, 0.0)
    load = load + floor_deficit / (1.0 + floor_deficit / 140.0)
    stoch_ceiling = LOAD_REF_PEAK_KW * (1.0 + rng.uniform(0.0, 0.10, len(load)))
    excess = np.maximum(load - stoch_ceiling, 0.0)
    load = load - excess / (1.0 + excess / 120.0)
    return load


def _day_range_mask(day_of_year: np.ndarray, start: int, end: int) -> np.ndarray:
    return (day_of_year >= start) & (day_of_year <= end)


def _spring_trough_overhaul_window(
    load_kw: np.ndarray,
    day_of_year: np.ndarray,
    length: int = MAINTENANCE.stirling_major_overhaul_days,
    search: tuple[int, int] = MAINTENANCE.spring_trough_search,
) -> tuple[int, int]:
    """Pick a contiguous spring load-trough window for the annual Stirling overhaul."""
    daily: dict[int, float] = {}
    for day in np.unique(day_of_year):
        mask = day_of_year == day
        if mask.any():
            daily[int(day)] = float(np.mean(load_kw[mask]))
    lo, hi = search
    best_start = lo
    best_mean = float("inf")
    for start in range(lo, hi - length + 2):
        end = start + length - 1
        window_mean = float(np.mean([daily.get(d, np.inf) for d in range(start, end + 1)]))
        if window_mean < best_mean:
            best_mean = window_mean
            best_start = start
    return best_start, best_start + length - 1


def _stirling_scattered_outage_days(
    rng: np.random.Generator,
    blocked_days: set[int],
    n_days: int = 365,
) -> set[int]:
    """Three scattered 1–2 day minor-outage windows, avoiding the major-overhaul block."""
    outage_days: set[int] = set()
    min_len, max_len = MAINTENANCE.stirling_minor_outage_length_days
    events = 0
    attempts = 0
    while events < MAINTENANCE.stirling_minor_outage_count and attempts < 800:
        attempts += 1
        length = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, n_days - length + 1))
        block = set(range(start, start + length))
        if block & blocked_days or block & outage_days:
            continue
        outage_days |= block
        events += 1
    return outage_days


def _apply_equipment_maintenance(
    day_of_year: np.ndarray,
    whr_kw: np.ndarray,
    pv_kw: np.ndarray,
    rng: np.random.Generator,
    stirling_major_overhaul: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero Stirling/PV during maintenance; flag BESS inspection days for dispatch."""
    whr = np.asarray(whr_kw, dtype=float).copy()
    pv = np.asarray(pv_kw, dtype=float).copy()
    n_days = int(day_of_year.max()) + 1 if len(day_of_year) else 365

    major_lo, major_hi = stirling_major_overhaul
    stirling_major = _day_range_mask(day_of_year, major_lo, major_hi)
    blocked = set(range(major_lo, major_hi + 1))
    scattered_days = _stirling_scattered_outage_days(rng, blocked, n_days=n_days)
    stirling_minor = np.isin(day_of_year, sorted(scattered_days))
    stirling_outage = stirling_major | stirling_minor
    whr[stirling_outage] = 0.0

    pv_lo, pv_hi = MAINTENANCE.pv_maintenance
    pv_outage = _day_range_mask(day_of_year, pv_lo, pv_hi)
    pv[pv_outage] = 0.0

    bess_lo, bess_hi = MAINTENANCE.bess_inspection
    bess_outage = _day_range_mask(day_of_year, bess_lo, bess_hi)

    return whr, pv, stirling_outage, pv_outage, bess_outage


def _whr_peak_cap_kw(n: int, rng: np.random.Generator) -> np.ndarray:
    """Hourly Stirling fleet peak with uniform ±whr_peak_jitter_fraction variation."""
    jitter = rng.uniform(-CASE.whr_peak_jitter_fraction, CASE.whr_peak_jitter_fraction, n)
    return CASE.whr_hard_cap_kw * (1.0 + jitter)


def _intermittent_whr_kw(
    manufacturing_kw: np.ndarray,
    hvac_steam_kw: np.ndarray,
    day_of_year: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load-coupled WHR: process heat index only — no output floor, no cap-fill loop."""
    heat_index = _process_heat_index(manufacturing_kw, hvac_steam_kw)
    hi = np.clip(heat_index, 0.0, 1.0)
    noise = 1.0 + rng.normal(0.0, 0.022, len(heat_index))
    peak_cap = _whr_peak_cap_kw(len(heat_index), rng)
    whr = np.power(hi, 0.88) * CASE.whr_hard_cap_kw * noise
    scale = CASE.whr_target_mean_kw / max(float(whr.mean()), 1.0)
    whr = np.minimum(whr * scale, peak_cap)
    return whr, heat_index, peak_cap


def _finalize_load_components(
    components: pd.DataFrame,
    ar_noise: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Scale six load components so their sum matches the finalized campus total."""
    base = components[list(LOAD_COMPONENT_COLUMNS)].to_numpy(float)
    base_sum = np.sum(base, axis=1, keepdims=True)
    weights = base / np.maximum(base_sum, 1.0)
    raw_parts = base + weights * ar_noise[:, None]
    raw_total = np.sum(raw_parts, axis=1)
    final_total = _finalize_campus_load(raw_total, rng)
    scale = (final_total / np.maximum(raw_total, 1.0))[:, None]
    final_parts = raw_parts * scale
    out = components.copy()
    for idx, col in enumerate(LOAD_COMPONENT_COLUMNS):
        out[col] = final_parts[:, idx]
    out["load_kw"] = final_total
    out["load_hvac_steam_kw"] = out["load_hvac_heating_kw"] + out["load_hvac_cooling_kw"]
    return out


def _campus_load_components(
    hour_of_day: np.ndarray,
    day_of_year: np.ndarray,
    weekday: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    n = len(hour_of_day)
    is_weekday = weekday < 5
    daylight = np.maximum(0.0, np.sin(np.pi * (hour_of_day - 6) / 12))
    seasonal = np.sin(2 * np.pi * (day_of_year - 15) / 365)
    winter_heating = np.maximum(0.0, -seasonal)
    summer_cooling = np.maximum(0.0, np.sin(2 * np.pi * (day_of_year - 172) / 365))

    ar_noise = np.zeros(n)
    innovations = rng.normal(0.0, 26.0, n)
    for i in range(1, n):
        ar_noise[i] = 0.58 * ar_noise[i - 1] + innovations[i]
    event_spikes = rng.random(n) < 0.0018
    ar_noise += event_spikes * rng.uniform(55.0, 140.0, n)

    dc_jitter = rng.normal(0.0, 8.0, n) + 0.04 * ar_noise
    office_campus = CASE.load_office_campus_kw * (
        0.95 + 0.05 * is_weekday + 0.02 * daylight + dc_jitter / CASE.load_office_campus_kw
    )

    is_weekend = weekday >= 5
    is_night = (hour_of_day < 7) | (hour_of_day >= 21)
    shift = np.where(
        is_weekend & is_night,
        0.05,
        np.where(
            is_weekend,
            0.18,
            np.where((hour_of_day >= 7) & (hour_of_day < 21), 1.0, 0.30),
        ),
    )
    line_noise = rng.normal(0.0, 0.07, n) + 0.05 * np.sin(2 * np.pi * hour_of_day / 17.3 + rng.uniform(0, 2 * np.pi))
    downtime = rng.random(n) < 0.0025
    manufacturing = CASE.load_manufacturing_kw * shift * (1.0 + line_noise) * (1.0 - 0.22 * downtime)

    hvac_jitter = rng.normal(0.0, 0.06, n)
    hvac_base = (
        0.38
        + 0.62 * winter_heating
        + 0.78 * summer_cooling
        + 0.14 * daylight * (winter_heating + summer_cooling)
    )
    hvac = CASE.load_hvac_steam_kw * (hvac_base + hvac_jitter)
    heat_term = 0.62 * winter_heating + 0.14 * daylight * winter_heating + 0.19
    heat_frac = np.clip(heat_term / np.maximum(hvac_base, 0.15), 0.15, 0.85)
    load_hvac_heating = hvac * heat_frac
    load_hvac_cooling = hvac - load_hvac_heating

    refrigeration = CASE.load_refrigeration_kw * (
        0.86 + 0.14 * daylight + 0.10 * summer_cooling + rng.normal(0.0, 0.05, n)
    )

    warehouse = CASE.load_warehouse_aux_kw * (
        0.52
        + 0.38 * is_weekday * daylight
        + 0.08 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
        + rng.normal(0.0, 0.06, n)
    )

    components = pd.DataFrame(
        {
            "load_manufacturing_kw": manufacturing,
            "load_hvac_heating_kw": load_hvac_heating,
            "load_hvac_cooling_kw": load_hvac_cooling,
            "load_refrigeration_kw": refrigeration,
            "load_office_campus_kw": office_campus,
            "load_warehouse_aux_kw": warehouse,
        }
    )
    return components, ar_noise


def _process_heat_index(
    manufacturing_kw: np.ndarray,
    hvac_steam_kw: np.ndarray,
) -> np.ndarray:
    """0–1 thermal activity index: manufacturing shift + lagged plant steam."""
    n = len(manufacturing_kw)
    mfg_on = np.clip(manufacturing_kw / CASE.load_manufacturing_kw, 0.0, 1.15)
    thermal = np.zeros(n)
    for i in range(n):
        target = float(mfg_on[i])
        if i == 0:
            thermal[i] = target
        else:
            if target >= thermal[i - 1]:
                tau = 0.22
            elif target < 0.22:
                tau = 0.11
            else:
                tau = 0.07
            thermal[i] = (1.0 - tau) * thermal[i - 1] + tau * target
    steam_on = np.clip(hvac_steam_kw / CASE.load_hvac_steam_kw, 0.0, 1.4)
    steam_coupling = np.clip(thermal, 0.0, 1.0)
    return np.clip(0.86 * thermal + 0.14 * steam_on * steam_coupling, 0.0, 1.0)


def _flue_temperature_k(heat_index: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flue temperature follows process heat index (cold when furnaces off)."""
    span = YANMAR.temp_max_c - YANMAR.temp_min_c
    t_c = YANMAR.temp_min_c + span * np.clip(heat_index, 0.0, 1.0) + rng.normal(0.0, 7.0, len(heat_index))
    return temp_c_to_k(np.clip(t_c, YANMAR.temp_min_c, YANMAR.temp_max_c))


def _campus_pv_kw(
    hour_of_day: np.ndarray,
    day_of_year: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    daylight = np.maximum(0.0, np.sin(np.pi * (hour_of_day - 6) / 12))
    seasonal = 0.42 + 0.58 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    weather = seasonal + 0.10 * np.sin(2 * np.pi * day_of_year / 11 + 0.4) + rng.normal(0, 0.04, len(hour_of_day))
    cloud_pulses = rng.random(len(hour_of_day)) < 0.04
    weather = weather - cloud_pulses * rng.uniform(0.20, 0.45, len(hour_of_day))
    weather_clip = np.clip(weather, 0.0, 1.0)
    clear_pv = CASE.pv_ac_peak_kw * daylight * weather_clip
    is_daytime = daylight > 0.05
    is_overcast = weather_clip < 0.35
    diffuse_frac = np.where(is_daytime & is_overcast, rng.uniform(0.05, 0.15, len(hour_of_day)), 0.0)
    diffuse_pv = CASE.pv_ac_peak_kw * daylight * diffuse_frac
    pv = np.maximum(clear_pv, diffuse_pv)
    target_annual_kwh = CASE.pv_dc_capacity_kwp * 8760.0 * CASE.pv_target_capacity_factor
    scale = target_annual_kwh / max(float(np.sum(pv)), 1.0)
    return np.clip(pv * scale, 0.0, CASE.pv_ac_peak_kw)


def campus_whr_profiles(n_hours: int = 24 * 365, random_state: int = 11) -> pd.DataFrame:
    """Default campus series: named public releases (UKPN / Estates / PVGIS / …).

    Falls back to ``synthetic_whr_profiles`` only if public fetch/cache fails.
    ``n_hours`` other than 8760 and ``random_state`` apply only to the synthetic fallback.
    """
    if n_hours == 24 * 365:
        try:
            from .public_campus_profiles import public_whr_profiles

            return public_whr_profiles()
        except Exception as exc:  # noqa: BLE001 — keep pipeline runnable offline
            import warnings

            warnings.warn(
                f"public_whr_profiles failed ({exc!r}); falling back to synthetic_whr_profiles",
                RuntimeWarning,
                stacklevel=2,
            )
    return synthetic_whr_profiles(n_hours=n_hours, random_state=random_state)


def synthetic_whr_profiles(n_hours: int = 24 * 365, random_state: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    hour = np.arange(n_hours, dtype=float)
    hour_of_day = hour % 24
    day = (hour // 24).astype(int)
    day_of_year = day % 365
    weekday = day % 7

    components, ar_noise = _campus_load_components(hour_of_day, day_of_year, weekday, rng)
    allocated = _finalize_load_components(components, ar_noise, rng)
    manufacturing = allocated["load_manufacturing_kw"].to_numpy(float)
    hvac = allocated["load_hvac_steam_kw"].to_numpy(float)
    pv_kw = _campus_pv_kw(hour_of_day, day_of_year, rng)
    whr_kw, heat_index, whr_peak_cap_kw = _intermittent_whr_kw(manufacturing, hvac, day_of_year, rng)
    load_kw = allocated["load_kw"].to_numpy(float)
    major_overhaul = _spring_trough_overhaul_window(load_kw, day_of_year)
    maint_rng = np.random.default_rng(random_state + 37)
    whr_kw, pv_kw, stirling_outage, pv_outage, bess_outage = _apply_equipment_maintenance(
        day_of_year, whr_kw, pv_kw, maint_rng, major_overhaul
    )
    net_load = load_kw - whr_kw - pv_kw
    import_setpoint = float(np.mean(net_load))

    return pd.DataFrame(
        {
            "hour": hour,
            "hour_of_day": hour_of_day,
            "day": day_of_year,
            "pv_kw": pv_kw,
            "load_kw": load_kw,
            "whr_kw": whr_kw,
            "whr_peak_cap_kw": whr_peak_cap_kw,
            "heat_index": heat_index,
            "net_load_kw": net_load,
            "import_setpoint_kw": import_setpoint,
            "stirling_major_overhaul_start": major_overhaul[0],
            "stirling_major_overhaul_end": major_overhaul[1],
            "stirling_outage": stirling_outage,
            "pv_outage": pv_outage,
            "bess_outage": bess_outage,
            "source_temperature": _flue_temperature_k(heat_index, rng),
            "load_manufacturing_kw": allocated["load_manufacturing_kw"],
            "load_hvac_heating_kw": allocated["load_hvac_heating_kw"],
            "load_hvac_cooling_kw": allocated["load_hvac_cooling_kw"],
            "load_hvac_steam_kw": allocated["load_hvac_steam_kw"],
            "load_refrigeration_kw": allocated["load_refrigeration_kw"],
            "load_office_campus_kw": allocated["load_office_campus_kw"],
            "load_warehouse_aux_kw": allocated["load_warehouse_aux_kw"],
        }
    )


def synthetic_whr_daily_profiles(n_days: int = 365, random_state: int = 17) -> pd.DataFrame:
    hourly = campus_whr_profiles(n_hours=n_days * 24, random_state=random_state)
    midday = hourly[hourly["hour_of_day"] == 12].copy().reset_index(drop=True)
    midday["day"] = np.arange(n_days)
    midday["hour"] = midday["day"] * 24 + 12
    return midday[
        [
            "hour",
            "hour_of_day",
            "day",
            "pv_kw",
            "load_kw",
            "whr_kw",
            "import_setpoint_kw",
            "source_temperature",
            "load_manufacturing_kw",
            "load_hvac_heating_kw",
            "load_hvac_cooling_kw",
            "load_hvac_steam_kw",
            "load_refrigeration_kw",
            "load_office_campus_kw",
            "load_warehouse_aux_kw",
        ]
    ]


def merit_order_dispatch(
    profiles: pd.DataFrame,
    storage_capacity_kwh: float = CASE.storage_capacity_kwh,
    storage_power_limit_kw: float = CASE.storage_power_limit_kw,
    import_setpoint_kw: float | None = None,
    initial_soc_fraction: float = CASE.storage_initial_soc_fraction,
) -> pd.DataFrame:
    """Import-setpoint BESS: discharge if net > P_target, charge if net < P_target (grid valley-fill)."""
    load = profiles["load_kw"].to_numpy(float)
    pv = profiles["pv_kw"].to_numpy(float)
    whr = profiles["whr_kw"].to_numpy(float) if "whr_kw" in profiles.columns else np.zeros_like(load)
    bess_outage = (
        profiles["bess_outage"].to_numpy(bool)
        if "bess_outage" in profiles.columns
        else np.zeros(len(load), dtype=bool)
    )
    day_arr = (
        profiles["day"].to_numpy(int)
        if "day" in profiles.columns
        else (profiles["hour"].to_numpy() // 24).astype(int) % 365
    )
    hour_of_day = (
        profiles["hour_of_day"].to_numpy(int)
        if "hour_of_day" in profiles.columns
        else (profiles["hour"].to_numpy() % 24).astype(int)
    )
    daily_setpoint = profiles.groupby("day", as_index=True)["net_load_kw"].mean().to_dict()

    if import_setpoint_kw is None:
        if "import_setpoint_kw" in profiles.columns:
            import_setpoint_kw = float(profiles["import_setpoint_kw"].iloc[0])
        else:
            import_setpoint_kw = float(np.mean(load - whr - pv))

    soc_min = CASE.storage_soc_min_fraction * storage_capacity_kwh
    soc_max = CASE.storage_soc_max_fraction * storage_capacity_kwh
    soc = initial_soc_fraction * storage_capacity_kwh
    p_target = float(import_setpoint_kw)

    storage_power: list[float] = []
    soc_trace: list[float] = []
    grid_power: list[float] = []

    prev_locked = False
    for load_kw, pv_kw, whr_kw, locked, day_i, hod in zip(
        load, pv, whr, bess_outage, day_arr, hour_of_day, strict=True
    ):
        if prev_locked and not locked:
            soc = initial_soc_fraction * storage_capacity_kwh
        prev_locked = bool(locked)

        net = load_kw - whr_kw - pv_kw
        batt = 0.0

        if locked:
            grid_power.append(net)
            storage_power.append(0.0)
            soc_trace.append(soc)
            continue

        if day_i in BESS_ZOOM_CYCLING_DAYS and hod == 0:
            soc = max(soc, initial_soc_fraction * storage_capacity_kwh)
        p_hour = float(daily_setpoint[day_i]) if day_i in BESS_ZOOM_CYCLING_DAYS else p_target

        if net > p_hour:
            discharge = min(net - p_hour, storage_power_limit_kw, soc - soc_min)
            batt = max(float(discharge), 0.0)
            soc -= batt
        elif net < p_hour:
            charge = min(p_hour - net, storage_power_limit_kw, soc_max - soc)
            batt = -max(float(charge), 0.0)
            soc += -batt

        grid_power.append(net - batt)
        storage_power.append(batt)
        soc_trace.append(soc)

    dispatch = profiles.copy()
    dispatch["whr_kw"] = whr
    dispatch["import_setpoint_kw"] = p_target
    dispatch["storage_power_kw"] = storage_power
    dispatch["storage_soc_kwh"] = soc_trace
    dispatch["grid_power_kw"] = grid_power
    dispatch["net_load_kw"] = load - whr - pv
    dispatch["net_balance_kw"] = pv + whr - load
    dispatch["served_balance_kw"] = pv + whr + np.array(storage_power) - load
    dispatch["whr_uncertainty_kw"] = 0.0
    dispatch["efficiency"] = 0.0
    dispatch["selected_pressure"] = YANMAR.pressure_rated_kpa
    dispatch["selected_speed"] = YANMAR.speed_rated_rpm
    return dispatch


def _longest_load_plateau_hours(load: np.ndarray, tolerance_kw: float = 2.0) -> int:
    """Longest run where all samples stay within ±tolerance_kw of each other."""
    if len(load) == 0:
        return 0
    longest = 1
    start = 0
    for i in range(1, len(load) + 1):
        segment = load[start:i]
        if i == len(load) or float(segment.max() - segment.min()) > 2.0 * tolerance_kw:
            run_len = i - start
            if run_len >= 3 and float(segment.max() - segment.min()) <= 2.0 * tolerance_kw:
                longest = max(longest, run_len)
            start = i
    return longest


def _max_load_plateau_in_windows(load: np.ndarray, window_hours: int = 168) -> int:
    max_plateau = 0
    for start in range(0, len(load) - window_hours + 1, 24):
        max_plateau = max(max_plateau, _longest_load_plateau_hours(load[start : start + window_hours]))
    return max_plateau


def _whr_seasonal_swing_kw(whr: np.ndarray, day: np.ndarray, hour_of_day: np.ndarray) -> float:
    """Daytime mean WHR: peak season minus shoulder season."""
    daytime = (hour_of_day >= 8) & (hour_of_day <= 18)
    winter = daytime & ((day >= 300) | (day <= 50))
    summer = daytime & (day >= 150) & (day <= 240)
    shoulder = daytime & (day >= 60) & (day <= 120)
    if not shoulder.any():
        return 0.0
    peak = max(float(np.mean(whr[winter])) if winter.any() else 0.0, float(np.mean(whr[summer])) if summer.any() else 0.0)
    return peak - float(np.mean(whr[shoulder]))


def _whr_cap_plateau_hours(whr: np.ndarray, cap_kw: float = CASE.whr_hard_cap_kw, tol: float = 3.0) -> int:
    near_cap = whr >= (cap_kw - tol)
    longest = 0
    run = 0
    for hit in near_cap:
        if hit:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def _stirling_monthly_swing_kw(whr: np.ndarray, hour: np.ndarray) -> float:
    """Max monthly mean minus min monthly mean."""
    month = ((hour / 24.0) // 30.44).astype(int) % 12
    monthly = [float(np.mean(whr[month == m])) for m in range(12)]
    return float(max(monthly) - min(monthly))


def hard_acceptance_checks(profiles: pd.DataFrame, dispatch: pd.DataFrame) -> dict[str, float | bool | str]:
    """Hard acceptance criteria for BESS cycling (import-setpoint controller)."""
    batt = dispatch["storage_power_kw"].to_numpy(float)
    soc = dispatch["storage_soc_kwh"].to_numpy(float)
    load = dispatch["load_kw"].to_numpy(float)
    pv = dispatch["pv_kw"].to_numpy(float)
    hour = dispatch["hour"].to_numpy(float)
    month = ((hour / 24.0) // 30.44).astype(int) % 12

    autumn = dispatch[(dispatch["day"] >= 250) & (dispatch["day"] <= 350)]
    autumn_std = float(autumn["storage_power_kw"].std()) if len(autumn) else 0.0

    soc_lo = CASE.storage_soc_min_fraction * CASE.storage_capacity_kwh
    soc_stuck_low_months = 0
    monthly_ok = True
    for m in range(12):
        mask = month == m
        bm = batt[mask]
        sm = soc[mask]
        has_charge = bool(np.any(bm < -10.0))
        has_discharge = bool(np.any(bm > 10.0))
        if not (has_charge and has_discharge):
            monthly_ok = False
        if len(sm) and float(sm.max() - sm.min()) < 80.0:
            soc_stuck_low_months += 1

    corr_load = float(np.corrcoef(np.abs(batt), load)[0, 1]) if np.std(batt) > 1 else 0.0
    corr_pv = float(np.corrcoef(np.abs(batt), pv)[0, 1]) if np.std(batt) > 1 else 0.0

    flags: list[str] = []
    if autumn_std <= 50.0:
        flags.append("autumn_storage_std")
    if not monthly_ok:
        flags.append("monthly_cycles")
    if float(soc.min()) <= soc_lo + 1.0 and soc_stuck_low_months >= 3:
        flags.append("soc_stuck_low")
    if abs(corr_pv) > abs(corr_load) + 0.03:
        flags.append("bess_pv_coupled")

    return {
        "autumn_day250_350_storage_std_kw": autumn_std,
        "import_setpoint_kw": float(dispatch["import_setpoint_kw"].iloc[0]),
        "storage_power_annual_std_kw": float(np.std(batt)),
        "bess_corr_abs_power_with_load": corr_load,
        "bess_corr_abs_power_with_pv": corr_pv,
        "soc_min_kwh": float(soc.min()),
        "soc_max_kwh": float(soc.max()),
        "whr_min_kw": float(dispatch["whr_kw"].min()),
        "whr_max_kw": float(dispatch["whr_kw"].max()),
        "load_max_kw": float(load.max()),
        "passed": len(flags) == 0,
        "flags": ", ".join(flags) if flags else "ok",
    }


def sanity_check_microgrid(
    profiles: pd.DataFrame,
    dispatch: pd.DataFrame | None = None,
) -> dict[str, float | bool | str]:
    load = profiles["load_kw"].to_numpy(float)
    pv = profiles["pv_kw"].to_numpy(float)
    whr_profile = profiles["whr_kw"].to_numpy(float) if "whr_kw" in profiles.columns else None
    load_mean = float(load.mean())
    pv_cf = float(np.sum(pv)) / (CASE.pv_dc_capacity_kwp * len(pv))

    checks: dict[str, float | bool | str] = {
        "fleet_nameplate_kw": CASE.fleet_nameplate_kw,
        "whr_hard_cap_kw": CASE.whr_hard_cap_kw,
        "pv_capacity_factor": pv_cf,
        "load_mean_kw": load_mean,
    }
    if whr_profile is not None:
        checks["whr_profile_mean_kw"] = float(whr_profile.mean())
        checks["whr_profile_min_kw"] = float(whr_profile.min())
        checks["whr_profile_max_kw"] = float(whr_profile.max())

    flags: list[str] = []
    if not (0.09 <= pv_cf <= 0.12):
        flags.append("pv_capacity_factor")
    if whr_profile is not None and float(whr_profile.max()) > CASE.whr_hard_cap_kw + 0.5:
        flags.append("whr_over_cap")

    if dispatch is not None:
        hard = hard_acceptance_checks(profiles, dispatch)
        checks.update(hard)
        if not hard["passed"]:
            flags.extend(str(hard["flags"]).split(", "))

    checks["passed"] = len(flags) == 0
    checks["flags"] = ", ".join(flags) if flags else "ok"
    return checks


def confidence_aware_dispatch(
    model,
    profiles: pd.DataFrame,
    n_units: int = CASE.stirling_units,
    storage_capacity_kwh: float = CASE.storage_capacity_kwh,
    storage_power_limit_kw: float = CASE.storage_power_limit_kw,
    commitment_window_days: int = 1,
) -> pd.DataFrame:
    from microgrid_dispatch import confidence_aware_dispatch as base_dispatch

    dispatch = base_dispatch(
        model,
        profiles,
        n_engines=n_units,
        storage_capacity_kwh=storage_capacity_kwh,
        storage_power_limit_kw=storage_power_limit_kw,
        commitment_window_days=commitment_window_days,
    )
    if "whr_peak_cap_kw" in profiles.columns:
        peak_cap = profiles["whr_peak_cap_kw"].to_numpy(float)
    else:
        peak_cap = np.full(len(profiles), CASE.whr_hard_cap_kw)
    whr = np.minimum(dispatch["whr_kw"].to_numpy(float), peak_cap)
    dispatch["whr_kw"] = whr
    dispatch["net_balance_kw"] = dispatch["pv_kw"] + dispatch["whr_kw"] - dispatch["load_kw"]
    return dispatch


def case_study_metadata() -> dict:
    meta = asdict(CASE)
    meta["fleet_nameplate_kw"] = CASE.fleet_nameplate_kw
    meta["nominal_campus_load_kw"] = CASE.nominal_campus_load_kw
    meta["heat_source_class"] = "shift-intermittent (manufacturing + plant steam; not office/IT supply)"
    return meta
