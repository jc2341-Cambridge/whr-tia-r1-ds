"""Build the 8760-hour campus series from named public releases.

Sources (cached under ``yanmar/data/public_campus/``):
  - UK Power Networks standard electricity-demand profiles (industrial, commercial)
  - Cambridge University Estates Building Energy Usage Archive (office proxy, 2019)
  - European Commission JRC PVGIS v5.3 hourly PV + T2m (Cambridge reference point)
  - DESNZ FiT load-factor statistics as the annual PV energy-scale anchor (CASE CF)
  - CIBSE TM46 / TM41 as end-use intensity and degree-day method context
  - Yanmar SE220-100 manufacturer heat-source box for Ts mapping

Southampton Estates open endpoints currently return HTTP 401 from this host; the
warehouse / refrigeration diurnal shapes therefore use the UKPN commercial profile
with CIBSE intensity context, which is the same engineering role stated in Section VI.

No random draws: every series is a deterministic transform of the cached public files.
"""
from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .dispatch import (
    CASE,
    LOAD_FLOOR_KW,
    LOAD_REF_PEAK_KW,
    MAINTENANCE,
    _process_heat_index,
    _spring_trough_overhaul_window,
    _day_range_mask,
)
from .engine_specs import YANMAR, temp_c_to_k

def _default_cache_dir() -> Path:
    """Prefer an existing cache (module-local or release ``data/public_campus``)."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "data" / "public_campus",
        here.parents[1] / "data" / "public_campus",  # whr-tia-r1-ds/data when file is code/yanmar/
        here.parents[2] / "data" / "public_campus",
    ]
    for c in candidates:
        if (c / "ukpn_hourly_2019like.csv").exists() or (c / "public_whr_profiles_8760.csv").exists():
            return c
    return candidates[0]


CACHE_DIR = _default_cache_dir()

PVGIS_SERIES = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
CAMBRIDGE_ESTATES_2019 = (
    "https://raw.githubusercontent.com/EECi/Cambridge-Estates-Building-Energy-Archive/"
    "main/building_data/processed_data/UCam_Building_b0/electricity/2019.csv"
)

# Cambridge reference point used in the provenance note / Section VI.
PVGIS_LAT = 52.205
PVGIS_LON = 0.119
PVGIS_YEAR = 2019

# Fixed minor Stirling outage days (deterministic substitute for RNG placement).
STIRLING_MINOR_OUTAGE_DAYS: tuple[tuple[int, int], ...] = (
    (40, 41),
    (200, 201),
    (330, 331),
)

# CIBSE TM41-style base temperatures (°C) for HVAC drive only.
HDD_BASE_C = 15.5
CDD_BASE_C = 18.0

UA = "Mozilla/5.0 (compatible; WHR-TIA-R1-research/1.0)"


def _ensure_cache() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json,text/csv,*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data:
        raise RuntimeError(f"empty HTTP body for {url}")
    return data


def _http_get_json(url: str, timeout: int = 120) -> dict:
    return json.loads(_http_get(url, timeout=timeout).decode("utf-8"))


def fetch_ukpn_hourly(force: bool = False) -> pd.DataFrame:
    """Half-hourly UKPN profiles → hourly mean industrial / commercial percentages."""
    out = _ensure_cache() / "ukpn_hourly_2019like.csv"
    raw_csv = _ensure_cache() / "ukpn_standard_profiles_raw.csv"
    if out.exists() and not force:
        return pd.read_csv(out)

    export_url = (
        "https://ukpowernetworks.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "ukpn-standard-profiles-electricity-demand/exports/csv"
        "?delimiter=%3B&list_separator=%2C&quote_all=false&with_bom=false"
    )
    if force or not raw_csv.exists():
        raw_csv.write_bytes(_http_get(export_url, timeout=300))

    hh = pd.read_csv(raw_csv, sep=";")
    hh["timestamp"] = pd.to_datetime(hh["timestamp"], utc=True)
    hh = hh.sort_values("timestamp").reset_index(drop=True)
    hh["hour_floor"] = hh["timestamp"].dt.floor("h")
    hourly = (
        hh.groupby("hour_floor", as_index=False)[["industrial", "commercial"]]
        .mean()
        .rename(columns={"hour_floor": "timestamp"})
    )
    # Align to 8760 hours starting at the first hour; trim/pad if needed.
    hourly = hourly.iloc[:8760].copy()
    if len(hourly) < 8760:
        raise RuntimeError(f"UKPN hourly series too short: {len(hourly)}")
    hourly["hour"] = np.arange(8760)
    hourly.to_csv(out, index=False)
    return hourly


def fetch_cambridge_office_hourly(force: bool = False) -> pd.DataFrame:
    """Cambridge Estates Building b0 electricity 2019 (kWh/h ≈ average kW)."""
    out = _ensure_cache() / "cambridge_estates_b0_electricity_2019.csv"
    if out.exists() and not force:
        return pd.read_csv(out)

    raw = _http_get(CAMBRIDGE_ESTATES_2019).decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    # Column name in archive: "equipment load [kWh]"
    value_col = [c for c in df.columns if c != "datetime"][0]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    series = pd.to_numeric(df[value_col], errors="coerce").ffill().fillna(0.0)
    if len(series) < 8760:
        raise RuntimeError(f"Cambridge Estates series too short: {len(series)}")
    out_df = pd.DataFrame(
        {
            "hour": np.arange(8760),
            "office_kwh": series.to_numpy(float)[:8760],
        }
    )
    out_df.to_csv(out, index=False)
    return out_df


def fetch_pvgis_hourly(force: bool = False) -> pd.DataFrame:
    """PVGIS v5.3 hourly PV power (W for 1 kWp) and T2m for Cambridge 2019."""
    out = _ensure_cache() / "pvgis_cambridge_2019_hourly.csv"
    if out.exists() and not force:
        return pd.read_csv(out)

    params = {
        "lat": PVGIS_LAT,
        "lon": PVGIS_LON,
        "startyear": PVGIS_YEAR,
        "endyear": PVGIS_YEAR,
        "pvcalculation": 1,
        "peakpower": 1,
        "loss": 14,
        "angle": 35,
        "aspect": 0,
        "outputformat": "json",
    }
    url = PVGIS_SERIES + "?" + urllib.parse.urlencode(params)
    payload = _http_get_json(url, timeout=180)
    records = payload["outputs"]["hourly"]
    df = pd.DataFrame(records)
    # P is W for 1 kWp; T2m °C.
    out_df = pd.DataFrame(
        {
            "hour": np.arange(len(df)),
            "pv_w_per_kwp": pd.to_numeric(df["P"], errors="coerce").fillna(0.0),
            "t2m_c": pd.to_numeric(df["T2m"], errors="coerce").fillna(10.0),
        }
    )
    if len(out_df) != 8760:
        # Leap / non-leap: trim or refuse.
        if len(out_df) > 8760:
            out_df = out_df.iloc[:8760].copy()
            out_df["hour"] = np.arange(8760)
        else:
            raise RuntimeError(f"PVGIS hourly length {len(out_df)} != 8760")
    out_df.to_csv(out, index=False)
    meta = {
        "source": "PVGIS v5.3 seriescalc",
        "lat": PVGIS_LAT,
        "lon": PVGIS_LON,
        "year": PVGIS_YEAR,
        "peakpower_kwp": 1,
        "loss_pct": 14,
        "angle_deg": 35,
        "aspect": 0,
        "url": url,
    }
    (_ensure_cache() / "pvgis_cambridge_2019_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return out_df


def _scale_shape_to_peak(shape: np.ndarray, peak_kw: float) -> np.ndarray:
    peak = float(np.nanmax(shape))
    if peak <= 0:
        return np.full_like(shape, peak_kw, dtype=float)
    return shape.astype(float) / peak * peak_kw


def _finalize_campus_load_deterministic(load_kw: np.ndarray) -> np.ndarray:
    """Soft floor + soft ceiling without RNG (fixed reference peak)."""
    load = np.asarray(load_kw, dtype=float)
    floor_deficit = np.maximum(LOAD_FLOOR_KW - load, 0.0)
    load = load + floor_deficit / (1.0 + floor_deficit / 140.0)
    ceiling = LOAD_REF_PEAK_KW * 1.05
    excess = np.maximum(load - ceiling, 0.0)
    load = load - excess / (1.0 + excess / 120.0)
    return load


def _hvac_from_t2m(
    t2m_c: np.ndarray,
    commercial_shape: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Degree-day style HVAC split, diurnal envelope from UKPN commercial."""
    hdd = np.maximum(0.0, HDD_BASE_C - t2m_c)
    cdd = np.maximum(0.0, t2m_c - CDD_BASE_C)
    diurn = _scale_shape_to_peak(commercial_shape, 1.0)
    diurn = 0.35 + 0.65 * diurn  # keep a base load
    heat_raw = hdd * diurn
    cool_raw = cdd * diurn
    # Allocate CASE HVAC steam peak across heating/cooling by annual energy share.
    heat_sum = float(heat_raw.sum()) + 1e-9
    cool_sum = float(cool_raw.sum()) + 1e-9
    # Target: heating + cooling mean ≈ CASE.load_hvac_steam_kw with seasonal dominance.
    target_mean = CASE.load_hvac_steam_kw
    # Use relative annual shares then scale so peak of sum ≈ CASE HVAC rating.
    heat = heat_raw / heat_sum
    cool = cool_raw / cool_sum
    # Blend: 55% heating / 45% cooling of mean energy (UK temperate campus).
    heat_kw = heat * (0.55 * target_mean * 8760.0)
    cool_kw = cool * (0.45 * target_mean * 8760.0)
    total = heat_kw + cool_kw
    peak = float(total.max())
    if peak > CASE.load_hvac_steam_kw:
        scale = CASE.load_hvac_steam_kw / peak
        heat_kw *= scale
        cool_kw *= scale
    return heat_kw, cool_kw


def _apply_maintenance_deterministic(
    day_of_year: np.ndarray,
    whr_kw: np.ndarray,
    pv_kw: np.ndarray,
    load_kw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    major = _spring_trough_overhaul_window(load_kw, day_of_year)
    major_lo, major_hi = major
    stirling_major = _day_range_mask(day_of_year, major_lo, major_hi)
    minor_days: set[int] = set()
    for lo, hi in STIRLING_MINOR_OUTAGE_DAYS:
        # Skip if overlaps major overhaul.
        if set(range(lo, hi + 1)) & set(range(major_lo, major_hi + 1)):
            continue
        minor_days |= set(range(lo, hi + 1))
    stirling_minor = np.isin(day_of_year, sorted(minor_days))
    stirling_outage = stirling_major | stirling_minor
    whr = np.asarray(whr_kw, dtype=float).copy()
    pv = np.asarray(pv_kw, dtype=float).copy()
    whr[stirling_outage] = 0.0
    pv_lo, pv_hi = MAINTENANCE.pv_maintenance
    pv_outage = _day_range_mask(day_of_year, pv_lo, pv_hi)
    pv[pv_outage] = 0.0
    bess_lo, bess_hi = MAINTENANCE.bess_inspection
    bess_outage = _day_range_mask(day_of_year, bess_lo, bess_hi)
    return whr, pv, stirling_outage, pv_outage, bess_outage, major


def _whr_from_heat_index(heat_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hi = np.clip(heat_index, 0.0, 1.0)
    peak_cap = np.full(len(hi), CASE.whr_hard_cap_kw, dtype=float)
    whr = np.power(hi, 0.88) * CASE.whr_hard_cap_kw
    scale = CASE.whr_target_mean_kw / max(float(whr.mean()), 1.0)
    whr = np.minimum(whr * scale, peak_cap)
    return whr, peak_cap


def _flue_temperature_k_deterministic(heat_index: np.ndarray) -> np.ndarray:
    span = YANMAR.temp_max_c - YANMAR.temp_min_c
    t_c = YANMAR.temp_min_c + span * np.clip(heat_index, 0.0, 1.0)
    return temp_c_to_k(np.clip(t_c, YANMAR.temp_min_c, YANMAR.temp_max_c))


def public_whr_profiles(force_refresh: bool = False) -> pd.DataFrame:
    """8760-hour campus DataFrame with the same columns as synthetic_whr_profiles()."""
    global CACHE_DIR
    CACHE_DIR = _default_cache_dir()
    frozen = CACHE_DIR / "public_whr_profiles_8760.csv"
    if frozen.exists() and not force_refresh:
        return pd.read_csv(frozen)

    ukpn = fetch_ukpn_hourly(force=force_refresh)
    office_raw = fetch_cambridge_office_hourly(force=force_refresh)
    pvgis = fetch_pvgis_hourly(force=force_refresh)

    n = 8760
    hour = np.arange(n, dtype=float)
    hour_of_day = hour % 24
    day_of_year = (hour // 24).astype(int) % 365

    industrial = ukpn["industrial"].to_numpy(float)[:n]
    commercial = ukpn["commercial"].to_numpy(float)[:n]
    t2m = pvgis["t2m_c"].to_numpy(float)[:n]

    # Manufacturing: UKPN industrial, peak-normalised to CASE manufacturing rating.
    manufacturing = _scale_shape_to_peak(industrial, CASE.load_manufacturing_kw)

    # Office: Cambridge Estates 2019 shape → CASE office rating (peak-normalised).
    office_shape = office_raw["office_kwh"].to_numpy(float)[:n]
    office = _scale_shape_to_peak(np.maximum(office_shape, 0.0), CASE.load_office_campus_kw)

    # Warehouse + refrigeration: UKPN commercial shapes (Southampton ODS unavailable).
    warehouse = _scale_shape_to_peak(commercial, CASE.load_warehouse_aux_kw)
    # Refrigeration: commercial envelope with mild summer lift from CDD.
    cdd = np.maximum(0.0, t2m - CDD_BASE_C)
    refrig_shape = commercial * (1.0 + 0.25 * cdd / max(float(cdd.max()), 1.0))
    refrigeration = _scale_shape_to_peak(refrig_shape, CASE.load_refrigeration_kw)

    heat_kw, cool_kw = _hvac_from_t2m(t2m, commercial)
    hvac_steam = heat_kw + cool_kw

    raw_total = manufacturing + office + warehouse + refrigeration + hvac_steam
    load_kw = _finalize_campus_load_deterministic(raw_total)
    # Re-allocate component shares so they sum to finalized load.
    base = np.column_stack(
        [manufacturing, heat_kw, cool_kw, refrigeration, office, warehouse]
    )
    base_sum = np.sum(base, axis=1, keepdims=True)
    weights = base / np.maximum(base_sum, 1.0)
    parts = weights * load_kw[:, None]
    manufacturing = parts[:, 0]
    heat_kw = parts[:, 1]
    cool_kw = parts[:, 2]
    refrigeration = parts[:, 3]
    office = parts[:, 4]
    warehouse = parts[:, 5]
    hvac_steam = heat_kw + cool_kw

    # PV: PVGIS shape × 1.6 MWp, AC clip, then DESNZ-anchored annual CF scale.
    pv = pvgis["pv_w_per_kwp"].to_numpy(float)[:n] / 1000.0 * CASE.pv_dc_capacity_kwp
    pv = np.clip(pv, 0.0, CASE.pv_ac_peak_kw)
    target_annual_kwh = CASE.pv_dc_capacity_kwp * 8760.0 * CASE.pv_target_capacity_factor
    pv_scale = target_annual_kwh / max(float(pv.sum()), 1.0)
    pv = np.clip(pv * pv_scale, 0.0, CASE.pv_ac_peak_kw)

    heat_index = _process_heat_index(manufacturing, hvac_steam)
    whr_kw, whr_peak_cap_kw = _whr_from_heat_index(heat_index)
    whr_kw, pv, stirling_outage, pv_outage, bess_outage, major = _apply_maintenance_deterministic(
        day_of_year, whr_kw, pv, load_kw
    )
    # Recompute heat index availability after outages is handled via stirling_outage flag.
    net_load = load_kw - whr_kw - pv
    import_setpoint = float(np.mean(net_load))
    source_temperature = _flue_temperature_k_deterministic(heat_index)

    prof = pd.DataFrame(
        {
            "hour": hour,
            "hour_of_day": hour_of_day,
            "day": day_of_year,
            "pv_kw": pv,
            "load_kw": load_kw,
            "whr_kw": whr_kw,
            "whr_peak_cap_kw": whr_peak_cap_kw,
            "heat_index": heat_index,
            "net_load_kw": net_load,
            "import_setpoint_kw": import_setpoint,
            "stirling_major_overhaul_start": major[0],
            "stirling_major_overhaul_end": major[1],
            "stirling_outage": stirling_outage,
            "pv_outage": pv_outage,
            "bess_outage": bess_outage,
            "source_temperature": source_temperature,
            "load_manufacturing_kw": manufacturing,
            "load_hvac_heating_kw": heat_kw,
            "load_hvac_cooling_kw": cool_kw,
            "load_hvac_steam_kw": hvac_steam,
            "load_refrigeration_kw": refrigeration,
            "load_office_campus_kw": office,
            "load_warehouse_aux_kw": warehouse,
            "t2m_c": t2m,
        }
    )
    _ensure_cache()
    prof.to_csv(CACHE_DIR / "public_whr_profiles_8760.csv", index=False)
    return prof


def write_frozen_inputs(path: Path | None = None, force_refresh: bool = False) -> Path:
    """Write the full public series and a slim annual_hourly_inputs.csv."""
    prof = public_whr_profiles(force_refresh=force_refresh)
    cache = _ensure_cache()
    full = cache / "public_whr_profiles_8760.csv"
    prof.to_csv(full, index=False)
    slim_path = path or (cache / "annual_hourly_inputs.csv")
    slim_cols = [
        c
        for c in (
            "hour",
            "day",
            "hour_of_day",
            "load_kw",
            "pv_kw",
            "source_temperature",
            "heat_index",
            "whr_kw",
            "stirling_outage",
            "pv_outage",
            "bess_outage",
        )
        if c in prof.columns
    ]
    prof[slim_cols].to_csv(slim_path, index=False)
    readme = cache / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Public campus inputs",
                "",
                "Built by `yanmar.public_campus_profiles.public_whr_profiles()`.",
                "",
                "## Sources",
                "",
                "- UKPN standard profiles: industrial → manufacturing; commercial → warehouse/refrigeration diurnal envelope",
                "- Cambridge Estates Building Energy Archive (b0, 2019) → office shape",
                "- PVGIS v5.3 Cambridge 2019 → PV shape + T2m; annual PV energy scaled to CASE CF (DESNZ-anchored)",
                "- CIBSE TM41 degree-day bases for HVAC split; TM46 intensity context in the manuscript",
                "- Yanmar SE220-100 500–800 °C box for deterministic Ts(h)",
                "",
                "No random seed. Re-fetch with `force_refresh=True`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return slim_path


if __name__ == "__main__":
    p = write_frozen_inputs(force_refresh=False)
    prof = public_whr_profiles()
    print("wrote", p)
    print(
        "load mean/peak",
        float(prof["load_kw"].mean()),
        float(prof["load_kw"].max()),
        "pv CF",
        float(prof["pv_kw"].sum()) / (CASE.pv_dc_capacity_kwp * 8760.0),
        "whr mean",
        float(prof["whr_kw"].mean()),
        "Ts mean C",
        float(prof["source_temperature"].mean() - 273.15),
    )
