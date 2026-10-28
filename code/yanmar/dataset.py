"""Generate and load Yanmar RK-simulated operating grids."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import FEATURE_COLUMNS, TARGET_COLUMNS, sparse_active_learning_split

from .engine_specs import PRESSURE_LEVELS_KPA, SPEED_LEVELS_RPM, YANMAR, temp_c_to_k, temp_levels_k
from .stirling_rk_solver import rk_power_efficiency

YANMAR_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = YANMAR_ROOT / "data" / "yanmar_operating_data.csv"


def generate_operating_grid() -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for t_c in YANMAR.temp_levels_c:
        t_k = temp_c_to_k(t_c)
        for p_kpa in PRESSURE_LEVELS_KPA:
            for rpm in SPEED_LEVELS_RPM:
                power, eff_pct = rk_power_efficiency(t_c, p_kpa, rpm)
                rows.append(
                    {
                        "source_temperature": t_k,
                        "source_temperature_c": t_c,
                        "mean_pressure": p_kpa,
                        "engine_speed": rpm,
                        "power": power,
                        "efficiency_percent": eff_pct,
                        "efficiency": eff_pct / 100.0,
                        "label_source": "rk4_second_order",
                    }
                )
    return pd.DataFrame(rows)


def save_operating_grid(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or DEFAULT_DATA_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_operating_grid()
    df.to_csv(path, index=False)
    return df


def load_operating_data(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or DEFAULT_DATA_PATH)
    if not path.exists():
        save_operating_grid(path)
    df = pd.read_csv(path)
    if "efficiency" not in df.columns and "efficiency_percent" in df.columns:
        df["efficiency"] = df["efficiency_percent"] / 100.0
    return df.sort_values(["source_temperature", "mean_pressure", "engine_speed"]).reset_index(drop=True)


def feature_target_arrays(df: pd.DataFrame):
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMNS].to_numpy(dtype=float)
    return x, y


__all__ = [
    "DEFAULT_DATA_PATH",
    "YANMAR_ROOT",
    "generate_operating_grid",
    "load_operating_data",
    "save_operating_grid",
    "sparse_active_learning_split",
    "temp_levels_k",
    "feature_target_arrays",
]
