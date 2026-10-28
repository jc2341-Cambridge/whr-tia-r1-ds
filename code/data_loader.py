from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# GM 4L23 Martini manual: the raw txt stores heater-tube temperature in deg F.
HEATER_TEMP_LEVELS_F = (1000.0, 1200.0, 1400.0)


def fahrenheit_to_kelvin(t_f: np.ndarray | pd.Series | float) -> np.ndarray:
    return (np.asarray(t_f, dtype=float) - 32.0) * 5.0 / 9.0 + 273.15


TEMP_LEVELS_K: tuple[float, float, float] = tuple(fahrenheit_to_kelvin(t) for t in HEATER_TEMP_LEVELS_F)
TEMP_K_LO, TEMP_K_MID, TEMP_K_HI = TEMP_LEVELS_K

DATA_COLUMNS = [
    "source_temperature",
    "mean_pressure",
    "engine_speed",
    "power",
    "efficiency_percent",
]
FEATURE_COLUMNS = ["source_temperature", "mean_pressure", "engine_speed"]
TARGET_COLUMNS = ["power", "efficiency"]


def _numbers_from_line(line: str) -> list[float]:
    normalized = line.replace("−", "-").replace("–", "-")
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", normalized)]


def load_raw_operating_data(path: str | Path = ROOT / "jordon源数据.txt") -> pd.DataFrame:
    """Load operating points and keep the latest correction for duplicate inputs."""
    path = Path(path)
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        nums = _numbers_from_line(line)
        if len(nums) == 5:
            rows.append(nums)

    df = pd.DataFrame(rows, columns=DATA_COLUMNS)
    # The text file contains an earlier block and a corrected repeated block.
    # Keeping the last duplicate preserves manual corrections without hard-coding line numbers.
    df = df.drop_duplicates(
        subset=["source_temperature", "mean_pressure", "engine_speed"],
        keep="last",
    ).sort_values(["source_temperature", "mean_pressure", "engine_speed"])
    df["efficiency"] = df["efficiency_percent"] / 100.0
    df["source_temperature"] = fahrenheit_to_kelvin(df["source_temperature"])
    return df.reset_index(drop=True)


def load_prior_prediction_table(path: str | Path = ROOT / "112数据集.txt") -> pd.DataFrame:
    """Load the legacy prediction table, retained only as a baseline reference."""
    path = Path(path)
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        nums = _numbers_from_line(line)
        if len(nums) == 7:
            rows.append(nums)

    cols = [
        "index",
        "source_temperature",
        "mean_pressure",
        "engine_speed",
        "actual_power",
        "legacy_predicted_power",
        "legacy_error_percent",
    ]
    return pd.DataFrame(rows, columns=cols)


def load_external_dataset(path: str | Path = ROOT / "36数据集.txt") -> pd.DataFrame:
    """Load an external small benchmark for optional transfer-style discussion."""
    path = Path(path)
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        nums = _numbers_from_line(line)
        if len(nums) == 6:
            rows.append(nums)

    cols = [
        "index",
        "engine_speed",
        "load",
        "actual_power",
        "reference_prediction",
        "error_percent",
    ]
    return pd.DataFrame(rows, columns=cols)


def train_test_split_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic sparse holdout resembling operating-point interpolation."""
    t1, t2, t3 = TEMP_LEVELS_K
    holdout_mask = (
        (df["source_temperature"].eq(t2) & df["mean_pressure"].eq(500) & df["engine_speed"].eq(2600))
        | (df["source_temperature"].eq(t2) & df["mean_pressure"].eq(1500) & df["engine_speed"].eq(1000))
        | (df["source_temperature"].eq(t2) & df["mean_pressure"].eq(2000) & df["engine_speed"].eq(1400))
        | (df["source_temperature"].eq(t2) & df["mean_pressure"].eq(2500) & df["engine_speed"].eq(1800))
        | (df["source_temperature"].eq(t3) & df["mean_pressure"].eq(500) & df["engine_speed"].eq(2200))
        | (df["source_temperature"].eq(t3) & df["mean_pressure"].eq(1000) & df["engine_speed"].eq(2600))
        | (df["source_temperature"].eq(t1) & df["mean_pressure"].eq(2000) & df["engine_speed"].eq(1800))
        | (df["source_temperature"].eq(t1) & df["mean_pressure"].eq(2500) & df["engine_speed"].eq(2200))
    )
    train = df.loc[~holdout_mask].reset_index(drop=True)
    test = df.loc[holdout_mask].reset_index(drop=True)
    return train, test


def sparse_active_learning_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Active-learning experiment split.

    Returns ``(baseline_train, oracle_pool, holdout)`` where:

    * ``baseline_train`` is a deliberately *sparse* measured subset (coarse
      sub-grid). The data-starved baseline surrogate is fitted on this set, so
      there is genuine room for the agents to improve it.
    * ``oracle_pool`` is every measured case outside the holdout. It plays the
      role of the high-fidelity solver/experiment that can be *queried* at the
      operating points the agents decide to add. The agents choose *where* to
      sample; the numerical labels come from this oracle, never from the LLM.
    * ``holdout`` is a held-out evaluation grid (never used for fitting).
    """
    pressures = sorted(df["mean_pressure"].unique())
    speeds = sorted(df["engine_speed"].unique())
    p_idx = {v: i for i, v in enumerate(pressures)}
    s_idx = {v: i for i, v in enumerate(speeds)}
    pi = df["mean_pressure"].map(p_idx)
    si = df["engine_speed"].map(s_idx)

    # Interior, evenly-spread holdout across every source temperature.
    holdout_mask = pi.isin([1, 3]) & si.isin([1, 4, 6])

    # Sparse coarse-grid baseline on the boundary/mid pressures and speeds,
    # disjoint from the holdout by construction (pressure index sets differ).
    baseline_mask = (~holdout_mask) & pi.isin([0, 2, 4]) & si.isin([0, 3, 7])

    holdout = df.loc[holdout_mask].reset_index(drop=True)
    oracle_pool = df.loc[~holdout_mask].reset_index(drop=True)
    baseline_train = df.loc[baseline_mask].reset_index(drop=True)
    return baseline_train, oracle_pool, holdout


def feature_target_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMNS].to_numpy(dtype=float)
    return x, y

