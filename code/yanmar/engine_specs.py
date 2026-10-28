"""Yanmar E-Stir SE220-100 equivalent parameters.

Sources:
- Yanmar Technical Review (2022/09): Table 2, SE220-100C/R
- Heat-source range 500-800 C, rated 750 C, 9.9 kW, 25 % efficiency
- Helium working gas, 800 rpm, 2.8 MPa absolute mean pressure
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YanmarEStirSpecs:
    model: str = "SE220-100"
    power_rated_w: float = 9900.0
    efficiency_rated: float = 0.25
    temp_rated_c: float = 750.0
    temp_levels_c: tuple[float, float, float] = (500.0, 650.0, 800.0)
    temp_min_c: float = 500.0
    temp_max_c: float = 800.0
    speed_rated_rpm: float = 800.0
    pressure_rated_kpa: float = 2800.0  # 2.8 MPa absolute
    displacer_cc: float = 2850.0
    power_piston_cc: float = 2470.0
    coolant_temp_k: float = 330.0  # ~57 C inlet, consistent with prior WHR case study
    working_gas: str = "helium"
    gamma: float = 1.667
    # Helium gas constant on a per-engine charge basis (calibrated scale factor).
    r_gas: float = 2080.0


# Operating grid for RK oracle (mirrors GM 4L23 structure: 3 x 5 x 8 = 120 cases).
PRESSURE_LEVELS_KPA: tuple[float, ...] = (1500.0, 2000.0, 2500.0, 2800.0, 3000.0)
SPEED_LEVELS_RPM: tuple[float, ...] = (400.0, 550.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1300.0)

YANMAR = YanmarEStirSpecs()


def temp_c_to_k(t_c: float) -> float:
    return t_c + 273.15


def temp_levels_k() -> tuple[float, float, float]:
    return tuple(temp_c_to_k(t) for t in YANMAR.temp_levels_c)
