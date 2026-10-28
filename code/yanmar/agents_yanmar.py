"""Multi-agent augmentation with RK4 oracle labelling for Yanmar cases."""
from __future__ import annotations

import json

import pandas as pd

from agents import CandidateCriticAgent, SolverLabelAgent, make_proposer
from data_loader import FEATURE_COLUMNS, TARGET_COLUMNS
from kimi_proposer import lattice_pool, tag_query_origin
from physics_constraints import PhysicsConstraintAgent

from .dataset import YANMAR_ROOT
from .engine_specs import YANMAR, temp_c_to_k
from .stirling_rk_solver import rk_power_efficiency


class RKSolverLabelAgent:
    """Labels candidate cases with the calibrated second-order RK4 Stirling oracle."""

    def __init__(self, physics_agent: PhysicsConstraintAgent | None = None) -> None:
        self.physics_agent = physics_agent or PhysicsConstraintAgent()

    def label(self, candidates: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        del reference  # RK oracle is physics-based; reference only bounds feasibility.
        out = candidates.copy()
        powers: list[float] = []
        effs: list[float] = []
        for _, row in out.iterrows():
            t_c = float(row["source_temperature"]) - 273.15
            p_w, eff_pct = rk_power_efficiency(t_c, float(row["mean_pressure"]), float(row["engine_speed"]))
            powers.append(p_w)
            effs.append(eff_pct / 100.0)
        labels = self.physics_agent.clip_predictions(
            __import__("numpy").column_stack([powers, effs])
        )
        out["power"] = labels[:, 0]
        out["efficiency"] = labels[:, 1]
        out["label_source"] = "rk4_oracle"
        return out


def yanmar_physics() -> PhysicsConstraintAgent:
    return PhysicsConstraintAgent(
        temperature_bounds=(temp_c_to_k(YANMAR.temp_min_c), temp_c_to_k(YANMAR.temp_max_c)),
        pressure_bounds=(1500.0, 3000.0),
        speed_bounds=(400.0, 1300.0),
        efficiency_bounds=(0.05, 0.40),
    )


def run_yanmar_augmentation(
    train: pd.DataFrame,
    initial_model,
    n_select: int = 28,
    oracle_reference: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_reference = train if oracle_reference is None else oracle_reference
    physics = yanmar_physics()
    proposer = make_proposer(audit_path=YANMAR_ROOT / "results" / "kimi_proposer_audit.json")
    critic = CandidateCriticAgent()
    solver = RKSolverLabelAgent(physics_agent=physics)

    proposed = proposer.propose(label_reference)
    valid = physics.validate_candidates(proposed)
    selected = critic.select(valid, train, initial_model, n_select=n_select)
    selected = tag_query_origin(selected, lattice_pool(label_reference))
    audit = getattr(proposer, "last_audit", None)
    if isinstance(audit, dict):
        counts = selected["candidate_source"].value_counts().to_dict()
        audit["selected_lattice"] = int(counts.get("lattice_retrieval", 0))
        audit["selected_off_lattice"] = int(counts.get("off_lattice_proposal", 0))
        path = getattr(proposer, "audit_path", None)
        if path is not None:
            path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    labelled = solver.label(selected, label_reference)
    augmented = pd.concat([train.assign(label_source="rk4_grid"), labelled], ignore_index=True)
    return augmented, labelled
