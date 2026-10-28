from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loader import FEATURE_COLUMNS, TARGET_COLUMNS
from physics_constraints import PhysicsConstraintAgent


@dataclass
class CandidateProposerAgent:
    """Agent that proposes feasible but denser operating cases."""

    n_candidates: int = 2500
    random_state: int = 7

    def propose(self, reference: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        bounds = {
            col: (float(reference[col].min()), float(reference[col].max()))
            for col in FEATURE_COLUMNS
        }
        samples = {
            col: rng.uniform(low, high, self.n_candidates)
            for col, (low, high) in bounds.items()
        }
        candidates = pd.DataFrame(samples)

        # Add a structured lattice so the selected points remain easy to plot and audit.
        lattice = pd.DataFrame(
            [
                (t, p, s)
                for t in np.linspace(*bounds["source_temperature"], 9)
                for p in np.linspace(*bounds["mean_pressure"], 13)
                for s in np.linspace(*bounds["engine_speed"], 17)
            ],
            columns=FEATURE_COLUMNS,
        )
        return pd.concat([candidates, lattice], ignore_index=True)


class CandidateCriticAgent:
    """Agent that scores cases using sparsity and model uncertainty."""

    def __init__(self, uncertainty_weight: float = 0.55, sparsity_weight: float = 0.45) -> None:
        self.uncertainty_weight = uncertainty_weight
        self.sparsity_weight = sparsity_weight

    def select(
        self,
        candidates: pd.DataFrame,
        reference: pd.DataFrame,
        model,
        n_select: int = 48,
    ) -> pd.DataFrame:
        x_ref = reference[FEATURE_COLUMNS].to_numpy(float)
        x_can = candidates[FEATURE_COLUMNS].to_numpy(float)
        mean = x_ref.mean(axis=0)
        std = np.where(x_ref.std(axis=0) == 0, 1.0, x_ref.std(axis=0))
        xr = (x_ref - mean) / std
        xc = (x_can - mean) / std

        distances = np.sqrt(((xc[:, None, :] - xr[None, :, :]) ** 2).sum(axis=2))
        sparsity = distances.min(axis=1)
        sparsity = _normalize(sparsity)

        _, pred_std = model.predict(x_can, return_std=True)
        uncertainty = _normalize(pred_std[:, 0]) + _normalize(pred_std[:, 1])
        uncertainty = _normalize(uncertainty)

        score = self.sparsity_weight * sparsity + self.uncertainty_weight * uncertainty
        chosen = candidates.copy()
        chosen["agent_score"] = score
        chosen["sparsity_score"] = sparsity
        chosen["uncertainty_score"] = uncertainty
        return self._diverse_top_cases(chosen, reference, n_select=n_select)

    def _diverse_top_cases(self, scored: pd.DataFrame, reference: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select high-score cases while avoiding a single boundary dominating the figure."""
        speed_bins = np.linspace(reference["engine_speed"].min(), reference["engine_speed"].max(), 7)
        pressure_bins = np.linspace(reference["mean_pressure"].min(), reference["mean_pressure"].max(), 6)
        temp_bins = np.linspace(reference["source_temperature"].min(), reference["source_temperature"].max(), 4)
        ranked = scored.sort_values("agent_score", ascending=False).copy()
        ranked["speed_bin"] = np.digitize(ranked["engine_speed"], speed_bins, right=True)
        ranked["pressure_bin"] = np.digitize(ranked["mean_pressure"], pressure_bins, right=True)
        ranked["temp_bin"] = np.digitize(ranked["source_temperature"], temp_bins, right=True)

        selected_rows = []
        bin_counts: dict[tuple[int, int, int], int] = {}
        speed_counts: dict[int, int] = {}
        pressure_counts: dict[int, int] = {}
        for _, row in ranked.iterrows():
            key = (int(row["temp_bin"]), int(row["pressure_bin"]), int(row["speed_bin"]))
            speed_key = int(row["speed_bin"])
            pressure_key = int(row["pressure_bin"])
            if bin_counts.get(key, 0) >= 2:
                continue
            if speed_counts.get(speed_key, 0) >= 5:
                continue
            if pressure_counts.get(pressure_key, 0) >= 7:
                continue
            selected_rows.append(row)
            bin_counts[key] = bin_counts.get(key, 0) + 1
            speed_counts[speed_key] = speed_counts.get(speed_key, 0) + 1
            pressure_counts[pressure_key] = pressure_counts.get(pressure_key, 0) + 1
            if len(selected_rows) == n_select:
                break

        if len(selected_rows) < n_select:
            already = {int(row.name) for row in selected_rows}
            for idx, row in ranked.iterrows():
                if int(idx) not in already:
                    selected_rows.append(row)
                if len(selected_rows) == n_select:
                    break

        out = pd.DataFrame(selected_rows).drop(columns=["speed_bin", "pressure_bin", "temp_bin"])
        return out.reset_index(drop=True)


class SolverLabelAgent:
    """Lightweight proxy for a thermodynamic solver.

    The paper can replace this with the full second-order solver. For reproducible
    local experiments, inverse-distance interpolation gives smooth physics-guided labels.
    """

    def __init__(self, physics_agent: PhysicsConstraintAgent | None = None, k_neighbors: int = 12) -> None:
        self.physics_agent = physics_agent or PhysicsConstraintAgent()
        self.k_neighbors = k_neighbors

    def label(self, candidates: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        x_ref = reference[FEATURE_COLUMNS].to_numpy(float)
        y_ref = reference[TARGET_COLUMNS].to_numpy(float)
        x_can = candidates[FEATURE_COLUMNS].to_numpy(float)

        mean = x_ref.mean(axis=0)
        std = np.where(x_ref.std(axis=0) == 0, 1.0, x_ref.std(axis=0))
        xr = (x_ref - mean) / std
        xc = (x_can - mean) / std
        dist = np.sqrt(((xc[:, None, :] - xr[None, :, :]) ** 2).sum(axis=2))
        nn = np.argsort(dist, axis=1)[:, : self.k_neighbors]
        w = 1.0 / (dist[np.arange(dist.shape[0])[:, None], nn] + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        labels = np.sum(y_ref[nn] * w[:, :, None], axis=1)
        labels = self.physics_agent.clip_predictions(labels)

        out = candidates.copy()
        out["power"] = labels[:, 0]
        out["efficiency"] = labels[:, 1]
        out["label_source"] = "proxy_solver"
        return out


def make_proposer(audit_path=None):
    """Kimi K2.6 when a Moonshot key is set; otherwise the seeded lattice."""
    from kimi_proposer import KimiCandidateProposerAgent, kimi_enabled

    if kimi_enabled():
        return KimiCandidateProposerAgent(audit_path=audit_path)
    return CandidateProposerAgent()


def run_multi_agent_augmentation(
    train: pd.DataFrame,
    initial_model,
    n_select: int = 48,
    oracle_reference: pd.DataFrame | None = None,
    physics: PhysicsConstraintAgent | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the active-augmentation loop.

    ``train`` is the (possibly sparse) set the current surrogate was fitted on;
    it drives both the sparsity reference and the proposal bounds. ``oracle_reference``
    is the high-fidelity solver/experiment that labels the selected query points.
    When omitted it falls back to ``train`` for backward compatibility.
    """
    label_reference = train if oracle_reference is None else oracle_reference
    physics = physics or PhysicsConstraintAgent()
    proposer = make_proposer()
    critic = CandidateCriticAgent()
    solver = SolverLabelAgent(physics_agent=physics)

    proposed = proposer.propose(label_reference)
    valid = physics.validate_candidates(proposed)
    selected = critic.select(valid, train, initial_model, n_select=n_select)
    labelled = solver.label(selected, label_reference)
    augmented = pd.concat([train.assign(label_source="experiment"), labelled], ignore_index=True)
    return augmented, labelled


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    if span < 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / span

