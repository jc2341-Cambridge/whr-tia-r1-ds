from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROPOSER_PROMPT = {
    "system": (
        "You are the OperatingCaseProposer in a physics-guided WHR workflow. "
        "Propose candidate Stirling operating points as JSON. "
        "Never invent power or efficiency labels. "
        "Keep source temperature, mean pressure, and engine speed inside the "
        "bounds supplied by the user. Prefer regions that are simultaneously "
        "sparse and uncertain relative to the labelled seed set. "
        "Return only a JSON object with key 'candidates'."
    ),
    "task": (
        "Given sparse WHR operating data, propose candidate operating regions "
        "that improve coverage without violating device constraints."
    ),
    "hard_rules": [
        "Do not generate power or efficiency labels.",
        "Keep source temperature, pressure, and engine speed inside validated bounds.",
        "Prefer regions that are simultaneously sparse and uncertain.",
        "Return JSON so the physics reviewer can audit every candidate.",
    ],
    "candidate_json_schema": {
        "source_temperature": "float",
        "mean_pressure": "float",
        "engine_speed": "float",
        "reason": "string",
    },
}

LLM_AGENT_PROTOCOL = {
    "system_boundary": (
        "The LLM is not used as a numerical oracle. It proposes operating cases, "
        "explains sparse regions, and emits machine-readable constraints. All "
        "numeric labels are produced by the solver-label agent or experimental data."
    ),
    "reported_proposer": {
        "model": "kimi-k2.6",
        "api": "Moonshot OpenAI-compatible chat completions",
        "temperature": 0.2,
        "seed": 7,
        "thinking": "disabled",
        "response_format": "json_object",
        "implementation": "kimi_proposer.py",
    },
    "agents": [
        {
            "name": "OperatingCaseProposer",
            "role": "Identify sparse or high-curvature regions in the temperature-pressure-speed space.",
            "input": ["clean operating data", "device bounds", "sampling-density summary"],
            "output_schema": {
                "candidate_region": "string",
                "reason": "string",
                "soft_constraints": ["string"],
            },
        },
        {
            "name": "PhysicsConstraintReviewer",
            "role": "Reject infeasible candidate regions before any labels are generated.",
            "input": ["candidate cases", "temperature/pressure/speed limits", "efficiency bounds"],
            "output_schema": {
                "accepted": "boolean",
                "violated_constraints": ["string"],
                "review_comment": "string",
            },
        },
        {
            "name": "UncertaintyCritic",
            "role": "Rank valid cases by epistemic uncertainty and distance from existing samples.",
            "input": ["bootstrap surrogate uncertainty", "nearest-neighbour distance"],
            "output_schema": {
                "agent_score": "float",
                "sparsity_score": "float",
                "uncertainty_score": "float",
            },
        },
        {
            "name": "DispatchExplainer",
            "role": "Translate model confidence into conservative WHR-PV-storage scheduling choices.",
            "input": ["surrogate mean", "surrogate uncertainty", "load/PV/waste-heat profiles"],
            "output_schema": {
                "selected_action": "string",
                "confidence_note": "string",
                "microgrid_rationale": "string",
            },
        },
    ],
}


def build_agent_trace(
    selected_cases: pd.DataFrame,
    metrics: dict,
    out_path: str | Path,
) -> dict:
    """Write a reproducible LLM-agent trace for the methods section and appendix."""
    top_cases = selected_cases.sort_values("agent_score", ascending=False).head(8)
    trace = {
        "protocol": LLM_AGENT_PROTOCOL,
        "prompt_template": PROPOSER_PROMPT,
        "selected_case_summary": [
            {
                "source_temperature": round(float(row["source_temperature"]), 3),
                "mean_pressure": round(float(row["mean_pressure"]), 3),
                "engine_speed": round(float(row["engine_speed"]), 3),
                "agent_score": round(float(row["agent_score"]), 4),
                "sparsity_score": round(float(row["sparsity_score"]), 4),
                "uncertainty_score": round(float(row["uncertainty_score"]), 4),
                "reason": _case_reason(row),
            }
            for _, row in top_cases.iterrows()
        ],
        "numerical_audit": {
            "baseline_power_mape": metrics["baseline"]["power_mape"],
            "baseline_efficiency_mape": metrics["baseline"]["efficiency_mape"],
            "agent_augmented_power_mape": metrics["agent_augmented"]["power_mape"],
            "agent_augmented_efficiency_mape": metrics["agent_augmented"]["efficiency_mape"],
            "interpretation": (
                "The augmentation is used for coverage and confidence-aware operation, "
                "not as a claim that every target metric must improve monotonically."
            ),
        },
    }
    Path(out_path).write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return trace


def _case_reason(row: pd.Series) -> str:
    pressure = float(row["mean_pressure"])
    speed = float(row["engine_speed"])
    if pressure >= 2200 and speed >= 2400:
        return "High-load/high-speed corner with limited experimental redundancy."
    if pressure <= 800 and speed >= 2400:
        return "Low-pressure/high-speed transition region with elevated extrapolation risk."
    if pressure >= 2200 and speed <= 900:
        return "High-pressure/low-speed region useful for separating pressure and speed effects."
    return "Interior sparse region selected by combined uncertainty and sampling-distance score."

