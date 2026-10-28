# -*- coding: utf-8 -*-
"""Kimi K2.6 proposer for Algorithm 1, line 4.

The language model proposes (Ts, p, ω) records and a short reason. It never
emits power or efficiency. Numerical labels stay with the solver / oracle.

Environment
-----------
MOONSHOT_API_KEY     required for a live run
MOONSHOT_BASE_URL    default https://api.moonshot.ai/v1
                     (China: https://api.moonshot.cn/v1)
WHR_PROPOSER         auto | kimi | offline
                     auto = Kimi if a key is set, else the seeded lattice
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loader import FEATURE_COLUMNS
from llm_protocol import PROPOSER_PROMPT

KIMI_MODEL = "kimi-k2.6"
KIMI_TEMPERATURE = 0.2
KIMI_SEED = 7
KIMI_N_CALLS = 40
KIMI_N_PER_CALL = 28
DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"


def kimi_enabled() -> bool:
    mode = os.environ.get("WHR_PROPOSER", "auto").strip().lower()
    has_key = bool(os.environ.get("MOONSHOT_API_KEY", "").strip())
    if mode == "offline":
        return False
    if mode == "kimi":
        if not has_key:
            raise RuntimeError("WHR_PROPOSER=kimi requires MOONSHOT_API_KEY.")
        return True
    return has_key


def lattice_pool(reference: pd.DataFrame) -> pd.DataFrame:
    """Retrieval memory R / X_ref: a structured feasible lattice (Algorithm 1)."""
    bounds = {
        col: (float(reference[col].min()), float(reference[col].max()))
        for col in FEATURE_COLUMNS
    }
    return pd.DataFrame(
        [
            (t, p, s)
            for t in np.linspace(*bounds["source_temperature"], 9)
            for p in np.linspace(*bounds["mean_pressure"], 13)
            for s in np.linspace(*bounds["engine_speed"], 17)
        ],
        columns=FEATURE_COLUMNS,
    )


def parse_candidate_payload(text: str) -> list[dict[str, Any]]:
    """Extract a JSON candidate list from a Kimi response (fences allowed)."""
    raw = (text or "").strip()
    if not raw:
        return []
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S)
    if fence:
        raw = fence.group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
    rows = payload.get("candidates", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            out.append(
                {
                    "source_temperature": float(row["source_temperature"]),
                    "mean_pressure": float(row["mean_pressure"]),
                    "engine_speed": float(row["engine_speed"]),
                    "reason": str(row.get("reason", "")),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _user_prompt(reference: pd.DataFrame, n_per_call: int) -> str:
    bounds = {
        col: (float(reference[col].min()), float(reference[col].max()))
        for col in FEATURE_COLUMNS
    }
    summary = (
        f"Labelled seed count: {len(reference)}. "
        f"Ts in [{bounds['source_temperature'][0]:.2f}, {bounds['source_temperature'][1]:.2f}] K; "
        f"p in [{bounds['mean_pressure'][0]:.1f}, {bounds['mean_pressure'][1]:.1f}] kPa; "
        f"N in [{bounds['engine_speed'][0]:.1f}, {bounds['engine_speed'][1]:.1f}] rpm. "
        "Propose points in sparse or high-uncertainty regions of this box."
    )
    return (
        f"{summary}\n"
        f"Return exactly {n_per_call} candidates as a JSON object "
        '{"candidates":[{"source_temperature":...,"mean_pressure":...,'
        '"engine_speed":...,"reason":"..."}]} '
        "with no power or efficiency fields."
    )


def _call_kimi(messages: list[dict[str, str]], seed: int) -> tuple[str, dict[str, int]]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["MOONSHOT_API_KEY"],
        base_url=os.environ.get("MOONSHOT_BASE_URL", DEFAULT_BASE_URL),
    )
    resp = client.chat.completions.create(
        model=KIMI_MODEL,
        messages=messages,
        temperature=KIMI_TEMPERATURE,
        seed=seed,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    usage = resp.usage
    tokens = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    content = resp.choices[0].message.content or ""
    return content, tokens


class KimiCandidateProposerAgent:
    """Algorithm 1 line-4 proposer: Kimi proposes; the lattice is retrieval memory."""

    def __init__(
        self,
        n_calls: int = KIMI_N_CALLS,
        n_per_call: int = KIMI_N_PER_CALL,
        random_state: int = KIMI_SEED,
        audit_path: str | Path | None = None,
    ) -> None:
        self.n_calls = n_calls
        self.n_per_call = n_per_call
        self.random_state = random_state
        self.audit_path = Path(audit_path) if audit_path else None
        self.last_audit: dict[str, Any] = {}

    def propose(self, reference: pd.DataFrame) -> pd.DataFrame:
        rag = lattice_pool(reference)
        kimi_rows: list[dict[str, Any]] = []
        n_valid = 0
        n_parse_fail = 0
        prompt_tokens = 0
        completion_tokens = 0
        system = PROPOSER_PROMPT["system"]
        user = _user_prompt(reference, self.n_per_call)
        for i in range(self.n_calls):
            try:
                text, tokens = _call_kimi(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    seed=self.random_state + i,
                )
            except Exception:
                n_parse_fail += 1
                continue
            prompt_tokens += tokens["prompt_tokens"]
            completion_tokens += tokens["completion_tokens"]
            parsed = parse_candidate_payload(text)
            if not parsed:
                n_parse_fail += 1
                continue
            n_valid += 1
            kimi_rows.extend(parsed)

        kimi = pd.DataFrame(kimi_rows, columns=[*FEATURE_COLUMNS, "reason"]) if kimi_rows else pd.DataFrame(
            columns=[*FEATURE_COLUMNS, "reason"]
        )
        proposed = pd.concat([rag.assign(reason="lattice_retrieval"), kimi], ignore_index=True)
        n_exact_lattice = 0
        if len(kimi):
            lat_keys = {
                (round(float(r.source_temperature), 8), round(float(r.mean_pressure), 8), round(float(r.engine_speed), 8))
                for r in rag.itertuples()
            }
            n_exact_lattice = sum(
                1
                for r in kimi.itertuples()
                if (round(float(r.source_temperature), 8), round(float(r.mean_pressure), 8), round(float(r.engine_speed), 8))
                in lat_keys
            )
        self.last_audit = {
            "model": KIMI_MODEL,
            "temperature": KIMI_TEMPERATURE,
            "seed": self.random_state,
            "thinking": "disabled",
            "n_calls": self.n_calls,
            "schema_valid_calls": n_valid,
            "parse_failures": n_parse_fail,
            "kimi_records": int(len(kimi)),
            "kimi_exact_lattice_duplicates": int(n_exact_lattice),
            "lattice_records": int(len(rag)),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            self.audit_path.write_text(json.dumps(self.last_audit, indent=2), encoding="utf-8")
        return proposed


def tag_query_origin(selected: pd.DataFrame, lattice: pd.DataFrame) -> pd.DataFrame:
    """Mark selected queries that sit on the retrieval lattice versus off-lattice."""
    lat = lattice[FEATURE_COLUMNS].to_numpy(float)
    out = selected.copy()
    origins: list[str] = []
    for _, row in out.iterrows():
        x = np.array([float(row[c]) for c in FEATURE_COLUMNS], dtype=float)
        dist = float(np.sqrt(((lat - x) ** 2).sum(axis=1)).min()) if len(lat) else np.inf
        origins.append("lattice_retrieval" if dist < 1e-6 else "off_lattice_proposal")
    out["candidate_source"] = origins
    return out
