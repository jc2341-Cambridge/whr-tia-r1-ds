# -*- coding: utf-8 -*-
"""Provider-agnostic proposer for Algorithm 1, line 4 (R1 revision).

Same prompt contract as kimi_proposer.py: the model receives the reference-box
bounds and seed count, and returns typed (Ts, p, ω) records with a reason. It
never emits power or efficiency.

Every call is archived (request, raw response, model string, fingerprint,
token counts, timestamp) so the merged pool can be frozen against API drift.

Environment
-----------
DEEPSEEK_API_KEY     required for provider "deepseek"
MOONSHOT_API_KEY     required for provider "kimi"
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loader import FEATURE_COLUMNS
from kimi_proposer import _user_prompt, lattice_pool, parse_candidate_payload
from llm_protocol import PROPOSER_PROMPT


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    model: str


PROVIDERS = {
    "deepseek": Provider("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-v4-pro"),
    "kimi": Provider("kimi", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY", "kimi-k2.6"),
}

TEMPERATURE = 0.2
N_CALLS = 40
N_PER_CALL = 28

KAG_DEVICE_NOTE = (
    "Device: Yanmar SE220-100 equivalent helium beta-type Stirling engine, rated 9.9 kW at "
    "Ts = 1023.15 K, p = 2800 kPa, N = 800 rpm."
)
KAG_PHYSICS_PRIORS = (
    "Physics priors: shaft power rises with mean pressure and with source temperature; "
    "efficiency is bounded by the Carnot limit and decreases with engine speed through "
    "flow and regenerator losses."
)


def kag_user_prompt(reference: pd.DataFrame, seeds: pd.DataFrame, n_per_call: int) -> str:
    """KAG variant: the retrieved seed records and device knowledge are placed in context."""
    bounds = {c: (float(reference[c].min()), float(reference[c].max())) for c in FEATURE_COLUMNS}
    lines = "\n".join(
        f"{r.source_temperature:.2f}, {r.mean_pressure:.1f}, {r.engine_speed:.1f}, {r.power:.1f}, {r.efficiency:.4f}"
        for r in seeds.itertuples()
    )
    return (
        f"Retrieved labelled seed records ({len(seeds)}), columns Ts [K], p [kPa], N [rpm], power [W], efficiency [-]:\n"
        f"{lines}\n"
        f"{KAG_DEVICE_NOTE}\n"
        f"Feasible box: Ts in [{bounds['source_temperature'][0]:.2f}, {bounds['source_temperature'][1]:.2f}] K; "
        f"p in [{bounds['mean_pressure'][0]:.1f}, {bounds['mean_pressure'][1]:.1f}] kPa; "
        f"N in [{bounds['engine_speed'][0]:.1f}, {bounds['engine_speed'][1]:.1f}] rpm.\n"
        f"{KAG_PHYSICS_PRIORS}\n"
        "Task: propose operating points that fill the gaps between the labelled seeds, so that a "
        "surrogate fitted to the seeds plus the new points generalises over the whole box. Use the "
        "seed labels to judge where the surface bends or interacts. Do not repeat seed points and do "
        "not restrict yourself to the seed levels or to a regular grid.\n"
        f"Return exactly {n_per_call} candidates as a JSON object "
        '{"candidates":[{"source_temperature":...,"mean_pressure":...,'
        '"engine_speed":...,"reason":"..."}]} '
        "with no power or efficiency fields."
    )


def _call(provider: Provider, model: str, messages: list[dict[str, str]], seed: int, retries: int = 4) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ[provider.key_env], base_url=provider.base_url, timeout=300)
    last_err = ""
    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                seed=seed,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            usage = resp.usage
            return {
                "ok": True,
                "attempts": attempt + 1,
                "latency_s": round(time.time() - t0, 2),
                "returned_model": resp.model,
                "system_fingerprint": getattr(resp, "system_fingerprint", None),
                "finish_reason": resp.choices[0].finish_reason,
                "content": resp.choices[0].message.content or "",
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            }
        except Exception as exc:  # network / rate limit; retried, then recorded
            last_err = f"{type(exc).__name__}: {exc}"[:500]
            time.sleep(2.0 * (attempt + 1))
    return {"ok": False, "attempts": retries, "error": last_err, "content": ""}


def run_proposer(
    reference: pd.DataFrame,
    out_dir: str | Path,
    run_id: int,
    provider: str = "deepseek",
    model: str | None = None,
    n_calls: int = N_CALLS,
    n_per_call: int = N_PER_CALL,
    seed_base: int = 7,
    workers: int = 8,
    variant: str = "bounds",
    seeds: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """One independent proposer run; writes raw JSONL, parsed records CSV, and audit JSON."""
    prov = PROVIDERS[provider]
    model = model or prov.model
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{provider}_run{run_id:02d}" if variant == "bounds" else f"{provider}-{variant}_run{run_id:02d}"

    system = PROPOSER_PROMPT["system"]
    if variant == "bounds":
        user = _user_prompt(reference, n_per_call)
    elif variant == "kag":
        if seeds is None:
            raise ValueError("variant 'kag' needs the labelled seed records")
        user = kag_user_prompt(reference, seeds, n_per_call)
    else:
        raise ValueError(f"unknown prompt variant {variant!r}")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    seeds = [seed_base + 1000 * run_id + i for i in range(n_calls)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda s: _call(prov, model, messages, s), seeds))

    rows: list[dict[str, Any]] = []
    n_valid = n_parse_fail = n_api_fail = 0
    raw_path = out_dir / f"{tag}_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for call_idx, (seed, res) in enumerate(zip(seeds, results)):
            parsed = parse_candidate_payload(res.get("content", "")) if res["ok"] else []
            if not res["ok"]:
                n_api_fail += 1
            elif not parsed:
                n_parse_fail += 1
            else:
                n_valid += 1
                for rec_idx, rec in enumerate(parsed):
                    rows.append({"run_id": run_id, "call": call_idx, "record": rec_idx, **rec})
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "call": call_idx,
                        "requested_model": model,
                        "seed": seed,
                        "temperature": TEMPERATURE,
                        "messages": messages,
                        "n_parsed": len(parsed),
                        **res,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    records = pd.DataFrame(rows, columns=["run_id", "call", "record", *FEATURE_COLUMNS, "reason"])
    records.to_csv(out_dir / f"{tag}_records.csv", index=False)

    lattice = lattice_pool(reference)
    lat_keys = {tuple(np.round(r, 8)) for r in lattice[FEATURE_COLUMNS].to_numpy(float)}
    rec_x = records[FEATURE_COLUMNS].to_numpy(float) if len(records) else np.empty((0, 3))
    bounds = {c: (float(reference[c].min()), float(reference[c].max())) for c in FEATURE_COLUMNS}
    in_box = np.ones(len(records), dtype=bool)
    for j, c in enumerate(FEATURE_COLUMNS):
        in_box &= (rec_x[:, j] >= bounds[c][0]) & (rec_x[:, j] <= bounds[c][1])

    ok = [r for r in results if r["ok"]]
    audit = {
        "provider": provider,
        "prompt_variant": variant,
        "base_url": prov.base_url,
        "requested_model": model,
        "returned_models": sorted({r["returned_model"] for r in ok}),
        "system_fingerprints": sorted({str(r["system_fingerprint"]) for r in ok}),
        "temperature": TEMPERATURE,
        "thinking": "disabled",
        "response_format": "json_object",
        "seed_base": seed_base + 1000 * run_id,
        "run_id": run_id,
        "n_calls": n_calls,
        "n_per_call_requested": n_per_call,
        "schema_valid_calls": n_valid,
        "parse_failures": n_parse_fail,
        "api_failures": n_api_fail,
        "records": int(len(records)),
        "unique_records": int(records[FEATURE_COLUMNS].drop_duplicates().shape[0]) if len(records) else 0,
        "records_in_box": int(in_box.sum()),
        "exact_lattice_duplicates": int(sum(tuple(np.round(r, 8)) in lat_keys for r in rec_x)),
        "lattice_records": int(len(lattice)),
        "prompt_tokens": sum(r["prompt_tokens"] for r in ok),
        "completion_tokens": sum(r["completion_tokens"] for r in ok),
        "wall_latency_s_sum": round(sum(r["latency_s"] for r in ok), 1),
        "utc_finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system_prompt": system,
        "user_prompt": user,
    }
    audit["total_tokens"] = audit["prompt_tokens"] + audit["completion_tokens"]
    (out_dir / f"{tag}_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def heuristic_proposer(reference: pd.DataFrame, n_records: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic non-LLM second proposer: scrambled-Sobol points in the reference box.

    Matches the LLM record count so pool density is equal across proposer variants.
    """
    from scipy.stats import qmc

    bounds = np.array([[float(reference[c].min()), float(reference[c].max())] for c in FEATURE_COLUMNS])
    u = qmc.Sobol(d=3, scramble=True, seed=seed).random(n_records)
    x = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    out = pd.DataFrame(x, columns=FEATURE_COLUMNS)
    out["reason"] = "sobol_heuristic"
    return out


def space_filling_pool(reference: pd.DataFrame, n_records: int, kind: str, seed: int = 0) -> pd.DataFrame:
    """Density-matched control pools: 'sobol', 'random' (uniform), or 'lhs'."""
    from scipy.stats import qmc

    if kind == "sobol":
        return heuristic_proposer(reference, n_records, seed)
    bounds = np.array([[float(reference[c].min()), float(reference[c].max())] for c in FEATURE_COLUMNS])
    if kind == "random":
        u = np.random.default_rng(seed).random((n_records, 3))
    elif kind == "lhs":
        u = qmc.LatinHypercube(d=3, seed=seed).random(n_records)
    else:
        raise ValueError(kind)
    out = pd.DataFrame(bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0]), columns=FEATURE_COLUMNS)
    out["reason"] = f"{kind}_control"
    return out
