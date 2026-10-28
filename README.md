# TIA 2026-ESC-1174 R1 — DeepSeek V4-Pro archive

Versioned release for *Physics-Guided Multi-Agent Operating Intelligence for Autonomous Dispatch in Waste Heat Recovery Microgrids* (IEEE Transactions on Industry Applications, manuscript **2026-ESC-1174**, revision R1).

This repository freezes the **DeepSeek V4-Pro** proposer runs used in the revision. It is a separate archive from the earlier public validation package. Kimi K2.6 records from the original submission were not archived and those results were withdrawn; a Kimi folder can be added later if those records are recovered.

Tag: **`r1-ds`**.

## What is frozen

| Item | Path |
|---|---|
| RK4 solver and Yanmar coefficients | `code/yanmar/stirling_rk_solver.py`, `code/yanmar/engine_specs.py` |
| 27 seed / 18 holdout / oracle pool | `data/seed27.csv`, `data/holdout18.csv`, `data/oracle_pool.csv` |
| 256-point validation and 512-point test sets | `data/validation_sobol256.csv`, `data/test_sobol512.csv` |
| Five DeepSeek bounds-prompt runs (raw JSONL, gated records, audit) | `records/ds/deepseek_run0{0–4}_{raw.jsonl,records.csv,audit.json}` |
| Five DeepSeek KAG-prompt runs (prompt ablation) | `records/ds/deepseek-kag_run0{0–4}_*` |
| Prompts | `records/ds/prompts/` |
| Seed ledger (`s = 7 + 1000r + i`) | `records/ds/seed_ledger.csv` |
| Selected batches and acquisition scores | `results/r1_proposer/r1_pool_revised_selected.csv`, `main_pipeline_batch.csv` |
| Annual synthetic series | `data/annual_hourly_inputs.csv` |
| Scripts for Tables IV–VIII and Figs. 6–17 | `code/yanmar/r1_*.py` |

The language model is a proposer only. It never writes power or efficiency labels. Every numerical label used in Tables IV–VIII comes from the RK4 solver.

Decoding contract (all five bounds runs and all five KAG runs): `deepseek-v4-pro`, temperature `0.2`, `json_object`, thinking disabled, seed `s = 7 + 1000·run_id + call_index`.

## Run

API keys are read from the environment only (`DEEPSEEK_API_KEY`). None is stored here.

```text
python -m pip install -r requirements.txt
set PYTHONPATH=code
python -c "from yanmar.dataset import load_operating_data; print(len(load_operating_data()))"
```

Table and figure scripts expect to be imported as `yanmar.*` with `PYTHONPATH=code`.

## Licence

Source code: MIT. Solver-labelled operating tables in `data/` are research artefacts for this paper, not a measured Yanmar map.
