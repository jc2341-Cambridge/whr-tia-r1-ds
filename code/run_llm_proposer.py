"""R1 proposer runs for Table III (independent runs, all calls archived).

    set DEEPSEEK_API_KEY=...
    python run_llm_proposer.py --provider deepseek --runs 5

Writes writing/Yanmar/results/r1_proposer/{provider}_runNN_{raw.jsonl,records.csv,audit.json}
and a summary CSV over runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_proposer import PROVIDERS, run_proposer  # noqa: E402
from yanmar.dataset import YANMAR_ROOT, load_operating_data, sparse_active_learning_split  # noqa: E402

OUT_DIR = YANMAR_ROOT / "results" / "r1_proposer"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek", choices=sorted(PROVIDERS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--first-run", type=int, default=0)
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variant", default="bounds", choices=["bounds", "kag"])
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    if not os.environ.get(PROVIDERS[args.provider].key_env, "").strip():
        raise SystemExit(f"Set {PROVIDERS[args.provider].key_env} and rerun.")

    df = load_operating_data()
    train, oracle_pool, _test = sparse_active_learning_split(df)
    reference = oracle_pool if len(oracle_pool) else train

    audits = []
    for run_id in range(args.first_run, args.first_run + args.runs):
        audit = run_proposer(
            reference, args.out, run_id, provider=args.provider, model=args.model,
            n_calls=args.calls, workers=args.workers, variant=args.variant, seeds=train,
        )
        audits.append({k: v for k, v in audit.items() if k not in ("system_prompt", "user_prompt")})
        print(json.dumps(audits[-1], ensure_ascii=False))

    stem = args.provider if args.variant == "bounds" else f"{args.provider}-{args.variant}"
    summary = Path(args.out) / f"{stem}_runs_summary.csv"
    prev = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    new = pd.DataFrame(audits)
    if len(prev):
        prev = prev[~prev["run_id"].isin(new["run_id"])]
    pd.concat([prev, new], ignore_index=True).sort_values("run_id").to_csv(summary, index=False)
    print("wrote", summary)


if __name__ == "__main__":
    main()
