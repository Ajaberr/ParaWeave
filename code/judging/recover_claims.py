#!/usr/bin/env python
"""One-off recovery: re-decompose rows that got 0 claims from the pre-fix DECOMP_SYS prompt
(unattributed first-person dialogue -- decomposer refused to guess an entity). Only touches rows
where claims was empty; everything else in the frozen claims file is left untouched.

  python eval/recover_claims.py <system>
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import LLM
from claims import decompose_many

GEN = Path(__file__).resolve().parent / "out" / "roster_gen"


def main():
    s = sys.argv[1]
    recover_in = GEN / f"recover_input.{s}.jsonl"
    affected = [json.loads(l) for l in recover_in.open(encoding="utf-8")]
    print(f"[{s}] recovering {len(affected)} rows")

    llm = LLM("claude")
    new_claims = decompose_many(llm, [r["summary"] for r in affected])
    recovered = sum(1 for c in new_claims if c)
    print(f"[{s}] recovered {recovered}/{len(affected)}")

    by_id = {(r["id"], r["condition"]): cl for r, cl in zip(affected, new_claims)}

    # patch into the frozen claims file (leave every other row untouched)
    claims_path = GEN / f"claims.ours.{s}.rest.jsonl"
    gen_path = GEN / f"gen.ours.{s}.rest.jsonl"
    gen_rows = [json.loads(l) for l in gen_path.open(encoding="utf-8")]
    claims_rows = [json.loads(l) for l in claims_path.open(encoding="utf-8")]
    patched = 0
    for gr, cr in zip(gen_rows, claims_rows):
        key = (gr["id"], gr["condition"])
        if key in by_id and by_id[key]:
            cr["claims"] = by_id[key]
            patched += 1
    with claims_path.open("w", encoding="utf-8") as fh:
        for cr in claims_rows:
            fh.write(json.dumps(cr, ensure_ascii=False) + "\n")
    print(f"[{s}] patched {patched} rows into {claims_path.name}")


if __name__ == "__main__":
    main()
