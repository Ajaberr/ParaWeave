#!/usr/bin/env python
"""Claude cross-judge harness (VM overloaded -> run the Claude decompose+judge via Opus 4.8 subagents,
which are a STRONGER judge than the VM's Sonnet 4.6 and need no API key). Mirrors judge_faith.py exactly.

  python eval/cj.py build  <gen_file> [n_slices]   -> eval/out/cj/in_<k>.jsonl  (slim slices for agents)
  python eval/cj.py agg     <gen_file>             -> read eval/out/cj/out_*.jsonl -> corpus scorecard

Each agent, per input row, decomposes summary -> atomic claims (claims.py DECOMP rules) then judges each
claim vs the dialect source AND the gold English ref (judge_faith.py labels/errors). Agent writes one JSON
object per row to out_<k>.jsonl: {id,dialect,condition,claims,dialect_verdicts,goldref_verdicts}.
"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict
import statistics

HERE = Path(__file__).resolve().parent
CJ = HERE / "out" / "cj"
LABELS = ["SUPPORTED", "PARTIAL", "NOT_SUPPORTED", "CONTRADICTED"]


def build(gen_file, n_slices):
    rows = [json.loads(l) for l in Path(gen_file).open(encoding="utf-8")]
    CJ.mkdir(parents=True, exist_ok=True)
    for f in CJ.glob("in_*.jsonl"):
        f.unlink()
    slim = []
    for r in rows:
        slim.append({
            "id": r["id"], "dialect": r["dialect"], "condition": r["condition"],
            "summary": r["summary"],
            "dialect_source": " ".join(r.get("source_sents") or [])[:8000],
            "goldref": " ".join(r.get("gold_reference") or [])[:8000],
        })
    # round-robin so every slice has a dialect mix (balanced agent load)
    slices = [[] for _ in range(n_slices)]
    for i, s in enumerate(slim):
        slices[i % n_slices].append(s)
    for k, sl in enumerate(slices):
        with (CJ / f"in_{k}.jsonl").open("w", encoding="utf-8") as fh:
            for s in sl:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {n_slices} slices to {CJ} ({len(slim)} rows, ~{len(slim)//n_slices}/slice)")


def _faith(verdicts):
    n = len(verdicts) or 1
    c = Counter(str(v.get("label", "NOT_SUPPORTED")).upper() for v in verdicts)
    sup, par = c.get("SUPPORTED", 0), c.get("PARTIAL", 0)
    nsup, con = c.get("NOT_SUPPORTED", 0), c.get("CONTRADICTED", 0)
    return {"n": len(verdicts), "faith": (sup + 0.5 * par) / n,
            "hd": (nsup + con) / n, "con": con / n,
            "errs": Counter(str(v.get("error", "none")).lower() for v in verdicts
                            if str(v.get("error", "none")).lower() != "none")}


def agg(gen_file):
    rows = {json.loads(l)["id"]: json.loads(l) for l in Path(gen_file).open(encoding="utf-8")}
    out_rows = []
    for f in sorted(CJ.glob("out_*.jsonl")):
        for l in f.open(encoding="utf-8"):
            l = l.strip()
            if l:
                out_rows.append(json.loads(l))
    print(f"judged rows: {len(out_rows)} / {len(rows)} gen rows")
    # per dialect x grounding x condition
    by = defaultdict(lambda: {"dialect": [], "goldref": []})
    tot = {"dialect": [], "goldref": []}
    errs = {"dialect": Counter(), "goldref": Counter()}
    con = {"dialect": [], "goldref": []}
    for o in out_rows:
        dv = o.get("dialect_verdicts") or []
        gv = o.get("goldref_verdicts") or []
        if not (dv or gv):
            continue
        d = o.get("dialect", "?")
        for g, verds in (("dialect", dv), ("goldref", gv)):
            if not verds:
                continue
            fv = _faith(verds)
            by[d][g].append(fv["faith"])
            tot[g].append(fv["faith"])
            con[g].append(fv["con"])
            errs[g].update(fv["errs"])
    print("\n# CLAUDE CROSS-JUDGE (Opus 4.8) faithfulness -- summarizer=GPT-5.5, judge=Claude")
    for g in ("dialect", "goldref"):
        if tot[g]:
            print(f"  ground={g:9s} faithfulness={statistics.mean(tot[g]):.3f}  "
                  f"contradiction={statistics.mean(con[g]):.3f}  "
                  f"top_errors={dict(errs[g].most_common(5))}  (n={len(tot[g])})")
    print("\n# BY DIALECT (faithfulness_rate: dialect-ground | goldref-ground)")
    for d in sorted(by, key=lambda x: -len(by[x]["dialect"])):
        dd = by[d]["dialect"]; gg = by[d]["goldref"]
        dm = statistics.mean(dd) if dd else float("nan")
        gm = statistics.mean(gg) if gg else float("nan")
        print(f"  {d:20s} dialect={dm:.3f} (n={len(dd):3d}) | goldref={gm:.3f} (n={len(gg):3d})")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    elif cmd == "agg":
        agg(sys.argv[2])
