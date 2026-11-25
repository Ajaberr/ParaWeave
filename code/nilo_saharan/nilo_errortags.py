"""Error-tag rates judged against English (per claim, pooled over all systems), source input / English input, as in the
paper's transfer error-tag table; Lugbara is recomputed first to check the definition reproduces the paper."""
import json, os, collections, sys
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"eval/official/out"
SYSTEMS = ["openai", "gptoss20", "qwen30a3b", "qwen8", "aya", "fanar", "sunflower9b", "sunflower32q4"]
TAGS = ["entity", "number", "unsupported", "contradiction", "relation", "overgeneralization"]
PAT = {"lgg": "gen.mlext.{s}.faith.claude.jsonl", "ach": "gen.ach.{s}.faith.claude.jsonl", "teo": "gen.teo.{s}.faith.claude.jsonl"}
res = {}
for L, pat in PAT.items():
    c = {"dialect": collections.Counter(), "gold_ref": collections.Counter()}; n = collections.Counter()
    for s in SYSTEMS:
        for l in open(os.path.join(OUT, pat.format(s=s)), encoding="utf-8"):
            r = json.loads(l); v = ((r.get("faith") or {}).get("gold_reference") or {}).get("verdicts") or []
            for x in v:
                n[r["condition"]] += 1; c[r["condition"]][str((x or {}).get("error") or "").split(":")[0].strip().lower()] += 1
    res[L] = {t: (c["dialect"][t] / n["dialect"], c["gold_ref"][t] / n["gold_ref"]) for t in TAGS}
    res[L]["_n"] = dict(n)
for t in TAGS:
    print(f"{t:20s}" + "".join(f"  {L}: {res[L][t][0]:.3f} / {res[L][t][1]:.3f}" for L in PAT))
print({L: res[L]["_n"] for L in PAT})
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nilo_errortags.json"), "w"), indent=1)
