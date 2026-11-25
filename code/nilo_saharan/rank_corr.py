"""Ranking agreement (Kendall tau-b) between the judge's English-reference ranking of the systems' source-input summaries
(F_src,En, the paper's summarizer ranking) and the rankings by ROUGE-L, BLEURT (against GPT-5.5's English-input summary)
and translation chrF, in every build. Open-weight systems only: GPT-5.5 wrote the reference summary and has no
translation probe. Local, from the open-weight pipeline's outputs."""
import json, os, statistics as st, sys
from scipy.stats import kendalltau
sys.stdout.reconfigure(encoding="utf-8")
OW = r"eval/official/openweight/ow"
R = json.load(open(os.path.join(OW, "results.json"), encoding="utf-8"))
F = {}
for l in open(os.path.join(OW, "ow_judge.jsonl"), encoding="utf-8"):
    j = json.loads(l)
    if j["condition"] == "dialect" and j["F_en"] is not None:
        F.setdefault((j["lang"], j["system"]), []).append(j["F_en"])
judge = {k: st.mean(v) for k, v in F.items()}
BUILDS = ["ar", "fa", "prs", "tg", "lgg", "ach", "teo"]
out = {}
for b in BUILDS:
    sy = sorted(s for (L, s) in judge if L == b and s != "openai")
    row = {}
    for name, get in (("ROUGE-L", lambda s: (R["refm"].get(f"{b}|{s}") or {}).get("rl_src")),
                      ("BLEURT", lambda s: (R["refm"].get(f"{b}|{s}") or {}).get("bl_src")),
                      ("chrF", lambda s: (R["trans"].get(f"{b}|{s}") or {}).get("chrf"))):
        pts = [(get(s), judge[(b, s)]) for s in sy if get(s) is not None]
        row[name] = (kendalltau([p[0] for p in pts], [p[1] for p in pts]).statistic, len(pts))
    out[b] = row
    print(b, {k: (round(v[0], 2), v[1]) for k, v in row.items()})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rank_corr.json"), "w"), indent=1)
