"""GPT-5.6-sol reasoning-effort sweep on the 240 Lugbara articles: the four F_{r,g} scores per level (same scoring as
every Lugbara row: SUPPORTED 1, PARTIAL 0.5, summaries with no claim excluded), bootstrap 95% CIs over articles,
no-claim counts, summary length, error mix, and tokens/cost per summary from the per-level usage logs. The 2026-09-21
gpt-5.6-sol low run and the GPT-5.5 roster row are printed alongside for reference. Writes ow/effort_results.json."""
import json, os, random, statistics as st, collections
ROOT = r"Documents\Work\lab\ArabicDetection"
OUT = os.path.join(ROOT, "eval", "official", "out")
LOGS = os.path.join(ROOT, "datasets", "official", "derived", "DiaBench-ML", "logs")
HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ["none", "low", "medium", "high", "xhigh"]
SYS = [(f"e{e}", f"openai-gpt56sol-e{e}") for e in LEVELS] + [("low 09-21", "openai-gpt56sol"), ("GPT-5.5 low", "openai")]
CELLS = [("dialect", "dialect"), ("dialect", "gold_reference"), ("gold_ref", "dialect"), ("gold_ref", "gold_reference")]
TAGS = ["entity", "number", "relation", "unsupported", "contradiction", "overgeneralization"]
def boot(vals, n=2000, seed=0):
    rnd = random.Random(seed); m = len(vals)
    bs = sorted(st.mean(vals[rnd.randrange(m)] for _ in range(m)) for _ in range(n))
    return bs[int(0.025 * n)], bs[int(0.975 * n)]
res = {}
print(f"{'run':12s} {'Lgb/Lgb':>8s} {'Lgb/En':>8s} {'95% CI':>13s} {'En/Lgb':>8s} {'En/En':>8s} {'gap':>6s} {'judge':>6s} {'noclaim':>7s} {'words':>6s}")
for name, sysn in SYS:
    p = os.path.join(OUT, f"gen.mlext.{sysn}.faith.claude.jsonl")
    if not os.path.exists(p):
        print(f"{name:12s} (not judged yet)"); continue
    R = [json.loads(l) for l in open(p, encoding="utf-8")]
    cell = {}
    for cond, g in CELLS:
        v = [r["faith"][g]["faithfulness_rate"] for r in R if r["condition"] == cond and (r.get("faith") or {}).get(g, {}).get("n_claims")]
        cell[(cond, g)] = v
    m = {k: st.mean(v) for k, v in cell.items()}
    lo, hi = boot(cell[("dialect", "gold_reference")])
    noclaim = sum(1 for r in R if r["condition"] == "dialect" and not ((r.get("faith") or {}).get("gold_reference") or {}).get("n_claims"))
    words = st.mean(len((r.get("summary") or "").split()) for r in R if r["condition"] == "dialect")
    tags = collections.Counter(); ncl = 0
    for r in R:
        fg = (r.get("faith") or {}).get("gold_reference") or {}
        if r["condition"] == "dialect" and fg.get("n_claims"):
            ncl += fg["n_claims"]
            for t, c in (fg.get("error_histogram") or {}).items():
                t = t.lower().split(":")[0]
                for full in TAGS:
                    if t.startswith(full[:5]): tags[full] += c
    gap = m[("gold_ref", "gold_reference")] - m[("dialect", "gold_reference")]
    judge = m[("gold_ref", "gold_reference")] - m[("gold_ref", "dialect")]
    res[name] = {"cells": {f"{a}|{b}": m[(a, b)] for a, b in CELLS}, "ci_src_en": [lo, hi], "noclaim": noclaim, "words": words,
                 "gap": gap, "judge": judge, "n": {f"{a}|{b}": len(cell[(a, b)]) for a, b in CELLS},
                 "tags_per_claim": {t: tags[t] / max(1, ncl) for t in TAGS}}
    print(f"{name:12s} {m[('dialect','dialect')]:8.3f} {m[('dialect','gold_reference')]:8.3f}   [{lo:.3f},{hi:.3f}] {m[('gold_ref','dialect')]:8.3f} "
          f"{m[('gold_ref','gold_reference')]:8.3f} {gap:6.3f} {judge:6.3f} {noclaim:7d} {words:6.1f}")
print("\nerror tags per claim, Lugbara input judged against English:")
for name in res:
    print(f"  {name:12s} " + " ".join(f"{t[:6]}={res[name]['tags_per_claim'][t]:.3f}" for t in TAGS))
print("\ntokens per summary (Bedrock Mantle usage, both input conditions) and estimated cost at $5/$30 per M:")
for e in LEVELS:
    p = os.path.join(LOGS, f"usage_gpt56sol_effort_{e}.jsonl")
    if not os.path.exists(p):
        continue
    U = [json.loads(l) for l in open(p, encoding="utf-8")]
    if not U: continue
    mi, mo = st.mean(u["in"] for u in U), st.mean(u["out"] for u in U)
    usd = sum(u["usd"] for u in U)
    res.setdefault(f"e{e}", {})["tokens"] = {"calls": len(U), "in": mi, "out": mo, "usd_logged": usd, "usd_per_480": 480 * (mi * 5 + mo * 30) / 1e6}
    print(f"  e{e:7s} calls={len(U):4d} mean in={mi:6.0f} mean out={mo:6.0f} (incl. reasoning) | logged ${usd:.2f} | full level ~${480 * (mi * 5 + mo * 30) / 1e6:.2f}")
json.dump(res, open(os.path.join(HERE, "ow", "effort_results.json"), "w", encoding="utf-8"), indent=1)
