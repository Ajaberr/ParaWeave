"""Free analyses from the existing Claude verdicts (no model calls): per build, (1) the judge check as agreement of the
source-reference verdicts with the English-reference verdicts on the same frozen claims, (2) whether the two references
rank the systems the same way, (3) the gate's rubric per build; plus a reproduction check of published table cells."""
import json, os, collections, statistics as st
from scipy.stats import kendalltau, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"Documents\Work\lab\ArabicDetection"
FAITH = {"SUPPORTED", "PARTIAL"}
NAMES = {"ar": "Arabic", "lgg": "Lugbara", "fa": "Farsi", "prs": "Dari", "tg": "Tajik"}
SYSN = {"openai": "GPT-5.5", "gptoss20": "gpt-oss-20B", "qwen30a3b": "Qwen3-30B-A3B", "qwen8": "Qwen3-8B", "aya": "Aya-23-8B",
        "fanar": "Fanar-1-9B", "dorna2": "Dorna2-8B", "sunflower9b": "Sunflower-9B", "sunflower32q4": "Sunflower-32B"}
J = [json.loads(l) for l in open(os.path.join(HERE, "ow", "ow_judge.jsonl"), encoding="utf-8")]
dial = {}
for l in open(os.path.join(HERE, "ow", "ow_articles.jsonl"), encoding="utf-8"):
    a = json.loads(l); dial[(a["lang"], a["id"])] = a["dialect"]

def kappa(c):
    tp, fn, fp, tn = c[(1, 1)], c[(1, 0)], c[(0, 1)], c[(0, 0)]
    n = tp + fn + fp + tn
    po = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n / n
    return (po - pe) / (1 - pe), n, fp / max(1, fp + tn), fn / max(1, tp + fn), (tp / max(1, tp + fn) + tn / max(1, fp + tn)) / 2

print("JUDGE CHECK: source-reference verdict vs English-reference verdict (truth), faithful = SUPPORTED or PARTIAL")
print(f"{'build':10s} {'input':7s} {'kappa':>6s} {'n':>7s} {'false acc':>9s} {'true rej':>8s} {'bal.acc':>7s}")
groups = collections.defaultdict(collections.Counter)
for r in J:
    if not r["lab_src"]:
        continue
    gs = [r["lang"]] + ([f"ar:{dial[(r['lang'], r['id'])]}"] if r["lang"] == "ar" else [])
    for a, b in zip(r["lab_src"], r["lab_en"]):
        for g in gs:
            groups[(g, r["condition"])][(int(b in FAITH), int(a in FAITH))] += 1
for g in ["ar", "ar:egyptian", "ar:gulf", "ar:levantine", "ar:maghrebi", "fa", "prs", "tg", "lgg"]:
    for cond in ("dialect", "gold_ref"):
        k, n, fa, tr, ba = kappa(groups[(g, cond)])
        print(f"{g:12s} {'src' if cond == 'dialect' else 'En':5s} {k:6.2f} {n:7d} {100 * fa:8.0f}% {100 * tr:7.0f}% {ba:7.2f}")

print("\nSYSTEM RANKING with source input: source reference vs English reference")
for L in ["ar", "fa", "prs", "tg", "lgg"]:
    m = collections.defaultdict(lambda: {"src": [], "en": []})
    for r in J:
        if r["lang"] == L and r["condition"] == "dialect":
            if r["F_src"] is not None: m[r["system"]]["src"].append(r["F_src"])
            if r["F_en"] is not None: m[r["system"]]["en"].append(r["F_en"])
    sy = sorted(m)
    a = [st.mean(m[s]["src"]) for s in sy]; b = [st.mean(m[s]["en"]) for s in sy]
    tau = kendalltau(a, b).statistic; rho = spearmanr(a, b).statistic
    best_src = sy[max(range(len(sy)), key=lambda i: a[i])]; best_en = sy[max(range(len(sy)), key=lambda i: b[i])]
    order_en = [SYSN[s] for s, _ in sorted(zip(sy, b), key=lambda x: -x[1])]
    print(f"{NAMES[L]:8s} n_sys={len(sy)} kendall tau={tau:+.2f} spearman={rho:+.2f} | best by src ref: {SYSN[best_src]}, by En ref: {SYSN[best_en]}")
    print("          " + ", ".join(f"{SYSN[s]} {x:.3f}/{y:.3f}" for s, x, y in sorted(zip(sy, a, b), key=lambda t: -t[2])))

print("\nREPRODUCTION CHECK, Lugbara table (source input: ref Lgb / ref En; English input: ref Lgb / ref En)")
for s in ["sunflower32q4", "sunflower9b", "openai", "gptoss20", "qwen8", "qwen30a3b", "fanar", "aya"]:
    cells = []
    for cond in ("dialect", "gold_ref"):
        for f in ("F_src", "F_en"):
            v = [r[f] for r in J if r["lang"] == "lgg" and r["system"] == s and r["condition"] == cond and r[f] is not None]
            cells.append(st.mean(v))
    print(f"  {SYSN[s]:14s} " + " ".join(f"{c:.3f}" for c in cells))

print("\nGATE RUBRIC per build (English side, Claude Sonnet 4.6 gate)")
for L, fn in (("lgg", "lugbara"), ("fa", "farsi"), ("prs", "dari"), ("tg", "tajik")):
    R = [json.loads(l) for l in open(os.path.join(ROOT, "datasets", "official", "derived", "DiaBench-ML", f"{fn}_eval.jsonl"), encoding="utf-8")]
    rb = [r["rubric_claude"] for r in R if r.get("rubric_claude")]
    print(f"  {NAMES[L]:8s} n={len(R)} coherence {st.mean(x['coherence'] for x in rb):.2f} | rated 5 {100 * sum(x['coherence'] == 5 for x in rb) / len(rb):.0f}% | "
          f"article_like {st.mean(x['article_like'] for x in rb):.2f} | summarizability {st.mean(x['summarizability'] for x in rb):.2f} | "
          f"sent/art {st.mean(len(r['dialect_article']) for r in R):.1f}")
