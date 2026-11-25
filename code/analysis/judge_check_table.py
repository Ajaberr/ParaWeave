"""Numbers for the appendix judge-check table (every system of each build, as the appendix paragraph states) and the
per-system Lugbara numbers the Discussion cites. English-reference verdict = truth; faithful = SUPPORTED or PARTIAL."""
import json, collections, statistics as st
from scipy.stats import kendalltau
OW = r"eval/official/openweight/ow"
FAITH = {"SUPPORTED", "PARTIAL"}
J = [json.loads(l) for l in open(OW + r"/ow_judge.jsonl", encoding="utf-8")]
C = collections.defaultdict(collections.Counter)
S = collections.defaultdict(collections.Counter)                  # per (lang, system) for source-input claims
for r in J:
    if not r.get("lab_src") or not r.get("lab_en"):
        continue
    for a, b in zip(r["lab_src"], r["lab_en"]):
        cell = (int(b in FAITH), int(a in FAITH))
        C[(r["lang"], r["condition"])][cell] += 1
        if r["condition"] == "dialect":
            S[(r["lang"], r["system"])][cell] += 1

def st4(c):
    tp, fn, fp, tn = c[(1, 1)], c[(1, 0)], c[(0, 1)], c[(0, 0)]
    n = tp + fn + fp + tn; po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n / n
    return (po - pe) / (1 - pe), fp / max(1, fp + tn), fn / max(1, tp + fn), n

def tau(L):
    m = collections.defaultdict(lambda: ([], []))
    for r in J:
        if r["lang"] == L and r["condition"] == "dialect":
            if r["F_src"] is not None: m[r["system"]][0].append(r["F_src"])
            if r["F_en"] is not None: m[r["system"]][1].append(r["F_en"])
    sy = sorted(m)
    return kendalltau([st.mean(m[s][0]) for s in sy], [st.mean(m[s][1]) for s in sy]).statistic, len(sy)

print(f"{'build':6s} {'k_src':>6s} {'k_en':>6s} | src-input: {'passF':>6s} {'rejT':>6s} | En-input: {'passF':>6s} {'rejT':>6s} | tau (n_sys)")
for L in ("ar", "fa", "prs", "tg", "lgg", "ach", "teo"):
    ks, pfs, rts, ns = st4(C[(L, "dialect")]); ke, pfe, rte, ne = st4(C[(L, "gold_ref")]); t, nsys = tau(L)
    print(f"{L:6s} {ks:6.2f} {ke:6.2f} | {100*pfs:9.0f}% {100*rts:5.0f}% | {100*pfe:8.0f}% {100*rte:5.0f}% | {t:+.2f} ({nsys})")
print("\nLugbara-input claims per system: share of English-false claims the judge passes against Lugbara / share of English-true claims it rejects")
for s in ("openai", "gptoss20", "qwen30a3b", "qwen8", "fanar", "aya", "sunflower9b", "sunflower32q4"):
    k, pf, rt, n = st4(S[("lgg", s)])
    print(f"  {s:14s} passes {100*pf:5.1f}%  rejects {100*rt:5.1f}%  (n={n})")
