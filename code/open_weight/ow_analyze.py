"""Open-weight runs, step 3 (local): join the VM outputs (ow/out/*) with the Claude verdicts and print every table.
Truth for the claim-level tables is the Claude judge's English-reference verdict (faithful = SUPPORTED or PARTIAL),
the reference the paper ranks systems on."""
import json, os, sys, collections, statistics as st
import numpy as np
from scipy.stats import rankdata, spearmanr, kendalltau, pearsonr
HERE = os.path.dirname(os.path.abspath(__file__))
OW = os.path.join(HERE, "ow"); OUT = os.path.join(OW, "out")
FAITH = {"SUPPORTED", "PARTIAL"}
BUILDS = ["ar", "fa", "prs", "tg", "lgg", "ach", "teo"]
NAMES = {"ar": "Arabic", "lgg": "Lugbara", "fa": "Persian", "prs": "Dari", "tg": "Tajik", "ach": "Acholi", "teo": "Ateso"}
SYSN = {"openai": "GPT-5.5", "gptoss20": "gpt-oss-20B", "qwen30a3b": "Qwen3-30B-A3B", "qwen8": "Qwen3-8B", "aya": "Aya-23-8B",
        "fanar": "Fanar-1-9B", "dorna2": "Dorna2-8B", "sunflower9b": "Sunflower-9B", "sunflower32q4": "Sunflower-32B"}

def key(r): return "|".join((r["lang"], r["system"], r["id"], r["condition"]))
def load(name):
    p = os.path.join(OUT, name + ".jsonl"); d = {}
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            try: r = json.loads(l); d[r["key"]] = r
            except Exception: pass
    return d
rows = [json.loads(l) for l in open(os.path.join(OW, "ow_rows.jsonl"), encoding="utf-8")]
J = {key(r): r for r in (json.loads(l) for l in open(os.path.join(OW, "ow_judge.jsonl"), encoding="utf-8"))}
A = {**load("align"), **load("align.rev")}; MC = load("minicheck"); NL = load("mdeberta"); E5 = load("e5"); BL = load("bleurt")
print(f"loaded: align {len(A)} minicheck {len(MC)} mdeberta {len(NL)} e5 {len(E5)} bleurt {len(BL)} of {len(rows)}")

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    npos, nneg = y.sum(), (1 - y).sum()
    if not npos or not nneg: return float("nan")
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

# ---------------- 1. claim level: how well does each checker recover the English-reference verdict --------------
print("\n[1] CLAIM-LEVEL AUC against the judge's English-reference verdict (all systems, both input conditions)")
SC = [("judge, source reference (binary)", None, "lab_src"),
      ("mDeBERTa, source premise", NL, "nli_src"), ("e5 evidence, source sentences", E5, "e5_src"),
      ("mDeBERTa, English premise", NL, "nli_en"), ("MiniCheck, English premise", MC, "mc_en"),
      ("e5 evidence, English sentences", E5, "e5_en")]
res1 = {}
RES = {'auc': {}, 'rho': {}, 'sys': {}, 'tau': {}, 'pairs': {}, 'refm': {}, 'trans': {}, 'trans_r': {}}
for cond_filter in (None, "dialect"):
    print(f"   -- input condition: {cond_filter or 'both'}")
    print("   " + f"{'checker':34s}" + "".join(f"{NAMES[b]:>9s}" for b in BUILDS) + "   n(claims) per build")
    ns = {}
    for name, src, fld in SC:
        vals = []
        for b in BUILDS:
            y, s = [], []
            for r in rows:
                if r["lang"] != b or (cond_filter and r["condition"] != cond_filter): continue
                k = key(r); j = J.get(k)
                if not j or not j["lab_en"]: continue
                if src is None:
                    sc = [1.0 if x in FAITH else 0.0 for x in j["lab_src"]]
                else:
                    if k not in src: continue
                    sc = src[k][fld]
                if len(sc) != len(j["lab_en"]): continue
                y += [1 if x in FAITH else 0 for x in j["lab_en"]]; s += sc
            vals.append(auc(y, s))
            if src is None:                      # the judge row covers every claim with aligned verdicts
                ns[b] = len(y); RES['auc'][f"n|{cond_filter or 'both'}|{b}"] = len(y)
                RES['auc'][f"pos|{cond_filter or 'both'}|{b}"] = int(sum(y))
            res1[(cond_filter, name, b)] = vals[-1]
            RES['auc'][f"{cond_filter or 'both'}|{name}|{b}"] = vals[-1]
        print("   " + f"{name:34s}" + "".join(f"{v:9.3f}" for v in vals))
    print("   " + " " * 34 + "".join(f"{ns.get(b, 0):9d}" for b in BUILDS))

# ---------------- 2. summary level: Spearman of each anchor with the judge ----------------------------------------
def sm(k):
    """per-summary anchor values; None when missing"""
    out = {}
    if k in A and A[k]["align_en"] is not None: out["align"] = A[k]["align_en"]
    if k in MC and MC[k]["mc_en"]: out["mc"] = st.mean(MC[k]["mc_en"])
    if k in NL and NL[k]["nli_en"]:
        out["nli_en"] = st.mean(NL[k]["nli_en"]); out["nli_src"] = st.mean(NL[k]["nli_src"])
    if k in E5 and E5[k]["e5_en"]:
        out["e5_en"] = st.mean(float(x >= 0.80) for x in E5[k]["e5_en"]); out["e5_src"] = st.mean(float(x >= 0.80) for x in E5[k]["e5_src"])
    return out
S = {key(r): sm(key(r)) for r in rows}
print("\n[2] PER-SUMMARY SPEARMAN rho (all systems, both input conditions)")
PAIRS2 = [("judge src-ref vs judge En-ref", "F_src", "F_en"), ("AlignScore (En) vs judge En-ref", "align", "F_en"),
          ("MiniCheck (En) vs judge En-ref", "mc", "F_en"), ("mDeBERTa En-premise vs judge En-ref", "nli_en", "F_en"),
          ("mDeBERTa src-premise vs judge src-ref", "nli_src", "F_src"), ("mDeBERTa src-premise vs judge En-ref", "nli_src", "F_en"),
          ("e5 src grounded vs judge src-ref", "e5_src", "F_src"), ("e5 En grounded vs judge En-ref", "e5_en", "F_en")]
print("   " + f"{'pair':40s}" + "".join(f"{NAMES[b]:>9s}" for b in BUILDS))
for name, a, b_ in PAIRS2:
    out = []
    for b in BUILDS:
        xs, ys = [], []
        for r in rows:
            if r["lang"] != b: continue
            k = key(r); j = J[k]; s = S[k]
            va = j.get(a) if a in ("F_src", "F_en") else s.get(a)
            vb = j.get(b_) if b_ in ("F_src", "F_en") else s.get(b_)
            if va is None or vb is None: continue
            xs.append(va); ys.append(vb)
        out.append(spearmanr(xs, ys).statistic if len(xs) > 10 else float("nan"))
        RES['rho'][f"{name}|{b}"] = out[-1]
    print("   " + f"{name:40s}" + "".join(f"{v:9.3f}" for v in out))

# ---------------- 3. system level: do the anchors rank the source-input summaries like the judge? ----------------
print("\n[3] SYSTEM MEANS, source-input summaries (judge F_src,En | F_src,src | AlignScore | MiniCheck | mDeB-En | mDeB-src | e5-src)")
tau = collections.defaultdict(dict)
for b in BUILDS:
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["lang"] != b or r["condition"] != "dialect": continue
        k = key(r); j = J[k]
        if j["F_en"] is not None: per[r["system"]]["F_en"].append(j["F_en"])
        if j["F_src"] is not None: per[r["system"]]["F_src"].append(j["F_src"])
        for f, v in S[k].items(): per[r["system"]][f].append(v)
    sy = sorted(per, key=lambda s: -st.mean(per[s]["F_en"]))
    print(f"  {NAMES[b]}")
    for s in sy:
        print(f"    {SYSN[s]:14s} " + " ".join(f"{st.mean(per[s][f]):.3f}" if per[s][f] else "  -  " for f in ("F_en", "F_src", "align", "mc", "nli_en", "nli_src", "e5_src")))
    RES['sys'][b] = {s: {f: st.mean(v) for f, v in per[s].items() if v} for s in sy}
    ref = [st.mean(per[s]["F_en"]) for s in sy]
    for f in ("F_src", "align", "mc", "nli_en", "nli_src", "e5_src", "e5_en"):
        if all(per[s][f] for s in sy):
            tau[b][f] = kendalltau(ref, [st.mean(per[s][f]) for s in sy]).statistic
            RES['tau'][f"{f}|{b}"] = tau[b][f]
print("  Kendall tau with the judge's English-reference ranking:")
print("   " + f"{'':10s}" + "".join(f"{NAMES[b]:>9s}" for b in BUILDS))
for f in ("F_src", "align", "mc", "nli_en", "nli_src", "e5_src", "e5_en"):
    print("   " + f"{f:10s}" + "".join(f"{tau[b].get(f, float('nan')):9.2f}" for b in BUILDS))

# ---------------- 4. known-true pairs --------------------------------------------------------------------------
p = os.path.join(OUT, "pairs.json")
if os.path.exists(p):
    print("\n[4] KNOWN-TRUE PAIRS (300 per variety)")
    P = json.load(open(p, encoding="utf-8"))
    RES['pairs'] = P
    for g in ["egyptian", "gulf", "levantine", "maghrebi", "fa", "prs", "tg", "lgg", "ach", "teo"]:
        if g in P:
            x = P[g]; print(f"   {g:10s} n={x['n']} e5 top1 {100 * x['e5_top1']:.1f}% top5 {100 * x['e5_top5']:.1f}% | mDeBERTa true {100 * x['nli_true_rate']:.1f}% "
                            f"(mean {x['nli_true_mean']:.2f}) neg {100 * x['nli_neg_rate']:.1f}% | En-En self {100 * x['nli_en_self_rate']:.1f}%")

# ---------------- 5. reference-summary metrics: ROUGE-L and BLEURT vs GPT-5.5's English-input summary -------------
from rouge_score import rouge_scorer
rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
ref = {(r["lang"], r["id"]): r["summary"] for r in rows if r["system"] == "openai" and r["condition"] == "gold_ref"}
print("\n[5] ROUGE-L / BLEURT against GPT-5.5's English-input summary (src input | En input)")
for b in BUILDS:
    for s in [x for x in SYSN if x != "openai"]:
        cells = []
        for cond in ("dialect", "gold_ref"):
            rl, bl = [], []
            for r in rows:
                if r["lang"] == b and r["system"] == s and r["condition"] == cond and r["summary"].strip() and ref.get((b, r["id"]), "").strip():
                    rl.append(rs.score(ref[(b, r["id"])], r["summary"])["rougeL"].fmeasure)
                    v = BL.get(key(r), {}).get("bleurt_ref")
                    if v is not None: bl.append(v)
            cells.append((st.mean(rl) if rl else None, st.mean(bl) if bl else None, len(rl)))
        if cells[0][2]:
            RES['refm'][f"{b}|{s}"] = {'rl_src': cells[0][0], 'rl_en': cells[1][0], 'bl_src': cells[0][1], 'bl_en': cells[1][1], 'n': cells[0][2]}
            f = lambda v: f"{v:.3f}" if v is not None else "  -  "
            print(f"   {NAMES[b]:8s} {SYSN[s]:14s} ROUGE-L {f(cells[0][0])} {f(cells[1][0])} | BLEURT {f(cells[0][1])} {f(cells[1][1])}  n={cells[0][2]}/{cells[1][2]}")

# ---------------- 6. translation probe: chrF vs faithfulness ----------------------------------------------------
import sacrebleu
TR = os.path.join(OW, "trans")
if os.path.isdir(TR):
    print("\n[6] TRANSLATION PROBE: mean chrF (sentence chrF averaged per article, then over articles) vs faithfulness")
    pts = []
    for b in BUILDS:
        fr = []
        for fn in sorted(os.listdir(TR)):
            if not fn.startswith(f"trans.{b}.") or not fn.endswith(".jsonl"): continue
            s = fn[len(f"trans.{b}."):-len(".jsonl")]
            perdoc = collections.defaultdict(list); nlong = n = 0
            for l in open(os.path.join(TR, fn), encoding="utf-8"):
                t = json.loads(l)
                h = t.get("en_hyp") or ""
                perdoc[t["id"]].append(sacrebleu.sentence_chrf(h, [t["en_gold"]]).score)
                n += 1; nlong += len(h.split()) > 2 * max(1, len(t["en_gold"].split()))
            docs = sorted(perdoc)
            chrf = st.mean(st.mean(perdoc[d]) for d in docs)
            fe = [J[f"{b}|{s}|{d}|dialect"]["F_en"] for d in docs if J.get(f"{b}|{s}|{d}|dialect") and J[f"{b}|{s}|{d}|dialect"]["F_en"] is not None]
            fs = [J[f"{b}|{s}|{d}|dialect"]["F_src"] for d in docs if J.get(f"{b}|{s}|{d}|dialect") and J[f"{b}|{s}|{d}|dialect"]["F_src"] is not None]
            # per-article correlation within this system
            xs, ys = [], []
            for d in docs:
                j = J.get(f"{b}|{s}|{d}|dialect")
                if j and j["F_en"] is not None: xs.append(st.mean(perdoc[d])); ys.append(j["F_en"])
            rdoc = spearmanr(xs, ys).statistic if len(xs) > 10 else float("nan")
            fr.append((s, chrf, st.mean(fe), st.mean(fs), rdoc, n, 100 * nlong / max(1, n)))
            RES['trans'][f"{b}|{s}"] = {'chrf': chrf, 'F_en': st.mean(fe), 'F_src': st.mean(fs), 'rdoc': rdoc, 'n': n, 'long': 100 * nlong / max(1, n)}
        for s, c, fe, fs, rdoc, n, lg in sorted(fr, key=lambda t: -t[1]):
            print(f"   {NAMES[b]:8s} {SYSN[s]:14s} chrF {c:6.2f} | F_src,En {fe:.3f} F_src,src {fs:.3f} | per-article rho {rdoc:+.2f} | n_sent {n} long {lg:.1f}%")
            pts.append((b, s, c, fe, fs))
        if len(fr) >= 3:
            RES['trans_r'][b] = {'r_en': pearsonr([t[1] for t in fr], [t[2] for t in fr]).statistic, 'r_src': pearsonr([t[1] for t in fr], [t[3] for t in fr]).statistic, 'n': len(fr)}
            print(f"   {NAMES[b]:8s} system-level r(chrF, F_src,En) = {pearsonr([t[1] for t in fr], [t[2] for t in fr]).statistic:+.3f}  "
                  f"r(chrF, F_src,src) = {pearsonr([t[1] for t in fr], [t[3] for t in fr]).statistic:+.3f}  (n={len(fr)})")
    if len(pts) >= 5:
        RES['trans_r']['pooled'] = {'r_en': pearsonr([p[2] for p in pts], [p[3] for p in pts]).statistic, 'rho_en': spearmanr([p[2] for p in pts], [p[3] for p in pts]).statistic, 'n': len(pts)}
        print(f"   POOLED over all build x system points: r(chrF, F_src,En) = {pearsonr([p[2] for p in pts], [p[3] for p in pts]).statistic:+.3f}, "
              f"Spearman {spearmanr([p[2] for p in pts], [p[3] for p in pts]).statistic:+.3f}, n={len(pts)}")

json.dump(RES, open(os.path.join(OW, 'results.json'), 'w', encoding='utf-8'), indent=1, default=float)
print('results ->', os.path.join(OW, 'results.json'))
