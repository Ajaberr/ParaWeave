"""Score Lugbara, Acholi and Ateso on the same 240 articles (identical English side), with the conventions of the
paper's tables (eval/official/rebuttal/result_cis.py, ow_prep.py): F = (supported + 0.5 partial) / claims, summaries
with no claim left out; judge check = claim-level verdict against the source article vs against the English article
(truth), faithful = SUPPORTED or PARTIAL, pooled over systems; Kendall tau-b between the systems' mean F_src,src and
F_src,En on source-input summaries. For Acholi/Ateso the English-input summaries, claims and English-reference verdicts
are Lugbara's (same English article); only their check against the new source article is new.

  python eval/official/nilo/nilo_analyze.py   -> eval/official/nilo/nilo_results.{json,md}"""
import json, os, statistics as st
from scipy.stats import kendalltau
ROOT = r"Documents\Work\lab\ArabicDetection"
OUT = os.path.join(ROOT, "eval", "official", "out"); HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = ["openai", "sunflower32q4", "sunflower9b", "gptoss20", "qwen8", "qwen30a3b", "fanar", "aya"]
SN = {"openai": "GPT-5.5", "qwen30a3b": "Qwen3-30B-A3B", "gptoss20": "gpt-oss-20B", "qwen8": "Qwen3-8B",
      "aya": "Aya-23-8B", "fanar": "Fanar-1-9B", "sunflower9b": "Sunflower-9B", "sunflower32q4": "Sunflower-32B"}
LANGS = {"lgg": "Lugbara", "ach": "Acholi", "teo": "Ateso"}
FAITH = {"SUPPORTED", "PARTIAL"}
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def parse(faith, claims):
    fd, fg = faith.get("dialect") or {}, faith.get("gold_reference") or {}
    a, b = fd.get("verdicts") or [], fg.get("verdicts") or []
    cm = None
    if len(a) == len(b) == len(claims) and claims:
        t = [int((y or {}).get("label") in FAITH) for y in b]; p = [int((x or {}).get("label") in FAITH) for x in a]
        tp = sum(ti and pi for ti, pi in zip(t, p)); fn = sum(ti and not pi for ti, pi in zip(t, p))
        fp = sum((not ti) and pi for ti, pi in zip(t, p)); cm = (tp, fn, fp, len(t) - tp - fn - fp)
    return {"F_src": fd.get("faithfulness_rate") if fd.get("n_claims") else None,
            "F_en": fg.get("faithfulness_rate") if fg.get("n_claims") else None, "n_claims": len(claims), "cm": cm}

def load(L, s):
    """-> {"dialect": {aid: parsed}, "gold_ref": {aid: parsed}} keyed by the Lugbara article id."""
    lgg = {(r["id"], r["condition"]): r for r in rd(os.path.join(OUT, f"gen.mlext.{s}.faith.claude.jsonl"))}
    D = {"dialect": {}, "gold_ref": {}}
    if L == "lgg":
        for (aid, c), r in lgg.items(): D[c][aid] = parse(r.get("faith") or {}, r.get("claims") or [])
        return D
    d = os.path.join(OUT, "nilo", L)
    src = os.path.join(d, f"gen.mlext.{s}.faith.claude.jsonl"); enin = os.path.join(d, f"gen.mlext.{s}_enin.faith.claude.jsonl")
    if os.path.exists(src):
        for r in rd(src):
            D["dialect"][r["id"].replace(f"{L}-", "lgg-")] = parse(r.get("faith") or {}, r.get("claims") or [])
    if os.path.exists(enin):
        for r in rd(enin):
            base = lgg[(r["lgg_id"], "gold_ref")]
            assert (base.get("claims") or []) == (r.get("claims") or []), (L, s, r["id"])   # the same frozen claims
            faith = {"dialect": (r.get("faith") or {}).get("dialect"), "gold_reference": (base.get("faith") or {}).get("gold_reference")}
            D["gold_ref"][r["lgg_id"]] = parse(faith, r.get("claims") or [])
    return D

def kappa(T):
    tp, fn, fp, tn = T; n = tp + fn + fp + tn
    po = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return (po - pe) / (1 - pe)

R = {}
for L in LANGS:
    per = {}; Tsrc = [0, 0, 0, 0]; Ten = [0, 0, 0, 0]; complete = True
    for s in SYSTEMS:
        D = load(L, s)
        if len(D["dialect"]) < 240 or len(D["gold_ref"]) < 240: complete = False
        m = lambda c, k: (st.mean(x[k] for x in D[c].values() if x[k] is not None) if any(x[k] is not None for x in D[c].values()) else None)
        per[s] = {"F_src_En": m("dialect", "F_en"), "F_En_En": m("gold_ref", "F_en"), "F_src_src": m("dialect", "F_src"),
                  "F_En_src": m("gold_ref", "F_src"), "no_claim_src": sum(1 for x in D["dialect"].values() if not x["n_claims"]),
                  "n_src": len(D["dialect"]), "n_en": len(D["gold_ref"])}
        for x in D["dialect"].values():
            if x["cm"]: Tsrc = [a + b for a, b in zip(Tsrc, x["cm"])]
        for x in D["gold_ref"].values():
            if x["cm"]: Ten = [a + b for a, b in zip(Ten, x["cm"])]
    sy = [s for s in SYSTEMS if per[s]["F_src_En"] is not None and per[s]["F_src_src"] is not None]
    tau = kendalltau([per[s]["F_src_src"] for s in sy], [per[s]["F_src_En"] for s in sy]).statistic if len(sy) > 2 else None
    tp, fn, fp, tn = Tsrc
    R[L] = {"complete": complete, "systems": per,
            "judge": {"kappa_src_input": kappa(Tsrc) if sum(Tsrc) else None, "kappa_en_input": kappa(Ten) if sum(Ten) else None,
                      "false_passed": fp / (fp + tn) if fp + tn else None, "true_rejected": fn / (tp + fn) if tp + fn else None,
                      "tau": tau, "n_claims_src": sum(Tsrc), "n_claims_en": sum(Ten)}}
json.dump(R, open(os.path.join(HERE, "nilo_results.json"), "w"), indent=1)

f2 = lambda x: "--" if x is None else f"{x:.3f}"
md = ["# Lugbara, Acholi, Ateso on the same 240 articles (same English side)\n",
      "## Summarizer: faithfulness against the English article (source input / English input)\n",
      "| System | " + " | ".join(f"{LANGS[L]} src | {LANGS[L]} En" for L in LANGS) + " |", "|---" * (1 + 2 * len(LANGS)) + "|"]
for s in SYSTEMS:
    md.append(f"| {SN[s]} | " + " | ".join(f"{f2(R[L]['systems'][s]['F_src_En'])} | {f2(R[L]['systems'][s]['F_En_En'])}" for L in LANGS) + " |")
md += ["\n## Judge: verdict against the source article vs against the English article (truth)\n",
       "| | " + " | ".join(LANGS.values()) + " |", "|---" * (1 + len(LANGS)) + "|"]
for k, name in (("kappa_src_input", "Cohen's kappa, source-input claims"), ("kappa_en_input", "Cohen's kappa, English-input claims"),
                ("false_passed", "False claims passed"), ("true_rejected", "True claims rejected"), ("tau", "Kendall tau (rankings)")):
    md.append(f"| {name} | " + " | ".join(f2(R[L]["judge"][k]) for L in LANGS) + " |")
md += ["\n## Source-reference scores and no-claim counts\n", "| System | " + " | ".join(f"{LANGS[L]} F_src,src | {LANGS[L]} F_En,src | no claim" for L in LANGS) + " |",
       "|---" * (1 + 3 * len(LANGS)) + "|"]
for s in SYSTEMS:
    md.append(f"| {SN[s]} | " + " | ".join(f"{f2(R[L]['systems'][s]['F_src_src'])} | {f2(R[L]['systems'][s]['F_En_src'])} | {R[L]['systems'][s]['no_claim_src']}" for L in LANGS) + " |")
md.append("\nComplete: " + ", ".join(f"{LANGS[L]}={R[L]['complete']}" for L in LANGS))
open(os.path.join(HERE, "nilo_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))
