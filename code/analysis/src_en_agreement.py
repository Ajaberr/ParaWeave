"""Per-claim agreement between the judge's verdict against the source-language article and against the English article
(same frozen claims; faithful = SUPPORTED or PARTIAL; English verdict as truth). English-input summaries (the RQ2 setting)
and, for reference, source-input summaries. Systems = the six of Figure fig:summ (four general + GPT-5.5 + the family
specialist), matching the judge figure's losses; 'all' = every system of the build (the appendix's pooling)."""
import json, collections
ROOT = r"eval/official/openweight/ow"
FAITH = {"SUPPORTED", "PARTIAL"}
GEN = {"openai", "gptoss20", "qwen30a3b", "qwen8", "aya"}
SPEC = {"ar": "fanar", "fa": "dorna2", "prs": "dorna2", "tg": "dorna2", "lgg": "sunflower9b", "ach": "sunflower9b", "teo": "sunflower9b"}
dial = {}
for l in open(ROOT + r"/ow_articles.jsonl", encoding="utf-8"):
    a = json.loads(l); dial[(a["lang"], a["id"])] = a["dialect"]
C = collections.defaultdict(collections.Counter)
for l in open(ROOT + r"/ow_judge.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if not r.get("lab_src") or not r.get("lab_en"):
        continue
    six = r["system"] in GEN or r["system"] == SPEC[r["lang"]]
    gs = [r["lang"]] + ([f"ar:{dial[(r['lang'], r['id'])]}"] if r["lang"] == "ar" else [])
    for a, b in zip(r["lab_src"], r["lab_en"]):
        cell = (int(b in FAITH), int(a in FAITH))            # (English verdict, source verdict)
        for g in gs:
            for pool in (["six", "all"] if six else ["all"]):
                C[(g, r["condition"], pool)][cell] += 1

def stats(c):
    tp, fn, fp, tn = c[(1, 1)], c[(1, 0)], c[(0, 1)], c[(0, 0)]
    n = tp + fn + fp + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n / n
    return n, po, (po - pe) / (1 - pe), fp / max(1, fp + tn), fn / max(1, tp + fn), (tp + fn) / n

NAME = {"fa": "Persian", "prs": "Dari", "tg": "Tajik", "ar": "Arabic (pooled)", "ar:egyptian": "  Egyptian", "ar:gulf": "  Gulf",
        "ar:levantine": "  Levantine", "ar:maghrebi": "  Maghrebi", "lgg": "Lugbara", "ach": "Acholi", "teo": "Ateso"}
for pool in ("six", "all"):
    print(f"\n=== systems: {pool} | English-input claims (RQ2)                      | source-input claims")
    print(f"{'variety':16s} {'n':>6s} {'agree':>6s} {'kappa':>6s} {'passFalse':>9s} {'rejTrue':>8s} {'En-faith':>8s} | {'agree':>6s} {'kappa':>6s}")
    for g in NAME:
        e = stats(C[(g, "gold_ref", pool)]); s = stats(C[(g, "dialect", pool)])
        print(f"{NAME[g]:16s} {e[0]:6d} {100*e[1]:5.1f}% {e[2]:6.2f} {100*e[3]:8.0f}% {100*e[4]:7.0f}% {100*e[5]:7.0f}% | {100*s[1]:5.1f}% {s[2]:6.2f}")
