"""Build the LaTeX for the open-weight appendix tables from ow/results.json (written by ow_analyze.py), so no number
is typed by hand. Prints each block and writes ow/tables.json {name: latex}."""
import json, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "ow", "results.json"), encoding="utf-8"))
B = ["ar", "fa", "prs", "tg", "lgg", "ach", "teo"]
BN = {"ar": "Arabic", "fa": "Persian", "prs": "Dari", "tg": "Tajik", "lgg": "Lugbara", "ach": "Acholi", "teo": "Ateso"}
SYSN = {"openai": "GPT-5.5", "gptoss20": "gpt-oss-20B", "qwen30a3b": "Qwen3-30B-A3B", "qwen8": "Qwen3-8B", "aya": "Aya-23-8B",
        "fanar": "Fanar-1-9B", "dorna2": "Dorna2-8B", "sunflower9b": "Sunflower-9B", "sunflower32q4": "Sunflower-32B"}
f3 = lambda v: "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
f2 = lambda v: "--" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.2f}" if v >= 0 else f"$-${-v:.2f}")
pct = lambda v: f"{100 * v:.1f}\\%"
T = {}

# ---- claim-level AUC ---------------------------------------------------------------------------------------
auc = R["auc"]
rows = [("\\multicolumn{8}{@{}l}{\\textit{Reads the source article}} \\\\", None),
        ("\\quad Judge, source reference", "judge, source reference (binary)"),
        ("\\quad mDeBERTa, source sentences", "mDeBERTa, source premise"),
        ("\\quad e5 evidence, source sentences", "e5 evidence, source sentences"),
        ("\\midrule\n\\multicolumn{8}{@{}l}{\\textit{Reads the English article}} \\\\", None),
        ("\\quad MiniCheck", "MiniCheck, English premise"),
        ("\\quad mDeBERTa, English sentences", "mDeBERTa, English premise"),
        ("\\quad e5 evidence, English sentences", "e5 evidence, English sentences")]
L = ["\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
     "\\textbf{Checker} & " + " & ".join(f"\\textbf{{{BN[b]}}}" for b in B) + " \\\\", "\\midrule"]
for lab, k in rows:
    L.append(lab if k is None else lab + " & " + " & ".join(f3(auc[f"both|{k}|{b}"]) for b in B) + " \\\\")
L += ["\\midrule", "Claims & " + " & ".join(f"{auc[f'n|both|{b}']:,}".replace(",", "{,}") for b in B) + " \\\\", "\\bottomrule", "\\end{tabular}"]
T["anchors_auc"] = "\n".join(L)

# ---- system-level Kendall tau --------------------------------------------------------------------------------
tau = R["tau"]
trows = [("Judge, source reference", "F_src"), ("AlignScore-large, English article", "align"), ("MiniCheck, English article", "mc"),
         ("e5 evidence, English sentences", "e5_en"), ("mDeBERTa, English sentences", "nli_en"),
         ("mDeBERTa, source sentences", "nli_src"), ("e5 evidence, source sentences", "e5_src")]
L = ["\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
     "\\textbf{Ranks the systems by} & " + " & ".join(f"\\textbf{{{BN[b]}}}" for b in B) + " \\\\", "\\midrule"]
for lab, k in trows:
    L.append(lab + " & " + " & ".join(f2(tau.get(f"{k}|{b}")) for b in B) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
T["anchors_tau"] = "\n".join(L)

# ---- known-true pairs ----------------------------------------------------------------------------------------
P = R["pairs"]
L = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
     "\\textbf{Variety} & \\textbf{e5 top-1 recovery} & \\textbf{mDeBERTa, true pair} & \\textbf{mDeBERTa, shuffled pair} \\\\", "\\midrule"]
for g, nm in [("egyptian", "Egyptian"), ("gulf", "Gulf"), ("levantine", "Levantine"), ("maghrebi", "Maghrebi"),
              ("fa", "Persian"), ("prs", "Dari"), ("tg", "Tajik"), ("lgg", "Lugbara"), ("ach", "Acholi"), ("teo", "Ateso")]:
    x = P[g]
    L.append(f"{nm} & {pct(x['e5_top1'])} & {pct(x['nli_true_rate'])} & {pct(x['nli_neg_rate'])} \\\\")
    if g == "maghrebi": L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}"]
T["anchors_pairs"] = "\n".join(L)

# ---- reference-summary metrics ---------------------------------------------------------------------------------
M = R["refm"]
L = []
for s in ["gptoss20", "fanar", "qwen8", "aya"]:
    x = M[f"ar|{s}"]
    L.append(f"{SYSN[s]} & {f3(x['rl_src'])} & {f3(x['rl_en'])} & {f3(x['bl_src'])} & {f3(x['bl_en'])} \\\\")
T["refm_ar_rows"] = "\n".join(L)
L = []
for b in ["fa", "prs", "tg", "lgg", "ach", "teo"]:
    for s in ["qwen30a3b", "gptoss20", "qwen8", "aya", "dorna2", "fanar", "sunflower32q4", "sunflower9b"]:
        x = M.get(f"{b}|{s}")
        if x:
            L.append(f"{BN[b]} & {SYSN[s]} & {f3(x['rl_src'])} & {f3(x['rl_en'])} & {f3(x['bl_src'])} & {f3(x['bl_en'])} \\\\")
T["refm_tx_rows"] = "\n".join(L)

# ---- translation probe ---------------------------------------------------------------------------------------
TR = R["trans"]; TRR = R["trans_r"]
order = ["gptoss20", "qwen30a3b", "qwen8", "aya", "fanar", "dorna2", "sunflower9b", "sunflower32q4"]
L = ["\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
     "\\textbf{System} & " + " & ".join(f"\\textbf{{{BN[b]}}}" for b in B) + " \\\\", "\\midrule"]
for s in order:
    cells = []
    for b in B:
        x = TR.get(f"{b}|{s}")
        cells.append("--" if not x else f"{x['chrf']:.1f} / {x['F_en']:.2f}")
    L.append(SYSN[s] + " & " + " & ".join(cells) + " \\\\")
L += ["\\midrule", "$r$ across systems & " + " & ".join(f"{TRR[b]['r_en']:.2f}" for b in B) + " \\\\", "\\bottomrule", "\\end{tabular}"]
T["translation"] = "\n".join(L)
T["_pooled"] = TRR.get("pooled")
json.dump(T, open(os.path.join(HERE, "ow", "tables.json"), "w", encoding="utf-8"), indent=1)
for k, v in T.items():
    print(f"===== {k}\n{v}\n")
