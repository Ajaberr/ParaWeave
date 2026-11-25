"""Acholi/Ateso judging inputs (2026-09-25). The English-input summaries of the Lugbara run are reused: the English
article is identical, so each system's English-input summaries, their frozen claims and their English-reference
verdicts carry over unchanged, and only the check against the new source article is new.

  python eval/official/nilo/nilo_prep.py
  -> eval/official/out/nilo/{ach,teo}/gen.mlext.<sys>_enin.jsonl + claims.mlext.<sys>_enin.jsonl (frozen, reused)
Source-input summaries (VM / GPT-5.5) are placed at eval/official/out/nilo/{ach,teo}/gen.mlext.<sys>.jsonl."""
import json, os
ROOT = r"Documents\Work\lab\ArabicDetection"
OUT = os.path.join(ROOT, "eval", "official", "out")
DML = os.path.join(ROOT, "datasets", "official", "derived", "DiaBench-ML")
SYSTEMS = ["openai", "gptoss20", "qwen30a3b", "qwen8", "fanar", "aya", "sunflower9b", "sunflower32q4"]
LANGS = {"ach": "acholi_eval.jsonl", "teo": "ateso_eval.jsonl"}
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]

for L, f in LANGS.items():
    arts = {a["lgg_id"]: a for a in rd(os.path.join(DML, f))}
    d = os.path.join(OUT, "nilo", L); os.makedirs(d, exist_ok=True)
    for s in SYSTEMS:
        gen = rd(os.path.join(OUT, f"gen.mlext.{s}.jsonl")); cl = rd(os.path.join(OUT, f"claims.mlext.{s}.jsonl"))
        assert len(gen) == len(cl) and all(g["id"] == c["id"] for g, c in zip(gen, cl)), s
        rows, claims = [], []
        for g, c in zip(gen, cl):
            if g["condition"] != "gold_ref": continue
            a = arts[g["id"]]
            assert a["gold_reference"] == g["gold_reference"], (s, g["id"])      # the same English article
            r = dict(g); r["id"] = a["id"]; r["lgg_id"] = g["id"]; r["reused_from"] = "lugbara"
            r["source_sents"] = a["dialect_article"]; r["source_text"] = " ".join(a["dialect_article"])
            rows.append(r); claims.append({"id": a["id"], "claims": c["claims"]})
        with open(os.path.join(d, f"gen.mlext.{s}_enin.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(d, f"claims.mlext.{s}_enin.jsonl"), "w", encoding="utf-8") as fh:
            for c in claims: fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(L, s, len(rows), "English-input summaries,", sum(1 for c in claims if c["claims"]), "with claims,",
              sum(len(c["claims"]) for c in claims), "claims")
