"""Merge the Acholi/Ateso judge outputs into the per-build format of every other build (one file per system with both
input conditions), so the open-weight metric pipeline (eval/official/openweight) can treat them like Lugbara.
  source-input rows : out/nilo/<L>/gen.mlext.<sys>.faith.claude.jsonl        (verdicts vs the new source + vs English)
  English-input rows: out/nilo/<L>/gen.mlext.<sys>_enin.faith.claude.jsonl   (verdicts vs the new source)
                      + the English-reference verdicts of the same frozen claims from Lugbara (same English article)
  -> out/gen.{ach,teo}.<sys>.faith.claude.jsonl"""
import json, os
OUT = r"eval/official/out"
SYSTEMS = ["openai", "gptoss20", "qwen30a3b", "qwen8", "aya", "fanar", "sunflower9b", "sunflower32q4"]
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
for L in ("ach", "teo"):
    for s in SYSTEMS:
        lgg = {(r["id"], r["condition"]): r for r in rd(os.path.join(OUT, f"gen.mlext.{s}.faith.claude.jsonl"))}
        src = rd(os.path.join(OUT, "nilo", L, f"gen.mlext.{s}.faith.claude.jsonl"))
        enin = rd(os.path.join(OUT, "nilo", L, f"gen.mlext.{s}_enin.faith.claude.jsonl"))
        rows = []
        for r in src:
            assert r["condition"] == "dialect"; rows.append(r)
        for r in enin:
            base = lgg[(r["lgg_id"], "gold_ref")]
            assert (base.get("claims") or []) == (r.get("claims") or [])
            r = dict(r); r["faith"] = {"dialect": (r.get("faith") or {}).get("dialect"),
                                       "gold_reference": (base.get("faith") or {}).get("gold_reference")}
            rows.append(r)
        with open(os.path.join(OUT, f"gen.{L}.{s}.faith.claude.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(L, s, len(rows))
