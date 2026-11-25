"""Open-weight metric runs, step 1 (local, no model calls): gather the released articles and every existing summary +
frozen claims for the five builds into three compact files.

  ow/ow_articles.jsonl  {lang, id, dialect, src_sents, en_sents}                      -> VM
  ow/ow_rows.jsonl      {lang, system, id, condition, summary, claims}                -> VM
  ow/ow_judge.jsonl     {lang, system, id, condition, lab_src, lab_en, F_src, F_en}   (local only; the Claude verdicts)

Arabic = the released cj-loop articles judged in roster_gen (2,170 of 2,330); transfer = every judged article."""
import json, os, collections
ROOT = r"Documents\Work\lab\ArabicDetection"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ow")
os.makedirs(OUT, exist_ok=True)
SCORE = {"SUPPORTED": 1.0, "PARTIAL": 0.5}
BUILDS = [("ar", os.path.join(ROOT, "eval", "official", "out", "roster_gen", "gen.ours.{s}.rest.faith.claude.jsonl"),
           ["openai", "gptoss20", "qwen30a3b", "fanar", "qwen8", "aya"]),
          ("lgg", os.path.join(ROOT, "eval", "official", "out", "gen.mlext.{s}.faith.claude.jsonl"),
           ["openai", "gptoss20", "qwen30a3b", "qwen8", "aya", "fanar", "sunflower9b", "sunflower32q4"])] + \
         [(t, os.path.join(ROOT, "eval", "official", "out", f"gen.{t}.{{s}}.faith.claude.jsonl"),
           ["openai", "gptoss20", "qwen30a3b", "qwen8", "aya", "dorna2"]) for t in ("fa", "prs", "tg")] + \
         [(t, os.path.join(ROOT, "eval", "official", "out", f"gen.{t}.{{s}}.faith.claude.jsonl"),
           ["openai", "gptoss20", "qwen30a3b", "qwen8", "aya", "fanar", "sunflower9b", "sunflower32q4"]) for t in ("ach", "teo")]   # 2026-09-25
src = {}; dial = {}
for l in open(os.path.join(ROOT, "datasets", "official", "derived", "DiaBench", "diabench.jsonl"), encoding="utf-8"):
    r = json.loads(l); src[r["id"]] = r["source"]; dial[r["id"]] = r["dialect"]
def dgroup(d):
    for k in ("egyptian", "gulf", "levantine", "maghrebi"):
        if (d or "").lower().startswith(k): return k
    return d or ""
arts, rows, judge = {}, [], []
cnt = collections.Counter(); bad = collections.Counter()
for lang, pat, systems in BUILDS:
    for s in systems:
        for l in open(pat.format(s=s), encoding="utf-8"):
            r = json.loads(l)
            i = r["id"].replace("ours-", "", 1)
            if lang == "ar" and src.get(i) != "cj-loop":
                continue
            ss, es = r.get("source_sents") or [], r.get("gold_reference") or []
            if len(ss) != len(es) or not ss:
                bad[(lang, "misaligned")] += 1; continue
            key = (lang, i)
            if key in arts:
                if arts[key]["src_sents"] != ss or arts[key]["en_sents"] != es:
                    bad[(lang, "article differs across systems")] += 1
            else:
                arts[key] = {"lang": lang, "id": i, "dialect": dgroup(dial.get(i)) if lang == "ar" else lang,
                             "src_sents": ss, "en_sents": es}
            claims = r.get("claims") or []
            rows.append({"lang": lang, "system": s, "id": i, "condition": r["condition"],
                         "summary": r.get("summary") or "", "claims": claims})
            fa = r.get("faith") or {}
            a = (fa.get("dialect") or {}).get("verdicts") or []
            b = (fa.get("gold_reference") or {}).get("verdicts") or []
            ok = len(a) == len(b) == len(claims) and len(claims) > 0
            judge.append({"lang": lang, "system": s, "id": i, "condition": r["condition"],
                          "lab_src": [v.get("label") for v in a] if ok else None,
                          "lab_en": [v.get("label") for v in b] if ok else None,
                          "F_src": (fa.get("dialect") or {}).get("faithfulness_rate") if (fa.get("dialect") or {}).get("n_claims") else None,
                          "F_en": (fa.get("gold_reference") or {}).get("faithfulness_rate") if (fa.get("gold_reference") or {}).get("n_claims") else None})
            cnt[(lang, s)] += 1
with open(os.path.join(OUT, "ow_articles.jsonl"), "w", encoding="utf-8") as f:
    for a in arts.values(): f.write(json.dumps(a, ensure_ascii=False) + "\n")
with open(os.path.join(OUT, "ow_rows.jsonl"), "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(os.path.join(OUT, "ow_judge.jsonl"), "w", encoding="utf-8") as f:
    for r in judge: f.write(json.dumps(r, ensure_ascii=False) + "\n")
per_lang = collections.Counter(a["lang"] for a in arts.values())
print("articles per build:", dict(per_lang))
print("rows:", len(rows), "claims:", sum(len(r["claims"]) for r in rows))
print("rows per build/system:", {f"{k[0]}/{k[1]}": v for k, v in sorted(cnt.items())})
print("sentences per build:", {L: sum(len(a["src_sents"]) for a in arts.values() if a["lang"] == L) for L in per_lang})
print("skipped:", dict(bad))
print("judge rows with aligned verdicts:", sum(1 for j in judge if j["lab_src"] is not None), "of", len(judge))
