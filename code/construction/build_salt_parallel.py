#!/usr/bin/env python
"""Acholi and Ateso versions of the 240 Lugbara benchmark articles (2026-09-25).

SALT is multi-way parallel: every English sentence was translated into each language, under one SALT id. So the
Lugbara articles need no new construction to exist in the other SALT languages: each Lugbara sentence is mapped to
its SALT id and replaced by that id's Acholi (ach) or Ateso (teo) translation, keeping the English side and the
sentence order. The three builds therefore share the English article exactly; only the source language changes.
Lugbara is Central Sudanic; Acholi (Western Nilotic) and Ateso (Eastern Nilotic) are its only relatives in SALT,
all three Nilo-Saharan in the traditional classification.

    python processing/official/build_salt_parallel.py
    -> datasets/official/derived/DiaBench-ML/{acholi,ateso}_eval.jsonl  (+ salt_parallel_report.json)

Source text: the full multi-way SALT text table (24,947 ids; 23,947 train / 500 dev / 500 test, the SALT splits) with
the columns ID, eng, lug, ach, teo, lgg, nyn. The official `Sunbird/salt` is gated, so it is read from the text columns
of the ungated `mekaneeky/Synthetic_Acholi_MMS` (a TTS set that carries the SALT text table unchanged), fetched once
with parquet column projection (no audio) into DiaBench-ML/_raw/salt_text_multiway.parquet; the smaller
`mekaneeky/salt-dataset` lacks about 1,070 ids and would drop 90 articles.
An article is kept only if every sentence maps to a SALT id whose Lugbara and English text match ours."""
import json, collections, difflib
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
DML = ROOT / "datasets" / "official" / "derived" / "DiaBench-ML"
RAW = DML / "_raw" / "salt_text_multiway.parquet"
norm = lambda t: " ".join(str(t or "").split())

def fetch_raw():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, HfApi
    fs, frames = HfFileSystem(), []
    for f in [s.rfilename for s in HfApi().dataset_info("mekaneeky/Synthetic_Acholi_MMS").siblings if s.rfilename.endswith(".parquet")]:
        with fs.open(f"datasets/mekaneeky/Synthetic_Acholi_MMS/{f}", "rb", block_size=2 ** 20) as fh:
            d = pq.ParquetFile(fh).read(columns=["ID", "eng", "lug", "ach", "teo", "lgg", "nyn"]).to_pandas()
        d["salt_split"] = f.split("/")[1].split("-")[0]; frames.append(d)
    pd.concat(frames, ignore_index=True).to_parquet(RAW)

if not RAW.exists(): fetch_raw()
salt = pd.read_parquet(RAW).rename(columns={"ID": "id", "eng": "eng_text", "ach": "ach_text", "teo": "teo_text", "lgg": "lgg_text"})
by_pair = collections.defaultdict(list)       # (lgg, eng) -> rows
by_eng = collections.defaultdict(list)
for r in salt.itertuples():
    by_pair[(norm(r.lgg_text), norm(r.eng_text))].append(r)
    by_eng[norm(r.eng_text)].append(r)
salt_rows = list(salt.itertuples())
pool = [json.loads(l) for l in open(DML / "poolfull_lugbara.jsonl", encoding="utf-8")]
pool_by_eng = collections.defaultdict(set)
for p in pool: pool_by_eng[norm(p["en"])].add(norm(p["ar"]))

arts = [json.loads(l) for l in open(DML / "lugbara_eval.jsonl", encoding="utf-8")]
out = {"ach": [], "teo": []}; stats = collections.Counter(); dropped = []; unmatched = []
for a in arts:
    rows = []
    for lg, en in zip(a["dialect_article"], a["gold_reference"]):
        hits = by_pair.get((norm(lg), norm(en)))
        if hits: stats["exact lgg+eng"] += 1
        else:
            # the composer occasionally dropped or changed a Lugbara word when copying (29 sentences): fall back on
            # the English side, accepting it only if the English sentence is unique in SALT
            hits = by_eng.get(norm(en))
            if hits and len({norm(h.lgg_text) for h in hits}) == 1: stats["english only (unique)"] += 1
            else:
                # last resort: the SALT row (same English if any, else any row) whose Lugbara is closest to ours;
                # accepted only when nearly identical (a copy slip), never a different sentence
                cands = hits or salt_rows
                best = max(cands, key=lambda h: difflib.SequenceMatcher(None, norm(h.lgg_text), norm(lg)).ratio())
                sim = difflib.SequenceMatcher(None, norm(best.lgg_text), norm(lg)).ratio()
                en_sim = difflib.SequenceMatcher(None, norm(best.eng_text), norm(en)).ratio()
                if sim >= 0.85 and en_sim >= 0.85: hits = [best]; stats["closest Lugbara (>=0.85)"] += 1
                else: hits = None; stats["unmatched"] += 1; unmatched.append((a["id"], lg, en, round(sim, 2), round(en_sim, 2)))
        rows.append(hits[0] if hits else None)
    if any(r is None for r in rows) or any(not norm(getattr(r, f"{c}_text")) for r in rows if r is not None for c in ("ach", "teo")):
        dropped.append(a["id"]); continue
    for code, name in (("ach", "acholi"), ("teo", "ateso")):
        out[code].append({"id": a["id"].replace("lgg-", f"{code}-"), "dialect": "", "lang": code,
                          "dialect_article": [norm(getattr(r, f"{code}_text")) for r in rows],
                          "gold_reference": a["gold_reference"], "source": f"salt-parallel-{name}",
                          "lgg_id": a["id"], "salt_ids": [str(r.id) for r in rows],
                          "salt_splits": [r.salt_split for r in rows], "tier": a.get("tier")})
for code, name in (("ach", "acholi"), ("teo", "ateso")):
    with open(DML / f"{name}_eval.jsonl", "w", encoding="utf-8") as fh:
        for x in out[code]: fh.write(json.dumps(x, ensure_ascii=False) + "\n")
n_sent = sum(stats.values())
rep = {"articles_in": len(arts), "articles_out": len(out["ach"]), "dropped": dropped, "sentence_matching": dict(stats),
       "unmatched": unmatched,
       "splits": dict(collections.Counter(s for x in out["ach"] for s in x["salt_splits"])),
       "mean_words": {c: round(sum(len(" ".join(x["dialect_article"]).split()) for x in out[c]) / max(1, len(out[c])), 1)
                      for c in out},
       "mean_words_lgg": round(sum(len(" ".join(a["dialect_article"]).split()) for a in arts) / len(arts), 1),
       "mean_words_en": round(sum(len(" ".join(a["gold_reference"]).split()) for a in arts) / len(arts), 1)}
json.dump(rep, open(DML / "salt_parallel_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
print("example ach:", out["ach"][0]["dialect_article"][:2]); print("example teo:", out["teo"][0]["dialect_article"][:2])
