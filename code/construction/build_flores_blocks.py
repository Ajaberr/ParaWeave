#!/usr/bin/env python
"""Fragment-level pools for the FLORES-derived varieties (2026-09-17).

FLORES ships 3-4 consecutive sentences per source URL and nothing else of the article. Sentence-level
construction on these pools finds no scenes (2,009 unrelated sentences -> 2/1/0 articles), so the unit of
composition becomes the URL fragment: the composer selects and orders fragments, never touches their inside.
An article assembled from k fragments spans k source articles, so nothing pre-existing is recovered.

    python processing/official/build_flores_blocks.py <pool_dir> farsi dari tajik

Reads  <pool_dir>/poolfull_<tag>.jsonl      (sentence rows with _group_id = split:URL)
Writes <pool_dir>/poolfull_<tag>_blk.jsonl  (one row per fragment: ar/en joined with a space, plus the
                                             sentence lists in _sents_ar/_sents_en for splitting back)
"""
import json, sys, collections
from pathlib import Path

pool_dir, tags = Path(sys.argv[1]), sys.argv[2:]
for tag in tags:
    groups = collections.OrderedDict()
    for l in (pool_dir / f"poolfull_{tag}.jsonl").open(encoding="utf-8"):
        r = json.loads(l); groups.setdefault(r["_group_id"], []).append(r)
    out = pool_dir / f"poolfull_{tag}_blk.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for gid, rows in groups.items():
            fh.write(json.dumps({"ar": " ".join(r["ar"].strip() for r in rows), "en": " ".join(r["en"].strip() for r in rows),
                                 "sub": rows[0].get("sub", tag), "_group_id": gid, "_n_sents": len(rows),
                                 "_sents_ar": [r["ar"] for r in rows], "_sents_en": [r["en"] for r in rows]},
                                ensure_ascii=False) + "\n")
    n = sum(len(v) for v in groups.values())
    print(f"{tag:6s}: {n} sentences -> {len(groups)} fragments (mean {n/len(groups):.1f} sentences) -> {out.name}")
