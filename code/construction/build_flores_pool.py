#!/usr/bin/env python
"""Stage flat, disconnected Farsi/Dari/Tajik pools for DiaBench's construction-pipeline transfer test,
sourced from the FLORES-200 tarball already downloaded locally (datasets/official/derived/DiaBench-ML/_raw/
flores200_dataset/, via processing/build_ml_extension.py's ungated-download discovery).

WHY THIS EXISTS (2026-08-13, per the author): after confirming Dari has NO genuinely-disconnected,
English-paired corpus anywhere (checked UD, BTEC/IWSLT, OpenSubtitles, eBible, Tatoeba -- all confirmed
absent; even the one 2026 paper written specifically about Farsi/Dari/Tajik fell back to FLORES-200 for
its own Dari eval), the author proposed a fix: instead of using FLORES's `url` field to reconstruct its
ALREADY-coherent source articles (which is what build_ml_extension.py does, and which doesn't test the
construction method -- see that script's history note), POOL sentences across ALL of a language's
~562 small `url`-groups (dev+devtest combined, avg 3.6 sentences/group) into ONE flat, undifferentiated
pool, exactly mirroring the trick already used for Naija (build_naija_pool.py flattens across 88
recordings the same way). Once the `url` signal is discarded for clustering purposes, it doesn't matter
that the sentences once belonged to a coherent article -- the construction pipeline has to rediscover
ANY structure from scratch, which is a fair, honest test. This ALSO solves the Dari problem for free: no
new external corpus needed, the data was already sitting on disk.

Each sentence keeps a `_group_id` field (its original FLORES `url`) even though the pool is flat --
this is NOT used for the sentence-level clustering test, but enables a SECOND variant the author also wants
tested: "group-to-group" clustering, where each small already-coherent FLORES group is kept intact as
one atomic chunk and whole chunks get clustered/merged into bigger composite articles, rather than
individual sentences. Both variants read this same pool file; build_ml_construction.py's granularity
flag decides which one runs. (Naija's pool also carries an analogous `_group_id` = `_orig_recording`,
so both pools support both granularities through the same code path.)

  python processing/build_flores_pool.py

Writes datasets/DiaBench-ML/poolfull_{farsi,dari,tajik}.jsonl -- schema {ar, en, sub, _group_id},
matching Naija's pool format (the `ar` key name is inherited from the original Arabic-only pipeline
code, reused UNCHANGED here -- it now just holds Farsi/Dari/Tajik text).
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLORES_DIR = ROOT / "datasets" / "DiaBench-ML" / "_raw" / "flores200_dataset"
OUT_DIR = ROOT / "datasets" / "DiaBench-ML"

VARIETIES = {"farsi": "pes_Arab", "dari": "prs_Arab", "tajik": "tgk_Cyrl"}
EN_CODE = "eng_Latn"


def load_split(code: str, split: str) -> tuple[list[dict], list[str]]:
    with (FLORES_DIR / f"metadata_{split}.tsv").open(encoding="utf-8") as f:
        meta = list(csv.DictReader(f, delimiter="\t"))
    src = (FLORES_DIR / split / f"{code}.{split}").read_text(encoding="utf-8").splitlines()
    en = (FLORES_DIR / split / f"{EN_CODE}.{split}").read_text(encoding="utf-8").splitlines()
    assert len(src) == len(meta) and len(en) == len(meta)
    return meta, list(zip(src, en))


def main():
    for tag, code in VARIETIES.items():
        pool = []
        for split in ("dev", "devtest"):
            meta, pairs = load_split(code, split)
            for m, (a, e) in zip(meta, pairs):
                if a.strip() and e.strip():
                    pool.append({"ar": a.strip(), "en": e.strip(), "sub": tag,
                                 "_group_id": f"{split}:{m['URL']}"})
        n_groups = len({p["_group_id"] for p in pool})
        out = OUT_DIR / f"poolfull_{tag}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for p in pool:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"{tag:6s} ({code}): {len(pool)} sentences across {n_groups} original url-groups -> {out}")


if __name__ == "__main__":
    main()
