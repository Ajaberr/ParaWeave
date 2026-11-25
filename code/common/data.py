#!/usr/bin/env python
"""Data adapters -> one common eval record for both corpora.

record = {
  id, corpus, dialect,
  source_sents : list[str],      # the Arabic source, sentence-split (dialect for ours; MSA for crosssum)
  source_text  : str,            # the source joined
  gold_reference: list[str]|None,# OUR SET ONLY: parallel human English/MSA translation, per sentence
  gold_summary : list[str]|None, # OUR SET ONLY: extractive gold summary sentences
  en_reference : str|None,       # CROSSSUM ONLY: the human abstractive English summary
}
"""
from __future__ import annotations
import json, os, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OURS = Path(os.environ.get("OURS_FILE",
            ROOT / "processing" / "out" / "useful" / "coherent_dataset.v6.sample.jsonl"))


def load_ours(limit: int | None = None) -> list[dict]:
    recs = []
    for i, line in enumerate(OURS.open(encoding="utf-8")):
        if limit and i >= limit:
            break
        a = json.loads(line)
        recs.append({
            "id": f"ours-{a.get('from_id', i)}",
            "corpus": "ours",
            "dialect": a["dialect"],
            "source_sents": a["dialect_article"] if isinstance(a["dialect_article"], list) else [a["dialect_article"]],
            "source_text": " ".join(a["dialect_article"]) if isinstance(a["dialect_article"], list) else str(a["dialect_article"]),
            "gold_reference": a.get("gold_reference") or [],
            "gold_summary": a.get("gold_summary") or [],                  # v5 has none -> faithfulness eval only
            "en_reference": " ".join(a.get("gold_summary") or []),        # empty for v5 (ref-metrics n/a; faithfulness ok)
            "sub": a.get("sub"), "tier": a.get("tier"), "source": a.get("source"),
        })
    return recs


BAGS = ROOT / "processing" / "out" / "useful" / "summ.bags.jsonl"   # build-time NON-summarizable (held-out)


def load_bags(limit: int | None = None) -> list[dict]:
    """The articles flagged summarizable=False at build time -> the known-negative group for summarizability."""
    recs = []
    for i, line in enumerate(BAGS.open(encoding="utf-8")):
        if limit and i >= limit:
            break
        a = json.loads(line)
        recs.append({
            "id": f"bags-{a.get('from_id', i)}", "corpus": "bags", "dialect": a["dialect"],
            "source_sents": a["dialect_article"], "source_text": " ".join(a["dialect_article"]),
            "gold_reference": a["gold_reference"], "gold_summary": a["gold_summary"],
            "en_reference": " ".join(a["gold_summary"]),
        })
    return recs


ML_EXTENSION = (Path(os.environ["ML_EXTENSION_FILE"]) if os.environ.get("ML_EXTENSION_FILE")   # 2026-09-17: any mlext-shaped file
                else ROOT / "datasets" / "DiaBench-ML" / "ml_extension.jsonl")


def load_ml_extension(limit: int | None = None) -> list[dict]:
    """Small multilingual proof-of-transfer set: Farsi/Dari/Tajik + Fon (processing/build_ml_extension.py).
    Same record shape as load_ours() -- `dialect` here holds the variety tag (farsi/dari/tajik/fon), not
    an Arabic dialect."""
    recs = []
    for i, line in enumerate(ML_EXTENSION.open(encoding="utf-8")):
        if limit and i >= limit:
            break
        a = json.loads(line)
        recs.append({
            "id": a["id"], "corpus": "mlext", "dialect": a["dialect"],
            "source_sents": a["dialect_article"], "source_text": " ".join(a["dialect_article"]),
            "gold_reference": a.get("gold_reference") or [], "gold_summary": [],
            "en_reference": "",   # no abstractive gold summary -- faithfulness eval only, same as "ours"
            "source": a.get("source"), "tier": a.get("tier"),
        })
    return recs


def load_crosssum(split: str = "train", limit: int | None = 500, seed: int = 0) -> list[dict]:
    """Arabic(MSA) article -> English human summary. Sampled (stable) for the quality leaderboard."""
    arc = ROOT / "datasets" / "cross_lingual" / "crosssum" / "data" / "arabic-english_CrossSum.tar.bz2"
    rows = []
    with tarfile.open(arc) as t:
        member = next(m for m in t.getmembers() if m.name.endswith(f"{split}.jsonl"))
        f = t.extractfile(member)
        for line in f:
            r = json.loads(line)
            rows.append(r)
    if limit and len(rows) > limit:
        import random
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    out = []
    for i, r in enumerate(rows):
        text = r.get("text", "")
        out.append({
            "id": f"crosssum-{i}",
            "corpus": "crosssum",
            "dialect": "msa",
            "source_sents": _split_ar(text),
            "source_text": text,
            "gold_reference": None,
            "gold_summary": None,
            "en_reference": r.get("summary", ""),
        })
    return out


def _split_ar(text: str) -> list[str]:
    import re
    parts = re.split(r"(?<=[\.\!\?؟۔])\s+", (text or "").strip())
    return [p for p in parts if p.strip()]


def load_records(corpus: str, **kw) -> list[dict]:
    if corpus == "ours":
        return load_ours(**kw)
    if corpus == "bags":
        return load_bags(**kw)
    if corpus == "crosssum":
        return load_crosssum(**kw)
    if corpus == "mlext":
        return load_ml_extension(**kw)
    raise ValueError(corpus)


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "ours"
    recs = load_records(c, limit=3)
    print(f"{c}: {len(recs)} sample records; keys={list(recs[0].keys())}")
    for r in recs:
        print(f"  {r['id']} [{r['dialect']}] {len(r['source_sents'])} sents | ref={r['en_reference'][:60]!r}")
