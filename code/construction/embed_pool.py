#!/usr/bin/env python
"""Build the cached English-side embeddings loop_assemble.py clusters on.

    python processing/official/embed_pool.py <pool_dir> <tag> [<tag> ...]

Reads  <pool_dir>/poolfull_<tag>.jsonl   (schema {ar, en, ...}; `ar` = the low-resource side, any language)
Writes <pool_dir>/emb_poolfull_<tag>.npy (float32, L2-normalised, one row per pool line, same order)

Production settings, identical to the Arabic SILVER build and the CrossSum validation:
  model        Qwen/Qwen3-Embedding-8B (local, GPU; ~16 GB in fp16)
  side         English (`en`) -- the high-resource side, never the source
  instruction  "Instruct: Group sentences that belong to the same conversation, scene, or connected event.\nQuery: "
PCA-50 is applied inside loop_assemble.py per round, so the full 4096-d vectors are stored here.
Env: EMBED_DEVICE (cuda:0), EMBED_BATCH (16), EMBED_DTYPE (float16).
"""
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cluster import STEmbedder

SCENE = "Instruct: Group sentences that belong to the same conversation, scene, or connected event.\nQuery: "

def main():
    pool_dir, tags = Path(sys.argv[1]), sys.argv[2:]
    assert tags, "give at least one pool tag, e.g. farsi"
    emb = STEmbedder("qwen3-8b", device=os.environ.get("EMBED_DEVICE", "cuda:0"),
                     batch_size=int(os.environ.get("EMBED_BATCH", "16")), instruction=SCENE,
                     dtype=os.environ.get("EMBED_DTYPE", "float16"))
    for tag in tags:
        src = pool_dir / f"poolfull_{tag}.jsonl"; dst = pool_dir / f"emb_poolfull_{tag}.npy"
        rows = [json.loads(l) for l in src.open(encoding="utf-8")]
        texts = [(r.get("en") or "").strip() for r in rows]
        empty = sum(1 for t in texts if not t)
        if empty:
            print(f"[{tag}] WARNING {empty} rows have an empty English side; they get a zero-ish embedding")
        E = emb.encode(texts).astype(np.float32)
        assert E.shape[0] == len(rows), (E.shape, len(rows))
        np.save(dst, E)
        print(f"[{tag}] {len(rows)} sentences -> {dst} {E.shape}", flush=True)

if __name__ == "__main__":
    main()
