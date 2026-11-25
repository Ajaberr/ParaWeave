"""Broad-topic clustering for the faithfulness / hallucination dataset.

Pipeline:
  1. pick a representative text per record by LANGUAGE PROMINENCE: English -> MSA -> dialect
     (highest-resource available language gives the cleanest topic signal),
  2. embed with LaBSE (multilingual, language-agnostic -> en/MSA/dialect land in one space),
  3. MiniBatchKMeans into K broad topics,
  4. label topics by their centroid-nearest exemplars; save assignments.

Heavy deps (numpy, scikit-learn, sentence-transformers/torch) are imported lazily, so this module
loads without them; you only need them at run time. Pure-ASCII source.

    from loaders import load_crosssum
    from cluster import cluster_records, save_clusters
    res = cluster_records(load_crosssum("arabic-english", "train"), n_topics=40)
    save_clusters(res, "processing/out/topics.crosssum.jsonl")
"""
from __future__ import annotations

import os

# Big multi-core servers (>128 cores) segfault OpenBLAS/sklearn when BLAS spawns more threads
# than it was built for. Cap BLAS threads before numpy/sklearn import (GPU embedding is unaffected).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "64")

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

LANG_PRIORITY = ("en", "msa", "dialect")
# Embedding models for multilingual (en/MSA/dialect) TOPIC clustering of full articles:
#   bge-m3    568M, 8192-ctx, strong Arabic  -> DEFAULT (best speed/quality/long-doc balance)
#   qwen3-8b  8B, SOTA multilingual, 32k-ctx -> max quality, needs ~16GB GPU, ~10-15x slower
#   labse     470M, 256-ctx ONLY             -> older; truncates long articles -> weaker topics
MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
    "labse": "sentence-transformers/LaBSE",
}
DEFAULT_MODEL = "bge-m3"


# -- 1. representative text by language prominence ----------------------------
def representative(by_lang: dict, priority=LANG_PRIORITY) -> tuple[str, str]:
    """Given {lang: text}, return (text, lang) for the highest-priority non-empty language."""
    for lang in priority:
        t = by_lang.get(lang)
        if t and t.strip():
            return t.strip(), lang
    return "", ""


def langs_from(rec) -> dict:
    """Map a loader record (SummPair / NormPair / DetItem) to {lang: text} using field
    conventions. en = English side, msa = Modern Standard Arabic, dialect = dialectal Arabic."""
    d: dict[str, str] = {}
    # NormPair: dialect_text / msa_text / english
    if hasattr(rec, "msa_text"):
        if getattr(rec, "english", None):
            d["en"] = rec.english
        if rec.msa_text:
            d["msa"] = rec.msa_text
        if rec.dialect_text:
            d["dialect"] = rec.dialect_text
        return d
    # SummPair: article (src_lang) + summary (tgt_lang); dialect set for dialectal summ
    if hasattr(rec, "article"):
        if getattr(rec, "tgt_lang", "") == "en" and rec.summary:
            d["en"] = rec.summary
        art_lang = "dialect" if getattr(rec, "dialect", None) else \
                   ("msa" if rec.src_lang == "ar" else rec.src_lang)
        if rec.article:
            d.setdefault(art_lang, rec.article)
        if getattr(rec, "tgt_lang", "") == "ar" and rec.summary:
            d.setdefault("dialect" if getattr(rec, "dialect", None) else "msa", rec.summary)
        return d
    # DetItem: just text; language unknown -> lowest tier
    if hasattr(rec, "text"):
        d["dialect"] = rec.text
    return d


# -- 2. embedding (LaBSE) -----------------------------------------------------
class STEmbedder:
    """SentenceTransformer embeddings (L2-normalized), for ANY HF model. Default BAAI/bge-m3.
    `model` accepts a MODELS key ('bge-m3' / 'qwen3-8b' / 'labse') or a full HF id.
    Requires sentence-transformers + torch (runs on GPU automatically if available)."""

    def __init__(self, model: str = DEFAULT_MODEL, device: Optional[str] = None,
                 batch_size: int = 64, trust_remote_code: bool = True,
                 instruction: Optional[str] = None, dtype: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        kw = {"model_kwargs": {"torch_dtype": dtype}} if dtype else {}   # e.g. float16 for 8B models
        self.model = SentenceTransformer(MODELS.get(model, model), device=device,
                                         trust_remote_code=trust_remote_code, **kw)
        self.batch_size = batch_size
        self.instruction = instruction   # instruction-tuned models (Qwen3-Embedding, e5-instruct)

    def encode(self, texts: list[str]):
        # `prompt` is prepended per text; for Qwen3 pass "Instruct: <task>\nQuery: ". None = no-op.
        return self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=True,
                                 prompt=self.instruction)


# -- 3. + 4. clustering + result ----------------------------------------------
@dataclass
class ClusterResult:
    texts: list[str]
    langs: list[str]
    sources: list[str]
    labels: list[int]          # topic id per item
    embeddings: object         # np.ndarray (n, d)
    centroids: object          # np.ndarray (k, d)

    def exemplars(self, topic: int, k: int = 5) -> list[str]:
        """The k texts whose embeddings are closest to a topic's centroid (a topic 'label')."""
        import numpy as np
        idx = [i for i, t in enumerate(self.labels) if t == topic]
        if not idx:
            return []
        sims = self.embeddings[idx] @ self.centroids[topic]
        order = np.argsort(-sims)[:k]
        return [self.texts[idx[o]] for o in order]

    def ordered_members(self, topic: int, *, dedup_cos: float = 0.9,
                        how: str = "chain") -> list[str]:
        """Members of `topic` arranged into a readable order for article assembly:
          0. drop near-duplicates -- scanning most-central first, skip any member with cosine >
             dedup_cos to one already kept (the data is repetitive; this stops 'X / X-reworded /
             X-again' runs that pure similarity ordering would line up adjacently);
          1. lead with the most central kept member (a natural topic sentence);
          2. how='chain': greedy nearest-neighbour walk -- each next member is the unused one most
             similar to the LAST placed, giving smooth local transitions; how='centroid': simply
             by centroid similarity.
        Returns the ordered list of texts (length <= cluster size, after de-duplication)."""
        import numpy as np
        idx = [i for i, t in enumerate(self.labels) if t == topic]
        if not idx:
            return []
        sub = np.asarray(self.embeddings, dtype=np.float64)[idx]
        sub = sub / np.maximum(np.linalg.norm(sub, axis=1, keepdims=True), 1e-9)
        cen = np.asarray(self.centroids[topic], dtype=np.float64)
        cen = cen / max(float(np.linalg.norm(cen)), 1e-9)
        to_cen = sub @ cen
        # 0. near-duplicate removal, central-first so the kept copy is the most representative
        kept: list[int] = []
        for c in np.argsort(-to_cen):
            c = int(c)
            if all(float(sub[c] @ sub[k]) <= dedup_cos for k in kept):
                kept.append(c)
        if how == "centroid" or len(kept) <= 2:
            return [self.texts[idx[c]] for c in kept]
        # 1.-2. central lead, then greedy similarity chain over the kept set
        sims = sub @ sub.T
        seed = max(kept, key=lambda c: to_cen[c])
        order, remaining = [seed], set(kept) - {seed}
        while remaining:
            last = order[-1]
            nxt = max(remaining, key=lambda c: sims[last, c])
            order.append(nxt)
            remaining.discard(nxt)
        return [self.texts[idx[c]] for c in order]

    def sizes(self) -> dict[int, int]:
        from collections import Counter
        return dict(Counter(self.labels))

    def separation(self) -> dict:
        """Cluster-quality metrics. Good clustering => inter-cluster distance > intra-cluster.
        Centroid-based (simplified silhouette) so it scales to large n (O(n*k) memory).
          inter_over_intra > 1   : nearest OTHER centroid is farther than own centroid (good)
          silhouette_approx ~ 1  : tight, well-separated topics ; ~0 overlapping ; <0 bad
        """
        import numpy as np
        emb = np.asarray(self.embeddings, dtype=np.float64)
        cen = np.asarray(self.centroids, dtype=np.float64)
        lab = np.asarray(self.labels)
        n, k = len(lab), len(cen)
        if k < 2:
            return {"clusters": k}
        e2 = (emb * emb).sum(1)[:, None]
        c2 = (cen * cen).sum(1)[None, :]
        dist = np.sqrt(np.maximum(e2 + c2 - 2.0 * (emb @ cen.T), 0.0))   # (n,k) point->centroid
        rows = np.arange(n)
        intra = dist[rows, lab]
        other = dist.copy()
        other[rows, lab] = np.inf
        nearest_inter = other.min(1)
        sil = (nearest_inter - intra) / np.maximum(intra, nearest_inter)
        cc = np.sqrt(np.maximum(c2 + c2.T - 2.0 * (cen @ cen.T), 0.0))
        return {
            "clusters": k,
            "mean_intra": round(float(intra.mean()), 4),
            "mean_nearest_inter": round(float(nearest_inter.mean()), 4),
            "inter_over_intra": round(float(nearest_inter.mean() / max(intra.mean(), 1e-9)), 3),
            "mean_centroid_dist": round(float(cc[np.triu_indices(k, 1)].mean()), 4),
            "silhouette_approx": round(float(np.nanmean(sil)), 3),
        }


def cluster_records(records: Iterable, n_topics: int = 40, *, embedder=None,
                    priority=LANG_PRIORITY, limit: Optional[int] = None,
                    dedup: bool = True, seed: int = 0,
                    target_size: Optional[int] = None) -> ClusterResult:
    """Pick representative text (by prominence) -> [dedup] -> embed -> MiniBatchKMeans into
    n_topics broad topics. `dedup` (default True) drops duplicate representatives BEFORE
    embedding -- important because MADAR shares one MSA sentence set across all 25 city files,
    so without it MADAR's repeats would swamp the topics (and waste embedding compute).
    `limit` caps records (CPU embedding is slow; sample first).
    `target_size`: if set, K is chosen so clusters average ~target_size items (K = unique/size,
    computed AFTER dedup) -- use this for article-sized clusters (e.g. 18 sentences/article)
    instead of a fixed broad-topic count. To cluster the raw dialect text, pass
    priority=("dialect",) so representative() returns dialect_text."""
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    texts, langs, sources = [], [], []
    seen: set[str] = set()
    dkey = None
    if dedup:
        from clean import dedup_key as dkey
    for i, r in enumerate(records):
        if limit and len(texts) >= limit:
            break
        text, lang = representative(langs_from(r), priority)
        if not text:
            continue
        if dkey is not None:
            k = dkey(text)
            if not k or k in seen:
                continue
            seen.add(k)
        texts.append(text)
        langs.append(lang)
        sources.append(getattr(r, "source", "?"))
    if not texts:
        raise ValueError("no representative texts extracted")

    if target_size:
        n_topics = max(1, round(len(texts) / target_size))
    emb = (embedder or STEmbedder()).encode(texts)
    k = min(n_topics, len(texts))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto", batch_size=2048)
    labels = km.fit_predict(emb).tolist()
    return ClusterResult(texts, langs, sources, labels, np.asarray(emb), km.cluster_centers_)


# -- persistence --------------------------------------------------------------
def save_clusters(res: ClusterResult, path: str) -> int:
    """Write one jsonl row per item: {topic, lang, source, text}. Returns row count."""
    import json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for t, lang, src, lab in zip(res.texts, res.langs, res.sources, res.labels):
            fh.write(json.dumps({"topic": lab, "lang": lang, "source": src, "text": t},
                                ensure_ascii=False) + "\n")
    return len(res.texts)


def topic_report(res: ClusterResult, k_exemplars: int = 4) -> str:
    """Human-readable summary: each topic's size, language mix, and exemplar texts."""
    from collections import Counter
    lines = [f"# separation: {res.separation()}",
             "#   (inter_over_intra>1 and silhouette_approx->1 = well-separated topics)", ""]
    sizes = res.sizes()
    for topic in sorted(sizes, key=lambda t: -sizes[t]):
        langmix = Counter(res.langs[i] for i, t in enumerate(res.labels) if t == topic)
        lines.append(f"# topic {topic}  (n={sizes[topic]}, langs={dict(langmix)})")
        for ex in res.exemplars(topic, k_exemplars):
            lines.append("   - " + ex[:120].replace("\n", " "))
    return "\n".join(lines)


# -- coherence ranking -> article-candidate shortlist -------------------------
def cluster_coherence(res: ClusterResult) -> dict:
    """Per-cluster tightness, for ranking article candidates. Embeddings are L2-normalized, so
    cosine-to-own-centroid measures how tightly members hug their topic (1 = identical direction);
    `margin` subtracts the cosine to the NEAREST OTHER centroid, so a high-margin cluster is both
    tight AND well-isolated from its neighbours. Returns {topic: {size, coherence, margin, ...}}."""
    import numpy as np
    emb = np.asarray(res.embeddings, dtype=np.float64)
    cen = np.asarray(res.centroids, dtype=np.float64)
    lab = np.asarray(res.labels)
    cn = cen / np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-9)
    en = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    cc = cn @ cn.T
    np.fill_diagonal(cc, -1.0)
    nearest = cc.max(1)                                    # cosine to closest rival centroid
    out = {}
    for t in range(len(cen)):
        idx = np.where(lab == t)[0]
        if len(idx) == 0:
            continue
        coh = float((en[idx] @ cn[t]).mean())
        out[t] = {"size": int(len(idx)), "coherence": round(coh, 4),
                  "nearest_centroid_cos": round(float(nearest[t]), 4),
                  "margin": round(coh - float(nearest[t]), 4)}
    return out


def rank_clusters(res: ClusterResult, *, size_band=None, by: str = "coherence", top_n=None):
    """Rank clusters tightest-first by `by` (coherence | margin), keeping only those whose size is
    in size_band so singletons and degenerate mega-blobs are dropped (they aren't article-sized).
    size_band defaults to (median/3 .. median*3) so it adapts whether clusters average ~18 or
    ~2000. Returns (rows, stats) where rows = [(topic, metrics), ...]."""
    import statistics
    coh = cluster_coherence(res)
    sizes = sorted(m["size"] for m in coh.values())
    if size_band is None:
        med = statistics.median(sizes) if sizes else 18
        size_band = (max(3, round(med / 3)), max(round(med * 3), round(med) + 5))
    lo, hi = size_band
    kept, small, big = [], 0, 0
    for t, m in coh.items():
        if m["size"] < lo:
            small += 1
        elif m["size"] > hi:
            big += 1
        else:
            kept.append((t, m))
    kept.sort(key=lambda kv: -kv[1][by])
    if top_n:
        kept = kept[:top_n]
    return kept, {"kept": len(kept), "dropped_small": small, "dropped_big": big,
                  "size_band": [lo, hi], "ranked_by": by}


def save_shortlist(res: ClusterResult, path: str, *, size_band=None, by: str = "margin",
                   top_n=None, group=None, dedup_cos: float = 0.9) -> dict:
    """Write a ranked shortlist of article-candidate clusters. Each jsonl row carries the cluster's
    members in `texts` already de-duplicated and chain-ordered (see ordered_members) -> ready to
    read/assemble as an article -- plus coherence stats, source/lang mix, and `article_len` /
    `dropped_dups` (how many near-duplicates the de-dup removed; high values flag a repetitive
    cluster that looks coherent only because it repeats itself). Also writes a sibling .md preview
    of the top 25. Returns drop/keep stats (logged, never silently truncated)."""
    import json
    from collections import Counter
    rows, stats = rank_clusters(res, size_band=size_band, by=by, top_n=top_n)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    md = [f"# article-candidate shortlist  ({group or 'all'})",
          f"# ranked by {by}; size_band={stats['size_band']}; kept {stats['kept']}, "
          f"dropped {stats['dropped_small']} too-small + {stats['dropped_big']} too-big",
          f"# texts are near-dup-removed (cos>{dedup_cos}) + chain-ordered; "
          f"high dropped_dups = repetitive cluster",
          f"# corpus separation: {res.separation()}", ""]
    with open(p, "w", encoding="utf-8") as fh:
        for rank, (t, m) in enumerate(rows, 1):
            members = res.ordered_members(t, dedup_cos=dedup_cos)   # de-dup'd + chain-ordered
            srcs = dict(Counter(res.sources[i] for i, l in enumerate(res.labels) if l == t))
            lgs = dict(Counter(res.langs[i] for i, l in enumerate(res.labels) if l == t))
            fh.write(json.dumps({"rank": rank, "cluster": t, "group": group, **m,
                                 "article_len": len(members),
                                 "dropped_dups": m["size"] - len(members),
                                 "sources": srcs, "langs": lgs,
                                 "exemplars": members[:3], "texts": members},
                                ensure_ascii=False) + "\n")
            if rank <= 25:
                md.append(f"## #{rank}  cluster {t}  (article_len={len(members)} of n={m['size']}, "
                          f"coh={m['coherence']}, margin={m['margin']}, srcs={srcs}, langs={lgs})")
                for s in members[:6]:
                    md.append("   - " + s[:140].replace("\n", " "))
                md.append("")
    p.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    return stats
