#!/usr/bin/env python
"""AUC on REAL FLORES-200 articles -- the parallel counterpart of crosssum_diagnostic.py.

WHY THIS EXISTS. The CrossSum control has real article boundaries but its Arabic is MSA, so it can only
validate the English side; the claim that the low-resource side is no better than character overlap
(0.4421 vs 0.4400) rests on the pipeline's own articles. FLORES-200 keeps the source URL of every
sentence (3-4 consecutive sentences per URL; 562 URL fragments over dev+devtest) and carries the SAME
sentences in the same order in every language. So the boundaries are external (nobody in this project
chose them) and identical across languages: Persian, Dari and Tajik -- the languages of the FLORES
builds -- and the FLORES Arabic varieties can be scored on the very articles English is scored on.

PROTOCOL = crosssum_diagnostic.py, except that FLORES is small enough to enumerate every pair instead of
sampling 60k: a positive pair is two sentences of the same URL fragment; negatives are two sentences of
different fragments under four regimes --
  random         every cross-fragment pair;
  topic          cross-fragment pairs sharing FLORES's `topic` field (the analogue of the BBC section,
                 kept to show that a coarse label makes the negatives easy);
  lexical_topk   each fragment against its ten most similar fragments by fragment-level char TF-IDF on
                 THAT SIDE's own text (the CrossSum hard negatives; drives the lexical baseline toward
                 chance by construction, so gaps are comparable);
  lexical_topk_en  the same neighbours mined on English and used for every side (identical pairs in
                 every language, so raw AUC is comparable across languages too).
Lexical baseline = char_wb (2,4) TF-IDF cosine on the same pairs. Embedding = Qwen3-Embedding-8B in
fp16 (the production embedder, processing/official/embed_pool.py), full 4096-d, arms none/topic/scene
with the instruction prepended exactly as in the CrossSum run. Compare GAPS, not raw AUC, across sets.

  python flores_auc.py <flores200_dataset_dir> <out.json> [lang_code ...]
Env: LEXICAL_ONLY=1 skips the model; EMB_DIR saves each side x arm matrix (fp16 .npy) and reuses it;
     EMB_BATCH (16), EMB_DTYPE (float16), EMBED_DEVICE (cuda:0), EMB_MODEL (Qwen/Qwen3-Embedding-8B).
"""
from __future__ import annotations
import collections, csv, json, os, sys, time
from pathlib import Path

import numpy as np

TOPK = 10
SCENE = "Instruct: Group sentences that belong to the same conversation, scene, or connected event.\nQuery: "
TOPIC = "Instruct: Identify the topic or theme of the given text\nQuery: "
ARMS = [("none", None), ("topic", TOPIC), ("scene", SCENE)]
TFIDF = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
LANGS = {"eng_Latn": "English", "pes_Arab": "Persian", "prs_Arab": "Dari", "tgk_Cyrl": "Tajik",
         "arb_Arab": "MSA", "arz_Arab": "Egyptian", "apc_Arab": "North Levantine",
         "ajp_Arab": "South Levantine", "ars_Arab": "Najdi", "acm_Arab": "Mesopotamian",
         "acq_Arab": "Ta'izzi-Adeni", "ary_Arab": "Moroccan", "aeb_Arab": "Tunisian"}


def auc(pos, neg):
    """Rank-sum AUC with ties at half credit (identical to crosssum_diagnostic.auc)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    r = np.empty(len(allv), float)
    r[order] = np.arange(1, len(allv) + 1)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, r)
    r = (sums / counts)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(fdir: Path, code: str):
    texts, frag, topic = [], [], []
    for split in ("dev", "devtest"):
        with (fdir / f"metadata_{split}.tsv").open(encoding="utf-8") as f:
            meta = list(csv.DictReader(f, delimiter="\t"))
        lines = (fdir / split / f"{code}.{split}").read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(meta), (code, split, len(lines), len(meta))
        for m, t in zip(meta, lines):
            texts.append(t.strip()); frag.append(f"{split}:{m['URL']}"); topic.append(m["topic"].strip().lower())
    return texts, frag, topic


def tfidf_cos(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    M = normalize(TfidfVectorizer(**TFIDF).fit_transform(texts))
    return (M @ M.T).toarray()


def mine(frag_texts, k=TOPK):
    """Top-k most lexically similar OTHER fragments, by fragment-level char TF-IDF."""
    S = tfidf_cos(frag_texts)
    np.fill_diagonal(S, -1.0)
    return np.argpartition(-S, k, axis=1)[:, :k]


def pairs_between(members, nn):
    out = set()
    for d, nbrs in enumerate(nn):
        for d2 in nbrs:
            for a in members[d]:
                for b in members[int(d2)]:
                    out.add((a, b) if a < b else (b, a))
    return np.array(sorted(out), dtype=np.int64)


def main():
    fdir, outp = Path(sys.argv[1]), Path(sys.argv[2])
    codes = sys.argv[3:] or list(LANGS)
    lexical_only = os.environ.get("LEXICAL_ONLY") == "1"
    emb_dir = Path(os.environ["EMB_DIR"]) if os.environ.get("EMB_DIR") else None
    if emb_dir:
        emb_dir.mkdir(parents=True, exist_ok=True)

    # structure is shared by every language: same sentences, same order, same URLs
    en_texts, frag, topic = load(fdir, "eng_Latn")
    ids = {f: i for i, f in enumerate(dict.fromkeys(frag))}
    doc = np.array([ids[f] for f in frag])
    members = collections.defaultdict(list)
    for i, d in enumerate(doc):
        members[int(d)].append(i)
    members = [members[d] for d in range(len(ids))]
    n = len(doc)
    iu, ju = np.triu_indices(n, k=1)
    same = doc[iu] == doc[ju]
    pos = np.stack([iu[same], ju[same]], 1)
    neg_random = np.stack([iu[~same], ju[~same]], 1)
    top = np.array(topic)
    st = (~same) & (top[iu] == top[ju])
    neg_topic = np.stack([iu[st], ju[st]], 1)
    en_frag_texts = [" ".join(en_texts[i] for i in m) for m in members]
    neg_lex_en = pairs_between(members, mine(en_frag_texts))
    sizes = collections.Counter(len(m) for m in members)
    print(f"FLORES dev+devtest: {n} sentences, {len(members)} URL fragments (sizes {dict(sorted(sizes.items()))}), "
          f"{len(set(topic))} topics | pos {len(pos)} | random {len(neg_random)} | topic {len(neg_topic)} | "
          f"lexical_topk_en {len(neg_lex_en)}", flush=True)

    out = {"config": {"topk": TOPK, "tfidf": {k: list(v) if isinstance(v, tuple) else v for k, v in TFIDF.items()},
                      "arms": {a: i for a, i in ARMS}, "n_sentences": n, "n_fragments": len(members),
                      "fragment_sizes": {str(k): v for k, v in sorted(sizes.items())}, "n_topics": len(set(topic)),
                      "n_pos": int(len(pos)), "model": os.environ.get("EMB_MODEL", "Qwen/Qwen3-Embedding-8B"),
                      "dtype": os.environ.get("EMB_DTYPE", "float16")},
           "langs": {}}
    if outp.exists():
        out["langs"] = json.loads(outp.read_text(encoding="utf-8")).get("langs", {})

    model = None
    for code in codes:
        texts, frag_c, _ = load(fdir, code)
        assert frag_c == frag, code
        frag_texts = [" ".join(texts[i] for i in m) for m in members]
        negs = {"random": neg_random, "topic": neg_topic,
                "lexical_topk": pairs_between(members, mine(frag_texts)), "lexical_topk_en": neg_lex_en}
        L = tfidf_cos(texts)
        take = lambda X, P: X[P[:, 0], P[:, 1]]
        lp = take(L, pos)
        res = {"name": LANGS.get(code, code), "n_empty": sum(1 for t in texts if not t),
               "n_neg": {k: int(len(v)) for k, v in negs.items()},
               "lexical": {k: auc(lp, take(L, v)) for k, v in negs.items()}, "arms": {}}
        print(f"\n=== {code} ({res['name']}) | " + " ".join(f"lex[{k}]={v:.4f}" for k, v in res["lexical"].items()),
              flush=True)
        if not lexical_only:
            for arm, instr in ARMS:
                f = emb_dir / f"flores_{code}_{arm}.npy" if emb_dir else None
                if f is not None and f.exists():
                    E = np.load(f).astype(np.float32)
                else:
                    if model is None:
                        from sentence_transformers import SentenceTransformer
                        t0 = time.time()
                        model = SentenceTransformer(os.environ.get("EMB_MODEL", "Qwen/Qwen3-Embedding-8B"),
                                                    device=os.environ.get("EMBED_DEVICE", "cuda:0"),
                                                    trust_remote_code=True,
                                                    model_kwargs={"torch_dtype": os.environ.get("EMB_DTYPE", "float16")})
                        print(f"    model loaded in {time.time() - t0:.0f}s", flush=True)
                    t0 = time.time()
                    E = np.asarray(model.encode(texts, batch_size=int(os.environ.get("EMB_BATCH", "16")),
                                                convert_to_numpy=True, normalize_embeddings=True,
                                                show_progress_bar=False, prompt=instr), dtype=np.float32)
                    print(f"    encoded {code}/{arm} {E.shape} in {time.time() - t0:.0f}s", flush=True)
                    if f is not None:
                        np.save(f, E.astype(np.float16))
                E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
                S = E @ E.T
                ep = take(S, pos)
                row = {}
                for k, v in negs.items():
                    a = auc(ep, take(S, v))
                    row[k] = {"AUC_embedding": a, "AUC_lexical": res["lexical"][k], "gap": a - res["lexical"][k]}
                print(f"    [{arm:5s}] " + "  ".join(f"{k}: {r['AUC_embedding']:.4f} ({r['gap']:+.4f})"
                                                   for k, r in row.items()), flush=True)
                res["arms"][arm] = row
        out["langs"][code] = res
        outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}", flush=True)


if __name__ == "__main__":
    main()
