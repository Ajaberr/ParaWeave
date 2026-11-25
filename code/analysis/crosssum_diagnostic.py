#!/usr/bin/env python
"""AUC_hard on REAL articles -- the non-circular version of the clustering diagnostic.

WHY THIS EXISTS. The DiaBench diagnostic defines a positive pair as "same document", but those
documents were built by the pipeline itself (GPT-5.5 proposed, Claude adjudicated), and the hard
negatives are "same cluster", where the clusters came from the english+scene arm -- one of the arms
being scored. So the metric partly measures agreement with its own construction. CrossSum articles
are real BBC documents whose boundaries nobody in this project chose.

IT ALSO SUPPLIES THE MISSING CONTROL. The paper claims Qwen3 fails on *dialectal* Arabic (0.4421).
Untested: does it fail on *Arabic*? CrossSum arabic-arabic is MSA. If MSA shows a healthy gap over
its lexical baseline while dialect sits at +0.002, that isolates dialect rather than the language.

HOW THE NEGATIVES ARE BUILT -- and why the obvious version is wrong.
A first attempt drew hard negatives from the same BBC URL category (middleeast, worldnews, ...).
That failed: `middleeast` alone spans ~772 articles about unrelated events, so character TF-IDF
scored AUC 0.7927 -- the "hard" negatives were trivially separable and the contrast measured nothing.
A newspaper section is not a topic.

So negatives are MINED: each article is paired with the articles most lexically similar to it
(article-level char TF-IDF, top-k nearest neighbours). This is standard hard-negative mining. Two
properties matter:
  * hardness is defined by TF-IDF, never by a neural embedding, so there is no circularity with the
    arms under test;
  * it drives the lexical baseline toward chance BY CONSTRUCTION, which is the regime DiaBench is
    already in (0.4400 / 0.4992). Only then is the embedding-minus-lexical gap comparable across the
    two datasets, since raw AUC is not -- the tasks differ in intrinsic difficulty.

All three negative regimes are reported from ONE embedding pass so the effect of negative choice is
visible rather than assumed: `random` (any other article), `category` (same BBC section, the flawed
version), `lexical_topk` (mined, the real one).

COMPARABILITY. Articles are truncated to 8 sentences (DiaBench mean 8.13, median 8) and only articles
with >= 8 are kept, so document length is not a confound. SEED and pair counts match the DiaBench
scripts. Embeddings are SAVED so negative sampling can be re-sliced without re-embedding.

  python crosssum_diagnostic.py <crosssum_data_dir> <out.json> [n_articles]
"""
from __future__ import annotations
import collections, json, os, random, re, sys, tarfile
from pathlib import Path

import numpy as np

SEED = 0
N_PAIRS = 60000
DOC_LEN = 8
TOPK = 10
SPLITS = {"msa": "arabic-arabic_CrossSum.tar.bz2",
          "english": "english-arabic_CrossSum.tar.bz2"}
SCENE = "Instruct: Group sentences that belong to the same conversation, scene, or connected event.\nQuery: "
TOPIC = "Instruct: Identify the topic or theme of the given text\nQuery: "
ARMS = [("none", None), ("topic", TOPIC), ("scene", SCENE)]
SENT = re.compile(r"(?<=[.!?؟۔])\s+")
TFIDF = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)


def category(url: str):
    for mark in ("/arabic/", "/news/", "/mundo/"):
        if mark in url:
            c = url.split(mark)[-1].split("/")[0]
            return re.sub(r"-?\d+$", "", c) or None    # world-europe-38120559 -> world-europe
    return None


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    r = np.empty(len(allv), float)
    r[order] = np.arange(1, len(allv) + 1)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, r)
    r = (sums / counts)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(data_dir: Path, tarname: str, n_articles: int, rng: random.Random):
    with tarfile.open(data_dir / tarname, "r:bz2") as tf:
        rows = []
        for m in sorted(x for x in tf.getnames() if x.endswith(".jsonl")):
            rows += [json.loads(l) for l in tf.extractfile(m)]
            if len(rows) > n_articles * 6:
                break
    keep = []
    for r in rows:
        cat = category(r.get("source_url", ""))
        if not cat:
            continue
        sents = [s.strip() for s in SENT.split(r["text"]) if len(s.strip()) > 15]
        if len(sents) >= DOC_LEN:
            keep.append({"sents": sents[:DOC_LEN], "cat": cat})
    if len(keep) > n_articles:
        keep = rng.sample(keep, n_articles)
    return keep


def mine_neighbours(art_texts, k=TOPK):
    """Top-k most lexically similar OTHER articles, by article-level char TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    A = normalize(TfidfVectorizer(**TFIDF).fit_transform(art_texts))
    nn, n = [], A.shape[0]
    step = 256
    for s in range(0, n, step):
        S = (A[s:s + step] @ A.T).toarray()
        for r in range(S.shape[0]):
            S[r, s + r] = -1.0                      # never itself
        nn.extend(np.argpartition(-S, k, axis=1)[:, :k].tolist())
    return nn


def main():
    data_dir, outp = Path(sys.argv[1]), Path(sys.argv[2])
    n_articles = int(sys.argv[3]) if len(sys.argv) > 3 else 2330
    embdir = outp.parent / "emb"
    embdir.mkdir(parents=True, exist_ok=True)
    out = {"config": {"doc_len": DOC_LEN, "n_pairs": N_PAIRS, "seed": SEED, "topk": TOPK,
                      "n_articles_requested": n_articles}, "sets": {}}

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(os.environ.get("EMB_MODEL", "Qwen/Qwen3-Embedding-8B"),
                                trust_remote_code=True)

    for name, tarname in SPLITS.items():
        rng = random.Random(SEED)
        arts = load(data_dir, tarname, n_articles, rng)
        texts, doc_of, cat_of = [], [], []
        for i, a in enumerate(arts):
            for s in a["sents"]:
                texts.append(s); doc_of.append(i); cat_of.append(a["cat"])
        doc_of = np.array(doc_of)
        print(f"\n=== {name}: {len(arts)} real articles | {len(texts)} sentences | "
              f"{len(set(cat_of))} BBC categories ===", flush=True)

        by_doc = collections.defaultdict(list)
        by_cat = collections.defaultdict(list)
        for i, (d, c) in enumerate(zip(doc_of, cat_of)):
            by_doc[int(d)].append(i); by_cat[c].append(i)

        nn = mine_neighbours([" ".join(a["sents"]) for a in arts])
        print(f"    mined top-{TOPK} lexical neighbours per article", flush=True)

        pos = [tuple(rng.sample(by_doc[rng.choice(list(by_doc))], 2))
               for _ in range(N_PAIRS // 2)]

        negs = {}
        # (a) random: any other article
        negs["random"] = []
        while len(negs["random"]) < N_PAIRS // 2:
            a, b = rng.randrange(len(doc_of)), rng.randrange(len(doc_of))
            if doc_of[a] != doc_of[b]:
                negs["random"].append((a, b))
        # (b) category: same BBC section, different article  -- the flawed version
        negs["category"] = []
        ck = [c for c, v in by_cat.items() if len({int(doc_of[i]) for i in v}) >= 2]
        guard = 0
        while len(negs["category"]) < N_PAIRS // 2 and ck and guard < N_PAIRS * 50:
            guard += 1
            v = by_cat[rng.choice(ck)]
            a, b = rng.sample(v, 2)
            if doc_of[a] != doc_of[b]:
                negs["category"].append((a, b))
        # (c) lexical_topk: mined nearest-neighbour articles -- the real hard negatives
        negs["lexical_topk"] = []
        for _ in range(N_PAIRS // 2):
            d = rng.randrange(len(arts))
            d2 = nn[d][rng.randrange(len(nn[d]))]
            negs["lexical_topk"].append((rng.choice(by_doc[d]), rng.choice(by_doc[int(d2)])))
        for k, v in negs.items():
            print(f"    negatives[{k:12s}] {len(v)}", flush=True)

        from sklearn.feature_extraction.text import TfidfVectorizer
        M = TfidfVectorizer(**TFIDF).fit_transform(texts)
        M = M.multiply(1.0 / (np.sqrt(M.multiply(M).sum(1)) + 1e-9)).tocsr()
        lex = lambda prs: np.array([float(M[a].multiply(M[b]).sum()) for a, b in prs])
        lp = lex(pos)
        res = {"n_articles": len(arts), "n_sentences": len(texts),
               "n_categories": len(set(cat_of)), "lexical": {}, "arms": {}}
        for k, v in negs.items():
            res["lexical"][k] = auc(lp, lex(v))
            print(f"    lexical char-TFIDF AUC [{k:12s}] = {res['lexical'][k]:.4f}", flush=True)

        for arm, instr in ARMS:
            tt = [instr + t for t in texts] if instr else texts
            E = np.asarray(model.encode(tt, batch_size=int(os.environ.get("EMB_BATCH", "8")),
                                        show_progress_bar=True, normalize_embeddings=True),
                           dtype=np.float32)
            np.save(embdir / f"cs_{name}_{arm}.npy", E)
            En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
            c = lambda prs, X: np.array([float(X[a] @ X[b]) for a, b in prs])
            ep = c(pos, En)
            row = {}
            for k, v in negs.items():
                a_emb = auc(ep, c(v, En))
                row[k] = {"AUC_embedding": a_emb, "AUC_lexical": res["lexical"][k],
                          "gap": a_emb - res["lexical"][k]}
                print(f"    [{arm:5s}][{k:12s}] emb={a_emb:.4f} lex={res['lexical'][k]:.4f} "
                      f"gap={a_emb-res['lexical'][k]:+.4f}", flush=True)
            res["arms"][arm] = row
            del E, En
        out["sets"][name] = res
        outp.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n===== REAL ARTICLES (CrossSum), by negative-sampling regime =====")
    print(f"{'set':9s}{'arm':7s}{'negatives':14s}{'embedding':>11s}{'lexical':>10s}{'gap':>9s}")
    for nm, r in out["sets"].items():
        for arm, row in r["arms"].items():
            for k, v in row.items():
                print(f"{nm:9s}{arm:7s}{k:14s}{v['AUC_embedding']:>11.4f}"
                      f"{v['AUC_lexical']:>10.4f}{v['gap']:>+9.4f}")
    print("\nDiaBench dialectal reference: embedding 0.4421 | lexical 0.4400 | gap +0.0021")
    print("Compare GAPs, not raw AUC -- the two datasets differ in intrinsic difficulty.")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
