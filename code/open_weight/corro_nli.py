#!/usr/bin/env python
"""Non-LLM faithfulness ANCHOR -- a reproducible entailment check to corroborate the LLM judge.

OUR SET ONLY: uses the parallel gold English reference as an English premise, the claim as the English
hypothesis -> pure English-English NLI, NO translation confound. PRIMARY: MiniCheck (flan-t5-large);
FALLBACK: mDeBERTa-v3-xnli. Report mean support; correlate with the LLM judge.

We deliberately AVOID AlignScore (torch-1.12 pin hell) and QAFactEval (allennlp conflict) -- MiniCheck
covers the same English-side niche with a clean modern install.

Tier-2: isolated venv.   pip install minicheck   (or just transformers for the mDeBERTa fallback)

  python eval/corro_nli.py <faith_file.jsonl> [minicheck|mdeberta]
"""
from __future__ import annotations
import json, os, sys, statistics
from pathlib import Path

_mc = None
_pipe = None


def minicheck_support(docs: list[str], claims: list[str]):
    """docs = premises (gold English), claims = hypotheses. Returns (labels 0/1, support probs)."""
    global _mc
    if _mc is None:
        from minicheck.minicheck import MiniCheck
        _mc = MiniCheck(model_name=os.environ.get("MINICHECK_MODEL", "flan-t5-large"))
    pred_label, raw_prob, _, _ = _mc.score(docs=docs, claims=claims)
    return pred_label, raw_prob


def mdeberta_entail(premises: list[str], hypotheses: list[str]) -> list[dict]:
    """Zero-shot NLI (also the cross-lingual anchor: mDeBERTa-v3-xnli is XNLI-trained, including Arabic,
    so an Arabic premise vs an English claim hypothesis is a genuine Arabic-side entailment check, not
    just embedding similarity). BATCHED via the pipeline's own batch_size -- one dataset-style call
    instead of one dict per pair, which was ~1000x slower in practice (see retrieve.py's same fix)."""
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline
        dev = 0 if torch.cuda.is_available() else -1      # use the (free) GPU if present; CPU is slow under load
        _pipe = pipeline("text-classification",
                         model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", top_k=None, device=dev,
                         truncation=True, max_length=512)
    inputs = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]
    out = []
    for scores in _pipe(inputs, batch_size=int(os.environ.get("MDEBERTA_BATCH", "32"))):
        if scores and isinstance(scores[0], list):          # top_k=None can nest one level
            scores = scores[0]
        d = {s["label"].lower(): s["score"] for s in scores}
        out.append({"support": round(d.get("entailment", 0.0), 4),
                    "contradiction": round(d.get("contradiction", 0.0), 4)})
    return out


def run(faith_file: Path, backend: str = "minicheck", premise: str = "gold") -> Path:
    """premise='gold' (default): English gold_reference as premise, pure English-English NLI, no
    translation confound. premise='source': Arabic source_sents as premise directly -- a genuine
    cross-lingual Arabic-side entailment check (mDeBERTa-v3-xnli is XNLI-trained, includes Arabic).
    MiniCheck's flan-t5-large backend is not validated for Arabic input, so premise='source' forces
    backend='mdeberta' regardless of the requested backend."""
    rows = [json.loads(l) for l in faith_file.open(encoding="utf-8")]
    if rows[0]["corpus"] != "ours":
        sys.exit("corro_nli runs on OUR set only (needs gold_reference/source_sents as premise)")
    if premise == "source":
        backend = "mdeberta"
    # SUMMAC-ZS anchor ('summac'): score each claim against EACH premise SENTENCE and MAX-pool, so a
    # claim supported anywhere in the reference scores high -- avoids the 512-token truncation that
    # doc-level NLI ('mdeberta') suffers on long references. idx maps every pair back to its (ri, ci).
    sentence_level = backend == "summac"
    pairs, idx = [], []
    for ri, r in enumerate(rows):
        field = "source_sents" if premise == "source" else "gold_reference"
        ref_sents = [s for s in (r.get(field) or []) if s and s.strip()]
        premises = ref_sents if (sentence_level and ref_sents) else [" ".join(ref_sents)]
        for ci, c in enumerate(r.get("claims") or []):
            for p in premises:
                pairs.append((p, c)); idx.append((ri, ci))
    docs = [p for p, _ in pairs]; hyps = [h for _, h in pairs]
    scores = None
    if backend == "minicheck":
        try:
            _, probs = minicheck_support(docs, hyps)
            scores = [float(x) for x in probs]
        except Exception as e:                      # minicheck not installed -> still get an anchor
            print(f"  [corro_nli] MiniCheck unavailable ({e}); falling back to mDeBERTa NLI")
            backend = "mdeberta"
    if scores is None:
        res = mdeberta_entail(docs, hyps)
        scores = [d["support"] for d in res]
    # max-pool duplicate (ri,ci) entries (sentence-level) -> best supporting reference sentence wins
    best: dict = {}
    for (ri, ci), s in zip(idx, scores):
        k = (ri, ci)
        if k not in best or s > best[k]:
            best[k] = s
    for (ri, ci), s in best.items():
        rows[ri].setdefault("nli_support", {})[ci] = round(float(s), 4)
    # per-summary min/mean support, and correlation hook with the LLM judge
    for r in rows:
        sc = list((r.get("nli_support") or {}).values())
        r["nli_summary"] = {"mean": round(statistics.mean(sc), 4) if sc else None,
                            "min": round(min(sc), 4) if sc else None, "backend": backend, "premise": premise}
    suffix = f".nli.{backend}.jsonl" if premise == "gold" else f".nli.{backend}.{premise}.jsonl"
    out = faith_file.with_name(faith_file.stem + suffix)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    allm = [r["nli_summary"]["mean"] for r in rows if r["nli_summary"]["mean"] is not None]
    print(f"{faith_file.name} [{backend}/premise={premise}] mean support = {statistics.mean(allm):.3f}  -> {out}")
    return out


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "minicheck",
        sys.argv[3] if len(sys.argv) > 3 else "gold")
