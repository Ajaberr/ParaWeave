#!/usr/bin/env python
"""Stage 1 -- GENERATE English summaries from Arabic with ChatGPT and/or Qwen3-8B.

Identical prompt across systems (fairness). One JSONL line per (record x condition).

  python eval/gen.py <corpus> <system> [conditions]
    corpus      ours | crosssum
    system      openai | qwen3-8b
    conditions  comma list. ours: dialect,gold_ref   crosssum: arabic   (defaults per corpus)

Env: OPENAI_API_KEY / OPENAI_MODEL, QWEN_BASE_URL / QWEN_MODEL, GEN_WORKERS (default 8),
     GEN_MAXWORDS (default 30), GEN_LIMIT (0=all), CROSSSUM_N (default 500).
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import LLM, parallel_map                                      # noqa: E402
import data as dio                                                    # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# GEN_SRC_LANG (2026-08, additive): lets the multilingual extension (Farsi/Dari/Tajik, see
# processing/build_persian_family.py) reuse this file unchanged instead of forking it -- default is
# byte-identical to the original hardcoded "Arabic" wording, so existing "ours"/"crosssum" runs
# (including anything reproducing already-measured/frozen Arabic results) are untouched.
SRC_LANG = os.environ.get("GEN_SRC_LANG", "Arabic")
SYS_PROMPT = (f"You are a precise cross-lingual summarizer. You read {SRC_LANG} (including dialect) "
              "and write English.")
USER_TMPL = ("Summarize the following {dialect}-" + SRC_LANG + " text in ONE English sentence of at "
             "most {maxw} words. Output ONLY the summary as plain declarative English -- no preamble, "
             "no quotes, no commentary. Do not add any fact that is not stated in the text.\n\nTEXT:\n{src}")

def _tmpl(dialect):
    """No dialect label (e.g. Lugbara) -> "the following Lugbara text", not "-Lugbara text" (2026-09-17)."""
    return USER_TMPL if dialect else USER_TMPL.replace("{dialect}-", "")


# what text we feed the model, per condition
COND_SOURCE = {
    "dialect":  lambda r: " ".join(r["source_sents"]),                # true test (read the dialect)
    "gold_ref": lambda r: " ".join(r["gold_reference"] or []),        # upper bound (read clean English)
    "arabic":   lambda r: r["source_text"],                           # crosssum MSA article
}
DEFAULT_CONDS = {"ours": ["dialect", "gold_ref"], "crosssum": ["arabic"], "mlext": ["dialect", "gold_ref"]}


def summarize(llm: LLM, source_text: str, dialect: str, max_words: int = 30) -> str:
    user = _tmpl(dialect).format(dialect=dialect, maxw=max_words, src=source_text[:8000])
    return llm.complete_text(SYS_PROMPT, user, temperature=float(os.environ.get("GEN_TEMP", "0.3")),
                             max_tokens=90)


def run_generation(corpus: str, system: str, conditions: list[str]) -> Path:
    out = OUT / f"gen.{os.environ.get('GEN_OUT_TAG', corpus)}.{system}.jsonl"   # GEN_OUT_TAG keeps per-language mlext runs apart (2026-09-17)
    if out.exists() and not os.environ.get("GEN_FORCE"):
        print(f"[skip] {out.name} already exists (reusing; GEN_FORCE=1 to regenerate)")
        return out
    limit = int(os.environ.get("GEN_LIMIT", "0")) or None
    kw = {"limit": limit} if corpus == "ours" else {"limit": int(os.environ.get("CROSSSUM_N", "500"))}
    recs = dio.load_records(corpus, **kw)
    dfilt = os.environ.get("DIALECT_FILTER")            # e.g. DIALECT_FILTER=maghrebi -> run a specialist on its slice
    if dfilt:
        keep = {d.strip() for d in dfilt.split(",")}
        recs = [r for r in recs if r.get("dialect") in keep]
        print(f"[dialect-filter] {dfilt} -> {len(recs)} records")
    maxw = int(os.environ.get("GEN_MAXWORDS", "30"))
    workers = int(os.environ.get("GEN_WORKERS", "8"))
    llm = LLM(system)
    jobs = [(r, c) for r in recs for c in conditions if COND_SOURCE[c](r).strip()]
    print(f"generating {corpus} x {system}: {len(recs)} records x {conditions} = {len(jobs)} summaries "
          f"(model={llm.model})")

    # identical prompt across backends (fairness); batched -> HF runs on GPU, OpenAI threads the API
    users = [_tmpl(r["dialect"]).format(dialect=r["dialect"], maxw=maxw, src=COND_SOURCE[c](r)[:8000])
             for r, c in jobs]
    summaries = llm.complete_text_many(SYS_PROMPT, users,
                                       temperature=float(os.environ.get("GEN_TEMP", "0.3")),
                                       max_tokens=int(os.environ.get("GEN_MAXTOK", "90")),
                                       workers=workers, label=f"{system} ")
    rows = []
    for (r, c), summ in zip(jobs, summaries):
        summ = (summ or "").strip()
        rows.append({"id": r["id"], "corpus": corpus, "dialect": r["dialect"], "system": system,
                     "model": llm.model, "condition": c, "summary": summ,
                     "n_words": len(summ.split()), "source_text": r["source_text"],
                     "source_sents": r["source_sents"], "gold_reference": r.get("gold_reference"),
                     "gold_summary": r.get("gold_summary"), "en_reference": r.get("en_reference")})
    out = OUT / f"gen.{os.environ.get('GEN_OUT_TAG', corpus)}.{system}.jsonl"   # GEN_OUT_TAG keeps per-language mlext runs apart (2026-09-17)
    with out.open("w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    import statistics
    wl = [x["n_words"] for x in rows]
    print(f"-> {out}  ({len(rows)} summaries, median {statistics.median(wl):.0f} words, max {max(wl)})")
    return out


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "ours"
    system = sys.argv[2] if len(sys.argv) > 2 else "openai"
    conds = sys.argv[3].split(",") if len(sys.argv) > 3 else DEFAULT_CONDS[corpus]
    run_generation(corpus, system, conds)


if __name__ == "__main__":
    main()
