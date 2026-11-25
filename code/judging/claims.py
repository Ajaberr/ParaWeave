#!/usr/bin/env python
"""Atomic claim decomposition (FActScore/FENICE front-end), reimplemented with our LLMs.

Splits an English summary into atomic, standalone, pronoun-resolved factual claims. Use a NEUTRAL
LLM (not the summarizer) to avoid self-bias. No factscore/fenice install -- just the prompt.
"""
from __future__ import annotations

DECOMP_SYS = ("You break a summary into atomic facts. An atomic fact states ONE thing, is fully "
              "self-contained (resolve every pronoun to an explicit entity), and copies no opinion "
              "or hedge. Output ONLY facts that the summary asserts. Many summaries are unattributed "
              "first-person dialogue with no named speaker -- in that case resolve 'I'/'me' to "
              "'the speaker' and 'you' to 'the listener' (or 'the addressee'); do NOT return zero "
              "claims just because no proper name is given.")

DECOMP_USER = ('Decompose this summary into atomic factual claims. Return ONLY JSON: '
               '{"claims": ["<claim 1>", "<claim 2>", ...]}. Each claim is one short standalone '
               'English sentence with all references made explicit. If the summary states nothing '
               'factual, return {"claims": []}.\n\nSUMMARY:\n')


def _claims_of(v) -> list[str]:
    claims = (v.get("claims") or []) if isinstance(v, dict) else []
    return [c.strip() for c in claims if isinstance(c, str) and c.strip()]


def decompose(llm, summary: str) -> list[str]:
    summary = (summary or "").strip()
    if not summary:
        return []
    return _claims_of(llm.complete_json(DECOMP_SYS, DECOMP_USER + summary,
                                        temperature=0.0, max_tokens=400))


def decompose_many(llm, summaries) -> list[list[str]]:
    """Batched decomposition -- one call per non-empty summary (HF batches on GPU; OpenAI threads)."""
    import os
    summaries = [(s or "").strip() for s in summaries]
    todo = [(i, s) for i, s in enumerate(summaries) if s]
    out: list[list[str]] = [[] for _ in summaries]
    if not todo:
        return out
    raws = llm.complete_json_many(DECOMP_SYS, [DECOMP_USER + s for _, s in todo],
                                  temperature=0.0, max_tokens=400,
                                  workers=int(os.environ.get("JUDGE_WORKERS", "8")),
                                  label="decompose ")
    for (i, _), v in zip(todo, raws):
        out[i] = _claims_of(v)
    return out
