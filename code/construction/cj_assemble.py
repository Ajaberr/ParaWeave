#!/usr/bin/env python
"""Claude-GATED assembly agent. GPT-5.5 carves articles from a cluster; CLAUDE (cross-judge, not the
builder) rates coherence 1-5 and flags the off-topic sentences; GPT re-assembles a tighter subset from
Claude's feedback; keep only when Claude says coh>=4 (& >=7 sents). Replaces the old GPT SELF-gate that
inflated its own work. Both models are API calls (run on the VM which has both keys).

  python eval/cj_assemble.py <clusters_file.jsonl> <out.jsonl> [max_refine=2]
Env: OPENAI_MODEL (gpt-5.5), ANTHROPIC_MODEL (claude-sonnet-4-6), ASM_WORKERS (8).
Env CJ_CACHE_DIR (2026-09-15): directory for an on-disk cache of the COMPOSER calls (gpt_break /
gpt_refine), keyed by sha256(model + system prompt + user prompt). A round that dies (network, spend cap,
kill) then costs nothing to redo on the composer side; the judge is never cached (cheap, and a
different judge must re-rate). Default: `_cj_cache` next to the pool files. CJ_CACHE_DIR=0 disables.
Env (2026-08-20 addition, for local-model construction tests): CJ_PROPOSER / CJ_JUDGE override which
LLM.__init__ system name is used for the proposer/judge roles (default "openai"/"claude", unchanged
for the existing Arabic pipeline). Must be two DIFFERENT models -- cross-judge is the whole point of
this file's design (a model gating its own proposals defeats it), so this refuses to run otherwise.
"""
import os, re, sys, json, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for line in next((p for p in (HERE.parent / ".env", HERE.parent.parent / ".env") if p.exists()), HERE.parent.parent / ".env").open(encoding="utf-8"):
    m = re.match(r"\s*([A-Z_]+)\s*=\s*(.+)", line)
    if m: os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
from llm import LLM  # noqa: E402

MIN_SENTS = int(os.environ.get("ASM_MIN_SENTS", "7"))
MIN_PAIRS = int(os.environ.get("ASM_MIN_PAIRS", "5"))   # shortest article the composer may propose (2026-09-17:
                                                        # env-configurable for the long-article Lugbara build; 5 = old behaviour)
MAX_CLUSTER = int(os.environ.get("ASM_MAX_CLUSTER", "250"))   # sentences shown to GPT per cluster (it DIVIDES)
PROPOSER_SYSTEM = os.environ.get("CJ_PROPOSER", "openai")
JUDGE_SYSTEM = os.environ.get("CJ_JUDGE", "claude")
# Name of the low-resource side as the prompts should call it. The pool schema keeps the field `ar`
# for every variety (Naija, Farsi, Dari, Tajik included); only the prompt wording changes.
SRC_LANG = os.environ.get("CJ_SRC_LANG", "dialectal Arabic")
assert PROPOSER_SYSTEM != JUDGE_SYSTEM, (
    f"CJ_PROPOSER and CJ_JUDGE must differ (got {PROPOSER_SYSTEM!r} for both) -- a model judging its "
    f"own proposals is exactly the self-gate bug this file's cross-judge design replaced.")
GPT, CLAUDE = LLM(PROPOSER_SYSTEM), LLM(JUDGE_SYSTEM)   # names kept for readability; hold whatever
                                                          # CJ_PROPOSER/CJ_JUDGE point at, not literally
                                                          # GPT/Claude unless left at the defaults.

# --- STRICT rubric baked into BOTH GPT formation and Claude gate (shared-setting = 3 = REJECT) ---
RUBRIC = ("STRICT COHERENCE RUBRIC (the ONLY thing that counts):\n"
 "  5 = ONE topic/event/scene: the SAME participants and a single continuous through-line (one conversation, "
 "one incident, one story).\n"
 "  4 = one scene with only minor drift.\n"
 "  3 = loosely related: a shared SETTING/DOMAIN but DIFFERENT sub-things -- e.g. several separate "
 "accident-related interactions, or a list of payment/shopping/tourism phrases. A shared setting/domain/topic "
 "is NOT coherence. REJECT (do NOT output).\n"
 "  1-2 = grab-bag / phrasebook / independent one-off utterances (tourism Qs, food orders, symptom lists, "
 "single-word commands).")

BREAK_SYS = (f"You are given a topical CLUSTER of {SRC_LANG} sentences with English translations (numbered "
 "SOURCE ||| EN pairs). Select and order sentences into ARTICLES that each score >=4 on this rubric.\n" + RUBRIC +
 f"\nEach article = {MIN_PAIRS}+ pairs forming a GENUINE single connected scene/story (rubric >=4). ALLOWED edits ONLY: "
 f"(1) reorder pairs for flow; (2) unify PROPER-NOUN names (same change to BOTH the {SRC_LANG} and the English). "
 "FORBIDDEN: rewording, adding connectives, changing common nouns/verbs, adding/removing content. Be STRICT: "
 "only output articles you are confident are >=4 (a single connected scene). A cluster that is only a shared "
 "setting yields NOTHING -- return []. If no single connected scene exists, return []. "
 'Return ONLY JSON: {"articles":[{"pairs":[{"ar":"..","en":".."}, ...]}, ...]}.')

# JUDGE: the SAME strict rubric coherence definition used for the whole dataset (shared setting = 3).
JUDGE_SYS = ("You rate the COHERENCE of a candidate article on this exact scale. Judge STRICTLY; a shared "
             "setting/domain is a 3, NOT a 4.\n" + RUBRIC)
JUDGE_USER = ('Numbered English sentences:\n__BODY__\n\nReturn ONLY JSON {"coherence": <1-5 per the rubric>, '
              '"drop":[<indices NOT part of the single connected scene>], "reason":"<=15 words"}.')

REFINE_SYS = BREAK_SYS
REFINE_USER = ('This article was judged loosely coherent. These sentence indices do not belong: __DROP__. '
               'Rebuild ONE tighter article as a connected subset (drop those + anything else off-scene; '
               f'reorder allowed; NO rewording/new content). Keep >={MIN_PAIRS} pairs or return empty. '
               'Numbered SOURCE ||| EN pairs:\n__BODY__\n\nReturn ONLY JSON {"pairs":[{"ar":"..","en":".."}, ...]}.')


_IDX_PREFIX = re.compile(r"^\d+\.\s+")   # strip a leaked "N. " -- gpt_break()'s prompt numbers each
                                          # input line ("i. ar ||| en") so the model can reference
                                          # sentences by index; weaker proposer models (confirmed with
                                          # gpt-oss-20b, 2026-08-21) sometimes echo that index straight
                                          # into their own output "ar"/"en" text instead of just using
                                          # it for selection. Real content starting with a literal
                                          # "N. " is rare enough that this strip is a strict cleanup.


def _pairs(a):
    out = []
    for p in (a.get("pairs") or []):
        if not isinstance(p, dict) or not p.get("ar") or not p.get("en"):
            continue
        out.append({**p, "ar": _IDX_PREFIX.sub("", p["ar"]), "en": _IDX_PREFIX.sub("", p["en"])})
    return out


_CACHE_DIR = os.environ.get("CJ_CACHE_DIR", "_cj_cache")


def _cached_json(sys_p, user_p, *, max_tokens):
    """GPT.complete_json with an on-disk cache keyed by (model, prompts). Only successful, non-empty
    JSON is cached, so a transient failure ({}) is retried next time rather than frozen in."""
    if _CACHE_DIR in ("0", ""):
        return GPT.complete_json(sys_p, user_p, temperature=0.0, max_tokens=max_tokens)
    key = hashlib.sha256((GPT.model + "\x00" + sys_p + "\x00" + user_p).encode("utf-8")).hexdigest()[:32]
    d = Path(_CACHE_DIR); d.mkdir(parents=True, exist_ok=True); f = d / f"{key}.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    v = GPT.complete_json(sys_p, user_p, temperature=0.0, max_tokens=max_tokens)
    if v:
        tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8"); tmp.replace(f)
    return v


def gpt_break(cluster):
    body = "\n".join(f"{i}. {p['ar']} ||| {p['en']}" for i, p in enumerate(cluster[:MAX_CLUSTER]))
    v = _cached_json(BREAK_SYS, body, max_tokens=4000)   # big cluster in, MANY articles out
    return [ps for a in (v or {}).get("articles", []) if len(ps := _pairs(a)) >= MIN_PAIRS]


def claude_judge(pairs):
    body = "\n".join(f"{i}. {p['en']}" for i, p in enumerate(pairs))
    v = CLAUDE.complete_json(JUDGE_SYS, JUDGE_USER.replace("__BODY__", body), temperature=0.0, max_tokens=300) or {}
    try:
        coh = int(v.get("coherence", 0))
    except Exception:
        coh = 0
    drop = [i for i in (v.get("drop") or []) if isinstance(i, int) and 0 <= i < len(pairs)]
    return coh, drop, str(v.get("reason", ""))[:120]


def gpt_refine(pairs, drop):
    body = "\n".join(f"{i}. {p['ar']} ||| {p['en']}" for i, p in enumerate(pairs))
    v = _cached_json(REFINE_SYS, REFINE_USER.replace("__DROP__", str(drop)).replace("__BODY__", body), max_tokens=1500)
    return _pairs(v or {})


def process_cluster(cluster, max_refine):
    """-> list of kept articles (each: pairs + claude_coherence + refine_rounds)."""
    kept = []
    for pairs in gpt_break(cluster):
        rounds = 0
        while True:
            coh, drop, reason = claude_judge(pairs)
            if coh >= 4 and len(pairs) >= MIN_SENTS:
                kept.append({"pairs": pairs, "claude_coherence": coh, "refine_rounds": rounds, "reason": reason})
                break
            if rounds >= max_refine or not drop or len(pairs) - len(drop) < MIN_PAIRS:
                break                                    # unfixable -> discard honestly
            new = gpt_refine(pairs, drop)
            if not new or len(new) >= len(pairs):
                break
            pairs = new; rounds += 1
    return kept


def main():
    clusters = [json.loads(l) for l in Path(sys.argv[1]).open(encoding="utf-8")]
    out = Path(sys.argv[2])
    max_refine = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    workers = int(os.environ.get("ASM_WORKERS", "8"))
    print(f"cj_assemble: {len(clusters)} clusters | GPT={GPT.model} CLAUDE={os.environ.get('ANTHROPIC_MODEL')} "
          f"max_refine={max_refine}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda c: process_cluster(c, max_refine), clusters))
    kept = []
    for cl, arts in zip(clusters, results):
        sub = cl[0].get("sub", "?") if cl else "?"
        for a in arts:
            kept.append({"sub": sub, "source": "cj-assembled", "claude_coherence": a["claude_coherence"],
                         "refine_rounds": a["refine_rounds"], "n": len(a["pairs"]),
                         "dialect_article": [p["ar"] for p in a["pairs"]],
                         "gold_reference": [p["en"] for p in a["pairs"]]})
    with out.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    resc = sum(a["refine_rounds"] > 0 for arts in results for a in arts)
    from collections import Counter
    print(f"KEPT {len(kept)} articles (Claude coh>=4 & >={MIN_SENTS} sents) | rescued-by-refine {resc} | "
          f"coh dist {dict(sorted(Counter(r['claude_coherence'] for r in kept).items()))} -> {out}", flush=True)


if __name__ == "__main__":
    main()
