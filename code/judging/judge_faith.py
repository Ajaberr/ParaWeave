#!/usr/bin/env python
"""Stage 2, ARM B (HEADLINE) -- source-grounded faithfulness via LLM-as-judge.

For each generated summary: decompose -> claims (neutral LLM), then a JUDGE LLM rules each claim
against the SOURCE as SUPPORTED / PARTIAL / NOT_SUPPORTED / CONTRADICTED with an AraHalluEval-style
error tag. CROSS-JUDGE: run with judge != summarizer to kill self-preference.

  python eval/judge_faith.py <gen_file.jsonl> <judge_system> [ground_on]
    judge_system   openai | qwen3-8b   (use the one that did NOT generate the summaries)
    ground_on      comma list subset of {dialect,gold_reference,arabic}  (default per corpus)

Env: DECOMP_SYSTEM (neutral decomposer, default openai), JUDGE_WORKERS (8), FAITH_GAMMA (3).
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import LLM, parallel_map        # noqa: E402
from claims import decompose, decompose_many   # noqa: E402,F401

LABELS = ["SUPPORTED", "PARTIAL", "NOT_SUPPORTED", "CONTRADICTED"]
ERRORS = ["none", "entity", "number", "relation", "unsupported", "contradiction", "overgeneralization"]

# JUDGE_SRC_LANG (2026-08, additive, mirrors gen.py's SRC_LANG): default matches the original hardcoded
# "Arabic" wording exactly, so already-measured/frozen Arabic judge runs are untouched.
JUDGE_SRC_LANG = os.environ.get("JUDGE_SRC_LANG", "Arabic")
JUDGE_SYS = (f"You are a strict cross-lingual faithfulness judge. You read {JUDGE_SRC_LANG} (including "
             "dialect) and English. Given a SOURCE text and CLAIMS taken from an English summary of it, "
             "decide for EACH claim whether the SOURCE supports it. Judge ONLY against the source; use "
             "no outside knowledge.")

JUDGE_USER = (
    "SOURCE:\n__SRC__\n\nCLAIMS:\n__CLAIMS__\n\n"
    "For each claim return a verdict. Labels: SUPPORTED (fully stated/entailed by the source), "
    "PARTIAL (partly supported, some detail missing or vague), NOT_SUPPORTED (the source does not "
    "state it -- fabricated/extrinsic), CONTRADICTED (the source states the opposite). "
    "error in { none | entity: wrong/invented named thing (person/place/org) | number: wrong quantity/"
    "date/amount/count | relation: right entities but wrong who-did-what/attribution/causality | "
    "unsupported: content absent from the source (fabrication) | contradiction: asserts the opposite of "
    "the source | overgeneralization: broader/stronger than the source supports }. "
    'Return ONLY JSON: {"verdicts":[{"i":1,"label":"...","error":"...","evidence":"<short source '
    'quote or empty>","reason":"<=15 words"}]} with one entry per claim index.')


def build_judge_prompt(source: str, claims: list[str]) -> str:
    listed = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    return JUDGE_USER.replace("__SRC__", source[:8000]).replace("__CLAIMS__", listed)


def parse_verdicts(v: dict, claims: list[str]) -> list[dict]:
    by_i = {}
    for d in ((v or {}).get("verdicts") or []):
        try:
            by_i[int(d.get("i"))] = d
        except Exception:
            pass
    out = []
    for i, c in enumerate(claims, 1):
        d = by_i.get(i, {})
        lab = str(d.get("label", "NOT_SUPPORTED")).upper()
        out.append({"claim": c,
                    "label": lab if lab in LABELS else "NOT_SUPPORTED",
                    "error": str(d.get("error", "none")).lower(),
                    "evidence": str(d.get("evidence", ""))[:200],
                    "reason": str(d.get("reason", ""))[:120]})
    return out


def judge_summary(judge: LLM, source: str, claims: list[str]) -> list[dict]:
    if not claims:
        return []
    v = judge.complete_json(JUDGE_SYS, build_judge_prompt(source, claims),
                            temperature=0.0, max_tokens=900)
    return parse_verdicts(v, claims)


def aggregate_verdicts(verdicts: list[dict], gamma: float) -> dict:
    n = len(verdicts) or 1
    c = Counter(v["label"] for v in verdicts)
    sup, par = c.get("SUPPORTED", 0), c.get("PARTIAL", 0)
    nsup, con = c.get("NOT_SUPPORTED", 0), c.get("CONTRADICTED", 0)
    faith = (sup + 0.5 * par) / n
    penalty = min(1.0, len(verdicts) / gamma)              # FActScore-style: don't reward too-few claims
    return {
        "n_claims": len(verdicts),
        "faithfulness_rate": round(faith, 4),
        "hallucination_density": round((nsup + con) / n, 4),
        "contradiction_rate": round(con / n, 4),
        "gamma_penalized": round(faith * penalty, 4),
        "label_counts": dict(c),
        "error_histogram": dict(Counter(v["error"] for v in verdicts if v["error"] != "none")),
    }


SOURCE_FOR = {
    "dialect":        lambda r: " ".join(r.get("source_sents") or []),
    "gold_reference": lambda r: " ".join(r.get("gold_reference") or []),
    "arabic":         lambda r: r.get("source_text", ""),
}
DEFAULT_GROUND = {"ours": ["dialect", "gold_reference"], "crosssum": ["arabic"],
                   "mlext": ["dialect", "gold_reference"]}


def run_faithfulness(gen_file: Path, judge_system: str, ground_on: list[str]) -> Path:
    out = gen_file.with_name(gen_file.stem + f".faith.{judge_system}.jsonl")
    if out.exists() and not os.environ.get("JUDGE_FORCE"):     # reuse -> no re-spend on re-runs
        print(f"[skip] {out.name} already exists (reusing; JUDGE_FORCE=1 to re-judge)")
        return out
    rows = [json.loads(l) for l in gen_file.open(encoding="utf-8")]
    corpus = rows[0]["corpus"]
    summarizer = rows[0]["system"]
    if judge_system == summarizer:
        print(f"  WARNING: judge==summarizer ({judge_system}); cross-judge with the other system to avoid self-bias")
    gamma = float(os.environ.get("FAITH_GAMMA", "3"))
    workers = int(os.environ.get("JUDGE_WORKERS", "8"))
    judge = LLM(judge_system)
    # 1. decompose ONCE and FREEZE -> every judge (and the human labels) see IDENTICAL claims, so the
    #    cross-judge / panel / meta-eval all align. (DECOMP_FORCE=1 re-decomposes from scratch.)
    claims_file = gen_file.with_name(gen_file.name.replace("gen.", "claims.", 1))
    claims_list = None
    if claims_file.exists() and not os.environ.get("DECOMP_FORCE"):
        cf = [json.loads(l) for l in claims_file.open(encoding="utf-8")]
        if len(cf) == len(rows):
            claims_list = [c.get("claims", []) for c in cf]
            print(f"[claims] reusing frozen {claims_file.name}")
    if claims_list is None:
        decomposer = LLM(os.environ.get("DECOMP_SYSTEM", "openai"))
        claims_list = decompose_many(decomposer, [r["summary"] for r in rows])
        with claims_file.open("w", encoding="utf-8") as fh:
            for r, cl in zip(rows, claims_list):
                fh.write(json.dumps({"id": r["id"], "claims": cl}, ensure_ascii=False) + "\n")
        print(f"[claims] froze {claims_file.name}")
    print(f"faithfulness: {gen_file.name} | summarizer={summarizer} judge={judge_system} ground_on={ground_on}")

    # 2. judge per ground_on source -- batched; rows with no claims skip the model entirely
    for r, cl in zip(rows, claims_list):
        r["claims"] = cl
    for g in ground_on:
        todo = [(k, build_judge_prompt(SOURCE_FOR[g](r), cl))
                for k, (r, cl) in enumerate(zip(rows, claims_list)) if cl]
        raws = (judge.complete_json_many(JUDGE_SYS, [u for _, u in todo], temperature=0.0,
                                         max_tokens=900, workers=workers, label=f"judge:{g} ")
                if todo else [])
        verds = [[] for _ in rows]
        for (k, _), v in zip(todo, raws):
            verds[k] = parse_verdicts(v, claims_list[k])
        for r, vs in zip(rows, verds):
            r.setdefault("faith", {})[g] = {"verdicts": vs, **aggregate_verdicts(vs, gamma)}

    out = gen_file.with_name(gen_file.stem + f".faith.{judge_system}.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # corpus-level summary print
    print(f"\n# {corpus} | summarizer={summarizer} | judge={judge_system}")
    for g in ground_on:
        import statistics
        fr = [r["faith"][g]["faithfulness_rate"] for r in rows if r["faith"][g]["n_claims"]]
        hd = [r["faith"][g]["hallucination_density"] for r in rows if r["faith"][g]["n_claims"]]
        eh = Counter()
        for r in rows:
            eh.update(r["faith"][g]["error_histogram"])
        print(f"  ground={g:14s} faithfulness={statistics.mean(fr):.3f}  "
              f"halluc_density={statistics.mean(hd):.3f}  top_errors={dict(eh.most_common(4))}")
    print(f"-> {out}")
    return out


def main():
    gen_file = Path(sys.argv[1])
    judge_system = sys.argv[2] if len(sys.argv) > 2 else "claude"     # Claude Opus 4.8 = standard judge
    corpus = json.loads(gen_file.open(encoding="utf-8").readline())["corpus"]
    ground = sys.argv[3].split(",") if len(sys.argv) > 3 else DEFAULT_GROUND[corpus]
    run_faithfulness(gen_file, judge_system, ground)


if __name__ == "__main__":
    main()
