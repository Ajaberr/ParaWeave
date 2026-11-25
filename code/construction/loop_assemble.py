#!/usr/bin/env python
"""Loop-until-dry assembly. Each ROUND: PCA+KMeans cluster the live pool -> GPT divides each big cluster
into articles -> Claude gates coh>=4 (refine per article, via cj_assemble.process_cluster) -> keep. Then
the LEFTOVER sentences (rejected-article + never-used) become the next round's pool and get RE-CLUSTERED
fresh -- sentences that couldn't connect before may group differently once the used ones are gone. Stop
when a round keeps < LOOP_DRY articles. Uses cached raw embeddings so every re-cluster is fast (no re-embed).

  python eval/loop_assemble.py <dialect> <out.jsonl>
Env: LOOP_ROUNDS(6) LOOP_DRY(5) CLUSTER_SIZE(150) PCA_DIM(50) MIN_KEEP(10) ASM_MAX_REFINE(2) ASM_WORKERS(8)
     LOOP_COMPOSE_ONLY=1: run ONLY the round-1 composer calls (gpt_break) on the live pool, filling the
     composer cache (cj_assemble CJ_CACHE_DIR), then exit without any judge call and without writing
     articles. Use it to pre-pay / pre-check the GPT side when the judge is not yet available; the real run
     then replays those clusters from the cache for free.
     LOOP_RESUME(1): if <out.jsonl> exists, its articles are loaded, their sentences retired from the pool
     and new rounds APPEND to it -- so a run that stopped (network, spend cap, kill) continues where it was
     instead of re-composing the same clusters. Each round's kept articles are written as soon as the round
     ends. LLM_BUDGET_USD_<PROVIDER> (see llm.py) stops the loop cleanly at the end of the current round.
"""
import os, sys, json
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cj_assemble as CJ   # reuses process_cluster (gpt_break -> claude_judge -> refine), strict prompts
from llm import BudgetExceeded, usage_summary

# --- IN-LOOP gate = the SAME holistic rubric used for independent verification (no gate/verify gap) ---
SRC_LANG = os.environ.get("CJ_SRC_LANG", "dialectal Arabic")   # see cj_assemble.SRC_LANG
RUB_SYS = (
 "You rate candidate source articles on four 1-5 axes. The first three are judged from the ENGLISH gold "
 f"reference; the fourth from the SOURCE text, which is {SRC_LANG}.\n"
 "- coherence: do the sentences form ONE topical unit? 5=one topic/event/scene; 3=loosely related (shared "
 "setting/domain but different sub-things); 1=grab-bag/phrasebook. A shared SETTING/DOMAIN is NOT coherence "
 "-- independent one-off utterances (tourism Qs, food orders, symptom lists, single-word commands) are 1-2.\n"
 "- article_like: does it read like a connected real document with flow (not a bag/list)? 5=flowing passage; "
 "3=choppy but related; 1=bag of independent lines.\n"
 "- summarizability: is there a compressible informational core? 5=clear central point+detail, condenses well; "
 "3=thin/diffuse; 1=nothing to summarize.\n"
 f"- dialectness: how colloquial is the SOURCE ({SRC_LANG})? 5=heavy spoken dialect/colloquial register; "
 "3=mixed; 1=pure formal standard written register.\n"
 "CRITICAL CALIBRATION: these articles are lightly MANUALLY POST-EDITED afterward, so DO NOT penalize fixable "
 "surface issues -- inconsistent entity names/referents, missing connectives, reorder needs, minor "
 "disfluencies/translation artifacts. Judge as if a careful editor cleaned those up (adding NO new facts). "
 "Penalize ONLY deep unfixable problems: topic scatter, no summarizable core, list/phrasebook structure, no info.")
RUB_USER = ('ENGLISH gold reference:\n__EN__\n\nSOURCE (' + SRC_LANG + ') article:\n__AR__\n\nReturn ONLY JSON: '
            '{"coherence":<1-5>,"article_like":<1-5>,"summarizability":<1-5>,"dialectness":<1-5>}')
def _bul(xs): return "\n".join(f"- {s}" for s in (xs or [])[:12])
def rubric_verify(pairs):
    u = RUB_USER.replace("__EN__", _bul([p["en"] for p in pairs])).replace("__AR__", _bul([p["ar"] for p in pairs]))
    v = CJ.CLAUDE.complete_json(RUB_SYS, u, temperature=0.0, max_tokens=150) or {}
    def g(k):
        try: return int(v.get(k, 0) or 0)
        except Exception: return 0
    return g("coherence"), {k: g(k) for k in ("coherence", "article_like", "summarizability", "dialectness")}

def norm(s): return (s or "").strip()


def run_loop(dialect, pairs, E, *, rounds=6, dry=5, csize=150, pca_dim=50, min_keep=10,
             max_refine=2, workers=8, used_init=None, on_round_kept=None):
    """The loop-until-dry round/recycle logic, extracted (2026-08-21) so it's reusable from other
    scripts (e.g. processing/loop_assemble_ml.py for the ML-transfer pools) without needing to
    duplicate it -- moved out of this file's former unguarded module-level code (which ran
    immediately on import, making the file unimportable) into a proper function, same logic,
    unchanged."""
    assert len(E) == len(pairs), f"emb/pairs mismatch {len(E)} vs {len(pairs)}"
    used = set(used_init or ())       # sentences already in kept articles (resume), never re-offered
    alive = [gi for gi in range(len(pairs)) if norm(pairs[gi]["ar"]) not in used]   # global indices in play
    kept_all = []
    rnd = 0
    stop = {}                          # set by a worker when a provider's spend cap is hit
    compose_only = os.environ.get("LOOP_COMPOSE_ONLY") == "1"
    for rnd in range(1, rounds + 1):
        if len(alive) < min_keep:
            break
        sub_alive = E[alive]
        d = min(pca_dim, sub_alive.shape[0], sub_alive.shape[1]) if pca_dim else 0
        Xr = PCA(n_components=d, random_state=0).fit_transform(sub_alive) if d else sub_alive
        # cluster per sub-dialect (positions index into `alive`)
        bysub = defaultdict(list)
        for pos, gi in enumerate(alive):
            bysub[pairs[gi].get("sub", "?")].append(pos)
        clusters = []                              # each = list of GLOBAL indices
        for sub, poss in bysub.items():
            if len(poss) < min_keep:
                continue
            k = max(1, len(poss) // csize)
            if k == 1:
                clusters.append([alive[p] for p in poss]); continue
            lab = MiniBatchKMeans(n_clusters=k, random_state=0, n_init="auto", batch_size=512).fit_predict(Xr[poss])
            cl = defaultdict(list)
            for p, l in zip(poss, lab):
                cl[int(l)].append(alive[p])
            clusters += [v for v in cl.values() if len(v) >= min_keep]

        def do(gidxs):
            if stop:
                return []                                   # cap already hit: skip remaining clusters
            try:
                if compose_only:
                    n = len(CJ.gpt_break([pairs[g] for g in gidxs])); return [{"_proposed": n}]
                return CJ.process_cluster([pairs[g] for g in gidxs], max_refine)
            except BudgetExceeded as e:
                stop["budget"] = str(e); return []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(do, clusters))
        candidates = [a for arts in res for a in arts]      # agent proposed + refined (still lenient)
        if compose_only:
            print(f"[{dialect}] COMPOSE-ONLY round {rnd}: {len(clusters)} clusters -> "
                  f"{sum(a['_proposed'] for a in candidates)} proposed articles cached in {CJ._CACHE_DIR}"
                  f"{' | ' + stop['budget'] if stop else ''}", flush=True)
            return [], rnd
        # RUBRIC-GATE each candidate; keep only genuine coh>=4, RECYCLE the rest (sentences stay in the pool)
        def vf(a):
            if stop.get("gate"):
                return a, 0, {}                             # judge cap hit: recycle, do not keep unjudged
            try:
                coh, rb = rubric_verify(a["pairs"]); return a, coh, rb
            except BudgetExceeded as e:
                stop["budget"] = str(e); stop["gate"] = True; return a, 0, {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            verds = list(ex.map(vf, candidates))
        round_kept = []
        for a, coh, rb in verds:
            if coh >= 4:
                a["rubric"] = rb
                round_kept.append(a)
                for p in a["pairs"]:
                    used.add(norm(p["ar"]))                 # ONLY verified-kept sentences leave the pool
        kept_all += round_kept
        if on_round_kept and round_kept:
            on_round_kept(round_kept)                       # persist NOW, not at the end of the run
        alive = [gi for gi in alive if norm(pairs[gi]["ar"]) not in used]
        print(f"[{dialect}] round {rnd}: {len(clusters)} clusters, {len(candidates)} candidates -> "
              f"KEPT {len(round_kept)} rubric-coh>=4 ({len(candidates)-len(round_kept)} recycled) | "
              f"total {len(kept_all)} | leftover {len(alive)}", flush=True)
        if stop:
            print(f"[{dialect}] STOPPING after round {rnd}: {stop['budget']} -- rerun with LOOP_RESUME=1 "
                  f"(default) to continue from {out_hint(dialect)}", flush=True)
            break
        if len(round_kept) < dry:
            break
    return kept_all, rnd


def out_hint(dialect):
    return os.environ.get("LOOP_OUT_HINT", f"loop_{dialect}.jsonl")


def article_row(dialect, a):
    return {"dialect": dialect, "source": "cj-loop", "rubric_claude": a["rubric"], "n": len(a["pairs"]),
            "dialect_article": [p["ar"] for p in a["pairs"]], "gold_reference": [p["en"] for p in a["pairs"]]}


if __name__ == "__main__":
    from collections import Counter

    dialect = sys.argv[1]; out = Path(sys.argv[2])
    ROUNDS = int(os.environ.get("LOOP_ROUNDS", "6"))
    DRY = int(os.environ.get("LOOP_DRY", "5"))
    CSIZE = int(os.environ.get("CLUSTER_SIZE", "150"))
    PCA_DIM = int(os.environ.get("PCA_DIM", "50"))
    MIN_KEEP = int(os.environ.get("MIN_KEEP", "10"))
    MAXREF = int(os.environ.get("ASM_MAX_REFINE", "2"))
    WORKERS = int(os.environ.get("ASM_WORKERS", "8"))

    pairs = [json.loads(l) for l in open(f"poolfull_{dialect}.jsonl", encoding="utf-8")]
    E = np.load(os.environ.get("EMB_FILE", f"emb_poolfull_{dialect}.npy"))   # ablation: swap AR/EN embeddings
    assert len(E) == len(pairs), f"emb_poolfull_{dialect}.npy has {len(E)} rows but the pool has {len(pairs)}"
    os.environ["LOOP_OUT_HINT"] = str(out)

    # RESUME: retire the sentences of already-kept articles and append new rounds to the same file
    prior = []
    if out.exists() and os.environ.get("LOOP_RESUME", "1") != "0":
        prior = [json.loads(l) for l in out.open(encoding="utf-8") if l.strip()]
    used_init = {norm(sent) for d in prior for sent in d["dialect_article"]}
    if prior:
        print(f"[{dialect}] RESUME: {len(prior)} articles already in {out} ({len(used_init)} sentences retired)", flush=True)
    print(f"[{dialect}] proposer={CJ.PROPOSER_SYSTEM}:{CJ.GPT.model} judge={CJ.JUDGE_SYSTEM}:{CJ.CLAUDE.model} "
          f"src_lang={SRC_LANG!r} rounds={ROUNDS} dry={DRY} csize={CSIZE} pool={len(pairs)}", flush=True)

    fh = out.open("a", encoding="utf-8")
    def persist(round_kept):
        for a in round_kept:
            fh.write(json.dumps(article_row(dialect, a), ensure_ascii=False) + "\n")
        fh.flush()
    kept_all, rnd = run_loop(dialect, pairs, E, rounds=ROUNDS, dry=DRY, csize=CSIZE, pca_dim=PCA_DIM,
                              min_keep=MIN_KEEP, max_refine=MAXREF, workers=WORKERS,
                              used_init=used_init, on_round_kept=persist)
    fh.close()
    total = len(prior) + len(kept_all)
    print(f"[{dialect}] DONE: {len(kept_all)} new rubric-verified articles over {rnd} rounds "
          f"({total} in file) | coh {dict(sorted(Counter(a['rubric']['coherence'] for a in kept_all).items()))} -> {out}",
          flush=True)
    print(usage_summary(), flush=True)
