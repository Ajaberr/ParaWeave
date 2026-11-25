"""GPT-5.6-sol (Bedrock Mantle) Lugbara summaries at one reasoning-effort level, both input conditions, with the exact
prompt and record format of eval/official/gen.py, written line by line so a failed call never loses the run
(rerun the same command to fill any gaps).

  python effort_gen.py <effort>        effort in none | low | medium | high | xhigh
  -> eval/official/out/gen.mlext.openai-gpt56sol-e<effort>.jsonl"""
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = r"Documents\Work\lab\ArabicDetection"
EVAL = os.path.join(ROOT, "eval", "official")
EFF = sys.argv[1]
assert EFF in ("none", "low", "medium", "high", "xhigh"), EFF
os.environ["ML_EXTENSION_FILE"] = os.path.join(ROOT, "datasets", "official", "derived", "DiaBench-ML", "lugbara_eval.jsonl")
os.environ["GEN_SRC_LANG"] = "Lugbara"
os.environ["OPENAI_MODEL"] = "openai.gpt-5.6-sol"
os.environ["OPENAI_REASONING_EFFORT"] = EFF
os.environ["OPENAI_MAX_COMPLETION"] = "32000"          # room for xhigh reasoning; billed on use only
os.environ.setdefault("LLM_BUDGET_USD_OPENAI", "30")   # per level
os.environ["LLM_USAGE_LOG"] = os.path.join(ROOT, "datasets", "official", "derived", "DiaBench-ML", "logs", f"usage_gpt56sol_effort_{EFF}.jsonl")   # per level
os.chdir(EVAL); sys.path.insert(0, EVAL)
import gen, data as dio, llm                            # gen.py's own prompt, templates and condition sources
SYSTEM = f"openai-gpt56sol-e{EFF}"
OUT = os.path.join(EVAL, "out", f"gen.mlext.{SYSTEM}.jsonl")
recs = dio.load_records("mlext", limit=10 ** 9)
jobs = [(r, c) for r in recs for c in ("dialect", "gold_ref") if gen.COND_SOURCE[c](r).strip()]
done = set()
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        try:
            x = json.loads(l); done.add((x["id"], x["condition"]))
        except Exception:
            pass
todo = [(r, c) for r, c in jobs if (r["id"], c) not in done]
L = llm.LLM(SYSTEM)
assert L.backend == "openai" and L.reasoning and "mantle" in str(L.client.base_url), (L.backend, L.reasoning, L.client.base_url)
print(f"[{EFF}] {len(recs)} articles, {len(jobs)} summaries, {len(done)} done, {len(todo)} to do | model {L.model}", flush=True)
lock = threading.Lock(); fh = open(OUT, "a", encoding="utf-8"); n_ok = n_err = 0
def one(rc):
    r, c = rc
    user = gen._tmpl(r["dialect"]).format(dialect=r["dialect"], maxw=30, src=gen.COND_SOURCE[c](r)[:8000])
    return rc, L.complete_text(gen.SYS_PROMPT, user, temperature=0.3, max_tokens=90).strip()
t0 = time.time()
with ThreadPoolExecutor(max_workers=int(os.environ.get("GEN_WORKERS", "6"))) as ex:
    futs = [ex.submit(one, rc) for rc in todo]
    for f in as_completed(futs):
        try:
            (r, c), summ = f.result()
        except llm.BudgetExceeded as e:
            print("BUDGET", e, flush=True); break
        except Exception as e:
            n_err += 1; print(f"  ERR {type(e).__name__}: {str(e)[:120]}", flush=True); continue
        row = {"id": r["id"], "corpus": "mlext", "dialect": r["dialect"], "system": SYSTEM, "model": L.model,
               "condition": c, "summary": summ, "n_words": len(summ.split()), "source_text": r["source_text"],
               "source_sents": r["source_sents"], "gold_reference": r.get("gold_reference"),
               "gold_summary": r.get("gold_summary"), "en_reference": r.get("en_reference"), "reasoning_effort": EFF}
        with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush(); n_ok += 1
            if n_ok % 60 == 0:
                print(f"  [{EFF}] {n_ok}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
fh.close()
print(f"[{EFF}] wrote {n_ok}, errors {n_err}, file now {sum(1 for _ in open(OUT, encoding='utf-8'))} lines", flush=True)
print(llm.usage_summary(), flush=True)
