"""Paid part of the Acholi/Ateso run (approved by the user 2026-09-25, about $50):
  1. GPT-5.5 source-input summaries (Bedrock Mantle, gen.py unchanged, as the Lugbara run);
  2. Claude Sonnet 4.6 batch judging, as effort_sweep/lgg_judge.sh:
     - reused English-input summaries (frozen Lugbara claims) against the new source article only;
     - source-input summaries: decompose once, then against the source and the English article.
Open-weight source-input summaries are pulled from the VM (/tmp/work/nilo) as each run finishes.
Per-process budget caps; usage logged to datasets/official/derived/DiaBench-ML/logs/usage_nilo_*.jsonl.

  python eval/official/nilo/nilo_paid.py        (resumable: finished outputs are skipped)"""
import os, shutil, subprocess, sys, time
ROOT = r"Documents\Work\lab\ArabicDetection"
EV = os.path.join(ROOT, "eval", "official"); N = os.path.join(EV, "out", "nilo")
DML = os.path.join(ROOT, "datasets", "official", "derived", "DiaBench-ML"); LOGS = os.path.join(DML, "logs")
LANGS = {"ach": ("Acholi", "acholi_eval.jsonl"), "teo": ("Ateso", "ateso_eval.jsonl")}
VMSYS = ["gptoss20", "qwen30a3b", "qwen8", "fanar", "aya", "sunflower9b", "sunflower32q4"]
SYSTEMS = ["openai"] + VMSYS
MAXJ = 8
base_env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
nlines = lambda p: sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else 0

def judge_cmd(L, gen_file, grounds, cap):
    name, evf = LANGS[L]
    env = dict(base_env, ML_EXTENSION_FILE=os.path.join(DML, evf), JUDGE_SRC_LANG=name, DECOMP_SYSTEM="claude",
               ANTHROPIC_MODEL="claude-sonnet-4-6", CLAUDE_BATCH="1", JUDGE_WORKERS="8",
               LLM_BUDGET_USD_ANTHROPIC=str(cap), LLM_USAGE_LOG=os.path.join(LOGS, f"usage_nilo_judge_{L}.jsonl"))
    return [sys.executable, "judge_faith.py", gen_file, "claude", grounds], env

running, queue, launched = [], [], set()
def submit(key, cmd, env, logname):
    if key in launched: return
    launched.add(key); queue.append((key, cmd, env, logname))
def pump():
    for p in running[:]:
        if p[1].poll() is not None:
            running.remove(p); log("finished", p[0], "exit", p[1].returncode)
    while queue and len(running) < MAXJ:
        key, cmd, env, logname = queue.pop(0)
        fh = open(os.path.join(LOGS, logname), "a", encoding="utf-8")
        running.append((key, subprocess.Popen(cmd, cwd=EV, env=env, stdout=fh, stderr=subprocess.STDOUT))); log("started", key)

# 1. GPT-5.5 source-input summaries, one process per language
gen_procs = {}
for L, (name, evf) in LANGS.items():
    dst = os.path.join(N, L, "gen.mlext.openai.jsonl")
    if nlines(dst) >= 240: continue
    env = dict(base_env, ML_EXTENSION_FILE=os.path.join(DML, evf), GEN_SRC_LANG=name, GEN_OUT_TAG=f"nilo_{L}",
               LLM_BUDGET_USD_OPENAI="8", LLM_USAGE_LOG=os.path.join(LOGS, f"usage_nilo_gen_{L}.jsonl"))
    fh = open(os.path.join(LOGS, f"nilo_gen_{L}_openai.log"), "a", encoding="utf-8")
    gen_procs[L] = subprocess.Popen([sys.executable, "gen.py", "mlext", "openai", "dialect"], cwd=EV, env=env,
                                    stdout=fh, stderr=subprocess.STDOUT)
    log("GPT-5.5 generation started", L)

# 2. reused English-input summaries against the new source article (claims frozen, no decomposition)
for L in LANGS:
    for s in SYSTEMS:
        gf = os.path.join(N, L, f"gen.mlext.{s}_enin.jsonl")
        if os.path.exists(gf.replace(".jsonl", ".faith.claude.jsonl")): continue
        cmd, env = judge_cmd(L, gf, "dialect", 4)
        submit(f"{L}/{s}_enin", cmd, env, f"nilo_judge_{L}_{s}_enin.log")

# 3. source-input summaries as they appear, then decompose + judge against both articles
def src_ready(L, s):
    dst = os.path.join(N, L, f"gen.mlext.{s}.jsonl")
    if nlines(dst) >= 240: return dst
    if s == "openai":
        p = gen_procs.get(L); tmp = os.path.join(EV, "out", f"gen.nilo_{L}.openai.jsonl")
        if (p is None or p.poll() is not None) and nlines(tmp) >= 240:
            shutil.copy(tmp, dst); return dst
        return None
    return None

def pull_vm():
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "gpu-server-env2",
                              "cd /tmp/work/nilo && ls ach/done teo/done"], capture_output=True, text=True, timeout=90).stdout
    except Exception as e:
        log("ssh failed", e); return
    cur = None
    for line in out.splitlines():
        line = line.strip()
        if line.endswith(":"): cur = line.split("/")[0]; continue
        if not line or cur not in LANGS: continue
        dst = os.path.join(N, cur, f"gen.mlext.{line}.jsonl")
        if nlines(dst) >= 240: continue
        r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"gpu-server-env2:/tmp/work/nilo/{cur}/out/gen.mlext.{line}.jsonl", dst],
                           capture_output=True, text=True, timeout=180)
        log("pulled", cur, line, nlines(dst), "rows", r.stderr.strip()[:100])

t0 = time.time()
while True:
    pull_vm()
    for L in LANGS:
        for s in SYSTEMS:
            gf = src_ready(L, s)
            if gf and not os.path.exists(gf.replace(".jsonl", ".faith.claude.jsonl")):
                cmd, env = judge_cmd(L, gf, "dialect,gold_reference", 6)
                submit(f"{L}/{s}", cmd, env, f"nilo_judge_{L}_{s}.log")
    pump()
    need = [f"{L}/{s}" for L in LANGS for s in SYSTEMS] + [f"{L}/{s}_enin" for L in LANGS for s in SYSTEMS]
    done = [k for k in need if os.path.exists(os.path.join(N, k.split("/")[0], f"gen.mlext.{k.split('/')[1]}.faith.claude.jsonl"))]
    log(f"judged {len(done)}/{len(need)} | running {len(running)} | queued {len(queue)}")
    if len(done) == len(need) and not running: break
    if time.time() - t0 > 6 * 3600: log("timeout"); break
    time.sleep(60)
log("ALL DONE")
