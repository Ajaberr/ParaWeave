#!/usr/bin/env python
"""Unified LLM client for the eval harness. Both 'ChatGPT' and the local Qwen3-8B are reached through
ONE OpenAI-compatible client -- Qwen3-8B is served by vLLM at a localhost /v1 endpoint, so the only
difference is base_url + model. No heavy deps: `pip install openai`.

Env:
  OPENAI_API_KEY              (required for system='openai')
  OPENAI_MODEL                generation/judge model for ChatGPT (default gpt-4o-2024-08-06)
  QWEN_BASE_URL              vLLM endpoint (OpenAI mode) (default http://localhost:8001/v1)
  QWEN_MODEL                 model id: served name (vLLM) OR HF repo (default Qwen/Qwen3-8B-FP8)
  QWEN_HF=1                  run Qwen IN-PROCESS via HuggingFace transformers -- NO server. Set
                             QWEN_MODEL=Qwen/Qwen3-8B (bf16); CUDA_VISIBLE_DEVICES picks the GPU.
  ANTHROPIC_API_KEY          (required for system='claude')   pip install anthropic
  ANTHROPIC_MODEL            Claude model (default claude-opus-4-8; e.g. claude-sonnet-4-6 / haiku-4-5)
  ANTHROPIC_THINKING         disabled (default) | adaptive    -- adaptive = deeper but costs more
  OPENAI_BASE_URL            on Bedrock Mantle the gpt-5.x ids are ONLY served under
                             https://bedrock-mantle.us-east-1.api.aws/openai/v1 (the bare /v1 prefix answers
                             "isn't supported on this route"; it serves gpt-oss and GET /v1/models)
  system='claude-bedrock'    Claude served by AWS Bedrock Mantle through its Anthropic-protocol route
                             (https://bedrock-mantle.us-east-1.api.aws/anthropic, header
                             anthropic-workspace-id=<project>); auth = the same OPENAI_API_KEY /
                             OPENAI_PROJECT_ID as the GPT-5.5 composer. BEDROCK_CLAUDE_MODEL picks the id
                             (default anthropic.claude-opus-4-8). No Batches API there.
  LLM_USAGE_LOG              path: append one JSON line per call {ts, provider, model, in, out, usd}
  LLM_BUDGET_USD_<PROVIDER>  hard spend cap per provider key (OPENAI / ANTHROPIC / BEDROCK_ANTHROPIC),
                             estimated from list prices in PRICES; exceeding it raises BudgetExceeded
                             (never swallowed) so a run stops instead of silently degrading.
"""
from __future__ import annotations
import os, sys, json, re, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_HF_LOCK = threading.Lock()     # serialize HF .generate (one model, one GPU)
_HF_CACHE: dict = {}            # model_id -> (tokenizer, model); load once per process


def _harmony_final(raw: str) -> str:
    """gpt-oss emits the harmony format: <|channel|>analysis<|message|>...CoT...<|channel|>final<|message|>ANSWER.
    Keep ONLY the final-channel answer (drop the reasoning)."""
    m = re.findall(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|channel\|>|$)", raw, re.S)
    if m:
        return m[-1].strip()
    if "final<|message|>" in raw:                     # template variants
        return raw.split("final<|message|>")[-1].split("<|")[0].strip()
    return re.sub(r"<\|[^|]*\|>", "", raw).strip()    # last resort: strip markers


def _load_dotenv():
    """Minimal .env loader (no python-dotenv dep): first .env found among eval/, cwd, repo-root."""
    here = Path(__file__).resolve().parent
    # eval/official/.env, cwd/.env, eval/.env, then the repo root -- the loop is normally run from a
    # staging directory that holds the pool files, so the repo-root .env must be found from there too.
    for p in (here / ".env", Path.cwd() / ".env", here.parent / ".env", here.parent.parent / ".env"):
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = re.sub(r"\s+#.*$", "", v).strip().strip('"').strip("'")   # drop trailing comments
                    if v:
                        os.environ.setdefault(k.strip(), v)
            return


_load_dotenv()


# ---- token/cost accounting ------------------------------------------------------------------
# List prices, USD per 1M tokens (input, output), matched by substring on the lower-cased model id so
# provider-prefixed ids ("openai.gpt-5.5-2026-04-23", "anthropic.claude-opus-4-8") resolve too.
# Checked 2026-09-15 (BenchLM OpenAI table; Anthropic pricing pages). Unknown ids fall back to the
# conservative _PRICE_DEFAULT and are flagged "assumed" in the summary. Bedrock may bill slightly
# differently; the estimate is for the spend cap and the usage report, the invoice is authoritative.
PRICES = {
    "gpt-5.5": (5.00, 30.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
_PRICE_DEFAULT = (5.00, 30.00)
_USAGE_LOCK = threading.Lock()
_USAGE: dict = {}          # (provider, model) -> {"calls","in","out","usd","assumed"}


class BudgetExceeded(RuntimeError):
    """Raised (and never swallowed by complete_json) once LLM_BUDGET_USD_<PROVIDER> is exhausted."""


def _budget_key(provider):
    return "LLM_BUDGET_USD_" + provider.upper().replace("-", "_")


def _price(model):
    low = (model or "").lower()
    for k, v in PRICES.items():
        if k in low:
            return v, False
    return _PRICE_DEFAULT, True


def _record_usage(provider, model, n_in, n_out, mult=1.0):
    """mult=0.5 for the Message Batches API (half price); tokens are logged in full."""
    (pi, po), assumed = _price(model)
    usd = (n_in * pi + n_out * po) / 1e6 * mult
    with _USAGE_LOCK:
        u = _USAGE.setdefault((provider, model), {"calls": 0, "in": 0, "out": 0, "usd": 0.0, "assumed": assumed})
        u["calls"] += 1; u["in"] += n_in; u["out"] += n_out; u["usd"] += usd
        total_provider = sum(v["usd"] for (p, _), v in _USAGE.items() if p == provider)
        log = os.environ.get("LLM_USAGE_LOG")
        if log:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     "provider": provider, "model": model, "in": n_in, "out": n_out,
                                     "usd": round(usd, 6)}) + "\n")
    cap = os.environ.get(_budget_key(provider))
    if cap and total_provider > float(cap):
        raise BudgetExceeded(f"{provider}: estimated spend ${total_provider:.2f} exceeds cap ${float(cap):.2f} "
                             f"({_budget_key(provider)})")


def _budget_left(provider):
    cap = os.environ.get(_budget_key(provider))
    if not cap:
        return True
    with _USAGE_LOCK:
        spent = sum(v["usd"] for (p, _), v in _USAGE.items() if p == provider)
    return spent <= float(cap)


def usage_summary() -> str:
    """Human-readable per-model table of calls / tokens / estimated USD for this process."""
    if not _USAGE:
        return "LLM usage: no API calls made"
    lines = ["LLM usage (estimated at list prices):"]
    for (p, m), u in sorted(_USAGE.items()):
        lines.append(f"  {p:18s} {m:36s} calls={u['calls']:5d} in={u['in']:9d} out={u['out']:8d} "
                     f"~${u['usd']:.2f}{' (assumed price)' if u['assumed'] else ''}")
    by_p = {}
    for (p, _), u in _USAGE.items():
        by_p[p] = by_p.get(p, 0.0) + u["usd"]
    lines.append("  total per provider: " + ", ".join(f"{p} ~${v:.2f}" for p, v in sorted(by_p.items())))
    return "\n".join(lines)


import atexit as _atexit
_atexit.register(lambda: print(usage_summary(), file=sys.stderr, flush=True) if _USAGE else None)   # stderr: stdout may be eval'd


class LLM:
    """system in {'openai','qwen3-8b'}. complete_text / complete_json with retry+backoff."""

    def __init__(self, system: str):
        self.system = system
        self.backend = "openai"          # "openai" (OpenAI-compatible HTTP) or "hf" (in-process transformers)
        self.provider = system           # accounting key; see LLM_BUDGET_USD_<PROVIDER>
        self.extra = {}
        if system == "openai" or system.startswith("openai-"):
            # "openai-<tag>" = the same backend under a distinct system name, so several OpenAI model ids
            # can be run on one corpus without their gen.<corpus>.<system>.jsonl files colliding
            # (e.g. OPENAI_MODEL=openai.gpt-5.6-terra ... gen.py mlext openai-gpt56terra). 2026-09-21.
            from openai import OpenAI
            self.provider = "openai"
            self.client = OpenAI()                       # reads OPENAI_API_KEY
            self.model = os.environ.get("OPENAI_MODEL", "gpt-5.5-2026-04-23")
            # gpt-5*/o-series are REASONING models: temperature is pinned to 1, max_tokens ->
            # max_completion_tokens, and they take reasoning_effort.
            # Bedrock Mantle exposes the same models under prefixed ids (e.g. "openai.gpt-5.5..."),
            # so match the family name anywhere after a separator, not only at the start.
            self.reasoning = bool(re.search(r"(^|[./:_-])(gpt-5|o1|o3|o4)(?![a-z])", self.model.lower()))
        elif system in ("qwen3-8b", "qwen", "qwen3"):
            self.model = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B-FP8")
            self.reasoning = False
            if os.environ.get("QWEN_HF"):
                self.backend = "hf"                       # load the weights HERE -- no server
                self._load_hf(device_env="QWEN_DEVICE")
            else:
                from openai import OpenAI
                base = os.environ.get("QWEN_BASE_URL", "http://localhost:8001/v1")
                self.client = OpenAI(base_url=base, api_key=os.environ.get("QWEN_API_KEY", "EMPTY"))
                # disable Qwen3 "thinking" -- <think> blocks blow past max_tokens and wreck length control
                self.extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        elif system in ("claude-bedrock", "bedrock-claude"):
            # Claude on AWS Bedrock Mantle, billed to the lab's AWS credits. Mantle's OpenAI-compatible
            # /v1/chat/completions REJECTS Anthropic models (400 "does not support"); the working route
            # is the Anthropic Messages protocol at <host>/anthropic/v1/messages with the project id
            # sent as `anthropic-workspace-id` (the OpenAI-Project header is refused there) -- found
            # 2026-09-15. Same key as the GPT-5.5 composer. First calls return 403 "aws-marketplace:
            # Subscribe" until the account's model subscription finishes in the background.
            import anthropic
            self.backend = "anthropic"
            self.provider = "bedrock-anthropic"
            host = os.environ.get("OPENAI_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/openai/v1")
            host = re.match(r"https?://[^/]+", host).group(0)          # scheme + domain only
            base = os.environ.get("BEDROCK_ANTHROPIC_BASE_URL", host + "/anthropic")
            self.client = anthropic.Anthropic(
                api_key=os.environ["OPENAI_API_KEY"], base_url=base,
                default_headers={"anthropic-workspace-id": os.environ.get("OPENAI_PROJECT_ID", "")})
            self.model = os.environ.get("BEDROCK_CLAUDE_MODEL", "anthropic.claude-opus-4-8")
            self.reasoning = False
            self.think = os.environ.get("ANTHROPIC_THINKING", "disabled")
            self._temp_ok = not any(t in self.model for t in
                                    ("opus-4-7", "opus-4-8", "sonnet-5", "fable", "mythos"))
        elif system in ("claude", "anthropic"):
            import anthropic                                   # pip install anthropic
            self.backend = "anthropic"
            self.provider = "anthropic"
            self.client = anthropic.Anthropic()               # reads ANTHROPIC_API_KEY
            self.model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
            self.reasoning = False
            self.think = os.environ.get("ANTHROPIC_THINKING", "disabled")   # disabled | adaptive
            # Opus 4.7/4.8, Sonnet 5 and Fable 5 REJECT temperature (400). Sonnet 4.6 / Haiku accept it
            # -> send temperature=0 there so the judge is deterministic & reproducible.
            self._temp_ok = not self.model.startswith(
                ("claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5", "claude-fable", "claude-mythos"))
        elif os.environ.get("VLLM_BASE_URL"):
            # vLLM OpenAI-compatible server (e.g. gpt-oss MXFP4, Qwen3-30B TP) -- tag = system name
            from openai import OpenAI
            self.client = OpenAI(base_url=os.environ["VLLM_BASE_URL"], api_key="EMPTY")
            self.model = os.environ.get("VLLM_MODEL", system)
            self.reasoning = False
            eb = {}
            if os.environ.get("VLLM_REASONING_EFFORT"):        # gpt-oss: minimal CoT so the summary isn't eaten
                eb["reasoning_effort"] = os.environ["VLLM_REASONING_EFFORT"]
            if os.environ.get("VLLM_NO_THINK"):                # Qwen3-30B: disable <think>
                eb["chat_template_kwargs"] = {"enable_thinking": False}
            if eb:
                self.extra = {"extra_body": eb}
        elif system == "hf" or os.environ.get("HF_MODEL"):
            # ANY HF causal LM as a NAMED system so files don't collide:
            #   HF_MODEL=FreedomIntelligence/AceGPT-v2-8B-Chat python eval/gen.py ours acegpt
            # -> gen.ours.acegpt.jsonl . (Recognized systems above are matched first.)
            self.backend = "hf"
            self.model = os.environ["HF_MODEL"]
            self.reasoning = False
            self._load_hf(device_env="HF_DEVICE")
        else:
            raise ValueError(f"unknown system {system!r} (set HF_MODEL to route it to an HF model)")

    # ---- HuggingFace in-process backend (no server) -----------------------------------------
    def _load_hf(self, device_env: str | None = None):
        """device_env, if given, names an env var (e.g. "HF_DEVICE"="cuda:0") that PINS this model to
        ONE explicit device instead of accelerate's device_map="auto" sharding it across every
        CUDA_VISIBLE_DEVICES-visible GPU. Added 2026-08-20 after two reproducible `CUDA error: an
        illegal memory access` crashes loading gpt-oss-20b with device_map="auto" across 5 GPUs (both
        with GPUs otherwise idle, so not contention) -- transformers 5.10.2's new threaded
        core_model_loading.py materializes/copies tensors to their target device from a thread pool,
        and that cross-device copy path is where both crashes' stack traces bottomed out. Two separate
        single-GPU loads (one per model, via two different device_env names since proposer+judge load
        in the SAME process) sidesteps that code path entirely and also caps this process at exactly
        the number of distinct devices callers explicitly opt into -- see cj_assemble.py's
        HF_DEVICE/QWEN_DEVICE usage."""
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError("QWEN_HF=1 needs transformers+torch: "
                               "pip install -U 'transformers>=4.51' accelerate torch") from e
        device = os.environ.get(device_env) if device_env else None
        if self.model in _HF_CACHE:
            self.tok, self.hf = _HF_CACHE[self.model]
        else:
            print(f"[hf] loading {self.model} in-process (no server)"
                  f"{f' on {device}' if device else ''} ...")
            trc = bool(os.environ.get("HF_TRUST_REMOTE_CODE"))   # Jais & some Arabic models need this
            _hftok = os.environ.get("HF_TOKEN")                  # gated repos (e.g. Aya): pass token explicitly
            self.tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=trc, token=_hftok)
            if self.tok.pad_token_id is None:
                self.tok.pad_token = self.tok.eos_token
            self.tok.padding_side = "left"                # decoder-only batched gen needs LEFT pad
            mkw = dict(device_map="auto", trust_remote_code=trc, token=_hftok)
            if os.environ.get("HF_4BIT"):                 # 4-bit (bitsandbytes) so 30B fits one 49G GPU
                from transformers import BitsAndBytesConfig, AutoConfig
                # Some checkpoints (e.g. gpt-oss-20b) bake their OWN quantization_config (Mxfp4Config)
                # into config.json; transformers refuses to load if we also pass a BitsAndBytesConfig
                # ("model is quantized with Mxfp4Config but you are passing a BitsAndBytesConfig" --
                # hit for real 2026-08-20). Load the config ourselves and clear that field first so ours
                # is the only one in play -- the raw weights are dequantized to full precision during
                # loading regardless of which scheme's declared, same as the no-BnB fallback path
                # already does ("MXFP4 quantization requires the kernels package... dequantizing to
                # bf16"), so BnB re-quantizing that materialized bf16 tensor is not lossy beyond what
                # BnB itself already is.
                cfg = AutoConfig.from_pretrained(self.model, trust_remote_code=trc, token=_hftok)
                if getattr(cfg, "quantization_config", None) is not None:
                    cfg.quantization_config = None
                mkw["config"] = cfg
                mkw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
                print(f"[hf] loading {self.model} in 4-bit (nf4)")
            else:
                mkw["torch_dtype"] = "auto"
            if device:                                     # explicit single-device pin, see device_env docstring
                mkw["device_map"] = {"": device}
            elif os.environ.get("HF_GPU0"):                # force everything on GPU 0 (no CPU offload)
                mkw["device_map"] = {"": 0}               # e.g. gpt-oss dequantized bf16 (~40G) on a 49G card
            self.hf = AutoModelForCausalLM.from_pretrained(self.model, **mkw)
            self.hf.eval()
            _HF_CACHE[self.model] = (self.tok, self.hf)
            print(f"[hf] ready ({self.model})")
        self.hf_device = next(self.hf.parameters()).device

    def _hf_generate(self, sys_p, user_ps, *, max_tokens, temperature, json_mode, label=""):
        import torch
        texts = []
        for up in user_ps:
            if json_mode and "json" not in (sys_p + " " + up).lower():
                up = up + "\n\nReturn a single valid JSON object."
            msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": up}]
            low = self.model.lower()
            ct = {}                                           # model-specific chat-template kwargs
            if "qwen3" in low or "qwen-3" in low or "sunflower" in low:   # Sunflower = Qwen3 fine-tune
                ct["enable_thinking"] = False                 # Qwen3 thinking off
            elif "gpt-oss" in low:
                ct["reasoning_effort"] = os.environ.get("GPTOSS_EFFORT", "low")   # reasoning model -> minimal CoT
            try:
                t = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **ct)
            except TypeError:
                t = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                t = sys_p + "\n\n" + up + "\n"                # no chat template -> plain prompt
            texts.append(t)
        bs = int(os.environ.get("HF_BATCH", "8"))
        maxlen = int(os.environ.get("HF_MAXLEN", "4096"))
        do_sample = bool(temperature and temperature > 0)
        out = []
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                           max_length=maxlen).to(self.hf_device)
            gkw = dict(max_new_tokens=max_tokens, do_sample=do_sample,
                       pad_token_id=self.tok.pad_token_id)
            if do_sample:
                gkw.update(temperature=float(temperature), top_p=0.8)
            with torch.no_grad(), _HF_LOCK:
                gen = self.hf.generate(**enc, **gkw)
            plen = enc["input_ids"].shape[1]              # left-pad -> all rows share prompt length
            is_oss = "gpt-oss" in self.model.lower()
            for row in gen:
                if is_oss:                                 # harmony format -> keep only the 'final' channel
                    out.append(_harmony_final(self.tok.decode(row[plen:], skip_special_tokens=False)))
                else:
                    out.append(self.tok.decode(row[plen:], skip_special_tokens=True).strip())
            print(f"  {label}[hf] {min(i + bs, len(texts))}/{len(texts)}")
        return out

    @staticmethod
    def _parse_json(txt):
        try:
            return json.loads(txt)
        except Exception:
            m = re.search(r"\{.*\}", txt or "", re.S)      # salvage a JSON object from prose
            try:
                return json.loads(m.group(0)) if m else {}
            except Exception:
                return {}

    def _anthropic_params(self, sys_p, user_p, *, max_tokens, json_mode, temperature=0.0):
        """Build the Messages-API kwargs (shared by the live call and the Batches path)."""
        if json_mode and "json" not in (sys_p + " " + user_p).lower():
            user_p = user_p + "\n\nReturn a single valid JSON object."
        kw = dict(model=self.model, system=sys_p,
                  messages=[{"role": "user", "content": user_p}])
        if self.think == "adaptive":
            kw["thinking"] = {"type": "adaptive"}
            kw["max_tokens"] = max(max_tokens, int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096")))
        else:
            kw["thinking"] = {"type": "disabled"}
            kw["max_tokens"] = max(max_tokens, 512)
            if self._temp_ok and temperature is not None:
                kw["temperature"] = float(temperature)     # deterministic judge (Sonnet 4.6 accepts it)
        return kw

    @staticmethod
    def _retry(fn, what, tries=8):
        """Transient-network retry for batch control-plane calls. A single ConnectTimeout must NOT
        kill a multi-hour run -> retry with backoff; only a deterministic 4xx aborts."""
        import anthropic as _a
        for att in range(tries):
            try:
                return fn()
            except (_a.APITimeoutError, _a.APIConnectionError, _a.InternalServerError) as e:
                if att == tries - 1:
                    raise
                print(f"    [batch] transient {type(e).__name__} on {what}; retry {att+1}/{tries}", flush=True)
                time.sleep(min(60, 3 * 2 ** att))
            except _a.APIStatusError as e:
                if getattr(e, "status_code", None) in (400, 401, 403, 404, 422):
                    raise
                if att == tries - 1:
                    raise
                time.sleep(min(60, 3 * 2 ** att))

    def _anthropic_batch_json(self, sys_p, user_ps, *, max_tokens, temperature=0.0, label=""):
        """Message Batches API -- HALF PRICE, async (<=24h, usually <1h). Offline judging only.
        RESUMABLE: batch id is persisted keyed by request-content hash, so a restart re-attaches to the
        already-submitted batch (no double-spend). All control-plane calls retry on transient errors.
        Results come back in ANY order -> key by custom_id, never by position."""
        import hashlib
        reqs = [{"custom_id": f"r{i}",
                 "params": self._anthropic_params(sys_p, up, max_tokens=max_tokens,
                                                  json_mode=True, temperature=temperature)}
                for i, up in enumerate(user_ps)]
        out = [{} for _ in user_ps]
        CHUNK = 90_000                                     # API cap is 100k requests / 256MB per batch
        ckdir = Path(__file__).resolve().parent / "out" / "_batch_ckpt"; ckdir.mkdir(parents=True, exist_ok=True)
        for s in range(0, len(reqs), CHUNK):
            part = reqs[s:s + CHUNK]
            key = hashlib.sha256((self.model + "\x00" + sys_p + "\x00" +
                                  "\x00".join(u for _, u in [(0, r["params"]["messages"][0]["content"]) for r in part])
                                  ).encode()).hexdigest()[:20]
            ck = ckdir / f"{key}.json"
            bid = None
            if ck.exists():                                # RESUME: re-attach instead of resubmitting
                try:
                    bid = json.loads(ck.read_text()).get("batch_id")
                    self._retry(lambda: self.client.messages.batches.retrieve(bid), "resume-check")
                    print(f"  {label}[batch] RESUMING {bid} (found checkpoint)", flush=True)
                except Exception:
                    bid = None
            if not bid:
                b = self._retry(lambda: self.client.messages.batches.create(requests=part), "create")
                bid = b.id
                ck.write_text(json.dumps({"batch_id": bid, "n": len(part)}))
                print(f"  {label}[batch] submitted {bid}  ({len(part)} requests)", flush=True)
            while True:
                st = self._retry(lambda: self.client.messages.batches.retrieve(bid), "poll")
                if st.processing_status == "ended":
                    break
                c = st.request_counts
                print(f"  {label}[batch] {st.processing_status} "
                      f"ok={c.succeeded} err={c.errored} left={c.processing}", flush=True)
                time.sleep(30)
            ok = err = 0; b_in = b_out = 0
            for res in self._retry(lambda: list(self.client.messages.batches.results(bid)), "results"):
                k = int(res.custom_id[1:])
                if res.result.type == "succeeded":
                    txt = "".join(bl.text for bl in res.result.message.content
                                  if getattr(bl, "type", None) == "text")
                    out[k] = self._parse_json(txt); ok += 1
                    u = getattr(res.result.message, "usage", None)
                    if u is not None:
                        b_in += getattr(u, "input_tokens", 0) or 0; b_out += getattr(u, "output_tokens", 0) or 0
                else:
                    err += 1
            if b_in or b_out:                              # batch = half price (2026-09-17: was unaccounted)
                _record_usage(self.provider, self.model, b_in, b_out, mult=0.5)
            print(f"  {label}[batch] {bid} ended: {ok} ok, {err} failed", flush=True)
            ck.unlink(missing_ok=True)                     # done -> clear checkpoint
        return out

    def _anthropic_call(self, sys_p, user_p, *, max_tokens, json_mode, temperature=0.0):
        """Claude Messages API (live). Opus 4.7/4.8 reject temperature -> gated by _temp_ok;
        thinking tokens count against max_tokens, so give headroom when adaptive thinking is on."""
        kw = self._anthropic_params(sys_p, user_p, max_tokens=max_tokens, json_mode=json_mode,
                                    temperature=temperature)
        if not _budget_left(self.provider):
            raise BudgetExceeded(f"{self.provider}: spend cap already reached, refusing new call")
        for attempt in range(5):
            try:
                r = self.client.messages.create(**kw)
                u = getattr(r, "usage", None)
                _record_usage(self.provider, self.model, getattr(u, "input_tokens", 0) or 0,
                              getattr(u, "output_tokens", 0) or 0)
                return "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
            except BudgetExceeded:
                raise
            except Exception as e:
                if getattr(e, "status_code", None) in (400, 401, 403, 404, 422):
                    raise                              # deterministic client error -- don't retry
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    def _call(self, sys_p, user_p, *, temperature, max_tokens, json_mode):
        if self.backend == "anthropic":
            return self._anthropic_call(sys_p, user_p, max_tokens=max_tokens, json_mode=json_mode,
                                        temperature=temperature)
        kw = dict(model=self.model,
                  messages=[{"role": "system", "content": sys_p},
                            {"role": "user", "content": user_p}], **self.extra)
        if self.reasoning:
            # reasoning tokens count against the budget, so give generous headroom; the prompt keeps
            # the *visible* output short. temperature is fixed at 1 (omitted -> default).
            kw["max_completion_tokens"] = max(max_tokens, int(os.environ.get("OPENAI_MAX_COMPLETION", "4096")))
            eff = os.environ.get("OPENAI_REASONING_EFFORT", "low")
            if eff:
                kw["reasoning_effort"] = eff
        else:
            if not getattr(self, "no_temperature", False):
                kw["temperature"] = temperature
            kw["max_tokens"] = max_tokens
        if json_mode and not getattr(self, "no_response_format", False):
            kw["response_format"] = {"type": "json_object"}
        if not _budget_left(self.provider):
            raise BudgetExceeded(f"{self.provider}: spend cap already reached, refusing new call")
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(**kw)
                u = getattr(r, "usage", None)
                content = r.choices[0].message.content or ""
                if u is not None and getattr(u, "prompt_tokens", None):   # completion_tokens includes reasoning
                    _record_usage(self.provider, self.model, u.prompt_tokens or 0, u.completion_tokens or 0)
                else:                                          # no usage block: estimate (~4 chars/token) so the
                    _record_usage(self.provider, self.model,   # spend cap can never be bypassed silently
                                  (len(sys_p) + len(user_p)) // 4, max(len(content) // 4, 256))
                return content
            except BudgetExceeded:
                raise
            except Exception as e:
                if getattr(e, "status_code", None) in (400, 401, 403, 404, 422):
                    raise                              # deterministic client error -- don't retry
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    def complete_text(self, sys_p: str, user_p: str, *, temperature=0.2, max_tokens=256) -> str:
        if self.backend == "hf":
            return self.complete_text_many(sys_p, [user_p], temperature=temperature,
                                           max_tokens=max_tokens)[0]
        return self._call(sys_p, user_p, temperature=temperature, max_tokens=max_tokens,
                          json_mode=False).strip()

    def complete_json(self, sys_p: str, user_p: str, *, temperature=0.0, max_tokens=512) -> dict:
        if self.backend == "hf":
            return self.complete_json_many(sys_p, [user_p], temperature=temperature,
                                           max_tokens=max_tokens)[0]
        if "json" not in (sys_p + " " + user_p).lower():   # OpenAI json mode 400s without the word "json"
            user_p = user_p + "\n\nReturn a single valid JSON object."
        try:
            txt = self._call(sys_p, user_p, temperature=temperature, max_tokens=max_tokens,
                            json_mode=True)
        except BudgetExceeded:
            raise                                          # a spend cap must stop the run, not return {}
        except Exception:
            txt = ""
        return self._parse_json(txt)

    def complete_text_many(self, sys_p, user_ps, *, temperature=0.2, max_tokens=256,
                           workers=8, label=""):
        """Batched text completion. HF: one GPU-batched .generate. OpenAI: threaded API fan-out."""
        user_ps = list(user_ps)
        if self.backend == "hf":
            return self._hf_generate(sys_p, user_ps, max_tokens=max_tokens,
                                     temperature=temperature, json_mode=False, label=label)
        return parallel_map(lambda u: self._call(sys_p, u, temperature=temperature,
                            max_tokens=max_tokens, json_mode=False).strip(),
                            user_ps, workers=workers, label=label)

    def complete_json_many(self, sys_p, user_ps, *, temperature=0.0, max_tokens=512,
                           workers=8, label=""):
        """Batched JSON completion. HF: GPU-batched .generate + salvage. OpenAI: threaded API fan-out.
        CLAUDE_BATCH=1 (anthropic backend): Message Batches API -- HALF PRICE, async."""
        user_ps = list(user_ps)
        if self.backend == "hf":
            raws = self._hf_generate(sys_p, user_ps, max_tokens=max_tokens,
                                     temperature=temperature, json_mode=True, label=label)
            return [self._parse_json(t) for t in raws]
        if self.backend == "anthropic" and os.environ.get("CLAUDE_BATCH") and len(user_ps) > 1:
            return self._anthropic_batch_json(sys_p, user_ps, max_tokens=max_tokens,
                                              temperature=temperature, label=label)
        return parallel_map(lambda u: self.complete_json(sys_p, u, temperature=temperature,
                            max_tokens=max_tokens), user_ps, workers=workers, label=label)


def parallel_map(fn, items, workers: int = 8, every: int = 50, label: str = ""):
    """Run fn(item) over items concurrently, order-preserving, with a progress print."""
    out = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for n, f in enumerate(as_completed(futs), 1):
            out[futs[f]] = f.result()
            if n % every == 0:
                print(f"  {label}{n}/{len(items)}")
    return out
