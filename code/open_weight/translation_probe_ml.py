#!/usr/bin/env python
"""Sentence-level translation probe for every build: each open-weight system translates every source sentence of the
released articles into English, and the output is scored against the human English sentence (chrF) afterwards.
Same prompt as eval/official/translation_probe.py with the language name swapped in, greedy decoding.

  python translation_probe_ml.py <lang> <system> <out.jsonl>
  env PROBE_LANG_NAME (e.g. "Dari (Afghan Persian)"), PROBE_MAX_TOKENS, and either VLLM_BASE_URL+VLLM_MODEL (local
  vLLM server) or HF_MODEL (in-process transformers). Open-weight only; llm.py's API branches are never taken.

RESUMABLE: appends per chunk, keyed on (id, sent_idx)."""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from llm import LLM  # noqa: E402
for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):   # llm.py may load a .env on import: drop any
    os.environ.pop(k, None)                                            # paid-endpoint key it found

LANG = os.environ.get("PROBE_LANG_NAME", "Arabic")
SYS_PROMPT = (f"You are a precise translator. Translate the given {LANG} sentence into English. "
              "Output ONLY the English translation as plain text -- no preamble, no quotes, no "
              "commentary, no transliteration.")
USER_TMPL = LANG + ":\n{src}\n\nEnglish:"
MAX_TOKENS = int(os.environ.get("PROBE_MAX_TOKENS", "160"))


def main():
    lang, system, out = sys.argv[1], sys.argv[2], sys.argv[3]
    ids = None
    if os.environ.get("PROBE_IDS"):                  # optional id whitelist (one id per line)
        ids = {l.strip() for l in open(os.environ["PROBE_IDS"], encoding="utf-8") if l.strip()}
    items = []
    for l in open(os.path.join(HERE, "ow_articles.jsonl"), encoding="utf-8"):
        a = json.loads(l)
        if a["lang"] != lang or (ids is not None and a["id"] not in ids):
            continue
        for j, (s, e) in enumerate(zip(a["src_sents"], a["en_sents"])):
            items.append({"lang": lang, "id": a["id"], "dialect": a["dialect"], "sent_idx": j, "src": s, "en_gold": e})
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            try:
                r = json.loads(l); done.add((r["id"], int(r["sent_idx"])))
            except Exception:
                pass
    todo = [it for it in items if (it["id"], it["sent_idx"]) not in done]
    print(f"[{lang}/{system}] {len(items)} sentences, {len(done)} done, {len(todo)} to translate ({LANG})", flush=True)
    if not todo:
        return
    assert not system.startswith(("openai", "claude", "anthropic", "bedrock")) and system not in ("qwen", "qwen3", "qwen3-8b"), \
        "local open-weight systems only"
    llm = LLM(system)
    assert (llm.backend == "hf") or (llm.backend == "openai" and os.environ.get("VLLM_BASE_URL", "").startswith("http://localhost")), \
        "local models only"
    workers = int(os.environ.get("GEN_WORKERS", "32"))
    n_long = n = 0
    with open(out, "a", encoding="utf-8") as fh:
        for s in range(0, len(todo), 512):
            ch = todo[s:s + 512]
            hyps = llm.complete_text_many(SYS_PROMPT, [USER_TMPL.format(src=it["src"]) for it in ch],
                                          temperature=0.0, max_tokens=MAX_TOKENS, workers=workers,
                                          label=f"translate[{lang}/{system}] {s}+ ")
            for it, h in zip(ch, hyps):
                h = (h or "").strip()
                fh.write(json.dumps({**it, "system": system, "en_hyp": h}, ensure_ascii=False) + "\n")
                n += 1
                n_long += bool(h) and len(h.split()) > 2 * max(1, len(it["en_gold"].split()))
            fh.flush(); os.fsync(fh.fileno())
            print(f"  [{lang}/{system}] {s + len(ch)}/{len(todo)} written", flush=True)
    print(f"[{lang}/{system}] done: {n} translations, {100 * n_long / max(1, n):.1f}% longer than twice the reference", flush=True)


if __name__ == "__main__":
    main()
