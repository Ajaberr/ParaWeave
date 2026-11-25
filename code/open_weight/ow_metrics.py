#!/usr/bin/env python
"""Open-weight faithfulness anchors on the released articles of all five builds (Arabic, Lugbara, Farsi, Dari, Tajik).
No API calls: every model here is a local open-weight checkpoint.

  python ow_metrics.py <mode>
    align      AlignScore-large (nli_sp), English article as premise, whole summary as hypothesis   [astest env]
    minicheck  MiniCheck-Flan-T5-Large, English article as premise, each claim                     [project env]
    mdeberta   mDeBERTa-v3-base-mnli-xnli, each claim against every sentence of the source article
               and of the English article, max-pooled per claim (SummaC-ZS style, no truncation)   [project env]
    e5         multilingual-e5-large-instruct evidence retrieval: best cosine of each claim over the
               source sentences and over the English sentences of its article                      [project env]
    pairs      known-true translation pairs per variety: e5 top-1 recovery among 300 same-variety
               sources, mDeBERTa true-pair vs shuffled-pair entailment                              [project env]
    bleurt     BLEURT-20-D12 of every summary against GPT-5.5's English-input summary              [venv-qwen]

Reads ow_articles.jsonl / ow_rows.jsonl next to this file and appends one line per summary to out/<mode>.jsonl,
keyed on lang|system|id|condition, so a restart skips what is done."""
from __future__ import annotations
import json, os, sys, time, random

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out")
os.makedirs(OUTD, exist_ok=True)
ALIGN_CKPT = os.environ.get("ALIGN_CKPT", "$WORK/AlignScore-large.ckpt")
INSTR = "Instruct: Retrieve the source sentence that states the claim\nQuery: "
PAIR_INSTR = INSTR


def load():
    arts = {}
    for l in open(os.path.join(HERE, "ow_articles.jsonl"), encoding="utf-8"):
        a = json.loads(l)
        arts[(a["lang"], a["id"])] = a
    rows = [json.loads(l) for l in open(os.path.join(HERE, "ow_rows.jsonl"), encoding="utf-8")]
    return arts, rows


def rkey(r):
    return "|".join((r["lang"], r["system"], r["id"], r["condition"]))


def todo_rows(rows, path):
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["key"])
            except Exception:
                pass
    todo = [r for r in rows if rkey(r) not in done]
    print(f"{len(rows)} summaries | done {len(done)} | to do {len(todo)}", flush=True)
    return todo


def write(fh, recs):
    for o in recs:
        fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def run_align(arts, rows):
    # OW_REVERSE=1: a second worker that takes the remaining rows from the end, into align.rev.jsonl
    rev = bool(os.environ.get("OW_REVERSE"))
    path = os.path.join(OUTD, "align.rev.jsonl" if rev else "align.jsonl")
    if rev:   # what the forward worker has not reached yet, taken from the end
        rows = todo_rows(rows, os.path.join(OUTD, "align.jsonl"))[::-1]
    todo = todo_rows(rows, path)
    if not todo:
        return
    from alignscore import AlignScore
    scorer = AlignScore(model="roberta-large", batch_size=32, device="cuda:0", ckpt_path=ALIGN_CKPT,
                        evaluation_mode="nli_sp")
    fh = open(path, "a", encoding="utf-8")
    t0 = time.time()
    for s in range(0, len(todo), 256):
        ch = todo[s:s + 256]
        idx = [i for i, r in enumerate(ch) if r["summary"].strip()]
        sc = scorer.score(contexts=[" ".join(arts[(ch[i]["lang"], ch[i]["id"])]["en_sents"]) for i in idx],
                          claims=[ch[i]["summary"] for i in idx]) if idx else []
        val = dict(zip(idx, sc))
        write(fh, [{"key": rkey(r), "align_en": (float(val[i]) if i in val else None)} for i, r in enumerate(ch)])
        print(f"  align {s + len(ch)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


def _claim_units(ch, arts):
    """unique (lang, id, claim) units of a chunk of summaries, and each summary's list of unit indices"""
    units, index, per_row = [], {}, []
    for r in ch:
        ids = []
        for c in r["claims"]:
            k = (r["lang"], r["id"], c)
            if k not in index:
                index[k] = len(units)
                units.append(k)
            ids.append(index[k])
        per_row.append(ids)
    return units, per_row


def run_minicheck(arts, rows):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    path = os.path.join(OUTD, "minicheck.jsonl")
    todo = todo_rows(rows, path)
    if not todo:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    ckpt = "lytang/MiniCheck-Flan-T5-Large"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).cuda().eval()
    bs = int(os.environ.get("MC_BATCH", "32"))
    fh = open(path, "a", encoding="utf-8")
    t0 = time.time()
    for s in range(0, len(todo), 1000):
        ch = todo[s:s + 1000]
        units, per_row = _claim_units(ch, arts)
        # MiniCheck's own input: "predict: " + document chunk + eos + claim; our English articles are one chunk each
        texts = ["predict: " + " ".join(arts[(L, i)]["en_sents"]) + tok.eos_token + c for (L, i, c) in units]
        order = sorted(range(len(texts)), key=lambda j: len(texts[j]))
        probs = [None] * len(texts)
        for b in range(0, len(order), bs):
            jj = order[b:b + bs]
            enc = tok([texts[j] for j in jj], max_length=2048, truncation=True, padding=True, return_tensors="pt").to("cuda")
            dec = torch.zeros((enc["input_ids"].size(0), 1), dtype=torch.long, device="cuda")
            with torch.no_grad():
                lg = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], decoder_input_ids=dec).logits.squeeze(1)
                p = torch.softmax(lg[:, [3, 209]].float(), dim=-1)[:, 1].cpu().tolist()
            for j, v in zip(jj, p):
                probs[j] = round(v, 4)
        write(fh, [{"key": rkey(r), "mc_en": [probs[u] for u in ids]} for r, ids in zip(ch, per_row)])
        print(f"  minicheck {s + len(ch)}/{len(todo)} ({time.time() - t0:.0f}s, {len(units)} claims)", flush=True)


_NLI = {}


def nli_model():
    if not _NLI:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        tok = AutoTokenizer.from_pretrained(name)
        m = AutoModelForSequenceClassification.from_pretrained(name).cuda().eval()
        lab = {v.lower(): int(k) for k, v in m.config.id2label.items()}
        _NLI.update(tok=tok, m=m, ent=lab["entailment"], con=lab["contradiction"])
    return _NLI


def nli_scores(prem, hyp, bs=256):
    """entailment and contradiction probabilities (softmax over the three NLI labels) for each pair"""
    import torch
    d = nli_model()
    order = sorted(range(len(prem)), key=lambda j: len(prem[j]) + len(hyp[j]))
    ent, con = [0.0] * len(prem), [0.0] * len(prem)
    for b in range(0, len(order), bs):
        jj = order[b:b + bs]
        enc = d["tok"]([prem[j] for j in jj], [hyp[j] for j in jj], truncation=True, max_length=320,
                       padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pr = torch.softmax(d["m"](**enc).logits.float(), dim=-1).cpu()
        for j, row in zip(jj, pr):
            ent[j] = float(row[d["ent"]]); con[j] = float(row[d["con"]])
    return ent, con


def run_mdeberta(arts, rows):
    path = os.path.join(OUTD, "mdeberta.jsonl")
    todo = todo_rows(rows, path)
    if not todo:
        return
    fh = open(path, "a", encoding="utf-8")
    t0 = time.time()
    for s in range(0, len(todo), 600):
        ch = todo[s:s + 600]
        units, per_row = _claim_units(ch, arts)
        prem, hyp, back = [], [], []            # back = (unit, side)
        for u, (L, i, c) in enumerate(units):
            a = arts[(L, i)]
            for side, sents in (("src", a["src_sents"]), ("en", a["en_sents"])):
                for sent in sents:
                    prem.append(sent); hyp.append(c); back.append((u, side))
        ent, con = nli_scores(prem, hyp)
        best = {}
        for (u, side), e, k in zip(back, ent, con):
            b = best.setdefault((u, side), [0.0, 0.0])
            if e > b[0]: b[0] = e
            if k > b[1]: b[1] = k
        out = []
        for r, ids in zip(ch, per_row):
            out.append({"key": rkey(r),
                        "nli_src": [round(best[(u, "src")][0], 4) for u in ids],
                        "nli_en": [round(best[(u, "en")][0], 4) for u in ids],
                        "con_src": [round(best[(u, "src")][1], 4) for u in ids],
                        "con_en": [round(best[(u, "en")][1], 4) for u in ids]})
        write(fh, out)
        print(f"  mdeberta {s + len(ch)}/{len(todo)} ({time.time() - t0:.0f}s, {len(prem)} pairs)", flush=True)


_E5 = {}


def e5_encode(texts, query=False):
    import numpy as np
    if not _E5:
        from sentence_transformers import SentenceTransformer
        _E5["m"] = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device="cuda")
    inp = [INSTR + t for t in texts] if query else list(texts)
    return np.asarray(_E5["m"].encode(inp, normalize_embeddings=True, batch_size=128, show_progress_bar=False,
                                      convert_to_numpy=True), dtype="float32")


def run_e5(arts, rows):
    import numpy as np
    path = os.path.join(OUTD, "e5.jsonl")
    todo = todo_rows(rows, path)
    if not todo:
        return
    fh = open(path, "a", encoding="utf-8")
    need = sorted({(r["lang"], r["id"]) for r in todo})
    sent_emb = {}
    t0 = time.time()
    flat, where = [], []
    for k in need:
        a = arts[k]
        for side in ("src", "en"):
            ss = a[side + "_sents"]
            where.append((k, side, len(flat), len(flat) + len(ss)))
            flat.extend(ss)
    E = e5_encode(flat)
    for k, side, a0, a1 in where:
        sent_emb[(k, side)] = E[a0:a1]
    print(f"  e5 sentences encoded: {len(flat)} ({time.time() - t0:.0f}s)", flush=True)
    for s in range(0, len(todo), 4000):
        ch = todo[s:s + 4000]
        units, per_row = _claim_units(ch, arts)
        Q = e5_encode([c for (_, _, c) in units], query=True)
        mx = {}
        for u, (L, i, c) in enumerate(units):
            for side in ("src", "en"):
                P = sent_emb[((L, i), side)]
                mx[(u, side)] = float((P @ Q[u]).max()) if len(P) else 0.0
        write(fh, [{"key": rkey(r), "e5_src": [round(mx[(u, "src")], 4) for u in ids],
                    "e5_en": [round(mx[(u, "en")], 4) for u in ids]} for r, ids in zip(ch, per_row)])
        print(f"  e5 {s + len(ch)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


def run_pairs(arts, rows):
    """Known-true pairs, same protocol as sanity_dialect_e5.py / sanity_dialect_nli.py: per variety, 300 aligned
    (source sentence, English translation) pairs of at least 15 characters with unique sources, seed 0."""
    import numpy as np, statistics
    groups = {}
    for (L, i), a in sorted(arts.items()):
        g = a["dialect"] if L == "ar" else L
        for s_, e_ in zip(a["src_sents"], a["en_sents"]):
            s_, e_ = (s_ or "").strip(), (e_ or "").strip()
            if len(s_) >= 15 and len(e_) >= 15:
                groups.setdefault(g, []).append((s_, e_))
    res = {}
    for g, pairs in sorted(groups.items()):
        rnd = random.Random(0); rnd.shuffle(pairs)
        seen, uniq = set(), []
        for s_, e_ in pairs:
            if s_ in seen:
                continue
            seen.add(s_); uniq.append((s_, e_))
            if len(uniq) >= 300:
                break
        srcs, golds = [p[0] for p in uniq], [p[1] for p in uniq]
        n = len(uniq)
        qe = e5_encode(golds, query=True); pe = e5_encode(srcs)
        sims = pe @ qe.T
        top1 = sum(1 for j in range(n) if int(np.argmax(sims[:, j])) == j)
        top5 = sum(1 for j in range(n) if j in np.argsort(-sims[:, j])[:5])
        neg = golds[:]; random.Random(1).shuffle(neg)
        for j in range(n):
            if neg[j] == golds[j] and n > 1:
                k = (j + 1) % n; neg[j], neg[k] = neg[k], neg[j]
        te, _ = nli_scores(srcs, golds); ne, _ = nli_scores(srcs, neg)
        ee, _ = nli_scores(golds, golds)          # English-to-English ceiling of the same NLI model
        res[g] = {"n": n, "e5_top1": round(top1 / n, 4), "e5_top5": round(top5 / n, 4),
                  "nli_true_rate": round(sum(1 for v in te if v >= 0.5) / n, 4),
                  "nli_true_mean": round(statistics.mean(te), 4),
                  "nli_neg_rate": round(sum(1 for v in ne if v >= 0.5) / n, 4),
                  "nli_neg_mean": round(statistics.mean(ne), 4),
                  "nli_en_self_rate": round(sum(1 for v in ee if v >= 0.5) / n, 4)}
        print(g, res[g], flush=True)
    json.dump(res, open(os.path.join(OUTD, "pairs.json"), "w", encoding="utf-8"), indent=1)


def run_bleurt(arts, rows):
    import torch
    from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
    path = os.path.join(OUTD, "bleurt.jsonl")
    ref = {(r["lang"], r["id"]): r["summary"] for r in rows if r["system"] == "openai" and r["condition"] == "gold_ref"}
    todo = todo_rows(rows, path)
    if not todo:
        return
    name = "lucadiliello/BLEURT-20-D12"
    tok = BleurtTokenizer.from_pretrained(name)
    m = BleurtForSequenceClassification.from_pretrained(name).cuda().eval()
    fh = open(path, "a", encoding="utf-8")
    t0 = time.time()
    for s in range(0, len(todo), 2048):
        ch = todo[s:s + 2048]
        idx = [j for j, r in enumerate(ch) if r["summary"].strip() and ref.get((r["lang"], r["id"]), "").strip()]
        vals = {}
        for b in range(0, len(idx), 64):
            jj = idx[b:b + 64]
            enc = tok([ref[(ch[j]["lang"], ch[j]["id"])] for j in jj], [ch[j]["summary"] for j in jj],
                      padding="longest", truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.no_grad():
                for j, v in zip(jj, m(**enc).logits.flatten().float().cpu().tolist()):
                    vals[j] = round(v, 4)
        write(fh, [{"key": rkey(r), "bleurt_ref": vals.get(j)} for j, r in enumerate(ch)])
        print(f"  bleurt {s + len(ch)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    arts, rows = load()
    print(f"[{mode}] {len(arts)} articles, {len(rows)} summaries  {time.strftime('%F %T')}", flush=True)
    {"align": run_align, "minicheck": run_minicheck, "mdeberta": run_mdeberta, "e5": run_e5,
     "pairs": run_pairs, "bleurt": run_bleurt}[mode](arts, rows)
    print(f"[{mode}] DONE {time.strftime('%F %T')}", flush=True)
