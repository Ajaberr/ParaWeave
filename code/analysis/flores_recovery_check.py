"""Recompute the FLORES recovery-control table (app:flores) from the released FLORES articles and the pools' _group_id
(split:URL), and print the FLORES AUC values the rewritten ablation appendix uses."""
import json, collections, statistics as st
ML = r"datasets\official\derived\DiaBench-ML"
for tag in ("farsi", "dari", "tajik"):
    pool = [json.loads(l) for l in open(fr"{ML}/poolfull_{tag}.jsonl", encoding="utf-8")]
    gid = {p["en"].strip(): p["_group_id"] for p in pool}
    frag = collections.defaultdict(list)
    for i, p in enumerate(pool):
        frag[p["_group_id"]].append(p["en"].strip())
    qual = {g for g, v in frag.items() if len(v) >= 4}
    arts = [json.loads(l) for l in open(fr"{ML}\{tag}_eval.jsonl", encoding="utf-8")]
    single = whole = order = 0; shares = []; recovered = set(); miss = 0
    for a in arts:
        en = [x.strip() for x in a["gold_reference"]]
        gs = [gid.get(x) for x in en]
        miss += sum(g is None for g in gs)
        c = collections.Counter(g for g in gs if g)
        main, k = c.most_common(1)[0]
        shares.append(k / len(en))
        if len(c) == 1 and None not in gs:
            single += 1; recovered.add(main)
            src = frag[main]
            if len(en) == len(src): whole += 1
            if en == [x for x in src if x in en]: order += 1
    print(f"{tag:6s} articles {len(arts)} | single-fragment {single} ({100*single/len(arts):.0f}%) | whole {whole} | "
          f"original order {order} | mean main-source share {st.mean(shares):.3f} | distinct recovered {len(recovered)} "
          f"of {len(qual)} qualifying ({100*len(recovered)/len(qual):.0f}%) | fragments {len(frag)} | unmatched sents {miss}")

F = json.load(open(r"eval/official/out/embedding/flores_auc.json", encoding="utf-8"))["langs"]
DIA = ["arz_Arab", "apc_Arab", "ajp_Arab", "ars_Arab", "acm_Arab", "acq_Arab", "ary_Arab", "aeb_Arab"]
R = "lexical_topk_en"
v = lambda c, arm: F[c]["arms"][arm][R]["AUC_embedding"]
print("\nFLORES, same pairs (mined on English): lexical / none / scene")
for c in ("eng_Latn", "pes_Arab", "prs_Arab", "tgk_Cyrl", "arb_Arab"):
    print(f"  {F[c]['name']:10s} {F[c]['lexical'][R]:.3f} {v(c,'none'):.3f} {v(c,'scene'):.3f}")
m = lambda f: sum(f(c) for c in DIA) / len(DIA)
print(f"  {'Arab dial mean':10s} {m(lambda c: F[c]['lexical'][R]):.3f} {m(lambda c: v(c,'none')):.3f} {m(lambda c: v(c,'scene')):.3f}"
      f" | none range {min(v(c,'none') for c in DIA):.3f}-{max(v(c,'none') for c in DIA):.3f}"
      f" | lex range {min(F[c]['lexical'][R] for c in DIA):.3f}-{max(F[c]['lexical'][R] for c in DIA):.3f}")
