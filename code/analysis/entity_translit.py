"""a collaborator's question: are 'entity' errors on dialect-input summaries wrong names or transliteration variants?
Released (silver) Arabic articles only (ids of ow_judge 'ar'), all six systems, claims judged against the ENGLISH
article. For each entity-tagged claim, the capitalised words not found in the English article are compared with the
article's capitalised words by difflib ratio: >= 0.6 counts as a spelling variant of a name in the article
(transliteration), otherwise the name is absent from the article. Writes entity_translit.json next to the script."""
import json, glob, re, difflib, collections, os
ROOT = r"eval/official"
silver = {json.loads(l)["id"] for l in open(ROOT + r"/openweight/ow/ow_judge.jsonl", encoding="utf-8") if '"lang": "ar"' in l}
STOP = {"The", "A", "An", "He", "She", "They", "It", "I", "We", "You", "His", "Her", "Their", "This", "That", "In", "On", "At",
        "Mr", "Mrs", "Ms", "Dr", "Speaker", "Listener", "Addressee", "Customer", "Person", "Man", "Woman"}
caps = lambda t: {w for w in re.findall(r"\b[A-Z][a-zA-Z'\-]+\b", t) if w not in STOP}
stats = collections.Counter(); per_cond = collections.defaultdict(collections.Counter); examples = []
for f in glob.glob(ROOT + r"/out/roster_gen/gen.ours.*.faith.claude.jsonl"):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        aid = r["id"].replace("ours-", "", 1)
        if aid not in silver:
            continue
        fa = r.get("faith") or {}
        ref = next((fa[k] for k in ("gold_reference", "gold_ref", "english") if k in fa), None)
        if not ref:
            continue
        en = " ".join(r.get("gold_reference") or [])
        en_caps = caps(en); en_low = en.lower()
        for v in ref.get("verdicts", []):
            if (v.get("error") or "").split(":")[0].strip().lower() != "entity":
                continue
            per_cond[r["condition"]]["entity"] += 1
            new = [w for w in caps(v["claim"]) if w.lower() not in en_low]
            if not new:
                per_cond[r["condition"]]["no_new_name"] += 1; continue
            best = max((difflib.SequenceMatcher(None, w.lower(), e.lower()).ratio(), w, e) for w in new for e in en_caps) if en_caps else (0, new[0], "")
            kind = "variant" if best[0] >= 0.6 else "absent"
            per_cond[r["condition"]][kind] += 1
            if kind == "variant" and len(examples) < 12 and r["condition"] == "dialect":
                examples.append((best[1], best[2], round(best[0], 2), r["id"], r["system"]))
out = {c: dict(v) for c, v in per_cond.items()}
for c, v in out.items():
    n = v["entity"]
    print(f"{c:9s} entity-tagged {n:5d} | name spelled like one in the article {v.get('variant',0):4d} ({100*v.get('variant',0)/n:.0f}%) | "
          f"name absent from the article {v.get('absent',0):4d} ({100*v.get('absent',0)/n:.0f}%) | no new capitalised name {v.get('no_new_name',0)} ({100*v.get('no_new_name',0)/n:.0f}%)")
print("examples (claim name -> article name, ratio):", examples)
json.dump({"per_condition": out, "examples": examples, "threshold": 0.6}, open(os.path.join(os.path.dirname(__file__), "entity_translit.json"), "w"), indent=1)
