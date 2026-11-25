#!/usr/bin/env python
"""Build a ParaWeave pool from SALT (Sunbird African Language Technology, HF `Sunbird/salt`,
CC BY-SA 4.0): 25k English sentences on agriculture / health / society, each translated into
Luganda (lug), Acholi (ach), Lugbara (lgg), Ateso (teo), Runyankole (nyn), Lusoga (xog), Rutooro (ttj).

    python processing/official/build_salt_pool.py <lang_code> [<tag>]
    e.g. python processing/official/build_salt_pool.py lgg lugbara

Writes datasets/official/derived/DiaBench-ML/poolfull_<tag>.jsonl with the same schema as every
other pool: {ar: <source-language sentence>, en: <English>, sub: <tag>, _group_id: <salt id>}.
The field is called `ar` for every variety; only the prompts' CJ_SRC_LANG changes.
Needs HF_TOKEN (the dataset is gated, auto-approved) in the environment or the repo-root .env.
Without a token, Lugbara (lgg) falls back to `mekaneeky/salt-llama-lgg-to-eng`, an ungated
repackaging of the same SALT text-all lgg/eng pairs (SALT ids preserved; 23,947/500/500 rows) in
`<s>[INST] Translate lgg to eng: <lgg> [/INST] <eng> </s>` form, which we parse back into pairs.
"""
import json, os, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets" / "official" / "derived" / "DiaBench-ML"

def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); v = re.sub(r"\s+#.*$", "", v).strip().strip('"').strip("'")
                if v: os.environ.setdefault(k.strip(), v)

def main():
    lang = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else lang
    load_env()
    tok = os.environ.get("HF_TOKEN")
    import pandas as pd
    from huggingface_hub import hf_hub_download
    if not tok:
        assert lang == "lgg", f"HF_TOKEN missing: Sunbird/salt is gated (auto-approve); only lgg has an ungated mirror, not {lang}"
        print("HF_TOKEN missing -> using the ungated SALT mirror mekaneeky/salt-llama-lgg-to-eng (same SALT lgg/eng pairs)")
        rx = re.compile(r"\[INST\]\s*Translate lgg to eng:\s*(.*?)\s*\[/INST\]\s*(.*?)\s*</s>\s*$", re.S)
        frames = []
        for split, f in (("train", "data/train-00000-of-00001-996f4551a67e9ca1.parquet"),
                         ("dev", "data/dev-00000-of-00001-585ed6fa7b1d126e.parquet"),
                         ("test", "data/test-00000-of-00001-dcd956181b43ed0e.parquet")):
            df = pd.read_parquet(hf_hub_download("mekaneeky/salt-llama-lgg-to-eng", f, repo_type="dataset"))
            m = df["text"].str.extract(rx)
            bad = int(m[0].isna().sum())
            if bad: print(f"  {split}: {bad} rows did not match the prompt template (dropped)")
            frames.append(pd.DataFrame({"id": df["ID"].astype(str), "lgg_text": m[0], "eng_text": m[1], "_split": split}).dropna())
        df = pd.concat(frames, ignore_index=True)
        cols = list(df.columns); print("columns:", cols)
        return write_pool(df, "lgg_text", "eng_text", "id", lang, tag)
    frames = []
    for split in ("train", "dev", "test"):
        f = hf_hub_download("Sunbird/salt", f"text-all/{split}-00000-of-00001.parquet", repo_type="dataset", token=tok)
        df = pd.read_parquet(f); df["_split"] = split; frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    cols = list(df.columns); print("columns:", cols)
    # SALT text configs carry one column per language (e.g. 'eng_text', 'lgg_text') or a nested 'text' dict.
    def col_for(code):
        for c in cols:
            if c.lower() in (code, f"{code}_text", f"text_{code}"): return c
        return None
    en_c, src_c = col_for("eng"), col_for(lang)
    if en_c is None or src_c is None:
        if "text" in cols and isinstance(df["text"].iloc[0], dict):
            df = pd.concat([df.drop(columns=["text"]), df["text"].apply(pd.Series)], axis=1); cols = list(df.columns)
            en_c, src_c = col_for("eng"), col_for(lang)
    assert en_c and src_c, f"could not find English/{lang} columns in {cols}"
    id_c = next((c for c in cols if c.lower() in ("id", "sentence_id", "salt_id")), None)
    write_pool(df, src_c, en_c, id_c, lang, tag)

def write_pool(df, src_c, en_c, id_c, lang, tag):
    n_in = len(df); rows = []; seen = set()
    for i, r in df.iterrows():
        src, en = str(r[src_c] or "").strip(), str(r[en_c] or "").strip()
        if len(src.split()) < 3 or len(en.split()) < 3: continue        # drop fragments
        key = (src, en)
        if key in seen: continue
        seen.add(key)
        rows.append({"ar": src, "en": en, "sub": tag, "_group_id": str(r[id_c]) if id_c else f"{r['_split']}-{i}"})
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"poolfull_{tag}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{lang}: {n_in} rows in SALT text-all -> {len(rows)} usable pairs -> {out}")
    for r in rows[:3]: print("  ", r["ar"][:80], "|||", r["en"][:80])

if __name__ == "__main__":
    main()
