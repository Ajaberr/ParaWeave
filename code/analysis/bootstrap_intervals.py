"""Bootstrap confidence intervals and paired significance tests for the paper's main results (rebuttal item 4).

LOCAL ANALYSIS ONLY: reads the existing Claude Sonnet 4.6 claim-level judge outputs; makes no API call, runs no model,
and writes only result_cis.json / result_cis.md next to this file.   Usage:  python result_cis.py [--diag]

Step 1 reproduces the paper's point estimates (Table 2 summarizer table, Table 3 judge check, the Lugbara table of
App. E, and the averages / margins / retention figures quoted in Sections 5-6). Step 2 bootstraps them.

Conventions (identical to eval/official/openweight/ow_prep.py + ow_local.py, the scripts behind the paper numbers):
  * F = faith[ref]["faithfulness_rate"] = (supported + 0.5 partial) / claims. A summary with n_claims == 0 is left
    out of the mean (the "No claim" column of the Lugbara table). System score = mean over the remaining articles.
  * Arabic = roster_gen records whose id minus "ours-" is a cj-loop article of DiaBench (2,170 of the 2,330 cj-loop
    articles were judged), split by DiaBench's 'dialect' field.
  * Judge check: claims whose source- and English-reference verdict lists both align with the frozen claim list;
    faithful = SUPPORTED or PARTIAL; the English-reference verdict is the truth; pooled over all systems of a build.
    False passed = FP / (FP + TN), true rejected = FN / (TP + FN), on source-input claims.
  * Kendall tau (tau-b) between the systems' mean F_src,src and mean F_src,En on source-input summaries.

Bootstrap: articles are resampled with replacement within each variety (stratified by variety; 2,000 resamples; one
np.random.default_rng(0) stream, drawn in the fixed order of VARS). Every statistic of a variety is recomputed on the
same resamples, so differences, ratios and model averages are paired by article, and pooled statistics (Arabic over
four dialects) are stratified by dialect. The judge statistics use the same article resamples, i.e. a cluster
bootstrap in which each article carries every claim of every system. 95% CI = 2.5 / 97.5 percentiles; p_boot is the
two-sided bootstrap p (2 x the smaller tail share at 0). Paired permutation tests flip the sign of the per-article
differences (100,000 draws, np.random.default_rng(0))."""
import json, os, sys, time, math, collections
import numpy as np
from scipy.stats import kendalltau, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "eval", "official", "out")
DIABENCH = os.path.join(ROOT, "datasets", "official", "derived", "DiaBench", "diabench.jsonl")
B, SEED, NPERM = 2000, 0, 100_000
FAITH = {"SUPPORTED", "PARTIAL"}
VARS = ["fa", "prs", "tg", "egyptian", "gulf", "levantine", "maghrebi", "lgg"]
DIALECTS = ["egyptian", "gulf", "levantine", "maghrebi"]
VN = {"fa": "Farsi", "prs": "Dari", "tg": "Tajik", "egyptian": "Egyptian", "gulf": "Gulf", "levantine": "Levantine",
      "maghrebi": "Maghrebi", "lgg": "Lugbara"}
SN = {"openai": "GPT-5.5", "qwen30a3b": "Qwen3-30B-A3B", "gptoss20": "gpt-oss-20B", "qwen8": "Qwen3-8B",
      "aya": "Aya-23-8B", "dorna2": "Dorna2-8B", "fanar": "Fanar-1-9B", "sunflower9b": "Sunflower-9B",
      "sunflower32q4": "Sunflower-32B", "spec": "Specialist"}
GEN4 = ["qwen30a3b", "gptoss20", "qwen8", "aya"]
SPEC = {"fa": "dorna2", "prs": "dorna2", "tg": "dorna2", "lgg": "sunflower9b", **{d: "fanar" for d in DIALECTS}}
ROWS = ["openai", "qwen30a3b", "gptoss20", "qwen8", "aya", "spec"]
SRC, EN = "dialect", "gold_ref"        # input conditions: summary written from the source / from the English article


def systems(v):
    if v in ("fa", "prs", "tg"):
        return ["openai", "qwen30a3b", "gptoss20", "qwen8", "aya", "dorna2"]
    if v == "lgg":
        return ["openai", "qwen30a3b", "gptoss20", "qwen8", "aya", "fanar", "sunflower9b", "sunflower32q4"]
    return ["openai", "qwen30a3b", "gptoss20", "qwen8", "aya", "fanar"]


# ------------------------------------------------------------------ paper values (for the reproduction check)
PAPER_SUMM = {  # (src, En) per variety, in VARS order; Table 2 (tab:summ)
    "openai":    [(.89, .92), (.89, .93), (.89, .92), (.87, .93), (.85, .92), (.86, .93), (.83, .92), (.47, .90)],
    "qwen30a3b": [(.79, .89), (.76, .88), (.66, .88), (.72, .89), (.67, .87), (.66, .88), (.57, .88), (.21, .88)],
    "gptoss20":  [(.74, .88), (.71, .89), (.58, .89), (.73, .86), (.68, .84), (.67, .85), (.65, .86), (.40, .84)],
    "qwen8":     [(.75, .88), (.71, .88), (.61, .89), (.66, .86), (.61, .85), (.61, .86), (.54, .86), (.29, .86)],
    "aya":       [(.81, .86), (.76, .85), (.16, .84), (.69, .79), (.62, .79), (.62, .78), (.55, .79), (.09, .83)],
    "spec":      [(.76, .84), (.70, .86), (.61, .87), (.72, .84), (.67, .84), (.66, .85), (.60, .85), (.50, .84)],
}
LGG_COLS = [("F_src_src", SRC, "src"), ("F_src_En", SRC, "en"), ("F_En_src", EN, "src"), ("F_En_En", EN, "en")]
PAPER_LGG = {  # App. E tab:lugbara: F_src,src  F_src,En  F_En,src  F_En,En  no-claim(Lugbara input)
    "sunflower32q4": (0.516, 0.608, 0.544, 0.858, 1), "sunflower9b": (0.482, 0.502, 0.526, 0.836, 0),
    "openai": (0.735, 0.471, 0.589, 0.897, 0), "gptoss20": (0.730, 0.396, 0.553, 0.839, 120),
    "qwen8": (0.620, 0.294, 0.577, 0.860, 57), "qwen30a3b": (0.554, 0.209, 0.540, 0.885, 30),
    "fanar": (0.541, 0.097, 0.554, 0.869, 2), "aya": (0.341, 0.087, 0.552, 0.825, 7)}
PAPER_JUDGE = {  # Table 3 (tab:judge-check), VARS order
    "kappa_src": [0.51, 0.57, 0.79, 0.56, 0.55, 0.58, 0.51, 0.21],
    "kappa_en": [0.47, 0.48, 0.45, 0.47, 0.43, 0.41, 0.32, 0.06],
    "false_passed_pct": [53, 49, 21, 46, 48, 43, 49, 58],
    "true_rejected_pct": [3, 2, 3, 4, 4, 5, 5, 19],
    "tau": [0.73, 1.00, 1.00, 0.73, 0.73, 0.73, 0.87, 0.29]}
PAPER_AVG = {"arabic": (0.84, 0.63), "farsi_dari": (0.88, 0.75), "tajik": (0.87, 0.50), "lugbara": (0.85, 0.25)}
PAPER_RET = {"sunflower9b": 60, "sunflower32q4": 71, "openai": 52}
PAPER_LGG_DIFF = {"sunflower9b": 0.03, "sunflower32q4": 0.14}


# ------------------------------------------------------------------ loading
def labels_rate(verdicts, n):
    if not n:
        return None
    labs = [(v or {}).get("label") for v in verdicts]
    return (sum(l == "SUPPORTED" for l in labs) + 0.5 * sum(l == "PARTIAL" for l in labs)) / n


def parse(r):
    fa = r.get("faith") or {}
    fd, fg = fa.get("dialect") or {}, fa.get("gold_reference") or {}
    a, b = fd.get("verdicts") or [], fg.get("verdicts") or []
    claims = r.get("claims") or []
    cm = None
    if len(a) == len(b) == len(claims) and len(claims) > 0:          # ow_prep.py alignment rule
        t = [int((y or {}).get("label") in FAITH) for y in b]           # English-reference verdict = truth
        p = [int((x or {}).get("label") in FAITH) for x in a]           # source-reference verdict = prediction
        tp = sum(ti and pi for ti, pi in zip(t, p)); fn = sum(ti and not pi for ti, pi in zip(t, p))
        fp = sum((not ti) and pi for ti, pi in zip(t, p)); tn = len(t) - tp - fn - fp
        cm = (tp, fn, fp, tn)
    ss, es = r.get("source_sents") or [], r.get("gold_reference") or []
    f_src = fd.get("faithfulness_rate") if fd.get("n_claims") else None
    f_en = fg.get("faithfulness_rate") if fg.get("n_claims") else None
    chk = [abs(x - y) for x, y in ((f_src, labels_rate(a, fd.get("n_claims"))), (f_en, labels_rate(b, fg.get("n_claims"))))
           if x is not None and y is not None]
    return {"F_src": f_src, "F_en": f_en, "n_src": fd.get("n_claims") or 0, "n_en": fg.get("n_claims") or 0,
            "n_claims": len(claims), "cm": cm, "misaligned": (len(ss) != len(es)) or not ss,
            "rate_dev": max(chk) if chk else 0.0}


def load_all():
    src_of, dial_of = {}, {}
    for l in open(DIABENCH, encoding="utf-8"):
        r = json.loads(l)
        src_of[r["id"]], dial_of[r["id"]] = r["source"], r["dialect"]
    D, files, diag = {}, [], collections.Counter()

    def ingest(path, system, variety_of):
        files.append(os.path.relpath(path, ROOT).replace("\\", "/"))
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            v, aid = variety_of(r["id"])
            if v is None:
                diag["arabic records skipped (not cj-loop)"] += 1
                continue
            d = D.setdefault((v, system, r["condition"]), {})
            if aid in d:
                diag[f"duplicate {v}/{system}/{r['condition']}"] += 1
            d[aid] = parse(r)

    for build, v in (("fa", "fa"), ("prs", "prs"), ("tg", "tg"), ("mlext", "lgg")):
        for s in systems(v):
            ingest(os.path.join(OUT, f"gen.{build}.{s}.faith.claude.jsonl"), s, lambda i, v=v: (v, i))

    def ar(i):
        aid = i.replace("ours-", "", 1)
        return (dial_of[aid], aid) if src_of.get(aid) == "cj-loop" else (None, None)

    for s in systems("egyptian"):
        ingest(os.path.join(OUT, "roster_gen", f"gen.ours.{s}.rest.faith.claude.jsonl"), s, ar)
    cj_total = collections.Counter(dial_of[i] for i in src_of if src_of[i] == "cj-loop")
    return D, files, diag, cj_total


def data_checks(D, diag, cj_total, verbose=False):
    """Integrity checks that could break exact reproduction; returned for the JSON, printed with --diag."""
    out = {"cj_loop_articles_in_diabench": dict(cj_total), "loader_notes": dict(diag), "per_variety": {}}
    for v in VARS:
        ids = set()
        for s in systems(v):
            for c in (SRC, EN):
                ids |= set(D.get((v, s, c), {}))
        rows = {}
        for s in systems(v):
            for c in (SRC, EN):
                d = D.get((v, s, c), {}); recs = list(d.values())
                rows[f"{s}|{c}"] = {
                    "records": len(d), "missing_articles": len(ids - set(d)),
                    "no_claim": sum(r["F_en"] is None for r in recs),
                    "n_claims_differs_between_refs": sum(r["n_src"] != r["n_en"] for r in recs),
                    "verdicts_unaligned": sum(r["cm"] is None for r in recs),
                    "misaligned_article": sum(r["misaligned"] for r in recs),
                    "max_abs_dev_stored_vs_recomputed_F": max([r["rate_dev"] for r in recs] or [0.0])}
        out["per_variety"][v] = {"articles": len(ids), "records": rows}
        if verbose:
            print(f"\n{VN[v]}: {len(ids)} articles")
            for k, x in rows.items():
                print(f"   {k:24s} " + " ".join(f"{a}={b}" for a, b in x.items()))
    return out


# ------------------------------------------------------------------ arrays and bootstrap machinery
def build_arrays(D):
    A = {}
    for v in VARS:
        ids = sorted(set().union(*(set(D.get((v, s, c), {})) for s in systems(v) for c in (SRC, EN))))
        pos = {a: k for k, a in enumerate(ids)}
        N = len(ids)
        F, CMS = {}, {}
        CM = {c: np.zeros((N, 4)) for c in (SRC, EN)}
        for s in systems(v):
            for c in (SRC, EN):
                fs, fe, cms = np.full(N, np.nan), np.full(N, np.nan), np.zeros((N, 4))
                for aid, r in D.get((v, s, c), {}).items():
                    k = pos[aid]
                    if r["F_src"] is not None:
                        fs[k] = r["F_src"]
                    if r["F_en"] is not None:
                        fe[k] = r["F_en"]
                    if r["cm"] is not None:
                        cms[k] = r["cm"]
                F[(s, c, "src")], F[(s, c, "en")], CMS[(c, s)] = fs, fe, cms
                CM[c] += cms
        A[v] = {"ids": ids, "N": N, "F": F, "CM": CM, "CMS": CMS}
    return A


def sc(x, idx):
    """Sum and count of the non-NaN entries of x: point values and (B,) bootstrap arrays over the resamples idx."""
    ok = ~np.isnan(x)
    x0 = np.where(ok, x, 0.0)
    return float(x0.sum()), int(ok.sum()), x0[idx].sum(1), ok[idx].sum(1)


def summ(pt, bs, n=None, diff=False):
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    d = {"est": float(pt), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
         "se": float(bs.std(ddof=1))}
    if n is not None:
        d["n"] = int(n)
    if diff:
        d["p_boot"] = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
        d["sig"] = bool(d["lo"] > 0 or d["hi"] < 0)
    return d


def perm_p(d, nperm=NPERM, seed=SEED):
    """Paired sign-flip permutation test of mean(d) = 0: two-sided and one-sided (greater) p-values."""
    d = np.asarray(d, float)
    n, obs = len(d), float(d.mean())
    rng = np.random.default_rng(seed)
    hit2 = hit1 = done = 0
    while done < nperm:
        k = min(20000, nperm - done)
        sg = rng.integers(0, 2, size=(k, n)).astype(np.float64) * 2 - 1
        mu = sg @ d / n
        hit2 += int((np.abs(mu) >= abs(obs) - 1e-12).sum())
        hit1 += int((mu >= obs - 1e-12).sum())
        done += k
    return (hit2 + 1) / (nperm + 1), (hit1 + 1) / (nperm + 1)


def holm(named_p):
    """Holm step-down adjustment of [(name, p), ...] -> {name: adjusted p}."""
    out, run = {}, 0.0
    ranked = sorted(named_p, key=lambda t: t[1])
    for r, (name, p) in enumerate(ranked):
        run = max(run, min(1.0, p * (len(ranked) - r)))
        out[name] = run
    return out


def tau_b_batch(X, Y):
    """Kendall tau-b for each row pair of X, Y (shape (b, k)); equals scipy.stats.kendalltau's default."""
    i, j = np.triu_indices(X.shape[1], 1)
    dx, dy = np.sign(X[:, i] - X[:, j]), np.sign(Y[:, i] - Y[:, j])
    n0 = len(i)
    den = np.sqrt((n0 - (dx == 0).sum(1)) * (n0 - (dy == 0).sum(1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (dx * dy).sum(1) / den


def judge_stats(T):
    """T[..., 4] = (tp, fn, fp, tn) with truth = English-reference verdict: kappa, false passed, true rejected, n."""
    tp, fn, fp, tn = (np.asarray(T[..., k], float) for k in range(4))
    n = tp + fn + fp + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return (po - pe) / (1 - pe), fp / (fp + tn), fn / (tp + fn), n


def rounds_to(x, paper, nd):
    """Compare at the paper's precision; also flag values within 1e-6 of a rounding boundary."""
    got = f"{x:.{nd}f}"
    scaled = x * 10 ** nd
    return got == f"{paper:.{nd}f}", got, abs(scaled - math.floor(scaled) - 0.5) < 1e-6


# ------------------------------------------------------------------ main analysis
def main():
    t0 = time.time()
    D, files, diag, cj_total = load_all()
    checks = data_checks(D, diag, cj_total, verbose="--diag" in sys.argv)
    if "--diag" in sys.argv:
        return
    A = build_arrays(D)
    rng = np.random.default_rng(SEED)
    IDX = {v: rng.integers(0, A[v]["N"], size=(B, A[v]["N"])) for v in VARS}   # fixed draw order = VARS
    MC = {}

    def m(v, s, c, ref="en"):
        """(point mean, (B,) bootstrap means, n articles) of F for system s, input c, reference ref."""
        k = (v, s, c, ref)
        if k not in MC:
            S, C, Sb, Cb = sc(A[v]["F"][(s, c, ref)], IDX[v])
            MC[k] = (S / C, Sb / Cb, C)
        return MC[k]

    R = {"meta": {"bootstrap_resamples": B, "seed": SEED, "rng": "numpy.random.default_rng(0), one stream, drawn per variety in order " + ",".join(VARS),
                  "ci": "95% percentile", "permutations": NPERM, "unit": "article (stratified by variety; Arabic pooled stats stratified by dialect)",
                  "articles": {VN[v]: A[v]["N"] for v in VARS}, "files": files,
                  "conventions": __doc__.split("Conventions")[1].split("Bootstrap:")[0].strip()},
         "data_checks": checks}

    # ================================================================ 1. reproduction of the paper's point estimates
    rep = collections.OrderedDict()
    L = rep["table2_summarizer"] = []
    for j, v in enumerate(VARS):
        for row in ROWS:
            s = SPEC[v] if row == "spec" else row
            for col, c, paper in (("src", SRC, PAPER_SUMM[row][j][0]), ("En", EN, PAPER_SUMM[row][j][1])):
                x = m(v, s, c)[0]
                ok, got, near = rounds_to(x, paper, 2)
                L.append({"variety": VN[v], "system": SN[s], "col": col, "computed": x, "rounded": got, "paper": paper,
                          "match": ok, "near_boundary": near})
    L = rep["lugbara_table"] = []
    for s, pv in PAPER_LGG.items():
        for (name, c, ref), paper in zip(LGG_COLS, pv[:4]):
            x = m("lgg", s, c, ref)[0]
            ok, got, near = rounds_to(x, paper, 3)
            L.append({"system": SN[s], "col": name, "computed": x, "rounded": got, "paper": paper, "match": ok, "near_boundary": near})
        nc = int(np.isnan(A["lgg"]["F"][(s, SRC, "en")]).sum())
        L.append({"system": SN[s], "col": "no_claim", "computed": nc, "rounded": str(nc), "paper": pv[4], "match": nc == pv[4], "near_boundary": False})
    L = rep["table3_judge"] = []
    tau_pts = {}
    for j, v in enumerate(VARS):
        stt = {}
        for c in (SRC, EN):
            stt[c] = judge_stats(A[v]["CM"][c].sum(0))
        S_ = systems(v)
        tau_pts[v] = kendalltau([m(v, s, SRC, "src")[0] for s in S_], [m(v, s, SRC, "en")[0] for s in S_]).statistic
        for key, x, nd, scale in (("kappa_src", stt[SRC][0], 2, 1), ("kappa_en", stt[EN][0], 2, 1),
                                  ("false_passed_pct", stt[SRC][1], 0, 100), ("true_rejected_pct", stt[SRC][2], 0, 100),
                                  ("tau", tau_pts[v], 2, 1)):
            ok, got, near = rounds_to(float(x) * scale, PAPER_JUDGE[key][j], nd)
            L.append({"variety": VN[v], "stat": key, "computed": float(x) * scale, "rounded": got, "paper": PAPER_JUDGE[key][j],
                      "match": ok, "near_boundary": near})

    # model averages (item 2), both conventions
    def avg(vs, models, c, pooled):
        pts, bss = [], []
        for s in models:
            if pooled:
                S = C = 0.0
                Sb = Cb = 0.0
                for v in vs:
                    a_, b_, ab, bb = sc(A[v]["F"][(s, c, "en")], IDX[v])
                    S, C, Sb, Cb = S + a_, C + b_, Sb + ab, Cb + bb
                pts.append(S / C); bss.append(Sb / Cb)
            else:
                ms = [m(v, s, c) for v in vs]
                pts.append(float(np.mean([x[0] for x in ms]))); bss.append(np.mean([x[1] for x in ms], axis=0))
        return float(np.mean(pts)), np.mean(bss, axis=0)

    AVG_GROUPS = [("arabic", DIALECTS, True, "pooled over the 2,170 Arabic articles, then averaged over the 4 models"),
                  ("arabic_dialect_mean", DIALECTS, False, "mean of the 4 dialect cells, then over the 4 models"),
                  ("farsi_dari", ["fa", "prs"], False, "mean of the Farsi and Dari cells, then over the 4 models"),
                  ("farsi_dari_pooled", ["fa", "prs"], True, "pooled over the 462 Farsi+Dari articles, then over the 4 models"),
                  ("tajik", ["tg"], True, "Tajik cells averaged over the 4 models"),
                  ("lugbara", ["lgg"], True, "Lugbara cells averaged over the 4 models")]
    AVG = {}
    L = rep["text_averages"] = []
    for name, vs, pooled, how in AVG_GROUPS:
        ee_pt, ee_b = avg(vs, GEN4, EN, pooled)
        se_pt, se_b = avg(vs, GEN4, SRC, pooled)
        AVG[name] = {"how": how, "F_En_En": summ(ee_pt, ee_b), "F_src_En": summ(se_pt, se_b),
                     "drop": summ(ee_pt - se_pt, ee_b - se_b, diff=True),
                     "relative_drop": summ(1 - se_pt / ee_pt, 1 - se_b / ee_b)}
        paper = PAPER_AVG.get(name)
        if paper:
            for lab, x, p in (("F_En_En", ee_pt, paper[0]), ("F_src_En", se_pt, paper[1])):
                ok, got, near = rounds_to(x, p, 2)
                L.append({"group": name, "stat": lab, "computed": x, "rounded": got, "paper": p, "match": ok, "near_boundary": near})
        else:
            for lab, x in (("F_En_En", ee_pt), ("F_src_En", se_pt)):
                L.append({"group": name, "stat": lab, "computed": x, "rounded": f"{x:.2f}", "paper": None, "match": None,
                          "note": "alternative convention, for comparison"})
    L = rep["text_lugbara_margins"] = []
    for s, p in PAPER_LGG_DIFF.items():
        x = m("lgg", s, SRC)[0] - m("lgg", "openai", SRC)[0]
        ok, got, near = rounds_to(x, p, 2)
        L.append({"system": SN[s], "stat": "F_src_En minus GPT-5.5", "computed": x, "rounded": got, "paper": p, "match": ok, "near_boundary": near})
    L = rep["text_retention"] = []
    for s, p in PAPER_RET.items():
        x = 100 * m("lgg", s, SRC)[0] / m("lgg", s, EN)[0]
        ok, got, near = rounds_to(x, p, 0)
        L.append({"system": SN[s], "stat": "100 x F_src_En / F_En_En", "computed": x, "rounded": got, "paper": p, "match": ok, "near_boundary": near})
    R["reproduction"] = rep
    R["reproduction_summary"] = {k: {"checked": sum(1 for x in v if x.get("match") is not None),
                                     "matched": sum(1 for x in v if x.get("match")),
                                     "mismatches": [x for x in v if x.get("match") is False],
                                     "near_rounding_boundary": [x for x in v if x.get("near_boundary")]} for k, v in rep.items()}

    # ================================================================ 2. CIs: Table 2 cells and comprehension gaps
    T2 = collections.OrderedDict()
    for v in VARS:
        for row in ROWS:
            s = SPEC[v] if row == "spec" else row
            se_, ee_ = m(v, s, SRC), m(v, s, EN)
            dp = A[v]["F"][(s, EN, "en")] - A[v]["F"][(s, SRC, "en")]         # NaN unless both summaries have claims
            S, C, Sb, Cb = sc(dp, IDX[v])
            T2[f"{v}|{row}"] = {"variety": VN[v], "system": SN[s], "row": row,
                                "F_src_En": summ(se_[0], se_[1], se_[2]), "F_En_En": summ(ee_[0], ee_[1], ee_[2]),
                                "gap": summ(ee_[0] - se_[0], ee_[1] - se_[1], diff=True),
                                "gap_strictly_paired": summ(S / C, Sb / Cb, C, diff=True)}
    R["table2_cis"] = T2

    # ================================================================ 3. averages quoted in the text
    R["text_averages_cis"] = AVG

    # ================================================================ 4. Lugbara: specialists minus GPT-5.5 (source input, English ref)
    v = "lgg"
    gpt = A[v]["F"][("openai", SRC, "en")]
    LG, praw = collections.OrderedDict(), []
    for s in ["sunflower9b", "sunflower32q4"]:
        a_, g_ = m(v, s, SRC), m(v, "openai", SRC)
        dfull = A[v]["F"][(s, SRC, "en")] - gpt
        S, C, Sb, Cb = sc(dfull, IDX[v])
        d = dfull[~np.isnan(dfull)]
        p2, p1 = perm_p(d)
        w = wilcoxon(d)
        sd = float(d.std(ddof=1))
        LG[s] = {"system": SN[s], "F_src_En": a_[0], "GPT55_F_src_En": g_[0],
                 "diff_of_means": summ(a_[0] - g_[0], a_[1] - g_[1], diff=True),
                 "paired_mean_diff": summ(S / C, Sb / Cb, C, diff=True),
                 "perm_p_two_sided": p2, "perm_p_one_sided_greater": p1, "wilcoxon_p_two_sided": float(w.pvalue),
                 "articles_better_tied_worse": [int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())],
                 "sd_of_article_differences": sd, "min_detectable_diff_80pct_power": 2.80 * sd / math.sqrt(len(d))}
        praw.append((s, p2))
    for s, p in holm(praw).items():                                         # Holm over the two two-sided tests
        LG[s]["perm_p_two_sided_holm"] = p
    a9, a32 = m(v, "sunflower9b", SRC), m(v, "sunflower32q4", SRC)
    LG["sunflower32q4_minus_sunflower9b"] = summ(a32[0] - a9[0], a32[1] - a9[1], diff=True)
    R["lugbara_specialists_vs_gpt55"] = LG

    # ================================================================ 5. judge agreement (Table 3), cluster bootstrap over articles
    JG = collections.OrderedDict()
    KB = {}
    for v in VARS:
        out = {}
        for c, tag in ((SRC, "source_input_claims"), (EN, "english_input_claims")):
            CMv = A[v]["CM"][c]
            Tpt = CMv.sum(0)
            Tb = np.stack([CMv[:, k][IDX[v]].sum(1) for k in range(4)], axis=1)
            kp, fpp, trp, npt = judge_stats(Tpt)
            kb, fpb, trb, _ = judge_stats(Tb)
            KB[(v, c)] = kb
            out[tag] = {"kappa": summ(kp, kb, npt), "false_passed": summ(fpp, fpb, Tpt[2] + Tpt[3]),
                        "true_rejected": summ(trp, trb, Tpt[0] + Tpt[1]),
                        "counts_tp_fn_fp_tn": [int(x) for x in Tpt]}
        S_ = systems(v)
        Xb = np.stack([m(v, s, SRC, "src")[1] for s in S_], 1)
        Yb = np.stack([m(v, s, SRC, "en")[1] for s in S_], 1)
        Xp = np.array([[m(v, s, SRC, "src")[0] for s in S_]])
        Yp = np.array([[m(v, s, SRC, "en")[0] for s in S_]])
        tb = tau_b_batch(Xb, Yb)
        assert abs(tau_b_batch(Xp, Yp)[0] - tau_pts[v]) < 1e-12
        out["kendall_tau"] = summ(tau_pts[v], tb, len(S_))
        out["kendall_tau"]["share_of_resamples_tau_eq_1"] = float((tb > 1 - 1e-9).mean())
        JG[v] = out
    # Lugbara against every other build (independent resamples of different articles)
    JG["lugbara_minus_other_builds"] = {
        f"{VN[o]}|{tag}": summ(judge_stats(A['lgg']['CM'][c].sum(0))[0] - judge_stats(A[o]['CM'][c].sum(0))[0],
                               KB[("lgg", c)] - KB[(o, c)], diff=True)
        for o in VARS[:-1] for c, tag in ((SRC, "kappa_source_input"), (EN, "kappa_english_input"))}
    # rank flip on Lugbara: specialists top-2 by the English reference and both in the bottom half by the Lugbara reference
    S_ = systems("lgg")
    Xb = np.stack([m("lgg", s, SRC, "src")[1] for s in S_], 1)
    Yb = np.stack([m("lgg", s, SRC, "en")[1] for s in S_], 1)
    rk_en = (-Yb).argsort(1).argsort(1)
    rk_lg = (-Xb).argsort(1).argsort(1)
    i9, i32 = S_.index("sunflower9b"), S_.index("sunflower32q4")
    top2_en = (rk_en[:, i9] <= 1) & (rk_en[:, i32] <= 1)
    bot_lg = (rk_lg[:, i9] >= len(S_) // 2) & (rk_lg[:, i32] >= len(S_) // 2)
    JG["lugbara_rank_flip"] = {"share_specialists_top2_by_english_ref": float(top2_en.mean()),
                               "share_specialists_bottom_half_by_lugbara_ref": float(bot_lg.mean()),
                               "share_both": float((top2_en & bot_lg).mean())}
    # per-system judge error rates on Lugbara-input claims (Section 6 text)
    JG["lugbara_per_system_source_input"] = {}
    for s in S_:
        CMs = A["lgg"]["CMS"][(SRC, s)]
        Tpt = CMs.sum(0)
        Tb = np.stack([CMs[:, k][IDX["lgg"]].sum(1) for k in range(4)], axis=1)
        _, fpp, trp, _ = judge_stats(Tpt)
        _, fpb, trb, _ = judge_stats(Tb)
        JG["lugbara_per_system_source_input"][SN[s]] = {"false_passed": summ(fpp, fpb, Tpt[2] + Tpt[3]),
                                                        "true_rejected": summ(trp, trb, Tpt[0] + Tpt[1])}
    R["judge_agreement_cis"] = JG

    # ================================================================ 6. retention on Lugbara: F_src,En / F_En,En
    RET, RB = collections.OrderedDict(), {}
    for s in ["sunflower9b", "sunflower32q4", "openai"]:
        se_, ee_ = m("lgg", s, SRC), m("lgg", s, EN)
        RB[s] = (se_[0] / ee_[0], se_[1] / ee_[1])
        RET[s] = {"system": SN[s], "retention": summ(*RB[s])}
    for s in ["sunflower9b", "sunflower32q4"]:
        RET[f"{s}_minus_openai"] = summ(RB[s][0] - RB["openai"][0], RB[s][1] - RB["openai"][1], diff=True)
    R["lugbara_retention_cis"] = RET

    # ================================================================ 7. other Section 5 claims on the same resamples
    S1 = []   # "the frontier model is the most faithful system whichever side of the article it is given"
    for v in VARS[:-1]:
        for col, c in (("src", SRC), ("En", EN)):
            others = [s for s in systems(v) if s != "openai"]
            g_ = m(v, "openai", c)
            ru = max(others, key=lambda s: m(v, s, c)[0])
            allb = np.stack([g_[1] - m(v, s, c)[1] for s in others], 1)
            S1.append({"variety": VN[v], "column": col, "runner_up": SN[ru],
                       "gpt55_minus_runner_up": summ(g_[0] - m(v, ru, c)[0], g_[1] - m(v, ru, c)[1], diff=True),
                       "share_of_resamples_gpt55_best": float((allb.min(1) > 0).mean())})
    S2 = []   # "the language specialists never beat the general open-weight models of their own size"
    for v in VARS[:-1]:
        sp = SPEC[v]
        for o in ["qwen8", "aya", "gptoss20", "qwen30a3b"]:
            for col, c in (("src", SRC), ("En", EN)):
                a_, b_ = m(v, sp, c), m(v, o, c)
                S2.append({"variety": VN[v], "specialist": SN[sp], "general_model": SN[o], "same_size": o in ("qwen8", "aya"),
                           "column": col, "specialist_minus_general": summ(a_[0] - b_[0], a_[1] - b_[1], diff=True)})
    R["other_claims"] = {"gpt55_best_every_column": S1, "specialist_vs_general": S2}
    R["meta"]["runtime_s"] = round(time.time() - t0, 1)

    with open(os.path.join(HERE, "result_cis.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if "--no-md" not in sys.argv:
        write_md(R)
    print(f"done in {time.time() - t0:.1f}s -> result_cis.json" + ("" if "--no-md" in sys.argv else ", result_cis.md"))
    return R


def write_md(R):
    pass


if __name__ == "__main__":
    main()
