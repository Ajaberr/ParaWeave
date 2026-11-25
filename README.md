# ParaWeave

Code, benchmark and results for **Finding the Parallels: A Path to Evaluating Summarization from Low-Resource Languages**.

Ahmed Jaber, Zachary Horvitz, Nicholas Deas, Parmida Shariat, Kathleen McKeown (Columbia University)

ParaWeave turns sentence-level parallel corpora into coherent document-level benchmarks. This repository contains the
benchmark, every model summary with its judged claims, the numbers behind the paper's tables and figures, and the code that
built and scored them. The code is MIT-licensed; every part of the data keeps its source corpus's license
([DATA_LICENSE.md](DATA_LICENSE.md)).

## Citation

```bibtex
@misc{jaber2026paraweave,
  title  = {Finding the Parallels: A Path to Evaluating Summarization from Low-Resource Languages},
  author = {Jaber, Ahmed and Horvitz, Zachary and Deas, Nicholas and Shariat, Parmida and McKeown, Kathleen},
  year   = {2026},
  note   = {Preprint}
}
```

## Contents

| Path | What it is |
|---|---|
| `data/` | The benchmark: 3,671 parallel articles in ten languages (3,661 released; see below) |
| `results/summaries/` | Every summary, its frozen atomic claims, and the judge's verdicts against both articles |
| `results/analysis/` | JSON files behind the tables and figures |
| `results/anchors/` | Per-claim scores of the open-weight checkers (AlignScore, MiniCheck, mDeBERTa, e5) |
| `results/translation_probe/` | Each open-weight system's sentence translations (chrF analysis) |
| `results/human_study/` | Survey items and aggregate results of the native-speaker study |
| `results/construction_logs/` | Round-by-round logs of the FLORES and Lugbara builds |
| `code/` | Construction pipeline, evaluation and analysis scripts |

## The benchmark (`data/`)

| File | Articles | Source corpus |
|---|---:|---|
| `arabic.jsonl` | 2,320 released of 2,330 (Egyptian 294, Gulf 914, Levantine 432, Maghrebi 680) | MADAR, via the AraBench release |
| `persian.jsonl`, `dari.jsonl`, `tajik.jsonl` | 232, 230, 159 | FLORES-200 |
| `lugbara.jsonl` | 240 | SALT |
| `acholi.jsonl`, `ateso.jsonl` | 240 each (the Lugbara articles in Acholi and Ateso) | SALT, multi-way parallel |

Ten of the 2,330 Arabic articles contain sentences from a corpus whose licence forbids redistribution (MDC); they are
withheld from this release and listed in `data/WITHHELD.md`, so `arabic.jsonl` has 2,320 articles.

Each line is one article: `source_sentences` (the low-resource language) and `english_sentences`, aligned
sentence by sentence. Every sentence is a human-written sentence of the source corpus; the pipeline only
selects and orders them (plus proper-noun unification in a few Arabic sentences). Arabic articles also carry
`dialect` and `country`; Acholi and Ateso articles carry the `lugbara_id` they were mapped from.

## Summaries and verdicts (`results/summaries/`)

One file per build, one line per summary:

- `article_id`, `system`, `input` (`source` or `english`), `summary`, `claims`
- `verdicts_vs_source`, `verdicts_vs_english`: one verdict per claim, with `label` (SUPPORTED, PARTIAL,
  NOT_SUPPORTED, CONTRADICTED), `error` tag and `evidence` quote.

Faithfulness is (supported + ½ partially supported) / claims. The summarizer evaluation uses
`verdicts_vs_english` for both inputs; the judge evaluation compares `verdicts_vs_source` with
`verdicts_vs_english` on the English-input summaries. Arabic covers the 2,162 released judged articles;
`lugbara_gpt-5.6-sol_reasoning_effort.jsonl` holds the reasoning-effort sweep.

## Where the paper's numbers come from

| Paper result | Files |
|---|---|
| Summarization faithfulness (scatter figure, full-results table) | `results/summaries/*.jsonl` |
| Judge evaluation (loss and agreement figures, judge-check table) | `results/summaries/*.jsonl`, `code/analysis/judge_check_table.py`, `src_en_agreement.py` |
| Embedding AUC on FLORES and CrossSum | `results/analysis/auc_flores.json`, `auc_crosssum.json`; `code/analysis/flores_auc.py`, `crosssum_diagnostic.py` |
| FLORES recovery control | `code/analysis/flores_recovery_check.py`, `results/construction_logs/` |
| Automatic anchors, translation quality | `results/anchors/`, `results/translation_probe/`, `results/analysis/anchors_translation_*.json` |
| Nilo-Saharan languages, ranking correlation | `results/analysis/nilo_saharan_*.json`, `ranking_correlation.json` |
| Reasoning effort | `results/analysis/reasoning_effort.json` |
| Confidence intervals | `results/analysis/bootstrap_intervals.json` |
| Error types and transliteration | `results/analysis/entity_errors_transliteration.json`, `code/analysis/entity_translit.py` |
| Native-speaker study | `results/human_study/` |

## Code (`code/`)

| Folder | What it does |
|---|---|
| `construction/` | Sentence pools (FLORES, SALT), embedding and clustering, and the compose, filter and recycle loop (`loop_assemble.py`) |
| `summarization/` | Summaries from the source or the English article (`gen.py`) |
| `judging/` | Claim decomposition and verification against either article (`claims.py`, `judge_faith.py`) |
| `open_weight/` | Open-weight systems, automatic anchors and the translation probe |
| `nilo_saharan/` | Acholi and Ateso runs and their analysis |
| `reasoning_effort/` | The gpt-5.6-sol reasoning-effort sweep |
| `analysis/` | Embedding AUC, bootstrap intervals and the appendix tables |
| `common/` | Model API wrapper (`llm.py`) and data loading (`data.py`) |

Prompts are in `construction/`, `summarization/` and `judging/` and in the paper's appendix. Run scripts from
`code/` with `PYTHONPATH=common:construction:summarization:judging` so imports across folders resolve.
Construction, summarization with GPT-5.5 and judging with Claude Sonnet 4.6 call commercial APIs
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) and are not bit-reproducible, so every output is released.
Open-weight systems run with transformers or vLLM (`code/requirements.txt`). The scripts were written
against the development layout (`datasets/official/...`, `eval/official/out/...`); the released files in
`data/` and `results/` are the same data under clearer names.

## Human study

The study was approved by the authors' institutional review board. Only aggregate results and the survey
items are released; no individual responses or annotator identifiers are included.

## Licenses

Code: MIT (`LICENSE`). Data keep the licenses of their sources (`DATA_LICENSE.md`).
