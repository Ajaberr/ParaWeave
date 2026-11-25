#!/usr/bin/env bash
# Decompose + judge one system's Lugbara summaries under both groundings: Sonnet 4.6 via the Batches API
# (as the Arabic run), decomposer = same model. Usage: lgg_judge.sh <system> [<system> ...]
set -uo pipefail
ROOT="."
cd "$ROOT/eval/official" || exit 1
export ML_EXTENSION_FILE="$ROOT/datasets/official/derived/DiaBench-ML/lugbara_eval.jsonl"
export JUDGE_SRC_LANG=Lugbara DECOMP_SYSTEM=claude ANTHROPIC_MODEL=claude-sonnet-4-6 CLAUDE_BATCH=1 JUDGE_WORKERS=8
export LLM_BUDGET_USD_ANTHROPIC=40 LLM_USAGE_LOG="$ROOT/datasets/official/derived/DiaBench-ML/logs/usage_lugbara_judge.jsonl"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
for sys in "$@"; do
  LOG="$ROOT/datasets/official/derived/DiaBench-ML/logs/judge_lgg_$sys.$(date +%Y%m%d-%H%M).log"
  echo "==== judge $sys $(date)" > "$LOG"
  python judge_faith.py "out/gen.mlext.$sys.jsonl" claude dialect,gold_reference >> "$LOG" 2>&1
  echo "==== $sys exit $? $(date)" >> "$LOG"
done
