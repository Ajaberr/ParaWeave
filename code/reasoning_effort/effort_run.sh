#!/usr/bin/env bash
# GPT-5.6-sol reasoning-effort sweep on the 240 Lugbara articles: generation at five effort levels in parallel
# (Bedrock Mantle, cap $30 per level), a second pass that fills any failed calls, then the Sonnet 4.6 batch judge
# under both groundings for every level (cap $40 per level), exactly as every other Lugbara system.
set -uo pipefail
S="scratch/"
LOGD="datasets/official/derived/DiaBench-ML/logs"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 GEN_WORKERS=4 LLM_BUDGET_USD_OPENAI=30
LEVELS="none low medium high xhigh"
echo "EFFORT START $(date +%T)"
declare -A W=([none]=4 [low]=8 [medium]=10 [high]=12 [xhigh]=14)   # slower levels get more parallel requests
for e in $LEVELS; do GEN_WORKERS=${W[$e]} python "$S/effort_gen.py" $e >> "$LOGD/gen_lgg_gpt56sol_e$e.log" 2>&1 & done
wait
for e in $LEVELS; do python "$S/effort_gen.py" $e >> "$LOGD/gen_lgg_gpt56sol_e$e.log" 2>&1; done
for e in $LEVELS; do echo "  gen e$e: $(wc -l < "eval/official/out/gen.mlext.openai-gpt56sol-e$e.jsonl") lines"; done
echo "GEN DONE $(date +%T)"
for e in $LEVELS; do bash "$S/lgg_judge.sh" "openai-gpt56sol-e$e" & done
wait
echo "JUDGE DONE $(date +%T)"
