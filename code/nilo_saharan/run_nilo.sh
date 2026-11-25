#!/usr/bin/env bash
# VM (gpu-server), 2026-09-25: source-input summaries of the 240 Lugbara articles in Acholi and Ateso, i.e. the same
# English articles with SALT's multi-way translations (processing/official/build_salt_parallel.py). Every system
# uses the exact recipe and code of its Lugbara run (lgg/run_gen_lgg.sh, run_qwen30_vllm.sh, run_sunflower9b.sh,
# lgg/run_gen_sunflower_q4.sh, and the lgg copies of gen.py/llm.py/data.py). Only the source-input ("dialect")
# condition is generated: the English input is the same article as for Lugbara, so those summaries are reused.
# Open-weight only; the API keys are unset so no paid call can happen.
#   bash run_nilo.sh setup            then, detached, one lane per GPU:   bash run_nilo.sh lane <0..4>
set -u
B=/tmp/work/nilo
SW=/mnt/nfs-server-pool2/user/hf_cache; TH=/tmp/work/hf; SUN=$WORK/hf_sun
declare -A LN=([ach]=Acholi [teo]=Ateso)
source ~/miniconda3/etc/profile.d/conda.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
unset OPENAI_API_KEY ANTHROPIC_API_KEY OPENAI_BASE_URL OPENAI_PROJECT_ID
export HF_TOKEN=$(grep -E '^HF_TOKEN=' /tmp/work/lgg/.env | head -1 | cut -d= -f2- | tr -d "\"'")

setup () {
  mkdir -p $B/logs
  for L in ach teo; do
    mkdir -p $B/$L/out $B/$L/done
    cp /tmp/work/lgg/gen.py /tmp/work/lgg/llm.py /tmp/work/lgg/data.py $B/$L/
  done
  ls -la $B/ach $B/teo
}

hf_gen () {   # gpu tag repo hf_home [VAR=val ...]   in-process transformers, as lgg/run_gen_lgg.sh
  local gpu=$1 tag=$2 repo=$3 hh=$4; shift 4
  conda activate project
  for L in ach teo; do
    [ -f $B/$L/done/$tag ] && { echo "[$L/$tag] done -> skip"; continue; }
    echo "===== [$L/$tag] $repo GPU $gpu $(date +%F_%T)"
    if ( cd $B/$L && env HF_HOME=$hh CUDA_VISIBLE_DEVICES=$gpu HF_MODEL=$repo ML_EXTENSION_FILE=$B/$L/${L}_eval.jsonl \
           GEN_SRC_LANG=${LN[$L]} GEN_WORKERS=4 HF_BATCH=8 HF_MAXLEN=4096 GPTOSS_EFFORT=low "$@" \
           python gen.py mlext $tag dialect ); then
      touch $B/$L/done/$tag; echo "===== [$L/$tag] OK $(date +%T) lines=$(wc -l < $B/$L/out/gen.mlext.$tag.jsonl)"
    else echo "===== [$L/$tag] FAILED $(date +%T)"; fi
  done
}

vllm_gen () {   # gpu tag repo hf_home port "server args" [client VAR=val ...]   one-GPU vLLM server, as run_*_vllm.sh
  local gpu=$1 tag=$2 repo=$3 hh=$4 port=$5 vargs=$6; shift 6
  while curl -s -o /dev/null http://localhost:$port/v1/models 2>/dev/null; do port=$((port + 20)); done
  conda activate env2
  HF_HOME=$hh VLLM_ENGINE_READY_TIMEOUT_S=2400 CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.openai.api_server \
    --model $repo --served-model-name $repo --port $port --dtype bfloat16 --enforce-eager $vargs > $B/logs/vllm_$tag.log 2>&1 &
  local vpid=$! ok=0
  for i in $(seq 1 170); do
    curl -s http://localhost:$port/v1/models 2>/dev/null | grep -q "$repo" && { ok=1; break; }
    kill -0 $vpid 2>/dev/null || break
    sleep 15
  done
  if [ $ok = 1 ]; then
    echo "===== [$tag] vLLM up on :$port $(date +%T)"
    conda activate project
    for L in ach teo; do
      [ -f $B/$L/done/$tag ] && { echo "[$L/$tag] done -> skip"; continue; }
      if ( cd $B/$L && env VLLM_BASE_URL=http://localhost:$port/v1 VLLM_MODEL=$repo ML_EXTENSION_FILE=$B/$L/${L}_eval.jsonl \
             GEN_SRC_LANG=${LN[$L]} "$@" python gen.py mlext $tag dialect ); then
        touch $B/$L/done/$tag; echo "===== [$L/$tag] OK $(date +%T) lines=$(wc -l < $B/$L/out/gen.mlext.$tag.jsonl)"
      else echo "===== [$L/$tag] FAILED $(date +%T)"; fi
    done
  else echo "===== [$tag] vLLM FAILED $(date +%T)"; tail -20 $B/logs/vllm_$tag.log; fi
  kill $vpid 2>/dev/null; sleep 15; kill -9 $vpid 2>/dev/null
}

case "${1:-}" in
  setup) setup ;;
  lane)
    echo "LANESTART $2 $(date +%F_%T)"
    case "$2" in
      0) hf_gen 0 qwen8 Qwen/Qwen3-8B $SW; hf_gen 0 aya CohereForAI/aya-23-8B $SW ;;
      1) hf_gen 1 fanar QCRI/Fanar-1-9B-Instruct $SW; hf_gen 1 gptoss20 openai/gpt-oss-20b $SW ;;
      2) vllm_gen 2 qwen30a3b Qwen/Qwen3-30B-A3B $SW 8131 "--quantization fp8 --gpu-memory-utilization 0.80 --max-model-len 4096 --max-num-seqs 32" \
           VLLM_NO_THINK=1 GEN_TEMP=0 GEN_WORKERS=16 ;;
      3) vllm_gen 3 sunflower9b Sunbird/Sunflower-Qwen3.5-9B $TH 8132 "--gpu-memory-utilization 0.85 --max-model-len 8192 --max-num-seqs 32" \
           VLLM_NO_THINK=1 GEN_TEMP=0 GEN_WORKERS=16 ;;
      4) hf_gen 4 sunflower32q4 Sunbird/Sunflower-32B $SUN HF_4BIT=1 HF_DEVICE=cuda:0 GEN_TEMP=0 HF_BATCH=8 ;;
    esac
    echo "LANEEND $2 $(date +%F_%T)" ;;
  *) echo "usage: run_nilo.sh setup | lane <0..4>" ;;
esac
