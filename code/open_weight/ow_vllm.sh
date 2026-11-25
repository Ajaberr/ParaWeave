#!/usr/bin/env bash
# One open-weight system through a local vLLM server on ONE GPU (the only reliable path on gpu-server), translating the
# listed builds' source sentences with translation_probe_ml.py, then the server is stopped.
#   ow_vllm.sh <gpu> <system> <repo> <hf_home> "<extra vllm args>" "<client env>" <lang>...
cd /tmp/work/owrun || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
GPU=$1; SYS=$2; REPO=$3; HH=$4; VARGS=$5; CENV=$6; shift 6
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60
export HF_TOKEN=$(grep -E '^HF_TOKEN=' /tmp/work/lgg/.env | head -1 | cut -d= -f2- | tr -d "\"'")   # only the HF token
unset OPENAI_API_KEY ANTHROPIC_API_KEY OPENAI_BASE_URL
declare -A LN=([ar]="Arabic" [lgg]="Lugbara" [fa]="Farsi (Iranian Persian)" [prs]="Dari (Afghan Persian)" [tg]="Tajik (Persian written in Cyrillic)")
mkdir -p logs out
echo "VSTART $SYS gpu=$GPU $(date +%F_%T)"
safe=$(echo "$REPO" | sed 's#/#--#')
if [ ! -d "$HH/hub/models--$safe/snapshots" ]; then
  for try in 1 2 3 4 5 6; do
    HF_HOME=$HH ~/miniconda3/envs/project/bin/hf download "$REPO" --max-workers 4 > "logs/fetch_$SYS.log" 2>&1 && { echo "=== [$SYS] fetched on try $try $(date +%T)"; break; }
    echo "=== [$SYS] fetch try $try failed $(date +%T)"; sleep 30
  done
fi
PORT=$((8120 + GPU))
while curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/models 2>/dev/null | grep -q 200; do PORT=$((PORT + 20)); done
conda activate env2
HF_HOME=$HH HF_HUB_OFFLINE=1 VLLM_ENGINE_READY_TIMEOUT_S=2400 CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
  --model $REPO --served-model-name $REPO --port $PORT --max-model-len 4096 --max-num-seqs 64 \
  --gpu-memory-utilization 0.80 --dtype bfloat16 --enforce-eager $VARGS > logs/vllm_$SYS.log 2>&1 &
VPID=$!
ok=0
for i in $(seq 1 170); do
  curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q "$REPO" && { ok=1; break; }
  kill -0 $VPID 2>/dev/null || break
  sleep 15
done
if [ $ok = 1 ]; then
  echo "=== [$SYS] server up on :$PORT $(date +%T)"
  for L in "$@"; do
    if env VLLM_BASE_URL=http://localhost:$PORT/v1 VLLM_MODEL=$REPO GEN_WORKERS=48 PROBE_LANG_NAME="${LN[$L]}" $CENV \
         python translation_probe_ml.py $L $SYS out/trans.$L.$SYS.jsonl; then
      echo "=== [$SYS/$L] OK $(date +%T) lines=$(wc -l < out/trans.$L.$SYS.jsonl)"
    else
      echo "=== [$SYS/$L] FAILED $(date +%T)"
    fi
  done
else
  echo "=== [$SYS] server FAILED $(date +%T)"; tail -25 logs/vllm_$SYS.log
fi
kill $VPID 2>/dev/null; sleep 15; kill -9 $VPID 2>/dev/null
echo "VEND $SYS $(date +%F_%T)"
