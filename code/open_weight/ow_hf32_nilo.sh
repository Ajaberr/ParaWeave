#!/usr/bin/env bash
# Sunflower-32B in 4-bit NF4 (bitsandbytes), in-process on ONE GPU, as for Lugbara (ow_hf32.sh), translating the
# Acholi and Ateso source sentences.   ow_hf32_nilo.sh <gpu>
cd /tmp/work/owrun || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate project
export HF_HOME=$WORK/hf_sun HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_DEVICE_ORDER=PCI_BUS_ID
unset OPENAI_API_KEY ANTHROPIC_API_KEY OPENAI_BASE_URL VLLM_BASE_URL
for L in ach teo; do
  if [ "$L" = ach ]; then N=Acholi; else N=Ateso; fi
  echo "HSTART sunflower32q4/$L gpu=$1 $(date +%F_%T)"
  if env HF_4BIT=1 HF_DEVICE=cuda:0 HF_BATCH=16 HF_MAXLEN=1024 CUDA_VISIBLE_DEVICES=$1 HF_MODEL=Sunbird/Sunflower-32B \
       PROBE_LANG_NAME=$N python translation_probe_ml.py $L sunflower32q4 out/trans.$L.sunflower32q4.jsonl; then
    echo "=== [sunflower32q4/$L] OK $(date +%T) lines=$(wc -l < out/trans.$L.sunflower32q4.jsonl)"
  else
    echo "=== [sunflower32q4/$L] FAILED $(date +%T)"
  fi
done
echo "HEND sunflower32q4 $(date +%F_%T)"
