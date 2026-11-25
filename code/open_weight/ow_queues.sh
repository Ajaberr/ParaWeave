#!/usr/bin/env bash
# Launch every open-weight job, detached, one queue per GPU:  ow_queues.sh <gA> <gB> <gC> <gD> <gM1> <gM2> <gM3>
#   A  Qwen3-30B-A3B fp8        Arabic (the one missing Arabic system) + Lugbara + Farsi + Dari + Tajik
#   B  Qwen3-8B, then Aya-23-8B  transfer builds (their Arabic translations already exist)
#   C  Fanar-1-9B and Sunflower-9B on Lugbara, then Dorna2-8B on the three Persian sets
#   D  gpt-oss-20B on the transfer builds, then Sunflower-32B NF4 (in-process, bitsandbytes) on Lugbara
#   M1 AlignScore-large, then BLEURT     M2 MiniCheck     M3 mDeBERTa, e5, known-true pairs
cd /tmp/work/owrun || exit 1
mkdir -p logs out
SW=/mnt/nfs-server-pool2/user/hf_cache; TH=/tmp/work/hf
A=$1; B=$2; C=$3; D=$4; M1=$5; M2=$6; M3=$7
setsid nohup bash -c "./ow_vllm.sh $A qwen30a3b Qwen/Qwen3-30B-A3B $SW '--quantization fp8' 'VLLM_NO_THINK=1' ar lgg fa prs tg" > logs/qA.log 2>&1 < /dev/null &
setsid nohup bash -c "./ow_vllm.sh $B qwen8 Qwen/Qwen3-8B $SW '' 'VLLM_NO_THINK=1' lgg fa prs tg; ./ow_vllm.sh $B aya CohereForAI/aya-23-8B $SW '' '' lgg fa prs tg" > logs/qB.log 2>&1 < /dev/null &
setsid nohup bash -c "./ow_vllm.sh $C fanar QCRI/Fanar-1-9B-Instruct $SW '' '' lgg; ./ow_vllm.sh $C sunflower9b Sunbird/Sunflower-Qwen3.5-9B $TH '' 'VLLM_NO_THINK=1' lgg; ./ow_vllm.sh $C dorna2 PartAI/Dorna2-Llama3.1-8B-Instruct $TH '' '' fa prs tg" > logs/qC.log 2>&1 < /dev/null &
setsid nohup bash -c "./ow_vllm.sh $D gptoss20 openai/gpt-oss-20b $SW '' 'VLLM_REASONING_EFFORT=low PROBE_MAX_TOKENS=768' lgg fa prs tg; ./ow_hf32.sh $D" > logs/qD.log 2>&1 < /dev/null &
setsid nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh; export HF_HOME=$TH HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=$M1; /mnt/nfs-server-pool2/user/envs/astest/bin/python ow_metrics.py align; HF_HOME=$SW HF_HUB_OFFLINE=1 ~/venv-qwen/bin/python ow_metrics.py bleurt" > logs/qM1.log 2>&1 < /dev/null &
setsid nohup bash -c "export HF_HOME=$TH HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=$M2; ~/miniconda3/envs/project/bin/python ow_metrics.py minicheck" > logs/qM2.log 2>&1 < /dev/null &
setsid nohup bash -c "export HF_HOME=$TH HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=$M3; ~/miniconda3/envs/project/bin/python ow_metrics.py mdeberta && ~/miniconda3/envs/project/bin/python ow_metrics.py e5 && ~/miniconda3/envs/project/bin/python ow_metrics.py pairs" > logs/qM3.log 2>&1 < /dev/null &
sleep 2; echo "launched: $(date +%F_%T)"; ps -u user -o pid,etime,cmd | grep -E "ow_|vllm|translation_probe" | grep -v grep | head -20
