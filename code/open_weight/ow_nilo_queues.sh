#!/usr/bin/env bash
# 2026-09-25: the open-weight metric suite on the Acholi and Ateso versions of the Lugbara articles. Translation probe
# for the seven open-weight Lugbara systems, plus every anchor/metric. ow_metrics.py is incremental (keyed on
# lang|system|id|condition), so the five earlier builds are skipped; pairs.json is recomputed with a per-variety seed,
# so the earlier varieties come out identical. Open-weight only, no API.
cd /tmp/work/owrun || exit 1
mkdir -p logs out
cp -n out/pairs.json out/pairs.before_nilo.json
grep -q 'ach\]="Acholi"' ow_vllm.sh || sed -i 's/\[tg\]="Tajik (Persian written in Cyrillic)")/[tg]="Tajik (Persian written in Cyrillic)" [ach]="Acholi" [teo]="Ateso")/' ow_vllm.sh
grep -c 'ach\]="Acholi"' ow_vllm.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
SW=/mnt/nfs-server-pool2/user/hf_cache; TH=/tmp/work/hf
setsid nohup bash -c "./ow_vllm.sh 0 qwen30a3b Qwen/Qwen3-30B-A3B $SW '--quantization fp8' 'VLLM_NO_THINK=1' ach teo" > logs/nA.log 2>&1 < /dev/null &
setsid nohup bash -c "./ow_vllm.sh 1 qwen8 Qwen/Qwen3-8B $SW '' 'VLLM_NO_THINK=1' ach teo; ./ow_vllm.sh 1 aya CohereForAI/aya-23-8B $SW '' '' ach teo" > logs/nB.log 2>&1 < /dev/null &
setsid nohup bash -c "./ow_vllm.sh 2 fanar QCRI/Fanar-1-9B-Instruct $SW '' '' ach teo; ./ow_vllm.sh 2 sunflower9b Sunbird/Sunflower-Qwen3.5-9B $TH '' 'VLLM_NO_THINK=1' ach teo" > logs/nC.log 2>&1 < /dev/null &
setsid nohup bash -c "./ow_vllm.sh 3 gptoss20 openai/gpt-oss-20b $SW '' 'VLLM_REASONING_EFFORT=low PROBE_MAX_TOKENS=768' ach teo; ./ow_hf32_nilo.sh 3" > logs/nD.log 2>&1 < /dev/null &
setsid nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh; export HF_HOME=$TH HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=4; /mnt/nfs-server-pool2/user/envs/astest/bin/python ow_metrics.py align; HF_HOME=$SW HF_HUB_OFFLINE=1 ~/venv-qwen/bin/python ow_metrics.py bleurt" > logs/nM1.log 2>&1 < /dev/null &
setsid nohup bash -c "export HF_HOME=$TH HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=4; ~/miniconda3/envs/project/bin/python ow_metrics.py minicheck" > logs/nM2.log 2>&1 < /dev/null &
setsid nohup bash -c "export HF_HOME=$TH HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=6; ~/miniconda3/envs/project/bin/python ow_metrics.py mdeberta && ~/miniconda3/envs/project/bin/python ow_metrics.py e5 && ~/miniconda3/envs/project/bin/python ow_metrics.py pairs" > logs/nM3.log 2>&1 < /dev/null &
sleep 3; echo "launched $(date +%F_%T)"
