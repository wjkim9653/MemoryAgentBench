#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/shared/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"

root=$(pwd)
file_name="qwen3_4b_vllm_letta_smoke.txt"
failed_file="outputs/qwen3-4b-vllm_failed_letta_smoke.txt"

while read -r agent_config dataset_config; do
  [[ -z "${agent_config:-}" ]] && continue
  [[ "$agent_config" =~ ^# ]] && continue

  echo "................Start................"
  echo "agent_config=${agent_config}"
  echo "dataset_config=${dataset_config}"

  if ! python main.py \
    --agent_config "configs/agent_conf/RAG_Agents/qwen3-4b-vllm/${agent_config}" \
    --dataset_config "configs/data_conf/${dataset_config}" \
    --max_test_queries_ablation 1 \
    --force; then
    mkdir -p outputs
    echo "${agent_config} ${dataset_config}" >> "${failed_file}"
    echo "FAILED: ${agent_config} ${dataset_config}"
  fi

  echo "................End................"
done < "${root}/bash_files/configs/${file_name}"
