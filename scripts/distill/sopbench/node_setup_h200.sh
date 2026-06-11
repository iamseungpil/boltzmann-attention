#!/bin/bash
# node_setup_h200.sh — one-shot env bootstrap on an H200 holder node (Track B coworker).
# Run as:  bash <(curl ...) or  git clone first, then  bash node_setup_h200.sh
# Needs: GH_TOKEN in env (passed via amlt submit_args) for the private repo clone.
set -ex
export HF_HUB_CACHE=/scratch/hf_cache
mkdir -p /scratch/logs /scratch/venvs /scratch/hf_cache /scratch/sopbench_runs /scratch/sft_runs
cd /scratch

# 1. clones (idempotent)
[ -d /scratch/SOPBench ] || git clone https://github.com/Leezekun/SOPBench.git /scratch/SOPBench
if [ ! -d /scratch/boltzmann-attention ]; then
  git clone -b facet-rft-2026 "https://${GH_TOKEN}@github.com/iamseungpil/boltzmann-attention.git" /scratch/boltzmann-attention
fi
cd /scratch/boltzmann-attention && git pull --ff-only && cd /scratch

# 2. venv: vllm serving + SOPBench runner in one env (py3.10 from image)
[ -d /scratch/venvs/sop_env ] || python3 -m venv /scratch/venvs/sop_env
PIP=/scratch/venvs/sop_env/bin/pip
$PIP install -q --upgrade pip
$PIP install -q "vllm==0.10.2" openai tqdm termcolor colorama pydantic anthropic "huggingface_hub[cli]"

# 3. training env deps (train node; harmless on eval node)
# <5 pin: trainer targets 4.x API (torch_dtype etc.); >=4.55.2 keeps vllm 0.10.2 happy
$PIP install -q "transformers>=4.55.2,<5" peft accelerate datasets

# 4. deploy two-stage patch into the SOPBench clone (idempotent, .bak backups)
/scratch/venvs/sop_env/bin/python /scratch/boltzmann-attention/scripts/distill/sopbench/apply_two_stage_patch.py /scratch/SOPBench

# 5. persist tokens/env for later ssh shells + sync loop (job env may not propagate)
grep -q HF_HUB_CACHE ~/.bashrc 2>/dev/null || echo "export HF_HUB_CACHE=/scratch/hf_cache" >> ~/.bashrc
[ -n "${HF_TOKEN:-}" ] && /scratch/venvs/sop_env/bin/hf auth login --token "$HF_TOKEN" > /dev/null 2>&1 || true

# 6. background-download Qwen2.5-32B-Instruct weights (~62GB)
nohup /scratch/venvs/sop_env/bin/hf download Qwen/Qwen2.5-32B-Instruct > /scratch/logs/hfdl_32b.log 2>&1 &
echo "SETUP_DONE (32B download continues in background: tail /scratch/logs/hfdl_32b.log)"
