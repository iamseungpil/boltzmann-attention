#!/bin/bash
# ===== dump A-prov + diff-grounding catch-rate (P4 §8-1) — waits for overnight, no race =====
# Deterministic handoff: waits for BOTH overnight orchestrators DONE (free GPU), then serves the
# A-prov LoRA, runs the harness with --dump, and runs the §5b diff-grounding verifier OFFLINE to
# report catchable-GBW recall + gold-correct false-reject (escape-hatch size + synonym need).
# A-prov chosen: direct GBW=4 (P4 §1). GPU brief (~5min serve+eval). Not racing the overnight.
set -u
GPU="${1:-0}"; PORT="${2:-8019}"
S=/home/woori/scratch
MA=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/ma
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
LOG=$S/dump_verify.log
exec > $LOG 2>&1; set -x; date
echo "waiting for overnight orchestrators DONE (free GPU) ..."
for i in $(seq 1 960); do                       # up to 8h
  grep -q OVERNIGHT_g0_DONE $S/ma_overnight_g0.log 2>/dev/null \
    && grep -q OVERNIGHT_g1_DONE $S/ma_overnight_g1.log 2>/dev/null && { echo "overnight done (i=$i)"; break; }
  sleep 30
done
sleep 20
cd /home/woori/workspace_common/boltzmann-attention-pi && git pull --ff-only
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --lora-modules aprov=$S/sft_runs/fact_A_prov --max-lora-rank 32 > $S/vllm_dumpverify.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q aprov && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -20 $S/vllm_dumpverify.log; exit 1; }

echo "===== generate per-case dump (A-prov) ====="
$PY $MA/m_sigma_transfer_eval_v4.py --base http://localhost:$PORT/v1 --model aprov --tag aprov --dump $S/dump_aprov.jsonl
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done

echo "===== §5b diff-grounding verifier catch-rate (offline, GPU 0) ====="
$PY $MA/ma_diff_grounding.py --dump $S/dump_aprov.jsonl --cases $S/ma_eval_cases.jsonl
echo DUMP_VERIFY_DONE; date
