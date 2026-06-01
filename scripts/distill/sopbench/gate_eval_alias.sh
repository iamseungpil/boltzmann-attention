#!/bin/bash
# gate_eval_alias.sh — re-eval the alias_s3_gate adapter (Phase-A serve failed). Runs on GPU1
# (parallel with s3_gate solo retrain on GPU0). Writes eval_alias_s3_gate/ so gate_recover's
# final breakdown picks it up. GPU-isolated (kills only GPU1 procs).
set +e
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
REPO=/home/woori/workspace_common/boltzmann-attention-pi
CLONE=/home/woori/scratch/SOPBench
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
SUM=$OUT/GATE_ALIAS_REEVAL.txt
AD=$RUNS/qwen7b_tbox_alias_s3_gate_lodo_bank
echo "=== alias_s3_gate re-eval start $(date) ===" > $SUM
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null
CUDA_VISIBLE_DEVICES=1 nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
  --lora-modules tbox_v2=$AD --port 9002 --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
  > $OUT/serve_alias_s3_gate.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 3 http://localhost:9002/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 4; done
if ! curl -s -m 3 http://localhost:9002/v1/models 2>/dev/null | grep -q tbox_v2; then
  echo "SERVE FAILED" >> $SUM; for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; exit 1; fi
echo "endpoint up $(date)" >> $SUM
cd $CLONE
env SOPBENCH_GATE=1 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3 SOPBENCH_VLLM_BASE_URL=http://localhost:9002/v1 $PY run_simulation.py \
  --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
  --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_alias_s3_gate --env_mode prompt \
  > $OUT/sim_alias_s3_gate.log 2>&1
$PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
  --output_dir $OUT/eval_alias_s3_gate > $OUT/evalout_alias_s3_gate.txt 2>&1
echo "alias_s3_gate: $(grep -E 'Mean Pass Rate' $OUT/evalout_alias_s3_gate.txt | tail -1)" >> $SUM
for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
sleep 4; rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null
$PY - >> $SUM 2>&1 <<'PYEOF'
import json,glob
def e0(x):
    e=x.get("evaluations"); return (e[0] if e else {}) if isinstance(e,list) else (e or {})
fs=glob.glob('/home/woori/scratch/sft_alias_run/eval_alias_s3_gate/bank/*.json')
if fs:
    d=json.load(open(fs[0])); T=[x for x in d if e0(x).get("action_should_succeed")]; F=[x for x in d if not e0(x).get("action_should_succeed")]
    both=sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly"))
    print(f"alias_s3_gate: should_T succ={sum(bool(e0(x).get('success')) for x in T)}/48 dirgraph={sum(bool(e0(x).get('dirgraph_satisfied')) for x in T)} goal={sum(bool(e0(x).get('action_called_correctly')) for x in T)} BOTH={both} || should_F={sum(bool(e0(x).get('success')) for x in F)}/86")
PYEOF
echo "=== alias re-eval DONE $(date) ===" >> $SUM
