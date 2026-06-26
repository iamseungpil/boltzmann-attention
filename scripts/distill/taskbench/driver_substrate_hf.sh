#!/bin/bash
# 둘째-기판 재현 (SELECTOR_DESIGN 큐 ⑷, 리뷰 1순위(a)) — TB huggingface sub500.
# 사전등록(동결): best-stack(SEL-1 β2 + SEL-4 Reviewer) 공식 link F1 ≥ k0 단일 +3pp.
#   미달 = "MM-특이" 정직 기록·선별 헤드라인 강등.
# 구성: AR8 = tb_lodo_hf 어댑터 + guided + temp0.8 K=8 (hf는 LODO held-out = 전이 설정)
#   / hetero = qwen3b·qwen14b·qwen3_4b·qwen3_14b (기존) + Track-B hf 4종 (extra)
# Run: setsid bash driver_substrate_hf.sh </dev/null >/dev/null 2>&1 &
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
SC=$R/scripts/distill/taskbench
TB=/home/woori/scratch/JARVIS_tb/taskbench
TBPRED=$TB/data_huggingface_sub500/predictions
TRACKB=$R/reports/facet_rft_2026/trackb_raw/preds/data_huggingface_sub500
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
S=/home/woori/scratch
exec > $S/substrate_hf.log 2>&1
set -x
cd $R && git pull --ff-only -q

kill_gpu() {
  for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader); do
    kill -9 $p 2>/dev/null; done; sleep 8
}

HM_HF="qwen3b,qwen14b,qwen3_4b,qwen3_14b"
EXTRA=""
for f in qwen25_32b qwen25_72b qwen3_235b_a22b_int4 qwen3_32b; do
  [ -f $TRACKB/$f.json ] && EXTRA="$EXTRA --extra $f=$TRACKB/$f.json"
done

# ---------- 1. AR8 샘플링 (GPU1) ----------
VTAG=tb_hfg_night
$IP $SC/tb_guided_patch.py $TB/inference.py || true
[ -f $S/tb_guided_hf_schema.json ] || $IP $SC/tb_guided_schema.py \
  --tool_desc $TB/data_huggingface/tool_desc.json --dep resource \
  --out $S/tb_guided_hf_schema.json
kill_gpu 1
CUDA_VISIBLE_DEVICES=1 VLLM_PORT=8203 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 --served-model-name base_model --enable-lora \
  --lora-modules ${VTAG}=$RUNS/qwen7b_tb_lodo_hf \
  --max-model-len 8192 --gpu-memory-utilization 0.85 > $S/vllm_${VTAG}.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8001/v1/models | grep -q "\"$VTAG\"" && ok=1 && break; sleep 10
done
[ $ok = 1 ] || { echo HF_SERVE_FAIL; exit 1; }
for k in 0 1 2 3 4 5 6 7; do
  [ -f $TBPRED/tb_hfg_k${k}.json ] && { echo "SKIP k$k"; continue; }
  rm -f $TBPRED/$VTAG.json
  (cd $TB && TB_GUIDED=1 TB_GUIDED_SCHEMA=$S/tb_guided_hf_schema.json \
    $IP inference.py --data_dir data_huggingface_sub500 --api_addr localhost --api_port 8001 \
    --api_key dummy --llm $VTAG --multiworker 8 --dependency_type resource --temperature 0.8)
  mv $TBPRED/$VTAG.json $TBPRED/tb_hfg_k${k}.json
  echo "HF_K_DONE $k"
done

# ---------- 2. 선별 (serve 유지 — SEL-4가 base_model 사용) ----------
# C0: k0 단일 통제
$IP $SC/tb_build_eval.py --tb_dir $TB --domain data_huggingface \
  --pred_file $TBPRED/tb_hfg_k0.json \
  --dst $TB/data_huggingface_sub500_eval_tb_hfg_k0 --llm tb_hfg_k0 > $S/hf_c0.txt 2>&1
# SEL-1
(cd $SC && $IP tb_select_official.py --tb_dir $TB --domain data_huggingface \
  --ar_tag tb_hfg_k --ar_group hfg --hm $HM_HF --prior_beta 2.0 $EXTRA \
  --out $TBPRED/tb_sel1_hf.json)
$IP $SC/tb_build_eval.py --tb_dir $TB --domain data_huggingface \
  --pred_file $TBPRED/tb_sel1_hf.json \
  --dst $TB/data_huggingface_sub500_eval_tb_sel1_hf --llm tb_sel1_hf > $S/hf_sel1.txt 2>&1
# SEL-4 (best-stack)
(cd $SC && $IP tb_reviewer_select.py --tb_dir $TB --domain data_huggingface \
  --ar_tag tb_hfg_k --ar_group hfg --hm $HM_HF $EXTRA \
  --endpoint http://localhost:8001/v1 --served base_model --lam 1.0 \
  --out $TBPRED/tb_sel4_hf.json)
$IP $SC/tb_build_eval.py --tb_dir $TB --domain data_huggingface \
  --pred_file $TBPRED/tb_sel4_hf.json \
  --dst $TB/data_huggingface_sub500_eval_tb_sel4_hf --llm tb_sel4_hf > $S/hf_sel4.txt 2>&1
grep -hE "link_binary_f1|node_micro_f1_no" $S/hf_c0.txt $S/hf_sel1.txt $S/hf_sel4.txt
kill_gpu 1
echo "SUBSTRATE_HF_DONE $(date)"
