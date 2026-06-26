#!/bin/bash
# ★야간 배치 2026-06-12 — E8(GPU1)∥[E1대기→E4→E6](GPU0)→E5(zero-GPU)→NIGHT_REPORT+git push
# 사전예측(동결): E8 lodo_hf 35.0→+1~3 / E4 in-domain slim 손실<3.1 / E6 held-out 선별갭>+5 / E5 회수 18%→≥30%
# log: /home/woori/scratch/tb_night.log, sentinel NIGHT_DONE
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
SC=$R/scripts/distill/taskbench
S=/home/woori/scratch
REPORT=$R/reports/facet_rft_2026/NIGHT_REPORT_2026_06_12.md
exec > $S/tb_night.log 2>&1
set -x
cd $R && git pull --ff-only -q

kill_vllm() { for p in $(nvidia-smi --id=$1 --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm | cut -d, -f1); do kill -9 $p; done; sleep 8; }
serve() { # gpu port tag adapter(없으면 base)
  if [ -n "$4" ]; then
    CUDA_VISIBLE_DEVICES=$1 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $2 \
      --served-model-name base_model --enable-lora --lora-modules ${3}=$4 \
      --max-model-len 8192 --gpu-memory-utilization 0.85 > $S/vllm_night_${3}.log 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$1 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $2 \
      --served-model-name ${3} --max-model-len 8192 --gpu-memory-utilization 0.85 \
      > $S/vllm_night_${3}.log 2>&1 &
  fi
  for i in $(seq 1 90); do curl -s localhost:$2/v1/models | grep -q "\"$3\"" && return 0; sleep 10; done
  echo "SERVE_FAIL_$3"; return 1
}
infer() { # port llm datadir dep [slimdir-via-datadir] [guided_schema] [extra]
  local P=$1 L=$2 D=$3 DEP=$4 SCH=$5; shift 5
  ( cd $TB && TB_GUIDED=1 TB_GUIDED_SCHEMA=$SCH \
    $IP inference.py --data_dir $D --api_addr localhost --api_port $P --api_key dummy \
    --llm $L --multiworker 8 --dependency_type $DEP "$@" )
}

# ---------- 스키마·slim 변형 준비 (zero-GPU) ----------
$IP $SC/tb_guided_patch.py $TB/inference.py || true
$IP $SC/tb_guided_schema.py --tool_desc $TB/data_huggingface/tool_desc.json --dep resource --out $S/tb_guided_hf_schema.json
$IP $SC/tb_guided_schema.py --tool_desc $TB/data_dailylifeapis/tool_desc.json --dep temporal --out $S/tb_guided_daily_schema.json || true
[ -f $S/tb_guided_mm_schema.json ] || $IP $SC/tb_guided_schema.py --tool_desc $TB/data_multimedia/tool_desc.json --dep resource --out $S/tb_guided_mm_schema.json
$IP - <<'EOF'
import json, shutil, os
TB="/home/woori/scratch/JARVIS_tb/taskbench"
for dom in ["data_huggingface","data_dailylifeapis"]:
    src=f"{TB}/{dom}_sub500"; dst=f"{TB}/{dom}_sub500_slim"
    os.makedirs(dst, exist_ok=True)
    for f in ["user_requests.json","graph_desc.json"]:
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    td=json.load(open(f"{TB}/{dom}/tool_desc.json"))
    keep={"id","input-type","output-type","parameters"}
    json.dump({"nodes":[{k:v for k,v in t.items() if k in keep} for t in td["nodes"]]}, open(f"{dst}/tool_desc.json","w"))
print("[slim] hf/daily sub500_slim ready")
EOF

# ---------- N0: E8 HF held-out guided (GPU1, 병렬) ----------
kill_vllm 1
serve 1 8001 tb_lodo_hf_guided $RUNS/qwen7b_tb_lodo_hf && {
  infer 8001 tb_lodo_hf_guided data_huggingface resource $S/tb_guided_hf_schema.json
  $IP $SC/tb_build_eval.py --tb_dir $TB --domain data_huggingface \
    --llm tb_lodo_hf_guided --dst $TB/data_huggingface_evalfull_tb_lodo_hf_guided
  echo "E8_DONE"
} &
E8_PID=$!

# ---------- E1 대기 (GPU0, 최대 70분) ----------
for i in $(seq 1 70); do grep -q GUIDED_BASE_DONE $S/tb_guided_base.log && break; sleep 60; done

# ---------- N1: E4 promptslim in-domain (GPU0, dpo2 서빙 재사용) ----------
kill_vllm 0
serve 0 8000 tb_dpo2_night $RUNS/qwen7b_tb_dpo2_mm || exit 1
for ARM in hf_full:data_huggingface_sub500:resource:hf hf_slim:data_huggingface_sub500_slim:resource:hf dl_full:data_dailylifeapis_sub500:temporal:daily dl_slim:data_dailylifeapis_sub500_slim:temporal:daily; do
  IFS=: read NAME DDIR DEP SK <<< "$ARM"
  rm -f $TB/$DDIR/predictions/tb_dpo2_night.json
  infer 8000 tb_dpo2_night $DDIR $DEP $S/tb_guided_${SK}_schema.json
  mv $TB/$DDIR/predictions/tb_dpo2_night.json $TB/$DDIR/predictions/tb_dpo2g_${NAME}.json
  case $NAME in hf_*) GOLDDOM=data_huggingface;; *) GOLDDOM=data_dailylifeapis;; esac
  $IP $SC/tb_build_eval.py --tb_dir $TB --domain $GOLDDOM \
    --pred_file $TB/$DDIR/predictions/tb_dpo2g_${NAME}.json \
    --dst $TB/${GOLDDOM}_sub500_eval_tb_dpo2g_${NAME} --llm tb_dpo2g_${NAME}
  echo "E4_ARM_DONE $NAME"
done

# ---------- N2: E6 held-out K=8 샘플링 (GPU0, 동일 서버) ----------
for k in 0 1 2 3 4 5 6 7; do
  rm -f $TB/data_multimedia_sub500/predictions/tb_dpo2_night.json
  infer 8000 tb_dpo2_night data_multimedia_sub500 resource $S/tb_guided_mm_schema.json --temperature 0.8
  mv $TB/data_multimedia_sub500/predictions/tb_dpo2_night.json $TB/data_multimedia_sub500/predictions/tb_dpo2g_mmk${k}.json
  echo "E6_K_DONE $k"
done
kill_vllm 0

# ---------- E8 합류 ----------
wait $E8_PID || true

# ---------- N3: E5 스코어러 v1 (zero-GPU) ----------
$IP $SC/tb_kgate_select.py --all $S/tb_rft/winners_rft2_mm.jsonl.all \
  --sft_jsonl $S/tb_sft/train_lodo_mm.jsonl \
  --tool_desc $TB/data_huggingface/tool_desc.json --tool_desc $TB/data_dailylifeapis/tool_desc.json \
  --graph_desc $TB/data_huggingface/graph_desc.json --graph_desc $TB/data_dailylifeapis/graph_desc.json \
  > $S/e5_v1_indomain.txt
cat $S/e5_v1_indomain.txt
PREDARGS=""; for k in 0 1 2 3 4 5 6 7; do PREDARGS="$PREDARGS --pred $TB/data_multimedia_sub500/predictions/tb_dpo2g_mmk${k}.json"; done
(cd $SC && $IP tb_kgate_heldout.py $PREDARGS \
  --gold $TB/data_multimedia/data.json --tool_desc $TB/data_multimedia/tool_desc.json \
  --graph_desc $TB/data_multimedia/graph_desc.json) > $S/e6_heldout.txt
cat $S/e6_heldout.txt

# ---------- N4: NIGHT_REPORT 생성 + push ----------
{
echo "# NIGHT REPORT 2026-06-12 (자동 생성 — tb_night_0612.sh)"
echo "> 사전예측: E8 +1~3 / E4 slim 손실<3.1 / E6 선별갭>+5 / E5 회수≥30%. 판정·박제는 Track A 아침 세션."
echo; echo "## E1 base+guided MM full (2×2 마지막 셀)"
grep -E "  link_binary_f1|  node_micro_f1_no" $S/tb_guided_base.log | tail -2
echo; echo "## E8 HF held-out guided (lodo_hf, full — unguided 통제=35.0)"
grep -E "  link_binary_f1|  node_micro_f1_no" $S/tb_night.log | head -2
ls $TB/data_huggingface_evalfull_tb_lodo_hf_guided/metrics/*.json >/dev/null 2>&1 && \
  $IP -c "import json,glob; m=json.load(open(glob.glob('$TB/data_huggingface_evalfull_tb_lodo_hf_guided/metrics/*.json')[0]))['overall_overall']; print('e-F1', m['link_binary_f1'], 'n-F1', m['node_micro_f1_no_matching'])"
echo; echo "## E4 promptslim in-domain (dpo2+guided; 통제: HF sub500 54.10 / daily 83.64 — unguided)"
for NAME in hf_full hf_slim dl_full dl_slim; do
  for GOLDDOM in data_huggingface data_dailylifeapis; do
    F=$TB/${GOLDDOM}_sub500_eval_tb_dpo2g_${NAME}/metrics/tb_dpo2g_${NAME}.json
    [ -f $F ] && $IP -c "import json; m=json.load(open('$F'))['overall_overall']; print('$NAME', 'e-F1', round(m['link_binary_f1'],4), 'n-F1', round(m['node_micro_f1_no_matching'],4))"
  done
done
echo; echo "## E5 스코어러 v1 (in-domain .all)"; cat $S/e5_v1_indomain.txt
echo; echo "## E6 held-out K=8 선별 (MM sub500, dpo2+guided, temp0.8)"; cat $S/e6_heldout.txt
echo; echo "원본 로그: tb_night.log·tb_guided_base.log / pred: tb_dpo2g_{hf,dl}_{full,slim}·tb_dpo2g_mmk0-7·tb_lodo_hf_guided"
} > $REPORT
cd $R && git add reports/facet_rft_2026/NIGHT_REPORT_2026_06_12.md && \
  git commit -m "NIGHT_REPORT 2026-06-12: E1/E4/E5/E6/E8 auto results (overnight batch)" -q && git push -q || echo PUSH_FAIL
echo "NIGHT_DONE $(date)"
