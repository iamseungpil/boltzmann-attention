#!/bin/bash
# P-lora 풀에 SEL-1+SEL-4 본격 적용 (2026-06-13 사용자 지시 — GPU0 유휴 활용, P-D0/결정론과 분리)
# 배경: P-lora(목적-다양 어댑터 8종 tb_dl_0~7) 다양성 0.1535·oracle 0.874인데 균등MBR 회수 18%뿐
#   (TB §8.9f). 본격 선별기(prior-가중 MBR + 7B Reviewer)로 oracle 0.874 회수 시도.
# ★8종이 서로 다른 proposer = --ar_group_per_slot (각 독립 그룹, prior-1표가 이종성 살림).
# 사전등록(SELECTOR ⑸ 후속·§7 측도):
#   C0 = tb_dl_0 단일 / SEL-1 = prior-MBR(per-slot) / SEL-4 = +Reviewer
#   예측: SEL-1 > C0 +5pp (다양성 회수) · SEL-4 ≥ SEL-1 (소수정답 구제). 회수율(sel-mean)/(oracle-mean)
#   현 18% → SEL-1+4로 ↑가 핵심 지표. + H6 합류 변형(P-lora8+H6)도 1회.
# GPU0 전용 (P-D0=GPU1·결정론=대기). 끝나며 GPU0 정리 → 결정론 드라이버 충돌 없음.
# log: /home/woori/scratch/plora_select.log, sentinel PLORA_SELECT_DONE
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
SC=$R/scripts/distill/taskbench
TB=/home/woori/scratch/JARVIS_tb/taskbench
TBPRED=$TB/data_multimedia_sub500/predictions
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
S=/home/woori/scratch
exec > $S/plora_select.log 2>&1
set -x
cd $R && git pull --ff-only -q

# GPU0 비었는지 확인 (P-D0는 GPU1 — 충돌 없어야)
if nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "PLORA_SELECT_ABORT_GPU0_BUSY"; exit 1
fi

eval_one() { # predfile llm_tag
  $IP $SC/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
    --pred_file $1 --dst $TB/data_multimedia_sub500_eval_$2 --llm $2 > $S/ps_$2.txt 2>&1
  echo -n "  $2: "; grep -hE "link_binary_f1" $S/ps_$2.txt | head -1
}

# C0: P-lora k0 단일 통제
eval_one $TBPRED/tb_dl_0.json plora_c0

# SEL-1: 순수 P-lora 8종 (per-slot prior-MBR, H6 제외)
(cd $SC && $IP tb_select_official.py --tb_dir $TB --domain data_multimedia \
  --ar_tag tb_dl_ --ar_group plora --ar_group_per_slot --no_hm --prior_beta 2.0 \
  --out $TBPRED/tb_plora_sel1.json)
eval_one $TBPRED/tb_plora_sel1.json plora_sel1

# SEL-1 + H6 합류 (P-lora8 + 기존 이종 6 = 14종)
(cd $SC && $IP tb_select_official.py --tb_dir $TB --domain data_multimedia \
  --ar_tag tb_dl_ --ar_group plora --ar_group_per_slot --prior_beta 2.0 \
  --out $TBPRED/tb_plora_h6_sel1.json)
eval_one $TBPRED/tb_plora_h6_sel1.json plora_h6_sel1

# SEL-4: per-slot + 7B Reviewer (GPU0 serve)
kill_gpu0() { for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader); do
  kill -9 $p 2>/dev/null; done; sleep 8; }
CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8150 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8002 --served-model-name base_model --max-model-len 8192 \
  --gpu-memory-utilization 0.85 > $S/vllm_plora_sel4.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8002/v1/models | grep -q base_model && ok=1 && break; sleep 10
done
if [ $ok = 1 ]; then
  (cd $SC && $IP tb_reviewer_select.py --tb_dir $TB --domain data_multimedia \
    --ar_tag tb_dl_ --ar_group plora --ar_group_per_slot --no_hm \
    --endpoint http://localhost:8002/v1 --served base_model --lam 1.0 \
    --out $TBPRED/tb_plora_sel4.json)
  eval_one $TBPRED/tb_plora_sel4.json plora_sel4
else
  echo "PLORA_SEL4_SERVE_FAIL"
fi
kill_gpu0

# 측도 보강 + 회수율 (SELECTOR §7): 다양성 분석 재실행 (P-lora 풀)
$IP $SC/tb_divgen_analyze.py --tb_dir $TB \
  --policy "P-lora=$TBPRED/tb_dl_*.json" \
  --out_prefix $TBPRED/tb_plora_div > $S/plora_div.txt 2>&1 || true
grep -aE "\[P-lora|oracle|이득" $S/plora_div.txt

{ echo "== [P-lora 선별] C0/SEL-1/SEL-1+H6/SEL-4"
  grep -hE "link_binary_f1" $S/ps_plora_c0.txt $S/ps_plora_sel1.txt $S/ps_plora_h6_sel1.txt $S/ps_plora_sel4.txt 2>/dev/null
} >> $S/day13_summary.txt
echo "PLORA_SELECT_DONE $(date)" | tee -a $S/day13_summary.txt
