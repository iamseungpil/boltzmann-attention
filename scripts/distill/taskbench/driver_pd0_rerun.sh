#!/bin/bash
# P-D0 clean 재실행 (2026-06-14 부검 교정본). GPU1 (determinism이 GPU0 점유 중).
# 교정: ①steps=full denoise(steps==max_new_tokens=512; 구 run은 512<768 과소-denoise=mask 잔류 붕괴 68%)
#       ②temp 0.2(구 0.8 → 디코드 noise 감소) ③degenerate(<=20ch) 카운터로 붕괴 모니터.
# 목적 = 형식게이트(parse_rate·degen)를 교란 제거 후 재측정. 우선 smoke(n=20 k=2), 통과 시 full N=50로 승격.
set -e
cd /home/woori/workspace_common/boltzmann-attention-pi
git pull --ff-only 2>&1 | tail -1
OUT=/home/woori/scratch/dream_p_d0_v2
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=1 /home/woori/venvs/seka_env/bin/python \
  scripts/distill/taskbench/tb_diffusion_sample.py \
  --data_dir /home/woori/scratch/JARVIS_tb/taskbench/data_multimedia \
  --out_prefix "$OUT/dream" --n 20 --k 2 --temperature 0.2 --max_new_tokens 512 \
  > "$OUT/pd0_v2.log" 2>&1
echo "PD0_V2_DONE $(date)" >> "$OUT/pd0_v2.log"
