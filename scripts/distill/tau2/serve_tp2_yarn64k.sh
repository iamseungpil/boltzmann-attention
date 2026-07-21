#!/bin/bash
# ★TP=2 단일 서버 64k (2026-07-21 §2bg-2): fp8 KV가 Ampere(sm86)서 triton fp8e4nv 미지원으로
#   ★NCCL_P2P_DISABLE=1 (2026-07-21): AMD-Vi IOMMU가 GPU간 P2P 차단(dmesg IO_PAGE_FAULT 실측)→NCCL 초기화 무한 행. SHM 폴백으로 우회.
#   엔진-사망(첫 요청·스모크 실측) → fp8 폐기. 대신 TP=2로 가중치 분할(~17GB/GPU)하면 fp16 KV
#   그대로 KV 풀 ~25GB/GPU = 64k 시퀀스 다중 수용·수치 열화 0. 토폴로지 변경: 단일 엔드포인트
#   8141(두 arm 공유·vllm 배칭). yarn 2.0=65536 커버.
NCCL_P2P_DISABLE=1 /home/woori/venvs/tau2_vllm_env/bin/vllm serve Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
  --port 8141 --enable-auto-tool-choice --tool-call-parser hermes \
  --tensor-parallel-size 2 \
  --max-model-len 65536 --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
  --enforce-eager --gpu-memory-utilization 0.90
