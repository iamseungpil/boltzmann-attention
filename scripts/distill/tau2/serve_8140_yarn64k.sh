#!/bin/bash
# ★64k 서빙 (2026-07-21 §2bg): CWE 3건이 44709~45707tok=구 상한(44672) 근소 초과 — 상한을 KV-fp8로 확장.
#   산술: fp16 KV ~256KB/tok(64L×8KVh×128d×2×2B)→44672≈11.2GB가 A6000 49GB 상한이었음.
#   fp8_e5m2 KV(~128KB/tok)→65536≈8.4GB<구 풀. yarn 1.375→2.0(32768×2=65536 커버).
#   ⚠️조건 변경 = 새 arm(rall6+)으로만 비교·구 결과와 혼합 금지. 품질(fp8 KV·yarn↑)은 023/095 스모크 무회귀로 검증.
CUDA_VISIBLE_DEVICES=0 /home/woori/venvs/tau2_vllm_env/bin/vllm serve Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
  --port 8140 --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 65536 --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
  --kv-cache-dtype fp8_e5m2 \
  --enforce-eager --gpu-memory-utilization 0.95
