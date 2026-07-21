#!/bin/bash
# ★48k 단일-GPU 서빙 (2026-07-21 §2bg-3·사용자 지시 "TP=2 말고 단일 GPU 최대로"):
#   가용 KV 풀 10.96GB 실측·fp16 256KB/tok → util 0.95=44.9k가 구 상한. util 0.97 실측 가용 11.91GiB → 최대 48640(12.00GiB 필요한 49152는 0.09GiB 부족·init 실패 실측).
#   yarn 1.5(32768×1.5=49152 정확)·2.0 대비 외삽 비용 최소. CWE 실측(44709~45707) 전부 커버.
#   fp8 KV(sm86 불가)·TP2(IOMMU NCCL) 모두 폐기 이력은 §2bg-2 참조.
CUDA_VISIBLE_DEVICES=1 /home/woori/venvs/tau2_vllm_env/bin/vllm serve Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
  --port 8141 --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 48640 --rope-scaling '{"rope_type":"yarn","factor":1.5,"original_max_position_embeddings":32768}' \
  --enforce-eager --gpu-memory-utilization 0.97
