#!/usr/bin/env bash
# x812_cloud_engine — vLLM 점화. ⛔플래그는 로컬 실행 프로세스 cmdline 원본 그대로([[84]] 파서 짝).
set -eu
BASE="$HOME"; PORT="${1:-8141}"
setsid "$BASE/venv_vllm/bin/vllm" serve Qwen/Qwen3.8-27B-FP8 \
  --port "$PORT" --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --max-model-len 131072 --gpu-memory-utilization 0.9 \
  --enable-prefix-caching --max-num-seqs 128 \
  </dev/null > "$BASE/logs/vllm_$PORT.log" 2>&1 &
echo "점화 중 (로그: $BASE/logs/vllm_$PORT.log) — 모델 적재에 3~8분"
for i in $(seq 1 120); do
  sleep 10
  ID=$(curl -s -m 5 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4 || true)
  [ -n "${ID:-}" ] && { echo "  ✅ 응답: $ID"; exit 0; }
done
echo "  ⛔ 20분 내 응답 없음 — 로그 확인"
echo "  1차 처방: VLLM_USE_FLASHINFER_SAMPLER=0 (CUDA<12 서버)"
exit 1
