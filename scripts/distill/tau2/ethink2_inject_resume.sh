#!/bin/bash
# RESUME: 14B THINK2(회수) + E-HORIZON inject arm 7B/14B. 무료·GPU1.
# 원 드라이버(ethink2_inject_driver.sh)가 14B THINK2 launch 직후 죽음(setsid 없이 부모 death 추정).
# 1.7B/4B/8B THINK2 jsonl은 이미 존재(skip). Qwen3-14B 서버는 8141에 idle로 생존 → 직접 프로브.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
TAU=$REPO/scripts/distill/tau2
OUT=$REPO/reports/facet_rft_2026/sim_results
GZ=$OUT/comp_retail_t4.results.json.gz
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
PORT=8141; BASE=http://localhost:$PORT/v1
SLOG=/home/woori/scratch/et2b_serve.log
MARK=/home/woori/scratch/ET2_INJECT_DONE
rm -f "$MARK"

serve() {
  local M="$1"
  echo "[serve] $M $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=1 nohup "$VLLM" serve "$M" --port $PORT --served-model-name "$M" \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --disable-log-requests > "$SLOG" 2>&1 &
  SPID=$!
  for i in $(seq 1 90); do
    curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$M" && { echo "[serve] ready ${i}0s"; sleep 3; return 0; }
    kill -0 $SPID 2>/dev/null || { echo "[serve] DIED"; tail -5 "$SLOG"; return 1; }
    sleep 10
  done
  echo "[serve] TIMEOUT"; return 1
}
stopserve() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 4; }

# ── Step 1: 14B THINK2 회수 (이미 떠 있는 Qwen3-14B 서버 재사용·재-serve 안 함) ──
if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "Qwen/Qwen3-14B"; then
  echo "[THINK2-14B resume] 서버 생존 확인·직접 프로브 $(date +%H:%M:%S)"
  "$PY" "$TAU/eref_probe.py" --gz "$GZ" --base "$BASE" --model Qwen/Qwen3-14B \
    --v2 "B:1,2,4" --n 36 --think on --max_tokens 12000 --workers 4 \
    --out "$OUT/ethink2_qwen3-14b_on.jsonl" 2>&1 | tail -8
else
  echo "[THINK2-14B] 서버 없음 → 재-serve"
  if serve Qwen/Qwen3-14B; then
    "$PY" "$TAU/eref_probe.py" --gz "$GZ" --base "$BASE" --model Qwen/Qwen3-14B \
      --v2 "B:1,2,4" --n 36 --think on --max_tokens 12000 --workers 4 \
      --out "$OUT/ethink2_qwen3-14b_on.jsonl" 2>&1 | tail -8
    stopserve
  fi
fi

# 포트 정리 (standing Qwen3-14B 서버 종료·GPU1 8141만·32B/8140 불가침)
echo "[cleanup] free port $PORT $(date +%H:%M:%S)"
fuser -k ${PORT}/tcp 2>/dev/null; sleep 8

# ── Step 2: E-HORIZON inject arm (Qwen2.5 {7B,14B} × base,verify,detect,inject) ──
for M in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct; do
  tag=$(echo "$M" | sed 's#.*/##;s/-Instruct//;s/\.//g' | tr 'A-Z' 'a-z')
  if serve "$M"; then
    echo "[INJECT] $M $(date +%H:%M:%S)"
    "$PY" "$TAU/eref_horizon.py" --base "$BASE" --model "$M" \
      --arms base,verify,detect,inject --H 30 --K 2 --runs 16 --workers 6 --think off \
      --out "$OUT/ehoriz_inject_${tag}.jsonl" 2>&1 | tail -12
    stopserve
  fi
done

cd "$REPO"
git pull --ff-only origin facet-rft-2026 2>&1 | tail -1
git add -f "$OUT"/ethink2_qwen3-14b_on.jsonl "$OUT"/ehoriz_inject_qwen25-7b.jsonl "$OUT"/ehoriz_inject_qwen25-14b.jsonl 2>/dev/null
git commit -q -m "E-THINK2 14B 회수 + E-HORIZON inject arm 7B/14B persist (무료)" 2>&1 | tail -1
git push origin facet-rft-2026 2>&1 | tail -1
touch "$MARK"
echo "[DONE] $(date +%H:%M:%S)"
