#!/bin/bash
# E-THINK 재실행(파싱수정·max_tokens↑) + E-HORIZON inject arm(sd/sc 결정적). 무료·GPU1.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
TAU=$REPO/scripts/distill/tau2
OUT=$REPO/reports/facet_rft_2026/sim_results
GZ=$OUT/comp_retail_t4.results.json.gz
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
PORT=8141; BASE=http://localhost:$PORT/v1
SLOG=/home/woori/scratch/et2_serve.log
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

# ── E-THINK 재실행 (Qwen3 × near-miss × think on·파싱수정·max_tokens 12000) ──
for M in Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B; do
  tag=$(echo "$M" | sed 's#.*/##;s/\.//g' | tr 'A-Z' 'a-z')
  if serve "$M"; then
    echo "[THINK2] $M think=on(fixed) $(date +%H:%M:%S)"
    "$PY" "$TAU/eref_probe.py" --gz "$GZ" --base "$BASE" --model "$M" \
      --v2 "B:1,2,4" --n 36 --think on --max_tokens 12000 --workers 4 \
      --out "$OUT/ethink2_${tag}_on.jsonl" 2>&1 | tail -8
    stopserve
  fi
done

# ── E-HORIZON inject arm (Qwen2.5 {7B,14B} × base,verify,detect,inject) ──
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
git add -f "$OUT"/ethink2_*.jsonl "$OUT"/ehoriz_inject_*.jsonl 2>/dev/null
git commit -q -m "E-THINK2(파싱수정) + E-HORIZON inject arm 결과 persist (무료)" 2>&1 | tail -1
git push origin facet-rft-2026 2>&1 | tail -1
touch "$MARK"
echo "[DONE] $(date +%H:%M:%S)"
