#!/bin/bash
# E-HORIZON + E-THINK 드라이버 (무료·GPU1 단독·리모트). 설계 = THINKING_HORIZON_LEVER_SURVIVAL_DESIGN §4.
#   E-HORIZON: Qwen2.5 사다리 × running-sum {base,verify,detect} → per-step scale-완만 vs verify-급격.
#   E-THINK  : Qwen3 사다리 × eref near-miss B:1,2,4 × {think off,on} → thinking이 참조오류 닫나.
# 실행: setsid bash ehoriz_ethink_driver.sh </dev/null >$LOG 2>&1 &  (진행률 가시·[[30]])
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
TAU=$REPO/scripts/distill/tau2
OUT=$REPO/reports/facet_rft_2026/sim_results
GZ=$OUT/comp_retail_t4.results.json.gz
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
PORT=8141
BASE=http://localhost:$PORT/v1
SLOG=/home/woori/scratch/ehz_serve.log
MARK=/home/woori/scratch/EHZ_ETHINK_DONE
rm -f "$MARK"
mkdir -p "$OUT"

serve() {  # $1=model  → SPID 설정·health 대기
  local M="$1"
  echo "[serve] $M on GPU1:$PORT $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=1 nohup "$VLLM" serve "$M" --port $PORT --served-model-name "$M" \
    --max-model-len 8192 --gpu-memory-utilization 0.90 --disable-log-requests \
    > "$SLOG" 2>&1 &
  SPID=$!
  for i in $(seq 1 90); do
    if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$M"; then
      echo "[serve] ready after ${i}0s"; sleep 3; return 0
    fi
    if ! kill -0 $SPID 2>/dev/null; then echo "[serve] DIED — see $SLOG"; tail -5 "$SLOG"; return 1; fi
    sleep 10
  done
  echo "[serve] TIMEOUT"; return 1
}
stopserve() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 4; }

# ─────────── E-HORIZON (Qwen2.5 사다리 = scale 완만 곡선 + verify 급격) ───────────
HZ_MODELS="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct"
for M in $HZ_MODELS; do
  tag=$(echo "$M" | sed 's#.*/##;s/-Instruct//;s/\.//g' | tr 'A-Z' 'a-z')
  if serve "$M"; then
    echo "[HORIZON] $M $(date +%H:%M:%S)"
    "$PY" "$TAU/eref_horizon.py" --base "$BASE" --model "$M" \
      --arms base,verify,detect --H 30 --K 2 --runs 12 --workers 6 --think off \
      --out "$OUT/ehoriz_${tag}.jsonl" 2>&1 | tail -40
    stopserve
  fi
done

# ─────────── E-THINK (Qwen3 사다리 × near-miss × think off/on) ───────────
TH_MODELS="Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B"
for M in $TH_MODELS; do
  tag=$(echo "$M" | sed 's#.*/##;s/\.//g' | tr 'A-Z' 'a-z')
  if serve "$M"; then
    for TH in off on; do
      mt=500; [ "$TH" = on ] && mt=3000
      echo "[THINK] $M think=$TH $(date +%H:%M:%S)"
      "$PY" "$TAU/eref_probe.py" --gz "$GZ" --base "$BASE" --model "$M" \
        --v2 "B:1,2,4" --n 36 --think "$TH" --max_tokens $mt --workers 6 \
        --out "$OUT/ethink_${tag}_${TH}.jsonl" 2>&1 | tail -20
    done
    stopserve
  fi
done

# ─────────── persist ───────────
cd "$REPO"
git add -f "$OUT"/ehoriz_*.jsonl "$OUT"/ethink_*.jsonl 2>/dev/null
git commit -q -m "E-HORIZON+E-THINK 결과 persist (per-step scale vs verify·thinking near-miss·무료)" 2>&1 | tail -2
git push origin facet-rft-2026 2>&1 | tail -2
touch "$MARK"
echo "[DONE] $(date +%H:%M:%S) marker=$MARK"
