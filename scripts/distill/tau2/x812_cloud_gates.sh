#!/usr/bin/env bash
# x812_cloud_gates — 착수 전 게이트 G1~G5 (계획서 x812 §4). ⛔통과 못 하면 런 금지.
set -u
BASE="$HOME"; PORT="${1:-8141}"
TB="$BASE/tau2-bench"; REPO="$BASE/boltzmann-attention-pi"; T2="$REPO/scripts/distill/tau2"
PY="$TB/venv/bin/python"; M="Qwen/Qwen3.8-27B-FP8"
FAIL=0
say(){ printf "%-4s %-28s %s\n" "$1" "$2" "$3"; }

echo "########## G1 서빙 모델 id"
ID=$(curl -s -m 10 "http://localhost:$PORT/v1/models" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
[ "$ID" = "$M" ] && say "✅" "G1 모델 id" "$ID" || { say "⛔" "G1 모델 id" "$ID (기대 $M)"; FAIL=1; }

echo "########## G2 KV 예산 (로컬 실측 171,749 보다 커야 80GB)"
KV=$(curl -s -m 10 "http://localhost:$PORT/metrics" | grep -oE 'kv_cache_size_tokens="[0-9]+"' | head -1 | grep -oE '[0-9]+')
CC=$(curl -s -m 10 "http://localhost:$PORT/metrics" | grep -oE 'kv_cache_max_concurrency="[0-9.]+"' | head -1 | grep -oE '[0-9.]+')
if [ -n "${KV:-}" ] && [ "$KV" -gt 171749 ]; then say "✅" "G2 KV" "$KV 토큰 · conc $CC (로컬 171,749 · 1.31)"
else say "⛔" "G2 KV" "${KV:-없음} — 80GB 가 아니거나 util 이 다르다"; FAIL=1; fi

echo "########## G3 툴콜 파싱 스모크 (3 태스크 · nt=1)"
source "$BASE/.openrouter_key"
export PYTHONPATH="src:$T2" T2_MAX_MODEL_LEN=131072
cd "$TB" || exit 1
TAG=x812_g3_smoke; rm -rf "$TB/data/simulations/$TAG"
$PY "$T2/t2_run_gated.py" --gate 0 --domain banking_knowledge --retrieval_config alltools \
  --agent_model "$M" --agent_base "http://localhost:$PORT/v1" \
  --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low \
  --task_ids task_001 task_002 task_005 --num_trials 1 --max_concurrency 3 --max_steps 200 \
  --save_to "$TAG" > "$BASE/logs/${TAG}.log" 2>&1
S=$(grep -c "SALVAGED=" "$BASE/logs/${TAG}.log"); T=$(grep -c '\*\*TRUNC\*\*' "$BASE/logs/${TAG}.log")
[ "$S" = "0" ] && [ "$T" = "0" ] && say "✅" "G3 파싱" "SALVAGED=0 TRUNC=0" || { say "⛔" "G3 파싱" "SALVAGED=$S TRUNC=$T ([[84]] 파서 짝 깨짐)"; FAIL=1; }

echo "########## G4 겹침 대조 — 로컬 4/4 인 001·002·005 를 nt=4 로 (12/12 여야 짝 성립)"
TAG4=x812_g4_pair; rm -rf "$TB/data/simulations/$TAG4"
$PY "$T2/t2_run_gated.py" --gate 0 --domain banking_knowledge --retrieval_config alltools \
  --agent_model "$M" --agent_base "http://localhost:$PORT/v1" \
  --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low \
  --task_ids task_001 task_002 task_005 --num_trials 4 --max_concurrency 4 --max_steps 200 \
  --save_to "$TAG4" > "$BASE/logs/${TAG4}.log" 2>&1
$PY - "$TB/data/simulations/$TAG4/results.json" <<'PY'
import json,sys,collections
r=json.load(open(sys.argv[1])); c=collections.Counter()
for s in r.get("simulations") or []:
    rw=(s.get("reward_info") or {}).get("reward")
    if rw is not None: c[s["task_id"]] += (1 if rw>=1.0 else 0); c[s["task_id"]+"_n"]+=1
ok=True
for t in ("task_001","task_002","task_005"):
    p,n=c.get(t,0),c.get(t+"_n",0); print("   %-10s %d/%d"%(t,p,n))
    if p!=4 or n!=4: ok=False
print("   =>", "✅ 12/12 짝 성립" if ok else "⛔ 어긋남 — base 잔여는 로컬에 남긴다(A/B 만 클라우드)")
sys.exit(0 if ok else 3)
PY
[ $? -ne 0 ] && FAIL=2

echo "########## G5 영속 왕복"
mkdir -p "$REPO/reports/facet_rft_2026/sim_results"
GZ="$REPO/reports/facet_rft_2026/sim_results/${TAG4}.results.json.gz"
gzip -c "$TB/data/simulations/$TAG4/results.json" > "$GZ"
cd "$REPO" && git add -f "$GZ" && git -c user.email=cloud@local -c user.name=cloud commit -q -m "results: $TAG4 (x812 G5)" && git push -q
git ls-files --error-unmatch "$GZ" >/dev/null 2>&1 && say "✅" "G5 영속" "tracked+push 확인" || { say "⛔" "G5 영속" "tracked 아님"; FAIL=1; }

echo
[ "$FAIL" = "0" ] && echo "=== 전체 통과 — base 잔여 투입 가능" \
 || { [ "$FAIL" = "2" ] && echo "=== G4 만 실패 — ⛔base 는 로컬 유지 · A/B 만 여기서" || echo "=== ⛔실패 있음 — 런 금지"; }
exit $FAIL
