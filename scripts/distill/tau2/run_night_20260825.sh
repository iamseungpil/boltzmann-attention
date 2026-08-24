#!/bin/bash
# 밤샘 배치 — 2026-08-25 (사용자 지시: *"전체 모두 다 할려면 얼마나 시간이 필요한가? 밤샘런 준비하라"*)
#
# ## 무엇을 하나 (전부 **무료**: 로컬 vLLM 두 대 · user-sim 호출 0 · 유료 런 0)
#   A. x453 속성 감사 **전수** — 선언 계열 4 · 클래스 71 · 문서 698편. 2 샤드로 GPU 병렬 (~85분)
#      왜: 표적 2계열 감사에서 **현행 16축 중 12축이 그 계열에서 한 번도 관측 안 됨**이 나왔다
#          (`declared_never_seen`). checking 용 표를 다른 계열에 갖다 댄 상태다.
#   B. x526 변환 — 감사의 **검산 통과 인용**을 사실표 형식으로 (결정론·LLM 0·수 초)
#   C. x451 재측정 — 확장 표로 ②범주(계좌 클래스) 격리. 이 축은 지금까지 **판정 자체가 없었다**
#      (x499b 정산: `MODEL_CAN 0 · MODEL_CANNOT 0`)
#   D. x525 n=8 — 074 전사 결손의 팔 순위를 굳힌다(현재 n=4)
#   E. 영속 — git add -f → commit → push → `git ls-files --error-unmatch` 확인([[30]])
#
# ## 이 배치가 **안 하는 것**
#   · 유료 런 0 · 라이브 코드/A2 수정 0 · 커밋되는 것은 **프로브 산출물뿐**
#   · 판정은 아침에 사람이 본다 — 스크립트는 수치를 만들 뿐 결론을 쓰지 않는다
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
TAU=$REPO/scripts/distill/tau2
REP=$REPO/reports/facet_rft_2026
LOG=/home/woori/scratch/x524run
PY=/home/woori/venvs/seka_env/bin/python
mkdir -p "$LOG"
cd "$TAU" || exit 1

say() { echo "[night $(date +%H:%M:%S)] $*"; }

# ── 가드: 유료 라이브 런이 돌면 붙지 않는다 ──────────────────────────────
if pgrep -f "[t]2_launch" >/dev/null; then
  say "REFUSING: 라이브 런이 돌고 있다"; exit 1
fi

# ── 0. 앞 프로브가 끝나기를 기다린다 (최대 60분) ─────────────────────────
say "0. 진행 중인 프로브 대기"
for i in $(seq 1 120); do
  if ! pgrep -f "[x]525_iso_vs_live_shape|[x]451_account_class_iso" >/dev/null; then break; fi
  sleep 30
done
say "0. 대기 종료 (남은 프로세스: $(pgrep -cf '[x]52|[x]451'))"

# ── A. x453 전수 감사 · 2 샤드 병렬 ──────────────────────────────────────
say "A. x453 전수 감사 시작 (shard 0 → 8140 · shard 1 → 8141)"
$PY -u x453_attr_coverage_audit.py --port 8140 --minclasses 5 --shard 0 --of 2 \
    --out "$REP/x453_attr_coverage_full_s0_2026_08_25.json" > "$LOG/x453_s0.log" 2>&1 &
P0=$!
$PY -u x453_attr_coverage_audit.py --port 8141 --minclasses 5 --shard 1 --of 2 \
    --out "$REP/x453_attr_coverage_full_s1_2026_08_25.json" > "$LOG/x453_s1.log" 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
say "A. 완료 (shard0 rc=$R0 · shard1 rc=$R1)"

# ── B. x526 변환 (샤드 둘 + 이미 있는 표적 감사 = 합집합) ────────────────
say "B. 사실표 확장"
BASE="$REP/x430_account_facts_llm_filled.json"
STEP1="$REP/_night_step1.json"
STEP2="$REP/_night_step2.json"
FULL="$REP/x430_account_facts_full_2026_08_25.json"
if [ -s "$REP/x453_attr_coverage_full_s0_2026_08_25.json" ]; then
  $PY -u x526_expand_facts_from_x453.py --audit "$REP/x453_attr_coverage_full_s0_2026_08_25.json" \
      --base "$BASE" --out "$STEP1" 2>&1 | tail -6
else
  say "B. ⚠shard0 산출 없음 — 기존 표를 그대로 승계"; cp "$BASE" "$STEP1"
fi
if [ -s "$REP/x453_attr_coverage_full_s1_2026_08_25.json" ]; then
  $PY -u x526_expand_facts_from_x453.py --audit "$REP/x453_attr_coverage_full_s1_2026_08_25.json" \
      --base "$STEP1" --out "$STEP2" 2>&1 | tail -6
else
  say "B. ⚠shard1 산출 없음"; cp "$STEP1" "$STEP2"
fi
$PY -u x526_expand_facts_from_x453.py --audit "$REP/x453_attr_coverage_targeted_2026_08_24.json" \
    --base "$STEP2" --out "$FULL" 2>&1 | tail -8
say "B. 완료 → $FULL"

# ── C·D. 재측정 두 개를 GPU 하나씩 병렬 ─────────────────────────────────
say "C. x451 ②범주 재측정 (확장 표 · 8141)"
$PY -u x451_account_class_iso.py --port 8141 --arms E_enum,F_facts,D_docs,N_sham \
    --tag full1 --facts "$FULL" > "$LOG/x451_full.log" 2>&1 &
PC=$!
say "D. x525 074 전사 팔 순위 n=8 (8140)"
$PY -u x525_iso_vs_live_shape.py --port 8140 --n 8 --arms A_probe,H_asklast,I_noref \
    --out "$REP/x525e_night_n8_2026_08_25.json" > "$LOG/x525e.log" 2>&1 &
PD=$!
wait $PC; RC=$?
wait $PD; RD=$?
say "C·D 완료 (x451 rc=$RC · x525 rc=$RD)"

# ── E. 영속 ([[30]] tracked 확인까지) ────────────────────────────────────
say "E. 영속"
cd "$REPO" || exit 1
rm -f "$STEP1" "$STEP2"
git add -f reports/facet_rft_2026/x453_attr_coverage_full_s*_2026_08_25.json \
           reports/facet_rft_2026/x430_account_facts_full_2026_08_25.json \
           reports/facet_rft_2026/x451_full1.json \
           reports/facet_rft_2026/x525e_night_n8_2026_08_25.json 2>/dev/null
git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q \
    -m "Night batch: full attribute audit, expanded fact table, category re-measure" || true
git push -q origin facet-rft-2026 || true
git ls-files --error-unmatch reports/facet_rft_2026/x430_account_facts_full_2026_08_25.json >/dev/null 2>&1 \
  && say "E. persisted+tracked OK" || say "E. ⚠tracked 확인 실패 — 아침에 직접 확인하라"

# ── 아침 요약 ────────────────────────────────────────────────────────────
say "=== 아침 요약 ==="
echo "--- x451 (②범주 · 확장 표) ---"; tail -12 "$LOG/x451_full.log" 2>/dev/null
echo "--- x525 (074 · n=8) ---";      tail -14 "$LOG/x525e.log" 2>/dev/null
echo "--- x453 전수 채택 ---"
grep -h "채택(≥5" "$LOG/x453_s0.log" "$LOG/x453_s1.log" 2>/dev/null
say "ALL DONE"
