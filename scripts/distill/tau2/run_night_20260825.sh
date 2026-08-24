#!/bin/bash
# 밤샘 배치 — 2026-08-25
#   사용자 지시 ①*"밤샘런 준비하라"* ②*"74 태스크도 끝내야 하니, **한개 gpu 로만 실험하라**"*
#   ⇒ 감사(②범주)는 **8141 한 대만** 쓰고, **8140 은 074 전용 레인**으로 비워 둔다.
#
# ## 레인 A — 8141 · ②범주 (계좌 클래스)
#   A1 x453 속성 감사 **전수**(선언 계열 4 · 클래스 71 · 문서 698편) — 단일 GPU ~170분
#      왜: 표적 2계열 감사에서 **현행 16축 중 12축이 그 계열에서 한 번도 관측 안 됨**
#          (`declared_never_seen`) — checking 용 표를 다른 계열에 갖다 댄 상태다
#   A2 x526 변환(결정론·LLM 0) — 검산 통과 인용을 사실표로
#   A3 x451 재측정 — 이 축은 지금까지 판정 자체가 없었다(x499b: MODEL_CAN 0 · MODEL_CANNOT 0)
#
# ## 레인 B — 8140 · 074 (전사 결손)
#   B1 x525 n=8 · A_probe·H_asklast·I_noref — 이긴 팔과 마지막 두 칸을 n=4 → n=8 로 굳힌다
#   B2 x525 n=8 · B_fmt·C_toolmsg·D_all·E_oneline·F_order·G_norow — **팔 순위표 완성**
#      (죽은 가설 다섯을 같은 n 으로 나란히 놓아야 표가 인용 가능해진다)
#
# ## 안 하는 것
#   유료 런 0 · 라이브 코드/A2 수정 0 · 결론 쓰기 0(스크립트는 수치만 만든다·판정은 아침에 사람이)
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
TAU=$REPO/scripts/distill/tau2
REP=$REPO/reports/facet_rft_2026
LOG=/home/woori/scratch/x524run
PY=/home/woori/venvs/seka_env/bin/python
mkdir -p "$LOG"
cd "$TAU" || exit 1

say() { echo "[night $(date +%H:%M:%S)] $*"; }

if pgrep -f "[t]2_launch" >/dev/null; then
  say "REFUSING: 라이브 런이 돌고 있다"; exit 1
fi

# ── 0. 앞 프로브 대기 (최대 60분) ────────────────────────────────────────
say "0. 진행 중인 프로브 대기"
for i in $(seq 1 120); do
  pgrep -f "[x]525_iso_vs_live_shape|[x]451_account_class_iso" >/dev/null || break
  sleep 30
done
say "0. 대기 종료"

# ── 레인 B (8140 · 074) — 먼저 띄운다. 레인 A 와 완전 독립 ───────────────
(
  say "B1. x525 n=8 (A_probe·H_asklast·I_noref) → 8140"
  $PY -u x525_iso_vs_live_shape.py --port 8140 --n 8 --arms A_probe,H_asklast,I_noref \
      --out "$REP/x525e_night_n8_2026_08_25.json" > "$LOG/x525e.log" 2>&1
  say "B1. 완료 rc=$?"
  say "B2. x525 n=8 (죽은 가설 다섯 + G) → 8140"
  $PY -u x525_iso_vs_live_shape.py --port 8140 --n 8 \
      --arms B_fmt,C_toolmsg,D_all,E_oneline,F_order,G_norow \
      --out "$REP/x525f_night_rank_2026_08_25.json" > "$LOG/x525f.log" 2>&1
  say "B2. 완료 rc=$?"
) > "$LOG/night_laneB.log" 2>&1 &
PB=$!

# ── 레인 A (8141 · ②범주) ───────────────────────────────────────────────
say "A1. x453 전수 감사 (단일 GPU 8141 · 클래스 71 · 문서 698)"
$PY -u x453_attr_coverage_audit.py --port 8141 --minclasses 5 \
    --out "$REP/x453_attr_coverage_full_2026_08_25.json" > "$LOG/x453_full.log" 2>&1
say "A1. 완료 rc=$?"

say "A2. 사실표 확장"
BASE="$REP/x430_account_facts_llm_filled.json"
STEP1="$REP/_night_step1.json"
FULL="$REP/x430_account_facts_full_2026_08_25.json"
if [ -s "$REP/x453_attr_coverage_full_2026_08_25.json" ]; then
  $PY -u x526_expand_facts_from_x453.py --audit "$REP/x453_attr_coverage_full_2026_08_25.json" \
      --base "$BASE" --out "$STEP1" 2>&1 | tail -8
else
  say "A2. ⚠전수 감사 산출 없음 — 기존 표 승계"; cp "$BASE" "$STEP1"
fi
$PY -u x526_expand_facts_from_x453.py --audit "$REP/x453_attr_coverage_targeted_2026_08_24.json" \
    --base "$STEP1" --out "$FULL" 2>&1 | tail -8
say "A2. 완료 → $FULL"

say "A3. x451 ②범주 재측정 (확장 표 · 8141)"
$PY -u x451_account_class_iso.py --port 8141 --arms E_enum,F_facts,D_docs,N_sham \
    --tag full1 --facts "$FULL" > "$LOG/x451_full.log" 2>&1
say "A3. 완료 rc=$?"

wait $PB
say "레인 B 종료"

# ── 영속 ([[30]] tracked 확인까지) ──────────────────────────────────────
say "E. 영속"
cd "$REPO" || exit 1
rm -f "$STEP1"
git add -f reports/facet_rft_2026/x453_attr_coverage_full_2026_08_25.json \
           reports/facet_rft_2026/x430_account_facts_full_2026_08_25.json \
           reports/facet_rft_2026/x451_full1.json \
           reports/facet_rft_2026/x525e_night_n8_2026_08_25.json \
           reports/facet_rft_2026/x525f_night_rank_2026_08_25.json 2>/dev/null
git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q \
    -m "Night batch: audit every family, expand the table, rank the transcription arms" || true
git push -q origin facet-rft-2026 || true
git ls-files --error-unmatch reports/facet_rft_2026/x430_account_facts_full_2026_08_25.json >/dev/null 2>&1 \
  && say "E. persisted+tracked OK" || say "E. ⚠tracked 확인 실패 — 아침에 직접 확인하라"

say "=== 아침 요약 ==="
echo "--- A3 x451 (②범주 · 확장 표) ---"; tail -12 "$LOG/x451_full.log" 2>/dev/null
echo "--- B1 x525 n=8 (이긴 팔 + 마지막 두 칸) ---"; tail -16 "$LOG/x525e.log" 2>/dev/null
echo "--- B2 x525 n=8 (팔 순위표) ---"; tail -22 "$LOG/x525f.log" 2>/dev/null
echo "--- A1 x453 전수 채택 ---"; grep -h "채택(≥5" "$LOG/x453_full.log" 2>/dev/null
say "ALL DONE"
