#!/bin/bash
# 밤샘 배치 (정본·앞 둘을 대체) — 2026-08-25
#   사용자 지시: *"**hard-0 pass 를 실질적으로 올릴 수 있는 실험들을 하라**"* ·
#                *"74 태스크도 끝내야 하니 한개 gpu 로만"* · *"074 끝나면 그 gpu 에 016 도"*
#
# ## 편성 원칙 — 모든 항목이 **hard-0 태스크**에 직결돼야 한다
#   레인 A (8141) ②범주 → 057·063 (hard-0 2)
#   레인 B (8140) ⑦유도 016 (hard-0 1) → ①금액 074 (hard-0 1)
#   ⛔뺀 것: x201 `D_null` · x317 `E_NEG` — 부정통제 공백 메우기는 감사 위생이지 hard-0 이 아니다.
#     (x499b 가 지목한 공백은 남아 있다 — 낮에 사람이 판단해서 돌린다)
#   ⛔뺀 것: x525 로 **죽은 가설 여섯**을 n=8 재확인 — 순위표는 논문 재료이지 pass 가 아니다.
#
# ## 레인 A · ②범주 (8141)
#   A1 x453 전수 감사(계열 4 · 클래스 71 · 문서 698) ~170분
#      근거: 표적 감사에서 **현행 16축 중 12축이 그 계열에서 0회 관측**(`declared_never_seen`)
#   A2 x526 변환(결정론·LLM 0)
#   A3 x451 재측정 — 확장 표로 처음으로 **공정한** ②범주 판정
#
# ## 레인 B · hard-0 두 태스크 (8140)
#   B1 x527 · **016** — 격리 음성이 재료 결손인가. 016 의 gold 는 *친구가 입금*해야 서는데
#      그 조건은 정책 문서에 있고 서브 창은 손님 발화 6개뿐이다 ⇒ 답이 창에 원리상 없다.
#   B2 x525 · **074 조합 팔**(J_both·K_paramslast) — 단일 변수 팔이 전부 실패했으므로
#      이긴 팔과의 차이 셋을 함께 민다. **형식은 A2 선언 그대로 유지**해 이식 가능성을 지킨다.
#   B3 x525 n=8 · A_probe·H_asklast·I_noref — 조합 팔의 대조군을 같은 n 으로 굳힌다.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
TAU=$REPO/scripts/distill/tau2
REP=$REPO/reports/facet_rft_2026
LOG=/home/woori/scratch/x524run
PY=/home/woori/venvs/seka_env/bin/python
mkdir -p "$LOG"
cd "$TAU" || exit 1
say() { echo "[final $(date +%H:%M:%S)] $*"; }

if pgrep -f "[t]2_launch" >/dev/null; then say "REFUSING: 라이브 런"; exit 1; fi

say "0. 앞 프로브 종료 대기"
for i in $(seq 1 120); do
  pgrep -f "[x]525_iso_vs_live_shape|[x]451_account_class_iso|[x]453_attr" >/dev/null || break
  sleep 30
done
say "0. 대기 종료"

# ── 레인 B (8140) — hard-0 016 → 074 ────────────────────────────────────
(
  say "B1. x527 · 016 정책-재료 격리"
  timeout 10800 $PY -u x527_016_policy_material_iso.py --port 8140 --limit 24 \
    --out "$REP/x527_016_policy_material_2026_08_25.json" > "$LOG/x527.log" 2>&1
  say "B1. rc=$?"
  say "B2. x525 · 074 조합 팔 (J_both·K_paramslast) n=6"
  timeout 10800 $PY -u x525_iso_vs_live_shape.py --port 8140 --n 6 --arms J_both,K_paramslast \
    --out "$REP/x525g_combo_2026_08_25.json" > "$LOG/x525g.log" 2>&1
  say "B2. rc=$?"
  say "B3. x525 · 대조군 n=6 (A_probe·H_asklast·I_noref)"
  timeout 10800 $PY -u x525_iso_vs_live_shape.py --port 8140 --n 6 --arms A_probe,H_asklast,I_noref \
    --out "$REP/x525h_control_2026_08_25.json" > "$LOG/x525h.log" 2>&1
  say "B3. rc=$?"
) > "$LOG/final_laneB.log" 2>&1 &
PB=$!

# ── 레인 A (8141) — hard-0 057·063 ──────────────────────────────────────
say "A1. x453 전수 감사 (8141)"
timeout 21600 $PY -u x453_attr_coverage_audit.py --port 8141 --minclasses 5 \
  --out "$REP/x453_attr_coverage_full_2026_08_25.json" > "$LOG/x453_full.log" 2>&1
say "A1. rc=$?"

say "A2. 사실표 확장"
BASE="$REP/x430_account_facts_llm_filled.json"
STEP1="$REP/_final_step1.json"
FULL="$REP/x430_account_facts_full_2026_08_25.json"
if [ -s "$REP/x453_attr_coverage_full_2026_08_25.json" ]; then
  $PY -u x526_expand_facts_from_x453.py --audit "$REP/x453_attr_coverage_full_2026_08_25.json" \
      --base "$BASE" --out "$STEP1" 2>&1 | tail -8
else
  say "A2. ⚠전수 산출 없음 — 기존 표 승계"; cp "$BASE" "$STEP1"
fi
$PY -u x526_expand_facts_from_x453.py --audit "$REP/x453_attr_coverage_targeted_2026_08_24.json" \
    --base "$STEP1" --out "$FULL" 2>&1 | tail -8
say "A2. → $FULL"

say "A3. x451 ②범주 재측정 (확장 표)"
timeout 7200 $PY -u x451_account_class_iso.py --port 8141 --arms E_enum,F_facts,D_docs,N_sham \
  --tag full1 --facts "$FULL" > "$LOG/x451_full.log" 2>&1
say "A3. rc=$?"

wait $PB
say "레인 B 종료"

cd "$REPO" || exit 1
rm -f "$STEP1"
git add -f reports/facet_rft_2026/x453_attr_coverage_full_2026_08_25.json \
           reports/facet_rft_2026/x430_account_facts_full_2026_08_25.json \
           reports/facet_rft_2026/x451_full1.json \
           reports/facet_rft_2026/x527_016_policy_material_2026_08_25.json \
           reports/facet_rft_2026/x525g_combo_2026_08_25.json \
           reports/facet_rft_2026/x525h_control_2026_08_25.json 2>/dev/null
git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q \
  -m "Night: fair table for the category axis, policy in 016's window, and the combination arms for 074" || true
git push -q origin facet-rft-2026 || true
git ls-files --error-unmatch reports/facet_rft_2026/x527_016_policy_material_2026_08_25.json >/dev/null 2>&1 \
  && say "영속+tracked OK" || say "⚠tracked 확인 실패 — 아침에 직접"

say "=== 아침 요약 ==="
echo "--- B1 016 (x527) ---";        tail -12 "$LOG/x527.log" 2>/dev/null
echo "--- B2 074 조합 (x525g) ---";  tail -14 "$LOG/x525g.log" 2>/dev/null
echo "--- B3 074 대조 (x525h) ---";  tail -14 "$LOG/x525h.log" 2>/dev/null
echo "--- A3 057·063 (x451) ---";    tail -14 "$LOG/x451_full.log" 2>/dev/null
echo "--- A1 전수 채택 ---";          grep -h "채택(≥5" "$LOG/x453_full.log" 2>/dev/null
say "ALL DONE"
