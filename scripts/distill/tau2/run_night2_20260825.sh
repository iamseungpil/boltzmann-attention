#!/bin/bash
# 밤샘 배치 2 — 8140 레인 후속 (사용자 지시: *"074 끝나면 그 gpu 에 016 도 실험하라"* ·
#   *"원래 계획에서 남은 실험이 있나? 내일 아침까지 돌릴 실험들을 배치로 준비하라"*)
#
# 앞 배치(`run_night_20260825.sh`)의 레인 B(x525 n=8 두 판)가 끝나면 이어서 8140 에서 돈다.
# 8141 은 건드리지 않는다 — 거기서 ②범주 전수 감사가 돌고 있다.
#
# ## 1. x527 — 016 격리 음성이 **재료 결손**인가 (hard-0 · x514 rank-1)
#   x516(후보집합)·x517(질문 프레임)이 gold 0/39 를 냈고 큐는 *"⑦유도 경로 없음"* 으로 적었다.
#   그런데 016 의 gold 는 **친구가 입금**해야 자격이 서는 것이고 그 조건은 **정책 문서**에 있다.
#   `formalize_intent_tool` 이 보는 것은 손님 발화 6개뿐 ⇒ 답에 필요한 사실이 창에 **원리상 없다**.
#   팔: A_asis(재현) · B_policy(그 sim 이 받은 자격조건 축자) · C_neutral(중립 물음) · N_sham(부정통제)
#
# ## 2. x201 — `D_null` 부정통제 **기록** (x499b 가 지목한 공백)
#   원장 축자: 코드엔 있으나 **수치 미기록** ⇒ 그 프로브의 능력 판정을 인용할 수 없다.
#
# ## 3. x317 — `E_NEG` **미실행분** (사전등록만 돼 있고 한 번도 안 돌았다)
#
# ⛔안 하는 것: x85 는 *프로브 수리*(열거에 gold 6/12)가 선행이라 **오늘 밤엔 안 돌린다** —
#   고치지 않고 돌리면 또 해석 불가 수치가 하나 더 쌓인다(원장 gap G5).
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
TAU=$REPO/scripts/distill/tau2
REP=$REPO/reports/facet_rft_2026
LOG=/home/woori/scratch/x524run
PY=/home/woori/venvs/seka_env/bin/python
mkdir -p "$LOG"
cd "$TAU" || exit 1
say() { echo "[night2 $(date +%H:%M:%S)] $*"; }

# ── 0. 앞 레인 B 대기 (최대 5시간) ───────────────────────────────────────
say "0. x525 (레인 B) 종료 대기"
for i in $(seq 1 600); do
  pgrep -f "[x]525_iso_vs_live_shape" >/dev/null || break
  sleep 30
done
say "0. 대기 종료 — 8140 확보"

# ── 1. x527 · 016 ────────────────────────────────────────────────────────
say "1. x527 016 정책-재료 격리 (8140)"
timeout 9000 $PY -u x527_016_policy_material_iso.py --port 8140 --limit 24 \
  --out "$REP/x527_016_policy_material_2026_08_25.json" > "$LOG/x527.log" 2>&1
say "1. 완료 rc=$?"

# ── 2. x201 · D_null 기록 ───────────────────────────────────────────────
say "2. x201 (D_null 부정통제 기록·8140)"
timeout 5400 $PY -u x201_type_axis.py --port 8140 > "$LOG/x201_night.log" 2>&1
say "2. 완료 rc=$?"

# ── 3. x317 · E_NEG 실행 ────────────────────────────────────────────────
say "3. x317 (E_NEG 미실행분·8140)"
timeout 5400 $PY -u x317_docgroup_route_iso.py --port 8140 > "$LOG/x317_night.log" 2>&1
say "3. 완료 rc=$?"

# ── 영속 ────────────────────────────────────────────────────────────────
say "E. 영속"
cd "$REPO" || exit 1
git add -f reports/facet_rft_2026/x527_016_policy_material_2026_08_25.json 2>/dev/null
git add -f reports/facet_rft_2026/x201*.json reports/facet_rft_2026/x317*.json 2>/dev/null
git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q \
    -m "Night batch 2: does 016 close when the policy is in the window, plus two missing negative controls" || true
git push -q origin facet-rft-2026 || true
say "=== 아침 요약 (배치 2) ==="
echo "--- x527 016 ---"; tail -14 "$LOG/x527.log" 2>/dev/null
echo "--- x201 D_null ---"; tail -8 "$LOG/x201_night.log" 2>/dev/null
echo "--- x317 E_NEG ---"; tail -8 "$LOG/x317_night.log" 2>/dev/null
say "ALL DONE"
