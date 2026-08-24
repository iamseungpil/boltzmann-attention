#!/bin/bash
# 밤샘 배치 3 — 074 기전 이등분 (사용자 지시: *"가르는 실험들 모두 배치에 추가하라"*)
#
# 앞 배치(`run_night_final_20260825.sh`)의 레인 B(B3=x525h)가 끝나면 8140 에서 이어 돈다.
# 8141 은 건드리지 않는다(②범주 전수 감사 진행 중).
#
# ## 무엇을 가르나
#   확정 사실: chk_2 에서 `=== REFERENCE ===` 블록이 있으면 cover 13~14/16 · 없으면 **16/16**(4/4).
#              ⚠단 chk_1·chk_3·chk_4 에서는 유무로 커버리지가 안 갈린다 — 효과는 **한 계좌**다.
#   가설 셋을 한 칸씩:
#     M_reflast    블록을 **원장 뒤**로만       → 돌아오면 원인은 존재가 아니라 **자리**(위치 효과)
#     M_refneutral **키 이름만** 중립으로       → 돌아오면 `account_id` 가 **선택 연산**으로 읽힌 것
#     M_refplain   같은 정보를 **문장 하나**로  → 돌아오면 두 번째 JSON 블록의 **형식 모방**
#   대조군은 이미 있다 — `D_all`(있음·13~14) · `I_noref`(없음·16/16).
#
# ⛔이 배치도 측정만 한다. 이기는 팔이 나와도 라이브 배선은 아침에 사람이 판단한다
#   ([[78]] ②: 이식 대상은 **선언에 있는 텍스트**여야 한다 — 프로브가 지은 문면이 이기면
#    그건 이식 후보가 아니라 A2 저작 항목이다).
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
TAU=$REPO/scripts/distill/tau2
REP=$REPO/reports/facet_rft_2026
LOG=/home/woori/scratch/x524run
PY=/home/woori/venvs/seka_env/bin/python
mkdir -p "$LOG"
cd "$TAU" || exit 1
say() { echo "[night3 $(date +%H:%M:%S)] $*"; }

say "0. 앞 배치 레인 B(x525h) 종료 대기"
for i in $(seq 1 720); do
  [ -s "$REP/x525h_control_2026_08_25.json" ] && break
  pgrep -f "[r]un_night_final_20260825" >/dev/null || { say "0. 앞 배치가 사라졌다 — 그대로 진행"; break; }
  sleep 30
done
for i in $(seq 1 120); do
  pgrep -f "[x]525_iso_vs_live_shape|[x]527_016" >/dev/null || break
  sleep 30
done
say "0. 8140 확보"

say "1. x525 기전 이등분 (M_reflast·M_refneutral·M_refplain) n=6"
timeout 14400 $PY -u x525_iso_vs_live_shape.py --port 8140 --n 6 \
  --arms M_reflast,M_refneutral,M_refplain \
  --out "$REP/x525i_refmech_2026_08_25.json" > "$LOG/x525i.log" 2>&1
say "1. rc=$?"

cd "$REPO" || exit 1
git add -f reports/facet_rft_2026/x525i_refmech_2026_08_25.json 2>/dev/null
git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q \
  -m "Bisect why the reference block costs rows: position, key name, or format" || true
git push -q origin facet-rft-2026 || true
say "=== 아침 요약 (배치 3) ==="
tail -22 "$LOG/x525i.log" 2>/dev/null
say "ALL DONE"
