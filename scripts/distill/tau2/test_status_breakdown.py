# -*- coding: utf-8 -*-
"""원장 상태별 세기 회귀 — 어느 행이 완료되지 않았는지 (순수함수·GPU 0).

근거 원장: C378. 표적 = task_010. 손님 축자 *"I got four friends to sign up, but I only
received bonuses for two."* 원장 4행 중 둘은 완료·하나는 진행 중·하나는 거절이고 gold 는
**그 거절된 행의 상품을 다시 제출**하는 것이다. 결함 = `row_keys` 가 날짜와 그룹 필드 둘만
선언해 **엔진이 상태를 아예 읽지 않았다**.

실행: py -3 test_status_breakdown.py
"""
import inspect
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_ledger as LG                                          # noqa: E402
from gate_interpreter import load_domain_a2                      # noqa: E402

FAIL = []
HERE = os.path.dirname(os.path.abspath(__file__))


def ok(cond, label, extra=""):
    print(("  OK   " if cond else "  FAIL ") + label + (("  — " + str(extra)) if extra else ""))
    if not cond:
        FAIL.append(label)


# ── 합성 고정물 (도메인 무관) ──────────────────────────────────────────────
SPEC = {"group_field": "g", "status_field": "st",
        "status_text": "of {total}: {breakdown}."}
ROWS = [{"g": "Alpha", "st": "DONE"}, {"g": "Bravo", "st": "DONE"},
        {"g": "Charlie", "st": "PENDING"}, {"g": "Delta", "st": "REFUSED"}]

print("\n§1 상태별로 묶고, 그 상태의 이름들을 함께 말한다")
out = LG.status_breakdown(ROWS, SPEC)
ok("DONE 2 — Alpha, Bravo" in out, "완료 둘이 이름과 함께", out)
ok("REFUSED 1 — Delta" in out, "거절 하나가 **이름으로 짚힌다** (재제출 표적)")
ok("of 4:" in out, "전체 건수도 말한다")

print("\n§2 나눌 것이 없으면 침묵한다 (발화 예산·Δspurious)")
ok(LG.status_breakdown([{"g": "A", "st": "DONE"}, {"g": "B", "st": "DONE"}], SPEC) == "",
   "상태가 한 종류뿐이면 안 말한다")
ok(LG.status_breakdown([], SPEC) == "", "행이 없으면 침묵")
ok(LG.status_breakdown(ROWS, {"group_field": "g", "status_field": "st"}) == "",
   "선언(status_text)이 없으면 침묵")
ok(LG.status_breakdown(ROWS, {"group_field": "g", "status_text": "x{total}{breakdown}"}) == "",
   "상태 필드 선언이 없으면 침묵")

print("\n§3 값이 없는 행은 세지 않는다 (모름을 상태로 바꾸지 않는다·[[25]])")
mixed = ROWS + [{"g": "Echo"}]
out2 = LG.status_breakdown(mixed, SPEC)
ok("Echo" not in out2, "상태가 빠진 행은 어느 그룹에도 안 들어간다")
ok("of 5:" in out2, "전체 건수는 그대로 5 (센 것과 있는 것을 구별한다)", out2[:24])

print("\n§4 엔진에 상태 어휘가 없다 ([[59]]·[[23]])")
src = inspect.getsource(LG.status_breakdown)
live = ("COMPLETE", "IN_PROGRESS", "REJECTED", "NO_PROGRESS", "referral_status")
ok(not [w for w in live if w in src.split('"""')[-1]],
   "함수 본문에 라이브 상태 값·필드명이 하나도 없다",
   [w for w in live if w in src.split('"""')[-1]])

print("\n§5 A2 두 층 + 라이브 spec (死코드 방지·[[24]])")


def grab(fn):
    d = json.load(io.open(os.path.join(HERE, "a2", fn), encoding="utf-8"))
    for x in d.get("ledger_metrics", []):
        if x.get("trigger_tool") == "get_referrals_by_user":
            return x
    return {}


a, b = grab("banking_knowledge.settings.json"), grab("banking_knowledge.gate.json")
for k in ("row_keys", "status_field", "status_text", "_note_status"):
    ok(a.get(k) == b.get(k) and a.get(k) is not None, "%s 가 정본·gate 바이트 동일" % k)
spec = {}
for x in (load_domain_a2("banking_knowledge") or {}).get("ledger_metrics", []):
    if x.get("trigger_tool") == "get_referrals_by_user":
        spec = x
ok(bool(spec.get("status_text")), "병합된 라이브 spec 이 들고 있다")
# ★가장 중요한 배선: 전사 목록에 그 필드가 없으면 모델이 안 뽑고, 그러면 이 기구는
#   조용히 영영 침묵한다(선언은 있는데 죽어 있는 형태·[[24]] 계보).
ok(spec.get("status_field") in (spec.get("row_keys") or []),
   "★status_field 가 row_keys 에 실려 있다 (없으면 모델이 전사하지 않는다)",
   spec.get("row_keys"))
ok("{total}" in spec.get("status_text", "") and "{breakdown}" in spec.get("status_text", ""),
   "자리표시자가 있다")
ok("do not state why" in spec.get("status_text", "").lower(),
   "문구가 **기록은 이유를 말하지 않는다**고 밝힌다 (엔진이 이유를 지어내지 않는다)")

print("\n§6 라이브 형태로 한 번 (010 원장 축자 구조)")
live_rows = [{"referred_account_type": "Bronze Rewards Card", "referral_status": "COMPLETE"},
             {"referred_account_type": "Gold Rewards Card", "referral_status": "COMPLETE"},
             {"referred_account_type": "Silver Rewards Card", "referral_status": "IN_PROGRESS"},
             {"referred_account_type": "Platinum Rewards Card", "referral_status": "REJECTED"}]
got = LG.status_breakdown(live_rows, spec)
ok("REJECTED 1 — Platinum Rewards Card" in got,
   "거절된 행의 상품이 이름으로 나온다 (= 이 태스크의 재제출 표적)")
ok("COMPLETE 2" in got, "이미 지급된 둘이 구별된다 (실패 궤적은 이 중 하나를 재제출했다)")

print("\n§7 창 산수 — 각 기록을 그 앞의 기록들과 맞댄다 (C379·손님의 *왜* 에 답하는 자리)")
WSPEC = dict(SPEC, date_field="d", window_days=9, window_max=2,
             date_formats=["%m/%d/%Y"],
             window_history_text="crowded: {crowded} (max {max} in {days}d)")
LIVE = [{"g": "Bronze", "d": "10/20/2025", "st": "COMPLETE"},
        {"g": "Gold", "d": "10/22/2025", "st": "COMPLETE"},
        {"g": "Platinum", "d": "10/25/2025", "st": "REJECTED"},
        {"g": "Silver", "d": "11/05/2025", "st": "IN_PROGRESS"}]
w = LG.window_history(LIVE, WSPEC)
ok("Platinum" in w, "창 한도에 닿아 있던 기록을 짚는다 (10/25 앞 9일에 2건)", w)
ok("Silver" not in w, "닿지 않은 기록은 안 짚는다 (11/05 앞 9일에 0건)")
ok("Bronze" not in w and "Gold" not in w, "앞이 비어 있던 기록도 안 짚는다")
# ★인과 금지: 엔진은 산수만 말한다
ok("because" not in w.lower() and "rejected" not in w.lower(),
   "★문장이 인과를 말하지 않는다 (상태 낱말도 안 쓴다·[[25]])", w)
ok(LG.window_history(LIVE, dict(WSPEC, window_max=3)) == "",
   "한도가 더 크면 닿는 기록이 없어 침묵한다")
ok(LG.window_history(LIVE[:1], WSPEC) == "", "기록이 하나면 맞댈 것이 없다")
ok(LG.window_history(LIVE, dict(WSPEC, window_history_text=None)) == "",
   "선언이 없으면 침묵")
ok(LG.window_history([{"g": "A"}, {"g": "B"}], WSPEC) == "", "날짜가 없으면 침묵")
live_spec = spec
ok(bool(live_spec.get("window_history_text")), "라이브 spec 이 선언을 들고 있다")
ok(live_spec.get("date_field") in (live_spec.get("row_keys") or []),
   "★date_field 가 row_keys 에 있다 (없으면 모델이 전사 안 해 조용히 죽는다)")
ok("retrieve the document" not in (live_spec.get("status_text") or "").lower(),
   "★상태 문구에서 **검색 지시**를 뺐다 — v010 에서 에이전트가 그 지시대로 상태 낱말"
   "(`referral status IN_PROGRESS REJECTED`)로 검색해 이유를 못 찾았다. 이유를 나르는 "
   "것은 이제 창 산수 문장이다.",
   (live_spec.get("status_text") or "")[-40:])

print("\n" + ("FAIL %d: %s" % (len(FAIL), FAIL) if FAIL else "PASS  (0 실패)"))
sys.exit(1 if FAIL else 0)
