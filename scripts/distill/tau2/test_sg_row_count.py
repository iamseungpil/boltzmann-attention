# -*- coding: utf-8 -*-
r"""`T2_SG_ROW_COUNT` 래칫 — **실물 궤적·실물 선언**으로 초 단위 검정 (모델 0 · 무료).

## 무엇을 고정하나

1. 술어(`_short_rows`)가 **적게 넘긴 것만** 잡는다 — 같으면·많으면·모르면 침묵.
2. 선언이 두 층(gate·specific)에 **동일**하게 있다([[24]] 한쪽만 고치면 死배선).
3. `return_template_short` 는 `return_template` 을 총액 문장 앞에서 **자른 것**이다 —
   새 도메인 문장 저작 0. 그리고 `{delta_total}` 이 거기 **없다**(그것이 이 레버의 요지다).
4. ★**실물 대조**: t7368 `task_072#s626729` 의 레코드 덤프에서 `type: atm_withdrawal` 을 세면
   Bluest **9** · Light Green **10** 이고, 같은 런 로그의 `operand-size` 는 서브가 각각
   **9 · 9** 를 넘겼다고 적어 뒀다 ⇒ 술어는 Light Green 에서만 서야 한다.
   그 자리가 정확히 `delta_total 5.0 ≠ gold 3.5` 가 나간 자리다.

⚠gold 는 여기 안 들어온다 — 위 3.5 는 **주석**이고 검정은 두 계수의 비교뿐이다([[23]]).

사용: PYTHONPATH=. py -3 test_sg_row_count.py
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_scaffold_get as SG                                        # noqa: E402

TOOL = "get_atm_fee_discrepancies"
CUT = "The signed total of the differences listed above"
FAIL = []


def chk(cond, what, extra=""):
    print("  %-4s %s%s" % ("ok" if cond else "FAIL", what, ("  %s" % extra) if extra else ""))
    if not cond:
        FAIL.append(what)


def tools_of(layer):
    p = os.path.join(HERE, "a2", "banking_knowledge.%s.json" % layer)
    d = json.load(io.open(p, encoding="utf-8"))

    def find(o):
        if isinstance(o, dict):
            if "scaffold_get_tools" in o:
                return o["scaffold_get_tools"]
            for v in o.values():
                r = find(v)
                if r:
                    return r
        if isinstance(o, list):
            for v in o:
                r = find(v)
                if r:
                    return r
        return None
    return find(d) or []


print("① 술어 — 적게 넘긴 것만 잡는다")
chk(SG._short_rows({"kind": "atm_withdrawal", "kind_rows": 10, "sub": 9}) == (1, "atm_withdrawal", 10),
    "10 중 9 → 부족 1")
chk(SG._short_rows({"kind": "atm_withdrawal", "kind_rows": 9, "sub": 9}) is None,
    "9 중 9 → 침묵(부정통제)")
chk(SG._short_rows({"kind": "atm_withdrawal", "kind_rows": 9, "sub": 11}) is None,
    "많이 넘기면 침묵 — 초과는 이 술어의 몫이 아니다")
chk(SG._short_rows({"kind": None, "kind_rows": 0, "sub": 9}) is None,
    "종류 미선언이면 침묵(모르는 것을 주장하지 않는다·[[25]])")
chk(SG._short_rows({"kind": "atm_withdrawal", "kind_rows": 0, "sub": 9}) is None,
    "원천에서 그 종류를 0건 세면 침묵")
chk(SG._short_rows(None) is None, "재료가 없으면 침묵")

print("")
print("② 선언 — 두 층 동일 · 총액 문장이 빠져 있다")
decl = {}
for layer in ("gate", "specific"):
    t = next((x for x in tools_of(layer) if x.get("name") == TOOL), None)
    chk(t is not None, "%s 층에 %s 선언이 있다" % (layer, TOOL))
    if t is None:
        continue
    decl[layer] = t
    chk((t.get("isolate") or {}).get("row_kind") == "atm_withdrawal",
        "%s · isolate.row_kind 가 선언돼 있다" % layer,
        repr((t.get("isolate") or {}).get("row_kind")))
    rts = str(t.get("return_template_short") or "")
    chk(bool(rts), "%s · return_template_short 가 있다" % layer)
    chk("{delta_total}" not in rts,
        "%s · short 판에는 `{delta_total}` 이 **없다**" % layer)
    for f in ("{missing}", "{read}", "{kind}", "{details}"):
        chk(f in rts, "%s · short 판이 %s 를 쓴다" % (layer, f))
    rt = str(t.get("return_template") or "")
    chk("{delta_total}" in rt,
        "%s · 정상 판에는 `{delta_total}` 이 그대로 있다(거동 불변)" % layer)
    chk(CUT in rt, "%s · 정상 판에 총액 문장이 있다" % layer)
    head = rt.split(CUT)[0].rstrip()
    chk(rts.startswith(head),
        "%s · short 판은 정상 판을 **자른 것**이다(새 도메인 문장 저작 0)" % layer)
if len(decl) == 2:
    chk(json.dumps(decl["gate"].get("return_template_short"), ensure_ascii=False)
        == json.dumps(decl["specific"].get("return_template_short"), ensure_ascii=False),
        "두 층의 short 판이 바이트 동일([[24]])")
    chk((decl["gate"].get("isolate") or {}).get("row_kind")
        == (decl["specific"].get("isolate") or {}).get("row_kind"),
        "두 층의 row_kind 가 같다")

print("")
print("③ 실물 대조 — t7368 task_072#s626729 (이 레버가 겨냥한 바로 그 자리)")
TAG = "bank_t7368_hard0_20260827"
SIM = "task_072#s626729"
try:
    import t2_forensic as F
    sim = next((s for s in F.sims(TAG) if F.simtag(s) == SIM), None)
except Exception as e:
    sim = None
    print("  (건너뜀) 궤적을 못 읽었다: %r" % (e,))
if sim is not None:
    kre = re.compile(r"(?im)^[ \t]*type:[ \t]*atm_withdrawal[ \t]*$")
    dumps = [len(kre.findall(str(m.get("content") or "")))
             for m in (sim.get("messages") or [])
             if "Record ID:" in str(m.get("content") or "")
             and kre.search(str(m.get("content") or ""))]
    print("  덤프별 `type: atm_withdrawal` 계수: %s" % dumps)
    chk(sorted(dumps) == [9, 10],
        "두 계좌의 인출 수가 9·10 이다(Bluest·Light Green)", str(sorted(dumps)))
    try:
        log = F.log_text(TAG) or ""
    except Exception:
        log = ""
    subs = [int(m) for m in re.findall(
        r"\[sim=%s\][^\n]*operand-size %s\.transactions: sub=(\d+)" % (re.escape(SIM), TOOL), log)]
    print("  로그가 적은 서브 산출 행 수: %s" % subs)
    chk(bool(subs) and set(subs) == {9},
        "서브는 두 계좌 모두 9 행을 넘겼다 — 10 짜리 계좌에서 한 행이 빠졌다", str(subs))
    if sorted(dumps) == [9, 10] and set(subs) == {9}:
        fires = [SG._short_rows({"kind": "atm_withdrawal", "kind_rows": n, "sub": 9})
                 for n in sorted(dumps)]
        chk(fires[0] is None and fires[1] == (1, "atm_withdrawal", 10),
            "술어가 9 짜리엔 침묵하고 10 짜리에서만 선다", str(fires))

# ── 대칭 축: `_over_rows` (2026-08-28 · t7378 task_074#s361454) ────────────────
#   그 sim 은 이 계좌의 `atm_withdrawal` 16 건보다 3 건 많은 **19 행**을 넘겼고 비교기가
#   그대로 더해 30.00 이 나갔다(옳은 값 14.50). coverage 분모가 **넘어온 행**이라 초과가
#   원리상 안 보였다. 술어는 `_short_rows` 의 대칭이고 같은 레버 아래 산다.
print("")
print("[대칭] _over_rows — **소속**으로 잡는다 (2026-08-29 술어 교체 · C+A)")
# ★갱신 이유(실측): 구판 술어 `sub > kind_rows` 는 밤샘 4런에서 **39회 발화 · 구제 0**,
#   막은 값은 gold 셋(27.00·4.75·3.70)이고 틀린 총액은 0회 막았다. '초과 1' 은 그 태스크의
#   상수이며 같은 조합이 t7378 에도 있었으나 거기서는 총액이 나가 **2/4 통과**했다([[57]]).
#   분모(`type:` 개수)가 상계가 아니기 때문이다. ⇒ 세기 대신 이물/충돌로 판정한다.
chk(SG._over_rows({"kind": "atm_withdrawal", "kind_rows": 16, "sub": 19,
                   "alien": 0, "conflict": 3}) == (3, "atm_withdrawal", 16),
    "같은 id 인데 내용이 다른 3행이면 선다", "실물 t7378 s361454 중복 3행")
chk(SG._over_rows({"kind": "atm_withdrawal", "kind_rows": 17, "sub": 18,
                   "alien": 0, "conflict": 0}) is None,
    "세기만 초과(+1)이고 이물·충돌 0 이면 침묵", "074 의 상수 (18,33,17)")
chk(SG._over_rows({"kind": "atm_withdrawal", "kind_rows": 16, "sub": 19,
                   "alien": 2, "conflict": 0}) == (2, "atm_withdrawal", 16),
    "이 원장에 없는 행 2건이면 선다")
chk(SG._over_rows({"kind": "atm_withdrawal", "kind_rows": 16, "sub": 16}) is None,
    "같으면 침묵")
chk(SG._over_rows({"kind": "atm_withdrawal", "kind_rows": 16, "sub": 9}) is None,
    "적으면 침묵 — 그 축은 _short_rows 몫")
chk(SG._over_rows({"kind": None, "kind_rows": 0, "sub": 19}) is None,
    "종류가 선언 안 됐으면 판정하지 않는다")
chk(SG._over_rows({"kind": "atm_withdrawal", "kind_rows": 0, "sub": 19}) is None,
    "원천에서 그 종류를 못 셌으면 판정하지 않는다")
chk(SG._over_rows(None) is None, "재료가 없으면 침묵")
chk(SG._short_rows({"kind": "atm_withdrawal", "kind_rows": 16, "sub": 19}) is None,
    "두 술어는 겹치지 않는다 (_short_rows 는 초과에 침묵)")

# 선언: `return_template_over` 는 **없어야 한다** (2026-08-29 C · 되돌림)
#   있으면 호출부가 총액 문장을 들어낸 문면으로 갈아타고, 그것이 074 에서 gold 총액을
#   39회 입막음하고 감사 도구 재호출 루프를 만들었다(최악 sim: 크레딧 도구 도달 0회).
#   탐지는 남는다 — 미선언이면 엔진이 `elif _over:` 로 떨어져 로그만 찍는다.
for _lay in sorted(decl):
    _t = decl[_lay]
    chk("return_template_over" not in _t,
        "%s 에 return_template_over 가 없다(되돌림)" % _lay)
    chk(bool(_t.get("_note_row_count_over")),
        "%s 에 되돌림 근거가 선언으로 적혀 있다" % _lay)
    chk("{delta_total}" in (_t.get("return_template") or ""),
        "%s: 기본 반환문은 총액을 그대로 말한다" % _lay)

print("")
print("RESULT: %s" % ("PASS" if not FAIL else "FAIL (%d) %s" % (len(FAIL), FAIL[:3])))
sys.exit(0 if not FAIL else 1)
