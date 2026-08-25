# -*- coding: utf-8 -*-
r"""`T2_WRITE_ARG_TYPE` · `T2_RULE_AT_WRITE` 래칫 — **선언만 읽는가 · 실제로 도는가** (2026-08-25).

## 무엇이 있었나 (t7354 라이브 전수)

도구가 `(boolean)` 으로 선언한 인자에 모델이 **문자열 `"Yes"`/`"No"`** 를 보낸다:

    085  written_statement_provided='Yes' · police_report_filed='No' · card_in_possession='Yes' …
         → 접수된 분쟁 **전건**(grpA1 t0 4 · grpB2 t0 4 · t1 4)
    040  gold 8건을 **축자 그대로 접수**하고도 `contacted_merchant` gold=True↔got='Yes' ·
         `eligible_for_provisional_credit` gold=False↔got='Yes' 로 **8/8** 어긋나 `db_match=false`

env 가 문자열을 받아 저장하므로 **호출은 성공하고 채점만 조용히 실패한다**. 의미는 모델이 이미
맞혔다(Yes↔True) — 엔진은 값을 바꾸지 않고 **선언된 타입만** 알린다([[62]]③④).

## 이 검정이 지키는 것

  ① 이름 목록은 **A2 선언**에서만 온다 — 엔진에 도메인 낱말 0([[05]]).
  ② 판정은 `isinstance(v, bool)` 하나 — 변환도 선택도 하지 않는다.
  ③ 기본 OFF · 도구별 sim 당 1회.
  ④ **OFF 에서도 관측 마커가 찍힌다** — 이 귀속의 반증 경로를 로그로 남긴다([[77]]③).
  ⑤ `T2_RULE_AT_WRITE` 는 **검색기가 아니다** — 선언된 문장만 읽는다(초판 검색기가 검산에서
     엉뚱한 문장을 집어 폐기한 자리다).
  ⑥ 두 함수가 **실제로 돈다**(死배선 아님 — 오늘 V2 가 그 교훈).
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

SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
FAIL = []


def chk(cond, what, got=""):
    print(("  OK  " if cond else "  XX  ") + what + (("  -- %s" % (got,)) if got else ""))
    if not cond:
        FAIL.append(what)


print("[1] 엔진에 도메인 낱말 0 · 판정은 isinstance 하나")
m = re.search(r'if en_fb is None and _sp\.get\("booleans"\):(.*?)if _sp\.get\("values"\):',
              SRC, re.S)
chk(m is not None, "타입 검사 갈래가 있다")
B = m.group(1) if m else ""
chk("isinstance(_ia.get(_bk), bool)" in B, "판정은 `isinstance(v, bool)` 하나다")
code = "\n".join(ln.split("#", 1)[0] for ln in B.splitlines())
DOM = ("dispute", "debit", "card", "police", "merchant", "possession", "statement", "credit")
hits = [w for w in DOM if w in code.lower()]
chk(not hits, "엔진 실행부에 도메인 낱말이 없다", ",".join(hits) or "없음")
chk('_sp["booleans"]' in B, "이름 목록은 **선언**에서 읽는다")
chk("bool(" not in code and "== 'Yes'" not in code and '== "Yes"' not in code,
    "엔진이 값을 **변환하지 않는다**")

print("[2] 기본 OFF · 상한 · 반증 경로")
chk('os.environ.get("T2_WRITE_ARG_TYPE") == "1"' in SRC, "플래그로만 켜진다")
chk("_t2_argtype_deny" in SRC, "도구별 sim 당 1회 원장이 있다")
chk("[T2_WRITE_ARG_TYPE] 관측(OFF)" in SRC, "OFF 에서도 관측 마커가 찍힌다([[77]]③ 반증 경로)")

print("[3] RULE_AT_WRITE 는 검색기가 아니다")
r = re.search(r"\ndef _declared_rules_for\(wc, a2\):(.*?)\n\n\n", SRC, re.S)
chk(r is not None, "선언 읽기 함수가 있다")
R = r.group(1) if r else ""
rcode = "\n".join(ln.split("#", 1)[0] for ln in re.sub(r'"""(?:.|\n)*?"""', "", R, count=1).splitlines())
chk("re.split" not in rcode and "sorted(" not in rcode,
    "문장을 나누지도 순위를 매기지도 않는다([[59]]·[[62]]④)")
chk('(a2 or {}).get("write_rules")' in R, "A2 `write_rules` 선언만 읽는다")

print("[4] **함수를 실제로 돌린다**")
try:
    import t2_gate_patch as G

    class _TC(object):
        def __init__(s, n, a):
            s.name, s.arguments = n, a

    wc = _TC("call_thing", {"agent_tool_name": "widget_filer_9001"})
    a2 = {"write_rules": [{"applies_to": "widget_filer_9001", "text": "Rule one."},
                          {"applies_to": "other_tool", "text": "Rule two."}]}
    got = G._declared_rules_for(wc, a2)
    chk(got == "- Rule one.", "선언된 그 도구의 문장만 돌려준다", repr(got))
    chk(G._declared_rules_for(_TC("call_thing", {"agent_tool_name": "nope"}), a2) is None,
        "선언이 없으면 None — 침묵한다([[25]])")
    chk(G._declared_rules_for(wc, {}) is None, "A2 가 비면 None")
except Exception as _e:
    chk(False, "헬퍼가 실행된다(死배선 아님)", repr(_e))

print("[5] A2 선언 — 출처가 붙어 있다 ([[23]])")
for p in ("a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json"):
    d = json.load(io.open(os.path.join(HERE, p), encoding="utf-8"))
    bs = [x for x in (d.get("write_arg_enum") or []) if isinstance(x, dict) and x.get("booleans")]
    ws = [x for x in (d.get("write_rules") or []) if isinstance(x, dict)]
    chk(len(bs) >= 2, "%s: 불리언 선언 %d" % (p.split("/")[-1], len(bs)))
    chk(all(x.get("_note_") for x in bs), "%s: 불리언 선언마다 `_note_` 출처" % p.split("/")[-1])
    chk(len(ws) >= 2 and all(x.get("text") and x.get("_note_") for x in ws),
        "%s: write_rules %d · 전부 text+_note_" % (p.split("/")[-1], len(ws)))

print("\n" + ("test_write_arg_type PASS" if not FAIL else "test_write_arg_type FAIL: %s" % (FAIL,)))
sys.exit(1 if FAIL else 0)
