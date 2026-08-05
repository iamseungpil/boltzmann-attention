# -*- coding: utf-8 -*-
"""Does the procedure walker say the right thing, and stay silent otherwise?

Cases are written against the real A2 declaration, not a fixture, because the thing
that has repeatedly failed in this repo is a predicate that passes its own fixture and
never matches production data ([[30]]). The trajectory being reproduced is the smoke's
task_051: the decision tool called with the request never submitted and one check never
run, and a months argument the tier table fixes.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_procedure as P  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"), encoding="utf-8"))
PROCS = A2.get("procedures") or []
fail = []


def check(name, cond, detail=""):
    print("  %s %s%s" % ("✓" if cond else "✗", name, ("  — " + detail) if detail and not cond else ""))
    if not cond:
        fail.append(name)


print("① 선언이 실려 있는가")
check("procedures 존재", len(PROCS) == 1, "%d" % len(PROCS))
proc = PROCS[0] if PROCS else {}
check("노드 7", len(proc.get("nodes") or []) == 7)
check("표 5", len(proc.get("tables") or {}) == 5)

print("\n② 051 재현 — 결정 도구를 먼저 부른다")
DECIDE = "approve_credit_limit_increase_5847"
executed = {"get_credit_limit_increase_history_4829", "get_payment_history_6183",
            "get_pending_replacement_orders_5765"}          # 스모크가 실제로 부른 것
missing, unobs = P.unmet_nodes(P.find_procedure(PROCS, DECIDE), DECIDE, executed)
check("요청 제출 누락 검출", "submit_request" in missing, str(missing))
check("분쟁 이력 누락 검출", "disputes" in missing, str(missing))
check("이미 부른 것은 미포함", "cooldown" not in missing and "payment_history" not in missing, str(missing))
check("관측 불가 노드는 분리", "amount_within_tier_cap" in unobs, str(unobs))

print("\n③ 표에서 유도되는 인자")
te = P.table_expectation(P.find_procedure(PROCS, "get_payment_history_6183"),
                         "get_payment_history_6183", {"card_tier": "entry"})
check("entry → months 6", te == ("months", 6), str(te))
te_mid = P.table_expectation(P.find_procedure(PROCS, "get_payment_history_6183"),
                             "get_payment_history_6183", {"card_tier": "mid"})
check("mid → months 3", te_mid == ("months", 3), str(te_mid))
check("등급 미제공이면 추측 안 함",
      P.table_expectation(P.find_procedure(PROCS, "get_payment_history_6183"),
                          "get_payment_history_6183", {}) is None)

print("\n④ 문구는 A2에서 온다 · 무관 호출엔 침묵")
notes = P.notes_for_call(PROCS, DECIDE, {}, executed, {"card_tier": "entry"})
check("미충족 문구 1개", len(notes) == 1, str(notes))
check("문구에 누락 단계 이름", "submit_request" in notes[0] if notes else False)
notes2 = P.notes_for_call(PROCS, "get_payment_history_6183", {"months": 12}, executed,
                          {"card_tier": "entry"})
check("months 12 → 교정 문구", any("6" in n for n in notes2), str(notes2))
notes3 = P.notes_for_call(PROCS, "get_payment_history_6183", {"months": 6}, executed,
                          {"card_tier": "entry"})
check("months 6이면 그 문구 없음", not any("months" in n for n in notes3), str(notes3))
check("절차 밖 도구엔 침묵", P.notes_for_call(PROCS, "KB_search_bm25", {}, executed) == [])

print("\n⑤ [[05]] 엔진 리터럴 0")
# 산문이 아니라 **실행되는 것**만 본다: 독스트링을 뺀 문자열 상수와 식별자.
# 첫 판은 원시 텍스트를 훑어 독스트링의 영어 단어("entry point")를 도메인 등급명으로 오인했다 —
# 계측기를 먼저 의심하라는 이 아크의 규율이 여기에도 적용된다(C279⑪).
import ast  # noqa: E402

tree = ast.parse(open(os.path.join(HERE, "t2_procedure.py"), encoding="utf-8").read())
docstrings = set()
for n in ast.walk(tree):
    if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)):
        d = ast.get_docstring(n, clean=False)
        if d:
            docstrings.add(d)
live = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in docstrings:
        live.add(n.value)
    elif isinstance(n, ast.Name):
        live.add(n.id)
    elif isinstance(n, ast.Attribute):
        live.add(n.attr)
blob = " ".join(sorted(live)).lower()
for lit in ("credit_limit", "months", "card_tier", "entry", "premium", "submit_credit", "dispute"):
    check("실행 코드에 '%s' 없음" % lit, lit not in blob)

print("\n⑥ 강제는 정책이 허가할 때만 (index + MUST 문장)")
check("이 절차는 강제 대상", P.is_mandatory(proc))
d1 = P.decide(PROCS, DECIDE, {}, executed, {"card_tier": "entry"})
check("선행 미충족 → deny", d1["verdict"] == "deny", str(d1["verdict"]))
d2 = P.decide(PROCS, DECIDE, {}, executed | {"submit_credit_limit_increase_request_7392",
                                             "get_user_dispute_history_7291"},
              {"card_tier": "entry"})
check("전부 충족 → pass", d2["verdict"] == "pass", str(d2))
check("절차 밖 도구 → pass·침묵",
      P.decide(PROCS, "KB_search_bm25", {}, executed)["verdict"] == "pass")
no_licence = dict(proc); no_licence.pop("_quote_order", None)
check("MUST 문장 없으면 강제 안 함", not P.is_mandatory(no_licence))
no_flag = dict(proc); no_flag["enforce"] = False
check("enforce 미선언이면 강제 안 함", not P.is_mandatory(no_flag))

print("\nRESULT2: %s" % ("ALL PASS" if not fail else "FAIL %s" % fail))
sys.exit(1 if fail else 0)
