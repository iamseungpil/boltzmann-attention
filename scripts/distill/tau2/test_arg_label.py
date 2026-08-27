#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""`T2_ARG_LABEL` 래칫 — 실제 궤적으로 초 단위 검정.

## 무엇을 잠그나 (측정 정본 = `x564_arg_producer_census.py` · 격리 = `x565_wrong_account_id_iso.py`)

`_provenance_deny` 는 *"문맥 어딘가에 있나"* 만 본다. env 는 레코드를 `필드: 값` 으로 찍으므로
**종류**의 답이 이미 문맥에 있는데 우리가 안 봤다. 085 축자: `user_id: f7d3a82c91` 이 나온 뒤
모델이 그 값을 `account_id` 로 넘긴다(계좌 목록이 오기 **전**·msg[24]) — 그 뒤 열 호출은 전부
옳다. 결손은 *"옳은 값을 못 고른다"* 가 아니라 **없는 값을 이웃 필드에서 빌린다** 이다.

## 다섯 (잡음 둘이 여기 들어 있다 — 둘 다 나를 물었다)

⑴ 085 msg[24] `account_id=f7d3a82c91` → **반려**하고 `user_id` 를 이름으로 댄다.
⑵ 085 msg[30] `account_id=chk_b4d92f7c28` → **침묵**(제 이름표로 나온 값).
⑶ **덤프 머리** `Record ID:` 의 `ID` 는 필드가 아니다 — 그것만 근거인 값은 침묵(079 18건).
⑷ **같은 축 동의어**(`phone`/`phone_number`)는 침묵 — 생산자 목록 동일성으로 판정(040 17건).
⑸ 이름표는 **레코드 덤프에서만** 읽는다 — 스키마 줄(`account_id: string (required)`)을 먹으면
   `string` 이 그 인자의 옳은 값이 된다.

## ⛔여기서 판정하지 않는 것
*"반려하면 통과하는가"* 는 런이 잰다([[69]]). 여기서는 술어가 옳게 갈리는지만 본다.

실행: PYTHONIOENCODING=utf-8 py -3 test_arg_label.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import gate_interpreter as GI                                       # noqa: E402
import t2_forensic as F                                             # noqa: E402
import t2_gate_patch as G                                           # noqa: E402

FAIL = []
A2 = GI.load_domain_a2("banking_knowledge") or {}


def chk(c, ok, extra=""):
    print(("  OK   " if ok else "  FAIL ") + c + (("  — " + extra) if extra else ""))
    if not ok:
        FAIL.append(c)


class _M(object):
    def __init__(self, m):
        self.role = m.get("role")
        self.content = m.get("content")


class _Orch(object):
    def __init__(self, ms):
        self._ms = [_M(m) for m in ms]

    def get_messages(self):
        return self._ms


class _TC(object):
    def __init__(self, name, args):
        self.name = name
        self.arguments = args


def sim_of(tag, task):
    return next((s for s in F.scored(tag) if F.task_id(s) == task), None)


def deny_at(tag, task, upto, name, args):
    s = sim_of(tag, task)
    if not s:
        return "sim 없음"
    labels = G._record_labels(_Orch((s.get("messages") or [])[:upto]))
    return G._label_mismatch_deny(_TC(name, args), A2, labels)


print("## ⑴⑵ 085 — 빌린 값은 반려, 제 이름표 값은 침묵")
d1 = deny_at("bank_t7363_hard0_20260827", "task_085", 24,
             "get_bank_account_transactions_9173", {"account_id": "f7d3a82c91"})
chk("msg[24] user_id 를 account_id 로 → 반려", bool(d1) and d1 != "sim 없음", str(d1)[:90])
chk("반려문이 `user_id` 를 이름으로 댄다([[64]])", bool(d1) and "user_id" in str(d1[1]))
chk("반려문이 무엇을 하면 풀리는지 댄다",
    bool(d1) and "get_all_user_accounts_by_user_id" in str(d1[1]))
d2 = deny_at("bank_t7363_hard0_20260827", "task_085", 32,
             "get_bank_account_transactions_9173", {"account_id": "chk_b4d92f7c28"})
chk("msg[32] 제 이름표로 나온 값 → 침묵", d2 is None, str(d2)[:80])

print()
print("## ⑶ 덤프 머리 `ID` 는 필드가 아니다")
# ★규칙 검정이다(사례 검정이 아니라) — 덤프는 `Record ID: X account_id: X` 처럼 같은 값을
#   두 이름으로 찍으므로 *"ID 로만 나온 값"* 은 실물에서 드물다. 잠그는 것은 **이름 하나**다:
#   `ID` 는 필드가 아니라 덤프의 머리이고, 그것만 근거이면 불일치가 아니다(x564: 079 18건).
d3 = G._label_mismatch_deny(_TC("x", {"card_id": "dbc_cr89a2b3c4_ev"}),
                            A2, {G._DUMP_HEAD: {"dbc_cr89a2b3c4_ev"}})
chk("`ID` 만 근거면 침묵", d3 is None, str(d3)[:80])
d3b = G._label_mismatch_deny(_TC("x", {"card_id": "dbc_cr89a2b3c4_ev"}),
                             A2, {"user_id": {"dbc_cr89a2b3c4_ev"}})
chk("다른 진짜 필드가 근거면 반려(대조)", bool(d3b), str(d3b)[:70])

print()
print("## ⑷ 같은 축 동의어는 침묵")
asr = {k: v for k, v in (A2.get("arg_source_reads") or {}).items()
       if not k.startswith("_") and isinstance(v, list)}
chk("phone/phone_number 가 같은 축", G._same_axis(asr, "phone", "phone_number"),
    str(asr.get("phone")))
d4 = G._label_mismatch_deny(_TC("x", {"phone": "215-555-0267"}),
                            A2, {"phone_number": {"215-555-0267"}})
chk("동의어 값은 침묵", d4 is None, str(d4)[:80])

print()
print("## ⑸ 이름표는 레코드 덤프에서만")
lab85 = G._record_labels(_Orch((sim_of("bank_t7363_hard0_20260827", "task_085")
                                or {}).get("messages") or []))
chk("account_id 이름표에 `string` 이 없다", "string" not in lab85.get("account_id", set()),
    str(sorted(lab85.get("account_id", set()))[:4]))
chk("account_id 이름표가 실제 id 를 담는다",
    any(v.startswith("chk_") for v in lab85.get("account_id", set())))

print()
print("## 플래그로만 켜진다")
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
chk("플래그 검사가 소스에 있다", 'os.environ.get("T2_ARG_LABEL") == "1"' in SRC)

print()
print("결과: %s" % ("모두 통과" if not FAIL else "실패 %d — %s" % (len(FAIL), FAIL)))
sys.exit(1 if FAIL else 0)
