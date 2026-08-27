# -*- coding: utf-8 -*-
r"""x564 — 식별자 인자의 값이 **선언된 생산자 read 에서 왔는가** (유료 0 · 코퍼스 계수).

## 왜 (2026-08-27 · t7364 부검에서 나온 원인)

`[PROVENANCE]` 의 술어는 `_ctx_has` — *"이 문자열이 문맥 어딘가에 있나"* 다. 그래서 **출처는
맞고 자리가 틀린** 값이 전부 통과한다. 실측 넷:

    074  `Dark Green Account`  손님 발화 msg[1] → msg[4] `get_atm_fee_discrepancies{account_id: …}`
    072  `Bluest Account`      같은 형태 (그 read 는 `chk_lj82d4f1a9` 류 id 를 받는다)
    085  `f7d3a82c91`          env 출력 msg[9](**user id**) → 나중에 `account_id` 자리로
    016  `friend_user_5839`    손님 발화 msg[37] → env 의 어느 레코드에도 없다

선언은 이미 답을 들고 있다 — `arg_source_reads` 13 인자가 각각 **어느 read 가 그 값을 낳는지**
적어 두었고(env desc 축자·[[23]] 확립), 지금 소비자는 `_fab_fix_note` 하나(이름 나열)뿐이다.

## 이 프로브가 하는 일 — **세기만** 한다

호출마다, 선언된 식별자 인자의 값이 **그 호출 이전에** 어디서 나왔는지 분류한다:

    producer   선언된 생산자 read 의 출력에 있다          ← 정상
    othertool  다른 도구 출력에는 있다(생산자 아님)        ← 085 형
    customer   손님 발화에만 있다                          ← 074·072·016 형
    absent     어디에도 없다                               ← 기존 PROVENANCE 가 잡는 것

⛔규칙을 제안하지 않는다. 어떤 분류가 **해로운가**는 이 표를 보고 정한다([[62]]).
특히 `verify_identity` 처럼 **손님 값이 제자리인** 호출이 얼마나 되는지가 규칙의 모양을 정한다.

사용: PYTHONIOENCODING=utf-8 py -3 x564_arg_producer_census.py
"""
import argparse
import collections
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

import gate_interpreter as GI                                       # noqa: E402
import t2_forensic as F                                             # noqa: E402
import t2_dominance as DOM                                          # noqa: E402


class _TC(object):
    def __init__(self, d):
        self.name = F.nameof(d)
        self.arguments = F.argsof(d)


def field_labels(ms, upto):
    """도구 출력의 **`필드: 값`** 을 축자로 — env 고정 포맷 전사(`_parse_record_dump` 와 같은 층).

    ⚠[[59]] 경계: 값의 뜻을 읽지 않는다. `(\w+):\s*(비공백)` 한 규칙뿐이고, 형식이 아니면
      아무것도 안 나온다(fail-open)."""
    out = {}
    for i, m in enumerate(ms[:upto]):
        if m.get("role") != "tool":
            continue
        c = " ".join(str(m.get("content") or "").split())
        for f, v in re.findall(r"(\w+):\s*([^\s,;]+)", c):
            out.setdefault(f, set()).add((i, v))
    return out


# ★잡음 둘 — 둘 다 **선언으로** 걷어낸다(엔진 리터럴 0).
#   ⑴ `Record ID:` 의 `ID` 는 필드 이름이 아니라 덤프의 머리다 — 같은 레코드의 옳은 값을
#      가리키므로 불일치가 아니다(079 `ID→card_id` 18건이 전부 이것).
#   ⑵ 같은 축의 **동의어**(`phone`/`phone_number`)는 `arg_source_reads` 의 생산자 목록이
#      **완전히 같다** — 그 동일성으로 판정한다(040 17건이 이것).
DUMP_HEAD = "ID"


def same_axis(asr, a, b):
    return bool(asr.get(a)) and asr.get(a) == asr.get(b)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="bank_t7364_hard0_20260827,bank_t7363_hard0_20260827,"
                                      "bank_t7356_grpA1_20260826,bank_t7356_grpA2_20260826,"
                                      "bank_t7356_grpA3_20260826,bank_t7356_grpA4_20260826,"
                                      "bank_t7356_grpB3_20260826")
    ap.add_argument("--domain", default="banking_knowledge")
    a = ap.parse_args(argv)
    a2 = GI.load_domain_a2(a.domain) or {}
    asr = {k: [str(x) for x in v]
           for k, v in (a2.get("arg_source_reads") or {}).items()
           if not k.startswith("_") and isinstance(v, list)}
    print("# x564 — 선언된 식별자 인자 %d" % len(asr))

    tally = collections.Counter()
    bytool = collections.defaultdict(collections.Counter)
    rows = []
    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        try:
            sims = F.scored(tag)
        except Exception:
            continue
        for s in sims:
            ms = s.get("messages") or []
            rw = (s.get("reward_info") or {}).get("reward")
            tid = F.task_id(s)
            # 도구 출력 → 그것을 낸 도구 이름
            out_by = []          # (index, tool_name, text)
            pend = {}
            for i, m in enumerate(ms):
                for tc in (m.get("tool_calls") or ()):
                    pend[i] = F.label(F.nameof(tc), F.argsof(tc))
                if m.get("role") == "tool":
                    nm = ""
                    for j in range(i - 1, max(-1, i - 3), -1):
                        if j in pend:
                            nm = pend[j]
                            break
                    out_by.append((i, nm, str(m.get("content") or "")))
            for i, m in enumerate(ms):
                for tc in (m.get("tool_calls") or ()):
                    args = DOM._args_dict(_TC(tc))
                    caller = F.inner_name(F.argsof(tc)) or F.nameof(tc)
                    for k, v in (args or {}).items():
                        if k not in asr or not isinstance(v, str):
                            continue
                        val = v.strip()
                        if len(val) < 4:
                            continue
                        prods = asr[k]
                        seen_prod = any(val in txt and any(p.split("_by_")[0] in nm or nm in p
                                                           for p in prods)
                                        for j, nm, txt in out_by if j < i)
                        seen_tool = any(val in txt for j, nm, txt in out_by if j < i)
                        seen_user = any(val in str(x.get("content") or "")
                                        for x in ms[:i] if x.get("role") == "user")
                        cls = ("producer" if seen_prod else
                               "othertool" if seen_tool else
                               "customer" if seen_user else "absent")
                        tally[cls] += 1
                        bytool[caller][cls] += 1
                        rows.append((tag, tid, rw, caller, k, val, cls))

    print("## 전체 분류 (호출·인자 단위)")
    tot = sum(tally.values())
    for c in ("producer", "othertool", "customer", "absent"):
        print("   %-10s %5d  (%.0f%%)" % (c, tally[c], 100.0 * tally[c] / tot if tot else 0))
    print()
    print("## 도구별 — `customer`/`othertool` 이 있는 것만")
    for t in sorted(bytool, key=lambda x: -(bytool[x]["customer"] + bytool[x]["othertool"])):
        c = bytool[t]
        if not (c["customer"] or c["othertool"]):
            continue
        print("   %-42s producer %-4d othertool %-4d customer %-4d absent %d"
              % (t[:42], c["producer"], c["othertool"], c["customer"], c["absent"]))
    print()
    print("## 표적 넷의 그 호출")
    for tag, tid, rw, caller, k, val, cls in rows:
        if val in ("Dark Green Account", "Bluest Account", "f7d3a82c91", "friend_user_5839"):
            print("   %-9s r=%-5s %-38s %-16s %-22s %s" % (tid, rw, caller[:38], k, val[:22], cls))
    return 0


if __name__ == "__main__":
    sys.exit(main())
