# -*- coding: utf-8 -*-
"""x612 — t7391_reg12 retail task 22 격리 재현 · `T2_GROUND` address2 치환 ([[78]] 격리→배선 규율)

## 무엇을 재는가
라이브(t7391 task 22) 의 turn 10 에서 **모델이 낸 인자**(`raw_data`)와 **실제 실행된 인자**
(`messages[10].tool_calls`)가 `address2` 한 칸에서 다르다:

    모델 : {"address2": "Apt 1"}          ← 완성(raw) 축자
    실행 : {"address2": "Suite 865"}      ← 커밋된 tool_call 축자 (tool_call **id 동일**)

이 스크립트는 엔진 함수(`t2_gate_patch`)만 불러 그 치환을 **오프라인으로 재현**한다.
모델 0 호출 · env 0 호출 · 라이브 재료(같은 sim 의 messages[0..9])만 쓴다.

## 팔
  A_repro   : 라이브 turn 10 의 문맥 + 모델 원본 인자 → `_first_fab_call` → `_grounded_candidates`
              → `_subst_arg_value`. 실행 인자와 **바이트 동일**해야 한다.
  B_prior   : 두 런(t7391 FAIL · hist_gpt52 PASS)에서 이 모델이 낸 `address2` **원본** 값을
              전수 집계한다. 우리 층 값이 문맥에 들어가기 **전/후**로 갈라 센다.
  C_feedback: 치환이 아니라 거절-재생성으로 갔다면 무엇을 지시했을지 — 후보 목록을 찍는다
              (`GROUND_FEEDBACK` 은 "Use ONLY one of these" 다).

## 왜 이 자리인가 (코드 경로)
  t2_gate_patch.py:69-70   DEFAULT_ARG_HINTS 에 "address" 가 있어 free-text 주소 슬롯이
                           **식별자 provenance** 검사 대상이 된다.
  t2_gate_patch.py:3068    후보 필터 `if len(s) < 4` → 빈 문자열은 **후보가 될 수 없다**.
  t2_gate_patch.py:8435-45 |C|=1 이면 조용히 제자리 치환하고 `continue` (거절·재생성 우회).
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAU2 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts", "distill", "tau2")
sys.path.insert(0, TAU2)
os.chdir(TAU2)
sys.stdout.reconfigure(encoding="utf-8")

import t2_gate_patch as G  # noqa: E402

SIMS = os.path.join(HERE, "sim_results")
FAIL = os.path.join(SIMS, "t7391_reg12.results.json.gz")
PASS = os.path.join(SIMS, "hist_gpt52_reg12_PASS.results.json.gz")


def sim(path, tid="22"):
    d = json.load(gzip.open(path, "rt", encoding="utf-8"))
    return [x for x in d["simulations"] if str(x["task_id"]) == tid][0]


class M(object):
    def __init__(self, m):
        self.role = m.get("role")
        self.content = m.get("content")
        self.error = False
        self.tool_calls = None


class TC(object):
    def __init__(self, name, args):
        self.name, self.arguments, self.id = name, args, "iso"


class AM(object):
    def __init__(self, tcs):
        self.tool_calls = tcs


def raw_calls(msg):
    """완성(raw_data) 안의 tool_call 을 [(name, args_dict)] 로."""
    out = []
    try:
        for t in (msg["raw_data"]["choices"][0]["message"].get("tool_calls") or []):
            out.append((t["function"]["name"], json.loads(t["function"]["arguments"])))
    except Exception:
        pass
    return out


def main():
    s = sim(FAIL)
    print("=" * 78)
    print("A_repro — turn 10 치환 재현")
    print("=" * 78)
    msgs = [M(m) for m in s["messages"][:10]]
    name, args = raw_calls(s["messages"][10])[0]
    print("  모델 원본 :", json.dumps(args, ensure_ascii=False))
    am = AM([TC(name, dict(args))])
    ctx = G._ctx_from_messages(msgs)
    fab = G._first_fab_call(am, ctx)
    print("  fab       :", None if fab is None else (fab[1], fab[2]))
    assert fab is not None, "fab 미검출 — 재현 실패"
    tc, k, v = fab
    cands = G._grounded_candidates(k, v, msgs, lenient=True)
    print("  후보 |C|  :", len(cands), cands)
    ok = len(cands) == 1 and G._subst_arg_value(tc, k, v, cands[0])
    print("  치환      :", ok, json.dumps(tc.arguments, ensure_ascii=False))
    live = s["messages"][10]["tool_calls"][0]["arguments"]
    print("  라이브 실행:", json.dumps(live, ensure_ascii=False))
    same = json.dumps(tc.arguments, sort_keys=True) == json.dumps(live, sort_keys=True)
    print("  ⇒ 바이트 동일:", same)

    print()
    print("=" * 78)
    print("B_prior — 이 모델이 낸 address2 **원본** 전수 (두 런·task 22)")
    print("=" * 78)
    rows = []
    for tag, sm in (("t7391_FAIL", s), ("histPASS", sim(PASS))):
        for i, m in enumerate(sm["messages"]):
            if m.get("role") != "assistant":
                continue
            for nm, a in raw_calls(m):
                if nm in ("modify_user_address", "modify_pending_order_address") \
                        and "address2" in a:
                    rows.append((tag, i, nm, a.get("address1"), repr(a.get("address2"))))
    for r in rows:
        print("  %-10s msg[%2d] %-28s %-20s address2=%s" % r)
    pre = [r for r in rows if r[4] == "''"]
    print("  빈 문자열(gold 값) %d / %d" % (len(pre), len(rows)))

    print()
    print("=" * 78)
    print("C_feedback — 치환 대신 거절-재생성이었다면 무엇을 시켰나")
    print("=" * 78)
    print("  " + G.GROUND_FEEDBACK.format(k=k, s=v, cands=", ".join(cands)))
    print("  ⇒ 후보 집합에 '' 는 원리상 없다 (t2_gate_patch.py:3068 `if len(s) < 4`).")


if __name__ == "__main__":
    main()
