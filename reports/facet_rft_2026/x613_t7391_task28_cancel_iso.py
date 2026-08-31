# -*- coding: utf-8 -*-
"""x613 — t7391_reg12 retail task 28 격리 재현 ([[78]] 격리→배선 규율).

물음 하나: **msg 15 의 `cancel_pending_order(reason="no longer needed")` 를
우리 층이 막을 수 있었는가, 그리고 왜 안 막았는가.**

팔 (전부 라이브 궤적 축자 재료 · gold 무참조 · 모델 호출 0):
  A_LIVE   현행 배선 그대로 — retail A2 의 `write_arg_grounding` 은 **부재** ⇒ specs=[] ⇒ skip.
  B_WAG    같은 엔진 함수에 retail 선언 한 줄(`cancel_pending_order.reason`)만 주면 어떻게 되나.
  N_NEG    [[57]] 부정통제 — 같은 선언 형식을 **근거가 실재하는 인자**(return 의 order_id·
           payment_method_id)에 걸면 통과해야 한다(무차별 deny 가 아님을 보인다).
  N_NEG2   [[57]] 두번째 — 같은 길이의 무관한 값(`"ordered by mistake"`, 정책의 다른 사유)도
           같은 이유로 미실재 ⇒ deny. 즉 사는 것은 *길이*가 아니라 **실재**다.
  C_RULE   `_declared_rules_for` 가 retail 에서 무엇을 돌려주나 (T2_RULE_AT_WRITE 의 재료).

⛔이 파일은 진단 전용이다. 엔진·A2 를 한 줄도 고치지 않는다.
"""
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAU2 = r"C:\workspace\ba-frft\scripts\distill\tau2"
sys.path.insert(0, TAU2)
RES = os.path.join(HERE, "sim_results", "t7391_reg12.results.json.gz")

import t2_gate_patch as G  # noqa: E402


class M(object):
    def __init__(self, role, content):
        self.role = role
        self.content = content


class TC(object):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


def load_sim(task_id="28"):
    d = json.load(gzip.open(RES, "rt", encoding="utf-8"))
    return [x for x in d["simulations"] if x["task_id"] == task_id][0]


def msgs_upto(sim, n):
    """msg 0..n-1 을 엔진이 보는 형태로 (role/content 만 쓴다)."""
    out = []
    for m in sim["messages"][:n]:
        out.append(M(m.get("role"), m.get("content") or ""))
    return out


def main():
    sim = load_sim("28")
    ctx = msgs_upto(sim, 15)          # msg 15(문제의 write) **직전**까지가 근거 코퍼스다

    cancel_tc = TC("cancel_pending_order",
                   {"order_id": "#W2575533", "reason": "no longer needed"})
    ret_tc = TC("return_delivered_order_items",
                {"order_id": "#W3792453", "item_ids": ["4293355847"],
                 "payment_method_id": "paypal_3024827"})

    a2r = json.load(io.open(os.path.join(TAU2, "a2", "retail.gate.json"), encoding="utf-8"))

    print("=" * 72)
    print("A_LIVE — 현행 retail 배선")
    live_specs = a2r.get("write_arg_grounding") or []
    print("  a2/retail.gate.json.write_arg_grounding =", live_specs, "(len=%d)" % len(live_specs))
    print("  _write_arg_ground_deny ->",
          repr(G._write_arg_ground_deny(ctx, cancel_tc, live_specs)))

    print("=" * 72)
    print("B_WAG — retail 선언 한 줄만 추가했다면 (엔진 무수정)")
    spec = [{"applies_to": "cancel_pending_order", "grounded_args": ["reason"]}]
    print("  _write_arg_ground_deny ->",
          repr(G._write_arg_ground_deny(ctx, cancel_tc, spec)))

    print("=" * 72)
    print("N_NEG — 근거가 실재하는 인자에 같은 형식을 걸면 (무차별 deny 가 아님)")
    spec_ok = [{"applies_to": "return_delivered_order_items",
                "grounded_args": ["order_id", "payment_method_id"]}]
    print("  _write_arg_ground_deny ->",
          repr(G._write_arg_ground_deny(ctx, ret_tc, spec_ok)))

    print("=" * 72)
    print("N_NEG2 — 정책의 *다른* 허용 사유도 대화에 미실재")
    tc2 = TC("cancel_pending_order",
             {"order_id": "#W2575533", "reason": "ordered by mistake"})
    print("  _write_arg_ground_deny ->",
          repr(G._write_arg_ground_deny(ctx, tc2, spec)))

    print("=" * 72)
    print("C_RULE — T2_RULE_AT_WRITE 의 재료 (`write_rules`)")
    print("  retail  _declared_rules_for(cancel_pending_order) ->",
          repr(G._declared_rules_for(cancel_tc, a2r)))
    a2b = json.load(io.open(os.path.join(TAU2, "a2", "banking_knowledge.gate.json"),
                            encoding="utf-8"))
    print("  banking write_rules 선언 수 =", len(a2b.get("write_rules") or []))

    print("=" * 72)
    print("D_CORPUS — 'no longer needed' 가 msg 0..14 의 어디에 있나")
    for i, m in enumerate(sim["messages"][:15]):
        c = (m.get("content") or "") + json.dumps(m.get("tool_calls") or [], ensure_ascii=False)
        if "no longer needed" in c:
            print("   HIT msg", i, m.get("role"))
    print("   (출력 없음 = 0건)")

    print("=" * 72)
    print("E_A2GAP — 엔진이 읽는데 retail 에 선언이 없는 A2 키")
    rk = set(a2r)
    bonly = {k for k in set(a2b) - rk if not k.startswith("_note")}
    src = ""
    for f in ("t2_gate_patch.py", "t2_scaffold_get.py", "t2_prekb_patch.py"):
        p = os.path.join(TAU2, f)
        if os.path.exists(p):
            src += io.open(p, encoding="utf-8", errors="ignore").read()
    used = sorted(k for k in bonly if ('"%s"' % k) in src)
    print("   retail keys=%d · banking-only=%d · 그중 엔진이 읽는 것=%d"
          % (len(rk), len(bonly), len(used)))
    print("   ", used)


if __name__ == "__main__":
    main()
