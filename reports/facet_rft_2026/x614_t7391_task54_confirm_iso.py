# -*- coding: utf-8 -*-
"""x614 — t7391_reg12 (retail) task 54 격리 재현 ([[78]] 격리→배선 규율).

물음 둘:
  ⒜ **msg 37 의 `return_delivered_order_items(#W4597054)` 를 우리 층이 막을 수 있었는가.**
     (그 write 는 gold 이지만 손님에게 **한 번도 제시된 적이 없다** — 손님이 msg 42 에서
      *"I didn't ask to return anything else"* 로 되물었고, 모델이 자기 gold write 를
      부인하면서 NL 축 총액 $3,646.68 이 $2,460.21 로 줄었다.)
  ⒝ **msg 30 의 `reason="ordered by mistake"` 를 우리 층이 막을 수 있었는가.**
     (gold = `"no longer needed"` — DB 축을 죽인 유일한 필드.)

팔 (전부 라이브 궤적 축자 재료 · gold 무참조 · 모델 호출 0):
  A_LIVE     현행 배선 그대로 — `GateInterpreter.check` 를 라이브와 같은 인자로 재생.
  B_ACTBIND  TASK_28 §7 P3 형 — 확인 창(직전 assistant 텍스트 ∪ 직전 user)에 **도구 어간**이
             실재하는가. 어간은 도구 이름에서 나온다(도메인 어휘 저작 0).
  N_NEG      [[57]] 부정통제 — 같은 술어를 msg 30 의 cancel 에 걸면 **통과**해야 한다
             (손님이 축자로 *"cancel both pending orders"* 라고 말했다).
  C_RULE     `_declared_rules_for` 가 retail 에서 무엇을 돌려주나 (T2_RULE_AT_WRITE 재료).
  D_WAG      `write_arg_grounding` 이 retail 에 있나 · 있다면 `reason` 을 잡나.
  E_CORPUS   `"no longer needed"` / `"ordered by mistake"` / `"financial issue"` 가
             결정점(msg 30) **직전까지** 어디에 몇 번 있었나 (role 별).
  F_TOTAL    NL 축이 요구한 `1186.47` 이 총액 계산 시점(msg 43) 문맥에 실재했나.

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
sys.stdout.reconfigure(encoding="utf-8")
RES = os.path.join(HERE, "sim_results", "t7391_reg12.results.json.gz")

import t2_gate_patch as G            # noqa: E402
from gate_interpreter import GateInterpreter, CONFIRM_RE   # noqa: E402


class M(object):
    def __init__(self, role, content):
        self.role = role
        self.content = content


class TC(object):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


def load_sim(task_id="54"):
    d = json.load(gzip.open(RES, "rt", encoding="utf-8"))
    return [x for x in d["simulations"] if x["task_id"] == task_id][0]


def msgs_upto(sim, n):
    return [M(m.get("role"), m.get("content") or "") for m in sim["messages"][:n]]


def last_user(sim, n):
    """엔진의 `_regen_last_user` 와 같은 술어 — msg n 직전 마지막 user 본문."""
    for i in range(n - 1, -1, -1):
        m = sim["messages"][i]
        if m.get("role") == "user" and m.get("content"):
            return i, m["content"]
    return -1, ""


def prev_assistant_text(sim, upto_user_i):
    """확인 창의 나머지 절반 — 그 user 발화 **직전**의 assistant 텍스트."""
    for i in range(upto_user_i - 1, -1, -1):
        m = sim["messages"][i]
        if m.get("role") == "assistant" and m.get("content"):
            return i, m["content"]
    return -1, ""


STEMS = {"cancel_pending_order": "cancel",
         "return_delivered_order_items": "return",
         "exchange_delivered_order_items": "exchange",
         "modify_pending_order_items": "modif",
         "modify_pending_order_address": "modif",
         "modify_pending_order_payment": "modif",
         "modify_user_address": "modif"}


def actbind(sim, msg_i, tool):
    """B_ACTBIND 술어: 확인 창(직전 assistant 텍스트 ∪ 직전 user)에 도구 어간이 실재하는가."""
    ui, ut = last_user(sim, msg_i)
    ai, at = prev_assistant_text(sim, ui)
    stem = STEMS.get(tool, tool.split("_")[0])
    win = (at + " " + ut).lower()
    return {"stem": stem, "user_i": ui, "assist_i": ai,
            "in_window": stem in win,
            "in_user": stem in ut.lower(), "in_assist": stem in at.lower()}


def main():
    sim = load_sim("54")
    a2r = json.load(io.open(os.path.join(TAU2, "a2", "retail.gate.json"), encoding="utf-8"))
    gates = a2r["gates"]

    WRITES = [(28, "cancel_pending_order", {"order_id": "#W4836353", "reason": "financial issue"}),
              (30, "cancel_pending_order", {"order_id": "#W4836353", "reason": "ordered by mistake"}),
              (30, "cancel_pending_order", {"order_id": "#W7342738", "reason": "ordered by mistake"}),
              (37, "return_delivered_order_items",
               {"order_id": "#W4597054",
                "item_ids": ["5669664287", "4900990404", "9862136885", "6777246137"],
                "payment_method_id": "gift_card_3491931"})]

    print("=" * 78)
    print("A_LIVE — 현행 G2 재생 (auth 확립 후 · resolvers 없음 = precondition skip)")
    for mi, tool, args in WRITES:
        gi = GateInterpreter(gates)
        gi.auth_user = "amelia_silva_7726"
        gi.state.presented_select = True      # 라이브: msg 28 이전에 G6 이 1회 소진됨
        ui, ut = last_user(sim, mi)
        ok, gid, why = gi.check(tool, args, last_user_msg=ut, transfer_msg_sent=None)
        m = CONFIRM_RE.search(ut or "")
        print("  msg %-3d %-30s allowed=%-5s gate=%-16s lastUser=msg%-3d token=%r"
              % (mi, tool, ok, gid, ui, m.group(0) if m else None))
    print("  ⇒ 라이브 결손 재현: msg 37 의 return 이 **cancel 확인**으로 통과한다.")

    print("=" * 78)
    print("B_ACTBIND — 확인 창에 도구 어간이 실재하는가 (도메인 어휘 저작 0)")
    for mi, tool, args in WRITES:
        r = actbind(sim, mi, tool)
        print("  msg %-3d %-30s stem=%-9s window=(assist msg%-3d ∪ user msg%-3d) in_window=%-5s "
              "(assist=%s user=%s)"
              % (mi, tool, r["stem"], r["assist_i"], r["user_i"], r["in_window"],
                 r["in_assist"], r["in_user"]))
    print("  ⇒ N_NEG: cancel 은 창에 실재(통과) · return 은 부재(차단) — 무차별 deny 아님([[57]]).")

    print("=" * 78)
    print("C_RULE — T2_RULE_AT_WRITE 재료 (retail vs banking)")
    for t in ("cancel_pending_order", "return_delivered_order_items"):
        print("  retail  _declared_rules_for(%-30s) -> %r" % (t, G._declared_rules_for(TC(t, {}), a2r)))
    try:
        a2b = json.load(io.open(os.path.join(TAU2, "a2", "banking_knowledge.gate.json"),
                                encoding="utf-8"))
        print("  banking write_rules 선언 수 =", len(a2b.get("write_rules") or []))
    except Exception as e:
        print("  banking a2 로드 실패:", repr(e))

    print("=" * 78)
    print("D_WAG — write_arg_grounding")
    live = a2r.get("write_arg_grounding") or []
    print("  a2/retail.gate.json.write_arg_grounding =", live, "(len=%d)" % len(live))
    ctx30 = msgs_upto(sim, 30)
    tc30 = TC("cancel_pending_order", {"order_id": "#W4836353", "reason": "ordered by mistake"})
    print("  A_LIVE  _write_arg_ground_deny ->", repr(G._write_arg_ground_deny(ctx30, tc30, live)))
    spec = [{"applies_to": "cancel_pending_order", "grounded_args": ["reason"]}]
    print("  B_WAG   _write_arg_ground_deny ->", repr(G._write_arg_ground_deny(ctx30, tc30, spec))[:180])
    tc_gold = TC("cancel_pending_order", {"order_id": "#W4836353", "reason": "no longer needed"})
    print("  N_NEG2  gold 값에도 같은 거부가 나오나 ->",
          repr(G._write_arg_ground_deny(ctx30, tc_gold, spec))[:120])

    print("=" * 78)
    print("E_CORPUS — 결정점(msg 30) 직전까지 각 문자열이 어디에 몇 번 있었나")
    for needle in ("no longer needed", "ordered by mistake", "financial issue"):
        per = {}
        for i, m in enumerate(sim["messages"][:30]):
            hay = json.dumps({"c": m.get("content") or "", "t": m.get("tool_calls") or []},
                             ensure_ascii=False).lower()
            n = hay.count(needle)
            if n:
                per.setdefault(m.get("role"), []).append((i, n))
        print("  %-20s -> %s" % (needle, per or "0건 (전 role · tool_calls 포함)"))

    print("=" * 78)
    print("F_TOTAL — NL 축이 요구한 값이 총액 계산 시점(msg 43) 문맥에 실재했나")
    for needle in ("1186.47", "1429.81", "1030.4", "3646.68"):
        hits = [i for i, m in enumerate(sim["messages"][:43])
                if needle in json.dumps({"c": m.get("content") or "",
                                         "t": m.get("tool_calls") or []}, ensure_ascii=False)]
        print("  %-9s msgs=%s" % (needle, hits if hits else "없음"))


if __name__ == "__main__":
    main()
