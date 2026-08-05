# -*- coding: utf-8 -*-
"""The four prescriptions from tonight's smoke, checked against the run that produced them.

Each one comes from a step that was read, not from a category:

  C1  the notice gate named `transfer_to_human_agents` while the discovery message said not to
      transfer, and gold wanted a protocol transfer (032/033/035 — 5 simulations)
  C2  the follow-up budget was one counter shared by two functions, so the give-nudge spent it
      and `chain[3]` — whose predicate held at the terminal turn of 028 — never spoke
  C4  `task_048` burned ten messages fetching a card's last four digits although the closure
      tool's signature does not take them; the trigger read prose, not the pending write
  C7  `task_053` was told eleven times that no tool output showed digits that step 32 does
      show — the customer ran that tool, so it is missing from the agent-side history, and the
      recovery text pointed at an account-record field that does not exist

The replays use the persisted trajectory of the run itself, so a regression shows up as the
same false denial returning.
"""

import glob
import gzip
import io
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as G      # noqa: E402
import gate_interpreter as GI  # noqa: E402

A2 = GI.load_domain_a2("banking_knowledge") or {}
SIM = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
fail = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL", (" — " + detail) if detail else ""))
    if not ok:
        fail.append(name)


def sims(tag="20260805n"):
    out = []
    for p in sorted(glob.glob(os.path.join(SIM, "bank_smk_gpu*_%s.results.json.gz" % tag))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            out.extend(json.load(f).get("simulations") or [])
    return out


def objs(messages):
    out = []
    for m in messages:
        tcs = [types.SimpleNamespace(name=tc.get("name"), arguments=tc.get("arguments"),
                                     id=tc.get("id"),
                                     requestor=tc.get("requestor", "assistant"))
               for tc in (m.get("tool_calls") or [])]
        out.append(types.SimpleNamespace(role=m.get("role"), content=m.get("content"),
                                         tool_calls=tcs or None, id=m.get("id"),
                                         tool_call_id=m.get("tool_call_id"),
                                         requestor=m.get("requestor", "assistant")))
    return out


# ── C1: 통지 게이트가 프로토콜 도구를 열어 둔다 ──────────────────────────────────────
gb2 = [g for g in (A2.get("gates") or []) if g.get("id") == "GB2_NOTICE_BEFORE_TRANSFER"]
ask = (gb2[0].get("ask") if gb2 else "") or ""
check("C1 통지문이 표준 이관을 유일 선택지로 말하지 않는다",
      "transfer_to_human_agents" in ask and "instead of the general one" in ask)

# ── C2: follow-up 예산이 사라졌다(공용 카운터·cap 참조 0) ─────────────────────────────
src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
check("C2 공용 follow-up 카운터가 없다", "_t2_followup\"" not in src and "_t2_followup," not in src)
check("C2 체인 발화가 예산 창에 걸리지 않는다", '_fu_mode = "normal"' in src)

# ── C4: 값 획득 넛지는 그 write가 실제로 걸려 있을 때만 ───────────────────────────────
va = A2.get("value_acquisition") or []
check("C4 선언이 레지스트리 정확 이름을 말한다",
      bool(va) and va[0].get("write", "").endswith("_4829"), str(va and va[0].get("write")))
# C4의 엔진 조건은 **철회**했다(원래 표적까지 침묵시켰다) — 남은 것은 이름 정확화뿐이고,
# 048의 last-4 우회는 열린 술어로 남겨 계속 계측한다.

# ── C7: 손님이 실행한 도구의 출력도 증거다 ───────────────────────────────────────────
s053 = [s for s in sims() if s.get("task_id") == "task_053"]
specs = A2.get("write_evidence_specs") or []
if s053:
    ms = objs(s053[0].get("messages") or [])
    call = types.SimpleNamespace(
        name="call_discoverable_agent_tool", id="t",
        arguments={"agent_tool_name": "file_credit_card_transaction_dispute_4829",
                   "arguments": json.dumps({"transaction_id": "txn_e9d195fe8e_001",
                                            "card_last_4_digits": "2791"})})
    check("C7 손님이 실행한 도구 출력이 증거로 인정된다",
          G._wev_deny_msgs(ms, call, specs) is None)
    # 증거가 정말 손님 쪽에만 있는지(테스트 자체의 부정 통제)
    only_user = [m for m in (s053[0].get("messages") or [])
                 if m.get("role") == "tool" and "Last 4 digits of card: 2791" in str(m.get("content") or "")]
    check("C7 부정 통제 — 그 증거는 손님 실행 결과에만 있다",
          bool(only_user) and all(m.get("requestor") == "user" for m in only_user))
else:
    check("C7 053 궤적 확보", False, "persisted 궤적 없음")

sp4 = [x for x in specs if x.get("id_key") == "card_last_4_digits"]
check("C7b 없는 필드를 가리키던 복구문이 사라졌다",
      bool(sp4) and "account record does NOT carry the digits" in sp4[0].get("feedback", ""))


# ── C9: 값을 받는 호출이 무엇인지는 환경 시그니처가 안다(표 없음) ─────────────────────
import types as _t


class _TK:
    def __init__(self, fns):
        self.tools = fns


def _f_dispute(transaction_id, card_last_4_digits, user_id):
    pass


def _f_close(credit_card_account_id, user_id):
    pass


_env = _t.SimpleNamespace(tools=_TK({"file_credit_card_transaction_dispute_4829": _f_dispute,
                                     "close_credit_card_account_7834": _f_close}),
                          user_tools=None)
check("C9 인자를 받는 도구를 시그니처에서 도출한다",
      G._arg_consumers(_env, "card_last_4_digits") == {"file_credit_card_transaction_dispute_4829"})
check("C9 해지 도구는 그 인자를 받지 않는다(048의 우회가 무의미한 이유)",
      "close_credit_card_account_7834" not in G._arg_consumers(_env, "card_last_4_digits"))
check("C9 환경이 없으면 조용히 빈 집합(무해)", G._arg_consumers(None, "card_last_4_digits") == set())


# ── 배선: `_ap_regen`은 (문구, tag) 두 인자를 받는다 ─────────────────────────────────
# 20260805q 실측: tag 없이 부른 C12가 TypeError로 **조용히 no-op**이 되어, 표적을 8번 잡고도
# 메시지가 한 번도 전달되지 않았다. py_compile은 이걸 못 본다 — 호출부를 AST로 센다.
import ast as _ast

_tree = _ast.parse(src)
_bad = [n.lineno for n in _ast.walk(_tree)
        if isinstance(n, _ast.Call) and getattr(n.func, "id", None) == "_ap_regen"
        and len(n.args) < 2]
check("배선 — `_ap_regen` 호출이 전부 tag를 넘긴다", not _bad, str(_bad[:4]))

print()
print("결과: %s" % ("ALL PASS" if not fail else "FAIL %d — %s" % (len(fail), fail)))
sys.exit(1 if fail else 0)
