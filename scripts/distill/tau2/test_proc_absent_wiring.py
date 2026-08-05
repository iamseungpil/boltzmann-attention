# -*- coding: utf-8 -*-
"""Would the absence trigger have spoken in `task_048`, and would it have said the one thing missing?

The unit tests for `t2_procedure` run on hand-built declarations, which cannot show
whether the wiring reads a real conversation the way the engine will. This runs the two
helpers the patch added — `_unlocked_names`, `_quiet_turns` — over the persisted 048
trajectory, the run where the same deny printed ten times while the model called the
right tool eight times and failed on the unlock every time.

The assertion that matters is the last one: at the point where the loop begins, the
message the declaration produces must name the callable tool *and* say it has not been
unlocked. That single fact is what the node id never carried.
"""

import glob
import gzip
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

import t2_gate_patch as G          # noqa: E402
import t2_procedure as P           # noqa: E402
import gate_interpreter as GI      # noqa: E402

TASK = os.environ.get("T_TASK", "task_048")
fail = []


def check(name, ok, detail=""):
    print("  %-56s %s%s" % (name, "PASS" if ok else "FAIL", (" — " + detail) if detail else ""))
    if not ok:
        fail.append(name)


def objs(messages):
    """궤적 dict → 엔진이 보는 모양의 객체(속성 접근·tool_calls는 name/arguments/id)."""
    out = []
    for m in messages:
        tcs = [types.SimpleNamespace(name=tc.get("name"), arguments=tc.get("arguments"),
                                     id=tc.get("id")) for tc in (m.get("tool_calls") or [])]
        out.append(types.SimpleNamespace(role=m.get("role"), tool_calls=tcs or None,
                                         content=m.get("content"), id=m.get("id"),
                                         tool_call_id=m.get("tool_call_id"),
                                         error=m.get("error", False)))
    return out


def load(task):
    for p in sorted(glob.glob(os.path.join(
            HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
            "bank_smk_gpu*_20260805g.results.json.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for s in json.load(f).get("simulations") or []:
                if s.get("task_id") == task:
                    return s
    return None


sim = load(TASK)
if sim is None:
    print("  · %s 궤적 미존재(스모크 g 미영속) — 배선 검정 건너뜀" % TASK)
    sys.exit(0)

a2 = GI.load_domain_a2("banking_knowledge") or {}
procs = a2.get("procedures") or []
msgs = objs(sim["messages"])
pat = (a2.get("discoverable_name_check") or {}).get("pattern")

print("① 헬퍼가 실제 궤적을 엔진과 같게 읽는다 (%s · %d msg)" % (TASK, len(msgs)))
unl = G._unlocked_names(msgs, a2)
check("unlock 이름을 A2 선언으로 수집한다", "log_credit_card_closure_reason_4521" in unl,
      "%d종" % len(unl))
check("한 번도 unlock 안 된 이름은 빠져 있다",
      "get_closure_reason_history_8293" not in unl)

print("\n② 부재 조건이 실제로 성립한 구간이 있다")
clo = next((p for p in procs if "closure" in (p.get("id") or "")), None)
check("해지 절차 선언이 있다", clo is not None)
nodes = {t for n in (clo.get("nodes") or []) for t in (P._tools_of(n) or [])}
peak = max(G._quiet_turns(msgs[:i + 1], nodes) for i in range(len(msgs)))
check("절차-무호출이 K=3 이상 이어진 구간이 있다", peak >= 3, "최대 %d턴" % peak)

print("\n③ 그 지점의 문구가 **호출 가능한 이름 + 잠금 상태**를 말한다")
# livelock 시작 = disputes·pending_replacement는 됐고 prior_attempts가 안 된 상태
executed = {"check_card_closure_eligibility", "get_user_dispute_history_7291",
            "get_pending_replacement_orders_5765"}
msg = P.absent_note(clo, executed, unl, pat) or ""
check("절차가 활성으로 잡힌다", bool(P.active_procedures(procs, executed)))
check("문구가 노드 id가 아니라 **도구 이름**을 준다",
      "get_closure_reason_history_8293" in msg)
check("문구가 **unlock 안 됨**을 말한다", "has not been unlocked" in msg)
check("문구가 자연어 질의를 준다", "get closure reason history" in msg)
check("동렬이 아니므로 NEXT가 붙는다", "NEXT:" in msg)

print("\n④ 동렬일 때는 고르지 않는다 (설계 §2.3)")
cli = next((p for p in procs if "limit_increase" in (p.get("id") or "")), None)
many = P.absent_note(cli, {"check_cli_eligibility",
                           "submit_credit_limit_increase_request_7392"}, set(), pat) or ""
check("4개 동렬에서 NEXT를 붙이지 않는다", "NEXT:" not in many)
check("대신 목록을 준다", "in any order" in many)

print("\nRESULT: %s" % ("ALL PASS" if not fail else "FAIL %s" % fail))
sys.exit(1 if fail else 0)
