# -*- coding: utf-8 -*-
"""x130 — db_match가 왜 실패했는지를 **replay로** 짚는다 (유료 0·모델 0).

`db_match`는 해시 동등성이라(`evaluator_env.py:118-126`) 실패해도 *어디서* 갈렸는지 말해 주지
않는다. 그런데 갈린 자리는 원리적으로 계산 가능하다 — 상태를 바꾸는 호출만이 해시를 움직이고,
어느 호출이 상태를 바꾸는지는 환경에 물어보면 된다(x129가 그 절반).

이 도구가 하는 일:

  · gold 액션을 하나씩 실행하며 해시를 찍어 **DB를 바꾸는 gold 호출**만 남긴다
  · 우리 궤적에서도 같은 판정으로 **DB를 바꾼 호출**만 남긴다
  · 둘을 맞대어 **빠뜨린 것 / 더 한 것**을 낸다 ← 이것이 db_match 실패의 내용이다

x120 §C는 제출 유형 집합만 봤다. 그래서 led_j task_100처럼 *"제출은 완벽한데 0점"* 인 경우를
설명하지 못했다 — 빠진 것이 제출이 아니라 **디스패처 계좌조회**(read인데 DB를 바꾼다)였기
때문이다. 여기서는 종류를 가리지 않고 상태-변경 전부를 본다.

⚠이 도구는 **진단**이지 저작이 아니다([[23]]). gold이 무엇을 요구하는지 읽는 것은 실패 원인을
아는 데 쓰고, A2에 그 값을 옮겨 적는 데 쓰지 않는다.

usage: x130_db_divergence.py --dir bank_stack_led_20260807j [--tasks task_100,task_101]
"""

import argparse
import glob
import gzip
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))


def _load(dirname):
    cands = [os.path.join(REPO, "reports", "facet_rft_2026", "sim_results", dirname + ".json.gz")]
    cands += glob.glob(os.path.join(os.path.expanduser("~"), "scratch", "tau2-bench",
                                    "data", "simulations", dirname, "results.json"))
    for p in cands:
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt", encoding="utf-8", errors="replace") as fh:
                return json.load(fh), p
    raise SystemExit("no results for %r" % dirname)


def _args_of(tc):
    a = tc.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a or {}


def _sig(name, args):
    """호출의 동일성 표지 — 이름 + 인자(정렬). 인자가 다르면 다른 상태 변경이다."""
    return "%s(%s)" % (name, json.dumps(args, sort_keys=True, ensure_ascii=False)[:160])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--domain", default="banking_knowledge")
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()

    from tau2.registry import registry
    get_env = registry.get_env_constructor(a.domain)
    tasks = {t.id: t for t in registry.get_tasks_loader(a.domain)()}

    data, src = _load(a.dir)
    print("source: %s\n" % src)
    want = set(t.strip() for t in a.tasks.split(",") if t.strip())

    gold_cache = {}
    for sim in data.get("simulations") or []:
        tid = sim.get("task_id")
        if want and tid not in want:
            continue
        task = tasks.get(tid)
        if task is None:
            continue

        # ── gold 쪽: DB를 바꾸는 액션만 추린다 ──────────────────────────────
        if tid not in gold_cache:
            env = get_env()
            init = task.initial_state
            if init is not None:
                env.set_state(initialization_data=getattr(init, "initialization_data", None),
                              initialization_actions=getattr(init, "initialization_actions", None),
                              message_history=getattr(init, "message_history", None) or [])
            prev, need = (env.get_db_hash(), env.get_user_db_hash()), []
            for act in ((task.evaluation_criteria.actions or [])
                        if task.evaluation_criteria else []):
                try:
                    env.make_tool_call(tool_name=act.name, requestor=act.requestor, **act.arguments)
                except Exception:
                    pass
                cur = (env.get_db_hash(), env.get_user_db_hash())
                if cur != prev:
                    need.append(_sig(act.name, act.arguments))
                prev = cur
            gold_cache[tid] = need
        need = gold_cache[tid]

        # ── 우리 쪽: **같은 해시 검정**을 적용한다 ──────────────────────────
        #   `_is_mutating_tool`은 선언을 읽을 뿐이라 `KB_search_dense`·`verify_identity`처럼
        #   선언상 mutating인데 해시를 안 바꾸는 것까지 '더함'으로 올린다(1차판이 그랬다).
        #   gold 쪽에 쓴 검정을 우리 쪽에도 그대로 써야 두 목록이 같은 뜻을 갖는다.
        env2 = get_env()
        init2 = task.initial_state
        if init2 is not None:
            env2.set_state(initialization_data=getattr(init2, "initialization_data", None),
                           initialization_actions=getattr(init2, "initialization_actions", None),
                           message_history=getattr(init2, "message_history", None) or [])
        prev2 = (env2.get_db_hash(), env2.get_user_db_hash())
        ours = []
        by_id = {m.get("id"): m for m in (sim.get("messages") or []) if m.get("role") == "tool"}
        for m in sim.get("messages") or []:
            for tc in (m.get("tool_calls") or []):
                nm = tc.get("name")
                out = by_id.get(tc.get("id"))
                if not nm or out is None or out.get("error"):
                    continue                                # 실패한 호출은 상태를 안 바꾼다
                try:
                    env2.make_tool_call(tool_name=nm,
                                        requestor=(out.get("requestor") or "assistant"),
                                        **_args_of(tc))
                except Exception:
                    pass
                cur2 = (env2.get_db_hash(), env2.get_user_db_hash())
                if cur2 != prev2:
                    ours.append(_sig(nm, _args_of(tc)))
                prev2 = cur2

        missing = [s for s in need if s not in ours]
        extra = [s for s in ours if s not in need]
        rw = (sim.get("reward_info") or {}).get("reward")
        print("=" * 96)
        print("== %s trial=%s ==  reward=%s" % (tid, sim.get("trial"), rw))
        print("   gold이 요구하는 상태변경 %d개 · 우리가 한 상태변경 %d개" % (len(need), len(ours)))
        for s in need:
            print("     %s %s" % ("✓" if s in ours else "✗ 빠짐", s))
        for s in extra:
            print("     ＋더함  %s" % s)
        if not missing and not extra:
            print("   ⇒ 상태변경 집합 일치 — db_match 실패라면 원인은 이 축이 아니다(인자 값 차이 등)")
        else:
            print("   ⇒ db_match 실패의 내용: 빠짐 %d · 더함 %d" % (len(missing), len(extra)))


if __name__ == "__main__":
    main()
