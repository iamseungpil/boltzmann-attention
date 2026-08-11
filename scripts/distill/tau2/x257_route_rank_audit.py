# -*- coding: utf-8 -*-
r"""x257 — 라우팅 rank 기계 추출 + **억제가 범인인지** 오프라인 판정 (유료 0 · 모델 0 · GPU 0).

## 왜 (설계 §7-2 · 원장 C427)

배달이 죽는 자리에 용의자가 **둘**이다:
  ⓐ `identical demand suppressed` — 지문이 **바이트 동일**만 보고 상태 변화를 안 본다(59회)
  ⓑ **15-way 배타 체인** — 같은 `tool_call` 에 하나만 나가고 `rw_fb`(CALL_FORM)는 **11위**다

C419⒠ 는 ⓐ 하나로 돌렸고 C427 이 그것을 부분 귀속이라 정정했다. 그런데 **어느 쪽이 실제로
그 턴을 삼켰는지는 아직 안 쟀다.** 고치기 전에 가른다([[62]] — 재고 나서 짓는다).

## 무엇을 하나

1. **rank 표를 코드에서 기계 추출**한다(사람이 옮겨 적지 않는다 — 두 벌이 되면 갈린다·[[03b]]).
2. `bank_cf_20260811b` 의 099 궤적에서 **커밋 히스토리만으로** 억제 지문 `_sig` 의 성분을
   각 턴마다 재계산해, **unlock 직후 턴에 지문이 바뀌었는지** 본다.
   - 지문이 **안 바뀌었으면** → 억제가 삼켰다(ⓐ). 지문 수리가 옳은 처방이다.
   - 지문이 **바뀌었으면** → 억제는 그 턴에 못 울었다 ⇒ 범인은 ⓑ(또는 제3)이고,
     지문을 고쳐도 배달률은 안 산다.

⚠재계산은 **엔진 함수를 그대로 불러서** 한다(`_executed_tool_names`). 우리가 다시 구현하면
   그 순간 두 벌이 된다.

실행: python x257_route_rank_audit.py
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

import t2_gate_patch as G                                        # noqa: E402

RES = "/home/woori/scratch/tau2-bench/data/simulations/bank_cf_20260811b/results.json"
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()


def rank_table():
    """체인 순서를 **소스에서** 뽑는다. 앞의 두 분기(`do_gate`·`main_prov`)를 포함해 센다."""
    chain = re.findall(r"elif (\w+_fb) is not None and c is \1\[0\]", SRC)
    names = dict(re.findall(r'\("([a-z_]+)", (\w+_fb)\)', SRC))
    rows = [(1, "do_gate", "gate", "정책 게이트 거부"),
            (2, "main_prov", "main_prov", "날조 값 provenance")]
    lbl = {v: k for k, v in names.items()}
    for i, v in enumerate(chain):
        rows.append((i + 3, lbl.get(v, "?"), v, ""))
    return rows


class _M(object):
    """커밋 메시지의 얇은 어댑터 — 엔진 함수가 기대하는 속성만 준다."""
    def __init__(s, d):
        s.role = d.get("role")
        s.content = d.get("content")
        s.error = bool(d.get("error"))
        s.id = d.get("id") or d.get("tool_call_id")
        s.tool_calls = [_TC(t) for t in (d.get("tool_calls") or [])]


class _TC(object):
    def __init__(s, d):
        f = d.get("function") or {}
        s.name = f.get("name") or d.get("name")
        a = f.get("arguments") if f else d.get("arguments")
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except Exception:
                a = {}
        s.arguments = a or {}
        s.id = d.get("id")


def main():
    print("== 라우팅 rank (코드에서 기계 추출) ==")
    for rk, tag, var, note in rank_table():
        print("  %2d  %-16s %-12s %s" % (rk, tag, var, note))
    tags = {t: rk for rk, t, _v, _n in rank_table()}
    print("\n  ⇒ wev(T2_ARG_EMPTY) = %s · resolve_write(T2_CALL_FORM) = %s"
          % (tags.get("wev"), tags.get("resolve_write")))

    if not os.path.exists(RES):
        print("\n(결과 파일 없음 — 지문 판정은 리모트에서)")
        return
    d = json.load(open(RES, encoding="utf-8"))
    print("\n== 억제 지문이 unlock 직후에 바뀌었나 (커밋 히스토리만·엔진 함수 재사용) ==")
    for s in sorted([x for x in d["simulations"] if x["task_id"] == "task_099"],
                    key=lambda x: x.get("trial")):
        msgs = [_M(m) for m in s["messages"]]
        unlock = None
        for i, m in enumerate(msgs):
            for tc in m.tool_calls:
                if (tc.name or "") == "unlock_discoverable_agent_tool":
                    unlock = i
        if unlock is None:
            print("  t%s (unlock 없음)" % s.get("trial"))
            continue
        # 지문 성분 중 **상태에 해당하는 둘**만 본다(표적·요건은 그 사이 안 바뀐다)
        def comp(upto):
            ex = G._executed_tool_names(msgs[:upto], None)
            nuser = sum(1 for m in msgs[:upto] if m.role == "user")
            return len(ex), nuser, ex
        before = comp(unlock + 1)          # unlock 호출 직전까지
        after = comp(unlock + 3)           # unlock 결과가 들어온 뒤
        same = (before[0], before[1]) == (after[0], after[1])
        called = any((tc.name or "") == "call_discoverable_agent_tool"
                     for m in msgs for tc in m.tool_calls)
        print("  t%s reward=%s  unlock@%d  len(executed) %d→%d · user %d→%d  "
              "⇒ 지문 %s  · call 했나 %s"
              % (s.get("trial"), s["reward_info"]["reward"], unlock,
                 before[0], after[0], before[1], after[1],
                 "**동일(억제 가능)**" if same else "변함(억제 불가)", called))
        newly = after[2] - before[2]
        if newly:
            print("       새로 '실행됨' 으로 잡힌 이름: %s" % ", ".join(sorted(newly)))
    print("\n※ 전부 '변함' 이면 억제는 그 턴에 못 울었다 ⇒ 지문 수리는 배달률을 안 산다."
          "\n  '동일' 이 있으면 그 시행은 억제가 삼킨 것이고 지문 수리가 정확한 처방이다.")


if __name__ == "__main__":
    main()
