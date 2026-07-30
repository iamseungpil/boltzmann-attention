#!/usr/bin/env python3
"""X10 실행 잔여 측정 — multi-act 게이트 3(우선순위)의 대조항 (2026-07-30).

대상 = `bank_day*front*` 전체(day2~9c). 배경: `MULTIACT_DECOMPOSITION_REVIEW §6` 게이트3 = "execution 잔여와의 우선순위 비교".
C140/C136이 **"절차를 다 retrieved 했는데 `apply_flag` 0 실행"**을 진짜 잔여로 지목했으나
**크기가 측정된 적이 없다**. 게이트1(인식)만 통과한 상태로 아키텍처에 착수하면 더 큰 조각을
놓칠 수 있다.

★측정 단위 문제(정직): 인식 잔여는 **발화-수준**(act 분류), 실행 잔여는 **sim-수준**(행동 완결)
이라 한 자에 놓을 수 없다. 공통 통화는 **실패 sim의 원인 귀속**으로 잡는다 — 실패한
`action_checks`를 원인별로 분류하고 그 census를 우선순위 근거로 쓴다.

분류 규칙(전부 궤적서 기계 판정·[[08]] per-case 보존):
  OPERAND   요구 도구를 **호출했으나** action_match 실패 → 인자·참조 오류(인식/operand 계열)
  EXEC_GAP  요구 도구가 **discoverable**(이름에 숫자 접미사 = KB에서 찾아야 함)인데 미호출 +
            그 이름이 **도구 출력에 등장**했다 → 알았는데 안 했다 = **C140이 지목한 실행 잔여**
  DECISION  요구 도구가 **항상 스키마에 있는 도구**(접미사 없음·프레임워크 공통 포함)인데 미호출
            → "알았는데 안 함"이 자동 참이라 EXEC_GAP과 구분해야 한다(**초판 결함 교정**:
            `transfer_to_human_agents`는 언제나 스키마에 있으므로 `name_in_outputs`가 무의미하고,
            실패의 성격은 **판단/준수**다. 초판은 이 11건을 EXEC_GAP에 섞었다)
  DISCOVERY 요구 도구 미호출 + 이름이 출력에 **등장하지 않음** → 발견 실패(정보 도달 문제)
  USER_SIDE 요구 action의 requestor가 user → 에이전트가 손님에게 실행시켜야 하는 계열
            (도구 이름 등장 여부를 같이 보고 EXEC/DISCOVERY로 세분)

용법: py -3 x10_execution_residual.py [--json out.json]
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
_SUF = re.compile(r"_\d{3,4}$")

# ★C244 자기정정: `unlock_`/`list_` 호출도 `agent_tool_name`을 갖는다. 그걸 "그 도구를 호출했다"로
#   세면 (a)X10에서 미호출을 OPERAND로 오분류 (b)X11에서 인자 없는 unlock을 짝지어 전 인자를
#   MISSING으로 오계상한다. **unlock/list는 실행이 아니다**(prekb `_effective_fams`·C159 교훈).
#   ⇒ **실행으로 세는 것은 dispatch(call_)·give_ 계열뿐**. 이름은 A2에서 읽는다(리터럴 0).
_A2_PATH = os.path.join(_HERE, "a2", "banking_knowledge.gate.json")


def _dispatch_cfg():
    try:
        ep = (json.load(open(_A2_PATH, encoding="utf-8")).get("eplan") or {})
    except Exception:
        return set(), set()
    exec_names = {ep.get("dispatch_tool")}
    nonexec = {ep.get("unlock_tool"), ep.get("list_tool")}
    return {x for x in exec_names if x}, {x for x in nonexec if x}


_EXEC_DISP, _NONEXEC_DISP = _dispatch_cfg()



def fam(n):
    return _SUF.sub("", str(n or ""))


def called_names(msgs):
    """궤적에서 실제 호출된 도구 fam 집합(디스패처 내부 이름 포함·requestor 무관)."""
    out = set()
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name")
            out.add(fam(nm))
            if nm in _NONEXEC_DISP:
                continue                       # ★C244: unlock/list = 실행 아님(inner 미계상)
            a = tc.get("arguments") or {}
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            for k in ("agent_tool_name", "discoverable_tool_name", "user_tool_name"):
                if isinstance(a, dict) and a.get(k):
                    out.add(fam(a[k]))
    return out


def name_in_outputs(msgs, name):
    """도구 이름(또는 그 fam)이 **어떤 도구 출력**에 등장했나 = 에이전트가 알 수 있었나.

    ★C140의 술어를 그대로 옮긴 것: 절차 문서가 도구 이름을 명시했다면 '알았다'로 본다.
    """
    f = fam(name)
    for m in msgs:
        if m.get("role") != "tool" or m.get("error"):
            continue
        c = str(m.get("content") or "")
        if f and (f in c or str(name) in c):
            return True
    return False


def classify(sim):
    """실패한 action_check 각각을 원인 분류. 반환 [(cls, name, requestor)]."""
    msgs = sim.get("messages") or []
    ri = sim.get("reward_info") or {}
    called = called_names(msgs)
    out = []
    for ac in (ri.get("action_checks") or []):
        if ac.get("action_reward"):
            continue                                  # 통과분은 대상 아님
        act = ac.get("action") or {}
        nm = act.get("name")
        # 디스패처 경유면 내부 이름이 진짜 요구 도구
        args = act.get("arguments") or {}
        inner = None
        if isinstance(args, dict):
            for k in ("agent_tool_name", "discoverable_tool_name", "user_tool_name"):
                if args.get(k):
                    inner = args[k]
                    break
        target = inner or nm
        req = act.get("requestor") or "assistant"
        discoverable = bool(_SUF.search(str(target)))   # 숫자 접미사 = KB 발견 필요
        if fam(target) in called:
            cls = "OPERAND"                           # 호출했는데 불일치
        elif not discoverable:
            cls = "DECISION"                          # 항상 스키마에 있음 → 판단/준수 실패
        elif name_in_outputs(msgs, target):
            cls = "EXEC_GAP"                          # 알았는데 안 함 (C140 지목)
        else:
            cls = "DISCOVERY"                         # 이름을 못 찾음
        out.append((cls, fam(target), req))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # ⚠초판 docstring은 "day6~9c"라 적었으나 glob은 **day 전체**를 잡는다(day2·day3 포함).
    #   범위가 넓은 게 데이터상 유리하므로 glob을 유지하고 **표기를 실제와 맞춘다**.
    files = sorted(glob.glob(os.path.join(_SIM, "bank_day*front*.results.json.gz")))
    tot_sim = fail_sim = 0
    cls_cnt = Counter()
    per_sim_cls = Counter()
    by_tool = defaultdict(Counter)
    rows = []
    for path in files:
        try:
            data = json.load(gzip.open(path, "rt", encoding="utf-8"))
        except Exception as e:
            print(f"[warn] {os.path.basename(path)}: {e}", file=sys.stderr)
            continue
        for sim in data.get("simulations") or []:
            ri = sim.get("reward_info") or {}
            rw = ri.get("reward")
            if rw is None:
                continue
            tot_sim += 1
            if rw >= 1.0:
                continue
            fail_sim += 1
            cs = classify(sim)
            if not cs:
                per_sim_cls["NO_FAILED_ACTION_CHECK"] += 1
                rows.append({"file": os.path.basename(path), "sim": sim.get("id"),
                             "classes": [], "note": "실패 sim인데 실패 action_check 0 "
                                                    "(db/communicate 축 실패)"})
                continue
            for c, t, r in cs:
                cls_cnt[c] += 1
                by_tool[c][t] += 1
            # sim 단위 지배 원인 = 최빈 클래스(동수면 EXEC_GAP 우선 표기 안 함·튜플로 기록)
            per_sim_cls["+".join(sorted({c for c, _, _ in cs}))] += 1
            rows.append({"file": os.path.basename(path), "sim": sim.get("id"),
                         "classes": [{"cls": c, "tool": t, "requestor": r} for c, t, r in cs]})

    print("=" * 76)
    print("X10 실행 잔여 — day 전체(day2~9c) front 런 실패 sim 원인 census")
    print("=" * 76)
    print(f"sim {tot_sim} · 실패 {fail_sim} ({100.0 * fail_sim / tot_sim if tot_sim else 0:.0f}%)")
    print()
    n = sum(cls_cnt.values())
    print(f"실패 action_check {n}건의 원인 분류:")
    for c, k in cls_cnt.most_common():
        print(f"  {c:10s} {k:4d}건 ({100.0 * k / n if n else 0:5.1f}%)")
    print()
    print("sim 단위(한 sim이 여러 클래스를 가질 수 있음):")
    for c, k in per_sim_cls.most_common():
        print(f"  {c:32s} {k:3d} sim")
    print()
    for c in ("EXEC_GAP", "DECISION", "DISCOVERY", "OPERAND"):
        if by_tool[c]:
            print(f"{c} 상위 도구: {dict(by_tool[c].most_common(6))}")
    print()
    print("=" * 76)
    print("게이트3 판정 재료")
    print("=" * 76)
    for c in ("OPERAND", "DISCOVERY", "EXEC_GAP", "DECISION"):
        k = cls_cnt[c]
        lab = {"OPERAND": "호출했으나 인자·참조 오류", "DISCOVERY": "이름을 못 찾음",
               "EXEC_GAP": "discoverable 알았는데 안 함(C140 지목)",
               "DECISION": "항상 스키마 있는 도구 미호출 = 판단/준수"}[c]
        print(f"{c:10s} {k:4d}건 = {100.0 * k / n if n else 0:5.1f}%  ({lab})")
    print()
    print("⚠단위 비교의 한계(정직): 인식 잔여(multi-act)는 **발화-수준**, 이 census는")
    print("  **action_check-수준**이라 한 자에 놓을 수 없다. 게이트3은 '어느 쪽이 큰가'를")
    print("  묻지만 이 표는 **실행-측 내부 구성**만 준다 — 인식 잔여를 같은 자로 옮기려면")
    print("  실패 sim마다 '인식 오류가 원인인가'를 궤적서 따로 귀속해야 한다(별건).")
    print("⚠[[08]]: 이 census는 집계다. 결론 전 EXEC_GAP 사례 2~3건 정독 필수.")

    if args.json:
        json.dump({"tot_sim": tot_sim, "fail_sim": fail_sim, "cls": dict(cls_cnt),
                   "per_sim": dict(per_sim_cls),
                   "by_tool": {k: dict(v) for k, v in by_tool.items()}, "rows": rows},
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
