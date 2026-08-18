# -*- coding: utf-8 -*-
r"""`requires_reads` **배선 검정** — 문면만 있고 목록이 없으면 게이트는 영영 안 걸린다
(2026-08-18 · t7314 스모크 050 포렌식에서 발견 · 死배선 4호).

## 무엇이 있었나

  · 엔진에 read-선행 게이트가 있다 — `t2_scaffold_get.py:1572` (`T2_SG_REQREADS=1`·go_stack ON).
  · A3(`banking_knowledge.specific.json`)의 `check_cli_eligibility` 에 **거부 문면이 저작돼 있다**
    (`requires_reads_feedback`: *"[READ-FIRST] … Missing required reads: {missing}"*).
  · 그런데 그 문면을 트리거하는 **목록 `requires_reads` 가 세 층 어디에도 없다** ⇒ `d.get("requires_reads")`
    가 빈 값 ⇒ **게이트가 한 번도 안 걸렸다**.

라이브 결과(t7314 treat·task_050): `check_cli_eligibility` 가 **선행 조회 0회**인 채로
*"ELIGIBLE - all tier requirements (account age, cooldown, utilization, payment history) are satisfied"*
를 발급했고, 그 뒤 한도 증액이 그대로 승인됐다(분쟁·교체주문·납부이력 **미조회**).
우리 도구가 **가짜 통과증**을 낸 것이고 [[25]](우리 도구는 100% 옳아야 한다) 정면 위반이다.

## 불변식

  ① `requires_reads_feedback` 를 선언한 도구는 **`requires_reads` 도 비어있지 않게** 선언한다.
     (문면만 있는 것 = 죽은 문면 = 이 사고의 형태)
  ② `requires_reads` 의 모든 이름은 그 도구의 **문면에 축자로** 등장한다 — 두 곳이 갈라지면 실패.
  ③ 이름에 `_NNNN` 접미사가 없다 — 엔진은 `_eff_tool_name` 으로 접미사를 지우고 대조한다.
  ④ **층 동기화**([[24]]): A3(`specific`) 와 `gate.json` 의 값이 같다.
  ⑤ **양성 대조**: 선언을 지운 사본에서 ①이 실제로 잡힌다.

오프라인 전용. 실행: py -3 test_requires_reads_wired.py
"""
import copy
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAILED = []


def chk(c, label):
    print(("  OK   " if c else "  FAIL ") + label)
    if not c:
        FAILED.append(label)


def load(name):
    p = os.path.join(HERE, "a2", name)
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else {}


def tools_of(doc):
    return {t.get("name"): t for t in (doc.get("scaffold_get_tools") or [])
            if isinstance(t, dict) and t.get("name")}


def audit(tools):
    """①②③ 위반 목록 — (도구, 사유)."""
    bad = []
    for nm, t in sorted(tools.items()):
        fb = t.get("requires_reads_feedback")
        rr = t.get("requires_reads") or []
        if fb and not rr:
            bad.append((nm, "문면은 있는데 requires_reads 가 없다(죽은 문면)"))
            continue
        for r in rr:
            if fb and r not in fb:
                bad.append((nm, "목록의 %r 가 문면에 없다(두 곳이 갈라졌다)" % r))
            if re.search(r"_\d+$", str(r)):
                bad.append((nm, "%r 에 _NNNN 접미사 — 엔진은 접미사를 지우고 대조한다" % r))
    return bad


def main():
    print("test_requires_reads_wired — 문면과 목록이 함께 있는가(死배선 4호)")
    spec, gate = load("banking_knowledge.specific.json"), load("banking_knowledge.gate.json")
    ts, tg = tools_of(spec), tools_of(gate)
    chk(bool(ts), "A3(specific) 에서 scaffold_get_tools 를 읽었다 (%d개)" % len(ts))

    bad = audit(ts)
    chk(not bad, "①②③ 위반 0 — %s" % ("없음" if not bad else
                                      " · ".join("%s: %s" % b for b in bad)))

    diff = [n for n in sorted(set(ts) | set(tg))
            if (ts.get(n) or {}).get("requires_reads") != (tg.get(n) or {}).get("requires_reads")]
    chk(not diff, "④ 층 동기화 — specific ↔ gate 의 requires_reads 가 같다 (%s)"
        % (", ".join(diff) or "전부 동일"))

    # ⑤ 양성 대조 — 선언을 지우면 ①이 잡혀야 한다
    probe = copy.deepcopy(ts)
    victim = next((n for n, t in probe.items() if t.get("requires_reads_feedback")), None)
    if victim:
        probe[victim].pop("requires_reads", None)
        chk(any(b[0] == victim for b in audit(probe)),
            "⑤ 양성 대조 — %s 의 선언을 지우면 검정이 잡는다" % victim)
    else:
        chk(False, "⑤ 양성 대조 불가 — requires_reads_feedback 를 가진 도구가 없다")

    print("")
    print("RESULT: %s" % ("ALL PASS" if not FAILED else "FAIL %d" % len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
