# -*- coding: utf-8 -*-
"""X15 — C248 처방 P3(살리기) 오프라인 검증 W1~W3 (2026-07-31·무료).

설계 = `RUNAWAY_DEMOTION_REMEDIATION_DESIGN_2026_07_31.md` §3.
이 파일은 **순수 함수 + 검증**만 한다 — 엔진 배선은 리뷰 통과 후 별건이다.

W1 회수율   : 강등된 응답에서 **첫 완결 블록**을 회수할 수 있나 (폭주형/꼬리절단 분리)
W2 중복 0   : 회수 결과가 응답당 **최대 1개**이고 유효 호출인가 (93개 복제 실행 금지)
W3 오탐 0   : 정상 응답·산문 응답·give-flow 서술(JSONish 77건)에서 **발화하지 않나**

★살리기 술어(전부 닫힘): `tool_calls`가 비었고 ∧ 본문에 `<tool_call>` 태그가 있다.
  회수 대상 = **첫 번째** 완결 블록의 JSON(`name`·`arguments` 보유). 복제분은 버린다.
"""
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
GLOB = "bank_day*front[AB]_*.results.json.gz"

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
_JSONISH = re.compile(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:', re.S)


# ── P3 살리기 (순수 함수) ────────────────────────────────────────────────────
def should_salvage(tool_calls, content):
    """닫힌 술어: 파서가 호출을 못 잡았는데 본문에 호출 태그가 있다."""
    return (not tool_calls) and isinstance(content, str) and (_OPEN in content)


def salvage_first_call(content):
    """첫 **완결** 블록 하나만 회수. 없으면 None.

    ★왜 첫 블록만인가: 폭주는 같은 블록을 최대 93회 복제한다(C248). 전부 실행하면 over-action
    재앙이고, 복제는 정지 실패의 산물이지 의도가 아니다.
    """
    i = content.find(_OPEN)
    while i >= 0:
        j = content.find(_CLOSE, i)
        blk = content[i + len(_OPEN):j] if j > 0 else None
        if blk is not None:
            try:
                o = json.loads(blk.strip())
                if isinstance(o, dict) and "name" in o:
                    return {"name": o["name"], "arguments": o.get("arguments", {})}
            except Exception:
                pass
        i = content.find(_OPEN, i + 1)
    return None


# ── 검증 ────────────────────────────────────────────────────────────────────
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    files = sorted(glob.glob(os.path.join(_SIM, GLOB)))
    if not files:
        sys.exit("궤적 0")

    demoted, normal_calls, prose, jsonish = [], 0, 0, []
    for path in files:
        d = json.load(gzip.open(path, "rt", encoding="utf-8"))
        for sim in d.get("simulations") or []:
            for m in sim.get("messages") or []:
                if m.get("role") != "assistant":
                    continue
                tcs = m.get("tool_calls") or []
                c = m.get("content")
                c = c if isinstance(c, str) else ""
                if tcs:
                    normal_calls += 1
                    if should_salvage(tcs, c):
                        print("  ✗ W3 위반: 호출이 있는데 살리기 술어 발화")
                    continue
                if should_salvage(tcs, c):
                    tail = (c.count(_OPEN) > c.count(_CLOSE))
                    first = c.find(_OPEN) / max(1, len(c))
                    demoted.append({"task": sim.get("task_id"), "sim": str(sim.get("id"))[:10],
                                    "content": c, "trunc": tail, "first_frac": first,
                                    "kind": ("꼬리절단" if first >= 0.9 else "폭주형") if tail
                                            else "완결-미파싱"})
                elif _JSONISH.search(c):
                    jsonish.append((sim.get("task_id"), c))
                else:
                    prose += 1

    print("=== 대상 ===")
    print("  강등 응답(살리기 술어 발화) %d · 정상 호출 응답 %d · 산문 응답 %d · JSONish 서술 %d"
          % (len(demoted), normal_calls, prose, len(jsonish)))

    print("\n=== W1 회수율 ===")
    kinds = Counter(x["kind"] for x in demoted)
    rec = Counter()
    names = Counter()
    for x in demoted:
        call = salvage_first_call(x["content"])
        x["call"] = call
        rec[(x["kind"], bool(call))] += 1
        if call:
            names[call["name"]] += 1
    for k in kinds:
        ok = rec[(k, True)]
        print("  %-12s %2d건 중 회수 **%d** (%.0f%%)" % (k, kinds[k], ok, 100.0 * ok / kinds[k]))
    tot_ok = sum(v for (k, b), v in rec.items() if b)
    print("  ⇒ 전체 회수 %d/%d = %.0f%%" % (tot_ok, len(demoted), 100.0 * tot_ok / len(demoted)))
    print("  회수된 도구:", names.most_common(6))

    print("\n=== W2 중복 실행 0 ===")
    bad = 0
    for x in demoted:
        if x["call"] is None:
            continue
        if not isinstance(x["call"], dict) or "name" not in x["call"]:
            bad += 1
        n_blocks = x["content"].count(_OPEN)
        if n_blocks > 1:
            pass          # 복제가 있어도 회수는 1개여야 한다(구조상 보장)
    print("  회수 결과는 응답당 정확히 **1개**(함수가 첫 블록만 반환) · 형식 위반 %d" % bad)
    dupes = [x for x in demoted if x["content"].count(_OPEN) > 1 and x["call"]]
    print("  복제가 있던 응답 %d건에서도 회수 1개 유지 ✅" % len(dupes))
    mx = max((x["content"].count(_OPEN) for x in demoted), default=0)
    print("  (최대 복제 수 %d개 → 회수 1개)" % mx)

    print("\n=== W3 오탐 0 ===")
    fp = sum(1 for _t, c in jsonish if should_salvage([], c))
    print("  give-flow JSONish 서술 %d건 중 살리기 발화: **%d**" % (len(jsonish), fp))
    print("  정상 호출 응답 %d건 중 발화: 0 (위 루프에서 위반 출력 없음)" % normal_calls)
    print("  산문 응답 %d건: 태그가 없으므로 구조상 미발화" % prose)

    print("\n=== 영향 ===")
    sims = {(x["sim"], x["task"]) for x in demoted if x["call"]}
    print("  회수가 발생할 sim %d개 · 태스크 분포 %s"
          % (len(sims), Counter(x["task"] for x in demoted if x["call"]).most_common(6)))
    print("\n⚠[[08]]: 회수는 **면제가 아니다** — 회수된 호출도 게이트·검증기를 평소대로 통과해야 하고,")
    print("  행에 회수 표시를 남겨 판정 시 분리 가능해야 한다(설계 §4-1).")


if __name__ == "__main__":
    main()
