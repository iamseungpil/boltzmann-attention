# -*- coding: utf-8 -*-
r"""x583 — 쓰기 도구 **발견 비용**을 잰다: 이름이 이미 배달됐는데 안 부르나 (모델 0 · 무료 · 계수만).

## 왜 (2026-08-28 밤)

`072#s373753` 이 t7375 에서 1.0, t7376 에서 0.0 인데 **코드가 같다**(29f09caf..66be7987 diff =
sim_results gz 뿐). 두 런 다 도구총액은 옳게 `14.0 · 3.5` 를 냈다 — 우리 총액 수리는 양쪽 다
작동했다. 갈린 것은 그 뒤다: 통과분은 msg[56] 에서 `apply_checking_account_credit_5829` 를
불렀고, 실패분은 58 msg 동안 KB 검색만 하다 손님이 끊었다.

⛔먼저 세운 가설 둘을 실물이 반증했다(기록):
  ⒜ *"user-sim 문면 차이(굵게 vs 평문·msg[5])가 원인"* → 분기의 **시작점**일 뿐 [[21]] 상 면책 불가.
  ⒝ *"소진 넛지(T2_SEARCH_EXHAUST)가 통과분을 살렸다"* → **순서가 반대다.** 넛지는 `msgs=60`,
     크레딧 적용은 `msg[56]` = **pass 뒤에 왔다.**

## 무엇을 세나 (닫힌 술어뿐 · gold 무참조)

그 도구가 필요한 sim 마다:
    문서에배달   그 이름이 **role=tool 메시지 본문**에 처음 나온 index (= 문맥에 들어온 시점)
    호출        `call_discoverable_agent_tool` 인자에 그 이름이 처음 들어간 index
검색 횟수(`KB_search*`)를 나란히 둔다. reward 는 참고 열이며 술어에 안 들어간다([[23]]).

## 무엇이 나왔나 (6런 · 34 sim)

    닿은 것 14 (reward=1 이 6)  ·  못 닿은 것 20 (reward=1 이 **0**)
    ★못 닿은 20 중 이름이 **이미 배달돼 있던 것 = 20/20**
    검색 횟수: 닿은 쪽 평균 5.1  <  못 닿은 쪽 평균 7.2   (더 찾아서 닫히지 않는다·[[63]])

=> 정보는 문맥에 있는데 안 쓴다 = **능력 결손이 아니라 부하**다([[62]] §1.4). 072 계열은
   `문서에배달=3` — 대화 거의 첫머리에 이름을 받고도 못 부른다.
   레버의 자리는 **전달**이고, 재료는 이미 `t2_gate_patch.py` 의 `_cand9`
   (레지스트리 ∩ 배달된 텍스트 − 이미 호출/해제)에 있다. 지금은 `_stubs>=2 AND _resign`
   뒤에 갇혀 있어 **낭비를 먼저 해야 도착한다**.
"""
import collections
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results"))
TOOL = "apply_checking_account_credit_5829"
TAGS = ["bank_t7375_072_20260828", "bank_t7376_treat_20260828", "bank_t7372_control_20260828",
        "bank_t7369_072_20260828", "bank_t7370_radius_20260828", "bank_t7368_hard0_20260827"]


def scan(tag, tool):
    p = os.path.join(BASE, tag + ".results.json.gz")
    if not os.path.exists(p):
        return []
    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
        sims = json.load(f).get("simulations") or []
    out = []
    for s in sims:
        msgs = s.get("messages") or []
        reach = deliv = None
        nkb = 0
        for i, m in enumerate(msgs):
            if m.get("role") == "tool" and tool in str(m.get("content") or "") and deliv is None:
                deliv = i
            for tc in (m.get("tool_calls") or []):
                nm = str(tc.get("name") or "")
                a = tc.get("arguments")
                a = a if isinstance(a, str) else json.dumps(a or {})
                if nm.startswith("KB_search"):
                    nkb += 1
                if tool in a and reach is None and "unlock" not in nm:
                    reach = i
        if deliv is None and reach is None:
            continue
        out.append({"tag": tag, "sim": "%s#s%s" % (s.get("task_id"), s.get("seed")),
                    "reward": (s.get("reward_info") or {}).get("reward"),
                    "delivered_at": deliv, "called_at": reach, "n_kb": nkb, "n_msgs": len(msgs)})
    return out


def main(argv=None):
    a = (argv or sys.argv[1:])
    tool = a[0] if a else TOOL
    tags = a[1:] or TAGS
    rows = []
    for t in tags:
        rows.extend(scan(t, tool))
    if not rows:
        print("행 0 — 재료를 못 읽었다"); return 1
    print("표적 도구: %s" % tool)
    print("%-8s %-22s %-5s %-11s %-8s %-6s %s"
          % ("런", "sim", "rew", "문서에배달", "호출", "KB검색", "msgs"))
    for r in rows:
        print("%-8s %-22s %-5s %-11s %-8s %-6d %d"
              % (r["tag"].split("_")[1], r["sim"], r["reward"],
                 r["delivered_at"] if r["delivered_at"] is not None else "-",
                 r["called_at"] if r["called_at"] is not None else "X",
                 r["n_kb"], r["n_msgs"]))
    ok = [r for r in rows if r["called_at"] is not None]
    no = [r for r in rows if r["called_at"] is None]
    had = [r for r in no if r["delivered_at"] is not None]
    print("")
    print("닿은 것 %d (reward=1 이 %d) · 못 닿은 것 %d (reward=1 이 %d)"
          % (len(ok), sum(1 for r in ok if r["reward"] == 1.0),
             len(no), sum(1 for r in no if r["reward"] == 1.0)))
    print("★못 닿은 sim 중 이름이 **이미 배달돼 있던 것: %d / %d**" % (len(had), len(no)))
    if ok:
        print("  KB검색 평균 — 닿은 쪽 %.1f · 못 닿은 쪽 %.1f"
              % (sum(r["n_kb"] for r in ok) / len(ok),
                 sum(r["n_kb"] for r in no) / max(1, len(no))))
    dst = os.path.join(BASE, "..", "x583_write_tool_discovery.json")
    with io.open(os.path.normpath(dst), "w", encoding="utf-8") as f:
        json.dump({"probe": "x583_write_tool_discovery", "date": "2026-08-28",
                   "tool": tool, "tags": tags, "rows": rows,
                   "limits": ["reward 는 참고 열 — 술어에 안 들어간다([[23]]).",
                              "'배달' 은 role=tool 본문의 축자 등장이지 모델이 읽었다는 증거가 아니다.",
                              "한 도구만 본다 — 다른 쓰기 도구로 일반화하려면 argv 로 다시 돌려라."]},
                  f, ensure_ascii=False, indent=1)
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
