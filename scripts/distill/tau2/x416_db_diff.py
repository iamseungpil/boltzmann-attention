# -*- coding: utf-8 -*-
r"""x416 - **올바른 실패 단위**로 다시: DB 채점 태스크는 변이(write) 집합만이 점수를 만든다

## 왜 (2026-08-19 발견)
`reward_basis` 실측: **DB 35 sim · ACTION 4 · 없음 1**. 그리고 미매치 gold 가 있는데도 통과한 sim 이 2건
(017 t1 · 050 t1). ⇒ **미매치 gold 는 실패 원인이 아니다.** DB 태스크의 실패 단위는 *최종 DB 상태*이고,
그것을 만드는 것은 **성공한 변이 호출의 집합**뿐이다. read 는 아무리 놓쳐도 점수에 영향이 없다.

여기서는 sim 마다:
   gold 변이 집합  vs  실제 성공 변이 집합
   -> MISSING(안 함) · WRONGARG(했는데 인자 다름) · EXTRA(gold 에 없는 변이) 로 가른다
`EXTRA` 가 중요하다 — 050 은 `approve_credit_limit_increase` 를 **두 번** 불러 DB 가 어긋났다.

변이 판정은 `a2/env_surface.json` 의 `mutates` 플래그(축자·환경 선언)로만 한다([[59]] 패턴매칭 아님).
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F
import x396_saying_vs_doing as C

ENVERR = ("Error:", "NOT_VERIFIED", "not been given", "Unknown", "Invalid", "cannot be",
          "[READ-FIRST]", "blocked by a policy gate")
WRAP = ("unlock_discoverable_agent_tool", "call_discoverable_agent_tool",
        "give_discoverable_user_tool", "call_discoverable_user_tool")


def mutating_set():
    p = os.path.join(HERE, "a2", "env_surface.json")
    d = json.load(io.open(p, encoding="utf-8"))
    t = d["banking_knowledge"]["tools"]
    return {k for k, v in t.items() if v.get("mutates")}


def flat(a):
    a = F.norm_args(a)
    if isinstance(a, dict) and isinstance(a.get("arguments"), dict):
        a = a["arguments"]
    if isinstance(a, dict) and isinstance(a.get("arguments"), str):
        try:
            a = json.loads(a["arguments"])
        except Exception:
            pass
    return a if isinstance(a, dict) else {}


def key(nm, a):
    return nm + "|" + json.dumps({k: str(v) for k, v in sorted(a.items())}, ensure_ascii=False)


def main():
    MUT = mutating_set()
    want = [x for x in sys.argv[1:] if x.startswith("task_")]
    print("=" * 112)
    print("x416 · DB 채점 단위 = 변이 집합 대조 (변이 도구 %d종·env_surface.mutates)" % len(MUT))
    print("=" * 112)
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            t = F.task_id(sim)
            if want and t not in want:
                continue
            ri = sim.get("reward_info") or {}
            basis = ",".join(ri.get("reward_basis") or [])
            # gold 변이
            gold = []
            for g in C.gold_rows(sim):
                nm = g["name"]
                if nm in MUT or nm in WRAP:
                    a = flat(g["args"])
                    if nm in MUT and a:
                        gold.append((nm, a, g["match"]))
            # 실제 성공 변이
            R = {m["id"]: " ".join(str(m.get("content") or "").split())
                 for m in (sim.get("messages") or []) if m.get("role") == "tool" and m.get("id")}
            act = []
            for m, tc in F.calls(sim):
                a = F.argsof(tc)
                nm = str(F.inner_name(a) or F.nameof(tc))
                if nm not in MUT:
                    continue
                if str(F.nameof(tc)) == "unlock_discoverable_agent_tool":
                    continue          # unlock 은 DB 를 안 바꾼다 (2026-08-19 계기 수정)
                fa = flat(a)
                if not fa:
                    continue
                body = R.get(tc.get("id"), "")
                if body and any(p in body for p in ENVERR):
                    continue
                act.append((nm, fa))
            gk = {key(n, a) for n, a, _ in gold}
            ak = [key(n, a) for n, a in act]
            print("\n### %s t%s reward=%s basis=%s" % (t, sim.get("trial"), ri.get("reward"), basis))
            miss = [(n, a) for n, a, mt in gold if key(n, a) not in ak]
            wrong, extra = [], []
            gnames = {n for n, _, _ in gold}
            for n, a in act:
                if key(n, a) in gk:
                    continue
                (wrong if n in gnames else extra).append((n, a))
            for lab, arr in (("MISSING(안 함)", miss), ("WRONGARG(인자 다름)", wrong),
                             ("EXTRA(gold 밖 변이)", extra)):
                if arr:
                    print("   %s %d" % (lab, len(arr)))
                    for n, a in arr[:6]:
                        print("      %-42s %s" % (n[:42], json.dumps(a, ensure_ascii=False)[:120]))
            if not (miss or wrong or extra):
                print("   변이 집합 일치")
    return 0


sys.exit(main())
