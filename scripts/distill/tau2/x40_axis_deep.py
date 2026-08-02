#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x40: x38의 **6축 전부**를 폭주 축과 같은 깊이로 정밀 분석 (무료·로컬).

사용자 지시(2026-08-02): *"6축 분해 한 것 전부 폭주처럼 철저히 재분석하고 대책을 설계서에 반영하라."*

폭주 축(x39)에서 확정된 방법 = **지시 스택 전수 + 결정점 궤적 전문**. 그 방법을 나머지 축에 적용한다:
  · 지시 스택 = ⑴env 정책 ⑵검색된 env KB 문서(해당 절차문) ⑶우리 엔진/A2 문구
  · 결정점 = gold 액션이 나왔어야 할 자리 전후의 본문·호출·도구 반환
질문은 매번 같다: **에이전트가 그때 읽고 있던 텍스트가 무엇을 지시했는가.**

축(x38):
  1 이관        004·008·012·014·035
  2 카드 선택   003·007·023·024
  3 discoverable 015·017·019·028·029·032·033·040·041
  4 write→이관  010·016
  6 gold 파손   005
"""
import argparse
import gzip
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--simdir", default=os.path.join(HERE, "..", "..", "..",
                                                 "reports", "facet_rft_2026", "sim_results"))
ap.add_argument("--tags", default="bank_qp32p1_gpu0_20260802,bank_qp32p1_gpu1_20260802")
ap.add_argument("--axis", default="2")
ap.add_argument("--chars", type=int, default=520)
A = ap.parse_args()

AXES = {"1": ["task_004", "task_008", "task_012", "task_014", "task_035"],
        "2": ["task_003", "task_007", "task_023", "task_024"],
        "3": ["task_015", "task_017", "task_019", "task_028", "task_029",
              "task_032", "task_033", "task_040", "task_041"],
        "4": ["task_010", "task_016"],
        "6": ["task_005"]}
FOCUS = {"1": ["transfer_to_human_agents"],
         "2": ["apply_for_credit_card", "check_card_application_fit"],
         "3": ["unlock_discoverable_agent_tool", "call_discoverable_agent_tool",
               "call_discoverable_user_tool", "give_discoverable_user_tool"],
         "4": ["submit_referral", "submit_transaction", "transfer_to_human_agents"],
         "6": ["log_verification", "change_user_email"]}

SIMS = {}
for tag in A.tags.split(","):
    p = os.path.join(A.simdir, tag.strip() + ".results.json.gz")
    if os.path.exists(p):
        for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations", []):
            SIMS[str(s.get("task_id"))] = s


def steps(sim):
    byid = {m.get("id"): str(m.get("content") or "")
            for m in (sim.get("messages") or []) if (m.get("role") or "") == "tool"}
    out = []
    for m in (sim.get("messages") or []):
        r = m.get("role") or ""
        if r == "tool":
            continue
        t = str(m.get("content") or "")
        if t.strip():
            out.append({"k": "USER" if r == "user" else "SAY", "text": t})
        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            a = fn.get("arguments", tc.get("arguments"))
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {"__raw": a}
            out.append({"k": "CALL", "name": str(fn.get("name") or ""),
                        "args": a if isinstance(a, dict) else {},
                        "req": tc.get("requestor") or r,
                        "out": byid.get(tc.get("id"), "")})
    return out


def one(txt, n):
    return " ".join(str(txt or "").split())[:n]


focus = FOCUS.get(A.axis, [])
for task in AXES.get(A.axis, []):
    sim = SIMS.get(task)
    if sim is None:
        print("⚠ %s 없음" % task)
        continue
    st = steps(sim)
    ri = sim.get("reward_info") or {}
    print("\n" + "=" * 100)
    print("# %s · 기준 %s %s · 종료 %s · 호출 %d"
          % (task, ",".join(ri.get("reward_basis") or []) or "—",
             ri.get("reward_breakdown") or {}, sim.get("termination_reason"),
             sum(1 for e in st if e["k"] == "CALL")))
    for ac in (ri.get("action_checks") or []):
        a = ac.get("action") or {}
        if a.get("name") in focus or not focus:
            print("  gold: %s(%s) [%s] match=%s"
                  % (a.get("name"), one(json.dumps(a.get("arguments"), ensure_ascii=False), 180),
                     a.get("requestor"), ac.get("action_match")))
    # 초점 도구 호출 전부 + 반환
    print("  ── 초점 호출 ──")
    for e in st:
        if e["k"] == "CALL" and (e["name"] in focus):
            print("   ▶ %s [%s] %s" % (e["name"], e["req"],
                                       one(json.dumps(e["args"], ensure_ascii=False), 220)))
            print("     ← %s" % one(e["out"], A.chars))
    # 관련 KB 문서(초점 도구 이름이 등장하는 검색 결과)
    print("  ── 관련 KB 문서 절차문 ──")
    seen = set()
    for e in st:
        if e["k"] != "CALL" or "search" not in e["name"].lower():
            continue
        o = e["out"] or ""
        for m in re.finditer(r"(?:^|\s)(\d+)\. ([^|]{6,90}?) ID: (doc_[a-z0-9_()]+)", o):
            title, did = m.group(2).strip(), m.group(3)
            if did in seen:
                continue
            seg = o[m.start():m.start() + 900]
            if any(f in seg for f in focus) or "Procedure" in seg or "must" in seg:
                seen.add(did)
                print("   · [%s] %s" % (did, title))
                print("     %s" % one(seg[len(m.group(0)):], A.chars))
    # 마지막 4스텝
    print("  ── 종결 4스텝 ──")
    for e in st[-4:]:
        if e["k"] == "CALL":
            print("   ▶ %s %s" % (e["name"], one(json.dumps(e["args"], ensure_ascii=False), 120)))
            print("     ← %s" % one(e["out"], 200))
        else:
            print("   ◆ %s: %s" % (e["k"], one(e["text"], 300)))
