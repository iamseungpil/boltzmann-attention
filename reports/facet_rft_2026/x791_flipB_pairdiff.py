#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x791 - flip pass/fail 짝의 **첫 갈림 턴** 탐지 (2026-09-05).

사용:  python x791_flipB_pairdiff.py <pairs.txt> [ctx]
pairs.txt 한 줄 = "task_id passTag passSim failTag failSim"

⛔ 판정하지 않는다. 찍기만 한다.
  - 두 sim 의 messages 를 (role, content, tool_calls[name+args]) 로 정규화해 index 0 부터 비교
  - 첫 불일치 index 와 그 전후를 축자로 찍는다
  - assistant 턴의 prompt_tokens 수열을 나란히 찍는다 (system prompt 차이 탐지용)
"""
import io, json, sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
_cache = {}


def load(tag):
    if tag not in _cache:
        _cache[tag] = json.load(open("%s/%s/results.json" % (SIMROOT, tag)))
    return _cache[tag]


def sig(m):
    role = m.get("role")
    c = m.get("content")
    c = "" if c is None else c
    tcs = []
    for tc in (m.get("tool_calls") or []):
        try:
            a = json.dumps(tc.get("arguments"), sort_keys=True, ensure_ascii=False)
        except Exception:
            a = str(tc.get("arguments"))
        tcs.append((tc.get("name"), a, tc.get("requestor")))
    return (role, c, tuple(tcs))


def show(m, maxc):
    role = m.get("role")
    c = m.get("content") or ""
    if isinstance(c, str):
        c = c.replace("\n", " \n ")
    tcs = []
    for tc in (m.get("tool_calls") or []):
        tcs.append("%s(%s)" % (tc.get("name"), json.dumps(tc.get("arguments"), ensure_ascii=False)))
    pt = ((m.get("usage") or {}) or {}).get("prompt_tokens")
    ct = ((m.get("usage") or {}) or {}).get("completion_tokens")
    return "%-9s pt=%s ct=%s | %s%s" % (role, pt, ct, c[:maxc], (" ;; " + " ;; ".join(tcs)) if tcs else "")


def main():
    pairs = [l.split() for l in open(sys.argv[1]).read().splitlines() if l.strip() and not l.startswith("#")]
    MAXC = int(sys.argv[2]) if len(sys.argv) > 2 else 420
    for task, ptag, psim, ftag, fsim in pairs:
        print("\n" + "=" * 100)
        print("### %s   PASS=%s/%s   FAIL=%s/%s" % (task, ptag, psim[:8], ftag, fsim[:8]))
        out = {}
        for lab, tag, sid in (("PASS", ptag, psim), ("FAIL", ftag, fsim)):
            d = load(tag)
            s = next(x for x in d["simulations"] if x["id"] == sid)
            info = d.get("info", {})
            ai = info.get("agent_info", {})
            print("  %s tag=%s commit=%s seed=%s trial=%s term=%s reward=%s nmsg=%d" % (
                lab, tag, str(info.get("git_commit"))[:8], s.get("seed"), s.get("trial"),
                s.get("termination_reason"), (s.get("reward_info") or {}).get("reward"), len(s["messages"])))
            print("       agent_llm=%s args=%s" % (ai.get("llm"), json.dumps(ai.get("llm_args"), ensure_ascii=False)[:300]))
            out[lab] = s
        A, B = out["PASS"]["messages"], out["FAIL"]["messages"]
        # prompt_tokens 수열 (assistant 만)
        for lab, M in (("PASS", A), ("FAIL", B)):
            pts = [(i, (m.get("usage") or {}).get("prompt_tokens")) for i, m in enumerate(M)
                   if m.get("role") == "assistant" and (m.get("usage") or {}).get("prompt_tokens")]
            print("  %s pt[:12]=%s" % (lab, pts[:12]))
        n = min(len(A), len(B))
        first = None
        for i in range(n):
            if sig(A[i]) != sig(B[i]):
                first = i
                break
        if first is None:
            first = n if len(A) != len(B) else None
        if first is None:
            print("  >> 전 메시지 동일 (길이도 같음)")
            continue
        print("  >> FIRST_DIVERGENCE index=%d  role_pass=%s role_fail=%s" % (
            first, A[first].get("role") if first < len(A) else "-",
            B[first].get("role") if first < len(B) else "-"))
        lo = max(0, first - 3)
        for i in range(lo, min(first + 3, max(len(A), len(B)))):
            print("  --- i=%d" % i)
            print("    P| " + (show(A[i], MAXC) if i < len(A) else "(없음)"))
            print("    F| " + (show(B[i], MAXC) if i < len(B) else "(없음)"))


if __name__ == "__main__":
    main()
