# -*- coding: utf-8 -*-
r"""MISS 턴 성격 감사 — 트리거 이설 시 **무엇을 막게 되는가** (설계 결정용·유료 0).

`bank_trigger_window_audit` 이 잰 MISS(근거 있음 ∧ 행동 미실행 ∧ **비-회피**)는 73% 다.
그 자리에 개입하려면 우리 발화면(`_ap_regen`)이 **재생성**을 하므로 그 턴의 도구 호출이 버려진다.
따라서 이설 전에 물어야 한다: MISS 턴은 무엇을 부르고 있었나?

분류(도구 이름 기준·구조 판정):
  DUP-READ    이미 같은 인자로 성공한 read 재호출     ← 막아도 손실 0(오히려 이득)
  NEW-READ    처음 보는 read                          ← 막으면 **정보 손실 위험**
  WRITE       write 계열(gold write 도구명)           ← 막으면 안 됨
  META        unlock/list/KB_search 등 발견 계열       ← 대체로 무해
사용: py -3 bank_miss_turn_audit.py <tag> [...]
"""
import collections
import gzip
import io
import json
import os
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import t2_forensic as F                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
SIMS = "/home/woori/scratch/tau2-bench/data/simulations"
TRANSFER = ("transfer_to_human_agents", "request_human_agent_transfer")
META_PREFIX = ("unlock_", "list_", "KB_search", "give_", "get_current_time", "shell")


def load(tag):
    """정본 = `t2_forensic.load`(리모트 라이브 결과 우선·gz 자동)."""
    return F.load(tag)


def write_tools(d):
    """정본 = `t2_forensic.write_tools`(사본 둘이 갈라져 있던 자리)."""
    return F.write_tools(d)


def calls(m):
    """(이름, 대상도구, 인자JSON, id) — 해제는 정본 `t2_forensic` 위임(사본 금지)."""
    out = []
    for tc in (m.get("tool_calls") or []):
        ar = F.argsof(tc)
        out.append((str(F.nameof(tc)), str(F.inner_name(ar)),
                    json.dumps(ar, ensure_ascii=False, sort_keys=True), tc.get("id")))
    return out


def run(tags):
    tot = collections.Counter()
    ex = collections.Counter()
    for tag in tags:
        d = load(tag)
        wt = write_tools(d)
        for s in d.get("simulations", []):
            msgs = s.get("messages") or []
            res = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
            basis, done = False, False
            seen_ok = set()                      # 성공한 (도구,인자) — 중복 read 판정용
            for m in msgs:
                role = m.get("role")
                if role == "user":
                    basis = False
                    continue
                if role == "tool":
                    if not m.get("error") and str(m.get("content") or "").strip() \
                            and not str(m.get("content")).lstrip().startswith("Error"):
                        basis = True
                    continue
                if role != "assistant":
                    continue
                cs = calls(m)
                names = {n for n, _, _, _ in cs} | {i for _, i, _, _ in cs if i}
                deflect = (not cs) or (names and names <= set(TRANSFER))
                if basis and not done and not deflect:
                    for nm, inner, sig, cid in cs:
                        eff = inner or nm
                        key = (eff, sig)
                        if eff in wt:
                            k = "WRITE"
                        elif key in seen_ok:
                            k = "DUP-READ"
                        elif any(eff.startswith(p) or nm.startswith(p) for p in META_PREFIX):
                            k = "META"
                        else:
                            k = "NEW-READ"
                        tot[k] += 1
                        ex[(k, eff)] += 1
                for nm, inner, sig, cid in cs:
                    r = res.get(cid) or {}
                    ok = not r.get("error") and not str(r.get("content") or "").lstrip(
                        ).startswith("Error")
                    eff = inner or nm
                    if ok:
                        seen_ok.add((eff, sig))
                        if eff in wt:
                            done = True
    n = sum(tot.values())
    print("MISS 턴이 부르던 호출 %d건" % n)
    for k in ("DUP-READ", "META", "NEW-READ", "WRITE"):
        print("  %-9s %3d (%.0f%%)" % (k, tot[k], 100.0 * tot[k] / n if n else 0))
    print("\n상위 도구:")
    for (k, nm), c in ex.most_common(10):
        print("  %-9s %-46s %d" % (k, nm[:46], c))
    print("\n※ DUP-READ+META 가 다수면 이설의 과차단 위험이 낮다. NEW-READ/WRITE 가 크면 "
          "재생성형 발화로는 이설하면 안 된다(비-재생성 채널이 선결).")


if __name__ == "__main__":
    run(sys.argv[1:] or ["bank_t7285_a_20260814g", "bank_t7285_b_20260814g",
                         "bank_t7287_a_20260814i", "bank_t7287_b_20260814i",
                         "bank_t7288_a_20260814j", "bank_t7288_b_20260814j"])
