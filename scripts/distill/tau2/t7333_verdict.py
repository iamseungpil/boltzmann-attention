# -*- coding: utf-8 -*-
r"""t7333 판정 — **⒡ 합성**(선언 문서 배달 + 후보별 값)의 라이브 A/B (2026-08-21·오프라인·LLM 0)

사전등록(`run_t7333_composed_ab_20260821.sh`)이 못 박은 것만 낸다:
    종점 = **reward 뿐**([[69]]) · 판정선 = C483 잡음 바닥 **±4/40** ⇒ 차 **2 미만이면 null**
    의무 3종([[70]]) = ⒜ 전체 reward 짝 ⒝ **태스크별 부호표** ⒞ 무엇을 팔았나
    동급 계측 = 조회 수 · 날조 · over-action(쓰기) · **배달 발화율**
⛔024 단독은 6 sim 이라 **부호만** 읽고 크기는 읽지 않는다(사전등록 축자).
"""
import collections
import glob
import gzip
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

BASE = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
TAG = "20260821c"
WRITE_HINTS = ("apply_for_credit_card", "open_bank_account", "submit_referral", "transfer",
               "log_verification", "unlock_discoverable_agent_tool", "call_discoverable_agent_tool")


def nameof(tc):
    return tc.get("name") or ((tc.get("function") or {}).get("name"))


def load(arm):
    sims = []
    for part in ("hot", "rest"):
        p = os.path.abspath(os.path.join(BASE, "bank_t7333_%s_%s_%s.results.json.gz" % (arm, part, TAG)))
        if not os.path.exists(p):
            print("⚠없음: %s" % p)
            continue
        d = json.load(gzip.open(p, "rt", encoding="utf-8"))
        sims.extend(d.get("simulations") or [])
    return sims


def stat(sims):
    s = {"n": len(sims), "reward": 0.0, "pass": 0, "reads": 0, "writes": 0, "fit": 0}
    per = collections.Counter()
    npass = collections.Counter()
    for x in sims:
        tid = str(x.get("task_id") or "")
        r = (x.get("reward_info") or {}).get("reward") or 0.0
        s["reward"] += r
        per[tid] += 1
        if r >= 1.0:
            s["pass"] += 1
            npass[tid] += 1
        for m in (x.get("messages") or []):
            for tc in (m.get("tool_calls") or []):
                nm = nameof(tc) or ""
                if nm.startswith("KB_search") or nm == "shell":
                    s["reads"] += 1
                if nm in WRITE_HINTS:
                    s["writes"] += 1
                if nm == "check_card_application_fit":
                    s["fit"] += 1
    s["per"], s["npass"] = per, npass
    return s


def log_count(arm, needle):
    n = 0
    for part in ("hot", "rest"):
        p = os.path.abspath(os.path.join(BASE, "bank_t7333_%s_%s_%s.log.gz" % (arm, part, TAG)))
        if not os.path.exists(p):
            continue
        with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if needle in line:
                    n += 1
    return n


def main():
    A, B = load("ctl"), load("treat")
    sa, sb = stat(A), stat(B)
    print("=" * 92)
    print("t7333 · ⒡ 합성 라이브 A/B · 종점 = reward · 판정선 |d| < 2 = null")
    print("=" * 92)
    print("%-10s %-6s %-10s %-8s %-8s %-8s %-8s" % ("팔", "n", "reward합", "pass", "조회", "쓰기", "fit"))
    for nm, s in (("ctl", sa), ("treat", sb)):
        print("%-10s %-6d %-10.2f %-8d %-8d %-8d %-8d"
              % (nm, s["n"], s["reward"], s["pass"], s["reads"], s["writes"], s["fit"]))
    d = sb["reward"] - sa["reward"]
    print("\n★reward 차(treat − ctl) = %+.2f · pass 차 = %+d ⇒ **%s**"
          % (d, sb["pass"] - sa["pass"], "null (판정선 미달)" if abs(d) < 2 else "판정선 통과"))

    print("\n[태스크별 부호표] ([[70]] 의무 — 합이 null 이어도 부호는 갈린다)")
    tasks = sorted(set(sa["per"]) | set(sb["per"]))
    print("%-12s %-10s %-10s %s" % ("태스크", "ctl", "treat", "부호"))
    for t in tasks:
        ca, cb = sa["npass"][t], sb["npass"][t]
        na, nb = sa["per"][t], sb["per"][t]
        sign = "=" if cb == ca else ("+" if cb > ca else "−")
        print("%-12s %-10s %-10s %s" % (t, "%d/%d" % (ca, na), "%d/%d" % (cb, nb), sign))

    print("\n[배달 발화 · 우리 층 계측]")
    for nm in ("ctl", "treat"):
        fired = log_count(nm, "T2_ARG_DOC_SUB")
        kept = log_count(nm, "격리 서브·선언")
        drop = log_count(nm, "넘긴 문서")
        memo = log_count(nm, "메모 재사용")
        val = log_count(nm, "documented_return_for_stated_spend")
        print("  %-6s 배달 로그 %3d (값 채택 %d · 철회 %d · 메모 %d) · 값주석 %d"
              % (nm, fired, kept, drop, memo, val))

    print("\n[무엇을 팔았나] Δ조회 %+d · Δ쓰기 %+d · Δfit %+d"
          % (sb["reads"] - sa["reads"], sb["writes"] - sa["writes"], sb["fit"] - sa["fit"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
