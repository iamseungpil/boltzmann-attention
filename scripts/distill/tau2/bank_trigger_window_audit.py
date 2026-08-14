# -*- coding: utf-8 -*-
r"""① 트리거 시점 관측 — write-착수 레버가 **옳은 자리에서 불리는가** (사용자 지시 2026-08-14).

현 트리거는 *"회피가 확정된 턴"*(도구 0 또는 transfer 만)이다. 그런데 옳은 시점은
**근거가 도착했고 아직 그 행동을 안 한 자리**다. 둘이 안 겹치면 레버는 정작 필요할 때
침묵하고(놓침), 근거 전에는 헛돈다(공전). 이 스크립트는 궤적에서 세 조건을 재구성해
그 겹침을 센다 — 엔진 변경 0·유료 0([[08]] 관측 우선).

턴 분류(assistant 턴마다):
  WINDOW   근거 있음 ∧ 그 행동 미실행   ← **옳은 시점**
  ⤷ FIRE   그중 회피 턴                 = 현 트리거가 잡는 자리
  ⤷ MISS   그중 비-회피 턴(다른 도구 호출 중) = **놓치는 자리**
  IDLE     근거 없음 ∧ 회피 턴          = 공전(무해하나 서브 호출 낭비)

근거 = 직전 손님 발화 이후의 **성공한 도구 결과**(t2_subcall.recent_tool_text 와 같은 규칙).
행동 = 그 런의 gold 채점표가 write 로 표시한 도구(벤치 메타데이터·분석 전용).

사용: py -3 bank_trigger_window_audit.py <tag> [<tag>...]
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

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
SIMS = "/home/woori/scratch/tau2-bench/data/simulations"
TRANSFER = ("transfer_to_human_agents", "request_human_agent_transfer")


def load(tag):
    p = os.path.join(SIMS, tag, "results.json")
    if os.path.exists(p):
        return json.load(io.open(p, encoding="utf-8"))
    with gzip.open(os.path.join(BASE, tag + "_results.json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)


def write_tools(d):
    """gold 채점표가 write 로 표시한 도구 이름(래퍼는 대상 도구까지)."""
    out = set()
    for s in d.get("simulations", []):
        for ck in ((s.get("reward_info") or {}).get("action_checks") or []):
            if ck.get("tool_type") != "write":
                continue
            a = ck.get("action") or {}
            ar = a.get("arguments") or {}
            out.add(str(a.get("name")))
            for k in ("agent_tool_name", "user_tool_name", "tool_name"):
                if ar.get(k):
                    out.add(str(ar[k]))
    return {n for n in out if n and n != "None"}


def call_names(m):
    out = []
    for tc in (m.get("tool_calls") or []):
        f = tc.get("function") or {}
        nm = f.get("name") or tc.get("name")
        ar = tc.get("arguments", f.get("arguments"))
        if isinstance(ar, str):
            try:
                ar = json.loads(ar)
            except Exception:
                ar = {}
        inner = (ar or {}).get("agent_tool_name") or (ar or {}).get("user_tool_name")
        out.append((str(nm or ""), str(inner or ""), tc.get("id")))
    return out


def run(tags):
    tot = collections.Counter()
    for tag in tags:
        d = load(tag)
        wt = write_tools(d)
        print("#" * 78)
        print("# %s · write 도구 %d종" % (tag, len(wt)))
        for s in d.get("simulations", []):
            msgs = s.get("messages") or []
            res = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
            basis, done = False, False
            c = collections.Counter()
            first_window = None
            for i, m in enumerate(msgs):
                role = m.get("role")
                if role == "user":
                    basis = False                      # 근거는 이번 손님 발화 이후분만
                    continue
                if role == "tool":
                    ok = not m.get("error") and not str(m.get("content") or "").lstrip(
                        ).startswith("Error")
                    if ok and str(m.get("content") or "").strip():
                        basis = True
                    continue
                if role != "assistant":
                    continue
                names = call_names(m)
                # 회피 = 도구 0 또는 transfer 만
                called = {n for n, _, _ in names} | {inn for _, inn, _ in names if inn}
                deflect = (not names) or bool(called) and called <= set(TRANSFER)
                if basis and not done:
                    c["FIRE" if deflect else "MISS"] += 1
                    if first_window is None:
                        first_window = (i, "FIRE" if deflect else "MISS")
                elif deflect and not basis:
                    c["IDLE"] += 1
                # 이 턴에서 write 가 성공했나
                for nm, inn, cid in names:
                    if (nm in wt or (inn and inn in wt)):
                        r = res.get(cid) or {}
                        if not r.get("error") and not str(r.get("content") or "").lstrip(
                                ).startswith("Error"):
                            done = True
            print("  %-10s trial=%s  FIRE %-3d MISS %-3d IDLE %-3d · 첫 window=%s" % (
                s.get("task_id"), s.get("trial"), c["FIRE"], c["MISS"], c["IDLE"],
                first_window))
            for k in ("FIRE", "MISS", "IDLE"):
                tot[k] += c[k]
    print("-" * 78)
    w = tot["FIRE"] + tot["MISS"]
    print("합계: WINDOW %d (FIRE %d = %.0f%% · **MISS %d = %.0f%%**) · IDLE(공전) %d" % (
        w, tot["FIRE"], 100.0 * tot["FIRE"] / w if w else 0,
        tot["MISS"], 100.0 * tot["MISS"] / w if w else 0, tot["IDLE"]))
    print("※ MISS 가 크면 트리거를 '회피'가 아니라 **근거 증가 ∧ 행동 미실행**으로 옮겨야 한다(②).")


if __name__ == "__main__":
    run(sys.argv[1:] or ["bank_t7287_a_20260814i", "bank_t7287_b_20260814i"])
