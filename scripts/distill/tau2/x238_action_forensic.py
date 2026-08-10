# -*- coding: utf-8 -*-
r"""x238 — **액션 에이전트는 진짜로 이득인가** (두 유료 런 전수 포렌식 · 새 유료 0 · 엔진 0).

## 왜 (handoff 2026-08-10 §8-1 · [[08]] · [[57]])

`bank_asubON_20260810` 이 **9/12**, 기준선 `bank_isoOFF_20260810` 이 **7/12** 다. 그러나 이것은
**런-간 비교**이고 12 vs 12 는 사전 등록 P4 기준에서 0.58↔0.75 도 가르지 못한다(C403). 오늘 하루
같은 모양의 주장이 **세 번 뒤집혔다**. ⇒ 집계에서 결론으로 직행하지 않고([[08]]) **궤적 전수**로
*무엇이 달라졌는지*를 먼저 본다. 이득/손실 판정은 이 인쇄물 **뒤에만** 쓴다.

## 사전 등록 (보기 전에 적는다)

  P0 팔 오염     `[T2_ACTION_SUB]` 마크가 ON 로그에만 있는가 (OFF=0 이어야 한다)
  P1 성적·종료   태스크별 pass · 종료사유 · 인프라 실패 0 인가
  P2 어디서 갈림 gold 액션별 `action_match` — **어느 칸**이 ON/OFF 에서 다른가
  P3 소유권      `requestor=user` 인 도구를 **에이전트가 직접** 부른 수(월권) ↔ 손님이 실행한 수
  P4 발화 계기   hands_over · self_claim · external  (문자열 대리지표 = **[M]**)
  P5 Δspurious   A2 가 선언한 액션 도구 중 **gold 이름 밖** 호출 수 · 게이트 거부 수
  P6 마크 상존   T2_KIND · T2_REDERIVE · T2_D1C · T2_OBJ_AXIS 가 양 팔에서 그대로 도는가

**읽는 법**: P2 가 같은 칸을 가리키고 P4 가 함께 움직여야 액션 에이전트의 이득이다. P4 만
움직이고 P2 가 안 움직이면 **발화만 예뻐진 것**이고, P5 가 늘면 그 이득은 **부작용과 상쇄**된다
(§1.3 — 부작용 없는 레버는 없다).

⚠gold 를 읽지만 이것은 **계측**이다 — 레버가 아니다([[23]] 는 A2/A3 에 관한 규율).

실행(리모트): python x238_action_forensic.py [ON태그] [OFF태그]
"""
import collections
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = "/home/woori/scratch/tau2-bench/data/simulations/%s/results.json"
LOG = "/home/woori/scratch/logs/%s.log"
DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge"

SELF = re.compile(r"\bI(?:'ve| have| will| am|'m)?\s+(?:already\s+)?(?:go(?:ne|ing) ahead and\s+)?"
                  r"(?:submit|submitted|submitting|process|processed|filed)\b", re.I)
EXTERNAL = re.compile(r"\b(website|web site|portal|mobile app|online banking|branch|"
                      r"sign in to your account|log in to your account)\b", re.I)
MARKS = ("T2_ACTION_SUB", "T2_KIND", "T2_REDERIVE", "T2_D1C", "T2_OBJ_AXIS", "T2_DECISION_ISOLATE")


def load(tag):
    p = SIM % tag
    d = json.load(open(p, encoding="utf-8"))
    return d.get("simulations") or []


def task_json(tid):
    p = os.path.join(DOM, "tasks", "%s.json" % tid)
    o = json.load(open(p, encoding="utf-8"))
    return o[0] if isinstance(o, list) and o else o


def gold_actions(tid):
    """권위본(`tasks/task_*.json`)에서 gold 액션 — 이름과 **소유자**([[54]]·101/102 포렌식 §0b)."""
    acts = ((task_json(tid).get("evaluation_criteria") or {}).get("actions") or [])
    names = set(a.get("name") for a in acts if isinstance(a, dict) and a.get("name"))
    user_owned = set(a.get("name") for a in acts
                     if isinstance(a, dict) and a.get("requestor") == "user" and a.get("name"))
    return names, user_owned


def a2_action_tools():
    p = os.path.join(HERE, "a2", "banking_knowledge.gate.json")
    g = json.load(open(p, encoding="utf-8"))
    out = set(g.get("action_tools") or [])
    out |= set(((g.get("eplan") or {}).get("write_tools")) or [])
    return out


def calls(sim):
    """(role, tool_name) 목록 — 누가 불렀는지가 소유권 판정의 전부다."""
    out = []
    for m in sim.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            f = tc.get("function") or {}
            out.append((m.get("role"), f.get("name") or tc.get("name")))
    return out


def denials(sim):
    """게이트 거부 = 도구 결과가 우리 문구로 시작하는 것."""
    n = 0
    for m in sim.get("messages") or []:
        if m.get("role") == "tool" and str(m.get("content") or "").lstrip().startswith("Error: ["):
            n += 1
    return n


def replies(sim):
    return [str(m.get("content") or "") for m in (sim.get("messages") or [])
            if m.get("role") == "assistant" and m.get("content")]


def marks(tag):
    p = LOG % tag
    c = collections.Counter()
    if not os.path.exists(p):
        return c
    for line in open(p, encoding="utf-8", errors="replace"):
        for k in MARKS:
            if "[%s]" % k in line:
                c[k] += 1
    return c


def per_run(tag):
    sims = load(tag)
    rows = []
    for s in sims:
        tid = s.get("task_id")
        gnames, uowned = gold_actions(tid)
        cs = calls(s)
        agent_calls = [n for r, n in cs if r == "assistant"]
        user_calls = [n for r, n in cs if r == "user"]
        acts = a2_action_tools()
        rep = replies(s)
        tail = " ".join(rep[-3:])
        checks = [(c.get("action") or {}).get("action_id", "?")
                  for c in ((s.get("reward_info") or {}).get("action_checks") or [])
                  if not c.get("action_match")]
        rows.append({
            "task": tid, "trial": s.get("trial"),
            "reward": (s.get("reward_info") or {}).get("reward"),
            "term": s.get("termination_reason"),
            "miss": checks,
            "usurped": sum(1 for n in agent_calls if n in uowned),      # P3 월권
            "user_ran": sum(1 for n in user_calls if n in uowned),
            "spurious": sum(1 for n in agent_calls if n in acts and n not in gnames),  # P5
            "denials": denials(s),
            "hands_over": sum(1 for n in uowned if n and n in tail),    # P4 (대리)
            "self_claim": 1 if SELF.search(tail) else 0,
            "external": 1 if EXTERNAL.search(tail) else 0,
            "turns": len(s.get("messages") or []),
        })
    return rows


def summarize(name, rows, mk):
    n = len(rows)
    ok = sum(1 for r in rows if r["reward"] == 1)
    print("\n%s  —  %d/%d" % (name, ok, n))
    print("  마크: " + " · ".join("%s=%d" % (k, mk.get(k, 0)) for k in MARKS))
    print("  종료사유: " + " · ".join("%s=%d" % (k, v) for k, v in
                                   collections.Counter(r["term"] for r in rows).most_common()))
    agg = collections.Counter()
    for r in rows:
        for k in ("usurped", "user_ran", "spurious", "denials", "hands_over",
                  "self_claim", "external"):
            agg[k] += r[k]
    print("  P3 월권=%d · 손님실행=%d | P5 Δspurious=%d · 게이트거부=%d | P4 hands_over=%d ·"
          " self_claim=%d · external=%d"
          % (agg["usurped"], agg["user_ran"], agg["spurious"], agg["denials"],
             agg["hands_over"], agg["self_claim"], agg["external"]))
    print("  %-9s %-6s %-5s %-14s %s" % ("task", "trial", "rew", "종료", "못 채운 gold 액션"))
    for r in sorted(rows, key=lambda x: (x["task"], str(x["trial"]))):
        print("  %-9s %-6s %-5s %-14s %s" % (r["task"], r["trial"], r["reward"], r["term"],
                                             ",".join(r["miss"]) or "-"))
    return agg


def main():
    on = sys.argv[1] if len(sys.argv) > 1 else "bank_asubON_20260810"
    off = sys.argv[2] if len(sys.argv) > 2 else "bank_isoOFF_20260810"
    rows_on, rows_off = per_run(on), per_run(off)
    mk_on, mk_off = marks(on), marks(off)

    print("=" * 96)
    print("P0 팔 오염 — [T2_ACTION_SUB]  ON=%d · OFF=%d   %s"
          % (mk_on.get("T2_ACTION_SUB", 0), mk_off.get("T2_ACTION_SUB", 0),
             "OK" if mk_on.get("T2_ACTION_SUB", 0) > 0 and mk_off.get("T2_ACTION_SUB", 0) == 0
             else "★오염 — 이 비교는 무효다"))
    a_on = summarize("ON  " + on, rows_on, mk_on)
    a_off = summarize("OFF " + off, rows_off, mk_off)

    print("\n" + "=" * 96)
    print("P1/P2 태스크별 (ON ↔ OFF)")
    tasks = sorted(set(r["task"] for r in rows_on + rows_off))
    for t in tasks:
        f = lambda rs: (sum(1 for r in rs if r["task"] == t and r["reward"] == 1),
                        sum(1 for r in rs if r["task"] == t))
        a, b = f(rows_on), f(rows_off)
        miss_on = collections.Counter(m for r in rows_on if r["task"] == t for m in r["miss"])
        miss_off = collections.Counter(m for r in rows_off if r["task"] == t for m in r["miss"])
        print("  %-9s ON %d/%d ↔ OFF %d/%d   못 채운 칸 ON=%s OFF=%s"
              % (t, a[0], a[1], b[0], b[1],
                 dict(miss_on) or "-", dict(miss_off) or "-"))
    print("\nP3~P5 합계 (ON ↔ OFF)")
    for k, lab in (("usurped", "월권(손님 도구를 에이전트가)"), ("user_ran", "손님이 실행"),
                   ("spurious", "Δspurious(gold 밖 액션)"), ("denials", "게이트 거부"),
                   ("hands_over", "hands_over[M]"), ("self_claim", "self_claim[M]"),
                   ("external", "external[M]")):
        print("  %-28s %3d ↔ %3d" % (lab, a_on[k], a_off[k]))
    print("\n※ 판정 규칙(사전 등록): P2 가 움직이지 않으면 이득이라 쓰지 않는다."
          "\n  P4 만 움직이면 '발화만 달라졌다'로 적고, P5 가 늘면 상쇄로 함께 적는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
