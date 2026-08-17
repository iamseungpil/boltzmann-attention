# -*- coding: utf-8 -*-
r"""x368 — **일반 프로토콜 술어의 라이브 위반 계수**(gold 무참조·LLM 0·비용 0).

## 사용자 지적 (2026-08-18 축자)

*"gold 보고 절차를 만들지 않아도, 정책 문서 모두 뒤지면 절차 일반화가 가능할 것 같다. …
  일반화 절차를 정책에서 찾거나 우리의 일반화 절차로 규정할 수 있으면, gold 에 각 태스크별로
  귀속된다고 볼 수 없다. 일반화 할 수 있는지에 집중하라"*

맞다. 그리고 출처는 정책 산문보다 **한 층 위**에 있다 — **환경의 도구 계약 축자**:

    call_discoverable_agent_tool : "Call an agent discoverable tool that **you have previously
                                    unlocked**. Use this **after unlocking a tool with
                                    unlock_discoverable_agent_tool**."
    call_discoverable_user_tool  : "Call a tool that **was given to you by the agent**."

⇒ 두 일반 규칙이 나온다(태스크 무관·도메인 무관·닫힌 술어·[[22]]):

    **G1** T 가 discoverable 이면 `call` 앞에 `unlock` 이 **선행**한다.
    **G2** T 가 **손님 도구**면(agent 도구 목록에 없고 `user_tools` 에 있다) 에이전트는 T 를 못 부른다
           — `give_discoverable_user_tool(T)` 로 건네야 **손님이** 부른다.

G2 가 바로 *"분쟁을 제기하시면 됩니다" 라고 **말만** 하고 안 건네는* 실패를 잡는다:
**어시스턴트 발화에 손님-도구 이름이 있는데 `give` 호출이 없다** — 레지스트리와 호출 로그만으로
판정된다(도메인 어휘 0 · gold 0 · 정규식 0).

## 이 프로브가 세는 것 (최근 라이브 런 전수)

    N_named   어시스턴트가 **손님 도구 이름을 발화**한 sim 수
    N_gave    그 sim 중 `give_discoverable_user_tool` 을 실제로 부른 수
    **G2_viol = N_named − N_gave**   ← 일반 규칙 위반(=레버가 살 자리)
    G1_viol   `call_discoverable_agent_tool(T)` 인데 앞선 `unlock(T)` 이 없는 호출 수
    또한 **레버 발화와의 교차표** — 위반 sim 에서 우리 레버가 발화했는가([[55]] 배관 먼저)

## 판정 (사전 고정)

    G2_viol ≥ 8 sim              → 일반 규칙만으로 살 자리가 크다 ⇒ 정책 DAG 없이도 축이 선다
    G2_viol 이 큰데 레버 발화 0   → **배선**(늦게 발화하거나 조건이 안 맞는다)
    G2_viol 이 큰데 레버 발화 >0  → **문구·형태**(발화는 하는데 안 듣는다) ⇒ 치환형으로
    G2_viol ≈ 0                  → 이 술어로는 살 것이 없다 ⇒ 축 폐기

실행: /home/woori/venvs/seka_env/bin/python x368_general_protocol_census.py
"""
import collections
import gzip
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x364_eligibility_axis_iso as E                             # noqa: E402

SIMDIR = os.path.join(E.REPORTS, "sim_results")
GIVE = "give_discoverable_user_tool"
UNLOCK = "unlock_discoverable_agent_tool"
CALL_A = "call_discoverable_agent_tool"
CALL_U = "call_discoverable_user_tool"
LEVERS = ("T2_USER_TOOL_NOTE", "T2_GIVE_EXEC_NUDGE", "T2_GIVE_RELEVANCE_NUDGE",
          "T2_UNINSTRUCTABLE", "T2_DUP_REPRESENT", "T2_UNCALLED_UNLOCK", "T2_PROCEDURE")


def registries():
    """**환경의 런타임 레지스트리**에서 두 집합을 읽는다 — 이것이 일반화의 출처다.

    ★2026-08-18 수리(1차 계수 무효): 첫 판은 문서에서 `*_1234` 꼴 토큰을 긁어 46종을 만들고
      **손님-측/에이전트-측을 안 갈랐다** — 에이전트 도구 이름을 발화하고 `give` 를 안 한 것까지
      위반으로 세어 168 sim 이 나왔다(과대). 환경이 두 집합을 **직접 선언**한다:

          KnowledgeUserTools  의 `__discoverable__` 메서드 = **손님이 부른다** → `give` 선행 필수
          KnowledgeTools      의 `__discoverable__` 메서드 = 에이전트가 부른다 → `unlock` 선행 필수

      데코레이터 속성 이름조차 `tools.DISCOVERABLE_ATTR` 에서 읽는다 ⇒ 우리가 쓴 도메인 어휘 0 ·
      gold 0 · 태스크별 분기 0([[05]]·[[22]] 닫힌 술어). 다른 도메인은 같은 규약이면 자동, 아니면
      **빈 집합 = 침묵**([[25]] 모르면 안 뺀다).
    """
    from tau2.domains.banking_knowledge import tools as T          # noqa: E402
    attr = getattr(T, "DISCOVERABLE_ATTR", "__discoverable__")

    def marked(cls):
        out = set()
        for n in dir(cls or ()):
            if n.startswith("_"):
                continue
            m = getattr(cls, n, None)
            if callable(m) and getattr(m, attr, False):
                out.add(n)
        return out

    return (marked(getattr(T, "KnowledgeUserTools", None)),
            marked(getattr(T, "KnowledgeTools", None)))


def main():
    if not os.path.isdir(SIMDIR):
        print("sim_results 없음 — 중단")
        return 1
    disc, agent_disc = registries()
    if not disc:
        print("환경이 손님-측 discoverable 레지스트리를 안 준다 — 침묵(이 술어는 이 도메인에 없다)")
        return 1
    print("x368 · **손님-측 discoverable %d종**(give 선행 필수) · 에이전트-측 %d종(unlock 선행 필수)"
          " — 전부 env 런타임 레지스트리 유래 · gold 0" % (len(disc), len(agent_disc)))
    print("   손님-측: %s" % ", ".join(sorted(disc)))
    print("판정(사전 고정): G2_viol ≥8 sim → 일반 규칙만으로 축이 선다 · 위반 큰데 레버 발화 0 → "
          "배선 · 발화>0 → 문구·형태(치환형으로) · G2_viol≈0 → 축 폐기\n")

    rows = []
    for fn in sorted(os.listdir(SIMDIR)):
        if not fn.endswith("_results.json.gz"):
            continue
        try:
            with gzip.open(os.path.join(SIMDIR, fn)) as f:
                data = json.load(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
        except Exception as e:
            print("  ⚠%s 열기 실패: %r" % (fn, e))
            continue
        sims = data.get("simulations") or data.get("results") or []
        for s in sims:
            msgs = s.get("messages") or []
            named, gave, calls_u, unlocked, bad_g1 = set(), 0, 0, set(), 0
            for m in msgs:
                role = m.get("role")
                txt = str(m.get("content") or "")
                for tc in (m.get("tool_calls") or ()):
                    nm = str((tc or {}).get("name") or "")
                    args = (tc or {}).get("arguments") or {}
                    tgt = str(args.get("discoverable_tool_name")
                              or args.get("agent_tool_name") or "")
                    if nm == GIVE:
                        gave += 1
                    elif nm == UNLOCK:
                        unlocked.add(tgt)
                    elif nm == CALL_A:
                        if tgt and tgt not in unlocked:
                            bad_g1 += 1
                    elif nm == CALL_U:
                        calls_u += 1
                if role == "assistant" and txt:
                    for d in disc:
                        if d in txt:
                            named.add(d)
            rows.append({"run": fn.replace("_results.json.gz", ""),
                         "task": str(s.get("task_id") or (s.get("task") or {}).get("id") or "?"),
                         "seed": str(s.get("seed")), "named": sorted(named), "gave": gave,
                         "call_u": calls_u, "g1_viol": bad_g1,
                         "reward": (s.get("reward_info") or {}).get("reward", s.get("reward"))})

    named_sims = [r for r in rows if r["named"]]
    g2 = [r for r in named_sims if r["gave"] == 0]
    print("sim 총 %d · 어시스턴트가 discoverable 이름을 발화한 sim **%d** · 그중 give 호출 0 = "
          "**G2 위반 %d sim**" % (len(rows), len(named_sims), len(g2)))
    print("G1 위반(unlock 없이 call) 총 %d회 · give>0 인 sim %d · 손님 호출>0 인 sim %d"
          % (sum(r["g1_viol"] for r in rows), sum(1 for r in rows if r["gave"]),
             sum(1 for r in rows if r["call_u"])))
    by = collections.Counter(r["task"] for r in g2)
    print("\nG2 위반 태스크 상위: %s"
          % ", ".join("%s×%d" % (t[5:], n) for t, n in by.most_common(15)))
    print("\nG2 위반 sim 의 pass: %d/%d"
          % (sum(1 for r in g2 if (r["reward"] or 0) >= 1), len(g2)))
    out = os.path.join(E.REPORTS, "x368_general_protocol.json")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
