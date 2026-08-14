# -*- coding: utf-8 -*-
"""조기 **인간-이관 이탈** 감사 — 에이전트가 일을 하는 대신 상담원에게 넘기고 끝내는가.

동기(2026-08-14 야간·G대표 12 전수 포렌식): `bank_fail_forensic_all` 로 t7286 을 읽으니
지배 형태는 C466 대로 MISS-NOTCALLED 였지만, 궤적 **끝**에 반복되는 형태가 하나 더 보였다 —
종료 직전 본문이 "A human agent is now on their way…" 류인 sim 이 눈으로 세도 여럿이다
(048·069·055·036). 그런데 **081 은 이관이 gold** 다. 즉 이관 자체는 결함이 아니고,
**gold 가 요구하지 않은 이관**만 결함이다. 집계로는 둘이 구분되지 않아 지금까지 안 보였다.

이 감사는 판정하지 않고 **네 수를 가른다**([[08]] — 집계에서 결론 직행 금지):
    gold-이관   gold action 에 이관이 있다      → 이관은 정답 행동
    이탈        gold 에 없는데 이관하고 끝냈다  → 후보 결함
    잔여        이탈 시점에 남아 있던 gold 액션 수(= 이관으로 포기한 일의 크기)
    도달        그 sim 이 실제로 맞힌 gold 수

⚠이것은 **관측**이다. 레버가 아니다 — 어떤 처방도 [[62]] 대로 격리 프로브로 결손을 먼저 재고
붙인다. 특히 "이관하지 마라"는 금지문은 [[63]]/[[42]] 상 역효과 전례가 있다(x301 B_WARN 0/8).

사용: py bank_bailout_audit.py <tag> [<tag>...]
"""
import collections
import sys

import t2_forensic as F

# tau2 하네스의 이관 프로토콜 이름(도메인 어휘 아님·gold action 이 쓰는 그 이름)
TRANSFER = ("transfer_to_human_agents", "request_human_agent_transfer")


def gold_names(sim):
    return [(a.get("action") or {}).get("name") or "" for a in F.gold_actions(sim)]


def gold_hits(sim):
    return sum(1 for a in F.gold_actions(sim) if a.get("action_match"))


def transfer_index(sim):
    """궤적에서 **어시스턴트가** 이관을 부른 첫 위치(호출 순번). 없으면 None."""
    for i, lb in enumerate(F.call_labels(sim)):
        base = lb.split(":")[-1]
        if base in TRANSFER or any(t in base for t in ("transfer_to_human", "human_agent_transfer")):
            return i
    return None


def run(tags):
    rows = []
    for tag in tags:
        for s in F.scored(tag):
            gn = gold_names(s)
            n_gold = len(gn)
            hit = gold_hits(s)
            ti = transfer_index(s)
            rows.append({
                "tag": tag, "sim": F.sim_key(s), "task": F.task_id(s),
                "reward": (s.get("reward_info") or {}).get("reward"),
                "term": F.term_reason(s),
                "gold_transfer": any(n in TRANSFER for n in gn),
                "transferred": ti is not None,
                "calls": len(F.call_labels(s)),
                "at": ti, "gold": n_gold, "ok": hit, "left": n_gold - hit,
                "tail": " ".join(F.assistant_text(s).split())[:110],
            })

    print("=" * 104)
    print("%-12s %-6s %5s %5s %5s  %-9s %-9s %s" % (
        "sim", "reward", "gold", "OK", "남은", "gold이관", "실이관", "종료 직전 본문"))
    print("-" * 104)
    for r in sorted(rows, key=lambda x: (-int(x["transferred"] and not x["gold_transfer"]),
                                         x["task"])):
        print("%-12s %-6s %5d %5d %5d  %-9s %-9s %s" % (
            r["sim"], r["reward"], r["gold"], r["ok"], r["left"],
            "예" if r["gold_transfer"] else "아니오",
            ("예(#%s)" % r["at"]) if r["transferred"] else "아니오", r["tail"]))

    bail = [r for r in rows if r["transferred"] and not r["gold_transfer"]]
    legit = [r for r in rows if r["transferred"] and r["gold_transfer"]]
    none = [r for r in rows if not r["transferred"]]
    print("\n# 요약 (sim %d)" % len(rows))
    print("  gold 가 이관을 요구한 sim        : %d" % len(legit))
    print("  gold 에 없는데 이관한 sim(이탈)  : %d" % len(bail))
    print("  이관 없음                        : %d" % len(none))
    if bail:
        left = [r["left"] for r in bail]
        print("  ⤷ 이탈 sim 이 포기한 gold 액션   : 합 %d · 중앙 %d · 최대 %d"
              % (sum(left), sorted(left)[len(left) // 2], max(left)))
        print("  ⤷ 이탈 sim 의 reward             : %s"
              % dict(collections.Counter(str(r["reward"]) for r in bail)))
        print("  ⤷ 태스크별                       : %s"
              % dict(collections.Counter(r["task"] for r in bail)))
    if none:
        print("  ⤷ 이관 없이 실패한 sim 의 남은 gold: 합 %d"
              % sum(r["left"] for r in none if r["reward"] != 1.0))
    print("\n※ 관측 전용. 이 표는 '이관이 나쁘다'를 말하지 않는다 — gold 이관과 이탈을 가를 뿐이다.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1:])
