#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""x371 — t7307 **부분** 판정 (창 순환으로 5/24 에서 중단·2026-08-18 밤).

런처 헤더의 **사전 고정 판정 ⓐ~ⓔ** 를 *잴 수 있는 만큼만* 적용한다.

## 계기 규율 (결론보다 먼저 적는다)

  ⚠**분모 순환 방지** — ⓐ 의 분모(*"손님-측 이름을 발화한 sim"*)를 우리 마커에서 뽑으면
    분자와 **같은 사건**이라 비율이 정의상 높아진다. t7303 ⓑ 가 정확히 그래서 무효였다
    (C502). 그래서 분모는 **궤적 어시스턴트 본문**에서 직접 검출한다 — 술어는
    `t2_search.handoff_missing` 과 같되 `given` 조건 **없이**(= 건넸든 아니든 *발화* 자체).
    이름 집합의 출처는 **env 레지스트리**(`get_discoverable_tools`)뿐이다(gold 0·[[23]]).
  ⚠**양성통제** — 마커가 뜬 sim 은 반드시 분모에 들어야 한다. 안 들면 **계기 결함**으로
    보고하고 비율을 인용하지 않는다([[08]]·C524 *"양성통제 없는 0 은 결손이 아니다"*).
  ⚠**pass 는 `reward` 로만**(C486 — `action_match` 는 소수점 표기로 무너진다).
  ⚠**중단 sim 은 궤적이 없다** ⇒ 분모를 못 재므로 ⓐ 에서 **뺀다**. 대신 로그-수준
    관측치를 **따로·이름 붙여** 인쇄한다(둘을 한 비율로 섞지 않는다).
  ⚠[[54]]/[[68]] 무관(리더보드 아님) · 이 런은 **승격 판단에 쓰지 않는다**(n 부족·헤더 ⓓ).

사용: PYTHONPATH=<tau2>/src python x371_t7307_judge.py
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_forensic as F  # noqa: E402

CTL = "bank_t7307_ctl_20260818b"
TREAT = "bank_t7307_treat_20260818b"
SUFFIX = ".results.json.gz"


def discoverable_names():
    """손님-측 discoverable 도구 이름 — **env 레지스트리에서만**(gold·도메인 어휘 0)."""
    try:
        from tau2.domains.banking_knowledge.environment import get_environment
        env = get_environment()
        return sorted(env.user_tools.get_discoverable_tools())
    except Exception as e:  # 못 얻으면 **추정하지 않는다** — 빈 집합 = 침묵([[25]])
        print("  ⚠env 레지스트리 획득 실패 → ⓐ 판정 불가: %r" % (e,))
        return []


def uttered(sim, names):
    """어시스턴트 본문에 손님-측 이름이 등장한 첫 index(없으면 None) — 마커 무관."""
    return F.first_named(sim, names) if names else None


def gave(sim):
    """`give_discoverable_user_tool` 를 실제로 호출한 sim 인가."""
    return any(F.nameof(tc) == "give_discoverable_user_tool" for tc in F.calls(sim))


def user_side_calls(sim):
    """손님(requestor=user)이 부른 도구 호출 수 — 절차가 끝까지 갔는지의 지표(ⓒ)."""
    n = 0
    for m in (sim.get("messages") or []):
        if str(m.get("role")) != "user":
            continue
        n += len(m.get("tool_calls") or [])
    return n


def marker_sims(tag, pattern):
    """로그에서 `pattern` 이 뜬 **sim 집합**(고유)."""
    try:
        return set(F.by_sim(tag, pattern).keys())
    except Exception as e:
        print("  ⚠로그 스캔 실패(%s): %r" % (tag, e))
        return set()


def arm_table(tag, names):
    rows = []
    for s in F.scored(tag, SUFFIX):
        rows.append(dict(
            task=F.task_id(s),
            key=F.simtag(s),
            reward=(s.get("reward_info") or {}).get("reward"),
            uttered=uttered(s, names) is not None,
            gave=gave(s),
            usercalls=user_side_calls(s),
            steps=len(s.get("messages") or []),
            term=F.term_reason(s),
        ))
    return rows


def main():
    print("=" * 78)
    print("x371 — t7307 부분 판정 (5/24 완주 · 창 순환으로 중단)")
    print("=" * 78)

    names = discoverable_names()
    print("\n[출처] env discoverable 손님-측 도구 %d 종: %s" % (len(names), ", ".join(names)))

    # ── ⓐ 1차 종점 (완주 sim 한정 · 분모는 궤적에서) ────────────────────────────
    print("\n" + "-" * 78)
    print("ⓐ 1차 종점 = 술어 발화율 — **완주 sim 한정**(궤적 있는 것만)")
    print("-" * 78)
    fired = {a: marker_sims(t, r"\[T2_HANDOFF\] named-but-not-given")
             for a, t in (("ctl", CTL), ("treat", TREAT))}
    for arm, tag in (("ctl", CTL), ("treat", TREAT)):
        rows = arm_table(tag, names)
        den = [r for r in rows if r["uttered"]]
        num = [r for r in den if r["key"] in fired[arm]]
        print("\n  [%s] 완주 %d sim" % (arm, len(rows)))
        for r in rows:
            print("    %-9s %-22s reward=%-5s 이름발화=%-5s give=%-5s 손님호출=%d steps=%3d %s"
                  % (r["task"], r["key"], r["reward"], r["uttered"], r["gave"],
                     r["usercalls"], r["steps"], r["term"]))
        print("    → 분모(이름 발화) %d · 분자(마커 발화) %d" % (len(den), len(num)))
        # 양성통제: 마커가 뜬 완주 sim 이 분모 밖이면 계기 결함
        keys = {r["key"] for r in rows}
        bad = [k for k in (fired[arm] & keys) if k not in {r["key"] for r in den}]
        if bad:
            print("    ⚠계기 결함 — 마커는 떴는데 분모 밖인 sim: %s" % ", ".join(bad))

    # ── 로그-수준 관측(중단 sim 포함) — ⓐ 와 **섞지 않는다** ────────────────────
    print("\n" + "-" * 78)
    print("[로그-수준 관측] 중단 sim 포함 · **비율 아님**(분모를 못 잰다)")
    print("-" * 78)
    for arm, tag in (("ctl", CTL), ("treat", TREAT)):
        started = marker_sims(tag, r"\[sim=")
        print("  [%s] 로그에 나타난 sim %d · 마커 발화 sim %d %s"
              % (arm, len(started), len(fired[arm]), sorted(fired[arm])))

    # ── ⓑ W4 발화 확인 (양팔 공통이어야 정상) ──────────────────────────────────
    print("\n" + "-" * 78)
    print("ⓑ W4 — `pending_user` 에 `call_discoverable_user_tool` 등장(양팔 공통이어야 정상)")
    print("-" * 78)
    for arm, tag in (("ctl", CTL), ("treat", TREAT)):
        w4 = marker_sims(tag, r"pending_user=\[[^\]]*call_discoverable_user_tool")
        print("  [%s] 등장 sim %d: %s" % (arm, len(w4), sorted(w4)))

    # ── ⓒ 절차 지표 ────────────────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("ⓒ 절차 — give 호출 sim · 손님(requestor=user) 호출")
    print("-" * 78)
    for arm, tag in (("ctl", CTL), ("treat", TREAT)):
        rows = arm_table(tag, names)
        print("  [%s] 완주 %d 중 give 호출 %d · 손님 호출이 있는 sim %d"
              % (arm, len(rows), sum(r["gave"] for r in rows),
                 sum(1 for r in rows if r["usercalls"] > 0)))
        deny = marker_sims(tag, r"\[T2_TOOL_SIGNATURE\] deny tool=give_discoverable_user_tool")
        print("       give 시그니처 deny 를 맞은 sim(중단 포함) %d: %s" % (len(deny), sorted(deny)))

    # ── ⓓ 2차 = pass (방향만 · 검정력 없음) ────────────────────────────────────
    print("\n" + "-" * 78)
    print("ⓓ pass — **방향만**(n 부족·승격 판단 금지)")
    print("-" * 78)
    for arm, tag in (("ctl", CTL), ("treat", TREAT)):
        rows = arm_table(tag, names)
        rw = [r["reward"] for r in rows]
        print("  [%s] %s → 합 %.1f / %d" % (arm, rw, sum(x or 0 for x in rw), len(rw)))
    # 짝지은 비교(같은 task·같은 seed 만)
    ct = {(r["task"], r["key"].split("#")[-1]): r["reward"] for r in arm_table(CTL, names)}
    tr = {(r["task"], r["key"].split("#")[-1]): r["reward"] for r in arm_table(TREAT, names)}
    both = sorted(set(ct) & set(tr))
    print("  짝지은 쌍 %d: %s" % (len(both), [(k[0], ct[k], tr[k]) for k in both]))

    # ── ⓔ 부작용 ───────────────────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("ⓔ 부작용 — 크래시 · 출력상한 · 창 순환(중단 원인)")
    print("-" * 78)
    for arm, tag in (("ctl", CTL), ("treat", TREAT)):
        for pat, lab in ((r"Traceback|CRITICAL|CWE", "크래시/예외"),
                         (r"output limit|truncguard|max_tokens", "출력상한"),
                         (r"\[T2_WINDOW\] open=resign", "resign 창"),
                         (r"same fingerprint \(seen=", "동일지문 접힘")):
            hits = F.by_sim(tag, pat)
            tot = sum(len(v) for v in hits.values())
            print("  [%s] %-12s sim %d · 총 %d %s"
                  % (arm, lab, len(hits), tot,
                     sorted(((k, len(v)) for k, v in hits.items()), key=lambda x: -x[1])[:3]))

    print("\n" + "=" * 78)
    print("판정은 이 출력 위에서만 쓴다. 인용 前 §계기 규율 4항을 다시 읽을 것.")
    print("=" * 78)


if __name__ == "__main__":
    main()
