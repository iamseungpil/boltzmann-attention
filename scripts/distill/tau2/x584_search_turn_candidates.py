# -*- coding: utf-8 -*-
r"""x584 — `_cand9` 를 **검색하려는 턴**으로 옮기면 반경이 얼마나 되나 (모델 0 · 무료 · 계수만).

## 왜 (2026-08-28 밤 · x583 에서 파생)

`x583`: 쓰기 도구를 못 부른 20 sim **전부**에서 그 이름은 이미 `role=tool` 본문으로 배달돼
있었고 그 20개는 전부 reward 0 이다 = **부하이지 능력 결손이 아니다**([[62]] §1.4).
재료는 `t2_gate_patch.py` 의 `_cand9`(레지스트리 ∩ 배달된 텍스트 − 이미 호출/해제)에 이미 있는데
게이트가 `_stubs>=2 AND _resign` 이라 **낭비를 먼저 해야 도착한다**(t7375 에선 pass 뒤에 왔다).

제안: 그 목록을 **에이전트가 또 검색하려는 턴**에 되짚는다. 배선 전에 반경을 잰다([[70]]·§6-1).

## 무엇을 세나 (닫힌 술어뿐 · gold 무참조 · 판단 0)

각 sim 을 시간순으로 걸으며 유지한다:
    delivered  `role=tool` 본문에 나온 **발견형 도구 이름**(`이름_숫자4`) 누적
    used       도구 호출 인자·이름에 나온 그 이름들 누적(해제 포함)
`KB_search*` 를 부르는 턴마다  cand = delivered − used  의 **크기와 내용**을 기록한다.

물음 셋:
    ⑴ 새로 발화하게 되는 **턴 수**는 몇인가 (반경)
    ⑵ 그 턴의 후보 목록은 **얼마나 긴가** — 길면 레지스트리 열거에 가까워 [[05]] Q3 위반이고
       over-action 위험이 커진다([[70]] 반대편 계측)
    ⑶ 못 닿은 sim 에서 **표적 도구가 그 후보 안에 실제로 들어 있나** (있어야 레버가 산다)

⛔이 프로브는 아무것도 고치지 않는다. 효과가 아니라 **반경**을 잰다.
⛔레지스트리를 a2 에서 못 읽으므로 이름 규칙(`[a-z][a-z0-9_]*_\d{4}`)으로 뽑는다 = 근사다.
"""
import collections
import gzip
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results"))
RE_DISC = re.compile(r"\b[a-z][a-z0-9_]*_\d{4}\b")
TARGET = "apply_checking_account_credit_5829"
TAGS = ["bank_t7376_treat_20260828", "bank_t7372_control_20260828", "bank_t7375_072_20260828",
        "bank_t7369_072_20260828", "bank_t7370_radius_20260828", "bank_t7368_hard0_20260827"]


def walk(sim):
    """검색 턴마다 (index, cand set) 을 낸다."""
    delivered, used = set(), set()
    events, reached = [], None
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") == "tool":
            delivered |= set(RE_DISC.findall(str(m.get("content") or "")))
        searched = False
        for tc in (m.get("tool_calls") or []):
            nm = str(tc.get("name") or "")
            a = tc.get("arguments")
            a = a if isinstance(a, str) else json.dumps(a or {})
            if nm.startswith("KB_search"):
                searched = True
            hit = set(RE_DISC.findall(a)) | set(RE_DISC.findall(nm))
            if TARGET in hit and "unlock" not in nm and reached is None:
                reached = i
            used |= hit
        if searched:
            events.append((i, sorted(delivered - used)))
    return events, reached


def main(argv=None):
    tags = (argv or sys.argv[1:]) or TAGS
    rows, sizes, n_turn_fire, n_turn_all = [], [], 0, 0
    for tag in tags:
        p = os.path.join(BASE, tag + ".results.json.gz")
        if not os.path.exists(p):
            print("(건너뜀) %s" % tag); continue
        with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
            sims = json.load(f).get("simulations") or []
        for s in sims:
            ev, reached = walk(s)
            if not ev:
                continue
            fire = [(i, c) for i, c in ev if c]
            n_turn_all += len(ev); n_turn_fire += len(fire)
            sizes.extend(len(c) for _, c in fire)
            tgt_in = [i for i, c in fire if TARGET in c]
            rows.append({"tag": tag, "sim": "%s#s%s" % (s.get("task_id"), s.get("seed")),
                         "reward": (s.get("reward_info") or {}).get("reward"),
                         "search_turns": len(ev), "would_fire": len(fire),
                         "cand_max": max([len(c) for _, c in fire] or [0]),
                         "target_offered_at": tgt_in[:3], "reached_at": reached})
    if not rows:
        print("행 0"); return 1

    print("%-8s %-22s %-5s %-7s %-7s %-7s %-18s %s"
          % ("런", "sim", "rew", "검색턴", "발화턴", "후보max", "표적을 짚는 턴", "실호출"))
    for r in rows:
        print("%-8s %-22s %-5s %-7d %-7d %-7d %-18s %s"
              % (r["tag"].split("_")[1], r["sim"], r["reward"], r["search_turns"],
                 r["would_fire"], r["cand_max"], r["target_offered_at"] or "-",
                 r["reached_at"] if r["reached_at"] is not None else "X"))

    miss = [r for r in rows if r["reached_at"] is None]
    miss_off = [r for r in miss if r["target_offered_at"]]
    print("")
    print("=" * 96)
    print("⑴ 반경: 검색 턴 %d 중 **후보가 있어 발화하게 되는 턴 %d** (%.0f%%)"
          % (n_turn_all, n_turn_fire, 100.0 * n_turn_fire / max(1, n_turn_all)))
    if sizes:
        ss = sorted(sizes)
        print("⑵ 후보 목록 크기: 중앙값 %d · 평균 %.1f · 최대 %d  (현행 코드는 6개로 자른다)"
              % (ss[len(ss) // 2], sum(ss) / len(ss), ss[-1]))
        print("   크기 분포: %s" % dict(collections.Counter(ss)))
    print("⑶ 표적을 못 부른 sim %d 중 **그 검색 턴에 표적이 후보로 제시됐을 것 = %d**"
          % (len(miss), len(miss_off)))
    print("   (제시 안 되는 %d 개는 이 레버가 못 산다)" % (len(miss) - len(miss_off)))

    dst = os.path.join(BASE, "..", "x584_search_turn_candidates.json")
    with io.open(os.path.normpath(dst), "w", encoding="utf-8") as f:
        json.dump({"probe": "x584_search_turn_candidates", "date": "2026-08-28",
                   "target": TARGET, "tags": tags, "rows": rows,
                   "limits": ["레지스트리를 이름 규칙으로 근사했다(a2 에 목록이 없다).",
                              "반경이지 효과가 아니다 — 발화가 행동을 바꾸는지는 런이 답한다.",
                              "gold 무참조 — reward 는 참고 열이다([[23]])."]},
                  f, ensure_ascii=False, indent=1)
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
