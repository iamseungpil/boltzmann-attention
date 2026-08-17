# -*- coding: utf-8 -*-
r"""x369 — **우리 배선 결함이 실제로 gold 절차를 몇 번 막았나** (오프라인·LLM 0·비용 0).

## 왜 (2026-08-17 워크플로 판정 §5-1)

배선 결함 6종이 확정됐지만 **A/B 가 0회**라 *"배선이 몇 pp 를 먹는가"* 를 말할 수 없다.
유료 런 없이 그 몫의 **상한**을 재는 유일한 경로가 궤적 재독이다. 세 가지를 센다:

    ① `operator-fab` deny 777회  — 거부된 이름이 **env 레지스트리에 실재**했는가
                                    (`T2_PROV_OURS` 死배선의 손해 상한)
    ② give 서명 deny 729회       — deny 뒤 **인자 없이 재발행해 성공**했는가
                                    (실질 차단인가 마찰인가)
    ③ A2 `action_tools` 상수      — 태스크별 `user_tools` 와 **불일치**가 몇 태스크인가
                                    (019 가족 손님-실행 분기가 구조적으로 막히는 범위)

## 읽는 법 (사전 고정)

    ①에서 실재율이 높다  → 우리가 **참인 이름을 발명이라 막았다** ⇒ 배선 손해 [S]
    ②에서 재발행 성공률 높음 → 마찰(지연·턴 소모)이지 차단 아님 ⇒ 손해 상한이 낮다
    ②에서 재발행 없음/실패 → **실질 차단** ⇒ 손해 [S]
    ③ 불일치 태스크 수 = 그 분기가 못 도는 범위(상한이지 실손해 아님)

⚠**상한이지 인과가 아니다**([[08]]). "막혔다"가 곧 "막지 않았으면 통과했다"는 아니다 —
  인과는 A/B 뿐이고, 이 계수는 그 A/B 를 **설계할 값어치가 있는지**를 정한다.

실행: /home/woori/venvs/seka_env/bin/python x369_wiring_damage_audit.py
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

import t2_forensic as F                                           # noqa: E402
import x364_eligibility_axis_iso as E                             # noqa: E402

SIMDIR = os.path.join(E.REPORTS, "sim_results")
LOGDIRS = (SIMDIR, "/home/woori/scratch/logs")


def registries():
    from tau2.domains.banking_knowledge import tools as T
    attr = getattr(T, "DISCOVERABLE_ATTR", "__discoverable__")

    def marked(cls):
        return set(n for n in dir(cls or ()) if not n.startswith("_")
                   and callable(getattr(cls, n, None))
                   and getattr(getattr(cls, n), attr, False))
    return marked(getattr(T, "KnowledgeUserTools", None)), marked(getattr(T, "KnowledgeTools", None))


def logs():
    """로그를 파일별로 연다(gz/plain 둘 다·[[30]] 파이프 버퍼 함정 회피)."""
    for d in LOGDIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            try:
                if fn.endswith(".log.gz"):
                    with gzip.open(p) as f:
                        yield fn, io.TextIOWrapper(f, encoding="utf-8", errors="replace").read()
                elif fn.endswith(".log"):
                    yield fn, io.open(p, encoding="utf-8", errors="replace").read()
            except Exception:
                continue


def main():
    udisc, adisc = registries()
    allnames = udisc | adisc
    print("x369 · env 레지스트리 손님-측 %d · 에이전트-측 %d" % (len(udisc), len(adisc)))

    # ── ① operator-fab: 거부된 이름이 실재했는가
    fab = collections.Counter()
    fab_real, fab_lines = 0, 0
    for fn, txt in logs():
        for ln in txt.split("\n"):
            if "operator-fab" not in ln:
                continue
            fab_lines += 1
            hit = sorted(n for n in allnames if n in ln)
            if hit:
                fab_real += 1
                for h in hit:
                    fab[h] += 1
    print("\n① operator-fab 줄 %d · 그중 **env 실재 이름을 담은 줄 %d** (%.0f%%)"
          % (fab_lines, fab_real, 100.0 * fab_real / max(fab_lines, 1)))
    for k, v in fab.most_common(10):
        print("     %-46s %d" % (k, v))

    # ── ② give 서명 deny 뒤 재발행 성공
    reissue = collections.Counter()
    for fn in sorted(os.listdir(SIMDIR)) if os.path.isdir(SIMDIR) else ():
        if not fn.endswith("_results.json.gz"):
            continue
        try:
            with gzip.open(os.path.join(SIMDIR, fn)) as f:
                data = json.load(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for s in (data.get("simulations") or data.get("results") or ()):
            gives = []
            for m in (s.get("messages") or ()):
                for tc in (m.get("tool_calls") or ()):
                    if F.nameof(tc) == "give_discoverable_user_tool":
                        a = F.argsof(tc)
                        gives.append((str(a.get("discoverable_tool_name") or ""),
                                      "arguments" in a))
            # 같은 도구를 **인자 있이** 부른 뒤 **인자 없이** 다시 부른 적이 있는가
            for i, (nm, witharg) in enumerate(gives):
                if not (nm and witharg):
                    continue
                later = any(n2 == nm and not w2 for n2, w2 in gives[i + 1:])
                reissue["재발행 성공" if later else "재발행 없음"] += 1
            reissue["give 총호출"] += len(gives)
            reissue["인자 붙여 부름"] += sum(1 for _n, w in gives if w)
    print("\n② give 호출 %d · 인자 붙여 부른 것 %d · 그중 **인자 없이 재발행 %d** · 재발행 없음 %d"
          % (reissue["give 총호출"], reissue["인자 붙여 부름"],
             reissue["재발행 성공"], reissue["재발행 없음"]))

    # ── ③ A2 action_tools 상수 ↔ 태스크 user_tools 불일치
    a2 = {}
    for name in ("banking_knowledge.settings.json", "banking_knowledge.specific.json"):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2", name)
        if os.path.exists(p):
            a2.update(json.load(io.open(p, encoding="utf-8")))
    acts = set()
    for v in (a2.get("action_tools") or ()):
        acts.add(str(v if isinstance(v, str) else (v or {}).get("name") or ""))
    print("\n③ A2 action_tools = %s" % (sorted(x for x in acts if x) or "(없음)"))
    miss = collections.Counter()
    rows = []
    for fn in sorted(os.listdir(E.M.TASKS_DIR)):
        if not fn.endswith(".json"):
            continue
        t = json.load(io.open(os.path.join(E.M.TASKS_DIR, fn), encoding="utf-8"))
        ut = set(str(u) for u in (t.get("user_tools") or ()))
        gap = sorted(ut - acts)
        if gap:
            rows.append((fn[:-5], gap))
            for g in gap:
                miss[g] += 1
    print("   태스크별 user_tools 중 A2 에 **없는** 것: %d 태스크" % len(rows))
    for k, v in miss.most_common():
        print("     %-34s %d 태스크" % (k, v))
    print("   해당 태스크: %s" % ", ".join(t[5:] for t, _g in rows[:40]))

    out = os.path.join(E.REPORTS, "x369_wiring_damage.json")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"fab_lines": fab_lines, "fab_real": fab_real,
                            "fab_by_name": dict(fab), "reissue": dict(reissue),
                            "a2_action_tools": sorted(acts), "user_tool_gap": rows},
                           ensure_ascii=False, indent=1, default=str))
    print("\n저장: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
