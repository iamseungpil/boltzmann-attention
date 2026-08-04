"""How often would a sim-scope pin fire, and would it aim at a tool the task needed?

The turn-scope count (77 declared-missing events) bounded a lever that cannot open in the
case it exists for: both sources of that count are computed inside a loop over the calls
the model just made, so a turn with no calls produces an empty set — and the no-call regen
is the 281-firing case P1 is aimed at (review of 2026-08-05, ★1).

Sim scope asks a different question, and one that does not need a call to answer: of the
prerequisite reads A2 declares, which has this simulation not executed yet? That set is
non-empty from the first turn, so the bound is no longer an event count — it is the number
of simulations reaching a regen while the set is non-empty, which the 1-per-sim cap turns
into "at most one pin per simulation".

What is worth pre-registering is therefore not how often it fires but WHERE IT AIMS. The
order rule is fixed here on principle, not by score:

  1. a declared prerequisite whose dependent tool this simulation already attempted
     (closed: it is in the call history) — the model has shown what it is trying to do
  2. otherwise A2 declaration order

Whether the aimed-at tool was in gold is then measured, not designed against ([[03b]]:
gold may score a rule, never author one).
"""

import argparse
import collections
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x50_says_not_does import ARMS, SIM, fam, norm_args  # noqa: E402
from x66_effective_tool_miss import agent_actions  # noqa: E402

A2DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")
_READ_PREFIX = ("get_", "list_", "check_", "search_", "view_", "read_", "fetch_")


def declarations(domain="banking_knowledge"):
    """(dependent tool → [prerequisite reads]) from both A2 keys, in declaration order."""
    out = []
    for suffix in ("specific", "gate"):
        p = os.path.join(A2DIR, "%s.%s.json" % (domain, suffix))
        if not os.path.isfile(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        for dep, reads in (d.get("require_tool_before") or {}).items():
            out.append((dep, list(reads or [])))
        for entry in (d.get("scaffold_get_tools") or []):
            if isinstance(entry, dict) and entry.get("requires_reads"):
                dep = entry.get("tool") or entry.get("name") or entry.get("get_tool") or "?"
                out.append((dep, list(entry["requires_reads"])))
        break        # specific wins; gate is its byte-identical mirror ([[24]])
    return out


def load(pattern):
    sims = []
    for p in sorted(glob.glob(f"{SIM}/{pattern}.results.json.gz")):
        sims.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])
    return sims


def gold_names(sim):
    out = set()
    for c in ((sim.get("reward_info") or {}).get("action_checks") or []):
        g = c.get("action") or {}
        a = norm_args(g.get("arguments"))
        inner = (a.get("agent_tool_name") or a.get("discoverable_tool_name")
                 or a.get("user_tool_name"))
        out.add(fam(inner or g.get("name") or ""))
    return out


def refcount(decls):
    """How many declared procedures depend on each prerequisite read.

    This is the closed, A2-derived measure of how much a missing read blocks: a read four
    declarations point at gates four paths, one that a single declaration points at gates
    one. Nothing here looks at a task or at gold.
    """
    c = collections.Counter()
    for _dep, reads in decls:
        for r in reads:
            c[fam(r)] += 1
    return c


def aim(decls, called_before, attempted_before, rule="hub", demanded=()):
    """The read a sim-scope pin would target, under a fixed order rule.

    decl   declaration order — the naive reading of "first item"
    hub    most-depended-on prerequisite first, ties by declaration order
    both rules put a prerequisite whose dependent this sim already attempted ahead of the
    rest, because the model has then shown which procedure it is trying to run.
    """
    rc = refcount(decls)
    ranked, seen = [], set()
    for i, (dep, reads) in enumerate(decls):
        for j, r in enumerate(reads):
            r = fam(r)
            if r in called_before or not r.startswith(_READ_PREFIX) or r in seen:
                continue
            # demand2h: 선언 하나만 의존하는 선행 read는 고정 대상에서 뺀다 — 막고 있는 경로가
            #   하나뿐이라 "무엇을 하려는지"의 증거가 약하다(A2에서 기계 계수·gold 무관).
            if rule == "demand2h" and rc[r] < 2:
                continue
            seen.add(r)
            attempted = 0 if (fam(dep) in attempted_before or r in demanded) else 1
            if rule.startswith("demand") and attempted:
                continue          # 수요 신호가 없으면 고정하지 않는다
            order = (i, j) if rule == "decl" else (-rc[r], i, j)
            ranked.append((attempted, order, r))
    if not ranked:
        return None
    ranked.sort(key=lambda t: (t[0], t[1]))
    return ranked[0][2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N97")
    ap.add_argument("--rule", default="hub", choices=["hub", "decl", "demand", "demand2", "demand3", "demand2h"])
    ap.add_argument("--trial", type=int, default=None,
                    help="0 = 규칙을 고른 절반 · 1 = 확인용 절반 (★A 분할 검증)")
    ap.add_argument("--list-misaim", action="store_true",
                    help="gold가 요구하지 않는 read를 고정하는 sim을 전부 찍는다 (★B)")
    ap.add_argument("--at", default="any", choices=["any", "first", "start"],
                    help="first = at the first assistant turn that made no call "
                         "(the regen case); start = before anything happened")
    args = ap.parse_args()

    sims = load(ARMS[args.arm])
    if args.trial is not None:
        sims = [x for x in sims if x.get("trial") == args.trial]
    decls = declarations()
    print("A2 선언 (의존도구 → 선행 read):")
    for dep, reads in decls:
        print("  %-40s %s" % (dep, ", ".join(reads)))
    print()

    tally = collections.Counter()
    targets = collections.Counter()
    by_signal = collections.defaultdict(lambda: [0, 0])
    misaim = []
    for s in sims:
        acts = agent_actions(s)
        called, attempted, demanded = set(), set(), set()
        why = set()          # 어느 신호가 수요를 세웠나: a=의존시도 b=READ-FIRST c=레코드부재
        target = None
        if args.at == "start":
            target = aim(decls, called, attempted, args.rule, demanded)
            tally["창 열림"] += 1 if target else 0
        else:
            # Walk the turns; the first assistant message with no tool call is where the
            # no-call regen happens, and the pin would be computed with what is known then.
            for m in s.get("messages") or []:
                if m.get("role") == "tool":
                    c = m.get("content")
                    # demand2: 레코드를 못 찾았다는 env 오류도 수요 신호로 본다 — 087이 정확히
                    # 그 모양이다(`Account '<user_id>' not found` → 산문 → 재생성).
                    # demand3: 문자열을 안 본다 — 도구 호출이 **오류로 끝났다**는 사실만 본다
                    #   (env가 세운 error 플래그. 도메인 어휘 0·완전 닫힘).
                    if args.rule == "demand3" and m.get("error"):
                        demanded |= {fam(r) for _d, rs in decls for r in rs}
                        why.add("c")
                    if (args.rule in ("demand2", "demand2h") and isinstance(c, str)
                            and "not found" in c.lower()):
                        demanded |= {fam(r) for _d, rs in decls for r in rs}
                        why.add("c")
                    if isinstance(c, str) and "READ-FIRST" in c:
                        for _dep, _reads in decls:
                            for _r in _reads:
                                if fam(_r) in c:
                                    demanded.add(fam(_r))
                                    why.add("b")
                    continue
                if m.get("role") != "assistant":
                    continue
                tcs = m.get("tool_calls") or []
                if not tcs:
                    tally["무호출 턴"] += 1
                    t = aim(decls, called, attempted, args.rule, demanded)
                    if t and target is None:
                        target = t                 # 1회/sim 캡 — 첫 성립 지점에서만
                        tally["첫 고정 지점 도달"] += 1
                    if args.at == "first":
                        break
                    continue
                for tc in tcs:
                    a = norm_args(tc.get("arguments"))
                    n = tc.get("name") or ""
                    inner = (a.get("agent_tool_name") or a.get("discoverable_tool_name")
                             or a.get("user_tool_name"))
                    eff = fam(inner or n)
                    called.add(eff)
                    attempted.add(eff)
                    if any(fam(dep) == eff for dep, _r in decls):
                        why.add("a")
        if target:
            tally["고정 발화"] += 1
            targets[target] += 1
            hit = target in gold_names(s)
            sig = "".join(sorted(why)) or "-"
            by_signal[sig][0] += 1
            by_signal[sig][1] += 1 if hit else 0
            if hit:
                tally["표적이 gold에 있음"] += 1
            else:
                misaim.append((s["task_id"], s.get("trial"), target, sig))
            if (s.get("reward_info") or {}).get("reward") == 1:
                tally["(그 sim은 pass)"] += 1

    print("sim %d · 기준 시점 = %s · 순서 규칙 = %s" % (len(sims), args.at, args.rule))
    for k, v in tally.most_common():
        print("  %-22s %d" % (k, v))
    print("\n수요 신호별 (a=의존도구 시도 · b=우리 층 결손통지 · c=레코드 부재 오류):")
    for sig, (n, h) in sorted(by_signal.items()):
        print("  %-10s 발화 %3d · gold 적중 %3d (%.0f%%)" % (sig, n, h, 100.0 * h / max(1, n)))
    print("\n표적 분포:")
    for t, n in targets.most_common(10):
        print("  %-46s %d" % (t, n))
    if args.list_misaim:
        print("\n★오조준 %d건 (gold가 요구하지 않는 read를 고정):" % len(misaim))
        for tid, tr, t, sig in sorted(misaim):
            print("  %-10s t%s  %-42s 신호=%s" % (tid, tr, t, sig))


if __name__ == "__main__":
    main()
