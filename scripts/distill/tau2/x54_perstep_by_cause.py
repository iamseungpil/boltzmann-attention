"""Per-step forensic, organised by cause rather than by task.

Task-level attribution does not survive a user simulator that rewrites the scenario
each run: the same task lands in a different cause class from run to run, so a lever
that does nothing still looks like it moved tasks around while the totals hold. The
unit that can move is the cause class, and the evidence for a cause is the step where
the run went wrong.

For every failed simulation this locates the decisive step for its class and reports
what was true *before* that step — not somewhere in the transcript, which is the
measurement error §1.4 warns about. A value that arrives after the decision cannot
explain the decision.

  user 실행·인자 불일치   the agent message that set the operand the customer then used
  ARG만                  the call whose arguments diverged from gold
  NEVER·정보도 없음        the last agent turn, with what retrieval had produced by then
  user 미실행             the last agent turn, and whether the tool was ever handed over
  말로 바꿈               the message claiming completion, and the calls made before it
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x50_says_not_does import (ARMS, DONE, SIM, fam, norm_args,  # noqa: E402
                               user_calls, values_present, variants)

LEVER = re.compile(r"\[(T2_[A-Z0-9_]+|GROUNDING WARNING|DUPLICATE-READ|coverage|matches)")


def load(pattern):
    out = []
    for p in sorted(glob.glob(f"{SIM}/{pattern}.results.json.gz")):
        out.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])
    return out


def context_before(sim, i):
    """Everything readable strictly before step i."""
    parts = []
    for m in (sim.get("messages") or [])[:i]:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        for tc in m.get("tool_calls") or []:
            a = tc.get("arguments")
            if a is None:
                a = (tc.get("function") or {}).get("arguments")
            parts.append(a if isinstance(a, str) else json.dumps(a or {}))
    return "\n".join(parts).lower()


def levers_before(sim, i):
    c = collections.Counter()
    for m in (sim.get("messages") or [])[:i]:
        if isinstance(m.get("content"), str):
            for t in LEVER.findall(m["content"]):
                c[t] += 1
    return c


def agent_calls(sim):
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name")
            a = tc.get("arguments")
            if a is None:
                a = (tc.get("function") or {}).get("arguments")
            args = norm_args(a)
            out.append((i, n, args))
            inner = (args.get("agent_tool_name") or args.get("discoverable_tool_name")
                     or args.get("user_tool_name"))
            if inner:
                ia = args.get("arguments")
                out.append((i, inner, norm_args(ia) if ia is not None else {}))
    return out


def user_call_steps(sim):
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "user":
            continue
        for tc in m.get("tool_calls") or []:
            out.append((i, tc.get("name") or "", norm_args(tc.get("arguments"))))
    return out


def last_agent_text_step(sim):
    last = None
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip():
            last = i
    return last


def classify(sim):
    """Same predicates as x50, returning the class plus the actions that produced it."""
    checks = (sim.get("reward_info") or {}).get("action_checks") or []
    a_missed = [c for c in checks if not c.get("action_match")
                and (c.get("action") or {}).get("requestor") == "assistant"]
    called = {fam(n) for _, n, _ in agent_calls(sim)}
    hay = context_before(sim, len(sim.get("messages") or []))

    if a_missed:
        kinds = set()
        for c in a_missed:
            g = c.get("action") or {}
            gargs = norm_args(g.get("arguments"))
            pres, nchk = values_present(gargs, hay)
            if fam(g.get("name")) in called:
                kinds.add("ARG")
            elif nchk and pres == nchk:
                kinds.add("SAYS")
            else:
                kinds.add("NEVER")
        cls = ("말로 바꿈" if "SAYS" in kinds else
               "ARG만" if kinds == {"ARG"} else
               "NEVER·정보도 없음" if kinds == {"NEVER"} else "ARG+NEVER 혼합")
        return cls, a_missed

    u_missed = [c for c in checks if not c.get("action_match")
                and (c.get("action") or {}).get("requestor") == "user"]
    uc = user_calls(sim)
    if any(fam(n) == fam((c.get("action") or {}).get("name")) for c in u_missed for n, _ in uc):
        return "user 실행·인자 불일치", u_missed
    if u_missed:
        return "user 미실행", u_missed
    return "DB만 불일치", []


def report(sim, cls, missed):
    """The decisive step for this class, and what held before it."""
    msgs = sim.get("messages") or []
    key = f"{sim['task_id']}/t{sim.get('trial')}"
    ac, uc = agent_calls(sim), user_call_steps(sim)
    lines = []

    def before(i):
        lv = levers_before(sim, i)
        return context_before(sim, i), lv

    if cls == "user 실행·인자 불일치":
        for c in missed:
            g = c.get("action") or {}
            gname, gargs = g.get("name"), norm_args(g.get("arguments"))
            hits = [(i, a) for i, n, a in uc if fam(n) == fam(gname)]
            if not hits:
                continue
            i, used = hits[0]
            hay, lv = before(i)
            diff = [k for k in sorted(set(gargs) | set(used)) if gargs.get(k) != used.get(k)]
            # Was the right value already stated somewhere the model could see?
            avail = {k: any(x in hay for x in variants(gargs.get(k))) for k in diff}
            guide = max([j for j in range(i) if msgs[j].get("role") == "assistant"
                         and isinstance(msgs[j].get("content"), str)
                         and msgs[j]["content"].strip()], default=None)
            lines.append(f"    손님 실행 step {i} · 틀린 키 {diff}")
            lines.append(f"      정답 값이 그 전에 문맥에 있었나: {avail}")
            lines.append(f"      직전 안내 step {guide}: "
                         f"{' '.join((msgs[guide].get('content') or '').split())[:150] if guide else '-'}")
            lines.append(f"      그 전까지 발화된 레버: {dict(lv) or '없음'}")
    elif cls in ("ARG만", "ARG+NEVER 혼합"):
        for c in missed:
            g = c.get("action") or {}
            gname, gargs = g.get("name"), norm_args(g.get("arguments"))
            tried = [(i, a) for i, n, a in ac if fam(n) == fam(gname)]
            if not tried:
                continue
            i, used = min(tried, key=lambda e: sum(
                1 for k in set(gargs) | set(e[1]) if gargs.get(k) != e[1].get(k)))
            hay, lv = before(i)
            diff = [k for k in sorted(set(gargs) | set(used)) if gargs.get(k) != used.get(k)]
            avail = {k: any(x in hay for x in variants(gargs.get(k))) for k in diff}
            lines.append(f"    호출 step {i} · {fam(gname)} · 틀린 키 {diff}")
            lines.append(f"      정답 값이 그 전에 문맥에 있었나: {avail}")
            lines.append(f"      그 전까지 발화된 레버: {dict(lv) or '없음'}")
    elif cls == "말로 바꿈":
        claim = next((i for i, m in enumerate(msgs)
                      if m.get("role") == "assistant" and isinstance(m.get("content"), str)
                      and DONE.search(m["content"] or "")), last_agent_text_step(sim))
        hay, lv = before(claim)
        names = [fam((c.get("action") or {}).get("name")) for c in missed]
        lines.append(f"    완료 주장 step {claim} · 그때까지 도구 호출 "
                     f"{sum(1 for i, _, _ in ac if i < claim)}회")
        lines.append(f"      놓친 gold: {names}")
        lines.append(f"      그 전까지 발화된 레버: {dict(lv) or '없음'}")
        lines.append(f"      문구: {' '.join((msgs[claim].get('content') or '').split())[:150]}")
    else:  # NEVER, user 미실행, DB만
        i = last_agent_text_step(sim) or len(msgs) - 1
        hay, lv = before(i)
        names = [fam((c.get("action") or {}).get("name")) for c in missed]
        present = {n: (n.lower() in hay) for n in names}
        lines.append(f"    마지막 발화 step {i} / 전체 {len(msgs)} · 도구 호출 {len(ac)}회 "
                     f"· KB_search {sum(1 for _, n, _ in ac if (n or '').startswith('KB_search'))}회")
        lines.append(f"      놓친 gold 이름이 그 전에 문맥에 있었나: {present}")
        lines.append(f"      그 전까지 발화된 레버: {dict(lv) or '없음'}")
        lines.append(f"      문구: {' '.join((msgs[i].get('content') or '').split())[:150]}")
    return key, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B4", choices=sorted(ARMS))
    ap.add_argument("--only", help="single cause class")
    args = ap.parse_args()

    sims = load(ARMS[args.arm])
    fails = [s for s in sims if ((s.get("reward_info") or {}).get("reward") or 0.0) != 1.0]
    groups = collections.defaultdict(list)
    for s in fails:
        cls, missed = classify(s)
        groups[cls].append((s, missed))

    print(f"[{args.arm}] 실패 {len(fails)} · 원인 클래스 {len(groups)}\n")
    for cls in sorted(groups, key=lambda k: -len(groups[k])):
        if args.only and args.only != cls:
            continue
        print(f"{'=' * 74}\n{cls}  —  {len(groups[cls])} sim\n{'=' * 74}")
        for s, missed in sorted(groups[cls], key=lambda x: (x[0]["task_id"], x[0].get("trial") or 0)):
            key, lines = report(s, cls, missed)
            print(f"  {key}")
            for ln in lines:
                print(ln)
        print()


if __name__ == "__main__":
    main()
