# -*- coding: utf-8 -*-
"""Would refusing a give of a tool the environment does not have ever refuse a right one?

`task_012` handed the customer `navigate_to_travel_notification`, a name that exists
nowhere. Two checks should have caught it and neither did: the suffix rule only covers
the unlock dispatcher, and the membership rule runs against the agent's own tool list
unless `T2_DISPATCH_ROLE_ENVSET` is set — which no driver sets. The membership branch is
implemented and evidenced (C257) but has never been registered, so this counts what
turning it on would do before it is turned on.

The gate is one number: **gives that gold itself asks for must all be inside the set**.
If even one is outside, the deny would block a correct answer and the lever stays off.
The suffix rule is deliberately not extended to gives here — two of banking's four
user-side discoverable tools carry no numeric suffix, so that rule would refuse them
every time. Membership is the closed predicate; spelling is not.

Free: reads persisted trajectories and the environment's own registry. No model calls.

  usage: x88_give_membership_census.py [arm] [domain]
"""

import collections
import glob
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x50_says_not_does import ARMS, SIM  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "N97B"
DOMAIN = sys.argv[2] if len(sys.argv) > 2 else "banking_knowledge"


def env_user_discoverable(domain):
    """The set the ENVSET branch compares against — read from the environment, not guessed.

    Mirrors `ToolKit.get_discoverable_tools()`: the same attribute, on the same class the
    environment builds its user toolkit from. A census that hard-codes the four names
    would prove nothing about the check that will actually run.
    """
    from tau2.environment.toolkit import DISCOVERABLE_ATTR
    mod = __import__("tau2.domains.%s.tools" % domain, fromlist=["*"])
    out = {}
    for cls_name in dir(mod):
        cls = getattr(mod, cls_name)
        if not isinstance(cls, type):
            continue
        names = {n for n in dir(cls)
                 if getattr(getattr(cls, n, None), DISCOVERABLE_ATTR, False)}
        if names:
            out[cls_name] = names
    return out


def inner_name(a):
    a = a if isinstance(a, dict) else {}
    return a.get("discoverable_tool_name") or a.get("user_tool_name")


GIVE = "give_discoverable_user_tool"

try:
    by_class = env_user_discoverable(DOMAIN)
except Exception as e:
    raise SystemExit("env not importable here (run on the remote): %r" % (e,))

# The user-side class is the one whose discoverable tools the agent hands over. Both
# classes are printed so a wrong pick is visible rather than silent.
print("환경 discoverable 집합 (%s)" % DOMAIN)
for k, v in sorted(by_class.items()):
    print("  %-22s %d종  %s" % (k, len(v), ", ".join(sorted(v))))
user_set = set()
for k, v in by_class.items():
    if "User" in k:
        user_set |= v
if not user_set:
    raise SystemExit("no user-side discoverable class found")
print("  ⇒ user-side 판정 집합: %d종\n" % len(user_set))

tally = collections.Counter()
out_sims, out_names = collections.defaultdict(set), collections.Counter()
reward_of, gold_out = {}, []
files = sorted(glob.glob(os.path.join(SIM, ARMS[ARM] + "*.results.json.gz")))
if not files:
    raise SystemExit("no runs matched arm %s" % ARM)
for p in files:
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("simulations") if isinstance(d, dict) else d):
        sid = (s.get("task_id"), s.get("trial"))
        tally["sim"] += 1
        rw = (s.get("reward_info") or {}).get("reward")

        # ① gold이 요구한 give — 이것이 하나라도 집합 밖이면 게이트는 gold를 막는다
        for c in (s.get("reward_info") or {}).get("action_checks") or []:
            act = c.get("action") or {}
            if act.get("name") != GIVE:
                continue
            nm = inner_name(act.get("arguments"))
            tally["gold_give"] += 1
            if nm and nm not in user_set:
                tally["gold_give_outside"] += 1
                gold_out.append((sid, nm))

        # ② 실제 give
        for m in s.get("messages") or []:
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") != GIVE:
                    continue
                nm = inner_name(tc.get("arguments"))
                tally["give"] += 1
                if nm and nm not in user_set:
                    tally["give_outside"] += 1
                    out_names[nm] += 1
                    out_sims[sid].add(nm)
                    reward_of[sid] = rw
                else:
                    tally["give_inside"] += 1

# sim 단위로 센다 — 호출 단위로 세면 한 sim이 같은 이름을 반복한 것이 모집단을 부풀린다.
outside_pass = sum(1 for sid in out_sims if reward_of.get(sid) == 1.0)

print("arm %s · sim %d" % (ARM, tally["sim"]))
print("  give 총 %d = 집합 안 %d / **집합 밖 %d**"
      % (tally["give"], tally["give_inside"], tally["give_outside"]))
print("  집합 밖 give가 있는 sim: %d (그 중 통과 %d)"
      % (len(out_sims), outside_pass))
print("\n  집합 밖 이름 상위:")
for nm, n in out_names.most_common(15):
    print("    %-52s %d회" % (nm, n))

print("\n★게이트 — gold이 요구한 give %d건 중 집합 밖 = **%d건**"
      % (tally["gold_give"], tally["gold_give_outside"]))
for sid, nm in gold_out[:20]:
    print("    %s  %s" % (sid, nm))
print("  판정: %s" % ("등재 가능 (오차단 0)" if tally["gold_give_outside"] == 0
                      else "**등재 금지** — 이 이름들을 막게 된다"))
