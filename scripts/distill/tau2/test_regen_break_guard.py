# -*- coding: utf-8 -*-
"""Every feedback the regeneration loop can raise must be able to hold the loop open.

The loop ends on a guard that lists the feedback variables which are still None. A
variable missing from that list is not a small omission: the loop exits before the
feedback is assembled into a message, so the lever prints its deny line and the model
never receives it, and the per-sim counter below the guard never increments so the cap
never binds. That is exactly what `proc_fb` did — eleven denies in one simulation
against a cap of six, none of them delivered (smoke g, `task_048`).

Comment discipline cannot prevent this; the next lever will be added the same way. So
the check is structural: parse the file, collect every `*_fb` the loop assigns, and
require the guard to name all of them.
"""

import ast
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = os.path.join(HERE, "t2_gate_patch.py")
fail = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL", (" — " + detail) if detail else ""))
    if not ok:
        fail.append(name)


def loops(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) \
                and node.test.value is True:
            yield node


def guard_of(loop):
    """The `if <all fb are None>: break` statement — the one whose test names many fb."""
    best = None
    for n in ast.walk(loop):
        if isinstance(n, ast.If) and len(n.body) == 1 and isinstance(n.body[0], ast.Break):
            names = {x.id for x in ast.walk(n.test) if isinstance(x, ast.Name)}
            fbs = {x for x in names if x.endswith("_fb")}
            if len(fbs) >= 2 and (best is None or len(fbs) > len(best[1])):
                best = (n, names)
    return best


tree = ast.parse(io.open(SRC, encoding="utf-8").read())
print("① 재생성 루프의 break 가드가 모든 피드백을 본다")
found = 0
for loop in loops(tree):
    assigned = {t.id for n in ast.walk(loop) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name) and t.id.endswith("_fb")}
    g = guard_of(loop)
    if not (assigned and g):
        continue
    found += 1
    missing = sorted(assigned - g[1])
    check("while@%d: 세우는 fb %d종이 전부 가드에 있다" % (loop.lineno, len(assigned)),
          not missing, "누락=%s" % missing if missing else "")

check("가드를 가진 재생성 루프를 찾았다", found >= 1, "found=%d" % found)

print("\n② 카운터·조립이 가드 **뒤**에 있다 (누락 시 무엇을 잃는지의 근거)")
for loop in loops(tree):
    g = guard_of(loop)
    if not g:
        continue
    after = [n.lineno for n in ast.walk(loop)
             if isinstance(n, ast.Attribute) and str(getattr(n, "attr", "")).endswith("_deny")
             and n.lineno > g[0].lineno]
    check("가드 뒤에 sim-cap 카운터가 있다", bool(after),
          "첫 카운터 @%s" % (min(after) if after else "없음"))
    break

print("\nRESULT: %s" % ("ALL PASS" if not fail else "FAIL %s" % fail))
sys.exit(1 if fail else 0)
