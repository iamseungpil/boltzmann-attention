# -*- coding: utf-8 -*-
"""[[08]] 결정적 검증: action_checks-완결 ⟺ db_match proxy tightness.
DB-basis sim만. X=전 gold args-row 충족 · Y=db_match. 2x2 교차표."""
import json, glob, sys, io
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def Nd(x):
    try:
        v = json.loads(x) if isinstance(x, str) else x
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}

def all_args_met(acs):
    """전 gold args-row(arguments 키 有 + agent_tool_name 有) 충족 여부. args-row 없으면 None."""
    n = 0
    for ac in acs:
        a = ac.get("action") or {}
        outer = Nd(a.get("arguments"))
        atn = outer.get("agent_tool_name", "")
        if not atn or "arguments" not in outer:
            continue
        n += 1
        met = ac.get("action_reward")
        if met is None:
            met = 1.0 if ac.get("action_match") else 0.0
        if float(met) < 1.0:
            return False, n
    return (True, n) if n else (None, 0)

def all_checks_met(acs):
    """전 action_check(name+args 모든 행) 충족 여부(더 엄격)."""
    n = 0
    for ac in acs:
        n += 1
        met = ac.get("action_reward")
        if met is None:
            met = 1.0 if ac.get("action_match") else 0.0
        if float(met) < 1.0:
            return False
    return True if n else None

tab = Counter()      # (X_args_met, Y_dbmatch)
tab_strict = Counter()
noargs = Counter()
for f in glob.glob("C:/tmp/traj/*_banking.json"):
    d = json.load(open(f, encoding="utf-8"))
    for s in d.get("simulations", []):
        ri = s.get("reward_info") or {}
        if tuple(ri.get("reward_basis") or []) != ("DB",):
            continue                                    # DB-basis만
        db = (ri.get("db_check") or {}).get("db_match")
        acs = ri.get("action_checks") or []
        X, nargs = all_args_met(acs)
        Xs = all_checks_met(acs)
        if X is None:
            noargs[str(db)] += 1
            continue
        tab[(bool(X), bool(db))] += 1
        tab_strict[(bool(Xs), bool(db))] += 1

def show(t, label):
    tot = sum(t.values())
    print("\n=== %s (n=%d DB-basis sim·args-row 有) ===" % (label, tot))
    print("            db_match=True   db_match=False")
    for X in (True, False):
        print("  X=%-5s     %6d          %6d" % (X, t[(X, True)], t[(X, False)]))
    # 일치도
    agree = t[(True, True)] + t[(False, False)]
    print("  일치(X==Y): %d/%d = %.1f%%" % (agree, tot, 100 * agree / max(tot, 1)))
    print("  X=1,Y=0 (checks-완결인데 DB불일치·over/order): %d" % t[(True, False)])
    print("  X=0,Y=1 (checks-미완인데 DB일치·checker 엄격): %d" % t[(False, True)])

show(tab, "X=전 args-row 충족 vs Y=db_match")
show(tab_strict, "X=전 action_check(name+args) 충족 vs Y=db_match")
print("\nargs-row 없는 DB-basis sim(assertion/pure-DB):", dict(noargs))
