"""When a call is made but graded wrong, is the argument wrong or just written differently?

task_040/t0 files all eight disputes, every one is accepted, and every one misses gold.
The differences there were `"2025-11-14"` against `11/14/2025` and `"Yes"` against `true`
— on a tool whose own docstring says `format MM/DD/YYYY` and `(boolean)`. That is a
different kind of failure from picking the wrong card, and it has a different fix, so it
should be counted separately before anything is built.

For every gold action the run missed, this pairs it with the call the agent actually made
(same tool, same primary identifier) and classifies each disagreeing key:

  bool-as-string   gold true/false, agent "Yes"/"No"/"true" — the docstring declares boolean
  date-format      the same calendar date written the other way round
  number-format    "75.00" against 75.0, or "$75" against 75.0
  case-space       same string modulo case and surrounding whitespace
  missing-key      gold has the key, the call omitted it (or passed null)
  extra-key        the call passed a key gold does not have
  value            genuinely different content — a different id, amount, or category

The number that matters is at the end: simulations where *every* disagreement is a
formatting one. Those are the ones a deterministic conformance check could close without
deciding anything, because in each of them the agent had already chosen correctly.

Caveat kept in view: most banking tasks are graded on final database state, not on these
action checks (C274). Argument equality is what writes those rows, so this is a proxy for
the ceiling, not a pass prediction.

  usage: x79_argdiff_census.py [--arm N97B] [--examples 12]
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x50_says_not_does import ARMS, SIM  # noqa: E402

ID_KEYS = ("transaction_id", "credit_card_account_id", "dispute_id", "referral_id",
           "user_id", "card_type", "discoverable_tool_name", "agent_tool_name")
BOOLISH = {"yes": True, "no": False, "true": True, "false": False, "y": True, "n": False}
DATE_PATTERNS = (
    (re.compile(r"^(\d{2})/(\d{2})/(\d{4})$"), lambda m: (m.group(3), m.group(1), m.group(2))),
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})"), lambda m: (m.group(1), m.group(2), m.group(3))),
    (re.compile(r"^(\d{2})-(\d{2})-(\d{4})$"), lambda m: (m.group(3), m.group(1), m.group(2))),
)


def as_date(v):
    s = str(v).strip()
    for pat, f in DATE_PATTERNS:
        m = pat.match(s)
        if m:
            return f(m)
    return None


def as_num(v):
    try:
        return float(re.sub(r"[,$\s]", "", str(v)))
    except Exception:
        return None


def classify(key, g, a):
    """One difference, one label. Order matters: the cheap explanations are tried first."""
    if a is None:
        return "missing-key"
    if g is None:
        return "extra-key"
    if isinstance(g, bool) and not isinstance(a, bool):
        return "bool-as-string" if str(a).strip().lower() in BOOLISH else "value"
    if isinstance(a, bool) and not isinstance(g, bool):
        return "bool-as-string" if str(g).strip().lower() in BOOLISH else "value"
    dg, da = as_date(g), as_date(a)
    if dg and da:
        return "date-format" if dg == da else "value"
    ng, na = as_num(g), as_num(a)
    if ng is not None and na is not None:
        return "number-format" if abs(ng - na) < 1e-9 else "value"
    if str(g).strip().lower() == str(a).strip().lower():
        return "case-space"
    return "value"


def inner(a):
    a = a if isinstance(a, dict) else {}
    nm = a.get("agent_tool_name") or a.get("discoverable_tool_name") or a.get("user_tool_name")
    sub = a.get("arguments")
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except Exception:
            sub = None
    if nm:
        return nm, (sub if isinstance(sub, dict) else {})
    return None, a


def key_of(tool, args):
    for k in ID_KEYS:
        if args.get(k):
            return (tool, str(args[k]))
    return (tool, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N97B")
    ap.add_argument("--examples", type=int, default=10)
    A = ap.parse_args()

    kinds = collections.Counter()
    per_key = collections.Counter()
    sims_all_format, sims_any_value, sims_unpaired = [], [], []
    examples = collections.defaultdict(list)

    for p in sorted(glob.glob(os.path.join(SIM, ARMS[A.arm] + "*.results.json.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        for s in (d.get("simulations") if isinstance(d, dict) else d):
            ri = s.get("reward_info") or {}
            missed = []
            for c in ri.get("action_checks") or []:
                if c.get("action_match"):
                    continue
                g = c.get("action") or {}
                nm, args = inner(g.get("arguments"))
                missed.append((nm or g.get("name"), args))
            if not missed:
                continue
            done = {}
            for m in s.get("messages") or []:
                for tc in m.get("tool_calls") or []:
                    nm, args = inner(tc.get("arguments"))
                    done.setdefault(key_of(nm or tc.get("name"), args), args)
            labels = []
            paired = 0
            for tool, gargs in missed:
                a = done.get(key_of(tool, gargs))
                if a is None:
                    continue                     # never called with that identifier — a different failure
                paired += 1
                for k in set(gargs) | set(a):
                    if gargs.get(k) == a.get(k):
                        continue
                    lab = classify(k, gargs.get(k), a.get(k))
                    labels.append(lab)
                    kinds[lab] += 1
                    per_key[(k, lab)] += 1
                    if len(examples[lab]) < A.examples:
                        examples[lab].append("%s/t%s %s.%s: %r → %r"
                                             % (s["task_id"], s.get("trial"), tool, k,
                                                gargs.get(k), a.get(k)))
            tag = "%s/t%s" % (s["task_id"], s.get("trial"))
            if not paired:
                sims_unpaired.append(tag)
            elif labels and all(l in ("bool-as-string", "date-format", "number-format", "case-space")
                                for l in labels):
                sims_all_format.append(tag)
            elif labels:
                sims_any_value.append(tag)

    tot = sum(kinds.values())
    print("=== 불일치 키 %d건 분류 (arm %s) ===" % (tot, A.arm))
    for k, n in kinds.most_common():
        print("  %-16s %5d  (%.0f%%)" % (k, n, 100 * n / max(1, tot)))

    print("\n=== 키별 상위 ===")
    for (k, lab), n in per_key.most_common(14):
        print("  %-34s %-16s %d" % (k, lab, n))

    print("\n=== sim 단위 ===")
    print("  호출은 했는데 **차이가 전부 표기형식**  : %d sim" % len(sims_all_format))
    print("     %s" % ", ".join(sims_all_format))
    print("  진짜 값 차이가 하나라도 있음             : %d sim" % len(sims_any_value))
    print("  gold 식별자로 호출 자체가 없음(짝 없음)  : %d sim" % len(sims_unpaired))

    print("\n=== 예시 ===")
    for lab in ("bool-as-string", "date-format", "number-format", "case-space", "value"):
        if examples[lab]:
            print("  [%s]" % lab)
            for e in examples[lab][:4]:
                print("     %s" % e)


if __name__ == "__main__":
    main()
