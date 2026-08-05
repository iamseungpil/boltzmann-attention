# -*- coding: utf-8 -*-
"""How often does a payload contradict the record it copies, and would refusing it ever refuse gold?

`task_018` lost on one typed digit: `rewards_earned: 1113` for a transaction the record
printed as `487 points`. The engine computed on what it was given, produced a discrepancy
that did not exist, and a seventh dispute went out. The check is closed — the record is
in the conversation — but a deny is only safe if gold's own calls never trip it.

  fire        payload rows that disagree with the record they name
  gold_trips  the same check run against gold's own action arguments
  sims        how many simulations contain at least one

`gold_trips` above zero is a stop: it would mean the record and gold disagree, and then
the disagreement is not the model's copying.

Free: persisted trajectories only.

  usage: x90_transcription_census.py [arm]
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

import t2_transcribe as T                       # noqa: E402
from t2_scaffold_get import _parse_record_dump as PARSE   # noqa: E402
from x50_says_not_does import ARMS, SIM         # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "N97B"


def a2_spec(domain="banking_knowledge"):
    import gate_interpreter as GI
    return (GI.load_domain_a2(domain) or {}).get("transcription_check") or {}


def as_dict(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


_raw = a2_spec()
# `_`로 시작하는 키는 주석·문구다(선언 규약) — 도구 스펙으로 순회하지 않는다.
spec_by_tool = {k: v for k, v in _raw.items()
                if not k.startswith("_") and isinstance(v, dict)}
if not spec_by_tool:
    raise SystemExit("A2에 transcription_check 선언이 없다 — 선언 후 다시 돌려라")

tally = collections.Counter()
sims_hit, examples, gold_ex = set(), [], []
for p in sorted(glob.glob(os.path.join(SIM, ARMS[ARM] + "*.results.json.gz"))):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("simulations") if isinstance(d, dict) else d):
        tally["sim"] += 1
        sid = (s.get("task_id"), s.get("trial"))
        recs = {}
        for m in s.get("messages") or []:
            if m.get("role") == "tool":
                try:
                    rows = PARSE(str(m.get("content") or ""))
                except Exception:
                    rows = []
                for r in rows:
                    for sp in spec_by_tool.values():
                        rid = r.get(sp.get("id_key"))
                        if rid:
                            recs[str(rid)] = r
            for tc in (m.get("tool_calls") or []):
                sp = spec_by_tool.get(tc.get("name"))
                if not sp:
                    continue
                bad = T.mismatches(sp, as_dict(tc.get("arguments")), recs)
                if bad:
                    tally["fire"] += 1
                    sims_hit.add(sid)
                    if len(examples) < 10:
                        examples.append((sid, tc.get("name"), bad[:2]))
        # gold 자신의 인자를 같은 검사에 건다 — 여기서 걸리면 deny는 gold을 막는다
        for c in (s.get("reward_info") or {}).get("action_checks") or []:
            act = c.get("action") or {}
            sp = spec_by_tool.get(act.get("name"))
            if not sp:
                continue
            bad = T.mismatches(sp, as_dict(act.get("arguments")), recs)
            if bad:
                tally["gold_trips"] += 1
                if len(gold_ex) < 6:
                    gold_ex.append((sid, act.get("name"), bad[:2]))

print("arm %s · sim %d · 선언된 도구 %s" % (ARM, tally["sim"], list(spec_by_tool)))
print("  전사 불일치가 있는 호출 **%d건** / **%d sim**" % (tally["fire"], len(sims_hit)))
for sid, tool, bad in examples:
    print("    %-16s %-28s %s" % ("%s/t%s" % sid, tool, bad))
print()
print("  ★게이트 — gold 자신의 인자가 같은 검사에 걸린 횟수 = **%d**" % tally["gold_trips"])
for sid, tool, bad in gold_ex:
    print("    %-16s %-28s %s" % ("%s/t%s" % sid, tool, bad))
print("  판정: %s" % ("등재 가능 (오차단 0)" if not tally["gold_trips"]
                      else "**등재 금지** — 원장과 gold이 어긋난다"))
