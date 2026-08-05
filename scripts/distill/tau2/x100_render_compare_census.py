# -*- coding: utf-8 -*-
"""What is the number-extraction in `same()` actually doing, and could a type rule do it?

`t2_transcribe._num` pulls the leading number out of a rendered value so that `$487.99`
and `487.99` are not called a mismatch. It is a spelling rule, and the session that
removed `T2_UNLOCK_NAME`'s suffix regex left it standing with a note to replace it — but
replacing it blind is how a check starts denying rows it used to pass ([[55]]: measure
before deciding, and the sims that pass are the ones a wrong tightening costs).

So this counts, over every declared transcription check in a persisted arm, what each
comparison would decide under three rules:

  strict   the two renderings agree as text, trimmed and case-folded
  typed    both sides parse as a number *in full* (`float(text)`) and agree numerically
  current  what `same()` decides today (leading-number extraction)

`typed_only` is the population that justifies keeping any parsing at all: comparisons the
current rule accepts, strict rejects, and a whole-string float cannot rescue — i.e. rows
where the environment printed a unit. Their renderings are printed verbatim, because the
question "is this a format convention or a domain fact?" is answered by looking at them.

Free: persisted trajectories and the declarations.

  usage: x100_render_compare_census.py [arm] [top_n]
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

import t2_transcribe as T                                   # noqa: E402
from t2_scaffold_get import _parse_record_dump as PARSE      # noqa: E402
from x50_says_not_does import ARMS, SIM                      # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "N97B"
TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 15


def a2():
    import gate_interpreter as GI
    return GI.load_domain_a2("banking_knowledge") or {}


def args_of(x):
    a = x.get("arguments") if isinstance(x, dict) else None
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


def strict(a, b):
    return T._txt(a).strip().lower() == T._txt(b).strip().lower()


def typed(a, b):
    """Both sides are numbers as a whole — no substring is taken out of anything."""
    try:
        return abs(float(T._txt(a).strip()) - float(T._txt(b).strip())) < 1e-9
    except (TypeError, ValueError):
        return False


A = a2()
TRS = {k: v for k, v in (A.get("transcription_check") or {}).items()
       if not k.startswith("_") and isinstance(v, dict)}

verdict = collections.Counter()
by_field = collections.Counter()
samples = collections.Counter()
bad = collections.Counter()
halfnum = collections.Counter()
sims = 0

for p in sorted(glob.glob(os.path.join(SIM, ARMS[ARM] + "*.results.json.gz"))):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("simulations") if isinstance(d, dict) else d):
        sims += 1
        recs = {}
        for m in s.get("messages") or []:
            if m.get("role") == "tool":
                try:
                    rows = PARSE(str(m.get("content") or ""))
                except Exception:
                    rows = []
                for r in rows:
                    for sp in TRS.values():
                        rid = r.get(sp.get("id_key"))
                        if rid:
                            recs[str(rid)] = r
                continue
            for tc in (m.get("tool_calls") or []):
                sp = TRS.get(tc.get("name"))
                if not sp:
                    continue
                idk = sp.get("id_key")
                for row in T._rows(args_of(tc).get(sp.get("arg"))):
                    src = recs.get(T._txt(row.get(idk)).strip())
                    if not src:
                        continue
                    for k, v in row.items():
                        if k == idk or k not in src:
                            continue
                        cur, st, ty = T.same(v, src[k]), strict(v, src[k]), typed(v, src[k])
                        if st:
                            verdict["strict_agree"] += 1
                        elif ty:
                            verdict["typed_only"] += 1      # float(전체)로 충분 — 추출 불요
                        elif cur:
                            verdict["extract_only"] += 1    # 추출이 있어야만 통과하는 자리
                            by_field[k] += 1
                            samples[(k, T._txt(v)[:24], T._txt(src[k])[:24])] += 1
                        else:
                            verdict["mismatch"] += 1
                            bad[(k, T._txt(v)[:24], T._txt(src[k])[:24])] += 1
                        # 위험 모집단: 한쪽만 수로 읽히는 표기(단위가 붙은 렌더링)
                        if typed(v, v) != typed(src[k], src[k]):
                            halfnum[(k, T._txt(v)[:20], T._txt(src[k])[:20])] += 1

tot = sum(verdict.values())
print("arm %s · sim %d · 비교 %d건" % (ARM, sims, tot))
for k in ("strict_agree", "typed_only", "extract_only", "mismatch"):
    print("  %-14s %6d  (%.1f%%)" % (k, verdict[k], 100.0 * verdict[k] / (tot or 1)))
print()
print("★추출이 있어야만 통과하는 자리 — 필드별:")
for k, n in by_field.most_common(TOP):
    print("   %-24s %d" % (k, n))
print()
print("★그 자리의 실제 표기(payload ↔ record):")
for (k, v, sv), n in samples.most_common(TOP):
    print("   %-20s %-26s %-26s ×%d" % (k, v, sv, n))
print()
print("★불일치로 남은 자리(payload ↔ record):")
for (k, v, sv), n in bad.most_common(TOP):
    print("   %-20s %-26s %-26s ×%d" % (k, v, sv, n))
print()
print("★한쪽만 수로 읽히는 표기(단위 렌더링 위험군): %d건" % sum(halfnum.values()))
for (k, v, sv), n in halfnum.most_common(TOP):
    print("   %-20s %-22s %-22s ×%d" % (k, v, sv, n))
