"""Which surface licenses closing a predicate — and does N2b's implemented test agree?

N2b (`t2_unknown_bool.py`) passes an argument when its name "실재한다 as a field of a
retrieved record", on the reasoning that a record is an authority that closes the
predicate's extension, so absence from it means false rather than unknown. That is a
closed-world assumption, and a closed-world assumption is only sound where somebody has
declared the extension complete ([[52]]: DCA/CWA are declarable by policy, UNA is not).

The implemented test is a substring scan over *every* role=tool message. Tool messages
are not one surface. Three appear in this arm:

  record    `Found N record(s) in 'users':` followed by `field: value` lines, and the
            dict-shaped output of our own compute tools. A table has an extension; the
            row either carries the field or the field does not exist for that row.
  document  a knowledge-base hit — `Score:` and `Content:` with policy prose. Prose
            *mentions* a concept. It never enumerates who satisfies it, so it licenses
            nothing.
  engine    our own notices ([READ-FIRST], [DUPLICATE-READ], VERIFIED …), which quote
            back arguments the agent itself supplied.

If a name reaches the licensing test through the document or engine surface, N2b passes
an argument whose extension nobody closed — the same inference it exists to block, made
by us instead of by the model. This counts, per surface, both what the implemented
predicate licenses and what a record-only predicate would, so the gap is a number rather
than a worry.

The second half asks the prior question: over the whole arm, does the DB carry the
relation at all? A name that never appears in any record of any of the 194 simulations
has no extension to fail against — the customer is its only authority, and the
prescription is ASK, not read.
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
from x50_says_not_does import ARMS, SIM, norm_args  # noqa: E402
from x66_effective_tool_miss import agent_actions  # noqa: E402

# A record listing from the DB search tool: a header naming the table, then one
# `  field: value` line per column. The header is what makes it a record rather than
# prose — it names the relation the rows are drawn from.
REC_HEADER = re.compile(r"Found \d+ record\(s\) in '([^']+)'")
REC_FIELD = re.compile(r"^\s{2,}([a-z_][a-z0-9_]*):", re.M)
# Key position in a dict repr or JSON object — our compute tools return these.
DICT_KEY = re.compile(r"['\"]([a-z_][a-z0-9_]*)['\"]\s*:")
# A knowledge-base hit. Prose, not rows.
DOC = re.compile(r"\bScore:\s*\d")
# Any occurrence of the bare name, which is what a substring test sees.
def _mentions(name, text):
    return re.search(r"\b%s\b" % re.escape(name), text) is not None


def surfaces(content):
    """(record_fields, is_document) for one tool message."""
    fields = set()
    if REC_HEADER.search(content):
        fields |= set(REC_FIELD.findall(content))
    fields |= set(DICT_KEY.findall(content))
    return fields, bool(DOC.search(content))


def tool_msgs(sim):
    for m in sim.get("messages") or []:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            yield m["content"]


# The arguments §2.2 of the design doc names, plus the two the fixture uses. Kept
# explicit so the count is comparable with the table already published there.
NAMED = ["invited", "premium_subscriber", "needs_purchase_protection", "business",
         "rho_bank_subscription", "subscribed"]

BOOLISH = {"true", "false", "yes", "no", "1", "0", ""}


def boolean_args(sims):
    """Every argument name whose supplied values are all boolean-shaped.

    N2b decides this from the live tool schema, which is the right source and is not
    reachable from a transcript. This is the offline stand-in: an argument the model
    only ever fills with true/false is one whose type is closed, whatever the schema
    says. It is a proxy, and it is here so the census covers every such argument rather
    than the four that happened to get named in a read-through.
    """
    seen = collections.defaultdict(set)
    for s in sims:
        for _w, _n, args in agent_actions(s):
            for k, v in (norm_args(args) or {}).items():
                if v is not None:
                    seen[k].add(str(v).strip().lower())
    return sorted(k for k, vals in seen.items()
                  if vals and vals <= BOOLISH and vals - {""})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N97")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{SIM}/{ARMS[a.arm]}.results.json.gz"))
    if not files:
        raise SystemExit(f"no runs matched {ARMS[a.arm]}")
    sims = []
    for p in files:
        sims.extend(json.load(gzip.open(p, "rt", encoding="utf-8"))
                    .get("simulations") or [])
    print(f"{a.arm}: {len(sims)} sims from {len(files)} files")

    TARGETS = sorted(set(NAMED) | set(boolean_args(sims)))
    print(f"targets: {len(TARGETS)} = {len(NAMED)} named + "
          f"{len(set(TARGETS) - set(NAMED))} boolean-valued\n  "
          + ", ".join(TARGETS) + "\n")

    # ── 1. Arm-wide: does the DB carry the relation at all? ──────────────────
    rec_fields = collections.Counter()   # name -> messages carrying it as a record field
    rec_sims = collections.defaultdict(set)
    doc_only = collections.Counter()     # name -> messages mentioning it in prose only
    tables = collections.Counter()
    for i, s in enumerate(sims):
        for c in tool_msgs(s):
            for t in REC_HEADER.findall(c):
                tables[t] += 1
            f, isdoc = surfaces(c)
            for n in TARGETS:
                if n in f:
                    rec_fields[n] += 1
                    rec_sims[n].add(i)
                elif _mentions(n, c):
                    doc_only[n] += 1

    print("── arm-wide surface of each name (tool outputs) ──")
    print(f"{'name':<30} {'record-field':>12} {'sims':>5} {'mention-only':>13}")
    for n in TARGETS:
        print(f"{n:<30} {rec_fields[n]:>12} {len(rec_sims[n]):>5} {doc_only[n]:>13}")
    print(f"\ntables seen: {dict(tables)}\n")

    # ── 2. Per supplied argument: what licenses it, under which predicate? ───
    # `implemented` reproduces t2_unknown_bool._is_record_field: the three substring
    # patterns over every tool message, no surface distinction.
    # Three candidate predicates, so the fix can be chosen on numbers:
    #   impl      what t2_unknown_bool ships — quoted name, or `name:` anywhere
    #   anchored  quoted name, or `name:` *at the start of its line*. A key occupies the
    #             head of its line; a sentence that happens to end in a colon does not.
    #             No knowledge of any tool's output format, so it transfers.
    #   keypos    anchored, and the quoted form must also be followed by a colon. A key
    #             is a name in key position; a name merely quoted is being talked about.
    #   record    the strict surface test above, which does know the record header
    counts = collections.Counter()
    per_name = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    disagree = []
    agree = collections.Counter()
    for s in sims:
        msgs = list(tool_msgs(s))
        for _wrap, name, args in agent_actions(s):
            for k, v in (norm_args(args) or {}).items():
                if k not in TARGETS:
                    continue
                quoted = any(('"%s"' % k) in c or ("'%s'" % k) in c for c in msgs)
                impl = quoted or any(("%s:" % k) in c for c in msgs)
                line = any(re.search(r"^\s*%s\s*:" % re.escape(k), c, re.M) for c in msgs)
                anch = quoted or line
                keyq = any(('"%s":' % k) in c or ("'%s':" % k) in c for c in msgs)
                keyp = keyq or line
                rec = any(k in surfaces(c)[0] for c in msgs)
                agree[(keyp, rec)] += 1
                verdict = ("record" if rec else
                           "substring-only" if impl else "absent")
                counts[verdict] += 1
                # N2b skips a None value before it ever tests the name, so only a
                # supplied value can actually be licensed by mistake.
                if impl and not rec and v is not None:
                    counts["effective-false-licence"] += 1
                    disagree.append((s.get("task_id"), s.get("trial"), name, k, v,
                                     anch, keyp))
                per_name[k][verdict] += 1
                if verdict == "substring-only" and len(examples[k]) < 3:
                    for c in msgs:
                        if ("%s:" % k) in c and k not in surfaces(c)[0]:
                            examples[k].append((name, c[:160].replace("\n", " ")))
                            break

    print("── each supplied argument, by what licenses closing it ──")
    print(f"{'name':<30} {'record':>7} {'substring-only':>15} {'absent':>7}")
    for n in TARGETS:
        c = per_name[n]
        if not sum(c.values()):
            continue
        print(f"{n:<30} {c['record']:>7} {c['substring-only']:>15} {c['absent']:>7}")
    print(f"{'TOTAL':<30} {counts['record']:>7} {counts['substring-only']:>15} "
          f"{counts['absent']:>7}")

    if examples:
        print("\n── what the substring test matched (record-only test would not) ──")
        for k, ex in examples.items():
            for tool, txt in ex:
                print(f"  {k} @ {tool}: {txt}")

    if disagree:
        print("\n── licences the shipped predicate grants on a supplied value ──")
        print(f"{'task':<12} {'tr':>2} {'tool':<30} {'arg':<12} {'value':<8} "
              f"{'anch':>5} {'keypos':>6}")
        for t, tr, tool, k, v, anch, keyp in disagree:
            print(f"{t:<12} {tr:>2} {tool:<30} {k:<12} {str(v)!r:<8} "
                  f"{str(anch):>5} {str(keyp):>6}")
        print(f"\neffective false licences: {counts['effective-false-licence']}"
              f"   line-anchored would still grant: "
              f"{sum(1 for d in disagree if d[5])}"
              f"   key-position would still grant: "
              f"{sum(1 for d in disagree if d[6])}")

    print("\n── key-position predicate vs the record surface (all supplied args) ──")
    print(f"  both license          {agree[(True, True)]}")
    print(f"  key-position only     {agree[(True, False)]}   (false licence remaining)")
    print(f"  record only           {agree[(False, True)]}   (would be lost)")
    print(f"  neither               {agree[(False, False)]}")


if __name__ == "__main__":
    main()
