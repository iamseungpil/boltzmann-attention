# -*- coding: utf-8 -*-
"""Which levers read an authority, and which guess — and which of the two misfires?

The design question the run forced: `T2_UNLOCK_NAME` refused `verify_identity` because the
name carries no numeric suffix, and told the model to search the knowledge base for a
suffixed name that does not exist. `verify_identity` is an ordinary tool. The environment
publishes exactly which tools are discoverable, so the fact was available and the lever
used a spelling rule instead. `task_019` lost identity verification at turn two and never
recovered.

That suggests a split that can be measured rather than argued:

  authority   the predicate reads the environment's registry, the policy's own sentence,
              or a record the conversation retrieved
  proxy       the predicate reads a pattern, a keyword, or an inference about the situation

and the question is whether misfires concentrate in one of them. A misfire here is a
lever speaking in a simulation about a tool gold never wanted — not proof of harm, but the
population where harm would live.

Source classification is declared below and printed, so it can be argued with directly.

Free: sidecar (what was sent, per sim) + gold.

  usage: x99_lever_source_audit.py <tag>
"""

import collections
import glob
import hashlib
import io
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

TAG = sys.argv[1] if len(sys.argv) > 1 else "20260805m"
LOGD = "/home/woori/scratch/logs"
SIMD = "/home/woori/scratch/tau2-bench/data/simulations"

# ★출처 분류(선언·감사 가능): 술어가 무엇을 읽는가.
SOURCE = {
    "POLICY GATE":      ("authority", "A2 gate_spec = 정책 축자"),
    "PROCEDURE":        ("authority", "A2 procedures = 정책 축자 + 호출 이력"),
    "WRITE-EVIDENCE":   ("authority", "도구 출력 실재"),
    "PROVENANCE":       ("authority", "문맥 실재(도구 출력/유저 발화)"),
    "CLAIM-PROVENANCE": ("authority", "주장 vs 실행 원장"),
    "WRITE-GROUNDING":  ("authority", "인자값 vs 문맥"),
    "SIGNATURE":        ("authority", "A2 tool_signatures = 선언 서명"),
    "LEDGER":           ("authority", "엔진 출력 vs 제출 이력"),
    "GIVE-EXEC":        ("authority", "give 후 실행 이력"),
    "FOLLOW-UP":        ("authority", "A2 체인 선언 + 호출 이력"),
    "PROTOCOL":         ("authority", "코퍼스: 그 도구를 정의한 문서"),
    # ── 대리(proxy) ──
    "missing its numeric suffix": ("proxy", "**철자 패턴**(`_\\d+$`) — env 레지스트리를 안 읽는다"),
    "DISCOVERY-REQUIRED": ("proxy", "상황 추정: '전용 도구가 필요할 것이다'"),
    "VALUE-ACQUIRE":    ("proxy", "상황 추정: '이 값이 필요할 것이다'"),
    "SEARCH-EXHAUST":   ("proxy", "중복 질의 계수(구조적이나 대상은 추정)"),
}


def which(text):
    for k, v in SOURCE.items():
        if k in (text or ""):
            return k, v
    return None, (None, None)


def sims():
    out = []
    for f in sorted(glob.glob(os.path.join(SIMD, "bank_smk_gpu*_%s" % TAG, "results.json"))):
        out.extend(json.load(open(f, encoding="utf-8")).get("simulations") or [])
    return out


def fp(sim):
    for m in sim.get("messages") or []:
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            return hashlib.sha1(m["content"].strip().encode("utf-8")).hexdigest()[:12]
    return None


def inner(a):
    a = a if isinstance(a, dict) else {}
    return (a.get("agent_tool_name") or a.get("discoverable_tool_name")
            or a.get("user_tool_name"))


gold_by_fp, task_by_fp = {}, {}
for s in sims():
    k = fp(s)
    task_by_fp[k] = s.get("task_id")
    g = set()
    for c in (s.get("reward_info") or {}).get("action_checks") or []:
        a = c.get("action") or {}
        n = inner(a.get("arguments")) or a.get("name")
        if n:
            g.add(n)
    gold_by_fp[k] = g

TOOLPAT = re.compile(r"`([a-z][a-z0-9_]{6,})`|'([a-z][a-z0-9_]{6,})'")
stat = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)
p = os.path.join(LOGD, "fb_%s.jsonl" % TAG)
if not os.path.exists(p):
    raise SystemExit("사이드카 없음: %s" % p)
for line in io.open(p, encoding="utf-8", errors="ignore"):
    try:
        r = json.loads(line)
    except Exception:
        continue
    txt = r.get("text") or ""
    key, (src, why) = which(txt)
    if not key:
        continue
    stat[key]["fire"] += 1
    stat[key]["_src"] = 0
    g = gold_by_fp.get(r.get("sim")) or set()
    named = {a or b for a, b in TOOLPAT.findall(txt[:400])} - {""}
    if named and g and not (named & g):
        stat[key]["offgold"] += 1
        if len(examples[key]) < 3:
            examples[key].append((task_by_fp.get(r.get("sim")), sorted(named)[:2]))

print("tag %s — 레버별 출처와 오발화" % TAG)
print("%-30s %-10s %-6s %-6s %s" % ("마커", "출처", "발화", "gold밖", "술어가 읽는 것"))
rows = sorted(stat.items(), key=lambda kv: -kv[1]["fire"])
for k, c in rows:
    src, why = SOURCE[k]
    n = c["fire"] or 1
    print("%-30s %-10s %-6d %-6d %s" % (k[:30], src, c["fire"], c["offgold"], why))
print()
for grp in ("authority", "proxy"):
    f = sum(c["fire"] for k, c in rows if SOURCE[k][0] == grp)
    o = sum(c["offgold"] for k, c in rows if SOURCE[k][0] == grp)
    print("  %-10s 발화 %-5d · gold-밖 %-5d = **%.1f%%**" % (grp, f, o, 100.0 * o / (f or 1)))
print()
print("  예시(gold-밖):")
for k, ex in examples.items():
    print("    %-28s %s" % (k[:28], ex))
