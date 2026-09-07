# -*- coding: utf-8 -*-
"""x902d — 2차: M1 오발 실물 확인 · M3 함수 실발화 · M5 blacklist 실재성 · M2 표본 검사."""
import sys, os, json, gzip, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t2_gate_patch as G


class TC(object):
    def __init__(s, d):
        s.name = d.get("name"); s.arguments = d.get("arguments")
        s.requestor = d.get("requestor") or "assistant"


class M(object):
    def __init__(s, d):
        s.role = d.get("role"); s.content = d.get("content")
        s.error = bool(d.get("error")); s.id = d.get("id")
        s.tool_call_id = d.get("tool_call_id")
        s.requestor = d.get("requestor") or "assistant"
        s.tool_calls = [TC(t) for t in (d.get("tool_calls") or [])]


class Orch(object):
    def __init__(s, m): s._m = m
    def get_messages(s): return s._m


sims = json.loads(gzip.open(sys.argv[1], "rt", encoding="utf-8").read())
for s in sims:
    s["M"] = [M(d) for d in s["msgs"]]
A2 = G._domain_a2("banking_knowledge")
PASS = lambda s: (s["reward"] or 0) >= 1.0

# ── M1 오발 실물: 통과 sim(task_037)에서 무엇을 반려했나 ────────────────────
print("=" * 78)
print("[M1-오발 실물] T2_ARG_LABEL 이 통과 sim 에 낸 반려의 근거 문면")
for s in sims:
    if s["task"] != "task_037" or not PASS(s):
        continue
    labs = G._record_labels(Orch(s["M"]))
    print("  sim tag=%s reward=%s" % (s["tag"], s["reward"]))
    print("  labels['account_id'] =", sorted(labs.get("account_id", []))[:8])
    print("  labels['credit_card_account_id'] =", sorted(labs.get("credit_card_account_id", []))[:8])
    print("  arg_source_reads['account_id'] =", (A2.get("arg_source_reads") or {}).get("account_id"))
    print("  arg_source_reads['credit_card_account_id'] =",
          (A2.get("arg_source_reads") or {}).get("credit_card_account_id"))
    for m in s["M"]:
        c = m.content if isinstance(m.content, str) else ""
        if m.role == "tool" and "cc_890389b165_silver" in c and "Record ID:" in c:
            k = c.find("cc_890389b165_silver")
            print("  --- env 덤프 축자 ---")
            print("  ", " ".join(c[max(0, k - 260):k + 160].split()))
            break
    break

# ── M3 실발화: _have_value_reask_fb / _value_acquire_fb ─────────────────────
print("=" * 78)
hv = (A2.get("have_value_reask") or [])
va = (A2.get("value_acquisition") or [])
fired_hv, fired_va = [], []
for s in sims:
    for i, m in enumerate(s["M"]):
        if m.role != "assistant":
            continue
        prior = s["M"][:i]
        r1 = G._have_value_reask_fb(m, prior, hv)
        if r1:
            fired_hv.append((PASS(s), s["task"], (m.content or "")[:150]))
        try:
            r2 = G._value_acquire_fb(m, prior, va, A2, None)
        except Exception:
            r2 = None
        if r2:
            fired_va.append((PASS(s), s["task"], (m.content or "")[:150]))
print("[M3] _have_value_reask_fb 실발화 = %d (통과sim %d) 태스크 %s"
      % (len(fired_hv), sum(1 for x in fired_hv if x[0]), sorted({x[1] for x in fired_hv})))
for x in fired_hv[:6]:
    print("   pass=%s %s :: %r" % (x[0], x[1], x[2]))
print("[M3] _value_acquire_fb 실발화 = %d (통과sim %d) 태스크 %s"
      % (len(fired_va), sum(1 for x in fired_va if x[0]), sorted({x[1] for x in fired_va})))
for x in fired_va[:6]:
    print("   pass=%s %s :: %r" % (x[0], x[1], x[2]))

# ── M5: blacklist 로 올라간 값이 실물 레코드에 실재하나 ─────────────────────
print("=" * 78)
ENV = json.load(open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))


class T(object):
    def __init__(s, n, d): s.name = n; s.description = d; s.openai_schema = {"description": d}


bl = G._static_blacklist([T(n, v.get("desc") or "") for n, v in
                          ENV["banking_knowledge"]["tools"].items()], placeholders=set())
alltext = "\n".join((m.content or "") for s in sims for m in s["M"] if isinstance(m.content, str))
print("[M5] blacklist 항목의 **회수분 실재** 여부 (실재 = 진짜 값을 금칙어로 올렸다):")
for v in sorted(bl):
    n = alltext.count(v)
    print("   %-26s 회수분 등장 %d 회 %s" % (v, n, "  ← 실물에 존재" if n else ""))

# ── M2: NLNUM 표본 정밀 — 통과 sim 에서 무엇을 '미검증'이라 불렀나 ──────────
print("=" * 78)
print("[M2-오발 실물] 통과 sim 에서 미검증으로 잡힌 금액의 문맥 정황")
cnt = 0
for s in sims:
    if not PASS(s):
        continue
    ctx = ""
    for m in s["M"]:
        if m.role == "assistant" and isinstance(m.content, str):
            un = G._unverified_amounts(m.content, ctx)
            if un and cnt < 4:
                cnt += 1
                a = un[0]
                num = a.lstrip("$€£¥ ")
                print("   task=%s 금액 %s · 숫자부 '%s' 가 ctx 에 있나=%s · 소수점 없이는=%s"
                      % (s["task"], a, num, num in ctx, num.split(".")[0] in ctx))
                print("      발화: %r" % " ".join(m.content.split())[:180])
        if m.role in ("user", "tool") and isinstance(m.content, str):
            ctx += " " + m.content.replace(",", "")
