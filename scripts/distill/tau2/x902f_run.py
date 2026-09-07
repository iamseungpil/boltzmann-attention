# -*- coding: utf-8 -*-
"""x902f — 4차: S1 공백턴 · REF-VERIFY 교차상점 침묵 · _split_claims_by_owner · DECL_ONEOF FP."""
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
        s.tool_calls = [TC(t) for t in (d.get("tool_calls") or [])]


sims = json.loads(gzip.open(sys.argv[1], "rt", encoding="utf-8").read())
for s in sims:
    s["M"] = [M(d) for d in s["msgs"]]
A2 = G._domain_a2("banking_knowledge")
ENV = json.load(open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))
PASS = lambda s: (s["reward"] or 0) >= 1.0

# ── S1b · give 턴 중 본문이 비어 있는 것 (원리상 항상 불성립) ────────────────
empty = tot = fail = 0
for s in sims:
    ut = ""
    for m in s["M"]:
        if m.role == "user" and isinstance(m.content, str):
            ut += " " + m.content
        if m.role != "assistant":
            continue
        if not any(str(t.name) == "give_discoverable_user_tool" for t in m.tool_calls):
            continue
        tot += 1
        c = (m.content or "").strip()
        if not c:
            empty += 1
        if not G._shared_span(c, ut, 4):
            fail += 1
print("[S1b] give 턴 %d · 술어 불성립 %d · 그중 **본문이 빈 턴** %d "
      "(도구만 낸 턴은 원리상 인용이 불가능 → 항상 regen)" % (tot, fail, empty))

# ── S2b · REF-VERIFY: 서로 다른 상점이 같은 토큰으로 '언급'되어 침묵하는가 ───
def toks(v, n=5):
    return {t.lower() for t in re.findall(r"[A-Za-z0-9]+", v) if len(t) >= n}


sil = []
for s in sims:
    ut = " ".join((m.content or "") for m in s["M"]
                  if m.role == "user" and isinstance(m.content, str)).lower()
    mer = set()
    for m in s["M"]:
        if m.role != "tool" or not isinstance(m.content, str):
            continue
        for mm in re.finditer(r"merchant_name:\s*([^\n]+?)(?:\s{2,}|\n|$)", m.content):
            mer.add(mm.group(1).strip())
    named = {v for v in mer if v.lower() in ut}          # 손님이 축자로 말한 상점
    for v in sorted(mer):
        if v in named or v.lower() in ut:
            continue
        hit = sorted(t for t in toks(v) if t in ut)
        if hit:
            share = sorted(o for o in named if hit[0] in o.lower())
            sil.append((s["task"], v, hit[0], share[:2]))
print("\n[S2b] 손님이 **축자로 말하지 않은** 상점인데 공유 토큰 때문에 '언급'으로 통과 = %d" % len(sil))
seen = set()
for t, v, tk, sh in sil:
    k = (t, v)
    if k in seen:
        continue
    seen.add(k)
    if len(seen) <= 10:
        print("   task=%s 미언급 상점 %-24r ← 토큰 %r (손님이 말한 것: %s)" % (t, v, tk, sh))

# ── S3b · _split_claims_by_owner : 자유문장 → 도구 지목(min_tok=2 기본) ──────
reg = sorted(n for n, v in ENV["banking_knowledge"]["tools"].items() if v.get("side") == "tools")
usr = sorted(ENV["banking_knowledge"].get("discoverable_user_tools") or [])
print("\n[S3b] _split_claims_by_owner(min_tok=2) — 실물 assistant 문장으로")
cases = [
 "checking the account status for the customer",
 "I will review your checking account activity",
 "close the loop on your credit card account",
 "give you the tool to read your card",
 "look into the referral bonus for your account",
]
for q in cases:
    own, theirs, unk = G._split_claims_by_owner([{"what": q, "tool": None}], set(reg), set(usr),
                                                registry=reg)
    print("   %-46r -> own=%s unknown=%d" % (q, [c.get("tool") for c in own], len(unk)))
# 코퍼스 전수: 문장 단위로 몇 %가 '유일 지목'을 받나
sent_tot = sent_pick = 0
picks = {}
for s in sims:
    for m in s["M"]:
        if m.role != "assistant" or not isinstance(m.content, str):
            continue
        for sent in re.split(r"[.!?\n]+", m.content):
            sent = sent.strip()
            if len(sent.split()) < 4:
                continue
            sent_tot += 1
            mm = G._tok_overlap(sent, reg, stem=True)
            if len(mm) == 1 and G._tok_hits(sent, mm[0]) >= 2:
                sent_pick += 1
                picks[mm[0]] = picks.get(mm[0], 0) + 1
print("   코퍼스 assistant 문장 %d 중 **유일 도구 지목**을 받는 문장 = %d (%.1f%%)"
      % (sent_tot, sent_pick, 100.0 * sent_pick / max(sent_tot, 1)))
print("   상위 지목 도구:", sorted(picks.items(), key=lambda x: -x[1])[:6])

# ── S4b · _DECL_ONEOF_RE 오발 점검: 뽑힌 값이 env 실제 enum 과 다른가 ───────
allm = [m for s in sims for m in s["M"]]
dp = G._declared_params_by_tool(allm)
bad = []
for t, d in dp.items():
    for k, (typ, vals) in d.items():
        for v in vals:
            if not re.match(r"^[a-z0-9_]+$", v) and not re.match(r"^[A-Za-z0-9 _\-/]+$", v):
                bad.append((t, k, v))
print("\n[S4b] _DECL_ONEOF_RE 로 뽑힌 열거값 중 **값처럼 안 생긴 것** =", len(bad), bad[:8])
