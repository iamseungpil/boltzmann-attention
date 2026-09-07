# -*- coding: utf-8 -*-
"""x902e — 3차: _shared_span(P1) · _mentioned(REF-VERIFY) · _tok_overlap(claim) · NLNUM 누락."""
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
PASS = lambda s: (s["reward"] or 0) >= 1.0
MIN = int(((A2.get("axis_notes") or {}).get("give_quote_min_tokens") or 4))

# ── S1 · _shared_span (T2_GIVE_QUOTE=ON) : give 직전 발화에 손님 축자 4-gram 이 있나
print("[S1] give_quote_min_tokens =", MIN)
fires = []
for s in sims:
    ut = ""
    for i, m in enumerate(s["M"]):
        if m.role == "user" and isinstance(m.content, str):
            ut += " " + m.content
        if m.role != "assistant":
            continue
        giv = [t for t in m.tool_calls if str(t.name) == "give_discoverable_user_tool"]
        if not giv:
            continue
        ok = G._shared_span(m.content or "", ut, MIN)
        fires.append((PASS(s), s["task"], ok, (m.content or "").strip()[:130]))
tot = len(fires); bad = [f for f in fires if not f[2]]
print("   give 직전 판정 = %d 회 · 술어 불성립(=regen 강제) = %d · 성립 = %d"
      % (tot, len(bad), tot - len(bad)))
print("   불성립 중 통과 sim =", sum(1 for f in bad if f[0]))
for f in bad[:4]:
    print("   [불성립] pass=%s %s :: %r" % (f[0], f[1], f[3]))
ok_pass = [f for f in fires if f[2]]
for f in ok_pass[:3]:
    print("   [성립]  pass=%s %s :: %r" % (f[0], f[1], f[3]))

# ── S2 · REF-VERIFY `_mentioned` 토큰 술어 (T2_REF_VERIFY=ON)
print("\n[S2] REF-VERIFY `_mentioned` (min_tok 기본 5)")
specs = (A2.get("ref_verify_specs") or A2.get("ref_verify") or [])
print("   A2 ref_verify specs =", json.dumps(specs, ensure_ascii=False)[:400])


def mentioned(val, utext, min_tok=5):
    if not val:
        return False
    if val.lower() in utext:
        return True
    for tok in re.findall(r"[A-Za-z0-9]+", val):
        if len(tok) >= min_tok and tok.lower() in utext:
            return True
    return False


# 실물: 도구 출력의 merchant_name 값 전체 ↔ 손님 발화
labels_all = {}
rows = []
for s in sims:
    ut = " ".join((m.content or "") for m in s["M"]
                  if m.role == "user" and isinstance(m.content, str)).lower()
    mer = set()
    for m in s["M"]:
        if m.role != "tool" or not isinstance(m.content, str):
            continue
        for mm in re.finditer(r"merchant_name:\s*([^\n]+?)(?:\s{2,}|\n|$)", m.content):
            mer.add(mm.group(1).strip())
    for v in sorted(mer):
        rows.append((s["task"], v, v.lower() in ut, mentioned(v, ut)))
exact = sum(1 for r in rows if r[2]); tokm = sum(1 for r in rows if r[3])
print("   상점명 표본 %d · 축자 일치 %d · 토큰(≥5) 완화로 '언급'된 것 %d (완화가 늘린 것 %d)"
      % (len(rows), exact, tokm, tokm - exact))
for r in rows:
    if r[3] and not r[2]:
        print("   [완화로 통과] task=%s merchant=%r" % (r[0], r[1]))
        if rows.index(r) > 60:
            break

# ── S3 · _tok_overlap : claim `what` → 도구 지목 (FIX-8)
print("\n[S3] _tok_overlap(claim.what → agent 도구) — 자유문장에서 '어느 도구를 말했나' 지목")
ENV = json.load(open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))
reg = sorted(n for n, v in ENV["banking_knowledge"]["tools"].items() if v.get("side") == "tools")
print("   registry n =", len(reg))
samples = [
    "checking the account status for the customer",
    "opening Green Fee-Free Account",
    "I will get the card details",
    "file the dispute for the Starbucks charge",
    "transfer you to a human agent",
    "look up your referral link",
]
for q in samples:
    m = G._tok_overlap(q, reg, stem=True)
    hits = G._tok_hits(q, m[0]) if len(m) == 1 else None
    print("   %-46r -> %s  tok_hits=%s" % (q, m[:4], hits))

# ── S4 · NLNUM 누락: 통화기호 없는 수치는 원리상 안 본다
print("\n[S4] _MONEY_RE 누락 — 통화기호 없는 수치 주장")
pat = re.compile(r"\b\d{1,3}(?:\.\d{1,2})?\s?%")
n_pct = n_pct_unver = 0
ex = []
for s in sims:
    ctx = ""
    for m in s["M"]:
        if m.role == "assistant" and isinstance(m.content, str):
            for mm in pat.finditer(m.content):
                n_pct += 1
                v = mm.group(0).rstrip("% ").strip()
                if v and v not in ctx:
                    n_pct_unver += 1
                    if len(ex) < 5:
                        ex.append((s["task"], PASS(s), mm.group(0),
                                   " ".join(m.content.split())[:120]))
        if m.role in ("user", "tool") and isinstance(m.content, str):
            ctx += " " + m.content.replace(",", "")
print("   assistant 발화의 백분율 주장 = %d · 그중 문맥 부재 = %d (MONEY_RE 는 0 건 본다)"
      % (n_pct, n_pct_unver))
for e in ex:
    print("   [누락] task=%s pass=%s %s :: %r" % e)
