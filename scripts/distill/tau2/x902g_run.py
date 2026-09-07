# -*- coding: utf-8 -*-
"""x902g — 5차: WRITE-EVIDENCE forbid_when_tokens 기간무시 · _hint_hit ↔ env 선언."""
import sys, os, json, gzip, re, datetime
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
TOK = "record(s) in 'credit_card_closure_reasons'"

# ── W1 · forbid_when_tokens 가 걸리는 출력의 **기록 날짜** ────────────────────
print("[W1] forbid_when_tokens 토큰 %r 가 든 도구 출력" % TOK)
n = 0
old = 0
ex = []
for s in sims:
    for m in s["M"]:
        if m.role != "tool" or not isinstance(m.content, str):
            continue
        if TOK not in m.content:
            continue
        n += 1
        ds = re.findall(r"\b(\d{2})/(\d{2})/(\d{4})\b", m.content)
        yrs = sorted({int(y) for _a, _b, y in ds})
        if yrs and max(yrs) < 2025:
            old += 1
        if len(ex) < 4:
            ex.append((s["task"], yrs, " ".join(m.content.split())[:230]))
print("   출력 %d 건 · 그중 최신 기록 연도가 2025 미만(=정책의 'past year' 밖) %d 건" % (n, old))
for t, y, c in ex:
    print("   task=%s years=%s :: %r" % (t, y, c))

# ── W2 · _hint_hit ↔ env 가 선언한 '식별자' ─────────────────────────────────
allm = [m for s in sims for m in s["M"]]
declared = {}
for m in allm:
    c = str(getattr(m, "content", "") or "")
    if "Parameters:" not in c:
        continue
    for name, typ, req, desc in G._DECL_PARAM_RE.findall(c):
        declared.setdefault(name, set()).add(desc.strip()[:120])
hints = A2["_hints"]
print("\n[W2] _hint_hit ↔ env 선언  (hints=%s)" % (sorted(hints),))
fp, fn = [], []
for name, descs in sorted(declared.items()):
    d = " ".join(descs).lower()
    is_id = ("identifier" in d) or bool(re.search(r"\bid\b", d)) or name.endswith("_id")
    hit = G._hint_hit(name, hints)
    if hit and not is_id:
        fp.append((name, sorted(descs)[0][:80]))
    if is_id and not hit:
        fn.append((name, sorted(descs)[0][:80]))
print("   env 선언 인자 %d · 오발(식별자 아닌데 식별자로 판정) %d · 누락 %d"
      % (len(declared), len(fp), len(fn)))
for a, b in fp:
    print("   [오발] %-26s %r" % (a, b))
for a, b in fn[:8]:
    print("   [누락] %-26s %r" % (a, b))

# ── W3 · free_text_drop 실발화 ──────────────────────────────────────────────
ftd = (A2.get("free_text_defaults") or {})
print("\n[W3] free_text_defaults =", ftd)
fires = []
for s in sims:
    corp = ""
    for m in s["M"]:
        if m.role == "assistant":
            for tc in m.tool_calls:
                ar = G._args_dict(tc) or {}
                inner = G._exact_tool_name(tc) or ""
                sub = ar.get("arguments")
                if isinstance(sub, str):
                    try:
                        sub = json.loads(sub)
                    except Exception:
                        sub = None
                tg = ftd.get(str(inner)) or ftd.get(str(tc.name))
                if not tg:
                    continue
                bag = sub if isinstance(sub, dict) else ar
                for k in tg:
                    v = bag.get(k)
                    if isinstance(v, str) and v.strip() and v.strip().lower() not in corp.lower():
                        fires.append((PASS(s), s["task"], inner, k, v[:90]))
        if isinstance(m.content, str):
            corp += " " + m.content
print("   실발화(=인자 제거) = %d (통과sim %d)" % (len(fires), sum(1 for f in fires if f[0])))
for f in fires[:5]:
    print("   pass=%s %s %s.%s = %r" % f)
