# -*- coding: utf-8 -*-
"""x902c — gate-core A1 감사 본체 (실물 회수분 76 sim · 8,770 msg)."""
import sys, os, json, gzip, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
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
ENV = json.load(open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))
SEL = {"agent_tool_name", "user_tool_name", "discoverable_tool_name", "tool_name"}
PASS = lambda s: (s["reward"] or 0) >= 1.0


def bucket(hits):
    """[[70]] 부호표: (전체 발화, 통과 sim 발화, 실패 sim 발화, 태스크 집합)"""
    p = sum(1 for h in hits if h[0])
    return len(hits), p, len(hits) - p, sorted({h[1] for h in hits})


# ══ M1 · _label_mismatch_deny (T2_ARG_LABEL=ON) ═══════════════════════════════
hits = []
for s in sims:
    labs = G._record_labels(Orch(s["M"]))
    for m in s["M"]:
        if getattr(m, "role", None) != "assistant":
            continue
        for tc in m.tool_calls:
            r = G._label_mismatch_deny(tc, A2, labs, selectors=SEL)
            if r:
                hits.append((PASS(s), s["task"], s["tag"], tc.name, r[1][:150]))
n, p, f, tasks = bucket(hits)
print("[M1] _label_mismatch_deny 발화 = %d (통과sim %d · 실패sim %d) 태스크 %s" % (n, p, f, tasks))
for h in hits[:6]:
    print("   pass=%s task=%s %s :: %s" % (h[0], h[1], h[3], h[4]))

# ══ M2 · _unverified_amounts / _MONEY_RE (T2_NLNUM_PROV=OFF) ══════════════════
hits = []
for s in sims:
    ctx = ""
    for m in s["M"]:
        if m.role == "assistant" and isinstance(m.content, str) and m.content.strip():
            un = G._unverified_amounts(m.content, ctx)
            if un:
                hits.append((PASS(s), s["task"], s["tag"], un[:4], m.content[:120]))
        if m.role in ("user", "tool") and isinstance(m.content, str):
            ctx += " " + m.content.replace(",", "")
n, p, f, tasks = bucket(hits)
print("\n[M2] _unverified_amounts 발화 = %d (통과sim %d · 실패sim %d) 태스크 %s" % (n, p, f, tasks[:20]))
for h in hits[:6]:
    print("   pass=%s task=%s amt=%s :: %r" % (h[0], h[1], h[3], h[4]))

# ══ M3 · reask_signals (have_value / value_acquire) ═══════════════════════════
specs = (A2.get("have_value_reask") or [])
sig = sorted({x.lower() for sp in specs for x in (sp.get("reask_signals") or [])})
print("\n[M3] reask_signals =", sig)
tot = fp = 0
ex_fp, ex_tp = [], []
ASK = ("?", "could you", "can you", "please provide", "please confirm", "what is",
       "i need", "may i", "let me know")
for s in sims:
    for m in s["M"]:
        if m.role != "assistant" or not isinstance(m.content, str):
            continue
        low = m.content.lower()
        if not any(x in low for x in sig):
            continue
        tot += 1
        asking = any(a in low for a in ASK)
        if not asking:
            fp += 1
            if len(ex_fp) < 5:
                ex_fp.append((s["task"], PASS(s), m.content.strip()[:190]))
        elif len(ex_tp) < 2:
            ex_tp.append((s["task"], m.content.strip()[:160]))
print("   신호 적중 assistant 발화 = %d · 그중 **질문 표지 0** (=재요청 아님) = %d (%.0f%%)"
      % (tot, fp, 100.0 * fp / max(tot, 1)))
for t, pa, c in ex_fp:
    print("   [오발] task=%s pass=%s :: %r" % (t, pa, c))

# ══ M4 · _is_effective_write vs env `mutates` ═════════════════════════════════
print("\n[M4] _is_effective_write ↔ env mutates")
for dom in ("banking_knowledge", "retail", "airline"):
    a2d = G._domain_a2(dom)
    fp, fn = [], []
    for name, v in sorted(ENV[dom]["tools"].items()):
        mut, pred = bool(v.get("mutates")), G._is_effective_write(name, a2d)
        if pred and not mut: fp.append(name)
        if mut and not pred: fn.append(name)
    a2p = G._a2_procedural(a2d)
    fn = [x for x in fn if G._SUFFIX_RE.sub("", x) not in a2p]   # A2 선언분은 뺀다
    print("   %-18s 오발(안바꾸는데 write)=%s | 누락(바꾸는데 non-write)=%s" % (dom, fp, fn))

# ══ M5 · _static_blacklist / _SUCH_AS_RE ═════════════════════════════════════
class T(object):
    def __init__(s, n, d): s.name = n; s.description = d; s.openai_schema = {"description": d}
tools = [T(n, v.get("desc") or "") for n, v in ENV["banking_knowledge"]["tools"].items()]
bl = G._static_blacklist(tools, placeholders=set())
print("\n[M5] _static_blacklist(banking desc) =", sorted(bl))

# ══ M6 · _declared_params_by_tool / _DECL_ONEOF_RE ═══════════════════════════
allmsgs = [m for s in sims for m in s["M"]]
dp = G._declared_params_by_tool(allmsgs)
enum_tools = {t: {k: v for k, v in d.items() if v[1]} for t, d in dp.items()}
enum_tools = {t: d for t, d in enum_tools.items() if d}
print("\n[M6] env 명세 파싱된 도구 = %d · 그중 열거값을 뽑은 도구 = %d" % (len(dp), len(enum_tools)))
for t, d in list(enum_tools.items())[:8]:
    print("   %-42s %s" % (t, {k: v[1] for k, v in d.items()}))
# 누락: 설명에 선택지가 있는데 "Must be one of:" 형이 아니면 못 뽑는다
miss = []
for m in allmsgs:
    c = str(getattr(m, "content", "") or "")
    if "Parameters:" not in c:
        continue
    for name, typ, req, desc in G._DECL_PARAM_RE.findall(c):
        if G._DECL_ONEOF_RE.search(desc):
            continue
        if re.search(r"\b(one of|either|'[a-z_]+'\s*(,|or)\s*'[a-z_]+')", desc, re.I):
            miss.append((name, desc[:150]))
seen = set(); u = []
for a, b in miss:
    if (a, b) not in seen: seen.add((a, b)); u.append((a, b))
print("   [누락] 'Must be one of:' 아닌 형태로 선택지를 말하는 인자 =", len(u))
for a, b in u[:6]:
    print("      %-26s %r" % (a, b))

# ══ M7 · sibling_paren_arg ═══════════════════════════════════════════════════
hits = []
for s in sims:
    for m in s["M"]:
        if m.role != "assistant":
            continue
        for tc in m.tool_calls:
            r = G.sibling_paren_arg(tc)
            if r:
                hits.append((PASS(s), s["task"], r))
n, p, f, tasks = bucket([(h[0], h[1]) for h in hits])
print("\n[M7] sibling_paren_arg 발화 = %d (통과sim %d · 실패sim %d) 태스크 %s" % (n, p, f, tasks))
for h in hits[:5]:
    print("   pass=%s task=%s %s" % (h[0], h[1], h[2]))
