# -*- coding: utf-8 -*-
r"""x848 — 레버 충돌 census (x845 자기검정 실패 수리판)

x845 가 049 의 `PROCEDURE ↔ E-PLAN` 을 못 찾은 이유는 도구명 매칭만 했기 때문이다.
**두 레버가 같은 것을 다른 이름으로 부른다** — PROCEDURE 는 도구 id(`apply_statement_credit_8472`),
E-PLAN 은 역할 이름(*"the retention offer"*). 대응은 A2 가 이미 선언한다:
`procedures[].nodes[] = {id, tool|tool_any}`. 그래서 별칭 사전은 **선언 출처**이지 우리가 지은 게 아니다.

또 하나 고침: 태그 없는 항목의 `kind`(subcall·prompt·reminder-*)를 태그로 쓰면 채널이 레버처럼
집계된다. **대괄호 태그가 있는 항목만** 센다 — subcall 은 프로브 질문이지 지시가 아니다.
"""
import json, gzip, os, re, glob, collections

TOOLS_PY = "/home/woori/scratch/tau2-bench/src/tau2/domains/banking_knowledge/tools.py"
A2 = "/home/woori/scratch/repo_rep1/scripts/distill/tau2/a2/banking_knowledge.gate.json"
ROOTS = ["/home/woori/scratch/logs", "/home/woori/scratch/x768"]

_src = open(TOOLS_PY, encoding="utf-8").read()
TOOLS = {n for n in re.findall(r"def ([a-z_][a-z_0-9]{4,})\(", _src) if not n.startswith("_")}
TOOLS.discard("get_current_time")

# ── 별칭: 역할 이름 -> 정규 표적 (A2 procedures[].nodes[] 선언 출처)
ALIAS = {}
_d = json.load(open(A2, encoding="utf-8"))
for p in (_d.get("procedures") or []):
    for nd in (p.get("nodes") or []):
        sid = nd.get("id")
        tl = nd.get("tool_any") or nd.get("tool") or []
        if isinstance(tl, str): tl = [tl]
        if not (sid and tl): continue
        canon = sorted(tl)[0]
        ALIAS[sid] = canon
        ALIAS[sid.replace("_", " ")] = canon
for t in TOOLS:
    ALIAS.setdefault(t, t)
# 긴 것 먼저 — "retention offer" 가 "offer" 보다 먼저 잡히게
KEYS = sorted(ALIAS, key=len, reverse=True)

TAG = re.compile(r"\[([A-Z][A-Z0-9 _\-']{2,30})\]")
SPLIT = re.compile(r"(?<=[.!?])\s+|\s+[-–—]\s+|;\s+|\n")
NEG = ["do not", "don't", "cannot", "can not", "must not", "never", "skip ",
       "skip both", "blocked", "forbid", "prohibit", "does not apply",
       "not by you", "instead of", "no longer", "without"]
POS = ["next:", "must ", "you must", "call ", "required", "require ",
       "apply it now", "now to complete", "first call", "proceed", "use "]

def polarity(c):
    c = c.lower()
    n = min([c.find(m) for m in NEG if m in c] or [10**9])
    p = min([c.find(m) for m in POS if m in c] or [10**9])
    if n == p == 10**9: return None
    return "FORBID" if n <= p else "REQUIRE"

acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(set)))
nrows = ntag = 0
files = set()
for r in ROOTS:
    files |= set(glob.glob(os.path.join(r, "**", "fb_*.jsonl*"), recursive=True))
for f in sorted(files):
    if os.path.getsize(f) > 40 * 1024 * 1024: continue
    op = gzip.open if f.endswith(".gz") else open
    try:
        with op(f, "rt", errors="ignore") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln: continue
                try: r = json.loads(ln)
                except Exception: continue
                st = str(r.get("simtag") or "")
                text = str(r.get("text") or r.get("feedback") or r.get("msg") or "")
                if not (st and text): continue
                nrows += 1
                m = TAG.search(text[:90])
                if not m: continue                    # ★대괄호 태그가 있는 것만 = 지시
                ntag += 1
                tag = m.group(1).strip()
                for clause in SPLIT.split(text):
                    for k in KEYS:
                        if k in clause:
                            pol = polarity(clause)
                            if pol: acc[(os.path.basename(f), st)][ALIAS[k]][pol].add(tag)
                            break
    except Exception:
        continue

pairs = collections.Counter(); ptools = collections.defaultdict(collections.Counter); hit = set()
for key, tools in acc.items():
    for t, pols in tools.items():
        for a in pols.get("REQUIRE", set()):
            for b in pols.get("FORBID", set()):
                if a == b: continue
                pk = tuple(sorted((a, b)))
                pairs[pk] += 1; ptools[pk][t] += 1; hit.add(key)
print("사이드카 항목 %d (태그 있는 지시 %d) · 별칭 %d · 충돌 sim %d"
      % (nrows, ntag, len(ALIAS), len(hit)))
print("\n== 충돌 태그쌍 (상위 18) ==")
for pk, n in pairs.most_common(18):
    top = ", ".join("%s(%d)" % (t, c) for t, c in ptools[pk].most_common(2))
    print("  %-40s %5d sim  %s" % (" ↔ ".join(pk), n, top))
print("\n== 자기검정: 049 의 PROCEDURE ↔ E-PLAN ==")
h = [(pk, pairs[pk], ptools[pk].most_common(3)) for pk in pairs
     if "PROCEDURE" in pk and any("PLAN" in x for x in pk)]
for pk, n, tt in h: print("  ★찾음: %s  %d sim  %s" % (" ↔ ".join(pk), n, tt))
if not h: print("  ★여전히 못 찾음")
