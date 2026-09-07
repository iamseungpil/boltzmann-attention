# -*- coding: utf-8 -*-
r"""x845 — 레버 충돌 census (오프라인·GPU 0)

물음: 우리 레버 둘이 **같은 sim 안에서 같은 도구에 반대 극성**을 지시한 일이 얼마나 있나.

왜 선언을 정적으로 읽지 않는가: 선언에서 트리거가 겹치는지 추론하면 «겹칠 수 있다» 까지만 나온다.
사이드카는 **실제로 발화한 것**의 기록이라 «겹쳤다» 를 준다([[08]]).

표적 어휘는 **env 에서 온다**(`tools.py` 의 def 이름 84개 + discoverable suffixed 이름) — 우리가
지어낸 목록이 아니다. 극성 판정은 **우리가 쓴 문면**에 대해서만 한다([[59]] 는 엔진이 *도메인
텍스트*를 뜯는 것을 금지하지, 오프라인에서 우리 출력을 세는 것을 금지하지 않는다).

한 sim 안에서 (도구 t) 에 대해 태그 A 가 REQUIRE, 태그 B 가 FORBID 면 충돌 1건.
049 의 `PROCEDURE ↔ E-PLAN` 을 **알려주지 않고 찾아내는가**가 이 도구의 자기검정이다.
"""
import json, gzip, os, re, sys, glob, collections

TOOLS_PY = "/home/woori/scratch/tau2-bench/src/tau2/domains/banking_knowledge/tools.py"
ROOTS = ["/home/woori/scratch/logs", "/home/woori/scratch/x768"]

_src = open(TOOLS_PY, encoding="utf-8").read()
VOCAB = sorted({n for n in re.findall(r"def ([a-z_][a-z_0-9]{4,})\(", _src)
                if not n.startswith("_")}, key=len, reverse=True)
VOCAB = [v for v in VOCAB if v not in ("get_current_time",)]

TAG = re.compile(r"\[([A-Z][A-Z0-9 _\-']{2,30})\]")
# 절 분리 — 우리 문면은 문장·세미콜론·대시로 끊긴다
SPLIT = re.compile(r"(?<=[.!?])\s+|\s+[-–—]\s+|;\s+|\n")

NEG = ["do not", "don't", "do NOT", "cannot", "can not", "must not", "never",
       "skip ", "skip both", "blocked", "forbid", "prohibit", "does not apply",
       "not by you", "instead of", "is not", "no longer", "without"]
POS = ["next:", "must ", "you must", "call ", "required", "require ", "apply it now",
       "now to complete", "first call", "proceed", "use "]

def polarity(clause):
    c = clause.lower()
    n = min([c.find(m) for m in NEG if m in c] or [10**9])
    p = min([c.find(m) for m in POS if m in c] or [10**9])
    if n == p == 10**9: return None
    return "FORBID" if n <= p else "REQUIRE"

def entries():
    files = []
    for r in ROOTS:
        files += glob.glob(os.path.join(r, "fb_*.jsonl")) + glob.glob(os.path.join(r, "fb_*.jsonl.gz"))
        files += glob.glob(os.path.join(r, "**", "fb_*.jsonl*"), recursive=True)
    seen = set()
    for f in sorted(set(files)):
        if os.path.getsize(f) > 40 * 1024 * 1024: continue
        op = gzip.open if f.endswith(".gz") else open
        try:
            with op(f, "rt", errors="ignore") as fh:
                for i, ln in enumerate(fh):
                    ln = ln.strip()
                    if not ln: continue
                    try: r = json.loads(ln)
                    except Exception: continue
                    st = str(r.get("simtag") or "")
                    if not st: continue
                    yield (os.path.basename(f), st, i, str(r.get("kind") or ""),
                           str(r.get("text") or r.get("feedback") or r.get("msg") or ""))
        except Exception:
            continue

# sim -> tool -> {polarity -> set(tags)}
acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(set)))
nrows = 0
for fn, st, idx, kind, text in entries():
    if not text: continue
    nrows += 1
    m = TAG.search(text[:80])
    tag = m.group(1).strip() if m else (kind or "?")
    key = (fn, st)
    for clause in SPLIT.split(text):
        for t in VOCAB:
            if t in clause:
                pol = polarity(clause)
                if pol: acc[key][t][pol].add(tag)
                break                       # 절 하나에 도구 하나만 귀속 (긴 이름 우선)

pairs = collections.Counter(); pair_tools = collections.defaultdict(collections.Counter)
sims_hit = set()
for key, tools in acc.items():
    for t, pols in tools.items():
        req, forb = pols.get("REQUIRE", set()), pols.get("FORBID", set())
        for a in req:
            for b in forb:
                if a == b: continue
                pk = tuple(sorted((a, b)))
                pairs[pk] += 1; pair_tools[pk][t] += 1; sims_hit.add(key)

print("사이드카 항목 %d · 표적 어휘 %d · 충돌이 있는 sim %d" % (nrows, len(VOCAB), len(sims_hit)))
print("\n== 충돌 태그쌍 (상위 20) ==")
print("  %-46s %6s  %s" % ("태그쌍", "sim수", "가장 잦은 표적 도구"))
for pk, n in pairs.most_common(20):
    top = ", ".join("%s(%d)" % (t, c) for t, c in pair_tools[pk].most_common(2))
    print("  %-46s %6d  %s" % (" ↔ ".join(pk), n, top))
print("\n== 자기검정: 049 가 보여준 PROCEDURE ↔ E-PLAN 이 나오는가 ==")
hit = [(pk, n) for pk, n in pairs.items()
       if {"PROCEDURE"} & set(pk) and any("PLAN" in x for x in pk)]
print("  %s" % (hit if hit else "★못 찾음 — 이 도구는 아직 못 쓴다"))
