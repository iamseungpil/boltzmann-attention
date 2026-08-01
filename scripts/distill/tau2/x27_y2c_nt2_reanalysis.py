# -*- coding: utf-8 -*-
"""x27: Y2-C nt=2 재분석 (2026-08-01) — 안정/흔들림·지문·채점 아티팩트·022 가드 인과.
사용 데이터: sim_results/bank_y1var_20260730, bank_y2c2_*_20260731, bank_y2cp2_*_20260801.
결과 → RESEARCH_MASTER §3 C272~C275. 실행: 아래 PART를 개별 실행(원본은 세션 스크래치의 y2c_reanalysis/part2~5).
"""

# ============================ PART: y2c_reanalysis.py ============================

"""Y2-C nt=2 재분석: 안정/흔들림 분리 + 결정론 지문 + 022/028 재확인."""
import gzip, json, sys, io
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
D = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"

def load(*names):
    sims = []
    for n in names:
        d = json.load(gzip.open(f"{D}\\{n}.results.json.gz"))
        sims.extend(d["simulations"])
    return sims

y1 = load("bank_y1var_20260730")
p1 = load("bank_y2c2_gpu0_20260731", "bank_y2c2_gpu0b_20260731", "bank_y2c2_gpu1_20260731")
p2 = load("bank_y2cp2_gpu0_20260801", "bank_y2cp2_gpu1_20260801")

y1t0 = {s["task_id"]: s for s in y1 if s["trial"] == 0}
y1t1 = {s["task_id"]: s for s in y1 if s["trial"] == 1}

def index(sims, label):
    idx = {}
    dups = []
    for s in sims:
        t = s["task_id"]
        if t in idx: dups.append(t)
        idx[t] = s
    if dups: print(f"[warn] {label} duplicate task_ids: {dups}")
    return idx

P1 = index(p1, "p1"); P2 = index(p2, "p2")
print(f"counts: Y1t0={len(y1t0)} Y1t1={len(y1t1)} P1={len(P1)} P2={len(P2)}")

tasks = sorted(set(P1) | set(P2))

def rew(s):
    if not s or not s.get("reward_info"): return None
    return int(s["reward_info"]["reward"] > 0.5)

def calls(s):
    """assistant tool-call sequence (name, sorted-args-json)"""
    out = []
    for m in s["messages"]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                out.append((tc["name"], json.dumps(tc.get("arguments", {}), sort_keys=True, ensure_ascii=False)))
    return out

def gold_summary(s):
    """(n_gold, n_matched, missing action names)"""
    ac = s["reward_info"].get("action_checks") or []
    miss = [a["action"]["name"] for a in ac if not a["action_match"]]
    return len(ac), sum(1 for a in ac if a["action_match"]), miss

# ---------- 1. per-task pass table + stability ----------
print("\n== per-task pass (Y1t0 / Y1t1 / P1 / P2) + P1vsP2 지문 ==")
classes = defaultdict(list)
fp_rows = []
for t in tasks:
    r = [rew(y1t0.get(t)), rew(y1t1.get(t)), rew(P1.get(t)), rew(P2.get(t))]
    c1, c2 = (calls(P1[t]) if t in P1 else None), (calls(P2[t]) if t in P2 else None)
    exact = (c1 == c2) if (c1 is not None and c2 is not None) else None
    names_eq = ([a for a,_ in c1] == [a for a,_ in c2]) if exact is not None else None
    # vs Y1 t0
    cy = calls(y1t0[t]) if t in y1t0 else None
    y_exact = (cy == c1) if (cy is not None and c1 is not None) else None
    p12 = (r[2], r[3])
    if p12 == (1,1): cls = "always-pass"
    elif p12 == (0,0): cls = "always-fail"
    else: cls = "FLIP"
    classes[cls].append(t)
    fp_rows.append((t, r, cls, exact, names_eq, y_exact, len(c1 or []), len(c2 or [])))
    print(f"{t}  Y1:{r[0]}/{r[1]}  P1:{r[2]} P2:{r[3]}  {cls:11s}  p1==p2 exact:{exact} names:{names_eq}  p1==Y1t0:{y_exact}  ncalls {len(c1 or [])}/{len(c2 or [])}")

print("\n== 안정성 분류 ==")
for k in ("always-pass","always-fail","FLIP"):
    print(f"{k}: {len(classes[k])}  {classes[k]}")

# pass counts
def psum(src): return sum(rew(s) or 0 for s in src.values())
print(f"\nPASS: Y1t0={psum(y1t0)}/32  Y1t1={psum(y1t1)}/{len(y1t1)}  P1={psum(P1)}/{len(P1)}  P2={psum(P2)}/{len(P2)}")
# crashed / no-reward sims
for lab, src in (("Y1t1",y1t1),("P2",P2)):
    bad = [t for t,s in src.items() if not s.get("reward_info")]
    if bad:
        for t in bad:
            s = src[t]
            print(f"[crash?] {lab} {t}: term={s.get('termination_reason')} dur={s.get('duration')} nmsg={len(s.get('messages') or [])}")

# termination reasons
print("\ntermination P1:", Counter(P1[t]["termination_reason"] for t in P1))
print("termination P2:", Counter(P2[t]["termination_reason"] for t in P2))

# ---------- 2. 022 coverage 재확인 ----------
print("\n== task_022 gold coverage (Y1t0/Y1t1/P1/P2) ==")
for lab, src in (("Y1t0",y1t0),("Y1t1",y1t1),("P1",P1),("P2",P2)):
    s = src.get("task_022")
    if not s: print(f"{lab}: absent"); continue
    n, m, miss = gold_summary(s)
    print(f"{lab}: gold {m}/{n} matched, missing({len(miss)}): {Counter(miss)}")

# ---------- 3. 028 재확인 ----------
print("\n== task_028 gold coverage ==")
for lab, src in (("Y1t0",y1t0),("Y1t1",y1t1),("P1",P1),("P2",P2)):
    s = src.get("task_028")
    if not s: print(f"{lab}: absent"); continue
    n, m, miss = gold_summary(s)
    ac = s["reward_info"]["action_checks"]
    missing_detail = [(a["action"]["name"], a["action"]["arguments"]) for a in ac if not a["action_match"]]
    print(f"{lab}: gold {m}/{n}, missing: {json.dumps(missing_detail, ensure_ascii=False)[:300]}")

# ---------- 4. 정확 중복 write (A2 write set, exact args) ----------
print("\n== 정확 중복 write (P2) ==")
# write set from tool_type in action_checks? Use per-sim executed calls: count exact duplicate call signatures among calls
for t in sorted(P2):
    c = calls(P2[t])
    dup = [k for k,v in Counter(c).items() if v > 1]
    # restrict to likely writes (heuristic: same as handoff — apply/close/dispute/transfer etc.)? report all exact dups with count
    if dup:
        wd = [(n_,a_,Counter(c)[(n_,a_)]) for n_,a_ in dup]
        print(f"{t}: {json.dumps(wd, ensure_ascii=False)[:300]}")

# ============================ PART: y2c_part2.py ============================

"""P2 실패 기전 분해 + P1 대비 기전 안정성 + 022/028/P2-only 하락 심층."""
import gzip, json, sys, io
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
D = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"

def load(*names):
    sims = []
    for n in names:
        sims.extend(json.load(gzip.open(f"{D}\\{n}.results.json.gz"))["simulations"])
    return sims

y1 = load("bank_y1var_20260730")
P1 = {s["task_id"]: s for s in load("bank_y2c2_gpu0_20260731","bank_y2c2_gpu0b_20260731","bank_y2c2_gpu1_20260731")}
P2 = {s["task_id"]: s for s in load("bank_y2cp2_gpu0_20260801","bank_y2cp2_gpu1_20260801")}
Y1t0 = {s["task_id"]: s for s in y1 if s["trial"]==0}

def rew(s):
    if not s or not s.get("reward_info"): return None
    return int(s["reward_info"]["reward"] > 0.5)

def calls(s):
    out = []
    for m in s["messages"]:
        if m.get("role")=="assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                out.append((tc["name"], json.dumps(tc.get("arguments",{}), sort_keys=True, ensure_ascii=False)))
    return out

def diag(s):
    ri = s["reward_info"]
    ac = ri.get("action_checks") or []
    missing = [(a["action"]["name"], a.get("action",{}).get("arguments")) for a in ac if not a["action_match"]]
    c = calls(s)
    names = [n for n,_ in c]
    # verify triple fingerprint
    triple = 0
    for i in range(len(names)-2):
        if names[i:i+3]==["verify_identity","get_current_time","log_verification"] or \
           names[i]=="get_current_time" and names[i+1]=="log_verification":
            triple += 1
    top = Counter(names).most_common(4)
    db = ri.get("db_check") or {}
    nl = ri.get("nl_assertions") or []
    nl_fail = [a.get("nl_assertion","")[:60] for a in nl if not a.get("met", a.get("passed", True))]
    comm = ri.get("communicate_checks") or []
    comm_fail = [str(a)[:60] for a in comm if not a.get("met", True)]
    return {
        "term": s["termination_reason"], "ncalls": len(c),
        "db_match": db.get("db_match"), "basis": ri.get("reward_basis"),
        "n_gold": len(ac), "n_miss": len(missing),
        "miss_names": Counter(n for n,_ in missing),
        "top_calls": top, "nl_fail": nl_fail, "comm_fail": comm_fail,
    }

print("== P2 실패 진단 (P1 결과 병기) ==")
for t in sorted(P2):
    r2, r1 = rew(P2[t]), rew(P1.get(t))
    if r2 != 0: continue
    d = diag(P2[t])
    d1 = diag(P1[t]) if t in P1 and P1[t].get("reward_info") else None
    print(f"\n-- {t} (P1={r1} P2=0) term={d['term']} ncalls={d['ncalls']} db={d['db_match']} basis={d['basis']}")
    print(f"   gold miss {d['n_miss']}/{d['n_gold']}: {dict(d['miss_names'])}")
    if d['nl_fail']: print(f"   nl_fail: {d['nl_fail']}")
    if d['comm_fail']: print(f"   comm_fail: {d['comm_fail']}")
    print(f"   top_calls: {d['top_calls']}")
    if d1 and r1==0:
        same_miss = dict(d1['miss_names'])==dict(d['miss_names'])
        print(f"   [P1 미스와 동일집합: {same_miss}]  P1 miss: {dict(d1['miss_names'])}")

# ---- P2-only 하락 4건 자세히 ----
print("\n\n== P2-only 하락 (002/017/021/025): 무엇이 달라졌나 ==")
for t in ["task_002","task_017","task_021","task_025"]:
    s1, s2 = P1[t], P2[t]
    c1, c2 = calls(s1), calls(s2)
    # divergence point
    k = 0
    for a,b in zip(c1,c2):
        if a!=b: break
        k += 1
    print(f"\n-- {t}: P1 {len(c1)}calls PASS / P2 {len(c2)}calls FAIL. 공통 prefix={k}")
    print(f"   P1 call {k}: {c1[k] if k<len(c1) else 'END'}")
    print(f"   P2 call {k}: {c2[k] if k<len(c2) else 'END'}")
    d2 = diag(s2)
    print(f"   P2 miss: {dict(d2['miss_names'])} db={d2['db_match']} nl_fail={d2['nl_fail']} comm={d2['comm_fail']}")

# ---- 022 executed vs gold ----
print("\n\n== task_022 executed discoverable calls vs gold ==")
for lab, src in (("P1",P1),("P2",P2)):
    s = src["task_022"]
    ac = s["reward_info"]["action_checks"]
    gold = [(a["action"]["name"], a["action"]["arguments"], a["action_match"]) for a in ac]
    exe = [c for c in calls(s) if "discoverable" in c[0]]
    print(f"\n[{lab}] executed discoverable ({len(exe)}):")
    for n,a in exe: print(f"   {n} {a[:130]}")
    print(f"[{lab}] gold ({len(gold)}):")
    for n,a,m in gold: print(f"   {'✓' if m else '✗'} {n} {json.dumps(a,ensure_ascii=False)[:130]}")

# ============================ PART: y2c_part3.py ============================

"""마무리 확인: 001 db실패 원인 / 002·025 첫 발화 비교 / write 정확중복 재계수 / 026 크래시."""
import gzip, json, sys, io
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
D = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"

def load(*names):
    sims = []
    for n in names:
        sims.extend(json.load(gzip.open(f"{D}\\{n}.results.json.gz"))["simulations"])
    return sims

y1 = load("bank_y1var_20260730")
p1s = load("bank_y2c2_gpu0_20260731","bank_y2c2_gpu0b_20260731","bank_y2c2_gpu1_20260731")
p2s = load("bank_y2cp2_gpu0_20260801","bank_y2cp2_gpu1_20260801")
P1 = {s["task_id"]: s for s in p1s}
P2 = {s["task_id"]: s for s in p2s}
Y1t0 = {s["task_id"]: s for s in y1 if s["trial"]==0}
Y1t1 = {s["task_id"]: s for s in y1 if s["trial"]==1}

# write tool set from gold action_checks
writes = set()
for s in p1s + p2s + y1:
    for a in (s.get("reward_info") or {}).get("action_checks") or []:
        if a.get("tool_type") == "write":
            writes.add(a["action"]["name"])
print("write set:", sorted(writes))

def calls(s, roles=("assistant",)):
    out = []
    for m in s["messages"]:
        if m.get("role") in roles and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                out.append((m.get("role"), tc["name"], json.dumps(tc.get("arguments",{}), sort_keys=True, ensure_ascii=False)))
    return out

# ---- 정확 중복 write 재계수 (4 런 전부) ----
print("\n== 정확 중복 write (write-집합·정확 인자 일치) ==")
for lab, src in (("Y1t0",Y1t0),("Y1t1",Y1t1),("P1",P1),("P2",P2)):
    hits = []
    for t, s in sorted(src.items()):
        if not s.get("messages"): continue
        c = [(n,a) for r,n,a in calls(s) if n in writes]
        dup = {k:v for k,v in Counter(c).items() if v>1}
        if dup: hits.append((t, {f"{n}({a[:60]})":v for (n,a),v in dup.items()}))
    print(f"{lab}: {len(hits)} sims — {[h[0] for h in hits]}")
    for t,d in hits: print(f"    {t}: {d}")

# ---- 001 P2: gold 전부 매치인데 db=False 원인 ----
print("\n== task_001 P2 (gold 1/1 ✓, db=False) ==")
s = P2["task_001"]
for r,n,a in calls(s, roles=("assistant","user")):
    print(f"  [{r}] {n} {a[:150]}")
ri = s["reward_info"]
print("db_check:", json.dumps(ri["db_check"], ensure_ascii=False)[:400])
# effect_timeline?
et = s.get("effect_timeline")
if et: print("effect_timeline:", json.dumps(et, ensure_ascii=False)[:600])

# user tool calls?
print("\nuser-side tool calls in 001 P2:", [ (n,a[:80]) for r,n,a in calls(s, roles=("user",)) ])

# ---- 002/025 첫 user 발화 비교 ----
print("\n== 첫 user 발화 (P1 vs P2) ==")
for t in ["task_002","task_025","task_017","task_021"]:
    for lab, src in (("P1",P1),("P2",P2)):
        s = src[t]
        first = next((m for m in s["messages"] if m.get("role")=="user" and m.get("content")), None)
        print(f"{t} {lab}: {(first['content'] if first else 'NONE')[:160]}")
    print()

# ---- 026 크래시 확인 ----
s = P2["task_026"]
print("== 026 P2 crash ==")
print("term:", s["termination_reason"], "info:", json.dumps(s.get("info"), ensure_ascii=False)[:400])

# ---- 022: user가 tool을 실제로 호출했는가 (전 런) ----
print("\n== 022: user-side call_discoverable_user_tool 수 ==")
for lab, src in (("Y1t0",Y1t0),("Y1t1",Y1t1),("P1",P1),("P2",P2)):
    s = src.get("task_022")
    uc = [ (n,a[:70]) for r,n,a in calls(s, roles=("user",)) ] if s and s.get("messages") else []
    print(f"{lab}: user tool calls={len(uc)}")
    for n,a in uc[:12]: print(f"    {n} {a}")

# ============================ PART: y2c_part4.py ============================

"""① 공백-리터럴 아티팩트 정량화(semantic re-match) ② pass 뒤집힘 여부 ③ 022 ba8b 가시성."""
import gzip, json, sys, io
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
D = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"

def load(*names):
    sims = []
    for n in names:
        sims.extend(json.load(gzip.open(f"{D}\\{n}.results.json.gz"))["simulations"])
    return sims

RUNS = {
    "Y1t0": [s for s in load("bank_y1var_20260730") if s["trial"]==0],
    "Y1t1": [s for s in load("bank_y1var_20260730") if s["trial"]==1],
    "P1":   load("bank_y2c2_gpu0_20260731","bank_y2c2_gpu0b_20260731","bank_y2c2_gpu1_20260731"),
    "P2":   load("bank_y2cp2_gpu0_20260801","bank_y2cp2_gpu1_20260801"),
}

def all_tool_calls(s):
    out = []
    for m in s["messages"]:
        if m.get("role") in ("assistant","user") and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                out.append(tc)
    return out

def norm(v):
    """중첩 JSON 문자열을 파싱해 동등 비교 가능한 형태로."""
    if isinstance(v, str):
        t = v.strip()
        if t.startswith("{") or t.startswith("["):
            try: return json.loads(t)
            except Exception: return v
    return v

def sem_eq(gold_args, exec_args):
    if set(gold_args.keys()) != set(exec_args.keys()):
        # evaluator compares only keys present in exec (compare_args=None → exec keys)
        pass
    keys = gold_args.keys()
    try:
        return all(norm(gold_args.get(k)) == norm(exec_args.get(k)) for k in keys)
    except Exception:
        return False

print("== 공백/포맷-리터럴 아티팩트 정량화: miss 중 semantic-match 존재 수 ==")
flip_candidates = []
for lab, sims in RUNS.items():
    tot_miss = art_miss = 0
    per_task = {}
    for s in sims:
        ri = s.get("reward_info")
        if not ri or not ri.get("action_checks"): continue
        execs = all_tool_calls(s)
        n_art = n_miss = 0
        for a in ri["action_checks"]:
            if a["action_match"]: continue
            n_miss += 1
            g = a["action"]
            sem = any(tc["name"]==g["name"] and sem_eq(g["arguments"], tc.get("arguments",{})) for tc in execs)
            if sem: n_art += 1
        tot_miss += n_miss; art_miss += n_art
        if n_art: per_task[s["task_id"]] = (n_art, n_miss)
        # would ACTION-basis task flip?
        if n_miss>0 and n_art==n_miss and (ri.get("reward_basis")==["ACTION"]):
            flip_candidates.append((lab, s["task_id"]))
    print(f"{lab}: 전체 miss {tot_miss} 중 semantic-match 존재(=아티팩트) {art_miss}")
    for t,(a,m) in sorted(per_task.items()): print(f"    {t}: {a}/{m}")
print("\nACTION-basis 판정 뒤집힘 후보:", flip_candidates)

# ---- 022 ba8b visibility ----
print("\n== 022: get_reward_discrepancies 반환 내 txn 목록 ==")
import re
for lab in ("Y1t1","P2"):
    sims = RUNS[lab]
    s = next(x for x in sims if x["task_id"]=="task_022")
    seen = []
    for m in s["messages"]:
        if m.get("role")=="tool":
            c = m.get("content")
            txt = json.dumps(c, ensure_ascii=False) if not isinstance(c,str) else c
            if "discrepan" in txt.lower() or "txn_" in txt:
                txs = re.findall(r"txn_[0-9a-f]{6,}", txt)
                if txs: seen.append((m.get("name") or "?", sorted(set(txs))))
    # union of all tool-returned txns
    from functools import reduce
    allt = sorted(set(t for _,ts in seen for t in ts))
    print(f"{lab}: tool 반환 txn 총 {len(allt)}종 / ba8b473f295d {'포함' if any('ba8b' in t for t in allt) else '❌미포함'}")
    for n,ts in seen[:6]: print(f"    [{n}] {len(ts)}종: {[t[:14] for t in ts][:12]}")

# ============================ PART: y2c_part5.py ============================

"""결정적 3확인: ①022 엔진 coverage 라인 전 런 ②첫 user 발화 동일성 ③2-trial 짝 검정."""
import gzip, json, sys, io, re
from collections import Counter
from math import comb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
D = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"

def load(*names):
    sims = []
    for n in names:
        sims.extend(json.load(gzip.open(f"{D}\\{n}.results.json.gz"))["simulations"])
    return sims

y1 = load("bank_y1var_20260730")
RUNS = {
    "Y1t0": {s["task_id"]: s for s in y1 if s["trial"]==0},
    "Y1t1": {s["task_id"]: s for s in y1 if s["trial"]==1},
    "P1":   {s["task_id"]: s for s in load("bank_y2c2_gpu0_20260731","bank_y2c2_gpu0b_20260731","bank_y2c2_gpu1_20260731")},
    "P2":   {s["task_id"]: s for s in load("bank_y2cp2_gpu0_20260801","bank_y2cp2_gpu1_20260801")},
}

print("== ① task_022: get_reward_discrepancies 반환의 coverage 라인 (전 런) ==")
for lab, src in RUNS.items():
    s = src.get("task_022")
    if not s or not s.get("messages"): print(f"{lab}: absent"); continue
    found = []
    for m in s["messages"]:
        if m.get("role")!="tool": continue
        c = m.get("content"); txt = c if isinstance(c,str) else json.dumps(c,ensure_ascii=False)
        if "coverage" in txt or "could not be verified" in txt:
            mm = re.search(r"\[coverage\][^\n]*", txt)
            ndisc = len(set(re.findall(r"txn_[0-9a-f]{6,}", txt)))
            found.append((ndisc, (mm.group(0) if mm else txt[:200])[:260]))
    print(f"{lab}: {len(found)}회")
    for n,l in found[:3]: print(f"    ({n}txn) {l}")

print("\n== ② 첫 user 발화 동일성 (Y1t0 vs Y1t1, P1 vs P2) ==")
def first_user(s):
    for m in s["messages"]:
        if m.get("role")=="user" and m.get("content"): return m["content"].strip()
    return None
def cmp_first(a, b, la, lb):
    same = diff = miss = 0; difftasks = []
    for t in sorted(set(a)&set(b)):
        fa, fb = (first_user(a[t]) if a[t].get("messages") else None), (first_user(b[t]) if b[t].get("messages") else None)
        if fa is None or fb is None: miss += 1; continue
        if fa == fb: same += 1
        else: diff += 1; difftasks.append(t)
    print(f"{la} vs {lb}: 동일 {same} / 상이 {diff} / 결측 {miss}")
    print(f"    상이: {difftasks}")
cmp_first(RUNS["Y1t0"], RUNS["Y1t1"], "Y1t0","Y1t1")
cmp_first(RUNS["P1"], RUNS["P2"], "P1","P2")
cmp_first(RUNS["Y1t0"], RUNS["P1"], "Y1t0","P1")

print("\n== ③ 2-trial 짝 검정 (Y1 {t0,t1} vs Y2-C {P1,P2}) ==")
def rew(s):
    if not s or not s.get("reward_info"): return None
    return int(s["reward_info"]["reward"] > 0.5)
rows = []
for t in sorted(RUNS["P1"]):
    a = [rew(RUNS["Y1t0"].get(t)), rew(RUNS["Y1t1"].get(t))]
    b = [rew(RUNS["P1"].get(t)), rew(RUNS["P2"].get(t))]
    if None in a or None in b:
        print(f"  [제외] {t}: Y1={a} Y2C={b}")
        continue
    rows.append((t, sum(a), sum(b)))
print(f"완전 짝 태스크 n={len(rows)}")
better = sum(1 for _,x,y in rows if y>x); worse = sum(1 for _,x,y in rows if y<x); tie = sum(1 for _,x,y in rows if y==x)
print(f"Y2-C 우세 {better} / 열세 {worse} / 동률 {tie}")
n = better+worse
p = sum(comb(n,k) for k in range(0, min(better,worse)+1))*2 / (2**n) if n else 1.0
print(f"부호검정 양측 p = {min(1.0,p):.4f}  (n={n})")
print(f"총 pass: Y1={sum(x for _,x,_ in rows)}/{2*len(rows)}  Y2C={sum(y for _,_,y in rows)}/{2*len(rows)}")
print("열세 태스크:", [t for t,x,y in rows if y<x])
print("우세 태스크:", [t for t,x,y in rows if y>x])

print("\n== 4런 전체 per-task pass 합 (0..4) — 안정/흔들림 ==")
tot = Counter()
for t in sorted(RUNS["P1"]):
    v = [rew(RUNS[l].get(t)) for l in ("Y1t0","Y1t1","P1","P2")]
    k = sum(x for x in v if x)
    tot[k] += 1
    print(f"  {t}: {v} → {k}/4" + ("  ★결정론적 실패" if k==0 else ("  ★결정론적 통과" if k==4 else "")))
print("분포(pass 수:태스크 수):", dict(sorted(tot.items())))
