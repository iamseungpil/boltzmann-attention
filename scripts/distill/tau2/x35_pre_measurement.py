#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x35: 지시-결함 설계서(r3)의 **사전-계측 4종** (무료·로컬·영속 데이터만·GPU 0).

정본 = `INSTRUCTION_DEFECT_REDESIGN_2026_08_01.md` §6-1. 각 항목은 **처방의 GO 조건**이다.

① 2e-2(반복 캡) — 정확 동일 재호출의 **실질 내용 동일률** + **K별 표적 시뮬 수 곡선**(K=3..12).
   ★한계(정직): 엔진이 dedup 스텁으로 **재실행을 억제한 read는 두 번째 실물 출력이 없다** ⇒
   동일률은 **양쪽 다 실물인 쌍**에서만 잴 수 있다. 그 부분모집단을 따로 보고한다.
   ★K>3 필수: C194 esc가 이미 `_n_rep>=3`에 있다(같은 카운터 재사용·중복 계수 금지).

② 2d(D3 출처 분화) — **설계서 §2d의 분류 축이 이 채널에 맞는지부터 검정한다.**
   실측: `[GROUNDING WARNING]`은 **에이전트가 넘긴 도구 파라미터**를 A2 `ground` 선언
   (`scalar_fields`/`array_fields`/`intent_fields` · `corpus`=ledger|kb|user)에 대조한 결과다
   (`t2_scaffold_get.py:1596`). ⇒ `row_fields`(다른 도구의 isolate 설정) 멤버십은 **적용 대상이 아니다**.
   대신 ⓐ 드롭 파라미터를 A2 ground 선언에 매핑하고 ⓑ **회복 여부**(같은 sim에서 그 파라미터가
   다른 값으로 경고 없이 통과)로 "다시 읽어라"의 이행 가능성을 근사한다 ⓒ 적용 모집단을 센다.

③ 2b(D1 종료 분기) — 조회 실패 후 ⓐ **다른 인자로 재조회해 성공한 비율**(= 즉시-ASK가 놓쳤을
   회복의 **상한**·R-F 대체 지표) ⓑ 손님이 대체 식별자를 실제로 준 정황.

④ 2a 케이스 3b — `_sg_details` 공집합 ∧ `ids` 비공집합이 **실재하는가**.
   궤적 텍스트엔 ratefix 템플릿이 details만 렌더하므로 관측 불가 ⇒ **op 코드 정적 census**로 답한다.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(HERE, "..", "..", "..",
                                              "reports", "facet_rft_2026", "sim_results"))
ap.add_argument("--glob", default="*.results.json.gz")
ap.add_argument("--out", default="")
A = ap.parse_args()

ENGINE_MARK = ("[DUPLICATE-READ]", "[GROUNDING WARNING]", "[GUIDANCE]", "[coverage]",
               "[quote-pin]", "[T2_", "★FEEDBACK", "[UNAVAILABLE]")


def strip_engine(text):
    """엔진이 덧붙인 구역을 벗겨 **도구 실물 내용**만 남긴다(비교용·§6-1① 오염 제거)."""
    t = str(text or "")
    for m in ENGINE_MARK:
        i = t.find(m)
        while i >= 0:
            j = t.find("\n", i)
            t = t[:i] + (t[j:] if j >= 0 else "")
            i = t.find(m)
    # esc 접미(문장 단위)와 공백 정규화
    t = re.sub(r"You have now issued this IDENTICAL call.*?again\.", " ", t, flags=re.S)
    return re.sub(r"\s+", " ", t).strip()


# ── 적재 ──────────────────────────────────────────────────────────────────────
SIMS = []
for f in sorted(glob.glob(os.path.join(A.dir, A.glob))):
    try:
        d = json.load(gzip.open(f, "rt", encoding="utf-8"))
    except Exception:
        continue
    tag = os.path.basename(f).replace(".results.json.gz", "")
    for s in d.get("simulations", []):
        msgs = s.get("messages") or []
        byid = {m.get("id"): str(m.get("content") or "")
                for m in msgs if (m.get("role") or "") == "tool"}
        seq = []
        for m in msgs:
            role = m.get("role") or ""
            if role == "user" and (m.get("content") or "").strip():
                seq.append({"kind": "user", "text": str(m.get("content"))})
            elif role == "assistant":
                for tc in (m.get("tool_calls") or []):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                    a = fn.get("arguments", tc.get("arguments"))
                    if not isinstance(a, str):
                        a = json.dumps(a, sort_keys=True, ensure_ascii=False)
                    seq.append({"kind": "call", "name": str(fn.get("name") or ""),
                                "args": a[:4000], "out": byid.get(tc.get("id"), "")})
        SIMS.append({"tag": tag, "task": s.get("task_id"), "trial": s.get("trial"),
                     "term": s.get("termination_reason"), "seq": seq,
                     "calls": [x for x in seq if x["kind"] == "call"]})
print("적재 시뮬 %d건 · 총 호출 %d\n" % (len(SIMS), sum(len(s["calls"]) for s in SIMS)))

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 92)
print("① 2e-2 반복 캡 — 실질 내용 동일률 + K별 표적 곡선")
same_real = diff_real = 0
diff_examples = collections.Counter()
for s in SIMS:
    last = {}
    for c in s["calls"]:
        k = (c["name"], c["args"])
        if k in last:
            a, b = last[k], c["out"]
            if "[DUPLICATE-READ]" not in a and "[DUPLICATE-READ]" not in b:
                if strip_engine(a) == strip_engine(b):
                    same_real += 1
                else:
                    diff_real += 1
                    diff_examples[c["name"]] += 1
        last[k] = c["out"]
tot_real = same_real + diff_real
print("  양쪽 다 **실물 출력**인 정확-재호출 쌍 = %d" % tot_real)
if tot_real:
    print("  · 실질 내용 **동일** %d (%.1f%%)  ← 재호출이 새 정보를 못 준 경우"
          % (same_real, 100.0 * same_real / tot_real))
    print("  · 실질 내용 **변화** %d (%.1f%%)  ← **정당한 재읽기**(캡이 죽이면 안 되는 것)"
          % (diff_real, 100.0 * diff_real / tot_real))
    print("  · 변화가 난 도구 top: %s"
          % ", ".join("%s×%d" % (k, v) for k, v in diff_examples.most_common(6)))
print("  ※ dedup 스텁이 재실행을 억제한 read는 두 번째 실물이 없어 이 모집단에서 제외됨(정직)")

print("\n  K별 표적(같은 (도구,인자)를 K회 **초과** 발행한 시뮬 / 그때 거부될 호출 수)")
print("  %-4s %-12s %-14s %s" % ("K", "표적 시뮬", "거부될 호출", "그중 실물-변화(과차단 하한)"))
kcurve = {}
for K in range(3, 13):
    nsim = ncall = nblock_diff = 0
    for s in SIMS:
        cnt = collections.Counter((c["name"], c["args"]) for c in s["calls"])
        over = {k: v for k, v in cnt.items() if v > K}
        if not over:
            continue
        nsim += 1
        ncall += sum(v - K for v in over.values())
        seen = collections.Counter()
        last = {}
        for c in s["calls"]:
            k = (c["name"], c["args"])
            seen[k] += 1
            if k in over and seen[k] > K and k in last:
                if ("[DUPLICATE-READ]" not in last[k] and "[DUPLICATE-READ]" not in c["out"]
                        and strip_engine(last[k]) != strip_engine(c["out"])):
                    nblock_diff += 1
            last[k] = c["out"]
    kcurve[K] = (nsim, ncall, nblock_diff)
    print("  %-4d %-12d %-14d %d" % (K, nsim, ncall, nblock_diff))

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 92)
print("② 2d 전제 검정 — GROUNDING 채널의 분류 축이 설계서 §2d와 맞는가")
A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"), encoding="utf-8"))
GROUND_PARAMS = {}     # param -> (도구, corpus, on_fail)
ROW_FIELDS = set()


def walk_a2(o, tool=None):
    if isinstance(o, dict):
        nm = o.get("name") if isinstance(o.get("name"), str) else tool
        for k, v in o.items():
            if k == "ground" and isinstance(v, dict):
                for grp in ("scalar_fields", "array_fields", "intent_fields"):
                    for fd in (v.get(grp) or []):
                        GROUND_PARAMS[str(fd.get("param"))] = (
                            nm, ",".join(fd.get("corpus") or []), fd.get("on_fail"), grp)
            elif k == "row_fields" and isinstance(v, list):
                ROW_FIELDS.update(str(x) for x in v)
            else:
                walk_a2(v, nm)
    elif isinstance(o, list):
        for v in o:
            walk_a2(v, tool)


walk_a2(A2)
print("  A2 `ground` 선언 파라미터 %d개 · `row_fields` %d개" % (len(GROUND_PARAMS), len(ROW_FIELDS)))
print("  두 집합의 교집합 = %s  ⇒ %s"
      % (sorted(set(GROUND_PARAMS) & ROW_FIELDS) or "∅",
         "**§2d의 row_fields 축은 이 채널에 적용되지 않는다**"
         if not (set(GROUND_PARAMS) & ROW_FIELDS) else "일부 겹침 — 재검토"))

DROP_RE = re.compile(r"were dropped:\s*(.+?)\.\s", re.S)
PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=")
dropped = collections.Counter()
unknown = collections.Counter()
warn_hits = 0
warn_sims = set()
recov = collections.Counter()      # param -> 회복(이후 경고 없이 같은 도구 통과)
norecov = collections.Counter()
for s in SIMS:
    hit = False
    for i, c in enumerate(s["calls"]):
        o = c["out"] or ""
        if "[GROUNDING WARNING]" not in o:
            continue
        hit = True
        warn_hits += 1
        m = DROP_RE.search(o)
        seg = m.group(1) if m else ""
        params = PAIR_RE.findall(seg)
        for p in params:
            if p in GROUND_PARAMS:
                dropped[p] += 1
            else:
                unknown[p] += 1
            # 회복 = 이후 같은 도구 호출이 경고 없이 돌아옴
            later = [x for x in s["calls"][i + 1:] if x["name"] == c["name"]]
            if later and "[GROUNDING WARNING]" not in (later[0]["out"] or ""):
                recov[p] += 1
            else:
                norecov[p] += 1
    if hit:
        warn_sims.add((s["tag"], s["task"], s["trial"]))
print("  경고 발화 %d회 · 시뮬 %d개 (적용 모집단 = ground 선언을 가진 도구를 부른 런)"
      % (warn_hits, len(warn_sims)))
print("  드롭 파라미터 (A2 ground 선언에 **실재**): %s"
      % (", ".join("%s×%d" % (k, v) for k, v in dropped.most_common(10)) or "없음"))
print("  선언에 **없는** 토큰(파싱 잡음 후보): %s"
      % (", ".join("%s×%d" % (k, v) for k, v in unknown.most_common(8)) or "없음"))
print("\n  파라미터별 이행 가능성 근사 (회복=값이 corpus에 실재했다는 증거)")
print("  %-24s %-10s %-8s %-8s %s" % ("param", "corpus", "회복", "미회복", "도구"))
for p in sorted(set(list(recov) + list(norecov)), key=lambda x: -(recov[x] + norecov[x])):
    if p not in GROUND_PARAMS:
        continue
    tool, corp, onfail, grp = GROUND_PARAMS[p]
    print("  %-24s %-10s %-8d %-8d %s" % (p, corp, recov[p], norecov[p], tool))

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 92)
print("③ 2b — 조회 실패 후 무엇이 실제로 일어났나")
LOOKUP = "get_user_information"
NOREC = "No records found"
n_fail = n_diffargs = n_diffargs_ok = n_user_after = 0
fail_sims = set()
for s in SIMS:
    for i, c in enumerate(s["seq"]):
        if c["kind"] != "call" or not c["name"].startswith(LOOKUP):
            continue
        if NOREC not in (c["out"] or ""):
            continue
        n_fail += 1
        fail_sims.add((s["tag"], s["task"], s["trial"]))
        later = [x for x in s["seq"][i + 1:] if x["kind"] == "call"
                 and x["name"].startswith(LOOKUP)]
        diff = [x for x in later if (x["name"], x["args"]) != (c["name"], c["args"])]
        if diff:
            n_diffargs += 1
            if any(NOREC not in (x["out"] or "") and "[DUPLICATE-READ]" not in (x["out"] or "")
                   for x in diff):
                n_diffargs_ok += 1
        if any(x["kind"] == "user" for x in s["seq"][i + 1:i + 6]):
            n_user_after += 1
print("  조회 실패(‘No records found’) %d회 · 시뮬 %d개" % (n_fail, len(fail_sims)))
print("  ⓐ 이후 **다른 인자로 재조회** %d회 · 그중 **성공** %d회"
      % (n_diffargs, n_diffargs_ok))
if n_fail:
    print("     ⇒ **즉시-ASK가 놓쳤을 회복의 상한 = %.1f%%** (%d/%d)"
          % (100.0 * n_diffargs_ok / n_fail, n_diffargs_ok, n_fail))
print("  ⓑ 실패 직후 5스텝 내 손님 발화가 있던 경우 %d회 (대체 식별자 제공 정황의 상한)" % n_user_after)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 92)
print("④ 2a 케이스 3b — details 공집합 ∧ ids 비공집합이 실재하는가 (op 코드 정적 census)")
src = open(os.path.join(HERE, "t2_compute.py"), encoding="utf-8").read()
ops_ids = set(re.findall(r"def (op_[a-z0-9_]+)", src))
det_pos = [m.start() for m in re.finditer(r"_sg_details", src)]
id_pos = [m.start() for m in re.finditer(r"out_ids\.append|\"ids\"|'ids'", src)]


def owner(pos):
    best = None
    for m in re.finditer(r"def ([a-z0-9_]+)\(", src):
        if m.start() <= pos:
            best = m.group(1)
        else:
            break
    return best


det_fns = collections.Counter(owner(p) for p in det_pos)
id_fns = collections.Counter(owner(p) for p in id_pos)
print("  `_sg_details`를 채우는 함수: %s" % ", ".join(sorted(x for x in det_fns if x)))
print("  ids(out_ids)를 채우는 함수 : %s" % ", ".join(sorted(x for x in id_fns if x)))
only_ids = sorted(x for x in id_fns if x and x not in det_fns)
print("  ★**ids만 채우고 details를 안 채우는 함수** = %s" % (", ".join(only_ids) or "없음"))
print("  ⇒ %s" % ("케이스 3b는 **현행 코드에서 도달 불가** ⇒ 분기를 만들지 말고 동치만 단위테스트로 못박는다"
                  if not only_ids else "케이스 3b 도달 가능 ⇒ details 기준 분기 신설 정당"))

if A.out:
    json.dump({"k_curve": kcurve, "same_real": same_real, "diff_real": diff_real,
               "ground_params": {k: list(v) for k, v in GROUND_PARAMS.items()},
               "dropped": dict(dropped), "recov": dict(recov), "norecov": dict(norecov),
               "lookup_fail": n_fail, "diffargs": n_diffargs, "diffargs_ok": n_diffargs_ok,
               "only_ids_fns": only_ids},
              open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n→ %s" % A.out)
