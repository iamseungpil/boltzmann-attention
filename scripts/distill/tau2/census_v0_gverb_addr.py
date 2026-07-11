#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V0 오프라인 census — GROUND-VERBATIM(설계 §1) + DISAMB-ADDR(설계 §4).

설계 정본: reports/facet_rft_2026/CENSUS_LEVERS_DESIGN_2026_07_11.md
COMP 실패 sims 전수를 궤적만으로(실행 0·무료) 분석:

  --lever gverb : write 인자 값 v에 대해 문맥(에이전트-기조회 tool 출력)의 동일-필드
                  후보 C 구성 → fuzzy-|C|=1 치환의 fix/break/neutral census.
                  empty-값(v="")은 별도 집계(설계 §1 채택 게이트: break>0이면 empty 제외).
  --lever addr  : 주소-필드 gold-불일치/누락 write에 대해 write 시점(누락=sim 종료) 문맥에
                  gold 주소(address1)가 실재했는지 → 문맥-실재 vs 미조회 분해 + 완전한
                  주소 레코드 수 |C| 분포(|C|>=2 = DISAMB 발화 조건).

판정 규칙(구체화):
  norm    = 소문자화 + 구두점→공백 + 공백 압축 (축약 확장 없음)
  fuzzy   = norm 일치 or 접두 토큰열(토큰 수 동일·각 토큰이 서로의 접두.
            단 숫자 토큰은 정확 일치만 — 번지수 "12" vs "123" 오매치 방지)
  exact   = 쓴 값 v가 write 이전 문맥 원문에 그대로 존재(단어경계 필수) → 무개입
  fix     = 치환값 c == gold ∧ 원래 v != gold / break = v == gold ∧ c != gold

usage:
  py -3 census_v0_gverb_addr.py --lever gverb --results comp_retail_t4.results.json.gz --tasks tasks.json
  py -3 census_v0_gverb_addr.py --lever addr  --results <gz> --tasks <tasks.json>
  py -3 census_v0_gverb_addr.py --selftest
"""
import argparse, gzip, json, os, re, sys
from collections import Counter, defaultdict

DEFAULT_FIELDS = "address1,address2,city,state,zip,country"


def load_json(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def args_of(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


# ---------- 정규화·fuzzy 판정 (설계 §1) ----------

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(s):
    """소문자화·구두점 제거·공백 압축. 축약 확장은 하지 않음(원문 보존 원칙)."""
    if s is None:
        return ""
    return " ".join(_PUNCT.sub(" ", str(s).lower()).split())


def tokens_prefix_eq(ta, tb):
    """접두 토큰열 판정: 토큰 수 동일 + 각 위치에서 한쪽이 다른쪽의 접두.
    숫자 토큰(번지수·zip)은 정확 일치만 — "12 Main" vs "123 Main" 오매치 방지."""
    if not ta or len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        if x.isdigit() or y.isdigit():
            return False
        if not (x.startswith(y) or y.startswith(x)):
            return False
    return True


def fuzzy_eq(v, c):
    """norm 후 일치 or 접두 토큰열 (예: "123 elm st" ~ "123 Elm Street")."""
    nv, nc = norm(v), norm(c)
    if not nv or not nc:
        return False
    return nv == nc or tokens_prefix_eq(nv.split(), nc.split())


def exact_in(v, ctx):
    """v가 문맥 원문에 '그대로' 존재 — 단어경계 필수(단순 substring이면
    "123 Elm St"가 "123 Elm Street" 내부에 잡혀 축약 케이스가 exact 오판됨)."""
    if not v:
        return False
    return re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", ctx) is not None


def classify(v, c, gold):
    """치환 v->c의 gold 대비 효과. gold=None(대응 gold 필드 없음)이면 neutral."""
    if gold is None:
        return "neutral"
    was, now = (v == gold), (c == gold)
    if now and not was:
        return "fix"
    if was and not now:
        return "break"
    return "neutral"


# ---------- 후보 추출 (tool 출력 JSON) ----------

def field_values(objs, field):
    """파싱된 tool 출력들에서 key==field 인 스칼라 값 수집(등장순·중복 제거·빈값 제외).
    t2_gate_patch._iter_scalars의 단순 재구현(임포트 금지 지시)."""
    out, seen = [], set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (dict, list)):
                    walk(v)
                elif k == field:
                    s = "" if v is None else str(v)
                    if s and s not in seen:
                        seen.add(s)
                        out.append(s)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for o in objs:
        walk(o)
    return out


def addr_records(objs, fields):
    """완전한 주소 레코드(=anchor 필드 fields[0]를 가진 dict) 단위 수집·중복 제거.
    DISAMB 발화 조건 |C|>=2 판정용 (설계 §4)."""
    anchor = fields[0]
    seen, out = set(), []

    def walk(o):
        if isinstance(o, dict):
            if anchor in o and not isinstance(o.get(anchor), (dict, list)):
                key = tuple((f, str(o.get(f))) for f in fields if f in o)
                if key not in seen:
                    seen.add(key)
                    out.append(dict(key))
            for v in o.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for o in objs:
        walk(o)
    return out


# ---------- 궤적 스캔 (ecomp_fail_census ctx_parts 패턴) ----------

def scan_sim(sim, write_tools):
    """messages 시간순 스캔 → (write events, ctx_parts, tool_jsons).
    ctx = user 발화 + 에러 아닌 tool 출력 텍스트(원문). event의 ctx_i/out_i는
    그 write *이전* 스냅샷 길이(해당 write의 결과는 미포함)."""
    msgs = sim.get("messages") or []
    res_by_id = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
    ctx_parts, tool_jsons, events = [], [], []
    for m in msgs:
        role, c = m.get("role"), m.get("content")
        if role == "user" and isinstance(c, str):
            ctx_parts.append(c)
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                nm = tc.get("name")
                if nm in write_tools:
                    tm = res_by_id.get(tc.get("id"))
                    ok = tm is not None and not tm.get("error")
                    events.append({"name": nm, "args": args_of(tc.get("arguments")),
                                   "ok": ok, "ctx_i": len(ctx_parts), "out_i": len(tool_jsons)})
        if role == "tool" and isinstance(c, str) and not m.get("error"):
            ctx_parts.append(c)
            try:
                tool_jsons.append(json.loads(c))
            except Exception:
                pass
    return events, ctx_parts, tool_jsons


def gold_writes(task, write_tools):
    """gold write actions (t5c_taskdiff.gold_writes 패턴·write_tools는 A2 유래)."""
    return [(x.get("name"), args_of(x.get("arguments")))
            for x in ((task.get("evaluation_criteria") or {}).get("actions") or [])
            if x.get("requestor", "assistant") == "assistant" and x.get("name") in write_tools]


def best_gold(args, cands):
    """같은 도구의 gold 후보 중 최소-diff 선택 (t5c diff_str 패턴)."""
    if not cands:
        return None
    return min(cands, key=lambda g: sum(1 for k in set(g) | set(args)
                                        if str(g.get(k)) != str(args.get(k))))


def iter_failed(sims):
    for s in sims:
        r = (s.get("reward_info") or {}).get("reward")
        if r is None or r >= 1:
            continue
        yield s


# ---------- 모드 gverb (설계 §1) ----------

def run_gverb(sims, tasks, write_tools, fields, out=sys.stdout):
    tot, rows = Counter(), []
    for s in iter_failed(sims):
        tot["fail_sims"] += 1
        tid = str(s.get("task_id"))
        gold_by_tool = defaultdict(list)
        for nm, ar in gold_writes(tasks.get(tid) or {}, write_tools):
            gold_by_tool[nm].append(ar)
        events, ctx_parts, tool_jsons = scan_sim(s, write_tools)
        for ev in events:
            if not ev["ok"]:
                continue  # DB 반영분(실행 성공 write)만 — fix/break는 DB-diff 기준
            ar = ev["args"]
            best = best_gold(ar, gold_by_tool.get(ev["name"]))
            ctx = "\n".join(ctx_parts[:ev["ctx_i"]])
            outs = tool_jsons[:ev["out_i"]]
            for f in fields:
                if f not in ar:
                    continue
                v = ar.get(f)
                v_s = "" if v is None else str(v)
                gv = None if best is None else best.get(f)
                gv_s = None if gv is None else str(gv)
                C = field_values(outs, f)
                if v_s == "":
                    # empty-케이스(t39형): |C|=1이면 치환 *후보*로만 별도 집계 (설계 §1)
                    if len(C) == 1:
                        verdict = classify(v_s, C[0], gv_s)
                        tot["empty_" + verdict] += 1
                        rows.append((tid, s.get("trial"), ev["name"], f,
                                     v_s, C[0], gv_s, "empty_" + verdict))
                    else:
                        tot["empty_nocand"] += 1
                    continue
                if exact_in(v_s, ctx):
                    tot["exact"] += 1  # 원문 그대로 문맥에 존재 → 무개입
                    continue
                m = [c for c in C if fuzzy_eq(v_s, c)]
                if len(m) != 1:
                    tot["multi_cand" if len(m) > 1 else "no_match"] += 1  # |C|>=2=DISAMB 관할·C=empty=무개입
                    continue
                c = m[0]
                if c == v_s:
                    tot["exact"] += 1  # 치환해도 동일 → no-op
                    continue
                verdict = classify(v_s, c, gv_s)
                tot[verdict] += 1
                rows.append((tid, s.get("trial"), ev["name"], f, v_s, c, gv_s, verdict))

    print("== GVERB census (design §1) ==", file=out)
    print("fail_sims=%d | fix=%d break=%d neutral=%d | empty_fix=%d empty_break=%d "
          "empty_neutral=%d empty_nocand=%d | exact(no-op)=%d no_match=%d multi_cand=%d"
          % (tot["fail_sims"], tot["fix"], tot["break"], tot["neutral"],
             tot["empty_fix"], tot["empty_break"], tot["empty_neutral"], tot["empty_nocand"],
             tot["exact"], tot["no_match"], tot["multi_cand"]), file=out)
    go = tot["fix"] >= 5 and tot["break"] == 0
    print("GO gate (fix>=5 and break==0): %s" % ("GO" if go else "NO-GO"), file=out)
    if rows:
        print("%-5s %-5s %-32s %-9s %s" % ("task", "trial", "tool", "field",
                                           "written -> substitute | gold | verdict"), file=out)
        for tid, tr, nm, f, v, c, g, verdict in rows:
            print("t%-4s %-5s %-32s %-9s '%s' -> '%s' | gold %r | %s"
                  % (tid, tr, nm, f, v, c, g, verdict), file=out)
    return tot, rows


# ---------- 모드 addr (설계 §4) ----------

def run_addr(sims, tasks, write_tools, fields, out=sys.stdout):
    anchor = fields[0]
    tot, cdist, rows = Counter(), Counter(), []
    per_task = defaultdict(Counter)
    for s in iter_failed(sims):
        tot["fail_sims"] += 1
        tid = str(s.get("task_id"))
        gold_by_tool = defaultdict(list)
        for nm, ar in gold_writes(tasks.get(tid) or {}, write_tools):
            gold_by_tool[nm].append(ar)
        events, ctx_parts, tool_jsons = scan_sim(s, write_tools)
        okwrites = [ev for ev in events if ev["ok"]]

        cases = []
        # (a) 주소 필드가 gold와 불일치한 write
        for ev in okwrites:
            best = best_gold(ev["args"], gold_by_tool.get(ev["name"]))
            if best is None:
                continue
            touched = [f for f in fields if f in best or f in ev["args"]]
            mism = [f for f in touched if str(best.get(f)) != str(ev["args"].get(f))]
            if mism:
                cases.append(("wrong", ev["name"], best, ev["args"],
                              ev["ctx_i"], ev["out_i"], mism))
        # (b) 주소 gold write 자체가 누락 (실행 수 < gold 수 근사·ctx=sim 종료 시점)
        for nm, cands in gold_by_tool.items():
            addr_golds = [g for g in cands if any(f in g for f in fields)]
            if not addr_golds:
                continue
            miss_n = len(cands) - sum(1 for ev in okwrites if ev["name"] == nm)
            for g in addr_golds[:max(0, miss_n)]:
                cases.append(("missing", nm, g, {},
                              len(ctx_parts), len(tool_jsons), list(fields)))

        for kind, nm, g, ar, ctx_i, out_i, mism in cases:
            tot["cases"] += 1
            tot["case_" + kind] += 1
            per_task[tid]["n_" + kind] += 1
            g1 = g.get(anchor)
            if g1 is None or norm(g1) == "":
                tot["no_gold_anchor"] += 1
                per_task[tid]["no_gold_anchor"] += 1
                rows.append((tid, s.get("trial"), kind, nm, "n/a", None,
                             g1, ar.get(anchor)))
                continue
            # 문맥-실재 판정: gold address1의 norm이 문맥 norm에 부분열(substring)로 존재
            present = norm(g1) in norm("\n".join(ctx_parts[:ctx_i]))
            nrec = None
            if present:
                tot["present"] += 1
                per_task[tid]["present"] += 1
                nrec = len(addr_records(tool_jsons[:out_i], fields))
                cdist[nrec] += 1
                if nrec >= 2:
                    tot["present_C_ge2"] += 1
            else:
                tot["not_found"] += 1
                per_task[tid]["not_found"] += 1
            rows.append((tid, s.get("trial"), kind, nm,
                         "present" if present else "not_found", nrec,
                         g1, ar.get(anchor)))

    print("== DISAMB-ADDR census (design §4) ==", file=out)
    print("fail_sims=%d addr_cases=%d (wrong=%d missing=%d) | context-present=%d "
          "not-looked-up=%d no_gold_anchor=%d"
          % (tot["fail_sims"], tot["cases"], tot["case_wrong"], tot["case_missing"],
             tot["present"], tot["not_found"], tot["no_gold_anchor"]), file=out)
    if tot["present"]:
        print("|C| distribution (present cases): %s | C>=2 (DISAMB-eligible): %d/%d (%.0f%%)"
              % (dict(sorted(cdist.items())), tot["present_C_ge2"], tot["present"],
                 100.0 * tot["present_C_ge2"] / tot["present"]), file=out)
    if per_task:
        print("%-6s %6s %8s %8s %10s" % ("task", "wrong", "missing", "present", "not_found"), file=out)
        for tid in sorted(per_task, key=lambda x: (len(x), x)):
            c = per_task[tid]
            print("t%-5s %6d %8d %8d %10d" % (tid, c["n_wrong"], c["n_missing"],
                                              c["present"], c["not_found"]), file=out)
    if rows:
        print("%-5s %-5s %-8s %-32s %-10s %-4s gold_addr1 | written_addr1"
              % ("task", "trial", "kind", "tool", "presence", "|C|"), file=out)
        for tid, tr, kind, nm, pres, nrec, g1, w1 in rows:
            print("t%-4s %-5s %-8s %-32s %-10s %-4s %r | %r"
                  % (tid, tr, kind, nm, pres, ("-" if nrec is None else nrec), g1, w1), file=out)
    return tot, rows


# ---------- selftest (외부 파일 불요) ----------

def _sim(msgs, tid, trial=0, reward=0.0):
    return {"task_id": tid, "trial": trial, "reward_info": {"reward": reward},
            "messages": msgs}


def _user(txt):
    return {"role": "user", "content": txt}


def _tool(mid, payload, error=False):
    return {"role": "tool", "id": mid,
            "content": payload if isinstance(payload, str) else json.dumps(payload),
            "error": error}


def _write(mid, name, args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": mid, "name": name, "arguments": json.dumps(args)}]}


def _task(tid, actions):
    return {"id": tid, "evaluation_criteria":
            {"actions": [{"name": n, "arguments": a} for n, a in actions]}}


def selftest():
    import io
    fields = [f.strip() for f in DEFAULT_FIELDS.split(",")]
    fails = []

    def ok(name, cond):
        fails.append(name) if not cond else None
        print("  %-58s %s" % (name, "PASS" if cond else "FAIL"))

    # 1) norm
    ok("norm: punct+case+space", norm(" 123  Elm St. ") == "123 elm st")
    ok("norm: None -> ''", norm(None) == "")
    # 2) fuzzy
    ok("fuzzy: prefix-token abbrev", fuzzy_eq("123 Elm St", "123 Elm Street"))
    ok("fuzzy: symmetric", fuzzy_eq("123 Elm Street", "123 Elm St"))
    ok("fuzzy: different street", not fuzzy_eq("456 Oak St", "123 Elm Street"))
    ok("fuzzy: digit tokens exact-only", not fuzzy_eq("12 Main St", "123 Main St"))
    ok("fuzzy: norm-equal case", fuzzy_eq("SEATTLE", "seattle"))
    ok("fuzzy: token count differs", not fuzzy_eq("123 Elm", "123 Elm Street"))
    ok("fuzzy: empty no-match", not fuzzy_eq("", "x"))
    # 3) 후보 추출
    payload = {"orders": [{"address": {"address1": "123 Elm Street", "city": "Seattle"}},
                          {"address": {"address1": "9 Old Rd", "city": "Dallas"}},
                          {"address": {"address1": "123 Elm Street", "city": "Seattle"}}]}
    ok("field_values: dedup + order",
       field_values([payload], "address1") == ["123 Elm Street", "9 Old Rd"])
    ok("field_values: empty excluded",
       field_values([{"address1": ""}], "address1") == [])
    # 4) 주소 레코드 파싱
    recs = addr_records([payload], fields)
    ok("addr_records: dedup record count", len(recs) == 2)
    ok("addr_records: fields captured", recs[0].get("city") == "Seattle")

    W = {"modify_addr"}
    # 5) gverb e2e: fix / break / empty_fix / multi_cand / exact
    tasks = {t["id"]: t for t in [
        _task("1", [("modify_addr", {"address1": "123 Elm Street", "city": "Seattle"})]),
        _task("2", [("modify_addr", {"address1": "123 Elm St"})]),
        _task("3", [("modify_addr", {"address1": "77 Pine Rd"})]),
        _task("4", [("modify_addr", {"address1": "123 Elm Street"})]),
    ]}
    sims = [
        # t1: 축약 복사(t17형) → fuzzy-|C|=1 치환이 gold 복구 = fix / city는 exact 무개입
        _sim([_user("change my address please"),
              _write("c1", "get", {}),  # 비-write는 events 미포함
              _tool("t1", {"address1": "123 Elm Street", "city": "Seattle"}),
              _write("c2", "modify_addr", {"address1": "123 Elm St", "city": "Seattle"}),
              _tool("c2", "ok")], "1"),
        # t2: 원래 gold와 일치하던 축약값을 치환이 어긋냄 = break
        _sim([_tool("t1", {"address1": "123 Elm Street"}),
              _write("c2", "modify_addr", {"address1": "123 Elm St"}),
              _tool("c2", "ok")], "2"),
        # t3: 빈 문자열 write(t39형)·|C|=1 → empty_fix
        _sim([_tool("t1", {"address1": "77 Pine Rd"}),
              _write("c2", "modify_addr", {"address1": ""}),
              _tool("c2", "ok")], "3"),
        # t4: fuzzy 후보 2개 → 무개입(multi_cand·DISAMB 관할)
        _sim([_tool("t1", {"a": {"address1": "123 Elm Street"}, "b": {"address1": "123 Elm Str"}}),
              _write("c2", "modify_addr", {"address1": "123 Elm St"}),
              _tool("c2", "ok")], "4"),
    ]
    buf = io.StringIO()
    tot, rows = run_gverb(sims, tasks, W, fields, out=buf)
    ok("gverb: fix==1", tot["fix"] == 1)
    ok("gverb: break==1", tot["break"] == 1)
    ok("gverb: empty_fix==1", tot["empty_fix"] == 1)
    ok("gverb: multi_cand==1", tot["multi_cand"] == 1)
    ok("gverb: exact(city)==1", tot["exact"] == 1)
    ok("gverb: rows have fix verdict", any(r[-1] == "fix" for r in rows))
    # 에러 write는 미집계
    tot2, _ = run_gverb([_sim([_tool("t1", {"address1": "77 Pine Rd"}),
                               _write("c2", "modify_addr", {"address1": "77 Pine Road"}),
                               _tool("c2", "denied", error=True)], "3")], tasks, W, fields, out=io.StringIO())
    ok("gverb: errored write skipped", tot2["fix"] + tot2["break"] + tot2["neutral"] == 0)

    # 6) addr e2e: 문맥-실재(|C|=2) vs 미조회(누락)
    tasks_a = {t["id"]: t for t in [
        _task("10", [("modify_addr", {"address1": "500 Oak Avenue", "city": "Dallas"})]),
        _task("11", [("modify_addr", {"address1": "1 Hidden Way", "city": "Reno"})]),
    ]}
    sims_a = [
        # t10: 오답 주소 write·gold 주소는 문맥 실재·레코드 2개 → present·|C|=2
        _sim([_tool("t1", {"orders": [
                  {"address1": "500 Oak Avenue", "city": "Dallas"},
                  {"address1": "9 Old Rd", "city": "Seattle"}]}),
              _write("c2", "modify_addr", {"address1": "9 Old Rd", "city": "Seattle"}),
              _tool("c2", "ok")], "10"),
        # t11: 주소 write 누락·gold 주소 미조회 → missing·not_found
        _sim([_user("please fix my address"),
              _tool("t1", {"orders": [{"address1": "9 Old Rd", "city": "Seattle"}]})], "11"),
    ]
    buf = io.StringIO()
    tot_a, rows_a = run_addr(sims_a, tasks_a, W, fields, out=buf)
    ok("addr: cases==2 (wrong+missing)", tot_a["cases"] == 2 and tot_a["case_wrong"] == 1
       and tot_a["case_missing"] == 1)
    ok("addr: present==1 not_found==1", tot_a["present"] == 1 and tot_a["not_found"] == 1)
    ok("addr: |C|>=2 on present case", tot_a["present_C_ge2"] == 1)
    ok("addr: missing row is not_found",
       any(r[2] == "missing" and r[4] == "not_found" for r in rows_a))

    if fails:
        print("SELFTEST FAILED: %s" % fails)
        sys.exit(1)
    print("ALL PASS (%d checks)" % 24)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lever", choices=["gverb", "addr"])
    ap.add_argument("--results", help="tau2 results json(.gz)")
    ap.add_argument("--tasks", help="tasks.json")
    ap.add_argument("--a2", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "a2", "retail.gate.json"),
                    help="write 도구 목록 유래(confirm gates)")
    ap.add_argument("--fields", default=DEFAULT_FIELDS,
                    help="대상 필드(csv·도메인 리터럴은 여기만)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.lever and a.results and a.tasks):
        ap.error("--lever/--results/--tasks required (or --selftest)")
    fields = [f.strip() for f in a.fields.split(",") if f.strip()]
    a2 = load_json(a.a2)
    write_tools = {t for g in a2["gates"] if g.get("kind") == "confirm"
                   for t in g.get("applies_to", [])}
    tasks = {str(t["id"]): t for t in load_json(a.tasks)}
    sims = load_json(a.results)["simulations"]
    if a.lever == "gverb":
        run_gverb(sims, tasks, write_tools, fields)
    else:
        run_addr(sims, tasks, write_tools, fields)


if __name__ == "__main__":
    main()
