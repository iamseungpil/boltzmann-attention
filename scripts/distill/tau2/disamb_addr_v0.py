#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DISAMB-ADDR V0 — 주소-오답 write의 문맥-후보/단서 census (설계 §4·CENSUS_LEVERS_DESIGN_2026_07_11).

comp gz의 실패 sim 전수에서 "주소류 필드가 gold와 불일치한 *성공* write"를 자동 탐지하고,
각 케이스에 대해 write 시점 문맥만으로(실행 0·무료):
  (i)   |C|  = 완전한 주소 레코드 후보 수 (t2_gate_patch._candidate_records 동형 자체구현 —
        임포트 금지 지시. anchor=address1을 가진 dict 단위·필드튜플 dedup·write 이전 tool 출력만)
  (ii)  gold_in_C = gold 주소가 후보에 실재하는가 (문맥-실재 vs 미조회 분해)
  (iii) user_cue  = 사용자 발화(write 이전)에 gold의 도시/주가 등장하는가 (결정론 예비신호)
        + cue_uniq = 그 단서가 gold *만* 가리키는가 (다른 후보 레코드와 비공유 = 판별력)
  (+)   n_mism = gold 대비 불일치 주소필드 수 — 현행 P-B 치환은 *단일 인자* 제자리 치환이라
        n_mism>=2 케이스는 address1만 고쳐선 chimera(부분 수정)로 여전히 fail → 채택 판단 재료.

서브콜 실측은 상위(리모트) 세션 몫 — --subcall 모드(openai-호환 POST)만 만들어두고 기본=오프라인 표.

판정·요약: gold_in_C 케이스의 user_cue 일치율 = DISAMB가 이길 사전 확률.
disamb_sub_args 추가는 일치율>=60%일 때만 (설계 §4·지시서).

usage:
  py -3 disamb_addr_v0.py --results reports/facet_rft_2026/sim_results/comp_retail_t4.results.json.gz
  py -3 disamb_addr_v0.py --results <gz> --subcall --base http://HOST:PORT/v1 --model NAME
  py -3 disamb_addr_v0.py --selftest
"""
import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

DEFAULT_FIELDS = "address1,address2,city,state,zip,country"

# ---------- 공용 로드·정규화 (census_v0_gverb_addr 규약 동일) ----------


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


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(s):
    if s is None:
        return ""
    return " ".join(_PUNCT.sub(" ", str(s).lower()).split())


def word_in(needle, hay_norm):
    """norm된 텍스트에서 단어경계 포함 검사 (도시명 다단어 안전·주코드 'DC' 오매치 방지)."""
    n = norm(needle)
    if not n:
        return False
    return re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", hay_norm) is not None


# ---------- 궤적 스캔 ----------


def scan_sim(sim, write_tools):
    """messages 시간순 → (write events, user_parts, agent_parts, tool_jsons).
    event의 user_i/agent_i/out_i = 그 write *이전* 스냅샷 길이(결과 미포함)."""
    msgs = sim.get("messages") or []
    res_by_id = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
    user_parts, agent_parts, tool_jsons, events = [], [], [], []
    for m in msgs:
        role, c = m.get("role"), m.get("content")
        if role == "user" and isinstance(c, str):
            user_parts.append(c)
        if role == "assistant":
            if isinstance(c, str) and c.strip():
                agent_parts.append(c)
            for tc in (m.get("tool_calls") or []):
                nm = tc.get("name")
                if nm in write_tools:
                    tm = res_by_id.get(tc.get("id"))
                    ok = tm is not None and not tm.get("error")
                    events.append({"name": nm, "args": args_of(tc.get("arguments")), "ok": ok,
                                   "user_i": len(user_parts), "agent_i": len(agent_parts),
                                   "out_i": len(tool_jsons)})
        if role == "tool" and isinstance(c, str) and not m.get("error"):
            try:
                tool_jsons.append(json.loads(c))
            except Exception:
                try:  # lenient: augment 텍스트가 JSON 뒤에 붙는 arm(v25e류) 구제
                    obj, _ = json.JSONDecoder().raw_decode(c.lstrip())
                    tool_jsons.append(obj)
                except Exception:
                    pass
    return events, user_parts, agent_parts, tool_jsons


def gold_writes(task, write_tools):
    return [(x.get("name"), args_of(x.get("arguments")))
            for x in ((task.get("evaluation_criteria") or {}).get("actions") or [])
            if x.get("requestor", "assistant") == "assistant" and x.get("name") in write_tools]


def best_gold(args, cands):
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


# ---------- 후보 열거 (_candidate_records 동형·자체구현) ----------


def addr_records(objs, fields):
    """anchor=fields[0](address1)를 스칼라로 가진 dict 단위 수집·필드튜플 dedup(등장순 보존)."""
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


def rec_matches_gold(rec, gold, fields):
    """gold 주소가 이 후보 레코드인가: anchor(norm) 일치 + 양쪽에 있는 나머지 필드 norm 일치."""
    anchor = fields[0]
    if norm(rec.get(anchor)) != norm(gold.get(anchor)) or not norm(gold.get(anchor)):
        return False
    for f in fields[1:]:
        if f in rec and f in gold and norm(rec.get(f)) != norm(gold.get(f)):
            return False
    return True


# ---------- 케이스 census ----------


def collect_cases(sims, tasks, write_tools, fields):
    """실패 sim 전수 → 주소-오답 성공-write 케이스 목록(사전순 스캔·per-write)."""
    cases = []
    for s in iter_failed(sims):
        tid = str(s.get("task_id"))
        gold_by_tool = defaultdict(list)
        for nm, ar in gold_writes(tasks.get(tid) or {}, write_tools):
            gold_by_tool[nm].append(ar)
        events, user_parts, agent_parts, tool_jsons = scan_sim(s, write_tools)
        for ev in events:
            if not ev["ok"]:
                continue
            ar = ev["args"]
            if not any(f in ar for f in fields):
                continue  # 주소류 인자 없는 write
            g = best_gold(ar, gold_by_tool.get(ev["name"]))
            if g is None:
                continue  # 같은 도구 gold 부재 = 판정 불가(스코프 밖)
            mism = [f for f in fields if (f in g or f in ar)
                    and str(g.get(f)) != str(ar.get(f))]
            if not mism:
                continue  # 주소 일치 write
            recs = addr_records(tool_jsons[:ev["out_i"]], fields)
            gold_rec = next((r for r in recs if rec_matches_gold(r, g, fields)), None)
            user_txt = norm("\n".join(user_parts[:ev["user_i"]]))
            cue, cue_uniq = None, False
            for cf in ("city", "state"):
                cv = g.get(cf)
                if cv is not None and word_in(cv, user_txt):
                    cue = "%s:%s" % (cf, cv)
                    others = [r for r in recs
                              if not rec_matches_gold(r, g, fields)
                              and norm(r.get(cf)) == norm(cv)]
                    cue_uniq = not others
                    break
            cases.append({
                "task": tid, "trial": s.get("trial"), "tool": ev["name"],
                "n_cand": len(recs), "gold_in_C": gold_rec is not None,
                "user_cue": cue is not None, "cue": cue or "-", "cue_uniq": cue_uniq,
                "n_mism": len(mism), "mism": ",".join(mism),
                "written_a1": str(ar.get(fields[0])), "gold_a1": str(g.get(fields[0])),
                "_recs": recs, "_gold": g, "_args": ar,
                "_user_parts": user_parts[:ev["user_i"]],
                "_agent_parts": agent_parts[:ev["agent_i"]],
            })
    return cases


def report(cases, out=sys.stdout):
    print("== DISAMB-ADDR V0 (design §4) — 주소-오답 성공-write 전수 ==", file=out)
    print("%-5s %-5s %-30s %3s %-9s %-8s %-8s %6s  %-22s %s"
          % ("task", "trial", "tool", "|C|", "gold_in_C", "user_cue", "cue_uniq",
             "n_mism", "cue", "written_a1 -> gold_a1"), file=out)
    for c in cases:
        print("t%-4s %-5s %-30s %3d %-9s %-8s %-8s %6d  %-22s %r -> %r"
              % (c["task"], c["trial"], c["tool"], c["n_cand"],
                 c["gold_in_C"], c["user_cue"], c["cue_uniq"], c["n_mism"],
                 c["cue"], c["written_a1"], c["gold_a1"]), file=out)
    tot = Counter()
    tot["cases"] = len(cases)
    present = [c for c in cases if c["gold_in_C"]]
    tot["gold_in_C"] = len(present)
    tot["C_ge2"] = sum(1 for c in cases if c["n_cand"] >= 2)
    tot["present_C_ge2"] = sum(1 for c in present if c["n_cand"] >= 2)
    tot["present_cue"] = sum(1 for c in present if c["user_cue"])
    tot["present_cue_uniq"] = sum(1 for c in present if c["cue_uniq"])
    tot["present_multi_mism"] = sum(1 for c in present if c["n_mism"] >= 2)
    print("\n== summary ==", file=out)
    print("cases=%d | gold_in_C(문맥-실재)=%d not_in_C(미조회-원천→E-PLAN L2 관할)=%d | |C|>=2 전체=%d"
          % (tot["cases"], tot["gold_in_C"], tot["cases"] - tot["gold_in_C"], tot["C_ge2"]),
          file=out)
    if present:
        rate = 100.0 * tot["present_cue"] / len(present)
        rate_u = 100.0 * tot["present_cue_uniq"] / len(present)
        print("gold_in_C 케이스: |C|>=2(DISAMB 발화가능)=%d/%d | user_cue=%d/%d (%.0f%%) | "
              "cue_uniq(판별력)=%d/%d (%.0f%%) | n_mism>=2(단일-인자 치환으론 chimera)=%d/%d"
              % (tot["present_C_ge2"], len(present), tot["present_cue"], len(present), rate,
                 tot["present_cue_uniq"], len(present), rate_u,
                 tot["present_multi_mism"], len(present)), file=out)
        lit = "fires (%.0f%% >= 60%%)" % rate if rate >= 60 else "no (%.0f%% < 60%%)" % rate
        print("decision gate (지시서 리터럴·user_cue>=60%%): %s" % lit, file=out)
        vetoes = []
        if tot["present_cue_uniq"] == 0:
            vetoes.append("cue_uniq=0 — 도시/주 단서가 후보를 판별하지 못함(전 후보 동일 도시)")
        if tot["present_multi_mism"] == len(present):
            vetoes.append("전 케이스 n_mism>=2 — P-B 단일-인자 치환(address1만)으론 gold 도달 "
                          "불가(chimera·Δfix 기대 0)")
        vetoes.append("A2 토큰 주의: _key_tokens('address1')={'address1'} — 'address' 문자열 "
                      "추가는 dead config·발화하려면 'address1'/'address2' 필요")
        if rate >= 60 and (tot["present_cue_uniq"] == 0
                           or tot["present_multi_mism"] == len(present)):
            print("structural veto:", file=out)
            for v in vetoes:
                print("  - " + v, file=out)
            print("recommendation: DO-NOT-ADD (사전신호가 승리를 지지하지 않음 — 격리 서브콜 "
                  "실측(--subcall)으로만 재판정)", file=out)
        elif rate >= 60:
            print("recommendation: ADD ('address1'/'address2' 토큰으로·'address' 아님)", file=out)
        else:
            print("recommendation: DO-NOT-ADD", file=out)
    else:
        print("decision gate: 판정 불가(gold_in_C 케이스 0) → 미추가", file=out)
    return tot


# ---------- --subcall (격리 서브콜 실측·리모트 상위 세션용·기본 미사용) ----------

SUBCALL_SYS = (
    "You are resolving ONE ambiguous tool-call argument for a customer-service agent. "
    "Read the conversation transcript and the candidate address records, then decide "
    "which single candidate address the user actually intends."
)


def build_subcall_prompt(c, fields):
    """_t5c_disamb_subcall 동형 프롬프트(전사=user/agent 텍스트만·후보=레코드 JSON)."""
    parts = []
    ui = list(c["_user_parts"])
    ai = list(c["_agent_parts"])
    # 단순 교차 전사(정확한 turn 순서 아님·서브콜엔 충분): user/agent를 등장순 병합 근사
    trans = []
    while ui or ai:
        if ui:
            trans.append("User: " + ui.pop(0).strip())
        if ai:
            trans.append("Agent: " + ai.pop(0).strip())
    prompt = (SUBCALL_SYS + "\n\n=== Conversation ===\n" + "\n".join(trans)[-6000:]
              + "\n\n=== Candidate address records ===\n"
              + "\n".join("- %s" % json.dumps(r, ensure_ascii=False) for r in c["_recs"])
              + "\n\nThe agent currently chose '" + c["written_a1"] + "'. Which single "
              "candidate does the user intend? Answer with EXACTLY one candidate "
              + fields[0] + " value, or UNSURE.")
    return prompt


def parse_answer(txt, cands):
    """t2_gate_patch._parse_subcall_answer 동형: 전체=후보 1개 or 경계-유일 부분매치."""
    raw = (txt or "").strip()
    t = raw.strip().strip('"\'`.,;: \n\t')
    for c in cands:
        if t == str(c).strip():
            return c
    found = []
    for c in cands:
        cs = str(c).strip()
        idx = raw.find(cs)
        if idx >= 0:
            before = raw[idx - 1] if idx > 0 else " "
            after = raw[idx + len(cs)] if idx + len(cs) < len(raw) else " "
            if not (before.isalnum() or after.isalnum()):
                found.append(c)
    return found[0] if len(found) == 1 else None


def run_subcall(cases, fields, base, model, api_key=None, max_cases=0, out=sys.stdout):
    import urllib.request
    anchor = fields[0]
    todo = [c for c in cases if c["gold_in_C"] and c["n_cand"] >= 2]
    if max_cases:
        todo = todo[:max_cases]
    print("\n== subcall 실측 (%d cases · base=%s model=%s) ==" % (len(todo), base, model),
          file=out)
    n_gold = n_keep = n_unsure = 0
    for c in todo:
        prompt = build_subcall_prompt(c, fields)
        body = {"model": model, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": "Bearer " + api_key} if api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                ans_txt = json.load(r)["choices"][0]["message"]["content"]
        except Exception as e:
            print("t%-4s tr%-3s ERROR %r" % (c["task"], c["trial"], e), file=out)
            continue
        pick = parse_answer(ans_txt, [str(r.get(anchor)) for r in c["_recs"]])
        gold_a1 = c["gold_a1"]
        if pick is None:
            n_unsure += 1
            verdict = "UNSURE"
        elif norm(pick) == norm(gold_a1):
            n_gold += 1
            verdict = "GOLD"
        else:
            n_keep += 1
            verdict = "OTHER(%r)" % pick
        print("t%-4s tr%-3s |C|=%d cue=%-20s -> %s" % (c["task"], c["trial"], c["n_cand"],
                                                       c["cue"], verdict), file=out)
    tot = n_gold + n_keep + n_unsure
    if tot:
        print("subcall summary: gold=%d/%d (%.0f%%) other=%d unsure=%d"
              % (n_gold, tot, 100.0 * n_gold / tot, n_keep, n_unsure), file=out)


# ---------- selftest ----------


def _sim(msgs, tid, trial=0, reward=0.0):
    return {"task_id": tid, "trial": trial, "reward_info": {"reward": reward}, "messages": msgs}


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
    fields = [f.strip() for f in DEFAULT_FIELDS.split(",")]
    fails = []

    def ok(name, cond):
        print("  %-62s %s" % (name, "PASS" if cond else "FAIL"))
        if not cond:
            fails.append(name)

    W = {"modify_pending_order_address"}
    orders = {"orders": [
        {"address": {"address1": "500 Oak Avenue", "address2": "Apt 1",
                     "city": "Washington", "state": "DC", "zip": "20001", "country": "USA"}},
        {"address": {"address1": "9 Old Rd", "address2": "",
                     "city": "Dallas", "state": "TX", "zip": "75201", "country": "USA"}},
    ]}
    gold = {"order_id": "#W1", "address1": "500 Oak Avenue", "address2": "Apt 1",
            "city": "Washington", "state": "DC", "zip": "20001", "country": "USA"}
    tasks = {"10": _task("10", [("modify_pending_order_address", gold)]),
             "11": _task("11", [("modify_pending_order_address", gold)]),
             "12": _task("12", [("modify_pending_order_address", gold)])}
    wrong = dict(gold, address1="9 Old Rd", address2="", city="Dallas",
                 state="TX", zip="75201")
    sims = [
        # t10: 오답(Dallas 프로필류) write·gold DC 레코드 문맥 실재·사용자 단서 "Washington"
        _sim([_user("ship it to my Washington place please"),
              _tool("t1", orders),
              _write("c2", "modify_pending_order_address", wrong),
              _tool("c2", "ok")], "10"),
        # t11: gold 미조회(문맥에 Dallas 레코드만) → gold_in_C=False
        _sim([_user("use the address from one of my orders"),
              _tool("t1", {"orders": [orders["orders"][1]]}),
              _write("c2", "modify_pending_order_address", wrong),
              _tool("c2", "ok")], "11"),
        # t12: gold 실재·단서 비판별(두 후보 모두 Washington) → cue_uniq=False
        _sim([_user("ship to my Washington place"),
              _tool("t1", {"orders": [
                  dict(orders["orders"][0]["address"]),
                  {"address1": "77 Pine Rd", "address2": "", "city": "Washington",
                   "state": "DC", "zip": "20002", "country": "USA"}]}),
              _write("c2", "modify_pending_order_address",
                     dict(gold, address1="77 Pine Rd", zip="20002")),
              _tool("c2", "ok")], "12"),
        # 성공 sim은 스캔 제외
        _sim([_tool("t1", orders),
              _write("c2", "modify_pending_order_address", wrong),
              _tool("c2", "ok")], "10", trial=9, reward=1.0),
        # 에러 write 제외
        _sim([_tool("t1", orders),
              _write("c2", "modify_pending_order_address", wrong),
              _tool("c2", "denied", error=True)], "10", trial=8),
    ]
    cases = collect_cases(sims, tasks, W, fields)
    ok("cases: 실패 sim의 오답 write만 3건", len(cases) == 3)
    c10 = next(c for c in cases if c["task"] == "10")
    ok("t10 |C|=2", c10["n_cand"] == 2)
    ok("t10 gold_in_C=True", c10["gold_in_C"] is True)
    ok("t10 user_cue(city:Washington)=True·uniq", c10["user_cue"] and c10["cue_uniq"])
    ok("t10 n_mism>=2 (전필드 오복사)", c10["n_mism"] >= 2)
    c11 = next(c for c in cases if c["task"] == "11")
    ok("t11 gold_in_C=False (미조회-원천)", c11["gold_in_C"] is False)
    c12 = next(c for c in cases if c["task"] == "12")
    ok("t12 cue 비판별 → cue_uniq=False", c12["user_cue"] and not c12["cue_uniq"])
    # 후보/gold 매칭 단위
    recs = addr_records([orders], fields)
    ok("addr_records: dedup 2", len(recs) == 2)
    ok("rec_matches_gold: DC 레코드", rec_matches_gold(recs[0], gold, fields))
    ok("rec_matches_gold: Dallas 레코드 불일치", not rec_matches_gold(recs[1], gold, fields))
    ok("word_in: 'DC' 단어경계 (McDonald 오매치 방지)",
       word_in("DC", norm("my DC place")) and not word_in("DC", norm("mcdcburger")))
    # 서브콜 프롬프트·파서 단위 (POST 없이)
    p = build_subcall_prompt(c10, fields)
    ok("subcall prompt: 후보·현재선택 포함",
       "500 Oak Avenue" in p and "currently chose '9 Old Rd'" in p)
    ok("parse_answer: exact", parse_answer("500 Oak Avenue", ["500 Oak Avenue", "9 Old Rd"])
       == "500 Oak Avenue")
    ok("parse_answer: 경계-유일 부분매치",
       parse_answer("The user intends 500 Oak Avenue.", ["500 Oak Avenue", "9 Old Rd"])
       == "500 Oak Avenue")
    ok("parse_answer: UNSURE→None", parse_answer("UNSURE", ["500 Oak Avenue"]) is None)
    import io
    tot = report(cases, out=io.StringIO())
    ok("report: gold_in_C=2", tot["gold_in_C"] == 2)
    if fails:
        print("SELFTEST FAILED: %s" % fails)
        sys.exit(1)
    print("ALL PASS (%d checks)" % 18)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", help="tau2 results json(.gz) — tasks 블록 내장(comp gz)")
    ap.add_argument("--tasks", help="tasks.json (미지정=results의 tasks 블록)")
    ap.add_argument("--a2", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "a2", "retail.gate.json"))
    ap.add_argument("--fields", default=DEFAULT_FIELDS)
    ap.add_argument("--subcall", action="store_true", help="격리 서브콜 실측(리모트 상위 세션용)")
    ap.add_argument("--base", help="openai-호환 base url (예: http://HOST:PORT/v1)")
    ap.add_argument("--model", help="서브콜 모델명")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.results:
        ap.error("--results required (or --selftest)")
    fields = [f.strip() for f in a.fields.split(",") if f.strip()]
    a2 = load_json(a.a2)
    write_tools = {t for g in a2["gates"] if g.get("kind") == "confirm"
                   for t in g.get("applies_to", [])}
    d = load_json(a.results)
    tasks_list = load_json(a.tasks) if a.tasks else (d.get("tasks") or [])
    tasks = {str(t["id"]): t for t in tasks_list}
    cases = collect_cases(d["simulations"], tasks, write_tools, fields)
    report(cases)
    if a.subcall:
        if not (a.base and a.model):
            ap.error("--subcall requires --base and --model")
        run_subcall(cases, fields, a.base, a.model, a.api_key, a.max_cases)


if __name__ == "__main__":
    main()
