#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E-ISO — 정보-맞춘 3단 격리 replay (SCAFFOLD_ENDGAME §L0 · 무료·로컬 32B).

질문: semantic 잔여(WRONG_REF_ORDER/ITEMS/PAYMENT)가 능력인가 부하인가 (§1.5 Q2 재실행).
방법: prov arm 실패 sim의 '첫 오기(誤記) write' 결정점마다, 에이전트가 실제 가진 정보로 3프로브:
  A 궤적-재현 : 결정점까지 전체 대화+도구출력 전사 -> 다음 write 결정 요청 (p_traj 재현)
  B 격리-원문 : 같은 정보(user 발화+도구 출력)만 잡음 없이 -> 같은 요청 (궤적-간섭 제거)
  C 격리-형식화: 문제-인자의 후보를 결정론 열거 + user 발화 -> 선택 요청 (형식화-부하 제거)
채점: 문제-인자(diff-arg)의 gold 값 일치(리스트=집합 비교). 정책 텍스트는 3프로브 공통(정보-맞춤).
판정: B>>A=궤적-간섭 부하 / B~A & C>>B=형식화-부하 / C도 낮음=능력·경계.

usage: ecomp_iso_probe.py --mode smoke|full [--port 8140] [--limit 999] [--conc 3]
"""
import argparse, gzip, json, re, sys, threading, queue
import urllib.request

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"
TASKS = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail/tasks.json"
POLICY = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail/policy.md"
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
WRITE_PAT = ("modify", "exchange", "return", "cancel")
ARG_PRIORITY = ["order_id", "item_ids", "new_item_ids", "payment_method_id",
                "address1", "address2", "city", "state", "zip", "country"]


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


def gold_writes(task):
    return [(x.get("name"), args_of(x.get("arguments")))
            for x in ((task.get("evaluation_criteria") or {}).get("actions") or [])
            if x.get("requestor", "assistant") == "assistant"
            and any(w in (x.get("name") or "") for w in WRITE_PAT)]


def first_wrong_write(sim, task):
    """(probe_index, tool_name, wrong_args, best_gold_args, diff_arg) or None.
    ★probe_index = min(첫 오기 write, 잘못된 값의 최초 assistant 언급) — **오염-전** 문맥.
    (v2: fail-분기서 write 직전은 이미 오제안→user 동의로 고착된 뒤라 gold-회복이 불공정.
     t61형 기전 = 대화-경로 오염 고착. 프로브는 첫 오염 이전에 잰다.)"""
    gb = {}
    for nm, ar in gold_writes(task):
        gb.setdefault(nm, []).append(ar)
    msgs = sim.get("messages") or []
    res_by_id = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
    found = None
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name") or ""
            if not any(w in nm for w in WRITE_PAT):
                continue
            tm = res_by_id.get(tc.get("id"))
            if tm is None or tm.get("error"):
                continue  # 성공 write만 (env-거부는 별개)
            ar = args_of(tc.get("arguments"))
            cands = gb.get(nm)
            if not cands:
                continue
            best = min(cands, key=lambda g: sum(1 for k in set(g) | set(ar)
                                                if str(g.get(k)) != str(ar.get(k))))
            dk = [k for k in ARG_PRIORITY if k in (set(best) | set(ar))
                  and str(best.get(k)) != str(ar.get(k))]
            if dk:
                found = (i, nm, ar, best, dk[0])
                break
        if found:
            break
    if not found:
        return None
    i, nm, ar, best, darg = found
    # 오염-전 인덱스: 잘못된 값(gold에 없는 wrong-marker)의 최초 assistant 언급 지점
    wv = ar.get(darg)
    gv = best.get(darg)
    gset = {str(x) for x in (gv if isinstance(gv, list) else [gv])}
    markers = [str(x) for x in (wv if isinstance(wv, list) else [wv]) if str(x) not in gset and len(str(x)) >= 4]
    probe_i = i
    if markers:
        for j, m in enumerate(msgs[:i]):
            if m.get("role") != "assistant":
                continue
            blob = (m.get("content") or "") if isinstance(m.get("content"), str) else ""
            for tc in (m.get("tool_calls") or []):
                blob += json.dumps(args_of(tc.get("arguments")), ensure_ascii=False)
            if any(mk in blob for mk in markers):
                probe_i = j
                break
    return probe_i, nm, ar, best, darg


def transcript(msgs, upto, mode):
    """mode='full'=역할 전부 / 'info'=user 발화+도구 출력만."""
    out = []
    for m in msgs[:upto]:
        r, c = m.get("role"), m.get("content")
        if r == "user" and isinstance(c, str) and c.strip():
            out.append("CUSTOMER: " + c.strip())
        elif r == "assistant" and mode == "full":
            if isinstance(c, str) and c.strip():
                out.append("AGENT: " + c.strip())
            for tc in (m.get("tool_calls") or []):
                out.append("AGENT->TOOL: %s(%s)" % (tc.get("name"),
                           json.dumps(args_of(tc.get("arguments")), ensure_ascii=False)))
        elif r == "tool" and isinstance(c, str) and c.strip():
            tag = "TOOL RESULT" if mode == "full" else "RECORD (fetched from database)"
            out.append("%s: %s" % (tag, c.strip()))
    t = "\n".join(out)
    return t[-80000:]


NUMID = re.compile(r"\b\d{7,}\b|#W\d+|\b(?:gift_card|credit_card|paypal)_\d+\b")


def candidates_for(msgs, upto, arg, wrong_val, gold_val):
    """도구 출력에서 arg-형(型) 후보 열거 (결정론·기조회 정보만)."""
    toks = set()
    for m in msgs[:upto]:
        if m.get("role") != "tool":
            continue
        c = m.get("content")
        if isinstance(c, str):
            toks |= set(NUMID.findall(c))
    def sig(v):
        v = str(v)
        if v.startswith("#"):
            return "order"
        if any(v.startswith(p) for p in ("gift_card", "credit_card", "paypal")):
            return "pay"
        return "num"
    want = sig(gold_val if not isinstance(gold_val, list) else (gold_val[0] if gold_val else ""))
    return sorted(t for t in toks if sig(t) == want)[:24]


def build_probes(sim, task, point, policy):
    i, nm, wrong, gold, darg = point
    msgs = sim.get("messages") or []
    gv = gold.get(darg)
    ex = json.dumps({darg: (["<id>", "..."] if isinstance(gv, list) else "<value>")})
    ask = ("\n\nYou are the retail agent at this exact point. You will eventually call `%s` "
           "for what the customer asked. Decide the CORRECT value of the argument `%s` "
           "(consult the records above and the policy). Respond with ONLY a JSON object "
           "exactly of this shape: %s (no prose, no other keys)." % (nm, darg, ex))
    A = transcript(msgs, i, "full") + ask
    B = transcript(msgs, i, "info") + ask
    cands = candidates_for(msgs, i, darg, wrong.get(darg), gv)
    C = (transcript(msgs, i, "info")
         + "\n\nYou must now call `%s`. For the argument `%s`, the ONLY possible values "
           "(from the records above) are:\n" % (nm, darg)
         + "\n".join("- " + c for c in cands)
         + "\n\nBased on what the customer asked for and the policy, choose the correct value(s) of `%s`. "
           "Respond with ONLY a JSON object: {\"%s\": <value or list>}." % (darg, darg))
    sysmsg = "You are a customer service agent for an online retail store. Follow this policy strictly:\n\n" + policy
    return {"A": (sysmsg, A), "B": (sysmsg, B), "C": (sysmsg, C)}, gv, cands


def call_llm(port, sysmsg, usermsg):
    body = json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": 300,
                       "messages": [{"role": "system", "content": sysmsg},
                                    {"role": "user", "content": usermsg}]}).encode()
    req = urllib.request.Request("http://localhost:%d/v1/chat/completions" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def judge(resp, darg, gold_val):
    m = re.search(r"\{.*\}", resp, re.S)
    got = None
    if m:
        try:
            got = json.loads(m.group(0)).get(darg)
        except Exception:
            got = None
    def norm(v):
        if isinstance(v, list):
            return tuple(sorted(str(x) for x in v))
        return str(v)
    if got is not None:
        return norm(got) == norm(gold_val), got
    # JSON 실패 시 문자열 포함 폴백 (리스트는 전원 포함)
    vals = gold_val if isinstance(gold_val, list) else [gold_val]
    return all(str(v) in resp for v in vals), resp[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--limit", type=int, default=999)
    ap.add_argument("--conc", type=int, default=3)
    ap.add_argument("--dump", default="/home/woori/scratch/eiso_results.jsonl")
    a = ap.parse_args()

    policy = open(POLICY, encoding="utf-8").read()
    tasks = {str(t["id"]): t for t in load_json(TASKS)}
    sims = load_json(SIM + "prov_e2e_retail_t4.results.json.gz")["simulations"]
    rows = [json.loads(l) for l in open("/home/woori/scratch/ecomp_census_prov.jsonl", encoding="utf-8")]
    targets = [r for r in rows if r["bucket"] in ("WRONG_REF_ORDER", "WRONG_ITEMS", "WRONG_PAYMENT")]
    if a.mode == "smoke":
        # 버킷별 2개 (t61 포함 보장)
        pick, seen = [], {"WRONG_REF_ORDER": 0, "WRONG_ITEMS": 0, "WRONG_PAYMENT": 0}
        for r in sorted(targets, key=lambda x: (x["bucket"], x["task_id"] != "61")):
            if seen[r["bucket"]] < 2:
                pick.append(r)
                seen[r["bucket"]] += 1
        targets = pick
    targets = targets[:a.limit]
    idx = {(str(s["task_id"]), s.get("trial")): s for s in sims}

    work = queue.Queue()
    for r in targets:
        work.put(r)
    out_lock = threading.Lock()
    results = []

    def worker():
        while True:
            try:
                r = work.get_nowait()
            except queue.Empty:
                return
            sim = idx.get((r["task_id"], r["trial"]))
            task = tasks.get(r["task_id"])
            if sim is None or task is None:
                continue
            point = first_wrong_write(sim, task)
            if point is None:
                with out_lock:
                    print("SKIP t%s/%s no-wrong-write" % (r["task_id"], r["trial"]), flush=True)
                continue
            probes, gv, cands = build_probes(sim, task, point, policy)
            # gold 정보가 probe 시점 문맥(도구출력)에 실재했나 — 부재면 선택 실패가 아니라
            # gather-선행 실패(조립)로 별도 버킷 (PREINFO·정직 분리)
            _ctx_tools = " ".join((m.get("content") or "") for m in (sim.get("messages") or [])[:point[0]]
                                  if m.get("role") == "tool" and isinstance(m.get("content"), str))
            _gvals = gv if isinstance(gv, list) else [gv]
            gold_in_ctx = all(str(v) in _ctx_tools for v in _gvals if v is not None)
            rec = {"task": r["task_id"], "trial": r["trial"], "bucket": r["bucket"],
                   "tool": point[1], "arg": point[4], "gold": gv, "gold_in_ctx": gold_in_ctx,
                   "wrote": point[2].get(point[4]), "ncand": len(cands), "probe_i": point[0]}
            for label in ("A", "B", "C"):
                sysmsg, um = probes[label]
                if label == "C" and (not cands or not any(
                        str(v) in cands for v in (gv if isinstance(gv, list) else [gv]))):
                    rec[label] = None  # gold가 후보 열거에 없음 = C 미적용(정직)
                    continue
                try:
                    resp = call_llm(a.port, sysmsg, um)
                    ok, got = judge(resp, point[4], gv)
                    rec[label] = bool(ok)
                    rec[label + "_got"] = str(got)[:100]
                except Exception as e:
                    rec[label] = None
                    rec[label + "_err"] = str(e)[:80]
            with out_lock:
                results.append(rec)
                print("EISO t%s/%s %s arg=%s A=%s B=%s C=%s gold=%s wrote=%s"
                      % (rec["task"], rec["trial"], rec["bucket"], rec["arg"],
                         rec.get("A"), rec.get("B"), rec.get("C"),
                         str(gv)[:40], str(rec["wrote"])[:40]), flush=True)

    threads = [threading.Thread(target=worker) for _ in range(a.conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open(a.dump, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    agg = {}
    for rec in results:
        key = (rec["bucket"], "info" if rec.get("gold_in_ctx") else "PREINFO")
        b = agg.setdefault(key, {"n": 0, "A": [0, 0], "B": [0, 0], "C": [0, 0]})
        b["n"] += 1
        for L in ("A", "B", "C"):
            if rec.get(L) is not None:
                b[L][1] += 1
                b[L][0] += 1 if rec[L] else 0
    print("\n== EISO 집계 (정답률·n=판정가능·PREINFO=gold 미조회 시점=gather-선행 실패) ==")
    for k, v in sorted(agg.items()):
        line = "%-17s %-8s n=%d" % (k[0], k[1], v["n"])
        for L in ("A", "B", "C"):
            c, n = v[L]
            line += "  %s=%s(%d/%d)" % (L, ("%.2f" % (c / n)) if n else "NA", c, n)
        print(line)
    print("EISO_DONE")


if __name__ == "__main__":
    main()
