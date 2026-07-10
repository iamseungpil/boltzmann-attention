#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""per-step 분산 census — "user-sim 노이즈" 주장의 해부 (2026-07-10·무료).

질문: flaky flip이 진짜 user-sim 문제인가, 에이전트 대처(brittleness) 문제인가?
방법: 같은 태스크의 pass-trial(참조) vs fail-trial을 스텝 정렬:
  (1) 최초 분기 주체 — user가 먼저 다른 발화? assistant가 동일-prefix에서 다른 행동?
      (assistant-first = vLLM 배칭 비결정/seed 미고정 = *우리* 비결정성, user-sim 아님)
  (2) fail-분기의 user 사실-토큰(ids·수치·옵션어) 집합이 pass-분기와 다른가?
      같음 → 같은 사실·표현만 다름인데 결과 뒤집힘 = **agent brittleness(대처 실패)**
      다름 → user-sim이 실제로 다른 정보를 냄(단, 에이전트 질문이 달라서일 수도 = 상류 확인 필요)
  (3) fail-분기 최종 write와 gold의 diff-인자 (무엇이 틀렸나)

usage: ecomp_divergence_census.py --results <gz> --tasks <tasks.json> [--task_ids 1,3,5,...]
"""
import argparse, gzip, json, re
from collections import Counter

WRITE_PAT = ("modify", "exchange", "return", "cancel")


def load_json(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def norm_msg(m):
    """비교용 정규화: (role, content-strip, tool_calls (name,args))"""
    tcs = []
    for tc in (m.get("tool_calls") or []):
        a = tc.get("arguments")
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except Exception:
                pass
        tcs.append((tc.get("name"), json.dumps(a, sort_keys=True, ensure_ascii=False)))
    c = m.get("content")
    return (m.get("role"), (c.strip() if isinstance(c, str) else ""), tuple(tcs))


FACT_RE = re.compile(r"#W\d+|\b\d{6,}\b|\$\d[\d,.]*|\b(?:small|medium|large|red|blue|black|white|green|yellow|pink|purple|gold|silver)\b", re.I)


def fact_set(msgs, role="user"):
    toks = set()
    for m in msgs:
        if m.get("role") != role:
            continue
        c = m.get("content")
        if isinstance(c, str):
            toks |= {t.lower() for t in FACT_RE.findall(c)}
    return toks


def final_writes(msgs):
    out = []
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name") or ""
            if any(w in nm for w in WRITE_PAT):
                a = tc.get("arguments")
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except Exception:
                        a = {}
                out.append((nm, a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--task_ids", default=None)
    a = ap.parse_args()

    tasks = {str(t["id"]): t for t in load_json(a.tasks)}
    sims = load_json(a.results)["simulations"]
    by_task = {}
    for s in sims:
        by_task.setdefault(str(s["task_id"]), []).append(s)

    tids = [t.strip() for t in a.task_ids.split(",")] if a.task_ids else sorted(by_task, key=int)
    init_ctr = Counter()
    fact_ctr = Counter()

    for tid in tids:
        group = by_task.get(tid) or []
        def ok(s):
            ri = s.get("reward_info") or {}
            dc = ri.get("db_check")
            return bool(dc.get("db_match")) if isinstance(dc, dict) else (ri.get("reward") or 0) >= 1
        passes = [s for s in group if ok(s)]
        fails = [s for s in group if not ok(s)]
        if not passes or not fails:
            continue
        ref = passes[0]
        rmsgs = ref.get("messages") or []
        for f in fails:
            fmsgs = f.get("messages") or []
            div_i, div_role = None, None
            for i in range(min(len(rmsgs), len(fmsgs))):
                if norm_msg(rmsgs[i]) != norm_msg(fmsgs[i]):
                    div_i = i
                    div_role = fmsgs[i].get("role")
                    break
            if div_i is None:
                div_i, div_role = min(len(rmsgs), len(fmsgs)), "length"
            init_ctr[div_role] += 1
            # user 사실-토큰 비교 (전 대화)
            fs_ref, fs_fail = fact_set(rmsgs), fact_set(fmsgs)
            same_facts = fs_ref == fs_fail
            fact_ctr["same_facts" if same_facts else "diff_facts"] += 1
            # gold diff (fail의 write)
            gold = [(x.get("name"), x.get("arguments")) for x in
                    ((tasks.get(tid, {}).get("evaluation_criteria") or {}).get("actions") or [])
                    if x.get("requestor", "assistant") == "assistant" and any(w in (x.get("name") or "") for w in WRITE_PAT)]
            fw = final_writes(fmsgs)
            wrong = []
            gold_by = {}
            for nm, ar in gold:
                gold_by.setdefault(nm, []).append(ar if isinstance(ar, dict) else {})
            for nm, ar in fw:
                cands = gold_by.get(nm) or []
                if not cands:
                    wrong.append((nm, "EXTRA"))
                    continue
                best = min(cands, key=lambda g: sum(1 for k in set(g) | set(ar) if str(g.get(k)) != str(ar.get(k))))
                dk = [k for k in set(best) | set(ar) if str(best.get(k)) != str(ar.get(k))]
                if dk:
                    wrong.append((nm, ",".join(sorted(dk))))
            missing = sum(len(v) for v in gold_by.values()) - sum(1 for nm, _ in fw if nm in gold_by)
            print("t%-4s trial%s div@%d by=%-9s facts=%s wrong=%s missing=%d"
                  % (tid, f.get("trial"), div_i, div_role,
                     "SAME" if same_facts else ("ref-only:" + ",".join(sorted(fs_ref - fs_fail))[:60]
                                                + "|fail-only:" + ",".join(sorted(fs_fail - fs_ref))[:60]),
                     wrong[:3], max(missing, 0)))
            # 분기점 두 변형 (정독용·170자)
            if div_i < min(len(rmsgs), len(fmsgs)):
                rv, fv = rmsgs[div_i], fmsgs[div_i]
                def brief(m):
                    c = m.get("content")
                    t = ";".join((tc.get("name") or "") for tc in (m.get("tool_calls") or []))
                    return ((c or "")[:150].replace("\n", " ") + ("|tc:" + t if t else ""))
                print("      PASS@div:", brief(rv)[:170])
                print("      FAIL@div:", brief(fv)[:170])
    print("\n== 집계 ==")
    print("최초 분기 주체:", dict(init_ctr))
    print("user 사실-토큰:", dict(fact_ctr))


if __name__ == "__main__":
    main()
