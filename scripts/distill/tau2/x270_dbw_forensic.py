# -*- coding: utf-8 -*-
r"""x270 — 2026-08-12 판정 런 전수 포렌식 (사용자 지시: *"pass 통계 알려주고, 실패 경우 모두
정밀 포렌식하라"*). 집계에서 결론으로 직행하지 않는다([[08]]).

대상: bank_dbw_on_20260812 · bank_dbw_off_20260812 · bank_batch4_20260812 (각 nt=4).

sim 마다 뽑는 것:
  · 종료 사유(termination_reason) · 턴 수 · reward
  · 실패한 칸(action_id) 과 그 칸의 도구
  · **우리 층이 무엇을 말했나**: 서브 산출(DOCDECIDE)·유예 발화·거부 수
  · **모델이 무엇을 썼나**: open_bank_account 인자(type/class) 전수
  · 공식 명칭 집합 소속 여부 (A3 doc_index 주어의 기계 전개 — 판정만·선별 0)

실행(리모트): python3 x270_dbw_forensic.py [태그...]
"""
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SIMS = os.environ.get("X270_SIMDIR", "/home/woori/scratch/tau2-bench/data/simulations")
LOGS = os.environ.get("X270_LOGS", "/home/woori/scratch/logs")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = ["bank_dbw_on_20260812", "bank_dbw_off_20260812", "bank_batch4_20260812"]


def official_names():
    """A3 doc_index 주어 → 공식 명칭 집합. 기계 전개뿐(엔진 선별 0·[[59]])."""
    p = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
    a = json.load(io.open(p, encoding="utf-8"))
    di = (a.get("policy_ontology") or {}).get("doc_index") or {}
    out = {}
    for g, subs in di.items():
        out[g] = {" ".join(w.capitalize() for w in str(k).split("_")) for k in subs}
    return out


def inner_args(tc):
    a = tc.get("arguments") or {}
    raw = a.get("arguments")
    try:
        return json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        return {}


def sim_log_marks(tag):
    """로그를 sim 태그별로 가르지 않는다 — 태그 단위 합계만(로그 sim 표기는 태스크명이라
    시행을 구분하지 못한다·handoff §7 경고)."""
    p = os.path.join(LOGS, tag + ".log")
    if not os.path.exists(p):
        return {}
    c = collections.Counter()
    for line in io.open(p, encoding="utf-8", errors="replace"):
        for k, pat in (("docdecide", "[T2_DOCDECIDE] →"),
                       ("hold", "[T2_DECIDE_BEFORE_WRITE] write 1턴 유예"),
                       ("search_fire", "[T2_SEARCH_AGENT] group="),
                       ("search_silent_now", "[T2_SEARCH_AGENT] now 미확정"),
                       ("search_done_all", "모두 처리됨 — 침묵"),
                       ("decide_any", "[T2_DECIDE_ANY]"),
                       ("route", "[T2_ROUTE]"),
                       ("generic_deny", "resolve the flagged call(s) first"),
                       ("arbitrate_supp", "identical demand suppressed")):
            if pat in line:
                c[k] += 1
    return c


def run(tag, names):
    p = os.path.join(SIMS, tag, "results.json")
    if not os.path.exists(p):
        print("== %s : 결과 없음" % tag)
        return
    d = json.load(io.open(p, encoding="utf-8"))
    marks = sim_log_marks(tag)
    sims = d["simulations"]
    bytask = collections.defaultdict(list)
    for s in sims:
        bytask[s["task_id"]].append(s)

    print("=" * 100)
    print("== %s  (sim %d)   로그 마커: %s" % (tag, len(sims), dict(marks) or "없음"))
    for t in sorted(bytask):
        ss = bytask[t]
        npass = sum(1 for s in ss if ((s.get("reward_info") or {}).get("reward") or 0) >= 1)
        print("\n  [%s] pass %d/%d" % (t, npass, len(ss)))
        for k, s in enumerate(ss):
            ri = s.get("reward_info") or {}
            rw = ri.get("reward") or 0
            term = (s.get("termination_reason") or s.get("info") or {})
            if isinstance(term, dict):
                term = term.get("termination_reason") or term.get("end_reason") or "?"
            fails = [(a.get("action") or {}).get("action_id")
                     for a in (ri.get("action_checks") or []) if not a.get("action_match")]
            msgs = s.get("messages") or []
            opens = []
            for m in msgs:
                for tc in (m.get("tool_calls") or []):
                    if "open_bank_account" in json.dumps(tc.get("arguments") or {}):
                        ia = inner_args(tc)
                        cls = str(ia.get("account_class"))
                        typ = str(ia.get("account_type"))
                        grp = ("business_checking_accounts" if typ == "business_checking"
                               else "business_savings_accounts" if typ == "business_savings"
                               else None)
                        member = (cls in names.get(grp, set())) if grp else None
                        opens.append((typ, cls, member))
            ntools = sum(len(m.get("tool_calls") or []) for m in msgs)
            print("    trial%d reward=%.2f 메시지 %d 도구 %d 종료=%s"
                  % (k, rw, len(msgs), ntools, str(term)[:34]))
            if fails:
                print("      실패 칸: %s" % ",".join(x for x in fails if x))
            if opens:
                for typ, cls, member in opens:
                    flag = "✅집합內" if member else ("❌집합外" if member is False else "· 축미상")
                    print("      write  type=%-18s class=%-30r %s" % (typ, cls, flag))
            else:
                print("      write  **호출 없음**")


def main():
    tags = sys.argv[1:] or DEFAULT
    names = official_names()
    print("공식 명칭 집합 (A3 doc_index 기계 전개):")
    for g in sorted(names):
        if "business" in g:
            print("  %-28s %s" % (g, sorted(names[g])))
    for t in tags:
        run(t, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
