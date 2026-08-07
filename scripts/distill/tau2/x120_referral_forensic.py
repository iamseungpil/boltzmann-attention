# -*- coding: utf-8 -*-
"""101/102가 무엇 때문에 0점인가 — 채점 성분별로, 궤적에서 직접.

이 도구가 있는 이유: 이 두 태스크의 실패를 `db_match` 한 칸으로 추적해 왔는데, 그 칸은
채점의 일부일 뿐이다. 102의 `reward_basis`는 **['DB','NL_ASSERTION']**(권위본 =
`tasks/task_102.json`)이라 DB가 통과해도 NL이 0이면 reward는 0이다. 그래서 여기서는
**성분을 분리해** 찍는다. 또 `submit_referral`의 레코드 id는
`generate_referral_id(user_id, account_type)`로 **인자에만 의존**하므로, 같은 유형을 두 번
제출해도 DB는 변하지 않는다 — DB를 깨는 것은 제출 **횟수**가 아니라 **서로 다른 유형의 집합**이다.
그 구분이 되지 않으면 "너무 많이 제출했다"는 진단이 원인을 가리킨다는 보장이 없다.

출력:
  §A 채점 성분      reward / DB / NL(판정문+근거) / gold 액션 대조
  §B 제출 원장      submit_referral 호출을 순서대로 — 누가·무엇을·결과가 무엇이었는지
  §C DB 원인        제출된 **유형 집합** vs gold 유형 집합 (여집합이 곧 DB 불일치의 내용)
  §D 선행 읽기      log_verification / get_referrals_by_user 가 언제 성공했는가
  §E 결정론 후보    원장에서 기계로 나오는 값(9일 창 잔여·유형별 연간 잔여)을 여기서 계산하고,
                    에이전트가 그 값을 실제로 말했는지 궤적에서 대조한다. 둘이 어긋나면
                    그 판정은 **LLM에 남겨 둔 닫힌 술어**다.

usage: x120_referral_forensic.py --dirs bank_f1_gpu0,bank_m1_gpu0 [--tasks task_101,task_102] [--prose]
"""

import collections
import datetime
import glob
import gzip
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
TAU2 = os.environ.get("GO_TAU2", "/home/woori/scratch/tau2-bench")
DOMAIN = os.path.join(TAU2, "data", "tau2", "domains", "banking_knowledge")
SIMBASES = [os.path.join(TAU2, "data", "simulations"),
            os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")]


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


DIRS = [d for d in (arg("--dirs") or "").split(",") if d]
TASKS = [t for t in (arg("--tasks") or "task_101,task_102").split(",") if t]
PROSE = "--prose" in sys.argv


def jopen(p):
    op = gzip.open if p.endswith(".gz") else io.open
    with op(p, "rt", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def load_task(tid):
    """★권위본은 `tasks/<id>.json`이다.

    같은 id가 `tasks.json`에도 있는데 **내용이 다르다**(102: 그쪽은 nl_assertions 0건·
    reward_basis=['DB']). 먼저 찾은 쪽을 쓰는 도구는 그래서 조용히 틀린 gold를 찍는다.
    실행된 런의 채점 성분과 일치하는 쪽이 권위본이므로 `tasks/` 를 우선한다.
    """
    p = os.path.join(DOMAIN, "tasks", tid + ".json")
    if os.path.exists(p):
        d = jopen(p)
        return (d if d.get("id") else (d.get("tasks") or [d])[0]), p
    for q in glob.glob(os.path.join(DOMAIN, "tasks*.json")):
        d = jopen(q)
        for t in (d.get("tasks") if isinstance(d, dict) else d) or []:
            if t.get("id") == tid:
                return t, q
    return None, None


def load_sims():
    out = []
    for base in SIMBASES:
        for d in DIRS:
            for p in glob.glob(os.path.join(base, d, "results.json")) + \
                     glob.glob(os.path.join(base, d + ".results.json.gz")):
                try:
                    dd = jopen(p)
                except Exception as e:
                    print("  (읽기 실패 %s: %s)" % (p, e))
                    continue
                for s in dd.get("simulations") or []:
                    s["_src"] = d
                    out.append(s)
    return out


def short(x, n=200):
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True)
    s = re.sub(r"[ \t]+", " ", s.replace("\n", " ⏎ ")).strip()
    return s if len(s) <= n else s[:n] + "…"


def args_of(tc):
    a = tc.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def name_of(tc):
    return tc.get("name") or (tc.get("function") or {}).get("name") or "?"


def walk(sim):
    """(turn, role, msg) — assistant 턴 번호를 붙여서."""
    turn = 0
    for m in sim.get("messages") or []:
        if m.get("role") == "assistant":
            turn += 1
        yield turn, m.get("role"), m


def tool_result_for(sim, call_id):
    for m in sim.get("messages") or []:
        if m.get("role") == "tool" and (m.get("id") == call_id or m.get("tool_call_id") == call_id
                                        or m.get("requestor_tool_call_id") == call_id):
            return str(m.get("content") or "")
    return ""


# ── 정책 상수는 **읽어 온 문서에서** 오지 않는다. 여기서는 원장 계산만 하고,
#    한도 값은 궤적에 에이전트가 말한 값과 대조하기 위한 참고로만 쓴다(강제 아님).
WINDOW_DAYS = 9
WINDOW_MAX = 2


def parse_date(s):
    for f in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s, f).date()
        except Exception:
            pass
    return None


def ledger_facts(task):
    """시딩된 원장에서 **기계로 나오는 값**: 9일 창 안의 건수, 유형별 누적."""
    ini = ((task.get("initial_state") or {}).get("initialization_data") or {}).get("agent_data") or {}
    refs = ((ini.get("referrals") or {}).get("data") or {})
    rows = list(refs.values())
    dates = [(parse_date(r.get("date") or ""), r.get("referred_account_type")) for r in rows]
    dates = [(d, t) for d, t in dates if d]
    latest = max((d for d, _ in dates), default=None)
    return rows, dates, latest


def main():
    sims = load_sims()
    if not sims:
        print("sim 0건 — --dirs 를 확인하라 (base=%s)" % SIMBASES)
        return
    for tid in TASKS:
        task, tpath = load_task(tid)
        mine = [s for s in sims if s.get("task_id") == tid]
        if not mine:
            continue
        ec = (task or {}).get("evaluation_criteria") or {}
        gold_types = [a.get("arguments", {}).get("account_type")
                      for a in (ec.get("actions") or []) if a.get("name") == "submit_referral"]
        rows, dates, latest = ledger_facts(task or {})
        cnt = collections.Counter(t for _, t in dates)
        print("=" * 104)
        print("== %s ==  권위본=%s" % (tid, os.path.basename(tpath or "?")))
        print("   reward_basis = %s   gold 제출 유형 = %s" % (ec.get("reward_basis"), gold_types))
        for a in ec.get("nl_assertions") or []:
            print("   nl 요건: %s" % a)
        print("   시딩 원장 %d건 · 최근 일자 %s · 유형별 %s"
              % (len(rows), latest, dict(cnt)))
        for s in sorted(mine, key=lambda x: (x.get("_src"), x.get("trial") or 0)):
            ri = s.get("reward_info") or {}
            print("-" * 104)
            print("[%s trial %s] reward=%s  breakdown=%s  종료=%s  메시지=%d"
                  % (s["_src"], s.get("trial"), ri.get("reward"),
                     ri.get("reward_breakdown"), s.get("termination_reason"),
                     len(s.get("messages") or [])))
            db = ri.get("db_check") or {}
            print("  §A DB=%s" % db.get("db_match"))
            for a in ri.get("action_checks") or []:
                print("     %s %-24s %s" % ("✓" if a.get("action_match") else "✗",
                                            (a.get("action") or {}).get("name"),
                                            short((a.get("action") or {}).get("arguments"), 120)))
            for c in ri.get("nl_assertions") or []:
                print("     %s NL: %s" % ("✓" if c.get("met") else "✗", short(c.get("nl_assertion"), 160)))
                if c.get("justification"):
                    print("        판정근거: %s" % short(c.get("justification"), 700))

            # §B 제출 원장
            subs, reads = [], []
            for turn, role, m in walk(s):
                for tc in (m.get("tool_calls") or []):
                    if role not in ("assistant", "user"):
                        continue
                    nm = name_of(tc)
                    a = args_of(tc)
                    res = tool_result_for(s, tc.get("id"))
                    if nm == "submit_referral":
                        subs.append((turn, role, a.get("account_type"), res))
                    elif nm in ("log_verification", "get_referrals_by_user", "verify_identity",
                                "get_user_information_by_name", "get_user_information_by_email",
                                "get_user_information_by_id"):
                        reads.append((turn, role, nm, res))
            print("  §B 제출 원장 (%d회)" % len(subs))
            for turn, role, at, res in subs:
                ok = "실패" if res.startswith("Failed") else ("성공" if res else "?")
                print("     턴%-3d %-6s %-26s %s  %s" % (turn, "손님" if role == "user" else "에이전트",
                                                        at, ok, short(res, 90)))
            got = [at for _, _, at, res in subs if not res.startswith("Failed")]
            uniq = sorted(set(got))
            goldset = sorted(set(gold_types))
            extra = [t for t in uniq if t not in goldset]
            miss = [t for t in goldset if t not in uniq]
            print("  §C 유형 집합: 제출=%s / gold=%s" % (uniq, goldset))
            print("     초과=%s   누락=%s   ⇒ DB 예상=%s (실측 %s)"
                  % (extra or "없음", miss or "없음",
                     "일치" if (not extra and not miss) else "불일치", db.get("db_match")))
            print("  §D 선행 읽기")
            for turn, role, nm, res in reads:
                bad = res.startswith("Failed") or "NOT_VERIFIED" in res or "Error" in res[:40]
                print("     턴%-3d %-6s %-28s %s %s" % (turn, "손님" if role == "user" else "에이전트",
                                                        nm, "⚠" if bad else " ", short(res, 110)))

            # §E 에이전트가 원장 수치를 말했는가
            said = []
            for turn, role, m in walk(s):
                if role != "assistant":
                    continue
                txt = str(m.get("content") or "")
                if not txt:
                    continue
                for pat, what in ((r"\b9[- ]day\b|rolling window|9일", "9일 창"),
                                  (r"\b6\s*/\s*6\b|six of six|6 of 6", "Gold Years 6/6"),
                                  (r"\b7\s*/\s*8\b|7 of 8|seven of", "Sky Blue 7/8"),
                                  (r"\bNovember 10|11/10/2025|Nov(ember)? 10", "Nov 10 원장 사실"),
                                  (r"TechFlow", "TechFlow 언급"),
                                  (r"Ember", "Ember 언급"),
                                  (r"5 years|five years|2020", "Ember 5년 주장")):
                    if re.search(pat, txt, re.I):
                        said.append((turn, what, txt))
            seen = set()
            print("  §E 원장 수치를 에이전트가 말했는가")
            for turn, what, txt in said:
                if what in seen and not PROSE:
                    continue
                seen.add(what)
                print("     턴%-3d %-16s %s" % (turn, what, short(txt, 220)))
            for what in ("9일 창", "Gold Years 6/6", "Sky Blue 7/8", "Nov 10 원장 사실"):
                if what not in seen:
                    print("     (없음) %s — 원장에서 기계로 나오는 값인데 발화 0" % what)


if __name__ == "__main__":
    main()
