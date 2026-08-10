# -*- coding: utf-8 -*-
r"""x232 — **97 태스크 전수 분류** (성격 × 근거원 × 실패 기전 × 손님 함정) · 유료 0 · 엔진 0.

## 왜 (사용자 지시 2026-08-10)

> *"추천계열 중에서도 오류 유형이 몇 가지로 나뉜다. 사용자의 거짓말이 의도적으로 섞인 것도
> 있다. 태스크의 주요 성격(카드조회·추천 등) · 오류 유형 · KB/DB 유형으로 구분하고, **지금
> 구조를 그대로 확장해서 해결할 수 있는 그룹**을 알려달라."*

우리는 지금까지 **추천 계열 4개**(098·099·100·010)만 팠다. 97 태스크 전체에서 그 구조가 어디까지
닿는지 모르면 이번 주 계획을 세울 수 없다.

## 축 (전부 **데이터에서** 나온다 — 손으로 붙이지 않는다)

  ⒜ **성격**   = gold 액션(쓰기 도구)의 종류. `evaluation_criteria.actions` 의 `name`.
  ⒝ **근거원** = `required_documents` 유무(KB 필요) · gold 인자가 DB 조회로만 닿는가.
  ⒞ **손님 함정** = `user_scenario.instructions` 의 자기-은폐/주장 지시
                   (*"ONLY MENTION … if asked"* · *"do not reveal"* · *"insist"* 류).
                   ⚠이건 **분석 스크립트**의 문자열 검사다 — 엔진은 이런 걸 하지 않는다([[59]]).
  ⒟ **이력**   = 지금까지 돈 **모든 런**의 그 태스크 pass/n (우리 구조가 이미 닿는가).
  ⒠ **실패 기전** = 실패 시행의 종료 사유 + 쓰기 유무(제출했는데 틀림 / 아예 안 씀 / 이관 / 폭주).

실행: python x232_task_taxonomy.py [출력.json]
"""
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

TASKS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/tasks.json"
SIMDIRS = ["/home/woori/scratch/tau2-bench/data/simulations/*/results.json",
           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../../reports/facet_rft_2026/sim_results/*.json.gz")]
# 손님이 **스스로 감추거나 주장하도록** 지시받은 흔적 (분석 전용)
TRAP = [("은폐", re.compile(r"only mention .{0,40}if (you are )?asked|do not (reveal|mention|say)"
                            r"|don't (reveal|mention|volunteer)", re.I)),
        ("주장", re.compile(r"\binsist\b|\bclaim\b|\bargue\b|push back|refuse to accept", re.I)),
        ("조건부", re.compile(r"\bonly if\b|\bunless\b|if and only if", re.I)),
        ("이탈", re.compile(r"take your business elsewhere|end the conversation|hang up", re.I))]


def load_tasks():
    return json.load(open(TASKS, encoding="utf-8"))


def gold_actions(t):
    ec = t.get("evaluation_criteria") or {}
    return [a.get("name") for a in (ec.get("actions") or []) if isinstance(a, dict)]


def history():
    """모든 런에서 태스크별 (pass, n) 과 실패 기전을 모은다."""
    agg = collections.defaultdict(lambda: {"n": 0, "pass": 0, "end": collections.Counter(),
                                           "wrote": 0, "fail_wrote": 0})
    seen = set()
    for pat in SIMDIRS:
        for p in sorted(glob.glob(pat)):
            try:
                f = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") \
                    else open(p, encoding="utf-8")
                d = json.load(f)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            for s in (d.get("simulations") or []):
                if not isinstance(s, dict):
                    continue
                key = s.get("id")
                if key in seen:
                    continue
                seen.add(key)
                a = agg[s.get("task_id")]
                ok = (s.get("reward_info") or {}).get("reward") == 1
                a["n"] += 1
                a["pass"] += 1 if ok else 0
                a["end"][s.get("termination_reason")] += 1
                wrote = any(((tc.get("function") or {}).get("name") or tc.get("name"))
                            not in (None,)
                            for m in (s.get("messages") or [])
                            for tc in (m.get("tool_calls") or [])
                            if m.get("role") == "user")
                a["wrote"] += 1 if wrote else 0
                if not ok and wrote:
                    a["fail_wrote"] += 1
    return agg


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "x232_taxonomy.json"
    tasks = load_tasks()
    hist = history()
    rows = []
    for t in tasks:
        tid = t.get("id")
        instr = ((t.get("user_scenario") or {}).get("instructions") or "")
        traps = [name for name, rx in TRAP if rx.search(instr)]
        docs = t.get("required_documents") or []
        acts = gold_actions(t)
        h = hist.get(tid) or {"n": 0, "pass": 0, "end": collections.Counter(),
                              "wrote": 0, "fail_wrote": 0}
        rows.append({"id": tid, "acts": acts, "n_acts": len(acts),
                     "kb": bool(docs), "n_docs": len(docs), "traps": traps,
                     "n": h["n"], "pass": h["pass"],
                     "rate": (h["pass"] / h["n"]) if h["n"] else None,
                     "fail_wrote": h["fail_wrote"],
                     "end": dict(h["end"])})
    json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def grp(key):
        g = collections.defaultdict(lambda: [0, 0, 0])      # tasks, pass, n
        for r in rows:
            k = key(r)
            g[k][0] += 1
            g[k][1] += r["pass"]
            g[k][2] += r["n"]
        return g

    print("### 97 태스크 전수 (이력 = 지금까지 돈 모든 sim)")
    print("\n[A] 성격 = gold 쓰기 도구")
    for k, (nt, ps, n) in sorted(grp(lambda r: " + ".join(sorted(set(r["acts"]))) or "(쓰기 없음)")
                                 .items(), key=lambda kv: -kv[1][0]):
        print("  %-46s 태스크 %2d · 이력 %4d sim · pass %s"
              % (k[:46], nt, n, ("%.0f%%" % (100.0 * ps / n)) if n else "-"))
    print("\n[B] 근거원")
    for k, (nt, ps, n) in sorted(grp(lambda r: "KB 필요(문서 %d)" % r["n_docs"] if r["kb"]
                                     else "DB 만").items(), key=lambda kv: -kv[1][0]):
        print("  %-46s 태스크 %2d · 이력 %4d sim · pass %s"
              % (k, nt, n, ("%.0f%%" % (100.0 * ps / n)) if n else "-"))
    print("\n[C] 손님 함정 (사용자 지시문에 심긴 것)")
    for k, (nt, ps, n) in sorted(grp(lambda r: "+".join(r["traps"]) or "(없음)").items(),
                                 key=lambda kv: -kv[1][0]):
        print("  %-46s 태스크 %2d · 이력 %4d sim · pass %s"
              % (k, nt, n, ("%.0f%%" % (100.0 * ps / n)) if n else "-"))
    print("\n[D] 이력 성적대")
    def band(r):
        if not r["n"]:
            return "미측정"
        x = r["rate"]
        return "0%" if x == 0 else ("<33%" if x < .33 else ("<67%" if x < .67 else "≥67%"))
    for k, (nt, ps, n) in sorted(grp(band).items(), key=lambda kv: -kv[1][0]):
        print("  %-46s 태스크 %2d · 이력 %4d sim · pass %s"
              % (k, nt, n, ("%.0f%%" % (100.0 * ps / n)) if n else "-"))
    print("\n[E] 실패했는데 **쓰기는 했다**(= 값이 틀렸다) 상위")
    for r in sorted(rows, key=lambda r: -r["fail_wrote"])[:15]:
        print("  %-9s 쓰고도 실패 %3d · 이력 %3d/%3d · KB %s · 함정 %s · %s"
              % (r["id"], r["fail_wrote"], r["pass"], r["n"], "O" if r["kb"] else "X",
                 ",".join(r["traps"]) or "-", " + ".join(sorted(set(r["acts"])))[:34]))
    print("\n저장: %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
