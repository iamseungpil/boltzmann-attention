# -*- coding: utf-8 -*-
r"""x486 — **task_098 은 무엇이 회귀시켰나** (2026-08-23·무료·오프라인·LLM 0).

사용자 지적 축자: *"98은 이전 수십 런에서 모두 pass 였다. 무엇이 회귀시켰는지 정확하게 다시
확인하라."* ⇒ 한 런의 차분(t7336↔t7346)으로는 부족하다. **전 런 전수**로 다음을 낸다.

  ① 098 의 런별 성적 전수 — 언제 처음 깨졌나(경계 런)
  ② 실패 sim 이 낸 `submit_referral.account_type` 값 — 무엇으로 갈렸나
  ③ 각 sim 의 우리-층 지표: 검색 서브가 **몇 번째 턴**에 답했나(`[T2_SEARCH_AGENT] group=`
     의 `turn=`), 우리 답(`[T2_DOCDECIDE] → …`)이 무엇이었나, 모델이 정박한 원값
     (`[T2_REDERIVE] raw='…'`)이 무엇이었나 — 전부 **영속 로그 축자**다.
  ④ 경계 런 앞뒤의 엔진 커밋 — 무엇이 그 사이에 들어왔나

판정은 하지 않는다 — 표만 낸다([[08]] 집계→결론 직행 금지). gold 는 `action_checks` 가
이미 담고 있는 것만 읽는다(따로 열지 않는다·[[23]]).
사본 0: 로딩·해제·액션 비교는 정본 `t2_forensic`([[67]]).
"""
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                      # noqa: E402

TASK = "task_098"
REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))

RE_TURN = re.compile(r"\[T2_SEARCH_AGENT\] group=(\S+).*?turn=(\d+)")
RE_DEC = re.compile(r"\[T2_DOCDECIDE\] → '([^']*)'")
RE_RED = re.compile(r"\[T2_REDERIVE\] raw='([^']*)'")


def log_facts(tag):
    """태그의 영속 로그에서 sim 별 우리-층 사실. 로그가 없으면 빈 dict(조용히 넘기지 않고 표시)."""
    try:
        txt = F.log_text(tag)
    except Exception:
        txt = ""
    if not txt:
        return {}
    out = collections.defaultdict(lambda: {"search_turns": [], "decided": [], "rederived": []})
    for ln in txt.splitlines():
        if TASK not in ln:
            continue
        m = re.search(r"\[sim=(task_098#s\d+)\]", ln)
        if not m:
            continue
        k = m.group(1)
        t = RE_TURN.search(ln)
        if t:
            out[k]["search_turns"].append((t.group(1), int(t.group(2))))
        d = RE_DEC.search(ln)
        if d:
            out[k]["decided"].append(d.group(1))
        r = RE_RED.search(ln)
        if r:
            out[k]["rederived"].append(r.group(1))
    return out


def submitted(sim):
    """궤적이 실제로 낸 `submit_referral` 인자 — 정본 해제를 거친다."""
    vals = []
    for t in F.trajectory_actions(sim):
        nm = t["inner"] or t["outer"]
        if nm == "submit_referral":
            vals.append((t["args"].get("account_type"), t["ok"]))
    return vals


def gold_type(sim):
    for g in F.gold_actions_flat(sim):
        if g["inner"] == "submit_referral" or g["outer"] == "submit_referral":
            return g["args"].get("account_type")
    return None


def main():
    rows = []
    for tag, sim in F.iter_all_sims(want_tasks=[TASK]):
        if sim.get("reward_info") is None:
            continue
        rows.append({"tag": tag, "key": F.simtag(sim) or "", "trial": sim.get("trial"),
                     "reward": (sim.get("reward_info") or {}).get("reward"),
                     "gold_type": gold_type(sim), "submitted": submitted(sim),
                     "n_calls": len(F.trajectory_actions(sim)),
                     "n_msgs": len(sim.get("messages") or [])})
    # 로그 사실은 태그당 한 번만 읽는다(대용량 gz).
    facts = {}
    for t in sorted({r["tag"] for r in rows}):
        facts[t] = log_facts(t)
    for r in rows:
        f = (facts.get(r["tag"]) or {}).get(r["key"]) or {}
        st = f.get("search_turns") or []
        r["first_search_turn"] = st[0][1] if st else None
        r["n_search"] = len(st)
        r["decided"] = f.get("decided") or []
        r["rederived"] = f.get("rederived") or []

    rows.sort(key=lambda r: (r["tag"], r["trial"] if r["trial"] is not None else 0))
    with io.open(os.path.join(REP, "x486_098_census.json"), "w", encoding="utf-8") as fp:
        json.dump({"task": TASK, "n": len(rows), "rows": rows}, fp, ensure_ascii=False, indent=1)

    P = [r for r in rows if r["reward"] == 1.0]
    print("== %s 전수 %d sim · pass %d · fail %d" % (TASK, len(rows), len(P), len(rows) - len(P)))
    print("\n== 런별 성적 (태그 정렬 = 대체로 시간순)")
    per = collections.OrderedDict()
    for r in rows:
        per.setdefault(r["tag"], []).append(r["reward"])
    for t, v in per.items():
        print("  %-34s %d/%d  %s" % (t[:34], sum(1 for x in v if x == 1.0), len(v),
                                     "★FAIL 포함" if any(x != 1.0 for x in v) else ""))

    print("\n== 제출된 account_type × 성적  (gold = %s)"
          % sorted({r["gold_type"] for r in rows if r["gold_type"]}))
    ct = collections.Counter()
    for r in rows:
        vals = tuple(sorted({v for v, _ok in r["submitted"] if v})) or ("(제출 0)",)
        ct[("PASS" if r["reward"] == 1.0 else "fail", vals)] += 1
    for (p, v), n in sorted(ct.items(), key=lambda kv: -kv[1]):
        print("  %-5s %-46s %d" % (p, ", ".join(v)[:46], n))

    print("\n== sim 별 (성적 · 제출값 · 검색서브 첫 턴 · 우리 답 · 모델 원값)")
    for r in rows:
        print("  %-30s %-5s sub=%-16s 검색첫턴=%-5s ×%-2s 우리답=%-26s 원값=%s"
              % (r["tag"][:30], "PASS" if r["reward"] == 1.0 else "fail",
                 ",".join(str(v) for v, _ in r["submitted"])[:16] or "-",
                 r["first_search_turn"], r["n_search"],
                 ",".join(r["decided"])[:26] or "-",
                 ",".join(r["rederived"])[:20] or "-"))

    print("\n== 검색서브 첫 턴 × 성적 (로그가 있는 sim 만)")
    band = collections.Counter()
    for r in rows:
        if r["first_search_turn"] is None:
            continue
        b = "turn≤8" if r["first_search_turn"] <= 8 else ("turn 9-19" if r["first_search_turn"] < 20
                                                          else "turn≥20")
        band[(b, "PASS" if r["reward"] == 1.0 else "fail")] += 1
    for k, n in sorted(band.items()):
        print("  %-10s %-5s %d" % (k[0], k[1], n))
    print("\n[JSON] %s" % os.path.join(REP, "x486_098_census.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
