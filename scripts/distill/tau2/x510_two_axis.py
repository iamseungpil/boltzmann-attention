# -*- coding: utf-8 -*-
"""x510 - 원인의 2축 결합: **per-step(스텝 축)** x **per-task(시나리오 축)**  (2026-08-24 사용자 지시).

사용자 지시(취지 축자):
  "pass 에 대한 원인이 계속 표류한다. per step 별 원인분석과 per task 별 원인분석이 둘 다
   표류한다. 둘 다 동시에 봐야 한다 - 태스크별 실제 시나리오 차이도 있고, 부르는 도구들의
   실제 per step 별 레버 차이도 있기 때문이다. 2축으로 원인을 나눠서 다시 분석하라."

왜 한 축만 보면 표류하는가 (이 프로브가 재는 것):
  · per-task 단독 = 050 처럼 런마다 기전이 옮겨 다닌다(x494 헤더).
  · per-step 단독 = 같은 도구/필드가 **어떤 태스크에서는 성공**하는데 그 대비가 접힌다.
    x506 은 스텝을 축 6개로 접었지만, **그 축이 태스크를 가로질러 같게 행동하는지**는 안 쟀다.
  => 여기서 재는 것은 딱 하나: **축의 태스크-불변성**.
       스텝-일반    그 도구/필드를 가진 모든 태스크에서 같은 비율로 실패한다 -> 레버는 도메인 일반
       태스크-조건부  어떤 태스크에선 성공하고 어떤 태스크에선 실패한다      -> 원인은 시나리오 조건
    두 칸은 처방이 다르다([[70]]: 조건부 발화 · 단 조건은 도메인 일반 닫힌 술어여야 한다).

계기: 정본만 - `t2_forensic.mutation_diff`(변이 집합·[[69]]) + 사이드카(`F.sidecar`) 원문.
      새 분류 0([[48]]·[[31]] 규칙 (1)). 새 런 0. gold 는 **어느 필드가 갈렸나 세는 진단**으로만([[23]]).
축표: 결손 축 (1)~(7) 은 `x509_axis_queue_2026_08_24.json` 정본을 **인용**한다. 재유도 금지([[74]]).

실행: PYTHONIOENCODING=utf-8 python x510_two_axis.py [tag ...]
"""
import collections
import glob
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F          # noqa: E402

SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")

# 필드 -> 결손 축 : x506 answer_3_six_axes + x508 corrected_marginal 축자 인용
FIELD_AXIS = {
    "amount": "1_금액", "amount_difference": "1_금액", "partial_refund_amount": "1_금액",
    "account_class": "2_범주", "card_type": "2_범주",
    "issue_noticed_date": "3_손님권위", "discovery_date": "3_손님권위",
    "address": "3_손님권위", "delivery_option": "3_손님권위",
    "resolution_requested": "3_손님권위",
    "eligible_for_provisional_credit": "4_자격",
    "expected_apy": "5_파생", "actual_apy": "5_파생",
    "account_id": "6_전사", "transaction_id": "6_전사", "card_id": "6_전사",
    "card_last_4_digits": "6_전사",
}
VERIFY_TOOLS = ("log_verification",)
HARD0 = ["016", "040", "055", "057", "063", "072", "074", "079", "085", "094"]
UNSTABLE8 = ["003", "004", "017", "033", "050", "073", "093", "098"]


def short(t):
    return (t or "").replace("task_", "")


def field_diff(done_args, gold_args):
    """같은 도구를 성공시켰는데 인자가 갈린 자리 - 필드 이름만 돌려준다(판단 0·[[59]])."""
    out = []
    keys = set(done_args or {}) | set(gold_args or {})
    for k in sorted(keys):
        a, b = (done_args or {}).get(k), (gold_args or {}).get(k)
        if F.norm_args({k: a}) != F.norm_args({k: b}):
            out.append(k)
    return out


def best_gold(name, args, golds):
    """gold 후보가 여럿이면 필드 불일치가 가장 적은 것과 짝짓는다(x506 caveat 3 과 같은 규칙)."""
    cands = [g for g in golds if g.get("name") == name]
    if not cands:
        return None, []
    scored = sorted(((len(field_diff(args, g.get("args"))), i, g)
                     for i, g in enumerate(cands)), key=lambda x: (x[0], x[1]))
    g = scored[0][2]
    return g, field_diff(args, g.get("args"))


def main(argv):
    pats = argv or ["bank_t7348_*", "bank_t7346_*", "bank_t7336_*"]
    files = []
    for p in pats:
        for suf in (".results.json.gz", "_results.json.gz"):
            files += sorted(glob.glob(os.path.join(SIMS, p + suf)))
    files = sorted(set(f for f in files if "smoke" not in os.path.basename(f)))
    if not files:
        print("결과 파일 없음: %s" % pats)
        return 1
    MUT = F.mutating_tools()
    print("결과 파일 %d개: %s" % (len(files),
                              ", ".join(os.path.basename(f).split(".")[0] for f in files)))

    tool_cell = collections.defaultdict(lambda: {"sims": 0, "reach": 0, "exec": 0,
                                                 "blocked": 0, "wrong": 0, "match": 0,
                                                 "deny_ours": 0, "runs": set()})
    axis_bad = collections.Counter()           # (task, axis) -> WRONGARG 필드 건수
    axis_ok = collections.Counter()            # (task, axis) -> 같은 축 필드가 맞은 건수
    task_sims = collections.Counter()
    task_pass = collections.Counter()
    per_sim = []

    for fp in files:
        run = os.path.basename(fp).split(".")[0]
        try:
            d = json.load(gzip.open(fp, "rt", encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for s in (d.get("simulations") or []):
            t = short(s.get("task_id"))
            if not t:
                continue
            try:
                md = F.mutation_diff(s, MUT)
                tried = F.attempted_mutations(s, MUT)
            except Exception:
                continue
            gold = md.get("gold") or []
            if not gold:
                continue
            task_sims[t] += 1
            rw = (s.get("reward_info") or {}).get("reward")
            if rw:
                task_pass[t] += 1
            gold_tools = set(g.get("name") for g in gold)
            wrong_names = set(e.get("name") for e in (md.get("wrongarg") or []))
            matched_names = set(e.get("name") for e in (md.get("matched") or []))
            for tool in gold_tools:
                c = tool_cell[(t, tool)]
                c["sims"] += 1
                c["runs"].add(run)
                mine = [x for x in tried
                        if (x.get("inner") or x.get("outer")) == tool
                        or tool in str(x.get("key") or "")]
                if mine:
                    c["reach"] += 1
                    if any(x.get("ok") for x in mine):
                        c["exec"] += 1
                    else:
                        c["blocked"] += 1
                        if any(str(x.get("deny") or "") == "ours" for x in mine):
                            c["deny_ours"] += 1
                if tool in wrong_names:
                    c["wrong"] += 1
                # ★엄격 정의: 그 sim 에서 이 도구가 **한 번도 갈리지 않고** 맞았을 때만 match.
                #   한 sim 이 같은 도구를 여러 번 부르면 matched 와 wrongarg 가 공존한다 -
                #   느슨히 세면 073 이 4/6 맞은 것처럼 보이지만 실제 무결은 1/6 이다.
                if tool in matched_names and tool not in wrong_names:
                    c["match"] += 1
            for e in (md.get("wrongarg") or []):
                g, diffs = best_gold(e.get("name"), e.get("args"), gold)
                for f_ in diffs:
                    axis_bad[(t, FIELD_AXIS.get(f_, "0_기타"))] += 1
            for e in (md.get("matched") or []):
                # ★`log_verification` 제외 (2026-08-24 자기검산):
                #   그 도구의 `address` 는 손님에게 물어야 하는 값이 아니라 **검증 대상 필드**다.
                #   포함시키면 (3)손님권위 열에 "무결 13 태스크" 라는 유령이 생긴다 - 실물은
                #   전부 `log_verification|address` 한 자리였다. 축 귀속은 **쓰기 도구**에만.
                if e.get("name") in VERIFY_TOOLS:
                    continue
                for f_ in (e.get("args") or {}):
                    ax = FIELD_AXIS.get(f_)
                    if ax:
                        axis_ok[(t, ax)] += 1
            miss_names = set(e.get("name") for e in (md.get("missing") or []))
            outc = {}
            for tool in gold_tools:
                if tool in matched_names and tool not in wrong_names:
                    outc[tool] = "match"
                elif tool in wrong_names:
                    outc[tool] = "wrong"
                elif tool in miss_names:
                    outc[tool] = "miss"
                else:
                    outc[tool] = "?"
            per_sim.append({"run": run, "task": t, "simtag": F.simtag(s), "reward": rw,
                            "miss": [e.get("aid") for e in (md.get("missing") or [])],
                            "wrong": sorted(wrong_names), "tool_outcome": outc})

    # (A) 축의 태스크-불변성
    by_tool = collections.defaultdict(dict)
    for (t, tool), c in tool_cell.items():
        by_tool[tool][t] = c
    rows = []
    for tool, per_t in by_tool.items():
        if len(per_t) < 2:
            continue
        rates = {}
        for t, c in per_t.items():
            n = c["sims"] or 1
            rates[t] = (n - c["match"]) / n
        hi, lo = max(rates.values()), min(rates.values())
        rows.append({"tool": tool, "tasks": rates, "spread": hi - lo,
                     "verdict": ("스텝-일반" if (hi - lo) <= 0.34 and lo >= 0.5
                                 else "태스크-조건부" if (hi - lo) >= 0.5
                                 else "혼합"),
                     "sims": sum(c["sims"] for c in per_t.values())})
    rows.sort(key=lambda r: (-r["spread"], -r["sims"]))

    print("")
    print("=" * 108)
    print("(A) 축의 태스크-불변성 - 같은 도구가 태스크를 가로질러 같게 실패하나 (2축 결합의 본론)")
    print("=" * 108)
    print("%-40s %6s %7s  %-14s %s" % ("tool (2+ 태스크의 gold)", "sims", "격차", "판정", "태스크별 실패율"))
    print("-" * 108)
    for r in rows:
        pt = " · ".join("%s %.0f%%(%d)" % (t, 100 * v, by_tool[r["tool"]][t]["sims"])
                        for t, v in sorted(r["tasks"].items(), key=lambda kv: -kv[1]))
        print("%-40s %6d %6.0f%%  %-14s %s"
              % (r["tool"][:40], r["sims"], 100 * r["spread"], r["verdict"], pt))

    # (B) 2축 표
    axes = sorted({a for (_, a) in list(axis_bad) + list(axis_ok)})
    tasks = [t for t in HARD0 + UNSTABLE8 if task_sims.get(t)]
    other = [t for t in sorted(task_sims) if t not in tasks]
    print("")
    print("=" * 108)
    print("(B) 2축 표 - 행=태스크(시나리오) · 열=결손 축(스텝) · 값=갈린 필드/그 축 전체")
    print("=" * 108)
    print("%-6s %5s %5s  %s" % ("task", "sims", "pass", " ".join("%-11s" % a for a in axes)))
    print("-" * 108)
    for t in tasks + other:
        cells = []
        for a in axes:
            bad, ok = axis_bad.get((t, a), 0), axis_ok.get((t, a), 0)
            cells.append("%-11s" % (("%d/%d" % (bad, bad + ok)) if (bad or ok) else "·"))
        mark = "H" if t in HARD0 else ("U" if t in UNSTABLE8 else " ")
        print("%-5s%1s %5d %5d  %s" % (t, mark, task_sims[t], task_pass[t], " ".join(cells)))

    # (C) 열 판정
    print("")
    print("=" * 108)
    print("(C) 열 판정 - 그 축이 '스텝 축(도메인 일반)'인가 '태스크 조건'인가")
    print("=" * 108)
    col = []
    for a in axes:
        have = [t for t in (tasks + other) if (axis_bad.get((t, a), 0) or axis_ok.get((t, a), 0))]
        fail = [t for t in have if axis_bad.get((t, a), 0)]
        clean = [t for t in have if not axis_bad.get((t, a), 0)]
        col.append({"axis": a, "tasks_with": have, "tasks_failing": fail, "tasks_clean": clean})
        print("%-11s 쓰는 태스크 %2d · 실패 %2d · 무결 %2d   실패=%s   무결=%s"
              % (a, len(have), len(fail), len(clean), ",".join(fail), ",".join(clean) or "-"))
    print("")
    print("판독: 무결 태스크가 있으면 그 축은 **능력 경계가 아니라 시나리오 조건**이다 -")
    print("      같은 필드를 어떤 시나리오에선 맞춘다. 처방은 조건부이고 조건은 도메인 일반")
    print("      닫힌 술어여야 한다([[70]]). 무결 0 이면 스텝 축(도메인 일반 레버).")

    # (D) ★per-step 레버 대조 - 같은 도구가 어떤 태스크/sim 에서는 맞는다. 그 자리에서
    #     우리 레버는 무엇이 달랐나. 사이드카가 우리 층 발화의 권위다(t2_forensic §사이드카).
    #     ⚠조인키 simtag(task#seed) 는 nt>1 의 trial 을 병합한다 - 발화의 유무만 읽고
    #       건수를 인과로 읽지 마라([[08]]).
    import re as _re
    MARK = _re.compile(r"\[([A-Z][A-Z0-9_\-]{2,})\]")
    sc_cache = {}

    def sc_for(tag):
        if tag not in sc_cache:
            try:
                sc_cache[tag] = F.sidecar(tag)
            except Exception:
                sc_cache[tag] = {}
        return sc_cache[tag]

    def signals(rec):
        rows_ = sc_for(rec["run"]).get(rec["simtag"]) or []
        sig = collections.Counter()
        for o in rows_:
            k = o.get("kind") or "?"
            sig["kind:" + k] += 1
            if o.get("cp2_tag"):
                sig["cp2:" + str(o["cp2_tag"])] += 1
            if k in ("tool-deny", "reminder-user"):
                for m in MARK.findall(o.get("text") or "")[:4]:
                    sig["mark:" + m] += 1
            if k == "route":
                sig["route:%s" % ("arrived" if o.get("arrived") else "lost")] += 1
                if o.get("folded"):
                    sig["route:folded"] += 1
        return sig, len(rows_)

    print("")
    print("=" * 108)
    print("(D) per-step 레버 대조 - 같은 도구, 맞은 sim vs 갈린 sim 에서 우리 층이 무엇을 했나")
    print("=" * 108)
    contrast = []
    focus = [r["tool"] for r in rows if r["spread"] >= 0.5] + \
            [r["tool"] for r in rows if r["verdict"] == "스텝-일반"]
    for tool in focus:
        groups = collections.defaultdict(list)
        for rec in per_sim:
            o = (rec.get("tool_outcome") or {}).get(tool)
            if o:
                groups[("match" if o == "match" else "fail")].append(rec)
        if not groups.get("match") or not groups.get("fail"):
            note = "대조 불가(%s 만 있음)" % (",".join(sorted(groups)) or "없음")
            print("")
            print("· %-42s %s" % (tool[:42], note))
            contrast.append({"tool": tool, "note": note,
                             "match_tasks": sorted(set(r["task"] for r in groups.get("match", []))),
                             "fail_tasks": sorted(set(r["task"] for r in groups.get("fail", [])))})
            continue
        agg = {}
        for g in ("match", "fail"):
            tot = collections.Counter()
            have = 0
            for rec in groups[g]:
                sig, n = signals(rec)
                if n:
                    have += 1
                for k in sig:
                    tot[k] += 1          # sim 수로 센다(건수 아님)
            agg[g] = {"n": len(groups[g]), "with_sidecar": have, "sig": tot,
                      "tasks": collections.Counter(r["task"] for r in groups[g])}
        keys = set(agg["match"]["sig"]) | set(agg["fail"]["sig"])
        deltas = []
        for k in keys:
            fm = agg["match"]["sig"][k] / max(agg["match"]["with_sidecar"], 1)
            ff = agg["fail"]["sig"][k] / max(agg["fail"]["with_sidecar"], 1)
            if abs(fm - ff) >= 0.34:
                deltas.append((abs(fm - ff), k, fm, ff))
        deltas.sort(reverse=True)
        print("")
        print("· %-42s 맞음 %d sim(%s) ↔ 갈림 %d sim(%s)"
              % (tool[:42], agg["match"]["n"],
                 ",".join("%s:%d" % kv for kv in agg["match"]["tasks"].most_common()),
                 agg["fail"]["n"],
                 ",".join("%s:%d" % kv for kv in agg["fail"]["tasks"].most_common())))
        if not deltas:
            print("    우리 층 발화 차이 없음(임계 0.34) - 레버는 양쪽에 같이 떴다")
        for _, k, fm, ff in deltas[:8]:
            print("    %-34s 맞음 %.0f%% ↔ 갈림 %.0f%%" % (k, 100 * fm, 100 * ff))
        contrast.append({"tool": tool,
                         "match": {"n": agg["match"]["n"], "tasks": dict(agg["match"]["tasks"]),
                                   "with_sidecar": agg["match"]["with_sidecar"]},
                         "fail": {"n": agg["fail"]["n"], "tasks": dict(agg["fail"]["tasks"]),
                                  "with_sidecar": agg["fail"]["with_sidecar"]},
                         "deltas": [{"signal": k, "match_rate": fm, "fail_rate": ff}
                                    for _, k, fm, ff in deltas]})
    print("")
    print("판독: 발화 차이가 없는데 결과가 갈리면 원인은 **레버가 아니라 시나리오**다(태스크 축).")
    print("      발화가 갈림 쪽에서만 낮으면 전달 결손(부하) - 우리 층이다([[55]]).")

    # (E) ★per-step 레버 프로파일 - 그 쓰기 호출이 나가기 **전에** 우리 층이 무엇을 말했나.
    #     (D) 는 sim 전체를 셌다. 스텝 주장을 하려면 순서를 봐야 한다.
    #     ⚠사이드카 `turn` 과 궤적 `msg_i` 는 다른 계기다. 범위는 비슷하지만(실측: turn 최대가
    #       messages 길이보다 조금 작다) **동일 눈금이 아니다** - 선후만 읽고 거리를 읽지 마라.
    print("")
    print("=" * 108)
    print("(E) per-step 레버 프로파일 - 쓰기 호출 **직전까지** 뜬 우리 층 표지 (sim 비율)")
    print("=" * 108)
    prof = collections.defaultdict(lambda: {"n": 0, "marks": collections.Counter()})
    for fp in files:
        run = os.path.basename(fp).split(".")[0]
        try:
            d = json.load(gzip.open(fp, "rt", encoding="utf-8", errors="replace"))
        except Exception:
            continue
        sc = sc_for(run)
        for s in (d.get("simulations") or []):
            t = short(s.get("task_id"))
            rows_ = sc.get(F.simtag(s)) or []
            if not rows_:
                continue
            try:
                md = F.mutation_diff(s, MUT)
            except Exception:
                continue
            for grp, items in (("갈림", md.get("wrongarg") or []), ("맞음", md.get("matched") or [])):
                for e in items:
                    tool = e.get("name")
                    if tool in VERIFY_TOOLS:
                        continue
                    mi = e.get("msg_i")
                    if mi is None:
                        continue
                    key = (t, tool, grp)
                    p = prof[key]
                    p["n"] += 1
                    seen = set()
                    for o in rows_:
                        if (o.get("turn") or 0) >= mi:
                            continue
                        if o.get("cp2_tag"):
                            seen.add("cp2:" + str(o["cp2_tag"]))
                        if (o.get("kind") or "") in ("tool-deny", "reminder-user"):
                            for m in MARK.findall(o.get("text") or "")[:4]:
                                seen.add(m)
                    for m in seen:
                        p["marks"][m] += 1
    print("%-6s %-38s %-5s %4s  %s" % ("task", "tool", "판정", "n", "직전까지 뜬 표지(비율 60%+)"))
    print("-" * 108)
    prof_out = []
    for (t, tool, grp), p in sorted(prof.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        top = [(m, c / p["n"]) for m, c in p["marks"].most_common() if c / p["n"] >= 0.6]
        prof_out.append({"task": t, "tool": tool, "verdict": grp, "n": p["n"],
                         "marks": {m: round(r, 3) for m, r in top}})
        print("%-6s %-38s %-5s %4d  %s"
              % (t, tool[:38], grp, p["n"],
                 " · ".join("%s %.0f%%" % (m, 100 * r) for m, r in top[:6]) or "-"))
    print("")
    print("판독: 같은 도구인데 '맞음' 행과 '갈림' 행의 표지가 다르면 그것이 **per-step 레버 차이**다.")
    print("      두 행이 같으면 레버는 무죄이고 차이는 시나리오에 있다(태스크 축).")

    FINDINGS = [
        "F1 스텝 축은 표류하지 않는다 - hard-0 의 쓰기 도구는 태스크가 달라도 거의 같은 비율로"
        " 갈린다: `open_bank_account_4821` 055/057/063 = 100%/100%/100% (격차 0) ·"
        " `deposit_check_3847` 055/057 = 100%/100% (격차 0) ·"
        " `apply_checking_account_credit_5829` 072/074/073 = 100%/100%/83% (격차 17%).",
        "F2 표류는 **태스크 축**에 있다 - 같은 필드 축이 어떤 시나리오에선 맞는다:"
        " (6)전사 는 8 태스크가 쓰는데 4 실패(040·055·079·085) 4 무결(072·017·073·093) ·"
        " (2)범주 는 5 중 024 만 무결. 이 둘은 능력 경계가 아니라 시나리오 조건이다.",
        "F3 ★(1)금액 축에서 우리 레버는 무죄다 - 같은 도구·같은 태스크에서 **맞은 호출과 갈린"
        " 호출의 직전 표지 프로파일이 동일**하다(072 갈림 11 ↔ 맞음 3 · 073 갈림 17 ↔ 맞음 8,"
        " 여섯 표지 전부 100%/100%). 발화 커버리지로는 이 축이 안 갈린다 -> [[62]](1) 대로"
        " 남은 물음은 격리에서의 **효능**이다(x509 S2 가 그대로 유효).",
        "F4 ★한 sim 안에서 스텝이 갈린다 - 079 는 동일 궤적에서 close/freeze/unfreeze 3 스텝이"
        " 맞고 `order_debit_card_5739` 만 갈리는데 직전 표지는 **여섯 개 모두 동일**하다."
        " 즉 원인은 레버 상태가 아니라 그 스텝이 요구하는 필드(`delivery_option` = 손님에게"
        " 물어야 하는 값)다. per-task 귀속으로는 이 대비가 보이지 않는다.",
        "F5 (3)손님권위 열은 3 태스크(040·079·085)뿐이다 - 어제 표에서 '무결 13' 로 보였던 것은"
        " `log_verification|address` 한 자리가 만든 유령이었다(이 프로브에서 제외·자기검산).",
        "F6 유일하게 레버 프로파일이 갈리는 자리 = `apply_for_credit_card`."
        " 맞음 13 sim 중 PROVENANCE 69% ↔ 갈림 4 sim 중 0%, 갈림 쪽엔 ISOLATED-FORMALIZATION"
        " 75%·ORDER 75%·TOOL-CHANNEL 50%·BLOCKED 50% 가 쌓인다. n 이 작고 인과 방향이"
        " 양쪽이므로([[08]]) 가설로만 둔다.",
    ]
    CAVEATS = [
        "조인키 `simtag`(task#seed) 는 nt>1 의 trial 을 병합한다 - 발화의 **유무**만 읽고"
        " 건수를 인과로 읽지 마라.",
        "사이드카 `turn` 과 궤적 `msg_i` 는 다른 계기다. 선후만 읽고 거리를 읽지 마라.",
        "t7348 halfB 는 14/20 진행분이라 074·079·085 의 분모가 작다(x506 과 같은 한계).",
        "gold 는 **어느 필드가 갈렸나 세는 진단**으로만 썼다([[23]]). 성적은 reward 다([[69]]).",
        "축 (1)~(7) 과 태스크별 한계 수치는 `x509_axis_queue_2026_08_24.json` 을 인용한 것이지"
        " 여기서 재유도한 것이 아니다([[74]]).",
    ]
    print("")
    print("=" * 108)
    print("결론")
    print("=" * 108)
    for s_ in FINDINGS:
        print("  " + s_)
    print("")
    for s_ in CAVEATS:
        print("  ⚠ " + s_)

    out = {"probe": "x510_two_axis", "date": "2026-08-24",
           "findings": FINDINGS, "caveats": CAVEATS,
           "lever_contrast": contrast,
           "step_lever_profile": prof_out,
           "files": [os.path.basename(f).split(".")[0] for f in files],
           "tool_invariance": [dict(r, tasks=r["tasks"]) for r in rows],
           "axis_task_matrix": {"%s|%s" % k: {"bad": v, "ok": axis_ok.get(k, 0)}
                                for k, v in axis_bad.items()},
           "axis_ok_only": {"%s|%s" % k: v for k, v in axis_ok.items() if not axis_bad.get(k)},
           "columns": col,
           "task_sims": dict(task_sims), "task_pass": dict(task_pass),
           "per_sim": per_sim}
    dst = os.path.join(OUT, "x510_two_axis_2026_08_24.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
