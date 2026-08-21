# -*- coding: utf-8 -*-
r"""x458 — 1단계 남은 두 축의 **기전 확정** (2026-08-21·오프라인·LLM 0·무료)

## 왜
`STAGE1_FULL_RUN_PLAN_2026_08_21.md` §2 census 가 1단계 40 sim(6/40 pass)에서 두 블록을 남겼는데
둘 다 *"기전 미확정"* 으로 미뤄져 있었다. 사용자 지적(2026-08-21): *"지금 포렌식으로 원인 조사할
수 있지 않나?"* — **맞다.** 궤적은 영속돼 있고 이 조사는 유료 런도 GPU 도 필요 없다.

    ⒜ `log_verification` **DUP 11**  (016·040·079·085)  같은 호출을 왜 두 번 하나
    ⒝ `apply_checking_account_credit_5829` **MISSING 13** (072·073·074)  왜 아예 안 부르나

## 무엇을 묻나 (닫힌 술어만 · 판정은 인쇄로 남기고 결론은 사람이)
⒜ DUP:
    · 같은 `(name, args)` 가 몇 번, 몇 턴 간격으로 반복되나
    · **두 호출 사이에 무엇이 있었나** — 도구 결과 / 손님 발화 / **우리 층 발화**
    · 반복 직전에 우리가 무언가 말했나([[55]] 우리 배관 먼저 — `proc_fb` 死배선 선례)
⒝ MISSING:
    · 그 도구가 궤적에 **한 번도 안 나오나**, 아니면 시도했는데 실패(ERROR/deny)인가
    · 발견 래퍼(`unlock`/`call_discoverable_*`)를 거쳤나 — 즉 **도구를 못 찾은 것**인가
    · 종료 사유·턴 수(상한 소진인가) · 대신 무엇을 썼나
    · 에이전트가 **하겠다고 말했나**(마지막 발화) — 말하고 안 했으면 knowing-doing 축

⛔여기서 처방을 만들지 않는다. 기전만 확정하고 [[62]] 순서대로 격리로 넘긴다.

사용: py x458_dup_and_missing_forensic.py [--tags bank_t7328_halfA_20260819r,...]
"""
import argparse
import collections
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

import t2_forensic as F                 # noqa: E402  정본 로더(사본 금지·[[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)
DUP_TOOL = "log_verification"
MISS_TOOL = "apply_checking_account_credit_5829"
DISPUTE_TOOL = "file_credit_card_transaction_dispute_4829"


def _role(m):
    return str(getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "") or "")


def _content(m):
    c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
    return str(c or "")


def dup_forensic(sims):
    """같은 `(도구, 인자)` 반복 — 간격과 **사이에 있던 것**을 센다."""
    rows = []
    for s in sims:
        msgs = s.get("messages") or []
        seen = {}
        idx_of = {}
        for i, m in enumerate(msgs):
            for tc in (m.get("tool_calls") or []):
                nm = F.nameof(tc)
                if nm != DUP_TOOL:
                    continue
                key = F.label(nm, F.argsof(tc))
                if key in seen:
                    j = idx_of[key]
                    between = msgs[j + 1:i]
                    kinds = collections.Counter(_role(b) for b in between)
                    # 두 호출 사이의 **도구 결과 문면**(우리 층 문구가 섞여 오는 자리)
                    tool_txt = " ".join(_content(b)[:200] for b in between if _role(b) == "tool")
                    user_txt = " ".join(_content(b)[:200] for b in between if _role(b) == "user")
                    rows.append({"task": F.task_id(s), "sim": F.sim_key(s),
                                 "key": key[:70], "first": j, "second": i,
                                 "gap_msgs": i - j, "between": dict(kinds),
                                 "tool_between": tool_txt[:220], "user_between": user_txt[:220]})
                seen[key] = seen.get(key, 0) + 1
                idx_of[key] = i
    return rows


def missing_forensic(sims, tool):
    """그 도구가 왜 없나 — 시도조차 없나 · 발견을 거쳤나 · 무엇을 대신 했나."""
    rows = []
    for s in sims:
        labels = [F.label(F.nameof(tc), F.argsof(tc)) for _m, tc in F.calls(s)]
        names = [F.nameof(tc) for _m, tc in F.calls(s)]
        inner = []
        for _m, tc in F.calls(s):
            n = F.nameof(tc)
            if n in F.WRAPPERS:
                inner.append(F.inner_name(F.argsof(tc)) or "")
        hit_direct = sum(1 for n in names if n == tool)
        hit_inner = sum(1 for n in inner if n == tool)
        # 그 도구를 부른 메시지의 결과가 오류였나
        err = 0
        for i, m in enumerate(s.get("messages") or []):
            for tc in (m.get("tool_calls") or []):
                if F.nameof(tc) == tool or F.inner_name(F.argsof(tc)) == tool:
                    for b in (s.get("messages") or [])[i + 1:i + 4]:
                        if _role(b) == "tool" and (b.get("error") or "error" in _content(b)[:40].lower()):
                            err += 1
        rows.append({"task": F.task_id(s), "sim": F.sim_key(s),
                     "direct": hit_direct, "via_wrapper": hit_inner, "errored": err,
                     "n_calls": len(names), "term": F.term_reason(s),
                     "mentions_tool_in_text": tool.split("_")[0] in F.assistant_text(s).lower(),
                     "last_text": F.assistant_text(s)[:200],
                     "writes": [x for x in labels if "apply" in x or "credit" in x][:6]})
    return rows


def apy_forensic(sims, tags):
    """⒞ 093·094 의 `submit_interest_discrepancy_report_7294` 세 인자가 **어디서 왔나**.

    C582 는 격리에서 *"서브가 base APY 문서에 못 닿아 0.0 으로 채운다"* 를 보였다. 그것이
    **라이브 궤적에서도 그 사슬인지**는 추론이었지 실측이 아니다 — 여기서 확인한다.
    셋은 출처가 다르다(핸드오프 §6):
        expected_apy       KB(정책이 그 클래스 APY 를 명시)  ← A3 색인·정확 전달이 닿는 자리
        actual_apy         고객 DB 레코드                     ← 모델 몫(C405ⓔ 경계)
        amount_difference  계산                               ← 값 레버 축
    gold 대조는 **진단 라벨**로만 쓴다 — A2/A3 저작에 쓰지 않는다([[23]]·`x451` 동형).
    """
    LOGMARK = ("get_correct_savings_apy", "T2_SG_GROUND", "T2_SG_ISOLATE", "SCAFFOLD_GET")
    logs = {}
    for t in tags:
        try:
            logs[t] = F.log_text(t)
        except Exception:
            logs[t] = ""
    rows = []
    for s2 in sims:
        sk = F.sim_key(s2)
        sent, subcalls = [], []
        for _m, tc in F.calls(s2):
            n = F.nameof(tc)
            if n == "submit_interest_discrepancy_report_7294":
                sent.append(F.argsof(tc))
            if n == "get_correct_savings_apy" or F.inner_name(F.argsof(tc)) == "get_correct_savings_apy":
                subcalls.append(F.argsof(tc))
        gold = [a for a in (F.gold_actions(s2) or [])
                if str(a.get("name") or "") == "submit_interest_discrepancy_report_7294"]
        # 그 sim 블록의 우리 층 자취
        marks = collections.Counter()
        for t, txt in logs.items():
            for line in txt.split(NLC):
                if sk and sk in line:
                    for k in LOGMARK:
                        if k in line:
                            marks[k] += 1
                    if "-> None" in line and "get_correct_savings_apy" in line:
                        marks["apy_returned_None"] += 1
                    if "source=0 rows" in line:
                        marks["src0_discard"] += 1
                    if "ungrounded operand" in line:
                        marks["gate1_drop"] += 1
        rows.append({"task": F.task_id(s2), "sim": sk,
                     "n_apy_subcalls": len(subcalls), "n_submit": len(sent),
                     "sent": [{k: v for k, v in (a or {}).items()
                               if k in ("expected_apy", "actual_apy", "amount_difference")}
                              for a in sent],
                     "gold": [{k: v for k, v in ((g.get("arguments") or {}) or {}).items()
                               if k in ("expected_apy", "actual_apy", "amount_difference")}
                              for g in gold],
                     "marks": dict(marks)})
    return rows


def _edit(a, b):
    """편집거리 — id 근접-변형 탐지용(C43 *"정박 치환·edit<=2 변형 70%"*). 형태만."""
    a, b = str(a or ""), str(b or "")
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def dispute_forensic(sims, tool):
    """⒟ 040 — `file_credit_card_transaction_dispute_4829` 26건의 정체.

    사용자 지시(2026-08-21): *"040 정밀 포렌식하여 원인 규명하라."* 계획 §2 는 이 블록을
    *"고객 DB 축이라 우리 층이 안 닿는다"* 로 미뤄 뒀는데, **그것은 처방 이야기이지 원인이
    아니다**. 원인을 가르는 술어는 하나다:

        ★ gold 값이 그 호출 **직전까지의 도구 출력 안에 이미 있었나**
            있었다  → 읽고도 틀린 것 = **정박 치환·전사 슬립**(부하·C43/C124 계보) → 격리가 산다
            없었다  → 못 읽은 것 = **탐색/발견 실패**(reach) → 다른 축

    같이 재는 것: 보낸 값이 문맥에 실재하는가(=다른 행에서 집어온 것 ↔ 날조) · gold 와의
    편집거리(근접 변형인가) · **필요 건수 대비 제출 건수**(coverage).
    gold 대조는 진단 라벨로만 쓴다([[23]]).
    """
    rows = []
    for s2 in sims:
        msgs = s2.get("messages") or []
        golds = [g for g in (F.gold_actions(s2) or []) if str(g.get("name") or "") == tool]
        sent_idx = []
        for i, m in enumerate(msgs):
            for tc in (m.get("tool_calls") or []):
                if F.nameof(tc) == tool or F.inner_name(F.argsof(tc)) == tool:
                    sent_idx.append((i, F.argsof(tc)))
        # 각 제출을 gold 중 **가장 가까운 것**과 짝짓는다(최소-diff·엔진 판단 0)
        pairs = []
        for i, args in sent_idx:
            ctx = " ".join(_content(b) for b in msgs[:i] if _role(b) in ("tool", "user"))
            best, bestd = None, 10 ** 9
            for g in golds:
                ga = g.get("arguments") or {}
                d = sum(1 for k in set(ga) | set(args or {})
                        if str((args or {}).get(k)) != str(ga.get(k)))
                if d < bestd:
                    best, bestd = ga, d
            diffs = []
            for k in sorted(set(best or {}) | set(args or {})):
                sv, gv = str((args or {}).get(k, "")), str((best or {}).get(k, ""))
                if sv == gv:
                    continue
                diffs.append({"arg": k, "sent": sv[:40], "gold": gv[:40],
                              "gold_in_context": bool(gv) and gv in ctx,
                              "sent_in_context": bool(sv) and sv in ctx,
                              "edit": _edit(sv, gv)})
            pairs.append({"msg": i, "n_diff": len(diffs), "diffs": diffs})
        rows.append({"task": F.task_id(s2), "sim": F.sim_key(s2),
                     "n_gold": len(golds), "n_sent": len(sent_idx),
                     "coverage_gap": len(golds) - len(sent_idx),
                     "term": F.term_reason(s2), "pairs": pairs})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--dup-tasks", default="016,040,079,085")
    ap.add_argument("--miss-tasks", default="072,073,074")
    ap.add_argument("--apy-tasks", default="093,094")
    ap.add_argument("--dispute-tasks", default="040")
    ap.add_argument("--out", default="x458_dup_missing.json")
    a = ap.parse_args()

    tags = [t for t in a.tags.split(",") if t.strip()]
    if not tags:
        tags = [F.tag_of_file(p) for p in F.all_result_files()
                if "t7328" in os.path.basename(p)]
    sims = []
    for t in sorted(set(tags)):
        try:
            sims += list(F.sims(t))
        except Exception as e:
            print("  ⚠%s 로드 실패: %r" % (t, e))
    print("=" * 96)
    print("x458 · 태그 %d · sim %d" % (len(set(tags)), len(sims)))
    print("  " + ", ".join(sorted(set(tags))))
    print("=" * 96)

    want_dup = {"task_" + x.strip() for x in a.dup_tasks.split(",") if x.strip()}
    want_miss = {"task_" + x.strip() for x in a.miss_tasks.split(",") if x.strip()}

    print("\n" + "─" * 96)
    print("⒜ %s 중복 — 같은 (도구, 인자) 반복" % DUP_TOOL)
    print("─" * 96)
    dups = dup_forensic([s for s in sims if F.task_id(s) in want_dup])
    print("반복 %d건 · 태스크 %s" % (len(dups), sorted({d["task"] for d in dups})))
    for d in dups[:14]:
        print("  %-9s gap=%-3d 사이=%s" % (d["task"], d["gap_msgs"], d["between"]))
        print("      key   %s" % d["key"])
        if d["tool_between"]:
            print("      tool  %s" % d["tool_between"][:150])
        if d["user_between"]:
            print("      user  %s" % d["user_between"][:150])
    gaps = sorted(d["gap_msgs"] for d in dups)
    if gaps:
        print("  간격 중앙 %d · 최소 %d · 최대 %d" % (gaps[len(gaps) // 2], gaps[0], gaps[-1]))

    print("\n" + "─" * 96)
    print("⒝ %s 미호출" % MISS_TOOL)
    print("─" * 96)
    miss = missing_forensic([s for s in sims if F.task_id(s) in want_miss], MISS_TOOL)
    for r in miss:
        print("  %-9s direct=%d wrapper=%d err=%d calls=%-3d term=%-16s"
              % (r["task"], r["direct"], r["via_wrapper"], r["errored"], r["n_calls"],
                 str(r["term"])[:16]))
        print("      쓴 것: %s" % (", ".join(r["writes"]) if r["writes"] else "(없음)"))
        print("      끝말: %s" % r["last_text"][:150].replace("\n", " "))

    print(NLC + "─" * 96)
    print("⒞ 093·094 — submit_interest_discrepancy_report_7294 세 인자의 출처")
    print("─" * 96)
    want_apy = {"task_" + x.strip() for x in a.apy_tasks.split(",") if x.strip()}
    apyr = apy_forensic([s2 for s2 in sims if F.task_id(s2) in want_apy], sorted(set(tags)))
    for r in apyr:
        print("  %-9s apy서브=%d submit=%d  우리층=%s"
              % (r["task"], r["n_apy_subcalls"], r["n_submit"], r["marks"] or "{}"))
        for i, sd in enumerate(r["sent"]):
            gd = r["gold"][i] if i < len(r["gold"]) else {}
            print("      보냄 %s" % json.dumps(sd, ensure_ascii=False))
            print("      gold %s   (진단 라벨·저작 미사용)" % json.dumps(gd, ensure_ascii=False))
        if not r["sent"]:
            print("      보냄 (없음 — 그 도구를 안 불렀다)")
            if r["gold"]:
                print("      gold %s" % json.dumps(r["gold"][0], ensure_ascii=False))

    print(NLC + "─" * 96)
    print("ⓟ 040 — %s 의 26건" % DISPUTE_TOOL)
    print("─" * 96)
    want_d = {"task_" + x.strip() for x in a.dispute_tasks.split(",") if x.strip()}
    disp = dispute_forensic([s2 for s2 in sims if F.task_id(s2) in want_d], DISPUTE_TOOL)
    agg = collections.Counter()
    per_arg = collections.defaultdict(collections.Counter)
    for r in disp:
        print("  %-9s gold %d · 제출 %d · 미제출 %d · term=%s"
              % (r["task"], r["n_gold"], r["n_sent"], r["coverage_gap"], str(r["term"])[:18]))
        agg["gold"] += r["n_gold"]
        agg["sent"] += r["n_sent"]
        agg["gap"] += max(0, r["coverage_gap"])
        for pr in r["pairs"]:
            for d in pr["diffs"]:
                agg["diff"] += 1
                per_arg[d["arg"]]["n"] += 1
                per_arg[d["arg"]]["gold_in_ctx"] += 1 if d["gold_in_context"] else 0
                per_arg[d["arg"]]["sent_in_ctx"] += 1 if d["sent_in_context"] else 0
                per_arg[d["arg"]]["near"] += 1 if 0 < d["edit"] <= 2 else 0
                print("      %-22s 보냄=%-22s gold=%-22s ctx(g/s)=%s/%s edit=%d"
                      % (d["arg"][:22], d["sent"][:22], d["gold"][:22],
                         d["gold_in_context"], d["sent_in_context"], d["edit"]))
    print(NLC + "  ★인자별 집계 — gold가 문맥에 있었나(있었으면 부하·없었으면 탐색)")
    for k, c in sorted(per_arg.items(), key=lambda kv: -kv[1]["n"]):
        print("    %-24s 틀림 %2d · gold∈문맥 %2d · 보낸것∈문맥 %2d · 근접변형(edit≤2) %2d"
              % (k[:24], c["n"], c["gold_in_ctx"], c["sent_in_ctx"], c["near"]))
    print("    합계  gold %d · 제출 %d · 미제출 %d · 인자불일치 %d"
          % (agg["gold"], agg["sent"], agg["gap"], agg["diff"]))

    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"tags": sorted(set(tags)), "n_sims": len(sims),
                   "dup_tool": DUP_TOOL, "dups": dups,
                   "miss_tool": MISS_TOOL, "missing": miss,
                   "apy_rows": apyr,
                   "dispute_tool": DISPUTE_TOOL, "dispute_rows": disp}, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
