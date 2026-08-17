# -*- coding: utf-8 -*-
r"""x361 — **상류(문서군 선택) 축**. 요청 밖 군 40% 를 판정-이월로 줄일 수 있나.

## 왜 (오늘 라이브 전수 계수·설계서 §2c)

t7305 32 sim: **군 결정 72회 중 29회(40%)가 손님이 요청하지 않은 군**이다
(055 `credit_cards` 9 · `business_*` 5 · `everyone_pay` 1 · 098 `business_*` 14). 024 는 **0회** —
요청이 명확하면 안 샌다 ⇒ 상류도 **미끼·모호성**과 같이 간다. 그리고 이 오류는 하류(클래스 선택)
레버의 이득을 **앞에서 깎는다**(설계서 §9 리스크 1).

## 셀 (태스크마다 · det · 요구원 2종)

    A_REF      라이브 그대로 — A2 `group_prompt` 로 군 목록을 한 번에 고르게 한다
    B_VERDICT  군마다 따로 *"손님이 이 군의 상품을 요청했나? **손님 말에서** 인용하라"* →
               엔진은 인용이 **손님 발화에 실재하는지만** 검산(`quote_in`) → 판정 줄만 올려 고르게
    D_NEG      같은 파이프라인에 **무관 요구**                      ← 부정통제

⚠[[06]] 선례: over-action 은 **게이트 금지 축**이다 ⇒ 이 프로브는 군을 **차단하지 않는다**.
  판정과 근거를 실어 모델이 다시 고르게 할 뿐이다(엔진 판단 0·[[66]]).

## 지표 (둘 다 본다 — 한쪽만 보면 과잉·과소를 못 가른다)

    HIT    그 태스크의 **요청 군**(gold 축들이 요구하는 군)을 골랐나
    EXTRA  요청 **밖** 군을 몇 개 골랐나  ← 낮을수록 좋다

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    B_HIT − A_HIT ≥ +2  ∧  B_EXTRA ≤ A_EXTRA  → 진행(합성 후보)
    B_EXTRA > A_EXTRA                          → 과잉 증가 ⇒ 폐기
    D_NEG 가 요청 군을 그대로 고르면(≥ n/2)     → 계기 무효(판정 줄이 답을 흘린다)

실행: T2_PROBE_URL=http://localhost:814X/... python x361_group_choice_iso.py [part] [nparts]
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import t2_probe as P                                              # noqa: E402
import t2_search as TS                                            # noqa: E402
import x341_docbody_verdict as X341                               # noqa: E402
import x351_order_lever_iso as X                                  # noqa: E402
import x357_verdict_carry_multitask as M                          # noqa: E402

ASKED = ("Below are a customer's own words.\n\n{req}\n\n"
         "Did the customer ask for a product in the group \"{group}\"? Answer on the first line "
         "with YES or NO. On the second line, quote VERBATIM the words of the customer that decide "
         "your answer. Nothing else.")
PICK = ("{lines}\n\nWhich document groups does the customer ask about? List every group that one "
        "of their requests is about, and no others. Reply with group names only.")


def yn(ans):
    up = " ".join(str(ans or "").split()).upper()
    return True if up.startswith("YES") else (False if up.startswith("NO") else None)


def cite_from(ans, text):
    for l in [x.strip().strip('"').strip() for x in str(ans or "").split("\n") if x.strip()][1:]:
        if TS.quote_in(l, text):
            return l
    return ""


def parse_groups(raw, names):
    """라이브와 **같은 규약**으로 군을 읽는다(`t2_search.formalize_groups` 축자 동형·정규식 0):
    등장 순서대로 담고, 긴 이름이 짧은 이름을 품으면 짧은 쪽은 뺀다."""
    low = " ".join(str(raw or "").split()).lower()
    out = sorted((g for g in names if g and g.lower() in low), key=lambda g: low.find(g.lower()))
    return [g for g in out if not any(o != g and g.lower() in o.lower() for o in out)]


def det_ask(body, maxtok=200):
    """온도 0 ×2 — 같으면 확정, 다르면 그 축을 **비결정**으로 표시(판정에서 뺀다)."""
    a = str((chat(body, None, 0.0, maxtok) or {}).get("content") or "")
    b = str((chat(body, None, 0.0, maxtok) or {}).get("content") or "")
    return a, (a.strip() == b.strip())


def main():
    part = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nparts = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    a2 = X.a2_load()
    po = a2.get("policy_ontology") or {}
    names = sorted((po.get("doc_index") or {}).keys())
    gtpl = str(po.get("group_prompt") or "")
    if not gtpl or not names:
        print("A2 group_prompt/doc_index 없음 — 중단(계기 결함)")
        return 1
    listing = "\n".join("  %s" % g for g in names)
    neg_sim = X.sim_of(X.AUX, X.NEG_TASK, X.NEG_SEED)
    neg = " ".join(str((neg_sim.get("messages") or [])[X.NEG_MSG].get("content") or "").split())

    jobs = []
    for fn in sorted(os.listdir(M.TASKS_DIR)):
        if not fn.endswith(".json"):
            continue
        tid = fn[:-5]
        if tid in M.EXCLUDE:
            continue
        axes = X341.gold_axes(tid)
        want = set(g for ax, gold in axes.items() for g in [M.group_for(ax, gold)] if g)
        if not want:
            continue
        req = M.instructions(tid)
        if not req:
            continue
        jobs.append({"task": tid, "want": sorted(want), "req": req})
    jobs = [j for i, j in enumerate(jobs) if i % nparts == part]
    print("x361 · 조각 %d/%d · 태스크 %d개 · 군 후보 %d개" % (part, nparts, len(jobs), len(names)))
    print("판정(사전 고정): B_HIT − A_HIT ≥ +2 ∧ B_EXTRA ≤ A_EXTRA → 진행 · B_EXTRA > A_EXTRA → "
          "폐기(과잉) · D_NEG 가 요청 군을 n/2 이상 고르면 계기 무효\n")

    res = []
    for j in jobs:
        req = j["req"]
        # A_REF — 라이브 축자 경로
        raw_a, det_a = det_ask(gtpl.format(groups=listing, text=req), 200)
        sel_a = parse_groups(raw_a, names)
        # B_VERDICT — 군마다 판정 + 손님 말 인용(엔진은 인용 실재만 검산)
        lines, neg_lines = [], []
        for g in names:
            for text, bucket in ((req, lines), (neg, neg_lines)):
                ans, _d = det_ask(ASKED.format(req=text, group=g), 160)
                v = yn(ans)
                why = cite_from(ans, text)
                bucket.append("- %s: %s%s" % (g, "ASKED" if v is True else
                                              ("NOT ASKED" if v is False else "UNCLEAR"),
                                              (" — \"%s\"" % why[:140]) if why else ""))
        raw_b, det_b = det_ask(PICK.format(lines="\n".join(lines)), 200)
        sel_b = parse_groups(raw_b, names)
        raw_d, _dd = det_ask(PICK.format(lines="\n".join(neg_lines)), 200)
        sel_d = parse_groups(raw_d, names)
        want = set(j["want"])
        row = {"task": j["task"], "want": sorted(want),
               "A": {"sel": sel_a, "hit": int(bool(want & set(sel_a))),
                     "extra": len([g for g in sel_a if g not in want]), "det": det_a},
               "B": {"sel": sel_b, "hit": int(bool(want & set(sel_b))),
                     "extra": len([g for g in sel_b if g not in want]), "det": det_b},
               "D": {"sel": sel_d, "hit": int(bool(want & set(sel_d))),
                     "extra": len([g for g in sel_d if g not in want])},
               "lines": lines}
        res.append(row)
        print("── %s · 요청군 %s\n     A %s (hit %d·extra %d%s) · B %s (hit %d·extra %d%s) · D %s"
              % (j["task"], row["want"], sel_a, row["A"]["hit"], row["A"]["extra"],
                 "" if det_a else "·⚠비결정", sel_b, row["B"]["hit"], row["B"]["extra"],
                 "" if det_b else "·⚠비결정", sel_d))

    n = len(res) or 1
    ah = sum(r["A"]["hit"] for r in res); bh = sum(r["B"]["hit"] for r in res)
    ae = sum(r["A"]["extra"] for r in res); be = sum(r["B"]["extra"] for r in res)
    dh = sum(r["D"]["hit"] for r in res)
    print("\n" + "=" * 96)
    print("n=%d · A_HIT %d · **B_HIT %d**(Δ%+d) · A_EXTRA %d · **B_EXTRA %d**(Δ%+d) · D_HIT %d"
          % (n, ah, bh, bh - ah, ae, be, be - ae, dh))
    if dh >= n / 2.0:
        print("   → ⛔계기 무효(부정통제가 요청 군을 그대로 고른다)")
    elif (bh - ah) >= 2 and be <= ae:
        print("   → **진행**(합성 후보)")
    elif be > ae:
        print("   → 폐기(요청 밖 군이 늘었다)")
    else:
        print("   → 미결")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                       "reports", "facet_rft_2026", "x361_part%d.json" % part)
    with io.open(os.path.normpath(out), "w", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % os.path.normpath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
