# -*- coding: utf-8 -*-
r"""x436 — **어느 제약이 gold 를 떨구나** (확대 표본 사후 귀속 · 2026-08-20 밤)

## 위치
`x431` 은 스펙 전체를 한 번에 걸어 *생존 수*만 남긴다. 카드축은 **과잉 필터**(003 gold 탈락·
063 전멸)이고 계좌축은 **미좁힘**(057 생존 36)인데, 합쳐진 숫자로는 **어느 제약이 범인인지**
말할 수 없다. 여기서는 기록된 스펙을 **하나씩** 걸어 그 자리를 특정한다.

## 규율
★LLM 을 부르지 않는다 — 입력은 `x431_wide*.json` 에 이미 영속된 스펙뿐이다(무료·결정론).
★필터·표·계열 선언은 전부 `x431` 정본을 그대로 import 한다([[67]] 사본 금지).
★계좌축은 **두 표**에 대고 각각 돌린다: 확대 표본이 실제로 쓴 표(`x430_account_facts_llm.json`
  = 값 192·미해결 704)와 코퍼스 전수 표(`_filled.json` = 값 346·미기재확정 550·미해결 0).
  둘이 갈리면 §8-4 의 "계좌축은 덜 좁힌다"는 **표 피복률의 산물**이지 스펙의 성질이 아니다.

사용: py -3 x436_constraint_attribution.py [--wide x431_wide1.json]
"""
import argparse
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

import x431_spec_selects as X  # noqa: E402

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")


def account_table(which):
    p = os.path.join(REP, "x430_account_facts_llm%s.json" % ("_filled" if which == "filled" else ""))
    with io.open(os.path.abspath(p), encoding="utf-8") as f:
        return json.load(f)


def cell(row, at):
    """그 칸의 상태를 사람이 읽는 꼴로 — 값 / 미기재확정(absent) / 미해결(빈칸)."""
    c = (row or {}).get(at) or {}
    if c.get("values"):
        v = X.cellval(c)
        return "=%s%s" % (c["values"][0], "" if v is None else "" if c.get("unit") != "boolean" else "(%.0f)" % v)
    if c.get("absent"):
        return "absent(문서가 없다고 말함)"
    return "미해결(안 뽑힘)"


passes = X.passes          # ★술어는 정본 하나뿐이다([[67]]) — 여기서 사본을 짓지 않는다


def candidates(tbl, fams):
    out = {}
    for cls, row in tbl.items():
        if not isinstance(row, dict):
            continue
        if fams and (row.get("_family") not in fams):
            continue
        out[cls] = row
    return out


def gold_row(cands, gold):
    g = X.clsname(gold)
    for cls, row in cands.items():
        if X.clsname(cls) == g:
            return cls, row
    for cls, row in cands.items():             # 계열 꼬리 차이 허용
        if X.clsname(cls).startswith(g) or g.startswith(X.clsname(cls)):
            return cls, row
    return None, None


def report(case, arg, tbl, label):
    cons = case["spec"] if isinstance(case["spec"], list) else []
    fams = X.arg_families(arg)
    cands = candidates(tbl, fams)
    caps = X.table_caps(tbl, fams)             # ★그 표가 지지하는 술어(x431 정본과 같은 유도)
    gcls, grow = gold_row(cands, case["gold"])
    print("  [%s] 후보 %d · gold %s" % (label, len(cands), gcls or "⛔표에 없음"))
    surv = set(cands)
    for i, con in enumerate(cons):
        alone = {c for c, r in cands.items() if passes(r, con)}
        surv = {c for c in surv if passes(cands[c], con)}
        mark = ""
        if grow is not None:
            ok = passes(grow, con)
            mark = "gold %s (%s %s)" % ("생존" if ok else "⛔탈락",
                                        con.get("attribute"), cell(grow, con.get("attribute")))
        sup = con.get("op") in (caps.get(con.get("attribute")) or ())
        print("    c%d %-24s %-6s %-10s 단독생존 %3d · 누적 %3d · %s%s"
              % (i, con.get("attribute"), con.get("op"), con.get("value"),
                 len(alone), len(surv), mark, "" if sup else "  ⚠표 미지지"))
        if con.get("because"):
            print("        근거: %s" % str(con["because"])[:110])
    print("    ⇒ 최종 생존 %d · gold_in %s" % (len(surv), bool(gcls and gcls in surv)))
    # ★능력 검산 후 — 표가 지지 못하는 술어를 **판정에서 뺐을 때**(x431 이 이제 하는 것).
    keep = [x for x in cons if x.get("op") in (caps.get(x.get("attribute")) or ())]
    if len(keep) != len(cons):
        s2 = {c for c, r in cands.items() if all(passes(r, x) for x in keep)}
        print("    ⇒ 능력 검산 후(미지지 %d건 제외): 생존 %d · gold_in %s"
              % (len(cons) - len(keep), len(s2), bool(gcls and gcls in s2)))
    return len(surv), bool(gcls and gcls in surv)



def declared_share(cases):
    """②의 **선언 분포** — 요구로 센 것 대 `question`/`background` 로 버린 것.

    057 이 36 에서 안 줄어드는 이유를 여기서 센다: 제약이 1 개뿐인 것은 모델이 속성을 못 봐서가
    아니라 **본 속성 대부분을 요구가 아니라고 선언**했기 때문인지 아닌지.
    """
    print(chr(10) + chr(10) + "=== ② 선언 분포 (요구 vs question/background) ===")
    tot_req = tot_drop = 0
    for c in cases:
        req = len(c["spec"] if isinstance(c["spec"], list) else [])
        drops = c.get("declared_non_requirement") or []
        kinds = {}
        for d in drops:
            kinds[d[1]] = kinds.get(d[1], 0) + 1
        tot_req += req
        tot_drop += len(drops)
        print("  %s t%s  요구 %d · 버림 %d %s  %s"
              % (c["task"], c["trial"], req, len(drops), kinds,
                 ", ".join(sorted({d[0] for d in drops}))[:80]))
    print("  ⇒ 합계 요구 %d · 버림 %d (버림 비율 %.0f%%)"
          % (tot_req, tot_drop, 100.0 * tot_drop / max(1, tot_req + tot_drop)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", default="x431_wide1.json")
    a = ap.parse_args()
    with io.open(os.path.abspath(os.path.join(REP, a.wide)), encoding="utf-8") as f:
        cases = json.load(f)
    card, _ = X.card_table()
    acc_asrun, acc_filled = account_table("asrun"), account_table("filled")
    for c in cases:
        arg = "card_type" if gold_row(candidates(card, X.arg_families("card_type")), c["gold"])[0] else "account_class"
        print("\n=== %s t%s · gold=%s · arg=%s · (x431 기록: 제약 %d · 생존 %d · gold_in %s)"
              % (c["task"], c["trial"], c["gold"], arg, c["n_constraints"], c["n_surv"], c["gold_in"]))
        if arg == "card_type":
            report(c, arg, card, "A2 카드표")
        else:
            n1, g1 = report(c, arg, acc_asrun, "확대표본이 쓴 표 192값")
            n2, g2 = report(c, arg, acc_filled, "전수 표 346값·absent 550")
            if (n1, g1) != (n2, g2):
                print("    ⚠표에 따라 갈린다: %d/%s → %d/%s" % (n1, g1, n2, g2))
    declared_share(cases)


if __name__ == "__main__":
    main()
