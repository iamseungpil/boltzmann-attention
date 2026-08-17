# -*- coding: utf-8 -*-
r"""x364b — x364 블록 ②(상류 군 선택)의 **빠진 부정통제**를 채운다.

## 왜 (내 프로브 설계 결함·[[57]])

`x364` 블록 ②는 A_GRP/B_GRP 만 돌렸다: 짝지은 27 태스크서 EXTRA **11 → 6** · EXTRA_BIZ **3 → 0** ·
HIT 손실 **0**. 바는 통과했지만 **부정통제가 없다** — 줄어든 것이 *자격의 내용* 때문인지 *머리에
한 줄이 더 붙은 자리* 때문인지 안 갈린다. x356 이 정확히 그 함정에 한 번 빠졌다(판정 줄이 답을
흘려 통제가 붕괴).

    A_GRP  라이브 축자(A2 `group_prompt`)                    ← 기준선
    B_GRP  + **자격 한 줄**(x364 가 검산해 남긴 것 그대로)      ← 레버
    D_NEG  + **자격을 뒤집은 줄**(같은 인용·라벨만 반전)        ← 부정통제

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    D_NEG 의 EXTRA_BIZ ≥ A_GRP 의 것  ∧  B_GRP < A_GRP   → 이득은 **내용**이다 ⇒ 상류축 생존
    D_NEG 도 B 와 같이 줄어든다                          → ⛔이득이 내용이 아니다 ⇒ **축 폐기**
    D_NEG 가 HIT 를 깨지 않는다(=자격을 뒤집어도 무반응)   → 그 줄이 **안 읽힌다**는 증거 ⇒ 폐기

⚠자격 줄은 **다시 만들지 않는다** — `x364_part*.json` 이 남긴 것을 축자로 쓴다(재현성·[[08]]).

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
      /home/woori/venvs/seka_env/bin/python x364b_group_elig_negctl.py [part] [nparts]
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

import t2_search as TS                                            # noqa: E402
import x341_docbody_verdict as X341                               # noqa: E402
import x351_order_lever_iso as X                                  # noqa: E402
import x357_verdict_carry_multitask as M                          # noqa: E402
import x364_eligibility_axis_iso as E                             # noqa: E402


def main():
    part = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nparts = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    a2 = X.a2_load()
    po = a2.get("policy_ontology") or {}
    gtpl = str(po.get("group_prompt") or "")
    names_g = sorted((po.get("doc_index") or {}).keys())
    listing = "\n".join("  %s" % g for g in names_g)
    form, src = {}, None
    for fn in sorted(os.listdir(E.REPORTS)):
        if fn.startswith("x364_part") and fn.endswith(".json"):
            d = json.load(io.open(os.path.join(E.REPORTS, fn), encoding="utf-8"))
            form.update(d.get("form") or {})
            src = src or d.get("src")
    if not (form and gtpl):
        print("x364 산출/A2 없음 — 중단(계기 결함)")
        return 1
    keys = [t for i, t in enumerate(sorted(form)) if i % nparts == part]
    print("x364b · 조각 %d/%d · 태스크 %d개 · 자격 재료원 %s(x364 축자 재사용)" %
          (part, nparts, len(keys), src))
    print("판정(사전 고정): D_NEG EXTRA_BIZ ≥ A ∧ B < A → 내용이 산다 · D 도 같이 줄면 폐기 · "
          "D 가 HIT/EXTRA 에 무반응이면 그 줄은 안 읽힌다 ⇒ 폐기\n")

    res = []
    for tid in keys:
        o = (form[tid].get("out") or {}).get(src) or {}
        if not (o.get("v") and o.get("q")):
            continue
        req = M.instructions(tid)
        want = set(g for ax, gold in X341.gold_axes(tid).items()
                   for g in [M.group_for(ax, gold)] if g)
        if not (req and want):
            continue
        line = E.LINE.format(v=o["v"], q=o["q"][:200])
        neg = E.LINE.format(v=E.flip(o["v"]), q=o["q"][:200])
        row = {"task": tid, "want": sorted(want), "v": o["v"], "arms": {}}
        for arm, text in (("A_GRP", req), ("B_GRP", line + "\n\n" + req),
                          ("D_NEG", neg + "\n\n" + req)):
            raw, det = E.det_ask(gtpl.format(groups=listing, text=text), 200)
            sel = TS.groups_in(raw, names_g)
            extra = [g for g in sel if g not in want]
            row["arms"][arm] = {"sel": sel, "det": det, "hit": int(bool(want & set(sel))),
                                "extra": extra,
                                "extra_biz": [g for g in extra if g.startswith("business_")]}
        res.append(row)
        print("   %-9s %-10s 요청 %-30s A %s | B %s | D %s"
              % (tid, o["v"], ",".join(sorted(want)),
                 ",".join(row["arms"]["A_GRP"]["sel"]) or "-",
                 ",".join(row["arms"]["B_GRP"]["sel"]) or "-",
                 ",".join(row["arms"]["D_NEG"]["sel"]) or "-"))

    print("\n" + "=" * 96)
    for arm in ("A_GRP", "B_GRP", "D_NEG"):
        rs = [r["arms"][arm] for r in res]
        print("%-6s n=%-3d HIT %-3d EXTRA %-3d **EXTRA_BIZ %d**"
              % (arm, len(rs), sum(x["hit"] for x in rs), sum(len(x["extra"]) for x in rs),
                 sum(len(x["extra_biz"]) for x in rs)))
    out = os.path.join(E.REPORTS, "x364b_part%d.json" % part)
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
