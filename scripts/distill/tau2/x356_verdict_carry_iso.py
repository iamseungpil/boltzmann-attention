# -*- coding: utf-8 -*-
r"""x356 — **낱개 판정을 다음 단계로 그대로 올린다**(사용자 제안 2026-08-17·축자:
*"각각의 결과를 다음 스텝에서 같이 넣고 판단하게 하면 안 되나?"*). 형태 축의 8번째이자 마지막.

## 왜 (x354·x355 가 남긴 정확한 구멍)

낱개 판정은 바를 넘었다 — `gold=OK` **6/8** · `Green=위반(근거 검산)` **6/8**(C513).
그런데 최종 선택은 **4/8** 에 머물렀다. 이유가 설계에 있다: 두 프로브는 판정으로 **문서를
걸러낸 뒤 다시 문서 전문**을 줬다 ⇒ 모델이 **재료를 다시 읽고 다시 정박**할 자리를 줬다.

이번엔 [[65]] 그대로 한다 — **재료는 두고 답만 올린다**: 각 후보에 대한 *판정 + 근거 인용* 한 줄씩만
싣고 고르게 한다. 문서 전문은 안 준다.

## 셀 (n=8 · 요구·판정은 x355 절차 축자 재사용 · 라이브 본문 구성)

    A_REF        후보 전원의 **문서 전문**(현행 라이브)             ← 기준선(3/8 근방)
    B_VERDICT    **판정 줄만**(전 후보·OK/위반 + 근거 인용)          ← ★사용자 제안
    C_OKONLY     **OK 판정 후보의 판정 줄만**(위반은 안 보인다)      ← 제거 + 요약 결합
    D_NEG        무관 요구로 만든 판정 줄                            ← 부정통제

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    B ≥6/8 ∧ A 대비 +5      → **판정 이월이 레버다**(새 결정론 0 — 모델의 판정을 모델에게 되돌림)
    C ≥6/8 ∧ B < C−4        → 위반 후보를 **보여주는 것 자체**가 해롭다(제거가 본체)
    B,C 둘 다 ≤4/8          → 형태 축 완전 종료(8/8 실패) ⇒ 배치가 아니라 **다른 축**
    D_NEG gold ≥3           → 계기 무효

실행(리모트·8141):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x356_verdict_carry_iso.py
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
import x351_order_lever_iso as X                                  # noqa: E402
import x354_per_candidate_iso as Q                                # noqa: E402
import x355_violation_form_iso as V                               # noqa: E402

MARKS = Q.MARKS
PER_DOC = 1800


def cite_line(ans, doc):
    """모델이 낸 근거 문장 중 **문서에 실재하는** 첫 줄(없으면 빈 문자열)."""
    for l in [x.strip().strip('"').strip() for x in str(ans or "").split("\n") if x.strip()][1:]:
        if TS.quote_in(l, doc):
            return l
    return ""


def main():
    a2 = X.a2_load()
    po = a2.get("policy_ontology") or {}
    corpus = {}
    for fn in sorted(os.listdir(X.DOCS)):
        d = json.load(io.open(os.path.join(X.DOCS, fn), encoding="utf-8"))
        corpus[str(d.get("id") or fn)] = str(d.get("content") or "")
    idx = (po.get("doc_index") or {}).get(Q.GROUP) or {}
    classes = [c for c in sorted(idx) if c != "_general_"]
    cand_line = str(po.get("decide_candidates_text")).format(
        candidates=", ".join(X.disp(c) for c in classes))
    tpl = str(po.get("doc_decide_prompt") or "")
    rq = str(po.get("requirement_prompt") or "")
    live = lambda ask, mat: tpl.format(ask=ask, material=mat)        # noqa: E731
    neg_sim = X.sim_of(X.AUX, X.NEG_TASK, X.NEG_SEED)
    neg = " ".join(str((neg_sim.get("messages") or [])[X.NEG_MSG].get("content") or "").split())

    def mat_of(cls):
        ids = [d for c in cls for d in (idx.get(c) or ())]
        docs, _m = TS.read_docs(list(dict.fromkeys(ids)), corpus=corpus)
        return TS.as_material(docs, (), per_doc=PER_DOC)

    print("x356 · 후보 %d개 · **판정 줄만 올리기**(재료 전문 없음·[[65]])" % len(classes))
    print("판정(사전 고정): B≥6/8 ∧ A 대비 +5 → **판정 이월이 레버** · C≥6 ∧ B<C−4 → 위반 후보를 "
          "보이는 것 자체가 해롭다 · B,C 둘 다 ≤4 → 형태 축 완전 종료 · D_NEG ≥3 → 계기 무효\n")

    res = {}
    for seed, (turn, arr_idx) in sorted(X.SIMS.items()):
        sim = X.sim_of(X.TAG, X.TASK, seed)
        if sim is None:
            continue
        q_arr = X.quotes(rq, X.user_text(sim, upto_idx=arr_idx), "q_arr/%s" % seed) or []
        if not q_arr:
            print("── seed %s · 인용 0 — 집계 제외" % seed)
            continue
        req = X.block(q_arr)
        lines, ok_lines = [], []
        for c in classes:
            m = mat_of([c])
            ans = str((chat(V.VIO.format(req=req, doc=m[:6000]), None, 0.0, 500) or {}).get("content") or "")
            v = V.verdict(ans)
            why = cite_line(ans, m)
            # ★판정 줄 = 모델이 낸 판정 + **문서에 실재하는** 근거만(검산 실패 근거는 안 싣는다)
            tagv = "CONFLICTS" if v is True else ("OK" if v is False else "UNCLEAR")
            line = "- %s: %s%s" % (X.disp(c), tagv, (" — " + why[:200]) if why else "")
            lines.append(line)
            if v is False:
                ok_lines.append(line)
        print("── seed %s · 판정 줄 %d(그중 OK %d)" % (seed, len(lines), len(ok_lines)))
        ask = req + "\n\n" + cand_line
        arms = [("A_REF", live(ask, mat_of(classes))),
                ("B_VERDICT", live(ask, "\n".join(lines)))]
        if ok_lines:
            arms.append(("C_OKONLY", live(ask, "\n".join(ok_lines))))
        # ★부정통제 수리(1차 실행 D_NEG 8/8 붕괴 = 내 설계 결함): 무관 요구를 **ask 에만** 바꾸고
        #   판정 줄은 진짜 요구로 만든 것을 그대로 뒀다 ⇒ 그 줄이 이미 답을 흘린다.
        #   통제는 **판정 줄 자체를** 무관 요구로 다시 만들어야 뜻이 있다.
        neg_lines = []
        for c in classes:
            m = mat_of([c])
            an = str((chat(V.VIO.format(req=X.block([neg]), doc=m[:6000]), None, 0.0, 500)
                      or {}).get("content") or "")
            vn = V.verdict(an)
            wn = cite_line(an, m)
            neg_lines.append("- %s: %s%s"
                             % (X.disp(c),
                                "CONFLICTS" if vn is True else ("OK" if vn is False else "UNCLEAR"),
                                (" — " + wn[:200]) if wn else ""))
        arms.append(("D_NEG", live(X.block([neg]) + "\n\n" + cand_line, "\n".join(neg_lines))))
        r = P.run("x356-%s" % seed, {"tag": X.TAG, "task": X.TASK, "cut": turn,
                                     "sim": seed, "base": ""},
                  arms, MARKS, "(판정은 전 sim 합산 후·위 문구 그대로)", "", None, 8, 3, det=True)
        res[seed] = {"lines": lines, "ok": len(ok_lines),
                     "arms": {k: {m: v2[m][0] for m in MARKS} for k, v2 in (r or {}).items()}}

    n = len(res) or 1
    hit = lambda arm: sum(1 for v in res.values()                    # noqa: E731
                          if v["arms"].get(arm, {}).get("SILVERPLUS", 0) > 0)
    a, b, c, d = hit("A_REF"), hit("B_VERDICT"), hit("C_OKONLY"), hit("D_NEG")
    print("\n" + "=" * 96)
    print("gold 적중: A_REF %d/%d · **B_VERDICT %d/%d** · C_OKONLY %d/%d · D_NEG %d/%d"
          % (a, n, b, n, c, n, d, n))
    if d >= 3:
        print("   → ⛔계기 무효")
    elif b >= 6 and (b - a) >= 5:
        print("   → **판정 이월이 레버다** ⇒ 설계 착수(새 결정론 0)")
    elif c >= 6 and b < c - 4:
        print("   → 위반 후보를 **보이는 것 자체**가 해롭다 ⇒ 제거가 본체")
    elif b <= 4 and c <= 4:
        print("   → 형태 축 **완전 종료**(8/8 실패) ⇒ 배치가 아니라 다른 축")
    else:
        print("   → 미결")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                       "reports", "facet_rft_2026", "x356_verdict_carry_iso.json")
    with io.open(os.path.normpath(out), "w", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % os.path.normpath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
