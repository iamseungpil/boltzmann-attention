# -*- coding: utf-8 -*-
r"""x355 — **제거형 질의**(위반 열거)로 물으면 되는가. 055 savings 축 마지막 프로브.

## 왜

`x354`(만족형 *"이 요구를 다 만족하나?"*)는 **4/8** 에서 정확히 gold 만 남겼지만 두 곳에서 무너졌다:
  ⑴ 요구에 벤치 미끼(*"something 'green' and/or more premium"*)가 섞이면 **아무도 만족 못 한다**(2/8 전멸)
  ⑵ `Green` 이 살아남는다(2/8)
그리고 YES 의 **근거 인용이 과반 검산 실패**해 사전 규칙상 판정을 보류했다(C512⒟).

이 프로브는 두 가지를 바꾼다:
  ★질문 형태 = **제거형**([[63]] 과 같은 모양) — *"다 만족하나?"* 가 아니라 *"손님이 말한 요구를
    **위반**하는 것이 있나?"*. 미끼는 '위반'이 아니므로 전멸이 일어나지 않는다.
  ★계기 수리 = `max_tokens` 200 → **500**(인용 줄 절단 의심·C512⒟) · 인용을 **한 문장**으로 못박고
    · 인용이 **문서**에서 왔는지 **요구**에서 왔는지 갈라 센다(어디서 왔는지 모르면 못 고친다).

## 규율

엔진은 ⑴첫 낱말이 `VIOLATES` 인지 `OK` 인지 읽고 ⑵인용이 그 문서에 **실재하는지** 검산할 뿐이다
(`t2_search.quote_in`·정규식 0). **근거가 검산되지 않은 '위반' 판정은 제거하지 않는다**(fail-safe —
조용한 오제거 금지). 남기고 빼는 기준은 끝까지 **모델의 판정**이고 엔진은 집행만 한다([[66]]·[[22]]).

## 셀 (n=8 · 요구 = 도착 뒤 인용 · 라이브 본문 구성)

    A_REF      전 후보(제거 없음)                  ← 기대 오답(`Green`)
    B_KEPT     **위반 판정 안 받은 후보만**          ← 1차 종점
    D_NEG      같은 파이프라인 + 무관 요구           ← 부정통제

## 판정 (사전 고정)

    ⓐ (`Green`=위반 ∧ gold=OK) ≥6/8 → **제거형이 맞는 형태** ⇒ 레버 설계 착수
    ⓑ B_KEPT gold ≥6/8 ∧ A_REF 대비 **+5 이상** → 1차 종점 통과
    ⓒ D_NEG gold ≥3 → 계기 무효
    ⓓ 근거 검산 실패가 위반 판정의 절반 초과 → **판정 보류**(계기 결함)
    ⓔ 전멸(살아남은 후보 0)이 3/8 이상 → 제거형도 과잉 ⇒ 형태 축 종료·019 가족으로 이동

실행(리모트·8140):
  T2_PROBE_URL=http://localhost:8140/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x355_violation_form_iso.py
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
import x354_per_candidate_iso as Q                                # noqa: E402  (라벨·유틸 재사용)

MARKS = Q.MARKS
PER_DOC = 1800
VIO = ("A customer stated these requirements:\n{req}\n\n"
       "Here is one product on file:\n{doc}\n\n"
       "Does this product CONFLICT with any requirement the customer stated? Answer on the first "
       "line with VIOLATES or OK. If VIOLATES, put on the second line ONE sentence copied exactly "
       "from the product text above that shows the conflict. Nothing else.")


def verdict(ans):
    """첫 낱말이 VIOLATES 인가 OK 인가 — 문자열 비교만(정규식 0). 못 읽으면 None."""
    up = " ".join(str(ans or "").split()).upper()
    if up.startswith("VIOLATES"):
        return True
    if up.startswith("OK"):
        return False
    return None


def where_cited(ans, doc, req):
    """인용이 **문서**에서 왔나 **요구**에서 왔나 아무 데도 아닌가 — 고치려면 알아야 한다."""
    for l in [x.strip().strip('"').strip() for x in str(ans or "").split("\n") if x.strip()][1:]:
        if TS.quote_in(l, doc):
            return "doc"
        if TS.quote_in(l, req):
            return "req"
    return "none"


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

    def mat_of(cls_list):
        ids = [d for c in cls_list for d in (idx.get(c) or ())]
        docs, _m = TS.read_docs(list(dict.fromkeys(ids)), corpus=corpus)
        return TS.as_material(docs, (), per_doc=PER_DOC)

    print("x355 · 후보 %d개 · 제거형 질의 · max_tokens=500(C512⒟ 절단 수리)" % len(classes))
    print("판정(사전 고정): ⓐ(Green=위반 ∧ gold=OK) ≥6/8 → 제거형이 맞는 형태 · ⓑB_KEPT gold ≥6/8 ∧ "
          "A 대비 +5 → 1차 종점 통과 · ⓒD_NEG ≥3 → 계기 무효 · ⓓ근거 검산 실패 과반 → 판정 보류 · "
          "ⓔ전멸 ≥3/8 → 형태 축 종료(019 가족으로 이동)\n")

    res = {"n": 0, "ok_gold": 0, "vio_green": 0, "both": 0, "wipe": 0,
           "cite": {"doc": 0, "req": 0, "none": 0}, "sims": {}}
    for seed, (turn, arr_idx) in sorted(X.SIMS.items()):
        sim = X.sim_of(X.TAG, X.TASK, seed)
        if sim is None:
            continue
        q_arr = X.quotes(rq, X.user_text(sim, upto_idx=arr_idx), "q_arr/%s" % seed) or []
        if not q_arr:
            print("── seed %s · 인용 0 — 집계 제외" % seed)
            continue
        req = X.block(q_arr)
        vd, cites, kept = {}, {}, []
        for c in classes:
            m = mat_of([c])
            ans = str((chat(VIO.format(req=req, doc=m[:6000]), None, 0.0, 500) or {}).get("content") or "")
            v = verdict(ans)
            vd[c] = v
            if v is True:
                w = where_cited(ans, m, req)
                cites[c] = w
                res["cite"][w] += 1
                if w != "doc":                      # ★근거 미검산 = 제거하지 않는다(fail-safe)
                    kept.append(c)
            else:
                kept.append(c)
        res["n"] += 1
        gok = vd.get(Q.GOLD_CLASS) is False
        gvi = vd.get(Q.WRONG_CLASS) is True and cites.get(Q.WRONG_CLASS) == "doc"
        res["ok_gold"] += gok
        res["vio_green"] += gvi
        res["both"] += (gok and gvi)
        res["wipe"] += (len(kept) == 0)
        print("── seed %s · 살아남음 %d/%d · gold=%s · Green=%s(근거 %s)"
              % (seed, len(kept), len(classes), "OK" if gok else "위반/미상",
                 "위반" if vd.get(Q.WRONG_CLASS) else "OK", cites.get(Q.WRONG_CLASS, "-")))
        print("     생존: %s" % [X.disp(c) for c in kept][:9])
        ask = req + "\n\n" + cand_line
        arms = [("A_REF", live(ask, mat_of(classes)))]
        if kept:
            arms.append(("B_KEPT", live(ask, mat_of(kept))))
            arms.append(("D_NEG", live(X.block([neg]) + "\n\n" + cand_line, mat_of(kept))))
        r = P.run("x355-%s" % seed, {"tag": X.TAG, "task": X.TASK, "cut": turn,
                                     "sim": seed, "base": ""},
                  arms, MARKS, "(판정은 전 sim 합산 후·위 문구 그대로)", "", None, 8, 3, det=True)
        res["sims"][seed] = {"verdicts": vd, "cites": cites, "kept": kept,
                             "arms": {k: {m: v[m][0] for m in MARKS} for k, v in (r or {}).items()}}

    n = res["n"] or 1
    hit = lambda arm: sum(1 for v in res["sims"].values()            # noqa: E731
                          if v["arms"].get(arm, {}).get("SILVERPLUS", 0) > 0)
    a, b, d = hit("A_REF"), hit("B_KEPT"), hit("D_NEG")
    print("\n" + "=" * 96)
    print("ⓐ 제거 판정: gold=OK %d/%d · Green=위반(근거검산) %d/%d · **둘 다 %d/%d** · 전멸 %d/%d"
          % (res["ok_gold"], n, res["vio_green"], n, res["both"], n, res["wipe"], n))
    print("ⓓ 위반 근거 출처: 문서 %d · 요구 %d · 없음 %d"
          % (res["cite"]["doc"], res["cite"]["req"], res["cite"]["none"]))
    print("ⓑ 최종 선택 gold: A_REF %d/%d · B_KEPT %d/%d · D_NEG %d/%d" % (a, n, b, n, d, n))
    bad = res["cite"]["req"] + res["cite"]["none"]
    if d >= 3:
        print("   → ⛔계기 무효")
    elif bad > (res["cite"]["doc"] + bad) / 2.0:
        print("   → ⚠근거 검산 실패 과반 ⇒ **판정 보류**(계기 수리 후 재측정)")
    elif res["wipe"] >= 3:
        print("   → 제거형도 과잉(전멸 %d) ⇒ **형태 축 종료**, 019 가족으로 이동" % res["wipe"])
    elif res["both"] >= 6 and b >= 6 and (b - a) >= 5:
        print("   → **제거형이 맞는 형태** ⇒ 레버 설계 착수(새 결정론 0·모델 판정 집행만)")
    else:
        print("   → 미결 ⇒ 예정대로 019 가족으로 이동(형태 축은 여기서 멈춘다)")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                       "reports", "facet_rft_2026", "x355_violation_form_iso.json")
    with io.open(os.path.normpath(out), "w", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % os.path.normpath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
