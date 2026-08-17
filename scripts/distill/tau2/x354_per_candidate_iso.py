# -*- coding: utf-8 -*-
r"""x354 — **후보를 하나씩 물으면 맞히나**. 레버냐 학습이냐를 가르는 갈림길 프로브.

## 왜 (C511 이 남긴 갈림길)

같은 2지선다인데 옆에 `Green` 이 있으면 **2/8**, `Gold Account` 가 있으면 **6/8** 이었다.
자료도 요구도 같다 ⇒ 판단 능력은 있고, 못 하는 것은 **스스로 오답을 지우는 일**([[63]]).
그러면 남은 물음은 하나다: *"고르라"* 대신 **후보마다 따로** 물으면 맞히는가.

    맞힌다  → 결손은 능력이 아니라 **묻는 방식(부하)** ⇒ 레버가 있다(전달·분해)
    틀린다  → 문서 해석 **능력** ⇒ 그때 비로소 학습/스케일 축이 정당([[13]]·[[42]])

## 어떻게 (엔진은 정답을 모른다)

후보 클래스마다: **요구 인용 + 그 클래스 문서**만 주고 *"이 요구들을 다 만족하나? YES/NO 와
그렇게 판정한 문서 문장을 인용하라"*. 엔진이 하는 일은 두 가지뿐 —
  ⑴ 답의 첫 낱말이 YES 인지 NO 인지 읽기(문자열 비교) ⑵ **인용이 그 문서에 실재하는지** 검산
     (`t2_search.quote_in`·C45 동형·정규식 0). 근거 없는 YES 는 **따로 센다**(조용히 안 버린다).
그 다음 **모델이 YES 라 한 후보만** 남겨 원래 결정 질문을 다시 던진다(제거는 모델의 판정을
집행하는 것이지 엔진의 판단이 아니다·[[66]]·[[22]]).

## 셀 (n=8 sim · 요구 인용은 요구가 도착한 뒤의 것 · 라이브 본문 구성)

    A_REF        전 후보 문서(제거 없음)             ← 기대 오답(`Green`)
    B_FILTERED   **YES 후보만** 남긴 문서            ← 1차 종점
    D_NEG        같은 파이프라인에 **무관 요구**      ← 부정통제(여기서 gold 나오면 계기 무효)

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    ⓐ 후보별 정확도  (gold=YES ∧ `Green`=NO) ≥6/8 → **묻는 방식이 문제였다** ⇒ 레버 진행
       `Green`=YES ≥6/8                          → **능력 문제** ⇒ 학습/스케일 축으로 이동
    ⓑ B_FILTERED gold ≥6/8                        → 레버가 실제로 답을 연다(A_REF 와 차 ≥5)
    ⓒ D_NEG gold ≥3                               → 계기 무효
    ⓓ 근거 없는 YES(인용 검산 실패)가 절반 넘으면   → 판정 보류(계기 결함)

실행(리모트·8140):
  T2_PROBE_URL=http://localhost:8140/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x354_per_candidate_iso.py
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

GROUP = "savings_accounts"
MARKS = {"SILVERPLUS": "Silver Plus", "GREEN": "Green Account", "GOLD": "Gold Account"}
GOLD_CLASS = "silver_plus_account"      # ★채점용 라벨(분석 전용·[[23]] — 프롬프트엔 안 들어간다)
WRONG_CLASS = "green_account_(savings)"
PER_DOC = 1800

FIT = ("A customer stated these requirements:\n{req}\n\n"
       "Here is one product on file:\n{doc}\n\n"
       "Does this product satisfy ALL of the stated requirements? Answer with YES or NO on the "
       "first line. On the second line, quote VERBATIM the one sentence from the product text "
       "above that decides your answer. Do not add anything else.")


def yes_no(ans):
    """첫 줄이 YES 인가 NO 인가 — 문자열 비교만(정규식 0). 못 읽으면 None."""
    head = " ".join(str(ans or "").strip().split(" ")[:1]).strip().strip("*:.,").upper()
    if head.startswith("YES"):
        return True
    if head.startswith("NO"):
        return False
    up = str(ans or "").upper()
    if up.startswith("YES"):
        return True
    if up.startswith("NO"):
        return False
    return None


def cited(ans, doc):
    """둘째 줄 이후의 인용이 그 문서에 **실재하는가**(정본 검산·강조 무시)."""
    lines = [l.strip().strip('"').strip() for l in str(ans or "").split("\n") if l.strip()]
    for l in lines[1:]:
        if TS.quote_in(l, doc):
            return True
    return False


def main():
    a2 = X.a2_load()
    po = a2.get("policy_ontology") or {}
    corpus = {}
    for fn in sorted(os.listdir(X.DOCS)):
        d = json.load(io.open(os.path.join(X.DOCS, fn), encoding="utf-8"))
        corpus[str(d.get("id") or fn)] = str(d.get("content") or "")
    idx = (po.get("doc_index") or {}).get(GROUP) or {}
    classes = [c for c in sorted(idx) if c != "_general_"]
    if GOLD_CLASS not in classes or WRONG_CLASS not in classes:
        print("클래스 라벨 불일치 — 중단(계기 결함): %s" % classes[:12])
        return 1
    cand_line = str(po.get("decide_candidates_text")).format(
        candidates=", ".join(X.disp(c) for c in classes))
    tpl = str(po.get("doc_decide_prompt") or "")
    rq = str(po.get("requirement_prompt") or "")
    live = lambda ask, mat: tpl.format(ask=ask, material=mat)        # noqa: E731
    neg_sim = X.sim_of(X.AUX, X.NEG_TASK, X.NEG_SEED)
    neg = " ".join(str((neg_sim.get("messages") or [])[X.NEG_MSG].get("content") or "").split())

    def mat_of(cls_list):
        ids = [d for c in cls_list for d in (idx.get(c) or ())]
        docs, _missing = TS.read_docs(list(dict.fromkeys(ids)), corpus=corpus)
        return TS.as_material(docs, (), per_doc=PER_DOC), docs

    print("x354 · group=%s · 후보 클래스 %d개 · gold 라벨=%r(채점 전용·프롬프트 밖)"
          % (GROUP, len(classes), GOLD_CLASS))
    print("판정(사전 고정): ⓐ(gold=YES ∧ Green=NO) ≥6/8 → **묻는 방식** 문제(레버 진행) · "
          "Green=YES ≥6/8 → **능력** 문제(학습 축) · ⓑB_FILTERED gold ≥6/8 ∧ A 와 차 ≥5 → 레버가 "
          "답을 연다 · ⓒD_NEG gold ≥3 → 계기 무효 · ⓓ근거 없는 YES 과반 → 판정 보류\n")

    res, per = {}, {"gold_yes": 0, "green_no": 0, "both": 0, "ungrounded": 0, "unread": 0, "n": 0}
    for seed, (turn, arr_idx) in sorted(X.SIMS.items()):
        sim = X.sim_of(X.TAG, X.TASK, seed)
        if sim is None:
            continue
        q_arr = X.quotes(rq, X.user_text(sim, upto_idx=arr_idx), "q_arr/%s" % seed) or []
        if not q_arr:
            print("── seed %s · 인용 0 — 집계 제외" % seed)
            continue
        req = X.block(q_arr)
        verdicts, ungrounded = {}, []
        for c in classes:
            m, docs = mat_of([c])
            if not m:
                continue
            ans = str((chat(FIT.format(req=req, doc=m[:6000]), None, 0.0, 200) or {}).get("content") or "")
            v = yes_no(ans)
            verdicts[c] = v
            if v is True and not cited(ans, m):
                ungrounded.append(c)
        yes = [c for c, v in verdicts.items() if v is True]
        unread = [c for c, v in verdicts.items() if v is None]
        gy = verdicts.get(GOLD_CLASS) is True
        gn = verdicts.get(WRONG_CLASS) is False
        per["n"] += 1
        per["gold_yes"] += gy
        per["green_no"] += gn
        per["both"] += (gy and gn)
        per["ungrounded"] += len(ungrounded)
        per["unread"] += len(unread)
        print("── seed %s · YES %d/%d 후보 · gold=%s · Green=%s · 근거없는YES %d · 못읽음 %d"
              % (seed, len(yes), len(verdicts),
                 {True: "YES", False: "NO", None: "?"}[verdicts.get(GOLD_CLASS)],
                 {True: "YES", False: "NO", None: "?"}[verdicts.get(WRONG_CLASS)],
                 len(ungrounded), len(unread)))
        print("     YES 목록: %s" % [X.disp(c) for c in yes][:8])
        ask = req + "\n\n" + cand_line
        m_all, _ = mat_of(classes)
        arms = [("A_REF", live(ask, m_all))]
        if yes:
            m_yes, _ = mat_of(yes)
            arms.append(("B_FILTERED", live(ask, m_yes)))
            arms.append(("D_NEG", live(X.block([neg]) + "\n\n" + cand_line, m_yes)))
        r = P.run("x354-%s" % seed, {"tag": X.TAG, "task": X.TASK, "cut": turn,
                                     "sim": seed, "base": ""},
                  arms, MARKS, "(판정은 전 sim 합산 후·위 문구 그대로)", "", None, 8, 3, det=True)
        res[seed] = {"verdicts": verdicts, "yes": yes, "ungrounded": ungrounded,
                     "arms": {k: {m: v[m][0] for m in MARKS} for k, v in (r or {}).items()}}

    print("\n" + "=" * 96)
    n = per["n"] or 1
    print("ⓐ 후보별 판정 (n=%d)" % per["n"])
    print("   gold=YES %d/%d · Green=NO %d/%d · **둘 다 맞음 %d/%d** · 근거없는 YES %d개 · 못읽음 %d개"
          % (per["gold_yes"], n, per["green_no"], n, per["both"], n, per["ungrounded"], per["unread"]))
    hit = lambda arm: sum(1 for v in res.values()                    # noqa: E731
                          if v["arms"].get(arm, {}).get("SILVERPLUS", 0) > 0)
    print("ⓑ 최종 선택 gold 적중: A_REF %d/%d · B_FILTERED %d/%d · D_NEG %d/%d"
          % (hit("A_REF"), n, hit("B_FILTERED"), n, hit("D_NEG"), n))
    if hit("D_NEG") >= 3:
        print("   → ⛔계기 무효")
    elif per["both"] >= 6 and hit("B_FILTERED") >= 6:
        print("   → **묻는 방식이 문제였다** ⇒ 후보별 자격질의+제거 레버 진행(새 결정론 0)")
    elif (n - per["green_no"]) >= 6:
        print("   → **능력 문제**(Green 을 만족이라 본다) ⇒ 학습/스케일 축")
    else:
        print("   → 미결(사전 문구 어느 칸에도 안 들어감)")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                       "reports", "facet_rft_2026", "x354_per_candidate_iso.json")
    with io.open(os.path.normpath(out), "w", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % os.path.normpath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
