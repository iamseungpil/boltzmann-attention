# -*- coding: utf-8 -*-
r"""x352 — **재료 형태가 원인인가**(덤프 ↔ 고른 소수 문서). C509 가 남긴 유일한 후보.

## 왜

C509: 요구를 제때 줘도(`B_ARRIVED`) 격리 서브는 gold `Silver Plus` 를 **0/8** 로 못 냈다.
그런데 같은 모델이 **라이브 메인**으로서 요구를 손에 쥐었을 때는 자기 KB 검색으로
`Silver Plus` 를 맞혔다(C508⒤ · ctl `s863145` msg 50~53 축자). 두 자리의 차이는 **재료 형태**다:

    우리 서브   `material_for` = 문서군 **92개 × 400자 = 40,649자** 덤프
    라이브 메인 KB 검색 몇 번 → **소수 문서를 깊게**

## 셀 (요구 인용은 전 셀 동일 = 요구가 도착한 뒤의 것 · 정규식 0)

    A_DUMP      라이브 재료(92문서×400자)                  ← 라이브 재현(기대 오답·C509 재현)
    B_PICK400   **LLM 이 고른 ≤5문서** × 400자              ← *선택*만 바꾼다
    C_PICK2000  같은 문서 × 2000자                          ← *깊이*까지 바꾼다
    D_NEG       C 구성 + **다른 태스크의 요구**              ← 부정통제([[57]])
    E_PICK_NOW  C 구성 + **결정 시점**의 인용                ← 형태만으로 되나(요구 없이)

문서 선택은 **LLM 이 한다**(A3 `doc_index` 의 id 목록을 보여주고 고르게 한다). 엔진은 고른 id 가
그 닫힌 집합에 **실재하는지만** 검산한다([[22]] 닫힌 술어·C45 동형·⛔0 ③ 새 결정론 0).

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    C_PICK2000 gold ≥6/8 ∧ A_DUMP ≤2/8 → **재료 형태가 원인** ⇒ 레버 = 전달 형태(덤프 → 선택+깊이)
    B ≥6/8                              → *선택*만으로 산다(깊이 불필요)
    C ≈ A (둘 다 낮음)                   → 형태도 아니다 ⇒ 결손은 **선택 능력**(경계 후보·[[63]])
    D_NEG gold ≥3                        → 계기 무효
    E ≥6/8                               → 요구 없이 형태만으로 된다(순서 무관·C509 와 정합)

⚠진단 인쇄: 고른 문서에 gold 문서가 들어갔는지 표시한다 — **분석 전용**([[23]]: 레버는 gold 를
  안 본다). 안 들어갔다면 실패의 이름은 *형태*가 아니라 *탐색*이다.

실행(리모트·8141):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x352_material_shape_iso.py
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
import x351_order_lever_iso as X                                  # noqa: E402  (사이트·인용 절차 재사용)

GROUP = "savings_accounts"
MARKS = {"SILVERPLUS": "Silver Plus", "GREEN": "Green Account", "GOLD": "Gold Account"}
GOLD_DOC_HINT = "silver_plus"          # ★진단 인쇄 전용(레버 아님·[[23]])
PICK = ("These are the document ids on file for the {group} (this list is complete):\n{ids}\n\n"
        "The customer said:\n{quotes}\n\n"
        "Which documents should be read to decide which one to recommend? Choose at most 5 ids "
        "from the list above, copied verbatim. Reply with a JSON array of ids and nothing else.")


def pick_docs(ids, quotes_text, group, label=""):
    """LLM 이 읽을 문서를 고른다 → 엔진은 **닫힌 집합 실재만** 검산(판단 0)."""
    raw = str((chat(PICK.format(group=group, ids="\n".join(ids), quotes=quotes_text),
                    None, 0.0, 400) or {}).get("content") or "")
    i, j = raw.find("["), raw.rfind("]")
    if i < 0 or j <= i:
        print("   ⚠[%s] 문서 선택 JSON 없음(응답 %d자)" % (label, len(raw)))
        return []
    try:
        rows = json.loads(raw[i:j + 1])
    except Exception as e:
        print("   ⚠[%s] 문서 선택 파싱 실패(%s)" % (label, type(e).__name__))
        return []
    keep = [r for r in rows if isinstance(r, str) and r in ids][:5]
    if len(keep) != len(rows):
        print("   ⚠[%s] 고른 %d개 중 색인에 실재 %d개(나머지 버림)" % (label, len(rows), len(keep)))
    return keep


def main():
    a2 = X.a2_load()
    po = a2.get("policy_ontology") or {}
    corpus = {}
    for fn in sorted(os.listdir(X.DOCS)):
        d = json.load(io.open(os.path.join(X.DOCS, fn), encoding="utf-8"))
        corpus[str(d.get("id") or fn)] = str(d.get("content") or "")
    dump, info = TS.material_for(a2, GROUP, now=X.NOW_CLOCK, corpus=corpus)
    if not dump:
        print("재료 생성 실패 — 중단(계기 결함)")
        return 1
    # ★id 목록 = **문서 id**(`docs_for`), 후보줄 = **클래스 슬러그**(`doc_index` 키). 둘은 다르다 —
    #   1차 작성에서 내가 섞었다(계기 결함 자가검출).
    ids = list(TS.docs_for(a2, GROUP))
    cands = ", ".join(X.disp(x) for x in sorted((po.get("doc_index") or {}).get(GROUP) or ()))
    cand_line = str(po.get("decide_candidates_text")).format(candidates=cands)
    tpl = str(po.get("doc_decide_prompt") or "")
    rq = str(po.get("requirement_prompt") or "")
    live = lambda ask, mat: tpl.format(ask=ask, material=mat)        # noqa: E731

    neg_sim = X.sim_of(X.AUX, X.NEG_TASK, X.NEG_SEED)
    neg = " ".join(str((neg_sim.get("messages") or [])[X.NEG_MSG].get("content") or "").split())

    print("x352 · group=%s · 덤프 %d자(문서 %d·색인 id %d) · gold 문서 힌트=%r(진단 전용)"
          % (GROUP, len(dump), info.get("kept", 0), len(ids), GOLD_DOC_HINT))
    print("판정(사전 고정): C≥6/8 ∧ A≤2/8 → **재료 형태가 원인**(레버=전달 형태) · B≥6/8 → 선택만으로 "
          "산다 · C≈A → 형태도 아님(선택 능력=경계 후보) · D_NEG≥3 → 계기 무효 · E≥6/8 → 요구 없이 "
          "형태만으로 된다\n")

    res = {}
    for seed, (turn, arr_idx) in sorted(X.SIMS.items()):
        sim = X.sim_of(X.TAG, X.TASK, seed)
        if sim is None:
            continue
        t_now = X.user_text(sim, upto_turn=turn)
        t_arr = X.user_text(sim, upto_idx=arr_idx)
        q_now = X.quotes(rq, t_now, "q_now/%s" % seed) or []
        q_arr = X.quotes(rq, t_arr, "q_arr/%s" % seed) or []
        if not q_arr:
            print("── seed %s · arrived 인용 0 — 이 sim 은 집계에서 뺀다(C509 와 같은 자리)" % seed)
            continue
        print("── seed %s · 결정 turn=%d · 요구 도착 msg=%d · 인용 now %d / arrived %d"
              % (seed, turn, arr_idx, len(q_now), len(q_arr)))
        chosen = pick_docs(ids, X.block(q_arr), X.disp(GROUP), "pick/%s" % seed)
        if not chosen:
            print("   ⚠문서 선택 0개 — 이 sim 은 집계에서 뺀다(계기)")
            continue
        has_gold = any(GOLD_DOC_HINT in c for c in chosen)
        print("   고른 문서 %d개: %s   · gold 문서 포함? **%s**(진단 전용)"
              % (len(chosen), [c[-42:] for c in chosen], "예" if has_gold else "아니오"))
        docs, missing = TS.read_docs(chosen, corpus=corpus)   # ★(문서, 없는 id) 튜플이다
        if missing:
            print("   ⚠읽기 실패 id %s — 재료에서 빠졌다(조용히 넘기지 않는다)" % missing[:3])
        m400 = TS.as_material(docs, (), per_doc=400)
        m2000 = TS.as_material(docs, (), per_doc=2000)
        ask_arr = X.block(q_arr) + "\n\n" + cand_line
        ask_now = (X.block(q_now) + "\n\n" + cand_line) if q_now else cand_line
        arms = [("A_REF", live(ask_arr, dump)),                     # A_DUMP = 라이브 재현
                ("B_PICK400", live(ask_arr, m400)),
                ("C_PICK2000", live(ask_arr, m2000)),
                ("D_NEG", live(X.block([neg]) + "\n\n" + cand_line, m2000)),
                ("E_PICK_NOW", live(ask_now, m2000))]
        print("   재료 크기: 덤프 %d자 · pick400 %d자 · pick2000 %d자"
              % (len(dump), len(m400), len(m2000)))
        r = P.run("x352-%s" % seed, {"tag": X.TAG, "task": X.TASK, "cut": turn,
                                     "sim": seed, "base": ""},
                  arms, MARKS, "(판정은 전 sim 합산 후·위 문구 그대로)", "", None, 8, 3, det=True)
        res[seed] = {"gold_doc": has_gold, "chosen": chosen,
                     "arms": {k: {m: v[m][0] for m in MARKS} for k, v in (r or {}).items()}}

    print("\n" + "=" * 96)
    n = len(res)
    hit = lambda arm: sum(1 for v in res.values()                    # noqa: E731
                          if v["arms"].get(arm, {}).get("SILVERPLUS", 0) > 0)
    print("gold(`Silver Plus`) 적중 sim 수 (n=%d)" % n)
    for arm in ("A_REF", "B_PICK400", "C_PICK2000", "D_NEG", "E_PICK_NOW"):
        print("   %-11s %d/%d" % (arm, hit(arm), n))
    print("   gold 문서를 고른 sim: %d/%d (진단·[[23]])"
          % (sum(1 for v in res.values() if v["gold_doc"]), n))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                       "reports", "facet_rft_2026", "x352_material_shape_iso.json")
    with io.open(os.path.normpath(out), "w", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % os.path.normpath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
