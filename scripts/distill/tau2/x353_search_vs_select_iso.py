# -*- coding: utf-8 -*-
r"""x353 — **탐색인가 선택인가**. C510 이 남긴 자리를 정면으로 가른다([[63]] 빼기 프로브).

## 왜

C510: 재료 형태(덤프↔고른 소수·깊이 5배)를 다 바꿔도 gold `Silver Plus` 는 **0/4**였고,
**gold 문서를 고른 sim 이 1/4**뿐이었다(한 sim 은 같은 상품의 판본 다섯 개를 골랐다).
그러면 실패의 이름이 둘 중 하나다:

    탐색  gold 문서가 손에 안 들어온다        ⇒ 레버 = 후보 확보(전달·검색)
    선택  손에 쥐어줘도 못 고른다             ⇒ 경계([[63]]·F3) — 전달로는 못 산다

## 셀 (요구 인용·본문 구성은 라이브 축자 그대로 · 재료만 바꾼다)

    A_REF        LLM 이 스스로 고른 ≤5문서                    ← x352 재현(기대 오답)
    B_GOLDIN     그 문서들 + **gold 문서를 강제로 넣는다**      ← 탐색 결손이면 여기서 열린다
    C_PAIR       **gold 문서 + 그 sim 의 오답 문서, 둘만**      ← 최소 2지선다(잡음 제거)
    D_NEG        C 구성 + **무관 요구**                        ← 부정통제(여기서 gold 나오면 계기 무효)
    E_MINUS      gold + **다른** 오답 하나(그 sim 의 오답을 뺀다) ← [[63]] 빼기: 제거가 인자인가

⚠gold 문서 주입은 **진단 전용**이다([[23]]): 이 프로브는 결손의 이름을 정하려는 것이지
  레버가 아니다. 레버는 gold 를 볼 수 없다.

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    B ≥6/8                  → 결손 = **탐색** ⇒ 레버 = 후보 확보(검색·전달·새 결정론 0)
    B ≤2/8 ∧ C ≥6/8         → 후보 **수**가 문제(잡음) ⇒ 레버 = 후보 축소
    B,C 둘 다 ≤2/8          → **선택 능력 = 경계**([[63]]) ⇒ 전달 축은 완전히 닫힌다
    E ≥ C+5                 → 오답 제거가 인자 = 빼기 결손 확증
    D_NEG gold ≥3           → 계기 무효

실행(리모트·8140):
  T2_PROBE_URL=http://localhost:8140/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x353_search_vs_select_iso.py
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

import t2_probe as P                                              # noqa: E402
import t2_search as TS                                            # noqa: E402
import x351_order_lever_iso as X                                  # noqa: E402
import x352_material_shape_iso as M                               # noqa: E402  (문서 선택 절차 재사용)

GROUP = "savings_accounts"
MARKS = {"SILVERPLUS": "Silver Plus", "GREEN": "Green Account", "GOLD": "Gold Account"}
GOLD_KEY = "silver_plus"          # ★진단 전용 주입 키([[23]] — 레버 아님)
WRONG_KEY = "green_account"       # 이 축의 실제 오답(라이브·격리 공통 8/8)
OTHER_KEY = "gold_account"        # E_MINUS 가 쓸 **다른** 오답
PER_DOC = 2000


def main():
    a2 = X.a2_load()
    po = a2.get("policy_ontology") or {}
    corpus = {}
    for fn in sorted(os.listdir(X.DOCS)):
        d = json.load(io.open(os.path.join(X.DOCS, fn), encoding="utf-8"))
        corpus[str(d.get("id") or fn)] = str(d.get("content") or "")
    ids = list(TS.docs_for(a2, GROUP))
    gold_ids = [d for d in ids if GOLD_KEY in d]
    wrong_ids = [d for d in ids if WRONG_KEY in d]
    other_ids = [d for d in ids if OTHER_KEY in d and "plus" not in d]
    if not (gold_ids and wrong_ids and other_ids):
        print("문서 id 를 못 찾음(gold %d·wrong %d·other %d) — 중단(계기 결함)"
              % (len(gold_ids), len(wrong_ids), len(other_ids)))
        return 1
    cands = ", ".join(X.disp(x) for x in sorted((po.get("doc_index") or {}).get(GROUP) or ()))
    cand_line = str(po.get("decide_candidates_text")).format(candidates=cands)
    tpl = str(po.get("doc_decide_prompt") or "")
    rq = str(po.get("requirement_prompt") or "")
    live = lambda ask, mat: tpl.format(ask=ask, material=mat)        # noqa: E731
    neg_sim = X.sim_of(X.AUX, X.NEG_TASK, X.NEG_SEED)
    neg = " ".join(str((neg_sim.get("messages") or [])[X.NEG_MSG].get("content") or "").split())

    def mat(doc_ids):
        docs, missing = TS.read_docs(list(dict.fromkeys(doc_ids)), corpus=corpus)
        if missing:
            print("   ⚠읽기 실패 %s" % missing[:2])
        return TS.as_material(docs, (), per_doc=PER_DOC)

    print("x353 · group=%s · 색인 %d문서 · gold %d개 · 오답(%s) %d개 · 다른오답(%s) %d개"
          % (GROUP, len(ids), len(gold_ids), WRONG_KEY, len(wrong_ids), OTHER_KEY, len(other_ids)))
    print("판정(사전 고정): B≥6/8 → **탐색** 결손(레버=후보 확보) · B≤2 ∧ C≥6 → 후보 수(축소) · "
          "B,C 둘 다 ≤2 → **선택=경계**([[63]]·전달 축 닫힘) · E≥C+5 → 빼기 결손 확증 · "
          "D_NEG gold ≥3 → 계기 무효\n")

    res = {}
    for seed, (turn, arr_idx) in sorted(X.SIMS.items()):
        sim = X.sim_of(X.TAG, X.TASK, seed)
        if sim is None:
            continue
        t_arr = X.user_text(sim, upto_idx=arr_idx)
        q_arr = X.quotes(rq, t_arr, "q_arr/%s" % seed) or []
        if not q_arr:
            print("── seed %s · arrived 인용 0 — 집계 제외" % seed)
            continue
        ask = X.block(q_arr) + "\n\n" + cand_line
        chosen = M.pick_docs(ids, X.block(q_arr), X.disp(GROUP), "pick/%s" % seed)
        if not chosen:
            print("── seed %s · 문서 선택 0 — 집계 제외(계기)" % seed)
            continue
        has_gold = any(GOLD_KEY in c for c in chosen)
        mine_wrong = [c for c in chosen if WRONG_KEY in c] or wrong_ids[:1]
        print("── seed %s · 인용 %d · 고른 문서 %d개 · gold 포함? %s"
              % (seed, len(q_arr), len(chosen), "예" if has_gold else "아니오"))
        arms = [
            ("A_REF", live(ask, mat(chosen))),
            ("B_GOLDIN", live(ask, mat(list(chosen) + gold_ids[:1]))),
            ("C_PAIR", live(ask, mat(gold_ids[:1] + mine_wrong[:1]))),
            ("D_NEG", live(X.block([neg]) + "\n\n" + cand_line, mat(gold_ids[:1] + mine_wrong[:1]))),
            ("E_MINUS", live(ask, mat(gold_ids[:1] + other_ids[:1]))),
        ]
        r = P.run("x353-%s" % seed, {"tag": X.TAG, "task": X.TASK, "cut": turn,
                                     "sim": seed, "base": ""},
                  arms, MARKS, "(판정은 전 sim 합산 후·위 문구 그대로)", "", None, 8, 3, det=True)
        res[seed] = {"gold_in_pick": has_gold, "chosen": chosen,
                     "arms": {k: {m: v[m][0] for m in MARKS} for k, v in (r or {}).items()}}

    print("\n" + "=" * 96)
    n = len(res)
    hit = lambda arm: sum(1 for v in res.values()                    # noqa: E731
                          if v["arms"].get(arm, {}).get("SILVERPLUS", 0) > 0)
    print("gold(`Silver Plus`) 적중 sim 수 (n=%d)" % n)
    for arm in ("A_REF", "B_GOLDIN", "C_PAIR", "D_NEG", "E_MINUS"):
        print("   %-9s %d/%d" % (arm, hit(arm), n))
    print("   (참고) LLM 이 스스로 gold 문서를 고른 sim: %d/%d"
          % (sum(1 for v in res.values() if v["gold_in_pick"]), n))
    b, c, e, d = hit("B_GOLDIN"), hit("C_PAIR"), hit("E_MINUS"), hit("D_NEG")
    if d >= 3:
        print("   → ⛔계기 무효(D_NEG %d)" % d)
    elif b >= 6:
        print("   → 결손 = **탐색** ⇒ 레버 = 후보 확보(전달)")
    elif b <= 2 and c >= 6:
        print("   → 후보 **수**가 문제 ⇒ 레버 = 후보 축소")
    elif b <= 2 and c <= 2:
        print("   → **선택 능력 = 경계**([[63]]) ⇒ 전달 축 닫힘")
    else:
        print("   → 미결(사전 문구대로 어느 칸에도 안 들어감)")
    if e >= c + 5:
        print("   → E_MINUS: 오답 제거가 인자 = **빼기 결손 확증**")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                       "reports", "facet_rft_2026", "x353_search_vs_select_iso.json")
    with io.open(os.path.normpath(out), "w", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % os.path.normpath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
