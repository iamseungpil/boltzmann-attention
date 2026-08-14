# -*- coding: utf-8 -*-
r"""x318 — 072 는 왜 **이번에만** 옳은 질의를 냈나, 그리고 bm25 대신 **shell(grep)** 이 맞나.

관측(t7291·n=1·귀속 불가):
  t7290  크레딧/환불 질의 **0건** → 도구 이름을 끝내 못 봄
  t7291  msg58 에서 `apply_statement_credit_8472` 가 신용카드용임을 알아챈 **직후** 스스로 검색 →
         bm25 가 `doc_..._017` 을 **rank 1(15.63)** 로 반환 → unlock 까지 감(호출은 안 함)
  ⚠우리 온톨로지 검색이 준 재료는 `group=checking_accounts` 라 017 이 **없다** — 라우팅은 여전히
    틀렸고 모델이 우회로를 뚫었다. "우리 재료가 오답 도구를 시도하게 만들었고 그 실패가 재검색을
    낳았다"는 **가설**일 뿐이다.

⇒ 이 프로브는 **질의 형성**만 격리한다. 채점은 모델이 낸 질의를 **실제 하네스 bm25 에 통과시켜**
   doc 017 이 top-10 에 들어오는지로 한다 — 엔진 판단 0·정답 문구 지정 0([[62]] ④).
   선행 실측(C481): 정책 어휘 질의는 rank 1, **환불 어휘 질의는 전부 미적중**(어휘 간극).

셀 6 (n=8·컷은 072 라이브 축자):
  A_REF        msg54 컷(포기 선언 직전·오답 도구 시도 **전**) · "낼 질의 하나"
  B_AFTERFAIL  msg58 컷(오답 도구가 신용카드용임을 안 **뒤**) · 같은 요구   ← 실패가 인자인가
  C_MATERIAL   msg54 컷 + **우리 온톨로지 재료**(틀린 군 113문서 제목) · 같은 요구
                                                                    ← 틀린 재료가 돕나 해치나
  D_GREP       msg54 컷 · "문서를 훑을 **grep 패턴**을 내라"          ← shell 축(사용자 제안)
  E_GREP_AF    msg58 컷 · grep 패턴                                  ← shell × 실패-후
  F_TITLES     msg54 컷 + **군 9개의 제목 표본** · 같은 요구          ← x317 계열(라우팅 대안)

채점(둘 다 기계적):
  질의 → 하네스 `create_bm25_retrieval_pipeline` top-10 에 `_017` 포함?
  grep → 패턴을 문서 본문에 정규식 적용 · 매치 문서에 `_017` 포함 ∧ **매치 수 ≤ 20**
         (전부 매치하는 `.` 같은 패턴은 찾은 것이 아니다 — 좁히기까지가 성공)

판정(사전 고정):
  A_REF ≥6                     → 실패 경험 없이도 낸다 ⇒ t7290 의 미발화는 **다른 이유**(부하·국면)
  A ≤2 ∧ B_AFTERFAIL ≥6        → **오답 시도의 실패가 질의를 낳는다**(우리 재료의 간접 기여 지지)
  C_MATERIAL < A               → 틀린 군 재료가 **해친다** ⇒ 라우팅 수리가 선결
  D_GREP ≥6 ∧ A ≤2             → **shell 이 bm25 보다 낫다**(사용자 가설 지지) ⇒ 검색 축 이설
  F_TITLES ≥6                  → 제목 표면화로 열린다(x317 과 같은 방향·더 싼 레버)
  전 팔 ≤2                     → 질의 형성은 프롬프트 축이 아니다

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x318_query_formation_iso.py [N]
"""
import collections
import glob
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import t2_search as S                                             # noqa: E402
import x313_bailout_iso as B                                      # noqa: E402

TAG, TASK = "bank_t7291_a_20260814n", "task_072"
TGT = "doc_bank_accounts_bank_accounts_(general)_017"
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
ASK_Q = ("\n[instruction] Do NOT call any tool. Reply with ONE knowledge-base search query "
         "only — the query string you would search with next, nothing else.")
ASK_G = ("\n[instruction] Do NOT call any tool. Reply with ONE regular-expression pattern only "
         "— the pattern you would grep the policy documents with next, nothing else.")


def load_docs():
    return [json.load(io.open(p, encoding="utf-8"))
            for p in sorted(glob.glob(os.path.join(DOCS, "*.json")))]


def bm25_pipe(raw):
    from tau2.domains.banking_knowledge.retrieval import create_bm25_retrieval_pipeline
    from tau2.domains.banking_knowledge.data_model import KnowledgeBase
    return create_bm25_retrieval_pipeline(
        KnowledgeBase(documents={d["id"]: d for d in raw}), top_k=10)


def score_query(pipe, q):
    q = " ".join(str(q or "").split())[:200].strip('"`\' ')
    if not q:
        return False
    try:
        res = pipe.retrieve(q)
    except Exception:
        return False
    return any((t[0] if isinstance(t, tuple) else str(t)) == TGT for t in res)


def score_grep(raw, pat):
    pat = " ".join(str(pat or "").split())[:200].strip('"`\' ')
    if not pat:
        return False
    try:
        rx = re.compile(pat, re.I)
    except Exception:
        return False
    hits = [d["id"] for d in raw if rx.search(d.get("content") or "")
            or rx.search(d.get("title") or "")]
    return (TGT in hits) and (len(hits) <= 20)      # 찾되 **좁혀야** 성공


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    raw = load_docs()
    pipe = bm25_pipe(raw)
    sim = next(s for s in F.scored(TAG) if F.task_id(s) == TASK)
    a2 = json.load(io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "a2/banking_knowledge.specific.json"), encoding="utf-8"))
    groups = list((a2["policy_ontology"].get("doc_index") or {}))
    byid = {d["id"]: d for d in raw}

    def titles(ids, k):
        return "; ".join((byid.get(i) or {}).get("title") or i for i in ids[:k])

    ctx54 = "\n".join([B.HEAD, "", B.transcript(sim, 54)])
    ctx58 = "\n".join([B.HEAD, "", B.transcript(sim, 58)])
    mat = "\n[note] Policy documents currently available to you: %s" % titles(
        S.docs_for(a2, "checking_accounts"), 12)
    tit = "\n[note] Document groups and examples of what each contains:\n" + "\n".join(
        "%s — %s" % (g, titles(S.docs_for(a2, g), 3)) for g in groups)
    print("x318 · %s/%s · 문서 %d · 목표=%s · n=%d\n" % (TAG, TASK, len(raw), TGT[-4:], n))

    arms = (
        ("A_REF", ctx54 + ASK_Q, "q"),
        ("B_AFTERFAIL", ctx58 + ASK_Q, "q"),
        ("C_MATERIAL", ctx54 + mat + ASK_Q, "q"),
        ("D_GREP", ctx54 + ASK_G, "g"),
        ("E_GREP_AF", ctx58 + ASK_G, "g"),
        ("F_TITLES", ctx54 + tit + ASK_Q, "q"),
    )
    res = {}
    for label, body, kind in arms:
        k = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, None, 0.0 if i == 0 else 0.7, 120)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            out = " ".join(str(r.get("content") or "").split())[:120]
            ok = score_query(pipe, out) if kind == "q" else score_grep(raw, out)
            k += ok
            cnt["hit" if ok else "miss"] += 1
            print("    [%s %02d] %s  %s" % (label, i, "HIT" if ok else "-", out[:80]), flush=True)
        res[label] = k
        print("%-12s %d/%d · %s\n" % (label, k, n, dict(cnt)))
    print("판정(사전 고정): A≥6 → 실패 경험 불요 · A≤2∧B≥6 → 오답 시도의 실패가 인자 · "
          "C<A → 틀린 군 재료가 해침 · D≥6∧A≤2 → shell 이 낫다 · F≥6 → 제목 표면화로 열림 · "
          "전 팔 ≤2 → 프롬프트 축 아님")
    print("측정치: " + " · ".join("%s=%d" % (k, v) for k, v in res.items()))


if __name__ == "__main__":
    main()
