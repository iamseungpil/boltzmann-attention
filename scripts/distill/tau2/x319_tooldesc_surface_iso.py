# -*- coding: utf-8 -*-
r"""x319 — **이름이 아니라 설명**을 표면화하면 열리나 (1회 정책 표면화의 비용-효과).

사슬:
  x316  이름 46개 → **0/8** · 정답 이름 1개 → 4/8 · 정답+방해자 4 → 0/8  (목록은 안 듣는다)
  x318  같은 컷 bm25 질의 **1/8** ↔ 오답 도구를 시도해 **설명을 읽은 뒤 8/8**
        축자: *"apply_statement_credit_8472 는 **신용카드 계좌용**이지 체킹용이 아니다"*
  ⇒ 가설: 듣는 것은 목록의 크기가 아니라 **경계 정보**다 — 설명은 *무엇이 아닌지*를 말하고
    이름은 말하지 않는다([[63]] 모델은 스스로 배제를 못 한다).

비용(사용자 지시 *"1회 정책 표면화면 그것도 비용으로 하면 된다"* · `t2_index_build` 실측):
  도구 설명 **91종 전부가 소스 docstring 에 이미 있다** — 생성 비용 **0**
  의미 도출이 필요한 문서는 **43개뿐**(698 중 action 문서 6%) — 나머지 655 는 제품명으로 닿는다

셀 5 (n=8·컷·채점은 x318 축자 = 모델이 낸 질의를 **하네스 bm25 에 통과**시켜 doc 017 회수):
  A_REF        x318 A_REF 재현(도움 없음)
  B_TOOLDESC   + **도구 설명 91종**(1회 정책 표면화·이름+한 줄)
  C_NAMESONLY  + **이름만 91종**(설명 제거)            ← x316 을 같은 컷에서 재현·크기 통제
  D_ACTIONDOC  + **action 문서 43건의 제목**(최소 문서 표면)
  E_NEG        B + **존재하지 않는 도구 1종**(그럴듯한 설명 포함)  ← 순응 부정통제

판정(사전 고정):
  B ≥6 ∧ C ≤2      → **설명이 인자다**(같은 크기·같은 자리에서 갈린다) ⇒ 설명 표면화 출시
  B ≈ C            → 크기가 인자이고 설명은 무관 ⇒ 표면화 축 폐기
  D ≥6             → 문서 제목만으로도 열림(더 싼 레버·43줄)
  E_NEG ≥3         → 가짜 도구를 지명 = 귀속 불가·[[25]] 위험 기록
  전 팔 ≤2         → 표면화로 안 열린다 ⇒ 학습 축

⚠부작용도 잰다: 응답이 **가짜/무관 도구를 지명**하는 비율을 함께 인쇄한다(설명이 순응을 키울 수 있다).

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x319_tooldesc_surface_iso.py [N]
"""
import collections
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
import t2_forensic as F                                           # noqa: E402
import t2_index_build as IB                                       # noqa: E402
import x313_bailout_iso as B                                      # noqa: E402
import x318_query_formation_iso as Q                              # noqa: E402

FAKE = "issue_fee_reimbursement_7710"
FAKE_DESC = "Reimburse a customer for fees charged in error on any account type."


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    import importlib
    mod = importlib.import_module("tau2.domains.banking_knowledge.tools")
    desc = IB.tool_names_from_module(mod)
    raw = Q.load_docs()
    pipe = Q.bm25_pipe(raw)
    idx = IB.build(Q.DOCS, desc)
    fac = idx["doc_facets"]
    action_titles = [f["title"] for f in fac.values() if f["kind"] == "action" and f["title"]]

    sim = next(s for s in F.scored(Q.TAG) if F.task_id(s) == Q.TASK)
    ctx = "\n".join([B.HEAD, "", B.transcript(sim, 54)])

    def line(t, d, cap=140):
        return "- %s: %s" % (t, " ".join(str(d).split())[:cap])

    tools_desc = "\n".join(line(t, desc[t]) for t in sorted(desc))
    tools_name = "\n".join("- %s" % t for t in sorted(desc))
    docs_title = "\n".join("- %s" % t for t in sorted(set(action_titles)))
    fake_block = tools_desc + "\n" + line(FAKE, FAKE_DESC)
    print("x319 · 도구 %d종(설명 %d자) · action 문서 %d건 · n=%d\n"
          % (len(desc), len(tools_desc), len(set(action_titles)), n))

    def note(txt, head):
        return ctx + "\n\n[note] " + head + "\n" + txt + Q.ASK_Q

    arms = (
        ("A_REF", ctx + Q.ASK_Q),
        ("B_TOOLDESC", note(tools_desc, "Tools available in this environment:")),
        ("C_NAMESONLY", note(tools_name, "Tools available in this environment:")),
        ("D_ACTIONDOC", note(docs_title, "Policy documents that describe agent actions:")),
        ("E_NEG", note(fake_block, "Tools available in this environment:")),
    )
    res = {}
    for label, body in arms:
        k = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, None, 0.0 if i == 0 else 0.7, 120)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            out = " ".join(str(r.get("content") or "").split())[:120]
            ok = Q.score_query(pipe, out)
            k += ok
            cnt["hit" if ok else "miss"] += 1
            if FAKE in out:
                cnt["fake-named"] += 1
            print("    [%s %02d] %s  %s" % (label, i, "HIT" if ok else "-", out[:80]), flush=True)
        res[label] = k
        print("%-12s %d/%d · %s (본문 %d자)\n" % (label, k, n, dict(cnt), len(body)))
    print("판정(사전 고정): B≥6∧C≤2 → **설명이 인자**(출시) · B≈C → 크기가 인자(폐기) · "
          "D≥6 → 제목 43줄로 충분 · E_NEG 가짜 지명 ≥3 → 귀속 불가 · 전 팔 ≤2 → 학습 축")
    print("측정치: " + " · ".join("%s=%d" % (k, v) for k, v in res.items()))


if __name__ == "__main__":
    main()
