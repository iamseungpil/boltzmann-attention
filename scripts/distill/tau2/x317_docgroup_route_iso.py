# -*- coding: utf-8 -*-
r"""x317 — 072 가 필요한 문서에 못 닿는 진짜 자리: **문서군 라우팅**.

사슬(C482 배선 수리 후 처음 보이게 된 것):
  배선 수리 → `T2_SEARCH_AGENT` 가 라이브에서 처음 재료를 전달(`group=checking_accounts ·
  문서 113 · now=2025-11-14`) → **그런데 072 가 필요한 문서는 그 안에 없다**.
  `apply_checking_account_credit_5829` 를 담은 **유일한** 문서
  `doc_bank_accounts_bank_accounts_(general)_017` 은 군 **`bank_accounts_bank_accounts`**(47건)에
  색인돼 있고, LLM 이 고른 `checking_accounts`(110건)에는 **없다**.

왜 그렇게 고르나(가설·이 프로브가 재는 것): 군 프롬프트는 *"손님이 **무엇에 대해** 묻는가"*를
묻는다. 손님은 체킹 계좌(Bluest·Light Green)를 말하므로 `checking_accounts` 가 자연스럽다.
그런데 **절차 문서**(크레딧을 어떻게 적용하는가)는 내부 군에 산다 — 군 이름만으로는 그 사실을
알 길이 없다. 즉 결손은 "고를 의지"가 아니라 **군 이름의 무정보성**일 수 있다.

⚠[[66]] — "환불 요청이면 bank_accounts 를 보라"는 **케이스 열거**다. 쓰지 않는다.
   여기서 재는 두 후보는 모두 도메인-일반이다:
     ⑴ **군이 무엇을 담는지 보여준다**(문서 제목 표본 — env 파일에서 기계 추출·저작 0)
     ⑵ **묻는 방식을 바꾼다**(손님의 화제 → *이 요청을 처리하는 데 필요한 정책 문서가 있을 만한 군*)
   엔진은 여전히 **고르지 않는다** — 군 목록을 주고 LLM 이 답한다([[62]] ④).

셀 5 (n=8·요청 텍스트는 072 라이브 축자·군 목록은 A2 색인 그대로):
  A_REF     현행 `group_prompt` 축자                         ← 오라우팅 재현
  B_TITLES  + 각 군의 **문서 제목 표본**(기계 추출)
  C_REWORD  질문만 도메인-일반으로 교체(화제 → 절차 소재)
  D_BOTH    ⑴+⑵
  E_NEG     B_TITLES + **존재하지 않는 군** 하나 추가         ← 이름 순응 부정통제

적중 = 답에 **`bank_accounts_bank_accounts` 가 포함**(다른 군을 함께 대도 무방 — 엔진이 결정점마다
하나씩 처리하므로 목록에 있으면 도달한다). 이 기준은 **정책 문서 소재**에서 나온 것이지 채점표에서
나온 것이 아니다([[23]]).

판정(사전 고정):
  A_REF ≤2                          → 오라우팅 재현(이하 전제)
  B_TITLES ≥6 ∧ E_NEG ≤2            → **군 내용 표면화가 레버**(기계 추출·도메인 일반)
  C_REWORD ≥6                       → 문구만으로 열린다(더 싼 레버)
  D_BOTH ≥6 ∧ B,C ≤2                → 둘 다 필요
  E_NEG ≥3                          → 이름 순응 = 귀속 불가(레버 주장 금지)
  전 팔 ≤2                          → 라우팅은 프롬프트로 안 닫힌다 ⇒ 색인 구조 축(군 재편)

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x317_docgroup_route_iso.py [N]
"""
import collections
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import t2_search as S                                             # noqa: E402

TAG, TASK = "bank_t7290_a_20260814m", "task_072"
TARGET_GROUP = "bank_accounts_bank_accounts"
FAKE_GROUP = "fee_disputes_and_refunds"
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
REWORD = ("A customer service agent needs to look up policy documents for the request below.\n"
          "Which of these document groups could contain the policy or procedure documents the "
          "agent needs in order to CARRY OUT this request? Note that the documents describing how "
          "to perform an action are not always in the group named after the product the customer "
          "is talking about. List every group that could apply, and no others.\nGroups:\n{groups}"
          "\n\nRequest and conversation:\n{text}\n\nReply with the group names only, one per line, "
          "or 'none' if none of them applies.")


def titles_by_group(a2, groups, per=6):
    """군마다 문서 제목 표본 — **파일에서 기계 추출**(저작 0·판단 0·`doc_index` 순서 그대로)."""
    out = {}
    for g in groups:
        rows = []
        for d in S.docs_for(a2, g)[:per]:
            p = os.path.join(DOCS, d + ".json")
            try:
                rows.append(json.load(io.open(p, encoding="utf-8")).get("title") or d)
            except Exception:
                rows.append(d)
        out[g] = rows
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    a2 = json.load(io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "a2/banking_knowledge.specific.json"), encoding="utf-8"))
    po = a2["policy_ontology"]
    groups = list(po.get("doc_index") or {})
    tpl = po["group_prompt"]
    sim = next(s for s in F.scored(TAG) if F.task_id(s) == TASK)
    users = [" ".join(str(m.get("content") or "").split())
             for m in sim["messages"] if m.get("role") == "user"]
    ask = " --- ".join(users[:4])[-6000:]          # 라이브와 같은 구성(손님 발화)
    tb = titles_by_group(a2, groups)
    plain = "\n".join(groups)
    withtitles = "\n".join("%s — 예: %s" % (g, "; ".join(tb[g][:4])) for g in groups)
    fake = withtitles + "\n%s — 예: Fee Dispute Handling; Refund Authorization" % FAKE_GROUP
    print("x317 · %s/%s · 군 %d개 · 목표군=%s · 요청 %d자 · n=%d\n"
          % (TAG, TASK, len(groups), TARGET_GROUP, len(ask), n))

    arms = (
        ("A_REF", tpl.format(groups=plain, text=ask), TARGET_GROUP),
        ("B_TITLES", tpl.format(groups=withtitles, text=ask), TARGET_GROUP),
        ("C_REWORD", REWORD.format(groups=plain, text=ask), TARGET_GROUP),
        ("D_BOTH", REWORD.format(groups=withtitles, text=ask), TARGET_GROUP),
        ("E_NEG", tpl.format(groups=fake, text=ask), FAKE_GROUP),
    )
    res = {}
    for label, body, watch in arms:
        k = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, None, 0.0 if i == 0 else 0.7, 200)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            txt = str(r.get("content") or "")
            ok = watch in txt
            k += ok
            cnt[("hit" if ok else "miss")] += 1
            if label != "E_NEG" and FAKE_GROUP in txt:
                cnt["fake"] += 1
            print("    [%s %02d] %s %s" % (label, i, "HIT" if ok else "-",
                                           " ".join(txt.split())[:70]), flush=True)
        res[label] = k
        print("%-10s %d/%d · %s\n" % (label, k, n, dict(cnt)))
    print("판정(사전 고정): A≤2 전제 · B≥6∧E≤2 → 군 내용 표면화가 레버 · C≥6 → 문구만으로 열림 · "
          "D≥6∧B,C≤2 → 둘 다 필요 · E≥3 → 이름 순응(귀속 불가) · 전 팔 ≤2 → 색인 구조 축")
    print("측정치: " + " · ".join("%s=%d" % (k, v) for k, v in res.items()))


if __name__ == "__main__":
    main()
