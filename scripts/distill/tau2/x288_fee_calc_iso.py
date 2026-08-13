# -*- coding: utf-8 -*-
r"""x288 — F2b 격리: 정답 혜택 문서가 손에 있으면 ATM-fee 차액을 계산하는가 (073 실물).

사전등록: `PROBE_X283_X285_DESIGN_2026_08_13.md` §x288.

배경(t7275 073 t0·t7273b 073 t1 실측): write 파이프라인은 열렸고 credit 도 실행되나
**금액이 오답**($5.00/$3.00 vs gold $9.50/$9.00/$1.50)이고 3계좌 중 2계좌만. 클래스 혜택
문서는 두 런 모두 미회수(검색이 business/teen 문서 회수). 이 프로브가 가르는 것:

  H-A 재료만 문제 → A_DOCS(문서 동봉) ≥6/8 정확 ⇒ 수리 = (A) 문서 표면화(전달·기존 기전 보강)
  H-B 계산 자체 결손 → A_DOCS 0~2/8 ⇒ 수리 = (B) scaffold_get ATM-fee 항목(F2b [P]·§1.2)

셀 3 (n=8·계기 = 응답 속 계좌별 차액 금액):
  A_LIVE  거래 3계좌 확보 직후 문맥 그대로(문서 없음) — 재현 대조
  A_DOCS  + 세 클래스의 ATM 관련 문서 축자(KB 원문·기계 선별: 클래스 접두 ∧ 'ATM' 포함)
  D_NEG   + 무관 문서(다른 클래스) 동수 — 길이/문서-존재 통제([[57]])

판정(사전 고정): 정확 = {9.50, 9.00, 1.50} 세 값 모두 출현(9.5/9/1.5 표기 허용).
  A_DOCS ≥6/8 ∧ A_LIVE ≤2/8 → (A)만 출시. A_DOCS ≤2/8 → (B) 병행. 중간(3~5) → n=16 재측정 1회.

실행(리모트·유휴 GPU): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x288_fee_calc_iso.py [N]
"""
import collections
import glob
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import x238_action_forensic as X                                  # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402
import x283_discovery_reach_probe as P                            # noqa: E402

TAG = "bank_t7275_b_20260813v"
TASK = "task_073"
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
CLASSES = ["blue_account", "green_account_(checking)", "light_green_account"]
FOILS = ["purple_account", "gold_years_account", "evergreen_account"]
GOLD = [("9.50", "9.5"), ("9.00", "9.00"), ("1.50", "1.5")]
ASK = ("Based on the November transactions you retrieved for the three checking accounts and "
       "the account fee policies, list PER ACCOUNT which ATM fees were charged incorrectly and "
       "the total credit due for each account (exact dollar amounts).")


def class_docs(classes):
    """클래스 접두 ∧ 본문에 'ATM' — 기계 선별(내용 판단 0·[[59]])."""
    out = []
    for cls in classes:
        for p in sorted(glob.glob(os.path.join(
                DOCS, "doc_checking_accounts_%s_*.json" % glob.escape(cls)))):
            try:
                j = json.load(io.open(p, encoding="utf-8"))
            except Exception:
                continue
            body = json.dumps(j, ensure_ascii=False)
            if "ATM" in body:
                title = j.get("title") or os.path.basename(p)
                content = j.get("content") or body
                out.append("[%s]\n%s" % (title, " ".join(str(content).split())[:1500]))
    return out


def amounts_hit(text):
    t = str(text or "")
    n = 0
    for variants in GOLD:
        if any(v in t for v in variants):
            n += 1
    return n


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK and s.get("reward_info") is not None]
    if not sims:
        print("궤적 없음")
        return
    sim = sims[0]
    # cut = 마지막 거래내역 tool 결과 다음
    cut = None
    for i, m in enumerate(sim["messages"]):
        if m.get("role") == "tool" and "get_bank_account_transactions" in str(m.get("content") or ""):
            cut = i + 1
    if cut is None:
        print("컷 없음")
        return
    docs_a = class_docs(CLASSES)
    docs_n = class_docs(FOILS)[:len(docs_a)]
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    print("073 t%s cut=%d · 정답문서 %d편 · 통제문서 %d편 · n=%d · URL=%s\n" % (
        sim.get("trial"), cut, len(docs_a), len(docs_n), n,
        os.environ.get("T2_PROBE_URL", "localhost:8140")))
    for label, docs in (("A_LIVE", None), ("A_DOCS", docs_a), ("D_NEG", docs_n)):
        out = []
        for i, m in enumerate(sim["messages"][:cut]):
            r, c = m.get("role"), " ".join(str(m.get("content") or "").split())
            tcs = [(tc.get("function") or {}).get("name") or tc.get("name")
                   for tc in (m.get("tool_calls") or [])]
            if tcs:
                out.append("[%s calls] %s" % (r, ", ".join(tcs)))
            if c:
                out.append("[%s] %s" % (r, c[:2500] if r == "tool" else c[:600]))
            for t in ours.get(i, ()):
                out.append("[system] %s" % t[:800])
        if docs:
            out.append("[tool] Retrieved policy documents:\n" + "\n\n".join(docs))
        out.append("[user] %s" % ASK)
        body = "\n".join(out)
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 700)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            k = amounts_hit(r.get("content"))
            cnt[k] += 1
        full = cnt.get(3, 0)
        print("%-7s 문맥 %6d자 · 3값전부 %d/%d · 분포 %s" % (
            label, len(body), full, n, dict(sorted(cnt.items()))))
    print("\n※ 판정(사전 고정): A_DOCS ≥6/8 ∧ A_LIVE ≤2/8 → (A) 문서 표면화만."
          "\n  A_DOCS ≤2/8 → (B) scaffold_get ATM-fee 항목 병행. 3~5 → n=16 재측정 1회.")


if __name__ == "__main__":
    main()
