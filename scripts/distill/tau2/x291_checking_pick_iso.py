# -*- coding: utf-8 -*-
r"""x291 — 075 checking-픽 격리: 후보 문서/총액 표가 손에 있으면 gold 픽이 나오는가.

사전등록: `X291_CHECKING_FIT_DESIGN_2026_08_13.md`(§1 셀·§2 판정 매트릭스 고정).

배경(t7275 075 t0 전수 정독): Green Fee-Free 문서 회수 0·계산 0으로 즉시 Purple 픽(함정)·
`check_card_application_fit` 오선택. 이 프로브가 가르는 것 = 픽 결손의 실패 단계
(회수/전달 ↔ F2b 산술 ↔ 자기-정박 ↔ 픽 자체(학습행)).

셀 5 (n=8·계기 = FINAL 줄의 클래스명·feefree 포함=hit):
  A_LIVE  msg00~09 그대로(문서 없음) — 재현 대조
  B_DOCS  + 전 클래스 ATM 문서 기계선별(클래스당 ≤2편·1200자)
  C_CALC  B_DOCS + 클래스별 총액 표((B)-동형 출력 모사·argmax 없음)
  D_NEG   + business 클래스 문서 동수 — 통제([[57]])
  E_FRESH msg00~01+usage msg+문서(중간 Purple 발화 제거) — 정박 분리([[18]])

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x291_checking_pick_iso.py [N]
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
TASK = "task_075"
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"

# C_CALC 표 — 계기 전용((B)-동형 출력 모사). 전 수치는 정책 스케줄 축자에서 산출
# (`ATM_FEE_SCHEDULE_VERBATIM` + green_ff_005 + gold_years_002)·argmax/추천 문구 없음.
CALC_TABLE = """Computed Rho-Bank ATM fee totals for the stated usage (18 foreign out-of-network withdrawals, $350 each, over 3 months), derived from the account fee schedules (out-of-network fee + foreign ATM fee both apply per withdrawal; free-withdrawal allowances deducted):
- green_fee-free_account: $0.00 out-of-network + $0.00 foreign = $0.00 total
- bluest_account: $36.00 + $0.00 = $36.00 total
- purple_account: $45.00 + $0.00 = $45.00 total
- gold_years_account: $0.00 + $63.00 = $63.00 total
- light_blue_account: $30.00 + $48.00 = $78.00 total
- light_green_account: $9.00 + $90.00 = $99.00 total
- evergreen_account: $45.00 + $126.00 = $171.00 total
- dark_green_account: $63.00 + $108.00 = $171.00 total
- blue_account: $54.00 + $189.00 = $243.00 total
- green_account_(checking): $54.00 + $189.00 = $243.00 total"""

ASK = ("Thanks. Based on what you already have here (do NOT call any tools now - just answer "
       "in text), which ONE personal checking account should I open to minimize my total ATM "
       "fees for that usage pattern (18 foreign out-of-network withdrawals of $350 each over "
       "3 months)? Briefly show the estimated total for the accounts you compared, then end "
       "your answer with exactly one line: FINAL: <account class name>")


def checking_classes():
    """클래스 열거 = 파일명에서 기계 도출(내용 판단 0·[[59]])."""
    out = set()
    for p in glob.glob(os.path.join(DOCS, "doc_checking_accounts_*.json")):
        m = re.match(r"doc_checking_accounts_(.+)_(\d+)\.json$", os.path.basename(p))
        if m:
            out.add(m.group(1))
    return sorted(out)


def top_docs(prefix, cls, cap=2, trunc=1200):
    """본문 'ATM' 포함 → (Foreign-ATM 우선, ATM 빈도 내림차순) 상위 cap 편."""
    scored = []
    for p in sorted(glob.glob(os.path.join(
            DOCS, "doc_%s_%s_*.json" % (prefix, glob.escape(cls))))):
        try:
            j = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        body = json.dumps(j, ensure_ascii=False)
        if "ATM" not in body:
            continue
        title = j.get("title") or os.path.basename(p)
        content = " ".join(str(j.get("content") or body).split())[:trunc]
        scored.append((("Foreign ATM" not in body), -body.count("ATM"), p,
                       "[%s]\n%s" % (title, content)))
    return [t[3] for t in sorted(scored)[:cap]]


def all_docs(prefix, classes, per=2):
    out = []
    for cls in classes:
        out.extend(top_docs(prefix, cls, cap=per))
    return out


def business_classes():
    out = set()
    for p in glob.glob(os.path.join(DOCS, "doc_business_checking_accounts_*.json")):
        m = re.match(r"doc_business_checking_accounts_(.+)_(\d+)\.json$", os.path.basename(p))
        if m:
            out.add(m.group(1))
    return sorted(out)


def final_pick(text):
    """FINAL 줄 정규화 — 부재 시 말미 300자 fallback(판정: miss 로 세되 fmt 로그)."""
    t = str(text or "")
    line = None
    for ln in reversed(t.splitlines()):
        if "FINAL" in ln.upper():
            line = ln
            break
    s = re.sub(r"[^a-z]", "", (line if line is not None else t[-300:]).lower())
    hit = "feefree" in s
    purple = "purple" in s
    return hit, purple, line is not None


def render(msgs, ours, docs=None, calc=None):
    out = []
    for i, m in enumerate(msgs):
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
    if calc:
        out.append("[tool] %s" % calc)
    out.append("[user] %s" % ASK)
    return "\n".join(out)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK and s.get("reward_info") is not None]
    if not sims:
        print("궤적 없음")
        return
    sim = sims[0]
    msgs = sim["messages"]
    cut = None
    for i, m in enumerate(msgs):
        if m.get("role") == "user" and "$350" in str(m.get("content") or ""):
            cut = i + 1
            break
    if cut is None:
        print("컷 없음($350 user msg)")
        return
    docs_b = all_docs("checking_accounts", checking_classes())
    docs_n = all_docs("business_checking_accounts", business_classes())[:len(docs_b)]
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    live = msgs[:cut]
    fresh = [msgs[0], msgs[1], msgs[cut - 1]]
    print("075 t%s cut=%d · 후보문서 %d편 · 통제문서 %d편 · n=%d · URL=%s\n" % (
        sim.get("trial"), cut, len(docs_b), len(docs_n), n,
        os.environ.get("T2_PROBE_URL", "localhost:8140")))
    arms = (("A_LIVE", live, None, None), ("B_DOCS", live, docs_b, None),
            ("C_CALC", live, docs_b, CALC_TABLE), ("D_NEG", live, docs_n, None),
            ("E_FRESH", fresh, docs_b, None))
    for label, base, docs, calc in arms:
        body = render(base, ours if base is live else {}, docs, calc)
        hits = purples = nofmt = 0
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 700)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            h, pu, fmt = final_pick(r.get("content"))
            hits += h
            purples += pu
            nofmt += (not fmt)
        print("%-7s 문맥 %6d자 · feefree %d/%d · purple %d · FINAL줄부재 %d" % (
            label, len(body), hits, n, purples, nofmt))
    print("\n※ 판정 매트릭스(사전 고정) = X291_CHECKING_FIT_DESIGN §2. A_LIVE≥6 또는"
          " D_NEG≥3 → 프로브 무효. B_DOCS≥6 → (A) 전달만. B_DOCS≤2 ∧ C_CALC≥6 → (B)"
          " checking-fit op. C_CALC≤2 → E_FRESH 로 정박/학습행 분기. 중간(3~5) → n=16 1회.")


if __name__ == "__main__":
    main()
