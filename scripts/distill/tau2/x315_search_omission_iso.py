# -*- coding: utf-8 -*-
r"""x315 — 072 의 진짜 결손: **필요한 행동을 검색하지 않는다**(그리고 스친 이름으로 대체한다).

포렌식(C478 이후·t7290 072 전 구간 정독):
  msg46~53  두 계좌의 fee 불일치를 **정확히** 산출(도구 출력도 정확)
  msg54     *"환불 처리를 언급만 하고 실행하지 않았다… 그런데 그 도구(`process_refunds`)가
            나에게 없다"* — **없는 이름을 지어내 놓고 부재를 결론**한다
  msg55     손님이 그제서야 화를 내며 이관 요구(→ C478: 이후 이관은 순응이지 이탈이 아니다)
  msg56     **savings** 크레딧 도구를 unlock(두 계좌 다 checking 인데) → 게이트 차단
  실제 필요한 `apply_checking_account_credit_5829` 는 **대화 전체에 한 번도 등장하지 않는다**.
  KB 질의 5건 = 신원확인·거래조회·계좌ID×2·이관 — **환불/크레딧 질의가 없다**.
  그 이름은 KB 문서(`doc_bank_accounts_bank_accounts_(general)_017`)에 실재한다 = 검색으로 도달 가능.
  스친 `apply_savings_account_credit_6831` 은 **무관한 질의 두 건의 부산물**로 본 유일한 크레딧-류
  이름이었다 ⇒ **가장 가까운 이름으로 대체**([[63]] 계열: 배제를 못 하고 근사치를 집는다).

⚠[[66]] — "네가 필요한 행동을 검색하지 않았다"는 **의도 분류**다. 엔진은 그걸 못 한다.
   그래서 팔을 **의도 0**인 것부터 쌓는다: 자기 질의 전사 → 미열람 문서 존재 → 이름 목록.
   판정은 *어디서부터 열리는가* 이고, 열린 지점이 곧 **허용 가능한 최소 레버**다.

셀 5 (n=8·컷 = msg54 직전 = 포기 선언 **전**·손님은 아직 이관을 요구하지 않았다):
  A_REF      라이브 문맥 축자                              ← 포기/대체 재현
  B_QUERIES  + "지금까지 낸 KB 질의는 다음뿐이다: <전사>"   ← 의도 0·순수 전사
  C_UNREAD   + "KB 에 아직 어느 질의로도 열람되지 않은 문서가 N건 있다"  ← 의도 0·집합 차
  D_NAMES    + KB 문서에 실재하는 **도구 이름 전부**(호출한 것 제외·수십 개) ← 이름 충분성
             (엔진은 **고르지 않는다** — 전부 나열하고 선택은 LLM·gold 무접촉·출처=env KB)
  E_NEG      + 같은 형식으로 **KB 에 없는 이름** 하나                 ← 순응 부정통제

판정(사전 고정·계기 = "checking 크레딧 도구를 부르려 하는가"가 아니라 **검색하는가 ∨ 실행 도구를
호출하려 하는가**. 무엇이 정답인지는 프로브가 말하지 않는다):
  A_REF ≤2                      → 포기/대체 재현 확인(이하 팔의 전제)
  B_QUERIES ≥6                  → **자기 질의 전사만으로 열린다** = 가장 싼 레버
  C_UNREAD ≥6 ∧ B ≤2            → 미열람 존재의 표면화가 레버
  D_NAMES ≥6 ∧ B,C ≤2 ∧ E ≤2    → 이름이 있어야 열린다(발견 축·검색 축 아님)
  D_NAMES ≥6 ∧ E_NEG ≥3         → 이름 순응 = 귀속 불가(레버 주장 금지)
  전 팔 ≤2                      → 검색-누락은 전달로 안 닫힌다 ⇒ 학습 축([[13]])

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x315_search_omission_iso.py [N]
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
import t2_forensic as F                                           # noqa: E402
import x313_bailout_iso as B                                      # noqa: E402

TAG, TASK, CUT = "bank_t7290_a_20260814m", "task_072", 54
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
TOOLNAME = re.compile(r"\b[a-z][a-z_]{5,}_\d{4}\b")
FAKE = "process_refunds_5140"      # KB 에 없는 이름(모델이 지어낸 `process_refunds` 형태를 본뜸)
SEARCH_WORDS = ("KB_search", "knowledge base", "search")


def kb_names():
    """KB 문서에 **실재하는** 도구 이름 — 출처는 환경의 지식베이스이지 채점표가 아니다([[23]])."""
    out = set()
    for p in sorted(glob.glob(os.path.join(DOCS, "*.json"))):
        try:
            out |= set(TOOLNAME.findall(io.open(p, encoding="utf-8", errors="replace").read()))
        except Exception:
            continue
    return out


def kb_queries(sim, cut):
    """지금까지 낸 KB 질의 — 순수 전사(판단 0)."""
    byid, qs = {}, []
    for m in (sim.get("messages") or [])[:cut]:
        for tc in (m.get("tool_calls") or []):
            byid[tc.get("id")] = (F.nameof(tc), F.argsof(tc))
        if m.get("role") == "tool" and "KB_search" in str(byid.get(m.get("id"), ("",))[0]):
            q = byid[m.get("id")][1].get("query")
            if q:
                qs.append(str(q))
    return qs


def called(sim, cut):
    out = set()
    for m in (sim.get("messages") or [])[:cut]:
        for tc in (m.get("tool_calls") or []):
            out.add(F.inner_name(F.argsof(tc)) or F.nameof(tc))
    return out


def searched(r):
    t = str(r.get("content") or "")
    return any(w.lower() in t.lower() for w in SEARCH_WORDS)


def hit(r):
    """열림 = **검색하려 하거나** 실행 도구를 부르려 한다(이관/포기 아님). 정답 지정 0."""
    t = str(r.get("content") or "")
    if re.search(r"(?i)(not available to me|cannot|can't|unable|transfer you to a human)", t) \
            and not searched(r):
        return False
    return bool(searched(r) or TOOLNAME.search(t))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sim = next(s for s in F.scored(TAG) if F.task_id(s) == TASK)
    base = "\n".join([B.HEAD, "", B.transcript(sim, CUT)])
    qs = kb_queries(sim, CUT)
    names = sorted(kb_names() - called(sim, CUT))
    ndocs = len(glob.glob(os.path.join(DOCS, "*.json")))
    print("x315 · %s/%s · cut=%d · 본문 %d자 · KB질의 %d건 · KB이름 %d개(미호출) · 문서 %d건 · n=%d\n"
          % (TAG, TASK, CUT, len(base), len(qs), len(names), ndocs, n))
    if not names:
        print("⚠KB 문서를 못 읽었다(리모트에서 실행하라) — D_NAMES 불가")

    def note(txt):
        return base + "\n\n[note] " + txt

    arms = [
        ("A_REF", base),
        ("B_QUERIES", note("The only knowledge-base queries issued so far in this conversation "
                           "are: %s." % "; ".join('"%s"' % q for q in qs))),
        ("C_UNREAD", note("The knowledge base contains %d documents; the queries issued so far "
                          "have returned only part of them." % ndocs)),
    ]
    if names:
        arms.append(("D_NAMES", note("Tool names that appear in the knowledge base and have not "
                                     "been called in this conversation: %s." % ", ".join(names))))
    arms.append(("E_NEG", note("Tool names that appear in the knowledge base and have not been "
                               "called in this conversation: %s." % FAKE)))
    res = {}
    for label, body in arms:
        k = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, None, 0.0 if i == 0 else 0.7, 1200)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            ok = hit(r)
            k += ok
            cnt["search" if searched(r) else ("act" if ok else "give-up/other")] += 1
            if label == "E_NEG" and FAKE in str(r.get("content") or ""):
                cnt["fake-named"] += 1
            print("    [%s %02d] %s" % (label, i, "HIT" if ok else "-"), flush=True)
        res[label] = k
        print("%-10s %d/%d · %s\n" % (label, k, n, dict(cnt)))
    print("판정(사전 고정): A≤2 전제 · B≥6 → 자기 질의 전사만으로 열림 · C≥6∧B≤2 → 미열람 표면화 · "
          "D≥6∧B,C≤2∧E≤2 → 이름이 있어야 열림 · D≥6∧E≥3 → 귀속 불가 · 전 팔 ≤2 → 학습 축")
    print("측정치: " + " · ".join("%s=%d" % (k, v) for k, v in res.items()))


if __name__ == "__main__":
    main()
