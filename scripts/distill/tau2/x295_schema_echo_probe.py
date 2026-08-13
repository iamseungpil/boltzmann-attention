# -*- coding: utf-8 -*-
r"""x295 — G1 스키마-에코 격리: env "unexpected keyword" 에러에 unlock 의 Parameters 블록을
에코하면 다음 호출이 선언 스키마로 수렴하는가 (082 실물).

배경(bank_n97_gpu1_batch_05_20260805 082 t1 전수 정독): unlock 출력(msg91)이 17개 파라미터
전체 스키마+enum 유효값을 나열했는데, 15msg 뒤 첫 dispute 호출(msg106)은 **발명 스키마**
(merchant/amount/date/nature_of_dispute…) — env 에러가 한 번에 한 kwarg 만 지목해 **한 겹씩
벗기며 8~18호출** 소모(대조: enum 에러는 유효값 목록 동봉 → 1호출 교정 = env 자체가 deny
품질의 자연 실험·[[64]]). 대화-내 실존 텍스트의 행동-지점 재제시가 레버 후보(판단 0).

셀 2 (n=8·082 msg106(첫 발명-스키마 호출) 직후 컷):
  A_CUR   + env 에러 원문만("unexpected keyword argument 'merchant'")
  B_ECHO  + env 에러 원문 + unlock 의 "Parameters: …" 블록 축자(대화 msg91 실존 텍스트)

계기: 다음 호출(file_debit_card_transaction_dispute)의 인자 키가 선언 필수 키를 몇 개
포함하는가 — full = 핵심 8키(transaction_id·account_id·card_id·user_id·dispute_category·
transaction_date·discovery_date·disputed_amount) 전부 포함 ∧ 발명 키 0.
판정(사전 고정): A_CUR full ≤2/8 ∧ B_ECHO full ≥6/8 → 스키마-에코 deny 출시(도메인-일반:
discoverable 도구 × env unexpected-kwarg/missing-args 에러 → unlock Parameters 블록 에코).
A_CUR ≥6/8 → env 에러만으로 충분 = 보류. 중간(3~5) → n=16 재측정 1회.

실행(리모트·8141·유료 런 종료 후): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x295_schema_echo_probe.py [N]
"""
import collections
import gzip
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
import x241_uncalled_unlock_probe as U                            # noqa: E402
import x291_checking_pick_iso as B                                # noqa: E402

ARCH = ("/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/"
        "sim_results/bank_n97_gpu1_batch_05_20260805.results.json.gz")
TASK = "task_082"
TOOL = "file_debit_card_transaction_dispute_6281"
REQ = ("transaction_id", "account_id", "card_id", "user_id", "dispute_category",
       "transaction_date", "discovery_date", "disputed_amount")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    d = json.load(gzip.open(ARCH, "rt", encoding="utf-8"))
    sims = [s for s in (d.get("simulations") or d.get("results") or d)
            if s.get("task_id") == TASK and s.get("trial") == 1]
    sim = sims[0]
    msgs = sim["messages"]
    # 컷 = 첫 발명-스키마 dispute 호출 어시스턴트 턴 직후 · 스키마 블록 = unlock 출력 축자
    cut = schema = err = None
    for i, m in enumerate(msgs):
        c = str(m.get("content") or "")
        if schema is None and m.get("role") == "tool" and "Tool unlocked: %s" % TOOL in c:
            mm = re.search(r"Parameters:.*", c, re.S)
            schema = mm.group(0) if mm else None
        if cut is None and m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                if TOOL in str(tc) and "merchant" in str(tc):
                    cut = i + 1
                    break
        if cut is not None and err is None and m.get("role") == "tool" \
                and "unexpected keyword" in c:
            err = " ".join(c.split())
            break
    if not (cut and schema and err):
        print("컷/스키마/에러 없음", cut, bool(schema), bool(err))
        return
    tools = U.tools_of(sim)
    base = B.render(msgs[:cut], {})
    base = base[:base.rfind("\n[user] ")]
    print("082 t1 cut=%d · 스키마 %d자 · n=%d · URL=%s\n" % (
        cut, len(schema), n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    for label, extra in (("A_CUR", err),
                         ("B_ECHO", err + "\nThe tool's declared parameters are:\n" + schema)):
        body = base + "\n[tool] %s" % extra
        full = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            got = None
            for tc in (r.get("tool_calls") or []):
                if TOOL not in str(tc):
                    continue
                a = tc.get("arguments") or (tc.get("function") or {}).get("arguments") or {}
                try:
                    a = json.loads(a) if isinstance(a, str) else a
                    inner = a.get("arguments")
                    inner = json.loads(inner) if isinstance(inner, str) else (inner or a)
                    got = set((inner or {}).keys())
                except Exception:
                    got = set()
                break
            if got is None:
                cnt["(no dispute call)"] += 1
                continue
            hit = sum(1 for k in REQ if k in got)
            invented = [k for k in got if k in ("merchant", "amount", "date",
                                                "nature_of_dispute", "additional_info",
                                                "written_statement", "dispute_reason")]
            ok = hit == len(REQ) and not invented
            full += ok
            cnt["full" if ok else "keys=%d inv=%d" % (hit, len(invented))] += 1
        print("%-7s full-schema %d/%d · %s" % (label, full, n, dict(cnt)))
    print("\n※ 판정(사전 고정): A_CUR ≤2/8 ∧ B_ECHO ≥6/8 → 스키마-에코 deny 출시(도메인-일반)."
          " A_CUR ≥6/8 → 보류. 중간(3~5) → n=16 1회.")


if __name__ == "__main__":
    main()
