# -*- coding: utf-8 -*-
r"""x295b — G1 스키마-에코 재격리 (x295 계기 결함 3건 수리판·2026-08-14).

배경(C466 G1 포렌식·082 t1 실물): 다인자 write 도구를 **발명 스키마**로 부르고 env 가 한 번에
**kwarg 하나씩만** 지목해 8~18 호출을 태운다(대조: enum 에러는 유효값 목록을 동봉해 1호출
교정 — env 자체가 deny 품질의 자연 실험·[[64]]). unlock 출력에는 **17개 파라미터 전체 명세가
이미 인쇄**돼 있으나 15 msg 뒤에는 쓰이지 않는다(근접성 = x287b 교훈).

x295(1차) 무효 사유 — 자기교정([[08]]):
  ⓐ 계기가 "dispute 호출 아님"을 전부 한 통에 담아 **무엇을 했는지 못 봤다**(프로즈/타 도구/
     빈 응답 구분 불가) ⓑ 컷이 발명-호출 **여러 번 뒤**라 이미 churn 중턱 ⓒ 통제 팔 부재
     (텍스트 양이 인자인지 **맞는 스키마**가 인자인지 못 가름).

셀 3 (n=8·컷 = **첫** 발명-스키마 호출 + 그 env 에러 직후):
  A_CUR    env 에러 원문만(라이브 축자)
  B_ECHO   + 그 도구의 unlock 출력 "Parameters: …" 블록 축자(대화 내 실존 텍스트)
  D_OTHER  + **무관 도구**의 Parameters 블록 동수(길이·형식 통제·[[57]])

계기(전건 라벨 기록): full = 필수 8키 전부 ∧ 발명 키 0 · partial = dispute 호출이나 키 미달 ·
  other:<도구> · (text) · (empty).
판정(사전 고정): A_CUR full ≤2/8 ∧ B_ECHO full ≥6/8 ∧ D_OTHER full ≤2/8 → 스키마-에코 출시
  (도메인-일반: discoverable 도구 × env kwarg/missing-args 에러 → 그 도구 unlock 명세 재표면화).
  A_CUR ≥6/8 → env 에러만으로 충분(보류). B_ECHO ≤2/8 → 근접 재제시로 안 닫힘(다른 축).
  D_OTHER ≥3/8 → 텍스트 양이 인자 = 프로브 무효. 중간(3~5) → n=16 재측정 1회.

실행(리모트·8141·유료 런 종료 후): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x295b_schema_echo_probe.py [N]
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
OTHER_TOOL = "get_bank_account_transactions_9173"     # 통제용(같은 대화에서 unlock 됨)
REQ = ("transaction_id", "account_id", "card_id", "user_id", "dispute_category",
       "transaction_date", "discovery_date", "disputed_amount")
INVENTED = ("merchant", "amount", "date", "nature_of_dispute", "additional_info",
            "written_statement", "dispute_reason", "description", "disputes")
ERRPAT = re.compile(r"unexpected keyword|Invalid arguments|missing \d+ required")


def schema_of(msgs, tool):
    """그 도구의 unlock 출력에서 Parameters 블록 축자(대화 내 실존 텍스트만)."""
    for m in msgs:
        c = str(m.get("content") or "")
        if m.get("role") == "tool" and ("Tool unlocked: %s" % tool) in c:
            mm = re.search(r"Parameters:.*", c, re.S)
            if mm:
                return mm.group(0).strip()
    return None


def label(msg):
    """전건 라벨([[08]] 정독용) — 무엇을 했는지 뭉뚱그리지 않는다."""
    for tc in (msg.get("tool_calls") or []):
        blob = str(tc)
        nm = str(tc.get("name") or (tc.get("function") or {}).get("name") or "")
        if TOOL in blob:
            a = tc.get("arguments") or (tc.get("function") or {}).get("arguments") or {}
            try:
                a = json.loads(a) if isinstance(a, str) else a
                inner = a.get("arguments")
                inner = json.loads(inner) if isinstance(inner, str) else (inner or a)
                keys = set((inner or {}).keys())
            except Exception:
                keys = set()
            hit = sum(1 for k in REQ if k in keys)
            inv = [k for k in keys if k in INVENTED]
            if hit == len(REQ) and not inv:
                return "full"
            return "partial(req=%d,inv=%d)" % (hit, len(inv))
        return "other:%s" % (nm or "?")
    return "(text)" if msg.get("content") else "(empty)"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    d = json.load(gzip.open(ARCH, "rt", encoding="utf-8"))
    sims = [s for s in (d.get("simulations") or d.get("results") or d)
            if s.get("task_id") == TASK and s.get("trial") == 1]
    if not sims:
        print("궤적 없음")
        return
    sim = sims[0]
    msgs = sim["messages"]
    # 컷 = **첫** 발명-스키마 호출 직후 + 그 결과 에러
    cut = err = None
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        blob = json.dumps(m.get("tool_calls") or [], ensure_ascii=False)
        if TOOL in blob and any(k in blob for k in INVENTED):
            for j in range(i + 1, min(i + 4, len(msgs))):
                c = str(msgs[j].get("content") or "")
                if msgs[j].get("role") == "tool" and ERRPAT.search(c):
                    cut, err = i + 1, " ".join(c.split())
                    break
        if cut:
            break
    sch = schema_of(msgs, TOOL)
    oth = schema_of(msgs, OTHER_TOOL)
    if not (cut and err and sch):
        print("컷/에러/스키마 없음:", cut, bool(err), bool(sch))
        return
    tools = U.tools_of(sim)
    base = B.render(msgs[:cut], {})
    base = base[:base.rfind("\n[user] ")] if "\n[user] " in base else base
    print("082 t1 cut=%d(첫 발명호출) · 스키마 %d자 · 통제 %s · n=%d · URL=%s\n" % (
        cut, len(sch), (len(oth) if oth else "없음"), n,
        os.environ.get("T2_PROBE_URL", "localhost:8140")))
    arms = [("A_CUR", err),
            ("B_ECHO", err + "\nThe tool's declared parameters are:\n" + sch)]
    if oth:
        arms.append(("D_OTHER", err + "\nThe tool's declared parameters are:\n" + oth))
    for lab, extra in arms:
        body = base + "\n[tool] " + extra
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            lb = label(r)
            cnt[lb] += 1
            print("  [%s %02d] %s" % (lab, i, lb))
        print("%-8s full %d/%d · %s\n" % (lab, cnt.get("full", 0), n, dict(cnt)))
    print("※ 판정(사전 고정): A_CUR ≤2 ∧ B_ECHO ≥6 ∧ D_OTHER ≤2 → 스키마-에코 출시."
          " A_CUR ≥6 → 보류 · B_ECHO ≤2 → 다른 축 · D_OTHER ≥3 → 무효(길이 인자).")


if __name__ == "__main__":
    main()
