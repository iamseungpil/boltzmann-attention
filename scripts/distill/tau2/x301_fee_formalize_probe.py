# -*- coding: utf-8 -*-
r"""x301 — fee 도구 입력 형식화 격리: rho 모순·중복 수수료를 모델이 형식화하는가 (073/072 표적).

배경(t7281 nt=4·073 4시행 **전부 8/11 동일** + 오프라인 검산·gold 미접촉[db 원장 직독]):
  계좌1 정답 차액 = $1.50(WELLS 초과) + $5.00(SHINHAN foreign 초과) + **$3.00(RHO-BANK 인출에
  NON-RHO 수수료)** = $9.50. 도구는 앞 둘만 내 **$6.50**.
  계좌2 = $3.00(WOORI 초과) + **$3.00(RHO-BANK 라인)** + **$3.00(같은 인출에 fee 2건)** = $9.00,
  도구는 $3.00.
⇒ 산술은 옳고(엔진), **입력 형식화**가 두 축에서 빠진다:
  ⓐ network 를 짝 인출의 description(RHO-BANK…)이 아니라 **fee 라인의 자기-라벨**
     ("NON-RHO ATM FEE")에서 읽는다 — 그 라벨이야말로 감사 대상이다(자기-참조 함정).
  ⓑ 한 인출에 fee 2건(집합-수준 관찰)은 라인별 스키마로는 표현조차 안 된다.

셀 4 (n=8·입력 = 073 계좌1·2의 원장 행 축자(db 직독·gold 미접촉)·계기 = 형식화 JSON 검사):
  A_CUR    현행 도구 param 설명 축자
  B_WARN   + 라벨-불신 한 줄(*"the fee line's own label is what you are auditing - read the
           network from the paired withdrawal's description"*)
  C_DUP    B_WARN + 중복 축 표면화 요청(같은 인출에 fee 2건이면 그 사실을 표기)
  D_NEG    A_CUR + 무관 경고문(길이 통제·[[57]])

계기: rho_ok = RHO-BANK 짝 fee 라인의 network 가 'rho' · dup_ok = 중복 fee 2건을 구분 표기.
판정(사전 고정): A_CUR rho_ok ≤2/8 ∧ B_WARN ≥6/8 → 문면 보강 출시(param 설명 개정).
  C_DUP dup_ok ≥6/8 → 중복 축도 모델이 낸다(엔진 확장 불요·[[62]] 최소).
  C_DUP ≤2/8 → 중복은 집합-술어라 모델이 못 냄 = 엔진 확장 후보로 별도 사전등록.
  D_NEG ≥6/8 → 길이/경고 자체가 인자 = 프로브 무효.

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x301_fee_formalize_probe.py [N]
"""
import collections
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

DB = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/db.json"
ACCTS = ("chk_kj93a7b2e1_1", "chk_kj93a7b2e1_2")
A2 = "a2/banking_knowledge.specific.json"
WARN = ("IMPORTANT: the fee line's own description (e.g. 'NON-RHO ATM FEE') is exactly what "
        "you are auditing - it may be wrong. Determine 'network' ONLY from the paired "
        "withdrawal's description: a RHO-BANK machine is 'rho' (a fee on a RHO-BANK withdrawal "
        "is itself the error you are looking for), a machine outside the U.S. is 'foreign', any "
        "other bank's machine is 'non_rho'.")
DUP = ("Also: if TWO fee lines belong to the SAME withdrawal, include both, and add "
       "\"duplicate_of\": \"<the other fee line's transaction_id>\" on the second one.")
NEG = ("Note: this bank's records are stored in reverse chronological order in some views, and "
       "amounts are shown in United States dollars with two decimal places throughout.")


def rows_of():
    db = json.load(io.open(DB, encoding="utf-8"))
    t = db["bank_account_transaction_history"]
    data = t.get("data") if isinstance(t, dict) else t
    out = []
    for r in (data or {}).values():
        if str(r.get("account_id")) in ACCTS and str(r.get("type")) in (
                "atm_withdrawal", "atm_fee"):
            out.append(r)
    return sorted(out, key=lambda r: (r["account_id"], str(r.get("date")),
                                      str(r.get("transaction_id"))))


def params_text():
    a = json.load(io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), A2),
                          encoding="utf-8"))
    e = next(t for t in a["scaffold_get_tools"] if t["name"] == "get_atm_fee_discrepancies")
    return e["params"]["transactions"]


def judge(txt):
    """rho_ok = RHO-BANK 짝 fee 가 'rho' · dup_ok = 중복 표기(둘 다 기계 검사)."""
    m = re.search(r"\[.*\]", str(txt or ""), re.S)
    try:
        arr = json.loads(m.group(0)) if m else []
    except Exception:
        return False, False, "parse-fail"
    if not isinstance(arr, list) or not arr:
        return False, False, "empty"
    by = {str(r.get("transaction_id")): r for r in arr if isinstance(r, dict)}
    # RHO 짝 fee 라인 = 원장 실측: 계좌1 btxn_kj07s5t6u7v9 · 계좌2 btxn_kj14u3v4w5x7
    rho_ids = ["btxn_kj07s5t6u7v9", "btxn_kj14u3v4w5x7"]
    hit = [str((by.get(i) or {}).get("network") or "").lower() for i in rho_ids]
    rho_ok = all(h == "rho" for h in hit if h) and any(h == "rho" for h in hit)
    dup_ok = any("duplicate" in json.dumps(r, ensure_ascii=False).lower() for r in arr)
    return rho_ok, dup_ok, "n=%d rho=%s" % (len(arr), hit)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    rows = rows_of()
    ledger = "\n".join(
        "%s | %s | %s | %s | %s" % (r["transaction_id"], r["date"], r["amount"],
                                    r["type"], r.get("description"))
        for r in rows)
    base_p = params_text()
    print("073 계좌2개 · 원장 %d행 · n=%d · URL=%s\n" % (
        len(rows), n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    arms = (("A_CUR", base_p), ("B_WARN", base_p + "\n" + WARN),
            ("C_DUP", base_p + "\n" + WARN + "\n" + DUP), ("D_NEG", base_p + "\n" + NEG))
    for label, spec in arms:
        body = ("You are auditing ATM fees on two checking accounts. Here are the account's "
                "transaction records:\n\n%s\n\nBuild the `transactions` argument for the "
                "get_atm_fee_discrepancies tool. The tool's specification for that argument "
                "is:\n%s\n\nReply with ONLY the JSON array." % (ledger, spec))
        cnt = collections.Counter()
        rho = dup = 0
        for i in range(n):
            try:
                r = chat(body, None, 0.0 if i == 0 else 0.7, 1800)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            a, b, info = judge(r.get("content"))
            rho += a
            dup += b
            cnt[info[:28]] += 1
            print("  [%s %02d] rho=%s dup=%s %s" % (label, i, a, b, info[:60]))
        print("%-7s rho_ok %d/%d · dup_ok %d/%d · %s\n" % (label, rho, n, dup, n, dict(cnt)))
    print("※ 판정(사전 고정): A_CUR rho ≤2 ∧ B_WARN rho ≥6 → param 문면 개정 출시."
          " C_DUP dup ≥6 → 중복 축도 모델 몫 · ≤2 → 엔진 확장 별도 사전등록."
          " D_NEG rho ≥6 → 무효(길이 인자).")


if __name__ == "__main__":
    main()
