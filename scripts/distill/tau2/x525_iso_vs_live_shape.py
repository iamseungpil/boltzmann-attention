# -*- coding: utf-8 -*-
r"""x525 — 전사 결손이 **재료의 자리** 때문인가를 이등분한다 (2026-08-24·무료·x524 후속)

## 관측 (x524 + 라이브 계기)
계약상 기대 행수를 원장에서 닫힌 술어로 계산하면 chk_1 18 · chk_2 16 · chk_3 16 · chk_4 16 이다
(인출 수 + 같은 날 수수료 2건인 날 수). 라이브 서브는 **18 / 14 / 17 / 17** 을 넘겼고(t7348 양
trial 동일), x524 격리(A_live)는 chk_2 에서 **16** 을 냈다 — **같은 6,752자 원문**을 받고서다
(라이브 계기 축자: `[T2_SG_ISOLATE] sub-view: record dump kept whole (6752 chars)`).

⇒ 재료는 같다. 다른 것은 **호출 형태**다. 코드에서 확인한 라이브 형태
(`t2_scaffold_get.py:766-770`):

    prompt = instructions + "\n\n=== REFERENCE ===\n" + json(ref) + "\n\n" + answer_format
    → 원장은 프롬프트에 **없다**. 서브가 getter 를 부르고 원장은 **도구 결과 메시지**로 온다.

같은 파일 :764 주석 축자: *"지시(형식 포함)가 재료보다 **앞**이다 — C578: 위치 하나가
26/26 ↔ 0/26 을 갈랐다."* ⇒ 자리는 이 코퍼스에서 이미 성적을 가른 축이다.

## 팔 (한 번에 하나만 라이브 쪽으로 민다)
    A_probe    x524 그대로 — instructions + params + 원장이 **user 메시지 안에**
    B_fmt      A + A2 `answer_format` 을 우리 임시 문구 대신 사용(원장은 여전히 user 안)
    C_toolmsg  **라이브 형태** — user(instructions+REFERENCE+answer_format·원장 없음)
               + assistant(tool_call) + tool(원장) 3메시지
  ⇒ C 가 라이브 행수(14/17/17)를 재현하면 원인은 **재료의 자리**이고 처방은 배달 형태다.
     C 도 계약값을 내면 원인은 다른 데 있다(라운드 수·이전 문맥·도구 목록).

## 채점 (닫힌 술어만·gold 미접촉)
    rows        산출 배열 길이 · expect = 인출 수 + 중복 수수료 일수
    ids_ok      낸 transaction_id 가 원장에 실재하는가
    cover       원장 인출 중 **그 날짜/그 인출에 대응하는 행이 있는가** — 인출 id 또는 그 날짜의
                수수료 id 중 하나라도 산출에 있으면 덮인 것으로 센다(A2 계약이 둘 다 허용한다)
    emitted     낸 id 전량을 그대로 저장한다(어느 행이 빠졌는지 사후에 보게)

사용: (리모트·cwd=scripts/distill/tau2) py -3 x525_iso_vs_live_shape.py --port 8140 --n 4
"""
import argparse
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x524_atm_row_transcription_iso as X   # noqa: E402  (정본 재사용·사본 금지 [[67]])

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RE_ID = re.compile(r"^\s*transaction_id:\s*(\S+)\s*$", re.M)
RE_TY = re.compile(r"^\s*type:\s*(\S+)\s*$", re.M)
RE_DT = re.compile(r"^\s*date:\s*(\S+)\s*$", re.M)
RE_ACC = re.compile(r"Transactions for account\s+(\S+)")


def records(text):
    """원장 → [(id, type, date)] (닫힌 술어·env 형식 그대로)."""
    out = []
    for b in re.split(r"\n(?=\s*\d+\.\s+Record ID:)", text):
        i, t, d = RE_ID.search(b), RE_TY.search(b), RE_DT.search(b)
        if i and t:
            out.append((i.group(1), t.group(1), d.group(1) if d else ""))
    return out


def expectation(text):
    """계약상 기대 행수 = 인출 수 + (같은 날 수수료가 2건 이상인 날의 초과분)."""
    recs = records(text)
    w = [r for r in recs if r[1] == "atm_withdrawal"]
    fees = [r for r in recs if r[1] == "atm_fee"]
    byday = {}
    for _, _, dt in fees:
        byday[dt] = byday.get(dt, 0) + 1
    extra = sum(v - 1 for v in byday.values() if v > 1)
    return len(w) + extra, w, fees


def coverage(emitted, w, fees):
    """인출이 덮였나 — 그 인출 id 또는 같은 날 수수료 id 가 산출에 있으면 덮인 것."""
    es = set(str(x) for x in emitted)
    covered = 0
    for wid, _, wdt in w:
        if wid in es or any(fid in es for fid, _, fdt in fees if fdt == wdt):
            covered += 1
    return covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--gz", default=os.path.join(X.SIMS, "bank_t7348_halfB_20260824.results.json.gz"))
    ap.add_argument("--task", default="task_074")
    ap.add_argument("--seed", default="373753")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--arms", default="A_probe,B_fmt,C_toolmsg")
    ap.add_argument("--out", default=os.path.join(X.REP, "x525_iso_vs_live_shape_2026_08_24.json"))
    a = ap.parse_args()

    decl = X.a2_decl()
    iso = decl.get("isolate") or {}
    instr = iso.get("instructions") or ""
    afmt = iso.get("answer_format") or ""
    params = ((decl.get("params") or {}).get("transactions")) or ""
    getter = (iso.get("getter_tools") or ["call_discoverable_agent_tool"])[0]
    if not instr or not afmt:
        raise SystemExit("A2 선언에 instructions/answer_format 이 없다 — 중단")

    leds = X.ledgers(a.gz, a.task, a.seed)
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    rows = []
    for idx, text in leds:
        exp, w, fees = expectation(text)
        acc = RE_ACC.search(text)
        acc = acc.group(1) if acc else "?"
        print("[x525] --- msg[%d] %s · 인출 %d · 수수료 %d · 계약 기대 %d행"
              % (idx, acc, len(w), len(fees), exp))
        ref = {"account_id": acc}
        for arm in arms:
            for k in range(a.n):
                if arm == "A_probe":
                    msgs = [{"role": "user", "content":
                             instr + "\n\n# Field contract\ntransactions: " + params +
                             "\n\n# Account transaction history\n" + text +
                             "\n\nReply with ONE JSON array only: the `transactions` value."}]
                elif arm == "B_fmt":
                    msgs = [{"role": "user", "content":
                             instr + "\n\n=== REFERENCE ===\n" +
                             json.dumps(ref, ensure_ascii=False, indent=1) +
                             "\n\n" + afmt + "\n\n=== RECORDS ===\n" + text}]
                elif arm == "C_toolmsg":
                    msgs = [
                        {"role": "user", "content":
                         instr + "\n\n=== REFERENCE ===\n" +
                         json.dumps(ref, ensure_ascii=False, indent=1) + "\n\n" + afmt},
                        {"role": "assistant", "content": "",
                         "tool_calls": [{"id": "c1", "type": "function",
                                         "function": {"name": getter, "arguments": json.dumps(
                                             {"agent_tool_name": "get_bank_account_transactions_9173",
                                              "account_id": acc}, ensure_ascii=False)}}]},
                        {"role": "tool", "tool_call_id": "c1", "content": text},
                    ]
                else:
                    continue
                try:
                    out = X.chat(a.port, msgs)
                except Exception as e:
                    rows.append({"msg": idx, "acc": acc, "arm": arm, "k": k, "error": str(e)[:200]})
                    print("      %-10s k=%d ERROR %r" % (arm, k, str(e)[:70]))
                    continue
                got = X.parse_rows(out)
                ids = [str((r or {}).get("transaction_id", "")) for r in (got or [])]
                real = [i2 for i2 in ids if i2 and i2 in text]
                cov = coverage(ids, w, fees)
                dup = sum(1 for r in (got or []) if isinstance(r, dict) and r.get("duplicate_of"))
                rows.append({"msg": idx, "acc": acc, "arm": arm, "k": k, "expect": exp,
                             "rows": (len(got) if got is not None else None),
                             "ids_real": len(real), "cover": cov, "withdrawals": len(w),
                             "dup_of": dup, "emitted": ids})
                print("      %-10s k=%d rows=%s/%d cover=%d/%d ids_real=%d dup=%d"
                      % (arm, k, (len(got) if got is not None else "parse_fail"), exp,
                         cov, len(w), len(real), dup))

    with io.open(a.out, "w", encoding="utf-8") as f:
        json.dump({"probe": "x525", "date": "2026-08-24",
                   "question": "전사 결손이 재료의 자리(도구 메시지) 때문인가",
                   "live_reference": {"chk_1": 18, "chk_2": 14, "chk_3": 17, "chk_4": 17,
                                      "source": "t7348 halfB 로그 [T2_SG_ISOLATE] operand-size (양 trial 동일)"},
                   "arms": arms, "n": a.n, "rows": rows}, f, ensure_ascii=False, indent=1)
    print("\n[x525] wrote %s" % a.out)
    print("[x525] 요약 — arm × 계좌 · rows(기대) · cover")
    for arm in arms:
        for idx, _ in leds:
            rs = [r for r in rows if r["arm"] == arm and r["msg"] == idx and r.get("rows") is not None]
            if rs:
                print("  %-10s msg[%2d] %s  rows=%s (기대 %d) · cover %s/%d"
                      % (arm, idx, rs[0]["acc"], [r["rows"] for r in rs], rs[0]["expect"],
                         [r["cover"] for r in rs], rs[0]["withdrawals"]))


if __name__ == "__main__":
    main()
