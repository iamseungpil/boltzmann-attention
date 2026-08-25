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


def regroup(text, mode):
    """레코드 덤프를 **재배열**한다 — 내용 변경 0(블록을 축자 그대로 옮길 뿐).

    ★왜 (2026-08-25·t7348 074 t1 실물): 원장에 `btxn_ar_dg_17f = -2.25` 가 **실재**하는데
      우리 도구는 `btxn_ar_dg_17 (charged $0.00, documented fee $2.25, difference $-2.25)` 로
      보고했다 — 서브가 그 수수료 줄을 **짝지어 주지 못했다**. 유령 음수 −2.25 가 환불액을
      깎았고, 되돌리면 chk_3 2.50+2.25=**4.75**=gold · chk_4 1.45+2.25=**3.70**=gold 로
      센트까지 맞는다. 드롭이 없던 chk_1 은 도구합 27.0 = gold 였고 모델이 그대로 제출했다.
    ⇒ 가설: 덤프가 **최신순**이라 `_17` 은 1번 행이고 그 수수료는 뒤쪽 어딘가에 떨어져 있다.
      각 인출 **바로 뒤**에 그 날 수수료를 놓으면 짝이 보이는가.
    ⚠엔진이 쓰는 문장 0 · 값 변경 0 · 판단 0 — 블록의 **순서**만 바꾼다([[59]]·[[62]]③④).

        pair     인출 뒤에 같은 날 수수료(그 다음에 나머지)
        datesort 같은 양의 재배열이되 **짝짓지 않는다**(인출 전량 → 수수료 전량·둘 다 날짜순)
                 = [[57]] 부정통제. 이것이 pair 만큼 움직이면 이득은 짝이 아니라 정렬이다.
    """
    blocks, head = [], ""
    parts = re.split(r"\n(?=\s*\d+\.\s+Record ID:)", text)
    if parts and not RE_ID.search(parts[0]):
        head, parts = parts[0], parts[1:]
    for b in parts:
        i, t, d = RE_ID.search(b), RE_TY.search(b), RE_DT.search(b)
        if i and t:
            blocks.append((i.group(1), t.group(1), d.group(1) if d else "", b))
    if not blocks:
        return text
    ws = [b for b in blocks if b[1] == "atm_withdrawal"]
    fs = [b for b in blocks if b[1] == "atm_fee"]
    rest = [b for b in blocks if b[1] not in ("atm_withdrawal", "atm_fee")]
    def _inter(rev):
        out, used = [], set()
        for w in sorted(ws, key=lambda x: x[2], reverse=rev):
            out.append(w)
            for f in fs:
                if f[2] == w[2] and f[0] not in used:
                    out.append(f)
                    used.add(f[0])
        return out + [f for f in fs if f[0] not in used]

    # ★2026-08-25 2차: x535 에서 **부정통제로 넣은 `datesort` 가 네 계좌를 전부 닫았다**
    #   (rows=expect · fee_paired 만점 · ids_real 만점 · n=3 결정론). 그런데 그 팔은 두 변수를
    #   동시에 바꾼다 — **방향**(오래된 순 ↔ 최신 순)과 **묶음**(인출전량→수수료전량 ↔ 번갈아).
    #   원본은 *최신순 + 번갈아* 다. 배선하기 전에 그 둘을 갈라야 한다([[57]]).
    #   그리고 *무의미한 재배열*로도 닫히면 이득은 순서가 아니라 **다시 렌더링한 것**이다 —
    #   `scramble` 이 그 통제다(id 문자열 뒤집기 정렬·결정론적·의미 0).
    if mode == "pair" or mode == "old_inter":
        out = _inter(False) + rest
    elif mode == "new_inter":
        out = _inter(True) + rest
    elif mode == "datesort" or mode == "old_group":
        out = sorted(ws, key=lambda x: x[2]) + sorted(fs, key=lambda x: x[2]) + rest
    elif mode == "new_group":
        out = (sorted(ws, key=lambda x: x[2], reverse=True)
               + sorted(fs, key=lambda x: x[2], reverse=True) + rest)
    elif mode == "scramble":
        out = sorted(blocks, key=lambda x: x[0][::-1])
    else:
        return text
    body = "\n".join("%d. %s" % (k + 1, b[3].strip().split(". ", 1)[-1])
                     for k, b in enumerate(out))
    return (head + "\n" + body) if head else body


def fee_paired(emitted, recs):
    """계약대로 **수수료 id 로** 짝지은 인출 수 / 수수료가 있는 인출 수.

    ★기존 `coverage` 로는 이 결손이 안 잡힌다 — 그 술어는 *인출 id 또는 같은 날 수수료 id*
      둘 중 하나면 덮인 것으로 세므로, `_17` 을 **자기 id 로** 낸 산출도 통과시킨다. 그런데
      선언 축자는 *"transaction_id: the id of the atm_fee line paired with this withdrawal;
      if this withdrawal has NO fee line, the withdrawal's own id"* 다. 수수료가 있는데
      인출 id 를 내면 도구가 *"수수료 미부과"* 로 읽고 유령 음수를 만든다(074 실물).
    """
    es = set(str(x) for x in emitted)
    ws = [r for r in recs if r[1] == "atm_withdrawal"]
    fs = [r for r in recs if r[1] == "atm_fee"]
    have = [w for w in ws if any(f[2] == w[2] for f in fs)]
    ok = 0
    for w in have:
        if any(f[0] in es for f in fs if f[2] == w[2]):
            ok += 1
    return ok, len(have)


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
                elif arm == "D_all":
                    # ★교락 해소 (2026-08-24): A_probe 는 `params` 계약과 **형식 문면**을 동시에
                    #   바꿨다. 이 팔은 라이브 형식(A2 answer_format)을 그대로 두고 `params` 만
                    #   더한다 — 16/16 이면 활성 성분은 **필드 계약 텍스트**다.
                    msgs = [{"role": "user", "content":
                             instr + "\n\n=== REFERENCE ===\n" +
                             json.dumps(ref, ensure_ascii=False, indent=1) +
                             "\n\n# Field contract\ntransactions: " + params +
                             "\n\n" + afmt + "\n\n=== RECORDS ===\n" + text}]
                elif arm == "N_wire":
                    # ★배선 검정 (2026-08-25): `t2_scaffold_get` 의 `T2_SG_PROMPT_V2` 가 만드는
                    #   프롬프트를 **그대로** 재현한다 — 라운드1(지시+REFERENCE 평문+필드계약) →
                    #   도구 결과 → 마감(answer_format). 이 팔이 16/16 이면 배선이 검정된 것이다.
                    msgs = [
                        {"role": "user", "content":
                         instr + "\n\n=== REFERENCE ===\naccount_id: " + acc +
                         "\n\n=== FIELD CONTRACT ===\ntransactions: " + params},
                        {"role": "assistant", "content": "",
                         "tool_calls": [{"id": "c1", "type": "function",
                                         "function": {"name": getter, "arguments": json.dumps(
                                             {"agent_tool_name": "get_bank_account_transactions_9173",
                                              "account_id": acc}, ensure_ascii=False)}}]},
                        {"role": "tool", "tool_call_id": "c1", "content": text},
                        # ★2026-08-25 정정: 마감 user 메시지에 **계약 + 원장 + 형식**을 함께 싣는다.
                        #   앞판(형식만)은 chk_2 에서 cover 15/16 이었고, 이기는 팔(J_both·
                        #   K_paramslast, 16/16)과의 유일한 차이가 **원장이 user 메시지 안이냐**였다.
                        {"role": "user", "content":
                         "=== FIELD CONTRACT ===\ntransactions: " + params +
                         "\n\n=== RECORDS ===\n" + text + "\n\n" + afmt},
                    ]
                elif arm == "M_reflast":
                    # ★기전 이등분 ⑵ 위치 (2026-08-25): `D_all` 과 **REFERENCE 위치만** 다르다.
                    #   블록을 원장 뒤로 보낸다 — 커버리지가 돌아오면 원인은 *존재*가 아니라 *자리*다.
                    msgs = [{"role": "user", "content":
                             instr + "\n\n# Field contract\ntransactions: " + params +
                             "\n\n" + afmt + "\n\n=== RECORDS ===\n" + text +
                             "\n\n=== REFERENCE ===\n" +
                             json.dumps(ref, ensure_ascii=False, indent=1)}]
                elif arm == "M_refneutral":
                    # ★기전 이등분 ⑶ 앵커링: 같은 값을 두되 **키 이름만** 중립으로 바꾼다.
                    #   `account_id` 는 *"이것과 매칭하라"* 는 선택 연산으로 읽힐 수 있다.
                    _neutral = {"context": list(ref.values())[0] if ref else ""}
                    msgs = [{"role": "user", "content":
                             instr + "\n\n=== REFERENCE ===\n" +
                             json.dumps(_neutral, ensure_ascii=False, indent=1) +
                             "\n\n# Field contract\ntransactions: " + params +
                             "\n\n" + afmt + "\n\n=== RECORDS ===\n" + text}]
                elif arm == "M_refplain":
                    # ★기전 이등분 ⑷ 형식 모방: 같은 정보를 **JSON 이 아니라 한 문장**으로.
                    #   두 번째 JSON 블록이 출력을 '매칭 요약' 형태로 흉내 내게 하는가.
                    _plain = ("The records below are the transaction history of account %s."
                              % (list(ref.values())[0] if ref else "?"))
                    msgs = [{"role": "user", "content":
                             instr + "\n\n" + _plain +
                             "\n\n# Field contract\ntransactions: " + params +
                             "\n\n" + afmt + "\n\n=== RECORDS ===\n" + text}]
                elif arm == "L_closeask":
                    # ★배선 가능한 형태 (2026-08-25): `I_noref` 가 chk_2 를 닫았지만(cover 16/16)
                    #   라이브는 REFERENCE 로 계좌를 지정해야 getter 를 부른다 — 통째로 뺄 수 없다.
                    #   그래서 **라이브 구조 그대로**(1라운드 REFERENCE + 도구 결과) 두고,
                    #   `_isolate_fetch` 의 **마감 라운드**(도구 없는 마지막 생성)에서만 계약·형식을
                    #   REFERENCE 없이 다시 묻는다. 이기면 수리는 그 마감 라운드 프롬프트 한 곳이다.
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
                        {"role": "user", "content":
                         instr + "\n\n# Field contract\ntransactions: " + params + "\n\n" + afmt},
                    ]
                elif arm == "J_both":
                    # ★단일 변수 팔이 전부 실패했으므로 **조합**을 친다 (2026-08-25):
                    #   이긴 팔 `A_probe` 는 라이브와 세 칸이 다르다 — ⑴REFERENCE 없음 ⑵요구가
                    #   원장 뒤 ⑶params 있음. H·I 가 각각 하나씩만 밀어 실패했으니 셋을 함께 민다.
                    #   ⚠단 형식 문면은 **A2 선언 그대로**(`answer_format`) 유지 — 이식 가능해야
                    #   의미가 있다([[78]] ②: 이식 대상은 선언에 있는 텍스트만).
                    msgs = [{"role": "user", "content":
                             instr + "\n\n# Field contract\ntransactions: " + params +
                             "\n\n=== RECORDS ===\n" + text + "\n\n" + afmt}]
                elif arm == "K_paramslast":
                    # 조합에서 params 위치만 다시 뒤로 — 계약이 **원장 뒤·형식 앞**.
                    msgs = [{"role": "user", "content":
                             instr + "\n\n=== RECORDS ===\n" + text +
                             "\n\n# Field contract\ntransactions: " + params +
                             "\n\n" + afmt}]
                elif arm == "H_asklast":
                    # ★남은 차이 ⑵ (2026-08-24): 이긴 팔 `A_probe` 만 **요구 문장이 원장 뒤**에 있다.
                    #   다른 모든 팔은 형식이 원장 앞이다(라이브도 그렇다). 그 한 칸만 민다 —
                    #   재료·형식 문면은 D_all 과 동일하고 **위치만** 다르다.
                    msgs = [{"role": "user", "content":
                             instr + "\n\n=== REFERENCE ===\n" +
                             json.dumps(ref, ensure_ascii=False, indent=1) +
                             "\n\n# Field contract\ntransactions: " + params +
                             "\n\n=== RECORDS ===\n" + text + "\n\n" + afmt}]
                elif arm == "I_noref":
                    # ★남은 차이 ⑴: 이긴 팔에는 REFERENCE 블록이 없다. 그 한 칸만 뺀다.
                    msgs = [{"role": "user", "content":
                             instr + "\n\n# Field contract\ntransactions: " + params +
                             "\n\n" + afmt + "\n\n=== RECORDS ===\n" + text}]
                elif arm == "F_order":
                    # ★위치 하나 (2026-08-24): `D_all` 은 계약을 **예시 앞**에 뒀고 13~14 였다.
                    #   같은 파일 :764 주석이 이미 이름 붙인 축 — *"지시(형식 포함)가 재료보다 앞이다
                    #   — C578: 위치 하나가 26/26 ↔ 0/26 을 갈랐다"*. 여기선 **계약이 예시 뒤**다:
                    #   모델이 마지막에 본 것이 *한 줄짜리 예시 행*이면 그것을 흉내 내 수수료 단위로
                    #   행을 만든다(chk_2 수수료 13개 → 13행)는 가설을 친다.
                    msgs = [{"role": "user", "content":
                             instr + "\n\n=== REFERENCE ===\n" +
                             json.dumps(ref, ensure_ascii=False, indent=1) +
                             "\n\n" + afmt +
                             "\n\n# Field contract\ntransactions: " + params +
                             "\n\n=== RECORDS ===\n" + text}]
                elif arm in ("D_old_group", "D_new_group", "D_old_inter",
                             "D_new_inter", "N_scramble"):
                    # ★2×2(방향 × 묶음) + 무의미 순서 통제. N_wire 와 **덤프 순서만** 다르다.
                    #   원본 = 최신순 + 번갈아 이므로 `D_new_inter` 가 N_wire 재현이어야 한다
                    #   (계기 생존 검사). `N_scramble` 이 닫으면 이득은 순서가 아니라 재렌더링이다.
                    _m = {"D_old_group": "old_group", "D_new_group": "new_group",
                          "D_old_inter": "old_inter", "D_new_inter": "new_inter",
                          "N_scramble": "scramble"}[arm]
                    _txt = regroup(text, _m)
                    msgs = [{"role": "user", "content":
                             instr + "\n\n=== REFERENCE ===\n" +
                             "\n".join("%s: %s" % (k2, v2) for k2, v2 in ref.items()) +
                             "\n\n# Field contract\ntransactions: " + params +
                             "\n\n=== RECORDS ===\n" + _txt +
                             "\n\n" + afmt}]
                elif arm in ("R_pairfee", "N_datesort"):
                    # ★수수료-짝 팔 (2026-08-25·t7348 074 t1 실물 뒤). N_wire 와 **한 가지만**
                    #   다르다: `=== RECORDS ===` 블록의 **순서**. 문면·값·형식 전부 동일하다.
                    #     R_pairfee   인출 바로 뒤에 그 날 수수료
                    #     N_datesort  같은 양의 재배열이되 짝짓지 않는다([[57]] 부정통제)
                    # ⚠2026-08-25 수리: 초판은 엔진 V2 갈래의 `sorted(keys)` 를 그대로 베껴
                    #   `NameError: keys` 로 죽었다(x535 1차). 이 프로브에서 필드 계약은
                    #   `params` 하나이고 다른 팔들이 쓰는 이름이 그것이다 — 그 이름을 쓴다.
                    _txt = regroup(text, "pair" if arm == "R_pairfee" else "datesort")
                    msgs = [{"role": "user", "content":
                             instr + "\n\n=== REFERENCE ===\n" +
                             "\n".join("%s: %s" % (k2, v2) for k2, v2 in ref.items()) +
                             "\n\n# Field contract\ntransactions: " + params +
                             "\n\n=== RECORDS ===\n" + _txt +
                             "\n\n" + afmt}]
                elif arm in ("P_pair", "P_noinv", "P_both", "N_len"):
                    # ★초과 행 팔 (2026-08-25). 관측(x525j 16창·결정론적):
                    #   chk_1 expect18/rows18 **초과 0** · chk_2·3·4 expect16/rows**17**.
                    #   초과의 정체는 계좌마다 다르고 셋 다 `_err` 접미 수수료에서 난다:
                    #     chk_2  `btxn_ar_lb_08f` 를 냈는데 원장엔 `btxn_ar_lb_08f_err` 뿐 = **id 날조**
                    #     chk_3  `btxn_ar_dg_02`(인출) + `btxn_ar_dg_02f_err`(같은 날) **둘 다** = 이중 덮음
                    #     chk_4  `btxn_ar_ev_03`     + `btxn_ar_ev_03f_err`             동일
                    #   선언은 규칙을 말한다 — *"ONE element per atm_withdrawal record"* ·
                    #   *"the id of the atm_fee line paired with this withdrawal; if this withdrawal
                    #   has NO fee line, the withdrawal's own id"*. 말하지 **않은** 것은 하나다:
                    #   *오류인 수수료 행도 여전히 그 인출의 짝인가*. `_err` 는 감사 대상이므로
                    #   모델이 *"정상 수수료가 아니니 이 인출엔 수수료가 없다"* 로 읽으면 인출 id 를
                    #   쓰고 `_err` 행도 따로 내서 **두 줄**이 된다 — 관측 그대로다.
                    #   ⚠팔은 **선언 텍스트에 한 칸 덧붙이기**뿐이다([[78]]). 엔진·형식·재료 불변.
                    #   ⚠기전이 둘이라 문장도 둘로 갈라 어느 쪽이 일하는지 본다(교락 금지).
                    _S_PAIR = ("A fee line that is itself the error you are auditing - wrong amount, "
                               "wrong network in its description, or one that should not have been "
                               "charged at all - is still the fee line paired with that withdrawal. "
                               "Such a withdrawal has a fee line, so use that fee line's "
                               "transaction_id for it, and it still gets exactly one element.")
                    _S_NOINV = ("Every transaction_id you write must be copied character for "
                                "character from a Record ID in the history above. An id you did "
                                "not read there does not exist, including one formed by shortening "
                                "or lengthening an id that is there.")
                    # 부정통제: 같은 길이·같은 자리·규칙 0([[57]]).
                    _S_LEN = ("This account's transaction history was retrieved from the bank's "
                              "records system and reflects the state of the account at the time of "
                              "retrieval. The history is provided for the audit described above and "
                              "covers the period the records system returned for this account id.")
                    _add = {"P_pair": _S_PAIR, "P_noinv": _S_NOINV,
                            "P_both": _S_PAIR + " " + _S_NOINV, "N_len": _S_LEN}[arm]
                    msgs = [
                        {"role": "user", "content":
                         instr + "\n\n=== REFERENCE ===\n" +
                         json.dumps(ref, ensure_ascii=False, indent=1) +
                         "\n\n" + afmt +
                         "\n\n# Field contract\ntransactions: " + params + " " + _add +
                         "\n\n=== RECORDS ===\n" + text}]
                elif arm == "G_norow":
                    # 예시 행 자체를 지운 판 — [[63]] 형태(더하기가 아니라 **제거**가 닫는가).
                    #   A2 `row_fields` 선언으로 예시를 대체한다(엔진 리터럴 0·선언에서 읽음).
                    _rf = ", ".join(iso.get("row_fields") or [])
                    _af2 = ('Reply with exactly one JSON object and nothing else: '
                            '{"transactions": [ ... one element per atm_withdrawal, '
                            'each with these fields: %s ... ]}' % _rf)
                    msgs = [{"role": "user", "content":
                             instr + "\n\n=== REFERENCE ===\n" +
                             json.dumps(ref, ensure_ascii=False, indent=1) +
                             "\n\n" + _af2 + "\n\n=== RECORDS ===\n" + text}]
                elif arm == "E_oneline":
                    # 계약 전문 대신 **A2 params 축자의 한 문장만** 붙인다 — 한 줄로 닫히나.
                    _one = ""
                    for _s in re.split(r"(?<=[.;])\s+", params):
                        if "EVERY atm_withdrawal" in _s:
                            _one = _s.strip()
                            break
                    msgs = [{"role": "user", "content":
                             instr + "\n\n=== REFERENCE ===\n" +
                             json.dumps(ref, ensure_ascii=False, indent=1) +
                             ("\n\n" + _one if _one else "") +
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
                # ★계기 수리 (2026-08-25·[[25]] 계기는 100% 정답 의무). 구판은
                #   `i2 in text` = **부분문자열** 대조였다. 그래서 원장에 없는 id 가
                #   '실재'로 통과했다 — 실물: chk_2 산출의 `btxn_ar_lb_08f` 는 레코드가
                #   아니고 원장엔 `btxn_ar_lb_08f_err` 만 있는데, 앞 문자열이 뒤 문자열의
                #   부분이라 `ids_real` 이 17/17 을 찍었다. 그 한 줄이 074 의 **초과 행**
                #   이었고, 계기가 그것을 *날조 아님* 으로 덮고 있었다.
                #   ⇒ 레코드 id **집합 소속**으로 바꾼다(닫힌 술어·판단 0).
                _ledger_ids = {i2 for i2, _t2, _d2 in records(text)}
                real = [i2 for i2 in ids if i2 and i2 in _ledger_ids]
                cov = coverage(ids, w, fees)
                _fp, _fn = fee_paired(ids, records(text))
                dup = sum(1 for r in (got or []) if isinstance(r, dict) and r.get("duplicate_of"))
                rows.append({"msg": idx, "acc": acc, "arm": arm, "k": k, "expect": exp,
                             "rows": (len(got) if got is not None else None),
                             "ids_real": len(real), "cover": cov, "withdrawals": len(w),
                             "fee_paired": _fp, "fee_total": _fn,
                             "dup_of": dup, "emitted": ids})
                print("      %-11s k=%d rows=%s/%d cover=%d/%d feepair=%d/%d ids_real=%d dup=%d"
                      % (arm, k, (len(got) if got is not None else "parse_fail"), exp,
                         cov, len(w), _fp, _fn, len(real), dup))

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
