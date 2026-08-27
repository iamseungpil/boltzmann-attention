# -*- coding: utf-8 -*-
r"""x577 — 부호합의 실패가 **전사**인가 **덧셈**인가 (무료 · 2026-08-28 · 사용자 지시 *"072 pass"*).

## 왜 이 프로브인가

`x542` 가 이 자리를 다섯 팔 × 24셀로 재서 **전부 0** 을 냈다(A_asis 축자 · B_fmt 렌더링 ·
C_sign 부호 명시 한 줄 · D_both · N_len 부정통제 동률). `A_asis` 가 라이브 오답(절댓값 합)을
재현했으므로 격리는 공정하다([[62]] 2b). 072 도 그 24셀 안에 있다 —
`task_072#s626729 msg48` · `#s361454 msg56·61` 세 자리 전부 **부호합 3.50 ↔ 산출 6.50**.

⇒ 배달로 살 수 있는 것은 끝났다. 남은 물음은 **한 칸 더 쪼개는 것**이다:

    ⓐ 모델이 부호를 **읽지** 못하나 (전사 결손)        → 엔진이 더하면 **엔진이 고르는 것**이 된다 ⇒ 중단
    ⓑ 부호는 읽는데 **더하지** 못하나 (산수 결손)       → 그 단계에만 결정론([[62]] ③·등대 §1.2 F2b)

x542 는 이것을 안 갈랐다 — 다섯 팔 전부 **합을 물었다**. 그래서 ⓐ와 ⓑ가 한 칸에 접혀 있다.

## 팔

    A_sum        x542 의 물음 그대로(합을 묻는다)                 ← 라이브 오답 재현 게이트
    B_transcribe **합을 묻지 않는다** — 행마다 `{id, difference}` 로 옮겨 적게만 한다.
                 엔진은 그 값들을 `t2_compute` 의 `sum` 으로 더한다(op 정본·사본 0).
    N_len        B 와 같은 물음 · 블록에 같은 길이 무관 문장 부착([[57]])

## 채점 — 닫힌 술어 · gold 미접촉

정답 = 블록의 `difference $X` 를 **부호 그대로** 더한 값(`x542.signed_sum` 정본). 규칙 출처는
선언 축자(`net correction` · `it shows as a negative difference`)이고 gold 는 안 본다([[23]]).

B 는 **두 가지를 따로** 센다:
    `전사일치` 모델이 옮겨 적은 값들의 다중집합 == 블록이 담은 값들의 다중집합 (부호 포함)
    `합일치`   엔진이 그 전사를 더한 값 == 부호합

## ⛔[[62]] 4문

  ①재봤나 — x542 5팔×24셀 전부 0(공정 격리). 이 프로브는 그 실패를 **두 단계로 가른다**.
  ②격리에서 성공하나 — 합은 **아니다**(x542). 전사는 **미측정**이고 그것이 이 물음이다.
  ③사라지는 모델 판단 — B 에서 사라지는 것은 **덧셈뿐**이다. 어느 행인지·부호가 무엇인지는
    전부 모델이 옮겨 적는다. 전사가 틀리면 엔진 합도 틀리므로 **측정 대상이 안 사라진다**.
  ④엔진이 argmax·"정답은 X" 를 내나 — 아니오. 행 선택 0 · 값 생성 0 · `sum` 하나뿐이다.

## ⚠[[70]] 이 축이 파는 것 (미리 적는다)

엔진이 순액을 내면 **순액을 스스로 짓는 연습이 사라진다**. 순액이 부호합이 아닌 자리
(예: 정책이 음수 행을 제외하라고 말하는 태스크)에서는 우리가 틀린다. ⇒ 라이브 배선 전
**태스크별 부호표 필수**이고, 이 프로브는 배선이 아니라 **측정**이다.

사용: (리모트·cwd=scripts/distill/tau2) py -3 x577_signed_split_iso.py --port 8140 --n 4
"""
import argparse
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_compute as CP                                             # noqa: E402  (sum op 정본)
import x542_signed_sum_iso as X542                                  # noqa: E402  (창·채점 정본)

NL = chr(10)
REP = X542.REP

# 합을 **묻지 않는다** — 옮겨 적게만 한다. 도메인 어휘 0·규칙 0([[66]]).
ASK_TRANSCRIBE = (
    NL + NL + "Transcribe the discrepancy lines above for account {acct} ONLY - not for any "
    "other account in this conversation." + NL +
    "Reply with a JSON array only, one object per line, each with exactly two keys: "
    '"id" (the transaction id, copied verbatim) and "difference" (the number shown for that '
    "line, copied verbatim INCLUDING its sign, as a JSON number). Do not add lines, do not "
    "leave any out, do not total them, do not round, and do not change any sign." + NL +
    'Example of the shape only: [{"id": "btxn_0000", "difference": 1.25}]')

RE_NUM = re.compile(r'"difference"\s*:\s*(-?[\d.]+)')
RE_ID = re.compile(r'"id"\s*:\s*"([^"]+)"')


def engine_sum(vals):
    """엔진 몫 = **덧셈 하나**. `t2_compute` 의 `sum` op 를 그대로 부른다(사본 0·[[67]])."""
    ctx = {"v": [{"x": v} for v in vals]}
    spec = {"op": "sum", "of": ["v.%d.x" % i for i in range(len(vals))]}
    out = CP.apply_op(spec, ctx)
    return None if out is None else round(float(out), 2)


def parsed(reply):
    """모델 답에서 전사값을 꺼낸다 — 계약 회수일 뿐 도메인 스캔이 아니다([[59]] 허용역)."""
    try:
        arr = json.loads(re.search(r"\[.*\]", reply, re.S).group(0))
        vals = [float(o["difference"]) for o in arr if isinstance(o, dict) and "difference" in o]
        ids = [str(o.get("id") or "") for o in arr if isinstance(o, dict)]
        if vals:
            return vals, ids
    except Exception:
        pass
    vals = [float(x) for x in RE_NUM.findall(reply or "")]
    return vals, RE_ID.findall(reply or "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--task", default="")
    ap.add_argument("--wiring-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(REP, "x577_signed_split_2026_08_28.json"))
    a = ap.parse_args(argv)

    # 자체 검산: 엔진 합이 정말 덧셈인가 (모델 0)
    assert engine_sum([1.5, -1.5, 3.0, 0.5]) == 3.5, "sum op 가 부호를 안 지킨다"

    cases, skipped = X542.windows()
    if a.task:
        cases = [c for c in cases if c["sim"].startswith(a.task)]
    if not cases:
        print("창 0 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 1

    print("# x577 — 부호합을 **전사 ↔ 덧셈**으로 가른다 · 창 %d개" % len(cases))
    for c in cases:
        print("   %-22s msg%-4s acct=%-16s 라이브 %.2f · 절댓값합 %.2f · **부호합 %.2f** · 행 %d"
              % (c["sim"], c["msg"], c["acct"][:16], c["live"], c["absum"], c["want"],
                 len(c["vals"])))

    if a.wiring_only:
        c = cases[0]
        print(NL + "--- A_sum 꼬리 ---" + NL + " ".join(
            (c["win"] + NL + NL + c["block"] + X542.ASK_T.format(acct=c["acct"])).split())[-320:])
        print(NL + "--- B_transcribe 꼬리 ---" + NL + " ".join(
            (c["win"] + NL + NL + c["block"] + ASK_TRANSCRIBE.format(acct=c["acct"])).split())[-420:])
        print(NL + "엔진 합 자체검산: sum([1.5,-1.5,3.0,0.5]) = %s" % engine_sum([1.5, -1.5, 3.0, 0.5]))
        return 0

    rows = []
    agg = collections.defaultdict(lambda: collections.Counter())
    for c in cases:
        base = c["win"] + NL + NL + c["block"]
        fill = c.get("filler")
        arms = {
            "A_sum": (base + X542.ASK_T.format(acct=c["acct"]), 16),
            "B_transcribe": (base + ASK_TRANSCRIBE.format(acct=c["acct"]), 420),
        }
        if fill:
            arms["N_len"] = (c["win"] + NL + NL + c["block"] + NL + fill
                             + ASK_TRANSCRIBE.format(acct=c["acct"]), 420)
        want_ms = sorted(round(v, 2) for v in c["vals"])
        for arm, (body, mx) in arms.items():
            for k in range(a.n):
                try:
                    rep = " ".join(str(X542.gen(a.port, body, mx)).split())
                except Exception as e:
                    print("  %-14s 호출 실패: %r" % (arm, e))
                    continue
                r = {"sim": c["sim"], "msg": c["msg"], "acct": c["acct"], "arm": arm, "k": k,
                     "want": c["want"], "absum": c["absum"], "live": c["live"], "raw": rep[:300]}
                if arm == "A_sum":
                    try:
                        got = round(float(re.search(r"-?[\d.]+", rep).group(0)), 2)
                    except Exception:
                        got = None
                    r["got"] = got
                    r["ok"] = (got is not None and abs(got - c["want"]) < 0.01)
                else:
                    vals, ids = parsed(rep)
                    r["n_rows"] = len(vals)
                    r["transcribed"] = vals
                    r["전사일치"] = (sorted(round(v, 2) for v in vals) == want_ms)
                    got = engine_sum(vals) if vals else None
                    r["got"] = got
                    r["ok"] = (got is not None and abs(got - c["want"]) < 0.01)
                    agg[arm]["전사일치"] += 1 if r["전사일치"] else 0
                agg[arm]["n"] += 1
                agg[arm]["ok"] += 1 if r["ok"] else 0
                agg[arm]["절댓값합"] += 1 if (r["got"] is not None
                                              and abs(r["got"] - c["absum"]) < 0.01) else 0
                rows.append(r)
                print("  %-22s msg%-4s %-13s k%d  got=%-8s ok=%-5s %s"
                      % (c["sim"], c["msg"], arm, k, r["got"], r["ok"],
                         ("전사일치" if r.get("전사일치") else
                          ("전사 %d행" % r["n_rows"] if arm != "A_sum" else ""))))

    print("")
    print("=" * 96)
    print("결과")
    print("=" * 96)
    for arm in ("A_sum", "B_transcribe", "N_len"):
        if not agg[arm]["n"]:
            continue
        c = agg[arm]
        print("  %-14s ok %2d/%-3d  절댓값합 %2d  %s"
              % (arm, c["ok"], c["n"], c["절댓값합"],
                 ("전사일치 %d/%d" % (c["전사일치"], c["n"]) if arm != "A_sum" else "")))
    print("")
    print("판독:")
    print("  A_sum 이 절댓값합을 재현해야 격리가 공정하다([[62]] 2b · x542 재확인).")
    print("  B 의 **전사일치가 높고 ok 도 높으면** 결손은 덧셈 하나다 ⇒ 그 단계에만 결정론이")
    print("  허용된다([[62]] ③). 엔진이 더하는 것 말고 하는 일이 없음은 위 자체검산이 보인다.")
    print("  B 의 **전사일치가 낮으면** 모델이 부호를 못 읽는 것이고, 그때 엔진이 더하면")
    print("  **엔진이 고르는 것**이 된다 ⇒ 배선하지 않는다.")
    print("  N_len 이 B 만큼이면 산 것은 물음이 아니라 길이다([[57]]).")

    out = {"probe": "x577_signed_split", "date": "2026-08-28",
           "scoring": "정답 = 블록의 `difference $X` 부호 합(`x542.signed_sum` 정본). gold 미접촉.",
           "engine_step": "덧셈 하나 — `t2_compute.apply_op({'op':'sum'})`. 행 선택 0·값 생성 0.",
           "arms": {"A_sum": "x542 의 합 물음 그대로",
                    "B_transcribe": "행별 전사만 묻고 엔진이 합산",
                    "N_len": "B + 같은 길이 무관 문장([[57]])"},
           "agg": {k: dict(v) for k, v in agg.items()},
           "skipped": skipped, "rows": rows,
           "limits": ["창·짝짓기·채점은 `x542_signed_sum_iso` 정본 그대로다(사본 0).",
                      "이것은 **측정**이고 배선이 아니다. 라이브 이식 전 [[70]] 태스크별 부호표 필수.",
                      "전사일치는 다중집합 비교라 행 **순서**는 안 본다."]}
    with io.open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
