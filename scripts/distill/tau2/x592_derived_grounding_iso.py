# -*- coding: utf-8 -*-
r"""x592 - `actual_apy` 접지 술어의 **거짓-음성**을 격리로 재고, 대안 술어를 같은 자리에서 잰다.

## 왜 (2026-08-29 · t7385~t7388 실측)

`TASK_094.md` §7-P1 은 **거짓-양성**(손님이 말한 5.0 이 통과)을 겨눴고 2026-08-26 에 배선됐다
(`ground.scalar_fields[actual_apy].corpus = ["ledger_tools"]`). 그런데 오늘 4 런에서 남은 것은
**반대편**이다 - `bank_t7388_hB2_20260829 task_094#s626729` 로그에서

    [T2_SG_GROUND] get_interest_correction: 3 ungrounded operand 드롭 ->
        actual_apy=5.1 (not found in the records - re-read the exact value); ...
    [T2_SCAFFOLD_GET] get_interest_correction -> None

이 **3회** 나온다. 그 `5.1` 은 gold 의 `actual_apy` 다. 원인은 한 선언 안의 자기모순이다:
A2 `params.actual_apy` 는 축자로 *"Derive it from the latest MONTHLY INTEREST CREDIT ...
monthly credit amount x 12 / principal x 100"* 이라며 **파생을 지시**하는데, 접지 술어는 그
파생값이 원장에 **문자로 실재**할 것을 요구한다 ⇒ **구조상 옳은 파생일수록 반드시 드롭된다**.

## 무엇을 재나 (모델 0 · 무료 · 결정론)

호출 시점의 원장을 정본 `t2_forensic.evidence_ctx_at` 으로 재구성하고, **정본 엔진 술어**
(`t2_scaffold_get._val_grounded`)를 그대로 호출해 두 판정을 나란히 낸다.

    P_now   값이 도구출력에 문자로 실재하나            <- 오늘 배선
    P_inv   그 값이 함의하는 **원장 리터럴**이 실재하나 <- 대안
            implied = principal x actual_apy x 0.01 / 12   (= 월 이자 크레딧)

`P_inv` 는 A2 공식의 역이고 엔진에 새 산수를 넣지 않는다(곱셈+상수뿐 - `t2_compute` 기존 op).
**gold 는 술어에 들어가지 않는다**([[23]]) - gold 값은 표의 마지막 칸에 *진단용*으로만 찍는다.

## 판정 규칙

  · `P_inv` 가 옳은 값을 살리고 **틀린 값을 안 살리면** 배선 자격([[76]]).
  · 한 건이라도 틀린 값을 살리면 그 값을 적고 배선하지 않는다(거짓-양성 반경).
  · `P_inv` 가 옳은 값도 못 살리면 **재료가 대화에 없다는 뜻**이고, 그때 필요한 것은 술어가
    아니라 `requires_reads`(§7-P2)다 - 그 경우도 표가 그대로 답한다([[78]] 격리 실패 = 재료 결손).
"""
import json
import sys

sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F
import t2_scaffold_get as S

TOOL = "get_interest_correction"
FIELD = "actual_apy"


def _f(x):
    try:
        return float(str(x).replace("$", "").replace(",", "").strip())
    except Exception:
        return None


def corpus_at(sim, i, drop_self=False):
    """그 호출 시점의 `ledger_tools` 코퍼스 = 도구 출력 전용(우리 경고 문면 제거).

    `drop_self=True` 면 **이 도구 자신의 이전 출력**을 뺀다. 우리 `return_template` 이
    `applied(actual) APY={actual_apy}%` 로 모델이 넘긴 값을 축자로 되울리므로, 그 출력이
    다음 호출의 코퍼스에 들어가면 **우리가 모델의 추측을 증명해 주는** 자기-접지가 된다
    ([[25]] · `_corpus_texts` 주석이 이미 자인한 축: *"도구가 한 번 뱉은 값은 그 다음
    호출부터 무조건 '실재'가 된다"*).
    """
    ev = F.evidence_ctx_at(sim, i)
    outs = dict(ev.get("__tool_outputs") or {})
    if drop_self:
        outs.pop(TOOL, None)
    return [S._strip_own_feedback(t) for t in outs.values()]


def grounders(sim, i, val):
    """그 값을 접지시킨 **도구 이름 목록** (어느 출력이 근거였나)."""
    ev = F.evidence_ctx_at(sim, i)
    out = []
    for nm, txt in (ev.get("__tool_outputs") or {}).items():
        if S._val_grounded(val, [S._strip_own_feedback(txt)], "number"):
            out.append(nm)
    return out


def implied(apy, principal):
    """A2 공식의 역: 이 APY 가 참이라면 원장에 있어야 할 **월 이자 크레딧**.
    A2 `op` 가 쓰는 상수(0.01 · 1/12)를 그대로 쓴다 - 새 산수 0."""
    a, p = _f(apy), _f(principal)
    if a is None or p is None:
        return None
    return round(p * a * 0.01 * (1.0 / 12.0), 2)


def calls_of(sim):
    """`get_interest_correction` 호출과 그 인덱스."""
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if not isinstance(m, dict):
            continue
        for tc in (m.get("tool_calls") or []):
            if F.nameof(tc) == TOOL:
                out.append((i, F.argsof(tc)))
    return out


def main(argv=None):
    tags = (argv or sys.argv[1:]) or ["bank_t7387_hB1_20260829", "bank_t7388_hB2_20260829"]
    rows = []
    for tag in tags:
        try:
            sims = F.sims(tag)
        except Exception as ex:
            print("(못 읽음) %s : %r" % (tag, ex))
            continue
        for s in sims:
            st = F.simtag(s)
            if "task_094" not in st:
                continue
            gold_apy = None
            for e in (F.mutation_diff(s, F.mutating_tools(), tag=tag) or {}).get("missing") or []:
                a = (e.get("args") or {})
                if "actual_apy" in a:
                    gold_apy = a.get("actual_apy")
            for i, args in calls_of(s):
                cx = corpus_at(s, i)
                cx_ex = corpus_at(s, i, drop_self=True)
                apy, prin = args.get(FIELD), args.get("principal")
                imp = implied(apy, prin)
                rows.append({
                    "tag": tag, "sim": st, "msg": i,
                    "actual_apy": apy, "principal": prin,
                    "P_now": bool(S._val_grounded(apy, cx, "number")),
                    "P_noself": bool(S._val_grounded(apy, cx_ex, "number")),
                    "grounders": grounders(s, i, apy),
                    "implied": imp,
                    "P_inv": bool(imp is not None and S._val_grounded(imp, cx, "number")),
                    "gold_apy": gold_apy,
                    "gold_P_now": bool(S._val_grounded(gold_apy, cx, "number")) if gold_apy else None,
                    "gold_implied": implied(gold_apy, prin),
                    "gold_P_inv": bool(implied(gold_apy, prin) is not None
                                       and S._val_grounded(implied(gold_apy, prin), cx, "number"))
                    if gold_apy else None,
                    "corpus_chars": sum(len(t) for t in cx),
                })
    print("=" * 124)
    print("%-20s %-20s %4s %8s %7s %8s %9s %6s | %6s %6s %s"
          % ("tag", "sim", "msg", "actual", "P_now", "P_noself", "implied", "P_inv",
             "gold", "g_inv", "접지원"))
    print("=" * 124)
    for r in rows:
        print("%-20s %-20s %4s %8s %7s %8s %9s %6s | %6s %6s %s"
              % (r["tag"][5:20], r["sim"], r["msg"], r["actual_apy"],
                 r["P_now"], r["P_noself"], r["implied"], r["P_inv"],
                 r["gold_apy"], r["gold_P_inv"], ",".join(r["grounders"])[:34]))
    ok_now = sum(1 for r in rows if r["gold_P_now"])
    ok_inv = sum(1 for r in rows if r["gold_P_inv"])
    wrong = [r for r in rows if r["gold_apy"] and _f(r["actual_apy"]) != _f(r["gold_apy"])]
    print()
    print("옳은 값이 살아남는가 :  P_now %d/%d   ·   P_inv %d/%d" % (ok_now, len(rows), ok_inv, len(rows)))
    print("틀린 값이 통과하는가 :  P_now %d/%d · P_noself %d/%d · P_inv %d/%d (분모=gold 아는 행)"
          % (sum(1 for r in wrong if r["P_now"]), len(wrong),
             sum(1 for r in wrong if r["P_noself"]), len(wrong),
             sum(1 for r in wrong if r["P_inv"]), len(wrong)))
    out = "/home/woori/scratch/x592_derived_grounding.json"
    try:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print("[saved]", out)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
