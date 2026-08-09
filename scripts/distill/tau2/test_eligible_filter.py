# -*- coding: utf-8 -*-
"""회귀 검정 (C337): **통과 집합**을 엔진이 걸러 준다 — 그리고 못 거를 때는 침묵한다.

무엇을 막는 검정인가 —
 ⒜ **문턱을 못 걸고 만든 '통과 집합'**. 경과일이 없으면 이 문장은 전체 표에 통과 도장을 찍는
    것과 같고, 실측상 그 상태의 최고액이 정확히 오답이었다(65일 손님에게 문턱 90짜리 상품).
    ⇒ `days=None` 이면 **침묵**해야 한다.
 ⒝ **자격 미달 상품이 목록에 남는 것**. 문턱을 넘지 못한 주어·연간 상한을 다 쓴 주어는
    나오면 안 된다.
 ⒞ **판단 재료를 빼앗는 것**. 통과 집합을 이름+보너스로 줄이면 099 계열이 안 닫힌다 —
    문턱이 아무도 못 거르는 손님에게는 남은 축(예치 하한 등)이 유일한 판별자다.
    ⇒ A2가 선언한 축이 **전부** 실려야 한다.
 ⒟ **우리가 argmax 까지 하는 것**. 정렬은 이름순이어야 한다([[05]] Q2: 고르는 것은 모델).

오프라인 전용: tau2·서버·LLM 불요. A2/A3는 repo의 정본 파일을 그대로 읽는다.
실행: py -3 test_eligible_filter.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as G                                     # noqa: E402
import t2_factdag as FD                                       # noqa: E402
import t2_ledger as LG                                        # noqa: E402
from t2_ledger import _num as _n                               # noqa: E402
from gate_interpreter import load_domain_a2                    # noqa: E402

DOMAIN = "banking_knowledge"
FAILED = []


def chk(cond, label):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        FAILED.append(label)


class _Agent(object):
    def __init__(self, ops):
        self._t2_ledger_ops = ops


def main():
    a2 = load_domain_a2(DOMAIN)
    if not a2:
        print("A2 없음 — skip")
        return 0
    specs = a2.get("ledger_metrics") or []
    spec = next((s for s in specs if s.get("eligible_text")), None)
    chk(spec is not None, "A2에 통과-집합 선언(eligible_text)이 있다")
    if spec is None:
        return 1
    cfg = spec.get("eligible") or {}
    rows = (a2.get("policy_ontology") or {}).get("rows") or ()
    show = list(cfg.get("show_axes") or ())
    crit = list(cfg.get("criteria") or ())
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in show}
    by_src = {c.get("operand"): c for c in crit}
    min_ax = (by_src.get("days") or {}).get("axis")
    lim_ax = (by_src.get("tally") or {}).get("axis")
    dep_ax = next((c["axis"] for c in crit if c.get("operand") == "stated"
                   and c.get("compare") == "ge"), None)
    mins = maps.get(min_ax) or {}
    lims = maps.get(lim_ax) or {}
    chk(bool(mins) and bool(lims), "선언한 필터 축이 A3에서 조회된다 (문턱 %d·상한 %d)"
        % (len(mins), len(lims)))
    chk(all(c.get("axis") in maps for c in crit),
        "모든 기준 축이 표시 축에 포함돼 있다(거르는 근거를 모델도 본다)")
    chk(any(c.get("operand") == "stated" for c in crit),
        "대화에서 오는 피연산자 기준이 선언돼 있다  ← 예치 하한을 표시만 하던 결손")

    # 문턱이 갈리는 자리를 **A3에서** 고른다(테스트가 도메인 어휘를 짓지 않는다).
    need = sorted({int(v[0]) for v in mins.values()})
    lo, hi = need[0], need[-1]
    chk(lo < hi, "문턱 값이 상품마다 다르다 (경계를 만들 수 있다: %d ~ %d)" % (lo, hi))
    days = (lo + hi) // 2
    low = sorted(s for s, v in mins.items() if int(v[0]) <= days)
    high = sorted(s for s, v in mins.items() if int(v[0]) > days)
    chk(bool(low) and bool(high), "그 경과일에서 통과/미달이 둘 다 존재한다 (%d/%d)"
        % (len(low), len(high)))

    # ── ⒝ 미달은 빠지고 통과는 남는다 ────────────────────────────────────────
    txt = LG.eligible_text(days, {}, maps, spec)
    chk(bool(txt), "통과 집합 문장이 생성된다")
    lines = [l.strip() for l in txt.splitlines() if l.startswith("  ")]
    named = {l.split(":")[0].strip() for l in lines}
    chk(all(s in named for s in low), "문턱을 넘은 주어가 전부 실린다")
    chk(not any(s in named for s in high), "문턱 미달 주어는 하나도 안 실린다  ← 이 레버의 전부")

    # 연간 상한을 다 쓴 주어도 빠진다
    victim = low[0]
    cap = int(lims[victim][0]) if victim in lims else None
    chk(cap is not None, "그 주어에 연간 상한이 있다(상한 검정을 만들 수 있다)")
    if cap is not None:
        txt2 = LG.eligible_text(days, {victim: cap}, maps, spec)
        named2 = {l.strip().split(":")[0].strip() for l in txt2.splitlines() if l.startswith("  ")}
        chk(victim not in named2, "연간 상한을 다 쓴 주어는 빠진다")
        chk(len(named2) == len(named) - 1, "그 하나만 빠진다(다른 주어는 그대로)")

    # ── ⒞ 선언한 축이 전부 실린다 ───────────────────────────────────────────
    rich = max(low, key=lambda s: sum(1 for ax in show if s in maps[ax]))
    line = next(l for l in lines if l.startswith(rich + ":"))
    missing = [ax for ax in show if rich in maps[ax] and ax not in line]
    chk(not missing, "그 상품에 기록된 축이 전부 실린다 (빠진 축: %s)" % (missing or "없음"))

    # ── ⒟ 정렬은 이름순 ─────────────────────────────────────────────────────
    order = [l.split(":")[0].strip() for l in lines]
    # ⚠**중립성 주장 철회(2026-08-09·C355)**: 이 줄은 원래 *"우리가 argmax 하지 않는다"* 의
    #   근거로 붙어 있었다. 그런데 내용·이름·개수·후보목록을 전부 고정하고 **행 순서만** 바꾸면
    #   답이 뒤집히고, 시험한 9배열 중 **이름순이 유일하게 오답**을 냈다(x172·32B·n=10).
    #   ⇒ 이름순은 중립이 아니라 **우리가 당기는 레버**다. 검정은 *현재 거동의 고정*으로만 남기고,
    #     중립성 주장은 여기서 하지 않는다. 정렬 변경은 복제(다른 태스크·정박·모델) 후에 판단한다.
    chk(order == sorted(order), "정렬은 이름순이다 (거동 고정·**중립성 주장 아님**·C355)")

    # ── ⒜ 못 거르면 침묵한다 (음성 통제·[[57]]) ─────────────────────────────
    chk(LG.eligible_text(None, None, maps, spec) == "",
        "걸 수 있는 기준이 하나도 없으면 침묵한다  ← 그런 '통과 집합'은 통과 도장이다")
    chk(LG.eligible_text(None, {}, maps, spec) != "",
        "누계가 **있으면**(빈 원장이라도) 상한 기준 하나로 발화한다")
    chk(LG.eligible_text(None, None, maps, spec, {dep_ax: 1}) != "" if dep_ax else True,
        "대화-피연산자 하나만 있어도 발화한다(기준 출처는 셋 다 대등하다)")
    chk(LG.eligible_text(days, {}, {}, spec) == "",
        "A3 조회가 비면 침묵한다")
    chk(LG.eligible_text(days, {}, maps, {}) == "",
        "A2가 문구를 선언하지 않으면 침묵한다(선언 없는 도메인은 거동 0)")

    # ── 대화-피연산자: 있으면 거르고, 없으면 안 거른다 ──────────────────────
    chk(dep_ax is not None, "≥ 방향의 대화-피연산자 기준이 있다 (%s)" % dep_ax)
    if dep_ax:
        dep = maps.get(dep_ax) or {}
        vals = sorted({_n(v) for v in dep.values()})
        cut = vals[len(vals) // 2]
        base = {l.strip().split(":")[0].strip()
                for l in LG.eligible_text(days, {}, maps, spec).splitlines()
                if l.startswith("  ")}
        got = {l.strip().split(":")[0].strip()
               for l in LG.eligible_text(days, {}, maps, spec, {dep_ax: cut}).splitlines()
               if l.startswith("  ")}
        too_big = {s for s, v in dep.items() if _n(v) > cut and s in base}
        chk(bool(too_big), "그 값으로 떨어져야 할 주어가 존재한다 (%d개)" % len(too_big))
        chk(not (got & too_big), "대화-피연산자로 실제로 걸러진다  ← 099를 가르는 축")
        chk(got == base - too_big, "그 축이 없는 주어는 그대로 남는다(모름 ≠ 탈락)")

    # ── 라이브 경로: `_limit_reduce_text` 를 통해 실제로 나가는가 ────────────
    tally_spec = next((s for s in specs if s.get("exhausted_text")), None)
    ops = {spec.get("trigger_tool"): {"spec": spec, "tally": {}, "days": days}}
    if tally_spec is not None:
        ops[tally_spec.get("trigger_tool")] = {"spec": tally_spec, "tally": {}, "days": None}
    live = G._limit_reduce_text(_Agent(ops), a2, [])
    live_rows = {l.strip().split(":")[0].strip()
                 for l in live.splitlines() if l.startswith("  ")}
    chk(bool(live_rows & set(low)),
        "라이브 호출부(`_limit_reduce_text`)에서 실제로 발화한다")
    chk(not (live_rows & set(high)), "라이브 문장에도 미달 주어가 없다")

    # ★문구 형태를 못박는다 (x151 3라운드 실측·유료 0). 행이 같아도 **꼬리말**이 붙으면
    #   레버가 뒤집힌다 — 100 은 5/5 → 2/5, 099 는 raw 4~5/5 를 0~1/5 로 떨어뜨렸다.
    #   모델이 *"자격은 이미 처리됐다"* 로 읽고 자기 검사를 끄기 때문이다. 여기서는
    #   **행 뒤에 산문이 붙지 않는 것**만 못박는다(문구 내용은 A2 몫).
    body = spec["eligible_text"]
    chk(body.rstrip().endswith("{eligible}"),
        "통과 집합 뒤에 산문이 붙지 않는다  ← 붙이면 레버가 뒤집힌다(x151 A5)")
    # 열 순서도 실측이다 — 판별 축이 뒤로 밀리면 같은 글자 수에도 099 가 4/5 → 0/5.
    chk(dep_ax and lim_ax and show.index(dep_ax) < show.index(lim_ax),
        "판별 축(대화-피연산자)이 상한 축보다 **앞**에 온다  ← C340: 뒤로 밀면 4/5→0/5")

    # 누계는 **A2가 지목한 선언**에서 온다 — 계좌 쪽 tally 를 섞으면 조용히 틀린다
    tf = cfg.get("tally_from")
    chk(tf and tf != spec.get("trigger_tool"),
        "누계 출처가 다른 선언으로 지목돼 있다 (%s)" % tf)
    if tf and cap is not None:
        ops[tf] = {"spec": ops.get(tf, {}).get("spec", tally_spec), "tally": {victim: cap},
                   "days": None}
        live2 = G._limit_reduce_text(_Agent(ops), a2, [])
        # ⚠판정 대상은 **통과-집합 행**뿐이다. 같은 반환에 `exhausted_text` 도 실리는데 그쪽은
        #   소진된 그룹의 이름을 (일부러) 말하므로, 문자열 전체로 보면 늘 걸린다.
        rows2 = {l.strip().split(":")[0].strip()
                 for l in live2.splitlines() if l.startswith("  ")}
        chk(victim not in rows2, "지목된 선언의 누계가 라이브 필터에 반영된다")
        chk(victim in live2, "그래도 소진 사실 자체는 다른 문장이 말한다(침묵 아님)")

    print("\n%s  (%d 실패)" % ("PASS" if not FAILED else "FAIL", len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
