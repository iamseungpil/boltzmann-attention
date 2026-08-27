# -*- coding: utf-8 -*-
"""일회성 — `T2_SG_ROW_COUNT` 선언 삽입 (2026-08-28 · gate/specific **두 층 동시**·[[24]]).

## 무엇을 넣나

`get_atm_fee_discrepancies` 에 세 칸:

    isolate.row_kind          "atm_withdrawal"   ← 분모의 종류. 출처 = env 도구 출력 스키마
    return_template_short     행이 모자랄 때 나가는 반환문 (총액 문장 없음 + 재공급 지시)
    _note_row_count           근거

## 왜 (실측 · t7368 `task_072#s626729`)

    msg[25] Bluest      레코드 32 · `type: atm_withdrawal`  9 → 서브  9   delta_total 14.0 = gold
    msg[35] Light Green 레코드 26 · `type: atm_withdrawal` 10 → 서브 **9** delta_total  5.0 ≠ 3.5

빠진 행 = `btxn_8c58b19a3628 (charged $0.00, documented fee $1.50, difference $-1.50)`
= **수수료 줄이 없는 인출**. 그런데 반환문은 `[coverage] 9 of 9 rows were checked (0 could not
be verified)` 였다 — 분모가 *넘어온 행 수*라 자기 자신을 잰다([[25]] 위반: 우리가 틀린 총액을
*"use it as the credit amount"* 라는 권위 문면으로 건넸다).

## 왜 이 모양인가

`return_template_short` 는 `return_template` 을 **총액 문장 앞에서 자른 것 + 결손 문장**이다.
새로 저작한 도메인 문장이 0 이고, 그 동일성을 `test_sg_row_count.py` 가 바이트로 고정한다.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TOOL = "get_atm_fee_discrepancies"
KIND = "atm_withdrawal"
CUT = "The signed total of the differences listed above"

# ★[[64]]: 무엇이 틀렸나 + 무엇을 하면 풀리나 둘 다 담는다. 도메인 낱말은 `{kind}` 뿐이고
#   그것은 런타임에 선언(`isolate.row_kind`)에서 주입된다.
SHORT_TAIL = (
    "This audit is INCOMPLETE and states no net correction: {missing} of the {read} {kind} "
    "record(s) that were read for this account did not reach the check, so any total computed "
    "from the lines above would be short. Read the account's records again and call again with "
    "EVERY {kind} of this account passed in - including the ones with no fee line, which are "
    "exactly the ones that go missing.")

NOTE = (
    "2026-08-28 · `T2_SG_ROW_COUNT`. 실측(t7368 task_072#s626729): Bluest 레코드 32 중 "
    "`type: atm_withdrawal` 9 → 서브 9 → delta_total 14.0 = gold · Light Green 26 중 10 → "
    "서브 **9** → delta_total 5.0 ≠ gold 3.5. 빠진 한 행이 "
    "`btxn_8c58b19a3628 (charged $0.00, documented fee $1.50, difference $-1.50)` = **수수료 "
    "줄이 없는 인출**이고, 그런데도 반환문은 `[coverage] 9 of 9 rows were checked (0 could not "
    "be verified)` 였다 — 분모가 *넘어온 행 수*라 자기 자신을 잰다(`t2_scaffold_get._short_rows` "
    "독스트링). 그 결과 우리 층이 **틀린 총액을 권위 문면으로** 건넸다([[25]] 위반). || "
    "출처([[23]]): `row_kind` 는 env 도구 출력의 `type:` 값이라 **기계도출**이고 gold 무참조 — "
    "`ledger_metrics.row_keys`/`group_field` 와 같은 계보. 문장은 `return_template` 을 총액 "
    "문장 앞에서 자른 것 + 재공급 지시뿐이라 새 도메인 저작 0. || 왜 프롬프트 수리가 아니라 "
    "검산인가: `T2_SG_PROMPT_V2` 는 같은 결손(수수료 줄 없는 인출 누락)을 **프롬프트 모양**으로 "
    "고쳤는데, 074 chk_2 에서 13~15/16 → 16/16 을 사고 072 Light Green 에서 10/10 → 9/10 을 "
    "팔았다(t7348 ↔ t7363·t7368). 프롬프트 섭동은 태스크마다 부호가 갈리고([[07]]) 그 손실은 "
    "위 coverage 문면 때문에 **조용하다**. 닫힌 검산은 못 본 태스크에서도 참이다([[22]]). || "
    "⚠[[70]] 파는 것: 행이 실제로 모자란 호출에서 총액 한 줄을 잃는다 — 지금 그 자리는 **틀린 "
    "총액**이 나가던 자리다. ⚠적게 넘긴 것만 본다(초과·중복은 다른 술어 몫). ⚠종류가 원천에 "
    "0건이면 판정하지 않는다. ⚠라이브 효과 미측정 — A/B 가 잰다.")


def find_tools(o):
    if isinstance(o, dict):
        if "scaffold_get_tools" in o:
            return o["scaffold_get_tools"]
        for v in o.values():
            r = find_tools(v)
            if r:
                return r
    if isinstance(o, list):
        for v in o:
            r = find_tools(v)
            if r:
                return r
    return None


def main():
    changed = []
    for layer in ("gate", "specific"):
        p = os.path.join(HERE, "a2", "banking_knowledge.%s.json" % layer)
        d = json.load(io.open(p, encoding="utf-8"))
        tools = find_tools(d) or []
        t = next((x for x in tools if x.get("name") == TOOL), None)
        if t is None:
            print("%-9s %s 없음 — 건너뜀" % (layer, TOOL))
            continue
        rt = str(t.get("return_template") or "")
        if CUT not in rt:
            print("%-9s ⛔`%s` 를 못 찾았다 — 템플릿이 바뀌었다. 중단." % (layer, CUT))
            return 2
        head = rt.split(CUT)[0].rstrip()
        t.setdefault("isolate", {})["row_kind"] = KIND
        t["return_template_short"] = head + " " + SHORT_TAIL
        t["_note_row_count"] = NOTE
        with io.open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
            f.write("\n")
        changed.append(layer)
        print("%-9s ok · row_kind=%s · short 템플릿 %d자 (원본 %d자)"
              % (layer, KIND, len(t["return_template_short"]), len(rt)))
    print("바뀐 층: %s" % (changed or "없음"))
    return 0 if len(changed) == 2 else 1


if __name__ == "__main__":
    sys.exit(main())
