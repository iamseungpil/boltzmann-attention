# -*- coding: utf-8 -*-
"""회귀 — 격리 서브의 REFERENCE 는 **에이전트의 전사**가 아니라 **도구 출력 원문**을 받는다.

★왜 (2026-08-22 · x481 격리 측정 · 사용자 지적):
  093 의 apy 가 4.025(정답 4.275)로 나온 원인은 `customer_products` **표기**였다.
  계좌 레코드는 `level: "Green Account"` 이고 KB 페어링도 `"Green Account (checking)"` 인데,
  에이전트는 그것을 **자기 말로 요약해** `"Green Checking Account"` 로 넘겼다. 그 이름은 KB
  어디에도 없으므로 서브가 페어링을 못 찾고 checking boost 를 통째로 놓친다.

★x481 (라이브 표기 고정·각 4회):
      에이전트 요약            checking 0/4 · 합 4.025   ← 라이브 재현
      요약 + 대응 지시 한 문장  checking 4/4 · 합 4.275
      **레코드 원문**          checking 4/4 · 합 4.275   ← 지시 문장 **없이** 해결
  ⇒ 설득하는 문장을 붙이는 대신 **재료를 제대로 준다**([[62]] 결정론기는 최소한).

★이 수리는 코드 자신의 설계 의도로 되돌리는 것이다. `_sub_fetch_formalize` docstring 축자:
  *"에이전트는 **참조(`ref_params`·예 account id)만** 넘기고, 서브가 레코드를 읽는다"* —
  그런데 `customer_products` 는 참조가 아니라 요약문이었다. [[71]](전달은 엔진이)·
  [[65]](재료가 메인을 거치면 손상된다)와도 같은 방향이다.

⚠[[59]] 경계: 엔진은 **도구 이름 대조 + 출력을 그대로 싣기**만 한다. 파싱·선택·요약 0 이고,
  어느 계좌인지·무엇이 boost 인지는 서브가 문서를 읽고 판단한다.
⚠[[70]] 무엇을 파는가: REFERENCE 가 길어진다(요약 한 줄 → 레코드 원문 수백 자). 서브 문맥이
  늘어 다른 성분을 놓칠 수 있다 — 다음 런 포렌식이 셀 것 = `[T2_SG_REFRAW]` 발화 수와
  그 sim 의 component 수.

오프라인 전용(모델 0·env 0). 실행: py -3 test_ref_from_outputs.py
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

LAYERS = ["a2/banking_knowledge.gate.json",
          "a2/banking_knowledge.specific.json",
          "a2/split/banking_knowledge.core.json"]
DECL = "get_correct_savings_apy"
SG = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


def iso_of(rel):
    d = json.load(io.open(os.path.join(HERE, rel), encoding="utf-8"))
    for t in (d.get("scaffold_get_tools") or []):
        if t.get("name") == DECL:
            return t.get("isolate") or {}
    return {}


print("\n[① 선언 — 3층 동일]")
sig = []
for rel in LAYERS:
    iso = iso_of(rel)
    rfo = iso.get("ref_from_outputs") or {}
    nm = rel.split("/")[-1]
    chk("%-34s ref_from_outputs 에 customer_products" % nm, "customer_products" in rfo, sorted(rfo))
    sel = (rfo.get("customer_products") or {}).get("producer_contains")
    chk("%-34s selector 가 기존 관행(producer_contains)" % nm, bool(sel), sel)
    chk("%-34s 측정 근거가 기록돼 있다(0/4 ↔ 4/4)" % nm,
        "checking 0/4" in str(iso.get("_note_ref_from_outputs") or "")
        and "4/4" in str(iso.get("_note_ref_from_outputs") or ""))
    sig.append(json.dumps(rfo, sort_keys=True, ensure_ascii=False))
chk("3층이 같은 선언", len(set(sig)) == 1, "%d 종" % len(set(sig)))

print("\n[② 엔진 배선 — 읽어 전달만]")
_blk = SG[SG.find("_rfo = iso.get(\"ref_from_outputs\")"):][:1600]
chk("배선: 선언이 있을 때만 돈다(미선언 거동 변화 0)", "if _rfo:" in _blk)
chk("배선: **원문** 필드를 쓴다(소문자본이 아니다)", "__tool_outputs_raw" in _blk)
chk("배선: 도구 **이름**만 대조한다([[59]] 파싱 0)",
    "producer_contains" in _blk and "str(n).lower()" in _blk)
chk("배선: 출력을 **그대로** 싣는다(요약·추출 0)", "join(_hit)" in _blk)
chk("배선: 못 찾으면 에이전트 인자로 폴백(fail-open)", "if _hit:" in _blk)
chk("계기: 발화를 인쇄한다(포렌식이 셀 수 있게)", "[T2_SG_REFRAW]" in _blk)

print("\n[③ 원문 필드가 기존 소비자를 깨지 않는다]")
_ev = SG[SG.find("def _evidence_ctx(orch):"):][:2200]
chk("기존 키 `__tool_outputs` 는 여전히 소문자화(grounding 대조 계약 보존)",
    '"__tool_outputs": {k: v.lower() for k, v in outs.items()}' in _ev)
chk("기존 키 `__user_text` 도 그대로", '"__user_text": " ".join(users).lower()' in _ev)
chk("새 키는 **원문**(lower 없음)", '"__tool_outputs_raw": dict(outs)' in _ev)
chk("새 키 추가 이유가 적혀 있다", "소문자" in _ev and "R3" in _ev)

print("\n[④ ⓒ 부정통제 — 오늘 수리분이 살아 있다]")
g = iso_of(LAYERS[0])
chk("ⓒ 잔액 티어 지시 보존", "tier table" in str(g.get("instructions") or ""))
chk("ⓒ ref_params 에 current_balance 보존", "current_balance" in (g.get("ref_params") or []))
chk("ⓒ 문법 스키마 보존", bool(g.get("operand_schema")))
chk("ⓒ docs 전달 선언 보존", isinstance(g.get("docs"), dict))
chk("⚠[[70]] 무엇을 파는가 명기(REFERENCE 가 길어진다)",
    "무엇을 파는가" in SG[SG.find("★R3 (2026-08-22"):][:2000]
    or "부풀" in SG[SG.find("★R3 (2026-08-22"):][:2000]
    or "미선언이면 거동 변화 0" in SG[SG.find("★R3 (2026-08-22"):][:2000])

print("\n[⑤ 같은 처방을 받은 둘째 도구 — get_interest_correction (x482)]")
# ★두 도구가 같은 뿌리의 결함을 앓았다: **재료를 서브에 제대로 전달하지 않은 것**.
#   apy 는 에이전트 전사가 계좌명을 바꿔 checking boost 를 잃었고(x481 0/4↔4/4),
#   이쪽은 `account_id` 만 주고 "getter 로 읽어라" 한 경로가 **격리에서 아예 살지 않았다**
#   — A_asis 답반환 **0/3** · N_neg 0/3(부정통제) · **B_raw 3/3**(principal 144000·apy 4.0).
IC = "get_interest_correction"
sig2 = []
for rel in LAYERS:
    d2 = json.load(io.open(os.path.join(HERE, rel), encoding="utf-8"))
    iso2 = {}
    for t in (d2.get("scaffold_get_tools") or []):
        if t.get("name") == IC:
            iso2 = t.get("isolate") or {}
    rfo2 = iso2.get("ref_from_outputs") or {}
    nm = rel.split("/")[-1]
    chk("%-34s 레코드·거래 두 재료를 선언" % nm,
        "account_records" in rfo2 and "transactions_raw" in rfo2, sorted(rfo2))
    chk("%-34s selector 가 기존 관행" % nm,
        all((rfo2.get(k) or {}).get("producer_contains") for k in rfo2))
    chk("%-34s 측정 근거 기록(0/3 ↔ 3/3)" % nm,
        "0/3" in str(iso2.get("_note_ref_from_outputs") or "")
        and "3/3" in str(iso2.get("_note_ref_from_outputs") or ""))
    chk("%-34s gold 미참조 명기([[23]])" % nm,
        "gold 미참조" in str(iso2.get("_note_ref_from_outputs") or ""))
    # ⓒ 부정통제 — 기존 계약이 살아 있다
    chk("%-34s ⓒ operand_keys 불변(principal·actual_apy)" % nm,
        (iso2.get("operand_keys") or []) == ["principal", "actual_apy"])
    chk("%-34s ⓒ getter 경로 선언 보존(폴백이 남아 있다)" % nm,
        bool(iso2.get("getter_tools")))
    sig2.append(json.dumps(rfo2, sort_keys=True, ensure_ascii=False))
chk("%-34s 3층이 같은 선언" % IC, len(set(sig2)) == 1, "%d 종" % len(set(sig2)))

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
