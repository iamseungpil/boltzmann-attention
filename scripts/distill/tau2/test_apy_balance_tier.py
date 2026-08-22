# -*- coding: utf-8 -*-
"""회귀 — 격리 서브가 **잔액-조건부 base APY 티어**를 고를 수 있게 선언돼 있다.

★왜 (2026-08-22 · x480 원인 확정 + x481 격리 프로브):
  093 은 네 런 연속 reward 0.0 인데 갈린 것은 **base APY 하나**였다 — 산출 2.775 ↔ 정답 4.275.
  정책 문서(`doc_savings_accounts_silver_account_003`)가 그 1.5 를 설명한다:
      | Below threshold       | Less than $10,000  | 2.5% |
      | At or above threshold | At least $10,000   | 4.0% |
  계좌 잔액은 144,000 이므로 문서상 base 는 4.0 이고, 검산도 맞는다 —
  `144000 × (4.275 − 4.0)/100 / 12 = 33.0` = gold amount.
  그런데 서브의 REFERENCE 에는 **잔액이 없었고**, 지시문은 base 가 하나인 것처럼 말했다.

★x481 격리 프로브 (4팔 × 4회·`reports/facet_rft_2026/x481_apy_tier_probe.json`):
      A_asis     0/4      기준선
      N_neg      0/4      부정통제([[57]]) — 반복해도 안 나온다
      B_bal      0/4      **잔액만 주면 전혀 안 된다**
      C_bal_hint 4/4      **잔액 + 지시 한 문장이면 100%**
  ⇒ 결정적인 것은 재료가 아니라 **지시**다. 잔액만 넣는 수리는 헛수고였을 것이고,
    프로브가 그것을 막았다([[62]] 0순위가 있는 이유).

⚠[[62]] 경계: 엔진은 티어를 **고르지 않는다** — 서브가 문서의 표를 읽고 고른다. 그래서 이
  검정의 핵심 항목은 *"지시문에 임계·값 리터럴이 없다"* 이다([[05]]). gold 미참조([[23]]).
⚠[[70]] 무엇을 파는가: 지시문이 한 문장 길어지고, 에이전트가 인자를 하나 더 채워야 한다.
  안 채우면 종전 거동으로 떨어진다(fail-open) — 다음 런 포렌식이 셀 것 = 이 인자가 실린 호출 수.

오프라인 전용(모델 0·env 0). 실행: py -3 test_apy_balance_tier.py
"""
import io
import json
import os
import re
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
BAL = "current_balance"
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


def tool_of(rel):
    d = json.load(io.open(os.path.join(HERE, rel), encoding="utf-8"))
    for t in (d.get("scaffold_get_tools") or []):
        if t.get("name") == DECL:
            return t
    return None


print("\n[① 잔액이 REFERENCE 에 실린다]")
for rel in LAYERS:
    t = tool_of(rel)
    nm = rel.split("/")[-1]
    if not t:
        chk("%-34s 선언 존재" % nm, False)
        continue
    iso = t.get("isolate") or {}
    chk("%-34s ref_params 에 %s" % (nm, BAL), BAL in (iso.get("ref_params") or []),
        iso.get("ref_params"))
    chk("%-34s params 에 %s 선언(에이전트가 채운다)" % (nm, BAL), BAL in (t.get("params") or {}))
    sf = [x.get("param") for x in ((t.get("ground") or {}).get("scalar_fields") or [])]
    chk("%-34s ground 가 %s 를 원장 대조한다(날조 차단)" % (nm, BAL), BAL in sf, sf)

print("\n[② 티어 지시가 두 지시문에 다 있다]")
KEY = "tier table"
for rel in LAYERS:
    t = tool_of(rel)
    nm = rel.split("/")[-1]
    iso = (t or {}).get("isolate") or {}
    chk("%-34s isolate.instructions" % nm, KEY in str(iso.get("instructions") or ""))
    docs = iso.get("docs")
    if isinstance(docs, dict) and docs.get("instructions"):
        chk("%-34s isolate.docs.instructions (라이브 기본 경로)" % nm,
            KEY in str(docs.get("instructions")))

print("\n[③ ★[[05]] 엔진/선언에 도메인 수치 리터럴 0 — 엔진이 고르지 않는다]")
# 프로브가 검증한 문구는 문서를 **읽으라**고만 한다. 임계(10000)나 값(4.0/2.5)이 지시문에
# 들어가면 그 순간 우리가 정답을 떠먹이는 것이고([[62]]), 다른 계좌로 전이되지 않는다([[05]]).
BAD = [r"10[,\.]?000", r"\b4\.0\s*%", r"\b2\.5\s*%", r"\b4\.275\b", r"\b2\.775\b"]
for rel in LAYERS:
    t = tool_of(rel)
    nm = rel.split("/")[-1]
    iso = (t or {}).get("isolate") or {}
    blob = " ".join([str(iso.get("instructions") or ""),
                     str(((iso.get("docs") or {}) if isinstance(iso.get("docs"), dict) else {})
                         .get("instructions") or ""),
                     str((t or {}).get("params", {}).get(BAL) or "")])
    hits = [p for p in BAD if re.search(p, blob)]
    chk("%-34s 지시·인자 설명에 임계/값 리터럴 0" % nm, not hits, hits)

print("\n[④ 출처·규율 표기]")
for rel in LAYERS:
    t = tool_of(rel)
    nm = rel.split("/")[-1]
    note = str((t or {}).get("_note_balance_tier") or "")
    chk("%-34s 프로브 근거가 기록돼 있다(A/N/B/C 수치)" % nm,
        "B_bal 0/4" in note and "C_bal_hint 4/4" in note)
    chk("%-34s gold 미참조 명기([[23]])" % nm, "gold 미참조" in note)

print("\n[⑤ 세 층이 같은 처방을 쓴다 ([[24]] 양방향)]")
sig = []
for rel in LAYERS:
    t = tool_of(rel) or {}
    iso = t.get("isolate") or {}
    sig.append(json.dumps({"ref": iso.get("ref_params"),
                           "ins": iso.get("instructions"),
                           "params": sorted(t.get("params") or {}),
                           "ground": t.get("ground")}, sort_keys=True, ensure_ascii=False))
chk("ref/지시/params/ground 가 3층 동일", len(set(sig)) == 1, "%d 종" % len(set(sig)))
# ⓒ 부정통제 — 기존 계약이 살아 있다(수리가 다른 것을 지우지 않았다)
g = tool_of(LAYERS[0]) or {}
iso = g.get("isolate") or {}
chk("ⓒ 기존 array 근거 계약(components 인용 검증) 보존",
    any((x or {}).get("param") == "components"
        for x in ((g.get("ground") or {}).get("array_fields") or [])))
chk("ⓒ operand_keys 불변(components)", (iso.get("ref_params") and
    (iso.get("operand_keys") or []) == ["components"]), iso.get("operand_keys"))
chk("ⓒ 문법 스키마(오늘 수리분) 보존", bool(iso.get("operand_schema")))

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
