# -*- coding: utf-8 -*-
"""회귀 — `isolate.answer_format` 의 값 자리에 **복사 가능한 실제 값**을 두지 않는다.

★왜 (2026-08-22 · t7337/t7338 093 실시간 포렌식 · [[55]] "우리 문구" 단계):
  `get_interest_correction` 의 격리 서브가 **두 런에서 모두** `principal=0.0; actual_apy=0.0`
  을 냈고, 그 값이 원장에 없어 폐기·폴백됐다(t7337 1회 · t7338 4회 — 비결정성이 아니라
  **재현되는** 실패다). 그 두 숫자는 우리가 준 형식 예시
      `Reply with exactly one JSON object and nothing else: {"principal": 0.0, "actual_apy": 0.0}`
  의 값과 **정확히 같다**. 저축계좌 잔액이 0.0 일 수 없으므로 계산 결과가 아니라 **예시 복사**다
  ([[42]] 선행연구: copy=induction-head — 모델은 형식 예시의 값을 그대로 베낀다).
  폐기 → 폴백에서 메인이 낸 추측값이 grounding 에 걸려 도구가 None → 모델이 amount 를
  자기 계산해 write → `T2_WRITE_EVIDENCE` deny → 반복(093 하나가 30분+ 를 태웠다).

⇒ 값 자리는 **자리표시자**여야 한다. 같은 파일이 이미 그 관행을 증명한다:
  `"source": "<verbatim quote>"` · `"date": "MM/DD/YYYY"` · `"transaction_id": "..."`.
  문자열 자리는 자리표시자인데 **숫자 자리만** 실제 값이었다 — 저작 시점의 누락이다.

⚠️[[70]] 무엇을 파는가: 예시가 덜 구체적이라 서브가 형식을 틀릴 여지가 는다(JSON 이 아닌
  답). 그래서 `<number>` 로 **타입은 남긴다**. 다음 런이 세는 것 = `마감-답 값이 서브 출력에
  부재(principal=0.0` 발화 수(0 이어야 한다) ↔ 서브 형식 오류로 인한 폐기 수.

오프라인 전용(모델 0·env 0). 실행: py -3 test_a2_answer_format_placeholder.py
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

# [[24]] A2 는 층이 여럿이고 **양방향**으로 동기화돼야 한다 — 하나만 고치면 죽은 코드가 된다.
LAYERS = ["a2/banking_knowledge.gate.json",
          "a2/banking_knowledge.specific.json",
          "a2/split/banking_knowledge.core.json"]
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


def afs_of(path):
    d = json.load(io.open(path, encoding="utf-8"))
    return {t.get("name"): ((t.get("isolate") or {}).get("answer_format") or "")
            for t in (d.get("scaffold_get_tools") or [])
            if ((t.get("isolate") or {}).get("answer_format"))}


def value_side(af):
    """`... nothing else: {…}` 의 **형식 본문**만 — 앞머리 산문은 검사 대상이 아니다."""
    return af.split(":", 1)[-1] if ":" in af else af


print("\n[① 값 자리에 복사 가능한 숫자가 없다]")
seen = {}
for rel in LAYERS:
    p = os.path.join(HERE, rel)
    if not os.path.exists(p):
        chk("층 존재: %s" % rel, False, "파일 없음")
        continue
    afs = afs_of(p)
    seen[rel] = afs
    for name, af in sorted(afs.items()):
        body = value_side(af)
        # 날짜 형식 표시(MM/DD/YYYY)는 값이 아니라 **형식**이라 예외다.
        probe = body.replace("MM/DD/YYYY", "")
        nums = re.findall(r":\s*(-?\d+\.?\d*)", probe)
        chk("%-34s %s" % (rel.split("/")[-1], name), not nums, nums or "")

print("\n[② 타입은 남긴다 — 자리표시자 형식]")
for rel, afs in seen.items():
    for name, af in sorted(afs.items()):
        body = value_side(af)
        if "<number>" in body:
            chk("%-34s %s: <number> 자리표시자" % (rel.split("/")[-1], name), True)

print("\n[③ 세 층이 바이트 동일 ([[24]] 양방향)]")
sigs = {rel: json.dumps(afs, sort_keys=True, ensure_ascii=False) for rel, afs in seen.items()}
chk("세 층의 answer_format 전체가 동일", len(set(sigs.values())) == 1,
    "%d 종" % len(set(sigs.values())))
chk("층마다 answer_format 이 같은 개수", len({len(a) for a in seen.values()}) == 1,
    {r.split("/")[-1]: len(a) for r, a in seen.items()})

print("\n[④ 부정통제 — 기존 자리표시자 관행이 보존됐다]")
g = seen.get(LAYERS[0], {})
chk("`<verbatim quote>` 인용 자리표시자 보존",
    any("<verbatim quote>" in v for v in g.values()))
chk("`MM/DD/YYYY` 날짜 형식 표시 보존",
    any("MM/DD/YYYY" in v for v in g.values()))
chk("열거형 표시(base|checking|...) 보존",
    any("|" in value_side(v) for v in g.values()))
chk("JSON 하나만 내라는 지시 보존",
    all("exactly one JSON object" in v for v in g.values()))

print("\n[⑤ 근본 수리 — 문법으로 형식을 강제한다 (T2_SG_SCHEMA)]")
# ★자리표시자는 *복사할 값*을 없앨 뿐 형식 준수를 **보장하지 않는다**(사용자 지적 2026-08-22:
#   "숫자대신 문자로 대체하면 복사하지 않는건가?"). 보장은 문법이 준다 — 격리 서브의 **마감
#   라운드는 구조적으로 도구가 없으므로**(`_tl = None if (_last or not tools)`) 거기에만 건다.
#   ⛔도구가 있는 라운드에 걸면 tool_calls 가 0 이 되어 서브가 레코드를 못 읽는다(C248).
SG = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
_blk = SG[SG.find("_last = (rnd == _maxr - 1)"):][:2600]

for rel, afs in seen.items():
    d = json.load(io.open(os.path.join(HERE, rel), encoding="utf-8"))
    for t in (d.get("scaffold_get_tools") or []):
        iso = t.get("isolate") or {}
        if not iso.get("answer_format"):
            continue
        sc = iso.get("operand_schema")
        nm = t.get("name")
        chk("%-34s %s: operand_schema 선언" % (rel.split("/")[-1], nm), bool(sc))
        if not sc:
            continue
        req = sc.get("required") or []
        chk("%-34s %s: derivation 이 첫 required(추론이 값보다 앞)"
            % (rel.split("/")[-1], nm), req[:1] == ["derivation"], req)
        chk("%-34s %s: 스키마에 복사 가능한 숫자 0"
            % (rel.split("/")[-1], nm),
            not [c for c in json.dumps(sc) if c.isdigit()])

chk("배선: 문법은 **도구가 없는 라운드에만** 걸린다(tool_calls 0 회피·C248)",
    "_tl is None and os.environ.get(\"T2_SG_SCHEMA\")" in _blk)
chk("배선: 스키마 출처가 A2 하나뿐이다(엔진 리터럴 0·[[05]])",
    'iso["operand_schema"]' in _blk and "operand_schema" in _blk)
chk("배선: vLLM 문법 계약(guided_json + xgrammar)",
    'guided_json' in _blk and 'xgrammar' in _blk)
chk("배선: 문법 경로에서 tools 를 제거한다",
    '_kw.pop("tools", None)' in _blk)
chk("배선: 기본 OFF(플래그 미설정이면 거동 변화 0)",
    'os.environ.get("T2_SG_SCHEMA") == "1"' in _blk)
chk("배선: 미선언 도구는 skip(fail-open)", 'and iso.get("operand_schema")' in _blk)
chk("래칫: go_stack 정본에 등재됐다([[73]])",
    "T2_SG_SCHEMA" in io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read())
_GS = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
_gsblk = _GS[_GS.find("★T2_SG_SCHEMA"):][:1400]
chk("⚠[[70]] 무엇을 파는가 명기(추론 자리를 잃는 것 ↔ derivation 으로 되산다)",
    "무엇을 파는가" in _gsblk and "derivation" in _gsblk)
chk("⚠도구 라운드 금지 근거가 등재문에 적혀 있다(C248)",
    "tool_calls" in _gsblk and "0" in _gsblk)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
