# -*- coding: utf-8 -*-
"""T2_SUB_REQUIREMENT 회귀 — **요구를 서브에 넣되 엔진은 뽑지 않는다**(2026-08-17·x343).

근거(x343·n=24=8×3·블록 편차 0): 결정 서브가 문서+후보줄만 받으면 `Gold Account` **24/24 오답**,
손님 요구를 축자로 받으면 gold **24/24 정답**, 무관한 요구면 **0/24**(부정통제 통과).
⇒ 라이브 실패의 원인은 재료가 아니라 **요구가 서브에 없다**는 것. 이 레버는 그 한 줄을 고친다.

불변식:
  ① 기본 OFF — 플래그 없으면 요구 주입 0(종전 거동·바이트 불변).
  ② **엔진은 추출하지 않는다** — 인용은 LLM 이 내고(A2 `requirement_prompt`), 엔진은
     `in` 연산으로 **원문 존재만** 확인한다. 이 경로에 정규식 0([[59]] 2026-08-16 강화판).
  ③ 검증 통과분만 싣는다 — 통과 0이면 아무것도 안 붙인다(fail-safe).
  ④ A2 프롬프트는 **도메인 어휘 0**(어느 도메인에서도 같은 문장·[[05]]).
  ⑤ 정본 진입점은 `t2_search.sub_requirements` 하나(사본 금지·[[67]]).
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
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
SEA = io.open(os.path.join(HERE, "t2_search.py"), encoding="utf-8").read()
OK = True


def chk(cond, msg):
    global OK
    OK = OK and bool(cond)
    print("  %s %s" % ("✓" if cond else "✗", msg))


blk = re.search(r'_reqs = \[\].{0,1600}?_ctpl = _po\.get\("decide_candidates_text"\)', SRC, re.S)
body = blk.group(0) if blk else ""

print("[①] 기본 OFF")
chk(bool(blk), "요구 주입 블록이 존재한다")
chk('os.environ.get("T2_SUB_REQUIREMENT") == "1"' in body, "플래그로만 켜진다")
chk('_po.get("requirement_prompt")' in body, "A2 프롬프트가 없으면 발화하지 않는다")

print("[②] 엔진은 뽑지 않는다 — 존재확인만")
chk("_ts.quote_in(_qs, _utxt)" in body,
    "인용의 **원문 존재**만 확인한다(정본 `quote_in`·2026-08-17 C510 로 이설)")
for pat, why in ((r"re\.(search|findall|match|finditer|compile|sub|split)", "정규식"),
                 (r"\.split\(\s*[\"']", "구분자 split")):
    chk(not re.search(pat, body), "주입 경로에 %s 없음" % why)
sub = re.search(r"def sub_requirements\(.*?return \[x for x in out", SEA, re.S)
sbody = sub.group(0) if sub else ""
chk(bool(sub), "정본 진입점 t2_search.sub_requirements 가 있다")
chk(not re.search(r"re\.(search|findall|match|finditer|compile)", sbody),
    "정본 진입점에도 정규식 없음 — JSON 경계는 find/rfind")
chk('raw.find("[")' in sbody and 'raw.rfind("]")' in sbody, "문자열 연산으로만 JSON 을 자른다")

print("[③] fail-safe")
chk('if _reqs:' in SRC, "검증 통과분이 있을 때만 붙인다")
chk("Customer's stated request:" in SRC, "요구는 머리에 붙는다(x343 구성)")

print("[④] A2 프롬프트는 도메인-일반")
a2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                       encoding="utf-8"))
tpl = ((a2.get("policy_ontology") or {}).get("requirement_prompt") or "")
chk(bool(tpl), "A2 에 requirement_prompt 가 있다")
chk("{messages}" in tpl, "손님 발화를 통째로 받는 자리 하나뿐")
low = tpl.lower()
for w in ("account", "card", "bank", "savings", "checking", "atm", "apy"):
    chk(w not in low, "도메인 낱말 '%s' 없음" % w)
gate = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"),
                         encoding="utf-8"))
chk(((gate.get("policy_ontology") or {}).get("requirement_prompt") or "") == tpl,
    "gate.json 과 바이트 동일(([[24]]) 동기화)")

print("\n%s" % ("PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
