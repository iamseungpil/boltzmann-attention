# -*- coding: utf-8 -*-
"""회귀 — 공식 명칭 소속 (T2_WRITE_ARG_ENUM · 오프라인 · 모델 0 · env 0 · 원장 C439⒠④).

이 레버는 **소속만** 본다. 그래서 검정도 경계다 —

  ⑴ 기본 OFF · ⑵ fail-open(축 미상·후보 없음 = 침묵) · ⑶ 집합 內는 통과(선택의 옳고 그름을
  판정하지 않는다 — `Hunter Green` 은 오답이어도 통과해야 한다) · ⑷ 엔진이 고르지 않는다
  (argmax/max/sorted 0) · ⑸ 후보는 A3 에서 오고 엔진이 도메인 텍스트를 뜯지 않는다([[59]])
  · ⑹ [[64]] 후보 명단을 함께 준다 · ⑺ 상한(livelock 방지) · ⑻ A2 두 층 동기 + 값 0개.
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
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


m = re.search(r"en_fb = None\n(.*?)\n            # ★결정-선행 write", SRC, re.S)
chk("블록을 찾았다", m is not None)
body = m.group(1) if m else ""
code = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("#"))

chk("기본 OFF", 'os.environ.get("T2_WRITE_ARG_ENUM") == "1"' in code)
chk("상한이 있다(livelock 방지)", "T2_WRITE_ARG_ENUM_CAP" in code and "_t2_enum_deny" in code)
# ★2026-08-22 갱신 ①: fail-open 술어가 `_subs` → **`_names`** 로 좁혀졌다(누수 수리).
#   `_subs` 는 있는데 표시명이 하나도 없는 그룹이 실재해(`_general_` 하나뿐인 그룹),
#   `_subs` 로 판정하면 **빈 후보 명단으로 deny** 한다 = [[64]] 가 비어 버린다.
chk("fail-open: 값·축·**표시명 후보** 없으면 침묵",
    re.search(r"if not \(_val and _grp and _names\):\s*\n\s*continue", code) is not None)
# ★2026-08-22 갱신 ②: 이 검정은 **낡아서 계속 FAIL 이었다**(수리 전 HEAD 에서도 False).
#   사이에 `T2_VERDICT_GATE` 분기가 들어오면서 `if _val in _names:` 바로 다음 줄이
#   `continue` 가 아니게 됐다. 주장은 그대로다 — **집합 內면 우리가 deny 하지 않는다**.
#   그래서 그 블록 안에 `continue` 가 있고 enum-deny 조립(`candidates=`)이 **없다**를 본다.
_inset = re.search(r"if _val in _names:\n(.*?)\n {28}self\._t2_enum_deny", code, re.S)
chk("집합 內는 통과(선택 판정 안 함)",
    _inset is not None and "continue" in _inset.group(1)
    and "candidates=" not in _inset.group(1))
for pat, why in ((r"\bargmax\b", "argmax"), (r"\bmax\s*\(", "max("),
                 (r"\bmin\s*\(", "min("), (r"\bsorted\s*\([^)]*\)\s*\[", "sorted[...]")):
    chk("선택기 없음: %s" % why, not re.search(pat, code))
chk("후보는 A3 doc_index 에서", "doc_index" in code)
chk("도메인 산문 파싱 없음(re/split 없음)",
    not re.search(r"re\.(search|findall|match)\s*\(", code)
    and not re.search(r"\.split\s*\(\s*[\"'][^_]", code))
chk("[[64]] 후보 명단을 준다", "candidates=" in code)
chk("_SRC8 등록", '("write_enum", en_fb)' in SRC)
chk("배타 체인 등록",
    re.search(r"elif en_fb is not None and c is en_fb\[0\]", SRC) is not None)

# A2 두 층 -------------------------------------------------------------------
specs = {}
for lay in ("specific", "gate"):
    p = os.path.join(HERE, "a2", "banking_knowledge.%s.json" % lay)
    a = json.load(io.open(p, encoding="utf-8"))
    specs[lay] = a.get("write_arg_enum")
chk("두 층에 있다", all(specs.values()), list(specs))
chk("두 층이 바이트 동일",
    json.dumps(specs["specific"], sort_keys=True, ensure_ascii=False)
    == json.dumps(specs["gate"], sort_keys=True, ensure_ascii=False))
sp = (specs["specific"] or [{}])[0]
blob = json.dumps(sp, ensure_ascii=False)
chk("스펙에 상품 이름 값이 0개 (ADB — 상품 수에 안 비례)",
    not re.search(r"Sky Blue|Hunter Green|Gold Saver|Beige|Lime Green", blob.split('"_note"')[0]))
chk("출처가 _note 에 적혀 있다([[23]])", "_note" in sp and "env" in str(sp.get("_note")))

try:
    import ast
    ast.parse(SRC)
    chk("파싱", True)
except Exception as e:
    chk("파싱", False, e)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
