# -*- coding: utf-8 -*-
"""T2_DOCS_AT_WRITE 회귀 — **배달을 결정 자리로 옮겼는가**(2026-08-16·t7304 재설계·리뷰 반영판).

## 왜 옮기나 (t7303 전수 실측·리뷰 ①이 지적하고 검증이 8/8 확인)

기존 배달 자리(`SEARCH_ON_PROCEED`)는 이름과 달리 **모델의 결정 자리가 아니다**:

    축         배달 turn        손님이 요구를 진술한 msg        간격
    checking   2 (8/8)          3 (8/8)                         +1
    savings    6 (7/8)·16(1/8)  23·39·69·37 / 34·40·7·59        중앙 +29.5

배달이 **8/8 전부 요구보다 먼저** 끝나고, 재료는 한 턴만 산다(비커밋 버퍼) ⇒ 결정 순간엔 없다.
반면 *선택을 담은 write 시도*는 요구가 다 진술된 뒤이고 모델이 값을 쓰겠다고 나선 순간이다.

불변식:
  ① 기본 OFF — 플래그 없으면 write 집합·배달 자리 모두 종전(바이트 불변).
  ② 플래그 시 write 집합에 **A2 가 이미 선언한 선택-인코딩 write** 만 더한다(새 A2 키 0):
     `choice_grounding[].tool` · `recommendation_verify.action_tool`.
     ⇒ 선언 없는 write(인증·조회)는 안 걸린다 = 부정통제 태스크 보존.
  ③ 예산은 **옮긴다, 늘리지 않는다** — 플래그 시 이른 proceed 자리 배달을 비운다.
  ④ 그 자리의 재료는 **문서 본문**(중앙 스위치가 decide=False 강제).
  ⑤ 컨텍스트 가드는 실측 보정(k=3.5·오버헤드 11,000 tok)을 쓴다.
  ⑥ 엔진이 값을 고르지 않는다 — 이 경로에 argmax·순위·정답 지목 0.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
OK = True


def chk(cond, msg):
    global OK
    OK = OK and bool(cond)
    print("  %s %s" % ("✓" if cond else "✗", msg))


wr = re.search(r"_wrset = _confirm_write_tools\(a2\).{0,2200}?_wc = next\(", SRC, re.S)
wbody = wr.group(0) if wr else ""

print("[①②] 기본 OFF · A2 선언만 더한다")
chk(bool(wr), "write 집합 구성부가 존재한다")
chk('os.environ.get("T2_DOCS_AT_WRITE") == "1"' in wbody, "플래그로만 확장된다")
chk('(a2 or {}).get("choice_grounding")' in wbody and '.get("tool")' in wbody,
    "choice_grounding[].tool 을 A2 에서 읽는다")
chk('"recommendation_verify"' in wbody and 'action_tool' in wbody,
    "recommendation_verify.action_tool 을 A2 에서 읽는다")
lits = re.findall(r'"(open_bank_account\w*|apply_for_\w+|submit_\w+|log_\w+)"', wbody)
chk(not lits, "엔진에 도메인 도구 리터럴 0 — 실제: %s" % (lits or "없음"))

print("[③] 예산은 옮긴다(늘리지 않는다)")
proc = re.search(r"T2_SEARCH_ON_PROCEED\"\) == \"1\".{0,2500}?deny 아님 · 재료 %d자 배달", SRC, re.S)
pbody = proc.group(0) if proc else ""
chk(bool(proc), "proceed 자리 경로가 있다")
chk('"" if os.environ.get("T2_DOCS_AT_WRITE") == "1"' in pbody,
    "플래그 시 이른 proceed 배달을 비운다(축 소비 방지)")
chk(SRC.count('_t2_searchagent_fired", 0) < 3') >= 2, "총 예산 3 게이트 불변")

print("[④] 재료 = 문서 본문")
sw = re.search(r'if \(os\.environ\.get\("T2_PROCEED_DOCBODY"\) == "1"\s*\n\s*'
               r'or os\.environ\.get\("T2_DOCS_AT_WRITE"\) == "1"\):\s*\n\s*decide = False', SRC)
chk(bool(sw), "중앙 스위치가 두 플래그 모두에서 decide=False 를 강제한다")
chk(SRC.count("decide = False") == 1, "강제는 한 곳뿐(자리별 플립 잔존 0)")

print("[⑤] 컨텍스트 가드 — 실측 보정")
chk("(_hist + len(_cp2)) / 3.5 > (44672 - 8192 - 1024 - 11000)" in SRC,
    "k=3.5 · 캡에서 오버헤드 11,000 tok 차감(t7303 472 콜 회귀)")
chk(SRC.count("[T2_DOC_DELIVERY] skipped") == 1, "가드는 소비 지점 한 곳")

print("[⑥] 엔진이 고르지 않는다")
dw = re.search(r"_saved = \(getattr\(self, \"_t2_axis_decision\".{0,1200}?"
               r"T2_DECIDE_BEFORE_WRITE\] write 1턴 유예", SRC, re.S)
dbody = dw.group(0) if dw else ""
chk(bool(dw), "write 유예 경로가 있다")
for pat, why in ((r"\bargmax\b", "argmax"), (r"정답은", "'정답은 X'"),
                 (r"sorted\([^)]*\)\[0\]", "sorted(...)[0]")):
    chk(not re.search(pat, dbody + wbody), "%s 없음" % why)
chk("make the call again with that value" in dbody or "call again" in dbody,
    "유예 문구는 **모델이 다시 낸다**(엔진이 대신 쓰지 않는다)")

print("\n%s" % ("PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
