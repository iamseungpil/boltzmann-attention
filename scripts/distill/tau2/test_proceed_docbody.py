# -*- coding: utf-8 -*-
"""T2_PROCEED_DOCBODY 회귀 — **배달 객체만 바꿨는가**(2026-08-16·t7304=S1 재설계).

설계서 = `S1_REDESIGN_T7304_2026_08_16.md` §2·§8.

불변식:
  ① 기본 OFF — 플래그 없으면 `decide=True`(종전 서브-결정 경로) 그대로.
  ② 플래그 시 그 자리만 `decide=False`(문서 본문) — **다른 배달 자리(PRECOMMIT 등)는 불변**.
  ③ 컨텍스트 가드 존재 — 보수 추정(자수/3)·초과 시 **건너뛰고 기록**(축약·선별 금지 [[62]]③).
  ④ 예산·슬롯·축 소비 불변 — 같은 카운터(`_t2_searchagent_fired < 3`)·같은 `_cp2_assign` 경유.
  ⑤ 엔진 선택 문장 0 — 이 블록에 argmax/정답 지목이 없다.
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


# SEARCH_ON_PROCEED 블록(플래그 정의부터 배달 로그까지)을 잡는다.
m = re.search(r'_docb = os\.environ\.get\("T2_PROCEED_DOCBODY"\) == "1".{0,2400}?'
              r'deny 아님 · 재료 %d자 배달', SRC, re.S)
body = m.group(0) if m else ""

print("[①] 기본 OFF")
chk(bool(m), "T2_PROCEED_DOCBODY 블록이 존재한다")
chk("decide=(not _docb)" in body, "플래그가 꺼져 있으면 decide=True(종전 경로)")

print("[②] 다른 배달 자리는 불변")
pre = re.search(r'if \(os\.environ\.get\("T2_DELIVER_PRECOMMIT"\).{0,1200}?재료 %d자"',
                SRC, re.S)
chk(bool(pre) and "decide=False" in pre.group(0) and "T2_PROCEED_DOCBODY" not in pre.group(0),
    "PRECOMMIT 자리는 이 플래그와 무관(종전 그대로)")
chk(SRC.count('os.environ.get("T2_PROCEED_DOCBODY")') == 1,
    "플래그 참조는 SEARCH_ON_PROCEED 자리 하나뿐")

print("[③] 컨텍스트 가드")
chk("_docb and len(_mp) > 5000" in body, "가드는 문서-본문 배달에만 건다(소형 재료는 종전대로)")
chk("(_hist + len(_mp)) / 3 > (44672 - 8192 - 1024)" in body,
    "보수 추정: (히스토리+배달물)/3 자 > 한도−출력상한−여유")
chk("[T2_DOC_DELIVERY] skipped" in body, "초과 시 건너뛰고 **기록**한다(부작용 표 계상)")
chk(not re.search(r"_mp\s*=\s*_mp\[", body), "축약하지 않는다 — 자르는 코드가 없다([[62]]③)")

print("[④] 예산·슬롯·축 소비 불변")
chk('_t2_searchagent_fired", 0) < 3' in SRC.split("_docb = ")[0][-2000:] or
    re.search(r'_t2_searchagent_fired", 0\) < 3\).{0,600}?_docb = ', SRC, re.S) is not None,
    "예산 3 게이트가 이 블록 앞에 그대로 있다")
chk('_cp2_assign(self, _mp, "SEARCH_ON_PROCEED")' in body, "슬롯은 같은 헬퍼(_cp2_assign) 경유")
sm = re.search(r"def _search_material\(agent, a2, messages, decide=True\):.{0,12000}"
               r"if not decide:.{0,600}?_done\.add\(_g\)", SRC, re.S)
chk(bool(sm), "_search_material 의 decide=False 경로도 축을 소비한다(_done.add)")

print("[⑤] 엔진 선택 문장 0")
for pat, why in ((r"\bargmax\b", "argmax"), (r"정답은", "'정답은 X'"),
                 (r"\bbest\b", "best 지목"), (r"\brecommend", "recommend 지목")):
    chk(not re.search(pat, body), "블록에 %s 없음" % why)

print("\n%s" % ("PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
