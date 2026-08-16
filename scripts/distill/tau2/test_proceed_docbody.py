# -*- coding: utf-8 -*-
"""T2_PROCEED_DOCBODY 회귀 — **배달 객체만 바꿨는가**(2026-08-16·t7304=S1 재설계·심사 반영판).

설계서 = `S1_REDESIGN_T7304_2026_08_16.md` §2·§8. 심사 3인 일치 수정 3건을 봉인한다.

불변식:
  ① 기본 OFF — 플래그 없으면 `decide` 인자 그대로(종전 경로·바이트 불변).
  ② **중앙 스위치** — 플래그 시 `_search_material` **함수 안**에서 `decide=False` 강제
     ⇒ 호출 자리 5곳(PRECOMMIT·MATERIAL_BYPASS·SEARCH_ON_PROCEED·VIEW_FB·DECIDE-FIRST) 자동 커버.
     한 자리만 플립하면 다른 자리가 축을 먼저 소비해(`_t2_search_done` 전역·영구) 문서가 그 축에
     영영 못 오는 누수가 생긴다 — 심사 3인 일치 지적.
  ③ 컨텍스트 가드는 **소비 지점 하나**(부착 직전) — 대용량(≥5k)만·보수 추정(자수/3)·초과 시
     **건너뛰고 기록**(축약·선별 금지 [[62]]③). 자리별 가드 잔존 0.
  ④ 대용량 anti-clobber — 미소비 ≥10k 배달물은 버리지 않고 **이어붙임**(`[T2_CP2_APPEND]`).
     소형끼리는 종전대로 덮어씀(ctl 바이트 불변).
  ⑤ 예산·슬롯·축 소비 불변 — 같은 카운터(`< 3`)·같은 `_cp2_assign`·decide=False 경로도 축 소비 **영속**.
  ⑥ 엔진 선택 문장 0.
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


fn = re.search(r"def _search_material\(agent, a2, messages, decide=True\):.{0,3000}?"
               r"import t2_search as _ts", SRC, re.S)
head = fn.group(0) if fn else ""

print("[①②] 기본 OFF · 중앙 스위치")
chk(bool(fn), "_search_material(…, decide=True) 시그니처 불변")
# ★2026-08-16 t7304: 스위치가 두 플래그(T2_PROCEED_DOCBODY · T2_DOCS_AT_WRITE) OR 로 확장됐다.
#   불변식은 같다 — **함수 안 한 곳**에서 decide 를 강제한다.
chk('os.environ.get("T2_PROCEED_DOCBODY") == "1"' in head and "decide = False" in head,
    "플래그 시 함수 안에서 decide=False 강제(호출 자리 무관)")
chk(SRC.count('os.environ.get("T2_PROCEED_DOCBODY")') == 1,
    "플래그 참조는 함수 안 한 곳뿐(자리별 플립 잔존 0)")
calls = re.findall(r"_search_material\(self, a2, state\.messages[^)]*\)", SRC)
chk(len(calls) >= 5, "호출 자리 ≥5 (전부 스위치가 덮는다) — 실제 %d" % len(calls))
chk(not any("_docb" in c for c in calls), "호출 자리에 자리별 decide 플립 없음")

print("[③] 컨텍스트 가드 — 소비 지점 하나")
cons = re.search(r'_cp2 = getattr\(self, "_t2_cp2_pending", None\).{0,3000}?'
                 r"이 턴 재생성 버퍼에 부착", SRC, re.S)   # 창 확대: 가드 주석이 길어졌다
cbody = cons.group(0) if cons else ""
chk(bool(cons), "소비 지점이 존재한다")
chk("len(_cp2) >= 5000" in cbody, "대용량(≥5k)만 검사")
chk("(_hist + len(_cp2)) / 3.5 > (44672 - 8192 - 1024 - 11000)" in cbody,
    "보수 추정(실측 보정 k=3.5·오버헤드 11,000)")
chk("[T2_DOC_DELIVERY] skipped" in cbody, "초과 시 건너뛰고 기록")
chk(SRC.count("[T2_DOC_DELIVERY] skipped") == 1, "가드는 한 곳뿐(자리별 가드 잔존 0)")
chk(not re.search(r"_cp2\s*=\s*_cp2\[", cbody), "축약하지 않는다([[62]]③)")

print("[④] 대용량 anti-clobber")
asg = re.search(r"def _cp2_assign\(self, text, tag\):.*?self\._t2_cp2_pending = text", SRC, re.S)
abody = asg.group(0) if asg else ""
chk("len(_prev) >= 10000" in abody and "[T2_CP2_APPEND]" in abody,
    "미소비 ≥10k 는 이어붙임(+로그)")
chk('text = _prev + "\\n\\n" + text' in abody, "이어붙임 = 구분자 결합(내용 무판단)")
chk("[T2_CP2_CLOBBER]" in abody, "소형은 종전대로 덮어쓰고 기록(ctl 바이트 불변)")

print("[⑤] 예산·슬롯·축 소비")
chk(SRC.count('_t2_searchagent_fired", 0) < 3') >= 2, "예산 3 게이트 그대로")
chk('_cp2_assign(self, _mp, "SEARCH_ON_PROCEED")' in SRC, "슬롯은 같은 헬퍼 경유")
sm = re.search(r"if not decide:.{0,900}?_done\.add\(_g\).{0,900}?agent\._t2_search_done = _done",
               SRC, re.S)
chk(bool(sm), "decide=False 경로가 축 소비를 영속한다")

print("[⑥] 엔진 선택 문장 0")
for pat, why in ((r"\bargmax\b", "argmax"), (r"정답은", "'정답은 X'")):
    chk(not re.search(pat, head + cbody + abody), "%s 없음" % why)

print("\n%s" % ("PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
