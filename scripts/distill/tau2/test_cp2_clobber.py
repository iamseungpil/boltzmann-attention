# -*- coding: utf-8 -*-
"""배달 슬롯(`_t2_cp2_pending`) **조용한 덮어쓰기 금지** 회귀 (2026-08-16·t7303 tag h).

## 무엇을 막는가

슬롯은 **하나**고 소비 지점도 하나(`[T2_DECISION_CARRY] … 부착`)다. 그래서 같은 턴 안에서
두 번째 배달이 첫 번째를 덮으면 첫 번째는 **영영 사라지는데, 로그에는 배달된 것처럼 찍힌다**.

실물 사고(t7303 tag h · task_055 4/4 sim):

    [T2_DELIVER_PRECOMMIT] 선-배달 turn=2 · 재료 50421자      ← 찍혔다
    [T2_SEARCH_ON_PROCEED] deny 아님 · 재료 247자 배달        ← 같은 턴에 덮어썼다
    [T2_DECISION_CARRY] 이 턴 재생성 버퍼에 부착 (247자)      ← 모델이 받은 것은 이것뿐
    (treat 로그 전체에 `부착 (50421자)` **0회** · 024/098 은 37833·37038자가 그대로 붙었다)

그 위에서 *"전달했는데 선택이 안 바뀐다"* 는 결론이 날 뻔했다. 실제로는 **전달된 적이 없다**
([[55]] 우리 배관 먼저 · [[25]] 우리 계기는 100% 정답 의무).

## 불변식

  ① 모든 배달 대입은 `_cp2_assign` 를 지난다(원시 `self._t2_cp2_pending = …` 은 헬퍼 안과
     소비 지점의 `None` 초기화 **둘뿐**).
  ② `_cp2_assign` 은 미소비 배달물을 다른 값으로 덮을 때 `[T2_CP2_CLOBBER]` 를 찍는다.
  ③ **거동은 안 바꾼다** — 여전히 덮어쓴다(부피를 그냥 얹으면 44,672 한도를 넘는다.
     같은 런에서 `ContextWindowExceededError` 5건이 전부 treat 에서 났다). 이 검정은
     *보이게 만드는 것*까지만 보장하고, 큐로 바꿀지·부착 시점을 옮길지는 설계 결정이다.
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


print("[①] 모든 대입이 헬퍼를 지난다")
raw = re.findall(r"^\s*self\._t2_cp2_pending = (.+)$", SRC, re.M)
chk(sorted(x.strip() for x in raw) == ["None", "text"],
    "원시 대입은 헬퍼(text)와 소비 지점(None) 둘뿐 — 실제: %s" % sorted(x.strip() for x in raw))
chk(SRC.count("_cp2_assign(self, ") >= 5,
    "배달 자리 5곳(PRECOMMIT·MATERIAL_BYPASS·ACT_DEMAND·SEARCH_ON_PROCEED·VIEW_FB)이 헬퍼 경유")
for tag in ("PRECOMMIT", "MATERIAL_BYPASS", "ACT_DEMAND", "SEARCH_ON_PROCEED", "VIEW_FB"):
    chk('_cp2_assign(self, ' in SRC and '"%s"' % tag in SRC, "태그 %s 존재" % tag)

print("[②] 덮어쓰기가 보인다")
m = re.search(r"def _cp2_assign\(self, text, tag\):.*?self\._t2_cp2_pending = text", SRC, re.S)
body = m.group(0) if m else ""
chk(bool(m), "_cp2_assign 정의가 있다")
chk("T2_CP2_CLOBBER" in body, "미소비 배달물을 버릴 때 [T2_CP2_CLOBBER] 를 찍는다")
chk("_prev and _prev != text" in body, "같은 값 재배달은 경보하지 않는다(잡음 방지)")
chk("len(_prev)" in body, "**버린 자수**를 남긴다 — 사고 당시 이 수가 없어 50421자 소실을 못 봤다")

print("[③] 거동 불변")
chk(body.strip().endswith("self._t2_cp2_pending = text"),
    "여전히 덮어쓴다(큐 아님) — 이 검정은 가시성만 보장한다")
consume = re.search(r"_cp2 = getattr\(self, \"_t2_cp2_pending\", None\).{0,400}?"
                    r"이 턴 재생성 버퍼에 부착", SRC, re.S)
chk(bool(consume), "소비 지점은 여전히 하나(재생성 버퍼 부착)")

print("\n%s" % ("PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
