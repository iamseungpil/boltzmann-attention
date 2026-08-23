# -*- coding: utf-8 -*-
"""행동 검정 — CP2 배달 생애 원장이 **닫힌 분할**인가 (`_cp2_open` / `_cp2_close` · R4).

★왜 (2026-08-23 · 원장 C502 · 감사 `CP2_QUEUE_AUDIT_2026_08_23.md`):
  t7303 A/B 가 무효가 된 이유는 결손이 아니라 **계기**였다 — 1차 종점이 `[T2_CP2_APPEND] …
  (queue)` 였는데 그 줄은 **플래그가 꺼진 팔에서는 존재할 수 없다**. 그래서 `0/8 → 8/8` 은
  측정이 아니라 처치 배정의 재인쇄였다(C502 축자).
  그리고 지금 쓰던 계기도 같은 병이었다(실측): 보관 사이드카 14파일 전수에서 `decision_carry`
  행의 `arrived` 가 **100% True**(303행·False 0)이고 그 행 수는 도달 수가 아니라 **VIEW_FB 대입
  수와 1:1** 이다 — 다섯 배달 자리 중 **하나만** 등재하고 있었다([[25]] 우리 계기는 100% 정답 의무).

⇒ 이 검정이 지키는 계약 넷:
  ① **닫힌 분할** — 배달물 하나는 `attached | clobbered | ctx_skip` 중 **정확히 하나**로 끝나거나
     미결로 남는다. `대입 = 종결 + 미결` 이 성립해야 도달률의 분모가 선다.
  ② **팔-대칭** — 원장 어디에서도 `T2_CP2_QUEUE` 를 읽지 않는다(소스로 강제).
  ③ **병합본은 조각 전부를 닫는다** — 부착 단위로 세면 큐 ON 이 2건을 1건으로 접어 도달률이
     구조적으로 낮아지고, 그 순간 두 팔의 분모 정의가 달라져 A/B 가 또 무효가 된다.
  ④ **거동 불변** — 슬롯·work·fb 어디에도 대입하지 않는다(모델 가시 바이트 0).

오프라인 전용(모델 0·env 0·사이드카는 가짜로 가로챈다). 실행: py -3 test_cp2_ledger.py
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OK = True


def chk(cond, msg):
    global OK
    print("  %s %s" % ("✓" if cond else "✗", msg))
    if not cond:
        OK = False


SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()


def body_of(name):
    m = re.search(r"\ndef %s\(.*?\n(?=\ndef |\n# ─|\Z)" % re.escape(name), SRC, re.S)
    return m.group(0) if m else ""


# ── 가짜 사이드카 — 실제 파일을 안 쓰고 행만 모은다 ────────────────────────────
ROWS = []


class _FakeSidecar(object):
    @staticmethod
    def record(kind, text, messages=None, **meta):
        ROWS.append(dict(meta, kind=kind, _n=len(text or "")))

    @staticmethod
    def record_many(items, messages=None, **meta):
        pass


class _FakeBeat(object):
    @staticmethod
    def current_sim():
        return "task_XXX#s1"

    @staticmethod
    def current_turn():
        return 7


# ★순서가 중요하다: **진짜 모듈로 먼저 import** 한 뒤에 가짜를 끼운다.
#   `t2_gate_patch` 는 모듈 레벨에서 `from t2_lever_beat import beat` 를 하므로, 먼저 가짜를
#   끼우면 import 자체가 죽는다(1차 실행에서 그렇게 죽었다). 원장 헬퍼는 **함수 안에서**
#   `import t2_fbsidecar` / `import t2_lever_beat` 를 하므로 나중에 갈아끼워도 잡힌다.
import t2_gate_patch as G                                            # noqa: E402

sys.modules["t2_fbsidecar"] = _FakeSidecar
sys.modules["t2_lever_beat"] = _FakeBeat


class Slot(object):
    """빈 슬롯. **앞 배달물을 넣을 때는 `seeded()` 를 쓴다** — 슬롯에 직접 꽂으면 원장에 열린 적이
    없어 `_cp2_close` 가 닫을 것을 못 찾는다(1차 실행에서 ③④가 그렇게 실패했다). 라이브에서는
    모든 대입이 `_cp2_assign` 을 지나므로 '열림 없는 pending' 은 생기지 않는다."""


def seeded(text, tag="PRECOMMIT"):
    """라이브와 같은 경로로 앞 배달물을 채운 슬롯."""
    s = Slot()
    del ROWS[:]
    G._cp2_assign(s, text, tag)
    del ROWS[:]
    return s


def run(fn, *a, **k):
    del ROWS[:]
    fn(*a, **k)
    return list(ROWS)


A = "a" * 243
B = "b" * 247
C = "c" * 100

print("[①] 대입이 분모를 만든다 — assign 행")
s = Slot()
r = run(G._cp2_assign, s, A, "SEARCH_ON_PROCEED")
chk(len([x for x in r if x.get("ev") == "assign"]) == 1, "새 배달물 1건 → assign 1행")
chk(r and r[0].get("agent") == "cp2", "채널이 'cp2' 다 — 기존 decision_carry 와 섞이지 않는다")
chk(r and r[0].get("cp2_tag") == "SEARCH_ON_PROCEED" and r[0].get("cp2_n") == 243,
    "태그와 자수를 남긴다")
chk(r and r[0].get("cp2_disp") == "fresh", "빈 슬롯 채우기는 disp=fresh")

print("\n[②] 같은 값 재대입은 **새 배달이 아니다**")
r = run(G._cp2_assign, seeded(A), A, "VIEW_FB")
chk(not r, "행 0 — 분모·분자를 동시에 부풀리지 않는다")

print("\n[③] 덮어쓰기 — 앞 건은 clobbered 로 **닫히고** 새 건이 열린다")
r = run(G._cp2_assign, seeded(A), B, "SEARCH_ON_PROCEED")
cl = [x for x in r if x.get("outcome") == "clobbered"]
op = [x for x in r if x.get("ev") == "assign"]
chk(len(cl) == 1, "앞 건 close(clobbered) 1행")
chk(len(op) == 1 and op[0].get("cp2_disp") == "clobber", "새 건 assign 1행 · disp=clobber")
chk(cl and cl[0].get("cp2_n") == 243, "닫힌 것은 **앞 건**(243자)이다")

print("\n[④] 큐 병합 — 두 조각이 **둘 다 열린 채** 남는다")
os.environ["T2_CP2_QUEUE"] = "1"
try:
    s2 = seeded(A)
    r = run(G._cp2_assign, s2, B, "SEARCH_ON_PROCEED")
    chk(not [x for x in r if x.get("outcome") == "clobbered"], "병합이면 clobbered 를 찍지 않는다")
    chk(len([x for x in r if x.get("ev") == "assign"]) == 1
        and r[-1].get("cp2_disp") == "append", "새 조각은 disp=append 로 열린다")
    chk(len(getattr(s2, "_t2_cp2_track", [])) == 2, "★슬롯에 **두 조각**이 열려 있다")
    r2 = run(G._cp2_close, s2, "attached", 492)
    outs = [x for x in r2 if x.get("ev") == "close"]
    chk(len(outs) == 2, "★한 번 닫으면 **조각 둘 다** 닫힌다(부착 단위가 아니라 배달물 단위)")
    chk(all(x.get("outcome") == "attached" and x.get("cp2_slot_n") == 492 for x in outs),
        "둘 다 같은 outcome·같은 슬롯 자수를 단다")
    chk(not getattr(s2, "_t2_cp2_track", []), "닫은 뒤 미결 0")
finally:
    os.environ.pop("T2_CP2_QUEUE", None)

print("\n[⑤] outcome 은 셋뿐이고 via 는 선택 라벨이다")
s3 = Slot()
run(G._cp2_assign, s3, C, "ACT_DEMAND")
r = run(G._cp2_close, s3, "attached", 100, "asub")
chk(r and r[0].get("cp2_via") == "asub", "via='asub' 를 적는다 — 커밋 생성기에 못 간 회차의 표식")
s4 = Slot()
run(G._cp2_assign, s4, C, "PRECOMMIT")
r = run(G._cp2_close, s4, "ctx_skip")
chk(r and r[0].get("outcome") == "ctx_skip" and "cp2_via" not in r[0],
    "via 를 안 주면 키 자체가 없다(잡음 0)")

print("\n[⑥] 닫힌 분할 — 대입 = 종결 + 미결")
s5 = Slot()
assigns = closes = 0
for i, (txt, tag) in enumerate([(A, "PRECOMMIT"), (B, "VIEW_FB"), (C, "ACT_DEMAND")]):
    assigns += len([x for x in run(G._cp2_assign, s5, txt, tag) if x.get("ev") == "assign"])
    closes += len([x for x in run(G._cp2_close, s5, "attached", len(txt)) if x.get("ev") == "close"])
chk(assigns == 3 and closes == 3, "대입 3 = 종결 3 · 미결 0 (검산식 성립)")

print("\n[⑦] ★팔-대칭 — 원장은 큐 플래그를 **읽지 않는다**")
led = body_of("_cp2_open") + body_of("_cp2_close")
chk("T2_CP2_QUEUE" not in led, "`_cp2_open`/`_cp2_close` 안에 T2_CP2_QUEUE 참조 0")
chk("arrived" not in led,
    "`arrived` 를 쓰지 않는다 — 부착=도달이라 그 이름은 부착 인쇄의 재인쇄가 된다")

print("\n[⑧] 거동 불변 — 원장은 슬롯·버퍼에 대입하지 않는다")
for bad in ("_t2_cp2_pending =", "_t2_cp2_said =", "work =", "fb ="):
    chk(bad not in led, "원장 안에 `%s` 대입 없음" % bad)

print("\n[⑨] 실패해도 런을 깨지 않는다")
bad_self = object()          # 속성 대입이 불가능한 객체
try:
    r = run(G._cp2_open, bad_self, A, "VIEW_FB", "fresh")
    chk(True, "예외를 올리지 않는다(무시 로그만)")
except Exception as e:
    chk(False, "예외가 올라왔다: %r" % (e,))

print("")
print("[9b] 한계 — 원장에 **열린 적 없는** pending 은 닫히지 않는다")
orphan = Slot()
orphan._t2_cp2_pending = A          # 원장을 우회해 꽂힌 상태(라이브에는 없다)
r = run(G._cp2_close, orphan, "attached", 243)
chk(not r, "행 0 — `close` 는 `open` 이 연 것만 닫는다. `open` 이 예외로 죽으면 그 배달물은 "
           "원장에서 통째로 사라진다(분모·분자 동시 누락). 그 예외는 stderr 로 인쇄되므로 "
           "감사 스크립트가 그 줄을 세야 한다.")

print("\n[⑩] 종결은 **생성이 돌아온 뒤**에만 찍힌다 (호출 자리)")
m = re.search(r"am = _am_sub or _gen\(self, work.*?\n(.*?\n){0,12}?.*?_cp2_close\(self, \"attached\"",
              SRC, re.S)
chk(bool(m), "`attached` 종결이 `am = … _gen(…)` **뒤**에 있다")
chk(SRC.count('_cp2_close(self, "attached"') == 1, "attached 종결 자리는 하나뿐")
chk('_cp2_close(self, "ctx_skip")' in SRC, "ctx_skip 종결이 창 초과 분기에 있다")
chk('_cp2_close(self, "clobbered")' in SRC, "clobbered 종결이 대입 분기에 있다")

print("\n%s" % ("PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
