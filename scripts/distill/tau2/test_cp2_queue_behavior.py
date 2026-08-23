# -*- coding: utf-8 -*-
"""`_cp2_assign` **거동** 검정 — 소스 문자열이 아니라 **함수를 실행해서** 잰다 (2026-08-23).

`test_cp2_clobber.py` 는 소스 문자열 대조(정규식)다. 그것은 *코드가 그렇게 쓰여 있다* 만
증명하고 *그렇게 동작한다* 는 증명하지 않는다. 이 파일은 진짜 `t2_gate_patch._cp2_assign` 를
import 해서 가짜 `self` 에 대고 돌리고, `sys.stderr` 로 나가는 계기 로그까지 잡아 대조한다.

§A 계약(설계서가 주장하는 것)
  ① `T2_CP2_QUEUE` OFF · 소형 pending + 소형 신규      → 덮어씀(구판 바이트 보존)
  ② OFF · ≥10k pending + 소형 신규                     → 이어붙임(구판 anti-clobber 생존)
  ③ ON  · 소형 + 소형                                  → 이어붙임 · 순서 = pending → 신규
  ④ ON  · `T2_CP2_APPEND_MAX` 초과                     → 덮어쓰되 **로그가 그 사실을 말한다**
  ⑤ 같은 텍스트 2회 배달                               → CLOBBER 로그 없음(오경보 없음)

§B 차등 검정(구판 replica 와 직접 비교)
  커밋 9d217b39 이전 구현을 **축자 replica** 로 두고, 플래그 OFF 에서 두 구현의 결과
  `_t2_cp2_pending` 바이트를 행렬로 대조한다. 커밋 메시지의 주장은
    *"Default off, so control bytes are unchanged until it is measured."*
  이고, 이 절은 그 주장을 실행으로 검산한다.

⛔이 파일은 원격 GPU·tau2 런을 일절 쓰지 않는다(순수 오프라인·무료).
"""
import io
import os
import sys
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as G  # noqa: E402

ASSIGN = G._cp2_assign

OK = True
FINDINGS = []


def chk(cond, msg):
    global OK
    OK = OK and bool(cond)
    print("  %s %s" % ("PASS" if cond else "FAIL", msg))
    return bool(cond)


class Slot(object):
    """가짜 `self` — 라이브 orchestrator 가 노출하는 것 중 이 함수가 만지는 것은 슬롯 하나뿐이다."""

    def __init__(self, pending=None):
        if pending is not None:
            self._t2_cp2_pending = pending


def run(slot, text, tag, queue=None, cap=None):
    """`_cp2_assign` 1회 실행 → (슬롯값, stderr 전문). env 는 호출 단위로 세팅/복원."""
    old = {k: os.environ.get(k) for k in ("T2_CP2_QUEUE", "T2_CP2_APPEND_MAX")}
    try:
        if queue is None:
            os.environ.pop("T2_CP2_QUEUE", None)
        else:
            os.environ["T2_CP2_QUEUE"] = str(queue)
        if cap is None:
            os.environ.pop("T2_CP2_APPEND_MAX", None)
        else:
            os.environ["T2_CP2_APPEND_MAX"] = str(cap)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            ASSIGN(slot, text, tag)
        return getattr(slot, "_t2_cp2_pending", None), buf.getvalue()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── 구판 replica (커밋 9d217b39 의 `-` 줄 축자) ────────────────────────────────
def old_assign(slot, text, tag, log):
    _prev = getattr(slot, "_t2_cp2_pending", None)
    if _prev and _prev != text and len(_prev) >= 10000 and text:
        log("[T2_CP2_APPEND] %s: 미소비 대용량 %d자 뒤에 %d자 이어붙임"
            % (tag, len(_prev), len(text)))
        text = _prev + "\n\n" + text
    elif _prev and _prev != text:
        log("[T2_CP2_CLOBBER] %s 가 미소비 배달물 %d자를 버리고 %d자로 덮어씀"
            % (tag, len(_prev), len(text or "")))
    slot._t2_cp2_pending = text


A = "A" * 243          # t7346 098#s626729 에서 **사라진** 배달물의 자수
B = "B" * 247          # 그것을 덮어쓴 SEARCH_ON_PROCEED 결정문의 자수
BIG = "D" * 50421      # t7303 tag h 에서 사라진 PRECOMMIT 문서 본문의 자수

print("§A 계약 검정 (실행)")

print("\n[①] 플래그 OFF · 소형 + 소형 → 덮어씀")
v, log = run(Slot(A), B, "SEARCH_ON_PROCEED", queue=0)
chk(v == B, "슬롯 = 신규 247자만 (구판 거동 보존)")
chk(A not in v, "앞 배달물 243자는 **사라진다** — 라이브 t7346 098 과 같은 소실")
chk("[T2_CP2_CLOBBER]" in log, "CLOBBER 로그가 찍힌다")
chk("243자를 버리고 247자로 덮어씀" in log, "버린 자수·덮은 자수를 축자로 남긴다")
chk("상한" not in log, "OFF 에서는 상한 문구가 붙지 않는다")

print("\n[②] 플래그 OFF · >=10k pending + 소형 신규 → 이어붙임 (구판 anti-clobber 생존)")
v, log = run(Slot(BIG), B, "SEARCH_ON_PROCEED", queue=0)
chk(v == BIG + "\n\n" + B, "슬롯 = 구 50421자 + 개행2 + 신규 247자")
chk("[T2_CP2_APPEND]" in log and "미소비 대용량" in log and "(대용량)" not in log,
    "구판 구제 경로는 **구판 축자 문구** 그대로 — 과거 런 로그를 grep 하는 포렌식이 두 문구를 "
    "받게 하지 않는다 (2026-08-23 수리: 초판이 OFF 문구까지 바꿨다)")
chk("[T2_CP2_CLOBBER]" not in log, "CLOBBER 는 찍히지 않는다")

print("\n[③] 플래그 ON · 소형 + 소형 → 이어붙임 · 순서 = pending → 신규")
v, log = run(Slot(A), B, "SEARCH_ON_PROCEED", queue=1)
chk(v == A + "\n\n" + B, "슬롯 = 앞 243자 + 개행2 + 뒤 247자 (둘 다 산다)")
chk(v.startswith(A) and v.endswith(B), "순서는 pending 이 앞·신규가 뒤")
chk("[T2_CP2_APPEND]" in log and "(queue)" in log, "APPEND 로그 · 사유 = queue")
chk("[T2_CP2_CLOBBER]" not in log, "CLOBBER 는 찍히지 않는다")

print("\n[③b] 플래그 ON · 같은 턴 3연속 배달 → 셋 다 순서대로 남는다")
s = Slot()
run(s, A, "PRECOMMIT", queue=1)
run(s, B, "ACT_DEMAND", queue=1)
v, _ = run(s, "C" * 10, "VIEW_FB", queue=1)
chk(v == A + "\n\n" + B + "\n\n" + "C" * 10, "3건 누적 · 배달 순서 보존")

print("\n[④] 플래그 ON · 상한 초과 → 덮어쓰되 로그가 그 사실을 말한다")
v, log = run(Slot(A), B, "SEARCH_ON_PROCEED", queue=1, cap=100)
chk(v == B, "이어붙이지 않고 신규로 덮어썼다")
chk("[T2_CP2_CLOBBER]" in log, "CLOBBER 로그")
chk("상한 100 초과라 이어붙이지 못함" in log, "로그가 **상한 때문**임을 축자로 밝힌다")

print("\n[④b] 상한 경계 — `len(prev)+len(new)+2 <= cap` 가 정확히 경계다")
p, n = "P" * 49, "N" * 49
v, _ = run(Slot(p), n, "T", queue=1, cap=100)          # 49+49+2 = 100 <= 100
chk(v == p + "\n\n" + n and len(v) == 100, "합계 == 상한이면 이어붙인다(길이 100)")
v, log = run(Slot(p), n, "T", queue=1, cap=99)         # 100 > 99
chk(v == n and "상한 99 초과" in log, "합계가 상한을 1 넘으면 덮어쓰고 사유를 밝힌다")

print("\n[⑤] 같은 텍스트 2회 배달 → 오경보 없음")
s = Slot()
run(s, A, "PRECOMMIT", queue=1)
v, log = run(s, A, "SEARCH_ON_PROCEED", queue=1)
chk(v == A, "슬롯은 그대로(중복 누적 없음)")
chk("[T2_CP2_CLOBBER]" not in log, "CLOBBER 로그 없음")
chk("[T2_CP2_APPEND]" not in log, "APPEND 로그도 없음 — 같은 값은 이어붙이지도 않는다")
v, log = run(Slot(A), A, "SEARCH_ON_PROCEED", queue=0)
chk(v == A and log == "", "OFF 에서도 동일 · stderr 전무")

print("\n[⑤b] 첫 배달(pending 없음) → 로그 없음")
v, log = run(Slot(), A, "PRECOMMIT", queue=1)
chk(v == A and log == "", "빈 슬롯 채우기는 조용하다")
v, log = run(Slot(""), A, "PRECOMMIT", queue=1)
chk(v == A and log == "", "빈 문자열 pending 도 falsy 라 조용하다")

print("\n[⑥] 빈 신규 배달물은 **슬롯을 지운다** (계약에 없던 경로)")
v, log = run(Slot(A), "", "VIEW_FB", queue=1)
chk(v == A, "큐 ON: 빈 배달물은 배달이 아니다 — pending 243자를 **유지**한다")
chk("빈 배달물" in log and "유지" in log, "유지했다는 사실을 로그로 남긴다")
v2, log2 = run(Slot(A), "", "VIEW_FB", queue=0)
chk(v2 == "" and "243자를 버리고 0자로 덮어씀" in log2,
    "OFF 는 구판 그대로 지운다(바이트 불변)")
chk("상한" not in log, "빈 배달은 상한 문구가 안 붙는다(text falsy)")
if v == "":
    FINDINGS.append(
        "빈 문자열 배달(`text=''`)은 큐 ON 에서도 이어붙임 분기(`and text`)를 못 타고 "
        "pending 을 **지운다**. 라이브에서 배달 재료가 빈 문자열로 계산되는 자리가 있으면 "
        "큐를 켜도 소실은 그대로다.")

print("\n§B 차등 검정 — 플래그 OFF 가 정말 구판과 같은 바이트를 내는가")
MATRIX = [
    ("소형↔소형", 243, 247),
    ("소형↔대형", 243, 50421),
    ("대형↔소형", 50421, 247),
    ("대형↔대형(t7303 실물 자수)", 50421, 50421),
    ("문서본문 2건(37833+37038)", 37833, 37038),
    ("상한 직하", 40000, 49998),
    ("상한 직상", 40000, 49999),
]
diverge = []
for name, lp, ln in MATRIX:
    prev, new = "p" * lp, "n" * ln
    got_new, log_new = run(Slot(prev), new, "PRECOMMIT", queue=0, cap=90000)
    s_old = Slot(prev)
    old_assign(s_old, new, "PRECOMMIT", lambda _m: None)
    got_old = s_old._t2_cp2_pending
    same = (got_new == got_old)
    print("  %s %-28s prev=%-6d new=%-6d 신판=%-7d자 구판=%-7d자"
          % ("same" if same else "DIFF", name, lp, ln, len(got_new), len(got_old)))
    if not same:
        diverge.append((name, lp, ln, len(got_new), len(got_old), log_new.strip()))

if diverge:
    print("\n  ⚠ OFF 인데 구판과 결과가 다른 조합 %d건:" % len(diverge))
    for name, lp, ln, n_new, n_old, log_new in diverge:
        print("    · %s (prev=%d, new=%d): 신판 %d자(=신규만) / 구판 %d자(=이어붙임)"
              % (name, lp, ln, n_new, n_old))
        print("      신판 로그: %s" % (log_new or "(없음)"))
    FINDINGS.append(
        "플래그 OFF 는 **바이트 불변이 아니다**. 신설된 `T2_CP2_APPEND_MAX` 조건이 "
        "`_queue` 밖에 걸려 있어서, 구판이 이어붙이던 `len(prev)>=10000 and "
        "len(prev)+len(new)+2 > cap` 영역이 이제 **덮어쓰기**로 바뀐다. go_stack.sh:375 가 "
        "`T2_CP2_APPEND_MAX=90000` 을 항상 export 하므로 라이브에서 유효하다. "
        "게다가 그 CLOBBER 로그에는 상한 문구가 **안 붙는다**(접미사 조건이 `_queue and ...`) — "
        "가장 큰 소실(문서 본문급)이 가장 이유 없이 사라진다([[64]]).")

chk(not diverge,
    "OFF 경로가 구판과 바이트 동일 — 커밋 메시지 축자 *\"Default off, so control bytes "
    "are unchanged until it is measured.\"*")

print("\n[§B-b] 로그 문구는 OFF 에서도 바뀌었다(계기 채널·에이전트 비가시)")
_, log_new = run(Slot(BIG), B, "PRECOMMIT", queue=0, cap=90000)
_buf = []
old_assign(Slot(BIG), B, "PRECOMMIT", _buf.append)
old_line = _buf[0] if _buf else ""
print("    구판: %s" % old_line)
print("    신판: %s" % log_new.strip())
if old_line and old_line != log_new.strip():
    FINDINGS.append(
        "OFF 에서도 APPEND 로그 문구가 바뀌었다(구판 `미소비 대용량 N자 뒤에` → 신판 "
        "`미소비 N자 뒤에 … (대용량)`). stderr 계기라 모델에는 안 보이지만, 과거 런 로그를 "
        "grep 하는 포렌식 스크립트가 있으면 문구를 둘 다 받아야 한다.")

print("\n[⑧] 큐 ON × 소비 지점 `_ctx_fits` 가드 상호작용 — 켠 쪽이 **덜** 배달할 수 있다")
# 소비 지점(t2_gate_patch.py:10248)은 배달물이 **>=5000자일 때만** 창 검사를 하고, 안 들어가면
# 슬롯을 통째로 `None` 으로 비운다(부분 배달 없음). 개별로는 5000 미만이던 두 배달물이
# 이어붙어 5000 을 넘으면, 그때부터 검사 대상이 된다.


class Msg(object):
    def __init__(self, content):
        self.role, self.content = "assistant", content


work = [Msg("h" * 80000)]                     # 긴 히스토리(자수 기준·실측 보정 산식은 `_ctx_fits`)
ta, tb = "a" * 4536, "b" * 2000               # 자수는 로컬 보고서 실측 배달물 크기대로
off_slot = Slot()
run(off_slot, ta, "PRECOMMIT", queue=0)
off_v, _ = run(off_slot, tb, "SEARCH_ON_PROCEED", queue=0)
on_slot = Slot()
run(on_slot, ta, "PRECOMMIT", queue=1)
on_v, _ = run(on_slot, tb, "SEARCH_ON_PROCEED", queue=1)
off_fit, _ = G._ctx_fits(work, off_v)
on_fit, _ = G._ctx_fits(work, on_v)
off_delivered = len(off_v) if off_fit else 0
on_delivered = len(on_v) if on_fit else 0
print("    OFF: 슬롯 %d자 · 가드 통과=%s → 모델이 받는 자수 %d"
      % (len(off_v), off_fit, off_delivered))
print("    ON : 슬롯 %d자 · 가드 통과=%s → 모델이 받는 자수 %d"
      % (len(on_v), on_fit, on_delivered))
chk(off_delivered > 0, "OFF 는 뒤엣것(2000자)만이라도 배달된다(<5000 이라 검사 자체를 안 받음)")
if on_delivered < off_delivered:
    FINDINGS.append(
        "큐 ON 이 **더 적게** 배달하는 국면이 있다. 개별로는 5000자 미만이라 창 검사를 아예 "
        "안 받던 배달물 둘(4536+2000)이 이어붙어 6538자가 되면 소비 지점 가드"
        "(`t2_gate_patch.py:10248` `if _cp2 and len(_cp2) >= 5000`)의 검사 대상이 되고, "
        "히스토리가 길면 **슬롯 통째로 None** 이 된다(부분 배달 없음) — OFF 는 2000자를 "
        "받았는데 ON 은 0자를 받는다. [[70]] 이 레버가 파는 것 중 하나가 이것이다.")

print("\n[⑦] 상한 env 가 정수가 아니면 **크래시**한다(try 밖·가드 없음)")
crashed = None
try:
    run(Slot(A), B, "PRECOMMIT", queue=1, cap="")
except Exception as e:
    crashed = "%s: %s" % (type(e).__name__, e)
print("    결과: %s" % (crashed or "예외 없음"))
if crashed:
    FINDINGS.append(
        "`T2_CP2_APPEND_MAX` 가 빈 문자열/비정수면 `int()` 가 `try` 밖에서 터진다 → %s. "
        "5개 배달 자리 전부가 이 경로를 지나므로 sim 이 죽는다. 현재 go_stack.sh:375 가 정수를 "
        "주므로 라이브에서는 잠복이다(같은 함수 주석이 `_sys` NameError 를 '잠복'이라 부른 것과 "
        "동종)." % crashed)

print("\n" + "=" * 72)
if FINDINGS:
    print("[FINDINGS] %d건 — 설계 서술과 실제 거동이 다르거나, 서술에 없던 경로" % len(FINDINGS))
    for i, f in enumerate(FINDINGS, 1):
        print("  (%d) %s" % (i, f))
    print("")
    print("  [[64]] 고치는 법(한 줄) — 상한 조건을 큐 분기 안으로 넣어 OFF 를 진짜 불변으로:")
    print("      if _prev and _prev != text and text and (")
    print("              (_queue and len(_prev) + len(text) + 2 <= _cap)")
    print("              or len(_prev) >= 10000):")
    print("    그리고 CLOBBER 접미사의 `_queue and` 를 떼어 상한 사유가 항상 보이게 한다.")
print("=" * 72)
print("\n%s" % ("PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
