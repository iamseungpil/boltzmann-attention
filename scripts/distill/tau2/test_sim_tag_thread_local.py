# -*- coding: utf-8 -*-
"""회귀 검정 (C325): sim 태그는 **스레드별**이고, stderr 줄마다 붙으며, 기존 파서를 안 깬다.

무엇을 막는 검정인가 — 구판은 `_SIM = [None]` 전역 하나를 프로세스가 공유했다. 러너는 sim을
`ThreadPoolExecutor`로 동시에 돌리므로(`tau2/runner/batch.py`), 마지막으로 이름을 심은
스레드의 값이 **다른 sim의 줄에 찍힌다**. 실측에서 beat 3줄이 전부 한 sim 이름을 달았는데
레코드마다 sim을 지니는 사이드카로 대조하니 반대쪽 sim의 것이었고, 그 잘못된 태그를 근거로
원장 등재까지 갔다가 되돌렸다. 태그는 *있다*보다 *맞다*가 중요하다([[25]]).

오프라인 전용(서버·LLM 불요). 실행: py -3 test_sim_tag_thread_local.py
"""
import io
import os
import re
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_lever_beat as LB                                    # noqa: E402

FAILED = []


def chk(cond, label):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        FAILED.append(label)


class _Task(object):
    def __init__(self, tid):
        self.id = tid


class _Orch(object):
    def __init__(self, tid):
        self.task = _Task(tid)


def main():
    # ── ① 두 스레드가 서로의 sim을 덮지 않는다 ────────────────────────────
    seen = {}
    barrier = threading.Barrier(2)

    def worker(tid):
        LB.set_sim_from(_Orch(tid))
        barrier.wait()                 # 상대가 자기 값을 심을 시간을 준다(경합 재현)
        seen[tid] = LB.current_sim()

    ts = [threading.Thread(target=worker, args=(t,)) for t in ("task_A", "task_B")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    chk(seen.get("task_A") == "task_A" and seen.get("task_B") == "task_B",
        "동시 두 스레드가 각자의 sim을 유지한다  ← C325가 막는 결손 (본 값: %s)" % seen)

    # ── ② 태거가 줄마다 프리픽스를 붙인다(부분 write도 줄 단위로 모은다) ──
    cap = io.StringIO()
    tagger = LB._TaggingStderr(cap)
    LB.set_sim_from(_Orch("task_Z"))
    tagger.write("[T2_LIMIT_REDUCE] emitted")     # 개행 없는 조각
    chk(cap.getvalue() == "", "개행 전에는 내보내지 않는다(줄이 쪼개지지 않는다)")
    tagger.write(" at decision point\n")
    out = cap.getvalue()
    chk(out == "[sim=task_Z] [T2_LIMIT_REDUCE] emitted at decision point\n",
        "조각 write가 한 줄로 합쳐지고 프리픽스가 한 번만 붙는다 (본 값: %r)" % out)

    # ── ③ 기존 로그 파서를 깨지 않는다 ────────────────────────────────────
    #     x134는 `search()` + **행말 앵커**다 → 접미사는 깨고 접두사는 안 깬다.
    X134 = re.compile(r"\[T2_STACK\] audit route=(\[.*?\]) chose=(\[.*?\]) differs=(True|False) "
                      r"suppressed=(\[.*?\])\s*$")
    cap2 = io.StringIO()
    tg2 = LB._TaggingStderr(cap2)
    tg2.write("[T2_STACK] audit route=[('a','b','c')] chose=[] differs=False suppressed=[]\n")
    line = cap2.getvalue().rstrip("\n")
    chk(line.startswith("[sim=task_Z] "), "태그가 붙었다")
    chk(X134.search(line) is not None, "x134 파서(행말 앵커)가 여전히 매칭한다")
    chk("[T2_LEVER] T2_X" in ("[sim=q] [T2_LEVER] T2_X"),
        "x44 파서(부분문자열 대조)도 영향 없다")

    # ── ④ sim을 모르면 종전 그대로(무기명) ────────────────────────────────
    done = {}

    def anon():
        c = io.StringIO()
        LB._TaggingStderr(c).write("plain line\n")
        done["v"] = c.getvalue()

    t = threading.Thread(target=anon)      # sim 미설정 스레드
    t.start()
    t.join()
    chk(done.get("v") == "plain line\n", "sim 미상이면 프리픽스 없음(거동 보존)")

    # ── ⑤ 설치는 멱등 ─────────────────────────────────────────────────────
    orig = sys.stderr
    try:
        chk(LB.install_stderr_tagger() is True, "최초 설치는 True")
        chk(LB.install_stderr_tagger() is False, "재설치는 False(이중 프리픽스 없음)")
    finally:
        sys.stderr = orig

    print("\n%s  (%d 실패)" % ("PASS" if not FAILED else "FAIL", len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
