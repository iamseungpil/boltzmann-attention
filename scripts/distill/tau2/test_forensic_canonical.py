# -*- coding: utf-8 -*-
"""포렌식 로더 사본-수 래칫 (2026-08-14 야간·사용자 지시 "라이브러리화 일반화를 잊지 말라").

궤적 읽기(결과 gz 경로·`load`·래퍼 해제 `nameof/argsof/inner_name/label`·`write_tools`)의 정본은
`t2_forensic` 하나다. 이 검정은 파일별 인라인 사본 수를 **기록된 예산 이하**로 강제한다.

왜 래칫이 필요한가 — 사본은 조용히 갈라진다(실측 2건):
  · `bank_miss_turn_audit.write_tools` 는 `tool_name` 을 안 봤고 `bank_trigger_window_audit` 은 봤다
    — 같은 이름의 함수가 서로 다른 집합을 반환하고 있었다.
  · C470 의 계기 4수리(래퍼 대상-도구 색인·중첩 `9.50↔9.5` 정규화·시도vs실행·seed 매칭)는 한
    사본에만 들어갔고, 그 갈라짐이 "073 의 성공 실행을 NOTCALLED 로 오분류"를 낳았다.

예산 근거(2026-08-14 야간 이관 종료 시점 실측):
  t2_forensic.py                 = 정본(제외)
  bank_fail_forensic_all.py      = 0 (얇은 위임 래퍼만 남김·출력 바이트 동일 확인)
  bank_miss_turn_audit.py        = 0
  bank_trigger_window_audit.py   = 0
  bank_bailout_audit.py          = 0 (처음부터 라이브러리 위에)
  미이관 잔여 = 아래 LEGACY — 옛 런 전용 포맷이라 그때 손대며 함께 이관할 것(늘리지 말 것)
"""
import io
import os
import re
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CANON = "t2_forensic.py"

# 인라인 사본의 표지 — 결과파일을 **직접 여는** 것과 래퍼를 **직접 해제**하는 것
PATTERNS = (
    re.compile(r'gzip\.open\('),
    re.compile(r'"_results\.json\.gz"'),
    re.compile(r'\.get\("function"\)\s*or\s*\{\}'),
    re.compile(r'"agent_tool_name"\)\s*or'),
)

BUDGET = {
    "bank_fail_forensic_all.py": 0,
    "bank_miss_turn_audit.py": 0,
    "bank_trigger_window_audit.py": 0,
    "bank_bailout_audit.py": 0,
}
# 옛 런 전용 포맷(태그 규칙이 다르거나 sim 구조가 옛것) — 손댈 때 함께 이관한다.
LEGACY = {
    "bank_nt4_forensic.py": 1,
    "bank_iso_forensic.py": 1,
    "bank_scaffold_forensic.py": 0,
    "bank_reach_forensic.py": 0,
    "bank_t7285_credit_forensic.py": 2,
    "bank_assertion_arm_forensic.py": 2,
    "bank_dup_exec_audit.py": 0,
    "bank_xmatch_forensic.py": 0,
}


def count(fn):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        return None
    s = io.open(p, encoding="utf-8", errors="replace").read()
    return sum(len(pat.findall(s)) for pat in PATTERNS)


def main():
    bad = []
    print("정본 = %s\n" % CANON)
    if not os.path.exists(os.path.join(HERE, CANON)):
        print("FAIL: 정본 %s 이 없다" % CANON)
        return 1
    for table, tag in ((BUDGET, "이관"), (LEGACY, "미이관")):
        for fn, budget in sorted(table.items()):
            n = count(fn)
            if n is None:
                print("  --   %-32s (파일 없음·건너뜀)" % fn)
                continue
            ok = n <= budget
            bad += [] if ok else ["%s: 사본 %d > 예산 %d" % (fn, n, budget)]
            print("  %s %-32s 사본 %d / 예산 %d  [%s]"
                  % ("ok  " if ok else "FAIL", fn, n, budget, tag))
    print()
    if bad:
        print("FAIL — 인라인 사본이 예산을 넘었다:")
        for b in bad:
            print("   · %s" % b)
        print("\n정본을 쓰라:  import t2_forensic as F  →  F.sims(tag) · F.calls(sim) · "
              "F.label(F.nameof(tc), F.argsof(tc)) · F.write_tools(tag)")
        print("정본에 없는 기능이면 **정본에 추가**하고 여기 예산은 그대로 두어라.")
        return 1
    print("PASS — 인라인 사본 예산 준수")
    return 0


if __name__ == "__main__":
    sys.exit(main())
