# -*- coding: utf-8 -*-
"""서브 관용구 사본-수 래칫 (2026-08-14·사용자 지시 "계속 같은 일을 반복하고 있다").

단발-격리 서브 호출(la.generate + UserMessage 관용구)의 정본은 `t2_subcall` 하나다.
이 검정은 파일별 인라인 사본 수를 **기록된 예산 이하**로 강제한다 — 새 채널이 인라인으로
짜면 여기서 떨어지고, 에러 문구가 정본으로 안내한다. 예산을 줄이는 것은 언제나 허용
(그때 이 표를 함께 줄여라)·늘리는 것은 리뷰 없이 불가.

예산 근거(2026-08-14 리팩토링 종료 시점 실측):
  t2_gate_patch.py 8 = 메인-루프 재생성 6(tools=self.tools·서브 아님) + 다중-메시지 서브 2
    (`_gen_action_sub`·8451 — 단발-프롬프트 계약이 아니라 이관 제외·확장 시 t2_subcall 에
    messages-list 지원을 추가하고 이관할 것)
  t2_scaffold_get.py 3 = 다회전 getter-루프 서브 3(428·504·911 — 도구 실행 동반·messages 누적·
    별도 가족·한 파일에 응집. 단발이던 sg_inject 3곳은 2026-08-14 이관 완료)
  나머지 엔진 파일 = 0 (전부 이관 완료)
"""
import io
import os
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET = {
    "t2_resolve.py": 0,
    "t2_ledger.py": 0,
    "t2_search.py": 0,
    "t2_source.py": 0,
    "t2_eplan_patch.py": 0,
    "t2_formalize_exec.py": 0,
    "t2_declfirst.py": 0,
    "t2_compute.py": 0,
    "t2_fbsidecar.py": 0,
    "t2_handoff_ground.py": 0,
    "t2_factdag.py": 0,
    "t2_gate_patch.py": 8,     # 메인-루프 6 + 다중-메시지 서브 2 (위 주석)
    "t2_scaffold_get.py": 3,   # 다회전 getter-루프 가족(428·504·911)
}
FAILS = []
for f, cap in sorted(BUDGET.items()):
    p = os.path.join(HERE, f)
    if not os.path.exists(p):
        continue
    n = io.open(p, encoding="utf-8", errors="replace").read().count("la.generate(")
    ok = n <= cap
    print("%-4s %-24s la.generate %d (예산 %d)" % ("PASS" if ok else "FAIL", f, n, cap))
    if not ok:
        FAILS.append(f)

# UserMessage 관용구도 같은 래칫 (정본 = t2_subcall.make_user_message)
UM_BUDGET = {"t2_resolve.py": 0, "t2_ledger.py": 3, "t2_search.py": 1, "t2_source.py": 0,
             "t2_eplan_patch.py": 0, "t2_formalize_exec.py": 0}
for f, cap in sorted(UM_BUDGET.items()):
    p = os.path.join(HERE, f)
    if not os.path.exists(p):
        continue
    n = io.open(p, encoding="utf-8", errors="replace").read().count("except TypeError")
    ok = n <= cap
    print("%-4s %-24s except-TypeError %d (예산 %d)" % ("PASS" if ok else "FAIL", f, n, cap))
    if not ok:
        FAILS.append(f + ":um")

print("=" * 60)
if FAILS:
    print("FAIL — 인라인 서브 관용구가 늘었다. 새 채널은 t2_subcall.sub_generate/"
          "parse_contract/grounded_calls 를 쓰라. 파일:", FAILS)
else:
    print("PASS — 사본 예산 준수")
sys.exit(1 if FAILS else 0)
