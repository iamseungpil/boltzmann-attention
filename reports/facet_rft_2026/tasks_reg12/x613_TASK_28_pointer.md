# TASK_28 (t7391_reg12) — 본문 위치 포인터

정본 본문 = `reports/facet_rft_2026/tasks__20260829/TASK_28.md`
격리 프로브 = `reports/facet_rft_2026/x613_t7391_task28_cancel_iso.py`

이 디렉터리(`tasks_reg12/`)에 `TASK_28.md` 를 직접 두지 못한 이유:
`.claude/hooks/scaffold_guard.py:200-201` 의 §74-b 런별-포렌식 예외 술어가
`^TASK_\d+[a-z]?\.md$` ∧ `r"/tasks_+\d{8}/"` 라 `tasks_reg12/TASK_28.md` 는 exit 2 다.
훅을 우회하지 않고 정본 명명 `tasks__<날짜>/TASK_<id>.md` 를 따랐다
(선례 `x602_TASK_3_pointer.md` · `x603_TASK_4_pointer.md` · `x603_TASK_9_pointer.md` ·
`x611_TASK_12_pointer.md` · `x612_TASK_22_pointer.md`).

한 줄 요약: `reward 0.0 = DB 0.0 × NL_ASSERTION 0.0`. 변이 집합 = MISSING 0 · WRONGARG 0 ·
DUP 0 · **EXTRA 1**. gold 에 없는 `cancel_pending_order(#W2575533, "no longer needed")`
한 건이 DB 를 죽였고, 그 환불 $1,619.34 가 총액에 섞여 NL 축의 $918.43 도 함께 죽었다.
손님은 msg 1 에서 그 write 를 금지했다(*"only the hose—if that means canceling the whole
order, please don't"*). 모델은 msg 13 에서 **다른 행동**(품목 제거·나머지 유지)을 설명한 뒤
msg 14 의 맨 `"Yes."` 로 msg 15 에서 전체 취소를 실행했고, `reason` 값은 대화 어디에도
없던 날조다(전 role·tool_calls 포함 0회).

우리 층 두 칸(둘 다 코드 경로 지목):
- `T2_WRITE_ARG_GROUND`(ON) · `T2_RULE_AT_WRITE`(ON) 가 **retail 선언 부재로 완전 무발화** —
  `t2_gate_patch.py:8161` · `:11710-11711` → `:3303-3321`. `write_arg_grounding`·`write_rules`
  는 `a2/banking_knowledge.*.json` 에만 있다. 격리 `x613` 에서 선언 한 줄이면 그 호출이
  정확히 거부된다(부정통제 2종 통과).
- `G2_CONFIRM_WRITE` 가 확인의 **대상**을 안 본다 — `gate_interpreter.py:387-390`.
  선언 predicate 는 *"of the action details"* 인데 구현은 `CONFIRM_RE.search(last_user_msg)`
  하나다. ⚠`TASK_12.md` 의 처방 P1 은 이 케이스를 **못 잡는다**(`prevTxt=True`).

⚠로그 미회수: `t7391_reg12.log.gz`·`fb_*`·`trace_*` **0건** ⇒ `[T2_*]` 마커 계수 불가([[30]]).
