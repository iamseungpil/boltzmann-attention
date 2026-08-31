# TASK_4 (t7391_reg12) — 본문 위치 포인터

정본 본문 = `reports/facet_rft_2026/tasks__20260829/TASK_4.md`

이 디렉터리(`tasks_reg12/`)에 `TASK_4.md` 를 직접 두지 못한 이유:
`.claude/hooks/scaffold_guard.py:201` 의 런별-포렌식 예외 술어가 `r"/tasks_+\d{8}/"` 라
`tasks_reg12/TASK_4.md` 는 exit 2 로 막힌다. 훅을 우회하지 않고 정본 명명
`tasks__<날짜>/TASK_<id>.md` (선례 `tasks__20260824/` · 형제 `x602_TASK_3_pointer.md`)를 따랐다.

한 줄 요약: reward 0.0 = `DB 1.0` × `NL_ASSERTION 0.0`.
변이 집합 clean(matched 2/2). 실패 칸은 `"10"` 숫자 하나 — 모델이 msg 26 에서 **12** 를 발화.
같은 seed 대조군(`hist_gpt52_reg12_PASS`)은 도구 출력 꼬리에 `[COMPUTED FACTS] … : 10` 을 받고
msg 10 에서 즉시 10 을 옮겨 pass. t7391 은 `T2_CALC` 플래그 부재로 그 문자열이 **0회** 생성됐다.
