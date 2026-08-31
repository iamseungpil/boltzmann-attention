# TASK_12 (t7391_reg12) — 본문 위치 포인터

정본 본문 = `reports/facet_rft_2026/tasks__20260829/TASK_12.md`

이 디렉터리(`tasks_reg12/`)에 `TASK_12.md` 를 직접 두지 못한 이유:
`.claude/hooks/scaffold_guard.py:201` 의 런별-포렌식 예외 술어가 `r"/tasks_+\d{8}/"` 라
`tasks_reg12/TASK_12.md` 는 exit 2 로 막힌다. 훅을 우회하지 않고 정본 명명
`tasks__<날짜>/TASK_<id>.md` (선례 `tasks__20260824/` · 형제 `x602_TASK_3_pointer.md`,
`x603_TASK_4_pointer.md`)를 따랐다.

딸린 격리 프로브(둘 다 `reports/facet_rft_2026/` 아래):
- `x611_t7391_task12_gate_iso.py` — task 12 게이트 격리 재현
- `x611b_t7391_confirm_census.py` — 런 전수 write 앞 확인상태 센서스

한 줄 요약: `reward 0.0 = DB 0.0 × NL_ASSERTION 1.0`. gold 는 write **0건**인데 에이전트가
`return_delivered_order_items` 를 **확인 없이 2건 실행**해 `db_match=false`.
막았어야 할 `G2_CONFIRM_WRITE` 가 **인증 턴의 "Sure"** 에 열렸다
(`gate_interpreter.py:16-18` CONFIRM_RE · `:387-390` confirm 술어 ·
`t2_gate_patch.py:1278-1285` `_last_user_text`). 격리 재현: last_user=msg[3] → allow,
last_user=msg[1] → deny(G2). 런 전수 실행 write 22건 중 5건이 최초요청/인증 턴 토큰으로 열렸다.
