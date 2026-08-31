# TASK_22 (t7391_reg12) — 본문 위치 포인터

정본 본문 = `reports/facet_rft_2026/tasks__20260829/TASK_22.md`

이 디렉터리(`tasks_reg12/`)에 `TASK_22.md` 를 직접 두지 못한 이유:
`.claude/hooks/scaffold_guard.py:200-201` 의 런별-포렌식 예외 술어가 `r"/tasks_+\d{8}/"` 라
`tasks_reg12/TASK_22.md` 는 exit 2 로 막힌다. 훅을 우회하지 않고 정본 명명
`tasks__<날짜>/TASK_<id>.md` (형제 `x602_TASK_3_pointer.md` · `x603_TASK_4_pointer.md` ·
`x611_TASK_12_pointer.md`)를 따랐다.

딸린 격리 프로브: `reports/facet_rft_2026/x612_t7391_task22_ground_iso.py`
(모델 0·env 0 · A_repro 바이트 동일 재현 · B_prior 원본 address2 전수 · C_feedback 문면 확인)

한 줄 요약: `reward 0.0 = DB 0.0 × NL_ASSERTION 1.0`. 최종 DB 가 gold 와 다른 칸은 **하나** —
주문 `#W9911714` 의 `address.address2` 가 gold `""` 인데 우리는 `"Suite 865"`(손님의 옛 주소
부속칸)를 남겼다. 그 값을 대화에 처음 들여놓은 것은 **우리 층의 `T2_GROUND` 제자리 치환**이다
(`t2_gate_patch.py:8435-8445` · msg[10] 에서 모델의 날조 `"Apt 1"` → `"Suite 865"`).
치환값이 msg[11] 도구 결과로 에코되자 모델은 이후 새-주소 write 3/3 을 `"Suite 865"` 로 냈다.
우리 값이 문맥에 들어오기 **전**에는 같은 모델이 같은 태스크에서 **8/8 로 `""`**(gold)를 냈다
(대조군 `hist_gpt52_reg12_PASS` task 22 = reward 1.0 포함).
