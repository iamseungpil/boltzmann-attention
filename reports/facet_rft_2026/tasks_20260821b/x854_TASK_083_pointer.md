# TASK_083 (bank_x806_base_nt4) — 포인터

정본 보고서: `reports/facet_rft_2026/tasks_20260908/TASK_083.md`

- 지시서 경로 `tasks_20260821b/TASK_083.md` 는 훅 `scaffold_guard.py:200`(정본 명명 = `/tasks_+\d{8}/TASK_<id>.md`, `20260821b` 의 `b` 불일치)에 막혀 x806 동기화 날짜(2026-09-08) 디렉터리에 두었다(`x852_TASK_060_pointer.md`·`x853_TASK_082_pointer.md` 선례).
- 지시서의 결과 파일 `bank_x806_base_nt4_B_20260821b.results.json.gz`·로그·`undefined.results.json.gz` 는 로컬에 없고, **x806 계열 task_083 은 결과·drv 로그·레인 로그 큐 흔적 어디에도 없다**(검색 경로 6종은 정본 §0). 지정 런 원인 = **UNPROVEN**.
- 대리 추적 = 동일 구성 base `bank_x644_q38base_bank78_20260830`(Q3.8 · gate 0 · alltools · gpt-5.2 user-sim) task_083 trial 0 + 로컬 task_083 18 sim 전수 변이표.
- 한 줄 판정: `reward_basis=['ACTION']` · gold dispute 4행이 env 필수 17키 중 `customer_max_liability_amount` 를 뺀 **16키** ⇒ 스키마대로의 정상 호출(x644 msg[51] 4/4 17키·env 접수)은 영구 불일치 = **env(벤치 gold 결함)**, 선행 §2-B2·1f-12 와 동일. 부차 model(x644 [43] «full $475, or just the $425 overage?» 자작 선택지 → 425 · duplicate 의 provisional_credit false). 선행 정정 1건: «후행 0 문자열 바이트 재현 불가» 는 틀렸다 — #1·#2 는 17번째 키만 빼면 바이트 동일. our_layer 0(gate 0 라 코드 경로 부재).
