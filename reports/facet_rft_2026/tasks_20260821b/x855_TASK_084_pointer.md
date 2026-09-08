# TASK_084 (bank_x806_base_nt4) — 포인터

정본 보고서: `reports/facet_rft_2026/tasks_20260830/TASK_084.md`

- 지시서 경로 `tasks_20260821b/TASK_084.md` 는 훅 `.claude/hooks/scaffold_guard.py:200`(정본 명명 = `/tasks_+\d{8}/TASK_<id>.md`, `20260821b` 의 `b` 불일치)에 막혀 실물 런 날짜(2026-08-30) 디렉터리에 두었다(`x853_TASK_082_pointer.md` 선례).
- 지시서의 결과 파일 `bank_x806_base_nt4_B_20260821b.results.json.gz` 는 로컬에 없고, **x806 계열 task_084 는 로컬 어디에도 없다**(검색 경로 5종은 정본 §0 · x818cloud 레인 큐에 084 가 오른 적 없음). x806 = `--gate 0` **base 팔(레버 0)** 이므로 같은 팔·같은 sha `fc0055d`·같은 모델·같은 user-sim 인 `bank_x644_q38base_bank78_20260830` task_084 t0 을 실물로 추적, ours 팔 6 sim(t7393 laneC·lev6b·re8141p11·smoke084·x713·x732)을 대조.
- 한 줄 판정: 1/1 reward 0 · DB 축 · read 9/9·unlock 7/7 완주(N97 08-04 의 71회 중복 read 소멸) · 실패 = dispute 인자 3행 · 결정 지점 msg41 · primary=model(같은-날짜 중복쌍 earliest 규칙이 msg5 문맥에 있는데 첫 줄 `d7f2` 선택 · 스스로 말한 $50 tier 를 0 으로 · Light Blue 한도 2 를 알고도 고객에게 고르게 하지 않고 EveryonePay 선행) · secondary=env(`customer_max_liability_amount` 규칙 KB 0건 + gold 행마다 규칙 상이 50↔412.88 — lev6b 가 이 한 필드로만 떨어짐) · user_sim 비인과 · our_layer 비인과(base 팔 · `x818_lanes/t2_base_worker.sh:31`). x737 §1f-5 #9 «KB 에 규칙 없음» 은 정정(doc_031 에 earliest 문장 실재).
