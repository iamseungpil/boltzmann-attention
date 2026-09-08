# TASK_087 (bank_x806_base_nt4) — 포인터

정본 보고서: `reports/facet_rft_2026/tasks_20260830/TASK_087.md`

- 지시서 경로 `tasks_20260821b/TASK_087.md` 는 훅 `C:/workspace/.claude/hooks/scaffold_guard.py:200`(정본 명명 = `/tasks_+\d{8}/TASK_<id>.md` — `20260821b` 의 `b` 불일치 · `_is_run_task_report=False`)에 막혀 실물 대리 런 날짜(2026-08-30) 디렉터리에 두었다(`x856_TASK_086_pointer.md`·`x855_TASK_084_pointer.md` 선례).
- 지시서의 결과 파일 `bank_x806_base_nt4_B_20260821b.results.json.gz` 는 로컬에 없고, **x806 계열 task_087 은 로컬 어디에도 없다**(전 x806 gz `zcat|grep task_087` 0 · x818cloud 레인 로그 5개 0 · 오염 manifest 0 · `find -iname "*task_087*"` 0). x806 = `--gate 0` base 팔이므로 같은 팔·같은 sha `fc0055d`·같은 모델·같은 user-sim 인 `bank_x644_q38base_bank78_20260830` task_087 t0(73 msg·user_stop) 을 주 대리로 추적, ours 팔 2 sim(`relane2b151`·`t7393_laneC`)을 대조.
- 한 줄 판정: 1/1 reward 0 · DB 축 · gold 변이 7 중 6 일치(log·unfreeze×2·velocity clear·close·order) · 실패 = dispute 인자 2칸(`discovery_date` 11/14↔11/06 · `customer_max_liability_amount` 500↔50) · 결정 지점 msg 38→49→61 · primary=**model**(«처음 알아챈 날»을 묻지 않고 오늘 날짜 대입 — 배달된 doc_031 «Date customer first noticed the issue» 무시) · secondary=**user_sim**(§11 «November 6th text» 조건절이 x644 에선 봉인, relane2b151 에선 개방) · our_layer 비인과(base 팔 · 로그 `T2_LEVER 0 · T2_GATE 0`). x737 §1f-12 «087 = model(이관 과일반화)» 은 relane2b151 에 한해 유지 · base 팔은 이관 없이 완주하고 인자에서 졌다.
