# TASK_085 (bank_x806_base_nt4) — 포인터

정본 보고서: `reports/facet_rft_2026/tasks_20260905/TASK_085.md`

- 지시서 경로 `tasks_20260821b/TASK_085.md` 는 훅 `scaffold_guard.py:200`(정본 명명 = `/tasks_+\d{8}/TASK_<id>.md`, `20260821b` 의 `b` 불일치) 에 막혀 실물 런 날짜(2026-09-05) 디렉터리에 두었다(`x852/x853/x854/x855_*_pointer.md` 선례).
- 지시서의 결과 파일 `bank_x806_base_nt4_B_20260821b.results.json.gz` 는 로컬에 없고, **x806 계열 task_085 는 로컬 어디에도 없다**(x806 체인 로그 축자: chainB *"→ task_070 (잔여 38) / Terminated"* · chainD *"→ task_074 (잔여 31) / Terminated"* — 큐가 085 에 닿기 전에 죽었다). 게다가 x806 은 `x812_cloud_worker.sh:47` `--gate 0`(우리 층 0). 실물 = `bank_t7393_laneC_20260905_1533`(주 · 1 trial · 75 msg) + `bank_lev6a_20260905_0935`(대조 · 1 trial · 74 msg) — 같은 bench sha fc0055dc · Q3.8-27B · gate 1. 로그 둘 다 미회수.
- 한 줄 판정: 2/2 reward 0 · DB 축 · read 9/9 · 3번째 분쟁 정정-스킵·4번째 금액 정정 **정답** · 실패 = `customer_max_liability_amount` 3칸(laneC 500/89.99/14.99 · lev6a 50/500/500 ↔ gold 50 고정) + `provisional_credit_eligible` 1칸(085_7 false↔true) · primary=**model**(doc_031 "of statement" 를 "of the transaction" 으로 읽어 $500 티어 · 한 sim 안에서 규칙 3종 자기모순) · secondary=**env**(도구 명세 "and the disputed amount" 가 min 유도 ↔ gold 는 disputed 무관 · doc_032 REQUIRED/NOT-REQUIRED 이중절 · statement 날짜 DB 부재) · our_layer=**비인과**(`T2_RULE_AT_WRITE` 기본 OFF `go_stack.sh:673` 로 liability 정책 행 `gate.json:11086` 미전달 · `T2_DISTINCT_ARGS` 호출부 `t2_gate_patch.py:13620-13629` 가 처방 `_dv[4]` 폐기 — 둘 다 재료는 msg3 에 이미 있었으므로 인과 불성립) · user_sim 정상.
- 선행 정정: `gate.json:11123` `_note` 의 «liability == min(티어, 실손실) 19/19» 는 085_9(gold 50 ↔ disputed 14.99) 가 반증.
