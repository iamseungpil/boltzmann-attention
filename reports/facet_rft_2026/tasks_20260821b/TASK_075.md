# TASK_075 — bank_x806_base_nt4 궤적 포렌식 (2026-09-08)

> 결론 먼저: **지정된 런의 task_075 궤적은 로컬에 존재하지 않는다.** 실패 궤적을 per-step 으로 추적할 재료가 0 이므로
> «실패 원인»은 **UNPROVEN** 이다. 대신 (a) 재료 부재를 검색 경로와 함께 박제하고, (b) 이 태스크가 x806 계열에서 실제로
> 어떻게 «실패로 집계»됐는지(런처 즉시-사망)를 로그 축자로 확정하며, (c) 로컬에 있는 유일한 task_075 완주 궤적(bank_p1, 1.0)
> 을 대조 기준으로 기록한다. 수리 실행·코드 수정은 하지 않았다(제안만).

## 0. 재료 부재 — 검색 경로 ([[77]] «없다»는 경로 없이 발화 금지)

| 지시된 파일 | 실재 | 검색 |
|---|---|---|
| `sim_results/bank_x806_base_nt4_B_20260821b.results.json.gz` | **없음** | `ls sim_results \| grep 20260821b` → `bank_t7333_smoke_*`, `bank_t7336_halfA/halfB_*`, `fb_*`, `trace_*` 뿐. halfA 20태스크·halfB 20태스크 모두 `task_075` 미포함(직접 로드해 `task_id` 집합 확인). |
| `sim_results/bank_x806_base_nt4_B_20260821b.log.gz` | **없음** | 동상 |
| 대조 기준선 `undefined.results.json.gz` | **없음** (이름 자체가 미정의) | — |
| x806 계열 task_075 | **없음** | `ls sim_results \| grep x806` → `bank_x806_base_nt4_task_{001..059}` (결손 있음) + `x818cloud_bank_x806_base_nt4_task_{060..100}` (074 는 drv 로그만, **075·077·078 등 부재**). `find ba-frft -iname "*task_075*"` → 아래 3개뿐. |
| 로컬에 있는 task_075 전부 | `bank_p1_task_075.results.json.gz`(+provenance) · `x818cloud_rep1_task_075_drv.log.gz`(181 B) | 후자는 결과 없음(§2). |
| 선행 보고서 중 task_075 절 | `TASKS_072_075_PREP_2026_08_13.md` 만 («074/075 결과 없음 — 첫 라이브 모양은 새 런에서») | `grep -l task_075 reports/facet_rft_2026/*.md` → 이 파일 + 무관 2건(BANKING_FLOOR_LEVER_FIT·X291 은 «075» 숫자 우연 일치). |

«이 태스크의 상태: 4» 는 결과 파일이 아니라 **레인 워커의 실패 카운트**로 보인다(§2 — 런처가 4회 즉시-사망하면 4 «실패»). 결과 json 이 없으므로 reward 0 이 4개 있는 것이 아니다.

## 1. 채점 축 (지시 1 단계 — 이용 가능한 유일한 task_075 sim 으로 확인)

`bank_p1_task_075` trial 0: `reward_basis=['DB']`, `db_check={'db_match': True, 'db_reward': 1.0}`, reward **1.0**.
gold 3행(`action_checks`): `log_verification(mv93f8a7b2, time_verified=2025-11-14 03:40:00 EST)` → `unlock:open_bank_account_4821` → `call:open_bank_account_4821{user_id, account_type=checking, account_class="Green Fee-Free Account"}`. DB 축이므로 «Green Fee-Free Account» 클래스가 열려야 통과 — PREP 노트의 예측(계산형 클래스 선택)과 일치.

## 2. x806 계열에서 task_075 가 «실패»로 집계된 실물 — 런처 즉시-사망 (our_layer)

`x818cloud_rep1_task_075_drv.log.gz` 전문(181 B):
```
./run_ours_task.sh: line 17: set: pipefail: invalid option name
./run_ours_task.sh: line 18: $'\r': command not found
./run_ours_task.sh: line 37: syntax error near unexpected token `$'in\r''
./run_ours_task.sh: line 37: `  case "$1" in'
```
- [[87]] 동일-출력 검사: rep1 lane 의 181 B drv 로그 **27개(task_001·002·006·008·010·014·017·025·031·036·044·048·049·051·056·057·058·062·063·064·067·068·070·075·081·095·100) md5 전부 동일** `975704cb…`. 도구(런처)가 죽은 것이지 27개 태스크가 실패한 것이 아니다.
- 레인 로그 `x818cloud_lane_rep1.log.gz`: 09-07 11:12 큐 35 시작 → 050·003·004·024·007·015·023·047 까지는 완주(결과 200 KB+ 영속) → 이후 로그 없이 끝. `x818cloud_lane_rep1_fb.log.gz`: 14:43 재시작 후 `-> task_050` 한 줄. 완주분 drv 로그도 이미 `model_profiles/Qwen__Qwen3.8-27B-FP8.env: line 7: $'\r': command not found` 를 찍고 있었다(env 는 주석줄만 CR 이라 살아남음). 13:44 이후 `run_ours_task.sh` 자체가 CRLF 본으로 갱신되어 파싱 단계에서 죽고, 워커가 큐를 드레인했다.
- 코드 경로: `scripts/distill/tau2/run_ours_task.sh:17` (`set -o pipefail` — CR 이 붙어 `pipefail\r`), `:37` (`case "$1" in\r`). 로컬 워킹카피 `CR 225 = LF 225`(전 줄 CRLF); `git show HEAD:…` 블롭은 `CR 0`; `git ls-files --eol` → `i/lf w/crlf`; `git config core.autocrlf` → `true`. 즉 **저장소는 정상이고 윈도우 체크아웃본을 그대로 전송한 것이 원인**. 호출자: `t2_lane_worker.sh:75` / `t2_lane_worker2.sh:90` (`bash ./run_ours_task.sh --arm viewmax2 …`) — 즉시-실패 가드 없음(메모리 30-remote-env 2026-09-07 절이 이미 같은 사고를 박제, «레인 워커에 120 초 즉시-실패 가드 → exit 3» 처방 미적용 상태에서 재발).
- 반증 조건: 리모트의 `grep -c $'\r' run_ours_task.sh` 가 0 이면서 같은 drv 로그가 나오면 이 귀속은 틀린 것.

## 3. 변이표 — 지정 런: 해당 없음 / 대조 기준(bank_p1 trial 0)

| run | trial | reward | gold | matched | missing | wrongarg | dup | blocked | extra |
|---|---|---|---|---|---|---|---|---|---|
| bank_x806_base_nt4 (지정) | — | **결과 없음** | — | — | — | — | — | — | — |
| bank_p1_task_075 (대조) | 0 | 1.0 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |

(`F.mutation_diff(sim, F.mutating_tools())` 정본 사용. `clean=True`.)

## 4. 대조 궤적(bank_p1 trial 0) step-by-step — 통과 경로의 결정 지점

엔진 5e269699(dirty)·bench fc0055d·`10.10.10.151:8141`·nt=1·2026-09-06 15:09→15:44(2101 s, 64 msgs, `user_stop`). x806 과 **engine sha 상이**(x806 = bench fc0055d 동일, 엔진 sha 는 x806 results 에 미기록) — 동일 조건 대조가 아니다.

1. msg 1 USER: «3-month photography trip … open a new personal checking account … cost me the least» → msg 2–28: `KB_search_bm25` 1회 + `shell` 17회로 personal checking 전 클래스(Blue·Green·Purple·Bluest·Light Blue·Green Fee-Free·Evergreen·Dark Green·Gold Years·General) ATM/foreign-fee 문서를 **전수 cat**. 후보군 회수 결손(098형) 없음.
2. msg 29 A-SAY: 사용량 2가지(횟수·금액)를 되묻는다. msg 30 USER: «3 months … 6 times per month (18 withdrawals total) … around $350».
3. **결정 지점 msg 31**: `get_checking_atm_fee_totals{months=3, withdrawals_per_month=6, withdrawal_amount=350}` → msg 32 TOOL «Documented ATM fee totals per personal checking account class for the stated usage …». 계산을 모델이 하지 않고 계산 도구(우리 층 `T2_COMPUTE` 계열)가 표로 준다 → msg 33 에서 Green Fee-Free 추천. F2b(계산 오류) 벽이 여기서 우회됐다.
4. msg 34 USER 승인+검증정보 → 35 `get_user_information_by_name` → 37 `verify_identity` → 39 `get_current_time` → **41 `log_verification`(gold 075_0 일치)**.
5. msg 43–49: 개설 전 자격확인(기존 계좌 수·closed-for-cause) — `unlock/call get_all_user_accounts_by_user_id_3847` → «No bank accounts found».
6. **msg 50 `unlock:open_bank_account_4821`(gold 075_1)** → msg 52 «Let me confirm the exact official account class name» 로 클래스명 문서 재확인 → 54–58 우회(transactions 툴 unlock·계좌 재조회 — 불필요·비변이) → **msg 60 `call:open_bank_account_4821{checking, "Green Fee-Free Account"}`(gold 075_2)** → 61 «Bank account opened successfully» → 63 USER `###STOP###`.

## 5. 레버 발화표 (지시 4 단계)

지정 런의 로그가 없어 **대조 불가**. rep1 drv 로그는 파싱 단계 사망이라 `T2_*` 레버 발화 0(레버 이전 단계). bank_p1 은 `.log.gz` 미회수(results+provenance 만)여서 `T2_SG_DOCS`·`T2_PIN_READ`·`T2_DEMANDED_STEP`·`T2_CLAIMPROV`·`T2_FOLLOWUP`·`T2_SEARCH_AGENT`·`FAB_STRIP`·`T2_ARG_PRODUCERS`·READ-FIRST·`T2_REQUIRE_DOC_DELIVER`·`T2_SEARCH_REARM` 의 발화/무시/오발화를 가릴 수 없다(provenance `levers_on` 에 전부 켜져 있었다는 선언만 있음). 궤적 문면에서 확인되는 것: 계산 도구 발화(msg 31–32, `T2_COMPUTE`/`get_checking_atm_fee_totals`) 1건.

## 6. 선행 판정과 대조

`TASKS_072_075_PREP_2026_08_13.md` §가족: 075 = «계산형 클래스 선택 … F2b compound(18회×$350) … 계산+후보군 커버리지». 대조 궤적(bank_p1)은 정확히 그 두 벽(전수 회수 + 계산 도구)을 통과해 1.0. **선행 예측과 모순 없음**. 실패 판정을 다룬 선행 절은 없다(«첫 라이브 모양은 새 런에서»).

## 7. 원인 확정

- 지정 런 task_075 «실패» 의 궤적 원인: **UNPROVEN — 재료 없음**. model / user_sim / env 어느 것도 주장할 근거 0.
- x806 계열에서 task_075 가 «실패 4» 로 보이는 원인: **our_layer(확정)** — `run_ours_task.sh:17,37` CRLF 파싱 사망 + `t2_lane_worker.sh:75` 즉시-실패 가드 부재 → 결과 파일 0 개, 워커가 실패로 계수. (rep1 27건 동일 md5 · 메모리 30-remote-env 2026-09-07 절과 같은 사고.)

## 8. 처방 후보 (실행 안 함 · 승인 필요 [[86]])

1. 리모트에서 `grep -c $'\r' run_ours_task.sh arms/*.env model_profiles/*.env` → `sed -i 's/\r$//'` 후 `bash -n`. 전송 경로를 `git pull`(리모트에서 LF 체크아웃)로 바꾸거나 `.gitattributes` 에 `*.sh text eol=lf` 추가(로컬 체크아웃도 LF 로).
2. `t2_lane_worker.sh` / `t2_lane_worker2.sh`: 태스크가 120 s 안에 종료하고 results.json 이 없으면 큐 맨 앞에 되돌리고 `exit 3`(메모리 처방 그대로).
3. 오케스트레이터의 태스크 «상태» 는 drv 종료코드가 아니라 **results.json 존재 + reward** 로 재산출. 현재 «4» 는 데이터가 아니다.
4. 이후 `bank_x806_base_nt4 task_075 nt=4` 를 실제로 돌려야 첫 실패 궤적이 생긴다. 그때 이 문서 §4 의 결정 지점(msg 31 계산 도구 / msg 60 클래스명)을 기준선으로 대조.
