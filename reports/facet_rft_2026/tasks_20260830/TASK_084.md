# TASK_084 (bank_x806_base_nt4 지시 · 실물은 x644 q38base 2026-08-30) — per-step 포렌식

> 작성 2026-09-08 · 전부 로컬 · 코드 수정 0 · 커밋 0. 지시서 경로 `tasks_20260821b/TASK_084.md` 는 훅 `.claude/hooks/scaffold_guard.py:200`(정본 명명 `/tasks_+\d{8}/TASK_<id>.md` — `20260821b` 의 `b` 불일치)에 막혀 **실물 런 날짜(2026-08-30) 디렉터리**에 둔다(`x853_TASK_082_pointer.md` 선례). 포인터 = `tasks_20260821b/x855_TASK_084_pointer.md`.

> 결론 먼저. **지시된 런(`bank_x806_base_nt4`)의 task_084 궤적은 로컬 어디에도 없다**(§0 검색 경로). x806 은 `--gate 0` **base 팔(레버 0개)** 이므로, 같은 팔·같은 bench sha `fc0055d`·같은 모델 `Qwen/Qwen3.8-27B-FP8`·같은 user-sim `gpt-5.2/low` 인 **`bank_x644_q38base_bank78_20260830` task_084 trial 0 을 실물로 추적**했고, ours 팔 6 sim(t7393 laneC·lev6b·re8141p11·smoke084·x713·x732)을 대조로 붙였다. 판정: **DB 축 · reward 0 · 읽기 사슬 9/9 완주 · 실패 = `file_debit_card_transaction_dispute_6281` 인자 3행**. 원인 **primary = model**(같은-날짜 중복쌍에서 «earliest» 규칙이 문맥에 있었는데 첫 줄(최신)을 골랐다 · Light Blue 한도 2 를 알고도 고객에게 고르게 하지 않고 EveryonePay 를 먼저 넣었다 · 스스로 말한 $50/$500 tier 를 호출에서는 0 으로 바꿨다), **secondary = env**(`customer_max_liability_amount` 는 KB 8편 어디에도 규칙이 없고 gold 값 자체가 행마다 다른 규칙 — 47.5 → **50**, 412.88 → **412.88** — 이라 닫힌 규칙으로 도달 불가), **our_layer = 비인과**(base 팔에는 레버가 없다 · `x818_lanes/t2_base_worker.sh:31` `--gate 0`). 선행 x737 §1f-0 «model» 판정과 같은 방향이되, x737 §1f-5 #9 의 *«KB 에 규칙 없음»* 은 **틀렸다** — doc_031 에 *"Dispute the earliest (first) transaction when multiple duplicates exist."* 가 있고 x644 msg5 문맥에 실재했다.

## 0. 재료 상태 — 지시된 실물은 없다 ([[77]] 검색 경로)

| 지시된 파일 | 실재 | 검색 경로 |
|---|---|---|
| `sim_results/bank_x806_base_nt4_B_20260821b.results.json.gz` · `.log.gz` | **없음** | `ls sim_results \| grep -iE "20260821b\|task_084\|x806.*B_"` → 20260821b 는 `bank_t7333_smoke_*`·`bank_t7336_halfA/halfB_*`·`fb_*`·`trace_*` 뿐. halfA/halfB 를 직접 로드해 `task_id` 집합 확인 → task_084 **미포함**(halfB = 016…098 20개) |
| 대조 `undefined.results.json.gz` | **없음**(이름 자체가 미충전 자리표시자) | — |
| x806 계열 task_084 | **없음** | `ls sim_results \| grep x806` → `bank_x806_base_nt4_task_{001..059}`(결번 있음) + `x818cloud_bank_x806_base_nt4_task_{053..100}`(060~074·076·079·089·098·100). `find ba-frft -iname "*task_084*"` → 0. `x818cloud_laneA/laneB.log.gz` 의 큐 진행(`→ task_*`) = A: 053·061·063·066·068·069 / B: 060·062·065·067·071·074 → **084 는 큐에 오른 적 없음**. `x818_lanes/dash.sh:44` 가 가리키는 run 디렉터리는 리모트(SSH 금지) |
| 로컬 task_084 전수 | **19 sim / 12 런 · 전부 reward 0.0** | 최근 `*.results.json.gz` 전수 로드 → `bank_all97_nt1_v2_20260718`·`bankxfer_floor_*`·`bank_n97_gpu1_main_2026080{4,5,6}`·`bank_smoke6_n97rx`·`bank_smoke084_2215`(09-01)·`bank_x644_q38base_bank78_20260830`·`bank_x713_nightB_20260901`·`bank_x732_qB_t3prime_20260901`·`bank_re8141p11_20260904`·`bank_lev6b_20260905_0935`·`bank_t7393_laneC_20260905_1533` |
| 선행 per-task 보고서 | `tasks_*/` 에 084 **없음** | `grep -ln task_084 reports/facet_rft_2026/*.md tasks_*/*.md` → `N97_TASKWISE_FORENSIC_2026_08_04`·`N97_LEVER_PRESCRIPTIONS_DESIGN_2026_08_04`·`BANKING_FLOOR_LEVER_FIT_2026_07_11`·`RESEARCH_MASTER`(C466·C523·C600)·`x737_next_run_plan_2026_09_04`(§1f-0·§1f-5 #8/#9)·`x738_q38_base97_census`(fail 55 목록) |

**x806 = base 팔 확정**: `x818_lanes/t2_base_worker.sh:27-33` / `x812_cloud/lane.sh:27-33` = `t2_run_gated.py --gate 0 … --num_trials 4 --max_concurrency 4` (`t2_run_gated.py:87` `--gate` default 0 · `:4` «--gate 1 이면 t2_gate_patch 적용»). `x818cloud_bank_x806_base_nt4_task_098_drv.log.gz` 의 `T2_` 줄 = **0**. x644 로그 머리 축자: *"레버 env : 0 개 (0 이어야 base)"* · *"clone HEAD : fc0055d"*. ⇒ **실물 대체 = `bank_x644_q38base_bank78_20260830` task_084 t0**(49 msg · `user_stop`). ours 팔 6 sim 은 «레버가 개입했으면 달랐나» 의 대조로만 쓴다(engine sha 상이 · n=1 · 태스크 단위 인과 서술 금지).

«이 태스크의 상태: 4» 는 로컬로 검증 불가(nt=4 이면 4/4 실패로 읽히나 결과 파일이 없다).

## 1. 채점 축

`sim['reward_info']`(x644·t7393 동일): `reward_basis=['DB']` · `db_check={'db_match': False, 'db_reward': 0.0}` · `nl_assertions=None` · `communicate_checks=None` · `env_assertions=[]`. **DB-해시 축**. `action_checks` 18행은 진단용: `log_verification`(084_0) · read unlock/call 9행(084_1~5) · `unlock file_debit…`(084_6) · **dispute call 3행(084_7·8·9 = 전부 match False)** · `unlock/call close_debit_card_4721`(084_10·11).

gold 3 dispute(전부 `file_debit_card_transaction_dispute_6281`):

| aid | txn | category | discovery | amount | type | contacted | prov | liability | card_action |
|---|---|---|---|---|---|---|---|---|---|
| 084_7 | `btxn_e8c3b09d2c45` (Bella's, 11/06) | duplicate_charge | 11/10 | 47.5 | pin_purchase | false | **true** | **50** | keep_active |
| 084_8 | `btxn_f9d4c10e3d56` (Whole Foods, 11/08) | incorrect_amount | 11/09 | **100.0** | signature_purchase | true | false | **50** | keep_active |
| 084_9 | `btxn_c2g7f43h6g89` (TechWorld, 10/05) | card_present_fraud | 11/14 | 412.88 | pin_purchase | false | true | **412.88** | close_and_reissue |

gold 에 EveryonePay(`btxn_b1f6e32g5f78`, $275) dispute 는 **없다** — 시나리오 §7 «Entry Tier 한도 2 · 기존 1건 OPEN → 하나만 · 고르게 하면 TechWorld».

## 2. 변이표 (`t2_forensic.mutation_diff(sim, F.mutating_tools())` · 손 비교 0 · WRONGARG 는 필드별 대조)

| run (팔·엔진) | msg | matched | missing | wrongarg | extra/dup/blocked |
|---|---|---|---|---|---|
| **x644_q38base 08-30 (base·실물 대체)** | 49 | 2 (`log_verification` m18 · `close_debit_card{6129,fraud_suspected}` m45) | 3 (084_7·8·9) | 3 (m41 병렬) | 0/0/0 |
| t7393_laneC 09-05 (ours·viewmax2+5) | 60 | 3 (+084_9 TechWorld **일치**) | 2 (084_7·8) | 2 | 0/0/0 |
| lev6b 09-05 (ours·a5417ce4) | 63 | **4** (084_7·084_8 **일치** + close) | 1 (084_9) | 1 | 0/0/0 |
| re8141p11 09-04 (ours·0b7d703b) | 68 | 2 | 3 | 3 | 0/0/0 |
| smoke084 09-01 (ours·7cc29877) | 53 | 4 (084_8·084_9 일치) | 1 (084_7) | 1 | 0/0/0 |
| x713_nightB 09-01 (ours) | 101 | 2 (084_9) | 3 (084_7·8·**11 close 미실행**) | 2 | 0/0/0 |
| x732_qB_t3prime 09-01 (ours) | 112 | 2 | 3 | 4 | 0/0/**1**(env `Invalid arguments`) |

### 2a. 필드별 대조 — x644 (보낸 값 ↔ gold · 일치 필드 생략)

| aid | 필드 | 보낸 값 | gold | 그 값이 문맥에 있었나 |
|---|---|---|---|---|
| 084_7 Bella's | `transaction_id` | `btxn_d7f2a98c1b34` | `btxn_e8c3b09d2c45` | msg35 거래 목록: *"Transactions are returned in reverse chronological order (most recent first)"*(msg26 도구 설명) 아래 2번 `d7f2a98c1b34`, 3번 `e8c3b09d2c45`(둘 다 11/06·-47.5) · doc_031(msg5 전문 도달) *"Dispute the earliest (first) transaction when multiple duplicates exist."* ⇒ earliest = **뒤에 나열된 3번** |
| | `provisional_credit_eligible` | `false` | `true` | doc_032(msg5 도달) *"Provisional Credit is REQUIRED when ALL … 2. Dispute Category: … 'duplicate_charge'"* · 서면진술 yes(msg15 §8) · 60일 내 |
| | `customer_max_liability_amount` | `0` | `50` | 모델 자신 msg14 *"Within 2 business days: up to $50"* · user msg15 *"noticed it on November 10th"*(11/06 목 → 11/10 월 = 2 영업일) |
| 084_8 Whole Foods | `customer_max_liability_amount` | `0` | `50` | 동일 · user msg15 *"noticed it on November 9th, the day after"* |
| | (`disputed_amount` 100.00 · `discovery_date` 11/09 · `contacted_merchant` true · `prov` false) | — | 일치 | (x644 는 여기 맞았다 · t7393/re8141p11 은 189.23 을 넣어 틀림) |
| 084_9 TechWorld | **호출 자체 없음** — 대신 EveryonePay `btxn_b1f6e32g5f78{card_not_present_fraud, 11/11, 275, person_to_person, prov=true, liab=500, close_and_reissue}` 를 넣었다 | — | gold 084_9 는 TechWorld | msg37 `get_debit_dispute_status` → *"Found 1 record(s) … status: OPEN"*(Light Blue) · doc_031 *"Entry Tier max 2"* · msg47 모델 자신 *"your Light Blue account has reached its maximum of two open disputes after filing the EveryonePay dispute"* — **한도를 알고 넣은 뒤에 통보** |

### 2b. 필드별 대조 — ours 대조 sim 의 dispute 행 (같은 축·같은 gold)

| run | 084_7 txn | 084_7 liab / prov | 084_8 amount / liab | 084_9 (TechWorld) liab / 그 외 |
|---|---|---|---|---|
| t7393_laneC | d7f2 (✗) | 0 / true | **189.23** / 0 | **412.88 ✓ 전 필드 일치** |
| lev6b | **e8c3 ✓** | **50 ✓ / true ✓ → 084_7 일치** | **100 ✓ / 50 ✓ → 084_8 일치** | **500** (✗ · 나머지 16/17 일치) |
| re8141p11 | d7f2 (✗) | **47.5** / true | 189.23 / **189.23** | 500 |
| smoke084 | d7f2 (✗) | 47.5 / true | 100 ✓ / 50 ✓ → 일치 | 412.88 ✓ → 일치 |
| x713 | d7f2 (✗) | 0 / true | 100 / 0 | 412.88 ✓ → 일치 (close 미실행) |
| x732 | e8c3 ✓ | 47.5 / **false** | 100 / 100 | m81 EveryonePay 먼저 → m108 TechWorld `card_not_present_fraud`·`keep_active` |

⇒ 7 sim 어느 것도 3행을 동시에 못 맞췄고, 틀린 필드는 **네 종류로 수렴**: ① 중복쌍 txn 선택(5/7 이 첫 줄 `d7f2`), ② `customer_max_liability_amount`(7/7 최소 1행 불일치 · 값은 0·47.5·189.23·500 제각각), ③ incorrect_amount 의 `disputed_amount`(2/7 이 189.23), ④ Light Blue 어느 건을 먼저(2/7 EveryonePay 선행).

## 3. x644 궤적 step-by-step — 결정 지점 (축자)

- **msg1 USER** *"Hi, I need to file several disputes on my debit cards."* → **msg2 A** *"Could you tell me a bit more about each transaction…"* → **msg3 USER** 4건을 한 번에 전부 진술(시나리오 «Respond reactively» 와 달리 선제 · 결손은 없음 · 기존 dispute 는 말하지 않음 = 시나리오 준수).
- **msg4-11 KB**: bm25/dense → **doc_031 전문(23,559자) msg5 에 도달**. 그 안에 liability 3단·*"Dispute the earliest (first) transaction when multiple duplicates exist."*·*"Entry Tier max 2 … per account"*·16 인자 규칙(단, `customer_max_liability_amount` 항목은 **없다** — §6 env)·card_action 매핑. doc_032(provisional 요건) 도 같은 턴에 도달.
- **msg12-13** `get_current_time` → *"2025-11-14 03:40:00 EST"*.
- **msg14 A (규칙 발화 = 결정 재료 ①)**: *"your maximum liability depends on when you first noticed them: — Within 2 business days: up to $50 — Within 60 days: up to $500 — After 60 days: you may not recover the funds"* + 8개 질문(발견일·PIN/서명·merchant 접촉·소지·PIN 유출·서면진술).
- **msg15 USER (결정 재료 ② — 필요한 값 전부 실재)**: *"Bella's Bistro duplicate ($47.50): I noticed it on November 10th"* · *"Whole Foods … I noticed it on November 9th"* · *"TechWorld … today, November 14th"* · *"Bella's Bistro: PIN transaction. I haven't contacted them yet"* · *"Whole Foods: Signature … I called Whole Foods on November 9th"* · *"in-person purchase … with a PIN"* · *"Evergreen: No … Light Blue: I'm unsure"* · *"you can use this conversation as my written statement."*
- **msg16-19** `get_user_information_by_name` → `log_verification`(gold 084_0 **matched**, `verify_identity` 없이 — base 팔이라 A2 variant 없음).
- **msg20-37 읽기 사슬**: unlock/call accounts(m20-23) → unlock 3종 + KB 4건 병렬(m24) → cards ×2 · transactions ×2 · dispute_status 병렬(m32). gold 084_1~5 **9/9 matched**. **msg35** Evergreen 목록: 1 `f9d4…`(11/08) · 2 `d7f2a98c1b34`(11/06 BELLAS -47.5) · 3 `e8c3b09d2c45`(11/06 BELLAS -47.5) · 4 direct deposit. **msg37** *"Found 1 record(s) in 'debit_card_disputes' … account_id: chk_e5a31c7d82 … status: OPEN"*.
- **msg38-40** unlock `file_debit…` + `close_debit_card`. 도구 설명(m39) 축자: *"customer_max_liability_amount: number (required) - The maximum dollar amount the customer could be liable for based on Regulation E reporting timing rules and the disputed amount. Use -1 for unlimited liability."*
- **★ msg41 A (결정 지점 — 고객에게 되묻지 않고 3건 병렬 write)**:
  1. Bella's: `transaction_id=btxn_d7f2a98c1b34`(목록 첫 Bella's = **최신**) · `provisional_credit_eligible=false` · `customer_max_liability_amount=0`.
  2. Whole Foods: `disputed_amount=100.00`·`discovery 11/09`·`contacted true`·`prov false`(여기까지 gold 일치) · `customer_max_liability_amount=0`.
  3. **EveryonePay** `btxn_b1f6e32g5f78 … card_not_present_fraud … 500 … close_and_reissue` — TechWorld 가 아니라 P2P 를 넣었다. 직전 어느 턴에도 «둘 중 무엇을 먼저» 라는 질문이 **없다**(msg14 의 8개 질문에 없음 · grep 0). 시나리오 §7 *"IF ASKED TO CHOOSE: The older TechWorld charge is for more money — let's prioritize that one"* 은 발화 기회를 얻지 못했다.
- **msg42-44 TOOL**: 셋 다 성공 · P2P 에 *"Provisional Credit: ISSUED - $275.00"*.
- **msg45-46** `close_debit_card_4721{dbc_82b5d1a6f3, fraud_suspected}` → gold 084_11 **matched**(순서는 gold 와 같다: dispute 뒤 close).
- **msg47 A (사후 통보)**: *"I couldn't file the $412.88 October 5 TechWorld charge yet. Although it remains within the 60-day filing window, your Light Blue account has reached its maximum of two open disputes after filing the EveryonePay dispute."* + 48시간 대기·주문 보류(gold 도 order 없음 = 일치).
- **msg48 USER**: *"Oh, I do have that other dispute from last week still open. I didn't realize there was a limit. So I can only file one more dispute on that account right now? In that case, the older TechWorld charge is for more money — let's prioritize TechWorld"* — 시나리오 §7 문장이 **이제서야** 나오지만 슬롯은 이미 소진 · *"my liability could be up to $500"* · `###STOP###`(49 msg · `user_stop`).

**정확한 결정 지점 = msg41 하나.** 그 턴에 (a) 중복쌍 첫 줄 선택 (b) prov false (c) liability 0/0 (d) P2P 선행이 동시에 확정됐고, 필요한 값(earliest 규칙·$50 tier·발견일·OPEN dispute 1건·max 2)은 **전부 msg5/14/15/35/37 에 실재**했다.

## 4. 레버 발화표

**x806/x644 (base 팔)**: 레버가 **존재하지 않는다** — `t2_base_worker.sh:31` `--gate 0` · x644 로그 *"레버 env : 0 개"* · `T2_` 줄 0. 따라서 `T2_SG_DOCS`·`T2_PIN_READ`·`T2_DEMANDED_STEP`·`T2_CLAIMPROV`·`T2_FOLLOWUP`·`T2_SEARCH_AGENT`·`FAB_STRIP`·`T2_ARG_PRODUCERS`·READ-FIRST·`T2_REQUIRE_DOC_DELIVER`·`T2_SEARCH_REARM` 전부 **미발화(부재)** 이고 «발화했는데 무시» 는 성립 불가. 직전 런 이후 들어간 수리·레버는 이 팔에 **개입한 적이 없다**.

**ours 팔 대조 (같은 gold · 로그/trace 있는 것만)**:

| 레버 | re8141p11 (trace) | smoke084 (log) | 판정 |
|---|---|---|---|
| `T2_DISTINCT_ARGS` (`t2_gate_patch.py:13616-13629`, `log`/`deny`) | turn55 *"file_debit_card_transaction_dispute_6281.customer_max_liability_amount == disputed_amount (47.5) — 선언상 달라야 한다"* · turn57 *"… (189.23) …"* | 2회(동형) | **발화했는데 무시** — 값은 그대로 47.5/189.23 으로 커밋(모드 `log`). 이 술어(liab == amount)는 084_9 gold(412.88 == amount) 와 **모순** — gold 가 행마다 다른 규칙이라 닫힌 술어가 못 잡는다(§6 env) |
| `T2_CLAIMPROV` | 66줄 · 전부 *"kind-index rescued … 원장에 있다"* / *"reserve window: no action-kind claim — skip"* | 48줄 동형 | 발화·무해(인자값 관할 아님) |
| `T2_EPLAN_LISTED_IS_READ` | 14줄 *"배달된 9건을 검토됨으로 보정(… btxn_d7f2a98c1b34, … btxn_e8c3b09…)"* | — | 두 중복 txn 모두 «읽음» 으로 보정 — 어느 쪽이 earliest 인지는 관할 밖 |
| `T2_SEARCH_EXHAUST` | turn52 *"nudge stubs=2 · retrieved-but-unused 21"* | — | 발화·비인과 |
| `T2_ARG_POLICY_AT_WRITE` (`t2_gate_patch.py:12736-12752`) | 미탑재(provenance levers_on 에 없음) | 미탑재 | **lev6b 에만 탑재** — lev6b 가 7 sim 중 유일하게 50/50 + `e8c3` 를 맞췄다(n=1 · 인과 미판정 · 로그 미회수) |
| `T2_PIN_READ`·`T2_SG_DOCS`·READ-FIRST·`T2_SEARCH_AGENT/REARM`·`T2_REQUIRE_DOC_DELIVER` | trace 에 해당 mark 0 | 0 | 미발화 — 읽기 사슬 9/9·doc_031/032 도달이라 필요 없었다 |
| `T2_DEMANDED_STEP`·`T2_FOLLOWUP`·`T2_ARG_PRODUCERS`·`FAB_STRIP` | mark 0 | 0 | 미발화(UNPROVEN — «고객에게 고르게 하라» 는 어느 레버의 술어도 아니다) |

re8141p11 msg51 자기보고 *"The first filing hit a reference error, which blocked the other two"* 뒤 거래 재조회 2회 → 같은 `d7f2` 로 재발행. 반려 교환 자체는 transcript 에 없음(`t2_gate_patch.py:7929` strip) — 어느 게이트였는지 UNPROVEN, 값 변화 0 이라 비인과.

## 5. 선행 판정 대조

| 선행 | 판정 | 이번 |
|---|---|---|
| `N97_TASKWISE_FORENSIC_2026_08_04` (08-04 n97) | t1 *"동일 호출 71회 → max_steps"* · `[DUPLICATE-READ]` 68회 무시 | **소멸** — 최근 7 sim 전부 dup 0 · `user_stop` · 읽기 9/9. 실패 층이 REACH→**ARG** 로 내려갔다 |
| `RESEARCH_MASTER` C466 | «task_084 unlock 주장 반증» | 일치 — unlock 7/7 matched, unlock 층 결손 없음 |
| `RESEARCH_MASTER` C600 ③ | «cap fail-open … k회 개방(039/041/084)» | ours 팔 재현 안 됨(blocked 0) · base 팔 해당 없음 |
| `x737_next_run_plan` §1f-0 (09-04 · re8141p11) | **model** · *"msg44 에서 gold 값을 정확히 유도했다가 msg46 에서 스스로 뒤집었다"* · 우리층무관 강등 | **같은 원인**. re8141p11 msg44 *"max liability $50 / $500"* → msg46 *"The $500 is a maximum liability ceiling … the ceiling doesn't reduce your exposure"* → m56/58 액수 그대로. x644 도 msg14 → msg41 동형(0). |
| `x737` §1f-5 **#8** «084 customer_max_liability 412.88 의 KB 유도 경로 (KB 결손 vs gold 임의값)» | 미답 | **답**: 캡처한 KB 8편(009·025·028·031·032·035·036·037) grep `customer_max_liability` = **0건**. 규칙은 도구 스키마 한 줄뿐. gold 는 084_7 에서 cap(50 > 47.5 인데 50), 084_9 에서 amount(412.88 < 500 인데 412.88) — **둘을 동시에 내는 닫힌 규칙 없음** ⇒ *KB 결손 ∧ gold 임의값* 둘 다 참 |
| `x737` §1f-5 **#9** «동일날짜 중복쌍 tie-break — KB 에 규칙 없음» | — | **정정**: doc_031 에 *"Dispute the earliest (first) transaction …"* 실재(x644 msg5·t7393 msg4·lev6b msg4 전부 도달). 동일 날짜의 «earliest» 는 도구 설명 *"reverse chronological order (most recent first)"* 로 결정 가능(= 뒤에 나열된 `e8c3`). lev6b·x732 는 실제로 그렇게 골랐다 |
| `x738` census | 084 = fail 55 에 포함(base 97) | 이 실물이 그 sim |

## 6. 원인 확정 (4주체 · [[77]] 4칸)

**primary = model (x644 t0 · DB 축 · n=1 · 3행 중 3행)**
- 주장: msg41 의 write 인자 3종이 문맥에 실재한 값과 모순되게 결정됐다. (a) `transaction_id` — doc_031 earliest 규칙(msg5) + reverse-chronological 설명(msg26) 이 있는데 첫 줄 `d7f2` 선택. (b) `customer_max_liability_amount=0` ×2 — msg14 에서 스스로 *"Within 2 business days: up to $50"* 라 말하고 msg15 에서 11/10·11/09 를 받았다. (c) Light Blue 한도 — msg37 OPEN 1건 + doc_031 «max 2» 를 알고(msg47 자백) 고객에게 묻지 않고 EveryonePay 를 넣었다; 시나리오는 «IF ASKED TO CHOOSE → TechWorld». (d) `provisional_credit_eligible=false`(duplicate) — doc_032 REQUIRED 목록에 duplicate_charge 가 있다.
- 근거: §3 축자(msg5·14·15·26·35·37·41·47).
- 반증 조건: doc_031 의 earliest 문장 또는 doc_032 의 duplicate_charge 행이 msg5 배달본에 **없었다면** (a)(d) 철회 — 확인: `t084_bank_x644…_t0.txt:37` 에 실재, doc_032 REQUIRED 목록 실재. (c) 는 msg41 이전에 «어느 것 먼저» 질문이 있었다면 철회 — grep 결과 0.
- 선행 확인: x737 §1f-0 (model) · N97(소멸) · C466.

**secondary = env (gold ↔ KB · 완벽한 에이전트도 DB 해시 미도달 가능)**
- 주장: `customer_max_liability_amount` 의 gold(50·50·412.88)는 KB 어느 문서의 규칙에서도 유도되지 않고 행 간에 규칙이 다르다. ours 7 sim 중 **가장 근접한 lev6b(16/17·16/17·16/17)가 이 한 필드(500 vs 412.88)로만 떨어졌다.**
- 근거: §5 #8 · 도구 스키마 축자 *"based on Regulation E reporting timing rules and the disputed amount"*(min 인지 cap 인지 미정의) · gold 084_7 = 50(amount 47.5) vs 084_9 = 412.88(amount 412.88).
- 반증 조건: required_documents 12편 중 캡처 안 된 4편에 «liability = min(cap, amount) except …» 류 규칙이 있으면 철회 — 로컬 KB 원문 없음(`a2/banking_knowledge.policy_facts.json` grep `liabil` 0건 · `find /c/workspace -iname "*bank_accounts*general*031*"` 0) → 리모트 회수 후 확인.
- 선행 확인: x737 #8 미답 → 이 절이 답.

**tertiary = user_sim: 비인과.** x644 user-sim 은 시나리오를 지켰다(기존 dispute 미발설 · 발견일 4건 정확 · «$500» 은 §9 시나리오 문장). msg3 의 선제 진술은 결손을 만들지 않았다.

**our_layer = 비인과 (확정).** base 팔 `--gate 0` = 레버·게이트·재생성 0 (`x818_lanes/t2_base_worker.sh:31` · `x812_cloud/lane.sh:31` · `t2_run_gated.py:87`). ours 대조에서도 값을 바꾼 레버 흔적 0 (`T2_DISTINCT_ARGS` 는 `log` 모드 발화·무시).

**분기점(ours 대조)**: lev6b 는 msg44 에서 *"file as a duplicate charge … noting you called them 11/09 … I will close card 6129"* 계획을 말하고 **PIN/서명을 되물은 뒤**(msg44-45) msg53 에 `e8c3`·50·50 을 넣었다 — x644/t7393 와 갈린 자리는 «계획 발화 → 확인 → write» 의 한 턴이지 읽기가 아니다. 남은 500 vs 412.88 은 §6 env.

## 7. 처방 후보 (실행 안 함 · 승인 필요 [[86]] · 제안만)

1. **gold/KB 정합성 표지**: 084 를 `customer_max_liability_amount` 행에 대해 «doc-unreachable» 로 표기하고 레버 A/B 분모에서 제외하거나, 채점을 그 필드 무시로 재산출해 «나머지 16 필드 도달률» 을 별도 계기로 둔다(082 §7-1 과 같은 표지). 리모트 KB 원문(required_documents 12편) 회수 후 #8 반증 조건 먼저 닫는다.
2. **닫힌 술어로 표현 가능한 것만 레버화(C600 판별식)**: (i) 같은 (date, description, amount) 중복쌍이 read 에 있고 write `transaction_id` 가 «reverse-chronological 목록의 뒤쪽» 이 아니면 deny — 닫힌 술어(정렬 규칙은 도구 설명이 선언). (ii) 계정당 OPEN dispute 수 + tier cap 을 read 에서 계산해 «남은 슬롯 1 · 후보 2 이상» 이면 write 전에 고객 선택 발화를 요구 — `T2_DEMANDED_STEP` 류 push 가 아니라 write-gate 로. (iii) `T2_DISTINCT_ARGS` 의 liab==amount 술어는 084_9 gold 와 모순이므로 **deny 승격 금지**.
3. base 팔 x806 task_084 실물(4 trial)이 내려오면 §3 msg41 결정 지점 4종(txn 선택·prov·liab·P2P 선행)을 trial 별로 대조 — 4/4 가 같은 필드에서 떨어지면 §6 primary 확정, liab 만 남으면 env 로 이관.

## 8. 산출·경로

- 실물 덤프: `scratchpad/t084_bank_x644_q38base_bank78_20260830_t0.txt`(전문) · `t084c_*.txt`(축약 4 런) · `t084_docs_raw.txt`(KB 8편 캡처).
- 변이 계산: `t2_forensic.mutation_diff` 정본, 44 mutating tools.
