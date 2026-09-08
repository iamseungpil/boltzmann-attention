# TASK_083 — bank_x806_base_nt4 궤적 포렌식 (2026-09-08)

> 결론 먼저: **지정된 런(`bank_x806_base_nt4`)의 task_083 궤적은 로컬에 존재하지 않는다** — 결과 gz 도, drv 로그도, 레인 로그의 큐 흔적도 0. 따라서 «이 런의 실패 원인»은 **UNPROVEN** 이다.
> 대신 (a) 재료 부재를 검색 경로와 함께 박제하고, (b) 로컬에 있는 **동일 구성 base 궤적**(`bank_x644_q38base_bank78_20260830` · Q3.8-27B-FP8 · `--gate 0` · alltools · user-sim gpt-5.2 · temp 0)을 대조 기준으로 per-step 추적해 이 태스크가 **어떤 지점에서 어떻게 0 이 되는지**를 축자로 확정하며, (c) 선행 판정(1f-12 · §2-B2)과 대조해 **한 항목을 정정**한다(«후행 0 문자열은 바이트 재현 불가» 는 틀렸다). 수리 실행·코드 수정 없음(제안만).
> 「이 태스크의 상태: 4」는 결과 파일이 아니라 오케스트레이터의 fails 카운트로 보인다 — 로컬에 reward 가 4개 있는 것이 아니다.
> 경로 주석: 지시서 경로 `tasks_20260821b/TASK_083.md` 는 훅 `C:\workspace\.claude\hooks\scaffold_guard.py:200`(정본 명명 = `/tasks_+\d{8}/TASK_<id>.md` · `20260821b` 의 `b` 불일치)에 막혀 x806 동기화 날짜 디렉터리에 두고, 지시서 디렉터리에는 `x854_TASK_083_pointer.md` 를 남겼다(`x852`·`x853` 선례).

## 0. 재료 부재 — 검색 경로 ([[77]] «없다»는 경로 없이 발화 금지)

| 지시된 파일 | 실재 | 검색 |
|---|---|---|
| `sim_results/bank_x806_base_nt4_B_20260821b.results.json.gz` | **없음** | `ls sim_results \| grep 20260821b` → `bank_t7333_smoke_*`, `bank_t7336_{smoke,halfA,halfB}_*` + `fb_*`/`trace_*` 만. 전 `*.results.json.gz` 를 열어 `"task_id": "task_083"` 을 센 결과 이들 파일은 **0건**. |
| `sim_results/bank_x806_base_nt4_B_20260821b.log.gz` | **없음** | 동상 |
| 대조 기준선 `undefined.results.json.gz` | **없음**(이름 자체가 미정의) | — |
| x806 계열 task_083 | **없음** | `ls \| grep "^bank_x806"` → task `001…059`(결손 있음) · `x818cloud_bank_x806_base_nt4_task_*` → `053,059…076,079,089,098,100`(074 는 drv 만). **083 은 결과·drv 로그 둘 다 없음.** `x818cloud_chain{A..D}/lane*/g3/g4.log.gz` 에 `task_083`·`083` 문자열 **0회** ⇒ base 큐에서 아직 안 돌았거나 안 내려왔다(리모트 확인 금지 조건이라 어느 쪽인지는 미확정). |
| 로컬에 있는 task_083 전부 | 13 파일 / 18 sim (표 §2-B) | 전 results gz 스캔. Q3.8 은 `x644`(base) · `t7393_laneC`·`relane2b151`·`x712_nightA`(ours) 4 sim, 나머지는 Q2.5 세대. |
| 선행 보고서 중 task_083 절 | `TASK_LEVER_MAP_AND_EXCLUSIONS_2026_08_16.md` §2-B2(:51-60) · `x737_next_run_plan_2026_09_04.md` §1f-12(:2782-2830) · :2465 · `FAILURE_AXIS_AND_FIX_ORDER_2026_08_15.md`:19 · `RESEARCH_MASTER.md` C491/C521 | `grep -l task_083 reports/facet_rft_2026/*.md` |

**x806 = `--gate 0` base 런이다**(`scripts/distill/tau2/x818_lanes/t2_base_worker.sh:32` `--gate 0 --domain banking_knowledge --retrieval_config alltools … --num_trials 4 --max_concurrency 4 --max_steps 200`). `t2_run_gated.py:219` `if a.gate:` 아래에서만 `t2_gate_patch` 가 적용되므로 **우리 레버는 이 런에 구조적으로 부재**하다(§4).

## 1. 채점 축 (지시 1단계 — 태스크 수준 사실, 런과 무관)

로컬 task_083 sim 18개 중 `reward_info` 가 있는 16개 **전부** `reward_basis=['ACTION']` · `reward_breakdown={'ACTION': …}` · `db_check={'db_match': False, 'db_reward': 0.0}`(채점 무관). DB 해시 축이 **아니다** — `dbdiff`/DB 필드 비교는 이 태스크에서 원인 규명 도구가 아니다([[69]]①).

gold `action_checks` 10행(x644 직독):
`log_verification`(7키) → `unlock_discoverable_agent_tool` ×5 → `call_discoverable_agent_tool:file_debit_card_transaction_dispute_6281` ×4. 마지막 4행은 `arguments` 가 **중첩 JSON 문자열**이고 파싱하면 **각 16키** — `customer_max_liability_amount` 가 없다(축자, GOLD[9]):

```
{"transaction_id": "btxn_333f5b136543", "account_id": "chk_dk83f5c2a1_gff", "card_id": "dbc_dk83f5c2a1_gff", "user_id": "dk83f5c2a1", "dispute_category": "unauthorized_transaction", "transaction_date": "11/09/2025", "discovery_date": "11/12/2025", "disputed_amount": 475.00, "transaction_type": "pin_purchase", "card_in_possession": true, "pin_compromised": "yes_shared", "contacted_merchant": false, "police_report_filed": false, "written_statement_provided": true, "provisional_credit_eligible": false, "card_action": "freeze_pending_investigation"}
```

env 가 선언하는 필수 인자는 **17개**(로컬 축자 — 도구 unlock 출력 x644 msg[49]: `customer_max_liability_amount: number (required) - The maximum dollar amount the customer could be liable for … Use -1 for unlimited liability.` / env 에러 `bank_n97_gpu1_main_20260806b` t1 msg[63]: `missing 15 required positional arguments: … 'provisional_credit_eligible', 'customer_max_liability_amount', and 'card_action'`).

## 2. 변이표

### 2-A. 지정 런 — 해당 없음

| run | trial | reward | gold | matched | missing | wrongarg | dup | blocked | extra |
|---|---|---|---|---|---|---|---|---|---|
| bank_x806_base_nt4 (지정) | 0..3 | **결과 없음** | — | — | — | — | — | — | — |

### 2-B. 로컬 task_083 전수 — `t2_forensic.mutation_diff`(정본) · gold 5 = unlock 1 + dispute call 4

| run (모델) | trial | reward | matched | missing | wrongarg | dup | blocked | extra | dispute call 키 수 / `customer_max_liability_amount` 포함 |
|---|---|---|---|---|---|---|---|---|---|
| **x644_q38base_bank78 (Q3.8 · base gate0)** | 0 | 0.0 | 1 | 4 | 4 | 0 | 0 | 1(freeze) | 17 ×4 / 포함 ×4 |
| t7393_laneC (Q3.8 ours) | 0 | 0.0 | 1 | 4 | 4 | 0 | 0 | 1 | 17 ×4 / 포함 |
| relane2b151 (Q3.8 ours) | 0 | 0.0 | 1 | 4 | 4 | 0 | 0 | 3 | 17 ×4 / 포함 |
| x712_nightA (Q3.8 ours) | 0 | 0.0 | 1 | 4 | 4 | 0 | 0 | 1 | 17 ×4 / 포함 |
| n97_gpu1_main (Q2.5) | 0 | 0.0 | 1 | 4 | 5 | 0 | 1 | 0 | 17 ×5 / 포함 |
| n97_gpu1_main (Q2.5) | 1 | — (context_window_exceeded) | 0 | 0 | 0 | 0 | 22 | 3 | 2~4키 오호출 6회(env 거부) → 17·18키 |
| n97_gpu1_batch_05 (Q2.5) | 1 | 0.0 | 1 | 4 | 2 | 0 | 3 | 0 | 17 ×5 |
| n97_gpu0_batch_05 t0/t1 · n97_gpu1_batch_05 t0 · cwe_batch_b · bx_t10_base · bx_t10_v2 · bankxfer t2/t4 ×2 | — | 0.0 | 0~1 | 4~5 | 0~1 | 0 | 0~4 | 0~3 | debit dispute 미호출 또는 credit 도구 오선택 |

읽기: **Q3.8 4 sim 은 모양이 완전히 같다** — unlock 은 맞고, dispute 4호출은 전부 `transaction_id` 로 gold 와 짝이 맞지만(missing 4 = wrongarg 4 는 같은 4건의 양면), 17번째 키가 실려 ACTION 불일치. x644 필드별 대조(§3-B 표)에서 4건 중 2건은 **17번째 키가 유일한 차이**이고 2건은 값 차이가 하나씩 더 있다.

## 3. step-by-step 결정 지점 추적 — 대조 기준 `bank_x644_q38base_bank78_20260830` task_083 trial 0 (60 msg · `user_stop` · reward 0.0)

⚠ 지정 런이 아니다. 같은 모델·같은 user-sim·같은 `--gate 0`·alltools·temp 0 이라 x806 의 가장 가까운 로컬 대리이지만, x806 4 trial 이 이 궤적을 밟았다는 주장은 하지 않는다.

### 3-A. 궤적 요약(축자는 결정 지점만)

| msg | 역할 | 내용 |
|---|---|---|
| [1] | user | 4건 분쟁 개요(«ATM that shorted me cash / ordered but never received / duplicate charge / family situation where someone I trusted took more than I authorized») |
| [2]-[9] | KB 검색 4회 + shell 2회 | [3]/[4]/[7] 에 **`Dispute the earliest (first) transaction when multiple duplicates exist.`** 와 `doc_…_032 Debit Card Provisional Credit Guidelines` 전문 배달 |
| [12]-[19] | read/verify | `get_user_information_by_name` → 2요소 확인 → `get_current_time` → `log_verification` ✓(gold 일치) |
| [22]-[37] | unlock/call | accounts → debit cards ×2 → transactions ×2 → `get_debit_dispute_status_7483`(«No debit card dispute…») — read 전부 수행, gold unlock 5/5 일치 |
| [41] | assistant | 4건 후보 제시(2번을 Adorama $324.99 로 오식별) + 질문 |
| [42] | user | 정정: «**$189.50** at **Premium Photo Frames Online** on **11/02/2025**» · GameStop: «I only authorized **$50**. I noticed the $475 charge on **11/12/2025**» |
| **[43]** | assistant | 추가 질문 8개. 결정 지점 ②: «6. What amount would you like to dispute — the **full $475**, or just the **$425 overage** (the amount above the $50 you authorized)?» · 결정 지점 ③: «contacting the merchant first is also required for provisional credit eligibility on that dispute» |
| **[44]** | user | «6) Please dispute **the $425 overage** (the amount above the $50 I authorized).» · «I did share my PIN with my nephew» |
| [48]-[50] | unlock | `file_debit_card_transaction_dispute_6281`(+`freeze_debit_card_3892`) — [49] 파라미터 17개 **전부 (required)**, 그중 `customer_max_liability_amount` |
| **[51]** | assistant | 결정 지점 ①: dispute 4호출 **각 17키**. 본문: «3. **Duplicate charge** — $67.25 (the earlier of the two identical charges)» · «4. **Unauthorized transaction** — $425.00 (the overage above your $50 authorization)» |
| [52]-[55] | tool | 4건 모두 성공(`Dispute ID: dsp_…`) — env 는 17키 호출을 **정상 접수** |
| [56]-[57] | write | `freeze_debit_card_3892(gff)` — gold 에 없음(EXTRA 1 · ACTION 축이라 감점 아님) |
| [58]-[59] | 종료 | 요약 → user «Yes, please close the card ending 8463 now and proceed with ordering a replacement…» 뒤 `user_stop` |

### 3-B. gold ↔ agent 필드별 대조 (msg[51] 4호출 · 파싱 후)

| # | txn | 17번째 키 | 그 외 차이 | 17번째 키 제거 후 **바이트 동일**? |
|---|---|---|---|---|
| 1 | btxn_36fd10197841 (ATM $150) | agent `150.00` / gold 부재 | 없음 | **True** |
| 2 | btxn_5230e065a1e7 ($189.50) | agent `189.50` / gold 부재 | 없음 | **True** |
| 3 | btxn_d937eaa1d21d ($67.25 dup) | agent `67.25` / gold 부재 | `provisional_credit_eligible` gold **true** ↔ agent **false** | False |
| 4 | btxn_333f5b136543 (GameStop) | agent `425.00` / gold 부재 | `disputed_amount` gold **475.00** ↔ agent **425.00** | False |

(바이트 비교 = agent 의 중첩 `arguments` 문자열에서 `, "customer_max_liability_amount": …` 만 제거한 뒤 gold 문자열과 `==`. #1·#2 는 `150.00`·`189.50` 후행 0 까지 **그대로 일치**한다.)

### 3-C. 결정 지점 판정

**① [49]→[51] 17키 호출 (4/4 · 지배 원인 · env)**. 필요한 정보는 문맥에 실재했고(도구 스키마가 `required` 로 명시), 모델은 스키마대로 정확히 호출했으며 env 는 [52]-[55] 에서 접수했다. 채점기는 gold 16키 문자열과 비교하므로 **정상 호출은 영구 불일치**다. 모델이 할 수 있는 «옳은» 행동이 점수와 양립하지 않는다 — 벤치 gold 결함(선행 §2-B2 와 동일). 반증 조건: 같은 도구를 쓰는 형제 태스크 gold 가 16키였다면 env 정의 변경으로 볼 수 있으나, 선행 1f-12 (d) «형제 6태스크 19 gold 액션 전부 17키, 083 만 16키» 가 반대다.

**② [43]→[44] 425 vs 475 (1/4 · model, user_sim 부차)**. KB 어디에도 «overage 만 분쟁» 개념이 없다(tool 출력 전수 grep — `overage`·`more than authorized`·`exceed…authoriz` 0건; [3] 에는 «Only use 'unauthorized_transaction' when fraud is NOT suspected (e.g., family member used card witho…)» 만 있음). 모델이 [43] 에서 **없는 선택지를 만들어 물었고** user-sim 이 [44] 에서 그 선택지를 골랐다. 값 획득 창은 있었고([42] «I noticed the $475 charge»), 정답은 문맥에 실재했다. user-sim 은 유도 질문에 답했을 뿐 오도하지 않았다(대본 축자는 미확인 — user-sim 층 부차).

**③ [43]→[51] `provisional_credit_eligible=false` (1/4 · model)**. [4] 에 배달된 `doc_…_032` 축자: REQUIRED 조건 = 적시 보고 · 카테고리(`'duplicate_charge'` 포함) · 서면 진술 · 계좌 OPEN, 그리고 NOT REQUIRED 항목 2 «Customer has not contacted merchant first (**for non-fraud disputes**)». 모델은 [43] 에서 후자를 duplicate 에 적용했다. gold 는 true. 문서가 «duplicate_charge 는 fraud 로 본다/안 본다» 를 명시하지 않으므로 해석 여지는 있으나, 같은 문서 REQUIRED 목록에 `'duplicate_charge'` 가 명시돼 있고 모델은 그 4조건을 모두 확인한 상태였다 ⇒ model (env 문서 모호성은 부차).

**④ 무효-호출 가지의 도달성**: 16키로 부르면 env 가 거부한다(로컬 축자 n97 t1 msg[63] `missing 15 required positional arguments … 'customer_max_liability_amount' …` — 다른 키가 더 빠진 사례이지만 필수 목록에 이 키가 있음은 확정). 거부된 호출이 ACTION 채점에 포함되는지는 **로컬에 증인이 없다**(16키 정확 호출 sim 0건 · `tasks.py` 소스 로컬 부재) — UNPROVEN 유지.

## 4. 레버 발화표

x806 은 `--gate 0`(`t2_base_worker.sh:32`) ⇒ `t2_run_gated.py:219 if a.gate:` 분기 밖 ⇒ `t2_gate_patch` 미적용. 아래 레버는 **전부 «구조적 미발화»** 이고 «발화했는데 무시»·«오발화» 는 성립할 수 없다. 이 sim 의 로그 줄 자체가 로컬에 없다(§0).

| 레버 | x806(지정) | 대조 x644(gate 0) | 비고 |
|---|---|---|---|
| `T2_SG_DOCS` · `T2_PIN_READ` · `T2_DEMANDED_STEP` · `T2_CLAIMPROV` · `T2_FOLLOWUP` · `T2_SEARCH_AGENT` · `FAB_STRIP` · `T2_ARG_PRODUCERS` · READ-FIRST · `T2_REQUIRE_DOC_DELIVER` · `T2_SEARCH_REARM` | 구조적 미발화(gate 0) | 구조적 미발화(gate 0) | 개입 없음 ⇒ «수리·레버가 개입하고도 못 샀나» 는 이 런에서 물을 수 없는 질문 |

참고(ours 팔): 선행 1f-12 가 ours 런(x742 번들 083)에서 `T2_COMPUTE op=filter in=0`·claimprov push(freeze→unfreeze→close) 를 기록했으나 ACTION 축이라 무영향으로 판정. 이번 로컬 ours 3 sim(t7393/relane2b151/x712)도 변이표 모양이 base 와 동일(missing 4/wrongarg 4)이다 — 레버 유무가 이 태스크의 결과를 바꾸지 않는다.

## 5. 선행 판정과 대조

| 항목 | 선행(§2-B2 2026-08-18/09-05 · 1f-12 2026-09-05) | 이번 |
|---|---|---|
| 채점 축 | `['ACTION']` | **동일** — 16 sim 전수 확인 |
| 지배 원인 | env(벤치 gold 결함: 17 required ↔ gold 16키) | **동일** — x644 base 에서도 4/4 호출 17키·env 접수·ACTION 0 |
| «정상 호출 가지» 관측 | relane2b151 `customer_max_liability_amount: 500.0` | **동일 가지** — base 도 같은 가지(x644 150/189.5/67.25/425) |
| «두 번째 차단막: gold 후행 0 문자열(`150.00`·`189.50`·`475.00`) — 무효-호출 경로로도 바이트 재현 불가» | 1f-12-0 083 행 | **정정**: 모델은 중첩 `arguments` 를 문자열로 직접 쓰고 `150.00`·`189.50`·`425.00` 을 그대로 썼다. #1·#2 는 17번째 키만 제거하면 **바이트 동일**(§3-B). 차단막은 «키 수» 하나뿐이다. |
| 값 오류의 층 | 1f-12: «[41] 유도 질문을 우리가 넣었나 판정 불가(D9)» — 425↔475 층 배정 불능 | base(gate 0) 에서도 **모델 스스로** «full $475, or just the $425 overage?» 를 물었다([43]) ⇒ 이 유도 질문은 우리 레버 산물이 아니라 **모델 고유** — D9 의 미결 한 칸이 닫힌다 |
| «Dispute the earliest (first) transaction» 위반(1f-12-2 #1: ours 083 이 second 를 골랐다) | ours 에서 관측 | base x644 는 **earlier 를 골랐고** txn id 가 gold 와 일치 — 태스크 불변 결함이 아니라 sim 변동 |
| 분모 제외 여부 | §2-B2 보류(자동 승격 금지) | 보류 유지 — 이 보고서는 관측만 추가. 승격은 [[68]]·x738 절차 |

## 6. 원인 확정

- **지정 런 `bank_x806_base_nt4` task_083**: 궤적 부재 ⇒ **UNPROVEN**. 리모트에 존재한다면 `/home/woori/iso_tau3/tau2-bench/data/simulations/bank_x806_base_nt4_task_083/results.json` · `/home/woori/scratch/logs/bank_x806_base_nt4_task_083_drv.log`(t2_base_worker.sh:26-34 경로) — 회수 후 §3 절차 재적용.
- **태스크 수준(로컬 증거로 확정)**: 주 원인 **env** — gold 4 dispute 액션이 env 필수 17키 중 `customer_max_liability_amount` 를 뺀 16키이고 채점이 ACTION(중첩 문자열 동일)이라, 스키마대로의 정상 호출은 4/4 영구 불일치. 부차 **model** — 대리 궤적에서 4건 중 2건은 값 오류가 추가로 있었고(425↔475 는 모델의 없는-선택지 질문, provisional_credit 은 문서 해석), 이는 17키 문제가 없었어도 그 sim 을 0 으로 만들었을 것이다.
- **our_layer**: 이 런은 gate 0 이라 우리 코드 경로가 궤적에 없다. 우리-층 주장 **0건**.

## 7. 처방 후보 (실행 안 함)

1. **회수**: x806 task_083 결과가 리모트에 있으면 sim_results 로 내려 §3 을 재적용. 없으면(큐 미처리) 이 태스크는 **유료 재실행 대상이 아니다** — 어떤 정상 궤적도 0 이다([[09]]).
2. **분류 결정**: §2-B2 «통과하려면 무효한 호출을 해야 한다» 를 [[68]]·x738 절차로 확정(분모 제외 여부). 이 보고서의 추가 관측 = base 도 같은 가지 · 바이트 차단막은 키 수 하나뿐.
3. **오케스트레이터 인자**: 지시서의 `bank_x806_base_nt4_B_20260821b.*`·`undefined.results.json.gz` 는 실재 파일명(`bank_x806_base_nt4_task_083.*` / `x818cloud_…`)과 다르다(TASK_075·x853 과 같은 증상). 태스크별 파일명 규약을 args 에 반영하지 않으면 다음 배치도 같은 «재료 부재» 보고가 반복된다.
4. **계기**: 1f-12-0 083 행의 «후행 0 바이트 재현 불가» 문장은 철회 대상(§5).

## 부록 — 사용 스크립트(스크래치)

`t083_census.py`(전수 변이표 · `t2_forensic.mutating_tools/mutation_diff`) · `t083_x644.py`(필드별 대조·궤적) · `t083_x644b.py`(축자 추출) · `t083_x644c.py`(바이트 반사실·KB grep) — `C:\Users\승원\AppData\Local\Temp\claude\C--workspace\6e081e70-44ce-498a-8d0c-09bf45d9812f\scratchpad\`.
