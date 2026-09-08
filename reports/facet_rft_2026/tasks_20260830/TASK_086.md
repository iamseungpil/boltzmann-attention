# TASK_086 (bank_x806_base_nt4 지시 · 실물은 x644 q38base 2026-08-30) — per-step 포렌식

> 작성 2026-09-08 · 전부 로컬 · SSH 0 · 코드 수정 0 · 커밋 0. 지시서 경로 `tasks_20260821b/TASK_086.md` 는 훅 `C:/workspace/.claude/hooks/scaffold_guard.py:200`(정본 명명 `re.search(r"/tasks_+\d{8}/", fp)` — `20260821b` 의 `b` 가 `\d{8}/` 에 걸려 False · 재현: `tasks_20260821b/TASK_086.md -> run_task_report=False`, `tasks_20260830/TASK_086.md -> True`)에 막혀 **실물 런 날짜(2026-08-30) 디렉터리**에 둔다(`x855_TASK_084_pointer.md`·`x853_TASK_082_pointer.md` 선례). 포인터 = `tasks_20260821b/x856_TASK_086_pointer.md`.

> **결론 먼저.** 지시된 런(`bank_x806_base_nt4`)의 task_086 궤적은 **로컬 어디에도 없다**(§0). x806 은 `--gate 0` **base 팔(레버 0)** 이므로 같은 팔·같은 bench sha `fc0055d`·같은 모델 `Qwen/Qwen3.8-27B-FP8`·같은 user-sim `gpt-5.2/low` 인 **`bank_x644_q38base_bank78_20260830` task_086 trial 0(66 msg · `user_stop` · reward 0.0)** 을 실물로 추적했고, ours 팔 3 sim(lev6c·re8141p11·t7393_laneC)을 대조로 붙였다. 판정: **DB 축 · 읽기 사슬 9/9 완주(날조 0·중복 0·BLOCKED 0) · 실패 = 쓰기 인자 7칸 + 미제출 1건**. 칸별 귀속 — ⑴ `customer_max_liability_amount` 500↔50 ×3(msg53): **model**(모델이 msg35 에서 스스로 «2 business days → $50» 을 고객에게 읊었고 msg36 에 발견일이 실재했는데 msg64 *"Since all disputes were reported within 60 days … $500"* 로 «신고일=오늘» 읽기를 택함) · 단 SupplyPro 행(발견 11/09·거래 10/31)은 문서 규칙으로 50 에 도달 불가 → **env**. ⑵ TechWorld 3칸(`card_not_present_fraud`/`online_purchase`/`pin_compromised=no` ↔ `card_present_fraud`/`pin_purchase`/`unknown`): **user_sim 1차**(대본의 *"I think someone skimmed my card"* 단서가 x644 user 발화 **0회** — ours 3 sim 은 전부 발화 · 대신 msg36 *"No, I don't believe either PIN was … compromised"* 로 카드 둘을 묶어 부정) + **model 2차**(msg33 거래 목록에서 ONLINE 표기 상품과 «MIAMI FL» 소재 표기 상품이 구분되는데 msg51 *"Since you were in Phoenix … card-not-present"* 로 자체 추론). ⑶ ATM deposit 미제출(086_13 MISSING): **user_sim 1차**(msg45 *"I'll go with what you have"* 뒤 msg52 *"Option B — let's not file"* 로 대본에 없는 철회) + **model 2차**(msg51 이 msg45 로 이미 닫힌 물음을 «no discrepancy» 선택지를 붙여 재질문). ⑷ `card_design` CLASSIC↔PREMIUM: **model**(doc_028 «5) Ask customer for preferred card design» 을 건너뛰고 배송 속도만 물음 · msg42 에 *"PREMIUM TIER: … PREMIUM design … design_fee $0 (complimentary)"* 실재 · 대본 §8 *"best free options"*). **our_layer = 비인과**(base 팔 · `x818_lanes/t2_base_worker.sh:31` `--gate 0` · x644 로그 사후검증 `T2_LEVER 0 · T2_GATE 0`). 선행 x737 §1f-7(«086 = model · WRONGARG 6 · 4× liability 0↔50» = re8141p11 sim 그 자체)과 **같은 방향**이되, 이번 실물에서 **user_sim 귀속 2건(TechWorld 단서 탈락·deposit 철회)이 새로 드러났다**. N97 08-04 의 «계좌 ID 4가지 형식 날조»(reach 실패)는 **소멸**.

---

## 0. 재료 상태 — 지시된 실물은 없다 ([[77]] 검색 경로)

| 지시된 파일 | 실재 | 검색 경로 |
|---|---|---|
| `sim_results/bank_x806_base_nt4_B_20260821b.results.json.gz` · `.log.gz` | **없음** | `ls sim_results \| grep -iE "20260821b\|task_086\|086"` → 20260821b 는 `bank_t7333_smoke_*`·`bank_t7336_halfA/halfB_*`·`fb_*`·`trace_*` 뿐 · `_086` 파일 0 |
| 대조 `undefined.results.json.gz` | **없음**(미충전 자리표시자) | — |
| x806 계열 task_086 | **없음** | `ls \| grep x806` → `bank_x806_base_nt4_task_{001..059}`(108 파일·결번 있음) + `x818cloud_bank_x806_base_nt4_task_{053,059..073,076,079,089,098,100}`(44 파일). 전 x806 results gz 를 `zcat \| grep '"task_086"'` → **0 HIT**. `x818cloud_laneA/laneB.log.gz` 큐 진행(`→ task_*`) = A: 053·061·063·066·068·069 / B: 060·062·065·067·071·074 → **086 은 큐에 오른 적 없음** |
| 로컬 task_086 전수 | **16 sim / 13 런 · 전부 reward 0.0 · 전부 `user_stop`** | 전 `*.results.json.gz` 로드(§0 스크립트) → 07-11 `bankxfer_floor_bank_t4`(2) · 07-18 `bank_all97_nt1_v2`(1) · 07-19 `bankxfer_floor_bank_t2`(2) · 08-04~06 `bank_n97_gpu1_main_*`(6)+`smoke8b_pin`(1) · **08-31 `bank_x644_q38base_bank78_20260830`(1)** · 09-06 `bank_lev6c_20260905_0935`(1) · 09-06 `bank_re8141p11_20260904_1100`(1) · 09-08 `bank_t7393_laneC_20260905_1533`(1) |
| 선행 per-task 보고서 | `tasks_*/` 에 086 **없음** | `find -iname "*TASK_086*"` 0 · `grep -l task_086 *.md tasks_*/*.md` → `N97_TASKWISE_FORENSIC_2026_08_04`(:162·:205·:439) · `x737_next_run_plan_2026_09_04`(§1f-7·P5) · `x738_q38_base97_census`(fail 55 목록에 086) · `TASK_LEVER_MAP_AND_EXCLUSIONS_2026_08_16:73` · `RESEARCH_MASTER`(C80·C89·C95) · `tasks_20260821b/TASK_077.md:106`(077–086 철회 인용) |

**x806 = base 팔 확정**: `scripts/distill/tau2/x818_lanes/t2_base_worker.sh:31-35` = `t2_run_gated.py --gate 0 --domain banking_knowledge --retrieval_config alltools --agent_model "Qwen/Qwen3.8-27B-FP8" … --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low --task_ids "$T" --num_trials 4 --max_concurrency 4 --max_steps 200`. x644 로그 머리 축자: *"clone HEAD : fc0055d / v1.0.1 태그 일치: fc0055dc / 레버 env : 0 개 (0 이어야 base)"*, 꼬리 축자(`:9749-9754`): *"[x644 사후검증] 우리-층 마커 (전부 0 이어야 base) T2_STACK 0 · T2_LEVER 0 · T2_SG_DOCS 0 · T2_SUPPRESS_AUTH 0 · T2_GATE 0"*. x644 는 nt=1·conc=1 이라 trial 은 **하나**뿐이다(지시서 «trial 이 둘이면 둘 다» 는 해당 없음).

«이 태스크의 상태: 4» 는 로컬로 검증 불가(x806 결과 파일이 없다). 로컬 16 sim 0/16 이므로 4/4 실패로 읽는 것과 모순되지는 않는다.

---

## 1. 채점 축

`sim['reward_info']`(x644·lev6c·re8141p11·t7393C 동일): `reward_basis=['DB']` · `db_check={'db_match': False, 'db_reward': 0.0}` · `nl_assertions=None` · `communicate_checks=None` · `env_assertions=[]`. **DB-해시 축**. `action_checks` 21행은 진단용 — read/unlock 계열(086_0~7·10·14·16) 11행 + **write 8행**(086_8·9·11·12·13 dispute 5 · 086_15 close · 086_17 order). x644 의 `action_match`: `log_verification`·unlock 7행 True · `call_discoverable_agent_tool` 전 행 False(벤치의 문자열 비교 — `unlock` 은 인자가 이름뿐이라 True).

**gold write 8행**(`tasks[task_086].evaluation_criteria.actions` 직독 · 5 dispute 전부 `file_debit_card_transaction_dispute_6281`):

| aid | txn(상호·일자) | category | discovery | amount | type | contacted | prov | **liability** | card_action |
|---|---|---|---|---|---|---|---|---|---|
| 086_8 | `btxn_c671c90e2724` QuickCash ATM 11/07 · gff | atm_cash_discrepancy | 11/07 | 100.0 | atm_withdrawal | false | true | **50** | keep_active |
| 086_9 | `btxn_a917670d8aaa` SupplyPro 10/31 · gff | goods_services_not_received | 11/09 | 189.99 | online_purchase | true | false | **50** | keep_active |
| 086_11 | `btxn_0fae026130ee` Phoenix Kitchen 11/08 · evergreen | incorrect_amount | 11/09 | 200.0 | signature_purchase | true | false | **50** | keep_active |
| 086_12 | `btxn_5690acdd3de5` TechWorld Miami 10/04 · evergreen | **card_present_fraud** | 11/14 | 625.0 | **pin_purchase** | false | true | **500** | close_and_reissue |
| 086_13 | `btxn_0788d2513c8c` Rho-Bank ATM #6172 deposit 11/06 · evergreen | atm_deposit_not_credited | 11/08 | **100.0** | atm_deposit | false | false | **50** | keep_active |
| 086_15 | `close_debit_card_4721` | `{card_id: dbc_cr47f8d2a1_evergreen, reason: fraud_suspected}` |||||||
| 086_17 | `order_debit_card_5739` | `{evergreen, EXPEDITED, delivery_fee 0, card_design **PREMIUM**, design_fee 0, "1520 East McDowell Road, Phoenix, AZ 85006"}` |||||||

(TechWorld 행 추가 gold: `pin_compromised: "unknown"` · `police_report_filed: false` · `card_in_possession: true`.) gold 에 CryptoMiner($350) dispute 는 **없다**(시나리오 노트 *"DISPUTE LIMIT enforcement on Entry Tier card (max 2) - user attempts 3 disputes, 3rd should be REJECTED"*). deposit 행 gold amount 100 = 시나리오 노트 *"user misremembers ATM deposit amount - agent must verify against DB and correct"*(DB 400 기록 · 고객 주장 입금 500 → 100).

---

## 2. 변이표 (`t2_forensic.mutation_diff(sim, F.mutating_tools(), tag=tag)` · 손 비교 0)

| run (팔·엔진) | msg | matched | missing | wrongarg | extra/dup/blocked | sidecar |
|---|---|---|---|---|---|---|
| **x644_q38base 08-30 (base·실물 대체)** | 66 | 2 (`log_verification` m13 · `close_debit_card{evergreen,fraud_suspected}` m60) | **6** (086_8·9·11·12·13·17) | **5** (m53 dispute 4 병렬 + m62 order) | 0/0/0 | absent(base·해당없음) |
| lev6c 09-05 (ours·a5417ce4·`T2_ARG_POLICY_AT_WRITE`+`T2_RULE_AT_WRITE` ON) | 50 | **3** (+086_8 ATM **일치** · +086_11 Phoenix **일치**) | 5 (086_9·12·13·15·17) | 2 (086_9·086_13) | 0/0/0 | absent |
| re8141p11 09-04 (ours·0b7d703b·`T2_RULE_AT_WRITE` ON·`ARG_POLICY_AT_WRITE` OFF) | 64 | 2 | 6 | **6** (dispute 5 + order) | 0/0/0 · `regen_blocked` **1**(turn 10·`unified_regen`·`T2_PHASE_PRECEDE` GB1 검증-선행 deny → 다음 턴 `get_current_time` 후 정상 진행 · 오답 칸과 비인과) | present(`simtag` join) |
| t7393_laneC 09-05 (ours·viewmax2+5) | 65 | **3** (+086_8 ATM · +086_12 **TechWorld 3칸 전부 일치**) | 5 (086_9·11·13·15·17) | 3 (086_9·11·13) | 0/0/0 | absent |

`missing` 6 ⊃ `wrongarg` 5 인 이유: `mut_key` = 이름+인자 전체라 인자 하나만 달라도 «같은 호출 없음»으로 센다. **실질 = 인자 틀림 5 호출 + 미제출 1 호출(086_13)**.

### 2a. 필드별 대조 — x644 (보낸 값 ↔ gold · 일치 필드 생략 · 원천 = msg53/m62 `tool_calls` 축자)

| aid | 필드 | 보낸 값 | gold | 그 값이 그 시점 문맥에 있었나 |
|---|---|---|---|---|
| 086_8 ATM | `customer_max_liability_amount` | **500** | 50 | doc_031(msg3·4·16·17·18·47 6회 도달) *"inform the customer of their liability exposure based on when they **noticed** the unauthorized activity: - Reported within 2 business days of statement: Maximum liability $50 - Reported within 60 days of statement: Maximum liability $500"* · user msg36 *"First noticed: Immediately on 11/07/2025"* · 모델 자신 msg35 *"$50 if reported within 2 business days of the statement, $500 within 60 days"* ⇒ **도달 가능** |
| 086_9 SupplyPro | `customer_max_liability_amount` | 500 | 50 | user msg36 *"First noticed: 11/09/2025"* · 거래 10/31 → 9일 ⇒ 위 규칙으로는 **500 이 맞다**. gold 50 은 KB 어느 문장으로도 안 나온다(x737 P5 «파생값 측정 자리» · 084 보고서 §1 «gold 행마다 규칙 상이» 와 동형) |
| 086_11 Phoenix | `customer_max_liability_amount` | 500 | 50 | user msg36 *"First noticed: 11/09/2025"* · 거래 11/08 → 1일 ⇒ **도달 가능** |
| 086_12 TechWorld | `dispute_category` | card_not_present_fraud | **card_present_fraud** | doc_031 *"'card_present_fraud': Physical card used fraudulently (not by customer) - USE THIS when fraud suspected and card was physically present - 'card_not_present_fraud': Online/phone transaction customer didn't make"* · msg33 거래 14 *"description: TECHWORLD ELECTRONICS MIAMI FL / type: debit_card_purchase"*(같은 목록의 온라인 상품은 *"CRYPTOMINER PREMIUM ONLINE"*·*"SUPPLYPRO WHOLESALE ONLINE"* 로 ONLINE 표기) · user msg36 *"Method: Not sure"* · msg45 *"I honestly don't know if it was in-person or online/phone"* · **대본의 «skimmed» 단서 x644 user 발화 0회** |
| | `transaction_type` | online_purchase | **pin_purchase** | 위 분류의 종속값(t7393C 는 `card_present_fraud`+`pin_purchase`+`unknown` 을 한 번에 냈다) |
| | `pin_compromised` | no | **unknown** | user msg36 *"**No**, I don't believe either PIN was shared, observed, or compromised"*(카드 둘 일괄) — 보낸 값은 **user 발화의 충실한 전사**. gold `unknown` 은 대본 *"I think someone skimmed my card at a gas station"* 에서만 나오는데 그 문장이 x644 에 **없다** |
| 086_13 deposit | **호출 자체 없음** | — | amount 100 · liab 50 | msg40 `get_atm_deposit_images` *"Amount Recorded: $400.00"* · user msg36 *"Deposited $500, only $300 was credited"* · msg45 *"If your records show $400 was credited for that deposit, then I'll go with what you have"* ⇒ 500−400 = **100 도달 가능**(lev6c·t7393C 가 실제로 100 제출) · 그런데 msg52 *"Option B — let's not file"* |
| 086_17 order | `card_design` | CLASSIC | **PREMIUM** | msg42(모델 자신의 msg41 검색 결과) *"PREMIUM TIER: - CLASSIC design: design_fee $0 - PREMIUM design (metallic silver): design_fee $0 (complimentary)"* · doc_028(msg16) *"5) Ask customer for preferred card design (CLASSIC, PREMIUM, or CUSTOM) and explain fees"* · 대본 §8 *"If asked about shipping speed or card design options: 'Just give me the best free options available for my tier'"* ⇒ **물었으면 도달**(re8141p11·t7393C user 는 물음에 *"PREMIUM (free metallic silver), please"* 로 답함) |

### 2b. 필드별 대조 — ours 대조 3 sim (같은 gold · n=1 · 엔진 sha 상이 · 태스크 인과 서술 금지)

| run | 086_8 ATM liab | 086_9 SupplyPro liab | 086_11 Phoenix liab | 086_12 TechWorld | 086_13 deposit | 086_15 close | 086_17 order |
|---|---|---|---|---|---|---|---|
| x644 (base) | 500 | 500 | 500 | CNP/online/no (3칸) | **미제출** | 일치 | CLASSIC |
| lev6c | **50 일치** | 500 | **50 일치** | **미제출**(user msg40 *"I can address the TechWorld/Miami charge after those if needed"* · msg49 *"hold off for now"* + 같은 턴 `###STOP###`) | amount 100 일치 · liab 500 | **미실행**(user_stop) | **미실행**(user_stop) |
| re8141p11 | **0** | **0** | **0** | CNP/online (2칸 · `unknown` 은 일치) | amount **200**(user 주장 300 채택·DB 400 무시) · liab 0 | 일치 | shipping_address *"Sabores del Sol, 1520 …"*(user msg59 가 그렇게 줌) |
| t7393C | **50 일치** | 500 | 500 | **3칸 전부 일치** | amount 100 일치 · liab 500 | **미실행**(msg63 교체 질문 → user msg64 답+`###STOP###`) | **미실행** |

`customer_max_liability_amount` 적중: 4 sim × 제출 dispute 15행 중 **50-gold 11행에서 적중 3**(lev6c 2·t7393C 1) — SupplyPro 는 **0/4**. 이 칸이 이 태스크의 지배 칸이다.

---

## 3. step-by-step 결정 지점 추적 (x644 t0 · 66 msg · 축자)

**읽기 사슬(msg2~34) — 흠 없음.** msg2 KB 검색 2발 → msg5 이름 요구 → msg7 `get_user_information_by_name` → msg9 2-factor 요구 → msg11 `get_current_time` → **msg13 `log_verification` 일치** → msg15·19 KB 5발(doc_028·031·009 도달) → **msg22 unlock 4종 병렬** → msg27 accounts → **msg29 `get_debit_cards`×2·`get_bank_account_transactions`×2·`get_debit_dispute_status` 병렬 5발** 전부 성공(msg30~34). N97 08-04 의 «계좌 ID 날조 4회» 는 여기 **없다** — `chk_cr47f8d2a1_gff`/`_evergreen`·`dbc_…` 전부 msg28·30·31 레코드에서 복사.

**msg35 (assistant)** — 6건 항목별 질문 + 규칙 낭독: *"If any transactions are unauthorized, your maximum liability depends on when you noticed them: **$50 if reported within 2 business days of the statement, $500 within 60 days, and potentially unlimited after 60 days**."* ← 모델이 규칙을 **정확히** 안다.

**msg36 (user)** — 6건 전부 + 확인 3건(§2a 인용). TechWorld: *"Method: Not sure, but I've never been to Miami — I was in Phoenix that whole week."* · PIN: *"No, I don't believe either PIN was shared, observed, or compromised."* · deposit: *"Deposited $500, only $300 was credited (missing $200). I don't have a receipt."* **대본의 «skimmed» 문장 없음.**

**msg37~40** — deposit 검증(`unlock get_atm_deposit_images` → *"Amount Recorded: $400.00 … IMAGES NOT AVAILABLE"*). 시나리오 노트의 «verify against DB» 를 **수행**.

**msg41~43** — close/reissue 절차 검색 → msg42 에 교체 카드 티어 표(*"PREMIUM TIER … Free expedited shipping … PREMIUM design (metallic silver): design_fee $0 (complimentary)"*) 도달.

**msg44 (assistant) — 한도·확인 질문** (§1 *"Entry Tier … maximum of 2 open disputes … Which two should I file first?"* · §2 TechWorld *"Date first noticed … today is 11/14/2025"* + *"in-person/physical … or online/phone? … If you genuinely don't know, let me know and I'll classify based on the available facts"* · §3 deposit *"our records show a $400 deposit … Could you confirm how much you actually deposited and how much showed as credited?"*).

**msg45 (user)** — *"Wait, there's a limit? … file these two first … QuickCash ATM … SupplyPro … I can come back for the CryptoMiner Premium charge later."*(대본 §4 그대로) · TechWorld *"I honestly don't know if it was in-person or online/phone"* · deposit *"Hmm… actually, I might be mixing it up with another deposit. **If your records show $400 was credited for that deposit, then I'll go with what you have.**"*(대본 §5 그대로 — 입금 500 주장은 철회하지 않았다).

**msg46~50** — «customer cannot say card-present or not» 검색(doc_031·034) → **msg49 unlock file tool** → msg50 도구 설명 도달: *"customer_max_liability_amount (number): The maximum dollar amount the customer could be liable for based on Regulation E reporting timing rules … Use -1 for unlimited liability."* · *"pin_compromised … 'unknown'"*.

**msg51 (assistant) — ★결정 지점 ①②③이 한 턴에**:
- ① TechWorld: *"**Classification:** Since you were in Phoenix the entire week and the charge is from Miami, I'll classify this as **card-not-present fraud** (the card was used without you, and you still have the physical card)."* — msg44 의 «classify based on the available facts» 를 실행했는데, 문맥의 사실(«MIAMI FL» 소재 표기·ONLINE 미표기·doc_031 «card was physically present → card_present»)과 반대 방향. 고객에게 되묻지 않고 확정.
- ② deposit: *"**Option A:** You deposited $500 and only $400 was credited → dispute the missing $100. **Option B:** You deposited $400 and the full $400 was credited → there's no discrepancy, and I won't file this one. **Option C:** Something else"* — msg45 가 이미 «기록 $400 수용」을 줬는데, 입금액을 400 으로 바꾸는 선택지를 모델이 **만들어 제시**.
- ③ 교체 카드: *"Please confirm: - Shipping address … - Delivery speed: Free expedited … or free standard"* — **card_design 질문 없음**(doc_028 5단계 생략).

**msg52 (user) — ★user_sim 이탈 지점**: *"For the Rho-Bank ATM deposit (9251): given what you're seeing in your records, I think it's **Option B** — let's **not file** a dispute for that deposit."* — 대본 §5 는 *"I'll go with what you have"* 까지이고 분쟁 철회 문장이 없다. 같은 턴 *"Delivery speed: Go with the best free option for my tier"*(디자인은 묻지 않았으므로 답도 없음).

**msg53 (assistant) — ★결정 지점 ④ 실행**: dispute 4 병렬 + unlock 2. 네 호출 전부 `"customer_max_liability_amount":500`. deposit 호출 **없음**. TechWorld `card_not_present_fraud`/`online_purchase`/`"pin_compromised":"no"`.
**msg54~57** — 4건 전부 성공(`dsp_3c585e9afa44`·`dsp_24a0475d1aa9`·`dsp_b6f75628c914`·`dsp_eab67a7a00f1`). env 는 인자값을 검증하지 않는다(BLOCKED 0).
**msg60~63** — close 일치 → order `"card_design":"CLASSIC"`.
**msg64 (assistant) — 규칙 오독의 자백 문장**: *"**Your liability:** Since all disputes were reported within 60 days, your maximum liability is **$500** per transaction under Regulation E."* ⇒ «reported» 를 오늘(11/14) 신고로 읽어 발견일과 무관하게 일괄 500. msg35 에서 «when you noticed» 로 옳게 읊은 규칙을 쓰기 시점에 다른 읽기로 바꿨다.
**msg65 (user)** — `###STOP###`.

---

## 4. 레버 발화표 (x644 = base 팔 · 로그 사후검증 블록 `:9749-9754` 축자)

| 레버 | x644(=x806 팔) | 판정 | ours 대조에서의 관찰 |
|---|---|---|---|
| `T2_SG_DOCS` | 0 | **미발화(설계상 OFF)** | lev6c·re8141p11 `levers_on` 에 ON |
| `T2_PIN_READ` | 0 | 미발화(OFF) | — |
| `T2_DEMANDED_STEP` | 0 | 미발화(OFF) | — |
| `T2_CLAIMPROV` | 0 | 미발화(OFF) | re8141p11 trace turn 3·41 `T2_GATE_REGEN claimprov` → `regen tool_calls=[]`(무해) |
| `T2_FOLLOWUP` | 0 | 미발화(OFF) | — |
| `T2_SEARCH_AGENT`(침묵 포함) | 0 | 미발화(OFF) | — |
| `FAB_STRIP` | 0 | 미발화(OFF) · 날조 자체가 0 | — |
| `T2_ARG_PRODUCERS` | 0 | 미발화(OFF) | — |
| READ-FIRST | 0 | 미발화(OFF) · 읽기는 자발적으로 9/9 | — |
| `T2_REQUIRE_DOC_DELIVER` | 0 | 미발화(OFF) | — |
| `T2_SEARCH_REARM` | 0 | 미발화(OFF) | — |
| **`T2_RULE_AT_WRITE` / `T2_ARG_POLICY_AT_WRITE`**(`t2_gate_patch.py:12688-12757` · 기본 OFF · `write_rules` = `a2/banking_knowledge.specific.json:10378`, 방출 가능값 50/500/-1) | 0 | 미발화(OFF) | lev6c(둘 다 ON) liability **2/4 적중** · re8141p11(`ARG_POLICY_AT_WRITE` OFF) **0/5**(전부 0) · n=1 씩 — 방향만 |
| `T2_DISTINCT_ARGS`(`:13616-13629`) | 0 | 미발화(OFF) | 이 태스크는 5 dispute 의 txn 이 전부 다르므로 표적 아님 |

**직전 런 이후 들어간 수리·레버가 이 궤적에 개입했는가**: **아니다 — 개입할 레버가 없다.** x806/x644 는 `--gate 0` 이라 `t2_gate_patch` 자체가 적용되지 않는다(`t2_run_gated.py:87` `--gate` default 0 · `:4` «--gate 1 이면 t2_gate_patch 적용»). 따라서 «개입하고도 왜 못 샀나» 는 이 궤적에 성립하지 않는 물음이고, ours 대조 sim 의 답은 §2b 와 x737 §1f-7 이 이미 적었다(write-point 전달 레버 미발화/부분 발화).

---

## 5. 선행 판정과 대조

| 선행 | 그때의 086 판정 | 이번 실물(x644) | 같은가 |
|---|---|---|---|
| `N97_TASKWISE_FORENSIC_2026_08_04:162` | *"계좌 ID를 **4가지 형식으로 날조**"* · `:439` «17→11 miss · P1+P3 발화» | 날조 0 · 읽기 9/9 · 계좌/카드 ID 전부 레코드 복사 | **달라졌다** — reach/discovery 실패 → **쓰기 인자값 실패**로 이동(v1.0.1 + Q3.8 + 검증 게이트 통과) |
| `RESEARCH_MASTER` C80(07-14) | 086 «16 ⋈ case = 전부 COVERAGE 미제출 · user 한도로 다 못 냄» | dispute 4/5 제출 · 미제출 1 은 한도가 아니라 user 철회 | 달라졌다(coverage 회복) |
| `x737_next_run_plan §1f-7`(09-04) | «086 우리층무관 · **model** · WRONGARG 6 (4× `customer_max_liability` 0↔50 등) · write_rules/distinct_args 실재하되 write-point 전달 레버 기본 OFF·미발화» — 그 sim = **re8141p11 0b7d703b**(§2 표의 6 wrongarg·liab 0×4 와 정확히 일치) | base 에서는 0 이 아니라 **500**(«신고일=오늘» 읽기) · TechWorld 3칸·deposit 미제출·CLASSIC 은 x737 이 다루지 않음 | **같은 방향 + 확장**: liability 는 model(+ SupplyPro 는 env) 유지 · **user_sim 귀속 2건 신규**(«skimmed» 단서 탈락 · Option B 철회) |
| `x737 P5` | «파생값 17칸은 수리가 아니라 측정 · compute_ops 08-19 의도적 공집합」 | `specific.json:68` 축자: *"file_debit_card_transaction_dispute.customer_max_liability_amount used thr=30 days while the policy text says 'within 2 business days of statement' … the engine filled a scored argument outright"* ⇒ 엔진이 값을 쓰지 않는 것은 **설계** | 유지 — 이 보고서도 수리 제안을 «문서 전달」에 한정한다 |
| `TASK_LEVER_MAP_AND_EXCLUSIONS_2026_08_16:73·:131` | «077–086 agent-realizable 아님 → 1.0.1 에서 수정됨 → 정상 표적」 | gold 17행 중 read/unlock 11행 + close 1행 = **12행이 실제로 실행됐다** | 유지(정상 표적) |

---

## 6. 원인 확정 (4주체 · 근거는 §3 축자만)

| 실패 칸 | primary | secondary | 반증 조건 |
|---|---|---|---|
| liability 500↔50 (ATM·Phoenix 2칸) | **model** — msg35 자기 낭독 + msg36 발견일 실재 + msg64 *"reported within 60 days … $500"* 오독 | — | 같은 문맥에서 다른 base sim 이 50 을 냈다면 model 확정 강화: lev6c(ours 이지만 규칙 문장이 write 시점에 전달됨) 2/2 적중 |
| liability 500↔50 (SupplyPro 1칸) | **env** — 발견 11/09·거래 10/31 → 문서 규칙으로 500 · gold 50 은 KB 0건 | model(발견일 기준으로도 500 이라 모델은 이 칸을 못 맞춘다) | KB 8편 어디에 «goods_services_not_received → $50» 이 있으면 철회. 4/4 sim 전부 500 또는 0 |
| TechWorld 3칸 | **user_sim** — 대본 §5 *"I think someone skimmed my card at a gas station"* 발화 0회(ours 3 sim 은 전부 발화) · msg36 PIN 일괄 *"No"* | **model** — msg33 «MIAMI FL» vs «ONLINE» 표기 + doc_031 «physically present» 가 있었는데 msg51 자체 추론으로 CNP 확정, 되묻지 않음 | t7393C: 같은 모델이 «skimmed·unsure» 단서를 받자 3칸 전부 적중 ⇒ 단서 유무가 갈랐다 |
| deposit 미제출 | **user_sim** — msg52 *"Option B — let's not file"* 는 대본 §5 에 없다(§5 는 «go with what you have» 까지) | **model** — msg45 로 이미 «기록 400 수용」이 왔는데 msg51 이 «no discrepancy» 선택지를 신설해 재질문 · lev6c 는 같은 재료로 msg43 *"I'll use the system record value … $100 disputed"* 자발 제출 | t7393C 도 같은 메뉴를 냈는데 user 가 *"file it as $100 missing"* 을 골랐다 ⇒ 메뉴가 아니라 user 선택이 갈랐다 |
| `card_design` CLASSIC↔PREMIUM | **model** — doc_028 5단계 «Ask … card design» 생략 · msg42 무료표 실재 · 대본 §8 은 «물으면 best free» | — | 물은 sim(re8141p11·t7393C)의 user 는 PREMIUM 을 골랐다 |
| our_layer | **비인과** — base 팔(`t2_base_worker.sh:31` `--gate 0`) · 로그 `T2_LEVER 0 · T2_GATE 0` · BLOCKED 0 · 날조 0 | — | — |

**sim 전체 primary = model**(칸 4/7 + 미제출의 2차 원인) · **secondary = user_sim**(TechWorld 단서 탈락·deposit 철회 — 두 사건 모두 x644 user-sim 발화에서만 관찰) · env 1칸.

---

## 7. 처방 후보 (제안만 · 수리 실행 0 · [[23]] gold 는 진단용)

1. **liability(지배 칸)** — `T2_ARG_POLICY_AT_WRITE`+`T2_RULE_AT_WRITE`(`t2_gate_patch.py:12697·:12736`) 의 write-point 전달을 기본 ON 후보로 A/B. 근거: lev6c(ON) 2/4 ↔ re8141p11(부분 OFF) 0/5 ↔ base 0/4 — n=1 이라 방향뿐. 전달 문구는 `specific.json:10378 write_rules` 의 doc_031 축자여야 하고 값 산출은 여전히 모델 몫([[62]]). SupplyPro 류 «비-fraud → 50» gold 는 KB 부재라 이 레버로도 닫히지 않는다 — **측정 자리로 남긴다**(x737 P5).
2. **TechWorld 분류** — 우리 층이 할 일은 없다(user_sim 단서 탈락은 벤치 user-sim 온도/효과 문제 · `--user_temp 0.0 --user_reasoning_effort low` 에서도 sim 간 발화가 달랐다). 굳이 잡으려면 `T2_SG_DOCS` 류가 «category 결정 문서 문장」(doc_031 «physically present / online-phone」)을 write 직전에 재전달하는 것 — 이미 문맥에 6회 실재했으므로 기대치 낮음.
3. **deposit 철회** — user_sim 귀속. 다만 «검증-후-정정」 태스크(노트 *"agent must verify against DB and correct"*)에서 모델이 «정정된 금액으로 제출」 대신 «철회 선택지」를 만드는 패턴은 `T2_COVERAGE_FOLLOWUP` 의 표적(gold 5건 중 4건 제출 뒤 종료)일 수 있다 — base 에는 없다.
4. **card_design** — doc_028 5단계(«Ask card design») 생략은 절차 누락이라 `T2_PROC_*`/`T2_DEMANDED_STEP` 표적. ours 2 sim 은 물었고 적중했다.
5. **재료** — x806 task_086 실물이 회수되면(리모트 `/home/woori/scratch/logs/bank_x806_base_nt4_task_086_drv.log`·`data/simulations/`) 이 보고서의 §2~§6 을 4 trial 로 재계산해야 한다. 특히 «skimmed» 단서 발화율과 Option-B 철회율이 trial 간 얼마나 흔들리는지가 user_sim 귀속의 양화다.

---

### 부록 A. 재현 스크립트(전부 로컬 · 스크래치)
- 변이표·필드 대조: `sys.path.insert(0,'.'); import t2_forensic as F; F.mutation_diff(sim, F.mutating_tools(), tag=tag)` + gold `tasks[].evaluation_criteria.actions` 를 `transaction_id` 로 join 해 `norm_args` 필드 비교.
- 궤적 덤프: `scratchpad/traj_086_{x644,lev6c,re8141p11,t7393C}_t0.txt` · 압축본 `cond_086_*.txt` · 태스크 정의 `task_086_def.json`.
- x644 로그: `sim_results/bank_x644_q38base_bank78_20260830.log.gz` `:7610`(*"65/78 … Running task task_086, trial 1"*) · `:7748`(결과 표) · `:9749-9754`(우리-층 마커 0).
