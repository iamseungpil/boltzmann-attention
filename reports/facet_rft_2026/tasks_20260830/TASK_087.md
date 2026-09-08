# TASK_087 — bank_x806_base_nt4 궤적 per-step 포렌식 (2026-09-08)

> 지시서 경로 `tasks_20260821b/TASK_087.md` 는 훅 `C:/workspace/.claude/hooks/scaffold_guard.py:200`(정본 명명 `/tasks_+\d{8}/TASK_<id>.md` — `20260821b` 의 `b` 불일치 → `_is_run_task_report=False`)에 막혀 실물 대리 런 날짜(2026-08-30) 디렉터리에 둔다(`x856_TASK_086_pointer.md`·`x855_TASK_084_pointer.md` 선례). 포인터 = `tasks_20260821b/x857_TASK_087_pointer.md`.

> 결론 먼저: **지정 런(`bank_x806_base_nt4`)에는 task_087 궤적이 없다** — 결과 json·drv 로그·레인 로그·오염 manifest 어디에도 없다(§0 검색 경로). «이 태스크의 상태: 4» 는 reward 0 네 개가 아니라 **미착수/미회수** 다. 따라서 «x806 에서의 실패 원인»은 **UNPROVEN(미측정)**.
> 대신 같은 task-set sha(`fc0055d`)·같은 모델(Qwen3.8-27B-FP8)·같은 user-sim(gpt-5.2 low)·**같은 base 팔(`--gate 0` · T2_LEVER 0)** 인 `bank_x644_q38base_bank78_20260830` task_087 t0 를 **주 대리**로 per-step 추적하고, ours 팔 2 sim(`bank_relane2b151_20260904_2138`·`bank_t7393_laneC_20260905_1533`)을 대조했다.
> **한 줄 판정(주 대리 x644·base)**: reward 0 · DB 축 · gold 변이 7 중 **6 일치**(log·unfreeze×2·velocity clear·close·order) · 실패 = `file_debit_card_transaction_dispute_6281` **인자 2칸**(`discovery_date` 11/14↔gold 11/06 · `customer_max_liability_amount` 500↔gold 50) · 결정 지점 **msg 38→49→61** · primary = **model**(«처음 알아챈 날»을 묻지 않고 오늘 날짜를 대입 · 배달된 doc_031 축자 «Date customer first noticed the issue» 무시) · secondary = **user_sim**(같은 «거래 확인» 질문에 x644 는 11/06 문자 단서를 봉인, relane2b151 은 개방 — 조건절 발화 분산) · our_layer **비인과**(base 팔). 로컬 task_087 완주 sim 17개 전부 reward 0 — 어느 스택에서도 통과 기록 없음.
> 수리 실행·코드 수정 없음(제안만). 표기 [[77]]: 원인 절 = ①주장+양화 · ②축자+파일:줄 · ③반증 조건 · ④선행 확인 경로.

## 0. 재료 부재 — 검색 경로 ([[77]] «없다»는 경로 없이 발화 금지)

| 지시된/기대되는 재료 | 실재 | 검색 경로 |
|---|---|---|
| `sim_results/bank_x806_base_nt4_B_20260821b.results.json.gz` · `.log.gz` | **없음** | `ls sim_results \| grep -iE "20260821b\|task_087\|x806.*B_"` → `20260821b` 는 `bank_t7333_smoke`·`bank_t7336_{smoke,halfA,halfB}`(+fb/trace) 뿐 |
| 대조 기준선 `undefined.results.json.gz` | **없음** | `ls \| grep undefined` → 0 |
| x806 계열 task_087 results/drv | **없음** | 로컬 국면 `bank_x806_base_nt4_task_{001..059}` + 클라우드 국면 `x818cloud_bank_x806_base_nt4_task_{022,053,059..074,076,079,089,098,100}` 전 파일 `zcat \| grep task_087` → **0 hit** |
| 레인 로그 | **0** | `x818cloud_laneA.log.gz`·`laneA_fb`·`laneB`·`lane_rep1`·`lane_rep1_fb` 5개 `zcat \| grep -c task_087` → 전부 0 |
| 오염 manifest | **없음** | `x818_cloud_srt_contamination_manifest.json` `grep -c 087` → 0(x806 키 23개: 022·053·059~074·076·079·089·098·100) |
| 전 트리 task_087 파일 | **0** | `find ba-frft -iname "*task_087*"` → 0 |
| 큐 위치 | 리모트라 직독 불가(SSH 금지) | `x812_cloud/goC.sh:2-3` *"우선군 소진 → … ②base 잔여 30(결손 7 + 022 + 후순위 24)"* · `goD.sh:2` *"base 잔여 32"* — 큐 `/root/q_base_rest.txt`. 087 은 «측정된 61» 밖 = 미착수 잔여군(TASK_078 §0 과 같은 판정) |
| 로컬 task_087 완주 sim | **17 sim · 전부 reward 0.0** | 전 `*.results.json.gz` 스캔: Qwen3.8 3(x644·relane2b151·t7393_laneC) · Qwen2.5 12(n97 gpu0 08-04/05/06/06b · smoke6/8b · all97_nt1_v2) · 구 sha 2(bankxfer_floor t2/t4) |

반증 조건: 리모트 `ls ~/logs /root/logs \| grep task_087` 또는 `grep task_087 /root/q_base_rest.txt` 가 drv 로그·큐 잔존을 보이면 «미착수» → «미회수(`x812_cloud/lane.sh:24-30 persist()` 는 results.json 없으면 침묵)» 로 바꾼다.

## 1. 채점 축 (지시 1단계)

x644 task_087 t0 `sim['reward_info']`: `reward_basis=['DB']` · `db_check={'db_match': False, 'db_reward': 0.0}` · task 정의 `evaluation_criteria`: `env_assertions=None`·`communicate_info=None`·`nl_assertions=None` ⇒ **DB-해시 단독 축**. `action_checks` 20행은 진단 표시일 뿐이고, **`call_discoverable_agent_tool` 행은 인자 문자열의 공백 문체 때문에 값이 같아도 전부 `action_match=false`** 로 찍힌다(예 087_2 `get_all_user_accounts` gold `{"user_id": "mt35a7c9d2"}` ↔ 보낸 `{"user_id":"mt35a7c9d2"}` → false). ⇒ 변이 판정은 §2 `mutation_diff` 만 쓴다(x737 §1f-12 «DB-basis 는 args_equal 필수» 규율).

gold 20 aid(mutating 굵게):
```
087_0  **log_verification**(Marcus Thompson · mt35a7c9d2 · 2025-11-14 03:40:00 EST)
087_1/2   unlock+call get_all_user_accounts_by_user_id_3847(mt35a7c9d2)
087_3/4   unlock+call get_debit_cards_by_account_id_7823(chk_blue)
087_5a/5b unlock+**unfreeze_debit_card_3893**(dbc_blue)
087_6/7   unlock+**clear_debit_card_fraud_alert_4892**(dbc_blue, velocity_clear)
087_8     call get_debit_cards(chk_gff)          087_9  **unfreeze**(dbc_gff)
087_10    call get_debit_cards(chk_evergreen)    087_11/12 unlock+call get_bank_account_transactions_9173(chk_evergreen)
087_13/14 unlock+**file_debit_card_transaction_dispute_6281**(btxn_70fb52ef4e29 · card_present_fraud · txn 11/06/2025 · **discovery_date 11/06/2025** · 499.99 · pin_purchase · possession true · pin no · merchant false · police false · statement true · provisional true · **liability 50** · close_and_reissue)
087_15/16 unlock+**close_debit_card_4721**(dbc_evergreen, fraud_suspected)
087_17/18 unlock+**order_debit_card_5739**(chk_evergreen · EXPEDITED · fee 0 · PREMIUM · fee 0 · 2847 Pine Street, Apt 12B, Seattle, WA 98101)
```
gold 는 evergreen 의 customer-initiated fraud alert 를 **clear 하지 않는다**(카드를 닫는다). 태스크 note 축자: *"Card 3 (Evergreen, 8934): customer-initiated fraud alert due to an ACTUAL fraudulent transaction … discover actual fraud, file dispute, close the compromised card, and order a replacement."*

## 2. 변이표 — 정본 `t2_forensic.mutation_diff(sim, mutating_tools())` (손 비교기 0)

| 파일 | 팔 | 일자 | trial | reward | msgs | matched | missing | wrongarg | extra | dup | blocked |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bank_x644_q38base_bank78_20260830` | **base · gate 0 · nt1** | 08-30 | t0 | 0.0 | 73 | **6** log@12 · unfreeze(blue)@45 · unfreeze(gff)@45 · clear(blue velocity_clear)@45 · close(evergreen fraud_suspected)@63 · order(chk_evergreen)@69 | 1 · file(evergreen)@g | **1 · file(evergreen)@61** | 0 | 0 | 0 |
| `bank_relane2b151_20260904_2138` | ours · gate 1 · engine `0b7d703b`(dirty) · seed 626729 | 09-04 | t0 | 0.0 | 58 | 4 log@13 · unfreeze×2@36 · clear(blue)@36 | **3** · file@g · close@g · order@g | 0 | 0 | 0 | 0 |
| `bank_t7393_laneC_20260905_1533` | ours · gate 1 | 09-05 | t0 | 0.0 | 68 | 5 log@16 · unfreeze×2@40 · clear(blue)@40 · close(evergreen)@64 | 2 · file@g · order@g | 2 · **clear(evergreen customer_verified)@40** · file(evergreen)@60 | 0 | 0 | 0 |
| `bank_n97_gpu0_main_20260806b` | Qwen2.5-32B · 구 스택 | 08-06 | t0 | 0.0 | 120 | 4 | 3 · file · close · order | 2 · clear(evergreen)@90 · file@116 | 0 | 0 | 2 · log@20,@24(«already exist») |
| `bank_n97_gpu0_main_20260806b` | 〃 | 08-06 | t1 | 0.0 | 160 | 5 | 2 · file · order | 3 · clear(evergreen)@105 · file@136 · order@156 | 0 | 0 | 3 |
| `bank_n97_gpu0_main_20260804` | 〃 | 08-04 | t0 / t1 | 0.0 | 62 / 146 | 1 / 5 | 6 / 2 | 0 / 3 | 0 | 0 | 0 / 2 |

x644 WRONGARG 필드별 대조(보낸 ↔ gold, 17칸): 15칸 동일(transaction_id·account_id·card_id·user_id·`card_present_fraud`·txn 11/06/2025·499.99·`pin_purchase`·possession true·pin `no`·merchant false·police false·statement true·provisional true·`close_and_reissue`) · **불일치 2칸**: `discovery_date` **11/14/2025** ↔ **11/06/2025** · `customer_max_liability_amount` **500** ↔ **50**. missing 1 = 같은 gold 항목(wrongarg 의 짝).
⚠ `request_human_agent_transfer`(relane2b151 msg49, **user 측** 호출)는 `a2/env_surface.json:623-626` 에 `mutates: true` 로 선언돼 있으나 `mutation_diff` 는 assistant 턴만 걷는다 → EXTRA 0 은 그 사정 안의 값이다(x737 §1f-5 #26 «087 EXTRA 재검» 은 여기서도 열려 있다).

## 3. 주 대리 `bank_x644_q38base_bank78_20260830` t0 — step-by-step 결정 지점 (축자)

| msg | 역할 | 사건 |
|---|---|---|
| 1 | user | *"THREE different debit cards decline on me today! … 'CODE 05' or 'Do Not Honor'"* |
| 2–5 | assistant | KB bm25/dense + `shell ls` — doc_003(error codes) 회수 |
| 6–13 | 검증 | 이름+DOB+email → `get_user_information_by_name` 2건 중 mt35a7c9d2 선택 → `get_current_time` → `log_verification` **성공(gold 087_0 일치)** |
| 14–27 | read | doc_009·028 검색 → unlock `get_all_user_accounts` → 3계좌(blue 3450 / gff 1275 / evergreen 12850) → unlock `get_debit_cards` → 3장 병렬: blue **FROZEN + velocity_blocked true** · gff **FROZEN** · evergreen **ACTIVE + fraud_alert_active true, alert_source customer_initiated** |
| 28–37 | read | doc_026(freeze/unfreeze) 검색 → unlock `get_bank_account_transactions` → 3계좌 병렬. evergreen: *"btxn_70fb52ef4e29 · date: 11/06/2025 · TECHZONE ELECTRONICS LOS ANGELES CA · -499.99"* |
| **38** | assistant | 진단 발화. evergreen 항: *"it has a **security alert that was initiated from your side of the account** (for example, through the app or a prior support request). I can clear it, but first I need you to confirm these recent transactions are yours: … TechZone Electronics, Los Angeles – $499.99 (11/06)"* · 질문 3개: unfreeze 둘? / *"Are the three Evergreen transactions above legitimate?"* / 7291 블록 해제? — **경보 자체(언제·왜 걸었나)는 묻지 않았다.** |
| **39** | user | *"I actually froze both my Blue (7291) and Green (4521) cards last week … please unfreeze both now. For the Evergreen transactions: Whole Foods $89.50 — yes … Netflix $34.99 — yes … TechZone Electronics in Los Angeles $499.99 — definitely NOT me. I was in Seattle that day."* ⇒ 대본 §12(거래 확인)만 개방 · **§11(«On November 6th, I got a text about a suspicious charge … I replied right away to put a fraud alert on the card»)은 미개방** — §11 의 조건절은 *"If agent asks about fraud alert on Evergreen card"* 인데 에이전트는 경보를 «알려주고» 거래를 «물었다». |
| 40–44 | assistant | unlock unfreeze/clear + KB 검색 → **msg 43 에 doc_031 전문 배달**: *"inform the customer of their liability exposure based on **when they noticed** the unauthorized activity: Reported within 2 business days of statement: Maximum liability $50 / within 60 days: $500"* · *"7. **discovery_date** (string, MM/DD/YYYY) - **Date customer first noticed the issue.**"* · 같은 배달의 doc_036: *"What You'll Need … **When you first noticed the issue**"* |
| 45–48 | write | unfreeze(blue)·unfreeze(gff)·clear(blue, velocity_clear) **셋 다 gold 일치** |
| **49** | assistant | *"**Your liability:** Because you're reporting this within 60 days, your maximum liability under federal law is $500."* → 질문 5개: ① 실물 카드 소지 ② PIN 노출 ③ 거래 유형 ④ 서면 진술 ⑤ 카드 폐쇄+재발급 — **«처음 알아챈 날»은 없다.** 오늘(11/14, msg 11 `get_current_time`)을 신고일로 놓고 60일 구간을 계산한 것. |
| **50** | user | *"1. Yes, I still have the physical Evergreen card … 2. No — my PIN was never compromised … 3. I'm not 100% sure, but **the text message I got** made it sound like it was a PIN purchase. 4. Yes … written statement. 5. Yes, please close the Evergreen card and order a replacement now. Also, I'd like to file a dispute"* — **«문자를 받았다»는 단서가 노출**(=거래 당시 통지가 있었음)됐으나 후속 질문 0. |
| 51–57 | read | doc_030·credit replacement·Evergreen = **PREMIUM TIER**(doc_005:20) 확인 |
| 58–60 | unlock | `file_debit_card_transaction_dispute_6281`(스키마 축자 *"discovery_date … Date when the user first noticed the issue"* · *"customer_max_liability_amount … based on Regulation E reporting timing rules"*) · `close_debit_card_4721` |
| **61** | **write** | `file_debit_card_transaction_dispute_6281(… "discovery_date":"11/14/2025" … "customer_max_liability_amount":500 …)` → env 수락 `dsp_8aa70a7acbbf` · Provisional Credit ISSUED — **결함이 DB 에 박힌 지점** |
| 63–64 | write | `close_debit_card_4721(evergreen, fraud_suspected)` **gold 일치** |
| 65–70 | write | 배송 옵션 제시(Premium: Expedited 무료·Premium 디자인 무료) → user *"Expedited (free) … Premium (metallic silver) … 2847 Pine Street"* → `order_debit_card_5739(EXPEDITED, 0, PREMIUM, 0, 주소)` **gold 일치** |
| 72 | user | *"Thank you so much for helping me sort all of this out! … ###STOP###"* — 대본 §20 정상 종결 |

**값 실재 여부**: msg 61 시점에 «11/06 = 처음 알아챈 날»은 문맥에 **없었다**(user 가 발화하지 않음). 카드 레코드에도 경보 일자 필드가 없다(`fraud_alert_active`·`alert_source` 뿐). 얻는 길은 **질문 하나**(§11 «경보를 묻기» 또는 doc_031/036 이 시키는 «언제 처음 알아챘나»)뿐이었고, 에이전트는 다른 5개는 물으면서 이것만 빠뜨렸다. 반면 «오늘 = 11/14»는 msg 11 에 실재 → 그것을 대입했다. liability 50 은 discovery 11/06 에서 도출되는 종속 값(2 영업일 이내 → $50)이라 **단일 결손이 2칸 불일치**를 만든다.

## 4. 대조 ours 팔 ① `bank_relane2b151_20260904_2138` t0 (x737 §1f-12 «087» 과 같은 sim · seed 626729)

| msg | 사건 |
|---|---|
| 40 | assistant: *"there's a security alert on it that appears to have been initiated from your side … Before I can clear it, could you confirm: **do you recognize your recent transactions**"* |
| **41** | user: *"A security alert? Oh right — on **November 6th** I got a text about a suspicious charge for like $500 at an electronics store in California. I was definitely in Seattle that day, so I replied right away to put a fraud alert on the card."* ⇒ **§11 개방 — discovery_date 재료가 문맥에 실재** |
| 42 | assistant: *"I'm not able to simply clear the alert myself — **that requires review by our security team**"* (msg 32 배달 doc_039 축자 *"Escalate to security team when: … Customer reports transactions they did not make"* 를 사건 전체 이관으로 읽음) |
| 44–48 | transactions 조회 → 11/06 TechZone 확인 → *"it needs to be reviewed by our security team, who can also help you file a dispute"* — `file_…_6281`·`close_…_4721`·`order_…_5739` 는 msg 20/43 의 shell/검색 결과에 이름이 실재 |
| 49–55 | user 도구 `request_human_agent_transfer` → GB2 deny 1회(fb turn 50 *"[POLICY GATE GB2_NOTICE_BEFORE_TRANSFER] blocked … Send the user exactly this message now: TRANSFER NOTICE"*) → 통지 후 `transfer_to_human_agents(fraud_or_security_concern)` 성공 |

여기서는 x644 와 **분기점이 다르다**: 재료는 있었고(msg 41) 쓰기 자체를 포기했다(msg 42/48). 판정 = model(x737 §1f-12 유지). user-sim 은 같은 종류의 질문에 §11 을 열어 줬다 — x644(msg 38→39)와의 차이는 에이전트 질문의 미세 차이(«확인해 달라» vs «알아보는가»)와 gpt-5.2 의 조건절 해석 분산이다 ⇒ x644 의 secondary=user_sim 근거.

## 5. 대조 ours 팔 ② `bank_t7393_laneC_20260905_1533` t0

| msg | 사건 |
|---|---|
| 29–37 | doc_039 두 번 배달(*"Clear the alert/block when: … For fraud alerts: The alert was customer-initiated AND customer confirms their recent transactions are legitimate"*) → transactions(evergreen) 조회 |
| **38·40** | user 발화 없이(직전 user = msg 9) 연속 assistant: *"I apologize for the confusion, Marcus. Let me actually execute the fixes now."* → *"**You're right, Marcus — I should have executed those fixes right away** rather than waiting."* → unfreeze×2 · clear(blue velocity_clear) · **clear(evergreen, customer_verified)** — 거래 확인(§12) **이전**에 evergreen 경보를 지웠다(doc_039 «AND customer confirms … legitimate» 위반 · gold 에 없는 쓰기 → DB 상이). 「You're right」의 상대는 손님이 아니라 우리 리마인더다(x737 §1f-12 D8 «push 발자국» 과 동형 — 단 laneC 의 fb 사이드카가 로컬에 없어 **어느 리마인더인지 미확정**). |
| 45–46 | 거래 목록 제시 → user: TechZone *"definitely not me"* + 동결 자백 |
| 47–59 | doc 검색 → dispute status 조회(0건) → PIN/서면 질문 2개(**처음 알아챈 날 없음**) → user 답 |
| **60** | `file_…_6281(… "discovery_date":"11/14/2025" … "customer_max_liability_amount":499.99 …)` — x644 와 같은 결손 + liability 에 분쟁 금액을 넣음(자기 논리 «60일 이내 $500»조차 아님) |
| 62–66 | close(evergreen, fraud_suspected) gold 일치 → 배송 옵션 제시 |
| **67** | user: *"Go with the best free option — so expedited … Premium metallic silver … 2847 Pine Street … Thank you so much … ###STOP###"* — 선택과 종결이 **한 메시지** ⇒ `order_debit_card_5739` 를 부를 턴이 없다(대본 §15+§20 병합 · user_sim 귀속). |

## 6. 레버 발화표

**x644(base)**: 로그 `bank_x644_q38base_bank78_20260830.log.gz:9751 "T2_LEVER 0"` · `:9754 "T2_GATE 0"` · 이 sim 줄은 진행 상태뿐(`Running task task_087, trial 1` … 1137s). 지시된 레버 11종 **전부 미발화(설계상)** — 개입 0.

**relane2b151(ours)** — `trace_bank_relane2b151_20260904_2138.jsonl.gz` task_087 줄 384 · fb 39:

| 레버 | 발화 | 판정 |
|---|---|---|
| `T2_CLAIMPROV`/`T2_CLAIM_PROV` | 98/10 마크 · reminder-user turn 5·31·50 — 문면 *"you told the customer you WOULD do the following … None: None; None: None; None: None"* | **오발화(내용 결손)** — x737 D8 «None: None» 재현. turn 31 은 이관 전 «약속 이행» 요구였으나 지목 대상이 비어 있어 모델이 무시 |
| `T2_GATE_REGEN` (claimprov) | turn 3·28·49 3회 — turn 49 *"(이후 무음)"* | 발화·무효 |
| `GB2_NOTICE_BEFORE_TRANSFER` | turn 50 tool-deny 1회 → 모델이 통지 발화 후 이관 | 발화·**준수했으나 사지 못함**(이관 자체는 허용 경로) |
| `WORK-INCOMPLETE` · `T2_TRANSFER_LEAVES_STEPS` | turn 49–50 *"3 item(s) the customer asked about and 0 you have actually acted on"* / `gap qty=3 executed=0` | 발화 · 모델은 «못 하는 것을 손님에게 알리라»는 허용 출구를 택함(msg 52) — 분자 3 은 참(dispute·close·order) |
| `T2_ACTIONREQ` | 13회 · turn 43·47 *"침묵: formalized_target=submit_transaction 가 이 대화 축자에 0회"* | 침묵(정상) — 표적 도구(discoverable) 지목 능력 없음 |
| `T2_UNUSED_GRANT` | 5회(unlock 뒤 미사용 알림) | 발화·전부 곧 사용됨 — 무해 |
| `T2_MATERIAL_GATE` | turn 49 `stop=other_lever(gate)` | 발화 없음(다른 레버에 양보) |
| `T2_SG_DOCS`·`T2_PIN_READ`·`T2_DEMANDED_STEP`·`T2_FOLLOWUP`·`T2_SEARCH_AGENT`·`FAB_STRIP`·`T2_ARG_PRODUCERS`·READ-FIRST·`T2_REQUIRE_DOC_DELIVER`·`T2_SEARCH_REARM` | **0** (trace grep) | 미발화 — 이 sim 의 결정 지점(«이관인가 분쟁인가»)은 어느 레버의 술어에도 없다 |

**t7393_laneC(ours)**: 로컬에 fb/trace/log 없음(results 만) → 발화 표 **작성 불가**(검색 경로: `ls sim_results \| grep t7393_laneC` → results.json.gz 1개). msg 38/40 의 «You're right — I should have executed» 는 리마인더 반응 문면이지만 어느 채널인지 **미확정**.

## 7. 선행 판정과 대조

| 선행 | 판정 | 이번과의 관계 |
|---|---|---|
| `N97_TASKWISE_FORENSIC_2026_08_04.md:16,96-102` (Qwen2.5) | t0 15-miss(`get_debit_cards` 에 user_id 투입→이관) / t1 2-miss(«동결해제·경보해제·분쟁접수·카드재발급 완주») | **달라졌다**: Qwen3.8 base 는 read 오류 0 · 관문 전부 통과 · 결손이 dispute **인자 2칸**으로 좁혀짐 |
| `N97_LEVER_PRESCRIPTIONS_DESIGN_2026_08_04.md:80` | 087/t0 «P1~P6 다 켜도 안 움직이는 6건» | 이번 base 는 P1 표적(관문) 자체가 없어졌다 |
| `BANK_OFFLINE_CEILING_PROXY_FORENSIC_2026_07_15.md:12` | gold 20 다단계 · dispute 완벽해도 close/order 없으면 0 | 역방향 실물: **close/order 완벽한데 dispute 2칸으로 0** |
| `TAU2_FRONTIER_…_2026_07_09.md:239` (gemini25pro) | 20 중 19 exact · dispute enum 2칸(`pin_purchase/card_present_fraud`) 오복사 | 같은 «1콜 2칸» 형이나 칸이 다르다 — 이번엔 enum 은 맞고 **날짜·liability**(ASK 결손 → COMPUTE 종속) |
| `x737_next_run_plan_2026_09_04.md:2792,2854,2895` (relane2b151 = 같은 sim) | model · doc_039 이관 과일반화 · «D 미판정 — 격리 프로브 선행» | **같은 sim, 같은 판정 유지**. 이번에 더한 것: base 팔은 이관하지 않고 완주 → «이관인가 분쟁인가» 분기는 gate 1 팔에서만 나타났다(n=1·n=1 — 인과 서술 금지 [[85]]) |
| `BANKING_FLOOR_LEVER_FIT_2026_07_11.md:131` | REACH m19/20[PROVall,FABTOOL,TIMEFAB,EARLYTR,RPTGAP] | TIMEFAB(시간 날조)의 후신 = 오늘 날짜 대입 · EARLYTR = relane2b151 의 이관 |

## 8. 원인 확정 ([[77]] 4칸)

**x644 t0 · primary = model**
① 주장: gold 변이 7 중 6 일치, 실패는 dispute 1콜의 `discovery_date`(1칸) + 그 종속값 liability(1칸). 모델이 «처음 알아챈 날»을 **묻지 않고** 오늘 날짜를 대입했다.
② 축자: msg 43 배달 doc_031 *"discovery_date … Date customer first noticed the issue"* · msg 59 스키마 *"Date when the user first noticed the issue"* ↔ msg 49 질문 5개에 해당 항목 없음 · msg 49 *"Because you're reporting this within 60 days … $500"* · msg 61 `"discovery_date":"11/14/2025"`(= msg 11 `get_current_time` 값) · user msg 50 *"the text message I got"* 단서 후속 0.
③ 반증 조건: user-sim 대본 §11 이 «에이전트가 경보를 언급만 해도 개방»이었다면 user_sim 귀속이 커진다 — 대본 축자는 *"If agent asks about fraud alert"* 이고 msg 38 은 «I can clear it, but first … confirm these transactions» 라 «물음»이 아니다. 또 gold liability 50 이 discovery 11/06 이 아닌 다른 규칙(예 «statement 기준»)에서 나온다면 COMPUTE 귀속으로 바뀐다 — doc_031 «2 business days of statement» 문구는 그 가능성을 남기나 gold 두 칸이 함께 움직이므로 ASK 결손 해석이 최소 가정이다.
④ 선행 확인: §7(N97·x737·CEILING·FRONTIER·FLOOR_FIT) — «오늘 날짜 대입(TIMEFAB)» 계열은 FLOOR_FIT 07-11 에 이미 태그돼 있었다.

**x644 t0 · secondary = user_sim**: ① 같은 «거래 확인» 요청에 relane2b151 msg 41 은 §11 문자 단서를 열고 x644 msg 39 는 닫았다(n=1 대 n=1). ② 축자 위. ③ 반증: 동일 프롬프트 반복 시 개방률이 0 또는 1 로 고정되면 «분산»이 아니다. ④ x737:2819 «087 §13/§14/§15 는 물으면 열리는 평범한 조건절».

**our_layer = 비인과(x644)**: base 팔. `x818_lanes/t2_base_worker.sh:31` `--gate 0` 동형 런 · 로그 `:9751 T2_LEVER 0 · :9754 T2_GATE 0`. relane2b151 에서는 CLAIMPROV «None: None»(오발화)·WORK-INCOMPLETE(발화·무시)가 있었으나 결정 지점(msg 42 이관 결정)보다 **뒤**(turn 31/50)에 섰다 → 비인과(x737 §1f-12 동일). laneC 의 «You're right — I should have executed»(msg 40)는 우리 리마인더가 유발한 **조기 clear(evergreen)** 의심이 있으나 사이드카 부재로 **UNPROVEN** — our_layer_claims 에 넣지 않는다.

**env**: `file_…_6281` 스키마가 `customer_max_liability_amount` 를 «Reg E 규칙으로 계산하라»고만 하고 기준일(statement vs 거래일 vs 신고일)을 주지 않는다(doc_031 «of statement» ↔ doc_037/036 «report within 2 business days») — 계산 기준의 모호함은 env 몫이나 이번 실패의 1차 원인은 아니다(기준일 자체를 안 물었다).

## 9. 처방 후보 (실행 안 함)

1. **ASK-슬롯 결손 검사(스키마 필드 → 손님-전용 정보)**: `discovery_date` 처럼 스키마 설명이 «customer first noticed/user first noticed» 인 필드는 «오늘 날짜/거래 날짜 복사» 를 차단하고 **손님 발화 축자 접지**를 요구. 술어는 도메인 어휘가 아니라 스키마 문면(«first noticed») 에서 나온다([[59]] 준수). 사정권: 이번 3 sim 중 2(x644·laneC) — 단 relane2b151 형(이관)은 못 산다.
2. **liability 종속값 재계산 리마인더**: `customer_max_liability_amount` 를 쓰기 전 discovery_date 와 함께 2칸을 한 묶음으로 제시(x644 는 discovery 만 고치면 50 이 따라온다).
3. relane2b151 형(이관 과일반화)은 x737 §1f-5 #21 «CODE 05 절차 선언 격리 프로브» 선행 — 이 보고서는 그 판정을 바꾸지 않는다.
4. laneC 의 «push → 조기 clear» 의심은 **fb 사이드카 회수 1건**(리모트 `~/logs/fb_bank_t7393_laneC_*.jsonl`)으로 확정/파기(D8 과 같은 건).
5. **x806 task_087 실측**은 큐 잔여분 회전 후에만 가능 — 승인 없이 재시작·큐 편집 금지([[86]]).
