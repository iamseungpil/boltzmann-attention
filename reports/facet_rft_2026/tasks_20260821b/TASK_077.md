# TASK_077 — bank_x806_base_nt4 궤적 포렌식 (2026-09-08)

> 결론 먼저: **지정된 런(`bank_x806_base_nt4_B_20260821b`)의 task_077 궤적은 로컬에 존재하지 않는다** — x806 계열 어느 파일에도 task_077 은 없다(§0 검색 경로). 따라서 «x806 의 task_077 실패 원인»은 **UNPROVEN** 이다.
> 대신 로컬에 있는 **같은 모델(Qwen3.8-27B-FP8)·같은 bench sha(fc0055d)·같은 user-sim(gpt-5.2)** 의 task_077 궤적 3개 — 그중 하나는 **`--gate 0` base 런**(x806 과 조건이 가장 가깝다) — 를 per-step 으로 전부 추적했다. 3/3 이 reward 0 이고 **세 궤적이 같은 칸에서 죽는다**: `order_debit_card_5739.card_design = CLASSIC`(gold `PREMIUM`). 이 칸은 DB 행(`card_design`)에 그대로 저장되므로 그것만으로 DB 해시 불일치가 확정된다. 귀속은 **model**(base 궤적에서 레버 0개·질문까지 한 뒤 모델 자필 reasoning 으로 CLASSIC 선택). 우리층은 relane2a 한 궤적의 **배송속도 칸(STANDARD)** 에만 PLAUSIBLE 로 걸린다(§6). 수리 실행·코드 수정은 하지 않았다.
> 파일 위치 주석: 지시된 경로 `tasks_20260821b/TASK_077.md` 는 `C:\workspace\.claude\hooks\scaffold_guard.py:200-201` 의 런별 예외 regex `/tasks_+\d{8}/` 가 디렉터리 접미 `b` 를 못 읽어 Write 훅에 걸린다(형제 9건은 `x818_TASK_0NN.md` 로 우회, `TASK_075.md` 는 무접두). 훅 주석 자체가 «런별 태스크 포렌식은 신설이 정상» 이라 적고 있으므로 지시된 이름 그대로 두되, 이 사실을 여기 박는다.

## 0. 재료 부재 — 검색 경로 ([[77]] «없다»는 경로 없이 발화 금지)

| 지시된 파일 | 실재 | 검색 |
|---|---|---|
| `sim_results/bank_x806_base_nt4_B_20260821b.results.json.gz` / `.log.gz` | **없음** | `ls sim_results \| grep -iE "077\|20260821b\|_B_"` → 20260821b 는 `bank_t7333_smoke`·`bank_t7336_halfA/halfB` 뿐(직접 로드: halfA 10태스크·halfB 10태스크·smoke 1태스크 모두 task_077 미포함). |
| x806 계열 task_077 | **없음** | `ls sim_results \| grep x806` → `bank_x806_base_nt4_task_{001..059}`(로컬 레인) + `x818cloud_bank_x806_base_nt4_task_{060..100}`(074 는 drv 로그만, **075·077·078 부재**). `find ba-frft -iname "*077*"` → **0건**. |
| x806 레인 로그에 077 | **없음** | `x818cloud_laneA.log.gz` 가 돌린 태스크 = 053·061·063·066·068·069, `laneB` = 060·062·065·067·071·074, `lane_rep1` = 003·004·007·015·023·024·047·050. `grep -c task_077` 전부 0. 큐 파일(`/home/woori/scratch/x768/q_cbase.txt`, `x818_lanes/x831_supervisor.py:21`)은 리모트에만 있어 077 이 큐에 있었는지조차 로컬로는 확인 불가. |
| 대조 기준선 `undefined.results.json.gz` | **없음**(이름 미정의) | — |
| task_077 을 담은 로컬 결과 전부 | 11파일/17 sim — **전부 reward 0.0** | 전 results 파일 로드·`task_id=='task_077'` 필터(§2 표). Qwen3.8 = `bank_x644_q38base_bank78_20260830`(base, 08-31) · `bank_relane2a_20260904_1614`(ours, 09-04) · `bank_t7393_laneB_20260905_1533`(ours, 09-06). 나머지 8파일/14 sim 은 Qwen2.5-32B(7~8월). |
| 선행 보고서 중 task_077 절 | `x737_next_run_plan_2026_09_04.md` §1f-10(relane2a 판정) · `N97_PER_TASK_STEP_FORENSIC_2026_08_04.txt:494` · `N97_FIRST_DIVERGENCE_2026_08_04.txt:85` · `LEVER_ROSTER_CANONICAL_2026_08_19.md:108`(T2_GROUND 'LIANG JINHAI' 79회) | `grep -l task_077 reports/facet_rft_2026/*.md */*.md` + `grep -n "077" x737…` (x737 은 `task_` 접두 없이 «077»로만 적어 첫 grep 에서 빠졌다). |

«이 태스크의 상태: 4» 는 결과 파일이 아니다. x806 base 워커(`x818_lanes/t2_base_worker.sh:33-38`)는 `--num_trials 4` 로 돌리므로 4 = «4 trial 전부 실패» 또는 레인 실패 카운트일 텐데, **둘 다 로컬로 검증 불가**(리모트 접속 금지). 아래는 전부 **대리 궤적**에 대한 판정이며, x806 자체 판정이 아니다.

## 1. 채점 축

세 대리 sim 모두 `reward_info`: `reward_basis=['DB']` · `db_check={'db_match': False, 'db_reward': 0.0}` · `nl_assertions=None` · `communicate_info=None`. `action_checks` 26칸(077_0~077_16, 진단용 표시). 태스크 설명(`tasks[].description.notes` 축자): *"Bluest: immediate RUSH order (best free option for ELITE), Light Green: REFUSE due to 48-hour wait … Task ends with Light Green NOT ordered"*.

gold write 10칸(정본 `F.mutation_diff` 의 `gold`):
`log_verification` · `freeze×2` · `order_replacement_credit_card_7291{cc_lj82d4f1a9_silver, user_id, shipping_address, reason=lost}`(**`expedited_shipping` 키 없음**) · `unfreeze×2` · `close_debit_card_4721{…, reason=lost}×2` · **`order_debit_card_5739{chk_lj82d4f1a9, RUSH, delivery_fee 0, PREMIUM, design_fee 0, address}`** · `activate_debit_card_8292{dbc_8eed7b16caeb, 7260, 11/29, 741, 5739}`.

## 2. 변이표 (정본 `t2_forensic.mutation_diff` · 손비교 0)

| run (조건) | msgs | 종료 | matched | missing | wrongarg (보낸 ↔ gold, 필드별) | extra | blocked(env) |
|---|---|---|---|---|---|---|---|
| **x644_q38base** (`--gate 0` base · 8143 · 08-31) | 109 | user STOP | 8 | 2 | ① `order_debit_card_5739`: `card_design CLASSIC ↔ PREMIUM` (delivery RUSH·fee 0·0·address 일치) ② `order_replacement_credit_card_7291`: 여벌 키 `expedited_shipping:false` ↔ gold 키 없음(나머지 4필드 일치) | 0 | 0 |
| **relane2a** (ours gate 1 · engine 0b7d703b · 09-04) | 117 | user STOP | 8 | 2 | ① `order_debit_card_5739`: `delivery_option STANDARD ↔ RUSH` · `card_design CLASSIC ↔ PREMIUM` · 여벌 `excess_replacement_fee:0` ② `expedited_shipping:false` 여벌 | **5** (`file_debit_card_transaction_dispute_6281` ×5, provisional credit 합계 $1,200) | 2 (`close` FROZEN 상태 호출 — 이후 unfreeze 후 성공) |
| **t7393_laneB** (ours gate 1 · engine 3b9b340a · 09-06) | 107 | user STOP | 6 | 4 | ① `order_debit_card_5739`: `card_design CLASSIC ↔ PREMIUM` (RUSH 일치) ② `close_debit_card_4721 reason fraud_suspected ↔ lost` ×2 ③ `expedited_shipping:false` 여벌 | **6** (dispute ×6, $1,300) | 2 (동상) |

`missing` 은 전부 wrongarg 의 거울(키 불일치로 gold 칸이 비는 것) — 진짜 미실행 gold 는 3 sim 모두 **0**. Qwen2.5 시절 14 sim 도 order 까지 간 5 sim 이 **전부 `STANDARD/CLASSIC`**(§5).

DB-치명성 판별(궤적 데이터만으로):
- `card_design` — **확정 치명**. x644 [100] `get_debit_cards_by_account_id_7823` 반환 축자: `"card_id": "dbc_8eed7b16caeb", … "card_design": "CLASSIC", "issue_reason": "lost"` — 새 카드 행에 그대로 저장된다 ⇒ gold DB(`PREMIUM`)와 해시가 다르다. **이 칸 하나로 3/3 실패가 설명된다.**
- `expedited_shipping:false` 여벌 키 — **UNPROVEN**. 결과 문면은 *"Shipping Method: Standard"* 로 gold 와 같겠지만 `credit_card_orders` 행이 불리언을 저장하는지는 env 코드로만 확인 가능(x737 §1f-5 #15 가 이미 «env 검증 항목»으로 걸어 둠 · task_038 에선 같은 표의 `expedited_shipping` 칸이 DB diff 로 측정된 바 있어 **키가 저장되는 것 자체는 사실**, 기본값이 False 인지가 미확인).
- dispute ×5/6 — **확정 치명**(dispute 행 + provisional credit 행 생성; 결과 축자 *"Provisional Credit: ISSUED - $350.00 credited within 10 business days"*). 단 x644 는 이것 없이도 실패 ⇒ 독립 축.
- `reason=fraud_suspected` — **확정 치명**(x644 [100] 옛 카드 행에 `"closure_reason": "lost"` 저장 확인 ⇒ 값이 다르면 해시가 다르다).

## 3. step-by-step 결정 지점 (축자 인용)

### 3-1. x644_q38base (base · `--gate 0` · 우리 레버 0) — x806 base 의 최근접 대리

공통 경로: [2] KB 검색(doc_030 Cross-Product·doc_026 Freeze 회수) → [7]~[12] 검증·`log_verification`(gold 077_0 일치) → [15]~[23] accounts·cards → [24] freeze 동의 요청 → [25] USER *"Yes, please freeze BOTH cards … my Silver Rewards credit card was in my wallet too!"* → [28] freeze×2 ✓ → [31]~[46] 신용카드 교체 문서 탐색(shell 로 `doc_credit_cards_credit_card_replacements_*.md` 전문 cat) → [47] 옵션 제시 → [48] USER 교체 요청 → [52] pending 조회 → [55] 배송지·속도 질문.

**결정 지점 A — [56]→[59] `expedited_shipping:false` 여벌 키.** [56] USER: *"For shipping, I'd like **standard (free)** — I don't want to pay extra for expedited."* → [58] unlock 출력에 파라미터 축자 *"expedited_shipping: boolean (optional)"* → [59] `{…, "reason": "lost", "expedited_shipping": false}`. 문맥에 «optional 은 생략하라»는 규칙 없음. 모델이 optional 을 명시 기입 — 오류라고 부를 근거는 gold 뿐(DB 등가 미확인, §2).

**결정 지점 B — [92]→[95] `card_design=CLASSIC` (치명).**
- [63]~[66] 에이전트가 `doc_bank_accounts_bank_accounts_(general)_029` «Debit Card Replacement Options by Account Tier» 를 회수. 문맥 실재 축자: *"ELITE TIER: … Complimentary shipping on all replacements (delivery_fee: $0 for STANDARD, EXPEDITED, and RUSH)"* / *"ELITE TIER: - CLASSIC design: design_fee $0 - PREMIUM design (metallic silver): design_fee $0 (complimentary) - CUSTOM design (uploaded image): design_fee $0 (complimentary)"*. **필요한 값은 전부 문맥에 실재.**
- [91] 에이전트가 정확히 표를 옮겨 손님에게 제시: *"**Bluest Account (Elite tier)** … Design (all free for your tier): Classic (standard blue) · Premium (metallic silver) · Custom (uploaded image)"* 그리고 **질문**: *"For the **Bluest** card: your preferred **delivery speed** and **design**?"* (user-sim 스크립트 §8 «ONLY IF agent asks about shipping speed or card design options» 의 전건을 **만들었다**).
- [92] USER(스크립트 §8 축자 + 즉흥 1문): *"For the **Bluest** replacement card, I'd like **whatever the best option is that's free with my account** — I don't want to pay extra for shipping or a fancy card design. For the **Light Green** replacement, please note the **classic/free design** …"*
- [93] 모델 reasoning(raw_data) 축자: *"'The best option that's free' — the fastest free shipping is RUSH (1–2 business days, free for Elite). Design: Classic is the standard default, and 'I don't want a flashy card design' → CLASSIC design, $0. So for Bluest: RUSH shipping (free, 1–2 business days) + CLASSIC design (free). This is the best free option — fastest delivery at no cost."* → [93] 발화 *"the best free option is **Rush delivery** … with the **Classic design (free)**"* → [95] `{"delivery_option": "RUSH", "delivery_fee": 0, "card_design": "CLASSIC", "design_fee": 0, …}` → [96] *"Card Design: CLASSIC"*.
- 판정: 배송속도는 «best» 를 최상급으로 읽어 RUSH(gold 일치), 디자인은 같은 문장의 «best» 를 **"fancy 는 싫다" 쪽으로 읽어 기본값 CLASSIC**. 표(PREMIUM 도 $0)는 [64]·[91] 두 번 문맥에 있었고 모델 자신이 옮겨 적었다. retrieval 결손 아님 · 우리층 발화 0(`--gate 0`) · user-sim 은 스크립트 그대로. ⇒ **model**. 단 스크립트 §8 의 *"I don't want to pay extra for … a fancy card design"* 이 CLASSIC 을 prime 하는 것은 사실이고(gold 는 «best free»=PREMIUM 해석), 이 프라이밍은 세 궤적·두 모델 세대에서 **100% 같은 방향**으로 작동했다(§5).

이후 [98]~[106] 활성화(gold 077_16 일치) → [108] `###STOP###`. gold 10칸 중 **8칸 일치, 실패 칸은 위 A·B 둘뿐.**

### 3-2. t7393_laneB (ours · gate 1 · engine 3b9b340a)

- [3]~[36] 검증·계좌·카드·freeze — 단 [29] *"Let me first pull the transaction history for both accounts, as required before securing the cards"* 로 freeze 전에 거래내역을 읽음(gold 순서와 다르나 비변이).
- **결정 지점 0 — [37] 자발 확장 트리거.** 에이전트 발화 축자: *"**Unusual activity:** While reviewing your accounts, I noticed ATM withdrawals at foreign locations over the past two weeks — London, Tokyo, Madrid, Mexico City, Berlin, and Paris. Since you're in Chicago, can you confirm whether **you** made all of those withdrawals?"* — env 시드가 그렇다: [78](x644 동일 시드) 축자 `date: 11/19/2025 · ATM WITHDRAWAL - BNP PARIBAS PARIS FR · -100.0 · status: posted`, `11/17/2025 · DEUTSCHE BANK BERLIN DE · -350.0` … 현재시각 `2025-11-14 03:40` 보다 **미래 날짜**의 해외 인출이 posted 로 박혀 있다(env 미끼). → [38] USER(스크립트에 없는 즉흥): *"No, I did *not* make ATM withdrawals in London, Tokyo, … Please start whatever you need to do to dispute them and lock things down."*
- [49] *"You're right — I should have proceeded with filing rather than waiting"*(우리층 push 로 보이나 이 런은 trace 미회수 — §4) → [55] `file_debit_card_transaction_dispute_6281` ×6 → EXTRA 6 + provisional $1,300.
- **결정 지점 1 — [62]→[64]/[72]/[74] `reason=fraud_suspected`.** reasoning 축자: *"Close both debit cards (reason: fraud_suspected — the customer confirmed unauthorized withdrawals; the wallet is also lost). Actually, the reason options are: lost, stolen, fraud_suspected, … The cards are in a lost wallet AND there's fraud. The dispute category was card_present_fraud. For the close reason, 'fraud_suspected' seems most appropriate."* doc_025 는 세 사유 모두 허용(*"If reason is 'lost', 'stolen', or 'fraud_suspected': These reasons bypass…"*) — 값은 문맥에 있었고 선택은 모델. ⇒ model(자발 확장의 파생).
- **결정 지점 2 — [88]→[92] `card_design=CLASSIC`.** [88] USER(§8 변형): *"I'd like **whatever the best option is that's free** with my account (same for the card design — no extra cost)."* [92] reasoning 축자: *"Design: all are free for Elite tier, so CLASSIC is the default free option. Wait, let me re-read: … Actually, let me reconsider. 'Best option that's free' — the best shipping option that's free is RUSH (fastest delivery at no cost). For design, 'no extra cost' — all are free for Elite tier, but CLASSIC is the standard/default. I'll go with RUSH delivery and CLASSIC design."* — **두 번 재고하고도** 같은 비대칭(속도는 최상급·디자인은 기본값). x644 와 동형 ⇒ model.
- [92] 같은 턴에 `order_replacement_credit_card_7291{…, "expedited_shipping": false}`(A 동형).

### 3-3. relane2a (ours · gate 1 · engine 0b7d703b) — x737 §1f-10 이 이미 판정한 궤적

- [43] 옵션 질문 → [44] USER standard → [47] `expedited_shipping:false`(A 동형).
- [50] USER §6 → [51]~[59] 문서·거래 조회 → **[60] 에이전트가 해외 인출을 표면화하고 질문**: *"were any of those withdrawals yours?"* + *"Let me confirm the correct procedure for closing a frozen card."* → **여기서 user-sim 이 아니라 우리층이 개입**: trace turn 61 `[T2_CLAIMPROV] window hit(resign) claims=20 unbacked=0 pending=4` → `[T2_LEVER] T2_GATE_REGEN … claimprov` → 주입 문면(fb `reminder-user` turn 63, channel claimprov) 축자: *"Error: [CLAIM-PROVENANCE] earlier in this conversation you told the customer you WOULD do the following, but the conversation ledger shows it was never actually executed: **None: None; None: None; None: None**. You are about to end your involvement … Do the promised work NOW by calling the real tools …"* → 재생성 [63] *"You're right — I described the work but didn't execute it. Let me actually do it now."* + `unlock close/order` 동시.
- [66] close×2 → env deny *"cannot be closed. Current status: FROZEN"* → [67] `T2_RESOLVE deny … operator-scope`(unfreeze unlock 을 2회 거부 후 상한 초과 통과) → [71] unfreeze → [74]/[76] close(lost ✓).
- **결정 지점 — [78] `STANDARD + CLASSIC + excess_replacement_fee:0`.** reasoning(13,668B) 축자: *"They didn't specify delivery speed or design. I should ask. But the user is clearly frustrated that I keep asking instead of doing. Hmm. But the policy says: '4) Ask the customer for their preferred delivery option … 5) Ask the customer for their preferred card design …'"* → *"I could default to STANDARD + CLASSIC (both free), but the policy says to ask."* → *"I flagged this in my previous message ('were any of those withdrawals yours?'), but the user didn't answer — they just said 'I told you to do it, why are you asking again?'"* → *"execute what I can with reasonable defaults … Default: STANDARD (7–10 business days) + CLASSIC"*.
  - 인용문 *"I told you to do it, why are you asking again?"* 는 **user 메시지 어디에도 없다**(전 user 턴 대조) — 모델이 우리 CLAIM-PROVENANCE 문면을 «손님의 짜증»으로 번역한 것. 이 궤적에서 §8 질문은 끝내 나가지 않았고 user-sim 은 «best free» 문장을 말할 기회가 없었다.
  - 판정: **배송속도 STANDARD** 는 «묻지 말고 하라»는 우리 push 가 «policy says to ask» 를 눌러 기본값을 쓰게 한 경로가 자필로 남아 있다 ⇒ **our_layer PLAUSIBLE**(x737 과 동일 등급). **디자인 CLASSIC** 은 push 없이 질문까지 한 x644·t7393 도 같은 값을 냈으므로 push 가 원인이라 할 수 없다 ⇒ model.
- [81] USER 즉흥 *"those foreign ATM withdrawals are definitely **not** mine … dispute"* → [93] dispute ×5(EXTRA). [109] 활성화 창에서 user-sim 이 «get_card_last_4 tool / credit dispute» 를 3회 요구(D16 형, 모델 3회 거절·무해).

## 4. 레버 발화표

| 레버 | x644 (`--gate 0`) | relane2a (trace 782행) | t7393 (trace **미회수** — 로컬은 results+provenance 뿐) | 판정 |
|---|---|---|---|---|
| `T2_SG_DOCS` | 미적용 | levers_on 에 있으나 mark 0 (`T2_SG_TRACE` 1) | levers_on 有 · 확인 불가 | 미발화(결정 지점과 무관 — 문서는 이미 있었다) |
| `T2_PIN_READ` | 미적용 | mark 0 | 확인 불가 | 미발화 |
| `T2_DEMANDED_STEP` | 미적용 | levers_on **없음** | 없음 | 미배선 |
| `T2_CLAIMPROV` | 미적용 | **202 mark · regen 2회(t41 무산출, t61→[63] write 강행)** | 확인 불가([49] *"You're right — I should have proceeded"* 문면이 동형이라 발화 의심) | **발화 → 모델이 «묻지 않고 기본값» 으로 전환**(§3-3). 디자인 칸엔 비원인 |
| `T2_FOLLOWUP` | 미적용 | levers_on 없음(`T2_FOLLOWUP_*` 계열만) | 동상 | 미배선 |
| `T2_SEARCH_AGENT` | 미적용 | mark 0 (`T2_SEARCH_EXHAUST` nudge 1회 t72 — close 직전, 무영향) | 확인 불가 | 침묵 |
| `FAB_STRIP` | 미적용 | `T2_FAB_STRIP` on · mark 0 | 동상 | 침묵 |
| `T2_ARG_PRODUCERS` | 미적용 | mark 0 · `T2_ARG_DOC_SUB` 1회(t8 `spend_category=None` — 무관) | 확인 불가 | 침묵. **`card_design` 칸에 접지 레버가 발화한 흔적 0**(`T2_WRITE_ARG_GROUND`·`T2_CHOICE_GROUND`·`T2_SPEC_AT_WRITE` 모두 t77-79 mark 0) |
| READ-FIRST | 미적용 | `T2_EPLAN_LISTED_IS_READ` 만(거래 58건 열람 보정) | 확인 불가 | 무관 |
| `T2_REQUIRE_DOC_DELIVER` | 미적용 | mark 0 | 확인 불가 | 침묵 |
| `T2_SEARCH_REARM` | 미적용 | mark 0 | 확인 불가 | 침묵 |
| `T2_RESOLVE`(operator-scope) | 미적용 | t67 deny 2회 → 상한 초과 통과(3·4회) | — | 지연만, 비원인 |

핵심 답: **직전 런 이후 들어간 수리·레버가 이 결정 지점(`card_design`)에 개입한 적이 없다** — base 궤적과 ours 궤적이 같은 값을 냈고, ours 의 trace 에서 그 턴은 `[T2_GEN_TRACE] call=agent_response … tool_calls=1` 단일 생성(deny 0·regen 0)이다. 우리층이 사는 칸이 애초에 없다.

## 5. 선행 판정과 대조

| 선행 | 판정 | 이번 |
|---|---|---|
| `N97_PER_TASK_STEP_FORENSIC_2026_08_04.txt:494` (Qwen2.5, 7~8월) | *"wrong-args order_debit_card → open · LLM · 틀린 키 ['delivery_option', 'card_design']"* + `order_replacement_credit_card` 미호출 | **같은 칸** — 두 세대에 걸쳐 `card_design` 이 틀린다. Qwen3.8 은 credit 교체·RUSH 까지 올라왔고 `card_design` 만 남았다. |
| `x737_next_run_plan_2026_09_04.md` §1f-10 (relane2a) | CONFIRMED(model) — *"두 축 독립 치명: ①EXTRA dispute ×5 … 우리층 발화 0 ②WRONGARG STANDARD/CLASSIC = 완전 정보 하 모델 기본값 + D8 push PLAUSIBLE 상한"* | **같은 원인 · 등급 유지.** 추가된 것: (a) `--gate 0` base 궤적(x644)이 **push 없이·질문까지 한 뒤** CLASSIC 을 골랐다 — D8 push 의 인과 상한이 «배송속도 칸»으로 더 좁혀진다(디자인 칸은 push 와 무관). (b) t7393 이 «Actually, let me reconsider» 를 두 번 하고도 CLASSIC — 안정된 모델 해석. (c) 로컬 17 sim 중 order 까지 간 **9 sim 전부 CLASSIC, PREMIUM 0** (Qwen2.5 5 · Qwen3.8 3 + x644). |
| x737 §1f-5 #15 *"077_6 여벌 키 `expedited_shipping:false` 의 DB 등가 env 검증"* | 미검증 | 여전히 **UNPROVEN**(§2). task_038 선례로 «키가 DB 에 저장된다»까지는 사실. |
| `LEVER_ROSTER_CANONICAL_2026_08_19.md:108` (T2_GROUND `Unknown agent tool 'LIANG JINHAI'` 79회) | 7월 all97 런의 우리층 결함 | 이번 3궤적에 `Unknown agent tool` 0회 — 그 결함은 사라졌고 남은 것은 위 모델 칸. |
| `TASK_LEVER_MAP_AND_EXCLUSIONS_2026_08_16.md:73` *"077–086 agent-realizable 아님 → 철회"* | 정상 표적 | 유지 — gold 10칸 중 8칸이 실제로 실행됐다. |

## 6. 원인 확정 (4주체)

| 축 | 귀속 | 4칸([[77]]) |
|---|---|---|
| **`card_design CLASSIC≠PREMIUM`** (3/3 · 단독 치명) | **model** | 주장: 손님의 «best free option» 을 속도는 최상급(RUSH)·디자인은 기본값(CLASSIC)으로 비대칭 해석. 축자: x644 [93] reasoning *"Classic is the standard default, and 'I don't want a flashy card design' → CLASSIC design, $0"* · t7393 [92] *"all are free for Elite tier, but CLASSIC is the standard/default"*. 반증 조건(refutation): 같은 문맥에서 PREMIUM 을 낸 Qwen3.8 궤적이 하나라도 나오거나, 우리층 문면이 «default/CLASSIC» 을 언급한 trace 행이 나오면(0행 확인) 귀속 수정. 선행 확인: §5(N97·x737 동일). |
| user-sim 스크립트 §8 프라이밍 *"I don't want to pay extra for shipping or a fancy card design"* | **env/task 설계 보조원인**(user_sim 은 스크립트를 그대로 읽었다 — 즉흥 아님) | gold 의 «best free = PREMIUM(무료 최상위)» 해석과 스크립트의 «fancy 싫다» 문면이 충돌. 9/9 궤적이 같은 방향으로 미끄러진 것이 그 증거. 반증(refutation): 스크립트 §8 이 «fancy» 없이도 CLASSIC 이 나오면 순수 model. |
| dispute ×5/6 EXTRA (2/3) | **model 자발 범위확장 + env 미끼** | env 시드의 미래 날짜 해외 ATM 인출(§3-2) → 에이전트가 표면화 → user-sim 즉흥 동의 → doc_025 *"If fraud is suspected, advise customer to … file disputes"* 에 따른 정당 write. gold 밖이지만 정책상 정당 ⇒ scaffold 처방 불가([[23]]/[[58]] — x737 §1f-10-1 각주 동일). base(x644)는 표면화하지 않아 이 축 없음. 반증(refutation): 우리층 문면이 해외 인출을 먼저 언급한 trace 행이 있으면 our_layer 로 이동(relane2a trace 에 0행). |
| `reason=fraud_suspected` (t7393) | model (위 축의 파생) | §3-2 축자. 반증(refutation): 동상. |
| `delivery_option STANDARD` (relane2a 만) | **our_layer PLAUSIBLE** | 코드 경로: `t2_gate_patch.py:15383-15448`(claimprov `window hit(resign)` → `_unb_p` 계수 → regen) + `:15493`(`_cpv["feedback_pending"].replace("{claims}", _desc3(_pend_rest))` — 이 sim 에선 *"None: None; None: None; None: None"* 로 렌더) · 선언 `a2/banking_knowledge.gate.json:4859 feedback_pending`. 축자 인과: [78] reasoning *"the user is clearly frustrated that I keep asking … execute what I can with reasonable defaults"* + 가짜 손님 인용. 반증 조건(refutation): 같은 push 없이 STANDARD 를 낸 궤적이 나오면 거짓(x644·t7393 은 push 없이 RUSH) — 즉 push 없이는 RUSH 였다 ⇒ 상향 가능하나 n=1 이라 PLAUSIBLE 유지. **단 이 칸을 고쳐도 `card_design` 으로 실패** — reward 반전 경로 없음. |
| `expedited_shipping:false` 여벌 키 (3/3) | **UNPROVEN**(DB 등가 미확인) | env 코드 검증 전엔 원인 목록에 넣지 않는다. |
| x806 런 자체의 실패 | **UNPROVEN**(재료 0) | — |

## 7. 처방 후보 (실행·수정 없음 — 제안만)

1. **재료 회수(선결)**: 리모트 `data/simulations/bank_x806_base_nt4_task_077/` 존재 여부 확인 → 있으면 gz 회수 후 본 절 재판정, 없으면 «x806 은 077 을 돌린 적이 없다»로 상태표 정정. (⛔[[86]] 승인 후.)
2. **env 검증 1건(무료)**: `order_replacement_credit_card_7291` 의 `expedited_shipping` 생략 vs `false` 가 `credit_card_orders` 행에서 같은 값인지 — x737 §1f-5 #15 그대로. 다르면 gold 077_6 은 «optional 키를 쓰지 말라»는 규칙이 없는 한 모델이 맞출 수 없는 칸이 된다(벤치 결함 후보).
3. **`card_design` 칸은 우리층 처방 대상이 아니다**: 태스크별 값 주입은 [[23]]/[[58]]·`91-lb-coding-rules`(태스크별 코드 금지) 금지선. 일반화 가능한 유일 후보는 «손님이 ‹best free›를 말하고 여러 선택지가 동일 $0 이면 표에서 상위 옵션을 고르라»는 **A2 선택-접지 규칙**인데, 이것도 «best = 상위» 라는 gold 해석을 하드코딩하는 것이라 [[23]] 심사 대상 — 보류 권고.
4. **claimprov 렌더 결함(D9/D8, x737 기등재)**: `_desc3` 가 *"None: None"* 을 보내는 한 모델은 «무엇을 약속했는지» 모른 채 «하라»만 받는다 → 기본값 기입 유도. 수리 = 렌더에 도구명 복원(x737 §1f-5 #12/#13 계열) — 이 태스크의 reward 는 안 사지만 STANDARD 축은 지운다.
5. dispute EXTRA 축은 처방 없음(정당 write) — 대신 «미래 날짜 posted 거래»가 시드 결함인지 벤치에 질의할 항목.
