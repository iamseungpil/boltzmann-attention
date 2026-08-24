# TASK_093 — t7348 halfA per-step 포렌식 (SAVINGS APY DISCREPANCY)

- 런: `bank_t7348_halfA_20260824` · agent = `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8` · user-sim = `openrouter/openai/gpt-5.2` · nt=2 · 로그 헤더 축자 `Save: bank_t7348_halfA_20260824`
- 성적: **trial 0 = 0.0 (seed 626729 · 49 msgs) · trial 1 = 1.0 (seed 373753 · 75 msgs)** ⇒ **1/2**
- 대조(직전 런·같은 계열) `bank_t7346_halfA_20260822`: **0/2** ⇒ 093 은 t7348 에서 **+1**
  ⚠단, **성분별로는 순증이 아니다** — t7346 은 **양 trial 모두 `get_correct_savings_apy -> 4.275`** 를 얻고 보고서를 MATCHED 시켰으며 남은 결손은 크레딧 write 미실행 하나뿐이었다. t7348 trial 0 은 **그 4.275 를 잃었다**(→ 4.0). 즉 **크레딧 write 는 사고 APY 는 판 교환**이다([[70]]).
- 로그 sim 태그 매핑: `s626729` = **trial 0** · `s373753` = **trial 1** (상태줄 `task_093.0(29s)` 가 s626729 라인과 인터리브 · 메시지 수 49/75 와 로그 라인 수 312/518 일치)
- 종료: 양 trial `###STOP###`(user_stop)

> **한 줄 결론**: trial 0 의 실패는 **이중으로 결정**돼 있다. ⑴ `expected_apy` 4.275→**4.0**: 모델이 방금 읽은 계좌 레코드를 `customer_products="No credit card accounts"` 로 **전사 누락**했고, 바로 그 누락을 막으려고 2026-08-22 에 저작된 우리 수리 **R3 `ref_from_outputs`(`customer_products`)가 이 런 전체에서 0회 발화**했다(死배선·producer 키 불일치). ⑵ `actual_apy` 4.0→**2.775**: 모델이 KB 도구문서의 **파라미터 예시값**(`e.g., 2.775 for 2.775%`)을 그대로 인자로 썼고, 우리 관문(`corpus:["ledger"]`)이 **KB 검색 출력까지 원장으로 세기 때문에** 그 예시값을 실재로 인정해 통과시켰다. ⑵ 는 **반사실이 같은 sim 안에 있다** — trial 1 에서 같은 관문이 `actual_apy` 를 드롭하자 모델이 두 턴 뒤 480×12/144000=4.00 을 **스스로 유도**했다.

---

## §1 채점축 (C583ⓖ · `sim['reward_info']` 직독)

```
trial 0: reward=0.0 · reward_basis=['DB'] · db_check={'db_match': False, 'db_reward': 0.0}
trial 1: reward=1.0 · reward_basis=['DB'] · db_check={'db_match': True,  'db_reward': 1.0}
env_assertions=[] (양 trial)
```

**DB-해시 축**이다. `action_checks` 는 진단 보조로만 읽는다([[69]]).
⚠ 주의: `action_checks[093_6]` 은 **양 trial 모두 `action_match=false`** 다(trial 1 은 reward 1.0). ACTION 축으로 읽었다면 표가 거짓말을 했을 자리다.

gold 변이 3건(`t2_forensic.gold_mutations` · unlock 은 DB 무변):

| aid | 도구 | 인자 |
|---|---|---|
| 093_0 | `log_verification` | Somchai Prasert / sp93k4m7n2 / … / `2025-11-14 03:40:00 EST` |
| 093_6 | `apply_savings_account_credit_6831` | `{sav_sp93k4m7n2_silver, **33**, interest_correction}` |
| 093_8 | `submit_interest_discrepancy_report_7294` | `{sav_sp93k4m7n2_silver, sp93k4m7n2, expected_apy **4.275**, actual_apy **4**, amount_difference **33**}` |

정답 산식(진단용): expected 4.275 = 4.0(잔액 144,000 = tier 2) + 0.25(Green checking→Silver 링크) + 0.025(relationship) · actual 4.0 = 480×12/144000 · 차액 = 144000×0.275%/12 = **33.00**

---

## §2 변이표 — 정본 `t2_forensic.mutation_diff` 만 (손 비교기 0 · C583ⓐ)

```python
sys.path.insert(0,'.'); import t2_forensic as F
mut = F.mutating_tools(); m = F.mutation_diff(sim, mut)
```

### trial 0 — `missing 2 · wrongarg 2 · extra 0 · dup 0 · blocked 0 · matched 1` (`clean=false`)

| 칸 | 내용 |
|---|---|
| MATCHED | `log_verification{…}` @ msg 23 |
| MISSING | `apply_savings_account_credit_6831{sav_sp93k4m7n2_silver, **33**, interest_correction}` (093_6) |
| MISSING | `submit_interest_discrepancy_report_7294{…, **4.275 / 4 / 33**}` (093_8) |
| WRONGARG | `submit_interest_discrepancy_report_7294{…, **4 / 2.775 / 147**}` @ msg 39 (`ok=True`) |
| WRONGARG | `apply_savings_account_credit_6831{…, **147**, interest_correction}` @ msg 45 (`ok=True`) |
| BLOCKED | **0** — 우리 게이트도 env 도 이 sim 의 어떤 write 도 막지 않았다 |

**WRONGARG 필드별 대조 — `submit_interest_discrepancy_report_7294` (msg 39)**

| 필드 | 보낸 값 | gold | 판정 | 값의 출처 |
|---|---|---|---|---|
| `account_id` | `sav_sp93k4m7n2_silver` | 동일 | ✅ | 계좌 레코드(msg 28) |
| `user_id` | `sp93k4m7n2` | 동일 | ✅ | 사용자 레코드(msg 12) |
| `expected_apy` | **4.0** | **4.275** | ❌ (−0.275pp) | **우리 도구 반환**(msg 36) |
| `actual_apy` | **2.775** | **4** | ❌ | **KB 도구문서의 파라미터 예시값**(msg 3/4) |
| `amount_difference` | **147.0** | **33** | ❌ | **우리 도구 반환**(msg 38) |

**WRONGARG 필드별 대조 — `apply_savings_account_credit_6831` (msg 45)**

| 필드 | 보낸 값 | gold | 판정 |
|---|---|---|---|
| `account_id` | `sav_sp93k4m7n2_silver` | 동일 | ✅ |
| `credit_type` | `interest_correction` | 동일 | ✅ |
| `amount` | **147.0** | **33** | ❌ (우리 도구 반환의 승계) |

⇒ **식별자·enum 4필드는 전부 일치**하고 어긋난 것은 **수치 3필드**뿐이다. 그 3필드 중 2개는 **우리 도구가 낸 값**이고, 나머지 1개(`actual_apy`)는 우리 관문이 **통과시킨** 값이다.

### trial 1 — `missing 0 · wrongarg 0 · extra 0 · dup 0 · blocked 3 · matched 3` (`clean=true`)

| 칸 | 내용 |
|---|---|
| MATCHED | `log_verification` @ 23 · `submit_interest_discrepancy_report_7294{…4.275/4/33}` @ 63 · `apply_savings_account_credit_6831{…33…}` @ 71 |
| BLOCKED(env) ×3 | msg 41 `correction_amount` 미지원 키 · msg 43 `period_start` 미지원 키 · msg 45 `user_id`/`amount_difference` 누락 — **전부 env 스키마 거부**(우리 게이트 아님) |

---

## §3 ★스텝 표 — trial 0 (seed 626729 · reward 0.0)  ※이 문서의 본체

열: step(messages 인덱스) | role | what | actor | kind | code_path(our_layer 만) | evidence(축자) | consequence | reward_causal

| step | role | what | actor | kind | code_path | evidence (축자) | consequence | reward_causal |
|---|---|---|---|---|---|---|---|---|
| **2** | assistant | `KB_search_bm25{"submitting interest discrepancy reports"}` + `unlock(submit_interest_discrepancy_report_7294)` — 절차 문서를 첫 턴에 확보 | model | 발견(성공) | — | `KB_search_bm25 {"query": "submitting interest discrepancy reports"}` | 절차 8단계·파라미터 정의가 문맥에 실림(긍정) | 무관 |
| **3/4** | tool(env) | 그 문서·도구 스펙이 **파라미터 예시값을 축자로 싣는다** | env | 예시-포이즈닝(재료) | — | `- expected_apy (number): The APY percentage the customer should have received (**e.g., 2.775 for 2.775%**)` | 원장에 `2.775` 가 심어진다 → step 37 의 인자 출처 | **준강** |
| **turn 2** | our_layer | 검색-에이전트가 축 `savings_accounts` 92편에서 **클래스를 'Gold Account' 로 결정**해 배달 (093 은 Silver) | our_layer | 오발화(재료 오배달) | `t2_search.py:740` `print("[T2_DOCDECIDE] → %r")` (결정 자체는 그 위 서브) | `[T2_DOCDECIDE] → 'Gold Account'` · `[T2_SEARCH_AGENT] group=savings_accounts · 문서 92 … turn=2` | Silver/Green 페어링 문서가 **turn 47 까지 배달 안 됨**(`T2_SEARCH_REARM … turn=47` = 모든 write 이후). 모델은 자력 검색으로 **business** 문서를 집는다(step 33) | 중 |
| **9→18** | assistant/our_layer | `verify_identity` NOT_VERIFIED → `get_user_information_by_email` → VERIFIED | our_layer(피드백)+model | 배관(정상) | — | `NOT_VERIFIED — the account record has not been fetched yet … the next step is the lookup.` → `VERIFIED — 4 identity values match` | 신원 확립 | 약 |
| **19** | our_layer | 계좌 접근 시도를 `GB1_VERIFY_BEFORE_ACCOUNT_ACCESS` 로 차단 (log_verification 이전) | our_layer | 게이트(정상) | `t2_gate_patch.py` GB1 | `[GB1_VERIFY_BEFORE_ACCOUNT_ACCESS] blocked by policy gate: customer identity verified and logged...` | 손님이 다음 턴에 접근 허가를 줌 → 순서 교정 | 약 |
| **23** | assistant | `log_verification{… time_verified:"2025-11-14 03:40:00 EST"}` | model | write(성공) | — | `Verification logged successfully. - User: Somchai Prasert (ID: sp93k4m7n2)` | **gold 093_0 MATCHED** | 무관(양성) |
| **27/28** | assistant/tool | `call_discoverable_agent_tool(get_all_user_accounts_by_user_id_3847)` — 레코드 확보 | model | 발견(성공) | — | `1. Record ID: **sav_sp93k4m7n2_silver** … level: **Silver Account** … current_holdings: **144000.00**` / `2. Record ID: **chk_sp93k4m7n2_green** … level: **Green Account**` / `Credit Card Accounts: No credit card accounts found.` | **4.275 를 유도할 재료가 이 시점에 전부 실재**(Green checking 포함). trial 1 의 msg 28 과 **바이트 동일** | 무관(양성) |
| **33** | assistant | `KB_search_dense{"Silver Savings Account APY components"}` → **business** 문서 `doc_business_savings_accounts_silver_plus_saver_001` 회수 | model | 재료 오선택 | — | 회수 문서 축자: `### Relationship APY Bonuses / If you hold an eligible **Business Rewards Card**, an APY bonus is added…` | 부스트를 **카드에만** 결부시키는 틀이 잡힌다 | 중 |
| **35** | assistant | ★**분기점** — `get_correct_savings_apy{savings_account_type:"Silver Account", customer_products:"**No credit card accounts**", current_balance:"144000.00"}` | model | 전사-누락(빼기 실패·[[63]]) | — | 같은 턴 산문 축자: *"since you **do not have any credit card accounts** associated with your user ID, we will consider **only the base APY**."* | 계좌 레코드 2행 중 **checking 행이 인자에서 소거**됨 | **강** |
| **35** | our_layer | R3 `ref_from_outputs.customer_products`(모델 전사를 **도구 출력 원문**으로 대체하는 수리)가 **발화하지 않음** | our_layer | 死배선 | `t2_scaffold_get.py:722` `_raw = (_evidence_ctx(orch).get("__tool_outputs_raw") or {})` + `t2_scaffold_get.py:1763-1767,1782` (`outs[nm]`·`nm=id2name[...]` = **외곽 도구명**) ↔ 선언 `a2/banking_knowledge.specific.json → get_correct_savings_apy.isolate.ref_from_outputs.customer_products.producer_contains=["accounts"]` | 로그 **`[T2_SG_REFRAW]` 0줄**(양 trial). 런 전체 25,496줄 중 REFRAW **1줄**뿐이고 그것도 `task_094 … get_interest_correction.account_records`(=`get_credit_card_accounts_by_user` 오매칭). 실측 outs 키: `['KB_search_bm25','KB_search_dense','**call_discoverable_agent_tool**','get_correct_savings_apy',…]` — `"accounts"` 를 포함하는 키 **0개** | 계좌 read 가 `call_discoverable_agent_tool` 을 거치므로 needle 이 영원히 안 맞는다 → **모델 전사가 그대로 서브로 감** | **강** |
| **35** | our_layer | 그 전사를 **REFERENCE 로 삼아** 문서-클래스 선택 서브가 돌고, **`['silver_account']` 9편**만 배달 | our_layer | 재료 결손(전달 실패) | `t2_scaffold_get.py:593-626` — `pick = ("… REFERENCE names an account and the customer's products …" % json.dumps(ref…))` · `ref` 는 `t2_scaffold_get.py:704` 의 모델 인자 | `[T2_SG_DOCS] get_correct_savings_apy: 클래스 **['silver_account']** · 문서 **9편** · 10562자 전달(검색 0)` ↔ trial 1 `['green_account_(checking)', 'silver_account']` · **12편** · 13619자 | Green→Silver 페어링 문서가 **서브 문맥에 없다** ⇒ checking +0.25 를 원리상 찾을 수 없음 | **강** |
| **35** | our_layer | 관문1(`_ground_operands`)이 1건 반려 → 서브 재시도 → 최종 **`sub=1 rows`**(base 만) | our_layer | 억제 되먹임 | `t2_scaffold_get.py:394-402` (`src_ok = … any(ns in nc …)` 인용 **전량** substring) | `[T2_SG_ISOLATE] fetch …: ground-피드백 1건 → 서브 재시도(1라운드)` → `[T2_SG_ISOLATE] operand-size …components: **sub=1 rows**` ↔ trial 1 `sub=3 rows` | relationship +0.025 도 소실(t7336 §2.4 와 동형 기전) | 준강 |
| **36** | tool(our) | `get_correct_savings_apy -> **4.0**` | our_layer | 오답 반환 | 위 3행의 합성 | `Correct savings APY under the stacking policy (…): **4.0%**. Compare this to the APY the system actually applied` | **gold 093_8 의 `expected_apy=4.275` 상실 확정** | **강** |
| **37** | assistant | ★`get_interest_correction{expected_apy:"4.0", **actual_apy:"2.775"**, principal:"144000.00", …}` — 480×12/144000 을 **유도하지 않고** 문서 예시값을 인자로 씀 | model | 날조(예시값 복사) | — | A2 param 문면은 유도를 명시 지시한다: *"Derive it from the latest MONTHLY INTEREST CREDIT …: monthly credit amount x 12 / principal x 100"* — 모델은 그 턴 산문에서 `actual_apy` 를 **한 번도 계산하지 않는다** | `actual_apy` 가 오염 | **강** |
| **38** | our_layer | 관문1 이 `period_start` 만 드롭하고 **`actual_apy=2.775` 는 통과**시킴 | our_layer | 관문 오통과 | `t2_scaffold_get.py:227-229` `_corpus_texts(…, "ledger")` = **에이전트 도구 출력 전량**(KB 검색 포함) + `t2_scaffold_get.py:_val_grounded` 수치-토큰 매칭 ↔ 선언 `get_interest_correction.ground.scalar_fields[actual_apy].corpus=["ledger"]` | `[GROUNDING WARNING] **1** input value(s) … dropped: period_start=10/01/2025` — `actual_apy` 는 경고에 없다. `2.775` 실재 위치 = **msg 3(`KB_search_bm25`)·msg 4(`unlock_…`) 의 파라미터 예시** | 오염값이 계산에 들어가 `= **147.0**` 산출 · 반환문이 *"Use this as the credit amount and the amount_difference, and use these APY values in the discrepancy report"* 로 **종단 write 인자 지정** | **강** |
| **39** | assistant | `submit_interest_discrepancy_report_7294{…, 4.0 / 2.775 / 147.0}` — 우리 반환문을 그대로 따름 | model | write(WRONGARG) | — | `Expected APY: 4.0% - Actual APY: 2.775% - Amount Difference: $147.00 - Status: PENDING_REVIEW` | **gold 093_8 상실** · `T2_WRITE_EVIDENCE` **0회 발화**(값이 실제로 우리 도구 출력에 있으므로 통과) | **강** |
| **43/45** | assistant | `unlock(apply_savings_account_credit_6831)` → `call_discoverable_agent_tool{… amount: 147.0 …}` | model | write(WRONGARG) | — | `Credit applied successfully! … Amount: **$147.00** - Previous Balance: $144000.00 - New Balance: $144147.00` | **gold 093_6 상실** · DB 해시 불일치 확정 | **강** |
| **48** | user_sim | `###STOP###` | user_sim | 종료 | — | `Okay, thank you for looking into this. I appreciate it. ###STOP###` | 회복 기회 소멸 | 무관 |

---

## §4 ★스텝 표 — trial 1 (seed 373753 · reward 1.0)  ※대조군

| step | role | what | actor | kind | code_path | evidence (축자) | consequence | reward_causal |
|---|---|---|---|---|---|---|---|---|
| **2/3** | assistant/env | trial 0 과 동일 검색·동일 문서(예시 `2.775` 포함) | env | 예시-포이즈닝 | — | `(e.g., 2.775 for 2.775%)` | 같은 포이즌이 원장에 있으나 **소비되지 않는다** | 무관 |
| **turn 2** | our_layer | 동일하게 `[T2_DOCDECIDE] → 'Gold Account'` 오배달 | our_layer | 오발화 | `t2_search.py:740` | `[T2_SEARCH_REARM] … turn=**73**` (모든 write 이후) | 두 trial 공통 ⇒ **분기 요인 아님** | 약 |
| **12→15** | assistant/user_sim | 모델이 user_id/account_id 를 손님에게 요구 → 손님이 모른다고 답하고 4-필드 제시 | model/user_sim | 부하(우회 턴) | — | `I don't have the user ID or the savings account ID handy right now.` | 26턴 소모(trial 0 대비 +) | 무관 |
| **16→24** | assistant/our_layer | `verify_identity` NOT_VERIFIED → 이름 조회 → VERIFIED → `log_verification` | our_layer(피드백)+model | 배관(정상) | — | `VERIFIED — 4 identity values match the account record` | **gold 093_0 MATCHED** | 무관(양성) |
| **27/28** | tool | 계좌 레코드 — **trial 0 msg 28 과 바이트 동일** | env | 재료 | — | `level: Silver Account … 144000.00` / `level: Green Account` | 두 trial 의 입력이 동일함의 증명 | 무관 |
| **29** | assistant | ★**분기점** — `get_correct_savings_apy{…, customer_products:"**Green Account (checking), Silver Account (savings)**", current_balance:"144000.00"}` | model | 전사(성공) | — | 인자 축자(위) — 레코드의 `level` 표기를 **그대로** 옮겼다 | REFERENCE 에 checking 이 실린다 | **강** |
| **29** | our_layer | 클래스 선택 서브가 **2개 클래스·12편** 배달 | our_layer | 전달(성공) | `t2_scaffold_get.py:593-626` | `[T2_SG_DOCS] … 클래스 **['green_account_(checking)', 'silver_account']** · 문서 **12편** · 13619자` · 이어 `operand-size …: **sub=3 rows**` · `1라운드`(ground 반려 0) | base+checking+relationship 3성분 확보 | **강** |
| **30** | tool(our) | `get_correct_savings_apy -> **4.275**` | our_layer | 정답 반환 | — | `Correct savings APY under the stacking policy (…): **4.275%**` | `expected_apy` 확보 | **강** |
| **35/37** | assistant/our_layer | 모델이 `actual_apy:"480.0 * 12 / 144000.0 * 100"`(수식 문자열)을 2회 전송 → 관문1이 **드롭** | model(오형식)+our_layer(정상 deny) | 관문(정상) | `t2_scaffold_get.py:_ground_operands` scalar_fields | `[GROUNDING WARNING] **2** input value(s) … dropped: **actual_apy=480.0 * 12 / 144000.0 * 100** …; period_start=…` · 반환 `Correction amount = … = (could not compute — check your arguments)` | ★**trial 0 의 반사실이 여기 있다** — 같은 관문이 오염된 `actual_apy` 를 드롭하자 모델이 **스스로 유도**로 전환한다 | **강** |
| **39** | assistant | `actual_apy` 를 **정확히 유도** | model | formalize(성공) | — | 산문 축자: `\[ \text{actual\_apy} = \left( \frac{480.00 \times 12}{144000.00} \right) \times 100 \]` → `= 4.00\%` · 호출 `actual_apy:"4.00"` | `actual_apy` 확보 | **강** |
| **40** | tool(our) | `Correction amount = … = **33.0**` (반올림 적용) | our_layer | 계산(성공) | `t2_scaffold_get.py:~2380` `result_round` | `[T2_SG_ROUND] get_interest_correction: **33.00000000000004 -> 33.0** (자릿수 2)` (9회) | 33 확보 | **강** |
| **41/43/45** | assistant/env | 보고서 write 3회 시도가 **env 스키마 거부** (`correction_amount` → `period_start` → 필수 2개 누락) | model | 스키마 오인자 | — | `Error: Invalid arguments: … got an unexpected keyword argument 'correction_amount'` 등 3종 | 6턴 소모 · BLOCKED 3건 | 중 |
| **47→61** | our_layer | ★`T2_WRITE_EVIDENCE` 가 **정답 write 를 8회 반려** — `"33.00"` ↔ 도구 출력 `33.0` 의 **바이트 비교** | our_layer | 과차단(자기 표현 오차) | `t2_gate_patch.py:1402` `if not found and **str(idv) in c** and all(t in c for t in tokens)` (수치 정규화 없음 · 같은 리포의 `t2_scaffold_get._val_grounded` 는 `_as_float`+`_nums_in` 로 **형식-불문** 매칭 = **두 게이트의 술어가 불일치**) ↔ 선언 `a2/banking_knowledge.specific.json → write_evidence_specs[submit_interest_discrepancy_report].id_key="amount_difference"` | deny 축자: `Error: [WRITE-EVIDENCE] the amount_difference (**33.00**) does not appear in any get_interest_correction tool output in this conversation.` · 도구 출력 축자: `= **33.0**. Use this as the credit amount` · 모델은 **동일 인자로 8회 재호출**(msg 49~61 산문 sha 동일 패턴) | **16 메시지 소모** · reward 를 죽일 뻔함 | **준강** |
| **62** | our_layer | deny 상한에 걸려 **게이트가 포기** | our_layer | 상한 | `t2_gate_patch.py:10504` `print("[T2_WRITE_EVIDENCE] deny cap %d reached — no further WEV denies this sim")` | `[T2_WRITE_EVIDENCE] **deny cap 8 reached** — no further WEV denies this sim` | ★**이 sim 이 통과한 직접 계기** — 게이트가 안 멈췄으면 정답 write 가 계속 막혔다 | **강** |
| **63** | assistant | `submit_interest_discrepancy_report_7294{…, 4.275 / 4.00 / 33.00}` | model | write(성공) | — | `Expected APY: **4.275%** - Actual APY: **4.0%** - Amount Difference: **$33.00**` | **gold 093_8 MATCHED** | **강** |
| **66** | user_sim | 크레딧 적용을 명시 요청 | user_sim | 촉구 | — | `can you also go ahead and **apply the $33.00 credit** if you're able to` | 두 번째 write 로 유도(t7346 이 놓친 자리) | **강** |
| **69/71** | assistant | `unlock(apply_savings_account_credit_6831)` → `{…, amount:"33.00", interest_correction}` | model | write(성공) | — | `Credit applied successfully! … Amount: **$33.00** … New Balance: $144033.00` | **gold 093_6 MATCHED** ⇒ `db_match=True` | **강** |

---

## §5 레버 발화표 (이 sim 줄만 · `[sim=task_093#…]` grep)

| 레버 | trial 0 | trial 1 | 판정 |
|---|---|---|---|
| `T2_SG_DOCS` | 1회 · **`['silver_account']` 9편** | 1회 · `['green_account_(checking)','silver_account']` 12편 | **발화·양쪽 다 정상 동작 — 그러나 입력(REFERENCE)이 오염돼 trial 0 은 재료 결손**. 死레버 아님 |
| `T2_SG_REFRAW` (R3 `ref_from_outputs`) | **0회** | **0회** | ★**死배선**(미발화). 런 전체 25,496줄 중 1줄뿐이고 그것도 오발화(094·카드 계좌) |
| `T2_SG_REQREADS` | 0회 | 0회 | 미발화(정상 — 계좌 read 가 APY 호출에 선행) |
| `T2_SG_GROUND` | 1회 (`period_start` 만 드롭) | 3회 (`actual_apy`+`period_start`) | **trial 0 = 오통과**(`2.775` 인정) · **trial 1 = 정상**(그 deny 가 정답 유도를 낳음) |
| `T2_SG_ROUND` | 0회 | **9회** (`33.00000000000004 -> 33.0`) | 발화·의도대로 작동. **그러나 목적(WEV deny 감소)은 미달** — `_note_result_round` 가 "다음 런이 셀 것"으로 지목한 지표가 **여전히 8 deny** |
| `T2_WRITE_EVIDENCE` | 0회 | **20줄 / deny 8회 + cap** | **trial 0 침묵**(오염값이 진짜 도구 출력이라 통과) · **trial 1 과차단**(정답을 막음) — 양방향 오작동 |
| `T2_SG_RESULT_RANGE` | 0회 | 0회 | 미발화(음수 케이스 없음 — t7336 §3.3 수리의 표적이 이번엔 안 나옴) |
| `T2_SEARCH_AGENT` | 배달 1회(turn 2·92편) + **침묵 4회** | 배달 1회(turn 2·92편) + **침묵 8회** | 발화하되 **클래스 오결정**(`T2_DOCDECIDE → 'Gold Account'`, 093=Silver) |
| `T2_SEARCH_REARM` | 2회 · turn **47** | 2회 · turn **73** | **오발화 시점** — 두 trial 모두 **모든 write 이후**에 silver 델타 8735자를 배달 |
| `T2_CLAIMPROV` | 20줄 | 29줄 | 발화·`unbacked=0` — 이 태스크에선 무해·무익 |
| `T2_PIN_READ` | 0 | 0 | 미발화 |
| `T2_DEMANDED_STEP` | 0 | 0 | 미발화 |
| `T2_FOLLOWUP` | 0 | 0 | 미발화 |
| `FAB_STRIP` | 0 | 0 | **미발화** — trial 0 의 `actual_apy=2.775`(예시값 복사)는 잡지 못했다 |
| `T2_ARG_PRODUCERS` | 0 | 0 | 미발화 |
| READ-FIRST | 0 | 0 | 미발화 |
| `T2_REQUIRE_DOC_DELIVER` | 0 | 0 | 미발화 |
| `T2_PROV` (참고) | 1회(`customer_name='John Doe'`) | 1회(동일) | ★대조 의미 있음 — **식별자 축에서는 "스키마 예시값" 을 정확히 잡는다**: `argument 'customer_name'='John Doe' … it looks invented (**e.g. a schema example value**)`. 같은 죄를 **수치 축**(`actual_apy=2.775`)에서는 아무도 안 잡았다 |

**직전 런 이후 들어간 수리가 이 궤적에 개입했는가**

| 수리(t7336→t7346/t7348) | 개입 | 결과 |
|---|---|---|
| `result_round`(A2·2026-08-22) | **개입함**(trial 1 · 9회) | 부동소수 잔차는 접었으나 **WEV deny 는 안 줄었다**(8회) — 남은 불일치는 `"33.00"` 대 `33.0` 의 **말미 0**이다 |
| `result_range`(A8) | 미개입(음수 미발생) | t7336 §3.3 재현 없음 = 그 수리는 이번 런에서 **미검정** |
| `requires_reads`(A6③) | 미개입(093) · 094 에서만 발화 | 093 은 read 선행이 이미 성립 |
| `ref_from_outputs`(R3) | **미개입(死)** | ★x481 이 *"레코드 원문 → checking 4/4·합 4.275"* 로 4/4 를 실증했던 그 수리가 **라이브에서 0회** — trial 0 실패의 우리-층 몫 |
| `_note_balance_tier` 지시(C_bal_hint) | **개입함** | t7336 의 근인(base tier 2.5 오선택)은 **재현되지 않았다** — trial 0 도 base 는 4.0 을 맞혔다. **이 수리는 샀다** |

---

## §6 선행 판정과의 대조 — 같은 원인인가 달라졌는가

| 문서 | 선행 판정 | t7348 실측 | 판정 |
|---|---|---|---|
| `t7336_tasks/T7336_TASK_093.md` | 근인 = `get_correct_savings_apy -> **2.75**`(base tier **2.5** 오선택 + relationship 관문1 드롭) · 모델이 넘긴 base 4.0 을 서브 산출이 **덮어씀** | **달라졌다.** base tier 는 이제 **4.0 으로 맞다**(잔액 전달 + 지시 수리가 샀다). 남은 결손은 **checking +0.25 · relationship +0.025** | **부분 해결 · 잔여 이동** |
| 같은 문서 `_note_balance_tier`(x481) | `C_bal_hint 4/4` — 결정적인 것은 재료가 아니라 지시 | 라이브에서 **재현**(base 4.0) | **일치** |
| `_note_ref_from_outputs`(R3·x481) | *"레코드 원문 → checking 4/4 · 합 4.275 (지시 문장 없이 해결)"* | 라이브 **0회 발화** ⇒ 격리에서 산 것을 **라이브에서 못 샀다** | **[[55]] 0단계 위반 사례 — 배선 생존 미확인** |
| `FAILURE_MASTER__20260822.md:172` | R3 판정 = *"표적 미도달 · 기대 상한 낮음"*, 근거 = *"반증자: **093 은 REFRAW 없이도 회복**(REFERENCE 에 레벨 이름만 있으면 됨)"* | **그 반증이 무너졌다.** t7346 양 trial·t7336 은 모델이 `"Green Checking Account"` 를 썼기에 회복했을 뿐이고, **t7348 trial 0 은 쓰지 않았다**(`"No credit card accounts"`) → 즉시 실패 | **선행 판정 갱신 필요 — 기대 상한 상향** |
| 같은 문서 `:295` A-4(`_evidence_ctx` 생산자 키 `_eff_tool_name` 정규화) | 기대 **"낮음"** | 실측 outs 키에 `"accounts"` 포함 **0개**(양 trial) — A-4 는 **093 을 직접 표적**한다 | **A-4 우선순위 상향 근거 확보** |
| `FAILURE_MASTER__20260822.md:179` | `T2_SG_DOCS` 는 093·094 에서 정상 발화 = 死레버 아님 | **재확인**(1회씩 발화) | **일치** |
| `STATE_OF_PLAY_2026_08_23.md:87-88`(x488/x489) | 093 을 막은 것은 우리 **문구**도 우리 **차단**도 아니다(둘 다 반증) · *"093 두 번째 write 미실행"* 이 잔여 | t7348 에서 **두 번째 write 는 실행됐다**(trial 1 msg 71 · 손님 촉구 msg 66 이 계기). trial 0 은 write 미실행이 아니라 **값 오염** | **잔여 축 이동**(미실행 → 값) |
| `t7346` 직전 런 실측 | 양 trial `-> 4.275` · 결손 = 크레딧 write 미실행 1건 | trial 0 이 **4.275 를 잃음** | **회귀 1건**([[70]] 절충 대상) |

---

## §7 원인 확정

**채점단위** = DB 해시 · **변이집합** = trial 0 `MISSING 2 / WRONGARG 2`(전부 수치 3필드) · **값의 KB 출처** = §2 표 · **우리 배선 발화** = §5.

### trial 0 — `cause_primary = our_layer` · `cause_secondary = model`

두 개의 **각자 치명적인** 결손이 겹쳤다(하나만 고쳐도 reward 는 안 돌아온다):

**결손 ① `expected_apy` 4.275 → 4.0**
- 발단(model): msg 35 에서 계좌 레코드 2행 중 checking 행을 인자에서 소거.
- 우리 몫(CONFIRMED): 그 소거를 막으려 저작된 R3 가 **0회 발화**. 코드 경로 = `t2_scaffold_get.py:722` 가 조회하는 `__tool_outputs_raw` 를 `t2_scaffold_get.py:1763-1767,1782` 가 **외곽 도구명**(`call_discoverable_agent_tool`)으로 키잉하므로, 선언 `producer_contains=["accounts"]` 가 은행 도메인의 discoverable getter 를 **구조적으로 못 뚫는다**. 실측 키 목록에 `"accounts"` 포함 0개.
- 반사실 근거[M]: 선언 자신의 `_note`(x481 격리 4회씩) *"**레코드 원문** checking 4/4 · 합 4.275 ← 지시 문장 없이 해결"* + 라이브 trial 1(레벨 이름이 실린 REFERENCE → 12편 → 4.275).

**결손 ② `actual_apy` 4 → 2.775**
- 발단(model): msg 37 에서 도구문서 예시값을 인자로 복사. A2 param 문면이 유도식을 축자로 지시했는데 따르지 않음.
- 우리 몫(CONFIRMED): 관문1 이 통과시킴. 코드 경로 = `t2_scaffold_get.py:227-229` 의 `ledger` 코퍼스가 **에이전트 도구 출력 전량**(=`KB_search_bm25`/`unlock_…` 출력 포함)이라, **도구 자신의 파라미터 예시**가 원장 근거가 된다. `_val_grounded` 의 docstring 이 이미 이 한계를 자인한다: *"다른 곳에 우연히 있는 틀린 값은 못 잡는다"*.
- 반사실 근거[S·같은 sim 내]: trial 1 msg 35/37 에서 동일 관문이 `actual_apy` 를 드롭하자 → msg 39 에서 모델이 **480×12/144000=4.00 을 스스로 유도**했다. 즉 **드롭이 정답 유도를 낳는다**는 것이 이 태스크 안에서 실측됐다.

⇒ 우리 층이 **두 자리 모두에서 설계된 방어를 집행하지 못했고**, 한 자리(②)는 같은 sim 안에 *집행했더라면 회복했다* 는 반사실이 있다. 그래서 primary 를 our_layer 로 둔다. 다만 두 발단은 모두 모델의 전사/날조이므로 secondary=model 을 병기한다(자기층 과잉귀속 경계·[[31]] 규칙 6).

### trial 1 — 통과했으나 **우리 게이트가 정답을 8회 막았고, 상한이 아니었으면 실패했을 것**
`T2_WRITE_EVIDENCE` 가 `"33.00" ∉ "…= 33.0."` 로 정답 write 를 8회 반려(`t2_gate_patch.py:1402` 순수 substring)했고, `deny cap 8 reached` 로 게이트가 포기한 **직후 턴**에 write 가 나갔다. 같은 리포의 `t2_scaffold_get._val_grounded` 는 같은 종류의 수치를 `_as_float`+`_nums_in` 로 **형식-불문** 비교한다 — **두 게이트의 술어가 불일치**한다. `result_round` 수리의 `_note` 가 스스로 지목한 검증 지표(*"`[T2_SG_ROUND]` 발화 수 ↔ 그 뒤 WEV deny 수(줄어야 한다)"*)는 **9 ↔ 8 로 미달**이다.

### 우리 층이 아닌 것(명시)
- **env**: 3회 스키마 거부(trial 1 msg 41/43/45)는 정당한 거부이고 모델의 인자 오류다. 도구문서의 `e.g., 2.775` 예시는 env 재료이며 **그 자체는 결함이 아니다**(예시는 정상적인 문서 관행) — 결함은 그것을 근거로 세는 우리 코퍼스 정의다.
- **user_sim**: 두 trial 모두 협조적이고, trial 1 의 msg 66 촉구는 오히려 두 번째 write 를 낳았다. `user-sim 요인` 으로 종결할 자리 없음([[21]]).
- **차단**: trial 0 BLOCKED=0. `x489`(우리 차단이 093 을 막았나) 의 **반증은 유지**된다.

---

## §8 처방 후보 (제안까지만 · 실행·코드 수정 없음)

| # | 표적 결손 | 후보 | 근거 | [[70]] 무엇을 파나 |
|---|---|---|---|---|
| P1 | ①(R3 死배선) | `_evidence_ctx` 의 생산자 키를 **효과 도구명**(`_eff_tool_name`·같은 파일이 READ-FIRST 에서 이미 쓰는 술어)으로 정규화 → discoverable getter 가 `producer_contains` 에 잡히게 | `FAILURE_MASTER:295` A-4 와 동일 항목이나 **기대 상한을 상향**할 근거가 이 보고서다(선행의 반증자 무효화) | REFERENCE 바이트 증가·클래스 과포함(비용은 바이트뿐·`T2_SG_DOCS` 로그로 가시) |
| P2 | ②(관문 오통과) | `ground.scalar_fields` 의 `corpus` 를 **생산자 한정**(예: 레코드/거래 getter 출력만)으로 좁힐 수 있게 선언 축 추가 — KB 검색 출력이 `ledger` 로 세지 않도록 | trial 1 의 드롭이 정답 유도를 낳은 **동일-sim 반사실**[S] · `_val_grounded` docstring 의 자인 | 진짜 KB 유래 값(예: 정책이 명시한 APY)까지 드롭될 수 있음 — **미측정** |
| P2' | ② 보완 | `T2_PROV` 가 식별자 축에서 이미 쓰는 *"schema example value"* 술어를 **수치 operand** 에도 적용(도구 자신의 param 문서에서 온 값 = 근거 아님) | 같은 런에서 `customer_name='John Doe'` 는 잡고 `actual_apy=2.775` 는 못 잡았다 | 예시값과 실제값이 우연히 같은 케이스 오차단 — **미측정** |
| P3 | WEV 과차단 | `t2_gate_patch.py:1402` 의 `str(idv) in c` 를 **수치일 때만** `_val_grounded` 동형(형식-불문 수치 매칭)으로 통일 — 같은 리포 안의 술어 불일치 해소 | trial 1 에서 정답 write 8회 반려·16 메시지 소모, cap 이 아니었으면 실패 | 수치 정규화가 `33` 과 `33.00`, `330` 등 인접값 오인정을 낳는지 — **경계 확인 필요** |
| P4 | 재료 오배달 | `T2_DOCDECIDE` 가 **계좌 레코드 read 이전**(turn 2)에 클래스를 확정하지 않도록(또는 read 직후 REARM 을 강제) | 두 trial 모두 turn 2 에 `'Gold Account'` 오결정 · REARM 이 turn 47/73(모든 write 이후) | 초반 배달 지연 = 초기 턴 정보 감소 |
| — | (재론 금지) | trial 1 의 env 스키마 거부 3건 · `T2_SG_RESULT_RANGE` | 전자는 모델 몫, 후자는 이번 런 **미검정** | — |

---

*작성 규율: 원인 귀속은 궤적 축자 인용만([[08]]) · `our_layer` 주장은 전부 파일:줄 또는 선언 키 지목 · gold(`reward_info`)는 진단용으로만 사용([[23]]) · 수리 실행·코드 수정 없음.*
