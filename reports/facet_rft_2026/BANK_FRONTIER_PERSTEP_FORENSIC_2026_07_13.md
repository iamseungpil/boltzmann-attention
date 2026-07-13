# banking frontier 17모델 전수 per-step 실패 정밀 분류 (2026-07-13·[[08]]·[[47]])

> 데이터: `C:/tmp/traj/*_banking.json` 17 frontier 모델(user-sim gpt-5.2·nt=4·388 sim/모델). [[47]] "raw 소실" 오류 교정(로컬 완비).
> 스크립트: `bank_frontier_perstep.py`(원인 분류) · `bank_frontier_argdiff.py`(arg-level). requestor-aware·action_match·[[48]] 서술형.

## 0. 한 줄 (결론)
**banking frontier-irreducible 격차 = ⋈ 경계가 아니라 *정책-구동 파라미터 계산/판정***. 발견한 도구(`call_discoverable_agent_tool`)에
넘길 nested 파라미터(책임상한·자격불리언·금액·APY·날짜)를 **정책 규칙+데이터로 계산**해야 하는데, **gpt-5.5(최강 37.4%)조차 틀린다.**
이는 **decidable**(형식화→정책적용→계산) = 프레임 **F2b(계산)/F1(정책)** 결정론 scaffold의 정확한 사정거리(경계 아님).

## 1. 모델별 pass + retrieval (17모델)
| 모델 | pass | retrieval | | 모델 | pass | retrieval |
|---|---|---|---|---|---|---|
| **gpt55** | **37.4%** | AllTools | | opus45 | 21.4% | AllTools |
| gpt54 | 30.7% | AllTools | | grok42 | 17.6% | AllTools |
| opus47 | 25.3% | AllTools | | gemini3pro | 15.7% | terminal |
| gpt52 | 24.7% | AllTools | | grok4fast | 14.2% | AllTools |
| opus46 | 24.5% | AllTools | | gpt52none | 12.4% | qwen_emb |
| gemini31pro | 22.5% | AllTools | | grok41fast | 12.4% | AllTools |
| sonnet45 | 22.4% | terminal | | gemini25pro | 12.8% | AllTools |
| gemini3flash | 20.6% | terminal | | **qwen397b** | 11.5% | **openai_emb** |
| | | | | **glm5** | 11.1% | **openai_emb** |
- **우리 retrieval(openai_embeddings)=frontier도 ~11%**(단 think 모델 too_many_errors 아티팩트). frontier 최상=37.4%.

## 2. POOLED 실패 원인 (전 모델·전 실패 gold-action)
| 원인 | 비중 | 정밀(arg-level) |
|---|---|---|
| **call_discoverable 파라미터 오류**(도구 맞음·nested 틀림) | ★지배 | `arguments` nested 12,037 mismatch |
| **operator-⋈**(틀린 도구명) | 27.6% | agent_tool_name: call 4,471+unlock 2,753 |
| reach-discovery(미호출) | 14.7% | call_discoverable 미호출 2,482 |
| coverage-미완 | 7.6% | |
| log_verification.time_verified(시각 날조) | — | 314 |
| apply.card_type(카드 ⋈) | — | 293 |
| transfer.summary/reason | — | 333 |
| submit_referral.account_type | — | 187 |
- **포기종결(조언/transfer) 4,660 · over-action 677**.

## 3. ★핵심 — call_discoverable_agent_tool 실패 정밀 분해 (frontier 최대 격차)
`call_discoverable_agent_tool` = 발견한 실제 뱅킹 오퍼레이션(분쟁접수·계좌개설·거래신고 등)을 파라미터와 함께 실행.
- **틀린 도구명 4,384**(operator-⋈) · **도구 맞음·파라미터 틀림 4,311** · 도구·파라미터 맞음·타기준실패 4,561 · 미호출 2,482.

**★어느 nested 파라미터가 틀리나 (도구는 맞춤·전 frontier):**
| 파라미터 | 오류수 | 유형 |
|---|---|---|
| `open_bank_account.account_class` | 624 | 범주 선택 |
| **`file_debit_card_transaction.customer_max_liability_amount`** | **602** | ★정책-계산(책임상한) |
| **`file_credit_card_transaction.eligible_for_provisional_credit`** | **439** | ★정책-판정(불리언) |
| `file_credit_card_transaction.card_last_4_digits` | 416 | 참조 조회 |
| **`submit_interest_discrepancy.amount_difference` / `.expected_apy`** | 295/243 | ★계산(차액·APY) |
| **`apply_savings/checking_account_credit.amount`** | 289/283 | ★계산(금액) |
| **`file_debit_card_transaction.pin_compromised` / `.provisional_credit_eligible`** | 206/195 | ★정책-판정 |
| `file_debit_card_transaction.disputed_amount` | 198 | 참조/계산 |
| `.transaction_type` / `.discovery_date` / `.issue_noticed_date` | 170/120/94 | 범주·날짜 |

**per-case 확증(task_085·전 frontier 0/66)**: 도구 `file_debit_card_transaction_dispute`·gold `customer_max_liability_amount`=**50** vs gpt55=**100**(전액). disputed=100·discovery=거래당일 → **정책(즉시신고 시 책임상한 $50)** 미적용. = 정책-계산 실패.

## 4. HARD CORE (전 frontier pooled pass ≤10% = irreducible)
- **45/97 태스크가 전 frontier서 ≤10% pass** (많은 태스크 pass=0/66). 지배원인: **파라미터-계산/판정 + operator-⋈**.
- 완전-0 태스크: t020·027·029·039·046·049·053·060·063·065·066·067·068·069·074·077·078·079·080·081·082·083·084·085·087...

## 5. ★함의 (thesis·다음)
1. **banking 격차의 실체 = 정책-구동 파라미터 계산/판정**(책임상한·자격불리언·금액·APY)·**decidable**. 프레임 F2b/F1의 결정론 scaffold 사정거리 = **경계 아님**. frontier(gpt55 37.4%)조차 못 여는 걸 scaffold가 열 여지 = thesis 강한 지지 후보.
2. **내가 만든 레버(L0-4: reach/verify/discovery/card-추천)는 엉뚱한 표적**(gpt-4.1 user-sim 아티팩트·C75). **진짜 레버 = 정책-파라미터 formalize→compute**(retail fexec 동형·[[05]] 정책규칙=A2·계산로직=도메인일반).
3. operator-⋈(틀린 도구명 4,384)=보조 격차(발견도구 선택). 이건 F3 성격(FIND).
4. caveat: `arguments` nested는 도구마다 스키마 상이 → 정책규칙 A2 인코딩 부담 큼. 실현가능성=별도 판단. reward_basis 대부분 DB·[[08]] 소표본 아님(17모델×388).

## 5b. ★★KEYSTONE 실증 — A2-선언 op가 frontier-irreducible 파라미터를 95% 재현 (사용자 아키텍처)
> 사용자 설계: **엔진 = 일반 op 라이브러리 + A2-스펙 인터프리터**(min/max/argmax/sum/diff/clamp/lookup_table/
> days_between...). A2가 도메인별 op 선언(어느 도구·어느 파라미터·어느 임계). scaffold/loop 불변·전이=A2만.

- **구현**: `t2_compute.py`(도메인-일반 op 라이브러리·리터럴 0·unit 14/14 `test_compute.py`). A2 op-스펙 = 예:
  `customer_max_liability_amount = {op:lookup_table, key:{op:days_between, a:tx_date, b:disc_date}, table:[{<=30→50},{else→500}]}`.
- **★gold 재현 실측(전 frontier 1109 케이스)**: threshold 2일=68% → **15~30일=95%**(1050/1109). 나머지 5%=412.88 특수분기(필드 추가=튜닝).
- **의미**: gpt55(37.4% 최강)조차 틀리는 `customer_max_liability_amount`(disputed 전액 오입력)를 **A2-선언 op + 일반 엔진이 95% 계산**. = ① 사용자 아키텍처 작동(엔진 고정·ABox만·[[05]]/[[11]]) ② **결정론 scaffold가 scale-불가 decidable 격차를 연다**(thesis 강증거·banking 키스톤).
- **잔여(라이브 keystone까지)**: (a) 에이전트가 입력(날짜) 수집=reach (b) 계산값을 call_discoverable nested args에 주입+검증(엔진 nested 확장) (c) 타 파라미터(provisional_credit_eligible=정책판정·amount_difference=diff·expected_apy) op-스펙 저작. 코어 형식화-가능성=실증됨.

## 6. 산출물
- `bank_frontier_perstep.py` · `bank_frontier_argdiff.py` · 데이터 `C:/tmp/traj/*_banking.json`.
