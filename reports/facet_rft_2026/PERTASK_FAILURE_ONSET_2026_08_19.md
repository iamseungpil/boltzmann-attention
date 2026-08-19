# 20태스크 실패 원인 — 궤적 정독 최종 (사용자 지시 2026-08-19)

> 자료 = `bank_t7326_half{A,B}_20260819q` 40 sim 전량. 계기 = `x413_trace_read.py`(정독) ·
> `x414_argdiff.py`(gold↔실제 인자) · `x416_db_diff.py`(**변이 집합 대조**) + KB 원문 대조.
> t7326 의 A2 gold-fit 오염은 **pass 를 만드는 방향**이라 실패 원인 분석에는 보수적으로 작용한다.

## ⛔전제 정정 — 미매치 gold 는 실패 원인이 아니다
```
reward_basis 실측:  DB 35 sim · ACTION 4 · 없음 1
미매치를 갖고도 통과:  017 t1 (4중 2 미매치) · 050 t1 (13중 1 미매치)
```
35/40 이 **최종 DB 상태**로만 채점된다. read 는 아무리 놓쳐도 점수와 무관하다.
⇒ 어제부터 세어 온 **"미매치 gold 144건" 은 실패 단위가 아니었다.** 무언급 1위였던
`get_bank_account_transactions_9173`(16건)은 **read** 다. 올바른 단위 = **성공한 변이 호출의 집합**.

---

## 태스크별 (통과 7/40 = 017t1 · 024t1 · 050t1 · 098×2 · 100×2)

| task | 부류 | 실패가 시작된 지점 (궤적 축자) |
|---|---|---|
| **003** | WRONGARG | `apply_for_credit_card` 를 **Platinum** 으로 (gold **Silver Rewards Card**). 나머지 인자 3개는 일치 |
| **004** | ACTION | 이관 `reason` = `customer_demands_after_unavailable_offer_refusal` (gold `account_ownership_dispute`). KB 표 축자: *"account_ownership_dispute \| … **identity verification failures requiring specialist**"* |
| **016** | MISSING | gold `submit_transaction` 은 `requestor=user`. 손님은 **에이전트가 구체 금액을 말해야** 실행. `spend at least $750` 이 11,073자 BM25 결과 **깊이 93.5%** 에 있었고 안 쓰임 |
| **017** | EXTRA/대체 | `submit_cash_back_dispute` 를 안 하고 **`update_transaction_rewards` 로 대체 실행**(EXTRA 4). 게다가 env 스키마에 없는 `correct_rewards`·`recorded_rewards` 를 지어냄 |
| **024** | WRONGARG | **Business Platinum** (gold **Business Bronze**). ⚠시작은 우리 층 — `check_card_application_fit` 이 `[GROUNDING WARNING] min_credit_limit=40000 … dropped` 를 내고 그 상태로 추천 |
| **033** | ACTION | t0 검색 **0회** · t1 grep 이 절차 없는 줄만 반환 → 선행 unlock/call 사슬 미실행. **x411(wrap ON)에서 0/5 → 4/5** |
| **040** | MISSING ×5 | `file_credit_card_transaction_dispute_4829` 를 **한 건도 안 함**. 손님에게 카드 끝 4자리를 요청하고 대기하다 종료 |
| **050** | EXTRA | `approve_credit_limit_increase_5847` **중복 호출**(2회차 `Previous $7500 → New $7500 · Increase $0.00`) → `db_match=false`. t1 은 1회 → 통과. `months=12`(gold 3)는 **read 라 점수 무관** |
| **055** | WRONGARG ×4 + MISSING | `Green Fee-Free/checking`·`Gold/savings` (gold `Purple/checking`·`Silver Plus/savings`) · `deposit_check` 미실행 |
| **057** | MISSING + WRONGARG | `Light Blue` (gold `Blue`) · `deposit_check(ac554…, 2000)` 미실행 |
| **063** | WRONGARG | `Bronze Rewards Card` (gold **Silver**) · savings `Gold Account` (gold **Silver Plus**) |
| **072** | MISSING ×2 | `apply_checking_account_credit`(chk_lj82…=14 · chk_538b…=3.5) **한 번도 안 함** |
| **073** | MISSING ×3 | 같음 (9.5 · 9 · 1.5). 거래 조회도 `chk_..._1` 에만 |
| **074** | WRONGARG ×4 | 4계좌 전부 호출했으나 **금액이 전부 틀림**: 1 / 2.5 / 1.25 / 1.5 (gold **27 / 14.5 / 4.75 / 3.7**). t1 은 아예 미실행 |
| **079** | MISSING + 자기취소 | `close_debit_card_4721`(stolen) 3건 미실행 · `order_debit_card` 인자 불일치. t1 은 **freeze → unfreeze** 로 자기 행동을 되돌리고 문맥 초과 |
| **085** | MISSING ×3 | `file_debit_card_transaction_dispute_6281` 미실행. t1 은 **날조 인자**로 1회 시도(`transaction_id: "tx123456"` · `card_id: "unknown"`) |
| **093** | WRONGARG | APY `2.95/3.33` (gold `4.275/4`) → 금액 `45.6` (gold `33`) |
| **094** | WRONGARG | APY `6.5/5` (gold `6.85/5.1`) → 금액 `120` (gold `140`) |
| **098** | PASS | 변이 집합 일치 |
| **100** | PASS | 변이 집합 일치. `submit_referral` 은 **손님이 실행** |

---

## 부류별 (실패 18 태스크)

| 부류 | 태스크 | 수 |
|---|---|---|
| **WRONGARG** — 실행했으나 값·종류가 다름 | 003 · 024 · 055 · 063 · 074 · 093 · 094 | **7** |
| **MISSING** — 그 변이를 아예 안 함 | 016 · 040 · 057 · 072 · 073 · 079 · 085 | **7** |
| **EXTRA/대체** — 중복하거나 다른 도구로 | 017 · 050 | 2 |
| **ACTION 채점** | 004 · 033 | 2 |

### WRONGARG 7건의 값은 전부 KB 에 있다
- 093·094 → `Credit Card APY Bonuses: Stacking Policy` + `Linked Checking Account APY Boost`
  (*"only the HIGHEST … do NOT stack"* / *"credit card APY bonuses **DO stack with** checking boosts,
  relationship bonuses, account tier bonuses"*) — 여러 표를 **선택+합산**해야 한다
- 003·024·063 → 카드 종류 표 · 055·057·063 → 계좌 클래스 표 · 074 → ATM 수수료 환급 규칙

⇒ 회수(문서 유무)도 배달(도착 여부)도 아니고 **표를 자기 사례에 매핑**하는 단계에서 갈린다.

---

## 레버 배치에 대한 함의
- **KB 압축(`policy_qa` wrap)** 이 닿는 것 = **033 하나** (실증됨 0/5→4/5). 040·085 는 절차 문서가
  필요할 수 있으나 미확인.
- **WRONGARG 7 + EXTRA 2 = 9 태스크**는 압축과 무관하다. 필요한 것은 **operand 검산**(두-커널 §검산기)과
  **완료된 변이 재실행 차단**(050 형·현 go_stack 의 dedup 레버는 read 대상).
- **MISSING 7** 은 또 다르다 — 040 은 손님 대기, 016·057 은 손님 도구 전달, 072·073·079·085 는 순수 미실행.

## 계기 한계
- gold 의 래퍼 행(`{"agent_tool_name": …}`)이 MISSING 에 섞인다 — DB 변이가 아니므로 빼야 한다.
- `EXTRA` 에서 `unlock_discoverable_agent_tool` 은 제외했다(2026-08-19 수정).
- 079 t1 은 `reward_basis` 가 비어 있다(문맥 초과) — 채점 자체가 안 됐다.
