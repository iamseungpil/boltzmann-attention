# VERDICT `VIOLATES 0` 가르기 — 런북 STEP 2 (2026-08-18)

> 등대 = `RESEARCH_MASTER.md`(§1 LOCK·§1.3 상쇄·§3 원장). 원장 = **C535**.
> 재료 = 영속 gz **로컬만**(GPU 0 · 유료 0 · LLM 호출 0). 도구 = `scripts/distill/tau2/x377_verdict_split.py`.
> 대상 = `bank_t7310_treat_20260818e`(12 sim) + `bank_t7312_treat_20260818g`(4 sim) 의 **판정 줄 21건**.
> ⚠t7313 은 아직 진행 중 — 완주 후 같은 스크립트로 **합쳐서** 다시 낸다(엔진 트리 해시 동일).

## 0. 무엇을 물었나 (사전 고정 · 핸드오프 §0-2 STEP 2)

`T2_VERDICT_CARRY` 는 트리거 자리 100% 발화인데 판정 줄의 절반이 `VIOLATES 0` 이다(핸드오프 §5⑴).
그것이 **옳은 침묵**(위반 후보가 정말 없다)인지 **무력**(판정이 안 갈린다)인지 아직 안 갈렸다.

**사전 고정 기준(축자)**: *"무정보 줄의 **과반에서 gold 가 OK 집합에 그냥 섞여 있으면 = 무력**."*
⇒ 이 문서는 그 기준을 **그대로** 적용한다(기준 재조정 없음).

**표적 판정 방법**: 그 군의 후보 표시명이 **gold 액션 인자 문자열에 실재하는가**(`in` 연산·정규식 0·
[[59]]). gold 는 **판정용 조회**이고 레버 재료로 넘어가지 않는다([[23]]).

## 1. 판정 줄 전수 (사이드카 `kind=verdict-lines` 축자)

```
task      tag    turn group                    n  OK  VIO  rw     표적          gold 안 후보
------------------------------------------------------------------------------------------------------------
task_016  t7310  4    credit_cards            10  10    0  0.0    있음          Silver Rewards Card=OK
task_024  t7310  4    business_credit_cards    7   7    0  0.0    있음          Business Bronze Rewards Card=OK
task_024  t7312  4    business_credit_cards    7   7    0  0.0    있음          Business Bronze Rewards Card=OK
task_024  t7312  6    credit_cards            10   5    5  0.0    있음          Bronze Rewards Card=VIOLATES
task_055  t7310  2    checking_accounts       10   6    4  0.0    있음          Purple Account=OK
task_055  t7310  4    savings_accounts         9   0    9  0.0    있음          Silver Plus Account=VIOLATES
task_055  t7310  15   credit_cards            10   2    8  0.0    없음          -
task_055  t7312  2    checking_accounts       10   6    4  0.0    있음          Purple Account=OK
task_055  t7312  6    savings_accounts         9   2    7  0.0    있음          Silver Plus Account=OK
task_055  t7312  22   credit_cards            10   1    9  0.0    없음          -
task_057  t7310  6    savings_accounts         9   3    6  0.0    없음          -
task_063  t7310  8    credit_cards            10   3    7  0.0    있음          Silver Rewards Card=OK
task_063  t7310  12   savings_accounts         9   2    7  0.0    있음          Silver Plus Account=OK
task_072  t7310  2    checking_accounts       10  10    0  0.0    없음          -
task_072  t7312  2    checking_accounts       10  10    0  0.0    없음          -
task_073  t7310  2    checking_accounts       10  10    0  0.0    없음          -
task_079  t7310  2    checking_accounts       10   9    1  0.0    없음          -
task_085  t7312  2    checking_accounts       10  10    0  0.0    없음          -
task_085  t7312  4    savings_accounts         9   9    0  0.0    없음          -
```

## 2. 교차표와 판정

```

## 교차표 — 무정보(VIOLATES 0) × 표적유무
         표적있음       표적없음      
무정보      3          6         
갈림       7          4         

무정보 줄 9 중 표적 있는 줄 3 (33%) — 사전 고정 기준: 과반이면 **무력**
판정: 무력 아님 — 무정보의 과반이 **표적 없는 자리**(발화 자체가 범위 밖)

## 무정보 줄의 gold 도구(그 자리에서 무엇을 해야 했나)
  task_016  t7310  turn=4   credit_cards           gold=log_verification, submit_transaction
```

### 판정 = **무력 아님** (사전 고정 기준 그대로)

무정보 9줄 중 gold 후보가 OK 집합에 섞인 줄은 **3**(33%) — 과반 미만이다.
⇒ *"판정 능력이 없다"* 는 형태의 결손은 **이 데이터로 지지되지 않는다**.

## 3. 그러면 무정보는 무엇인가 — **요구가 배제 술어가 아닌 자리**

무정보 9줄을 궤적·gold 로 갈라 보면 두 종류뿐이고, 둘 다 *판정 능력*이 아니라 **적용 범위** 문제다.

| 종류 | 태스크 | 손님 요구(축자) | 왜 아무것도 안 걸리나 |
|---|---|---|---|
| ⒜ **제품 선택 자체가 없다** | 072 · 073 · 093 · 085 · 016 | *"something seems off with my ATM fees"* · *"my monthly interest just seems… low?"* · *"I need help with some disputes"* · *"I still haven't received my referral bonus"* | gold 는 전부 **조회·정정·분쟁 접수**(`get_bank_account_transactions_9173` · `apply_checking_account_credit_5829` · `file_debit_card_transaction_dispute_6281` · `submit_transaction`). 고를 제품이 없으니 **배제할 후보도 없다** |
| ⒝ **요구가 최대화다** | 024(1차·`business_credit_cards`) | *"I want to open a new business credit card that will give me the **best return** on this purchase"* | *"최선"* 은 **어떤 후보도 위반시키지 않는다**. 갈리려면 argmax 인데 그것은 ⛔0 이 금지한 자리다(엔진이 최댓값을 내면 측정 대상이 사라진다) |

⇒ 이 레버는 **빼기 도구**다([[63]]). 무정보 줄은 *뺄 것이 없는 자리*에서 났다.
같은 태스크라도 요구가 배제형으로 바뀌면 즉시 갈린다 — **024 는 2차(`credit_cards`·turn 6)에서
`VIOLATES 5`** 로 갈렸고, 055 는 세 축 전부 갈렸다.

## 4. ★1차 기입 철회 — 문서 본문은 **어느 팔에서도** 에이전트에 안 간다

처음에 나는 *"무정보 줄이 문서 본문을 대체한다 → 072 는 규정 본문이 필요한데 사라졌다"* 로 적었다.
**틀렸다.** 배달 경로를 직접 확인한 결과:

- `decide=True` 경로가 에이전트에게 주는 것은 `decided_by_docs_text` = **이름 한 줄**이다
  (*"It answers: Blue Account."*). 문서 본문(`_mat`)은 **결정 서브까지만** 간다 — 양팔 동일.
- 궤적 전수에 `It answers` **0건** · 주입은 **뷰 경로**(사이드카가 축자로 기록).

⇒ `_vmat` 치환이 바꾸는 것은 *에이전트가 본 재료*가 아니라 **서브가 무엇을 보고 그 이름을 골랐나**다.
원장 C535ⓓ 에 철회로 남겼다([[03b]] 스스로 플래그).

## 5. ★대신 나온 것 — **양팔 공통** 범위 밖 주입 (사이드카 축자)

| 태스크 | 성격 | t7310 ctl | t7310 treat |
|---|---|---|---|
| 072 | ATM 수수료 환불 | `Blue Account` | `Blue Account` |
| 073 | ATM 수수료 환불 | `Blue Account` | **`Green Account (checking)`** |
| 079 | (검증 자리) | `Blue Account` · `General` | **`Evergreen Account`** |
| 093 | 이자 정정 | `Gold Account` | **`Platinum Account`** |
| 085 | 직불카드 분쟁 | (t7312) `Blue`·`Gold` | (t7312) `Blue`·`Gold` |
| 055 | **진짜 제품 선택** | turn 42·44 `Gold Account` | **주입 기록 0** ⚠ |

주입 건수: t7310 **ctl 8 / treat 5** · t7312 ctl 5 / treat 3.

⒜ **제품을 고르는 자리가 아닌 태스크에 제품 추천이 들어간다 — 양팔 다.** 즉 범위 문제의 주소는
   `VERDICT_CARRY` 가 아니라 **검색-에이전트 축 자체**다([[55]] 우리 층 먼저).
⒝ 판정 줄이 이름을 **바꾼다**(073·079·093) — 내용 0 인 줄로 고른 이름이 문서로 고른 이름과 다르다.
⒞ ⚠**미해결(계기)**: 055 는 treat 가 축 3개를 다 처리했는데(`축 처리 완료` ×3) 주입 기록이 0 이다.
   배달·큐 소비 경로를 아직 안 봤다 — 결론 내지 않는다([[08]]).

## 6. 격리 ① 최종 설계 (①″ · 8141 전용 · 사용자 승인 대기)

핸드오프 §5⑴ 원안(*"무력한 컷에서 요구를 배제 술어로 재진술"*)은 **없는 결손을 잰다**(§3 Ⓐ).
§5 가 연 자리를 대신 잰다 — **범위 밖 주입이 다음 행동을 틀어놓는가**.

| arm | 결정 시점 문맥 | 무엇을 가르나 |
|---|---|---|
| **A_REF** | 라이브 대화 축자, 주입 **없음** | 기준선 |
| **B_INJ** | + 라이브 주입 **축자**(*"It answers: Blue Account."*) | 주입의 비용 |
| **D_NEG** | + 같은 형식·**다른 제품 이름** | 모델이 이름을 읽는가(계기) |

- 컷 = 범위 밖 주입이 실제로 있었던 자리(072·073·079·093·085 · 양팔) + **양성통제** = 055
  (주입이 정당한 자리 — 여기서 B 가 A 보다 낫거나 같아야 계기가 산다).
- 채점 = **방출된 tool_call 이름**(접미사까지 일치·x370 규약) — 그 자리의 진행 방향
  (`unlock_/call_discoverable_agent_tool` + 조회 도구)인가, 제품 추천·개설 쪽으로 새는가.
  **gold 무참조**([[23]]) · 결정론.
- 사전 고정 판정:
  · **B < A 가 과반 컷** → 주입 자체가 비용 ⇒ 처방 = **발화 범위 조건**(고르는 자리에서만)
  · **A ≈ B** → 주입 무해 ⇒ 이 자리는 레버가 아니다(원인은 딴 데)
  · **D_NEG ≈ B** → 모델이 이름을 안 읽는다 = **계기 무효**(결과 인용 금지)
  · **A 도 실패** → 그 결손은 주입과 무관

⛔처방은 이 격리 뒤에만([[62]]). 지금 확정된 것은 **원인 진술과 인벤토리**뿐이다.
