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

## 4. ★부작용 후보 — 무정보 줄이 **문서 본문을 대체한다** (미측정)

`t2_gate_patch.py:2835~2855` 축자: `_vlines` 가 생기면 `_vmat = chr(10).join(_vlines)` 로
**결정 서브에게 가는 재료가 문서 본문에서 이름 목록으로 바뀐다**. 무정보 줄에서 그 목록은
*내용이 0* 이다(전부 `OK`, 인용 0).

실측(072·t7310): 양팔 다 `[T2_SEARCH_AGENT] group=checking_accounts · 문서 113` 을 읽었는데,
treat 의 결정 서브가 받은 것은 **10줄의 이름 + OK** 였다. 072 의 gold 는 **ATM 수수료 규정으로
환불액을 계산**하는 일이다($14.00 · $3.50) — 그 규정 본문이 필요한 자리에서 본문이 사라졌다.

⚠**이것은 아직 가설이다**([[62]]·§1.3): 손해는 측정되지 않았다(양팔 다 0.0). 다음 격리가 잴 것.

## 5. 이 결과가 바꾸는 것 — 격리 ① 재설계

핸드오프 §5⑴ 의 격리 ①은 *"무력한 컷(072·085·093)에서 요구를 배제 술어로 재진술하면 갈리는가"*
였다. **그 컷들은 애초에 제품을 고르는 자리가 아니다**(위 ⒜) ⇒ 그 프로브는 없는 결손을 잰다.

**대체 설계 ①′ — 무엇을 팔았나를 잰다**(072형·정보-맞춘 격리·8141 전용)

| arm | 결정 서브가 받는 재료 | 무엇을 가르나 |
|---|---|---|
| **A_DOC** | 문서 본문(현행 ctl 경로) | 본문이 있으면 되는가 = 양성통제 |
| **B_LINES** | 무정보 판정 줄(전부 OK·현행 treat) | 본문 대체의 **비용** |
| **C_BOTH** | 판정 줄 + 본문 | 대체가 아니라 **덧붙이기**면 회복되는가 |
| **D_NEG** | 다른 태스크의 판정 줄 | 계기 검정(비슷하면 무효) |

과제 = 072 의 수수료 환불액 산출(닫힌 정답 · gold 대조는 **방향만**·C486).
사전 고정: **B < A 면 대체가 비용**이고 처방은 *덧붙이기 또는 범위 축소*. **A 도 실패하면** 그
결손은 이 레버와 무관하다(다른 자리).

⛔처방은 이 격리 뒤에만 고른다([[62]]). 지금 확정된 것은 **원인 진술**뿐이다.
