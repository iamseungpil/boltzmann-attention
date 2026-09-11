# 아직 지는 태스크의 정밀 포렌식 — 수리가 안 먹었나, 역효과인가 (2026-09-11)

> `fs_`(90160cad) 사전런 뒤 손해 15 중 **아홉이 아직 진다**:
> `048 −3 · 016 −3 · 007 −3 · 040 −2 · 081 −2 · 070 −1 · 019 −1 · 028 −1 · 054 −1`.
> 방법: 태스크별로 gold(`evaluation_criteria.actions`)와 실제 궤적을 맞대고, **sim 단위로**
> 「우리가 그 sim 에서 말했나 ↔ 그 sim 이 이겼나」를 센다. 태스크 사이 비교는 오늘 실측한 잡음(±3)에
> 묻히지만 **같은 태스크 안 sim 사이는 조건이 같다.**

## 세 태스크, 세 가지 다른 병

### `task_007` (base 4/4 → 1/4) — **우리 문장은 한 마디도 없었다**

```
trial 2  r=1  28보  발화 없음  개입 tools=1 fold=0 ask=3 regen=0
trial 3  r=0  26보  발화 없음  개입 tools=1 fold=0 ask=5 regen=0   빠진gold=apply_for_credit_card
trial 0  r=0  24보  발화 없음  …                                   빠진gold=apply_for_credit_card
trial 1  r=0  49보  발화 없음  …                                   빠진gold=apply_for_credit_card
```
gold 는 **단 하나** `apply_for_credit_card` 이고 진 셋은 그것을 **아예 부르지 않았다**.
우리 층은 네 sim 전부에서 **조언 0 · 접기 0 · 재생성 0 · 표지 발화 0**. 남는 개입은
**스캐폴드 도구 17종 주입**(`tools=1`)과 대화 밖 서브호출뿐이다.

⇒ **수리의 역효과가 아니다. 우리가 쓴 문장 때문도 아니다.** 후보는 둘뿐이다 —
①도구 주입이 모델의 선택을 흔든다 ②잡음. 007 은 네 팔에서 `cb 2/4 · ub 3/4 · mn 4/4 · fs 1/4`
로 흔들린 바로 그 태스크이고 **그 팔들 사이에 우리 층의 행동 차이가 없다**(모두 발화 0).
다만 **base 는 4/4 로 흔들리지 않는다** — 주입이 분산을 만든다는 가설이 산다. 미결.

### `task_016` (base 4/4 → 1/4) — **운 sim 은 둘 다 졌다**

```
trial 3  r=1  발화 없음                  빠진gold=-
trial 1  r=0  발화 [LEDGER] x1  regen=1  빠진gold=submit_transaction
trial 2  r=0  발화 없음                  빠진gold=submit_transaction
trial 0  r=0  발화 [LEDGER] x1  regen=1  빠진gold=submit_transaction
        우리가 말한 sim 0승 2패   ·   말 안 한 sim 1승 1패
```
축자: `[LEDGER] You are ending your part here and the run record shows no action for this:`
**`Check the status of the missing referral bonus`**

gold 는 `log_verification > submit_transaction` 이다. **우리는 손님이 청한 것을 「추천 보너스 상태
확인」으로 지목하는데 gold 가 요구하는 것은 거래 제출이다.** 떠나려는 모델을 **엉뚱한 미해결 항목으로
되돌린다.** `ARM` 커밋이 *"naming the referral enquiry rather than the transaction the gold submits"*
라고 적은 진술은 **맞았다**(다만 그 커밋이 근거로 댄 «016 에서 2회» 는 `ub_` 사이드카에선 0회였다).

⇒ **역효과 방향.** n=2 라 결정적이진 않지만, 기전이 문장에 그대로 적혀 있다.
`mn_`(이 레버를 뺀 팔)이 이 태스크에서 3/4 였던 것과 방향이 같다.

### `task_048` (base 3/4 → 0/4) — **우리 지적은 옳았고, 그래도 진다**

```
네 sim 전부 발화 있음 · 전부 패 (0승 4패)
trial 3  [LEDGER]1 [PROCEDURE]3   88보   빠진gold = 없음
trial 0  [LEDGER]1 [PROCEDURE]2 [CLAIM-PROVENANCE]1  105보  빠진gold = 없음
trial 2  [LEDGER]1 [ORDER]1 [PROCEDURE]4  111보  빠진gold = 없음
trial 1  [LEDGER]1 [PROCEDURE]4  101보  빠진gold = 없음
```
**네 sim 다 gold 24개 액션을 전부 수행했다.** 그런데도 DB 판정이 0 이다.

처음에는 `[PROCEDURE]` 가 거짓 경보로 보였다 — turn 32 에서
*"steps before `get_closure_reason_history_8293` … not done yet: **pending_replacement**"* 라고
말하는데, 궤적에는 `get_pending_replacement_orders_5765` 가 turn 11·15·19 에 이미 있었기 때문이다.
**그러나 틀린 것은 내 읽기였다.** 이 절차는 `id_key=credit_card_account_id` — **카드별로 도는
원장**이다(`lb1_requirements.scoped()`). 인자를 보면:

```
msg22  get_pending_replacement / get_closure_reason   → cc_..._green
msg28  close_credit_card_account                      → cc_..._green
msg32  get_pending_replacement , get_closure_reason   → cc_..._eco     ← 같은 메시지에서 동시에
msg41  get_pending_replacement                        → cc_..._gold
msg71  get_pending_replacement , get_closure_reason   → cc_..._crypto
```
손님은 카드가 넷이다. msg32 에서 모델은 **eco 카드**의 `prior_attempts` 를 그 카드의 `disputes`·
`pending_replacement` 없이(같은 턴에 동시에) 호출했다. **우리 지적은 정확했다.**

지는 원인은 다른 데 있다 — 세 sim 이 gold 에 **없는 write** `apply_credit_card_account_flag_6147`
을 추가로 낸다(gold 액션 목록에 `flag` 는 없다). 그리고 `addressed`·`offer` 노드는 **도구가 없는
노드**(`tool=None`)라 발화로만 충족되는데, 우리는 `apply_statement_credit_8472` 앞에서
*"not done yet: **addressed**"* 라고 말한다. **도구 없는 단계를 「안 됐다」고 말하면 모델은 그것을
할 도구를 찾는다** — 가장 그럴듯한 것이 retention flag 다. 이 연결은 **가설이고 아직 미확정**이다.

**반증조건**: `[PROCEDURE] … addressed` 가 울지 않은 sim 에서도 flag 호출이 나오면 이 가설은 틀린다.

## 종합 — 물음에 대한 답

| 태스크 | 수리가 안 먹었나 / 역효과인가 | 근거 |
|---|---|---|
| `007` | **둘 다 아니다.** 우리 층이 침묵했다 | 네 sim 발화 0 · 접기 0 · 재생성 0 |
| `016` | **역효과 방향** | 운 sim 0승 2패 · 문장이 gold 아닌 항목을 지목 |
| `048` | **지적은 옳고 결과는 나쁘다** | `[PROCEDURE]` 는 카드별 원장에서 정확 · gold 는 다 했는데 여분 write 로 진다 |

세 병이 서로 다르므로 **한 레버를 끄는 것으로는 셋 다 못 고친다.** 그리고 오늘 실측한 잡음(태스크당
±3, 우리 층이 침묵한 007 에서 세 팔이 1·3·4)을 감안하면 **007 은 지금 데이터로 판정 자체가 불가**다.

## 다음에 할 것 (싼 것부터)

1. **도구 없는 노드의 발화**를 손본다 — `addressed`·`offer` 처럼 `tool=None` 인 단계를
   *"안 됐다"* 로 말하지 말고 **무엇이 그것을 충족시키는지**를 말한다(모델이 도구를 지어내지 않도록).
   `048` 의 여분 write 가설을 먼저 확정한 뒤에.
2. `016` 의 `[LEDGER]` 가 **gold 가 요구하는 미해결 항목**을 지목하는지 검사하는 게이트.
   지금은 손님이 말한 아무 문장이나 집는다.
3. `007` 은 **주입 자체를 끄고 한 번** 재는 것 말고는 가릴 길이 없다(문장이 0이므로).

---

# 프롬프트 수준 대조 — 우리가 창에 넣은 글자 vs gold (2026-09-11 추가)

사이드카의 모든 레코드에 `len` 이 있다. 모델이 **실제로 읽는** 것은 셋뿐이다 —
`lb-inject`(우리가 넣은 내용) · `lb-advice`(손님 자리의 우리 문장) · `lb-tool`(우리 검증도구의 답).
`lb-ask` 는 별도 모델 호출이라 창에 안 들어간다. gold 도구 이름을 언급하는지로 갈랐다.

```
task  base->ours  창에넣은글자  gold언급  gold무관  무관%    advice/inject/tool
028   3/4->2/4    102,210      23,045    79,165    77.5%   15/135/5
019   4/4->3/4     54,019      18,436    35,583    65.9%    9/ 52/4
073   3/4->4/4     52,368      18,436    33,932    64.8%    4/  0/12
040   2/4->0/4     48,342       4,955    43,387    89.8%   28/  0/50
070   2/4->1/4     36,303      23,045    13,258    36.5%    7/  0/5
036   2/4->2/4     31,779      23,215     8,564    26.9%   17/  0/2
058   4/4->4/4     19,979           0    19,979   100.0%    1/  0/7
054   2/4->1/4     11,768       1,328    10,440    88.7%   15/  0/4
043   4/4->4/4     10,643       7,442     3,201    30.1%   16/  0/3
023   4/4->4/4      8,916         162     8,754    98.2%    7/  0/4
049   3/4->4/4      7,310       5,986     1,324    18.1%   14/  0/0
098   4/4->4/4      7,243           0     7,243   100.0%    1/  0/3
081   2/4->0/4      6,749           0     6,749   100.0%    6/  0/6
048   3/4->0/4     14,258       9,674     4,584    32.2%   22/  0/5
007   4/4->1/4      2,315           0     2,315   100.0%    0/  0/1
016   4/4->1/4        685           0       685   100.0%    2/  0/0

base 수준 이상 7개 : 평균 19,748자 (gold무관 11,857 = 63%)
아직 지는     9개 : 평균 30,739자 (gold무관 21,796 = 77%)
                              ×1.6배        ×1.8배
```

## 무엇을 붓고 있나 — 「조언」이 아니라 **기계 출력**이다

```
[lb-inject] GROUP Silver Rewards Card docs=11 rows=2 REPLY  { "txn_d3b830f4a2a4": {
              "base_rate": 1, "exclusion_quote": "", "promo_mult": 1, "promo_window_months": 0, ...
[lb-tool]   RESULT Transactions whose recorded reward does NOT match the expected reward under the
              reward-rate policy (each needs a cash back dispute). **The CORRECT total reward per
              policy is shown next to each id, so you can state it to the customer**
[lb-tool]   RESULT Provisional credit for this dispute (reason unauthorized_fraudulent_charge,
              amount 342.50 ...): **ELIGIBLE - all five conditions of the Provisional Credit
              Eligibility Guidelines hold**
```
`028` 은 **inject 135건 61,393자**(원시 JSON 덩어리), `040` 은 **도구 답 50건 38,634자**(분쟁마다
자격 판정 한 편)이다. 서브호출도 `028` 218회 · `040` 155회 · `019` 110회로 함께 폭증한다.

## 두 가지 결론

**1. 부피가 승패와 같은 방향으로 움직인다.** 지는 쪽이 창 글자 1.6배, gold 무관 글자 1.8배다.
   ⚠ 다만 **반례가 양쪽에 있다** — `016` 은 685자만 넣고도 졌고, `058`(19,979자·무관 100%)과
   `073`(52,368자)은 이겼다. **부피 하나로 설명되지 않는다.** 오늘 잰 잡음(태스크당 ±3)도 감안해야 한다.

**2. 더 무거운 것은 부피가 아니라 종류다.** 위 축자들은 정보가 아니라 **판정**이다 —
   *"ELIGIBLE — 다섯 조건이 모두 성립"*, *"정확한 보상 총액이 각 id 옆에 있으니 손님에게 말하면 된다"*.
   이것은 `[[92-engine-never-judges]]` 가 금지한 자리다: **결정기가 판단을 대신하면 측정 대상이 사라진다.**
   조건부 설명·hidden/disable 은 **부르는 횟수**는 줄였지만(17→15 주입, 호출 대부분 2~5배 감소),
   **한 번 불렸을 때 쏟아지는 양과 그 안의 판단**은 그대로다.

## 검증된 것 · 아직 아닌 것

- ✅ 조건부/숨김·끔은 **동작한다**: 주입 17→15(`verify_identity` 숨김 · `get_interest_correction` 끔),
  호출은 `007` 3→1 · `016` 5→0 · `036` 10→2 · `048` 25→5 · `047` 12→0.
- ❌ **`040` 만 거꾸로 11→50 이다.** 손해 −2 인 태스크이고, 우리 도구 답이 38,634자를 차지한다.
  조건이 이 태스크에서 왜 안 먹는지는 미확인 — 다음 표적.
- ❓ 부피↔승패는 **방향만** 맞고 반례가 있다. 「많이 말해서 진다」는 아직 **가설**이다.
  반증: `028`·`040`·`019` 에서 도구 답의 **판정 문장만** 줄이고(수치는 남기고) 재측정했을 때
  점수가 안 움직이면 부피 가설은 틀린다.
