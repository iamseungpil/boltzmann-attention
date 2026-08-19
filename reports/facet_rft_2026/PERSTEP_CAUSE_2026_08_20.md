# 태스크별·스텝별 원인 확정 — t7326 · t7328 (80 sim)

> 사용자 지시 2026-08-20: *"7328과 7326 정밀 포렌식하여 태스크별로 원인 확정하라. 특히 실패 원인들을
> 태스크별로 per step 으로 확정하라. 계속 포렌식에서 정확한 원인을 못찾고 있다."*
>
> 계기 = `x422_perstep_cause.py` + `t2_forensic.mutation_diff`(신규·정본). 산출 = `x422_perstep_cause.json`.
> ⚠**이 문서는 레버를 제안하지 않는다.** [[62]] 순서 ①(격리로 결손 측정) 앞의 **관측 단계**다.

---

## §0 먼저 — 계기가 틀려 있었다 (이것이 "원인을 못 찾던" 이유의 절반)

원인을 세기 전에 [[25]] 대로 계기를 검산했다. 결함 4종이 나왔고 넷 다 **결손을 있는 곳이 아닌 곳에서**
세고 있었다.

| # | 결함 | 실물 오차 |
|---|---|---|
| ⓐ | `t2_forensic.inner_name` 이 `discoverable_tool_name` 을 안 봤다 | `give_/call_discoverable_user_tool` 의 **대상이 통째로 안 보임**. 017 t1 은 `reward=1.0` 인데 gold 3건이 MISSING, 같은 실행 5건이 EXTRA 로 찍혔다 |
| ⓑ | 거절 판정을 substring 으로 했다(`Invalid`·`cannot be`·`Unknown`) | 성공 응답 본문에도 흔한 말이다 — 실측 `cannot be` 217건·`Invalid` 79건이 성공 응답 안에 있다. 성공한 write 가 "막힘"으로 분류됐다 |
| ⓒ | `unlock`·`give`(부여)를 변이로 셌다 | 부여는 DB 를 안 바꾼다. gold 가 부풀고 한 실행이 MISSING+EXTRA 두 칸에 동시에 찍혔다 |
| ⓓ | 중복 호출이 집합 연산에서 사라졌다 | 050 의 승인 **2회**는 key 가 gold 안에 있어 EXTRA 도 WRONGARG 도 아니다 ⇒ "변이 집합 일치"로 보였다(실측 3 sim) |

수리 후 **정본**(`t2_forensic`)에 올렸다: `mutating_tools` · `flat_args` · `mut_key` · `deny_kind` ·
`gold_mutations` · `attempted_mutations` · `mutation_diff`. 거절 주체는 substring 이 아니라 **발신자**로
가른다 — 환경은 `Error:` 로 **시작**하고(tau2 규약), `[READ-FIRST]`·`NOT_VERIFIED` 는 우리 것이며
(`tau2-bench/src` grep 0 확인), *"has not been given to you by the agent"* 는 반대로 **환경 것**이다.

### ★검산 (이 문서의 모든 수치가 이 위에 있다)

```
reward = 1.0   ⟺   변이 집합 일치        80 sim 중 76 일치
어긋난 4건 = 전부 ACTION 채점(004·033) — DB 해시로 채점되지 않는 태스크
```

DB 채점 sim 에서는 **예외 0**이다. 실패 단위는 이제 추정이 아니라 검산된 것이다([[69]]).

---

## §1 결손 사다리 — 250 행

결손 하나마다 궤적에서 **기계 판정**한다(산문 판단 0·[[59]]). 칸마다 처방 축이 다르다.

| 칸 | 뜻 | 수 |
|---|---|---|
| `WRONGARG` | 불렀는데 값이 다르다 | **67** |
| `TRIED-OTHER-ARGS-BLOCKED-env` | 옳은 도구를 틀린 인자로 불러 환경이 거절 | **56** |
| `ARRIVED-NOT-NAMED` | 이름이 도착했는데 산문에도 안 올렸다 | 35 |
| `NAMED-NOT-CALLED` | 이름을 말해 놓고 안 불렀다 | 29 |
| `EXTRA` | gold 에 없는 변이를 성공시켰다 | 29 |
| `DUP` | 같은 gold 호출을 배수로 실행 | 22 |
| `DELIVERY-MISS` | 이름이 궤적에 **한 번도 안 왔다** | 12 |

> ★**지배적 실패는 "무엇을 부를까"가 아니라 "어떤 인자로 부를까"다** — 67+56 = **123/250 = 49%**.
> `BLOCKED-ours`(우리 게이트가 막은 변이)는 **0건**이다. 이 두 런에서 우리 게이트는 write 를 막지 않았다.

### 1b. 값은 **거기 있었다**

WRONGARG 필드 177개의 **우리가 쓴 값**의 출처:

```
tool-result 155 · user-said 10 · self-said 5 · NOWHERE(날조) 5 · 판정불가 2
```

그리고 **gold 값**이 그 호출 이전에 도착해 있던 경우 **124/177**. 미도착 53 중 대부분(094 20 · 093 9)은
**계산으로만 나오는 값**이다(문서에 있을 수 없다). ⇒ 이 축의 실패는 **발명이 아니라 오선택**이다.

---

## §2 태스크별 확정 원인 (20)

pass 는 4 sim(t7326 t0/t1 · t7328 t0/t1) 기준.

| 태스크 | pass | 확정 원인 | 근거(축자·궤적) |
|---|---|---|---|
| **003** | 0/4 | **후보를 못 좁힘 — 우리 도구가 기준을 안 걸었다** | 우리 `check_card_application_fit` 이 Platinum·Gold·Silver **셋 다 eligible** 로 반환하고 note 에 *"not applied (no input given): annual_fee, min_payment_pct, cashback, virtual_card, min_score"*. 손님은 *"biggest spending category is travel"* 을 말했는데 그것이 질의로 **형식화되지 않았다**. 궤적 10 msg — 검색 1회 후 바로 신청 |
| **004** | 1/4 | **닫힌 enum 오선택** (`reason`) | 단일 gold 액션 `transfer_to_human_agents`. 통과 sim = `account_ownership_dispute`, 실패 sim = `customer_demands_after_unavailable_offer_refusal`. `summary` 자유문은 달라도 match=True ⇒ 판정 필드는 enum 하나 |
| **016** | 0/4 | **검색 실패(우리 층)** | gold `submit_transaction`(손님-측 도구) 이름이 궤적에 **0회 도착**. 지배 문서 *"Refer a Friend…"* 는 4 sim 중 **1번만**, score **0.5262** 로 도착 — 1위는 무관한 *"Beige Account Referral Program"* score 9.3180. 4 sim 전부 이관으로 종료 |
| **017** | 1/4 | **손님-측 도구 인자 형식** | `submit_cash_back_dispute_0589` 를 부여 없이/틀린 인자로 호출 → 환경 거절. 부수로 gold 밖 `update_transaction_rewards_3847` 실행(EXTRA) |
| **024** | 1/4 | **닫힌 enum 오선택**(`card_type`) | gold `Business Bronze Rewards Card` ↔ 우리 `Business Platinum Rewards Card`(4/4 동일 방향). gold 값은 호출 前 **도착해 있었다** |
| **033** | 0/4 | **검색 실패(우리 층)** | gold 가 요구하는 `initial_transfer_to_human_agent_1822`·`_0218` 이름이 궤적에 **0회 도착**. 호출은 `verify_identity`·`shell`·`transfer_to_human_agents` 뿐. 5 액션 중 최대 1개만 match |
| **040** | 0/4 | **인자 형식 + enum 오선택** | `file_credit_card_transaction_dispute_4829`(15인자) 를 8회·6회 반복 거절. 이어서 `dispute_reason` 을 `duplicate_charge`(gold) 대신 `unauthorized_fraudulent_charge`→`goods_services_not_as_described`→`refund_never_processed` 로 **돌려가며** 재시도. `card_last_4_digits` 0581↔1652 오선택 |
| **050** | 1/4 | **중복 실행** | `approve_credit_limit_increase_5847` **1회 초과**(3 sim 공통). 값·순서는 맞다 |
| **055** | 0/4 | **enum 오선택 → 하류 전파** | `account_class` gold `Purple Account` ↔ 우리 `Green Fee-Free`/`Gold`/`Blue`/`Gold Plus`(+ `account_type` checking↔savings). 그 결과 새 계좌 id 가 달라져 하류 `deposit_check_3847` 도 어긋난다 |
| **057** | 0/4 | **enum 오선택 → 하류 전파** | `account_class` gold `Blue Account` ↔ `Light Blue`/`Dark Green`/`Green Fee-Free`. `deposit_check_3847` 은 4 sim 중 3에서 **이름조차 도착 안 함** |
| **063** | 0/4 | **enum 오선택** | `Silver Plus Account` ↔ `Silver`/`Gold`, `Silver Rewards Card` ↔ `Bronze Rewards Card` |
| **072** | 0/4 | **도착했는데 실행 안 함 + 값 오차** | `apply_checking_account_credit_5829` 가 4 sim 중 3에서 `ARRIVED-NOT-NAMED`(도착·미지목). 4 sim 중 2는 이관으로 종료. 실행한 1건은 amount 14↔12 |
| **073** | 1/4 | **완결(F4)** | gold 3건(9.5 / 9 / 1.5) 중 실패 sim 은 **1건만** 실행(값은 정확). 통과 t0 는 거래조회 **12회**·실패 t1 은 **3회** ⇒ 수집 자체가 덜 됐다 |
| **074** | 0/4 | **값 오차(계산) + 계좌 오선택** | `amount` gold 27 ↔ 우리 1 / 2.5 / 1.25 / 1.5 (계좌별로 쪼개 넣음) · `account_id` `_1` ↔ `_2/_3/_4`. 8건은 환경 거절 |
| **079** | 0/4 | **over-action + 미종료** | gold 변이 **0건**인데 freeze×3 → unfreeze×3 → close×3 → order×3 실행(EXTRA 26). t7326 t1 은 `context_window_exceeded`, t7328 t0 는 `max_steps`(152 msg) |
| **085** | 0/4 | **두 층이 trial 마다 갈린다** | t0: 17인자 도구에 4~7개만 주고 `atm_location`·`amount_withdrawn`·`dispute_reason` 등 **없는 키 발명** → 전부 거절. t1: **17/17 정확**히 채워 성공하지만 **틀린 거래행**(`btxn_c3d4…` vs gold `btxn_b2c3…`) ⇒ 형식 통과 후 잔여는 ⋈. 부수로 `log_verification` 6회 중복 |
| **093** | 0/4 | **계산(F2b)** | `amount_difference` gold 33 ↔ 우리 45.6 / 30 / 210 / 0 · `expected_apy` 4.275 ↔ 2.95 / 2.75 / 4 / 4.25. 하류 `apply_savings_account_credit` 의 amount 가 그대로 따라 틀린다 |
| **094** | 0/4 | **계산(F2b)** | gold 값 **20/20 미도착** = 문서에 없는 값 ⇒ 계산으로만 나온다. `amount_difference` 140 ↔ 120 / 180 / 92 · `expected_apy` 6.85 ↔ 6.5 / 6.25 |
| **098** | 4/4 | — | 4/4 안정 |
| **100** | 4/4 | — | 4/4 안정 |

### 2b. 결손은 **독립이 아니다**

`055`·`057`·`063` 의 `deposit_check`/두 번째 계좌 결손은 `open_bank_account` 의 `account_class` 오선택의
**하류**다. `093`·`094` 의 `apply_savings_account_credit(amount)` 는 `amount_difference` 계산의 하류다.
⇒ **250 행을 250 원인으로 읽으면 안 된다.** 태스크당 상류 결정점은 대개 **1개**다.

---

## §3 ★새 측정 — 스키마는 **거리**에 따라 죽는다 (부정통제 내장)

발견-도구(`call_discoverable_*`)는 인자 스키마가 **산문으로 한 번** 도착하고 끝난다. 네이티브 도구는
스키마가 도구 목록에 상주한다. 인자 3개 이상 도구의 변이 호출 350건을, **스키마 도착 지점에서 호출까지의
거리**로 갈랐다. 형식 OK = 필수 인자 전부 있고 없는 키 없음.

| 스키마 → 호출 거리 | 형식 OK | 형식 X | OK율 |
|---|---|---|---|
| ≤ 4 메시지 | 62 | 16 | **79%** |
| 5 – 12 | 29 | 11 | 72% |
| 13 – 30 | 39 | 30 | 57% |
| > 30 | 38 | 30 | 56% |
| **네이티브(스키마 상주)** | **95** | **0** | **100%** |

실물: 085 t0 은 스키마가 msg59 에 **17개 파라미터 + enum 목록까지 축자로** 도착했는데(`Tool unlocked:` 응답)
msg88 부터의 호출은 `atm_location` 같은 **없는 키를 발명**했다. 같은 태스크 t1 은 도착 직후(msg76) 호출해
**17/17** 을 맞췄다.

⇒ 이것은 능력이 아니라 **재제시 거리**의 문제로 보인다. 네이티브 도구 **95/95** 가 그 자리에서 부정통제다
(같은 모델·같은 궤적·같은 턴 — 다른 것은 스키마가 상주하느냐뿐).
⚠**단 이것은 관측이지 개입이 아니다.** 인과 주장은 [[62]] ①격리(스키마를 호출 시점에 붙인 팔 vs 안 붙인 팔)
를 거친 뒤에만 할 수 있다. 거리와 형식 실패가 **함께 긴 궤적에서 는다**는 교락(문맥 길이)이 아직 안 갈렸다.

---

## §4 인용 금지 · 미확립

1. **"250 결손 = 250 원인"** — §2b 하류 전파 때문에 틀린 읽기다.
2. **스키마 거리 표를 인과로 인용 금지** — 관측이다(교락: 문맥 길이·태스크 난이도).
3. **`BLOCKED-ours` 0건**은 *"우리 게이트가 무해하다"*가 아니라 *"이 두 런에서 write 를 막은 적이 없다"*는
   좁은 진술이다. read 게이트(`[READ-FIRST]`·`NOT_VERIFIED`)의 손해는 여기서 안 쟀다.
4. 앞 세션의 실패 분류표(WRONGARG 7 · MISSING 7 · EXTRA 2)는 §0 결함 위에 있었다 — **§2 표로 대체**한다.

---

## §5 다음 측정 (레버 아님 · [[62]] ① 순서대로)

| 축 | 격리 물음 | 태스크 |
|---|---|---|
| 인자 형식 | 호출 시점에 스키마를 **재제시**하면 형식 실패가 닫히나 (부하만 제거·정보 동일) | 085 · 040 · 017 |
| 닫힌 enum | 후보 목록을 **그 자리에** 대령하면 옳은 멤버를 고르나 | 003 · 004 · 024 · 055 · 057 · 063 |
| 계산 | 계산 단계만 격리하면 맞나 (F2b — 등대는 결정론 실행이 답이라 적었다) | 093 · 094 · 074 |
| 완결 | 3건 중 1건에서 멈춘 자리에서, 남은 2건을 아는가 | 073 · 072 |
| 검색 | 지배 문서·도구 이름이 왜 안 오나 (질의 형식화 vs 색인) | 016 · 033 |
| 범위 | gold 변이 0인 태스크에서 왜 13건을 쓰나 | 079 |

계기: `x422_perstep_cause.py`(사다리) · `t2_forensic.mutation_diff`(채점 단위) ·
`x422_perstep_cause.json`(80 sim 전수 행).
