# Y2-A(폐기 런) 실패 원인 per-step 포렌식 — 인자 수준 (2026-07-31 · 무료)

> 대상 = 폐기된 첫 발사 `bank_y2a_*`의 **완료 15 sim**(전부 trial 0). 도구 = `x21_persim_forensic.py`
> (채점 규약은 C245 축자 재구현 `x12`를 **그대로 재사용**·[[03b]]).
> ⚠**완료 순서 편향** — 빨리 끝난 sim에 치우쳐 있다. 분포를 모집단으로 읽지 말 것([[08]]).
> ⚠이 런은 **V7이 죽은 경로에 있던 스택**이다(`Y2_DESIGN` §12). 그래도 실패 *원인*은 유효하다.

## 0. 분류 분포 (15 sim · 실패 gold action 18건)

`TOP_VALUE` 8 · `NAME_ABSENT` 5 · `PRED_EXTRA_KEY` 2 · `NESTED_VALUE` 2 · `NESTED_SERIAL` 1
db_match × 결과 = (True,pass) 5 · **(True,fail) 2**(=ACTION-basis) · (False,fail) 8

---

## 1. transfer `reason` enum 오선택 — TOP_VALUE/ENUM (004·008)

| task | gold | 모델이 넣은 값 |
|---|---|---|
| 004 | `account_ownership_dispute` | `customer_demands_after_unavailable_offer_refusal` |
| 008 | `customer_demands_after_unavailable_offer_refusal` | (다른 enum) |

★**두 태스크가 서로의 정답을 골랐다.** 즉 enum 선택이 상황-특정적이지 않다. `summary`는 gold가
빈 문자열이고 비교 대상이 아니다(diff에 `reason`만 뜬다) — **오직 enum 하나로 갈린다**.

정책은 이미 "MOST SPECIFIC reason"을 요구하고 A2 `gates.ask`에 그 문구가 있다(C212 B2). 즉
**프롬프트로는 이미 말했고 안 지켜진다** ⇒ 남은 축은 scaffold가 아니라 F3(의미 매칭) 계열이다.
닫힌 술어로 만들 방법이 보이지 않는다 — [[22]] 관점에서 **열린 술어**.

## 2. 도구 미호출 — NAME_ABSENT (005·010·014·016·020)

| task | 안 부른 도구 | 성격 |
|---|---|---|
| 005 | `log_verification` **+** `change_user_email` | 인증 체인 자체가 안 돎(둘 다) |
| 010 | `submit_referral` | user-side action |
| 014 | `transfer_to_human_agents` | agent action(미호출) |
| 016 | `submit_transaction` | user-side action |
| 020 | `give_discoverable_user_tool` | give 자체를 안 함 |

user-side action(010·016)은 **에이전트가 직접 못 부르고 고객이 실행**해야 하는 것이라, 실패 형태가
"안 시켰다"이다 — 오늘 env에서 도출하게 만든 실행-주체 판정(`_exec_side`)이 겨냥하는 지점이다.

## 3. ★`discoverable_tool_name` 접미사 누락 (020) — **새 발견·닫을 수 있음**

```
gold: submit_cash_back_dispute_0589
pred: submit_cash_back_dispute        ← 접미사 _0589 누락
pred: file_reward_dispute             ← 존재하지 않는 이름
```

A2 `discoverable_name_check`는 `name_pattern`(`[a-z_]+_[0-9]{4}`)으로 **정확히 이걸 잡도록** 돼 있다.
그런데 그 레버(`T2_UNLOCK_NAME`)는 **`unlock_discoverable_agent_tool`만** 검사한다. 020의 실패는
**user-side `call_discoverable_user_tool`** 경로라 **적용되지 않는다.**

- **정책 근거 있음**: *"Provide the exact tool name as specified in the knowledge base"* ·
  *"Do not invent or guess"* ⇒ [[23]] 통과.
- **A2 비용 0**: 기존 `discoverable_name_check` 재사용(도구 목록만 확장).
- ⇒ **처방 후보 1순위**: name check를 user-side 디스패처 경로로 확장.

## 4. ★같은 `transaction_id` 반복 — coverage 실패 (020) — **새 발견**

gold는 **서로 다른 4건**의 dispute를 요구한다(`…e403` · `…f506` · `…e410` · `…e411`).
모델은 **전부 첫 번째 txn만** 반복해서 넣었다. 즉 "여러 건"을 한 건으로 축약한다.

C251의 *"022 `transaction_id` REF 오류 11건 · 029 6건"*과 같은 계열이고, RESEARCH_MASTER §1.4의
**F4 coverage**(all/both/every 미완)에 해당한다. `eplan`/`follow_up_chains`가 겨냥하는 축이지만
이 런에서는 닫지 못했다.

## 5. nested 여분 키 `correct_rewards` (020) — V7의 사각

```
gold: {"user_id": …, "transaction_id": …}
pred: {"user_id": …, "transaction_id": …, "correct_rewards": 6300}
```

V7은 **top-level 키**만 본다(`give(discoverable_tool_name)` 서명). 디스패처의 **nested `arguments`
안 여분 키**는 검사 범위 밖이다. 자연스러운 확장 후보지만 **정책 근거를 먼저 찾아야 한다** —
정책이 규정하는 것은 give의 서명이지 내부 도구의 인자 집합이 아니다([[23]]).

## 6. 변형 오선택 — NESTED_VALUE (015)

```
gold: card_name = "Platinum Rewards Card"
pred: card_name = "Crypto-Cash Back Card"
```
RESEARCH_MASTER §1.4의 **F2 변형 선택**(실재하나 틀린 변형). 격리 천장 대비 능력 격차로 귀속된
축이고, scaffold로 닫는 축이 아니다.

---

## 7. 처방 우선순위 (닫을 수 있는 것부터)

| # | 표적 | 근거 | A2 비용 | 판정 |
|---|---|---|---|---|
| 1 | **user-side 경로 name check**(§3) | 정책 축자 有 | **0** | 닫힌 술어 — **구현 권고** |
| 2 | coverage 다건 처리(§4) | F4·기존 eplan | 0 | 기존 레버 조정 |
| 3 | nested 여분 키(§5) | **정책 근거 미확립** | ? | 근거 먼저 |
| 4 | transfer enum(§1) | 프롬프트로 이미 말함 | — | **열린 술어 — 처방 없음** |
| 5 | 변형 오선택(§6) | F2 능력 축 | — | scaffold 축 아님 |

⚠**표본 15 sim·trial 1개**다. 이 우선순위는 **가설**이고, Y2-B 완주 후 같은 포렌식을 전수로
돌려 재확인해야 한다.


---

## 8. ★모집단 규모 (Y1 전수 63 sim · 실패 gold action 202건 · 2026-07-31 추가)

§7의 우선순위는 15 sim에서 세운 **가설**이었다. Y1 전수로 재니 **순위가 뒤집혔다.**

| 분류 | 수 | 비율 |
|---|---|---|
| `NAME_ABSENT` | 77 | **38%** |
| `NESTED_VALUE` | 67 | **33%** |
| `TOP_VALUE` | 36 | 18% |
| `NESTED_SERIAL` | 13 | 6% |
| `PRED_EXTRA_KEY` | 9 | **4%** ← V7의 표적 |

### 8-1. 두 축이 지배한다

**① `transaction_id` 오참조 — NESTED_VALUE 67건 중 64건(96%) = 전체 실패의 32%**

| 내부 키 | 종류 | 수 |
|---|---|---|
| **`transaction_id`** | **REF** | **64** |
| `discrepancy` | NUM | 10 |
| `correct_rewards` | NUM | 6 |
| `card_name` | OTHER | 2 |

C251의 *"022 REF 오류 11건 · 029 6건"*이 전수에서 **64건**으로 확인됐다. **단일 키가 전체 실패의
1/3을 만든다.**

**② dispatcher 체인 미호출 — NAME_ABSENT 77건 중 55건(71%)**

| 안 부른 도구 | 수 |
|---|---|
| **`call_discoverable_agent_tool`** | **43** |
| `unlock_discoverable_agent_tool` | 12 |
| `call_discoverable_user_tool` | 10 |
| `transfer_to_human_agents` | 4 |
| 나머지(각 1~2) | 8 |

C251의 *"032·033 dispatcher 경로 미사용"*이 전수에서 **unlock→call 체인 55건**으로 확인됐다.

### 8-2. 순위 정정

| # | 표적 | 규모 | 상태 |
|---|---|---|---|
| **1** | `transaction_id` 오참조 | **32%** | ★[[18]] **정보-맞춘 격리 프로브가 선결** — 경계(F3)인지 전사-슬립인지 미판정 |
| **2** | dispatcher 체인 미호출 | **27%** | 기존 eplan/followup 축·조정 대상 |
| 3 | `arguments`/`agent_tool_name` 값 | 18% | |
| 4 | V7(give 서명) | **4%** | 오늘 살렸으나 **표적이 작다** |
| ⛔ | 접미사 오류 | **0%** | 설계 폐기(`DISC_NAME_EXACT_DESIGN` 참조) |

⚠**V7에 대한 정정**: 앞서 "give 105회 중 82회(78%) 위반"을 근거로 규모가 크다고 했는데, 그건
**호출 비율**이지 **실패 기여도**가 아니다. 채점 실패에서 차지하는 몫은 **4%**다. 오늘 살린 것은
옳지만(정책 위반은 위반이다) **pass에 미치는 영향은 작다**고 봐야 한다.

### 8-3. [[18]] 의무 — `transaction_id` 축은 아직 분류할 수 없다

전체의 1/3이지만 **원인 분류가 안 끝났다**. [[18]]이 못박은 절차:
*"F3/⋈ 경계 판정 전 무조건 정보-맞춘 격리 프로브(A_minimal vs B_fullctx)"* — C124에서 같은
형태의 wrong-pick이 **경계가 아니라 전사-슬립 + 자기-정박**으로 판명된 전례가 있다.

⇒ 다음 무료 작업 = **`transaction_id` REF 64건의 정보-맞춘 격리 프로브 설계**. 그 결과가
"부하"면 scaffold로 닫히고, "경계"면 map으로 남는다.
