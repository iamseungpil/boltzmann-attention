# t7295 실패 태스크 정밀 포렌식 (2026-08-15)

> 원자료: `sim_results/bank_t7295_b_20260815n_results.json.gz`(영속·tracked) + arm a 라이브
> 스냅샷(`/home/woori/scratch/tau2-bench/data/simulations/bank_t7295_a_20260815n`·41/44 시점).
> 도구: `bank_fail_forensic_all.py`(궤적) · `dbdiff_task.py`(DB·이번에 전수화) · `x324`(계기 검산).
> 대상: **채점된 72 sim · 통과 16**. arm a 3 sim 은 아직 진행 중이라 제외.

---

## §1 먼저 — 판정 계기를 바꿨다

**⒜ `action_match` 는 판정에 쓸 수 없다(C486 확장·기전 확정).**
하네스는 `tasks.py:195` 에서 `tool_args == action_args` 로 비교하는데, discoverable 래퍼의
`arguments` 는 **문자열**이다. 모델이 JSON 을 들여쓰기해 내면 의미가 같아도 False 다.
실측(x324): 050·048 의 7건이 `{"user_id": "584f9c5d00"}` ↔ `{\n  "user_id": "584f9c5d00"\n}`
차이만으로 불일치 — 전부 **성공 실행된 호출**이었다.
⇒ 점수 영향은 없다: banking 97 태스크 중 **DB 기준 88 · ACTION 기준 9**(`task_004·008·012·
014·032·033·034·035·083`)이고 **t7295 태스크엔 ACTION 기준이 하나도 없다.** 오염된 것은
우리 계기뿐이다.

**⒝ 그래서 판정은 DB diff 로 한다 — 그리고 그 계기는 검증됐다.**
`db_match` 는 전체 DB 해시 동일성이라 부분점수가 없다. 72 sim 전수에서
**`db_match` ⇔ `reward==1` 이 100% 일치**했다(불일치 0). 원인 규명은 `dbdiff_task.py` 의
필드-수준 재귀 diff 로 한다.

---

## §2 결과 — 실패는 **네 덩어리**다 (56 실패 sim)

| 군집 | sim | 태스크 | DB 상 실패의 정체 |
|---|---|---|---|
| **A. 계좌 개설/해지 미이행** | **19** | 061(4)·070(4)·055(3)·071(3)·036(3)·063(1)·075(1) | `accounts.data` |
| **B. 수수료 환급 미이행** | **9** | 072(3)·074(3)·073(2)·(072#3 리플레이 불능) | `accounts.current_holdings` + `bank_account_transaction_history` |
| **C. 카드/분쟁 작업 미이행** | **17** | 048(4)·049(3)·087(3)·085(2)·081(2)·050(3) | `credit_card_*` · `debit_card*` |
| **D. 단일 필드 근소 실패** | **8** | 003(3)·099(1)·010(1)·069(3) | 레코드는 만들었는데 **값 하나**가 다름 |

### ★A 군집이 최대다 — 그리고 원인이 균일하다
- **071 (0/3·3 sim 버킷 동일)**: 계좌를 **만들긴 했다**. user·type·상태·날짜 전부 정답.
  틀린 것은 **등급(class) 하나**뿐 —
  gold `business_checking/**Sky Blue**` + `business_savings/**Gold Saver Account**`
  ↔ 우리 `business_checking/**Lime Green**` + `business_savings/**Bronze Saver Account**`.
  ⇒ 도달·형식·집행 전부 성공, **상품 등급 선택**에서만 진다.
- **061 (0/4·4 sim 버킷 동일)**: gold 는 savings(`Silver Plus Account`) **개설** + 기존 계좌
  **해지**(`status CLOSED`·`closure_reason`·`date_closed`·`early_closure_fee_waived=False`).
  우리는 **둘 다 안 했다**(궤적서도 `open_bank_account_4821`·`close_bank_account_7392`
  **NOTCALLED 26**). 종료 본문은 도구를 **부적용이라고 스스로 판정**하고 물러난다.
- **070 (0/4)**: 같은 미이행 + 종료 본문이 *"안전한 링크를 드릴 테니 거기서 신청하세요"* —
  **존재하지 않는 채널을 날조**해 이행을 회피한다.

### ★B 군집 — 가정이 틀렸다
어제까지의 서사는 *"환급 차감을 안 해서 12.00 ↔ gold 14.00"* 이었다. DB 는 다르게 말한다:
```
DIFF .accounts.data.chk_lj82d4f1a9.current_holdings: gold='$127464.00' pred='127450.00'
DIFF .accounts.data.chk_538bfb9cba.current_holdings: gold='$1851.00'   pred='1847.50'
ONLY-GOLD .bank_account_transaction_history…amount: 14.0  (chk_lj82d4f1a9)
ONLY-GOLD .bank_account_transaction_history…amount: 3.5   (chk_538bfb9cba)
```
`pred` 는 **초기값 그대로**다 — 금액이 틀린 게 아니라 **크레딧이 한 건도 안 들어갔다**.
게다가 072 는 gold 가 **두 계좌에 두 건**(14.00 + 3.50)을 요구한다(어제 규명한 $14 는 그중
하나였다). ⇒ **§5 '환급 차감 되살리기'는 상류가 막힌 채로는 무의미하다.** 선결은 산술이
아니라 `apply_checking_account_credit_5829` 가 **착지하지 못하는 이유**다(궤적: unlock 후
호출이 2회 시도되고 전부 실패·`MISS-UNLOCKONLY`).

### ★D 군집 — 가장 싼 자리
- **003 (1/4)**: `credit_card_applications` 가 gold/pred 양쪽에 존재 = **다른 카드로 신청**.
- **099 (3/4)** 1건: referral 레코드 필드 하나. **010 (3/4)** 1건: referral **누락**.
- 즉 이 8 sim 은 도달·집행이 다 됐고 **값 선택 하나**로 0 이 된다.

---

## §3 기각된 가설 둘 (무료로 닫았다)

- **"read-coverage 부족이 db_match 를 깬다"** → 기각. `agent_discoverable_tools` 가 해시에
  들어가는 건 사실이지만, **READ-COVERAGE 만으로 실패한 sim 은 56건 중 0건**이다. 언제나
  실제 write 차이와 함께 나온다. 독립 레버 아님.
- **"우리 층이 궤적을 망가뜨려 채점을 잃었다"** → 미확정, 우리 결함 아님 쪽. 리플레이 불능
  3 sim(072#3·085#2·063#0)의 정체는 **모델이 낸 10-way 병렬 호출에 대해 도구 응답이 순서를
  바꿔 돌아온 것**이고(msg 56 의 호출 순서 ↔ 57·58 응답 순서 역전), 하네스의
  `get_actions_from_messages` 가 **위치 짝맞춤**을 강제해 깨진다. 라이브 하네스는 예외 없이
  0 으로 채점했다. ⇒ 3/72 는 **독립 재현이 안 되는 채점**으로 표시하고, 우리 층 귀속은 하지 않는다.

---

## §4 지속-실패(루프)의 자리

arm a 에서 세 sim 이 **1.4h·1.2h·0.8h** 째 돌고 있다(069·074·085). 어제 073 을 2.4h 물린 것과
같은 계열이다. arm a 전체 `T2_WINDOW` **739회 중 `open=resign` 579회(78%)** —
sim 당 우리-층 마크가 `T2_CLAIMPROV` 254 · `T2_WRITE_SUB` 154 처럼 세 자리로 쌓인다.
⇒ 종료를 막는 것은 손님도 모델도 아니라 **우리 층의 과잉-지속**이라는 §8-2 의 방향이
계량으로 뒷받침된다(단 인과는 미검정 — 격리 필요).

---

## §5 이 포렌식이 지시하는 다음 순서

1. **A 군집(19 sim·최대)** — 071 은 [[62]] 대로 **격리부터**: 정책 문면만 주고 등급을 고르게 하는
   프로브(A_ref vs B_궤적문맥)로 결손이 능력인지 부하인지 먼저 잰다. 061/070 은 종류가 달라
   (**미이행 + 회피 날조**) 따로 잰다.
2. **B 군집** — 산술 복구 이전에 `apply_checking_account_credit_5829` **미착지 규명**이 선결.
3. **D 군집(8 sim)** — 가장 싼 자리. 값 선택 하나.
4. arm a 종료 후 최종 판정(①배선 ②레버 ③성적) 및 원장 C488.

> 계기 변경 2건(§1)은 `dbdiff_task.py` docstring 에 박았다.
