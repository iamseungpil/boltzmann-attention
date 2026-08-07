# 101·102 실패 재포렌식 — 결정론이 놓친 네 개의 수 (2026-08-07)

> 이전 분석(핸드오프 포함)에 기대지 않고 원본에서 다시 세운 것이다. 수치는 전부 이 문서에
> provenance와 함께 남긴다. 재현 명령은 §9.
> 대상 = 101 **20 trial**(10 arm × nt2) · 102 **22 trial**(11 arm × nt2), 2026-08-06~07 라이브.

---

## §0 먼저 잡은 계기 결함 2건 — 이게 없었으면 이 문서의 수치가 전부 틀렸다

**(a) 도슨트가 arm 런을 못 봤다.** `x109 --tag`가 `bank_n97_gpu*`만 glob해서 101/102 arm 런
(`bank_f1_gpu0` 등)은 **한 건도 안 실렸다**. "sim 없음"이 부재의 증거처럼 읽혔다. → `--dirs`
추가·사이드카 1:1 이름 매칭(커밋 `1f4de270`).

**(b) 권위본이 아닌 gold를 읽고 있었다.** 같은 id가 두 파일에 있고 **내용이 다르다**:

| 파일 | task_102 `reward_basis` | `nl_assertions` |
|---|---|---|
| `tasks.json` | `['DB']` | 0건 |
| **`tasks/task_102.json`** (권위본) | **`['DB','NL_ASSERTION']`** | **1건** |

런의 `reward_breakdown`에 `NL_ASSERTION`이 실려 있다는 사실이 권위본을 가른다. 먼저 찾은 쪽을
쓰는 도구는 **런이 쓰지도 않은 gold**를 찍는다.

⇒ **`db_match`는 102 관문의 절반이었다.** [S]

---

## §1 채점 구조 (권위본 직독) [S]

`tau2/evaluator/evaluator.py`: `reward = Π(reward_basis에 든 성분)`.

| | 101 | 102 |
|---|---|---|
| `reward_basis` | `['DB']` | `['DB','NL_ASSERTION']` |
| gold 제출 유형 | `Purple Account`, `Sky Blue Account` | `Sky Blue Account` |
| nl 요건 | 없음 | *"agent should recommend TechFlow Labs (not Ember Analytics) … recognizing that Ember Analytics exceeds the 4-year company age limit"* |

`submit_referral`은 **손님 도구**이고, 레코드 id는 `generate_referral_id(user_id, account_type)` —
**인자에만 의존**한다. 그래서 같은 유형 재제출은 DB 무변화다. DB를 깨는 건 제출 *횟수*가 아니라
**서로 다른 유형의 집합**이다. [S]

---

## §2 전수 성분표 [M]

| | 101 (20 trial) | 102 (22 trial) |
|---|---|---|
| reward | **0/20** | **0/22** |
| DB | **0/20** | **3/22** (m1 t0·t1, g1 t1) |
| NL_ASSERTION | — | **0/22** |
| gold `log_verification` ✓ | 19/20 | 21/22 |
| gold `get_referrals_by_user` ✓ | **2/20** | **21/22** |
| gold `submit_referral` ✓ | 3/40 | 16/22 |

DB가 통과한 3건도 reward 0이다. **102는 어떤 런에서도 사정거리 안에 들어온 적이 없다.**

---

## §3 DB 축의 원인은 기계적으로 닫혔다 [M]

제출된 **유형 집합**만으로 `db_match`를 예측했더니 **101 20/20 · 102 22/22 실측과 일치**했다.
즉 DB 축은 "제출 유형 집합 ≠ gold 유형 집합"이 전부다. 초과 유형의 예:

- 102: `Gold Years`(6/6 소진) · `Blue`(5/5 소진) · `Light Green`(3/3 소진) · `Dark Green` · `Purple`
- 101: `Beige` · `Bluest` · `Gold Plus` · `Silver Zoom Card` · `Business Platinum Rewards Card` …

101에서 **`Sky Blue`는 20 trial 중 0건 제출**됐다(gold 요구). `Purple`은 3건.
제출된 이름 중 코퍼스에 없는 것은 `Silver Business` 하나뿐 — **날조가 아니라 엉뚱한 실재 상품으로
새는 것**이다.

---

## §4 결정론으로 닫히는 네 개의 수, 그리고 실측 [S 진리 / M 실측]

무엇을 제출할 수 있는가는 **네 개의 수**로 끝난다. 넷 다 (원장 행) × (정책 상수)의 함수이고 대화의
어떤 말에도 의존하지 않는다 — [[22]]의 **닫힌 술어**, [[50]]의 이관 3조건(유한·정책유계·전수열거) 충족.

**정책 축자 출처** (`documents/`):
- *"at most 2 referral bonuses in any rolling **9-day** window … applies across all checking account types"*
  (신용카드용 **7-day** 변종이 코퍼스에 따로 있다 — 디코이)
- Sky Blue: *"Confirm your company is within 4 years of formation"* · *"Annual limit: 8 referrals per year"*
- 유형별 상한은 코퍼스가 **최소 7개 문형**으로 쓴다(`Annual limit:` · `up to N … per calendar year` ·
  `Annual maximum` · `Maximum referrals` · `Maximum per year` · `| Annual cap |` · `| Maximum referrals per year |`).

**기계 진리 (102, 오늘=2025-11-14, 원장 29행):**

| 유형 | 사용 | 상한 | 연간 잔여 |
|---|---|---|---|
| Sky Blue | 7 | 8 | **1** |
| Gold Years | 6 | 6 | 0 |
| Blue | 5 | 5 | 0 |
| Light Green | 3 | 3 | 0 |
| **Dark Green** | 5 | **6** | **1** |
| Purple | 1 | 6 | 5 |
| Lime Green | 2 | 12 | 10 |

창 안(≤9일) 1건(11/10 Purple) ⇒ **창_잔여 = 1**.
(101은 창 안 0건 ⇒ 창_잔여 2. 그래서 101의 gold가 2건이다.)

> ⚠**Dark Green 상한은 6이다**(문서 축자 `Maximum per year: 6 referral bonuses`). 태스크 notes는
> "Dark Green (5/5)"라 적지만 그건 gold-side 주석이고, 에이전트의 권위는 문서다([[23]]).
> ⇒ 102에서 **연간 상한만으로는 답이 하나로 좁혀지지 않는다.** 답을 정하는 건 **창_잔여=1**이다.

**모델이 이 수들을 다뤘는가** (문형 결속 채점 · x121):

| | 101 (20 trial) | 102 (22 trial) |
|---|---|---|
| 원장 **계수** 주장 | **0건** (전 trial) | 35✓ / 8✗ |
| 정책 **상한** 주장 | 2✓ / 0✗ (1 trial) | **2✓ / 6✗** |
| **소진** 판정 | 0건 | 31✓ / **14✗** |
| **9일 창** 언급 | **2/20 trial** | **3/22 trial** |

**읽는 것은 되고, 상수와 비교하는 것이 안 된다.** 102는 원장을 세는 데는 대체로 성공하고
(*"you have already made 6 referrals for the Gold Years Account, 5 … Dark Green, and 7 … Sky Blue"* —
전부 정확), **상한에서 무너진다**. 축자 실패 사례:

- *"The limit for the Sky Blue Account is **7** referrals per year"* — 자기가 센 사용량을 상한으로 삼았다(실제 8).
- *"7 referrals for the Sky Blue account type. **Since you have already reached the limit**…"* — 계수는 맞고 비교가 틀렸다.

그리고 **답을 정하는 수(창_잔여)는 19/22에서 아예 언급되지 않는다.**

---

## §5 우리 층이 통제하지 못하는 채널 — 산문이 곧 write다 [M]

`submit_referral`은 손님 도구다. 우리 게이트는 **에이전트의 호출만** 막을 수 있고, 손님의 write는
보지도 못한다. 그런데 DB 채점은 그 write로 결정된다. 손님은 무엇을 근거로 인자를 고르는가:

| | 제출 | 직전 에이전트 답변에 그 유형이 있었는가 |
|---|---|---|
| 101 | 87건 | **87/87 (100%)** |
| 102 | 63건 | **61/63 (97%)** |

**산문이 곧 write다.** 우리 ORDER 문구는 이미 *"Telling them to run it earlier is the same as doing
it early yourself"* 라고 **말한다** — 그런데 **집행은 도구 호출에만 걸려 있다.** 답변에 상품을
나열하는 행위 자체는 어떤 게이트도 지나지 않는다. 101의 어떤 답변은 **상품 15종을 나열**했다.

---

## §6 102의 NL 축: 피연산자가 도달 불가능하다 [S]

- `Ember Analytics` / `TechFlow Labs`는 **task JSON 밖에 존재하지 않는다** — `db.json` 0회, 문서 0회.
- 회사 설립일을 돌려주는 READ 도구는 **없다**(도구 전수 확인). 회사 테이블 자체가 없다.
- 102에서 회사 나이에 대한 유일한 진술은 **손님의 것이고, 그것이 틀렸다**(주장 3년 / 실제 5년).

⇒ **"Ember는 5년"은 어떤 경로로도 유도할 수 없다.** 판정문 22건 전부가 *"never mentioned or applied
the 4-year company age limit"* 계열이다. TechFlow가 실제로 선택된 2건조차 *"selected by the customer,
not because the agent identified Ember as ineligible"* 로 기각됐다.

그래서 이 축을 **사실을 알아내서** 이기는 길은 없다. 남는 길은 하나 — 정책 조건의 피연산자가
**고객 주장뿐이면 그 조건을 미확립으로 취급**하는 것이다([[25]]·사용자 지시 *"고객이 레코드 내용이나
정책을 얘기하면 권위가 아니다"*). Sky Blue 문서가 스스로 *"Gather company formation documents"* 라고
요구한다. 두 후보 다 미확립이지만 문턱까지의 여유가 다르다(주장 3년 vs 2년, 문턱 4년) ⇒ 미확립일 때
**문턱 여유가 큰 쪽**이 남는다 = TechFlow. gold를 보지 않고 도달하는 유일한 경로다.

> 우리 `[SOURCE]` 게이트는 **이미 이 일을 한 번 하고 있다**: 101 궤적에서
> *"you stated 4 thing(s) as fact that the policy documents decide, without having the document:
> 'maximum of 11 referrals per year' …"* 로 **날조된 상한을 잡았다**. 대상이 '상한'에만 걸려 있고
> **'자격 조건의 피연산자'로 확장되어 있지 않을 뿐이다.**

---

## §7 101과 102는 **다른 실패**다 [M]

같은 손님·같은 원장·같은 게이트인데 갈라진다:

| | 101 | 102 |
|---|---|---|
| 원장 조회 | **2/20** | 21/22 |
| 원장 계수 발화 | **0 trial** | 14 trial |
| 실패 위치 | **원장에 도달하기 전** | 원장은 읽고 **상수와 비교하는 데서** |

우리 `[ORDER]` 문구는 두 태스크에서 **바이트 동일**이고 둘 다 6회 발화했다. 즉 게이트 부재가 아니다.
차이는 손님이다 — 102 손님은 referral 이력을 **먼저 틀리게 말해** 검증을 유발하고, 101 손님은
*"I can't even count them all"* 이라 말한다. **우리 게이트는 조회를 '나중에 할 것'으로 미루기만 하고
(*"Still outstanding after that (do not do them in this reply)"*), 다시 강제하지 않는다.**
101은 그 뒤로 조회 없이 끝난다.

---

## §8 그래서 결정론의 실패는 어디인가

네 자리다. 전부 도메인-일반이고 전부 정책 축자에서 나온다(gold 경유 0).

| | 결정론이 owner여야 할 판정 | 지금 | 실측 근거 |
|---|---|---|---|
| **L1** | `연간_잔여(유형) = 상한 − 계수` — 잔여 0이면 그 유형은 후보에서 제외 | LLM이 상한을 날조/오비교 | 상한 2✓/6✗ · 소진 31✓/14✗ |
| **L2** | `창_잔여 = 2 − |9일 내 행|` — 이 수가 제출 가능 건수의 상한 | 19/22에서 미언급 | 제출 0~5건으로 흔들림 |
| **L3** | 답변에 나열한 상품 집합 = 손님이 실행할 write 집합 → **산문을 게이트 대상으로** | 도구 호출만 게이트 | 87/87 · 61/63 |
| **L4** | 정책 조건의 피연산자가 고객 주장뿐이면 **미확립** | `[SOURCE]`가 '상한'에만 적용 | NL 0/22 · Ember 도달 불가 |

**[[05]] 경계 점검**: 넷 다 (1) 모델 학습 불변 (2) scaffold 엔진은 **도메인-일반 술어**만 추가
(원장 유형별 계수·시간창 계수·답변 내 상품 열거·조건 피연산자 출처) — banking 리터럴 0
(3) A2에는 `상한 문형 7종`·`윈도 선언`만 얹힌다(둘 다 정책 문서 축자 출처). retail·telecom으로
ABox-swap 시 같은 술어가 그대로 선다.

**부정통제 의무([[57]])**: L1~L4 중 무엇을 켜든, *무내용 재시도* arm 없이 효과를 주장하지 않는다.
현재 22 trial에서 같은 구성이 DB 2/2 통과와 0/2 실패를 모두 냈으므로 nt=2 비교는 [D]다.

---

## §9 재현

```bash
cd /home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2
P=/home/woori/venvs/seka_env/bin/python
D102=bank_f1_gpu0,bank_g1_gpu0,bank_m1_gpu0,bank_r1_gpu0,bank_q1_gpu0,bank_c1c_gpu0,bank_c1b_gpu0,bank_c13_gpu0,bank_arb1_gpu0_102,bank_arb2_gpu0_102,bank_arb3_gpu1
D101=bank_f1_gpu1,bank_g1_gpu1,bank_q1_gpu1,bank_m1_gpu1,bank_c1c_gpu1,bank_c1b_gpu1,bank_c13_gpu1,bank_arb3_gpu0,bank_arb2_gpu1_101,bank_arb1_gpu1_101
$P x120_referral_forensic.py --tasks task_102 --dirs $D102   # 성분·제출 원장·유형 집합
$P x121_ledger_reduction_audit.py --tasks task_102 --dirs $D102 --show  # 계수/상한/소진 채점
$P x122_prose_is_the_write.py --tasks task_101 --dirs $D101  # 산문 = write
```

**증거 등급**: §1·§4 진리·§6 = [S](권위본·정책 문서·도구 전수) · §2·§3·§4 실측·§5·§7 = [M](전수 궤적 대조).
레버 효과는 아직 아무것도 [M]이 아니다.
