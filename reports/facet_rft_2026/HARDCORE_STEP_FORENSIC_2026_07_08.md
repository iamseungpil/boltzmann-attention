# HARD-CORE 전수 스텝 포렌식 — frontier와의 모든 차이 (2026-07-08·무료)

> 상위 = `RESEARCH_MASTER.md`. C15(집계 분해)의 **스텝-레벨 근인 확정**. gpt-4.1 = 0.
> 방법: task-레벨 짝짓기 → **HARD CORE**(o4-mini ≥3/4 ∧ gpt-4.1 ≥3/4 ∧ ours ≤1/4) **10 task** 전수 스텝 대조
> (gold action열 vs 우리 호출열[ERR 표기] vs o4-mini 통과 호출열). **역방향(ours≥3 ∧ 두 frontier ≤1) = 0 task.**
> 스크립트=scratchpad `gaptasks`·`stepdiff`·`rbw`·`nl_reads`(재현).

---

## 0. 먼저 — ★두 개의 반증 (내 진단 교정)
1. **"우리는 읽기 전에 쓴다" = scaffold 아티팩트.** RBW(write의 order를 사전 조회한 비율): **ours+scaffold 21.8%** ·
   **32B floor 95.5%** · o4-mini 99.8% · gpt-4.1/claude 100%. reads/sim: ours 0.47 vs floor 2.48 vs frontier 2.4–2.9.
   ⇒ `present`가 주문 정보를 **주입**하므로 에이전트가 `get_order_details`를 부를 이유가 없다(우리 sim의 **66%가 0회**·floor는 4%).
   **도구 호출의 부재 ≠ 정보의 부재.** (E3와 정합: 정보 present인데 오선택.)
2. **within-arm 상관이 반대**: 안 읽고 쓴 sim 실패율 **33%** < 읽고 쓴 **43%**. NL도 0-read 5.3% vs ≥1-read 4.5%.
   ⇒ **읽기-부족 인과 가설 기각.**
   ※ 단 **db_match=True 조건부 NL 실패율**은 실재: ours **7.3%** vs o4-mini 3.6% · gpt-4.1 4.5% · floor 6.3%.

## 1. HARD CORE 10 task — 스텝별 최초 발산과 근인
| task | ours(4) | o4/g41 | 우리 스텝 (요지) | o4-mini | **최초 발산 → 근인** |
|---|---|---|---|---|---|
| **t17** | 0 | 4/4 | find_user → **곧바로** modify_address(`"123 Elm St"`) | order 조회 후 `"123 Elm Street"` | **값 충실도**: 주소 **약어**("St"≠"Street") — id 날조 아님·선택 아님 = **N1 verbatim 실패** |
| **t37** | 0 | 4/3 | modify_items **item 5개**(gold 3) + new_id 날조(+1) → ERR → 재시도도 **5개** | item **3개** 정확 | **쓰기 내부의 item-집합 과포함** = **N2 (write-scope)** |
| **t40** | 0 | 4/4 | write 정확·**db_match=True** | 동일 | **NL 보고 실패**("Mastercard로 결제됨"을 안 알림) = 순수 communication |
| **t57** | 1 | 4/3 | 조회 후 **cancel_pending_order** | **write 0** | **gold=write 없음인데 파괴적 cancel** = over-action(치명) |
| **t63** | 1 | 3/4 | 주문 2개 중 1개만 조회 · payment=**gift_card**(gold=paypal) | 둘 다 조회·paypal | **N3 payment-method 선택 오류**(+탐색 부족) |
| **t68** | 0 | 4/4 | order 조회 **0** → 총액 보고 불가 | 조회 후 보고 | **요약 밖 사실 미조회 → NL 실패**(총액 $829.43) |
| **t86** | 1 | 3/4 | modify_user_address만 · **modify_items 누락** | 5주문 조회 후 **둘 다** | **미완(F4 coverage)** |
| **t91** | 1 | 4/3 | **틀린 주문**에 exchange(ERR×2) → return만 성공·exchange **영영 안 함** | 3주문 조회·둘 다 정확 | **⋈ 틀린 주문 → 미완** |
| **t105** | 1 | 3/3 | exchange(gold와 동형 args) → **ERR** · db 불변 | write 0으로 통과 | **도구가 우리 호출 거부**(인자 미세차) + task 모호 |
| **t111** | 1 | 3/3 | new_id 날조(+1) ERR → 수정 성공 · **#W9810810 통째 누락** | 4주문 조회 | **coverage 누락**(+날조 1회) |

## 2. 근인 분류표 (기존 vs ★신규)
| 근인 | 정의 | hard-core 사례 | 전수(C15) | 기존/신규 |
|---|---|---|---|---|
| ⋈ 참조매칭 | 틀린 주문 선택 | t91 | 37 (vs g41 16) | 기존 F3 |
| coverage/미완 | gold write 일부 누락 | t86·t91·t111 | 21 | 기존 F4 |
| over-action(파괴적) | gold=무write인데 상태변경 | **t57** | 5 (frontier 0~2) | 기존([[06]] 게이트금지) |
| NL/communication | db 맞으나 보고 실패 | t40·t68 | 23 (조건부 7.3%) | 기존 |
| operand 날조 | 없는 id 발명 | t37·t111 | 32 call(환경이 거부) | 기존(증상·C12) |
| **N1 값 충실도(verbatim)** | 실재 값의 **약어/변형** 기입("123 Elm St") | **t17** | 미집계 | **★신규** |
| **N2 write-scope(item-집합)** | 한 write 안에서 **요청 안 한 item까지** 포함 | **t37** | 미집계 | **★신규**(over-action과 구분: 같은 write 내부) |
| **N3 payment-method 선택** | 원결제/기프트카드 규칙 위반 | **t63** | other-operand 3~12 | **★신규(명시)** |
| **N4 도구 거부(인자 미세차)** | gold와 동형인데 tool이 ERR | **t105** | all-errored 16 | **★신규** |
| **N5 scaffold의 읽기 억제** | present 주입 → `get_order_details` 0회(66%) | t68 | 상관 미확인 | **★신규(부작용·[P])** |

## 3. 판정
- **단일 상류 원인은 없다.** 10개 hard-core가 **8개 서로 다른 근인**으로 갈린다. C15의 "5조각"이 스텝-레벨에선 **더 쪼개진다**.
- **읽기-부족 가설 기각**(§0). 정보는 present가 준다. **결손은 *정보 부재*가 아니라 *주어진 정보 위의 선택·범위·충실도*.**
- **역방향 0 task**: 우리가 두 frontier를 모두 robust하게 이기는 task는 **없다**. (대칭 크레딧·[[03]]#9 기록.)
- **신규 근인 4개(N1~N4)는 전부 "값·범위의 정밀도"** — 참조매칭(⋈)·보고(NL)와 함께 **frontier가 우리보다 잘하는 것은 "정확히 그 값, 정확히 그 범위, 정확히 그 사실"** 이다. 능력의 이름은 **precision**이지 planning도 reading도 아니다.
- **N5는 우리 제1원리의 또 다른 사례**: present가 grounding을 사고 **읽기 주도성을 판다**(0.47 vs 2.48 reads). 다만 within-arm
  인과는 미확인 → **[P]**.

## 4. 미결·다음
- N1·N2의 **전수 집계**(현재 hard-core 사례만) — 무료.
- N5의 인과: present-off arm(=floor)과 NL 조건부율 비교(6.3% vs 7.3%)는 시사적이나 n 작음.
- t105류 **도구 거부**의 인자 차이 정독(all-errored 16의 근인).
- ⋈은 여전히 최대 조각(vs gpt-4.1 +21) — E3가 "절반은 탐색"이라 했으나 present가 이미 열거 중 ⇒ **탐색도 아님** → 경계 재확인 필요.
