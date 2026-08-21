# 1단계 20 태스크 전체 런 계획 (2026-08-21)

> 상위 = `RESEARCH_MASTER.md` · 계기 = 사용자 지시 축자:
> *"원래 96 태스크의 1단계 20 태스크 돌리기로 한거 아닌가? 1단계 전체 태스크 돌리기 위해서 필요한
> 계획 준비하라"* · *"실패 원인을 정밀하게 per step 포렌식하라. pass 를 올릴 방법이 없는지 확인하라"*

---

## §0 ⛔먼저 자백 — t7333 은 1단계가 아니었다

1단계 정본(= t7326/t7328 이 쓴 집합) 20 태스크:

```
003 004 016 017 024 033 040 050 055 057 063 072 073 074 079 085 093 094 098 100
```

t7333(합성 A/B)이 쓴 8 태스크는 `003 024 025 001 070 055 047 063` 이고 **1단계에 속하는 것은
넷뿐**(003·024·055·063)이다. 레버가 발화할 수 있는 자리를 census 로 골랐기 때문인데, 그 선택은
**1단계 성적으로 읽으면 안 된다**. 특히 024 가 6/20(30%)의 비중을 차지했고 1단계에서는 1/20 이다.
⇒ **C579 의 +4 는 "t7333 태스크 집합에서의 +4"** 이고 1단계 주장으로 승격할 수 없다.

---

## §1 실패 per-step 포렌식 — 레버가 발화했는데도 왜 실패했나

`t7333_failure_forensic.py`(변이 집합 = reward 의 실패 단위·[[69]]). 두 무리로 **완전히 갈린다**.

### ⒜ 실패한 결정이 **레버가 닿지 않는 다른 인자**다 (047·055·063·070)

| 태스크 | 실패 변이 | 모델이 고른 것 ↔ gold |
|---|---|---|
| 055 | `open_bank_account_4821.account_class` ×2 + `deposit_check_3847` MISSING | Green Fee-Free ↔ **Purple** · Gold ↔ **Silver Plus** |
| 063 | `open_bank_account_4821.account_class`(savings) 양 trial MISSING | Platinum/Bronze ↔ **Silver Plus** |
| 070 | `open_bank_account_4821.account_class`(business_checking) | Lime Green·Cobalt Blue ↔ **Sky Blue** |
| 047 | `log_credit_card_closure_reason_4521` **env 가 거절**(BLOCKED) · `close_credit_card_account_7834` EXTRA · `apply_for_credit_card` MISSING | 해지 흐름 |

★`account_class` 는 `spend_category` 와 **같은 모양의 결정**이다 — 문서가 정의하는 닫힌 목록에서
하나 고르기. 엔진의 배달 루프는 **이미 `catalog_arg_docs` 의 모든 인자를 돈다**. 즉 코드가 아니라
**선언이 없어서** 안 닿았다.

### ⒝ 카드 선택 자체가 틀렸는데 **범주와 무관**하다 (003·001)

* **003**: gold = `Silver Rewards Card`. ctl t0 → `Gold`, treat t0 → `Platinum`. 범주는 두 팔 다
  `travel` 로 **맞았다**. 실패는 **다중 제약 하의 비교**(FX 0 · 구매보호 · 한도 ≥ 100k · 연회비 최소).
* **001**(1단계 밖): treat t0 → `Silver`(gold `Gold`), 그 sim 은 `spend_category` 가 **없다**
  ⇒ **−1 은 값 레버 쪽 손실**로 귀속된다(배달이 손대지 않은 sim).

---

## §2 pass 를 올릴 자리 — 1단계 전수 census

`stage1_failure_census.py` · 바탕 = `t7328`(1단계 20 × nt2 = 40 sim) · **6/40 pass**.

| 도구 | 실패 종류 | 건수 | 태스크 |
|---|---|---|---|
| `file_credit_card_transaction_dispute_4829` | MISSING 14 · WRONGARG 12 | **26** | 040 |
| `apply_checking_account_credit_5829` | MISSING | 13 | 072·073·074 |
| `log_verification` | **DUP** | 11 | 016·040·079·085 |
| `open_bank_account_4821` | MISSING 8 · WRONGARG 7 | **15** | 055·057·063 |
| `submit_interest_discrepancy_report_7294` | WRONGARG | 6 | 093·094 |
| `apply_for_credit_card` | MISSING 4 · WRONGARG 4 | 8 | 003·024 |

**WRONGARG 가 걸린 인자 = 다음 레버의 주소**이고, 축이 셋으로 갈린다:

| 인자 | 건수 | 축 | 배달(선언)이 닿나 |
|---|---|---|---|
| `open_bank_account_4821.account_class` | 7 | **카탈로그**(문서가 목록을 정의) | **닿는다** |
| `apply_for_credit_card.card_type` | 4 | **카탈로그** | 닿는다(단 003 은 다중 제약 비교) |
| `dispute.transaction_id` · `purchase_date` · `card_last_4_digits` | 12·12·7 | **레코드 참조**(고객 DB) | ⛔**안 닿는다** — C405ⓔ 경계상 고객 DB 는 모델 몫. `ref_iso` 축이다 |
| `dispute.dispute_reason` · `contacted_merchant` · `card_action` | 10·8·4 | 카탈로그/발화-유래 혼재 | **미판정** — 격리로 갈라야 한다 |
| `report.expected_apy` · `actual_apy` · `amount_difference` | 6·6·6 | **계산** | 값 레버 축(C562) |
| `log_verification` DUP | 11 | **중복 실행** | 다른 축(050 형) |

⇒ 정직한 크기: **배달 선언이 사는 것은 `account_class` 15 건(3 태스크) + `card_type` 8 건(2 태스크)**
이고, 최대 블록(dispute 26 건)은 **참조·계산 축이라 배달이 못 산다**.

---

## §2b ⛔§2 정정 — 궤적 전수 포렌식 결과 (2026-08-21 오후 · C583 · `x458`)

§2 의 census 는 **호출 이름을 그대로** 셌고, 이 도구들은 발견 래퍼를 탄다. 정본
`t2_forensic.mutation_diff`(래퍼 해제·GRANTS 제외·DUP 계수)로 다시 세니 세 군데가 틀렸다.

| §2 가 적은 것 | 실제 (40 sim 전수) |
|---|---|
| dispute 26 = `transaction_id`·`purchase_date`·`card_last_4_digits` **참조 축**, 우리 층이 안 닿음 | ⛔**그 셋은 불일치 0** — 맞히고 있다. 실제 실패는 `eligible_for_provisional_credit` **12** · `contacted_merchant` **7** · `card_action` **4** = **정책 판정**이고 KB 에 문서가 있다(`Provisional Credit Guidelines`·`Regulation E`) ⇒ **A3 문서 축** |
| `apply_checking_account_credit` **MISSING 13** (072·073·074) | ⛔**대부분 불렀다**(래퍼 경유·072 wrapper 3 · 073 wrapper 4·2). 진짜 미호출은 **074 두 sim**뿐이고 끝말이 *"The corrections have been applied"* = **완료 사칭**(knowing-doing) |
| `log_verification` **DUP 11** = 기전 미확정 | ⛔**우리 층이 만든다** — 085 반복 4건 중 3건이 우리 문구(`… you may now call log_verification`) 직후, 040 은 `[DUPLICATE-READ]` stub 을 실패로 읽고 재시도 |

### ★§2 가 세지 않은 블록 — `BLOCKED 105`

시도했으나 거절당한 변이가 **105건**이고 **거절자는 전부 `env`**(우리 층 0).

```
057=41  040=30  085=13  017=8  079=7  094=3  055=2  063=1
open_bank_account_4821                    env 41   "Error: Account eligibility requirements not met."
file_credit_card_transaction_dispute_4829 env 30   그중 "Unknown discoverable tool" 16 · "already been filed" 7
file_debit_card_transaction_dispute_6281  env 13   "Invalid arguments" 6
```

**057 의 41 건이 전부 자격 미달이다.** 그것이 같은 날 감사(`x453`)가 클래스별 값·문서 id·오프셋까지
확보한 축(`minimum_opening_deposit`·`ongoing_minimum_balance`·`minimum_balance_requirement`)이다
⇒ **057 은 A3 자격 축의 정면 표적**이고, §5 가 057 을 *"다른 MISSING 도 함께 안고 있다"* 로만
적어 둔 것은 과소평가였다.

### 이 정정이 §4 계획에 미치는 것

* **P1 의 기대 효과가 커진다** — A3 자격 문턱이 057(41 BLOCKED)·055·063 을 겨눈다.
* **P2 로 미룬 dispute 26 은 P1 과 같은 축**(문서)일 수 있다 — 참조 격리가 아니라 정책 문서 전달.
* **DUP 은 레버가 아니라 우리 문구 수리**다 — 유료 런 없이 고칠 수 있다.
* 074 의 완료 사칭은 **처방이 없는 축**([[46]] 라이브 A/B null) — 계획에서 기대치를 빼야 한다.

## §3 §[[05]] 세 질문 (설계서 상설·[[17]])

1. **도메인-특화를 순증시키나?** 엔진은 그대로다 — `catalog_arg_docs` 에 `account_class` 선언을
   **더할 뿐**이고 루프는 이미 일반이다. 선언 출처는 **문서 제목·정책 축자**여야 하고 gold 는
   보지 않는다([[23]]). 못 대는 값은 **넣지 않는다**.
2. **유동적 판단을 동결하나?** 아니다 — 판단은 격리 서브(=같은 모델)가 하고 엔진은 읽어 넘기고
   인용 실재만 검산한다.
3. **scaffold 가 도메인 행동을 수행하나?** 정책 문서 읽기뿐(C405ⓔ 확정 경계). 고객 DB 는 안 읽는다
   — 그래서 `transaction_id` 류에는 **의도적으로 안 닿는다**.

---

## §4 계획 (순서를 바꾸지 말 것)

### P0 — 무료 선결 (지금)
1. **t7334(값-단독) 판정** — t7333 의 +4 를 배달/값에 귀속. `val < 6` 이면 합성이 일하는 것.
2. **`account_class` 배달 선언 초안** — KB 문서 제목만으로 계열별 문서 id 색인 작성([[23]] 준수).
3. **격리 프로브**(`x448` 관용구 재사용): 무문서 ↔ 선언 문서 ↔ 부정통제 ↔ 검색(bm25/dense).
   ⛔여기서 안 갈리면 **선언하지 않는다**([[62]]).

### P1 — 1단계 전체 A/B (유료)
```
태스크 : 1단계 20 정본 · nt=2 ⇒ 팔당 40 sim · 두 팔 80 sim
팔     : ctl(전부 unset) ↔ treat(P0 이 통과시킨 조합)
배치   : ctl=8140 · treat=8141 병렬 · GO_CONCURRENCY=1
바탕   : ⛔t7328 을 ctl 로 재사용 금지 — sha `275bb222` 로 엔진이 달라 A/B 가 성립하지 않는다
시간   : t7333 실측 20 sim/팔 ≈ 5.75h ⇒ **40 sim/팔 ≈ 11~12h**(하룻밤)
판정   : reward 뿐 · 판정선 C483 ±4/40 ⇒ **|차| < 4 는 null**(40 sim 규모의 원잡음 그대로)
의무   : 전체 짝 · **태스크별 부호표** · 무엇을 팔았나(조회·쓰기·날조) · 레버별 발화율
동결   : 스모크 뒤 `freeze.py --on` · 종료 시 off · gz 영속 + `git ls-files` tracked 확인
```

### P2 — 그다음 축 (배달이 못 사는 자리)
* `log_verification` **DUP 11** — 중복 실행 축(050 형). 무료 포렌식으로 먼저 기전 확정.
* dispute 26 건 — **참조-격리**(`ref_iso`) 축. C124/C125 계보.
* `expected_apy`/`amount_difference` — **값 레버**(C562) 적용 대상. `compute_ops` 선언 확인부터.

---

## §5 ⛔이 계획이 못 사는 것 (미리 적어 둔다)

* 1단계 바탕이 **6/40** 이라 바닥이 낮다. 배달이 `account_class` 를 전부 고쳐도 상한은
  **3 태스크**(055·057·063)이고, 각 태스크는 **다른 MISSING 도 함께** 안고 있다(055 는
  `deposit_check_3847` 도 빠진다) ⇒ **변이 하나를 고쳐도 그 sim 이 pass 로 가지 않을 수 있다**.
* 040 은 1단계 실패의 최대 블록인데 **배달 축이 아니다**.
* t7333 의 +4 는 태스크 집합이 달라 **1단계 예측치가 아니다**.
