# x807 — A2 `scaffold_get_tools` 전수 KB 대조 감사 (banking_knowledge · 2026-09-06)

**계기**: `get_reward_discrepancies` 가 우리 도구임이 확인되자([[23]] 감사 의무 발생) 10개 전수로 확대.
**방법**: 각 도구 `op` 의 **리터럴 상수**를 뽑아 `data/tau2/domains/banking_knowledge/documents/*.json` 축자와 1:1 대조.
**출처 제한**: 대조 대상은 KB 문서·`db.json` 스키마뿐. `tasks.json`/gold 는 **판정 근거로 쓰지 않았다**([[23]]).

---

## 1. 결론표 (10 도구)

| # | 도구 | 리터럴 칸 | 판정 |
|---|---|---|---|
| 0 | `get_reward_discrepancies` | 6 | ❌ **2칸 오류** (아래 §2) |
| 1 | `verify_identity` | 0 | ❌ **술어 결함**(기존 확인·§4) |
| 2 | `check_rebate_qualification` | 0 | ✅ 파라미터 구동 |
| 3 | `get_correct_savings_apy` | 0 | ✅ 리듀서만 |
| 4 | `get_interest_correction` | 1 (`1/12`) | ✅ KB *"Interest is credited monthly"* |
| 5 | `check_card_closure_eligibility` | 1 (`<=0`) | ✅ 구조 |
| 6 | `check_cli_eligibility` | 36 | ⚠ **경계 1칸 오류 + 커버리지 1칸**(§3) |
| 7 | `check_card_application_fit` | 13행 | ✅ 요율 정확(Platinum **10.0**) |
| 8 | `get_atm_fee_discrepancies` | 18 | ✅ **전수 일치** |
| 9 | `get_checking_atm_fee_totals` | 40 | ✅ **전수 일치** |

---

## 2. ❌ `get_reward_discrepancies` — 요율표 2칸

| 카드 | A2 선언 | KB 축자 | 출처 | 판정 |
|---|---|---|---|---|
| Gold Rewards Card | 2.5 | "Cash back on all purchases: **2.5%**" | `doc_credit_cards_gold_rewards_card_*` | ✅ |
| Bronze Rewards Card | 1 | "Understanding Your **1%** Cash Back Rewards" | `…bronze_rewards_card_002` | ✅ |
| Diamond Elite Card | 5 | "Earning **5%** Cash Back on All Purchases" | `…diamond_elite_card_*` | ✅ |
| EcoCard Green / 기타 | 5 / 1 | "Green: **$5.00** … Other: **$1.00** per dollar" | `…ecocard_002` | ✅ |
| **Platinum Rewards Card** | **4** | "Platinum Rewards Card: **Earning 10% Cash Back**" | `…platinum_rewards_card_002` | ❌ |
| **Silver Rewards Card** | **4 (평률)** | "How to Earn **4%** Cash Back on **Travel and Software**" + 기타 1% | `…silver_rewards_card_002` | ❌ |

- **Platinum 4 는 사업자 카드의 값**이다 — `doc_business_credit_cards_business_platinum_rewards_card_002` 축자 *"Business Platinum Rewards Card: Earning **4%** Cash Back"*. 소비자 카드 이름에 사업자 값을 붙였다.
- **Silver 는 카테고리 조건부**인데 평률로 눌렀다. 비-카테고리 거래(Groceries/Shopping 등)의 정답은 1%.
- **자기모순 증거**: 같은 A2 의 `check_card_application_fit` 카탈로그는 이미 `Platinum … cashback 10.0` · `Silver … base_cashback 1.0, category_rates {travel:4, software:4}` 로 **정확히** 적어 두었다. 두 도구가 같은 카드에 다른 값을 쓴다.

### 2-1. 파급 (실측)
우리 팔 sim **37,415** · 이 도구 호출 **838회**. 인자에 실린 거래 카드 분포:

| 카드 | 건수 | 우리 표 | 결과 |
|---|---|---|---|
| Silver Rewards Card | **2,258** | 4 (평률) | 비-카테고리 거래 **오탐** |
| Business Silver Rewards Card | 1,991 | (없음) | abstain |
| EcoCard | 1,809 | 5/1 | 정상 |
| Gold Rewards Card | 1,044 | 2.5 | 정상 |
| Business Platinum | 677 | (없음) | abstain |
| Crypto-Cash Back | 363 | (없음) | abstain |
| Diamond Elite | 307 | 5 | 정상 |
| Business Bronze | 293 | (없음) | abstain |
| **Platinum Rewards Card** | **69** | 4 | **전건 오탐** |
| Business Gold | 36 | (없음) | abstain |

### 2-2. ⛔ 더 큰 결함 — **이름 절단이 abstain 을 무력화한다**
엔진은 정상적으로 기권한다(`t2_compute.py` 축자: `if en is None or act is None: skipped += 1; continue`).
그런데 **모델이 `"Business Silver Rewards Card"` 에서 `Business ` 를 떨어뜨려 넘긴다**.

`task_020` 실측 — db 상 해당 사용자의 거래는 **전부 `Business Silver Rewards Card`** 인데,
인자 문자열 집계는 정확한 이름 **679** vs 절단형 `Silver Rewards Card` **571**.

절단되면 abstain 이 아니라 **소비자 Silver=4** 에 걸린다:

| 거래 | 카테고리 | 금액 | 기록 | 진짜 기대(Business Silver 10%) | 우리 기대(절단→4) | 결과 |
|---|---|---|---|---|---|---|
| `txn_a8f1c2d3e403` | Travel | $315.00 | 3,150 | 3,150 ✅ **정상** | 1,260 | **오탐** |
| `txn_a8f1c2d3e412` | Travel | $342.00 | 3,420 | 3,420 ✅ **정상** | 1,368 | **오탐** |

실제 반환 문면(축자): `Transactions whose recorded reward does NOT match the expected reward (these require a cash back dispute): txn_a8f1c2d3e401, txn_a8f1c2d3e402, txn_a8f1c2d3e404, txn_a8f1c2d3e405, …` — 정상 거래가 목록에 들어 있다.

⇒ **"우리가 없던 도구를 줘서 벌었다"는 `020` 서사는 이 값 위에 얹혀 있었고, 지금 근거가 없다.**

---

## 3. ⚠ `check_cli_eligibility`

임계값 3표(36칸)는 KB `doc_credit_cards_credit_card_account_logistics_005/_006` 과 **전수 일치**
(연령 120/90/60 · 쿨다운 120/90/60 · 이용률 70/80/90% · 연속납입 6/3/3).
티어 매핑도 KB 축자 *"Premium-tier and above (Gold, Business Gold, Platinum, Business Platinum, Diamond Elite Card)"* 와 일치.

**결함 2건**:

1. **경계 부호** — 우리 op: `current_balance > credit_limit × rate` → NOT_ELIGIBLE.
   KB 축자: *"Utilization **at or above** the threshold does not qualify"* / *"If your premium-tier utilization is **exactly 90%**, it does not meet the 'below' requirement"*.
   ⇒ `>` 가 아니라 **`>=`** 여야 한다. 정확히 임계에 걸린 건을 우리는 ELIGIBLE 로 발급한다.
   (연령·쿨다운·납입은 KB *"on or after the day the minimum is reached"* 와 `<` 가 정합 ✅)

2. **커버리지** — `Crypto-Cash Back` 누락. KB 축자 *"Entry Tier (Bronze Rewards Card, EcoCard, Business Bronze Rewards Card, **Crypto-Cash Back Card**)"*. `default: null` → abstain 이라 오답은 아니나 db 15건 미커버.

---

## 4. ❌ `verify_identity` (기존 확인 재게)

`op: {"op":"match_verdict","a":"provided","b":"record","threshold":2}` — **양변을 모두 모델이 채운다**.
매처가 역할-무시 평문 부분문자열(`t2_gate_patch.py:593` `if s.lower() in ctx: return True`)이라
**레코드 에코가 항상 통과**한다. `task_004` 실패의 직접 원인(위조 `VERIFIED` → gold 전제 소멸).

---

## 5. ✅ 검증 통과한 것 (재유도 금지 · [[40]])

- `get_atm_fee_discrepancies`: 8 계좌등급 × {out-of-network, foreign} + 환급 상한 2 = **18칸 전수 일치**.
  Blue min(1%,3.00) · Bluest 2.00/0 · Green 3.00/max(3%,5.00) · Purple 2.50/0 · Dark Green max(1%,1.50)/min(2.5%,6.00) ·
  Evergreen min(1%,2.50)/max(2%,3.00) · Light Green 4free→1.50 / 계단(≤100→2.00, ≤300→3.50, >300→5.00) ·
  Light Blue 2free→2.50 / 2free→4.00. 환급 상한 Bluest $50 · Purple $30.
  ★Light Green 계단의 경계도 KB 축자 *"Amounts exactly at a threshold are charged the **lower** tier's fee"* 와 `<=` 로 정합.
- `get_checking_atm_fee_totals`: 위 8 + `Green Fee-Free`(0/0) + `Gold Years`(0 / $3.50) = **40칸 전수 일치**.
- `check_card_application_fit` 요율: Platinum 10.0 · Gold 2.5 · Silver base1/cat4 · Diamond 5 · Bronze 1 — **정확**.
- `get_interest_correction` 의 `1/12`: KB 저축계좌 문서 다수 *"Interest is credited monthly."*

---

## 6. ⚠ 감사가 **닫지 못한 것**

- `check_card_application_fit` 의 비-요율 칸(`min_score` 640/660/680/700/720/735/750/765/780 · `limit_max` · `annual_fee` · `min_payment_pct`)은 **미대조**. 요율만 봤다.
- 이름 절단률은 `task_020` 한 건에서만 쟀다(679:571). 전 태스크 분포 미측정.
- 오탐이 **실제 reward 를 얼마나 깎았는지**는 미측정 — [[69]] 채점단위(변이 집합 MISSING/WRONGARG/EXTRA)로 재야 한다.

---

## 7. 수리 — **A~E 전부 적용 완료** (2026-09-06 20:40 · 사용자 승인 *"모두 한번에 가라"*)

적용 층 = **3개 동시**([[24]] 양방향): `a2/banking_knowledge.gate.json` · `…specific.json` · `a2/split/banking_knowledge.core.json`
(구판 op md5 3층 모두 `e62761e406`/`bda714d4c0` 로 동일 → 갈라진 층 없음). `a2/frozen/` 은 baseline 이라 미변경.

| # | 변경 | 실물 |
|---|---|---|
| **A** | Platinum `4 → 10` | `steps.rate` 교체에 포함 |
| **B** | Silver 평률 → 카테고리 조건부 | `{"op":"case","key":"r.category","cases":{"Travel":4,"Software":4},"default":1}` |
| **C** | CLI 이용률 `>` → `>=` | **2칸**(선행 CLI 없음 분기 + 쿨다운 통과 분기). 연령·쿨다운·납입의 `<` 는 **미변경**(KB *"on or after the day the minimum is reached"* 와 정합) |
| **D** | 미등재 카드 **4종** 등재 | Business Silver(T/S 10·기타 1) · Business Gold(Operations 2.5·기타 1) · Business Platinum(T/S/Media 4·기타 **1.5**) · Crypto-Cash Back(2) |
| **E** | 카드명 절단 금지 | `params.transactions` 에 축자 추가 — *"copy credit_card_type EXACTLY … INCLUDING any leading 'Business ' … a wrong card name makes this tool compute the wrong expected reward and flag correct transactions as disputes."* |

`steps.rate` 는 구판 `if_then`(EcoCard 특례) 를 **단일 `case`** 로 통일했다. 엔진 `case` 는 중첩 op 를 재귀 평가하고
키 비교가 `strip().lower()` 라 대소문자 차이에 안전하다(`t2_compute.py:793-800` 축자).
구판 DAG 는 `steps._prev_rate_x807_` 에 보존.

### 7-1. ⛔ D 에서 **의도적으로 뺀 3종** ([[22]] 열린 술어)
선언된 leaf `(credit_card_type, category)` 로 **닫히지 않는** 카드는 등재하지 않고 `default: null` 기권을 유지했다.
등재했다면 §2-2 와 **같은 종류의 오탐**을 새로 만들었을 것이다.

| 카드 | 왜 안 닫히나 (KB 축자) |
|---|---|
| `Business Bronze Rewards Card` | 머천트 기반 0% 제외 — *"The following specific merchants earn **0% cash back**"* (WeWork·Regus·Industrious·Gusto·ADP·Paychex·Rippling) + *"Slack·Zoom·HubSpot·Salesforce … After the initial 12-month period … earn 0% cash back"* |
| `Silver Zoom Card` | promo 배수 — *"Transportation and logistics category purchases earn **3 × 3.0%** during the promo period"* (기간 미선언) |
| `Green Rewards Card` | 머천트 기반 — *"Purchases at **qualifying sustainable merchants** earn 3.0%"* · db 거래 **0건** |

### 7-2. 검증 (전부 통과)

- **`x807` 요율 DAG 실행 검증 16/16** — db 실측 행 + 합성 경계. 핵심 3칸:
  `txn_a8f1c2d3e403`(Business Silver·Travel·$315·3,150) → **정상**(구판 오탐) ·
  `Platinum·$100·1,000` → **정상**(구판 오탐) · `Silver·Groceries·$200·200` → **정상**(구판 오탐).
  `Business Bronze`·`Silver Zoom` → **기권 2/2**.
- **CLI 경계 5/5** — 정확히 90%/80%/70% 전부 `NOT_ELIGIBLE_UTILIZATION`, 한 단위 아래는 `ELIGIBLE`.
- **`test_a2_three_layer.py` ALL PASS** — banking 실키 66 일치.
- **회귀 0** — `test_lever_reachable.py` · `test_c201_stage2.py` · `test_sg_record_order.py` 3건 실패는
  구판 파일로 되돌려도 **동일하게 실패**(기존 결함). `test_sg_row_count.py` PASS.

### 7-3. ⛔ 아직 안 한 것
- **리모트 미동기화**. base nt=4 가 3레인 도는 중이라 [[86]] 부칙(런 중 pull/동기화 금지)을 지켰다.
  런 종료 후 `ps -eo cmd | grep "[t]2_run_gated"` 가 빈 것을 확인하고 승인 받아 별도 단계로 한다.
- **커밋 안 함**(3파일 수정 상태). 209줄 추가 · 31줄 삭제.
- **[[70]] 파는 쪽 미측정**. A·C 는 오탐만 줄여 ⊖ 후보가 없지만, B·D 는 판정 커버리지를 늘려
  새 deny 턴·전달 바이트를 만든다. **런이 판정한다.**
- **[[81]]**: 이 수리는 A2 선언층이라 새 계기가 없다 — 첫 런에서 `_sg_stats.skipped` 가
  구판(사업자 카드 전건 기권) 대비 **줄었는지**로 발화를 확인할 것.
