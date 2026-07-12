# 출처 선언 4지선다(GET/FIND/INFER/ASK) — learn·loop·prompt 세 처방 실측 비교 (2026-07-11)

> 상위 = `RESEARCH_MASTER.md`. 선행 = `C42_SHORT_CONTEXT_SOLVES_FOURWAY`(clean이면 base도 완벽·무-gradient) ·
> `C43_ANCHORED_SUBSTITUTION_NOT_WM`(날조=정박치환) · `C44_SOURCE_DECLARATION_LEVER`(출처선언+검증기 67→0%).
> 재현 = `scripts/distill/tau2/c51_fourway_prescriptions.py` (엔진: c47_dprime·c48_dprime_full·e11a 재사용).
> 결과 JSONL = 리모트 `sim_results/c51_fourway_prescriptions.jsonl` (스크립트만 커밋 · [[30]]).
> 규율: [[08]] per-case 포렌식 · [[05]] A2만 변경·엔진 고정 · [[09]] 무료-우선(로컬 32B·user-sim 0).

---

## 0. 한 줄 (판정 자리 — 결과 후 확정)
**세 처방은 같은 4지선다에서 서로 다른 것을 산다.** prompt는 행동을 *움직이되* 옳은 쪽으로 못 고정([[42]] 천장) ·
loop은 producer-존재 지점의 ASK 누수(t17형)를 결정론으로 *닫는다*(getbfAsk→1.0·날조→0) · learn은 clean 합성서
gradient가 0이라 **오염(정박) 데이터를 먼저 만들어 학습필요성을 실증하기 전엔 착수 불가**.

## 1. 결정점과 4지선다

write 직전 인자 하나마다: **"이 인자값의 출처는? {GET·FIND·INFER·ASK}"** + 갈래별 실행.

| 갈래 | 의미 | 옳음 조건 (검증기·decidable·DB내용 주입 0) |
|---|---|---|
| **GET** | producer 도구 호출로 값 산출 | 지목 도구가 그 인자의 producer (A2 매핑) |
| **FIND** | 문맥(사용자 발화∪도구출력)서 실재 값 복사 | 값이 접두 문맥에 문자열로 실재 |
| **INFER** | 문맥 값들로 유도 (최저가·합계 등) | (retail write-arg서 gold=INFER 부재 — §2 주) |
| **ASK** | 사용자에게 물음 | 그 인자를 산출하는 producer가 하나도 없어야 함 |

gold 라벨 = gold 값의 *거처* 로 자동 도출 (결정 시점 접두 기준 · `c47.gold_label`):
FIND(접두에 실재) · GET(접두엔 없으나 DB producer로 획득 가능) · ASK(어디에도 없음) · NO-WRITE(그 write 자체가 gold에 없음=과행동).

## 2. 모집단 (둘 다 tau2 retail 실 궤적 = in-vivo·오염·정박 재료 있음)

대상 tag = `fl32b_floor_retail_t4` (user_stop만·infra 0).

| 모집단 | n | 정의 | 역할 |
|---|---|---|---|
| **FAB** | 30 | 원 궤적서 *날조가 난* write 인자 결정점 (`find_violations`) | 처방이 날조/ASK누수를 닫는지 |
| **CLEAN** | 30 | 원 궤적서 *grounded* 였던 결정점 (`clean_points`) | Δspurious(멀쩡한 결정 파손) 대조 |

- **clean 합성 대조(C42)**: 짧고 정박재료 없는 합성에선 base(7B·32B)가 4지선다를 이미 완벽히 푼다(fabricate 0.00·gather 0.87).
  ⇒ 처방 차이는 clean에서 나오지 않는다. **오염된 긴-문맥(FAB)이 차이를 드러내는 유일 무대.** (C42·C43 확정 — 재현 불요.)
- **INFER 주**: retail write-arg(item/payment/address) gold엔 INFER가 없다(계산형 기준은 F2b 예산 결정점 = 별 모집단).
  4지선다엔 선택지로 제시되나 이 모집단선 gold=INFER=0 → INFER 정확도는 본 실험 범위 밖(별도 F2b 프로브 필요).

## 3. 처방 arm — 엔진·A2 공통 ([[05]] 준수)

세 arm 모두 **같은 결정점·같은 A2(`PRODUCER`={인자→producer 도구} 매핑 하나·값 주입 0)·같은 4지선다 인터페이스**를 쓴다.
차이는 오직 *처방 레버* 다.

| arm | 레버 | 구현 | [[05]] |
|---|---|---|---|
| **base** | 없음 (4지선다만) | 단일 호출·검증기/재발화 없음 (`build_base`) | 엔진·A2 공통 |
| **prompt** | 강화 규칙문 + **도메인-일반** 예시 | 단일 호출·system에 규칙 주입 (`STRONG_RULE`) | 예시에 retail gold 값 0 (일반 서술만) |
| **loop** | 결정론 controller | 검증기(producer 있으면 ASK 금지·FIND 문맥실재·GET producer일치)+재발화(≤3)+**소진 시 GET 강제** (`c48.run`) | 검증기=decidable·DB내용 주입 0 |
| **learn** | DPO/SFT로 GET-first 성향 설치 | **설계만** (§6·유료/데이터 게이트) | 도메인-일반 스킬 목표 |

- **prompt 예상**([[42]]): prompt-only는 긴 문맥서 copy/anchor prior를 못 이긴다 → 행동은 움직이나 천장. 검정 대상.
- **loop = t17 처방의 일반화**: t17은 PERARG 발화 후 getter 대신 ASK로 샜다(NEWSTACK §G). loop은 producer 존재 시 ASK를
  결정론으로 막고 getter를 강제 → t17형 누수를 원리적으로 닫음.

## 4. 지표 (arm별 · 세 개)

| 지표 | 정의 | 무엇을 잡나 |
|---|---|---|
| **(a) 4지선다 정확도** | 선언 출처 == gold 라벨 (판정가능=gold∈{GET,FIND,ASK}) | 옳은 출처를 *선언* 하는가 |
| **(b) 최종 인자** | 날조율(FIND 값 문맥에 없음) · find_exact(값==gold) · **find_wrong**(문맥엔 있으나 ≠gold = ⋈ 경계) | 선언대로 실행 시 값이 gold인가 · ⋈ 잔여 |
| **(c) GET-before-ASK 준수** | producer 존재 지점서 ASK 를 *안* 고른 비율 | t17형 누수(producer 있는데 ASK) |

- **(b) 캐비어트**: GET 갈래는 격리 프로브서 실제 도구 실행이 없어 값이 *유예* 된다(vclass='get'=출처정답·값보류).
  ⋈(후보 2+개서 옳은 값 선택)은 GET/FIND 어느 갈래로도 이 레버가 못 닫는 경계(C46·C3b) → find_wrong 로 계측.

---

## 5. 결과 [M] (실측 후 채움)

> arm=base,prompt,loop · n=30/모집단 · 32B(localhost:8140) · 단일 clean run([M]).

### 5.1 집계표
_(실행 완료 후 c51 출력으로 채움)_

### 5.2 per-case 포렌식 (≥6 · [[08]])
_(loop이 t17형 GET-강제로 닫는지 · prompt가 [[42]]대로 천장인지 · base→loop/base→prompt 뒤집힌 결정)_

### 5.3 FABRICATION 배분표 — "출처선언" 행 (방법 × 케이스)
_(처방별 수치 반영)_

---

## 6. learn arm — 데이터 요건 + 게이트 (설계만·미착수)

**학습 표적** = "긴 오염 문맥에서도 GET-first(producer 있으면 GET·ASK 최후)를 유지" = 부하 내성(결정 자체 아님·C42).

**착수 게이트 (순서·먼저 걸리면 정지)**:
1. **타당성 게이트 (C42 무-gradient 차단)**: clean 짧은 합성선 base가 이미 완벽 → gradient 0 → 학습 불가.
   ⇒ **D7 정박 합성 필수**(C43 §4.1): 창에 *정답과 같은 형식의 근접-오답 id* 를 배치해 base가 정박치환을 저지르는
   합성을 만들고, **base가 tau2 수준(≥30%)으로 날조함을 먼저 실증**. 통과 못 하면 학습 착수 금지.
2. **데이터 구성 (C38·C39·선행 §5b.3 반영)**: 음성사례(정답 함수 제외·라벨=기권/GET) 포함 · **on-policy rejected**
   (off-policy DPO는 valid↔fabricated id가 edit-dist 1-2=CHES 극高 → likelihood displacement로 정답 밀어냄·C43 §6.4) ·
   발명형 rejected(C39).
3. **무효 조건**: loop이 이미 무료로 pass비용 0에 날조/ASK누수를 닫으면([[13]] scaffold<learn 우선) learn의 순증분이
   loop 대비 무엇인지 먼저 규정 — "부하 내성 흡수(scaffold 없이도 GET-first)" 외엔 중복.

**learn 착수 조건 (요약)**: (i) D7 정박합성서 base 날조 ≥30% 실증 ∧ (ii) loop 대비 순증분(scaffold 제거) 규정 ∧
(iii) 유료 승인([[09]]) ∧ (iv) on-policy 손실(RLVR/DPO-Positive).

## 7. 판정 (결과 후 확정)
- **prompt 천장**: _(base 대비 Δ · [[42]] 검정)_
- **loop 강제**: _(getbfAsk·날조·Δspurious)_
- **learn 필요성**: 1차 = §6 게이트 (clean 무-gradient → 오염데이터 선실증 필수). 미착수.
