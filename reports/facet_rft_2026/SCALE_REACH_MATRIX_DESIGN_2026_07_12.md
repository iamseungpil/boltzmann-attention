# scale-reach 행렬 설계서 — 기전 × 난이도 × 크기 × 도메인 (2026-07-12)

> 상위 = `EXPERIMENT_DESIGN_RIGOR_2026_07_12`(E1-E6 셀-상세·FE1/R1-R7) · `THINKING_HORIZON §8-§10` · 원장 C56/C66/C69/C71/C72.
> 규율: [[05]] A2만·엔진 도메인-일반 · [[08]] 기전·per-case · [[09]] 무료 우선 · [[40]] 등급.
> **상태**: 설계(리뷰 후 구현). 사용자 지적(2026-07-12): "synth만 아니라 synth/retail/banking을 난이도별 기전-분해해 *scale이 어느 문제를 어디까지 푸는지* 측정하라."

## 0. 목표 = scale의 도달 지도
**질문**: 모델 크기(0.5B→32B→frontier)가 **어느 기전을·어느 난이도까지 scale로 푸는가**, 어디서 포화(scale가 해결)하고 어디서 flat(잔여=레버 필요)인가.
**현 상태(정직)**: 지금까지 대부분 **synth(우리 사다리)** + retail/banking은 frontier 집계·proxy. **retail/banking을 *우리 사다리*로 기전×난이도 분해한 것 거의 없음.** 이 설계서가 그 공백을 격리-프로브(무료)로 채운다.

## 1. 행렬 (셀 = {우리 사다리 0.5-32B + frontier 앵커} 성능곡선 → 포화/flat + 닫는 레버)
| 기전(근본기능) | 격리 프로브 | 난이도 축 | 도메인 | 현황 | 예상 scale 거동 |
|---|---|---|---|---|---|
| **M1 상태추적/sd** | E-HORIZON(순수-산술·FE1-b) | 값범위·K·H | synth | ✅ 일부(C72·32B) | 쉬우면 32B 포화·어려우면 flat |
| **M2 참조/near-miss** | E-REF/E-THINK | decoy 밀도(L1-L4) | retail(상품) | ✅ C66/C69 + E-THINK2 진행 | thinking/prov 닫음(§ E-THINK2 ②) |
| **M3 ⋈ 의미모호** | ⋈-probe(NEW) | 유효후보 수 | retail(주문) | ❌ NEW(C56/C59 부분) | **전 scale flat**(C56) |
| **M4 reach/발견** | banking-reach probe(NEW) | 발견체인 깊이 | banking(도구) | ❌ NEW(frontier=§3.2f) | banking 지배 잔여 |
| **M5 coverage/완결** | coverage-probe(NEW) | 항목 수 | retail/telecom | ❌ NEW(C52 frontier) | 전-도메인 flat |
| **M6 compute** | calc-probe | 자릿수·연산 | synth/retail | ✅ C-B 부분 | 전 scale flat→offload |
| **M7 compliance** | 게이트 census | — | 전 도메인 | ✅ C1 [S] | scale-불변 |

**핵심 산출**: 각 (기전, 난이도)서 성능이 크기와 함께 **포화하는 임계 크기**(scale가 푸는 지점) vs **flat**(어느 크기도 못 품=레버 영토). = capability×scale×lever 지도(논문 코어).

## 2. 프로브 스펙 (격리·무료·결정론gold·[[05]] 클린)
각 프로브 = **한 기전만 격리**(다른 기전 gold-제공)·user-sim 0·우리 사다리 전수 + frontier 앵커.

### M1 상태추적/sd — E-HORIZON (기존·FE1-b 반영)
- 순수-산술 running-sum(lookup 금지)·난이도=값범위/K/H. arms{base,verify,detect,inject}. 기전=sd(inject 자기일관·F9). 난이도 보정=FE1-a 사전등록(32B base∈[0.75,0.88]).

### M2 참조/near-miss — E-REF/E-THINK (기존·확장)
- retail 상품 바인딩·decoy 밀도 L1-L4(C69). arms{base, +thinking, +prov-constraint(E2)}. **E-THINK2 발견**: thinking이 near-miss 닫음(4b L4 0.28→0.97·8b/14b 대기). ⇒ **M2 = decidable**(thinking OR prov 닫음)·⋈(M3)와 구분.

### M3 ⋈ 의미모호 — ⋈-probe (NEW)
- **설계**: retail 주문서 기준("my recent order")이 **복수 유효후보** 매칭(전부 owned·near-miss 아님)·의도로만 결정. 후보 수 난이도. arms{base, +thinking, +enumerate(C59)}.
- **판정(사전등록)**: 전 scale·thinking flat(못 닫음) → M3=진짜 잔여([[40]] "scale/learn/ASK 영토"). enumerate가 부분(C59 +31pp)이나 잔여~.34.
- **gold-independence**: 기준-매칭은 결정론(gold=유일 의도)이나 *의도 자체*가 모호=경계. [[05]]: A2 0(측정).

### M4 reach/발견 — banking-reach probe (NEW·최고가치)
- **동기**: banking 지배기전(reach·§3.2f 발견체인 universal-fail 28/28)을 **우리 사다리로 처음 분해**(현재 frontier만).
- **설계**: banking 도메인(도구+KB+정책)·목표 제시·**필요 도구가 unlock 선행조건 뒤**(발견체인). 지표=올바른 도구 발견+unlock+호출. 난이도=체인 깊이(1/3/6+). arms{base, +controller(gather/discover 강제·E-PLAN)}.
- **판정**: 체인깊이↑서 base 급락(banking 곡선 재현) ∧ controller 회복 → reach=controller 영토. 격리(user-sim 0·gold=§3.2f 발견체인).
- **[[05]]**: controller=도메인-일반 gather/discover 강제·banking 정책사실=A2.

### M5 coverage/완결 — coverage-probe (NEW)
- **설계**: "모든 Y에 X" task·항목 수 난이도. 지표=완결율. arms{base, +완결게이트(읽기강제·C52 G1)}.
- **판정**: 항목수↑서 base 하락(전-도메인 flat·C52) ∧ 게이트 완결 → coverage=게이트 영토.

### M6 compute — calc-probe (기존·C-B)
- 자릿수/연산 난이도. arms{base, +calc-offload}. 예상 전 scale flat→offload 1.00(C-B 대후보선택 0.02 vs 결정론 1.00).

### M7 compliance — 게이트 census (기존·C1)
- scale-불변 [S]·per-write rate flat. 난이도축 없음(위반=이분).

## 3. 측정 프로토콜 (포화 탐지 → 레버 귀속)
1. **각 셀**: 우리 사다리 0.5/1.5/3/7/14/32B(+frontier 앵커=submission.json 무료) × 난이도 × arms. runs≥30·**R1 CI**.
2. **포화 탐지(사전등록)**: base(scale) 곡선서 *포화 임계 크기*(Δ<ε 되는 지점) 또는 *flat*(전 사다리 저조). 난이도별로 임계 이동.
3. **레버 귀속(R4·per-case)**: 잔여(flat 부분)를 닫는 레버 측정(arm 대비)·**기전 확인**(F9식 self-consistency/per-case·집계 직행 금지).
4. **frontier 앵커**: 상단(0/18 banking·retail 챔피언)은 submission.json 역산(단 M4=§3.2f 직접·M1 synth엔 frontier 없음·주의).

## 4. 엄격성 (F1-F9 + FE1 코드화)
- **R1 CI**·**R2 대조군(detect/arm)**·**R3 gold-independence**(verify=재계산·prov=출처제약·controller=recency·전부 gold 아님)·**R4 기전 not 집계**·**R5 사전등록 양분기**·**R6 [[05]] 감사(셀별)**·**R7 confound 배제(parse율·termination)**.
- **★도메인-교차 축 금지(리뷰 F-banking)**: banking p_step=pass1 역산(H가정)·E-HORIZON=직접측정 → **같은 곡선/축 금지**. 각 도메인 자기-축·병렬 제시(외적타당도 vs 내적타당도).
- **★기전 단일성(FE1-b)**: 각 프로브는 한 기전만·난이도 상향이 다른 기전 유입 안 함(synth=산술만·lookup 금지).

## 5. 무료 / 기보유 / 유료
- **무료(격리 프로브·우리 사다리)**: M1-M6 전부. user-sim 0·결정론gold.
- **기보유**: M2(C66/C69)·M6(C-B)·M7(C1)·M4-frontier(§3.2f)·M3-부분(C56/C59).
- **NEW 구현**: M4 banking-reach(최우선)·M3 ⋈-probe·M5 coverage-probe.
- **유료 없음**: e2e(E-XFER-bank류)는 결론-후-확인 대기열([[09]]).

## 6. 산출 = 도달 지도 → 논문 주장
| 주장 | 셀 근거 | 등급목표 |
|---|---|---|
| scale는 상태추적을 난이도-임계까지 산다 | M1(E-HORIZON) | [M] |
| near-miss=decidable(thinking/prov) | M2(E-THINK2·E2) | [M] |
| ⋈=전 scale flat=잔여 | M3 | [M] |
| reach=controller 영토(banking) | M4 | [미검]→[M] |
| coverage=게이트 영토 | M5 | [미검]→[M] |
| compute=offload | M6 | [M] |
| compliance=scale-불변 | M7 | [S] |
⇒ **"scale는 M1(임계까지)·M6(못)·M7(못)... 각 기전마다 도달이 다르다"** = 논문 capability×scale×lever 지도 완성.

## 7. 실행 순서 (전 무료·리뷰 후)
1. **진행 중**: E-THINK2 8b/14b(M2 near-miss·F3 판정) · E3 inject 7B/14B(M1 sd).
2. **최우선 NEW**: **M4 banking-reach probe**(우리 사다리로 banking 지배기전 첫 분해) · M1 E-HORIZON FE1 파일럿(난이도 보정) · E2 prov-constraint(M2 레버).
3. **다음**: M3 ⋈-probe · M5 coverage-probe.
4. 완료마다: R1-R7·per-case·원장 C-신규·[[40]] 등급·행렬 셀 채움.

## 8. 리뷰 포인트
1. M4 banking-reach probe = banking 액션모델(discoverable/unlock/shell) 격리가 충실한가·gold=§3.2f 발견체인 사용 타당?
2. 도메인-교차 "같은 축 금지"(R7·§4)로 3-도메인을 *병렬*(외적 vs 내적 타당도)로만 — 통합 곡선 유혹 차단 충분?
3. NEW 프로브 3종(M4/M3/M5)의 [[05]]·gold-independence 사전 검증 항목 충분?
4. frontier 앵커: M1(synth)엔 frontier 없음·M4(banking)만 §3.2f 직접 — 앵커 비대칭 어떻게 정직 표기?
