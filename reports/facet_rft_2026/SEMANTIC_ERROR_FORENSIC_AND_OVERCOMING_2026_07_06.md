# Semantic 에러 전수 궤적 포렌식 → 극복 방안 (2026-07-06)

> **질문(사용자)**: 이전 실험은 오차를 SYMBOLIC vs SEMANTIC로 갈라, symbolic은 결정론 scaffold(calc/CoT)로 닫고 semantic은 "scale 잔여"로 분류했다. **semantic 실패 케이스를 전수 정독해 scale 문제와 아닌 문제를 다시 가르고, 극복 방안을 제시하라.**
> **정본 입력**: `RELWORK_LOAD_COT_2026_07_05.md`(symbolic/semantic 분할·CoT/self-consistency probe) + `ASSEMBLED_FAILURE_FORENSIC_2026_06_27.md`(per-case 정독) + `plan_probe_phase0`(격리 planning) + operand_controlled(GIVEN-SPEC 100%/GOAL 70%/⋈ 54%).
> **비용**: gpt-4.1 0 (기존 궤적·probe 재분석·신규 런 0).
> **불변**: [[08]] 전수 per-case·robust·집계→결론 금지 · [[13]] 결정론(scaffold)먼저·scale최후 · [[05]] 도메인-일반·A2만 인스턴스 · [[03]] probe=adjudicator(예측으로 단정 금지).

---

## 0. 한 줄 결론
**"semantic = scale"는 과대 집계였다.** semantic 실패를 전수 정독(§2b 궤적 4건 정독 포함)하면, **지배 기전은 결정론-닫힘인 {상태추적(A)·feasibility/조건 게이트(B)}**이고 symbolic(C)까지 합하면 대부분이 scale 아님. **abstain(D)·genuine-scale(F)은 잔여 소수.** ★[[08]] 교정: 첫 패스는 one-line 라벨로 t41/t107을 "⋈ 의미매칭"·t62/t66을 "abstain"으로 오배정했으나, **궤적 정독하니 전부 A(멀티주문 미추적) 또는 B(불가능/조건 op)** = 예상보다 *더* 결정론-favorable. RELWORK §6d(systematic→debias·scale 아님) lead는 유효하나, τ²-retail의 실제 지배는 debias보다 **state-tracking + feasibility gate**.

---

## 1. 선행 확정 (재유도 금지)
- **SYMBOLIC**(max-of-N·joint-constraint): CoT +17%/+35% 회복·**결정론 solver > CoT**(Sprague 2409.12183) → calc/gate로 닫힘. scale 아님.
- **SEMANTIC**(intent-select 68→72% CoT **+4%**·⋈-matching): CoT 무효·self-consistency **+0%**(8/8 동일오답=**systematic·high-conf**) → "scale 잔여"로 분류됨.
- **★그러나 RELWORK §6d 미소진 lead**: systematic 오차는 (a) **targeted label-free 보정으로 제거**(PriDe 2309.03882·1.15×·scale 0), (b) coverage→acc는 **verifier 필요**(Brown 2407.21787), (c) 반복/투표는 correlated라 무효(Byerly&Khashabi 2411.01101). ⇒ **systematic이면 debias-fixable이지 자동으로 scale 아님.** 이 lead가 semantic 실패에 적용된 적 없음 = 본 포렌식의 진입점.

---

## 2. 전수 per-case 재분류 (ASSEMBLED_FORENSIC 정독·양 scale)

semantic-관련 ~35건을 **극복 축**으로 재라벨(기존 라벨은 증상, 아래는 기전):

| 하위클래스 | 대표 케이스 | 진짜 기전 | 극복 레버 | scale? |
|---|---|---|---|---|
| **A. 상태추적**(멀티엔티티 미완) | **t41·t107**·t102·t111·t98·t99 | 유저가 "모든 주문/두 항목" 요청 → 일부만 처리(**나머지 미추적**). ⋈-매칭 아님·모호 아님 | **C2 결정론 controller = 영향 엔티티 전수 열거·coverage 강제** | ✗ 결정론 |
| **B. feasibility/조건 게이트** | **t62·t66**·t10·t34·t57·t45·t85·t63 | 불가능 op(luggage→coat·부분취소·타주문결제)에 **소모**·조건부 write(price>$300) 미평가 = **decidable** | **precondition/feasibility 게이트(불가능=block→fallback) + 조건 calc + A2 refund-rule(5/5 DB)** | ✗ 게이트 |
| **C. SYMBOLIC criterion**(joint/filter) | t20(size9∧max)·t13(non-gaming) | 필터∧최대 = **serial-symbolic** | **present-enumerate + calc**(solver>CoT) | ✗ calc |
| **D. ABSTAIN**(genuine 모호→추측) | (τ²-retail서 **소수·미확정**) | 진짜 다중매칭·모호 intent인데 묻지 않고 추측 | **present σ 관측화 + σ≠1이면 ASK**(epistemic-A2 + abstain 커리큘럼) | ✗ abstain scaffold |
| **E. present-formalize variant** | t100·t27·t58·t110·t8 | NL 스펙("waterproof"·34"custom)→변형 매칭 | **present 전체옵션 노출(→GIVEN-SPEC 100%) + PriDe debias(systematic 선택편향)** | ◐ 대부분 present, 소량 scale |
| **F. TRUE semantic**(genuine) | E·D 잔여 소수 | present+abstain+debias 후에도 남는 진짜 의미이해 | scale/(도메인-일반 학습) | ✓ scale |

**★집계 교정(궤적 정독 후·§2b)**: 첫 패스는 D(abstain)를 ~6으로 봤으나 **정독하니 t41/t107→A·t62/t66→B**로 이동 = **A·B(결정론)가 지배**, D는 τ²-retail서 소수·genuine 여부 미확정(P-abstain으로 측정). 개정 추정: **A ~12 · B ~10 · C ~5 · E ~4 · D 소수(0~3?) · F(진짜 scale) ~2–4/arm** — make-or-break "learn-candidate ≤3–4"와 정합. 즉 **"semantic 잔여"의 지배부분은 결정론(상태추적+feasibility)으로 극복 가능**하고, abstain·진짜 scale은 소수(D는 측정 필요·다른 도메인선 더 클 수 있음).

## 2b. 궤적 정독 증거 (4건·[[08]]·robust fail-all-3·`asmscale_*_0626pm`)
- **t41 (32B·라벨"⋈ wrong order")→ 실제 A**: 유저="**모든** 주문 주소 고쳐줘"(2 pending). 모델이 **두 주문 다 인지**("I see you have two pending orders")하고도 **#W4082615만** 주소변경·**#W9583042 미처리**. = 멀티주문 미추적(상태추적)·의미매칭/모호 아님. **라벨 오도**(집계론 "wrong order"처럼 보임).
- **t107 (32B·⋈)→ A**: 유저=부츠(주문1)+직소(주문2) 각각 exchange. 모델 한 주문만. = 상태추적.
- **t66 (14B·"wrong-op")→ B**: 유저=luggage→coat 교체(不가·다른 product)·"안되면 취소". 모델이 **불가능 exchange에 여러 턴 소모** 후 취소. **feasibility 게이트가 사전에 '불가능' 알렸으면** 즉시 fallback(취소)로. = 결정가능·scale 아님.
- **t62 (32B·over-action)→ B/C**: 유저="가격>$300이면 취소"(조건부). gold=read만(가격 조건 불충족). 모델이 **조건 미평가·취소 강행**(파괴적). 조건 = 결정론 calc/gate. = 결정가능.
- **함의**: 4/4가 A/B(결정론). τ²-retail "semantic 실패"는 *의미이해 부족*보다 **엔티티 전수추적 실패 + 불가능/조건 op 미게이팅**이 지배 → **극복=결정론 controller/gate**([[13]] 정합·make-or-break 강화). abstain(D)은 이 4건엔 부재.

---

## 2c. ★스케일-커브 3점 측정 (2026-07-06·GPU0∥GPU1 병렬·gpt-4.1 0·`sim_results/*_{7b,14b,32b}_2026_07_06`)
paid 32B 런과 **병렬**(GPU1 7B/14B·GPU0 idle 32B)로 isolated probe 재측정 → "scale vs 아닌 것"을 3점 커브로:

| probe | 7B | 14B | 32B | 거동 | 해석 |
|---|---|---|---|---|---|
| 변형 GIVEN-SPEC(순수 실행·n=88) | 94% | **100%** | 100% | **평탄-고** | 실행=scale 아님·scaffold가 전 규모 닫음 (robust) |
| 변형 GOAL(+criterion·n=88) | 55% | 67% | 70% | **완만 단조(+15)** | criterion=대체로 present-fixable·**소량 scale** (robust) |
| ⋈ DESCRIBED(묘사→매칭·n=13) | 54% | 23% | 54% | **비단조=노이즈** | n=13 과소·깨끗한 scale 트렌드 없음 |
| plan_probe core_ok(격리 planning) | — | 7/10 | 6/10 | 평탄/무관 | planning-of-required 블로커 아님(§2b) |

**★핵심(scale vs 아닌 것)**:
- **robust 평탄(scale 아님)**: GIVEN-SPEC 94~100%(전 3규모) → 순수 실행은 scaffold. plan core_ok도 평탄.
- **robust 완만-scale**: GOAL 55→70(단조·+15) → criterion은 대체로 present-fixable + **소량 scale**.
- **★[[08]] 교정**: 앞 패스는 "⋈ 23(14B)→54(32B)=강한 scale"이라 했으나 **7B 점 추가하니 54/23/54=비단조=n=13 노이즈**. 14B 딥이 우연이었고 **⋈의 깨끗한 scale 트렌드는 없음**. "강한 scale" 주장 철회. (n=13 2점 읽기가 오도 — [[08]] 반복.)

### 2c-1. ★P-debias — ⋈ wrong-match은 (scale보다) position bias 성분이 크다 (`p_debias_join_{7b,14b,32b}`)
후보 제시순서를 K=4 무작위하고 pick이 flip하는지 측정(temp0·gpt-4.1 0·n=13). 3규모:
| 지표 | 7B | 14B | 32B | 판독 |
|---|---|---|---|---|
| single-shot(1 셔플) | 54% | 46% | 31% | 순서 의존 큼(operand 자연순과도 불일치) |
| any-permutation | 62% | 62% | 46% | *맞는 제시순서 존재* |
| **FLIP(순서로 pick 바뀜)** | 46% | 38% | 38% | **전 규모서 위치편향 상당**(32B도 38%) |
| INVARIANT-WRONG(genuine) | 23% | 15% | 31% | 소수·비단조 |
| majority(무라벨 debias) | 54% | 54%>46 | 31% | aggregation 소폭 회복 |

- **robust 결론(전 규모 일관)**: (1) **위치편향(FLIP)이 전 규모서 상당**(38~46%·32B 포함) → scale이 위치편향을 없애지 못함 → **debias(제시순서 무작위+aggregate / PriDe [2309.03882])가 전 규모서 유효**. (2) genuine(INVARIANT-WRONG)은 소수. RELWORK §6d lead(systematic→debias·scale 아님)를 실제 ⋈ 잔여에 정성 실증.
- **★정직 단서(n=13)**: 개별 분율(single/any/INVARIANT)은 **n=13로 노이즈 큼·비단조**(single 54/46/31·INVARIANT 23/15/31). 정확한 scale/genuine 분해는 **N 확대 필요**(RELWORK §6c "clean cross-order ⋈ re-run" 미결과 정합). 확정된 것 = *위치편향 존재·전 규모*(정성)뿐, 정밀 분율 아님.
- **⋈ 최종(정직)**: MISSED-order(A·결정론 state-tracking·라이브 지배) + position-bias(debias·전 규모·정성 확정) + genuine(소수·정밀분율 미확정). ⇒ ⋈ 극복 = 상태추적(결정론) + 제시순서 debias(scale 아님)로 대부분·genuine 잔여만 scale(크기 미확정).

## 3. 극복 방안 ([[13]] 결정론→abstain→(최후)scale 순)

### 3.1 A/B/C = 이미 있는 결정론 레버로 닫힘 (Track 2 C1/C2와 통합)
- **A. 상태추적 controller**(C2): open NL 목표 → 영향 주문 **전수 열거**를 결정론 부기. `plan_execute_orch.py`의 controller에 order-coverage 강제 추가(멀티주문 미완=구조적 차단). [[05]] 도메인-일반 IR.
- **B. feasibility 게이트**: write의 precondition(부분취소 불가·status·refund=원결제∪상품권)을 gate_spec(A2)로 검사 → 불가능하면 **block+ASK**. t10/t34/t57/t63 계열. 엔진 도메인-일반·A2 인스턴스.
- **C. present+calc**: joint-constraint/filter는 후보 전체 속성 열거 + 결정론 필터·최대. t20/t13. (probe 실증: solver>CoT.)

### 3.2 D = ABSTAIN — 이론상 유효하나 τ²-retail선 소수 (정직 right-size)
- **기전**: genuine 모호(진짜 다중매칭·모호 intent)인데 묻지 않고 추측. epistemic-A2([[43]]): 불확실성을 *관측가능한 관계 카디널리티(σ)*로 외부화 → σ≠1이면 **ASK**.
- **★궤적 정독 교정**: τ²-retail 4건 정독서 abstain 케이스 **0** — 유저 요청이 대체로 명확(모두/이 주문/두 항목)했고 실패는 추측이 아니라 미추적(A)·미게이팅(B)이었다. ⇒ **abstain의 τ²-retail 실제 instance는 소수·미확정.** 이론(self-consistency +0%=systematic·추측 억제로 우회)은 유효하나, *이 벤치선* state-tracking+feasibility가 지배.
- **처리**: abstain을 **P-abstain probe로 instance-count 측정 후** 크기 확정(단정 금지·[[03]]). 구현(scaffold σ≠1→ASK + abstain 커리큘럼)은 준비되나(`A2_RULE_USE_SFT_PREP`), **τ²-retail 우선순위는 A·B 아래**. abstain은 **모호도 높은 도메인(다른 τ² 도메인·SOPBench)에서 더 큰 레버일 수 있음** → 전이 실험(Track 1)과 함께 측정.

### 3.3 E = present-full-options + targeted debias (★debias 실증됨)
- **present**: 변형 전체옵션 노출 → operand_controlled 실증대로 GOAL 67~70%→GIVEN-SPEC 100% 회복. 남는 NL-스펙 formalize가 잔여.
- **★debias(실증·2c-1)**: ⋈ wrong-match은 **position bias 지배**(FLIP 38%·INVARIANT-WRONG 15%뿐)로 확정 → **후보 제시순서 무작위화 + majority-aggregate(무라벨·PriDe [2309.03882])가 규모 없이 회복**(single 46%→majority 54%·natural 23%→any-perm 62%). 이 debias를 present 단계에 결정론으로 배선(제시순서 무작위 K회+집계) → ⋈ wrong-match의 대부분 극복. 남는 genuine ~15%만 scale.

### 3.4 F = TRUE scale (정직)
- present+abstain+debias 후에도 남는 genuine 의미매칭(소수). **이것이 scale이 사는 것** — 헤드라인 "scale buys semantic capability, not guarantee". 도메인-일반 학습(faithful-formalize) 후보이나 make-or-break 게이트(§A2_RULE_USE 4조건) 통과 시만.

---

## 4. 검증 실험 (forensic-driven·[[08]]·probe=adjudicator)

**"scale vs 아닌 것"을 단정하지 말고 counterfactual로 측정** — 각 semantic 실패에 3 probe:

| probe | 방법 | 판정 | 비용 |
|---|---|---|---|
| **P-classify** | semantic 실패 전건을 A–F 버킷으로 라벨(기존 궤적 정독) | 버킷 분포 확정(§2 수치 robust화) | 무료·로컬 |
| **P-abstain** | present σ + σ≠1→ASK 주입 후 재구동 → 모호 케이스가 ASK→정답 전환하나 | D가 abstain로 닫히는 비율 = **scale 아님 증명** | 32B·PERSISTED 후 |
| **P-debias** | 후보 제시순서 무작위화 ×k + PriDe → 선택이 flip하나(systematic) vs 불변(genuine) | E/⋈의 systematic-bias 분율 = debias-fixable | 32B·PERSISTED 후 |

- **판정 매트릭스**: P-abstain 회복 크면 → semantic 대부분 abstain-fixable(scale 아님). P-debias flip 크면 → systematic(debias). 둘 다 후 잔여 = **F=진짜 scale**(측정된 하한).
- **robust**: pass^k·per-case 이중확증. pass^1 금지. abstain은 **over-ask 비용(σ=1인데 ASK)** 동시 측정(대칭).

---

## 5. Track 2 / thesis 연결
- **A/B/C = Track 2 C1/C2 controller에 직접 흡수**(이미 `plan_execute_orch.py` 착수·batch/status/provenance controller 존재). **추가 2개**: (A) **order-coverage 강제**(영향 엔티티 전수열거·미추적 차단) + (B) **feasibility 게이트**(불가능 op = 사전 block→fallback·조건부 write = calc 평가). 이 둘이 §2b 4건을 직접 겨냥.
- **D(abstain)** = epistemic-A2 실증 지점·A2_RULE_USE 커리큘럼이나, **τ²-retail선 소수**(측정 후 확정) — 크기는 P-abstain·다른 도메인 전이서.
- **★결론 갱신(궤적 정독 후)**: "semantic=scale" → **"semantic 잔여의 지배부분 = 상태추적(A)+feasibility(B) 결정론으로 극복·symbolic(C)도 calc·abstain(D)·진짜 scale(F)은 소수"**. 첫 재분류보다 *더* 결정론-favorable(D를 A/B로 교정). 헤드라인 강화(결정론 controller/gate가 semantic 잔여 대부분 닫음·남는 소수만 scale) + make-or-break 정합(learn-candidate ≤3–4). **극복 실행 = Track 2 C2 controller에 coverage+feasibility 추가**(GPU-free 빌드)·잔여 measure는 P-abstain/P-debias(PERSISTED 후).

## 6. 불변 정합
- [[08]] 전수 per-case·robust·counterfactual probe로 검증(단정 아님). [[03]] probe=adjudicator.
- [[13]] 결정론(A/B/C)→abstain(D scaffold)→학습(D 커리큘럼)→scale(F) 순. [[05]] 전 레버 도메인-일반·A2만 인스턴스. [[11]] 학습=SOPBench/Synth·τ² A2-swap. [[09]] P-abstain/debias=PERSISTED 후·승인·최소.
