# Escape-Scope Diagnostic — make-or-break 첫 측정 설계 (2026-06-24 · rev2 리뷰반영)

> **위치**: `EPISTEMIC_A2_THESIS_2026_06_23.md` §6 "★선행 진단(escape 범위 측정)"의 구현 설계. SOAR impasse(§3 블록)가 분류축. **이 측정이 abstain-커리큘럼 SFT(풀 make-or-break) 전체를 GO/NO-GO**. 학습 *전* 단계 = 지금 진행 가능. [[05]]·[[10]]·[[11]] 준수.
> **rev2(2026-06-24 리뷰반영)**: #1 base arm 순환 제거(predicate 재추출 폐기→궤적-선택으로 분류) · #2 faithful 조작정의 못박음 · #3 select-probe 추가(capability-bound 측정) · #4 Stage-1=정성카탈로그/Stage-2=비율(필수) · #5 결정점 DB state.

## 0. 한 줄
**우리 escape("빈/모호한 관계→ASK")가 32B의 실패를 *얼마나* 덮는가**를 측정. 두 **독립 사실**을 깨끗이 분리:
- **(I) 아키텍처 사실**: 실패가 *DB 카디널리티*로 표면화되나(|σ|≠1)? = escape 기회 너비.
- **(II) capability 사실**: 후보집합을 주면 32B가 정답을 고르나? = grounding은 되는데 self-formalize만 실패(학습여지) vs 후보 줘도 못 함(capability-bound).
- ⇒ "Opus가 술어 잘 썼나"가 아니라 위 두 사실을 잰다. abstain-SFT 방향의 정직한 게이트.

## 1. 핵심 질문 + 단일 베팅
- **escape ceiling(아키텍처)**: *명세-충실* σ를 참 DB에 돌렸을 때, 32B 실패 중 **|σ|≠1**로 표면화되는 비율 = escape가 *원리상* 잡을 수 있는 상한.
- **★단일 베팅**: 최대 gap 클래스 **wrong-ORDER 47%(7 task)**가 **tie impasse(ⓐ·참DB |σ|>1) 냐**, **확신적 mis-ground(ⓑ·|σ|=1·모델이 딴 걸) 냐**.
  - **★사전 경고(정직·load-bearing)**: tau2 다수 태스크는 유저 요청이 주문을 *유일하게* 결정(정보 있음·모델이 매핑 실패) → wrong-order가 **ⓑ일 공산이 실제로 높음** → 그러면 escape는 *좁고* faithful-formalize(§3)가 본체. **이 진단이 그 추측을 DB 사실로 결판.**

## 2. 분류 스키마 (SOAR impasse → ⓐ/ⓑ) — ★궤적-선택 기반(rev2 #1)
**모델 predicate를 추출/재질의하지 않는다**(순환·노이즈). 분류 = **(i) faithful-σ의 참 DB 카디널리티** + **(ii) 궤적에서 모델이 *실제로* 고른 entity/operator/arg** 대조:

| 단계 | 판정 | SOAR impasse | 분류 |
|---|---|---|---|
| (i) 참 DB \|σ\| = **0** | (해당 tuple 없음) | no-change | **ⓐ** (모델이 묻지 않고 행동) |
| (i) 참 DB \|σ\| = **>1** | (후보 여럿) | tie | **ⓐ** (모델이 묻지 않고 궤적서 한 개 찍음) |
| (i) \|σ\|=1, but precond 위반 | constraint | **ⓐ′** (G-gate 영역·기존 레버) |
| (ii) \|σ\|=1 & 모델이 그 유일정답 entity 고름·**operator 틀림** | non-impasse | **ⓑ-act** |
| (ii) \|σ\|=1 & 정답 entity+op·**arg 값 틀림** | non-impasse | **ⓑ-op** (B2-resolve) |
| (ii) \|σ\|=1 & 모델이 **딴 entity 고름** | non-impasse | **ⓑ** (mis-ground·침묵 잔여) |

- **ⓐ = escape가 원리상 잡음**(|σ|≠1이 토큰으로 표면화). **ⓑ = 침묵 잔여**(참 DB는 단일·모델이 틀림 → 카디널리티로 안 드러남).
- predicate 재추출 단계 *제거*: \|σ\|=1에서 "모델이 자기 술어로 σ=1" 같은 건 자기 커밋이라 무조건 ⓑ로 보임(정보0). 실제 정보는 **궤적의 행동 vs 참 정답** 대조에만 있음.
- **ⓐ 안 2차 태그([[10]] A/B·커리큘럼용)**: ask-needed(정보부재→유저 질문·SOAR 외부) vs compute-resolvable(argmax/rank·B2·SOAR 내부 subgoal).

## 3. 입력 + Stage 구조 (rev2 #4 — n=15는 비율에 무력)
- **Stage-1 = 정성 카탈로그(헤드라인 아님)**: gap-task 15개(gpt-4.1 pass ∧ 32B fail-all-3). **케이스별** {faithful 술어·참 DB |σ|·궤적 선택·분류·근거·select-probe 결과}. **비율 GO/NO-GO를 여기 걸지 않는다**(wrong-order 7개서 1개 재분류=14% 흔들림=딥리서치 n≈25 비판과 동형).
- **Stage-2 = 비율(필수·선택 아님)**: 32B retail floor **전체 실패**(robust=fail-all-3 우선) → ⓐ/ⓑ 비율 + CI. + airline/banking(cross-domain escape 너비). **§6 임계는 전부 여기에 건다.**
- 신호: fail-all-3(robust) 우선·pass^1 점추정 단독 무효([[06-NOW]]).

## 4. 방법 (rev2 #1·#2·#3·#5 반영)
### 4.1 Arm-I = 아키텍처(escape 너비) — faithful-σ + 궤적
- **faithful 술어 조작정의(★rev2 #2·정의적 함정 차단)**: = **유저 발화의 리터럴 제약을 정확히 인코딩** — *제약 누락=가짜 tie / 제약 발명=가짜 ⓑ* **둘 다 금지**. 경험적 내용은 "완전-명세 술어 하에 *참 DB가 실제로* >1을 담는가"라는 **DB 사실**에 있음(Opus가 명세를 다 넣었나를 재는 게 *아님*). 출처 = gold action seq + user-scenario.
- **σ 실행 = 결정점 직전 DB state(★rev2 #5)**: 멀티스텝 gap은 앞선 tool이 state를 바꾼 뒤가 관련 카디널리티 → 초기상태 아니라 *그 실패 결정 직전* 상태에 σ.
- 분류 = §2 표(궤적 선택 대조). 도메인분기 0·A2 σ/⋈/π만([[05]]). gpt-4.1 불요.

### 4.2 Arm-II = capability(select-probe) — ★rev2 #3
- **tie(|σ|>1) & ⓑ(|σ|=1) 케이스에서**: 32B에게 **faithful 후보집합을 주고 정답 선택**을 시킴(grounding을 떠먹임).
  - **정답 → "grounding은 됨·self-formalize만 실패"** = 학습여지(GO 기회).
  - **후보 줘도 틀림 → capability-bound** = §6 NO-GO(escalate/scale)의 결정적 신호.
- 이 probe가 #1의 "재질의"를 *올바른 자리*(분류 아니라 capability 판정)에 둠. select-probe만 32B 호출(소량·로컬·throttle 없음).

## 5. 출력
1. **Arm-I: ⓐ/ⓑ split**(impasse 타입별·**참 DB 카디널리티 기반**): escape 기회 너비. **wrong-order가 어디 떨어지나**(단일 베팅 결판).
2. **Arm-II: select-probe pass율**(tie·ⓑ 케이스): grounding-됨 vs capability-bound 분해.
3. **ⓐ 내 ask-needed vs compute-resolvable**: 커리큘럼 타깃 비율.
4. impasse 타입 × gap-클래스 교차표(wrong-order/action/operand/budget/over-action → tie/no-change/constraint/non-impasse).

## 6. GO / NO-GO (정직·★기회≠학습성공·rev2 #3·#4)
**비율 임계는 Stage-2에만**(Stage-1 과반에 걸지 말 것). **GO는 두 사실 *모두* 필요**:
- **GO(커리큘럼 지을 가치 있음·"강함/될 것" 아님)**: Arm-I ⓐ가 과반(특히 wrong-order=tie) **AND** Arm-II select-probe 통과(후보 주면 32B가 고름=학습여지). → ⓐ 넓음 = *escape 기회* 넓음이지 학습성공 보장 아님.
- **부분 GO(명세-충실 우선)**: Arm-I ⓐ 좁음(wrong-order=ⓑ) → escape보다 **faithful-formalize 학습**이 본체. 여전히 학습대상이나 ASK 아님(커리큘럼 무게중심 이동).
- **NO-GO**: (a) 5개 실패가 유한관계로 깔끔 표현 안 됨, or (b) **Arm-II에서 후보 줘도 32B 틀림(capability-bound)** 이 압도 → 진짜 경계 = escalate/scale(thesis §6 NO-GO).

## 7. 측정 주의 / 함정
- **★faithful 판정이 절반은 정의적(rev2 #2·과소평가 금지)**: 같은 wrong-order가 술어를 넓게(제약 누락)면 tie(ⓐ)·완전 명세면 σ=1(ⓑ)로 갈림. §1 경고가 맞다면 완전-명세는 *거의 정의상* σ=1=ⓑ. → **S2 소검증 핵심 점검항목 = "술어가 유저 제약을 빠짐없이·없는 것 없이 담았나"**(누락/발명 둘 다 spot-check).
- **conflate 금지(이미 분리)**: Arm-I(아키텍처)와 Arm-II(capability)를 섞지 말 것 — "escape 좁다"가 카디널리티 사실인지 모델 capability인지 분리(G5=0가 binding 아닌 capability였던 함정과 동형).
- **base 순환 제거됨(rev2 #1)**: 32B predicate 추출/재질의로 분류하지 않음(자기 커밋=무조건 ⓑ=정보0). 분류는 궤적 선택 vs 참 정답.
- σ 엔진 미비분(`grounding.json` 부분 실물)은 *측정 전* 보강·gap 5개 소검증서 σ 출력 sane 확인 후 전수.

## 8. 구현 단계
1. ✅ **S0 설계**(이 문서·rev2).
2. **S1 harness 빌드** `escape_scope_diag.py`: gap sim 로드 → **결정점 직전 state** 추출 → **Arm-I**(faithful 술어[큐레이션]→σ→궤적선택 대조→§2 분류) + **Arm-II**(tie·ⓑ 케이스 32B select-probe). `grounding.json` σ 보강. repo 커밋([[30]]).
3. **S2 소검증(대면·무인 금지)**: gap 5개로 (a)σ 출력 sane (b)**faithful 술어가 유저 제약 빠짐없이·없는것없이** (c)궤적 선택 추출 정확 (d)select-probe 동작 — 4항 확인.
4. **S3 Stage-1 정성 카탈로그**: 15 케이스 분류+근거+probe(비율 결론 금지).
5. **S4 Stage-2 비율(필수)**: 32B retail 전체 실패 → ⓐ/ⓑ + CI + select-probe율 → §6 GO/NO-GO. + cross-domain.
6. 산출 = `ESCAPE_SCOPE_RESULTS_2026_06_2x.md` + 판정 → 풀 make-or-break(SFT) 설계 or escalate.

## 9. 불변 (discipline)
- 정적 진단·**tau2 학습 0**([[11]])·A2 관계(σ/⋈/π)만·도메인분기 0([[05]])·gpt-4.1 불요(select-probe=로컬 32B만).
- harness repo 커밋·**S2 4항 소검증 전 무인 전수 launch 금지**(틀린 split·가짜 tie/ⓑ=방향 오도).
- 풀 SFT(make-or-break 본체)는 별개·딥리서치 방법(IDK targets·learning-to-defer `w0r8slp20`) 수령 후. 이 진단 = 그 *선결 게이트*(아키텍처 기회 + capability 두 사실).
