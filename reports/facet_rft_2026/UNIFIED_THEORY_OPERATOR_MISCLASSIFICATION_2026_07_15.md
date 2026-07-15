# 통합 이론 — 모든 실패 = open/closed 경계에서 해소연산 오분류 (2026-07-15)

> 정본. 사용자 통합 통찰(경계에서 과확신/under-action·formalize=Find/Get/Ask) + [[16]] 재유도 + H_min.
> 입력: C88(voting-vacuity)·C89(4하위유형)·C90(2층)·C91(closure)·[[16]] GENERALIZED_SCAFFOLD·[[44]] SOAR·C81(compute)·reach forensic.
> 등급: 이론 [D]·구성요소 [M](각 실패 실측)·통합비율 [?](결정적 실험 대기).

## 0. 한 문장
**모든 tool-use 실패 = LLM이 각 잔여 정보-갭의 *해소연산*을 오분류한다** — closed(GET/FIND/COMPUTE로 닫힘)를 추측하거나(과확신), open(ASK만)을 추측하거나, 갭을 아예 안 닫는다(under-action).

## 1. 해소연산 = [[16]] GET→FIND→INFER→ASK (INFER→ASK 결정론화·§4c)
각 연산이 서로 다른 open→closed 갭을 닫는다:
| 연산 | 닫는 갭 | world |
|---|---|---|
| **GET** | 알지만 미조회(메모리/컨텍스트) | closed |
| **FIND** | 열거가능·미열거(검색/DB·명시제약 filter) | closed |
| **COMPUTE** | 도출가능·미도출(정책/산술) | closed |
| **ASK** | 외부만 앎 | **open** |
- [[16]] §4c: **INFER(유효후보 추측=오류원) 삭제** → FIND후 1개면 사용·**≥2면 ASK**(결정론). 당신 "나열+ASK"=이것.
- COMPUTE = INFER의 *결정론* 부분(추측 아닌 도출)만 남긴 것 = 엔진 gate/calc.

## 2. 실패 = 오분류 두 방향
- **과확신(over-action)**: closed를 추측(GET/FIND/COMPUTE 안 하고 INFER) 또는 open을 추측(ASK 안 하고 committed). = 확신에 찬 오답.
- **under-action**: 갭을 아예 안 닫음(FIND/enumerate 안 하고 종료). = 조기종료.
- 공통 = **경계 오판**: "이게 closed(내가 닫을 수 있음)인가 open(물어야 함)인가"를 모름.

## 3. ★엄밀 매핑 — 우리 모든 실패가 이 하나로 (기존 궤적 데이터)
| 실패 | 실측 | 오분류 | 맞는 연산 |
|---|---|---|---|
| **⋈ mis-anchoring** | E-REGIME plan 560·n_disputes≥2=852/853·gold∈support 0/29 | ≥2 후보(open)를 confident 1개로 committed | **ASK**(≥2→ASK) |
| **reach under-action** | reach forensic closed-world 69.2% | 미완(open 갭) done 취급 | **FIND/GET**+H_min종료 |
| **reach open-world** | reach forensic 30.8% | 진짜 open | **completeness-ASK** |
| **compute** | C81 liability 51% 오답·systematic | 도출가능(COMPUTE-closed)을 추측 | **COMPUTE/verify** |
| **hallucination** | C45 날조 67% | open을 confident closed로 | GET/ASK |
- **⋈이 결정적**: 852/853 다중후보(≥2)인데 gold가 8샘플에 0 = 애매(open)를 확신(closed)으로 처리·재샘플로도 못 뒤집음 = **지식결함 아니라 경계오판**.

## 4. H_min = 최소질문 + [[16]] over-ask 문제 해결
- [[16]] §4c 열린 문제: "≥2→항상 ASK = over-ask 비용(측정필요)".
- **H_min이 답**: GET/FIND/COMPUTE로 닫을 수 있으면 먼저(질문0)·남은 진짜 open만 **H_min bit·VOI 순서로** ASK → 최소질문.
- 종료조건 = 잔여목표 엔트로피 > floor면 continue(문제2·under-action 차단).

## 5. H_min 기반 분해 = SOAR 일반화·MAKER 포섭 ([[44]])
- **분해 경계 = H_min > floor인 지점**(정보-갭). 각 subtask = "한 갭을 GET/FIND/COMPUTE/ASK로 닫기".
- **SOAR impasse**([[44]]): 이산 "결정불가"→subgoal. 우리 "≥2→ASK"=impasse 결정론판·**H_min=연속판**(impasse=갭 하나의 특수경우) → **SOAR 일반화**.
- **MAKER**: 구조적 micro-step(과분할). H_min=정보-갭 분할(필요분할만) → **MAKER 포섭**(최소·정보정당화).

## 6. novelty (PREEMPTION_SCAN 정합)
- 부품(GET/FIND/COMPUTE/ASK 각각·엔트로피-ask·decomposition)=선점→인용.
- **미선점 = 이 통합**: 모든 실패를 *하나의 연산-오분류*로 진단 + {GET/FIND/COMPUTE/ASK} 결정론 라우터 + H_min(종료·최소질문·분해). "correlated open problem"을 연산-오분류 taxonomy로 분해(MAKER blob 대비).

## 7. 정직한 잔여 2개 (과장 방지)
1. **compute는 순수 경계 아님**: 도출가능(closed) 내 추측 = "guess vs COMPUTE"(open-vs-closed 아님)·같은 연산-오분류 프레임엔 들어옴.
2. **경계 판정 자체의 calibration**: "≥2 감지"는 결정론(스키마)이나 "언제 confident해도 되나"(INFER→ASK 보정)는 [[16]] §6·C38/C42 미확립 = 진짜 make-or-break·learn 잔여.

## 8. ★결정적 실험 — 실행결과 + [[08]] 정직 교정
**`bank_failure_operator_label.py`(로컬 17궤적·gold-dispute 4139·실패 2858) 실행:**
- 분포: **COMPUTE 36.0%(1030) · FIND 25.4%(725) · GET/ASK 19.1%(547) · completeness-ASK 11.3%(323) · ⋈-ASK 8.2%(233)**.
- ⚠**"매핑률 100%"는 tautological**(4범주 exhaustive 설계) → *증거 아님*. **정보=분포**: 실패가 4연산에 **고르게 분산**(한 연산 지배 아님)=taxonomy 비-degenerate(필요조건).
- **★진짜 검증(non-trivial·반증가능) = operator-*choice* 오류인가(연산 안 씀) vs within-operator 실행오류(연산 썼는데 실패)**:
  - ⋈: **gold∈support 0/29**(E-REGIME) → 재샘플로 못 고침 = 연산-choice(ASK/verify 필요). ✅
  - compute: **✅실증(`bank_compute_regime.py`·880 cases)** — 결정론 verify(keystone lookup) **94.4%** vs agent in-situ **48.9%**(~2배). resample focused: **greedy-wrong 0**(모델이 격리서 liability 계산 가능). ⇒ **compute 실패=능력 아니라 부하/operator-choice**(모델 COMPUTE 가능한데 in-situ서 안 씀)·fix=부하없는 COMPUTE(verify 94%). voting=격리서 잉여·in-situ서 vacuous → **compute의 답은 항상 COMPUTE(voting 아님)**.
  - reach: 69% enumerable/queryable=FIND 안 씀(under-action). ✅ 부분.
- **★세 버킷 다 연산-choice 오류로 확인**(⋈ gold∈support0·compute 격리94%/insitu49%·reach 69%FIND미사용) = within-operator 실행노이즈 아님 → 통합의 non-trivial 검증 성립.

## 8b. ★"부하" 정체 규명 (`bank_load_diagnosis.py`·liability 오답 n=450·[M])
격리(94%)→in-situ(49%) gap="부하"를 gather-vs-compute로 엄밀 분해:
- **(b) COMPUTE 오류(입력=gold인데 liability틀림): 380(84.4%)** ← 지배
- (c) 혼합(입력+계산 둘다 틀림): 67(14.9%) · (a) GATHER(입력틀림·자기입력엔 정확): 3(0.7%)
- ⇒ **"부하"=context가 계산을 *망가뜨림*이 아니라 agent가 COMPUTE를 *실행 안 함***(정확입력 쥐고도 omit/None·capability는 격리 94%). = operator-misclassification(COMPUTE 미invoke).
- **★loop이 막나(사용자 #3)**: **YES** — 84.4%(pure compute)=입력정확→결정론COMPUTE(수집입력 엔진계산 94.4%)가 직접 닫음(loop이 강제 실행) / 14.9% 혼합=FIND/GET+COMPUTE / 0.7% gather=FIND/GET. **~85%가 COMPUTE 연산으로 직접 닫힘.** (부하=미실행이라 격리/강제로 닫힘·distraction-degrade면 안 닫혔을 것.)
- 예시: `11/07→11/14`(7일=$50) agent=**None**(계산 자체 skip). ⇒ 정체=계산-skip.

## 8c. ★부하 N-scaling — horizon=부하 직접 검정 (`bank_load_nscaling.py`·실패sim·[M])
liability-wrong·reach-miss를 N(sim당 gold dispute 수)으로 층화(field수는 전부 15-19 불변→축=N):
| N | reach-miss% | liability-wrong%(id-correct) |
|---|---|---|
| 1 | 36.8 | 23.6 |
| 2 | 32.1 | 13.2 |
| 3 | 26.6 | **60.4** |
| 4 | 17.5 | 48.6 |
| 5+ | 24.3 | 58.8 |
- **★compute: N=1,2(13-24%)→N≥3(48-60%) 계단상승 ~2.5×** → **실행 step의 부하가 N↑에 악화 = 강한 "horizon=부하" compute서 부분지지**(단 N=4<N=3·매끈X). = 과제 길수록 COMPUTE-skip↑ → 곱 붕괴 가속.
- **★reach: N↑일수록 *감소*(36.8→17.5)** → **부하 아니라 salience/engagement 기전**(유저가 다 나열하면 다 제출·N=1이면 미착수 잦음). horizon-load와 별개.
- **정정 결론**: 강한 버전(step당 부하가 N↑에 악화)은 **compute(=horizon AND-chain 실행)엔 성립**·**reach엔 불성립**. ⇒ **horizon 붕괴 = 실행 step의 N-악화 부하(compute) + salience(reach) + H_min갭(⋈/open)** — 하나 아니라 세 기전(앞 "약한 버전만"도 정정).
- caveat: N=dispute수 축만·field수 불변이라 field-level 부하 미분리·N=4 계단 비매끈(소표본 or 태스크 구성).

## 8d. ★모든 버킷 operator-skip 분해 (②·LOAD vs GAP·기존결과 종합)
각 버킷을 "isolated-capable but in-situ-skip(LOAD·loop이 강제로 fix)" vs "isolated서도 실패(GAP·ASK 필요)"로:
| 버킷 | isolated test | 판정 | 연산 | loop이 fix? |
|---|---|---|---|---|
| **compute** | 격리 94%·in-situ 49%(§8b) | **LOAD**(COMPUTE-skip)·N-scale(§8c) | COMPUTE 강제 | ✅ 결정론 엔진(94%) |
| **reach** | 완비도구有·69% 봤는데 미act(reach forensic) | **LOAD**(FIND-skip)·salience | FIND 강제(전계좌 열거) | ✅ 강제 열거+H_min종료 |
| **⋈** | 격리 formalize도 실패(E-REGIME voting 0.1%·gold∈support 0) | **GAP**(진짜 애매) | ASK(≥2 disambig) | △ ASK(H_min잔여) |
| **gather**(pin등) | 정보 모델밖(non-decidable) | **GAP** | ASK | △ ASK |
- **★2 LOAD(compute·reach)=loop이 연산강제로 직접 fix / 2 GAP(⋈·gather)=ASK(H_min잔여·작음)**.
- ⇒ **loop이 LOAD 2버킷(대부분)을 닫으면, 잔여=GAP 2버킷=H_min ASK(~7질문)**. horizon 붕괴의 지배분(compute-skip·reach-skip)이 loop로 닫힘 = **①E-PLAN 배선이 pass 올릴 것으로 예측**(다음 실증).
- **반증 hunt(TODO)**: agent가 *올바른 연산을 쓰고도* 실패한 케이스(within-operator)를 수동감사로 탐색 → 그 비율이 통합의 진짜 반증. (예: ASK했는데 user-답으로도 실패·COMPUTE 맞는데 정책 애매.)

## 8e. ★① E-PLAN 오프라인 replay ([[09]] 무료 先·`bank_operator_replay.py`·[M])
실패 sim의 id-correct dispute에 결정론 COMPUTE(liability lookup) 강제 적용:
- 오답 dispute 1576: **FLIP(COMPUTE로 완전정답) 224(14.2%)** · 부분개선 186(11.8%) · 미개선 1166(74%·non-compute도 틀림).
- **★sim-level: 실패 sim 1024 중 COMPUTE만으로 완전 pass 197(19.2%)**.
- ⇒ **연산 하나(COMPUTE)가 offline로 실패의 19.2%를 pass로 flip = "연산강제가 pass 올림" 무료 실증(①)**. 나머지 74%=non-compute(gather/⋈) 필드도 틀림→**FIND/ASK 추가 필요**(전체 loop=복리).
- **라이브 e2e(유료·[[09]] 승인·컨트롤러 배선 미구축[[14]])**: 배선된 loop이 이 19.2%+FIND+ASK를 실제 달성하나 = make-or-break. GAP 버킷(⋈/gather ASK)은 user 답 필요→offline 검증불가(gold쓰면 cheating)·라이브만.
- **caveat**: offline=COMPUTE 완벽적용 상한·라이브 formalize/실행 정확도로 감쇄·reach 0제출은 COMPUTE로 안 닫힘(FIND 별도).
