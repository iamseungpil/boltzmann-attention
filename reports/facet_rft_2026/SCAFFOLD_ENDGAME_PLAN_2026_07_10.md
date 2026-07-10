# SCAFFOLD ENDGAME — 남은 scaffold 레버 전량 소진 프로그램 (2026-07-10 · [D] · 리뷰 반영판)

> **목표**: retail 32B에서 scaffold-계열 레버로 살 수 있는 것을 전부, **측정된 상쇄 합성**(등대 §1.3)으로 소진한다.
> 입력: C21 격차 원장(vs o4-mini DB +23) · E-COMP census(`RETAIL_PASS_COMPOSITION_DESIGN §3b`·prov-arm 193 fails) ·
> `FIXABLE_FAIL_CENSUS_2026_07_06`(51 fails·결정론-fixable 51%) · 타 세션 리뷰(2026-07-10: 우선순위 5 + 정직 프레임 2).
> **프레임(정직)**: 큰 조각(날조)은 C45가 먹었다. 남은 것은 +2~9pp 조각들의 합성 — 이게 모트 논지의 실전이다.
> retail 쥐어짜기는 유일한 이득 축이 아니다: ①전이(banking/airline·특허 A 본체) ②crossover 사다리(E5) ③P3/P4가
> 더 큰 연구 이득이고, **이 프로그램은 그 증거를 두껍게 하는 역할**. GPU는 전이 실측에 우선권.

## 0. 라운드 구조
| 라운드 | 내용 | 비용 | 상태 |
|---|---|---|---|
| **R0** | E-COMP: COMP → 체크포인트 → COMP+D (게이트+prov+calc/nested+DISAMB) | 912 sims | 🔄 스모크 실행중 |
| **R1** | 신규 레버 4종 **무료 빌드+오프라인 검증** (아래 §1·병행 가능) | 0 | [D] |
| **R2** | re-census(COMP+D 실패 위) → **COMP2**(+precond/retry/feasibility·1런 합산) + **E-PLAN**(별도 1런) | ~912 sims | 승인 대기 |
| **R3** | E5 7B 사다리(unified 스택으로 crossover 강화) + 마무리 원장/논문 반영 | 소액 | 대기 |

## 1. R1 레버 4종 (전부 도메인-일반 엔진 + A2 사실·[[05]] 3질문 답 명기)

### L1. 대화-precondition controller (E-PRECOND) — over-action 30 sims(6.6pp) 표적·미시도 최대 조각
- 근거: C50("불가능성은 DB 아니라 대화에 산다 → 남는 후보=대화-precondition controller/ASK")·C25(8/12=정책-불가/철회 수행)·타 세션 리뷰 1순위.
- **v1 결정가능 코어 = 요청-대장(ledger)**: 대화에서 결정론 추출 가능한 신호만 —
  파괴적 write의 **대상 entity id가 사용자 발화에 미등장 ∧ 어시스턴트 제안→사용자 확인(CONFIRM_RE) 쌍도 부재** →
  **1회 재확인 재생성**(DISAMB형·deny 아님·게이트 금지 선례 [[06]] 준수). 철회(retraction) 감지는 **v1 제외**(semantic·정직).
- [[05]]: (1)도메인-특화 0(entity-id 추출=識別-hint 재사용·A2) (2)동결 0(재확인 후 선택=모델) (3)수행 0. 반대편 계측=**re-confirm된 정당 write의 이탈 census**.
- 기대: over-action 30 중 대상-미언급형 부분집합(census로 사전 측정·무료) × 전환율. C25상 8/12가 "요청은 있으나 정책-불가"라 **v1 사정거리는 부분집합** — 사전 census가 GO 게이트.

### L2. retry/diversify 이식 — ZERO_ATT 13(2.9pp·retry플래그 6)
- 기존 rule①②(정확-반복 차단·연속실패 K 다양화·apply() 경로 구현 존재)를 unified regen에 이식. 결정론·저비용.
- 반대편 계측: 차단이 유효 재시도를 막은 flip(=repeat가 사실 성공했을 케이스) per-case.

### L3. feasibility·refund-rule (A2 preconditions 확장) — FIXABLE census 1순위(~10건 상당)
- op-불가능성(부분취소=연산 부재 교시·G7 계열) + refund=원결제∪gift(정책사실·DB 5/5). E10(C50 NO-GO=DB-상태 게이트)과 구분: 이건 도구스키마/정책으로 결정가능.
- 엔진=기존 preconditions/constraints kind에 general op(`member_of`+resolver) 추가·필드=A2. **파생필드 리스크(G7 _note) → Δspurious≤0 계측 조건부**(제1원리).
- prov arm 잔여 WRONG_PAYMENT 8건과 겹침 — DISAMB와 이중커버(먼저 걸리는 쪽·마커로 귀속).

### L4. plan/execute live 배선 (E-PLAN) — MISSED+ZERO_NEV 47 sims(10.3pp) 표적·최대 headroom
- C14(부하 실증)·`plan_execute_orch.py` 빌드+오프라인 ALL PASS(2026-07-06·[[05]] 가드 내장)·**live e2e 미실행**.
- 남은 일 = tau2 에이전트 루프 통합: 첫 사용자 요청 확정 후 **plan-spec 1회 생성(모델)** → 결정론 controller가
  완료 추적·미완 항목 재프롬프트(값 주입 0·규칙0). 아키텍처 변경이라 **R2에서 단독 arm**(귀속 분리).
- 주의(FIXABLE §0 재프레이밍): 32B fail 16 중 14가 격리 계획선 core_ok·controller 0발화 — 이득은 batch/status 수정이
  아니라 **완료-추적(coverage walk)**에서 나와야 함. 설계 시 controller의 walk-reminder를 주기능으로.

### (옵션 L5) GROUND 후보-표면화 — zero-write repair 조각(상한 +4 sims·C27)
- C40(격리: getter-follow 16→56%)·rule-0 클린(기조회 출력만 표면화). R2 arm에 플래그 변형으로만(효과 작음·복잡도 증가 주의).

### L0. ★E-ISO — 정보-맞춘 3단 격리 replay (진단·레버 배정의 선결·무료급 · 2026-07-10 신설)
- **동기(사용자 재고 지시)**: "semantic 잔여가 사실 부하 아닌가" — C13(부하 없음=능력)은 **정보-빈약 프로브**였고 재검(E1′ PhA)은 C23으로 하향·미실행. 이후 증거는 부하 쪽: C59 열거 +31pp(구조화하면 고름)·C60 표현-민감(표면 분산이 결정 오염)·C14(격리 계획선 정답). §1.5 Q2를 semantic 슬라이스에 규율대로 재실행.
- **설계**: census 실패 결정점(WRONG_ITEMS/REF/PAYMENT 77 + C60 flip 쌍)마다 에이전트가 실제 가진 정보를 고정,
  **A 궤적-재현**(전체 prefix) / **B 격리-원문**(같은 정보·잡음 압축) / **C 격리-형식화**(명시 열거+요청 재진술).
  판정: B≫A=궤적-간섭 부하(분리/controller) · B≈A∧C≫B=**형식화-부하**(직렬화·열거=예상) · C 낮음=진짜 능력/경계(그때만 learn/scale).
- **출력 = learn 표적의 사전 필터**: E6′ 표적(paraphrase-invariance·over-action)을 "부하 몫 제거 후 잔여"로 정의 — [[13]]·§1.5 순서 준수. 기반 코드=`c51_disambig_boundary.py`(C59) 확장. 32B 로컬만(무료·GPU 한가할 때).
- **horizon 별도 처방 확인**: C43 "메모장 무효"는 날조 한정 — horizon/coverage엔 미검. banking "완주-후-불일치 45%"=단계별 결과 기록 부재 서명 → **E-PLAN walk + banking 절차조립 controller가 그 처방**(learn/scale 아님·지속=분해 축·taxonomy 정합).

### L6. prov p4-비용 차단 패키지 (2026-07-10 신설 · `RETAIL_PASS_COMPOSITION_DESIGN §3d/§3e`)
- 근거: C53 보강 — prov는 p1을 사고 **p4를 판다**(짝 −5.3pp·감쇠 +7.5pp 초과·prov-lost 15 중 12는 gpt-4.1 4/4 = 레버-유발 flaky).
- **P1 최소-침습 arg-머지**(regen 후 flagged-arg 외 원본 복원·결정론) · **P2 원리-디폴트 검증기**(=L3와 동일물·refund∈{원결제,명시 gift}) · **P3 write-only 트리거**(측정 플래그).
- 순서: COMP/COMP+D는 GO-arm 정확 중첩 그대로 先측정(귀속) → L6은 R2 수정 arm. 관측성([T2_PROV] stderr)은 확보됨.

## 2. R2 arm 구성·귀속 규율
- **COMP2 = COMP+D + {L1·L2·L3}** — 1런 합산이 정당한 이유: 세 레버의 표적 버킷이 **서로소**(over-action / zero-att / payment-feasibility)이고 발화 마커가 구분됨(stderr 카운터) → per-case 귀속 가능. 각 레버 반대편 계측 필수. 대조군=COMP+D(짝 flip census 1차).
- **E-PLAN = 별도 456** — 루프 아키텍처 변경이라 합산 금지.
- 판정(공통): 짝 flip ∧ Δspurious≤0 ∧ 위반0 유지 ∧ 표적 버킷 per-case 감소. 실패한 레버는 **개별 제거**(§1.3 죽은 레버 목록에 등재).

## 3. 경계·소유권 (정직·침범 금지)
- **learn 축(E6′)** = scaffold 아님·P4 본체·C38 데이터 게이트 선행 — 이 프로그램 밖(병행 축·⋈/변형 +6~8 조각의 장기 답).
- **slot-filling 라우터 v2**(결정 인터페이스 교체·T6 FILL .94~1.0) = **E-AMB T5-B·이론 세션 소유** — 중복 실험 금지. 우리 몫은 DISAMB(재확인형)까지·E-COMP 결과(switched census)가 T5-B의 사전 데이터로 흘러감. T6h 교훈 공유: 통계 디폴트=전이불가 트릭·원리 디폴트만.
- **scaffold로 안 닫히는 잔여**(정직): 정책-불가/철회 수행의 semantic 절반·잔여 ⋈·NL 보고형 3건 → P3 경계 주장 + learn/scale 축.
- **도달 목표(retail 32B)**: R0 후 0.63±0.03 → R2 후 **0.66~0.70**. frontier(0.741) 잔여 격차 = 경계 축 = 논문 주장 그 자체.
- **GPU 우선권**: 전이 실측(banking→airline) > R0/R2. R1은 무료라 항상 병행 가능.

## 4. 상태
- [D]·타 세션 리뷰 우선순위 반영(2026-07-10). R1 착수는 무료라 즉시 가능. R2는 R0 결과+사용자 승인 후.
