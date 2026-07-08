# Agentic AI · Graph RAG · Palantir AIP → 우리 프레임워크 활용 (딥리서치 회수·2026-07-06)

> 딥리서치(108 agent·25 소스·117 claim→25 검증: **18 confirmed / 7 killed**)의 **synth 단계가 placeholder("test")로 glitch** — RELWORK_LOAD_COT DR2와 동일 실패. 검증된 claim·votes·소스는 온전하여 **직접 회수·재합성**([[08]] 정합·killed claim은 인용 금지). 원본 = `tasks/wqfdne75b.output`.

## 0. 판정 요약 (adopt / adapt / avoid / already-ahead)
| 메커니즘 | 검증 | 우리 컴포넌트 | 판정 |
|---|---|---|---|
| Palantir Ontology(objects/links/**action-types**) | 3-0 ✓ | ABox·gate | **already-ahead**(gate) + **adapt**(ABox=objects+**links**) |
| Graph-RAG KG 엔티티(멀티홉·entity-resolution) | 3-0 ✓ (단 graph *구축*=LLM-soft, contested) | ⋈ present-scaffold | **★ADAPT**(DB서 결정론 entity-graph) |
| Routine 구조 planning IR | 3-0 ✓ (41→96% GPT-4o) | C1 plan/execute | **ADOPT**(우리 controller 검증) |
| 결정론 orchestration=agentic 동급 | 3-0 ✓ (통제연구) | scaffold thesis | **already-ahead·인용** |
| 실행-상태 구조 메모리(MAGE·Zep·A-MEM) | ✓ | C2 state-tracking | **ADAPT**(execution-state store) |
| 현실 점검(τ-bench<50%·prod agent<10step) | ✓ | 헤드라인 | **인용 지지** |
| **"구조로 소형=대형"(Qwen3-14B Routine 95.5%≈GPT-4o)** | **0-3 ✗ KILLED** | — | **AVOID**(인용 금지) |
| **FAOS 온톨로지-grounding 강주장**(p<.001·3-layer 22벤치 전이·inverse-parametric) | **전부 killed(1-2·0-3)** | relwork | **AVOID 인용 + 우리 relwork서 반증 활용** |
| Plan-then-Execute의 ReAct 대비 우위·injection 내성 | **0-3·1-2 ✗ killed** | — | **AVOID**(hype 미생존) |

## 1. 채택 제안 (우리 open-problem 직결)

### ★1. ⋈ 잔여 = DB서 결정론 entity-graph present-scaffold (최고 가치)
- **근거(✓)**: Graph-RAG는 **멀티홉/교차엔티티**서 vector RAG 능가(HippoRAG evidence-recall 87.9~90.9%); entity-resolution denoising(엔티티 40% 제거)이 QA 개선; SPLIT-RAG=결정론 subgraph 라우팅; OG-RAG=결정론 hypergraph 검색.
- **핵심 caveat(killed/contested)**: LightRAG/GraphRAG의 graph *구축*은 **LLM 추출=soft·비결정론**(claim "KG=결정론" 1-2로 약화). ⇒ graph가 결정론인 건 **깨끗한 구조화 데이터**일 때만.
- **우리 적용**: τ²는 **깨끗한 DB**(orders→items→products→users FK 링크)라 **LLM 추출 불요 → DB서 entity-graph를 결정론 구축**. ⋈ = 이 링크 그래프를 traverse해 후보 엔티티를 **관계맥락과 함께 present**(order→그 주문의 items/이름/status). = 우리 present/calc scaffold의 **링크-구조화 후보제시**. **첫 딥리서치(⋈ non-scale)와 수렴** — ⋈은 scale 아니라 결정론 graph present로.
- **판정 ADAPT**: Graph-RAG의 LLM-추출은 버리고(우리는 DB로 결정론), 링크-traverse present만 채택. **[[05]] 정합**(엔진=일반 graph traverse·ABox=도메인 링크).

### 2. ABox = objects + **links** + action-preconditions (Palantir 패턴)
- **근거(✓·전부 3-0)**: Palantir Ontology = Object Types(엔티티)+Link Types(관계)+**Action Types(거버넌스 write-back 트랜잭션)**. "action types=human/agentic 결정, pipelines=자동" 명시. Agent가 objects/links traverse(Shipment→Order→Customer), SQL JOIN 아님.
- **우리 적용**: **Action Types ≈ 우리 GATE**(실행 전 거버넌스 write) → **already-ahead**(우리=도메인-일반·결정론·모델무관·gate_spec 파라미터화 / Palantir=고객별 enterprise config·비전이). **ADAPT**: ABox를 flat operator 목록이 아니라 **objects+links+action-precondition**으로 구조화 → 링크가 ⋈에 관계골격 제공(제안1) + action-precondition이 feasibility 게이트 자연 표현.
- **전이성**: Palantir 온톨로지 자체는 **고객별=비전이**(우리 ABox-swap 전이의 foil). 단 objects/links/actions **패턴**은 도메인-일반.

### 3. C1/C2 controller 검증·정련 (Routine·결정론 orchestration)
- **근거(✓)**: Routine=plan↔execution 사이 구조 planning IR(step-order+parameter-passing)로 GPT-4o tool-calling **41.1→96.3%**. 통제연구(COBOL→Python)서 **결정론 orchestration=LLM-controlled 동급**(+토큰↓).
- **우리 적용**: **ADOPT** — 우리 `plan_execute_orch`(plan-spec+결정론 walk)가 바로 이 패턴. Routine의 parameter-passing IR = controller 정련 참고. 통제연구 = 우리 "결정론 scaffold=orchestration 대체" 외부 지지 인용.
- **★AVOID**: 같은 논문의 "Qwen3-14B distill Routine 95.5%≈GPT-4o(구조로 소형=대형)"은 **적대검증서 0-3 killed** → **인용 금지**. 구조는 *같은 모델*을 41→96 올리나, 소형=대형 증거는 미생존. 우리 crossover는 **우리 데이터로만** 주장.

### 4. 실행-상태 메모리(execution-state, not semantic) — C2 state-tracking
- **근거(✓)**: **MAGE** — 메모리를 semantic 유사도로 조직하면 실행 궤적이 fragment됨 → **실행-상태로 구조화**해야(장기 horizon). Zep/Graphiti=시간인지 KG(대화+구조데이터 융합). A-MEM=구조화 노트.
- **우리 적용**: **ADAPT** — controller의 state-tracking(class-A order-coverage)을 **결정론 execution-state store**(어느 엔티티 건드림·무엇 완료·pending)로. MAGE의 "state≠semantics"가 설계원칙. 우리 semantic-forensic의 상태추적 fix와 정합.

## 2. 인용·positioning (relwork/특허)
- **지지 인용(✓)**: τ-bench(gpt-4o<50% pass·pass^8<25%)·τ²(pass^1→pass^4 열화)=frontier 신뢰성 부족=우리 헤드라인. prod agent **68%가 <10 step서 human 개입·92.5% human 전달**=긴 자율실행 불가=우리 compliance/scaffold 논거. 결정론 orchestration=agentic 동급(통제연구)=우리 thesis 외부지지.
- **★반증 활용(killed)**: **FAOS(2604.00555)의 강주장 전부 적대검증 실패** — ontology-grounding 우위(1-2)·3-layer 22벤치 전이(0-3)·inverse-parametric(0-3). [[46]] FAOS=foil 정합·이제 **반증까지 확보** → relwork서 "FAOS의 전이-스켈레톤 주장은 재현·검증 취약"으로 양보 아닌 **구별** 강화.

## 3. 종합 (우리에게 없던 것 vs 이미 앞선 것)
- **이미 앞섬**: 결정론·도메인-일반·모델무관 GATE(vs Palantir 고객별 action-type); ABox-swap 전이(vs 온톨로지 비전이·FAOS 전이주장 반증); scale-불변 compliance.
- **우리가 얻을 것(신규 채택)**: (a) **DB-결정론 entity-graph로 ⋈ present**(제안1·최고 ROI·첫 딥리서치와 수렴), (b) **ABox를 objects+links+action-precondition으로 구조화**(제안2·⋈골격+feasibility 자연표현), (c) **execution-state store**로 state-tracking(제안4).
- **경계 준수**: 채택은 전부 결정론 scaffold/ABox 구조 층 — [[05]](도메인-일반·A2만 인스턴스)·[[11]](전이=ABox-swap) 위반 0. graph는 present-scaffold(엔진), 링크는 ABox. 학습레버 불변.
