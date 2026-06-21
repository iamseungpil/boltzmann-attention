# Future Direction — 능력-레버 프레임워크를 추론 일반(ARC-AGI 등)으로 일반화 (2026-06-21)

> ★**후속/병렬 방향. 현재 논문1(tool-use 비용)·논문2(path-selection) focus를 흐리지 마라**([[03-anti-drift]]). 이 문서 = 비전 박제·미검증·지금 빌드 아님.
> 상위 = `PLATFORM_DESIGN`·`CAPABILITY_LEVER_ALLOCATION`·`THESIS_STATEMENT`·`GENERATOR_ALGEBRA_DESIGN`·`NL_PROCEDURE_OFFLOAD_THEORY`(energy-Lie).

## 0. 질문 (사용자 2026-06-21)
우리 접근(능력-레버 배정·decidable→offload·behavior-vs-capacity·cheap-replication probe)이 tool-use를 넘어 **추론 일반(ARC-AGI 등 AGI 벤치)서 "LLM 추론이 어디까지 무엇을 하고·무엇을 어떻게 싸게 발전시키나"를 탐색·일반화**하는 데 쓸 수 있나?

## 1. 답 = 방법론은 일반화·비용 낙관은 약하게 전이
- **방법론(추론 해부 도구) 일반화 ✅**: 과제→atomic 능력 분해 → 각 능력 분류(**decidable→offload / behavior-default→싸게(프롬프트·최소학습) / capacity→scale·search**) → cheapest 레버 + scale-경계 *지도화*. = tool-use 특정 아님.
- **비용 낙관(작은+scaffold≈큰·싸게) 약하게 전이 ⚠️**: tool-use=난이도 대부분 decidable(offload 큼). 순수 추론=핵심 난이도가 *귀납 그 자체*(irreducible)→offload 여지 작음→**더 진짜 capacity/search-bound**. 비용 win 작고, 작은 모델이 싸게 못 따라잡는 영역 多 예상.

## 2. ARC-AGI 맵핑 (구조적으로 잘 맞음)
| ARC 단계 | 우리 분류 |
|---|---|
| 그리드 지각/객체 파싱 | 일부 decidable(offload) |
| **규칙 가설 귀납** | ★irreducible 추론(capacity 핵심) |
| **가설을 train 예제에 검증** | ✅ decidable→offload(결정론 인터프리터) |
| test 적용 | decidable 실행 |
- = **최강 ARC solver 구조**(LLM/search가 DSL 프로그램 제안 → 인터프리터 검증) = 우리 **generator+결정론-verifier 분담**([[10]]) 그대로. 프레임워크가 그 분해를 *예측*.
- **통합 기여**: program-synthesis·test-time search·PRM verifier를 **한 틀(decidable→offload·irreducible→reason)** 로 통합.

## 3. behavior-vs-capacity probe를 추론에 (핵심 일반화 도구)
- ARC 추상-유형별: 작은 모델이 *잠재로 함*(격리/프롬프트/search로 싸게=behavior-default) vs *진짜 못 함*(capacity). = scale-curve·격리·prompt-effect로 측정.
- ⇒ **"추론을 어디서 싸게 키우고(behavior+offloadable-verify) 어디가 진짜 scale/search-bound(capacity)인지의 *지도*"** = 추론 이해 기여(비용 절감보다 *경계 특성화*).

## 4. 정직 경계 (과주장 금지)
- ARC=설계상 scale-저항(암기 무력화)→"scale이 산다"도 약하고 답이 "대부분 capacity/search-bound"로 날 공산. 그것도 valid 발견.
- 검증 가능성 의존: ARC verify는 깨끗 decidable / open 추론(증명·commonsense)은 verify 약함→offload 레버 약화. 프레임워크 파워 = decidable 비율에 비례.
- **미검증**: 증거는 tool-use뿐. ARC 일반화=구조적 그럴듯·실측 0.

## 5. repo theory 연결 (씨앗 존재)
- **generator-algebra/closure**(`GENERATOR_ALGEBRA_DESIGN`) = ARC DSL-primitive 집합 대응(유한 생성원·닫힘).
- **energy-Lie**(`NL_PROCEDURE_OFFLOAD_THEORY §10`·boltzmann-attention) = 추론 기하(basin·전이·β*).
- = 추론 일반화의 이론 토대 일부 이미 있음.

## 6. 다음 (★현재 작업 후·드리프트 금지)
- 논문1/2 완료 후 검토. 1차 = ARC-AGI(or 1 reasoning bench)에 능력-분해 + generate/verify + behavior-vs-capacity probe 파일럿 → "추론의 offload/behavior/capacity 비율" 1점 측정.
- 헤드라인 후보: **"추론의 cost-anatomy 지도 — 무엇이 decidable-offload·behavior-cheap·genuinely-scale/search-bound인가"**(tool-use 프레임의 추론 확장).
- ⚠️ 지금은 안 함. C8 회수·C10 구현이 우선.
