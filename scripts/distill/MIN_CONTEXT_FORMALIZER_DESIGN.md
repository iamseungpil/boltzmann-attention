# MSC — DAG-조건부 최소-충분 formalized context (입력측 offload) 설계 DRAFT — 2026-06-16

> 계기: M-A서 selector(출력측)가 in-domain 패배 + 그 원인이 *정보 비대칭*(arm B가 availability 못 봄)으로 판명 → 레버는 **출력 포맷이 아니라 *입력*에 있다**(사용자 통찰). 상위 = `ABOX_CONFIG_FORMALIZATION_DESIGN_2026_06_15.md`·이론 `ALGEBRAIC_DERIVATION_CLOSURE` §5.10(γ∘σ).
> 불변 = [[feedback-nl-formalize-llm-selection-deterministic]]·[[feedback-capability-vs-artifact-elicitation]]. ⚠**선행연구·신규성 = 딥리서치 `w2i00droj`(입력 formalize)·`wf_3f814306`(constrained-decode) 도착 후 §2/§7 정련.**

## 0. 한 줄
**DAG의 각 결정노드마다, 결정론 scaffold가 그 결정에 *필요한 최소 충분 정보*만 typed-dependency closure로 계산·필터·formalize해 LLM에 전달한다.** = 입력측 결정론 offload. LLM의 잔여 추론을 floor 위·최소·구조화 상태로 만들어 *고정 모델서 정확도↑·작은 모델이 큰 모델 대체* 가능.

## 1. 동기 (확정된 관찰)
- **출력측 selector는 in-domain 패배**(14B B 0.406 ≪ A 0.719) — 단 원인=프롬프트 정보 비대칭(B가 availability·joint 카탈로그 못 봄)=실험결함.
- ⇒ 진짜 레버 = **입력 정보의 *충분성*+*형식***. 같은 정보를 (a) 충분히 (floor 위) (b) 최소로 (lost-in-the-middle 회피) (c) 형식화(덜 추론하게) 주면 정확도↑.
- 두 실패 영역 분리: **info-limited**(답 미결정·어떤 모델도 실패) vs **reasoning-limited**(유도가능·깊은추론 필요). MSC는 전자를 *보장 제거*(floor 위)하고 후자를 *경감*(formalize).

## 2. 원리 — 최소성 = typed-dependency closure (≠ 임베딩 유사도)
각 결정노드 n의 LLM 결정(어떤 도구·인자·criteria)은 **typed 의존**을 가짐(도구 스키마+provenance):
- 예: "exchange 변형선택" 결정 = `f(NL, old_item.options, product.AVAILABLE_variants[관련 옵션키])`.
- **최소 충분 context = 그 typed 의존의 결정론 closure**(어떤 getter 출력·어떤 속성)을 world-state(ABox)서 계산·필터(available)·투영(관련 옵션키만)·formalize(typed/표).
- = 결정에 대한 **충분통계량(sufficient statistic)**. 더 주면 distractor·long-context 저하, 덜 주면 floor 아래.
- ★임베딩 retrieval 아님 — **타입-그래프 closure**(정확·완전·도메인일반). (대형 A2는 §6.5b 임베딩 *선택* + closure *완전성* 혼합 가능.)

## 3. 메커니즘 (per-node)
```
for each decision node n in (incrementally-built) DAG:
  1. type(n)         = 스키마+provenance로 n이 소비하는 정보클래스 식별 (도메인일반 규칙)
  2. deps(n)         = world-state(ABox)서 typed-dependency closure 계산 (resolver 의존그래프)
  3. slice           = deps(n) 필터(available/유효) + 투영(관련 속성만) = 최소 충분
  4. ctx(n)          = formalize(slice)  (typed/표·pre-derived·정렬)
  5. LLM 결정         = decide(NL, ctx(n))   # 잔여추론 = 작음·floor 위·구조화
  6. 결정론 검증/resolve (기존 selector+resolver 또는 concrete — ctx 충분하면 concrete로 족함)
```
- **scaffold가 derivation(필터·조인·투영)을 흡수** = 입력측 offload. LLM은 *남은* 결정만.

## 4. 왜 thesis를 강화하나
- **입력 formalize가 정확도↑ (고정 모델)** — 문헌 풍부 예상(PAL/PoT·표추론·schema-linking; 딥리서치 확정).
- **입력 offload가 scale 대체 가능** → 작은 on-prem 모델 + MSC ≈ 큰 모델 + raw = **주권 결과**([[feedback-sovereignty-not-small-model]]). selector(출력)가 못 산 걸 입력측이 살 수 있음.
- **transfer**: MSC 규칙(type(n)·deps(n))=도메인일반·**ABox swap = 데이터 swap**(규칙 불변) → 무재학습 전이. (§5.10 γ의 일부를 *입력준비*로 결정론화 → 학습할 γ 잔여↓.)
- **selector와의 관계**: MSC가 최소충분 context(가용 후보집합)를 주면 LLM 잔여결정이 작아져 **concrete-emit로 족함**(selector vs concrete 무의미해짐)·날조도↓(id가 minimal context 안에 있음). ⇒ MSC가 더 강한 레버·selector는 multi-turn 날조방지 특수case.

## 5. 최소성 정의 & 검증
- **정의(연역)**: typed-dependency closure (결정론·resolver 그래프서).
- **검증(실증)**: 정보 ablation L0–L3 × scale — 한 필드 빼서 정확도 하락=그 필드 필요 → **측정된 floor가 계산된 closure와 일치하는지**. (L0 marginal=floor아래 / L1 full+avail / L2 가용필터+투영=MSC / L3 거의 pre-resolved.)
- L2(MSC)가 7B를 14B/32B-의-L1로 끌면 = **입력 offload가 scale 대체** 입증.

## 6. 평가 (사전등록)
- 지표: ①정보 floor(L0→L1 점프) ②MSC(L2) 정확도 vs A/Bfair(L1) @ 고정모델 ③scale 대체(L2@7B vs L1@큰모델) ④token 비용(MSC가 적은 token으로 동등) ⑤전이(ABox-swap: 같은 MSC 규칙, 새 도메인 카탈로그).
- 반례가드: MSC 이득이 단순 "정보 더 줌"인지(L1 already 충분) vs "최소+형식화"인지 — L1(full raw) vs L2(MSC) 직접대조로 분리.

## 7. Scope / 신규성 (정직·딥리서치 후 확정)
- **닫음 후보**: world-state가 typed-closure로 슬라이스 가능한 결정(exchange·정책게이트 prereq). 
- **안 닫음**: 자유텍스트 의존·closure 불명확한 결정·정책 NL→GATE_SPEC(별 라인).
- **신규성 후보(딥리서치 확인중)**: "입력 formalize가 추론↑"·"lost-in-the-middle"은 *기존*. 우리 후보 = **(i) 최소성을 *typed-DAG closure*로 정의(임베딩 아님) (ii) min-info floor를 *측정*해 info-limited/reasoning-limited 분리 (iii) 입력 offload의 *scale 대체*를 정량 (iv) ABox-swap 전이**. 디코딩 처방(CRANE)·일반 RAG와 차별. = `w2i00droj` 도착 후 박제.

## 8. 다음
1. 진행중 GPU 실험(7B/14B·32B 재실행) 정리 → 14B B 궤적 트레이스(info-limited 비중) + A vs Bfair(비대칭 해소 확인).
2. **정보 ablation L0–L3 × scale** = MSC 1차 검증(harness 재사용·프롬프트 변형).
3. 딥리서치 2건(입력 formalize·constrained-decode) 도착 → §2/§7 선행연구·신규성 정련.
