# MSC — DAG-조건부 최소-충분 formalized context (입력측 offload) 설계 DRAFT — 2026-06-16

> 계기: M-A서 selector(출력측)가 in-domain 패배 + 그 원인이 *정보 비대칭*(arm B가 availability 못 봄)으로 판명 → 레버는 **출력 포맷이 아니라 *입력*에 있다**(사용자 통찰). 상위 = `ABOX_CONFIG_FORMALIZATION_DESIGN_2026_06_15.md`·이론 `ALGEBRAIC_DERIVATION_CLOSURE` §5.10(γ∘σ).
> 불변 = [[feedback-nl-formalize-llm-selection-deterministic]]·[[feedback-capability-vs-artifact-elicitation]]. ⚠**선행연구·신규성 = 딥리서치 `w2i00droj`(입력 formalize)·`wf_3f814306`(constrained-decode) 도착 후 §2/§7 정련.**

## 0. 한 줄
**DAG의 각 결정노드마다, 결정론 scaffold가 그 결정에 *필요한 최소 충분 정보*를 typed-dependency closure로 계산·formalize해 LLM에 전달한다.** = 입력측 결정론 offload. LLM의 잔여 추론을 floor 위·구조화 상태로 만들어 *정확도↑·(info-limited regime서) 작은 모델이 큰 모델 대체* 가능.
> **★리뷰 반영(2026-06-16)**: 이 설계의 **가장 강하고 방어가능한 기여 = floor 측정(L0–L3)이 write-벽이 info-limited인지 reasoning-limited인지 *측정으로 분리***(=M-A 리뷰의 fabrication-vs-reasoning 질문에 깨끗한 답). MSC *메서드*(typed-closure)는 근접 선행 多(schema-linking·structured RAG)이라 헤드라인 무게는 **측정 기여**에. 아래 4 scoping 必: (1) MSC=**Bfair-게이트 가설**(피벗이 confounded 0.406 선취 금지) (2) **reasoning 잔여는 두 아키텍처 불변**(MSC 천장=reasoning floor) (3) **scale 대체=info-limited regime 한정**([[feedback-capability-vs-artifact-elicitation]] 함정) (4) ablation은 **filter/format/info 3-way 분리**·"최소"≠순수결정론(투영=soft).

## 1. 동기 — ★Bfair-게이트 가설 (확정 전제 아님)
- 출력측 selector가 in-domain 패배(14B B 0.406 ≪ A 0.719)**처럼 보였으나** 원인=프롬프트 정보 비대칭(B가 availability·joint 카탈로그 못 봄)=**실험결함**. ⇒ 그 0.406은 **confounded** — "selector 약함"의 증거 아님.
- ★**따라서 MSC는 *Bfair-게이트 가설*이다**(피벗이 confounded 결과를 선취하면 안 됨):
  - **Bfair서 B≈A** → 출력-포맷 무관 → **입력이 레버**(MSC 정당).
  - **Bfair서 B>A** → 포맷도 도움 → MSC가 *유일* 레버 아님(출력측도 병행).
  - ⇒ §8-1(A vs Bfair)이 **MSC의 전제 검증 게이트**. 그 전까지 MSC=조건부 제안.
- (게이트 통과 시) 레버 = **입력 정보의 *충분성*+*형식***: (a) 충분히 (floor 위) (b) 형식화(덜 추론하게) (c) 결정공간 축소(가용 필터). lost-in-the-middle은 *형식화*로 처리(§3 — lossy NL-투영 회피).
- 두 실패 영역 분리: **info-limited**(답 미결정·어떤 모델도 실패) vs **reasoning-limited**(유도가능·깊은추론 필요). MSC는 전자를 *보장 제거*(floor 위)하고 후자를 *경감*(formalize).

## 2. 원리 — 최소성 = typed-dependency closure (≠ 임베딩 유사도)
각 결정노드 n의 LLM 결정(어떤 도구·인자·criteria)은 **typed 의존**을 가짐(도구 스키마+provenance):
- 예: "exchange 변형선택" 결정 = `f(NL, old_item.options, product.AVAILABLE_variants[관련 옵션키])`.
- **충분 context = 그 typed 의존의 결정론 full closure**(어떤 getter 출력·어떤 속성)을 world-state(ABox)서 계산·필터(available)·formalize(typed/표).
- = 결정에 대한 **충분통계량(sufficient statistic)**. 더 주면 distractor·long-context 저하, 덜 주면 floor 아래.
- ★**"최소"의 긴장(리뷰)**: full closure = 결정론·충분(보장). 단 "관련 옵션키만 *투영*"은 **NL→스키마 grounding 재도입**(M-A 값-역매칭) = 순수 결정론 아님. ⇒ **충분성=full closure로 보장**·context-bloat은 **형식화(표 구조)로 처리**(손실 NL-투영 회피). 투영을 쓰면 **soft step으로 명시**(결정론 보장 밖). = "최소"는 *형식*으로, "충분"은 *closure*로.
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

## 4. thesis 관계 — ★scope된 주장 (리뷰 반영)
- **입력 formalize가 정확도↑ (고정 모델)** — 문헌 풍부 예상(PAL/PoT·표추론·schema-linking; 딥리서치 확정).
- **★reasoning 잔여는 두 아키텍처(M-A selector·MSC)에 *불변*** (핵심): 둘 다 **fabrication + info-limitation만** 친다. "어느 variant·NL fallback 해석·synonym(Google Home→Assistant)"은 **NL 의미의 함수**라 typed-closure가 *안 준다*.
  - M-A: LLM이 criteria emit → 단 criteria가 틀린 추론일 수 있음.
  - MSC: LLM이 최소 context(후보 id 포함) 받아 emit → 날조 0이나 어느 후보는 여전히 NL-추론.
  - ⇒ **"selector 포섭/write-벽 제거"는 틀린 주장**. 올바른 주장 = **MSC는 결정공간을 *축소*(가용 필터 20→3)해 reasoning을 *쉽게* 하나, 환원불가 NL-의존 선택(reasoning floor)은 남긴다.** MSC 성공천장 = reasoning floor·**L0–L3가 그걸 잰다**.
  - 정련(나의 추가): floor 자체는 불변이나 **도달 *난이도*는 MSC가 공간축소로 낮춤** → "불변"은 floor(환원불가핵)에 한함·난이도는 줄어듦. (그래서 MSC가 selector보다 강한 레버인 건 맞음 — 입력공간을 줄이니까.)
- **★scale 대체 = info-limited regime 한정**([[feedback-capability-vs-artifact-elicitation]] 함정): "작은모델+MSC ≈ 큰모델"은 **write-벽이 info-limited일 때만**. reasoning-limited면 7B는 완벽 info에도 추론 못 함·큰 모델은 함 → 입력으로 scale 대체 *불가*. **floor 측정(L0–L3)이 regime 판정 게이트**·헤드라인을 그 regime에 조건화.
  - 정련(나의 추가): 엄밀히는 *연속*이다 — MSC가 난이도를 작은모델 capacity 아래로 낮추는 *만큼* 대체 성립. floor 측정이 그 위치를 정량. (보수적 헤드라인=info-limited 한정·정확한 그림=연속.)
- **transfer**: MSC 규칙(type(n)·deps(n))=도메인일반·**ABox swap = 데이터 swap**(규칙 불변) → 무재학습 전이. (§5.10 γ의 일부를 *입력준비*로 결정론화 → 학습할 γ 잔여↓.)

## 5. 최소성 정의 & 검증
- **정의(연역)**: typed-dependency closure (결정론·resolver 그래프서).
- **검증(실증)**: 정보 ablation × scale — 한 필드 빼서 정확도 하락=그 필드 필요 → **측정된 floor가 계산된 closure와 일치**.
- ★**3-way 분리(리뷰)**: L2(MSC)는 *적은 info(필터)* + *좋은 구조(형식)*가 섞임 → 둘을 가르려면:
  - **L0** marginal만 (floor 아래)
  - **L1** full catalog + availability, raw 포맷 (현 A/Bfair)
  - **L2a** MSC-info(가용 필터·후보 축소) + **raw** 포맷 → *필터/공간축소* 효과 분리
  - **L2b** MSC-info + **formalized**(표·정렬) → *형식화* 효과 분리(L2a 대비)
  - **L3** 거의 pre-resolved (극단 offload·reasoning floor만 남김)
  - ⇒ (L1→L2a)=필터/공간축소 / (L2a→L2b)=형식화 / (L0→L1)=정보 floor. 셋이 분리됨.
- L2b가 7B를 14B/32B-의-L1로 끌면 = **입력 offload가 scale 대체**(단 §4 info-limited 조건).

## 6. 평가 (사전등록)
- 지표: ①정보 floor(L0→L1 점프) ②MSC(L2) 정확도 vs A/Bfair(L1) @ 고정모델 ③scale 대체(L2@7B vs L1@큰모델) ④token 비용(MSC가 적은 token으로 동등) ⑤전이(ABox-swap: 같은 MSC 규칙, 새 도메인 카탈로그).
- 반례가드: MSC 이득이 단순 "정보 더 줌"인지(L1 already 충분) vs "최소+형식화"인지 — L1(full raw) vs L2(MSC) 직접대조로 분리.

## 7. Scope / 신규성 (정직·딥리서치 후 확정)
- **닫음 후보**: world-state가 typed-closure로 슬라이스 가능한 결정(exchange·정책게이트 prereq). 
- **안 닫음**: 자유텍스트 의존·closure 불명확한 결정·정책 NL→GATE_SPEC(별 라인).
- **신규성 — ★측정 기여에 무게(리뷰)**: 
  - *메서드*(typed-closure context 선택)는 **근접 선행 多** — schema-linking(text-to-SQL)·structured RAG·KG-grounded gen·tool-retrieval 타입필터. 헤드라인으로 약함. (딥리서치 `w2i00droj` 도착 후 차별 정밀화.)
  - *가장 방어가능한 기여* = **(ii) min-info floor *측정*으로 info-limited vs reasoning-limited 분리** — 측정 기여라 메서드보다 깨끗. = M-A 리뷰의 fabrication-vs-reasoning 질문에 깨끗한 답.
  - 부차 = (i) 최소성=typed-DAG closure(임베딩 아님) (iii) 입력 offload의 scale-대체 *정량*(info-limited 한정) (iv) ABox-swap 전이.
- **★constrained-decode 딥리서치(`whv0jinf9`·완료) 반영**: schema *제약*(JSON 포맷 자체 아님)이 추론손실 구동(Tam: Haiku 86.5→23.4)·완화법(CRANE/two-stage/in-schema-reasoning) **출판됨**(우리 Bcot/Btwo/structural_tag가 그 채택). **"reason-then-format + decoupled deterministic resolver를 *agentic NL→tool-arg*에 결합"은 미출판 gap** = 우리 selector/MSC 라인의 메서드-신규성 *후보*(단 측정 기여가 더 안전).

## 8. 다음
1. 진행중 GPU 실험(7B/14B·32B 재실행) 정리 → 14B B 궤적 트레이스(info-limited 비중) + A vs Bfair(비대칭 해소 확인).
2. **정보 ablation L0–L3 × scale** = MSC 1차 검증(harness 재사용·프롬프트 변형).
3. 딥리서치 2건(입력 formalize·constrained-decode) 도착 → §2/§7 선행연구·신규성 정련.
