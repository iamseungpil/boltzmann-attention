# B(절차예산) 스케일 실험 — superlative/comparative를 *in-head 매핑*으로 푸는 임계 크기 측정 (설계) — 2026-06-17

> 명제(사용자): "the most/the better"는 알고리즘 혼재지만 **파라미터가 충분히 크면 *얕은 모델도 forward-pass 매핑*으로 푼다**(절차 직렬실행 X·병렬회로 학습). **어느 크기서 풀리는지 측정.** = 유계 절차예산 B(L,width)가 연산 깊이 d를 넘는 *임계 크기* 측정. 상위 = `LIE_ABSTRACTION_THEORY_2026_06_17.md §7c-7d-bis`.

## 0-bis. ★★THESIS 재정렬 (2026-06-17·사용자) — *작은 모델이 결정론 구조로 superlative를 극복하나*가 핵심
스케일 사다리(언제 매핑으로 풀리나·§아래)는 **baseline**이고, **우리 thesis = 7B(소형·주권타깃)가 *결정론 구조의 도움*으로 most/best/better를 극복해 *거대모델 매핑을 비용·정확서 지배*하나**다.

### 핵심 교정 — 연산-IR (정적 $select 폐기)
우리 정적 `$select`가 superlative서 실패한 건 **답-criteria("zoom=max")를 강제**했기 때문(max는 값이 아님). **고친 결정론 구조**:
- **LLM = 연산 *명명*(얕은 인식·B 안)**: `{op: argmax/argmin/rank-k/filter/comparative, attr: <X>, among: {filter}, anchor: <현재>, fallback: [...]}`. 답이 아니라 *어떤 연산*인지.
- **엔진 = 실행(깊은·B=∞)**: filter→sort by attr→extremum/rank/anchor-비교→id. 정확·임의 N·임의 깊이.
- 예 "highest resolution waterproof": LLM `{op:argmax, attr:resolution, among:{waterproof:yes}}` → 엔진이 max 계산. ("less bright": `{op:comparative, attr:brightness, dir:less, anchor:current}` → 엔진.)

### ★THESIS 시험 (4 arm·superlative/comparative 케이스)
| arm | 정체 | 예측 |
|---|---|---|
| **A. 7B in-head 단독** | NL→답 id 직접 | **실패**(파일럿 d3=0.20) |
| **B. 7B + 결정론 구조** | NL→연산-IR(LLM 명명)→엔진 실행 | **≈ oracle**(극복) |
| **C. 거대(32/72B) in-head 단독** | 매핑 | 부분(d3≈0.50)·비쌈 |
| **D. oracle** | gold 연산 직접 | 1.0 |
- **thesis 주장: B ≈ D ≫ A · 그리고 B ≥ C (구조가 scale을 대체·소형으로 거대 매핑 지배).**
- **분해 측정**: (i)*연산-인식 정확도*(LLM이 옳은 op-IR emit하나 = 얕은 분류·소형도 높아야·이론 예측) (ii)*end-to-end*(인식+엔진). (i) 높으면 (ii)≈oracle = thesis.
- 실패 분해: 연산-오인(argmax↔argmin·"less"방향·attr오인·filter누락) = 인식 실패(개선타깃)·엔진은 무오류.

### 스케일 사다리의 역할 (baseline)
거대모델 in-head 매핑의 *임계크기/비용*(아래 §1-7)이 = **arm C가 *얼마나 비싼지*** = B(작은+엔진)가 지배하는 폭. "풀린다(매핑)"의 증명이 곧 "그 비용을 offload가 친다"의 증명.

## 0. 한 줄
**합성 통제-깊이 selection 과제를 *in-head(CoT 없음)*로 1.5B→235B 스케일 사다리에 돌려, superlative(max/min)·ranked(2nd)·comparative(anchor)·nested 각각이 *매핑만으로* 풀리는 *임계 파라미터*를 측정. + 결정론 offload(=1.0·B=∞) 대조로 "임계가 크면 offload가 비용 지배"를 박는다.**

## 1. 왜 합성인가 (τ² 폐기·통제)
τ²는 *깊이 × comprehension × 어휘 × 가용성* 혼재 → B 측정 오염. 합성으로 **깊이만 통제·표면 등방화·order는 정수로 명시**(order-지식 confound 제거) → *순수 절차예산*을 잰다.

## 2. 과제 (통제-깊이 in-head selection)
- **카탈로그**: N개 item = {item_id(랜덤), 속성들}. 속성: 범주형(랜덤 토큰) + **수치 ordinal**(정수 1..99·max/min 명확).
- **쿼리(깊이 d 통제)**:
  | d | 연산 | 예 | 추측기준선 |
  |---|---|---|---|
  | 1 | filter | "attr_a=v 인 item" | 1/N |
  | 2 | extremum∘filter | "attr_b가 **최대**인 item (attr_a=v 중)" | 1/(필터후 수) |
  | 2 | comparative∘anchor | "현재(anchor)보다 attr_b **큰** item" | — |
  | 3 | ranked∘2filter | "attr_a=v ∧ attr_c=w 중 attr_b **2번째 최대**" | — |
  | 4 | nested | "(filter) 중 attr_b max들 중 attr_d min" | — |
- **출력 = item_id**(in-head·직접). gold = 결정론 계산. **등방화**(예제마다 새 스키마·랜덤 id) → *item 암기 불가·연산만*.
- **변수**: 깊이 d ∈ {1..4}·**리스트크기 N ∈ {5,10,20,50}**(같은 d라도 N↑=스캔깊이↑) · 연산타입.

## 3. 통제 (엄격)
- **단일 family**(Qwen2.5 dense) 핵심사다리 → L·width 함께 자람(아키텍처 confound 0). 각 크기의 **L(layers)·d_model 기록**.
- **bf16**(양자화=용량 confound 회피·불가피한 대형만 표시).
- **in-head·CoT 없음**(매핑/공간 B 측정)·temp0·고정 프롬프트·**n≥200/셀**(검정력·N별).
- 조건축: **{in-head}** 주(매핑) · **{+CoT}** (B 외부연장 대조) · **{결정론 offload}** (=1.0·B=∞ 기준선).
- 추측기준선 명시(acc ≫ 1/N 이어야 연산 수행).

## 4. ★스케일 사다리 (1.5B → 235B)
| 크기 | 모델 | L·width(기록) | 누가 |
|---|---|---|---|
| 0.5B·1.5B·3B·7B | Qwen2.5-{0.5,1.5,3,7}B | | **woori** (단일 GPU) |
| 14B | Qwen2.5-14B | | woori/coworker |
| 32B·72B | Qwen2.5-{32,72}B (bf16) | | **coworker** (H100×4 TP) |
| ~235B | **Qwen3-235B-A22B** (MoE) | active 22B | coworker (★MoE confound: active≠total·*확장점*·dense사다리와 분리해석) |
- **dense 깨끗 사다리 = 0.5–72B**(Qwen2.5)·235B는 MoE라 *별표 확장*(총-params vs active 구분).
- 가능하면 Llama-3.1-{8,70,405}B로 *교차-family* 1줄(family-불변 확인).

## 5. ★측정·판정 (반증가능)
- **acc(d, N, S)** 전수 → 각 (연산 d, N)의 **임계크기 S\***(acc≥0.9 최소 S·또는 로지스틱 변곡).
- **핵심 곡선**:
  1. **acc(S) | 고정(d,N)** = 스케일링 곡선 — *임계 S\* 존재·매핑으로 풀림 증명*(사용자 명제).
  2. **S\*(N) | 고정 연산**(예 max) — *N↑이면 필요크기↑인가* = B(width)가 스캔깊이로 자람.
  3. **S\*(d)** — 깊이별 임계.
  4. **in-head vs +CoT**: CoT가 작은 S서 큰-S-in-head를 대체하나(외부 B연장).
  5. **offload=1.0 전 d/N** — 깊이-불변 기준.
- **사전등록 예측(이론)**: (a)각 (d,N)에 *유한 S\* 존재*(충분히 크면 매핑으로 풀림). (b)**S\* 단조↑ in (d,N)**(깊을수록·길수록 큰 모델). (c)in-head S\*가 *L에 더 묶임*(깊이예산=직렬층)·width는 스캔폭. (d)CoT가 S\*를 *낮춤*(외부직렬). (e)offload는 S 무관 1.0.
- **thesis 결론**: S\*(d,N)가 *크면* → **소형+결정론(sort/select)이 *거대모델 매핑*을 비용·정확·주권서 지배**(같은 답을 7B+엔진이 1/N×비용으로). 즉 *"풀린다"의 증명이 곧 "offload가 옳다"의 증명*(매핑 비용 = 거대모델).

## 6. 산출물/구현
- `synth_depth.py`(신규): 통제-깊이 카탈로그+쿼리+gold·등방화·N/d/연산 인자.
- `depth_eval.py`(신규): served 모델 in-head/CoT eval·acc(d,N) per-case·결정론 oracle 대조.
- `depth_scale_batch.sh`: 크기별 serve→eval→집계(woori 소형·coworker 대형).
- 결과 박제 `M_A_RESULTS §15`·곡선 S\*(d,N) phase diagram.

## 7. 비용/feasibility (★가능한가 = 예)
- **추론-only**(학습 0). 소형(≤14B)=woori 단일GPU 수분/크기. 32/72B=coworker H100×4 TP. 235B=coworker 대형노드+양자화.
- 총 ≈ 7크기 × {in-head,CoT} × 4 N × 4 d × n200 = 추론량 큼이나 *대형이 비용 지배*·병렬. **woori 1.5–14B / coworker 32–235B** 분담이면 1-2일.
- 235B는 *확장점*(MoE 별표)·핵심 dense 사다리(0.5–72B)만으로도 명제 판정.

## 8. 한 줄
**"superlative는 충분히 큰 모델이 in-head 매핑으로 푼다"를 1.5B→235B 사다리서 *임계크기 S\*(d,N)*로 증명·측정. 그리고 S\*가 깊이·N로 자람을 보여 *소형+결정론 offload가 거대모델 매핑을 지배*함을 같은 실험서 박는다.** = 유계예산 B(L,width) 이론의 엄격 통제 시험.
