# Steering-as-Control 실험 설계 (Phase 2c/2d+)

**작성**: 2026-05-29 (Phase 2c/2d 진행 중)
**상태**: 살아있는 설계 문서 — 다음 세션 실행용
**보완 대상**: `STEERING_EFFICACY_ANALYSIS.md` (H1–H7 효능 분석)
**Owner branch**: `facet-rft-2026`

---

## 0. 목적

단일·상수 steering(+1.5%p)을 넘어, **온톨로지 관계 기반의 실시간 제어(closed-loop facet-steering)** 로 확장하는 설계.
핵심 질문: *"steering을 RFT-동치까지 일반화하면서, RFT가 못 하는 실시간 적응 장점을 살릴 수 있는가, 그리고 그 효과 크기가 제어층으로 쓸 만한가."*

---

## 1. Framing: steering ↔ RFT

### 1.1 수학적 관계
- **상수 steering** `h ← h + α·v` = layer 출력 **bias 편집** = weight 편집의 부분집합(입력-독립, layer별 rank-1 가산).
- **mean-of-diff 벡터 = advantage-weighted(RWR) 1-step gradient의 bias-부분공간 사영** (contrast pair = high/low advantage). → 우리 steering = **"대조 advantage 하 bias-한정 1-step RFT"**.
- **완전 RFT = strict superset**: 모든 weight × 입력의존 × 다단계 × 진짜 reward. ⟹ RFT ≥ steering (도달 reward).
- **동치를 만들려면**: steering을 **학습가능 + 입력의존** `v(h)` 로 = **LoRA/adapter** 또는 학습된 컨트롤러. 이를 RFT reward로 최적화 = **parameter-efficient RFT 그 자체** (handoff Phase 3 cross-attn LoRA가 이것).
- 사다리: 상수 v → 게이트(입력의존 약) → 컨트롤러 v(h)(입력의존 강) → **학습형 v(h)=LoRA=RFT-동치**.

### 1.2 함수 class hierarchy — "RFT를 가변 steering/LoRA로"
모든 변형(RFT full-FT, static LoRA, 가변 LoRA, closed-loop steering)은 **같은 목적함수 `max E[R]`** 를 최적화한다 — 다른 건 오직 **함수 class(파라미터 family)**.

포함관계:
```
상수 steering ⊂ static LoRA ⊂ 가변(입력조건) LoRA ⊂ closed-loop steering controller
(bias 1-step)   (저rank ΔW)    (hypernetwork ΔW(x))    (state→개입, 에피소드 내 가변)
```

| 레벨 | 정체 | RFT 관계 |
|---|---|---|
| **L0 static LoRA** | 고정 `ΔW=BA`, RL reward 학습 | **= PEFT-RFT.** rank 충분 시 full-FT와 **exact 동치**(RFT 업데이트는 저-intrinsic-dim) |
| **L1 가변 LoRA** | `ΔW(x)=g_φ(x)` hypernetwork, φ를 RL 학습 | **conditional/fast-weight RFT.** static LoRA(상수 g)를 특수해로 포함 → **⊇ RFT** |
| **L2 closed-loop steering** | `Δh(state_t)` 에피소드 내 가변, 컨트롤러 RL 학습 (base=고정 환경) | **control policy / meta·hierarchical RL.** RFT가 *구조적으로 못 하는* 실시간 적응 추가 |

**정확한 명제**:
- **static LoRA = RFT** (exact, rank 충분 시).
- **가변 LoRA/steering ⊇ RFT** — 상수 컨트롤러(=static LoRA=RFT)를 **특수해로 포함**하고 실시간 적응을 더함. *엄밀 등호 아님*(가변이 상위 class). "RFT를 가변화" = **RFT 목적함수를 더 큰 조건부·closed-loop class 위에서 푸는 것**.

**구성법**: base freeze, 동일 RL machinery(GRPO/PPO/RWR)로 학습 대상만 교체 — (L1) `H(state)→LoRA params` hypernetwork / (L2) 컨트롤러 `π(steer|state)`, action=steering 벡터.

**No-free-lunch**: 표현력↑ → RFT 천장 접근/초과하나 **cheap·reversible·modular·real-time 장점 상실**. 약한 끝 = actuator ceiling(C5), 강한 끝 = full 조건부 → RFT 비용으로 회귀. **연구 베팅 = "최소 충분 표현력" + L2의 고유 실시간성(에피소드 내 적응)**.

**사다리 연결**: Rung 6(학습형 v(h)/LoRA) = **L0→L1 = "RFT 가변화"**, Rung 7(외부 RL 컨트롤러) = **L2**. 정당성(C5 actuator·C1 감지·C4 데이터)은 하위 Rung에서 선검증.

### 1.3 함의
- 상수 steering = **RFT의 lower-bound / feasibility probe** (선형 부분공간에 reward 방향 있나 싼 확인).
- **steering null ⊬ RFT null** (RFT는 비선형·입력의존 이득 추가 보유). 우리 약한 결과는 RFT 하한만 말함.

### 1.4 실시간 steering의 고유 장점 (RFT가 *구조적으로* 못 함)
1. 재학습 0 (벡터 덧셈, 무시할 비용) 2. 즉시 온라인 적응(ms) 3. per-context/도메인/유저 특화(1 base + 라이브러리) 4. 합성성(inference blend) 5. 가역·무망각·안전(weight 불변) 6. 연속 knob(α) 7. **★에피소드 *내* closed-loop 제어** (RFT는 가중치 고정이라 불가).
- **종합**: RFT=느린 기반(고천장) + steering=빠른 실시간 제어층 → **보완재**. steering = test-time adaptation 층.
- **단서**: 장점은 **효과 크기가 충분할 때만** 가치화 (현재 +1.5%p·gating null → 크기 충분성이 핵심 미지수).

---

### 1.5 비용 분리와 LoRA-RFT → 가변화 2단 경로 (실시간 적응)

**핵심 구분 — 학습 비용 ≠ 실시간 적응 비용**:
- LoRA-RFT가 비싼 건 *학습*(rollout+RL). 학습된 LoRA는 적용은 싸지만 **정적**(실시간 변경 불가).
- 가변 steering의 실시간 변경은 학습을 *피해서* 얻는 게 아니라, **학습 1회 + inference에서 선택/합성/생성**으로 얻음.

**구조적 수렴**: 입력의존 잔차 개입 Δh(x)를 self/cross-attn 출력에 더함 = 그 sublayer 가중치를 입력별로 바꾸는 것과 동치(ΔW·x). 게이팅/라우팅/하이퍼넷을 얹으면 = 『가중치 변경처럼 작동하는 가변 steering』. → **가변 steering ≡ (게이트된) LoRA** (별개 아님, 같은 대상의 두 이름).

**불가능 정리**: 『학습 0 + 실시간 + RFT급』 3자 동시 달성은 불가 — 하드코딩 가변 steering = bias-1step-RFT(class-hierarchy 최약점)이고 §2.1/§2.4 null이 증거. RFT 성능엔 최적화가 *어딘가* 반드시 필요(공짜 점심 없음).

**실시간 + RFT급을 동시에 얻는 2 아키텍처** (일회성 학습 산물을 실시간 가변 적용):

| | 학습(1회, 비쌈) | inference(실시간, 쌈) | 가변 단위 |
|---|---|---|---|
| **A. LoRA 라이브러리 + 라우팅** (mixture-of-LoRA) | 강한 LoRA N개 오프라인 | 문맥/도메인/턴별 hot-swap·blend | 턴별 |
| **B. 하이퍼넷/컨트롤러 per-input 생성** (amortized RFT) | 컨트롤러 1회 | 문맥별 개입 실시간 생성 | 토큰/턴별 |

→ 둘 다 정적 LoRA-RFT가 못 하는 **에피소드 내 실시간 적응**을 줌. 이것이 가변 steering의 진짜 가치이며 **불가능하지 않음**.

**2단 경로 (순서 불가역)**:
1. **LoRA-RFT (강도 확보)** = Rung 6 진입: 학습된 강한 개입이 RFT급 lift를 내나 + 온톨로지가 *reward*로 유효한가 확인 (actuator 상한 확보).
2. **가변화 (실시간 적응)** = Rung 7 / 위 A·B: 그 개입을 conditional/routed/per-turn으로 올려 실시간 적응·모듈성 부여.
- **역순 불가**: 2를 1 없이 하드코딩하면 §2 null로 회귀. ⟹ **LoRA-RFT는 가변 steering의 경쟁자가 아니라 *전제*.**

## 2. 지금까지의 실증 결과 (context)

### 2.1 게이팅 null (Phase 2c, N=60)
| 조건 | pass^1(all) | transfer%(svc) | med_chars |
|---|---|---|---|
| α=0 | 0.176 | 27.5% | 12259 |
| raw α=0.5 | 0.192 | **37.5%** | 13714 |
| decay | 0.183 | 30.0% | 12778 |
| orth | 0.200 | 25.0% | 14010 |

**효과 없는 정확한 이유**:
1. **decay 전제가 거꾸로** — validates는 집계로 escalation을 *촉진*(transfer +10pp). 후반 steering 제거 → 도움신호 제거 → transfer↓(10중 5 task에서 하락). −0.50 "trap"은 예외였지 규칙 아님.
2. **orth ≈ 무작동** — 제거량 = cos(v,h) ≈ 0(고차원) → orth ≈ raw. (확정: cos(v,h) 실측 필요, GPU ~5분.)
3. **검정력 부족** — 기반 효과 작고(±noise) N=60 CI ±0.10. transfer%↔pass^1 디커플링(노이즈 지배).

### 2.2 상보 구조 (표상 공간, layer 12–14)
- 전체 평균 cos=0.016 (거의 직교).
- **AXIS-1 (실행/지속/검증 ↔ 이탈/복구/에스컬레이션)**:
  - EXEC 군집 {step_realizes_tool, parameter_feeds, **retry_after_fail**, and_join, validates(+enables)} ↔ RECOVER 군집 {**error_fallback**, conditional_on, compensates} (cos −0.3~−0.5).
  - **retry_after_fail(EXEC) vs error_fallback(RECOVER) = 상보 대립** ← −0.50 메커니즘의 축.
- 기타 축: mandatory_in_flow↔optional_in_flow(−0.31), plan_step_precedes↔plan_revised_to(−0.28), conditional_on↔and_join(−0.39).

### 2.3 성공 driver
- service task: **transfer_to_human_agents 발동 = 성공의 압도적 결정요인** (성공 trial 83% 발동 vs 실패 18%). transfer는 대화 **90% 지점**(콘텐츠 ~2000tok 후) 발동 = "위치"가 아닌 "실패누적 후 전이" 사건.

---

### 2.4 관계-sweep 인과검증 (Phase 2d, N=60, α=0.5, gate=none) — Rung 2 결과
| relation | pass^1(all) | pass^1(svc) | transfer%(svc) |
|---|---|---|---|
| error_fallback (RECOVER 극) | 0.183 | 0.300 | 30% |
| retry_after_fail (EXEC 극) | 0.217 | 0.400 | 45% |
| (validates raw, N120) | 0.192 | 0.300 | 37.5% |
| (a0, N119) | 0.176 | 0.275 | 27.5% |

- **AXIS-1 인과 미지지/반전**: 표상에서 retry↔error_fallback은 대립(cos<0)이나 행동상 **retry_after_fail(EXEC 극)이 transfer·pass 최고** — 예측 반대. → 표상 상보 ≠ 인과 상보.
- 모든 조건 noise band [0.176-0.217] 내(N≤120). 효과 비특이적(모든 steering이 transfer를 a0 대비 올림), pass^1 미반영(H2 ceiling 지지).
- **C5(actuator 강도) 부정 경향 / C3(relation→behavior) 미지지.** → §9 피벗.

## 3. 검증 링크 (closed-loop controller가 서려면 — AND 조건)

| 링크 | 주장 | 검증 | 상태 |
|---|---|---|---|
| **C1 READ** | live 궤적에서 relation 상태·변화를 신뢰성 있게 감지 | 프로브를 trajectory에 적용·정답대조 | 미검증(H3) |
| **C2 PREDICT** | relation-변화가 성공/실패를 예측 | 로그에서 transition↔outcome | 미검증 = Rung 3 |
| **C3 WRITE** | relation steering이 인과적으로 행동을 바꿈 | 실제 주입·측정 | **미지지(2026-05-29, §2.4)** |
| **C4 DATA** | 엣지 가치 추정에 충분한 데이터 | 분산/표본 분석 | 미확보(RL엔 부족) |
| **C5 STRENGTH** | action(steering) 효과가 제어할 만큼 큼 | 효과 크기 측정 | 미입증(gating null) |

> 루프는 C1∧C2∧C3(+C4/C5). **직관 기반 루프는 이미 한 번 틀림(decay)** → 내용은 측정 증거에서.

---

## 4. 실험 사다리 (각 rung: 가설 / 방법 / 지표 / go-no-go)

- **Rung 1 — 게이팅 (decay/orth)**: ✅ 완료. **null** (§2.1). → 게이트-다운 방향 기각.
- **Rung 2 — (a) 상보 relation 인과검증 [C3,C5]**: ✅ 완료(2026-05-29). **결과: AXIS-1 인과 미지지/반전** (error_fallback 0.183/transfer30% < retry_after_fail 0.217/transfer45%, 예측 반대). C5 약함·C3 미지지 → §9 피벗. (§2.4)
  - 가설: error_fallback → transfer% ↑; retry_after_fail → ↓ (AXIS-1 인과성).
  - 방법: gate=none, α=0.5, layers 12–14, N=60(trials2). error_fallback(GPU0) ∥ retry_after_fail(GPU1). baseline=raw α0.5(N120, 37.5%), α0(27.5%).
  - 지표: transfer% + pass^1 + verbosity. **go**: transfer%가 error_fallback > validates > retry_after_fail 순으로 갈림(인과 상보 확인) → relation-교체가 유효 actuator.
- **Rung 3 — (b) 성공-조건부 관계-전이 그래프 [C2]**: 로그에서 (relation-state → state) 전이에 P(성공|전이) 라벨. **무예산.** go: 착취 가능 구조(전이가 outcome 가름) 존재 → 컨트롤러 정당화.
- **Rung 4 — (c) 룰엔진 컨트롤러 (read→decide→write, 심볼릭) [C2+C3]**: P1 룰 `count(retry_after_fail)≥2 OR error_fallback.observed → validates⇒error_fallback`, turn-level(Path B), prefix-cache가 과거 보존. 지표: transfer% / pass^1.
- **Rung 5 — 프로브 신뢰성 검증 [C1]**: PCLI 프로브가 live 궤적의 relation을 맞추나(정답 대조). go 후에만 프로브-라우티드 컨트롤러.
- **Rung 6 — 입력의존/학습형 컨트롤러**: v(h) 학습(또는 LoRA) → **RFT-동치 수렴**.
- **Rung 7 — 외부 RL/확률그래프 컨트롤러**: trial로 그래프 가치 추정 → offline RL 또는 룰. **C4(데이터) 충족 시에만**; 현 데이터론 룰 우선.

### 병행(무예산/저비용)
- **orth cos(v,h) 실측** (Rung 2 종료 후, GPU ~5분) — orth 무작동 가설 확정.
- **terseness/결정성 벡터 추출** (오프라인) — caveman의 벡터화. 스케줄/룰에 drop-in.

---

## 5. 온톨로지 관계 활용 taxonomy (상보-전환 외)

| 군 | 모드 | 비고 |
|---|---|---|
| **A 합성** | 가산합/부분공간/관계산술 (직교성 활용) | 단일효과 작음 주의 |
| **B Read** | **관계=RFT 과정보상**(thesis 핵심) / facet 타임라인 / 이탈탐지 | C1 의존 |
| **C 제거** | 유해 relation ablation(retry 등 빼기) / 직교투영 제거 | Rung 2서 retry 유해 시 즉시 |
| **D 그래프구동** | 워크플로 그래프 상 "다음 필수 엣지"로 steer / precond→effect 체이닝 | sparse 이산액션에 적합 |
| **E Training-time** | relation 커리큘럼 / **relation-조건 LoRA(=RFT)** / 보조헤드(H3 완화) | 큰 commitment, 큰 lift 가능 |
| **F Meta** | facet 좌표 분해 / steerability 랭킹 | — |

유망 top3: **B-4(RFT 과정보상)**, **D-9(그래프 구동)**, **C-7(유해 relation ablation)**.

---

## 6. 상보 관계 + 전환 규칙 카탈로그

상보 쌍(steering 양극): **P1 validates/retry ↔ error_fallback**(★), P2 mandatory↔optional, P3 plan_committed↔plan_revised/backtrack, P4 checkpoint↔backtrack, P5 precondition↔effect, P6 achieves_goal(over-run 방지).

전환 규칙(이벤트 구동, 양방향):
```
INIT steer=validates@0.5
R1 IF count(retry_after_fail)≥2 OR error_fallback.observed OR no(state_transition⁺)≥2턴 OR mandatory pending
   THEN steer=error_fallback@0.4[+orth]
   (B→A) IF state_transition⁺ OR precondition newly_met THEN steer=validates@0.5
R2 IF repeated_remediation≥3 THEN steer=error_fallback@0.6
R3 IF goal-state met THEN steer=achieves_goal (확인·종료)
강화: A극서 state_transition⁺ 연속 → α↑
```
탐지: 심볼릭(기기 status 미개선/툴 실패 토큰) 우선, 프로브(H3 검증 후).
> ⚠️ §2.1 발견(validates가 escalation 촉진)에 따라 R1의 "validates 제거" 효과는 재검토 필요 — Rung 2가 error_fallback이 *더 강한* escalation actuator인지 확인 후 규칙 확정.

---

## 7. 지표 · 검정력 · go/no-go

- **지표**: pass^1(all/svc), **transfer_to_human_agents 발동률**(핵심), verbosity(chars/turns), (확장) relation-transition 경로.
- **검정력**: N=60 CI ±0.10 → 방향탐색용. 유망 조건은 **N≥120(가능시 240)** 재실행.
- **Phase 3 gating**: lift ≥ +3%p in ≥2 모델 → Phase 3(LoRA=RFT). 현재 미달 → 제어 사다리로 크기 충분성 탐색.
- **Strategic NO-GO**: 32B/70B는 8B서 actuator 유효(C3,C5) 확인 후에만.

---

## 8. 정직한 한계 / 리스크
- **천장**: steering은 기존 행동 재가중일 뿐 새 능력 X (H2). 적응이 능력을 요구하면 RFT만 가능.
- **actuator 약함(C5)**: gating null·+1.5%p → 제어 레버가 작을 위험. Rung 2가 1차 판정.
- **감지 신뢰성(C1)**: 실시간 제어는 센싱만큼만 좋음 — H3 미검증.
- **데이터 기근(C4)**: RL 컨트롤러엔 현 N의 수십 배 필요 → 룰 우선.
- **제어 안정성**: 실시간 보정 과보정/진동 → 게이팅 비자명.
- **귀인 복잡도**: 컨트롤러+다벡터+게이트 → ablation 필수. user-sim 노이즈 교란.

---

**핵심 한 줄**: *상수 steering은 "bias-1step-RFT"라는 RFT의 정찰병이며, 이를 입력의존·학습형(LoRA=RFT-동치)으로 올리되 RFT가 못 하는 실시간 closed-loop 제어 장점을 살리는 것이 목표 — 단 그 전에 actuator 효과 크기(C5, Rung 2)와 감지 신뢰성(C1, Rung 5)을 검증해야 한다.*

---

## 9. 결과 종합 및 LoRA-RFT 피벗 (2026-05-29 PM) — DECISION

**종합**: Qwen-7B에서 상수 single-relation steering(validates/error_fallback/retry_after_fail) + 게이팅(decay/orth)이 **전부 baseline noise band [0.176-0.217] 내** (N≤120). 표상-공간 facet 상보 구조가 인과 행동 제어로 이어지지 않음(AXIS-1 반전). 효과 약하고 비특이적 → **C5 부정 경향, C3 미지지.**

**결정 (사용자, 2026-05-29)**: class-hierarchy(§1.2)가 가리키는 대로 — 상수 inference steering(=bias-1step-RFT, 최약점)에서 hand-tuning 중단, **학습 끝(LoRA-RFT, L0=PEFT-RFT)으로 피벗. power test(N=240) 생략, 본체 직행.**

**근거**: (i) 상수 steering null은 class-hierarchy상 예상된 것(최약점); (ii) H2 capability ceiling 지지 → RFT는 능력 추가 가능; (iii) 원 Go/No-Go = facet-RFT +5%p in ≥2 domain.

**다음 = §1.5 2단 경로의 1단계 (LoRA-RFT 강도 확보; 성공 시 2단계 가변화로 승급)**: base=Qwen2.5-7B + LoRA(attn/mlp), objective=τ² task reward(GRPO/RWR), rollout=telecom + gpt-4o-mini user_sim(예산 주의), eval=동일 30-task pass^1. 기존 infra 정찰: run_lora_hybrid_pipeline.sh, scripts/ocq/lora_train_metatool_v*.py (RFT/SFT 여부 + peft/trl 가용성 확인 필요). ⚠️ RL rollout이 OpenRouter 대량 소비 → 소규모부터, 잔액 모니터링.

> 마스터 설계서: reports/EXPERIMENT_DESIGN_v1_7_facet_rft.md v1.21 §7 Phase 2a 박스 / §10 Go/No-Go / §12.

---

## 10. 도메인 전이 + 합성-온톨로지 학습 (경제성·일반성 축) — 2026-05-29

**동기**: 학습 비용 ≈ 도메인당 학습 × N도메인. 온톨로지 관계가 *도메인-일반*이면 **1회 학습 후 전이**로 비용 amortize — thesis의 경제 정당성이자 novelty(cross-domain pre-defined ontology). → 마스터 Phase 5에 묻힌 도메인 일반화를 **중심 축으로 격상**.

**4 도메인** (τ²-bench): telecom(2285, small 20) / retail(114) / airline(50) / banking_knowledge(97, nonmeta 13).
- ⚠️ banking_knowledge는 knowledge-QA 성향 가능(tool-action 비중 확인 필요) — 도메인 style 이질성은 전이의 강한 시험.
- 자산: per-domain ontology 추출본 telecom/retail/airline ✓, **banking 미추출(AFOD 필요)**. 합성 relation 데이터 contrast_pairs_v3.json ✓(steering 추출원).

**전이 주장 3분리**: (a) 스키마 전이(auto, 약한 saving), (b) **개입(벡터/policy) 전이 — 진짜 prize**, (c) reward 전이 — **이미 rule-based 도메인-일반(공짜)**. 핵심 = (b).

**일반성 gradient** (학습원 agnostic 정도 ↑ → 주장 깨끗·경제성 ↑, 전이 리스크 ↑):
1. 합성 텍스트 → steering = 이미 보유, **null**(§2).
2. **합성 agentic → RFT → 4도메인 zero-shot = 북극성**(최강 주장, 高리스크: synthetic→real gap > 도메인간 gap).
3. 1도메인 RFT → others 전이.
4. in-domain RFT = floor.

**de-risk 측정 사다리**: Floor(ontology-개입이 real in-domain서 작동? telecom real LoRA-RFT, null이면 상한 막힘) → Gradient(in-domain → 1도메인전이 → 합성→4) 로 *어디서 전이가 깨지는지* 위치 측정.

**4도메인 전이 실험** (각 도메인 d):
- 조건: B0_d(무개입, 선측정) / SYN→d(합성학습 zero-shot) / TEL→d(telecom학습 전이) / d-RFT(in-domain 상한)
- 지표: pass^1, 전이율 = (X→d − B0_d) / (d-RFT − B0_d)
- 판정: SYN→d > B0_d (4도메인 평균 유의) → 도메인-일반 온톨로지 1회 학습 → 어디서나 (★최강 주장)

**선행 필요**: (i) retail/airline/banking **B0 baseline**(현재 telecom만), (ii) banking ontology **AFOD 추출**, (iii) **합성 agentic 학습원 설계**(42-relation 템플릿을 합성 tool-use 시나리오로 확장 — 성패 핵심 변인), (iv) reward는 schema에서 auto(공짜).

**Go/No-Go 강화**: in-domain +X%p 가 아니라 **재학습 없이 held-out 도메인서 +X%p 전이**. §1.5 가변 steering 비전(아키텍처 A)과 결합 시 = **도메인-일반 relation-LoRA 라이브러리 + 도메인별 라우팅**(1회 학습 → 어디서나).

---

## 11. Facet-guided distillation (capability 주입 + cold-start 해소) — 2026-05-29

**동기**: 7B 최대 블로커 = (i) H2 capability ceiling(steering·self-RFT가 *없는 능력*을 못 만듦), (ii) self-GRPO가 baseline 0.18에서 성공 rollout 희소 → sparse-reward cold-start(gradient 기아). 강한 teacher가 7B가 스스로 못 만드는 성공 궤적을 제공 → 두 문제 동시 우회. → distillation은 옵션이 아니라 **floor(LoRA-RFT)가 lift를 보이게 하는 실질 enabler.**

**형태 = facet-guided (온톨로지-필터) distillation**:
- teacher(GPT-4o / Qwen-72B) 궤적 생성(합성 or 멀티도메인) → ontology-violation reward로 필터/가중(precedes/requires/mutex 준수 궤적만) → student(Qwen-7B) LoRA-SFT.
- 문서 T4-RFT의 rejection-SFT 옵션의 **teacher 버전** → Phase 4 path β 변형 슬롯. GRPO보다 쌈(teacher 생성 1회, on-policy 루프 없음).

**★confound 격리 (thesis 보호, 필수)**: plain distillation = teacher 행동 복제 ≠ 온톨로지 기여. plain이 큰 lift이고 필터가 +ε면 기여는 distillation이지 온톨로지가 아님. → **반드시 ablation: unfiltered distill vs facet-filtered distill** 로 온톨로지 marginal value 격리 (cross-domain·합성-학습과 동일 규율).

**2단계 결합 (우아)**: ① distill → capability 확보(성공률↑, reward non-sparse화) → ② facet-RFT → 온톨로지 adherence 정제. distill이 cold-start 깨고 RFT가 facet-specific 미세조정.

**경제/북극성 정합**: distill-once(합성 or 1도메인) → 4도메인 zero-shot = 합성-온톨로지 북극성의 실질 수단(RL보다 쌈 + capability 주입). 산물은 static(실시간 X)이나 §1.5 2단 경로 1단계(강도 확보)이므로 무방 → 이후 가변화 승급.

**teacher 옵션**: GPT-4o(API, trajectory/rejection distill) / Qwen-72B(local GPU, logit·representation distill 가능, activation 접근 ✓). 마스터 Tier-4 API 모델(상한선 참조 전용)을 **teacher로 재활용**하는 신규 lever.

**한계**: confound(위, 최우선) / capability 주입은 온톨로지와 orthogonal → 귀인 흐림 위험 → related work 포지셔닝(ontology-guided distill vs plain distill novelty) / static.
