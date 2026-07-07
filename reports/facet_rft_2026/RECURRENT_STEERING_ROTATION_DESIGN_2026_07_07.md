# 학습-wing 메커니즘 후보 B — recurrent steering×rotation (damped-Hamiltonian DEQ adapter) 설계 (2026-07-07)

> **★★2026-07-07 RETRACTED — 기계적으로 성립하지 않음(cold analysis). 대체=`LEARNED_WING_MECHANISM_DESIGN_2026_07_07.md`.**
> 철회 사유: (1)LIE_GROUP 회전은 KV *양자화* basis 변경=**내적 보존=출력 불변**(⟨Mk,Mq⟩=⟨k,q⟩)이라 계산을
> 더하지 않음 — "재-attend"는 틀린 서술. (2)residual-stream loop은 attention/MLP 재실행 없이는 활성을 relax만
> 할 뿐 coverage/⋈(문맥 재읽기=attention 필요)를 못 함. (3)A<1(수축)=답이 아니라 활성의 완화판으로 수렴=
> 계산적 무능(안정성이 표현력을 죽임). (4)RLVR(토큰-공간)↔latent DEQ loop(implicit gradient)=미정합·구호 조합.
> (5)STEERING §8 H2(재가중)·C5(약함)는 열린 리스크 아니라 negative prior. 아래는 기록 보존용(오설계 표본).

> **위치**: `NEXT_LEVERS_DESIGN_2026_07_07.md`(rev2) §3 "학습된 도메인-일반 스킬 wing"의 **메커니즘 후보 B**.
> 후보 A = four-bench SFT TBox. 둘 다 **동일 make-or-break(τ² ABox-swap 전이로 잔여 닫는가)**로 겨룬다.
> **통일하는 두 자산(=샛길 아님·기존 이론 병합)**:
> - `STEERING_CONTROL_DESIGN.md`: steering 사다리 L0(static LoRA=RFT-exact)→L2(closed-loop controller·
>   Rung 6-7·학습형 v(h)). **정직한계 §8: C5 actuator 약함(+1.5%p·gating null)·H2 재가중일 뿐 새 능력 아님.**
> - `math/paper/lie_group/LIE_GROUP_UNIFICATION.md`: KV 캐시 RoPE 토러스 T^{d/2}⊂SO(d)·회전작용
>   $e^{tA}$($A\in\mathfrak{so}(d)$)·SpinQuant=Cayley-SGD로 $\mathfrak{so}(d)$ 위 데이터-의존 최적회전 학습.
> **불변**: rev2 make-or-break 면제 없음·[[13]] 학습먼저·[[05]] 도메인-일반·[[11]] τ² swap·[[03]] 무료관문 먼저.

---

## 0. 한 줄
잔여(reasoning: coverage·⋈·criterion)를 닫을 **학습-wing을 "SFT로 스킬 주입"이 아니라 "test-time에 표현을
답-평형으로 relax시키는 학습된 recurrent 연산자"**로 구성한다. 연산자 = **회전(보존·KV-rotation/Lie) ∘
steering(소산·closed-loop controller)** = **감쇠 해밀토니안 흐름** = DEQ(고정점=암묵 무한깊이). base freeze·
adapter만 RLVR 학습(=Rung 6-7). **A<1 수축은 steering α로 튜닝·측정**(별도 난제 아님).

---

## 1. 연산자 정의 (두 이론에 앵커)
hidden state(또는 KV) $x_t$ 위의 recurrent 연산자:
$$x_{t+1} = \underbrace{M(\Omega_\theta)}_{\text{보존·회전}}\cdot\big(x_t + \underbrace{\alpha\, g_\theta(x_t)}_{\text{소산·steering}}\big),\qquad M=\exp(\Omega),\ \Omega\in\mathfrak{so}(d),\ \Omega^\top=-\Omega$$
- **소산항(steering)**: $x\to x+\alpha g_\theta(x)$ = 답-다양체로 끌어당김. **STEERING §1 Rung 6 학습형 $v(h)$**(=입력의존
  LoRA=RFT-동치). α = 마찰·연속 knob(§1.4).
- **보존항(회전)**: $M=\exp(\Omega_\theta)$ = **LIE_GROUP Mode I 회전작용**·norm 보존($\langle Mx,My\rangle=\langle
  x,y\rangle$·명제 2.1.3). 매 loop 문맥을 "다른 각도로 재-attend". $\Omega$ = 저계 skew(블록 SO(2)^{d/2}⊂SO(d)
  계층서 자유도 선택) → **SpinQuant식 Cayley-SGD로 $\mathfrak{so}(d)$ 위 학습**.
- **loop = 감쇠 해밀토니안 이산화** → 평형 $x^*=R_\theta(x^*)$로 relax = **DEQ**(Bai 2019·암묵 무한깊이) =
  Parcae $L(T)=L_\infty+Z e^{-zT}$(열평형 완화).

### 1.1 ★A<1이 공짜·튜닝·측정가능 (Q3 해소)
- 순수 회전은 spectral norm=1(A=1·궤도만). steering 추가 → $\|R'\|\approx(1-\alpha)<1$.
- **$A=(1-\alpha)$ = STEERING이 이미 잰 그 α**. A<1이 별도 난제가 아니라 **기존 alpha_grid/phase2 데이터+T-sweep으로
  측정**. 회전은 안정성 깨지 않고(등거리) 탐색만 추가.

### 1.2 LoRA-후보강·RLVR (사전학습 아님)
- base freeze. adapter = $\{\Omega_\theta(\text{저계 skew}),\ g_\theta(\text{LoRA-MLP})\}$·α(스칼라/스케줄).
- 학습 = STEERING L2 controller RL machinery(GRPO/PPO/RWR)·검증가능 보상(τ²/four-bench 성공). loop 수 T·연산자
  = 정책. = **"RFT 목적함수를 closed-loop recurrent class 위에서"**(STEERING §1.2 L2).

---

## 2. 왜 이게 rev2 잔여에 맞나 (그리고 안 맞을 수 있나·정직)
- **맞는 근거**: 잔여=reasoning(다단계 처리)·knowledge 아님(§forensic)·loop이 잘 사는 유형(Parcae). test-time에
  "더 오래 생각"=표현을 답-평형으로 relax = coverage 전수·⋈ 해소 같은 다단계를 암묵깊이로.
- **★안 맞을 수 있는 정직 한계(중심 리스크·STEERING §8)**:
  - **C5(actuator 약함)**: 단일-step steering=+1.5%p·gating null. loop이 이 약한 actuator를 **누적**해 유의미해지는가,
    아니면 Parcae처럼 **weak single-step에서 곧 포화**하는가 = **make-or-break의 핵**. 선행 데이터는 단일-step 약함을
    이미 말함 → loop-누적 가설은 미검·낙관 금지.
  - **H2(재가중일 뿐 새 능력 X)**: steering은 기존 행동 재가중. 잔여가 **새 능력**을 요구하면 소산-steering만으론
    부족. → **회전이 델타**: KV를 *이동*(재-attend)시켜 재가중 넘어 새 구성 접근 가능성. 단 선형-ish 반복은
    표현력 한계 → **loop에 비선형 read 필요**(§5 리스크).

---

## 3. Feasibility 관문 (무료→저→유료·[[09]]·[[03]])
| # | 관문 | 방법 | GO/NO-GO | 비용 |
|---|---|---|---|---|
| **F1** | **A<1 존재·튜닝** | 기존 steering α로 $A=1-\alpha$·hidden 반복 수렴률 측정(phase2 데이터+소규모) | A<1 achievable | 무료~저 |
| **F2** | **loop이 누적하나(C5 극복)** | 한 잔여 probe(coverage/⋈)서 T=1,2,4,8… → accuracy가 **단일-step 위로 유의 상승**하나·$L(T)$ 포화점 | 상승 유의(≥+3%p·§7 gating) vs 곧 포화 | 저 |
| **F3** | **비선형 read 필요성** | 순수 affine 연산자 vs (steering+회전+1 nonlinear layer) 비교 | 비선형 없이도 이득? | 저 |
| **F4** | **RLVR 학습·전이** | adapter RLVR(four-bench·도메인-일반) → **τ² ABox-swap 전이** measure | rev2 make-or-break | 유료·승인 |

**F1·F2가 핵심 무료-관문**: A<1인데 T↑가 단일-step 위로 안 오르면(=C5가 loop으로도 안 풀림·Parcae 조기포화)
→ **후보 B 조기 기각**·four-bench SFT(A)나 rev2 경계-지도로. RLVR(F4)은 F2 통과 후에만.

---

## 4. 후보 A(SFT) vs 후보 B(recurrent) 비교 축
| 축 | A: four-bench SFT | B: recurrent steering×rotation |
|---|---|---|
| 스킬 설치 | 가중치에 직접(op·state-track) | test-time 동역학(평형=답) |
| prior-negative | §17/§23D 역전이(domain-op SFT) | C5 약함·H2 재가중(steering §8) |
| 델타(왜 다른가) | §19·§22·§28(생성원·decomp·native) | 회전(H2 넘음)·loop-누적(C5)·일반 추론절차 |
| A2/도메인성 | 학습벤치·τ² swap | 동일(도메인-일반 연산자) |
| cost | 학습 1회·추론 정상 | 추론 **T× 느림**(cost-knee 계상) |
| 가역/무망각 | SFT=가중치 변경 | **base freeze·adapter·가역**(STEERING §1.4) |
- **둘 다 rev2 make-or-break로 판정**·둘 다 무료 관문(A=전이델타 실증 / B=F1·F2) 먼저. **병렬 겨룸**·이기는 쪽이
  학습-wing.

---

## 5. 정직한 리스크 (과열 방지·[[03]])
1. **C5(중심)**: 단일-step actuator 약함 실측 → loop-누적이 이를 넘는지 **F2가 1차 판정**·못 넘으면 사망.
2. **H2**: 재가중 한계 → 회전이 넘는지·**비선형 read(F3)** 필요할 수 있음(그럼 "경량"이 약해짐).
3. **표현력**: 순수 affine 반복=고정점 relax만(진짜 다단계 추론 X). $g_\theta$ 비선형·회전이 이를 완화하나 미증명.
4. **Parcae 포화**: test-time loop은 학습깊이서 포화(L∞)·"공짜 무한개선" 없음. T×latency=cost 페널티(계상 필수).
5. **RLVR-recurrent 안정성**: 흐름 RL은 발산/붕괴 위험·α 수축이 안전판이나 미검.
6. **전이 면제 없음**: F4가 rev2 make-or-break. B라고 역전이 면제 아님.
7. **열역학 이론 분리**: damped-Hamiltonian/DEQ/열평형은 **사후 설명틀**로만·경험(F1-F4) 먼저·이론 먼저 얹기 금지.

---

## 6. 단계 실행 (리뷰 후)
1. **F1 A-probe**(무료): phase2 steering 데이터 재분석 + Qwen hidden 반복 수렴률 → A(α) curve.
2. **F2 loop-gain probe**(저): coverage/⋈ probe서 T-sweep accuracy·$L(T)$ 피팅·포화점. **GO/NO-GO**.
3. **F3 비선형 필요성**(저): affine vs +1층 비교.
4. F2 GO → **F4**: adapter RLVR(four-bench·도메인-일반)+τ² 전이(유료·승인). 후보 A와 병렬/비교.
5. 분기: B 전이 성공→학습-wing=B(가역·무망각 이점) / 실패→A 또는 rev2 경계-지도.

## 7. 사전등록·불변
- **성공 기준(F2·F4)**: T↑가 단일-step 위로 유의(≥+3%p·STEERING §7 gating 재사용)·τ² 전이로 잔여 닫힘.
- **over-claim 금지**: "loop이 scale 대체" 단정 금지(Parcae 포화·C5). 실패면 후보 A/경계-지도(rev2 §4).
- **[[05]]**: 연산자·학습=도메인-일반·A2 불변. **[[13]]**: 학습(A/B) 먼저·scaffold/scale 최후.
- **cost 정직**: T×latency를 paper cost-knee 목적함수에 명시(B의 대가).
