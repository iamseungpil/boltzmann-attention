# 학습-wing 메커니즘 재설계 — reachability-gated 도메인-일반 reasoning (2026-07-07)

> **위치**: `NEXT_LEVERS_DESIGN_2026_07_07.md`(rev2) §3 "학습된 도메인-일반 스킬 wing"의 **정합 재설계**.
> **폐기**: 후보 B(recurrent steering×rotation)=기계적 불성립·RETRACTED(`RECURRENT_STEERING_ROTATION_DESIGN`
> 헤더). steering=재가중·rotation=출력불변 basis변경·loop=활성 relax → reasoning 못 함.
> **원리**: scaffold(present+gate)는 정보를 *제시*하고 compliance를 *보장*한다(done). 학습-wing은 **제시된 정보를
> 올바르게 *쓰는* reasoning**을 설치한다. 그 mechanism을 **냉정한 필요조건(reachability)**으로 설계한다.
> **불변**: rev2 make-or-break(τ² ABox-swap 전이)·[[11]] τ² 미학습·[[13]] 학습먼저·[[05]] 도메인-일반·[[03]]
> 무료관문 먼저·over-claim 금지.

---

## 0. 한 줄
학습-wing = **도메인-일반 reasoning 학습**(scaffold-제시 정보를 쓰는 다단계 추론). 두 축: **목적**(SFT-imitation
/ RLVR-verifiable-reward)·**형식**(direct / explicit-CoT). **★RLVR의 필요조건 = reachability**(정답 행동이
sampling으로 도달가능해야 reward 신호 존재). **선행 측정이 버킷별 reachability를 이미 말한다** → RLVR-CoT는
**부분 레버**(symbolic 유망·⋈ 의문), 나머지는 genuine 경계(map+fleet).

---

## 0.5 ★★2차 리뷰 정정 (R-1 온도·R-2 게이트 복원 — 학습-wing 부담 축소)
- **R-1(reachability 증거 정정)**: ⋈-unreachable 근거였던 "self-consistency 8/8 동일오답"은 **temp 0.8**서
  잰 것(vacuous 아님·확인)·**단 max-of-N(symbolic) parallel-voting 증거**이지 ⋈이 아님. CoT-reachability
  판별자 = **voting이 아니라 CoT delta**(symbolic +17/35 reachable vs semantic +4 약함). ⇒ **G2 = best-of-N-
  with-CoT(temp>0·다양 추론체인)**·직접-sampling G2는 vacuous. **G2를 four-bench(학습신호)+τ²(천장) 둘 다**(R-3).
- **★R-2(게이트 복원·rev2 과잉정정 교정)**: SEMANTIC_ERROR_FORENSIC per-case(t41/t107)가 이미 확정 —
  **"⋈ wrong-order" 라벨 대부분이 실제 state-tracking(A)**("모든 주문" 2개 인지하고 1개만·=coverage). 버킷
  추정 **A(상태추적)~12·B(feasibility)~10·C(symbolic)~5·genuine-scale(F)~2-4/arm(소수)**. ⇒ **지배 잔여 =
  결정론 게이트/컨트롤러**. rev2가 rev1의 **steering-inject R-레버(dead·R1)**와 **결정론 게이트 G-레버**를
  함께 버렸으나 **게이트는 R1(steering=0)에 안 걸림**(inject-copy 아니라 compliance-class 강제). **복원**:
  - **coverage-controller**(양화사+A2열거+미커버시 stop-deny·regen-gate): 결정론·replay-safe·[[05]] A2-구동·
    compliance 게이트와 같은 class. 모델이 2주문 인지하고 멈춤→완주 강제(행동 대행 아님).
  - **feasibility-gate**(불가능/should-not write 차단·precondition): 결정론.

**⇒ 정정된 전체 그림**:
| 잔여 | 닫는 주체 | class |
|---|---|---|
| coverage/상태추적(A·지배) + feasibility(B) | **결정론 게이트**(복원) | scaffold-gate(done-class·[[13]] 먼저) |
| ⋈ 참조-매칭(묘사→엔티티·71/98 일부) | present+confirm(게이트) + **매칭부분=learn** | 학습-wing(소수) |
| criterion-NL 매핑(bigger→capacity·CoT +17/35) | **learn**(reachable) | 학습-wing |
| ~~steering-inject(변형/총액 값 주입)~~ | **죽음** | R1 |
| genuine-scale(F ~2-4/arm) | 경계 | map+fleet |
- **학습-wing 부담 = 작은 genuine 잔여(⋈-매칭 + criterion-NL)**·reachability-gated. 지배는 결정론 게이트. §5의
  A/A'(SFT/RLVR-CoT)는 이 **작은 학습-잔여**에만 적용.
- **R-4(CoT 포맷·비용)**: A'(RLVR-CoT)는 CoT가 **native tool_call 포맷을 깨면 안 됨**(§23E op-IR 텍스트→파서붕괴·
  pass 0.075 전례). CoT=assistant content·tool_call 별도 FC 프로토콜/replay 보존 **검증 필수** + CoT 토큰비용
  cost-knee 계상(B의 T×와 같은 정직).
- **R-5(필요조건)**: G2 pass@N>0(reachable)은 **RL 신호 존재**일 뿐 **학습·전이 성공 아님**(credit-assignment·
  reward-hack·전이갭 잔존). 전이 make-or-break가 충분조건 판정.

---

## 1. reasoning-wing이 잔여에 맞는 근거·안 맞는 곳 (냉정)
잔여 중 **학습이 닫을 부분(⋈-매칭·criterion-NL)**은 "base가 present된 정보를 못 쓰는" 다단계 추론 실패. "더
careful하게 단계적으로 추론"이 원리적으로 겨냥. **그러나 CoT/RL이 통하려면 정답 행동이 모델 support 안에 있어야
한다.** (지배 잔여 coverage/feasibility는 §0.5대로 결정론 게이트가 닫음·학습 불요.)

## 2. ★핵심 필요조건 — reachability (RL이 작동하는 전제)
- RLVR(GRPO/PPO)은 **롤아웃을 sampling → 정답이면 보상 → 그 궤적을 upweight**. **정답이 한 번도 sampling되지
  않으면 positive reward=0 → 학습 신호 없음 → RL 무의미.**
- 즉 **"RL로 못 여는 것"의 진단 = best-of-N(자기일관성)에서 정답이 나오는가.** 나오면 RL이 증폭 가능·안 나오면
  RL 불가(scale/다른 접근 필요).

## 3. 버킷별 reachability — 선행 측정 (SEMANTIC_ERROR_FORENSIC·RELWORK_LOAD_COT)
| 잔여 버킷(clean nt=4) | 선행 증거 | reachable? | RLVR-CoT 전망 |
|---|---|---|---|
| **변형-select/criterion·calc** (symbolic·최대 21/26) | max-of-N·joint-constraint서 **CoT +17/+35%**·solver>CoT | **✅ reachable** | **유망**(단계추론이 실익) |
| **coverage** (17/16) | completion-miss(모델이 2주문 *인지*하고 1개만)=wrong-belief 아님·부분 도달 추정 | ◐ 부분 | **maybe**(측정 필요·G2) |
| **⋈ wrong-order** (7/10) | intent/⋈-matching서 **CoT +4%·self-consistency +0%(8/8 동일오답=systematic·high-conf)** | **✗ 도달 불가** | **의문**(RL 신호 없음) |
| order-total | 순수 집계·CoT/solver reachable | ✅ | 유망(또는 scaffold-calc) |

**★결론(냉정)**: RLVR-CoT는 **symbolic 잔여(변형·criterion·calc=최대 버킷)에 잘 맞고**, **⋈은 systematic-오답이라
RL이 신호를 못 얻어 의문**, coverage는 미확정(측정). = **부분 레버·universal 아님.**

## 4. 무료 feasibility 관문 (학습 前·무엇이 도달가능한지 측정·[[09]])
| # | 관문 | 방법 | 판독 | 비용 |
|---|---|---|---|---|
| **G1** | **prompted-CoT 버킷별** | 현 모델에 "step-by-step 신중 추론" 프롬프트 → 버킷별 pass 변화 | symbolic↑ 재현? coverage/⋈↑? | 무료(프롬프트만·유료 user-sim 소량) |
| **G2** | **best-of-N reachability 버킷별** | 각 잔여 task를 N=8 sampling → 정답이 *한 번이라도* 나오나(pass@N) | pass@N≫pass^1 → RL-reachable·pass@N≈0 → RL 불가 | 저(생성만·검증 무료) |

**G2가 RLVR의 GO/NO-GO** (후보 B의 A<1과 같은 위상): pass@N≈0인 버킷은 **RL로 못 엶**(scale/경계). pass@N 높은
버킷만 RLVR 투자 가치.

## 5. mechanism — A(primary) + RLVR(조건부 hedge) [2차 리뷰 반영]
| 후보 | 위상 | 목적/형식 | 근거 |
|---|---|---|---|
| **A: four-bench LoRA-SFT** | **primary** | imitation·direct(또는 CoT-trace) | four-bench=**gold 궤적 有 → SFT가 표본효율**·base-freeze·가역·무망각(LoRA)·§19/§22/§28 델타 |
| **A': RLVR-CoT** | **조건부 hedge** | verifiable-reward·explicit reasoning | **A가 §17 역전이할 때만** 정당(RL=generalize>SFT-memorize)·reachable(G2) 버킷 한정 |
- **★2차 리뷰 수렴**: 후보 B(steering-loop) 폐기 후, "test-time compute" 아이디어는 (i) latent-loop=autonomous
  입력소거·선형=단일adapter → **A로 붕괴**(RECURRENT doc ★1/★2), (ii) 토큰-공간 CoT=attention 재읽기(★1 무관)
  지만 **궤적 있으면 SFT가 도구**(§4 함정3). ⇒ **learned-wing = four-bench LoRA-SFT → τ² 전이가 primary이자
  거의 유일 경로**. LoRA-SFT가 B의 이점(base-freeze·가역·무망각)을 $T\times$ 비용·형식파탄 없이 다 가짐.
- **RLVR은 hedge**: SFT-on-traces가 §17처럼 역전이하면 → reachable 버킷서 RLVR-CoT(RL이 표면매핑 회피 가능성).
  궤적 있으므로 **SFT 먼저·RLVR은 SFT 실패 시**. 둘 다 도메인-일반·τ² 미학습·전이 make-or-break로 판정.
- unreachable 버킷(⋈ 8/8 systematic·G2 pass@N≈0)은 **SFT도 RL도 못 닫음** → 경계(§7).

## 6. make-or-break·판정 분해 (R-new-3: 미설치 vs 미전이)
1. **held-out four-bench 전이**: 학습 도메인 외 four-bench 도메인서 잔여 닫히나(스킬 설치+in-substrate 전이).
2. **τ² cross-bench 전이**: τ²로 닫히나. ①✅②✗ = 스킬 실재하나 표현/포맷 갭(§17). ①✗ = 학습 자체 실패.
- 지표: 공식 compute_metrics pass^1..4 same-k·**버킷별 포렌식**([[08]])·reachable 버킷 pass↑.

## 7. 정직한 경계 (contingency·rev2 §4)
- **scaffold(steering=0·present 미사용)도, 학습(SFT 역전이·RLVR 미도달)도 못 닫는 버킷** = **genuine residual**
  (특히 ⋈ systematic·8/8 동일오답). → **닫기 포기·특성화**: 능력별 (scaffold-decidable / reasoning-reachable /
  scale-bound) 지도 + fleet(escalate). moat=compliance+cost-knee는 pass-parity 없이 성립.
- **즉 이 재설계의 정직한 예측**: RLVR-CoT가 **symbolic 잔여(최대 버킷)는 닫을 수 있으나 ⋈은 못 닫을 가능성 큼**
  → 부분 성공 + 경계 census가 결과물. "모든 잔여를 학습으로 닫는다"는 단정 금지.

## 8. 실행 순서 (무료 먼저)
1. **G1 prompted-CoT**(버킷별·거의 무료): symbolic +17/35 재현?·coverage/⋈ 반응?
2. **G2 best-of-N**(저): 버킷별 pass@8. **RLVR GO/NO-GO**.
3. G2 reachable 버킷 → **A' RLVR-CoT**(four-bench 학습·유료·승인)·A(SFT)와 병렬.
4. **전이 make-or-break**(§6)·버킷별 판정.
5. unreachable 버킷 → 경계-지도+fleet(rev2 §4).

## 9. B 철회 요약 (재발 방지·[[03]])
steering=residual 재가중(작음·H2)·rotation=출력불변 양자화 basis·loop=활성 relax(수축=무능)·RLVR↔latent
미정합. **용어 aesthetic("세 스레드 통일")이 기전을 대체했다** = 표류 표본. 교훈: **모든 mechanism은 "이게 잔여를
닫는 *계산*을 실제로 더하는가"로 검문**(present-만·relax-만·basis변경-만은 계산 아님).
