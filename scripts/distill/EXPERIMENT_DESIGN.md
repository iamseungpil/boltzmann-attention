# ★ EXPERIMENT DESIGN — MASTER (단일 권위본, 이것부터 읽을 것)

> 2026-06-01 신설. **설계서 난립 방지용 단일 진입점.** 목표·현재 실험 순서·헤드라인 지표를 여기서 고정한다.
> 세부는 §7 문서지도의 detail 문서로. **방향이 흔들리면 이 문서 §0~§3만 다시 읽는다.**

---

## §0. 목표 (한 문장, 변하지 않음)
**자연어 멀티턴 요청을, 도메인별 구조화 온톨로지(ABox)로 재해석해 내부적으로 절차(dirgraph)를 추론·실행하는 agentic planner를, 작은 모델 weight(TBox)에 학습시키고, 본 적 없는 도메인은 ABox 교체만으로 재학습 0 전이한다.**
- **TBox(weight, 학습·전이)** = "NL 요청 + ABox 어휘 → dirgraph(절차) 도출 + 실행" 스킬. dirgraph는 **모델 출력**(컨닝 아님). L0(결정론)는 NL→dirgraph 불가 → 이 매핑이 비자명·대체불가 기여.
- **ABox(in-context swap, 후속 xattn)** = 도메인 도구 affordance + NL 정책. goal precondition '정답 구조'는 안 떠먹임.
- injection/steering(구 라인)은 agentic서 null로 폐기됐다 재정의(§3 Rung3). FM SO.P/CAP-CPT의 weight-baking과 달리 **TBox/ABox 분리 → 전이**가 핵심 차별.

## §1. 벤치마크 · 평가
- **주 = SOPBench(Zekun Li, 2503.08669)**, 7도메인. **파일럿 = bank(N=134: should_T 48 / should_F 86)**. rule oracle `env/evaluator.py`(LLM judge無).
- **헤드라인 지표**:
  1. **LODO 전이** (6도메인 학습 → held-out bank, ABox swap, **재학습 0**) — 1급 결과.
  2. **should_T = dirgraph+ ∩ goal+ 동시충족** (BOTH). ⚠️**총 Mean Pass Rate는 거부로 부풀려져 헤드라인 금지**(s1 0.605=거부 degenerate 함정).
  3. **should_F gross** gain/loss(net 금지).
- **평가 세팅**: 현재 정적-user(`default_response` 덤프, leaderboard 정합). **멀티턴 user_sim**(`--user_model`)은 더 어려운 robustness 축(천장 불변 — user_sim도 user_known만 앎; PartB cred-부재 해소 안 됨).
- **ablation(매 rung)**: 빈/틀린 ABox→붕괴(온톨로지 실사용) · L0 vs L1(in-context) vs L2(학습) · **alias on/off**.

## §2. 현재 진단 (어디까지 왔나)
- **arm-1(LLM-alone) bank 5.2% / arm-3 naive 0% / arm-3v2(in-context 구조, 무학습) gating 무시 / arm-4a(학습 L2) 26.1%(should_T 4/48)**.
- **★run_scripted(결정론 직접제어): 완전게더 A+B+C = 37/48(oracle), 프롬프트-LLM = 4/48.** → **should_T binding = 도구 선택(=게더: 어떤 검증/establish 도구를 부르는가)**. "직접제어 ≫ 프롬프트" 실증.
- **★현 SFT 진단(전수조사)**: **게더는 학습됨**(dirgraph 36-45/48, LODO 전이) **but 터미널 게이트(게더 후 act vs STOP) 못 박음** → **BOTH 0-2**. s1(STATUS)=게더후 STOP / s3(NL)=안게더 act. gate-token(Exp-4d)=부분(act 3→13, BOTH 0→2). **= 병목이 게더에서 터미널 게이트로 이동.**
- **결함 제외 천장**: should_T 정직천장 ~34/48(8 PartA 코드결함 + 6 PartB cred-부재 제외; run_scripted oracle 37).

## §3. ★실험 순서 = 학습 사다리. 핵심 = **게더-act 경합(순서) 해소: "게더 미완이면 ACT 금지"를 *학습*** (가드 아님)

> 병목(전수조사) = 게더는 학습됨(dirgraph 36-45) **but 게더 AND act 공존 못함(BOTH 0-2)**: 게더후 STOP(s1)·안게더 act(s3) = **순서 경합**. bank는 체크 1-3개라 conjunction-*계산*은 안 어려움 → 병목 = **act/STOP 결정·순서**(s1은 READY 보여줘도 over-STOP). 목표 = **"ready(required 전부 게더)면만 ACT"를 weight에 학습**(결정론 가드 offload 아님). 세 레버 = **①SFT readiness-게이트 → ②DPO preference → ③RFT 페널티**. (순서 정당화: process supervision은 통계적 우위 없음[Jia ICML25] → ①② 먼저, ③로 경계.)

### Phase 1 = Rung 1 = ① SFT: per-step readiness 게이트 토큰 〔현재〕
- **목표**: "게더 미완이면 ACT 금지"를 SFT로 학습(순서 경합 해소).
- **방법(deep-research 2026-06-01 확정, 적대검증 20/25)**: 매 스텝 타깃 = **`ready=<true|false>; <행동>`** (educated 중간 게이트):
  - **게더 미완** → `ready=false; <다음 게더 도구>` — ★학습 데이터에 **`ready=false` 뒤 ACT가 한 번도 안 나옴 → "미완→ACT 금지"를 구조적으로 학습**(positive-only SFT로도 순서 강제).
  - **게더 완료** → `ready=true; all_verified=<T/F>; <ACT|STOP>` (all_verified = 게더된 체크 truths의 **AND**; true→ACT/false→STOP).
  - ready 판정 = required 다 게더됐나(prompt observed/history서 셈) = act결정보다 **단순 sub-task** → 명시 supervise면 학습 [educated 중간타깃 Abbe NeurIPS24; parity+CoT 분리 Kim&Suzuki ICML25]. **★educated 필수**(그냥 "생각공간"은 globality barrier 못 깸).
  - **scratchpad(all_verified)는 conjunction-*계산*용**(체크 많은 복잡 도메인 일반화); **bank엔 readiness 게이트가 핵심**(작은 conjunction). 같은 토큰 메커니즘.
- **한계(SFT positive-only)**: "조기 ACT는 *나쁨*" 음성 신호 부재 → ready 오판시 조기 act 잔존 → **② DPO**.
- **근거 논문**: Nye scratchpad(2112.00114)·Teaching Arithmetic(2307.03381)·**Kim&Suzuki parity+CoT 증명(2410.08633)**·**Abbe globality+inductive scratchpad(2406.06467, NeurIPS'24)**·Feng CoT expressivity(2402.12875)·least-to-most(2205.10625).
- **소스코드 재활용 판정**: 학습코드(`aryol/inductive-scratchpad`, `lee-ny/teaching_arithmetic`)는 **from-scratch 합성과제(parity/arithmetic)라 우리 LoRA-SFT엔 직접 재활용 불가**. **재활용 = 데이터포맷 레시피**(inductive carry-update + educated 중간토큰)를 `build_tbox_planner_sft.py` teacher에 적용. 트레이너는 `lora_train_chat_toolcall.py` 유지.
- **⚠️근거 한계(정직)**: 증거 대부분 parity/arithmetic/automata(parity=XOR≠AND, 증명은 1-layer) → 7B tool-use 전이는 **analogical**. 과대주장 5 sub-claim은 검증서 killed(SRL 등). 일반 원리(educated 중간토큰 supervision)는 강함, gate 분해는 동기부여된 instantiation.
- **현황**: Exp-4d(gate-token)=부분(act 3→13·BOTH 0→2). Exp-4e=절차분해(inductive scratchpad). **목표 BOTH 0-2 → 다수.**

### Phase 1.5 = Rung 1.5 = ② DPO: 순서 preference (SFT 음성-부재 보강)
- **GT 쌍**(reward model 불요): **chosen** = 게더→올바른 분기(완전 궤적) / **rejected** = (a) **조기 act**(게더 미완 상태서 ACT/goal 호출), (b) should_T인데 게더→STOP(과잉거부), (c) should_F인데 act(과소거부).
- **DPO loss** → 조기 act·과/소거부를 **명시 dispreferred**. init = Rung1 SFT, KL to SFT(ref=frozen). = SFT positive-only가 못 주는 **음성 신호**를 GT-유도 대조로 직접 주입.
- **✅구현·단위검증**: `build_dpo_pairs.py`(readiness-gate SFT jsonl → {prompt,chosen,rejected}; rejected=조기ACT/과·소거부) + `dpo_train.py`(수동 DPO, policy=SFT-init LoRA·ref=frozen SFT·assistant-token logp DPO loss, trl-free). Exp-4e-dpo.

### Phase 2 = Rung 2 = ③ RFT (GRPO): 조기 ACT 패널티 + BOTH 보상
- **init = Rung1/1.5 어댑터**. 수동 GRPO 루프(`grpo_reward.py` 검증, trl 회피, KL to SFT).
- **reward = SOPBench rule evaluator**(결정론·judge無·무료): **+w_pass·BOTH(dirgraph∧goal 성공)** + w_proc·dirgraph진행(dense, sparse cold-start 구제) **− w_early·(조기 ACT 호출수)**(=게더 미완 상태 ACT/goal 직접 페널티), should_F면 **+올바른 STOP**(dual-axis gross).
- 조기 act 롤아웃→음성 advantage→억제(순서 위반 직접 페널티). outcome-only도 조기act=dirgraph실패→0(암묵 페널티); 명시 process 패널티는 dense·빠름.
- **✅구현**: `sopbench_reward.py`(reward=BOTH+dirgraph진행−early-ACT, dual-axis, GT-grounded·judge無; 자체테스트: gather-act 1.3>refuse 0.2>premature −0.4) + `grpo_train_sopbench.py`(수동 GRPO: group-norm advantage + −adv·logp + KL **update core ✅**; rollout assembly) + `_plan_v2` `SOPBENCH_RLLOG` 훅(planner step 로깅)·temp 샘플링. ⚠️**rollout-serving 오케스트레이션은 Rung2 진입시 원격검증**(task 격리). Exp-4f.
- **SFT(Rung1) vs RFT 역할(연구 확정)**: process supervision은 outcome 대비 **통계적 우위 없음**(알고리즘적 credit-assignment만) [Jia ICML'25, 2502.10581] → **Rung1 SFT 중간토큰 supervision이 primary**(rare-gate도 dense 신호); RFT는 보조(outcome-only RL은 게이트가 드물게 샘플돼 신호0일 때 실패 → process/SRL식 dense reward로 보완). 즉 **게이트는 SFT 분해로 먼저, RFT로 경계 다듬기**.

### Phase 3 = Rung 3 — 내재화: xattn ABox-memory + steering (직접 주입)
- 원 ★novelty(B5*). ABox를 프롬프트(토큰) 아닌 **xattn 메모리로 직접 주입**. TBox=xattn weights/ABox=메모리 M 스왑(토큰0·전이).
- **근거**: MetaTool basis-specificity(직접개입 specific, random/프롬프트 재현불가) + run_scripted(37≫4). **진입조건 = Rung1-2가 in-context 천장 근접 후**. Exp-4g.

> **공정성/anti-cheat (전 rung 적용)**: **도구명 ALIAS 마스킹**(그래프 전체 일관) — 이름 암기 차단·NL↔설명 의미매칭 강제 = LODO 전이 타당성 게이트. **source3**(STATUS 정답지 미렌더, NL 정책만). 진짜 anti-cheat = alias ON + source3.

### §3.9 선행 연구 — 단계별 학습 사다리 (staged SFT→preference→RL) · 커리큘럼 〔2026-06-01 추가, 전부 arXiv 검증〕
> §3 기존 인용(Nye·Abbe·Kim&Suzuki·Feng·least-to-most)은 "readiness-게이트 sub-task가 *학습 가능한가*"(analogical, parity/arithmetic 증명)를 다룬다. **아래는 사다리 *구조 자체*(왜 SFT → preference → RL 순서로 쌓나)의 직접 LLM-post-training 선례** — analogical 아님.
- **SFT→선호→RL 정준 파이프라인(우리 ①②③의 직계 선례)**:
  - **InstructGPT** [Ouyang et al. 2022, `2203.02155`, NeurIPS'35]: SFT → reward model → PPO RL의 3단계 사후학습 원형. 우리 사다리의 골격.
  - **DPO** [Rafailov et al. 2023, `2305.18290`]: reward model·in-loop RL 없이 선호를 policy/ref log-ratio로 재매개화한 BCE 손실. = ② `dpo_train.py`가 구현한 방법의 원전.
  - **★Tülu 3** [Lambert et al. 2024, `2411.15124`]: open recipe **SFT → DPO → RLVR(RL with Verifiable Rewards)**. 우리 사다리 **①SFT → ②DPO → ③RFT(SOPBench rule reward)와 거의 동형** — RLVR의 "검증가능 보상"이 우리의 결정론 rule evaluator와 정확히 대응. 가장 직접적인 선례.
  - **★DeepSeek-R1** [DeepSeek-AI 2025, `2501.12948`]: 순수 RL(R1-Zero)은 가독성·언어혼합 붕괴 → **cold-start SFT + multi-stage 후 RL**로 해소. **"RL 전에 SFT 발판이 필요"**의 강한 증거 = 우리 "①② 먼저, ③로 경계 다듬기"를 직접 지지.
- **process vs outcome 감독(③ reward 설계 근거, 양면 제시·정직)**:
  - **Lightman et al. "Let's Verify Step by Step"** [2023, `2305.20050`, PRM800K]: MATH에서 **process 감독 > outcome 감독**(중간 step별 보상). 우리 dense dirgraph-progress·early-ACT 패널티(③)의 경험적 지지.
  - ⚠️**대척점(이미 §3에 인용)**: Jia ICML'25 [`2502.10581`]: process는 outcome 대비 **통계적 우위 없음**(알고리즘적 credit-assignment 이점만) → **①② SFT/DPO를 primary로, ③ RFT는 보조**(드문 게이트 신호0 구제)라는 우리 순서의 정당화. 두 결과를 함께 둠 = process가 *언제* 도움되는지(hard 추론·dense step 보상)는 경험적, *통계적 일반우위*는 아님.
- **커리큘럼 (easy→hard, 게이트 sub-task 우선)**:
  - **Bengio et al. "Curriculum Learning"** [2009, ICML]: 쉬운→복잡 순 제시 = continuation method(매끄러운 목표부터). **readiness 판정(required 다 게더?)은 act-결정보다 단순 sub-task** → 먼저 명시 supervise(①)하고 어려운 순서-결정을 나중(②③)에 다듬는 우리 분해의 근거.
- **반복 자기개선 사다리 (③ RFT가 자기 데이터로 부트스트랩, Rung2 옵션)**:
  - **STaR** [Zelikman et al. 2022, `2203.14465`]: 정답 rationale만 모아 FT→반복. **ReST-EM "Beyond Human Data"** [Singh et al. 2023, `2312.06585`; 원 ReST `2308.08998`]: generate→**검증가능 피드백으로 필터**→FT→반복(EM). = ③ GRPO 롤아웃을 rule evaluator로 필터해 재학습하는 우리 루프의 선례(reward hacking 없는 GT-grounded 자기학습).
- **정직 단서**: 이 선례들은 대형 모델(수십B~)·math/instruction 도메인 결과 → 7B LoRA + SOPBench tool-use 전이는 **레시피 차용**(단계 순서·검증가능 보상)이지 성능 보장 아님. 헤드라인은 우리 LODO BOTH 실측으로만 주장.

## §4. 성공 게이트 (사전등록)
- **G-SFT(Rung1, ① readiness-게이트)**: BOTH(dirgraph∩goal) 현 0-2 → **≥ 다수**(예 ≥15/48) + **조기 ACT율↓**(ready=false서 ACT 거의 0). 못 넘으면 ② DPO.
- **G-DPO(Rung1.5, ②)**: DPO가 SFT 대비 **조기 act율 추가 감소 + BOTH 상승**. 음성 신호 효과 확인.
- **G-RFT(Rung2, ③)**: RFT가 SFT/DPO BOTH 대비 **유의 상승** + early-ACT 페널티로 순서위반↓ + should_F gross 회귀 ≤기준.
- **G-전이(헤드라인)**: held-out bank LODO가 in-domain의 **≥70%** 회수, **재학습 0**.
- **G-ablation**: 빈/틀린 ABox → **붕괴**(온톨로지 실사용 증명).
- **G-xattn(Rung3 진입)**: Rung1-2 in-context 천장 확인 후, 직접주입이 프롬프트 잔여천장 상회 + ABox-ablation 붕괴.

## §5. 트랙 분업
- **Track A (우리, 7B/14B)**: Rung1-2 구현·파일럿·진단. 현재 bank LODO.
- **Coworker (32B/72B)**: 대형 모델 arm 매트릭스·alias·user_sim (별도 `COWORKER_EXPERIMENT_PLAN.md`). **32B 바닐라는 leaderboard 인용(재측정 금지).**
- **특허**: 별 트랙(중복 도구 SELECTION 내재화; MetaTool/ToolBench). 본 라인=계획·전이.

## §6. 인프라 불변식 (반복 사고 방지)
- 로컬 python=Store 스텁 → **모든 측정 리모트(rr.ps1)**, RC·scanned 확인 후만 "실측" 인용. **fabrication 금지**.
- `rr.ps1 pkill -f`가 자기 셸 self-match→SSH drop → **문자열 split**(`"x""y"`)로 회피. vLLM kill-9 잔여 `/dev/shm/vllm*` → GPU wedge → **shm 정리+GPU별 PID kill**. 2잡/48GB=OOM위험(solo 권장). eval=GPU별 PID-kill 격리.
- 배포=git pull(SFTP 금지). seka_env(py3.12, peft 0.19 있음=학습 가능). 어댑터명 `tbox_v2`=FC통과.

## §7. 문서 지도 (이 문서가 마스터; 나머지는 detail)
| 문서 | 역할 | 상태 |
|---|---|---|
| **이 문서** `EXPERIMENT_DESIGN.md` | **목표·순서·지표 권위본** | ★마스터 |
| `WORKFLOW_ONTOLOGY_DESIGN.md` | TBox/ABox 전체 스펙·planner L0/L1/L2·§9 LLM-in-loop·prior art | detail (개념 원본) |
| `TASK_CONSTRAINT_DESIGN.md` | should_T 병목 진단·게이트·§8.6 전수진단·§8.7 사다리 상세 | detail (Rung1-2 상세) |
| `GRPO_REWARD_DESIGN.md` | RFT reward 함수·GRPO 루프(Rung2 상세) | detail |
| `SOPBENCH_EXPERIMENT_RESULTS.md` | 모든 실측 결과(Exp-1~4) 누적 | 결과 권위본 |
| `COWORKER_EXPERIMENT_PLAN.md` | 32B/72B 분업 | detail (Track B) |
| `TASK_CONSTRAINT_{DESIGN_REVIEW,IMPL_REVIEW}.md` | 리뷰 라운드 | 참조 |
| `EXPERIMENT_DESIGN_v1_7_facet_rft.md`, `steering_paper/*` | 구 facet-rft/steering 라인 | ⚠️superseded(개념 참조만) |

**규칙: 목표·실험 순서 변경은 이 문서 §0~§4에서만. detail 문서는 구현 세부만.**
