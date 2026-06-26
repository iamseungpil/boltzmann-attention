# A2-규칙사용 SFT 준비 (2026-06-25) — priority-4 make-or-break 조작화

> 정본 학습타깃 = `EPISTEMIC_A2_THESIS_2026_06_23.md` §line49 (A2-규칙-사용 학습). 본 doc = 그 정의를 *기존 인프라에 구체 매핑*(재사용 vs 신규)·NO-GO·열린결정. 설계먼저([[03]])·재유도 금지([[20]][[40]]). **A2_FRONTEND(NL→A2 생성)=별도 논문·제외**([[06]] 2026-06-25 범위분리) — 본 SFT는 **A2 수작성·고정 전제**, 모델이 *고정 A2를 faithful하게 *사용*하기*를 학습.

## 0. 무엇을 학습하나 (★잔여 단정 금지·조건부·2026-06-25 리뷰 정정)
- **타깃의 *종류*(settled)**: 학습 잔여가 *있다면* 그것은 **faithful-formalize형**(NL→관계 predicate·operator·operand 선택의 충실도)이지 **abstain-SFT 아님**(escape=0/15·thesis §3). 이건 종류 확정.
- **★타깃의 *존재·크기·내용*(미지·측정 게이트)**: **학습 잔여 ≡ "결정론 present+gate가 닫고 *남는* 측정된 잔여"**. 이게 무엇인지·존재하는지는 **priority-2(present-nested) 회수 *전*엔 미지**. arc의 직접 증거가 단정을 *반증*하는 쪽임:
  - Probe-B 7/7 — 후보 제시되면 formalize·⋈ *작동* → "formalize 못 함"은 잔여 아님(스킬 present).
  - present+g15 포렌식 — L0(26→16)·over·order를 *결정론으로* 닫는 중([[catalog]] §3.5 L0/OVER=결정론영역).
  - 재설계 트리거 = "arm not learn"으로 점화([[06]]) — 잔여=orchestration-under-load, 결정론 scaffold가 메움.
  ⇒ **잔여를 operand L2/L3·operator L0·task58로 미리 단정하면 트리거·[[13]]·[[05]]와 자기모순**(결정론 점화 직후 learn-first 회귀). 진짜 learn 잔여 후보 = *scaffold가 σ로 미리 못 만드는 criterion을 모델이 formalize해야 하는* 케이스(σ-enumerable이면 nested-present가 결정론으로 닫음). **그 케이스의 실재 = priority-2 + SFT-vs-base 비교로만 확정.**
- **★NO-GO 정직**: 결정론(present/nested+gate)이 잔여를 다 닫거나 잔여가 capability-bound면 → **SFT 불요·§5 NO-GO 직행**. 그건 논문 약화 아니라 헤드라인=결정론게이트+TCO 강화([[06]] 정합·학습=잔여보조).
- **궤적 SFT 형태**: 벤치서 "유한 관계규칙 제시(σ/candidate-set) → formalize(intent→predicate) → check → select; 빈/모호면 ASK" 를 학습. **도메인-일반·내용X**(P-primitive·관계규칙만)·**abstain 케이스 + 결정가능→행동 대칭케이스 둘 다 포함**(over-ask 방지).
- **전이**: 학습은 SOP/TB/Synth서만 → tau2는 **A2-swap·재학습0**([[11]]). e2e = 학습된 TBox 모델([[01]]).

## 1. 재사용 vs 신규 (인프라 감사·survey 기반)
### 재사용 (있음·재건 금지)
| 자산 | 용도 | 비고 |
|---|---|---|
| `lora_train_chat_toolcall.py` | multi-turn tool-use LoRA-SFT (Qwen2.5·assistant-only mask) | 학습 엔진. seq=10240·R16. |
| `build_abstract_sft.py`·`build_tbox_sft.py` | 도메인-일반 "Plan:" 추상화·TBox-isolation(concrete tool_calls=-100 mask) | **A2-σ-use 궤적의 골격** — 단 A2 σ-presentation 주입 필요(§2). |
| `sopbench/build_tbox_planner_sft.py` | gold 결정시퀀스(READY/BLOCKED+operator/STOP)·cross-domain LODO | gate-준수 시퀀스 = A2 check/select의 직접 선례. |
| `tb_build_sft.py`·`ma/synth_*.py`·`build_solo_data_cfb.sh` | TaskBench graph·Synth content-op·cfbsynth fetch-first 데이터 | 3벤치 데이터빌더 ([[01]]). |
| `grpo_reward.py`·`procedure_scorecard.py` | 결정론 보상(pass+seq_F1−over+arg)·5축 채점 | LLM-judge 0. SFT 후 GRPO(S2) 옵션. |
| `gate_interpreter.py`·`t2_gate_patch.py`·`a2/{retail,airline}.gate.json` | A2 집행 scaffold(고정·도메인-일반) | tau2 전이 평가 기반. |
| `escape_scope_diag.py`·`escape_layer_decomp.py`·`escape_det_census.py` | ⓐ/ⓑ·층(L0/L1/L2/L3)·결정론 census | **SFT 전후 결정론 측정 도구**(이미 가동·검증됨). |

### ★Builder 감사 (2026-06-25 실코드 대조·D3 잔여-직결)
- **SOP operator/order = `sopbench/build_tbox_planner_sft.py` 빌더 있음**(input=planner prompt+READY/BLOCKED·target=operator/STOP·mask·shuffle·ABox=prompt·LODO). **⚠️ 단 L0 분리 선결(2026-06-25 리뷰 #2)**: L0(operator)는 두 종류 — **eligibility-operator**(modify↔exchange를 status로 결정 = [[catalog]]§3.5 *결정론 게이트*·present+g15가 이미 26→16 닫음) vs **control-flow operator-sequencing**(*학습가능*). **g15 후 *남는* L0 잔여가 어느 쪽인지 확인이 선결** — eligibility-잔여면 결정론게이트 몫(SOP-SFT로 또 닫음=[[05]]/[[13]] 위반). 학습은 sequencing 잔여에만.
- **Synth operand = `ma/m_sigma_data.py` $ref ⚠️ C4/M-σ settled 음성 계열 (2026-06-25 리뷰 #1·★최중요 화해)**: $ref(operand=값날조 아닌 *관계*로 학습)는 **이미 시도·전이 실패한 그것** — [[20]] 확정: **"M-σ in-dist 96%(derivation 학습가능)·M-D 전이 음성(C8 1차)"**·[[06]] C4 확정: cfbsynth SFT(52)/DPO(35) abstract→real **전이 실패·autofetch(결정론)만 작동**. 101/102 "123 Elm St" 날조 = copy-prior fabrication = **C4가 다룬 바로 그 클래스**. → **operand $ref-SFT는 settled 음성 재유도 위험**([[20]][[40]]). **operand의 *검증된* 레버 = 결정론 present/autofetch**(present-nested arm이 지금 측정 중). **블로커: "왜 이번 $ref가 cfbsynth/M-σ 전이실패한 곳서 성공하나"를 명시 못 하면 → operand-learn 보류·결정론 present로.**
- 함의(정정): 빌더는 있으나 **두 절반 다 "학습 vs 결정론" 경계 화해가 선결** — operand=C4/M-σ 전이음성 계열(보류 위험)·operator=eligibility 부분 게이트-redundant. 데이터빌더 재건은 불요지만 *타깃 정당성*이 미확정. 신규작업(format정렬·abstain·평가셋)은 화해 통과분에만.

### 신규 (gap·만들어야 함)
1. **A2-σ-use 궤적 빌더** — 기존 build_abstract_sft를 확장: 각 결정점에 *A2가 제시한 candidate-set/gate-verdict*를 컨텍스트로, 타깃=formalize 선택(어느 predicate/entity/operator/operand). present/nested(이미 구현)와 *동형*인 입력형. ⚠️ M_A 교훈: concrete-emit SFT=over-calling 아티팩트(2× gold) → **abstract+mask·diversity 필수**([[12]]).
2. **abstain+대칭 커리큘럼 데이터** — empty-σ→ASK·σ>1→ASK·σ=1→act 케이스 *명시 합성*(벤치 변형). 균형(act:ask)·over-ask 비용 측정용.
3. **held-out formalize 평가셋** — prospective (NL, predicate-set, gold-target-spec) triplet. 시드 = 15 gap + present/nested 잔여(L2/L3·task58류). ⚠️ **15 gap은 thin·편향**(gpt4.1-pass∧32B-fail = capability-gap set·모호 배제·대부분 orchestration이라 formalize 케이스 얇음). → **확장 필수: present/nested 잔여 + S4(retail 전체) 케이스 추가.** train=SOP/TB/Synth·eval=tau2-swap이라 held-out 자체는 OK(전이측정). SFT 전후 formalize-EM/gate-F1 기준선.

## 2. 데이터형 스펙 (A2-σ-use 궤적·핵심 신규)
- 입력(매 결정점): `[NL 요청] + [도구 스키마] + [A2-σ: 이 결정점의 candidate-set/gate-verdict]` — present/nested가 런타임에 주입하는 것과 *동일* 형식(학습=추론 일치).
- 타깃(assistant): formalize 결정 = (a)applicable predicate/operator (b)entity/operand 선택(σ에서 copy·날조0) (c)σ=0/>1이면 ASK. **concrete tool-name/arg는 mask 가능**(도메인-일반 학습=어느 *관계*인지·구체는 A2-swap).
- 도메인-일반 보장([[12]] diversity): 표현(verb/phrase)·구조(chain/DAG/branch) 다양성 sweep(ma/synth_depth 축 재사용). 단일템플릿 금지.
- abstain/대칭: 같은 골격에 σ-cardinality만 바꿔(0/1/>1) act vs ask 라벨 대칭 생성.

## 3. 학습→전이 파이프라인 (thesis line49)
```
S0  벤치(SOP/TB/Synth) A2-σ-use 궤적 SFT (도메인-일반·abstain+대칭·diversity)
      base = Qwen2.5-14B (D1: 14B 단독 먼저=빠른 GO/NO-GO) → NO-GO 경계 확인용 7B 추가(스케일-floor)
S1  (옵션) GRPO RFT (grpo_reward·결정론) — SFT가 *잔여*를 못 닫으면만
전이 {base vs A2-trained} × scaffold(A2 집행) on tau2  ★A2-swap·tau2 학습0
```

## 4. 검정 (결정론 지표·[[08]]·pass^1 금지)
### 4.0 ★분리 (2026-06-25 리뷰 정정): 결정론-arm 몫 ≠ 학습-arm 성공기준
- **L0(operator)·OVER(over-action)·order = 결정론-arm 몫** ([[catalog]] §3.5 결정론영역·present+g15가 이미 부분 닫음 L0 26→16). 이걸 *학습*으로 닫는지 테스트하면 = 더 싼 레버 두고 학습 사용([[05]]/[[13]] 위반). → **학습 성공기준에서 제외.** present/nested+gate 측정(escape_det_census)이 담당.
- **학습-arm 성공기준 = present/nested로도 *안* 닫히는 잔여에만**(존재 시).
### 4.1 학습-arm 검정 (present-불가 잔여 한정)
1. **결정론-불가 operand-formalize 잔여 Δ**: priority-2(present-nested)가 닫고 *남은* operand 케이스(σ-enumerable 아닌 criterion-formalize)가 {base→A2-trained}서 줄어드나. **잔여가 비면 이 검정 무의미→NO-GO.**
2. **새 도메인 A2-swap 전이**: airline(또는 bank) A2-swap서 retail-학습 효과 전이(재학습0). = 학습이 *도메인-일반 스킬*인지(표면매핑 아닌지·[[12]]).
3. **대칭 비용**: over-ask/false-defer rate(결정가능한데 ASK) — 학습이 act↔ask 대칭인가(한쪽 과적합 아님).
### 4.2 결정론-arm 검정 (참고·이미 측정중·학습 무관)
- G5-was-0·L0·OVER·order census = present/g15/nested arm 결과(`PRESENT_G15_DET_CENSUS`·priority-2 회수). 학습 성공기준 아님.

## 5. NO-GO (정직·thesis line51·우선 게이트)
- **(0·선결) 잔여 부재**: present/nested+gate가 잔여를 다 닫으면 → 학습 잔여 없음 → **SFT 불요**(결정론+TCO 헤드라인 강화·[[06]]). priority-2 회수가 1차 판정.
- **(0'·★선결 화해·2026-06-25 리뷰) settled 음성 재유도 차단**: 이 arc는 학습으로 **3회 실패·결정론으로 성공** 패턴 — C4(cfbsynth SFT/DPO copy)·M-σ(derivation 전이음성·[[20]])·G5(eligibility-steer≈0). 따라서 SFT 착수 전 잔여가 **(a)실재**(present-nested 후 남음)·**(b)C4/M-σ 계열 아님**(operand-copy/derivation이면 전이음성 재유도)·**(c)gate-redundant 아님**(eligibility-operator이면 결정론 몫)·**(d)Probe-B처럼 격리하면 됨**(capability-bound 아님)을 *모두* 통과해야. 못 통과하는 잔여는 결정론(present/autofetch/gate) 몫 → SFT 제외.
- **(a) 표현불가**: 남은 잔여가 유한관계로 깔끔 표현 안 되면.
- **(b) capability-bound**: "formalize"가 그 크기서 학습으로 *안* 줄면(특히 mis-formalize=ⓑ) → 진짜 경계 = escalate/scale([[13]]). 결론 = "scaffold offload 천장 + 학습 한계 지도".

## 6. 결정 (D1·D3 확정 2026-06-25·D2 게이트)
- **D1 (확정)**: **{7B,14B} 멀티스케일**·단 *단계적* — **14B 단독 SFT-only 먼저**(빠른 GO/NO-GO·M_A: 7B reasoning-bound) → 7B는 NO-GO 경계/스케일-floor 확인용 추가. GRPO(S1)=SFT가 잔여 못 닫을 때만.
- **D3 (확정·단 화해 게이트)**: 잔여-직결 우선 — 단 §5(0') 통과분에만. **operand(Synth $ref) = C4/M-σ 전이음성 계열 → 화해 못 하면 *보류*(결정론 present로)**. **operator(SOP) = eligibility-잔여 분리 후 sequencing분만**. 즉 D3는 "벤치 선택"은 확정이나 *어느 잔여를 학습 타깃으로 삼나*는 §5(0') 통과가 선결. 3벤치 균형은 1차 GO 후.
- **D2 (확정·게이트)**: **GPU-free 준비물 = 지금 착수 OK**(평가셋·궤적빌더 프로토·커리큘럼 스펙·§7). **단 SFT *실행* 결정 = priority-2(present-nested) 회수 게이트** — "결정론이 *안* 닫는 operand/criterion-formalize 잔여가 실재하고 학습 여지(Probe-B처럼 격리하면 됨) 있음"이 확인될 때만. 잔여 비거나 capability-bound면 §5 NO-GO 직행.

## 7. 지금 GPU-free 착수 가능 (D2=지금이면)
- held-out formalize 평가셋 시드(15 gap + present/nested 잔여 → (NL,predicate,gold) triplet).
- A2-σ-use 궤적 빌더 프로토타입(build_abstract_sft 확장·소량 dry-build로 형식 검증).
- abstain+대칭 커리큘럼 합성기 스펙.
(학습 실행은 GPU 필요 → priority-2 런 완료 후.)
