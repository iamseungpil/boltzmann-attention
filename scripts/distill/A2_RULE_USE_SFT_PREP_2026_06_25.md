# A2-규칙사용 SFT 준비 (2026-06-25) — priority-4 make-or-break 조작화

> 정본 학습타깃 = `EPISTEMIC_A2_THESIS_2026_06_23.md` §line49 (A2-규칙-사용 학습). 본 doc = 그 정의를 *기존 인프라에 구체 매핑*(재사용 vs 신규)·NO-GO·열린결정. 설계먼저([[03]])·재유도 금지([[20]][[40]]). **A2_FRONTEND(NL→A2 생성)=별도 논문·제외**([[06]] 2026-06-25 범위분리) — 본 SFT는 **A2 수작성·고정 전제**, 모델이 *고정 A2를 faithful하게 *사용*하기*를 학습.

## 0. 무엇을 학습하나 (타깃 고정·재론 금지)
- **타깃 = faithful-formalize** (NL→관계 predicate·operator·operand 선택의 충실도), **abstain-SFT 아님**(escape=0/15 settled·[[06]]). 이번 present+g15 포렌식이 확인한 잔여 원천 = ⓑ mis-formalize(operand L2/L3·operator L0·task58 wrong-variant) = *유일하게 LLM만 할 일(번역)을 불안정하게 함*(thesis §2 translator).
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

### 신규 (gap·만들어야 함)
1. **A2-σ-use 궤적 빌더** — 기존 build_abstract_sft를 확장: 각 결정점에 *A2가 제시한 candidate-set/gate-verdict*를 컨텍스트로, 타깃=formalize 선택(어느 predicate/entity/operator/operand). present/nested(이미 구현)와 *동형*인 입력형. ⚠️ M_A 교훈: concrete-emit SFT=over-calling 아티팩트(2× gold) → **abstract+mask·diversity 필수**([[12]]).
2. **abstain+대칭 커리큘럼 데이터** — empty-σ→ASK·σ>1→ASK·σ=1→act 케이스 *명시 합성*(벤치 변형). 균형(act:ask)·over-ask 비용 측정용.
3. **held-out formalize 평가셋** — prospective (NL, predicate-set, gold-target-spec) triplet (~50건). 현 자산서 즉시 시드 가능: **이미 분석된 15 gap task + present/nested 잔여(L2/L3·task58류)**가 gold 라벨된 formalize 케이스. SFT 전후 formalize-EM/gate-F1 측정 기준선.

## 2. 데이터형 스펙 (A2-σ-use 궤적·핵심 신규)
- 입력(매 결정점): `[NL 요청] + [도구 스키마] + [A2-σ: 이 결정점의 candidate-set/gate-verdict]` — present/nested가 런타임에 주입하는 것과 *동일* 형식(학습=추론 일치).
- 타깃(assistant): formalize 결정 = (a)applicable predicate/operator (b)entity/operand 선택(σ에서 copy·날조0) (c)σ=0/>1이면 ASK. **concrete tool-name/arg는 mask 가능**(도메인-일반 학습=어느 *관계*인지·구체는 A2-swap).
- 도메인-일반 보장([[12]] diversity): 표현(verb/phrase)·구조(chain/DAG/branch) 다양성 sweep(ma/synth_depth 축 재사용). 단일템플릿 금지.
- abstain/대칭: 같은 골격에 σ-cardinality만 바꿔(0/1/>1) act vs ask 라벨 대칭 생성.

## 3. 학습→전이 파이프라인 (thesis line49)
```
S0  벤치(SOP/TB/Synth) A2-σ-use 궤적 SFT (도메인-일반·abstain+대칭·diversity)
      base = Qwen2.5-{7B floor, 14B 주력}  (M_A: 7B reasoning-bound·14B=주력 후보)
S1  (옵션) GRPO RFT (grpo_reward·결정론) — SFT가 formalize 못 닫으면
전이 {base vs A2-trained} × scaffold(A2 집행) on tau2  ★A2-swap·tau2 학습0
```

## 4. 검정 (결정론 지표·[[08]]·pass^1 금지)
1. **G5-was-0 pass전환**: present+g15서 operator/over-action 잔여(L0/OVER)가 *학습 후* 결정론 census(escape_det_census)서 닫히나.
2. **operand grounding(L2/L3)**: task58류 wrong-variant가 학습으로 줄어드나 (present-nested 측정과 직접 비교).
3. **새 도메인 A2-swap 전이**: airline(또는 bank) A2-swap서 retail-학습 효과 전이(재학습0).
4. **대칭 비용**: over-ask/false-defer rate(결정가능한데 ASK) — 학습이 act↔ask를 대칭으로 했나(한쪽 과적합 아님).

## 5. NO-GO (정직·thesis line51)
- (a) 잔여(L0/L2/L3)가 유한관계로 깔끔 표현 안 되면, 또는 (b) "formalize"가 그 크기서 **capability-bound**(특히 mis-formalize=ⓑ가 학습으로 *안* 줄면) → 그게 진짜 경계 = escalate/scale([[13]] 흡수 우선순위 scale→learn→scaffold). 이 경우 본 논문 결론 = "scaffold offload 천장 + 학습 한계 지도".

## 6. 열린 결정 (사용자)
- **D1 base 스케일·범위**: (a) 14B 단독 SFT-only 먼저(빠른 GO/NO-GO) vs (b) {7B,14B} 멀티스케일 vs (c) SFT+GRPO 풀스택. → 비용·정보 트레이드.
- **D2 첫 산출**: present-nested(priority-2) 결과 회수 *후* 시작 vs 지금 GPU-free 준비물(궤적빌더·formalize 평가셋·커리큘럼)부터 착수.
- **D3 궤적 데이터 출처**: 3벤치 전부 vs 잔여(operand/operator)에 가장 직결된 벤치 우선(Synth content-op = operand·SOP = operator/order).

## 7. 지금 GPU-free 착수 가능 (D2=지금이면)
- held-out formalize 평가셋 시드(15 gap + present/nested 잔여 → (NL,predicate,gold) triplet).
- A2-σ-use 궤적 빌더 프로토타입(build_abstract_sft 확장·소량 dry-build로 형식 검증).
- abstain+대칭 커리큘럼 합성기 스펙.
(학습 실행은 GPU 필요 → priority-2 런 완료 후.)
