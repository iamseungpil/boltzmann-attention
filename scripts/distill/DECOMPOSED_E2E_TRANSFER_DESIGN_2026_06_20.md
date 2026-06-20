# 분해-협업 e2e 전이 설계 (2026-06-20) — 미빌드 step-scaffold로 벤치-TBox를 tau2에 전이

> **진입 = 06-NOW(드리프트 교정 블록) + 이 설계.** tau2 학습 0 (도메인-타겟 금지·[[11]]). 상위 권위 = `INTEGRATED_TBOX_DESIGN §5/§7`(분해 아키텍처·*폐기됐다 재개*)·`CROSS_BENCH_TRANSFER_PLAN`·`FIXED_VS_VARIABLE`. 불변 = [[03-anti-drift]][[05-fixed-vs-variable]][[00-thesis]].
> ★사용자 결정(2026-06-20): 프론티어 표가 벤치-TBox 단일-LoRA 전이=음성(base 미만) 보임 → **설계가 주장하나 미측정인 분해-협업 e2e를 빌드·측정**.

## 0. 왜 재개인가 (폐기 재론 아님·anti-drift 정조준)
- **프론티어 실측(2026-06-20·학습0)**: base 7B 0.24 / 32B 0.60 / frontier ~0.81. **벤치-TBox ABox-swap 전이 전부 base 미만**: solo_cfb_mid 0.16·cons 0.12·sts 0.00·fact_* 0.02. ⇒ 현 형태(단일 LoRA를 full-agent로) 전이 = **음성**.
- **§5 분해 아키텍처는 2026-06-18 폐기됨 — 단 *짓지 않고*.** 폐기 사유(`INTEGRATED_SCAFFOLD_IMPL v2:4`): "facet3 단독 어댑터가 τ² 전체 맥락서 resolve_selection **0회 호출**." = **맨 어댑터를 full 맥락에 던졌을 때 안 불림.** 그걸 고치는 본체 = **step-scaffold(각 스페셜리스트를 자기 facet 맥락에 격리 호출)**인데, 복잡도 우려로 *미빌드*하고 단독LoRA로 피벗 → 그 단독LoRA도 실패(위).
- **격리하면 작동한다는 직접 증거**: forced-selection replay = 선택 격리 시 7B ground_OK 0.14→**0.62**(06-NOW). ⇒ 빠진 건 *라이브 에이전트에서 맥락을 격리하는 오케스트레이션*뿐. **두 시도(맨 어댑터·단독LoRA) 다 실패·분해 본체는 미측정** → 빌드 정당.

## 1. ★타깃 = 실제 병목 (선택 아님·정정)
- **selection = 소수경로(~5%)**(06-NOW·overnight census). **진짜 pass 동력 = action-execution/flow(DB-state) + 에러복구**: 7B 0.19 vs 32B 0.55 격차 = order_id grounding 날조(R1b·7B 27%)·write 행동 누락·P7 recovery 실패(too_many_errors 7B 36 vs 32B 0).
- ⇒ 분해는 **flow(SOP)·threading(TaskBench)·grounding(CFB)** facet 중심. selection(synth)은 한 facet일 뿐.
- 목표 = **분해-협업 7B > base 7B(0.24)**, 상대 Pareto-지배(절대수 약속 금지·`INTEGRATED_TBOX §5b` Risk B). 헤드라인 = ABox-swap airline 전이(재학습0).

## 2. 아키텍처 = 결정론 step-router + per-facet 스페셜리스트 (미빌드 본체)
주입점(코드 확인): `LLMAgent.generate_next_message`(모델이 tool_calls emit) + `BaseOrchestrator._execute_tool_calls`(이미 t2_resolve_patch가 hook).
- **step-router(신규·결정론·도메인-일반)**: 에이전트 턴마다 *현 sub-결정의 facet 타입* 분류 → 매칭 스페셜리스트 LoRA를 **격리 맥락**(자기 facet 학습 분포에 맞춘 축소 프롬프트)으로 호출. v1 폐기사유(0회 호출) 정조준.
- **스페셜리스트 = 기존 LoRA 재사용**(머지 안 함): SOP(flow/gate)·TaskBench(threading)·CFB(grounding)·synth(content-op). vLLM multi-LoRA per-request adapter 선택(`--lora-modules`·요청별 model명).
- **결합/offload = 결정론**(이미 존재): resolve 엔진(`t2_resolve_patch`)·provenance 검증(R1b·날조거부)·GateInterpreter(gate_spec). 
- **★도메인-일반성 제약**: router가 retail FSM을 하드코딩하면 위반([[05]]). router는 **gate_spec/ABox + facet-타입 신호로만** 구동(grep `if domain`=0). airline에 unchanged 작동이 일반성 증명.

## 3. 측정 (`INTEGRATED_TBOX §7.3` 부활·tau2 학습0)
- **3-way**: base 7B(0.24) / 스페셜리스트-분해(router·offload 없이) / 스페셜리스트+분해+offload(resolve/provenance/gate). 헤드라인 = > 둘 다.
- **autopsy**: facet3 0회 문제 해소(스페셜리스트 이제 불리나)·order_id 날조 소멸·recovery 개선.
- **decidable-비율**(thesis 핵심·§7.2): 결합 결정 중 결정론으로 닫히는 비율 vs 학습 필요. 대부분 det = offload 지배·소형 충분.
- **전이 매트릭스**: 같은 시스템 ABox만 swap → retail·airline 동시. = 일반성 유일 증거.
- 보상 = 결정론(DB-match·gated). 32B 궤적 348 = facet별 정답 **오라클**(대조용·학습 아님).

## 4. 단계 (최소→확장·v2 복잡도 우려 존중)
- **S-min**: 가장 큰 병목 1 facet(order_id grounding = CFB fetch-first + TaskBench threading + provenance)부터 router 격리 → base 초과하나. **빠른 생사 판정.**
- **S1**: 4 facet router 전체 + decidable-비율.
- **S2**: ABox-swap airline 전이(헤드라인).
- 각 단계 GO/NO-GO. NO-GO(분해도 base 못 넘음) = thesis 음성 → 비용결론으로 정직 후퇴.

## 5. Risk
- **R1 router 복잡도/맥락비용**(v2 폐기 사유): S-min 1-facet로 최소화·det-우선.
- **R2 절대천장 = real-NL facet *인식***(`§5b` Risk B): 헤드라인=상대 Pareto지 절대수.
- **R3 router가 일반 아니면**(retail 하드코딩) 전이 무효 → gate_spec/ABox 구동 강제·airline unchanged 시험.
- **R4 LoRA-swap 서빙 비용**: per-request adapter 선택은 vLLM 지원·검증 필요.

## 6. 리뷰 안건 (구현 전 확정 필요)
1. **facet-타입 분류 신호** = 무엇으로 router가 "지금 grounding/flow/selection"을 도메인-일반으로 판정하나? (gate_spec 상태 + 직전 tool 출력 + 미충족 인자?) — 핵심 미정.
2. **격리 맥락 구성** = 스페셜리스트에 주는 축소 프롬프트 = 어디까지 자름(full policy 빼고 facet 지시만? 도구 부분집합?).
3. **S-min facet 선택** = order_id grounding(최대 병목) 맞나.
4. **router 결정론 vs 잔여 consensus-LoRA**(§5 결합) — 처음엔 순수 결정론으로?
