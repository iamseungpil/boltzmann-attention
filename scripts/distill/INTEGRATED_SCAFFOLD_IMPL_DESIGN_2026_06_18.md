# 통합 구현 설계 v2 (2026-06-18) — ★단독 통합 LoRA + 결정론 offload + ABox-swap

> **v2 전환 (2026-06-18 · C0 실측 · 사용자 결정)**: ~~분해 멀티-스페셜리스트 + step-orchestrator (v1)~~ → **단독 통합 native LoRA 1개**.
> C0 실측: facet3 *단독* 어댑터는 τ² 전체 e2e 맥락(retail policy system + 17 도구 + 멀티턴 user-sim)서 resolve_selection **0회 호출**(synth 격리선 §28 1.00) = 맥락 불일치. 멀티-스페셜리스트 = step-orchestrator 복잡 + 맥락 비용. ⇒ 한 LoRA가 flow·threading·content-op를 일관 처리.
> **폐기 (v1)**: step-orchestrator · route() · SPECIALIST 맵 · multi-LoRA 서빙 · combine · 메타-판정 · decidable-비율 · wrongly-decided.
> **유지 불변**: 학습 TBox(base 아님) · 고정={TBox weights + Scaffold 엔진}/변경={ABox} · operand offload(§23D) · 실 τ² user-sim e2e 측정만. 상위 권위 = `INTEGRATED_TBOX_DESIGN`(§5 단독 전환 노트)·`FIXED_VS_VARIABLE.md`·`GATE_INTERPRETER_UNIFICATION_DESIGN`. 메모리 = `06-NOW`·`05-fixed-vs-variable`·`03-anti-drift`.

---

## 1. 아키텍처 = 단독 통합 LoRA + 결정론 offload
- **학습 (TBox · LoRA 1개 = `qwen7b_solo_sts`)**: SOP(flow/gate 명명) + TaskBench(data-flow threading) + Synth(content-op routing = resolve_selection emit). native tool_calls 직접 emit.
- **offload (결정론 · 고정 엔진)**: resolve 엔진(operand 조립·keep-rest·concrete item_id 해소) · GateInterpreter(gate enforce) · provenance/GROUND(concrete grounding). operand의 *delta 명명*만 LoRA · *조립/해소*는 엔진(§23D 라우팅 퇴행 회피).
- **ABox (swap · 재학습 0)**: 도구 카탈로그 + resolve config + GATE_SPEC. 전이 = retail↔airline · τ²→SOP-Bench.

---

## 2. 주입점 = `_execute_tool_calls` 1곳 (step-orchestrator 불요)
단독 LoRA가 native tool_calls(도메인 도구 + resolve_selection)를 직접 emit → `_generate_next_message` 패치 불요(v1 step-orchestrator 폐기). `_execute_tool_calls`만 패치:
- **resolve_selection 인터셉트** → 엔진 grounding (`t2_resolve_patch`·구현됨·crash-free).
- **GateInterpreter** gate enforce (`t2_gate_patch` → GateInterpreter).
- = `apply_integrated()` = resolve patch + gate patch. 벤치 측정 인터페이스 불변.

---

## 3. (A) GateInterpreter 통일
설계·스키마·kind closure = `GATE_INTERPRETER_UNIFICATION_DESIGN_2026_06_18.md` 그대로. 결선: `RetailGate` → `GateInterpreter(load_gate_spec(domain), resolvers)` · kind dispatch(auth/confirm/ownership/notice/preconditions) · 도구멤버십=gate_spec 데이터 · `GATE_DOMAINS` 하드셋 폐기 · **per-bench 분기 0**(grep `if domain`=0). retail/airline 같은 인터프리터 unchanged·gate_spec만 swap.

## 4. (B) resolve_selection wiring + operand offload (구현됨)
- 카탈로그 노출: `resolve_selection(op, attr?, among?, dir?, k?, set?)` (anchor_id 모델-가시 제외·`tau2_op_resolver:74-77`). `Environment.get_tools` 패치(검증: env에 17개·resolve 포함).
- 실행: `_execute_tool_calls`서 `tau2_op_resolver.resolve_op_tau2` 호출 · catalog/anchor를 직전 fetch(get_product_details/get_order_details)서 grounding.
- **operand = LLM delta 명명 + 엔진 조립**: LoRA가 op + intensional delta(`set`) emit · 엔진이 keep-rest copy + multi-attr 조립 + concrete item_id 해소(extensional). multi-attr 과소추출은 per-attr 분해 명명(LLM)→엔진 조립으로 회피(§22). **operand 전용 학습 금지**(§23D 퇴행).

---

## 5. 학습 데이터 (단독 합본 · 구현됨)
- **SOP**(`fc_build/sop_rand.jsonl`·alias) + **TaskBench**(`fc_build/tb_all_v4.jsonl`·alias) + **Synth content-op**(`route_native.jsonl`·**no_alias**=resolve_selection 실명·τ² 노출명 일치). CFB 제외.
- `fc_build_sft.py`(per-traj 랜덤 alias=grounding 강제·QC·bench-balance) → `sft_solo_sts.jsonl`. 빌드 실측: synth 6020 · taskbench 7000 · sopbench 5028 · resolve_selection 도구 6020행.
- 도구 = `build_solo_data.sh`(데이터) · `build_solo_train.sh`(1ep·r64·seq8192 → `qwen7b_solo_sts`).
- ⚠️ alias 규약: SOP/TB는 alias(R1 grounding·lexical 암기 차단) · synth는 no_alias(resolve_selection을 τ²서 그 이름으로 노출하므로 실명 유지).

---

## 6. 측정 = 2-way + 전이 (분해 decidable-비율 폐기)
- **arm0 base**(retail pass^1 **0.205**·실측) / **arm1 단독 LoRA + offload**(resolve + GateInterpreter).
- **헤드라인 = arm1 > arm0 상대 Pareto-지배 + ABox-swap 무재학습 전이**(retail→airline · τ²→SOP-Bench). 절대수 약속 금지.
- **★측정 분산(전수 확정 2026-06-18)**: base run-to-run **±2-3 pass(±0.05)** (agent 7B 비결정성 · 동일 조건서 11/40 양방향 flip · §0.175 vs 0.205 = 노이즈). **⇒ 유의 = Δ≥3-4 pass 또는 multi-trial(num_trials↑·pass^k)**. 작은 Δ는 노이즈. agent seed 고정 + vllm enforce-eager·max-num-seqs=1로 분산 축소 가능(비용↑).
- autopsy: **resolve_selection이 실제 불리나**(C0 facet3 단독 0회가 단독 통합으로 해소됐나·핵심) · order_id 날조 소멸 · gate 준수.

---

## 7. 단계 (현 진행)
- ✅ C0(facet3 단독 0회 → 단독 전환 트리거) · resolve wiring(`t2_resolve_patch`) · 데이터 합본(`sft_solo_sts`).
- 🔄 단독 LoRA 학습(`qwen7b_solo_sts`·1ep·r64·seq8192).
- ⏳ retail 실 e2e(`--resolve 1` + gate) → **resolve_selection 불리나** + base 0.205 대비 **Δ≥3-4 pass?**
- ⏳ GateInterpreter 통일(airline gate_spec) → ABox-swap 전이 매트릭스(retail↔airline·SOP-Bench).

---

## 8. 자가심사 (리뷰 안건 · 치팅 방어)
- **thesis-정합**: 단독 통합 LoRA = 학습 TBox(도메인-일반 flow+threading+content-op 명명) · base 아님. offload = decidable(resolve/gate/operand 조립/concrete). ABox swap = 전이. ✅
  - ⚠️ **단독 LoRA가 절차(flow 순서)도 학습 = procedure-offload 위반 아님**: thesis = LLM이 절차-타입 *명명/분류*(P1-P9) 학습(도메인-일반·전이). route()가 절차를 *발명*하면 위반이었으나(v1 폐기), TBox가 학습하면 thesis 그대로. **ABox-swap 전이(retail→airline·τ²→SOP)가 도메인-일반성 증명** = monolith지만 도메인-타겟 아님([[11-transfer-direction]]).
- **per-bench 분기 0**: resolve·GateInterpreter 전부 카탈로그/spec-도출(grep `if domain`=0·CI).
- **operand offload**(§23D 퇴행 회피) · **contamination 0**(τ²·SOP-Bench held-out) · 보상 결정론.
- **분해-측정 포기 비용(정직)**: decidable-비율(학습 vs offload 분담선 오케스트레이션 실측)은 단독 전환으로 포기. 헤드라인 = 통합 TBox+offload Pareto-지배 + ABox-swap 전이로 축소.

## 9. 결정 로그
- **단독 전환**(2026-06-18·C0 facet3 0회 실측·사용자): 분해 멀티/step-orchestrator/route/combine/decidable-비율 폐기.
- **operand offload 유지**(§23D)·CFB 폐기·synth no_alias·anchor_id 제외(`_emit_args`).
- **측정**: 통합 Pareto+전이. base 분산 ±2-3 pass → 유의 Δ≥3-4 또는 multi-trial.
- **유지**: GateInterpreter 통일(A)·resolve wiring(B)·고정/변경 경계·실 e2e만.
