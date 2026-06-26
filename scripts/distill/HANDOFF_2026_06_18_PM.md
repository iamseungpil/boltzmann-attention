# HANDOFF 2026-06-18 PM — ★오프라인 op-eval 신뢰불가 판명·실 user-sim e2e로 가야 함 (다음 세션 진입점)

> 이 세션 후반 = 실수 누적으로 사용자 신뢰 상실. 정직한 기록·다음 세션은 *실 user-sim e2e*만 신뢰할 것. 권위 = `ma/M_A_RESULTS.md §28-34`·불변 = `FIXED_VS_VARIABLE.md`·메모리 `03-anti-drift`·`04-current-position`.

## 0. ★이 세션의 핵심 교훈 (반복 금지)
**내가 쓴 오프라인 op-eval(`tau2_op_eval`·모델에 태스크 텍스트 직접 주고 단발로 op-IR/resolve 채점)은 실제 τ²를 *충실히 재현하지 못한다*.** 그래서 §17-§32의 τ² 전이/operand 수치는 **전부 신뢰불가 프록시**다.
- τ²는 **멀티턴 user-sim**: user가 `task_instructions`(detail)를 *대화로 점진 공개*·agent가 물어서 얻음.
- 내 오프라인 eval: `reason_for_call`(요약)만 주면 **정보 부족**(cabin/attr 누락)·`task_instructions` 전체 주면 **노이즈 과다**(persona·복수 goal). **둘 다 실 세팅과 불일치.**
- ⇒ **신뢰가능한 측정 = 실 τ² user-sim e2e(`t2_run_gated`)뿐.** 오프라인 op-eval 프록시 폐기.

## 1. ★다음 세션 첫 행동 (사용자 지시)
1. **base 7B 실 τ² user-sim e2e → pass^1 ≈ 0.17 재현 확인**(retail anchor·§24 retail_base_anchor 0.175). = 파이프라인 검증. 도구 = `tau2/real_e2e_base.sh`(작성됨·미실행·base 7B·retail+airline·gpt-4.1 user-sim) 또는 `tau2_eval_adapter.sh`/`t2_run_gated.py`.
2. **그 다음 개선되는지** — offload(결정론 resolve·gate)/학습 TBox가 0.17을 올리나. **실 user-sim e2e로만 측정.**
3. 오프라인 op-eval로 결론 내지 말 것(이 세션 실수의 근원).

## 2. 신뢰가능 (SETTLED·synth 한정·trust)
- **§28 facet-3 native = synth held-out op-naming 1.00**(7-op mixed·새 어휘·native resolve_selection 포맷·base 0.76→trained 1.00). = op-IR(§21)이 native 형식서 재현·§23E 다리 *synth 레벨* 확보. **단 synth-only·τ² 주장 아님.**
- **§22-23 밤샘(width/K/decomp·synth)**: under-extraction=소형조건부(frontier gpt-4.1 평탄·decomp 7B 0.51→0.87·14B 1.0·llama8b 실패)·wide-train 라우팅 퇴행(§23D)·diversity 캡(§23C). **synth 결과.** §33에 width 전체표 확정(단 70B/72B 미완).
- **아키텍처/설계(이 세션·trust)**: `FIXED_VS_VARIABLE.md`(고정={TBox+Scaffold}/변경={ABox}·ABox=함수스키마config+GATE_SPEC·§2bis)·`EXPERIMENT_DESIGN.md`(전면개편·복원본)·`GATE_INTERPRETER_UNIFICATION_DESIGN`·`ABOX_CONFIG_FORMALIZATION`·`INTEGRATED_TBOX_DESIGN`. 메모리 00-05.

## 3. ★오염/철회 (DO NOT TRUST·재유도 금지)
- **§29-32 τ² operand 분석 전부 = 오프라인 op-eval 기반 = 신뢰불가**: "facet-3 τ² 역전이(0.19<0.34)"·"operand 3분해"·"enum-snap"·"wrong-value-selection 잔여" — 전부 프록시 결함. 실 user-sim서 재측정 필요.
- **§34 두 버전 다 불완전**: (a)"airline 케이스 underspecified=벤치 결함" **철회**(리더보드 작동=벤치 정상·사용자 교정) (b)"ma_gold_extract가 reason_for_call만 써서=추출버그" → 고쳐서 task_instructions로 재측정하니 **retail 0.34→0.25 악화·airline 0.59 불변** = 이 가설도 불완전(full info=노이즈). ⇒ **오프라인 eval 자체가 문제**(§0).
- **§31 = §22-23 재유도**(anti-drift 위반·이미 settled를 다시 함).
- τ³가 airline 27태스크 수정(README·SABER) — 내 클론이 옛 결함 태스크일 수 있음(별도 확인).

## 4. 상태/도구
- **GPU**: 이 세션 잡들 정리 필요(GPU0 재측정 잔류 가능·`nvidia-smi`로 확인 후 kill).
- **`ma_gold_extract`**: `_full_nl`(task_instructions 포함)로 수정함 — 단 오프라인 eval 폐기하므로 무관.
- **adapter `facet3_native_ep1`**: synth content-op 라우터(resolve_selection emit)·**τ² agent로 직접 못 씀**(τ² 도구는 exchange 등이지 resolve_selection 아님). e2e엔 부적합.
- **신규 도구(이 세션)**: `synth_to_nativefc.py`·`synth_native_eval.py`·`tau2_op_eval --native`·`tau2_native_operand_autopsy.py`·`tau2_operand_cause_confirm.py`·`facet3_native.sh`·`facet3_eval_rerun.sh`·`facet3_tau2_native.sh`·`real_e2e_base.sh`(미실행). **op-eval 계열은 §0 이유로 신뢰불가.**
- **실 e2e 도구(신뢰)**: `t2_run_gated.py`(user-sim gpt-4.1·gated)·`tau2_eval_adapter.sh`·`tau2_autopsy.py`·`tau2_collapse_autopsy.py`.

## 5. 큰 그림 (어디로)
- thesis/아키텍처는 settled(`FIXED_VS_VARIABLE`·`EXPERIMENT_DESIGN`): 학습 TBox + 고정 scaffold(GateInterpreter) + ABox-swap. 조건 ②(TBox formalize-전이)·③(scaffold 통일)이 실험 타깃.
- **단 측정은 실 user-sim e2e로**. 이 세션이 오프라인 프록시로 헤맨 게 핵심 실수.
- 다음 = base 0.17 재현 → offload/TBox가 올리나 실 e2e로 검증 → §24-25 autopsy(order_id 날조·collapse)가 실 벽.
