# 격리 실험 (2026-06-18) — facet별 NL→formalize *출력* 전이를 confound 없이 확정

> 동기: 지금까지 전이 해석이 흔들린 근본원인 = 실험이 전부 **e2e/STACK/net**으로 측정 → LLM의 NL→formalize가 scaffold(실행)·offload(resolve)·어휘(도구명)와 *섞임*. theory는 명확(LLM = flow·data 양쪽서 NL→formalize·intensional typed 구조). ⇒ **각 facet의 formalize *출력*을 직접 채점·confound 격리·ABox-swap 전이 측정**. 진입=메모리 `04`·`01`. 불변 = formalize 출력 채점(e2e 아님)·재학습0 swap·결정론 채점.

## 공통 프로토콜 (4 facet 동일)
- **입력**: NL + 타깃 도메인 ABox(catalog/tools/policy) [+ flow는 state/history].
- **LLM 출력**: 그 facet의 **intensional formal 구조만**(concrete ID 아님·typed selector).
- **채점**: gold formal 구조와 **결정론 매치**(exact/F1) — **e2e success 아님**(scaffold/offload 안 거침).
- **전이**: source(도메인/벤치) 학습 → ABox swap → 타깃(held-out 도메인·τ²) **재학습0**. base floor·in-domain ceiling 동시.
- **격리(핵심)**: scaffold(게이트/실행) 제거·offload(concrete resolve) 제거·어휘(도구명 ground) 제거 → **순수 formalize-전이만 측정**.
- 판정: formalize-출력 전이가 **base 초과 + held-out≈in-domain**이면 그 facet의 NL→formalize 전이 *확정*. (e2e가 아니라 formalize 자체.)

## facet (1) flow-formalize 전이
- **출력**: 다음-스텝 TYPE ∈ {gather(slot)·act(op)·confirm·stop·refuse} + 순서(gather-before-act·confirm-before-write).
- **gold**: SOPBench gold 궤적의 각 스텝 정답 action-type.
- **metric**: per-step action-type 정확도 + 순서 정확도. **scaffold 미실행**(모델 emit type만 채점) = STACK confound 제거(77.3% 문제 해소).
- **전이**: t1c LODO 어댑터(6도메인 학습)→held-out 도메인 + τ². base 대비.
- **격리 확정 질문**: flow-formalize(NL→action-type)가 *adapter weight로* 전이하나 — scaffold가 나르는 e2e와 분리. (현 26.87% STACK은 regime-flag 혼합·이건 순수 type-emit.)
- 자산: SOPBench gold(있음)·`qwen7b_tbox_t1c_lodo_*`(있음). 신규=action-type 추출·offline 채점기 `flow_formalize_eval.py`.

## facet (2) threading-formalize 전이
- **출력**: 각 인자의 **reference**(어느 이전 스텝 출력서 옴·`<node-j>` 인덱싱) = intensional 바인딩.
- **gold**: TaskBench gold task_links(의존 그래프).
- **metric**: reference-link F1, **도구명 grounded(주어짐)** → 어휘-간섭 confound 제거(net≈0 상쇄 해소·TB:184). threading 규율만 채점.
- **전이**: tb_lodo(HF+MM 학습→daily/τ²), 도구명 ground한 채 threading-F1. base 대비(self-ref율).
- **격리 확정 질문**: 참조-인덱싱 규율이 *weight로* 전이하나(어휘간섭 빼고) — TB:184가 "전이됨"이라 했으나 net으로 가려짐. 이건 직접.
- 자산: TaskBench gold(`tb_sft` 있음)·`qwen7b_tb_lodo_*`(있음)·`tb_census.py`(self-ref 계산·확장). 신규=grounded-name threading-F1 전이 채점.

## facet (3) content-op-formalize 전이 (★keystone·native 다리)
- **출력**: **native tool_call** `resolve_selection(op,attr,among,set)`(op-명명+operand·anchor_id 모델제외).
- **gold**: §21 gold op-IR를 native로.
- **metric**: op-명명 정확도 + operand 정확도, **native 포맷서**(op-IR "Output ONLY JSON" 아님 → §23E 회피).
- **전이**: synth-native LoRA → retail+airline config-swap. **§21(held-out 1.00·τ² 0.44)을 native로 재현하나**.
- **격리 확정 질문**: 유일하게 증명된 cross-bench 학습-전이(§21·op-IR)가 **native 포맷서 살아남나**. = 통합의 단일 최중요 실험. 죽으면(§23E) 다리 재설계.
- 자산: `synth_depth.py`(생성기·있음)·`tau2_op_resolver`(엔진·있음)·`tau2_op_eval`(채점·있음). 신규=`synth_to_nativefc.py`(생성기→native tool_call 직생성)·native eval.

## facet (4) operand-formalize(set) 전이
- **출력**: **set**(어느 attr→어느 값·keep-rest delta·intensional/typed) — concrete 아님.
- **gold**: synth/τ² gold set.
- **metric**: set exact-match/per-attr F1, **엔진 resolve *전*의 emit된 set** 채점 → engine-match·concrete-resolution(offload) 분리(§20-B 혼동 해소).
- **전이**: synth multi-attr LoRA → retail+airline, set-formalize 정확도.
- **격리 확정 질문**: operand-formalize(set 추출)가 *formalize 스킬로* 전이하나. **theory=LLM 일(per-attr 명명)** vs 리뷰어=offload 타깃. §23D "학습하면 라우팅 퇴행"은 *e2e/라우팅 안에서* 학습한 confound — 격리하면 set-formalize 전이는 미측정. **이 실험이 theory vs 리뷰어를 가름.**
- 주의: set이 deep multi-attr이면 per-attr 명명(LLM)+assembly(engine) 경계(§22)·여기선 *명명*만 채점.
- 자산: `synth_depth.py`(multi-attr substitute·있음)·`ma_gold_extract`(τ² set gold·있음)·`tau2_op_eval`(by-op·있음). 신규=set-formalize 격리 채점(engine 전).

## 실행 순서 (확정 우선순위)
1. **facet (3) native keystone** — `synth_to_nativefc.py` 빌드→synth-native LoRA→retail+airline native op-명명 전이. (유일 증명전이의 native 생존 = 통합 가부 결정.)
2. **facet (4) set-formalize 격리** — theory vs 리뷰어 판정(operand 전이가 학습-스킬인가 offload인가). offline·싸다.
3. **facet (1)(2) formalize-전이** — action-type·threading-ref 격리 채점(기존 어댑터·offline). scaffold/어휘 confound 없이 weight-전이 직접.
- 전부 offline 채점(e2e 시뮬 불요·키0·싸다). 결과 = `M_A_RESULTS §28` + 이 문서 표.

## 정직
- formalize-전이 양성 ≠ e2e 양성(e2e엔 scaffold/offload/복구 더 필요). 이 실험은 *LLM-leg의 전이*만 확정 = theory 검증.
- in-domain ceiling·base floor 동시 측정(전이 gap 정의). held-out≈in-domain이어야 전이.
- facet (4)가 theory 편(전이 양성)이면 §23D는 "e2e-내 학습"의 아티팩트로 재해석·facet4=학습-leg. 음성이면 리뷰어 편(offload).
