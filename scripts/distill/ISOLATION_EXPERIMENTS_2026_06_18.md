# 격리 실험 (2026-06-18) — facet별 NL→formalize *출력* 전이를 confound 없이 확정

> 동기: 지금까지 전이 해석이 흔들린 근본원인 = 실험이 전부 **e2e/STACK/net**으로 측정 → LLM의 NL→formalize가 scaffold(실행)·offload(resolve)·어휘(도구명)와 *섞임*. theory는 명확(LLM = flow·data 양쪽서 NL→formalize·intensional typed 구조). ⇒ **각 facet의 formalize *출력*을 직접 채점·confound 격리·ABox-swap 전이 측정**. 진입=메모리 `04`·`01`. 불변 = formalize 출력 채점(e2e 아님)·재학습0 swap·결정론 채점.

## 0. ★maximal thesis = ⟨TBox 고정 + scaffold 고정⟩ + ABox-only swap → transactional 벤치 전부 (가능성 분해)
명제(최강형·사용자 2026-06-18): **TBox(학습·weight)와 scaffold(결정론)는 도메인·벤치 무관하게 *미수정*, ABox config만 swap**으로 새 벤치(tau2·SOP-Bench·…)를 푼다. = 전이.

가능 조건 4개 (각 실태·정직):
1. **closure**: 벤치 primitive ⊆ P1-P9 ∪ 생성원. transactional는 닫힘(census·생성원대수 §8). **AppWorld(G_loop)·WebArena(G_ground)·TravelPlanner(G_csp)는 scope-out** → "25개 전부"=과주장·정직형=**transactional ~17-19개**.
2. **TBox formalize-전이**: 고정 TBox가 새 벤치서 NL→formalize 정확. facet 3 cross-bench **증명**(§21)·**1,2,4 = 미검증 = 이 실험(§facet 1-4)이 측정**.
3. **scaffold 벤치-일반성**: *한* 결정론 엔진(GATE_SPEC 해석기+resolve+orchestrator)·ABox로만 파라미터·**per-bench 분기 0**. **현재 FALSE**(SOPBench t2_gate ≠ tau2 RetailGate=별도 코드) → §S(아래)가 keystone.
4. **ABox 충분성**: A1(카탈로그)=기계적·**A2(정책 NL→GATE_SPEC)=유일 난제**.

⇒ **"가능한가" = ②(facet 1-4 격리전이) + ③(scaffold 통일 ablation) 둘 다 통과 시 가능**(transactional scope). 이 문서 = ② 측정. ③ = §S.

**격리 원칙(공통)**: 각 facet 실험서 **TBox·scaffold 고정·ABox만 swap·재학습0**. cross-bench(→tau2) = within-bench LODO보다 강한 진짜 전이.

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

## §S. ★scaffold 벤치-일반성 keystone (조건 ③ — maximal thesis의 가장 어려운 leg)
- **질문**: 한 결정론 엔진(GATE_SPEC 해석기 + resolve + step-orchestrator)을 ABox(정책→GATE_SPEC·카탈로그)로만 파라미터화 → SOPBench·tau2 둘 다 **per-bench 코드분기 0**(grep `if bench`/`if domain` = 0)으로 작동하나.
- **현 실태**: 별도 scaffold(t2_gate·RetailGate) → 통일 필요. plan §2b "GATE_SPEC replay=일반"이 설계.
- **실험**: ① 두 벤치의 게이트/순서를 **GATE_SPEC(ABox)로 표현** → 일반 해석기가 replay. ② grep `if bench`=0 검증. ③ ABox-ablation(빈/틀린 GATE_SPEC→붕괴). ④ 동일 엔진 unchanged로 SOPBench+tau2+(airline) 작동.
- **판정**: 한 엔진+ABox-only가 다벤치 작동 = ③ 양성·"그냥 결정론 프로그램" 공격 방어(마스터 §1: procedure-offload 금지=답지·fact-offload OK). per-bench 분기 있으면 = bench-베이킹 = thesis 실패.
- = facet 전이(②)와 *직교*·둘 다 필요. (TBox 전이는 facet 실험·scaffold 일반성은 여기.)

## 실행 순서 (확정 우선순위·진행상태)
1. **facet (3) phase-1** [★가동중·`facet3_native.sh`] — synth-native LoRA → synth held-out native op-명명. **method-gate**(native formalize-채점이 작동? §23E 생존?). 양성이면 같은 방법이 1·2·4에 적용됨.
2. **facet (3)+(4) phase-2** — 위 *동일 LoRA* → retail+airline native eval(`synth_native_eval` + τ²-case 변환). op-명명(=facet 3 cross-bench) + set/operand 정확도(=facet 4 cross-bench). **facet 4가 여기서 piggyback** → "operand=offload(리뷰어) vs LLM-formalize(theory)" 직접 판정.
3. **facet (1)(2) cross-bench** — 고정 SOPBench(t1c)·TaskBench(tb_lodo) 어댑터 → **tau2** action-type/threading-ref formalize 채점(ABox만 swap·재학습0·scaffold 미실행). tau2 gold action-type/threading 추출 신규. (1·2는 phase-1이 method 검증 후 빌드.)
4. **§S scaffold 통일** — 별 트랙(조건 ③·직교).
- 의존: phase-1(method) → 2,3. facet 3·4는 한 LoRA로 동시. 1·2는 별 어댑터.
- 전부 offline 채점(e2e 시뮬 불요·키0·싸다). 결과 = `M_A_RESULTS §28` + 이 문서 표.

## 정직
- formalize-전이 양성 ≠ e2e 양성(e2e엔 scaffold/offload/복구 더 필요). 이 실험은 *LLM-leg의 전이*만 확정 = theory 검증.
- in-domain ceiling·base floor 동시 측정(전이 gap 정의). held-out≈in-domain이어야 전이.
- facet (4)가 theory 편(전이 양성)이면 §23D는 "e2e-내 학습"의 아티팩트로 재해석·facet4=학습-leg. 음성이면 리뷰어 편(offload).
