# 통합 TBox 설계 (2026-06-18) — 4벤치(SOPBench+TaskBench+CFB+Synth) = 분해된 멀티-스페셜리스트 오케스트레이션

> ★아키텍처 = 단일 merged LoRA 폐기. per-bench 스페셜리스트 LoRA(기존 자산) + 얽힌 결정의 typed-스텝 분해 + 결합(결정론 우선·잔여만 consensus LoRA) + 결정론 offload + ABox-swap. "통합"=weight 머지 아니라 *오케스트레이션*. (§5가 핵심.)

> 진입 = 메모리 `01-four-bench-tbox`·`04-current-position`. 불변 = `03-anti-drift`·`11-transfer-direction`·`10-roles-deterministic`. 상위 = `CROSS_BENCH_TRANSFER_PLAN_2026_06_14.md`(Option B native)·`THESIS_STATEMENT_2026_06_16.md §7`·`NATIVE_FC_CONVERTER_DESIGN_2026_06_14.md`. 사용자 지시 = "통합 TBox 설계하라"·"설계먼저→리뷰→구현".

## 0. 확인 — 통합 TBox는 *존재하지 않는다* (직접 검증, 2026-06-18)
- TaskBench native-FC ✓ (`scratch/tb_sft/*`·`fc_build/tbnfc_*` = `{tools, messages[tool_calls]}`).
- SOPBench native-FC ✓ (v-series가 사용·`sft_alias_run/lodo_train_*`).
- CFB native-FC ✓ (v7 = grounded 2-hop).
- **Synth = op-IR JSON만** (`route_sft.jsonl`: system "Output ONLY JSON" → `{"op":"filter",...}`). **native-FC 버전 없음.**
- **4벤치를 한 native 포맷으로 묶은 데이터·모델 = 없음.** v7 = SOPBench+TaskBench(+CFB), synth content-routing 미포함. ⇒ 통합 TBox = 신규 학습.

## 1. 목표 (thesis 헤드라인)
**4벤치의 facet을 per-bench 스페셜리스트 LoRA(기존 자산)로 두고, 얽힌 결정을 typed 스텝으로 *분해*해 오케스트레이션** + 결합은 결정론-우선·잔여만 consensus LoRA + 결정론 offload(resolve/provenance/gate) + ABox-swap 전이. (단일 merged LoRA = 폐기·§5.) 헤드라인 = **분해된 협업이 base와 monolith를 상대 Pareto-지배 + ABox-swap 무재학습 전이(retail→airline·τ²→SOP-Bench) + 결합의 decidable-비율 실측**. (절대수 약속 금지·§5b.)

## 2. 공통 표현 = native tool-calling (Option B)
표준 OpenAI function-calling(`{tools, messages: [...tool_calls{name,arguments}...]}`)·vLLM `--tool-call-parser hermes`. 포맷매칭 브리지 제거 = 공격표면 소멸. config=포맷 validity(XGrammar) / 학습=내용(어느 도구·인자). **전 4벤치를 이 한 포맷으로.** (op-IR "Output ONLY JSON"은 §23E서 native agent 붕괴 → 폐기.)

## 3. ★핵심 설계 — 4벤치 → 공통 native 포맷 (각 벤치의 학습 primitive 보존)
| 벤치 | 학습 primitive | native 포맷 매핑 | 데이터 |
|---|---|---|---|
| SOPBench | gather-first(R2)·gate/decision-offload(R3)·순서·write-confirm(P5/P6/P8) | 도메인 도구 tool_calls + 순서/게이트 규율 supervise | 재사용(native 변환 존재) |
| TaskBench | tool-DAG threading·node/edge 선택(R1/R4/R6·P2a/P3) | 멀티 tool_calls·출력→인자 threading | 재사용(`tb_sft`·`tbnfc_*`) |
| CFB | grounded 2-hop fetch-first(R1b·P2b) | getter tool_call → 결과서 인자 grounding → action tool_call | 재사용(v7 데이터) |
| **Synth** | content 생성원 op-routing(filter/argmax/.../substitute/create) | **★resolve-tool로의 native tool_call** (아래 3a·op-IR 폐기) | **신규 변환 필요** |

### 3a. ★Synth → native: content-op = resolve-tool 호출 (신규·핵심 기여)
op-IR(`{"op":"argmax",...}` 텍스트 출력)을 폐기하고, **도구 카탈로그(ABox)에 결정론 `resolve_selection` 도구를 노출**:
**스키마 = 기존 op-IR 그대로**(신규 설계 불필요): `resolve_op_tau2`가 소비하는 `{op, attr?, among?, dir?, k?, set?}`를 도구 JSON-schema로 노출(`tau2_op_resolver.py:57`). synth 추상 catalog→도구 스키마는 among/attr 1:1.
```
resolve_selection(op: argmax|argmin|rank|filter|comparative|substitute|create,
                  attr?: <name>, among?: {<attr>:<val>}, dir?, k?, set?: {<attr>:<val>}) -> item_id
```
- **★anchor_id는 모델-가시 인자에서 *제외*** (리뷰 필수보정·`tau2_op_resolver.py:74-77`): 모델이 anchor_id를 채우면 = order_id 날조(R1b/§25 collapse)를 resolve_selection 안으로 재수입. anchor(수정대상 item)는 **offload 층이 직전 fetch 결과(context anchor)로 grounding**해 엔진에 주입. 모델은 op·set만 명명. = §3b를 R1b-정합으로.
- 모델은 NL→op를 **native tool_call**로 명명(`resolve_selection(op=argmax, attr=score, ...)`). = synth 라우팅을 *어휘만 다른 같은 학습*(§21 양성)으로 보존하되 포맷은 native.
- 도구 *구현* = 결정론 엔진(`resolve_op_tau2`·`resolve_operation`) = 깊은 실행 offload(B=∞). 모델은 *위임만*([[10-roles-deterministic]]).
- synth 학습데이터 = **생성기(`synth_depth.py`)서 native-FC 직생성**(op-IR `route_sft.jsonl` 재변환 *아님* — op-IR 프레이밍 상속 회피): assistant 턴 = `tool_calls:[resolve_selection(...)]`(anchor_id 제외), tool 턴 = 엔진 결과. (신규 `synth_to_nativefc.py` = 생성기 직호출.)
- **이게 §23E를 푼다**: 모델은 native tool_call을 emit(op-IR 텍스트 아님) → 파서 인식·도구 실행 → collapse 없음.

### 3b. resolve-tool은 *real-bench action*과 분리 (offload 컴퓨테이션·내재화/외재화 경계)
- thesis 구분: **내재화 = TBox LoRA 하나**(위임/오케스트레이션 스킬) / **외재화 = 결정론 전부**(resolve 엔진·provenance·gate·ABox). resolve_selection 노출 = 내재화된 TBox가 외재화된 엔진에 *명시 위임*하는 경계 = thesis 정합(인터셉트는 위임을 숨겨 내재화 안 됨 → 폐기).
- τ²/real에서 resolve_selection은 *real 도구를 대체하지 않음* — 모델이 real action tool(exchange_…)의 *인자(item_id 등)를 계산*하려 호출하는 **제공된 컴퓨테이션 도구**(코드-인터프리터/계산기 동형). 벤치 성공기준(최종 DB 상태) 불변. = 치팅 아님(§8).

## 4. 결정론 실행 층 (offload·config-도출·학습 아님)
- **resolve_selection 구현** = content-op 엔진(§3a).
- **R1b provenance 검증기** = 인자값 ∈ {user, 직전 tool 출력}·날조/스키마-example 거부(CFB가 학습한 fetch-first의 결정론 가드). 기존 `t2_gate_patch` provenance 재사용(단 deny→재생성·일반).
- **gate** = G1-G4 정책 replay(SOPBench가 학습한 순서/confirm의 결정론 집행·GATE_SPEC).
- 전부 **ABox/스키마 도출·도메인-일반**(per-domain 분기0). [[10-roles-deterministic]]·[[11-transfer-direction]].

## 5. ★아키텍처 = 분해된 멀티-스페셜리스트 (단일 merged LoRA 폐기)
**단일 merged LoRA = 폐기**(monolith=thesis 반대). 얽힌 결정(한 tool-call이 게이트+grounding+threading 동시 요구)을 *합쳐 한 LoRA에 내재화*하는 건 틀림 — 얽힘은 머지의 이유가 아니라 **분해의 이유**. 분해를 오케스트레이션 레벨까지:

- **스페셜리스트 LoRA = 기존 학습 자산 재사용**(per-facet·머지 안 함): SOP(게이트-상태)·TaskBench(threading)·CFB(grounding)·synth(content-op 명명). 각자 자기 facet을 *한 스텝*에 산출. 각 LoRA는 도메인-일반 TBox·ABox-swap 전이.
- **얽힌 결정 → 순차 typed 스텝**(결정론 scaffold가 몰아감·thesis §2 "typed 증분 스텝"): 예 "지금 exchange 호출?" = (1)게이트-상태 확인[SOP/det-gate] (2)인자 grounding 확인[CFB/provenance] (3)order_id threading 확인[TaskBench/det] (4)content-op resolve[synth→resolve 엔진·§3a].
- **★결합(combine) = 별도 *다음* 스텝**: facet 출력들을 "emit / 보류 / 어느 도구" 결정으로 합침.
  - **결합 규칙이 decidable → 결정론 알고리즘**(offload·예: gate∧grounded∧resolved → emit).
  - **decidable 아님 → consensus/통합 LoRA를 *그 결합 결정만* 학습**(작은 잔여 학습·다른 facet 학습 안 건드림).
- **★이게 측정하는 것 = thesis 핵심**: 결합의 *얼마가 decidable(offload)이고 얼마가 학습(consensus LoRA) 필요*인가. 대부분 decidable이면 = offload 지배·소형 충분. = "decidable→결정론 / 환원불가→학습" 분담선을 오케스트레이션 레벨서 직접 실측.

**효과**: ⓐ Risk A(벤치 간 간섭) 소멸 — blend 없음·각 LoRA 스페셜리스트 유지. ⓑ 기존 학습 LoRA 그대로 사용(신규 대량학습 불요). ⓒ 신규 학습은 synth→native 스페셜리스트 + (필요시)결합 consensus LoRA만.

**내재화/외재화**: 내재화 = 스페셜리스트 LoRA들 + (필요시)consensus LoRA(전부 TBox·weight) / 외재화 = 결합 알고리즘·resolve 엔진·provenance·gate·ABox·step-scaffold.
- ⚠️ **wide-train 교훈(§23D)**: binding(set)을 모델에 욱여넣으면 라우팅 퇴행 → 각 facet 스페셜리스트는 자기 facet만·결합도 det 우선.

## 5b. ★Risk 사전등록 (낮은 절대수를 나중에 "새 벽"으로 격상하는 표류 차단·[[03-anti-drift]] 6)
- ~~Risk A (벤치 간 간섭)~~ = **분해 아키텍처(§5)로 소멸**(blend 없음). 잔여 위험 = 스텝-scaffold가 얽힌 facet을 올바른 순서로 분해하는가 + 결합 consensus LoRA가 실제로 필요한 결합 수(decidable 비율).
- **Risk B — 절대 e2e 천장은 offload 아니라 real-domain 라우팅 *인식*일 공산.** 증명된 전이 = **0.44(op-선택 정확도)지 1.00 아님**(§21). 이게 **라우팅-leg 예상 천장**. τ² 측정은 offload가 collapse(order_id 날조)를 소멸시키는 **상대 우위(Pareto-지배)**를 보일 것·절대수는 messy real NL op-인식에 묶임. **헤드라인 = 상대(Pareto-지배). 절대수 약속 금지.**

## 6. ABox (per-domain swap·재학습 0)
- A1 = 도구 카탈로그(도메인 도구 + resolve_selection + getter) — 기계 생성.
- A1' = resolve_selection config(attr 타입 ordinal/categorical·vocabulary) — 카탈로그서 도출.
- A2 = 정책→GATE_SPEC(front-end 난제).
- **swap 시험**: retail↔airline(같은 시스템 unchanged)·τ²→SOP-Bench(Amazon). 빈/틀린 A1/A2 → 붕괴(ablation).

## 7. 측정 (synth-first)
1. **스페셜리스트 sanity**: 각 LoRA(SOP·TaskBench·CFB·synth-native)가 자기 facet held-out서 작동(synth는 §21 1.00 회귀 없나·native 포맷서). = 스텝 출력 신뢰.
2. **★결합 decidable-비율**(thesis 핵심): τ² 궤적서 "emit/보류/도구" 결합 결정 중 결정론 알고리즘으로 닫히는 비율 vs consensus LoRA 필요 비율. 대부분 det = offload 지배.
3. **τ² e2e 3-way**: base(0.17 floor) / **스페셜리스트-only**(분해 오케스트레이션·결정론 결합/offload 없이) / **스페셜리스트+분해결합+offload**(resolve/provenance/gate). **헤드라인 = > 둘 다(상대 Pareto-지배)**. autopsy: collapse(order_id 날조) 소멸? (절대수 약속 금지·§5b Risk B.)
4. **전이 매트릭스**: 같은 시스템(스페셜리스트+결합+offload), ABox만 swap → retail·airline·(SOP-Bench) 동시 작동. 재학습0. **= offload-일반성의 유일 증거**(§8b).
5. 보상 = 결정론(τ² DB∧NL∧comm·compliant-pass gated).

## 8. ★자가심사 (리뷰 안건)
**(a) thesis-정합?** — ✓. 학습=도메인-일반(op-명명·순서·fetch-first·threading=4벤치 P-primitive)·offload=decidable(resolve/provenance/gate)·ABox=사실·전이=swap. [[00-thesis]] 그대로. base+bespoke(UNIFIED) 아님(학습 TBox가 에이전트).
**(b) tau2-치팅 공격 방어:**
- resolve_selection = **도메인-일반 decidable 도구**(ABox-파라미터·카탈로그서 config 도출), 손-코딩 retail 정답 아님. **전이(retail→airline 같은 도구·§19 오라클 32/32+27/27)가 일반성 증명.**
- real action 도구 대체 아님 = 인자 *계산* 보조(계산기 동형)·벤치 기준 불변(§3b).
- 라우팅은 **학습**(§21 전이 양성)·scaffold가 op 고르지 않음. provenance/gate = per-domain 분기0(grep `if domain`).
- contamination 0 = τ²·SOP-Bench는 학습에서 held-out.

### 8b. ★synth 깨끗한 수치는 offload-일반성 증거가 *아니다* (자가차단·[[03-anti-drift]] 6)
synth는 정의상 *라우팅 벤치* → resolve_selection이 일의 100%를 함(정당). 따라서 synth closure는 **라우팅 증거일 뿐 offload-일반성 증거가 아님.** offload-일반성의 유일 증거 = **grep-clean(per-domain 분기0) config로 τ²/SOP-Bench ABox-swap 전이**(§7.4). synth 수치로 일반성 주장 금지.

## 9. 결정 (리뷰 1차 반영·확정/잔여)
1. **[확정] resolve_selection 스키마 = 기존 op-IR 그대로**(`{op,attr?,among?,dir?,k?,set?}`·신규 설계 불요)·**anchor_id 모델-가시 제외**(offload가 context anchor grounding·§3a). synth-nfc = 생성기 직생성.
2. **[확정] 단일 merged LoRA 폐기 = 분해된 멀티-스페셜리스트**(기존 per-bench LoRA 재사용·머지 안 함). 얽힌 결정=typed 스텝 분해·결합=결정론 우선·잔여만 consensus LoRA(§5).
3. **[확정] 노출(expose)** — 내재화/외재화 경계상 정합(위임=내재화·실행=외재화·§3b). 인터셉트 폐기. anchor_id grounding 보정과 묶음.
4. **[확정] synth-first + 4벤치 회귀 배터리**(τ² 전·Risk A·§7.2).
5. **[잔여] 스텝-scaffold 설계**(얽힌 결정→typed 스텝 분해 순서)·**결합 알고리즘 명세**(어느 결합이 decidable·언제 consensus LoRA)·resolve_selection의 t2 도구 레지스트리 wiring.

## 10. 단계
1. 리뷰(이 문서·§5 분해 아키텍처·§5b·§8·§9) → 확정.
2. synth 스페셜리스트: `synth_to_nativefc.py`(신규·`synth_depth.py` 생성기서 native-FC 직생성·resolve_selection tool_calls·anchor_id 제외) → synth-native LoRA 학습 + resolve_selection 도구 스키마/구현. (SOP·TaskBench·CFB는 기존 LoRA 재사용.)
3. **스텝-scaffold + 결합 알고리즘**(결정론 우선) 구현 — facet 스페셜리스트 오케스트레이션. 스페셜리스트 sanity(§7.1).
4. **결합 decidable-비율 측정**(§7.2) → 잔여 결합만 consensus LoRA 학습(필요시).
5. e2e 3-way(§7.3) → autopsy → ABox-swap 전이 매트릭스(§7.4) → `M_A_RESULTS §28` 박제.

## 11. 공격 방어 체크리스트 (리뷰어 사전대응·CROSS_BENCH §5)
- [ ] 포맷매칭/resolve-tool에 per-domain 분기 0 (grep `if domain`).
- [ ] ABox-ablation: 빈/틀린 A1/A2 → 붕괴 실측.
- [ ] 동일 시스템 unchanged로 retail+airline 작동(per-bench 1회).
- [ ] 학습벤치(τ²·SOP-Bench) contamination 0.
- [ ] 보상 전부 결정론(LLM-judge 0).
