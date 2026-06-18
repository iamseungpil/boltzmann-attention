# 통합 TBox 설계 (2026-06-18) — 4벤치(SOPBench+TaskBench+CFB+Synth)를 한 native-FC TBox로

> 진입 = 메모리 `01-four-bench-tbox`·`04-current-position`. 불변 = `03-anti-drift`·`11-transfer-direction`·`10-roles-deterministic`. 상위 = `CROSS_BENCH_TRANSFER_PLAN_2026_06_14.md`(Option B native)·`THESIS_STATEMENT_2026_06_16.md §7`·`NATIVE_FC_CONVERTER_DESIGN_2026_06_14.md`. 사용자 지시 = "통합 TBox 설계하라"·"설계먼저→리뷰→구현".

## 0. 확인 — 통합 TBox는 *존재하지 않는다* (직접 검증, 2026-06-18)
- TaskBench native-FC ✓ (`scratch/tb_sft/*`·`fc_build/tbnfc_*` = `{tools, messages[tool_calls]}`).
- SOPBench native-FC ✓ (v-series가 사용·`sft_alias_run/lodo_train_*`).
- CFB native-FC ✓ (v7 = grounded 2-hop).
- **Synth = op-IR JSON만** (`route_sft.jsonl`: system "Output ONLY JSON" → `{"op":"filter",...}`). **native-FC 버전 없음.**
- **4벤치를 한 native 포맷으로 묶은 데이터·모델 = 없음.** v7 = SOPBench+TaskBench(+CFB), synth content-routing 미포함. ⇒ 통합 TBox = 신규 학습.

## 1. 목표 (thesis 헤드라인)
**한 native-FC LoRA를 4벤치 혼합으로 학습 = P1-P9 flow + content 생성원 전부 내재화한 도메인-일반 TBox.** 그 위에 결정론 offload + ABox → e2e. 헤드라인 = **협업(TBox+offload)이 base와 bare-TBox를 Pareto-지배 + ABox-swap 무재학습 전이(retail→airline·τ²→SOP-Bench)**.

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
```
resolve_selection(op: argmax|argmin|rank|filter|comparative|substitute|create,
                  attr: <name>, criteria: {...}, anchor_id?: <id>, set?: {...}) -> item(s)
```
- 모델은 NL→op를 **native tool_call**로 명명(`resolve_selection(op=argmax, attr=score, ...)`). = synth 라우팅을 *어휘만 다른 같은 학습*(§21 양성)으로 보존하되 포맷은 native.
- 도구 *구현* = 결정론 엔진(`tau2_op_resolver.resolve_op_tau2`·`synth_depth.resolve_operation`) = 깊은 실행 offload(B=∞). 모델은 *명명만*([[10-roles-deterministic]]).
- synth 학습데이터 = `route_sft.jsonl`을 native로 재변환: assistant 턴 = `tool_calls:[resolve_selection(...)]`, tool 턴 = 엔진 결과. (신규 변환기 `synth_to_nativefc.py`.)
- **이게 §23E를 푼다**: 모델은 native tool_call을 emit(op-IR 텍스트 아님) → 파서 인식·도구 실행 → collapse 없음.

### 3b. resolve-tool은 *real-bench action*과 분리 (offload 컴퓨테이션)
τ²/real에서 resolve_selection은 *real 도구를 대체하지 않음* — 모델이 real action tool(exchange_…)의 *인자(item_ids 등)를 계산*하려 호출하는 **제공된 컴퓨테이션 도구**(코드-인터프리터/계산기 동형). 벤치 성공기준(최종 DB 상태) 불변. = 치팅 아님(§8).

## 4. 결정론 실행 층 (offload·config-도출·학습 아님)
- **resolve_selection 구현** = content-op 엔진(§3a).
- **R1b provenance 검증기** = 인자값 ∈ {user, 직전 tool 출력}·날조/스키마-example 거부(CFB가 학습한 fetch-first의 결정론 가드). 기존 `t2_gate_patch` provenance 재사용(단 deny→재생성·일반).
- **gate** = G1-G4 정책 replay(SOPBench가 학습한 순서/confirm의 결정론 집행·GATE_SPEC).
- 전부 **ABox/스키마 도출·도메인-일반**(per-domain 분기0). [[10-roles-deterministic]]·[[11-transfer-direction]].

## 5. 학습
- base Qwen2.5-7B·LoRA(r16·native target_modules)·`lora_train_chat_toolcall.py`(loss=assistant-only).
- **혼합 데이터** = SOPBench-nfc ∪ TaskBench-nfc ∪ CFB-nfc ∪ **Synth-nfc(신규)**. 비율 = 균형(벤치별 동수·primitive 커버 우선)·표현 다양성 유지([[12-diversity-required]]·단일템플릿 금지).
- 1차 = ep1~3 sweep·in-dist 검증 후 held-out.
- ⚠️ **wide-train 교훈(§23D)**: binding(set 추출)을 모델에 욱여넣으면 라우팅 퇴행 → set/concrete는 offload(resolve-tool)로, 학습은 op-명명·순서·fetch-first·threading만.

## 6. ABox (per-domain swap·재학습 0)
- A1 = 도구 카탈로그(도메인 도구 + resolve_selection + getter) — 기계 생성.
- A1' = resolve_selection config(attr 타입 ordinal/categorical·vocabulary) — 카탈로그서 도출.
- A2 = 정책→GATE_SPEC(front-end 난제).
- **swap 시험**: retail↔airline(같은 시스템 unchanged)·τ²→SOP-Bench(Amazon). 빈/틀린 A1/A2 → 붕괴(ablation).

## 7. 측정
1. **synth(통제 최종 벤치) closure**: 통합 TBox가 7-op 라우팅 held-out 유지(§21 1.00 회귀 없나)·native 포맷서.
2. **τ² e2e 3-way**: base(0.17 floor) / **TBox-bare**(라우팅O·offload 없이) / **TBox+offload**(resolve-tool+provenance+gate). 협업 > 둘 다 = 헤드라인. autopsy: collapse(order_id 날조) 소멸?
3. **전이 매트릭스**: 같은 통합 TBox+offload, ABox만 swap → retail·airline·(SOP-Bench) 동시 작동. 재학습0.
4. 보상 = 결정론(τ² DB∧NL∧comm·compliant-pass gated).

## 8. ★자가심사 (리뷰 안건)
**(a) thesis-정합?** — ✓. 학습=도메인-일반(op-명명·순서·fetch-first·threading=4벤치 P-primitive)·offload=decidable(resolve/provenance/gate)·ABox=사실·전이=swap. [[00-thesis]] 그대로. base+bespoke(UNIFIED) 아님(학습 TBox가 에이전트).
**(b) tau2-치팅 공격 방어:**
- resolve_selection = **도메인-일반 decidable 도구**(ABox-파라미터·카탈로그서 config 도출), 손-코딩 retail 정답 아님. **전이(retail→airline 같은 도구·§19 오라클 32/32+27/27)가 일반성 증명.**
- real action 도구 대체 아님 = 인자 *계산* 보조(계산기 동형)·벤치 기준 불변(§3b).
- 라우팅은 **학습**(§21 전이 양성)·scaffold가 op 고르지 않음. provenance/gate = per-domain 분기0(grep `if domain`).
- contamination 0 = τ²·SOP-Bench는 학습에서 held-out.

## 9. 열린 결정 (리뷰서 확정)
1. **synth-nfc 변환**: resolve_selection 인터페이스 최종 스키마(op 인자 집합)·synth 추상 catalog를 도구 스키마로 어떻게 렌더.
2. **혼합 비율** + base를 공유로 한 단일 LoRA vs 역할-파이프라인(멀티-LoRA). (단일 권장 = 한 모델 헤드라인.)
3. **resolve-tool을 τ²에 노출 vs 내부 인터셉트**: §3b(노출·계산기) 채택 — 단 리뷰 확인.
4. 측정 우선순위: synth closure 먼저(통제·싸다) → τ² 전이.

## 10. 단계
1. 리뷰(이 문서·§8·§9) → 확정.
2. `synth_to_nativefc.py`(신규) + resolve_selection 스키마·구현. 기존 native-FC 3벤치 데이터 정리.
3. 혼합 학습(GPU) → in-dist + synth held-out 검증.
4. e2e 측정(§7) → autopsy → 전이 매트릭스 → `M_A_RESULTS §28` 박제.

## 11. 공격 방어 체크리스트 (리뷰어 사전대응·CROSS_BENCH §5)
- [ ] 포맷매칭/resolve-tool에 per-domain 분기 0 (grep `if domain`).
- [ ] ABox-ablation: 빈/틀린 A1/A2 → 붕괴 실측.
- [ ] 동일 시스템 unchanged로 retail+airline 작동(per-bench 1회).
- [ ] 학습벤치(τ²·SOP-Bench) contamination 0.
- [ ] 보상 전부 결정론(LLM-judge 0).
