# 설계서: SOPBench/TaskBench → native function-calling 궤적 변환기 (2026-06-14, 리뷰용)

> 상위 = `CROSS_BENCH_TRANSFER_PLAN_2026_06_14.md` §4.1. 불변 = memory `feedback-thesis-tbox-transfer-direction`·`feedback-selector-verifier-deterministic`. **구현 전 리뷰 대기.**

## 0. 목표
SOPBench(op-graph)·TaskBench(tool-graph) 학습데이터를 **단일 native OpenAI function-calling 궤적 포맷**으로 변환 → 하나의 TBox(공유 base + LoRA)가 R-규율(R1/R2/R4)을 *벤치-불변·전이가능*하게 학습. 추론 = vLLM config(`--tool-call-parser`·`guided_json`/XGrammar). **표현 발명 없음** = 표준 function-calling 채택.

## 1. 입력 포맷 (실측 2026-06-14)
**TaskBench** (`data_*/data.json`, 단발 DAG):
```
{instruction, tool_nodes:[{task:"play_movie_by_title", arguments:[{name:"title",value:"Inception"}]}],
 tool_links:[{source,target}], tool_steps}  +  tool_desc.json=[{id,desc,parameters:[{name,type,desc}]}]
```
- 인자값 **grounded**(instruction서 추출). 단발·tool 결과 없음(오프라인 plan).

**SOPBench t1c** (`sft_alias_run/lodo_train_t1c.jsonl`, 단계별 plan):
```
{domain, goal, target_kind, messages:[{user: "Pick SINGLE next tool... USER REQUEST:... POLICY:... TOOLS(op_X: needs[..]; gives[..])", assistant:"ready=false; op_40"}]}
```
- op **별칭**(op_40)·target=다음 1 op·인자 없음. 인자는 user request서 getter-map 해소(`offload_log`: `argmismatch`/`args_unresolvable` 추적). 멀티스텝=op-graph deps 순서.

## 2. 목표 출력 포맷 (native function-calling 궤적)
```jsonc
{ "tools": [ {"type":"function","function":{"name":<alias>,"description":<desc>,"parameters":<JSON schema(A1/A5)>}} ],
  "messages": [
    {"role":"system","content":<role + POLICY(A2-유도 제약)>},        // ABox·마스킹
    {"role":"user","content":<instruction>},                          // ABox·마스킹
    {"role":"assistant","tool_calls":[{"name":<alias>,"arguments":{...}}]}, // ★supervise
    {"role":"tool","content":<결과>},                                  // 마스킹
    ... (gather 반복) ...,
    {"role":"assistant","tool_calls":[{"name":<goal alias>,"arguments":{...}}]}, // ★supervise (ACT)
    {"role":"assistant","content":"<done>"}                            // ★supervise (R3 종료 신호 — 단 결정=게이트)
  ] }
```
- **loss = assistant-only**(labels −100 except assistant turn = tool_calls·done). TBox/ABox 분리 강제기둥#1.

## 3. 변환 로직 (벤치별)
### 3a. TaskBench → native
1. `tool_links` 위상정렬 → 노드 순서.
2. 각 노드 → assistant `tool_calls:[{name:alias(task), arguments: {n:v from tool_nodes}}]` + 합성 tool 결과(`"ok"` 또는 인자 echo).
3. system = generic role + (TaskBench는 정책 없음 → 최소 role). tools = tool_desc → function schema.
4. **가르치는 것**: R1(인자 grounded copy)·R4(도구 선택)·R6(구조/deps). **R2 약함**(합성 결과).

### 3b. SOPBench → native
1. op-graph deps(`needs[..]`) 위상정렬 = gather→goal 순서. (단계별 t1c를 *replay 체인*으로 결합.)
2. 각 op → assistant `tool_calls:[{name:op_X, arguments:{getter-map 해소값}}]` + 합성 결과(env true/false/값).
3. goal op = ACT. 성공 후 `done`.
4. system = SOPBench POLICY(역할·접근통제 — A2-유도). tools = op 카탈로그 → function schema(needs는 schema 밖, system/암묵).
5. **가르치는 것**: R2(gather 선행 순서)·R4. 인자 = R1.
- ⚠️**인자 해소 가능분만 변환**(`argmismatch=0 ∧ args_unresolvable 없음`) = 청정 supervise; 나머지 제외(정직 분모).

## 4. 핵심 설계 결정 (리뷰 포인트)
- **D1 별칭 유지**: 두 벤치 도구명 *별칭화*(lexical 암기 차단·NL설명↔도구 의미매칭 강제). function tools= 필드가 alias+description 제공 → 모델은 description으로 매칭. alias on/off = ablation(헤드라인=alias). **통합 alias 네임스페이스**(벤치 간 충돌 방지: `sop_*`/`tb_*` prefix 또는 전역 재번호).
- **D2 SOPBench 멀티턴 = op-graph replay**: 단계별 t1c를 deps 순서로 체인. 합성 결과는 env 시맨틱(true/false/값).
- **D3 TaskBench 순차화**: DAG를 *순차* tool_call 턴(병렬 multi-call 아님)으로 = SOPBench와 포맷 통일. 합성 결과 삽입.
- **D4 인자 = R1 supervise 타깃**: TaskBench=tool_nodes값 / SOPBench=getter-map 해소값. *컨텍스트(요청·이전결과)서 복사*가 학습 신호.
- **D5 loss=assistant-only**: system/user/tool = −100.
- **D6 ABox 제공**: tools(A1)=function 필드 / policy(A2)=system. 둘 다 마스킹·추론시 swap.
- **D7 QC**: ①스키마 로드 유효 ②R-규율 보존(gather 순서·인자 grounded) ③round-trip 샘플 감사 ④loss-mask 정합.

## 5. 통합 (단일 학습셋)
두 변환 출력이 *동일* native 포맷 → 단순 concat → 단일 SFT. 도메인-mix LODO(강제기둥#3). 비율·alias 네임스페이스 통제.

## 6. 미해결/리스크 (정직)
1. **대화(ask-user) 부재**: 두 벤치 다 멀티턴 user-sim 아님 → τ²의 ask-user는 **base 모델 폴백** 또는 후속 합성 augment. *이 변환기 범위 밖* 명시.
2. **TaskBench 합성 결과**: "결과 무의미" 학습 위험 → 결과를 plausible하게/마킹. R2 본체는 SOPBench가 운반.
3. **SOPBench 인자 커버리지**: 해소가능분만 → 학습셋 축소 가능(D7 필터). 규모 측정 필요.
4. **alias 충돌·스키마 추출**: SOPBench op 카탈로그→JSON schema 변환(parameters 타입) 정합 필요.
5. **R3 종료**: `done` 신호는 supervise하되 *결정*은 추론시 결정론 게이트(R3 불변) — 학습은 종료 *형식*만.

## 7. 단계 구현 (리뷰 후)
- **P1**: TaskBench→native 변환기(`fc_convert_taskbench.py`) — DAG 순차화·인자·tool_desc→schema. 샘플 5 검증.
- **P2**: SOPBench→native 변환기(`fc_convert_sopbench.py`) — op-graph replay·getter-map 인자·필터. 샘플 5 검증.
- **P3**: 통합·alias 네임스페이스·loss-mask 빌더(`fc_build_sft.py`) → 단일 JSONL. QC(§4 D7).
- **P4**: TBox LoRA 학습(Qwen2.5-7B) → held-in eval(native tool-call 정확도·R-규율 census). *전이 테스트는 별도 단계.*

## 8. 산출물
- 변환기 3종 + 통합 SFT JSONL + QC 리포트 + 샘플 궤적 10(감사용).
- 검증 기준: 스키마 100% 유효 · gather-순서 보존율 · 인자 grounded율 · loss-mask 정합 100%.
