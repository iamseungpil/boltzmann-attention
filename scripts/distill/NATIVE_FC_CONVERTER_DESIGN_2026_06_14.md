# 설계서 v2: SOPBench/TaskBench → native function-calling 궤적 변환기 (2026-06-14, 리뷰 반영)

> 상위 = `CROSS_BENCH_TRANSFER_PLAN_2026_06_14.md` §4.1. 불변 = memory `feedback-thesis-tbox-transfer-direction`·`feedback-selector-verifier-deterministic`.
> **v2 변경(리뷰 반영)**: SOPBench 소스 t1c→**FC 성공 rollout**(P0 확정)·D1 전역재번호·D3 병렬허용·R1 하위기술 분리·리스크#1/#2 본문승격·#5 해소.

## 0. 목표
SOPBench(FC rollout)·TaskBench(tool-graph)를 **단일 native OpenAI function-calling 궤적**으로 변환 → 단일 TBox가 R-규율(R1/R2/R4)을 벤치-불변·전이가능하게 학습. 추론 = vLLM config(`--tool-call-parser`·`guided_json`/XGrammar). 표현 발명 없음.

## 1. 입력 포맷 (실측 2026-06-14)
**TaskBench** (`data_*/data.json`, 단발 DAG): `{instruction, tool_nodes:[{task, arguments:[{name,value}]}], tool_links:[{source,target}]}` + `tool_desc.json`=A1. 인자값 grounded(instruction서 verbatim). **결과 없음**(오프라인 plan).

**SOPBench FC rollout** (★P0 확정 소스 — t1c 폐기) `SOPBench/output/<domain>/ast_<teacher>-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json`:
- 리스트[task], 각 task = `{domain, setup, task, interactions:[{prompt, interaction, database}], evaluations:[{success,...}]}`.
- `interaction` = **실제 native FC 메시지 시퀀스**: assistant text + `tool_calls:[{name, arguments:{실제값}}]` + tool 결과(`(True,False)` 등) + 종료 도구 `exit_conversation`.
- **`evaluations[*].success` 필터** → 검증된 궤적. **인벤토리 = 5599 성공 / 11274** (7도메인×15+교사, tool_full). should_T(행동)+should_F(거부=exit_conversation) 둘 다.
- ⇒ **op순서·인자·결과 전부 보존 → 합성 없이 1:1 사상**. R1 주장 = *실행된 인자*(진짜).

## 2. 목표 출력 포맷 (native function-calling 궤적)
```jsonc
{ "tools":[{"type":"function","function":{"name":<alias>,"description":<desc>,"parameters":<JSON schema(A1/A5)>}}],
  "messages":[
    {"role":"system","content":<role + POLICY(A2)>},                  // ABox·마스킹
    {"role":"user","content":<instruction>},                          // ABox·마스킹
    {"role":"assistant","content":<옵션 텍스트>,"tool_calls":[{"id,"type":"function","function":{"name":<alias>,"arguments":<json>}}]}, // ★supervise
    {"role":"tool","tool_call_id":..,"content":<결과>},               // 마스킹
    ... ,
    {"role":"assistant","tool_calls":[{...goal/exit_conversation...}]} // ★supervise (ACT/종료)
  ] }
```
- **loss = assistant-only**(labels −100 except assistant turn). TBox/ABox 분리 강제기둥#1.

## 3. 변환 로직
### 3a. SOPBench FC rollout → native (★1:1 변환, 합성 없음)
1. `success=True` 태스크만 (검증-distill). 교사 선택 = 강교사 우선(gpt-5·o4-mini-high·claude-3-7) + 도메인 다양성 우선(교사 다양성보다).
2. `interaction` 메시지를 표준 OpenAI 포맷으로 정규화(role 결측분 user/assistant/tool 추론·tool_call_id 부여).
3. 도구명 **전역-재번호 alias**(§4 D1). tools= = 도메인 tool 카탈로그→function schema(A1). system = task prompt/policy(A2).
4. **가르치는 것**: R2(실 gather→결과→act 순서·*실 결과* 보유)·R4·R1(실행된 인자). 종료 = `exit_conversation` 도구(리스크#5 해소).

### 3b. TaskBench → native (DAG, 병렬허용)
1. `tool_links`로 DAG 구성. **위상 *레벨*별**로: 같은 레벨(상호 비의존) 노드 = **한 assistant 턴에 복수 tool_calls**(native FC 병렬 = 진짜 R6). 선형 강제 안 함(D3).
2. 각 tool_call = {name:alias(task), arguments: tool_nodes값}. tool 결과 = **합성**(인자서 결정론 도출한 plausible 값·리스크#2).
3. **가르치는 것**: R1(verbatim copy)·R4·R6(병렬 구조). R2 약함(합성결과).

## 4. 핵심 결정 (v2)
- **D1 alias = 전역 재번호** (리뷰: `sop_*`/`tb_*` prefix **기각** — 벤치-식별 단서가 벤치-특이 행동 조건화 유발 = LODO 자기파괴). 두 벤치 도구를 **단일 전역 alias 공간**으로 재번호, 벤치 식별 단서 0. alias on/off = ablation.
- **D2 SOPBench = FC rollout 1:1 변환** (t1c 폐기). 합성·getter-map 불요.
- **D3 TaskBench = 병렬 허용** (리뷰: 강제 순차화 **기각** — native FC의 병렬 multi-call 이점 폐기·임의 위상순서 = 가짜 순서신호 주입·R6 선형화 모순). DAG 병렬분기 = 한 턴 복수 tool_calls.
- **D4 R1 하위기술 분리**: TaskBench = **verbatim-copy**(instruction서 인자 복사) / SOPBench = **executed-args**(rollout 실행값). 둘 다 R1이되 다른 하위기술로 귀속 기록. 해소불가/실패분 제외(success 필터가 자동 처리).
- **D5 loss=assistant-only**. **D6 ABox**: tools(A1)=function·policy(A2)=system, 마스킹·swap.
- **D7 QC**: ①스키마 로드 유효 ②R-규율 보존(gather순서·인자 grounded) ③round-trip 감사 ④loss-mask 정합 ⑤**결과-민감도 census**(결과 뒤집으면 다음 호출 바뀌나 = 합성결과 오염 실측).

## 5. 통합
두 출력 = 동일 native 포맷 → concat. **SOPBench:TaskBench 비율을 R2 보존이 측정되는 선에서 통제**(리스크#2). 전역 alias 공간·도메인-mix LODO.

## 6. 리스크 (본문 결정 승격)
- **★#1 ask-user 부재 = τ² 전이의 미검증 gate** (본문 승격): SOPBench FC rollout 다수가 정적-user(명료화 질문 없음). 멀티턴 명료화는 τ² 난이도 본체인데 동결 thesis가 base 능력에 둔 가정은 *어디서도 미검증*. **결정: 변환기 범위 밖으로 두되 — "τ² 전이는 이 미검증 가정에 gated"를 명시·전이 실패 시 #1 의심처로 등록.** (완화 후보: SOPBench `usr_adv-*`/`usr_gpt-4o-*` 멀티턴-user rollout 일부 합류로 대화 노출.)
- **★#2 TaskBench 합성결과 오염** (본문 승격·이제 TaskBench *전용* — SOPBench는 실결과): 통합셋에 무의미 결과가 섞이면 "tool 결과 무시" 전역 학습 → R2 오염. **완화: (a) 인자서 결정론 plausible 값 (b) SOPBench:TaskBench 비율 통제 (c) 결과-민감도 census(D7⑤)로 오염 실측.**
- **#3 커버리지/규모**: SOPBench=5599 성공(충분). TaskBench=깨끗. P0 잔여 = 도메인 균형·교사 선택 정책.
- **#4 스키마 추출** (P0 측정): SOPBench 도메인 tool 카탈로그→JSON schema(parameters 타입) 정합 확인. TaskBench tool_desc→schema는 깨끗.
- **#5 종료** (해소): SOPBench `exit_conversation` = 실제 도구 → 정상 tool_call로 supervise. 리터럴 done 불요·게이트 충돌 없음.

## 7. 단계 구현
- **P0 (완료/잔여)**: ✅SOPBench FC rollout 소스 확정(5599 성공). 잔여 = SOPBench tool카탈로그→schema 추출 정합(#4)·교사/도메인 균형 정책.
- **P1a TaskBench 변환기** (`fc_convert_taskbench.py`): DAG 레벨별 병렬(D3)·인자·tool_desc→schema·합성결과(결정론). 샘플 5 검증. **소스 깨끗 → 즉시 착수 가능.**
- **P1b SOPBench 변환기** (`fc_convert_sopbench.py`): success 필터·interaction 정규화·전역 alias·tool카탈로그→schema. 샘플 5 검증.
- **P2 통합·alias 전역재번호·loss-mask 빌더** (`fc_build_sft.py`) → 단일 JSONL + QC(§4 D7, 결과-민감도 census 포함).
- **P3 TBox LoRA 학습**(Qwen2.5-7B) → held-in eval. *전이 테스트(SOP-Bench·τ²)는 별도 단계.*

## 8. 산출물·검증기준
변환기 2종 + 통합 빌더 + SFT JSONL + QC 리포트 + 샘플 궤적 10. 기준: 스키마 100% 유효 · gather순서 보존율 · 인자 grounded율 · loss-mask 정합 100% · **결과-민감도(합성결과 오염 지표)**.
