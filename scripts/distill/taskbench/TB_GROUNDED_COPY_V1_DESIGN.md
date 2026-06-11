# grounded-copy v1 — 추론-시 도구명 제약 디코딩 (설계 노트, 2026-06-11)

> 배경: 결과문서 §9.5 — v0(name-snap 후처리)로 RFT2+snap=52.5(held-out 첫 base 추월).
> v0 한계 실측: daily 미스냅 689건 = 의미적 패러프레이즈("install software"→`software_management`) = 문자열 매칭 사정거리 밖.
> v1 = 생성 자체를 valid 도구명 집합 안으로 제약(의미 매칭은 모델이 수행).

## 1. 리모트 환경 사실 (2026-06-11 확인)
- vllm **0.11.0** (`tau2_vllm_env`). 구조화 디코딩 = **per-request, 서버 플래그 불요**(기본 backend=auto→xgrammar).
- 요청 문법 (0.11.0 권장, 0.12 호환):
  `extra_body={"structured_outputs": {"json": <schema>}}`
  (`guided_json`은 0.11에서 deprecated-동작·0.12에서 제거 — 사용 금지. `response_format={"type":"json_schema",...}`도 동작=가장 버전-안정.)
- xgrammar는 `enum`(수백 문자열)·`$defs`/`$ref`·중첩 배열 지원. `minItems`/`pattern`류만 회피.
- LoRA 호환: logits bitmask는 forward 후 적용 — `--lora-modules` 서빙과 직교, model=LoRA명 + extra_body 동시 사용 가능.
- 비용: 스키마당 1회 grammar 컴파일(수백 enum이면 초 단위) 후 캐시 — **도메인당 더미 요청 1회 pre-warm** 필수. batch≥8 동시 제약 요청에서 CPU bitmask 병목 보고 있음(multiworker 8 사용 중 → 측정 시 throughput 주시).

## 2. 스키마 (도메인별 1개, tool_desc.json에서 생성)
```json
{
  "type": "object",
  "properties": {
    "task_steps": {"type": "array", "items": {"type": "string"}},
    "task_nodes": {"type": "array", "items": {
      "type": "object",
      "properties": {"task": {"$ref": "#/$defs/ToolName"}, "arguments": {"type": "array"}},
      "required": ["task"]}},
    "task_links": {"type": "array", "items": {
      "type": "object",
      "properties": {"source": {"type": "string"}, "target": {"type": "string"}}}}
  },
  "required": ["task_steps", "task_nodes", "task_links"],
  "$defs": {"ToolName": {"type": "string", "enum": ["<tool_desc.json 전체 이름>"]}}
}
```
- ⚠️ 구현 전 확인 2건: ①daily(temporal)의 실제 출력 키 구조(arguments dict형·task_links 의미 차이) — inference.py 프롬프트/파서 직독 ②`task_steps/task_nodes/task_links` 외 키 유무. 스키마는 실제 base 출력 몇 개로 검증 후 고정.
- 프롬프트는 그대로 유지(제약은 마스킹일 뿐 내용을 가르치지 않음 — Qwen 공식 권고).

## 3. 통합 지점
- TaskBench `inference.py`(리모트 `/home/woori/scratch/JARVIS_tb/taskbench`)의 OpenAI 호출부에 extra_body 주입. 플래그(예: env `TB_GUIDED=1`)로 on/off — A/B 동일 파이프 유지.
- 적용 대상: held-out full 추론(특히 daily). 비교군: 동일 어댑터 ±guided, 그리고 v0 snap 겹침(v1이 snap을 대체하는지/보완하는지).

## 4. 판정 (사전 등록)
- **주지표**: daily held-out edge-F1 — v0 무효(snap Δ0)였던 축. v1이 daily를 회복하면 "의미-변형은 제약-선택으로 해소" 입증.
- **부지표**: valid_frac→1.0(정의상), node-F1/recall, MM/HF에서 v0 snap과 동급 이상인지.
- **리스크**: 제약이 task_steps 자유텍스트 품질·조기종결(누락축)에 영향 주는지 census로 확인(제약은 누락을 못 고침 — 기대 Δ0, 악화만 감시).
- 성공 시: TaskBench = propose(weight) + gate(결정론 제약)의 2번째 실증 → thesis 패키지 헤드라인 갱신.

## 5. 출처 (조사 에이전트 2026-06-11)
- vLLM structured outputs 문서(v0.10.1/v0.11.0)·release notes(0.12 제거)·PR#22740(per-request backend deprecate)·issue#15762(xgrammar enum)·SqueezeBits 벤치(스키마-재사용 시 오버헤드 미미)·Qwen 문서(프롬프트 형식 설명 유지 권고). Qwen3 thinking-모드 이슈는 Qwen2.5 비해당.
