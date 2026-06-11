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

## 6. ★선행연구 (litreview 2026-06-11, 적대검증 — 메커니즘 novelty 주장 금지)
**기제 자체는 확립된 계보 — 절대 novelty 주장 금지:**
- **GENRE** (De Cao et al., ICLR'21 spotlight, 2010.00904): 고정 이름 집합(위키 엔티티 6M)을 prefix-trie 제약 빔서치로 생성 — "유효 이름만 생성"의 정전.
- **PICARD** (EMNLP'21)·**Synchromesh** (ICLR'22)·**Geng et al. GCD** (EMNLP'23): 점진 파서/completion-engine/입력-종속 문법 제약 디코딩 — 우리 per-domain enum은 Geng의 input-dependent grammar 인스턴스.
- **★ToolDec** (2310.07075, NeurIPS'23 MATH-AI 워크숍 — 본회의 아님): **도구명 토큰-trie + 인자 FSM 마스킹, 가장 근접 선행**. v2 제목이 "Don't Fine-Tune, Decode" = FT-적대 프레이밍. **FANTASE** (Findings EMNLP'24, 2407.13945): API명 trie+reranker. **ToolGen** (ICLR'25, 2410.03439): 도구=단일 토큰화+제약 빔서치(임베딩-레벨 변형).
- 인프라 상품화: Outlines(2307.09702)·xgrammar(2411.15100, vLLM 기본 백엔드=우리 스택)·OpenAI Structured Outputs(2024-08, adherence 93%→100%)·GBNF/Guidance.
- 단점 이론: **Grammar-Aligned Decoding** (NeurIPS'24, 2405.21047) — per-step 마스킹은 LM 분포를 왜곡(greedy trie-커밋) ⇒ 우리 worsened 77/3647이 그 비용의 실측(작음). Tam et al.(EMNLP'24 Ind., 제약이 추론 저해) vs dottxt 반박 — 논쟁 live.
**미점유 영역 (novelty는 여기) — 검증 결과 선행 0건:**
1. **TaskBench/도구-그래프 플랜 생성에 CD 적용 수치** (TaskBench는 측정만, DiG-Plan은 diffusion).
2. **★FT×CD 요인분해 (2×2: base/FT × free/constrained) + census 귀속** — ToolDec=적대 프레이밍만, ToolGen=결합하되 미분해. 우리 §10(레버 장부: DPO v2 55.95 + guided 57.22, 각 축 귀속)이 정확히 이 빈칸. "Don't Fine-Tune, Decode"에 대한 답 = "둘은 다른 층을 푼다(합성이 최선)".
3. **제약이 해/독이 되는 조건의 실증** (GAD 이론 ↔ 우리 daily +8.0·worsened 77 실측).
