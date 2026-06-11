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

## 6.5 ★선행연구 대비 우리 차별점 (1:1 대조, 전부 실측 출처 표기 — 논문 related-work 원고 재료)

| 선행 | 그들의 주장/한계 | 우리의 차별 (증거) |
|---|---|---|
| **ToolDec** ('23) | "Don't Fine-Tune, Decode" = CD가 FT를 **대체**. frozen 모델·단일 도구호출 벤치(ToolEval 등)·문법오류→0. FT와의 결합·분해 없음 | **대체가 아니라 분업**: 같은 모델·같은 벤치 2×2 — base+snap **+0.4뿐**(§9.5 통제; base+guided 미실측이나 base parse-드롭 12/5584뿐이라 동급 예상 — 2×2 완결하려면 1런 추가) ↔ FT(rft2+dpo2)+guided **+7.2**(§9.6·9.5b). CD는 어휘/문법축만 풀고(L1), 누락·정책축(L5)은 on-policy DPO만 풀었음(§9.6 v2). **합성이 양쪽 단독을 모두 추월** = ToolDec 프레이밍의 직접 반증 데이터 |
| **ToolDec/FANTASE 공통** | 단일 호출/API-시퀀스 선택. **왜 무효명이 생기는지** 기제 없음 | **무효명의 인과 진단**: base는 valid 0.98-1.0(베끼기 완벽) → **SFT가 간섭을 주입**(0.987→0.946, §8) → CD는 "모델 결함 보정"이 아니라 **학습 부작용의 회복장치**라는 기제 확정(32B census로 외삽 검증 중, §8.5). 적용 대상도 멀티노드 플랜 그래프(edge-F1) |
| **ToolGen** (ICLR'25) | 도구=단일 토큰 추가 = **무거운 FT 필수**(47K 어휘 확장 재학습)·새 도구마다 재학습·FT/CD 기여 미분해 | **제약층은 학습 0·도구셋 교체=스키마 교체뿐**(per-request, §1) — 재학습0 전이 thesis와 정합. 기여 분해는 census로 정량(§9.5b 귀속 ①valid ②parse ③macro 분리) |
| **GENRE/PICARD/Synchromesh/Geng** ('21-23) | 엔티티/SQL/코드 도메인의 기제 확립. FT 상호작용·에이전트 플랜 무관 | 기제는 그대로 계승(인용) — 우리는 **에이전트 플랜 벤치 첫 수치 + FT 상호작용**이 기여 |
| **GAD** (NeurIPS'24, 이론) | per-step 마스킹=분포 왜곡(greedy trie-커밋) 경고. 실증 빈약 | **왜곡 비용의 실측**: worsened 77/3647=**2.1%**뿐, 순이득 +8.0(§9.5b) — name-slot enum처럼 짧고 의미-구분되는 제약에선 왜곡≪이득. 역으로 **제약이 이기는 조건**도 실측: 의미-패러프레이즈(snap 0)는 모델-내 의미매칭에 위임해야 풀림(v0/v1 경계, §9.5) |
| **Tam et al.** (EMNLP'24, "제약이 추론 저해") | 형식 제약이 성능 저하 주장 (dottxt 반박과 논쟁 중) | **도메인별 조건 분리로 논쟁에 데이터 제공**: 자유텍스트 필드는 비제약(이름 슬롯만 마스킹)·P/R 동반 상승(§9.5b)=서술 품질 무손상 — "무엇을 제약하느냐"가 변수임을 실측 |

**프레임워크-수준 차별 (CD를 넘어, 어느 선행도 없는 것):**
1. **층위 분업 법칙** (결과문서 §10): CD는 L1(심볼)의 도구일 뿐 — L2 gather·L3 게이트 offload(SOPBench 15→29)·L4 하이브리드·L5 양방향-DPO까지 **층위별 승자 지도 + 각 층 실측**. 선행들은 전부 단일 층의 단일 도구.
2. **census→처방 절차**: base 결핍 측정→레버 선택을 *학습 전 예측*으로 동결(32B Δ=−5 사전등록, §8.5) — 모델·크기 불문 적용되는 배포 의사결정 도구. 선행은 모델별 일회성 결과.
3. **정책축의 학습 조건 발견**: 종결-캘리브레이션은 weight-학습 가능하되 **신호 양방향 필수**(v1 단방향 net−15 ↔ v2 양방향 신기록, §9.6) — CD 문헌이 다루지 않는 직교축.
4. **2-벤치 일관 실증**: 같은 분업 구조가 플랜-예측(TaskBench: guided/snap)과 절차-실행(SOPBench: offload/DGGATE)에서 독립 재현 — 단일 벤치 결과가 아니라 구조적 주장.
