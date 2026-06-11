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
- **PICARD** (EMNLP'21)·**Synchromesh** (ICLR'22)·**Geng et al. GCD** (EMNLP'23): 점진 파싱(스키마 식별자 제약은 **lexing 모드**)/completion-engine(Brzozowski 도함수)/입력-종속 문법. 우리 enum은 Geng의 input-dependent grammar 인스턴스 — 단 **per-request 독법으로**("요청과 함께 도착하는 도구 카탈로그가 문법을 유도"; per-domain 고정 문법으로 쓰면 stretch).
- **★ToolDec** (2310.07075, **인용은 v3** 2024-06-04; NeurIPS'23 MATH-AI 워크숍 포스터 — 본회의 아님·ICLR'24 불채택 확인): **도구명 토큰-trie + 인자 FSM 마스킹, 가장 근접 선행**. ⚠️주의 2건: ①제목은 "Don't Fine-Tune, Decode"지만 **본문은 FT 모델 위에도 얹음**(Table 1 ToolLLM/ToolkenGPT ±ToolDec, 최대 +21pt; §5 "complementary to existing approaches" 자인) — "그들=대체 프레이밍만" 주장 **금지** ②"names-only generalization" 인용은 **v1 한정**(v2/v3에서 삭제) — 버전 명시 필수. 범위는 단일/순차 호출(플랜-그래프 無)·제약은 OpenAPI **문법 수준**(이름+인자 타입). **FANTASE** (Findings EMNLP'24, 2407.13945): **CTST** trie+경량 reranker(RoBERTa), DSTC8/API-Bank — ⚠️**SFT 모델 ±SCD도 실측함**(Table 3, +12.7~17 최대 이득) — 단 ICL군=13B vs SFT군=7B로 base 교란·상호작용 분석 0. **ToolGen** (ICLR'25, 2410.03439): 도구=단일 토큰(47K 어휘 확장)+3단계 학습+제약 빔서치 — ⚠️"−constraining" ablation 있음(retrieval ≈0효과)+환각 ±제약 비교(7%→0%); 없는 것=end-to-end 과제성능의 제약-기여 분해·no-FT+제약 셀(설계상 정의 불가: 제약 집합=학습된 어휘). "신규 도구=재학습"은 우리 추론(그들 Appendix F가 unseen 일반화 열세 자인).
- 인프라 상품화: Outlines(2307.09702)·xgrammar(2411.15100, vLLM 기본 백엔드=우리 스택)·OpenAI Structured Outputs(2024-08: gpt-4-0613 <40% → gpt-4o 일반훈련 93% → +CD **100%**)·GBNF/Guidance.
- 단점 이론: **Grammar-Aligned Decoding** (NeurIPS'24, 2405.21047) — per-step 마스킹·국소 재정규화가 LM 분포 왜곡(기대-미래-문법성 무시; **샘플링-분포 차원** 주장 — "greedy 커밋"으로 쓰면 과장). "공유-prefix 이름 오선택" 인스턴스화는 **우리 따름정리**(그들 예시는 이항 문자열·이름/식별자 無). 그들 실험=SyGuS 36문제+파싱·Mistral-7B 단일·KL만 — task-수준 왜곡비용 측정 없음 + ASAp조차 수렴 느림·정확도 혼조 자인 ⇒ 우리 worsened 77/3647(2.1%) 실측의 빈칸 확정. Tam et al.(EMNLP'24 Ind., 제약 strictness↑=추론 성능↓ — 단 JSON-mode 붕괴 일부는 key-순서 아티팩트·블랙박스 flag) vs dottxt 반박(프롬프트 교란 지적) — 논쟁 live. ⚠️선제 인용: Tam §4.1에 "분류과제(DDXPlus 49진단)는 JSON-mode 답-공간 제약이 성능 ↑" 관찰 있음 — 프롬프트-수준·통제분리 없음이지만 인접 신호로 미리 인용할 것.
**미점유 영역 (novelty는 여기) — 5-에이전트 전문 적대검증 후 경화된 형태 (2026-06-11):**
1. **TaskBench 표준 프로토콜(n-F1/e-F1)에 inference-time·training-free CD 첫 수치** — 좁은 형태로 SAFE(인용 그래프 ~180편 전수+서베이 확인). 근접 작업과의 구분 명시 필수: GRAFT(2605.11706)=**학습된** tool-token 어휘(마스킹 아님·비표준 지표)·ToolGen=ToolBench·ToolDec=순차 단일호출·DiG-Plan=diffusion·GNN4Plan/GNNVerifier=모델-측 GNN. "도구-그래프 플랜 생성 전반 최초" 같은 넓은 표현 금지.
2. **★same-base-model 통제 2×2 + 상호작용·기제 귀속** — FT+CD *결합 자체*는 선행에 있음(ToolDec Table 1 stacking·FANTASE Table 3 SFT±SCD·ToolGen 부분 ablation) ⇒ 차별점은 결합의 존재가 아니라: ①**동일 base·동일 데이터 통제 factorial**(선행은 전부 ±FT 축에서 base 모델이 다름 — ToolDec: Mistral vs LLaMA-FT·FANTASE: 13B vs 7B) ②**이득의 census 귀속**(어느 오류축을 누가 풀었나: CD=어휘/문법, on-policy DPO=누락/정책) ③**인과 기제**(무효명=모델 결함이 아니라 SFT-주입 간섭이라는 발생론 — 선행 전부 부재). ToolDec §5의 "complementary" *관찰*을 우리가 *정량 분해*로 완성하는 구도.
3. **제약 득실 조건의 task-수준 실증** (GAD=KL·소규모만 ↔ 우리 daily +8.0·worsened 2.1%·P/R 동반상승 실측) + 제약 **층위**의 차이(선행=문법/이름; 우리는 SOPBench 정책-게이트까지 의미 수준 — 동일 분업 구조의 2-벤치 재현).

## 6.5 ★선행연구 대비 우리 차별점 (1:1 대조, 전부 실측 출처 표기 — 논문 related-work 원고 재료)

| 선행 | 그들의 주장/한계 | 우리의 차별 (증거) |
|---|---|---|
| **ToolDec** (v3 '24) | 제목은 "Don't Fine-Tune, Decode"지만 ⚠️**FT 모델 위 stacking도 실측**(Table 1 ToolLLM/ToolkenGPT ±ToolDec +21pt·§5 "complementary" 자인). 단 ±FT 축의 base가 다름(Mistral vs LLaMA-FT)=비통제·상호작용 분석 0·문법-수준 제약·단일/순차 호출 | **그들의 "complementary" 관찰을 통제 분해로 완성**: 동일 base(Qwen2.5-7B)·동일 데이터에서 2×2 — base+snap **+0.4**(§9.5 통제; base+guided 미실측, parse-드롭 12/5584뿐이라 동급 예상 — 완결용 1런 잔여) ↔ FT(rft2+dpo2)+guided **+7.2**(§9.6·9.5b). **CD가 푼 축(L1 어휘/문법)과 on-policy DPO가 푼 축(L5 누락/정책)을 census로 분리 귀속** — 선행 어디에도 없는 건 결합이 아니라 이 귀속 |
| **ToolDec/FANTASE 공통** | FANTASE도 ⚠️SFT±SCD 실측(Table 3, +12.7~17 최대이득) — 단 ICL=13B vs SFT=7B 교란·상호작용 분석 0. 양쪽 모두 **왜 무효명이 생기는지** 기제 없음(FANTASE는 오류 분류만: 환각 14% 등) | **무효명의 인과 진단**: base는 valid 0.98-1.0(베끼기 완벽) → **SFT가 간섭을 주입**(0.987→0.946, §8) → CD는 "모델 결함 보정"이 아니라 **학습 부작용의 회복장치**라는 발생론 확정(32B census로 외삽 검증 중, §8.5). 적용 대상도 의존-링크 멀티노드 플랜 그래프(edge-F1) — 선행은 단일/순차 호출 |
| **ToolGen** (ICLR'25) | 도구=단일 토큰(47K 어휘 확장)+3단계 학습 필수. ⚠️부분 ablation 있음("−constraining" retrieval ≈0·환각 ±제약 7%→0%) — 없는 것: end-to-end 과제성능의 제약-기여 분해·no-FT+제약 셀(설계상 정의 불가: 제약 집합=학습된 어휘 자체). 신규 도구 재학습=우리 추론(App.F unseen 열세 자인) | **제약층은 학습 0·도구셋 교체=스키마 교체뿐**(per-request, §1) — 재학습0 전이 thesis와 정합·no-FT+제약 셀이 우리 설계에선 자연 존재. end-to-end 기여 분해는 census로 정량(§9.5b 귀속 ①valid ②parse ③macro 분리) |
| **GENRE/PICARD/Synchromesh/Geng** ('21-23) | 엔티티/SQL/코드 도메인의 기제 확립(PICARD 스키마-식별자=lexing 모드). FT 상호작용·에이전트 플랜 무관 | 기제는 그대로 계승(인용) — 우리는 **TaskBench 표준 프로토콜 첫 CD 수치 + 통제 FT 상호작용**이 기여. Geng input-dependent 인용은 per-request 독법으로("요청과 함께 오는 도구 카탈로그가 문법 유도") |
| **GAD** (NeurIPS'24, 이론) | per-step 마스킹·국소 재정규화=분포 왜곡 증명(**샘플링-분포 차원** — "greedy 커밋" 표현 금지). 실험=SyGuS 36문제·Mistral-7B·KL만, task-수준 비용 없음·ASAp도 수렴 느림 자인 | **왜곡 비용의 task-수준 실측**: worsened 77/3647=**2.1%**뿐, 순이득 +8.0(§9.5b) — name-slot enum처럼 짧고 의미-구분되는 제약에선 왜곡≪이득. "공유-prefix 이름 오선택" 인스턴스화는 우리 따름정리로 명시(그들 예시는 이항 문자열). 역으로 **제약이 이기는 조건**도 실측: 의미-패러프레이즈(snap 0)는 모델-내 의미매칭 위임으로 풀림(v0/v1 경계, §9.5) |
| **Tam et al.** (EMNLP'24, "제약이 추론 저해") | 형식(구조) 제약만 — slot-only 마스킹 없음. JSON-mode 붕괴 일부=key-순서 아티팩트·블랙박스 flag. ⚠️§4.1에 "분류과제(DDXPlus)는 답-공간 제약이 성능↑" 인접 관찰 있음(프롬프트-수준·비통제) — **선제 인용할 것** | **slot-수준 조건 분리로 논쟁에 데이터 제공**: 자유텍스트 필드는 비제약(이름 슬롯만 마스킹)·P/R 동반 상승(§9.5b)=서술 품질 무손상 — "무엇을 제약하느냐"가 변수임을 통제 실측. DDXPlus 관찰은 suggestive prior signal로 인용 후 "engineered slot-only+통제분리는 우리가 처음" |

**프레임워크-수준 차별 (CD를 넘어, 어느 선행도 없는 것):**
1. **층위 분업 법칙** (결과문서 §10): CD는 L1(심볼)의 도구일 뿐 — L2 gather·L3 게이트 offload(SOPBench 15→29)·L4 하이브리드·L5 양방향-DPO까지 **층위별 승자 지도 + 각 층 실측**. 선행들은 전부 단일 층의 단일 도구.
2. **census→처방 절차**: base 결핍 측정→레버 선택을 *학습 전 예측*으로 동결(32B Δ=−5 사전등록, §8.5) — 모델·크기 불문 적용되는 배포 의사결정 도구. 선행은 모델별 일회성 결과.
3. **정책축의 학습 조건 발견**: 종결-캘리브레이션은 weight-학습 가능하되 **신호 양방향 필수**(v1 단방향 net−15 ↔ v2 양방향 신기록, §9.6) — CD 문헌이 다루지 않는 직교축.
4. **2-벤치 일관 실증**: 같은 분업 구조가 플랜-예측(TaskBench: guided/snap)과 절차-실행(SOPBench: offload/DGGATE)에서 독립 재현 — 단일 벤치 결과가 아니라 구조적 주장.
