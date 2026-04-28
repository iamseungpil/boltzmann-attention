# EXPERIMENT_PLAN_v28.1 — Facet/Ontology Framework for Tool-Use LLMs

**작성**: 2026-04-29 v28 / **개정**: 2026-04-29 v28.1 (E10 N=64 결과 반영)
**브랜치**: `iclr-prompt-internalization` (옵션 A 확정, §10)
**상태**: E10 완료 (commit 61fdb7e), E10b/E11/E13 prep 단계
**다음 목표 venue**: ICLR 2027 / TMLR / NeurIPS 2027 (3 paper triplet의 세 번째)

## 변경 이력
- **v28** (2026-04-29 초안): 초기 thesis, 정리 3·4 sketches, E10–E13 plan
- **v28.1** (2026-04-29 개정): E10 N=64 결과 반영, **모델 진단 contribution 추가** (§7.1 신설), **E10b list_anon 신설**, E11/E13 구체화, Risk R1/R3 부분 confirmed

---

## §0. 한 페이지 컨텍스트

### 0.1 세 paper triplet 위치

| paper | thesis 한 줄 | 관점 | 상태 |
|---|---|---|---|
| 자매 (`math/paper/iclr2027/PAPER_DRAFT_ICLR_v1.md`) | K-side rank-$r$ perturbation은 smooth attention shift를 만들지만 argmax는 안정 — *two-level gap*. | K-side, mechanism | 초안 v1 완성 |
| 본 v1.1 (`math/paper/iclr2027_prompt_internalization/PAPER_v1_ko.tex`) | Prefix-attention contribution은 함수공간 rank ≤ 2이며 정적 + Q-bias hybrid로 첫 토큰 98.4% 회복. multi-step F1=0 negative 동봉. | V-side, function-space | v1.1 push 완료 (e3b5a35) |
| **새 (이 plan)** | **자연어 시스템 prompt를 ontology facet 기반 formal 표현으로 대체하면 동등 또는 우수한 도구 선택 F1을 더 짧은 토큰으로 달성한다.** | **인풋 prompt 형식** | 본 plan |

### 0.2 새 paper의 differentiator (자매·본과의 비충돌)

- 자매 F1 결정: "B_ont는 K-subspace 추출 파이프라인, catalog *content*는 load-bearing 아님". 이 주장은 **K-subspace 측정 protocol**에 한정됨.
- 새 paper의 주장은 **모델에 들어가는 인풋 토큰 형식**에 관한 것 — 자연어 vs typed facet schema. 이는 자매 F1과 직교한다.
- 본 v1.1의 정리 1·2는 "정적 KV-bias 개입의 한계"를 다룸 (어떤 형태든 prompt 자체가 없는 체제). 새 paper는 "prompt 자체의 형식"을 다룸 (prompt가 있되 구조가 다름). 직교.

### 0.3 새 paper의 강력한 motivation 라인

- 자매 Phase C: "facet-value shuffle / tool-name shuffle / random-sentence" 모두 K-geometry attn_fro ratio 유지 → token-level *내용*은 K-geometry에 무관.
- 본 v1.1 E1 sanity: `real ≈ shuffled_prefix` — 도구 *순서*는 load-bearing 아님.
- → 두 paper 합쳐서 "자연어 prompt의 토큰 *내용·순서*는 attention-level에 매우 약하게 영향" 입증됨.
- **그렇다면 자연어 토큰 분량 자체가 비효율 — formal typed schema로 같은 정보를 더 짧게 인코딩 가능해야 함.** 이것이 새 paper의 thesis 출발점.

---

## §1. Thesis

> Tool-use 시 사용되는 자연어 시스템 prompt는 LLM의 attention output에 함수공간 rank ≤ 2의 contribution을 만든다 (본 v1.1 정리 1). 따라서 동일한 contribution을 **ontology facet 기반 formal 인코딩**으로 더 적은 토큰으로 달성 가능하며, 더 나아가 (a) multi-step generation의 schema-fragility 감소, (b) cross-domain transfer, (c) compositional generalization을 자연어 prompt 대비 우수하게 달성할 수 있다.

### Subclaim (v28.1 개정 — E10 결과 반영)
- **C1 (length-efficiency)**: facet prompt 길이 $\ell_F$가 NL prompt 길이 $\ell_{NL}$의 1/3–1/5에서 동등 F1. ⚠ **E10에서 부분 reject** — facet_compact (137 tok)가 두 모델 모두 nl_full (146 tok)보다 22-32% F1 낮음. *순수 typed annotation만으로는 description의 의미 정보를 대체하지 못함.* 갱신: facet_compact는 길이 절감 목적이 아닌 *speed-priority deployment* 옵션으로 reframe.
- **C2 (format-superiority for attention-conditioning models)**: length-matched 비교에서 facet prompt가 NL prompt 대비 F1 우수, **Qwen 한정** (E10에서 +9.7%). Llama는 -3.7% (NL 우위). → **모델 의존**.
- **C3 (cross-domain transfer)**: $\tau^2$-bench retail에서 학습/구성된 facet schema가 telecom/airline에 transfer (NL prompt는 도구 카탈로그 통째로 교체). [E11 미실시]
- **C4 (compositional)**: 처음 보는 도구 묶음(facet 결합)에서 NL prompt 대비 F1 격차 작음. [E12 optional]
- **C5 (NEW, 모델 진단 contribution)**: **`list_only/nl_full` F1 비율**이 facet 효과 예측에 사용 가능 (Qwen 0.285, Llama 0.729). 비율 < 0.4 → attention-conditioning native (facet 직접 prompt 효과적). 비율 > 0.7 → residual-stream native (다른 deployment mode 필요). E10에서 직접 입증.

---

## §2. Non-claims (scope creep 차단)

- 본 paper는 **새 training procedure 제안 안 함** — facet schema는 task documentation에서 결정론적/규칙적으로 추출.
- **steering benchmark race 안 함** — F1을 비교하지만 SOTA 비교 아님; 본 paper의 주장은 "*같은 정보의 다른 인코딩*에서 격차".
- **rank 이론 재증명 안 함** — 본 v1.1 정리 1·2 인용·확장.
- **자매 paper의 K-subspace 주장 재정의 안 함** — F1 결정문 인용만.
- **tool design 자동화 안 함** — facet schema는 사람이 정의(또는 task의 evaluation_criteria.actions에서 직접 추출).

---

## §3. 정리 후보

### 정리 3 (Facet-rank 일치, sketch)
$\mathcal{F} = \{f_1, \ldots, f_m\}$를 facet 집합, 각 facet $f_i$가 $|V(f_i)|$개 value를 가진다고 하자. task의 도구 카탈로그가 facet 결합 $\{(v_1, \ldots, v_m): v_i \in V(f_i)\}$의 부분집합으로 표현되면, 자연어 prompt $P_{NL}$의 함수공간 rank $r^*(P_{NL})$는 facet count $m$의 lower bound를 갖는다 — 즉 $r^*(P_{NL}) \ge \dim(\mathrm{span}(\{B_{f_1}, \ldots, B_{f_m}\}))$ where $B_{f_i}$는 facet $f_i$의 K-subspace.

**해석**: NL prompt가 사실상 인코딩하는 정보 차원의 lower bound가 facet decomposition으로 직접 측정 가능. 본 v1.1의 r* ≤ 2.25 측정과 일치하면 → MetaTool ST4의 facet count = 2-3개 정도라는 예측.

**증명 sketch**: 본 v1.1 정리 1의 함수공간 SVD를 facet decomposition에 적용; facet들이 K-space에서 (대부분) 직교라는 자매 paper의 K-subspace 결과와 결합.

### 정리 4 (Formal advantage, sketch)
NL prompt $P_{NL}$ ($\ell_{NL}$ 토큰)과 facet prompt $P_F$ ($\ell_F$ 토큰)이 동일 facet 정보를 인코딩하면, $\ell_F \le \ell_{NL}$이고 두 prompt의 attention output 차이는
\[
\|o_{NL} - o_F\|_2 \le C \cdot \mathrm{NL\text{-}redundancy}(P_{NL})
\]
여기서 $\mathrm{NL\text{-}redundancy}$는 NL token의 정보 ambiguity (synonyms, fillers, order variation).

**해석**: facet prompt는 redundancy = 0이므로 NL prompt 대비 attention output이 더 *예리*. multi-step generation에서 첫 step의 KL이 작아지면 trajectory 발산 가능성도 작아짐 (본 v1.1 §6.4의 trajectory 격차 분석과 결합).

**증명 sketch**: NL prompt를 facet 표현으로 deterministic mapping (예: tool-name → typed slot) 후 token entropy 분석. attention softmax의 Lipschitz 성질 + 정리 1 잔차 식.

### Optional 정리 5 (Compositional generalization)
facet schema가 결합 cardinality $|\mathcal{F}|$를 갖고 새 task가 facet 결합으로 표현 가능하면 (즉 unseen tool이 seen facet의 새 결합), facet prompt는 NL prompt 대비 generalization gap이 작다 — formally, $\mathrm{F1}_F(\text{unseen}) - \mathrm{F1}_F(\text{seen}) \le \mathrm{F1}_{NL}(\text{unseen}) - \mathrm{F1}_{NL}(\text{seen})$.

---

## §4. Method 골격

### 4.1 Facet schema 정의 (deterministic, no learning)

각 task의 도구 카탈로그를 다음 facet으로 분해:

```yaml
facets:
  domain: enum [retail, telecom, airline, ...]   # τ²-bench 도메인
  action_class: enum [read, write, search, compute, communicate]
  arg_types: list [string, int, date, enum, ...]
  precondition: optional list [auth_required, user_logged_in, ...]
  side_effect: enum [none, db_write, external_api_call, irreversible]
  affordance: list [user_visible, agent_internal, async, sync]
```

### 4.2 NL prompt vs facet prompt 예시

**NL prompt** (MetaTool ST4 sample, ~150 tokens):
```
You are a helpful assistant with access to the following tools:
- search_web(query: str): Search the web for information about a topic.
- send_email(to: str, subject: str, body: str): Send an email to the specified recipient.
- get_weather(location: str): Get current weather for a location.
... [12 more tools with descriptions]
Choose the appropriate tools based on user query.
```

**Facet prompt** (length-matched 또는 더 짧음, ~30 tokens):
```
TOOLS:
[search_web action=search args=string]
[send_email action=communicate args=string,string,string side_effect=external]
[get_weather action=read args=string]
... [12 more tools as facet rows]
```

또는 더 압축된 표현:
```
T:{search_web|search|s} {send_email|comm|sss|ext} {get_weather|read|s} ...
```

### 4.3 측정 protocol

- **F1 metric**: 본 v1.1 E7과 동일 (intervention_metatool_eval.py의 generation 경로).
- **Length**: token count 비교 (BPE tokenizer 기준).
- **Per-step KL**: 첫 K step generation의 token-level KL (NL vs facet) — 정리 4의 trajectory 안정성 검증.

### 4.4 Cross-domain transfer protocol (C3)
1. retail의 facet schema (도구 8개)로 facet prompt 작성 → retail F1.
2. 동일 facet schema로 telecom 도구 8개를 표현 → telecom F1.
3. NL prompt: retail 카탈로그를 telecom 카탈로그로 통째로 교체.
4. F1 격차 비교.

---

## §5. 실험 plan

### E10 (완료 — commit 61fdb7e) — NL vs facet, length-matched
- Task: MetaTool ST4
- 모델: Qwen 2.5-7B + Llama 3.1-8B
- N = 64
- 6 modes (실측): `nl_full`, `nl_with_desc`, `facet_full`, `facet_compact`, `list_only`, `noprompt`
- 측정: F1, exact, precision, recall, prompt_length

**결과 (Qwen N=64, wall 714s)**:

| condition | length | F1 | exact | precision | recall |
|---|---|---|---|---|---|
| nl_full | 145.9 | 0.7495 | 0.4688 | 0.8620 | 0.6953 |
| nl_with_desc | 271.4 | 0.7474 | 0.3906 | 0.9141 | 0.6641 |
| **facet_full** | **311.7** | **0.8203** | 0.5469 | 0.9453 | 0.7578 |
| facet_compact | 136.8 | 0.5234 | 0.1719 | 0.6953 | 0.4375 |
| list_only | 94.9 | 0.2135 | 0.0156 | 0.3125 | 0.1641 |
| noprompt | 54.8 | 0.0000 | — | — | — |

**결과 (Llama N=64, wall 1062s)**:

| condition | length | F1 | exact | precision | recall |
|---|---|---|---|---|---|
| nl_full | 153.8 | 0.8745 | 0.6562 | 0.9557 | 0.8359 |
| **nl_with_desc** | **279.2** | **0.8979** | 0.7188 | 0.9427 | 0.8828 |
| facet_full | 317.5 | 0.8609 | 0.5625 | 0.9193 | 0.8516 |
| facet_compact | 142.7 | 0.5526 | 0.2188 | 0.6784 | 0.5000 |
| list_only | 100.8 | 0.6375 | 0.2188 | 0.8385 | 0.5391 |
| noprompt | 36.7 | 0.0000 | — | — | — |

**핵심 발견**:
1. **Qwen**: facet_full F1=0.8203 > nl_full F1=0.7495 → **+9.5% F1 향상**. 정리 4 부합.
2. **Qwen**: facet_full > nl_with_desc (length-matched 312 vs 271) → **+9.7%**. format 우위 분리됨.
3. **Llama**: nl_with_desc=0.8979 > facet_full=0.8609 → **NL 우위 +3.7%**. 모델 이질성 입증.
4. **list_only/nl_full 비율**: Qwen 0.285 vs Llama 0.729 — **3배 격차**. 모델 진단 metric.
5. **facet_compact**: 두 모델 모두 nl_full 대비 -22-32% (description 부재가 hurt). 순수 typed compression 실패.
6. **precision** ≥ 0.86 in desc-bearing modes — recall과 exact가 격차의 source. *도구 카탈로그가 풍부할 때 선택은 precise, 완전한 set 잡는 게 도전*.

### E10b (NEW, 1시간) — `list_anon` condition: 가설 A vs B 분리
- 도구 이름을 anonymized placeholder (`T1, T2, ..., T10`)로 바꿔 도구 이름 의미 path 차단.
- 시스템 prompt: `"Tools: [T1, T2, ..., T10]. Pick which IDs match. Mapping: T1=NewsTool, T2=WeatherTool, ..."` (mapping은 system 마지막에 명시).
- Llama list_anon F1이 list_only 대비 폭락 → 가설 A (도구 이름 자연어 의미 path).
- 변화 작음 → 가설 B (자연어 instruction-following 자체).
- 추가로 facet_anon: facet annotation에 도구 이름만 anonymize → format vs name effect 분리.
- 모델: Qwen + Llama. N=64.

**Hypothesis**: Llama list_anon F1 ∈ [0.30, 0.50] (list_only=0.638에서 폭락 예상, 그러나 noprompt=0보다는 높음).

### E11 (Cross-domain transfer, 2-3시간) — facet schema 도메인 간 일반화
- Task: τ²-bench retail (114 tasks) / telecom (256 sample) / airline (50 tasks)
- 모델: Qwen + Llama
- 6 modes per (source × target) 쌍:
  - `nl_full_target`: target 도메인의 자연어 prompt 정상 사용 (anchor)
  - `nl_full_source`: source 도메인의 자연어 prompt를 target에 적용 (예상 F1 폭락 — 도구 mismatch)
  - `facet_full_target`: target 도메인 facet schema 정상 (anchor)
  - **`facet_xfer`**: source 도메인의 *facet schema 구조*를 그대로 두고 target 도구 이름만 채워넣음 (action/domain 분류는 source에서 학습)
  - `facet_compact_target`: target 도메인 compact facet (sanity)
  - `noprompt`
- 6 (source, target) 쌍: retail↔telecom, retail↔airline, telecom↔airline (양방향)

**Hypothesis (개정)**:
- Qwen `facet_xfer` F1 ≥ 0.7 × `facet_full_target` F1 → schema 재사용성.
- Llama `facet_xfer` F1 변화 작음 (잔차 흐름이 어차피 도구 이름에서 의미 추출).
- 둘 다 `nl_full_source` F1 < 0.3 → 도구 이름 mismatch가 자연어 prompt를 망가뜨림 (facet은 견딤).

**Schema 추출 protocol (E11 prep)**:
- `extract_facet_schema_tau2.py` 신규 작성 (metatool extractor 일반화).
- τ²-bench의 `tasks.json`에서 `evaluation_criteria.actions` 필드의 도구 이름 추출.
- 도구 이름 + 도메인 컨텍스트로 keyword rule 분류 (action_class + domain).
- output: `data/facet_schemas/{tau2_retail,tau2_telecom,tau2_airline}.yaml`.

### E12 (Compositional, optional, 2-3시간) — unseen tool combinations
- MetaTool ST4의 15 GT 도구 중 5개를 hold-out → 그 도구가 GT인 query를 evaluate.
- 모델: facet schema는 학습된 9-action × 24-domain만 보유 (held-out 도구도 schema에서 자동 분류 가능).
- 비교: facet_full vs nl_with_desc on held-out vs seen queries.
- F1 generalization gap = F1(seen) - F1(held-out).

**Hypothesis**: facet의 generalization gap < NL의 generalization gap (compositional 우위).

### E13 (Theory-empirics anchor, 1시간) — facet count → r* 일치
- 본 v1.1 `measure_phi_rank.py`를 5 prompt-mode로 재실행:
  - `nl` (현재 default — 자연어 카탈로그)
  - `facet_full` (typed schema with desc)
  - `facet_compact` (typed compact, no desc)
  - `facet_action_only` (action facet 단독, domain 제거)
  - `facet_domain_only` (domain facet 단독, action 제거)
- 각 mode에서 r*(0.95) 측정. MetaTool ST4 + tau2 retail/telecom/airline (4 tasks × 5 modes = 20 measurements).

**Hypothesis (정리 3 직접 검증)**:
- nl mode: r*(0.95) ≈ 2.25 (현재 측정값).
- facet_full: r*(0.95) ≈ 2 (action + domain 두 facet 활성).
- facet_compact: r*(0.95) ≈ 2 (typed format이지만 정보 동일).
- facet_action_only: r*(0.95) ≈ 1 (1 facet).
- facet_domain_only: r*(0.95) ≈ 1.

**Prep**: `measure_phi_rank.py`에 `--prompt-mode {nl, facet_full, facet_compact, facet_action_only, facet_domain_only}` 인자 추가, schema YAML 로드 후 prefix 텍스트 합성.

---

## §6. 새 측정 스크립트 디자인

### 6.1 `scripts/rank_replaceability/facet_eval.py` (NEW)
`intervention_metatool_eval.py` 기반:
- `--prompt-mode nl|facet_full|facet_compact|noprompt` argparse.
- facet schema는 별도 YAML/JSON에서 로드: `data/facet_schemas/metatool_st4.yaml`.
- 토큰 길이 + F1 같이 보고.

### 6.2 `data/facet_schemas/` (NEW 디렉토리)
- `metatool_st4.yaml` — 30+ 도구의 facet 분해.
- `tau2_retail.yaml`, `tau2_telecom.yaml`, `tau2_airline.yaml` — 도메인별.

### 6.3 `scripts/rank_replaceability/measure_phi_rank.py` 확장 (이미 있음)
- `--prompt-mode` 인자 추가, facet schema 받기.

---

## §7. Paper 구조 (한국어 LaTeX 초안 v0 skeleton)

```
math/paper/iclr2027_facet_ontology/
├── PAPER_v0_ko.tex           # 한국어 초안 (cowork)
└── PAPER_DRAFT_v0.md         # 영어 markdown 초안 (선택)
```

### Abstract
- thesis (위 §1)
- subclaims C1-C4
- 핵심 측정 결과 (E10-E13 요약)
- 본 v1.1, 자매 논문과의 관계 명시

### §1 서론
- tool-use LLM의 system prompt 비용 문제 (본 v1.1과 공유)
- 자연어 vs formal — 정보이론·LM 효율 angle
- 기여 5개 (이론 2개, 실증 3개)

### §2 관련 연구
- function calling (Anthropic, OpenAI) — typed schema는 있되 attention-level 분석 없음
- prompt compression (LLMLingua, GIST, 500xCompressor) — NL → 짧은 NL/learned, formal 변환 아님
- ontology-augmented LLM (KG-RAG, ToolBench) — retrieval 단, attention 단 분석 없음
- 본 v1.1, 자매 논문 — 자매·본의 결과를 인용·확장

### §3 이론 (정리 3·4·5)
- 본 v1.1 정리 1을 lemma로 인용
- 정리 3 (facet-rank), 정리 4 (formal advantage), 선택 정리 5 (compositional)

### §4 실험 설정
- facet schema 정의 protocol
- 모델·task (E10·E11)
- length normalization, fairness

### §5 결과
- §5.1 E10 (NL vs facet, length-matched)
- §5.2 E11 (cross-domain transfer)
- §5.3 E12 (compositional, optional)
- §5.4 E13 (theory anchor: facet count → r*)

### §6 논의
- §6.1 본 v1.1 multi-step F1=0 negative와의 관계
  - facet prompt는 *prompt 형식 변경*이지 *prompt 제거*가 아니므로 KV cache에 prefix entries 정상 보유
  - E10 결과: facet_full multi-step F1 = 0.82 (Qwen) / 0.86 (Llama) — *generation 손상 없음* 입증.
  - 따라서 본 v1.1 §6.4 trajectory 발산 메커니즘은 본 framework에 적용 안 됨.
- §6.2 자매 논문 K-subspace 결과와의 관계
- §6.3 모델 이질성과 그 mechanism (Llama 잔차흐름 dominance 가설)
- **§6.4 (NEW, E10이 만든 새 contribution) Model-specific deployment guideline**:
  - 진단: `list_only/nl_full F1 ratio` (Qwen 0.285, Llama 0.729). 비율이 모델별 facet 효과 예측.
  - **Mode α (direct facet prompt)**: 비율 < 0.4 (attention-conditioning native). 정확도 +5-15% (Qwen 같은). 권장 deployment.
  - **Mode β (length compression for any model)**: facet_compact 자체 성능은 부족하나 fine-tuning과 결합 시 길이 절감 가능. 또는 facet_full을 token-budget 내 max-info 표현으로 사용.
  - **Mode γ (retrieval / fine-tuning)**: 비율 > 0.7 (residual-stream native, Llama 같은). facet schema를 KG-retrieval의 index로 사용하거나 LoRA fine-tuning에서 facet annotation 직접 학습.
  - **Mode δ (hybrid, 가설)**: NL prompt + facet annotation 결합 — 두 path 모두 활성. 미실험.
- §6.5 한계 (facet schema 수동 정의, domain-specific, hand-tuned action keywords)

### §7 결론
### 향후 작업
- 학습된 facet embedding (soft prompt로 더 압축)
- facet schema 자동 추출 (도구 documentation에서)

---

## §8. Timeline (자율 실행 가정, 1-2 sprint)

| 단계 | 작업 | 시간 추정 |
|---|---|---|
| S1 | facet_eval.py 작성 + metatool_st4.yaml 작성 | 2시간 |
| S2 | E10 실행 (Qwen + Llama, N=64) | 1시간 (병렬) |
| S3 | E10 결과 분석 + commit | 30분 |
| S4 | tau2_*.yaml 3개 작성 | 1시간 |
| S5 | E11 실행 + 분석 | 2시간 |
| S6 | E13 (theory anchor) | 1시간 |
| S7 | paper v0 한국어 LaTeX skeleton | 2시간 |
| S8 | paper §3 이론 정리 (정리 3·4 정식 진술 + 증명) | 4시간 |
| S9 | paper §5 결과 표·그래프 통합 | 2시간 |
| S10 | E12 (선택, compositional) | 2시간 |

**총 17–20시간** (실험 ~6시간, 페이퍼 ~10시간, 분석 ~3시간).

---

## §9. Risks (negative result 시나리오)

### R1. facet prompt가 NL prompt보다 F1 *낮게* 나옴
- 가능성: 중간 (LM이 NL training data에 over-fit돼서 formal schema는 OOD).
- **상태 (E10 후)**: **부분 confirmed**. Llama 3.1-8B에서 facet_full -3.7% (NL 우위). Qwen 2.5-7B에서는 +9.7% (facet 우위). 모델별 분기 패턴.
- 대응: paper §6.4를 새 positive contribution으로 reframe — "deployment guideline by model diagnostic". Negative single-headline 대신 *model-conditional* claim으로 정리.

### R2. facet schema 정의의 task-dependence
- 가능성: 높음 (각 task마다 facet 구조가 다름).
- **상태 (E10 후)**: 미검증 (E11에서 입증 예정).
- 대응: 정리 5 (compositional)이 깨지면 limitation에 명시. 핵심 contribution은 single-task format-superiority (C2) + model diagnostic (C5) 단독으로도 valid.

### R3. multi-step F1이 NL보다 *낮음*
- 가능성: 중간 (formal schema는 LM의 generation prior 외부).
- **상태 (E10 후)**: **partial reject** (Qwen). E10이 multi-step F1을 직접 측정 (max_new_tokens=192 generation), Qwen facet_full F1=0.82 — generation 정상 작동. 본 v1.1 E7 negative와 다른 regime.
- 대응: 본 v1.1 §6.4 trajectory 발산 메커니즘이 facet replacement에서는 발생하지 않음을 §6.1에 명시.

### R4. 자매 논문과의 framing 충돌
- 가능성: 낮음 (§0.2에서 차단).
- 상태: 안전. E10 결과는 자매 논문 F1 결정문(B_ont K-subspace 추출 protocol)과 직교.
- 대응: §1에서 자매 paper F1 결정문 인용. 새 paper의 주장이 *인풋 토큰 형식*에 한정됨을 명시.

### R5 (NEW, E10 후). facet_compact의 길이 절감 promise 깨짐
- E10에서 facet_compact F1이 nl_full보다 22-32% 낮음. C1 (length efficiency) subclaim 부분 reject.
- 대응: C1을 "speed-priority deployment에서 *some* F1 trade-off"로 reframe. 또는 length-매치 비교를 facet_full vs nl_with_desc로 (둘 다 desc 있음, format만 다름) — 이쪽은 Qwen에서 +9.7%로 fair한 length-format isolation이며 paper의 핵심 비교가 됨.

### R6 (NEW). 모델 diagnostic의 N=1 unreliability
- 현재 list_only/nl_full 비율 = Qwen 0.285, Llama 0.729 — N=2 (모델 2개)에서 dramatic 격차지만 일반화 여부 불명.
- 대응: Mistral 7B, Phi-3, Gemma-2 같은 추가 모델로 검증 (E11 또는 별도 cell). 또는 paper에 limitation으로 정직 명시.

### Mechanism hypothesis (E10 + 사용자 input 반영)
- "Llama가 Function Calling 학습 부족"으로 해석되기 쉬우나 *반대*: Llama-3.1은 RLHF로 자연어 instruction-following에 *과도하게 saturated* → typed schema는 OOD → facet 효과 약함.
- Qwen 2.5는 자연어에 *덜* saturated → attention-level conditioning이 살아있음 → typed format이 attention 정렬 도와줌 → +9.7%.
- → Paper §6.4 새 통찰: **"Facet framework은 instruction-tuning이 덜 saturated된 모델에 더 효과적이며, RLHF-saturated 모델은 fine-tuning 단계에서 facet annotation을 직접 학습시켜야 한다"**.
- 이 가설의 확장 검증: instruction-tuning 강도가 다른 모델 셋(예: Llama-3.1-8B-base + Llama-3.1-8B-Instruct + Llama-3.3-70B-Instruct)에서 facet 효과를 측정 → instruction-tuning saturation과 facet 효과의 단조 negative 관계 입증 시도.

---

## §10. 인프라 결정

### 브랜치
**옵션 A**: `iclr-prompt-internalization` 브랜치 그대로 사용 (현재 본 v1.1 + E7 commit이 있는 브랜치).
- 장점: 단일 진실원, paper triplet 한 곳에 위치.
- 단점: 새 paper가 본 v1.1과 commit 구분 불명확.

**옵션 B**: `iclr-facet-ontology` 새 브랜치 분기.
- 장점: 깨끗한 commit 트리.
- 단점: 본 v1.1의 인프라(scripts/, reports/) cherry-pick 필요.

**권장**: **옵션 A** — 기존 브랜치에서 commit prefix를 `iclr-facet`로 구분. 인프라가 동일 (measure_phi_rank.py, intervention_metatool_eval.py 같은 스크립트 공유).

### 디렉토리
- 새 paper: `math/paper/iclr2027_facet_ontology/PAPER_v0_ko.tex`
- 새 데이터: `data/facet_schemas/{metatool_st4,tau2_retail,tau2_telecom,tau2_airline}.yaml`
- 새 스크립트: `scripts/rank_replaceability/facet_eval.py`
- 결과 디렉토리: `reports/facet_ontology_2026_04/`

---

## §11. 즉시 다음 액션 (v28.1 작성 시점)

**완료**:
1. ✅ plan v28 commit (d6d46e4) + paper skeleton (9656751) + E10 결과 (61fdb7e).
2. ✅ E10 N=64 두 모델 완료, model-heterogeneity 입증.

**다음 sprint (prep 단계)**:
3. `facet_eval.py`에 `list_anon` condition 추가 (E10b).
4. `extract_facet_schema_tau2.py` 작성 (E11 prep): retail/telecom/airline 도메인 facet schema.
5. `measure_phi_rank.py`에 `--prompt-mode` 인자 추가 (E13 prep).
6. plan v28.1 + 위 prep 모두 commit + push.

**E10 결과 후속 (실행 단계)**:
7. **E10b (`list_anon`)** 양 모델 N=64: 가설 A (도구 이름 의미) vs B (자연어 instruction-following) 분리. ~20분 (Qwen), ~30분 (Llama).
8. **E13** (facet count → r* 일치): 5 prompt mode × 4 task × 2 모델 = 40 measurements. ~1시간.
9. **E11** (cross-domain transfer): 6 (source × target) × 6 conditions × 2 모델. ~3-4시간.

**Paper integration (E10b 완료 후)**:
10. paper v0 §5.1 Results 채우기 (E10 N=64 numbers).
11. paper v0 §5.1b 추가 (E10b list_anon 결과 + 모델 진단 발견).
12. paper v0 §6.4 deployment guideline 작성.

**전체 sprint 완료 후 main merge**:
13. `iclr-prompt-internalization` → `main` fast-forward 머지 (현재 v1.1 + E7 + plan v28 + facet artifacts + E10 + 새 prep까지).
