# EXPERIMENT_PLAN_v28 — Facet/Ontology Framework for Tool-Use LLMs

**작성**: 2026-04-29
**브랜치**: `iclr-prompt-internalization` (또는 새 branch 분리 검토 — §10 참조)
**상태**: 초안 v0 — 아직 실험 시작 전
**다음 목표 venue**: ICLR 2027 / TMLR / NeurIPS 2027 (3 paper triplet의 세 번째)

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

### Subclaim
- **C1 (length-efficiency)**: facet prompt 길이 $\ell_F$가 NL prompt 길이 $\ell_{NL}$의 1/3–1/5에서 동등 F1.
- **C2 (multi-step robustness)**: facet prompt가 NL prompt보다 multi-step F1이 통계적으로 유의하게 우수 (특히 도구 ≥ 3개 task).
- **C3 (cross-domain transfer)**: $\tau^2$-bench retail에서 학습/구성된 facet schema가 telecom/airline에 transfer (NL prompt는 도구 카탈로그 통째로 교체).
- **C4 (compositional)**: 처음 보는 도구 묶음(facet 결합)에서 NL prompt 대비 F1 격차 작음.

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

### E10 (Power-test, 1-2시간) — NL vs facet, length-matched
- Task: MetaTool ST4
- 모델: Qwen 2.5-7B + Llama 3.1-8B
- N = 64 (E7과 일치, 비교 가능성)
- 4 mode: `nl_full` (현재 시스템 prompt) / `facet_full` (facet schema, 길이 비슷) / `facet_compact` (압축 facet, 1/3 길이) / `noprompt`
- 측정: F1, exact, precision, recall, length, first-step KL

**Hypothesis**: facet_full과 nl_full F1 차이 ≤ 5%. facet_compact F1이 nl_full의 80% 이상.

### E11 (Cross-domain, 2-3시간) — facet schema transfer
- Task: τ²-bench retail / telecom / airline
- 모델: Qwen + Llama
- Source: retail에서 정의한 facet schema
- Target: telecom + airline (도구 이름만 schema에 채워넣음)
- 비교: source-domain NL prompt를 target에 그대로 사용 (예상 F1 폭락) vs facet schema transfer

**Hypothesis**: facet transfer가 NL transfer 대비 F1 격차 50% 이상 줄임.

### E12 (Compositional, optional) — unseen tool combinations
- MetaTool ST4의 일부 tool 묶음을 hold-out → facet 결합으로 표현 가능한지 평가.
- F1 generalization gap 비교.

### E13 (Theory-empirics anchor) — facet count → r* 일치
- 본 v1.1 measure_phi_rank.py를 facet 개수별로 재실행: facet 1개만 노출 → r* 측정, 2개 → r* 측정, ...
- 정리 3 ("rank lower bound = facet count")의 직접 실증 검증.

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
- 본 v1.1 multi-step F1=0 negative와의 관계
  - facet prompt는 *prompt 형식 변경*이지 *prompt 제거*가 아니므로 KV cache에 prefix entries 정상 보유
  - 따라서 본 v1.1의 KV-cache 오염 누적 문제는 발생하지 않음 (예측: facet prompt multi-step F1 > 0)
- 자매 논문 K-subspace 결과와의 관계
- 한계 (facet schema 수동 정의, domain-specific)

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
- 대응: §6에 정직하게 보고. 본 v1.1과 마찬가지로 negative result도 valid contribution. 정리 3·4 자체는 lower bound 진술이므로 framework은 살아남음.

### R2. facet schema 정의의 task-dependence
- 가능성: 높음 (각 task마다 facet 구조가 다름).
- 대응: 정리 5 (compositional)이 깨지면 limitation에 명시. 핵심 contribution은 single-task length-efficiency (C1) 단독으로도 valid.

### R3. multi-step F1이 NL보다 *낮음*
- 가능성: 중간 (formal schema는 LM의 generation prior 외부).
- 대응: §6에 분석. trajectory 안정성 비교 (per-step KL)로 mechanism 이해.

### R4. 자매 논문과의 framing 충돌
- 가능성: 낮음 (§0.2에서 차단).
- 대응: §1에서 자매 paper F1 결정문 인용. 새 paper의 주장이 *인풋 토큰 형식*에 한정됨을 명시.

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

## §11. 즉시 다음 액션 (이 plan commit 후)

1. **plan commit**: 이 파일을 `reports/EXPERIMENT_PLAN_v28_facet_ontology_2026_04_29.md`로 push.
2. **paper skeleton 생성**: `math/paper/iclr2027_facet_ontology/PAPER_v0_ko.tex` 빈 골격 (서언 + 빈 섹션).
3. **E10 power-test prep**: `scripts/rank_replaceability/facet_eval.py` 작성 + `data/facet_schemas/metatool_st4.yaml` 작성.
4. **E10 실행**: Qwen + Llama background.
5. **E10 결과 분석 + 본 plan 업데이트**.
