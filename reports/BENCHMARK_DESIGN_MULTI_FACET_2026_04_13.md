# Benchmark Design: Ontology K-Bias Steering Across Tool Selection & Plan Generation

**Date**: 2026-04-13 (updated)
**Thesis**: Ontology-based K-bias vector steering consistently improves tool selection AND plan generation across diverse benchmarks and scenarios.
**Non-goal**: Benchmark 자체가 novelty가 아님. 다양한 기존 벤치마크에서의 일관된 효과 입증이 목적.

---

## 1. 3-Tier Evaluation Framework

| Tier | 검증 대상 | 질문 |
|------|----------|------|
| **T1: Tool Selection** | 올바른 도구 하나 고르기 | K-bias가 large catalog에서 정확한 도구를 찾는가? |
| **T2: Plan Generation** | 도구 조합 DAG 만들기 | K-bias가 올바른 도구 순서/의존성을 생성하는가? |
| **T3: Multi-Turn Workflow** | 대화 속 연속 도구 호출 | K-bias가 goal switching, context 누적에서도 동작하는가? |

---

## 2. Benchmark Matrix

### T1: Tool Selection

| Benchmark | Tools | Samples | Format | Metric | Status |
|-----------|-------|---------|--------|--------|--------|
| **MetaTool Subtask1** | 199 | 995 | 1-turn, 10 candidates | Top-1 acc | **DONE** (+11pp Qwen, +10pp Llama) |
| **CRMArena-Pro** | 25 + SOQL | 1,170 | Multi-turn CRM | exact/fuzzy match | Cloned |
| **C³-Bench** (tool selection subset) | 256 groups | 255 | Multi-step | Tool selection F1 | Cloned |

### T2: Plan Generation (도구 조합)

| Benchmark | Tools | Samples | Plan 구조 | Metric | Status |
|-----------|-------|---------|----------|--------|--------|
| **TaskBench** (Microsoft) | 23/40/40 (3 domains) | ~28K | **DAG** (Node/Chain/DAG) | **node-F1 + edge-F1** | Cloned |
| **AppBench** | 10 apps × 11-23 APIs | 800 (4 categories) | **Graph** (seq + parallel) | **graph-level Success Rate** | Cloned |
| **C³-Bench** (plan subset) | 256 groups | 255 | dependency_list | AP + OP | Cloned |

### T3: Multi-Turn Workflow

| Benchmark | Tools/conv | Conversations | Turns | Metric | Status |
|-----------|-----------|---------------|-------|--------|--------|
| **CONFETTI** (Amazon) | 15 | 506 | multi | Per-turn function call acc | Cloned |
| **CRMArena-Pro** (interactive) | 25 | ~2K | 5-8 | Task accuracy | Cloned |
| **τ²-bench** | 13-15 | 117/50 | multi | Score | Cloned |

---

## 3. K-Bias 적용 방법 (Tier별)

### T1: Tool Selection (기존 방식)

```
Input: "Search for savings product visit stats"
Hook: K' = K + α · B_ont · B_ont^T · K
Output: 올바른 tool name
Metric: Top-1 accuracy
```

### T2: Plan Generation (NEW)

**K-bias가 plan의 정확도를 높이는 메커니즘**:

```
Input: "Chicago 주말 날씨 알려주고, 비 올 확률 높으면 실내 레스토랑 추천해줘"

Without K-bias → model generates:
  Step 1: getCityForecast(city="Chicago") ← correct
  Step 2: searchRestaurants(city="Chicago") ← wrong (should be indoor only)
  
With K-bias (ontology: weather→forecast, restaurant→indoor/outdoor→preference) →
  Step 1: getCityForecast(city="Chicago", startDate=..., endDate=...)
  Step 2: searchRestaurants(city="Chicago", cuisine_type="any", indoor_only=true)  ← dependency-aware
```

**평가**: TaskBench의 DAG에서
- **node-F1**: K-bias가 올바른 tool 선택을 개선하는가?
- **edge-F1**: K-bias가 올바른 dependency link를 생성하는가?
- AppBench의 graph-level Success Rate 개선 여부

### T3: Multi-Turn Workflow

```
Turn 1: User asks "분석해줘" → K-bias helps select analysis tool
Turn 2: User clarifies "캠페인별로" → K-bias maintains campaign facet focus
Turn 3: User switches "이번엔 코호트로" → K-bias shifts cohort facet weight
```

**평가**: CONFETTI의 goal-switching turns에서 K-bias 효과.

---

## 4. 벤치마크별 상세 데이터

### TaskBench (Plan Generation — PRIMARY)

```
3 domains:
  HuggingFace: 23 tools, 225 links (tool→tool edges)
  Multimedia:  40 tools
  Daily Life:  40 tools

Graph types:
  Node (단일 tool): 30%
  Chain (순차): 70%  
  DAG (병렬+의존): 80%
  (sampling ratio 3:7:8 ≈ 28K total samples)

Plan format:
  {"nodes": [tool_ids], "links": [{"source": A, "target": B, "type": "text"}]}

Metrics:
  n-F1 = F1(predicted tool set, gold tool set)
  e-F1 = F1(predicted edges, gold edges)
```

**K-bias 적용**: tool_desc.json의 23/40/40개 tool에서 6-facet ontology 구축 → B_ont → plan generation 시 K-bias 적용 → n-F1 / e-F1 비교

### AppBench (Plan Generation — SECONDARY)

```
4 complexity categories:
  SS (Single-app Single-API):  200 samples
  SM (Single-app Multi-API):   200 samples  
  MS (Multi-app Single-API):   200 samples
  MM (Multi-app Multi-API):    200 samples → GPT-4o 2.0% success

Plan format:
  {"used_app": ["Trains"], "used_api": [{"findtrains": {params}}], "result_arguments": [...]}

Metric:
  Success Rate = (correct apps AND correct APIs AND correct params AND correct dependencies)
```

### CONFETTI (Multi-Turn — PRIMARY)

```
506 conversations, each with ~15 available tools
Dialog acts: follow-up, correction, new goal, chained calls

Format per conversation:
  {"id": "...", "question": [turns], "function": [15 tool schemas]}
```

### C³-Bench (Hybrid — Tool Selection + Plan)

```
255 tasks, 256 tool groups
Each task: answer_list with dependency_list per action

Metrics:
  AP (Accomplish Progress): fraction of subtasks completed
  OP (Optimal Path Rate): was the shortest valid path taken?
```

---

## 5. 실험 실행 계획

### Phase 1: T1 완성 (Week 1)

| Task | Benchmark | 작업 |
|------|-----------|------|
| 1.1 | MetaTool | **DONE** |
| 1.2 | CRMArena-Pro | 6-facet ontology 구축 → B_ont → 1,170 tasks eval |
| 1.3 | C³-Bench tool selection | 256 tool groups → ontology → eval |

### Phase 2: T2 Plan Generation (Week 2-3)

| Task | Benchmark | 작업 |
|------|-----------|------|
| 2.1 | TaskBench HuggingFace | 23 tools ontology → B_ont → plan generation with K-bias → n-F1 / e-F1 |
| 2.2 | TaskBench Multimedia | 40 tools ontology → B_ont → eval |
| 2.3 | AppBench MM | 10 apps × APIs ontology → B_ont → graph-level Success Rate |
| 2.4 | C³-Bench plan | dependency resolution with K-bias → AP / OP |

### Phase 3: T3 Multi-Turn (Week 3-4)

| Task | Benchmark | 작업 |
|------|-----------|------|
| 3.1 | CONFETTI | Per-turn tool call accuracy with K-bias |
| 3.2 | CRMArena-Pro interactive | Interactive mode + K-bias |
| 3.3 | τ²-bench retail | K-bias on retail workflow |

### Phase 4: Cross-Tier Analysis (Week 4)

| Analysis | 내용 |
|----------|------|
| 4.1 | All Tier 결과 집계: "K-bias가 T1/T2/T3 모두에서 일관되게 개선" |
| 4.2 | Per-facet ablation: 어떤 facet이 어느 tier에서 가장 기여하는가? |
| 4.3 | α sweep per benchmark: 최적 α가 benchmark에 따라 다른가? |
| 4.4 | Model comparison: Qwen vs Llama per tier |

---

## 6. 논문 Story Line

```
Section 1: Introduction
  "LLM tool use에서 ontology가 중요한 이유"

Section 2: Method
  "Multi-facet ontology K-bias steering"
  - 6-facet ontology construction
  - K-bias hook: K' = K + α·B·B^T·K
  - Per-head adaptive rank (pad-to-max + skip pathological layers)

Section 3: Tool Selection Experiments (T1)
  MetaTool: +11pp (Qwen), +10pp (Llama)
  CRMArena-Pro: enterprise CRM에서도 효과
  C³-Bench: 256 tool groups에서도 효과

Section 4: Plan Generation Experiments (T2)  ← KEY NOVELTY
  TaskBench: node-F1 ↑, edge-F1 ↑ (DAG 구조가 개선)
  AppBench: graph-level Success Rate ↑
  "K-bias는 개별 tool 선택뿐 아니라 tool 간 의존성 인식도 향상"

Section 5: Multi-Turn Workflow Experiments (T3)
  CONFETTI: goal-switching turn에서 효과
  CRMArena interactive: multi-turn CRM에서 효과

Section 6: Analysis
  Cross-model: Qwen + Llama 일관
  Per-facet ablation
  B_ont construction: pad-to-max + skipL0 필수
  Mistral failure analysis (negative result)
  
Section 7: Related Work
  Prior art: CAA, RepE, PASTA, ASA, Focus Directions → all steering but not ontology-based
  KV quant: connection to coworker's rotation paper (dual-use B_ont)
```

---

## 7. Cloned Repos Summary

| Repo | Path | Tools | Samples |
|------|------|-------|---------|
| CRMArena | `external/CRMArena/` | 25 | 4,280 |
| C³-Benchmark | `external/C3-Benchmark/` | 256 groups | 255 |
| CONFETTI | `external/confetti/` | 86 (15/conv) | 506 conv |
| TaskBench | `external/JARVIS/taskbench/` | 23+40+40 | ~28K |
| AppBench | `external/AppBench/` | 10 apps | 800 |
| τ²-bench | `external/tau2-bench/` | 13-15 | 117+50 |
| MetaTool | `/tmp/MetaTool/` | 199 | 995 |
