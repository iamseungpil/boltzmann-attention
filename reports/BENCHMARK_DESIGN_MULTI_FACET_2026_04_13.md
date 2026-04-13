# Benchmark Design: Multi-Facet Multi-Turn Tool Selection

**Date**: 2026-04-13
**Goal**: 공개 벤치마크 기반 ontology K-bias tool selection 검증

---

## 1. 벤치마크 선정 근거

20+ 벤치마크 survey 결과, **"role-dependent tool disambiguation"을 테스트하는 공개 벤치마크가 없음**. 이것이 우리 논문의 novelty dimension.

### 선정된 벤치마크 조합

| # | Benchmark | 용도 | Tools | Turns | 공개 |
|---|-----------|------|-------|-------|------|
| **B1** | MetaTool Subtask1 | Single-turn baseline (이미 완료) | 199 | 1 | Yes |
| **B2** | CRMArena-Pro | Enterprise multi-turn + persona | 27 + SOQL | 5-8 | Yes |
| **B3** | CONFETTI | Multi-turn goal switching + disambiguation | 86 APIs (15/conv) | multi | Yes |
| **B4** | C³-Bench | Large catalog + inter-tool dependency | 256 tool groups | multi-step | Yes |
| **B5** | **MF-Disambig** (신규 설계) | Role-dependent facet disambiguation | CRMArena 기반 확장 | multi | **생성** |

---

## 2. 각 벤치마크별 K-bias 적용 계획

### B1: MetaTool Subtask1 (DONE)

- 995 queries × 199 tools, single-turn
- Qwen +11.16pp, Llama +10.25pp 확인 완료
- B_ont: 4-facet (function_action, io_type, domain, tool_category)

### B2: CRMArena-Pro

**구조** (실측):
- DB: 16 tables (CRMArena) / 27 tables (Pro B2B)
- 9 task types × 130 samples = 1,170 total
- Tasks: knowledge_qa, monthly_trend_analysis, top_issue_identification, case_routing, named_entity_disambiguation, handle_time, transfer_count, policy_violation_identification, best_region_identification
- 25 Python wrapper functions (get_cases, get_agents, calculate_average_handle_time 등)
- Personas: Service Agent, Analyst, Manager

**K-bias 적용**:
```
Step 1: CRMArena의 25개 tool에서 6-facet ontology 구축
  F1 Purpose:    analysis / routing / identification / qa
  F2 Function:   get / calculate / search / find / submit
  F3 Structure:  case / order / product / agent / knowledge
  F4 Parameter:  date_range / agent_id / case_id / product_id
  F5 Domain:     service / sales / operations / knowledge
  F6 Stakeholder: agent / analyst / manager

Step 2: B_ont 구축 (Qwen/Llama on CRMArena tool descriptions)
Step 3: 1,170 tasks에서 K-bias vs no_steer eval
  Metric: task accuracy (fuzzy_match / exact_match per task type)
```

### B3: CONFETTI (Amazon, ACL 2025)

**구조** (실측):
- 506 conversations, 각 대화에 ~15 tools 할당
- Multi-turn with goal correction/switching
- Dialog act annotations (follow-up, correction, new goal)

**K-bias 적용**:
```
Step 1: 506개 대화의 15 tools/conv에서 facet ontology 구축
Step 2: Per-turn tool selection에 K-bias 적용
Step 3: Eval: per-turn function call accuracy
  Focus: goal-switching turns에서의 K-bias 효과
```

### B4: C³-Bench (Tencent)

**구조** (실측):
- 255 tasks, 256 tool groups (각 group에 multi-tool)
- DAG dependency structure (dependency_list per action)
- Multi-step with observation feedback

**K-bias 적용**:
```
Step 1: 256 tool groups에서 6-facet ontology 구축
Step 2: Multi-step task에서 K-bias 적용
  Metric: Accomplish Progress (AP), Optimal Path Rate (OP)
  Focus: inter-tool dependency 해결에 K-bias가 기여하는지
```

### B5: MF-Disambig (신규 설계 — CRMArena 기반)

**기존 벤치마크에 없는 것**: 같은 query에 대해 stakeholder에 따라 정답 tool이 다른 시나리오.

**설계**:

CRMArena의 Salesforce 객체 구조 위에, role-dependent tool disambiguation 레이어를 추가.

#### 5.1 Disambiguation 시나리오

CRMArena의 9개 task type을 3개 persona에 매핑:

| Query (동일) | Service Agent 정답 | Analyst 정답 | Manager 정답 |
|-------------|-------------------|-------------|-------------|
| "Show me case trends" | get_cases (→ list recent cases) | get_month_to_case_count (→ monthly aggregation) | calculate_region_average_closure_times (→ regional summary) |
| "Who handles the most?" | get_agents_with_max_cases (→ agent name) | get_agent_handled_cases_by_period (→ detailed breakdown) | find_id_with_max_value (→ top performer ID for KPI) |
| "What's the issue?" | get_email_messages_by_case_id (→ case detail) | get_issue_counts (→ issue frequency) | search_knowledge_articles (→ policy lookup) |
| "Check the product" | search_products (→ product lookup) | get_order_item_ids_by_product (→ order analysis) | get_purchase_history (→ revenue impact) |

#### 5.2 데이터 생성 방법

```python
# CRMArena 25개 tool에서 disambiguation triplet 생성
# 각 triplet: (ambiguous_query, {persona: correct_tool})

for query_template in ambiguous_queries:  # ~30 templates
    for persona in ["service_agent", "analyst", "manager"]:
        # CRMArena DB에서 실제 데이터로 ground truth 생성
        correct_tool = persona_tool_mapping[query_template][persona]
        correct_params = generate_params_from_db(correct_tool, crmarena_db)
        
        yield {
            "query": query_template,
            "persona": persona,
            "system_prompt": f"You are a {persona}. Use CRM tools to help.",
            "ground_truth_tool": correct_tool,
            "ground_truth_params": correct_params,
            "distractor_tools": other_persona_tools + random_tools,
        }

# Total: ~30 queries × 3 personas = 90 disambiguation instances
# + 30 queries × no persona (ambiguous baseline) = 30 instances
# Total: 120 instances
```

#### 5.3 Facet-Conditioned Steering

```python
# Persona → F6 (Stakeholder) facet의 조건부 α 조정
alpha_config = {
    "service_agent": {"F1": 0.3, "F2": 0.3, "F3": 0.3, "F4": 0.3, "F5": 0.3, "F6_agent": 0.5},
    "analyst":       {"F1": 0.3, "F2": 0.3, "F3": 0.3, "F4": 0.3, "F5": 0.3, "F6_analyst": 0.5},
    "manager":       {"F1": 0.3, "F2": 0.3, "F3": 0.3, "F4": 0.3, "F5": 0.3, "F6_manager": 0.5},
}
```

---

## 3. Evaluation Metrics

### Per-Benchmark Metrics

| Benchmark | Primary | Secondary |
|-----------|---------|-----------|
| **B1 MetaTool** | Top-1 accuracy (Δ pp) | no_match rate |
| **B2 CRMArena** | Task accuracy (fuzzy/exact match) | Per-task-type accuracy |
| **B3 CONFETTI** | Per-turn function call accuracy | Goal-switch turn accuracy |
| **B4 C³-Bench** | Accomplish Progress (AP), Optimal Path (OP) | Dependency resolution rate |
| **B5 MF-Disambig** | **Facet-Conditioned Accuracy (FCA)** | Per-persona F1, Homonym Resolution |

### Aggregate Metrics

| Metric | Definition |
|--------|-----------|
| **Multi-Bench Average Δ** | Average K-bias improvement across B1-B4 |
| **FCA** (new) | P(correct tool \| query, persona) — B5 only |
| **Disambiguation Lift** | FCA(with K-bias) - FCA(no_steer) — our main novelty claim |

---

## 4. Implementation Plan

### Week 1: Ontology + B2 Baseline

| Day | Task |
|-----|------|
| D1 | CRMArena 25 tools에서 6-facet ontology JSON 생성 |
| D1 | B_ont 구축: Qwen + Llama on CRMArena tool descriptions |
| D2 | CRMArena eval pipeline 작성 (SOQL → tool selection format 변환) |
| D3 | B2 eval: 1,170 tasks × {no_steer, K-bias α=0.3} × Qwen |

### Week 2: B3 + B4 + B5 Data Generation

| Day | Task |
|-----|------|
| D4 | CONFETTI eval pipeline (506 conversations → per-turn eval) |
| D5 | C³-Bench eval pipeline (255 tasks → multi-step eval) |
| D6 | MF-Disambig data generation (30 queries × 3 personas × params) |
| D7 | MF-Disambig eval: 120 instances × {no_steer, K-bias, facet-conditioned} |

### Week 3: Cross-Benchmark Analysis

| Day | Task |
|-----|------|
| D8 | All benchmarks × {Qwen, Llama} × {no_steer, K-bias} |
| D9 | Per-facet ablation on B5 (remove one facet, measure FCA delta) |
| D10 | Paper table generation + analysis |

---

## 5. 논문 Contribution Mapping

| Contribution | Evidence Source |
|-------------|---------------|
| K-bias generalizes across models | B1 (Qwen +11pp, Llama +10pp) |
| K-bias works on enterprise CRM tasks | B2 (CRMArena-Pro) |
| K-bias improves multi-turn goal switching | B3 (CONFETTI) |
| K-bias helps inter-tool dependency resolution | B4 (C³-Bench) |
| **Facet-conditioned K-bias enables role-dependent disambiguation** | **B5 (MF-Disambig) — NOVEL** |
| B_ont construction: pad-to-max + skipL0 | Mistral ablation study |

---

## 6. Cloned Repos

| Repo | Location | Status |
|------|----------|--------|
| CRMArena | `external/CRMArena/` | Cloned, DB inspected |
| C³-Benchmark | `external/C3-Benchmark/` | Cloned, 255 tasks + 256 tools |
| CONFETTI | `external/confetti/` | Cloned, 506 conversations |
| τ²-bench | `external/tau2-bench/` | Previously cloned |
| MetaTool | `/tmp/MetaTool/` | In use |
