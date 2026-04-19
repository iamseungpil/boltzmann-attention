# E1 — Engineering-Interface Benchmark

**Status**: spec-draft (2026-04-19)
**Origin**: 2026-04-19 brainstorm reframe session. User argument: "LLM 내재화가 외부 ontology 로 직접 프롬프트 제어한 CoT/ToT 효과와 상호 변환 가능하나, 비용/단순/표준화 차이는 크다."
**Claim framing**: Not new capability; **more efficient/standardized control interface** for capabilities CoT/ToT also reach.

---

## §1. Rationale

**Problem addressed**: ICL Bayesian null (Xie 2022). LLM already has ontology from text pretraining. "External ontology injection" is not new information; it is **re-weighting**. Strictly applied, this null would delete CoT/ToT as contributions too — but the field accepts them because they are better **control interfaces**.

**Our claim (survives ICL null)**:
> Attention-level F12/F13 achieves parity with CoT/ToT on task space X, with superior cost / variance / HP-dim / standardization.

**Precedent**: LoRA (Hu 2021), RepE (Zou 2023), CAA (Panickssery 2024), Constitutional AI (Bai 2022) — all accepted "new interface for existing capability" as contribution.

---

## §2. Four measurable axes

| Axis             | Attention-level (F12/F13)   | Prompt-level (CoT/ToT)      | Metric                                    |
|------------------|-----------------------------|-----------------------------|-------------------------------------------|
| Inference tokens | ≈ base (hook, no extra)     | CoT 2–5×, ToT 10–50×        | output token count per query              |
| HP dim           | (α, axis, schedule) ~3–4    | prompt template ~∞          | # of tunable hyperparams                  |
| Paraphrase var   | low (weight-level)          | high (prompt-sensitive)     | std(accuracy) across 10 paraphrases       |
| Peak accuracy    | TBD                         | ToT strong                  | top-1 on MetaTool Subtask4                |

**Interpretation**: win 2-of-4 → contribution holds; win 0–1 → branch dies.

---

## §3. Protocol (brief)

1. **Shared task**: MetaTool Subtask4 N=200 (Qwen2.5-7B), same as F12/F13 eval.
2. **Baselines added** (E1-specific, piggyback on F12/F13 run):
   - CoT (Wei 2022): "Let's think step by step." prepended.
   - ToT (Yao 2023): 3 candidate reasoning paths, self-evaluation.
   - Self-Consistency (Wang 2022): N=5 CoT samples + majority vote.
3. **Paraphrase robustness** (shared resource with F12/F13):
   - 10 prompt paraphrases (held-out set).
   - Measure accuracy std per condition.
4. **Cost instrumentation**:
   - Record: wall-clock, output tokens, FLOPs estimate (tokens × model FLOPs/token).
5. **HP sensitivity curve**:
   - F12: sweep α ∈ {0.01, 0.05, 0.1, 0.2}
   - CoT: hold fixed (no HP)
   - ToT: sweep `n_branches ∈ {2, 3, 5}`, `n_depth ∈ {2, 3}`
   - Report dim-of-search-space + sensitivity slope.

---

## §4. 4-outcome pre-reg (MOFCISS-precedent compliant)

| Outcome (2-of-4 axes win?) | Interpretation                               | Paper action                                        |
|----------------------------|----------------------------------------------|-----------------------------------------------------|
| Win 4/4                    | Unambiguous interface superiority            | Main §5.X "Engineering-interface superiority"       |
| Win 2–3/4                  | Partial; specify which axes                  | §5.X with scope-limit to winning axes               |
| Win 0–1/4 with acc parity  | No interface win but parity                  | §6 Discussion footnote; drop main claim             |
| Lose on accuracy           | Prompt-level strictly dominates              | E1 branch dormant; paper pivots away from reframe   |

---

## §5. Dependency

- **Runs piggyback on F12/F13** — same prompts, same N, same model. Additional GPU cost ≤ 3 hr for CoT/ToT/SC baselines.
- **Independent of F12/F13 outcome**: E1 emits useful data regardless (even if F12/F13 null, E1 gives us CoT/ToT cost baseline for future comparison).
- **Triggers reframe when**: F12/F13 both marginal (<+3pp) AND E1 wins 2-of-4 → paper §1 reframes to engineering-interface as primary contribution.

---

## §6. Code/artifact layout

```
branches/E1_engineering_reframe/
├── SPEC.md            (this file)
├── STATUS.md          (updated as runs complete; currently empty)
├── scripts/           (will be added: eval_cot_baseline.py, eval_tot_baseline.py, eval_sc_baseline.py, measure_e1.py)
├── prompts/           (10 paraphrases; to be added)
└── results/           (per-axis JSON outputs; to be added)
```

---

## §7. Risks

1. **"Mutually convertible" over-claim**: empirical only, scope-limited to MetaTool Subtask4. Do not generalize beyond tested task.
2. **ToT as strong baseline**: if ToT wins accuracy by >5pp, "parity" claim dies and E1 must scope-limit to non-peak tasks.
3. **Variance measurement noise**: 10 paraphrases may be too few. If std overlap is large, either raise N or retire the axis.
4. **Reviewer framing attack**: "This is just efficiency, not science." Counter: LoRA/RepE/CAA/ConstAI precedent; §2 Related Work must cite these four explicitly.
5. **Scope creep into H-HOT territory**: do NOT claim "qualitatively different" at E1 level; that is H-HOT's job.

---

## §8. Cross-references

- Session reframe: this session's transcript (not saved to memory yet; see change log below when added).
- NEW_THEOREM_TEST.md §2.5 Group 6 (Lu 2022 order effects, Templeton 2024 SAE) — tangential theoretical anchor.
- Master tracking: `reports/new_theorem_test/EXPERIMENT_BRANCH_TRACKING.md` §4.E1.
- CoT/ToT/SC papers as baselines (to be cited in §2 Related Work of main ICLR draft).
