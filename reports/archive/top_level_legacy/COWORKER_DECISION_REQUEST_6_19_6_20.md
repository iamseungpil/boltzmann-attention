# Decision Request: §6.19 Mahalanobis-Kantorovich & §6.20 HEAT Sections

**To**: `iamseungpil` (coworker)
**From**: `mais` (Claude session)
**Date**: 2026-04-08
**Subject**: Paper inclusion decision — cut or demonstrably fix

---

## Context

Codex review (2026-04-08) gives **borderline weak accept (5.5/10)** conditional on 5 fixes. Condition V8 is:

> "§6.19 Mahalanobis-Kantorovich and §6.20 HEAT are either cut or demonstrably fixed"

Codex also said:
> "If the paper slips the timeline or if §6.19/§6.20 remain and trigger a second retraction cycle, it falls back to reject."

**This is your sections** — you wrote them. I need your decision on how to proceed.

---

## Current Status of §6.19 (Mahalanobis-Kantorovich)

**Contents** (LIE_GROUP_UNIFICATION.md lines 3450-4306):
- 6.19.1 Euclidean MSE의 비최적성
- 6.19.2 Mahalanobis quantizer definition
- 6.19.2.1 **Core theorem**: MK strictly dominates Lloyd-Max in attention distortion
- 6.19.2.2 MK optimality range: Banach space distortion metric hierarchy
- 6.19.2.x **Axis 2 failure qualitative explanation** (quantitative verification labeled as future work)
- 6.19.3 Q-weighted PCA (E1 extension)
- 6.19.4 E1+E2+E3 = fokvq_full

**Risks identified by Codex**:
- "§6.19.2.1 core theorem" uses single-head single-token case; multi-layer extension hand-waved
- "§6.19.2.x Axis 2 failure qualitative explanation" defers quantitative verification
- Our Next-9 attempted Mahalanobis Lloyd full-model → **982 PPL catastrophe** (numerical)

**What this means**:
- The theorem in §6.19.2.1 may be correct in principle
- But it has not been demonstrated to work in practice at full-model PPL
- Mais-side Next-9 failure is evidence AGAINST the method's practical viability
- Including §6.19 without demonstrably fixing Next-9 failure risks second retraction

### Options for §6.19

**Option 1: Cut entirely**
- Remove §6.19.2.1 core theorem claim
- Remove §6.19.3 Q-weighted PCA (already demoted by PCA-Q alignment discovery)
- Keep only §6.19.1 (Euclidean MSE sub-optimality critique) as motivation
- **Pros**: Honest, safe, supports "understanding paper" framing
- **Cons**: Loses formal claim on Fisher-optimality

**Option 2: Demonstrably fix**
- Re-implement Mahalanobis Lloyd with cascade weighting + numerical stability (per §6.23.14)
- Run Next-9 v2 with proper fixes
- Verify full-model PPL beats at least baseline Lloyd
- **Pros**: Preserves formal contribution
- **Cons**: Risk of another failure; 2-3 days of implementation work needed

**Option 3: Restrict claim**
- Keep §6.19.2.1 theorem but explicitly restrict to "single-head, local distortion measurement"
- Add explicit disclaimer: "Multi-layer PPL transfer is open; Next-9 numerical attempt failed"
- Do NOT claim MK as a method
- **Pros**: Preserves theoretical content without method overclaim
- **Cons**: Weakens the original §6.19 framing

**Mais recommendation**: **Option 3 (Restrict claim)**
- Keeps your theoretical contribution visible
- Matches §6.23 explanatory framing (not method)
- No new experiments needed (fits 28-day timeline)
- Aligns with "understanding paper" positioning

---

## Current Status of §6.20 (HEAT Axis 3)

**Contents** (LIE_GROUP_UNIFICATION.md lines 4307-4454):
- 6.20.1 기존 방법의 한계
- 6.20.2 HEAT 이론 요약
- 6.20.3 HEAT 기반 위치 인식 양자화
- 6.20.4 NIAH 예측과의 연결

**Risks identified by Codex**:
- HEAT (Hamiltonian Energy Attention Tracking?) is coworker-specific theory
- Not measured against WF(floor=2) directly
- NIAH prediction needs independent verification
- If this is the "Hamiltonian" approach mentioned in some coworker docs, it may overlap with unresolved physics analogies

**What this means**:
- §6.20 may be a complete and valid theory on its own
- But its relationship to WF(floor=2) (the current best method) is unclear
- Including §6.20 without showing how HEAT beats or combines with WF risks reviewer confusion

### Options for §6.20

**Option 1: Cut entirely**
- Move HEAT to separate follow-up paper
- Main paper focuses on Axis 1 + Axis 2 + Axis 3 WF
- **Pros**: Clean, focused, fits timeline
- **Cons**: Loses Axis 3 theoretical contribution

**Option 2: Demonstrably connect to WF**
- Show: HEAT allocation = WF allocation + position-dependent correction
- Benchmark HEAT vs WF(floor=2) at same budget
- If HEAT > WF, include; otherwise cut
- **Pros**: Grounds HEAT empirically
- **Cons**: Requires NIAH/long-context experiments, ~3-5 days

**Option 3: Position as orthogonal axis**
- §6.20 applies to long-context (NIAH) whereas §6.19/§6.23 are short-context (PPL)
- Frame HEAT as "dynamic token selection layer on top of static quantization"
- Explicitly state: "HEAT and WF(floor=2) are orthogonal; we leave their combination to future work"
- **Pros**: Preserves §6.20 without method overclaim
- **Cons**: Reviewer may ask why it's in this paper then

**Mais recommendation**: **Option 1 (Cut entirely) for NeurIPS 2026**
- §6.20 HEAT is a strong standalone theory
- But mixing it with PPL-focused §6.16-§6.23 dilutes the paper
- Save for a focused long-context paper (ICLR 2027?)
- This keeps the NeurIPS paper under 9 pages and focused

---

## Suggested Decision Matrix

| Section | Recommended | Alternative | Justification |
|---|---|---|---|
| §6.19 MK Fisher | **Option 3 (Restrict)** | Option 1 (Cut) | Preserves theorem content, explanatory only |
| §6.20 HEAT | **Option 1 (Cut)** | Option 3 (Orthogonal) | Focus NeurIPS paper on short-context |

This gives us a **focused NeurIPS paper** with:
- §6.16 Framework (joint)
- §6.16.3 Pre-RoPE PCA theorem (joint, proven)
- §6.19.1 Motivation for anisotropic metrics (cut §6.19.2+)
- §6.21 KVTC comparison (your section)
- §6.22 Verification (joint)
- §6.23 Per-head outlier explanatory framework (mais-primary)
- PCA-Q Natural Alignment discovery (mais-primary, V1 pending)

---

## What I Need from You

Please respond with:

1. **§6.19 decision**: Option 1 / 2 / 3 / other?
   - If Option 2 (demonstrably fix), who will implement? (Mais can help but needs your theoretical guidance.)
   - If Option 3 (restrict), please draft the restriction clause yourself (it's your theorem)

2. **§6.20 decision**: Option 1 / 2 / 3 / other?
   - If keep, please confirm what empirical evidence you have for HEAT vs WF(floor=2)

3. **Timeline commitment**: Can you provide the decision + any necessary edits to §6.19/§6.20 within 3 days (by 2026-04-11)?
   - If yes, we can write the full paper draft by 2026-04-18
   - If no, please let me know and we'll adjust the 28-day schedule

4. **Paper authorship split**: Once §6.19/§6.20 are resolved, we can split sections:
   - mais primary: §6.23, PCA-Q alignment, Theorems A/B/C/G
   - iamseungpil primary: §6.16-§6.18, §6.21 KVTC comparison, §6.22 verification, KVTC + TurboQuant benchmarks
   - Joint: Intro, related work, conclusions

---

## Why This Matters

Codex's review explicitly warns:

> "If §6.19/§6.20 remain and trigger a second retraction cycle, it falls back to reject."

We just did one retraction cycle (the CWF SOTA overclaim). The review is fair but firm: we cannot afford another retraction. The safest path is to explicitly restrict or cut the at-risk sections **now**, before the paper goes to reviewers.

**Option 3 (restrict)** for §6.19 and **Option 1 (cut)** for §6.20 together minimize risk while preserving the strongest contributions:

1. Theorem 6.16.3 (Pre-RoPE PCA, proven) — untouched
2. PCA-Q natural alignment (V1 measurement pending) — new
3. Per-head > Shared PCA (+46.3%) — untouched
4. 5-hypothesis rejection — untouched
5. MSE-PPL gap explanatory framework — intact

This gives us a paper with 5 strong contributions and zero method overclaims.

---

## Status of Related Work

**V1 (Principal angle measurement)**: Running now on mais GPU 1. Expected completion ~15 min. If mean principal angle < 5° across k=16,32,64, paper upgrades to 7/10 accept potential.

**Llama CWF**: Your GPU 2, in progress. Will be used as ablation (not method) in Section 6.4.

**MMLU**: Your GPU 0,1 + mais GPU 0. Critical for Section 6.5.

**V2 (2-bit anomaly restriction)**: Done in PAPER_OUTLINE_UNDERSTANDING.md §3.3 (2026-04-08).

**V4 (KVTC protocol)**: Needs documentation from your V15-1 run.

---

## Response Format

Please reply via commit message or new file in `reports/` with:

```
§6.19: Option [1/2/3/other]
§6.20: Option [1/2/3/other]
Timeline: [commit by YYYY-MM-DD]
Specific edits: [draft text or reference to existing content]
```

---

*Sent: 2026-04-08 (mais Claude session)*
*Related: RETRACTION_MESSAGE_TO_COWORKER_2026-04-08.md, PAPER_OUTLINE_UNDERSTANDING.md*
