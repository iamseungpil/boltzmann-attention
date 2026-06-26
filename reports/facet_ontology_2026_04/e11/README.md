# E11 cross-domain transfer — diagnostic preserve (incomplete)

**Status (2026-05-14):** abandoned mid-sweep. Cohort closed by the
Facet-Aware Verifier-Guided RFT pivot
(`memory/project_research_pivot_2026_05_14.md`). One of six planned sweep
cells ran (qwen, retail → telecom, N=64); the rest were never executed.
Files in this directory are preserved here because the **one cell that did
run is itself diagnostic for the pivot's central claim** (§1.2 of the
pivot memo), not because E11 will be continued as designed.

## What ran

| file | source | target | model | N | wall |
|---|---|---|---|---|---|
| `qwen_retail_to_telecom_n64.json` | tau2_retail | tau2_telecom | Qwen 2.5-7B-Instruct | 64 | (logged in `reports/.../logs/` on remote disk) |
| `smoke_qwen_retail_to_telecom_n4.json` | tau2_retail | tau2_telecom | Qwen 2.5-7B-Instruct | 4 | smoke |

What was *not* run (5 cells): qwen retail→airline, telecom→retail,
telecom→airline, airline→retail, airline→telecom; and the entire
Llama leg.

## Result (qwen retail→telecom, N=64)

F1_macro mean per condition:

| condition | F1 mean | note |
|---|---|---|
| `nl_full` (target NL prompt) | **0.049** | NL baseline on tau2 telecom multi-tool queries |
| `nl_full_source` | 0.000 | source's tool list, target's query — expected sanity floor |
| `facet_full` (target typed schema) | **0.220** | ≈ 4.5× nl_full |
| `facet_xfer` (target tools w/ source domain label) | 0.134 | partial transfer signal |
| `facet_compact` (no descriptions) | 0.134 | description loss confirms E10b pattern |
| `noprompt` | 0.000 | floor |

## Why preserved

The qwen `nl_full` 0.049 vs `facet_full` 0.220 gap on tau2 telecom is
**much larger than the +9.7% MetaTool gap reported in E10** (commit
61fdb7e). This is *not* the headline result for the new pivot, but it is
consistent with — and provides early support for — the pivot memo §1.2
framing that multi-step tool calling on τ²-bench-style queries is
effectively long-context with high schema-enforcement bandwidth
requirements, where NL prompts collapse first. Re-using these numbers
later requires bigger N + CI; treat as anecdotal until reproduced.

## Why not extended

Continuing the 6×2 = 12 sweep is not on the path to the pivot's primary
contribution (Facet-Aware Verifier-Guided RFT on τ²-bench pass^1, see
`memory/project_research_pivot_2026_05_14.md` §5 sprint plan). The
"facet ontology as universal IR" framing in pivot §3.1 also reframes
cross-domain transfer as a property of the IR (verifier/sampler/curriculum)
rather than as a prompt-format property to be measured directly, so the
specific E11 design no longer maps cleanly to the new contribution.

If transfer ever becomes a paper requirement, the relevant
infrastructure is in `scripts/rank_replaceability/facet_eval.py`
(--task, --source-schema, `build_facet_xfer_prompt`) and
`scripts/rank_replaceability/e11_sweep.sh`.
