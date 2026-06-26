# ⚠️ COST INCIDENT 2026-06-16 — ~$600 OpenRouter drain via Claude on the shared key

## What happened
The OpenRouter key labeled **`facet-rft-tau2-user-sim`** was billed ~$600, split across
**Claude Sonnet 4.6 (~$246)** and **Claude Opus 4.8 (~$132)** — concentrated June 14+, esp. today.
The user-sim is supposed to be **gpt-4.1**.

## Root cause (structural)
- That OpenRouter key is **shared for ALL τ² OpenRouter calls — agent + user-sim + judge**, not just
  the user-sim. The key NAME is misleading.
- Claude is **~15–30× the price of gpt-4.1**, and agentic τ² runs are token-heavy (multi-turn
  trajectories × tasks × trials × long tool-schema prompts). So a single run with
  `--agent_llm anthropic/claude-opus-4.8` (or a teacher arm with `claude-sonnet-4.6`) drains hundreds of $.
- No per-key spend cap → unbounded.

## What was NOT the cause (verified)
- The committed τ² pipeline (`coupling_eval.sh`, `t2_run_gated.py`, `driver_frontier_*`, `prov_eval.sh`,
  taskbench drivers) all correctly pass `--user_llm openrouter/openai/gpt-4.1`. **user-sim config is correct.**
- The M-σ v4 experiments (exp0 / factorial / overnight / `m_sigma_transfer_eval_v4.py`) are **local vLLM only**
  (localhost) — zero OpenRouter calls.
- Coworker's committed node scripts (`node_run_ma_72b.sh`, `node_run_factorial_iso.sh`) are **local-only**.

## Most likely source
An **ad-hoc / node-local run** of a τ² agentic driver with `--agent_llm`/`--user_llm` overridden to a
Claude model (Sonnet 4.6 = aux "stronger-teacher"; Opus 4.8 = recent), OR tau2-bench's NATIVE harness
with its default Claude agent. Not in the committed repo. **The OpenRouter dashboard Activity/Requests
log (filter model=Opus 4.8) pins the exact timestamps + source IP/app.**

## Immediate actions
1. **[USER — only you can] Rotate/disable the OpenRouter key + set a hard per-key spend cap** on the
   dashboard. This is the only thing that stops ANY node (incl. coworker's) using the shared key NOW.
2. **[DONE — code guard]** `t2_run_gated.py` now **REFUSES** any `--user_llm`/`--agent_llm` containing
   anthropic/claude/opus/sonnet/haiku unless `--allow-frontier` is passed (and prints a loud warning).
   Covers all drivers that route through the runner.
3. **[coworker] Use only gpt-4.1 (or local) for τ² OpenRouter calls.** No Claude agent/user/teacher arms
   without an explicit budget + spend cap.

## Rule going forward
τ² OpenRouter calls = **gpt-4.1 only**. Claude/frontier arms require: spend cap set FIRST +
`--allow-frontier` + explicit sign-off. Prefer a **separate, capped key** for any frontier experiment.
