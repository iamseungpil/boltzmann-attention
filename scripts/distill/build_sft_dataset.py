#!/usr/bin/env python
"""Facet-guided Distillation — SFT dataset extractor.

Reads shipped tau2-bench teacher trajectories (results/final/*.json), keeps the
TRAIN-split *successful* trajectories, and emits a chat-format JSONL suitable for
multi-turn tool-use LoRA-SFT (assistant-only loss).

System prompt + tool schemas are reconstructed from the live tau2 environment so
they exactly match what an agent would see at eval time:
    system  = SYSTEM_PROMPT.format(domain_policy=env.get_policy(),
                                   agent_instruction=AGENT_INSTRUCTION)
    tools   = [t.openai_schema for t in env.get_tools()]

Each output line:
    {
      "messages": [ {role: system|user|assistant|tool, ...} ],
      "tools":    [ openai function schema, ... ],
      "meta":     {domain, task_id, teacher, trial, reward, variant, source_file}
    }

Variants
--------
  plain   : every train-split trajectory with reward==1.0  (this script's default)
  facet   : additionally requires ontology-clean (precedes/requires/mutex/guardrail
            violations == 0).  Enabled with --ontology; the per-domain ontology
            module under scripts/ontology/ supplies the violation counter.

Run (on remote, tau2 env):
  /home/woori/venvs/seka_env/bin/python scripts/distill/build_sft_dataset.py \
      --domains telecom retail airline \
      --out reports/facet_rft_2026/phase4_distill/sft_data \
      --variant plain
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved relative to the boltzmann-attention checkout that holds the
# tau2-bench submodule + shipped results).  Overridable via --tau2-root.
# ---------------------------------------------------------------------------
DEFAULT_TAU2_ROOT = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench"

# Teacher-result files we trust as distillation sources: standard gpt-4.1 user_sim,
# *default* agent variant only (no-user / op / workflow ablations excluded so the
# policy + interaction protocol match the eval setting).  gpt-4.1-mini shipped
# under the "base" variant tag — accept it too.
ALLOWED_VARIANTS = {"default", "base"}


def _require_tau2(tau2_root: str):
    src = os.path.join(tau2_root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT  # noqa: E402
    from tau2.registry import registry  # noqa: E402

    return registry, SYSTEM_PROMPT, AGENT_INSTRUCTION


def build_system_and_tools(registry, SYSTEM_PROMPT, AGENT_INSTRUCTION, domain: str):
    """Reconstruct the exact agent system prompt + tool schemas for a domain."""
    env = registry.get_env_constructor(domain)()
    policy = env.get_policy()
    tools = [t.openai_schema for t in env.get_tools()]
    system = SYSTEM_PROMPT.format(domain_policy=policy, agent_instruction=AGENT_INSTRUCTION)
    return system, tools


def load_train_ids(tau2_root: str, domain: str) -> set[str]:
    p = Path(tau2_root) / "data" / "tau2" / "domains" / domain / "split_tasks.json"
    splits = json.loads(p.read_text())
    return set(splits["train"])


def iter_teacher_files(tau2_root: str, domain: str):
    """Yield (path, teacher, variant) for shipped result files of `domain`.

    Filename layout: <teacher>_<domain-token>_<variant>_<user-model>_<ntrials>.json
    We require the domain token to equal `domain` exactly (rejects telecom-workflow)
    and the variant to be in ALLOWED_VARIANTS.
    """
    final = Path(tau2_root) / "data" / "tau2" / "results" / "final"
    user_tag = "gpt-4.1-2025-04-14"
    for path in sorted(final.glob("*.json")):
        name = path.stem  # strip .json
        if user_tag not in name:
            continue
        # split off the trailing "_<user-model>_<ntrials>"
        head = name.split(f"_{user_tag}_")[0]  # <teacher>_<domain-token>_<variant>
        parts = head.split("_")
        if len(parts) < 3:
            continue
        variant = parts[-1]
        domain_token = parts[-2]
        teacher = "_".join(parts[:-2])
        if domain_token != domain:
            continue
        if variant not in ALLOWED_VARIANTS:
            continue
        yield path, teacher, variant


# ---------------------------------------------------------------------------
# Trajectory -> chat conversion
# ---------------------------------------------------------------------------
def convert_messages(raw_msgs: list[dict]) -> list[dict] | None:
    """Convert recorded tau2 messages to OpenAI/HF chat format.

    Recorded roles: assistant | user | tool.
      assistant: {role, content, tool_calls:[{id,name,arguments(dict),requestor}], ...}
      tool:      {id, role, content, requestor, error, ...}
      user:      {role, content, tool_calls?, ...}

    DUAL-CONTROL DOMAINS (telecom): both the agent AND the user simulator hold
    tools.  The agent we distill never sees the user's tool calls or their
    results — only the user's natural-language messages.  So we keep only
    requestor=="assistant" tool activity and the user's text turns:
      - user message tool_calls (requestor=="user")  -> dropped
      - tool responses with requestor=="user"        -> dropped
      - user message with empty content (tool-call-only turn) -> dropped
    Single-control domains (retail/airline) have no user tools -> no-op there.

    Returns None if the trajectory is malformed (e.g. an agent-side tool
    response with no matching agent call) so the caller can skip it.
    """
    out: list[dict] = []
    open_call_ids: set[str] = set()
    for m in raw_msgs:
        role = m.get("role")
        if role == "user":
            # drop user-side tool_calls (agent-invisible); keep text turns only
            content = m.get("content")
            if content:
                out.append({"role": "user", "content": content})
        elif role == "assistant":
            content = m.get("content")
            tcs = m.get("tool_calls") or []
            msg: dict = {"role": "assistant", "content": content if content is not None else ""}
            if tcs:
                conv_tcs = []
                for tc in tcs:
                    cid = tc.get("id")
                    if cid is None:
                        return None
                    open_call_ids.add(cid)
                    args = tc.get("arguments", {})
                    conv_tcs.append(
                        {
                            "id": cid,
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                # canonical OpenAI: arguments is a JSON string
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    )
                msg["tool_calls"] = conv_tcs
            out.append(msg)
        elif role == "tool":
            if m.get("requestor") != "assistant":
                # user-side tool response: the agent never sees it -> drop
                continue
            cid = m.get("id")
            if cid not in open_call_ids:
                # agent-side tool response with no preceding call -> malformed
                return None
            open_call_ids.discard(cid)
            content = m.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            out.append({"role": "tool", "tool_call_id": cid, "content": content})
        else:
            # unknown role (system shouldn't appear in recorded transcript) -> skip msg
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["telecom", "retail", "airline"])
    ap.add_argument("--tau2-root", default=DEFAULT_TAU2_ROOT)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--variant", choices=["plain", "facet"], default="plain")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--max-per-task",
        type=int,
        default=0,
        help="cap successful trajectories kept per task_id (0 = no cap)",
    )
    ap.add_argument("--ontology", action="store_true",
                    help="(facet variant) require ontology-clean trajectories")
    args = ap.parse_args()

    registry, SYSTEM_PROMPT, AGENT_INSTRUCTION = _require_tau2(args.tau2_root)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_path = out_dir / f"sft_{args.variant}_{args.split}_all.jsonl"
    combined_f = combined_path.open("w", encoding="utf-8")

    grand = Counter()
    manifest = {"variant": args.variant, "split": args.split, "domains": {}}

    for domain in args.domains:
        system, tools = build_system_and_tools(
            registry, SYSTEM_PROMPT, AGENT_INSTRUCTION, domain
        )
        split_ids = load_train_ids(args.tau2_root, domain) if args.split == "train" \
            else set(json.loads((Path(args.tau2_root) / "data" / "tau2" / "domains" / domain / "split_tasks.json").read_text())[args.split])

        per_task = defaultdict(int)
        stats = Counter()
        dom_path = out_dir / f"sft_{args.variant}_{args.split}_{domain}.jsonl"
        dom_f = dom_path.open("w", encoding="utf-8")

        for path, teacher, variant in iter_teacher_files(args.tau2_root, domain):
            data = json.loads(path.read_text())
            sims = data.get("simulations", [])
            for s in sims:
                tid = s.get("task_id")
                stats["seen"] += 1
                if tid not in split_ids:
                    continue
                stats["in_split"] += 1
                reward = (s.get("reward_info") or {}).get("reward", 0.0)
                if reward < 0.999:
                    continue
                stats["success"] += 1
                if args.max_per_task and per_task[tid] >= args.max_per_task:
                    stats["capped"] += 1
                    continue
                conv = convert_messages(s.get("messages") or [])
                if conv is None:
                    stats["malformed"] += 1
                    continue
                # ontology filter hook (facet variant) — wired in step 2
                if args.variant == "facet" and args.ontology:
                    # placeholder: violations counter to be supplied by
                    # scripts/ontology/tau2_<domain>_ontology.py
                    pass
                per_task[tid] += 1
                rec = {
                    "messages": [{"role": "system", "content": system}] + conv,
                    "tools": tools,
                    "meta": {
                        "domain": domain,
                        "task_id": tid,
                        "teacher": teacher,
                        "variant_tag": variant,
                        "trial": s.get("trial"),
                        "reward": reward,
                        "source_file": path.name,
                    },
                }
                line = json.dumps(rec, ensure_ascii=False)
                dom_f.write(line + "\n")
                combined_f.write(line + "\n")
                stats["kept"] += 1

        dom_f.close()
        grand.update(stats)
        manifest["domains"][domain] = {
            "kept": stats["kept"],
            "success": stats["success"],
            "in_split": stats["in_split"],
            "malformed": stats["malformed"],
            "capped": stats["capped"],
            "n_tools": len(tools),
            "n_tasks_covered": len(per_task),
            "split_size": len(split_ids),
            "out_file": str(dom_path),
        }
        print(f"[{domain}] kept={stats['kept']} success={stats['success']} "
              f"in_split={stats['in_split']} malformed={stats['malformed']} "
              f"tasks_covered={len(per_task)}/{len(split_ids)} tools={len(tools)}")

    combined_f.close()
    manifest["total_kept"] = grand["kept"]
    (out_dir / f"manifest_{args.variant}_{args.split}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"\nTOTAL kept={grand['kept']} -> {combined_path}")
    print(f"manifest -> {out_dir / f'manifest_{args.variant}_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
