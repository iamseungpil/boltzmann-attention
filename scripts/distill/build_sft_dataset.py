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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ontology_filter import agent_tool_sequence, count_ontology_violations  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (resolved relative to the boltzmann-attention checkout that holds the
# tau2-bench submodule + shipped results).  Overridable via --tau2-root.
# ---------------------------------------------------------------------------
DEFAULT_TAU2_ROOT = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench"
# per-domain ontology data modules live in scripts/ontology/ (sibling of this dir)
DEFAULT_ONT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ontology")
)

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
# Auxiliary teacher: our OpenRouter Sonnet-4.6 runs (NOT primary distillation).
# These live under data/simulations/... (not in git) and are used as a stronger-
# teacher arm + a user_sim ablation (gpt-4o-mini vs gpt-4.1).  Keep them OUT of
# the shipped `plain` set so those comparisons stay un-confounded.
# Only the 4 valid runs are listed; the buggy/smoke runs are intentionally omitted:
#   gen_retail_...162303 (--task-set bug: telecom tasks leaked into a retail run),
#   gen_airline_...162622 (2 sims), airline_smoke*, calib_sonnet (early failures).
# ---------------------------------------------------------------------------
AUX_SONNET_SUBDIR = "data/simulations/reports/facet_rft_2026/phase4_distill"
AUX_SONNET_SOURCES = [
    # (domain, user_sim, relpath-under-AUX_SONNET_SUBDIR)
    ("telecom", "gpt-4o-mini", "gen_telecom_train_sonnet_20260529_154937/B0_telecom_train.json/results.json"),
    ("telecom", "gpt-4.1",     "gen_telecom_train_sonnet_u41_20260529_165055/B0_telecom_train.json/results.json"),
    ("retail",  "gpt-4o-mini", "gen_retail_train_sonnet_20260529_162430/B0_retail_train.json/results.json"),
    ("airline", "gpt-4o-mini", "gen_airline_train_sonnet_20260529_163445/B0_airline_train.json/results.json"),
]


def _user_sim_slug(user_sim: str) -> str:
    return user_sim.replace("-", "").replace(".", "")


def iter_aux_sonnet(tau2_root: str, domain: str):
    """Yield (path, teacher, user_sim) for the Sonnet-4.6 aux runs of `domain`.

    `teacher` is read from the run's info block when present, else "sonnet-4.6".
    """
    base = Path(tau2_root) / AUX_SONNET_SUBDIR
    for dom, user_sim, rel in AUX_SONNET_SOURCES:
        if dom != domain:
            continue
        path = base / rel
        if not path.exists():
            continue
        teacher = "sonnet-4.6"
        try:
            info = json.loads(path.read_text()).get("info") or {}
            # info may carry the agent llm id; fall back gracefully
            agent_llm = (info.get("agent_info") or {}).get("llm") or info.get("agent_llm")
            if agent_llm:
                teacher = str(agent_llm)
        except Exception:
            pass
        yield path, teacher, user_sim


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


def _split_ids(tau2_root: str, domain: str, split: str) -> set[str]:
    p = Path(tau2_root) / "data" / "tau2" / "domains" / domain / "split_tasks.json"
    return set(json.loads(p.read_text())[split])


def build_group(out_dir, group_key, domain, system, tools, sources, split_ids, args, combined_f):
    """Build one output group.

    sources: list of (path, teacher, variant_tag, user_sim).
    Returns (stats: Counter[int], extras: dict).
    """
    per_task = defaultdict(int)
    stats = Counter()
    teachers = set()
    if args.source == "shipped":
        fname = f"sft_{args.variant}_{args.split}_{group_key}.jsonl"
    else:
        fname = f"sft_{args.variant}_{args.source}_{args.split}_{group_key}.jsonl"
    dom_path = out_dir / fname
    with dom_path.open("w", encoding="utf-8") as dom_f:
        for path, teacher, variant_tag, user_sim in sources:
            data = json.loads(path.read_text())
            for s in data.get("simulations", []):
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
                # ontology filter (facet variant): keep only ontology-clean trajectories
                onto_total, onto_breakdown = 0, {}
                if args.variant == "facet" and args.ontology:
                    tool_seq = agent_tool_sequence(conv)
                    onto_total, onto_breakdown, _ = count_ontology_violations(
                        domain, tool_seq, args.ont_dir
                    )
                    if onto_total > 0:
                        stats["onto_violation"] += 1
                        for k in onto_breakdown:
                            stats[f"onto_{k}"] += onto_breakdown[k]
                        continue
                per_task[tid] += 1
                teachers.add(teacher)
                meta = {
                    "domain": domain,
                    "task_id": tid,
                    "teacher": teacher,
                    "user_sim": user_sim,
                    "source": args.source,
                    "variant_tag": variant_tag,
                    "trial": s.get("trial"),
                    "reward": reward,
                    "source_file": path.name,
                }
                if args.variant == "facet" and args.ontology:
                    meta["ontology_violations"] = onto_total  # 0 for kept records
                rec = {
                    "messages": [{"role": "system", "content": system}] + conv,
                    "tools": tools,
                    "meta": meta,
                }
                line = json.dumps(rec, ensure_ascii=False)
                dom_f.write(line + "\n")
                if combined_f is not None:
                    combined_f.write(line + "\n")
                stats["kept"] += 1
    extras = {
        "tasks_covered": len(per_task),
        "teachers": sorted(teachers),
        "out_file": str(dom_path),
    }
    return stats, extras


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["telecom", "retail", "airline"])
    ap.add_argument("--tau2-root", default=DEFAULT_TAU2_ROOT)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--variant", choices=["plain", "facet"], default="plain")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--source",
        choices=["shipped", "aux_sonnet"],
        default="shipped",
        help="shipped=free multi-teacher final/ (primary distillation); "
             "aux_sonnet=our OpenRouter Sonnet-4.6 runs (auxiliary: stronger-teacher "
             "+ user_sim ablation; grouped per user_sim, kept separate from shipped)",
    )
    ap.add_argument(
        "--max-per-task",
        type=int,
        default=0,
        help="cap successful trajectories kept per task_id (0 = no cap)",
    )
    ap.add_argument("--ontology", action="store_true",
                    help="(facet variant) require ontology-clean trajectories")
    ap.add_argument("--ont-dir", default=DEFAULT_ONT_DIR,
                    help="dir with tau2_<domain>_ontology.py modules")
    args = ap.parse_args()

    registry, SYSTEM_PROMPT, AGENT_INSTRUCTION = _require_tau2(args.tau2_root)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Assemble output groups: (group_key, domain, sources[(path,teacher,variant_tag,user_sim)]).
    # shipped -> one group per domain; aux_sonnet -> one group per (domain, user_sim).
    groups = []
    for domain in args.domains:
        if args.source == "shipped":
            srcs = [(p, t, v, "gpt-4.1") for (p, t, v) in iter_teacher_files(args.tau2_root, domain)]
            if srcs:
                groups.append((domain, domain, srcs))
        else:  # aux_sonnet
            by_us = defaultdict(list)
            for (p, t, us) in iter_aux_sonnet(args.tau2_root, domain):
                by_us[us].append((p, t, "aux", us))
            for us, srcs in by_us.items():
                groups.append((f"{domain}_{_user_sim_slug(us)}", domain, srcs))

    if not groups:
        print(f"no sources found for source={args.source} domains={args.domains}")
        return 1

    if args.source == "shipped":
        combined_path = out_dir / f"sft_{args.variant}_{args.split}_all.jsonl"
        man_path = out_dir / f"manifest_{args.variant}_{args.split}.json"
    else:
        combined_path = out_dir / f"sft_{args.variant}_{args.source}_{args.split}_all.jsonl"
        man_path = out_dir / f"manifest_{args.variant}_{args.source}_{args.split}.json"
    combined_f = combined_path.open("w", encoding="utf-8")

    grand = Counter()
    manifest = {"variant": args.variant, "source": args.source, "split": args.split, "groups": {}}
    sys_cache = {}

    for group_key, domain, srcs in groups:
        if domain not in sys_cache:
            sys_cache[domain] = build_system_and_tools(
                registry, SYSTEM_PROMPT, AGENT_INSTRUCTION, domain
            )
        system, tools = sys_cache[domain]
        split_ids = _split_ids(args.tau2_root, domain, args.split)
        stats, extras = build_group(
            out_dir, group_key, domain, system, tools, srcs, split_ids, args, combined_f
        )
        grand.update(stats)
        manifest["groups"][group_key] = {
            "domain": domain,
            "kept": stats["kept"],
            "success": stats["success"],
            "in_split": stats["in_split"],
            "malformed": stats["malformed"],
            "capped": stats["capped"],
            "n_tools": len(tools),
            "n_tasks_covered": extras["tasks_covered"],
            "split_size": len(split_ids),
            "teachers": extras["teachers"],
            "user_sims": sorted({s[3] for s in srcs}),
            "out_file": extras["out_file"],
        }
        if args.variant == "facet" and args.ontology:
            manifest["groups"][group_key]["onto_violation_dropped"] = stats["onto_violation"]
            manifest["groups"][group_key]["onto_breakdown"] = {
                k[5:]: stats[k] for k in ("onto_mutex", "onto_precedes", "onto_requires", "onto_guardrail")
                if stats[k]
            }
        onto_note = (f" onto_dropped={stats['onto_violation']}"
                     if (args.variant == "facet" and args.ontology) else "")
        print(f"[{group_key}] kept={stats['kept']} success={stats['success']} "
              f"in_split={stats['in_split']} malformed={stats['malformed']}{onto_note} "
              f"tasks_covered={extras['tasks_covered']}/{len(split_ids)} tools={len(tools)}")

    combined_f.close()
    manifest["total_kept"] = grand["kept"]
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nTOTAL kept={grand['kept']} -> {combined_path}")
    print(f"manifest -> {man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
