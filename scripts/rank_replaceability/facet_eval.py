#!/usr/bin/env python3
"""facet_eval.py — E10 of EXPERIMENT_PLAN_v28 facet/ontology framework.

NL prompt vs ontology facet prompt comparison on MetaTool ST4 multi-tool
selection. NO intervention, NO model surgery -- only prompt-format variation.

Conditions (per query):
  nl_full         : current FC_SYSTEM_TEMPLATE + tool name list (anchor, no desc)
  nl_with_desc    : NL system prompt + tool name + short description (length-matched with facet_full)
  facet_full      : typed schema (name | action | domain : short_desc), length-matched with nl_with_desc
  facet_compact   : compressed facet (name|action_abbrev|domain), no desc, length-matched with nl_full
  list_only       : tool names only (sanity baseline, no system framing)
  noprompt        : user query only (bottom)

Reports F1, exact, precision, recall per condition + token length statistics
+ first-step KL between conditions (optional, future).

Usage:
  /home/woori/venvs/seka_env/bin/python3.12 facet_eval.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --schema data/facet_schemas/metatool_st4.yaml \\
      --max-samples 64 --device cuda:0 \\
      --conditions nl_full,facet_full,facet_compact,list_only,noprompt \\
      --out reports/facet_ontology_2026_04/qwen_e10_n64.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from measure_phi_rank import load_metatool_st4, load_tau2  # type: ignore
from intervention_metatool_eval import (  # type: ignore
    FC_SYSTEM_TEMPLATE,
    build_full_prompt,
    build_noprompt,
    extract_tool_names,
    compute_metrics,
    generate_text,
)


# =============================================================================
# Tiny YAML loader (avoid PyYAML dep -- our schema is flat)
# =============================================================================

def bootstrap_ci(
    values: List[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple:
    """Percentile bootstrap CI of the mean. Returns (lo, hi)."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    return (
        float(np.quantile(boot_means, alpha / 2)),
        float(np.quantile(boot_means, 1 - alpha / 2)),
    )


def load_facet_schema(path: str) -> Dict[str, Dict[str, str]]:
    """Parse the flat YAML produced by extract_facet_schema_metatool.py.

    Expected structure:
        tools:
          ToolName:
            action: read
            domain: news
            desc_short: "..."
    """
    schema: Dict[str, Dict[str, str]] = {}
    in_tools = False
    cur_name = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip().startswith("#") or not line.strip():
                continue
            if line.startswith("tools:"):
                in_tools = True
                continue
            if not in_tools:
                continue
            # Tool entry header: 2-space indent, ends with colon
            if line.startswith("  ") and not line.startswith("    "):
                name = line.strip().rstrip(":").strip()
                # Unquote if quoted
                if (name.startswith('"') and name.endswith('"')) or (
                    name.startswith("'") and name.endswith("'")
                ):
                    name = name[1:-1]
                cur_name = name
                schema[cur_name] = {}
                continue
            # Field: 4-space indent
            if line.startswith("    ") and cur_name:
                kv = line.strip()
                if ":" in kv:
                    k, _, v = kv.partition(":")
                    v = v.strip()
                    # Unquote string values
                    if v.startswith('"') and v.endswith('"'):
                        try:
                            v = json.loads(v)
                        except json.JSONDecodeError:
                            v = v[1:-1]
                    schema[cur_name][k.strip()] = v
    return schema


# =============================================================================
# Prompt builders
# =============================================================================

NL_WITH_DESC_SYSTEM_TEMPLATE = (
    "You are a tool-selection agent. Given a user query, emit ONE OR MORE "
    "<tool_call> blocks naming tools from this list:\n"
    "{tool_rows}\n"
    "Format each tool call exactly as:\n"
    '<tool_call>{{"name": "ToolName", "arguments": {{}}}}</tool_call>\n'
    "Emit MULTIPLE blocks if the query needs multiple tools. "
    "Output ONLY tool_call blocks; no explanation."
)


FACET_FULL_SYSTEM_TEMPLATE = (
    "You are a tool-selection agent. Choose ALL applicable tools. "
    "Available tools — format: tool_name | action | domain : description\n"
    "{tool_rows}\n"
    "Format each call exactly as:\n"
    '<tool_call>{{"name": "ToolName", "arguments": {{}}}}</tool_call>\n'
    "Emit MULTIPLE blocks if the query needs multiple tools. "
    "Output ONLY tool_call blocks; no explanation."
)


FACET_COMPACT_SYSTEM_TEMPLATE = (
    "Tool-selection agent. TOOLS: {tool_rows}\n"
    'Emit <tool_call>{{"name":"X","arguments":{{}}}}</tool_call> per tool. '
    "Multiple if needed. Only tool_call blocks."
)


LIST_ONLY_SYSTEM_TEMPLATE = (
    "Tools: [{tools}]. "
    'Emit <tool_call>{{"name":"X","arguments":{{}}}}</tool_call> blocks.'
)

# D0 fix-(마): nl_full with same minimal framing as list_only (deconfound system-prompt richness)
NL_SYSLESS_SYSTEM_TEMPLATE = (
    "You are a tool-selection agent. "
    "Tools: [{tools}]. "
    'Emit <tool_call>{{"name":"X","arguments":{{}}}}</tool_call> blocks. '
    "Multiple if needed. Only tool_call blocks."
)


# E10b -- anonymized tool names. Tools are referenced by placeholder IDs
# (T1..Tn) so that any task knowledge the model has from the natural
# tool-name strings is screened out. The real names appear ONLY in a
# mapping at the bottom of the system prompt, forcing the model to read
# the mapping to recover the catalogue.
LIST_ANON_SYSTEM_TEMPLATE = (
    "Tools (by ID): [{anon_ids}]. ID-to-name mapping: {mapping}. "
    'Emit <tool_call>{{"name":"X","arguments":{{}}}}</tool_call> blocks '
    "using the real tool names from the mapping."
)


# Same idea but with the facet annotation kept on the placeholder IDs.
# Separates "does the model gain from typed format?" from "does the
# model gain from natural-language tool names?".
FACET_ANON_SYSTEM_TEMPLATE = (
    "You are a tool-selection agent. Choose ALL applicable tools. "
    "Available tools — format: tool_id | action | domain : description\n"
    "{tool_rows}\n"
    "ID-to-name mapping (use real names in tool_call): {mapping}\n"
    "Format each call exactly as:\n"
    '<tool_call>{{"name": "ToolName", "arguments": {{}}}}</tool_call>\n'
    "Emit MULTIPLE blocks if the query needs multiple tools. "
    "Output ONLY tool_call blocks; no explanation."
)


# Cap on per-tool description length (chars) for length-matched comparisons.
DESC_MAX_CHARS = 60


def _trim_desc(desc: str, n: int = DESC_MAX_CHARS) -> str:
    desc = desc.strip()
    if len(desc) <= n:
        return desc
    return desc[: n - 1].rstrip() + "…"


def build_nl_with_desc_prompt(
    tokenizer, query: str, tools: List[str], schema: Dict[str, Dict[str, str]]
) -> str:
    """NL prompt with per-tool short description (length-matched with facet_full)."""
    rows = []
    for t in tools:
        s = schema.get(t, {})
        desc = _trim_desc(s.get("desc_short", ""))
        rows.append(f"- {t}: {desc}")
    sys_msg = NL_WITH_DESC_SYSTEM_TEMPLATE.format(tool_rows="\n".join(rows))
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def build_facet_full_prompt(
    tokenizer, query: str, tools: List[str], schema: Dict[str, Dict[str, str]]
) -> str:
    rows = []
    for t in tools:
        s = schema.get(t, {})
        action = s.get("action", "read")
        domain = s.get("domain", "general")
        desc = _trim_desc(s.get("desc_short", ""))
        rows.append(f"- {t} | {action} | {domain} : {desc}")
    sys_msg = FACET_FULL_SYSTEM_TEMPLATE.format(tool_rows="\n".join(rows))
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def build_facet_compact_prompt(
    tokenizer, query: str, tools: List[str], schema: Dict[str, Dict[str, str]]
) -> str:
    # Compact: ToolName|a|d  ToolName|a|d  ...
    rows = []
    for t in tools:
        s = schema.get(t, {})
        action = s.get("action", "read")[:1]  # 1 char abbreviation: r/s/c/m/...
        domain = s.get("domain", "general")
        rows.append(f"{t}|{action}|{domain}")
    sys_msg = FACET_COMPACT_SYSTEM_TEMPLATE.format(tool_rows=" ".join(rows))
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def build_list_only_prompt(tokenizer, query: str, tools: List[str]) -> str:
    sys_msg = LIST_ONLY_SYSTEM_TEMPLATE.format(tools=", ".join(tools))
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def build_nl_sysless_prompt(tokenizer, query: str, tools: List[str]) -> str:
    """D0 fix-(마): nl_full with minimal framing identical to list_only (same template structure).
    Isolates effect of tool-list format from system-prompt richness."""
    sys_msg = NL_SYSLESS_SYSTEM_TEMPLATE.format(tools=", ".join(tools))
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def _anon_ids(n: int) -> List[str]:
    """T1, T2, ..., Tn"""
    return [f"T{i+1}" for i in range(n)]


def _mapping_str(ids: List[str], tools: List[str]) -> str:
    """T1=NewsTool, T2=WeatherTool, ..."""
    return ", ".join(f"{i}={t}" for i, t in zip(ids, tools))


def build_list_anon_prompt(tokenizer, query: str, tools: List[str]) -> str:
    """Tool names hidden behind T1..Tn placeholders; mapping at bottom of system."""
    ids = _anon_ids(len(tools))
    mapping = _mapping_str(ids, tools)
    sys_msg = LIST_ANON_SYSTEM_TEMPLATE.format(
        anon_ids=", ".join(ids), mapping=mapping
    )
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def build_facet_anon_prompt(
    tokenizer, query: str, tools: List[str], schema: Dict[str, Dict[str, str]]
) -> str:
    """facet_full but tool names replaced with T1..Tn placeholders.
    Real names appear ONLY in the mapping line (so the placeholder Ti carries
    only the typed facet annotation, not the natural tool name semantics)."""
    ids = _anon_ids(len(tools))
    rows = []
    for tid, t in zip(ids, tools):
        s = schema.get(t, {})
        action = s.get("action", "read")
        domain = s.get("domain", "general")
        desc = _trim_desc(s.get("desc_short", ""))
        rows.append(f"- {tid} | {action} | {domain} : {desc}")
    mapping = _mapping_str(ids, tools)
    sys_msg = FACET_ANON_SYSTEM_TEMPLATE.format(
        tool_rows="\n".join(rows), mapping=mapping
    )
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


# =============================================================================
# E11 cross-domain transfer prompt builders
# =============================================================================

def build_nl_full_source_prompt(
    tokenizer, query: str, source_tools: List[str]
) -> str:
    """Target query but source's tool list as candidates -- expected F1 ~ 0
    (tool-name mismatch). Reuses the standard NL FC system template."""
    return build_full_prompt(tokenizer, query, source_tools)


def build_facet_xfer_prompt(
    tokenizer,
    query: str,
    target_tools: List[str],
    target_schema: Dict[str, Dict[str, str]],
    source_domain: str,
) -> str:
    """Target tools relabeled with source domain string and given a synthetic
    name-only description (target's canonical desc stripped). Action labels
    come from target_schema (the same universal classifier that produced the
    source schema). Tests whether the typed-format gain transfers when the
    domain label is mismatched and target descriptions are unavailable."""
    rows = []
    for t in target_tools:
        s = target_schema.get(t, {})
        action = s.get("action", "read")
        domain = source_domain
        words = t.replace("_", " ").strip().lower()
        desc_short = _trim_desc(f"{source_domain}: {words}")
        rows.append(f"- {t} | {action} | {domain} : {desc_short}")
    sys_msg = FACET_FULL_SYSTEM_TEMPLATE.format(tool_rows="\n".join(rows))
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--metatool-path",
        default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json",
    )
    p.add_argument("--schema", required=True, help="target facet schema YAML path")
    p.add_argument(
        "--task",
        default="metatool_st4",
        choices=["metatool_st4", "tau2_retail", "tau2_telecom", "tau2_airline"],
        help="evaluation task: queries + GT come from this dataset",
    )
    p.add_argument(
        "--source-schema",
        default=None,
        help="source-domain facet schema YAML (E11 cross-domain transfer; "
        "required when conditions include nl_full_source / facet_xfer)",
    )
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument(
        "--bootstrap-n-boot",
        type=int,
        default=1000,
        help="bootstrap resamples for 95%% CI on per-condition F1/precision/recall. "
        "Set 0 to skip (CI fields become NaN).",
    )
    p.add_argument(
        "--conditions",
        default="nl_full,nl_with_desc,facet_full,facet_compact,list_only,noprompt",
        help="comma-separated subset of {nl_full, nl_with_desc, facet_full, "
        "facet_compact, list_only, list_anon, facet_anon, noprompt, "
        "nl_sysless, nl_full_source, facet_xfer}",
    )
    # D0 assumption fixes
    p.add_argument(
        "--d0",
        action="store_true",
        default=False,
        help="Enable all D0 assumption fixes: --query-mode native --gt-mode unique",
    )
    p.add_argument(
        "--query-mode",
        default="dict",
        choices=["dict", "native"],
        help="(나) dict=legacy str(instructions_dict); native=reason_for_call+known_info",
    )
    p.add_argument(
        "--gt-mode",
        default="full",
        choices=["full", "unique"],
        help="(가) full=legacy multiset trajectory; unique=ordered dedup of tool names",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    # Apply --d0 shorthand
    if args.d0:
        args.query_mode = "native"
        args.gt_mode = "unique"
    valid = {
        "nl_full",
        "nl_with_desc",
        "facet_full",
        "facet_compact",
        "list_only",
        "nl_sysless",
        "list_anon",
        "facet_anon",
        "noprompt",
        "nl_full_source",
        "facet_xfer",
    }
    for c in conditions:
        if c not in valid:
            raise ValueError(f"unknown condition '{c}' (valid: {valid})")

    schema = load_facet_schema(args.schema)
    print(f"[schema] {len(schema)} tools loaded from {args.schema}", flush=True)

    # Source schema (E11 cross-domain only).
    source_schema: Dict[str, Dict[str, str]] = {}
    source_tools_list: List[str] = []
    source_domain: str = ""
    xfer_conds = {"nl_full_source", "facet_xfer"}
    needed_xfer = xfer_conds & set(conditions)
    if needed_xfer:
        if not args.source_schema:
            raise ValueError(
                f"conditions {sorted(needed_xfer)} require --source-schema"
            )
        source_schema = load_facet_schema(args.source_schema)
        source_tools_list = sorted(source_schema.keys())
        # Source domain label: read any tool's domain field (uniform per file).
        if source_schema:
            source_domain = next(iter(source_schema.values())).get("domain", "general")
        print(
            f"[source] {len(source_tools_list)} tools, domain='{source_domain}' "
            f"from {args.source_schema}",
            flush=True,
        )

    if args.task == "metatool_st4":
        items = load_metatool_st4(args.metatool_path, args.max_samples, full_schema=False)
    else:
        domain = args.task.split("_", 1)[1]
        items = load_tau2(
            domain,
            args.max_samples,
            full_schema=False,
            query_mode=args.query_mode,
            gt_mode=args.gt_mode,
        )
    print(f"[data] task={args.task} N={len(items)}", flush=True)

    print(f"[model] loading {args.model}", flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    metrics = {c: [] for c in conditions}
    generations = {c: [] for c in conditions}
    prompt_lengths = {c: [] for c in conditions}  # token count

    t0 = time.time()
    for i, item in enumerate(items):
        query = item["query"]
        tools = item["candidates"]
        gt = item["gt"]
        if isinstance(gt, str):
            gt_list = [gt]
        elif isinstance(gt, list):
            gt_list = [str(g) for g in gt]
        else:
            gt_list = []

        prompts = {}
        if "nl_full" in conditions:
            prompts["nl_full"] = build_full_prompt(tokenizer, query, tools)
        if "nl_with_desc" in conditions:
            prompts["nl_with_desc"] = build_nl_with_desc_prompt(tokenizer, query, tools, schema)
        if "facet_full" in conditions:
            prompts["facet_full"] = build_facet_full_prompt(tokenizer, query, tools, schema)
        if "facet_compact" in conditions:
            prompts["facet_compact"] = build_facet_compact_prompt(tokenizer, query, tools, schema)
        if "list_only" in conditions:
            prompts["list_only"] = build_list_only_prompt(tokenizer, query, tools)
        if "nl_sysless" in conditions:
            prompts["nl_sysless"] = build_nl_sysless_prompt(tokenizer, query, tools)
        if "list_anon" in conditions:
            prompts["list_anon"] = build_list_anon_prompt(tokenizer, query, tools)
        if "facet_anon" in conditions:
            prompts["facet_anon"] = build_facet_anon_prompt(tokenizer, query, tools, schema)
        if "noprompt" in conditions:
            prompts["noprompt"] = build_noprompt(tokenizer, query)
        if "nl_full_source" in conditions:
            prompts["nl_full_source"] = build_nl_full_source_prompt(
                tokenizer, query, source_tools_list
            )
        if "facet_xfer" in conditions:
            prompts["facet_xfer"] = build_facet_xfer_prompt(
                tokenizer, query, tools, schema, source_domain
            )

        for cond in conditions:
            ptxt = prompts[cond]
            tok_count = len(tokenizer(ptxt).input_ids)
            prompt_lengths[cond].append(tok_count)
            gen = generate_text(model, tokenizer, ptxt, args.max_new_tokens, args.device)
            # nl_full_source presents source's tool list; collect preds against
            # union of source+target so the model's emitted source-tool names
            # are visible in the metric (otherwise filter loses everything).
            if cond == "nl_full_source":
                cand = list(set(source_tools_list) | set(tools))
            else:
                cand = tools
            pred = extract_tool_names(gen, cand)
            m = compute_metrics(pred, gt_list)
            metrics[cond].append(m)
            generations[cond].append(gen[:512])

        if (i + 1) % 4 == 0 or i == len(items) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            f1_now = {c: float(np.mean([x["f1"] for x in metrics[c]])) for c in conditions}
            print(
                f"[{i+1}/{len(items)}] {elapsed:.1f}s rate={rate:.2f}/s  "
                + " ".join(f"{c[:8]}={f1_now[c]:.3f}" for c in conditions),
                flush=True,
            )

    # Aggregate
    summary = {}
    n_boot = max(0, int(args.bootstrap_n_boot))
    for cond in conditions:
        ms = metrics[cond]
        if not ms:
            continue
        f1_vals = [m.get("f1", 0.0) for m in ms]
        prec_vals = [m.get("precision", 0.0) for m in ms]
        rec_vals = [m.get("recall", 0.0) for m in ms]
        if n_boot > 0:
            f1_lo, f1_hi = bootstrap_ci(f1_vals, n_boot=n_boot)
            prec_lo, prec_hi = bootstrap_ci(prec_vals, n_boot=n_boot)
            rec_lo, rec_hi = bootstrap_ci(rec_vals, n_boot=n_boot)
        else:
            f1_lo = f1_hi = prec_lo = prec_hi = rec_lo = rec_hi = float("nan")
        summary[cond] = {
            "n": len(ms),
            "f1_mean": float(np.mean(f1_vals)),
            "f1_sd": float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0,
            "f1_ci95_lo": f1_lo,
            "f1_ci95_hi": f1_hi,
            "f_05_mean": float(np.mean([m.get("f_05", 0.0) for m in ms])),
            "eu_mean": float(np.mean([m.get("eu", 0.0) for m in ms])),
            "jaccard_mean": float(np.mean([m.get("jaccard", 0.0) for m in ms])),
            "exact_mean": float(np.mean([m.get("exact", 0.0) for m in ms])),
            "precision_mean": float(np.mean(prec_vals)),
            "precision_ci95_lo": prec_lo,
            "precision_ci95_hi": prec_hi,
            "recall_mean": float(np.mean(rec_vals)),
            "recall_ci95_lo": rec_lo,
            "recall_ci95_hi": rec_hi,
            "n_pred_mean": float(np.mean([len(m.get("pred", [])) for m in ms])),
            "prompt_len_mean": float(np.mean(prompt_lengths[cond])),
            "prompt_len_min": int(np.min(prompt_lengths[cond])),
            "prompt_len_max": int(np.max(prompt_lengths[cond])),
        }

    out = {
        "model": args.model,
        "task": args.task,
        "schema_path": args.schema,
        "source_schema_path": args.source_schema,
        "source_domain": source_domain,
        "source_tools": source_tools_list,
        "n_samples": len(items),
        "max_new_tokens": args.max_new_tokens,
        "conditions": conditions,
        "d0_query_mode": args.query_mode,
        "d0_gt_mode": args.gt_mode,
        "summary": summary,
        "details": metrics,
        "prompt_lengths": prompt_lengths,
        "generations_sample": {c: generations[c][:3] for c in conditions},
        "wall_seconds": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[done] saved -> {out_path}", flush=True)

    # Pretty-print summary table
    print()
    hdr = "  ".join(f"{c:>16}" for c in conditions)
    print(f"{'metric':<18} {hdr}")
    print("-" * (20 + 18 * len(conditions)))
    for k in [
        "n",
        "prompt_len_mean",
        "f1_mean",
        "f1_sd",
        "f_05_mean",
        "exact_mean",
        "precision_mean",
        "recall_mean",
    ]:
        if k == "n":
            row = "  ".join(f"{summary[c][k]:>16d}" for c in conditions)
        else:
            row = "  ".join(f"{summary[c][k]:>16.4f}" for c in conditions)
        print(f"{k:<18} {row}")
    # CI rows (mean [lo, hi])
    for k in ["f1", "precision", "recall"]:
        row = "  ".join(
            f"[{summary[c][f'{k}_ci95_lo']:.3f},{summary[c][f'{k}_ci95_hi']:.3f}]".rjust(16)
            for c in conditions
        )
        print(f"{k+'_ci95':<18} {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
