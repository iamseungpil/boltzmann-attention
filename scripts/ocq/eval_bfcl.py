#!/usr/bin/env python3
"""eval_bfcl.py — BFCL v3 proxy evaluation with Q-coverage and K-bias hooks.

This file is a screening harness, not an official BFCL evaluator.
It uses function-name set overlap rather than BFCL's AST-based scoring, so
its output must be treated as proxy evidence only.

Its practical role is to compare simple (1-function) versus parallel or
multiple (multi-function) behavior under the same steering hooks.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocq"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--b-ont", required=True)
    p.add_argument("--subsets", nargs="+", default=["simple", "parallel", "multiple"])
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--beta", type=float, default=-0.1)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    return p.parse_args()


def download_bfcl(subset_name):
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                        f"BFCL_v3_{subset_name}.json", repo_type="dataset")
    ans_f = None
    try:
        ans_f = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                                f"possible_answer/BFCL_v3_{subset_name}.json", repo_type="dataset")
    except:
        pass
    data = [json.loads(l) for l in open(f)]
    answers = [json.loads(l) for l in open(ans_f)] if ans_f else [None] * len(data)
    return data, answers


def build_bfcl_prompt(tokenizer, entry):
    question = entry["question"]
    functions = entry["function"]
    if isinstance(question, list) and isinstance(question[0], list):
        messages = question[0]
    elif isinstance(question, list):
        messages = question
    else:
        messages = [{"role": "user", "content": str(question)}]

    tools_desc = []
    for fn in functions:
        tools_desc.append(f"- {fn['name']}: {fn.get('description','')}")
    tools_str = "\n".join(tools_desc)

    system_msg = f"You are a helpful assistant with access to the following tools:\n{tools_str}\nRespond with <tool_call> blocks."
    full_messages = [{"role": "system", "content": system_msg}] + messages
    return tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)


def extract_function_names(text):
    names = []
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        names.append(m.group(1))
    if not names:
        for m in re.finditer(r'<tool_call>\s*\{[^}]*"name"\s*:\s*"([^"]+)"', text):
            names.append(m.group(1))
    return names


def get_gt_function_names(answer_entry):
    """Extract GT function names from BFCL answer. Handles multiple schemas:
      - {'result': [{'name': ..., ...}, ...]}
      - {'ground_truth': [{'fn_name_1': {...args...}}, {'fn_name_2': {...args...}}]}  (BFCL v3 parallel_multiple)
      - [{'name': ...}, ...]
    """
    if answer_entry is None:
        return []
    # BFCL v3 parallel/parallel_multiple/multiple schema
    if isinstance(answer_entry, dict) and "ground_truth" in answer_entry:
        ans = answer_entry["ground_truth"]
        names = []
        if isinstance(ans, list):
            for item in ans:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if k in ("name", "id"):
                            continue
                        # In BFCL v3, key IS the function name, value is args dict
                        names.append(k)
        return names
    if isinstance(answer_entry, dict) and "result" in answer_entry:
        ans = answer_entry["result"]
    elif isinstance(answer_entry, list):
        ans = answer_entry
    elif isinstance(answer_entry, str):
        try:
            ans = json.loads(answer_entry)
        except:
            return []
    else:
        ans = answer_entry

    names = []
    if isinstance(ans, list):
        for item in ans:
            if isinstance(item, dict) and "name" in item:
                names.append(item["name"])
            elif isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict) and "name" in parsed:
                        names.append(parsed["name"])
                except:
                    pass
    elif isinstance(ans, dict) and "name" in ans:
        names.append(ans["name"])
    return names


def compute_f1(pred, gt):
    if not gt:
        return {"precision": 0, "recall": 0, "f1": 0, "exact": 0}
    pred_set = set(pred)
    gt_set = set(gt)
    tp = len(pred_set & gt_set)
    precision = tp / max(len(pred_set), 1)
    recall = tp / max(len(gt_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    exact = int(pred_set == gt_set)
    return {"precision": precision, "recall": recall, "f1": f1, "exact": exact}


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eval_metatool_subtask4 import install_q_bias_hooks, install_kbias_hooks

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).eval()

    bdict = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    B_ont = bdict["B_ont"] if isinstance(bdict, dict) else bdict
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // model.config.num_attention_heads)

    all_results = {
        "benchmark": "BFCL_v3_proxy",
        "evaluation_status": "proxy_only",
        "scorer": "function_name_set_overlap_v1",
        "official_bfcl_ast_scorer": False,
        "prompt_template_id": "bfcl_chat_template_v1",
        "decode_policy": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tok.eos_token_id,
        },
        "runtime_config": {
            "device": args.device,
            "torch_dtype": "torch.bfloat16",
            "attn_implementation": "eager",
        },
        "results": {},
    }

    for subset_name in args.subsets:
        print(f"\n{'='*50}\n[bfcl] Subset: {subset_name}", flush=True)
        data, answers = download_bfcl(subset_name)
        data = data[:args.max_samples]
        answers = answers[:args.max_samples]
        print(f"[bfcl] N={len(data)}", flush=True)

        for method in ["no_steer", f"qbias_b{args.beta}", f"kbias_a{args.alpha}"]:
            print(f"\n[eval] {subset_name} / {method}", flush=True)
            per_sample = []
            t0 = time.time()

            for i, (entry, ans_entry) in enumerate(zip(data, answers)):
                prompt = build_bfcl_prompt(tok, entry)
                ids = tok(prompt, return_tensors="pt")["input_ids"].to(args.device)

                if method == "no_steer":
                    ctx = torch.no_grad()
                elif method.startswith("qbias"):
                    ctx = install_q_bias_hooks(model, B_ont, beta=args.beta,
                                               n_kv=n_kv, head_dim=head_dim)
                elif method.startswith("kbias"):
                    ctx = install_kbias_hooks(model, B_ont, alpha=args.alpha,
                                              n_kv=n_kv, head_dim=head_dim)

                with torch.no_grad(), ctx:
                    out_ids = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                             do_sample=False, pad_token_id=tok.eos_token_id)

                gen_text = tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)
                pred_names = extract_function_names(gen_text)
                gt_names = get_gt_function_names(ans_entry)
                metrics = compute_f1(pred_names, gt_names)

                per_sample.append({
                    "id": entry.get("id", i),
                    "pred": pred_names, "gt": gt_names,
                    "metrics": metrics,
                    "gen_head": gen_text[:200],
                })

                if i % 20 == 0:
                    print(f"  [{i}/{len(data)}] pred={pred_names[:2]} gt={gt_names[:2]} f1={metrics['f1']:.3f}", flush=True)

            runtime = time.time() - t0
            macro = {k: sum(s["metrics"][k] for s in per_sample) / max(len(per_sample), 1)
                     for k in ["precision", "recall", "f1", "exact"]}

            print(f"  {method}: F1={macro['f1']:.4f} Exact={macro['exact']:.4f} ({runtime:.1f}s)", flush=True)

            key = f"{subset_name}_{method}"
            all_results["results"][key] = {
                "subset": subset_name, "method": method,
                "n": len(per_sample), "macro": macro,
                "runtime_s": runtime, "per_sample": per_sample,
            }

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
