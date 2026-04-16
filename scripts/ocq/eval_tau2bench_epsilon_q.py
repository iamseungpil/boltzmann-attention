#!/usr/bin/env python3
"""eval_tau2bench_epsilon_q.py — τ²-bench retail per-step tool prediction + ε_q plan predictor.

Per-step proxy: for each task, construct single-turn FC prompt at each step
(oracle GT for prior steps), generate tool_call, measure ε_q and accuracy.
Then compute AUROC(min_eps_q → conversation success).
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
from contextlib import nullcontext

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from eval_metatool_subtask4 import extract_tool_names  # noqa


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--b-ont", required=True)
    p.add_argument("--tasks-json", default=str(REPO / "external/tau2-bench/data/tau2/domains/retail/tasks.json"))
    p.add_argument("--max-tasks", type=int, default=50)
    p.add_argument("--beta", type=float, default=-0.1)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--hook-layer", type=int, default=13)
    p.add_argument("--out", required=True)
    return p.parse_args()


TOOLS_DESC = {
    "find_user_id_by_name_zip": "Find user ID by first name, last name, and zip code.",
    "find_user_id_by_email": "Find user ID by email address.",
    "get_user_details": "Get user details by user ID.",
    "get_order_details": "Get order details by order ID.",
    "get_product_details": "Get product details by product ID.",
    "get_item_details": "Get item details by item ID.",
    "calculate": "Calculate a mathematical expression.",
    "cancel_pending_order": "Cancel a pending order.",
    "exchange_delivered_order_items": "Exchange items in a delivered order.",
    "return_delivered_order_items": "Return items from a delivered order.",
    "modify_pending_order_items": "Modify items in a pending order.",
    "modify_pending_order_address": "Modify the shipping address of a pending order.",
    "modify_pending_order_payment": "Modify the payment method of a pending order.",
    "modify_user_address": "Modify a user's address.",
    "transfer_to_human_agents": "Transfer the conversation to human agents.",
}


def build_fc_prompt(tokenizer, task, step_idx, gt_actions):
    tools_list = "\n".join(f"- {name}: {desc}" for name, desc in TOOLS_DESC.items())
    reason = task["user_scenario"]["instructions"]["reason_for_call"]
    known = task["user_scenario"]["instructions"].get("known_info", "")

    prior_steps = ""
    for i in range(step_idx):
        a = gt_actions[i]
        prior_steps += f"\nStep {i+1}: Called {a['name']}({json.dumps(a.get('arguments',{}))})"

    user_msg = f"""Customer request: {reason}
Known information: {known}
{f'Prior actions taken:{prior_steps}' if prior_steps else 'This is the first action.'}

Select the next tool to call from the available tools below.
Available tools:
{tools_list}"""

    messages = [
        {"role": "system", "content": "You are a retail customer service agent. Select the appropriate tool for the next step. Respond with a single <tool_call> block."},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    B_ont = B_ont.to(args.device, dtype=torch.float32)
    L_total, H_kv, d, R = B_ont.shape
    print(f"[bont] shape={B_ont.shape}", flush=True)

    tasks = json.load(open(args.tasks_json))
    tasks = [t for t in tasks if len(t["evaluation_criteria"]["actions"]) > 0]
    tasks = tasks[:args.max_tasks]
    print(f"[data] {len(tasks)} tasks", flush=True)

    # ε_q hook
    eps_q_vals = []
    L_hook = args.hook_layer
    B_hook = B_ont[L_hook]  # (H_kv, d, R)

    def eps_q_hook(mod, inp, out):
        k = out
        if k.dim() == 3:
            B, T, D_all = k.shape
            k = k.view(B, T, H_kv, d)
        elif k.dim() == 4:
            pass
        else:
            return
        k_f = k.float()
        proj = torch.einsum("bthd,hdr->bthr", k_f, B_hook)
        energy = (proj ** 2).sum(-1)
        k_norm_sq = (k_f ** 2).sum(-1) + 1e-8
        eps = (energy / k_norm_sq).mean(dim=-1)  # (B, T) avg over heads
        eps_q_vals.append(eps[:, -1].item() if eps.dim() == 2 else eps.item())

    # Q-coverage hook (simple stationary β)
    beta = args.beta
    def install_qbias(model, B_ont_tensor, beta_val):
        hooks = []
        for l_idx in range(L_total):
            attn = model.model.layers[l_idx].self_attn
            B_l = B_ont_tensor[l_idx]  # (H_kv, d, R)
            P = torch.einsum("hdr,her->hde", B_l, B_l)  # (H_kv, d, d)

            def make_hook(P_proj):
                def hook_fn(mod, inp, out):
                    q = out
                    orig_shape = q.shape
                    if q.dim() == 3:
                        B, T, D_all = q.shape
                        q = q.view(B, T, H_kv, d)
                    q_biased = q.float() + beta_val * torch.einsum("bthd,hde->bthe", q.float(), P_proj)
                    return q_biased.to(out.dtype).view(orig_shape)
                return hook_fn

            hooks.append(attn.q_proj.register_forward_hook(make_hook(P)))
        return hooks

    results = []
    t0 = time.time()

    for ti, task in enumerate(tasks):
        gt_actions = task["evaluation_criteria"]["actions"]
        n_steps = len(gt_actions)
        step_results = []
        task_eps_q_min = 1.0

        for s_idx in range(n_steps):
            gt_tool = gt_actions[s_idx]["name"]
            prompt = build_fc_prompt(tok, task, s_idx, gt_actions)

            ids = tok(prompt, return_tensors="pt")["input_ids"].to(args.device)

            # Hook ε_q on k_proj
            eps_q_vals.clear()
            h_eps = model.model.layers[L_hook].self_attn.k_proj.register_forward_hook(eps_q_hook)

            # Q-coverage hooks
            q_hooks = install_qbias(model, B_ont, beta) if beta != 0 else []

            with torch.no_grad():
                out_ids = model.generate(
                    ids, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tok.eos_token_id,
                )

            h_eps.remove()
            for h in q_hooks:
                h.remove()

            gen_text = tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)
            pred_tools = extract_tool_names(gen_text, list(TOOLS_DESC.keys()))
            pred_tool = pred_tools[0] if pred_tools else ""
            correct = (pred_tool == gt_tool)

            step_eps = eps_q_vals[-1] if eps_q_vals else 0.0
            task_eps_q_min = min(task_eps_q_min, step_eps)

            step_results.append({
                "step": s_idx, "gt": gt_tool, "pred": pred_tool,
                "correct": correct, "eps_q": step_eps,
            })

        n_correct = sum(1 for s in step_results if s["correct"])
        success = n_correct / max(n_steps, 1) >= 0.8

        results.append({
            "task_id": task["id"], "n_steps": n_steps,
            "n_correct": n_correct, "success": success,
            "min_eps_q": task_eps_q_min, "steps": step_results,
        })

        if ti % 5 == 0:
            print(f"  [{ti}/{len(tasks)}] steps={n_steps} correct={n_correct}/{n_steps} "
                  f"success={success} min_eps_q={task_eps_q_min:.4f}", flush=True)

    runtime = time.time() - t0
    n_tasks = len(results)
    n_success = sum(1 for r in results if r["success"])

    # AUROC
    from sklearn.metrics import roc_auc_score
    labels = [int(r["success"]) for r in results]
    scores = [r["min_eps_q"] for r in results]
    if len(set(labels)) > 1:
        auroc = roc_auc_score(labels, scores)
    else:
        auroc = float("nan")

    print(f"\n{'='*50}", flush=True)
    print(f"Tasks: {n_tasks}, Success: {n_success}/{n_tasks} ({100*n_success/max(n_tasks,1):.1f}%)", flush=True)
    print(f"AUROC(min_eps_q → success): {auroc:.4f}", flush=True)
    print(f"Beta: {beta}, Runtime: {runtime:.1f}s", flush=True)
    print(f"{'='*50}", flush=True)

    blob = {
        "model": args.model, "beta": beta, "hook_layer": L_hook,
        "n_tasks": n_tasks, "n_success": n_success,
        "success_rate": n_success / max(n_tasks, 1),
        "auroc_min_eps_q": auroc, "runtime_s": runtime,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
