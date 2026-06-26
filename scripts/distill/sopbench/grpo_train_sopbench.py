#!/usr/bin/env python3
"""
grpo_train_sopbench.py — §3 Rung2 ③: manual GRPO RFT on SOPBench (trl-free; project policy).
Policy = base + trainable LoRA initialised from the Rung1 SFT (or DPO) adapter; reference =
frozen SFT/DPO adapter. Reward = sopbench_reward (BOTH + dirgraph-progress − early-ACT penalty,
dual-axis). GRPO: G rollouts / prompt, advantage = (r − group_mean)/group_std, loss =
−adv · logp(planner_output | prompt) + beta · KL(policy‖ref) over the logged planner steps.

TWO-PHASE per iteration (re-serve between, like the eval orchestrators):
  (A) ROLLOUT  : serve current adapter (vLLM, temp>0); run_simulation G× per task with
                 SOPBENCH_RLLOG → planner (prompt,output) steps + the rule-evaluator verdict.
  (B) UPDATE   : load HF policy+ref; compute GRPO loss over the logged steps; step LoRA; save.

This file implements (B) fully (reward+advantage+GRPO LoRA update) and the rollout ASSEMBLY
from RLLOG+eval. The rollout-SERVING orchestration reuses the eval scripts' serve pattern and
needs one remote validation pass at Rung2 (task isolation: run_simulation per-task or group
RLLOG by `goal`). Entered only after Rung1 SFT (+ optional DPO) results are in.

RUN (update core, given a collected rollouts.jsonl):
  python scripts/grpo_train_sopbench.py update --base Qwen/Qwen2.5-7B-Instruct \
     --init-adapter <rung1_sft> --rollouts rollouts.jsonl --out-dir <grpo_out> --beta 0.04
rollouts.jsonl row = {"reward": float, "group_id": str, "steps":[{"prompt":..,"output":..}]}
(reward precomputed via sopbench_reward; group_id = task/goal id for GRPO normalization)
"""
import argparse, json
import torch
from collections import defaultdict


def seq_logp(model, tok, prompt, completion, device, max_len=2048):
    pre = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True)
    full = tok.apply_chat_template([{"role": "user", "content": prompt},
                                    {"role": "assistant", "content": completion}],
                                   tokenize=True, add_generation_prompt=False)
    comp_len = max(1, len(full) - len(pre))
    if len(full) > max_len:
        full = full[len(full) - max_len:]
    ids = torch.tensor([full], device=device)
    out = model(ids).logits[:, :-1, :]
    lp = torch.log_softmax(out.float(), dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
    return lp[-comp_len:].sum()


def grpo_update(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.base)
    pol = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map=args.device),
        args.init_adapter, is_trainable=True)
    ref = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map=args.device),
        args.init_adapter)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    pol.config.use_cache = False

    rows = [json.loads(l) for l in open(args.rollouts, encoding="utf-8") if l.strip()]
    # GRPO advantage per group (precomputed reward; normalize within group_id)
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("group_id", "g")].append(r)
    import statistics
    for g, rs in groups.items():
        rw = [x["reward"] for x in rs]
        m = statistics.mean(rw); sd = statistics.pstdev(rw) or 1.0
        for x, r in zip(rs, rw):
            x["adv"] = (r - m) / sd

    opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], lr=args.lr)
    step = 0
    for ep in range(args.epochs):
        for i, row in enumerate(rows):
            adv = row.get("adv", 0.0)
            if abs(adv) < 1e-6 or not row.get("steps"):
                continue
            loss = 0.0
            for s in row["steps"]:
                lp_pol = seq_logp(pol, tok, s["prompt"], s["output"], args.device, args.max_seq_len)
                with torch.no_grad():
                    lp_ref = seq_logp(ref, tok, s["prompt"], s["output"], args.device, args.max_seq_len)
                # policy-gradient (−adv·logp) + KL-to-ref regularizer
                loss = loss - adv * lp_pol + args.beta * (lp_pol - lp_ref)
            loss = loss / max(1, len(row["steps"])) / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % args.log_every == 0:
                    print(f"  ep{ep} step{step} loss={loss.item()*args.grad_accum:.4f}", flush=True)
    pol.save_pretrained(args.out_dir); tok.save_pretrained(args.out_dir)
    print(f"[done] GRPO adapter -> {args.out_dir}", flush=True)


def assemble(args):
    """Build rollouts.jsonl from RLLOG files + eval output. Each rollout = one run_simulation
    sample (temp>0); group_id = goal+task index. Reward via sopbench_reward.
    NOTE: requires per-task isolation in the RLLOG (group steps by `goal`); validate on remote."""
    import glob, sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sopbench_reward import reward
    data_tasks = {}
    raw = json.load(open(args.tasks))           # {goal: [task,...]} (SOPBench <domain>_tasks.json)
    for goal, lst in raw.items():
        for t in lst:
            data_tasks.setdefault(goal, t if isinstance(t, dict) else {})
    # eval records (one per task) — used for the verdict (dirgraph/action_called/success)
    ev_by_goal = defaultdict(list)
    for f in glob.glob(args.eval_glob):
        for rec in json.load(open(f)):
            ev = rec.get("evaluations", [{}])
            ev_by_goal[rec["task"].get("user_goal")].append((rec["task"], ev[0] if ev else {}, rec))
    n = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for rl in glob.glob(args.rllog_glob):
            steps = [json.loads(l) for l in open(rl, encoding="utf-8") if l.strip()]
            if not steps:
                continue
            goal = steps[0].get("goal")
            recs = ev_by_goal.get(goal) or []
            if not recs:
                continue
            task, ev, rec = recs[0]                # NOTE: maps to first task of goal — refine per-task on remote
            tool_seq = [m.get("tool_name") for it in rec.get("interactions", [])
                        if isinstance(it, dict) for m in it.get("interaction", [])
                        if isinstance(m, dict) and m.get("tool_name")]
            r = reward(ev, tool_seq, task)
            w.write(json.dumps({"reward": r["reward"], "group_id": goal,
                                "steps": [{"prompt": s["prompt"], "output": s["output"]} for s in steps]}) + "\n")
            n += 1
    print(f"[assemble] {n} rollouts -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update")
    u.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    u.add_argument("--init-adapter", required=True)
    u.add_argument("--rollouts", required=True)
    u.add_argument("--out-dir", required=True)
    u.add_argument("--device", default="cuda:0")
    u.add_argument("--beta", type=float, default=0.04)
    u.add_argument("--lr", type=float, default=1e-6)
    u.add_argument("--epochs", type=int, default=1)
    u.add_argument("--grad-accum", type=int, default=8)
    u.add_argument("--max-seq-len", type=int, default=2048)
    u.add_argument("--log-every", type=int, default=10)
    a = sub.add_parser("assemble")
    a.add_argument("--rllog-glob", required=True)
    a.add_argument("--eval-glob", required=True)
    a.add_argument("--tasks", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "update":
        grpo_update(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
