#!/usr/bin/env python3
"""
dpo_train.py — §3 Rung1.5 ②: manual DPO on readiness-gate ordering pairs (trl-free; project
policy avoids trl/seka_env conflicts). Policy = base + trainable LoRA initialised from the Rung1
SFT adapter; reference = base + the SAME SFT adapter, FROZEN. DPO loss over (prompt, chosen,
rejected) using assistant-only token log-probs:

    L = -log sigmoid( beta * [ (logp_pol(chosen) - logp_ref(chosen))
                              - (logp_pol(rejected) - logp_ref(rejected)) ] )

Teaches the negative signal SFT cannot: disprefer premature-ACT (ready=false), over-refuse,
under-refuse. KL to the SFT policy is implicit in the DPO objective (ref term + beta).

ENV: seka_env (torch + peft). ~2x 7B in memory (policy+ref) -> fits one 48GB GPU.
RUN: python scripts/dpo_train.py --base Qwen/Qwen2.5-7B-Instruct \
        --sft-adapter <rung1_sft_dir> --pairs dpo_s1.jsonl --out-dir <dpo_out> --beta 0.1
"""
import argparse, json, copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--sft-adapter", required=True, help="Rung1 SFT LoRA dir (policy init + ref)")
    p.add_argument("--pairs", required=True, help="DPO pairs jsonl (build_dpo_pairs.py)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def seq_logp(model, tok, prompt, completion, device, max_len):
    """Sum log-prob of the `completion` (assistant turn) tokens given `prompt` (user turn)."""
    pre = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True)
    full = tok.apply_chat_template([{"role": "user", "content": prompt},
                                    {"role": "assistant", "content": completion}],
                                   tokenize=True, add_generation_prompt=False)
    comp_len = max(1, len(full) - len(pre))      # # of completion (assistant) tokens
    if len(full) > max_len:                      # keep the TAIL (completion intact), drop prompt head
        full = full[len(full) - max_len:]
    ids = torch.tensor([full], device=device)
    # memory: full-seq logits (seq x 152K vocab, x2 graphs) OOM at long seq -> run the
    # transformer body on the full sequence but apply lm_head ONLY to the completion span.
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = core.model(input_ids=ids).last_hidden_state          # [1, L, H] (LoRA active)
    span = hidden[:, -(comp_len + 1):-1, :]                       # states predicting completion tokens
    logits = core.lm_head(span)                                   # [1, comp_len, V] only
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    targets = ids[:, -comp_len:].unsqueeze(-1)
    tok_lp = logprobs.gather(-1, targets).squeeze(-1)[0]
    return tok_lp.sum()


def main():
    a = parse_args()
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(a.base)
    # single base + dual adapters: "default"=policy (trainable), "ref"=frozen copy.
    # (two full 7B models + both policy graphs OOM on 48GB at long seq)
    print("[load] base + dual adapters (policy trainable / ref frozen)", flush=True)
    pol = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype, device_map=a.device),
        a.sft_adapter, adapter_name="default", is_trainable=True)
    pol.load_adapter(a.sft_adapter, adapter_name="ref")
    pol.set_adapter("default")
    pol.config.use_cache = False
    # activation memory: 2x 7B + long-seq backward without checkpointing OOMs on 48GB
    pol.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pol.enable_input_require_grads()

    pairs = [json.loads(l) for l in open(a.pairs, encoding="utf-8") if l.strip()]
    if a.max_examples:
        pairs = pairs[:a.max_examples]
    opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], lr=a.lr)
    step = 0
    for ep in range(a.epochs):
        for i, ex in enumerate(pairs):
            pol.eval()  # ref pass deterministic (no dropout)
            with torch.no_grad():
                pol.set_adapter("ref")
                lp_ref_c = seq_logp(pol, tok, ex["prompt"], ex["chosen"], a.device, a.max_seq_len)
                lp_ref_r = seq_logp(pol, tok, ex["prompt"], ex["rejected"], a.device, a.max_seq_len)
            pol.set_adapter("default")
            pol.train()  # REQUIRED: transformers gradient checkpointing only fires when training=True
            lp_pol_c = seq_logp(pol, tok, ex["prompt"], ex["chosen"], a.device, a.max_seq_len)
            lp_pol_r = seq_logp(pol, tok, ex["prompt"], ex["rejected"], a.device, a.max_seq_len)
            margin = (lp_pol_c - lp_ref_c) - (lp_pol_r - lp_ref_r)
            loss = -torch.nn.functional.logsigmoid(a.beta * margin) / a.grad_accum
            loss.backward()
            if (i + 1) % a.grad_accum == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % a.log_every == 0:
                    acc = (margin > 0).float().item()
                    print(f"  ep{ep} step{step} loss={loss.item()*a.grad_accum:.4f} "
                          f"margin={margin.item():.3f} chosen>rej={acc:.0f}", flush=True)
    pol.save_pretrained(a.out_dir, selected_adapters=["default"]); tok.save_pretrained(a.out_dir)
    print(f"[done] DPO adapter -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
