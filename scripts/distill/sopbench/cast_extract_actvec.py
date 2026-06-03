"""cast_extract_actvec.py — CAST probe step 1: extract the ACT-vs-REFUSE behavior vector
from the TRAINED adapter, controlling for prompt content (same terminal-state prompt,
contrast the ACT vs STOP decision completion). mean(ACT)-mean(STOP) per layer.

Rationale (EXPERIMENT_DESIGN Rung3 재검토 2026-06-03): permitted over-refusal has a PRIOR-BIAS
component (training permitted true:false = 1:1.9). A steering vector toward the ACT/permitted=true
direction, applied gated, may counter that bias. This vector is the behavior vector; we apply it
with the existing _steering_vllm_server_gated.py and sweep alpha on bank eval.
NOTE (honest): this fixes BIAS, not the grounding/serial-compute parts of the collapse.

Output format matches _steering_vllm_server_gated.py loader:
  { "metadata": {...}, "vectors": { "actvec": { <layer_idx:int>: Tensor[d_model] } } }

RUN (remote, seka_env, from PI repo or anywhere with PYTHONPATH ok):
  PYTHONPATH=<CLONE> python scripts/distill/sopbench/cast_extract_actvec.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter <RUNS>/qwen7b_tbox_alias_s3_scratch_lodo_bank \
    --train-jsonl /home/woori/scratch/sft_alias_run/lodo_train_alias_s3_scratch.jsonl \
    --out /home/woori/scratch/sft_alias_run/cast_actvec_alias_s3.pt \
    --device cuda:0 --layers 10-23 --max-pairs 300
"""
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_layers(spec, n):
    spec = spec.strip()
    if spec == "all":
        return list(range(n + 1))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", default="10-23")
    ap.add_argument("--max-pairs", type=int, default=300)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"[cast] loading base {args.base_model} + adapter {args.adapter}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map=args.device,
        attn_implementation="eager")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    layer_idxs = parse_layers(args.layers, n_layers)
    print(f"[cast] n_layers={n_layers} d_model={d_model} capture_layers={layer_idxs}")

    # collect terminal-state prompts (assistant target starts with 'ready=true')
    prompts = []
    for line in open(args.train_jsonl, encoding="utf-8"):
        m = json.loads(line)["messages"]
        if m[1]["content"].strip().startswith("ready=true"):
            prompts.append(m[0]["content"])
        if len(prompts) >= args.max_pairs:
            break
    print(f"[cast] {len(prompts)} terminal-state prompts")

    ACT = "ready=true; preconds_verified=true; permitted=true; ACT"
    STOP = "ready=true; preconds_verified=true; permitted=false; STOP"

    sums = {l: {"pos": None, "neg": None} for l in layer_idxs}
    cnt = 0

    @torch.no_grad()
    def last_hidden(user_prompt, decision):
        msgs = [{"role": "user", "content": user_prompt},
                {"role": "assistant", "content": decision}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(args.device)
        out = model(**enc, output_hidden_states=True)
        res = {}
        for l in layer_idxs:
            res[l] = out.hidden_states[l][0, -1, :].detach().to(torch.float32).cpu()
        return res

    t0 = time.time()
    for i, p in enumerate(prompts):
        hp = last_hidden(p, ACT)
        hn = last_hidden(p, STOP)
        for l in layer_idxs:
            sums[l]["pos"] = hp[l] if sums[l]["pos"] is None else sums[l]["pos"] + hp[l]
            sums[l]["neg"] = hn[l] if sums[l]["neg"] is None else sums[l]["neg"] + hn[l]
        cnt += 1
        if (i + 1) % 50 == 0:
            print(f"[cast] {i+1}/{len(prompts)} elapsed={time.time()-t0:.1f}s")

    vectors = {"actvec": {}}
    for l in layer_idxs:
        v = (sums[l]["pos"] - sums[l]["neg"]) / cnt
        vectors["actvec"][l] = v
        print(f"[cast] layer {l}: ||mean-diff||={v.norm():.4f}")

    torch.save({"metadata": {"base": args.base_model, "adapter": args.adapter,
                             "n_layers": n_layers, "d_model": d_model,
                             "layers": layer_idxs, "n_pairs": cnt,
                             "contrast": "ACT vs STOP (same terminal prompt)"},
                "vectors": vectors}, args.out)
    print(f"[cast] saved -> {args.out}  ({cnt} pairs, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
