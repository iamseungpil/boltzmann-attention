#!/usr/bin/env python3
"""Phase transition with v3 patcher (verified working)."""
import sys, json, torch, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from exp4_2_v3_full_quant_ppl import AttentionKQuantPatcher, find_attention_modules
from exp_phase_transition import make_niah_prompt

DTYPE = torch.bfloat16
MODEL = "mistralai/Mistral-7B-v0.3"


def run_niah_v3(model, tokenizer, quant_layers, bits=2,
                depths=[0.3, 0.5, 0.7], reps=3):
    if not quant_layers:
        scores = []
        for d in depths:
            for _ in range(reps):
                ids = make_niah_prompt(tokenizer, 4096, d).cuda()
                S = ids.shape[1]
                gen = model.generate(ids, max_new_tokens=20, do_sample=False)
                text = tokenizer.decode(gen[0, S:], skip_special_tokens=True)
                scores.append(1.0 if "7392" in text else 0.0)
                del gen; torch.cuda.empty_cache()
        return sum(scores) / len(scores)

    p = AttentionKQuantPatcher(model, "uniform", bits)
    p.patch()
    attn_mods = find_attention_modules(model)
    quant_set = set(quant_layers)
    for i, (name, mod) in enumerate(attn_mods):
        if i not in quant_set and name in p.original_forwards:
            mod.forward = p.original_forwards[name]
    p.active = True
    p.reset_stats()

    scores = []
    for d in depths:
        for _ in range(reps):
            ids = make_niah_prompt(tokenizer, 4096, d).cuda()
            S = ids.shape[1]
            gen = model.generate(ids, max_new_tokens=20, do_sample=False)
            text = tokenizer.decode(gen[0, S:], skip_special_tokens=True)
            scores.append(1.0 if "7392" in text else 0.0)
            del gen; torch.cuda.empty_cache()

    p.active = False
    p.unpatch()
    gc.collect()
    torch.cuda.empty_cache()
    return sum(scores) / len(scores)


def main():
    print(f"PHASE TRANSITION V3: {MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE, trust_remote_code=True,
        attn_implementation="eager"
    ).cuda().eval()

    results = {}

    # FP16
    print("FP16", flush=True)
    s = run_niah_v3(model, tokenizer, [])
    results["fp16"] = {"niah": round(s, 3), "n": 0}
    print(f"  NIAH={s:.3f}", flush=True)

    # First-N sweep
    for N in [4, 8, 12, 16, 20, 24, 28, 32]:
        tag = f"first_{N}"
        print(tag, flush=True)
        s = run_niah_v3(model, tokenizer, list(range(N)))
        results[tag] = {"niah": round(s, 3), "n": N, "pattern": "first"}
        print(f"  NIAH={s:.3f}", flush=True)

    # Last-N sweep
    for N in [4, 8, 12, 16, 20, 24, 28, 32]:
        tag = f"last_{N}"
        print(tag, flush=True)
        s = run_niah_v3(model, tokenizer, list(range(32 - N, 32)))
        results[tag] = {"niah": round(s, 3), "n": N, "pattern": "last"}
        print(f"  NIAH={s:.3f}", flush=True)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Config':<12s} {'N':>3s} {'NIAH':>6s}")
    for tag in ["fp16"] + [f"first_{N}" for N in [4,8,12,16,20,24,28,32]] + \
               [f"last_{N}" for N in [4,8,12,16,20,24,28,32]]:
        if tag in results:
            r = results[tag]
            print(f"{tag:<12s} {r['n']:>3d} {r['niah']:>6.3f}")

    out_dir = Path("/scratch/boltzmann/results/phase_transition")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v3_phase_transition_2bit.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
