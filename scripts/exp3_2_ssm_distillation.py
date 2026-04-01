"""
Experiment 3-2: Attention → Low-Dimensional SSM Distillation
=============================================================
Distill each Transformer attention block into a discrete SSM
with state dimension N in {16, 32, 64, 128, 256}.
Measure NMSE between original attention output and SSM output.

Hypothesis: N<=128 achieves NMSE<0.1 for majority of layers.
"""
import json, sys, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openai-community/gpt2-medium"
DEVICE = "cuda:0"
MAX_SEQ_LEN = 128
NUM_SAMPLES = 40
BATCH_SIZE = 2
STATE_DIMS = [16, 32, 64, 128, 256]
NUM_LAYERS = 24
TRAIN_STEPS = 500
LR = 1e-3
OUTPUT_DIR = Path("/mnt/input/fokvq/results/exp3_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXTS = [
    "The transformer architecture revolutionized natural language processing.",
    "In statistical mechanics the Boltzmann distribution describes state probabilities.",
    "Quantum computing leverages superposition and entanglement for faster computation.",
    "The human genome contains approximately three billion base pairs.",
    "Climate change models predict rising sea levels worldwide.",
    "Deep learning achieved remarkable success in vision and speech.",
    "The standard model describes electromagnetic weak and strong nuclear forces.",
    "Protein folding is a grand challenge in computational biology.",
    "Reinforcement learning aligns large language models with human preferences.",
    "Information theory quantifies limits of data compression and communication.",
]


class DiscreteSSM(nn.Module):
    """Simple discrete SSM: x[t] = A x[t-1] + B u[t], y[t] = C x[t]"""
    def __init__(self, input_dim, state_dim, output_dim):
        super().__init__()
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.01)
        self.state_dim = state_dim

    def forward(self, u):
        # u: (batch, seq, input_dim)
        batch, seq_len, _ = u.shape
        x = torch.zeros(batch, self.state_dim, device=u.device)
        outputs = []
        for t in range(seq_len):
            x = torch.tanh(F.linear(x, self.A) + F.linear(u[:, t], self.B))
            y = F.linear(x, self.C)
            outputs.append(y)
        return torch.stack(outputs, dim=1)  # (batch, seq, output_dim)


def extract_attention_io(model, inputs, layer_idx):
    """Extract input and output of attention block at layer_idx."""
    attn_input = None
    attn_output = None

    def hook_pre(module, args):
        nonlocal attn_input
        attn_input = args[0].detach()

    def hook_post(module, args, output):
        nonlocal attn_output
        if isinstance(output, tuple):
            attn_output = output[0].detach()
        else:
            attn_output = output.detach()

    block = model.transformer.h[layer_idx].attn
    h1 = block.register_forward_pre_hook(hook_pre)
    h2 = block.register_forward_hook(hook_post)

    with torch.no_grad():
        model(**inputs)

    h1.remove()
    h2.remove()
    return attn_input, attn_output


def run_experiment():
    print(f"{'='*60}")
    print(f"Exp 3-2: SSM Distillation")
    print(f"Model: {MODEL_NAME}, States: {STATE_DIMS}")
    print(f"{'='*60}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded in {time.time()-t0:.1f}s")

    texts = (TEXTS * ((NUM_SAMPLES // len(TEXTS)) + 1))[:NUM_SAMPLES]

    # Collect attention I/O for all layers
    print("Collecting attention I/O...")
    layer_data = {l: {'inputs': [], 'outputs': []} for l in range(NUM_LAYERS)}

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=MAX_SEQ_LEN).to(DEVICE)
        for l in range(NUM_LAYERS):
            ai, ao = extract_attention_io(model, inputs, l)
            layer_data[l]['inputs'].append(ai.cpu())
            layer_data[l]['outputs'].append(ao.cpu())
        del inputs
        torch.cuda.empty_cache()
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  Batch {i//BATCH_SIZE+1}/{len(texts)//BATCH_SIZE}")

    # Concatenate
    for l in range(NUM_LAYERS):
        layer_data[l]['inputs'] = torch.cat(layer_data[l]['inputs'], dim=0)
        layer_data[l]['outputs'] = torch.cat(layer_data[l]['outputs'], dim=0)

    del model
    torch.cuda.empty_cache()

    # Distill each layer with each state dimension
    print("\nDistilling...")
    results = {
        "experiment": "3-2_ssm_distillation",
        "model": MODEL_NAME,
        "state_dims": STATE_DIMS,
        "num_layers": NUM_LAYERS,
        "train_steps": TRAIN_STEPS,
        "num_samples": NUM_SAMPLES,
        "per_layer": {},
    }

    input_dim = layer_data[0]['inputs'].shape[-1]
    output_dim = layer_data[0]['outputs'].shape[-1]

    # Test subset of layers for speed (every 4th)
    test_layers = list(range(0, NUM_LAYERS, 4))  # [0, 4, 8, 12, 16, 20]

    for N in STATE_DIMS:
        print(f"\n  State dim N={N}")
        for l in test_layers:
            X = layer_data[l]['inputs'].to(DEVICE)
            Y = layer_data[l]['outputs'].to(DEVICE)

            # Normalize
            x_std = X.std() + 1e-8
            y_std = Y.std() + 1e-8
            X_norm = X / x_std
            Y_norm = Y / y_std

            ssm = DiscreteSSM(input_dim, N, output_dim).to(DEVICE)
            optimizer = torch.optim.Adam(ssm.parameters(), lr=LR)

            best_loss = float('inf')
            for step in range(TRAIN_STEPS):
                # Mini-batch
                idx = torch.randint(0, X_norm.shape[0], (min(8, X_norm.shape[0]),))
                pred = ssm(X_norm[idx])
                loss = F.mse_loss(pred, Y_norm[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()

            # Evaluate NMSE on all data
            with torch.no_grad():
                pred_all = ssm(X_norm)
                nmse = F.mse_loss(pred_all, Y_norm).item() / (Y_norm.var().item() + 1e-8)

            key = f"N{N}_L{l}"
            results["per_layer"][key] = {
                "state_dim": N, "layer": l,
                "nmse": float(nmse), "best_train_loss": float(best_loss),
            }
            print(f"    L{l:2d}: NMSE={nmse:.4f}")

            del ssm, optimizer, X, Y, X_norm, Y_norm
            torch.cuda.empty_cache()

    # Summary
    for N in STATE_DIMS:
        nmses = [results["per_layer"][f"N{N}_L{l}"]["nmse"] for l in test_layers]
        mean_nmse = np.mean(nmses)
        below_01 = sum(1 for n in nmses if n < 0.1)
        results[f"N{N}_mean_nmse"] = float(mean_nmse)
        results[f"N{N}_layers_below_0.1"] = below_01
        results[f"N{N}_fraction_below_0.1"] = below_01 / len(test_layers)
        print(f"\n  N={N}: mean NMSE={mean_nmse:.4f}, {below_01}/{len(test_layers)} layers < 0.1")

    # Hypothesis test
    n128_frac = results.get("N128_fraction_below_0.1", 0)
    results["hypothesis_supported"] = n128_frac > 0.5
    results["runtime_seconds"] = time.time() - t0

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}. Runtime: {results['runtime_seconds']:.0f}s")
    return results


if __name__ == "__main__":
    r = run_experiment()
    sys.exit(0)
