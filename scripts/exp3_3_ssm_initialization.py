"""
Experiment 3-3: Learned ω SSM Initialization
=============================================
Compare 5 SSM initialization strategies:
1. HiPPO (standard baseline)
2. Random
3. Fixed RoPE ω
4. YaRN-scaled ω
5. Spectral-weighted ω (from attention energy spectrum)

Train small Mamba-like SSM on language modeling, compare convergence.
Using GPT-2 Medium's attention spectrum to derive ω values.
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
BATCH_SIZE = 4
STATE_DIM = 64
TRAIN_STEPS = 300
LR = 1e-3
OUTPUT_DIR = Path("/mnt/input/fokvq/results/exp3_3")
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


def hippo_initialization(N):
    """HiPPO-LegS initialization for SSM A matrix."""
    A = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if i > j:
                A[i, j] = -(2*i + 1)**0.5 * (2*j + 1)**0.5
            elif i == j:
                A[i, j] = -(i + 1)
    return A


def rope_frequencies(N, base=10000):
    """RoPE-style frequency initialization."""
    freqs = 1.0 / (base ** (torch.arange(0, N, 2).float() / N))
    return freqs


def yarn_frequencies(N, base=10000, scale=4.0):
    """YaRN-scaled frequency initialization."""
    freqs = rope_frequencies(N, base)
    return freqs / scale


def extract_spectral_weights(model, tokenizer, texts, device):
    """Extract attention energy spectrum from GPT-2."""
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads

    spectral_energy = torch.zeros(head_dim // 2 + 1, device=device)
    count = 0

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_SEQ_LEN).to(device)

        qk_store = {}
        def make_hook(idx):
            def hook_fn(module, input, output):
                hidden = input[0]
                qkv = module.c_attn(hidden)
                q, k, _ = qkv.split(model.config.n_embd, dim=-1)
                bsz, slen, _ = q.shape
                q = q.view(bsz, slen, num_heads, head_dim).transpose(1, 2)
                k = k.view(bsz, slen, num_heads, head_dim).transpose(1, 2)
                qk_store[idx] = (q.detach(), k.detach())
            return hook_fn

        hooks = []
        for idx, block in enumerate(model.transformer.h):
            hooks.append(block.attn.register_forward_hook(make_hook(idx)))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        # Average spectral energy across all layers
        for idx in qk_store:
            q, k = qk_store[idx]
            q_fft = torch.fft.rfft(q.float(), dim=-1)
            k_fft = torch.fft.rfft(k.float(), dim=-1)
            energy = (q_fft.abs() * k_fft.abs()).mean(dim=(0, 1, 2))
            spectral_energy += energy
            count += 1

        del inputs, qk_store
        torch.cuda.empty_cache()

    spectral_energy /= count
    return spectral_energy.cpu()


class SimpleSSM(nn.Module):
    """SSM with configurable A matrix initialization."""
    def __init__(self, input_dim, state_dim, output_dim, A_init):
        super().__init__()
        self.A = nn.Parameter(A_init.clone())
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.01)
        self.state_dim = state_dim

    def forward(self, u):
        batch, seq_len, _ = u.shape
        x = torch.zeros(batch, self.state_dim, device=u.device)
        outputs = []
        for t in range(seq_len):
            x = torch.tanh(F.linear(x, self.A) + F.linear(u[:, t], self.B))
            y = F.linear(x, self.C)
            outputs.append(y)
        return torch.stack(outputs, dim=1)


def create_A_init(method, N, spectral_weights=None):
    """Create A matrix initialization for given method."""
    if method == "hippo":
        return hippo_initialization(N)
    elif method == "random":
        return torch.randn(N, N) * 0.01
    elif method == "rope":
        freqs = rope_frequencies(N)
        A = torch.zeros(N, N)
        for i in range(min(len(freqs), N//2)):
            A[2*i, 2*i] = -0.5
            A[2*i, 2*i+1] = freqs[i]
            A[2*i+1, 2*i] = -freqs[i]
            A[2*i+1, 2*i+1] = -0.5
        return A
    elif method == "yarn":
        freqs = yarn_frequencies(N)
        A = torch.zeros(N, N)
        for i in range(min(len(freqs), N//2)):
            A[2*i, 2*i] = -0.5
            A[2*i, 2*i+1] = freqs[i]
            A[2*i+1, 2*i] = -freqs[i]
            A[2*i+1, 2*i+1] = -0.5
        return A
    elif method == "spectral":
        if spectral_weights is None:
            return torch.randn(N, N) * 0.01
        # Weight frequencies by attention energy spectrum
        n_freqs = min(len(spectral_weights), N//2)
        weights = spectral_weights[:n_freqs]
        weights = weights / weights.sum()  # normalize

        base_freqs = rope_frequencies(N)[:n_freqs]
        weighted_freqs = base_freqs * (1 + weights * 10)  # amplify by spectral weight

        A = torch.zeros(N, N)
        for i in range(n_freqs):
            A[2*i, 2*i] = -0.5
            A[2*i, 2*i+1] = weighted_freqs[i]
            A[2*i+1, 2*i] = -weighted_freqs[i]
            A[2*i+1, 2*i+1] = -0.5
        return A
    else:
        raise ValueError(f"Unknown method: {method}")


def run_experiment():
    print(f"{'='*60}")
    print(f"Exp 3-3: SSM Initialization Comparison")
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

    texts = (TEXTS * ((NUM_SAMPLES // len(TEXTS)) + 1))[:NUM_SAMPLES]

    # Extract spectral weights from GPT-2
    print("Extracting spectral weights...")
    spectral_weights = extract_spectral_weights(model, tokenizer, texts, DEVICE)

    # Collect attention I/O for distillation target (layer 12 = middle)
    target_layer = 12
    print(f"Collecting attention I/O (layer {target_layer})...")
    all_inputs, all_outputs = [], []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=MAX_SEQ_LEN).to(DEVICE)

        attn_in = None; attn_out = None
        def hook_pre(module, args):
            nonlocal attn_in; attn_in = args[0].detach()
        def hook_post(module, args, output):
            nonlocal attn_out
            attn_out = output[0].detach() if isinstance(output, tuple) else output.detach()

        block = model.transformer.h[target_layer].attn
        h1 = block.register_forward_pre_hook(hook_pre)
        h2 = block.register_forward_hook(hook_post)
        with torch.no_grad():
            model(**inputs)
        h1.remove(); h2.remove()
        all_inputs.append(attn_in.cpu())
        all_outputs.append(attn_out.cpu())
        del inputs

    X = torch.cat(all_inputs, dim=0)
    Y = torch.cat(all_outputs, dim=0)
    x_std = X.std() + 1e-8; y_std = Y.std() + 1e-8
    X_norm = X / x_std; Y_norm = Y / y_std

    del model
    torch.cuda.empty_cache()

    # Train SSM with each initialization
    methods = ["hippo", "random", "rope", "yarn", "spectral"]
    input_dim = X.shape[-1]
    output_dim = Y.shape[-1]

    results = {
        "experiment": "3-3_ssm_initialization",
        "model": MODEL_NAME,
        "target_layer": target_layer,
        "state_dim": STATE_DIM,
        "train_steps": TRAIN_STEPS,
        "methods": {},
    }

    for method in methods:
        print(f"\n  {method}...")
        A_init = create_A_init(method, STATE_DIM, spectral_weights)
        ssm = SimpleSSM(input_dim, STATE_DIM, output_dim, A_init).to(DEVICE)
        optimizer = torch.optim.Adam(ssm.parameters(), lr=LR)

        losses = []
        X_dev = X_norm.to(DEVICE)
        Y_dev = Y_norm.to(DEVICE)

        for step in range(TRAIN_STEPS):
            idx = torch.randint(0, X_dev.shape[0], (min(8, X_dev.shape[0]),))
            pred = ssm(X_dev[idx])
            loss = F.mse_loss(pred, Y_dev[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ssm.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if step % 100 == 0:
                print(f"    step {step}: loss={loss.item():.4f}")

        # Final NMSE
        with torch.no_grad():
            pred_all = ssm(X_dev)
            nmse = F.mse_loss(pred_all, Y_dev).item() / (Y_dev.var().item() + 1e-8)

        results["methods"][method] = {
            "final_nmse": float(nmse),
            "final_loss": float(losses[-1]),
            "best_loss": float(min(losses)),
            "loss_curve": [float(l) for l in losses[::10]],  # every 10th
            "converged": losses[-1] < losses[0] * 0.5,
        }
        print(f"    NMSE={nmse:.4f}, best_loss={min(losses):.4f}")

        del ssm, optimizer
        torch.cuda.empty_cache()

    # Ranking
    ranked = sorted(results["methods"].items(), key=lambda x: x[1]["final_nmse"])
    results["ranking"] = [{"rank": i+1, "method": m, "nmse": d["final_nmse"]} for i, (m, d) in enumerate(ranked)]
    results["best_method"] = ranked[0][0]
    results["spectral_is_best"] = ranked[0][0] == "spectral"
    results["runtime_seconds"] = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Ranking:")
    for r in results["ranking"]:
        print(f"  {r['rank']}. {r['method']}: NMSE={r['nmse']:.4f}")
    print(f"Best: {results['best_method']}")
    print(f"Spectral is best: {results['spectral_is_best']}")
    print(f"{'='*60}")

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    r = run_experiment()
    sys.exit(0)
