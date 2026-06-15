#!/usr/bin/env python
"""Olver 차원법칙 실험 (ALGEBRAIC_DERIVATION_CLOSURE §5.14 실험2·최강 반증가능, 2026-06-15).

가설 H-차원 (Olver 여차원): 표면군 차원 s 늘리면 불변 부분공간 유효차원 n−s 단조↓.
사전등록 예측: eff_dim_invariant(naming) > (+value) > (+format) · 변이부 차원(=s) 단조↑.
반증: 무관계/비단조 → H-차원 기각.

방법(§5.11 𝒜=T_k∘P_G):
  각 입력에 표면군 G의 무작위 변환 K개 적용 → 표현 추출 → orbit-평균 = P_G 불변부 추정.
  불변 eff-dim = between-input orbit-mean 공분산 PR. 변이 eff-dim = within-orbit 잔차 PR(≈s).
  3 조건(누적군): naming ⊂ naming+value ⊂ naming+value+format.

표면 = *진짜 군*만(§5.14 caveat2): naming=tool/field 재명명(치환), value=식별자 reformat,
       format=구조 렌더 변형(유한). 의미/패러프레이즈=비군 → 제외.

Usage: olver_dimension_experiment.py --model <path|hf> --n 40 --k 16 --layers 14,21,28 --out <json>
"""
import argparse, json, random, re, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RND = random.Random(42)
ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"


def rand_tok(n=8):
    return "".join(RND.choice(ALNUM) for _ in range(n))


def fmt_preserve(s):
    """식별자 포맷 보존 무작위 치환 (#W123→#W..., 숫자→숫자, 이메일→이메일)."""
    def repl_char(c):
        if c.isdigit():
            return RND.choice("0123456789")
        if c.isalpha():
            return RND.choice("abcdefghijklmnopqrstuvwxyz") if c.islower() else RND.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return c
    return "".join(repl_char(c) for c in s)


# ── 입력 사양 추출 (sop_rand2) ──
def load_specs(path, n):
    specs = []
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        tools = d.get("tools") or []
        msgs = d.get("messages") or []
        sys_m = next((m["content"] for m in msgs if m.get("role") == "system"), "")
        usr_m = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        call = None
        for m in msgs:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                fn = m["tool_calls"][0].get("function", m["tool_calls"][0])
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                call = (fn.get("name"), args or {})
                break
        if not (tools and usr_m and call):
            continue
        tnames = [t["function"]["name"] for t in tools]
        fields = sorted({k for t in tools for k in (t["function"].get("parameters", {}).get("properties", {}) or {})})
        specs.append({"sys": "You are a tool-using assistant.", "user": usr_m[:200], "tnames": tnames[:27],
                      "fields": fields[:40], "call": call})
        if len(specs) >= n:
            break
    return specs


# ── ★통제 sweep: 표면토큰의 fraction f 만 무작위 relabel/reformat (같은 변환·분량만 증가) ──
# 군차원 s ∝ (relabel된 토큰수) ∝ f → Olver n−s 직선 테스트 ("수 vs 수" 최강).
def augment_frac(spec, f):
    tnames, fields = spec["tnames"], spec["fields"]
    cname, cargs = spec["call"]
    # 후보 표면토큰 = tool명 ∪ field명 ∪ 값(len>=4). 각각 독립 relabel-가능 dim.
    surf_tools = list(tnames)
    surf_fields = list(fields)
    surf_vals = [str(v) for v in cargs.values() if isinstance(v, str) and len(str(v)) >= 4]
    pool = [("T", t) for t in surf_tools] + [("F", x) for x in surf_fields] + [("V", v) for v in surf_vals]
    m = int(round(f * len(pool)))
    chosen_idx = set(RND.sample(range(len(pool)), m)) if m and len(pool) else set()
    tmap, fmap, vmap = {}, {}, {}
    for j, (kind, val) in enumerate(pool):
        if j not in chosen_idx:
            continue
        if kind == "T":
            tmap[val] = "tool_" + rand_tok(6)
        elif kind == "F":
            fmap[val] = "arg_" + rand_tok(5)
        else:
            vmap[val] = fmt_preserve(val)
    tnames2 = [tmap.get(t, t) for t in tnames]
    cname2 = tmap.get(cname, cname)
    cargs2 = {fmap.get(k, k): vmap.get(str(v), v) for k, v in cargs.items()}
    user2 = spec["user"]
    for v, v2 in vmap.items():
        user2 = user2.replace(v, v2)
    argstr = ", ".join("%s=%s" % (k, v) for k, v in cargs2.items())
    tools = "Available tools: " + ", ".join(tnames2)
    return "%s\n%s\nUser: %s\nAssistant action: %s(%s)" % (spec["sys"], tools, user2, cname2, argstr)


# ── 표현 추출 ──
@torch.no_grad()
def reps(model, tok, texts, layers, device, bs=8):
    """last non-pad 토큰 표현 (action에 민감; mean-pool은 불변 system에 희석됨)."""
    out = {L: [] for L in layers}
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple [n+1] of [b,seq,d]
        last = enc["attention_mask"].sum(1) - 1  # 마지막 non-pad 인덱스 (right-pad 가정)
        bidx = torch.arange(len(batch), device=device)
        for L in layers:
            pooled = hs[L][bidx, last]  # [b, d] last-token
            out[L].append(pooled.float().cpu().numpy())
    return {L: np.concatenate(v, 0) for L, v in out.items()}


def eff_dim(X):
    """participation ratio = (Σλ)²/Σλ² of covariance."""
    X = X - X.mean(0, keepdims=True)
    C = X.T @ X / max(1, len(X))
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 1e-10]
    if ev.size == 0 or (ev ** 2).sum() == 0:
        return 0.0
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/woori/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
    ap.add_argument("--data", default="/home/woori/scratch/fc_build/sop_rand2.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--layers", default="14,21,28")
    ap.add_argument("--fracs", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--out", default="/home/woori/scratch/olver_result.json")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    fracs = [float(x) for x in a.fracs.split(",")]

    specs = load_specs(a.data, a.n)
    print("[data] %d input specs" % len(specs), flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    device = "cuda:0"

    result = {"layers": layers, "n": len(specs), "k": a.k, "fracs": fracs, "by_layer": {}}
    for L in layers:
        result["by_layer"][L] = {}

    for f in fracs:
        all_texts, idx = [], []
        for i, sp in enumerate(specs):
            for _ in range(a.k):
                all_texts.append(augment_frac(sp, f))
                idx.append(i)
        R = reps(model, tok, all_texts, layers, device)
        idx = np.array(idx)
        for L in layers:
            X = R[L]
            mus = np.stack([X[idx == i].mean(0) for i in range(len(specs))])
            resid = np.concatenate([X[idx == i] - X[idx == i].mean(0) for i in range(len(specs))], 0)
            inv_dim = eff_dim(mus)
            var_dim = eff_dim(resid) if f > 0 else 0.0
            var_frac = float(resid.var() / X.var()) if f > 0 else 0.0
            result["by_layer"][L]["%.2f" % f] = {"inv_dim": inv_dim, "var_dim": var_dim, "var_frac": var_frac}
            print("[f=%.2f L%d] inv_dim=%.2f var_dim=%.2f var_frac=%.3f" % (f, L, inv_dim, var_dim, var_frac), flush=True)

    json.dump(result, open(a.out, "w"), indent=1)
    # 사전등록 판정: f↑(군차원↑) → inv_dim 단조↓ (Olver n−s), var_dim 단조↑
    print("\n=== ★H-차원 판정 (예측: f↑ → inv_dim 단조↓·var_dim 단조↑) ===", flush=True)
    for L in layers:
        iv = [result["by_layer"][L]["%.2f" % f]["inv_dim"] for f in fracs]
        vv = [result["by_layer"][L]["%.2f" % f]["var_dim"] for f in fracs]
        # Spearman 류 단조성: 인접쌍 감소/증가 비율
        inv_dec = sum(iv[j] > iv[j + 1] for j in range(len(iv) - 1))
        var_inc = sum(vv[j] < vv[j + 1] for j in range(len(vv) - 1))
        # 선형 상관 (f vs inv_dim)
        corr = float(np.corrcoef(fracs, iv)[0, 1])
        print("  L%d: inv_dim %s (감소쌍 %d/%d·corr=%.2f) | var_dim %s (증가쌍 %d/%d)" %
              (L, [round(x, 1) for x in iv], inv_dec, len(iv) - 1, corr,
               [round(x, 1) for x in vv], var_inc, len(vv) - 1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
