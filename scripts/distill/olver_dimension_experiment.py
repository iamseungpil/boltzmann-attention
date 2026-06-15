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


# ── 표면군 변환 (누적) ──
def augment(spec, cond):
    """cond ∈ {naming, value, format}. 누적: value⊃naming, format⊃value+naming."""
    tnames, fields = spec["tnames"], spec["fields"]
    cname, cargs = spec["call"]
    # naming: tool/field 재명명 (bijection)
    tmap = {t: "tool_" + rand_tok(6) for t in tnames}
    fmap = {f: "arg_" + rand_tok(5) for f in fields}
    tnames2 = [tmap[t] for t in tnames]
    cname2 = tmap.get(cname, cname)
    cargs2 = {fmap.get(k, k): v for k, v in cargs.items()}
    user2 = spec["user"]
    # value: 식별자 값 reformat (naming 위에)
    if cond in ("value", "format"):
        newargs = {}
        for k, v in cargs2.items():
            if isinstance(v, str) and len(v) >= 4:
                v2 = fmt_preserve(v)
                user2 = user2.replace(v, v2)  # user 발화의 동일값도 일관 치환
                newargs[k] = v2
            else:
                newargs[k] = v
        cargs2 = newargs
    # format: 고차원 연속 섭동 (4-템플릿=저차원·magnitude 아티팩트 → arg-순서 셔플 +
    #         무작위 구분자/공백 = 많은 독립 dim·군차원 진짜 증가). naming/value는 fmt=고정.
    fmt_random = (cond == "format")
    return render(spec["sys"], user2, tnames2, cname2, cargs2, fmt_random)


def render(sysm, user, tnames, cname, cargs, fmt_random):
    items = list(cargs.items())
    if fmt_random:
        RND.shuffle(tnames)          # tool-list 순서 (permutation dim)
        RND.shuffle(items)           # arg-pair 순서 (permutation dim)
        sep = RND.choice([", ", " , ", ",  ", ", "])  # 무작위 구분자/공백
        tsep = RND.choice([", ", " | ", " / ", "  "])
        eq = RND.choice(["=", " = ", ": ", "="])
    else:
        sep, tsep, eq = ", ", ", ", "="
    argstr = sep.join("%s%s%s" % (k, eq, v) for k, v in items)
    tools = "Available tools: " + tsep.join(tnames)
    return "%s\n%s\nUser: %s\nAssistant action: %s(%s)" % (sysm, tools, user, cname, argstr)


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
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/woori/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
    ap.add_argument("--data", default="/home/woori/scratch/fc_build/sop_rand2.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--layers", default="14,21,28")
    ap.add_argument("--out", default="/home/woori/scratch/olver_result.json")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    specs = load_specs(a.data, a.n)
    print("[data] %d input specs" % len(specs), flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    device = "cuda:0"

    conditions = ["naming", "value", "format"]
    result = {"layers": layers, "n": len(specs), "k": a.k, "conditions": conditions, "by_layer": {}}
    for L in layers:
        result["by_layer"][L] = {}

    for cond in conditions:
        # 각 입력 i에 K augmentation → 표현
        all_texts, idx = [], []
        for i, sp in enumerate(specs):
            for _ in range(a.k):
                all_texts.append(augment(sp, cond))
                idx.append(i)
        R = reps(model, tok, all_texts, layers, device)
        idx = np.array(idx)
        for L in layers:
            X = R[L]  # [n*k, d]
            # orbit-mean per input = 불변부 P_G 추정
            mus = np.stack([X[idx == i].mean(0) for i in range(len(specs))])  # [n, d]
            # 변이부 잔차 (within-orbit) pooled
            resid = np.concatenate([X[idx == i] - X[idx == i].mean(0) for i in range(len(specs))], 0)
            inv_dim = eff_dim(mus)        # 불변 유효차원 (between-input orbit-means)
            var_dim = eff_dim(resid)      # 변이 유효차원 (≈ 궤도차원 s)
            var_frac = float(resid.var() / X.var())  # 변이 분산 비율
            result["by_layer"][L][cond] = {"inv_dim": inv_dim, "var_dim": var_dim, "var_frac": var_frac}
            print("[%s L%d] inv_dim=%.2f var_dim=%.2f var_frac=%.3f" % (cond, L, inv_dim, var_dim, var_frac), flush=True)

    json.dump(result, open(a.out, "w"), indent=1)
    # 사전등록 판정
    print("\n=== ★H-차원 판정 (예측: inv_dim 단조↓·var_dim 단조↑) ===", flush=True)
    for L in layers:
        iv = [result["by_layer"][L][c]["inv_dim"] for c in conditions]
        vv = [result["by_layer"][L][c]["var_dim"] for c in conditions]
        inv_mono = iv[0] > iv[1] > iv[2]
        var_mono = vv[0] < vv[1] < vv[2]
        print("  L%d: inv_dim %s %s | var_dim %s %s" %
              (L, [round(x, 1) for x in iv], "✓단조↓" if inv_mono else "✗",
               [round(x, 1) for x in vv], "✓단조↑" if var_mono else "✗"), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
