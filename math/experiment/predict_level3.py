"""
Level 3 예측: FP16 어텐션 gap 측정 → NIAH/LITM 절대 성공률 예측
================================================================

Level 1: Σ₀만으로 MSE 순서 예측 (완료)
Level 2: D_actual로 PPL 예측 (완료)
Level 3: FP16 gap + σ_error → NIAH/LITM 절대 성공률 예측 (이 스크립트)

이론 공식:
  P(NIAH 성공 | 위치 p) = Φ(gap(p) / σ_error(M))

  gap(p) = a_needle(p) - max_{i≠needle} a_i
         = FP16에서 needle의 어텐션 점수 - 2위 점수

  σ_error(M) = √(D_attn(M) / d)
             = √(G(M,Σ₀) · c_Q(B) / d) · 2^{-B}

  D_attn은 Level 1에서 계산한 공분산 통계량에서 결정.

방법별 σ_error 비율:
  fokvq_full:    σ_base × √(lm_ratio / am_gm_w)
  PreRoPE_PCA:   σ_base × 1.0
  KVTC:          σ_base × √(eta_head)
  TurboQuant:    σ_base × √(R_aniso)
  KIVI:          σ_base × √(R_aniso × hadamard_extra)  (클래스 C 밖, 근사)

Usage:
  python predict_level3.py --model-name Qwen/Qwen2.5-7B --device cuda:1
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# 이론 상수
C_UNI = {2: 0.1175, 3: 0.03268, 4: 0.00874}
C_LM  = {2: 0.09497, 3: 0.02329, 4: 0.00561}


def parse_args():
    p = argparse.ArgumentParser(description="Level 3: NIAH/LITM absolute prediction")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--context-len", type=int, default=4096)
    p.add_argument("--needle-depths", nargs="+", type=float,
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--n-calib-tokens", type=int, default=2000)
    p.add_argument("--output-dir", type=str, default="results/predictions_level3")
    p.add_argument("--cache-dir", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ============================================================================
# 모델 로딩
# ============================================================================

def load_model(model_name, device, cache_dir=""):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    d_model = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    d_head = d_model // n_heads
    n_layers = config.num_hidden_layers
    G = n_heads // n_kv_heads

    model_info = {
        "name": model_name, "d_model": d_model, "d_head": d_head,
        "n_heads": n_heads, "n_kv_heads": n_kv_heads,
        "n_layers": n_layers, "G": G,
    }

    lowered = model_name.lower()
    dtype = torch.bfloat16 if any(k in lowered for k in ["qwen","llama","mistral"]) else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
        attn_implementation="eager", trust_remote_code=True, **kwargs)
    model.eval()
    return tokenizer, model, model_info


# ============================================================================
# Step 1: Pre-RoPE 공분산에서 σ_error 계산
# ============================================================================

def compute_sigma_errors(model, tokenizer, model_info, device, n_calib=2000, cache_dir=""):
    """캘리브레이션에서 Pre-RoPE Σ₀ 계산 → 방법별 σ_error 산출"""
    from datasets import load_dataset

    d = model_info["d_head"]
    n_kv = model_info["n_kv_heads"]
    n_heads = model_info["n_heads"]
    n_layers = model_info["n_layers"]
    G = model_info["G"]

    # WikiText-2 캘리브레이션
    ds_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", **ds_kwargs)
    text = "\n".join([t for t in ds["text"] if t.strip()])
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :n_calib].to(device)
    print(f"  캘리브레이션: {ids.shape[1]} 토큰")

    # K, Q 수집 (hook)
    K_data, Q_data = {}, {}
    hooks = []

    def make_hook(storage, layer_idx):
        def fn(module, inp, out):
            storage.setdefault(layer_idx, []).append(out.detach().cpu().float())
        return fn

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parts = name.split('.')
        layer_idx = None
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break
        if layer_idx is None:
            continue
        if name.endswith('.k_proj'):
            hooks.append(module.register_forward_hook(make_hook(K_data, layer_idx)))
        elif name.endswith('.q_proj'):
            hooks.append(module.register_forward_hook(make_hook(Q_data, layer_idx)))

    with torch.no_grad():
        for s in range(0, ids.shape[1], 512):
            model(ids[:, s:min(s+512, ids.shape[1])])
    for h in hooks:
        h.remove()

    # 통계량 계산
    all_R_aniso, all_am_gm_w, all_G_PCA, all_G_norot = [], [], [], []
    all_Sigma_K = []

    for li in range(n_layers):
        K_cat = torch.cat(K_data[li], dim=1).squeeze(0).reshape(-1, n_kv, d).numpy()
        Q_cat = torch.cat(Q_data[li], dim=1).squeeze(0).reshape(-1, n_heads, d).numpy()

        for hi in range(n_kv):
            K_h = K_cat[:, hi, :]
            K_c = K_h - K_h.mean(0, keepdims=True)
            Sigma_K = (K_c.T @ K_c) / K_h.shape[0]
            all_Sigma_K.append(Sigma_K)

            eigs = np.maximum(np.linalg.eigvalsh(Sigma_K), 1e-30)
            am = eigs.mean()
            gm = np.exp(np.mean(np.log(eigs)))
            all_R_aniso.append(am / gm)
            all_G_PCA.append(gm)

            diag = np.maximum(np.diag(Sigma_K), 1e-30)
            all_G_norot.append(np.exp(np.mean(np.log(diag))))

            # Q 가중치
            Sigma_Q_eff = np.zeros((d, d))
            for g in range(hi * G, min((hi + 1) * G, n_heads)):
                Q_g = Q_cat[:, g, :]
                Q_c = Q_g - Q_g.mean(0, keepdims=True)
                Sigma_Q_eff += (Q_c.T @ Q_c) / Q_g.shape[0]
            Sigma_Q_eff /= G

            V = np.linalg.eigh(Sigma_K)[1][:, ::-1]
            w = np.maximum(np.diag(V.T @ Sigma_Q_eff @ V), 1e-30)
            w_n = w / w.sum()
            all_am_gm_w.append(w_n.mean() / np.exp(np.mean(np.log(w_n))))

    # η_head (레이어 내 공유 패널티, log-det)
    eta_per_layer = []
    for li in range(n_layers):
        sigs = all_Sigma_K[li * n_kv:(li + 1) * n_kv]
        if len(sigs) < 2:
            eta_per_layer.append(1.0)
            continue
        shared = np.mean(sigs, axis=0)
        logG_s = np.mean(np.log(np.maximum(np.linalg.eigvalsh(shared), 1e-30)))
        logG_h = np.mean([np.mean(np.log(np.maximum(np.linalg.eigvalsh(s), 1e-30))) for s in sigs])
        eta_per_layer.append(max(float(np.exp(logG_s - logG_h)), 1.0))

    stats = {
        "R_aniso": float(np.mean(all_R_aniso)),
        "AM_GM_w": float(np.mean(all_am_gm_w)),
        "G_PCA": float(np.mean(all_G_PCA)),
        "G_norot": float(np.mean(all_G_norot)),
        "eta_head": float(np.mean(eta_per_layer)),
        "hadamard_ratio": float(np.mean(all_G_norot)) / float(np.mean(all_G_PCA)),
    }

    print(f"  R_aniso={stats['R_aniso']:.2f}, η_head={stats['eta_head']:.3f}, AM/GM(w)={stats['AM_GM_w']:.3f}")
    print(f"  G_PCA={stats['G_PCA']:.6f}, Hadamard={stats['hadamard_ratio']:.3f}")

    # 방법별 σ_error 계산 (비트별)
    sigma_errors = {}
    for bits in [2, 3, 4]:
        # 기준: PreRoPE PCA + WF + Uniform
        # D_attn_base = d · G_PCA · c_uni(B)  (2^{-2B}는 c_uni에 포함 안 됨)
        # 실제: D_attn_base_per_dim = G_PCA · c_uni(B) · 2^{-2B}
        # 하지만 c_uni 자체가 "N(0,1)의 b-bit 양자화 MSE/분산" 이므로
        # D_j = λ_j · c_uni(b_j)  (water-filling 후)
        # D_attn = Σ w_j · D_j ≈ G_PCA · c_uni(B) (water-filling 등가)
        # σ_error = √(D_attn / d) = √(G_PCA · c_uni(B) / d)
        # 하지만 이것은 2^{-2B}를 포함하지 않는다... c_uni가 이미 MSE/분산 비율이므로
        # 실제 MSE = λ_j · c_uni(B) (단위: 원래 스케일의 분산)
        # σ_error = √(mean_j(w_j · λ_j · c_uni(B))) = √(G_PCA · c_uni(B))... 이것은 근사

        # 더 정확하게: water-filling 후 각 차원의 MSE는 θ (water level)
        # D_total = d · θ = d · GM(λ) · c_uni(B)
        # → per-dim MSE = GM(λ) · c_uni(B)
        # σ_error = √(per-dim D_attn) = √(GM(λ) · c_uni(B))

        base_var = stats["G_PCA"] * C_UNI[bits]  # 기준 방법의 per-dim 어텐션 왜곡
        sigma_base = np.sqrt(base_var)

        lm_ratio = C_LM[bits] / C_UNI[bits]

        sigma_errors[bits] = {
            "fokvq_full":    sigma_base * np.sqrt(lm_ratio / stats["AM_GM_w"]),
            "PreRoPE_PCA":   sigma_base,
            "KVTC":          sigma_base * np.sqrt(stats["eta_head"]),
            "TurboQuant":    sigma_base * np.sqrt(stats["R_aniso"]),
            "No_rotation":   sigma_base * np.sqrt(stats["hadamard_ratio"]),
            # KIVI: 비회전 + 채널별 → No_rotation과 유사하지만 비대칭 양자화 효과로 약간 다름
            "KIVI_approx":   sigma_base * np.sqrt(stats["hadamard_ratio"] * 1.2),  # 20% 보수적 추정
        }

    return stats, sigma_errors


# ============================================================================
# Step 2: FP16 어텐션 gap 측정 (NIAH 시뮬레이션)
# ============================================================================

def measure_attention_gaps(model, tokenizer, model_info, device,
                           context_len=4096, depths=[0.1,0.3,0.5,0.7,0.9]):
    """
    NIAH 시뮬레이션: needle을 삽입하고 FP16에서 어텐션 gap을 측정한다.
    gap(p) = needle 위치의 어텐션 점수 - 2위 점수
    """
    print(f"\n[Step 2] FP16 어텐션 gap 측정 (context={context_len})")

    # needle/haystack 텍스트
    needle = "The secret password is: HAMBURGER-42."
    haystack_unit = "This is a filler sentence for testing purposes. It contains no useful information whatsoever. "

    # 컨텍스트 구성
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
    haystack_tokens = tokenizer.encode(haystack_unit * 200, add_special_tokens=False)

    gaps_by_depth = {}

    for depth in depths:
        # needle 삽입 위치
        needle_pos = int(context_len * depth)
        needle_pos = max(1, min(needle_pos, context_len - len(needle_tokens) - 1))

        # 입력 구성: haystack[:pos] + needle + haystack[pos:]
        hay_before = haystack_tokens[:needle_pos]
        hay_after = haystack_tokens[:context_len - needle_pos - len(needle_tokens)]
        input_ids = hay_before + needle_tokens + hay_after
        input_ids = input_ids[:context_len]

        input_tensor = torch.tensor([input_ids], device=device)

        # needle 토큰 범위
        needle_start = len(hay_before)
        needle_end = needle_start + len(needle_tokens)

        # FP16 순전파 — output_attentions=True는 OOM 위험
        # 대신 hook으로 마지막 1/4 레이어의 어텐션만 수집
        n_layers = model_info["n_layers"]
        n_heads_total = model_info["n_heads"]
        start_layer = n_layers * 3 // 4  # 마지막 1/4만

        attn_collected = {}  # layer_idx -> (n_heads, seq) 마지막 토큰의 어텐션

        hooks_attn = []

        def make_attn_hook(layer_idx):
            def fn(module, args, kwargs, output):
                # Qwen2/Llama/Mistral의 attention forward는
                # (attn_output, attn_weights, past_kv)를 반환
                # output_attentions=True일 때만 attn_weights가 있음
                # hook에서 직접 Q,K를 계산하는 것이 더 안전
                pass  # 아래에서 Q,K hook으로 대체
            return fn

        # Q, K hook으로 어텐션 점수를 직접 계산 (메모리 효율적)
        Q_for_gap = {}
        K_for_gap = {}

        def make_qk_hook(storage, layer_idx):
            def fn(module, inp, out):
                if layer_idx >= start_layer:
                    storage[layer_idx] = out.detach()  # GPU에 유지
            return fn

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            parts = name.split('.')
            layer_idx = None
            for p in parts:
                if p.isdigit():
                    layer_idx = int(p)
                    break
            if layer_idx is None or layer_idx < start_layer:
                continue
            if name.endswith('.q_proj'):
                hooks_attn.append(module.register_forward_hook(make_qk_hook(Q_for_gap, layer_idx)))
            elif name.endswith('.k_proj'):
                hooks_attn.append(module.register_forward_hook(make_qk_hook(K_for_gap, layer_idx)))

        with torch.no_grad():
            model(input_tensor)

        for h in hooks_attn:
            h.remove()

        # Q, K에서 어텐션 점수 직접 계산 (마지막 토큰 query만)
        d_head = model_info["d_head"]
        n_kv = model_info["n_kv_heads"]
        G_ratio = model_info["G"]

        layer_gaps = []
        for layer_idx in range(start_layer, n_layers):
            if layer_idx not in Q_for_gap or layer_idx not in K_for_gap:
                continue
            Q_raw = Q_for_gap[layer_idx]  # (1, seq, n_heads*d_head)
            K_raw = K_for_gap[layer_idx]  # (1, seq, n_kv*d_head)

            seq_len = Q_raw.shape[1]
            Q_h = Q_raw.reshape(1, seq_len, n_heads_total, d_head).transpose(1, 2)  # (1, n_heads, seq, d)
            K_h = K_raw.reshape(1, seq_len, n_kv, d_head).transpose(1, 2)  # (1, n_kv, seq, d)

            # GQA: K를 반복하여 Q와 맞춤
            K_expanded = K_h.repeat_interleave(G_ratio, dim=1)  # (1, n_heads, seq, d)

            # 마지막 토큰의 query만
            q_last = Q_h[:, :, -1:, :]  # (1, n_heads, 1, d)

            # 어텐션 점수: q_last @ K^T / sqrt(d)
            attn_scores = (q_last @ K_expanded.transpose(-1, -2)).squeeze(2) / (d_head ** 0.5)
            # (1, n_heads, seq)

            # softmax
            attn_probs = F.softmax(attn_scores.float(), dim=-1)  # (1, n_heads, seq)
            attn_probs = attn_probs[0]  # (n_heads, seq)

            # needle gap
            needle_attn = attn_probs[:, needle_start:needle_end].sum(dim=-1)  # (n_heads,)
            non_needle_mask = torch.ones(seq_len, device=device, dtype=torch.bool)
            non_needle_mask[needle_start:needle_end] = False
            max_non_needle = attn_probs[:, non_needle_mask].max(dim=-1).values

            gap_tensor = (needle_attn - max_non_needle).cpu().float().numpy()
            layer_gaps.append(gap_tensor)

            # 메모리 해제
            del Q_raw, K_raw, Q_h, K_h, K_expanded, attn_scores, attn_probs

        del Q_for_gap, K_for_gap
        torch.cuda.empty_cache()

        if layer_gaps:
            avg_gap = np.mean([g.mean() for g in layer_gaps])
            min_gap = float(min(g.min() for g in layer_gaps))
            max_gap = float(max(g.max() for g in layer_gaps))
        else:
            avg_gap, min_gap, max_gap = 0.0, 0.0, 0.0

        gaps_by_depth[depth] = {
            "avg_gap": float(avg_gap),
            "min_gap": min_gap,
            "max_gap": max_gap,
            "needle_pos": needle_pos,
        }

        print(f"  depth={depth:.1f}: avg_gap={avg_gap:.4f}, min={min_gap:.4f}, max={max_gap:.4f}")

    return gaps_by_depth


# ============================================================================
# Step 3: 절대 성공률 예측
# ============================================================================

def predict_absolute(gaps, sigma_errors, model_info, bits_list):
    """gap과 σ_error로 NIAH/LITM의 절대 성공률을 예측한다."""
    from scipy.stats import norm

    results = {}

    for bits in bits_list:
        se = sigma_errors[bits]
        results[bits] = {}

        for depth, gap_info in gaps.items():
            gap = gap_info["avg_gap"]
            results[bits][depth] = {}

            for method, sigma in se.items():
                if sigma < 1e-12:
                    p = 1.0
                elif gap <= 0:
                    # gap이 음수 = FP16에서도 needle이 밀림 → 양자화 관계없이 실패
                    p = float(norm.cdf(gap / sigma))
                else:
                    p = float(norm.cdf(gap / sigma))
                results[bits][depth][method] = {"p_success": p, "sigma": float(sigma), "gap": float(gap)}

    return results


# ============================================================================
# 출력
# ============================================================================

def print_results(gaps, sigma_errors, predictions, model_info, stats, bits_list):
    """결과를 예측표로 출력한다."""
    from scipy.stats import norm

    model = model_info["name"]
    d = model_info["d_head"]

    print(f"\n{'=' * 100}")
    print(f"Level 3 예측 결과: {model}")
    print(f"{'=' * 100}")

    # gap 프로파일
    print(f"\n[FP16 어텐션 gap 프로파일]")
    print(f"  {'depth':>6} {'avg_gap':>10} {'min_gap':>10} {'max_gap':>10}")
    print(f"  {'─' * 40}")
    for depth in sorted(gaps.keys()):
        g = gaps[depth]
        print(f"  {depth:>6.1f} {g['avg_gap']:>10.4f} {g['min_gap']:>10.4f} {g['max_gap']:>10.4f}")

    # σ_error 표
    print(f"\n[방법별 σ_error]")
    methods = ["fokvq_full", "PreRoPE_PCA", "KVTC", "TurboQuant", "KIVI_approx"]
    for bits in bits_list:
        print(f"\n  {bits}-bit:")
        print(f"  {'방법':<18} {'σ_error':>10} {'σ/σ_base':>10}")
        print(f"  {'─' * 40}")
        base = sigma_errors[bits]["PreRoPE_PCA"]
        for m in methods:
            se = sigma_errors[bits][m]
            print(f"  {m:<18} {se:>10.6f} {se/base:>10.3f}x")

    # NIAH 절대 성공률
    for bits in bits_list:
        print(f"\n{'─' * 100}")
        print(f"[{bits}-bit NIAH 절대 성공률 예측 (%)]")
        print(f"{'depth':>6}", end="")
        for m in methods:
            print(f" {m:>15}", end="")
        print()
        print("─" * 90)

        for depth in sorted(predictions[bits].keys()):
            print(f"{depth:>6.1f}", end="")
            for m in methods:
                p = predictions[bits][depth][m]["p_success"]
                print(f" {p*100:>14.1f}%", end="")
            print()

    # KVTC 보고값과 비교 가능한 형태의 요약
    print(f"\n{'=' * 100}")
    print(f"KVTC 비교용 요약 (비트별 평균 NIAH 성공률)")
    print(f"{'=' * 100}")

    depths = sorted(gaps.keys())
    for bits in bits_list:
        print(f"\n[{bits}-bit]")
        print(f"{'방법':<18} {'평균':>8} {'최소(중간)':>10} {'최대(끝)':>10}")
        print("─" * 50)
        for m in methods:
            vals = [predictions[bits][d][m]["p_success"] for d in depths]
            avg = np.mean(vals)
            mn = min(vals)
            mx = max(vals)
            print(f"{m:<18} {avg*100:>7.1f}% {mn*100:>9.1f}% {mx*100:>9.1f}%")

    # fokvq_full이 KVTC를 이기는 조건 분석
    print(f"\n{'=' * 100}")
    print(f"fokvq_full vs KVTC: depth별 성공률 차이")
    print(f"{'=' * 100}")
    for bits in bits_list:
        print(f"\n[{bits}-bit]")
        print(f"{'depth':>6} {'fokvq_full':>12} {'KVTC':>12} {'차이':>10} {'fokvq 우위':>10}")
        print("─" * 55)
        for depth in depths:
            p_full = predictions[bits][depth]["fokvq_full"]["p_success"]
            p_kvtc = predictions[bits][depth]["KVTC"]["p_success"]
            diff = p_full - p_kvtc
            win = "✓" if diff > 0.001 else ("≈" if abs(diff) < 0.001 else "✗")
            print(f"{depth:>6.1f} {p_full*100:>11.1f}% {p_kvtc*100:>11.1f}% {diff*100:>+9.1f}% {win:>10}")


# ============================================================================
# main
# ============================================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    tokenizer, model, model_info = load_model(args.model_name, args.device, args.cache_dir)

    # Step 1: 공분산 → σ_error
    print(f"\n[Step 1] Pre-RoPE 공분산 → σ_error 계산")
    stats, sigma_errors = compute_sigma_errors(
        model, tokenizer, model_info, args.device, args.n_calib_tokens, args.cache_dir)

    # Step 2: FP16 gap 측정
    gaps = measure_attention_gaps(
        model, tokenizer, model_info, args.device,
        args.context_len, args.needle_depths)

    # 모델 해제
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: 절대 성공률 예측
    print(f"\n[Step 3] 절대 성공률 예측")
    predictions = predict_absolute(gaps, sigma_errors, model_info, args.bits)

    # 출력
    print_results(gaps, sigma_errors, predictions, model_info, stats, args.bits)

    # 저장
    model_key = args.model_name.replace("/", "_")
    out_file = output_dir / f"level3_{model_key}_{datetime.now():%Y%m%d_%H%M%S}.json"

    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(out_file, "w") as f:
        json.dump({
            "model": model_info, "stats": stats,
            "gaps": {str(k): v for k, v in gaps.items()},
            "sigma_errors": {str(k): {m: float(v) for m, v in se.items()} for k, se in sigma_errors.items()},
            "predictions": {str(b): {str(d): {m: v for m, v in mv.items()} for d, mv in bv.items()} for b, bv in predictions.items()},
            "args": vars(args), "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=conv)

    print(f"\n저장: {out_file}")


if __name__ == "__main__":
    main()
