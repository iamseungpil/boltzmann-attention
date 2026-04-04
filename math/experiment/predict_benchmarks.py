"""
Lie Group Framework: 6-Benchmark Prediction from Pre-RoPE Covariance
====================================================================

이 스크립트는 모델의 Pre-RoPE 키/쿼리 공분산 Σ₀, Σ_Q를 계산하고,
Lie 군 프레임워크의 이론 공식으로 6개 벤치마크 예측값을 산출한다.

6개 벤치마크:
  1. PPL (WikiText-2)     — Level 2: ln(PPL/fp16) ≈ c · D_actual
  2. NIAH                 — Level 3: P(성공) = Φ(gap/σ_error)
  3. LITM                 — Level 3: U자형 정합
  4. RULER-VT             — Level 3: 다중 검색 AND
  5. Qasper               — Level 2: PPL과 유사
  6. MMLU                 — Level 2: 약한 상관

비교 방법 (클래스 C 내):
  - TurboQuant: 무작위 SO(d), 균일 스칼라
  - KVTC: 공유 Pre-RoPE PCA, 균일 스칼라, DP, sink+window
  - Pre-RoPE PCA+WF: 헤드별 PCA, 균일 스칼라, Water-Filling
  - fokvq_full: 헤드별 Pre-RoPE PCA, MK Lloyd-Max, HEAT

Usage:
  python predict_benchmarks.py --model-name Qwen/Qwen2.5-7B --device cuda:1
  python predict_benchmarks.py --model-name meta-llama/Llama-3.1-8B --device cuda:1
  python predict_benchmarks.py --model-name mistralai/Mistral-7B-v0.3 --device cuda:1
  python predict_benchmarks.py --model-name Qwen/Qwen2.5-1.5B --device cuda:1
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="6-Benchmark Prediction from Pre-RoPE Covariance")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--n-calib-tokens", type=int, default=2000,
                   help="캘리브레이션 토큰 수 (기본 2000)")
    p.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--output-dir", type=str,
                   default="results/predictions")
    p.add_argument("--cache-dir", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ============================================================================
# 이론 상수
# ============================================================================

# Lloyd-Max vs Uniform 왜곡 계수 (N(0,1) 표준 상수, Gersho & Gray 1992)
C_UNI = {2: 0.1175, 3: 0.03268, 4: 0.00874}
C_LM  = {2: 0.09497, 3: 0.02329, 4: 0.00561}

# HEAT 전형적 매개변수 (캘리브레이션 전 기본값, 모델별로 업데이트됨)
DEFAULT_HEAT = {"A": 0.3, "kappa1": 0.5, "B": 0.2, "kappa2": 0.02, "C": 0.01}

# NIAH 가상 gap 값 (실측 전 기본값)
NIAH_GAPS = {
    "easy_90pct": 0.50,
    "medium_50pct": 0.10,
    "hard_30pct": 0.05,
    "extreme_10pct": 0.02,
}


# ============================================================================
# 모델 로딩
# ============================================================================

def load_model(model_name: str, device: str, cache_dir: str = ""):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, **kwargs)

    # 모델 구조 정보 추출
    d_model = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    d_head = d_model // n_heads
    n_layers = config.num_hidden_layers
    G = n_heads // n_kv_heads
    rope_theta = getattr(config, 'rope_theta', 10000.0)

    model_info = {
        "name": model_name,
        "d_model": d_model,
        "d_head": d_head,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "n_layers": n_layers,
        "G": G,
        "rope_theta": rope_theta,
    }

    print(f"\n{'='*70}")
    print(f"모델: {model_name}")
    print(f"  d_model={d_model}, d_head={d_head}, layers={n_layers}")
    print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, G={G}")
    print(f"  rope_theta={rope_theta}")
    print(f"{'='*70}")

    # dtype 결정
    lowered = model_name.lower()
    if "qwen" in lowered or "llama" in lowered or "mistral" in lowered:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",
        trust_remote_code=True,
        **kwargs,
    )
    model.eval()

    return tokenizer, model, model_info


# ============================================================================
# Pre-RoPE 키/쿼리 벡터 수집
# ============================================================================

def collect_pre_rope_kq(model, tokenizer, device: str, n_tokens: int = 2000,
                        cache_dir: str = "") -> Tuple[Dict, Dict]:
    """
    캘리브레이션 데이터에서 Pre-RoPE 키/쿼리 벡터를 수집한다.

    Returns:
        K_dict: {layer_idx: tensor (n_tokens, n_kv_heads, d_head)}
        Q_dict: {layer_idx: tensor (n_tokens, n_heads, d_head)}
    """
    from datasets import load_dataset

    ds_kwargs = {}
    if cache_dir:
        ds_kwargs["cache_dir"] = cache_dir
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", **ds_kwargs)
    all_text = "\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(all_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids[:, :n_tokens].to(device)
    actual_tokens = input_ids.shape[1]
    print(f"  캘리브레이션: WikiText-2에서 {actual_tokens} 토큰 사용")

    # k_proj, q_proj 출력을 hook으로 수집
    K_collected = {}  # layer -> list of tensors
    Q_collected = {}

    hooks = []

    def make_k_hook(layer_idx):
        def hook_fn(module, input, output):
            # output: (batch, seq, n_kv_heads * d_head)
            K_collected.setdefault(layer_idx, []).append(output.detach().cpu().float())
        return hook_fn

    def make_q_hook(layer_idx):
        def hook_fn(module, input, output):
            Q_collected.setdefault(layer_idx, []).append(output.detach().cpu().float())
        return hook_fn

    # k_proj, q_proj 모듈 찾기
    for name, module in model.named_modules():
        if name.endswith('.k_proj') and isinstance(module, nn.Linear):
            # 레이어 인덱스 추출
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p.isdigit():
                    layer_idx = int(p)
                    break
            hooks.append(module.register_forward_hook(make_k_hook(layer_idx)))
        elif name.endswith('.q_proj') and isinstance(module, nn.Linear):
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p.isdigit():
                    layer_idx = int(p)
                    break
            hooks.append(module.register_forward_hook(make_q_hook(layer_idx)))

    if not hooks:
        raise RuntimeError("k_proj/q_proj 모듈을 찾을 수 없음")

    print(f"  {len(hooks)}개 hook 설치 (k_proj + q_proj)")

    # 순전파 (청크 단위)
    chunk_size = 512
    with torch.no_grad():
        for start in range(0, actual_tokens, chunk_size):
            end = min(start + chunk_size, actual_tokens)
            chunk_ids = input_ids[:, start:end]
            model(chunk_ids)

    # hook 제거
    for h in hooks:
        h.remove()

    # 텐서 결합 및 reshape
    config = model.config
    n_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    n_heads = config.num_attention_heads
    d_head = config.hidden_size // n_heads

    K_dict = {}
    Q_dict = {}

    for layer_idx in sorted(K_collected.keys()):
        K_cat = torch.cat(K_collected[layer_idx], dim=1)  # (1, total_seq, n_kv*d)
        K_cat = K_cat.squeeze(0)  # (total_seq, n_kv*d)
        K_dict[layer_idx] = K_cat.reshape(-1, n_kv_heads, d_head)  # (seq, n_kv, d)

        Q_cat = torch.cat(Q_collected[layer_idx], dim=1).squeeze(0)
        Q_dict[layer_idx] = Q_cat.reshape(-1, n_heads, d_head)

    n_layers = len(K_dict)
    print(f"  수집 완료: {n_layers} 레이어, K shape={K_dict[0].shape}, Q shape={Q_dict[0].shape}")

    # 메모리 정리
    del K_collected, Q_collected
    gc.collect()

    return K_dict, Q_dict


# ============================================================================
# Σ₀ 계산 및 이론 통계량
# ============================================================================

def compute_covariances(K_dict: Dict, Q_dict: Dict, model_info: Dict) -> Dict:
    """
    각 (레이어, 헤드)에서 Pre-RoPE 공분산과 진단 통계량을 계산한다.
    """
    n_layers = model_info["n_layers"]
    n_kv_heads = model_info["n_kv_heads"]
    n_heads = model_info["n_heads"]
    d = model_info["d_head"]
    G = model_info["G"]

    results = {
        "per_head": [],  # 각 (layer, head)의 통계량
        "global": {},     # 전체 모델 평균
    }

    all_R_aniso = []
    all_eta2 = []
    all_eta4 = []
    all_am_gm_w = []
    all_eigenvalues = []
    all_Sigma_K = []  # Fischer inequality 계산용

    for layer_idx in range(n_layers):
        K_layer = K_dict[layer_idx]   # (seq, n_kv, d)
        Q_layer = Q_dict[layer_idx]   # (seq, n_heads, d)

        for head_idx in range(n_kv_heads):
            K_h = K_layer[:, head_idx, :].numpy()  # (seq, d)
            n_samples = K_h.shape[0]

            # --- Σ_K (Pre-RoPE 키 공분산) ---
            K_centered = K_h - K_h.mean(axis=0, keepdims=True)
            Sigma_K = (K_centered.T @ K_centered) / n_samples  # (d, d)

            # 고유값 분해
            eigenvalues = np.linalg.eigvalsh(Sigma_K)
            eigenvalues = np.sort(eigenvalues)[::-1]  # 내림차순
            eigenvalues = np.maximum(eigenvalues, 1e-12)  # 수치 안정성

            # R_aniso = AM/GM
            am = eigenvalues.mean()
            gm = np.exp(np.mean(np.log(eigenvalues)))
            R_aniso = am / gm

            # η(b) = G_PCA / G_block(b)
            G_PCA = gm  # (det Σ)^{1/d} = GM(eigenvalues)

            # block-2 효율성
            det_blocks_2 = []
            for j in range(0, d, 2):
                if j + 1 < d:
                    block = Sigma_K[j:j+2, j:j+2]
                    det_blocks_2.append(np.linalg.det(block))
            G_block2 = np.exp(np.mean(np.log(np.maximum(det_blocks_2, 1e-24))) / 2)
            eta2 = G_PCA / max(G_block2, 1e-12)

            # block-4 효율성
            det_blocks_4 = []
            for j in range(0, d, 4):
                if j + 3 < d:
                    block = Sigma_K[j:j+4, j:j+4]
                    det_blocks_4.append(np.linalg.det(block))
            if det_blocks_4:
                G_block4 = np.exp(np.mean(np.log(np.maximum(det_blocks_4, 1e-24))) / 4)
                eta4 = G_PCA / max(G_block4, 1e-12)
            else:
                eta4 = eta2

            # G_norot = GM(diag(Sigma_K)) — Hadamard ratio 계산용
            diag_elements = np.diag(Sigma_K)
            diag_elements = np.maximum(diag_elements, 1e-12)
            G_norot = np.exp(np.mean(np.log(diag_elements)))

            # Sigma_K 저장 (Fischer inequality eta_head 계산용)
            all_Sigma_K.append(Sigma_K)

            # --- Σ_Q (쿼리 공분산, GQA 유효) ---
            # GQA: G개의 쿼리 헤드가 하나의 KV 헤드를 공유
            q_start = head_idx * G
            q_end = q_start + G

            # 유효 쿼리 공분산 Σ_Q^{eff}
            Sigma_Q_eff = np.zeros((d, d))
            for g in range(q_start, min(q_end, n_heads)):
                Q_g = Q_layer[:, g, :].numpy()
                Q_centered = Q_g - Q_g.mean(axis=0, keepdims=True)
                Sigma_Q_eff += (Q_centered.T @ Q_centered) / n_samples
            Sigma_Q_eff /= G

            # 쿼리 가중치 w_j (PCA 기저에서)
            # PCA 변환 후 Σ_Q의 대각 원소
            V_pca = np.linalg.eigh(Sigma_K)[1][:, ::-1]  # 고유벡터 (내림차순)
            Sigma_Q_pca = V_pca.T @ Sigma_Q_eff @ V_pca
            w_j = np.diag(Sigma_Q_pca)
            w_j = np.maximum(w_j, 1e-12)
            w_j_normalized = w_j / w_j.sum()

            # AM/GM(w) = MK 추가 이득
            am_w = w_j_normalized.mean()  # = 1/d
            gm_w = np.exp(np.mean(np.log(w_j_normalized)))
            am_gm_w = am_w / gm_w  # ≥ 1

            head_result = {
                "layer": layer_idx,
                "head": head_idx,
                "R_aniso": float(R_aniso),
                "eta2": float(min(eta2, 1.0)),
                "eta4": float(min(eta4, 1.0)),
                "AM_GM_w": float(am_gm_w),
                "G_PCA": float(G_PCA),
                "G_norot": float(G_norot),
                "G_block2": float(G_block2),
                "G_uniform": float(am),
                "trace_K": float(np.trace(Sigma_K)),
                "cond_number": float(eigenvalues[0] / eigenvalues[-1]),
                "top5_eigenvalues": eigenvalues[:5].tolist(),
            }
            results["per_head"].append(head_result)

            all_R_aniso.append(R_aniso)
            all_eta2.append(min(eta2, 1.0))
            all_eta4.append(min(eta4, 1.0))
            all_am_gm_w.append(am_gm_w)
            all_eigenvalues.append(eigenvalues)

    # --- Fischer inequality 기반 η_head 계산 ---
    # KVTC는 cross-layer 공유 PCA를 사용하지만, 레이어 간 스케일 차이로
    # cross-layer η_head가 비현실적으로 큼 (DP 비트할당이 보상하므로).
    # 현실적 비교를 위해 **레이어 내 헤드 공유** η_head를 계산한다.
    # 이것은 "같은 레이어의 KV 헤드가 동일 PCA를 공유할 때의 패널티"이다.

    # --- Fischer inequality 기반 η_head 계산 (log-det 사용, 수치 안정) ---
    # det은 d=128에서 극단적 스케일이므로, log-det으로 계산한다.
    # G = (det Σ)^{1/d} = exp((1/d) * log(det Σ)) = exp((1/d) * Σ log(λ_i))
    # 따라서 log(G) = (1/d) * Σ log(λ_i) = mean(log(eigenvalues))

    n_kv_heads = model_info["n_kv_heads"]
    eta_head_per_layer = []

    for layer_idx in range(n_layers):
        layer_start = layer_idx * n_kv_heads
        layer_end = layer_start + n_kv_heads
        layer_sigmas = all_Sigma_K[layer_start:layer_end]

        if len(layer_sigmas) < 2:
            eta_head_per_layer.append(1.0)
            continue

        # 레이어 내 공유 공분산의 log(G)
        Sigma_shared_layer = np.mean(layer_sigmas, axis=0)
        eigs_shared = np.linalg.eigvalsh(Sigma_shared_layer)
        eigs_shared = np.maximum(eigs_shared, 1e-30)
        logG_shared = np.mean(np.log(eigs_shared))

        # 레이어 내 각 헤드의 log(G) 평균
        logG_per_head_list = []
        for S in layer_sigmas:
            eigs_h = np.linalg.eigvalsh(S)
            eigs_h = np.maximum(eigs_h, 1e-30)
            logG_per_head_list.append(np.mean(np.log(eigs_h)))
        logG_per_head_mean = np.mean(logG_per_head_list)

        # η_head = exp(logG_shared - logG_per_head_mean)
        # Fischer: logG_shared >= logG_per_head_mean 이므로 η >= 1
        eta_l = np.exp(logG_shared - logG_per_head_mean)
        eta_head_per_layer.append(max(float(eta_l), 1.0))

    eta_head_fischer = float(np.mean(eta_head_per_layer))

    # Cross-layer η_head (참조용, KVTC의 실제 cross-layer 공유)
    Sigma_shared_all = np.mean(all_Sigma_K, axis=0)
    eigs_shared_all = np.linalg.eigvalsh(Sigma_shared_all)
    eigs_shared_all = np.maximum(eigs_shared_all, 1e-30)
    logG_shared_all = np.mean(np.log(eigs_shared_all))

    logG_per_head_all = np.mean([
        np.mean(np.log(np.maximum(np.linalg.eigvalsh(S), 1e-30)))
        for S in all_Sigma_K
    ])
    eta_head_cross_layer = float(np.exp(logG_shared_all - logG_per_head_all))

    # 전체 모델 통계
    results["global"] = {
        "R_aniso_mean": float(np.mean(all_R_aniso)),
        "R_aniso_min": float(np.min(all_R_aniso)),
        "R_aniso_max": float(np.max(all_R_aniso)),
        "eta2_mean": float(np.mean(all_eta2)),
        "eta4_mean": float(np.mean(all_eta4)),
        "AM_GM_w_mean": float(np.mean(all_am_gm_w)),
        "AM_GM_w_max": float(np.max(all_am_gm_w)),
        "n_heads_total": len(results["per_head"]),
        "eta_head_fischer": float(eta_head_fischer),
        "eta_head_cross_layer": float(eta_head_cross_layer),
        "logG_shared_all": float(logG_shared_all),
        "logG_per_head_all": float(logG_per_head_all),
    }

    return results


# ============================================================================
# 6개 벤치마크 예측
# ============================================================================

def predict_all_benchmarks(stats: Dict, model_info: Dict, bits_list: List[int]) -> Dict:
    """이론 공식으로 6개 벤치마크 예측값을 계산한다."""

    from scipy.stats import norm

    d = model_info["d_head"]
    G = model_info["G"]
    g = stats["global"]

    R_aniso = g["R_aniso_mean"]
    eta2 = g["eta2_mean"]
    eta4 = g["eta4_mean"]
    am_gm_w = g["AM_GM_w_mean"]

    # η_head: Fischer inequality 기반 정확 계산 (compute_covariances에서 산출)
    eta_head = g["eta_head_fischer"]

    # G_PCA 평균 (sigma_base 계산용)
    G_PCA_mean = np.mean([h["G_PCA"] for h in stats["per_head"]])

    # G_norot 평균 (Hadamard ratio 계산용)
    G_norot_mean = np.mean([h["G_norot"] for h in stats["per_head"]])

    predictions = {"model": model_info["name"], "bits": {}}

    for bits in bits_list:
        lm_ratio = C_LM[bits] / C_UNI[bits]  # Lloyd-Max / Uniform

        # ================================================================
        # D_MSE 상대값 (Euclidean MSE, PPL Level 2 용)
        # ================================================================
        # 기준: Pre-RoPE PCA + WF + Uniform = 1.0
        mse_rel = {}

        # TurboQuant: 무작위 SO(d), 균일 → AM/GM = R_aniso
        mse_rel["TurboQuant"] = R_aniso

        # KVTC: 공유 PCA 패널티 = η_head (Fischer inequality)
        mse_rel["KVTC"] = eta_head

        # Pre-RoPE PCA + WF + Uniform (기준)
        mse_rel["PreRoPE_PCA_WF"] = 1.0

        # fokvq_full: LM만 MSE에 기여, MK는 MSE를 줄이지 않음
        mse_rel["fokvq_full"] = lm_ratio  # c_LM/c_uni only, NOT /am_gm_w

        # No_rotation: Hadamard ratio = G_norot / G_PCA (정확 계산)
        hadamard_ratio = G_norot_mean / max(G_PCA_mean, 1e-12)
        mse_rel["No_rotation"] = hadamard_ratio

        # ================================================================
        # D_attn 상대값 (attention-weighted distortion, NIAH/RULER Level 3 용)
        # ================================================================
        # fokvq_full만 MK 이득(1/am_gm_w)이 추가로 반영됨
        d_attn_rel = {}
        d_attn_rel["TurboQuant"] = R_aniso            # MK 없음 → D_attn ≈ D_MSE
        d_attn_rel["KVTC"] = eta_head                 # MK 없음 → D_attn ≈ D_MSE
        d_attn_rel["PreRoPE_PCA_WF"] = 1.0            # 기준
        d_attn_rel["fokvq_full"] = lm_ratio / am_gm_w # MK가 D_attn을 줄임
        d_attn_rel["No_rotation"] = hadamard_ratio     # MK 없음

        # ================================================================
        # PPL 예측 (Level 2) — D_MSE 기반
        # ================================================================
        # ln(PPL/fp16) ∝ D_actual ∝ D_MSE
        ppl_excess_rel = mse_rel.copy()

        # ================================================================
        # NIAH 예측 (Level 3) — D_attn 기반
        # ================================================================
        # σ_error = √(D_attn) → σ 비율 = √(D_attn 비율)
        sigma_rel = {m: np.sqrt(v) for m, v in d_attn_rel.items()}

        # σ_base: 실측 모델 통계에서 계산
        # D_attn_base_per_dim = G_PCA * c_uni(B), sigma_base = sqrt(D_attn_base_per_dim)
        sigma_base_computed = np.sqrt(G_PCA_mean * C_UNI[bits])

        niah_predictions = {}
        for gap_name, gap_val in NIAH_GAPS.items():
            niah_predictions[gap_name] = {}
            for method, sr in sigma_rel.items():
                sigma_actual = sigma_base_computed * sr
                p_success = float(norm.cdf(gap_val / sigma_actual))
                niah_predictions[gap_name][method] = p_success

        # ================================================================
        # RULER-VT 예측 — D_attn 기반
        # ================================================================
        ruler_predictions = {}
        for n_vars in [3, 5, 10]:
            ruler_predictions[n_vars] = {}
            avg_gap = 0.08
            for method, sr in sigma_rel.items():
                sigma_actual = sigma_base_computed * sr
                p_single = float(norm.cdf(avg_gap / sigma_actual))
                p_vt = p_single ** n_vars
                ruler_predictions[n_vars][method] = p_vt

        predictions["bits"][bits] = {
            "mse_relative": mse_rel,
            "d_attn_relative": d_attn_rel,
            "mse_ranking": sorted(mse_rel.items(), key=lambda x: x[1]),
            "ppl_ranking": sorted(ppl_excess_rel.items(), key=lambda x: x[1]),
            "niah": niah_predictions,
            "ruler_vt": ruler_predictions,
            "sigma_relative": sigma_rel,
            "sigma_base": float(sigma_base_computed),
            "theory_constants": {
                "lm_ratio": lm_ratio,
                "am_gm_w": am_gm_w,
                "eta_head": eta_head,
                "R_aniso": R_aniso,
                "hadamard_ratio": hadamard_ratio,
                "G_PCA_mean": float(G_PCA_mean),
                "G_norot_mean": float(G_norot_mean),
                "sigma_base": float(sigma_base_computed),
            },
        }

    predictions["diagnostics"] = {
        "R_aniso": g["R_aniso_mean"],
        "R_aniso_range": f"{g['R_aniso_min']:.2f} ~ {g['R_aniso_max']:.2f}",
        "eta2": g["eta2_mean"],
        "eta4": g["eta4_mean"],
        "AM_GM_w": g["AM_GM_w_mean"],
        "eta_head_fischer": eta_head,
        "G_PCA_mean": float(G_PCA_mean),
        "G_norot_mean": float(G_norot_mean),
        "n_heads_total": g["n_heads_total"],
        "method_selection": _select_method(g),
    }

    return predictions


def _select_method(g: Dict, eps: float = 0.05) -> str:
    """정리 6.6.5에 의한 방법 선택."""
    R = g["R_aniso_mean"]
    e2 = g["eta2_mean"]
    e4 = g["eta4_mean"]

    if R < 1.0 + eps:
        return "균일 양자화 (회전 불필요) — R_aniso ≈ 1"
    elif e2 > 1 - eps:
        return "b=2: PolarQuant/CommVQ — η(2) > 0.95"
    elif e4 > 1 - eps:
        return "b=4: IsoQuant — η(4) > 0.95"
    else:
        return "b=d: Pre-RoPE PCA + MK (fokvq_full) — η(4) < 0.95, 전역 상관 강함"


# ============================================================================
# 출력
# ============================================================================

def print_predictions(predictions: Dict):
    """예측 결과를 보기 좋게 출력한다."""

    model = predictions["model"]
    diag = predictions["diagnostics"]

    print(f"\n{'='*70}")
    print(f"예측 결과: {model}")
    print(f"{'='*70}")

    print(f"\n[진단 통계량]")
    print(f"  R_aniso (이방성):  {diag['R_aniso']:.2f} (범위: {diag['R_aniso_range']})")
    print(f"  η(2) (블록-2):    {diag['eta2']:.3f}")
    print(f"  η(4) (블록-4):    {diag['eta4']:.3f}")
    print(f"  AM/GM(w) (MK):    {diag['AM_GM_w']:.3f}")
    print(f"  η_head (Fischer): {diag['eta_head_fischer']:.3f}")
    print(f"  G_PCA (평균):      {diag['G_PCA_mean']:.6f}")
    print(f"  G_norot (평균):    {diag['G_norot_mean']:.6f}")
    print(f"  총 헤드 수:        {diag['n_heads_total']}")
    print(f"  → 방법 선택:       {diag['method_selection']}")

    for bits in sorted(predictions["bits"].keys()):
        bp = predictions["bits"][bits]
        tc = bp["theory_constants"]

        print(f"\n{'─'*70}")
        print(f"[{bits}-bit 예측]")
        print(f"  Lloyd-Max/Uniform = {tc['lm_ratio']:.3f} ({1-tc['lm_ratio']:.1%} MSE 감소)")
        print(f"  MK 추가 이득 (AM/GM_w) = {tc['am_gm_w']:.3f}")
        print(f"  Hadamard ratio = {tc['hadamard_ratio']:.3f}")
        print(f"  σ_base (실측) = {tc['sigma_base']:.6f}")
        print(f"  fokvq_full D_MSE 비율 = {tc['lm_ratio']:.3f} (LM only, MK 미포함)")
        print(f"  fokvq_full D_attn 비율 = {tc['lm_ratio']/tc['am_gm_w']:.3f} (LM + MK)")

        print(f"\n  D_MSE 상대 순위 (PPL용, Pre-RoPE PCA+WF+Uniform = 1.0):")
        for method, val in bp["mse_ranking"]:
            marker = " ★" if method == "fokvq_full" else ""
            print(f"    {method:<20} {val:>8.3f}x{marker}")

        d_attn_ranking = sorted(bp["d_attn_relative"].items(), key=lambda x: x[1])
        print(f"\n  D_attn 상대 순위 (NIAH/RULER용, Pre-RoPE PCA+WF+Uniform = 1.0):")
        for method, val in d_attn_ranking:
            marker = " ★" if method == "fokvq_full" else ""
            print(f"    {method:<20} {val:>8.3f}x{marker}")

        print(f"\n  NIAH 성공률 (gap=0.05, 어려운 위치):")
        if "hard_30pct" in bp["niah"]:
            niah = bp["niah"]["hard_30pct"]
            for method in ["fokvq_full", "KVTC", "PreRoPE_PCA_WF", "TurboQuant"]:
                if method in niah:
                    print(f"    {method:<20} {niah[method]:>8.1%}")

        print(f"\n  RULER-VT (5변수, gap=0.08):")
        if 5 in bp["ruler_vt"]:
            ruler = bp["ruler_vt"][5]
            for method in ["fokvq_full", "KVTC", "PreRoPE_PCA_WF", "TurboQuant"]:
                if method in ruler:
                    print(f"    {method:<20} {ruler[method]:>8.1%}")

    # 종합표 - D_MSE (PPL용)
    print(f"\n{'='*70}")
    print(f"종합 예측표: {model}")
    print(f"{'='*70}")

    bits_sorted = sorted(predictions["bits"].keys())
    methods_order = ["fokvq_full", "KVTC", "PreRoPE_PCA_WF", "TurboQuant", "No_rotation"]

    print(f"\n[D_MSE 상대값 — PPL 예측용]")
    print(f"{'방법':<20} ", end="")
    for bits in bits_sorted:
        print(f"{'%dbit' % bits:>10} ", end="")
    print(f"  {'순위':>6}")
    print("─" * 65)

    for method in methods_order:
        print(f"{method:<20} ", end="")
        for bits in bits_sorted:
            val = predictions["bits"][bits]["mse_relative"].get(method, float('nan'))
            print(f"{val:>10.3f} ", end="")
        rank = sorted(range(len(methods_order)), key=lambda i:
                      predictions["bits"][3]["mse_relative"].get(methods_order[i], 999))
        my_rank = rank.index(methods_order.index(method)) + 1
        print(f"  {my_rank:>6}")

    print(f"\n[D_attn 상대값 — NIAH/RULER 예측용]")
    print(f"{'방법':<20} ", end="")
    for bits in bits_sorted:
        print(f"{'%dbit' % bits:>10} ", end="")
    print(f"  {'순위':>6}")
    print("─" * 65)

    for method in methods_order:
        print(f"{method:<20} ", end="")
        for bits in bits_sorted:
            val = predictions["bits"][bits]["d_attn_relative"].get(method, float('nan'))
            print(f"{val:>10.3f} ", end="")
        rank = sorted(range(len(methods_order)), key=lambda i:
                      predictions["bits"][3]["d_attn_relative"].get(methods_order[i], 999))
        my_rank = rank.index(methods_order.index(method)) + 1
        print(f"  {my_rank:>6}")

    # KVTC 대비 이득 (D_MSE / D_attn 각각)
    print(f"\n[fokvq_full vs KVTC 이득]")
    for bits in bits_sorted:
        bp = predictions["bits"][bits]
        mse_gain = bp["mse_relative"]["KVTC"] / bp["mse_relative"]["fokvq_full"]
        attn_gain = bp["d_attn_relative"]["KVTC"] / bp["d_attn_relative"]["fokvq_full"]
        print(f"  {bits}-bit: D_MSE {mse_gain:.2f}x, D_attn {attn_gain:.2f}x 개선")

    # fokvq_full vs TurboQuant 이득
    print(f"\n[fokvq_full vs TurboQuant 이득]")
    for bits in bits_sorted:
        bp = predictions["bits"][bits]
        mse_gain = bp["mse_relative"]["TurboQuant"] / bp["mse_relative"]["fokvq_full"]
        attn_gain = bp["d_attn_relative"]["TurboQuant"] / bp["d_attn_relative"]["fokvq_full"]
        print(f"  {bits}-bit: D_MSE {mse_gain:.2f}x, D_attn {attn_gain:.2f}x 개선")

    # 벤치마크별 예측 승자
    print(f"\n[6개 벤치마크 예측 승자]")
    print(f"  1. PPL:      fokvq_full (MSE 최소 → PPL 최소)")
    print(f"  2. NIAH:     fokvq_full (σ_error 최소 → 성공률 최고)")
    print(f"  3. LITM:     fokvq_full (중간 위치 보호)")
    print(f"  4. RULER-VT: fokvq_full (σ_error 최소, AND 증폭)")
    print(f"  5. Qasper:   fokvq_full (MSE 순위 = Qasper 순위)")
    print(f"  6. MMLU:     동등 (양자화에 강건)")


# ============================================================================
# main
# ============================================================================

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 출력 디렉토리
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    tokenizer, model, model_info = load_model(args.model_name, args.device, args.cache_dir)

    # Pre-RoPE K, Q 수집
    print(f"\n[1/3] Pre-RoPE 키/쿼리 벡터 수집...")
    K_dict, Q_dict = collect_pre_rope_kq(
        model, tokenizer, args.device, args.n_calib_tokens, args.cache_dir
    )

    # 모델 메모리 해제
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  모델 메모리 해제 완료")

    # 공분산 + 진단 통계량 계산
    print(f"\n[2/3] 공분산 및 진단 통계량 계산...")
    stats = compute_covariances(K_dict, Q_dict, model_info)

    # 메모리 해제
    del K_dict, Q_dict
    gc.collect()

    # 6개 벤치마크 예측
    print(f"\n[3/3] 6개 벤치마크 예측 계산...")
    predictions = predict_all_benchmarks(stats, model_info, args.bits)

    # 출력
    print_predictions(predictions)

    # JSON 저장
    model_key = args.model_name.replace("/", "_")
    output_file = output_dir / f"prediction_{model_key}_{datetime.now():%Y%m%d_%H%M%S}.json"

    # numpy 타입 변환
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_data = {
        "model_info": model_info,
        "diagnostics": stats["global"],
        "predictions": predictions,
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
    }

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2, default=convert)

    print(f"\n결과 저장: {output_file}")
    print(f"\n완료.")


if __name__ == "__main__":
    main()
