"""probe_ontology_v3.py

Phase 0 v3 + A7 실험: T1 vs A6 실증 비교를 위한 probing.

v2 대비 변경:
  - 12종 관계 모두 probing
  - 관계별 기하학 분류(directional / symmetric / categorical) 기록
  - A7 예측 검증: 이론 예측 방법(A6/T1)과 실측 probing 성능 비교
  - 출력: 관계별 best_layer, best_acc, geometry, predicted_method 포함 JSON

Usage:
  python3 scripts/ontology/probe_ontology_v3.py \\
    --pairs reports/facet_rft_2026/phase0_probing/contrast_pairs_v3.json \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --device cuda:0 \\
    --out-dir reports/facet_rft_2026/phase0_probing/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_hidden_states(
    texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int = 8,
    max_length: int = 300,
) -> np.ndarray:
    model.eval()
    all_hidden: List[np.ndarray] = []

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start: batch_start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, return_dict=True)

        hs = out.hidden_states          # tuple: (n_layers+1, batch, seq, d)
        mask = enc["attention_mask"]
        last_idx = mask.sum(dim=1) - 1  # (batch,)

        batch_hidden = []
        for layer_hs in hs:
            idx = last_idx.view(-1, 1, 1).expand(-1, 1, layer_hs.size(-1))
            last_tok = layer_hs.gather(1, idx).squeeze(1)  # (batch, d)
            batch_hidden.append(last_tok.cpu().float().numpy())

        all_hidden.append(np.stack(batch_hidden, axis=1))  # (batch, n_layers+1, d)

        if (batch_start // batch_size) % 5 == 0:
            print(f"  [{batch_start + len(batch)}/{len(texts)}]", flush=True)

    return np.concatenate(all_hidden, axis=0)  # (N, n_layers+1, d)


def probe_layer(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, aucs = [], []
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[va])
        proba = clf.predict_proba(X[va])[:, 1]
        accs.append((preds == y[va]).mean())
        try:
            aucs.append(roc_auc_score(y[va], proba))
        except ValueError:
            aucs.append(0.5)
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
    }


def plot_curves(
    layer_results: Dict[str, List[Dict]],
    out_path: str,
    threshold: float = 0.70,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # 기하학별 색상
    color_map = {
        "directional":  "#1f77b4",  # 파랑 계열 (A6 후보)
        "symmetric":    "#ff7f0e",  # 주황 계열 (T1 후보)
        "conditional":  "#d62728",
        "categorical":  "#9467bd",
    }
    geo_map = {
        "precedes": "directional", "requires": "directional",
        "enables": "directional", "parameter_feeds": "directional",
        "validates": "directional", "retry_after_fail": "directional",
        "compensates": "directional",
        "mutex": "symmetric", "parallel_safe": "symmetric",
        "conditional_on": "conditional",
        "workflow_role": "categorical", "precondition_state": "categorical",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 상단: 방향성 (A6 후보)
    ax = axes[0]
    ax.set_title("Directional Relations — A6 Rotation Candidate", fontsize=12)
    for pt, data in layer_results.items():
        if geo_map.get(pt) != "directional":
            continue
        accs = [d["acc_mean"] for d in data]
        ax.plot(range(len(accs)), accs, label=pt, linewidth=1.8)
    ax.axhline(threshold, linestyle="--", color="red", alpha=0.7, label=f"Threshold {threshold:.0%}")
    ax.axhline(0.5, linestyle=":", color="gray", alpha=0.4, label="Chance")
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("acc (5-fold CV)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 하단: 대칭/범주형 (T1 후보)
    ax = axes[1]
    ax.set_title("Symmetric / Categorical Relations — T1 Additive Candidate", fontsize=12)
    for pt, data in layer_results.items():
        if geo_map.get(pt) == "directional":
            continue
        accs = [d["acc_mean"] for d in data]
        ax.plot(range(len(accs)), accs, label=pt, linewidth=1.8)
    ax.axhline(threshold, linestyle="--", color="red", alpha=0.7)
    ax.axhline(0.5, linestyle=":", color="gray", alpha=0.4)
    ax.set_ylim(0.4, 1.05); ax.set_xlabel("Layer index"); ax.set_ylabel("acc (5-fold CV)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot] → {out_path}")
    plt.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=(
        "/home/woori/workspace_common/boltzmann-attention-pi/"
        "reports/facet_rft_2026/phase0_probing/contrast_pairs_v3.json"
    ))
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=300)
    p.add_argument("--out-dir", default=(
        "/home/woori/workspace_common/boltzmann-attention-pi/"
        "reports/facet_rft_2026/phase0_probing/"
    ))
    p.add_argument("--threshold", type=float, default=0.70)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pairs: {args.pairs}")
    with open(args.pairs) as f:
        data = json.load(f)
    pairs = data["pairs"]
    meta = data["metadata"]
    print(f"  {meta['n_total']} total | v={meta['version']}")
    for pt, cnt in meta.get("by_type", {}).items():
        print(f"    {pt}: {cnt}")

    # 관계별 분류
    rel_types = list(meta.get("by_type", {}).keys())
    by_type: Dict = {}
    for pt in rel_types:
        sub = [pp for pp in pairs if pp["pair_type"] == pt]
        if not sub:
            continue
        by_type[pt] = {
            "texts":    [pp["text"] for pp in sub],
            "labels":   np.array([pp["label"] for pp in sub], dtype=int),
            "geometry": sub[0].get("geometry", "unknown"),
            "predicted_method": sub[0].get("predicted_method", "?"),
        }
        n, npos = len(sub), sum(pp["label"] for pp in sub)
        print(f"  {pt}: {n} ({npos} pos / {n-npos} neg)  "
              f"geometry={by_type[pt]['geometry']}  pred={by_type[pt]['predicted_method']}")

    print(f"\nLoading model: {args.model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded in {time.time()-t0:.1f}s — {n_layers} layers")

    layer_results: Dict[str, List[Dict]] = {}
    for pt, info in by_type.items():
        texts, labels = info["texts"], info["labels"]
        print(f"\n[{pt.upper()}] ({info['geometry']}, pred={info['predicted_method']}) "
              f"Extracting {len(texts)} samples ...")
        t0 = time.time()
        hidden = extract_hidden_states(
            texts, model, tokenizer, args.device, args.batch_size, args.max_length
        )
        print(f"  Done {time.time()-t0:.1f}s  shape={hidden.shape}")

        t0 = time.time()
        res = [probe_layer(hidden[:, li, :], labels) for li in range(n_layers + 1)]
        print(f"  Probing done {time.time()-t0:.1f}s")

        best = int(np.argmax([r["acc_mean"] for r in res]))
        flag = "✓ PASS" if res[best]["acc_mean"] >= args.threshold else "✗ FAIL"
        print(f"  Peak: L{best:02d} acc={res[best]['acc_mean']:.3f}  {flag}")
        layer_results[pt] = res

    # ── A7 예측 검증 요약 ──────────────────────────────────────────────────────
    GEO_PREDICTION = {
        "directional": "A6",
        "symmetric":   "T1",
        "conditional": "T1",
        "categorical": "T1",
    }
    summary = {}
    overall_pass = False
    for pt, res in layer_results.items():
        best = int(np.argmax([r["acc_mean"] for r in res]))
        best_acc = res[best]["acc_mean"]
        passed = best_acc >= args.threshold
        if passed:
            overall_pass = True
        info = by_type[pt]
        summary[pt] = {
            "best_layer": best,
            "best_acc": best_acc,
            "best_auc": res[best]["auc_mean"],
            "go_nogo_pass": passed,
            "geometry": info["geometry"],
            "predicted_method": info["predicted_method"],
            "layer_data": res,
        }

    out_json = out_dir / "probing_results_v3_telecom.json"
    with open(out_json, "w") as f:
        json.dump({
            "metadata": {
                "model": args.model, "domain": "telecom",
                "n_layers": n_layers, "threshold": args.threshold,
                "overall_pass": overall_pass, "version": "v3",
            },
            "summary": summary,
        }, f, indent=2)
    print(f"\nResults → {out_json}")

    out_png = str(out_dir / "probing_curve_v3_telecom.png")
    plot_curves(layer_results, out_png, args.threshold)

    # ── 최종 출력 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 0 v3 — A7 예측 검증 요약")
    print("=" * 70)
    print(f"  {'관계':22s} {'기하학':12s} {'예측':>4s}  L  acc   auc   GO")
    print("  " + "-" * 60)

    n_correct_pred = 0
    a6_accs, t1_accs = [], []

    for pt in sorted(summary.keys()):
        s = summary[pt]
        geo = s["geometry"]
        pred = s["predicted_method"]
        flag = "✓" if s["go_nogo_pass"] else "✗"
        print(f"  {pt:22s} {geo:12s} {pred:>4s}  "
              f"L{s['best_layer']:02d} {s['best_acc']:.3f} {s['best_auc']:.3f}  {flag}")
        if pred == "A6":
            a6_accs.append(s["best_acc"])
        else:
            t1_accs.append(s["best_acc"])

    print("  " + "-" * 60)
    if a6_accs and t1_accs:
        a6_mean = np.mean(a6_accs)
        t1_mean = np.mean(t1_accs)
        print(f"\n  A6 후보 관계 평균 acc : {a6_mean:.3f} (N={len(a6_accs)})")
        print(f"  T1 후보 관계 평균 acc : {t1_mean:.3f} (N={len(t1_accs)})")
        print(f"\n  → 다음 단계 A7 실험에서 A6/T1 적용 시 이 acc 차이가 pass^1 개선으로 이어지는지 검증")

    verdict = "PASS — Phase 1 진행" if overall_pass else "FAIL — 데이터/관계 재검토"
    print(f"\n→ OVERALL: {verdict}")
    print("=" * 70)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
