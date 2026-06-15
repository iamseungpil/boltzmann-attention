#!/usr/bin/env python
"""Olver 정의적 검증 (M-Olver 정의적·2026-06-15) — O(d)연속군 + cross-domain probe.

============================ ★사전등록 (PRE-REGISTRATION) ============================
이 docstring + git 커밋 타임스탬프 = 측정 *전* 예측 고정 증명. 결과로 수정 금지(falsifiable).
2차 Olver 부정→3차 전수분석이 high-f 퇴화 오염으로 정정. 정의적 확정엔 3결함 동시해소 필요:
  (1) 이산-relabel degeneracy → O(d) 연속군(norm-보존·퇴화 원리상 없음)
  (2) post-hoc f-range → 사전등록
  (3) 전제-스크린 → cross-domain probe 전이테스트(§5.14 실험1)

Part A — O(d) 연속군: surface-token 임베딩을 s-차원 무작위부분공간서 직교회전. s∈{0,2,4,8,16,32}.
  ★예측 A: inv_dim이 s↑에 *전 s-range·전 layer* 단조↓ (이산 f≥0.8 반등 없이). corr(s,inv)<−0.7 전 layer.
  반증 A: 어느 layer corr≥−0.3 or 비단조 → inv-측 부정 확정.

Part B — cross-domain invariant-probe: A=HF B=MM native-FC. label=n_nodes(1/2/3+).
  inv rep=orbit-mean μ(K surface-aug 평균)·raw rep=h(무aug). logreg 전이 A→B.
  ★예측 B: inv-probe A→B acc > raw-probe A→B acc (불변부=전이부). (inv−raw)>0.10. 둘다>chance.
  반증 B: raw_A→B ≥ inv_A→B → 「추상=불변」 기각.

★통과기준(고정): A 양성(전 layer corr<−0.7) AND B 양성(inv−raw>0.10) → Olver 정의적 확정.
  하나라도 음성 → 전제-스크린으로만. 단 1회·결과로 기준수정 금지.
=====================================================================================

Usage: olver_definitive.py --part A|B|both --adapter <dir> --n 48 --k 32 --layers 10,18,28
"""
import argparse, json, os, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RND = np.random.RandomState(42)
MODEL_DEFAULT = "/home/woori/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"


# ---------- 데이터: native-FC TaskBench (도메인별·n_nodes label) ----------
def load_domain(path, n):
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        nn = d.get("_meta", {}).get("n_nodes", 0)
        # 프롬프트 텍스트 = system+user+tools 요약 + 첫 assistant action
        msgs = d.get("messages", [])
        usr = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        tools = [t["function"]["name"] for t in (d.get("tools") or [])][:30]
        call = None
        for m in msgs:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                fn = m["tool_calls"][0].get("function", m["tool_calls"][0])
                call = (fn.get("name"), fn.get("arguments"))
                break
        if not (usr and tools and call):
            continue
        label = 0 if nn <= 1 else (1 if nn == 2 else 2)  # 1/2/3+
        text = "You are a tool-using assistant.\nAvailable tools: %s\nUser: %s\nAction: %s(%s)" % (
            ", ".join(tools), usr[:200], call[0], str(call[1])[:120])
        # surface 문자열(회전 대상) = tool명 + call명 + 값
        surf = list(set(tools + [call[0]] + re.findall(r"[A-Za-z0-9_.#@]{4,}", str(call[1]))))
        rows.append({"text": text, "label": label, "surf": surf})
        if len(rows) >= n:
            break
    return rows


# ---------- 표현 추출 (옵션: surface-token 임베딩 O(d) 회전) ----------
def surface_token_mask(tok, text, surf, device):
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
              truncation=True, max_length=512)
    offs = enc.pop("offset_mapping")[0].tolist()
    spans = []
    low = text.lower()
    for s in surf:
        st = 0
        sl = s.lower()
        while True:
            i = low.find(sl, st)
            if i < 0:
                break
            spans.append((i, i + len(s)))
            st = i + 1
    mask = torch.zeros(len(offs), dtype=torch.bool)
    for ti, (a, b) in enumerate(offs):
        if a == b:
            continue
        for (sa, sb) in spans:
            if a < sb and b > sa:
                mask[ti] = True
                break
    return {k: v.to(device) for k, v in enc.items()}, mask.to(device)


def rot_in_subspace(d, s, rng):
    """d차원서 무작위 s-부분공간 직교회전 행렬 R (norm-보존). (U, R_s) 반환."""
    if s <= 0:
        return None
    U = np.linalg.qr(rng.randn(d, s))[0]            # [d,s] orthonormal
    A = rng.randn(s, s); Q = np.linalg.qr(A)[0]      # random O(s)
    return torch.tensor(U, dtype=torch.float32), torch.tensor(Q, dtype=torch.float32)


@torch.no_grad()
def reps_rotated(model, emb_layer, tok, rows, layers, s, K, device):
    """각 입력 i에 K개 O(d)-회전 augmentation → layer별 표현 [n,K,d]."""
    d = emb_layer.weight.shape[1]
    out = {L: [] for L in layers}
    idx = []
    for i, r in enumerate(rows):
        enc, mask = surface_token_mask(tok, r["text"], r["surf"], device)
        base_emb = emb_layer(enc["input_ids"])  # [1,seq,d]
        for _ in range(K):
            emb = base_emb.clone()
            if s > 0 and mask.any():
                UR = rot_in_subspace(d, s, RND)
                U, Q = UR[0].to(device).to(emb.dtype), UR[1].to(device).to(emb.dtype)
                sub = base_emb[0][mask]                  # [m,d]
                coord = sub @ U                          # [m,s]
                rotated = sub + (coord @ (Q - torch.eye(s, device=device, dtype=emb.dtype))) @ U.T
                emb[0][mask] = rotated
            hs = model(inputs_embeds=emb, attention_mask=enc["attention_mask"],
                       output_hidden_states=True).hidden_states
            last = enc["attention_mask"].sum(1) - 1
            for L in layers:
                out[L].append(hs[L][0, last[0]].float().cpu().numpy())
            idx.append(i)
    return {L: np.array(v) for L, v in out.items()}, np.array(idx)


def eff_dim(X):
    X = X - X.mean(0, keepdims=True)
    C = X.T @ X / max(1, len(X))
    ev = np.linalg.eigvalsh(C); ev = ev[ev > 1e-10]
    return 0.0 if ev.size == 0 else float((ev.sum() ** 2) / (ev ** 2).sum())


def part_A(model, emb_layer, tok, rows, layers, K, device):
    s_list = [0, 2, 4, 8, 16, 32]
    print("\n=== Part A: O(d) 연속군 (s sweep, K=%d) ===" % K, flush=True)
    res = {L: {"s": s_list, "inv": [], "var": []} for L in layers}
    for s in s_list:
        R, idx = reps_rotated(model, emb_layer, tok, rows, layers, s, K, device)
        for L in layers:
            X = R[L]
            mus = np.stack([X[idx == i].mean(0) for i in range(len(rows))])
            resid = np.concatenate([X[idx == i] - X[idx == i].mean(0) for i in range(len(rows))], 0)
            iv, vv = eff_dim(mus), (eff_dim(resid) if s > 0 else 0.0)
            res[L]["inv"].append(iv); res[L]["var"].append(vv)
            print("  [s=%d L%d] inv=%.2f var=%.2f" % (s, L, iv, vv), flush=True)
    print("\n--- ★예측 A 판정 (corr(s,inv)<−0.7 전 layer·단조↓) ---")
    ok = True
    for L in layers:
        c = float(np.corrcoef(s_list, res[L]["inv"])[0, 1])
        mono = all(res[L]["inv"][i] >= res[L]["inv"][i + 1] - 1e-6 for i in range(len(s_list) - 1))
        passed = c < -0.7
        ok = ok and passed
        print("  L%d: inv=%s corr=%.2f %s %s" % (L, [round(x, 2) for x in res[L]["inv"]], c,
              "✓<−0.7" if passed else "✗", "단조↓" if mono else "비단조"))
    print("PART_A=%s" % ("PASS" if ok else "FAIL"))
    return res


def part_B(model, emb_layer, tok, dom_a, dom_b, layers, K, device):
    from sklearn.linear_model import LogisticRegression
    print("\n=== Part B: cross-domain invariant-probe 전이 (A=HF B=MM·label=n_nodes) ===", flush=True)
    def feats(rows):
        Rinv, Rraw, idx = reps_rotated(model, emb_layer, tok, rows, layers, 8, K, device)
        # inv = orbit-mean·raw = s=0 단일 forward
        Rraw0, idx0 = reps_rotated(model, emb_layer, tok, rows, layers, 0, 1, device)
        inv = {L: np.stack([Rinv[L][idx == i].mean(0) for i in range(len(rows))]) for L in layers}
        raw = {L: Rraw0[L] for L in layers}
        y = np.array([r["label"] for r in rows])
        return inv, raw, y
    iA, rA, yA = feats(dom_a)
    iB, rB, yB = feats(dom_b)
    print("\n--- ★예측 B 판정 (inv_A→B > raw_A→B·차>0.10) ---")
    ok = True
    for L in layers:
        def acc(Xtr, ytr, Xte, yte):
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, ytr)
            return clf.score(Xte, yte)
        inv_ab = acc(iA[L], yA, iB[L], yB)
        raw_ab = acc(rA[L], yA, rB[L], yB)
        chance = max(np.bincount(yB)) / len(yB)
        diff = inv_ab - raw_ab
        passed = diff > 0.10 and inv_ab > chance
        ok = ok and passed
        print("  L%d: inv_A→B=%.3f raw_A→B=%.3f diff=%.3f chance=%.3f %s" %
              (L, inv_ab, raw_ab, diff, chance, "✓" if passed else "✗"))
    print("PART_B=%s" % ("PASS" if ok else "FAIL"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="both", choices=["A", "B", "both"])
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dom_a", default="/home/woori/scratch/fc_build/tbnfc_huggingface.jsonl")
    ap.add_argument("--dom_b", default="/home/woori/scratch/fc_build/tbnfc_multimedia.jsonl")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--layers", default="10,18,28")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        print("[adapter]", a.adapter, flush=True)
    model.eval()
    emb_layer = model.get_input_embeddings()
    device = "cuda:0"
    rows_a = load_domain(a.dom_a, a.n)
    print("[data] A=%d" % len(rows_a), flush=True)
    if a.part in ("A", "both"):
        part_A(model, emb_layer, tok, rows_a, layers, a.k, device)
    if a.part in ("B", "both"):
        rows_b = load_domain(a.dom_b, a.n)
        print("[data] B=%d" % len(rows_b), flush=True)
        part_B(model, emb_layer, tok, rows_a, rows_b, layers, a.k, device)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
