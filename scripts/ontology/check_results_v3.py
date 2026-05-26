"""check_results_v3.py — Phase 0 v3 결과 확인 + PCLI Intervention Map 자동 생성.

PCLI (Probing-Calibrated Layerwise Intervention):
  Phase 0 probing 결과 (τ, L_peak, curve_pattern) 세 값으로
  각 관계의 최적 개입 방법, 레이어, 계수를 자동 결정.
"""
import json, os, math

REPORT = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/phase0_probing"
path   = f"{REPORT}/probing_results_v3_telecom.json"
log    = f"{REPORT}/probe_v3_run.log"
MAP_OUT = f"{REPORT}/intervention_map.json"

N_LAYERS = 29          # Qwen2.5-7B: L00–L28
EARLY_PEAK_CUTOFF = 5  # best_layer <= 5 → early_peak
MID_LATE_CUTOFF   = 9  # best_layer >= 10 → mid_late
FLAT_STD_THRESH   = 0.030  # std < 0.03 → flat curve
TAU_SKIP          = 0.70   # probing acc 미달 → skip
TAU_STRONG        = 0.85   # 높은 신뢰도 기준
BASE_ALPHA        = 0.30   # 기준 steering 계수


# ──────────────────────────────────────────────
# PCLI 핵심 함수
# ──────────────────────────────────────────────

def classify_curve_pattern(layer_data, best_layer):
    """레이어 곡선 모양을 분류: early_peak / flat / mid_late"""
    accs = [d["acc_mean"] for d in layer_data]
    n    = len(accs)
    mean = sum(accs) / n
    std  = math.sqrt(sum((a - mean) ** 2 for a in accs) / n)

    if std < FLAT_STD_THRESH:
        return "flat"
    if best_layer <= EARLY_PEAK_CUTOFF:
        return "early_peak"
    if best_layer >= MID_LATE_CUTOFF:
        return "mid_late"
    return "mid_late"   # 6–9 구간도 mid_late로 분류


def calibrate_alpha(tau, best_layer):
    """
    α = BASE_ALPHA × tau_factor / settling
    tau_factor : τ=0.70 → 1.0, τ=1.00 → 0 (강한 인코딩 = 작은 α)
    settling   : (N_LAYERS-1-L_peak)/(N_LAYERS-1) (남은 레이어 많을수록 안정)
    """
    tau_factor = max(0.10, (1.0 - tau) / (1.0 - TAU_SKIP))
    settling   = max(0.10, (N_LAYERS - 1 - best_layer) / (N_LAYERS - 1))
    return round(BASE_ALPHA * tau_factor / settling, 3)


def select_method(tau, pattern, geometry):
    """
    반환: (method_code, notes)
    method_code:
      "skip"      — τ < TAU_SKIP, 개입 효과 불충분
      "A8"        — KV Cache Steering (flat 또는 moderate τ mid_late)
      "A6_peak"   — Per-head Rotation @L_peak (early_peak + directional)
      "T1_qonly"  — Q-only Additive @L_peak (그 외 개입 가능 케이스)
    """
    if tau < TAU_SKIP:
        return "skip", f"τ={tau:.3f} < {TAU_SKIP} — 선형 분리 불충분, 개입 skip"

    if pattern == "flat":
        return "A8", "flat curve → KV Cache Steering (증폭 없음, α 불필요)"

    if pattern == "early_peak":
        if geometry == "directional":
            return "A6_peak", "early_peak + directional → A6 Per-head Rotation @L_peak"
        else:
            return "T1_qonly", "early_peak + symmetric/categorical → Q-only T1 @L_peak"

    # mid_late
    if geometry == "directional" and tau >= TAU_STRONG:
        return "T1_qonly", "mid_late + directional + τ≥0.85 → Q-only T1 @L_peak"
    if geometry == "directional":
        return "A8", "mid_late + directional + moderate τ → A8 더 안정적"
    # symmetric / categorical mid_late
    if tau >= TAU_STRONG:
        return "T1_qonly", "mid_late + τ≥0.85 → Q-only T1 @L_peak"
    return "A8", "mid_late + moderate τ → A8 (안정성 우선)"


def build_intervention_map(summary):
    """
    summary dict → PCLI intervention map dict.
    각 관계 항목:
      pattern, method, layer, alpha, tau, geometry, notes
    """
    imap = {}
    for rel, s in summary.items():
        tau      = s["best_acc"]
        best_l   = s["best_layer"]
        geometry = s["geometry"]
        ld       = s["layer_data"]

        pattern          = classify_curve_pattern(ld, best_l)
        method, notes    = select_method(tau, pattern, geometry)
        alpha            = None if method in ("skip", "A8") else calibrate_alpha(tau, best_l)

        imap[rel] = {
            "geometry"  : geometry,
            "tau"       : round(tau, 4),
            "best_layer": best_l,
            "best_auc"  : round(s.get("best_auc", 0.0), 4),
            "pattern"   : pattern,
            "method"    : method,
            "alpha"     : alpha,
            "go_nogo"   : s["go_nogo_pass"],
            "notes"     : notes,
        }
    return imap


# ──────────────────────────────────────────────
# 결과 파일 로드
# ──────────────────────────────────────────────

if not os.path.exists(path):
    print("RESULT FILE NOT FOUND")
    if os.path.exists(log):
        with open(log) as f:
            lines = f.readlines()
        print("Last 20 lines of log:")
        for l in lines[-20:]:
            print(l, end="")
    raise SystemExit(1)

with open(path) as f:
    d = json.load(f)

print(f"Version : {d['metadata'].get('version')}")
print(f"Model   : {d['metadata']['model']}")
print(f"OVERALL : {'PASS' if d['metadata']['overall_pass'] else 'FAIL'}")
print()


# ──────────────────────────────────────────────
# §1. 기본 probing 결과 테이블 (기존 형식 유지)
# ──────────────────────────────────────────────

import numpy as np

GEO_ORDER = ["directional", "symmetric", "conditional", "categorical"]
a6_preds, t1_preds = [], []

print(f"  {'관계':22s} {'기하학':12s} {'예측':>4s}  L   acc   auc   GO  {'accs (L0→Lpeak→L28)':>25s}")
print("  " + "─" * 90)

for geo in GEO_ORDER:
    for pt, s in sorted(d["summary"].items(), key=lambda x: x[1]["geometry"]):
        if s["geometry"] != geo:
            continue
        flag = "✓" if s["go_nogo_pass"] else "✗"
        pred = s["predicted_method"]
        ld   = s["layer_data"]

        acc0   = ld[0]["acc_mean"]
        accpk  = s["best_acc"]
        accend = ld[-1]["acc_mean"]
        trend  = f"L00={acc0:.3f} → pk={accpk:.3f} → L28={accend:.3f}"

        print(f"  {pt:22s} {s['geometry']:12s} {pred:>4s}  "
              f"L{s['best_layer']:02d} {s['best_acc']:.3f} {s['best_auc']:.3f}  {flag}  {trend}")

        if pred == "A6":
            a6_preds.append(s["best_acc"])
        else:
            t1_preds.append(s["best_acc"])

print()
print("═" * 70)
print("A7 예측 검증 (이론: 방향성→A6, 대칭/범주형→T1)")
if a6_preds:
    print(f"  A6 후보 관계 평균 best_acc: {np.mean(a6_preds):.3f}  (N={len(a6_preds)})")
if t1_preds:
    print(f"  T1 후보 관계 평균 best_acc: {np.mean(t1_preds):.3f}  (N={len(t1_preds)})")
if a6_preds and t1_preds:
    diff = np.mean(a6_preds) - np.mean(t1_preds)
    print(f"  Δ(A6-T1) probing acc     : {diff:+.3f}")
    if diff > 0.02:
        print("  → 방향성 관계가 더 선형 분리 가능 → A6 유리")
    elif diff < -0.02:
        print("  → 대칭/범주형 관계가 더 선형 분리 가능 → T1 전반 우위")
    else:
        print("  → 두 그룹 유사 → A7 실제 pass^1 실험으로 최종 판단")
print("═" * 70)


# ──────────────────────────────────────────────
# §2. PCLI Intervention Map 생성 및 출력
# ──────────────────────────────────────────────

print()
print("═" * 90)
print("PCLI — Probing-Calibrated Layerwise Intervention Map")
print("  각 관계의 최적 개입 방법·레이어·계수를 probing 결과에서 자동 결정")
print("═" * 90)

imap = build_intervention_map(d["summary"])

# 통계 집계
method_counts = {}
for v in imap.values():
    method_counts[v["method"]] = method_counts.get(v["method"], 0) + 1

skip_n   = method_counts.get("skip", 0)
a8_n     = method_counts.get("A8", 0)
a6_n     = method_counts.get("A6_peak", 0)
t1_n     = method_counts.get("T1_qonly", 0)
total_n  = len(imap)
active_n = total_n - skip_n

print(f"\n  전체 {total_n}종  →  활성 {active_n}종 / skip {skip_n}종")
print(f"  A6_peak={a6_n}  T1_qonly={t1_n}  A8={a8_n}  skip={skip_n}")

# 방법별 테이블 출력
METHOD_ORDER = ["A6_peak", "T1_qonly", "A8", "skip"]
PATTERN_SYM  = {"early_peak": "↗", "flat": "─", "mid_late": "↗↗"}

print()
print(f"  {'관계':24s} {'geo':12s} {'τ':>5s} {'L':>3s} {'pat':>5s} {'method':>10s} {'α':>6s}  notes")
print("  " + "─" * 95)

for meth in METHOD_ORDER:
    items = [(k, v) for k, v in imap.items() if v["method"] == meth]
    if not items:
        continue
    for rel, v in sorted(items, key=lambda x: x[1]["best_layer"]):
        pat_sym = PATTERN_SYM.get(v["pattern"], "?")
        alpha_s = f"{v['alpha']:.3f}" if v["alpha"] is not None else "  —  "
        go_s    = "✓" if v["go_nogo"] else "✗"
        print(f"  {rel:24s} {v['geometry']:12s} {v['tau']:>5.3f} "
              f"L{v['best_layer']:02d} {pat_sym:>5s} {v['method']:>10s} "
              f"{alpha_s:>6s}  {go_s} {v['notes'][:45]}")


# ──────────────────────────────────────────────
# §3. 개입 레이어 충돌 분석
# ──────────────────────────────────────────────

print()
print("═" * 70)
print("개입 레이어 충돌 분석 (같은 레이어에 복수 관계 개입 시 간섭 위험)")
layer_to_rels = {}
for rel, v in imap.items():
    if v["method"] in ("A6_peak", "T1_qonly"):
        L = v["best_layer"]
        layer_to_rels.setdefault(L, []).append((rel, v["method"]))

conflict = False
for L in sorted(layer_to_rels):
    rels = layer_to_rels[L]
    flag = "⚠ 간섭 위험" if len(rels) > 1 else "  안전"
    if len(rels) > 1:
        conflict = True
    print(f"  L{L:02d}: {flag}  {[r for r,_ in rels]}")

if not conflict:
    print("  → 모든 활성 관계가 서로 다른 레이어 사용. 간섭 없음.")
print("═" * 70)


# ──────────────────────────────────────────────
# §4. Intervention Map JSON 저장
# ──────────────────────────────────────────────

out = {
    "metadata": {
        "version"     : d["metadata"].get("version"),
        "model"       : d["metadata"]["model"],
        "pcli_version": "1.0",
        "base_alpha"  : BASE_ALPHA,
        "tau_skip"    : TAU_SKIP,
        "tau_strong"  : TAU_STRONG,
        "n_layers"    : N_LAYERS,
        "summary": {
            "total"   : total_n,
            "active"  : active_n,
            "skip"    : skip_n,
            "A6_peak" : a6_n,
            "T1_qonly": t1_n,
            "A8"      : a8_n,
        }
    },
    "intervention_map": imap
}

with open(MAP_OUT, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"\n  Intervention Map 저장 완료: {MAP_OUT}")
print(f"  → Phase 2 구현 시 이 파일을 직접 로드하여 개입 스케줄 구성")
