# 실험 계획 v19: 이론 심화 + 실험 검증

**날짜**: 2026-04-06
**목표**: KVTC 사후 정당화를 넘어, 이론이 새로운 방법을 도출하는 논문으로 전환

## ⚠️ 정정 노트 (2026-04-06 후반)

T3, T4 초기 구현에 다음 결함이 발견되어 수정되었음:

1. **weighted_lloyd_col은 no-op이었다.** 스칼라 weight로 col을 곱하고 Lloyd-Max 후 나누는 방식은 standard Lloyd-Max와 수학적으로 동일 (스케일 불변). 진짜 attention-aware는 차원별 비트 할당(water-filling) 필요. → `attn_quant` 메서드를 차원별 bit 재분배(`weighted_bit_allocation`)로 재구현.

2. **T4 qw_pca 회전이 이론과 불일치했다.** 명제 6.19.6의 V_qw는 k̃ = Σ_Q^{1/2}·k 공간의 고유벡터이므로 k에 직접 적용할 수 없음. → Σ_Q^{1/2} 변환 → V_qw 회전 → 양자화 → V_qw^T → Σ_Q^{-1/2} 단계로 구현.

3. **GQA Σ_Q 계산 오류**. `Cov(mean_g q^g) ≠ (1/G)·Σ_g Cov(q^g)`. 전자는 그룹 내 분산을 잃음. → 모든 G 헤드를 stack하여 공분산 계산.

4. **κ_Q, κ_K, ‖δa‖∞ 측정 부재**. 이론(b_crit, R₃ bound)이 요구하는 양들이 측정되지 않아 검증 불가능했음. → calibrate()에서 κ 측정 출력, eval_ppl()에서 ‖δk‖∞/√d proxy 측정.

또한 LIE_GROUP_UNIFICATION.md 정리 6.19.16-18의 다음 결함이 정정됨:
- R₃ bound: `(1/3)·‖δa‖∞³/min_i p_i` → `(8/3)·‖δa‖∞³` (대수 오류)
- b_crit 닫힌 공식 (`(1/2)log₂(κ_Q·κ_K) + O(1)`)와 모델별 표 철회 ((κ-1)(κ-1) 미증명, O(1) 차원 인자 은폐)
- 6.19.2.x V3 검증의 "이론 4.7 vs 실험 4.6 일치" 정량 주장 철회 (사후 적합 + 단위 불일치)
- 03_theory.tex의 명제 3-5와 검증 절도 동일하게 정정

## ⚠️ 추가 정정 노트 (2026-04-06 후반 라운드 2)

리뷰어 재지적으로 다음 추가 결함이 발견되어 수정됨:

5. **‖δa‖∞ proxy가 무의미했음**. 이전 정정에서 `||δk||∞/√d`를 기록했으나, 진짜 ‖δa‖∞은 `max_i |q·δk_i|/√d`로 q와 결합 필요. → q_proj forward_pre_hook으로 layer별 q 캡처, k_proj hook에서 q와 결합하여 진짜 ‖δa‖∞ 측정. (단, pre-RoPE q이며 post-RoPE는 향후 작업).

6. **FP16 baseline 부재**. PPL 비율이 아닌 ΔCE = log(PPL/PPL_fp) 비교를 위해 FP16 baseline 측정 추가. 모든 실험 결과에 `delta_ce` 필드 추가.

7. **qw_pca round-trip 단위 테스트 추가**. bits=16 (효과적으로 양자화 없음)에서 qw_pca 경로의 PPL이 FP16과 |rel_err| < 0.001 이내인지 검증. 실패 시 회전 구현 버그 경고.

8. **min_bits=2 기본값**. T3 attn_quant에서 1-bit 할당 가능성을 차단 (실험 오염 방지).

9. **T6 추가**. attn_quant + qw_pca 결합 (`qw_pca_attn` 메서드) — 이론상 최대 이득 예측 영역.

10. **LIE_GROUP의 "정확한 유도" 표현 정정**. R₃ bound는 leading-order bound (‖δa‖∞ ≪ 1 가정)이지 엄밀 Lagrange bound가 아님을 명시.

11. **03_theory.tex contribution 재포지셔닝**. "MSE-PPL 역전의 최초 정량 설명" 주장 제거 → "KV 양자화의 Softmax 섭동 프레임워크" + AWQ/KVQuant/Atom 선행 연구 명시 + 차이점 (full covariance vs per-channel scaling, RoPE 분포 무관 상쇄, 3축 직교 분해)을 도입부에 명시.

## 잔존 작업 (정량 contribution을 위해 필수)

1. T3/T4/T6 실험 실행 → 측정된 ‖δa‖∞ 값으로 정리 6.19.17 충분 조건 직접 검증
2. AWQ/KVQuant baseline 재현 → KV 양자화 영역에서의 정량적 비교
3. 2-bit Uniform = FP16 anomaly 재현 및 진단
4. Post-RoPE q 캡처 (현재는 pre-RoPE)
5. log-PPL (ΔCE) 공간에서의 일관된 보고 (PPL 비율 사용 금지)

---

## T1: Softmax 섭동 분석 (MSE→PPL 다리)

```
의도: MSE 최적성이 PPL 최적성으로 전이되는 조건을 이론적으로 도출.
      현재 "MSE≠PPL"은 경험적 관찰. 이론적 설명 없음.

이론:
  attention_error ≈ J · Q · δK^T / √d
  여기서 J = diag(p) - pp^T (softmax Jacobian)
  
  E[||attention_error||²] ∝ tr(Σ_Q · Σ_err) / d
  
  → attention 오차는 Σ_Q-가중 MSE에 비례
  → attention sink (p_i ≈ 1) 근처에서 J의 eigenvalue가 작아져 오차 감소
  → 그러나 sink 토큰의 key 오차는 전체 분포를 shift하여 catastrophic

가설:
  H1: 1차 섭동이 MSE-PPL 상관(R²>0.85)을 설명
  H2: 2차 항이 2-bit PPL 역전의 원인

검증: 기존 데이터로 수치 검증 (추가 실험 불필요)
```

## T2: RoPE 주파수별 오차 분석

```
의도: RoPE 주파수 θ_i = 10000^{-2i/d}에 따라 양자화 오차의
      영향이 다름을 이론적으로 보이고, 주파수 인지 비트 배분을 도출.

이론:
  PCA 성분 j의 양자화 오차가 attention에 미치는 영향:
  δa_j(pos) = q_j · δk_j · cos(θ_j · pos + φ)
  
  위치 평균: E_pos[δa_j²] = (1/2) · q_j² · δk_j²  (고주파)
  위치 상관: Cov(δa_j(p1), δa_j(p2)) ∝ cos(θ_j·(p1-p2))
  
  → 고주파(큰 θ): 오차가 위치 간 uncorrelated → 평균화
  → 저주파(작은 θ): 오차가 위치 간 correlated → 누적

가설:
  H1: 저주파 PCA 성분에 더 많은 비트를 주면 PPL 개선
  H2: 현재 WF(분산 비례)와 주파수 인지 배분은 다름

검증: PCA 성분별 주파수-분산 상관 분석 + 주파수 인지 WF 실험
```

## T3: Attention-Aware 최적 양자화기

```
의도: Lloyd-Max PPL 실패를 해결하는 새로운 양자화기를
      이론에서 도출. "이론 → 새 방법 → 실험 개선"의 완결.

이론:
  표준 Lloyd-Max: min E[(x - Q(x))²]
  Attention-aware: min E[w² · (x - Q(x))²]
  
  여기서 w_j = √(Σ_Q)_{jj} (j번째 차원의 query 가중치)
  
  Weighted Lloyd-Max의 codebook:
  - w가 큰 차원: centroid를 더 촘촘하게 배치
  - w가 작은 차원: 덜 촘촘해도 됨

가설:
  H1: attention-aware quantizer PPL < uniform PPL
  H2: attention-aware quantizer PPL < Gaussian Lloyd-Max PPL
  H3: 2-bit에서 이득이 가장 큼 (오차 영향이 크므로)

검증: 구현 → 3모델 × {uniform, Lloyd-Max, attention-aware} × {2,3}bit PPL
```

## T4: GQA 커플링 분석

```
의도: GQA에서 G개 쿼리 헤드의 유효 쿼리 공분산을 도출하고,
      G에 따라 PCA vs query-weighted PCA의 차이가 어떻게 변하는지 분석.

이론:
  Σ_Q_eff = (1/G) Σ_{g=1}^{G} Σ_Q^{(g)}
  G→∞: Σ_Q_eff → isotropic → 표준 PCA 최적
  G=1 (MHA): Σ_Q_eff = Σ_Q → query-weighted PCA 최적

가설:
  H1: G가 큰 모델(Qwen G=7)에서 PCA와 QW-PCA 차이 미미
  H2: G가 작은 모델(Llama G=4)에서 QW-PCA 이득 더 큼

검증: 4모델의 G값과 PCA 이득의 상관 분석
```

## T5: Value 캐시 양자화 이론

```
의도: Key만 다루는 현재 논문을 K+V 통합 이론으로 확장.

이론:
  output = Σ_t α_t · v_t
  δ_output = Σ_t α_t · δv_t
  
  V 양자화 최적 회전: V의 공분산 PCA (RoPE 없으므로 단순)
  V 양자화 최적 비트 배분: attention weight 가중 분산 비례
  
  → 높은 α를 받는 토큰(sink)의 v는 더 정밀하게 양자화
  → Key와 다른 최적 전략

가설:
  H1: V에 PCA를 적용하면 추가 PPL 개선
  H2: K+V 동시 최적화가 개별 최적화보다 이득

검증: K-only vs K+V 양자화 PPL 비교
```

---

## 실행 순서 (병렬화)

```
즉시 (이론 유도, GPU 불필요):
  T1: Softmax 섭동 수식 유도 + 수치 검증
  T2: 주파수별 오차 분석 수식 유도
  T4: GQA 커플링 분석 (기존 데이터)

GPU 필요 (tops caiman):
  T3: Attention-aware quantizer 구현 + 실험
  T5: V 캐시 양자화 실험
  + 진행 중: MMLU, 3-bit MMLU queue
```
