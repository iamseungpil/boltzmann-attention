# 실험 계획 v20: 통합 최종 계획

**날짜**: 2026-04-06
**상태**: v19 + coworker develop merge + MMLU 중간 결과 반영

---

## 현재 완료된 것

| 항목 | 결과 | 논문 반영 |
|------|------|:---------:|
| 정리 1: RoPE 상쇄 (Pre-RoPE PCA 최적) | 분포 무관, 624/624 MSE 확인 | ✅ |
| 명제 2: Attention-weighted 최적 회전 | 이론 유도 완료 | ✅ |
| 명제 3: Attention error bound | 이론 유도 완료 | ✅ |
| **명제 4: MSE→PPL 전이 체인** | **Coworker, 6단계+3차 나머지** | ✅ (merge) |
| **명제 5: MSE 순위→PPL 순위 보존** | **Coworker, 충분 조건** | ✅ (merge) |
| **명제 6: b_crit 임계 비트** | **Coworker, 이론 4.7 vs 실측 4.6** | ✅ (merge) |
| Pre-RoPE vs Post-RoPE PPL (4모델) | 3-bit 4/4, 2-bit 2/4 | ✅ |
| Baseline 비교 (KIVI/GEAR/Random/PCA) | PCA>KIVI 10/10, PCA>GEAR 7/9 | ✅ |
| KVTC 비교 (Llama) | +46.3% | ✅ |
| Per-head MSE (정리 방어) | 112/112 Pre<Post | ✅ |
| Gaussianity check | κ₄~0.5 | ✅ |
| WF floor ablation | floor=2 최적 | ✅ |

## 현재 진행 중

| 항목 | 상태 | GPU |
|------|------|:---:|
| **MMLU Qwen PCA 2-bit** | 실행 중 (~12시간) | GPU 1 |
| **MMLU Llama PCA 2-bit** | 실행 중 (~12시간) | GPU 2 |

**중간 결과:**
- Qwen: FP16=74.3%, NoRot 2b=58.7%, PCA 2b=진행 중
- Llama: FP16=65.6%, NoRot 2b=**40.2%** (25.4%p 폭락!), PCA 2b=진행 중

## 남은 실험 (우선순위순)

### P1: MMLU PCA 결과 대기 (진행 중)
```
의도: PCA가 NoRot보다 MMLU에서 우수한지 확인
가설: Qwen PCA > 58.7%, Llama PCA > 40.2%
검증: 현재 실행 결과 대기
```

### P2: 3-bit MMLU (MMLU 후 즉시 queue)
```
의도: 이론이 가장 강한 3-bit에서 downstream 검증
가설: 3-bit에서 PCA MMLU > NoRot MMLU > ... (PPL 순서 일치)
검증: 동일 스크립트, --bits 3
GPU: 1, 2 (MMLU 2-bit 완료 후)
```

### P3: Coworker 진단 실험 E1 (Mahalanobis Lloyd-Max)
```
의도: b_crit 이론의 예측을 실험적으로 검증
      Mahalanobis Lloyd-Max가 standard Lloyd-Max보다 PPL 개선하는지
가설: MK PPL < standard Lloyd-Max PPL (특히 b < b_crit에서)
검증: exp_axis2_axis3_diagnosis.py --experiment E1
GPU: 1 또는 2 (MMLU 후)
```

### P4: Query-weighted PCA 실험
```
의도: 명제 2의 예측 검증 — QW-PCA가 표준 PCA보다 PPL 개선하는지
가설: QW-PCA가 2-bit Mistral/Llama에서 표준 PCA 역전 해소
검증: run_theory_experiments.py --exp t4
GPU: 1 또는 2
```

### P5: KVTC 비교 확장 (Qwen, Mistral)
```
의도: +46.3% (Llama) 결과를 다모델로 확장
검증: 공유 PCA vs 헤드별 PCA PPL
```

### P6: 3-bit MMLU + Baseline 비교
```
의도: 전체 비교 테이블 완성
```

## 논문 남은 수정 사항

1. MMLU 결과 → 논문에 downstream 테이블 추가
2. b_crit 수치 검증 (κ(Σ_Q), κ(Σ_K) 실측)
3. Coworker 이론의 증명 appendix 추가
4. 영문 번역 (한국어 → 영어)
5. Figure 업데이트 (MMLU + b_crit 시각화)
