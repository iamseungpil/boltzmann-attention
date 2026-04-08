# 실험 계획 v21: QW-WF 발견 반영 + 남은 실험

**날짜**: 2026-04-07
**상태**: QW-WF 3모델 검증 완료, MMLU 진행 중, b_crit 4모델 완료

---

## 완료된 실험

| 항목 | 결과 | 논문 반영 |
|------|------|:---------:|
| 정리 1: RoPE 상쇄 | 624/624 MSE, 3-bit 4/4 PPL | ✅ |
| Baseline 비교 (KIVI/GEAR/Random/PCA) | PCA>KIVI 10/10, PCA>GEAR 7/9 | ✅ |
| Pre vs Post-RoPE (4모델) | 3-bit 4/4, 2-bit 2/4 | ✅ |
| KVTC 비교 (Llama) | +46.3% | ✅ |
| **QW-WF (3모델)** | **Qwen +10.9%, Llama +32.9%, Mistral +10.1%** | **✅ 방금 반영** |
| **PCA-Q 정렬 (3모델)** | **0.6-2.5°** | **✅ 방금 반영** |
| **QW-PCA 실패 진단** | **H2/H4/H5 확인** | **✅ 방금 반영** |
| **b_crit 실측 (4모델)** | **median 9.8-11.0** | ✅ |
| Gaussianity check | κ₄~0.5 | ✅ |
| Per-head MSE (정리 방어) | 112/112 | ✅ |
| Lloyd-Max PPL 실패 | 3모델 catastrophic | ✅ |

## 현재 진행 중

| 항목 | GPU | 상태 |
|------|:---:|:----:|
| MMLU Qwen PCA 2-bit | 0 | 실행 중 |
| MMLU Llama PCA 2-bit | 1 | 실행 중 |
| Pre vs Post (Qwen/Mistral/14B 재확인) | 3 | 실행 중 |

중간 결과: Qwen FP16=74.3%, NoRot=58.7% | Llama FP16=65.6%, NoRot=40.2%

## 남은 실험 (우선순위순)

### R1: MMLU 결과 분석 + 논문 반영
```
의도: PCA가 downstream에서도 NoRot보다 우수한지 확인
가설: PCA MMLU > NoRot MMLU (2-bit, 3-bit)
상태: 실행 중, 수시간 내 완료 예상
```

### R2: QW-WF tops caiman 재현
```
의도: QW-WF의 로컬 결과가 다른 환경에서도 재현되는지 확인
상태: ✅ 완료 (Qwen-14B 포함, 결과 일치)
```

### R3: Coworker develop 결과 통합
```
의도: 13개 실험 결과 (sensitivity allocation, per-layer breakdown) 활용
주의: develop에서 paper/ 파일이 삭제됨 — cherry-pick으로 실험 결과만 가져올 것
```

### R4: 논문 최종 정리
```
- 영문 번역
- Figure 업데이트 (QW-WF 비교 차트)
- Appendix: QW-PCA 실패 진단 상세, b_crit 실측 테이블
```

## 논문 현재 상태

제목: "Pre-RoPE PCA는 KV 캐시 양자화에서 최적이다: 회전 기반 방법의 통일적 분석과 증명"
페이지: 20p (본문 ~12p + appendix)
기여 5가지:
1. RoPE 상쇄 정리 (분포 무관)
2. 헤드별 PCA > 공유 PCA (+46.3%)
3. Same-harness 비교 + 경계 분석
4. MSE-PPL 간극 설명 모델 (b_crit)
5. **QW-WF (10-33% PPL 개선)** ← 새로 추가
