# 실험 계획 v22: SOTA 비교 전략 + CWF 검증

**날짜**: 2026-04-08
**전략**: B+C+D (same-harness ablation 유지 + KVTC 검증 + MMLU downstream)
**논문 포지셔닝**: Understanding paper (이론 + 구조 분석), NOT SOTA system

---

## 현재 진행 중

| GPU | 실험 | 시작 | 상태 |
|:---:|------|:----:|:----:|
| 0 | Qwen MMLU PCA 2-bit | Apr 6 | 실행 중 (FP16=74.3%, NoRot=58.7% 완료) |
| 1 | Llama MMLU PCA 2-bit | Apr 7 | 실행 중 (FP16=65.6%, NoRot=40.2% 완료) |
| 2 | Llama CWF sweep | Apr 7 | Phase 1 sensitivity 진행 중 |
| 3 | 유휴 | — | — |

## 완료된 실험 (14개)

총 14개 실험 완료. 핵심 결과:
- 정리 검증: 624/624 MSE, 4/4 PPL (3-bit)
- PCA > 다른 회전: 10/10 KIVI, 7/9 GEAR, 6/9 Random
- KVTC 비교: +46.3% (헤드별 > 공유)
- PCA-Q 정렬: 0.6-2.5°, ρ=0.655
- Lloyd PPL 실패: 3축 통일 분석, 5가설 기각
- CWF: Mistral avg=3.5에서 5.73 (v3 WF 5.82 돌파)
- QW-WF: negative result (PCA-Q 정렬 때문)

## 남은 실험 (우선순위순)

### P0: MMLU 결과 수집 (진행 중)
```
의도: Downstream에서도 PCA > NoRot 확인
가설: PCA 2-bit MMLU > NoRot 2-bit MMLU
검증: GPU 0,1 완료 대기
리뷰어 방어: "PPL만으로 부족하다" 반박
```

### P0: Llama CWF (진행 중)
```
의도: CWF 3번째 모델 cross-verification
가설: CWF avg=3.5에서 v3 WF(7.16) 근접 또는 돌파
검증: GPU 2에서 실행 중
리뷰어 방어: "Mistral에서만 작동" 반박
```

### P1: Unified benchmark (5-seed error bars)
```
의도: 표 간 숫자 불일치 해소 + error bars 제공
가설: 1-2% PPL 차이는 calibration noise
검증: 단일 calibration으로 모든 표 재생성
스크립트: run_unified_benchmark.py (post_pca bug fix 필요)
예상: ~5시간 on 2 GPUs
리뷰어 방어: "숫자가 표마다 다르다", "error bars 없다"
```

### P1: KVTC 공식 코드 검증 (Option C)
```
의도: 기존 시스템에 우리 이론을 적용하여 개선 가능 입증
가설: KVTC 코드에서 공유 PCA → 헤드별 PCA로 교체하면 PPL 개선
검증: KVTC GitHub 코드 clone, PCA 부분만 교체, 동일 eval
리뷰어 방어: "실제 시스템에서도 작동하나?"
NOTE: KVTC GitHub (github.com/OnlyTerp/kvtc) 코드 확인 필요
```

### P2: 3-bit MMLU (GPU 0,1 완료 후)
```
의도: 3-bit에서도 downstream 확인
가설: PCA 3-bit MMLU > NoRot 3-bit MMLU
검증: 2-bit 완료 후 이어서 실행 (이미 스크립트에 queued)
```

### P2: 논문 영문 번역
```
의도: NeurIPS 제출
상태: 한국어 초안 25p 완성. 영문 번역 필요
```

## SOTA 비교 전략 (B+C+D)

### Option B: Published numbers 참조 (논문에 반영 완료)
- 기존 방법의 보고된 수치를 별도 단락에서 인용
- "다른 평가 설정"임을 명시
- 우리 숫자와 같은 표에 넣지 않음

### Option C: KVTC 검증 (P1)
- KVTC 코드에서 shared PCA → per-head PCA 교체
- 우리 이론의 예측을 독립 시스템에서 확인
- "이론이 기존 시스템을 개선할 수 있다"는 가장 강력한 증거

### Option D: MMLU (P0, 진행 중)
- PCA 2-bit/3-bit MMLU ≥ NoRot MMLU 확인
- "downstream에서도 degradation 없음" 방어

## 논문 현재 상태

- 25 pages, 빌드 성공, 에러 없음
- 6개 기여 (정리, 헤드별PCA, PCA-Q정렬, 3축간극, per-layer+5기각, sensitivity)
- QW-WF: negative result로 정직하게 재정의
- SOTA 비교: understanding paper 프레이밍 완료
- 남은 이슈: error bars (P1), MMLU (P0), KVTC 검증 (P1)
