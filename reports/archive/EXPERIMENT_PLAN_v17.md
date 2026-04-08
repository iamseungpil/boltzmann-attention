# 실험 계획 v17: NeurIPS 최종 보강 (Downstream + Scale + SpinQuant)

**날짜**: 2026-04-05
**기반**: Codex R4 overclaim 리뷰 + NeurIPS gap 분석
**목표**: CRITICAL gap 해소 (downstream task, 14B 모델)

---

## Exp V17-1: Downstream Task 평가 (MMLU + GSM8K)

```
의도 (Intent):
  PPL 이득이 실제 downstream 정확도로 전환되는지 확인.
  WikiText-2 PPL만으로는 NeurIPS 리뷰어가 불충분하다고 판단할 것.

가설 (Hypothesis):
  H1: 2-bit에서 Pre-RoPE PCA의 MMLU 정확도 > TurboQuant MMLU 정확도
  H2: 정확도 순서가 PPL 순서와 일치 (PCA+WF > PCA+Uni > TurboQuant > NoRot)
  H3: 3-4bit에서는 차이가 미미 (PPL과 동일 패턴)

검증 방법 (Verification):
  모델: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B
  방법: FP16, NoRot+Uni, TurboQuant, PCA+Uni, PCA+WF(f=2)
  비트: 2, 3
  Task: MMLU (5-shot), GSM8K (8-shot CoT)
  
  구현: lm-evaluation-harness + k_proj hook 양자화
    - model.forward를 감싸서 k_proj 출력에 hook 적용
    - FP16 baseline은 hook 없이 실행
  
  성공 기준: 2-bit에서 PCA+WF > TurboQuant (MMLU accuracy 1%p 이상)
  
  시간: 모델당 ~3시간/방법 × 5방법 × 3모델 = ~45시간
  우선순위: Llama-3.1-8B 먼저 (KVTC 비교 모델)
```

## Exp V17-2: 14B 모델 스케일링 (Qwen2.5-14B)

```
의도 (Intent):
  결과가 7-8B를 넘어 14B 규모에서도 유지되는지 확인.
  NeurIPS 리뷰어는 규모 일반화를 요구할 것.

가설 (Hypothesis):
  H1: Qwen2.5-14B에서도 Pre-RoPE PCA+WF가 TurboQuant를 2-bit PPL에서 상회
  H2: 이득 비율이 7B와 유사하거나 더 큼

검증 방법 (Verification):
  모델: Qwen2.5-14B (GQA, A100 48GB에 적합)
  방법: FP16, TurboQuant, PCA+Uni, PCA+WF(f=2)
  비트: 2, 3, 4
  Task: WikiText-2 PPL + MMLU (5-shot)
  
  메모리: FP16 weights ~29GB + activations ~4GB = ~33GB (48GB A100 적합)
  시간: PPL ~1시간/방법 × 4방법 = ~4시간, MMLU ~3시간/방법 × 4방법 = ~12시간
```

## Exp V17-3: SpinQuant 비교 (문헌 인용 + 간이 구현)

```
의도 (Intent):
  가장 강한 Class C baseline (학습된 회전)과의 비교.
  SpinQuant이 없으면 리뷰어가 "가장 강한 baseline을 빠뜨렸다"고 지적.

가설 (Hypothesis):
  H1: Pre-RoPE PCA가 SpinQuant PPL의 95% 이내 달성
  H2: PCA는 캘리브레이션 불필요 (SpinQuant은 학습 필요) → 비용 대비 우위

검증 방법 (Verification):
  Option A (1시간): SpinQuant 논문의 published numbers 인용
    - Table에 "SpinQuant (reported)" 행 추가, 차이점 명시
  Option B (2-3일): Cayley SGD 간이 구현
    - skew-symmetric A를 최적화하여 quantization MSE 최소화
    - 캘리브레이션 128 sequences, 100 epochs
  추천: Option A + "SpinQuant은 학습 비용이 필요한 반면 PCA는 zero-cost" 프레이밍
```

---

## 실행 순서 (최적화)

```
Day 1:
  - [AM] 구현: run_downstream.py (lm-eval + k_proj hook)
  - [PM] Smoke test: Llama-8B FP16 MMLU 1 subtask
  
Day 2:
  - [FULL] Llama-8B: FP16/TurboQuant/PCA+WF × MMLU 전체 (병렬 불가, 순차)
  - [OVERNIGHT] Qwen-7B MMLU

Day 3:
  - [AM] Mistral-7B MMLU
  - [PM] Qwen-14B PPL + MMLU 시작

Day 4:
  - GSM8K 평가 (선택적, MMLU가 더 중요)
  - SpinQuant 문헌 수치 정리
  - 결과 분석 + 논문 업데이트
```
