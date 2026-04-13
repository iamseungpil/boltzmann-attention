# 실험 계획 v20: 통합 최종 계획

**날짜**: 2026-04-06
**상태**: v19 + coworker develop merge + MMLU 중간 결과 반영

## 핵심 원칙

이 계획은 다음의 구분을 전제로 한다.

1. **증명된 것**
   - RoPE 상쇄와 Pre-RoPE PCA의 Class C 내 MSE 기준 최적성
2. **설명 모델**
   - attention-weighted surrogate
   - MSE$\to$PPL 국소 전이
   - `b_crit` 위험도 진단
3. **실험 목표**
   - (a) 증명된 부분을 직접 검증
   - (b) 설명 모델이 실제로 어느 구간에서 유효한지 경계를 측정
   - (c) practical winner 여부는 별도의 downstream/long-context 실험으로 판정

---

## 현재 blocker / 정합성 이슈

1. **Downstream 하네스 drift**
   - 현재 `run_downstream.py`는 `fp16 / no_rot / turbo / turboquant_rand / pre_pca_uni / pre_pca_wf2 / fokvq_e2 / fokvq_e2_residual / kivi_residual`만 지원한다.
   - 다만 이 집합은 여전히 전체 v3 registry의 부분집합이며, 각 method의 provenance도 runner-local 구현 / shared v3 quantizer import / proxy control이 섞여 있다.
   - 따라서 parity는 ``이름 지원``만으로 닫히지 않는다. 공통 method(`fp16`, `no_rot`, `pre_pca_uni`, `pre_pca_wf2`, `fokvq_e2`, `fokvq_e2_residual` 및 대응 proxy)에 대해
     matched model/task/prompt/calibration/scoring 설정에서 v3 PPL 하네스와 downstream 하네스가 허용 오차 내 정합함을 확인해야 한다.
2. **Baseline 해석 제한**
   - v3 PPL 하네스의 `kivi_residual`, `turboquant_rand`는 공식 baseline 재현이 아니라 same-harness proxy control이다.
   - 논문과 보고서에서는 ``KIVI-style / TurboQuant-inspired proxy``로만 표기한다.
   - parity/provenance gate가 닫히기 전 pilot 결과는 proxy baseline ranking, cross-model generalization, paper headline claim,
     official reproduction claim의 근거로 사용할 수 없다.
3. **진단 지표 위상 정리**
   - `GQA mismatch`는 quantizer 품질 지표가 아니라 구조 설명용 진단이다.
   - acceptance rule에 넣지 않고 descriptive diagnostic으로만 사용한다.
4. **Theory-to-metric 표본 부족**
   - 현재 `D_attn`, top-k overlap, Hamiltonian 진단은 첫 배치/첫 헤드 기반의 샘플링 진단이다.
   - P3를 완료하려면 per-model/per-bit 다중 샘플 집계로 확장해야 한다.
5. **실행 중 MMLU 결과의 위상**
   - 현재 GPU 1, 2의 MMLU는 실험이 잘 돌고 있는지 확인하는 \emph{범위 제한 smoke+pilot}다.
   - baseline-complete 논문 근거로 쓰기 전 하네스 확장이 선행되어야 한다.
6. **Axis 2 확장 계획의 범위 통제**
   - `spherical / L1 / Fisher / lattice / Wasserstein` 전체를 메인 플랜에 합치면 현재 증거 수준을 넘는다.
   - Axis 2는 우선 `spherical` 단일 gate 실험과 `metric hierarchy` 서술 정제까지만 포함한다.
7. **Plan-misaligned artifact 격리 필요**
   - parity/provenance 강화 전에 생성된 downstream artifact 중 method semantics, dtype, prompt/scoring path가 불명확한 것은
     authoritative evidence에서 제외하고 `pilot/quarantine`으로 표기해야 한다.
   - 재사용 가능한 artifact는 provenance backfill이 가능할 때만 승격 후보로 둔다.
8. **Downstream env parity / provenance 누락**
   - downstream 결과가 `dtype`, `attn_implementation`, `seed`, `git_head`, calibration artifact identity를 남기지 않으면
     same-harness pilot도 환경 차이로 오독될 수 있다.
   - 특히 unified-plan method name이 runner-local 구현인지, shared v3 quantizer import인지, proxy control인지
     artifact만 보고 식별할 수 있어야 하므로 requested method와 implementation key/source를 함께 남겨야 한다.

---

## 현재 완료된 것

| 항목 | 결과 | 논문 반영 |
|------|------|:---------:|
| 정리 1: RoPE 상쇄 (Pre-RoPE PCA 최적) | 분포 무관, 624/624 MSE 확인 | ✅ |
| 명제 2: Attention-weighted 최적 회전 | 이론 유도 완료, 실증 강화 필요 | ⚠️ |
| 명제 3: Attention error bound | 이론 유도 완료, 실증 강화 필요 | ⚠️ |
| **명제 4: MSE→PPL 전이 체인** | **Coworker merge 완료, 국소 surrogate로 해석** | ⚠️ |
| **명제 5: MSE 순위→PPL 순위 보존** | **충분 조건 유도 완료, 경계 검증 필요** | ⚠️ |
| **명제 6: b_crit 임계 비트** | **위험도 진단식으로 사용, 다모델 검증 필요** | ⚠️ |
| Pre-RoPE vs Post-RoPE PPL (4모델) | 3-bit 4/4, 2-bit 2/4 | ✅ |
| Baseline 비교 (same-harness proxy) | PCA>KIVI-style 10/10, PCA>GEAR-style 7/9 | ✅ |
| KVTC 비교 (Llama) | +46.3% | ✅ |
| Per-head MSE (정리 방어) | 112/112 Pre<Post | ✅ |
| Gaussianity check | κ₄~0.5 | ✅ |
| WF floor ablation | floor=2 유망, 다모델/downstream 보강 필요 | ⚠️ |

## 현재 진행 중

| 항목 | 상태 | GPU |
|------|------|:---:|
| **MMLU Qwen PCA 2-bit** | 실행 중, scope-limited pilot | GPU 1 |
| **MMLU Llama PCA 2-bit** | 실행 중, scope-limited pilot | GPU 2 |

**중간 결과:**
- Qwen: FP16=74.3%, NoRot 2b=58.7%, PCA 2b=진행 중
- Llama: FP16=65.6%, NoRot 2b=**40.2%** (25.4%p 폭락!), PCA 2b=진행 중

## 남은 실험 (우선순위순)

### P0: 논문 서사 정합화
```
의도: develop에서 가져온 강한 이론 문구와 현재 실험 근거의 강도를 맞춘다
가설: "증명"과 "설명 모델"을 분리하면 논문 전체의 내부 일관성이 올라간다
검증:
  - abstract/introduction/theory/discussion에서 claim hierarchy 일치
  - b_crit, MSE→PPL, attention-weighted 회전을 universal law가 아니라 surrogate로 표기
  - WF는 메인 알고리즘이 아니라 검증된 개선축으로 기술
```

### P0.5: Downstream 하네스 정합화
```
의도: downstream 결과가 unified plan의 method vocabulary와 같은 실험을 의미하도록 만든다
가설:
  - run_downstream.py가 same-harness method 이름과 의미를 맞추면
    현재 MMLU pilot과 이후 downstream 결과의 해석 오류를 크게 줄일 수 있다
검증:
  - method alias 또는 explicit support를 통해 최소한 `turboquant_rand`, `fokvq_e2`, `fokvq_e2_residual`, `kivi_residual`
    중 지원 가능한 범위를 명시
  - unsupported method는 조용히 fallback하지 않고 명시적 오류로 실패
  - 결과 JSON에 `requested_method`, `canonical_method`, `implementation_key`, `implementation_source`, `claim_label`,
    `support_level`, `method_provenance`, `baseline_family`, `implementation_scope`,
    `official_reproduction=false/true`, `pilot_only=true/false`, `same_harness_only=true`,
    `overclaim_guard`, `architecture_support`, `calibration_source`를 저장
  - 결과 JSON에 `git_head`, `harness_name/version`, `task list`, `prompt/scoring path`, `dataset revision`,
    `seed`, `calibration artifact identity`, `model/tokenizer revision`, `device`, `dtype`,
    `attn_implementation`, `pilot_scope`를 저장
  - 공통 method/config에 대해 v3 PPL 하네스와 matched-config parity smoke를 수행하고, 해석 가능한 범위와 비정합 범위를 명시
  - tiny Llama smoke + unsupported GPT-2 clean failure + one supported quantized smoke를 모두 통과
```

### P1: MMLU PCA 결과 대기 (진행 중)
```
의도: 현재 downstream 하네스가 최소한 PCA vs NoRot 방향성은 재현하는지 확인
가설: 동일 하네스/동일 task/동일 calibration/동일 scoring path에서 PCA가 NoRot보다 우수하다
검증:
  - 현재 실행 결과 대기
  - same task list, same prompt/scoring path, same seed policy, same calibration source를 유지한 delta 비교로 기록
  - 표준오차 또는 반복 실행이 없으면 ``within-harness directional pilot``로만 기록
  - parity/provenance gate 전에는 baseline-complete 비교 근거로 승격하지 않음
```

### P2: 3-bit MMLU (MMLU 후 즉시 queue)
```
의도: scope-limited downstream 하네스에서 3-bit 안정 구간의 방향성을 먼저 확인
가설: 3-bit에서는 Pre-RoPE PCA가 NoRot보다 안정적으로 우수하고, 2-bit보다 격차가 작다
검증:
  - 동일 스크립트, --bits 3
  - same task list, same prompt/scoring path, same seed policy, same calibration source를 유지
  - 결과는 하네스 정합성 확보 전까지 ``within-harness directional pilot``로만 사용
GPU: 1, 2 (MMLU 2-bit 완료 후)
```

### P3: Theory-to-metric 검증 실험
```
의도: attention surrogate와 b_crit가 실제로 어떤 범위까지 유효한지 측정
가설:
  - 3/4-bit에서는 D_attn, attention-logit distortion, PPL 순위가 비교적 잘 정렬된다
  - 2-bit에서는 정렬이 무너지고 모델별 편차가 커진다
검증:
  - per-model/per-bit로 D_attn vs PPL 상관 측정
  - top-k attention rank overlap 측정
  - measured κ(Σ_Q), κ(Σ_K)와 역전 구간의 정성적 대응 확인
  - 첫 배치 단일 샘플이 아니라 다중 layer/head/window 집계로 확장
GPU: 1 또는 2
```

### P4: KVTC 비교 확장 (Qwen, Mistral)
```
의도: +46.3% (Llama) 결과가 특정 모델 우연이 아닌지 확인
가설: 공유 PCA 대비 헤드별 PCA 이득은 2-bit에서 가장 크고 3/4-bit에서 축소된다
검증:
  - 모델: Qwen, Mistral
  - 비트: 2/3/4-bit
  - same-harness full-K PPL에서 공유 PCA vs 헤드별 PCA를 비교
  - acceptance: 2-bit에서 두 모델 모두 head-wise PCA가 공유 PCA보다 개선, 3/4-bit에서는 이득 축소 또는 동률
```

## Pilot 승격 규칙

pilot 결과는 다음 조건을 모두 만족할 때에만 ``reportable downstream evidence``로 승격한다.

1. `P0.5`의 provenance 필드가 모두 artifact에 저장되어 있을 것
2. 공통 method/config parity smoke가 통과했을 것
3. 적어도 하나의 반복 실행 또는 uncertainty estimate가 있을 것
4. 결과 해석 범위가 ``official reproduction``인지 ``same-harness proxy pilot``인지 명시되어 있을 것

## Artifact 격리 규칙

다음 조건 중 하나라도 만족하면 기존 결과물은 `quarantine`으로 분류한다.

1. method provenance가 저장되어 있지 않음
2. dtype / seed / prompt-scoring path / calibration source가 누락됨
3. 현재 `P0.5`에서 정의한 matched-config parity gate 이전 결과임
4. proxy baseline이 official reproduction처럼 읽힐 여지가 있음

### P5: WF(floor=2) 다모델/downstream 검증
```
의도: WF가 메인 방법이 아니라 ``증명된 회전 위의 실용 확장``으로서 의미가 있는지 확인
가설:
  - 2-bit에서만 일관된 이득
  - 3-bit에서는 미세 이득 또는 동률
  - 4-bit에서는 사실상 차이 없음
검증:
  - Qwen/Llama/Mistral MMLU 2/3-bit에 PCA+Uni vs PCA+WF(floor=2) 추가
  - 표준편차와 과목별 편차까지 확인
```

### P6: Axis 2 spherical gate experiment
```
의도: axis 2 실패가 단순한 "PCA 관점의 실패"가 아니라 Euclidean/MSE형 양자화 목표의 geometry mismatch인지 확인한다
가설:
  - 간단한 spherical/polar 계열 quantizer는 axis-2-sensitive 설정에서 Euclidean scalar quantizer보다
    attention-structure 진단 또는 PPL에서 더 나은 신호를 줄 수 있다
  - 이 항목은 새로운 핵심 기여 주장이 아니라 exploratory stress test다
검증:
  - 선행 조건: P3의 다중 샘플 진단 집계와 하네스 정합성 확보
  - 1차 gate: Qwen 2-bit + Mistral 2-bit에서 same-harness full-K PPL, key MSE, attention-logit distortion 비교
  - 비교 대상: PCA+Uniform, PCA+WF(floor=2), 기존 polar/mag-phase 계열
  - 합격 기준: matched budget에서 하나 이상의 practical metric이 일관되게 개선되고 collapse가 없을 것
  - 불합격 시: 메인 플랜 확장 없이 appendix/exploratory negative result로만 기록
GPU: 1 또는 2 (P1-P5 이후)
```

### P7: Anisotropy-aware metric hierarchy statement 정제
```
의도: axis 2 음성 결과를 과장 없이 설명할 수 있도록 "무엇이 최적이고 무엇이 아닌가"를 metric hierarchy로 정리한다
가설:
  - L2 / attention-weighted quadratic / KL 2차 근사 / worst-case retrieval형 metric을 분리하면
    Lloyd-Max 실패와 uniform 생존을 더 명확히 설명할 수 있다
검증:
  - 03_theory와 appendix에서 proved result / surrogate / exploratory metric을 명시적으로 구분
  - 현재 evidence ledger와 충돌하는 강한 optimality 문구 제거
  - spherical gate 결과가 없더라도 독립적으로 읽히는 statement로 유지
```

### P8: Query-weighted PCA / MK-Lloyd-Max 및 기타 axis 2 방법은 탐색 항목으로 격하
```
의도: theory surrogate의 constructive extension이 실제로 필요한지 확인
가설:
  - QW-PCA, MK-Lloyd-Max, L1 Lloyd, Fisher, lattice, Wasserstein은 주 논문 핵심이 아니라 부록/탐색 결과에 가깝다
검증:
  - 기존 핵심 실험(P1-P7) 완료 후에만 실행
  - PCA baseline을 넘지 못하면 appendix negative result로 정리
  - spherical gate가 통과하지 않으면 axis 2 method family 확장 자체를 보류
```

## 논문 남은 수정 사항

1. abstract/introduction/theory/method에서 `증명`, `surrogate`, `constructive extension`, `proxy baseline`을 명시적으로 분리
2. MMLU 결과 → 논문에 downstream 테이블 추가
3. b_crit 수치 검증 (κ(Σ_Q), κ(Σ_K) 실측)과 함께 `risk diagnostic`로 표기
4. Coworker 이론의 증명 appendix 추가하되, empirical law처럼 서술하지 않기
5. Figure 업데이트:
   - MMLU
   - D_attn vs PPL
   - measured `b_crit` vs observed reversal regime
6. 영문 번역 (한국어 → 영어)
