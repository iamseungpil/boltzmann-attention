# 실험 계획 v27: Query-Dynamic Risk Paging + Tiny Recursive Innovation Cache

## 0. 결정 요약

이번 라운드에서 살아남긴 두 축은 다음이다.

1. **QDRP (Query-Dynamic Risk Paging)**
   - 저비트 base KV + 보조 residual/shadow를 저장해 두고, decode 시점에 **query 조건부 위험도**가 높은 page만 refinement.
   - 핵심은 `proxy score`가 아니라 **margin + quantization uncertainty**를 함께 쓰는 위험도 목적함수다.

2. **TRIC (Tiny Recursive Innovation Cache)**
   - 레이어 간 KV를 단순 공유하지 않고, **shared tiny recursive predictor**로 다음 레이어 KV를 예측한 뒤 **innovation residual만 저장/양자화**.
   - 핵심은 cross-layer reuse 자체가 아니라 **predictive coding / conditional rate-distortion** 프레이밍이다.

버린 축:
- Query-conditional PCA dimension selection:
  - attention MSE 이점은 있었지만 저장량이 줄지 않고 RoPE mismatch가 남는다.
- Tiny latent-state cache:
  - predictor + decoder + sparse correction까지 한 번에 가면 이론도 구현도 과하다.

---

## 1. 선행연구 기준점

### Query-dynamic 쪽

겹치는 기준선:
- H2O / SnapKV / Quest / RocketKV:
  - query 또는 proxy attention으로 토큰/페이지를 남기거나 선택
- MixKVQ:
  - query-aware mixed precision, 그러나 주로 **static allocation**
- TurboQuant:
  - base + residual storage 자체의 강한 baseline
- Expected Attention:
  - attention weight를 예측해 미래 중요도를 추정

이 축에서 살아남으려면:
- “query를 쓴다”로는 부족
- “top-k score selection”으로도 부족
- **quantization uncertainty-aware retrieval/refinement objective**가 핵심이어야 함

### Cross-layer 쪽

겹치는 기준선:
- MiniCache / Cross-Layer Attention:
  - depth redundancy 활용, layer sharing/merging
- xKV / AQUA-KV:
  - cross-layer prediction / residual coding

이 축에서 살아남으려면:
- “cross-layer redundancy를 쓴다”로는 부족
- per-layer adapter나 단순 interpolation이면 새롭지 않음
- **shared recursive predictor + innovation coding**으로 분리해야 함

---

## 2. 방법 1: QDRP

## 2.1 의도

긴 문맥에서 모든 토큰을 균일하게 더 정밀하게 읽는 대신, **현재 query에서 실제로 랭킹 뒤집힘 위험이 큰 page만** refinement하여 retrieval을 회복한다.

## 2.2 이론 프레이밍

양자화된 점수:

`ŝ_t = q^T k̂_t / sqrt(d),   s_t = q^T k_t / sqrt(d),   ε_t = k_t - k̂_t`

query가 주어졌을 때 score noise 분산:

`Var(s_t - ŝ_t | q) = q^T Σ_ε,t q / d`

현재 quantized winner를 `b`라 하면 challenger token `t`의 flip risk:

`P_flip(t,b | q, k̂) ≈ Φ(-(ŝ_b - ŝ_t) / sqrt(Var_t + Var_b))`

page `p`의 위험도는 token-level risk의 합/상계로 둔다:

`R_page(p) = Σ_{t in p} P_flip(t,b | q, k̂)`

budget `B_page` 하에서 top-risk page를 refinement하는 greedy는 union-bound 기준 최적.

단, budget이 2 이상이면 pure risk만으로는 현재 high-score winner page를 놓칠 수 있다.
따라서 실제 selector는 다음 두 경우를 분리한다.

- `B_page = 1`:
  - pure risk paging
- `B_page >= 2`:
  - **hybrid selector**
  - 1장은 score-max page 보존, 나머지는 risk page

핵심은 이것이 **decoder side information q가 있는 conditional refinement**라는 점이다.
목표는 KV MSE 최소화가 아니라 **argmax / top-k mismatch risk 감소**다.

## 2.3 기존 방법과의 구분

- H2O / SnapKV / Quest / RocketKV:
  - 높은 score page를 고른다. 우리는 **높은 score가 아니라 높은 mismatch risk**를 고른다.
- MixKVQ:
  - bit allocation은 저장 시점 중심. 우리는 **decode-time page activation**.
- TurboQuant:
  - residual을 모든 토큰에 균일 적용. 우리는 **stored refinement를 selective activation**.

## 2.4 핵심 가설

- H1:
  - page-level risk score가 raw score / raw margin보다 실제 miss page를 더 잘 찾는다.
- H2:
  - low-budget (`B_page=1`)에서는 `risk-page > score-page`가 성립해야 한다.
- H3:
  - multi-page budget에서는 `hybrid(score + risk)`가 pure score / pure risk보다 낫거나 최소 동률이어야 한다.
- H4:
  - dense PPL에서는 개선이 작거나 거의 없고, retrieval 쪽에서만 이득이 커야 한다.

## 2.5 Kill Criteria

- K1:
  - low-budget에서 oracle risk paging이 raw-score paging을 이기지 못하면 즉시 중단
- K2:
  - proxy risk가 oracle과 큰 간극을 보이면 실전 경로 보류
- K3:
  - hybrid selector까지 넣어도 retrieval gain이 없고 PPL도 그대로면 novelty가 약함

## 2.6 검증 순서

### Stage A: Risk metric calibration
- metric:
  - AUROC, Recall@budget, miss-page recall
- 비교:
  - raw score, margin, risk

### Stage B: Oracle page refinement
- 동일한 refinement budget에서:
  - score-page vs margin-page vs risk-page vs hybrid vs oracle
- metric:
  - true winner recovery, top-k overlap, NIAH success

### Stage C: Proxy implementation
- 2-bit base score에서 page risk 추정
- NIAH / retrieval-depth curve

### Stage D: Bounded dense-task sanity
- WikiText-2 PPL parity
- latency / extra pass overhead

---

## 3. 방법 2: TRIC

## 3.1 의도

레이어마다 KV를 독립 저장하는 대신, depth 방향 redundancy를 **shared tiny recursive predictor**로 설명하고, 설명되지 않는 innovation만 저장한다.

## 3.2 이론 프레이밍

토큰/헤드 단위 layer progression을 approximate Markov chain으로 둔다:

`K_{l+1} = f_phi(K_l, h_l) + ξ_l`

여기서:
- `f_phi`:
  - 모든 layer에서 공유되는 tiny recursive predictor
- `ξ_l`:
  - innovation residual

저장해야 할 정보의 핵심은 `K_{l+1}` 전체가 아니라:

`H(K_{l+1} | K_l, h_l)`

즉 conditional innovation entropy다.

예측이 좋을수록 residual covariance가 줄어들고, 동일 비트에서 residual quantization distortion도 줄어든다.
이건 predictive coding / conditional rate-distortion / innovation coding 프레임과 직접 연결된다.

## 3.3 기존 방법과의 구분

- MiniCache / CLA:
  - layer merging / sharing
- xKV / AQUA-KV:
  - cross-layer prediction은 겹치지만, 우리는 **per-layer module bank가 아니라 shared recursive predictor + innovation diagnostics**를 전면에 둔다.

핵심 차별점:
- predictor를 layer마다 따로 두지 않음
- 먼저 **depth process가 정말 low-innovation인지**를 통계적으로 검증
- predictor uncertainty를 residual bit budget과 연결

## 3.4 핵심 가설

- H1:
  - simple copy/interpolation보다 shared linear/recursive predictor가 residual energy를 유의하게 줄인다.
- H2:
  - residual covariance가 whitening되거나 저랭크화되어 innovation coding이 유리해진다.
- H3:
  - predictor confidence로 residual bit allocation을 하면 uniform residual coding보다 낫다.

## 3.5 Kill Criteria

- K1:
  - copy-last-layer 또는 shared linear predictor를 tiny recursive predictor가 못 이기면 중단
- K2:
  - residual energy는 줄어도 total bits + predictor overhead에서 손해면 중단
- K3:
  - retrieval/PPL proxy가 baseline과 비슷하면 논문 축으로 채택하지 않음

## 3.6 검증 순서

### Stage A: Depth redundancy diagnosis
- metric:
  - per-layer conditional R^2
  - residual covariance trace / spectrum
  - whiteness or concentration diagnostics
- baseline:
  - copy-last, linear shared map, tiny recursive predictor

### Stage B: Innovation coding proxy
- predictor residual만 quantize
- same-bit direct-quantize vs predict+innovation
- metric:
  - reconstruction MSE
  - query-logit MSE

### Stage C: End-to-end bounded patch
- 일부 layer group에만 적용
- retrieval / short PPL sanity

---

## 4. 구현 우선순위

### 먼저 구현할 것

1. `QDRP` 진단 스크립트
   - 이유:
     - 기존 `exp_flip_calibration.py`, `exp_cliffkv_niah.py`를 바로 재사용 가능
     - oracle-vs-proxy, score-vs-risk가 명확해 빠르게 죽이거나 살릴 수 있음

2. `TRIC` 진단 스크립트
   - 이유:
     - 바로 end-to-end 하네스에 넣기보다, depth innovation이 실제로 작은지 먼저 확인해야 함

### 나중으로 미룰 것

- full v3 harness integration
- full TurboQuant coupling
- latent-state decoder 방식

---

## 5. 이번 라운드 코드 기준

이번 라운드에서 필요한 것은 다음 두 가지다.

1. `QDRP`용:
   - synthetic + trace-level smoke 가능한 page-risk selector
   - risk > score가 최소 synthetic에서 성립해야 함

2. `TRIC`용:
   - shared tiny recursive predictor 실험기
   - synthetic depth process에서 copy/linear 대비 개선이 재현되어야 함

이 두 smoke가 통과한 뒤에만 autoresearch loop를 시작한다.
