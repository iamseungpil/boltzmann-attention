# Track B — 기전 논문 골격 (2026-07-19 · 사용자 지시)

**가제**: "Degenerate-Cue Interference: An Associative-Memory Account of Same-Rule Failures in Batched LLM Judgments"
**한 줄 논지**: softmax 어텐션=연상기억(modern Hopfield)에서, **같은 규칙 조항에 추론-결합한 선행 항목들은 상관 패턴
(준-축퇴 상태)**을 이루어 target의 조항-검색 질량을 희석한다(ΔF=ΔE−T·log g). 어휘-앵커 항목(Δ大)은 면역이고
의미-연합 항목(Δ小)은 k*=e^{βΔ−θ}에서 계단 실패한다 — 행동·어텐션·인과개입 3층에서 검증한다.

## 이론 (§작성 계획)
1. **접속 정리**: Ramsauer+ 2020 modern Hopfield — 분리 잘 된 패턴=지수 용량·상관 패턴=준안정 혼합으로 검색 실패.
2. **축퇴 특수화**: 같은-조항 처리 행 k개 = 준-축퇴 저장 패턴 g≈k → 결합 odds = βΔE − log g (엔트로피 잠식).
3. **3단 인과 사슬(구조적 가능성)**: (i) causal 프리필서 개입 행이 조항 value 흡수(residual)
   (ii) k_i=W_K h_i라 key가 조항-방향 성분 획득 (iii) softmax 정규화가 질량 분할·value 혼합 readout.
4. **"추론"의 환원**: 판단-의존 = 어휘 앵커 없이 의미-연합 마진 Δ小. 명시-앵커 = 준-중복 토큰 매칭 Δ大 = k*→∞.
5. 도출 예측: P1 causal 비대칭(후행 무해·✅arm4 확인) · P2 log k 감쇠/계단(✅k*=2) · P5 유사성-게이팅(✅이중해리)
   · **P3 어텐션 질량 a_C(k) 감소·비유사선 유지 [미검증]** · **P6 edge-knockout 시 판정 회복 [미검증]** · P4 온도→k* 지수 [미검증].

## 행동 증거 (완료 [S] — Track A와 공유)
배제 사슬 4단·입/출력 분리·이중 해리·k*=2 계단·합성 confound 기각 (§2m).

## 남은 필수 실험 (Track B 완결 조건 — 이것 없이 기전 주장 불가)
- [ ] **B0 행동 재현 전제**: full-fidelity 프롬프트 logit 판독으로 어느 크기가 k-계단을 재현하는지 (진행 중 — 크기 스윕).
      1차 P3(14B·축약 프롬프트)는 k=0부터 실패 = 전제 미충족으로 무효 처리(정직 기록).
- [ ] **B1 = P3 어텐션 곡선**: 재현 모델서 판정-쿼리→조항-C 토큰 질량 a_C(k)·echo 질량·유사 vs 비유사 대조.
      (KV-cache 2-pass 트릭: 마지막 쿼리만 어텐션 계산 → 메모리 무시 가능. 32B 필요 시 :8140 일시 중단·GPU 측정.)
- [ ] **B2 = P6 인과 개입(금표준)**: **4D attention-mask knockout** — target-행 이후 쿼리들이 개입-행 토큰에 어텐션 못
      하게 차단(위치 id는 보존=위치 효과와 내용 간섭 분리) → P(5) 회복하면 인과 확정. 층별 knockout으로 결합 층 국소화.
- [ ] B3 = P4 온도: logit 온도 조작 → k* 지수 이동 확인.
- [ ] B4 (선택) 조항-방향 기하: 개입-행 key들의 W_K v_C 방향 사영이 k와 함께 성장하는지 직접 측정 — (ii) 단계 검증.

## 관련연구 (기전 축)
Ramsauer+ 2020(Hopfield)·Bietti+ 2023(연상기억 관점)·attention sink/StreamingLLM·duplicate-token/induction heads·
retrieval heads·Found-in-the-Middle(위치 어텐션 보정)·Unable to Forget(log-linear=우리 log g 항의 행동 발현 주장)·
fan effect/cue overload(Anderson·Watkins — 인지과학 대응·release-from-PI 패러다임 차용 명시).

## HEAT/Boltzmann-attention 접속 (사용자 이론)
토큰-거리 감쇠 항과 별개로 **의미 축퇴 엔트로피 항**(g=유사 선행 세그먼트 수)을 자유에너지에 추가하는 확장 —
LiM(거리 항)과 cue overload(축퇴 항)를 한 식으로 통합하는 이론 절 후보. (선생님 이론 프레임과의 정식 결합은 별도 논의.)

## 그림 계획
F1 이론 개요(축퇴 자유에너지) / F2 행동 3층(배제·해리·계단) / F3 a_C(k) 곡선(유사/비유사) / F4 knockout 회복
/ F5 크기별 k*(Δ(size) 해석) / F6 층별 국소화.

## 리스크·정직 조항
- 행동-재현 모델을 못 찾으면(only 32B) GPU 측정 필수 — :8140 중단 필요(리스크 관리: 프로브 일시 불가).
- knockout이 회복 안 되면: 희석(어텐션) 아닌 값-혼합/표상 오염 경로 — 모델 수정 필요(그 자체로 결과).
- "새 어텐션 문제" 주장 금지 — 연상기억 정리의 실전 발현+단위 확정으로 포지셔닝.
