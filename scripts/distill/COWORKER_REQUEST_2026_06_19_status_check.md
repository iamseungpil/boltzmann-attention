# Coworker 상태 확인 (2026-06-19) — 06-18 lane2 실 e2e 회수 + 다음 단계 정렬

> 자기완결·짧음. 직전 요청 = `COWORKER_REQUEST_2026_06_18_lane2_real_e2e.md`. woori 쪽엔 그 **결과 doc이 없고**(repo엔 `COWORKER_RESULTS_2026_06_17_scale.md`=오프라인만), woori 공유 리모트엔 scale e2e 시뮬 없음, HF dataset `iamseungpil/sopbench-trackb-h200`은 private(woori 401). ⇒ **회수 여부를 woori가 확인 불가 → 먼저 묻습니다.**

## 1. ★확인 요청 (이거 먼저)
**06-18 요청(`lane2_real_e2e` = 32B/72B base, 실 τ² user-sim e2e, pass^1, `--resolve`·`--gate` off, gpt-4.1 user-sim) 이 돌았나요?**
- (a) **돌았으면**: 결과 위치(HF path / amlt job)와 per-(크기×도메인) `pass^1`·`pass^k`·`mean_reward`·`n`를 `COWORKER_RESULTS_2026_06_19_*.md`로 회수 부탁. (woori가 못 봄 = private dataset.)
- (b) **안 돌았으면**: 우선순위만 알려주세요(아래 §2가 측정축을 바꾸므로 안 돌았으면 06-18 그대로 말고 §2로 갱신 권장).

## 2. ★다음 단계 정렬 (06-18 이후 woori 진전 — 측정축이 바뀜)
woori가 이번 세션(2026-06-19)에 **grounding-spec wiring**을 구현·검증했습니다(`A2_GROUNDING_WIRING_DESIGN`·`t2_resolve_patch` 재건·도메인-일반 관계대수 투영·retail+airline). 핵심 진전:
- **§7 조건부 측정**: `P(ground_OK | resolve-emitted ∧ producer-present)` = 선택-formalize 순수 정확도(spec-fail/C0/P2b 분해). bare pass보다 진단적.
- **측정 결론(7B solo_cfb_mid 실 e2e)**: 병목 = **formalize**(모델이 *검색-키*[product_type·origin/dest]를 *선택 among*에 섞음·기준 under-specify). 엔진 which-output 일반화로 ground_OK 0→~12%·잔여는 formalize(scale/학습 몫).
- **비용모델 우선순위**(`13-absorption-priority`): scale→학습(무망각)→(최후)scaffold/A2. ⇒ **"scale가 선택-formalize를 푸나"가 핵심 질문.**

### 함의 (요청 갱신 제안)
- **06-17 오프라인 op-eval은 이미 선택-formalize scale-plateau를 시사**(retail new_item_id acc 0.44/0.38/0.41 @ 32/72/235B·235B↓). 단 τ² task 프록시론 철회됨.
- **base는 resolve_selection을 안 emit**(0회) → `ground_OK|present`를 *base e2e*로는 못 잼(C0=0). ⇒ scale에서 선택-formalize를 재려면 **forced/prompted selection**(오프라인 op-eval을 grounding-spec resolver로 *통제 재측정*) 또는 각 크기 학습 어댑터 필요.
- 따라서 새 큰 실험(225B 포함) 전에 **(1) 06-18 회수 확인 → (2) woori가 측정축(forced-selection vs pure-e2e)을 확정 → (3) 그때 grounding-aware scale 요청서 발행**이 순서. 지금은 **회수 확인만** 부탁.

## 3. 분담 (확정)
- **woori**: ≤ **32B int8**까지 로컬(A6000 49GB). 1.5/7/14/32B-int8 곡선 하단.
- **coworker**: **32B fp16 · 72B · 225B**(대형·TP·H200). woori가 못 도는 곡선 상단.
- 측정·집계·박제 = woori. coworker = inference-only·temp0·frozen·COST GUARD(user-sim=gpt-4.1·Claude 금지·키는 coworker 환경).

## 4. TL;DR
1. **06-18 실 e2e 돌았나? 결과 어디?** (woori가 private dataset 못 봄.)
2. 안 돌았으면 보류 — woori가 grounding-aware 측정축 확정 후 §3 분담(32Bfp16/72B/225B)으로 갱신 요청 발행.
3. 큰 새 실험 아직 발주 아님 — 회수·정렬 먼저(중복·낭비 방지).
