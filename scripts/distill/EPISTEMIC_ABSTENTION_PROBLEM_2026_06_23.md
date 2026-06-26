# 추론 실패의 정밀 재정의 = Epistemic 결함("5번째 보기" 부재) (2026-06-23·사용자 통찰)

> 진입: `FLOW_DISCIPLINE_RESULTS_2026_06_23 §32B capability-gap 분해`(5 실패유형) + 사용자 재정의(2026-06-23).

## 0. 사용자 통찰 (정밀 정의)
LLM의 근본문제 = **"모르는 것을 모른다고 인증"하도록 학습하기 어렵다.** 비유: K개 보기를 학습한 학생은 *모든* 질문을 그 K에 끼워맞추려 하고, K에 답이 없는 K+1번째 문제에서 **"정답 없음(5번째 보기)"을 선택할 줄 모른다**(객관식 4개만 배운 학생이 '답 없음' 옵션을 모름). ⇒ 모델이 **"모름/불확실/내가 틀릴 수 있음"을 표현·행동**하도록 학습돼야 저 실패들이 풀린다.

## 1. ★메타-원인: 5 실패유형이 epistemic 결함 하나로 수렴
| 실패유형(§gap) | epistemic 재해석 |
|---|---|
| ① wrong-ORDER 선택(47%) | 후보(여러 주문) 中 *확신 없는데* 하나를 강행 — "어느 것인지 불확실"을 인정·질문 못 함 |
| ③ 복구/루프(틀린 행동 4-6회 반복) | 틀렸다는 신호를 받고도 *같은 선택 반복* — "내가 틀리고 있다→다른 후보 or 모름" 전환 못 함 |
| ④ 예산 제약충족 | 해 없음/불확실인데 빈 시도 반복 — "이 제약으론 안 됨"을 인정 못 함 |
| ⑤ 과행동(T62) | 행동 불필요/요청 모호한데 *강행* — "행동 안 함/되묻기"라는 선택 부재 |
| ② intent→action | 매핑 불확실인데 한 도구 강행 — 되묻기 부재 |
- ⇒ 공통 = **calibrated "모름/불확실" 신호 부재 → confabulate(강행) + loop(반복)·ASK(질문) 안 함.** = forced-answer 편향(학습 데이터에 unanswerable/abstain 타깃 결핍 → 환각).

## 2. ★제안 메커니즘: iterate → recognize-failure → ASK
- 리스트/후보 결정: best 후보 시도 → 실패신호(gate-deny / tool-error / 불일치)면 → 다음 후보 → ...
- 실패 누적 추적 → 임계(K회 시도 OR 후보 소진 OR 저신뢰)서 **강행 중단하고 사용자에게 *질문***(clarify) 또는 human escalate.
- = "틀린 선택 반복(loop)" → "유한 탐색 후 질문"으로 전환 → wrong-order·loop·budget 회복.

## 3. ★분담: scaffold(결정론) vs learn(epistemic)
- **scaffold(decidable·offload 가능)**: 실패 시도 추적·"K회 실패 or 후보 소진 → ASK/escalate" 강제. = 기존 `T2_RETRY_CONTROLLER` *재프레임* — DIVERSIFY("다르게 해봐")가 소형모델 해쳤음(검증됨) → **복구 행동을 "사용자에게 *질문*"으로** 바꾸는 게 핵심(질문=항상 안전·정보획득). "언제 멈추고 물을지"의 *트리거*는 decidable(K·소진).
- **learn(어려운 핵심·사용자 지적)**: ①*calibrated 불확실성* — 언제 확신 없는지(질문 vs 강행) 판단 ②*clarification 생성* — 무엇을 물을지. 이게 forced-answer 편향 때문에 학습 난제.
- **열린 질문(딥리서치 타깃)**: 학습된 불확실성 없이 **scaffold만으로(K-소진 트리거 + 질문)** 충분한가, 아니면 calibrated abstention을 *학습*해야 하나? (= offload vs learn 경계를 epistemic에 적용.)

## 4. 기존결과 연결
- ⑤ abstention은 §gap의 한 유형이자 ①③④의 *메타-원인* — 따로 + 근본 둘 다.
- retry-controller(존재·DIVERSIFY) = scaffold leg의 1차 시도였으나 소형 해침 → "질문/escalate" 행동으로 재설계 필요.
- §35/§6c "복구(P7)" = 이 epistemic-recovery의 외형. "다르게 재시도"보다 "모름 인정→질문"이 정밀 처방일 수 있음.

## 5. 다음 (딥리서치 2건 병렬)
- #1(진행중 `wqidut74b`): scale 외 추론향상 + 분해 가설.
- #2(이 문서 타깃): **epistemic — 모름 학습 난제·abstention/calibration/selective-prediction·clarification-question(ask vs guess)·iterate-then-ask/learning-to-defer·scaffold가 학습된 불확실성 대체 가능한가.** 5 실패유형에 레버 매핑.
- 결론 후: scaffold leg(K-소진→질문 controller) + learn leg(calibrated abstention·필요시) 설계.
