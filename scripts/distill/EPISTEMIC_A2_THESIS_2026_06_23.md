# Epistemic-A2 Thesis Spine (2026-06-23) — "모름"을 외부 관계로 우회하고 그 사용을 학습한다

> 이 세션 대화 arc(사용자 통찰 연쇄)의 압축 결론. 근거: `FLOW_DISCIPLINE_RESULTS_2026_06_23`(32B gap 분해·G5=0·편차=user-sim)·`EPISTEMIC_ABSTENTION_PROBLEM_2026_06_23`·[[05]]·[[10]]·[[42]]·[[02]]·[[07]]. 딥리서치 2건(`wqidut74b` 분해·`w0r8slp20` abstention/defer)이 방법을 채울 예정.

## 0. Root 진단 — 왜 frontier도 같은 병
LLM 학습은 전부 **P(다음토큰|문맥) 형성**일 뿐, *그 분포의 신뢰도(메타)*를 학습하지 않는다. 그래서 **"모른다"를 토큰으로 만들기가 구조적으로 어렵다**(자기 출력분포 내성이 목적함수 밖). RLHF는 단정적·도움되는 답을 보상 → 환각 인센티브화. ⇒ **scale 문제 아님·모든 frontier 동일.**

## 1. 실패의 통일 (epistemic)
32B capability-gap(gpt-4.1 pass∧32B fail) 전수분해 = **wrong-ORDER 선택 47% · wrong-ACTION 20% · operand 13% · 예산 13% · over-action 7%**, 2차증상=loop/복구실패. 스케일별: gap 7B 50%→14B 20%→32B 13%(경계는 밖으로) **단 실패 *종류*는 스케일-불변**(loop/복구 ~35% 전 스케일 1위). 공통 메타원인 = **"모름/불확실(어느 후보·행동·해 있나)"을 표현·행동 못 함** = "K개 보기 학습 → 5번째 '정답 없음' 선택 불가". (cf. G5(precondition-steering) 인과효과=0: 가이드를 줘도 *쓰도록 학습 안 됨*. 편차=gpt-4.1 user-sim 비결정성 카오스증폭(flip 절반)→pass^1 점비교 무효·결정론 신호만 신뢰.)

## 2. ★아키텍처 — A2=유한 관계, 모델=관계 쓰기, scaffold=관계 집행
- **A2 = 효과적으로 유한한 *관계/규칙*.** *관계 스키마(predicate·rule 형태)=도메인-일반*(retail/airline/banking 동일)·*내용(tuple·값)=도메인별 swap*. = DB 스키마(고정 관계)+도메인 row. 유한성=[[02]] 생성원 closure. (현 `grounding.json` π/⋈/σ·`gate.json` 관계형이 부분 실물.)
- **scaffold(고정)** = 관계 연산 *집행*(σ/⋈/agg·빈결과 판정 = decidable).
- **모델(학습·TBox)** = (1)NL→관계 predicate *formalize* (2)규칙따라 내용 *확인·선택* (3)**빈/모호 결과면 ASK**. 도메인-일반·A2-swap 전이.

## 3. ★핵심 escape — "모름"을 내성이 아니라 *관찰가능한 빈-관계*로
목적함수가 내성형 IDK를 못 만드니, **모름을 외부화**한다:
- scaffold가 σ 계산 → **빈/모호 결과가 모델 문맥에 *토큰*으로 등장.** 1개→선택·**0개→모름→질문**·>1개→모호→질문.
- 모델은 모름을 *무에서 생성* 안 하고 **보이는 빈-결과에 "ask"를 출력**(평범한 입력→출력 매핑·학습가능). ⇒ **불가능한 메타과제를 학습가능한 구체과제로 변환.** 목적함수 한계 우회.
- formalize의 "none"도 **A2 유한 predicate 어휘**로 → "유한 후보 中 맞는 것 없음"=checkable 유한조건(열린 내성 아님).

## 4. 학습 대상 확정 — 무엇을 배우나 (내용 아님)
**도메인-일반 스킬 = "관계규칙 따라 formalize → check+select → 빈/모호면 ASK"** + **abstain 커리큘럼**(학습데이터에 빈/모호→ASK 케이스 *명시 포함*=‘5번째 보기 존재’를 예시로 가르침). 
- 왜 *학습*(프롬프트 아님): [[42]] 소형은 in-context 규칙 무시(prompt-ceiling)→학습이 설치.
- 왜 *외부신호-응답*(내성 아님): 목적함수가 내성 IDK 못 만듦→빈-관계가 외부 손잡이.
- G5 null 해소: scaffold가 가리켜도(외부) 모델이 *쓰도록 학습* 안 됨 → **곱(scaffold×A2-사용학습)** 이어야. G5는 첫 항만 켠 것.

## 5. 기여 (frontier 미보유)
frontier도 *외부 손잡이 없이 내성*하라 요구받아 confabulate. **소형 + 관계형 scaffold(외부 손잡이) + 모름 커리큘럼**은 손잡이가 있어 **frontier가 못 하는 grounded-abstain을 함** = 비용우위(~23×·TCO) + 능력. [[41]] "소형>큰"이 아니라 *systematization+grounded-abstain+경계지도*.

## 6. ★중심 실험 (make-or-break)
**A2-규칙-사용 학습**: 벤치(SOP/TB/CFB/Synth)서 "유한 관계규칙 제시→formalize/check/select·빈→ASK" 궤적 SFT(도메인-일반·내용X·abstain 케이스 포함) → {base vs A2-trained} × scaffold(A2 집행) on tau2(**A2-swap·재학습0**). 검정: ①G5서 0이던 게 학습 후 *pass 전환* ②새 도메인 A2-swap 전이 ③모름-커리큘럼이 over-action↓·loop↓·wrong-order→ask↑. per-scale(7/14/32/72B) = **hook 커버리지×학습 곡선**(어느 크기서 닫히나·어디부터 escalate).
- 정직 NO-GO: (a) 5개가 유한관계로 깔끔 표현 안 되거나 (b) "관계-사용/formalize-none"이 그 크기서 capability-bound면 → 그게 진짜 경계(escalate/scale).
- 측정 주의: user-sim 편차~0.11 → gap=fail-all-3 같은 robust 신호 + 결정론 신호(위반카운트·궤적 census) + 다수trial. pass^1 점추정 단독 무효.

## 7. 분담 한 줄 (최종)
**A2=유한관계(내용만 도메인) · scaffold=관계 집행(빈결과 판정) · 모델=관계 formalize+check+select+빈→ASK(학습·전이) · 모름=빈관계에 grounding해 *외부신호 응답*으로 학습(내성 아님·abstain 커리큘럼).**
