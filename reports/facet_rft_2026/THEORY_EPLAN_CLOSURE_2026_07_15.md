# 이론 검증 — 2층 라우터(outer loop + inner router)의 closure (2026-07-15)

> 정본. 사용자 이론 통찰(H_min/엔트로피가 loop 두 난제를 푸나·open-world는 ASK로 닫히나) 검증.
> 입력: C88(voting-vacuity)·C89(correlated 하위유형)·C90(2층)·C83/§16-18(H_min)·C80(reach)·PREEMPTION_SCAN.
> 등급: 이론 [D](논증)·H_min 측정 [S](C83)·banking reach 열거가능성 [M부분](C80 §14.6·재forensic 대기).

## 0. 주장
**2층 라우터는 새 primitive 없이 3원소 {결정론 열거(GET/FIND), 엔트로피-gate(H_min), ASK}로 닫힌다.** outer loop은 inner router와 같은 도구의 across-item 적용이다.

## 1. 2층 구조 (C90)
- **inner router (within-item·per-step)**: 오류 영역별 {voting|verify|ASK} — variance>0→voting·decidable-systematic→verify·non-decidable→ASK. (C88/C89·PREEMPTION_SCAN서 core 미선점 확인.)
- **outer loop (across-item·E-PLAN)**: 항목 열거→격리(per-item 컨텍스트)→완료추적. 격리=선행 decomposition(양보).
- 두 난제(C90): ①열거/discovery(reach) ②결정론 loop-강제(under-action).

## 2. 문제2 (loop 강제·정지) = H_min이 완전히 해결 [D]
- 정지조건을 LLM 자율 아니라 **엔트로피 기준**으로: `CONTINUE while H(남은목표|지금까지) > H_min_floor · STOP at floor`.
- under-action(조기종료·C80 100% user-턴)이 원리적 차단 — "할 일 남았나"를 결정론 정보량으로 판정.
- = [[07]] control-not-prompt의 정보이론판 · VOI/EIG(Info-Gain `2606.03135`)를 **loop 종료조건**에 적용.

## 3. 문제1 (열거/discovery) = 엔트로피는 support 위에서만 정의됨 — 2분
**핵심 구분: H(X)는 후보공간(support) 위 불확실성. discovery는 support 확장.**
- **closed-world (열거가능 support)**: "원천 나열 → 각 항목 in-scope?" → **GET/FIND(결정론) + 항목별 엔트로피**로 닫힘. [D]
- **open-world (unknown-unknowns·support 밖)**: 없는 항목은 엔트로피에 0 기여 → 엔트로피 단독 *못 봄*. **→ ASK로 닫음(아래 §4).**

## 4. ★open-world closure = completeness-ASK (사용자 통찰·핵심)
- unknown-unknowns = 정의상 non-decidable → **ASK**(우리 프레임 그대로).
- 구체: **completeness-confirmation ASK** — "찾은 것 = X·Y·Z. 이게 전부입니까?" → 사용자가 완비성 인증/누락추가.
- **정당성**: 에이전트 목표 = *사용자 의도* → **사용자가 완비성 oracle**. "그게 다예요" = ground truth → open→closed 전환.
- **비용 bounded**: 무한 아니라 완주 시 1회 확인 ASK → H_min floor 도달(STOP 신호=잔여 엔트로피 0). per-candidate 확인 필요 시 H_min/VOI가 순서·수 최소화.
- **환원불가 잔여**: 사용자*도* 모르고 열거가능 원천*도* 없을 때만. **유계 과제(유한 거래이력)=거래이력이 완비 후보원천 → 잔여 없음.**

## 5. closure 정리 (E-PLAN)
유계·열거가능 다중-item 과제에서 outer loop은 다음으로 닫힌다:
1. **열거** = 유계원천 GET/FIND(결정론) → 알려진 support.
2. **per-item** = inner router {voting|verify|ASK}.
3. **완비성** = closed-world(support coverage) + open-world(**completeness-ASK**·사용자 인증).
4. **종료** = 엔트로피-gate(잔여목표 H > floor면 계속·completeness-ASK "그게 다"가 H→floor).
→ **새 primitive 0.** 두 난제는 *naive LLM*엔 어렵지만(under-act·자기완비성 인증 불가), **결정론 엔트로피-gate + ASK 컨트롤러**로 닫힘.

## 6. banking 실측 (reach forensic·`bank_reach_forensic.py`·로컬 17 frontier 궤적·1048 0-제출·[[08]] 3분·[M])
- **A. enumerable(agent가 tool result/user서 봄)→under-action(ACT): 526(50.2%)**
- **B. queryable(원천 존재·다른 run이 surface·이 run 미조회)→under-action(ENUM): 199(19.0%)**
- **C. never-surfaced(어느 run도 tool result에 없음=open-world 후보·상한): 323(30.8%)**
- ⇒ **closed-world(A+B·원천 존재·실패=under-action) = 725(69.2%)** = C80 §14.6 "70%"와 정합. → **entropy-gate + 강제 열거로 닫힘**(문제2·문제1-closed).
- ⇒ **open-world 후보(C) = 323(30.8%·상한)** → **completeness-ASK 필요**(완비 GET 도구 존재 시 상당수 closed-world 이동=상한).
- **★도구 확인 완료(2026-07-15·`listtool` 분석)**: banking에 **완비 거래목록 도구 존재** — `get_bank_account_transactions`(528회·계좌 전거래)·`get_credit_card_transactions_by_user`(420회·유저 전거래·호출당 최대 77 txn). ⇒ **완비 GET/FIND 원천 존재 → banking reach는 원리적으로 ~완전 closed-world.**
- **★정직 재수정(v2)**: C(31% never-surfaced)는 **진짜 open-world 아니라 under-action on 완비열거**(agent가 전 계좌/카드를 안 열거·원천은 존재). ⇒ **banking open-world 잔여 ≈ 0**(앞 "31% load-bearing"은 과대평가·교정). **completeness-ASK = banking서 드문 fallback**·지배 fix = **강제 완비열거(FIND: 전 계좌→get_*_transactions) + H_min 종료**. → C91 closure 강화(문제1 대부분 closed-world·문제2 entropy-gate).
- **잔여**: 31%가 계좌를 알지만 미열거(under-action)인지 계좌자체 미발견인지 추가 forensic 가능·단 완비도구 존재로 상한은 확정.

## 7. novelty 위치 (PREEMPTION_SCAN 정합)
- 부품: 엔트로피/VOI-ask(Info-Gain)·decidable-offload(PAL)·decomposition(TDP) = 선점·인용.
- **미선점 = 이 closure 자체**: {열거+엔트로피-gate+completeness-ASK}로 outer loop을, {voting|verify|ASK}로 inner를, **하나의 엔트로피/decidability 프레임으로 2층 통일**. DR `wf_23666af5-a45`가 "엔트로피-over-열거가능-support coverage" + "completeness-ASK" 선점여부 확인 중.

## 8. caveat
- 이론 [D]·논증. H_min 계산가능성 [S](C83 은행 4.27/2.60bit). banking 열거가능성 [M부분](§6 forensic 대기).
- completeness-ASK는 사용자=intent-oracle 전제(유계과제서 성립·무계/사용자-무지식은 잔여).
- 엔트로피-gate는 H(잔여목표|ctx) 계산 필요=calibration 전제(§18 conformal 보정·C83 caveat 상속).
