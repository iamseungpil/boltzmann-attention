# FETCH=결정론 / SELECT=LLM·learn — 두 날개 경계 정식화 (2026-06-22·사용자)

> 사용자 통찰(2026-06-22): "autofetch는 결정론이라 유동성 적다. fetch 후보가 여럿일 때 *고르기*는 결정론이 힘들고 통계적 LLM이 더 잘한다." → 이 경계를 정식화하고 다음 learn을 B(selection)로 재타깃.
> 상위 = `RULE_LEVER_COST_EFFICIENCY_PROGRAM`·§35(A vs B)·`CFBSYNTH_P2B_P4_DESIGN`(COPY vs COMPUTE)·[[10-roles-deterministic]].

## §1. 경계 (확정)
fetch-first/grounding 능력은 *두 부분*으로 쪼개진다:
- **(A) FETCH = "값 없으면 실제 후보를 *가져와라*·날조 금지"** → **결정론(autofetch/provenance gate)**. 정답이 하나(날조 금지)라 *유동적 판단 아님* = 뺏을 유동성이 없음. 결정론 적합.
- **(B) SELECT = "가져온 실후보 중 user 의도에 맞는 *하나를 골라라*"** → **두 갈래**(CFBSYNTH §3·[[10]] 정합):
  - **(B1) 의미/기술 매칭**(filter) = **LLM·learn**: "파란 키보드"·"이 이름의 계좌" = NL→레코드 의미매칭 = 유동·통계적 = LLM 강점. ★사용자가 지목한 부분.
  - **(B2) 순서/집계**(argmax/rank/Nth) = **결정론 resolve 엔진**: "가장 싼"·"최근 것" = decidable 계산 = 엔진(COMPUTE·[[10]] "선택기=결정론"은 *이것*을 의미). 단 기준(criteria) 추출=LLM.
  - ⇒ "select=LLM"은 **B1(의미매칭)**·B2(순서집계)는 결정론. 혼동 주의.

## §2. 우리 구현이 이미 이 철학 (검증)
`t2_gate_patch._autofetch_text`: provenance-deny 시 producer를 결정론 호출 → **그 출력(=실후보 목록)을 모델에 *주입*** + "copy a REAL value." **후보 하나를 자동선택하지 않음** — 선택은 모델. ⇒ autofetch = 후보 *제공*(날조 차단), 선택 = LLM. = 이 경계 그대로.
- ⇒ **autofetch의 [[05]] "유동성 동결" 우려는 좁다**: 동결=날조금지(비유동)뿐·유동적 *선택*은 LLM에 보존.

## §3. 실측이 경계를 확증
autofetch on → 실패가 **A(provenance/날조)→B(operand/잘못된 선택)로 이동**(retail A_notfound 0.27→0.14·B_wrong_write 잔존). = autofetch가 닫는 것=A(결정론 적합)·남는 것=B(LLM/learn). §35 "B=scale-불변 학습잔여(operand)" 정합.

## §4. ★learn 재타깃 = A(날조)가 아니라 B(selection)
- **왜 SFT/DPO가 실패했나(2026-06-22)**: 둘 다 "**날조하지 마(A)**"를 learn으로 가르치려 함 → A는 결정론(autofetch)이 더 잘하는 비유동 영역 → learn 헛돎·abstract→real 전이 실패(SFT 52·DPO 35 ≈ base).
- **재타깃**: learn은 **B(가져온 실후보 중 의미적 선택)**를 겨눈다. = §35 operand·CFBSYNTH의 P4-select(COMPUTE)·content-op(resolve_selection). 선택은 *prior 억제*가 아니라 *능력 추가*라 SFT(양성예시)가 적합할 공산(penalty 불요).
- **데이터**: cfbsynth의 list+select 구조 (autofetch가 후보 제공 가정 → 날조 confound 제거 → 순수 selection 학습). user-NL-description → matching record id. 도메인-일반(익명 도구·tau2 0).
- **eval**: autofetch ON(후보 제공) 고정 + {base vs B-learn} → B_wrong_write/grounded_other(선택실패) 감소하나. = 두 날개 곱(결정론 fetch × learn select).

## §5. 아키텍처 결론
> **결정론(scaffold/autofetch) = 실후보 *가져옴*(날조 차단·비유동) · LLM/learn = 실후보 중 *고름*(의미적·유동).**
- 새 도메인 전이: autofetch=엔진+A2 producer-map(A2-swap) · select=모델(도메인-일반 능력·learn으로 강화). banking서 검증 예정(account/user_id 후보 제공→선택).
- ⇒ fetch-first "한 규칙"은 사실 **A(offload)+B(learn)** 두 날개 합. 곡선에서 A는 결정론 knee·B는 learn knee.

## §6. 다음
1. banking prior 확인됨(user_id 날조: 1234567890·user123456) → banking.gate.json(producer-map: user_id←get_user_information_by_name/email) 저작 → autofetch arm = A-offload 전이 실증.
2. B-selection learn 설계·빌드(`C10_OPERAND` 계열·cfbsynth P4-select·SFT) → autofetch×B-learn 곱 측정.

**불변**: [[10-roles-deterministic]](선택기·검증기 역할)·[[05]](유동성 동결 좁힘)·§35·[[13]]. 상위 `RULE_LEVER_COST_EFFICIENCY_PROGRAM`.
