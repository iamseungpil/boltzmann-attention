# C10 (Operand-Formalize) 최소 LoRA 설계 — scale-불변 *유일* 학습잔여를 무붕괴로 내재화 (2026-06-21)

> 상위 = `EXPERIMENT_DESIGN §0★★`·`CAPABILITY_LEVER_ALLOCATION §10`(L-vs-E)·`ma/M_A_RESULTS §32-35`. 불변 [[00-thesis]][[11-transfer-direction]][[12-diversity-required]][[13-absorption-priority]].
> 위치 = 둘째 기둥의 ***학습* leg**. C3=엔진(offload)·C8=scaffold(retry)·**C10=유일하게 *학습*이 정당한 능력**(scale도 scaffold도 안 됨).

## 0. 가설·프레임 (리뷰 위험2 reframe — "유일 학습 leg"가 아니라 *학습 정당성의 최종 판별*)
**operand-formalize는 scale-불변·decidable 아님(scaffold/scale 둘 다 못 함). 단 *instruction(무료·학습0·무망각)이 이미 0.62=32B 천장(62)에 도달*(forced-replay) → C10의 진짜 질문 = "LoRA가 *무료 instruction을 유의하게 초과*하나."**
- **LoRA ≫ instruction** → *정당한 학습 leg 존재*([[00-thesis]] "LLM=NL→formalize" 학습이 *최소 한 곳* 산다).
- **LoRA ≤ instruction** → operand도 prompt-fixable = **정당한 학습 0** = 논문 완전 **cost-conclusion**(engine+scaffold+prompt·학습0·프론티어표 정합). ← 이것도 *강한 정직 결과*.
- ⇒ C10 = "어떤 학습이라도 정당한가"의 **결정 실험.** GO든 NO-GO든 논문 기여. (드리프트된 tau2-학습 아님·벤치학습→ABox-swap.)
- **★competitor = escalate가 아니라 *무료 instruction*** (§5 1차 비교군).

### 0a. ★§23D 선제 방어 (리뷰 위험1·핵심) — C10은 §23D 퇴행의 *재시도가 아님*
`§23D`(메모리04) 박제: "operand(**wide-substitute under-extraction**) 학습 → τ² **라우팅** 퇴행 0.44→0.30." C10이 operand를 학습하므로 명시 방어:
- **§23D 퇴행 = (i) wide under-extraction 학습 탓 / C10 = (ii) *value-comprehension 핵만* + behavior-token 마스킹 → 라우팅(op-naming) *미접촉*.** = 다른 것을 학습.
- ⚠️ **§23D 퇴행은 *라우팅*(task-특정)이라 일반능력 held-out이 *못 잡음*** → §7 GO에 **라우팅-acc 무퇴행**을 *별도 조건*으로(아래).

## 1. 동기 (실측)
- **scale-불변**(`§35`): operand(B) 7B 83→14B 66→32B 62 plateau = **scale이 못 고침**(C3 grounding 76→3과 정반대). 32B도 62 실패 = 모두의 천장.
- **능력은 *있고* default가 틀림**: forced-replay 격리+instruction → 7B operand 0.14→**0.62**(`§6`). ⇒ capability gap 아니라 *발현/default* gap → 최소 LoRA로 default 교정 가능 가설.
- **operand 두 부분**(`§32-34`): (i)**under-extraction**(multi-attr keep-rest 과소추출) → decomp(offload-ish) (ii)**wrong-value-selection**(NL 값 오독 "31inch"→gold"28inch") = **순수 formalize 정확도·offload 불가 = LoRA 타깃 핵심**. (i)는 scaffold, (ii)만 학습.

## 2. 무엇을 학습 (정확히)
- **타깃 = NL→operand 정확도**: 후보 간 *차이* 속성 명명(검색/카테고리 키 제외)·NL서 *옳은 값* 읽기·날조 0. = facet-4 operand-formalize.
- **비-타깃(혼동 금지)**: concrete resolution(기준→item)=엔진(C9)·under-extraction=decomp·grounding=엔진(C3). 이들 LoRA로 굽지 마라([[00-thesis]] 위반).
- ⇒ LoRA는 **NL-comprehension 핵**만(§32-34 (ii)).

## 3. 데이터 (★§3.5 붕괴 회피·드리프트 금지)
- **§3.5 교훈**: 추상 단일도구 SFT가 full-agent 파탄(solo_*/fact_* 날조·빈arg). ⇒ **실·다양 데이터·실도구·concrete 값** 필수.
- **소스 = 학습벤치(Synth/CFB operand facet)** — *도메인-일반* operand-formalize. raw 카탈로그(실 color/size + 검색키 섞임)→차이속성/값 추출. **tau2 학습 금지**([[11]])=전이 타깃.
- **다양성**([[12]]): 표현/구조 다양(L×S×P×R)·단일템플릿 금지(역전이).
- ReST-style 옵션: 강한 instruction/32B로 정답 operand 궤적 생성→필터→학습(단 *벤치*서·tau2 아님).

## 4. 무붕괴 (anti-collapse·§3.5 직접 해소)
- **최소 rank LoRA**(r4-8·solo_sts r64 아님)·**replay**(일반 tool-use 혼합·비율 sweep)·**per-iter held-out 일반능력 eval**(깨지면 replay↑/rank↓/stop)·early-stop.
- **behavior-token 마스킹**: operand 결정 토큰에만 loss.
- 목표 = operand-acc↑ ∧ held-out 일반능력 *불변*.

## 5. 실험 arms (리뷰 위험2 — instruction = ★1차 경쟁자)
| arm | 구성 | 의미 |
|---|---|---|
| base | 7B floor (격리 step·무instruction) | operand-acc baseline |
| **+instruction** | 격리 step + operand 지시(학습0·무망각) | ★**1차 경쟁자**(0.62·무료 천장) |
| **+operand-LoRA** | 최소 r·replay·격리 step | ★학습 — **instruction을 *초과*해야 정당** |
| +escalate | operand step만 32B/72B | L-vs-E 비교(§10) |
- 평가 = **scaffold 내 *operand step 격리*** (single-facet를 full-agent로 돌리지 마라·§3.5). instruction·LoRA *같은 격리 step*서 비교(apples-to-apples).
- ⚠️ **live-vs-isolated 주의**(C6 교훈): instruction은 *격리 step*서 0.62지만 *live 멀티턴*선 7B 무효였음(§35 크기의존). 격리 step이 배포 형태면 instruction이 유효 경쟁자·LoRA가 그걸 넘어야. (instruction OpEx=매콜 토큰 / LoRA=1회 CapEx → 고빈도면 동-acc라도 LoRA OpEx 이득 가능·§7 부차조건.)

## 6. 측정
- **operand-accuracy**(§32-34 지표·offline 신뢰가능? → 실 e2e operand step도)·**transfer pass**(retail+airline ABox-swap)·**무붕괴**(held-out 일반능력 Δ)·**L-vs-E 비용**(LoRA build+OpEx vs escalate OpEx).

## 7. GO / NO-GO (리뷰 위험1·2 반영 — 라우팅보존 + instruction 초과)
**학습-정당 GO = 3조건 *모두*:**
1. ★**LoRA ≫ instruction**(유의 초과·"0.62 근접=매칭"은 GO 아님·매칭이면 무료 instruction 승) — 리뷰 위험2.
2. ★**라우팅-acc 무퇴행**(§23D 0.44→0.30 *재현 안 됨*·별도 측정·일반 held-out이 못 잡음) — 리뷰 위험1.
3. **무붕괴**(held-out 일반능력 불변) ∧ **벤치→tau2 전이 보존**(ABox-swap·재학습0).
- **= 정당 학습 leg 존재**(thesis 학습이 최소 한 곳 산다).

**NO-GO도 *강한 정직 결과* (둘 다 논문 기여):**
- **LoRA ≤ instruction** → operand prompt-fixable = **정당 학습 0 = 완전 cost-conclusion**(engine+scaffold+prompt·학습0). ← 프론티어표 정합·강한 결과.
- **라우팅 퇴행 재현**(조건2 실패) → §23D 확정·operand 학습은 라우팅 희생 = E/instruction이 답.
- **무붕괴 깨짐**(replay↑로도) → rank↓·본질이면 E.
- **전이 안 됨**(acc↑이나) → [[12]] 다양성 부족. **e2e pass≈**(acc↑이나·§44) → operand=소수경로·L-vs-E가 E/instruction으로(위험3).
- ★**L-vs-E-vs-instruction 판정**: cost(instruction·매콜토큰 OpEx·CapEx0) vs cost_L(LoRA CapEx/빈도+저OpEx) vs cost_E(escalate). operand 빈도 × instruction-대비-이득(둘 다 작을 수 있음·위험3) 으로 결정.

## 8. Risk
- §3.5 붕괴 재발(추상·narrow) → 실·다양·최소rank·replay로 차단·per-iter eval.
- single-facet full-agent mismatch → scaffold 내 operand step *격리* 평가만.
- under-extraction(decomp) vs NL-comprehension(LoRA) 혼동 → 타깃 (ii)만.
- offline op-eval 신뢰불가([[03]]) → 실 e2e operand step 재확인.

## 9. 프레임워크 정합
- operand = irreducible NL→formalize = **[[00-thesis]] 학습 본체**(decidable→offload 후 *남는* 것)·도메인-일반·ABox-swap 전이([[11]]). = 둘째기둥 중 *유일한 정당 학습*.
- C3(엔진)+C8(scaffold)+C10(최소학습) = "scale 능력 = 능력별 *가장 싼 레버*로 분해"의 세 leg. C10이 "학습이 *어디서* 정당한가"의 경계.
- L-vs-E(§10): operand이 L(학습)인지 E(escalate)인지도 *측정*으로 — 작은 몫이면 E가 쌀 수도.

## 10. 다음 (설계 확정·C8 후)
1. 벤치 operand-formalize 데이터 빌드(실·다양·Synth/CFB·tau2 제외) + 무붕괴 학습 드라이버(r4-8·replay).
2. arms eval(base/instruction/LoRA/escalate) + held-out 일반능력 + 전이.
3. 결과 → `M_A_RESULTS §35d`(operand=L vs E 판정).
