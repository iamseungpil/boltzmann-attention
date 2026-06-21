# C10 (Operand-Formalize) 최소 LoRA 설계 — scale-불변 *유일* 학습잔여를 무붕괴로 내재화 (2026-06-21)

> 상위 = `EXPERIMENT_DESIGN §0★★`·`CAPABILITY_LEVER_ALLOCATION §10`(L-vs-E)·`ma/M_A_RESULTS §32-35`. 불변 [[00-thesis]][[11-transfer-direction]][[12-diversity-required]][[13-absorption-priority]].
> 위치 = 둘째 기둥의 ***학습* leg**. C3=엔진(offload)·C8=scaffold(retry)·**C10=유일하게 *학습*이 정당한 능력**(scale도 scaffold도 안 됨).

## 0. 가설 (한 줄)
**operand-formalize(NL서 옳은 값/차이속성 추출)는 scale-불변·decidable 아님 = scaffold/scale 둘 다 못 함 → *최소 전이 LoRA*(무붕괴·도메인-일반·ABox-swap)가 유일 cheap 레버.** = [[00-thesis]]의 "LLM 학습=NL→formalize"가 *진짜* 시험되는 자리(드리프트된 tau2-학습 아님).

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

## 5. 실험 arms
| arm | 구성 | 의미 |
|---|---|---|
| base | 7B floor | operand-acc baseline |
| +instruction | forced-replay 재현(학습0) | 학습-불요 상한(0.62 재현?) |
| **+operand-LoRA** | 최소 r·replay | ★학습 효과·무붕괴 |
| +escalate | operand step만 32B/72B | **L-vs-E** 비교(§10) |
- 평가 = **scaffold 내 *operand step 격리*** (single-facet를 full-agent로 돌리지 마라·§3.5 mismatch). flow engine이 operand 결정만 LoRA에 위임.

## 6. 측정
- **operand-accuracy**(§32-34 지표·offline 신뢰가능? → 실 e2e operand step도)·**transfer pass**(retail+airline ABox-swap)·**무붕괴**(held-out 일반능력 Δ)·**L-vs-E 비용**(LoRA build+OpEx vs escalate OpEx).

## 7. GO / NO-GO (지표 분리·C8 리뷰 교훈)
- **operand-GO**: operand-acc 유의↑(0.62 instruction 상한 근접 or 초과) ∧ held-out 일반능력 불변(무붕괴) ∧ **벤치→tau2 전이 보존**(ABox-swap·재학습0).
- **경계/약**: operand-acc↑인데 전이 안 됨(벤치 표면학습) → [[12]] 다양성 부족 진단. or operand-acc↑인데 full e2e pass≈(operand가 그 태스크 병목 아님) → 정직 기록(operand=소수 태스크 결정).
- **NO-GO**: 무붕괴 깨짐(replay↑로도) → rank↓·본질이면 operand=escalate(E)가 답(L-vs-E가 E로 판정·여전히 유효=비용결론).
- ★**L-vs-E 판정**: cost_L(LoRA build/빈도+OpEx) vs cost_E(operand step escalate). operand 빈도·LoRA 효과로 결정.

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
