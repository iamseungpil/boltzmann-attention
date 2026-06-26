# Escape-Scope Diagnostic — Stage-1 정성 카탈로그 (2026-06-24·수동·n=15)

> 설계 = `ESCAPE_SCOPE_DIAGNOSTIC_LAYERS_AUG_2026_06_24.md`. 입력 = gap 15 task(gpt4.1 pass∧32B fail-all-3). 방법 = 층(escape_layer_decomp 기계적) + faithful σ(db.json 수동 조회·유저 리터럴 제약 인코딩) + §3.5 레버. **★정성·방향 신호이지 비율 아님(n=15)**·**레버-종류 분할이지 해결 분할 아님(§6 caveat)**·σ 분류는 수동 판단(S2 대면검증 대상).

## 1. 케이스별 분류표
| task | 층 | faithful σ (참 DB) | 분류 | 레버 | 비고 |
|---|---|---|---|---|---|
| 71 | L1 | σ_{state=DC}(orders)=**{#W5270061} 유일** | **ⓑ mis-ground** | 학습-formalize | 32B→Charlotte 주문. "DC"를 filter로 인코딩 실패 |
| 72 | L0/L1 | σ_{DC}=유일 (+operator 혼동) | **ⓑ mis-ground** | 학습-formalize | modify_items↔address 혼동+wrong order |
| 74 | L1 | σ_{pending∧items=5}=**{#W3189752} 유일** | **ⓑ mis-ground** | 학습-formalize | 32B→3-item 주문 취소. "five items"=cardinality formalize 실패 |
| 101 | L1 | σ_{2 watches}=**{#W4219264} 유일** | **ⓑ mis-ground** | 학습-formalize | 32B→speaker 주문에 주소변경 |
| 102 | L1/L3 | σ_{2 watches}=유일 +주소⋈ | **ⓑ mis-ground** | 학습-formalize | 주소 출처(다른 주문 ⋈) 틀림 |
| 17 | L3-attr | 주문 명시(|σ|=1)·기존주소 복사 | **ⓑ-op (verbatim)** | 학습-formalize-fidelity | "123 Elm St"≠"123 Elm Street" 약어 |
| 8 | L3 | brightness=high∧avail (tie→prefer ord) | **ⓑ-op** + ⓐ-resolve-ord | 결정론-B2 (+OVER) | 32B→battery(avail=False) 선택=availability 무시 |
| 36 | L2 | cheapest-per-item (각 min price) | ⓑ-op(resolve) +OVER | 결정론-B2 (+stop) | "switch all to cheapest"=B2 |
| 37 | L2 | cheapest-per-item | ⓑ-op(resolve) +OVER | 결정론-B2 (+stop) | 36과 동형($1150) |
| 29 | L3/multi | bamboo∧28(tie→max price) + hose⋈pending | ⓑ +L0/L1 혼동 | 결정론-B2 + ⋈ + formalize | 2-exchange 혼동·needle(⋈) |
| 34 | L0 | policy: partial-cancel-pending 불가→addr | L0 wrong-branch +OVER | 결정론-eligibility (+stop) | 32B→틀린 branch(부분수정 시도) |
| 38 | L0 | arithmetic: cheapest-sum>thresh→cancel | L0 wrong-branch | 결정론-arith (+stop) | 32B→cancel 안 함(계산 누락) |
| 62 | OVER | gold=**0 write**(조건평가→무행동) | OVER (3 spurious writes) | 결정론-stop/commit | adversarial+조건. 32B 과행동 |
| 41 | MISS | (gold 3-write 중 1만·jigsaw=fewest=B2) | MISS (incomplete) | 상류 + B2 | 미완료 |
| 85 | MISS | (gold 1-write·0 traj write) | MISS (incomplete) | 상류 | exchange 도달 못 함 |

## 2. ★헤드라인 (정성·방향·n=15)
1. **★escape(ⓐ-ask·진짜 "모름→유저질문") = 0/15.** *단 한 건도* tiebreaker 없는 진짜 모호가 없다. "if multiple, prefer X"(grey/white/silver/256GB/most-expensive/cheapest/battery>USB>AC)가 **도처(8·29·71·72·74·101·102)** 인데 **전부 유저-제공 ordinal 규칙 = B2-resolve(결정론)**, ASK 아님. → **벤치 설계가 유저 preference를 주어 *묻을 필요를 없앤다*. epistemic-abstain escape는 이 표면에서 사실상 빈손.**
2. **결정론이 본체**: B2-ordinal(prefer/cheapest/most-expensive·8·36·37·+tiebreaker 도처) + eligibility/arithmetic branch(L0·34·38) + stop/commit(OVER·62·+secondary 도처). = 가장 큰 몫·*적용표면*은 결정론.
3. **★학습 잔여 = faithful-formalize(ⓑ mis-ground)**: **71·72·74·101·102** = σ가 *유일정답*인데 32B가 틀림(DC·five-items·two-watches를 filter로 인코딩 실패). = thesis §2 **boundary-translator(formalize) 실패** 그 자체. **abstain 아니라 formalize.** ⓑ-op(17 verbatim·8 availability)도 formalize-fidelity.
4. **MISS 2(41·85)** = 상류 미완(recovery/capability).

→ **사전등록 트리거(§6.5) 방향 강하게 점화 쪽**: escape narrow(=0!) 확증·결정론 지배·learn 잔여=formalize(abstain 아님). **단 아래 caveat로 *확정*은 보류.**

## 3. ★정직 caveat (확정 보류 이유)
- **레버-종류 ≠ 해결(§6)**: "결정론-B2/eligibility 적용표면"이 *pass 전환*을 뜻하지 않음. [[06]] eligibility-steer(G5)=0 전례. L0/OVER/B2 전환은 **별도 실런(FLOW_DISCIPLINE류)** 몫·이 정적 카탈로그 밖.
- **★Arm-II 미실행 = ⓑ 판정 미완**: 5개 ⓑ mis-ground(71·72·74·101·102)이 **학습여지(후보 주면 32B가 고름)인지 capability-bound(후보 줘도 틀림)인지 아직 모름.** = make-or-break의 결정적 미지수. select-probe 필수.
- **수동 σ 판단**: 일부 debatable(특히 29 ⋈·multi). S2 대면검증 대상.
- **n=15 정성**: 비율 아님. "escape=0"은 방향으론 robust(15/15 tiebreaker-or-unique)하나 너비 수치는 S4(retail 전체).

## 4. ★taxonomy 구멍 (수동-먼저가 노출·설계 환류)
- **(A) ⋈-join 서브클래스 누락**: 29(garden hose="pending 주문의 그 type")·101/102(주소="다른 주문에서")=**cross-entity 관계 조인(⋈)**. σ(단일 컬렉션 filter) 아님·B2(ordinal) 아님·B1(semantic) 아님. = **별도 레버(관계-join=결정론 IF scaffold ⋈·단 모델이 cross-ref formalize)**. §3.5 표에 ⋈ 행 추가 필요.
- **(B) tiebreaker는 *항상* ordinal**(이 벤치): B1-semantic("eco-friendly")는 0/15. "prefer grey"도 color-우선순위=ordinal. → ⓐ-resolve-semantic(B1) 표면도 사실상 빈손. learn 잔여가 *formalize 단일*로 더 좁혀짐.
- **(C) ⓑ-op 분리**: verbatim-copy(17)·availability-filter-miss(8) = mis-ground(entity 틀림)과 다른 *attr/operand* 오류. 별도 태그.
- **(D) "branch 선택"(34·38·62)**: 조건부 의도("if X then... else...")의 *어느 가지*인지 = 계산/정책으로 결정(결정론)·단 의도 formalize 필요. = L0의 서브타입(eligibility/arith branch).

## 5. ★Arm-II select-probe — v1 결함 + Probe-B(raw) 교정 (2026-06-25·다른세션 리뷰)
- **v1(`escape_arm2_probe.py`)=무효(spoonfeed)**: 후보를 판별필드 *추출·라벨*("shipped to Washington, DC | 5 item(s)")해 표로 줌 → "어느 게 DC?"=라벨 컬럼 lookup(formalize 아님). 진짜 ⓑ=①criterion formalize ②전 주문 fetch ③raw 중첩 address dict서 추출 ④매칭인데 v1은 ①②③ collapse·④만 격리 → 자명한 15/15. **+ 102는 실패한 ⋈(주소출처) 아니라 watch-id(안 실패한 부분) probe = 오-probe.**
- **Probe-B(`escape_arm2_probe_v2.py`·raw)**: 후보=raw 주문 dict(중첩 address·미카운트 items)·criterion 필드 *미라벨*·102/101은 실제 ⋈(NY주소=다른주문) probe. → 모델이 formalize+extract+⋈ 직접.

| probe | 테스트 | gold | 32B (3 trial) | 판정 |
|---|---|---|---|---|
| 71·72 | DC주문(address.state 추출) | #W5270061 | 3/3 | GROUNDED |
| 74 | 5-item(items 카운트) | #W3189752 | 3/3 | GROUNDED |
| 101·102w | 2-watch(카운트) | #W4219264 | 3/3 | GROUNDED |
| **101x·102x** | **⋈ NY주소(다른주문서 추출)** | 144 Lakeview Drive | **3/3** | **GROUNDED** |

- **★Probe-B 판정**: raw·중첩추출·**⋈까지 7/7** → 리뷰어 가설("raw면 formalize 드러나 실패")은 **반증**. **disambiguation·formalize-from-raw·extract·⋈ = 32B에 *존재*(capability-bound 아님)** — v1보다 *엄밀히*.
- **★단 교훈은 남음(정직)**: Probe-B도 **단일턴·focused 격리**(풀-플로우 orchestration 부하 제거). 원궤적 ⓑ(71: 데이터 있었는데 틀림)와 차이 = *멀티스텝 부하 속 그 순간 disambiguation에 집중* 실패. ⇒ **스킬은 있음·orchestration-under-load가 실패원.** = autofetch-σ-present(결정점서 focused 후보제시)가 *그 부하를 덜어주면* 작동 스킬이 carry. **여전히 풀-플로우 전환은 미입증(=arm).**

## 5.5 ★S2 multi-label 재집계 (리뷰 반영·"결정론 지배" 정정)
카탈로그 §1이 일부 lever를 오배정 → 재집계:
- **task 8**: primary = **σ-filter-miss(formalize·availability=False 무시)** 이지 B2 아님(prefer-X는 secondary). → **formalize.**
- **29·102**: **⋈(cross-entity join)=primary** → ⋈-formalize(단일σ/B2 아님).
- **72**: gold 양 op 다 #W5270061(DC)·traj 2×items on #W7032009(wrong) → **ⓑ mis-ground**(L0는 정렬 artifact·확인됨).
- **재집계(정성·n=15)**: **formalize-ⓑ ≈ 8~9**(71·72·74·101·102·8·17·29·102 ⋈/verbatim/mis-ground) = *단독 최대* · 순수 결정론(34·38·36·37·62)= 5 · **abstain=0** · MISS=2. ⇒ **"결정론 지배" 아니라 "formalize 지배 + 결정론 공동 + abstain=0".**

## 6. ★종합 판정 (정정·Probe-B 후)
- **escape(ⓐ-ask)=0** (확고) · **formalize-ⓑ = 단독 최대 클래스**(결정론 공동·지배 아님·§5.5) · **formalize 스킬은 capability-bound 아님**(Probe-B 7/7 incl ⋈) · 실패원 = **orchestration-under-load**.
- ⇒ 그림 = thesis §2 정합 강화: **LLM의 formalize/select는 *있고 작동*(translator 본령)·scaffold가 FETCH+orchestration 부하를 덜면 carry.** "결정론이 다 닫고 LLM 미미"가 아니라 **"LLM 스킬 present·orchestration 부하가 그걸 무너뜨림 → scaffold가 부하 offload"**.
- **★트리거 = 아직 점화 보류(정정)**: v1 근거(spoonfeed)는 무효였고, Probe-B는 *스킬 존재*만 입증·**풀-플로우 전환(autofetch-σ-present가 부하 덜어 pass로?)은 미측정.** [[06]] eligibility-steer=0 전례 = 경계. **arm이 점화/기각.**
- **make-or-break(정정)**: "abstain 학습"(표면0) → **"autofetch-σ-present scaffold(orchestration 부하 offload) + 작동 formalize/select"**. 잔여 학습 = orchestration robustness(or thin). formalize 지배라 *모델 역할이 "결정론 지배"가 시사한 것보다 큼*.
- **★Probe-B가 잔여를 *정밀화*(가설 반증=결과)**: select·raw-formalize·⋈ 다 작동 → **유일 잔여 = "σ_{criterion} 결과를 결정점에 *명시 선택지로 제시*"** 단계(71: 데이터 *있었는데* 무시 → "fetch" 아니라 "제시"). ⇒ arm 메커니즘 본체 = Probe-B 제시 형식 재현(`AUTOFETCH_SIGMA_ARM_DESIGN` §1 제시-중심). "raw면 formalize 실패"는 *반증됨*(7/7).

## 7. 다음
- **(결정·task #4)** autofetch-σ arm(`AUTOFETCH_SIGMA_ARM_DESIGN_2026_06_25`) = 결정점 σ 후보제시 → gap 풀 e2e pass 전환(=lever-type→해결·orchestration 부하 덜면 작동 스킬 carry?). [[05]] 도메인-일반.
- S2 잔여: ⋈ 서브클래스(A) 정식화·taxonomy 구멍 B-D·정렬 모호. S4(retail 전체 비율).
