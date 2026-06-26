# Autofetch-σ Scaffold Arm — 설계 (2026-06-25): make-or-break 전환검증 (lever-type→해결)

> **위치**: escape-scope 진단(`ESCAPE_SCOPE_STAGE1_CATALOG_2026_06_24` §6-7)의 *확정* 실런. Arm-II가 **단일턴 SELECT=15/15 GROUNDED**를 입증 → 이 arm은 **"FETCH offload + 작동 select가 *풀 멀티턴* task를 pass로 전환하나"**를 측정. = 사전등록 트리거(§6.5)를 데이터로 확정. [[05]]·[[06]]·[[10]]·[[11]] 준수.
>
> **★[[05]] 가드-기록 (scaffold_guard 3문·정직)**: (1)A2 도메인특화 순증? = candidate-source producer-map 1개 추가(get_user_details.orders)·단 candidate_source/anchor_source는 grounding.json에 부분 실물=*최소*. (2)유동판단 동결? = **No**·select(formalize+매칭)=모델 유지(Probe-B 작동)·scaffold는 제시만. (3)도메인 행동 수행? = **★Yes**·후보 조립·제시=**autofetch류 절차 offload**(read-only·write 아님). ⇒ **Q3=yes→기본 NO·*측정으로만* 정당화.** 그래서 이건 default scaffold 변경이 *아니라* **측정 arm**(flag-gated `select_confirm`·floor 대비·GO/NO-GO). arm=이 절차-offload가 pass 전환하나의 측정기. GO=정당화·NO-GO=기각(adopt 안 함).

## 0. 핵심 가설 (진단이 세움)
- **진단 사실**: ⓑ mis-ground(71·72·74·101·102)는 capability-bound 아님 — 후보+criterion 떠먹이면 SELECT 완벽. 실패는 멀티스텝 속 **FETCH+orchestration**(후보 조립+criterion 집중)에.
- **가설(H)**: 결정점서 scaffold가 **(a)후보 컬렉션 FETCH(결정론)+(b)discriminating-attr로 명시 제시(σ-present)** 하면 → 모델 SELECT(작동)로 ⓑ가 **pass 전환**. = FETCH=offload·SELECT=모델([[10]]/FETCH_SELECT).
- **반가설(H0·정직)**: 후보 제시해도 풀-플로우서 전환 안 됨 → 실패가 candidate-presentation보다 깊음(orchestration/multi-step capability) → autofetch 가설 기각·escalate. ([[06]] eligibility-steer(G5)=0 전례 = 경계등.)

> **★Probe-B(raw·⋈)가 H를 *정밀화*(2026-06-25·다른세션 리뷰)**: 가설 "raw면 formalize-from-raw 실패 드러남"은 **반증**(7/7 GROUNDED·⋈ 포함). ⇒ select·formalize-from-raw(criterion 추출·count·⋈) **둘 다 작동(후보 제시되면)**. **유일 잔여 = σ_{criterion} 결과를 *결정점에 명시 선택지로 올리는* 단계.** 71 원실패 = get_user_details 데이터 *있었는데도* 무시 → **"fetch"가 아니라 "*제시*(explicit choice-set)"가 핵심.** mere-fetch는 또 무시될 수 있음(71). Probe-B가 통한 *이유*=그 제시. = arm 메커니즘의 본체는 (2) 제시이지 (1) fetch 아님.

## 1. 메커니즘 (write-time select-confirm gate·신규 kind·★제시-중심)
owned-entity write(modify/cancel/exchange on order/item/variant) 시도 시 scaffold 개입:
1. **FETCH(결정론·전제)**: entity-type 전체 후보 retrieve — order→user의 orders / item→order의 items / variant→product의 variants. (A2 producer-map·기존 autofetch [[06]]/§35.) *단 fetch 자체는 충분조건 아님(71).*
2. **★σ-present = 핵심(신규)**: 후보를 *discriminating attr*와 함께 **명시 선택지 리스트로 결정점에 주입** = **Probe-B가 통한 *그* 형태**(order: id+status+ship-to+items / variant: options+available). 모델이 안 모아도 scaffold가 한자리에 올려줌.
3. **select-confirm**: 모델이 자기 formalize한 criterion으로 후보서 타깃 *재확인* 후 write 커밋. criterion formalize·select = 모델 몫(Probe-B로 작동 입증·thin·translator).
- ※ **잔여의 정확한 위치(Probe-B 확정)**: select❌·raw-formalize❌·fetch△ → **"σ_{criterion} 결과를 결정점에 선택지로 *제시*"** 단계. 그래서 (2)가 본체.

## 2. A2-구동 candidate-source (도메인-일반·[[05]] keystone)
- candidate-source = A2 producer-map(entity-type→producer·field). retail: order→`get_user_details.orders`·item→`get_order_details.items`(anchor_source)·variant→`get_product_details.variants`(candidate_source·grounding.json 실물). **retail 하드코딩 0·grep if-domain=0.**
- airline A2-swap: reservation→`get_user_details.reservations` 등 producer만 교체·엔진/kind 불변. = 전이가 producer-map swap.
- ⚠️ **게이트 증식 금지([[05]] 최대 함정)**: 신규 kind=*일반* select-confirm(GateInterpreter dispatch)이지 failure-type별 retail 게이트 아님.

## 3. 측정 (arm·robust)
- **Baseline**: floor(gate0) gap 15 — 기존 fail-all-3.
- **Arm-AF**: autofetch-σ-select kind ON, gap 15 → **per-task pass 전환**(특히 ⓑ 5개).
- **Arm-AF+G14**: + 기존 G1-G4(eligibility L0·stop/commit OVER) → L0/OVER 클래스까지 결합 전환.
- **메트릭**: pass(robust nt=3·fail-all-3·gpt-4.1 user-sim) + **회귀 census**: over-ask·over-action·false-block(autofetch가 *틀린* 후보 제시/과제시로 깨는가). 결정론 신호(위반카운트·궤적 census) 우선·pass^1 점추정 보조.
- **분해**: ⓑ(71·72·74·101·102) 전환율 = H 직접검정. L0(34·38·72)·OVER(62)·ⓑ-op(8·17) = AF만으론 안 닫힐 것(별 레버) → AF+G14·operand 별도.

## 4. 판정 (트리거 확정)
- **GO(전환 확증)**: ⓑ 5개가 AF로 pass 전환(robust) + 회귀(over-ask/false-block) 미미 → **make-or-break 재정의 확정**: 결정론 autofetch-σ + 작동 select가 본체·abstain-SFT 불요. → 다음=A2-swap 전이(airline) + retail 전체 + TCO.
- **부분 GO**: 일부만 전환(orchestration 잔존) → AF가 필요조건이나 불충분·잔여=multi-step capability(thin SFT or scale).
- **NO-GO(H0)**: AF로도 ⓑ 미전환(단일턴 select는 되는데 풀-플로우 안 됨) → 실패=candidate-presentation 너머 orchestration capability → escalate. (= [[06]] G5=0의 autofetch판 재현 위험·정직 직면.)

## 5. ★정직 경계 (lever-type≠해결·G5 구별)
- 진단은 "ⓑ=FETCH-addressable 적용표면"만 봤음. **이 arm이 *해결*(pass 전환)을 처음 측정.** 전까지 "autofetch가 닫는다"는 *가설*.
- **G5 전례와 구별**: G5=*steer*(힌트 주입·"이 도구 써라")로 inert. AF=*FETCH+명시후보 제시*(힌트 아니라 select-ready 집합). Arm-II가 후자선 select 작동 입증 → AF는 G5와 *다른* 개입. 단 풀-플로우 전환은 여전히 미입증 → 이 arm이 답.
- over-action/stop은 AF가 *안* 고침(별 레버=commit gate) → AF 단독 전환율은 ⓑ-층에 한정 해석.

## 6. 구현 단계
1. **S1**: GateInterpreter에 `select_confirm` kind(FETCH producer-map + σ-present 포맷 + confirm). `t2_gate`/`gate_interpreter.py` 확장·A2 producer-map(retail.gate.json/grounding.json 재사용). `--validate` 무회귀(over-deny=0).
2. **S2 소검증(대면·무인 금지)**: gap 71로 AF arm 1 task 돌려 (a)후보 제시 포맷 sane (b)write 전 개입 (c)select 정확 — 확인.
3. **S3 실런**: Arm-AF·Arm-AF+G14 on gap 15(nt=3) → 전환율 + 회귀 census. (GPU·로컬 32B + gpt-4.1 user-sim COST GUARD.)
4. **S4**: GO면 retail 전체 + airline A2-swap 전이.
5. 산출 = `AUTOFETCH_SIGMA_RESULTS_2026_06_2x.md` + 트리거 확정/기각.

## 7. 불변
- [[05]] AF=도메인-일반 A2 producer-map·retail 하드코딩0·게이트 증식 금지 · [[11]] tau2 학습0·전이=producer-map swap · [[06]] robust(fail-all-3·다수trial)·pass^1 단독 무효 · [[30]] 리모트 ssh_run·CRLF→LF·COST GUARD(user-sim=gpt-4.1).
- **S2 소검증 전 무인 전수 금지.** over-ask/false-block 회귀 *반드시* 측정(autofetch 과제시=새 실패원).
