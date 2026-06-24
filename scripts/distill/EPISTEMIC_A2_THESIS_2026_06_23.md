# Epistemic-A2 Thesis Spine (2026-06-23) — "모름"을 외부 관계로 우회하고 그 사용을 학습한다

> 이 세션 대화 arc(사용자 통찰 연쇄)의 압축 결론. 근거: `FLOW_DISCIPLINE_RESULTS_2026_06_23`(32B gap 분해·G5=0·편차=user-sim)·`EPISTEMIC_ABSTENTION_PROBLEM_2026_06_23`·[[05]]·[[10]]·[[42]]·[[02]]·[[07]]. 딥리서치 2건(`wqidut74b` 분해·`w0r8slp20` abstention/defer)이 방법을 채울 예정.

## 0. Root 진단 — 왜 frontier도 같은 병
LLM 학습은 전부 **P(다음토큰|문맥) 형성**일 뿐, *그 분포의 신뢰도(메타)*를 직접 학습하지 않는다. 그래서 **"모른다"를 신뢰가능하게 토큰으로 만들기가 어렵다.** ★**정밀화(적대검증 후·과한 표현 교정)**: 내성이 *불가능*이 아니라 — 내성은 **분포의 직접 읽기가 아니라 proxy(유창성·확률질량 등) 기반 *추론***이고, 그 proxy가 비진단적일 때 *체계적으로 어긋나* 신뢰가능 IDK를 못 만든다. RLHF는 단정적·도움되는 답을 보상 → 환각 인센티브화. ⇒ **scale 문제 아님·모든 frontier 동일.**

> **★선행 정합 (인용·확정)**: Kalai, Nachum, Vempala, Zhang, *"Why Language Models Hallucinate"*, OpenAI, 2025 (arXiv:2509.04664). **핵심 = hallucination은 평가/채점이 "추측(guessing)"을 "기권(abstain·IDK)"보다 보상하기 때문에 지속·불가피.** 시험 비유: 0/1 채점에서 틀려도 감점 없으면 빈칸(0점 확정)보다 찍기(기대값 양수)가 *수학적으로 최적* → 모델은 "좋은 시험꾼"=허세(bluff)를 학습. 처방 = "scoreboard를 바꿔라"(틀린 자신감 감점·보정된 IDK 부분점수) + **explicit confidence targets**. ⇒ 본 §0(불가피·scale무관·목적함수/인센티브 산물)을 *독립 논증으로 직접 지지*하고, §4 abstain 커리큘럼 ≈ 논문의 "explicit confidence targets". (cf. §3 SOAR impasse = *비내성·구조적* 결정불능 감지의 더 오래된 선행 — 내성 calibration을 아예 우회.)
> - **단(정직·확정 vs 논쟁 구분)**: 논문의 처방은 "인센티브 고치면 모델이 기권을 *배울 수 있다*"는 쪽 = base 모델에 쓸 만한 calibration 신호가 *있고* post-training/RLHF가 그걸 망가뜨린다는 통설을 전제. 정밀화와 정합 — 신호가 *없는* 게 아니라 *직접접근이 아닌 추론*이라 비진단 proxy서 어긋남. **§3 외부화는 그보다 한 걸음 더** = calibration 자체가 모델 내성(proxy 추론)에 의존하는 잔여를 scaffold-계산 빈관계로 우회(논문 미다룸·우리 추가 기여).

> **★인간 평행 (적대검증·확정·딥리서치 `w974l2gpa` 21/25 confirmed)**: 인간 메타인지도 *동일 구조* — 자기 지식상태의 **직접 읽기가 아니라 단서(cue) 기반 추론**이다(Koriat accessibility 1993·cue-utilization 1997·perception 유비 2008 = 학계 합의·옛 Hart 1965 직접접근설 폐기). 단서가 비진단적이면(유창성·빈도 착시) *체계적으로 어긋나* 능력착각·과신 → **메타무지("unknown unknowns")는 병리가 아니라 무정보 단서의 *기본 출력*.** 자기-모름 인식은 *유계·기반의존*(우 aPFC/BA10·Fleming 2010 Science·2014 Brain·병변시 선택적 손상)이고 *간헐·상태의존*(TOT/고-FOK 없으면 chance). **"항상 자각하는 새 뇌구조"로 *훈련 설치*는 회의적**(Rouy 2022 사전등록 재현=훈련효과 *부재* 중등도 증거·낙관 주장들 반증). ⇒ **인간의 안정적 자각 ≈ 새 구조가 아니라 *외부화(과학적 방법·기록·동료심사)+훈련된 습관/태도*(El Kassar: 무지엔 환원불가 태도 차원).**
>   - **함의(우리 설계 정당화)**: ① escape(빈관계 외부화)는 *임시방편이 아니라 인간 인지의 모사* — 인간도 직접 읽기 대신 외부/proxy 단서로 우회. ② §4 abstain *커리큘럼*(empty→ASK *습관* 학습)이 "내성 능력 훈련"보다 옳은 길 = finding 5(특질훈련 회의)+6(습관/태도)이 지지. ③ 모델 내성을 신뢰축으로 두지 말라 = finding 4(자기지식=유계·취약·기반의존).
>   - **정직 한계**: AI↔인간 평행은 *구조적 유비*(단서추론 ≈ 분포 비직접접근)지 head-to-head 검증 아님. Dunning-Kruger·작화증(Gazzaniga·Korsakoff)·WYSIATI·Nisbett&Wilson(1977)은 이번 배치 미검증(framing only) → 별도 라운드 보강 예정. 권위본 = `EPISTEMIC_HUMAN_PARALLEL_2026_06_25.md`.

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
- **★escape 범위(정직·load-bearing)**: 이 우회는 **올바르게 formalize됐는데 빈/모호한** 실패만 잡는다. *오해결*(모델이 σ predicate를 틀리게 formalize→scaffold가 틀린 관계에 σ 계산→**비어있지 않은 틀린 1개** 반환→ask 트리거 안 됨)은 **침묵 잔여**로 escape 통과. ⇒ "모호(σ=0/>1)"와 "오해결(σ=1·틀림)"은 다르고, 관찰가능한 빈-관계로 표면화되는 건 *전자뿐*. **출구 = 명세-충실 formalize(섣불리 1개로 narrowing 금지·"조건 맞는 *모든* tuple")**면 진짜 모호가 cardinality로 표면화. 잔여 침묵 = 진짜 mis-formalize.
- formalize의 "none"은 **scaffold가 각 유한 predicate를 입력에 대조해 "매치 없음"을 판정**할 때만 외부화된다(checkable). 모델이 "none"을 *emit*하면 = §0 내성 문제가 formalize 단계로 올라온 것일 뿐(외부화 *안* 됨) → **그 경우는 외부화 안 된 잔여로 표시**(escape 적용 불가).

> **★선행 정합 (SOAR·인용·확정)**: Newell·Laird·Rosenbloom, *SOAR: An Architecture for General Intelligence* (1987); Laird, *The Soar Cognitive Architecture* (2012). **핵심 = SOAR는 결정불능을 *모델 내성*이 아니라 *아키텍처가 선호(preference)의 부재로 관찰*해 1급 사건 `impasse`로 표면화하고 자동 subgoal을 생성**(universal subgoaling). 즉 본 §3 "모름을 관찰가능한 빈-관계로 외부화"는 SOAR impasse 메커니즘의 재발견 — *구조적·비내성 결정불능 감지*를 40년 전 형식화한 선행. §0(내성 IDK 불가)을 *독립 선행으로 지지*하고, Kalai et al.의 "학습된 calibration" 노선과 대비해 **impasse=아키텍처 산물(구조적 기권)** 각도를 보탬.
>   - **동형 매핑**: tie impasse(후보 여럿·선호 부족)=σ>1 모호→ASK · no-change impasse(적용 operator 없음)=σ=0 빈→ASK · constraint-failure=precondition 위반(G-gate) · operator propose→select→apply = 모델 formalize→check→select · productions(균일 if-then)=A2 유한관계 · 결정사이클(결정론)+impasse 감지=scaffold(σ+gate). ⇒ 우리 scaffold = **SOAR 결정론 코어의 최소판**(σ+gate). [[10]] 선택기·검증기=결정론과 정합.
>   - **★우리가 소유하는 delta (= 인용해도 안 잡아먹히는 이유·load-bearing)**: ⓐ **SOAR는 production이 *옳다*고 전제**(손으로 짠 심볼 규칙)이라 본 §3 ⓑ "비어있지않은-틀림(mis-formalize)" 침묵 잔여가 *없음* — 우리 난제는 **LLM의 formalize(NL→관계)가 틀릴 수 있음**이고 그게 §6 make-or-break 첫 측정(ⓐ/ⓑ split). ⓑ **impasse 해소 방향 분담**: SOAR는 *내부* subgoal(하위공간 탐색)로 품; 우리는 "0개→모름"=*외부* ASK(정보부재라 내부탐색으로 못 만듦)·operand/ranking(B2)=*내부* 결정론 resolve(=SOAR subgoal 방향). ⇒ SOAR가 우리 A/B 분담에 impasse taxonomy를 줌(계산으로 푸는 impasse vs 물어야 하는 impasse). ⓒ **chunking ≠ 우리 SFT**(온라인 심볼 컴파일  vs 오프라인 gradient): 비유 느슨함; 단 chunking 과일반화 문헌 = 우리 ⓑ 잔여(틀린 impasse 해소를 규칙으로 굳힘)의 경고등. ⓓ **★결정론 우세 ≠ SOAR 반박·획득경로 차이**(escape-scope 진단이 결정론게이트로 기울 때 정합 검토·`ESCAPE_SCOPE_DIAGNOSTIC_LAYERS_AUG` §9): SOAR도 유능행동=recognition 지배·impasse는 프런티어 잔여(정합)이고, SOAR는 그 결정론 게이트를 *chunking으로 학습*(결정화된 학습)인데 우리는 *decidable을 손작성 scaffold+A2-swap으로 offload*([[10]] decidable→offload). 같은 종착·다른 획득(SOAR=runtime chunk·도메인재학습 / 우리=design-time author·relation-swap=TCO 효율). ⓐ-resolve(선호충분→impasse아님)=SOAR 결정절차 그 자체. *유일* divergence=잔여 학습이 LLM서 capability-bound면(§6 NO-GO-b)이나 그건 LLM 사실(SOAR도 chunk 과일반화=ⓑ 인정). ⇒ escape narrow면 헤드라인=결정론게이트+TCO([[06]])·abstain=잔여보조(§4 예고 정밀화·표류방지).
>   - **★스코프 한계([[05]]·[[03]] 드리프트 차단)**: SOAR는 **framing/선행근거 + 실패 census 정밀화(gap을 impasse 타입으로 재라벨)**로만 사용 — 결정사이클·chunking 엔진 *이식 금지*(우리 scaffold가 이미 최소판). 리뷰어 "LLM 붙인 SOAR impasse 아님?" 선제 차단 = 위 delta(LLM formalize 불확실성 + 학습된 ASK + 배포비용 각도).

## 4. 학습 대상 확정 — 무엇을 배우나 (내용 아님)
**도메인-일반 스킬 = "관계규칙 따라 formalize → check+select → 빈/모호면 ASK"** + **abstain 커리큘럼**(학습데이터에 빈/모호→ASK 케이스 *명시 포함*=‘5번째 보기 존재’를 예시로 가르침). 
- **★대칭 필수(over-ask 방지)**: "빈/모호→ASK"만 가르치면 모델은 *항상 묻기*를 학습(false-ASK=결정가능한데 물음=tau2 pass 붕괴·과거 false-block의 쌍대). 커리큘럼에 **"결정가능(σ=1)→행동, 묻지 마" 케이스를 균형있게** 포함. learning-to-defer(`w0r8slp20`)의 threshold 트레이드오프 = 이 균형의 방법.
- 왜 *학습*(프롬프트 아님): [[42]] 소형은 in-context 규칙 무시(prompt-ceiling)→학습이 설치.
- 왜 *외부신호-응답*(내성 아님): 목적함수가 내성 IDK 못 만듦→빈-관계가 외부 손잡이.
- G5 null 해소: scaffold가 가리켜도(외부) 모델이 *쓰도록 학습* 안 됨 → **곱(scaffold×A2-사용학습)** 이어야. G5는 첫 항만 켠 것.

## 5. 기여 (아키텍처·크기무관 + 소형=배포비용)
- **기여의 본체 = *아키텍처*(크기무관)**: 모름을 외부 빈-관계로 표면화 + 그 사용을 학습(내성 우회). frontier도 *같은 관계-scaffold + 같은 커리큘럼*이면 grounded-abstain을 함 → "frontier가 *못* 함"은 과주장(깨짐). 기여 = 이 **factoring**이지 소형 고유 해자 아님.
- **소형 각도 = 그 아키텍처를 *배포하는 비용*** → [[06]] TCO 헤드라인으로 회귀(~23×). 이 메커니즘 = **싼 다리를 *더 싸고* + *더 신뢰가능(grounded-abstain)*하게** 만드는 것(헤드라인 비선결·싼 다리 개선).
- 포지셔닝: [[41]] "소형>큰" 아님 = *systematization + (아키텍처)grounded-abstain + 경계지도*. **§6 미실증 위에 co-헤드라인 금지**(measured 후 격상).

## 6. ★중심 실험 (make-or-break)
**A2-규칙-사용 학습**: 벤치(SOP/TB/CFB/Synth)서 "유한 관계규칙 제시→formalize/check/select·빈→ASK" 궤적 SFT(도메인-일반·내용X·abstain 케이스 포함·**결정가능→행동 대칭케이스 포함**) → {base vs A2-trained} × scaffold(A2 집행) on tau2(**A2-swap·재학습0**). 검정: ①G5서 0이던 게 학습 후 *pass 전환* ②새 도메인 A2-swap 전이 ③모름-커리큘럼이 over-action↓·loop↓·wrong-order→ask↑ **④over-ask/false-defer rate(결정가능한데 물음)=대칭 비용**.
- **★선행 진단(기존 데이터·GPU0·escape 범위 측정)**: gap task를 formalize→σ로 돌려 5개 실패가 **ⓐ빈/모호 관계(escape가 잡음) vs ⓑ비어있지않은-틀림(오해결·escape 통과)**로 분류. 대부분 ⓑ면 escape는 좁음 → 명세-충실 formalize(§3) 비중이 결정적. = make-or-break의 *첫* 측정. **(이 ⓐ/ⓑ split = SOAR가 *안 풀어도 됐던* 영역: SOAR는 production이 옳다고 전제했기에 mis-formalize=ⓑ가 없음 → 우리 고유 기여 지점. §3 SOAR 블록 delta-ⓐ.)** 실패를 impasse 타입(tie/no-change/constraint)으로 재라벨하면 분류가 더 원칙적.
- 정직 NO-GO: (a) 5개가 유한관계로 깔끔 표현 안 되거나 (b) "관계-사용/formalize"가 그 크기서 capability-bound(특히 mis-formalize=ⓑ가 학습으로 안 줄거나)면 → 그게 진짜 경계(escalate/scale).
- 측정 주의: user-sim 편차~0.11 → gap=fail-all-3 같은 robust 신호 + 결정론 신호(위반카운트·궤적 census) + 다수trial. pass^1 점추정 단독 무효.

## 7. 분담 한 줄 (최종)
**A2=유한관계(내용만 도메인) · scaffold=관계 집행(빈결과 판정) · 모델=관계 formalize+check+select+빈→ASK(학습·전이) · 모름=빈관계에 grounding해 *외부신호 응답*으로 학습(내성 아님·abstain 커리큘럼·대칭 결정가능→행동).**
> ★범위(§3): escape는 *빈/모호로 표면화되는* 모름만 잡음. 오해결(σ=1·틀림)은 명세-충실 formalize가 닫아야 할 침묵 잔여 = make-or-break 첫 측정(ⓐ/ⓑ split).
