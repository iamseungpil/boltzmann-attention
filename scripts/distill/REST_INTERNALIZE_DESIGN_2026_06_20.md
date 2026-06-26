# ReST 내재화 설계 — 검증기-필터 self-training으로 *도메인-일반 discipline*을 forgetting-통제 내재화 (2026-06-20·리뷰 반영)

> **자립 설계서**(리뷰용). 이 세션이 좁혀온 싼 *학습* 레버. 동기·교정:
> - **narrow 추상 SFT가 full-agent서 파탄**(전수: fact_full 형식·solo_* 날조 96%·#9 떠먹인 데이터). ⚠️단 *간섭/lr 미격리*라 "narrow-data가 THE 원인"은 단정 금지(위험4) → weight 학습은 *데이터·forgetting* 통제가 관건.
> - **steering=약함**(철회 원인확인: agentic +1.6점). ⇒ **ReST = 싼 학습 레버**(reward/preference 0·검증기 필터).
> - 메모리=[[13-absorption-priority]](scale→학습→최후 scaffold/A2)·[[12-diversity-required]]·[[05-fixed-vs-variable]](고정=LoRA+scaffold·변경=A2)·[[00-thesis]](decidable→offload)·[[06-NOW]].

## 0. 목표 (한 줄·리뷰1 교정)
**작은 모델이 검증-정답 궤적으로 LoRA-SFT(+replay)해, *도메인-일반 discipline*(A1 provenance-copy·A2 producer-호출·순서)을 내재화 — *concrete-offload(resolve/gate)는 scaffold에 유지*. forgetting은 통제(보장 아님·§5 측정).**

## 0.5 ★★A2-min 정합 (리뷰1·최우선·이게 모순↔상보를 가름)
"scaffold 없이"는 [[05-fixed-vs-variable]](고정=scaffold·decidable→offload)와 *정면충돌*·잘못된 framing. 정정:
- **ReST가 내재화 = *discipline*(학습가능 flow·도메인-일반)**: 값 없으면 *producer 도구를 부른다*(A2)·*그 출력서 복사*(A1 무날조)·*인증 먼저*(순서). = "어느 타입 선행이 필요한가"의 SHAPE. order_id를 만들 때 *get_user_details를 부르는 행동* = discipline(학습).
- **scaffold에 유지 = *concrete-offload*(decidable)**: resolve argmax/select·gate 평가(G1-G4)·provenance 검사(값∈{user,tool}). = "정확히 어느 값/허용되나"(계산). 이건 weight로 굽지 *않음*(00-thesis 위반).
- ⇒ "order_id grounding 내재화" = *producer 호출+복사 discipline* 학습 ≠ *값 계산* offload. **ReST·A2-min·decidable→offload가 한 그림**: ReST가 discipline(고정 LoRA의 일부)을 강화·resolve/gate는 그대로 scaffold. "scaffold 제거"가 아니라 "discipline은 weight·concrete는 scaffold" 경계를 *흐리지 않게* 학습.
- **금지**: ReST가 resolve/gate가 하던 decidable을 weight로 되돌리면 = A2-min 무효·thesis 자기모순. discipline-only 타깃 엄수.

## 1. ★핵심 통찰 = 스캐폴드(프롬프트/게이트)로 정답 생성 → ReST로 *discipline* 내재화
- **스캐폴드(게이트+프롬프트)는 정답을 *늘린다***: 게이트(T2_PROVENANCE)가 날조 차단·프롬프트(NOFAB)가 룰 명시 → base보다 *discipline-올바른 궤적* 多(매트릭스 게이트 날조 55→33%↓ 실측).
- **그 정답 궤적으로 *discipline*을 구움**: 모델이 *producer 부르고 출력서 복사*하는 flow(A1/A2)를 스스로 하게 = forgetting-통제 context-distillation.
⇒ **프롬프트/게이트(=정답 *생성 보조*) → ReST가 discipline을 weight로.** 단 ★resolve/gate의 *concrete-offload는 추론 시 유지*(§0.5)·여기서 "독립"=프롬프트 없이도 discipline 발현(게이트는 soundness 안전망으로 남김).

## 2. 파이프라인 (생성→필터→LoRA+replay→반복)
1. **생성(generate)**: 강한 생성기로 τ² retail e2e 다회 → 궤적. 생성기 = base+**프롬프트(NOFAB)+게이트(T2_PROVENANCE)** (정답률↑·§4) 또는 32B(0.55 pass)·temp>0로 다양성.
2. **필터(verify)**: **reward≥1(DB-match=full task 성공)** 궤적만 keep. ★검증기=DB-match(결정론·τ² 보상)지 ground_OK(소수경로) 아님. = 검증가능 신호·reward model/preference/human 0.
3. **타깃화(distill target)**: keep 궤적을 (NL, 도구스키마, *모델 tool-call 시퀀스*) SFT 쌍으로. **프롬프트(NOFAB)를 입력서 제거**(프롬프트 없이 discipline 발현=내재화). ★단 *resolve/gate concrete-offload는 유지*(§0.5)·타깃은 *discipline 시퀀스*(producer 호출·복사·순서)지 concrete 값 계산 아님. 도구명=*실 retail 도구*(#9 교정: 추상 alias 금지).
4. **학습(SFT·LoRA)**: 작은 rank LoRA로 keep 궤적 학습 + **replay(§5)**.
5. **반복(ReST)**: 개선 모델로 다시 생성→필터→학습 (K 라운드·정답률 수렴까지).

## 3. ★세션 교정 반영 (왜 이번엔 안 깨지나)
| 과거 실패 | ReST 교정 |
|---|---|
| #9 떠먹인 추상(color/size 0줄) | **실 raw 카탈로그·실 도구·full-agent 궤적** |
| solo_* 날조 96%(catastrophic forget) | **모델 자기 정답**(on-distribution)·replay·small LoRA |
| fact_full 형식파탄 | keep=*형식 OK인 정답만*(검증기가 malformed 자동 배제) |
| 단일템플릿 역전이([[12]]) | temp>0·다도메인·다궤적 = 다양성 |
| ground_OK=소수경로 | 검증기=**DB-match(전체 task)** |

## 4. ★부트스트랩 + coverage 선택편향 (리뷰2·crux·미해결 위험)
- base retail pass≈0.19 → 정답궤적 적음. yield↑: 프롬프트+게이트·temp/다회.
- **★위험(crux)**: keep=reward≥1(~19%)이 *discipline 불필요한 쉬운 태스크*에 쏠릴 수 있음(선택편향). **discipline이 진짜 필요한 하드 태스크(다홉·모호)는 base+gate도 여전히 날조(33%) → 거기 정답궤적이 거의 없어 ReST가 핵심을 못 배움**(데이터 부재). 게이트가 yield 55→33%↑해도, 남은 33%가 곧 discipline-critical이면 못 가르침. **= ReST의 진짜 관문.**
- **32B-증류는 *주장 변경*(가볍게 두지 말 것)**: 32B 생성→7B 학습 = self-improvement 아니라 *distillation(32B 교사)*·7B는 32B(0.55) 상한·on-prem서 32B도 필요 → "소형 자기개선" 헤드라인 약화(sovereignty 피치↓). 쓰면 *명시*. (단 proactive 우려는 약함: SFT가 request→정답시퀀스 배우면 deny 없이 proactive 학습됨·문제는 coverage지 proactive 아님.)

## 4b. ★사전등록 진단 (위험2 falsify)
- **(a) scaffold-drop eval**(§7): 학습 후 *게이트 빼고* 평가 → 하드 태스크서 날조 *복귀*하면 = discipline 미내재화(쉬운 태스크만 배움). discipline 내재화의 진짜 시험.
- **(b) coverage 진단**: keep 궤적이 *discipline-critical 태스크*(다홉·모호·gate가 막은)를 대표하나 vs 쉬운 쪽 쏠림 — 태스크 난이도×keep율 분포. 쏠리면 = 하드태스크 정답 합성/증류 필요.

## 5. ★forgetting 통제 (핵심·과거 파탄의 직접 해소)
- **replay**: keep(task) : 일반데이터(원 instruct/일반 tool-use) 혼합비 sweep(예 1:1~1:4). narrow-only가 fact_full/solo_* 파탄 원인.
- **small rank LoRA**(r8-16)·**early-stop**·**per-iter held-out 일반능력 eval**(steering §4와 동일 held-out) → 라운드마다 일반능력 *불변* 확인, 깨지면 replay↑/rank↓/stop.
- 목표 = task pass↑ ∧ held-out 일반능력 *불변*. (forgetting이 보이면 즉시 비율 조정.)

## 6. 비용·효과 (마스터 표 행)
| 방법 | 학습 | reward/pref/label | forgetting | 효과 |
|---|---|---|---|---|
| **ReST(검증기 필터·LoRA+replay)** | LoRA×K라운드(싸) | **0**(DB-match 결정론) | **통제됨**(replay) | ★내재화·스캐폴드 독립 |
| SFT(떠먹인) | LoRA | 0 | 파탄(실측) | 망가짐 |
| DPO/RLVR | RL루프 | preference/reward model(비쌈) | 위험 | — |
| steering | 0 | — | 0 | 약(철회실증) |
| gate(ABox) | 0 | — | 0 | 결정론·보완 |

## 7. GO / NO-GO (사전등록·리뷰2/4)
- **GO**: 7B+ReST가 retail pass > base 유의↑ ∧ **held-out 일반능력 *불변*(forgetting 통제 측정·보장 아님)** ∧ **★scaffold-drop(게이트 제거)서 하드태스크 날조 *안* 복귀**(=discipline 내재화·§4b-a) ∧ coverage가 discipline-critical 대표(§4b-b).
- **NO-GO**: pass 안 오름(yield/coverage·하드태스크 정답부재→증류) / 일반능력 깨짐(replay↑로도 안 되면 rank↓·본질이면 forgetting 한계) / scaffold-drop서 날조 복귀(=쉬운 태스크만 배움·discipline 미내재화=위험2 적중).
- ⚠️ "무망각"=*목표·per-iter 측정*이지 보장 아님(ReST=weight-change SFT·replay가 줄이지 제거 아님·위험4).

## 7.5 ★positioning (리뷰3·기여 오독 금지)
- **ReST 내재화 = known 메커니즘**(STaR/ReST·검증기 self-training). "소형이 스캐폴드 없이 retail 정답"은 ToolLLM/ToolOrchestra가 하는 것 = **novelty 아님**.
- **★기여 = 내재화된 게 *도메인-일반 discipline*이고 *ABox-swap 전이*(S3 airline)**. 헤드라인 = **전이**, "scaffold-독립 retail" 아님. ReST는 [[13-absorption-priority]] 학습 레버(도구)·기여는 그 위 전이.

## 8. 빌드 단계
- **S0**: 생성 파이프(base+NOFAB+gate retail e2e 다회·temp>0) → keep(reward≥1) → (NL,tools,calls) SFT jsonl(실도구·프롬프트만 제거·resolve/gate 유지) + **coverage 진단(§4b-b)** + replay 일반데이터.
- **S1**: small-rank LoRA SFT(replay비 sweep) → retail pass + held-out 일반능력(*forgetting 통제* 확인) + **scaffold-drop eval(§4b-a·discipline 내재화 시험)**.
- **S2**: ReST 반복(2-3 라운드)·정답률 수렴.
- **S3 (★헤드라인=전이·리뷰3)**: discipline 내재화되면 → **ABox-swap airline 전이**(재학습0)·비-게이트 규칙(operand) 적용. *이게 기여*(scaffold-독립 아님).
- 자산: `t2_run_gated`(생성)·DB-match 보상(필터)·`t2_failcensus_deep`(진단)·기존 LoRA 학습 드라이버(`build_solo_*`·replay 추가)·매트릭스/method_compare 기준선.

## 핵심 한 줄 (리뷰 반영)
**게이트+프롬프트로 검증-정답 궤적을 싸게 생성 → 작은 LoRA+replay로 *도메인-일반 discipline*(producer 호출·복사·순서)을 forgetting-통제 내재화(reward/pref 0) — *concrete-offload(resolve/gate)는 scaffold에 유지*(A2-min 상보·decidable→offload 불변). 기여=novelty가 *전이*(ABox-swap·S3)지 scaffold-독립 아님. 관문=하드(discipline-critical) 태스크 coverage·scaffold-drop서 날조 안 복귀.**
