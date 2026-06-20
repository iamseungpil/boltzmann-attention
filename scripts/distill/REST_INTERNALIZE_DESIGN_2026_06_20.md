# ReST 내재화 설계 — 검증기-필터 self-training으로 규칙을 무망각 내재화 (2026-06-20)

> **자립 설계서**(리뷰용). 이 세션 전수실험이 좁혀온 *유일하게 남은 싼 내재화 경로*. 동기·교정:
> - **narrow 추상 SFT가 일반능력 파탄**(전수: fact_full 형식파탄·solo_* 날조 96%·#9 떠먹인 데이터) → weight 학습은 *데이터·forgetting* 통제가 관건.
> - **steering=약함**(철회논문 원인확인: agentic서 +1.6점·취약) → 가중치 무변경 레버는 한계. ⇒ **ReST = 싼 *학습* 레버**(reward/preference 0·검증기 필터).
> - 메모리=[[13-absorption-priority]](scale→**학습(무망각)**→최후 scaffold/A2)·[[12-diversity-required]](단일템플릿 SFT=역전이·다양성 필수)·[[06-NOW]].

## 0. 목표 (한 줄)
**작은 모델이 *자기가 낸 검증-정답 full-agent 궤적*으로 LoRA-SFT(+replay)해, 올바른 절차(인증→get_user_details로 order_id grounding→정확 write)를 *스캐폴드 없이* 내재화 — 일반능력 보존(무망각).**

## 1. ★핵심 통찰 = 스캐폴드로 정답 생성 → ReST로 내재화
전수실험이 준 두 사실의 결합:
- **스캐폴드(게이트+프롬프트)는 정답을 *늘린다***: 게이트(T2_PROVENANCE)가 날조를 실행 전 차단, 프롬프트(NOFAB)가 룰 명시 → base보다 *올바른 궤적* 多 생성(매트릭스서 게이트가 날조 55→33%↓ 실측).
- **그 정답 궤적을 모델에 *구우면* 스캐폴드 불요**: 게이트가 만든 올바른 행동을 모델이 *스스로* 하게 = context-distillation 사촌.
⇒ **스캐폴드 = 정답 *생성기*, ReST = 그걸 weight로 *내재화*.** 부트스트랩→자기증류→독립.

## 2. 파이프라인 (생성→필터→LoRA+replay→반복)
1. **생성(generate)**: 강한 생성기로 τ² retail e2e 다회 → 궤적. 생성기 = base+**프롬프트(NOFAB)+게이트(T2_PROVENANCE)** (정답률↑·§4) 또는 32B(0.55 pass)·temp>0로 다양성.
2. **필터(verify)**: **reward≥1(DB-match=full task 성공)** 궤적만 keep. ★검증기=DB-match(결정론·τ² 보상)지 ground_OK(소수경로) 아님. = 검증가능 신호·reward model/preference/human 0.
3. **타깃화(distill target)**: keep 궤적을 (NL, 도구스키마, *모델 tool-call 시퀀스*) SFT 쌍으로. **스캐폴드 아티팩트(게이트 deny·프롬프트) 제거** → 모델이 *스캐폴드 없이* 그 행동 재현하게(내재화). 도구명=*실 retail 도구*(#9 교정: 추상 alias 금지).
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

## 4. 부트스트랩 (낮은 base pass 문제)
- base retail pass≈0.19 → 정답 궤적 적음. ⇒ **생성기 강화로 yield↑**: (a)프롬프트+게이트(정답↑·실측) (b)32B 생성→7B 학습(*강생성기→약학습기 증류*) (c)temp/다회로 정답 모음.
- 게이트가 만든 정답(날조 차단된 궤적)을 7B에 구움 = "게이트가 가르치고 모델이 외움" → 추론 시 게이트 없이도(저비용). 단 게이트 잔존도 무방(보완).

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

## 7. GO / NO-GO
- **GO**: 7B+ReST가 **retail pass > base(0.19) 유의↑ ∧ held-out 일반능력 불변(무망각) ∧ 날조율↓**. 이상적=스캐폴드(프롬프트/게이트) *없이도* 정답(내재화 실증).
- **NO-GO**: pass 안 오르거나(정답 궤적 부족·yield 문제→생성기 강화) / 일반능력 깨짐(replay↑로도 안 되면 forgetting 본질·rank↓) / keep 궤적이 너무 적음(부트스트랩 실패→32B 증류).
- **헤드라인**: "스캐폴드(게이트)로 만든 검증-정답을 작은 모델에 *무망각 내재화* → 소형이 스캐폴드 없이 올바른 절차" = [[00-thesis]] *학습 레그*를 *데이터·forgetting 교정*으로 살림. ABox-swap 전이는 그 위.

## 8. 빌드 단계
- **S0**: 생성 파이프(base+NOFAB+gate로 retail e2e 다회·temp>0) → keep(reward≥1) 궤적 → (NL,tools,calls) SFT jsonl(실도구·스캐폴드 제거). + replay 일반데이터 확보.
- **S1**: small-rank LoRA SFT(replay비 sweep) → τ² retail pass + held-out 일반능력 → 무망각 확인.
- **S2**: ReST 반복(2-3 라운드)·정답률 수렴. 스캐폴드-drop eval(내재화).
- **S3**: 마스터 표 통합 + ABox-swap(airline) 전이·비-게이트 규칙(operand) 적용.
- 자산: `t2_run_gated`(생성)·DB-match 보상(필터)·`t2_failcensus_deep`(진단)·기존 LoRA 학습 드라이버(`build_solo_*`·replay 추가)·매트릭스/method_compare 기준선.

## 핵심 한 줄
**게이트+프롬프트로 *검증-정답 full-agent 궤적*을 싸게 생성 → 작은 LoRA+replay로 *무망각* 내재화(reward/preference 0) → 소형이 스캐폴드 없이 올바른 절차(order_id grounding·정확 write). #9 떠먹임·catastrophic forget을 *실궤적+검증기+replay*로 교정한, 이 세션이 남긴 유일한 싼 학습 경로.**
