# 수술적 no-fabrication 룰 강화 — CAA steering (+SAE 옵션) 설계 (2026-06-20)

> **자립 설계서**(리뷰용). 동기 = 전수 궤적분석(2026-06-20)이 밝힌 핵심: **narrow 추상 SFT(LoRA)가 일반 tool-use 능력을 *손상*** — fact_full=tool-call 형식 깨짐(인자누락→에러루프)·solo_*=order_id 날조(96%). ⇒ weight를 바꾸는 방법은 catastrophic forgetting 위험. **가중치 무변경·룰만 외과적 강화 = steering/SAE.**
> 재사용 자산 = `_extract_steering_vectors.py`(CAA mean-of-diff·contrast pairs→층별 residual 벡터)·`_steering_vllm_server.py`/`_gated`(추론 시 forward-hook으로 β·v 더함)·`exp_ggb_residual_steer.py`(residual steering 레퍼런스)·`reports/.../phase2_steering`(alpha_grid 선례)·withdrawn NeurIPS steering 논문. 메모리=[[13-absorption-priority]](scale→학습→최후 scaffold/A2·*steering=학습 아래 무망각 레버*)·[[06-NOW]].

## 0. 목표 (한 줄)
**order_id 날조(스키마 예시값 #W0000000 복사) 행동을, 가중치 무변경·일반능력 보존하며 추론 시 *방향벡터*로 억제**한다. = "fetch-first/provenance" 룰을 외과적으로 강화. forgetting 0.

## 0.5 ★리뷰 반영 — 위험·재프레임 (steering 부스터리즘 경계)
- **(A) ★팀 steering 논문 철회 원인 *확인 완료*(2026-06-20·FACT_BASE 정독) = steering이 *같은 agentic 영역서 약함***: MetaTool multi-tool-selection서 best steering F1 0.731→0.747 **(+1.6점뿐)**·β=-0.1서만 양성·"더 키우면 급격 붕괴"·K-side 증폭은 *음성*(Qwen 0.73→0.685·Llama→0.31 파탄). 헤드라인 못 세워 철회. ⇒ **no-fab(절차규칙·selection보다 어려움) steering도 약할 강한 사전경고.** 단 철회본=*Q-side ontology* 방법·내 제안=*residual-CAA*라 동일결론 아님(약한 prior).
- **★결론(우선순위 강등)**: no-fab steering = 저가치(약한 레버 실증 × 게이트가 이미 해결). **배포 답=게이트.** 싼 내재화 1순위 = **ReST/context-distill**(→ `REST_INTERNALIZE_DESIGN`)로 이동. steering/SAE는 ReST가 막히거나 *비-게이트 규칙*서 더 정밀 필요할 때 *작게* 재시도(residual-CAA만·기대 ±2점).
- **(B) steering은 *거친 방향*(sentiment·refusal·단일개념)엔 강하나 *절차·조건부 규칙*("값 없으면 producer 먼저 호출")엔 약할 수 있음** — 단일 선형방향이 조건부 procedural 로직을 못 잡을 위험(딥리서치 caveat·GCD=form-not-meaning 동류). ⇒ no-fab이 *clean linear direction*인지 자체가 가설.
- **(C) ★no-fab은 *이미 게이트가 결정론으로 해결*** (`T2_PROVENANCE`: 날조 id 실행 전 차단·무비용·무망각·decidable→offload thesis 정합). ⇒ **배포용 no-fab 답 = 게이트**, steering 아님. **steering의 진짜 가치 = 게이트 *불가*한 규칙**(semantic operand-selection·clarify 타이밍·judgment).
- **★재프레임**: 이 설계의 1차 산출 = "*가중치 무변경 surgical 강화가 되는가·한계가 무엇인가*"(**method 검증**)·no-fab은 *깨끗이 측정되는 testbed*. **게이트(≈perfect)가 baseline** — steering이 게이트를 *못 이기면* no-fab은 게이트로 두고 **steering은 비-게이트 규칙(§5b)으로 옮긴다.** 즉 본 실험의 진짜 질문 = "surgical-steering이 *어떤 클래스의 룰*에 *얼마나* 먹히나", no-fab 절대수치 아님.

## 1. CAA steering 메커니즘 (재사용·발명 0)
1. **contrast pairs** (날조 vs grounding 행동쌍) → 2. **mean-of-differences로 층별 residual 벡터 v_ℓ 추출**(`_extract_steering_vectors.py`) → 3. **추론 시 forward-hook으로 chosen layer residual에 β·v_ℓ 더함**(`_steering_vllm_server.py`) → 4. **β(alpha) sweep**으로 강도 조절. (Anthropic GGB식·STEER_LAYERS=[7,15,23] 출발.)

## 2. contrast pairs = 날조 vs grounding (★실 궤적서 마이닝)
우리는 이미 두 행동의 실 데이터를 가졌다(전수분석 산출):
- **negative(억제 대상)** = 날조 턴: `get_order_details({order_id:"#W0000000"})`·`find_user_id_by_email({email:"johndoe@example.com"})` (스키마 예시값 복사). solo_*·base 실패궤적서 다수.
- **positive(강화 대상)** = grounding 턴: 유저에 물음("이름·우편번호 알려주세요")·`get_user_details(user_id)`로 주문목록 fetch·실 출력서 id 복사 (base task9식).
- **★matched minimal pair 필수**(리뷰B 교정·confound 제거): mean-of-diff는 "유일 차이=fabricate vs ground"를 가정하나, 실 sim의 neg/pos 턴은 *다른 태스크·대화상태·표면*이라 벡터가 confound(early-vs-late·ask-vs-act)를 잡을 수 있음. ⇒ **같은 컨텍스트(동일 prefix)에서 *오직 다음 호출만* 날조 vs grounding으로 다른 쌍**을 구성. 방법: prefix(인증 직후·order_id 필요 시점) 고정 → neg=`get_order_details(#W0000000)`·pos=`get_user_details(user_id)`. 합성으로 minimal-pair 대량생성(도메인-일반: retail/airline/추상 여러 예시값).
- ⚠️ **도메인-일반성**: 특정 id가 아니라 *"예시값 복사 vs 출처서 fetch"* 방향. 여러 도메인·여러 placeholder로 구성해 단일-도메인 벡터 회피.
- **검증**: 추출 벡터가 *진짜 fabricate-direction*인지 = held-out minimal-pair서 neg-logprob↑/pos-logprob↑ 확인(벡터 sanity) 후 e2e.

## 3. 추출·적용·sweep (실험)
- **추출**: `_extract_steering_vectors.py --model Qwen2.5-7B --pairs nofab_pairs.json --layers all` → v_ℓ (d_model) per layer.
- **적용**: `_steering_vllm_server.py`로 serve + 층/β 지정. τ² retail e2e(`t2_run_gated`)로 평가.
- **β-sweep**: β ∈ {0(=base), 2, 4, 8, 16} × layers {[15] / [7,15,23] / all-mid}. alpha_grid 선례 재사용.
- **token 위치**: 전 위치 vs assistant-생성 위치만(GGB는 전위치). 1차=전위치.

## 4. 평가 (★일반능력 보존이 핵심)
- **룰 효과**: order_id 날조율(`t2_failcensus_deep` A%)·"not found"율 ↓.
- **★일반능력 보존 = 별도 held-out 벤치**(리뷰E 교정·같은 retail pass로 퉁치지 말 것): steering은 전 토큰에 벡터를 더해 *유창성/일반능력*을 깰 수 있음. ⇒ **steering-무관 held-out**(예: 일반 instruction-following 소세트·MMLU 소표본·또는 τ²의 *non-fab* 태스크 pass)을 β별로 측정해 **일반능력 *불변* 확인**. fab↓ ∧ held-out 불변 = 외과적 성공. held-out도 떨어지면 = blunt(β↓·layer 좁힘·gen-position-only).
- **token 위치**: 전위치(blunt) vs *assistant 생성위치만*(외과적). 일반능력 보존 위해 **생성위치-only 우선** 검토.
- ⚠️ **steering-server × 도구호출 통합 리스크**(리뷰F): `_steering_vllm_server.py`는 GGB 텍스트생성용 — `--enable-auto-tool-choice`·hermes 파서·τ² agentic 루프와 호환되는지 **S1서 먼저 smoke**(호환 안 되면 hook을 tau2_vllm_env에 포팅).
- **비교 기준선**: base(β0)·prompt(NOFAB)·**★게이트(T2_PROVENANCE·≈perfect)**·SFT(solo_*). = 마스터 표 행.

## 5. 조합 (prompt + steering)
- prompt(NOFAB·룰 *명시*) + steering(룰 *준수 강화*) = 직교·합성 기대. 셀: {prompt, steer, prompt+steer}.
- 가설: prompt는 룰을 알려주나 7B가 약하게 따름(전수분석) → steering이 *따르게* 밀어줌.

## 5b. ★steering의 진짜 타깃 = 비-게이트 규칙 (리뷰C·재프레임의 귀결)
no-fab은 게이트가 결정론 해결(§0.5C)이라 *method 검증 testbed*. **surgical-steering이 testbed서 먹히면(게이트 baseline 인지하며), 진짜 응용 = 게이트 불가한 규칙:**
- **operand-selection 의미정확**(어느 valid item·NL 의도): decidable 아님(ABSTENTION §1 의미잔여)·게이트 못 막음 → steering/SAE 후보.
- **clarify 타이밍·ask-vs-fetch judgment**(P7 reactive): RL 외 싼 대안으로 steering.
- ⇒ **2차 실험 = 비-게이트 규칙에 동일 steering 파이프 적용**(no-fab서 검증된 방법으로). no-fab 절대수치보다 *이 일반화*가 헤드라인.

## 6. SAE 옵션 (더 외과적·후속)
- **메커니즘**: "예시값 복사/placeholder 날조" feature를 SAE로 찾아 *억제*, 또는 "provenance/fetch" feature *증폭*. 단일 feature = 가장 외과적.
- **비용/리스크**: Qwen2.5-7B용 residual SAE 필요(공개 SAE 있으면 재사용·없으면 학습=큼). CAA가 안 되거나 더 정밀 필요할 때. **1차는 CAA(싸고 인프라 있음)·SAE는 2차.**

## 7. 비용·효과 (마스터 표 위치)
| 방법 | 학습비용 | forgetting | 추론비용 | 효과(예상) |
|---|---|---|---|---|
| prompt(NOFAB) | 0 | 0 | +컨텍스트 | 약(7B) |
| **CAA steering** | **추출~수분(무gradient)** | **0** | hook~0·+컨텍스트0 | ★중-강(외과적) |
| prompt+steer | 0 | 0 | +컨텍스트 | 강 |
| SAE 수술 | SAE학습 or 공개SAE | 0 | ~0 | ★★정밀(2차) |
| SFT/ReST(LoRA) | 학습 | **위험(실측 파탄)** | 0 | 망가짐(narrow) |
| ABox-swap gate | 0 | 0(weight 무관) | scaffold | 결정론 차단(보완) |

## 8. GO / NO-GO (falsifiable·게이트 baseline 인지)
- **GO (method)**: β-sweep의 어떤 (layer,β)서 **fab율 유의↓(27%→<10%) ∧ held-out 일반능력 불변**. = *가중치 무변경 surgical 강화가 된다* 실증(forgetting 0). **단 no-fab 절대수치는 게이트(≈0 fab)가 이미 달성 — steering이 게이트를 *못 이겨도* method는 valid**(게이트 불가 규칙으로 §5b 이행이 목적).
- **NO-GO (method 자체)**: 어느 (layer,β)서도 fab↓ 못 하거나(방향 없음·procedural 한계=리뷰B 적중) / fab 줄면 held-out도 같이 깨짐(blunt·외과 불가). ⇒ **steering은 이 룰 클래스에 부적합** → SAE(더 정밀) 또는 ReST(replay forgetting통제)로.
- **헤드라인(정직)**: "weight 학습이 일반능력을 깨는 곳에서(실측), *어떤 클래스의 룰*이 가중치 무변경으로 외과 강화되나 — 되는 한계와 안 되는 한계." = [[13-absorption-priority]] *scale↔학습 사이 무망각 레버*의 *경계 지도*. (steering이 만능이란 주장 아님·팀 철회논문 교훈 반영.)

## 9. 빌드 단계
- **S0 (선결·2개)**: ① **철회 steering 논문 *왜 철회됐나* 확인**(`archive_neurips2026_withdrawn/`·어느 한계서 깨졌나=리뷰A) → 같은 함정 회피. ② **matched minimal-pair `nofab_pairs.json`** 합성(동일 prefix·neg날조/pos grounding·다도메인 ~300쌍).
- **S1**: `_extract_steering_vectors.py`로 v_ℓ 추출 → 벡터 sanity(held-out minimal-pair) → `_steering_vllm_server.py` *도구호출 smoke*(§4 리스크) → β-sweep × layers, τ² retail 50태스크, fab율 + **held-out 일반능력**.
- **S2**: prompt+steer 조합 + gen-position-only. 게이트 baseline 대비.
- **S3 (★진짜 헤드라인)**: method가 서면 → **비-게이트 규칙(§5b·operand-selection·clarify)에 동일 파이프** = "어떤 룰 클래스가 무망각 외과강화되나" 경계지도.
- **S4**: 마스터 표(prompt/steer/SFT/gate/ReST × 7B/32B) + SAE는 CAA 한계 보고 결정.
- 자산: §1 steering 코드·`t2_run_gated`·`t2_failcensus_deep`·매트릭스/method_compare 기준선.

## 핵심 한 줄 (리뷰 반영)
**narrow SFT가 일반능력을 깨는 게 실측됐다(fact_full 형식파탄·solo_* 날조). no-fab *배포 답은 이미 게이트*(결정론·무망각). steering의 진짜 질문 = "*가중치 무변경 외과강화가 어떤 룰 클래스에 되나·안 되나*"(no-fab=testbed·게이트=baseline) → 되면 비-게이트 규칙(operand·clarify)으로 일반화, 안 되면 SAE/ReST. 팀 철회논문이 경고이자 출발점. = scale↔학습 사이 무망각 레버의 *경계지도*, steering 만능론 아님.**
