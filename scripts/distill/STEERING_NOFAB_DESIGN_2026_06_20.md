# 수술적 no-fabrication 룰 강화 — CAA steering (+SAE 옵션) 설계 (2026-06-20)

> **자립 설계서**(리뷰용). 동기 = 전수 궤적분석(2026-06-20)이 밝힌 핵심: **narrow 추상 SFT(LoRA)가 일반 tool-use 능력을 *손상*** — fact_full=tool-call 형식 깨짐(인자누락→에러루프)·solo_*=order_id 날조(96%). ⇒ weight를 바꾸는 방법은 catastrophic forgetting 위험. **가중치 무변경·룰만 외과적 강화 = steering/SAE.**
> 재사용 자산 = `_extract_steering_vectors.py`(CAA mean-of-diff·contrast pairs→층별 residual 벡터)·`_steering_vllm_server.py`/`_gated`(추론 시 forward-hook으로 β·v 더함)·`exp_ggb_residual_steer.py`(residual steering 레퍼런스)·`reports/.../phase2_steering`(alpha_grid 선례)·withdrawn NeurIPS steering 논문. 메모리=[[13-absorption-priority]](scale→학습→최후 scaffold/A2·*steering=학습 아래 무망각 레버*)·[[06-NOW]].

## 0. 목표 (한 줄)
**order_id 날조(스키마 예시값 #W0000000 복사) 행동을, 가중치 무변경·일반능력 보존하며 추론 시 *방향벡터*로 억제**한다. = "fetch-first/provenance" 룰을 외과적으로 강화. forgetting 0.

## 1. CAA steering 메커니즘 (재사용·발명 0)
1. **contrast pairs** (날조 vs grounding 행동쌍) → 2. **mean-of-differences로 층별 residual 벡터 v_ℓ 추출**(`_extract_steering_vectors.py`) → 3. **추론 시 forward-hook으로 chosen layer residual에 β·v_ℓ 더함**(`_steering_vllm_server.py`) → 4. **β(alpha) sweep**으로 강도 조절. (Anthropic GGB식·STEER_LAYERS=[7,15,23] 출발.)

## 2. contrast pairs = 날조 vs grounding (★실 궤적서 마이닝)
우리는 이미 두 행동의 실 데이터를 가졌다(전수분석 산출):
- **negative(억제 대상)** = 날조 턴: `get_order_details({order_id:"#W0000000"})`·`find_user_id_by_email({email:"johndoe@example.com"})` (스키마 예시값 복사). solo_*·base 실패궤적서 다수.
- **positive(강화 대상)** = grounding 턴: 유저에 물음("이름·우편번호 알려주세요")·`get_user_details(user_id)`로 주문목록 fetch·실 출력서 id 복사 (base task9식).
- **마이닝**: 밤샘/매트릭스 sim에서 (a)"not found" 직전 assistant 날조 호출 = neg, (b)get_user_details→실 order_id 사용 = pos. 같은 *직전 컨텍스트*서 pos/neg 추출 = 깨끗한 대조. 목표 ~200-500쌍(도메인-일반 위해 retail+airline+추상 섞기).
- ⚠️ **도메인-일반성**: 특정 id(#W0000000)가 아니라 *"예시값 복사 vs 출처서 fetch"* 방향을 잡아야 전이. contrast를 *행동 패턴*으로(여러 도메인·여러 예시값) 구성.

## 3. 추출·적용·sweep (실험)
- **추출**: `_extract_steering_vectors.py --model Qwen2.5-7B --pairs nofab_pairs.json --layers all` → v_ℓ (d_model) per layer.
- **적용**: `_steering_vllm_server.py`로 serve + 층/β 지정. τ² retail e2e(`t2_run_gated`)로 평가.
- **β-sweep**: β ∈ {0(=base), 2, 4, 8, 16} × layers {[15] / [7,15,23] / all-mid}. alpha_grid 선례 재사용.
- **token 위치**: 전 위치 vs assistant-생성 위치만(GGB는 전위치). 1차=전위치.

## 4. 평가 (★일반능력 보존이 핵심)
- **룰 효과**: order_id 날조율(`t2_failcensus_deep` A%)·"not found"율 ↓.
- **★일반능력 보존**(steering이 blunt하면 다 깨짐): pass^1 + *날조 아닌* 실패모드(B operand·format) 불변 확인. **steering 후 pass가 *오르거나 유지*하며 fab만 ↓ = 성공**. pass가 떨어지면 = 벡터가 너무 거침(β↓ 또는 layer 좁힘).
- **비교 기준선**: base(β0)·prompt-only(NOFAB)·SFT(solo_*·매트릭스). = 마스터 표에 행 추가.

## 5. 조합 (prompt + steering)
- prompt(NOFAB·룰 *명시*) + steering(룰 *준수 강화*) = 직교·합성 기대. 셀: {prompt, steer, prompt+steer}.
- 가설: prompt는 룰을 알려주나 7B가 약하게 따름(전수분석) → steering이 *따르게* 밀어줌.

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

## 8. GO / NO-GO (falsifiable)
- **GO**: β-sweep의 어떤 (layer,β)서 **fab율 유의↓(예 27%→<10%) ∧ pass 유지/상승 ∧ 비-fab 실패모드 불변**. = 룰만 외과적 강화 실증·forgetting 0.
- **NO-GO**: fab는 주나 pass도 같이 떨어짐(벡터 거침·일반능력 손상) / 또는 fab 안 줌(방향 약함→pairs 재구성). 
- **헤드라인**: "weight 학습이 일반능력을 깨는 곳에서, steering이 룰만 외과적으로 강화(forgetting 0)" = 비용모델 [[13-absorption-priority]]의 *scale↔학습 사이 무망각 레버* 실증.

## 9. 빌드 단계
- **S0**: contrast pairs 마이닝(sim서 neg=날조턴·pos=grounding턴) → `nofab_pairs.json` (~300쌍·다도메인).
- **S1**: `_extract_steering_vectors.py`로 v_ℓ 추출 → `_steering_vllm_server.py` serve → β-sweep × layers, τ² retail 50태스크, fab율+pass.
- **S2**: prompt+steer 조합 + 일반능력 보존 확인(비-fab 실패 불변).
- **S3**: 마스터 표 통합(prompt/steer/SFT/gate/ReST × 7B/32B). + SAE는 CAA 결과 보고 결정.
- 자산: 위 §1 steering 코드·`t2_run_gated`·`t2_failcensus_deep`·매트릭스 기준선.

## 핵심 한 줄
**narrow SFT가 일반능력을 깨는 게 실측됐으니(fact_full 형식파탄·solo_* 날조), 룰은 *가중치 무변경 steering(CAA, 인프라 보유)*으로 외과적 강화 — fab만 ↓·pass 보존·forgetting 0. 안 되면 SAE feature 수술. = scale↔학습 사이의 무망각 레버.**
