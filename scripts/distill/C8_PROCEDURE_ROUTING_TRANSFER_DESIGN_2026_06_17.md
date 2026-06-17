# C8 — 절차 생성원 어휘를 TBox로 내재화 → 어휘·스키마 재등방화 held-out 전이 (설계) — 2026-06-17

> 상위 = `NL_PROCEDURE_OFFLOAD_THEORY_2026_06_17.md §6·§7e·§7f` · 직전 실측 = `ma/M_A_RESULTS §15`(comparative 진단·gloss 회복). 불변 = [[feedback-thesis-tbox-transfer-direction]]·[[feedback-nl-formalize-llm-selection-deterministic]]·[[feedback-no-fundamental-claims-from-convenience-data]].

## 0. 사용자 질문이 이 실험이다
> "절차와 관련된 상식 용어(연산자)는 미리 학습으로 **TBox에 고정**하고, 도메인 용어집은 **ABox로 swap**하는 게 가능한가?"

이걸 반증가능 시험으로 옮김: **절차 생성원 어휘(argmax/argmin/rank/filter/comparative 라우팅)를 등방화 추상 데이터로 *gloss 없이* SFT(TBox 내재화) → 어휘·스키마가 *전부 다른* held-out 등방화 셋에서 gloss 없이 라우팅되나.**
- **전이 → 절차어휘가 진짜 TBox로 고정됨**(소형서·도메인불변) = 사용자 가설 입증·§6 예측 (c) 양성.
- **무전이 → "내재화"는 in-context 보간이었음**(gloss 떠먹일 때만 됨) = 무엇이 고정가능인지 재정의.

## 1. 왜 gloss를 학습 prompt에서 빼나 (핵심 통제)
§15 실측: gloss(연산자 어휘 정의)를 *in-context*로 주면 comparative 0.00→1.00·N-불변. **그건 시험의 *상한*이지 내재화가 아니다** — 프롬프트로 떠먹인 라우팅. C8의 본질 = 이 라우팅을 *가중치로* 옮겨 **gloss 없는** held-out서도 되나. 따라서 **학습·평가 prompt 모두 gloss 제거**(v1 plain OP_SPEC). 모델이 NL "just greater than item <id>" → `comparative` 매핑을 *데이터로* 배워야 한다.

## 2. arm (4)
| arm | 학습 | 평가 prompt | 평가셋 | 예측·역할 |
|---|---|---|---|---|
| **S0** base·무학습 | — | gloss 無 | held-out iso | comparative recog ≈ **0**(filter 붕괴)·하한·= §15 v1 |
| **S1** base·무학습 | — | gloss **有** | held-out iso | comparative recog ≈ **1.00**·상한(내재화 목표치)·= §15 v2 |
| **★S2** trained | gloss 無 SFT | gloss 無 | **held-out iso**(새 어휘·스키마) | **C8 측정. S1(상한) 닿으면 양성·S0(0) 머물면 무전이** |
| S3 trained | gloss 無 SFT | gloss 無 | **train iso**(in-dist) | 학습성공 sanity. S3 높고 S2 낮으면 = 표면결합(M-σ 음성 재현) |

- **판정축 = comparative recognition**(S0=0 → S1=1.00 양극단이 가장 깨끗). 부수 = 전 5 op recognition·engine-acc·per-op·N-sweep.
- **핵심 대조 S2 vs S3**: in-dist만 오르고 held-out 무전이 = 표면결합(어휘 암기). **S2가 사활.**

## 3. 데이터 (등방화 분리 = held-out '도메인')
- 도구 = `synth_depth.py`(iso=1). **학습 = seed 0·평가 held-out = seed 777**(겹침 0·attr 토큰·스키마 전부 다름). in-dist sanity = seed 0 별도 holdout split.
- 전 5 op 균등·N 혼합({5,10,20,50} 섞어 B-budget 무관 라우팅 학습)·n_train ≥ 4000(op당 ~800).
- **포맷(chat·`lora_train_chat_toolcall` 소비)**: system="Output ONLY JSON." / user= `arm_B` prompt(**gloss 無 OP_SPEC** + NL + attrs) / assistant= gold `op_ir` JSON. assistant-only loss.
- 등방화가 '도메인 swap'의 합성 대응: 새 attr 토큰·스키마 = 새 도메인 어휘. 절차어휘(op 라벨)만 불변 → 그것만 전이되면 TBox 격리 성공.

## 4. 측정·판정 (사전등록)
- primary: **held-out comparative recognition**(S2). 보조: 전 op recog·engine-acc·N별.
- 양성(TBox 고정): S2 comparative recog ≥ 0.8(S1 상한의 80%↑)·전 op held-out recog ≈ S1.
- 음성(무전이): S2 ≈ S0(comparative ~0)·**단 S3 in-dist는 높음** → 어휘-표면 암기·절차 미내재화.
- 부분: 일부 op만 전이 → *어떤 절차가 내재화가능*인지 지도(가해 레벨 §4와 상관: L1 filter·L2 argmax/min 먼저 전이? comparative(관계참조) 마지막?).
- ⚠️ **정직 경계**([[feedback-no-fundamental-claims-from-convenience-data]]): 이 시험은 *순수 절차어휘*(attr이 이미 등방 토큰). 실도메인 "더 밝은"→order축 매핑(ABox 지식)은 *2차 시험*(실벤치 swap). 1차는 절차어휘 전이만 격리 — "도메인 전이 일반" 단정 금지·범위=합성 등방.

## 5. 구현 단계
1. `c8_build_sft.py`: synth_depth(seed0) → chat SFT JSONL(gloss 無 prompt). + held-out(seed777)·in-dist-holdout 평가셋 jsonl.
2. `depth_eval.py`에 `--gloss {0,1}` 토글(현 v2 spec=gloss有 / v1 plain=gloss無) — S0/S1/S2/S3 한 도구로.
3. `lora_train_chat_toolcall.py`로 7B LoRA SFT(seed0 train). GPU0(woori)·arm C(coworker 32/72B)와 분리.
4. eval 4 arm → `M_A_RESULTS §16`(C8 결과) 박제 + 이론 §7f 닫기(양성=내재화 입증·음성=재정의).

## 6. 위치
thesis 사활(§7f "LLM의 추상화 내재화가 진짜냐 표면보간이냐"). 양성이면 = **소형 on-prem LLM이 절차어휘를 TBox로 들고 도메인만 ABox swap** = 주권-leg 분담 설계의 직접 증거. 음성이어도 진단적(무엇이 고정가능인지·v4-v7 무전이의 이론적 자리매김).
