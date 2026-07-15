# Track B 상세 설계 — F3 스키마-분류 스킬 SFT (전부 무료·리모트 GPU·2026-07-16)

> 사용자 지적 정정: **Track B는 유료 아님**. SFT=리모트 A6000·eval=로컬 vLLM·**user-sim(gpt-5.2) 안 씀 = API 0 = 무료**.
> few-shot 실험이 표적을 좁힘(§0). 입력: C99 base-eval·few-shot(dispute_reason 98% fraud mode-collapse·프롬프트 무효)·[[11]]/[[42]]/[[12]].

## 0. 표적 (few-shot 실험이 확정)
- **닫는 것 = 강한-prior 서사 enum**(dispute_reason형): 프롬프트(정의·anti-prior·few-shot 반례) 전부 무효·98% "fraud" mode-collapse. **사실-도출 enum**(dispute_category)은 few-shot로 이미 열림(55→81.7%)=**제외**.
- **스킬(도메인일반)**: "제공된 enum 스키마 정의를 읽고 NL 분류·salient prior로 안 덮기". banking엔 ABox-swap 전이(스키마=eval서만 공급·학습에 0·[[11]]).

## 1. 비용 (정정)
| 단계 | 자원 | 비용 |
|---|---|---|
| synth 생성 | 로컬/리모트 CPU | 무료 |
| LoRA SFT + DPO | 리모트 A6000 | **무료**(자체 GPU·API 0) |
| 전이 eval (bank_f3_eval) | 리모트 vLLM | **무료**(user-sim 없음) |
| (선택) tau2 e2e 최종확인 | gpt-5.2 user-sim | 유료([[09]]·make-or-break 아님) |
- ⚠️ **GPU 제약**: 두 A6000 ~44.5GB 점유(vLLM 8140/8141). 32B LoRA 학습 = vLLM 하나 정지(GPU 확보) 필요·[[30]] 조율.

## 2. 학습 데이터 = 도메인일반 스키마-분류 (벤치·banking 미학습·[[11]])
- **생성기**: 다양한 *합성 taxonomy*(banking 아님) × NL 상황 → 정답 enum. 각 taxonomy = 5~10 카테고리 + 정의.
  - 예 도메인: support-ticket 유형·product-defect·insurance-claim·HR-request·content-moderation 등 — **다도메인**([[12]] 다양성).
- **★필수 = prior-conflict 케이스**(few-shot 실험이 이게 핵심임을 실증): surface-plausible(직관·salient) ≠ 정의상 정답. 강한-prior 유발 카테고리(각 taxonomy에 "가장 흔한/직관적" 카테고리를 두고, NL이 그걸 암시하나 정의상 다른 답).
- **다양성([[12]])**: taxonomy 구조·카테고리 수·NL 표현·prior-conflict 유형 변형. 단일템플릿 금지(표면매핑 역전이).
- **재사용**: `t2_a2_concrete_gen`·`cfbsynth_v2`(합성 스캐폴드)·`t2_formalize_exec`(NL→formalize 골격) 확장.

## 3. 방법 ([[42]] 처방)
- **SFT**: (schema 정의 + NL) → 정답 enum. diverse synth. LoRA 32B(`lora_train_metatool_v3` 재사용·진행률 가시 [[30]]).
- **prior-suppression DPO/NPO**: pair (정답 enum) ≻ (prior-default enum). `cfbsynth_dpo_pairs.py` 패턴 재사용. mode-collapse의 salient-default에 페널티.
- SFT 먼저(스킬 설치) → DPO(prior 억제). 각 단계 후 eval.

## 4. 전이 검증 (무료·make-or-break)
- SFT/DPO'd 32B → **`bank_f3_eval`**(banking F3·banking 스키마 학습에 0·held-out 전이).
- **지표**: ① dispute_reason 정확도 base 35%→? (majority 39% 초과·98% fraud mode-collapse 붕괴) ② 예측분포(fraud 편중 해소) ③ dispute_category 무회귀(≥55%) ④ 미학습 banking 스키마 전이.
- **대조군**: base 32B(35%/55%)·zero/strict/few-shot(전부 35%).

## 5. 성공기준·make-or-break
- **GO**: dispute_reason > majority·fraud-편중 붕괴·미학습 banking 전이·dispute_category 무회귀. = **소형+학습 스킬이 프롬프트-불가 F3를 연다**([[41]] 헤드라인·frontier도 이 스킬 없음).
- **NO-GO**: SFT 후에도 mode-collapse 지속 or banking 전이 실패(과적합) → F3 강한-prior=진짜 경계(learn 축까지 닫힘)·명제는 결정론+사실-도출-F3(few-shot)로 유지.
- **부분**: dispute_reason 개선하나 <frontier — 경계-완화 정량.

## 6. 순서 (전부 무료 리모트)
1. **synth 생성기 v0**(다도메인 taxonomy + prior-conflict·로컬 무료·다양성 QC).
2. **base eval on synth**(held-out synth 스키마·SFT 前 baseline·prior-conflict서 mode-collapse 재현 확인).
3. **GPU 확보**(vLLM 하나 정지·[[30]]) → **LoRA SFT** → synth eval → **DPO** → synth eval.
4. **전이 eval**(bank_f3_eval·banking held-out) = make-or-break.
5. (선택·유료) tau2 banking e2e 최종.

## 7. 규율 가드
- [[11]] 벤치(synth)서만·banking 스키마 학습에 0(eval서만)·전이=ABox-swap. [[12]] 다양성 필수·단일템플릿=역전이. [[42]] SFT설치+DPO. [[30]] 진행률 가시·결과 gzip 영속·GPU 충돌금지. [[05]] 스킬=도메인일반·엔진 리터럴0. [[08]] SFT 후 예측분포 전수(mode-collapse 붕괴 실증)·집계직행 금지.
- **모트 계측**: 과-분류(prior 억제 역효과=over-correction) 계측·held-out 역전이 0 확인.
