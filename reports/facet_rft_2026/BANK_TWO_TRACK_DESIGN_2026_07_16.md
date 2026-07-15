# 2-트랙 설계 — 결정론 배선(Track A) + F3 스키마-분류 스킬(Track B) (2026-07-16)

> 사용자 지시: 두 방향 다 설계서 반영.
> 입력: C93(dispute-proxy 발각)·C94(per-step 다층·100% user_stop)·C95(frontier 극복 model-invariant·F3 경계)·C96((a)(b)(c)).
> 종합 진단: frontier 실패 극복 = **결정론 28~49%(Track A) + F3 의미참조(Track B)**. 두 트랙이 상보(전자=under-action/discovery, 후자=semantic reference).
> 규율: [[05]] 엔진일반·ABox만 · [[11]] 벤치서 학습·타깃 금지 · [[03]] 설계먼저 · [[09]] 유료 게이트 · [[42]] prompt-ceiling→train · [[16]] scaffold LOCK.

## 0. 왜 두 트랙 (진단 요약)
| 잔여 | % (관측가능) | 성격 | 트랙 |
|---|---|---|---|
| under-action/discovery/compute | ~28~49% | **결정론** (coverage/FIND/COMPUTE/GET-⋈) | **A (배선)** |
| F3 의미참조 (상황→정책 enum) | ~26% | **의미매핑** (NL→taxonomy·prior-override) | **B (스킬 학습)** |
| 선택 enum·data-부재 | ~6% | ASK (user-원천) | A (ASK 연산) |
| pure-DB blind | ~19% | 오프라인 관측 밖 | (측정=live/DB-replay) |
- **핵심 프레임(사용자)**: 지식(taxonomy)=ABox(검색/선언·A2류·**학습 금지**) vs 스킬(스키마-대령→분류·prior 안 덮기)=TBox(**벤치서 한 번 학습·ABox-swap 전이**). LoRA/가중치는 스킬 전용(도메인-일반)·팩트는 retrieval.

---

## 1. Track A — 결정론 per-step DAG 컨트롤러 라이브 배선 (28~49% 실현)
### 1.1 목표
C96 DAG 컨트롤러(`bank_eplan_controller.py --dag`·오프라인 27.8~49%)를 **라이브 in-situ서 실현**. 지배 레버 = under-action(종료 100% user_stop) → **H_min 강제열거+coverage-track**.

### 1.2 배선 (t2_eplan_patch 진화·`_note_eplan` 3한계 해소)
| 한계 | 현상 | 해소 (생성-레벨·write강제금지·설계 §2 REPLAY_SAFE) |
|---|---|---|
| (a) 비-JSON 도구출력 | 포맷문자열→`_extract_entity_ids` ∅ | 포맷-파서(txn/account/card id 정규식·`bank_reach_forensic` TXN 패턴) → listed 채움 |
| (b) per-entity reader 귀속 | `get_*_transactions`=user_id키·디스패처내부 | eplan spec 2단(account 열거→per-account 상세)·entity_key 갱신 훅 |
| (c) confirm 게이트 부재 | `_confirm_write_tools=∅`→deny 비발화 | write 도구집합=gold DAG 파생·H_min 미충족 시 **생성-레벨 리마인더**(히스토리 커밋 금지·상한 1회+step-budget) |
- **개입 = 작업버퍼만** (L1/L2 deny·CP5 리마인더)·**write 강제 금지**([[14]]). read/discovery만 강제.
- inner: COMPUTE(ABox 규칙·liability/amount_difference 확증분)·GET-⋈(decidable).

### 1.3 검증·게이트
- **무료**: 배선 후 오프라인 스모크 — 컨트롤러가 궤적 replay서 ALL-CLOSED 27.8% 재현·라이브 발화 마커(stderr) 확인.
- **유료 [[09]]**: 로컬 banking 도메인 부재 → 라이브=리모트/유료. 승인+scope(태스크수·trial수) 필수. user-sim=gpt-5.2.
- **성공기준**: 라이브 pass가 floor 대비 결정론 극복분(coverage/discovery 닫힘)만큼 상승·over-action Δ≤0(게이트 자기역효과 계측·§1.3 모트).

### 1.4 산출·순서
`t2_eplan_patch` 3한계 코드(무료) → 오프라인 스모크 → [[09]] 승인 → 라이브 e2e. **Track A 먼저**(결정론·리스크 낮음·B의 기반).

---

## 2. Track B — F3 도메인-일반 "스키마-분류" 스킬 학습
### 2.1 문제 정의 (C95 §5.6·per-case)
F3 상황→정책 enum(dispute_reason·dispute_category): 고객 NL("환불 안 됨")을 **정책 taxonomy의 정확한 enum**으로 매핑. task_040 실증: 고객 "fraud"↔gold "not_as_described" — **정의는 in-context(tool 스키마)에 있으나 모델이 적용 안 함**([[42]] prior-override). ⇒ **지식 결손 아닌 스킬 결손.**

### 2.2 분리 (사용자 프레임·핵심)
- **지식(taxonomy) = ABox**: enum 정의·정책 분류표는 **KB retrieval / A2 선언**으로 공급(이미 in-context). **학습/가중치 절대 아님**(LoRA=팩트에 부적합·전이 파괴·[[11]]).
- **스킬 = TBox(도메인-일반·한 번 학습)**: "**대령된 스키마 정의를 읽고 → NL 상황을 해당 enum으로 분류 → 자기 prior로 안 덮기**". 도메인 불문 동형(스키마만 바뀜). ABox-swap로 banking 전이.

### 2.3 학습 (벤치서만·[[11]]·[[42]] 처방)
- **원천**: synth schema-content op(FAOS schema-content)·SOPBench(control-flow)·TaskBench — **"provided-schema→classify" 태스크로 합성**(다양성 [[12]]·단일템플릿 금지). **banking 분쟁 데이터 학습 금지**([[03]] 죽는 실험).
- **방법([[42]])**: SFT 설치(consult-schema-then-classify 궤적) + **prior-override 억제**(DPO/NPO: 스키마 무시하고 추측한 출력에 penalty). 프롬프트론 안 닫힘(prompt-ceiling).
- **기존 인프라 연결**: `t2_formalize_exec`(NL→formalize)·FAOS schema-content — Track B = 그 formalize 스킬의 "closed-enum 분류" 특화. [[10]] LLM=formalize·결정론=선택.

### 2.4 전이 검증 (τ² banking·미학습 전이)
- ABox=banking KB taxonomy 공급·TBox=벤치-학습 스킬(banking 미학습) → F3 enum 정확도 측정.
- **make-or-break**: **~0.44 F3 천장**(§1.4c·scale·CoT·RL·budget 전부 실패)을 도메인-일반 스킬-SFT가 깨나. 깨면 = **"소형+학습 스킬 > frontier"**([[41]] 헤드라인·frontier도 이 스킬 없어 막힘).
- **음성이면**: F3=진짜 경계 확정(learn 축까지 닫힘)·명제는 결정론 28~49%+F3-패리티로 유지(frontier도 못 감).

### 2.5 정직한 유보·리스크
- **미검증 베팅**(C38: learn 축 cfbsynth 결손 재현 실패로 미시험). 0.44가 scale/CoT/RL 불변 → 스킬-SFT가 깰지 불확실.
- **과적합 위험**: closed-enum 분류가 벤치 템플릿에 표면매핑되면 역전이([[12]]). 다양성·구조 변형 필수.
- **경계 판단**: 깨도 부분(예: 44→70%)일 수 있음. 목표=frontier 초과 or 패리티+결정론 우위.

---

## 3. 두 트랙 합성 = 전체 매커니즘
```
per-step DAG-walk (Track A 배선):
  갭마다 op ← {GET | FIND | COMPUTE | ASK | classify-schema}
    ├ under-action/discovery/compute → 결정론 (Track A·28~49%)
    ├ 선택 enum·data부재 → ASK (Track A)
    └ 상황→정책 enum → schema-classify 스킬 (Track B·F3)
  H_min 종료 (전 갭 닫힘)
```
- Track A = outer/inner 결정론 loop(coverage/discovery/compute). Track B = inner의 F3 slot(schema-classify).
- **상보**: A 없이 B만=여전히 under-action으로 실패(C94 78% 다층). B 없이 A만=F3서 frontier 패리티(28~49% + 공유천장). **둘 다=frontier 초과 잠재.**

## 4. 순서·게이트·규율
1. **Track A 무료 배선**(`_note_eplan` 3한계·오프라인 스모크) — 먼저·리스크 낮음·B 기반.
2. **Track B 무료 설계·합성**(벤치 schema-classify 태스크·V0 census·prior-override 프로브) — A와 병행 가능(무료).
3. **[[09]] 유료**: Track A 라이브 e2e → (성공 시) Track B 학습+전이 e2e. 각 승인+scope.
- **규율 하드**: [[05]] 엔진일반·리터럴0 · [[11]] 벤치학습·banking 미학습 · [[03]] 설계먼저·anti-drift · [[16]] scaffold LOCK · [[08]] per-case·집계직행 금지 · [[42]] prompt-ceiling.
- **모트 계측**: 모든 게이트에 반대편(Δspurious≤0·over-action Δ≤0). Track B는 오분류-역효과(과-분류) 계측.

## 5. 성공기준 (측정)
- **Track A**: 라이브 pass가 오프라인 결정론 상한(27.8~49%)을 in-situ 달성·over-action Δ≤0.
- **Track B**: τ² banking F3 enum 정확도 > 0.44(천장 돌파)·미학습 전이·역전이 0.
- **합성**: 소형(32B)+A+B가 banking pass서 frontier(gpt55 37.4%) 패리티 이상.
