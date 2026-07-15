# frontier 실패궤적 전수 × 우리 매커니즘 극복 판정 (2026-07-16)

> 사용자 지시: "모든 frontier 모델 실패궤적을 분석하여 우리 매커니즘이 극복할 수 있는지 확인."
> 데이터: 17 frontier 리더보드 제출 banking 궤적(`C:/tmp/traj/*_banking.json`·user-sim gpt-5.2·[[47]] 목록 일치).
> 스크립트: `bank_frontier_mechanism.py`·`bank_tierA_grounding.py`·`bank_grounding_bounds.py`(+ `bank_perstep_decomp` 재사용). 로컬 무료.
> ★[[08]] forensic guard가 grounding 오탐(enum-schema)을 잡아 naive 73.5%→교정 52.3%. per-case 정독 반영.

## 0. 한 줄
17 frontier 전부 banking 실패-과다(pass 9.8~37.4%). 실패의 **~52%를 우리 per-step 균일 연산-loop이 결정론+GET로 극복**(coverage/discovery/compute/grounding)하고, **최대 잔여 23.1%는 F3 의미참조 경계(enum NL→정규화)** = LOCKED-frame이 늘 지목한 **frontier-공유 잔여**. 극복 사정권은 **model-invariant**(Tier-D 32~49%가 pass율과 무관·scale이 못 건드림).

## 1. frontier pass율 (17 모델·388 sim·banking)
gpt55 37.4 · gpt54 30.7 · opus47 25.3 · gpt52 24.7 · opus46 24.5 · gemini31pro/sonnet45 22.4 · opus45 21.4 · gemini3flash 20.6 · grok42 17.5 · gemini3pro 15.7 · grok4fast 14.2 · gemini25pro 12.6 · gpt52none/grok41fast 12.4 · glm5 11.1 · qwen35 9.8.
→ **최강 frontier도 62%+ 실패** = banking = frontier 못 사는 영역(whitespace).

## 2. ★model-invariance (극복 tier가 pass율과 무관)
per-model 결정론-closable(Tier-D) 비율(DB-basis 실패 기준):
| 극단 | pass% | Tier-D | Tier-A(의미) | Blind |
|---|---|---|---|---|
| gpt55(최강) | 37.4 | 43.3% | 41.5% | 13.4% |
| gpt54 | 30.7 | 48.8% | 38.9% | 10.7% |
| opus45 | 21.4 | 31.9% | 41.6% | 22.2% |
| gemini25pro | 12.6 | 47.9% | 33.3% | 17.2% |
| qwen35(최약) | 9.8 | 43.5% | 8.7%* | 47.8%* |
- **Tier-D = 31.9~48.8%·pass율과 상관 없음**(gpt55 43.3% ≈ qwen 43.5%). ⇒ 우리 매커니즘이 겨냥하는 실패(under-action/discovery)는 **약↔강 모델을 가르는 축이 아니라 scale-불변 보편 잔여**. (*glm5/qwen=DB-basis 표본 small·pure-DB 지배.)

## 3. ★sim-레벨 극복 판정 (DB-basis 실패 4262·[[08]] enum-오탐 교정)
| tier | % | 우리 레버 |
|---|---|---|
| **D+X+compute (결정론)** | **47.6%** | coverage-track+H_min(under-action)·강제열거 FIND(reach)·⋈-decidable GET·ABox COMPUTE(+eligibility/apy 규칙확장) |
| **그라운딩-closable (data present)** | **4.6%** | GET 원문-치환(id/amount/date가 tool-record에 존재) |
| **극복 소계** | **52.3%** (관측가능 중 **64.2%**) | |
| F3-의미경계 (enum NL→정규화) | 23.1% | **경계**(ASK로 부분·NL→symbolic 매핑=능력) |
| ASK (data 부재) | 6.0% | user-원천 ASK |
| Blind (pure-DB) | 18.6% | 오프라인 밖(DB-replay/live) |
- **극복 사다리(정직)**: 현 ABox(liability만) 결정론 ≈**42.6%** → +compute 규칙확장(eligibility/apy/difference) ≈**47.6%** → +GET그라운딩 ≈**52.3%**.

## 4. ★F3 경계 = 이론 정합 (forensic 회수)
- naive grounding 25.9%(→극복 73.5%)는 **enum-schema 오탐**: `card_action`·`closure_reason`·`transaction_type` 같은 enum 값이 tool 스키마/KB 문서에 정의로 등장해 "present"로 오판(per-case 정독 task_039/047/086서 발각). literal 존재≠case 그라운딩.
- 교정: **enum 필드(NL→정규화)는 그라운딩 무효 → F3 의미참조로 분류**. `bank_grounding_bounds`: enum 1759개 literal-present 100%(오염 확증)·data 필드는 present 62.7%(tool-record GET 정당).
- ⇒ 최대 잔여 **F3-의미경계 23.1% = LOCKED-frame §1.1 F3(⋈ 참조매칭·의도)** = "scale·thinking·scaffold 어느 레버도 못 여는 경계"(§1.2)·**frontier와 공유**. **우리 매커니즘의 실패가 아니라 우리 프레임이 예측한 그 경계**. enum의 67%는 user 발화에 존재(→ASK-confirm 부분 사정권)이나 NL→symbolic 정규화 자체=F3 능력.

## 5. 결론 (사용자 질문 답)
- **"우리 매커니즘이 frontier 실패를 극복하나?"** → **관측가능 실패의 ~64%(전체 52%)를 결정론+GET로 극복**. 지배 레버 = under-action/discovery(coverage 40.7%+FIND 27.2%·C94)를 닫는 per-step 균일 loop{GET/FIND/COMPUTE/ASK}+H_min.
- **극복 못 하는 부분 = 정확히 F3 의미경계(23.1%)** — 우리 프레임이 처음부터 "잔여 경계·frontier 공유"로 못박은 그것. + pure-DB blind 18.6%(오프라인 관측 한계·능력 아님).
- **model-invariance** = 이 잔여가 scale로 안 닫힘의 직접 증거(C94 종료 100% user_stop·[[45]] load-이론 정합). = **소형+매커니즘이 frontier에 도달하는 근거**(00-thesis).

## 6. caveat
- action_checks proxy(reward=DB) ~90% tight(C93). pure-DB 18.6%는 per-step 관측 불가(하한 아님).
- read/write=이름-prefix·enum/data=필드명 휴리스틱. compute-like 극복은 **ABox 규칙 확장 전제**(현재 미구현·liability만).
- 그라운딩=tool-record literal(정규화 저평가·enum 제외로 하한). F3 23.1% 중 일부는 ASK-closable(과대). sim=all-layer AND(보수).
- glm5/qwen35=DB-basis 표본 small(58/69)·pure-DB 지배(별도).
