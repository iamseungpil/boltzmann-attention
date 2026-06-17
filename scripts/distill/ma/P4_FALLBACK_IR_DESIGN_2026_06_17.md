> **★격상 (2026-06-17) = [`../LIE_ABSTRACTION_THEORY_2026_06_17.md`](../LIE_ABSTRACTION_THEORY_2026_06_17.md)**: P4(변형선택/superlative)는 표기-깊이 d(e)↑ 케이스. **정적 순서-연산 IR도 정적이면 해로움**(CoT probe 실측) → **연산-IR(LLM이 argmax/comparative *명명*·엔진 실행)로 격상**. 7B 실측: argmax/rank=극복(1.00)·**comparative만 명명 실패**(절차의미=명명 어려움) = P4의 진짜 잔여. 통제 측정 = `../B_BUDGET_SCALE_DESIGN_2026_06_17.md`.

# P4 fallback 해결 (리뷰용 DRAFT) — 조건 제어흐름 결정론-offload + 순서-연산 IR + CoT probe — 2026-06-17

> 출처: 06-17 τ² write-벽 전수 autopsy(`m_sigma_transfer_eval_v4.py` + gold-reachability 검사). 확정: write-벽 = **조건부 fallback / 다속성 변경 reasoning**(gold 21/32가 다속성)·resolver는 sound(oracle criteria→gold 32/32 unique)·실패는 STRUCTURAL(lexical≈0)·`ok`에 숨은 **grounded-but-wrong(~9건)**. 상위 = `M_SIGMA_V4_UNION_CORPUS_DESIGN.md`. 불변 = [[feedback-nl-formalize-llm-selection-deterministic]]·[[feedback-selector-verifier-deterministic]]·[[feedback-capability-vs-artifact-elicitation]].

## 0. 한 줄
**조건분기("1차 없으면 2차")는 결정론 영역인데 LLM 출력에 잘못 맡겨져 있었다. 트리거("available 없으면")는 resolver의 본업이므로 resolver가 소유하고, LLM은 *순서 있는 연산 리스트*(set/relax)만 emit한다. 그러면 분기 표현이 평평한 리스트가 되어 구조적 실패(F1-F4)가 무력화된다. grounded-but-wrong(GBW)은 구조로 안 고쳐지나 *결정론 diff-grounding 검증기*(§5b)로 *포착·회복*된다 — LLM-judge 아님. 학습 처방 전, CoT probe로 capability vs artifact를 가르고(§4), GBW는 크기×elicitation 2D sweep으로 조건#4(잔여추론⊆소형)를 시험한다(§4b).**

**★전략 묶음**: GBW catch(§5b 결정론 검증기+retry)는 GBW 크기-sweep(§4b)의 *탈출구*다 — 잡히는 GBW는 소형 모델로도 닫혀 주권 thesis 보존. **단 탈출구는 retry-수렴(P7)에 기대고 그것도 scale-bound일 수 있다(재귀·리뷰 보강C)** → §4b가 retry-수렴률을 크기별 *측정*해 "탈출구"를 가정→검증으로. §5b가 못 잡는 잔여(primary↔fallback) ∩ §4b scale-저항 ∩ retry-비수렴 = P4가 thesis 위협하는 진짜 교집합.

## 1. 진단 결박 (autopsy 권위본 = `M_A_RESULTS §12`/exp0)
- 실패 분류(M0·n=29·*구 계측*): **STRUCTURAL `fail_no_available`=8**(criteria target 미가용) + **GBW ~9 (추론치**·구 계측이 `ok`=21에 뭉뚱·case-correct 12와의 산술차로 추정) + LEXICAL≈0.
- **★새 계측(`1ffd176`) 라이브 검증**: A-prov 직접측정 = `CORRECT=10·WRONG_VARIANT(GBW)=4·STRUCT=8·LEXICAL=5`. GBW가 *직접 관측됨*(M0 GBW는 §8-0 dump가 확정). exp0 split_*.json 대조: base 14/29·concrete 15/29·typed 14/29 ✓.
- factorial 신호(6/8·n=29 노이즈): 단독축 전부 base 미만(M0 0.41·A-iso 0.28·A-nl 0.34·A-prov 0.31)이나 **C-in(ISO+NL)=0.52 > base 0.48 = 첫 추월·양성 interaction**(fail_no_available 8→4). PROV 단독은 LEXICAL=5 도입(해로움).
- gold-reachability: gold 항상 available·exact-match **unique 32/32**·tie 0 → **resolver 무죄**(oracle criteria면 100%). ⇒ 병목 = 순수 NL→criteria formalize, 특히 **조건분기**.
- base(in-head literal 0.48) > M0($select 0.41): **분기를 구조화 emit하는 게 머릿속 추론보다 어렵다**(F4).

## 2. 실패모드 (라벨·처방 매핑)
| 모드 | 설명 | 재구성(§3)이 고치나? |
|---|---|---|
| **F1** 실패조건→새제약 오인 | "없으면 clicky·RGB·full"의 RGB·full을 새 변경으로 | ✅ 무력화(§3.3·no-op 흡수) |
| **F2** keep-rest 누락 | fallback이 이미-변경 아이템 기준인데 유지 놓침 | ✅ 누적이 구조적 보장 |
| **F3** 분기 생략 | 1차만 emit·fallback 없음 | ✅ 평평한 리스트 |
| **F4** 구조화>in-head | 중첩 조건 표현이 어려움 | ◐ 완화(리스트가 중첩보다 쉬움) |
| **GBW** grounded-but-wrong | 엉뚱한 속성 변경→실재 오답 변형 | ❌ 구조 무관 — but **결정론 diff-grounding 검증기로 catch+retry**(§5b)·잔여=primary↔fallback 혼동 |

## 3. ★재구성 = 순서 있는 연산 리스트 (제어흐름은 resolver 소유)
### 3.1 IR
```jsonc
new_item_ids: [ { "$select": [
    {"set": {"<attr>": "<val>"}},     // 값 지정 변경 (1순위)
    {"set": {"<attr>": "<val>"}}      // 누적 적용 (2순위 fallback)
] } ]                                  // 순서 = 선호순위, 첫 available 채택
```
- LLM은 **트리거 조건("없으면")을 안 쓴다** — 쓸 슬롯 없음(트리거는 resolver 소유).
- **★set-only로 출하**(리뷰 보강3): autopsy fallback("clicky·none·full")은 *명시 다속성 set*이지 "속성 풀기"가 아니다. `relax{attr}`(값 미지정 자유)는 (a) 추상 synth서 거리 메트릭 없어 **tie-break 미정의** (b) retail-exchange에 free-attribute fallback 없을 공산 → **critical path서 제거**. **벤치 딥리서치가 free-attribute/ranking fallback 존재를 확인한 뒤에만** relax+tie-break 추가(§6-1).

### 3.2 resolver 시맨틱 (결정론·기존 `ma_resolver` 폴드 재사용)
- `cur = old_options`; 각 `set` 연산을 **누적 override** 적용 → `cur ⊕ op` → available 유일매칭이면 반환·아니면 다음. 첫 available 우선. 구 `by/fallback` 스키마 backward-compat 유지.
- (relax/tie-break는 §3.1대로 보류 — 데이터가 요구할 때.)

### 3.3 IR이 무엇을 하나 (정직: 제거 아니라 형태 단순화 — 리뷰 보강2)
- **F1 over-spec *손해* 흡수**: 실패조건 restate(RGB·full)가 **old와 일치하는 경우** *안 바뀐 값*이라 `{backlight:RGB}` 적용=**no-op·무해**. 단 **올바른 fallback 델타({backlight:none})는 여전히 emit해야** — 그 *어떤* 델타냐는 잔여 formalize 추론(=F3). (restate 값이 old와 *다른* 1차-미가용 criteria면 누적이 같은 미가용 타깃 재생산 → 그건 델타-emit 추론이 잡아야.)
- **F2 keep-rest 자동**(누적). **F3/F4 = 중첩분기→평평한 리스트로 *형태 단순화***(추론 제거 아님). ⇒ **§4 CoT probe가 이 "형태 단순화 이득"을 *측정*** (P-new-CoT vs P-old-CoT).
- additive·revising fallback 둘 다 override 폴드로 일반 처리.

## 4. ★CoT probe (capability vs artifact·학습 전·GPU≈0)
n=29 base 7B inference. **2-stage**(리뷰 보강4): **free CoT 먼저 → 별도 추출**(한 번에 CoT→스키마 강제하면 forced-JSON 왜곡이 probe에 섞여 capability를 못 가름·CRANE/딥리서치 정합). `m_sigma_cot_probe.py` 신규.
| 셀 | 출력(2-stage) | 측정 |
|---|---|---|
| **P-lit** | free CoT → 최종 item_id(literal) | 분기를 *추론*으론 푸나? |
| **P-old-CoT** | free CoT → 추출 by/fallback | 구 스키마 formalize 손실 |
| **P-new-CoT** | free CoT → 추출 순서-연산(§3) | 재구성 formalize 손실 |
| 기준 | M0 0.41 · base-no-CoT-lit 0.48 | |

**★regime 게이트(리뷰 보강1·중요):** 위 셀은 전부 **카탈로그 통째 주는 단발**이고, 거기선 base in-head이 유리(0.48>M0 0.41)다. 그러니 **카탈로그-withhold 조건**(기존 `--withhold`로 get_product_details 드롭 → 모델이 카탈로그 못 봐 *구조화 강제*·resolver만 카탈로그 보유 = 분해가 *필요한* regime)을 각 셀에 병행한다. τ²도 에이전트가 카탈로그를 보지만, 분해의 값은 *큰 카탈로그/fetch 안 한* regime서 나므로 — withhold-셀이 deployment regime의 대리.

**사전등록 판독(스코프 명시):**
- **단발-given 한정**: P-lit ≫ 0.48 → 능력 있음·강제 JSON artifact. **단 "무학습 종결" 결론은 *작은 주어진 카탈로그 한정*** — withhold-셀/큰-카탈로그서 구조화가 이기면 라인 유지(deployment regime 미확인 채 죽이지 말 것·v4 §9 pass@1 bridge와 연결).
- P-new-CoT > P-old-CoT → 재구성이 inference-time에 검증(형태 단순화 이득·§3.3).
- P-lit ≈ 0.48 ∧ withhold-셀도 낮음 → 진짜 reasoning 천장 → 대조 synth(§5).
- P-new-CoT ≪ P-lit → 델타 IR 자체 lossy → IR 재설계.

## 4b. ★GBW 2D 크기 sweep (thesis-임계·조건#4 시험)
GBW가 *크기(capability)* 문제인지 *형식(artifact)* 문제인지를 [[feedback-capability-vs-artifact-elicitation]] 규율대로 **크기×elicitation 2D로 격리**(섞지 말 것). M-A floor sweep이 이미 "binding 벽 ≠ scale"을 시사하나 GBW-격리는 미측정.
| | forced-emit | CoT |
|---|---|---|
| 7B | base 측정됨 | §4 probe |
| 14B | ○ | ○ |
| 32B (coworker) | ○ | ○ |
| 72B (coworker) | ○ | ○ |
- **★n 확장이 선결(리뷰 보강B)**: GBW 4/29(±2~3)·interaction ±1~2는 scale 추세 분해 불가. τ² exchange는 29 상한일 공산 → **§4b는 *synth 평가셋*(n≥100·controllable fallback 난이도)으로 능력 측정**, τ²-29는 transfer 헤드라인 유지. synth→synth 약점은 *capability 측정*엔 무해(전이 주장 아님). **★스코프: synth GBW-sweep은 *구조적 fallback-추론* 능력을 잼**(어느 속성 change/keep/순서); τ² GBW의 *어휘/의미* 성분(synonym "Google Home"→값)은 **ABox 제공 몫·모델 일 아님**([[feedback-nl-formalize-llm-selection-deterministic]]) → 오히려 구조를 어휘서 *격리*해 더 깨끗.
- **지표 = (i) GBW율 격리**(`ok_wrong_variant`·계측 `1ffd176`) **(ii) new_item_ids 정확도 (iii) structural-fail율** (scale이 어디서 돕나 분해) **(iv) ★retry-수렴률(리뷰 보강C)**: §5b reject 후 재시도가 통과하나 — *크기별*. 탈출구(아래)가 "가정"이 아니라 *검증*이 되게. retry cap = §6-4 spraying 규율 공유.
- **싼 셀 먼저**(7B/14B × both), **그 다음 coworker 32/72B**.
- **판정 = 조건#4**([[project-decomposition-optimality-contribution]]): GBW가 소형으로 닫히면 LLM-leg가 sLLM 충분=주권 성립·큰 모델 요구면 sovereignty-leg 위협. **§5b catch + retry-수렴과 교차**(전략 묶음).

## 5. GBW + capability 결핍 레버 (구조와 별도)
- **CoT 추출(④)**: emit 전 "(a)요청 변경 (b)*이건 품절목표 묘사지 새 변경 아님* (c)fallback 델타" 명시 → F1·GBW.
- **대조 synth(③)**: 미니멀 페어 — 같은 NL, (a)실패조건이 1차목표 restate(criteria 아님) vs (b)진짜 다속성 변경. + GBW 하드네거티브(유혹적 오답 속성변경을 정답으로 교정).
- (relax/tie-break는 §3.1대로 보류 — set-only 출하·데이터 요구 시만.)

## 5b. ★GBW catch = 결정론 diff-grounding 검증기 (LLM-judge 아님)
resolver 단독은 GBW를 못 잡는다(criteria 구조적 유효). **출력에 둘째 결정론 검증기**를 건다([[feedback-selector-verifier-deterministic]] 준수). **이중 역할**: (i)*진단* — catchable GBW율 측정(§8-1·탈출구 크기) (ii)*배포 루프* — catch→retry로 정확도↑(§4b-iv retry-수렴이 이걸 크기별 측정). **양방향 검사**(리뷰 보강A):
- **(a) commission(주):** `old → 선택 variant` diff의 **바뀐 모든 속성 새 값이 NL grounding**되나·아니면 reject. 예: backlight→white인데 "white"가 NL에 없음 → GBW 포착.
- **(b) omission(약·보강A):** NL에 attest된 *카탈로그-값 토큰*이 old∨chosen에 다 출현하나 — 빠진 required-change 부분 포착. 단 *완전 결정론 불가*(어느 NL언급이 required냐 = 모델이 실패하는 그 comprehension) → weak 근사·false-reject 위험(multi-primary+revert 케이스).
- **★synonym-precision 한계(실측 박제)**: gold fallback `none`이 NL엔 "**no backlight**" → substring 미스 → **synonym map(ABox) 없으면 정답을 false-reject**. negation 값(none/no/without)이 대표 함정. ⇒ **검증기 정밀도 = synonym map 품질에 의존**·runner의 false-reject 타일리가 실측(catch율과 *함께* 봐야 함). 값→속성 grounding = 카탈로그 value-space + synonym(ABox)·**LLM 판단 0**.
- **emit-gate(보조):** `set(attr,val)`을 *val이 NL attest된* 경우만 허용(사전 차단·D5/getter_map 동형).
- **back-translation(보조):** 선택 criteria를 템플릿 문장 렌더 → NL overlap 검사.
- **retry 회복:** 검증기 reject → 모델 재시도(다른 속성)·결정론 통과까지 = GBW를 *silent 오답→포착·회복 오류*로 전환(P4+P7 통합 지점·§6-5).
- **잔여(이것도 못 잡음):** 바뀐 속성이 전부 NL-grounding 됐는데 틀림 = **primary↔fallback 혼동**(어느 mentioned 속성이 primary인지 오판). §4b scale-저항과 교차 시 P4의 진짜 위협(§0 전략 묶음).
- **선구현 권고:** §8 순서 전에 *지금 가진 base*에 diff-grounding을 얹어 "잡히는 GBW 비율"을 먼저 측정 → §4b sweep 해석 틀을 미리 제공(싸고, catch율이 thesis 탈출구 크기를 정량화).

## 6. 위험 / 구멍 (정직·리뷰 훅)
1. **ranking-fallback 미커버**: "없으면 *가장 싼* tactile" 같은 *순위* fallback은 속성-델타로 표현 불가. retail-exchange=속성-델타지만 airline=시간순위 가능 → 재구성은 **속성-델타 fallback 전용**·순위는 별 sub-primitive(벤치 딥리서치가 존재 확인).
2. **트리거=가용성 가정**: resolver가 트리거 소유하려면 항상 "available 없음"이어야. P4 정의(가용성 매칭)엔 성립·"너무 비싸면" 같은 다른 트리거 나오면 깨짐 → 스코프 명시.
3. **relax tie (보류)**: relax 도입 시만 — 추상 토큰 거리 메트릭 없어 tie-break 미정의 → set-only로 회피(§3.1). 데이터가 free-attribute fallback 요구하면 그때 tie-break 정의.
4. **델타 spraying**: 많은 델타 뿌려 아무거나 걸림 노리는 퇴행 → 첫-available이 이른 오답 안착 → grammar로 델타 수 cap·순서 규율.
5. **★단발 측정타당성 = 게이트(리뷰 보강1)**: 카탈로그 통째 주는 단발선 base in-head 유리(0.48>0.41) → probe 판독 "무학습 종결"이 *deployment regime 미확인 거짓 종결* 위험. → §4 regime 게이트(카탈로그-withhold 셀)·판독 스코프 "작은 주어진 카탈로그 한정"·v4 pass@1 bridge 연결.
6. **GBW 잔존**: §3은 구조만·GBW catch는 §5b 검증기·잔여(primary↔fallback)는 §4b. 섞지 말 것.

## 7. 선행연구 연결 (딥리서치 검증 지점·`w2ueso1g5` P4·`wrdn8dh77` 벤치)
- §3의 `set/relax 순서 + 가용성 폴드` = DB **preference query / constraint relaxation(skyline·소프트제약 완화순서)**와 동형(2차 angle). 딥리서치가 (a) relaxation-order IR이 semantic parsing서 검증됐나 (b) coarse-to-fine/least-to-most가 분기에 쓰였나 (c) 대조 synth가 검증 기법인가를 확인 → 처방 확정.

## 8. 순서
0. **per-case dump**(계측 `1ffd176`·미실행·factorial 종료 후 첫 타) — 실패분포(F1-F4 vs GBW vs primary↔fallback)를 *측정*. 재구성이 옳은 레버인지·GBW 비중을 판정(§리뷰 #1).
1. **diff-grounding 검증기 선구현**(§5b) — 지금 base에 얹어 "잡히는 GBW 비율" 측정 = thesis 탈출구 크기 정량화·§4b 해석틀.
2. **CoT probe**(§4·GPU≈0·base는 기존 vLLM에 얹기) — **2-stage(free CoT→추출)** + **카탈로그-withhold 셀 병행** → capability vs artifact + 재구성 inference 검증(regime 스코프).
3. resolver에 **순서-연산 `set`-only 지원**(폴드·구 스키마 호환). relax/tie-break는 벤치 딥리서치가 free-attribute fallback 확인 후에만.
4. **n 확장(선결·보강B)**: synth 평가셋 n≥100(controllable fallback) — §4b sweep이 GBW 추세 분해 가능하게(τ²-29는 transfer 헤드라인 유지).
5. **GBW 2D 크기 sweep**(§4b·7B/14B 먼저·coworker 32/72B) — GBW율 + new_item + retry-수렴률(보강C) — 조건#4 시험.
6. dump+probe="artifact" → 재구성+CoT+검증기 종결. "capability 결핍" → synth gold 순서-연산 IR 재추출 + 대조 synth(§5).
7. 단발 천장 시 → 멀티턴 회복(P4+P7 통합·§6-5).
- 결과 박제 `M_A_RESULTS §13` — **factorial은 3-seed band로**(C-in 0.52 추월=n=29 1~2케이스·점추정 금지).

## 9. 한 줄
**fallback의 조건분기를 LLM에서 resolver로 옮긴다(또 하나의 결정론-offload). LLM=순서-연산(set-only) 리스트만 emit·resolver=트리거+누적(relax/tie-break는 데이터 요구 시만). 누적-override가 F1(no-op흡수)·F2(keep-rest)·F3/F4(평평한 리스트)를 구조적으로 단순화(추론 제거 아님·probe가 이득 측정). GBW는 결정론 diff-grounding 검증기(commission+omission)+retry로 *포착·회복*(§5b·LLM-judge 아님·정밀도는 synonym/ABox 의존)·잔여=primary↔fallback. dump로 실패분포 측정 → CoT probe로 capability vs artifact → n≥100 synth로 GBW 2D 크기 sweep(GBW율+retry-수렴)으로 조건#4 시험. ★검증기 catch+retry-수렴(§5b/§4b)이 크기 sweep의 탈출구 — 단 retry도 scale-bound일 수 있어 *측정*으로 검증.**
