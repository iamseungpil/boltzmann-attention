# P4 fallback 해결 (리뷰용 DRAFT) — 조건 제어흐름 결정론-offload + 순서-연산 IR + CoT probe — 2026-06-17

> 출처: 06-17 τ² write-벽 전수 autopsy(`m_sigma_transfer_eval_v4.py` + gold-reachability 검사). 확정: write-벽 = **조건부 fallback / 다속성 변경 reasoning**(gold 21/32가 다속성)·resolver는 sound(oracle criteria→gold 32/32 unique)·실패는 STRUCTURAL(lexical≈0)·`ok`에 숨은 **grounded-but-wrong(~9건)**. 상위 = `M_SIGMA_V4_UNION_CORPUS_DESIGN.md`. 불변 = [[feedback-nl-formalize-llm-selection-deterministic]]·[[feedback-selector-verifier-deterministic]]·[[feedback-capability-vs-artifact-elicitation]].

## 0. 한 줄
**조건분기("1차 없으면 2차")는 결정론 영역인데 LLM 출력에 잘못 맡겨져 있었다. 트리거("available 없으면")는 resolver의 본업이므로 resolver가 소유하고, LLM은 *순서 있는 연산 리스트*(set/relax)만 emit한다. 그러면 분기 표현이 평평한 리스트가 되어 구조적 실패(F1-F4)가 무력화된다. 단 grounded-but-wrong(GBW)은 grounding 오류라 구조로 안 고쳐지므로 별 레버(CoT·대조 synth). 학습 처방 전, CoT probe로 capability vs artifact를 먼저 가른다.**

## 1. 진단 결박 (autopsy 권위본 = `M_A_RESULTS §12`/exp0)
- 실패 분류(M0·n=29): **STRUCTURAL `fail_no_available`=8**(criteria target 미가용) + **GBW ~9**(`ok`이나 gold≠·resolver 못 잡음·계측수정 `1ffd176`으로 분리) + LEXICAL≈0.
- gold-reachability: gold 항상 available·exact-match **unique 32/32**·tie 0 → **resolver 무죄**(oracle criteria면 100%). ⇒ 병목 = 순수 NL→criteria formalize, 특히 **조건분기**.
- base(in-head literal 0.48) > M0($select 0.41): **분기를 구조화 emit하는 게 머릿속 추론보다 어렵다**(F4).

## 2. 실패모드 (라벨·처방 매핑)
| 모드 | 설명 | 재구성(§3)이 고치나? |
|---|---|---|
| **F1** 실패조건→새제약 오인 | "없으면 clicky·RGB·full"의 RGB·full을 새 변경으로 | ✅ 무력화(§3.3·no-op 흡수) |
| **F2** keep-rest 누락 | fallback이 이미-변경 아이템 기준인데 유지 놓침 | ✅ 누적이 구조적 보장 |
| **F3** 분기 생략 | 1차만 emit·fallback 없음 | ✅ 평평한 리스트 |
| **F4** 구조화>in-head | 중첩 조건 표현이 어려움 | ◐ 완화(리스트가 중첩보다 쉬움) |
| **GBW** grounded-but-wrong | 엉뚱한 속성 변경→실재 오답 변형 | ❌ **구조 무관·grounding 레버 필요**(§5) |

## 3. ★재구성 = 순서 있는 연산 리스트 (제어흐름은 resolver 소유)
### 3.1 IR
```jsonc
new_item_ids: [ { "$select": [
    {"set":   {"<attr>": "<val>"}},   // 값 지정 변경
    {"relax": "<attr>"}               // 값 미지정(아무 available 허용)
] } ]                                  // 순서 = 선호순위, 첫 available 채택
```
- LLM은 **트리거 조건("없으면")을 안 쓴다** — 쓸 슬롯 없음(트리거는 resolver 소유).

### 3.2 resolver 시맨틱 (결정론·기존 `ma_resolver` 폴드 재사용)
- `cur = old_options`; 각 연산을 **누적 override** 적용 → `cur ⊕ op` → available 유일매칭이면 반환·아니면 다음.
- `set`: 값 지정(tie 없음). `relax{attr}`: 그 속성 자유 → 후보 여럿이면 **결정론 tie-break**(기본: old 값에 가장 가까운 것·명문화·§5.3).
- 첫 available 우선. 구 `by/fallback` 스키마 backward-compat 유지.

### 3.3 왜 구조적으로 F1-F4 무력화 (핵심 성질)
- **누적-override 흡수**: 실패조건 restate(RGB·full)는 *안 바뀐 old 값*이라 `{backlight:RGB}` 적용=**no-op·무해**. over-spec 손해 소멸(F1).
- **누적=keep-rest 자동**(F2). **평평한 순서 리스트=분기 표현 단순화**(F3·F4).
- additive("백라이트 추가변경")·revising("clicky→tactile 재변경") 둘 다 override 폴드로 일반 처리.

## 4. ★CoT probe (capability vs artifact·학습 전·GPU≈0)
n=29 base 7B inference. 4셀, 기존 eval 케이스 재사용(`m_sigma_cot_probe.py` 신규).
| 셀 | 출력 | 측정 |
|---|---|---|
| **P-lit** | CoT→최종 item_id(literal) | 분기를 *추론*으론 푸나? |
| **P-old-CoT** | CoT→구 by/fallback | 구 스키마 formalize 손실 |
| **P-new-CoT** | CoT→순서-연산(§3) | 재구성 formalize 손실 |
| 기준 | M0 0.41 · base-no-CoT-lit 0.48 | |

**사전등록 판독:**
- P-lit ≫ 0.48 → **능력 있음·강제 JSON artifact** → **학습 불요·재구성+CoT로 종결**.
- P-new-CoT > P-old-CoT → **재구성이 inference-time에 검증**(학습 전 IR 우월성).
- P-lit ≈ 0.48 → 진짜 reasoning 천장 → 대조 synth(§5) 필요.
- P-new-CoT ≪ P-lit → 델타 IR 자체 lossy → IR 재설계.

## 5. GBW + capability 결핍 레버 (구조와 별도)
- **CoT 추출(④)**: emit 전 "(a)요청 변경 (b)*이건 품절목표 묘사지 새 변경 아님* (c)fallback 델타" 명시 → F1·GBW.
- **대조 synth(③)**: 미니멀 페어 — 같은 NL, (a)실패조건이 1차목표 restate(criteria 아님) vs (b)진짜 다속성 변경. + GBW 하드네거티브(유혹적 오답 속성변경을 정답으로 교정).
- **tie-break 명문(§3.2)**: relax 후 다수 available → old-근접 결정론 규칙.

## 6. 위험 / 구멍 (정직·리뷰 훅)
1. **ranking-fallback 미커버**: "없으면 *가장 싼* tactile" 같은 *순위* fallback은 속성-델타로 표현 불가. retail-exchange=속성-델타지만 airline=시간순위 가능 → 재구성은 **속성-델타 fallback 전용**·순위는 별 sub-primitive(벤치 딥리서치가 존재 확인).
2. **트리거=가용성 가정**: resolver가 트리거 소유하려면 항상 "available 없음"이어야. P4 정의(가용성 매칭)엔 성립·"너무 비싸면" 같은 다른 트리거 나오면 깨짐 → 스코프 명시.
3. **relax tie 폭증**: relax가 후보 여럿 남기면 tie → tie-break 규칙 필수(§3.2). NL이 값 주면 set(tie 없음) 기본.
4. **델타 spraying**: 많은 델타 뿌려 아무거나 걸림 노리는 퇴행 → 첫-available이 이른 오답 안착 → grammar로 델타 수 cap·순서 규율.
5. **단발 eval 과소평가**: 카탈로그를 통째 주는 단발선 base in-head이 통해 $select를 과소평가. 재구성의 진짜 값은 **fetch 필요·카탈로그 부분제시 멀티턴**서 — 평가 셋업도 그쪽이 정직(별 항목).
6. **GBW 잔존**: §3은 구조만·GBW는 §5 레버 의존. 섞지 말 것.

## 7. 선행연구 연결 (딥리서치 검증 지점·`w2ueso1g5` P4·`wrdn8dh77` 벤치)
- §3의 `set/relax 순서 + 가용성 폴드` = DB **preference query / constraint relaxation(skyline·소프트제약 완화순서)**와 동형(2차 angle). 딥리서치가 (a) relaxation-order IR이 semantic parsing서 검증됐나 (b) coarse-to-fine/least-to-most가 분기에 쓰였나 (c) 대조 synth가 검증 기법인가를 확인 → 처방 확정.

## 8. 순서
1. **CoT probe 4셀**(§4·GPU≈0·factorial 종료/레인 빔 즉시·base는 기존 vLLM에 얹기 가능) — capability vs artifact + 재구성 inference 검증. 첫 타.
2. resolver에 **순서-연산(set/relax) 지원**(폴드·구 스키마 호환·tie-break 명문).
3. probe="artifact" → 재구성+CoT 종결. "capability 결핍" → synth gold를 순서-연산 IR로 재추출 + 대조 synth(§5).
4. 단발 천장 시 → 멀티턴 회복(P4+P7 통합·§6-5).
- 결과 박제 `M_A_RESULTS §13`.

## 9. 한 줄
**fallback의 조건분기를 LLM에서 resolver로 옮긴다(또 하나의 결정론-offload). LLM=순서-연산(set/relax) 리스트만 emit·resolver=트리거+누적+tie-break. 누적-override가 F1(no-op흡수)·F2(keep-rest)·F3/F4(평평한 리스트)를 구조적으로 무력화. GBW는 grounding 레버(CoT·대조 synth) 별도. 학습 전 CoT probe로 capability vs artifact를 먼저 가른다 — artifact면 무학습 종결.**
