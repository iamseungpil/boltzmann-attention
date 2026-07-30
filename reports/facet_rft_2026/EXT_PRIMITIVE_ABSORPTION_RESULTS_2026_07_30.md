# EXT 재판정 — 공용 primitive 흡수율 실측 (X6-i) · 2026-07-30

> 무료 트랙. C235(§13e) 설계 TODO ①callback 인터페이스 목록 ②공용 primitive 목록 ③EXT 10키 재판정에
> **실측으로** 답한다. handoff §6-④.
> 도구 = `scripts/distill/tau2/x6i_ext_primitive_absorption.py` (재현 1커맨드)
> 선행 = C234 §13d(EXT 화이트리스트) · C235 §13e(엔진 고정+callback) · C237(capex/opex 정렬) · C238(리터럴 감사)

## §0. 한 문단

C235가 남긴 질문 "공용 primitive가 E1/E2를 얼마나 흡수하나(→C_ext 축소분)"는 추측할 일이 아니라
**세는 일**이었다. A2가 선언한 op와 엔진이 실행하는 op를 맞춰 보니 **28/28 = 흡수율 100%**,
**callback 구현체 요구 0**이다. 즉 현 3도메인에서 E1(계산 명세)의 **도메인별 실행 코드 비용은 0**이고
A2는 식(what)만 선언한다 — §13e의 설계 의도가 이미 코드에 성립해 있다. (엔진은 39 op을 지원하고
도메인이 28을 쓰므로 **미사용 여유 11**.)

**그런데 그 100%는 "공용 라이브러리 1개"가 아니라 서로 다른 인터프리터 3개의 *합집합* 기준이고,
3자 교집합은 0이다.** `t2_compute.py`(28 op·라이브) · `gate_interpreter.py`(9) ·
`bank_eplan_controller.py`(9·산술 전담)이 각자 다른 어휘를 갖는다. 따라서 §13e의 "엔진 = 단일·고정
+ 공용 도구 라이브러리"는 **의도로는 성립하고 물리적으로는 미성립**이며, 이 분열은 [[03b]]가 금지한
**술어 이중화**다. 통합이 처방이고, **통합 시 11 op를 이식하지 않으면 E-PLAN 계산 경로가 조용히
죽는다**([[14]] 배선의 선결 조건).

이 조사 중 **내 오진 3건을 스스로 잡았다**(§4): 초판 흡수율 50%는 인터프리터 1개만 수확한
아티팩트였고, "라이브 버그 확정"은 호출부를 추적하기 전에 내린 성급한 단정이었고, 선언-op 추출
워커에 버그가 있어 분모를 28→14로 절반 과소 계상했다.

### [[05]] 3질문 ([[17]] 상설 의무)

| # | 질문 | 답 | 근거 |
|---|---|---|---|
| **Q1** | scaffold/A2의 도메인-특화를 순증시키나? | **No** | 측정만. 처방(인터프리터 통합)은 **엔진 내부 중복 제거**라 A2 불변·op 어휘 순감 없음(39 유지)·도메인 특화 0 |
| **Q2** | 모델의 유동적 판단을 결정론에 동결하나? | **No** | 새 op·새 스펙 0. 이미 선언된 식의 실행 위치만 통합 |
| **Q3** | scaffold가 모델 대신 도메인 행동을 수행하나? | **No** | 계산 offload는 **기존** 메커니즘([[10]] LLM=formalize·엔진=실행). 신규 수행 0 |

⇒ 전부 No. 단 통합은 엔진 변경이므로 **설계·리뷰 후 구현**([[03]]-7).

## §1. 방법

**op 어휘를 문서에서 읽지 않고 코드에서 수확한다** — `t2_compute.py` docstring은 지원 op를 15개로
적었으나 실제 코드는 28개다(문서 낡음). 수확 = `op == "X"` / `op in ("X","Y")` 패턴.

**A2 선언 op는 전수 재귀 추출** — `a2/*.json` + `a2/split/*.json`의 모든 중첩 dict에서 `"op"` 키 값.
계산식은 트리라(`round(multiply(divide(subtract(...))))`) 최상위만 보면 놓친다.

**커버 판정** = 그 op가 인터프리터에 **이미 구현**돼 있어 A2가 식만 선언하고 실행은 공용 코드가
하는 경우. 미커버 = 도메인이 실행 코드를 새로 공급해야 함 = **callback 후보**.

## §2. 실측 — 인터프리터가 3개이고 교집합이 0이다

| 모듈 | op 수 | 자기만 보유 | 역할 |
|---|---|---|---|
| `t2_compute.py` | **28** | 21 | **라이브 경로**(`t2_scaffold_get` → `apply_op`). 자칭 "도메인-일반 compute op 라이브러리" |
| `gate_interpreter.py` | 9 | 7 | 게이트 술어 평가(`argmax_where`·`argmin_where`·`disjoint`·`equal_len`·`count`·`lookup`·`most_recent`) |
| `bank_eplan_controller.py` | 9 | 4 | E-PLAN 컨트롤러(**산술 `add`·`divide`·`round`·`subtract` 전담**) — [[14]] e2e 미배선 |
| **합집합** | **39** | — | — |
| **3자 교집합** | **0** | — | ★공유 op이 **하나도 없다** |

- `t2_compute.py` **결손 11 op**(타 모듈엔 있음): `add`·`argmax_where`·`argmin_where`·`count`·
  `disjoint`·`divide`·`equal_len`·`lookup`·`most_recent`·`round`·`subtract`.
- 같은 계산이 **두 어휘로 이중 표현**돼 있다 — 이자 차액을 라이브 경로는
  `get_interest_correction`(`diff`+`multiply`)로, 컨트롤러 경로는
  `compute_ops.submit_interest_discrepancy_report`(`subtract`+`divide`+`round`)로 각각 적었다.
- 실측 확인: `t2_compute.apply_op`에 후자 스펙을 넣으면 **`None`**, `lookup_table` 스펙은 정상(50).
  `bank_eplan_controller.compute_fields`에 넣으면 정상(33.0·selftest (7)이 gold 2건 검증).

## §3. 흡수율 — 28/28 = 100% · callback 요구 0

| | |
|---|---|
| 엔진 지원 op (3 모듈 합집합) | **39** |
| A2가 선언한 서로 다른 op | **28** |
| 공용 커버 | **28** |
| 미커버(callback 후보) | **0** |
| **흡수율** | **100%** (op 종류 기준) |
| 미사용 여유 | 11 |

사용 빈도 상위(선언 노드 수): `const`×30 · `if_then`×22 · `multiply`×18 · `case`×16 ·
`compare`×16 · `days_between`×6 · `group_reduce`/`ref`/`select_discrepant`/`str_eq`×4.
retail 전용: `argmax_where`·`argmin_where`·`count_where`·`disjoint`·`equal_len`·`sum`.

> ⚠️이 측정은 **EXT 키에 한정하지 않고 A2 전체의 op 선언**을 센다(CORE의 `scaffold_get_tools`
> 포함). EXT만 세면 표본이 12 노드로 줄어 결론이 약해지므로, 넓게 세고 §3 표에서 EXT를 따로 뗀다.
> 흡수율이 EXT-only가 아니라 **A2-전체 기준**이라는 점을 인용 시 명시할 것.

### EXT 키 재판정 (C235 TODO ③)

| 도메인 | 키 | C234 분류 | op | 판정 |
|---|---|---|---|---|
| banking | `compute_ops` | E1 계산명세 | 8 | **공용실행** (도메인 코드 0) |
| banking | `field_ops` | E1 계산명세 | 0 | 데이터만(field→operator 분류) |
| banking | `action_tool_executor`·`function_agents` | E2 도구셋 | 0 | 데이터만 |
| banking | `identifying_arg_types` | E3 스키마상수 | 0 | 데이터만 |
| retail | `calc_specs` | E1 계산명세 | 4 | **공용실행** |
| retail | `calc_tool` | E2 도구셋 | 0 | 데이터만(도구 이름) |
| retail | `variant_spec`·`variant_operand` | E3 스키마상수 | 0 | 데이터만 |
| airline | `identifying_arg_types` | E3 스키마상수 | 0 | 데이터만 |

⇒ **E1/E2/E3 전부 "데이터 또는 공용실행"**이고 도메인 실행 코드를 요구하는 항목이 **없다**.

⚠️**분류 불일치 기록**: `a2/split/*.ext.json`은 C234 화이트리스트 **이전** 산출물이라, C234가
CORE 흡수(7) 또는 폐기(2)로 재분류한 키가 아직 EXT 파일에 남아 있다(banking 6·retail 3 = `ref_iso`·
`reference_filter`·`analysis_producers`·`assertion_operands`·`param_cap_check`·`view_field_annotations`·
`placeholders`·`regen_resolver_specs`·`tool_error_specs`). **split 재생성이 C234 반영 대기 중**
(`x6c_a2_emit_split.py` 재실행 + 화이트리스트 연동). 이 doc의 §3 표는 C234 분류를 기준으로 읽을 것.

## §4. 자기정정 3건 (반복 방지)

0. **선언-op 추출 워커 버그 — 분모를 절반으로 과소 계상했다.** 초판 `_find_ops`가
   `if k != "op"`로 재귀해 **`"op"` 키의 값이 중첩 dict일 때 그 서브트리를 통째로 건너뛰었다**
   (`scaffold_get_tools[i]["op"]`가 정확히 그 형태). banking 실제 22 op을 8로 셌고 전체가 14로
   나왔다. 정규식 ground-truth(`"op":\s*"..."`)와 대조해 발견 → 수정 후 **28**. 결론(100%·미커버 0)은
   불변이나 **표본이 2배**가 됐다. ⇒ **교훈: 재귀 추출기는 항상 독립 수단(정규식·전수 덤프)과
   교차검증하라.** C238에서도 같은 계열의 도구 결함을 겪었다(전체-단어 매칭이 정규식 가지를 놓침).

1. **"흡수율 50%"는 아티팩트였다.** 초판은 `t2_compute.py` **하나만** 수확해서 7/14가 미커버로
   나왔다. `argmax_where`·`disjoint`·`equal_len`은 `gate_interpreter.py`에, 산술 3종은
   `bank_eplan_controller.py`에 **이미 구현**돼 있었다. ⇒ **교훈: "미구현"을 선언하기 전에 전 모듈을
   grep하라.** 커버리지 분모/분자를 한 파일로 잡으면 없는 결손을 만든다.
2. **"라이브 버그 확정"은 성급한 단정이었다(철회).** `apply_op`이 `None`을 반환하는 것을 보고
   라이브 결함이라 단정했으나, **호출부를 추적하니** 라이브 경로는 `scaffold_get_tools`(8 GET
   도구·op 18종)이고 **그 18종은 전부 `t2_compute.py`에 있다**. 산술 스펙은 E-PLAN 컨트롤러용이며
   그 경로는 자체 인터프리터로 정상 동작한다. ⇒ **[[08]] 위반**(집계·단발 관측에서 결론 직행).
   정확한 진술은 **§6-P1의 조건부 위험**("통합 시 11 op 미이식이면 E-PLAN 계산 경로가 조용히 죽는다")
   이며, 현 상태는 결함이 아니다.

## §5. C235 TODO 답변

### ① callback 인터페이스 목록 — **현재 공집합**
미커버 op 0이므로 **지금 필요한 callback 구현체가 없다**. 이는 §13e의 "callback이 필요한 잔여 =
공용 primitive로 표현 불가한 도메인 고유 산식"이 **현 3도메인에서는 비어 있다**는 뜻이다.
⇒ 특허/논문 문구는 "callback을 둔다"가 아니라 **"둘 수 있으나 현 실측에서는 요구량 0"**이 정확하다.
(단 §8 한계 — 이건 상한 증명이 아니라 현재 회계다.)

### ② 공용 primitive 목록 — **39 op (단 3분열)**
`add`·`argmax`·`argmax_where`·`argmin`·`argmin_where`·`bool_expr`·`bucket_month_window`·`case`·
`catalog_filter`·`clamp`·`compare`·`const`·`count`·`count_field_matches`·`count_where`·
`date_between`·`date_in_window`·`days_between`·`diff`·`disjoint`·`divide`·`equal_len`·`filter`·
`group_reduce`·`if_then`·`lookup`·`lookup_table`·`match_verdict`·`match_verdict_grounded`·`max`·
`min`·`most_recent`·`multiply`·`ref`·`round`·`select_discrepant`·`str_eq`·`subtract`·`sum`

[[16]] §8 상한("태스크-정답-특정 도구 금지") 대조: 39개 전부 **일반 연산·집계·날짜·비교**이고
태스크 정답을 내장한 것은 없다. `select_discrepant`·`match_verdict_grounded`는 이름이 특수해
보이지만 각각 "행 집합에서 불일치 행 선택"·"제시값 vs 기록 일치 판정"이라 도메인-일반이다.

### ③ EXT 재판정 — §3 표. **C_ext의 실행-코드 성분 = 0**

## §6. 처방 (설계·리뷰 대상 · 미구현)

**P1. 인터프리터 통합** — [[03b]] "술어 이중화 금지"의 직접 적용. `t2_compute.py`를 단일 정본으로
하고 결손 11 op를 이식, `gate_interpreter.py`·`bank_eplan_controller.py`는 그것을 호출한다.
- ⚠**선결**: 이식 없이 통합하면 **E-PLAN 계산 경로가 조용히 죽는다**(`apply_op`이 실패를 `None`으로
  반환하는 설계라 미지원 op와 "계산 불가"가 구분되지 않는다). [[14]] E-PLAN 배선 전 반드시 처리.
- **회귀 위험**: 동일 op 이름이 모듈마다 **다른 의미**일 수 있다(예: `count` vs `count_where`).
  이식 전 op별 의미 대조표를 만들고, 3 모듈의 기존 selftest를 통합 후에도 전부 통과시킨다.

**P2. 미지원 op 침묵 제거** — `apply_op`이 미지원 op를 만나면 `None`(무음) 대신 **경고 마크**를
남긴다. 무음 실패는 C238의 "죽은 중복"·본 조사의 오진을 둘 다 낳은 구조적 원인이다.

**P3. `a2/split` 재생성** — C234 화이트리스트 반영(§3 분류 불일치 해소).

**P4. docstring 동기화** — `t2_compute.py`가 지원 op를 15개로 적었다(실제 28). 문서-코드 괴리가
"미구현" 오판의 재료가 된다.

## §7. C_ext 회계 (capex/opex · C237 프레임)

| 항목 | 값 | 근거 |
|---|---|---|
| **capex**(1회·전 도메인 공유·증분 0) | 인터프리터 + **39 op 구현** | 도메인 무관 재사용 |
| **C_ext 실행코드**(도메인당) | **0** | 흡수율 100%(28/28)·미커버 0 |
| **C_ext 선언**(도메인당) | EXT 기준 banking 8 op-노드 / retail 4 / airline 0 (A2 전체로는 banking 22 op종·retail 6) | A2에 식을 쓰는 비용만 |
| **C_cb**(callback 구현체) | **0** | 현 3도메인 실측 |

⇒ C237의 "유한성 = opex 유계성의 다른 이름"에 대한 **정량 근거 1건**: E1 성분의 도메인별 opex는
**선언 비용뿐이고 코드 비용은 0**이다. 단 이는 **3도메인 회계**이며 상한 증명이 아니다(§8).

## §8. 한계 (정직 기록)

1. **상한 증명이 아니다.** 흡수율 100%는 "현 3도메인이 실제로 선언한 op" 기준이다. 새 도메인이
   미지원 연산(예: 비선형 금융 산식·확률 계산)을 요구하면 그때 callback이 필요하다. 이 수치를
   "어떤 도메인도 callback을 요구하지 않는다"로 인용하면 **C233이 이미 교정한 유한성 과잉주장의
   재발**이다.
2. **op 종류 기준**이다. 사용 빈도·복잡도 가중이 아니다(빈도는 §3 참고). 또 흡수율 분모는
   **A2 전체 op 선언**이며 EXT-only가 아니다(§3 주석).
3. **수확이 정규식**이다. op을 dict 디스패치나 `getattr`로 처리하는 구현은 놓친다(현 3모듈은 모두
   `if op == ...` 형태라 해당 없음을 확인).
4. **airline은 E1/E2가 아예 없다**(`identifying_arg_types` 1키뿐). 3도메인 중 계산 스펙을 가진
   것은 banking·retail 둘이므로, 흡수율의 실효 표본은 **2도메인**이다.

## 부록 — 재현

```bash
cd scripts/distill/tau2 && PYTHONIOENCODING=utf-8 py -3 x6i_ext_primitive_absorption.py --json out.json
```
