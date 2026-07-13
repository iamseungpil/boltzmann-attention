# COVERAGE_LOOP_DESIGN — coverage = FIND-subset 루프 통일 (walk 폐기·in-flight step-loop)

> 2026-07-13. 상위: `GENERALIZED_SCAFFOLD_ARCHITECTURE_2026_07_12`(LOCK·GET→FIND→ASK) · `A1_V3_DESIGN_2026_07_13`(§1 COVER·L9) · `A1_REGRESSION_PERSTEP_FORENSIC_2026_07_13`.
> **역반영 완료**: 본 문서의 루프+가드 전체 세트가 `A1_V3_DESIGN` v3.1 개정(헤더·§0·§1·§2b COV/L10/L4-tie/L11/L12·§3·§5·§6·§8d/§9c 철회)으로 편입됨 — 구현 순서·조합은 그쪽이 정본.
> 근거 데이터: `sim_results/genv3_probe.results.json.gz`(A1-v3 18-probe·per-step 전수 포렌식·이 문서 §1).
> 불변: [[05]] 도메인일반·A2만 · [[10]] 선택기/검증기=결정론·LLM=formalize만 · Δspurious≤0 계측 · gold-independence(리마인더에 DB내용 0).

## 0. 결론 (한 줄)
**T2_EPLAN_WALK(개수-기반 stop-time walk) 폐기. coverage는 이미 LOCK된 GET→FIND(fexec)→ASK 루프의 cardinality 일반화(one→subset)로 흡수 — in-flight·후보별 step-loop·내용(predicate) 기반.** 새 온톨로지 아님: disamb(choose-one)와 coverage(choose-subset)는 "getter가 후보 ≥2 surface" 같은 상황의 두 terminal semantics.

## 1. 포렌식 재분류 — "coverage-miss 8"은 집계 착시 [M]
genv3_probe 8개 실패를 per-step 전수(gold action_checks vs 실행 tool_calls vs 대화)로 재분류:

| task | 집계 라벨 | per-step 확정 원인 | 재분류 |
|---|---|---|---|
| t41 | coverage-miss | 주소수정 1 intent가 주문 2개 span(#W9583042 누락) | ✅**진짜 coverage** |
| t81 | coverage-miss | 취소 대상 2주문 중 1개만(#W9722559 누락) | ✅**진짜 coverage** |
| t92 | coverage-miss | 반품 2주문 중 1개만(#W3239882는 read조차 0) | ✅**진짜 coverage** |
| t35 | coverage-miss | user 오주장(laptop∈#W8528674)을 grounded 데이터보다 우선→틀린 도구(create_new_order) | ❌ selection-binding |
| t64 | coverage-miss | 복합기준(방수∧가격≤지불∧argmax해상도) 변형품 오선택 | ❌ L4b 복합 criterion |
| t76 | coverage-miss | 2번째 cancel reason을 1번째서 복사("ordered by mistake"≠gold "no longer needed") | ❌ 인자 carryover |
| t97 | coverage-miss | 주소 날조("123 Broadway")→user-confirm 세탁. gold 주소는 미조회 주문 #W3407479에 실재 | ❌ 누락 GET+날조(t96형) |
| t54 | coverage-miss | **v3 인스턴스 = infra**: user-sim이 known_info에 없는 "black size 8" confabulate(DB 유일 부츠=size 12)→동정 교착→user 요청 transfer→성공 후 hold-loop 149msgs 소진. set-op formalize 자체는 성공(msg46). v2 arm은 부분실행+cancel 1 누락=L9 coverage-miss | ⚠️ infra-noise(v3)·L9 가족(v2)+종결가드 |

**⇒ coverage 레버의 진짜 타깃 = 3개(t41·t81·t92). 나머지 5 = 별개 레버(§7)·coverage 작업에 혼입 금지([[03]]).**
(핸드오프 2026-07-13 §4①의 "coverage-miss 8" 표기는 이 표로 교정.)

포렌식 방법 명세([[08]]): 종료사유 분포 = user_stop 7·max_steps 1·crash/infra 0(노이즈 배제). 분류 근거 = 결정론 신호(gold action_checks의 match=False 항목 vs 실행 tool_calls diff·인자 값 대조·궤적 정독 8/8) — pass-집계 아님. **단 nt=1 단일 궤적 귀속**: 원인 *유형* 분류는 결정론이나 태스크별 재현성은 [D] — A1-v2 nt4 집계(진행 중)와 표적 probe로 [M] 승격.

### 1b. 대조군 교차표 — A1-v2 arm vs A1-v3 probe 실패 signature [M]
| task | arm 간 signature | 판정 |
|---|---|---|
| t92 | **동일**(22msg·같은 단일 return·#W3239882 무접촉 양쪽) | coverage 실패 arm-불변·재현성 높음 |
| t64 | **동일**(같은 오선택 6117189161 양쪽) | 복합-criterion 실패 arm-불변 |
| t97 | **동일 날조값**("123 Broadway"/10007 양쪽 — 모델 prior의 전형 NYC 주소·temp0) | 날조 arm-불변·결정론적 재현 |
| t35 | 코어 동일(랩탑 modify 누락 양쪽·v3만 create_new_order 추가 시도) | binding 실패 arm-불변 |
| t41 | 패턴 동일(#W9583042 주소 누락 양쪽·v3는 modify_user_address 추가=부분개선) | coverage 실패 stable |
| t81 | **패턴 동일·인스턴스 flip**: 두 cancel 중 정확히 1개만 실행이 양쪽 공통·*어느 쪽*인지는 arm 간 뒤바뀜(v2=#W9722559만·v3=#W3289292만) | "하나만 하고 종료" 패턴=구조적·선택=stochastic |
| t76 | **성격 이동**: v2=2번째 cancel 자체 누락(=coverage-miss)·v3=2번째 cancel 실행하되 reason carryover | coverage 루프가 write를 끌어내면 잔여=enum-인자 — 레버 2개 다 필요 |
| t54 | **★교정**: v3 "gold action_checks 비어있음"은 **max_steps 아티팩트**(채점 미수행·"Simulation terminated prematurely") — 실제 gold(d["tasks"])는 write 3+communicate 1(cancel×2·return×1·환불 $3,646.68). v2는 부분 실행(cancel #W7342738 누락=coverage-miss)·v3는 user-sim confabulation("black size 8" 날조·§7)으로 경로 자체가 다름 | 태스크 성격=L9 가족·v3 인스턴스=infra |
**교차표 함의**: ① 실패 7/8이 양 arm 동형 재현(순수 nt1 노이즈 아님 — §1 caveat 완화)·t54만 arm 간 경로 상이(infra 개입) ② t76은 coverage 루프+enum 가드 복합 타깃 ③ t97 동일 날조값·t64 동일 오선택 = temp0 결정론 재현 — 레버 없으면 nt 늘려도 안 닫힘.

## 2. walk 사인(死因) 3가지 [M]
walk은 로그상 **7회 발화**(죽지 않았음)·그럼에도 0/8:
1. **stop-time**: user가 "###STOP###"으로 대화를 닫은 *후* 리마인더 → 재개 불능(t81·t92 궤적 꼬리 확인). advisory 한 줄로 닫힌 대화 못 연다([[42]] prompt-ceiling과 동형).
2. **개수 semantics**: `요청수량 N vs write수 M` 비교는 (a) t41형("한 intent가 2 record span"·write 3회≥N=2→gap 미탐) 원리적 미탐 (b) qty 과-파싱 스푸리어스 실재(`walk gap: qty=8 executed=0`·`qty=9` — 요청에 없는 수량). 게이트 자신의 over-action 위험(등대 모트).
3. **advisory(비강제·글로벌)**: 발화한 케이스에서도 32B가 무시. 강제 채널(deny/regen)이 아닌 권고 주입은 통제 아님([[07]] control-not-prompt).

## 3. 설계 — coverage = FIND-subset step-loop (in-flight)
### 3a. 통일 원리
```
getter가 후보 ≥2 surface (user orders, product variants, …)
 ├─ selection(disamb): 정답 cardinality=1 → GET→FIND(fexec)→ 1?use : ≥2?enumerate-ASK   [기존·작동]
 └─ coverage:          정답 cardinality=부분집합 → 같은 루프, terminal만 "M 전원에 write"  [본 설계]
```
formalize 출력에 **`cardinality: one | subset | all`** 1필드 추가가 통일의 전부. L9 complement("X 빼고 전부")는 `all − predicate`의 특수형.

### 3b. 루프 (결정론 controller가 소유·[[10]])
```
트리거(결정론): record-type R에 write 의도 등장 ∧ R의 enumerate된 후보 ≥2
 0. FORMALIZE (LLM 1회·sim당 intent별 캐시): 요청 → {predicate, cardinality}
    · "그 두 스케이트보드 반품" → {item.name~skateboard ∧ order.status=delivered, subset}
    · "#W123 취소"(명시 단일 ID·복수 단서 0) → {id=#W123, one} → 개입 0  ← Δspurious 제1가드
 1. cardinality ∈ {subset, all}이면 step-loop:
    for c in 후보 − examined (한 번에 하나·K-read cap):
       deny+regen 피드백 = "read c의 details 먼저" (기존 L2 채널·read만 강제·[[05]])
    → 전부 읽히면(또는 cap) 엔진이 predicate를 grounded records에 평가(execute_formalized) → M
 2. predicate 형식화 불능 / M=∅ → enumerate-ASK 낙하 (LOCK §4c·후보 나열+사용자 선택·1회 cap)
 3. write 단계: M 전원 대상. 각 write는 기존 confirm 게이트 불변. ledger가 M∖acted 추적.
 4. 백스톱(구 walk 자리): 종결 시도 시 gap = M∖acted (결정론 set-diff·내용 기반).
    gap≠∅면 gap 멤버 나열 리마인더 1회 — qty 파싱 없음 → §2② 스푸리어스 원천 소멸.
```
### 3c. walk 대비 기전 대응 (사인별 해독제)
| walk 사인 | 본 설계 |
|---|---|
| stop-time(닫힌 대화) | **in-flight**: 첫 write 의도 시점에 M 고정·즉시 걷기 시작 |
| 개수 세기(N 파싱) | **내용 매칭**: predicate를 grounded 후보에 결정론 평가. qty 파싱 폐지 |
| advisory 글로벌 권고 | **step-loop 단일 명령 + deny/regen 강제 채널**: "다음: c 읽어라" 후보별 micro-스텝. disamb가 같은 채널로 over-block 67%→0 실증(C44-C48). 글로벌 coverage 판단→후보별 membership 판단 분해 = load-reduction([[45]]) |

### 3d. A2 (도메인일반·[[05]] 체크)
- 기존 필드 재사용: `list_enumerator`(GET)·`detail_reader`(read 스텝)·`items_key`. **신규 필드 불요** — cardinality는 formalize(LLM) 산출이지 A2 아님.
- 엔진 리터럴 0: predicate 평가=`execute_formalized`(기존)·후보=record dict(grounded). 전이=ABox-swap 그대로.

### 3e. Δspurious 가드 (등대: 게이트 자신도 over-action)
1. 명시 단일 ID + 복수 단서 0 → cardinality=one → **개입 0** (제1방어선).
2. K-read cap(기본 6)·ASK cap 1·formalize retry ≤2 (t54형 thrash 방지).
3. write 강제 절대 없음 — read 지시·ASK·set-diff 리마인더까지만([[05]]).
4. 가드 태스크 고정 계측: t58(qty-conflation)·t32(over-action)·t27(품목-급 수량) — **Δspurious≤0 미달 시 레버 기각**.
5. over-ask 비용 [?]: enumerate-ASK 낙하 빈도 실측 항목(LOCK §4c caveat 승계).

## 4. 진짜 3케이스 예측 [D — probe로 [M] 승격 대기]
- **t92**: predicate("skateboard item") 평가가 각 주문 read를 강제 → #W3239882 발견·M에 포함 → 반품 2건. (현 궤적: 2번째 주문 read 0 — read 강제 자체가 해독제.)
- **t41**: "주소를 잘못 입력한 것 같다" → predicate(pending ∧ address=옛주소) → M={#W9583042,#W4082615} 둘 다 수정.
- **t81**: "안 쓰는 것들"=형식화 불능 → enumerate-ASK("pending 주문 4개: … 어느 것?") → user-sim이 #W9722559 특정 기대.
- 보너스 **t35**(오귀속이지만 같은 기계): predicate("17-inch laptop")를 두 주문 grounded items에 평가 → #W9672333 (user 오주장 무시) — selection-binding도 FIND 기계가 그대로 해결. §7.

## 5. 구현 스케치 (기존 접점 재사용·신규 표면 최소)
- **제거**: `t2_eplan_patch.apply()`의 walk wrap(`_check_termination` 패치)·`walk_required_n`/`qty_item_covered`/`cp5_gap_reminder`의 qty 경로. env `T2_EPLAN_WALK` 폐기 → **`T2_COVERAGE_LOOP=1`** 신설.
- **트리거·스텝 배선**: `t2_gate_patch.unified()`의 write-인터셉트(disamb와 같은 자리·1083행 패턴). 후보/examined는 기존 `build_ledger_from_messages` ledger. read-지시 deny=기존 `discovery_precondition`/`l2_feedback` 채널 확장(대상 c 1개 명시로 좁힘).
- **FORMALIZE·평가**: `t2_formalize_exec.build_formalize_prompt`+`parse_formalize`(cardinality 필드 추가)+`execute_formalized`(subset 반환 지원 — 현 argmax/filter에 "matching all" 모드).
- **ASK 낙하**: 기존 DISAMB enumerate 피드백(`DISAMB_ENUM_FEEDBACK`) 재사용.
- **백스톱 set-diff**: walk의 종결 후크 자리 재사용하되 입력이 M∖acted(결정론)로 교체.

## 6. 검증 계획 (전부 무료 先·[[09]])
1. **오프라인 replay [0원]**: genv3_probe·genv2_a1v2 기록 궤적의 getter 출력에 predicate 평가 — "M을 올바르게 산출하나"(t41/81/92 + 가드 t58/32/27). formalize는 로컬 Qwen 격리 프롬프트.
2. **formalize 정확도 probe [0원]**: 3케이스 요청문 → {predicate, cardinality} 변환 정확도(로컬).
3. **표적 라이브 probe [소액·승인 후]**: 3 coverage + 5 가드 + help-8 무회귀 = ~16 task nt1.
4. full 실측은 3 통과 후 별도 승인.

## 7. 5개 오귀속 배정 (병렬 per-step 포렌식으로 검증 — t97/t35/t76 확정 [M]·t64/t54 대기)
| task | per-step 확정 원인 | 레버 | scaffold/learn |
|---|---|---|---|
| t97 | **확정**: user 재공유 거부(msg23)→tool call 0회로 "123 Broadway" 날조(msg24·전수 grep: 이전 어떤 tool 출력·user 발화에도 0회)→user-sim yes 세탁(msg25)→DB 오염. gold 주소 경로(#W3407479 GET)는 orders 리스트에 grounded인데 미호출 | **L3 origin-prov 결정론 차단 가능 확정**: write 인자(주소 문자열)가 이전 tool 출력∪user 발화에 verbatim 부재→deny. 차단 후 L2 fetch-first(#W3407479 GET)로 교정 유도. 의미 이해 불요 | **scaffold(순결정론)** |
| t35 | **수정**: "user 주장 일괄 추종" 아님 — **선택적 grounding 실패**. 같은 tool 출력(msg23: laptop 1684786391∈#W9672333·pending)으로 스피커 오매핑은 반박 성공, 랩탑 매핑은 무검증 수용. 미조회 주문 status를 pending으로 날조(msg24·실제 delivered)→모순 바인딩(item 1684786391 ∉ #W8528674.items)→가짜 오류 프레이밍(msg36·modify 시도 0회)→create_new_order(미존재 tool·Error). 대체 item 선정은 gold 일치 — 틀린 건 오직 주문 바인딩 | **바인딩 멤버십 검사 결정론 차단 가능 확정**: write/plan의 item_id는 대상 주문의 최신 get_order_details.items 멤버여야 함→위반 deny. 올바른 주문 = 이미-fetch된 주문들에서 문자열 매칭(기계적) | **scaffold(순결정론)** |
| t76 | **확정**: 2번째 cancel reason을 1번째서 관성 복사. user 동기(msg51 "maple이 없어서")에 mistake 계열 0건(전수)·gold="no longer needed". yes/no 확인(msg53)이 오히려 잘못된 값을 승인 통로로 세탁 | **탐지=결정론·값 결정=의미**: (인자 미attested ∧ 직전 write 인자와 동일)→carryover 플래그. 처방 = yes/no 확인을 **개방형 ASK로 강등**("실수 주문인가요, 더 이상 필요 없으신가요?") — yes/no는 user-sim이 rubber-stamp함이 실증됨 | scaffold(탐지+ASK강등)·값은 INFER |
| t64 | **가설 수정**: formalize는 정확했음(msg14 "waterproof·highest res·≤$502.28" 언어화 정확·필터도 옳음). 실패 = **선택 단계 first-fit satisficing**: predicate 만족 후보가 {6700049080($466.75), 6117189161($481.50)} **4K 동률 2원소**인데 열거·비교 없이 첫 후보 단수 제시. COMPUTED FACTS가 "cheapest available=6700049080"를 명시 주입했는데도 무시(advisory 무력·[[42]] 재실증) | **결정론 완결 가능**: predicate 기계평가(스키마 필드만) + **동률 tie-break(min price=환불극대)** 규칙으로 유일해=gold. caveat [D]: tie-break 규칙의 gold-정렬은 이 태스크 1건 근거·A2 MENU 선언 형태로([[05]]) | **scaffold(전수평가+tie-break)** |
| t54 | **가설 대부분 반박**: set-op formalize는 성공(msg46 complement 실행안 정확 언어화)·transfer는 user 명시요청 순응. 1차 원인 = **user-sim confabulation**(known_info에 없는 "black size 8"을 날조·DB 유일 부츠=size 12 → 동정 교착·**시뮬 인프라 결함**). 2차 = transfer 성공 후 **hold-loop 149/201 msgs(74%)** 무행동 소진. (v2 arm에선 같은 태스크가 부분실행+cancel 1 누락 = L9 coverage-miss로 실패 — 태스크 성격은 L9 가족·v3 인스턴스는 인프라 노이즈) | ① 인프라: user-sim confab은 에이전트 레버로 gold 비보장 — **집계서 infra-noise로 분리 표기** ② **transfer-후 종결 가드**(transfer 성공 tool 결과 후 세션 종결·또는 연속 k회 무-tool 근사중복 발화→종료) = 순결정론·비용 74% 절약(reward 회복과는 무관) | scaffold(종결 가드)+**infra 분리** |
**⇒ "5개=learn행"은 아니다 — 오히려 5개 중 learn 몫이 거의 없다.** t97(provenance)·t35(멤버십)·t64(전수평가+tie-break)·t54(종결 가드) = 순결정론. 의미(INFER/learn)가 남는 곳은 t76 enum 값 결정 하나. 단 t54 reward 회복은 user-sim 결함에 막힘(레버 무관).

### 7a. ★교차 패턴 (t97·t35·t76 공통·신규 발견 [M])
1. **가짜 오류/정책 프레이밍**: 3태스크 모두 직전 tool call 0회인데 "I encountered an issue/policy requirement"로 자발 결론을 포장(t97 msg24·t35 msg36·t76 msg52). 날조/carryover 값의 **운반 템플릿**. → 결정론 탐지 후보: "오류 주장 ∧ 직전 tool call 없음" = 날조 신호.
2. **확인-세탁(confirmation laundering)**: user-sim은 에이전트가 제시한 구체 값을 검증 없이 yes 승인(t97 msg25·t76 msg53). **yes/no 확인은 ungrounded 값의 방어가 아니라 승격 장치** — §3 ASK 설계에 반영: 에이전트-제시값 확인은 개방형 ASK로(값을 user가 생산)·에이전트가 값을 제안하는 yes/no 금지 후보.
3. **가드 스펙트럼**: 반증 데이터가 이미 컨텍스트에 있으면(t97 주소·t35 멤버십) 순결정론 차단. 값 생산에 의미가 필요하면(t76 enum) 탐지→ASK 강등까지가 결정론 몫 — [[10]] 분담 그대로.

## 8. 열린 문제
- over-ask/over-read 비용 실측(§3e-5) — Δspurious 계측과 함께 probe 항목.
- formalize가 cardinality를 안정 출력하는가(subset vs one 오판 = 잠재 스푸리어스) — §6-2가 선행 관문.
- t81형 "형식화 불능→ASK"의 user-sim 응답 품질(특정 안 해주면 gap 미해소) [?].
- t64 tie-break(동률→min price) 규칙의 gold-정렬 근거가 현재 1태스크 [D] — 타 태스크/도메인 probe로 검증. A2 MENU 필드(`tie_break`) 선언 형태 검토.
- t54형 user-sim confabulation = 집계 오염원 — **집계 규약에 infra-noise 분리 표기** 필요(에이전트 레버 평가에서 제외).
- transfer-후 종결 가드·근사중복 발화 종결 가드 = 비용 레버(74% 절약)·reward 무관 — 구현 우선순위 별도.
