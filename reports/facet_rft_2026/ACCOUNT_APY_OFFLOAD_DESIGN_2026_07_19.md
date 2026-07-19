# ACCOUNT-APY OFFLOAD 설계 (D+G군 L2 계산층 대책 · 2026-07-19 밤)

> 계기: 사용자 지시 — D+G(APY 최적화·조사 ~22태스크) per-step 전수 포렌식(62 미스 diff) 후 *"대책을 미리 만들어 둬야겠다"*.
> 앵커: `RATE_SUBAGENT_DESIGN §2b~2e`(rate-formalize keystone·021 라이브 1.0)·[[16]] GET/FIND/INFER/ASK·
> [[10]] 분담(LLM=생성기·엔진=결정론)·§2m cue overload(다후보 조항-결합 간섭)·UNIFIED_TAXONOMY(계산·참조-기준형).

## 0. 문제 정의 (실측 [S] · bank_all97_nt1_v2 D+G 14궤적 62미스 전수 diff)
- **L1 조립(발견/계획) = 미스 55/62** — unlock/call 체인 미진입. **T3 보강(e3a8654f)의 표적** — 본 설계 scope 밖.
  (batch_a2(057·064~069)가 T3-스택 L1 닫힘율의 첫 측정.)
- **L2 계산/선택 = 7미스이나 결정적** — L1을 뚫은 궤적이 죽는 지점. 본 설계의 표적:
  1. 계좌-선택 오류: 055 `Silver Plus`←`Green` · 066 `savings+Green/Evergreen`←`checking+Bluest` · 071 `Gold Saver`←`Bronze Saver`
  2. 카드-추천 오류: 059 `Silver`←`Gold` · 063/066 `Silver`←`EcoCard` (=023 사인·T3가 레버 버그는 픽스)
  3. 이자 보정액 산술: **095** `expected_apy 6.85←3.35`·`amount_difference 98.0←0` (조사 체인 6/9 통과 후 사망)
- 도메인 사실(KB doc_..._046 등·probe 정독): **boost 스택 규칙** — checking-boost는 최고 1개만(비스택)·
  credit-card APY bonus 최고 1개·relationship/tier bonus는 스택. 즉 올바른 APY = base + max(checking_boosts)
  + max(card_bonuses) + Σ(스택형) — **조항-결합 비교 = LLM이 눈대중으로 지는 지점**(ratefix 이전의 rate 오독과 동형).

## 0b. 설계 리뷰 반영 (2026-07-19 밤·사용자 리뷰 7건 — ①④ 필수 반영 완료·②③⑤⑥ 스펙 수정·⑦ 상설)
①: 신규 op을 `apy_argmax`/`interest_delta`(도메인-리터럴 이름·엔진 컨벤션 첫 위반) → **범용 프리미티브 1개
`group_reduce`** + 기존 op 합성으로 교체(§2). ②: `days` 파라미터 폐기 → `period_start/end` 복사+엔진 `days_between`.
③: `base_apy_source` grounding 대칭 요구. ④: kind 오분류=grounding 사각 → 프로브 별도 축 + unknown-kind 명시
플래그(silent drop 금지). ⑤: `constraints`=에코-전용으로 명시. ⑥: 무회귀 결론은 유료 대조런만(오프라인=참고).
⑦: [[05]] 3질문 답 §1에 상설(이하)·**향후 모든 설계서에 3질문 섹션 상설**(두 연속 누락→[[07]] 논리로 관례 불가).

### [[05]] 3질문 (상설 섹션)
- **Q1 A2 순증? Yes** — GET 도구 2 + `stack_rules`. 정당화 = §4 측정 게이트. `stack_rules`는 KB 사실의 A2 복제본이므로
  **§4-0 gold census가 복제 정합성 검증을 겸한다**(KB 조항 ↔ stack_rules 대조를 census 항목에 포함).
- **Q2 유동성 동결? Yes — 의도적** — 조항-결합 산술 = concrete → [[10]]상 offload 정당. 근거 = 눈대중 패배 7미스 실측
  (055/059/063/066/071/095) + fresh 097(8/18·계산층 사망).
- **Q3 도메인 행동 수행? No** — 후보 발굴·조항 해석 = LLM(서브), 엔진은 선언된 combinator 산술만.

## 1. 분담 설계 (ratefix keystone 완전 동형 · [[05]]/[[10]]/[[03b]])
```
LLM(격리 서브·INFER):  후보별 "APY 구성요소" formalize — base_apy · boost 후보(종류·값·출처 조항)
                        ※ 눈대중 합성 금지 — 구성요소만 낸다 (ratefix의 base_rate/promo 동형)
엔진(결정론·GET 반환):  스택 규칙 적용(max-per-종류 + Σ스택형) · 후보 비교(argmax) · 보정액 산술
                        ※ 도메인 상수 0 — 스택 규칙의 *구조*(어느 종류가 max-1인지)도 A2 선언
게이트(기존 재사용):    write(open/close/apply_credit/submit_report)의 인자 검증은 기존 스택
                        (provenance·reference_filter·WEV-류) — 본 설계는 값 *생산* 측만
```

## 2. A2 스키마 (scaffold_get_tools 확장 — **엔진 신규 프리미티브 1개**·나머지 전부 데이터/기존 op 합성)

### 2-0. 엔진 신규 프리미티브 `group_reduce` (리뷰① 반영 — 도메인-일반·합성 가능)
```
group_reduce(over=<list>, group_by=<field>, reducers={<group값>: "max1"|"sum", ...},
             unknown_policy="flag", value_field=<field>) → scalar + {flags}
```
- 항목을 `group_by` 필드로 묶고, 그룹별 A2-선언 reducer(max1/sum) 적용 후 총합. 이름·상수 도메인 리터럴 0 —
  기존 `argmax·diff·multiply·days_between·const`와 표현트리 문법으로 합성:
  - **effective APY** = `base + group_reduce(boosts, by=kind, reducers=stack_rules)`
  - **계좌-선택** = 기존 `argmax`(over=candidates, key=위 합성)
  - **보정액** = `multiply(diff(effective, applied), principal, div(days_between(period_start, period_end), day_basis))`
    — `day_basis`(365)도 A2 파라미터.
- **unknown-kind 정책(리뷰④)**: `reducers`에 없는 kind 존재 시 **silent drop 금지** — 해당 boost 미합성 + 반환에
  `[UNKNOWN-KIND: <k> — not composed; verify manually]` 플래그를 낸다([[03b]] 인접 회피).

### 2a. `get_best_account_option` (계좌-선택·055/066/071 + 최적화군 057/064~069 표적)
```json
{"name": "get_best_account_option",
 "description": "USE THIS when the customer wants to choose/open the account (or account+card combo) that maximizes APY. Pass the candidate options you found in the KB with their APY components; it applies the declared stacking rules deterministically and returns the best option with the effective APY.",
 "params": {"candidates": "JSON array: [{option: <name>, base_apy: <number>, base_apy_source: <verbatim doc quote>, boosts: [{kind: 'checking'|'card'|'relationship'|'tier'|..., value: <number>, source: <verbatim doc quote>}...]}]",
            "constraints": "ECHO-ONLY: the user's stated hard constraints, copied from the conversation. NOT applied by this tool — returned back so YOU verify them against the winner."},
 "return_template": "Best option by effective APY under stacking rules, among {n} provided candidates: {result}. Per-candidate: {details}. Constraints (verify these yourself against the winner — this tool only compared APY): {constraints}",
 "op": {"op": "argmax", "over": "candidates",
        "key": {"op": "sum", "of": ["r.base_apy", {"op": "group_reduce", "over": "r.boosts", "group_by": "kind", "value_field": "value", "reducers": {"checking": "max1", "card": "max1", "relationship": "sum", "tier": "sum"}, "unknown_policy": "flag"}]}}}
```
- grounding(리뷰③): `base_apy_source`·`boosts[].source` **둘 다** 원문 인용 요구 — §2e `_norm_ground` 검증·탈락 시
  해당 구성요소 드롭+플래그·전멸 시 폴백. `constraints`는 **에코-전용**(리뷰⑤) — 엔진 필터 아님을 반환문이 자백.
- return_template의 "among {n} provided candidates"(리뷰-경미): 도구는 **비교만** 고친다 — 후보 발굴(L1) 누락에
  과신을 입히지 않도록 반환문 자체가 경계를 드러냄.

### 2b. `get_interest_correction` (보정액·093~097 표적)
```json
{"name": "get_interest_correction",
 "params": {"principal": "<조회값 복사>", "period_start": "<조회값 복사(문자열)>", "period_end": "<조회값 복사>",
            "correct_components": "2a와 같은 components 스키마(sources 포함)",
            "applied_apy": "실제 적용된 APY — 계좌 조회 출력에서 복사"},
 "op": {"compose": ["group_reduce(correct_components)→effective", "diff(effective, applied_apy)",
                    "multiply(principal, days_between(period_start, period_end)/day_basis)"], "day_basis": 365},
 "return_template": "correct APY {apy_correct}% vs applied {apy_applied}% → interest difference over {days} days (computed from the given period): {amount}"}
```
- **`days` 파라미터 폐기(리뷰②)**: 일수 세기는 LLM 약점(095가 정확히 이 태스크) — `period_start/end` **문자열 복사**를
  받아 엔진 `days_between`이 계산. `applied_apy`는 grounding 미요구 — **조회 출력에 실재하는 값의 복사이므로
  기존 fab-provenance(ctx-실재 검사)가 커버하는 축**(위험 수용 근거 명시).

### 2c. 추천 합류 (엔진 변경 0)
- `recommendation_verify`(T3서 research_tool 픽스됨)의 formalize 프롬프트가 **2a 반환을 근거로 인용**하도록
  research 경로에 `get_best_account_option`을 권고(우리 도구는 이미 도구목록 주입됨) — 눈대중 추천 차단.

## 3. 함정 → 제약 (전부 선례 실측)
| 함정 | 제약 |
|---|---|
| §2d 결함1: 후보 전체 일괄 formalize = 부하 재생산(172K 전례) | 후보별/문서군별 격리 + max_batch cap (카드→카테고리 격리 선례) |
| §2d 결함2: formalize 환각(잘못된 boost 값) | temp 0 + `boosts[].source` **원문 인용 grounding**(_norm_ground 재사용·탈락 시 해당 boost 드롭·전멸 시 폴백) |
| [[03b]] 엔진-formalize 금지 | 엔진은 숫자 합성만·KB 파싱 0·후보 발굴도 LLM(서브) 몫 |
| 게이트 자신의 역효과(등대 모트) | 새 GET 도구 주입이 기존 흐름을 방해하는지 Δspurious 계측(주입만으로 성립하는 T2_SG_EXCLUDE 대조 지원됨) |
| 026 gold버그 선례(벤치 자체 오류) | gold 검증 census를 대책 *前* 1회: D+G gold의 APY 값이 자기 정책과 일치하는지(§5-0) |

## 4. 측정 계획 (무료 먼저 · [[09]])
0. **gold 자기일관 census**: D+G gold 인자(APY·account_class)를 KB 정책으로 재유도 — 벤치버그 분리 +
   **stack_rules ↔ KB 조항 정합성 검증 겸임**(§0b Q1).
1. **formalize 프로브**(GPU0·rate-프로브 동형·`bank_apy_formalize_probe.py`): 055/066/071/095의 실제 후보·계좌로
   — 축 3개: ①값 정답률 ②grounding 탈락률 ③**kind 분류 정확도(리뷰④ — grounding이 못 잡는 판단 지점·별도 축)**.
   기준: ratefix 선례(카테고리 격리서 오독 0)와 동급.
2. **엔진 단위테스트**: `group_reduce`(max1/sum·unknown-flag) + 합성식이 gold 값 재현(095의 6.85/5.625/98.0 등).
3. **유료 스모크(승인 후)**: L2-실패 대표 2태스크(095·066) 단일변수(`T2_SG_EXCLUDE`로 도구 유/무 대조).
4. **무회귀(리뷰⑥ 정정)**: 도구 주입=프롬프트 분포 변화라 **오프라인 재생으로는 원리적으로 결론 불가**
   (기록 궤적은 무도구 프롬프트에 조건화·§2m cue overload가 이 설계 자신의 부작용 채널 — 도구 설명 2개=cue 추가).
   오프라인=참고만·**결론은 §4-3의 T2_SG_EXCLUDE 유/무 유료 대조런**. 기존 PASS 계열(018~029) 대표 1-2태스크를
   대조런 scope에 포함.

## 4b. P0 실측 결과 (2026-07-19 심야·엔진+census 완료·[S])
- **엔진 `group_reduce` 구현·단위 5/5 PASS** (`test_group_reduce.py`): max1/sum·unknown-flag·argmax 합성·
  interest_delta 합성(eff 6.85·delta 1.225·92일). 합성 확장(sum/diff nested-op·argmax per-record key)·
  argmax missing-key 제외(잠재버그 수정·활성 소비자 0=retail은 argmax_where). 커밋됨.
- **§4-0 gold census (formula 반) [S]**: 8 interest-correction 중 **6개가 `balance×Δapy×1yr` 정확 재현**
  (095/094=$8000·093=$12000·097diamond=$10000·096bronze=$2500·096goldplus=$5000=라운드) →
  **gold formula 자기일관·026식 벤치버그 아님**. 097-silver($8333)/platinum($5833)만 non-round=부분기간 추정
  (engine period_start/end가 처리·census 미종결분 2개=DB balance+기간 필요). ⇒ `interest_delta` 접근 검증됨.
- **미완(다음)**: §4-0 stack_rules↔KB 반(expected_apy 6.85 등을 KB boost 조항서 재유도 — 계좌별 KB doc 정독 필요·
  A2 stack_rules 작성과 동시) · §4-1 formalize 프로브(GPU 필요·T3런 후).

## 5. 구현 단계
- P0: `t2_compute`에 **`group_reduce` 프리미티브 1개**(도메인-일반·unknown-flag 포함) + 단위테스트(§4-2).
  2a/2b는 기존 op(argmax·sum·diff·multiply·days_between)과의 표현트리 합성 — 신규 op 추가분은 이 1개뿐(리뷰①).
- P1: A2 2a/2b 선언 + isolate 스펙(카드 격리 선례 복제) + §4-0/4-1 프로브.
- P2: §4-3 스모크 → D+G 확대. (L1은 T3/batch_a2 결과에 따라 별도 트랙.)

## 6. 논문 연결 (Track A)
- ratefix(카드 rate)→account-APY(계좌)로 **같은 분담 패턴의 두 번째 인스턴스** = "lever 배분 방법론"의 도메인-내
  일반화 증거. £L1(발견=controller) vs L2(계산=formalize→결정론)의 이층 구조 자체가 축2(조립 vs 계산) 분리의 실측 사례.
