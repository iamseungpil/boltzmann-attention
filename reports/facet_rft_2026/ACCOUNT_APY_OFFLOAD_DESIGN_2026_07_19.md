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

## 1. 분담 설계 (ratefix keystone 완전 동형 · [[05]]/[[10]]/[[03b]])
```
LLM(격리 서브·INFER):  후보별 "APY 구성요소" formalize — base_apy · boost 후보(종류·값·출처 조항)
                        ※ 눈대중 합성 금지 — 구성요소만 낸다 (ratefix의 base_rate/promo 동형)
엔진(결정론·GET 반환):  스택 규칙 적용(max-per-종류 + Σ스택형) · 후보 비교(argmax) · 보정액 산술
                        ※ 도메인 상수 0 — 스택 규칙의 *구조*(어느 종류가 max-1인지)도 A2 선언
게이트(기존 재사용):    write(open/close/apply_credit/submit_report)의 인자 검증은 기존 스택
                        (provenance·reference_filter·WEV-류) — 본 설계는 값 *생산* 측만
```

## 2. A2 스키마 (scaffold_get_tools 확장 — 엔진 신규 op 2개 외 전부 데이터)

### 2a. `get_best_account_option` (계좌-선택·055/066/071 + 최적화군 057/064~069 표적)
```json
{"name": "get_best_account_option",
 "description": "USE THIS when the customer wants to choose/open the account (or account+card combo) that maximizes APY (or meets stated constraints). Pass the candidate options you found in the KB with their APY components; it applies Rho-Bank stacking rules deterministically and returns the best option with the effective APY.",
 "params": {"candidates": "JSON array: [{option: <account/card name>, base_apy: <number>, boosts: [{kind: 'checking'|'card'|'relationship'|'tier'|..., value: <number>, source: <doc quote>}...]}]",
            "constraints": "JSON object of the user's hard constraints (deposit amount, disallowed types, required perks) — copy from the conversation, do not invent"},
 "return_template": "Best option by effective APY under stacking rules: {result}. Per-candidate effective APY: {details}",
 "op": {"op": "apy_argmax", "over": "candidates",
        "stack_rules": {"checking": "max1", "card": "max1", "relationship": "sum", "tier": "sum"}},
 "isolate": { "…rate-formalize isolate 동형…": "후보별 격리 formalize — inject_docs=계좌/카드 문서군·row_fields
   화이트리스트·max_batch(§2d cap)·quote-grounding은 boosts[].source에 원문 요구" }}
```
- **엔진 신규 op `apy_argmax`** (t2_compute): 종류별 max1/sum 합성 → effective APY → argmax + per-후보 상세.
  스택 *규칙표는 A2 데이터*(`stack_rules`) — 엔진은 "max1"/"sum" 두 combinator만 안다(도메인 리터럴 0).

### 2b. `get_interest_correction` (보정액·093~097 표적)
```json
{"name": "get_interest_correction",
 "params": {"principal": "...", "days": "...", "correct_components": "위와 같은 components 스키마", "applied_apy": "실제 적용된 APY(조회값 복사)"},
 "op": {"op": "interest_delta", "uses_stack_rules": true},
 "return_template": "correct APY {apy_correct}% vs applied {apy_applied}% → interest difference over {days} days: {amount}"}
```
- **엔진 신규 op `interest_delta`**: effective-APY(2a 재사용) − applied → 원금×Δ×days/365 (반올림 규칙 A2).
  095의 3필드(expected_apy·actual_apy·amount_difference)를 직접 생산.

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
0. **gold 자기일관 census**: D+G gold 인자(APY·account_class)를 KB 정책으로 재유도 — 벤치버그 분리.
1. **formalize 프로브**(GPU0·rate-프로브 동형·`bank_apy_formalize_probe.py`): 055/066/071/095의 실제 후보·계좌로
   서브가 components를 정확히 내는지 — 셀별 정답률·grounding 탈락률. 기준: ratefix 선례(카테고리 격리서 오독 0)와 동급.
2. **엔진 op 단위테스트**: stack_rules 조합·argmax·interest_delta (gold 값 재현 — 095의 98.0 등).
3. **유료 스모크(승인 후)**: L2-실패 대표 2태스크(095·066) 단일변수(`T2_SG_EXCLUDE`로 도구 유/무 대조).
4. Δspurious: 새 도구 주입 arm에서 기존 PASS 계열(018~029) 무회귀 확인은 프로브·오프라인 재생으로.

## 5. 구현 단계
- P0: `t2_compute`에 `apy_argmax`·`interest_delta` op + 단위테스트(§4-2) — 엔진 일반 combinator.
- P1: A2 2a/2b 선언 + isolate 스펙(카드 격리 선례 복제) + §4-0/4-1 프로브.
- P2: §4-3 스모크 → D+G 확대. (L1은 T3/batch_a2 결과에 따라 별도 트랙.)

## 6. 논문 연결 (Track A)
- ratefix(카드 rate)→account-APY(계좌)로 **같은 분담 패턴의 두 번째 인스턴스** = "lever 배분 방법론"의 도메인-내
  일반화 증거. £L1(발견=controller) vs L2(계산=formalize→결정론)의 이층 구조 자체가 축2(조립 vs 계산) 분리의 실측 사례.
