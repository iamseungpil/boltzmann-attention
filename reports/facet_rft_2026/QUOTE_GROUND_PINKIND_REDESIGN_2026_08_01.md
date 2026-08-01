# QUOTE-GROUND 재설계: pin_kind 선언 + 종류별 닫힌-검사 라우팅 (2026-08-01)

> **상태 = 설계서·리뷰 대기(구현 금지)**. 근거 원장 = **C275**(quote-ground false-abstain이 022 실패의 1차 원인·"call again" 지시가 이행 불가능한 모순) · **C276**(gold-free 전수 계수: 현행 구조적 false-abstain 9종·별칭 채널 불요·범주어 false-apply 표면·⑤ 대문자 게이트 철회).
> 원칙 권위 = [[22]] 근거-우선 formalize 계약(2026-08-01 따름정리+보강): **의미 판단·근거·근거의 종류까지 LLM이 형식 산출, 엔진은 종류별 닫힌 필요조건 라우팅만.**
> 관련 코드(현행): `t2_scaffold_get.py:613-632`(C197 가드) · `t2_scaffold_get.py:1400-1434`(C195 coverage 문구) · `t2_compute.py select_discrepant`(P4 `_missing` 계상) · A2 `banking_knowledge.gate.json` ratefix `isolate`(quote_must_contain_field=merchant_name).

## 0. 요약 (5줄)

C197 가드는 "제외문 quote가 행 merchant_name을 **축자 포함**해야 rate 인정"인데, KB는 상인을 짧은 브랜드명("Target")으로 부르고 레코드는 긴 이름("Target - Eco Collection")이라 **옳은 강등 9종이 구조적으로 abstain**된다(C276①·022 ba8b가 그중 하나=4런 전패 기여). 수정 = 가드의 열린 술어(정책 이름↔상인 지시 동일성)를 문자열로 닫으려는 시도를 중단하고, **LLM이 핀(`exclusion_policy_merchant`)과 핀의 종류(`exclusion_pin_kind`)를 formalize로 선언**, 엔진은 **종류별 닫힌 필요조건만 라우팅**한다(named=포함-사슬 검사·category=등가 미적용+표면화). 부수 수정 = C195 coverage 문구의 **모순 지시 분화**(record-유래 결핍만 "call again"·sub-유래는 "unverified" 정직 표기 — C275 ⑤정정의 직접 원인 제거). 전부 플래그 게이트(`T2_QUOTE_PIN`·기본 OFF)·A2 선언 구동·엔진 도메인 리터럴 0.

## 1. 문제 (실측 요약 — 재유도 금지·원장 인용)

1. **C275 인과 사슬**: isolate 서브가 ba8b에 Target-제외를 옳게 적용(rate=1+축자 quote) → C197 가드가 quote 안 "Target - Eco Collection" 축자를 요구 → KB엔 "Target"뿐 → rate 드롭(abstain) → select_discrepant 행 skip → coverage 문구가 "call again with the completed input" 지시 → **그 지시가 이행 불가능**(결핍 3필드는 sub-산출값·레코드에 없고·params가 rate 제공 금지) → agent는 KB_search로 이행 시도까지 했으나 닫을 수 없음 → 9/10 dispute → db 실패. 4런(Y1t0/t1·P1/P2) 동일.
2. **C276 전수 계수**(merchant 138종 × KB 11,719라인·결정론·gold 0):
   - 현행 술어 구조적 false-abstain = **9종**(Target - Eco Collection·Microsoft 365[둘 다 라이브 드롭 관측]·Apple Store·Dell Technologies·Home Depot·LinkedIn Ads·Slack Technologies·Zoom Video·Electronics Express Miami).
   - 선행 n-gram 다리로 **9/9 회수 가능**.
   - **비-부분문자열 브랜드 별칭 사례 0** ⇒ 열린 별칭 채널 불요.
   - **새 위험**: 비선행 매칭 35종이 전부 일반 범주어("market"·"airlines"·"restaurant") = 단순 핀+등가/포함이면 범주어가 false-apply 다리("grocery markets"↔Thrive Market)로 019형 부활.
   - 대문자-시작 게이트는 **철회**(C276⑤: Title-Case 범주어 `Electronics Merchants`·`General Retailers`가 코퍼스 내 반례·소문자 브랜드 방향도 실패).

## 2. 설계

### 2a. A2 변경 (ratefix variant `isolate` — 데이터만)

`operand_schema`에 2필드 추가(문구는 도메인-일반·값은 sub 산출):

```json
"exclusion_policy_merchant": "<if you downgraded this item because the policy EXCLUDES this specific merchant: the merchant name EXACTLY as the policy text names it, copied verbatim from your exclusion_quote (policies often use a short brand name, e.g. 'Target'). Empty otherwise>",
"exclusion_pin_kind": "<'named_merchant' if the exclusion sentence names this specific merchant; 'category' if it excludes a TYPE of merchant or purchase (e.g. 'general retailers', 'gaming subscriptions') and you judged this item to belong to that type; empty if no exclusion applies>"
```

`instructions`에 표기-지식 1문장 추가(엔진 강제 아님·LLM 가이드): *"In policy prose, category words are usually lowercase common nouns; a specific merchant is usually a proper name — quote the sentence and copy the exact name it uses."*

가드 바인딩 선언 신설(엔진이 읽는 필드명 바인딩·구조는 도메인-일반):

```json
"quote_pin": {"policy_field": "exclusion_policy_merchant", "kind_field": "exclusion_pin_kind", "row_field": "merchant_name"}
```

기존 `quote_must_contain_field`는 플래그 승격 후 제거 예정(과도기 공존·OFF 경로가 사용).

### 2b. 엔진 변경 (`t2_scaffold_get.py` 가드 블록 — 도메인-일반·리터럴 0)

플래그 `T2_QUOTE_PIN=1`(기본 OFF). ON이고 A2에 `quote_pin` 선언이 있으면 기존 검사 **대체**:

| 단계 | 검사 (전부 닫힘) | 불성립 처리 |
|---|---|---|
| 공통 | `len(quote) ≥ quote_min` ∧ `norm(quote) ∈ docnorm` | rate 드롭 + 사유 표면화 (현행 유지 — 날조 차단) |
| kind=`named_merchant` | `pin ≠ ∅` ∧ `norm(pin) ∈ norm(quote)` ∧ `norm(pin) ∈ norm(row[row_field])` | rate 드롭 + **사유 표면화**: `policy names '<pin>' but this row's merchant is '<merchant>' — mapping rejected` + **재질의 1회**(이행 가능: 올바른 핀/종류로 재산출 요구) |
| kind=`category` | (포함/등가 검사 **미적용**) | — 항상 통과하되 `_sg_details`와 반환에 **category-마크** 병기: `category-based ('<pin or quote 요지>') — membership unverified` |
| kind 결측 (quote는 있고 rate 강등형) | — | 드롭 아닌 **재질의 1회**("declare exclusion_pin_kind") → 실패 시 abstain(안전측·사유 명시) |

- "rate 강등형" 판정: 엔진은 강등 여부를 의미로 판단하지 않는다 — `exclusion_quote ≠ ∅`인 행을 대상으로만 라우팅(닫힘: 필드 존재 검사).
- 재질의 = 기존 isolate 재호출 경로 재사용(`max_rounds` 소진 내·해당 행만·guard 피드백 문구 포함). 신규 판단 0.

### 2c. C195 coverage 문구 분화 (`t2_scaffold_get.py:1413-1421` — 모순 지시 제거)

`_missing` 필드를 두 그룹으로 분할해 문구를 가른다(멤버십 = A2 선언 대조·판단 0):

- **record-유래**(`iso.row_fields` ∋ 필드): 현행 문구 유지 — *"Read the missing value(s) from the records… call again with the completed input."*
- **sub-유래**(`iso.operand_schema` ∋ 필드): *"…could not be verified from the policy documents; these rows remain UNVERIFIED. Do NOT supply rates or promo values yourself — tell the customer which transactions could not be checked."*

이건 계측 문구의 **사실-모순 수정(버그픽스)**이므로 `T2_QUOTE_PIN`과 독립으로 즉시 적용을 제안한다(§8 리뷰 포인트 R3).

### 2d. 명시적 비채택 (재론 방지)

- ~~토큰-교집합/head-토큰 완화~~ — 열린 술어의 문자열 동결(이 대화 1차 회귀·기각).
- ~~대문자-시작·토큰-최대 게이트~~ — C276⑤ 철회(2차 회귀·코퍼스 내 반례).
- ~~열린 별칭 채널~~ — C276③ 사례 0.
- ~~row-측 `merchant_brand` 핀(양측-핀 등가)~~ — **P2 예비로 강등**: 같은 sub·같은 문맥에서 채우면 오류가 상관돼 P1(policy-핀 ⊆ merchant 포함) 대비 실이득 0이고, 탈상관하려면 별도 호출(+비용). Δspurious 실측이 §6 임계를 넘으면 별도-문맥 채움으로 승격 검토.
- ~~미검증-행 엔진 자동 재질의 루프(무조건)~~ — 술어 불충족이 원인일 땐 동일 실패 반복(C275 실증). 재질의는 **guard 피드백을 실은 1회**만(2b).

## 3. [[05]] 결정-시점 3질문 (의무 섹션)

1. **scaffold 또는 A2의 도메인-특화 순증?** — scaffold: 순증 0(신설 로직은 `quote_pin` 선언을 읽는 라우팅·필드명 리터럴 0·`grep merchant` 0 유지). A2: +2 스키마 필드·+1 바인딩 선언 = **도메인-특화 데이터 순증 있음(정직)** — 단 출처는 전부 정책/환경 구조(KB 표기 관행·레코드 필드명)이지 gold가 아니며([[23]] 준수·§6 검증도 gold-diff는 계측만), 차등 본체=A2라는 [[05]] §3 관례 내.
2. **유동적 판단을 결정론에 동결?** — **아니오가 이 설계의 목적**: 현행 가드가 동결해 둔 열린 판단(정책 이름↔상인 지시 동일성)을 **LLM으로 반환**하고, 엔진엔 닫힌 필요조건(축자 포함·필드 존재)만 남긴다. 동결 방향이 아니라 **해동 방향**.
3. **scaffold가 모델 대신 도메인 행동 수행?** — 아니오. 엔진은 검사·라우팅·표면화만. 강등 판단·핀 선택·범주 소속 판단 전부 LLM.

## 4. [[22]] 술어·처방 분류 (§3 2문 체크·의무 섹션)

| 검사/처방 | ①술어 닫힘? | ②처방 닫힘? | 판정 |
|---|---|---|---|
| quote ∈ 주입문서 (축자) | 닫힘(문자열 실재·변이 무관 — 표기 그대로 대조) | 드롭=형식-층(날조 차단·시점 무관) | 강제 정당 |
| pin ∈ quote / pin ∈ merchant_name | 닫힘(포함) | named 불성립→드롭+재질의 1회 | 강제 정당(필요조건임을 자인·§7) |
| pin_kind 라우팅 | **판단은 열림 → LLM 선언**·엔진은 선언값 멤버십(닫힘) | category→표면화만(강제 0) | [[22]] 3호 준수(표면화 강등) |
| kind 결측 | 필드 존재(닫힘) | 재질의 1회→abstain(안전측) | 강제 아님(이행 가능 지시) |
| coverage 문구 분화 | 필드 출처 멤버십(닫힘) | 문구만(강제 0) | 표면화 |

열린 잔여(명시): (a) named 선언의 진실성(핀이 정말 그 상인을 지시하는가) (b) category 선언의 소속 판단 — 둘 다 LLM 몫·표면화·§6 Δ계측·(장기) learn([[13]]).

## 5. 케이스 매트릭스 (구현 시 픽스처로 전사)

| # | 케이스 | 입력 요지 | 기대 |
|---|---|---|---|
| 1 | ba8b (C275) | quote=Target 제외문·pin="Target"·kind=named·merchant="Target - Eco Collection" | **통과** → rate=1 생존 → 022 discrepancy 10/10 |
| 2 | Microsoft 365 | pin="Microsoft"·kind=named | 통과 |
| 3 | 019형 | Thrive Market 행·quote=ThredUp 문장·pin="ThredUp"·kind=named | **드롭**(pin ∉ merchant)+사유+재질의 |
| 4 | Delta | Delta Sky Club 행·pin="Delta Airlines"·kind=named | 드롭(포함 불성립) |
| 5 | 범주어 게이밍 | pin="market"·kind=named·quote="…grocery markets…" | *통과함*(포함 성립) — **잔여**: kind 오선언 1중 오류·category-마크 없이 강등. §6 Δspurious의 계측 표적(닫힌 검사로 못 막음을 정직 기재) |
| 6 | 정직한 범주 주장 | kind=category·quote="gaming subscriptions are excluded" | 통과+**category-마크 표면화** |
| 7 | kind 결측 | quote 있음·kind="" | 재질의 1회→abstain |
| 8 | quote 날조 | quote ∉ docs | 드롭(현행 동일) |
| 9 | 회귀 | 기존 test_c197 계열(019 차단·0행 무판정 문구) | OFF 경로 불변·ON 경로 동등 이상 |

## 6. 검증 계획 (순서 고정·[[09]])

1. **오프라인 단위**(무료): `test_c277_quotepin.py` — §5 매트릭스 9종 + 기존 회귀. 리모트 배터리 PASS 후 배포.
2. **x28 정적 재검**(무료): 계수기에 새 술어 시뮬 추가 — C276① 9종이 named-핀 경로로 전부 회수되는지(핀=각 다리 n-gram 가정) + 범주어 35종이 category 경로로 빠지는지.
3. **022 replay**(무료·GPU만): P2 궤적의 77행 입력 재실행 → ba8b rate 생존 → `select_discrepant` 10건 → coverage 라인 "77 of 77" 확인.
4. **라이브**(유료·승인 후): 다음 Y2 계열 런에 `T2_QUOTE_PIN=1` 편입([[19]] 합성 스택). **판정은 pass 비교가 아니라 기전 지표**(C272 시간-표류 교훈): 022 discrepancy 집합 정확도(replay-대조)·**Δspurious 양방향** — false-abstain(드롭 사유 로그 중 semantic 매핑 실재 건수) / false-apply(category-마크·named-강등의 gold-diff — *계측만*, A2 수정 근거로 사용 금지 [[23]]). 스모크로 신규 마크 라이브 발화 확인 후 full([[30]]).
5. **승격/기각 임계(사전등록)**: 회귀(§5-9) 0 ∧ 022 replay 10/10 → go_stack ON 승격. 라이브 false-apply 신규 발생이 false-abstain 회수분을 초과하면 OFF 원복(플래그 롤백=1줄).

## 7. 상쇄·위험 (§1.3 — 이 레버가 파는 것)

1. **category 오선언(케이스 5)**: 등가류 검사를 우회하는 유일한 문 — 닫힌 검사로 못 막는다(브랜드-됨=열림). 대가 지불 방식 = 표면화(마크)+Δ계측+장기 learn. **숨은 실패가 아니라 감사 가능한 주장**이라는 것이 현행 대비 순개선.
2. **스키마 부하 +2필드**: formalize 품질 저하 가능(스키마 비대·§2h max_batch 상호작용) — 계측: operand 채움률·range_retry 빈도(기존 로그 라인 재사용).
3. **재질의 1회**: sub 호출 증가(행-단위·bounded) — 계측: inject 라인 수 diff.
4. **표면화 자체의 역효과**(C270⚠ 동형): category-마크·unverified 문구가 agent의 over-action(무근거 재시도 남발)을 유발하는지 — Δspurious에 "동일-인자 재호출 3회+" 지문 포함.

## 8. 리뷰 포인트 (사용자 결정 요청)

- **R1**: P1(policy-핀 포함)로 시작하고 P2(양측-핀·별도-문맥)는 Δ실측 후 승격 — 동의하는가? (§2d 근거: 같은-문맥 채움은 상관 오류라 이득 0)
- **R2**: category 경로를 "통과+마크"로 두는 것(강등 적용됨·미검증 표시) vs "통과하되 rate를 abstain"(보수) — 전자 제안. 근거: 후자는 정직한 범주 제외(General Retailers=Target 공동 제외가 정답인 사례)까지 죽여 false-abstain을 재생산.
- **R3**: coverage 문구 분화(§2c)를 플래그와 독립인 즉시 버그픽스로 — 동의하는가?
- **R4**: 재질의 1회의 위치 — guard 불성립 행만 대상·max_rounds 내 재사용. 추가 라운드 예산 필요 없다고 판단.

## 9. 구현 순서 (리뷰 승인 후)

① `t2_scaffold_get.py` 가드 교체(플래그)+문구 분화 → ② A2 스키마/바인딩 추가 → ③ `test_c277_quotepin.py` + 회귀 → ④ x28 정적 재검 → ⑤ 022 replay → ⑥ 커밋/push/리모트 배터리 → ⑦ 스모크→라이브 편입(승인). 롤백 = `T2_QUOTE_PIN` OFF(1줄).
