# QUOTE-GROUND 재설계: pin_kind 선언 + 종류별 닫힌-검사 라우팅 (2026-08-01)

> **상태 = rev2(2026-08-01 2차 리뷰 반영) — §8b가 확정 스펙, §2·§5·§7은 rev1 구현 상태의 기록이며 R5 관련 항목은 *철회 표시*로 남긴다(삭제하지 않음 = 회귀 감사 추적).**
> **⚠남은 사용자 결정 1건 = [[23]] 신범주("운영자 지식") 승인**(§8b ⒡). 승인 전 표 저작·구현 진입 금지.
> rev1 구현분(`T2_QUOTE_PIN`)은 **기본 OFF·라이브 배포 0** — rev2가 그 일부를 대체한다(원장 C278→C279). 근거 원장 = **C275**(quote-ground false-abstain이 022 실패의 1차 원인·"call again" 지시가 이행 불가능한 모순) · **C276**(gold-free 전수 계수: 현행 구조적 false-abstain 9종·별칭 채널 불요·범주어 false-apply 표면·⑤ 대문자 게이트 철회).
> 원칙 권위 = [[22]] 근거-우선 formalize 계약(2026-08-01 따름정리+보강): **의미 판단·근거·근거의 종류까지 LLM이 형식 산출, 엔진은 종류별 닫힌 필요조건 라우팅만.**
> 관련 코드(현행): `t2_scaffold_get.py:613-632`(C197 가드) · `t2_scaffold_get.py:1400-1434`(C195 coverage 문구) · `t2_compute.py select_discrepant`(P4 `_missing` 계상) · A2 `banking_knowledge.gate.json` ratefix `isolate`(quote_must_contain_field=merchant_name).

## 0. 요약 (5줄)

C197 가드는 "제외문 quote가 행 merchant_name을 **축자 포함**해야 rate 인정"인데, KB는 상인을 짧은 브랜드명("Target")으로 부르고 레코드는 긴 이름("Target - Eco Collection")이라 **옳은 강등 9종이 구조적으로 abstain**된다(C276①·022 ba8b가 그중 하나=4런 전패 기여). 수정 = 가드의 열린 술어(정책 이름↔상인 지시 동일성)를 문자열로 닫으려는 시도를 중단하고, **LLM이 핀(`exclusion_policy_merchant`)과 핀의 종류(`exclusion_pin_kind`)를 formalize로 선언**, 엔진은 **종류별 닫힌 필요조건만 라우팅**한다(named=포함-사슬 검사·category=등가 미적용+표면화). 부수 수정 = C195 coverage 문구의 **모순 지시 분화**(record-유래 결핍만 "call again"·sub-유래는 "unverified" 정직 표기 — C275 ⑤정정의 직접 원인 제거). 전부 플래그 게이트(`T2_QUOTE_PIN`·기본 OFF)·A2 선언 구동·엔진 도메인 리터럴 0.

## 1. 문제 (실측 요약 — 재유도 금지·원장 인용)

1. **C275 인과 사슬**: isolate 서브가 ba8b에 Target-제외를 옳게 적용(rate=1+축자 quote) → C197 가드가 quote 안 "Target - Eco Collection" 축자를 요구 → KB엔 "Target"뿐 → rate 드롭(abstain) → select_discrepant 행 skip → coverage 문구가 "call again with the completed input" 지시 → **그 지시가 이행 불가능**(결핍 3필드는 sub-산출값·레코드에 없고·params가 rate 제공 금지) → agent는 KB_search로 이행 시도까지 했으나 닫을 수 없음 → 9/10 dispute → db 실패. 4런(Y1t0/t1·P1/P2) 동일.
2. **C276 전수 계수**(merchant 138종 × KB 11,719라인·결정론·gold 0):
   - 현행 술어 구조적 false-abstain = ~~9종~~ → **6종 정정(C276★① 2026-08-01·매칭 원문 전수 대조)**: Target - Eco Collection·Microsoft 365[둘 다 라이브 드롭 관측]·Apple Store·Dell Technologies·Zoom Video·Slack Technologies(전부 정책 불릿 축자 명명 확인). **허위 3종 제외**: Home Depot("home-**sharing** platforms" 파생 합성어)·Electronics Express Miami("Hardware/Electronics Merchants" 범주 제목)·LinkedIn Ads("LinkedIn **Learning**"=타-상인) — 허위 유형이 정확히 케이스 11·category 경로의 대상이라 설계 방향 불변.
   - 선행 n-gram 다리로 **6/6 회수 가능**.
   - **비-부분문자열 브랜드 별칭 사례 0** ⇒ 열린 별칭 채널 불요.
   - **새 위험**: 비선행 매칭 35종이 전부 일반 범주어("market"·"airlines"·"restaurant") = 단순 핀+등가/포함이면 범주어가 false-apply 다리("grocery markets"↔Thrive Market)로 019형 부활.
   - 대문자-시작 게이트는 **철회**(C276⑤: Title-Case 범주어 `Electronics Merchants`·`General Retailers`가 코퍼스 내 반례·소문자 브랜드 방향도 실패).

## 2. 설계

### 2a. A2 변경 (ratefix variant `isolate` — 데이터만)

`operand_schema`에 2필드 추가(문구는 도메인-일반·값은 sub 산출):

```json
"exclusion_policy_merchant": "<if you downgraded this item because the policy EXCLUDES this specific merchant: the merchant name EXACTLY and COMPLETELY as the policy text names it, copied verbatim from your exclusion_quote — copy the WHOLE name the policy uses (e.g. 'Target'), never a fragment of it. Empty otherwise>",
"exclusion_pin_kind": "<'named_merchant' if the exclusion sentence names this specific merchant; 'category' if it excludes a TYPE of merchant or purchase (e.g. 'General Retailers', 'Gaming Subscription Merchants') and you judged this item to belong to that type; empty if no exclusion applies>"
```

**[[23]] 출처 의무(발견 2 반영·배포 전 확인 완료 2026-08-01)**: 예시 문자열 전부 env-축자 — `'Target'`·`'General Retailers'` = `doc_credit_cards_ecocard_004`(*"### General Retailers (even for eco-friendly product purchases)"* / *"- Target"*) · `'Gaming Subscription Merchants'` = `doc_business_credit_cards_business_silver_rewards_card_005`(*"**Gaming Subscription Merchants (Software Exclusion):**"*·리모트 grep 확인). 각 신설 A2 항목에 `_note_` 필수: 위 축자 출처 + "gold 경유 0" 명기.

`instructions`에 지시-대상 기준 1문장 추가(엔진 강제 아님·LLM 가이드·**발견 3 반영** — 표기 기준 문장은 C276⑤ 철회 상관물의 재도입이라 기각): *"A merchant pin names ONE business; words describing a TYPE of merchant or purchase ('retailers', 'markets', 'airlines') are a category even when capitalized as a heading."*

가드 바인딩 선언 신설(엔진이 읽는 필드명 바인딩·구조는 도메인-일반·**R5 앵커 + 발견 4 문구 템플릿 포함**):

```json
"quote_pin": {
  "policy_field": "exclusion_policy_merchant",
  "kind_field": "exclusion_pin_kind",
  "row_field": "merchant_name",
  "pin_anchor": "leading",
  "reject_note": "the policy text you quoted names '{pin}', but this row's merchant is '{merchant}' — the mapping was rejected and this row's rate was NOT applied.",
  "retry_prompt": "Re-check this item: copy the merchant name exactly and completely as the policy sentence names it into exclusion_policy_merchant, or if the exclusion is by TYPE of merchant/purchase, set exclusion_pin_kind='category'.",
  "category_note": "category-based exclusion ('{pin}') — merchant membership unverified.",
  "unverified_note": "could not be verified from the reward-rate policy for this card; the row(s) remain UNVERIFIED. Do NOT supply rates or promo values yourself — tell the customer which transactions could not be checked."
}
```

- ~~**R5 `pin_anchor: "leading"`**~~ → **☠철회(rev2·§8b ⒜)**. 사유: 이 술어가 검사하는 것은 연산의 모양(위치)이 아니라 **지시 동일성(열림)**이며, "A2 선언이라 도메인-일반"이라는 완화도 구제가 못 된다(선언값 자체가 동결된 휴리스틱 판단). 실파손 = 문자-수준 `startswith`가 `'Target'`→`'Targeting Solutions'`에 성립(같은 절 토큰-연속 규칙과 자기모순·단위테스트가 적발)·토큰-수준으로 고쳐도 케이스 11 통과. **2차 리뷰 추가 판정: 같은 비판이 평범 포함(`pin ∈ merchant`)에도 그대로 적용된다** — 비-부분문자열 별칭이 실재하는 ABox에서 깨지는 코퍼스-우연 필요조건이므로, **유사도·접두·포함을 전부 제거**하는 것이 일관된 유일 귀결(§8b ⒝ 식별표).
- **발견 4**: `reject_note`/`retry_prompt`/`category_note`/`unverified_note` 문구는 전부 A2 선언(placeholder `{pin}`/`{merchant}` 치환·`range_retry_prompt` 선례). 엔진 기본값은 도메인 어휘 0의 중립 문구("the source documents") — `grep merchant|policy` 0 유지.

기존 `quote_must_contain_field`는 플래그 승격 후 제거 예정(과도기 공존·OFF 경로가 사용).

### 2b. 엔진 변경 (`t2_scaffold_get.py` 가드 블록 — 도메인-일반·리터럴 0)

플래그 `T2_QUOTE_PIN=1`(기본 OFF). ON이고 A2에 `quote_pin` 선언이 있으면 기존 검사 **대체**:

| 단계 | 검사 (전부 닫힘) | 불성립 처리 |
|---|---|---|
| 공통 | `len(quote) ≥ quote_min` ∧ `norm(quote) ∈ docnorm` | rate 드롭 + 사유 표면화 (현행 유지 — 날조 차단) |
| kind=`named_merchant` | `pin ≠ ∅` ∧ `norm(pin) ∈ norm(quote)` ∧ ~~`row.startswith(pin)`~~ **(R5 철회 → rev2 §8b: 표 멤버십으로 대체)** | rate 드롭 + **사유 표면화**(A2 `reject_note` 치환) + **재질의 1회**(A2 `retry_prompt`) |
| kind=`category` | (포함/앵커 검사 **미적용**) | — 항상 통과하되 `_sg_details`와 반환에 **category-마크** 병기(A2 `category_note` 치환) |
| kind 결측 **또는 열거 밖 값**(발견 6: 오타 "named" 등 = 결측 동일 취급) (quote는 있고 강등형) | 필드 존재·열거 멤버십(닫힘) | 드롭 아닌 **재질의 1회**("declare exclusion_pin_kind") → 실패 시 abstain(안전측·사유 명시) |

**구현 규칙(C276★① 교훈·rev2에서도 유지)**: 남기는 모든 문자열 검사(quote∈docs·pin∈quote)는 raw substring이 아니라 **정규화 후 토큰-연속-부분열**로 구현한다("target"이 "targeting"에 매칭되는 부류 차단). 단 이것이 막는 것은 **부분-단어 매칭뿐**이고, 토큰이 실재해도 지시가 다른 경우(파생 합성어·범주 제목·타-상인)는 문자열로 못 막는다 — rev2는 그 축을 **식별표 멤버십**으로 옮긴다(§8b).

- "rate 강등형" 판정: 엔진은 강등 여부를 의미로 판단하지 않는다 — `exclusion_quote ≠ ∅`인 행을 대상으로만 라우팅(닫힘: 필드 존재 검사).
- 재질의 = 기존 isolate 재호출 경로 재사용(`max_rounds` 소진 내·해당 행만·guard 피드백 문구 포함). 신규 판단 0.

### 2c. C195 coverage 문구 분화 (`t2_scaffold_get.py:1413-1421` — 모순 지시 제거)

`_missing` 필드를 두 그룹으로 분할해 문구를 가른다(멤버십 = A2 선언 대조·판단 0):

- **record-유래**(`iso.row_fields` ∋ 필드): 현행 문구 유지 — *"Read the missing value(s) from the records… call again with the completed input."*
- **sub-유래**(`iso.operand_schema` ∋ 필드): A2 `unverified_note` 문구(§2a) — "UNVERIFIED·직접 값 공급 금지·고객에 고지".
- **엣지 규칙(발견 5·결정론)**: 필드가 양쪽에 다 있으면 **row_fields 우선**(="call again"·레코드에서 완성 가능하므로 이행 가능 지시가 정당) / 어느 쪽에도 없으면 **sub-유래 취급**(안전측 — 이행 불가능 지시를 내는 것보다 unverified 정직 표기).

이건 계측 문구의 **사실-모순 수정(버그픽스)**이므로 `T2_QUOTE_PIN`과 독립 적용 — 단 **R3 조건(리뷰 확정): "즉시" = repo 반영·다음 런부터**. 진행 중 라이브 런(pass3 등)에는 배포하지 않는다(런 내 비교가능성 오염 — C272 시간-표류와 같은 축).

### 2d. 명시적 비채택 (재론 방지)

- ~~토큰-교집합/head-토큰 완화~~ — 열린 술어의 문자열 동결(이 대화 1차 회귀·기각).
- ~~대문자-시작·토큰-최대 게이트~~ — C276⑤ 철회(2차 회귀·코퍼스 내 반례).
- ~~열린 별칭 채널~~ — C276③ 사례 0.
- ~~row-측 `merchant_brand` 핀(양측-핀 등가)~~ — **P2 예비로 강등**: 같은 sub·같은 문맥에서 채우면 오류가 상관돼 P1(policy-핀 ⊆ merchant 포함) 대비 실이득 0이고, 탈상관하려면 별도 호출(+비용). Δspurious 실측이 §6 임계를 넘으면 별도-문맥 채움으로 승격 검토.
- ~~미검증-행 엔진 자동 재질의 루프(무조건)~~ — 술어 불충족이 원인일 땐 동일 실패 반복(C275 실증). 재질의는 **guard 피드백을 실은 1회**만(2b).

**선택-레버(재론 금지 목록 아님·2차 리뷰 선택 제안·미채택)**: category 주장의 그룹명이 표 키와 norm-불일치
(의역)하면 산문-경로로 새는 우회 채널이 생긴다. quote 축자 강제가 대부분 막지만, **축자 quote 안에 표-키가
실재하면 그 키로 강제 라우팅**하는 것은 열거-집합 대조라 닫혀 있고 채널을 좁힌다. 단 **라우팅 규칙 순증**
이므로 Δ계측과 함께 별도 판단(기본 미채택).

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

## 5. 케이스 매트릭스 (rev1 픽스처 = `test_c278_quotepin.py` 현행)

> **rev2 기대 갱신(리뷰 세부 7)**: 5=표 조회 실패(공백)→재질의→abstain / 10=표 엔트리에 관사-접두 행을
> 포함시켜 저작하면 통과(저작 시점 판단) / **11=표에서 `LinkedIn Learning`→`[]`(판단된 무대응)이라 차단**
> — 즉 rev2에서 케이스 11은 잔여가 아니라 **닫힌다**. 아래 표의 "기대"는 rev1 구현 기준 기록.

| # | 케이스 | 입력 요지 | 기대 |
|---|---|---|---|
| 1 | ba8b (C275) | quote=Target 제외문·pin="Target"·kind=named·merchant="Target - Eco Collection" | **통과** → rate=1 생존 → 022 discrepancy 10/10 |
| 2 | Microsoft 365 | pin="Microsoft"·kind=named | 통과 |
| 3 | 019형 | Thrive Market 행·quote=ThredUp 문장·pin="ThredUp"·kind=named | **드롭**(pin ∉ merchant)+사유+재질의 |
| 4 | Delta | Delta Sky Club 행·pin="Delta Airlines"·kind=named | 드롭(포함 불성립) |
| 5 | 범주어 핀 (R5로 봉쇄) | pin="market"·kind=named·merchant="Thrive Market" | **드롭+재질의**(비선행 — 앵커 불성립) → sub가 category 재선언 시 마크 경로로 정직 회수 |
| 6 | 정직한 범주 주장 | kind=category·quote="**Gaming Subscription Merchants (Software Exclusion):**"(축자) | 통과+**category-마크 표면화** |
| 7 | kind 결측 | quote 있음·kind="" | 재질의 1회→abstain |
| 7b | kind 열거 밖(발견 6) | kind="named"(오타) | 결측과 동일(재질의→abstain) |
| 8 | quote 날조 | quote ∉ docs | 드롭(현행 동일) |
| 9 | 회귀 | 기존 test_c197 계열(019 차단·0행 무판정 문구) | OFF 경로 불변·ON 경로 동등 이상 |
| 10 | **앵커 잔여 A — 관사-접두 이름형** | 가상: 정책이 "Cheesecake Factory"로 명명·행="The Cheesecake Factory" | 드롭+재질의→abstain(**정당 강등의 false-abstain·표면화됨**). 코퍼스 실측 0(x28: The-접두 상인의 제외-문맥 명명 없음)·발생 시 A2 `pin_anchor` 미선언으로 해제 가능 |
| 11 | **앵커 잔여 B — 부분-핀 접두 충돌**(자기감사 신규 발견) | 정책 불릿 "- LinkedIn Learning"에서 핀을 "LinkedIn"으로 **부분 복사**·행="LinkedIn Ads" (동형: "home-sharing platforms"→핀 "home"→"Home Depot" — C276★①이 실증한 파생-합성어 유형) | *통과함*(pin∈quote ∧ 앵커 성립) = **false-apply 잔여**. 완전-복사는 sub 지시(§2a "COMPLETELY")로 유도·토큰-최대 강제는 이름-경계 판정(열림)이라 기각. 표면화·Δ계측 표적 |

## 6. 검증 계획 (순서 고정·[[09]])

1. **오프라인 단위**(무료): `test_c277_quotepin.py` — §5 매트릭스 9종 + 기존 회귀. 리모트 배터리 PASS 후 배포.
2. **정적 재검**(무료·**rev2 기준으로 재작성**·리뷰 세부 7): ~~x28 앵커 시뮬~~ → **표-시뮬**로 교체 —
   ⓐ C276★① 정정 후 **진짜 6종**(Target·Microsoft·Apple·Dell·Zoom·Slack)이 표 멤버십으로 회수되는지
   (**주의: 회수=술어 충족-가능성 회복이지 6종 전부 강등이 정답이라는 뜻 아님**)
   ⓑ 019(ThredUp→Thrive Market) 차단 보존 ⓒ 케이스 11(LinkedIn Learning→`[]`) 차단
   ⓓ 비선행 범주어 35종이 named 경로에서 **전부 조회-실패**로 빠지는지(표 키에 없음).
3. **022 replay**(무료·GPU만·**sub 재호출 필수** — 신규 필드는 궤적에 없음 · **`T2_SG_ISOLATE_TRACE` ON 의무** — ⒠의 "개체 혼동 vs 범주 과적용" 관측 불가의 원인이 정확히 trace 미설정이었다·2차 리뷰): P2 궤적의 77행 입력 재실행 → ba8b rate 생존 → `select_discrepant` 10건 → coverage 라인 "77 of 77" 확인. **발견 7**: replay 산출물에 **드롭 사유별 계수**(quote-날조/핀-비포함/~~앵커-불성립~~/**표-조회 실패 — 공집합(판단된 무대응) vs 공백(미저작)** 분리/kind-결측/sub-미채움) 포함 — 실패 시 엔진 결함인지 sub 채움 품질인지 표 미비인지 즉시 귀속 가능하게(리뷰 세부 7).
4. **라이브**(유료·승인 후): 다음 Y2 계열 런에 `T2_QUOTE_PIN=1` 편입([[19]] 합성 스택). **판정은 pass 비교가 아니라 기전 지표**(C272 시간-표류 교훈): 022 discrepancy 집합 정확도(replay-대조)·**Δspurious 양방향** — false-abstain(드롭 사유 로그 중 semantic 매핑 실재 건수) / false-apply(category-마크·named-강등의 gold-diff — *계측만*, A2 수정 근거로 사용 금지 [[23]]). 스모크로 신규 마크 라이브 발화 확인 후 full([[30]]).
5. **승격/기각 임계(사전등록)**: 회귀(§5-9) 0 ∧ 022 replay 10/10 → go_stack ON 승격. 라이브 false-apply 신규 발생이 false-abstain 회수분을 초과하면 OFF 원복(플래그 롤백=1줄).

## 7. 상쇄·위험 (§1.3 — 이 레버가 파는 것·R5 반영 갱신)

1. **category 오선언**: 앵커 검사를 우회하는 문 — 닫힌 검사로 못 막는다(브랜드-됨=열림). 단 R5 채택으로 "선행 위치의 범주어 핀"(코퍼스 실측 0)으로 표면이 좁아졌고, 케이스 5의 주 경로(비선행 범주어)는 재질의→category 재선언으로 흡수. 대가 지불 방식 = 표면화(마크)+Δ계측+장기 learn. **숨은 실패가 아니라 감사 가능한 주장**이라는 것이 현행 대비 순개선.
1b. ~~R5 앵커의 자기 잔여 2종~~ → **rev2에서 소멸**(앵커 철회·표 멤버십이 케이스 10·11을 저작 시점 판단으로 흡수). **남는 원칙은 유지**: 이 아크에서 문자열 규칙 **4개**(토큰-교집합·대문자·토큰-최대·**선행 앵커**)가 전부 열린 술어 동결로 기각됐다 — 잔여는 규칙이 아니라 **유한 열거(표)·계측·learn**으로. 새 잔여 = **표 자체의 품질**(저작 오류·미저작 공백)이며 이는 감사 가능한 데이터라 규칙보다 다루기 쉽다.
2. **스키마 부하 +2필드**: formalize 품질 저하 가능(스키마 비대·§2h max_batch 상호작용) — 계측: operand 채움률·range_retry 빈도(기존 로그 라인 재사용).
3. **재질의 1회**: sub 호출 증가(행-단위·bounded) — 계측: inject 라인 수 diff.
4. **표면화 자체의 역효과**(C270⚠ 동형): category-마크·unverified 문구가 agent의 over-action(무근거 재시도 남발)을 유발하는지 — Δspurious에 "동일-인자 재호출 3회+" 지문 포함.

## 8. 리뷰 판정 (2026-08-01 사용자 리뷰·반영 완료)

- **R1 동의(확정)**: P1(policy-핀)로 시작·P2(양측-핀 별도-문맥)는 Δ실측 후 승격. C276②가 policy-핀 단독 회수를 이미 보임(**수치는 9/9 → C276★① 정정 후 6/6**). ⚠rev2에서는 P1/P2 구분 자체가 무의미해진다 — 행-측 핀도 표 멤버십이 대체.
- **R2 동의(확정)**: category = 통과+마크. abstain은 정당 공동-제외(General Retailers→Target·C276⑥ 축자)까지 죽여 false-abstain 재생산.
- **R3 조건부 동의(확정)**: 문구 분화는 버그픽스로 독립 적용하되 **"즉시"=repo 반영·다음 런부터** — 진행 중 라이브 런(pass3)에 배포 금지(런 내 비교가능성).
- **R4 동의(확정)**: guard-불성립 행만·`sg_inject_retry` 경로 재사용·추가 예산 0.
- ~~**R5 채택**~~ → **☠R5 철회(2차 리뷰·rev2·§8b ⒜)**. 1차 리뷰의 "위치 관계라 닫힘" 방어는 **계산가능성(결정론 검사 가능)과 [[22]]의 닫힘(변이 불변)을 혼동**한 것이고, 리뷰어가 그 오류를 자인했으며 나도 그 프레이밍을 수용한 책임이 있다(**4번째 회귀·이번엔 사용자 지적이 아니라 논거에 넘어간 경우**). 같은 비판이 `pin ∈ merchant` 평범 포함에도 적용 ⇒ 유사도·접두·포함 **전부 제거**가 일관된 귀결.
- **발견 2~7 반영 위치**: 2→§2a 출처 의무(gaming subscriptions 리모트 grep 확인 완료) · 3→§2a instructions 재작성(지시-대상 기준) · 4→§2a/§2b A2 문구 템플릿 · 5→§2c 엣지 규칙 · 6→§2b/§5-7b · 7→§6-3 드롭-사유 계수.

## 8b. ★rev2 제안 — R5 철회 + 정책-참조 상인 식별표(사용자 제안·2026-08-01 저녁·리뷰 대기)

**계기**: 사용자 지적 2건 — ⑴ "패턴 매칭 안 하기로 한 것 아닌가"(→ `row.startswith(pin)`=지시 동일성을
문자열로 동결·**R5는 4번째 회귀**이며 리뷰어의 "위치 관계라 닫힘" 방어를 내가 수용한 것이 오류) ⑵ "A2에
별도 약어집을 두면 안 되나".

**⒜ R5 철회**: `pin_anchor` 폐기. 근거 = 술어가 묻는 것이 *연산의 모양*(위치)이 아니라 *지시 동일성*(열림)
이고, 실제로 `'Target'`→`'Targeting Solutions'`로 깨졌으며 케이스 11은 지금도 통과한다. **가드 계열의
실측 가치도 음수**: 라이브 발화 3종 중 **2종이 오차단**(ba8b=022 상실 · `txn_a8f1c2d3e404` Microsoft 365 =
Business Silver 정책이 "Hardware/Electronics Merchants" 아래 Microsoft를 실제 제외 ⇒ 서브가 옳았다·db+KB
확인)·1종만 옳음(Thrive).

**⒝ 정책-참조 상인 식별표(A2)** — 열린 술어를 **열거로 닫는다**([[16]] A2 EXT "닫힌 3종"과 동류·enum 상수의
연장). 형식 = `policy_merchant_rows: {<정책이 쓰는 이름>: [<카탈로그 merchant_name…>]}`.
엔진 검사 = **집합 멤버십**(문자열 유사도·접두·포함 전부 불필요):

| 케이스 | 표 조회 | 결과 |
|---|---|---|
| ba8b | "Target" → {Target - Eco Collection, Target} | 행 ∈ 집합 → **통과**(회수) |
| 019 | "ThredUp" → {ThredUp} | Thrive Market ∉ 집합 → **차단**(현행 유일 정당 발화 보존) |
| Microsoft 365 | "Microsoft" → {Microsoft 365} | 통과(회수) |
| 케이스 11 | "LinkedIn Learning" → 카탈로그 대응 0 | 어떤 행도 ∉ → **차단**(⚠현 설계가 못 막던 잔여가 닫힌다) |
| 범주 주장 | 표에 없음 | category 경로(마크·미검증) — 불변 |
| 표 미수록 상인 | 조회 실패 | abstain+표면화(안전측·표 갱신 신호) |

**규모 실측(gold-free·`x28` 계열)**: 제외-문맥 불릿 고유명 **30개** 중 카탈로그 대응이 있는 것 **11개** —
`Amazon·Apple(→Apple Store)·Coursera·Dell(→Dell, Dell Technologies)·Microsoft(→Microsoft 365)·Salesforce·
Slack(→Slack Technologies)·Target(→Target - Eco Collection, Target)·ThredUp·WeWork·Zoom(→Zoom Video)`.
⇒ 표는 **카탈로그(138) 크기가 아니라 정책 언급 수에 유계**. 기존 rate 표(`Gold Rewards Card: 2.5`)와 같은
A2 도메인 데이터 등급.

**☠⚠️치명적 주의(자기 플래그·[[03b]])**: 위 11개 목록은 내가 **`norm(m)==n or startswith(n+" ")` = 방금
기각한 접두 규칙으로 기계 생성**한 것이다. 이대로 A2에 실으면 **폐기된 휴리스틱의 출력을 데이터로 동결**
하는 것이라 런타임에서 뺀 의미가 사라진다. ⇒ **측정치(=11·유계성)만 채택하고 내용은 재작성**한다:
표 내용은 정책 문서와 카탈로그를 보고 **판단으로 저작**(사람 또는 LLM 저작+사람 검토)하며, 접두 규칙
산출물을 그대로 쓰지 않는다. 검수 항목 = 접두로 안 잡히는 쌍(비-접두 별칭)이 있는지·접두로 잡히지만
실은 다른 업체인 쌍이 있는지.

**⒡ [[23]] 출처 판정 — ✅사용자 승인 완료(2026-08-01)·★2차 리뷰 조건 1에 따라 *층 분리***:

| 층 | 내용 | 출처 범주 | 승인 |
|---|---|---|---|
| **층1** | 제외-그룹 → 정책이 열거한 멤버 이름(`General Retailers → {Target, Walmart, Amazon}`) | **env 기계도출**(제목-깊이 상속 파서로 축자 추출·x28 동류) | 기존 범주·**신규 승인 불요** |
| **층2** | 정책 이름 → 카탈로그 행(`Target → {Target - Eco Collection, Target}`) | **운영자 지식**(지시 동일성 판단) | **신범주·승인 대상 ✅승인됨** |

즉 승인이 실제로 소비되는 것은 **층2뿐**이고 층1은 감사가 축자 대조로 끝난다. (현실 정합: 은행은 merchant
normalization/DBA 테이블을 운영 자산으로 갖는다 — 벤치 편법 아님.) **승인 조건 3(전부 필수·이행 완료)**:
1. **전수 저작·실패-사례-주도 금지.** 30 불릿 × 카탈로그 **전수**로 저작한다. 동기가 된 ba8b·MS365 두 쌍만
   고치는 저작은 **사실상 gold 경유**다(그 두 쌍은 gold-측정된 라이브 실패에서 왔다). 전수 저작이라야
   내용의 출처가 env(정책 불릿×카탈로그)+세계지식으로 정화된다.
2. **엔트리별 provenance** `_note_`: 정책 불릿 축자 ↔ 카탈로그 행 · 저작 근거 · **gold 미열람** 명기.
3. **갱신 루프 규율**: 갱신 신호는 **표면화된 abstain 로그**(핀·quote는 env-측 실재)→env 재판단으로만.
   **라이브 gold-diff를 표 갱신 근거로 쓰는 순간 [[23]] 위반**이다. (이 문장을 A2 `_note_`에도 복사한다.)

**[[05]] 비용**: A2 **+30 엔트리**(대응 11 + 무대응 19·아래 ⒢-1). 엔진 순증 = 멤버십 조회 1회(리터럴 0).
전이 = 새 도메인마다 표 저작 — **A2-swap 범위 내**이나 capex 순증이므로 특허 §유한성 서술에 반영 필요.

**⒢ 확정 스펙(2차 리뷰 세부 1~4 반영)**

1. **30 불릿 전수를 명시 엔트리로** — 대응 있는 11개(비공집합) + **대응 없는 19개를 `[]`로 명시**.
   이러면 조회 결과가 결정론적으로 셋으로 갈린다: **행 ∈ 집합**=통과 / **집합은 있는데 행 ∉**(공집합 포함)
   = **판단된 무대응 → 차단**(정당·abstain 소음 없음·케이스 11이 여기) / **키 자체가 없음** = **표 공백**
   → 재질의 1회 → abstain + **갱신 신호**.
2. **조회 실패 시 재질의 1회 유지**(R4 일관): 핀이 조각-복사("LinkedIn"만)거나 오타일 수 있으므로
   A2 `retry_prompt`("copy the WHOLE name") 1회 후 abstain.
3. **표 키 대조 = `norm` 정확-동등**(포함 아님). 포함을 되살리면 표의 의미가 사라진다.
4. **저작 파이프라인 = LLM 저작 + 사람 검토** 권고(도메인당 capex↓ = 전이 비용 논거 유지·검토 로그가
   그대로 provenance). 검수 항목 = 비-접두 별칭 쌍이 누락됐는지 · 접두-동형이지만 실은 **다른 업체**인 쌍이
   섞였는지(예: `Dell`↔`Delta …`, `Target`↔`Targeting …`).
5. **A2 형태**(`quote_pin` 내부·바인딩 국소화):
   `"policy_group_rows": {"<정책 제외-그룹 이름>": ["<카탈로그 merchant_name>", …]}`
   — 키는 단일 상인 불릿(`"Target"`) **또는 열거를 동반한 제목**(`"Thrift and Resale Markets"`·⒞).
6. **★단일 라우팅 표(2차 리뷰 조건 3 — ⒝의 옛 조회 표를 대체)**: kind × 표-상태 6칸.

| | **수록·전수 열거** | **수록·비전수/개방-표지** | **미수록** |
|---|---|---|---|
| `named_merchant` | 행∈집합=**pass** / 행∉(∅ 포함)=**reject_member**(확정·재질의 없음) | 표에 **키로 넣지 않는다**(⒢-7) ⇒ 미수록으로 흐름 | **lookup_missing** → 재질의 1회 → abstain + 갱신 신호 |
| `category` | 같음(그룹-키 조회·019가 여기서 닫힌다) | 위와 같음 | **category**(통과+마크·R2·**열린 잔여 정직 보존**) |

7. **★전수성 판단은 새 열린 판단이다(2차 리뷰 조건 2·핵심)**: 열거를 *기각 근거*로 쓰는 것은 그 열거가
   **전수일 때만** 정당하다 — 예시-열거("including/such as")면 미열거 정당 구성원 기각 = **ba8b-동형
   false-abstain 재생산**(=표면-규칙성 동결의 5번째 후보). 그래서 가정하지 않고 **쟀다**:
   **제외-그룹 25 = 열거형 21 / 산문형 4 · 열거형 21개 중 개방-표지 0개**(결정론 census·계약문도 축자로
   *"The following **specific merchants**…"* = 예시가 아니라 지정). ⇒ 이 ABox의 열거는 닫혀 있다.
   **전이 규율**: 새 ABox마다 이 census를 다시 돌리고, **개방-표지가 있는 그룹은 키로 넣지 않는다**
   (미수록 → category 마크로 안전측). 판정과 근거는 A2 `_note_exhaustive`에 영속.
8. **키 모호성 규칙**: 여러 문서에 반복되는 총칭 제목(`Common exclusions`·`Important exclusions`)은
   **키 금지**(구성원 집합이 갈린다). 도메인-유일 제목만 키(예: `What is excluded from the higher
   sustainability points rate on EcoCard`).
9. **★"회수"는 보장이 아니다(2차 리뷰 조건 3)**: 닫힌 검사가 보장하는 것은 **오적용 차단까지**다.
   019에서 rate 5로의 *회수*는 재질의에서 서브가 제외 주장을 철회하는 데 의존한다(확률). 서브가 같은
   주장을 반복하면 그룹 검사가 다시 기각 → abstain(행 상실·단 **표면화됨**). ⇒ 픽스처·판정 기대를
   **2단으로 분리**: 차단=결정론(사전등록 임계) / 회수=재질의 성공 시(관측·계측 대상).

**⒞ ★범주도 대부분 닫힌다(추가 실측)**: 이 KB의 제외 범주는 **거의 전부 구성원을 열거**한다 —
`### General Retailers → {Target, Walmart, Amazon}` · `### Thrift and Resale Markets → {ThredUp}` ·
`**Gaming Subscription Merchants** → {Xbox Game Pass, PlayStation Plus, Nintendo Switch Online}` ·
`**Hardware/Electronics Merchants** → {Apple, Microsoft, Dell}`. ⇒ 표의 키를 **정책 제외-그룹**(단일 상인
불릿 *또는* 열거를 동반한 제목) 단위로 두면, named든 category든 **하나의 멤버십 조회로 해소**된다.
019가 그 실례: 서브가 "Thrift and Resale Markets"를 근거로 삼아도 그 그룹의 구성원은 {ThredUp}뿐이라
`Thrive Market ∉` → **차단**. ⇒ §5 케이스 5·11 + 019가 전부 같은 닫힌 검사로 닫힌다.

**미해결(진짜 잔여)**: **열거 없는 산문 범주**만 열려 있다 — 예: silver_003 *"Vacation rentals or
home-sharing platforms coded under real estate or 'miscellaneous' categories"*(구성원 목록 없음).
이 경우만 category 경로·표면화 유지(R2 불변)·Δ계측.

**⒟ 019의 실제 해악(수치)**: `txn_f093f96e2001` = EcoCard·Green·$175.00·recorded **175 points**.
정상 rate 5 ⇒ expected **875** ≠ 175 = **진짜 discrepancy**. 서브가 exclusion을 적용하면 rate 1 ⇒
expected 175 = recorded ⇒ **불일치가 사라진다(false-negative)**. 즉 이 오적용은 *있어야 할 dispute를
지운다*. 현행 가드는 그것을 막는 대신 rate를 드롭해 **행을 미판정으로** 만들었다(역시 dispute 없음·단
"1 row could not be verified"로 표면화는 됨). ⒝⒞의 표는 이 행을 **판정 가능 상태로 되돌린다**(제외 기각
→ rate 5 → discrepancy 검출).

**⒠ 미확정(정직)**: 서브가 ⑴두 회사를 동일 업체로 착각했는지 ⑵"Thrift and Resale **Markets**" 범주에
`Thrive **Market**`이 속한다고 판단했는지는 **현 로그로 구분 불가**(isolate trace 미설정·서브 출력이
궤적에 없음). 구조적으로 ⑵가 유력(제목이 범주형·이름에 "Market" 공유). **pin_kind 필드 자체가 이 구분을
처음으로 관측 가능하게 만드는 계기**다 — 서브가 named인지 category인지 선언하므로. 확인 프로브(무료·
GPU 필요) = pass3 완주 후 1콜.

## 9. 구현 순서 (rev2·2차 리뷰 조건 5 반영 — **①~⑥ 완료·⑦⑧ 미착수**)

**단계 0 = 표 저작(신설·조건 5)** — ✅완료: ⓐ층1 기계 추출(제목-깊이 상속 파서 재사용·축자) → ⓑ그룹별
**exhaustive 판정**(census 21/0·근거 축자를 `_note_exhaustive`에) → ⓒ층2 카탈로그 대응 **판단 저작**
(LLM 초안+사람 검토·**접두 산출물 불사용**·검수 항목 = LinkedIn/Delta/AWS/Thrive/Electronics 5쌍) →
ⓓ**30 불릿 전수를 명시 엔트리로**(대응 없음 = `[]`로 명시 ⇒ 조회 실패의 의미가 "판단된 무대응"과
"표 공백"으로 결정론 분리) → ⓔ도메인-유일 상위 제목 1건 추가(⒢-8 모호성 규칙 준수).

① 엔진 `_quote_pin_check` 표 멤버십 교체 + `_split_missing_fields` 문구 분화 ✅ → ② A2 `policy_group_rows`
(46 엔트리)+문구 템플릿 ✅ → ③ `test_c278_quotepin.py` **30검정**(케이스 3·4·5·11 기대를 표-의미론으로
재작성·케이스 12 신설) + 회귀 9종 ✅ → ④ **x29 표-시뮬** ✅(아래 임계) → ⑤ 커밋/push/리모트 배터리 ✅ →
**⑥ 022 replay(미착수·`_isolate_trace` ON 필수)** → **⑦ 스모크→라이브 편입(미착수·승인 필요)**.
롤백 = `T2_QUOTE_PIN` OFF(1줄).

**④ 사전등록 임계(조건 4)·실측 결과**: 진짜 **6/6 회수** ✓ · **허위 3종 비회수** ✓(`home`↛Home Depot·
`Electronics`↛Electronics Express Miami·`LinkedIn Learning`↛LinkedIn Ads — *앵커는 이 3종을 잘못 회수했다*
= 표의 우월성이 가장 선명한 검정) · 019 차단(named·category 양 경로) ✓ · 케이스 11 차단 ✓ ·
**비선행 범주어 49종 전부 lookup_missing** ✓ · 접두-동형 타업체 4쌍 비통과 ✓ ⇒ `x29_table_sim.py` PASS.
⚠**시뮬 하네스 자기결함 1건**: 1차 실행서 실 A2 필드명 대신 축약 키를 넘겨 **전건 kind_missing**이 났다
(계측기를 먼저 의심하라 — C276★① 교훈 재적용). 수정 후 재실행.
