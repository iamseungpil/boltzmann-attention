# Lie-대수적 추상화 위계 — LLM이 잘하는 것 vs 못하는 것의 수학적 골격 (이론 DRAFT) — 2026-06-17

> 동기: τ² write-벽 전수추적서 "best/most/더 밝은" 류가 실패. 이건 *단어 하나*가 아니라 **리스트-구성 → 정렬 → 첫째 선택**이라는 *연산을 품은 개념*이다. = 십진 위치표기가 무한집합을 유한과정으로 바꾸는 것과 동형. 추상화 *레벨*이 LLM 적성을 가른다는 thesis를 Lie 군으로 형식화.
> 상위/연결: Olver 표면군 저차원불변(`EXPERIMENT_DESIGN §5.13-5.14`)·`ALGEBRAIC_DERIVATION_CLOSURE`·`PRIMITIVE_COVERAGE_MATRIX`·[[feedback-nl-formalize-llm-selection-deterministic]]·[[project-decomposition-optimality-contribution]].

## 0. 한 줄
**모든 연산 O는 두 층으로 쪼개진다: 유한차원 *대수* 𝔤(생성원 = 추상화·도메인불변·저차원)와 그것이 지수사상 exp으로 *생성*하는 *군* G(궤도 = 실행·무한/비유계). LLM은 *대수*(생성원을 *명명*)에 강하고 *군 실행*(궤도를 *적분/순회*)에 약하다. 결정론은 군을 실행(exp을 돌린다). 추상화 레벨 = 생성원 위계서의 높이(가해성). 학습 가능한 것 = 생성원 명명(저차원·전이)이지 궤도 실행이 아니다.**

## 1. 두 층 — name/algebra vs value/execution
| | 대수 𝔤 (NAME) | 군 G = exp(𝔤) (EXECUTE) |
|---|---|---|
| 정체 | 생성원 = 규칙 = "어떤 연산·어떤 순서관계·어떤 파라미터" | 생성원을 데이터 위에 *적분*한 결과 = 실행된 답 |
| 차원 | **유한·저차원**(Olver n−s 불변) | 무한/비유계(궤도 크기·리스트 길이) |
| 도메인 | **불변**(생성원은 도메인-일반) | 구체(이 카탈로그·이 값) |
| 누가 | **LLM** (명명·인식·전이) | **결정론** (exp을 돌림) |
| 실패모드 | 생성원 오인(드묾·저차원이라) | 환각·날조·중도단락(무한과정 실행 실패) |

**핵심 정리(분담 경계)**: LLM은 **𝔤를 emit**(연산의 생성원 명명)하고, 결정론 엔진은 **exp(𝔤)를 실행**(궤도 순회). 경계는 "이 연산을 *명명*하라(대수)" vs "이 연산을 *완수*하라(군)"이다.

## 2. ★정준 예: 위치표기 = 무한을 유한 생성원으로 (네 비유의 형식화)
- 십진/아라비아 표기: 곱하기-base = **1-파라미터 부분군의 생성원**(자리이동). 수 = 생성원들의 *유한 단어*(자릿수열). ℕ(무한)이 {유한 기호 × 위치}로 생성됨 = `exp(Σ dᵢ Xᵢ)`.
- **수의 *이름*(자릿수열)은 대수 = 유한·조작가능. 수의 *값/크기*는 exp을 돌려야(Σdᵢbⁱ) = 군 실행.**
- **실증 검증**: LLM은 수를 *읽고 쓴다*(이름=대수 잘함) but **여러 자리 산술서 무너진다**(값=군 실행 못함). = 정확히 algebra-good / group-execute-bad 분리. 위치표기는 *고추상*(저차원 생성원) 인코딩이라 명명이 쉽고 실행은 기계로 offload 가능. 탈리/로마숫자는 *저추상*(압축 없음)이라 둘 다 어려움.
- **함의**: 올바른 IR = 연산의 *생성원을 노출*하는 표기(무한을 유한단어로). 잘못된 IR(우리 정적 $select)은 생성원이 아니라 *전개된 결과*를 강제 → 군 실행을 LLM에 떠넘김 → 실패.

## 3. ★"best/most/더 밝은"의 분해 (τ² 실패의 정체)
"best" = 순서관계 ≼의 **extremum 연산**. 군적으로:
1. 리스트 구성(집합) — Sₙ-대칭(무순서).
2. 순서 ≼ 부과 = **대칭 깨기**(Sₙ → 정렬된 canonical rep). ≼ 자체("brightness 오름"·"price 내림") = **생성원**.
3. extremum 선택 = 궤도의 canonical 원소 = exp 실행(정렬+첫째).
- **LLM-good(대수)**: "best=extremum이고 순서는 brightness↓다"를 *인식·명명* — 유한·도메인일반·저차원. **순서관계가 생성원.**
- **LLM-bad(군)**: 100개를 *실제 정렬*해 max 뽑기 = 궤도 순회. → 결정론.
- **τ² 실증(전수추적)**: t52 "max zoom"·t64 "highest res"·t29 "most expensive"가 실패한 건 *순서를 몰라서가 아니라* 우리가 **정적 `{attr:val}`로 *전개된 결과*를 강제**해 LLM에 군-실행을 떠넘겼기 때문. **처방 = LLM이 생성원 `{op:extremum, order:zoom↓}`만 emit·엔진이 정렬+선택.** (availability-forced t9/94/107도 동일: gold이 keep-rest 위반 = 궤도가 가용성으로 굴절 = 실행은 엔진이.)

## 4. ★추상화 *레벨* = 생성원 위계의 가해성(solvability)
연산의 어려움 = 생성원 대수의 구조적 복잡도. Galois/Lie 가해성 위계로:
| 레벨 | 대수 구조 | 예 | LLM/결정론 |
|---|---|---|---|
| L0 | 자명 {e} | 복사·identity(값 주어짐) | LLM 자명 |
| L1 | abelian(가환) | 독립 필터·exact match | LLM 명명·엔진 실행(병렬) |
| L2 | nilpotent/solvable | extremum·sort·조건 fallback(순차 비교 = 유한 닫힌 절차) | **LLM 생성원 emit·엔진 exp** |
| L3 | 합성·비가환 solvable | 중첩 조건·다단 fallback·관계참조 | LLM 생성원 *단어* emit(저차원이면)·엔진 실행 |
| L∞ | **비가해(non-solvable)** | 무계 탐색·닫힌형 환원 없음 | **둘 다 어려움**(thesis 밖) |
- **thesis가 사는 영역 = 가해(solvable) 구간**: 생성원이 유한 가해 구조 → *유한 생성원-단어*(LLM 명명가능) + *유한 결정론 실행*. tool-use control/data-flow(P1-P9)가 이 구간이라는 게 `ALGEBRAIC_DERIVATION_CLOSURE`의 "연산자 닫힘" 주장 = **유한차원 가해 대수**라는 뜻.
- **"abstraction level이 LLM good/bad를 가른다"의 정확한 의미**: LLM은 *대수 레벨*(생성원 명명)까지 강하고 *군 레벨*(궤도 실행) 위는 약하다. 한 연산이 "명명하라"로 제시되면 LLM 영역, "완수하라"로 제시되면 군-실행 = 결정론 영역. **레벨을 *낮추는*(군→대수로 lift) 것 = 설계의 본질**(생성원 emit으로 들어올림).

## 5. ★왜 *작은* 모델로 충분한가 (thesis 결박)
- 𝔤는 **저차원**(Olver n−s 불변·생성원 몇 개). 소형 모델도 저차원 대수를 *담을* 용량 충분. scale이 사는 건 *군 실행*(암기·다단 적분 근사)인데 그건 **결정론에 offload**하니 scale 불요. = M-A floor sweep "binding 벽 ≠ scale"의 *이유*: binding이 군-실행이면 scale도 못 풀고(무계), 대수면 소형도 됨. **핵심 = binding을 *군-실행*으로 강제하지 말고 *생성원 emit*(대수)로 재표상.**

## 6. ★학습가능성 (딥리서치 질문의 이론적 답)
- **학습 대상 = 𝔤-식별**(NL → 어떤 연산·어떤 순서관계·어떤 생성원). 이건 **저차원·도메인불변 불변량** → *학습가능·전이가능*(ABox-swap). 군 실행은 학습 안 함(엔진).
- **분류/라우팅(직전 턴)의 정체 = "이 부분은 대수(명명·LLM)냐 군(실행·결정론)이냐"를 LLM이 식별** = 𝔤로의 사영(projection onto algebra). 이 사영이 저차원이라 학습·전이된다는 게 예측.
- **예측(반증가능)**: (a) 생성원 emit으로 재표상하면 superlative/조건 성공률이 base in-head을 *넘는다*(LLM 명명 + 엔진 실행 > LLM 군-실행). (b) 𝔤-식별 정확도는 scale-둔감(저차원)·실행 정확도는 결정론이라 100%. (c) 𝔤-식별은 도메인 전이(불변). (d) 비가해 연산(무계 탐색)서만 둘 다 무너짐 = thesis 경계.

## 7. 설계 함의 (P4 재설계 = 군→대수 lift)
1. **IR = 생성원-단어**: `{op, order/relation, params}` (extremum·filter·copy·fallback-seq). 정적 결과 아님. LLM이 *연산을 명명*.
2. **엔진 = exp 실행기**: 명명된 생성원을 실제 리스트 위에 적분(정렬·필터·extremum·가용성) + grounding(날조차단).
3. **LLM-분류 = 𝔤-사영**: 요청을 생성원(대수=LLM)과 실행(군=엔진)으로 분해. *전체 context 위에서* 명명(필터 먼저 금지 — 명명은 전체 위에서).
4. **측정 = 군→대수 lift가 base를 넘나**: (LLM 생성원 emit + 엔진 실행) vs (base in-head 군-실행) vs (정적 IR = 군을 LLM에 떠넘김). 3-way.

## 7b. ★재정초 (2026-06-17 교정) — Lie/산수는 특수예·진짜 축 = *표기가 알고리즘을 내장하는가*
사용자 교정: Lie-산수 비유에 갇혔다. 핵심은 **표기(notation)가 *denotation만* 하는가 vs *알고리즘을 내장*하는가**다.
- **denotation-only 표기**(로마/한자 숫자·리터럴 값·개체명): 기호가 *사물을 가리킴*. 조작 절차 없음. → **결정론 ground/lookup**.
- **알고리즘-내장 표기**(아라비아 위치표기·"best/most/more-than"·비교급·양화사): 한 기호가 *절차를 압축*(집합구성→순서→extremum) — *기호 조작으로 연산*. 아라비아 숫자가 수학을 가능케 한 이유 = **표기에 산술 알고리즘이 결합**(로마숫자엔 없음). → **그 절차를 펼쳐 실행**.
- **난이도 축 = 내장 절차의 *복잡도류*** (van Benthem **semantic automata**): "every/some"=유한상태(자명)·**"most/짝수개"=계수 필요**(push-down↑)·superlative=리스트+전순서+extremum·"제약하 최적"=무계탐색. **양화사/비교급이 *오토마타 복잡도류를 내장*한다는 게 "단어가 알고리즘을 품는다"의 엄밀판.**

### 분류(LLM)의 정체 — 3단
1. **denotation vs 절차-내장** 인식(기호의 종류).
2. 절차-내장이면 **절차 *타입*과 복잡도류** 인식(어떤 semantic automaton).
3. 라우팅: denotation→결정론 ground / 절차→(저복잡=유한상태면 LLM 직접·계수↑/무계면 절차-spec emit→엔진 실행).

### ★학습가능성의 정답 (딥리서치 질문)
**절차-타입을 *인식*(분류)하는 것은 저복잡도** — 기호의 범주를 *읽는* 것이지 *실행*이 아니다(van Benthem: 오토마타 *타입*은 유한 라벨·데이터 위 *실행*이 무계). ⇒ **소형 학습자가 고복잡 절차를 *실행* 못 해도 *분류*는 학습가능·전이가능.** 이게 "LLM이 잘하는 것/못하는 것의 구분을 학습할 수 있나"의 이론적 답 = **구분(절차-타입 인식)은 denotational-저차원이라 학습됨·실행은 결정론.**

### 선행연구 결박 (딥리서치 `w3d906s6n` 체계화 대상)
- 수학사: Cajori·Netz·Rotman·Nesselmann(수사→음절→기호)·Iverson "notation as tool of thought"·Leibniz characteristica.
- 언어학: 절차적 vs 개념적 의미(Relevance Theory·Blakemore)·**van Benthem semantic automata**·일반양화사(Barwise-Cooper)·동적의미론(meaning-as-instruction).
- 기호학: **Goodman 표기 이론**(notationality=모호성없는 기계조작 가능조건)·Peirce 도형추론.
- 인지: **표상효과**(Zhang&Norman·왜 아라비아 곱셈이 쉽고 로마는 어려운가)·외재인지.
→ Lie(§1-7)는 *연속/생성* 특수예·이 표기론(절차-내장·복잡도류·notationality)이 *일반 골격*.

## 7c. ★표기 깊이(notation depth) — 측정법 (2026-06-17 사용자 지시)
NL 표기는 *깊이*가 다르다: 상형문자처럼 대상에 **1:1 매핑(denotation·깊이0)**부터, **알고리즘·추상화를 내장한 표기**(깊음)까지. **깊을수록 LLM의 탐색·분류가 어렵다.** ⇒ 깊이 `d(e)`를 *측정*해야 — 그게 LLM-실행가능 경계를 *예측*한다.

### 정의: d(e) = e를 denotational normal form으로 펼치는 *연산자-중첩 깊이*
e를 *절차 골격*으로 파싱해, **순수 denotation(값·개체 lookup)에 닿을 때까지의 unfold(환원) 단계 수** = 중첩된 연산자(양화·extremum·비교·집계·조건)의 깊이.
| d | 정체 | τ² 예 |
|---|---|---|
| **0** | 순수 denotation(1:1) | "주문 #W123"·"the keyboard" (이름→사물) |
| **1** | denotational 항 위 *연산자 1개* | "clicky 스위치"(필터)·"the cheapest"(extremum) |
| **2** | 연산자∘연산자 | "the cheapest **waterproof**"(extremum∘filter)·"현재보다 **less bright**"(anchor-비교) |
| **≥3** | 중첩·관계 | "두번째로 싼 방수 + 현재보다 가벼운"·다단 조건 fallback |

### 3개의 *수렴하는* 형식 측도 (다 같은 깊이를 다른 각도로)
1. **연산자-중첩 깊이 / quantifier rank** — 논리형의 최대 중첩(superlative=∃∀ rank2…). *구문적·계산가능.* (1차 측도·실용.)
2. **의미 타입 차수(type order)** — entity(0)·predicate⟨e,t⟩(1)·양화/extremum⟨⟨e,t⟩,·⟩(2)·중첩(3+). *더 추상 객체 위 연산일수록 깊음.*
3. **van Benthem semantic-automata 류** — 각 연산자의 *평가 복잡도*(유한상태<계수<…). *깊이의 계산적 무게* — 같은 rank라도 "most"(계수)가 "every"(유한상태)보다 깊음.
→ **정초 측도 = Bennett의 *logical depth***(콤팩트 기호를 denotation으로 펼치는 *계산시간*) = "기호에 *압축된 알고리즘의 양*". 위 1-3은 그 계산가능 대리. 상형=O(1)·"best over n"=O(n log n)·중첩=합성.

### ★측정 프로토콜 (실측·반증가능)
1. τ² 각 요청 e를 절차골격으로 파싱 → `d(e)` 계산(rank + type-order, automata류로 가중).
2. **`d(e)` vs LLM per-case 분류/선택 실패율 상관**(전수추적 데이터 재사용). 
3. **예측**: 실패율 ↑ in d·**임계 `d*`**(LLM 분류 지평) 존재·결정론 엔진은 **d-불변**(어떤 깊이도 실행). ⇒ **분담 경계 = `d*`**: LLM은 `d ≤ d*` 골격 인식·emit / 엔진은 실행.
4. **scale 의존성**: `d*`가 모델크기로 *얕게* 오르나(§4b sweep과 교차)·아니면 *결정론 offload*가 유효 `d*`를 ∞로(깊이는 엔진이 흡수). thesis = 후자.

### IR 설계 함의 (Arabic-numeral 교훈의 정량화)
**좋은 표기 = 깊은 연산을 *얕게* 조작가능케 함**(아라비아 숫자가 곱셈을 얕은 기호조작으로). 우리 정적 `$select`는 깊은 연산을 *LLM에 깊게* 떠넘겨 실패. ⇒ **IR 목표 = LLM-대면 `d`를 최소화**(절차를 얕은 생성원-단어로 노출·실행 깊이는 엔진이 흡수). `d(e)` 측정이 *어떤 IR이 깊이를 더 얕게 만드나*의 비교 척도가 된다.

### 측정 도구 (구현)
`notation_depth.py`(신규): τ² 요청 → 연산자-골격 파싱(superlative/comparative/filter/conditional/relational 태깅) → d(rank·type-order·automata가중) → per-case d + 실패율 상관표(`M_A §14`). 딥리서치 `w3d906s6n`(semantic automata·Goodman notationality·logical depth)가 측도 선택·가중을 정련.

## 8. 정직 (이론 지위)
- 이건 *생산적 형식틀*(Lie 대수↔군 = name↔execute의 동형)이지 증명된 정리 아님. "가해성=추상화레벨"은 *은유적 위계*(엄밀 Galois 군 아님)·반증가능 예측(§6)으로 검증.
- 엄밀화 경로: (i)tool-use primitive를 실제 유한생성 대수로 구성(닫힘 companion 확장) (ii)생성원-수 = eff-dim 측정(olver_dimension_experiment 재사용) (iii)§6 예측 실험.
- 한 줄: **LLM=대수(생성원 명명·저차원·학습·전이)·결정론=군(exp 실행·무계). 추상화 레벨=가해성 위계. 설계=군을 대수로 들어올려 명명은 LLM·실행은 엔진. 십진 위치표기가 무한을 유한 생성원으로 바꾼 바로 그 수.**
