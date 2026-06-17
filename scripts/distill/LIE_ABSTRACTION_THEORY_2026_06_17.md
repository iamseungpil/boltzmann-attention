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

## 7d. ★시간(절차) vs 공간(파라미터) — 깊이를 *어디에* 지불하나 (thesis의 아키텍처적 근거·2026-06-17)
사용자 통찰: 빅모델이 작은모델보다 깊은 표기를 잘 매핑하는 건 *시간적 절차*가 아니라 *공간적 파라미터*를 키워 극복한 것. 그리고 시간↔공간은 *LLM 아키텍처의 호환성*에 기초. → Transformer 복잡도이론으로 형식화하면 thesis의 *뿌리*가 나온다.

### 사실 1: forward pass = *유계-깊이* 병렬회로
- log-정밀 Transformer 1회 forward = **uniform TC⁰**(상수-깊이 threshold 회로·Merrill-Sabharwal). *직렬 계산깊이 ≈ layer 수 L*(상수)·나머지는 병렬(width).
- ⇒ **forward pass의 직렬-깊이 예산 = L**. scale은 L을 *느리게*(~log) 키우고 width/암기회로를 *크게* 키운다.

### 사실 2: 표기깊이 d(e) = 직렬 unfold 깊이 (= Bennett logical depth)
"best/most/중첩"을 펼치는 건 본질상 *직렬*(정렬·계수·재귀). d(e) = 그 직렬 깊이.

### ★깊이를 지불하는 3가지 길 (= 시간/공간/offload 삼분)
| 지불 | 메커니즘 | 한계 | 우리 실측 |
|---|---|---|---|
| **공간(scale)** | 더 큰 L + 암기 병렬회로로 unfold를 *병렬화/암기* | **d ≤ L만**. 상수-깊이 회로는 *직렬-깊이 d≫L*를 **크기 무관 못 함**(AC⁰의 PARITY 불가류 하한) | **floor sweep: 32B/72B도 binding 벽 못 깸** = binding이 logically-deep → 어떤 forward pass도 불가 |
| **시간-in-LLM(CoT)** | 토큰마다 직렬스텝 추가 → 깊이 무계(Li et al: CoT면 직렬/P-complete 가능) | **스텝당 오차 누적** → 소형은 plateau | **Sstep: CoT 7B→0.656≈14B 후 평탄** |
| **시간-in-엔진(결정론)** | 실제 직렬기계가 unfold 실행 | **무계·오차0·저비용** | **offload 승리(thesis)** |

### ★시간↔공간 호환성의 정확한 의미
- forward-pass 깊이 L(병렬-시간/공간)과 CoT 길이 T(직렬-시간)는 *같은 계산의 두 축* — **CoT = 고정-깊이 회로를 시간축으로 *펼침***(Merrill-Sabharwal CoT 확장·Feng et al). 둘은 아키텍처적으로 교환가능, **단 비대칭**: 공간(L)은 *상수 깊이*(유계)·시간(CoT/엔진)만 *무계 깊이*.
- **빅모델 = 직렬 unfold를 *병렬회로로 암기/근사*해 공간으로 지불** — 빠르나(1패스) *유계 깊이*·파라미터 낭비·주권 위배. **logically-deep(병렬화 불가) 연산은 크기 무관 forward pass로 못 함.**

### ★thesis의 아키텍처적 정당화 (왜 작은모델+scaffold ≥ 빅모델)
1. 깊은 d는 **어떤 forward pass(어떤 크기)도** 못 함(유계-깊이 하한) → scale 무효(=floor sweep).
2. CoT는 무계지만 *LLM이 매스텝 정확*해야 → 소형 오차누적 plateau.
3. **결정론 엔진 = 완벽·저비용 직렬기계** → unfold를 무계·정확 실행.
⇒ **LLM은 *얕은 분류*(d ≤ d*·절차-타입 인식·소형 L 예산 안)만·깊은 unfold는 엔진에 offload.** 빅모델은 (a)깊은 d서 어차피 유계한계에 막히고 (b)얕은-d 회로 암기에 파라미터 낭비 → **소형+엔진이 지배.** = `DECOMPOSITION_OPTIMALITY`·"binding 벽≠scale"의 *근본 이유*.

### 예측 (반증가능)
- (a) forward-pass(no-CoT) regime서 **d* ∝ L(layer 수)이지 total-params 아님** — 깊이천장은 깊이예산(L)이 정함.
- (b) CoT가 유효 d*를 늘리나 *오차율×깊이*로 소형 plateau(Sstep 정합).
- (c) **결정론 offload가 유효 d* = ∞**(엔진이 직렬깊이 흡수) → 소형+엔진이 d-불변.
- (d) 분류(절차-타입 인식)는 d-얕음 → 소형 L 예산 안 → 학습·전이(§7b 학습가능성과 정합).
→ 딥리서치(`w3d906s6n` notationality·`wuwr9839y` routing) + **Transformer-복잡도 문헌**(Merrill-Sabharwal TC⁰·CoT 확장·Feng/Li serial·Bennett depth)이 이 절을 정련·검증.

## 7e. ★토대 이론 — 왜 LLM이 자연어를 잘 푸는가, 그리고 거기서 도출되는 역할분담 (2026-06-17)
> 사용자 지시: LLM이 왜 NL을 잘 푸는지 *철학적·과학적* 이론을 세우고, 거기서 LLM-잘하는것 / 알고리즘-잘하는것 / 애매한것(둘 다→병합)의 **새 역할분담**을 도출. = thesis의 *왜*.

### 명제 (한 문단)
**자연어는 *유계-깊이 병렬 인지*(인간 뇌·~100-step 직렬한계, Feldman)를 위해 진화한 *압축된 인터페이스*다. 그래서 NL 이해의 대부분은 *얕은 병렬 연상*(분포의미의 보간)이고, 정확히 이것이 Transformer forward pass(TC⁰·유계-깊이 병렬연상기)가 잘하는 일이다 — LLM이 NL을 잘 푸는 건 *구조적 합치*(둘 다 얕은-병렬). 그러나 NL은 곳곳에 *깊은 알고리즘 주머니*(superlative·양화·산술·다단·정확참조 = §7c d↑)를 *내장*하는데, 거기서는 LLM도 *유계-깊이 한계*에 막힌다(§7d) — 그리고 그 지점은 **인간도 머릿속이 아니라 *외부 직렬 도구*(아라비아-숫자 알고리즘·종이·계산기·정렬절차)로 offload하는 바로 그 지점**(확장된 마음, Clark-Chalmers). ⇒ 올바른 LLM-에이전트 = *인간 인지를 그대로 모사*: **LLM = 뇌의 얕은-언어-연상 핵 / 알고리즘 = 외부의 정확-깊은-절차 도구.****

### 왜 *얕은 병렬*인가 — 4겹 근거
- **과학(계산)**: forward pass = TC⁰ 유계-깊이 병렬(Merrill-Sabharwal). 잘하는 건 *얕은-병렬*뿐(§7d).
- **과학(인지/진화)**: 뇌는 대규모 병렬·직렬깊이 유계(100-step rule). **언어는 그런 뇌가 *실시간 처리 가능*하도록 진화·압축**(고빈도=청크=얕음·Zipf 효율부호). → 언어는 *설계상* 얕은-병렬 처리가능.
- **과학(통계)**: 의미는 분포기하(Harris/Firth "company it keeps") = 저차원 매니폴드(우리 Olver 추상화) → 경사하강이 학습·보간.
- **철학**: Wittgenstein *의미=사용* (연상이 포착)·Frege *뜻(grasp=연상) vs 지시(compute=절차)*·Brandom *질료적(즉각) vs 형식적(계산) 추론*. LLM은 *사용/뜻/질료추론*(얕은연상)에 강하고 *지시계산/형식추론*(깊은절차)서 막힘.

### ★역할분담 (명제의 귀결)
| | LLM (뇌-언어 핵) | 알고리즘 (외부 도구) |
|---|---|---|
| 잘함 | 얕은 연상·맥락·화용·애매성 해소·denotation·**절차-타입 *분류*(어디에 깊은 주머니가 있나)** | 깊은 직렬 unfold·정확·대규모 상태·날조차단·exact 참조 |
| 왜 | 얕은-병렬 = forward pass 적합 | 무계-직렬·오차0 = 실제 기계 |
| 한계 | d > d* 깊은 unfold 불가(§7d) | 애매성·맥락·화용 모름 |

- **분류(절차-주머니 식별)가 LLM 일인 이유**: "여기 superlative/산술/다단이 있다"를 *인식*하는 건 *표면 연상*(얕음·d-낮음) — 실행(깊음)과 분리. ⇒ **소형도 분류는 학습·전이**(§7b·§7d-d).

### ★애매한 경계 — 둘 다 하고 병합 (사용자 지시)
명확히 얕음→LLM만·명확히 깊음→알고리즘만. **경계(denotation인지 절차인지·읽기 애매)** = **둘을 *동시에* 돌려 병합**:
- LLM = *맥락 prior*(가능한 해석들·화용적 우선)·알고리즘 = *exact 제약*(DB 실재·feasibility·grounding).
- **병합 = LLM 해석분포 ∩ 알고리즘 feasibility** (LLM이 제안·알고리즘이 거르고 ground). 예 "Google Home"= LLM이 "Google Assistant 의도"로 읽고 → 알고리즘이 카탈로그서 실재 매핑 검증. 생성기(LLM)–검증기(알고리즘) 양방향. = 애매성은 *맥락(LLM)×형식(알고리즘)* 곱으로 해소([[feedback-selector-verifier-deterministic]] 결정론 검증기와 정합).

### 인간-모사가 *왜* 최적인가 (주권·비용으로 닫힘)
인간이 큰수 곱셈을 머리로 안 하고 아라비아-알고리즘으로 종이에 하듯 — **깊은 d를 LLM-파라미터(공간·유계·비쌈)나 LLM-CoT(오차누적)로 *내재화*하지 말고 외부 결정론 도구로 offload.** 그래서 *작은* LLM(얕은 언어핵)+엔진(깊은 도구)이 *큰* LLM(깊이를 파라미터로 욱여넣음)을 비용·정확·주권서 지배(§7d·`DECOMPOSITION_OPTIMALITY`). **= 새 역할분담: LLM을 *언어가 진화한 그 일*(얕은연상+분류)에만 쓰고, 못하는 깊은 절차는 알고리즘으로 대체.**

### 반증가능·측정
(a) LLM 실패가 §7c d(e)에서 갈림(얕음 성공·깊음 실패). (b) 분류(절차-주머니 식별) 정확도 ≫ 실행 정확도·소형서도 높음. (c) 병합(LLM∩알고리즘)이 단독 둘보다 애매-경계서 우월. (d) 깊이천장 ∝ L(§7d). → 두 딥리서치 + Transformer-복잡도 + 인지/언어철학 문헌이 정련.

## 7f. ★내재화 → 전이(C8)의 재정식화 (2026-06-17·이론을 우리 전이문제로 닫음)
> 사용자: LLM이 *flat 표기 암기*가 아니라 *추상화를 내재화*한다고 보면, 우리 전이도 "*내재화 가능한 것*을 학습시킬 수 있나"의 문제가 된다.

### 재정식화
**전이(C8) = "내재화 *가능한 추상화*를 *고립시켜 학습타깃화*할 수 있나"** — '전이가 되나?'가 아니라 '*무엇*을 학습시키면 그게 내재화·전이되나'.
- **내재화 가능(전이됨)** = 절차-타입 분류·생성원·automaton-type (얕음 d-low·§7d / 저차원불변·§7e). → *소형도 내재화·도메인 전이*.
- **내재화 불가(암기·비전이)** = flat 표면(어휘·도구명)·concrete 값(order_id·item_id). → 표면결합·도메인종속.

### 설계 = 학습타깃을 *내재화가능 추상화로 고립*
- **등방화** → flat 표면 제거(표면결합 차단·§M-σ).
- **resolver-offload** → concrete 값 제거(답이 아니라 절차-타입을 학습·[[feedback-nl-formalize-llm-selection-deterministic]]).
- ⇒ 남는 학습신호 = **절차-분류(생성원 인식)** = 내재화가능 = 전이 타깃. (정적 criteria도 폐기 — §7b-d: *절차-타입*을 emit·concrete/실행은 엔진.)

### 과거 전이실패의 재해석 (이론이 *예측*했어야 할 것)
v4-v7·M-σ 음성 = **flat/concrete를 학습타깃화**(표면결합·암기) → 무전이는 *당연*. 이론 함의: 그건 "내재화 불가한 것"을 학습시킨 것. **처방 = 내재화가능한 절차-분류(진짜 추상)만 고립 학습.**

### ★핵심 미해결 = 깨끗한 C8 시험 (반증가능)
**LLM의 "추상화 내재화"가 *진짜*(전이)냐 *정교한 표면보간*(비전이)이냐** = 이론의 사활.
- 시험: **고립된 절차-타입 분류**(denotation/superlative/comparative/conditional…)를 *등방화 추상 데이터*로 학습 → *held-out 도메인* 전이 측정.
- **전이 → LLM이 진짜 추상(절차-타입)을 내재화·C8 양성** = thesis 입증.
- **무전이 → 분류*조차* 표면결합** = "내재화"는 보간이었음 → 무엇이 내재화가능인지 재정의.
- = §7e-(b)(분류정확도≫실행·소형서도 높음) × 도메인전이를 *함께* 측정. 우리 등방화-synth(M-σ v3/v4) 골격 재사용·**타깃만 concrete→절차-타입으로 교체.**

## 7d-bis. ★엄밀 교정 — TC⁰는 점근적, 고정모델은 *유계 절차예산* B(L,width) (2026-06-17 측정 후)
> 내 자기비판("실패 ops는 TC⁰니 깊이-한계 아님")이 *부정확*했다. 측정(§아래)이 양쪽 과장을 규율.

### 교정된 진술
- **TC⁰ ∈ 은 *점근적***: "어떤 poly-size 상수깊이 회로가 *존재*"일 뿐, *고정 크기* 모델이 그 함수를 *구현한다*는 게 아니다. 내가 이 둘을 혼동했다.
- **고정 Transformer = 유계 절차예산 B(L·width)**: 한 forward pass가 *내부적으로 수행 가능한 절차 깊이*가 유한. **d(e) ≤ B 성공·d(e) > B 하락.**
- **실측 정합**: dump base 7B = d=1 **0.83** → d=2 0.58 → d=3 **0.20** (단조하락) = **B≈2 신호**(7B가 1-2단 절차는 하고 3단은 못 함). base도 깊이서 실패 → *순수 아티팩트 아님*(내 §7d 비판 반증). ops가 TC⁰여도 *고정 7B가 d=3을 구현 못 함* → *깊이-한계도 참*(단 회로류가 아니라 *예산*).
- **★사용자 가설 = B가 params·layer로 자람** → **반증가능 시험: 깊이-정확도 *무릎 d\**이 모델크기로 *우측 이동*하나?** (14B/32B/72B in-head dump로 측정 — 깊은 케이스가 크기로 fail→success면 B 성장 확정.)

### IR-아티팩트와의 재화해 (같은 축의 두 얼굴)
- **정적 IR 강제 = 모델이 *in-head 반복절차에 B를 쓰는 걸 박탈*** → 유효 B↓ → 깊이서 더 나쁨. 실측: structured < base이고 *base-맞고-structured-틀린 6케이스 전부 d≥2*(IR이 깊은 데서 B를 더 깎음). ⇒ **깊이-예산과 IR-아티팩트는 *별개가 아니라* 같은 축**(IR은 B 접근을 막는 방식).
- **세 지불 재정렬**: 공간(B를 키움·유계·느림·비쌈) / 시간-CoT(B를 외부 토큰으로 연장·무계·오차누적) / 시간-엔진(B=∞·정확). 정적-IR은 *B 박탈*(최악).

### ⇒ 정정된 결론
"binding/깊이 실패"는 **(a)하드 회로-한계**(틀림·TC⁰)도 **(b)순수 forced-JSON 아티팩트**(틀림·base도 깊이서 실패)도 아니라 **(c)유계 절차예산 B(L,width)를 d(e)가 초과**다. scale은 B를 *키우나*(무릎 우측이동 시험중)·결정론 offload는 B=∞로 *깊이를 흡수*(주권·비용 우월). 측정이 내 양쪽 과장(전부-아티팩트 / 전부-회로한계)을 다 깎았다 — **데이터정합 = 유계예산.**

## 9. ★선행연구 정초 (딥리서치 salvage 합성·2026-06-17) — 이론이 문헌에 *엄밀히* 선다 + 1 교정
> 두 딥리서치(routing `wuwr9839y`·notation `w3d906s6n`)는 synthesis 직전 killed됐으나 search/fetch/verify 완료 → 검증된 claim 207건 salvage(`_dr_salvage`). 핵심만 합성. **대부분 우리 이론을 *발명 아닌 검증된 조각의 미점유 교차점*으로 확인**하고, **한 곳(Roman/Arabic)을 교정**한다.

### A. "절차-내장 표기"는 *실재하는 형식 범주* (§7b 정초)
- **Relevance Theory 절차 vs 개념 의미**(Blakemore·Wilson): 언어요소는 *개념*(denotation·계산에 들어가는 표상)을 부호화하거나 *절차*(추론 계산 자체·따라야 할 추론-루트)를 부호화. "but/so/this"는 *개념 아니라 절차*. = 우리 denotation vs 절차-내장 분리 *그대로*. ★"절차의미는 sub-personal *machine-language*·의식화 저항" → **왜 comparative/anchor를 *명명* 못 했나**(7B 실측 B=0.00) 설명: 절차의미는 명명 어렵다.
- **동적의미론**(Groenendijk-Stokhof-Veltman·Heim): 의미 = *context-change potential*(정보상태 업데이트 *명령*)·정적 진리조건 아님. ★denotation·절차의미 *공존*(진리조건=업데이트의 *전제조건*) = 우리 "애매경계=둘 다·병합" 정합.
- **Goodman notationality**(Languages of Art): 기호계가 *모호성 없는 기계조작*을 지원할 조건 = disjointness + finite differentiation. notational(digital·결정) vs dense(analog·기계화 불가). 그림=실패·악보=통과. = 우리 "결정론 실행가능 표기" 형식 기준.

### B. 복잡도 위계가 *정확*하고 *인지적으로 실재* (§7c-7d 정초)
- **★van Benthem semantic automata (정확)**: "every/some"=permutation-invariant *acyclic 유한상태*·"짝수개"=*cyclic 유한상태*·**"most/less-than-half"=유한상태 불가·*pushdown(스택)* 필요**(Chomsky 한 단계 위·context-free). = "단어가 *특정 복잡도 알고리즘*을 내장"의 *증명*. 우리 superlative=pushdown류.
- **★인지 실재(fMRI/RT)**: pushdown("most")은 *작업기억(PFC) 동원*·유한상태("all")는 안 함·RT가 *최소-오토마타 복잡도로 스케일*. = 우리 "깊은 d=무너짐"의 *생물학적* 대응(d↑=작업기억↑).
- **학습성 ∝ 복잡도**: monotone/quantity 양화사가 LSTM서 빨리 학습·**NN이 "most" 검증서 *비율-민감(ANS) 절차를 자발 재현*** = *내장 절차(집합구성+크기비교)가 학습됨*(lookup 아님). LZ/Kolmogorov로 양화사 복잡도 *측정가능*. = 우리 §7c d(e) 측정 정당.
- **RASP-L**: Transformer length-일반화 ⟺ *짧은 길이-독립 RASP 프로그램* = 절차의 구조복잡도가 학습성 결정. = 우리 유계예산 B(§7d-bis) 정합.
- **iterated 양화사 = pushdown으로 *합성***(자동기계 합성) = 우리 "깊이=연산 중첩" 정합.

### C. ★교정 — 표기는 *계산가능성*이 아니라 *비용*을 정한다 (Zhang-Norman·내 Roman 비유 정정)
- **★Roman·Arabic은 *계산적으로 유사*** — Roman이 알고리즘 계산에 *부적합한 게 아니라 단지 *기본단계 수가 많을* 뿐. **표상은 *계산가능성*이 아니라 *비용 프로파일*을 정한다.** → 내 "Roman엔 알고리즘 없다"는 *틀림*. 옳은 진술 = **위치표기는 곱셈을 *싸게*(적은 단계) 만들지 *가능*하게가 아니다.**
- **★Arabic vs Roman = *반대 비용 트레이드오프***: Arabic=큰 *내부* 자원(100 덧셈사실·작업기억) / Roman=적은 내부·많은 *외부*(지각-운동) 단계(Roman 덧셈 ~30× 더 많은 READ). **= 우리 §7d-bis의 *공간(내부 B) vs 외부(CoT/엔진) 단계* 트레이드오프 그 자체.** 표기/IR의 가치 = *LLM-대면 단계수(비용)*를 줄이는 것(아라비아=내부암기로·우리 IR=엔진 offload로).
- ⇒ 교정이 이론을 *강화*한다: "깊이"는 *계산가능성 한계*가 아니라 *단계-비용*이고(B 초과 시 무너짐도 비용/오차), scale은 내부비용으로·offload는 외부 결정론으로 지불 — §7d-bis 유계예산 그림과 정확히 일치.

### D. 라우팅/분류는 *학습되고 전이된다* — 단 *자기평가는 불신* (§7e-7f 정초)
- **학습된 라우팅**(발명 아님·검증됨): **RTR**(model+추론전략 공동 라우팅·OOD 전이·71.7%↓토큰)·**A2FM**(instant/reasoning/agentic 모드 RL 라우팅·비용 45%↓)·**xRouter**(answer-vs-delegate RL)·**ARM2**(NL/code/vision 포맷 RL 선택·code offload)·**ToolkenGPT**(tool 호출=토큰예측). = "LLM이 분류·라우팅을 *학습*"이 *기성 패러다임*.
- **★전이**: **RITE**(math-only RL 학습 → *교차도메인 SOTA*) = 분해/도구-라우팅이 *학습·전이 가능 스킬*(우리 §7f C8 핵심 *지지*).
- **When2Call**: when-to-call이 *학습됨*(RPO≫SFT·8B F1 31.9→52.4·hallucination 1.2%)·기성모델 불신(Qwen72B F1 32.8·23% 환각). **tool-necessity가 hidden state서 *생성 전* 선형해독(AUROC 0.89-0.96)** = 라우팅 신호 *내부 존재*(우리 "얕은 분류=내재화가능" 정합).
- **★자기평가 불신**(중요·[[feedback-selector-verifier-deterministic]] 강화): verbalized confidence는 *과신*·OOD서 붕괴·Type-2 SDT로 "겉보기 calibration=criterion-placement지 진짜 메타인지 아님". **⇒ 라우팅을 *introspection 아니라 *학습/외부신호*로*** = 우리 "검증기=결정론" 정합.

### E. offload 경계 — 확립됐으나 *항상 이기진 않음* (정직)
- **PAL**: LLM 분해 / 결정론 interpreter 실행 = 우리 경계 *그대로*·직접 CoT를 큰 차로 이김.
- **LLM-as-formalizer 스케일**: NL→PDDL formalizer가 직접 planner 압도(100 blocks 100% vs 직접 20%)·"NL→formal은 LLM·지수탐색은 결정론 solver"가 *경험 최적 분담*. = thesis 핵심.
- **★반례(정직)**: LLM-as-formalizer가 직접 solver에 *15/24서 짐*·formalize 시 *해결추론 누출*(hard-code). ⇒ **offload가 *항상* 우월 아님** — search-heavy/스케일서 이기고, 단순서 formalize 오버헤드가 짐. 우리 7B 실측(comparative B=0.00)도 *명명 실패* 반례와 정합. **경계 자체가 측정 대상**(§7c d·B-budget 실험).

### 종합 (이론 지위 상향)
**우리 이론은 발명이 아니라 *검증된 조각들*(절차의미·semantic automata·notationality·representational-effect·학습된 라우팅·formalize-offload)의 *미점유 교차점***: **유계예산 B(L,width) × 표기-깊이 d(e) × 결정론 offload로 깊이흡수 × 분류는 학습·전이**. 1 교정(Roman=비용 아닌 가능성)이 오히려 §7d-bis(공간/시간/단계-비용)를 강화. 반례(formalize 항상은 아님·comparative 명명실패)가 *경계를 측정 대상*으로 못박음.

## 8. 정직 (이론 지위)
- 이건 *생산적 형식틀*(Lie 대수↔군 = name↔execute의 동형)이지 증명된 정리 아님. "가해성=추상화레벨"은 *은유적 위계*(엄밀 Galois 군 아님)·반증가능 예측(§6)으로 검증.
- 엄밀화 경로: (i)tool-use primitive를 실제 유한생성 대수로 구성(닫힘 companion 확장) (ii)생성원-수 = eff-dim 측정(olver_dimension_experiment 재사용) (iii)§6 예측 실험.
- 한 줄: **LLM=대수(생성원 명명·저차원·학습·전이)·결정론=군(exp 실행·무계). 추상화 레벨=가해성 위계. 설계=군을 대수로 들어올려 명명은 LLM·실행은 엔진. 십진 위치표기가 무한을 유한 생성원으로 바꾼 바로 그 수.**
