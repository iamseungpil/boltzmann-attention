# 선행연구 정본 — "판단을 LLM에서 결정론 엔진으로 옮겨도 되는가"의 3조건 기준 (2026-08-01)

> **상태 = 딥리서치 5클러스터 완료 · 인용 가능.**
> **시험 대상 명제**(사용자 2026-08-01): *"경계는 고정이 아니라 열거 가능성에 따라 움직인다. 어떤 판단이든
> 그 외연이 **(i) 유한**하고 **(ii) 정책에 유계**이고 **(iii) 전수로 열거돼 있으면** 엔진 쪽으로 옮길 수 있다.
> 셋 중 하나라도 안 되면 LLM+표면화가 정답이다. 그리고 이 세 조건은 **가정하는 게 아니라 재는 것**이다."*
> **방법** = 병렬 딥리서치 5클러스터(①KR 폐쇄세계·완전성 추론 ②집행가능성 형식이론 ③DST categorical 슬롯
> ④신경-기호 배분 기준 ⑤적대적 신규성 검증), 전 인용 1차 출처 대조.
> **연결** = [[22]](닫힌/열린 술어) · [[10]](분담) · [[49]]/C277(매핑-경계) · [[46]](노벨티 지도) ·
> C276(merchant census) · C279(식별표 멤버십) · §1.3(상쇄법칙)

---

## §0. 한 문단 (판정)

**세 조건은 전부 이미 형식화돼 있다.** (i)=Reiter의 **domain closure axiom**, (iii)=**CWA**, 둘의 결합은
**Reiter 1984의 DCA+UNA+CWA 삼중 재구성과 동형**이다. (ii)는 KR의 **`LCWA=⟨S,P̄,Ψ⟩`(window of expertise)**와
DB의 **`Compl(R(s̄);G)`(table completeness statement)**가 독립적으로 같은 형태에 도달했다. 실무 쪽도
DST의 **categorical/non-categorical 슬롯**이 2020년에 스키마 필드로 표준화했고, **MultiWOZ 2.2는 임계값
(학습셋 distinct value <50)으로 그 판정을 *계산*한다** — "재는 것"조차 선례가 있다. **⇒ "새 기준"으로
헤드라인을 걸면 즉사한다.** 그러나 문헌 20편 이상이 **열거 완전성을 공리로 놓고 끝낸다**. Cortés-Calabuig은
"Ψ의 검증 가능성"을 **미해결로 남기고**, shielding은 abstraction 위반을 게임에서 **"승리"로 계산**해 보장이
침묵 소실되며, Carr는 *"provided that the partial model is adequate"*라 쓴다. 측정한 예외는 넷뿐이다
(Etzioni Counting Rule · cardinality assertion · AMIE+ §7.4 · **CMU 2604.15579**). 그리고 마지막 것은
**τ²-Bench를 쓰는 직접 경쟁자**다. 남는 델타는 ⓐ 세 조건을 **할당 게이트**로 세운 진술 ⓑ census 대상이
**자연어 정책 코퍼스** ⓒ **역방향 실패(예시를 전수로 오인)의 명명·계측**이다. 단 **(i)은 그대로 쓰면 안 된다** —
유한은 결정가능성의 충분조건도 필요조건도 아니다(§5).

---

## §1. 조건별 선행 지형 (전체 대조표)

| 조건 | 정본 선행 | 등급 | ASSUME / MEASURE |
|---|---|---|---|
| **(i) 유한** | **DCA** `∀x(x=a₁∨…∨aₘ)` (Reiter, JACM 27(2), 1980) — 문헌 전체에서 (i)만 단독으로 다루는 유일 공리 | [S] | **ASSUME**(공리로 선언) |
| | **HRU 1976** — *"the finiteness of real resources does make safety decidable"* | [S] | — |
| | **Cardinality assertion** `\|σ(Kᵢ)\|=n` (Razniewski et al., CSUR 2024 Def 5) | [S] | **★MEASURE** |
| **(ii) 정책-유계** | **`LCWA=⟨S,P̄,Ψ(ȳ)⟩`** — Ψ = *"window of expertise"* (Cortés-Calabuig et al., LPNMR 2005) | [S] | ASSUME |
| | **`Compl(R(s̄);G)`** table completeness statement (Razniewski & Nutt, VLDB 2011) | [S] | 선언은 ASSUME·**파생은 결정 절차로 계산** |
| | **Semantic DMN**(RuleML+RR 2017) — 온톨로지가 허용하는 값으로 completeness check를 제한 | [M] | ASSUME |
| | **LLM-Modulo**(ICML 2024) — hard critic 자격 = *"works off of a model"* | [S] | ASSUME |
| **(iii) 전수 열거** | **CWA**(Reiter 1978) · Clark completion · circumscription · only-knowing | [S] | **ASSUME**(Reiter 축자: *"presumes total knowledge"*) |
| | **Reasonable property** = `∀σ: P̂(σ) is decidable` (Ligatti et al., TISSEC 2009 Def 2.1) | [S] | 증명 대상(측정 아님) |
| | **`pre*(P∩U)` decidable** (Basin et al., TISSEC 2013 Thm 7) + **강제가능성 판정 자체가 PSPACE-complete** | [S] | **결정 문제로는 MEASURE·커버리지는 아님** |
| | **AlphaGeometry**(Nature 2024) — *"exhaustively derives the deduction closure"* vs 보조작도의 무한 분기 | [M] | 도메인 서술(일반 원리로 미승격) |
| **"재라"** | **AMIE+ §7.4** — PCA 성립률을 YAGO2에서 실측·실패 유형 분류 | [S] | **★MEASURE** |
| | **Etzioni Counting Rule**(AI 1997 Thm 11) — 카디널리티 대조로 LCW 획득 | [S] | **★MEASURE**(단 W의 카디널리티는 다시 선언) |
| | **MultiWOZ 2.2** — 학습셋 distinct value <50이면 categorical | [S] | **★MEASURE (유한성만)** |
| | **CMU 2604.15579** — 요구사항 전수 분류: **74% symbolic 강제 가능·그중 95% 단순** | [S] | **★★MEASURE (유일한 완전 대응)** |

### ★Reiter 1984 본문 확보 (2026-08-01·VERIFIED-PRIMARY) — **선점하지 않는다, 단 마진이 얇다**

경로: Springer/ACM/CiteSeerX/ResearchGate/HathiTrust 전부 차단 → **archive.org 스캔본
`onconceptualmode0000unse`의 full-text 엔드포인트**로 OCR 문단 전문 추출(leaf↔인쇄면 오프셋 16 검증).

**① 세 가정의 축자 형식 (p.191 초록 — 책 전체에서 DCA·UNA는 이 면에만 등장, volume 색인으로 확인)**
- DCA: *"The individuals occurring in the database are all and only the existing individuals."*
- UNA: *"Individuals with distinct names are distinct."*
- CWA: *"The only possible instances of a relation are those implied by the database."*
⇒ **"세 조건이 함께 닫힌 관계구조를 규정한다"는 통찰은 1984년에 이미 있다. 노벨티로 주장하면 즉사.**

**② iff 정의도 실재한다 (§3.1 p.209)**: *"A first order theory T ⊆ Ψ is a relational theory of R iff it
satisfies the following properties"* + **Theorem 3.1(p.210)** 유일 모델 동치(양방향).
**★그러나 이 iff의 조건은 전부 *이론 T의 구문*에 대한 것이다** — "T가 DCA·UNA·completion 공리를
담고 있는가". **세계·도메인·판단이 실제로 닫혀 있는지에 대해서는 한 마디도 하지 않는다.**
Theorem 3.1은 적용가능성 판정이 아니라 **proof-theoretic ↔ model-theoretic 번역의 충실성 정리**다.
§4.2.4(p.227)가 규범적 어조에 가장 근접하나 어휘가 *"is likely to impose"* / *"should"* = **설계
desiderata**이지 테스트가 아니다.

**③ 검증·측정 논의는 없다 (결정적 negative finding)**. 가장 근접한 것이 §5(p.230, p.232)의
*"open for inspection"* = **인간의 육안 검토**. 오히려 p.203에서 DCA/CWA를 세계에 관한 사실이 아니라
**데이터베이스의 인식적 입장**으로 재규정해 **검증 대상 자체를 소거**한다.

**④ 실패 모드는 있으나 *이론 쪽*이다.** p.214 선언적 사실 추가 → 비일관 · p.218 null이
*"contradicting our presumed ignorance"* · **p.227–228에서 자신의 1978 CWA를 스스로 기각**
(*"unsuitable"* — null 오취급 + 선언 정보에서 비일관). 그럼에도 처방은 *"some representation of the
closed world assumption is necessary"* = **공리를 고쳐라**이지 *"이 도메인엔 적용하지 마라"*가 아니다.
**"open world"라는 표현은 pp.191–238에 단 한 번도 나오지 않는다.**

**⑤ 쿼리 축은 이 챕터에 없다.** §2.4(p.202)의 유일한 연결이 방향이 반대 — *"There is no need for the
concept of a safe query."* (DCA를 쿼리 안전성 조건을 **제거**하는 근거로 사용). existential vs
universal/negative 구분은 **JACM 1980 소관**.

**★방어선 (셋 중 하나라도 흐려지면 Reiter가 그대로 선행기술이 된다)**
1. 조건이 **이론이 아니라 관측 가능한 도메인/판단**에 대해 진술된다.
2. 성립 여부를 **stipulate가 아니라 measure**한다.
3. 실패 시 처방이 **"공리 보강"이 아니라 "닫힌 취급 자체의 기각"**(=표면화)이다.

### 세 조건이 논리적으로 독립임을 문헌이 증언한다
**Clark completion은 CET(=UNA)를 포함하되 DCA는 배제한다.** 즉 (iii)을 주면서 (i)은 주지 않는 형식이
실재한다. 세 조건을 **연언**으로 요구하는 것이 자의적 묶음이 아니라는 근거로 쓸 수 있다.

### CWA의 붕괴 조건 — 우리 게이트에 직접 적용
Reiter TR-77-16 **Example 5.1**: `DB: Pa ∨ Pb`에서 `DB⊬Pa`·`DB⊬Pb`이므로 CWA가 `¬Pa,¬Pb`를 둘 다 추가 →
**비일관**. Theorem 5.1: Horn이면 일관. ⇒ **정책 산문이 선언(disjunction)을 담으면 폐쇄 가정이 무너진다.**
Theorem 3.1: CWA 하에서 minimal answer는 전부 definite = **지식 간극이 정의상 존재할 수 없음**(= 간극을
탐지할 수단이 원천 봉쇄).

---

## §2. ★유일한 완전 경쟁자 — CMU 2604.15579 (최우선 정독)

> Yining Hong, Yining She, Eunsuk Kang, Christopher S. Timperley, Christian Kästner (CMU).
> *Don't Make Models Guess Security and Safety: Symbolic Guardrails for Domain-Specific AI Agents.*
> **arXiv:2604.15579** (v1 2026-04-16, v2 2026-07-05 제목 변경).

- 축자: *"tools are limited, so verifiable requirements can be enumerated in advance"* ⇒ **(i)+(iii)**
- 축자: *"plausibly enforced symbolically, that is, through deductive reasoning in program code"* ⇒ **(ii)**
- 축자(자기 한계): *"the enforceable rate should be read as a conservative lower bound"*
- **분류 축**: neural=귀납(LLM-as-judge) / symbolic=연역. **LLM이 명세를 생성·실행하면 결정론 아님**으로
  판정 — GuardAgent·NeMo Guardrails·ShieldAgent를 명시적으로 neural에 배치([[10]]과 같은 규율).
- **측정**(⚠2026-08-01 초록 직독으로 정정 — 벤치 이름 오기 교정 + **3부 구성**임이 드러남):
  **(1)** 에이전트 보안·안전 벤치 **80개 systematic review** → **85%가 verifiable requirement 미명시**
  (**61% 전무 · 24% 고수준 목표만**) **(2)** **τ²-Bench · CAR-bench**(⚠CRMArena-Pro 아님) **· MedAgentBench**
  applicability 분석 → **74% symbolic 강제 가능 · 그중 95%는 단순·저비용** **(3)** ★**같은 세 벤치에서
  symbolic guardrail의 실증 평가** — *"improve security and safety without sacrificing utility, and often
  improve it"*. **코드·아티팩트 전량 공개**(github.com/hyn0027/agent-symbolic-guardrails).
  ⇒ **분류 연구가 아니라 시스템 평가까지 포함** — 겹침이 C281 초판 기록보다 크다.
- symbolic 수단 6종 카탈로그: API validation · schema constraint · temporal logic · information flow ·
  user confirmation · response template.

**우리와의 델타(현재 판단)**: 이들 기준은 **수단 기준**(6종 카탈로그 중 하나에 걸리는가)이고 우리는
**대상 기준**(술어 외연의 유한성·전수성)이라 카탈로그 확장에 독립적이다. 역으로 이 논문은 세 조건의
**경험적 지지 증거**(74%/95%)로 인용 가능하다. ⚠**같은 벤치(τ²)를 쓰는 인접 그룹이므로 scoop 위험 최상 —
원문 정독을 최우선 대기열에 올린다.**

---

## §3. "측정"이 실제 델타임 — 문헌의 자백 4건

1. **Cortés-Calabuig et al. §6**: `Ψ`가 window를 모호하지 않게 정의해야만 완전성을 얻는데(`Ψ=¬Q(x)`는
   검증 불가), **어떤 LCWA 표현이 실용적인지 판별하는 조건을 미해결로 남긴다.**
2. **Alshiekh et al.(AAAI 2018)**: safety game의 안전 상태 집합이 `Fg=(F×QM)∪(Q×(QM\FM))` — **abstraction이
   틀렸음이 드러나는 상태를 system player의 "승리"로 계산**한다. 축자: *"only needs to work correctly in
   environments that conform to the abstraction"* ⇒ **abstraction 밖의 일이 벌어지면 shield는 경보 없이
   보장을 잃는다.** ba8b형 침묵 오차단의 RL 판본.
3. **Carr et al.(AAAI 2023)**: 전이 그래프 **지지집합의 정확한 열거**가 Thm 4·5의 전제인데 §4.1은
   *"provided that the partial model is adequate"*로 **가정**한다.
4. **David et al. 2026**: `finite Σ`+`finite Res`+decidable predicates로 (i)(ii)는 일치시키나 (iii)은
   **Effect Observability Assumption으로 격리만** 하고 검증 절차를 제시하지 않는다.

### 방법론적 선례 = AMIE+ §7.4 (우리 절차와 동형)
가정(PCA)을 세운 뒤 → **실제 성립률 실측** → **실패 유형 분류** → **range 제한 처방**. 축자로 자기 전작을
정정한다: *"it did not evaluate whether this assumption is true"*. [[03b]] 자기감사 규율과 동형.

**★그들이 찾은 실패 유형이 우리 문제와 같다** — PCA가 깨지는 대표 사례가 `locatedIn`/`livesIn`의
**입도 불일치**(도시/지방/국가/대륙). 이는 C276의 **`Target` vs `Target - Eco Collection`**, **범주어 핀
(`market`) vs 브랜드 핀**과 **같은 실패 클래스**다. 처방도 동일 — **관계의 range를 제한하면 준함수가 되어
가정이 다시 성립**. C276/C279의 `pin_kind` 선언·식별표 멤버십 전환이 이 처방의 우리 판본이다.

---

## §4. 실무 선점 — DST와 tool schema는 이미 이 기준으로 돌아간다

| 항목 | 사실 | 등급 |
|---|---|---|
| **SGD** `is_categorical` 스키마 필드 | *"If true, the slot has a fixed set of possible values."* — 값 열거 불가면 non-categorical | [S] |
| SGD 실측(릴리즈 schema 파싱) | train 215 slot 중 **categorical 53(24.7%)**, 값 집합 크기 median **3** | [S]신규측정 |
| **BFCL v4 실측**(공개 문헌 최초로 보임) | live(실사용자) 파라미터의 **enum 보유 21.5~23.1%**, 크기 median 3~4. **합성 벤치는 3%**로 실세계를 크게 과소표현 | [S]신규측정 |
| **MultiWOZ 2.2** | 학습셋 distinct value **<50이면 categorical**로 *계산*. ontology 값의 **21.0%가 DB에 매핑 불가** | [S] |
| **DS-DST**(*SEM 2020) | "Find or Classify?" — 열거 가능하면 picklist, 아니면 span | [S] |

**★수렴**: SGD 24.7% ≈ BFCL 21~23% ≈ MW2.2 34.4%, 값 집합 크기 median 3~6. **20년·두 분야·같은 답 —
닫힌 술어는 전체의 약 1/4이고, 닫혔을 때는 매우 작다.**

**★사후 소진성 검증(신규)**: BFCL gold answer를 선언 enum과 대조 — simple_python 48개 전부 내부,
live_simple은 `"N/A"` 센티널 20개 제외하고 **진짜 위반 0건**. **(iii)이 측정 가능하고, 측정하면 통과할 수
있다는 존재 증명**이며 이 측정을 한 논문은 없다.

### ⚠기준에 불리한 증거 4건 (반드시 다룰 것)
1. **DS-Picklist(전부 닫음, 54.39/53.30) > DS-DST hybrid(52.24/51.21)** — 열거가 가능하면 **하이브리드보다
   전부 닫는 게 낫다**. hybrid는 최적이 아니라 열거 불가능성에 대한 *타협*.
2. **TripPy(열거 없이 copy, 55.29) > DS-Picklist(53.30)** — 열거 없이도 이긴다.
3. **ADB(AAAI 2021)**: known-ratio 25→50→75%로 inventory가 촘촘해질수록 open-class F1이
   **84.56→78.44→66.47 하락**. ⇒ **열거를 촘촘히 만들수록 그것이 새는 것을 탐지하기 어려워진다.**
   A2를 키울수록 밖을 못 본다는 뜻 — 우리 설계에 직접 경고.
4. **닫아도 선택은 보장되지 않는다**: Structured Output Benchmark(2604.25359)에서 schema-constrained
   decoding의 **value accuracy 효과 −0.007~+0.033 ≈ 0**, 2607.07026은 잔여 실패에 **"wrong enum
   selection"**을 명시. 형식 유효성은 100%가 되지만 **집합 안에서 옳은 원소를 고르는 정확도는 안 오른다**
   (§1.3 상쇄법칙).

---

## §5. ★조건 (i)은 그대로 쓰면 안 된다 — 필요한 수정 3건

### 5-1. 유한은 충분조건도 필요조건도 아니다
- **불충분**: Ray 2026 Thm 6 — *"Nontriviality is undecidable with two global ℕ-counters with increment,
  guarded decrement, and zero-test"* ⇒ **유한 알파벳이어도 감소 가능 카운터 둘이면 결정 불가.**
  필요한 것은 **유계(bounded) ∧ 단조(monotone) ∧ 분리가능(separable)**. [D]단독저자 preprint
- **불필요**: **Sandhu SPM**(JACM 1988) — 무한 생성이어도 **acyclic이면 tractable**. 축자:
  *"analysis is tractable … provided certain restrictions are imposed on subject creation"* [S]
- ⇒ **(i)을 "extension FINITE"가 아니라 "extension finitely foldable"로 재정식화**하는 것이 정확하다.

### 5-2. 결정가능 ≠ 실행가능 (비용 축 누락)
**HRU 1976 §5**: 유한하면 safety는 결정 가능하지만 **NP-complete**(mono-operational) /
**PSPACE-complete**(create 없음)이고, 같은 문단에서 *"this bound will not make the decision … 'easy'"*라고
못 박는다. **세 조건에 "유한"만 있고 "열거·판정 비용"이 없으면 HRU가 그대로 반례**가 된다.

### 5-3. 누락된 축 3개
| 축 | 출처 | 내용 |
|---|---|---|
| **질의 형태** | Li & Tripunitara (TISSEC 2006) | 같은 정책·같은 상태도 **semistatic query면 다항, 일반 query면 coNP-complete**. ⇒ 결정가능성은 술어만의 성질이 아니라 **⟨술어, 질의 형태⟩ 쌍**의 성질 |
| **개입 능력** | Ligatti et al. (TISSEC 2009) | truncation(차단만)=**safety**, edit(억제+삽입)=**renewal**, safety ⊊ renewal. ⇒ **게이트가 무엇을 할 수 있느냐가 강제 클래스를 바꾼다.** 세 조건은 술어만 말하고 게이트 형태를 말하지 않음 |
| **관측 vs 통제** | Basin et al. (TISSEC 2013) Def 3 | *"An enforcement mechanism cannot terminate the system when observing an only-observable action."* 시계 tick·역할 부여는 관측되나 통제 불가 ⇒ safety여도 강제 불가 |

### 5-4. Schneider 축과의 관계 정리 (혼동 금지)
Schneider(TISSEC 2000)의 축은 **시간적 형태**(safety vs liveness)이지 **외연 구조**가 아니다. 그는 (i)을
**요구하지 않는다**(상태 집합 countably infinite 허용). 세 조건이 (i)을 추가해 얻는 것은 *강제가능성*이
아니라 **결정 복잡도**다. 반대로 **Alpern-Schneider 분해 정리는 조상이 아니라 좌표계**로 쓰는 게 정확하다 —
임의 property = safety ∩ liveness이므로, *"이 판단은 못 옮긴다"*가 아니라 **"safety 성분만 옮기고 liveness
성분은 LLM에 남긴다"**가 가능하다. 이것이 [[22]] 표면화의 형식적 뒷받침이다.

**(iii)의 최근접 형식 조상 = Basin et al. Thm 7**: `P`가 `(U,O)`-enforceable ⟺ (1) `(U,O)`-safety
(2) **`pre*(P∩U)`가 decidable set** (3) `ε∈P`. 그리고 **그 판정 자체가 정규 언어에서 PSPACE-complete,
문맥자유에서 결정 불가**. ⚠단 Basin이 결정하는 것은 ***명세***의 강제가능성이지 ***커버리지***가 아니다 —
**이 간극이 whitespace의 정확한 위치**다.

---

## §6. 역방향 실패 — 이름은 없고 계측은 있다

### 6-1. 명명 공백 (3갈래 독립 조사가 모두 부재 확인)
KR/DB·법률·NLP 어디에도 **"예시 목록을 전수 목록으로 오인하는" 실패의 정본 명칭이 없다.** 가장 가까운
것들도 어긋난다 — Etzioni의 **"LCW miss"는 방향이 반대**(성립하는데 못 도출하는 과잉보수), McCarthy의
*"this mistake"*는 고유명 없음, qualification problem은 원인이지 실패 자체가 아님, AMIE+의 "granularity
differences"는 한 유형의 이름.

**문헌 밖 대응어**(⚠1차 미검증·인용 전 확인): 법해석 *expressio unius est exclusio alterius* /
*ejusdem generis*, 화용론 exhaustivity implicature, 비형식논리 false dilemma. **법률 실무는 이 실패를
명명했을 뿐 아니라 "including but not limited to"라는 방지 문구까지 표준화**했다.

### 6-2. 계측 증거 5갈래 (전부 우리 §1.3 상쇄법칙의 사례)
| 증거 | 수치 |
|---|---|
| **Reality Check**(2505.13252) | 형식화가 **24개 model-dataset 조합 중 15개에서 LLM-as-solver보다 열세** |
| **LINC**(EMNLP 2023 Outstanding) | **precision 93%↑(vs 81%) / recall 60%↓(vs 75%)** — 정밀도를 사고 재현율을 판다. 실패 L1(암묵 정보 누락)·L2(표현 선택으로 명시 정보 손실) = **(iii) 실패의 관측형** |
| **Constraint Tax**(2605.26128·preprint) | schema validity 61.5→**100%**, answer accuracy 19.7→**11.0%**, **wrong-but-valid-schema 49.5→88.9%** — 결정론 강제가 오류를 **은폐** |
| **AgentSpec**(ICSE 2026) | 규칙 강제 **recall 70.96%** — ~29%를 놓침 = 규칙으로 안 닫히는 열린 술어 |
| **Grammar-Aligned Decoding**(NeurIPS 2024) | 국소적으로만 유한 열거하면 **전역 확률분포가 왜곡** ⇒ (iii)은 국소가 아니라 **전역 외연**에 요구돼야 함 |

**원전급 자기한정**: SATLM(NeurIPS 2023) — *"guarantee the correctness of the answer **with respect to the
parsed specification**"*. 건전성이 파스에 상대적임을 스스로 못 박았다.

---

## §7. 예상 반론 2건 (대응 준비 필요)

1. **fail-direction 논변** — **RTBAS**(CMU, 2502.08966)는 결정론 IFC 안에 **확률적 dependency screener**
   (정확도 81%)를 박아 넣고 축자: *"incorrect decisions by our dependency screener approaches cannot
   compromise security"* — 오류가 over-tainting(확인 증가) 또는 under-tainting(작업 실패)으로만 귀결되게
   배치. **Progent의 SMT monotonic confinement도 같은 논리.** ⇒ **"외연이 열려 있어도 오류 방향을 봉쇄하면
   결정론 시스템에 넣을 수 있다"** — 세 조건이 다루지 않는 케이스. **대응 논거 필수.**
2. **일반화 논변** — **LlamaFirewall**(Meta) 축자: *"the absence of deterministic solutions to mitigate
   these risks"*, 결정론은 배포마다 일반화가 어렵다. 세 조건의 domain-specific 전제와 정면 충돌하며
   **동시에 최고의 foil**.

---

## §8. 우리 작업으로의 함의

1. **C276 census의 위치가 문헌으로 확정됐다** — "완전성을 선언하는 논리"는 40년간 완성됐지만 "선언이
   틀렸음을 탐지하는 방법"은 이름조차 없다. census는 그 자리에 있다.
2. **★missing-mass 승격 제안(무료·후속)**: 현재 census는 **센 것**이다. 여기에 **Good-Turing missing mass**
   (Lee & Böhme, **ICLR 2025** — missing mass를 *labeling saturation* 판정 기준으로 정식화)를 얹으면
   **안 본 것의 질량**까지 추정된다. 그러면 (iii)이 "전수인가"라는 이진 판정에서 **"미관측 질량이 ε 미만"**
   이라는 **연속 판정**으로 올라가고, [[46]]에 기록한 **Schaeffer 방어(연속 지표 병행)**와도 맞물린다.
   ⚠**species-richness estimator를 NLP label inventory·정책 열거에 적용한 연구는 발견되지 않음 = whitespace.**
   (2024 CSUR 서베이 Def 3: `Recall = |σ(K)|/|σ(Kᵢ)|` — *"KB recall is a real-valued concept, while KB
   completeness is a binary concept"*.)
3. **A2 비대화 경고**: ADB의 known-ratio↑ → open-class F1↓(84.56→66.47)는 **A2를 촘촘히 할수록 밖을 못
   본다**는 뜻이다. [[13]] 흡수 우선순위와 충돌하지 않도록 **A2 확장 시 open-detection 지표를 병행 계측**할 것.
4. **입도 처방 이식**: AMIE+의 range 제한 처방 = C279 식별표 멤버십 전환. 문헌 근거가 붙었다.

---

## §9. 인용 위생 — 정정 12건 (전부 내 지시 전제의 오류)

| 잘못 알려진 것 | 검증된 사실 |
|---|---|
| Moore autoepistemic 초기판 = IJCAI-85 | **IJCAI-83**, pp. 272–279 |
| Etzioni LCW 선행 = AAAI-94 | **KR'94**, pp. 178–189 |
| EQL-Lite 저자 4인 | **5인** — **Domenico Lembo** 누락 |
| Cortés-Calabuig 후속 = KR 2006 | **KR 2008**(81–91) + 별도 **LPAR 2006** |
| Motro = "relative sound/complete **view**" | **query completeness statement** + view 기반 rewriting. view 라벨링은 Calvanese/Grahne 계보 |
| Ligatti et al. 2009 = TISSEC 12(2) Art.9 | **12(3), Article 19** |
| Kambhampati "Can LLMs Really Reason and Plan?" = CACM 논문 | **BLOG@CACM 포스트**(심사 없음). 정식 인용은 *Annals NYAS* 1534:15–18 |
| ROUTE = AAAI 2025 | **ICLR 2025** |
| Ask-before-Plan에 GPT-4 결과 | **없음** — proprietary는 GPT-3.5 하나 |
| CaMeL 저자 | **10인** — Andreas Terzis 누락 주의 |
| IsolateGPT 5저자 | **Umar Iqbal** |
| McCarthy 1980 논문에 CWA 연결 | 1980년 본문에 "closed world" **0회**. 근거는 **1986년** *"This generalizes the closed world assumption."* |

**⚠함정**: McCarthy의 Stanford `circumscription.pdf`는 헤더에 **1986**이 찍혀 있으나 내용은 **1980년 논문**.

### 인용 불가 / 미검증
- **NexusBench·NexusRaven**: 논문·arXiv 없음, blog `@misc`만. "GPT-4 대비 7%"는 vendor 자기보고 [D]고정.
- **Falcone et al. STTT 2012의 계층별 enforceable 경계** — 원문 확보 실패(HAL 차단·Springer 유료). 초록만 검증.
- **Reiter 1984 본문 미열람**(서지만 3중 확인) — 우리 주장의 최대 위협이므로 **원문 확보 필수**.
- Bauer-Leucker-Schallhart TOSEM 2011 초록 축자(유료벽) → Bollig 2026 강의노트 인용 권장.
- Darari et al. RDF completeness statement 내부 구문·복잡도(저자 PDF 404).
- Lifschitz 1985 정리 내용 · Sandhu TAM 1992 정리 진술 · AKBC 2016 본문.
- **preprint 상태**(미심사): Progent · FIDES · RTBAS · LlamaFirewall · Constitutional Classifiers ·
  Ray 2026 · David 2026 · Constraint Tax · Format Tax · Blind-Spot Mass.
- §6-1 문헌 밖 대응어(expressio unius 등) — 1차 미검증.
- ⚠**요약 모델 날조 3건 적발**(DS-DST 결과표 통째·BERT-DST 공저자·CLINC 수치) — 전부 원문 재독으로 교정.
  **수치 인용 전 원문 대조 원칙 유지.**

---

## §10. 논문 프레이밍 규율 (적대 검증 결론)

**금지**: "새로운 기준(novel criterion)"으로 헤드라인. Reiter 1984 / Lutz et al. 2015 / DMN 2016 /
MultiWOZ 2.2 중 하나로 즉사한다.

**권고 문구**:
> "우리는 CWA·domain closure(Reiter 1984)와 decision-table completeness(Calvanese et al. 2016)를,
> KB 완전성 측정(Galárraga et al. 2017)의 방법론과 결합하여 **LLM-엔진 할당의 사전 게이트로 재배치**한다."

**용어 충돌 필수 처리**: [[22]]의 **"닫힌 술어(closed predicate)"**는 **Lutz, Seylan, Wolter (IJCAI 2015)
"Ontology-Mediated Queries with Closed Predicates"**의 정식 개념명과 같다. **인용 없이 쓰면 표절로 읽힌다.**
델타 명시: 우리는 그 폐쇄성을 **선언이 아니라 측정**하고, 그 측정을 **할당 결정에 쓴다**.

**모트** = (3조건 연언) × (NL 정책 코퍼스 census) × (할당 게이트) × (위반 시 false enforcement라는 **측정된
상쇄**). [[46]]의 패턴과 동일 — 부품은 전부 선점, 모트는 합성.

**★Reiter 대응 문장(§1 ④의 방어선을 문면화)**: 세 조건의 *목록*은 Reiter 1984의 것이므로 양보·인용한다.
우리 것은 목록이 아니라 **그 조건들의 지위 전환**이다 — Reiter에서 셋은 *이론이 담아야 할 공리*
(stipulated, syntactic)이고, 우리에게 셋은 *도메인에 대해 성립 여부를 재는 술어*(measured, extensional)이며,
위반 시 처방이 Reiter의 *"공리를 고쳐라"*가 아니라 **"닫힌 취급을 기각하고 표면화하라"**다.
⚠**이 세 마디 중 하나라도 약해지면 Reiter가 그대로 선행기술이 된다** — 논문 문면·초록·기여 목록
전부에서 셋을 함께 유지할 것.
