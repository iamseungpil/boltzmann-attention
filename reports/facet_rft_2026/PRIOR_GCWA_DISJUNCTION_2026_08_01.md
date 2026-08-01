# 선행연구 정본 — 정책의 OR를 닫힌 술어로 다루기: GCWA 계보와 그 비용 (2026-08-01)

> **상태 = 딥리서치 4클러스터 완료 · 인용 가능.**
> **출발 질문**(사용자 2026-08-01): *"CWA에서 OR가 깨진다면, 이후 연구에서 OR를 포함한 연구가 있나?"*
> **방법** = 4클러스터(①Minker 원전 형식화 ②circumscription·ASP 계보 ③복잡도·실행가능성
> ④LLM 에이전트 적용 화이트스페이스), 전 인용 1차 출처 대조.
> **연결** = [[22]](닫힌/열린 술어) · [[50]]/C281·C283(엔진-이관 3조건) · §1.3(상쇄법칙) · C276/C279(merchant)

---

## §0. 한 문단

Reiter의 CWA는 `Pa ∨ Pb`에서 `¬Pa`·`¬Pb`를 둘 다 도출해 **비일관**이 된다. 처방은 1982년에 나왔다 —
**Minker의 GCWA**: *"어떤 minimal model에도 없을 때만 부정한다."* 이후 40년간 EGCWA·CCWA·ECWA·
WGCWA로 갈라졌고, 계산 층에서는 **ASP(answer set)가 대체**했다. 우리에게 중요한 결과는 셋이다.
**(1) 이름을 정확히 써야 한다** — "모든 minimal model이 만족하는 것만 강제"는 GCWA가 아니라 **EGCWA**
(`EGCWA(P)=MM(P)`)이고, GCWA는 **원자·부정원자로 제한된 단편**이라 임의 논리식 연역의 완전성이
미해결이다. **(2) 비용은 우리 체제에서 거의 0이다** — 강제 조건에 부정이 없으면 `MM(T)⊨F ⟺ T⊨F`라
minimal model이 **아예 필요 없고**, 부정을 강제해도 K개 disjunction×d분지면 `O(dᴷ)`이며 우리 banking의
실측 **K=2~3**이다. **(3) 화이트스페이스가 좁고 진짜다** — 인식적 disjunction(D2)을 정책 언어
의미론으로 다루는 에이전트 가드레일은 **하나도 없으나**, 문제 진술 자체는 **PolicyBank가 선점**했다.
그리고 ⚠**DL의 "closed predicate"(Lutz et al.)는 이 계보와 무관하다** — 최소화가 아니라 **데이터에
고정**이고 참고문헌에 Minker·GCWA·minimal model이 **0건**이다. [[22]] 서술에서 두 계보를 잇지 말 것.

---

## §1. 형식 위계 — 무엇이 무엇의 단편인가

```
        naive CWA (Reiter 1978)          ← 논리합에서 비일관
             │ 일관할 때만
             ▼
        GCWA (Minker 1982)               ← 결론이 원자·부정원자로 제한
             │ + 부정 clause
             ▼
        EGCWA (Yahya & Henschen 1985)    ← EGCWA(P) = MM(P)  ★우리가 원하는 것
             │ + 술어 분할 ⟨P̂;Q̂;Ẑ⟩
             ▼
        ECWA (Gelfond·Przymusinska·Przymusinski 1989)
             ≡ circumscription (명제, Lifschitz 정식화)
```

**★가장 중요한 정정**: *"GCWA ≡ minimal-model entailment ≡ circumscription"은 성립하지 않는다.*
GCWA는 최소모델 함의를 **원자·부정원자로 제한한 단편**이고, 완전한 등가물은 **ECWA**다.
- Suchenek 축자: *"both cwa and GCWA restrict conclusions of cwaS to atomic and negated atomic sentences"* [S]
- Cadoli & Lenzerini (AAAI-90) 축자: *"the ECWA is equivalent to circumscription, at least for propositional formulae"* [S]

### 정의 (Minker 1982 — ★1차 정의는 구문적이다)
```
1차(구문):   ¬A ∈ GCWA(P)  ⟺  ∀K. ( P ⊢ A ∨ K  ⟹  P ⊢ K )      (K = ground positive disjunction)
동치(모델):  ¬A ∈ GCWA(P)  ⟺  A ∉ M  for every minimal Herbrand model M
```
Chomicki & Subrahmanian(UNC TR89-036) 축자: *"He also showed that the above definition has a
model-theoretic counterpart"* — **구문 정의가 원 정의, minimal-model은 Minker가 별도로 보인 정리**.

### 서지 (교차 확인 완료)
| 항목 | 서지 | 등급 |
|---|---|---|
| **GCWA** | Jack Minker, *On Indefinite Databases and the Closed World Assumption*, **CADE-6, LNCS 138, pp.292–308, 1982**, DOI `10.1007/BFb0000066` | [S]서지(본문 미확보·유료) |
| **EGCWA** | Yahya & Henschen, *Deduction in Non-Horn Databases*, **JAR 1(2):141–160, 1985**, DOI `10.1007/BF00244994` | [S]서지 |
| **CCWA** | Gelfond & Przymusinska, *Negation as Failure: Careful Closure Procedure*, **AIJ 30(3):273–287, 1986** | [S]서지 |
| **ECWA** | Gelfond, Przymusinska, Przymusinski, *On the Relationship Between Circumscription and Negation as Failure*, **AIJ 38(1):75–94, 1989** (학회판 PODS 1986, pp.133–139) | [S]서지 |
| **WGCWA** | Rajasekar, Lobo & Minker, **JAR 5:293–307, 1989** — ★Minker 본인이 GCWA를 **약화**시킴 | [M] |
| **통합 교과서** | Lobo, Minker, Rajasekar, *Foundations of Disjunctive Logic Programming*, **MIT Press 1992**, 307쪽(§5.1 GCWA p.106·§5.3 p.127·§6.1 WGCWA p.141) | [S] |

⚠**오귀속 주의 4건**
1. **"Horn에서 CWA로 축약"은 Minker가 아니라 Shepherdson**(*Negation in Logic Programming*, in Minker
   ed. 1988, **Theorem 32.5**) 귀속.
2. **Lifschitz "Computing Circumscription"(IJCAI-85)은 CWA와 무관** — 전문 전수 검색에서
   Minker·CWA·closed-world·minimal model **0건**. 맞는 논문은 **Lifschitz, AIJ 27(2):229–235, 1985**
   *"Closed-world databases and circumscription"*이나 **전문 입수 실패**(Unpaywall `oa_status: closed`)
   ⇒ 정리 번호·조건절 **[?]**. 내용은 citing paper 2건의 독립 재진술로만 확인.
3. **"Saturation, nonmonotonic reasoning and the CWA"(AIJ 25, 1985)는 Bossu & Siegel**이지 Lifschitz 아님.
4. **Semantic Scholar가 Minker 1982를 1987년으로 오기** — dblp·Crossref는 1982. S2를 연도 근거로 쓰지 말 것.

---

## §2. GCWA에 대한 비판 5종 (인용 시 함께 적을 것)

| # | 비판 | 처방 |
|---|---|---|
| ① | 결론이 원자에 갇혀 **부정 clause를 못 뽑음** | EGCWA |
| ② | **전부-닫기만 가능**, 술어 선택 불가 | CCWA (`P̂=A(P)`면 GCWA로 축약 — Eiter&Gottlob 축자) |
| ③ | EGCWA도 분할 불가 | ECWA ≡ circumscription |
| ④ | **너무 강함** — `p∨q`를 배타적으로 읽어 `{p,q}` 모델을 버림 | WGCWA / DDR (coNP로 저렴) |
| ⑤ | 데이터 교환에서는 **너무 약함** | Hernich(LMCS): Reiter CWA·EGCWA는 *"too strong"*, GCWA는 *"too weak"* |

⚠**④는 단정 금지**: Dix(Dagstuhl Sem. Rep. 150, 1996)가 반박한다 — `{p∨q, q∨r, r∨s}`의 minimal model
`{q,r}`은 두 번째 절을 **포괄적으로** 읽는다. *"minimal/maximal model"*이라 말할 것이지
inclusive/exclusive 용어는 단순 사례에만 적절. **"GCWA = exclusive-or"로 쓰면 반박당한다.**

---

## §3. ★복잡도 — "비싸다"는 우리 체제에서 틀렸다

| 문제 | 제약 | 복잡도 | 출처 |
|---|---|---|---|
| ECWA/CIRC/**EGCWA** 연역 | 일반 | **Π₂ᵖ-complete** | Eiter&Gottlob TCS'93 **Thm 3.3** |
| **`MM(T) ⊨ x` (양의 원자)** | 일반 | **coNP-complete = 고전 연역과 동일** | TCS'93 §3.1 · Dantsin et al. CSUR'01 **Thm 6.1(i)** |
| GCWA 연역, F=임의 논리식 | 일반 | Π₂ᵖ-hard, 상계 **Δ₃ᵖ[O(log n)]**, **완전성 미해결** | AMAI'95 **Prop 16·Thm 17** |
| ECWA/EGCWA/CIRC | **2CNF(Krom)** | **coNP-complete**(한 레벨 하강) | TCS'93 **Thm 3.4** |
| CWA·GCWA·EGCWA | **Horn** | **P(다항)** — 모두 일치 | Cadoli & Lenzerini AAAI-90 **Fact 1** |
| minimal model finding/checking | **HCF** | **P(다항)** | Ben-Eliyahu-Zohary & Palopoli AIJ'97 |
| skeptical ASP | **backdoor 크기 k** | **FPT `O(2ᵏn²)`** | Fichte & Szeider AAAI'13 |
| 동일 | **treewidth k** | `O(2^{2k+2}‖Π‖)` | Fichte et al. LPNMR'17 **Thm 1** |
| minimal model **checking** | 일반 CNF | **coNP-complete** | Cadoli IPL'92 |
| GCWA | **1차·함수기호** | **Π⁰₂-complete = 결정불가** | Chomicki & Subrahmanian IPL'90 |

### ★실무 판정 (3단계)
- **(A) 강제 조건에 부정이 없으면 minimal model이 불필요**: `MM(T)⊨F ⟺ T⊨F`
  (TCS'93 §3.1 축자: *"MM(T) |= F iff T |= F, hence the problem is in co-NP"*). Horn이면 **선형시간**.
  **컴플라이언스 게이트 대다수가 여기 해당**("이 조건이 성립해야 한다" 형태).
- **(B) 부정을 강제해도 K작으면 `O(dᴷ)`**: d=2·K=20 → 약 10⁶회 선형 계산. 정책은 정적이라
  **사전계산으로 흡수**. ⚠이 `O(dᴷ)` 유도는 **에이전트 파생이지 인용 가능 정리가 아님** —
  인용은 Fichte & Szeider의 FPT 결과로.
- **(C) 탈출구 3개 독립 증명**: HCF(다항·**DLV가 런타임에 실제 검사**) / 2CNF(coNP) / backdoor·treewidth(FPT).

### ⚠실무 낙관에 대한 부분 철회 (클러스터 간 대조로 교정)
ASP Competition 7에서 구조적 Σ₂ᵖ 도메인은 **20/20·인스턴스당 ~12초**로 풀린다. **그러나**
조직위가 non-HCF 논리합 서브트랙을 *"sparsely populated"*라 자인하고, 랜덤 논리합 인스턴스는
여전히 대량 timeout이며, **확인된 산업 사례의 논리합 사용은 head-cycle-free guess&check(=NP급)에
그친다.** ⇒ *"논리합이 실무에서 잘 풀린다"*는 **HCF 조건부**로만 쓸 것.

⚠**정정**: Eiter & Gottlob 1995는 **brave reasoning을 다루지 않는다**(축자: *"we will not consider
brave reasoning in our analysis"*). 증명한 것은 Consistency Σ₂ᵖ-complete와 **cautious** Entailment
Π₂ᵖ-complete다.

---

## §4. ★★[[22]]에 대한 경고 — DL "closed predicate"는 이 계보가 아니다

Lutz, Seylan, Wolter (IJCAI 2015 / LMCS 2019)의 참고문헌을 **전수 추출해 검색**한 결과:

| 검색어 | IJCAI 2015(23건) | LMCS 2019(60여 건) |
|---|---|---|
| Minker / GCWA / minimal model | **0** | **0** |
| Gelfond / Lifschitz / McCarthy | **0** | **0** |
| answer set / stable model | **0** | **0** |
| Reiter | **0** | 1 — 단 **Reiter 1992** *"What should a database know?"*이지 1978 CWA 아님 |

그리고 **형식적으로도 다르다** — Lutz et al. §1 축자: *"the interpretation of CWA predicates is fixed
to what is explicitly stated"*. **데이터에 고정하는 것이지 최소화가 아니다.**

⇒ **두 계보는 "부분적 CWA"라는 동기만 공유하고 합류한 적이 없다.** [[22]]·[[50]]에서 "닫힌 술어"를
쓸 때 **용어 충돌(Lutz et al. 인용 의무)은 유지하되, GCWA 계보와 잇는 서술은 하지 말 것.**

---

## §5. LLM 에이전트 적용 — 화이트스페이스 판정

### ★D1 / D2 구분 (논문 초반에 못박을 것)
| | 뜻 | minimal model 필요? |
|---|---|---|
| **D1** 진리함수적 OR | 완전히 알려진 concrete 인자 위의 `a OR b` | **불필요** — 단일 모델 즉시 판정 |
| **D2** 인식적 disjunction | 정책 KB가 "A 또는 B 적용"이라 **어느 쪽인지 엔진이 모름** | **여기서만 필요** |

### 가드레일 6종 감사 — **D2를 다루는 것은 0건**
| 시스템 | 정책 언어 | D1 | D2 | 기본값 |
|---|---|---|---|---|
| **AgentSpec**(2503.18666) | DSL + Python 술어 | **NO** — 문법에 OR 연산자 자체가 없음(`⟨Pred⟩ ::= True\|False\|!⟨Pred⟩\|⟨DomainSpecificPred⟩`)·본문 축자 *"conjunctions of predicates"* | NO | denylist |
| **Progent**(2504.11703) | DSL(JSON Schema) | **YES**(`e or e'`·`anyOf`) | NO | **allowlist**(축자: *"we block the tool call by default for security"*) |
| **CaMeL**(2503.18813) | Python 콜백(DSL 거부) | YES(Python 표현력) | NO | allowlist→사용자 확인 |
| **GuardAgent**(2406.09187) | 자연어→LLM 생성 Python | **NO**(속성=값 conjunction) | NO | denylist |
| **NeMo/Colang**(2310.10501) | 흐름 DSL | **YES**(`or when`·`else when`) | NO | **open-world**(미정의 입력을 LLM이 지어냄) |
| **CMU 심볼릭**(2604.15579) | 없음(하드코딩) | 논의 없음 | NO | allowlist |

**minimal model / skeptical vs credulous / closed-world assumption / over-·under-approximation 이라는
용어는 6편 전체에서 단 한 번도 등장하지 않는다.** Progent만 CWA를 *구현*하되 이름을 안 붙이고,
CaMeL의 STRICT/NORMAL만 approximation 선택을 *구현*하되 이름을 안 붙인다.

### ☠문제 진술은 선점됐다 — PolicyBank (arXiv:2604.15505)
축자: *"imprecise quantifiers or exemplar lists that the agent interprets as exhaustive"* —
**예시 목록을 소진적으로 읽는 실패를 그대로 명명**했고 *"Under-Specified Exceptions"*가
*"causing the agent to reject legitimate edge cases"*(과잉차단)까지 포함한다.
**단 해법은 형식의미론이 아니라 NL 메모리·검색**이고 본문에 disjunction·minimal model·skeptical이 없다.
⇒ 프레이밍: **"PolicyBank가 NL 메모리로 *완화*한 것을, 우리는 형식 의미론으로 *보장*한다."**

### 판정 = **PARTIALLY NOVEL**
**미선점**: NL 정책의 disjunction을 *제거하지 않고* 인식적 disjunctive KB로 보존한 뒤,
**복수 minimal model에 걸친 cautious(skeptical) consequence만 tool-call 게이트에서 집행**하는 것.
세 공동체가 만난 적이 없다 — 논리 쪽은 EGCWA를 갖되 에이전트를 안 보고, 가드레일 쪽은 정책 언어가
구성상 Horn/conjunctive라 **D2를 표현조차 못 하며**, 법률-형식화 쪽은 처방이 *제거*(DDL 프롬프트 축자:
*"If you want to represent a disjunction, please use multiple rules"*)이거나 *노출*이지 **모호성 하 집행**이 아니다.

**예상 공격**: *"cautious entailment는 clingo 플래그 하나 아닌가"* ⇒ 방어를 **집행 의미론**에 실을 것 —
"모든 minimal model에서 위반일 때만 차단"이 **soundness 보존 under-approximation**임을 정리로 세우고
**over-blocking 감소를 계측**한다([[19]] Δspurious≤0과 맞물림).

---

## §6. 우리 문제에의 적용 — 동일성 축

C283 이후 확정된 대로, banking의 잔여는 닫힘(DCA/CWA)이 아니라 **동일성(UNA)**이다.
`"Target - Eco Collection"`이 정책의 `"Target"`과 같은 개체인가 — **읽기가 둘**이다.

- **credulous**(어느 한 읽기에서라도 제외) → 차단 → **한 해석에만 근거한 거짓 집행**
- **skeptical**(모든 읽기에서 제외일 때만) → 통과 → **거짓 집행 0**, gold 대비 under-enforce

⇒ **"안전한 방향으로 틀린다"의 형식 이름이 skeptical(cautious) entailment**다.

**비용은 우리 규모에서 무시할 수준**(census 실측):
```
target -> ['Target', 'Target - Eco Collection']
dell   -> ['Dell',   'Dell Technologies']
delta  -> ['Delta Airlines', 'Delta Sky Club']   (제외문맥 밖)
```
**K = 2~3, d = 2 → 모델 4~8개.** Π₂ᵖ를 걱정할 자리가 아니다.

---

## §7. 함정 · 미검증

**함정 3**
1. **GCWA를 쓰지 말 것** — 임의 논리식 GCWA 연역은 Π₂ᵖ 멤버십조차 미증명(상계 Δ₃ᵖ). 원하는 것은 **EGCWA**.
2. **grounding blowup** — 명제 복잡도는 괜찮아도 1차 정책을 큰 도메인에 grounding하면 `‖P‖`가 폭발.
   1차+함수기호는 **결정불가**(Π⁰₂).
3. **minimal model checking이 coNP-complete** ⇒ "후보 모델 하나를 받아 minimal인지 검증"하는 설계는
   피하고 **열거 기반(dᴷ)**으로.

**미검증 [?]**
- **Minker 1982 본문 전체 미확보**(Springer 유료·OA 0건) ⇒ 정의·정리 번호는 2차 문헌 귀속.
  단 4개 독립 문헌(Chomicki–Subrahmanian / Gries / Suchenek / Eiter–Gottlob)이 **일치**.
- **Lifschitz AIJ 27(2) 1985 전문 미확보** — 정리 번호·조건절 미확인.
- Yahya & Henschen 1985 · GPP 1986/1989 본문 미확보(서지만).
- Przymusinski 1988 챕터 본문 미독 — "perfect model은 minimal model"은 GL88 Thm 1+Cor 1의 **합성 유도**.
- ⚠**Baral 2003이 GCWA를 ASP 하위로 포섭한다는 서술은 근거 없음** — 책 전체에 GCWA가 **2회**만 등장하고
  KR 형식체계 비교 절에 **항목이 없음**. 인용 금지.
- Eiter & Gottlob TCS'93 addendum 페이지 불일치(118:115 vs 118:315) 미해결.

**공개 접근 대체 출처**(Minker 원문이 유료이므로 원장 인용용): Chomicki & Subrahmanian UNC TR89-036 ·
Eiter & Gottlob AMAI'95(저자 아카이브) · Gries CEUR Vol-477 paper 45 · Suchenek · Hernich arXiv:1107.1456 ·
Dix Dagstuhl Sem. Rep. 150.

## §8. 살아남은 것

Przymusinski 본인의 통합 진술이 가장 간명하다 — *"not F holds if F is false in all minimal models."*
(여기서 Minker 1982·Gelfond et al. 1989·McCarthy 1980을 **한 줄에 나란히** 인용).
실무로 살아남은 것은 **answer set**이고 최소성은 solver 안에 흡수되어 사용자에게 보이지 않는다.
**GCWA·circumscription은 개념적 조상으로 남았고 계산 층은 ASP가 대체했다.**
