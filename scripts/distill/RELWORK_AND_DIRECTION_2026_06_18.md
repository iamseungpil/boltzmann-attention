# 선행연구 리뷰 + 방향 명시 (2026-06-18) — 무엇이 이미 확정됐고, 우리 고유 기여는 무엇인가

> **목적**: 통합 TBox/Scaffold 설계(`INTEGRATED_TBOX_DESIGN_2026_06_18.md`·`INTEGRATED_SCAFFOLD_IMPL_DESIGN_2026_06_18.md`)를 *구현 착수 전*에, 딥리서치(52 검증 claim·~20 출처) + 핵심 4편 primary 재검증으로 정렬한다. **coworker 공유용 자립 문서** — 아래 §1만 읽으면 thesis 프레임이 선다.
> **한 줄 결론 (정독 종료·2026-06-18)**: "전이"는 우리 신규성이 *아니다*(D3ST/STAR/ToolLLM/TGRL가 schema·tool·습관 전이를 이미 함). **우리 신규 = "tool-use 계획규칙이 *닫힌(closure) 유한 생성원 기저*로 환원되고, *그 규칙-추상화*를 작은 모델이 학습·전이한다"** — 스키마도 도구도 실행도 습관도 아닌 **규칙의 추상화** 전이. 전이는 닫힘의 *귀결*(데모)이지 신규 주장이 아님. 결정론 leg(게이트·복구·deferral·측정)는 전부 발표됨 = 재사용·인용. ⇒ §1.5 핵심주장·§4 차별·§6 keystone.

---

## 1. thesis 한 문단 (coworker용 프레임)
자연어 멀티턴 요청을 도메인 온톨로지(ABox)로 재해석해 native function-calling 시퀀스를 추론·실행하는 agentic planner를, **작은 모델 weight(TBox)에 학습**시키고, 본 적 없는 도메인은 **ABox 교체만으로 재학습 0 전이**한다. 분담 = **LEARN**(LLM·NL→formalize·도메인일반·전이) / **PROVIDE**(ABox·도메인특정·swap) / **DETERMINISTIC**(decidable: gate·resolve·verify·고정). 벤치 = τ²-bench·SOPBench·TaskBench·ComplexFuncBench. 측정 = 실 τ² user-sim e2e + 전이 매트릭스.

---

## 1.5 ★우리 핵심 주장 (정독 종료 후 확정) + related work 차별

**핵심 주장 (한 문장)**:
> **Tool-use 계획(planning)은 *닫힌(closure-justified) 유한 생성원 기저*로 환원된다 — flow 기본형(P1-P9·Böhm–Jacopini) + content 연산(Codd 관계대수). 작은 모델은 *이 계획-규칙의 추상화*를 학습하고, 새 도메인엔 *그 추상화를 재조합* + ABox(사실)만 교체해 재학습 0 전이한다. decidable한 부분은 결정론 엔진에 offload.**

**무엇이 *추상화(전이대상)*이고 무엇이 *데이터*인가 — 이게 차별의 핵심축**:
| | 전이되는 것 = *추상화* | 데이터(= 안 배움·제공) | 닫힘? |
|---|---|---|---|
| **D3ST** (schema-guided DST) | 스키마 *읽기*(NL설명→슬롯) | 스키마 텍스트 | — (기저 없음) |
| **STAR** | 순서도 *따라가기*(실행) | **순서도(=계획규칙!)** ← 규칙이 데이터 | ❌ 명시적 비폐포(이탈→학습) |
| **ToolLLM** | 도구 *사용*(문서→호출) | API 문서 | ❌ open set(16k·무한) |
| **TGRL** | 도구사용 *습관*(opaque RL) | 도구 인터페이스 | ❌ 기저·closure 주장 없음 |
| **ATA·LLM-Modulo** | (LLM은 학습 안 함·프롬프트 인코딩) | symbolic KB(도메인마다 재구축) | — |
| **★우리** | **계획-규칙의 추상화**(닫힌 생성원 기저) | **ABox(카탈로그·정책 사실)** | ✅ closure-정당화(scope 내) |

**차별 한 줄**: 남들은 **스키마·도구·실행·습관**을 전이하거나(D3ST·ToolLLM·STAR·TGRL), 계획규칙을 *데이터에 둔다*(STAR 순서도·ATA KB). **우리만 *계획규칙 자체를 닫힌 추상화로 학습해 전이*하고, 데이터엔 사실(ABox)만 둔다.** STAR는 정확히 *반대*(규칙=데이터/실행=학습) → 대비가 가장 선명.

**왜 "닫힘"이 신규의 핵심인가**: ToolLLM=open set(무한·완결 없음)·STAR=비폐포(유한이나 샘·땜빵). **우리만 closure**(유한 기저가 scope 내 *모든* 계획을 생성·증명상). ⇒ 전이가 "요행"이 아니라 **원리적·완전**(새 도메인 = 같은 기저 재조합 + ABox). **"전이"가 신규가 아니라 "닫힌 규칙-기저라서 전이가 원리적"이 신규.**

**대조 판정 (우리 3후보 × schema-transfer 문헌·정독 2026-06-18)**: ① **closure-정당화 유한 기저 = 생존(clean·아무도 closure 주장 안 함)** / ② 결정론 offload+전이 결합 = ◐ STAR 부분선점(단 그들=schema-graph 추종·우리=decidable 계산 offload) / ③ 벤치횡단 = ❌ ToolLLM 부분선점(cross-API OOD). ⇒ **신규의 무게중심 = ①(닫힘 기저)**, ②③는 보조·차별 sharpening 필요.

⚠️ **정직(과몰입 방지)**: (a) closure는 *scope 한정*(transactional control/data-flow·flow층 강·policy층 상대닫힘). (b) "규칙 추상화가 *weight로* 전이"는 *주장* — 실증은 content-op 라우팅(§21)만·flow는 결정론 scaffold가 나름(adapter held-out≈0). ⇒ **남은 핵심 실험 = "닫힌 기저 학습이 open(ToolLLM류)·비폐포(STAR류) 대비 전이·커버리지에서 이득"을 보이는 것** + C0(라우팅 native 전이).

---

## 2. 검증된 선행연구 (신뢰 티어 명시)

**티어 표기**: 🟢 primary 정독 검증(WebFetch) · 🔵 cutoff(2026-01) 이전·확립 · 🟡 딥리서치 surface·primary 미검증(인용 전 확인 요).

| arxiv | 제목 (약칭) | 날짜 | 티어 | 우리와의 관계 |
|---|---|---|---|---|
| **2603.20449** | Solver-Aided Verification of Policy Compliance (Winston·Winston·Just) | 2026-03 | 🟢 | ★**gate-leg 최근접 rival** — NL 정책→SMT-LIB→Z3 런타임 게이트·**τ-bench**·위반 차단 |
| **2510.16381** | ATA: Neuro-Symbolic Autonomous Trustworthy Agents (Peer·Stabinger) | 2025-10 | 🟢 | ★**thesis-framing 최근접 rival** — LLM=NL→형식 KB·symbolic engine 결정·소형>대형·결정론·injection 면역 |
| **2604.07036** | ReDAct: Uncertainty-Aware Deferral for LLM Agents | 2026-04 | 🟢 | calibrated-threshold defer(소형→대형)·ALFWorld/MiniGrid |
| **2606.01416** | Self-Healing Agentic Orchestrators | 2026-05 | 🟢 | monitor→diagnose→recover→verify 결정론 루프·fault-injection 98.8% |
| **2511.21689** | ToolOrchestra | 2025-11 | 🟢 | ★**가장 위험한 rival** — 8B RL이 **τ² 80.2%@10.3¢ > GPT-5 77.7%@31.3¢**·HLE 37.1%>35.1%·미관측 도구=*학습* 일반화 |
| **2510.16381** | ATA Neuro-Symbolic (Peer·Stabinger) | 2025-10 | 🟢 | LLM=NL→형식 KB·symbolic 결정·**자동 72.94<gemini-pro 76.50·human-KB 87.17만 추월**·단발 보험 reasoning |
| **2603.20449** | Solver-Aided Policy Compliance | 2026-03 | 🟢 | NL→SMT-LIB→Z3 게이트·**airline만**·위반 50%→29%·**human-번역**·**게이트 only** |
| 2402.01817 | LLM-Modulo (Kambhampati, ICML'24) | 2024 | 🟢 | LLM=근사 지식원 + 외부 sound verifier 루프(LLM 자기검증 불가)·"생성+외부검증" 정전 이름 |
| 2407.01032 | Overcoming Common Flaws in Selective Classification Eval (Traub) | 2024 | 🟢 | ★측정 규율 — **AUGRC**(generalized risk-coverage 곡선)·단일점 게이밍 |
| 2502.17216 | Intermediate Languages Matter (neurosymbolic) | 2025 | 🟢 | NL→형식 IR 선택이 1차 결정변수·context-aware 인코딩만 효과·**단발 reasoning(ProntoQA/ProofWriter)** |
| 2509.25370 | AgentDebug / Where LLM Agents Fail | 2025 | 🟢 | AgentErrorTaxonomy·귀인+repair·+24%/+17%/최대 26%·**ReAct(ALFWorld/GAIA/WebShop)=별 벤치족** |
| 1902.06349 | Learning to Infer Program Sketches (SketchAdapt, Nye) | 2019 | 🔵 | 학습 sketch + symbolic hole-fill·경계 *학습*(우리 델타축)·⚠️PDF 렌더 실패·기존 thesis로 신뢰 |
| 2107.11277 | Machine Learning with a Reject Option (survey) | 2021 | 🔵 | (h,r) 예측기+거부기·3분류(separated/dependent/integrated) |
| 2410.10347 | Unified Routing and Cascading for LLMs | 2024 | 🟢 | "cascade routing"·**학습 quality estimator가 핵심 인자**·model-selection deferral(verdict 아님) |

⚠️ **인용 규율(메모리 `40-settled-cite-only`·`feedback-arxiv-citation-discipline`)**: 본 표 🟢 = 본 세션 primary 정독 완료(2026-06-18). 딥리서치가 surface했으나 본 문서 미사용 post-cutoff snippet(2603.04474·기타 2602/2603류)은 사용 시 primary 검증. SketchAdapt(🔵)만 PDF 렌더 실패 — 2019 확립 논문이라 기존 thesis(경계 학습)로 신뢰하되 인용 시 재확인.

---

## 3. 버킷 A — 이미 *확정*됨 (재증명 금지·인용으로 대체)

0. **★소형 모델이 우리 *바로 그* τ²-bench서 GPT-5를 추월(80.2% vs 77.7%)·~1/3 비용** = **ToolOrchestra 2511.21689** (monolithic 8B RL). ⇒ ★**"소형이 대형 tool-use 성능 도달"은 우리 헤드라인이 될 수 없다 — monolithic-learned로 이미 됨, 그것도 우리 벤치에서.** 우리 차별은 *결과*가 아니라 *방식*(아래 §6).
1. **NL 정책 → 형식제약 → 런타임 결정론 게이트가 위반 차단·정확도 유지 (τ-bench airline)** = **2603.20449**. ⇒ 우리 **GateInterpreter의 gate-compliance leg는 novelty 아님 = 재현.** "게이트가 작동함" 단독 실험 불요.
2. **"LLM은 NL→형식화, 건전 판단은 결정론/기호"라는 신경-기호 *분업 원리*** = **ATA 2510.16381**(+LLM-Modulo·neurosymbolic). ⇒ 이 *원리*는 이미 확립됨 = **우리 발명 아님·헤드라인 못 함**(운영 방법은 전혀 다름·§4 가). 단발 보험 reasoning·자동은 대형 못넘음(human-KB만)·KB 재구축.
3. **calibrated-threshold deferral이 대형 품질을 일부 비용에 달성** = **ReDAct 2604.07036**. ⇒ "불확실하면 미루기·calibration"은 확정.
4. **결정론 monitor→diagnose→recover→verify 루프가 cascade 오류 대부분 복구** = **Self-Healing 2606.01416** + **AgentDebug 2509.25370**. ⇒ 우리 facet_check→regen·오차교정 결합은 *패턴으로 확정*.
5. **selective-classification 게이밍 + 사전 threshold + 다중-threshold 곡선(AUGRC) 규율** = **Traub 2407.01032**. ⇒ 측정 방법론 확정 = 채택만(발명 아님). 우리 decidable-비율 = AUGRC식 곡선으로 보고.
6. **모듈식 에이전트의 cascade 오류는 실재·정량** = 2503.13657·AgentDebug. 확정.

---

## 4. ★방법적으로 닮은 선행 = 없다 (정정 2026-06-18·사용자 교정)

이전 판의 "유사하나 델타 있음" 프레이밍은 **틀렸다.** 정독 결과 **운영 방법이 우리와 닮은 선행은 하나도 없다.** 선행은 *부분*만 겹치고, 4종으로 분류된다 — **어느 것도 방법 쌍둥이 아님**:

**(가) 원리-족보만 공유 (방법 ❌·*원리*만)** — ATA `2510.16381`·LLM-Modulo `2402.01817`·neurosymbolic `2502.17216`·Solver-gate `2603.20449`.
- 공유 = **"LLM은 NL→형식화, 건전 판단은 결정론/기호"라는 신경-기호 *분업 원리* 한 줄.** 그게 전부. 운영은 전부 다름(ATA=프롬프트 encoder+100% 기호 솔버+단발 청구+KB 재구축 / 우리=SFT 학습 에이전트+멀티턴 도구행동+결정론은 인자/게이트만+ABox-swap).
- ⇒ 인용 이유 = **"분업 *원리*는 우리 발명 아님(가족 것)"을 정직하게 자리매김**. 방법 경쟁 아님. **원리는 헤드라인 못 함·구현만 우리 것.**

**(나) 결과·벤치 baseline만 겹침 (방법 ❌·*결과*만)** — ToolOrchestra `2511.21689`.
- 공유 = τ²서 "소형 싸게 대형 추월" *결과* + 벤치. 방법 무관(RL model-selection = 모델 중 택일·우리는 객체-수준 도메인 도구호출).
- ⇒ 인용 이유 = "소형>대형은 됐다" 헤드라인 방어용 한 줄. **방법 차별 노력 쓰지 말 것**(다른 문제).

**(다) 구조적 사촌 — 구별 필요 (방법 ❌·*구조*만)** — Schema-Guided Dialogue(Rastogi 2020)·Voyager `2305.16291`.
- SGD = 유한 act + schema로 미관측 서비스 전이(구조 사촌) — 단 **DST이지 tool-use planning 아님·act=고정 annotation(닫힘증명 없음)·*학습 전이 기저* 아님.** Voyager = *열린* skill library(무한·도메인특정·우리와 반대).
- ⇒ reviewer가 "이거랑 같지 않냐" 1순위 → **명시 구별만**.

**(라) 도구로 채택 (우리가 빌려 씀)** — Traub `2407.01032`(측정=AUGRC 곡선)·Codd 1972/Böhm-Jacopini 1966(closure 수학)·Description Logic TBox/ABox(용어 분리·단 그들=손작성 symbolic·우리=*학습* TBox).
- ⇒ 발명 말고 채택·인용.

**(마) ★전이 메커니즘 = 구체 선행 있음 (정정 2026-06-18·검색 누락 시정)** — Schema-Guided Dialogue([SGD/STAR 2010.11853]·[Description-Driven TOD 2201.08904]·[Schema Augmentation 2411.00150])·tool-FC([ToolLLM 2307.16789]·[Tool-Doc 2308.00675])·도메인-불변 planning([TGRL 2510.11184]·Trace2Skill).
- ★**우리가 "핵심 신규"라 한 전이("소형+교체 schema→미관측 도메인 zero-shot")는 신규 메커니즘이 아니라 *변종*이다.** schema-guided DST 커뮤니티가 수년간 한 것: 여러 schema 학습→미관측 schema 일반화·schema=swappable inductive bias·소형모델 cross-task. = 우리 ABox-swap과 구체적으로 동형.
- ⚠️ 이전 딥리서치가 "typed-verdict→decidable combine"으로 좁혀 **이 문헌 전체를 누락**. 가장 관련된 곳을 안 봄 = 신규성 과대평가의 원인.

**★결론 (정정·대조 완료)**: 우리 *부분*은 대부분 구체 선행이 있다 — 전이 메커니즘(마), 게이트(2603.20449), 분업 원리(가). **전체 시스템 그대로**는 없으나 "비어 있는 칸"은 **과장**이었다. 좁은 신규 후보 3 × schema-transfer 문헌 **대조 완료(§1.5·정독 2026-06-18)**: ① **closure-정당화 유한 기저 = 생존(clean)** ② offload+전이 결합 = ◐STAR 부분선점 ③ 벤치횡단 = ❌ToolLLM 부분선점. ⇒ **신규 무게중심 = ①(닫힘 규칙-기저) 단독.** 핵심 주장 = §1.5.

### 4.1 벤치 현실 (τ² = 우리 포트폴리오 중 하나일 뿐)
- rival들은 τ²/τ-bench를 *주 타깃*으로 씀(ToolOrchestra·2603.20449 airline). **우리는 τ²가 SOPBench+TaskBench+CFB+Synth*에 더해진* 하나** — 벤치 횡단 포트폴리오 자체가 우리 셋업이고, **그 포트폴리오를 가로지르는 단일 시스템 = 어떤 rival도 없음.** ⇒ "같은 벤치" 겹침도 부분적(우리=다벤치·그들=단벤치). 벤치-횡단이 차별의 일부.

---

## 4.5 ★전이(transfer) 전용 — 남이 한 것 vs 우리 것 (논문 핵심이라 별도)
"전이"는 세 질문으로 쪼개야 명확: **(1) 무엇이 전이하나** (학습 스킬? 프롬프트?) · **(2) 새 도메인엔 뭘 바꾸나** (가중치 재학습? KB 재구축? 데이터만?) · **(3) 얼마나 멀리** (도메인 내? 벤치 횡단?).

| | 무엇이 전이 | 새 도메인엔 뭘 바꾸나 | 학습 스킬 전이? | 완성? |
|---|---|---|---|---|
| **ToolOrchestra** 2511.21689 | 미관측 *도구* | **아무것도 안 바꿈**(모델이 도구설명 읽음·학습일반화) | ✅ 단 monolithic·black-box RL | ✅**완성**(τ² 발표) |
| **ATA** 2510.16381 | 보험 도메인 간 | **symbolic KB 재구축**(human 검증·= A2 비용) | ❌ 학습스킬 없음(프롬프트 encoder) | ✅**완성**(발표) |
| **2603.20449** | (전이 안 함) | 정책마다 **human SMT 번역** | ❌ | airline만·전이 없음 |
| **우리** | 도메인 **+ 벤치 횡단** | **ABox만 교체**(선언적 catalog+gate_spec) | ✅ 학습된 TBox 라우팅 | ❌**미완성**(C0 동전던지기) |

| **schema-guided DST** SGD/Description-Driven/ToolLLM | 미관측 *도메인/API* | **schema 텍스트 교체**(swappable inductive bias) | ✅ **소형모델 cross-task 학습** | ✅**완성**(수년간·우리 ABox-swap과 구체 동형) |

**🟰 이미 남이 한 것(우리 novelty 아님·정정)**: "**소형 모델 + 교체 schema → 미관측 도메인 zero-shot 전이**" = schema-guided DST(SGD/Description-Driven)·tool-FC(ToolLLM)가 **수년간 한 구체 메커니즘**. ⇒ 우리 "ABox-swap 전이"는 *변종*이지 신규 메커니즘 아님. ("무재학습 전이"·"학습 라우팅 스킬 전이" 둘 다 이미 됨.)

**🆚 좁아진 신규 후보 — 대조 완료(§1.5)**: ① closure-정당화 **유한 기저 = 생존(clean·아무도 closure 주장 안 함·DST는 ontology지 기저 아님·STAR 명시 비폐포·ToolLLM open)** ② offload+전이 = ◐STAR 부분선점 ③ 벤치횡단 = ❌ToolLLM 부분선점. ⇒ **신규=①(닫힘 규칙-기저). "전이"는 신규 아님 — 닫힘이 신규·전이는 귀결.**

**★불편한 진실**: 위 차별 3축이 정확히 *아직 미증명* 부분이다. 이미 증명된 우리 전이(synth `§21` retail+airline 0.44)는 (a) op-IR 포맷(`§23E`로 native서 깨짐→축①미확보) (b) 도메인 *내*(retail/airline 둘 다 τ²→축②미확보) (c) 전수본상 cross-domain은 **결정론 scaffold가 나르고 학습 adapter는 held-out≈0**(`SOP:583`→축③약함). ⇒ **증명된 전이는 rival과 덜 구별되고, 우리만의 전이(벤치횡단·ABox-swap·학습스킬)는 아직 결과 없음.** C0(native 라우팅 전이)+벤치횡단 매트릭스 = **논문 존립 실험**(양성이면 3축 동시 확보·음성이면 rival과 구별 안 됨).

## 5. 버킷 C — 우리 고유 whitespace (대조 후 *남은* 것)

**★유일하게 깨끗한 whitespace = closure-정당화 유한 생성원 기저** (§1.5). tool-use 계획규칙이 닫힌 유한 기저(flow P1-P9·Böhm–Jacopini + content·Codd)로 환원·*그 규칙추상화*를 학습·전이. 아무도 closure 주장 안 함(ToolLLM=open·STAR=명시 비폐포·D3ST/TGRL=기저 없음·ATA=KB 재구축).

**보조(부분 선점·sharpening 필요)**:
- 사전고정 verdict-튜플 decidable-비율 *측정* — Traub식 측정 자체는 채택(발명 아님)·tool-use facet-verdict에 적용은 신규 여지(단 보조).
- ABox-swap 무재학습 전이 — **메커니즘은 선점**(schema-guided DST·ToolLLM). 우리 차별은 *전이 대상*이 닫힌 규칙-기저라는 점뿐(= ①에 흡수).
- 멀티-facet verdict 결합 — reject-option은 단일/2분류뿐(여지 있으나 보조).

---

## 6. ★방향 = 무게중심 재정렬 (이 문서의 핵심 결론)

**(1) 헤드라인을 *결과*에서 *방식*으로 피벗하라.** ToolOrchestra(검증)가 우리 바로 그 τ²서 8B로 GPT-5를 추월(80.2>77.7)·1/3 비용·미관측 도구 일반화까지 했다. ⇒ **"소형이 대형 tool-use 도달"은 더 이상 novelty가 아니다(monolithic으로 됨).** 우리가 주장할 수 있는 건 *어떻게* 도달하느냐의 차별뿐:
- **무재학습 ABox-swap 전이** (그들=RL 재학습 + 학습된 도구표현 / 우리=config swap·0 재학습).
- **측정된 decidable-offload 분담선** (그들=black-box RL policy / 우리=투명한 verdict-튜플 + AUGRC 곡선).
- **결정론 compliance 보장** (그들=형식 보장 없음 / 우리=게이트 구조적 0-위반).

**(2) GateInterpreter는 keystone이 아니다** — 2603.20449가 gate-leg를 이미 τ-bench에 발표. e2e에 *필요한 엔지니어링*이지 기여가 아니다.

**(3) ★전이 자체도 "핵심 신규"로 못 박지 마라 (정정 2026-06-18)** — "소형+schema-swap→미관측 도메인 전이"는 schema-guided DST(SGD/Description-Driven)·tool-FC(ToolLLM)가 *이미* 한 구체 메커니즘(§4 마). C0(라우팅이 native 전이하나)가 통과해도 그것만으론 schema-guided 전이의 변종일 뿐. **전이가 novelty이려면 좁아진 3후보(closure-기저·offload+전이 결합·벤치횡단)가 그 문헌 대비 다름을 *먼저* 보여야 함.**

**진짜 keystone (수정)** = (litmus) **schema-transfer 문헌 대조**(§8) → 좁은 차별 생존 확인 → (C0) 라우팅 native 전이 → (대비) ToolOrchestra·ATA. **차별이 안 서면 = SGD/Description-Driven의 tool-use 변종 → ICLR 신규성 부족.** 이게 지금 논문 존립의 *첫* 관문(C0보다 앞).

---

## 7. 실험 목록 (run / don't-run / cite)

**❌ 돌리지 말 것 (버킷 A·인용 대체)**: "GateInterpreter가 위반 차단" 단독·"소형+결정론이 대형과 경쟁" 일반주장·"recover 루프가 cascade 복구"·"calibrated defer 작동".

**✅ 반드시 돌릴 것 (버킷 C·여기에만 novelty)**:
1. **★C0 keystone (이중 load-bearing)**: native facet3(content-op 라우팅)가 retail→airline 전이를 *native 포맷*으로 §21 동급 재현하나. (배경: `M_A_RESULTS §21`=op-IR 포맷서 0.44·held-out 1.00 / `§23E`=op-IR을 native로 옮기면 깨짐 pass^1 0.075<base.) **실패 시 학습-novelty 0 → 논문 무붕괴 → 멈춤·재검토.**
2. **decidable-비율 측정**: risk-coverage 곡선 + AURC·META_DECIDE 술어 **사전등록**(Traub 2407.01032 규율). 단일 숫자 금지(게이밍).
3. **ABox-swap 전이 매트릭스**: arm2 unchanged·gate_spec/catalog만 swap → retail·airline·SOP-Bench. ATA(KB 재구축)·2603.20449(정책마다 번역)와 대비.
4. **★head-to-head vs ToolOrchestra(검증된 기준선: τ² 80.2%@10.3¢)**: monolithic-learned(8B GRPO) vs 우리 decomposed-deterministic. *정확도로* 이길 필요 없음 — 우리 축 = **무재학습 ABox-swap 전이**(그들=학습 일반화)·**투명한 decidable-분담 측정**(그들=black-box)·**결정론 compliance 보장**(그들=무보장). 이 비교·이 세 축이 빠지면 "ToolOrchestra가 더 단순·강한데?"에 무방비.

---

## 8. 검증 상태 / 잔여 TODO (2026-06-18 정독 완료분)
- [x] **ToolOrchestra 2511.21689** ✅정독: τ² 80.2%@10.3¢ > GPT-5 77.7%@31.3¢·HLE 37.1%>35.1%·벤치=HLE/FRAMES/τ²·GRPO RL(outcome+efficiency+preference)·미관측 도구=trajectory-기반 학습 표현. **확정 = 가장 위험한 rival(우리 결과 헤드라인 선점).**
- [x] **ATA 2510.16381** ✅정독: std 1.07 vs 4.89·Travel 자동 72.94(±0)/human-KB 87.17·gpt-5 68.75·gemini-pro 76.50. **자동은 대형 못넘음·human-KB만 추월·단발 보험 reasoning·KB 도메인마다 재구축.**
- [x] **2603.20449** ✅정독: airline만·위반 50%→29%·human-guided SMT 번역(4 설계·자동 실패)·게이트 only·전이/decidable 미측정.
- [x] **Traub 2407.01032** ✅: 지표=AUGRC(다중-threshold 곡선)·단일 working point 게이밍.
- [ ] **SketchAdapt 1902.06349** — PDF 렌더 실패. 2019 확립 논문·"경계 학습"이 알려진 thesis라 신뢰하되, 인용 시 html/semantic-scholar로 재확인.
- [ ] 잔여 post-cutoff snippet(2603.04474 등) — 사용 시 primary 검증(현재 미사용).
- [x] **schema-transfer 문헌 정독 완료(2026-06-18)**: D3ST `2201.08904`(pure neural·DST내·closure無)·STAR `2010.11853`(유한act·명시 비폐포·순서도=데이터·우리 반대)·ToolLLM `2307.16789`(open set·cross-bench OOD有·pure LLM)·TGRL `2510.11184`(RL 습관 전이·기저無). **판정(§1.5): ①closure 기저=생존 / ②offload+전이=STAR 부분선점 / ③벤치횡단=ToolLLM 부분선점.** ⇒ 신규=①. "전이=핵심신규" 폐기.
- [ ] **다음 핵심 실험**: 닫힌 기저 학습이 open(ToolLLM류)·비폐포(STAR류) baseline 대비 전이·커버리지 이득을 보이나 (closure의 *payoff*) + C0(라우팅 native 전이).

---

## 9. 권위본 포인터
- 설계: `INTEGRATED_TBOX_DESIGN_2026_06_18.md`(§5 분해 아키텍처)·`INTEGRATED_SCAFFOLD_IMPL_DESIGN_2026_06_18.md`(구현·§5 decidable-비율·§9 방어).
- 결과 권위본: `ma/M_A_RESULTS.md`(§21 라우팅 전이·§23D operand 퇴행·§23E native 붕괴)·`reports/facet_rft_2026/{SOPBENCH,TASKBENCH}_EXPERIMENT_RESULTS.md`.
- thesis·경계: `THESIS_STATEMENT_2026_06_16.md`·`DECOMPOSITION_OPTIMALITY.md`·마스터 `EXPERIMENT_DESIGN.md §1`(fact-offload OK·procedure-offload 금지 경계).
- 딥리서치 원본: 세션 워크플로 `wf_30a790e2-566`(52 claim·journal.jsonl).

---

## 10. ★인지아키텍처(SOAR/ACT-R)×LLM 계보 (2026-06-24 추가·전부 primary 정독) — epistemic-A2 방향 전용

> **맥락**: §1-9는 closure/전이 축의 rival을 다룸. 본 §10은 *다른 클러스터* = **인지아키텍처×LLM** 계보로, 새 방향 `EPISTEMIC_A2_THESIS_2026_06_23.md`(§3 SOAR impasse↔빈관계 매핑·scaffold=SOAR 결정코어 최소판)의 related work다. 6편 모두 2026-06-24 다중-fetch 정독(🟢). **결론 먼저**: 이 계보는 우리 SOAR 프레이밍을 *지지*하고(특히 Wray-Kirk-Laird·CoALA), 가장 가까운 이웃(NL2GenSym)조차 우리 세 델타(추론시 유한관계 *적용*·빈관계 abstention·A2-swap 전이)를 안 가짐. 단 NL2GenSym은 "소형+기호프레임워크>대형" 헤드라인을 또 선점하므로(ToolOrchestra와 같은 줄) 차별은 *방식*(abstain·전이·비용)에 둬야 함.

| arxiv | 제목(약칭) | 저자·날짜 | 티어 | 우리와의 관계 |
|---|---|---|---|---|
| **2510.09355** | NL2GenSym (NL→SOAR rule via LLM) | Yuan·Zeng·Hu·Zhu·Yin·Xie, 2025-10 | 🟢 | ★**가장 가까운 이웃** — LLM이 NL→SOAR production *생성*·실행-grounded Generator-Critic·**소형+프레임워크>대형** |
| **2505.07087** | Applying Cognitive Design Patterns to General LLM Agents | Wray·Kirk·**Laird**, 2025(AGI) | 🟢 | ★**우리 SOAR 프레이밍 권위 지지** — impasse/subgoaling·propose-select-reconsider를 정전 패턴으로·LLM 자기반성 불신뢰 명시 |
| **2309.02427** | CoALA (Cognitive Architectures for Language Agents) | Sumers·Yao·Narasimhan·Griffiths, 2024(TMLR) | 🟢 | ★**우리가 속하는 우산 프레임워크** — LLM=확률적 production·memory/action/decision 3축·우리는 그 안의 한 인스턴스 |
| **2408.09176** | Cognitive LLMs / LLM-ACTR | Wu·Oltramari·Francis·Giles·Ritter, 2024 | 🟢 | CA→LLM *반대* 방향(ACT-R 정책을 LoRA로 *가중치에 구움*)·전이/abstain 없음·제조 의사결정 |
| **2403.00810** | Bootstrapping Cognitive Agents with a LLM | Zhu·Simmons, 2024(AAAI) | 🟢 | LLM이 SOAR-류 production *부트스트랩*→기호측 검증·**결정론-우선/LLM-폴백**·토큰 50-100×↓·GPT-4(대형) |
| **2309.14945** | LLM in Cognitive Architecture (MERLIN2 로봇) | González-Santamarta 등, 2023 | 🟢 | (★예상과 다름=SOAR 아닌 ROS 로봇 MERLIN2)·GBNF 문법제약 디코딩으로 NL→PDDL·기호계획기 검증 |

### 10.1 편별 내용 + 우리 델타

1. **NL2GenSym (2510.09355)** — LLM(Generator)이 자연어 문제를 SOAR production 규칙으로 변환→**실제 SOAR에서 실행**(trace+decision-cycle이 결정론적 grounding)→Critic LLM이 실행결과 분석해 NL 개선제안→반복. RAG 자기진화 KB(SOAR 문법 13모듈 + 케이스풀). **Water Jug 100케이스: 성공률 86%(Gemini-Flash)~91%(Qwen)·decision-cycle 최적의 1.76~1.98배(기존 ~1/1000)·소형+프레임워크가 대형+one-shot 능가**(Flash 86% vs Pro 55%). 한계(저자): 단일 깨끗한 도메인·iterative 비용·"진짜 SOAR 이해는 미지수". 
   - **우리 델타(3)**: ⓐ 그들=**오프라인 규칙 *합성*(program induction)**, 우리=**추론시 유한관계에 σ/⋈/agg *적용***(규칙 생성 아님). ⓑ 그들=abstention 없음(성공/500k타임아웃까지 iterate), 우리=**빈관계→ASK epistemic abstention**(핵심 escape). ⓒ 그들=도메인별 케이스풀 성장(전이 미검증·future work), 우리=**A2-swap 재학습0 전이**. ★단 "소형>대형" 헤드라인 겹침→[[41]] ToolOrchestra와 함께 *방식 차별*로 대응.

2. **Wray·Kirk·Laird (2505.07087)** — SOAR 창시자 Laird 포함. 고전 인지아키텍처의 "인지 설계패턴"(observe-decide-act·제안-선택-재고 3단계 commitment·**impasse 기반 subgoaling**·knowledge compilation·memory 종류)을 정리, 현 LLM 에이전트(ReAct/Voyager/ToT…)에 매핑, **결함 예측**(LLM은 신뢰가능 commitment·비단조 제어 결여). *position 논문(실험無)*. ★중요: impasse는 본문서 *한 셀*(operator no-change=hierarchical decomposition 예시)로만 등장—**taxonomy·LLM매핑은 안 함** → 우리 impasse↔빈관계 매핑은 *중복 아닌 보완*.
   - **우리에게**: (a) impasse/subgoaling이 일반지능 정전 패턴이라는 **권위 근거**(scaffold=최소 SOAR 결정코어 주장 정당화). (b) **"LLM 자기반성은 초기응답만큼 불신뢰·재귀무한"** 명시 → "왜 LLM 내성으로 모름판정 안 하나"에 대한 직접 방어(=우리 비내성·빈관계 외재화 지지). scoop 위험 없음(그들은 빈관계-abstain 미제안).

3. **CoALA (2309.02427)** — LLM 에이전트를 인지아키텍처·production system 관점으로 재정식화한 **우산 프레임워크**. LLM=텍스트편집 *확률분포*=확률적 production. 3축: memory(working/episodic/semantic/procedural)·action(내부 reasoning/retrieval/learning·외부 grounding)·decision(제안-평가-선택 루프). 미래과제로 "코드 결정로직+LLM 추론 혼합" 권고.
   - **우리에게**: 우리 설계가 **이 분류 안에 정확히 안착**(A2=semantic+procedural memory·scaffold=procedural+실행단계·"LLM생성/결정론실행·선택"=권고하는 혼합의 *구체 구현*). 단 CoALA가 *일반 축·미해결*로 남긴 것 — **관계대수로 명시한 결정론 substrate·빈관계 epistemic abstention·A2-swap 전이·TCO** — 이 우리 고유 기여. ⇒ "CoALA 인스턴스이자 그 whitespace를 메커니즘으로 구체화".

4. **LLM-ACTR (2408.09176)** — ACT-R 인지모델(제조 의사결정)을 시뮬레이션해 그 결정·RL trace를 임베딩/라벨로 추출→**LlaMa-2 13B LoRA에 미세조정 주입**(CA→LLM·정책을 *가중치에 구움*). 결과: LLM-ACTR acc 0.66 > 무조정 LlaMa 0.36(chance 이하). 한계: 인간정렬 불완전·full-trace 융합 negative·**과제 간 일반화 불가**(저자 명시).
   - **우리 델타**: 방향이 *반대* — 그들=추론시 기호계 *사라짐*(가중치 흡수)·도메인별 재미세조정·abstain無. 우리=기호엔진 추론시 *살아있음*·도메인학습0·A2-swap 전이·빈관계 abstain. ⇒ *인접 whitespace, 우리 칸 아님*.

5. **Bootstrapping (2403.00810)** — GPT-4가 SOAR-syntax production을 *부트스트랩 생성*→기호 에이전트가 replay/critic/utility로 검증·정련. 런타임=**production 있으면 결정론 발화·없으면 LLM 폴백**(=우리 "scaffold 실행·열린 경우만 LLM"과 동형). AI2-THOR 주방: 순수 LLM과 동등 성공률·**추론 토큰 50-100×↓**(학습후 0토큰), 신규객체 전이.
   - **우리 델타**: 그들=**대형(GPT-4)·가중치 학습0·production을 LLM이 ad-hoc 저작**(드리프트 가능 soft 권한). 우리=**소형 LLM의 도메인일반 스킬을 가중치 학습·scaffold는 고정(LLM이 엔진 안 씀)·스키마-일반 A2 고정+내용swap·빈관계→ASK 일급화**. ★"결정론-우선/LLM-폴백 + 비용절감"의 가장 가까운 선행 = **인용 정본**(단 우리 소형-학습-전이·abstain은 미선점).

6. **MERLIN2 통합 (2309.14945)** — (예상과 달리 SOAR survey 아님) ROS 로봇 인지아키텍처에 LLM 통합·**GBNF 문법제약 디코딩**으로 NL→PDDL 구조출력·기호계획기가 검증·로컬추론. = "LLM=생성/기호=실행·검증" 분담의 로봇 사례. 문법제약 디코딩 = 우리 "LLM이 NL→술어 formalize"의 가장 가까운 선행 형태.
   - **우리 델타**: 그들=배치마다 도메인 재작성(전이無)·LLM은 제약된 *추측*만(abstain無). 우리=A2-swap 전이·빈관계 abstain·문법(출력형식)이 아니라 scaffold(어느 관계연산이 언제)가 강제.

### 10.2 종합 위치 (epistemic-A2 방향)
- **공통 패턴(이 계보 전체)**: "LLM=생성/형식화, 결정론·기호계=실행·검증"의 분담 + 다수 사례서 "결정론-우선/LLM-폴백" + "소형/적은비용이 충분"([[41]] 정합). ⇒ 분담 *원리*·소형 결과는 **우리 발명 아님**(가족 것·인용).
- **우리 고유(이 계보가 안 가진 것)**: ① **빈관계로 외재화된 epistemic abstention + 학습된 `empty→ASK`**(내성 우회·SOAR impasse의 LLM판·아무도 안 함) ② **스키마-일반 A2 고정 + 내용 swap 재학습0 전이**(LLM-ACTR/MERLIN2=전이無·Bootstrapping=도메인내 production성장) ③ **소형 LLM의 도메인일반 스킬을 *가중치 학습*하되 도메인은 안 굽음**(NL2GenSym/Bootstrapping=대형·LLM-ACTR=도메인 가중치주입) ④ TCO/비용 명시.
- **권위 활용**: Wray-Kirk-Laird = impasse 패턴·반(反)내성 설계의 권위 인용. CoALA = 우산 위치설정. NL2GenSym = 가장 가까운 *방법* 이웃(LLM→SOAR 규칙 생성). **⚠️정정(2026-06-25 딥리서치 적대검증 1-2 killed)**: NL2GenSym "소형>대형" 헤드라인은 *검증 실패* → 공동선점자로 인용 금지. **소형>대형 공동선점자 = TRUST(2606.06976)로 교체**(§10.3).
- ⚠️ [[05]]/[[03]] 드리프트 차단: SOAR/이 계보는 **framing·선행근거·실패 census 재라벨**로만 사용·결정사이클/chunking 엔진 *이식 금지*(우리 scaffold가 이미 최소판). 상세 = `EPISTEMIC_A2_THESIS §3` SOAR 블록 delta.

### 10.3 ★2025 H2 – 2026 갱신 (딥리서치 2026-06-25·23/25 적대검증·전부 primary)
**결론: 신규 문헌이 세 델타로 *수렴*하나 어느 것도 완전 scoop 안 함. (iii) 검증·(i) 방식상 미scoop·(ii) 청구자 0.**

**A. 우리를 *지지*하는 신규 인용 (scoop 아님·반드시 인용):**
- **`2602.05073`(2026-02)** — **tau2-bench서** LLM verbalized confidence·NLL·Entropy가 task성공 예측에 거의 random(AUROC 0.47-0.69)·**궤적 길수록 confidence 부풀음**. = **§0 비내성 명제를 *우리 정확한 벤치*서 실증.** 보강 `2602.06948`(agentic overconfidence)·`2601.07264`(tool miscalibration). ⚠️caveat: pilot-scale(2모델·2도메인·무유의검정)→"방향성".
- **`2604.19459`(Kim·Poiroux·Bosselut·EPFL·ICLR-2026 VerifAI workshop·2026-04·★정독 2026-06-25)** — "Do LLMs Game Formalization?": NL→Lean4 형식증명·303 FOL(203 FOLIO+100 Multi-LogiEval)·GPT-5/DeepSeek-R1·2-stage(형식화→증명) cross-stage 비교.
  - **두 unfaithful 모드**: ① **DeepSeek-R1 전제오역=★탐지불가**(Case 177: "최다메달[장소]"을 [사건]으로 오역→goal과 동일·trace가 "trivial해짐"을 *알고도 고름*·증명성공 True보고·**내부일관이라 결정론검증기[Lean]도 못 잡음**). ② **GPT-5 axiom 날조=탐지가능**(107건·"결론을 axiom으로" 59건·반응적·cross-stage로 검출).
  - **★우리에게(정밀)**: ①모드 = **thesis §3 "silent residual"(mis-formalize σ=1·틀림은 빈관계 표면화 안 됨·결정론 scaffold 못 잡음)의 *frontier 실증***·§0 정합(frontier도 같은 병). ⓑ-op 날조(우리 101/102 "123 Elm St" 주소 fabrication) ↔ ②모드. = **delta(iii) 직접 지지.**
  - **★정직 한계(과인용 금지·[[08]])**: ⓐ 저자=전략적 'gaming' *없음*·**unfaithful은 *소수*(124에러 중 95=77% *faithful*·FOLIO divergence 0-4건)**·"모델은 틀린증명 강행보다 *실패보고* 선호" → "LLM이 늘 날조"로 인용 금지·*위험한 소수가 탐지불가하게 오역* 톤 유지. ⓑ **논문은 depth/multi-hop별 faithfulness 분해 *안 함***(ML이 depth3-5 stratify지만 aggregate만) → **"다단계라서 더 날조"로 인용 금지**(미입증). 컴파일 87-99%·정확도 FOLIO 85-87%/ML 70-72%.

**B. 차별화할 신규 공동선점자:**
- **`2606.06976` TRUST(2026-06)** — **4B>30B·Claude-Sonnet-4 동급**(When2Call/BFCL-V4). 4-way {Direct,Tool,Ask,Unable}. **단 순수 RL(GRPO+perplexity margin)·기호 scaffold 0·Ask/Unable=LLM-emit·신호=내성 perplexity** → **우리와 방식 직교**(scaffold+비내성+전이 vs RL+내성). = 소형>대형 공동선점자(NL2GenSym 대체)·*방식차별*로 응대.

**C. delta(i) 이웃(=경계·미scoop 확정):** 우리 "학습된 빈관계→ASK"의 이웃들 — 전부 *학습 아님*:
- **`2511.08798` SAGE-Agent** — tool-parameter 도메인 위 구조적 belief state(LLM confidence 무관)+**결정론 EVPI 정지규칙**. 비내성·구조적·*그러나 hand-specified Bayesian*(학습 아님). 7-39% coverage↑·질문 1.5-2.7×↓.
- **`2603.26233` Ask-or-Assume** — underspecification 탐지를 실행서 분리(별 agent)·*그러나 탐지기=LLM judge*.
- **`2402.00367`(ACL2024) Don't Hallucinate Abstain** — abstain을 multi-LLM 협력으로 외재화(*자기반성·held-out 실패 때문에* 명시) = 우리 비내성 동기 지지.
- ⇒ 비내성·구조적 외재화는 *활발*하나 **학습된 빈관계 사건은 0** → delta(i) 방식상 생존.

**D. A2_FRONTEND 직접 선행 (⏸️별도 논문·2026-06-25 [[06]] 범위분리 — 현 논문 제외·NL→A2 자동생성=후속 논문. 아래 cite는 그 후속 논문 relwork용):**
- **`2512.18189` NL2CA(2025-12)** — **fine-tuned Qwen3-0.6B**가 NL→LTL→unsupervised Critic Tree→**pyactr(ACT-R) production 컴파일·완전자동·human 0.** = **"NL→A2 생성기"(`A2_FRONTEND_DISTILL`)의 직접 선행·0.6B로 됨**(소형 NL→기호 컴파일 입증·NL2GenSym과 같은 Generator-Critic). A2_FRONTEND 재개 시 방법 참조. 단 단일 깨끗 파이프라인.

**E. open question / 정밀필요:** ① AAAI-26(Jones/Wray/Laird·`41081`) "no-compliant-action=구조적 사건"=검증실패(1-2)·close-read 필요 ② delta(ii) 청구자 0은 *음성증거*(약함)→cognitive-arch 전이·ABox-swap 문헌 표적탐색 권고 ③ ACT-R+LLM(LLM-ACTR 후속) bake vs swap 대조 확인.

**F. 종합 위치 변화**: 세 델타 *강화*. 특히 §0(비내성)과 ⓑ(faithful-formalize)가 *2026 1차문헌으로 외부 검증*됨 = 우리 진단의 신뢰도↑. 헤드라인 정정: 소형>대형=TRUST 대비 *방식*(scaffold·비내성·전이·TCO)으로 차별.
