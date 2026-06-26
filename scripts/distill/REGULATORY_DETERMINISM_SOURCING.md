# REGULATORY_DETERMINISM_SOURCING — 규제 1차원문 판정: "결정론 요구"인가 "로깅/검증가능성 요구"인가
> 2026-06-10. §18.2 항목4 (zero-GPU, load-bearing) 수행 결과. 발주 = `FIELD_GAP_LLM_VALUE_DESIGN.md` §15.4 open-1 ("§15.3 moat의 사활").
> 지시문 자체가 적대적 프레이밍("로깅이면 충분"이면 그렇게 말하라)이었고, **결과는 그쪽이다.**

---

## §0 판정 (헤드라인)

**판정 = (c) "로깅+검증+감독이면 충족" — 주요 프레임워크는 결정론을 명시 요구하지 않는다.**

- 조사 범위(EU AI Act·SR 11-7/SR 26-2·FDA SaMD/GMLP/PCCP·EU MDR·GDPR Art.22·MiFID II RTS 6) 중 ***결정론/재현가능*을 명시한 1차원문은 EU MDR Annex I §17.1 단 1건**("repeatability" — 의료기기 한정, "in line with their intended use" 한정어 부착).
- EU AI Act는 **전문(144pp OJ, recital 포함) 기준 "deterministic"=0회, "reproducib\*"=0회, "repeatab\*"=0회** (공식 authentic text 전수 grep, 3-vote 검증 [V2]). 어휘는 전부 traceability / logging / "appropriate level of accuracy" / oversight.
- SR 11-7은 애초에 **통계적·불확실한 모델을 전제하고 관리하는 체제**다(모델 정의 자체가 "statistical... estimates"). 결정론 요구가 있을 수가 없는 구조.
- **⇒ §15.3 moat의 "규제가 결정론 요구" leg = 철회 확정. "검증가능성(auditability/verifiability)" leg로 후퇴한다** (§4 수정안). 이는 §17.9 리뷰6-1이 이미 수행한 강등("moat = soundness의 *검증가능성*")을 1차원문으로 *확정*하는 것이며, thesis 본문 변경은 후퇴 방향으로 단조(추가 붕괴 없음).

**단, 후퇴는 전면 항복이 아니다 — textual하게 생존하는 3개 footholds:**
1. **★SR 26-2 정의-비대칭 (이번 sourcing 최대 수확, 2026-04-17 발효)**: 미 은행 모델리스크 지침의 "model" 정의가 **결정론 rule-based 프로세스를 명시적으로 제외** → 결정론 게이트는 MRM(검증·effective challenge·문서화·재검증 주기) 부담 *체제 밖*. 동시에 **생성형·에이전틱 AI는 지침 scope 밖**(=정착된 준수 경로 부재). 즉 "결정론이라서 요구를 충족"이 아니라 **"결정론이라서 요구 자체가 면제"** — 준수비용 비대칭으로서의 moat.
2. **EU MDR Annex I §17.1** — 유일한 명시적 "repeatability" 조문 (단 의료기기 도메인 한정·qualified).
3. **de-facto (b) 주장**: AI Act Art.14 감독("correctly interpret"), Annex IV("general logic of the AI system") 문서화, SR의 validation/effective challenge — 전부 *결정론 시스템이 구조적으로 싸고 강하게 충족*. stochastic 정책의 "effective policy"는 정적 검증 불가, 표본 통계로만 사후 입증 가능. — 이것이 후퇴 후 헤드라인이다.

---

## §1 방법 · 출처 신뢰도 표기

- **[V1]** = 이 세션에서 1차원문을 직접 확보·추출·키워드 전수 카운트한 것 (SR 11-7/SR 26-2는 공식 PDF 로컬 추출, EU AI Act는 조문 미러 fetch, GMLP는 MHRA 공동게재 정부원문, GDPR 원문 미러).
- **[V2]** = 딥리서치 워크플로(40 agents)에서 claim별 3-vote 적대검증을 통과한 것. 워크플로는 중단(killed)됐으나 journal(`wf_76d7bdb4-7b2`)에 verify 결과가 박제되어 회수함. 특히 EU AI Act 인용들은 EUR-Lex/EU Publications Office **authentic text 대조 word-for-word 확증**.
- **[S]** = 검색/스니펫 경유(원문 전문 미확보). 논문 인용 전 원문 재확인 요.

---

## §2 프레임워크별 조문·판정

판정 분류(발주 정의): **(a)** 결정론/재현가능의 명시적 텍스트 요구 / **(b)** de-facto 실무 요구(검증 체제가 stochastic에 비현실적) / **(c)** 로깅·감사가능성 요구 — logged stochastic으로 텍스트상 충족 가능.

### 2.1 EU AI Act (Regulation (EU) 2024/1689) — 판정 **(c)**

**Art. 12 (Record-keeping)** [V1][V2]:
> "1. High-risk AI systems shall **technically allow for the automatic recording of events (logs)** over the lifetime of the system."
> "2. In order to ensure **a level of traceability** of the functioning of a high-risk AI system **that is appropriate to the intended purpose** of the system, logging capabilities shall enable the recording of events relevant for: (a) identifying situations that may result in ... a risk ...; (b) facilitating the post-market monitoring ...; (c) monitoring the operation ..."

— 순수 로깅 의무. traceability조차 "intended purpose에 *적절한 수준*"으로 상대화. 결정로직 자체에 대한 요구 0.

**Art. 14 (Human oversight)** [V1][V2 word-for-word]:
> "1. High-risk AI systems shall be designed and developed in such a way ... that they can be **effectively overseen by natural persons** during the period in which they are in use."
> "(4)(c) to **correctly interpret** the high-risk AI system's output, taking into account, for example, the interpretation tools and methods available; (d) to decide ... not to use the high-risk AI system or to otherwise **disregard, override or reverse the output** ..."

— 요구 = 해석가능성 + 인간 override 경로. 재실행 재현성·결정론 요구 없음.

**Art. 15 (Accuracy, robustness and cybersecurity)** [V1][V2]:
> "1. High-risk AI systems shall be designed and developed in such a way that they achieve **an appropriate level of accuracy**, robustness, and cybersecurity, and that they **perform consistently in those respects** throughout their lifecycle."
> "3. The levels of accuracy and the relevant accuracy metrics ... shall be **declared in the accompanying instructions of use**."

— "perform consistently"는 *수준의 시계열 일관성*이지 per-input 결정론이 아님. 결정적 반증: **Art. 15(4)는 "고위험 AI 시스템 that continue to learn after being placed on the market"를 명시 규율**(편향 피드백루프 완화 요구) [V2] — 즉 이 법은 **적응형/비결정 시스템을 전제로 쓰였다.**

**Annex IV (Technical documentation)** [V1]:
> "the **general logic of the AI system** and of the algorithms; the key design choices ...; **what the system is designed to optimise for**, and the relevance of the different parameters"
> "validation and testing procedures used, including ... **metrics used to measure accuracy, robustness**"

— "general logic" *서술* 의무이지 logic의 결정론 의무 아님. LLM 시스템도 설계 서술로 텍스트상 충족 가능.

**전문 어휘 전수** [V2, 공식 OJ authentic text grep]: deterministic=0 · reproducible/reproducibility=0 (유일 "reproduc\*" hit는 저작권 맥락 "reproductions and extractions of works") · repeatable/repeatability=0 · traceability≈7(전부 식별/감사 맥락).

**logged-stochastic 충족?** **예 (텍스트상).** 로깅+선언된 정확도 수준+감독 인터페이스를 갖춘 stochastic LLM은 Art.12/13/14/15를 문언상 충족 가능.

### 2.2 US 은행 — SR 11-7 (2011) 및 ★SR 26-2 (2026, 대체) — 판정 **(c) + 구조적 비대칭**

**SR 11-7 원문** [V1, 공식 PDF 21pp 추출; "reproduc/replicat/repeatab"=0회, "determin\*"은 전부 일반동사 "determine"]:

모델 정의 (p.3):
> "the term model refers to a quantitative method, system, or approach that **applies statistical, economic, financial, or mathematical theories, techniques, and assumptions to process input data into quantitative estimates**."

— 정의 자체가 통계적 추정. 한계·불확실성은 결격사유가 아니라 *관리 대상*:
> "Model risk increases with greater model complexity, **higher uncertainty about inputs and assumptions**, broader use, and larger potential impact."

요구하는 것 = 검증·도전·문서화 [V1+V2]:
> "Model validation is the set of processes and activities intended to **verify that models are performing as expected, in line with their design objectives and business uses**." [V2]
> "A guiding principle for managing model risk is **'effective challenge'** of models, that is, critical analysis by objective, informed parties who can identify model limitations and assumptions and produce appropriate changes." [V1]
> "Documentation of model development and validation should be **sufficiently detailed so that parties unfamiliar with a model can understand how the model operates, its limitations, and its key assumptions**." [V1]

**★SR 26-2 (2026-04-17, Fed/OCC/FDIC 공동 — SR 11-7·SR 21-8 "supersedes and replaces")** [V1, 공식 PDF 14pp 추출]:

모델 정의의 명시 제외 (이번 sourcing 최대 수확):
> "The term 'model' in this guidance **excludes** simple arithmetic calculations, such as those found within spreadsheets, **as well as deterministic rule-based processes and software** where there are no statistical, economic, or financial theories underpinning their design or use."

생성형·에이전틱 AI의 지위 (각주 3):
> "**Generative AI and agentic AI models are novel and rapidly evolving. As such, they are not within the scope of this guidance.** Nonetheless, a banking organization's risk management and governance practices should guide the determination of appropriate governance and controls for any tools, processes, or systems not covered in this document. However, the principles described in this guidance apply to traditional statistical and quantitative models and **non-generative, non-agentic AI models**."

**해석 (조심스럽게)**:
- (i) 결정론 요구는 여기에도 **없다** — 체제는 여전히 validation/monitoring/documentation. 판정 (c) 유지.
- (ii) 그러나 **결정론 rule-based 프로세스는 '모델' 정의에서 제외** = MRM 장치(독립 검증·effective challenge·재검증 주기·모델 인벤토리) 의무의 *적용 대상이 아님*. **stochastic 모델은 안에, 결정론 게이트는 밖에** — "규제가 결정론을 요구"가 아니라 **"결정론은 규제 부담을 면제받는다"**는 비대칭. 이것이 우리가 실제로 주장할 수 있는 *정확한* 형태다.
- (iii) 생성형·에이전틱 AI는 **승인된 준수 경로 자체가 미정착**("not within the scope") — 은행이 LLM-agent 자기집행을 준수근거로 쓰려면 정착된 지침 없이 자체 거버넌스로 방어해야 함 = 보수적 기관의 채택장벽. 단 이것은 *불법*이 아니라 *불확실성*이므로 과대주장 금지.
- ⚠️ 주의: 우리 시스템의 NL→구조 front-end(학습 모델)는 (ii)의 면제 대상이 아닐 수 있음(통계적 학습물). 정직한 구도 = "**집행 게이트는 MRM-면제 결정론, front-end는 통상 모델 검증 대상**" — 검증 표면적을 게이트 *생성* 단계로 국소화한다는 주장으로 연결.

**logged-stochastic 충족?** **예** — SR 11-7/26-2 모두 통계 모델 전제 체제. 단 생성형·에이전틱은 SR 26-2 기준 *정착 경로 부재*.

### 2.3 의료 — EU MDR · FDA(SaMD/GMLP/PCCP) — 판정 **MDR=(a)-qualified, FDA=(b), GMLP=(c)**

**EU MDR 2017/745 Annex I §17.1** [V1, 검색결과 원문 인용 — 논문 인용 전 EUR-Lex 재확인 권장 [S]→[V1] 승급 필요]:
> "Devices that incorporate electronic programmable systems, including software, or software that are devices in themselves, shall be designed to ensure **repeatability, reliability and performance in line with their intended use**."

— **조사 전체에서 유일한 명시적 "repeatability" 조문.** 단 (i) 의료기기 한정, (ii) "in line with their intended use" 한정어로 notified body 해석 여지(연속학습 기기의 적합성 평가 논쟁이 실제로 이 조문 중심으로 진행 중). 분류: **(a)-qualified**.

**FDA "locked algorithm"** [S, 2019 AI/ML SaMD discussion paper — *nonbinding*]:
> a "locked" algorithm = one that "**provides the same result each time the same input is applied to it and does not change with use**"

— 전통 패러다임이 locked를 전제했다는 *역사적* 근거. 단 (i) discussion paper = 구속력 없음, (ii) locked/adaptive 구분의 초점은 *시간에 따른 변경*(재학습)이지 추론시 stochasticity 자체가 아님, (iii) **FDA의 궤적은 locked 요구에서 *멀어지는* 방향**: PCCP 최종지침(2024-12)은 사전명세된 변경(pre-specified, verified per change protocol)을 허용 — 단 런타임 연속학습은 여전히 미지원 [V2 스니펫·S]. 분류: **(b)** — 명시 요구는 아니나 검증 체제가 비-locked에 실질 부담.

**GMLP 10원칙 (FDA/Health Canada/MHRA 공동, MHRA 게재본)** [V1]:
- 10원칙 전문에서 **"deterministic"/"reproducible"/"repeatable" 0회.**
- 어휘 = "robustness and generalisability"(원칙5), "Deployed models have the capability to be **monitored in 'real world' use**"(원칙10), human-AI team(원칙7). 분류: **(c)**.

**logged-stochastic 충족?** GMLP·PCCP 체제상 **예**(사전명세+모니터링 갖추면). MDR §17.1은 "intended use에 부합하는 repeatability"의 입증 부담이 stochastic에 실질적으로 큼 — 의료만 (a) 인접.

### 2.4 기타 — GDPR Art.22 · MiFID II RTS 6 — 판정 **(c) / (b)-light**

**GDPR Art. 22** [V1]:
> "1. The data subject shall have the right not to be subject to a decision **based solely on automated processing** ... 3. ... the data controller shall implement suitable measures to safeguard ... **at least the right to obtain human intervention** ..., to express his or her point of view and **to contest the decision**."

— 안전장치 = 인간개입·이의제기. "deterministic"/"reproducible" 0회. 분류: **(c)**. (참고: contest권의 실효성 논변은 "결정 근거의 사후 설명가능성"을 요구하는 방향으로 CJEU가 해석 중 — 검증가능성 논거에 보조적으로 유리하나 결정론 요구는 아님 [S].)

**MiFID II RTS 6 (Commission Delegated Regulation (EU) 2017/589)** [V2 스니펫 — 조항 번호·문구는 논문 인용 전 원문 재확인 요]:
> Art. 5: "clearly delineated methodology" for testing before deployment; Art. 6(1): conformance testing that the system "interacts with the trading venue's matching logic ... as intended"; Art. 7: non-live testing that the algorithm "**does not behave in an unintended manner**".

— "behave as intended" = 행동-경계 테스트 의무. 결정론 단어 없음. stochastic 알고리즘에 "intended 행동의 사전 정의·테스트"가 실질 부담이므로 **(b)-light**, 텍스트상은 (c).

---

## §3 분류 총괄표

| 프레임워크 | 핵심 조문 | "determin/reproduc/repeatab" 명시? | 판정 | logged-stochastic 텍스트상 충족? |
|---|---|---|---|---|
| EU AI Act | Art.12/13/14/15, Annex IV | **0회 (전문 grep, OJ authentic)** | **(c)** | 예 (Art.15(4)가 적응형 시스템 명시 규율) |
| US 은행 SR 11-7→**SR 26-2** | 전문 | 0회 (요구로서는) | **(c)** | 예 — 단 **결정론 rule-based는 '모델' 정의 제외**(부담 면제), **genAI/agentic은 scope 밖**(경로 미정착) |
| EU MDR | **Annex I §17.1** | **"repeatability" 명시 (유일)** | **(a)-qualified** | 곤란 (intended-use 한정어로 해석 여지) |
| FDA SaMD/PCCP | 2019 discussion paper·PCCP 2024 | "locked" 정의 (nonbinding) | **(b)** | 부분 (사전명세 변경은 허용, 런타임 학습 미지원) |
| FDA/HC/MHRA GMLP | 10원칙 | 0회 | (c) | 예 |
| GDPR | Art.22 | 0회 | (c) | 예 (인간개입·이의제기 갖추면) |
| MiFID II RTS 6 | Art.5–7 | 0회 | (b)-light | 대체로 예 ("intended 행동" 입증 부담) |

---

## §4 thesis 영향 — §15.3/§15.4/§17.9 반영 지시 (이 문서가 권위)

1. **§15.3 "규제가 결정론 요구" leg = 철회 (확정).** §15.4 open-1 해소. §15.3의 "compliance는 *결정론·감사가능·추적가능*을 요구" 문장에서 **"결정론·" 삭제** — 규제 어휘는 감사가능·추적가능·검증가능까지만 지지된다.
2. **후퇴 착지점 = "검증가능성(verifiability) moat" (§17.9 리뷰6-1과 동일 좌표 — 이미 강등해 둔 위치로 정확히 떨어짐, 추가 붕괴 없음)**: 규제가 요구하는 것은 traceability·validation·oversight·effective challenge이고, **결정론 게이트는 이를 by-construction으로 충족**(정책-집행 동치성을 정적 검사 가능, 어떤 입력에도 인코딩된 제약 위반 0을 *검사*로 보장), **stochastic 자기집행은 표본 통계로만 사후 충족**(effective policy 자체가 검사 불가능, per-release 재검증). 직교성 주장("capability가 못 푸는 축")은 *요구의 존재*가 아니라 *충족 방식의 비용·강도 비대칭*으로 재정식화.
3. **신규 leg 추가 (SR 26-2 정의-비대칭, textual)**: "결정론 rule-based 집행은 미 은행 MRM의 '모델' 정의에서 명시 제외 = 검증·재검증 체제 밖; 생성형·에이전틱 AI는 동 지침 scope 밖 = 정착 준수경로 부재" — 이것은 1차원문 인용 가능한 *준수비용 비대칭*. 단 ⚠️ 우리 front-end(학습 모델)는 면제 비대상 — 주장은 "검증 표면적을 게이트-생성 단계로 국소화"까지만.
4. **도메인 층화**: 결정론-텍스트가 실재하는 곳 = 의료(MDR §17.1)뿐. 은행(우리 substrate)은 (c)+비대칭, 마케팅(CDP 일반)은 규제 압력 자체가 약함 — §17.9 리뷰6-5(도메인 의존)와 정합.
5. **금지 문구 / 허용 문구 (논문·특허 공통)**:
   - ❌ "Regulations require deterministic decision logic" — **1차원문 미지지, 사용 금지.**
   - ❌ "Stochastic models cannot legally be deployed in regulated industries" — 반증됨(SR 체제 자체가 통계모델 관리 체제).
   - ✅ "Regulatory frameworks mandate traceability, validation, and effective human oversight (EU AI Act Arts. 12/14; SR 26-2) rather than determinism per se; **deterministic policy gates satisfy these obligations by construction — and in US banking guidance fall outside the model-risk perimeter entirely (SR 26-2) — whereas stochastic self-enforcement can demonstrate compliance only statistically, per sample, with the effective policy unverifiable ex ante.**"
   - ✅ (의료 한정 보조) "Where regulation does textually demand repeatability (EU MDR Annex I §17.1), deterministic execution satisfies it directly."
6. **§18.2 항목4 = DONE.** 항목5(bitter-lesson sourcing)·항목6(erosion 테스트)은 별도.

---

## §5 출처

- EU AI Act (Reg. (EU) 2024/1689): EUR-Lex `https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng` (authentic); 조문 미러 `artificialintelligenceact.eu/article/{12,14,15}/`, `/annex/4/` [V1]; 전문 grep은 EU Publications Office cellar 본 [V2].
- SR 11-7: `https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107a1.pdf` (21pp, 로컬 추출) [V1].
- **SR 26-2**: `https://www.federalreserve.gov/supervisionreg/srletters/SR2602.pdf` (2026-04-17, Fed/OCC/FDIC; SR 11-7·21-8 대체) [V1]. OCC Bulletin 2026-13 동본.
- EU MDR 2017/745 Annex I §17.1: EUR-Lex CELEX:32017R0745 [S — 인용 전 재확인].
- FDA AI/ML SaMD Discussion Paper (2019-04): `fda.gov/files/medical devices/published/US-FDA-Artificial-Intelligence-and-Machine-Learning-Discussion-Paper.pdf` [S]; PCCP 최종지침(2024-12) [S].
- GMLP 10원칙: `gov.uk/government/publications/good-machine-learning-practice-for-medical-device-development-guiding-principles` (FDA/HC/MHRA 공동) [V1].
- GDPR Art.22: `gdpr-info.eu/art-22-gdpr/` [V1-미러].
- MiFID II RTS 6 (Reg. (EU) 2017/589): EUR-Lex CELEX:32017R0589 [V2-스니펫].
- 딥리서치 journal: 세션 `wf_76d7bdb4-7b2` (40 agents, 3-vote verify; killed mid-synthesis, verify 결과 회수).
