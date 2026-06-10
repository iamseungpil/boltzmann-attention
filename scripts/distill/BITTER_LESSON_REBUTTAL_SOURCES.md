# BITTER LESSON 반박 1차원문 소싱 — §15.2-① 최대 개념위협 닫기 (2026-06-10)

> **지위**: `FIELD_GAP_LLM_VALUE_DESIGN.md` §18.2 항목5 수행 산출물 (zero-GPU 트랙). §15.4 open-2("Sutton-vs-neurosymbolic 1차원문이 검증셋에 0 = 최대 개념위협 미반박") 해소.
> **방어 대상 thesis (§17.9 동결)**: "보장(검증가능) soundness 하 *높은 coverage*를 {소형·저비용} × {감사가능 결정론게이트} × {재학습0 전이} *패키지*로 (= precision=1[인코딩제약 대비]서 recall 최대화)".
> **공격 (§15.2-①)**: "bitter lesson — 구조 주입은 scale이 이긴다. 손-설계 스키마는 스케일링이 1–2년 내 obsolete시킬 과도기 버팀목."
> **검증 방법**: 모든 인용 = 2026-06-10 1차원문 직접 fetch (Sutton 원문은 crosspost 경유 — incompleteideas.net SSL 오류; arXiv abs 페이지; DeepMind/Nature; Brooks 블로그 원문). §15.4 딥리서치 `w0yix88gp` 기검증(3-vote) 항목은 재fetch 없이 교차참조로 표기.
> **요약 판정**: 공격은 **범주 오류 위에 서 있다** — bitter lesson은 *해법-지식 주입*(how to think)을 겨냥하지, *문제-명세 집행*(what counts as correct)이나 *결정론 검증 method*를 겨냥하지 않는다(후자는 오히려 Sutton이 권장하는 "search" 계열 meta-method). 단, capability-침식 부분은 옳으므로 §15.3 기존 양보(정확도 헤드라인=시한부) 유지 — §7 정직 양보 참조.

---

## §1. Sutton 원문 — 정확히 무엇을 주장하나 (인용 박제)

**출처**: Richard S. Sutton, *The Bitter Lesson* (2019-03-13), http://www.incompleteideas.net/IncIdeas/BitterLesson.html (fetch는 crosspost https://braddelong.substack.com/p/hoistedcrosspost-richard-s-sutton 경유, 2026-06-10).

원문 핵심 문장 (verbatim):
1. **일반 명제**: "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."
2. **scale하는 두 방법 = search *and* learning**: "The two methods that seem to scale arbitrarily in this way are **search and learning**."
3. **금지 대상 = '우리가 생각하는 방식'의 주입**: "We have to learn the bitter lesson that building in **how we think we think** does not work in the long run."
4. **금지 대상 구체화 = 마음의 콘텐츠**: "The actual contents of minds are tremendously, irredeemably complex; … They are not what should be built in, as their complexity is endless; instead we should build in only the **meta-methods** that can find and capture this arbitrary complexity."
5. **단기-이득 패턴**: 지식 주입은 "always helps in the short term, and is personally satisfying to the researcher, but in the long run it plateaus and even inhibits further progress."
6. **발견 vs 소유**: "We want AI agents that can discover like we can, **not which contain what we have discovered**."

**정독 결과 — 원문이 주장하지 *않는* 것 3가지 (반박의 토대)**:
- (i) 원문의 공격 범위는 **"how we think we think" = 인간이 추정한 *해법* 구조·도메인 휴리스틱·마음의 콘텐츠**다. **외부에서 주어지는 task 명세/제약**(우리의 경우: 고객 SOP·규제 정책)은 논의 대상조차 아님 — 명세는 "인간이 생각하는 방식에 대한 추측"이 아니라 **문제 정의 그 자체**.
- (ii) 원문은 **search를 learning과 동급의 승자 method로 명시** — 결정론적 탐색/검증(우리 게이트 = 제약그래프 위 결정론 check + fail-safe abstain)은 bitter lesson의 *피해자* 목록이 아니라 *승자* 목록에 속한 계열. AlphaGo/AlphaZero(Sutton 본인이 승자 사례로 인용)부터가 learning + **결정론 MCTS search** 하이브리드.
- (iii) 원문은 **무엇을 보장(guarantee)할 수 있는가에 대해 일언반구도 없음** — 주장 전체가 *capability/성능* 축. soundness·감사가능성·컴플라이언스라는 우리 헤드라인 축은 원문의 적용 범위 밖.

---

## §2. 반박 라인 A — 범주 오류: 게이트는 "지식 주입"이 아니라 "명세 집행 + meta-method" (원문 자체로 방어, 최강)

**논거** (외부 인용 불요 — §1 원문 인용만으로 성립):
1. **정책 제약 = 문제 명세, 해법 지식 아님.** 은행 SOP "잔고 확인 전 이체 금지"는 "인간이 생각하는 방식에 대한 연구자의 추측"(Sutton의 금지 대상)이 아니라 **task의 정답 조건**이다. scale이 10⁶배 늘어도 *무엇이 정답인지*는 모델이 발명할 수 없고 외부에서 주어져야 한다 — 명세가 사라지지 않는 한 명세-집행 레이어의 가치는 scale과 무관. (Sutton: "not which contain what we have discovered" — 우리 게이트는 *discovered content*가 아니라 *주어진 constraint*를 집행.)
2. **결정론 게이트 = "search" 계열의 domain-general meta-method.** 게이트의 알고리즘(제약그래프 충족 검사·미충족 prereq 결정론 구동·fail-safe abstain)은 도메인 무관 — 도메인-특정 부분은 *데이터*(ABox/제약 인스턴스)로 들어옴. 이는 Sutton이 권장한 "meta-methods that can find and capture this arbitrary complexity"의 형태 그 자체. **Exp-5 재학습0 전이(scaffold가 7도메인 전이)가 이 domain-generality의 실측 증거** — 손-설계 콘텐츠였다면 전이 자체가 불가능했을 것.
3. **우리 파이프라인의 학습 부분은 Sutton-compliant.** thesis의 어려운 절반(NL→구조 coverage)은 *learning*(소형모델 학습·frontier-1-call induce)으로 푼다 — 손-설계 지식 축적이 아님. 즉 패키지의 분업이 정확히 Sutton의 두 승자 method 분업: **learning(coverage 제안) + search/verification(soundness 집행)**.

**적용**: §15.2-①의 "구조 주입은 scale이 이긴다" 공격은 우리 시스템을 1950–90년대식 *지식공학*(해법 휴리스틱 hand-coding)으로 오분류할 때만 성립. 정확한 분류 = learning + 결정론 search/verify 하이브리드 = bitter lesson의 승자 패턴.

---

## §3. 반박 라인 B — 보장축 직교성: scale은 capability를 사지만 soundness 보장은 못 산다 (1차 문헌)

| # | 출처 | 핵심 인용/결과 (verbatim 위주) | 적용 논거 |
|---|---|---|---|
| B1 | **Xu, Jain, Kankanhalli, "Hallucination is Inevitable: An Innate Limitation of LLMs"** (arXiv 2401.11817, 2024; learning-theory 형식 증명) | "LLMs **cannot learn all the computable functions and will therefore inevitably hallucinate** if used as general problem solvers" — 결과는 **모델 규모·데이터와 무관**한 형식적(diagonalization) 한계. | hallucination=0은 *어떤* scale에서도 도달 불가 → "o-class가 신뢰성을 네이티브화"의 극한이 존재 → precision=1이 요구되는 regime에서 외부 결정론 게이트는 영구 필요. 정확히 우리 #1 leg. |
| B2 | **Kalai, Nachum, Vempala, Zhang, "Why Language Models Hallucinate"** (OpenAI, arXiv 2509.04664, 2025) | "Hallucinations … originate simply as **errors in binary classification**"; "language models are optimized to be good test-takers, and **guessing when uncertain improves test performance**." | frontier 개발사 자체 분석이 hallucination을 scale 문제가 아니라 **통계·인센티브 구조** 문제로 진단 — guessing-보상 구조가 지속되는 한 모델은 abstain 대신 그럴듯한 오답을 냄. 우리 게이트의 fail-safe **abstain은 정확히 이 인센티브를 외부에서 뒤집는 장치**(모델이 학습으로 얻기 어려운 속성을 구조로 보장). |
| B3 | **Dziri et al., "Faith and Fate: Limits of Transformers on Compositionality"** (arXiv 2305.18654, NeurIPS 2023) | transformers는 "solve compositional tasks by **reducing multi-step compositional reasoning into linearized subgraph matching**, without necessarily developing systematic problem-solving skills"; "performance can **rapidly decay with increased task complexity**." | SOP-following의 본질=compositional 절차 실행. 패턴매칭 기반인 한 깊이↑서 붕괴 — 우리 SOPBench 실측(gathered_then_REFUSE·premature·fabrication)과 동형. 구조적 한계 → 게이트가 메우는 빈틈은 scale로 안 닫힘. |
| B4 | **Mirzadeh et al., "GSM-Symbolic"** (Apple, arXiv 2410.05229, 2024) | "Adding a single clause that seems relevant … causes **significant performance drops (up to 65%) across all state-of-the-art models**"; "current LLMs **cannot perform genuine logical reasoning**; they replicate reasoning steps from their training data." | "SOTA 전 모델"에서 표면 변화에 취약 = capability 상승이 robustness 상승이 아님. 감사가능성이 필요한 규제 도메인에서 이 분산 자체가 결격 — 결정론 게이트는 분산 0. |
| B5 | **Valmeekam, Stechly, Kambhampati** (arXiv 2409.13373; **§15.4 기검증** `w0yix88gp` 3-vote) | o1: plain Blocksworld 97.8% **but obfuscated 52.8%·20–40스텝 23.6%**; symbolic planner(Fast Downward)=100% 불변. | reasoning-model의 이득은 실재하나 robust·long-horizon·domain-general 아님 — "1–2년 내 obsolete" 시간표의 실측 반례. |
| B6 | **(constraint fabrication)** (arXiv 2505.12151; **§15.4 기검증**) | reasoning-model 전반(o1-mini/o3-mini/R1/Claude-3.7/Gemini-2.5/Grok-3)이 **프롬프트에 없는 제약을 환각**(false-error의 67–94%). | 제약을 fabricate하는 모델은 제약-충실성을 weight에 내재화 못함 = 제약의 **외부 결정론 표현·집행** 논거 직격. |
| B7 | **(LRM-Modulo)** (arXiv 2410.02162; **§15.4 기검증**) | o1조차 **correctness 보장 0** — soundness는 외부 verifier에서; o1=cost/time/guarantee/perf 트레이드의 한 점. | "determinism은 capability와 직교"의 직접 진술 — scale 축과 보장 축은 다른 축. |

**라인 B 종합**: bitter lesson이 옳게 예측하는 것은 *기대 성능*의 상승이지 *worst-case 보장*이 아니다. B1(형식 증명)·B2(인센티브 구조)는 보장 부재가 scale의 함수가 아님을, B3–B6은 현 frontier에서의 실측 붕괴를, B7은 직교성의 명시 진술을 제공. **thesis 헤드라인이 §15.3/§17.9에서 capability가 아니라 soundness-검증가능성 패키지로 이미 재좌표된 이상, bitter lesson의 사정거리 밖.**

---

## §4. 반박 라인 C — 구조-옹호 진영 1차 문헌 (권위·생태계)

| # | 출처 | 핵심 인용 (verbatim 위주) | 적용 논거 |
|---|---|---|---|
| C1 | **Rodney Brooks, "A Better Lesson"** (2019-03-19, https://rodneybrooks.com/a-better-lesson/ — Sutton 직접 반박문) | "for most machine learning problems today **a human is needed to design a specific network architecture** for the learning to proceed well"; "a **better lesson** … we have to take into account the **total cost of any solution**, and that so far they have all required substantial amounts of human ingenuity." | ① 인간 구조설계는 제거된 게 아니라 **재배치**됨(지식공학→아키텍처/목적함수/벤치 설계) — "구조 0"인 진영은 없음, 차이는 구조를 *어디에* 두느냐. ② **total cost 프레임 = 우리 {소형·저비용} leg와 동일 축**: frontier 콜비용 vs 소형+게이트 amortization(§9-2)이 정확히 Brooks의 비교 프레임. |
| C2 | **Gary Marcus, "The Next Decade in AI: Four Steps Towards Robust AI"** (arXiv 2002.06177, 2020) | "general-purpose learning and ever-larger training sets" 패러다임 대신 "**hybrid, knowledge-driven, reasoning-based approach**, centered around cognitive models"가 robust AI 경로. | scaling-only에 대한 가장 직접적인 학술 반론 포지션 페이퍼. *robustness*가 목표일 때 hybrid가 경로라는 주장 = 우리 soundness leg의 진영 좌표. |
| C3 | **Garcez & Lamb, "Neurosymbolic AI: The 3rd Wave"** (arXiv 2012.05876, 2020; AI Review 2023) | "concerns about **trust, safety, interpretability and accountability** of AI were raised by influential thinkers"; 필요한 것 = "well-founded knowledge representation and reasoning to be integrated with deep learning and for **sound explainability**." | 신경-기호 통합의 동기 자체가 capability가 아니라 **trust/safety/accountability** — bitter lesson이 다루지 않는 축의 survey-급 정식화. 우리 "감사가능"의 문헌 닻. |
| C4 | **Henry Kautz, "The Third AI Summer"** (AAAI-20 Engelmore Lecture; AI Magazine 43(1):105–125, 2022) | neurosymbolic 시스템 6-type taxonomy(통합 강도 순). 우리 패키지 = Neuro\|Symbolic 협력형(LLM 제안 + symbolic 검증 in-loop)에 해당. | AAAI 주류가 신경-기호 통합을 3차 여름의 정식 의제로 좌표화 — "구조 주입=과거 회귀"가 아니라 현 단계 분류체계가 존재. 우리 위치를 taxonomy 좌표로 지정 가능(related-work 방어). |
| C5 | **Yann LeCun, "A Path Towards Autonomous Machine Intelligence"** (OpenReview, 2022) | 제안 = "configurable predictive **world model**" + 모듈형 인지 아키텍처 + 계층 계획; 출발 질문 = "How could machines learn to reason and plan?" (현 패러다임으로 부족하다는 함의). | 딥러닝 진영 내부 최고 권위조차 **구조화된 모듈 아키텍처**(=구조 주입의 한 형태)를 경로로 제시 — "scale만으로 충분" 진영이 딥러닝 진영 전체가 아님. |
| C6 | **François Chollet, "On the Measure of Intelligence"** (arXiv 1911.01547, 2019) | "**unlimited priors or unlimited training data allow experimenters to 'buy' arbitrary levels of skills** for a system, in a way that masks the system's own generalization power." | scale은 *skill*을 사는 것이지 *일반화*를 사는 게 아님 — 우리 {재학습0 전이} leg(스킬 구매가 아닌 transfer 측정: LODO·alias-마스킹·cross-bench)의 방법론적 정당화. 평가 함정(=forward guard "frontier 이기나로 평가 금지")과도 정합. |
| C7 | **Kambhampati et al., "LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks"** (arXiv 2402.01817, **ICML 2024**; §15.4 기검증) | "auto-regressive LLMs **cannot, by themselves, do planning or self-verification** (which is after all a form of reasoning)"; LLM = "universal approximate knowledge sources" + 외부 **model-based verifiers**와 양방향 통합. | LLM(제안)+외부 결정론 verifier(보장) 분업이 ICML 메인트랙 정식 패러다임 — 우리 패키지의 분업 구조와 1:1. reasoning-model 시대에도 지속(§15.4 판정①). |

---

## §5. 반박 라인 D — frontier 최전선 자체가 neuro-symbolic 하이브리드 (실증 카운터)

| # | 출처 | 결과 | 적용 논거 |
|---|---|---|---|
| D1 | **Trinh et al., "Solving olympiad geometry without human demonstrations"** (Nature 625, 2024 = **AlphaGeometry**; DeepMind 블로그 fetch 검증) | neural LM(직관·construct 제안) + **symbolic deduction engine**(formal logic, "rational and explainable") 하이브리드가 IMO 기하 25/30(금메달리스트 평균 25.9 근접; 직전 최고 AI=10). LM 단독은 "often **lack the ability to reason rigorously or explain their decisions**." | scale 진영의 본진(DeepMind)이 최고 난도 추론에서 **결정론 기호엔진을 결합해야** 도달 — "구조는 scale이 이긴다"의 살아있는 반례. 분업 구조(신경=제안/기호=검증·집행)가 우리 패키지와 동형. |
| D2 | **AlphaProof / IMO 2024 silver** (DeepMind, 2024; AlphaProof+AlphaGeometry-2) | 증명을 **Lean 형식 검증기** 위에서 RL — 보상 자체가 결정론 verifier. | 동일 패턴 반복: **RLVR의 'V'가 결정론 검증기** — scale 시대의 학습 사다리가 결정론 구조를 *연료로* 씀. "scale이 구조를 폐기"가 아니라 "scale은 검증가능 구조 위에서 작동". 우리 §16 레시피(outcome-RFT의 보상=Guard-2/TSR 결정론 게이트)와 동형. |
| D3 | (보조·비학술) **Karpathy** (X, 2025-10, Sutton 인터뷰 반응) | "Sutton's 'The Bitter Lesson' has become a bit of **biblical text in frontier LLM circles**" — 이면서 본인도 LLM이 그 교리와 긴장 관계임을 논함. | 위협 ①이 실재하는 *사회적 통념*임을 인정하면서(정직), 그 통념의 경전 해석이 frontier 내부에서도 논쟁 중임을 보임. |

---

## §6. 반박 라인 E — Sutton 본인(2025): "LLM은 bitter-lesson-pilled가 아니다" (공격 삼단논법의 내부 붕괴)

**출처**: Dwarkesh Patel 팟캐스트 "Richard Sutton – Father of RL thinks LLMs are a dead end" (2025-09, https://www.dwarkesh.com/p/richard-sutton) + Patel 후기 (https://www.dwarkesh.com/p/thoughts-on-sutton).

- Sutton 입장(인터뷰·후기 요약, 검증 fetch 2026-06-10): LLM은 "massive computation을 쓰는 방법이지만 **동시에 lots of human knowledge를 집어넣는 방법**" — 인간 텍스트 모방(imitation) 기반이므로 bitter lesson의 승자 패턴이 아니며, 경험학습(continual/on-the-job learning) 시스템에 의해 "**another instance of the bitter lesson**"으로 superseded될 것.
- **적용 논거**: §15.2-① 공격의 삼단논법 = "(대전제) bitter lesson: scale이 구조를 이긴다 → (소전제) o-class LLM scaling이 그 scale이다 → (결론) 게이트 obsolete". **원저자가 소전제를 부정** — LLM-scaling은 bitter lesson이 보증하는 경로가 아니라 그 자체가 "인간 지식 주입" 사례. 따라서 "bitter lesson의 권위"를 빌려 LLM-scaling이 우리 게이트를 폐기한다고 주장할 수 없음 — 권위의 원천이 그 사용법을 기각.
- ⚠️**양날 주의 (정직)**: Sutton의 대안은 경험-RL이지 구조 주입이 아님 — 이 카드는 "bitter lesson ⇒ *현 LLM-scaling*이 흡수" 형태의 공격만 무력화하며, "미래의 경험학습 agent가 흡수"라는 더 깊은 형태는 남는다. 그 형태에 대한 방어는 라인 A(명세는 발견 대상이 아님)·B(보장 직교)가 담당.

---

## §7. 정직 양보 — bitter lesson이 *맞는* 부분 (motivated-reasoning 방지, §15.4 규율 승계)

1. **capability/정확도 헤드라인 침식은 인정 유지**: §15.3 판정("정확도·비용·전이 *정확도성능* 헤드라인 = reasoning-model+스케일링에 침식 = 시한부; 이걸로 frontier와 싸우면 진다") **변경 없음**. 본 문서는 이 양보를 뒤집지 않는다 — thesis가 산 것은 헤드라인을 soundness-검증가능성 패키지로 옮겼기 때문(§17.9).
2. **formalizer-우위 erosion 선제 인용 유지** (arXiv 2412.09879, §15.4 c12/13): top reasoning-model(o3-mini/R1)에선 formalizer 우위가 simple-PDDL에서 부분 침식 — messy-NL-SOP 확장 여부는 **미검증(open, §18.2 항목6)**. 이 카드는 우리가 *직접 인용해 선제 방어*한다는 §15.4 방침 그대로.
3. **손-설계 도메인 콘텐츠는 실제 위험** (§15.2-②와 정합): per-domain 온톨로지/실행기를 *손으로* 설계·유지하는 형태라면 bitter lesson 비판이 정당하게 꽂힌다(그건 지식공학이 맞음). 방어가 성립하는 것은 (a) 게이트 *알고리즘*이 domain-general이고(전이 실측) (b) 도메인 부분이 *명세 데이터*이며 (c) NL→구조 컴파일을 *학습*으로 풀 때 — **즉 thesis의 front-end 학습 leg(Exp-B blind-E1)가 성공해야 이 방어가 완결**. 실패 시 §15.2-② 후퇴("per-domain DSL 손설계")가 현실화되고 bitter lesson 공격이 부분 재개방됨을 박제.
4. **Brooks/Marcus/LeCun의 권위는 '진영 존재' 증거이지 '진영 승리' 증거가 아님**: 라인 C는 "구조-옹호가 fringe가 아니다"까지만 지지. 승부 자체는 라인 A(범주)·B(직교 형식 결과)·D(frontier 실증)가 담당.

---

## §8. §15.2-① 처분 제안 (설계서 갱신용 문구)

> **§15.2-① 갱신안**: "(= bitter lesson; ~~`wheyskq29`로 검증 중~~ → **반박 소싱 완료 `BITTER_LESSON_REBUTTAL_SOURCES.md` 2026-06-10**: ①범주 오류 — Sutton 원문의 금지 대상은 해법-지식 주입('how we think we think')이지 명세-집행 아님; 게이트=search-계열 meta-method(원문이 권장) + 명세는 발견 대상이 아님 ②보장축 직교 — hallucination 불가피성 형식증명(2401.11817)·OpenAI 자체 진단(2509.04664)·compositional collapse(2305.18654/2410.05229) = precision=1은 scale로 구매 불가 ③frontier 자체가 hybrid(AlphaGeometry Nature'24, RLVR의 V=결정론 verifier) ④Sutton 본인이 'LLM은 bitter-lesson-pilled 아님' 진술(2025). **잔존 유효 부분 = capability 헤드라인 침식(§15.3 양보 유지)·front-end 학습 leg 성공 조건부(§7-3)**.)"

**검증셋 등재 후보 (딥리서치 검증셋 0건 → 본 문서로 1차 충족; 추후 3-vote 정식 등재 시 우선순위)**: 2401.11817(형식증명·load-bearing) > Sutton 원문(범주 오류의 근거 텍스트) > 2402.01817(이미 등재) > Nature AlphaGeometry > 2509.04664 > Brooks/Marcus/Garcez-Lamb/Kautz/LeCun/Chollet(진영 좌표용).

---

## 부록 — 전체 출처 목록 (fetch 검증 상태)

| 출처 | 식별자 | 검증 |
|---|---|---|
| Sutton, *The Bitter Lesson* (2019) | incompleteideas.net/IncIdeas/BitterLesson.html | ✅ crosspost fetch (원 사이트 SSL 오류) |
| Brooks, *A Better Lesson* (2019) | rodneybrooks.com/a-better-lesson/ | ✅ 원문 fetch |
| Marcus, *The Next Decade in AI* (2020) | arXiv 2002.06177 | ✅ abs fetch |
| Garcez & Lamb, *Neurosymbolic AI: The 3rd Wave* (2020) | arXiv 2012.05876 | ✅ abs fetch |
| Kautz, *The Third AI Summer* (2022) | AI Magazine 43(1):105–125 | ✅ 서지 검증 (taxonomy 세부는 2차) |
| LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) | OpenReview BZ5a1r-kVsf | ✅ abs fetch |
| Chollet, *On the Measure of Intelligence* (2019) | arXiv 1911.01547 | ✅ abs fetch |
| Kambhampati et al., LLM-Modulo (ICML 2024) | arXiv 2402.01817 | ✅ abs fetch + §15.4 기검증 |
| Trinh et al., AlphaGeometry (Nature 2024) | Nature 625 + DeepMind blog | ✅ blog fetch (Nature 본문 paywall) |
| Xu et al., *Hallucination is Inevitable* (2024) | arXiv 2401.11817 | ✅ abs fetch |
| Kalai et al., *Why Language Models Hallucinate* (2025) | arXiv 2509.04664 | ✅ abs fetch |
| Dziri et al., *Faith and Fate* (NeurIPS 2023) | arXiv 2305.18654 | ✅ abs fetch (venue는 기지정보) |
| Mirzadeh et al., *GSM-Symbolic* (2024) | arXiv 2410.05229 | ✅ abs fetch |
| Sutton–Patel 인터뷰 + 후기 (2025) | dwarkesh.com/p/richard-sutton, /p/thoughts-on-sutton | ✅ 검색 결과 본문 검증 |
| Valmeekam et al. (o1 PlanBench) | arXiv 2409.13373 | §15.4 기검증 (재fetch 생략) |
| LRM-Modulo 트레이드 | arXiv 2410.02162 | §15.4 기검증 |
| constraint fabrication | arXiv 2505.12151 | §15.4 기검증 |
| formalizer erosion (반대증거, 선제 인용) | arXiv 2412.09879 | §15.4 기검증 |
