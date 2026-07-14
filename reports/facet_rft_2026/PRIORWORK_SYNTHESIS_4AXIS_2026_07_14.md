# 선행연구 종합 — 4축 딥리서치 harvest 정본 (2026-07-14)

> 정본(provenance anchor). handoff LATE-10 §0·§3 지시 = 딥리서치 1·2·3 선행지형 종합 + 4번 harvest → [[41]]·[[46]] 갱신.
> 방법: 4 workflow journal harvest (`subagents/workflows/<runId>/journal.jsonl` result 이벤트).
> **증거등급([[47]] 규율)**: 4축 **전부 verify 통과 = [M]**. 축1·2 = 원 딥리서치서 verify(refuted 42·31). 축3·4 = **2026-07-14 verify-only workflow `wf_33f2f4c7-64b`로 사후 검증**(107 claim·refute-by-default·web-grounded).
> **[[08]] 포렌식 통과**: 근접 pre-emption 2편(2606.30531·2601.07264) 모두 **실존 확인**(다중 verifier가 PDF 추출·Table 수치·verbatim 인용 대조·arXiv HTML/ACL Anthology 독립확인) = 환각 아님.

## ★verify 결과 종합 (2026-07-14·`wf_33f2f4c7-64b`·107 claim·에러 0)
- **축3 horizon: 82/86 통과** (central+supporting). refuted 4건 = **전부 라벨/프레이밍 overreach·출처 환각 아님**(논문 실존):
  - axis3-11·40·43 = **동일 논문 `arXiv:2603.29231` "Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents"**(Khanal/Tao/Zhou·2026-03-31·실존): 수치 정확하나 ①"RDC+MOP=**exponential** decay 조작화"는 overreach(논문은 **super-exponential**·exponential=초과대상 baseline·MOP=엔트로피 collapse detector) ②**GDS=Graceful Degradation Score(부분점수 metric)**를 "goal-directed success"로 오명명 ③"document processing"→"data-processing" 오기. **★핵심 실증(memory-scaffold 6/10 악화·SE 0.90→0.44 vs doc 0.74→0.71·domain-stratified)은 유효**·라벨만 교정.
  - axis3-80 = **`arXiv:2601.07577` "Beyond Entangled Planning: TDP"**(실존): 토큰절감 74.5-82.4%를 TravelPlanner에 오귀속(실제=**HotpotQA(82.4/82.1%)+ScienceWorld(74.5/70.4%)만**·진짜 최소 70.4%·quote stitched). TDP 핵심(task-decouple·성능 heterogeneity)은 유효.
- **축4 이론: 21/21 전부 통과** — §17-18 수학토대(Ramsauer `2008.02217` Hopfield≡attention·Fano·Hill exp(H)=k_eff·energy-conformal·softmax비신뢰→conformal) **전부 [M] 확립**.
- **novelty 영향 없음**: refuted 4건은 라벨 교정이지 존재/방향 반박 아님. decomposition 선점·reliability≠scale·수학토대 전부 생존.
- **확보 arXiv ID(신규)**: reliability-science `2603.29231` · TDP `2601.07577` · ToT `2305.10601` · Huang self-correct `2310.01798` · execution-horizon `2509.09677` · Ramsauer `2008.02217`.

---

## 축1 — 참조-앵커링 (⋈ / reference resolution) — `wf_d9dbbf1a-af3` (verify 42, 통과 grade [M])

### ★근접 pre-emption 판정: **부분 선점, "직접 선점" 아님** (verifier 다수결 = overreach 반박)
- **arXiv:2606.30531 "Entity Binding Failures in Tool-Augmented Agents" (Babu & Indukuri, v1 2026-06-29)** — **실존·PDF 검증**.
  - 형식화: `EntityBindingFailure(a)=1[t(a)=t* ∧ ê(a)≠e*]` (맞는 action type·틀린 target·"rescheduling the wrong event"). candidate set `C(m,S)`, `Ambiguous(m,S)=1[|C(m,S)|>1]`.
  - 실측: 60 task × 5 backend × 6 method = 1,800 run. **0.0% wrong-tool** vs **24.0-26.0% wrong-entity**(action baseline). Entity-retrieval = direct prompting **동일 26.0%** → "entity binding is not merely a retrieval problem. Retrieval can surface candidate entities, but it does not decide whether the request uniquely identifies one of them."(§VII-B) 8-범주 모호성 taxonomy(name collision·temporal·near-duplicate·cross-system…).
  - **★우리 delta(선점 안 됨·verifier 만장에 가깝게 확인)**:
    1. **축 불일치**: 그들 = TOOL(operation) vs ENTITY(target). 우리 = **field/value EXTRACTION vs referent LINKING**. 그들은 "linking-not-OPERATION"만 실증·**extraction을 solved dimension으로 측정 안 함**. 우리 C79 = extraction 정당대상 69.5% 도달·랜덤오류 1.5% vs anchoring 66% mis-pairing = **extraction-vs-anchoring 분해**가 그들 축과 다름.
    2. **0.0% wrong-tool = by-design 아티팩트**: 논문 §VIII Limitations 자백("designed to isolate 'right tool, wrong target'"·"controlled diagnostic, not comprehensive benchmark").
    3. **세팅 불일치**: 그들 = single-step, 외부 enterprise entity resolution(§VIII 명시적으로 multi-step/conversational 제외). 우리 = **multi-turn 한 대화 내 다중 co-disputed record 앵커링**(C79: hard-core ⋈ 852/853 = 다중-dispute).
  - **판정**: "referent-binding이 tool-selection보다 지배" 일반 framing = 강한 부분중첩(**양보·인용 필수**). 남는 whitespace = ①extraction-vs-anchoring 축 ②multi-turn 다중-referent 앵커링 ③verify-or-ASK offload ④H_min 정량.

### 지지·인접 선행 (양보·인용)
- **arXiv:2408.09846 "Continual DST via Reason-of-Select Distillation" (ACL2024 Findings·peer-reviewed)** — **실존**. "Value Selection Quandary": 오류 **94.5%**가 의미유사 wrong value(추출 아님·선택 실패). 세부: 45% 최근값·30% stale 고수. **우리 정박치환/near-miss 동형**. ⚠ caveat: continual-DST·long-dialogue(turn>10)·소형모델 scope·**value-update tracking**이지 다중-referent 앵커링은 아님(refuter 1건: "which-referent vs extraction" gloss는 overreach).
- **arXiv:2511.08798 "SAGE-Agent / Structured Uncertainty guided Clarification" (2025-11)** — **실존**. 모호성 = under-specified **PARAMETER**(schema level·POMDP·Bayesian EVPI clarification). **in-context 다중 referent 선택은 다루지 않음** → 우리 앵커링을 pre-empt 안 함(whitespace 지지).
- **arXiv:2503.00564 "ToolDial" (ICLR2025)** — **실존**. Appendix A.8이 "Extracting Input Value from **Multiple Result** Error"를 generic extraction error와 **별개 클래스**로 명명(예: API가 다중 payment record 반환·모델이 잘못 선택). ⚠ **반박된 claim**: "DST error가 action-prediction보다 far exceed"는 **거짓**(Table 4: action-prediction이 더 자주 실패). DST가 selection+extraction을 fuse → 분해는 whitespace로 남김.
- **arXiv:2503.01940 "AskToAct" (EMNLP2025)** — **실존**. 모호성 = missing/omitted parameter recovery via clarification(57% 회복). under-specification이지 다중-referent 선택 아님.
- **Jurafsky & Martin SLP3 Ch.26** — coreference = mention detection(추출) + clustering(linking) **분리**. gold-mention isolation·mention-pair→mention-ranking(경쟁 antecedent 중 선택). **근간 계보**(우리 extraction/linking 분리의 고전 뿌리)·단 LLM tool-use서 selection 지배는 별개 명제.

---

## 축2 — Calibration·저-ASK abstention — `wf_ce8e3320-3e9` (verify 31, 통과 grade [M])

### ★근접 pre-emption 판정: **abstention측 성숙·선점**·**우리 whitespace = ground-vs-ASK 라우팅(joint ASK-rate↓)**
- **arXiv:2601.07264 "The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents" (Xuan et al.·ACL2026·aclanthology 2026.acl-long.520)** — **실존·peer-reviewed**.
  - 발견: evidence tool(web search)=noise로 **severe overconfidence** 유발 / verification tool(code interpreter)=결정론 feedback으로 ground·miscalibration 완화.
  - 방법: **RL fine-tuning으로 task accuracy + calibration 동시 최적화**(reward 벤치·noisy web·math 전이).
  - **★우리 delta(선점 안 됨·verifier 명시 caveat)**: 그들 calibration = **confidence-correctness** 정렬. 우리 = **ASK-rate/over-clarification 라우팅**(ground 가능→결정론verify·불가→ASK). "ground-vs-ask로 ASK-rate↓"의 joint objective는 그들 축 아님 = relevance limit.
- **arXiv:2502.06884 "Learning Conformal Abstention Policies (CAP)" (Tayebati et al.·2025·PMLR v304)** — **실존**. RL+conformal·per-instance adaptive abstention threshold. static threshold 비판. +22.19% AUROC(hallucination)·+21.17% AUARC. ⚠ tool-use agent 아님(LLM/VLM QA).
- **arXiv:2405.01563 "Mitigating LLM Hallucinations via Conformal Abstention" (DeepMind·2024)** — **실존**. conformal abstention·hallucination-rate 보장(exchangeability 전제·marginal)·long-form서 덜 보수적.
- **arXiv:2307.09254 "Selective Generation for Controllable LMs" (Lee et al.·NeurIPS2024 Spotlight)** — **실존**. IDK abstention·FDR-E(textual entailment 기준) 제어·distribution-free PAC. ⚠ **abstain-when-uncertain측만**·**IDK로 라우팅이지 ground-vs-ask 아님** → joint objective whitespace 지지.
- **arXiv:2504.14154 "SConU: Selective Conformal Uncertainty" (ACL2025)** — **실존**. conformal p-value·per-sample exchangeability outlier 필터·miscoverage 관리.
- 인접(verify 언급·미하베스트): 2602.06948 "Agentic Uncertainty Reveals Agentic Overconfidence", 2601.15778 "Agentic Confidence Calibration"(일관·모순 아님).

---

## 축3 — Horizon (verify 통과 82/86 = [M] · `wf_33f2f4c7-64b`) — 원 harvest `wf_f18b104d-b2a`

> C70(§SCALE_DYNAMIC_CONTAMINATION)·C82·C83과 상당 중첩. **verify 완료**(refuted 4=라벨교정·위 §verify결과).

### A. horizon = per-step 신뢰도 곱 (우리 §15 논리곱의 핵심 선행)
- **arXiv:2509.09677 "The Illusion of Diminishing Returns / execution horizon" (Sinha/Geiping·ICLR2026)** — **이미 C70 등재**. `H_s(p)=⌈ln(s)/ln(p)⌉`·marginal per-step gain → exponential horizon. **self-conditioning**(자기오류가 다음오류 확률↑·correlated). thinking 모델은 self-condition 안 함(turn-100 안정). **scale이 execution horizon 상승**(소형이 near-perfect per-step이어도). = 우리 §15-16·F6 직접 선행.
- **METR "Measuring AI Ability to Complete Long Tasks" (2025-03-18)** — 50%-task 시간지평 ~50min(Claude 3.7)·**7개월마다 doubling**. horizon = capability/scale 축 지지.
- 블로그(2026-06-03): Lusser's Law(직렬 신뢰도 곱)·95%/step → 10step 59%·50step 7%.

### B. 오류 correlated/cascading (i.i.d. p^N 아님 — 우리 §15 상관 지지)
- 다중에이전트 contagion `βρ(A)>δ`(2026-03-04): 6 framework 중 5가 100% 최종 infection(**reviewer/QA role 있어도**). detection alone 무력(rollback 없으면 3.1%).
- off-canonical tool call **+22.7pp** 자기강화(2026-02-22): reliability = **capability/scale 아님**("cannot be improved by capability scaling alone")·mid-trajectory 모니터 restart **+8.8pp**. canonical path adherence(+0.060 Jaccard)·drift는 점진적(전반 50% 무차이).
- cascading hallucination RAG(2026-06-03): stage-level 검증 -82.1% vs output-level -18.5%.
- compositional depth 비선형 붕괴(2026-04-13): "model scaling alone unlikely to resolve"·planning/memory 지배(72.5% process-level).

### C. scale이 horizon을 단조로 사지 않음 (우리 [[45]] load·[[46]] crossover 지지 + pre-emption 위험)
- **`arXiv:2603.29231` "Beyond pass@1: Reliability Science Framework"**(Khanal et al.·2026-03-31): 4 metric RDC/VAF/**GDS(Graceful Degradation Score·부분점수)**/MOP(엔트로피 collapse). **domain-stratified decay**(SE GDS 0.90→0.44·**document processing** flat 0.74→0.71)·**frontier 최고 meltdown 19%**·**memory(episodic scratchpad) scaffold 6/10 악화·나머지 neutral**·**decomposition = highest-leverage**. ★논문 주장 = reliability decay는 **super-exponential**(i.i.d. geometric을 초과)·단 memory 결론은 "per-task calibration 없이 배치 말라"(scaffold 무용 아님)·1개 naive scratchpad만 테스트(over-generalize 주의).
- ⚠ **pre-emption 위험**: "decomposition이 최고 레버·scaling이 안 고침"은 이미 다수 주장 → 우리 delta는 **verify-or-ASK + H_min 정보이론 floor + decidable-vs-nondecidable 라우팅 + DERIVE/DEFAULT 닫기**(정량)여야 함. generic "decompose"로는 부족.

### D. exponential을 깨는 decomposition+voting (heavily occupied — 양보·차별 필요)
- **MAKER (2025-11)**: Maximal Agentic Decomposition + first-to-ahead-by-k voting + red-flag로 **100만+ step 0오류**(20-disk Hanoi). cost Θ(s ln s).
- TDP "Task-Decoupled Planning" (`arXiv:2601.07577`·2026-01): supervisor→DAG·per-subtask 격리 컨텍스트·active node만 replan·**token -70.4~82.4%(HotpotQA+ScienceWorld·TravelPlanner 아님)**·3벤치 성능 heterogeneity 안정.
- ReAcTree (2025-11): 계층 subgoal·61% vs 31%(ReAct)·**7B가 37% 도달**(소형이 대형 rival)·Behavior-Tree sequence = 명시적 AND.
- ToT (NeurIPS2023): tree search·backtrack·Game-of-24 74% vs 4% CoT.

### E. 자기교정은 외부 feedback 없이 per-step 신뢰도 못 올림 (우리 결정론-verify 정당화)
- Huang et al. ICLR2024 (2310.01798): intrinsic self-correction 실패·때로 악화.
- Kamoi et al. TACL2024: LLM-only feedback로 성공 사례 없음·**large-scale fine-tuning이 self-correction 활성**(학습으로 삼).
- 2024-12-19: oracle label 없으면 실패·질문반복+SFT 처방.
- (대조) self-reflection이 도움(2024-05-05)이나 **oracle correctness 신호 하**서만.

### F. test-time scaling plateau
- (2026-06-16 블로그): repeated sampling Elo = log-compute 선형·agent sublinear·top human superlinear로 추월·agent ~24h plateau.

**축3 종합**: horizon = 논리곱·correlated·scale-부분내성은 **강하게 선점**. 우리 순수 delta = **verify-or-ASK controller(decidable→결정론verify·non→ASK) + H_min 정보이론 정량(§16·C83) + DERIVE/DEFAULT로 N_eff→H_min 접기**. decomposition 자체는 양보.

---

## 축4 — 이론 엄밀 (Hopfield·conformal·VOI) (verify 통과 21/21 = [M] · `wf_33f2f4c7-64b`) — 원 harvest `wf_0d395776-1f2`

> §17-18 수학 형식화의 **지지 인용**(pre-emption 아님). **verify 완료·전건 통과.**

- **Ramsauer et al. "Hopfield Networks is All You Need" (ICLR2021·2008.02217)**: modern continuous Hopfield update = transformer attention **정확 등가**(1 step). 3 fixed-point 체제(global avg·**metastable subset avg**·single-pattern)·지수 저장용량(≈2^(Nf/2))·**층depth별**(초층 global·상층 metastable partial avg). = 우리 "attention=1 Hopfield step"·"confusable subset→metastable"·H_min^ref=log₂ k_eff 토대.
- **Fano 부등식**(Bayes error가 H(Y|X)로 lower-bound·2013 paper는 upper/lower 둘 다 도출): 엔트로피만으로 선택오류확률 pin 안 됨. Infomax는 F-score/cost-sensitive risk에 non-optimal(minimize entropy ≠ minimize decision error) → **ASK 순서=VOI가 objective-의존** 지지·단순 엔트로피 최소화 부적절.
- **Hill number exp(H) = effective number of types**(2019): uniform이면 정확히 n → **k_eff = e^H = 유효 candidate 수** 근거.
- **energy-based conformal reweighting**(2026-02-23): pre-softmax logit energy(Helmholtz free energy)가 softmax보다 나은 uncertainty·nonconformity reweight로 efficiency/adaptiveness↑(approximate·정리 아님).
- **softmax/entropy 비신뢰**(2026-04-06·2026-02-23): softmax는 진짜 certainty 나쁜 지표·hard input서 overconfident·conformal set이 그 편향 상속 → **calibration 편향은 conformal(분포무가정)로 보정** 필요·runtime 단일 forward-pass H_min은 miscalibrated면 비신뢰 = §18 online H_min의 calibration 전제 지지.

**축4 종합**: §17-18(H_min = log₂ k_eff·ASK = 에너지갭·conformal 보정·VOI 순서) 수학 토대 = 확립된 선행이 지지. Hopfield≡attention·Hill·Fano·conformal 전부 **established**(단 verify 재확인 필요).

---

## 즉시 반영 (메모리·ledger)
- **[[41]]**: 축1·축2 근접 pre-emption 2편 + delta 정련 추가.
- **[[46]]**: reference-anchoring·calibration 이웃을 foil/양보 목록에 추가·우리 crossover moat 불변.
- **등대 ledger**: C84(축1 ⋈)·C85(축2 calibration)·C86(축3 horizon·verify 82/86 [M])·C87(축4 이론·verify 21/21 [M]).
- **✅ verify 완료**(2026-07-14·`wf_33f2f4c7-64b`·107 claim): 축3 82/86·축4 21/21 통과 → **4축 전부 [M]**. refuted 4=라벨 교정(위 §verify결과). 남은 미검증 = tangential 10건(central/supporting 아님·drop됨).
