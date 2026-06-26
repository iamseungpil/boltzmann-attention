# Related Work — 선별/MBR/Verifier 딥리드 (full-text, fit/divergence/error 분석)

**날짜**: 2026-06-14 · **입력**: `research_selector_lit_2026_06_12.md` §5 검증서지 · **그라운딩**: `SELECTOR_DESIGN.md`(SEL-1~5, V-라인), TB결과 §8.9b–§8.9h.
**방법**: 각 논문 ar5iv 전문(method+table+limitation) WebFetch. 본 세션 직독으로 검증된 verbatim만 인용; 미직독은 §A에서 명시. 우리 설정 4-제약 = (a) 이종 제안 풀 (b) 선별 시점 gold-free (c) judge ≤7B/on-prem (d) 구조적 JSON DAG 출력.

랭킹 = load-bearing(우리 SEL-x를 직접 떠받치거나 반증하는 정도) 내림차순.

---

## TIER 1 — 설계 기둥 (각 SEL-x를 직접 떠받침)

### 1. Coder-Reviewer Reranking — [arXiv:2211.16490] → SEL-4의 직계
- **서지(직독 검증)**: "Coder Reviewer Reranking for Code Generation", Tianyi Zhang, Tao Yu, Tatsunori B. Hashimoto, Mike Lewis, Wen-tau Yih, Daniel Fried, Sida I. Wang. ICML 2023(PMLR v202).
- **CLAIM/METRIC(verbatim)**: Reviewer 점수 = p(x|y)(=p(instruction|code)), Coder = p(y|x); 결합 = "log p(x|y)p(y|x) = log p(x|y) + log p(y|x)". 최대 이득 "**up to 17% absolute accuracy gain**". 구체표: Codex002 HumanEval **62.5% (N.Coder-Reviewer) vs Coder 45.1%**; InCoder6B Plotting **35.5% vs 15.5%**.
- **모델 크기/공정성**: 6 dataset × 8 model(Codex 175B/12B, CodeGen 16B/6B/2B, InCoder 6B/1B). 표준 코드벤치(HumanEval/MBPP/Spider/NL2Bash/Plotting). 동일 모델의 self-baseline(Coder reranking·MAP) 대비 일관 개선. **공정** — 단 48 model-dataset 중 5건은 CodeGen에서 역전(사전훈련 차 귀속, 저자 명시). 175B 위주라 7B-only 검증은 우리가 채워야 함.
- **FIT(SEL-4)**: 역방향 likelihood p(instr|plan)이 합의와 직교 신호라는 우리 SEL-4 채택 근거의 1차 출처. 결정적 verbatim: "**reranking with the Coder model often mistakenly prefers degenerate solutions, e.g., extremely short code or repetitive solutions**" — 우리 D2 brevity-prior 부검·게이트 역선택과 정확히 같은 병리, 그리고 Reviewer 항이 그 교정. SEL-4가 hf에서 −0.3pp였던 것(기판-의존)은 이 논문의 "짧은 입력=약신호"(Plotting류) 한계와 정합.
- **DIVERGENCE/ERROR**: 우리 prior report가 "+17%p absolute"를 큰 수로 강조했는데 원문은 "up to 17% absolute" = **상한치(특정 모델/셋)**이지 평균 아님 — paper에 쓸 땐 "up to" 유지 필수. 또 Coder×Reviewer는 **곱**이지 선형결합이 1순위 — 우리 SEL-4는 z-결합/ε-밴드를 썼으니 "Coder-Reviewer의 곱-결합을 z-정규화 합성으로 변형"으로 정직하게 표기.
- **RW 문장**: "We adopt the reverse-likelihood signal of Coder-Reviewer reranking (Zhang et al., ICML 2023), which corrects the degenerate-solution bias of forward-likelihood reranking, but combine it with a consensus-utility term rather than using it standalone."

### 2. Smoothie — [arXiv:2412.04692] → SEL-1의 직계
- **서지(직독)**: "Smoothie: Label Free Language Model Routing", Neel Guha, Mayee F. Chen, Trevor Chow, Ishan S. Khare, Christopher Ré (Stanford). NeurIPS 2024.
- **CLAIM/METRIC(verbatim)**: 임베딩 위 latent-variable graphical model로 label 없이 per-sample 품질 추정. "correctly identifying the optimal model on **9/14 tasks**"; multi-task routing "outperforms…unsupervised routing methods by **up to 10 points** accuracy and supervised routing methods by up to 5.0 points"; NLG에서 학습 가중치-실품질 Spearman 평균 **0.72**; prompt selection "up to 18 points".
- **공정성**: baseline = Random, Best-on-Val(labeled), Labeled-kNN, **PairRM**. 14 task. 일부 비교가 labeled baseline까지 포함 = 공정. 단 "up to 10 points"는 multi-task **routing**의 상한.
- **FIT(SEL-1)**: label-free proposer 품질 prior = SEL-1 "proposer-prior 가중 prop-MBR"의 방법론적 출처. SEL-1이 β=2로 채택(공식 66.48→67.22)된 것이 이 노선의 우리-실측 검증.
- **DIVERGENCE/ERROR**: ★중요 — Smoothie는 **routing**(입력당 모델 1개 선택)이지 후보-집합 selection 아님. 우리 prior report는 §2.4에서 이 차이를 맞게 적었으나 §1 요약은 "최근접 선행"으로만 표기 — paper엔 "routing-not-selection" 차이 명시 필요. **결정적 추가 발견**: Smoothie의 자기-한계 = "its multivariate Gaussian graphical model currently uses a **diagonal covariance matrix. This assumes independent error vectors for each generation**" — 즉 Smoothie도 **소스 간 상관(우리 다수-블록 편향)을 가정으로 배제**. 이는 우리 Novelty #1(소스-상관 보정)을 직접 보강: 가장 가까운 선행조차 상관을 못 다룬다.
- **RW 문장**: "Like Smoothie (Guha et al., NeurIPS 2024) we estimate per-generator quality without labels, but whereas Smoothie routes each input to one model under a diagonal-covariance (independent-error) assumption, we select among a candidate set and explicitly down-weight correlated same-policy votes."

### 3. MBR bias-diversity 분해 — [arXiv:2410.15021] → Novelty #1의 이론 옆자리
- **서지(직독 — ★제목 정정)**: arXiv 현행 제목 = "**Theoretical Aspects of Bias and Diversity in Minimum Bayes Risk Decoding**", Hidetaka Kamigaito, Hiroyuki Deguchi, Yusuke Sakai, Katsuhiko Hayashi, Taro Watanabe (NAIST·U.Tokyo). 우리 prior report가 "Diversity Explains Inference Scaling Laws"를 v2 정식 제목처럼 단 것은 **부정확** — 직독한 ar5iv 본문 제목은 "Theoretical Aspects…"였고 ACL 2025 게재 여부는 본 세션 재확인 못 함(§A).
- **CLAIM/METRIC(verbatim)**: 추정오차 분해 = "(û_i − ū_i)² = **Bias − Diversity**" (diversity는 음수로 **차감**, ensemble 관례). Bias = 인간추정-utility 괴리, Diversity = pseudo-reference 간 utility 추정 분산. 트레이드오프: "When the bias term approaches zero…the diversity term also approaches zero".
- **공정성**: 이론 논문(MT/요약). 실험 WMT19 En-De 85.9→86.1 BLEU(8 model·64 sample) = 작은 효과지만 이론 검증용. 표준 setup.
- **FIT**: 우리 "선별=다양성 함수"(TB §8.9d, P-lora 회귀 기울기 +0.077 SIG)의 이론 대응. proposer-1-vote = pseudo-reference 분포 재가중으로 bias 항 축소.
- **DIVERGENCE/ERROR**: ★우리 Novelty #1을 견고히 함 — **이 논문은 명시적으로 단일 모델 i.i.d. 가정**("y∼P(y|x)", "No analysis of sample correlation within a single sample set", Appendix H가 sampling·metric 다양성 상호작용을 미해결로 명시). 즉 우리 "같은 정책 K샘플=1표, 이종 모델=독립 증거"의 source-aware 보정은 이 분해의 빈 칸이 맞다. prior report §3.3의 부호 표기("bias + diversity")는 **부호 오류** — 원문은 **Bias − Diversity**(다양성 클수록 오차↓). 의미는 같으나 식 인용 시 정정 필수.
- **RW 문장**: "Kamigaito et al.'s bias–diversity decomposition of MBR estimation error (û−ū)²=Bias−Diversity assumes i.i.d. pseudo-references from a single policy; the heterogeneous-pool regime, where same-policy samples are correlated, is left open — the gap our source-aware vote re-weighting fills."

### 4. Multi-Agent Verification (MAV) — [arXiv:2502.20379] → SEL-2 기각 해석 + V-라인 재진입 조건
- **서지(직독)**: "Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers", Shalev Lifshitz, Sheila A. McIlraith, Yilun Du. 2025.
- **CLAIM/METRIC(verbatim)**: Aspect Verifier = "Off-the-shelf LLMs prompted to verify specific aspects…through binary True/False approvals". BoN-MAV = 후보 n개 → m verifier의 binary approval → "selecting the output with the most approvals"(점수 = "sum of the positive votes"). weak-to-strong: "combining weaker verifiers can improve the performance of even stronger generator LLMs". 수치: Gemini-1.5-Flash MATH **BoN-MAV 66.0 vs SC 59.0 vs RM 61.7**.
- **공정성**: ★주의 — **무조건 작동 아님**. GPQA-diamond "BoN-MAV and RM achieve comparable results"(**tie**, Pro 49.0=49.0), HumanEval도 RM과 동급(사실상 역전). MATH/MMLU-Pro만 명확 우세. 표준벤치, 동일 generator 비교 = 공정하나 cherry 위험 있는 셀렉티브 헤드라인.
- **FIT(SEL-2/V-라인)**: ★우리 SEL-2 기각(soft-approval validity vote)의 해석을 확정. MAV 작동 전제 2개를 우리는 못 갖췄다: (1) "we select the subset ℳᵈ⊆ℳ which **maximizes the average performance…on a validation set**"(축 부분집합을 val로 선별) (2) 20 verifier·축 다양성. 우리 SEL-2는 **단일 게이트 신호**(gmem)의 soft화 = 축 다양성 0 → 기각은 모순이 아니라 조건차. SELECTOR_DESIGN §6 V-라인(val 100/test 400 분할 + greedy 축선별)이 정확히 MAV 레시피를 이식한 것.
- **DIVERGENCE/ERROR**: prior report §4 RQ4가 MAV를 "약한 verifier 다수 집계가 SC·RM보다 좋게 스케일"로만 요약하고 **tie/역전 셀을 누락** — paper엔 "on MATH/MMLU-Pro, with held-in validation subset selection"이라는 조건을 반드시 동반(그 조건 없이 인용하면 우리 SEL-2 음성과 모순돼 보임). MAV 자기-한계 명시: 단일 generator·verifier confidence 미반영·2 base LLM뿐.
- **RW 문장**: "BoN-MAV (Lifshitz et al., 2025) shows aggregated weak-verifier approvals can beat self-consistency, but only with axis-diverse verifiers and held-in validation subset selection — conditions absent in our single-axis soft-gate, which (consistent with their HumanEval/GPQA ties) underperformed."

### 5. Imperfect-verifier ceiling — [arXiv:2411.17501] → SEL 설계원칙 #2의 이론 닻
- **서지(직독 — ★제목 확인)**: ar5iv 본문 제목 = "**Inference Scaling fLaws: The Limits of LLM Resampling with Imperfect Verifiers**", Benedikt Stroebl, Sayash Kapoor, Arvind Narayanan (Princeton). prior report가 v3 제목을 "The Limits of Inference Scaling Through Resampling"로 적은 것은 **본 세션 직독본과 불일치** — 직독본은 "Inference Scaling fLaws…"였다(§A에 버전 확인 권고).
- **CLAIM/METRIC(verbatim)**: "there is no free lunch…indefinite accuracy improvement through resampling can only be realized if the 'verifier' is perfect." 불완전 verifier는 FP>0 → "Resampling cannot decrease this probability, so it imposes an **upper bound** to the accuracy." "for realistic cost-benefit ratios, the optimal number of samples K is **finite and very low**"(본문 K≤5). "**weaker models exhibit a higher probability of producing false positives**".
- **공정성**: HumanEval+/MBPP+ 코딩, human-written 테스트. 표준벤치. 단 저자 명시 한계 = 코딩 한정·LLM-생성 테스트면 결론 바뀔 수 있음.
- **FIT(원칙 #2)**: 우리 "검증기-주도 선별 단독 라인은 구조적 천장 → verifier는 영원히 veto"의 직접 근거. 게이트 역선택 = FP가 후보 간 계통적(이종 풀)인 악화판. "optimal K very low"는 우리 K=14가 이미 적정구간 → 이득은 샘플증설 아닌 선별기 품질이라는 결론과 정합.
- **DIVERGENCE/ERROR**: prior report가 "최적 샘플수 10 미만"이라 적었는데 직독본은 더 강하게 **"finite and very low", 다수 케이스 K≤5** — "<10"은 약하게 인용한 것. 우리 K=14는 "very low" 구간을 다소 초과하나, 우리 풀은 verifier-주도가 아니라 합의-주도라 정리의 천장 가정과 직접 충돌하지 않음(우리 쪽이 더 유리).
- **RW 문장**: "Stroebl et al. (2024) prove imperfect verifiers (FP>0) cap resampling accuracy regardless of budget and that optimal sample counts are very low; our anti-selection finding is the systematic-FP instance of this bound, motivating consensus-as-chooser with verifier-as-veto rather than verifier-led selection."

---

## TIER 2 — 노선 정당화 (제약·베이스라인을 떠받침)

### 6. Self-Consistency — [arXiv:2203.11171]
- **서지(직독)**: "Self-Consistency Improves Chain of Thought Reasoning in Language Models", Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou (Google). ICLR 2023.
- **CLAIM(verbatim)**: diverse reasoning path 샘플 → "marginalize…aggregate by choosing the most consistent answer". GSM8K "**+17.9%**", SVAMP +11.0%, AQuA +12.2%, StrategyQA +6.4%, ARC-challenge +3.9%(전부 abs).
- **공정성**: 표준 추론벤치, 동일 모델 greedy 대비. 공정. 단 **answer-level 정확매칭** 전제 = 우리 구조 DAG엔 직접 적용 불가(USC가 이 갭을 메움).
- **FIT**: MBR/prop-MBR의 모(母) 기법(Bertsch가 SC를 MBR 특수례로 증명). Δhetero(+13.6)의 동질판 직관.
- **RW 문장**: "Self-consistency (Wang et al., ICLR 2023) marginalizes over reasoning paths via exact-match voting; our setting generalizes this to structured DAGs where exact match fails, via a graph-utility MBR."

### 7. Universal Self-Consistency (USC) — [arXiv:2311.17311]
- **서지(직독)**: "Universal Self-Consistency for Large Language Model Generation", Xinyun Chen 외 9인(Google). 2023.
- **CLAIM(verbatim)**: LLM이 직접 "select the most consistent response among multiple candidates"(실행·정확매칭 불요). "USC **matches the execution-based self-consistency** performance on both benchmarks, while USC does not utilize code execution"(BIRD-SQL·ARCADE).
- **공정성/한계(verbatim)**: ★우리 SEL-q⑹(USC-식 7B 일괄선택)의 직접 경고 — "number of samples supported by USC is **bounded by the context length**"; "accuracy on GSM8K **decreases with 16 samples**"(위치/장문 약점); "most consistent response is **not necessarily the best one**".
- **FIT(SEL 후순위 베이스라인)**: 실행 없이 합의선택 = 우리 gold-free·no-sandbox 제약에 부합. 단 K=14 DAG는 컨텍스트 부담 → 우리가 USC를 "베이스라인 표 채우기"로만 둔 판단을 16-sample 열화 증거가 뒷받침.
- **RW 문장**: "USC (Chen et al., 2023) lets the model pick the most consistent free-form output without execution, but its accuracy degrades past ~16 candidates due to long-context position bias — why we use it only as a baseline for our K=14 pool."

### 8. It's MBR All the Way Down — [arXiv:2310.01387]
- **서지(직독 — 부제 보강)**: "**It's MBR All the Way Down: Modern Generation Techniques Through the Lens of Minimum Bayes Risk**", Amanda Bertsch, Alex Xie, Graham Neubig, Matthew R. Gormley (CMU). 2023.
- **CLAIM(verbatim)**: "several recent methods…can be written as **special cases of MBR**"(self-consistency, range voting, output ensembling). "MBR provides reliable several-point improvements across metrics for a wide variety of tasks **without any additional data or training**". 한계: MBR "inherits the weaknesses and biases of the gain metric…susceptible to overfitting to the metric".
- **공정성**: 통합/리뷰성 논문, CNN/DM ROUGE-1 46.89 vs beam 43.x. 표준. 공정.
- **FIT**: 우리 "utility=평가척도 동형(edge-F1)이 우월"(Novelty #3)의 모(母) 프레임 — utility가 metric일수록 bias↓. SC/voting을 MBR으로 통합하는 우리 서술의 인용 근거.
- **DIVERGENCE**: 이 논문의 "metric overfitting" 경고 = 우리 edge-F1 utility의 잠재 리스크(채점 metric과 utility를 동형으로 두면 metric-게이밍 위험) — paper의 limitations에 정직 기재할 것.
- **RW 문장**: "We follow Bertsch et al. (2023) in viewing voting/ensembling as MBR special cases, and instantiate MBR with a graph-utility homomorphic to the evaluation metric — accepting their caveat that metric-homomorphic utility risks metric overfitting."

### 9. MBR-Exec — [arXiv:2204.11454]
- **서지(직독)**: "Natural Language to Code Translation with Execution", Freda Shi, Daniel Fried, Marjan Ghazvininejad, Luke Zettlemoyer, Sida I. Wang (Meta AI). EMNLP 2022.
- **CLAIM(verbatim)**: 실행결과로 의미동등 근사 후 MBR. "MBR-exec **consistently improves over all execution-unaware selection methods**". MBPP 58.2 vs 47.x, Spider 63.6 vs 50.8, NL2Bash 58.5 vs 53.0.
- **공정성**: 표준 코드벤치, frozen Codex. 공정.
- **FIT(Novelty #3)**: 우리 "구조 출력의 metric-homomorphic utility"는 MBR-Exec(실행 utility)과 n-gram MBR 사이의 빈 칸 — sandbox 없는 DAG에서 그래프 정적분석(타입전파·슬롯체결, V-1의 A4/A5)이 "유사-실행" 대용물. 직접 선행 없음.
- **RW 문장**: "Where MBR-Exec (Shi et al., EMNLP 2022) marginalizes over execution-equivalent programs, our no-sandbox structured setting substitutes static DAG analysis (type propagation, slot binding) as a 'pseudo-execution' utility."

### 10. PoLL — [arXiv:2404.18796]
- **서지(직독)**: "Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models", Pat Verga 외(Cohere). 2024.
- **CLAIM(verbatim)**: 패널 = **Command R (35B) + Claude Haiku + GPT-3.5**. "outperforms a single large judge"(NQ κ **0.763 vs GPT-4 0.627**; Arena Pearson 0.917 vs 0.817), "**over seven times less expensive**" than GPT-4 Turbo. "PoLL exhibits less intra-model bias due to…disjoint model families".
- **공정성**: ★주의 — 패널이 **Haiku(35B급 proprietary)·GPT-3.5 포함** = "소형 오픈 온프렘"이 아님. 우리 ≤7B/on-prem 제약과 **부분 불일치**. "less expensive"는 GPT-4 단일 대비지 7B-only 대비 아님.
- **FIT(SEL-5)**: "소형 다수 ≥ 대형 단일 judge" + "judge 가문 다양성이 intra-bias↓" = 우리 7B pairwise judge·H6 이종 judge 재활용 아이디어 정당화.
- **DIVERGENCE/ERROR**: ★prior report는 "소형 3종 패널"로 적었으나 그 "소형"이 Haiku·GPT-3.5(API·비-7B) = **on-prem 주권-leg 논거로는 직접 쓸 수 없음**. paper에서 PoLL을 "frontier 불필요"의 일반 근거로만 인용하고 "on-prem 7B 검증은 본 연구 기여"로 분리할 것.
- **RW 문장**: "PoLL (Verga et al., 2024) shows a panel of smaller, family-diverse judges beats a single GPT-4 judge at ~1/7 cost; we extend the panel-diversity argument to an on-prem ≤7B regime its API-based panel (Haiku, GPT-3.5) does not cover."

### 11. Prometheus 2 — [arXiv:2405.01535]
- **서지(직독)**: "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models", Seungone Kim, Juyoung Suk, Shayne Longpre, Bill Yuchen Lin 외. EMNLP 2024.
- **CLAIM(verbatim)**: Mistral/Mixtral 기반 7B·8x7B, direct assessment+pairwise 겸용, 두 평가자 가중-병합(α=0.5/DARE). "highest correlation with both human evaluators and proprietary LM-based judges"(Vicuna 0.64+, MT-Bench 0.66+; HHH pairwise 85.52%). 한계: joint training이 "negative task transfer".
- **공정성**: 오픈 evaluator 중 최고 상관(표준 평가벤치). 공정 — 단 일반텍스트 평가 학습 → 우리 DAG 도메인 적응 필요.
- **FIT(SEL-5)**: 7B pairwise judge 기성품 후보의 실재 증명(우리 train-gold로 LoRA judge fine-tune의 대안).
- **RW 문장**: "Prometheus 2 (Kim et al., EMNLP 2024) demonstrates a 7B open evaluator can match proprietary-judge correlation, supporting our ≤7B pairwise-judge tier, though its general-text training requires DAG-domain adaptation."

### 12. Self-Certainty (Scalable BoN) — [arXiv:2502.18581]
- **서지(직독)**: "Scalable Best-of-N Selection for Large Language Models via Self-Certainty", Zhewei Kang, Xuandong Zhao, Dawn Song. ICML(arXiv v3; prior report의 "NeurIPS 2025"은 본 세션 미확인 — §A).
- **CLAIM(verbatim)**: self-certainty = uniform 대비 KL-divergence 기반("−1/nV ∑∑ log(V·p)"). "scales efficiently with increasing sample size N, mirroring reward models…without their computational burden", "generalizes effectively to open-ended responses, where self-consistency is inapplicable". 한계: "self-certainty alone underperforms self-consistency on questions with unique answers".
- **공정성**: ★중요 직독 발견 — **전 실험이 단일 모델**(Llama-3.1-8B/DeepSeek-R1-Distill/Qwen-2.5-Coder-32B). "**does NOT address or compare self-certainty scores across different models…No discussion of heterogeneous model calibration**". 표준벤치, 동일모델. 공정하나 이종 비교는 다루지 않음.
- **FIT/DIVERGENCE**: 우리 prior report가 "이종 logprob 스케일 비교 불능(보정 필요)"이라 적은 것이 **정확** — 이 논문이 명시적으로 cross-model 보정을 다루지 않음을 확인. 우리 풀은 **같은 base의 LoRA**라 logprob 스케일이 예외적으로 비교 가능 = 0-추가-pass 보조신호로 합성 가능(우리 고유 이점). "unique-answer에선 SC에 밀림"은 우리 DAG(unique-ish 구조)에선 단독 사용 금지 신호.
- **RW 문장**: "Self-certainty (Kang et al.) gives a reward-model-free, N-scalable confidence from output distributions but is validated only intra-model; our shared-base LoRA pool is a rare regime where its logprob signal is cross-candidate comparable."

---

## TIER 3 — 맥락/경계 (선행 인정·차별점)

### 13. Symbolic Mixture-of-Experts — [arXiv:2503.05641]
- **서지(직독 — ★제목 정정)**: 현행 arXiv 제목 = "**Symbolic Mixture-of-Experts: Adaptive Skill-based Routing for Heterogeneous Reasoning**", Justin Chih-Yao Chen, Sukwon Yun, Elias Stengel-Eskin, Tianlong Chen, Mohit Bansal (UNC). prior report의 "Skill-Based Mixture-of-Experts(v4)·ICML 2026"은 **직독본과 불일치** — 직독 ar5iv 제목은 여전히 "Symbolic Mixture-of-Experts"이며 "Skill-Based" 리네임·ICML 2026 게재는 확인 못 함(§A, 인용 시 정정).
- **CLAIM(verbatim)**: instance별 skill 추론→k expert 모집→aggregator 합성. "outperforms the most competitive multi-agent baseline, **Self-MoA, by 8.15% (absolute) on average**"(MMLU-Pro/AIME/GPQA/MedMCQA). "integrate **16 models on a single GPU** with a time cost comparable to…4 GPUs".
- **공정성**: Self-MoA 대비 +8.15(표준 추론벤치). 공정.
- **FIT**: 우리 멀티-LoRA 1서버 운용의 학술 대응(16 model/1 GPU = H6+AR8 배칭 정당화). 단 라우팅+aggregator 융합 ≠ 우리 순수 selection.
- **RW 문장**: "Symbolic-MoE (Chen et al., 2025) recruits heterogeneous experts per instance and batches 16 models on one GPU — the systems analogue of our multi-LoRA single-server pool, though it fuses rather than selects."

### 14. Mixture-of-Agents — [arXiv:2406.04692]
- **서지(직독)**: "Mixture-of-Agents Enhances Large Language Model Capabilities", Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, James Zou. 2024.
- **CLAIM(verbatim)**: 계층 proposer→aggregator. "MoA using only open-source LLMs…**AlpacaEval 2.0…65.1% compared to 57.5% by GPT-4 Omni**".
- **공정성**: AlpacaEval 2.0(LLM-judge 벤치) — open-only가 GPT-4o 추월. 공정하나 judge-기반 벤치 특성 유의.
- **FIT(설계원칙 #4, fusion 기각)**: ★우리 fusion-기각의 직접 근거 — "the aggregator does **not simply select**…but potentially performs sophisticated aggregation…actively generates new synthesized text". 즉 MoA는 새 텍스트 생성 = 우리 evaluator의 구조-동형성·합법 DAG 제약과 충돌. 우리가 selection을 택한 이유의 외부 대비점.
- **RW 문장**: "MoA (Wang et al., 2024) achieves SOTA by generative fusion of agent outputs, but fusion synthesizes new text — incompatible with our structural-validity constraint, so we restrict to after-inference selection."

### 15. More Agents Is All You Need — [arXiv:2402.05120]
- **서지(직독)**: Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, Deheng Ye. 2024(prior report의 "TMLR 2024" 게재는 본 세션 미확인 — §A).
- **CLAIM(verbatim)**: "performance…scales with the number of agents"; "relative performance gain η becomes more significant when the relative difficulty…increases"(어려운 task 28–200% vs 쉬운 8–16%).
- **FIT**: 우리 Δhetero=+13.6의 일반판(sampling-and-voting 스케일). 단 동질 샘플 — 우리 이종-풀 보정과 구별.
- **RW 문장**: "Consistent with More Agents (Li et al., 2024), our pool gains scale with candidate count, but our heterogeneous-source correction targets the correlated-vote regime their homogeneous sampling does not."

### 16. Generative Verifiers (GenRM) — [arXiv:2408.15240]
- **서지(직독)**: "Generative Verifiers: Reward Modeling as Next-Token Prediction", Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, Rishabh Agarwal. ICLR 2025(arXiv).
- **CLAIM(verbatim — ★숫자 정정)**: verifier를 Yes/No next-token 생성으로; GenRM-CoT는 CoT 후 다수결. GSM8K "Gemma-9B GenRM-CoT…verify…Gemini 1.0 Pro…20% improvement…(**73%→92.8%**)". "scales favorably as we increase dataset size as well as model capacity".
- **공정성**: 표준(GSM8K/알고리즘). 공정.
- **DIVERGENCE/ERROR**: ★prior report가 "GSM8K 73→**93.4%**"라 적었으나 직독본은 **73→92.8%**(Gemma-9B가 Gemini 1.0 Pro 출력 검증). **수치 오인용 정정 필요**.
- **FIT(SEL-5)**: 우리가 ranker를 학습한다면 분류 head보다 생성식 검증이 7B에서 유리하다는 처방.
- **RW 문장**: "If we fine-tune a selector, GenRM (Zhang et al., ICLR 2025) argues a generative next-token verifier scales better than a classification head — relevant to our SEL-5 learned-judge option."

### 17. Lightweight Reranking (UCS) — [arXiv:2307.06857]
- **서지(직독)**: "Lightweight reranking for language model generations", Siddhartha Jain, Xiaofei Ma, Anoop Deoras, Bing Xiang (AWS). ACL 2024.
- **CLAIM(verbatim)**: UCS = Unigram Consistency Score, "UCS(i,j)=(1/|V|) v_i·v_j"(토큰 존재 벡터 내적). self-consistency를 실행·외부모델 없이 open-ended로 일반화. Codex002 HumanEval 0.435→0.568(Consensus-WUCS).
- **FIT(Novelty #3 경계)**: 표면 n-gram MBR의 대표 — 우리 metric-homomorphic 그래프 utility가 메우는 빈 칸의 한쪽 극(다른 극=MBR-Exec).
- **RW 문장**: "UCS (Jain et al., ACL 2024) reranks by surface n-gram overlap; we replace surface similarity with metric-homomorphic graph utility (edge-F1) to respect DAG structure."

### 18. AlphaCode — [arXiv:2203.07814]
- **서지(직독)**: "Competition-Level Code Generation with AlphaCode", Yujia Li 외(DeepMind). Science 2022.
- **CLAIM(verbatim)**: filter(example test 통과)→learned test-input 생성→"clustering on program behaviour…selecting one solution from each cluster from largest to smallest performed best, perhaps because…correct solutions tend to behave the same and…are grouped into larger clusters".
- **FIT(설계원칙 #1)**: "filter=veto, cluster/consensus=chooser" 분업의 원형. 우리 §8.9b 분업이 재발견 아닌 정합임을 박제. 단 AlphaCode는 실행거동 클러스터(sandbox 필요) — 우리는 정적 그래프 utility로 대체.
- **RW 문장**: "Our filter-then-consensus pipeline mirrors AlphaCode's filter→behavioral-cluster→representative scheme (Li et al., 2022), but substitutes static graph-behavior for execution clustering."

### 19. DOCE — [arXiv:2408.13745]
- **서지(직독)**: "DOCE: Finding the Sweet Spot for Execution-Based Code Generation", Haau-Sing Li, Patrick Fernandes, Iryna Gurevych, André F. T. Martins. 2024.
- **CLAIM(verbatim)**: "the importance of filtering based on **trial unit tests, a commonly used technique whose effect has not been reported in previous works**", "simple and effective". MBR-Exec+filter 74.5 vs no-filter 52.1(CodeLlama-7B).
- **FIT(설계원칙 #1)**: "filter가 가장 간과된 단순·유효 전략" = 우리 validity-veto의 외부 검증. 단 trial-test=실행 필요(우리는 정적 게이트).
- **RW 문장**: "DOCE (Li et al., 2024) finds trial-test filtering an under-reported but strong lever; absent execution, our static validity gate plays the veto role it assigns to test filtering."

### 20. Large Language Monkeys — [arXiv:2407.21787]
- **서지(직독)**: Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini. 2024.
- **CLAIM(verbatim)**: coverage "scales…over four orders of magnitude"(c≈exp(ak^−b)). ★"all sample selection methods **fail to reach the coverage upper bound and saturate before reaching 100 samples**" while coverage continues to rise. SWE-bench Lite 15.9%(1)→56%(250).
- **FIT(원칙 #2)**: oracle(coverage)는 계속 오르나 verifier-free 선별은 100샘플 전 plateau = 우리 "이득의 남은 자리는 샘플증설 아닌 선별기"의 직접 근거. Δhetero +13.6(oracle 천장 vs 단일-best)의 일반판.
- **RW 문장**: "Brown et al. (2024) show coverage scales for 4 orders of magnitude while verifier-free selection plateaus before 100 samples — locating our bottleneck at selection quality, not sample count."

### 21. MAP 부적합성 — [arXiv:2005.10283]
- **서지(직독)**: "Is MAP Decoding All You Need? The Inadequacy of the Mode in Neural Machine Translation", Bryan Eikema, Wilker Aziz (U.Amsterdam). COLING 2020.
- **CLAIM(verbatim)**: "the most likely translations…accumulate so little probability mass that the mode can be considered **essentially arbitrary**"; "some of the known pathologies…are due to **MAP decoding and not to NMT's statistical assumptions**"; MBR을 holistic 대안으로.
- **FIT(설계원칙 #3)**: likelihood-단독 재랭킹 금지·게이트 역선택의 이론적 뿌리(MAP의 서열이 품질 서열 아님). Coder-Reviewer 병리의 NMT 원조.
- **RW 문장**: "Eikema & Aziz (COLING 2020) attribute likelihood-decoding pathologies to the MAP rule, not the model — the root of the degenerate-preference we observe in deterministic gate anti-selection."

### 22. Semantic Uncertainty — [arXiv:2302.09664]
- **서지(직독)**: "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation", Lorenz Kuhn, Yarin Gal, Sebastian Farquhar (Oxford). ICLR 2023.
- **CLAIM(verbatim)**: 양방향 함의로 의미-클러스터링 후 클러스터 분포 엔트로피; unsupervised·단일모델·무수정(DeBERTa-large MNLI; TriviaQA 92.7%/CoQA 95.3% 클러스터 정확도).
- **FIT(SEL-3/abstention)**: "클러스터 분포 엔트로피=불확실성" = 우리 margin-기반 abstention(승자-클러스터 점유율, 1-2위 갭)의 신호 출처. SEL-3가 작동(TB §8.9c)한 것이 이 노선 검증.
- **RW 문장**: "We reuse semantic-entropy-style cluster dispersion (Kuhn et al., ICLR 2023) as the confidence signal for margin-based abstention over candidate clusters."

---

## §A. prior selector_lit 리포트 정정 (본 세션 직독 기준)

1. **[Kamigaito+24] 제목**: prior report는 "Diversity Explains Inference Scaling Laws"를 v2 정식 제목으로 표기 + "ACL 2025". 직독 ar5iv 본문 제목 = "**Theoretical Aspects of Bias and Diversity in Minimum Bayes Risk Decoding**". → 인용 제목을 이것으로 정정; "Diversity Explains…"는 별 제목/버전으로 미확정 처리. **ACL 2025 게재 여부 미재확인**.
2. **[Kamigaito+24] 식 부호**: prior report §RQ1이 "bias + diversity"로 적었으나 원문은 "(û−ū)² = **Bias − Diversity**"(다양성 차감). 의미 동일하나 식 인용 시 부호 정정.
3. **[Zhang+24 GenRM] 수치**: prior report "GSM8K 73→**93.4%**" = 오인용. 직독 원문 = "**73%→92.8%**"(Gemma-9B GenRM-CoT가 Gemini 1.0 Pro 검증, 20% 개선). 정정.
4. **[Stroebl+24] 제목·K**: prior report v3 제목 "The Limits of Inference Scaling Through Resampling". 직독 ar5iv 본문 제목 = "**Inference Scaling fLaws: The Limits of LLM Resampling with Imperfect Verifiers**". 또 "최적 샘플수 10 미만"은 약함 — 원문 "**finite and very low**"(다수 K≤5). 제목·강도 정정(버전 메타 §A 미확정).
5. **[Chen+25-MoE] 제목·venue**: prior report "Skill-Based Mixture-of-Experts(v4)·ICML 2026". 직독 ar5iv 제목 = "**Symbolic Mixture-of-Experts: Adaptive Skill-based Routing for Heterogeneous Reasoning**". "Skill-Based" 리네임·ICML 2026 = **미확인** → 현행 제목으로 인용.
6. **[Jiang+23 LLM-Blender] ranker 백본**: prior report "RoBERTa-급". 직독 = PairRanker **DeBERTa(400M)**, GenFuser=Flan-T5-XL(3B). 정정(RoBERTa→DeBERTa). 또 PairRanker는 O(n²) 호출 비용 명시 = 우리 shortlist 압축 정당화.
7. **[Verga+24 PoLL] 패널 "소형"의 실체**: prior report "소형 3종 패널". 실제 = Command-R(35B)+**Claude Haiku**+**GPT-3.5**(=API·proprietary 포함). "on-prem ≤7B 주권-leg" 논거로 직접 쓸 수 없음 — frontier-judge-불필요의 일반 근거로만 인용하고 7B-on-prem 검증은 우리 기여로 분리.
8. **[Kang+25] venue**: "NeurIPS 2025" 미재확인(직독은 "ICML" 표기 — §A 미확정).
9. **[Smoothie] 한계 보강(누락)**: diagonal-covariance = 독립오차 가정 → 소스 상관 미처리. 우리 Novelty #1을 강화하는 결정적 한계인데 prior report 미언급 — 반드시 추가.
10. **[MAV] tie/역전 누락**: prior report가 MAV를 "SC·RM보다 좋게 스케일"로만 요약. 실제 GPQA tie·HumanEval 동급. 인용 시 "with axis-diverse verifiers + held-in subset selection" 조건 동반 필수(없으면 우리 SEL-2 음성과 모순돼 보임).
11. **[Coder-Reviewer] "+17%p"**: "**up to** 17% absolute"(상한, 평균 아님) — "up to" 유지.

## §B. 우리 thesis에 가장 결정적인 3편

1. **Smoothie [2412.04692]** — SEL-1(채택·공식 +0.74pp)의 방법론 출처이자, diagonal-covariance 한계가 우리 source-correlation Novelty의 공백을 직접 입증. fit+novelty 양면 최대 load-bearing.
2. **MBR bias-diversity 분해 [2410.15021]** — "선별=다양성 함수"(우리 회귀 +0.077 SIG, D-oracide 게이트)의 이론 닻이며, **단일-모델 i.i.d. 가정 명시**가 우리 이종-풀 보정을 논문화-가능한 빈 칸으로 확정.
3. **Coder-Reviewer [2211.16490]** — SEL-4(신기록 스택 +10.3pp/+12.9pp의 한 축)의 직계이며, "degenerate-solution 선호" verbatim이 우리 게이트 역선택 병리에 문헌-동형의 이름을 부여(설계원칙 #3).
보강 닻: **Stroebl+24 [2411.17501]**(verifier 천장 = 원칙 #2)이 4번째 결정적.

## §C. DROP 권고 (불공정/제약-불일치 인용)

- **PoLL [2404.18796]을 "on-prem 소형 judge" 직접 근거로 쓰는 것은 DROP**. 패널이 Haiku·GPT-3.5(proprietary API) 포함 = 우리 ≤7B/망분리 제약과 불일치. "frontier 단일 judge 불필요"의 일반 근거로만 유지, 주권-leg 논거에서는 제거. (논문 자체는 유효 — 우리 *프레이밍*이 불공정해질 위험.)
- **MAV [2502.20379]을 무조건 "약-verifier 집계가 SC/RM 우월"로 인용하는 것은 부분 DROP** — GPQA/HumanEval tie를 누락하면 cherry-pick. 반드시 조건(축 다양성+val subset 선별) 동반, 아니면 우리 SEL-2 음성과 충돌.
- 그 외 핵심 인용 중 "부당승/커스텀벤치" 사유의 전면 DROP 대상 없음(전부 표준벤치·동일-모델 baseline). MoA/Symbolic-MoE는 fusion·routing이라 우리와 task 사분면이 다름 — selection 차별점만 주장하면 유효.
