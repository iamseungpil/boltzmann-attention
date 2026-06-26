# 아키텍처/디코딩 개입으로 PLANNING·PROCEDURE·TOOL-SELECTION 개선 — 적대검증 서베이 (diffusion 제외)

> 2026-06-14. 작성자=research agent. 적대검증 규율: 모든 1차출처 arXiv abs/HTML 확인 후만 인용·핵심수치 verbatim·model size & fairness flag 명기·메모리 인용 금지(미검증=Unverified leads).
> 범위: "표준 left-to-right AR을 넘어 구조적/agentic 생성을 개선하는 아키텍처·디코딩 개입" 7축. **diffusion LM은 명시적 제외**(별도 `research_diversity_*`·디퓨전 라인 참조).
> 이미 우리 문서가 커버한 라인(아래 §0)은 "already in our docs"로 표기하고 **2024-2026 신규·미커버**에 새 가치 집중.

---

## §0. 이미 우리 문서가 커버한 것 (재보고 금지, 좌표만)
- **depth/iteration-recurrence** (Universal TF 1807.03819 · Looped/Giannou 2301.13196 · Huginn 2502.05171 · RELAY 2502.08482 · n-RASP-L 2409.15647 · Xu&Sato "To CoT or To Loop?" 2505.19245): `SEARCH_INTERNALIZATION_LITREVIEW.md §9`. **핵심 결론(승계)**: pretrained-7B latent-recurrence retrofit은 오늘 방법 無(Huginn=from-scratch 800B토큰·LoRA 비호환). 결정론 DAG 평가엔 loop이 이론적 정답이나 우리 병목(gather/도구선택)이 serial-depth인지 미확인.
- **탐색 내재화** (Searchformer 2402.14083 · TS-LLM 2309.17179 · ReST-MCTS* 2406.03816 · Math-Shepherd 2312.08935 · AlphaLLM 2404.12253): `SEARCH_INTERNALIZATION_LITREVIEW.md §2-6`. Searchformer=교사초과 키스톤(단 from-scratch enc-dec).
- **grounded AND/OR 트리평가** (Feng 2305.15408 · Abbe 2406.06467 · CRvNN/Beam-Tree RvNN 2305.19999 등): `RUNG1_V3_TREE_EVAL_LITREVIEW.md`.
- **xattn 메모리 주입** (RETRO·Flamingo·CALM·G-Retriever): `EXPERIMENT_DESIGN.md §3.10` B5* 3단 사다리.

본 서베이의 신규 기여 = **latent CoT(축2)·planning/pause/filler 토큰(축3)·非AR 非diffusion 생성(축4)·structure-aware DAG 디코딩(축5)·tool-as-token/MoE tool routing(축6)·value-guided decoding(축7)** + 이들이 우리 "different-AND-right diversity" / gold-free selection 풀에 들어갈 수 있는가.

---

## §1. 핵심 답변 요약 (강한 증거 → 과대주장 랭킹 + 우리 설정 적합도)

우리 설정 = **Qwen2.5-7B + LoRA, K-sample + gold-free 선별, JSON-DAG(set/graph of tool calls) 출력**. "재학습 불필요(decoding-time)"와 "vLLM 멀티-LoRA 배포 호환"이 채택 1순위 필터.

**증거 강도 랭킹 (planning/tool gain 직접증거 기준):**

| 순위 | 라인 | 증거 강도 | 우리 적합도 | 한 줄 |
|---|---|---|---|---|
| **1** | **Grammar/constrained decoding (XGrammar 2411.15100 등)** | ★★★ (배포검증·우리가 이미 의존) | **이미 우리 것** | JSON-DAG 유효성을 *아키텍처 레이어*로 보장 = K-sample 풀의 "right(파싱가능)" 하한. 디코딩-타임·재학습0·vLLM 호환. |
| **2** | **Stream-of-Search (2404.03683)** | ★★★ (from-scratch지만 명확·AR 우위) | 중 (trace-distill로 SFT 적용) | 탐색 trace(실패·백트랙 포함)를 텍스트로 학습 → +25% 탐색정확도·교사 미해결 36% 해결. 단 250M from-scratch. |
| **3** | **Pause/filler tokens (2310.02226 · 2404.15758)** | ★★ (gain 작음·합성과제·학습난) | 낮~중 (LoRA로 토큰추가 가능하나 dense supervision 필요) | "추가 계산폭"은 실재하나 1B서 GSM8K +1%p에 그침·filler는 학습 매우 어려움. |
| **4** | **Latent CoT / Coconut (2412.06769)** | ★★ (수학서 AR에 패배·불안정) | **낮음** (retrofit 불가·우리엔 distraction) | ProsQA서 CoT 압도(97.0 vs 77.5)이나 GSM8K서 **패배**(34.1 vs 42.9). GPT-2급·학습 불안정. |
| **5** | **Tool-as-token / dense tool routing (ToolkenGPT 2305.11554)** | ★★ (frozen LLM·embedding만) | 중 (특허 트랙과 정합) | massive-tool 선택을 toolken 임베딩으로. frozen-LLM·재학습0 도구추가. 우리 도구수 작아 한계효용↓. |
| **6** | **Any-order / 非AR 非diffusion 생성 (A3 2601.13228 · Insertion 1902.03249 · SUNDAE 2112.06749)** | ★★ (A3는 신선·검증중) | 중 (개념적 닻; 우리 set-출력에 직결) | **A3: 순수 AR이 any-order 생성서 diffusion과 동급/우위** → "diffusion 없이도 unordered set 생성·다양성 가능" 닻. |
| **7** | **Value-guided / PRM decoding (ReST-MCTS* 등 — already in docs)** | ★★ (math 한정) | 중 (우리 결정론 검증기와 정합) | 우리 gold-free selector가 사실상 outcome-verifier 역할 → 별도 value-head는 보완. |

**과대주장·unfair 경고(skeptical):**
- **"우리 새 아키텍처가 AR을 이긴다"는 대부분 from-scratch·합성·소형(GPT-2/250M)·custom-bench**. Coconut(GPT-2)·SoS(250M GPT-Neo)·Planning Tokens(Phi-1.5/Llama2)·pause(1B). **7B-LoRA-tool-DAG 전이는 0건.**
- **Coconut**: ProsQA 97.0 헤드라인은 강하나 **같은 논문서 GSM8K는 CoT에 짐**(34.1<42.9) — "planning이면 우위, math면 열위"의 좁은 조건. 우리 JSON-DAG는 어느쪽?
- **Transformers Struggle to Learn to Search (2412.04703, He He 그룹)**: "그래프 크기↑서 학습 어려워지고 **파라미터 늘려도 해소 안 됨**, CoT도 큰 그래프선 못 고침." = 탐색-내재화 라인 전반의 천장 경고.
- **Latent-CoT survey (2505.16782v2)**: "현 방법은 여전히 explicit CoT에 **열위**, 주로 **학습 불안정** 때문" — latent reasoning 전반의 정직한 현주소.

---

## §2. 축별 발견 (검증 인용 + fairness flag + verbatim 수치)

### 축 2 — Latent reasoning / continuous thought
- **Coconut — "Training LLMs to Reason in a Continuous Latent Space"** (arXiv **2412.06769v3**, Hao·Sukhbaatar·Su·Li·Hu·Weston·Tian; v1 2024-12-09, last rev 2025-11-03). Base=**GPT-2**. 메커니즘: 마지막 hidden state를 디코딩 않고 다음 입력 임베딩으로 재주입(언어모드↔잠재모드). "continuous thoughts can encode multiple alternative next steps, allowing the model to perform a **breadth-first search (BFS)** rather than committing prematurely to a single deterministic path."
  - **verbatim**: ProsQA — CoT **77.5±1.9** vs Coconut **97.0±0.3**. GSM8k — CoT **42.9±0.2** vs Coconut **34.1±1.5**.
  - **FAIRNESS**: ⚠️ same-size 비교(둘 다 GPT-2 SFT)지만 **GSM8k서 AR-CoT에 패배**. planning-heavy(ProsQA)만 우위. 비표준 bench(ProsQA=합성). **retrofit 불가**(아키텍처 학습 필요).
- **Token Assorted** (2502.03275, VQ-VAE 잠재토큰+텍스트 혼합으로 trace 단축) — 효율 라인, planning gain은 부차. 인용만.
- **Latent-CoT survey** (2505.16782v2): "Current methods still underperform explicit CoT approaches, largely due to the **instability of training**." "models trained with latent CoT techniques often **struggle with novel problem structures**." → 일반화 취약·해석불가 명시.
- **판정(우리)**: ❌ **distraction**. 7B-LoRA retrofit 경로 없음·math서 AR에 짐·우리 출력은 verifiable JSON(잠재가 아니라 *명시* 구조가 검증 대상). Coconut의 "BFS-in-latent"는 우리 K-sample 다양성을 *명시적으로* 이미 달성.

### 축 3 — Explicit search/planning baked-in + pause/filler/planning tokens
- **Stream of Search (SoS)** (arXiv **2404.03683**, Gandhi·Lee·Grand·Liu·Cheng·Sharma·Goodman). 모델=**GPT-Neo 250M, from scratch**, 50만 탐색 trace(diverse heuristic solvers), task=**Countdown**(game-of-24 일반화).
  - **verbatim**: "SoS pretraining increases search accuracy by **25%** over models trained to predict only the optimal search trajectory." 그리고 STaR+APA 후 "solve **36%** of previously unsolved problems, including problems that cannot be solved by any of the heuristic solvers."
  - **FAIRNESS**: ✅ 동일아키텍처 baseline(optimal-only) 대비 명확한 우위·교사초과까지. ⚠️ **from-scratch 250M·단일 합성 도메인**. 7B 전이 미검증. (=`SEARCH_INTERNALIZATION` Searchformer와 같은 패러다임: 실패 trace 포함이 핵심.)
- **System-1.x** (2407.14414): fast/slow planning 균형 학습 — SoS 계열, 인용 리드.
- **Think before you speak: Pause Tokens** (arXiv **2310.02226**, Goyal et al., ICLR'24). Decoder-only **130M·1B**, C4 causal pretrain.
  - **verbatim**: 1B 모델서 8개 task gain, "most prominently, an **18% EM** gain on the QA task of SQuAD, **8%** on CommonSenseQA and **1%** accuracy on the reasoning task of **GSM8k**."
  - **FAIRNESS**: ⚠️ gain이 task별 크게 갈리고 **reasoning(GSM8k)은 +1%p**로 미미. pretrain+finetune 둘 다 pause 필요(추가-only는 약함).
- **Let's Think Dot by Dot: filler tokens** (arXiv **2404.15758**, Pfau·Merrill·Bowman). "Transformers can use meaningless filler tokens ('......') ... to solve two hard algorithmic tasks they could not solve without intermediate tokens." 단 **"Learning to use filler tokens is difficult and requires specific, dense supervision to converge."** 이론: quantifier-depth(1차논리)로 유용성 클래스 특징.
  - **FAIRNESS**: ✅ 양/음 모두 정직(추가계산 실재하나 학습 매우 어려움·합성과제). 자연 데이터선 표준학습으로 안 나타남.
- **Planning Tokens** (arXiv **2310.05707**, Wang et al., UCSB/MSR/Mila). 각 추론스텝 앞 high-level planning token(임베딩 추가, **+0.001%** params). 모델=Phi-1.5(1.3B)/Llama2-7B/13B.
  - **verbatim (SQ-VAE variant)**: Llama2-13B GSM8K **44.6→50.6 (+6.0pt)**, AQUA **41.3→43.9 (+2.6)**, MATH **7.2→8.5 (+1.3)**. Llama2-7B GSM8K **38.2→40.0 (+1.8)**, AQUA **36.6→41.3 (+4.7)**. 평균 "**3.3% accuracy points**" over 3 models.
  - **FAIRNESS**: ✅ same-base finetune baseline 대비·표준 math bench·실 7B/13B. gain 작지만 robust·**LoRA-친화(임베딩만 추가)**. ⚠️ math word problem 한정(DAG/tool 아님).
- **판정(우리)**: SoS = **trace-distill 경로로 가장 매력적**(우리 결정론 검증기로 cost-aware 탐색 trace 생성→SFT, 이미 §3.10 빌드경로와 정합). Planning Tokens = **저비용 보완 실험 후보**(LoRA로 단계별 plan-token 추가, A2 dirgraph 스텝에 plan-token 부착). pause/filler = gain 미미·distraction.

### 축 4 — 非AR 非diffusion 생성 (set/any-order)
- **A3: "Autoregressive Models Rival Diffusion Models at ANY-ORDER Generation"** (arXiv **2601.13228v1**, Du·Fang·Yang·Zhang·Wei·Wang·Wang; 2026-01-19). Any-order Any-subset AR. 평가 **1B/3B/8B**, baseline=Plaid(1B)/Dream(7B)/DiffuLlama(7B)/LLaMA-3.1-8B.
  - **verbatim**: "A3 outperforms diffusion-based models while maintaining flexible decoding." TriviaQA QA — A3-8B **19.4** vs DiffuLlama-7B 18.5 / Dream-7B 18.3. PIQA — A3-8B **78.1** vs DiffuLlama **63.3**. Story infilling(ROCStories ROUGE) — A3-8B **19.2/4.6/18.6** vs DiffuLlama **23.3/5.5/21.2**(infilling은 diffusion이 ROUGE 우위).
  - **FAIRNESS**: ✅ 동급 규모 diffusion과 직접 대결·표준 bench. ⚠️ 신규(v1·2026-01)·infilling선 diffusion이 ROUGE 더 높음(혼합). **핵심 함의**: 순수 AR이 any-order/parallel 유연성을 가질 수 있다 = **"unordered tool-SET 생성에 diffusion이 필수 아님"의 강한 닻**.
- **Insertion Transformer** (arXiv **1902.03249**, Stern·Chan·Kiros·Uszkoreit, ICML'19): 임의 위치 삽입·부분AR. "accommodates arbitrary orderings ... can be trained to maximize entropy over all valid insertions." WMT'14 En-De 검증.
- **Levenshtein Transformer** (arXiv **1905.11006**, Gu·Wang·Zhao, NeurIPS'19): insert+delete 원자연산·동적 길이·정련(refinement) 가능. "comparable performance with much-improved efficiency."
- **SUNDAE** (arXiv **2112.06749**, Savinov et al.): step-unrolled denoising(非AR, diffusion 아님). WMT'14 En→De **26.25 BLEU**(非AR 중 Transformer-base에 최근접). "filling in **arbitrary blank patterns** in a template."
- **XLNet** (1906.08237): permutation LM = any-order AR의 원조(생성보다 표현학습 목적).
- **판정(우리)**: A3 = **개념적 1순위 닻** — 우리가 diffusion proposer를 *대체*할 후보로 "any-order AR"을 제시 가능(같은 7B base에 any-order objective LoRA?는 미검증·고위험). Insertion/Levenshtein/SUNDAE = **set/그래프 출력의 순서무관성**에 직결되나 전부 MT·번역 시대·decoder-only LLM 미적용 → 인용·개념차용만(재구현은 distraction).

### 축 5 — Structure-aware decoding / 그래프·DAG 아키텍처
- **Grammar-constrained / structured decoding**: **XGrammar** (arXiv **2411.15100**) — pushdown automata로 CFG·JSON schema를 토큰마스크로 강제, 배포최적화. **Outlines**(FSM/PDA, O(1) valid-token lookup). JSONSchemaBench(2501.10868) = 프레임워크 벤치. ATLAS-RTC(2603.27905)=token-level runtime control로 agent 출력 폐루프.
  - **FAIRNESS**: ✅ 배포검증·우리가 **이미 JSON 출력에 의존하는 레이어**. "right(파싱가능 JSON-DAG)"를 디코딩-타임에 보장 = K-sample 풀의 유효성 하한·재학습0·vLLM 호환.
- **Pointer Networks** (arXiv **1506.03134**, Vinyals·Fortunato·Jaitly): 출력이 입력원소를 가리키는 attention-pointer = 가변 출력공간(convex hull/TSP/Delaunay). **set→seq·tool-from-catalog 선택에 구조적 정합**(toolken의 attention 버전).
- **Set-to-Sequence 리뷰** (2103.09656): unordered set 입출력 방법 정리 — 우리 tool-SET 출력의 직접 배경.
- **graph/DAG decoder**: encoder-decoder로 인접행렬·acyclic graph 생성(Transformer encoder + pointer query) 라인 존재(서치결과 다수)이나 **LLM-tool-DAG 직접 적용 1차논문은 빈약** = 갭.
- **판정(우리)**: grammar-constrained = **이미 우리 것·1순위 유지**(A2 compile 출력 schema 강제). pointer-net 스타일 tool-pointer = **A2/도구선택에 흥미로운 아키텍처 차용**(catalog에서 가리키기)이나 LoRA-decoder에 붙이는 건 비표준·고위험. 트리/그래프 디코딩 = `RUNG1_V3_TREE_EVAL`서 이미 "decoder-only LLM=구조bias 無" 결론.

### 축 6 — Tool-selection-specific 아키텍처
- **ToolkenGPT** (arXiv **2305.11554v4**, Hao·Liu·Wang·Hu, NeurIPS'23 oral). 각 도구=새 토큰(toolken) 임베딩을 LM head에 삽입·**LLM frozen**, toolken 임베딩만 학습. "plug in an arbitrary number of tools by expanding the set of toolkens on the fly." 도메인=numerical reasoning/KBQA(KAMEL 234 toolken)/embodied plan. (GSM8K-XL · LLaMA2-13B서 보고; 정확수치는 도메인별·여기선 미인용.)
  - **FAIRNESS**: ✅ frozen-LLM·재학습0 도구추가·massive-tool 확장성. ⚠️ tool 문서 활용 못함·"도구 쓸지 말지" 오판(→Toolken+ 2410.12004이 rerank+reject로 보완).
- **Toolformer** (arXiv **2302.04761**, Schick et al., NeurIPS'23): self-supervised로 API call 위치/인자 학습(어느 call이 future-token 예측 돕는지 loss로 선별). Base=**GPT-J 6.7B**, GPT-3보다 우위(zero-shot). = prompting 아닌 **데이터/학습-레벨** tool 내재화.
- **Tool retrieval+rerank**: ToolRerank(2024)·Chain-of-Tools(2503.16779, frozen LLM서 massive unseen tool)·Agent-as-a-Graph(2511.18194, KG 기반 tool/agent retrieval+wRRF). Outcome-aware semantic router(2603.13426, LLM추론 없이 latency-제약 tool선택).
- **MoE-for-tools**: 전용 "MoE=tool expert" 1차논문은 약함(대부분 RAG-MoE/retrieval-MoE) = 갭. (특허 트랙 MetaTool/ToolBench와 분리.)
- **판정(우리)**: ToolkenGPT/Toolformer = **특허 트랙(도구 SELECTION 내재화)과 직접 정합**(`EXPERIMENT_DESIGN §5` 특허=별 트랙). 본 thesis 라인(A2 compile·gold-free selection)엔 우리 도구수 작아 한계효용↓ → **특허 트랙으로 좌표화, thesis엔 보조**.

### 축 7 — Verifier/planner-integrated 아키텍처
- **already in docs**: ReST-MCTS*(2406.03816)·Math-Shepherd(2312.08935)·TS-LLM value-fn(2309.17179)·AlphaLLM(2404.12253) = process-reward/value-head 통합 디코딩. value-guided MCTS decoding(few lookahead로 토큰선택).
- **신규/skeptical**: **Transformers Struggle to Learn to Search** (arXiv **2412.04703v2**, Saparov·...·Najoung Kim·He He). "As the input graph size increases, the transformer has greater difficulty ... This difficulty is **not resolved even as the number of parameters is increased**." + 큰 그래프선 CoT도 못 고침.
  - **FAIRNESS**: ✅ 통제된 부정결과 — value/search-내재화 라인 전반의 **규모-한계 경고**.
- **판정(우리)**: 우리 **gold-free 결정론 검증기 = 사실상 outcome-verifier**(ReST-MCTS*의 process-reward를 최종 gate-정답서 추론하는 우리 방식과 정합). 별도 학습 value-head = depth/cost-aware 게더에 보완 후보(TS-LLM)이나 우리 트리 얕음(2-7조건)이라 우선순위 낮음.

---

## §3. Cross-cut — diffusion proposer의 대안/보완 vs 직교 후보생성기 vs 재학습 여부

우리 다양성 목표 = **different-AND-right (D-oracle>0)**: 서로 다르면서 검증기 통과하는 K 후보를 풀에 모아 gold-free 선별.

| 라인 | diffusion 대안/보완? | 직교 후보생성기(풀 합류)? | 재학습 vs 디코딩-타임 |
|---|---|---|---|
| **Grammar-constrained (XGrammar)** | 보완(모든 proposer의 유효성 레이어) | — (생성기 아닌 필터) | **디코딩-타임·재학습0** ✅ |
| **Any-order AR (A3)** | **★대안 후보**(diffusion 없이 any-order 다양성) | 가능(any-order LoRA arm) | **재학습**(objective 변경·고위험·7B 미검증) |
| **Stream-of-Search distill** | 보완(탐색다양성을 trace로) | 가능(SoS-style LoRA arm = 다른 분포의 후보) | **재학습**(SFT trace distill) |
| **Planning Tokens** | 직교 | 가능(plan-token LoRA arm = heterogeneous 후보) | **재학습**(임베딩 추가·저비용 LoRA) |
| **Insertion/Levenshtein** | 대안(편집기반 set생성) | 개념적(decoder-LLM 미적용) | 재학습(아키텍처 교체·distraction) |
| **Latent CoT/Coconut** | 대안 주장이나 retrofit불가 | ❌ | 재학습(불안정·distraction) |
| **Pointer-net tool선택** | 직교 | 개념차용 | 재학습(비표준 head) |
| **ToolkenGPT/Toolformer** | 직교(도구선택 전용) | 가능(toolken arm) | 재학습(임베딩/SFT, frozen-LLM) |
| **Value-head/PRM (TS-LLM)** | 보완(선별기 강화) | ❌ (선별측) | 재학습(value-fn) |

**핵심 통찰**:
1. **gold-free selection 풀에 이질적 후보를 넣는 우리 전략은 "여러 디코딩/학습 개입을 *직교 arm*으로 합치는" 것과 동형.** diffusion proposer가 하려던 "different distribution의 후보 주입"을 **(a) any-order AR LoRA, (b) SoS-trace LoRA, (c) planning-token LoRA**가 각각 *재학습은 들지만 vLLM 멀티-LoRA 호환*인 대체/보완으로 제공 가능. 이미 우리는 "heterogeneous LoRA pool" 위에 선별(N2 공식 +8.8)을 베팅 중 → **이 라인들은 그 풀의 새 멤버 후보**.
2. **재학습0(디코딩-타임)으로 즉시 얻는 유일한 강한 카드 = grammar-constrained decoding**. 나머지 다양성 카드는 전부 추가 LoRA 학습(=arm) 필요.
3. **A3(any-order AR)는 "diffusion 없이도 set/any-order 다양성"의 이론·실증 닻** → 우리가 디퓨전 라인을 P-D(-1) census로 선별기로 교체한 결정(메모리)을 *논거 보강*. 단 7B any-order objective LoRA는 미검증 고위험 → 제안서엔 "닻"으로만, 실험은 저비용부터.

---

## §4. 정직한 한 줄 판단 — 우리 3대 베팅 대비 추가가치 vs distraction

우리 기존 베팅: (a) 결정론 게이트 + A2 compile, (b) gold-free selection over heterogeneous LoRA pool, (c) graph-guided internalization(§3.10).

**추가할 가치 있음:**
- **① Grammar-constrained decoding을 A2 출력 schema에 명시 레이어로 박제** — 이미 의존중이나 *아키텍처 개입으로 공식화*하면 "right(파싱·schema유효) 하한 보장 = 풀의 D-oracle 분모 안정화". 재학습0·즉시. **(a)·(b) 양쪽 강화.**
- **② Planning-token LoRA를 (b) 풀의 새 arm으로 저비용 시험** — +0.001% params·LoRA친화·heterogeneous 후보 공급. A2 dirgraph 스텝마다 plan-token. 단 gain 작음(math서 +1.8~6pt)이라 "풀 다양성 기여" 가설로만, 헤드라인 아님.
- **③ SoS-style trace distill** — 우리 결정론 검증기로 cost-aware 탐색 trace(실패·백트랙 포함) 생성→SFT = §3.10 빌드경로 정합·교사초과 잠재. (b)/(c) 보완. 중비용.
- **④ A3(any-order AR)는 "diffusion 불필요" 논거로 관련연구에 박제**(실험 아닌 framing).

**distraction(추가 말 것):**
- **Latent CoT / Coconut** — 7B retrofit 불가·math서 AR에 짐·불안정. 우리 K-sample이 이미 "BFS-in-latent"를 *명시적*으로 달성. ❌
- **Insertion/Levenshtein/SUNDAE 재구현** — MT시대·decoder-LLM 미적용·아키텍처 교체비용 과대. 개념차용만. ❌
- **Pause/filler tokens** — reasoning gain +1%p·filler는 학습 극난. ❌
- **In-model looping/recurrence retrofit** — `SEARCH_INTERNALIZATION §9` 이미 "LoRA-on-7B 부정"(Huginn). ❌
- **Pointer-net/graph-decoder head 교체** — `RUNG1_V3_TREE_EVAL` "decoder-only LLM=구조bias 無" 정합·비표준 head 고위험. ❌
- **ToolkenGPT/Toolformer/MoE-tool** — 우리 도구수 작아 thesis 한계효용↓ → **특허 트랙으로 이관**(중복 아님). ◐

**천장 경고 박제**: 2412.04703("transformers struggle to learn to search, 파라미터 늘려도 안 됨")는 탐색-내재화/value-head 라인 전반의 규모-한계 → ③·value-head 투자 전 "우리 트리가 그 regime인가(얕음 2-7조건)" 선확인. (=`SEARCH_INTERNALIZATION §9` "serial-depth가 우리 병목인지 선결"과 동일 가드.)

---

## §5. 검증 bibliography (primary, arXiv) / Unverified leads

**Verified (abs/HTML 확인·수치 verbatim):**
- 2412.06769 (Coconut, v3) — ProsQA 97.0 vs 77.5 / GSM8k 34.1 vs 42.9 (GPT-2)
- 2404.03683 (Stream of Search) — 250M GPT-Neo from scratch, +25% 탐색정확도, 36% 미해결 해결
- 2310.02226 (Pause Tokens, ICLR'24) — 1B, SQuAD +18% EM, GSM8k +1%
- 2404.15758 (Let's Think Dot by Dot, filler) — filler 학습난·dense supervision 필요
- 2310.05707 (Planning Tokens) — Llama2-13B GSM8K 44.6→50.6, 평균 +3.3pt, +0.001% params
- 2412.06769 latent-CoT 일반론 + 2505.16782v2 (survey) — "underperform explicit CoT, instability of training"
- 2601.13228 (A3 any-order AR, v1) — A3-8B TriviaQA 19.4 > DiffuLlama 18.5 / PIQA 78.1 vs 63.3
- 1902.03249 (Insertion Transformer, ICML'19) · 1905.11006 (Levenshtein Transformer, NeurIPS'19) · 2112.06749 (SUNDAE, 26.25 BLEU) · 1906.08237 (XLNet)
- 2411.15100 (XGrammar) · 2501.10868 (JSONSchemaBench) — structured/grammar decoding
- 1506.03134 (Pointer Networks) · 2103.09656 (Set-to-Sequence review)
- 2305.11554v4 (ToolkenGPT, NeurIPS'23 oral, frozen LLM) · 2410.12004 (Toolken+) · 2302.04761 (Toolformer, GPT-J 6.7B) · 2503.16779 (Chain-of-Tools) · 2511.18194 (Agent-as-a-Graph) · 2603.13426 (Outcome-aware router)
- 2412.04703v2 (Transformers Struggle to Learn to Search) — 파라미터↑로 미해소
- 2502.03275 (Token Assorted, VQ-VAE 잠재토큰) · 2407.14414 (System-1.x)

**Unverified leads (제목·존재만 확인, 수치 미검증 → 인용 전 직독 필수):**
- 2601.21358 / PLaT "Latent CoT as Planning: Decoupling Reasoning from Verbalization" (검색결과 제목만)
- 2509.16278 "Language Modeling with Learned Meta-Tokens" · 2509.22131 R-Capsule · 2510.13879 "Catch Your Breath: Adaptive Computation" (adaptive-step 라인 추가후보)
- 2602.01148 "Capabilities and Fundamental Limits of Latent CoT" · 2512.21711 "Do Latent Tokens Think?" (latent 한계 적대분석 — distraction 판정 보강용)
- 2603.27905 ATLAS-RTC (agent token-level runtime control) · 2507.16768 WGrammar (구조디코딩 가속)
- ToolkenGPT 도메인별 verbatim 수치(GSM8K-XL/KAMEL) — PDF만 존재, HTML 미확보 → 인용시 직독.

**기각·주의**: PDF WebFetch가 본 환경서 파싱 불가(바이너리) → 모든 수치는 abs/HTML/ar5iv 또는 검증된 2차요약 교차확인분만 인용. ToolkenGPT 정확수치는 미확정으로 본문서 생략.
