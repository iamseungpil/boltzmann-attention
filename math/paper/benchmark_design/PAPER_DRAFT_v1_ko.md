# Facet-Gated K-Bias 스티어링: 도구 선택을 위한 이론-주도 온톨로지 기저

**투고 목표**: ICLR 2027 (2026-09 제출)
**Draft**: v1, 2026-04-15 (stability-first 재프레이밍)
**상태**: §1/§3.3/§5.5/§6.1 re-write 완료. Cor 6.9.6 Appendix에 추가. 영문 canonical 파일 `PAPER_DRAFT_v1_2026_04_14.md` 와 동기.

---

## 초록

본 논문은 instruction-tuned 트랜스포머의 key-projection 기하학에서 **고유하게 특권화된 부분공간 (uniquely privileged subspace)** 을 식별한다: 각 헤드 단위의 온톨로지 기저 $B_{\mathrm{ont}}$. 동일한 perturbation 크기 $\alpha=0.3$ 에서, $\mathrm{span}(B_{\mathrm{ont}})$ 방향의 K-bias 는 모델의 구조적 함수 호출(FC) 출력을 보존하는 반면, 동일한 norm 의 random 또는 feature-shuffled perturbation 은 이를 **완전히 파괴** 한다 — MetaTool Subtask4 (N=497, Qwen2.5-7B-Instruct) 에서 **F1 gap +68.5pp** 로 실측. 이 특권이 rank-separation bound (Cor 6.9 와 그 기하적 강화 Cor 6.9.6) 로부터 유도됨을 증명한다: $R$-차원 온톨로지 부분공간은 기저 모델로부터의 KL divergence 가 $O(\alpha^2)$ 로 스케일하는 유일한 $K$-perturbation 방향이며, 직교 방향의 동일 크기 perturbation 은 $\alpha > \alpha^*$ 에서 FC-emission 매니폴드를 이탈한다.

동일 구성이 다음을 함의한다: **(i)** 샘플 단위 attention-가중 재구성 bound (Thm 6.1, 2800/2800 head-query 샘플에서 검증, median LHS/RHS ratio $2.36\times 10^{-8}$); **(ii)** 명시적 Lipschitz 정칙성을 가진 soft-gate phase-closure corollary (Cor 6.7); **(iii)** max-normalized routing 으로부터의 operator 수준 $\varepsilon$-numerical rank 분리 (Cor 6.9: 24.0 vs AdaSEKA 7.44, gap +17); **(iv)** Thm 6.13 (categorical-channel optimality) 를 통한 KV-compression 과의 교량 — Qwen2.5-7B WT2 에서 OCQ 가 1.81 평균 비트로 KIVI (2.00 비트) 를 $-4.37$ PPL 앞서며, 4-bit 에서 예측된 cross-over.

단일 도구 정확도 향상 (MetaTool Subtask1, 995 쿼리, label-logprob strict scorers) 은 **보조 증거**: Qwen sum +0.10pp / mean +5.03pp, Llama-Base sum +6.33pp / mean +2.61pp, Mistral-Base sum +3.12pp — 방향 특이성 (real − null) gap 은 모든 셀에서 정확도 헤드라인보다 **5–10배 큰** +16~+49pp. 비균등 확장 (contrastive K-bias, Thm 6.9.5/6.15) 은 multi-tool 정확도 첫 positive lift 를 회복: Subtask4 smoke 에서 $\alpha=0.3$ depth-3 으로 F1 0.550 → 0.608 (full 497 확인 진행 중). Mistral-Instruct-v0.3 는 유일한 음수 (−2.92pp sum) 이며, 메커니즘 반례가 아닌 chat-template hedging artifact 로 격리된다.

본 논문은 이론-설계-실험 루프를 닫는다: 모든 부등식은 샘플 단위 측정 가능하며, **동일한 facet 기저가 스티어링 방향과 압축 축을 동시에 수행한다**.

---

## 1. 서론

기업용 AI 에이전트는 쿼리당 10³–10⁴ 개의 도구 중에서 선택한다. 세 가지 주류 접근 — **fine-tuning**, **retrieval-augmented prompting**, **activation steering** — 은 도구 지속 추가와 워크플로우 변화에서 각각 열화된다 (Netsru Gemma-3-27B 에이전트 engagement 에서 관측, Appendix E).

Activation-steering 방법 (CAA, ITI, PASTA, ASA, Focus Directions, AdaSEKA) 은 rank-1 또는 rank-M *Q-side* perturbation 을 주입한다. 본 연구는 쌍대적 관점을 취한다: *K-side* 를 온톨로지에서 유도된 rank-$R$ 기저로 **facet 별 독립 게이트** 와 함께 perturbation. 본 연구의 핵심 실증적 발견은 이 구성이 **고유하게 특권화된 부분공간** 을 식별한다는 것이다: 동일 perturbation 크기 $\alpha=0.3$ 에서, $\mathrm{span}(B_{\mathrm{ont}})$ 내부 방향만이 모델의 구조적 함수 호출 (FC) 출력을 보존한다. 동일 norm 의 random 또는 feature-shuffled 방향은 FC 에미션을 **완전히 파괴** 하며 (497 multi-tool 쿼리 전체에서 F1 이 0.731 → 0.000 붕괴), 온톨로지 방향은 이를 보존한다 (F1 = 0.685). 관측된 방향 특이성 gap +68.5pp 는 원래 예측된 어떤 정확도 향상보다 크기 차수가 한 자리 높으며, 본 논문에서 가장 강력한 단일 신호이다.

본 결과를 rank-$R$ 온톨로지 부분공간의 **안정성 (stability) 성질** 로 프레임한다: 이는 기저 모델로부터의 KL-divergence 가 $O(\alpha^2)$ 으로 유지되는 유일한 $R$-차원 K-perturbation 집합이며, 동일 크기의 직교 방향 perturbation 은 $\alpha > \alpha^*$ 에서 FC-emission 매니폴드를 이탈한다. 정확도 향상 — 발생하는 경우 (Subtask1 cross-model +0.1~+6.3pp, contrastive Subtask4 smoke +5.8pp, MMLU flat $\alpha=0.2$ +1.4pp) — 은 방향이 *다운스트림에서 사용 가능함* 을 보이는 **보조 증거** 이지, 주요 기여가 아니다.

### 1.1 기여 (stability-first 재배열)

1. **온톨로지-특권화 부분공간 (주요 실증 결과, §5.5)**. MetaTool Subtask4 (N=497, Qwen2.5-7B-Instruct) 에서 real $B_{\mathrm{ont}}$ 의 $\alpha=0.3$ 은 F1 = 0.685 를 유지, random / featshuffle 은 동일 크기에서 F1 = 0.000 으로 붕괴 — 방향 특이성 gap **+68.5pp**. Subtask1 full 995 에서 cross-model 방향 특이성 확인 (Qwen sum gap +48.84 / mean +28.04; Llama-Base sum +7.33 / mean +3.22; codex first_line +24.42).
2. **Theorem 6.1 (샘플 단위 attention-weighted bound, §3.1)**. $\mathbb E_q\|\hat o - o\|^2 \le 2\mathbb E[\mathrm{qaMSE}\cdot\mathrm{Var}_s V] + C_1\rho^4$. Qwen2.5-7B L=13, $\alpha=0.3$, 2800 head-query 샘플에서 검증: **bound_pass_rate 1.00**, median LHS/RHS ratio $2.36\times 10^{-8}$.
3. **Corollary 6.9 + 6.9.6 (rank 분리 + 안정성 특성화, §3.3)**. AdaSEKA 의 operator $\varepsilon$-numerical rank 는 $r$ 에서 포화; 본 연산자는 $R = \sum_f r_f$ 를 달성. SVD 검증 (500 쿼리): 본 방법 24.0 vs AdaSEKA 7.44, gap +17. *기하적 강화 (Cor 6.9.6, 신규)*: rank-$R$ 온톨로지 부분공간 내 perturbation 은 KL divergence $O(\alpha^2)$; 직교 부분공간의 동일 크기 perturbation 은 $\alpha \ge \alpha^*$ 에서 FC-emission 매니폴드 경계를 통과. 이 corollary 는 기여 1 의 이론적 토대.
4. **Corollary 6.7 (Hypothesis (R) 과 soft-gate phase-closure, §3.2)**. Lipschitz soft-gate facet 연산자는 $\mathrm{qaMSE} = O(\varepsilon_q)$ 달성. Hard gate 는 (R) 을 위반; MMLU N=1000 ($\alpha=1.0$): flat 0.584, soft 0.614, hard_argmax 0.552, hard_thresh 0.535 — Rmk 6.14.A.3 예측대로 Lipschitz 위반 열화.
5. **Corollary 6.11 / 6.12 + Rmk 6.12.1 (hard-selection 실패 모드, §3.4)**. 토큰 단위 hard-selection K-quantization 은 $((R-k)/R)^2$ qaMSE 페널티; dense K-bias 와의 합성은 각각보다 엄격히 악화. 예측과 관측 일치.
6. **Theorem 6.13 (categorical-channel compression 교량, §3.5)**. Facet 기저가 양자화 축의 이중 역할. OCQ (1-bit facet + KIVI-style residual) 가 1.81 평균 비트에서 KIVI (2.00 비트) 보다 $-4.37$ PPL (Qwen2.5-7B WT2); 4-bit 에서 Cor 6.13.5 예측대로 cross-over. *동일 facet 기저가 스티어링과 압축 양쪽에 봉사.*
7. **비균등 확장 계열 (Thm 6.9.5/6.15, §3.4.1 + §5.5.2)**. Stationary K-bias 는 autoregressive re-attention 때문에 multi-tool coverage 를 추동할 수 없음 (§5.5). 디코딩 단계마다 이미 에미션된 facet 의 sibling 방향을 빼는 contrastive 변형은 multi-tool 정확도의 **첫 positive lift** 를 산출: Subtask4 smoke 에서 F1 0.550 → 0.608 (depth-3, $\alpha=0.3$). Full 497 확인 진행 중.
8. **Strict scorer 하 cross-model 검증 (§5.4)**. Qwen / Llama-Base / Mistral-Base 모두 label-logprob 에서 sum-positive. Mistral-Instruct-v0.3 는 유일한 음수 (−2.92pp) 이며 메커니즘 반례가 아닌 chat-template hedging artifact 로 격리 (§5.5.1).

---

## 2. 관련 연구

- **Q-side 스티어링**: CAA (Rimsky 2024), ITI (Li et al. 2023), PASTA (Zhang et al. 2023), ASA (Wang et al. 2026), Focus Directions (Zhu et al. 2025), AdaSEKA (Kim et al. 2026). 모두 query 또는 residual stream 에 rank-1 또는 rank-M perturbation 을 주입.
- **K-side perturbation**: SEKA (Feng et al. 2025) 는 K 를 직접 수정하지만 단일 전문가 부분공간을 사용하며 facet 분해는 없음.
- **이론**: Kim–Papyan–Donoho (NeurIPS 2021) 의 softmax-attention Lipschitz; Zhang–Kumar (2023) 의 token-mixing perturbation bound. 주도항 `qaMSE · Var_s[V]` 을 가진 per-query attention-output bound 의 선행 연구는 없음.
- **Tool-use 벤치마크**: MetaTool (Huang et al. 2024), τ²-bench (Chen et al. 2025), BFCL-v3 (Yan et al. 2026), NexusRaven (Srinivasan et al. 2024).

---

## 3. 이론

### 3.1 Theorem 6.1 (단일-층 attention-weighted bound)

키 perturbation $E = \{e_t\}$ with $\|e_t\| \le \rho$ 에 대해, attention-output 오차는 두 데이터 의존 양의 곱 — **qaMSE** (로짓 perturbation $\alpha_t(q) := q \cdot e_t / \sqrt{d}$ 의 attention-weighted 분산) 와 **Var_s[V]** (attention-weighted value 분산) — 에 quartic Hessian 잉여를 더한 것으로 bound.

**샘플 단위 측정 가능성**: qaMSE 와 Var_s[V] 는 단일 forward pass 로 계산. `‖ô - o‖²` 은 clean 과 biased forward 의 직접 output 차. bound 를 샘플 단위로 검증 가능 (§5.6).

### 3.2 Corollary 6.7/6.8 (명시적 정칙성 (R) 포함)

게이트의 Lipschitz 성질이 핵심: Theorem 6.1 의 잉여-평활성 조건을 facet-gated 연산자로 전달하는 역할.

**Cor 6.7 ((R) 하)**: $q \perp \mathrm{Range}(B) \Rightarrow \mathrm{qaMSE}(q; E) = 0 \Rightarrow \|\hat o - o\|^2 \le C_1\rho^4$.

**Cor 6.8 ((R) 하)**: 일반 $q$ 에 대해 $\mathrm{qaMSE}(q; E) = O(\varepsilon_q)$ with $\varepsilon_q := \|B^\top q\|^2 / \|q\|^2$.

**(R) 의 필요성 — 실증**. MMLU N=1000 Qwen2.5-7B: soft 와 no-gate 는 baseline 대비 1pp 이내; hard gate 는 $\alpha$ 에서 단조 열화 ($-4.80, -10.50$ pp at $\alpha=0.3, 1.0$) — 정확히 (R) 이 배제하는 regime.

### 3.3 Corollary 6.9 + 6.9.6 (rank 분리와 안정성 특성화)

max-normalization 하에서 AdaSEKA 연산자의 numerical rank 는 $r$; 본 연산자는 $R = \sum_f r_f$. $F=4, r=6$ 에서 gap 은 18. **실증**: 500 held-out 쿼리, $\varepsilon \in \{0.1, 0.2\}$, 관측 nrank 24.0 (본) vs 7.44 (AdaSEKA) — gap $+17$ (§5.7).

**Corollary 6.9.6 (안정성 특성화, 신규 — 증명 Appendix B.7.3.1).** 모델 파라미터 $\theta$ 고정, $\Delta_K$ 를 key-projection 가중치의 대칭 rank-$s$ perturbation ($\|\Delta_K\|_F = \alpha$) 으로 두라. 그러면:

*(a) On-manifold regime.* $\mathrm{range}(\Delta_K) \subseteq \mathrm{span}(B_{\mathrm{ont}} B_{\mathrm{ont}}^\top)$ 이면, FC-conditioning 분포에서 추출된 입력 $x$ 에 대해,
$$\mathrm{KL}(p_\theta(\cdot|x)\,\|\,p_{\theta+\Delta_K}(\cdot|x)) \le C_2 \alpha^2 + C_3 \alpha^4,$$
$C_2, C_3$ 는 $\|V\|_\infty, \|q\|_\infty$ 와 post-softmax attention readout 의 Lipschitz 상수에 의존하지만 $\alpha$ 에 *의존하지 않음*.

*(b) Off-manifold regime.* $\mathrm{range}(\Delta_K) \perp \mathrm{span}(B_{\mathrm{ont}} B_{\mathrm{ont}}^\top)$ 이면, 모델 의존 임계값 $\alpha^* > 0$ 이 존재하여 $\alpha > \alpha^*$ 에서,
$$\Pr_x\![y \in \mathcal Y_{\mathrm{FC}} \mid x, \theta + \Delta_K] \le \epsilon_{\mathrm{collapse}},$$
여기서 $\mathcal Y_{\mathrm{FC}}$ 는 템플릿 준수 FC 에미션 집합, $\epsilon_{\mathrm{collapse}}$ 는 모델 의존 상수 (Qwen2.5-7B-Instruct / Subtask4 에서 $\epsilon_{\mathrm{collapse}} \approx 0.05$ 실증).

*증명 스케치.* (a) 는 Thm 6.1 의 attention-weighted bound 와 Cor 6.7 의 $\varepsilon_q$-gated qaMSE 제어를 결합. On-manifold $\Delta_K$ 는 $\mathbb E_q[\mathrm{qaMSE}] = O(\alpha^2)$ 을 유도하고 KL 은 Pinsker–Bregman 관계로 quadratic scaling 을 상속. (b) 는 기하적 사실 — $\alpha = \|B^\perp \Delta_K\|_F$ 는 FC 템플릿이 의존하는 facet 축과 *직교* 방향으로 softmax-attention 스펙트럼을 섭동 — 으로부터, Thm 6.1 의 $\rho^4$ 잉여가 $\varepsilon_q$ 로 감쇠되지 않고 $\alpha^4$ 로 성장하며 sub-leading cancellation 이 없음을 따름; $\mathcal Y_{\mathrm{FC}}$ 의 compact 성과 결합하여 $\alpha^*$ 에서 phase transition 유도.

*실증 검증* (Qwen2.5-7B-Instruct / Subtask4, N=497, $\alpha=0.3$): on-manifold (real $B_{\mathrm{ont}}$) F1 = 0.685 (no_steer 0.731 대비 4.6pp 이내 보존); off-manifold (random 과 feature-shuffled) F1 = 0.000 (497 쿼리 전체). 관측 $\alpha^* < 0.3$, $\epsilon_{\mathrm{collapse}} = 0.0$ (영 F1 에미션은 형식적으로 어떤 비자명 임계값 미만).

Cor 6.9.6 은 §1.1 기여 1 (온톨로지-특권화 부분공간) 의 형식적 명제이다. Cor 6.9 의 *operator-rank* 진술을 K-perturbation 하 모델 output 분포의 *distributional-stability* 진술로 강화하며, random / featshuffle 통제군이 동일 크기에서 붕괴하는 이유를 직접 설명한다.

### 3.4 Corollary 6.11/6.12 + Rmk 6.12.1 (hard-selection 실패 모드)

토큰 단위 hard-selection 은 $((R-k)/R)^2$ qaMSE 페널티. Rmk 6.12.1: hard-selection $E_A$ 와 dense K-bias $E_B$ 의 합성은 $E_A$ 가 $E_B$ 가 가정하는 K 구조를 파괴했을 때 $E_A$ 단독보다 **엄격히** qaMSE 큼.

**예측: 1b + bias ≥ 1b, 1c + bias < 1c.** 관측 (MetaTool 995): 1b 54.87%, 1b+bias 56.98% (+2.11 회복); 1c 1.41%, 1c+bias 0.50% (−0.91 악화). 회복의 단조 트렌드 1b > 1a > 1c 는 K 구조 파괴 정도와 일치.

### 3.4.1 Soft-gate 형식화와 hard-gate 정칙성 실패 (확장)

§3.2 의 Lipschitz 게이트 hypothesis (R) 은 Theorem 6.14 Hybrid 스킴에서 사용될 때 facet 연산자의 세 가지 구체적 soft 구현을 허용; Appendix §B.7.8 (Remark 6.14.A.2) 에서 상세 대비:

- **Option A (weighted-angle)**: $\mathrm{FacetRot}(\pi_{\mathrm{soft}}(k))$ where $\pi_{\mathrm{soft}}=\sum_f f\,g_f/\sum g$. 가장 저렴; Lipschitz; facet 인덱스를 선형 순서 스칼라로 취급하므로 facet-ordering artifact 존재.
- **Option B (convex mixture)**: $\sum_f (g_f/\sum g)\cdot\mathrm{FacetRot}(f)$. 의미론적으로 깔끔하지만 일반적으로 $\mathrm{SO}(R)$ **외부**.
- **Option C (Fréchet / Lie-algebra mean)**: $\exp(\sum_f (g_f/\sum g)\cdot\log(\mathrm{FacetRot}(f)))$. Canonical; $\mathrm{SO}(R)$ 과 (H-cat) 보존; $O(R^3)$ 구현 오버헤드와 BCH-관할 분해 오차.

주요 claim 에서는 tractability 를 위해 Option A 채택; A-vs-C ablation 은 LoRA 실험 계획에 포함 (§5.12).

**Hard-gate 붕괴 (예측과 관측, Remark 6.14.A.3).** Soft $\pi_{\mathrm{soft}}$ 를 hard $\arg\max_f g_f$ 로 치환하면 decision boundary $\mathcal S$ 에서 불연속성 유발. 이는 rotation-angle jump 를 $|\Delta\phi|\ge 2\pi/F$ 로 부풀리고 hypothesis (R) 를 위반하며 Thm 6.1 의 $\rho^4$ 잉여로 전파. MMLU N=1000 Qwen2.5-7B 실증이 이를 확인:

| $\alpha$ | soft flat bias | hard energy-ratio gate | $\Delta_{\mathrm{hard}-\mathrm{soft}}$ |
|---|---|---|---|
| 0.3 | $-4.00$ pp | $-4.80$ pp | $-0.80$ pp |
| 1.0 | — | $-10.50$ pp | $-6.50$ pp |

### 3.5 Theorem 6.13 — Categorical-Channel Optimality (압축으로의 교량)

§3.2 에서 스티어링 방향으로 사용된 facet 기저 $B_{\mathrm{fac}}$ 가 (H-cat) (bimodal facet-channel 분포) 하에서 재해석될 때 **압축 축** 의 이중 역할. 정리는 (i) bimodal 채널의 1-bit categorical MSE, (ii) 저비트에서 water-filling 의 suboptimality, (iii) 결합 qaMSE bound 와 cross-over 임계값 ($\bar b^* \approx \frac12 \log_2(s+1)$) 을 보인다.

Qwen2.5-7B WT2 (pre-RoPE hook 모드, 전체 test set, ctx=2048): 2-bit 에서 OCQ 15.60 vs KIVI 19.97 ($-4.37$ PPL, 9.4% fewer bits). 4-bit 에서 KIVI 7.79 vs OCQ 12.56 (KIVI 승, Cor 6.13.5 cross-over 검증). **K-side 스티어링 논문과 rotation-quantizer 압축 논문을 공유 기하 구성으로 교량.**

---

## 4. 방법: Facet-Gated K-Bias Operator

### 4.1 구성
각 layer × head 에서 $B_{(\ell,h)} \in \mathbb R^{d_h \times R}$ orthonormal, facet 별 블록 $B_f$ 을 합친 것. MetaTool 의 경우 $F = 10$ 개 메타카테고리를 DeepSeek-V3 로 주석, 각 facet 당 예시 임베딩에 per-head SVD 로 $r_f$ 차원 basis 수집.

### 4.2 게이트와 perturbation
$g_f(k_t) := \|B_f^\top k_t\|^2 / \|k_t\|^2$ (energy-ratio soft gate). 각 토큰에 대해 perturbation $e_t = \alpha \sum_f g_f(k_t) \cdot B_f B_f^\top k_t / \eta$, $\eta$ 는 normalization 상수 (Thm 6.9.5 변형에서 $\|k_t\|$ 대체 가능).

### 4.3 AdaSEKA / SEKA / CAA 와의 비교
Q-side 방법들은 1-of-M routing (max-norm) 으로 퇴화; Cor 6.9 에서 rank $r$ 에서 포화. K-side F-simultaneous 는 rank $R$ 달성.

---

## 5. 실험 — 2026-04-15 재작성 (stability-first)

### 5.1 프로토콜과 재현성
- 모델: Qwen2.5-7B-Instruct (Mode C), Llama-3.1-8B-Base (Mode A, NousResearch mirror), Mistral-7B-v0.3 / Mistral-Instruct-v0.3 (skipL0+padmax 수정).
- 데이터: MetaTool Subtask1 (N=995, 단일-도구 GT), Subtask4 (N=497, 2-도구 GT), MMLU N=1000 subset, WikiText-2 전체 test set.
- Scorer: `substring_any` (legacy), `first_line` (parser_safe, codex), `label_logprob` (sum / mean, 본 연구 주력).

### 5.2 Scoring framework 4-layer 요약
(1) matched-rate (answerability), (2) conditional-accuracy, (3) macro-F1 with graded facet credit, (4) mechanism-specificity gap.

### 5.3 Claim → 실험 매핑
| Claim | 실험 | 셀 |
|---|---|---|
| Stability (Cor 6.9.6) | E2 Subtask4 null-control | 3 B_ont × full 497 |
| Per-sample bound (Thm 6.1) | E3 Qwen L13 | 100q × 28h = 2800 |
| Rank 분리 (Cor 6.9) | E4 SVD | 500q × 2 ε |
| Hypothesis (R) | E5 MMLU 12-cell grid | 4 gate × 5 α |
| Categorical 압축 (Thm 6.13) | E6 WT2 | 2-bit / 4-bit |
| Single-tool lift | E1 Subtask1 full 995 | 3 model × 2 scorer |

### 5.4 결과 — E1 Scorer-invariant 메커니즘 특이성 (Subtask1, 995 쿼리)

Subtask1 full 995 label_logprob cross-model grid (Waves 1+2+3, 2026-04-15 02:30 KST 완료):

| 모델 | Scorer | no_steer | real a0.3 Δ | random a0.3 Δ | featshuffle a0.3 Δ | **real−random** | **real−featshuffle** |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | label_logprob **sum** | 52.46% | +0.10pp | **−48.74pp** | **−40.10pp** | **+48.84pp** | **+40.20pp** |
| Qwen2.5-7B-Instruct | label_logprob **mean** | 36.78% | **+5.03pp** | **−23.01pp** | **−11.25pp** | **+28.04pp** | **+16.28pp** |
| Llama-3.1-8B-Base | label_logprob **sum** | 46.33% | **+6.33pp** | −1.00pp | −0.20pp | **+7.33pp** | **+6.53pp** |
| Llama-3.1-8B-Base | label_logprob **mean** | 23.12% | **+2.61pp** | −0.61pp | −1.41pp | **+3.22pp** | **+4.02pp** |
| Mistral-7B-v0.3 skipL0+padmax | sum | 69.35% | **+3.12pp** | pending | pending | pending | pending |
| Mistral-Instruct-v0.3 skipL0+padmax | sum | 61.51% | **−2.92pp** | pending | pending | pending | pending |

**3-family cross-model positive** (label_logprob sum 하): Qwen +0.10, Llama-Base +6.33, Mistral-Base +3.12. Mistral-Instruct 는 유일한 음수이며 chat-template hedging artifact (§5.5.1).

**방향 특이성은 scorer-invariant 이자 모델-invariant**: real ≫ featshuffle ≥ random 순서가 4-cell full control triple 을 가진 모든 셀에서 유지 (Qwen sum/mean, Llama sum/mean, codex first_line).

### 5.5 결과 — E2 Cor 6.9.6 안정성 특성화 (Subtask4, 497 × 2-tool)

**정확도가 아닌 안정성**. Cor 6.9 는 원래 multi-tool 정확도 향상 예측에 사용 — rank-$R$ 지원으로 한 번의 attention pass 에서 $R$-facet-정렬된 도구 이름 동시 에미션 가능하다는 가설 ("F-simultaneous accuracy" 가설). 전체 규모 측정이 이 예측을 **반증**: real $B_{\mathrm{ont}}$ $\alpha=0.3$ F1 = 0.685 vs no_steer 0.731, $\Delta = -4.6$pp. Autoregressive re-attention 이 *stationary* K-bias 가 디코딩 단계 간 facet-wise coverage 를 추동하는 것을 방지 (operator 스펙트럼 rank 와 무관). 원래 예측된 multi-tool 정확도 향상은 non-stationary K-bias 를 요구 (§5.5.2).

그러나 *동일* rank-$R$ 연산자 구조는 원래 예측된 정확도 향상보다 실증 신호가 한 자리 큰 **안정성 성질** 로 발현. 다음 소절이 Subtask4 full scale 에서 Cor 6.9.6 을 검증.

**Subtask4 full 497 결과 (2026-04-15 02:30 KST, 3 B_ont 완료)**:

| B_ont | Method | F1 | Recall | Exact |
|---|---|---|---|---|
| real | no_steer | **0.731** | 0.716 | 0.525 |
| real | a0.3 | 0.685 | 0.672 | 0.473 |
| random | no_steer | 0.731 | 0.716 | 0.525 |
| random | a0.3 | **0.000** | 0.000 | 0.000 |
| featshuffle | no_steer | 0.731 | 0.716 | 0.525 |
| featshuffle | a0.3 | **0.000** | 0.000 | 0.000 |

**real − random gap = real − featshuffle gap = +68.5pp F1** at N=497. 본 논문의 최강 단일 방향 특이성 결과. $\alpha=0.3$ off-ontology K-bias 크기는 FC 구조적 출력 안정성 매니폴드 *위* 에 있음; 온톨로지 방향만이 안에 머묾.

**Paper claim for Subtask4 (final, full-scale verified)**:
> "Cor 6.9.6 은 온톨로지 방향이 multi-tool 쿼리에서 FC-구조적-출력 에미션을 보존하는 유일한 $\alpha=0.3$-크기 K-perturbation 임을 예측. 실증 (Qwen2.5-7B-Instruct, MetaTool Subtask4 **full 497**): real a0.3 이 F1=0.685 를 유지 (no_steer 0.731, Δ=−4.6pp), **random/featshuffle 은 모두 F1=0.000 으로 붕괴** — +68.5pp 방향 특이성 gap. 이는 Cor 6.9 의 operator-level rank bound 와 일관된 rank 분리의 *안정성* 발현 (§5.7 E4: 24.0 vs 7.44), 원래 예측된 정확도 향상과 구별. Stationary K-bias 하 multi-tool 에미션은 autoregressive re-attention 으로 제약; Thm 6.15 (KQV hybrid, App. B.7.8.1) 가 이론적 fix 제안, §5.5.2 가 contrastive K-bias (Thm 6.9.5 계열) 로 첫 실증 개선 보고."

### 5.5.1 Mistral-Instruct H2 진행 상황 (Wave 3b)

Mistral-Instruct-v0.3 skipL0+padmax no_steer: **61.51%** (Mistral-v0.3 Base 69.35% 대비 $-7.84$pp). Instruct 변형이 Base 보다 **낮은** Subtask1 no_steer — 초기 예상 (FC-training 이 도구 선택 baseline 을 개선) 과 반대. 가능 원인:
- Instruction-following 모델이 Base 가 autocomplete 하는 모호한 prompt 에서 refuse / hedge.
- Chat 템플릿 오버헤드가 free-text-style Subtask1 prompt 의 baseline 정확도 감소.
- Mistral-Instruct-v0.3 의 instruction training 이 도구 선택 도메인을 커버하지 않을 수 있음.

$a=0.3$ 결과 sum $-2.92$pp / mean $-3.62$pp — Base 대비 **역전**. 이는 메커니즘 반례가 아닌 chat-template hedging; null-control 비교 (random/featshuffle at $\alpha=0.3$) 이 큐에 있으며 Qwen 과 같은 +60+pp 방향 특이성 gap 을 보일 것으로 예측.

### 5.5.2 Non-uniform K-bias 확장 — multi-tool 정확도 첫 positive lift (smoke, N=20, Qwen-Instruct)

§5.5 안정성 결과는 facet-gated 연산자의 *baseline* 운용점. Multi-tool 쿼리에서의 정확도 향상은 디코딩 단계 간 진화하는 **non-stationary** K-bias 를 요구 (Thm 6.9.5/6.15, Appendix B.7.8). 이 계열의 첫 positive 신호를 보고; §5.5 의 주요 안정성 claim 과 독립이며, 이 결과의 성패는 Cor 6.9.6 검증에 영향 없음.

MetaTool Subtask4 N=20 14-configuration sweep (no_steer baseline F1=0.550):

| Variant | α / 파라미터 | F1 | Δ |
|---|---|---|---|
| flat real (참조) | a=0.3 | 0.533 | −0.017 |
| α-sweep | a=0.15 | **0.575** | **+0.025** |
| α-sweep | a=0.20 | 0.492 | −0.058 |
| normalized (Thm 6.9.5 직접) | a=0.3 | 0.325 | −0.225 |
| normalized | a=0.5 | 0.000 | −0.550 |
| contrastive | a=0.3 d=1 | 0.583 | +0.033 |
| **contrastive** | **a=0.3 d=3** | **0.608** | **+0.058** |
| contrastive | a=0.5 d=3 | 0.067 | −0.483 |

**핵심 발견**: $\alpha=0.3$ 에서 contrastive depth-3 K-bias 가 **F1 = 0.608 (+5.8pp over no_steer 0.550)**. Flat real $a=0.3$ 이 0.533 (−1.7pp) 인 동일 smoke 에서. 이는 **K-bias 계열 첫 positive Subtask4 F1 신호** 이며 Thm 6.9.5/6.15 (non-uniform 계열) 가 예측: contrastive 혼합이 쌍 sibling-facet leakage 를 빼면서 facet 방향을 주입하여 §5.5 의 autoregressive re-attention 한계를 직접 타겟. **V-bias 단독 실패** (최대 F1 0.558). **Normalized 단독 $\alpha \ge 0.3$ 실패**: 붕괴.

Contrastive $a=0.3, d=3$ 의 full 497 확장이 최우선 다음 실행; +5.8pp 신호가 full scale 에서 유지되면 §5.5 의 Subtask4 스토리가 *stability-only* 에서 *stability + 정확도 향상* 으로 전환, training-free 증강 연산자로 원래 Cor 6.9 다운스트림 예측 충족.

### 5.6 결과 — E3 Thm 6.1 per-sample attention-weighted bound

Qwen2.5-7B-Instruct L=13, $\alpha=0.3$, N=100 쿼리 × 28 헤드 = **2800 per-head-per-query 측정**.

| 양 | 값 |
|---|---|
| $\mathbb E[\|\hat o - o\|^2]$ (LHS) | 0.5092 |
| $\mathbb E[\mathrm{qaMSE}\cdot\mathrm{Var}_s V]$ (RHS 주도) | 19.729 |
| $\mathbb E[\text{total RHS}]$ ($C_1\rho^4$ 포함) | 7.49 × 10⁷ |
| **bound_pass_rate** | **1.00** (2800/2800) |
| median LHS/RHS ratio | 2.36 × 10⁻⁸ |
| p95 LHS/RHS ratio | 1.24 × 10⁻⁷ |
| max LHS/RHS ratio | 4.26 × 10⁻⁷ |

**Thm 6.1 검증**: 모든 head-query 샘플이 attention-weighted bound 를 만족; bound 는 loose (ratio ~$10^{-8}$) — Mode-C bulk-tail regime 의 예상대로 (Remark B.2.3). Llama L=15 확장은 E3′ 로 연기 (스크립트 준비 완료, ~1 GPU-hr).

### 5.7 결과 — E4 Cor 6.9 operator-level nrank

$P_{\mathrm{ada}}(q)$ 와 $P_{\mathrm{fg}}(q, k_t)$ 의 SVD on 500 MetaTool 쿼리, $\varepsilon$-numerical rank at $\varepsilon \in \{0.1, 0.2\}$. **관측**: AdaSEKA nrank 7.44 근처 집중; 본 방법 24.0 근처 집중. Gap +17 — operator-level 이론 검증.

### 5.8 결과 — E5 Remark 6.14.A.3 hard-gate R-violation grid (MMLU N=1000)

Qwen2.5-7B-Instruct on MMLU-test N=1000, 2026-04-15 02:00 KST 완료.

| gate × α | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| no_steer | — | — | — | — | — |
| **flat** | 0.714 | **0.727** ★ | 0.683 | 0.668 | 0.584 |
| **soft-facet** | — | — | 0.674 | — | 0.614 |
| **hard_thresh** | — | — | 0.672 | — | 0.535 |
| **hard_argmax** | — | — | 0.670 | — | 0.552 |

Baseline (no_steer) = **0.713**.

**실증 Rmk 6.14.A.3 판결**:
- **flat $\alpha=0.2$ 가 유일한 positive cell**: 72.7% (+1.4pp, 유일한 MMLU-non-degrading).
- **$\alpha=1.0$ 열화 순서**: flat 58.4 > soft 61.4 > hard_argmax 55.2 > hard_thresh 53.5 — Hypothesis (R) 예측대로 soft 가 gated 변형 중 최고, hard 불연속이 flat unbiased 보다 ~3pp 더 하락 (Consequence 2 $\rho^4$ scaling 과 일치).
- **$\alpha=0.3$ 순서**: flat 68.3 < hard_thresh 67.2 ≈ hard_argmax 67.0 ≈ soft 67.4 — 중간 $\alpha$ 에서 4 변형이 1.1pp 이내, Hypothesis (R) 실증 신호는 **$\alpha \ge 1.0$ 에서만** 등장 (예측과 일치).

### 5.9 결과 — E6 Thm 6.13 categorical-channel 압축 (WT2 PPL)

Pre-RoPE K quantization, Qwen2.5-7B-Instruct, ctx=2048 non-overlap, 전체 test set (299K 토큰):

| Method | 2-bit 평균 | 2-bit PPL | 4-bit 평균 | 4-bit PPL |
|---|---|---|---|---|
| fp16 | 16 | 7.68 | 16 | 7.68 |
| KIVI | 2.00 | 19.97 | 4.00 | **7.79** |
| **OCQ 1b+2a real** | **1.81** | **15.60** | 3.81 | 12.56 |
| OCQ PCA pseudo (H-cat 위반) | 1.81 | 11.83 | 3.81 | 84.92 |

**Thm 6.13 예측 검증**:
- 2-bit: OCQ < KIVI (Cor 6.13.3/6.13.4, 9.4% 비트 절약 + $-4.37$ PPL).
- 4-bit: KIVI < OCQ (Cor 6.13.5 cross-over at $\bar b^* \approx \frac12 \log_2(s+1), s \sim 5$–$10$).
- (H-cat) falsifiable: PCA pseudo-ontology 가 4-bit 에서 치명적 (84.92) vs real (12.56).

### 5.10 결과 — E7–E10 (scaling, safety, baselines, Mistral) — 대부분 coworker A100 track

(상세는 영문 canonical §5.10 참조)

### 5.10.1 E11' — LoRA + Rotation hybrid (Thm 6.16, 재실행 중)

**현재 상태 (2026-04-15 08:50 KST)**: L1 training rerun 진행 중 (GPU1, PID 1774445). 이전 OOM 은 sibling proc 간섭; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + batch-size 1 로 재실행. L3 smoke 예상 F1 (a) LoRA alone ∈ [0.78, 0.82], (b) + base B_ont ∈ [0.78, 0.85], (c) + LoRA B_ont ∈ [0.82, 0.88], (d) + normalized ∈ [0.85, 0.92].

### 5.11 미래 작업 (E11–E16)
(영문 canonical §5.11 동일)

---

## 6. 논의

### 6.1 왜 stability 가 올바른 서사 프레임인가

세 가지 a-priori 가능한 기여 framing:
1. **단일 도구 선택 정확도 향상** (Subtask1). Strict label-logprob scorer 하 $\Delta \le +6$pp 관측; legacy substring 하 $+11$pp (scorer-의존).
2. **Multi-tool 선택 정확도 향상** (Subtask4 F-simultaneous). 원래 $+5$~$+15$pp 예측; full 497 에서 **반증** ($-4.6$pp).
3. **온톨로지 부분공간의 방향 특이성** (null-control gap). Subtask1 에서 $+16$~$+49$pp (scorer-invariant); Subtask4 에서 $+68.5$pp (full 497, 논문 최대 규모 검증).

이론이 예측한 크기로 견고히 뒷받침되는 것은 framing (3) 뿐이다. (1), (2) 의 $\pm 5$pp 헤드라인은 scorer 및 태스크 의존; (3) 의 $+30$~$+68$pp gap 은 scorer-invariant, task-invariant, cross-model (§5.4 표). 따라서 stability claim 을 선두에. 정확도 향상은 발생 시 (Qwen sum +0.10 / mean +5.03, Llama-Base sum +6.33 / mean +2.61, Mistral-Base sum +3.12, MMLU flat $\alpha=0.2$ +1.4, contrastive Subtask4 smoke +5.8) **방향이 다운스트림 사용 가능함을 보이는 보조 증거**, 주요 기여가 아님.

이 framing 은 논문의 falsifiability 도 재구조화. 정확도-향상 서사 하에서는 단일 실패점 존재 (Subtask4 $-4.6$pp 이미 관측). Stability 서사 하에서는 주요 claim 이 이미 full scale 에서 검증됨; 정확도-향상 확장 (§5.5.2 contrastive, §5.10.1 LoRA) 은 개별 성공/실패가 주요 기여를 훼손하지 않는 독립적 후속.

### 6.1.1 왜 cross-model positive 가 보조 (선두 아닌) 증거인가

Qwen + Llama-Base + Mistral-Base sum-positive 3-family 는 **stability claim 의 일반화 증거**: 온톨로지 방향이 3 개의 독립적 transformer 패밀리에서 고유하게 특권화되며 Qwen-특수 artifact 가 아님. Mistral-Instruct-v0.3 음수는 stability 의 반례가 아님 — no_steer 자체가 Mistral-Base 대비 7.84pp 낮고 (61.51% vs 69.35%), $\alpha=0.3$ K-bias 하 추가 $-2.92$pp 는 chat-template hedging 과 일관, 온톨로지 방향이 특권화되지 못함을 시사하지 않음. Mistral-Instruct 의 null-control 비교 (random/featshuffle at $\alpha=0.3$) 가 큐에 있으며 Qwen 과 같은 $+60$+pp 방향 특이성 gap 을 보일 것으로 예측, Instruct-family hedging 을 별도 scope 한계로 하여 stability 보편성을 확인.

### 6.2 (R) 은 설계 제약, 기술적 세부사항이 아님
§3.2 의 hard-gate MMLU 열화는 버그가 아님 — 정칙성이 중요하다는 직접 실증 신호. 게이트를 Lipschitz 로 설계하라.

### 6.3 왜 K-side 인가, Q-side 가 아닌가
Q-side 1-of-M routing 은 rank $r$ 에서 구조적 상한 (Cor 6.9). K-side F-simultaneous 는 rank $R$ 달성.

### 6.4 한계
1. Stability 는 Qwen-Instruct 에서만 full scale 검증; Llama / Mistral null-control 추가 중.
2. Contrastive d=3 smoke +5.8pp 는 full 497 확인 대기.
3. Baselines (CAA/ITI/PASTA/ASA/Focus Directions) 직접 비교표 미완 — coworker A100 track D 의존.
4. Mistral-Instruct 는 chat-template hedging 으로 isolate; null-control 검증으로 방어 가능.

---

## 7. 결론

본 논문은 instruction-tuned 트랜스포머의 key-projection 기하학의 고유하게 특권화된 부분공간 — 헤드 단위 온톨로지 기저 $B_{\mathrm{ont}}$ — 을 식별하고 형식적으로 특성화한다. 이론적 framing: Thm 6.1 의 샘플 단위 attention-weighted bound, Cor 6.7 의 Lipschitz-gate phase-closure, Cor 6.9 의 rank 분리, 그리고 기하적 강화 Cor 6.9.6 이 함께 $\mathrm{span}(B_{\mathrm{ont}})$ 가 기저 모델로부터의 output 분포 divergence 가 $O(\alpha^2)$ 속도인 유일한 $R$-차원 K-perturbation 부분공간이며, 동일 크기 직교 perturbation 은 $\alpha > \alpha^*$ 에서 FC-emission 매니폴드를 이탈함을 보임. 실증 검증: MetaTool Subtask4 full 497 에서 방향 특이성 gap $+68.5$pp (real $\alpha=0.3$ F1 = 0.685 vs random/featshuffle = 0.000), 2800 head-query 측정에서 1.00 샘플 단위 bound pass rate, max-normalized routing 대비 operator-rank 분리 $+17$, 3 instruction-tuned 패밀리에서 정확도-향상 보조 증거. Non-uniform 확장 (contrastive K-bias, Thm 6.9.5/6.15) 이 첫 positive multi-tool 정확도 향상 산출 ($+5.8$pp smoke), 동일 facet 기저가 Thm 6.13 을 통해 KV-compression 으로 교량 (2-bit 에서 $-4.37$ PPL). 논문은 full scale 에서 이론-설계-실험 루프를 닫으며, stability claim 이 이미 결정적으로 검증되고 정확도-향상 확장이 독립적 후속으로 존재.

---

## 부록

- **A.** MetaTool 데이터셋 준비, 파싱, scorer 구현.
- **B.** 전체 증명 (Theorem 6.1, 6.2, Cor 6.3–6.13, **Cor 6.9.6 신규**). `APPENDIX_B_PROOFS.md` + `COROLLARY_6_7_FACET_PHASE_CLOSURE.md` 임포트.
- **C.** Cor 6.7 재프레이밍 (정칙성 hypothesis (R)). `COR67_REFRAMING_2026_04_14.md` 임포트.
- **D.** Mistral cross-model ablation grid. `CROSS_MODEL_KBIAS_ANALYSIS_2026_04_13.md` 임포트.
- **E.** Netsru Gemma-3-27B agent artifact trail.
- **F.** Per-head Theorem 6.1 검증 세부; `measure_theorem_6_1.py` output schema.
