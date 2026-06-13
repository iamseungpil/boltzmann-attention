# Deep-Research: LLM 에이전트 프레임워크 메트릭 배터리 — 문헌 표준 조사 (2026-06-12)

> **목적**: `EXPERIMENT_DESIGN.md` §1.6 framework-tier 메트릭 배터리(F1–F7, 사전등록 v1)의 추정량·CI 절차를 문헌 표준으로 치환해 **v2 동결**하기 위한 근거 조사.
> **인용 규율**: 본 보고서의 Verified bibliography(§4) 전 항목은 **이 세션에서 arXiv abs/원문/공식 문서 fetch로 직접 검증**(제목·저자·버전·핵심 주장). 수식은 원문(또는 ar5iv 전문)에서 추출. 검증 못한 항목은 §5 Unverified leads로 격리.
> 범위 근거 문서: `EXPERIMENT_DESIGN.md` §1.5–1.6, `BENCH_PORTFOLIO_FRAMEWORK_DESIGN.md` §1·§3.5–3.8 (τ² retail 게이트 44→0·pass^1 parity 실측 포함).

---

## 1. 핵심 답변 요약 — §1.6 v2 동결 권고

| # | v1 상태 | **v2 권고** | 근거 |
|---|---|---|---|
| F1 어댑터 비용 | 신규 발명·novelty 위험 | **유지(우리 발명 명시)** — 문헌에 per-benchmark 어댑터 비용 곡선의 표준 메트릭 **없음 확인**. 가장 가까운 선례 = HAL의 3차원(모델×scaffold×벤치) 분해 [HAL, 2510.11977]가 "scaffold가 결과를 좌우한다"를 정량화하나 *비용* 축은 아님. census-tier 한정 유지 | §2.4 |
| F2 전이 보존율 | 관행 정합 | **유지** + per-domain 보고(집계 단일값 금지) + task-level bootstrap CI. 교차벤치 평균 금지는 §2.5의 집계 규율 적용 | §2.5 |
| F3 일관성 | τ-bench 계보 | **확정**: τ-bench pass^k의 unbiased 추정량 `E_task[C(c,k)/C(n,k)]` 채택(n=4, k=1..4) — τ² 리더보드 공식 프로토콜(≥4 trials, Pass^1–4)과 정확히 일치. ⓟ1 판정은 paired(task-matched) bootstrap difference-in-differences. 민감도 분석으로 G-Pass@k_τ [2412.13147] 옵션 | §2.1 |
| F4 무위반 | 통계 절차 대기 | **확정**: 위반 0/N → **one-sided 95% Clopper-Pearson 상한 `1−0.05^{1/N}` (≈ rule of three 3/N)** 을 주 보고, Jeffreys 상한을 민감도로. **단, "구조적 0"(게이트 구성상 불가능)과 "표본적 0"(측정상 미관측)을 분리 보고** — CI는 후자에만 의미 있음(아래 §2.2 핵심 통찰) | §2.2 |
| F5 선별 회수율 | 표준명 대기 | **표준명 없음 확인 → "우리 발명" 명시 유지**. (sel−mean)/(oracle−mean) 정규화의 문헌 표준명을 발견하지 못함(가장 가까운 구조 = E-AURC의 oracle-정규화 [1805.08206]). ⓟ2(paired bootstrap 95% CI 동반)는 Koehn 2004 paired bootstrap resampling 계보로 정당화 | §2.3, §2.5 |
| F6 abstain 품질 | 표준(selective prediction) | **확정**: RC-curve + AURC + **E-AURC**(oracle-정규화) [1805.08206] + **coverage@risk≤r\*** (배포 친화 단일점, Geifman & El-Yaniv 2017의 risk-control 관점). 에이전트 변형(abstain→handoff)은 2025–26에 활발하나 아직 단일 표준 없음 — selectively-quitting [2510.16492]·Trust-or-Escalate [2407.18370] 인용으로 포지셔닝 | §2.3 |
| F7 비용 | 문헌 확정 대기 | **확정**: ①토큰(모델-불변) + USD(가격 스냅샷 날짜 명시) per trajectory — τ² 리더보드 공식 cost 필드와 호환 ②**cost-of-pass = 기대비용/정답률** [2504.13359] ③accuracy×cost **Pareto frontier** 플롯 [2407.01502, HAL]. 우리 확장 "cost-of-consistent-pass"(비용/pass^4)는 발명 표시 | §2.4 |

**오늘 결과("write-violation 44→0, pass^1 parity")의 표준 보고형** (§2.2.4): ① matched task set에서 **paired Δpass^1 + bootstrap 95% CI**(parity는 "CI가 0 포함"이 아니라 **CI 폭 자체를 명시**: "Δ = +0.23pp, 95% CI [−x, +y]") ② 위반은 **0/N (N=게이트 관할 write-기회 수, 사전등록 census)** + one-sided 95% 상한 3/N ③ 게이트 관할 내 0은 **구조적 보장**(게이트 spec 정확성 검증으로 뒷받침, Guard-2 패턴)이고 CI는 **관할 밖 잔여 위반**에만 적용됨을 명시.

**프로토콜 드리프트 경고(즉시 실무 영향)**: τ²-bench 공식 리더보드는 user-sim 모델을 **리더보드에 병기**하며 현재 **gpt-5.2를 권장** — 우리 run7은 gpt-4.1 계열 judge/sim. 리더보드 숫자와의 직접 비교는 user-sim 명시 없이는 무효(§2.5.3). 내부 ±게이트 비교(internal-consistent)는 영향 없음.

---

## 2. Research Questions별 상세

### 2.1 RQ1 — pass@k vs pass^k: 정확한 정의·분산·trial 수

#### 2.1.1 Chen et al.의 unbiased pass@k (HumanEval 계보)

[Chen et al. 2021, arXiv:2107.03374v2] §2.1 — 원문 검증(ar5iv 전문):

```
pass@k := E_Problems[ 1 − C(n−c, k) / C(n, k) ]
```

- n = 문제당 생성 샘플 수, c = 그중 정답 수, k ≤ n. **"k개 중 적어도 1개 성공" 확률의 unbiased 추정량**.
- 원문이 명시한 함정: naive 추정 `1−(1−p̂)^k` (p̂=경험적 pass@1)은 **"consistent underestimate"** (원문 Fig.13; n이 5k를 넘어도 바이어스 잔존).
- 수치 안정 구현(원문 코드):
  ```python
  1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
  ```
- 원문 세팅: **n=200, k≤100** — 즉 n ≫ k 체제에서 사용.

#### 2.1.2 τ-bench의 pass^k — "전부-성공" 일관성 지표

[Yao, Shinn, Razavi, Narasimhan, τ-bench, arXiv:2406.12045v1, ICLR 2025 (OpenReview roNSXZpUDN)]:

> pass^k = "the probability that it succeeds on all k independent trials of a task", 추정량:

```
p̂ass^k := E_task[ C(c, k) / C(n, k) ]      (n trials, c successes, k ≤ n)
```

- pass@k와 정확히 대칭: pass@k = "at least one of k", pass^k = "**all** of k". 둘 다 **초기하(hypergeometric) 기반 unbiased 추정량** — n개 trial에서 k개를 비복원 추출했을 때 전부 성공일 확률의 기대값이 정확히 `C(c,k)/C(n,k)`.
- τ-bench 원문은 **8 trials**로 pass^8까지 보고(GPT-4o retail pass^8 < 25% vs pass^1 ≈ 61% = 일관성 결핍의 정량화).
- ⚠️ 검증 메모(정직): 이 수식의 원문 PDF 직접 추출은 실패(인코딩); arXiv abs로 논문·metric 존재 검증 + 2차 소스(EmergentMind topic page) 인용문 + **독립 교차검증**: G-Pass@k 논문 [2412.13147v5, ar5iv 전문 검증]이 base case로 **동일 수식** `G-Pass@k = E_Questions[C(c,k)/C(n,k)]`를 명기.

#### 2.1.3 일반화: G-Pass@k_τ (부분-일관성 허용)

[Liu et al., "Are Your LLMs Capable of Stable Reasoning?", arXiv:2412.13147v5] — ar5iv 전문에서 수식 검증:

```
G-Pass@k_τ = E_Questions[ Σ_{j=⌈τ·k⌉}^{c}  C(c,j)·C(n−c, k−j) / C(n,k) ]
mG-Pass@k  = 2 ∫_{0.5}^{1.0} G-Pass@k_τ dτ      (τ-곡선 AUC 요약)
```

τ=1.0이면 τ-bench pass^k와 일치. **k개 중 ⌈τk⌉개 이상 성공** 확률 — "4/4 전부"가 너무 가혹할 때(분모 작은 우리 체제) τ=0.75(3/4) 곡선이 유용한 민감도 분석. 단 헤드라인은 τ² 리더보드 호환을 위해 pass^k(τ=1) 유지.

#### 2.1.4 분산과 trial 수 — 우리 스케일(114 tasks × 4 trials)

**문헌 검증 결과**: pass@k 추정의 분산 분석은 최근에야 본격화 [Kazdan et al., "Efficient Prediction of Pass@k Scaling", arXiv:2510.05197v1 — beta-binomial 피팅·표본 한계 체제 분석]. "n ≥ 2k–4k" 같은 경험칙은 블로그 수준만 발견(§5 leads) — **공식 표준 없음**. 따라서 아래는 **자체 유도(파생임을 명시)**:

- per-task 추정량 `C(c,k)/C(n,k)`는 unbiased이므로 교차-task 평균의 분산은 `(1/T²)Σ_t Var_t`. **k=n(우리 pass^4, n=4)일 때 추정량은 indicator(c=4)로 퇴화** → per-task Bernoulli, 전체 SE = `sqrt(P₄(1−P₄)/T)`.
  - T=114, P₄≈0.1 → SE ≈ 2.8pp (95% CI ≈ ±5.5pp). P₄≈0.2 → ±7.4pp. **단일 arm 절대값엔 굵은 CI**.
  - k<n이면 `C(n,k)`개 부분집합 평균이라 분산이 그보다 작음(예: pass^2는 C(4,2)=6개 쌍 평균 = naive 2-split보다 효율적) — unbiased 추정량을 쓰는 실질 이득.
- **ⓟ1 판정(Δpass^4 > Δpass^1)은 paired로 설계해야 검정력 확보**: 게이트±가 같은 114 task에 매칭되므로, task-level **paired bootstrap**으로 `(Δpass^4 − Δpass^1)`의 difference-in-differences CI를 직접 산출(10k resample). 비페어드 비교는 위 ±5~7pp 노이즈에 묻힘.
- **소표본 CI 일반론**: [Bowyer, Aitchison, Ivanova, arXiv:2503.01747v3, ICML 2025 spotlight] — **"수백 datapoint 미만에서 CLT 기반 오차막대는 불확실성을 극적으로 과소추정"** → frequentist(exact)/Bayesian 대안 권고. T=114는 정확히 이 경고 체제. → CLT-SE 단독 보고 금지, bootstrap(percentile) 또는 Wilson/Jeffreys 병기.
- 분산 구조 일반론: [Miller, "Adding Error Bars to Evals", arXiv:2411.00640v1] — 평가 문항을 super-population에서 추출된 표본으로 개념화, 두 모델 차이는 paired 분석, (멀티-trial은 task로) **클러스터링** 처리. 우리 456 trajectory는 114 클러스터 — pass^1 SE도 **task-cluster bootstrap**이 정답(trial을 iid 456개로 취급 금지).

#### 2.1.5 τ²/τ-bench 리더보드의 실제 보고 방식 (공식 문서 fetch 검증)

`sierra-research/tau2-bench` 공식 `docs/leaderboard-submission.md` (2026-06-12 fetch):
- **"we strongly prefer results with at least 4 trials per domain for statistical reliability"** → **Pass^1–Pass^4 보고**(미실시 항목 null 허용; voice는 Pass^1만 관행).
- **전 task·base split 강제**(`--task-ids`/`--num-tasks` 필터 금지) — 우리 retail 114 전수와 일치.
- agent-llm·user-llm·인자 **전 run 동일** + **user-sim 모델이 리더보드에 병기됨**. "We recommend using `gpt-5.2` as the user simulator for the most accurate results."
- cost는 **선택 필드: 도메인당 평균 USD/trajectory**.

⇒ **F3 v2 = n=4, pass^1..pass^4 곡선 = 리더보드 프로토콜과 1:1 호환** (우리가 추가하는 것은 CI와 paired 판정뿐).

### 2.2 RQ2 — 위반 0건의 신뢰구간: rule of three / Clopper-Pearson / Jeffreys

#### 2.2.1 Rule of three (계보의 기원)

[Hanley & Lippman-Hand, "If nothing goes wrong, is everything all right? Interpreting zero numerators", JAMA 249(13):1743–1745, 1983 — PubMed 6827763 검증]: n건 중 사건 0건 관측 시 **95% 신뢰 상한 ≈ 3/n** (n>30). 유도: one-sided exact 상한 `p_U = 1 − α^{1/n}`, α=0.05 → `1 − 0.05^{1/n} ≈ 3/n` (−ln 0.05 ≈ 3.0). 임상 안전성 보고의 사실상 표준.

#### 2.2.2 Exact(Clopper-Pearson)와 Jeffreys — 통계학 권위 비교

[Brown, Cai & DasGupta, "Interval Estimation for a Binomial Proportion", Statistical Science 16(2):101–133, 2001 — projecteuclid 검증]:
- **Wald 구간은 소표본·경계(p≈0)에서 커버리지 붕괴** — "chaotic coverage properties … far more persistent than appreciated". p≈0인 위반율에 Wald/CLT 사용 금지.
- **Clopper-Pearson(1934, exact)**: 모든 n·p에서 명목 커버리지 보장하나 **"wastefully conservative"**. x=0일 때 닫힌형: one-sided 95% 상한 = `1 − 0.05^{1/N}` (≈3/N), two-sided 95%의 상한 = `1 − 0.025^{1/N}` (≈3.69/N).
- **권고**: 소 n에서 **Wilson 또는 equal-tailed Jeffreys**, 큰 n에서 Agresti-Coull. Jeffreys = Beta(½,½) prior → x=0이면 posterior Beta(½, N+½), 95% 상한 = `qbeta(0.95; 0.5, N+0.5)` ≈ 1.9/N (우리 파생 수치; N=100 → ≈1.9%).

#### 2.2.3 LLM 안전/컴플라이언스 평가에서의 실태

- [Bowyer et al., 2503.01747v3]: 소표본 LLM eval에서 CLT 금지·exact/Bayesian 권고 (ICML 2025 spotlight) — **우리 F4가 따를 1차 권위**.
- [Beyer et al., "LLM-Safety Evaluations Lack Robustness", arXiv:2503.02574v2]: 안전 평가 전반이 "small datasets, methodological inconsistencies, unreliable evaluation setups"로 노이즈 — 불확실성 보고 자체가 비표준임을 방증.
- 개별 시스템: Clopper-Pearson을 쓰는 인증형 평가(QuaCer-B 등)는 2차 소스로만 확인(§5). **종합: 분야 표준은 미성숙 — 임상 통계(rule of three)와 통계학 정론(Brown et al.)을 직수입하는 것이 가장 방어적.**

#### 2.2.4 F4 v2 보고형 — "compliance free at pass^1" 클레임의 표준형

**핵심 통찰(이번 조사에서 가장 중요한 설계 결정)**: 결정론 게이트의 0은 두 종류다.
1. **구조적 0(structural zero)**: 게이트 관할 내 위반은 *구성상* 불가능(deny가 차단). 여기에 표본 CI를 붙이는 것은 범주 오류 — 올바른 뒷받침은 **게이트 spec의 정확성 검증**(GATE_SPEC vs 정책 원문 대조, Guard-2 mirror 패턴) + 관할 census(write-기회 중 게이트가 본 비율).
2. **표본적 0(sampled zero)**: end-to-end 측정에서 잔여 위반(게이트 관할 밖 위반 분류군 포함) 0건 → **여기에 0/N 상한 CI 적용**.

**권고 보고형(사전등록)**:
- 분모 N = **사전등록된 "위반 기회" census**(예: 정책 관할 write-시도 수; trajectory 수 아님 — 기회 단위가 임상 계보의 분모와 동형). N 정의를 v2에 박제.
- 보고: "violations 0/N; one-sided 95% upper bound `1−0.05^{1/N}` (rule of three ≈ 3/N) [Hanley & Lippman-Hand 1983; Brown et al. 2001], Jeffreys 95% upper `qbeta(.95,.5,N+.5)` 병기".
- 예시 수치: N=300 기회 → 상한 1.0%/0.64%(CP/Jeffreys); N=44 → 6.6%/4.2%. **"0이지만 상한 x%"가 헤드라인 문장형**.
- pass^1 parity 쪽: matched 112-task에서 **paired Δpass^1 + task-bootstrap 95% CI** — "parity"는 점추정 근접이 아니라 **CI 폭 공개**로만 주장(우리 0.1853 vs 0.1830 = Δ+0.23pp는 CI가 수 pp일 것 → "no measurable cost, CI [−a,+b]"로 서술). 동일 trial 쌍이 있으므로 McNemar식 discordant-pair 카운트 병기 가능.
- 클러스터 주의 [Miller 2411.00640]: 위반 기회는 trajectory 내 상관 → N을 iid로 취급한 상한은 약간 낙관적일 수 있음 — trajectory-level 0/M(M=trajectory 수) 상한을 보수적 병기로 제공.

### 2.3 RQ3 — AURC / risk-coverage: 정의와 에이전트 변형

#### 2.3.1 표준 정의 (원문 검증)

- **Selective prediction 기초** [Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks", NeurIPS 2017 — proceedings 페이지 검증]: 예측기 f + 선택함수 g; **selective risk = (수락 표본의 경험 손실)/coverage**; 신뢰도 임계로 **목표 risk를 고신뢰 보장**(SGR) — "coverage@guaranteed-risk" 관점의 원조.
- **AURC / E-AURC** [Geifman, Uziel & El-Yaniv, "Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers", arXiv:1805.08206v4, ICLR 2019 — ar5iv 전문에서 수식 검증]:

```
AURC(κ, f | V_n)   = (1/n) Σ_{θ∈Θ} r̂(f, g_θ | V_n)
   — 신뢰도 κ로 정렬해 생기는 모든 임계 θ에서의 selective risk 평균 (RC-곡선 아래 면적)
E-AURC(κ, f | V_n) = AURC(κ, f | V_n) − AURC(κ*, f | V_n)
   — κ* = 오답을 정답 아래로 완벽 정렬하는 oracle; E-AURC=0이 최적, [0,1] 무단위
```

- **AURC를 1차 지표로 격상한 권위** [Jaeger et al., "A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification", arXiv:2211.15259, ICLR 2023 oral]: calibration/OOD/selective prediction으로 갈라진 평가들을 "분류기 실패 검출"로 통합, **AURC를 주 지표로 권고**.

#### 2.3.2 에이전트 적응 (abstain → human handoff)

2024–26 에이전트 문헌에서 risk-coverage의 직접 이식은 아직 표준화 전이나 세 계보가 검증됨:
- [Bonagiri et al., "Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety", arXiv:2510.16492v3 (2026-02)]: **에이전트가 궤적 중간에 quit** — safety–helpfulness trade-off로 보고(ToolEmu, 12 LLM). 안전 +0.39/3.0, helpfulness −0.03 = "abstain이 거의 무료" 형태의 주장 구조가 우리 게이트 클레임과 동형.
- [Jung, Brahman & Choi, "Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement", arXiv:2407.18370, ICLR 2025]: **선택적 평가 + 단계적 escalation**(약한 모델→강한 모델→인간)에 **사용자 지정 보장 수준** — abstention→handoff 파이프라인의 provable 버전. F6에서 "abstain의 목적지가 인간"임을 포지셔닝할 때 인용.
- [Wen et al., "Know Your Limits: A Survey of Abstention in LLMs", arXiv:2407.18418v3, TACL]: abstention을 query/model/human-values 3관점으로 정리한 우산 서베이 — 평가 방법론 카탈로그.
- 보조: [Rabanser et al., 2602.16666v3 — §2.4 참조]의 Predictability 차원이 ECE/AUROC/Brier로 "신뢰도가 성공을 분별하는가"를 측정 — AURC의 에이전트 인접 변형.

#### 2.3.3 F6 v2 권고

- GROUNDED_BIZ abstain-GT 위에서: 신뢰도 score(게이트/선별기 산출) 기준 **RC-곡선 + AURC + E-AURC** 보고(E-AURC가 oracle-정규화라 task mix 간 비교 가능 — F5의 정규화 철학과 동일 계열).
- 배포 단일점: **coverage@risk≤r\*** (r\* 사전등록, 예: 위반·오답 위험 ≤5%에서 자동처리 커버리지 몇 %) — Geifman 2017의 보장-risk 관점이 "abstain=인간 핸드오프 비용" 서사에 직결.
- abstain 자체의 빈도는 지표가 아님(전부 abstain하면 risk 0) — **반드시 곡선/E-AURC로**.

### 2.4 RQ4 — 비용 곡선·비용-정규화 보고

검증된 계보 (전부 abs fetch):
- [Kapoor, Stroebl, Siegel, Nadgir, Narayanan, "AI Agents That Matter", arXiv:2407.01502v1, 2024]: ①정확도 단독 보고 비판 — **"jointly optimizing accuracy and cost"**, cost-controlled 평가 ②벤치 과적합·holdout 부재·재현성 부재 비판. 에이전트 비용 보고 의제의 출발점.
- [Erol, El, Suzgun, Yuksekgonul, Zou, "Cost-of-Pass: An Economic Framework for Evaluating Language Models", arXiv:2504.13359v2 (v2 2026-02-26)]: **cost-of-pass = "the expected monetary cost of generating a correct solution"** (≈ 평균 시도비용/정답률), **frontier cost-of-pass** = 가용 모델·인간 전문가 풀에서의 최소 cost-of-pass — "경제적 생산성" 단위.
- [Kapoor, Stroebl, Kirgis et al., "Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation", arXiv:2510.11977v1, 2025]: 표준화 harness·**모델×scaffold×벤치 3차원**·21,730 rollouts/$40k·LLM-aided log inspection(벤치 게이밍 탐지). 비용을 1급 축으로 둔 리더보드 인프라의 현행 표준.
- τ² 리더보드 공식 cost 필드 = **도메인당 평균 USD/trajectory** (§2.1.5).

**F7 v2 권고**: ①**토큰 수**(모델-불변·가격변동 면역)를 1차, USD는 가격 스냅샷 날짜 명시로 2차 ②**cost-of-pass = E[비용/trajectory]/pass^1** [2504.13359 형식] ③arm 비교는 **success×cost Pareto 플롯**(게이트·선별·guided가 각각 비용을 얼마나 추가하고 점수를 얼마나 사는가) [2407.01502 형식] ④우리 확장 "**cost-of-consistent-pass**"(비용/pass^4) = 일관성까지 산 비용 — **발명 표시 필수**(문헌 없음). F1(어댑터 비용 곡선)은 검색 결과 표준 부재 확인 — HAL의 scaffold 차원이 "scaffold가 성능 좌우"를 보였을 뿐 **개발-비용 정량화 메트릭은 없음** → v1대로 "우리 발명·census-tier 한정" 유지.

### 2.5 RQ5 — 교차-벤치 집계 규율과 프로토콜 드리프트

#### 2.5.1 집계 방법의 표준과 함정

- [Liang, Bommasani et al., "Holistic Evaluation of Language Models (HELM)", arXiv:2211.09110v2, 2022]: 멀티-metric(정확도·calibration·robustness·공정성·bias·toxicity·효율) × core scenarios 표준화. 집계는 mean win rate 계열.
- [Nitsure, Mroueh et al., "Risk Aware Benchmarking of Large Language Models", arXiv:2310.07132, ICML 2024]: **mean-win-rate 집계 비판** — 실패 모드를 무시·모델 풀에 따라 순위 변동. 대안 = 1·2차 **stochastic dominance** 기반 통계 검정 + metrics-portfolio 순위(계량금융 mean-risk 모델 차용).
- [Perlitz et al., "Do These LLM Benchmarks Agree? Fixing Benchmark Evaluation with BenchBench", arXiv:2407.13696v2]: **Benchmark Agreement Testing(BAT)** — 벤치 간 비교에서 "간과된 방법론 선택(정규화·모델 풀 선택 등)이 결론을 뒤집음"을 40+ 벤치로 실증, 표준 절차 + BenchBench 패키지.
- [Miller, arXiv:2411.00640v1]: 문항=super-population 표본 → 벤치별 SE 동반, 모델 간 차이는 paired. (집계 이전에 per-bench 불확실성이 선행.)
- [Bowyer et al., 2503.01747v3]: 소형 벤치(<수백 문항) CLT 금지.

**종합 규율(F-tier 전체에 적용)**: ①교차-벤치 단일 평균 금지 — per-bench 네이티브 헤드라인(우리 2-tier 규율의 tier-1과 정확히 일치) ②집계가 필요하면 win-rate/rank 계열 + **모델 풀 의존성·크기 둔감성 한계 명시** [2310.07132] ③F2/F3 같은 framework-tier 지표는 **벤치별로 따로 보고하고 "방향 일치 횟수"(sign consistency)로만 종합** — z-score 평균도 분포 가정이 깨지는 소 task 수에서 위험.

#### 2.5.2 프로토콜 드리프트 — user-sim이 1급 프로토콜 변수

- τ² 공식 문서(§2.1.5): user-sim 모델은 자유지만 **리더보드에 병기**, 권장 gpt-5.2; agent/user 인자 전 run 동일 강제. (τ-bench 1세대 README는 gpt-4o가 기본 user 모델 — 세대 간에도 이미 드리프트.)
- [Zhou et al., "Mind the Sim2Real Gap in User Simulation for Agentic Tasks", arXiv:2603.11245v1, 2026]: **실인간 451명 × 165 task vs LLM 시뮬레이터 31종** — 시뮬 유저는 "excessively cooperative, stylistically uniform, lack realistic frustration" → **점수 상방 편향**; User-Sim Index(USI) 제안; 시뮬레이터 능력↑ ≠ 충실도↑. ⇒ **user-sim 버전이 다르면 절대 점수 비교는 원천 무효**라는 직접 증거.
- [HAL, 2510.11977]: 드리프트의 구조적 해법 = 단일 harness로 전 모델 재실행(제3자 재평가). 우리 R8(내부-일관 비교)와 동일 철학.

**v2 규율**: 리더보드 인용 시 (user-sim 모델·버전, judge 모델, trial 수, task split) 4-튜플을 함께 인용; 하나라도 다르면 "비교 불가" 표기. **우리 run7 (judge=gpt-4.1-2025-04-14)은 4-튜플 명시로만 리더보드 옆에 놓을 수 있음** — 게이트 효과 주장 자체는 internal-paired라 면역.

#### 2.5.3 F5의 paired bootstrap 근거

[Koehn, "Statistical Significance Tests for Machine Translation Evaluation", EMNLP 2004 — ACL Anthology W04-3250, 제목·저자·venue 검증]: 시스템 쌍 비교의 paired bootstrap resampling 표준 출처(NLP 계보). ⓟ2(±단일점 금지, paired bootstrap 95% CI 동반)의 인용 근거로 채택. (전문 텍스트 추출은 실패 — 방법 귀속은 통설이며 제목·venue만 fetch 검증임을 명시.)

---

## 3. 제안 §1.6 v2 표 (동결안)

| # | 축 | 추정량 (수식) | CI 절차 | 보고형 | 문헌 지위 |
|---|---|---|---|---|---|
| F1 | 어댑터 비용 곡선 | 벤치당 수동 LOC/시간 + 기계화율% | — (census) | 벤치 순서별 곡선, headline 금지 | **우리 발명(표준 부재 확인)** |
| F2 | 전이 보존율 | held-out/in-domain 공식 success 비 (재학습0) | per-domain task-bootstrap 95% CI | 도메인별 표 + 집계는 sign-consistency만 | 관행 정합 [HELM·HAL 표준화 철학] |
| F3 | 일관성 | **p̂ass^k = E_task[C(c,k)/C(n,k)]**, n=4, k=1..4 [τ-bench 2406.12045; 동형 2412.13147] | task-level cluster bootstrap (10k); ⓟ1은 paired diff-in-diff CI | pass^1..4 곡선 ±CI; Δpass^4 vs Δpass^1 분리; (민감도) G-Pass@4_{0.75} | 표준 (τ² 리더보드 프로토콜 호환) |
| F4 | 무위반 soundness | 위반 건수/사전등록 기회 N; 0이면 **상한 1−0.05^{1/N} ≈ 3/N** | CP one-sided 95% 주, Jeffreys `qbeta(.95,.5,N+.5)` 병기 [Hanley 1983; Brown et al. 2001] | "0/N, 95% UB x%" + **구조적 0(게이트 spec 검증) vs 표본적 0 분리** | 표준 (임상→직수입) |
| F5 | 선별 회수율 | (sel−mean)/(oracle−mean) | **paired bootstrap 95% CI** [Koehn 2004 계보] (ⓟ2) | 점추정+CI, oracle·mean 절대값 병기 | **우리 발명(표준명 부재 확인; E-AURC 정규화와 동계열로 포지셔닝)** |
| F6 | abstain 품질 | RC-곡선; **AURC = (1/n)Σ_θ r̂(f,g_θ)**; **E-AURC = AURC−AURC(κ*)** [1805.08206] | task bootstrap on AURC | AURC·E-AURC + **coverage@risk≤r\*** (r\* 사전등록) | 표준 (selective prediction; 에이전트 변형은 2510.16492·2407.18370 인용) |
| F7 | 비용-정규화 | 토큰/trajectory(1차)·USD(스냅샷 날짜); **cost-of-pass = E[cost]/pass^1** [2504.13359] | bootstrap CI | success×cost Pareto 플롯 [2407.01502]; (발명) cost-of-consistent-pass = E[cost]/pass^4 | 표준 + 발명 1 |

**사전등록 예측 v2**: ⓟ1 유지(판정 절차를 paired diff-in-diff bootstrap으로 구체화). ⓟ2 유지(Koehn 계보 인용 확보). **추가 ⓟ3(권고)**: F4 보고에서 게이트 관할 census(전체 write-기회 중 게이트 적용 비율)가 ≥95%일 것 — 구조적 0 주장의 전제 조건을 수치화.

---

## 4. Verified bibliography (전부 이 세션에서 fetch 검증)

검증 방법 표기: [A]=arXiv abs 페이지, [F]=원문/ar5iv 전문 수식 추출, [P]=publisher/PubMed/proceedings, [D]=공식 repo 문서.

1. Evaluating Large Language Models Trained on Code, M. Chen et al., arXiv:2107.03374**v2**, 2021. [A][F: pass@k 수식·numpy·n=200]
2. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains, S. Yao et al., arXiv:2406.12045**v1**, 2024; ICLR 2025 (OpenReview roNSXZpUDN). [A][P] (pass^k 수식은 2차 소스+동형 수식 교차검증 — §2.1.2 메모)
3. τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment, V. Barres et al., arXiv:2506.07982**v1**, 2025. [A]
4. tau2-bench 공식 `docs/leaderboard-submission.md`, sierra-research (GitHub), 2026-06-12 fetch. [D: ≥4 trials·Pass^1–4·user-sim 병기·gpt-5.2 권장·cost 필드]
5. Are Your LLMs Capable of Stable Reasoning?, J. Liu et al., arXiv:2412.13147**v5**, 2024–25. [A][F: G-Pass@k_τ·mG-Pass@k 수식]
6. If nothing goes wrong, is everything all right? Interpreting zero numerators, J.A. Hanley & A. Lippman-Hand, JAMA 249(13):1743–1745, 1983. [P: PubMed 6827763]
7. Interval Estimation for a Binomial Proportion, L.D. Brown, T.T. Cai & A. DasGupta, Statistical Science 16(2):101–133, 2001. [P: Project Euclid 10.1214/ss/1009213286]
8. Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints, S. Bowyer et al., arXiv:2503.01747**v3**, ICML 2025 spotlight. [A]
9. Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations, E. Miller, arXiv:2411.00640**v1**, 2024. [A]
10. Selective Classification for Deep Neural Networks, Y. Geifman & R. El-Yaniv, NeurIPS 2017. [P: proceedings.neurips.cc]
11. Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers, Y. Geifman, G. Uziel & R. El-Yaniv, arXiv:1805.08206**v4**, ICLR 2019. [A][F: AURC·E-AURC 수식]
12. A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification, P.F. Jaeger et al., arXiv:2211.15259, ICLR 2023 (oral). **[F]**(2026-06-14 전문 정독, relwork_metrics §1.12: verbatim "We propose to use AURC as the primary metric" 확인 → 스니펫 태그서 승격) [+P: iclr.cc oral 페이지 확인]
13. Know Your Limits: A Survey of Abstention in Large Language Models, B. Wen et al., arXiv:2407.18418**v3**, TACL. [A]
14. Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety, V.K. Bonagiri et al., arXiv:2510.16492**v3**, 2025–26. [A]
15. Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement, J. Jung, F. Brahman, Y. Choi, arXiv:2407.18370, ICLR 2025. [A]
16. AI Agents That Matter, S. Kapoor, B. Stroebl, Z.S. Siegel, N. Nadgir, A. Narayanan, arXiv:2407.01502**v1**, 2024. [A]
17. Cost-of-Pass: An Economic Framework for Evaluating Language Models, M.H. Erol et al., arXiv:2504.13359**v2** (2026-02-26). [A]
18. Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation, S. Kapoor, B. Stroebl, P. Kirgis et al. (31인), arXiv:2510.11977**v1**, 2025. [A]
19. Towards a Science of AI Agent Reliability, S. Rabanser, S. Kapoor, P. Kirgis, K. Liu, S. Utpala, A. Narayanan, arXiv:2602.16666**v3**, ICML 2026. [A][F: 12 metrics·K=5 runs·Cout 수식]
20. Holistic Evaluation of Language Models (HELM), P. Liang, R. Bommasani et al., arXiv:2211.09110**v2 (rev 2023-10-01**; 최초 제출 2022-11). [A] — 4-tuple/버전 규율상 인용 시 **v2 개정일 2023-10-01 핀**(relwork_metrics §1.10).
21. Do These LLM Benchmarks Agree? Fixing Benchmark Evaluation with BenchBench, Y. Perlitz et al., arXiv:2407.13696**v2**, 2024. [A]
22. Risk Aware Benchmarking of Large Language Models, A. Nitsure, Y. Mroueh et al., arXiv:2310.07132, ICML 2024. [A(부분)+검색으로 저자·venue 확인(IBM Research 페이지)]
23. Mind the Sim2Real Gap in User Simulation for Agentic Tasks, X. Zhou et al., arXiv:2603.11245**v1**, 2026. [A]
24. Statistical Significance Tests for Machine Translation Evaluation, P. Koehn, EMNLP 2004 (ACL W04-3250). [P: 제목·저자·venue만; 전문 미추출 — paired bootstrap 귀속은 통설]
25. Position: LLM-Safety Evaluations Lack Robustness, T. Beyer et al., arXiv:2503.02574**v2**, 2025–26. [A]
26. Efficient Prediction of Pass@k Scaling in Large Language Models, J. Kazdan et al., arXiv:2510.05197**v1**, 2025. [A]

## 5. Unverified leads (인용 금지 — 후속 검증 후보)

- **"n ≥ 2k–4k면 pass@k 안정"** 경험칙: 블로그(leehanchung.github.io 2025-09)·검색 합성에서만 발견. 공식 출처 미확인 — v2에는 자체 분산 유도(§2.1.4)로 대체.
- **pass@k = U-statistic, Hoeffding 점근 분산**: 검색 합성에 등장(아마 2510.05197 또는 2505.15201 본문) — 본문 미확인.
- **QuaCer-B**(Clopper-Pearson 기반 LLM bias certifier): 2차 리뷰 사이트로만 확인.
- **Clopper & Pearson 1934 (Biometrika 26:404–413)**: 원 논문 미fetch — Brown et al. 2001 경유로만 인용할 것.
- **Vote'n'Rank**(사회선택이론 벤치 집계, arXiv:2210.05769 추정)·**AbstentionBench**(Meta 2025 추정): 미검증.
- **Pass@K Policy Optimization** (arXiv:2505.15201): RL 학습용 pass@k — abs 미fetch.
- **Efficient Benchmarking of AI Agents** (arXiv:2603.23749)·**Locally Confident, Globally Stuck** (arXiv:2604.00375): 검색 결과로만 존재 확인.
- τ-bench pass^k 수식의 **원문 1차 확인**: PDF 텍스트 추출 실패(로컬 도구 제약). EmergentMind 인용문 + G-Pass@k 동형 수식으로 교차검증했으나, v2 동결 전 원문 §metric 1회 육안 확인 권장 (리모트에 PDF 뷰어 있음).
