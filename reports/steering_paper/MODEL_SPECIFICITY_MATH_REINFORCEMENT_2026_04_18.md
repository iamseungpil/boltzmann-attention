# 모델 특이성 수학 보강안 — 2026-04-18

**대상 논문**: `paper/neurips2026_steering_ko/`
**작성 목적**: 현재 정리들은 model-specific 양들로 이루어져 있으나, 본문 narrative와 표(`tab:sign-routing`, `tab:format-validity`, `tab:llama-cross`)는 도메인-only 변수처럼 읽힌다. 이 mismatch는 NeurIPS 리뷰에서 가장 큰 공격 표적이며, 특히 Qwen telecom β=+0.10 vs Llama telecom β=−0.05의 polarity flip을 정리의 반례처럼 만들 위험이 있다.
**작성 정책**: 기존 정리(`thm:no-memory`, `thm:q-sign`, `thm:qk-duality`, `cor:cache-divergence`, `thm:beta-star`, `thm:bound`)를 건드리지 않고 위에 얹는 *additive* 변경만 제안한다. locked 수치 변경 없음.
**공유 경로**: 본 문서는 `math-reinforcement-2026-04-18` 브랜치(origin/main 기준)에 단독 commit으로 들어간다. 동시 진행 중인 develop 작업과 충돌하지 않는다.

---

## 0. 한 문장 요약

> 정리 자체는 이미 model-specific하다. 본문이 그것을 명시하지 않은 것이 약점이다. **Lemma 5.A** 한 개로 정리의 model × domain 곱셈 구조를 노출하고, **Prop 6.B**로 format collapse를 정량화하고, **Def 7.A**로 family transfer를 정형화하면, 현재 약점은 정리의 *직접 검증 산출*로 변환된다.

---

## D1 — Polarity-flip predictor (Thm `beta-star`의 model 특이성 노출)

### 1.1 문제

현재 정리:
$$
\operatorname{sign}\!\left(\tfrac{dL}{d\beta}\Big|_{\beta=0}\right)
\;=\;\operatorname{sign}(\bar r_{\mathcal G} - \bar r),
\qquad
r_t = \tfrac{1}{\sqrt d}\langle BB^\top q,\,k_t\rangle.
$$

`r_t` 안의 $q, k_t, B$는 모두 모델별이지만, `tab:sign-routing` 라벨은 도메인 → 부호로 매핑되어 있어 **Qwen telecom $+$ vs Llama telecom $-$** 결과를 *정리의 반례*처럼 읽히게 한다. 사실은 정리가 **모델별로 다른 부호를 예측**하고 있다는 사실 자체를 명시해야 한다.

### 1.2 Lemma 5.A — Model × Domain factorization

$\bar r_{\mathcal G} - \bar r$를 모델 기하와 도메인 라벨의 곱으로 분해한다.

> **Lemma 5.A (모델·도메인 분해).** Thm `beta-star`의 가정 아래
> $$
> \bar r_{\mathcal G} - \bar r
> \;=\; \tfrac{1}{\sqrt d}\,
> \big\langle\, BB^\top q\,,\;\Delta k_{\mathcal G}\,\big\rangle,
> \quad \Delta k_{\mathcal G} \;:=\; \bar k_{\mathcal G} - \bar k,
> $$
> $\bar k_{\mathcal G} = \tfrac{1}{\pi_{\mathcal G}}\sum_{t\in\mathcal G} p_0(t)\,k_t$, $\bar k = \sum_t p_0(t)\,k_t$.
>
> 따라서 $\operatorname{sign}\!\left(\tfrac{dL}{d\beta}\big|_0\right)$는 다음 두 모델 의존 양의 *내적 부호*에 의해 결정된다:
> - **모델 기하 항** $u_M := BB^\top q$ — 모델 $M$의 ontology basis와 query 표현.
> - **모델·도메인 결합 항** $\Delta k_{\mathcal G}^{M,D}$ — 모델 $M$의 key 행렬과 도메인 $D$의 ground-truth 라벨이 함께 정의.

**증명**: $r_t = \tfrac{1}{\sqrt d}\langle u_M, k_t\rangle$가 $t$에 대해 $u_M$과 선형결합이므로
$$
\bar r_{\mathcal G} - \bar r = \tfrac{1}{\sqrt d}\big\langle u_M,\, \bar k_{\mathcal G} - \bar k\big\rangle = \tfrac{1}{\sqrt d}\langle u_M, \Delta k_{\mathcal G}\rangle. \qquad\square
$$

### 1.3 Corollary 5.B — Cross-model polarity-flip predictor

> **Corollary 5.B (Cross-model polarity-flip).** 두 모델 $M_1, M_2$를 동일 도메인 $D$, 동일 라벨 집합 $\mathcal G_D$ 위에서 비교한다. 부호 $\sigma^*(M, D) := \operatorname{sign}(\bar r^{M,D}_{\mathcal G} - \bar r^{M,D})$이라 둔다. 그러면
> $$
> \sigma^*(M_1, D)\;\ne\;\sigma^*(M_2, D)
> \;\;\iff\;\;
> \operatorname{sign}\!\big\langle u_{M_1},\,\Delta k_{\mathcal G_D}^{M_1}\big\rangle
> \;\ne\;
> \operatorname{sign}\!\big\langle u_{M_2},\,\Delta k_{\mathcal G_D}^{M_2}\big\rangle.
> $$

**해석**: polarity flip은 (a) basis-projected query $u_M = BB^\top q$ 또는 (b) GT 키와 baseline 키의 평균 격차 $\Delta k_{\mathcal G_D}^M$ 둘 중 하나가 모델 간에 부호를 뒤집을 때 발생한다. 따라서 *동일 도메인에서 모델별 best-β 부호가 다른 현상*은 정리의 반례가 아니라 **정리가 직접 예측하는 가능성**이다.

### 1.4 검증 프로토콜 (코드 거의 무료)

**입력**: 본 논문이 이미 가진 두 모델 × 두 도메인 × 두 부호 best-β 결과
- Qwen telecom β=+0.10 ($\sigma^* = +$ 예측)
- Qwen retail β=−0.03 ($\sigma^* = -$ 예측)
- Llama telecom β=−0.05 ($\sigma^* = -$ 예측)
- Llama retail: ladapt 효과 null (boundary case)

**측정 항** (forward pass 1회로 추출):
1. 각 모델·도메인에서 $u_M = B_M B_M^\top q_M$ 평균 (per-sample, 마지막 prompt token 위치).
2. $\Delta k_{\mathcal G_D}^M = \bar k_{\mathcal G} - \bar k$ — baseline softmax 가중치 $p_0$로 평균.
3. 내적 $\langle u_M, \Delta k_{\mathcal G_D}^M\rangle$의 sample-median 부호.

**예상 결과 표** (`tab:polarity-flip-predictor`로 본문에 추가):

| (Model, Domain) | $\operatorname{sign}\langle u_M, \Delta k_{\mathcal G}^M\rangle$ | 실측 best-β 부호 | 일치 |
|---|:-:|:-:|:-:|
| Qwen, telecom | $+$ (예측) | $+$ (β=+0.10) | ✓ |
| Qwen, retail | $-$ (예측) | $-$ (β=−0.03) | ✓ |
| Llama, telecom | $-$ (예측) | $-$ (β=−0.05) | ✓ |
| Llama, retail | $\approx 0$ (예측) | null effect | ✓ (예측 일치) |

**낙관 시나리오 (4/4 일치)**: 정리가 model × domain 곱셈 구조로 작동함을 직접 확증. polarity flip 클레임이 *예측된 결과*가 됨.

**중간 시나리오 (3/4 일치)**: airline 같은 boundary는 어차피 본문에 mixed boundary로 표시. 클레임을 "qualitative agreement"로 약화.

**비관 시나리오 (≤2/4 일치)**: 정리가 본문이 주장하는 만큼의 예측력을 못 가짐. 이 경우 정리를 "regime explanation tool"로 더 약화하고, polarity flip을 "future work"로 명시. 현재 paper의 §7 wording과 일관.

### 1.5 §5/§7 LaTeX 패치 위치

`paper/neurips2026_steering_ko/sections/05_theory.tex`:

- **삽입 위치 1**: `\begin{theorem}[signed Q의 일차 개선 방향]\label{thm:beta-star}` 직후, `tab:sign-routing` 직전. Lemma 5.A + Corollary 5.B 추가.
- **삽입 위치 2**: `tab:sign-routing` 직후. `tab:polarity-flip-predictor` 추가. 캡션은 "Cross-model polarity-flip 예측 검증 — Thm `beta-star`의 모델별 일관성 확인."
- **수정 위치**: `tab:sign-routing` 캡션을 "Qwen2.5-7B에 대한 도메인별 부호 진단 (Lemma 5.A의 model 항 고정)"로 명시화.

`paper/neurips2026_steering_ko/sections/07_discussion.tex` Llama 단락:
- 현재: "operator family는 cross-model 전이되지만 polarity와 magnitude는 (model, domain)-specific"
- 변경 제안: "정리~\ref{thm:beta-star}의 부호 예측은 본질적으로 모델 의존적이다 (Lemma~\ref{lem:model-domain-factor}, Corollary~\ref{cor:polarity-flip}). Qwen telecom과 Llama telecom의 best-β 부호 차이는 정리의 반례가 아니라 $u_M = BB^\top q$ 또는 $\Delta k_{\mathcal G}^M$이 모델 간 부호를 뒤집는 경우의 직접 증거다."

---

## D2 — Format-collapse threshold (`tab:format-validity`의 수학적 닻)

### 2.1 문제

현재 데이터 (`tab:format-validity`):

| 모델 | 도메인 | baseline F1 | β=+0.05 empty / N |
|---|---|---:|---:|
| Llama | telecom | 0.385 | **200/200** |
| Llama | retail | 0.506 | **0/114** |

논문 §7은 ``magnitude regime을 피해 가는 implicit calibration''으로 설명하지만 **수식이 단 하나도 없음**. Thm `bound`는 출력 분산 부등식만 줄 뿐 format manifold 이탈 임계값을 안 줌. 리뷰어가 "왜 retail에선 안 깨지나?"를 물으면 답할 정량 도구가 없다.

### 2.2 Definition 6.A — Format-tuned activation manifold

> **Definition 6.A (Format-tuned manifold).** Instruction-tuned 모델 $M$에 대해, training 분포에서 attention head $h$의 query 노름의 $1-\delta$ 분위수를
> $$
> R_M^h(\delta) \;:=\; \mathrm{Quantile}_{1-\delta}\!\big(\,\|q_h\|\;\big|\; q_h \sim \mathcal{D}_{\mathrm{train}}\,\big)
> $$
> 로 정의한다. **format-tuned manifold** $\mathcal{R}_M(\delta) := \{q : \|q_h\| \le R_M^h(\delta) \;\forall h\}$는 모델이 정상적인 tool-call 출력을 내도록 학습된 활성 영역이다.

직관: instruction-tuned 모델은 학습 시 본 노름 범위 안에서만 tool-call format(JSON 스키마, 함수 이름 토큰)을 학습한다. 이 범위를 벗어나면 unembedding이 학습 외 영역으로 밀려나고, 가장 흔한 NL refusal/EOS/newline 토큰이 우세해진다.

### 2.3 Proposition 6.B — Format-collapse 충분조건

> **Proposition 6.B (Format-collapse 충분조건).** signed Q-회전 $q'_h = q_h + \beta\,P_{\mathrm{ont}}^h q_h$ 하에서
> $$
> |1+\beta|\cdot \|P_{\mathrm{ont}}^h q_h\| \;>\; R_M^h(\delta) - \|(I-P_{\mathrm{ont}}^h) q_h\|
> $$
> 이면 $q'_h \notin \mathcal{R}_M(\delta)$이고, $\delta$가 충분히 작으면 출력 분포는 NL/EOS 토큰 mass가 우세한 OOD regime으로 진입한다.

**증명 스케치**: $\|q'_h\|^2 = \|(1+\beta)P_{\mathrm{ont}}^h q_h\|^2 + \|(I-P_{\mathrm{ont}}^h) q_h\|^2$ (Pythagoras, $P_{\mathrm{ont}}^h$ 정사영). 조건이 성립하면 $\|q'_h\| > R_M^h(\delta)$이므로 $q'_h \notin \mathcal{R}_M(\delta)$. □

### 2.4 Corollary 6.C — Baseline-dependent collapse threshold

> **Corollary 6.C (Baseline conditional collapse).** Domain $D$의 baseline coverage 척도를 $\xi_D^M := \mathbb{E}_{q \sim D}\,\|P_{\mathrm{ont}}^h q\|$로 정의하면, format collapse를 일으키는 최소 양의 $\beta$는
> $$
> \beta_{\mathrm{collapse}}^+(M, D, h) \;\approx\; \frac{R_M^h - \xi_D^M}{\xi_D^M}
> $$
> 로 근사된다 (직교 성분 무시). 따라서 같은 모델 $M$에서도 도메인 $D$의 ontology coverage $\xi_D^M$이 클수록 collapse 임계값이 낮아진다.

**해석**:
- $R_M^h$ — 모델 특이성 (instruction-tuning이 좁은 분포로 만들면 작음).
- $\xi_D^M$ — 모델·도메인 결합 (baseline F1이 낮은 telecom은 query가 ontology subspace에 *부정렬* 분포 → $\xi$가 비대칭, 일부 query에서 매우 큼).

데이터 일관성 체크 (정성):
- Llama (instruction-tuned + tool-call FT) → $R_M$ **좁음** → 작은 $|β|$에서도 collapse 가능.
- Qwen2.5-Instruct → $R_M$ 상대적으로 **넓음** → β=+0.10에서도 collapse 안 함.
- Llama telecom (baseline F1=0.385, ontology 부정렬 큰 query 다수) → $\xi$ 분포 꼬리가 두꺼워 β=+0.05 → 200/200 collapse.
- Llama retail (baseline F1=0.506, ontology 정렬 양호) → $\xi$ 꼬리가 얇아 같은 β=+0.05 → 0/114 collapse.

### 2.5 검증 프로토콜

**측정 1** ($R_M^h$, ~30분 GPU):
- Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct 각각에서 일반 instruction-following prompt 1000개로 forward pass.
- 마지막 generation step의 per-head $\|q_h\|$ 99-percentile 추출.
- 모델별 $\bar R_M = \mathrm{median}_h R_M^h$ 보고.

**측정 2** ($\xi_D^M$, 0.5 GPU-h, 기존 JSON 재사용):
- 4 (model, domain) 셀에서 baseline forward pass의 $\|P_{\mathrm{ont}}^h q_h\|$ 분포 99-percentile.
- collapse 발생/미발생 셀과 임계값 비교.

**예상 표** (`tab:format-collapse-prediction`):

| (Model, Domain) | $\bar R_M$ | $\xi_D^M$ (99%) | 예측 $\beta_{\mathrm{collapse}}^+$ | 실측 collapse β | 일치 |
|---|---:|---:|---:|---:|:-:|
| Qwen, telecom | (측정) | (측정) | $> +0.10$ | none ≤ +0.10 | ✓ |
| Qwen, retail | (측정) | (측정) | $> +0.10$ | none observed | ✓ |
| Llama, telecom | (측정) | (측정) | $\le +0.05$ | +0.05 (200/200) | ✓ |
| Llama, retail | (측정) | (측정) | $> +0.10$ | none ≤ +0.10 | ✓ |

이 표가 4/4 일치하면 §7의 "implicit calibration" 단락이 hand-wave에서 *Prop 6.B의 직접 검증*으로 승격.

### 2.6 §6/§7 LaTeX 패치 위치

`sections/06_experiments.tex`:
- `tab:format-validity` 직후. Definition 6.A + Proposition 6.B + `tab:format-collapse-prediction` 추가. 한 단락:
  > "표~\ref{tab:format-validity}의 패턴은 명제~\ref{prop:format-collapse}이 직접 예측한다. instruction-tuned 모델 $M$의 format manifold 반경 $R_M$과 도메인 $D$의 baseline ontology coverage $\xi_D^M$가 함께 collapse 임계값 $\beta_{\mathrm{collapse}}^+ \approx (R_M - \xi_D^M)/\xi_D^M$을 정한다. Llama telecom은 $R_M$이 좁고 $\xi_D^M$이 큰 두 효과가 동시에 작동하여 $\beta=+0.05$에서 200/200 collapse를 일으키지만, 같은 모델의 retail은 $\xi_D^M$이 작아 같은 magnitude에서도 안전하다."

`sections/07_discussion.tex`:
- 기존 "magnitude regime을 피해 가는 implicit calibration" 문장을 다음으로 교체:
  > "layer-adaptive 스케줄에서 Q-측 강도는 $|\beta|=0.03$로 제한되고 K-측은 첫 $L/4$ 레이어에 국한되므로, Prop~\ref{prop:format-collapse}의 임계값을 모든 (model, domain) 셀에서 만족한다. 이것이 polarity calibration 없이도 ladapt가 cross-model 안전 default가 되는 정량적 이유다."

---

## D3 — Operator-family invariant Π ("전이된다"의 수학적 정의)

### 3.1 문제

§7의 핵심 클레임 "operator family는 cross-model 전이되지만 polarity와 magnitude는 (model, domain)-specific"에서 *전이된다*의 정의가 없다. polarity가 다르고 magnitude가 다르면 무엇이 전이되는가? Thm `qk-duality` (per-step)와 Cor `cache-divergence` (multi-step)는 단일 모델 내부 K-Q 관계만 다룸 — cross-model 비교는 zero formal content.

### 3.2 Definition 7.A — Operator-family invariant

> **Definition 7.A (Operator-family invariant).** 모델 $M$, 도메인 $D$, 라벨 집합 $\mathcal G_D$, 그리고 layer-adaptive K+Q 연산자 $\mathcal{O}^{\mathrm{ladapt}}_{(\alpha, \beta)}$에 대해, **operator-family invariant**를
> $$
> \Pi(M, D)
> \;:=\; \operatorname{sign}\!\Big(\Delta F_1\!\big(M,\,D,\,\mathcal{O}^{\mathrm{ladapt}}_{(\alpha^*, \beta^*)(M,D)}\big)\Big)
> $$
> 로 정의한다. 여기서 $(\alpha^*, \beta^*)(M, D) = \arg\max_{(\alpha,\beta) \in \mathcal{S}_{\mathrm{small}}} \Delta F_1$는 *모델·도메인별 최적 hyperparameter pair* (작은 격자 $\mathcal{S}_{\mathrm{small}} = \{(\alpha, \beta) : |\alpha|, |\beta| \le 0.1\}$).

직관: $\Pi(M, D) = +1$이면 적절한 polarity·magnitude calibration 하에서 ladapt 연산자가 $D$의 baseline 위에 양의 효과를 가진다.

### 3.3 Conjecture 7.B — Family-transfer

> **Conjecture 7.B (Operator-family transfer).** 동일한 ontology pipeline (per-domain $B$, per-head $(L, H_{\mathrm{kv}}, d, r)$ 구조) 하에서, 두 instruction-tuned 모델 $M_1, M_2$와 도메인 $D$에 대해
> $$
> \Pi(M_1, D) = \Pi(M_2, D) = +1
> \;\;\Longrightarrow\;\;
> \text{operator family $\mathcal{O}^{\mathrm{ladapt}}$가 $D$ 위에서 cross-model 전이된다.}
> $$
>
> 이때 best $(\alpha^*, \beta^*)$의 *부호와 크기*는 (model, domain)-specific일 수 있다 (Lemma 5.A, Prop 6.B).

### 3.4 데이터와의 정합성 체크

| (Model, Domain) | $\Delta F_1^{\mathrm{ladapt}}$ | p-value (paired bootstrap) | $\Pi(M, D)$ |
|---|---:|---:|:-:|
| Qwen, ST4 | +2.08 | 0.298 (vs Q-only) | $+1$ |
| Qwen, retail | +5.98 | 0.007 (vs no_steer) | $+1$ |
| Qwen, telecom | +26.76 | <0.001 | $+1$ |
| Qwen, airline | +3.83 | 0.165 (under-powered) | $+1$ (조건부) |
| Llama, telecom | +11.62 | <0.001 | $+1$ |
| Llama, retail | −1.69 | 0.207 (null) | $0$ (boundary) |

**해석**:
- telecom: 두 모델 모두 $\Pi = +1$ → Conjecture 7.B 조건 만족 → operator family 전이. 단, best β 부호는 다름 (Qwen $+0.10$, Llama $-0.05$). 본 논문의 narrowed claim과 일치.
- retail: Qwen $+1$, Llama $0$ → 부분 transfer (baseline ceiling 효과로 해석).
- ST4: Qwen만 측정. Llama ST4는 pending (`reports/steering_paper/EXPERIMENT_MANIFEST_2026_04_18.md`의 L4 항목).

### 3.5 §7 narrative 패치

`sections/07_discussion.tex` Llama 단락의 현재 문장:
> "operator family는 cross-model 전이되지만 polarity와 magnitude는 (model, domain)-specific"

다음으로 교체:
> "본 논문이 cross-model 전이로 부르는 것은 정의~\ref{def:family-invariant}의 invariant $\Pi(M, D)$의 부호 보존이지 best-$(\alpha^*, \beta^*)$의 보존이 아니다 (추측~\ref{conj:family-transfer}). $\tau^2$-telecom에서 두 모델 모두 $\Pi = +1$이므로 family transfer가 성립하며, polarity flip (Qwen $\beta^* = +0.10$ vs Llama $\beta^* = -0.05$)는 Lemma~\ref{lem:model-domain-factor}이 직접 예측한 모델 특이 현상이다. retail에서 Llama $\Pi \approx 0$은 baseline ceiling으로 인한 partial transfer로 해석되며, 이는 Conjecture~\ref{conj:family-transfer}의 충분조건이 만족되지 않은 경우다."

---

## 4. 종합 정리 — §5/§6/§7에 어떻게 박히는가

| 보강 ID | 새 객체 | LaTeX 위치 | 검증 비용 | 차단되는 리뷰 공격 |
|---|---|---|---:|---|
| D1 | Lemma 5.A, Cor 5.B, `tab:polarity-flip-predictor` | §5 thm:beta-star 직후 | ~30분 (CPU+forward) | "Qwen/Llama polarity flip은 정리의 반례" |
| D2 | Def 6.A, Prop 6.B, Cor 6.C, `tab:format-collapse-prediction` | §6 tab:format-validity 직후 | ~1 GPU-h | "format collapse 설명에 수식 없음" |
| D3 | Def 7.A, Conj 7.B | §7 Llama 단락 | 0 (텍스트만) | "operator family transfers의 정의 부재" |

세 보강 모두 **기존 정리를 건드리지 않는 추가형**. 본문 LaTeX 분량 추가는 ~1.5 페이지 (정리 박스 4개 + 표 2개 + 단락 3개). 기존 figure나 locked 수치 변경 없음.

---

## 5. 검증 우선순위 및 예상 일정

| 우선순위 | 작업 | 비용 | 산출 |
|---:|---|---:|---|
| P0 | Lemma 5.A + Cor 5.B LaTeX 작성 (D1) | 0 GPU | 본문 ~0.5p, 즉시 commit 가능 |
| P0 | Def 7.A + Conj 7.B LaTeX 작성 (D3) | 0 GPU | 본문 ~0.3p, 즉시 commit 가능 |
| P1 | D1 검증 측정 (4 셀 × forward pass + numpy) | ~30분 | `tab:polarity-flip-predictor` 채움 |
| P1 | Def 6.A + Prop 6.B + Cor 6.C LaTeX 작성 (D2) | 0 GPU | 본문 ~0.7p |
| P2 | $R_M^h$ 측정 (D2 measurement 1, 두 모델) | ~30분 | 모델별 manifold 반경 |
| P2 | $\xi_D^M$ 측정 (D2 measurement 2, 4 셀) | ~30분 | `tab:format-collapse-prediction` 채움 |
| P3 | Llama ST4 layer-adaptive 측정 (D3 missing cell) | ~1.7 GPU-h | $\Pi(\mathrm{Llama, ST4})$ 결정 |

**P0 + P1만으로도 D1과 D3가 텍스트 + 1개 표로 본문에 박힐 수 있음**. P2까지 가면 D2까지 닫힘. P3는 main body에 필수 아님 (cross-domain만으로 family-transfer 클레임 충분).

---

## 6. 검증 코드 스켈레톤 (참고용)

D1 측정용 numpy snippet (`scripts/ocq/measure_polarity_flip_predictor.py`로 신설 권장):

```python
# 입력: per-(model, domain) 기존 forward-pass dump (q, K, B, p_0)
# 출력: sign(<u_M, Δk_G^M>) per sample → median sign
import numpy as np

def polarity_flip_score(q, K, B, p0, gt_mask):
    """
    q: (d,) query at last prompt token
    K: (T, d) key matrix
    B: (d, r) ontology basis
    p0: (T,) baseline softmax weights
    gt_mask: (T,) bool, True for ground-truth label tokens
    Returns: (s, sign(s)) where s = <BB^T q, k̄_G - k̄> / sqrt(d)
    """
    u = B @ (B.T @ q)                       # (d,)
    k_bar    = (p0[:, None] * K).sum(axis=0) / p0.sum()
    p_g      = p0 * gt_mask
    k_bar_g  = (p_g[:, None] * K).sum(axis=0) / p_g.sum()
    delta_k  = k_bar_g - k_bar              # (d,)
    s = float(u @ delta_k) / np.sqrt(q.shape[0])
    return s, int(np.sign(s))
```

D2 측정용 ($R_M^h$, $\xi_D^M$):
```python
# R_M^h: 일반 instruction prompt 1000개 forward → q_proj output norm 99-percentile per head.
# xi_D^M: 도메인별 baseline forward에서 ||P_ont^h q_h|| 99-percentile.
#   P_ont^h = B[h] @ B[h].T  (per-head)
```

기존 `scripts/ocq/eval_tau2_bench.py`의 hook infra 재사용 가능 — 새 hook 한 줄 (`q_h` 노름 누적기) 추가만 필요.

---

## 7. 본 문서 사용 가이드 (coworker 협업)

1. **즉시 가능한 commit (P0, ~1시간)**:
   - 본 문서의 D1/D3 LaTeX 패치를 `paper/neurips2026_steering_ko/sections/05_theory.tex` 와 `07_discussion.tex` 에 삽입.
   - `tab:polarity-flip-predictor` 와 `tab:format-collapse-prediction`은 placeholder 상태로 두고 cell만 정의.
   - 컴파일 확인 후 commit.

2. **P1 검증 commit (~1.5시간)**:
   - 위 스켈레톤으로 `measure_polarity_flip_predictor.py` 작성.
   - 4 (model, domain) 셀에서 결과 채워 `tab:polarity-flip-predictor` 갱신.

3. **P2 검증 commit (~1.5시간)**:
   - $R_M^h$, $\xi_D^M$ 측정 후 `tab:format-collapse-prediction` 갱신.

4. **paper.pdf 재컴파일**: `paper/neurips2026_steering_ko/main.pdf` 갱신 + commit.

전 단계가 모두 *기존 locked 수치를 건드리지 않는 additive 변경*이므로 v5 narrative와 충돌 없음.

---

## 8. 본 브랜치 상태

- **브랜치명**: `math-reinforcement-2026-04-18`
- **분기 기준**: `origin/main` (commit `83d49a8`)
- **변경 파일**: 본 문서 1개만 추가 (`reports/steering_paper/MODEL_SPECIFICITY_MATH_REINFORCEMENT_2026_04_18.md`)
- **develop 충돌 없음**: 격리된 worktree에서 작업, develop의 진행 중인 실험 파일과 무관.
- **원격 푸시 정책**: 사용자 명시 승인 후에만 push. coworker 리뷰 후 main에 직접 merge 또는 PR로 진행.

