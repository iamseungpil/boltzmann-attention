# Related work — scale ↔ 부하-내성 정량 (DR#2·2026-07-07·`wf_c3937c7b`)

> RELWORK_LOAD_COT(단일스텝)·RELWORK_AGENTIC_HORIZON(멀티스텝 기전)의 **정량 scale-축** 보완. 26소스·110클레임→
> 25검증(19확증·6기각). peer-reviewed 강함(RULER COLM24·Lost-in-Middle TACL24·STRING ICLR25·Zhou Nature24·METR/Sinha
> NeurIPS25). 목적=cost-optimal 레버맵(scale가 어디까지 삼)·특허/논문 "규모가 사는 것/못 사는 것" 근거.

---

## 0. 한 줄
문헌이 **scale가 신뢰성 있게 줄이는 부하 vs scale-불변/포화 부하**를 정량으로 가른다: **scale가 삼=agentic horizon
(per-step reliability 복리)·long-context 부분** / **scale가 못 삼(불변)=self-conditioning·reliability(confident-wrong)
·(우리) coverage·premature-halt**. ★**"for-all coverage/조기중단을 scale-축으로 통제측정한 연구=전무=우리 whitespace 확증.**

## 1. scale가 신뢰성 있게 사는 축

### 1.1 ★agentic horizon (가장 강한 정량 결과)
- **METR/Kwa 2503.14499(NeurIPS25)**: 50%-신뢰 과제 **시간-horizon이 ~7개월마다 2배**(2019-25·**release-date 지수·R²=0.98**·Claude-3.7≈50분).
- **Sinha 2509.09677(NeurIPS25)**: horizon이 per-step 정확도에 **hyperbolic** $H_s(p)=\lceil \ln s/\ln p\rceil$ → 작은 per-step 이득이 지수적으로 긴 과제. 큰 모델(Qwen3 4-32B·Gemma3)이 **작은 모델 single-turn 100%여도 훨씬 많은 턴 지속**.
- 기제 = **per-step reliability 복리 $p^H$**(raw reasoning 아님).
- ★**caveat(measured vs asserted)**: METR의 "per-step reliability가 주동인"은 **정성 주장**(§5·정량 분해=future work). METR fit은 **release-date**지 **param-count 아님** → 깨끗한 파라미터-스케일링 법칙 아님. $p$ vs $p^H$ 인과분해는 Sinha가 형식화하나 METR가 실측분해 안 함.

### 1.2 long-context = scale 부분적·단 광고치 훨씬 아래 포화
- **Lost-in-Middle(Liu 2307.03172·TACL24)**: 7B=recency-only·**13B/70B=U자(primacy가 scale로 출현)**·단 **U자 안 평탄해짐**(13B·70B 둘 다 U)·중간열화 "장문모델서도 지속".
- **RULER(COLM24)**: 유효길이 ≪ 광고치 — Qwen2-72B 128K주장/유효 32K·Mistral-Large-2407(123B) 4K 96.2%→128K 23.7%(-72.5pt 단조).
- **STRING(2410.18745·ICLR25)**: 유효 ≤ 학습길이 절반·**70B+(Llama-3.1·Qwen2)서도 지속**(+10pt fix 여지).
- ★caveat: 유효-shortfall *크기*가 scale-불변인지 통제 ablation 없음(지속은 보임·flat은 미측정).

## 2. ★scale가 못 사는 축 (불변/악화) — 우리 thesis 핵심

### 2.1 ★self-conditioning = 가장 깨끗한 scale-불변 (Sinha 2509.09677)
- "**scaling model size does not mitigate self-conditioning**"·200B+(Kimi-K2·DeepSeek-V3·Qwen3 4-235B)도 자기과거오류↑→열화↑·**큰 모델이 오히려 더 강하게** 자기조건화("불변"은 보수적 표현). long-context(scale 도움)와 **명확히 분리**.
- caveat: 단일 preprint(NeurIPS25 accepted)·합성 task.

### 2.2 ★reliability = scale-불변/악화 (Zhou Nature 2024·peer-reviewed)
- GPT·LLaMA·BLOOM 전 family: "**어떤 저난이도 안전영역도 확보 못 함**"·"scaling+shaping이 **회피를 confident-wrong로 교환**"(ultracrepidarianism). 실패=**제거 아니라 변형**·일부 **악화**. ★우리 coverage 직접측정 아님(난이도-일치 실패)·불변 방증.

### 2.3 long-horizon = 정성적 failure-composition shift (2604.11978·post-cutoff)
- 3100+궤적·7범주(κ=0.84): "**단순 성공률 하락 아니라 실패구성의 구조적 전환**"(planning·memory/forgetting 지배)·base-scaling만으론 불충분. ★우리 whitespace에 가장 근접하나 **scale-축 통제 안 함**. (scale-불변 강주장은 **기각 1-2**.)

## 3. ★★WHITESPACE 확증 = 우리 기여
- **"do X for all Y" coverage / premature-halt을 scale-vs-load 축으로 통제측정한 연구 = 전무.** 최근접(2604.11978)=단일·post-cutoff·해당 차원 param-ablation 없음. **⇒ 우리 tau2 forensic(14B≈32B coverage·premature-halt 동률)=진짜 측정 whitespace.** (DR#1 whitespace와 합류: iso-scaffold×cross-scale on 실제 tool-use도 미확립.)
- **미커버 RQ(증거 부재로 열림)**: distractor-vs-scale(item7·검증 소스 0)·WM-capacity-vs-scale(item3·"Unable to Forget" 여기선 미확증)·compositional-depth-vs-scale(item4·Faith-and-Fate 생존 클레임 0).

## 4. 기각(인용 금지)
METR half-life 상수-hazard(1-2) · 2604.11978 scale-불변 강주장(1-2) · METR sigmoid-plateau 반론(0-3) · 큰 positional gap(0-3) · "큰 모델이 U자 제거"(0-3) · lost-in-middle=학습혼합만(0-3).

## 5. 우리 thesis/문서 편입
- **"부하 두 원천" 문헌 확증**: scale=horizon/per-step-reliability(reducible)·**self-conditioning/reliability/coverage=invariant**. 우리 실측(coverage 17≈16·32B≈14B)과 정합.
- **★moat 근거 강화**: coverage=scale-invariant이 **학계 미측정**(whitespace) → 우리 14B/32B iso-scaffold coverage 결과=novel·측정 기여.
- **cost-optimal 레버맵**: scale는 horizon(per-step)만 삼 → 나머지(self-conditioning·coverage·compliance)는 scaffold/게이트가 담당(thinking은 self-conditioning 완화=DR#1). 특허 "규모가 사는 것 vs 못 사는 것" 정량 근거.
- 신규 cite: **2503.14499**(METR horizon)·**2509.09677**(Sinha·self-conditioning=scale-invariant·keystone)·**2307.03172**(Lost-in-Middle TACL)·**RULER**(COLM)·**2410.18745**(STRING ICLR)·**Zhou Nature 2024**·**2604.11978**(composition-shift·post-cutoff caveat).
- ★규율: post-cutoff·single-preprint(2509.09677·2604.11978) caveat·METR=release-date≠param·"measured vs asserted" 명시.
