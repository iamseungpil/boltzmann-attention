# 선점 스캔(원문 정독) — correlated-error 근본원인 해결이 선점됐나 (2026-07-15)

> 정본. novelty 방어. DR `wf_b34e89f3-aad` stall(12h·deep-read서 hang) → **근접 이웃 4편 PDF 원문 직접 정독**(WebFetch 소형모델 아님·[[40]] 신뢰도 [M]).
> 렌즈(사용자): 이 논문들이 **correlated의 근본원인을 찾아 해결**하나, 아니면 "voting 안 된다" 관찰·부품 패치인가.
> 우리 주장(C88/C89): correlated는 하나 아님 → 근본원인별(statistical/decidable-systematic/plan/non-decidable) 진단 후 매칭 레버 라우팅. **핵심 = 정답이 샘플분포에 *없을 때*(gold∈support 0) aggregation을 *떠나* 원인별로 생성**.

## 결론(원문 정독)
**4편 모두 우리 core를 선점하지 않음.** 셋(VeriPlan·GoV·Info-Gain)=부품(verify/ASK) 또는 무관. **Minority Sentinel만 correlated 근본원인을 인지하나, 해법이 aggregation 안에 머물러 우리와 disjoint regime.**

## 1. Minority Sentinel (`2606.29270`·SIGIR'26) — ★가장 근접·그러나 disjoint
- **근본원인 인지 = 우리와 동일**: Condorcet 독립가정 위반·"LLMs share pretraining corpora → errors strongly correlated"·"Tyranny of the Majority"(Estornell&Liu 형식증명·correlated서 MV가 오답 lock-in).
- **해법 = counting 개선(aggregation 내부)**: 3-agent debate서 2:1 divergence(39.1%) 중 **minority가 정답인 25.5%**를 LightGBM debate-fingerprint로 감지·flip. 자기표현: *"the problem is not 'how to argue' but **how to count**"*·"safety valve at the voting aggregation layer".
- **★결정적 disjoint(우리 차별의 핵심)**: recovery 상한 = **정답이 minority로 *존재*할 때**(divergent의 25.5%·oracle margin 10pp). **우리 regime = 정답이 어느 샘플에도 *없음*(E-REGIME ⋈ gold∈support 0/29)** → recover할 minority 부재 → Minority Sentinel **원리적 무력**. 즉 그들=분포 안에 정답 있음(counting 문제)·우리=분포에 정답 없음(computation/plan/information 문제). **두 regime 상호배타.**
- 판정: correlated 근본원인 *인지*는 공유(인용). 그러나 "aggregation을 떠나 원인별 생성"은 **미선점**.

## 2. Info-Gain Clarification (`2606.03135`·ICML계열) — ASK 부품
- RL로 **info-gain reward**(EIG=H(G*|x)−H(G*|x,Q,A)·Bayesian belief update)로 **when-to-ask 정책 학습**(retail 모호성).
- 원문 grep: **floor/lower-bound/H_min/decidable/verify = 전무**·correlated 무관.
- 판정: "info-gain으로 언제 물을까"=ASK측 부품 선점(인용). **H_min을 *floor*(질문수 하한)로·DERIVE/DEFAULT로 접기·verify-or-ASK router 안에 배치 = 미선점**(그들=when-to-ask·우리=how-few=하한+router).

## 3. VeriPlan (`2502.17898`) — verify 부품(whole-plan)
- **전체 plan**에 결정론 formal model-checking(PRISM)·user rule는 **upfront** 정의(undecidable sub-step 트리거 아님)·voting 없음·correlated 무관.
- 판정: 결정론 verify 존재 증명(인용)·**per-step decidability 라우팅·ASK escalation = 미선점**.

## 4. Graph-of-Verification (`2506.12509`) — verify 부품(구조화)
- 추론을 DAG로 분해·구조적 검증(결정론+self-verify 혼합). per-step {voting|verify|ask} 라우팅·decidability 키·correlated 대응 = 없음.
- 판정: 검증 구조화 선행·라우터 미선점.

## 5. 종합 판정 (원문 [M])
- **부품은 개별 선점**(verify: VeriPlan/GoV/PAL · ASK/VOI: Info-Gain · voting-override: Minority Sentinel) → 인용·양보.
- **우리 core = 미선점**: ①정답-부재(gold∈support 0) regime에서 **aggregation을 떠나** ②correlated를 근본원인별 하위유형으로 진단(statistical/decidable-systematic/plan/non-decidable) ③각각 매칭 레버(voting/verify/E-PLAN/ASK)로 라우팅 ④H_min-floor로 ASK 예산 접기. **어느 논문도 안 함.**
- **★가장 날카로운 차별(Minority Sentinel 대비)**: 그들 상한=정답이 minority로 present(counting)·우리 대상=정답 absent(생성 필요). **disjoint regime** = 정면충돌 아님·상보.

## 6. 인용해야 할 특성화 선행 (양보)
- correlated errors가 voting 무력화: `2606.29270`(Tyranny of Majority·Estornell&Liu 형식증명)·`2411.01101`(Self-Consistency Falls Short·positional bias)·`2605.29800`(Nine Judges·correlated undermine eval).
- (DR harvested claim [?]·미검증) "external verification, not consensus, enables added samples to help"(verified vs unverified domain) — 우리 verify>voting 지지·arxiv id 미확정·재검색 필요.

## 7. caveat
- 4편=PDF 원문 정독 [M]. DR harvested 15 claim=deep-read [?](verify 미도달·재검증). 특성화 papers 일부 arxiv id 미확정.
- Minority Sentinel "gold∈support 0서 무력"은 *논리적 귀결*(그들 recovery 정의상)·그들이 그 regime을 명시 테스트한 건 아님(우리 E-REGIME이 그 regime 계측).
