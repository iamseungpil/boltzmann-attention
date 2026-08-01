# 선행연구 정본 — 자유 대화 → 유한 enum/operand 매핑에서 유사어·경계는 스케일로 닫히는가 (2026-08-01)

> **상태 = 딥리서치 검증 완료 · 인용 가능.** 사용자 질문(2026-07-31): *"LLM은 무한계 자유 대화를
> 검증기/실행기가 쓸 유한계 enum·도구 operand로 매핑한다. 이때 유사어·경계는 스케일로만 해결될
> 수 있나 — 선행연구가 있나?"*
> **방법** = 병렬 딥리서치 5클러스터(①유사어·조합성 ②모호성·인간 불일치 ③OOS·도구선택
> ④비-스케일 레버 ⑤화이트스페이스 적대 검증), 전 항목 arXiv/ACL Anthology/OpenReview/BFCL
> 라이브 CSV 원문 대조(2026-08-01 기준). 수치는 원문 표에서 직접 추출한 것만 [S], 2차 출처·요약
> 유래는 [M], 검증 실패는 ⚠️로 명기.
> **연결** = [[41]](rival·whitespace) · [[42]](prompt-limits) · [[43]](환각 유인) · [[44]](SOAR) ·
> [[45]](load 이론) · [[46]](Paper1 노벨티) · [[22]](닫힌/열린 술어) · task_023(C251 ⑤).

---

## §0. 한 문단 (질문의 답)

**축을 둘로 갈라야 답이 된다.** (a) **표현 변이 축**(유사어·paraphrase — 정답 매핑은 결정돼 있고
표면만 변함)은 스케일이 **크게 먹되 완전히 닫지 못한다** — 2021년 20~30pp였던 유사어 갭이
2025년 LLM 시스템에서도 6~11pp 잔존하고, 가장 깨끗한 스케일-sweep(T5 60M→11B·PaLM 8B→540B)
에서 fine-tuning 곡선은 조합 split에서 **flat/음수**다. (b) **경계 축**(어느 enum에도 안 속하거나
후보 둘 다 명분이 있는 미결정 사례)은 **동일-계열 sweep에서 반복적으로 scale-flat**이고
(BFCL Qwen3/Llama·When2Call Qwen 7B→72B F1 32.0→32.8·AbstentionBench 20모델·WSD hardEN),
역상관 사례까지 있다(ToolSandbox·reasoning-RL −24%). 경계를 실제로 닫는 실증 수단은 스케일이
아니라 **비-스케일 레버**다: 되묻기(+8~20pp·모델 고정), 경계-타깃 훈련(When2Call-RPO +65%·
DiaFORGE 27B가 GPT-4o를 +27pp 능가), 유인 교정(Kalai 하한은 모델 크기와 무관). 그리고 문헌의
가장 정밀한 단서는 **enum 제약 그 자체가 병목**이라는 것 — 같은 모델이 유한 inventory 강제
매핑에선 82%인데 자유 서술로 풀면 98%다(WSD). 단 "스케일 무용"은 과장이다 — 경계 축도 계열
의존적으로 느리게 오르는 경우가 있고(Gemma-3), frontier의 irrelevance는 2024→2026에 61→85로
실제 개선됐다. **방어 가능한 주장 강도 = "경계 축은 in-inventory 정확도보다 훨씬 느리고
불균일하게 오르며, 닫는 실증 수단은 스케일이 아니다."**

---

## §1. 질문의 정식화 — 두 축 분해

무한계 발화 → 유한 enum/operand 매핑의 실패는 성질이 다른 두 성분이다.

| 축 | 정의 | 오류 성질 | 우리 대응물 |
|---|---|---|---|
| **(a) 표현 변이** | 정답 매핑은 결정 — 입력 표면(유사어·paraphrase·조합)만 변함 | epistemic — 원리상 데이터/스케일이 감소시킴 | F3 표면 축 (τ² 지평에선 scale이 흡수 — [47]) |
| **(b) 경계/미결정** | 정답 매핑 자체가 미결정 — annotator 불일치·복수 후보 모두 명분·OOS | task-inherent/irreducible — 어떤 스케일도 원리상 못 닫음 | [[22]] 열린 술어 · task_023(C251 ⑤: KB가 오도구를 명시 승인) |

⚠️**용어 규율**: (b)를 "aleatoric"으로 부르는 것은 Baan et al. (arXiv:2307.15703)이 NLG에서
비판한 이분법이다 — 논문에서는 **"irreducible/task-inherent (aleatoric-류)"**로 병기할 것. [S]

---

## §2. (a)축 — 유사어·조합성: 스케일이 크게 먹되 잔여가 측정됨

### 확정 증거 (전부 원문 수치 [S])

| 논문 | venue | 핵심 수치 | 스케일 판정 |
|---|---|---|---|
| **Spider-Syn** (Gan et al., arXiv:2106.01065) | ACL 2021 | 유사어 치환 시 exact match −21~−29pp (RAT-SQL+BERT 69.7→48.2). 방어 기법도 부분 회복만 | 사전학습 강화가 갭을 줄이되 못 없앰 |
| **Dr.Spider** (Chang et al., arXiv:2301.08881) | ICLR 2023 | 17 perturbation. 최강 모델도 전체 −14.0pp·최난(DBcontent-equivalence) −50.7pp. **Codex는 스키마 측엔 강하고 NL paraphrase 측엔 fine-tuned보다 약함** | **비대칭** — 스케일이 한 강건성 축을 사며 다른 축을 팖 |
| **CFQ** (Keysers et al., arXiv:1912.09713) | ICLR 2020 | random split >95% vs MCD split <20% (전 아키텍처) | — |
| **Furrer et al.** (arXiv:2007.08970 ⚠️arXiv-only) | preprint | **T5 5단 sweep**: MCD-mean 28.0→31.2→34.8→40.2→40.9% — 200× 파라미터에 +13pp·포화 곡선. SCAN은 비단조 | **No** — 감쇠 수익 |
| **Qiu et al.** (arXiv:2205.12253) — ★최강 직접 증거 | EMNLP 2022 | **T5 60M→11B + PaLM 8B→540B 동일-과제 sweep**: fine-tuning은 조합 split에서 **flat/음수**(GeoQuery template: T5-small 85.8 → T5-11B 86.1). ICL은 양수(PaLM 8B 25.9→540B 74.1)나 소형 fine-tuned에 미달 | **No**(FT) / **Partially**(ICL) |
| **SLOG** (Li et al., arXiv:2310.15040) | EMNLP 2023 | 구조적 일반화: 사전학습 T5 40.6% ≈ LLaMA-7B 40.1% ≈ scratch Transformer 27.1% ≪ 구조-인지 파서 70.8% | **No** — 사전학습이 구조 축에 무력 |
| **Solid-SQL** (COLING 2025) · **ROUTE** (AAAI 2025) | — | 2025년 LLM 시스템에서도 Spider-Syn 갭 **6~11pp 잔존** (MAC-SQL 76.3→65.1) | 줄지만 0이 안 됨 |

### 의무 인용 nuance — Drozdov et al. (ICLR 2023, arXiv:2209.15003)

code-davinci-002 + **dynamic least-to-most 분해 scaffold**로 CFQ MCD 평균 **95.0%**·COGS 99.2%
도달 — 문헌에서 "갭이 닫힌" 유일급 사례. **그러나 같은 모델 vanilla few-shot은 80.8%**
(CoT 87.2%) — 닫은 것은 스케일이 아니라 **분해 스캐폴드(method 레버)**다. ⇒ 반례가 아니라
우리 명제(비-스케일 레버가 닫는다)의 지지 사례로 인용. [S]

**(a)축 소결**: 스케일-반응적이되 **완전 폐쇄 사례는 스케일 단독으로는 미발견**. 잔여는
method(분해)·augmentation(재표현 생성)이 닫는다.

---

## §3. (b)축 — 경계/미결정: scale-flat 증거 스택

### 3-1. 인간 불일치·모호성 (gold 자체가 미결정)

| 논문 | venue | 핵심 수치 [S] |
|---|---|---|
| **ChaosNLI** (Nie et al., arXiv:2010.03532) | EMNLP 2020 | 100 annotator/item. **MNLI의 31.8%에서 다수결이 원 gold를 뒤집음**. 저일치 사례에서 모델 정확도 ~0.5(random). 모델-인간 분포 JSD 0.22~0.31 vs 인간 baseline 0.042~0.070 (3~7배) |
| LLM 후속 (Lee et al., arXiv:2305.13788) | EMNLP 2023 | LLM도 불일치 분포 포착 실패·저일치 급락 재현 |
| **AMBIENT** (Liu et al., arXiv:2304.14399) | EMNLP 2023 | GPT-4 disambiguation 인간평가 정답 **32%** vs 데이터셋 기준 90%. T/F 인식 63%(random 50%). **scale trend 비단조가 논문 명시** |
| **AmbigQA** (Min et al., arXiv:2004.10645) | EMNLP 2020 | 자연 발생 질문(NQ-open)의 **>50%가 구조적으로 모호** — 모호성은 입력의 속성 |
| **Plank** (arXiv:2211.02570) | EMNLP 2022 | HLV = noise가 아니라 signal. 복수 타당 답 = 본 질문의 "미결정 gold" 정본 프레임 |
| **BoN 무효** (Ruiz et al., arXiv:2510.12516) | LeWiDi-2025 | verifiable 정답 없는 disagreement 과제에서 **test-time scaling(Best-of-N)도 무효** |

### 3-2. OOS·도구 선택 경계 (우리 도메인과 동형)

| 증거 | 핵심 수치 [S] |
|---|---|
| **CLINC150** (Larson et al., EMNLP 2019) + LLM 후속 (Wang et al., LREC-COLING 2024) | BERT: in-scope 96.9 vs OOS recall 40.3 (−56.6pp). LLM 시대에도 비대칭 유지: ChatGPT **in-scope 격차 13pp vs OOS 격차 56pp** (vs fine-tuned UniNL) |
| **BFCL 라이브 CSV** (2026-08-01 직접 fetch) — ★동일-계열 sweep | **Qwen3 0.6B→32B: Overall +24.8pp / Irrelevance 80.8→76.4 flat** · **Llama-3.x 1B→70B: Overall +21.1pp / Irrelevance ~52 flat** · xLAM-2 부분 상승 · **Gemma-3 1b→27b: 33→74 상승(반례)** |
| **When2Call** (Ross et al., NAACL 2025) | **Qwen2.5 7B→72B: F1 32.0→32.8 — 10배 스케일에 flat**. Llama 8B→70B 16.6→37.8(느린 상승). GPT-4o 61.3 — frontier도 천장에서 멂. 훈련(RPO)으로는 8B 31.9→52.4 (+65%). 축자: "does not necessarily improve with model size" |
| **ToolSandbox** (Apple, arXiv:2408.04682) | **역상관** — Insufficient-Information 시나리오에서 강한 모델일수록 도구·인자를 더 지어냄 |
| **API-Bank** (EMNLP 2023) | GPT-4 오류의 **67.9% = 오도구 검색/선택** — 인자 오류(7.1%)의 ~10배 |
| **AbstentionBench** (Meta FAIR, arXiv:2506.09038) | 20 frontier × 20 데이터셋: abstention은 **scale-flat**("scaling models is of little use") + **reasoning-RL이 −24% 악화** |
| **QuestBench** (GDM, NeurIPS 2025 D&B) | **완전-명세 버전은 풀면서 "무엇을 물을지"는 못 찾음** (frontier 40~50%) — 풀이 능력과 경계 식별의 해리를 격리 실증 |
| **τ²-bench no-user ablation** (arXiv:2506.07982) | 사용자 제거 시 GPT-4.1 +18pp·4o-mini +25pp — 실패 큰 몫이 상호작용/모호성 축 |
| **WSD hardEN** (Meconi et al., arXiv:2509.13905) | 진짜 fine-grained 경계 사례: GPT-4o **45.6%**·대부분 LLM ~40% — **경계만 scale-flat** |

### 3-3. 반대 방향 증거 (정직 기록 — 주장 강도를 여기에 맞춘다)

1. **frontier irrelevance는 실제 개선됨**: 2024 GPT-4 61.4% → 2026 frontier 77~87%. 단
   near-perfect 아니고, **소형 특화가 frontier를 능가**(Hammer2.1-**7b** 90.1 · GPT-5-mini 91.0
   vs Claude Opus 4.5 84.7) — 닫는 것은 훈련 타깃팅이라는 해석이 데이터와 정합.
2. **Gemma-3 계열은 경계 축도 스케일 상승**(33→74) — "전 계열 flat"이 아니라 **불균일**.
3. **WSD determined 영역은 스케일이 닫음**: Llama 1B 56% → GPT-4o 82.3% — 옛 시스템 천장
   (73~77%) 돌파·supervised SOTA와 무차이. ⚠️**"IAA≈75-80%에서 포화·LLM도 못 넘음"의 소박한
   버전은 이제 반박된다** — 반드시 "경계 잔여만 flat"으로 정밀화해서 인용.
4. **분포 학습 가능성**: soft-label fine-tuning이 불확실성 **분포 추정**을 개선(Uma et al.
   JAIR 2021 · Inoshita 2026: JSD 0.45→0.30) — "되묻기/기권만이 해법"은 과장. 제3의 레버로 병기.
5. BFCL Relevance Detection은 표본 ~16개 추정(점수가 전부 6.25% 배수)·Ministral-8B가
   irrelevance 100%/relevance 0%(항상-거부 퇴화 정책) — **단일 지표 해석 금지**.

---

## §4. ★가장 정밀한 단서 — enum 제약 그 자체가 병목

Meconi et al. (arXiv:2509.13905·Navigli AAAI 2026): 같은 모델이 —

- **유한 sense inventory로 강제 매핑**: 82.3% (expert 인간 91.25%와 격차 잔존)
- **자유 서술로 의미 설명**: 최대 **98%**

⇒ 결손은 "의미를 모름"이 아니라 **"아는 의미를 유한 label 공간에 정렬하는 행위"**에 있다.
이것이 사용자 질문("무한계→유한계 매핑")의 가장 직접적인 문헌 답이며, [[22]](닫힌 술어만
scaffold·열린 술어는 LLM+표면화)의 독립 지지 증거다. 우리 어휘로: **formalize 부하는 모델
내부 표상이 아니라 인터페이스에서 발생**한다.

---

## §5. 비-스케일 레버 — 문헌 실증 수치

| 레버 | 대표 실증 [S] | 효과 | 우리 대응 |
|---|---|---|---|
| **되묻기(clarification)** | CLAM (arXiv:2212.07769 ⚠️preprint): +20pp adjusted acc·모델 고정 / ClarifyGPT (FSE 2024): GPT-4 Pass@1 +9.8pp / Tell Me More·IN3 (ACL 2024): 불필요 subtask 22.2→1.9% | +8~20pp를 **모델 고정 하에** | ASK ([[16]]) |
| **경계-타깃 훈련** | When2Call-RPO: 8B F1 +65% / DiaFORGE (SAP): 특화 Gemma-3-**27B** 89% vs GPT-4o 62% / **Mistral-Interact 7B가 GPT-4를 vagueness 판정에서 이김**(85.2 vs 82.4) | 소형 특화 > frontier | 학습 두 날개 ([00]) |
| **분해 스캐폴드** | Drozdov (ICLR 2023): vanilla 80.8 → least-to-most 94.4 (동일 모델) | +14pp | scaffold 결정론 분담 ([[10]]) |
| **기권/선택적 예측** | Kamath (ACL 2020): 80% 정확도 조건 coverage +8pp — 단 **트레이드**(능력을 만들지 않음) | risk-coverage 최적화 | INFER-calibration ([16] 유일 잔여) |
| **분포 학습** | soft-label FT: JSD 0.45→0.30 (Inoshita 2026) | 분포 추정만 개선 | — (제3 레버·신규 등재) |
| **유인 교정** | Kalai et al. (arXiv:2509.04664): **singleton-rate 하한 err ≥ sr − …은 모델 크기와 무관**. 주류 벤치 대다수가 IDK에 0점 → 추측이 기권을 지배 | 채점 수정 없인 어느 스케일도 기권 안 함 | [[43]] 형식 보강 |
| **impasse→구조 전환** | SOAR (Laird·Newell·Rosenbloom, AIJ 1987) 정본 확인 | — | escape/ASK 분해 ([[44]]) |

**calibration-행동 해리 (핵심 정리)**: calibration **신호**는 스케일과 동행하고(Kadavath
arXiv:2207.05221 — 큰 모델일수록 P(True) 개선), abstention **행동**은 flat이며 reasoning-RL은
역행(AbstentionBench). **병목은 앎이 아니라 행동 정책과 유인** — [45] load 이론과 정확히 정합.

---

## §6. 화이트스페이스 재판정 ([[46]] 갱신 입력)

**판정 = PARTIALLY TRUE (진성이되 좁혀 주장).** "동일 task·동일 harness·**동일-family** size
사다리에서 대화→유한 inventory 매핑 부하를 격리하고 (a)표면변이 / (b)경계 오류를 **분리
보고**"한 발표 문헌은 적대 수색에도 **미발견**. 단 부품은 전부 선점 — novelty는 conjunction에만:

| foil (명시 인용 의무) | 가진 것 | 빠진 것 |
|---|---|---|
| Rabinovich & Anaby-Tavor (IBM, arXiv:2504.00914) | PARA/SYNO vs toolkit-확장 분리 측정 | size 사다리 없음·(b)가 미결정 아닌 distractor-혼동 |
| DiaFORGE (arXiv:2507.03336) | 3B~70B 6단 + abstention/false-positive 분리 | **이종 family**(교란)·synonym 축 없음 |
| Skills scaling law (arXiv:2605.16508) | inventory-routing 오류 분해·log-감쇠 법칙 | sweep 축 = library size ≠ model size |
| T-Eval (ACL 2024) | Qwen 7B→72B 동일-family 사다리 | 분해 축이 파이프라인 단계 |
| BFCL·When2Call | 동일-계열 flat 실측 | 매핑-부하 격리 아님·(a)/(b) 분리 안 함 |

**문구 규율**: "no published work does X" 금지 → "**no work combines (i) same-family scale
ladder, (ii) isolated dialogue→inventory mapping load, (iii) surface-variation vs
underdetermination error separation, in a single controlled sweep**"으로 쓰고 위 5편을 foil로
명시 인용.

**방법론 지뢰 (crossover 실험 설계에 선반영 의무)**: Schaeffer et al. (NeurIPS 2023 Oral,
arXiv:2304.15004) — emergence는 discontinuous metric의 인공물. 우리 pass^all·flip 판정이
정확히 그 metric 클래스 ⇒ "경계는 scale-flat" 주장 시 **per-error-type continuous rate 병행
보고**를 실험 설계에 박아야 리뷰 방어 가능. (McKenzie inverse-scaling은 prior-override 계열
논거로 [[42]]와 합치 — 인접 인용.)

⚠️이 지형은 2026년 들어 월 단위로 채워지고 있다(2605·2607 신간 다수) — **출고 시점 재수색 필수.**

---

## §7. 초기 답변([D]) 대비 정정 목록 (자기정정)

| 초기 주장 | 실측 | 성격 |
|---|---|---|
| "WSD는 IAA ~80%에서 포화·경계 상한이 과제에 있음" | determined 영역은 스케일이 천장 돌파(82.3%·SOTA 무차이). **flat인 것은 hardEN 경계뿐**(45.6%) | **정밀화 필수** — 소박 버전은 반박됨 |
| "BFCL은 irrelevance가 frontier 최약 축" | 2024엔 참(61 vs 86). **2026 현재 최약은 relevance**(Opus 4.5가 62.5)·irrelevance는 77~87로 개선 | 시점 갱신 |
| "해법은 되묻기/기권" | + 분포 학습(제3 레버) + 경계-타깃 훈련(최대 효과) 병기 필요 | 과소 열거 |
| CLAM 인용 | 학회 게재 미확인 preprint·제목에 "with **Generative** LMs" | 인용 형식 |
| "The Art of Saying No" 서베이 | **존재하지 않음** → "Know Your Limits" (TACL 2025·구제목 "The Art of Refusal") | 환각 교정 |
| Kuhn "CLAM" 2022 단독 연도 | v1 2022-12 / v2 2023-02 | 소소 |

## §8. 인용 등급·검증 실패 목록 (정직 기록)

- **⚠️미검증·인용 금지**: NexusBench(정량 표 검증 실패) · Ask-before-Plan 세부 수치(abstract에
  없음·본문 필요) · HammerBench/ToolDial(존재만 확인).
- **⚠️venue 주의**: Furrer 2020(arXiv-only) · CLAM(preprint) · Dr.Spider "top-5%" 뱃지(2차 출처) ·
  AbstentionBench venue(OpenReview 봇차단·arXiv로 인용) · Intent-Detection-in-the-Age-of-LLMs
  (venue 미확정).
- **τ² 인용 시**: Amazon AGI `tau2-bench-verified`(태스크-정책 불일치 교정판) 존재 — 원본 수치
  인용 시 병기.
- BFCL 수치는 **라이브 리더보드 CSV**(2026-08-01) — 논문 아닌 리더보드 인용임을 명시할 것.

## §9. 우리 연구에의 함의

1. **모트 주장 구조 확정**: "(a) 스케일-흡수 가능 / (b) 경계 scale-flat·비-스케일 레버 관할"은
   [45]와 동형이고, 이제 각 마디에 검증 인용이 붙는다. task_023 = (b)의 사내 witness.
2. **[46] crossover 실험이 정확히 빈칸을 때림** — 단 (i) Mistral-Interact(7B>GPT-4 판정)와
   2601.08196을 선점 인용으로 양보 (ii) Schaeffer 방어(continuous rate 병행) (iii) foil 5편
   명시가 리뷰 방어 조건.
3. **A2/scaffold 설계 지지**: §4(enum 제약이 병목·자유 서술 98%)는 [[22]] "열린 술어는 LLM+
   표면화"의 문헌 근거 — 경계 사례를 scaffold가 판정하려 들지 말고 표면화(ASK·제시)하라는
   설계 원칙이 문헌과 수렴.
4. **유료 실험 우선순위 불변**: 경계 축은 스케일로 안 닫히므로 7B+비-스케일 레버 스택의
   가치가 문헌으로 재확인 — E-큐·Y-트랙 우선순위 유지.

## 부록 — 클러스터별 검증 원본

딥리서치 5클러스터의 원본 보고(전 인용 URL 포함)는 세션 transcript에 있음. 본 문서의 모든
수치는 해당 보고에서 원문-추출([S])만 승격한 것. 재검증 시 각 표의 arXiv ID로 원문 직독.
