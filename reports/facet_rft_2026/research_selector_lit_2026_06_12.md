# 이종-풀 후보 선별(Selection over Heterogeneous LLM Proposal Pools) 문헌 딥리서치

**날짜**: 2026-06-12 · **발주**: §8.9b 선별기 1차 실측 후속 (TASKBENCH_EXPERIMENT_RESULTS.md §8.8–§8.9b)
**우리 설정**: tool-use plan(JSON DAG) 생성 → 벤치 evaluator 채점. 제약 = **gold-free 선별 · 결정론/저비용(judge ≤7B) · 구조적 출력**. 실측: Δhetero oracle +13.6, 결정론 validity gate = 역선택(0.54 < mean 0.671), prop-MBR+filter 0.753(회수 44%), 공식 +8.8.
**인용 규율**: 본문 모든 레퍼런스는 이 세션에서 arXiv abs(또는 PMLR/검색 다중소스) fetch로 제목·저자·버전 검증 완료. 각 수치 옆에 출처 층위(abstract/secondary) 표기. 미검증 항목은 §6에 격리.

---

## 1. 핵심 답변 요약 — 문헌이 가리키는 다음 수

1. **우리가 수렴한 "filter-as-veto + consensus-as-chooser" 패턴은 문헌의 정석과 정확히 일치한다.** AlphaCode(거동-필터→클러스터링→대표 제출)[Li+22], DOCE(trial-test 필터→MBR — "필터가 가장 간과된 단순·유효 전략")[Li+24-DOCE], Lightweight reranking(쌍별 통계만으로 선별)[Jain+23] 모두 "검증 신호는 거부권, 선택권은 합의"의 변형이다. §8.9b의 분업 결론은 재발견이 아니라 문헌 정합 — 이 축은 더 짜낼 것이 적다.
2. **다음 한 수로 문헌이 가장 강하게 지지하는 것은 reverse-likelihood (Reviewer) 스코어의 합성이다.** Coder-Reviewer[Zhang+22]: p(plan|instr)가 아니라 **p(instr|plan)** 을 7B로 채점 — 우리 제약(결정론·gold-free·7B·1 forward pass/후보) 전부 충족, likelihood 단독 재랭킹의 퇴화-선호(우리 gate 역선택·D2 brevity prior와 같은 류)를 정면으로 교정하며 +17%p absolute까지 보고(abstract). MBR 합의와 직교 신호(소수-정답 구제 가능).
3. **proposer-1-vote의 이론적 자리**: MBR 추정오차의 bias-diversity 분해[Kamigaito+24]가 "유틸리티 추정 분산(diversity)이 스케일링 이득을 설명하고, 상관된 샘플은 유효 표본수를 줄인다"는 프레임을 제공 — 같은 정책의 K샘플 합산-1표는 사실상 *유효표본 보정*. 단 source-aware 상관 보정 자체를 다룬 논문은 못 찾았다(→§4 novelty).
4. **이종-풀 label-free 선별의 최근접 선행 = Smoothie**[Guha+24]: 임베딩 위 latent-variable 모델로 per-sample 품질 추정(라벨 0) → 라우팅. 우리와 차이: 일반 임베딩(우리: 평가척도-동형 edge-F1 utility), 라우팅(우리: 후보 선별), 구조 출력 아님. **Smoothie-식 proposer 품질 prior를 vote 가중에 넣는 것이 보정-없는-prior 문제(§8.9b 잔여 후보)의 문헌 답안.**
5. **7B judge는 '단일 frontier judge 대체'로 이미 정당화돼 있다**: PoLL(소형 3종 패널 > GPT-4 단일 judge, 비용 1/7)[Verga+24], Prometheus 2(7B/8x7B 오픈 evaluator가 인간·GPT-4 상관 최고)[Kim+24], MAV(약한 verifier 다수 결합 → 더 강한 generator도 개선 = weak-to-strong verification)[Lifshitz+25]. 우리의 "7B pairwise judge" 후보는 문헌상 무리수가 아니라 주류 처방.
6. **단, 검증기-불완전성의 상한 정리를 기억할 것**: 불완전 verifier의 false-positive율은 resampling으로 줄지 않고 정확도 상한을 박는다[Stroebl+24]. 우리 gate 역선택은 이 정리의 "FP가 계통적(이종 후보 간 편향)일 때" 악화판 — verifier-주도 선별 단독 라인은 구조적으로 천장이 있고, 합의-주도+verifier-거부권 하이브리드가 맞다.
7. **abstain은 selective prediction의 표준 기계로 처리 가능**: 합의 margin(1위-2위 utility 갭)을 confidence로 쓰는 risk-coverage 운용[Geifman&El-Yaniv17; Wen+24 survey]. 회수 44%의 나머지를 "낮은 margin → HITL"로 돌리는 설계는 문헌 직역이다.
8. **요약 처방(비용 오름차순)**: ⑴ Smoothie-식 proposer-prior 가중 prop-MBR(0원) → ⑵ ε-밴드 cost-aware utility(이미 실측, 0원) → ⑶ 7B reverse-likelihood 합성(후보당 1 pass) → ⑷ MBR-shortlist top-3~5에만 7B pairwise judge 토너먼트(PairRanker-식; 우리는 train-time gold로 ranker 미세조정 가능) → ⑸ margin-기반 abstention으로 HITL 라우팅.

---

## 2. 연구질문별 결과

### RQ1. 구조적 출력의 MBR: utility, 실패모드, 이론

**기본 프레임.** MBR은 max-prob이 아니라 후보 간 기대 utility 최대 출력을 고른다. Bertsch et al.은 self-consistency·output ensembling·range-voting 류 다수 기법이 MBR의 특수례임을 보이고 "추가 데이터/학습 없이 광범위 과제에서 수 포인트 개선"을 정리했다 (It's MBR All the Way Down, arXiv:2310.01387 v1, 2023; abstract). 우리 §8.9b의 "MBR utility를 평가척도와 동형(edge-F1)으로 두는 게 우월"이라는 실측은 이 프레임의 직접 귀결(utility ≈ 평가 metric일수록 bias↓).

**왜 MAP/likelihood가 아니라 MBR인가 (이론).** Eikema & Aziz: NMT에서 "모델의 최빈 출력(mode)에 쌓이는 확률질량이 너무 작아 mode는 본질적으로 자의적"이며, 관찰된 병리는 모델이 아니라 MAP 의사결정 규칙의 문제 — MBR이 분포 전체를 존중하는 대안 (Is MAP Decoding All You Need?, arXiv:2005.10283, COLING 2020; abstract). 이는 RQ3의 likelihood 역선택 병리의 뿌리이기도 하다.

**수렴 보증.** Ichihara et al.: 참조 가설 n개의 Monte-Carlo MBR은 가정 하에 O(n^-1/2)로 최적해에 접근하고, 여러 시나리오에서 MAP보다 빠르게 수렴 (Theoretical Guarantees for Minimum Bayes Risk Decoding, arXiv:2502.12685 v3, 2025; abstract). K=8~14 풀이면 추정 분산이 아직 크다 — 우리 풀 확대(AR8+H6=14)는 이 축에서도 정당.

**bias-diversity 분해 (우리 proposer-1-vote의 이론 옆자리).** Kamigaito et al.: MBR 추정오차 = bias(utility와 인간평가의 괴리) + diversity(품질 추정의 변동성)로 분해, 표본수 증가의 이득은 diversity 항으로 설명되고 둘의 동시 개선엔 트레이드오프 (Diversity Explains Inference Scaling Laws, arXiv:2410.15021 v2, ACL 2025; abstract. v1 제목 "Theoretical Aspects of Bias and Diversity in MBR Decoding"). **시사**: 같은 정책 8표는 pseudo-reference 분포를 편향시켜 bias 항을 키운다 — proposer-1-vote는 reference 분포의 재가중(소스당 균등)으로 읽힌다. 단 이 논문도 *상관 소스 보정*은 명시적으로 다루지 않음.

**구조 출력에서의 utility 선택지(코드 문헌).**
- **실행-기반**: MBR-Exec — 소수 테스트 입력에 실행해 같은 의미의 프로그램을 주변화, "실행-무시 선별 전부를 일관되게 능가" (Natural Language to Code Translation with Execution, Shi et al., arXiv:2204.11454 v2, EMNLP 2022; abstract). AlphaCode — 대량 샘플→프로그램 *거동* 기반 필터/클러스터링→클러스터 대표 ~10개 제출 (Li et al., arXiv:2203.07814 v1, 2022; abstract). DOCE — 후보생성·n-best 재랭킹·MBR·self-debug 통합 비교, **trial unit test 필터가 "간과돼 온 단순·유효 전략"**, 실행-기반 vs 실행-무시 갭 큼 (Li et al., arXiv:2408.13745 v4, 2024; abstract).
- **실행-불가 시 표면 통계**: Jain et al. — 쌍별 n-gram/unigram 일치(UCS)만으로 self-consistency의 일반화로서 재랭킹, 코드·요약·번역에서 강건 개선, 이론·시뮬레이션 분석 포함 (Lightweight reranking, arXiv:2307.06857, ACL 2024; abstract+secondary[리뷰 사이트의 UCS 명칭]).
- **우리 위치**: 우리는 실행 sandbox가 없으므로 MBR-Exec 직역 불가. 대신 **DAG의 "유사-실행" = 그래프 정적분석(타입 전파·입출력 슬롯 체결)** 이 실행-기반 utility의 대용물 후보 — 문헌에 직접 대응물 없음(§4).

**알려진 실패모드 정리**: ① 다수-블록 편향 — 상관 샘플이 합의를 지배(우리 §8.9b ①에서 실측; 문헌에선 bias 항으로만 암시) ② utility-평가척도 불일치(bias) ③ 저품질 풀에서 "평균으로의 회귀"(MBR은 oracle이 아니라 consensus를 고름 — oracle 회수율 상한의 원인) ④ 후보수 적을 때 추정 분산 [Ichihara+25]. ①에 대한 명시적 처방을 제시한 논문은 발견 못 함.

### RQ2. Self-consistency의 이종-풀 일반화 / answer-level 앙상블

**계보.** Self-Consistency: 다양한 reasoning path 샘플→최빈 답 주변화, GSM8K +17.9%p 등 (Wang et al., arXiv:2203.11171 v4, ICLR 2023; abstract). 자유형식 출력으로의 확장 = USC: LLM 스스로 "가장 일관된 후보"를 고르게 함, **코드 생성에서 execution-기반 voting과 동급 성능을 실행 없이 달성** (Universal Self-Consistency, Chen et al., arXiv:2311.17311 v1, 2023; abstract). 비용 측 일반화 = Adaptive-Consistency: 합의가 이르게 형성되면 샘플링 중단, 표본 예산 최대 7.9× 절감·정확도 −0.1% 미만 (Aggarwal et al., arXiv:2305.11860 v2, EMNLP 2023; abstract). → 우리 선별기에 USC식 "7B에게 K후보를 한 컨텍스트로 주고 고르게 하기"는 저비용 베이스라인으로 추가 가치 있음(단 K=14 DAG는 컨텍스트 부담).

**이종 모델 앙상블 (answer-level).**
- **LLM-Blender** = 이 분야의 원형: PairRanker(입력+후보쌍 cross-attention 인코딩으로 쌍별 비교; ChatGPT 랭킹과 최고 상관) + GenFuser(상위 후보 융합 생성) (Jiang et al., arXiv:2306.02561 v3, ACL 2023; abstract). PairRanker는 우리 "7B pairwise judge"의 직계 선행 — 단 RoBERTa-급 학습 ranker라 우리처럼 train-time gold가 있으면 7B 미만으로도 가능함을 시사.
- **Mixture-of-Agents**: 계층화된 제안→재생성 구조로 오픈모델만으로 AlpacaEval 2.0 65.1% (GPT-4o 57.5%) (Wang et al., arXiv:2406.04692 v1, 2024; abstract). 융합(fusion)-형 — 우리 설정에선 evaluator가 구조 동형성을 요구하므로 fusion은 위험(비합법 DAG 생성 리스크), 선별(selection)이 안전.
- **More Agents Is All You Need**: 단순 sampling-and-voting만으로 성능이 agent 수에 따라 스케일, 이득은 과제 난도와 상관 (Li et al., arXiv:2402.05120 v2, TMLR 2024; abstract). 우리 Δhetero=+13.6의 일반판.
- **Skill-MoE(구 Symbolic-MoE)**: 인스턴스별 스킬 추론→이종 expert 모집→aggregator가 k출력 합성, 평균 +8.15%p, **16 expert를 GPU 1장에 배칭** (Chen et al., arXiv:2503.05641 v4, ICML 2026; abstract). 우리 멀티-LoRA 1서버 운용의 학술 대응물.
- **분류 체계**: LLM Ensemble 서베이가 ensemble-before(라우팅)/during(토큰 융합)/after(완성 후 선별·융합) 3분류 제공 (Chen et al., arXiv:2502.18036 v6, 2025; abstract). 우리는 순수 *after-inference selection* — 서베이 기준으로도 가장 배포-경량 사분면.
- **Smoothie** (라우팅이지만 최근접): 라벨 0으로 LLM 출력 임베딩 위 latent-variable graphical model을 세워 per-sample 품질 점수 추정, 14개 중 9개 task에서 최적 모델 식별·라우팅 baseline +10%p (Guha et al., arXiv:2412.04692 v1, NeurIPS 2024; abstract). **핵심 이전 가치**: weak-supervision식 "정답 없이 모델 신뢰도 추정" — proposer 품질 prior를 보정 데이터 없이 얻는 방법론.

### RQ3. Verifier-free / gold-free 선별 — 그리고 역선택 병리

**합의/엔트로피 계열.** Kuhn et al.의 semantic entropy: 출력을 양방향 함의로 의미-클러스터링 후 클러스터 분포의 엔트로피로 불확실성 측정 — 비지도·단일모델·무수정 (Semantic Uncertainty, arXiv:2302.09664 v3, ICLR 2023 Spotlight; abstract; Nature 2024 확장판은 §6 참조). 우리 번역: **proposer-1-vote 합의 후 "클러스터 분포의 엔트로피" = abstention 신호** (RQ5와 접합).

**likelihood/self-confidence 계열과 그 병리.**
- Coder-Reviewer: "샘플링 후 모델 likelihood 재랭킹은 **퇴화해(degenerate) 보이는 해를 선호하는 경향**" — Reviewer 점수 p(instruction|program)를 곱해 교정, 6 데이터셋·8 모델에서 일관 개선, 최대 +17%p absolute (Zhang et al., arXiv:2211.16490 v1, ICML 2023; abstract). **우리 D2 brevity-prior 부검 및 gate 역선택과 같은 족보의 병리에 대한 가장 직접적인 문헌 처방.**
- MAP 부적합성 [Eikema&Aziz20, RQ1 재인용]: likelihood 서열 자체가 품질 서열이 아님 — *결정론 스코어의 서열화가 풀이 이종일수록 깨진다*는 우리 발견의 이론적 친척. 단, **"deterministic validity gate가 이종 풀에서 mean 이하로 역선택"이라는 형태의 보고는 문헌에서 찾지 못했다** (§4 novelty #2).
- Self-certainty: 출력 확률분포만으로 응답 품질 추정(보상모델 무), N 스케일링이 reward model급, self-consistency가 못 미치는 open-ended에 일반화 (Kang et al., arXiv:2502.18581 v3, NeurIPS 2025; abstract). 우리 멀티-LoRA vLLM은 logprob을 공짜로 주므로 **0-추가-pass 보조 신호**로 합성 후보. 단 *이종* 모델 간 logprob 스케일은 비교 불능(보정 필요) — 같은 base의 LoRA들이라 우리 풀은 예외적으로 유리.
- **round-trip consistency**: RTC — 모델이 코드→설명→코드 왕복 후 의미 동등성 체크, 휴먼 라벨 없이 모델 평가, 기존 벤치 성능과 강한 상관 (Allamanis et al., arXiv:2402.08699, ICML 2024/PMLR 235; abstract+PMLR 페이지). 평가 프레임이지만 후보-선별 utility로 직역 가능: plan→7B가 자연어 요약→원 요청과의 유사도 = gold-free 후보 점수. Coder-Reviewer의 p(instr|plan)은 이것의 1-pass 근사.

**종합**: 문헌의 gold-free 신호는 ① 합의(MBR/SC/semantic clustering) ② 자기-신뢰(likelihood/self-certainty — 병리 동반) ③ 역방향/왕복(reviewer·RTC) ④ 약지도 latent 품질(Smoothie)의 4족. 우리는 ①만 깊게 팠고 ③④가 미개척 — 직교 신호라 합성 여지가 가장 크다.

### RQ4. 소형 judge·약한 verifier의 선별력

- **PoLL**: 서로 다른 가문 소형 3모델 패널(command-r, gpt-3.5, haiku)이 GPT-4 단일 judge를 6 데이터셋에서 능가, intra-model bias↓, **비용 1/7 이하** (Verga et al., arXiv:2404.18796 v2, 2024; abstract+secondary[패널 구성은 검색 요약]). → "frontier judge 불가" 제약은 성능 포기가 아님. 또한 *judge 다양성*이 핵심 변수 — 우리 H6 풀의 이종성을 judge 쪽에도 재활용 가능(서로 다른 LoRA가 서로의 후보를 채점).
- **Prometheus 2**: Mistral 기반 오픈 evaluator(7B/8x7B), direct assessment+pairwise ranking 겸용, 8개 벤치에서 오픈 evaluator 중 인간·GPT-4 상관 최고 (Kim et al., arXiv:2405.01535 v2, EMNLP 2024; abstract). 7B pairwise judge의 기성품 후보(단 일반 텍스트 평가 학습이라 DAG 도메인 적응 필요).
- **GenRM**: verifier를 분류기가 아니라 next-token 생성으로 학습(CoT 검증·다수결 test-time compute 활용), Best-of-N에서 GSM8K 73→93.4% 등, 모델 크기·test-time compute에 유리하게 스케일 (Zhang et al., arXiv:2408.15240 v3, ICLR 2025; abstract). 우리가 ranker를 *학습*한다면 분류 head보다 생성식 검증이 7B에서 더 유리하다는 처방.
- **상한 정리 (경고측)**: Large Language Monkeys — coverage(oracle)는 4 자릿수 스케일에 걸쳐 멱법칙으로 계속 오르지만 **검증기 없는 도메인에선 majority voting·reward model이 수백 샘플에서 plateau** (Brown et al., arXiv:2407.21787 v3, 2024; abstract). Inference Scaling fLaws/Limits — verifier가 불완전(FP>0)하면 resampling 정확도에 상한, 현실 가정에선 **최적 샘플수가 10 미만**인 경우 多, 약한 모델일수록 FP율 높음 (Stroebl et al., arXiv:2411.17501 v3, 2024-2026; abstract). → 우리 K=14는 "최적 샘플수" 관점에서 이미 적정 구간; 이득의 남은 자리는 샘플 증설이 아니라 **선별기 품질**이라는 우리 결론과 일치.
- **weak-to-strong**: Burns et al. — 약한 supervisor 라벨로 강한 모델을 fine-tune해도 supervisor를 일관되게 추월(GPT-2-급 supervisor+confidence loss로 GPT-4에서 GPT-3.5-급 회복) (arXiv:2312.09390 v1, 2023/ICML 2024; abstract). MAV — **여러 약한 aspect verifier의 승인-집계(BoN-MAV)가 self-consistency·단일 reward model보다 좋게 스케일하고, 약한 verifier 조합이 더 강한 generator를 개선(weak-to-strong verification)** (Lifshitz et al., arXiv:2502.20379 v1, 2025; abstract). → 우리 validity-특징들(파싱·self-loop·dangling·타입)을 *서열*이 아니라 **독립 approval 투표**로 집계하는 것이 gate 역선택의 문헌-제안 수리법.

**문헌 수치로 본 "7B judge vs frontier judge" 갭**: 직접 수치 비교는 과제 의존이라 단일 숫자는 없으나, PoLL(소형 패널≥GPT-4 judge)·Prometheus 2(상관 최고치)·MAV(약한 verifier 집계의 우월 스케일링)의 3건이 일관되게 "**소형 다수 ≥ 대형 단일**"을 보고. 반대 방향 경고는 fLaws의 "약한 모델일수록 FP율↑" — *단일* 약한 verifier에 선택권을 몰아주지 말 것.

### RQ5. Risk-coverage / 선택적 예측과 abstention

- 고전 토대: Geifman & El-Yaniv — 학습된 분류기에 rejection 함수를 붙여 **사용자가 지정한 위험 수준을 고신뢰로 보장하며 coverage를 거래** (Selective Classification for DNNs, arXiv:1705.08500, NeurIPS 2017; abstract는 다중 소스 검색으로 확인 — abs 페이지 fetch는 제목만 반환). 핵심 기계: 신뢰 점수로 정렬→임계값=원하는 risk를 만족하는 최대 coverage.
- LLM abstention 서베이: Wen et al. — abstention을 query/모델/인간가치 3관점으로 정리, 평가 지표로 coverage@acc·AURCC(risk-coverage 곡선 아래 면적)·abstain-ECE 등 (Know Your Limits, arXiv:2407.18418 v3, TACL 2024; abstract+secondary[지표 상세는 검색 요약]).
- 합의-신호의 재활용: semantic entropy[Kuhn+23]·합의 margin·Adaptive-Consistency의 정지 기준[Aggarwal+23]은 전부 "합의 강도→confidence"의 변형 — **MBR 선별기에서 abstention은 공짜다**: 1위-2위 기대 utility 갭, 승자 클러스터 점유율, proposer-투표 만장일치도가 즉시 confidence 점수가 된다.
- 우리 운용 번역: 선별 후 confidence 하위 τ%를 HITL로 — "회수 44%"는 *전량 자동* 기준이고, risk-coverage 곡선을 그리면 "coverage 70%에서 oracle 회수 60%+" 같은 운용점 선택이 가능해진다(예상; 곡선은 zero-cost로 기존 census에서 즉시 산출 가능 — 다음 실측 후보 1순위).

---

## 3. 우리 설정용 후보 선별기 설계 (랭킹)

전제: 풀=AR8+H6(같은 base 멀티-LoRA, vLLM 1서버), 현 최고 실현 0.753/oracle 0.856, 공식 +8.8. 비용 단위 = 후보당 추가 forward pass.

| # | 설계 | 기대 이득 근거 | 비용 | 리스크 |
|---|---|---|---|---|
| 1 | **Smoothie-prior 가중 prop-MBR**: proposer별 품질 prior를 라벨 없이 추정(출력 임베딩/상호 utility의 latent-variable 모델, Smoothie 직역)해 1표에 가중 | Smoothie 9/14 task 최적모델 식별·+10%p[Guha+24]; 우리 H6 solo mean이 0.348~0.757로 극단 이질 → prior 한 번에 잡으면 qwen3b류 노이즈표 제거. §8.9b 잔여 후보 "proposer 품질 prior(보정 필요)"의 보정-불요 해법 | **0** (기존 출력 재계산) | prior가 in-domain 통계에 과적합; lodo류 신규 어댑터 추가 시 재추정 필요 |
| 2 | **validity 특징의 soft-approval 집계 (BoN-MAV식)**: 파싱·self/dangle·타입-호환·스키마 각각을 독립 0/1 승인으로, 합의 점수에 가산(서열 아님) | MAV: 약한 verifier 다수의 approval 집계가 SC·RM보다 스케일 우월[Lifshitz+25]; 우리 v0 실패는 *lexicographic 서열* 탓이지 특징 탓이 아님(§8.9b "거부권" 실측이 이미 절반 증명) | **0** | 가중치 1~2개 튜닝 필요(ε-밴드 실측처럼 zero-cost 스윕 가능) |
| 3 | **7B reverse-likelihood (Reviewer) 합성**: 후보 plan을 조건으로 원 요청문의 logprob을 7B로 1 pass 채점, MBR 점수와 선형 결합 또는 ε-밴드 타이브레이크 | Coder-Reviewer 최대 +17%p·6셋 일관[Zhang+22]; 합의와 직교(다수-블록이 틀릴 때 소수-정답 구제 = 회수 44%→상향의 주 경로); 같은 base LoRA라 logprob 스케일 비교 가능 | 후보당 1 pass (K=14, ~수초/문항) | 요청문이 짧으면 신호 약함; plan→NL 직렬화 템플릿 설계 필요 |
| 4 | **MBR-shortlist + 7B pairwise judge**: prop-MBR로 top-3~5 압축 후 7B(Prometheus-2-식 또는 자가 LoRA judge)로 쌍별 토너먼트. **우리는 train 도메인 gold가 있으므로 judge를 공식 metric 라벨로 fine-tune 가능(선별 시점 gold-free 제약과 무모순)** | PairRanker가 랭킹 상관 최고[Jiang+23]; GenRM: 생성식 verifier 학습이 BoN에서 대폭 이득·소형에서도 스케일[Zhang+24]; PoLL/Prometheus2로 7B 정당화 | shortlist 쌍당 1 pass (5후보=10 pass) | 학습 들어감(LoRA 1개); judge의 in-domain 편향 — held-out 도메인 검증 필수 |
| 5 | **risk-coverage abstention 레이어**: 1위-2위 utility 갭·승자 클러스터 점유율로 confidence → 하위 τ는 HITL | 선택적 예측 표준[Geifman&El-Yaniv17]; LLM abstention 지표 기성[Wen+24]; semantic-entropy류 신호 재활용[Kuhn+23] | **0** (census에서 곡선 즉시 산출) | 이득이 "정확도"가 아니라 "운용점" — 논문 기여로는 보조 축 |
| 6 | **USC-식 7B 일괄 선택**: K후보를 한 컨텍스트에 넣고 7B가 "가장 일관된 plan" 지목 | USC가 코드에서 execution-voting과 동급[Chen+23]; 구현 5분급 베이스라인 | 문항당 1 long pass | K=14 DAG는 컨텍스트 길이·위치 편향 리스크; 결정론 아님(greedy면 결정론) |
| 7 | (보류) **fusion(GenFuser/MoA식 후보 융합 재생성)** | MoA 등 이득 크지만[Wang+24], 비합법 DAG 생성 리스크·evaluator 동형성 요구와 충돌, 결정론 상실 | 1+ pass | 우리 제약과 정면 충돌 — 선별이 우월하다는 내부 실측(D2 교훈: 생성-side 개입의 교락) |

**권장 시퀀스**: (1)+(2) 동시 [전부 zero-cost, 사전등록: 0.753→0.78± 기대] → (5)로 risk-coverage 곡선 박제 → (3) 1 GPU-시간급 파일럿 → 효과 있으면 (4)를 "학습-선별기" 본 라인으로. (6)은 베이스라인 표 채우기용.

---

## 4. 우리 설정의 NOVELTY — 문헌이 비워둔 칸

1. **이종-풀 MBR의 소스-상관 보정이 무이론 지대**: bias-diversity 분해[Kamigaito+24]·수렴 보증[Ichihara+25] 모두 i.i.d.(단일 정책) 샘플 가정. "같은 정책 K샘플=합산 1표, 이종 모델 일치=독립 증거"(proposer-1-vote)는 우리가 실측으로 찾은 규칙이고(0.716→0.751), 이를 pseudo-reference 분포의 importance-reweighting으로 정식화하면 **그 자체로 논문 기여**. full-dedup의 붕괴(0.731→0.471, "다중성=증거" 소거) 대비가 깨끗한 ablation.
2. **결정론 validity gate의 역선택(anti-selection) 현상 자체가 미보고**: likelihood의 퇴화-선호[Zhang+22]·MAP 부적합[Eikema&Aziz20]·verifier FP 상한[Stroebl+24]은 친척이지만, "**동질 풀에서 +α였던 결정론 게이트가 풀이 이종화되는 순간 mean 이하로 떨어진다**"는 형태(게이트 특징이 proposer 정체성과 교락 → 특정 가문의 표면 관례를 품질로 오인)는 발견 못 함. 메커니즘 가설("특징-서열이 풀 간 분포 이동에 비강건, 특징-거부권은 강건")까지 얹으면 독립 절 분량.
3. **평가척도-동형(metric-homomorphic) utility의 구조 출력 MBR**: 코드 문헌은 실행(MBR-Exec/AlphaCode/DOCE) 아니면 표면 n-gram[Jain+23]의 양극단. **실행 불가능한 구조 출력(tool DAG)에서 "채점 metric과 동형인 그래프-utility(edge-F1)"를 MBR utility로 쓰는 중간지대**는 비어 있다 — node-Jaccard 대비 edge-F1 우월 실측이 utility-정합성 가설의 증거.
4. **4중 제약의 교집합이 무인지대**: (a) 이종 제안 풀 + (b) 선별 시점 gold-free + (c) judge ≤7B(frontier API 금지, on-prem) + (d) 구조적 JSON DAG 출력. 문헌은 각 축을 따로 다룸 — LLM-Blender(이종+학습 ranker, 자유 텍스트), Smoothie(이종+gold-free, 라우팅·일반 임베딩), PoLL/Prometheus(소형 judge, 평가용), MBR-Exec/DOCE(구조 출력, 실행 필요·단일 모델). 교집합을 채우는 시스템 보고 자체가 기여. 특히 **주권-leg(망분리 on-prem sLLM) 시나리오에서 '같은 base의 LoRA 가족 풀 + 결정론 선별 = 추가 서빙비 ~0으로 +8.8 공식'** 은 배포 논거로 문헌에 없는 조합.
5. **ε-밴드 cost-aware utility(§8.10 후속 실측)**: MBR utility에 비용 선호를 명시적 ε-밴드 타이브레이크로 넣는 것 — MBR 문헌의 utility 설계는 품질 단일축이고, 비용-품질 교환비를 *선별기-side*에서 조절 가능한 손잡이로 제시한 사례를 찾지 못함(D2 weight-side 실패와의 대조가 스토리 완성).

**경계 사례(novelty 아님)**: "filter→consensus" 분업(AlphaCode·DOCE 선행), "소형 judge 패널"(PoLL), "K-스케일링의 선별 병목"(Monkeys·fLaws) — related work에서 선행 인정하고 차별점만 주장할 것.

---

## 5. 검증된 서지 (전건 본 세션 fetch-검증; 표기 = 검증 방법)

| 인용 | 서지 | 검증 |
|---|---|---|
| [Wang+22] | Self-Consistency Improves Chain of Thought Reasoning in Language Models, Xuezhi Wang et al., arXiv:2203.11171 v4, ICLR 2023 | abs fetch ✓ (수치=abstract) |
| [Li+22] | Competition-Level Code Generation with AlphaCode, Yujia Li et al., arXiv:2203.07814 v1, 2022 (Science 2022 게재본 존재) | abs fetch ✓ |
| [Shi+22] | Natural Language to Code Translation with Execution (MBR-Exec), Freda Shi et al., arXiv:2204.11454 v2, EMNLP 2022 | abs fetch ✓ |
| [Zhang+22] | Coder Reviewer Reranking for Code Generation, Tianyi Zhang et al., arXiv:2211.16490 v1, ICML 2023 (PMLR v202) | abs fetch ✓ (+17%p=abstract) |
| [Eikema&Aziz20] | Is MAP Decoding All You Need? The Inadequacy of the Mode in NMT, Bryan Eikema, Wilker Aziz, arXiv:2005.10283, COLING 2020 | abs fetch ✓ |
| [Kuhn+23] | Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG, Lorenz Kuhn, Yarin Gal, Sebastian Farquhar, arXiv:2302.09664 v3, ICLR 2023 Spotlight | abs fetch ✓ |
| [Jiang+23] | LLM-Blender: Ensembling LLMs with Pairwise Ranking and Generative Fusion, Dongfu Jiang, Xiang Ren, Bill Yuchen Lin, arXiv:2306.02561 v3, ACL 2023 | abs fetch ✓ |
| [Jain+23] | Lightweight reranking for language model generations, Siddhartha Jain et al., arXiv:2307.06857, ACL 2024 | 검색(arXiv 목록+ACL anthology) ✓ (UCS 명칭=secondary) |
| [Bertsch+23] | It's MBR All the Way Down, Amanda Bertsch et al., arXiv:2310.01387 v1, 2023 | abs fetch ✓ |
| [Chen+23] | Universal Self-Consistency for LLM Generation, Xinyun Chen et al., arXiv:2311.17311 v1, 2023 | abs fetch ✓ |
| [Aggarwal+23] | Let's Sample Step by Step: Adaptive-Consistency, Pranjal Aggarwal et al., arXiv:2305.11860 v2, EMNLP 2023 | abs fetch ✓ |
| [Shen+23] | TaskBench: Benchmarking LLMs for Task Automation, Yongliang Shen et al., arXiv:2311.18760 v4, NeurIPS 2024 | abs fetch ✓ (우리 벤치 근거) |
| [Burns+23] | Weak-to-Strong Generalization, Collin Burns et al., arXiv:2312.09390 v1, 2023 (ICML 2024) | abs fetch ✓ |
| [Allamanis+24] | Unsupervised Evaluation of Code LLMs with Round-Trip Correctness, Miltiadis Allamanis, Sheena Panthaplackel, Pengcheng Yin, ICML 2024 (PMLR 235; arXiv:2402.08699) | PMLR 페이지 fetch ✓ |
| [Li+24-Agents] | More Agents Is All You Need, Junyou Li et al., arXiv:2402.05120 v2, TMLR 2024 | abs fetch ✓ |
| [Verga+24] | Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (PoLL), Pat Verga et al., arXiv:2404.18796 v2, 2024 | abs fetch ✓ (패널 구성 상세=secondary) |
| [Kim+24] | Prometheus 2: An Open Source Language Model Specialized in Evaluating Other LMs, Seungone Kim et al., arXiv:2405.01535 v2, EMNLP 2024 | abs fetch ✓ |
| [Wang+24] | Mixture-of-Agents Enhances Large Language Model Capabilities, Junlin Wang et al., arXiv:2406.04692 v1, 2024 | abs fetch ✓ |
| [Brown+24] | Large Language Monkeys: Scaling Inference Compute with Repeated Sampling, Bradley Brown et al., arXiv:2407.21787 v3, 2024 | abs fetch ✓ |
| [Wen+24] | Know Your Limits: A Survey of Abstention in LLMs, Bingbing Wen et al., arXiv:2407.18418 v3, TACL 2024 | abs fetch ✓ (지표 상세=secondary) |
| [Li+24-DOCE] | DOCE: Finding the Sweet Spot for Execution-Based Code Generation, Haau-Sing Li et al., arXiv:2408.13745 v4, 2024 | abs fetch ✓ |
| [Zhang+24] | Generative Verifiers: Reward Modeling as Next-Token Prediction, Lunjun Zhang et al., arXiv:2408.15240 v3, ICLR 2025 | abs fetch ✓ |
| [Kamigaito+24] | Diversity Explains Inference Scaling Laws: ... MBR Decoding, Hidetaka Kamigaito et al., arXiv:2410.15021 v2, ACL 2025 (v1 제목 "Theoretical Aspects of Bias and Diversity in MBR") | abs fetch ✓ |
| [Stroebl+24] | The Limits of Inference Scaling Through Resampling, Benedikt Stroebl, Sayash Kapoor, Arvind Narayanan, arXiv:2411.17501 v3 (v1-2 제목 "Inference Scaling fLaws"), 2024 | abs fetch ✓ |
| [Guha+24] | Smoothie: Label Free Language Model Routing, Neel Guha et al., arXiv:2412.04692 v1, NeurIPS 2024 | abs fetch ✓ |
| [Ichihara+25] | Theoretical Guarantees for Minimum Bayes Risk Decoding, Yuki Ichihara, Yuu Jinnai et al., arXiv:2502.12685 v3, 2025 | abs fetch ✓ |
| [Kang+25] | Scalable Best-of-N Selection for LLMs via Self-Certainty, Zhewei Kang, Xuandong Zhao, Dawn Song, arXiv:2502.18581 v3, NeurIPS 2025 | abs fetch ✓ |
| [Chen+25-Ens] | Harnessing Multiple LLMs: A Survey on LLM Ensemble, Zhijun Chen et al., arXiv:2502.18036 v6, 2025 | abs fetch ✓ |
| [Lifshitz+25] | Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers, Shalev Lifshitz, Sheila A. McIlraith, Yilun Du, arXiv:2502.20379 v1, 2025 | abs fetch ✓ |
| [Chen+25-MoE] | Skill-Based Mixture-of-Experts (v4; 구 "Symbolic Mixture-of-Experts"), Justin Chih-Yao Chen et al., arXiv:2503.05641 v4, ICML 2026 | abs fetch ✓ |
| [Geifman&El-Yaniv17] | Selective Classification for Deep Neural Networks, Yonatan Geifman, Ran El-Yaniv, arXiv:1705.08500, NeurIPS 2017 | abs fetch=제목만; 저자·내용은 다중 검색 소스(NeurIPS proceedings 링크·dblp·arXiv pdf) — **준검증** |

## 6. 미검증 리드 (인용 금지 — 후속 검증 후보)

- Farquhar, Kossen, Kuhn, Gal — *Detecting hallucinations in LLMs using semantic entropy*, **Nature 630 (2024)**: 검색에서 PubMed/Nature 링크 확인했으나 원문 페이지 fetch 안 함. [Kuhn+23]의 저널 확장판으로 추정.
- Stahlberg & Byrne — *On NMT Search Errors and Model Errors* (arXiv:1908.10090 추정, "빈 문자열이 mode"): likelihood 병리의 고전. 미fetch.
- Ohashi et al. — *On the True Distribution Approximation of MBR Decoding* (arXiv:2404.00752): 검색 결과로만 확인.
- Freitag et al. — epsilon-sampling 기반 MBR 후보 생성 (MT): 검색 노출 안 됨, ID 미상.
- *Multi-Prompt MBR* (arXiv:2407.15343): 검색 목록에서만 확인 — 제안 다양화 축(우리 H6의 prompt판)과 관련 가능.
- *Functional Overlap Reranking* (arXiv:2311.03366): 코드 클러스터-중첩 재랭킹. 검색 목록만.
- *Semantic Entropy Probes* (arXiv:2406.15927): hidden state에서 SE 근사(1-pass). 검색 요약만.
- Kamath et al. — *Selective Question Answering under Domain Shift* (ACL 2020): 분포-이동 하 abstention — 우리 "이종 풀=분포 이동" 프레임에 유관. 미검증.
- *Learning Conformal Abstention Policies* (arXiv:2502.06884): conformal 보장 abstention. 검색 목록만.
- SOTA 추가 후보(2025-26)로 "heterogeneous candidate selection for agent/tool plans"를 표적 검색했으나 직접 대응 논문은 발견 못 함 — §4 novelty 주장과 정합(부재 증명은 아님; AAAI/ICLR 2026 프로시딩 재검 권장).

---
*작성: deep-research 세션 2026-06-12. 본 파일은 §8.9b "deep-research 합류 후 선별기 설계서(detail)" 단계의 입력물.*
