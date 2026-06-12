# 이종-풀 선별기 설계 (detail, 2026-06-12 야간 — 문헌 deep-research 합류판)
> 📌 **구조 안내**: 마스터 = `EXPERIMENT_DESIGN.md` (§7 문서지도). 결과 권위 = `../reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md` **§8.9b**(선별기 1차 실측)·**§8.10**(N2 공식 +8.8). 문헌 근거 = `../reports/facet_rft_2026/research_selector_lit_2026_06_12.md`(31 fetch-검증 인용 — 이하 [SEL-RPT]). 지표 규율 = 마스터 **§1.6 v2**(F5·ⓟ2).

## 0. 문제 좌표 (실측 — 재유도 금지, §8.9b 인용)
- 이종 풀(동일 7B base의 LoRA 변종 AR8+H6=14 제안)의 **oracle 천장 0.856 vs 단일-best 0.720 = Δhetero +13.6** — 정답은 풀 안에 있고, 문제는 **gold 없이 고르기**.
- 1차 사다리 실측: raw MBR 0.716 → **proposer-1표 0.751** → +validity 필터 **0.753** (oracle-회수율 44%, F5 정의) → **공식 척도 확정 +8.8**(선별 66.5 vs k0 57.7).
- **★게이트 역선택**: 결정론 validity gate를 *선택기*로 쓰면 0.54 < 풀 평균 0.671 — 검증기 점수의 서열화가 이종 풀에서 깨짐.
- 제약(불변): **gold-free**(선별 시점 정답 무) · **결정론/저비용**(judge ≤7B, frontier API 불가 = 주권-leg) · **구조 출력**(JSON tool-call DAG — 합법성·구조 utility 계산 가능).

## 1. 설계 원칙 (문헌 정합 — [SEL-RPT] §1)
1. **분업 불변: 검증 신호 = 거부권(veto), 합의 신호 = 선택권(chooser)** — AlphaCode(필터→클러스터)·DOCE("trial-test 필터가 가장 간과된 유효 전략")와 동형. 우리 §8.9b 결론은 문헌 정석의 재발견 = 이 축은 추가 여지 적음.
2. **검증기-주도 선별 단독 라인은 구조적 천장** — 불완전 verifier의 FP율은 resampling으로 안 줄고 정확도 상한을 박음(Stroebl+24). 게이트 역선택은 그 악화판(FP가 후보 간 계통적). ⇒ 검증기는 영원히 veto/보조-vote.
3. **likelihood-단독 재랭킹 금지** — 퇴화-해 선호 병리(Coder-Reviewer가 명시; MAP 부적합성 Eikema&Aziz). D2 brevity-prior·게이트 역선택과 같은 족보.
4. **fusion(후보 융합 생성) 기각** — GenFuser/MoA 류는 비합법 DAG 생성 리스크 = evaluator 구조-동형성·결정론 제약과 충돌. 우리는 순수 **after-inference selection** 사분면.
5. zero-cost 우선 (memory `feedback-zero-cost-diagnosis-strongest-case`): 기존 rollout 산출물로 분리 검증 가능한 레버부터.

## 2. 선별기 사다리 (사전등록 — 비용 오름차순, 각 단 독립 판정)
| 단 | 방법 | 비용 | 문헌 계보 | 사전등록 예측 |
|---|---|---|---|---|
| **SEL-1** | **proposer-prior 가중 prop-MBR**: Smoothie-식 label-free 품질 추정(이종 출력 위 latent-variable)으로 proposer별 prior 추정 → 1표 가중 | **0원** (기존 rollout 재분석) | Smoothie [2412.04692, NeurIPS'24] + MBR bias-diversity 분해 [2410.15021] | 회수율 44%→**≥50%** (prior가 약-proposer 표 디스카운트) |
| **SEL-2** | **soft-approval validity votes**: 게이트/검증 신호를 이진 veto가 아닌 **연속 승인 점수의 보조 투표**로 합산 (lexicographic 서열화 금지) | 0원 | Multi-Agent Verification [2502.20379] (약-verifier 승인 집계 = weak-to-strong) | SEL-1 위에 +1~3%p (역선택 신호의 정보 회수) |
| **SEL-3** | **margin-기반 abstention**: 1위-2위 utility 갭 + 승자-클러스터 점유율 → confidence → risk-coverage 운용(F6 기계 그대로) | 0원 | Geifman&El-Yaniv'17 · semantic-entropy 클러스터 엔트로피 [2302.09664] | 비-회수 56%의 고위험 부분을 HITL로 — **coverage@risk 곡선 신규 산출**(점수 예측 아님) |
| **SEL-4** | **7B reverse-likelihood 합성**: p(instruction\|plan)을 7B로 채점해 MBR utility와 곱/합성 (Reviewer 항) | 후보당 1 pass | Coder-Reviewer [2211.16490, ICML'23 — 최대 +17%p] | 회수율 **≥60%** 후보 (MBR과 직교 — 소수-정답 구제) |
| **SEL-5** | **MBR-shortlist(top-3~5) + 7B pairwise judge 토너먼트**: train-time gold로 judge 미세조정 합법 | shortlist당 ~10 pass | LLM-Blender PairRanker [2306.02561] · PoLL [2404.18796] · Prometheus 2 [2405.01535] | 최종 단 — SEL-4까지의 잔여 oracle 갭에서 판단 |
- **판정 규율(전 단 공통)**: F5 회수율 = (선별−mean)/(oracle−mean), **paired bootstrap 95% CI 동반**(ⓟ2) · 내부-일관 비교(동일 풀·동일 평가) · 공식 척도 확인은 sub500 N2-프로토콜 재사용 · 단별 즉시-기각 조항 = CI가 직전 단 대비 0 이득 포함 시 그 단 폐기.
- **시드 확장 별도 축**: MBR 수렴 O(n^-1/2) [2502.12685] — K=14는 추정분산 잔존, 풀 확대는 선별기와 독립 레버(GPU 비용 발생 — 후순위).

## 3. Novelty 좌표 (논문 자리 — [SEL-RPT] §4 검증)
1. **상관-소스 투표 보정**: 같은 정책 K샘플의 합의 지배(다수-블록 편향)를 source-aware로 보정하는 선행 **부재** — proposer-1표·prior 가중의 이론화 자리(bias-diversity 분해의 옆).
2. **결정론 게이트의 이종-풀 역선택**: 보고 사례 **부재** — 우리 census가 첫 실측.
3. **metric-homomorphic 구조 utility**: 실행-기반 MBR(MBR-Exec)과 n-gram MBR 사이의 빈 자리 — DAG 정적분석(타입 전파·슬롯 체결) = "유사-실행" utility.
4. 4-제약 교집합(이종풀 × gold-free × ≤7B judge × JSON DAG) 직접 선행 무.

## 4. 실행 큐 (마스터 §0-§4 순서 변경 없음 — GPU 큐 등재용)
- ⑴ SEL-1+2+3 (0원, 기존 N2 rollout 재분석 — 단일 스크립트 `tb_selector_v2.py` 신규) → F5 v2 행 + PORTFOLIO 갱신.
- ⑵ 적중 시 SEL-4 (7B Reviewer 채점 배치 — GPU 소량) → 선별기 합성 평가(v3+guided 57.90 위 N2-식).
- ⑶ SEL-5는 ⑵ 잔여 갭 보고 결정. 병행: τ²에도 동일 사다리 이식 검토(pass^k 풀이 이미 있음 — K=4 trials를 풀로).
