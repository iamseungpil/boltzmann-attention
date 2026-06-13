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
| **SEL-5** | **MBR-shortlist(top-3~5) + 7B pairwise judge 토너먼트**: train-time gold로 judge 미세조정 합법 | shortlist당 ~10 pass | LLM-Blender PairRanker [2306.02561] (ranker=**DeBERTa-400M**·O(n²) → shortlist 압축 정당) · **Prometheus 2 [2405.01535]**(=실재 7B 오픈 evaluator·on-prem 닻) · ⚠️PoLL [2404.18796] | 최종 단 — SEL-4까지의 잔여 oracle 갭에서 판단. ⚠️**PoLL 인용 규율(relwork_selector §C)**: 패널=Command-R-35B+Haiku+GPT-3.5(proprietary API) = "**frontier 단일 judge 불필요**"의 일반근거로만 인용, **≤7B/on-prem 주권-leg 근거로는 금지** |
- **판정 규율(전 단 공통)**: F5 회수율 = (선별−mean)/(oracle−mean), **paired bootstrap 95% CI 동반**(ⓟ2) · 내부-일관 비교(동일 풀·동일 평가) · 공식 척도 확인은 sub500 N2-프로토콜 재사용 · 단별 즉시-기각 조항 = CI가 직전 단 대비 0 이득 포함 시 그 단 폐기.
- **시드 확장 별도 축**: MBR 수렴 O(n^-1/2) [2502.12685] — K=14는 추정분산 잔존, 풀 확대는 선별기와 독립 레버(GPU 비용 발생 — 후순위).

## 3. Novelty 좌표 (논문 자리 — [SEL-RPT] §4 검증)
1. **상관-소스 투표 보정**: 같은 정책 K샘플의 합의 지배(다수-블록 편향)를 source-aware로 보정하는 선행 **부재** — proposer-1표·prior 가중의 이론화 자리(bias-diversity 분해의 옆).
2. **결정론 게이트의 이종-풀 역선택**: 보고 사례 **부재** — 우리 census가 첫 실측.
3. **metric-homomorphic 구조 utility**: 실행-기반 MBR(MBR-Exec)과 n-gram MBR 사이의 빈 자리 — DAG 정적분석(타입 전파·슬롯 체결) = "유사-실행" utility.
4. 4-제약 교집합(이종풀 × gold-free × ≤7B judge × JSON DAG) 직접 선행 무.

## 4. 실행 큐 (마스터 §0-§4 순서 변경 없음 — GPU 큐 등재용)
- ⑴ ✅완료(06-12 심야): **SEL-1 채택**(β=2 SIG·공식 66.48→67.22)·SEL-2 기각·SEL-3 작동. 권위 = TB결과 §8.9c.
- ⑵ ✅완료(06-13): **SEL-4 신기록 — 최적 dpo2g-풀 + 7B Reviewer = 공식 0.6803**(+0.81pp, 사전등록 적중) = k0 대비 **+10.3pp**, best-stack=SEL-1+SEL-4. 음성 2건(풀확장·v3g풀)은 다양성-부검으로 종결(PORTFOLIO §3.7d: oracle 동일·다양성 −33% = 선별=다양성 함수 정량 확정). 권위 = TB결과 §8.9d.
- ⑶ ✅**SEL-5 기각 (2026-06-14, `tb_pairwise_select.py`·`driver_sel5.sh`)**: MBR-shortlist + 7B pairwise judge 토너먼트(순서편향 제거)를 best-stack dpo2g 풀에 적용 — **shortlist=3 = 0.6690(26 flips)·shortlist=5 = 0.6635(41 flips)**, **둘 다 SEL-1 0.6722·SEL-4 0.6803 미달**. **단조 악화**(개입↑=flip↑=점수↓) = same-base 7B judge가 MBR 합의를 뒤집을 때 맞기보다 틀림. ⇒ **SEL-2(soft-approval)에 이어 SEL-5도 기각 = 설계원칙 #2(검증기/judge-주도 선별의 구조적 천장·Stroebl 불완전-verifier) 두 번째 실증**. 잔여 oracle 갭(0.680→0.856)은 **same-base pairwise judge로 닫히지 않음** — 군중 합의(SEL-1)+거꾸로검증(SEL-4)이 스위트스팟. ⚠️**"실효 천장" 주장은 보류(2026-06-14 정정)**: 검증된 음성은 *same-base judge/pairwise/soft-validity류 한정*. 미검증·비-중복 레버 잔존: **(B1) self-certainty/logprob**(우리 shared-base = cross-candidate 비교가능 *희귀 레짐*, relwork_selector §12 — judge와 다른 신호라 SEL-5 음성이 배제 못 함·zero-GPU) · **(C) 갭 0.176의 eval-tie/노이즈 분해**(선별가능분 미측정 — 천장 선언 선결) · (B2) utility 재설계 · (B3) SEL-1·4 결합/λ 튜닝 · ②의 *different-base 7B judge*. ※"①풀 다양성"은 새 proposer 추가=생성기-강등 라인과 충돌(또는 기존풀 D-oracle 게이팅=오라클 0.856에 bounded한 선별-curation)이고, "②축-다양 검증기"는 이미 **V-line(§6)·V-2 MAV(day13 부분실행)** 예정 — 둘 다 "새 프런티어" 아님. 차기 zero-cost 진단 = **B1 + C**. 병행: τ² 이식은 대화형 trajectory-선별 의미론 검토 후.
- ⑷ **둘째-기판 재현 (사전등록 2026-06-14 — 추세리뷰 위험2[단일-기판 의존] 방어)**: SEL-1+SEL-4 best-stack을 **TB huggingface sub500**(이종 도메인·기존 예측: lodo_hf±guided·Track-B hf preds)에서 1회 재현. 예측: k0 단일 대비 **≥+3pp 공식**(MM +10.3pp가 기판-특이 아님 입증; 미달 = "MM-특이" 정직 기록·헤드라인 강등). 선행 = hf AR8 K=8 샘플링(GPU ~2h). ✅**적중 (2026-06-13): C0 0.3504 → SEL-1 0.4791 = +12.9pp (사전등록 4배 초과) — 헤드라인 자격 획득**. 단 SEL-4는 hf서 −0.3pp = 기판-의존 옵션으로 정정. 권위 = TB결과 §8.9e. ✅드라이버 `driver_substrate_hf.sh` (AR8=tb_lodo_hf+guided temp0.8 — LODO held-out 전이 설정·hetero=qwen 4종+Track-B hf 4종).
- ⑸ **다양성-생성 실험 (사전등록 2026-06-14, 리뷰 2순위 — '선별=다양성 함수'를 사후 부검에서 예측 법칙으로 승격)**: matched-compute K=8 생성정책 3종 비교(MM sub500) — ①P-temp: 단일 어댑터(dpo2) temp {0.5,0.8,1.1,1.4}×2 ②P-lora: 어댑터 8종 각 1샘플(이종 체크포인트 — 현 최적 가설) ③P-prompt: 단일 어댑터 temp0.8 + 시스템-프롬프트 변주 8종. **사전등록**: (i)풀 다양성(평균 쌍별 1-F1)이 per-id 선별이득을 예측(회귀 기울기>0, paired bootstrap CI 0 제외) (ii)순위 예측 P-lora > P-temp > P-prompt (iii)다양성 순위=선별이득 순위 일치. 의의 = 2502.12118 heterogeneity 전제의 실측 검증(이론-실험 페어링) + **P-D(diffusion) 재개의 선결 게이트**(AR-내 다양성 천장 확정 후 "AR로 못 만드는 다양성"만 diffusion 정당화). 딥리서치 합류 완료(research_diversity_2026_06_14.md, 35 검증인용) — 설계 확정 반영:
  - **★P-unguided arm 추가 (문헌 1순위 개입, 0-GPU)**: guided JSON이 다양성을 구조적으로 깎음(GAD/ASAp NeurIPS'24 = GCD 분포왜곡 증명·Price of Format 2505.18949 = 구조토큰이 고온에서도 다양성 붕괴) ⇒ 4번째 정책 **P-unguided: guided OFF + validity-filter/snap 후처리**(현 AR8은 전부 guided=다양성 자해 가능성). 예측: P-unguided 다양성 > P-temp(guided) ∧ 선별 후 공식 동등+ — 적중 시 "풀은 unguided, 최종 출력만 guided" 기본값 전환.
  - P-prompt 근거 확보: multi-prompt bank→MBR (Heineman+ 2407.15343, EMNLP'24 — 이득이 "더 다양한 후보공간" 귀속 명시)·Verbalized Sampling(1.6-2.1× 다양성)·arithmetic sampling(+3-5pp SC).
  - v3g 붕괴의 정준 인용 확보: Kirk+ 2310.06452(RLHF 다양성 절감)·DivPO 2501.18101("post-training sharpens the distribution") — **v4 재채굴 사이클에 DivPO-식 쌍선별/GEM 도입 prereg**(이중 게이트: k0 강도 ∧ 풀 다양성 ≥ rft2 수준).
  - LoRA-풀을 MBR 후보 생성기로 쓰는 발표 선행 **부재 확인** = +10.3pp 스택의 기여 좌표(검증된 공백).
  - ✅**실행 완료 (2026-06-13, TB §8.9f)**: **P-lora(목적-다양 어댑터 8종) 다양성 0.1535 = 단일정책 ~10배·H6 6배**·oracle 0.874·이득 +0.0175 / **회귀 gain~diversity 기울기 +0.077 SIG[CI 0.020,0.140]** = 예측 법칙 승격(사전등록 적중). P-unguided>P-temp = guided 다양성 자해 확인. ✅**P-lora 본격 선별 완료 (2026-06-13, TB §8.9g)**: C0 0.567→SEL-1 0.604(+3.7)→+H6 0.633(+6.6)→SEL-4 0.615. **nuance = 다양성 최고인데 best-stack(0.680) 미달** = 개별품질↓(목적-편향 어댑터) → sel≈mean+회수×(oracle-mean). ✅**통합풀 음성+부검 (TB §8.9h)**: 22종 SEL-1 0.626/SEL-4 0.636 < best-stack 0.680. 곱-가정 기각 — mean ↑(+0.011)·oracle 불변·**회수율 61%→40% 붕괴**가 원인. P-lora=쌍별다양성 0.15지만 **D-oracle≈0**(새 정답 무·합의만 교란). ⇒ **풀 admission = D-oracle>0 게이트**(쌍별다양성 부적합); 최적 = dpo2g+H6 유지. §7 측도보강이 정확히 예측.

## 5. 외부근거 직독 추가 (2026-06-14, 4편 원문 정독 — 상세·verbatim = `FIELD_GAP_LLM_VALUE_DESIGN.md` §5.5)
1. **이론 닻 신규 — VB/VF 분리정리** [`2502.12118` v2]: 검증-채널 이득 **Ω̃(H/√n)**의 전제 = base 분포 **heterogeneity + anti-concentration** → §0 Δhetero +13.6·N2 "다양성 함수" 기제의 이론 대응물 = **"다양성 없으면 검증-선별 이득도 없다"가 정리 수준에서 성립**(우리 E6 실측과 동형). ⚠️정식 대상=파인튜닝 — 선별 사상은 확장해석 명시 후 인용.
2. **SEL-2 기각 해석 확정 — MAV 직독** [`2502.20379` v1]: BoN-MAV 작동 전제 = 다축 검증기 다양성 + **held-in validation으로 검증기 부분집합 선별**; GPQA tie·HumanEval 역전 = 무조건 작동 아님. ⇒ 우리 NS 기각 = **모순 아닌 조건차**(단일 게이트 신호·축 다양성 0·validation-선별 무). 재도전 조건 = 검증기 축 다양화+집합 선별 — 현 우선순위 낮음(SEL-4/5가 선행).
3. **설계 원칙 1 외부증거** [`2506.12928` v1, GAIA agent-TTS]: **list-wise(후보 상대비교) > scoring > voting** + 상시 reflection 해로움 = "병렬 K-제안+상대비교 선별 > 순차 수정"의 에이전트-도메인 독립 증거.
4. ❌**철회 (2026-06-14, relwork_diversity §5 — 내용 불일치)**: `2601.15808`을 "게이트 역선택의 외부 동형(검증-측 스케일링 천장 = 오기각 지속)"으로 인용한 것은 **오인용**. 직독 abstract상 실제 논문 = Wan et al. *"Inference-Time Scaling of Verification: Self-Evolving Deep Research Agents"*(GAIA/XBench 검증-스케일, ACL'26 Findings)로 "오기각(correct→incorrect) 천장"과 무관. ⇒ **게이트 역선택의 외부 동형은 imperfect-verifier 천장 정리 [`2411.17501`, Stroebl+24: FP>0 ⇒ resampling 정확도 상한·최적 K 매우 작음]로 대체**(relwork_selector §5/§B). 본 ID 재인용은 full-text 재검증 후에만.
## 6. ★V-라인: 검증-다양성 (multi-axis soft 재진입 — 2026-06-14 정밀 설계, SEL-2 기각의 조건 충족판)
> 재진입 조건(§5 MAV 직독에서 박제): ①검증기 축 *다양화* ②held-in validation으로 부분집합 선별. SEL-2 기각 = 단일 축(gmem)의 soft화 — 정보가 하드필터에 이미 소진된 신호의 재포장이었음. SEL-4(+0.81pp)가 soft 신호 성공례인 이유 = **직교 축**. ⇒ V-라인 = 직교 축을 *체계적으로* 늘리고 MAV-레시피로 집계.

### V-1 결정론 축 라이브러리 (0원·CPU — 즉시 구현 가능, `tb_axes.py` 신규)
| 축 | 정의 | 상태 |
|---|---|---|
| A1 gmem | 링크의 도메인-그래프 멤버십 비율 | 기존 (SEL-2 단독 기각분) |
| A2 valid_frac | 노드명 카탈로그 유효 비율 | 기존 (하드필터와 중복 주의) |
| A3 struct | nself+ndangle (음수화) | 기존 |
| **A4 인자-정합** | 노드별 arguments 수/형이 tool_desc params와 일치하는 비율 — **"유사-실행" 정적분석의 1보** (§3 novelty #3 구현 시작) | ★신규 |
| **A5 DAG-위상** | 고립 노드 수·약연결 컴포넌트 수(음수화)·루트/리프 존재 | ★신규 |
| A6 reviewer | SEL-4 p(instr\|plan) z-점수 | 기존 (LLM-soft 축) |
- 전 축 후보-단위 스칼라로 정규화(0원 — 기존 preds 재분석). A4/A5가 진짜 신규 정보(하드필터·gmem과 비중복).

### V-2 MAV-식 validation-선별 집계 (0원 — V-1 후 즉시)
- **분할 규율**: sub500 → val 100 / test 400 (id-층화, seed 고정·사전 공개). val gold 사용은 합법(train-gold 동급) — 분할·동결 절차 보고 의무.
- **집계 3형 비교** (val에서 cutpoint/가중 보정 → 동결 → test 1회):
  (i) **이진 승인 합산** (MAV-충실: 축별 val-보정 cutpoint → 승인 수 합산을 MBR utility에 가산)
  (ii) **z-선형 결합** (SEL-4 일반화: val에서 ridge/grid 가중)
  (iii) lexicographic 서열 (v0 동형 — **음성 통제**: 역선택 재현 예상)
- **축 부분집합 선별**: val에서 greedy forward selection (축 추가가 val 선별-F1 올릴 때만 채택 — MAV의 "검증기 엔지니어링" 대응).
- **사전등록**: ⓥ1 test에서 (i) 또는 (ii) ≥ SEL-1+4 베이스라인(0.6803) **+0.5pp** → V 채택 / ⓥ2 (iii)이 최하 = 역선택 기제의 내부 통제 적중 / ⓥ3 A4·A5 중 ≥1축이 greedy 선별에 잔존(신규 축의 정보성). 전 판정 paired bootstrap CI.

### V-3 합성 2×2 (capstone — 후보-다양성 × 검증-다양성 가산성)
- arms: {풀: 현최적(dpo2g-AR8+H6), +D4(또는 +cross-family — P-D1' 승자)} × {선별: SEL-1+4, +V축}.
- **사전등록**: ①두 처방 가산적(상호작용 ≥ −0.3pp — 잠식 없음) ②최종 stack 공식 link F1 **≥ 69** ③D-기여는 V축 있을 때 더 큼(소수-정답 구제 채널 강화 — P-D2' 예측과 일관).
- 의의: "다양성 처방의 두 면(후보/검증)"이 합성 가능함을 보이면 R6의 완성형 = **diverse-propose → multi-axis-verify → consensus+rescue-select**.

### 실행 순서·비용 (GPU 큐 조율)
0원 즉시: V-1 구현 → V-2 (기존 풀 재분석) ∥ GPU 순서: SEL-q⑸(AR-내 천장) → P-D0/P-D1'(또는 P-D-alt) → V-3. 전 단계 즉시-기각 조항 = CI 0 포함 시 해당 단 폐기.

## 7. ★다양성 측도 보강 (2026-06-14 사용자 지적 — 0원·CPU, P-lora 회귀 재분석)
**문제 (P-lora 결과 부검, TB §8.9f)**: 현 다양성 측도 = 평균 쌍별 (1−link_F1). 한계 ①**"다르게 틀림"과 "다르게 맞음" 미구분**(단순 거리 — 둘 다 틀린 후보쌍도 다양성으로 계상) ②링크집합 1~3개라 F1이 거칠게 점프(0/0.5/1) → 절대값 작게(0.15) 압축. ⇒ per-id 회귀 기울기 작아 보임(+0.077). **그러나 진짜 신호는 oracle**: P-lora oracle 0.874(P-temp 0.774 대비 +0.10 = 다양성이 정답을 풀에 넣은 양)인데 선별 회수율 18%뿐 = **병목은 다양성 아닌 선별기**.
**보강 측도 (사전등록, `tb_divgen_analyze` v2)**:
- **D-oracle = oracle(풀) − max single-policy** : "다양성이 정답을 *추가로* 풀에 넣은 순 기여"(틀림-다양성 배제). 이게 선별이득의 진짜 상한.
- **D-unique = 정답(edge-F1=1 또는 ≥τ) 후보의 distinct 링크집합 수** : "서로 다른 *정답* 경로 수"(ND 교훈의 unique-correct census와 동형).
- **회귀 재설계**: per-policy(점=정책, n=4~6)로 **D-oracle ~ 다양성** + **선별이득 ~ D-oracle** 2단 — per-id 노이즈 회피. 사전등록: 선별이득/D-oracle = 회수율이 핵심 지표(현 18% → SEL-1+4 적용 후 ↑ 목표). ⓡ1 D-oracle이 1−F1보다 선별이득과 상관 강함(R² 비교).
- **연결**: 회수율이 진짜 병목이므로 **다음 GPU 1순위 = P-lora 풀 + SEL-1+SEL-4 본격 적용**(oracle 0.874 회수 시도) — 측도 보강은 그 이득을 정확히 귀속(다양성-기여 vs 선별-회수 분해).

## 7. ★XGrammar validity-floor 설계 실험 (2026-06-14, `tb_validity_floor.py`, zero-GPU — relwork_arch §3b#1)
> 질문: grammar-constrained decoding(enum 스키마 = `taskbench/tb_guided_schema.py`)이 보장하는 "valid JSON-DAG 하한"이 **선별기 D-oracle 분모를 안정화**하는가. floor 부재 시 후보가 새는 2층 = tier-1 sig=None(구조파손→완전드롭)·tier-2 ok=False(도구명 enum위반→격하).

**실측 (dpo2g best-stack 8 AR + 6 hetero, n=6980)**:
| 풀군 | valid% | invalid-name | name-snap 회수 |
|---|---|---|---|
| guided AR8 (dpo2g0-7) | 94~95% | 24~30/500 | 0 (오타 아님) |
| **qwen3b (약 hetero)** | **56.5%** | 216 | 52→66.9% |
| qwen14b/3_4b/3_14b | 74~79% | 102~128 | 3~6 |
| tb_lodo_hf/daily | 96~98% | 12~19 | 6 |
- **tier-1(구조파손) = 전 풀 0** → 파스-레벨 floor 이미 충족. 갭은 전부 **tier-2(도구명)**.
- **name-snap repair = 78/806(9.7%)만 회수** — 무효 도구명 대부분이 **오타 아닌 의미적 환각**(최근접 valid 부재).
- **D-oracle 분모 영향 = +0.16 cand/id (12.35→12.50)** = 한계적.

**★결론 (다소 negative·전략 강화)**: validity floor의 분모-안정화는 **한계적 = 위생 레버지 선별 레버 아님**. ①강한 arm(dpo2g·lodo)은 이미 94%+ ②약한 arm(qwen3b 56.5%)의 무효는 **의미적 환각**이라 snap-repair(9.7%)도 XGrammar enum-강제도 **valid-but-wrong**(=day-5 "다르게 틀림" MBR 합의 교란)으로 바꿀 뿐 D-oracle 무기여. ⇒ **validity ≠ D-oracle 재확인**(§0 게이트 역선택·§7 D-oracle 측도와 동일 족보). XGrammar는 (a)강한 arm의 ~5% enum 누수 정리 = 최종출력 위생 floor로만 채택 (b)약한 arm을 풀에 넣을 땐 enum-강제가 **denominator를 valid-wrong 잡음으로 부풀려 오히려 해로울 수 있음**(D-oracle 게이트 우선 원칙 §6/§7과 정합). **결합제약=선별기** 결론 강화 — floor는 선별기 천장을 못 올림.

**floor 2-arm 비교(GPU 차기, 본 실험이 동기 약화)**: A guided-at-gen(현 AR8, 다양성↓ per P-unguided) vs B unguided+snap-repair-floor(다양성 보존) — 단 본 측정상 B의 snap 회수가 9.7%뿐이라 "unguided+repair가 guided를 대체"는 약함. unguided arm의 *다양성* 이득이 floor 손실을 상쇄하는지는 ⑸ P-unguided 결과와 결합해 판단(생성기-arm 강등 하에 후순위).

## 8. ★천장 진단 라운드: C(갭분해) + B1-zero(자기일치) (2026-06-14 — "0.6803=천장?" 정밀 검증)
> 사용자 비판("①②는 강등/예정이라 새 경로 아님")이 촉발. 천장 단정 전 zero-GPU 진단.

**C — oracle 갭 분해 (`tb_gap_decompose.py`)**: no_gap 84.6%. 갭(독립-group 기준): **selectable 50.7%**(oracle를 ≥2독립그룹 달성인데 더 지지받는 *오답*에 밀림)·needle 42.4%(1그룹뿐)·gold-limited 6.9%. ⚠️**기준 민감**: distinct-plan 기준이면 selectable 3.6%로 붕괴(정답 그래프는 보통 1종류라 거의 다 needle로 오분류) → **독립-group이 정확**. ⇒ 갭의 ~절반은 *원리상* 선별가능.

**B1-zero — 자기-일치도 빈도 = 확신 신호? (`tb_selfagree.py`, 사용자 발의·zero-GPU)**: 확신 = AR 8샘플 중 동일답 빈도. ①진단: **8/8 만장일치 92%(3648/3950)→F1 0.795**·분리(8%)는 비단조 저F1. ②선별: **MBR+agree=0.6726 ≈ SEL-1 0.6722(무변화)**·AR-mode 단독=0.6024(<SEL-1). ⇒ **agreement = MBR 합의와 중복 = 선별 레버 아님**(92% 만장일치라 고를 게 없고, 분리 시 빈도가 정답 안 가리킴).

**★종합 — 천장 주장의 정밀화**: 갭의 ~50%는 "selectable"이나, 그건 *정답이 더 인기있는 오답에 outvoted*된 케이스라 — **합의(MBR)로도(이미 짐) 자기확신(agreement, 곧 B1 logprob도)으로도 못 건짐**(모델이 그 소수정답을 더 확신하지 않음). ⇒ **same-base 신호(합의·자기확신) 소진 확정**. selectable 갭은 **생성기와 독립인 신호**(다른-base 검증기 / 다수오류를 잡는 결정론 체커 = ②/V-line §6)로만. needle 42%는 ①(생성-다양: 정답을 더 많은 독립출처가 내게)로만. **메타: 사용자의 "zero-GPU 먼저" 지시가 GPU B1을 (예측적으로) 대체** — agreement 무력 = 같은 "모델 자기확신"인 B1도 ≈SEL-1 예측(B1 완료 시 확인).
