# Deep-Research: 후보 풀 다양성의 의도적 증대 (선별/MBR-기반 test-time compute용) — 2024–2026 문헌 조사

> 작성 2026-06-14 (deep-research 하네스, 전 인용 arXiv abs 페이지 1차 검증·버전 명시).
> 발주 맥락 = TB결과 §8.9b–d(+10.3pp 공식 = 이종-풀 선별; v3g AR8 다양성 −33% 붕괴·대형 단일샷 무이득·"선별 이득=다양성 함수") + PORTFOLIO §3.7d + FIELD_GAP §5.5(VB/VF 분리정리) + TB_DIFFUSION_PROPOSER_DESIGN(P-D0/P-D1 사전등록).
> 출처 등급 표기: Ⓐ=abs 초록 직접 확인 / Ⓣ=본문 표·수치(직접 열람) / Ⓢ=검색 스니펫·2차 — **Ⓢ 단독 수치는 인용 금지 대상**.

---

## 1. 핵심 답변 요약 — 우리 세팅 처방 랭킹 (비용 오름차순)

전제(우리 실측과 문헌의 합치): 선별 이득은 **풀 이질성**의 함수이고(§8.9d ND/NC 음성 2건), 이는 이론적으로 VB/VF 분리정리의 heterogeneity+anti-concentration 전제[Setlur+ 2502.12118v2]와, 실증적으로 "diverse prompts → Best-of-N 오류율 유의 하락"[Wang+ 2502.11027v4]과 정확히 같은 그림이다. 문헌은 우리의 3-계층(0-GPU 디코딩 → 어댑터-풀 설계 → diffusion 제안기) 비용 사다리를 지지하며, 각 층에 즉시 prereg 가능한 레버가 있다.

| # | 레버 | 비용 | 기대 기제 | prereg-가능 예측 |
|---|---|---|---|---|
| 1 | **풀-멤버 unguided(또는 완화-제약) 샘플링** — guided JSON은 k0 단일샷에만 유지, K-샘플 풀은 raw+snap | 0-GPU (재생성만) | GCD는 분포를 왜곡해 grammar-내 고-우도 모드로 질량 집중[GAD 2405.21047v3]; 구조 토큰 자체가 다양성 붕괴 주범·고온에서도 지속[2505.18949v1] | unguided AR8의 intra-pool 1-F1 다양성 > guided AR8 (현 0.016 기준 ≥+30% 회복), oracle ↑, SEL-1+4 공식 ≥+0.5 |
| 2 | **멀티-프롬프트 뱅크 K-샘플** — 같은 어댑터라도 인스트럭션 패러프레이즈 p개 × K/p 샘플 | 0-GPU | 단일 프롬프트는 생성 접근법의 부분집합만 커버; 프롬프트-뱅크→MBR이 "더 다양하고 더 좋은 후보 공간" 입증[Heineman+ 2407.15343v2, EMNLP 2024]; 이론·실증[2502.11027v4] | v3g-AR8(붕괴 풀)에 8-프롬프트 적용 시 다양성 0.016→≥0.024 복원·NC-C1 0.6598 ≥+1 |
| 3 | **샘플링 기하 교체: arithmetic/quasi-random 샘플링** | 0-GPU (구현 비용: vLLM 외부 루프) | 코드북-기반 병렬 다양 샘플링 = 같은 분포에서 저분산·고커버리지 추출[Vilnis+ 2210.15458v2]; SC +3–5pp GSM8K·MBR COMET +0.45–0.89[2411.06251v2] | K=8 동일 예산에서 oracle ↑·prop-MBR ↑ (효과량 소~중) |
| 4 | **멀티-샘플 전용 온도 재최적화 + 열거형 프롬프팅** | 0-GPU | BoN/SC 최적 온도 ≠ 단일샷 최적 온도, 엔트로피-기반 자동 선정[TURN 2502.05234v2, ICML 2025]; "n개 후보 한 번에/순차 조건부" 열거가 독립 샘플링보다 다양성↑·품질 동등[2509.17570v1]·Verbalized Sampling 1.6–2.1×[2510.01171v3] | 온도 스윕 0.8→{1.0,1.2}+min-p형 절단: 다양성↑·validity-필터 후 oracle ↑ (min-p 자체 효과는 계쟁 중 — §RQ1) |
| 5 | **어댑터-풀 설계 원칙 박제: H6 유지 + 목적-이질 멤버 추가** | 어댑터 학습 소량 | LoRA 앙상블=확신-있되-서로-다른 예측[2310.00035v2·2402.12264v2]; 다양성은 seed가 아니라 **데이터/목적 축**에서 옴(우리 H6 실측과 합치); nuclear-norm 류 명시적 어댑터-다양성 정규화 선례[2401.00243v1] | GEM-SFT 멤버 1 + DivPO 멤버 1 추가 시 풀 oracle ≥+1·선별 ≥+0.5 |
| 6 | **v4 학습 사이클에 다양성-보존 목적 채택** — DivPO-식 쌍 선택·GEM-식 SFT loss | 학습 1회 | post-training이 분포를 sharpen해 K-샘플 다양성 파괴(v3g −33%의 문헌 대응물)[Kirk+ 2310.06452v3·DivPO 2501.18101v4·VS 2510.01171v3]; DivPO = "희귀하되 고품질=chosen, 흔하되 저품질=rejected"로 sharpening 자체를 목적에서 차단 | v4 모델의 AR8 intra-pool 다양성 ≥ rft2 수준 유지 ∧ k0 비열화 — **"단일 강함과 풀 다양성의 양립"이 v4의 사전등록 이중 관문** |
| 7 | **Dream-7B diffusion 제안기 (P-D0/P-D1 그대로)** | GPU 1–1.5일 | 도메인-일치 직접 증거: 도구-그래프 계획에서 masked denoising이 Pass@10 커버리지 0.320→0.943 (AR 대비, matched compute)[DiG-Plan 2606.05728v1, IJCAI-ECAI 2026] — 단 우리 P-D(-1)이 보인 "싼 이종성 +13.6"이 선행 베이스라인 | 사전등록 v2 그대로(Dream 한계가치 > 이종-AR 한계가치 요구). **신규 정보: dLLM용 CFG-제약 디코딩 존재[2508.10111v1] = P-D0 형식-리스크① 완화 수단** |

**한 줄 처방**: 1·2번(0원)을 dpo2g-풀·v3g-풀 양쪽에 즉시, 6번을 v4 사전등록에 합류, 7번은 강등 상태 유지(1–6 적용 후 잔여 oracle 갭에서만 발사).

---

## 2. RQ별 상세

### RQ1. 디코딩-시점 다양성 — 무엇이 "유용한" 다양성을 늘리나

**(a) 온도/절단 계열 — 효과는 실재하나 min-p 단독 주장 계쟁 중.**
- Min-p(top-token 확률로 동적 절단)[Nguyen+ **2407.01082v8**, 2025-11-20 Ⓐ]: 고온에서 품질·다양성 동시 개선 주장, ICLR 2025 oral(Ⓢ). **그러나** Schaeffer+ [**2506.13681v2**, 2025-06-19 Ⓐ]가 4개 증거선 전부 재분석 — "하이퍼파라미터 통제 시 min-p는 품질·다양성·트레이드오프 어느 것도 개선하지 못함" 결론. ⇒ **min-p를 다양성 레버로 단독 채택 금지**; 쓴다면 "고온 안정화 보조" 정도로, 자체 prereg 필요.
- 온도 자체는 멀티-샘플 추론에서 재최적화 대상: TURN[Du, Yang, Welleck **2502.05234v2**, ICML 2025 Ⓐ] — BoN/majority-vote용 최적 온도를 라벨 없이 엔트로피-기반으로 자동 선정, 고정-온도 베이스라인을 일관 추월. 우리 K-샘플은 temp0.8 고정 관성 — 0원 스윕 가치 있음.

**(b) 다양성→BoN 인과 — 2025년에 직접 답한 논문이 있다.**
- Wang+ [**2502.11027v4**, 2025-12-19 Ⓐ] "On the Effect of Sampling Diversity in Scaling LLM Inference": **이론**(diverse prompt 샘플의 BoN 후 오류율이 stationary prompt보다 유의 하락) + **diversity–fidelity 트레이드오프 원리**에서 섭동 스타일들을 도출, EM@100 +10.8%(추론)·+9.6%(수학)·Pass@100 +9.5%(코드). **결정적 caveat: majority-voting 집계에선 다양성 이득이 축소·소멸** — 우리 census(§8.9b)에서 "다수-블록 편향 raw MBR < proposer-1표 MBR"로 본 것과 동일한 기제 진술. ⇒ 다양성 증대는 **선별기가 합의-편향을 보정할 때만**(prop-가중·verifier 채널) 현금화된다 — 우리 스택이 이미 그 형태.
- 커버리지 스케일링의 원전: Large Language Monkeys[Brown+ **2407.21787v3**, 2024-12-30 Ⓐ] — coverage가 샘플 수에 log-linear(4 자릿수), 자동 검증기 있는 도메인은 그대로 이득(SWE-bench Lite 15.9%→56% @250샘플), **검증기 없는 도메인은 majority-vote/RM 선별이 수백 샘플에서 plateau**. 우리 K=8–14 영역은 plateau 이전이지만, "커버리지↑가 곧 실현이득 아님 = 선별이 병목" 경고는 E6/P-D2 채택기준과 동일.
- 사전 분포를 바꾸지 않고 추출 기하만 바꾸는 계열: **Arithmetic sampling**[Vilnis+ **2210.15458v2** Ⓐ, ICML 2023 — 2024 이전이나 본 주제의 canonical] 코드북-병렬 다양 디코딩, 기대값 무편향·beam-수준 다양성 보장, WMT에서 reward 추정 분산 절반·BLEU 갭 63% 폐쇄. 후속 재검증[Parashar+ **2411.06251v2**, 2025-04-27 Ⓐ]: GSM8K self-consistency +3–5pp·WMT19 MBR COMET +0.45–0.89, 추가 비용 무시 수준. ⇒ **"같은 모델·같은 분포에서 더 고르게 뽑기"는 공짜 레버로 실증돼 있음** (vLLM 표준 API 밖이라 구현 비용은 정직 기록).

**(c) 프롬프트-수준 확률화(0-GPU 최강 후보군).**
- SimpleStrat[Wong+ **2410.09038** Ⓐ(v2 HTML 확인, abs 페이지 버전·날짜는 v1 2024-10-11 기준)]: 모델 스스로 답-공간을 층화(strata)하고 층을 랜덤 선택해 그 안에서 샘플 — "온도↑=다양성↑" 통념 반박(온도는 개별 품질을 깎으면서 진짜 답-분포 근사도 보장 못 함). 회수형 QA 중심이지만 "구조화된 다양화 > 온도"라는 방향 증거.
- Troshin+ [**2509.17570v1**, 2025-09-22 Ⓐ]: 독립 병렬 샘플링 vs **열거(n개 1-pass)·반복(이전 후보 조건부)** 비교 — 비독립 전략이 동일 예산에서 다양성↑·품질 동등.
- Verbalized Sampling[Zhang+ **2510.01171v3**, 2025-10-10 Ⓐ]: "응답 분포를 말로 출력하라"는 훈련-無 프롬프트 전략, 직접 프롬프트 대비 다양성 1.6–2.1×, 능력 큰 모델일수록 이득↑. (기제는 RQ2 참조 — typicality bias 우회.)
- 입력-층 노이즈(seed-conditioning)[Nagarajan+ **2504.15266v4**, 2025-08-28 Ⓐ]: 창의-조합 과제에서 **입력층 노이즈 주입이 출력층 온도 샘플링과 동등~우월** — "프롬프트-수준 확률화" 계열의 이론적 응원이자, multi-token/diffusion 접근이 next-token 근시안을 넘는다는 주장(RQ5와 연결).
- DPP-기반: 디코딩-시점 DPP는 2024–26에 의외로 얇음. 확인된 것은 **학습-시점** DQO[Chen+ **2509.04784v3**, ICLR 2026 Ⓐ — 응답 임베딩 DPP 행렬식을 RL 보상에 합성] 뿐. (diffusion 병렬 디코딩용 DPP "D5P4" 2603.19146는 Ⓢ만 — 미검증 리드.)

**RQ1 판정**: "유용한 다양성"(서로 다른 *정답* 모드 커버)을 늘린다고 인과적으로 입증된 것은 ①프롬프트-수준 섭동/뱅크[2502.11027·2407.15343] ②추출-기하(arithmetic)[2411.06251] ③구조화 층화[2410.09038]·열거[2509.17570]이고, 단순 온도↑/min-p는 노이즈-측 다양성 위험(개별 품질 하락) 또는 효과 계쟁. **전부 우리 caveat과 동형의 단서**: 합의-류 집계는 다양성 이득을 깎아 먹으니 선별기가 합의-편향을 보정해야 함.

### RQ2. 정렬-유도 다양성 붕괴 — v3g 소견의 canonical 인용

**확정 인용 3종 (우리 §8.9d/§3.7d 소견의 문헌 대응물):**
1. **Kirk+ [2310.06452v3, ICLR 2024 Ⓐ]** — "RLHF는 SFT 대비 출력 다양성을 유의하게 감소(다중 지표), 대신 OOD 일반화는 우월" = 일반화↔다양성 트레이드오프의 canonical. 우리 "v3g 단일은 더 강한데(0.772>0.762) 풀 다양성은 −33%" = 이 트레이드오프의 선별-관점 재발견.
2. **DivPO [Lanchantin+ 2501.18101v4, 2025-05-22 Ⓐ]** — 문제 진술 자체가 우리 문장: "표준 post-training은 출력 분포를 sharpen해 생성 다양성을 줄인다." 처방 = 쌍 선택 규칙 교체(chosen=희귀∧고품질, rejected=흔함∧저품질): persona 다양성 +45.6%·스토리 +74.6%·일반 IF 다양성 +46.2% (winrate +2.4% 동반, DPO 대비).
3. **Verbalized Sampling [2510.01171v3 Ⓐ]** — 붕괴의 **데이터-수준 기제**: 선호 데이터의 **typicality bias**(주석자가 친숙한 텍스트를 체계적으로 선호; 인지심리학 소견) 형식화·선호 데이터셋에서 실증. ⇒ 알고리즘(역-KL)만이 아니라 쌍 자체가 모드-붕괴를 가르친다 — **우리 D1/D2 "대조축 전이" 기제 명제와 같은 족**(쌍의 암묵 축이 도메인-일반 prior로 전이; typicality도 하나의 암묵 축).

**기제 보강 (역-KL mode-seeking 라인):**
- f-DPO[Wang+ **2309.16240v1** Ⓐ]: DPO의 역-KL을 JS/forward-KL/α-divergence로 일반화 — divergence 선택이 "정렬 성능 vs 생성 다양성" 균형을 직접 결정.
- Soft Preference Learning[Slocum+ **2511.08594v1**, 2025-10-29 제출 Ⓐ]: KL 정규화기가 다수-의견 과대대표·다양성 희생의 주범 — KL의 entropy/cross-entropy 항 분리로 다양성 미세 제어, "temperature scaling의 Pareto 개선", repeated-sampling 과제 정확도↑.
- RLVR 판: DPH-RL[Li+ **2509.07430v4**, 2026-03-03 Ⓐ] — Pass@1↑·Pass@k↓ 역설의 원인 = mode-seeking divergence; forward-KL/JS 류 mass-covering f-divergence를 rehearsal로 써 Pass@1과 Pass@k 동시 개선(수학·SQL). 개념 동일 계열로 conceptual diversity 감소 실증[Murthy+ **2411.04427v3** Ⓐ — 정렬 모델이 instruct-만 모델보다 개념 다양성↓, 인간 수준엔 전 모델 미달].

**다양성-보존 post-training (채택 후보 구체안):**
- **GEM**[Li+ **2408.16673v2**, ICLR 2025 Ⓐ]: SFT의 CE를 **엔트로피-정규화 역-KL 게임-이론 정식화**로 교체 — downstream 동등·다양성↑, **test-time scaling(BoN)에서 chat·코드 이득으로 현금화**(초록 기준; "up to 7점" 수치는 Ⓢ GitHub/리뷰 — 본문 표 재확인 후 인용). 망각 완화 부수효과.
- **DARLING**[Li+ **2509.02534**, Meta FAIR Ⓐ]: online RL에서 학습된 partition function으로 의미-다양성을 측정해 품질 보상과 **곱-합성** — 다양성 명시 최적화가 탐색을 촉진해 품질 자체도 ↑ (비검증·검증 과제 양쪽).
- (보너스, RQ6와 교차) **Price of Format**[Yun+ **2505.18949v1** Ⓐ]: 다양성 붕괴를 일으키는 건 정렬만이 아니라 **구조 토큰의 존재 자체** — 고온에서도 지속.

**RQ2 판정**: v3g 소견(33% 다양성 붕괴·oracle 불변)은 문헌이 3중으로 예측하는 현상(divergence 기제 + 데이터 typicality 기제 + 실측 canonical). **단 문헌의 어느 것도 "oracle은 그대로인데 intra-pool 다양성만 죽어 선별 headroom이 붕괴"를 우리만큼 깨끗히 분리하지 않음** — pool-autopsy C는 그 자체로 기여 후보. v4 사이클에 DivPO-식 쌍 선택(우리 채굴기에 1-규칙 추가: chosen의 풀-내 빈도 패널티) + GEM-식 SFT가 문헌-지지 처방.

### RQ3. 멀티-프롬프트 / 입력 섭동 앙상블

- **Multi-Prompt MBR**[Heineman, Dou, Xu **2407.15343v2**, EMNLP 2024 Ⓐ] — 우리 설계에 가장 직결. 프롬프트 뱅크에서 후보를 디코딩해 MBR로 앙상블; 개선의 귀속을 명시적으로 "단일 프롬프트보다 **더 다양하고 더 품질 높은 후보 공간** 추정"에 둠. 조건부 생성 과제 전반·복수 모델에서 일관 이득.
- 이론·대규모 실증 짝 = 2502.11027v4(RQ1) — 프롬프트 섭동 스타일을 diversity-fidelity 원리에서 *도출*하고 BoN 이득 입증. 이 두 편이 "AR8 슬롯의 프롬프트-축 다양화"의 직접 근거.
- 페르소나/시스템-프롬프트 섭동의 **선별-이득** 직접 증거는 얇음: 페르소나 프롬프팅이 합성 데이터 lexical 다양성을 올린다는 측정 연구(2505.17390 Ⓢ)·아이디어 다양성 향상(Ⓢ)은 있으나 "페르소나 풀 → MBR/BoN 이득" 형태의 검증된 논문은 미발견 — **미검증 리드**로 분류. 실무적으론 multi-prompt MBR이 같은 자리를 이미 커버.
- 주의(우리 데이터와의 정합): 우리 도구-DAG 출력은 정답 공간이 좁음(개방형 생성과 다름) — 프롬프트 섭동이 *정답 모드*가 아니라 *형식 변주*만 늘릴 위험. ⇒ prereg에 "다양성↑ ∧ oracle↑ 동시"를 관문으로(다양성만 올라가면 노이즈-측).

### RQ4. 체크포인트/어댑터-수준 다양성

- LoRA 앙상블의 원전[Wang, Aitchison, Rudolph **2310.00035v2** Ⓐ]: 어댑터 다수 유지 비용 ≈ 단일 모델 — 정확도·캘리브레이션 동시 개선. 멀티-LoRA 1-서버 배포(우리 H6 ≈ 추가비 0)의 문헌 대응물.
- 멤버 거동[Balabanov, Linander **2402.12264v2** Ⓐ]: 앙상블 멤버들이 "확신-있되 서로-다른" 예측 — 과적합 영역에서도 사전 지식 보존 관찰. = "diverse-but-individually-strong"의 UQ-측 실증.
- **명시적 어댑터-다양성 유도** 선례[Zhai+ **2401.00243v1** Ⓐ]: reward LoRA 앙상블에서 **LoRA 행렬 연접의 nuclear norm 최대화**로 다양성 정규화(목적은 RLHF 과최적화 방지용 UQ). ⇒ "풀을 위해 어댑터를 *서로 다르게* 학습"하는 직접 기술이 존재 — H6의 자연 발생 다양성(데이터/목적/백본 3축)을 의도 설계로 승격 가능.
- **갭(정직)**: "LoRA/체크포인트 풀을 *MBR 후보 생성기*로 쓰고 선별 이득을 측정"한 2024–26 논문은 본 조사에서 미발견 — multi-prompt MBR(프롬프트-축)과 LoRA-앙상블(UQ-축)의 교집합이 비어 있음. **우리 H6+prop-MBR+Reviewer 스택(+10.3pp 공식)은 이 갭에 정확히 앉아 있음** = 기여 주장 가능 좌표 (선행 부정 단정은 금지·추가 적대검증 1회 권장).
- 무엇이 어댑터-풀을 "다양하되 개별-강"하게 만드나 — 문헌 종합: seed-만 차이는 약하고(앙상블 고전 소견), **데이터 샤드·목적함수·정렬 단계 차이**가 기능적 다양성의 본체[2310.06452의 SFT↔RLHF 다양성 차·2402.12264의 과적합-영역 보존성]. 우리 실측(H6=LODO 데이터-축+백본-축 어댑터가 이득 운반, AR8 같은-어댑터 샘플은 운반 못 함)과 합치.

### RQ5. Diffusion / 비-AR 제안기

- **DiG-Plan**[Li, Zhang **2606.05728v1**, 2026-06-04 제출, IJCAI-ECAI 2026 Ⓐ]: 도구-그래프 계획에서 AR 디코딩의 early-commitment를 지적, diffusion 제안기(도구-셋 반복 정제)+AR 정제기(의존성 예측) 2-단 — **masked denoising이 matched compute에서 Pass@10 커버리지 0.320→0.943**, TaskBench 상대 +10%, API-Bank 일반화. 우리 설계서가 이미 인용한 그 논문; abs 재검증 완료(수치는 초록 Ⓐ — 단 그들 프로토콜 "TaskBench-23 501" 비표준, 수치 이식 금지 유지).
- 모델 자체: **Dream 7B**[Ye+ **2508.15487v1** Ⓐ — AR-초기화·토큰-수준 적응 노이즈 스케줄, 동급 AR 비견 + planning·arbitrary-order·infilling 강점] / **LLaDA 8B**[Nie+ **2502.09992v3** Ⓐ — LLaMA3-8B 비견, reversal curse 해소 주장].
- **"상보적 오류 프로파일" 직접 증거는 부재**: 코드 생성 전수 실증[Li+ **2509.11252v2** Ⓐ, 9개 dLLM×4 벤치]도 성능 비견·길이 외삽 우위까지만 — AR vs diffusion의 오류-모드 직교성을 측정한 논문, **AR+diffusion 후보를 한 풀에 섞어 선별한 논문 모두 미발견**(DiG-Plan은 직렬 파이프라인이지 혼합-풀 선별이 아님). ⇒ P-D1의 풀-분해 설계(oracle(AR8+D4)−oracle(AR8))는 문헌 공백을 직접 측정하는 실험 — 기여 좌표 둘째.
- 운영 리스크의 문헌 답: **dLLM CFG-제약 디코딩이 가능해짐**[Mündler, Dekoninck, Vechev **2508.10111v1** Ⓐ — additive infilling 정식화, C++/JSON에서 구문 거의-완전·기능 보존, 오버헤드 실용]. P-D0 최대 리스크(형식 준수)의 보강 수단 — 단 위 GAD 소견상 제약이 diffusion 분포도 왜곡할 수 있음을 같은 prereg로 감시(raw+snap 변형 병기 유지).

### RQ6. 제약 디코딩 × 다양성 — "guided가 다양성을 깎는가": 깎는다 (수렴 증거 3계열)

1. **분포 왜곡 기제**[GAD: Park+ **2405.21047v3**, NeurIPS 2024 Ⓐ]: 마스크-기반 GCD는 "문법적이지만 LLM 우도에 비례하지 않는" 출력을 만든다 — 저-우도 prefix가 문법 제약 아래 과대 선택되거나, 분포가 조건부-진분포에서 벗어남. 처방 = **ASAp**(샘플 이력으로 미래 문법성 기대를 근사해 LLM의 *문법-조건부* 분포에 점근 정렬). 다양성 함의: 제약-조건부 분포 자체를 보존하는 샘플링이 가능 — "constrained sampling을 분포-정렬형으로" 가 곧 완화책.
2. **구조-토큰의 다양성 붕괴**[Price of Format: Yun+ **2505.18949v1** Ⓐ]: 다양성을 지배하는 건 구조 토큰의 유무이며 고온 샘플링으로도 안 풀림; 형식 일관성은 구조 과제(GSM8K·IFEval)에만 이득, 다양성은 전 도메인 억제.
3. **형식 제약의 품질 비용**[Tam+ **2408.02442** Ⓐ, EMNLP 2024 Industry]: JSON-mode 등 강한 제약이 추론 성능을 깎음(분류는 도움) — 품질 채널까지 포함한 비용. 반론·후속(엄밀 측정) 계열: JSONSchemaBench[Geng+ **2501.10868v3** Ⓐ — 6개 프레임워크 효율/커버리지/품질 평가; **다양성 차원은 초록에 없음** — 표-수준 확인 전 인용 보류).

**우리 세팅 번역**: guided의 +1.3–1.4(단일샷, N1)는 진성이지만 그것은 *형식-실패 제거* 채널이고, K-샘플 풀에서는 같은 마스킹이 후보들을 grammar-내 고-우도 모드로 몰아 **intra-pool 다양성을 추가 압축**할 개연성(기제 1+2). 현 AR8=guided 샘플이라는 점은 P-D1 공정성 한계로 이미 박제되어 있는데, 문헌은 이를 "풀-측 레버"로 격상시킨다: **풀은 unguided+validity-필터+snap(우리 v0 분업: 검증=거부권·선택=합의), 최종 출력만 guided** 구성이 문헌-정합 기본형. ASAp-류 분포-정렬 제약 샘플링은 vLLM 표준 스택 밖(구현 비용) — 차선으로 충분.

### RQ7. Quality-Diversity 프레이밍

- **QDAIF**[Bradley+ **2310.13032v4**, ICLR 2024 Ⓐ]: MAP-Elites의 변이·평가를 LM 피드백으로 — 품질 유지하며 지정 탐색-공간 커버리지↑(창작 도메인, 인간평가 정합). LLM을 QD의 in-context 변이 연산자로 쓰는 후속[**2404.15794** Ⓢ — abs 미검증, 리드].
- 학습-시점 QD-류: DQO[**2509.04784v3**, ICLR 2026 Ⓐ — DPP 행렬식 보상]·DARLING[2509.02534 Ⓐ] = "novelty+fitness 동시 보상"의 RL 구현.
- **우리 세팅 번역**: K=8–14 예산에서 인스턴스-당 MAP-Elites 아카이브는 과잉. QD의 쓸모는 **풀-설계 수준의 사고틀** — 어댑터 포트폴리오를 행동-축(도메인×목적×백본) 그리드의 elite들로 보고, "새 멤버는 기존 niche와 겹치지 않으면서 niche-내 최강일 때만 편입"(§3.7d ND 교훈 "다양성-기여 검증된 proposer만 편입"의 형식화). 이 관점에서 H6은 자연 발생 MAP-Elites 아카이브.

---

## 3. 우리 두 후보 수에 대한 문헌 판정

### (a) Dream-7B diffusion 제안기 — **조건부 지지 (강등 유지가 문헌-정합)**
- **지지**: 도메인-일치 직접 증거 DiG-Plan(커버리지 0.32→0.94)·dLLM 성숙도(Dream/LLaDA 동급 AR 비견)·seed-conditioning/multi-token 계열의 이론적 응원[2504.15266v4]·형식 리스크 완화 수단 등장[2508.10111v1].
- **유보**: ①"상보적 오류 프로파일" 및 "AR+diffusion 혼합-풀 선별"의 발표된 증거 0 — 가설이지 정리 아님 ②우리 P-D(-1) 실측: 싼 이종성(멀티-LoRA)만으로 Δoracle +13.6 — 문헌의 어떤 결과도 "diffusion의 한계가치가 *이종-AR 풀 위에서*도 양(+)"을 보이지 않음(DiG-Plan 비교 대상은 단일-정책 AR 샘플링) ③LLM Monkeys·E6 교훈: 커버리지↑≠실현이득(선별 병목).
- **판정**: 사전등록 v2(Dream 한계가치 > 이종-AR 한계가치, AR8+H6 위에서 측정) **그대로 유지**가 문헌이 권하는 정확한 실험. 발사 우선순위는 1–6 레버(특히 unguided-풀·멀티-프롬프트) 소진 후. 양성이면 "혼합-풀 AR+diffusion 선별 첫 실증"의 기여 좌표.

### (b) 검증기 축 다양화 후 soft/multi-axis 집계 재시도 — **지지 (재시도 정당)**
- 이론 닻: VB/VF 분리정리[2502.12118v2 Ⓐ] — 검증-채널 이득의 전제가 base 분포 heterogeneity+anti-concentration. **풀 다양화는 정확히 이 전제를 만드는 조작** → 검증기-측 headroom이 다양화와 함께 커진다는 방향 예측(정식 대상은 파인튜닝 알고리즘 — test-time 사상은 확장 해석임을 §5.5 단서대로 명시).
- 실증 정합: 합의-류 집계는 다양성 이득을 못 받음[2502.11027 caveat·LLM Monkeys plateau] ↔ 검증/BoN 채널은 받음 — 우리 SEL-4(Reviewer p(instr|plan), 합의와 직교) +0.81이 이 그림의 우리-측 실현. SEL-2(soft-approval) 음성은 graded 신호가 gmem 1개뿐이던 정보량 문제로 부검됐는데, 문헌의 다축 선례(QDAIF의 품질+다양성 2축·UP-RLHF의 reward+uncertainty 2축[2401.00243]·DMBR의 utility+diversity 합성[Jinnai+ **2401.05054v2** Ⓐ])는 **축이 2개 이상이고 서로 직교할 때** soft 합성이 작동함을 시사.
- **prereg-가능 형태**: 풀=unguided-다양화 후, z-합성 축 = {prop-MBR 합의, Reviewer 역-우도, proposer-prior(SEL-1), validity-등급(이진→graded 승격: snap-거리·dangling 수), (선택) 비용 ε-밴드} — 예측: 다양화-풀에서 soft 합성 > hard-filter+합의 (현행), 동질-풀에선 차이 없음(SEL-2 재현). 음성 시 "soft 집계는 축-수가 아니라 풀-다양성의 함수가 아니었다"로 기제 분리.

---

## 4. 서지

### 4.1 검증 완료 (전부 arXiv abs 페이지 직접 확인, 버전·확인일 2026-06-13/14)

| # | arXiv | 버전 | 제목 (축약) | 역할 |
|---|---|---|---|---|
| 1 | 2502.11027 | v4 (2025-12-19) | On the Effect of Sampling Diversity in Scaling LLM Inference | 다양성→BoN 인과·majority-vote caveat |
| 2 | 2407.01082 | v8 (2025-11-20) | Turning Up the Heat: Min-p Sampling | min-p 원 주장 |
| 3 | 2506.13681 | v2 (2025-06-19) | Min-p, Max Exaggeration | min-p 반박 |
| 4 | 2502.05234 | v2 (2025-06-16) | Optimizing Temperature for LMs with Multi-Sample Inference (TURN) | 멀티-샘플 온도 자동화 |
| 5 | 2407.21787 | v3 (2024-12-30) | Large Language Monkeys | coverage 스케일링·선별 plateau |
| 6 | 2210.15458 | v2 (2023-06-01) | Arithmetic Sampling | 병렬 다양 디코딩 (canonical, 2024 이전) |
| 7 | 2411.06251 | v2 (2025-04-27) | Quasi-random Multi-Sample Inference | arithmetic→SC/MBR 이득 재검증 |
| 8 | 2410.09038 | (v2 HTML 확인) | SimpleStrat | 층화 > 온도 |
| 9 | 2509.17570 | v1 (2025-09-22) | Asking a LM for Diverse Responses | 열거/반복 > 독립 샘플링 |
| 10 | 2510.01171 | v3 (2025-10-10) | Verbalized Sampling | typicality bias·훈련-無 다양화 |
| 11 | 2504.15266 | v4 (2025-08-28) | Roll the dice & look before you leap | seed-conditioning·multi-token 창의성 |
| 12 | 2310.06452 | v3 (2024-02-19) | Understanding the Effects of RLHF (Kirk+) | RLHF 다양성 감소 canonical |
| 13 | 2408.16673 | v2 (2025-04-05) | Preserving Diversity in SFT (GEM) | 엔트로피-정규화 SFT→BoN 이득 |
| 14 | 2501.18101 | v4 (2025-05-22) | Diverse Preference Optimization (DivPO) | 다양성-보존 DPO 쌍 선택 |
| 15 | 2511.08594 | v1 (2025-10-29) | Diverse Preference Learning (Soft Preference Learning) | KL 분해 기제·제어 |
| 16 | 2509.07430 | v4 (2026-03-03) | The Choice of Divergence (DPH-RL) | RLVR pass@k 붕괴·f-divergence |
| 17 | 2509.02534 | (v1 계열) | DARLING (Meta FAIR) | 다양성×품질 online RL |
| 18 | 2309.16240 | v1 (2023-09-28) | f-DPO | divergence 선택↔다양성 (canonical, 2024 이전) |
| 19 | 2411.04427 | v3 (2025-07-07) | One fish, two fish (conceptual diversity) | 정렬→개념 다양성↓ |
| 20 | 2407.15343 | v2 (2024-10-03) | Multi-Prompt MBR (EMNLP 2024) | 프롬프트-뱅크→MBR 직접 근거 |
| 21 | 2310.00035 | v2 (2023-10-04) | LoRA ensembles for LLM fine-tuning | 어댑터 앙상블 원전 |
| 22 | 2402.12264 | v2 (2025-05-20) | UQ in fine-tuned LLMs using LoRA ensembles | 멤버 diverse-but-confident |
| 23 | 2401.00243 | v1 (2023-12-30) | UP-RLHF (diverse reward LoRA ensembles) | nuclear-norm 어댑터-다양성 정규화 |
| 24 | 2508.15487 | v1 (2025-08-21) | Dream 7B | diffusion 제안기 모델 |
| 25 | 2502.09992 | v3 (2025-10-18) | LLaDA (Large Language Diffusion Models) | 대안 dLLM |
| 26 | 2606.05728 | v1 (2026-06-04) | DiG-Plan (IJCAI-ECAI 2026) | diffusion 제안 커버리지 0.32→0.94 |
| 27 | 2509.11252 | v2 (2025-11-02) | Beyond Autoregression (dLLM 코드 실증) | 오류-프로파일 직접 증거 부재 확인 |
| 28 | 2508.10111 | v1 (2025-08-13) | Constrained Decoding of Diffusion LLMs with CFGs | Dream 형식-리스크 완화 |
| 29 | 2405.21047 | v3 (2025-12-12) | Grammar-Aligned Decoding (ASAp, NeurIPS 2024) | GCD 분포 왜곡·완화 |
| 30 | 2408.02442 | (v1 계열, EMNLP 2024 Ind.) | Let Me Speak Freely? | 형식 제약의 품질 비용 |
| 31 | 2505.18949 | v1 (2025-05-25) | The Price of Format: Diversity Collapse | 구조 토큰→다양성 붕괴 |
| 32 | 2501.10868 | v3 (2025-02-27) | JSONSchemaBench | 제약 프레임워크 평가(다양성 축 초록 無) |
| 33 | 2310.13032 | v4 (2023-12-07) | Quality-Diversity through AI Feedback (ICLR 2024) | QD×LLM 원전 |
| 34 | 2509.04784 | v3 (2026-03-02) | DQO: Post-training for Diverse High-Quality Responses (ICLR 2026) | DPP-보상 학습-시점 다양화 |
| 35 | 2502.12118 | v2 (2025-02-18) | Scaling Test-Time Compute Without Verification or RL is Suboptimal | VB/VF 분리정리 (FIELD_GAP §5.5 닻 재확인) |

### 4.2 미검증 리드 (abs 미확인 또는 Ⓢ-만 — 인용 전 1차 검증 필수)
- **D5P4** (2603.19146?) — diffusion 병렬 디코딩의 DPP 선택. 디코딩-시점 DPP의 유일 후보.
- **2404.15794** — LLMs as In-context AI Generators for Quality-Diversity (QD 변이 연산자).
- **2505.17390** — 페르소나 프롬프팅의 lexical 다양성 측정 (RQ3 페르소나-축).
- **2509.21791** — 구조화 출력 영향의 인과추론 정량화 (RQ6 보강).
- min-p "ICLR 2025 18위·oral" 등 수용 디테일 — Ⓢ.
- GEM "BoN 최대 +7점" 수치 — GitHub/리뷰 Ⓢ, 본문 표 확인 후 인용.
- DPP **디코딩-시점** 적용·"AR+diffusion 혼합-풀 선별"·"LoRA-풀을 MBR 후보 생성기로" — **3개 모두 발표 문헌 미발견(갭)**: 부정 단정 대신 "본 조사에서 미발견"으로만 기술.

### 4.3 우리 문서 좌표 (역링크)
- 실측 권위: `reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md` §8.9b–d / `scripts/distill/BENCH_PORTFOLIO_FRAMEWORK_DESIGN.md` §3.7d (NC 다양성 −33%·ND 상쇄 부검).
- 이론 닻: `scripts/distill/FIELD_GAP_LLM_VALUE_DESIGN.md` §5.5 (2502.12118 확장-해석 단서 포함).
- 사전등록: `scripts/distill/taskbench/TB_DIFFUSION_PROPOSER_DESIGN.md` (P-D0/P-D1/P-D2 v2) — 본 보고서 §3(a)가 우선순위 입력.
