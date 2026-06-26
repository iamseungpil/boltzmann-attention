# 탐색 내재화(search→weight) 선행연구 — 적대검증 서베이 + 북극성 아키텍처 근거

> 2026-06-03. deep-research #2 (103 에이전트·21 소스→86 주장→25 검증, 24 confirmed/1 killed).
> 동기: TBox/ABox 분리 + 그래프-가이드 자율 agent(gather-judge-act-until-resolved)를 **weight에 내재화**해 전이. 외부 탐색(run_scripted 오라클 37/48)을 내재화·전이가능하게.
> 짝문서: `RUNG1_V3_TREE_EVAL_LITREVIEW.md`(트리평가 학습). 권위본 결과=`reports/.../SOPBENCH_EXPERIMENT_RESULTS.md`.

## 1. 한 줄
**탐색을 weight에 내재화해 "교사 초과 일반화"가 나온다는 직접 증거 有(Searchformer 키스톤)** — 전부 math/puzzle·verifiable·from-scratch. **RISK 1(NL→마스킹트리 추론+전이)은 선행 전무 = 우리 고유 기여이자 최대 미검증 리스크.**

## 2. 검증된 핵심
| 클러스터 | 논문(arXiv/venue) | 결과 | 판정 |
|---|---|---|---|
| 외부탐색(제거대상) | ToT `2305.10601`(NeurIPS23)·RAP `2305.14992`(EMNLP23)·LATS `2310.04406`(ICML24)·GoT(AAAI24 29720)·DPTS `2502.16235`(ACL25) | CoT 4%→74%(ToT GoT24); RAP LLaMA-33B>GPT4-CoT +33%; LATS 92.7% HumanEval; **GoT 그래프가 트리보다 정확도+비용 둘 다 우위(>31%↓)** | ANALOGICAL(동기; 우리는 제거) |
| **★내재화 키스톤** | **Searchformer `2402.14083`(TMLR, Meta)** | A* trace 증류 + **shorter-trace bootstrapping(expert-iter)** → unseen Sokoban **93.7% 최적·교사보다 26.8% 적은 스텝** = **교사초과 일반화 직접증거**. 메커니즘=full trace 모방→자기 짧은 trace로 반복 FT 압축 | ✅ 키스톤(단 from-scratch enc-dec·maze; 2505.13775이 *메커니즘* 비판, 수치는 아님) |
| 깊이(RISK 2) | **TS-LLM `2309.17179`(ICML24)** | **학습 value function** 증류(PolicyImprove→Distill→Eval) → **depth-64**(ToT~10/RAP~7) | ◐ 적응책(단 RLHF 토큰트리=ANALOGICAL) |
| 학습신호 | ReST-MCTS* (NeurIPS24, `2406.03816`)·Math-Shepherd(ACL24 `2312.08935`)·AlphaLLM `2404.12253`(NeurIPS24) | **per-step process reward를 최종정답만으로 자동추론(human label 0)**; 탐색-trace self-train > ReST^EM/Self-Rewarding | ✅ 레시피(우리 완벽검증기와 정합) |
| 7B 자기진화 | rStar-Math `2501.04519`(ICML25) | 우월교사 없이 MCTS trace 자기진화 *가능* | ◐ (★"58.8→90 MATH"는 **기각 1-2, 인용금지**) |

## 3. 우리 3대 리스크 판정
| 리스크 | 판정 | 근거/함의 |
|---|---|---|
| RISK 2 깊이/조건수 | ◐ 적응책 | TS-LLM value-fn(depth-64) → partial-gather-state value 학습. ANALOGICAL. |
| RISK 3 OR>AND(grounded 평가) | ❌ 미해결 | 분리한 소스 0; OR-붕괴가 RISK1 누수인지 미지 → **직접 측정 필수** |
| RISK 1 NL→마스킹트리+전이 | ❌❌ 전무 | structure-inference 확정주장 0 = **headline novelty이자 최대 리스크** |

## 4. 새 내재화 학습법 — 인용 확보(정정)
- **① 탐색-trace 증류** → **Searchformer**(교사초과·압축 부트스트랩). SUPPORTED(analogical).
- **value-function 학습**(신규) → **TS-LLM**(depth-64). RISK 2.
- **expert-iteration**(SFT/DPO/RFT 외) → **ReST-MCTS*/AlphaLLM**(MCTS-trace 재학습 > filtered-SFT).
- **process-reward-from-outcome** → **Math-Shepherd/ReST-MCTS***(최종 gate-정답→스텝라벨). ★우리 결정론 evaluator로 즉시 가능.
- **② 반사실 grounding** + **③ 잠재트리 hard-EM(NL→트리)** → **선행 무 = 우리 고유**(③=headline).

## 5. 우리 설정용 새 방법 스케치 (문헌 근거)
- **A. cost-aware 탐색-trace 증류**(Searchformer+ReST-MCTS*): GT 시스템 위 cost-aware 탐색으로 (어느 leaf 게더·AND/OR 집계·백트랙) trace 생성 → inductive derivation(기록값 read-back)으로 증류 → **짧은 trace 부트스트랩**. 학습신호=최종 gate-정답(검증기)에서 스텝 reward 추론. RISK 2/3(깊이·OR 분기탐색)·교사초과. 반증최저비용=trace-distill vs clean-SFT를 조건수 2/4/6/8서 비교.
- **B. partial-gather value-fn 증류**(TS-LLM): v(게더상태)=최종 gate 성공확률 → gather 순서·early-stop(cost-aware)·깊이. 반증=value-guided vs 무 비교(depth-decay 곡선).
- **C. 구조-탐색-후-증류(RISK 1, headline·UNSUPPORTED)**: NL정책서 후보 AND/OR 파스 K개 탐색 → **GT gate-라벨과 일치하는 구조 채택(검증기)** → NL→승자구조 증류 → ABox swap 전이. = Searchformer의 *구조* 버전. 선행 무 → 1급 결과/반증 대상.

## 6. 북극성 아키텍처 검증
"그래프-가이드 gather-act-until-resolved → 내재화"(TBox=일반논리·탐색 / ABox=가이드룰 swap / 그래프=affordance검색 / 자율루프)는 **Searchformer 패러다임의 도메인확장**: 외부탐색(run_scripted/그래프)→trace 증류(Searchformer)→value-fn(TS-LLM)→전이(우리 고유 RISK1). 빌드경로 문헌상 정합. (상세=`EXPERIMENT_DESIGN` Rung3 + target-architecture.)

## 7. 정직한 한계
모든 내재화 결과가 **math/puzzle·verifiable·from-scratch/value-net** — **LoRA-7B tool-use AND/OR 제약트리 over 기록값은 없음.** RISK 1·3 미커버. Searchformer 26.8%·GoT 62%는 best-case. fast-moving(2023-25), 2025 비판(2505.13775)은 메커니즘 한정.

## 8. 소스(primary)
2305.10601·2305.14992·2310.04406·AAAI24-29720·2502.16235·2402.14083·2309.17179·2406.03816(ReST-MCTS*)·2312.08935·2404.12253·2501.04519·2404.03683·2405.14838·2407.06023. 기각: 2501.04519의 58.8→90 수치.

## 9. 재탐색 (2026-06-03 PM) — depth-recurrence (Universal/Looped Transformer)로 serial-depth 내재화 (deep-research #4, 104 에이전트·22소스·25주장, 21 confirmed/4 killed)
> 동기: CoT 토큰 대신 **weight-tied recurrence**로 serial 계산깊이를 얻어 트리/탐색 평가를 latent로 내재화 가능한가 + pretrained 7B retrofit 가능한가.
> **결론: 결정론 트리평가엔 looping이 이론적 정답이나, pretrained 7B latent-recurrence retrofit은 오늘 방법 無 → 실현경로=RELAY식 trace 증류(=우리 inductive/multi-call의 loop-aligned판). serial-depth가 우리 병목인지 선결 확인 필요.**

| 발견 | 논문 | 판정·함의 |
|---|---|---|
| ★**결정론 DAG/트리평가: loop ∝ 깊이, CoT ∝ 크기 (형식 분리)** | **Xu & Sato 2505.19245 "To CoT or To Loop?"** (Thm 4.7/4.4/5.6) | ✅ HIGH. **우리 AND/OR/chain precond 트리(bounded-fan-in 결정론 DAG)=looping 우위 영역.** "recurrent induced bottom-up GoT" 이론 검증. stochastic 샘플링은 CoT 우위(looping FPAUS 불가) → **평가=loop, 탐색=CoT**(통합 아님; "unified" 0-3 기각) |
| recurrence가 TC0 초과(serial depth=iter) | **UT 1807.03819**(Turing-complete, 가정下)·**Giannou 2301.13196**(looped 13-layer=프로그래머블 컴퓨터) | ✅ HIGH. 단 **constructive 존재증명(hand-designed weights)·학습/pretrained 결합법 불명** |
| adaptive-step looping이 length/depth 일반화↑ | **Fan et al. ICLR25 2409.15647** (n-RASP-L) | ✅ HIGH. 단 (1)iterative 해법 (2)step 감독 (3)정지규칙 필요 |
| ★**retrofit/증류 경로** | **RELAY 2502.08482**: loop-iter↔CoT-step 정렬+중간감독→looped로 학습길이초과 CoT 생성→**AR 모델 SFT** | ✅ HIGH. = recurrence를 *trace 생성기*로. ⚠️**소형 from-scratch·합성과제만, pretrained 7B 미검증** |
| ❌**latent recurrence retrofit 불가** | **Huginn-3.5B 2502.05171**: from-scratch(800B토큰)·Prelude/Loop/Coda=vanilla retrofit 비호환. **latent<CoT**(GSM8K 5% vs 25-38%, Lu 2507.02199 probing=해석가능 latent CoT 거의無) | ✅ HIGH. **LoRA-on-7B 목표 결정적 부정** |
| recurrence 단독≠일반화 | UT+ACT Sudoku 6-8% vs TRM 87.4% (2604.21999/2510.04871) | ◐ MED(과일반화; "올바른 학습 필요" clause만 타당. TRM도 recursive-depth) |

**기각(신뢰금지)**: "CoT=loop 단일메커니즘"(0-3)·"memory token 필수"(0-3)·"Huginn 50B급 능력"(1-2, FLOPs일 뿐)·"recurrence는 CoT데이터 불요 이점"(0-3).
**gap**: ①pretrained 7B 레이어 post-hoc looping(LoRA loop-stable)=無 ②RELAY @7B·현실과제=無 ③**우리 트리가 serial-depth 병목인지 자체가 미확인**(얕음 2-7조건; 우리 진단=gather/도구선택 병목 → recurrence가 비-문제 풀 위험).
**실행 함의**: 7B 내부 looping 대신 ①**call-레벨 loop**(harness가 frontier 한 층씩 재호출=손수 만든 looped TF, 즉시 구현가능=명시적 multi-call recurrent bottom-up) + ②RELAY식으로 단일-pass 증류. **선결: BOTH 조건수/깊이 분해**(평탄→recurrence 불필요 / decay→투자가치).
소스: 2505.19245·1807.03819·2301.13196·2409.15647·2502.08482·2502.05171·2507.02199·2604.21999·2510.04871·2509.25239.
