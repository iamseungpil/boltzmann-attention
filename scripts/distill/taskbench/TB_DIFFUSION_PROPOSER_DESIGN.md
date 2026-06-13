# 이종 제안기(diffusion) × 결정론 선별 — 설계 (detail, 2026-06-12)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**에서 각 문서의 역할·상태 확인; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> 동기 = E6 확정 (TB결과 §8.8): best-stack(dpo2+guided) 정책은 K=8 샘플이 수렴 → 선별 갭 +1.4뿐.
> 잔여 L6는 **수렴 정책의 샘플 분포 밖** ⇒ 처방 = 제안 다양성의 인위적 복원. 외부 근거 = DiG-Plan.
> ⚠️**수치 정정 (2026-06-14 정독, `research_digplan_deepread_2026_06_14.md`)**: 기존에 "AR Pass@10 0.32
> vs diffusion 0.94 = TaskBench-23 501"로 기록했으나 **틀림** — 0.32/0.94는 **합성 토이**(2층·128dim·
> 23bit 벡터, 논문 Table 1)이고 TaskBench가 아님. **실제 TaskBench 이득은 ~10% 상대**(ToolF1 0.661→0.729·
> held-out Oracle@10 0.735→0.787)이며 **단일샷 ToolF1은 무승부**(0.355 vs 0.349). 0.32 vs 0.94는
> greedy(T=0, Pass@10≡Pass@1) AR ↔ 확률 diffusion 비교라 **모델클래스 아닌 샘플링-엔트로피 측정** = tilt.
> 기제(early-commitment 회피)만 차용·수치 이식 금지 (인용위생 — 허위 레퍼런스 제재 규율 준수).

## 1. 가설·프레임 정합
- **가설 H-D**: 이종(비-AR) 제안기를 풀에 섞으면 **풀-oracle이 상승**한다 (다르게 틀리는 제안 → 정답 구조가 풀에 들어올 확률↑). 선별은 기존 결정론 게이트(R6) — 마스킹 아닌 선별이라 GAD 왜곡 무관.
- §10.5 정합: R6(K-제안+검증-선별)의 **제안자 슬롯만 교체** — propose-then-gate 구조 불변. A1-A5 어댑터 재사용(스키마·gold·평가기 동일).

## 2. 후보 제안기 (공개 가중치, 우리 GPU 적합)
| 모델 | 크기 | 비고 |
|---|---|---|
| **Dream-7B** (`Dream-org/Dream-v0-Instruct-7B`) | 7B, bf16 ~16GB | DiG-Plan이 쓴 라인. HF 공개. 자체 diffusion generate 루프(vllm 불가) |
| LLaDA-8B-Instruct (`GSAI-ML/LLaDA-8B-Instruct`) | 8B | 대안/2차 |

## 3. 실험 단계 (사전등록 — 2026-06-12 리뷰 반영 v2: P-D(-1) 신설·분해행·ⓑ전제조건·P-D2 채택기준)
**★P-D(-1) 이종-AR 풀 census (zero-GPU, P-D0 선행 — 메타규칙 "GPU 전 zero-cost 진단")**:
  디스크의 기존 sub500 MM 예측으로 "이종이면 되는가"를 공짜로 분리. 풀(사전등록):
  AR8 = `tb_dpo2g_mmk0-7`(E6 동일) / **H6** = {qwen3b, qwen14b, qwen3_4b, qwen3_14b, tb_lodo_hf, tb_lodo_daily}
  (family×size×학습변형 3축 이종; id 겹침 486-500/500 실측). 행: oracle(AR8) [E6 재현 통제] /
  oracle(AR8+H6) / oracle(AR4=k0-3 고정) / oracle(AR4+H6).
  - **판정**: Δhetero=oracle(AR8+H6)−oracle(AR8) **≥+2** → 싼-이종성으로 충분 ⇒ P-D1 성공 기준 상향
    ("Dream 한계가치 > 이종-AR 한계가치"; 배포 2-모델 비용 정당화 부담) / **<+2** → AR끼리 닮게 틀림의
    1차 증거 ⇒ H-D 본형(비-AR 기제 필요) 강화. ⚠️해석 가드: H6=greedy 단일샘플=싼-이종성의 *하한*
    (temp0.8 K-샘플 이종-AR 아님) — 음성이어도 "AR 다양성 한계 확정"으로 과대해석 금지.
  - **✅실행·판정 완료 (2026-06-12, 결과 권위 = TB결과 §8.9)**: **Δhetero=+13.6**(0.720→0.856, H6단독 0.847,
    인플레이션 가설 기각 census 통과) ⇒ 싼-이종성으로 coverage 해결. **동시 발견 = 혼합 풀에서 게이트 붕괴**
    (v0/v1 0.71→0.52-0.54 < mean) ⇒ **binding constraint는 선별기로 이동, P-D0/P-D1은 조건부 강등**
    (Dream 한계가치는 AR+H6 위에서만 의미·"싼 대안 대비 순이득" 입증 부담). H6=같은 base LoRA들=멀티-LoRA
    1서버 ≈ 추가비 0. 다음 = 이종-풀 robust 선별기 설계(신규 detail 문서로, 마스터 §7 경유).
**P-D0 스모크 (반나절, GPU 1장)**: Dream-7B로 MM 프롬프트 50개 × K=4 생성 →
  ①**형식 준수 관문(이중)**: JSON 파싱율 ≥0.5 **∧ snap-후 valid_frac ≥0.8**(게이트가 실제 소비 가능한
  비율 — 파싱돼도 구조 전파손 케이스 차단). 미달이면 템플릿/few-shot 보강 1회 재시도, 그래도 미달 시 LLaDA로 교체.
  ②다양성 측정: distinct-plan율·노드셋 Jaccard.
**P-D1 본 측정 (1일, GPU 1장)**: MM sub500 × K_d=4 생성 → 풀 분해 4행(`tb_kgate_heldout.py`, 동일 id셋):
  - **oracle(AR4=k0-3) / oracle(AR4+D4) / oracle(AR8) / oracle(AR8+D4)** + Dream-only oracle
  - **검정(사전등록)**: Dream 한계가치 = oracle(AR4+D4)−oracle(AR4) vs AR 한계가치 = oracle(AR8)−oracle(AR4).
    ⓐ Dream 한계가치 > AR 한계가치 ∧ oracle(AR8+D4)−oracle(AR8) ≥ +2 (P-D(-1)이 Δhetero≥2면 "> 이종-AR 한계가치"도 요구)
      = H-D 채택 → P-D2. **+2 임계는 paired bootstrap 95% CI(id-단위 resample) 병기** — CI가 0 걸치면 보류.
    ⓑ 상승 없음 = 기각. **단 (iii)"diffusion 다양성이 표준 프로토콜로 전이 안 됨" 해석은 전제 2개 충족 시만**:
      P-D0 형식 관문 통과 ∧ Dream-only oracle ≥ 0.7×oracle(AR8). 미달 → "형식/모델 한계로 측정불가" 강등
      (형식 사고가 음성결과로 둔갑 금지 — LODO 직렬화-범인 함정 동형).
  - 선별 후 공식 edge(tb_build_eval)도 병기 — census-식과 분리 보고.
  - 공정성 한계(명시): unguided AR K-샘플 디스크에 없음(재생성=GPU 비용) → AR8=guided. 비대칭은 Dream에
    불리한 방향=ⓐ에 보수적이므로 채택 결론엔 안전, ⓑ 해석에만 위 전제조건으로 방어.
**P-D2 (조건부, ⓐ시)**: 혼합-풀 + 게이트(v1+) 선별의 best-stack 대비 **공식 edge-F1 순이득 ≥ +1 = 채택**
  (E6 교훈: oracle↑≠실현이득 — 현 게이트 회수 18-22.6%라 oracle+2≈실현+0.4뿐; 게이트가 AR 오류형태에
  튜닝돼 diffusion 제안을 더 못 고를 위험 포함). 미달 = "oracle-only 호기심" 분류. 채택 시 패키지 보고에
  **배포 비용 열(제안기 2-모델 = 추론비 ~2×) 병기** — {소형·저비용} 주장과 충돌 방지.

### ★P-D0 실행 부검 (2026-06-14 전수 궤적조사 — `dream_p_d0/dream_k{0..3}.json`, 사용자 발주 "엄격 원인분석")
**판정: parse_rate=0/41=0% 는 diffusion 음성으로 *해석 불가* — 설계서 line 47-49 가드레일("형식 사고가 음성결과로 둔갑 금지") 발동 확정. 두 교란(런 미완 + seed-구동 디코드 붕괴)을 제거하기 전 형식게이트 수치 무의미.**
- **표본**: 11/50 레코드만 처리(handoff "~21:40 timeout 종료" — N=50 미달) × k4 = 41샘플. 인자 = **defaults**(`steps=512 < max_new_tokens=768`·`temp=0.8`·`top_p=0.95`·`alg=entropy`·`alg_temp=0.0`; ps 실측).
- **3 실패모드 (전수 census)**: ⓐ**붕괴 ≤17자 = 28/41(68%)**(empty 15 + 스텁 `{"task_steps": ["` 13). ⓑ**near-complete ≥100자 = 9/41(22%)**. ⓒ partial 4.
- **★결정적 증거 — 붕괴는 레코드 난이도 아닌 *seed 확률성*에 지배**: 같은 레코드가 k에 따라 **0↔141로 요동**(10074769: k0=141·k1-3=0 / 10010819: k1=112·나머지 0 / 10076784: 141·0·0·143). **4/11 레코드가 한 입력에서 붕괴와 near-complete를 동시 산출**. `n_tools`도 무상관(nt=1이 전부붕괴[10090642 all-0]·전부성공[10103534 all~125] 양쪽). ⇒ ⓐ붕괴 = **디코드 불안정 artifact**(능력·난이도 아님). 1순위 기제 = **steps(512) < max_new_tokens(768)** 과소-denoise로 mask 잔류 → `skip_special_tokens`가 제거 → 빈/스텁 디코드; temp 0.8 가중.
- **★진짜 신호 — diffusion은 *내용*은 되고 *직렬화*가 깨짐**: ⓑ 9건 전부 **올바른 도구**(Audio-to-Video·Topic Generator·Audio Splicer·Text-to-Image) + **지시-정합 인자**(example.wav/mp4·environmental conservation) 선택, 단 **국소 괄호/따옴표 손상**(`]]]`·`]"}`·`,}]}`·stray `"`)으로 parse 실패. = **diffusion any-order 직렬화 약점**(닫는 토큰이 내용 확정 전 배치) = §3c DiG-Plan(diffusion-only edge 0.128)·§3d A3 예측과 정합. **⇒ AR-refiner 하이브리드(§3b·사용자 제안)가 정공임을 실측이 재확인**; parse_rate 단독 게이트는 *planning을 serialization과 혼동*해 diffusion을 과소평가.
- **clean 재실행 config(교란 제거)**: ①**steps = max_new_tokens**(full denoise·mask 잔류 0) ②**temp 0(형식게이트는 결정론; 다양성은 P-D1 별건)** ③**N=50 전수** ④**mask-잔류/미충전 위치 카운터를 로깅에 추가**(디코드 완결성 확증) ⑤alg_temp·Dream 권장 preset 검증. 재실행 전엔 형식게이트 PASS/FAIL 판정 보류.

> 인용위생 체크박스: DiG-Plan(2606.05728) 1차 검증 = 프로토콜 디테일(TaskBench-23 501, Pass@10 수치) 원문 확인됨
> · 논문 본문 인용 전 R8 절차(버전 명시·수치 재검증) 필수, 수치 이식 금지 유지.

## 3b. ★v3 승격 설계 (2026-06-14 — §3.7d 다양성 부검 후 "조건부 강등" 해제 경로)
**승격 근거(실측 3건, PORTFOLIO §3.7d)**: ①v3g 부검 — oracle 동일(0.896=0.895)·AR8 내부다양성 −33%만으로 선별이득 붕괴 = **AR 자기-샘플 다양성은 정책이 강해질수록 고갈** ②대형 AR 단일샷 추가 무이득(중복) ③H6(싼-이종성)은 이미 풀에 소진. ⇒ AR-밖 생성기제 = 풀에 남은 마지막 다양성 공급원 후보. **선결 게이트 = SELECTOR_DESIGN 큐 ⑸(다양성-생성 실험)가 AR-내 천장을 먼저 확정** — temp/prompt 변주로 닿는 다양성이면 diffusion 불요(비용 우위).

**단계 갱신 (P-D1/2를 현-최적 풀·현-최적 선별기 기준으로 재정의)**:
- **P-D0 (불변 + 완화책 1건)**: §3 형식 이중관문 그대로 (파싱 ≥0.5 ∧ snap-후 valid_frac ≥0.8; 실패 시 few-shot 보강 1회→LLaDA 교체). `tb_diffusion_sample.py` 기존. **신규 완화 옵션(딥리서치 2026-06-14)**: dLLM용 CFG-제약 디코딩 존재(`2508.10111` — diffusion에도 문법 제약 가능) — 형식 관문 실패 시 LLaDA 교체 전 시도. 단 "diffusion 한계가치 > AR8+H6" 채택 기준은 불변(딥리서치도 동일 권고 — AR+diffusion 혼합-풀 선별의 발표 증거 부재 = 우리가 하면 첫 실측).
- **P-D1' (혼합-풀 census, zero-eval)**: 현-최적 풀(dpo2g-AR8+H6, oracle 0.896) + D4 — 판정 3행:
  ⓐ oracle Δ = oracle(+D4)−0.896 (paired bootstrap CI)
  ⓑ **★unique-correct census**: D-후보만 정답(edge-F1 1위가 D이며 전 AR/H 후보 < 0.5)인 id 수 — **직교 기여의 정밀 척도** (oracle Δ보다 민감; ND 교훈 = 중복 proposer는 oracle도 못 올림)
  ⓒ 풀 다양성 Δ (평균 쌍별 1-F1) + D↔AR 평균 거리 (이종성 정량)
  **채택 기준**: ⓐ CI>0 ∧ ⓑ ≥10 ids → P-D2. ⓑ<10 = "중복-이종" 기각(ND와 동일 분류).
- **P-D2' (선별 실현, 2-arm 분해 — 신규)**: 혼합-풀 위 **P-D2a = SEL-1만 / P-D2b = SEL-1+SEL-4**.
  **사전등록 예측: Δ(P-D2b) > Δ(P-D2a)** — D-후보는 구조적으로 소수파라 합의(MBR)가 못 고르고, Reviewer(p(instr|plan))가 소수-정답 구제 채널 (SEL-4 +0.81pp 기제의 직접 시험). 채택 = 공식 link F1 ≥ **68.03+1.0**. 배포 비용 열(2-모델 추론비) 병기 의무 유지.
- **★P-D-alt (신설 — 싼 아키텍처-이종성 대조군, P-D0 실패 시 1순위 폴백)**: **cross-family AR + guided** (Llama-3.1-8B-Instruct 또는 Mistral-7B-Instruct + 동일 JSON 스키마 guided). 근거: guided가 형식을 균일화하므로 family-혼합의 형식 리스크 소거; H6은 전부 Qwen-family = family 축 미개척. vLLM 표준 서빙(Dream 대비 인프라 비용 ~0). 판정 = P-D1'와 동일 3행 — **Dream vs cross-family의 unique-correct 비교가 "비-AR 기제 필요성"의 최종 분리 실험** (cross-family로 충분하면 diffusion 기각·논문엔 "아키텍처-이종성이면 충분" 더 강한 결론).

## 3c. ★DiG-Plan 정독 + 수학적 우위 분석 (2026-06-14, `research_digplan_deepread_2026_06_14.md`)
- **DiG-Plan = 실재** (arXiv:2606.05728 v1, 2026-06-04, Yansi Li & Zhuosheng Zhang, "Mitigating Early Commitment for Tool-Graph Planning via Diffusion Guidance"). **최종 시스템 = diffusion-proposer + AR-refiner 하이브리드**(diffusion-only는 edge ToolF1 0.128 처참 — 저자 자인). ★**= 사용자 제안("형식=AR·다양성=diffusion")이 논문의 최종 구조와 일치** → AR-refiner 경로(§3b P-D 형식구제)가 정공법임을 외부 확증.
- **수학적 우위 — 증명가능 부분(조건부)**: ①**고정순서 AR은 order-invariant(집합) 타깃에 ≥0 KL 페널티** [XLNet·ARDM] — 도구-SET·DAG는 부분순서라 좌→우 factorization이 가짜 순서비용 부과 ②**AR 오류는 최대 quadratic 누적** T·ε≤R≤T²·ε [Arora `2204.01171`] — early-commitment 연쇄오류. **단 정직한 한계(증명가능)**: AR·diffusion 둘 다 joint의 universal approximator → "diffusion 항상 승"은 **거짓**. 이득은 **모델 클래스가 아니라 디코딩 regime(Pass@k spread)** — DiG-Plan 단일샷 무승부(0.355 vs 0.349)가 확증. ③우리 VB/VF 정리(`2502.12118`): 선별 이득은 풀 heterogeneity 전제 — **"diffusion이 heterogeneity를 높이는가"는 경험적 = P-D가 측정할 것**.
- **P-D 프로토콜 갱신(정독 반영)**: ⓐ**비교 baseline = hot-T AR**(temp>0·다양 — greedy 금지, 논문 tilt 재현 회피) ⓑ**주 지표 = Δheterogeneity + unique-correct(D-oracle)**(단일샷 우위 아님 — 기대값을 coverage/tool-set-recall로 하향, §8.9h 곱-부검 정합) ⓒ채택 시 **AR-refiner 하이브리드**(diffusion 골격→AR 형식화, 사용자 제안=DiG-Plan 구조). 기대 = "modest D-oracle"(논문 실제 +10% 상대), 단일샷 대박 아님.

## 3d. ★비-diffusion 대안 — 아키텍처 서베이 (2026-06-14, `research_arch_planning_survey_2026_06_14.md`)
diffusion이 "AR-밖 다양성"의 유일 후보가 아님. 검증된 대안 3종(전부 1차 검증):
- **★A3 any-order AR (`2601.13228`)** — ⚠️**정독 정정 (relwork_arch 2026-06-14)**: A3-8B는 **표준 좌→우 AR에 크게 짐**(TriviaQA 19.4 vs LLaMA-3.1-8B 52.1)·"diffusion 필적"은 **2B vs 65B 토큰 효율-정규화 한정**(동일-예산 비교 행 없음·약한 diffusion baseline). ⇒ **인용 형태 = "any-order 생성이 AR 패러다임 내 달성가능·동급-스케일 diffusion과 (데이터 소량으로) 경쟁" framing 닻으로만** — "AR이 diffusion 이긴다"·"강한 생성기" 주장 **금지**. 우리 7B 스케일선 강한 생성기로 부적합. diffusion-not-required는 *framing*으로 유지하되 P-D 대체 후보로는 약함.
- **grammar-constrained decoding (XGrammar `2411.15100`)**: retraining-free·vLLM 호환 = **우리가 이미 쓰는 guided의 정식화**. A2 출력 스키마 층으로 K-샘플 풀의 "valid JSON-DAG(=right의 하한)" 보장 — diffusion 형식 리스크(P-D0)의 정공 대안.
- **Stream-of-Search trace-distill (`2404.03683`)**: 검증기로 비용-인지 탐색 trace(실패·백트랙 포함) 생성→SFT, teacher-exceeding = §3.10 빌드 경로 일치.
- **회의(게이트)**: "Transformers Struggle to Learn to Search" (`2412.04703`) — 큰 그래프는 파라미터↑로 안 풀림 = search-internalization 라인 전체 천장 경고. 함정(채택 금지): Coconut/latent-CoT(retrofit 불가·math서 짐), pause/filler 토큰(+1%p), insertion/Levenshtein 재구현.
- **종합 함의(P-D 우선순위 재평가)**: diffusion 라인은 ①형식 리스크(P-D0) ②인프라 비용(vLLM 비호환) ③DiG-Plan 실제 이득 ~10%(§3c)에 더해, **④A3 any-order AR이라는 더 싼 대안 존재**. ⇒ **P-D0 형식게이트 결과 + P-D-alt(cross-family AR) + A3-닻을 종합 후 착수 판단** — diffusion 자체보다 "AR-밖 다양성이 D-oracle>0인가"가 본질이고, A3가 그 답을 더 싸게 줄 수 있음.

## 4. 구현 노트
- 생성 루프: Dream repo의 diffusion_generate API(HF transformers 기반, trust_remote_code) — `tb_diffusion_sample.py` 신규 (프롬프트 = inference.py와 동일 문자열 재사용, 출력 = inference.py 호환 predictions jsonl로 기록 → 기존 채점·조인 도구 전부 재사용).
- 후처리 사다리: parse-fix(reformat) → name-snap(v0) → 풀 합류. guided는 불가(서빙 스택 비호환) — **불공정 비교 방지를 위해 AR 풀도 snap-기준으로 정렬한 변형 병기**.
- 자원: 다운로드 ~16GB(디스크 1.3T 여유) — **D1/D2 학습과 무충돌(네트워크/디스크만)**, GPU는 학습 종료 후.

## 5. 리스크 (정직)
①형식 준수가 최대 리스크(P-D0가 게이트) ②diffusion 추론 속도(스텝 수 × 길이 — sub500×8이면 수 시간) ③DiG-Plan 기제의 프로토콜-의존 가능성(그래서 ⓑ도 1급) ④신규 의존성(trust_remote_code) — seka_env 격리 설치.
