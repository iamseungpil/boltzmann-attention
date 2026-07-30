# 무료 트랙 1차 결과 — X1 전독·X2 재진술 측정(예비)·X7 MFIX-조사 (2026-07-30)

> `EXPERIMENT_PLAN_PATENT_PAPERS` rev1의 즉시-가능 3건 실행 기록. 리뷰 수정 1 적용:
> X2 수치는 전부 **[M-예비·추출기 gold 감사 전]** — 논문 인용 금지·감사 후 승격.

## §1. X1 — rival 3편 전독 판정 (전부 비선점 확정 [S-lit])

| 논문 | 실체 | 판정 |
|---|---|---|
| **ATLAS-RTC** (2603.27905) | decode-time **형식-계약만**(drift 감지→biasing/masking/rollback·"decoding artifacts" 표적·+20~37.8pp 첫-시도) — 수치/사실 주장의 원장 검증 **없음**·다회턴 재진술 검사 없음 | **수치 선언-채널·재진술 충실도 비선점 확정.** decode-time 통제 선행으로 인용 |
| **Paper2Data** (2604.16317) | 도시-데이터 메타데이터 추출(단일-문서·human-annotated eval) — **verbatim-span/fuzzy-matching 기전 부재**(DR2 리서치 agent의 과잉 귀속이었음) | **quote-검증(demand-원장) 기전 양보 불필요.** 인접 추출 연구로만 인용 |
| **Gecko** (2602.19218) | **대화→checklist 생성 없음** — rules+LLM으로 도구명/인자 validity 검사·응답 합성·시뮬레이션 refinement(GATS). 런타임 게이트 아님·유저-발화 근거 검증 없음 | **W1(수요-원장) 거리 확대** — 최근접이 아니라 인접. 시뮬레이션-피드백 선행으로 인용 |

⇒ **특허 청구 문구의 전독-게이트 해소**(DR2 §8 잔여 소거). 스케치 §5-4 감시 대상 갱신.

## §2. X2 — 재진술-충실도 예비 측정 ([M-예비]·도구=`x2_restatement_fidelity.py`)

- 정의: assistant 산문의 통화-금액이 그 시점까지의 원장(도구 출력∪유저 발화)에 부재 =
  미접지. 정규화 변형 집합(콤마·float 정형·후행 0·정수형) 대조.
- **banking (front32 계열 10런·251 sims·провenance=`sim_results/bank_day*front*`)**:
  금액 3,809건 중 미접지 92건(**2.4%**)·미접지 보유 sim 8개. **군집이 계산-무거운 태스크에
  집중**: 023(32+32건=월합계·부족분 암산 — handoff [S] 기전의 상류 그대로)·019(17건=지출
  총계)·020/024/025. pass-sim 5 vs fail-sim 87(**association만** — 인과 주장 금지·역인과 규율).
- **retail 스케일 축 (2,840 sims·asmscale/asmregen 계열)**: 미접지율 **7B 34.2% → 14B
  21.7~16.8% → 32B 23.9~17.1%**. 스팟 정독: 가격차 암산("269.16−272.33=−3.17")·환불 총액 —
  **피연산자는 접지·파생값만 미접지 = 동일 클래스 교차-도메인 재현.**
- 해석(예비): "미접지 재진술"의 실체 = **파생값 암산**(합계·차액) — 미접지≠오류이므로
  논문② 주장은 v2(재계산 대조 오류율) 후에. banking 2.4% vs retail 17~34% 격차 = 도메인의
  계산 밀도 차이로 보이나 감사 전 단정 금지.
- **[[08]] 위생 명시**: ①제외분 — bank_day2back{A,B}·day4front{A,B}(부분)·day9front{A,B}는
  결과 파일에 simulations=0/부분(사유 미조사·기여 0으로 자동 제외) — **감사 시 사유 확인
  필수** ②포함 sims의 종료사유 층화(정상 vs ctxover/crash별 미접지율) 미실시 — gold 감사와
  함께 수행(절단 sim이 분모를 왜곡할 수 있음) ③궤적 정독=예시 2도메인 스팟 완료(023 월합계·
  retail 가격차)·전수 아님.
- **다음(계획 rev1 반영)**: ①층화 gold 100건+ 수작업 라벨(X8과 공유)→정밀도/재현율 병기
  ②v2=파생값 재계산 오류율 ③종료사유 층화 ④frontier 궤적(리모트 pull) 확장.

## §3. X7 — MFIX 조사 (무료분·핵심 발견 2)

1. **★task instructions는 고정이다** — 시나리오-합성 단계는 없다. `banking_knowledge/
   tasks.json`의 `user_scenario.instructions`가 페르소나·제약 전문을 보유(확인: task_002에
   "willing to pay up to an **effective** amount of 100 dollars a year in fees" **명문 실재**).
   ⇒ C215의 "시나리오 재생성"의 실체 = **발화-수준 렌더링 변동**(고정 지시문을 gpt-5.2가
   매 런 다르게 서술·temp 0에도 서버 비결정론) — 하네스 결함이 아님.
   **002 재분류 단서**: "$100 상한"은 user-sim 창작이 아니라 태스크 명문 — gold(Platinum
   $200)와의 충돌은 "effective"(실효 비용=수수료−리베이트) 해석에 달림 → **태스크-정의
   모호성**이며, 런별로 user-sim이 raw-fee로 강하게 발화하면 gold 배제. C215 (d) 항목의
   부분 재분류 후보(하네스→태스크 모호성+렌더링 변동).
2. **seed 배선은 존재한다**: `orchestrator.py:99,424`("reproducibility of agent and user
   behavior")·`user_simulator.py:283 set_seed`. 단 openrouter/gpt-5.2가 seed를 존중하는지
   미확인(기존 관측: temp 0에도 비결정론 = 미존중 가능성 높음).
- **⇒ Y1 설계 입력 (고정 수단 후보·비용 순)**:
  (a) seed 고정 유효성 마이크로-확인(태스크 2~3개×2회 동일-seed 재현성·유료지만 극소·승인)
  (b) **첫-발화 고정**(한 런의 opening을 스크립트 주입 — 20/32 실질 변경이 첫-발화에서
  측정된 만큼 앵커 효과 큼·상호작용 이후 턴은 여전히 변동)
  (c) user-sim을 로컬 결정론 모델로 교체(무료·단 기존 런과 비교성 단절)
  (d) 불가 시 nt≥2 편차 폭(기존 계획).

## §4. 산출물·상태

- 도구: `scripts/distill/tau2/x2_restatement_fidelity.py`(신규)·예시 400건
  `reports/facet_rft_2026/x2_examples.jsonl`(스팟 정독용).
- X1 완료 [S-lit] / X2 phase-1 완료 [M-예비] / X7 조사분 완료 [S](코드·JSON 정독).
- 대기: X2 gold 감사·v2 / X4·X5 / X3(DECLARATION_FIRST 리뷰-확정 선결) / Y1 승인.
