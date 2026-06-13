# 벤치-불변 프레임워크 × 벤치 포트폴리오 (detail 문서, 2026-06-12)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**에서 각 문서의 역할·상태 확인; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> ⚠️**마스터 = `EXPERIMENT_DESIGN.md`** (§1.5 요약·§7 문서지도 등재). 목표·순서 변경은 마스터에서만 — 이 문서는 규칙표·포트폴리오 선정근거·어댑터 명세의 **구현 세부**.

> **목적 명제 (사용자, 2026-06-12)**: 특정 벤치 최적화가 아니라, **다양한 벤치를 최소한의 자동 노력으로 전부 커버하는 기본 규칙 내재 프레임워크**. 벤치 서베이는 선택용이 아니라 커버리지 행렬의 타깃 목록.
> 상호링크: 규칙·근거 요약 = `../reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md` **§10·§10.5**(레버 장부·층위 분류) · 벤치 생사 전수조사 = 동 **§1.5** · 선행연구 차별점 = `taskbench/TB_GROUNDED_COPY_V1_DESIGN.md` **§6.5** · SOPBench 실측 = `../reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md` · 대형모델 분담 = `../reports/facet_rft_2026/COWORKER_REQUEST_TB_SCALE.md` **§7-8** · thesis 좌표 = `FIELD_GAP_LLM_VALUE_DESIGN.md` **§15.4(정정 포함)·§17.9·§18** · CDP 벤치 설계 = `GROUNDED_BIZ_AGENT_BENCH_DESIGN.md`.

## 1. 벤치-불변 규칙 R1-R8 (프레임워크 내재 — 전부 실측 근거)
| # | 규칙 | 근거 (출처) |
|---|---|---|
| R1 | 심볼(도구명·필드명)은 생성 금지, 컨텍스트 복사 — enum/문법 마스크로 집행 | TB §9.5b (guided daily +8.0·무효 0/13k) |
| R2 | 인스턴스 사실은 act 전 gather 선행 | SOP active-H3 6→15·gather LODO 전이 |
| R3 | 허가/결정 판단은 모델 emit 금지 — 결정론 게이트 offload | SOP 3-NULL LOCK·DGGATE 15→29/34 |
| R4 | 의미 매칭은 모델, 출력 공간은 제약 (하이브리드) | TB §9.5 v0/v1 경계 |
| R5 | 정책류 행동(종결·길이)은 on-policy **양방향** 선호학습만; 규율-완비 base에 모방-SFT 금지 | TB §9.6 (v1 net− ↔ v2 신기록)·§8.5 (32B −5.4 prereg 적중) |
| R6 | 구조 선택(L6)은 K-제안 + 결정론 검증-**선별** (마스킹 아님 — 분포왜곡 회피) | TB §8.7 (edge-snap NULL)·DiG-Plan Pass@10·GAD |
| R7 | 배포 전 base census → 레버 선택 (이득 = base 결핍의 함수; 크기·family 불문) | TB §8/§8.5/§8.6 (1.5B~235B 실측) |
| R8 | 측정 규율: 집계 후 즉시 궤적 census·사전등록·내부-일관 비교·인용=원문 검증 | 전 문서의 방법론 + arXiv 규율 |

## 2. 벤치-어댑터 A1-A5 (벤치당 새로 쓰는 유일한 것)
| # | 추출물 | 자동화 | 견본 |
|---|---|---|---|
| A1 | 도구 카탈로그 → enum 스키마 | 기계적 | `taskbench/tb_guided_schema.py` |
| A2 | **정책/SOP NL → 제약 구조(게이트 입력)** | **★유일 연구-난제 = thesis의 학습 front-end** | SOP Guard-2 재구성(구조 제공시)·학습 front-end(NL시) |
| A3 | 평가기 → 보상·채굴 신호 (RFT/DPO) | 래핑 | `tb_dpo_mine.py`·sopbench_reward |
| A4 | 도메인 경계 → LODO 분할 | 기계적 | lodo 스크립트군 |
| A5 | 출력 스키마 → guided 문법 | 기계적 | schema `--dep` 분기 |

**주장 형태**: 새 벤치 커버 비용 = A1/A4/A5(기계적)+A3(래핑)+**A2** — A2의 자동화 정도가 기여의 크기(per-domain authoring 비용 제거).

## 3. 실험 벤치 포트폴리오 (2026-06-12 제안 — A2 난이도 스펙트럼 구성)
| Tier | 벤치 | A2 난이도 | 축 | 역할 (실증 규칙) | 상태 |
|---|---|---|---|---|---|
| 1 | TaskBench | 없음(정책 무) | 플랜-예측 | R1·R4·R5·R6·R7 | ✅ 완료 (외부 동결 — TB §1.5; 내부-일관 유지) |
| 1 | SOPBench | 구조 제공 | 실행-준수 | R2·R3 (per-constraint 진단 = 유일 자산) | ✅ 완료 (upstream 비유지보수·Amazon SOP-Bench와 이름충돌 주의) |
| 1 | **★τ²/τ³-bench** | **순수 NL 정책** | 실행-준수 | **A2 front-end 필요성의 실증 무대** + R3 + pass^k(게이트=일관성) | 신규 1순위 — 유일 활성 frontier 리더보드(~30모델 '26-05) |
| 2 | **Amazon SOP-Bench** | SOP 텍스트 | 실행-준수 | **R7을 12도메인 LODO로** (전이 스케일업, 2,000+태스크) | 신규 2순위 (어댑터 대부분 기계적·CC-BY-NC) |
| 2 | AppWorld | 없음(암묵) | 실행-검증 | R3 타-환경 (collateral-damage 단위테스트) | 신규 3순위 ('26-02 활성) |
| 3 | ODCV-Bench | 유혹-시나리오 | 거부-축 | "게이트=KPI-유혹 위반 0" (frontier 30-50% 위반) | 스팟 (40개·저비용) |
| 조건부 | WorFBench | 그래프 생성 | 플랜-예측 | TaskBench 후계 — 플랜-축 외부비교 복원 | 구조-축 조사 도착 후 확정 |

**선정 논리**: ①A2 스펙트럼 4점(없음→구조→semi-SOP→순수NL) 완성 ②τ²=활성 리더보드(외부 비교)+NL정책(front-end 데모)+pass^k(게이트 일관성 보상) ③Amazon=12도메인 전이 ④ODCV=결정론-leg 한 줄 헤드라인.

**실행 순서**: ⑴τ² retail 어댑터(A1 기계+A3 래핑+A2 수동-1회 — 수동본이 front-end 자동화의 GT가 됨)→7B+게이트 평가 ⑵Amazon 12-도메인 LODO 행렬 ⑶AppWorld·ODCV 스팟. 대형모델(32B+) arm은 Track-B(coworker) 분담 — `COWORKER_REQUEST_TB_SCALE.md` §8.

## 3.5 τ² retail 어댑터 스코핑 (E3, 2026-06-12 — 리모트 클론·구조 실측)
- 클론 = woori `/home/woori/scratch/tau2-bench` (sierra-research, depth 1). 도메인 = retail(114 tasks)·airline·telecom·banking_knowledge(τ³)·mock.
- **A1-A5 추출물 실측**: A2 = `domains/retail/policy.md` **136줄 NL 정책** — 핵심 게이트 4종이 **SOPBench와 동형**: ①**인증-선행**("authenticate ... even when the user already provides the user id" = LOGINFIRST 동형) ②**쓰기-전-확인**(cancel/modify/return/exchange 전 명시 confirm = goal-call 게이트) ③단일-유저 범위(타 유저 요청 deny) ④정책-외 거부 + transfer 규정. / A3 = tau2 evaluator(DB-state 등가 reward) 래핑 / A4 = 도메인 3+1 / A1 = tau2 패키지 내 도구 정의(코드 — 추출 스크립트 필요) / A5 = 대화형(툴콜 스키마 — guided는 tool-call JSON에 적용).
- **A2 수동 컴파일 v1 전망**: 게이트 4종 + per-action 인자 규칙(나머지 ~100줄) — SOPBench Guard-2 절차(정책→graph 재구성→evaluator 대조) 재사용 가능. 수동본 = front-end 자동화의 GT.
- 다음: ✅①A1 도구 추출 스크립트 ✅②retail 정책 수동 컴파일(게이트 4종) — §3.6 / ③7B+게이트 vs 7B-alone pass^1/pass^k 첫 측정 (GPU 가용 시).

## 3.6 τ² retail A1+A2 구현·검증 결과 (2026-06-12 — push)
- **A1 완료** = `tau2/t2_extract_tools.py` → 16 도구(READ 7·WRITE 7·GENERIC 2) openai-schema+enum+타입맵 (`/home/woori/scratch/tau2_adapter/`).
- **A2 완료** = `tau2/t2_gate.py`: G1 인증-선행(user-scoped 9종 = WRITE 7+get_user/order_details; 카탈로그 READ·calculate 면제)·G2 쓰기-전-확인(직전 user 턴 확인-regex, live 전용)·G3 단일-유저(user_id 인자 + order_id→db owner resolve)·G4 transfer 고정문구(post-hoc 헬퍼).
- **★스코핑 발견**: per-action 인자규칙(~100줄: status·reason enum·동일상품·잔액·환불처·1회성)은 **tools.py가 전부 자체 집행(raise)** — 게이트 가치는 대화-수준 G1-G3에 집중. SOPBench(DGGATE가 graph 전체 재구성)보다 어댑터가 더 얇음 = 프레임워크 주장(A2 비용↓) 강화.
- **Guard-2 동형 검증 PASS** (gold 114 태스크·550 액션 replay): PassA G1-순서 위반 **0** / PassB G3 over-deny **0** (GT 유저=gold 인자 order-owner·user_id 합의, multi-user 0). ⚠️naive replay는 G1 deny 86 — **gold가 인증 READ 생략(46/114, DB-state 보상이라 READ 불요)**한 observed-proxy 아티팩트로 무효 처리 (SOPBench 함정 동형 — 재발견 금지).
- 다음 = ③측정: tau2 orchestrator에 게이트 hook(에이전트 툴콜 인터셉트→deny 시 게이트 메시지 반환, SOPBench two_stage_client 패턴) + 7B(vllm OpenAI-호환)±게이트 pass^1/pass^k retail 114.

## 3.7 ★τ² retail 7B base ±게이트 1차 측정 (2026-06-12 — run7, 표준 user-sim·judge=gpt-4.1-2025-04-14 via OpenRouter = 리더보드 프로토콜-호환)
| arm | pass^1 | pass^2 | pass^3 | pass^4 | 실행된 인증-전 위반 (write) |
|---|---|---|---|---|---|
| 7B base (nogate) | 0.184 | 0.089 | 0.068 | 0.061 | 121 (**43 write**) / 53 sims |
| 7B base + 게이트 | 0.147 | 0.061 | 0.044 | 0.035 | 53 시도→deny (write 실행 **~1**) |
- **양축 판정 (F3×F4 분리 — 마스터 §1.6)**: ⓕ4 soundness = **write-위반 43→1 (≈98% 차단)** — 결정론-leg 설계대로 작동. ⓕ3 helpfulness = pass^1 −3.7pp·pass^4 −2.6pp — **ⓟ1(Δpass^4>Δpass^1, 양수 전제) 기각**: 이 7B+passive-deny 구성에선 게이트가 일관성 레버가 아님.
- **궤적 census 귀속**: deny 65건(G1 86%·G2 12%·G3 2%)·**deny 경험 sim의 92%(48/52)가 실패** = 7B가 deny 메시지("authenticate first via find_user_id_...")로부터 복구 못 함. 단 base 자체가 nogate 81.6% 실패(DB-state 295 지배) = R7 base census가 가리키는 능력 바닥이 1차 병목.
- **SOPBench 동형 해석**: passive deny의 한계 = SOPBench passive-H3(6)와 동형 — 거기선 active-H3(게이트가 누락 getter 구동)로 6→15. τ²의 G1은 대화 정보(이메일/이름+zip)가 필요해 자동-구동 불가 ⇒ 처방 후보: ①**사전 scaffold**(시스템 프롬프트에 게이트 규칙 명시 = 사후 deny→사전 회피) ②deny-시 복구 절차 주입 ③compliance-first 배포 regime에선 현 트레이드오프 자체가 가치(위반 0 보장 헤드라인 + 성공비용 명시).
- 절대좌표: 표준 user-sim이라 리더보드-비교 가능 — 7B base 18.4%는 frontier(60-80%대)와의 갭 실측. 인프라: judge 하드 gpt-4.1 기본값·json.loads 직접 호출(json_object 강제 필요)·resume 대화형 프롬프트 — 3함정 전부 수정 커밋.

### 3.7b ★N3 = gate_r2 (G1 deny→복구절차 메시지) 사전등록 판정 (2026-06-12 — 114×4 full 재실행, 동일 프로토콜)
| arm | pass^1 | pass^2 | pass^3 | pass^4 | deny→fail | 인증-전 write 실행 |
|---|---|---|---|---|---|---|
| nogate | 0.184 | 0.089 | 0.068 | 0.061 | — | 44 |
| gate r1 (단순 deny) | 0.147 | 0.061 | 0.044 | 0.035 | 92.3% (48/52) | 0 |
| **gate r2 (복구 메시지)** | **0.191** | 0.074 | 0.035 | 0.018 | **80.8% (42/52)** | **0** |
- **사전등록 판정: ①deny→fail<50% FAIL · ②pass^1≥0.184 PASS · ③write-차단 유지 PASS ⇒ conjunction FAIL** (정직 기록). 단 ①은 **오캘리브레이션 판명**: r2의 nodeny 실패율 81.5% ≈ deny→fail 80.8% — deny-sim 실패율은 base 실패율 아래로 내려갈 수 없는데 기준을 50%로 걸었음. **올바른 척도 = deny의 한계 피해(deny−nodeny 실패율 갭): r1 +7.9pp → r2 −0.7pp = 소거.**
- **★헤드라인: 게이트가 pass^1에서 무비용화** — matched 112-태스크(infra-error 4 sims 제외 동일셋) r2 0.1853 vs nogate 0.1830 = parity. **write-위반 44→0 유지** ⇒ "compliance 무료(pass^1 기준)" 달성. 분석 = `tau2/t2_gate_r2_verdict.py`.
- **★귀속 (복구 census — r1 재분석 포함)**: 복구 *행동*은 r1에서도 이미 96%(G1-deny 47 sims 중 auth-after-deny 45·원도구 재시도 42) — **r1의 단순 deny 메시지("authenticate first")로도 모델은 기계적으로 복구했음**. r2 메시지가 바꾼 건 행동이 아니라 **복구 후 성공률**(4/41→9/36 pass): 4단계 절차(재시도 금지→필요입력 질문→satisfier 호출→원행동 재개)가 복구 *품질*(대화 흐름 보존)을 올림. "deny→fail 92%=복구 불능" 해석(§3.7)은 절반만 옳았음 — 병목은 복구 여부가 아니라 복구 과정의 대화 붕괴.
- **잔여 음성**: pass^4 0.0625→0.0179(matched) — 일관성 축은 여전히 악화(전-trial 통과 태스크 7→2개, 소수-n 노이즈 유의·user-sim temp 0.7 분산). ⓟ1(게이트=일관성 레버)은 이 구성에서 계속 기각.
- 영구실패 4 sims(task17×3·task99×1) = `infrastructure_error`·대화 0건 — OpenRouter측 에러, 모델 무관. n=452/456. G2 deny 8→13(G1 54·G3 0).
- **★벤치 pass는 compliance-blind (2026-06-12 심야 census)**: nogate에서 인증-전 WRITE를 실행한 21 sims 중 **6건이 bench-pass** — τ² 보상(DB-state 등가+NL assertion)은 절차 위반을 직접 벌점화하지 않음(gold가 인증 READ 생략 46/114 — §3.6 동형; 정책 원문은 "have to authenticate ... even when the user already provides the user id"로 의무 명시). ⇒ **F3(pass)×F4(위반) 2축 분리 보고가 필수**인 직접 증거.
- **★F4b compliant-pass (2026-06-13, 사용자 발의 — 마스터 §1.6 등재·사후-정의·census-tier)**: pass∧위반-무. 변형 3종 실측 (n=456/arm):
  | arm | bench pass^1 | write-clean | strict-clean | **FULL-clean (G1+G2+G3+G4)** | 위반 sims |
  |---|---|---|---|---|---|
  | nogate | 0.1842 | 0.1711 | 0.1601 | **0.1425** | G1=56·**G2=44**·G3=3·G4=1 |
  | gate r1 | 0.1469 | 0.1469 | 0.1469 | 0.1469 | G4=2뿐 (G1/G2/G3=0) |
  | **gate r2** | **0.1908** | 0.1908 | 0.1908 | **0.1908** | **전부 0** |
  ①**FULL-clean에선 게이트 우위 +4.8pp**(0.1908 vs 0.1425; 점추정 — CI 동반 전 단정 금지). ②**교차검증**: 동일 사후-replay 검출기가 게이트 arm서 G1/G2/G3=0 — live 게이트와 사후-검출기의 상호 검증. ③nogate 제2 위반축 = **G2 미확인-WRITE 44 sims**(G1 56과 비슷한 규모 — "확인 없이 변경"이 인증-생략만큼 흔함). ④G4(transfer 문구)는 게이트 비집행(post-hoc 헬퍼) — r1서 2건 잔존·r2 0. ⑤pass^4(일관성)는 FULL-clean에서도 nogate 0.0263 > r2 0.0179 (정직 기록).
- **★F4b eval-후크 상시화 (2026-06-13, 사용자 발의 "SOPBench처럼")**: 검사기를 공용 모듈 `tau2/t2_compliance.py`로 분리(spec 상태기계 replay = **A2 산출물 3중 재사용**: 집행/측정/GT) — `t2_run_gated`가 **모든 평가 직후 자동 산출** + `simulations/<arm>/compliance.json` 사이드카. 3-arm 회귀검증 동일치 재현. 원칙: τ² 네이티브 evaluator는 불변(벤치 동결·리더보드 비교 보존) — compliance는 2-tier 분리 산출. SOPBench는 evaluator가 경로-채점 내장이라 공식 success가 이미 compliant-pass = 이 갭은 τ²류(결과-채점 벤치) 전용 어댑터 비용.
- **다음 처방 후보**: deny-복구는 종결 — 남은 갭은 base 능력(nogate 81.6% 실패·DB-state 지배). 게이트-side 추가 레버 없음 ⇒ A2 front-end(§3.8-3.9)·base 학습 라인으로 이관.
- ⚠️**프로토콜 드리프트 경고 (메트릭 리서치 2026-06-12)**: τ² 공식 리더보드는 user-sim을 병기하며 현재 **gpt-5.2 권장** — 본 행렬(run7/r2)은 gpt-4.1-2025-04-14. **외부 리더보드 숫자와 비교 시 user-sim 4-tuple(user-sim·judge·trials·split) 명시 필수**, 내부 ±게이트 비교는 무영향. 보고 표준형(paired Δpass^1+bootstrap CI·0/N+rule-of-three 상한·구조적0/표본적0 분리) = `reports/facet_rft_2026/research_framework_metrics_2026_06_12.md` §2.2.4 — §1.6 v2 동결 시 채택.

### 3.7c ★N4 = gate r3 (G4 deny-게이트 + 중립템플릿) 사전등록 판정 (2026-06-13 새벽, 114×4 재실행)
| arm | pass^1 | pass^2 | pass^3 | pass^4 | 위반(G1/G2/G3/G4) | denies |
|---|---|---|---|---|---|---|
| nogate | 0.1842 | 0.0892 | 0.0680 | 0.0614 | 56/44/3/1 | — |
| gate r2 | 0.1908 | 0.0737 | 0.0354 | 0.0179 | 0/0/0/0(운) | 52 sims |
| **gate r3** | **0.1952** | **0.1038** | **0.0708** | 0.0541 | **0/0/0/0(집행)** | 105 sims (G4 **65**) |
- **판정**: ①G4 위반 0 **PASS**(이번엔 운 아닌 집행 — G4 deny 65건이 문구 송신을 유도) ②pass^1 r2 동등 **PASS**(0.1952, 오히려 +0.4pp) ③G4 deny "1~3건" 예측은 **대폭 기각**(65건 — transfer 시도가 예상보다 훨씬 빈번; r2의 G4=0은 그 65건 중 우연히 문구가 앞섰던 게 아니라 deny 없이도 따라온 운이었음을 시사... 부검 필요시 r2 transfer 빈도 census).
- **★pass^2-4 대폭 회복**: r2의 일관성 붕괴(pass^4 0.0179)가 r3에서 **0.0541**로 — nogate(0.0614) 근접. **compliant-pass FULL = bench pass = 0.1952** → nogate FULL-clean 0.1425 대비 **+5.3pp 우위로 확대**. ⓟ1(Δpass^4>Δpass^1)도 r3-vs-nogate에선 거의 회복(−0.7pp). ⚠️**혼입 주의(정직)**: r3 = G4 게이트 + 템플릿 문구 변경("once this is done") 동시 적용 — 일관성 회복의 귀속은 분리 불가(소수-n 분산도 잔존). compliance 후크 첫 실전 = 자동 산출 작동 확인.

### 3.7d ★전수 궤적 부검 — pass^4 정체·r3 기제·음성 2건 (2026-06-13 사용자 발주, `t2_passk_autopsy.py`·`tb_pool_autopsy.py`, zero-GPU)
- **★pass^4 정체 = user-sim 분산이 주범 (게이트 레버 무관)**: 전 arm fail-trial 종결사유 **user_stop 압도**(nogate 94/104·gate 84·r2 91·r3 75; too_many_errors 7~14). 같은 태스크 4-trial이 갈리는 건 user-sim(gpt-4.1, temp 0.7)이 매번 다르게 조기 종료하기 때문 — A3: pass-trial이 fail보다 턴수 짧음(17 vs 20, 깔끔히 끝나면 성공·헤매면 user가 끊음). A2 within-task(난이도 통제): deny-trial fail율이 nodeny보다 +5~24% 높지만(게이트가 일관성 손해의 *일부*) 절대 다수 실패는 deny 무관. ⇒ **ⓟ1("게이트=일관성 레버") 검증은 이 측정 프로토콜에선 user_stop 분산에 묻혀 거의 불가** = 메트릭 한계(F3 보고 시 user-sim seed 분산 동반 필수). 일관성 진짜 레버 = base 능력(짧고 정확한 궤적)이지 게이트 아님.
- **★r3 성공 기제 = transfer 차단의 net 양성 (A5/A6)**: r3에서 **transfer 실행 63→0**(G4 게이트가 전부 deny, 모델이 재시도 대신 포기). 그런데 transfer는 원래 실패 경로(nogate transfer 71실행→pass 8 = 11%만 성공) — **차단이 곧 이득**. task-flip r2→r3: UP 25 vs DOWN 20(net +5). UP엔 transfer-task 다수 + G1 deny 제거분(task44 c1→3). **단 DOWN 20 중 G4 deny가 새로 망친 태스크 다수**(task0 c2→0·51 c3→1·8·16 — transfer가 필요했는데 deny로 막힘). ⇒ **개선 = G4를 deny가 아니라 offload**(scaffold가 문구 직접 송신 후 transfer *통과*) — deny형은 transfer-필요 태스크를 죽이고 transfer-불필요 태스크만 구제(양날). 데이터가 보류했던 offload 정공법을 이제 정당화.
- **★음성 NC(v3g 풀) = oracle 동일·다양성 붕괴 (tb_pool_autopsy C — 결정적)**: dpo2g vs v3g — **oracle 0.896 = 0.895(동일·천장 불변)**·AR8-mean 0.762 < 0.772(v3g 단일이 더 강함 ✓)인데 **AR8 내부 다양성 0.024 → 0.016 (−33%)**. DPO가 분포를 날카롭게 해 K8 후보가 서로 더 비슷 → 합의-선별이 식별할 차이 소멸 → 선별 headroom 붕괴(65.98). **"선별 이득=다양성 함수" 정량 확정** — 강한 단일 모델일수록 자기-앙상블 선별엔 불리.
- **★음성 ND(풀 확장) = 개선·악화 상쇄 (tb_pool_autopsy B)**: 선택 변경 69건 중 improved 29 vs worsened 24(net +5뿐)·새↔기존 중복성 0.469. worsened의 새-선택이 qwen32b/72b 15/7 = **대형 단일샷이 가끔 합의를 이기지만 틀린 합의를 이기는 만큼 맞는 합의도 깨뜨림**. ⇒ 풀 무차별 확장 금지, 다양성-기여 검증된 proposer만 선별 편입.
- **종합 개선 3방향**: ①F3/pass^4는 user-sim seed 분산 보고로 정직화(게이트 일관성 주장 철회·base 능력 라인으로 이관) ②G4 deny→**offload 전환**(transfer-필요 태스크 구제) — PORTFOLIO §3.8 큐의 정공법을 데이터가 승격 ③선별 풀은 **다양성 명시 증대**(이종 temp/base/프롬프트)가 강한-단일 K샘플보다 우월 + SEL-4 직교 신호.
- **★ⓟ1 분산통제 arm 반증 (2026-06-13 day13 [F], `retail_7b_gate_r3_ut0`)**: user-sim temp **0.0** 고정해도 pass^4 **0.018** (r3 temp0.7의 0.054보다 *낮음*)·pass^1 0.177(<r3 0.195).
- **★★ⓟ1 진짜 원인 = 에이전트 생성 비결정성 (전수 궤적조사 2026-06-13, `t2_p1_autopsy.py`)**: ut0(user temp0)·agent temp0 둘 다 결정론 설정인데도 **4-trial 행동 시퀀스가 0/111 동일**(인자포함·도구명 둘 다 0%) = **vLLM 추론 자체가 비결정**(연속배칭·부동소수점·prefix 캐싱 — temp0 greedy 명목과 무관). flaky 태스크 분해: **행동-다름(생성비결정) 100%(44/44) vs 같은행동-다른채점(judge잡음) 0%** = **분산은 전적으로 agent 생성에서, 채점기는 결정론**. arm-교차: 두 arm **공통실패 45%**(난이도-결정·user 무관)·과반-flip 16%(user temp 영향은 소수). ⇒ **확정 결론: pass^4 천장 = ①agent 생성 비결정성(분산 단일 원천) + ②base 능력 바닥(공통실패 45%) — 둘 다 게이트-외 요인이고 user-sim 분산은 주범이 아니었음**(부검의 user_stop은 *종결사유* 라벨일 뿐 분산원 아님). **ⓟ1 영구 철회 확정**. 처방: 일관성 레버는 게이트가 아니라 (a) 추론 결정론화(seed·batch1) (b) base 능력 — pass^4 보고 시 "vLLM 생성 비결정성" caveat 의무.

### 3.7e ★frontier F4b census (2026-06-13 — 추세리뷰 숙제②, retail 114×1 census-tier, agent=gpt-4.1·4-tuple 동일)
| arm | bench pass^1 | **FULL-clean** | 위반 sims |
|---|---|---|---|
| 7B nogate | 0.184 | 0.143 | **56** (G1+G2+G3+G4) |
| 7B + gate (r3) | 0.195 | **0.195** | 0 |
| **gpt-4.1 nogate** | **0.816** | 0.798 | **4** (전부 G2 미확인-WRITE) |
| **gpt-4.1 + gate** | 0.788 | 0.788 | **0** |
- **사전등록 판정**: ⓕA 위반>0 **PASS**(4 sims — 단 ODCV-외삽보다 약함: G1 인증은 frontier가 자발 준수 0건, 위반축은 G2 확인-생략) · ⓕA FULL<bench PASS(−1.8pp) · ⓕB 위반 0 **PASS = 게이트 model-agnostic 입증** · ⓕB pass FA±3pp PASS(−2.8pp, 경계; FULL-기준 −1.1pp). bench 0.816은 예측 상한(0.80) +1.6pp 초과(정직).
- **정직한 서사(과대주장 금지)**: frontier는 네이티브 compliance가 상당히 좋다(위반 4/114) — "frontier도 크게 깎인다"는 강형 주장은 **기각**. 성립하는 주장: ①소형에선 게이트가 FULL-clean을 **+5.3pp 올리며** 위반 56→0 (변혁적) ②frontier에선 게이트가 **−1.1pp FULL 비용으로 구조적-0 보장**(보험·감사가능성 — 표본적 0이 아닌 구조적 0의 제도 가치, §1.6 F4 분리 그대로) ③게이트는 모델 불문 동작 = 인프라. ⚠️trials=1·n=114 — 위반율 CI 넓음(4/114, 95% 상한 ~8.9%), 비교주장 시 CI 동반.

### 3.8 ★A2 산출물 ↔ R3 템플릿 분리 (2026-06-12 사용자 지시 — 수동 프롬프트 금지)
- 구조: **A2 컴파일 산출물 = `GATE_SPEC` 구조 데이터**(JSON 덤프 `tau2_adapter/retail_gate_spec.json`: predicate·satisfier도구→필요입력·applies_to·terminal여부) / **R3-side 불변 템플릿 `render_recovery`**가 전 deny 메시지를 spec에서 *생성* — 도메인 문자열 hand-authoring 0. 새 도메인 비용 = spec 컴파일뿐(메시지·게이트 자동).
- 검증: 생성 메시지 3종이 수동본과 의미 동등(G1 복구 4단계·G2 확인 절차·G3 terminal 거부 안내) ∧ replay 재검증 PASS(0/0). N3(진행 중)는 수동본으로 시작했으나 생성본과 의미 동등이라 결과 대표성 유지.
- 다음 검증 목표(사전등록 후보): **airline 도메인 spec만 컴파일해 게이트+메시지 무수정 작동** = "새 벤치 비용=A2" 주장의 2-도메인 실증.
- **★G4 의무형 deny-게이트 (2026-06-13 구현 — 사용자 결정: 차선책 채택, offload는 보류 대안)**: G4(transfer 시 고정문구)는 의무형(부작위 위반)이라 차단형 게이트의 원관할 밖 — **차선 집행 = 의무를 사전조건으로 변환**: 문구 미송신 상태의 `transfer_to_human_agents` 호출을 deny + N3식 복구 메시지("정확히 이 문구를 먼저 송신 후 재시도"). 구현 = GATE_SPEC `G4_TRANSFER_MSG`(ask-kind) + `check(transfer_msg_sent=...)` + patch가 어시스턴트 발화 이력 스캔. **한계 명시: 보장 아님**(모델이 복구를 따라야 함 — offload만이 구조적 0; N3 실측상 복구-추종률 96%라 기대 효과 큼). 검사기 의미론도 정합화: G4 위반 = transfer 실행 ∧ 문구가 대화 전체에 부재(순서-무관 — "후"만 보면 게이트-준수 대화를 오검출). **사전등록(다음 τ² gate 실행에서 검증·airline 번들)**: ①G4 위반 사실상 0(완전 보장은 비주장) ②pass^1 영향 ±0 ③deny 1~2건 추가 발생(G4 deny→복구 비용은 무시 수준 예측).
- **★A2 생성기 문제의 좌표 (2026-06-12 문답 박제)**: 산출물(GATE_SPEC)+결정론 검증기(replay) 인터페이스 확립으로 **생성기가 교체-가능 부품화** — ①프로그램 파서=L0(NL 이해 천장 낮음) ②frontier LLM 1회 컴파일=실용 최적(도메인당 1회라 비용 무관·검증기 동반 필수) ③훈련 소형 LLM=thesis(주권/Δ-robustness/전이 주장에서만 정당화). 셋은 경쟁이 아니라 파이프라인(검증기 불변·frontier가 GT 생성·소형이 내재화). ⇒ 마스터의 "세 컴파일러 대조군"(NL→graph 천장)이 GATE_SPEC 컴파일을 무대로 실측 가능: 동일 검증기로 L0/frontier/소형 채점, retail+airline 2-도메인.

### 3.9 ★P-A2-0 실행 = PASS — frontier(Fable-5 in-session) airline 컴파일 replay-clean (2026-06-12)
- **결과**: airline policy.md(166줄, 미답 도메인) → `specs/airline_gate_spec_fable5.json` 6게이트 단일샷 컴파일 → 상태-추적 replay 검증(50태스크 gold 전수) **over-deny 0/108** (G3 소유권 39·G4 취소적격 11·G5 수정규칙 28·G6 지불조성 30). G1/G2는 대화-전용이라 replay 불가(retail 동일 한계). **GT 파이프라인 생존성 확인 + frontier 단일샷 baseline 확보** (`A2_FRONTEND_DISTILL` P-A2-0 게이트 통과 → P-A2-1 합성 데이터엔진 진행 가능).
- **★도메인-간 A2 두께 실측**: retail=도구 자체집행이라 대화-게이트 4종뿐 ↔ **airline="The API does not check" 명시 2회 = 결정론 인자-게이트가 본체**(취소적격·수정규칙·지불조성). A2 비용·게이트 가치가 도메인의 함수 — 커버리지 행렬(§4)의 실측 첫 행.
- **부수 실증 2건**: ①스키마 한계 — db_check가 서술 문자열(검증기가 의미 구현) = `A2_FRONTEND_DISTILL` 리스크④ 실증, DSL 필요 ②**검증기도 상태-추적 필수** — 초판 static-DB replay의 over-deny 2건이 전부 "선행 gold WRITE 효과 미적용" 아티팩트(cabin 업그레이드→취소 적격 등). 부검으로 컴파일 무결 확인 후 검증기 수정.

## 4. 커버리지 행렬 (벤치 × A1-A5 가용성 × R1-R8 적용처) — [작성 중: landscape census·구조-축 조사 도착 후 완성]

## 5. 메타 (인용·측정 규율)
- 벤치 수치 인용 함정 6종 = TB §1.5 ④ (재분할·백본 드리프트·지표 약화·파이프라인 재구성·in-domain 학습 비교·도메인 전치). ToLeaP "GPT-4o" 행 인용 금지.
- 신규 벤치 투입 시 절차: base census(R7) → 사전등록 → 어댑터 → 측정 → 궤적 census(R8).
