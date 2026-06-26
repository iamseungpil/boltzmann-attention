# B0 Vanilla Baseline 분석 보고서 (v1 · 16K partial)

> **Status**: superseded — 본 보고서는 **2026-05-27 02:15 시작 run의 B0 부분**만 다룬다.
> 같은 run의 B1·B2는 vLLM 사망으로 무효 (전체 456 sims connection error).
> v2 (max-model-len=32768, B0+B1+B2 모두 재실행)는 별도 보고서로 작성될 예정.
>
> **v1 raw data 보존 위치**: `reports/facet_rft_2026/phase1_baseline/base_n114_v1_16k_partial/`

**실행**: 2026-05-27 02:15 ~ 07:42 KST (5h 27m)
**대상**: τ²-bench telecom, base split (N=114), trials=4, max_steps=200, max_concurrency=8
**모델**: Qwen2.5-7B-Instruct (local vLLM, port 8000, max-model-len=16384)
**Agent**: B0 (Vanilla — default `llm_agent`, scaffolding 없음)

산출물:
- 원본: `reports/facet_rft_2026/phase1_baseline/base_n114_v1_16k_partial/B0_telecom_base.json/results.json` (109 MB)
- 자동 metric: `base_n114_v1_16k_partial/analysis/B0_analysis.json`, `B0_analysis.md`
- 심화 metric: `base_n114_v1_16k_partial/analysis/B0_deep.json`
- 본 보고서: `B0_analysis_report.md` (해석 + 결정 근거)

---

## 1. Headline

| 지표 | 값 | 비고 |
|---|---|---|
| Total simulations | 456 | 114 × 4 trials |
| Evaluated (infra error 제외) | 421 | |
| **Avg reward (evaluated)** | **0.0475** | tau2 leaderboard 표준 |
| Avg reward (all, infra=0) | 0.0439 | infra error를 fail로 처리 |
| Full-credit (reward ≥ 1.0) | 20/421 = 4.75% | 강한 통과 |
| Wilson 95% CI | [0.031, 0.072] | full-credit 비율 |

**pass^k (task-level, 4 trials 중 ≥1 성공)**:

| k | passed tasks | pass^k |
|---|---|---|
| 1 | 2/114 | 0.0175 |
| 2 | 8/114 | 0.0702 |
| 3 | 11/114 | 0.0965 |
| **4** | **15/114** | **0.1316** |

**핵심**: 단일 trial로는 1.75%만 성공. 4 trials best-of로 13.16%까지 회복. **단 1개 task도 4 trials 모두 성공하지 못함**.

---

## 2. Termination 분포 — 가장 진단적인 신호

| Termination | n | 비중 | 의미 |
|---|---|---|---|
| user_stop | 254 | 55.7% | user simulator가 종료 — agent가 답을 줬다고 본 경우 |
| **max_steps** | **153** | **33.6%** | **200 step 끝까지 결판 안 남 — agent 무한루프** |
| infrastructure_error | 35 | 7.7% | vLLM/Context 에러 (분석 §6) |
| too_many_errors | 14 | 3.1% | tool call 에러 누적으로 강제 종료 |

**Termination × reward 교차표**:

| Termination | n | avg_reward | pass_rate (≥1) |
|---|---|---|---|
| user_stop | 254 | 0.0787 | 7.87% |
| max_steps | 153 | **0.0000** | **0.00%** |
| infrastructure_error | 35 | 0.0000 | 0.00% |
| too_many_errors | 14 | 0.0000 | 0.00% |

→ **max_steps에 닿은 sims는 단 하나도 reward를 얻지 못함**. 즉 agent가 200 step 안에 답을 못 내면 그 trial은 100% 실패. `user_stop`까지 가야 통과 가능성이 존재 (7.87%).

---

## 3. Category별 양상 — 매우 분산된 능력 프로파일

| Category | n_sims | avg_reward | user_stop pass_rate | max_steps n | 진단 |
|---|---|---|---|---|---|
| **service_issue** | 116 | **0.1379** | **19.8% (16/81)** | 33 | 가능성 있음 |
| mobile_data_issue | 144 | 0.0278 | 5.3% (4/76) | 50 | 거의 실패 |
| **mms_issue** | **196** | **0.0000** | **0.0% (0/97)** | 70 | **전체 실패** |

**mms_issue 0/97 user_stop pass**: user simulator 기준 종료된 97 sims 중 **단 한 건도 reward ≥ 1.0이 아님**. mms 도메인은 7B 모델 vanilla로는 사실상 불가능.

**max_steps × category × persona** (해당 termination 발생 횟수):

| Category | Easy | Hard | None | 합 |
|---|---|---|---|---|
| service_issue | 15 | 6 | 12 | 33 |
| mobile_data_issue | 17 | 16 | 17 | 50 |
| **mms_issue** | **23** | **26** | **21** | **70** |

→ mms_issue는 persona 종류와 무관하게 max_steps 도달. 모델이 `mms` 도구체인을 못 끌고 감.

---

## 4. Persona 효과 — 직관과 반대

| Persona | n_total | n_evaluated | avg_reward |
|---|---|---|---|
| **Hard** | 144 | 136 | **0.0735** |
| Easy | 152 | 140 | 0.0429 |
| None | 160 | 145 | 0.0276 |

**Hard가 가장 잘 나옴**. 일반적으로 hard persona는 더 어렵다고 기대되지만, 본 데이터에서는 반대.
- 가설: Hard persona는 더 구체적인 정보를 일찍 주거나, 종료 신호를 명확히 줘서 agent가 결판을 빨리 봄.
- None persona는 가장 약함 — persona 정보 없이 user simulator가 모호하게 행동하면 agent가 혼란.

---

## 5. Trial 일관성 — Stochasticity 매우 큼

| 양상 | tasks |
|---|---|
| consistent pass (4/4) | **0** |
| consistent fail (0/4) | 99 (86.8%) |
| inconsistent (1~3/4) | 15 (13.2%) |
| per-task mean reward std | 0.0692 |

**0 tasks이 모든 trial 통과** — Qwen2.5-7B vanilla로는 어떤 task도 안정적이지 않음.

**Inconsistent 15 tasks의 패턴** (sample 일부):

| task | rewards | terminations |
|---|---|---|
| service_issue · airplane_mode_on \| lock_sim_card_pin [Easy] | 0,0,1,1 | user_stop ×4 |
| service_issue · airplane_mode_on \| break_apn_settings \| contract_end_suspension \| lock_sim_card_pin [None] | 1,1,0,0 | u,u,max,u |
| mobile_data_issue · airplane_mode_on \| data_mode_off [None] | 0,1,0,0 | u,u,max,u |

→ inconsistent 15는 모두 service_issue 또는 mobile_data_issue. **mms_issue엔 inconsistent도 없음** (전부 fail).
→ trial이 늘어나면 pass^k가 올라가는 게 이 15 tasks 덕분 (pass^4 - pass^1 = 13 추가).

---

## 6. Infrastructure error 35건 상세

| 종류 | 건수 | 발생 시점 (첫/마지막) | 단순 rerun으로 해결? |
|---|---|---|---|
| **ContextWindowExceededError** | 18 | 02:32 ~ 07:27 (분산) | ❌ max-model-len=16384가 부족. 32768 필요 |
| **Connection error** | 16 | 07:41:27 ~ 07:42:09 (~42초간 끝부분 클러스터) | ✅ 건강한 vLLM이면 발생 안 함 |
| Unknown tool replay | 1 | 04:09:33 | ? `check_roaming_status` 도구 lookup 실패 — tau2 내부 이슈 |

### 6.1 ContextWindowExceededError 패턴
- 입력 token 16,396 ~ 16,481 토큰 (max 16,384 초과 — 32~97 tokens overflow)
- 전 시간대(첫 17분 ~ 5h 12m)에 걸쳐 발생 → 특정 시점 이슈 아닌 **구조적 문제**
- max_steps에 도달한 sims의 message count 평균 201 (vLLM tokens로는 16K 초과 가능)
- **해결**: max-model-len을 32768로 올리거나 max_steps를 낮춰야 함

### 6.2 Connection error 패턴
- 16건 모두 07:41:27 ~ 07:42:09의 42초 구간 — vLLM 사망 직전
- 모두 `[mms_issue]` 카테고리 (이미 mms는 reward 0이라 영향 최소)
- 이건 단순 vLLM 재가동으로 해결

### 6.3 Unknown tool error (1건)
- `check_roaming_status`라는 도구가 환경에 없는데 model이 replay에서 호출
- tau2 내부 retry 4번 모두 같은 결과 → model이 특정 prompt에서 환각 tool 이름 생성
- 단순 rerun 시 같은 seed에서 같은 환각 가능. 다만 1건이라 영향 미미

---

## 7. DB check vs Reward — 불일치 패턴

τ²-bench는 `env_assertion`(상태 검사) + `db_check`(DB 일치) + `action_checks`(필수 액션 호출)로 reward를 계산.

| DB match | Pass (reward≥1) | Fail | 합 |
|---|---|---|---|
| match | 12 | 25 | 37 |
| nomatch | 8 | 209 | 217 |
| unchecked | 0 | 167 | 167 |

- **match-pass (12)**: 깨끗한 통과
- **match-fail (25)**: DB는 맞췄지만 다른 평가 기준(env_assertion 등) 실패 — agent가 일부만 처리
- **nomatch-pass (8)**: DB는 못 맞췄는데 통과 — env_assertion이 db보다 느슨한 경우
- **unchecked-fail (167)**: db_check 자체가 적용 안 된 sim (action 검증만으로 평가)

→ DB match와 pass는 강한 상관이 아님. 평가 다층화돼 있어서 어느 한 axis로 환원 불가.

---

## 8. Reward basis 분포

| Basis | 횟수 | 비중 |
|---|---|---|
| ENV_ASSERTION | 254 | 80.9% |
| ACTION | 60 | 19.1% |

**ACTION 카운트**:
- 평균 2.44건/sim (max 12)
- 167 sims는 action_checks 비어 있음 (env_assertion만으로 평가)

**ENV_ASSERTION 카운트**:
- 평균 1.06건/sim (max 3)

평가 기준이 task마다 다르고 layered. ENV_ASSERTION이 dominant.

---

## 9. Duration / Message length

| Termination | mean sec | median sec | p95 | mean msg | max msg |
|---|---|---|---|---|---|
| user_stop | 92.0 | 85.2 | 182.1 | 47.7 | 118 |
| **max_steps** | **218.2** | **213.6** | **295.6** | **201.2** | **206** |
| too_many_errors | 123.4 | 110.6 | 210.3 | 65.4 | 110 |
| infrastructure_error | 0.0 | 0 | 0 | 0 | 0 |

- max_steps에 닿은 sims는 평균 218초 (3분 38초) + 201 messages — agent가 끝없이 시도
- user_stop 92초 평균 — 정상 종료는 1.5분 정도
- max_steps 153건 × 218초 = **약 9.3시간**이 max_steps 처리에 소모됨. 전체 5h 27m 동안 동시성 8로 돌렸으므로 이게 critical path

---

## 10. Best/Worst tasks

**Worst (avg_reward = 0.0)**: 99 tasks 모두. 상위 10건 모두 mobile_data_issue / mms_issue. (전체 mms 47개 task가 여기 포함)

**Best (avg_reward > 0)** — 5건이 0.5 달성:
- `service_issue · airplane_mode_on|break_apn_settings|contract_end_suspension|lock_sim_card_pin [None]` — 0.5
- `service_issue · break_apn_settings|lock_sim_card_pin|overdue_bill_suspension|unseat_sim_card [Easy]` — 0.5
- `service_issue · airplane_mode_on|contract_end_suspension|lock_sim_card_pin|unseat_sim_card [Hard]` — 0.5
- `service_issue · airplane_mode_on|lock_sim_card_pin [Easy]` — 0.5
- `mobile_data_issue · airplane_mode_on|bad_network_preference [Hard]` — 0.5

**관찰**: 4건이 `airplane_mode_on + lock_sim_card_pin` 조합 — 이 두 이슈는 agent가 비교적 잘 처리.

---

## 11. Hallucination retry

- Total retries used: **0**
- Sims with ≥1 retry: **0/456**

→ tau2의 hallucination check이 발동된 적이 없음. 모델의 tool call이 schema 측면에서는 깨끗했다는 뜻 (의미는 별개).
   하지만 §6.3의 `Unknown tool` 1건은 4 attempts 후 영구 실패 — 이건 hallucination_retry와 별개 경로.

---

## 12. 종합 진단

### 12.1 모델 능력 프로파일
- **service_issue**: 19.8% user_stop pass rate — Qwen2.5-7B 한계까지 사용 가능
- **mobile_data_issue**: 5.3% — 어렵지만 일부 시도 통과
- **mms_issue**: 0% — 7B 한계 명확. **Vanilla로 풀 수 없음**

### 12.2 구조적 병목
1. **max_steps=200 한계** (33.6% sims에서 발동)
   - 종료 못 한 sims는 100% reward 0
   - max_steps 도달 sims의 평균 message count 201 (≈ 200 step 가득)
   - 이건 agent가 **언제 그만둬야 할지 모름** — vanilla에 종료 기준이 없음
2. **Context window 16384 오버플로** (18 / 456 = 3.95%)
   - message 201 doc의 token 수가 16K 넘는 경우
   - max-model-len 32K로 올리면 해결
3. **Stochasticity** — 0 tasks가 4/4 통과
   - temperature=0인데도 (agent_llm temp 0.0) tool-call routing의 stochastic noise

### 12.3 무엇이 B1/B2를 의미있게 만드는가
- **B1 (ReAct)**: Thought/Action 강제로 **agent의 종료 시점 판단**을 개선할 수 있는지 — max_steps 비중 감소가 핵심 지표
- **B2 (Text-serialized ontology)**: 42 relations text 주입이 **mms_issue 0%**를 깨고 약간이라도 들어 올리는지가 핵심
- B0가 0인 mms_issue에서 B2가 1건이라도 통과하면 Δ=+0.005 시그널. 더 의미있는 효과는 service_issue/mobile_data_issue에서 +5%p 이상 차이.

### 12.4 leaderboard 보고 형태
- **B0 pass^1 = 0.0475 (95% CI [0.031, 0.072]) on telecom-base N=114, 4 trials**
- 보조 metric: pass^4 = 13.16% (확률적 best-of-4)
- Category breakdown 필수 — overall은 mms가 끌어내림

---

## 13. B1/B2 재실행 권고 사항

분석에 비추어:

### 13.1 vLLM 설정 변경 권고
| 설정 | 기존 | 변경 권고 | 근거 |
|---|---|---|---|
| GPU | 0 | **1** | GPU0는 mais 사용자 점유 (port 8010) |
| Port | 8000 | **9000** | 우리 vllm 사망 후 동일 포트 회피, mais의 8010과 분리 |
| max-model-len | 16384 | **16384 유지** | B0(421/456)과 동일 조건. 32K로 올리면 cross-baseline 비교 깨짐 |
| dtype | bfloat16 | 동일 | |
| max_steps (runner) | 200 | 동일 | 공식 leaderboard 조건 |

### 13.2 B0 재실행 여부 — 권고: **하지 않음**
근거:
- 35 infra error 중 18은 ContextWindow(구조적, max-model-len 안 바꾸면 재현), 16은 vLLM 사망 잔여(이미 알려진 잡음), 1은 모델 환각(rerun으로 변화 보장 없음)
- 18 ContextWindow는 max-model-len 32K로 올리면 사라지지만, 그러면 B0 421 + 추가 35의 **조건 불일치** 발생 → 평균 reward 계산 시 분모 통일 어려움
- 보존 가치: **B0 421 evaluated를 그대로 두고 분모 421로 평균/pass^k 보고** (이미 작성한 위 metric)
- 만약 더 깨끗한 비교가 필요하면 → Phase 1 v2에서 모든 baseline을 **max-model-len=32768, max_steps=200**으로 통일 재실행 (≈ 16-18h 추가 소요)

### 13.3 B1/B2만 재실행 — 즉시 가능
- max-model-len=16384, port=9000, GPU1로 B1+B2 동작
- 예상 소요: 5~6h (B0와 유사한 task 수, vLLM 안 죽으면)
- B1/B2도 18건 정도 ContextWindowExceededError가 나올 가능성 ─ 그래도 B0와 동일 조건이라 비교는 깨끗

---

## 14. 다음 단계

1. **(즉시)** vLLM 재가동 — GPU1, port 9000, max-model-len 16384
2. **(즉시)** B1/B2 재실행 — `phase1_runner.py --variants B1 B2 --base-url http://127.0.0.1:9000/v1`
3. **(완료 후)** 동일 분석 스크립트로 B1, B2 metric 산출
4. **(비교 단계)** B0 vs B1 vs B2 cross-comparison 보고서 작성
   - Δ pass^1, Δ pass^4
   - Category × baseline (mms_issue가 깨지는지가 핵심)
   - Termination 분포 변화 (max_steps 비중 감소 = scaffolding 효과)
5. **(메타 결정)** B2가 mms_issue / mobile_data_issue에서 B0 대비 +5%p ≥ 2 categories면 Phase 2 진입; 아니면 max-model-len/max_steps 재설계 후 v2 실험
