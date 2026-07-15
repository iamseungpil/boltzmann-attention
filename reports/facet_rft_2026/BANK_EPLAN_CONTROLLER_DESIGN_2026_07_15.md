# banking E-PLAN 오프라인 컨트롤러 설계 (2026-07-15·handoff §0-a)

> ⚠️ **2026-07-15 정정 (구현 중 [[08]] 발각)**: 아래 §0-3의 **dispute-only ceiling(COMPUTE 19.2%→LOAD 60%)은 폐기**.
> reward=DB-state(81%)·dispute=실패의 20%뿐 → dispute action_checks는 proxy 부분모델.
> 정본 = `BANK_OFFLINE_CEILING_PROXY_FORENSIC_2026_07_15`(§3.6 전-액션 재앵커: coverage-only 20.4%·args 36.7%·pure-DB blind 20.2%).
> 아래 설계는 **엔진 5원소·ABox·[[05]] 원칙은 유효**하나 harness scope가 dispute→전-액션으로 확대됨. 엔진 구현=`bank_eplan_controller.py`(all-action 기본).


> [[03]] 설계먼저 · [[05]] 도메인일반 엔진(리터럴0·ABox서만 도메인지식) · [[14]] 결정론 컨트롤러(read/discovery 강제·write 강제 금지) · [[09]] 무료 오프라인(유료 e2e=b).
> 목적 = **아우터 loop 컨트롤러**(across-item)를 짓고, 실패궤적 replay로 **오프라인 ceiling 재현**(유료 前 필수 게이트). inner COMPUTE는 keystone(C81)서 이미 배선.

## 0. 무엇을·왜
- **입력 사실(확립)**: COMPUTE-only ceiling 19.2%(`bank_operator_replay`)·reach closed-world 69.2%(`bank_reach_forensic`)·liability오답 84.4%=COMPUTE미실행(`bank_load_diagnosis`)·⋈=plan 67.9%(C89).
- **구축물** = `bank_eplan_controller.py`: **순수 엔진**(state→decisions·오프라인 단위테스트 가능) + **오프라인 harness**(궤적서 per-sim state 복원→엔진 walk→ceiling 재현). `plan_execute_orch.py`(retail) 분리구조 미러.
- **왜 지금**: (b) 라이브 e2e 배선의 controller 로직을 먼저 순수·무료로 확정하고, 오프라인 ceiling(COMPUTE 19.2%→LOAD ~60%→GAP ~40%)을 재현해 make-or-break 목표선을 고정.

## 1. 엔진 = 도메인일반 아우터 loop (C90 outer·C91 closure)
루프 5원소 — 전부 ABox(`banking_knowledge.gate.json`) 참조·엔진 하드코딩 0:
```
CONTROLLER(state, abox):
  1. FIND-enumerate : abox.eplan(list_enumerator→detail_reader)로 전 계좌/카드 거래 열거
                       → known support(후보 txn 집합). [discovery 강제·read-only]
  2. coverage-track : gold-대상(dispute 필요) txn마다 제출 보장·미제출 감지. [H_min: 잔여>0이면 미완]
  3. per-item COMPUTE: 제출 dispute마다 abox.compute_ops(liability lookup_table)로
                       compute필드 결정론 채움. [keystone·이미 배선]
  4. ASK            : ⋈-비결정(≥2 후보·reference_filter on_ambiguous=none) 필드 → ASK. [[16]]§4c
  5. H_min-stop     : 모든 reachable dispute 커버될 때까지 continue(잔여 엔트로피>floor·under-action 차단).
```
- **엔진 리터럴 0 검증**: banking "liability 50/500" 규칙·도구명·필드명이 코드에 0. 전부 `compute_ops`·`eplan`·`reference_filter` ABox서 로드. banking→retail 교체 = ABox 교체만.

## 2. 오프라인 harness = 궤적 state 복원 (per-sim·[[08]] per-case 정직)
17 궤적(`C:/tmp/traj/*_banking.json`)의 **실패 sim**마다 복원:
- `gold_disputes {tid: gold_args}` = action_checks(정답 dispute 집합).
- `agent_subs {tid: agent_args}` = agent 실제 제출.
- `seen_txns` = agent가 본(tool result/user) txn (= enumerable).
- `universe_txns` = 그 task의 **어느 run이든** tool result에 등장한 txn (= queryable·완비열거 상한).

엔진이 각 gold dispute를 판정:
| reachable? | 조건 | 처리 |
|---|---|---|
| **agent 제출** | `tid∈agent_subs` | agent record + COMPUTE 재계산 |
| **closed-world A** | `tid∈seen_txns` | FIND-enumerate가 이미 surface → 제출가능(id-correct free) |
| **closed-world B** | `tid∈universe_txns` | 완비열거(list→detail)가 surface → 제출가능 |
| **open-world** | 어디에도 없음 | 결정론 불가 → **completeness-ASK** 잔여(GAP) |

제출가능 dispute마다 COMPUTE 후 잔여 오답필드:
- compute필드(liability류)면 → COMPUTE가 닫음.
- 비결정(⋈ pin·on_ambiguous) → **ASK** 잔여(GAP).

## 3. Ceiling 분해 (재현 목표·확립수치와 대조)
sim이 컨트롤러 후 pass하려면 = **모든 gold dispute가 (reachable ∧ 전필드 정답)**.
| arm | 켜는 연산 | 예상 | 근거 |
|---|---|---|---|
| **COMPUTE-only** | COMPUTE만(제출된 것 한정) | **~19.2%** | `bank_operator_replay` SOLID |
| **LOAD** | COMPUTE+FIND-enumerate+coverage | **~60% 상한** | +closed-world reach(69.2%) 회복 |
| **GAP 잔여** | (닫히지 않음) | **~40%** | open-world ASK + ⋈ 비결정 + gather |
- **정직 게이트**: LOAD가 확립수치와 크게 어긋나면 harness 결함 의심·per-case 정독([[08]]). aggregate→결론 직행 금지.
- GAP은 결정론으로 못 닫음이 **정상**(open-world=completeness-ASK·⋈=경계). LOAD가 make-or-break 상한.

## 4. 경계·비목표 (scope 규율)
- **NOT**: write 강제(설계 절대선·§write강제금지)·라이브 호출(=b·유료)·신규 도메인지식 코드화.
- **엔진 순수성**: tau2 임포트 없이 harness만 궤적 파싱. 엔진 = dict-in/dict-out.
- **caveat**: (i) universe_txns=완비열거 상한 근사(실제 list→detail가 그 txn 전부 surface함을 가정) (ii) id-correct free 가정(열거가 정답 record 노출) (iii) 궤적=frontier(gpt-5.2 등)≠라이브 32B. 이건 **상한**이지 라이브 예측 아님.

## 5. 산출·다음
- `bank_eplan_controller.py`(엔진+harness) · 오프라인 스모크 로그 · ceiling 표 → 원장 C93 후보.
- 확정 후 (b): 엔진을 `t2_eplan_patch` 배선진화(_note_eplan 3한계 해소)로 라이브 배선·gpt-5.2 user-sim·[[09]] 승인.
