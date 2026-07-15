# (c) 전-액션 per-step DAG 컨트롤러 — 구축 + 라이브 배선 계획 (2026-07-16)

> 사용자 지시 (a)(b)(c) 중 (c). **구축·오프라인 검증 완료·유료 라이브 실행은 [[09]] 승인+scope 前 금지.**
> 엔진 = `bank_eplan_controller.py`(dag_plan/run_dag_replay·selftest PASS). [[05]] 도메인일반·[[14]] 결정론·리터럴0.

## 0. 무엇을 지었나
C90/C92/C94의 **균일 per-step 연산-loop**을 실제 컨트롤러로 구현:
```
dag_plan(sim): gold DAG의 매 미충족 스텝 → 연산 분류 →
  미호출 read  → FIND(강제열거·결정론)
  미호출 write → COVERAGE(H_min-강제·결정론)
  호출·오답    → 필드별 최악: COMPUTE(ABox규칙) / GET-⋈(id·decidable) / F3-enum(NL→정규화) / F3-judgment / GATHER
  all_closed = 모든 스텝이 결정론 closable (FIND/COVERAGE/COMPUTE/GET-⋈)
```
- outer(coverage across items)·inner(per-item operator)이 **하나의 walk**로 통일(사용자 통찰·C94).

## 1. 오프라인 replay 결과 (`--dag`·DB-basis 실패 4262·infra제외)
| 판정 | % | |
|---|---|---|
| **ALL-CLOSED (순수구조 결정론 극복)** | **27.8%** (관측가능 중 36.3%) | FIND/COVERAGE/COMPUTE/GET만으로 전 스텝 닫힘 |
| 잔여 inner-router 필요 | 48.7% | F3-enum 1743(지배)·GATHER 356·F3-judgment 292 |
| Blind (pure-DB) | 23.5% | action-check 없음·오프라인 밖 |
- **2 bound (정직)**: **27.8%(하한·순수구조·그라운딩 미인정)** ~ **49.1%(상한·C95 +GET그라운딩)**. 차이=data-field 그라운딩 크레딧. 컨트롤러는 무맥락 구조판정이라 보수적 하한.
- 잔여 지배 = **F3-enum(NL→정규화)** = 라이브 inner router가 풀어야 할 의미매핑(make-or-break).

## 2. 라이브 배선 계획 (t2_eplan_patch 진화·유료 前 문서만)
정본 엔진(`t2_eplan_patch.py`)의 `_note_eplan` 3 구조한계 해소가 배선 전제:
- **(a) 비-JSON tool 출력**: banking 도구=포맷문자열→`_extract_entity_ids` 파싱 실패=listed ∅. → 포맷-파서 추가(txn/account id 정규식·`bank_reach_forensic`의 TXN 패턴 재사용).
- **(b) per-entity reader 귀속**: `get_*_transactions`가 user_id 키·디스패처 내부라 entity_key 미갱신. → eplan spec `entity_key`를 account 열거→per-account 상세로 2단 배선.
- **(c) confirm 게이트 부재**: `_confirm_write_tools=∅`→L1/L2 deny 비발화. → write 도구 집합을 gold DAG서 도출·H_min 미충족 시 생성-레벨 리마인더(히스토리 커밋 금지·설계 §2 REPLAY_SAFE).
- **개입 = 생성-레벨만**(작업버퍼·설계 절대선): FIND-강제·coverage-리마인더·H_min-continue. **write 강제 금지**([[14]]).

## 3. 유료 라이브 e2e 게이트 ([[09]])
- **금지**: 승인+scope 없이 유료 실행. 로컬 tau2 banking 도메인 부재→라이브=리모트/유료만.
- **make-or-break**: 배선 loop이 (i) 오프라인 결정론 상한(27.8~49%)을 라이브서 달성하나 (ii) **F3-enum inner router(NL→정규화)**를 소형모델이 하나 — 이게 진짜 시험대(C95 §5.6·offline 밖).
- **user-sim = gpt-5.2**([[47]]·handoff §4). scope=태스크수·trial수 승인 필요.

## 4. 산출·상태
- `bank_eplan_controller.py`: dag_plan/run_dag_replay + 산술 op(subtract/multiply/divide/round) + amount_difference ABox 규칙. selftest 8종 PASS.
- **다음(무료)**: `_note_eplan` (a)(b)(c) 한계 해소 코드(생성-레벨)·오프라인 스모크. **다음(유료)**: [[09]] 승인 후 라이브 e2e.
- caveat: 오프라인 상한=proxy(reward=DB)·pure-DB 23.5% blind·F3 라이브-gated.
