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

## 2.5 ★배선 착수 — (a) 해소 + (b)(c) 정밀진단 (2026-07-17·지배 레버로 전환·C101)
> 사용자 지시 "지배 레버로 가라"(F3=작은 slice 종결) → coverage/reach 라이브 배선 착수. `t2_eplan_patch.py` 3한계.
- **✅ (a) 해소·오프라인 스모크 PASS**: `_extract_entity_ids` 비-JSON fallback(도메인일반 `entity_key: value` 텍스트 추출·entity_key=ABox·리터럴0·[[05]]). banking 거래목록 포맷문자열 → transaction_id 5개 추출(기존 ∅)·JSON 무회귀·빈입력 안전. commit.
- **★(b) 정밀진단(전수 검증)**: `get_credit_card_transactions_by_user`는 **1761/1761 호출 전부 user_id 키**(account_id 0). ⇒ banking detail_reader = **bulk user-keyed**(한 호출로 전 거래 surface) = retail per-entity 모델과 **구조적으로 다름**. 게다가 dispute write=`call_discoverable_agent_tool`(디스패처)로 transaction_id가 **nested args**. ⇒ **eplan entity_key=account_id가 어느 실제 엔티티와도 불일치**(reads=user-keyed·writes=transaction-nested). **결론: banking 지배 레버=account-level examined 아니라 transaction-level write-coverage**(surface된 disputable 거래를 다 dispute했나=C94 under-action).
- **★(b) 수정 방향(설계결정)**: ABox eplan `entity_key: account_id→transaction_id` + `list_enumerator: get_*_transactions_by_user`(bulk reader가 disputable 단위 열거) + `build_ledger_from_messages`가 디스패처 write의 **nested transaction_id 추출**(executed). 그러면 (a)파서로 listed=surface된 txn·coverage_gap=required−disputed=under-action 검출. credit/debit 2reader = eplan 확장 필요.
- **(c) confirm 게이트**: 미착수(write_tools=디스패처 nested라 gold DAG서 도출 필요).
- **✅ (b) 방향 오프라인 실증 PASS**(`bank_eplan_coverage_probe.py`·실 스캐폴드 PlanLedger+coverage_gap·963 실패 sim·DB-basis): **listed 채워짐 76%**((a)파서 작동·기존 ∅)·**coverage_gap>0 44%**(under-action 검출)·gap 1346 중 **surfaced-not-disputed 40%(COVERAGE·정보보유→H_min 리마인더 표적)** / not-surfaced 60%(REACH·FIND-enumerate). ⇒ **라이브 메커니즘이 banking under-action 올바로 식별 확증**·C94(COVERAGE 40.7%/FIND 27.2%)와 transaction 레벨 정합·라이브 배선 de-risk. caveat: not-surfaced 60%는 debit/디스패처 목록 포맷 파서 미커버로 일부 과대 가능.
- **다음(무료)**: (b) 커밋(ABox eplan entity_key→transaction_id·bulk reader=enumerator·`build_ledger_from_messages` 디스패처 nested write 추출) + (c) confirm게이트(write_tools=디스패처 도출). 그 후 라이브 e2e([[09]] 유료).

## 3. 유료 라이브 e2e 게이트 ([[09]])
- **금지**: 승인+scope 없이 유료 실행. 로컬 tau2 banking 도메인 부재→라이브=리모트/유료만.
- **make-or-break**: 배선 loop이 (i) 오프라인 결정론 상한(27.8~49%)을 라이브서 달성하나 (ii) **F3-enum inner router(NL→정규화)**를 소형모델이 하나 — 이게 진짜 시험대(C95 §5.6·offline 밖).
- **user-sim = gpt-5.2**([[47]]·handoff §4). scope=태스크수·trial수 승인 필요.

## 4. 산출·상태
- `bank_eplan_controller.py`: dag_plan/run_dag_replay + 산술 op(subtract/multiply/divide/round) + amount_difference ABox 규칙. selftest 8종 PASS.
- **다음(무료)**: `_note_eplan` (a)(b)(c) 한계 해소 코드(생성-레벨)·오프라인 스모크. **다음(유료)**: [[09]] 승인 후 라이브 e2e.
- caveat: 오프라인 상한=proxy(reward=DB)·pure-DB 23.5% blind·F3 라이브-gated.
