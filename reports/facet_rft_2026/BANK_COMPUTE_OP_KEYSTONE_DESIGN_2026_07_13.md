# banking 정책-파라미터 compute-op keystone 설계서 (2026-07-13)

> [[03]] "설계 먼저". 구현(nested 배선 + op-스펙 확장) 전 설계. 근거 = C76(banking frontier-irreducible 격차 =
> 정책-구동 파라미터 계산/판정·decidable) + 사용자 아키텍처(엔진=일반 op + A2-인터프리터·전이=A2만).
> 실증 완료: `t2_compute.py` 일반 op가 `customer_max_liability_amount` gold **95% 재현**(§키스톤 proof).

## 0. 한 줄 (목표)
**gpt55(frontier 최강 37.4%)조차 틀리는 banking 정책-파라미터를, 도메인-일반 compute-op 엔진 + A2-선언 규칙으로
결정론 계산·검증해 닫는다.** 엔진/loop 소스 불변, banking↔retail 전이 = A2(compute_ops)만 교체. thesis 키스톤.

## 1. 배경 (C76 근거)
- frontier 17모델 전수(gpt-5.2 user-sim): banking pass 11~37%. **hard core 45/97 태스크 전 frontier ≤10%**.
- 지배 실패 = `call_discoverable_agent_tool` **nested 파라미터 오류**(도구 맞음·파라미터 틀림 4311·+틀린 도구명 4384).
- 틀린 파라미터 = **정책-계산/판정**: liability 602·provisional_credit 439·amount/apy·pin·card_action.
- per-case t085: gold 책임상한 50 vs gpt55 100(disputed 전액·정책 미적용). = decidable(정책규칙+수집데이터→계산).

## 2. 아키텍처 (사용자 설계 · [[05]]/[[11]] keystone)
```
  ┌─ 엔진(고정·도메인-일반) ────────────────────────────┐
  │  t2_compute.apply_op(spec, ctx)  =  op 라이브러리      │
  │   {const,ref,min,max,argmin,argmax,sum,count_where,   │
  │    diff,clamp,days_between,lookup_table,bool_expr,     │
  │    formalize}  ← 도메인 리터럴 0                        │
  │  + scaffold 인터프리터: A2 compute_ops 읽고 dispatch    │
  └───────────────────────────────────────────────────┘
  ┌─ A2 (도메인 데이터·유일 가변) ──────────────────────┐
  │  compute_ops[tool_prefix][param] = { op-스펙 }        │
  │  = "어느 도구의 어느 파라미터를 어느 op·어느 임계로"     │
  └───────────────────────────────────────────────────┘
  전이 retail↔banking = compute_ops 교체. 엔진/loop 소스 diff = 0.
```
- **원칙(사용자)**: A2에 도메인마다 필요한 내부 op(min/max/argmax/lookup 등) 선언·변경 가능. scaffold는 A2의
  op-스펙(설명)을 읽고 dispatch만. 도메인 특화 로직은 전부 A2.

## 3. [[05]] 포지셔닝 — 왜 offload(autofetch)가 아닌가
- **핵심 경계**: compute는 **에이전트가 *이미 수집한* 사실(ctx.records/params/user) 위에서만** 동작. DB를 새로
  읽지 않는다(=autofetch 금지·C34 규칙0 준수). 기존 `calc_specs`(read record서 count/sum)와 동일 규율.
- **fact-gate이지 절차-offload 아님**: 정책-결정값(liability=정책상한)을 계산해 에이전트 값과 **대조·틀리면 교정**
  = 컴플라이언스 검증(F1)·decidable 계산(F2b). 게이트가 auth를 검증하듯, 정책-파라미터를 검증. 값을 대신
  *생성*해 주입하는 게 아니라(그건 offload), 에이전트 값이 정책과 맞는지 **검증**하고 틀리면 규칙을 알린다.
- **[[05]] 3질문**: (1) op=도메인일반·규칙=A2데이터(특화 순증 아님) (2) 정책규칙=결정론 사실(auth-술어와 동형·
  유동판단 동결 아님) (3) 도메인 행동 수행 아님(검증+교정·에이전트가 재호출). ⇒ 클린.
- ⚠**주의 경계**: 규칙이 NL-판단(예: provisional_credit_eligible="가이드라인 따라 판단")이면 op=**formalize**(LLM
  서브콜)로 위임 = learn 날개. 결정론 op(lookup/diff/bool_expr) vs formalize op 구분을 A2가 선언.

## 4. Op 라이브러리 (기존 + 확장)
| op | 상태 | 용도(banking 예) |
|---|---|---|
| const·ref | ✅ | 고정값·수집 record/user 값 참조(disputed_amount·pin_compromised) |
| min·max·clamp·sum·diff | ✅ | amount_difference=diff·상한 clamp |
| argmin·argmax·count_where | ✅ | 최저가·집계 |
| days_between·lookup_table | ✅ | liability=lookup_table(days_between)·95% 실증 |
| **bool_expr**(and/or/not+compare) | **신규** | provisional_credit_eligible·card_action=정책 불리언식 |
| **formalize**(LLM 서브콜→값) | **신규** | NL-판단형 파라미터(account_class·가이드라인 판정)·learn 날개 |
- 신규 op 2종도 **도메인-일반**(연산자·서브콜은 엔진·조건/프롬프트는 A2).

## 5. A2 compute_ops 스키마 (제안)
```json
"compute_ops": {
  "file_debit_card_transaction_dispute": {          // agent_tool_name prefix (숫자접미 무관)
    "customer_max_liability_amount": {
      "op": "lookup_table",
      "key": {"op": "days_between", "a": "params.transaction_date", "b": "params.discovery_date"},
      "table": [{"cmp": "<=", "thr": 30, "result": 50},
                {"cmp": "<=", "thr": 60, "result": 500},
                {"result": {"op": "ref", "path": "params.disputed_amount"}}]
    },
    "provisional_credit_eligible": {
      "op": "bool_expr", "all": [
        {"ref": "params.written_statement_provided", "==": true},
        {"ref": "params.disputed_amount", "<=": 500}]     // 정확 조건=KB 가이드라인서 확정
    }
  },
  "submit_interest_discrepancy": {
    "amount_difference": {"op": "diff", "a": "params.expected_interest", "b": "params.actual_interest"}
  }
}
```
- **키 = agent_tool_name prefix**(엔진이 prefix 매칭). param = nested 인자명.
- 값 op-스펙은 §4 op 사용. ref 경로 = `params.*`(nested 인자)·`records[*].*`(수집)·`user.*`(발화).

## 6. Nested-arg 배선 (핵심 신규·엔진 확장)
- 문제: 파라미터가 `call_discoverable_agent_tool.arguments`(JSON 문자열) **안**. 기존 resolve_write는 top-level만.
- 설계:
  1. **감지**: am.tool_calls서 `call_discoverable_agent_tool` 호출 → `discoverable_tool_name`으로 compute_ops
     prefix 매칭 → 대상 param 목록.
  2. **ctx 구축**: `params`=nested arguments(파싱)·`records`=이전 성공 tool-result 파싱·`user`=발화값.
  3. **compute+대조**: 각 param에 apply_op → 계산값. 에이전트 nested값과 비교(수치=tol·범주=eq).
  4. **deny+교정**: 불일치 시 deny(feedback="{param}은 정책상 {계산값}이어야 함(현재 {에이전트값})·재호출").
     tool_call 앵커(nested 있으니 call 존재)·cap N/sim·Δspurious 계측(정답 param 오검 0 목표).
  5. formalize op는 서브콜(fexec 동형)·실패=None=미개입(안전).
- **주입 아님**: 계산값을 nested에 몰래 넣지 않음. 검증+교정만(=[[05]] fact-gate). (silent 교정 옵션은 T5-C
  silent-repair 패턴 재사용 가능하나 기본=deny+regen.)
- 위치: `t2_gate_patch` T2_RESOLVE 블록 내 resolve_write 직후(Lever 4 recommendation과 동렬)·별도 함수
  `resolve_compute_params(am, msgs, a2, ...)`.

## 7. 파라미터 커버리지 (지배 파라미터 → op·근거 데이터)
| 파라미터 | 오류수 | op | gold 값형 |
|---|---|---|---|
| customer_max_liability_amount | 602 | lookup_table(days) | 50/500/412.88 ✅95% |
| eligible/provisional_credit_eligible | 439+ | bool_expr(정책조건) | True/False |
| amount_difference | 295 | diff | 수치 |
| expected_apy | 243 | lookup_table(account_class) or ref | 수치 |
| apply_*_account_credit.amount | 289/283 | ref/diff | 수치 |
| pin_compromised | 206 | ref(user 발화) | no/yes_*/unknown |
| card_action | — | bool_expr(dispute_category별) | keep/cancel/close/freeze |
| account_class | — | ref(record)/formalize | 15 범주 |
| card_last_4_digits·transaction_id | 416 | ref(record) | 참조 |
- 정확 조건/임계 = **KB 정책 문서서 확정**(doc_bank_accounts_031 류·오프라인 저작).

## 8. 오프라인 검증 계획 (유료 前·[[09]])
1. 파라미터별 op-스펙 저작 → **전 frontier gold 재현율**(1109 liability=95% 완료·나머지 param 반복).
2. **Δspurious**: frontier가 *맞춘* param에 op 적용 시 오검 0 확인(정답 안 깨야).
3. 목표: 지배 5~8 param 각 ≥90% 재현 → 라이브 착수 자격.
4. 유닛: `test_compute` 확장(bool_expr·formalize·nested 파싱).

## 9. 라이브 keystone 계획 (표준·[[09]] 승인)
- 표준 gpt-5.2 user-sim·G(compute_ops off) vs GR(on)·표적=hard-core param 태스크·nt1.
- 판정: GR이 G 대비 **hard-core param 태스크 pass↑ ∧ Δspurious≤0 ∧ tme 안정**. = A2-only 전이가 frontier-격차를 닫나.
- 전이 증명: 같은 엔진·retail compute_ops(기존 calc_specs 흡수)·banking compute_ops만 추가 = diff A2-only.

## 10. 리스크 / 미해결
- **입력수집 의존(reach)**: compute는 에이전트가 날짜·필드를 수집해야 동작. 미수집이면 op=None(미개입). ⇒
  compute 이득 = 수집된 태스크 한정. reach는 별도(discovery)·이 설계 범위 밖.
- **formalize op의 [[05]] 경계**: NL-판단 param은 learn 날개. 결정론 op로 최대 커버·formalize는 최후.
- **prefix 키 충돌**: agent_tool_name prefix가 여러 도구 매칭 가능 → param명으로 2차 필터·A2가 정확 prefix 선언.
- **retail 회귀**: compute_ops는 신규 필드(미기재 도메인=무발동). 기존 calc_specs와 통합/공존 결정 필요(초기=별도 유지).
- **412.88류 잔여**: 특수분기(추가 필드)·완전커버 아님·정직 보고.
- **nested 교정의 대화-교란**(C62 regen 손상): deny+regen이 흐름 깨면 silent-repair(T5-C) 전환 고려.

## 11. 구현 순서 (설계 승인 후)
1. op 확장: `bool_expr`·`formalize` + `test_compute` 유닛.
2. 오프라인 저작+검증: liability(완)·amount_difference·provisional_credit·expected_apy 각 gold 재현.
3. `resolve_compute_params` + nested 배선(`t2_gate_patch`) + 유닛(stub 루프).
4. banking A2 compute_ops 저작(KB 정책서 조건 확정).
5. 오프라인 Δspurious 게이트 → 라이브 keystone(승인).

## 12. 산출물(예정)
- 엔진: `t2_compute.py`(op·있음) + `resolve_compute_params`(t2_gate_patch·신규) · 유닛 `test_compute`·`test_compute_params`.
- A2: `banking_knowledge.gate.json` `compute_ops` 블록. 데이터: `C:/tmp/traj/*_banking.json`(gold 재현 근거).
