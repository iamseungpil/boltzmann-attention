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

## 3. [[05]] 포지셔닝 — 정당한 결정론-계산 offload (리뷰 R1 반영·2026-07-13)
> ★리뷰 교정(R1): 이전판 "offload 아님·검증만"은 **방어적 과소진술이자 틀린 경계**였다. thesis 본체가
> **결정론 분담(offload)**([[00]]): 작은 LLM이 decidable 부분을 결정론에 위임. decidable 정책값을 계산하는 건
> 위반이 아니라 **scaffold 날개 그 자체**. "값 못 건넴" 가짜 제약을 만들지 말 것.
- **compute는 값을 산출한다 — 그게 맞다.** decidable 정책-파라미터(liability=$50)를 결정론 계산해 채우는 것 =
  thesis의 결정론 날개(F2b 계산·F1 정책 적용). auth-게이트(pass/fail만)와 **억지 동형화 금지**: 게이트는 이산
  술어를 검증하고, compute는 값을 계산한다 — 둘 다 정당한 결정론 분담의 다른 형태.
- **진짜 [[05]] 경계 = 계산의 *입력 출처***:
  - ✅ **정당**: 에이전트가 *이미 수집한* 사실(ctx.records/params/user) 위 계산. DB를 새로 읽지 않음.
  - ⛔ **금지(autofetch·C34 규칙0)**: 미수집 DB record를 scaffold가 대신 fetch. (그건 절차-offload=진짜 위반.)
  - 기존 `calc_specs`(read record서 count/sum)가 정확히 이 규율 = 선례.
- **[[05]] 3질문**: (1) op=도메인일반·규칙/임계=A2데이터(특화 순증 아님) (2) 정책규칙=결정론 사실(유동판단
  동결 아님·decidable) (3) 도메인 행동 수행 = **예, 그러나 정당**(decidable 계산 offload=thesis 날개·autofetch와
  구분되는 건 *입력이 수집된 사실*이라는 점). ⇒ 클린.
- ⚠**결정론 op vs formalize op**: 규칙이 이산·decidable(liability·diff·bool_expr)이면 결정론 offload(값 산출).
  규칙이 NL-판단(account_class·"가이드라인 따라")이면 op=**formalize**(LLM 서브콜)=learn 날개. A2가 구분 선언.

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

## 5c. ★★재설계 피벗 — reference-filter 레버 (사용자 지시 "참조 reach 축"·2026-07-13)
> STEP 0(§8-0) 판정: compute-alone=7.5%(미달). 재설계 = 참조/reach 축. **참조-ID(transaction_id·account_id·
> card_id·card_last_4)가 hard-core param 실패의 지배 버킷**(~8500). 이 축을 [[08]] 3분·필터가능성 정량함.

- **참조-ID 실패 3분**(`bank_reference_scope.py`): ⋈오선택(gold∈수집·여럿중틀림) **92.7%** · reach 4.2% · 날조 3.2%.
- **★"⋈=경계" 반증 — 대부분 필터가능**(per-case t085 + 정량): ⋈ 케이스서 user가 **금액+날짜 식별정보 제공 39%·
  부분 29%·무 32%** = **≥68%가 식별정보 보유**(무 32%도 merchant/서수 개연). t085: gold거래=user가 "11/05·ATM·
  $100부족"으로 유일 특정·수집 record 5건 중 필터가능. ⇒ **순수 F3 경계 아니라 F2b 필터**(에이전트가 필터 실패).
- **★레버 = reference-filter (keystone 엔진 동형·retail fexec 일반화)**:
  1. **formalize**(LLM 날개): user 발화 → 식별기준(date·amount·merchant·type) predicate. (retail fexec_filter_decide 동형)
  2. **deterministic filter**(엔진): 수집된 record(get_*_transactions 등) 위 filter → 매칭 record → 그 id. (수집사실
     위 계산·§3 정당·autofetch 아님)
  3. **verify/silent-repair**: 에이전트 id를 필터 id와 대조·불일치 시 결정론op면 silent 치환(R2).
- **사정권**: 참조-⋈(92.7% of 참조실패) × 필터가능(≥39~68%) = **hard-core 지배 버킷의 큰 몫**(compute 7.5% ≫ 능가).
  R4 게이트 통과 후보. 잔여(식별정보 無·다중매칭)=진짜 ⋈ 경계(수용).
- **op 추가**: `filter`(over records, by formalized predicate → return id) — 도메인일반. formalize=learn 날개.

## 5d. ★★축 종합 판정 — reference-filter가 고사정권 강레버 (2026-07-13·★파서오염 교정판)
> ⚠**교정(자기교정·[[08]])**: 초판 "banking=F3⋈경계 지배·필터 31.8%"는 **파서 오염**이었다 — 타 모델 record
> 포맷서 date/amount/type/description 미추출→None==None 허위동일(555/615 None·"57개 동일" 부조리로 발각).
> **파싱신뢰 케이스만(good field ≥70%)** 재측정이 정본.
- **★transaction_id ⋈ 결정론-필터 유일식별 = 83%**(전 15모델·n=798·파싱신뢰만·gemini/gpt/grok/opus/sonnet 전부
  80-88% 일관). 진짜중복(全필드동일) 17%·ASK가능 0(중복은 동일이라 ASK불가·소수).
- **★reference-filter = 고사정권 강레버**: 참조실패가 hard-core 지배버킷 · ⋈비율 ~93%(gold수집·오선택) × 필터가능
  83% ≈ **참조실패의 ~77%**. compute(7.5%)를 압도. **사용자 참조/reach 재설계 방향이 정답.**
- **레버 = formalize(user발화→식별기준: date·amount·merchant) → 결정론 filter(수집record) → id (keystone 엔진·retail
  fexec 일반화·§5c)**. 잔여=진짜중복 17%+범주분류(formalize).
- **★진짜중복 17%의 정체(사용자 질문 계기·per-case t084 전수)**: transaction_id만 다르고 date/amount/desc/type/status
  *완전 동일* record 2건(중복청구). **속성으로 구분 불가·ASK도 user가 구분 못함**. 위치도 clean rule 아님(gold=나중
  66%·먼저 34%·정확2쌍 n=171). ⇒ **결정불가 잔여(벤치 인디터미너시·완벽에이전트도 못맞힘)**. 위치휴리스틱("나중")
  =부분(66%)이나 34% 오답으로 순효과 불명. **레버 정직한 천장 = 83%**(결정가능부)·17%=수용.
- **결정 필요**: **(a) reference-filter 레버 구축**(고사정권·R4 게이트 통과·filter op+formalize+ABox) 권장 /
  (b) compute는 소슬라이스로 병기 / (c) 진짜중복 17%=수용 or 위치op.
- ⚠**caveat([[08]]·과확신 방지)**: (1) **83%=파싱신뢰 부분표본**(798/3097·26%·나머지 74%는 모델포맷 파싱실패로
  스킵·표본편향 가능·완전검증엔 포맷별 파서 필요) (2) **83%=결정론-필터 유일식별 *천장***(gold record가 기준으로
  유일하다는 것)·**formalize half(user발화→date/amount/merchant 기준 추출) 오차는 별도**·전체 레버=formalize×filter
  < 83% (3) per-case 정독 t085(ATM 필터가능·CityFit 중복=⋈) 1건뿐·추가 정독 필요. ⇒ **강한 후보이나 [M] 잠정**·
  다음=포맷-강건 파서 or 라이브 formalize 검증으로 확정.

## 6. Nested-arg 배선 (핵심 신규·엔진 확장)
- 문제: 파라미터가 `call_discoverable_agent_tool.arguments`(JSON 문자열) **안**. 기존 resolve_write는 top-level만.
- 설계:
  1. **감지**: am.tool_calls서 `call_discoverable_agent_tool` 호출 → `discoverable_tool_name`으로 compute_ops
     prefix 매칭 → 대상 param 목록.
  2. **ctx 구축**: `params`=nested arguments(파싱)·`records`=이전 성공 tool-result 파싱·`user`=발화값.
  3. **compute+대조**: 각 param에 apply_op → 계산값. 에이전트 nested값과 비교(수치=tol·범주=eq).
  4. **★기본 메커니즘 = 결정론 op는 silent-repair(리뷰 R2 반영)**: 불일치 시 scaffold가 **계산값으로 nested를
     제자리 치환**(대화 무교란·replay-clean·T5-C 패턴). 근거(R2): 발견이 "frontier조차 이 값을 *계산 못 한다*"
     이므로 deny+regen("50이어야 함")은 (a) 값을 우회로 건네주는 셈 (b) C62 대화-교란을 부른다. "규칙만 알리고
     모델이 계산"은 작동 안 함(모델이 못 하니 실패). ⇒ **결정론 op(lookup/diff/bool_expr·확정값)=silent-repair가
     정석**(최대 결정론 offload·§3 정당). cap N/sim·Δspurious 계측(정답 param 오치환 0 목표).
  5. **formalize op·불확실 op만 deny/ASK**: LLM 서브콜 값이거나 신뢰도 낮으면 silent 금지 → deny-피드백 or ASK.
  6. formalize op는 서브콜(fexec 동형)·실패=None=미개입(안전).
- 위치: `t2_gate_patch` T2_RESOLVE 블록 내 resolve_write 직후(Lever 4 recommendation과 동렬)·별도 함수
  `resolve_compute_params(am, msgs, a2, ...)`. silent 치환은 am.tool_calls의 nested arguments를 직접 수정(기록
  응답 커밋 전)·replay-clean(C62 감사 준수).

## 7. 파라미터 커버리지 (지배 파라미터 → op·근거 데이터)
| 파라미터 | 오류수 | op | gold 값형 |
|---|---|---|---|
| customer_max_liability_amount | 602 | lookup_table(days) | 50/500/412.88 ⚠95%는 역산·§8-2 gold-blind=89.4% |
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
### 8-0. ★STEP 0 = compute-사정권 정량 — ✅실행 완료·판정 (리뷰 R4·`bank_compute_scope.py`·2026-07-13)
- **측정(traj 무료·hard-core 45태스크)**: 우측도구 호출된 실패 param 유형별:
  | 유형 | 비중 | 처방축 |
  |---|---|---|
  | **참조/ID**(transaction_id·account_id·card_id·card_last_4·날짜·disputed_amount) | **~지배(8500)** | **F3 ⋈(경계)+reach+prov** |
  | 범주 분류(contacted_merchant·dispute_reason/category·transaction_type·pin·account_class·card_action) | ~4900 | 의미/formalize(learn) |
  | 불리언 정책(eligible/provisional_credit) | ~1500 | bool_expr/formalize |
  | **결정론 계산**(customer_max_liability_amount·amount) | **~1100 = 7.5%** | **compute(이 설계)** |
- **★판정 = 미달(R4 게이트 ≥30% 대비 7.5%)**: **순수 결정론-compute는 hard-core param 실패의 7.5%뿐**.
  banking hard core는 **⋈-참조(올바른 거래/계좌 지목=프레임 F3 경계)+범주분류(formalize)**가 지배·계산 아님.
- **⇒ 결론(정직)**: **compute-alone 키스톤은 무신호 예상**(라이브 G-vs-GR 유의차 어려움). compute는 작지만 실재하는
  슬라이스(frontier 못 여는 decidable). **의미있는 banking 이득 = ⋈-참조/reach + 분류(formalize)** — 결정론 compute
  단독이 아니라 그 축들의 결합. R4 리뷰가 7.5% 슬라이스 과투자를 막음. **다음 = 재설계**(§10 재검·⋈/reach 우선 or
  compute를 소규모 실증-only로).

### 8-1. ★gold-blind 저작 게이트 (리뷰 R3·[[11]] 순환 방지·L4C-R2 재발 방지) — ✅저작 완료 2026-07-14
- **liability 테이블(days≤30→$50)을 gold 값에서 역산하면 95%는 순환**([[11]] 위반).
- **규율**: op-스펙 임계/조건은 **KB 정책 문서(doc_bank_accounts_031류·Regulation E)서 eval-blind 저작** — 저작
  시 gold 값 보지 않음. **재현율은 blind 테이블의 *사후* 검증**(저작 후 1회 측정). blind 저작 provenance를 doc에 기록.
- **✅ 완료(2026-07-14·gold-blind)**: `banking_knowledge.gate.json` `compute_ops` 저작 — **liability**(doc_036/031:
  ≤2 business days→$50·≤60 days→$500·후→전액) + **provisional_credit_eligible**(doc_032: timely≤60d ∧ category∈5종 ∧
  written_statement). 엔진 `bool_expr` op 추가(도메인일반·3값논리) + lookup_table key-None abstain 수정. **provenance+caveat**
  (business-day≈calendar 근사·date기준=issue_noticed→now·account_status 조건 생략)=`_note_compute_ops`. **gold-blind 유닛
  `test_compute_params` 2/2 PASS**(기대값=정책서 유도·gold 안 봄). 재현율(gold 대조)=§8-2 미측정(저작엔 미반영).
- **미저작(다음 pass)**: card_action(doc_031/credit_014·3%뿐 저순위)·partial_refund·credit provisional 정밀조건.

### 8-2. ★★재현율 측정 (gold-blind 저작의 *사후* 검증·2026-07-14·`bank_compute_slice`류 인라인·[S])
> 저작한 blind op-스펙에 **gold 입력을 넣어 gold 값 재현** 대조(저작엔 미반영·[[11]]). 구 §7 "95%"는 **역산**(폐기).
- **★liability(customer_max_liability_amount·n=1109)**: **89.4%**(proxy T1=30) / 73.6%(정책-literal T1=2). 저작 오류 §8-2가
  교정: ①필드=**transaction_date→discovery_date**(issue_noticed 아님·스키마 확인) ②구조=**min(disputed_amount, tier_cap)**
  (clamp·412.88=min(412.88,500)이지 별tier 아님) ③$50 tier 임계=**tx→disc proxy**(정책 '2 business days *of statement*'는
  tx기준 미명시·statement_date 부재→billing-cycle proxy·데이터 [10,40) 전부 89.4%·특정 gold 역산 아님). 잔여 10.6%=proxy 미적합.
- **★debit provisional_credit_eligible(n=1353)**: **86.8%**. bool_expr ALL(timely=tx→disc≤60 ∧ dispute_category∈5종 ∧
  written_statement_provided). 잔여 13.2%.
- **credit eligible_for_provisional_credit(2552·최대)**: **미저작** — credit args가 `written_statement_provided` 부재라 조건셋
  상이(purchase_date/issue_noticed_date 사용)·별도 정책 pass 필요. 0%-abstain 스펙은 제거(노이즈).
- **★정직 종합**: gold-blind 재현 = **liability 89.4% · debit provisional 86.8%**(95% 역산과 달리 진짜 수치). 실효 compute 이득
  = 재현율 × slice(651·§14.8) × (id/gather/reach 선결). Δspurious(frontier 맞춘 param 오치환)=미측정(다음). business_days
  미구현(calendar 근사)·credit provisional 미커버가 상한 제약.
- 등급: 재현율 [S](궤적전수·gold-blind) · 임계 proxy [M](정책 미명시·데이터공백서 [10,40) 등가) · credit/card_action 미저작.

### 8-3. ★★Δspurious 측정 = 레버 선택성(§1.3 모트 실증·2026-07-14·[S] go/no-go)
> silent-repair가 frontier의 *맞은* param을 오치환하나. agent-correct(agent==gold) 케이스서 op≠gold = Δspurious(순손해).
| 필드 | agent-correct 중 오치환 | agent-wrong 교정(gain) | ★순효과 |
|---|---|---|---|
| **customer_max_liability_amount** | 27 / 431 (6.3%) | 375 / 414 (90.6%) | **+348 (강한 승)** |
| **provisional_credit_eligible** | 82 / 909 (9.0%) | 78 / 156 (50%) | **−4 (순손해)** |
- **★모트 실증(§1.3 "하나 사면 하나 판다")**: liability=agent 못함(49% 틀림)→compute 큰 이득(+348). provisional=agent 잘함
  (85% 맞음)→얻을 것 적고 오치환이 맞은답 깨서 **순≈0/음(−4)**. ⇒ **compute는 *선택적* 적용**: agent가 못하는 필드(liability)만·
  잘하는 필드(provisional)엔 금지. Δspurious=0 아님(6-9%)이나 gain이 압도할 때만 순양성.
- **⇒ 레버 확정 = LIABILITY compute 단독**(+348). debit provisional **드롭**(net−4). §14.8 "651 slice"는 provisional 이득 포함
  과대평가·실 net-positive는 liability 지배.
- **★spurious 정독 검증([[08]]·아티팩트 배제)**: liability 27 오치환 = 진짜 op 오류(예 gold=50·op=14.99·tx→disc=5·amt=14.99).
  원인=벤치 특유 **tier 불일치**($50 tier=flat 50 even amt<50 / $500 tier=min(amt,500)). 정책서 미유도·[[11]] 역산 금지라 미해소.
  ⇒ 6.3% spurious는 irreducible(정직)·단 gain 375≫27이라 **순 +348 견고**(결론 불변).
- 등급: Δspurious/순효과 [S](궤적전수·agent-vs-gold-vs-op 3자·spurious 정독) · "liability만 적용" [S].

### 8-4. ★liability flat 재구성 + business_days + credit 조사 (2026-07-14·사용자 지시)
- **★liability = flat 구조로 교정**(config 스윕 측정): {flat vs min}×{calendar vs business}×임계. **flat+calendar+T1=30 = 재현 94.7%**
  (min 89.4%↑)·**Δspurious 2.1%**(6.3%↓)·**순 +366**(348↑). 근거: $50 tier=flat 50("maximum liability IS $50" literal·gold=50
  even amt<50·§8-3 spurious 정독). min은 내 추론(can't-exceed-tx)이었으나 벤치 미적용→flat이 더 gold-blind. table=[≤30→50,≤60→500,else→amt].
- **★business_days op 구현**(`_business_days_between`·주말제외·days_between spec에 `business:true`): Reg E '2 business days' 정확계산용.
  단 liability 최적=calendar+T1=30(proxy)가 이미 94.7%라 business 우위 없음(스윕: biz+T1=5도 94.7% 동률). **일반 op로 유지**(향후용).
- **★credit 조사(사용자 '커버' 지시)**: credit dispute 유일 유의미 compute 필드=eligible_for_provisional_credit(21.6% 오답). 정책
  doc_015 5조건 중 account-age(credit_card_accounts.date_of_account_open 파싱가능)·category·amount≥25로 저작·측정 →
  **순 +13·0 spurious·단 degenerate**([[08]] 검증: 평가가능 919 전부 gold=False·True케이스 65% abstain). ⇒ **marginal**(liability +366 대비
  무의미)·account-record 파싱 인프라 대비 값 미미 → **[[13]] 미탑재**. 진짜 credit 이슈=card_last_4(19.5%)=**참조 문제(어느 카드)**·
  compute 아님(향후 reference-filter류 확장 후보). card_action(4.6%)·저순위.
- **⇒ compute 키스톤 최종 = liability 단독**(flat·94.7%·+366). credit/provisional/card_action 전부 데이터-주도 탈락.
- 등급: liability flat [S](스윕 측정) · business_days [D](구현·유닛) · credit marginal [S](degenerate 검증).

### 8-5. ★오프라인 통합 replay (배선 실발화 검증·무료·[[09]] 스모크 대체·2026-07-14)
> [[30]] 유닛≠라이브발화. 유료 스모크는 32B-reach 장애(handoff §1)로 firing 적어 낭비 → **실 궤적 replay로 배선 실발화 확인**.
- 실 frontier 궤적의 dispute 호출 메시지를 `resolve_compute_params`에 실 am/msgs로 통과: **755회 발화**(240 sim)·gold-검증가능
  발화 중 **교정→gold 일치 90.9%**(491/540·불일치 49=flat-nuance 잔여). ⇒ **배선이 실데이터서 정상 발화·정확**. 유닛(stub)+통합(real) 완비.
- **라이브 caveat**: 오프라인선 frontier가 dispute 도달해 755 발화. **32B 라이브는 reach 장애(handoff §1)로 firing↓** 예상 → 라이브
  keystone은 reach-가능 셋업(frontier or reach 레버 결합) 필요·[[09]] 승인. 오프라인 replay(755·90.9%·+366)가 정량 정본.

### 8-2. 파라미터별 검증
1. (8-1 blind 저작 후) 파라미터별 → **전 frontier gold 재현율**(liability=95%는 *역산*이라 **재저작 필요**·8-1 규율).
2. **Δspurious**: frontier가 *맞춘* param에 op/silent-repair 적용 시 오치환 0 확인(정답 안 깨야).
3. 목표: 사정권(8-0) param 각 ≥90% blind 재현 → 라이브 착수 자격.
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
- **retail 회귀**(R5): compute_ops는 신규 필드(미기재 도메인=무발동). 기존 calc_specs와 **초기=별도 유지**·단
  **op 커널은 공유**(t2_compute 단일)해 중복 op 정의 회피. 향후 calc_specs를 compute_ops로 흡수 가능(엔진 동일).
- **prefix-키 충돌**(R5): agent_tool_name prefix 다중매칭 시 param명 2차 필터·A2가 정확 prefix 선언으로 해소.
- **412.88류 잔여**: 특수분기(추가 필드)·완전커버 아님·정직 보고.
- **nested 교정의 대화-교란**(C62 regen 손상): deny+regen이 흐름 깨면 silent-repair(T5-C) 전환 고려.

## 11. 구현 순서 (리뷰 반영·2026-07-13)
0. **★STEP 0 = compute-사정권 정량**(R4·무료·최우선): hard-core param을 사정권/밖 2분. **유의미할 때만 이하 진행**.
1. op 확장: `bool_expr`·`formalize` + `test_compute` 유닛.
2. **gold-blind 저작**(R3): KB 정책 문서서 op-스펙 임계 저작(gold 안 봄) → 사후 재현율. liability=**재저작**(현 95%는 역산).
3. `resolve_compute_params` + nested 배선(`t2_gate_patch`) + **결정론 op=silent-repair 기본**(R2) + 유닛(stub 루프).
4. banking A2 compute_ops 저작(KB 정책서 조건 확정).
5. 오프라인 Δspurious 게이트 → 라이브 keystone(승인).

## 12. 산출물(예정)
- 엔진: `t2_compute.py`(op·있음) + `resolve_compute_params`(t2_gate_patch·신규) · 유닛 `test_compute`·`test_compute_params`.
- A2: `banking_knowledge.gate.json` `compute_ops` 블록. 데이터: `C:/tmp/traj/*_banking.json`(gold 재현 근거).

## 13. ★★오프라인 reference-filter REPLAY 결과 (2026-07-14·[S]/[D] 혼합·정본 수치)
> **측정 방식**: §5c 레버를 실제 검증. `bank_keystone_replay.py`(무료·로컬)가 17 frontier 궤적의 transaction_id
> ⋈ 오선택 케이스마다 **실제 A2 `reference_filter` 규칙**(gate json line 102-115)을 `t2_compute.apply_op(op=filter)`
> 엔진으로 replay. criteria(date/type/merchant/amount)는 **gold record서 파생 = perfect-formalize 천장**.
> **provenance**: `scripts/distill/tau2/bank_keystone_replay.py`(재현: `PYTHONIOENCODING=utf-8 py bank_keystone_replay.py`)·
> 데이터 `C:/tmp/traj/*_banking.json`(17모델)·A2 스펙 `a2/banking_knowledge.gate.json` `reference_filter`. bank_filter_repro와
> **동일 모집단**(hard-core pass≤10%·⋈ 케이스 = `action_checks[].action_match=false` + call_discoverable + transaction_id 상이).

### 13.1 수치 (n=853 ⋈ 오선택 케이스)
| criteria (perfect-formalize) | 교정성공(filter==gold) | filter오답 | 미해결→none | ★결정가능부 교정률 |
|---|---|---|---|---|
| **(a) date+type** (merchant 미사용·하한) | 640 (75.0%) | **0** | 213 (25.0%) | **91.6%** |
| **(b) date+type+merchant** (브랜드토큰) | 646 (75.7%) | **0** | 207 (24.3%) | **92.4%** |
| **(c) date+type+amount** (금액 formalize) | 699 (81.9%) | **0** | 154 (18.1%) | **100.0%** |
- **진짜중복 = 154 (18.1%)**: date/amount/type/description 全동일 record ≥2(중복청구) → `on_ambiguous=none`이 **정당 abstain**.
  완벽 에이전트·완벽 formalize도 못 맞힘 = 벤치 인디터미너시. **결정가능부 = 699**(= n − 진짜중복).
- **★핵심**: 비-중복 ⋈ 케이스(699)는 **date+type+amount 결정론 filter로 100% gold 도달**(0 오답). date+type만으로도 91.6%.

### 13.2 해석 (keystone 사정거리)
- **레버 실증**: frontier 17모델이 틀린 참조-⋈의 **81.9%(전체)/100%(결정가능부)**를 도메인-일반 filter op + A2 규칙만으로
  결정론 교정. §5c 레버가 hard-core 지배버킷(참조-⋈)의 큰 몫을 닫음을 정량 확인. (compute-alone 7.5% §8-0 대비 압도.)
- **merchant(어려운 NER)는 거의 무관**: (a) date+type만으로 결정가능부 91.6%·merchant 추가 +0.8pp뿐. 천장이 **저모호
  구조적 formalize**(날짜·type enum·금액)에 기댐 → formalize half 리스크 낮음(keystone 강화).

### 13.3 ★정직성 캐비엇 ([[08]]·과대주장 방지)
1. **②filter오답=0은 *구조적*** — criteria를 gold서 파생하므로 gold는 항상 자기 기준 만족→매칭셋 포함→유일매칭이면
   필연 gold. 이 replay는 **filter의 reach 천장**을 잴 뿐, **imperfect-formalize 하의 Δspurious(오치환율)를 재지 않는다**.
   Δspurious는 별도 게이트(§8-2)·**미측정**. "0 오답"을 Δspurious=0으로 오독 금지.
2. **전 수치 = perfect-formalize 천장**. 실제 레버 = formalize정확도 × 이 천장. formalize half 실측(user발화→criteria)은
   미측정(§0 선택·유료). 단 §13.2대로 어려운 formalize(merchant)는 거의 불요.
3. **미해결(비-중복 53)의 정체 = 다중매칭**(merchant 브랜드토큰 조잡·예 'PREMIUM PHOTO FRAMES'서 'PREMIUM'만
   뽑아 'SPOTIFY PREMIUM'과 충돌). amount 추가 시 전부 해소. 근본한계 아님.
4. **모집단 = 파싱신뢰 필터 미적용**(§5d 83%는 798 부분표본). 파싱갭은 미해결↑(보수적)·허위교정 아님(gold 항상 매칭셋).
   ∴ 75.0~81.9%는 **하한 성격**. merchant_phrase 변형(65%)은 stripped-token 비연속 substring 아티팩트 = 무효(단일 토큰이 강건).
5. **[S]/[D] 등급**: filter reach 천장 = [D](결정론 replay·재현가능)·진짜중복 18.1% = [S](per-case t084 전수·설계 §5d) ·
   **전체 라이브 교정률 = [?]**(formalize half + Δspurious + 다중턴 pass 미측정·§0 다음).

### 13.4 다음 (§0 handoff)
- (선택·유료) formalize half 실측: ⋈ user발화 batch → `formalize_reference_criteria`(gpt-4.1 소액) → criteria 정확도 → 전체 교정률 확정.
- 라이브 keystone: §1 장애물(32B가 dispute discovery 前 막힘·HANDOFF LATE-3 §1) 때문에 marginal 산출 불가 → 이 오프라인 replay가 정량 경로.

## 14. ★★formalize half 실측 — 리모트 32B e2e (2026-07-14·무료 on-prem·[S]추출/[M]앵커/[?]라이브)
> **방식**: `bank_keystone_formalize.py`가 ⋈ 케이스의 user 발화를 **실제 LLM formalize**(리모트 vLLM
> Qwen2.5-32B·`formalize_reference_criteria` 동형 프롬프트·localhost:8140·**API비용0**)로 돌려 식별기준을
> 뽑고 결정론 filter로 gold 도달 여부 측정. §13은 gold서 파생(perfect-formalize 천장)·이건 **user발화서 실제
> formalize** = 전체 교정률 = formalize정확도 × filter천장. provenance: `bank_keystone_formalize.py`·
> 입력 `sim_results/bank_xmatch_cases.jsonl.gz`(853·`bank_keystone_extract.py`)·출력 `bank_xmatch_formalize.results.json`.

### 14.1 ★구조적 발견 = banking hard-core ⋈는 ~100% 다중-dispute
- **단일-dispute 1 / 다중-dispute 852** (853 중). banking hard-core 참조문제는 본질적으로 **한 대화서 여러 거래를
  동시 dispute**하는 구조. ⇒ formalize는 "여러 공동-dispute 중 *지금 이* dispute가 어느 거래인가"를 먼저 풀어야 함.

### 14.2 수치 (C3 date+merch+amount·전 dispute셋 대조 재분류)
| 결과 | 수 | 의미 |
|---|---|---|
| 교정 (this action의 gold) | 28 (3.3%) | 이 action_check gold 정확 도달 |
| **오답 → 다른 *정당* dispute 대상** | **565 (66.2%)** | formalize가 실재 dispute 거래 찾음·단 이 action 아님 = **mis-pairing(앵커링)** |
| none (0/≥2 매칭) | 247 (29.0%) | 진짜중복 18.1% 포함 |
| **오답 → dispute셋 밖 record (진짜 오류)** | **13 (1.5%)** | formalize가 무관 record 잡음 |
- **★formalize가 정당 dispute 대상 도달 = 69.5%**(28+565) · **진짜-랜덤 오류 = 1.5%**.

### 14.3 해석 (formalize half의 진짜 binding)
- **32B formalize *추출*은 좋다**: date/merchant/amount NL 추출이 정당 dispute 대상 **69.5% 도달**·랜덤 오류 **1.5%뿐**.
  per-case 포렌치(§forensic): crit "FitLife Premium/11/10/89.99" → filter가 "FITLIFE PREMIUM MONTHLY 11/10 -89.99"
  정확 매칭. 필드추출은 병목 아님.
- **★binding = 다중-dispute 앵커링**(66.2% mis-pair): 레버의 **전역 formalize**(전 user 발화)가 "지금 이 dispute가
  어느 거래"를 못 가림 → 정당하나 *다른* 공동-dispute 거래로 감. = 참조/완결성 문제(§5c claim·[[45]] reach). 
  banking 지배 binding(reach/coverage/reference·C52·C71)과 정합.
- **Δspurious 안전**: 진짜-랜덤 오류 1.5%·나머지는 정당대상(69.5%) 또는 abstain(29%). formalize+filter가 무관 target을
  거의 안 만듦 = filter의 0-wrong-by-construction과 정합.

### 14.4 ★정직성 캐비엇
1. **per-action-check 페어링은 라이브보다 엄격** — 라이브 레버는 *각 dispute 호출 시점*에 돌아 agent가 *지금 거는*
   dispute를 교정. 내 오프라인은 formalize 출력을 특정 action_check gold에 (chosen-id 시간절단으로) 페어링 →
   dispute 순서 정렬이 어긋나면 mis-pair. ∴ **라이브 formalize-half 교정률 ∈ [3.3% (엄격페어링 하한), 69.5%
   (정당대상 도달 상한)]**. 참값은 dispute-순서 정렬에 의존·오프라인선 앵커 없이 확정 불가.
2. **천장(§13) 대비 gap = 필드값 아니라 앵커(어느 거래)**: gold파생 criteria(§13)가 100% 결정가능부 도달한 건
   *올바른 거래에 앵커됐기 때문*. 32B의 gap = 앵커링이지 date/amount/merchant 값 아님.
3. **레버 함의**: reference-filter가 다중-dispute 도메인서 유효하려면 **per-dispute 앵커**(지금 dispute의 특정 맥락)가
   필요. 전역 formalize는 "*한* dispute"는 풀되 "*이* dispute"는 못 풂. = 향후 레버 정련 방향(앵커 신호).
4. 등급: 필드추출 품질 [M](32B·포렌직) · 앵커링-binding [M] · 라이브 교정률 [?]([3.3, 69.5] 범위).

### 14.5 ★★[[08]] 포렌식 교정 = "⋈ 지배" 오염 발각·진짜 지배는 COVERAGE (2026-07-14·forensic-guard 촉발·방향전환)
> **경보**: [3.3,69.5] 좁히려 per-dispute 앵커 진단 중 궤적 정독서 **모집단 오염 발각**. task_086 정독: 16 "⋈ case"가
> **전부 동일 chosen(c90e2724)·distinct chosen=1**인데 gold 5개·user 발화 *"there's a limit? can't file all three"*.
> = **⋈ 오선택 아니라 COVERAGE 실패**(한도로 미제출). 추출이 `same[0]`(첫 호출)을 미제출 gold마다 페어링해 오분류.

- **★궤적 전수 재정량**(로컬 17모델·hard 936 실패 sim·agent 실제 호출 vs gold·`bank_xmatch_forensic.py`).
  스코프 주의: gold transaction_id는 3도구(credit_dispute 2552·debit_dispute 1352·rewards 244)에 걸침 — **disputes(credit+debit 결합) 한정** = 정본:
  | 분류 (gold 3904) | 수 | 의미 |
  |---|---|---|
  | agent 올바른 id 제출 | **2904 (74.4%)** | transaction_id 선택은 대부분 정확 |
  | **wrong id 제출 = 진짜 ⋈** | **159 (4.1%)** | 실제 오선택은 소수 |
  | **미제출 = COVERAGE** | **1000 (25.6%)** | 지배 실패 |
  - **★missed(1000) 분해**: **A.0제출(dispute 하나도 안냄=REACH/DISCOVERY) 804 (80%)** · B.한도언급 110 (11%) · C.부분제출후 미완(F4/F5) 86 (9%).
- **★결론(방향전환)**: **transaction_id ⋈는 banking 지배 실패가 *아니다*** — 제출 시 id 정확·진짜 오선택 4.1%뿐.
  **지배 = COVERAGE(25.6%)·그중 80%가 "dispute를 하나도 안 냄" = REACH/DISCOVERY 실패**(도구 unlock/discovery 체인 前 막힘).
  = **원장 C52/C71(banking binding=reach/coverage)·C76(compute param)·handoff §1(32B가 dispute 도달 0)과 3중 독립 수렴**.
  (⚠초판 C80 "⋈222/missed1121/27%"는 전-discoverable-도구 스코프 오염·위가 dispute-한정 정본.)
- **★오염된 선행 주장 교정**: C77("⋈ 지배버킷·82% filterable")·C78·C79의 ***prevalence*** 주장은 오염 집합(853) 위였음.
  - **살아남는 것**: filter 엔진의 **record 유일식별 능력**(C78 결정가능부 100%·gold record가 criteria로 유일)은 *데이터 사실*로 유효 — 단 사정권이 853이 아니라 **진짜 ⋈ 222**.
  - **철회**: "⋈가 hard-core 지배·reference-filter가 큰 레버"(C77 §5d·§8-0). reference-filter 사정권 = 222(작음).
- **★레버 재정렬**: banking 큰 레버 = **COVERAGE/completion**(요구 dispute 전부 제출·[[14]] E-PLAN·§1.4 F4), reference-filter 아님.
  단 F4 completion은 **write 강제 금지**(§1.5 Q5) → coverage 게이트는 "미제출 감지→ASK/재시도 유도"(read/plan 강제)로.
- 등급: 재정량 [S](궤적 전수·결정론 id-집합 비교) · "⋈ 비지배" [S] · reference-filter 사정권 159 [M].

### 14.6 ★★missed 지배원인 정밀화 = UNDER-action/over-deferral (도구발견 실패 아님) (2026-07-14·226 0제출 sim 정독+정량)
> missed(1000)의 80%=0제출(804·226 sim). "reach/discovery 실패"로 명명했으나 정밀 포렌직서 **재명명**.
- **도달 단계 분포**(0제출 226 sim): 검증前 stall 16% · unlock중 10% · **dispute도구 unlock했으나 조회前 53%** · 거래조회했으나 미제출 17% · 무행동 4%. ⇒ **70%가 dispute 도구를 이미 unlock**(도구 못 찾은 게 아님).
- **★종료 신호(하드)**: **100%가 user 턴서 종료**(user-STOP 만족/떠남 40% + user 대기 60%). = **agent가 마지막에 행동 안 하고 텍스트(질문/요약)로 user에 넘김**·dispute 자율제출 0.
- **precondition 요구 81%**(keyword proxy) · **diversion 17%**(결제·폐쇄 등 다른 action은 실행). per-case: task_041=카드 있는데 "끝4자리 필요"라 자기-차단→user "지금 없음"→STOP · task_045=결제/계좌폐쇄 하고 dispute는 미제출→user 만족 STOP.
- **★재명명 = UNDER-action / over-deferral**(도구발견 실패 ✗): 도구 도달했고 거래 조회했는데도 **필요정보 보유 상태서 precondition 되묻거나 diversion**해 dispute 미제출·user에 넘김. = 과확신의 **반대편 꼬리(over-asking/over-deferral)**·"act vs advise"(C74)의 agent-실행 dispute판.
- **레버 함의**: **action-required/persistence 게이트 for agent-실행 dispute**(정보 보유 decidable 확인→act 유도) — 단 §1.5 Q5(정보 진짜 부재 시 ASK 정당) 준수 → **"필요필드가 조회record에 있나" 결정론 확인 후에만 act-nudge**. = coverage 레버의 정확한 형태(reference-filter 아님·[[14]] E-PLAN 인접).
- 등급: 0제출 stage분포 [S]·"UNDER-action 재명명" [M](종료100%user턴=하드·precondition 81%=proxy)·act-gate 처방 [D설계].

### 14.7 ★★act-gate 전제 REFUTED = 되묻기는 정당한 field-gathering·진짜 lever는 COMPUTE (2026-07-14·[[08]] 전제검증)
> §14.6이 "정보 보유 상태서 over-defer"로 명명하고 act-gate(행동 강제)를 제안. **구현 前 전제검증**([[08]]/[[03]]):
> gold dispute 액션의 **요구 인자 스키마**를 전수(n=3959)—전제 반증.
- **★gold dispute = 13-25 필드 요구**(단순 transaction_id 아님): user-제공 필수(contacted_merchant 100%·dispute_reason·
  resolution_requested·card_in_possession·pin_compromised·police_report_filed·written_statement_provided·phone/email/address 66%·
  issue/purchase_date) + **computed/정책**(card_action 100%·**provisional_credit_eligible 66%·customer_max_liability_amount 28%·
  partial_refund 9%**) + reference(transaction_id).
- **★전제 반증**: agent의 "되묻기"(card_last_4·"contacted merchant?" 등)는 **spurious 과요구 아니라 *진짜 필요 필드 gathering***
  (KB "the agent must gather comprehensive information" 준수). ∴ **act-gate로 행동 강제 = 필수필드 빠진 틀린 dispute = §1.5 Q5
  위반 = C74/C75(action-required 레버 prior-negative) 재현.** ⇒ **act-gate 폐기**([[08]] 전제검증이 구현 前 차단).
- **★재명명(정본) = HORIZON + COMPUTE**(over-deferral 아님): 각 dispute가 13-25 필드(다수 user-gather + 3-4 computed)·hard task가
  4-10 dispute → **field-gathering horizon**(C71 p_step^H·H≈8 정합)·대화가 user-STOP/예산 前 미완결 = coverage. 참조/발견/deferral 아님.
- **★진짜 decidable lever = dispute 인자의 COMPUTE 필드**(§7 원안 회귀): `customer_max_liability_amount`(28%·gpt55도 50 vs 100 틀림·
  lookup_table 95% 실증)·`provisional_credit_eligible`(66%·bool_expr)·`partial_refund_amount`(9%)·`card_action`(100%·정책 bool_expr).
  = **키스톤 compute 엔진(§4·§7)이 정확히 이걸 닫음**. frontier-irreducible·decidable·[[05]] 클린(수집사실 위 계산).
- **⇒ 방향 확정**: reference-filter(⋈ 4%)도 act-gate(refuted)도 아니라 **compute 키스톤(§7)이 banking의 유효 decidable slice**.
  다음 = compute 필드 실패율 실측(2904 id-correct 중 compute-param-fail분) + §8-1 gold-blind 저작 → 라이브.
- 등급: dispute-스키마 [S](전수 3959)·act-gate refuted [S]·compute lever [M](§7·liability 95% 실증·slice 미실측).

### 14.8 ★★COMPUTE slice 실측 = id-correct dispute의 22.4% (전체 gold의 16.7%)·verified (2026-07-14·`bank_compute_slice.py`·[S])
> §14.7 재지목(compute가 유효 lever)을 실측. id-correct dispute 쌍(agent가 올바른 transaction_id 제출·2904)의
> 필드를 gold와 대조 → compute-필드 오답률 + compute-closability(오답필드 전부 compute면 엔진이 fail→pass).
- **★compute-필드 오답률(id-correct 2904 중)**: **customer_max_liability_amount 51.1%**(450/880·최대 단일필드)·
  eligible_for_provisional_credit 22.3%(410/1840)·provisional_credit_eligible 14.7%(156/1064)·partial_refund 13.6%·
  card_action 3.1%(90/2904). **verified 실오류**(spot-check): liability agent None/0/47.5 vs gold 50·card_action keep_active
  vs cancel_and_reissue·provisional True vs False = norm 아티팩트 아닌 정책-계산 오류.
- **★compute-closability 분해(2904)**: pass(전필드정확) 1376(47.4%) · **compute만 오답=엔진이 닫음 651(22.4%)** ·
  혼합(compute+other) 374(12.9%) · noncompute만 503(17.3%).
- **★★compute slice = 651 dispute**(id-correct의 **22.4%** · 전체 gold-dispute 3904의 **16.7%**). liability 주도.
  = **키스톤 compute(§7)의 실측 사정권** — reference-filter(⋈4%)의 4배·frontier-irreducible(gpt55도 liability 51% 틀림)·decidable.
- **noncompute 잔여**(503+혼합374): card_last_4_digits(369·참조/도출)·pin_compromised(181·gather)·transaction_type(121)이 지배.
  card_last_4는 조회 카드record서 도출 가능성(별도 ref 레버 후보)·pin류는 user-gather(경계).
- **★캐비엇**: 651 = **compute가 gold를 재현한다는 가정 하 천장**. 실효 = 651 × §8-1 gold-blind 재현율(liability 현 95%는
  역산·재저작 필요). + id-correct(참조)·gather·reach가 선결(compound). 순 라이브 이득 < 651.
- **⇒ 종착 결론(forensic 전체)**: banking decidable lever 우선순위 = **① compute(651·16.7%·§7 엔진 有) > ② reference-filter
  (⋈ 159·4%) ≫ act-gate(refuted 0)**. 나머지(gather·reach·horizon)=능력/대화 축(scaffold 밖 or E-PLAN). **compute 키스톤이
  banking의 정답 slice** — 첫 설계(C76·§7)가 옳았고 ⋈ 우회(C77-79)가 오염된 곁길이었음.
- 등급: compute slice 22.4%/16.7% [S](전수·verified) · 실효상한 [M](gold-blind 재현율 의존) · 우선순위 [S].

## 15. ★★"거래 고정→파생" 아키텍처 검증 = 논리곱 collapse 시도의 정직한 결론 (2026-07-14·사용자 설계)
> 사용자 통찰: horizon=거대 논리곱(∧ 필드). "필드 하나하나 풀지 말고 ①거래(root) 결정론 고정 → ②나머지 slot-fill/link/compute
> 파생 → ③user-대화 필드만 남김"으로 N을 줄여 곱을 회복. 상관(필드가 거래 이해서 뭉침)을 역이용(root 고정→다발 결정화).

### 15.1 아키텍처 (사용자 3단계·[[05]] 클린·[[00]] 논제)
```
1단계 [고정]: 어느 거래냐 = 결정론 reference-filter (⋈·유일 hard step)
2단계 [파생]: A2 규칙으로 거래→필드 (slot-fill 거래레코드 · link card_id→last_4 · compute liability)
3단계 [gather]: user-대화 필드만 (~8개·짧아진 horizon)
```
엔진=일반(slot-fill/link/compute), A2=거래→필드 매핑. 논리곱 처방 중 "종속항 접어 N 축소".

### 15.2 ★검증 = per-레코드 slot-fill 일치율 (id-correct·거래레코드 존재·[S])
| 필드 | 거래레코드 slot-fill→gold 일치 | agent 오답률 | 파생 판정 |
|---|---|---|---|
| transaction_date | **100%** | 0.2% | 깨끗 |
| account_id | **100%** | 0% | 깨끗 |
| disputed_amount | 67.6% | 7% | 부분(partial dispute/format) |
| transaction_type | 29.5% | 11% | enum 불일치(매핑 필요) |
| card_last_4 | 링크 messy(외래키 부재) | 19.5% | 파생 난 |

### 15.3 ★★잔인한 정렬 = slot-fill 레버 순이득 ~0 (moat 재현·핵심 결론)
- **파생 깨끗한 필드(date·account 100%) = agent가 이미 맞히는 것**(0.2%·0%). **agent가 틀리는 필드(type 11%·card_last_4
  19.5%) = 파생 안 되는 것**(30%·messy). ⇒ **slot-fill 레버는 fix할 게 없음**(고칠 필드는 파생불가·파생가능은 이미정답).
  = compute(+1.3pp)·provisional(−4)·§8-3 모트의 재현: **레버는 agent가 못하는 곳에만 이득인데, 그곳은 결정론화가 어려운 곳.**
- **∴ Stage 2(파생 slot-fill) = 대부분 redundant.** 별도 slot-fill 엔진 미구현([[13]]·순이득 근거 없음).
- **★가치는 Stage 1에 집중**: 거래 고정(⋈)이 상관 다발의 root — 틀리면 딸린 필드 *같이* 틀림. root 하나 잡는 게 핵심이고 = **이미
  구현된 reference-filter**(silent-repair). 즉 사용자 아키텍처는 **reference-filter가 왜 핵심 레버인지를 논리곱으로 재증명**.
- **한계 재확정**: horizon collapse는 부분적. ①거래-파생 필드는 이미 agent가 처리·②agent 오답은 파생불가(hard NER/link)·
  ③user-대화 ~8필드 irreducible. ⇒ **banking pass의 큰 상승은 여전히 구조적 난제**(scale 영역·§8-5·C71).
- 등급: slot-fill 일치율 [S](per-레코드) · "파생=이미정답/오답=파생불가 정렬" [S] · Stage2 redundant [M].

## 16. ★★H_min 방법 확정 + banking 측정 = gather-horizon의 진짜 정보는 ~15bits (2026-07-14·사용자·[S])
> 사용자: horizon 실패=곱≠1. verify-or-ASK로 각 인자를 1(결정론 verify)/externalize(ASK)로 몰면 silent p^N 붕괴 해결.
> 관건=H_min(꼭 ASK할 irreducible bits) 최소화. 방법 확정+측정.

### 16.1 H_min 계산 방법 (확정·threshold 명시)
필드 f마다:
1. **derivable?** 레코드 slot-fill/policy compute 재현율 ≥ **τ_derive(0.95)** → **DERIVE**·H_min+=0 (verify/compute/link, ASK 불요).
2. else **mode 확률 ≥ τ_default(0.9)** → **DEFAULT**·assume mode·잔여 소량 (신호 있을때만 확인).
3. else → **ASK(H_min)**.
- **H_min = ASK 집합의 JOINT 엔트로피**(상관 제거)·대화당 공유필드 amortize.
- **★τ_default = 위험-길이 knob**: 높이면 ASK↑(안전·길다)·낮추면 DEFAULT↑(짧다·비-default시 오류). 최적=결정론적 VOI(default iff P(비-default)×cost_err < cost_ask).

### 16.2 banking 측정 (per dispute-type·joint 엔트로피·[S])
- 필드별 marginal H: transaction_id 5.95·dispute_category 2.87·discovery_date 2.66·dispute_reason 2.49·pin 1.22·contacted 0.76-0.97
  · liability 0.60·issue_noticed 0.31·card_in_possession 0.26·police_report **0.00**(100% false)·written_statement **0.00**(100% true).
- **DERIVE**(정보량0·거래/정책/링크 결정): id·date·amount·type·card·account·PII·liability·provisional·partial·card_action(←category). ~18필드.
- **DEFAULT**(mode≥0.9): police_report(false)·written_statement(true)·card_in_possession(true·95.6%)·issue_noticed(today·95.5%)·resolution(full_refund·92.9%). ~5필드.
- **★ASK(H_min·joint)**: **DEBIT 4.27 bits**(category+pin+contacted+discovery·상관 3.2bit 절감) · **CREDIT 2.60 bits**(reason+contacted).
- **★대화 H_min ≈ 2.6-4.3 × H(4.2) ≈ 11-18 bits ≈ 잘 고른 질문 5-7개.** vs naive 26필드=수십~수백 bits → **~5× 이상 축소**.

### 16.3 ★전략적 재해석 (gather-horizon 비관 재반전)
- agent의 gather-horizon 실패는 **정보량이 커서가 아님**(실제 ~15bit) — DERIVE 안 하고 재질문·DEFAULT 안 쓰고 다 물어 **대화 부풀림→user-STOP 前 미완**.
- ⇒ **DERIVE+DEFAULT+VOI-ASK로 N_eff를 H_min(~7질문)까지 접으면 완주 = non-scale 레버.** "gather-horizon=scale영역"(§8-5/C82) **부분 철회** — irreducible은 ~15bit(작음)·나머지는 효율/elicitation 문제(비-scale).
- verify-or-ASK 아키텍처(§15 이어): decidable 인자→결정론 verify→1(scale 불요)·H_min 인자→VOI-ASK→user가 1로. 잔여=**H_min(~15bit)×user협조**뿐.
- **caveat**: (a)엔트로피는 gold값 기준(perfect elicitor 하한·agent 불확실성은 별도) (b)DERIVE는 derive/link 기전이 실제 작동 가정(card_last_4 링크 messy면 ASK로 이동·H_min↑) (c)discovery_date ASK/default 경계 (d)조건부-context 엔트로피면 더 낮음(H_min 상한).
- 등급: H_min 방법 [D]·banking 측정 [S](joint 엔트로피·per-type) · "gather=효율문제" [M](derive기전 작동 가정).

### 16.4 다음 (verify-or-ASK controller)
H_min 확정으로 처방 명확: **①DERIVE(거래고정+slot-fill/compute/link·verify→1) ②DEFAULT(저엔트로피 assume+신호확인) ③VOI-ASK(H_min ~7질문만·최적순서·confident시 중단)**. 이게 completion을 verify-or-ASK로 강제하는 controller. scale 불요분(①②③) 실측=다음.
