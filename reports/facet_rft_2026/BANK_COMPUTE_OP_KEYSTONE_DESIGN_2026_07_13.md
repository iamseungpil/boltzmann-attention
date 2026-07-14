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

### 8-1. ★gold-blind 저작 게이트 (리뷰 R3·[[11]] 순환 방지·L4C-R2 재발 방지)
- **liability 테이블(days≤30→$50)을 gold 값에서 역산하면 95%는 순환**([[11]] 위반).
- **규율**: op-스펙 임계/조건은 **KB 정책 문서(doc_bank_accounts_031류·Regulation E)서 eval-blind 저작** — 저작
  시 gold 값 보지 않음. **재현율은 blind 테이블의 *사후* 검증**(저작 후 1회 측정). blind 저작 provenance를 doc에 기록.

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
