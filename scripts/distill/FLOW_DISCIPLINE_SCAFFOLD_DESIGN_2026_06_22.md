# Flow-Discipline Scaffold — 설계서 (2026-06-22·목표 = 32B/72B → gpt-4.1 도달·학습0)

> 진입: `06-NOW`(단일 진실원) + `THIRTYB_VS_FRONTIER_GAP §6c`(both-fail 11 분석) + `05-fixed-vs-variable`(불변).
> 사용자 순서지정: **1.00(top-frontier) 도전 보류 → 먼저 32B(또는 72B)를 gpt-4.1 능력에 도달시킨다.** 이 설계서 = 그 첫 빌드.

## 0. 목표 & 성공기준
- **목표**: 32B-int8을 *학습 0·도메인-일반 scaffold만*으로 gpt-4.1 수준에 올린다. 측정 = retail e2e **pass^1: 32B floor 0.60 → target ≈ gpt-4.1 0.82**.
- **GO**: 32B+flow-scaffold pass^1 이 0.82 근방 도달(학습0) **AND** airline A2-swap만으로 동작(코드수정0·grep if-domain=0). 72B는 같은 scaffold로 상한 확인(반증: scaffold가 32B만 돕고 72B 무효면 = 32B-특이 보철).
- **NO-GO / 정직경계**: 게이트가 닫는 부분이 *premature/over-action*(아래 §3 비커버)면 = scaffold로 못 닫는 flow-판단 → learn/planning 잔여로 분리 기록.

## 1. 증거기반 — 32B→gpt-4.1 갭의 정체 (재능력 아닌 *첫시행 신뢰성*)
- 32B pass^1 0.60 / **pass-any-3 0.77 ≈ gpt-4.1 0.82** (`§35`·`THIRTYB §1`) ⇒ 갭 = *첫시행 완수율*.
- both-fail 11 中 32B 실패유형(`THIRTYB §6c`·실 e2e 궤적):
  - **eligibility 위반 / 잘못된 도구**: pending 주문에 `exchange`(T109/110)·processed 주문에 `modify`(T20) → 도구 에러 → **루프/too_many_errors 종료**(§35: 7B 36-43회·32B도 잔존).
  - **id 환각**: product-id를 item_id로(T21 `1656367028`).
  - **grounding 방향역전**: old/new 주소 반대(T109/110).
  - **중복·포기**: 같은 write 재시도(자동에러)·no-write.
- = §35 "7B→32B 갭=DB-state/flow/복구·too_many_errors 루프"가 *32B→gpt-4.1에 잔존*(작아졌으나 첫시행서).

## 2. ★메커니즘 확인 (도구 소스·설계 정정의 근거)
`src/tau2/domains/retail/tools.py` 직접 확인:
| write 도구 | 내부 precondition | 성공 후 status |
|---|---|---|
| modify_pending_order_items | `status=="pending"` | `"pending (item modified)"` |
| exchange_delivered_order_items | `status=="delivered"` | `"exchange requested"` |
| return_delivered_order_items | `status=="delivered"` | `"return requested"` |
| cancel_pending_order | `status=="pending"` | `"cancelled"` |
- ⇒ **도구가 이미 (a) wrong-status write를 에러로 거부 (b) 성공 후 status-lock(둘째 write 자동 에러).**
- **함의 1 (commit-once 불요)**: 중복 write는 도구 status-lock이 이미 막음 → 별도 commit-once gate = 대부분 redundant. **설계서서 제외**(과설계 회피).
- **함의 2 (scaffold의 진짜 가치 = pre-execution steering)**: 도구 에러는 *사후*·암호적("non-pending")이고 error-budget을 태우며 32B를 루프로 몬다. scaffold가 *실행 전* 같은 조건을 검사해 **방향지시 메시지**("이 주문은 delivered다 → modify 말고 exchange/return을 써라")로 변환하면: 에러예산 보존 + wrong-tool 루프 차단 + 올바른 도구로 유도.
- **함의 3 (premature-write 비커버)**: T2/T100/T105 실패 = *첫* write가 미완성/조급했고 그게 commit돼 lock. 손상은 둘째가 아니라 첫 write → precondition/commit-once로 **안 고쳐짐**. = gather-fully-before-write 판단(§3 비커버·learn/planning 잔여).

## 3. [[05]] 결정질문 — 행동 전 명시 (필수)
1. **scaffold/A2 도메인-특화 *순증*?** — 엔진 추가(preconditions kind·steering)=도메인-일반(도구명0). A2 추가(어느 status·어느 도구가 write·target-arg)=도메인 *정책 사실* → `a2/<domain>.gate.json`(=기존 GATE_SPEC와 동일 범주·auth/confirm/ownership과 같은 policy). **새 종류의 특화 아님**·단 A2-growth는 발생 → **transfer(airline) 측정으로만 정당화**.
2. **유동 판단을 결정론에 동결?** — eligibility(status==pending)=**decidable**(DB lookup·유동성 없음)·중복=decidable. **유동 판단 동결 아님**(=provenance/autofetch와 동일하게 offload-적합). ✓
3. **scaffold가 도메인 행동을 *수행*?** — **아니오. 가드(차단/유도)만**·write를 대신 안 함(autofetch와 달리 "수행" 없음). = 순수 사실-게이트. ✓
- ⇒ 결정질문 1만 yellow(A2-growth) → **airline transfer 측정이 GO 조건에 포함**(못 넘으면 도메인특화 의심). 2·3=green. **학습0·tau2 학습 없음**([[11]] 준수).

## 4. P1 설계 (코어·기존 인프라 재사용)

### 4a. Precondition-steering gate (신규 gate-kind `preconditions`)
- **엔진**(`gate_interpreter.py`·도메인 분기 0): `check()`에 `kind=="preconditions"` 분기 추가. 기존 ownership과 *동일 메커니즘*(resolver_path로 read-only getter 호출·error budget 무소비) 재사용 — owner_field 대신 임의 field 읽어 허용집합 membership 검사.
- **A2 스키마**(`a2/<domain>.gate.json` gates에 추가·도메인 사실):
```json
{ "id":"G5_STATUS_PRECONDITION", "kind":"preconditions",
  "predicate":"target order is in a state that permits this action",
  "checks":[
    {"applies_to":["modify_pending_order_items","cancel_pending_order"],
     "resolver_path":["order_id","get_order_details","status"], "allow":["pending"],
     "steer":"this order is not pending; for a delivered order use exchange/return instead"},
    {"applies_to":["exchange_delivered_order_items","return_delivered_order_items"],
     "resolver_path":["order_id","get_order_details","status"], "allow":["delivered"],
     "steer":"this order is not delivered; for a pending order use modify/cancel instead"}
  ]}
```
- **동작**: write 호출 전 resolver로 target status 읽음 → `allow`에 없으면 deny + `steer` 메시지(방향지시·render_recovery 확장). 도구 자체 에러보다 *먼저*·*명확히* → 루프/에러예산 절감. (resolver 캐시: ownership이 이미 같은 주문 fetch → 1회 read 공유.)
- **불변**: `allow`/`resolver_path`/`applies_to`=A2(도메인사실)·엔진은 "field 읽어 membership" 일반로직. `grep if-domain=0` 유지.

### 4b. Retry/loop controller 활성·검증 (기존 `T2_RETRY_CONTROLLER`)
- 이미 구현됨(`t2_gate_patch.py` rule①정확반복차단·rule②연속실패K guard+DIVERSIFY). **미검증=32B→frontier 레버로서.** P1서 ON + tune(`T2_RETRY_K`).
- 역할: precondition-steering이 *못* 막은 잔여 에러(예: 올바른 도구인데 인자 틀림)서 루프/포기 차단 = §35 too_many_errors 종료(32B 큰 사인) 직접 타격.
- §35c "retry=잘못된레버"는 *grounding* 맥락(autofetch가 대체). 여기 *완수/루프* 맥락은 재검토 가치(`THIRTYB §6c`).

### 4c. (선택) id-type 검증 — product-id-as-item-id
- 기존 GROUND `_key_tokens` 재사용: arg `item_ids` 값이 컨텍스트서 *오직* `product_id` 키 아래만 등장(토큰 불일치)이면 flag. provenance(값이 컨텍스트에 있음)는 통과시키므로 *타입* 검증이 추가축. **P1.5(저비용 추가)**·효과작으면 보류.

## 5. P2 (이후·gpt-4.1→Opus 잔여·이 빌드 범위 밖)
- **info-aggregate 엔진**: "가용 옵션 수"·tracking# = 카탈로그/주문 결정론 집계→모델 보고(autofetch 사촌·T2/3/4/103 = gpt-4.1 실패축). 32B→gpt-4.1엔 부차(32B는 eligibility가 먼저).
- **operand B2-resolve**: max/min 랭킹(T20/109)=결정론 resolve([[10]] B2). both-fail서 최소축(2/11)·우선순위 하/.
- **premature/over-action**(§3 비커버): gather-before-write 판단 = learn/planning 별도 질문.

## 6. 측정 계획
- **arm**(retail 114·pass^1·3 trial denoise): ① 32B floor ② 32B + precondition-steering(4a) ③ 32B + 4a + retry-controller(4b) ④ (선택) +id검증(4c). **gpt-4.1 0.82 = 기준선**.
- **메트릭**: (주) pass^1·(진단) 실패-census Δ = §6c 유형별(eligibility위반·wrong-tool·too_many_errors종료·id환각) 감소량(규칙-격리·global pass=보조·`RULE_LEVER §2`).
- **transfer(GO 필수)**: airline A2-swap(`a2/airline.gate.json`에 동형 preconditions checks 추가)·동일 엔진·동일 코드 → airline pass Δ. 도메인특화면 여기서 실패.
- **72B**: arm③ 72B = 상한·scaffold 일반성(32B-특이 아님) 확인.
- **데이터**: 32B floor=`on_n32int8_floor_retail`(기존)·gpt-4.1=`retail_gpt41_nogate`(기존). 신규 = scaffold-on 런.

## 7. 위험 & GO/NO-GO
- **R1 (A2-growth)**: preconditions specs = A2 비대 → ⑤일반화비용↑. 완화 = airline transfer 측정(못 넘으면 NO-GO·도메인특화 판정). 정직: status-policy는 진짜 도메인사실이라 A2가 맞는 자리(엔진 아님).
- **R2 (도구가 이미 막음 → scaffold 무효익)**: precondition이 도구 에러를 *앞당기는* 것뿐이면 pass 게인 0 가능. 검증 = arm② Δ. 게인의 출처가 "에러예산 절감+steering"인지 census로 확인.
- **R3 (premature-write 비커버가 천장)**: §3 손상이 첫-write라 게이트 무력 → pass 천장이 gpt-4.1 못 미침. 그럼 정직히 "scaffold 한계·learn 잔여" 기록(둘째기둥 경계).
- **R4 (resolver read 비용·에러)**: 매 write마다 get_order_details 1회(ownership과 공유). error budget 무소비(read-only)·캐시.
- **GO** = arm③ ≈ 0.82 AND airline transfer 양성 AND 72B 무붕괴.

## 8. 빌드 순서
1. `gate_interpreter.py`: `check()`에 `preconditions` 분기 + `render_recovery` steer 확장(엔진·도메인0). 로컬 스모크(retail+airline 동형).
2. `a2/{retail,airline}.gate.json`: `G5_STATUS_PRECONDITION` checks 추가(도메인 status 사실).
3. `--validate`(기존 키스톤 검증): 양도메인 PassA/B 무회귀 + grep if-domain=0.
4. 측정 드라이버: arm①~③ 32B retail(`overnight_*` 패턴·healthcheck·진행률 가시 [[30]]) + airline transfer.
5. 회수·census Δ → GO/NO-GO → (GO면) 72B 상한 + 사용자 리뷰.

**불변**: 학습0·tau2 미학습([[11]])·grep if-domain=0([[05]])·설계먼저([[03]])·진행률가시([[30]]). 권위 상위 = `RULE_LEVER_COST_EFFICIENCY_PROGRAM`(이 레버=scaffold 날개)·`THIRTYB_VS_FRONTIER_GAP §6c`(증거).
