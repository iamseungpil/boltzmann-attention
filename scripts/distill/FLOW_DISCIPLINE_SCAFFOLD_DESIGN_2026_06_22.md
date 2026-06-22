# Flow-Discipline Scaffold — 설계서 (2026-06-22·목표 = 32B/72B → gpt-4.1 도달·학습0)

> 진입: `06-NOW`(단일 진실원) + `THIRTYB_VS_FRONTIER_GAP §6c`(both-fail 11 분석) + `05-fixed-vs-variable`(불변).
> 사용자 순서지정: **1.00(top-frontier) 도전 보류 → 먼저 32B(또는 72B)를 gpt-4.1 능력에 도달시킨다.** 이 설계서 = 그 첫 빌드.

## 0. 목표 & 성공기준 (★리뷰#1 반영·2026-06-22 — 천장-정합 재설정)
- **목표(정정)**: 32B를 *학습0·도메인-일반 scaffold만*으로 **scaffold-addressable 실패 클래스(eligibility/wrong-tool·loop)를 닫는다**. "gpt-4.1 0.82 매치"는 **scaffold 단독 임계로 두지 않는다** — 데이터상 도달 불가(아래).
- **★왜 0.82는 scaffold 단독 임계가 아닌가(§1b census)**: 첫시행-신뢰성 복구의 상한 = 모델이 *애초에* 풀 수 있는 비율 = **pass-any-3 = 0.772 < 0.82**. 게다가 실패 155건 中 **scaffold-addressable(eligibility 16.1% + loop 12.9%) ≈ 최대 25%**·나머지 ~75%는 *틀린 write*(operand/premature/wrong-target=capability·gate 무력). ⇒ scaffold-only 천장은 0.772보다 *훨씬 아래*. 0.82는 learn(operand)+info까지 더해야 도달.
- **GO(재설정)** = ① arm②서 **eligibility/wrong-tool 실패 클래스 ≥50% 감소** ② arm③서 **loop-like 실패 ≥50% 감소** ③ **false-block rate ≈ 0**(도구보다 더 막지 않음·리뷰#2) ④ **airline A2-swap서 동일 클래스-감소**(transfer·코드0·grep if-domain=0) ⑤ **잔여(operand/premature/wrong-write/info ~75% + 0.772→0.82 capability) 정직 분해·learn/operand/info 기둥에 명시 배정**. pass^1 게인은 *부차 지표*(addressable mass에 의해 상한·소폭 예상).
- **NO-GO / 정직경계**: addressable 클래스조차 안 닫히거나(steer가 fail→success 전환 못 함) false-block net-harm이면 = scaffold 레버 음성. *premature/over-action*(§3 비커버)·operand·0.77천장 = 처음부터 learn/capability 잔여(scaffold 실패 아님).

## 1. 증거기반 — 32B→gpt-4.1 갭의 정체 (재능력 아닌 *첫시행 신뢰성*)
- 32B pass^1 0.60 / **pass-any-3 0.77 ≈ gpt-4.1 0.82** (`§35`·`THIRTYB §1`) ⇒ 갭 = *첫시행 완수율*.
- both-fail 11 中 32B 실패유형(`THIRTYB §6c`·실 e2e 궤적):
  - **eligibility 위반 / 잘못된 도구**: pending 주문에 `exchange`(T109/110)·processed 주문에 `modify`(T20) → 도구 에러.
  - **id 환각**: product-id를 item_id로(T21 `1656367028`).
  - **grounding 방향역전**: old/new 주소 반대(T109/110).
  - **중복·포기**: 같은 write 재시도(자동에러)·no-write.

## §1b. ★전수 실패-census (32B floor 114×3·리뷰#1 반영·`/tmp/rev_b.sh`·기존 데이터)
- pass^1 = **0.596** / pass-any-3 = **0.772** (천장 확인).
- 실패 trial **155건** 분해:
  - **종료 = 전부 `user_stop`** (too_many_errors=0). ⇒ §1 초안의 "too_many_errors 루프 종료" 주장 **정정**: 이 floor 데이터선 도구 에러를 user-sim이 보고 멈출 뿐 에러-cap 종료 아님 (§35의 36-43회는 7B/타 config 추정).
  - **eligibility/status 도구-에러 포함 = 16.1%** (scaffold steer-addressable 상한).
  - **loop-like(num_errors≥4) = 12.9%** (retry-controller-addressable 상한).
  - **no-write(write 전 포기) = 9%**.
  - **나머지 ~75% = write를 했으나 *틀림*** (operand/variant·premature/incomplete·wrong-target) = **gate 무력·capability/learn**.
- ★**함의**: scaffold-addressable(eligibility∪loop) ≈ **최대 25%**. = flow-discipline scaffold는 *소수 레버*. 32B→gpt-4.1 0.82 전체갭의 대부분은 *틀린-write*(operand/premature) = learn/operand 몫. **이 빌드는 그 25%를 닫고 75%를 정직 분해하는 게 임무**(둘째기둥 경계 instrument).

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
- **함의 2 (scaffold 가치 = pre-execution steering·단 좁음)**: 도구 에러("Non-pending order cannot be modified")는 *사후*고 32B를 wrong-tool 루프로 몬다. scaffold가 *실행 전* status를 읽어 **방향지시 메시지**로 전환 → 올바른 도구로 유도. ★단 §4a/§1b 정합: gate-deny도 `num_errors++`라 **에러예산 절감은 아님**·가치는 *메시지 품질*(fail→success 전환 가능성)뿐. 또 §1b census상 too_many_errors 종료가 없으므로(전부 user_stop) "루프 종료 방지"보다 "wrong-tool을 right-tool로 전환"이 정확한 메커니즘.
- **함의 3 (premature-write 비커버)**: T2/T100/T105 실패 = *첫* write가 미완성/조급했고 그게 commit돼 lock. 손상은 둘째가 아니라 첫 write → precondition/commit-once로 **안 고쳐짐**. = gather-fully-before-write 판단(§3 비커버·learn/planning 잔여).

## 3. [[05]] 결정질문 — 행동 전 명시 (필수)
1. **scaffold/A2 도메인-특화 *순증*?** — 엔진 추가(preconditions kind·steering)=도메인-일반(도구명0). A2 추가(어느 status·어느 도구가 write·target-arg)=도메인 *정책 사실* → `a2/<domain>.gate.json`(=기존 GATE_SPEC와 동일 범주·auth/confirm/ownership과 같은 policy). **새 종류의 특화 아님**·단 A2-growth는 발생 → **transfer(airline) 측정으로만 정당화**.
2. **유동 판단을 결정론에 동결?** — eligibility(status==pending)=**decidable**(DB lookup·유동성 없음)·중복=decidable. **유동 판단 동결 아님**(=provenance/autofetch와 동일하게 offload-적합). ✓
3. **scaffold가 도메인 행동을 *수행*?** — **아니오. 가드(차단/유도)만**·write를 대신 안 함(autofetch와 달리 "수행" 없음). = 순수 사실-게이트. ✓
- ⇒ 결정질문 1만 yellow(A2-growth) → **airline transfer 측정이 GO 조건에 포함**(못 넘으면 도메인특화 의심). 2·3=green. **학습0·tau2 학습 없음**([[11]] 준수).

## 4. P1 설계 (코어·기존 인프라 재사용)

### 4a. Precondition-steering gate (신규 gate-kind `preconditions`)
- **엔진**(`gate_interpreter.py`·도메인 분기 0): `check()`에 `kind=="preconditions"` 분기 추가. 기존 ownership과 *동일 메커니즘*(resolver_path로 read-only getter 호출·error budget 무소비) 재사용 — owner_field 대신 임의 field 읽어 허용집합 membership 검사.
- **★매칭 정합(리뷰#3 반영·리모트 tools.py 확인)**: 모든 write 도구 = **정확 매칭** `status != "pending"/"delivered"`(raise "Non-pending/delivered order cannot be ..."). ⇒ G5 `allow`도 **정확-매칭**으로 도구와 정렬(over-block 0=false-block 없음·도구가 거부할 것만 거부). 단 modify 성공 후 status=`"pending (item modified)"` → 도구·G5 둘 다 2차 modify 거부(commit-once 불요 재확인 ✓).
- **★steer 정합(리뷰#3 핵심 버그 픽스)**: 단일 steer 문자열 금지 — **현재 status 값에 따라 분기**해야 함(blanket "use exchange"는 `"pending (item modified)"`엔 오유도). A2 = status-class별 steer 맵:
```json
{ "id":"G5_STATUS_PRECONDITION", "kind":"preconditions",
  "predicate":"target order is in a state that permits this action",
  "checks":[
    {"applies_to":["modify_pending_order_items","cancel_pending_order"], "allow":["pending"],
     "resolver_path":["order_id","get_order_details","status"]},
    {"applies_to":["exchange_delivered_order_items","return_delivered_order_items"], "allow":["delivered"],
     "resolver_path":["order_id","get_order_details","status"]}
  ],
  "steer_by_status_class": {
    "delivered":"this order is delivered — use exchange_delivered_order_items / return_delivered_order_items",
    "pending":"this order is pending — use modify_pending_order_items / cancel_pending_order",
    "processed":"this order is processed (in fulfillment) and cannot be modified, exchanged, returned, or cancelled by an agent",
    "_acted":"this order has already been acted on (its status shows a prior modify/exchange/return/cancel) — do NOT retry; the requested change is already recorded or must be handled by a human"
  }}
```
  엔진: deny 시 현재 status를 읽어 → status가 "...(item modified)"/"...requested"/"cancelled" 등 *이미-행동* 패턴이면 `_acted` steer, 아니면 status-class 키로 steer 선택. **blanket-cross-tool 유도 금지.**
- **동작**: write 호출 전 resolver로 status 읽음 → `allow`에 없으면 deny + status-aware steer. (resolver 캐시: ownership이 이미 같은 주문 fetch → 1회 read 공유.)
- **★정직: 게이트 value = 메시지 품질뿐(에러예산 절감 아님·리뷰#3 census 정합)**: 도구가 이미 정확 에러("Non-pending order cannot be modified")를 내고, gate-deny도 `num_errors += 1` 하므로 **에러-비용 동일**. gate의 유일 이득 = steer가 32B를 *옳은 도구로 전환*시켜 후속 wrong-tool 루프를 줄임(eligibility 16% 클래스에서 fail→success 전환 가능성·단 입증 필요). = "신뢰성 복구"라기보다 좁은 "능력 유도".
- **불변**: `allow`/`resolver_path`/`applies_to`/steer맵=A2(도메인사실)·엔진은 "field 읽어 membership + status-class steer" 일반로직. `grep if-domain=0` 유지.

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

## 6. 측정 계획 (리뷰#2·#4 반영)
- **arm**(retail 114·3 trial denoise): ① 32B floor ② 32B + precondition-steering(4a) ③ 32B + 4a + retry-controller(4b) ④ (선택) +id검증(4c). gpt-4.1=`retail_gpt41_nogate`·floor=`on_n32int8_floor_retail`(둘 다 기존).
- **★1차 메트릭 = 규칙-격리 실패-census Δ**(pass^1 아님·`RULE_LEVER §2` "y=규칙격리 census Δ"): §1b 클래스별 — (a)eligibility/wrong-tool 실패 수 (b)loop-like 수 (c)no-write 수. GO 판정은 이 클래스 감소로(§0).
- **★false-block rate (리뷰#2·필수 신규 메트릭)**: arm②/③서 **gate-deny 했으나 그 write가 실제론 옳았던 비율** = scaffold가 도구보다 *더* 막은 net-harm. 측정 = deny된 (tool,args)를 gold write-set과 대조(옳은 write를 막았나) + floor 대비 *새로* 실패한 task(arm②서 floor pass였는데 fail). **GO 조건 ③ = false-block ≈ 0.**
- **★#4 census 구분(retry-controller)**: arm③서 loop가 줄 때 **success로 전환 vs 다른-실패로 전환**을 궤적서 구분(같은 task가 pass 됐나, 아니면 wrong-write로 끝났나). 후자면 = 능력갭(retry 무익) → loop 클래스도 일부만 addressable.
- **pass^1 (부차)**: 보고하되 GO 임계 아님(addressable mass에 상한·소폭 예상). gpt-4.1 0.82·pass-any-3 0.772 = 참조선.
- **transfer(GO 필수·리뷰#1 ④)**: airline A2-swap(`a2/airline.gate.json`에 동형 G5 + steer맵 추가)·동일 엔진·코드0 → airline서 **동일 클래스-감소 + false-block≈0**. 도메인특화면 여기서 실패.
- **72B**: arm③ 72B = 상한·scaffold 일반성(32B-특이 보철 아님) 확인.

## 7. 위험 & GO/NO-GO (리뷰 반영)
- **R1 (A2-growth)**: G5 specs = A2 비대 → ⑤일반화비용↑. 완화 = airline transfer(못 넘으면 NO-GO·도메인특화 판정). 정직: status-policy는 진짜 도메인사실이라 A2가 맞는 자리(엔진 아님).
- **R2 (도구가 이미 막음→무효익)**: §4a 정직노트대로 gate value=메시지 품질뿐(에러예산 절감 아님). pass 게인 0일 수 있음 → GO를 pass 아닌 *클래스-감소+fail→success 전환*으로 판정(§0).
- **★R3 (scaffold = 소수 레버·리뷰#1 데이터)**: §1b — addressable ≈ 최대 25%·나머지 75%=틀린-write(operand/premature)=gate 무력. ⇒ scaffold-only로 gpt-4.1 0.82 도달 **불가**가 예상값. NO-GO 아니라 *설계된 경계*: 이 빌드는 25% 닫고 75%를 learn/operand/info로 분해-배정(둘째기둥 boundary instrument). **0.82 미달=실패 아님**(target 재설정·§0).
- **R4 (false-block net-harm·리뷰#2)**: steer 오유도/over-block이 floor-pass task를 깨뜨릴 위험. 완화 = 정확-매칭 정렬(§4a)+status-aware steer + false-block 메트릭 필수. **false-block>0이면 NO-GO**(도구보다 더 막으면 안 됨).
- **R5 (resolver read 비용)**: 매 write마다 get_order_details 1회(ownership과 공유·read-only·error budget 무소비·캐시).
- **★GO(재설정)** = ①eligibility/wrong-tool 클래스 ≥50%↓ ②loop 클래스 ≥50%↓(且 fail→success 전환 입증·#4) ③**false-block ≈ 0** ④airline transfer 동일 ⑤잔여 75% 정직 분해. (pass^1 게인=부차·0.82는 scaffold 단독 임계 아님.)
- **NO-GO** = 클래스 안 닫힘 OR false-block>0 OR airline transfer 실패 OR 72B 붕괴.

## 8b. ✅ STEP 1-3 구현 완료 (2026-06-22·commit 8646636)
- **엔진**(`gate_interpreter.py`): `preconditions` 분기 + `pick_steer`(status-class·_acted 우선) + `resolve_field`(resolve_owner 일반화·status는 캐시금지 fresh read). grep if-domain/도구명=0(주석 제외).
- **A2**(`retail.gate.json` G5): 검증된 4 도구만(modify/cancel→pending·exchange/return→delivered)·address/payment 제외(status-무관→false-block 회피)·정확-매칭 도구-정렬.
- **로컬 스모크 PASS**: pick_steer 전 status-class(특히 "pending (item modified)"→_acted) + preconditions allow/deny + false-block 회피(resolver None / order_id 부재 → allow).
- **`--validate` 결과**: retail PassA=0·**PassB ownership over-deny=0(무회귀)**·G5 gold-replay deny=1(task64=benign·도구-정렬·gold가 같은 pending 주문에 exchange+modify 동시기재) / airline PassA/B=0(G5 부재 no-op·무회귀). validate를 precondition-aware로 분리(ownership PassB 깨끗).

## 8c. ⚠️★airline transfer 발견 (리뷰#1④ GO에 영향·정직 기록)
- **리모트 tools.py 확인: airline write 도구(book/cancel/update_reservation_*)에 status-precondition 전무**(유일 status 체크=flight 검색 availability). ⇒ **retail의 status-lifecycle precondition = retail-특유 정책 형태**. 엔진(pick_steer+kind)은 도메인-일반이나, **airline엔 동형 status-precondition 인스턴스가 없어 G5의 효과-transfer를 airline status-swap으로 *실증 불가*.**
- **그러나 airline에도 *attribute*-precondition은 존재**(정책: basic-economy는 항공편 변경불가·24h 무료취소·cabin 규칙). = 같은 `preconditions` kind로 인코딩 가능(resolver가 reservation.cabin/created_at 읽고 allow 판정). ⇒ **transfer 데모 경로 = airline attribute-precondition을 G5 동형으로 추가**(engine 코드0·A2만)·step4 prep. 이게 "kind 일반성"의 진짜 transfer 증거.
- **[[05]] 함의(정직)**: G5의 *retail status 내용*은 transfer 안 됨(retail 정책). transfer되는 건 **엔진 + preconditions kind**(airline attribute-precond로 실증해야). status-string 형태로 airline 무효는 도메인특화가 아니라 *도메인이 그 정책을 안 가짐*(airline엔 notice 게이트밖에 없는 것과 동형). GO④ 재정의: "airline에 attribute-precond G5 추가 → 동일 엔진으로 동작 + class-감소" (status-swap 아님).

## 8. 빌드 순서 (★리뷰 조건부승인: step1-3 즉시 OK·step4 전 #1·#2·#3 박기 = 완료)
- ✅ **리뷰 전제 3건 반영 완료**: #3(리모트 tools.py 정확-매칭 확인→steer status-aware·§4a)·#1(census→target 재설정·§0/§1b)·#2(false-block 메트릭·§6). step4 진행 가능.
1. `gate_interpreter.py`: `check()`에 `preconditions` 분기 + status-aware steer(`steer_by_status_class`·`_acted` 패턴). 로컬 스모크(retail+airline 동형).
2. `a2/{retail,airline}.gate.json`: `G5_STATUS_PRECONDITION`(정확-매칭 allow + steer맵) 추가.
3. `--validate`: 양도메인 PassA/B 무회귀 + grep if-domain=0 + **false-block 스모크**(modified-status 주문에 `_acted` steer 뜨는지·delivered에 right-tool steer).
4. 측정: arm①~③ 32B retail + airline transfer(`overnight_*`·healthcheck·진행률가시 [[30]]). 메트릭 = §6 클래스Δ + false-block + #4 전환구분.
5. 회수 → GO/NO-GO(§7) → (GO면) 72B 상한 + 사용자 리뷰. 잔여 75% = learn/operand/info 기둥 배정 문서화.

**불변**: 학습0·tau2 미학습([[11]])·grep if-domain=0([[05]])·설계먼저([[03]])·진행률가시([[30]]). 권위 상위 = `RULE_LEVER_COST_EFFICIENCY_PROGRAM`(이 레버=scaffold 날개)·`THIRTYB_VS_FRONTIER_GAP §6c`(증거).
