# 원인-근거 해결 레버 설계 — 클린 nt=4 실패 극복 (2026-07-07)

> **입력**: `CLEAN_NT4_FAILURE_FORENSIC_2026_07_07.md`(실패 버킷·기전·극복레버 초안).
> **목적**: 각 실패 원인에 대응하는 **결정론 레버를 구현-수준으로 설계**.
> **불변(설계 제약·절대준수)**: [[05]] 엔진=도메인-일반·정책/필드=A2(`a2/<domain>.gate.json`)·retail 하드코딩
> 0·ABox만 변경. **replay-safe**(REPLAY_SAFE_GATE_DESIGN): READ-증강(reads는 eval-replay서 skip=안전) 또는
> generation-level regen-gate(히스토리 clean)만 허용·WRITE 히스토리 오염 금지. [[13]] 결정론 먼저·genuine
> 잔여만 학습. [[03]] 설계 먼저·구현은 리뷰 후.

---

## 0. 설계 원리 — 두 메커니즘에만 얹는다
모든 신규 레버는 **기존 replay-safe 두 축의 확장**으로 한정(새 subsystem 금지):
- **(R) READ-증강**(`compute_facts`/`nested_candidate_summary`/`candidate_summary` 계열): READ tool 응답에
  결정론 계산·후보를 텍스트로 첨부. eval-replay서 READ는 skip → **replay-safe**. A2 `*_specs` 구동.
- **(G) generation-gate/controller**(`apply_gate_regen`): WRITE/종료 시점서 결정론 검사·deny→작업버퍼
  피드백+재생성. 히스토리엔 compliant만 커밋 → **replay-safe**. A2 `gates` 구동.

각 레버: **엔진 op(도메인-일반)** + **A2 필드(도메인 인스턴스)** + **replay 안전성** + **decidable core vs
genuine 잔여**(정직 분리) 명시.

---

## 1. 변형-선택 calc (within-order·최대 버킷 32B 21·14B 26) — 메커니즘 (R)

**원인(기전)**: 우주문·정확하나 `new_item` **변형을 오선택**(63: 2635 vs gold 3254). 유저 묘사(cheapest·
bigger·brighter battery>USB·i9/256GB·red)를 변형 item_id에 매핑 실패. present-nested가 변형을 *보여주나*
argmax/filter를 모델이 틀림.

**설계 — `select_specs`(R·nested present 확장)**: trigger READ의 변형집합에 대해 **결정론 선택-결과를 계산·주입**.
- **엔진 op(도메인-일반)**:
  - `argmax/argmin(field)` — cheapest/most-expensive/biggest/highest-X (price·capacity·resolution 등).
  - `filter(attr=val)` — red/leather/waterproof/i9/256GB (attribute 완전일치).
  - `rank(pref-order)` — "battery>USB>AC" 선호순위 → 가용 최상위.
  - `match-keep(field=source-value)` — "same size shoe"·"same color" → 원본 필드값 유지.
- **A2 필드**: `select_specs:[{trigger_tool, variant_field(=nested_field), id_field, criterion_map}]`.
  `criterion_map`=유저-표현 토큰→(op,field) 매핑(cheapest→argmin price·bigger→argmax capacity…). 도메인-일반
  토큰 사전 + 도메인 필드명만 A2.
- **주입 형식**(census 마커): `[VARIANT SELECT — deterministic]: for 'cheapest' → item_id=X (price=..)`.
  모델은 이 값을 복사(report-conversion). READ-증강이라 replay-safe.
- **★decidable core vs genuine 잔여**:
  - decidable: cheapest/most-expensive/biggest/same-X/색·소재·용량 완전일치 = 대부분.
  - **genuine 잔여**: 모호 NL→필드 매핑("bigger"가 capacity인지 size인지 도메인 애매)·복합 선호. → 소량.
    **여기만** 소형 학습 formalizer 후보([[13]] 최후·paper1 "learned path-selection residual"). scaffold는
    **전체 변형+속성을 명시 제시**해 이 잔여를 최소화(현재도 present하나 계산-주입이 추가 offload).
- **타깃**: 63·8·37·100·45(변형오선) 등. 우선순위 **1**(최대 버킷·순수 READ-증강이라 저위험).

---

## 2. Coverage controller (상태추적·32B 17·14B 16) — 메커니즘 (G)

**원인**: "**모든**/양쪽/각각 X" 요청에 일부만 처리(41: 2 주문 인지하고도 1개만). 전수-열거 부재.

**설계 — `coverage_specs`(G·종료-게이트)**: 모델이 **종료(stop/transfer)하려 할 때** 요청 범위의 미커버
엔티티가 있으면 deny→regen("아직 #W... 미처리").
- **엔진(도메인-일반)**:
  1. **scope 감지**: 최근 유저 발화에서 범위 양화사(all·every·both·each·모두·전부) + 대상 타입 검출(결정론
     키워드·도메인-일반).
  2. **엔티티 집합 열거**: A2 producer로 인증유저의 대상 집합(예: pending orders)을 결정론 fetch.
  3. **커버리지 추적**: 커밋된 히스토리서 실행된 write의 대상 order/item 집합.
  4. **종료-게이트**: scope 감지 ∧ (집합 − 커버) ≠ ∅ 이면 stop/transfer deny → "미처리: #W.. — 처리 후 종료".
- **A2 필드**: `coverage_specs:[{scope_tokens, entity_producer, entity_id_field, applies_when_tool_class}]`.
- **replay 안전**: 종료를 지연시켜 write를 더 유도할 뿐·히스토리엔 실행된 write만 → clean.
- **★decidable core vs 잔여**: 양화사+대상타입 명시 케이스=decidable. **잔여**: 암묵 범위("정리해줘"가 무엇을
  포함하는지) = 소형 학습/ASK. over-ask 방지 위해 **명시 양화사에만 발동**(보수적·false-block 회피).
- **타깃**: 41·103·20·74·98(부분)·112 등. 우선순위 **2**.

---

## 3. Cross-order present (⋈ 참조·32B 7·14B 10) — 메커니즘 (R)

**원인**: 주문을 잘못된 키로 식별(71: "최근"으로 골라 오선택·gold=DC주소 주문)·아이템을 잘못된 주문에
conflate(98: 다른 주문 아이템을 한 주문에). 현 present는 **한 주문 내부**만.

**설계 — `xref_specs`(R·present 확장)**: 인증-후 사용자-레벨 READ(get_user_details 등)에 **주문↔속성·
아이템↔주문 매핑을 명시 제시**.
- **엔진(도메인-일반)**: producer로 각 엔티티의 disambiguating 속성 fetch → 매핑 테이블 주입:
  - order→{address, status, item-summary} (71·109: 주소로 매칭).
  - item→order (98·107: 어느 주문에 무슨 아이템·conflate 방지).
- **A2 필드**: `xref_specs:[{root_producer, child_producer, present_fields, id_field}]`(order/reservation
  swap=필드만 교체).
- **주입**: `[ORDERS — match by these before any write]: #W1{addr:DC, items:[bike]} · #W2{addr:Charlotte,
  items:[lamp]}`. READ-증강·replay-safe.
- **★decidable/잔여**: 속성-매칭(주소·아이템)=decidable. 잔여: 유저 묘사가 어느 속성인지 모호=소량.
  user-sim 오확인(71)이 마스킹하는 케이스는 present가 올바른 후보를 강제 노출해 완화.
- **타깃**: 71·79·98·107·109·12. 우선순위 **3**.

---

## 4. Order-total calc (calc·32B 6·14B 3) — 메커니즘 (R)

**원인**: 주문 총액 오산 보고(67·68: $919.67 vs gold $829.43).

**설계 — `calc_specs`에 order-total 추가**(기존 calc 엔진 그대로): trigger=get_order_details, op=`sum`,
item_field=price(+할인/조정 필드 A2). 주입 `[COMPUTED FACTS]: order_total: 829.43`. 이미 있는 `compute_facts`
`sum` op 재사용·A2 spec만 추가. READ-증강·replay-safe. **decidable 100%**(단순 집계). 우선순위 **4**(저위험·즉효).
- 단서: gold 총액이 item-price 합과 다르면(할인·세금) A2에 조정필드 명시 필요 — 케이스 확인 후.

---

## 5. Feasibility/should-not gate (over-action·32B 4·14B 7) — 메커니즘 (G)

**원인**: gold=무write인데 실행(12: 질문만인데 return)·불가능 op 실행.

**설계 — 기존 `preconditions`/`constraints` gate 확장**:
- **불가능 op**: 부분취소·cross-order 결제 등 = precondition 위반 → 기존 게이트가 이미 block(확장=A2 checks 추가).
- **should-not-write(질문만)**: 유저가 write를 **명시 요청 안 함**인데 write 시도 → 어려움(intent). 보수적
  설계: **write 직전 "이 write를 유저가 명시 요청했는가"를 확인-게이트(G2 confirm 강화)**로 — 이미 G2가 confirm
  요구하나, "질문"에 대한 confirm이 write로 오해되는 케이스. **잔여(genuine intent)** = 소량·학습/ASK 후보.
- **replay 안전**: G(regen). **decidable**: 불가능 op·precondition. **잔여**: should-not(intent) 소량.
- 우선순위 **5**.

---

## 6. Orchestration loop/no-write (32B 13·14B 17) — 부분 (G·정직)
**원인**: 동일호출 반복(loop)·미실행(no-write). **혼합**(일부 결정론·일부 load).
- **loop-guard(G)**: 동일-args 반복 write를 재생성 유도(다양화). 단 과거 retry_controller가 예산소진으로
  **해로웠음**([[06-NOW]]) → **regen 방식(예산 1tick·§R1)으로 재설계 시 무해할 수 있음**·재측정 필요.
- **no-write**: joint-constraint서 멈춤(64) = §1 변형-select가 후보를 계산해주면 완화. 순수 orchestration
  capacity 잔여는 **load(scale-의존)**·plan-execute controller 후보. **정직: 이 버킷 일부는 scale/load**.
- 우선순위 **6**(혼합·부분).

---

## 7. 우선순위·검증·정직 경계

### 우선순위 (ROI = 버킷크기 × decidability × 저위험)
| # | 레버 | 메커니즘 | 버킷 32B/14B | 위험 |
|---|---|---|---|---|
| 1 | 변형-select calc | R | 21/26 | 저(READ-증강) |
| 2 | coverage controller | G | 17/16 | 중(종료-게이트·over-ask 측정) |
| 3 | cross-order present | R | 7/10 | 저(READ-증강) |
| 4 | order-total calc | R | 6/3 | 저(기존 calc 확장) |
| 5 | feasibility gate | G | 4/7 | 중(should-not=intent 잔여) |
| 6 | loop-guard | G | (혼합) | 중(과거 해로움·재측정) |

### 검증 (구현 후·[[08]][[09]])
- 각 레버는 **A/B smoke**(레버 on/off·같은 task)로 **표적 버킷 pass↑ + over-block/over-ask 0 확인** 후 full.
- **replay 회귀**: infra=0 유지(READ-증강은 자명·G는 regen 경로).
- **[[05]] census**: `grep -c "if.*retail\|하드코딩"`=0·airline/bank A2-swap 등가 확인.
- 공식 compute_metrics pass^1..4로 재측정(같은-k·floor 대비).

### ★정직 경계 (over-claim 금지)
- decidable core = 변형-select 대부분·coverage(명시양화사)·cross-order(속성매칭)·order-total·불가능op.
- **genuine 잔여(scaffold로 못 닫음)**: 모호 NL→필드 매핑·암묵 scope·should-not intent·순수 orchestration
  load. **소량이나 존재** → paper1의 "learned path-selection residual" + load(scale)로 정직 귀속. 전부
  결정론이라 주장 금지.
- 규모-불변: 32B·14B 동일 버킷이므로 레버는 두 규모 공통(scale은 빈도).

## 8. 다음 (리뷰 후 구현 순서)
1. **§4 order-total calc**(최저위험·즉효) → smoke.
2. **§1 변형-select calc**(최대버킷·READ-증강) → smoke A/B.
3. **§3 cross-order present**(READ-증강) → smoke.
4. **§2 coverage controller**(G·over-ask 측정 동반) → smoke.
5. **§5 feasibility**·**§6 loop-guard**(측정 하에).
각 단계 full 재런 전 A/B smoke로 표적효과·부작용0 확인([[09]] 무료 먼저).
