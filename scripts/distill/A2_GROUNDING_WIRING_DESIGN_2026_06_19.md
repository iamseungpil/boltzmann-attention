# A2 Grounding Wiring 설계 — 엔진↔A2 포맷 연결 (keystone·2026-06-19)

> **자립 델타 문서**(리뷰용·2026-06-19 리뷰 4위험 반영). 상위 = `A2_FORMAT_SPEC_2026_06_19`(A2=3부 타입드 합성·여기선 ① dataflow + ② catalog의 *grounding 규약 구현*) · `A2_MINIMIZATION_FRONTIER_DESIGN_2026_06_19`(이 wiring = **S0 decidable-ablation의 구현 전제**) · `ABSTENTION_AS_DECIDABLE_2026_06_19`(§5a fetch-vs-ask 라우팅·구조적-sound-or-abstain).
> 불변 = `05-fixed-vs-variable`(scaffold 불변·`grep "if domain"=0`) · `00-thesis`(decidable=offload·LLM=formalize) · `03-anti-drift`(룰7 설계먼저). 이론 = `GENERATOR_ALGEBRA §1`(content-op=관계대수 Codd-닫힘·grounding-spec 투영도 동일 기저=§2b).
> 직전 = `HANDOFF_2026_06_19 §1.2` — "현 `t2_resolve_patch`는 ②일부(retail variants)만·①DAG·③rules 미통합 → ground_OK=0의 직접 원인."
> **★리뷰 반영(4위험)**: §2b 포맷 닫힘=관계대수(explode=unnest·ad-hoc 아님·3-도메인 사전등록) / §2c grounding-spec=고정 PROVIDE(shrinkable-A2 아님) / §5a fetch-우선 분기 / §6 구조적-sound≠correct·fabrication제거=offload-인프라 / §7 조건부 GO/NO-GO(spec-fail·C0·P2b 분해).

## 0. 문제 (정확히 무엇이 깨졌나)
`t2_resolve_patch._resolve`는 LLM이 emit한 `resolve_selection(op,...)`를 가로채 결정론 resolver(`tau2_op_resolver.resolve_op_tau2`)로 실행한다. resolver는 도메인-일반(ordinal/categorical·`op∈{filter,argmax,argmin,rank,comparative,substitute,create}`)이다. **그런데 그 사이의 *grounding*** — "어느 tool 출력서 catalog를 뽑고, 어느 출력서 anchor를 뽑나" — 이 `_ground_retail()`에 **retail 구조(`variants`/`items`/`product_id`)로 하드코딩**돼 있다.

⇒ 두 가지 동시 결함:
1. **기능 결함**: retail 외 도메인은 catalog/anchor를 못 잡아 `resolve_op_tau2`에 빈 catalog → `None` → `ground_OK=0`.
2. **thesis 위반**(`05-fixed-vs-variable §현위반`): 군-실행 엔진에 *구체*(ABox)를 박음 = RetailGate 하드코딩과 같은 죄. "fix scaffold, swap ABox" 주장이 성립 안 함.

## 1. 처방 = grounding을 A2 데이터로 외화 (코드→선언)
`_ground_retail`(코드)을 제거하고, **A2의 ① dataflow 부분에 "grounding-spec"**(선언적 JSON)을 둔다. 엔진은 spec를 *읽어* catalog/anchor를 뽑는다. **엔진 코드는 도메인 무관·`grep "if domain"=0`.**

이는 `A2_FORMAT_SPEC §1`의 ① dataflow(타입드 DAG: producer/consumes)를 *grounding에 필요한 최소 투영*으로 구체화한 것이다(full DAG는 후속·여기선 selection-grounding에 쓰는 엣지만).

## 2. grounding-spec 포맷 (도메인-일반·제거가능 A2 컴포넌트)
목표 = tool 출력 컨테이너를 `resolve_op_tau2`가 먹는 **relational catalog 행** `{item_id, options:{attr:val}, available:bool}`로 *선언적* 투영. 컨테이너 형태(map/list)·옵션 형태(평면 dict / nested enum-keyed dict)를 데이터로 기술한다.

```jsonc
// A2.grounding (도메인당 1개 파일 a2/<domain>.grounding.json — ABox swap 대상)
{
  "candidate_source": {            // ① "catalog는 어느 producer의 어느 출력서, 어떻게 투영"
    "producer": "get_product_details",   // P2b: 이 도구가 선행 호출돼야 함(DAG 엣지)
    "field": "variants",                 // 출력 내 후보 컨테이너 경로(dotted·""=출력 자체)
    "container": "map",                  // "map"(키=item_id) | "list"
    "id_key": "item_id",                 // list일 때 행 id 필드(map이면 키 사용)
    "options_path": "options",           // (선택) 행 안의 *평면* 옵션 dict 경로 → 그대로 options
    "fields": {},                        // (선택) {attr: dotted-path} 행서 추가 평면 속성 복사
    "explode": null,                     // (선택) §2a nested enum-dict → 행 분해
    "available_path": "available"        // (선택) 가용 플래그 경로(없거나 explode면 §2a 규칙)
  },
  "anchor_source": {               // ① anchor(수정 대상 현 항목) — substitute/comparative만
    "producer": "get_order_details",
    "field": "items", "container": "list",
    "match": {"on": "product_id", "from_candidate": "product_id"},  // 후보 product와 동일 행
    "id_key": "item_id"
  }
}
```
### 2a. `explode` = 관계대수 **unnest** (airline이 *강제한 ad-hoc 아님*·항상 닫힌 집합에 있던 연산)
한 컨테이너 원소의 속성이 **enum 키로 나뉜 dict**(예 airline `prices: {business:.., economy:.., basic_economy:..}`)면, 원소 하나를 *키마다 한 행*으로 분해(=unnest). **새 primitive가 아니라 §2b 관계대수 기저의 unnest 연산.**
```jsonc
"explode": {                                        // = unnest(prices, available_seats) by cabin
  "key_attr": "cabin",                              // dict 키 → 이 categorical 속성
  "from": { "price": "prices", "seats": "available_seats" }  // attr ← dict[key]
}
// → flight HAT001 한 개가 3행: {item_id:"HAT001", options:{cabin:"economy", price:230, seats:6}}, ...
```
- **★`available` = spec-driven 술어**(위험4 교정·엔진 기본값 금지): 가용성은 `available` 필드의 *선언적 술어*로 기술하지, 엔진이 `seats>0`을 박지 않는다(그건 잠복 도메인-의미=한 층 위 하드코딩). 예:
```jsonc
"available": { "attr": "seats", "pred": "gt", "value": 0 }   // airline: seats>0
"available": { "path": "available" }                         // retail: 플래그 그대로
"available": true                                            // 술어 없음(전부 가용)
```
- **엔진 연산**(고정·도메인무관): trace서 `producer` 출력 찾아 `field`로 컨테이너 추출 → 각 원소를 `options_path`(평면) ∪ `fields`(복사·project) ∪ `explode`(unnest)로 0+행 투영 → `available` 술어 평가 → `resolve_op_tau2`. `anchor_source` 있으면 동일 trace서 anchor 도출(후보와 `match.on` 일치 행=join).
- **literal 금지 유지**: anchor/catalog id는 *엔진이* trace서 ground(모델 emit 무시·`tau2_op_resolver:73-77` 그대로) → order_id/flight_number 날조 차단(P1).

### 2b. ★포맷 닫힘 = 관계대수 투영 (위험1 교정·핵심) — N=2-맞춤이 아니라 닫힌 투영언어
grounding-spec primitive는 ad-hoc 나열이 아니라 **관계대수 연산자**다 — tool 출력 JSON을 relational rows로 투영하는 언어:
| spec primitive | 관계대수 | 역할 |
|---|---|---|
| `field`/`options_path`/`fields` | **π project** | 필드 추출 |
| `explode` | **unnest** | nested 컬렉션 평탄화 |
| `anchor_source.match` | **⋈ join** | 후보 ⋈ anchor(공유 키) |
| `available` 술어 | **σ select** | 가용 필터 |
- 관계 투영은 **Codd-닫힘**(content-op `filter/argmax/project`와 *같은 기저*·`GENERATOR_ALGEBRA §1`). ⇒ `explode`는 "airline이 강제한 확장"이 아니라 *언제나 닫힌 집합에 있던 unnest*. 포맷 성장이 아니다.
- **★사전등록(falsifiable)**: **포맷 닫힘 = 3+번째 도메인(telecom·SOPBench 등)이 신규 spec-primitive 0으로 적합**(unnest/join/project/select 조합으로). 신규 primitive가 필요하면 = 포맷 미닫힘(도메인마다 DSL 성장=한 층 위 하드코딩)·thesis 위험. 안 하면 N=2용 포맷을 "도메인-일반"이라 부르는 것.
- **측정 단위 정합**(위험3과 연결): "A2 크기"로 세는 건 *도메인 사실*(어느 producer·어느 attr)이지 *투영 연산 집합*(닫힘·고정)이 아니다. §2c.

### 2c. ★grounding-spec = 고정 PROVIDE·shrinkable-A2 아님 (위험3 교정·A2_MIN 헤드라인 정합)
grounding-spec는 **엔진의 고정 입력**(72B 모델이라도 엔진은 `variants`가 catalog임을 알아야 resolve)이다 — 모델 크기와 무관·scale-sweep서 *안 줄어든다*. A2-최소화 thesis("모델이 덜 필요로 하나")의 측정 단위에 grounding-spec를 넣으면 "scale 무관 불변"이 나와 헤드라인을 깎는다(범주 혼동).
- **명시 분리**: **고정 PROVIDE**(grounding-spec·엔진이 필요로 하는 사실·도메인당 1회·swap만) **vs shrinkable-A2**(모델 크기↑면 덜 받아도 되는 규칙·gate_spec 등). grounding-spec = **전자**.
- `A2_MIN §3` 측정 단위서 grounding-spec를 *shrinkable* 컴포넌트서 제외하고 "고정 PROVIDE 비용"으로 별도 회계(변경-흡수 §6엔 들어감: producer rename 시 spec-edit). ⇒ S0가 측정하는 건 "engine+PROVIDE가 decidable을 처리하나"(기능)이지 "scale로 줄어드나"(그건 shrinkable-A2의 질문)가 아니다.

## 3. retail 매핑 (현 하드코딩의 1:1 외화 — 동치 검증용)
위 예시 JSON이 곧 retail spec. `_ground_retail`의 동작과 정확히 동치:
- `variants`(map, 키=item_id, `.options`, `.available`) → candidate. `get_order_details.items`서 `product_id` 일치 행의 `item_id` → anchor.
- **검증 기준**: spec-driven 경로가 현 retail e2e서 `_ground_retail`과 *같은* item_id를 내면 외화 무손실(회귀 0) 확인 후 airline 추가.

## 4. airline 매핑 (사용자 결정: 처음부터 흡수 — 도메인-일반성 강증명)
airline 데이터 구조(remote `domains/airline/data_model.py` 확인): `search_direct_flight → list[DirectFlight]`, 각 flight = `{flight_number, origin, destination, scheduled_departure_time_est, date, available_seats: dict[cabin,int], prices: dict[cabin,int]}`. `CabinClass = {business, economy, basic_economy}`.

**선택(P4) 매핑**: "가장 싼 economy 직항 예약" → LLM이 `resolve_selection(op=argmin, attr=price, among={cabin: economy})` emit → 엔진이 §2a explode로 flight×cabin 행 합성 → `among` cabin=economy 필터 → price argmin → **flight_number 반환**(literal 금지·엔진 ground). cabin은 `ORD_WORDS["cabin"]`로 ordinal·price는 numeric → `resolve_op_tau2` **그대로** 동작.

```jsonc
// a2/airline.grounding.json
{
  "candidate_source": {
    "producer": "search_direct_flight", "field": "", "container": "list",
    "id_key": "flight_number",
    "fields": { "departure": "scheduled_departure_time_est", "date": "date" },
    "explode": { "key_attr": "cabin", "from": { "price": "prices", "seats": "available_seats" } }
  }
  // anchor_source 없음(예약=create/argmin·anchor 불요). cabin 업그레이드(substitute)는 §4a.
}
```
- **이게 포맷 도메인-일반성의 증명**: 같은 엔진·같은 `resolve_op_tau2`가 retail(map·평면 options)과 airline(list·explode)을 *오직 spec 차이*로 처리. 엔진 `if domain`=0.

### 4a. cabin 업그레이드(substitute) — 후속 (keystone 범위 밖·미차단)
`update_reservation_flights`로 cabin만 올리는 변경은 anchor=현 reservation flight·`set={cabin: business}`의 substitute. 단 airline은 cabin이 *예약 속성*이라 "새 flight_number 선택"과 결합(같은 flight, 새 cabin) → §4 explode 행 위 substitute로 표현 가능하나, *예약 변경 도구 시그니처 결합*은 별 검증. **keystone = 예약-선택(argmin/filter) 먼저 ground_OK>0**, cabin-substitute는 직후 동일 포맷서 검증.

## 5. 엔진 변경 (최소·1파일)
`t2_resolve_patch.py`:
- `_ground_retail(outs)` **삭제** → `_ground(outs, gspec)`: gspec(§2) 읽어 candidate/anchor 투영(π/unnest/⋈/σ). 경로 추출은 generic dotted-path getter.
- `_resolve(orch, tc)`: A2 grounding-spec 로드(도메인 config 파일·아래) → `_ground` → `resolve_op_tau2`.
- **A2 로드 경로**: `apply(a2_grounding_path)` 인자 또는 env var로 도메인 spec JSON 주입. **도메인 식별은 spec 파일 선택(런처)으로**, 코드 분기 아님.
- `scripts/distill/tau2/a2/{retail,airline}.grounding.json` 신설(§3·§4 내용).

### 5a. ★ground 실패 라우팅 = fetch-우선 분기 (위험B 교정·bare ask면 τ²서 깨짐)
`ground_OK=0`엔 두 원인이 섞이고, **결정론으로 구분 가능**(`ABSTENTION_AS_DECIDABLE` 참조):
```
ground_OK=0 → producer 출력이 trace에 있나?
   ├ 아니오 + producer가 A2-스키마에 존재(=fetchable) → FETCH 신호("get_product_details를 먼저 불러라")  ← ask 아님·P2b
   ├ 아니오 + producer 부재/값이 user·tool 어디에도 없음            → ASK/escalate(P7)
   └ 예지만 resolve 비유일/anchor 불명                              → ASK(clarify) 또는 FETCH(누락 producer)
```
- τ²의 order_id 미ground는 **거의 (2)=fetchable-but-not-fetched**(`get_user_details` 미호출). bare `ground_OK=0→ask`로 보내면 user-sim이 order_id 모름 → 거짓-실패. **반드시 fetch-우선**.
- 즉 abstention-as-decidable이 A2(producer-존재)에 의존 — 그 producer-존재 체크(스키마-decidable·지난 턴 graded-DAG)가 ask-vs-fetch를 가른다.
- **현 abstain 메시지**(`t2_resolve_patch:119-121`)는 이미 "fetch 먼저" 톤이나 *조건 분기 없이 항상 같은 메시지* → producer-존재에 따라 FETCH vs ASK로 **분기**시킨다.
- **범위 한정(위험C/E)**: 이 라우팅은 *막힌 단일 스텝의 차단*까지만 깨끗. ASK 후 user 답 받아 재시도하는 multi-turn recovery=반응형 P7·gold deny 없어 static-SFT 불가(`PRIMITIVE_MATRIX:94-95`·RL 별개·미해결). "구조적 무결"은 단일-스텝 차단으로 한정.

## 6. 자가심사 (anti-drift 룰7·`A2_MIN §9`)
- **치팅면**: 엔진 `grep -n "if .*domain\|retail\|airline\|variants\|product_id" t2_resolve_patch.py` = **0**(전부 spec서 읽음·CI 가드). real 도구 미대체(resolve_selection은 보조 도구·concrete는 trace ground).
- **thesis정합**: 학습=op-naming(formalize·불변) / **grounding=결정론 엔진 + A2 spec(swap)** / soundness 게이트=별도. = decidable(which-producer·deep-select)을 *가르치지 않고* offload(`00-thesis`).
- **게이밍**: ground_OK는 "엔진이 trace서 실제 fetch된 catalog로 resolve 성공"만 카운트(빈 catalog→`None`→fail). resolver가 유일 후보를 반환(`len==1`) — 임의 추측 아님.
- **★보장 범위 = 구조적-sound-or-abstain ≠ correct-or-abstain**(위험A): 엔진이 막는 건 **구조적 오류**(fabrication·미인증·미확인·resolve-비유일)뿐. **구조적으론 멀쩡한데 의미적으로 틀린 경우**(grounded·unique·gate-pass·reversible인데 *잘못된 valid item* 선택·NL 의도 오독)는 어떤 구조조건도 안 걸림. ⇒ 헤드라인 = "구조적 무결 보장(fabrication 0) + 의미 잔여는 격리된 selective-prediction", **"correct-or-abstain"이라 쓰지 말 것.** fabrication 제거 후 그 의미-오류 잔여가 *지배적일 수 있음*(τ² 실패 상당수=날조 아닌 wrong-valid-selection) → 잔여="작다"는 가정 말고 **측정 대상**(§7-③).
- **★fabrication→0은 offload-인프라지 학습-기여 아님**(위험C/E): ground-실패 라우팅(§5a)은 결정론 엔진 변경이라 fabrication을 trivially 0으로(미ground 값을 엔진이 절대 안 흘림). = offload 다리(`2603.20449`/R1b-gate 동류·known engineering)지 *모델이 뭘 학습했나*가 아님. **"fabrication 사라짐"을 thesis 결과로 오독 금지** — 필요 인프라지 novelty 아님. Novelty는 그 위에서 **A2-formalize가 전이하나**.
- **정직**: 이건 **decidable 부분(grounding/select)을 결정론이 처리함**의 실증이지, e2e pass 상승 약속 아님(pass는 상류 op-naming·게이트·의미선택에도 의존). ground_OK>0가 1차 통화.

## 7. GO / NO-GO (= A2_MIN S0 decidable-ablation) — ★조건부 측정 (위험2 교정)
**bare `ground_OK`로 GO/NO-GO 하면 오진**: `ground_OK=0`엔 세 원인이 섞여(아래), spec가 완벽해도 (b)/(c)면 안 오른다. 반드시 **분해**해서 본다(안 하면 1시간 뒤 데이터를 오독).
- **분모 분해(3원인)**:
  - (a) **spec-fail** ← *측정 대상*: resolve emit·producer 호출됐는데 spec가 catalog/anchor를 못 뽑음.
  - (b) **C0=resolve 미emit**: 모델이 `resolve_selection`을 애초에 안 부름(solo_sts=0회 전례·**지배 교란 가능**).
  - (c) **P2b=producer 미호출**: 검색/fetch getter를 안 불러 trace에 catalog 부재.
- **★핵심 지표 = 조건부**: `P(ground_OK | resolve-emitted ∧ producer-called)` = spec 품질의 *순수* 측정. + 분모율 **`P(resolve emitted)`**(C0)·**`P(producer called | resolve emitted)`**(P2b) 별도 보고. 그러면 "왜 낮은가"가 [spec-fail | C0 | P2b] 셋으로 분해돼 진단가능.
- **arm**: (i) engine+A2-spec vs (ii) spec 없는(현 0-ground) vs (iii) 모델만(엔진 off). 각 arm서 위 조건부+분모율.
- **GO**: 조건부 `P(ground_OK | emitted ∧ called)`가 (ii)≈0 → (i) 유의>0(대다수)이고, retail이 `_ground_retail`(하드코딩)과 동치. ⇒ "decidable 부분을 engine+A2가 결정론 처리" 실증 → S1(모델 sweep·A2 ablation).
- **NO-GO 분기(원인별 처방 다름)**:
  - 조건부가 낮음(spec-fail) → 포맷/spec 결함. 진짜 NO-GO·포맷 재설계.
  - C0 지배(resolve 미emit) → 모델이 op-naming 안 함(학습/프롬프트 문제·resolve wiring 무관) → C0 트랙.
  - P2b 지배(producer 미호출) → 상류 gather 병목 → §5a fetch-우선 라우팅 + P2b 트랙.
- **③ 의미-오류 잔여율**(위험A): ground_OK인 케이스 중 *gold item_id 불일치*(구조 통과·의미 틀림) = "잔여가 작은가"의 실측. wiring 후 첫 측정에 포함.

## 8. 열린 질문 (리뷰 훅)
1. ~~airline 흡수 vs 분리~~ → **결정(사용자): 처음부터 흡수**(§4 explode). retail+airline 둘 다 §2 포맷·동일 엔진.
2. ~~spec 위치~~ → **결정(사용자): 새 `a2/<domain>.grounding.json`**. 기존 `t2_a2_*`(GATE_SPEC=③ rule-set 트랙)와는 별개·나중 정합.
3. **grounding-spec ⊂ A2_FORMAT_SPEC ① dataflow-DAG**의 부분집합인가, 별 컴포넌트인가? (A2-최소화 측정 단위 정의에 영향 — `A2_MIN §3`.) → 본 설계는 "① dataflow의 selection-grounding 투영"으로 위치.
4. **resolve_selection 도구 docstring 도메인-일반화**: 현재 retail-맛(variants 언급). airline도 같은 도구로 emit하려면 도메인-중립 재서술 필요(item/flight 모두 "candidate"). 구현 시 반영.
5. **airline 선택 태스크가 trace에 search_direct_flight 출력을 실제로 남기나**: explode 전제 = 상류 P2b(검색) 호출됨. 안 남으면 ground 불가(=상류 gather 병목·NO-GO 신호).

## 9. 한 줄
**grounding을 `_ground_retail`(코드)서 A2 grounding-spec(닫힌 관계대수 투영언어·고정 PROVIDE)으로 외화 → 엔진 `if domain`=0, retail+airline을 *오직 spec 차이*로(=포맷 도메인-일반·§2b), ground-실패는 fetch-우선 분기로(§5a·fabrication 구조적 0=offload-인프라), 조건부 측정으로 spec-fail/C0/P2b 분해(§7).** keystone = "fix scaffold, swap ABox"를 grounding에서 처음으로 성립 + 구조적-sound-or-abstain 축 추가(의미 잔여는 측정 대상). novelty=그 위 A2-formalize 전이.
