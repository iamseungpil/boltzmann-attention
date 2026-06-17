# Unified Concrete-Value Offload — τ² e2e (2026-06-18)

> 권위 = `ma/M_A_RESULTS.md §19-26`. 불변 = `feedback-nl-formalize-llm-selection-deterministic`·`feedback-selector-verifier-deterministic`·`feedback-thesis-tbox-transfer-direction`.

## 0. 왜 (수렴 진단 §26)
τ² 전 벽이 **한 뿌리**로 수렴: **LLM이 fetch/계산해야 할 *구체값*을 발명/오선택**.
- write-벽 = order_id/item_id **날조** (§25: base서 order_id 날조 91·item 15).
- GBW = new_item_ids **오선택** (§24: 5%).
- width = multi-attr set **binding 실패** (§22: 소형 벽).
- collapse 60% = fetch-id **날조** → not-found → retry-same loop (§25).

**단일 처방**: LLM은 *얕은* 일(op/도구 명명·referent NL 기술·attr별 변경 결정)만, **구체값(read-id·write-변형·multi-attr 조합)은 결정론 resolver**. = LLM이 구체값을 *절대 생산 안 함* → 날조 불가(구조적).

## 1. 원칙 (사용자 지시 정식화)
- **width**: 각 attribute는 **LLM이 개별 결정**(does attr X change? to what?), **조합은 결정론**(target=old⊕changes, new_item_id 매칭). = decomposition(§22 실증: 7B 0.51→0.87).
- **파라미터**: **단계별 수집(staged collection)** — order_id/item_id/payment를 한 번에 날조하지 말고, 직전 tool 결과서 **점진 grounding**(user→orders→order→items). LLM은 *어느 것*인지 NL로 referent만, resolver가 fetch된 상태서 concrete 확정.

## 2. 아키텍처 = referent 도구 래퍼 + resolver (native tool-call 유지)
§23E 교훈: op-IR로 재학습하면 native tool-call 포맷 붕괴. ⇒ **재학습 0**. 대신 **도구를 referent 버전으로 래핑**하고 그 사이에 결정론 resolver. agent는 여전히 native tool_call emit(포맷 불변).

### 2a. read-step = staged parameter collection
- **상태저장소**: 대화 진행하며 fetch된 DB 사실(user·orders·items) 누적.
- **referent 도구**(concrete id 대신 NL referent 받음):
  - `authenticate(name, zip|email)` → 실제 find_user → user_id 저장.
  - `find_order(description)` → resolver가 user의 주문목록 조회(없으면 fetch) → description 매칭 → order_id. **날조 0**(목록에 있는 것만)·모호하면 후보 반환→LLM 선택.
  - `find_item(order, description)` → 그 주문의 item 목록서 description 매칭 → item_id.
  - `get_payment(method_hint)` → user 결제수단서 매칭.
- **staged 강제**: 각 referent는 *이미 grounded된 상태*서만 resolve·선행 fetch 누락시 resolver가 자동 fetch. order_id는 **항상 user의 fetch된 주문서** 나옴(발명 차단). `#` 포맷도 resolver가 정규화.

### 2b. write-step = per-attr decomposition
- `exchange(order, item, request_nl)`:
  1. resolver가 item의 option attr 열거.
  2. 각 attr마다 LLM에 width-1 질의: "request가 <attr> 바꾸나? 뭐로(<enum> 중)?" (§22 decomp arm).
  3. 엔진: target = old ⊕ changes, **new_item_id 결정론 매칭**, 값은 카탈로그 enum **최근접 스냅**(wrong_value 정규화 처리).
- LLM=attr별(얕음)·결정론=조합. (§23D wide-학습 회피=routing 불변.)

## 3. LLM이 *여전히* 하는 것 (얕음·잘함)
- 라우팅(어느 도구/op) — §21 recognition 높음.
- attr별 변경 결정 — 예산 내.
- referent NL 기술.
**안 함**: concrete id·multi-attr 조합·포맷.

## 4. τ² 구현
- tau2 env 도구 레지스트리에 **referent 래퍼 + resolver** 주입(원 도구는 resolver가 내부 호출).
- resolver = `tau2_op_resolver`(substitute/create) + **id-grounding 모듈**(order/item by description·매칭). 결정론.
- agent = base 7B(재학습 0·routing만)·native tool-call.
- 측정: e2e gated pass^1 vs base 0.17. 예상: order_id-날조 collapse(91)+GBW(5%)+width 동시 제거 → 큰 상승.

## 5. 구현 단계 (첫 행동)
1. `tau2_offload_tools.py` — referent 도구 정의(authenticate/find_order/find_item/get_payment/exchange) + 상태저장소.
2. `tau2_id_resolver.py` — description→id grounding(fetch된 상태서 매칭·모호시 후보).
3. write `exchange` referent에 §22 decomp(per-attr LLM 질의) + 엔진 조립 + enum 스냅.
4. tau2 env에 도구 swap(원 도구 숨기고 referent 노출) — env config or tool-registry patch.
5. `t2_run_gated` 재실행(referent 도구셋) → e2e pass^1. autopsy로 collapse 소멸 확인.

## 6. 정직 / 잔여
- 새 표면 = **referent 매칭 오류**(description→id) — 단 유계·결정론 디버그 가능·모호시 LLM에 후보 환원.
- retry-복구: referent 진짜 모호시 resolver가 후보 반환→LLM 재선택(루프 차단).
- decomp 비용 = write당 n_attr 질의(τ² catalog attr 소수라 저렴).
- 측정 = 40 tasks·gated·base 7B. 성공판정 = pass^1 ≫ 0.17 & collapse(order_id 날조) 소멸.
- 신규성 경계: referent-tool 패턴은 기존 사례 있음(인용 필요)·기여 = NL→op/referent 분담 + 결정론 grounding의 **e2e 날조차단 실증**.
