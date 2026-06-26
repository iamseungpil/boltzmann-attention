# Escape-Scope Diagnostic — 층화(layered) σ 보강 설계 (2026-06-24)

> **모doc**: `ESCAPE_SCOPE_DIAGNOSTIC_DESIGN_2026_06_24.md`(rev2)의 보강. S1 harness preview가 드러낸 것 반영. thesis `EPISTEMIC_A2_THESIS_2026_06_23` §3 escape-범위·§6 make-or-break.

## 0. 동기 — S1 preview가 깬 가정 (정직)
rev2는 실패를 **order-층 disambiguation** 하나로 가정했다. **S1 preview(15 gap)가 반증**:
- **다수 gap은 32B가 *gold 주문을 이미 골랐다*** (task 8·17·34·41·101·102) → 실패는 order-층 아님.
- **단일 주문 task**(36/37/38·daiki=후보 1개)는 order-disambiguation *불가능* → 실패는 하류.
- **틀린 주문**(71·72=σ=1 ⓑ mis-ground)은 *소수*.
- ⇒ **order-층 σ만으론 gap 다수를 분류 못 함.** 실패가 어느 *층*인지부터 찾아야 한다. order-층 escape는 좁고, 진짜 게이트는 하류 층 분포에 있음.

## 1. 층 구조 (failure locus)
write tool call = (operator, order_id, item_ids, new_item_ids, options/payment). gold vs 궤적을 **field-by-field 비교 → first-divergence 층**이 실패 locus:

| 층 | divergence | 관계/σ 대상 | 예 |
|---|---|---|---|
| **L0 operator** | tool name 틀림 | (관계 아님·action) | 주소수정↔취소·교환↔반품 |
| **L1 order** | order_id 틀림 | σ(user의 orders) | task 71 DC주문 |
| **L2 item** | item_ids 틀림 | σ(그 order의 items=anchor_source) | 어느 품목 교환 |
| **L3 variant** | new_item_ids/options 틀림 | σ(product variants=candidate_source) | black lamp·medium polyester |
| **OVER** | gold에 없는 추가 write | (action·over-action) | task 101/102 여분 주문 |

- L2/L3 = `retail.grounding.json` **anchor_source(order items)·candidate_source(product variants)**가 이미 실물 → 그 σ 재사용.
- 한 task가 여러 층 실패 가능 → **first-divergence를 primary**로, 나머지는 secondary 태그.

## 2. 각 층 σ + ⓐ/ⓑ 분류 (rev2 §2를 층별로)
각 층에서 **faithful 술어**(유저 발화의 그 층 리터럴 제약) → σ(결정점 state) → |σ| + 궤적선택 대조:
- **|σ|=0** → no-change impasse → **ⓐ** ("해당 없음→ASK/불가통지")
- **|σ|>1** → tie impasse → **ⓐ-tie** *단 §3 단서 적용*
- **|σ|=1** → 유일정답: 궤적이 그걸 골랐나 → 아니면 **ⓑ**(mis-resolve·침묵)
- **L0/OVER** = impasse 아님 → **ⓑ-act**(escape 밖·operator-select/over-action)

## 3. ★핵심 정밀화 — tie의 절반은 escape 아님 (resolve)
**rev2가 놓친 것**: |σ|>1(tie)이라도 **유저가 tiebreaker를 줬으면 escape-ASK가 *아니라* 내부 resolve**.
- 예 task 71: "backpack medium polyester, **if multiple colors prefer grey**" → color tie를 유저가 *미리 해소*. → σ_{size=medium∧material=polyester} 다중이라도 **prefer grey로 resolve**지 ASK 아님.
- ⇒ ⓐ-tie를 쪼갬: **ⓐ-ask**(tiebreaker 없음·진짜 모호→ASK) vs **ⓐ-resolve**(tiebreaker 있음·SOAR 선호충분→impasse 아님).
- **★ⓐ-resolve를 또 갈라야 함([[10]] B1/B2 정정·리뷰#1)**: tiebreaker가 **ordinal(prefer grey·cheapest·max)** 이면 **B2=argmax/rank=결정론 offload**(학습 아님); **semantic("eco-friendly"·"a warm color")** 이면 **B1=의미매칭=학습**. → rev2의 "B2-resolve 학습"은 용어오류(B2=결정론). 학습은 B1뿐.
- **escape(ASK)가 잡는 건 ⓐ-ask 뿐.**

### 3.1 ★filter vs tiebreaker 조작 규칙 (헤드라인 좌우·리뷰#3·S2-(f) 게이트)
ⓐ-ask/ⓐ-resolve를 가르는 게 escape 너비를 직접 결정 → 큐레이터-의존 뒤집힘 방지 위해 규칙 명문화:
- **hard requirement**(size=medium·material=polyester·state=DC) → **filter**(σ에 넣음).
- **선호/순서 표현**("prefer X"·"cheapest"·"max"·"if multiple…") → **tiebreaker**(σ에서 빼고 → ordinal=B2 / semantic=B1).
- **모호 표현**("a nice one"·미지정) → **ⓐ-ask 후보**(tiebreaker 아님·진짜 모호).
- 규칙 적용 후 S2-(f) spot-check. 규칙으로 안 갈리는 경계는 케이스별 *근거 명시*(카탈로그에 기록).

## 3.5 ★층 → 레버 매핑 (리뷰#1·#2 핵심 — 진단의 본체)
진단은 escape 너비만이 아니라 **gap을 {결정론-게이트 / 결정론-resolve / 학습 / capability}로 분할**한다. 각 층/라벨의 레버:

| 층/라벨 | 레버 | 종류 |
|---|---|---|
| **ⓐ-ask** | ASK(빈관계 escape) | **학습**(abstain 커리큘럼) — *Stage-1 표면=0* |
| **L0 operator** (modify↔exchange) | eligibility/status gate(pending→modify·delivered→exchange) | **결정론**(G-gate·[[06]]) |
| **L0 branch** (조건부 "if X then…else") | 정책/계산 분기(partial-cancel 불가→addr·cheapest-sum>thresh→cancel) | **결정론**(eligibility/arith·hole-D) |
| **OVER** (여분 write·gold=0 write 포함) | stop-when-satisfied / commit-once gate | **결정론**(G-gate) |
| **ⓐ-resolve ordinal** (prefer grey/cheapest/most-expensive) | argmax/rank | **결정론**(B2 offload·[[10]]) |
| **ⓐ-resolve semantic** ("eco-friendly") | 의미매칭 | **학습**(B1) — *Stage-1 표면≈0(tiebreaker 전부 ordinal)* |
| **⋈-join** (hole-A·"다른 주문의 주소"·"pending 주문의 그 type") | cross-entity 관계조인(σ 아님·B2 아님) | **혼합**: scaffold ⋈ 가능(결정론) + 모델이 cross-ref *formalize*(학습)·**Probe-B서 present→7/7 작동** |
| **ⓑ mis-ground** (σ=1·model 딴 entity) | faithful-formalize(present로 해소) | **학습**·*capability-bound 아님*(Probe-B raw 7/7) |
| **ⓑ-op** (hole-C·verbatim "St"≠"Street"·availability 무시) | formalize-fidelity / copy-scaffold | **학습**(operand·B1 아님) |

→ **S2 재집계(정정·`STAGE1_CATALOG` §5.5)**: formalize-ⓑ(mis-ground+⋈+op) ≈ 8~9/15 *단독 최대* · 순수 결정론(L0/OVER/B2-ord)= 5 · **ⓐ-ask=0·B1-semantic≈0** · MISS=2. ⇒ **"학습 잔여=formalize 지배(ask/semantic 아님)" + 결정론 공동.** ⓑ·⋈ 모두 *present(Probe-B)로 작동* → 잔여의 정확한 위치 = **"σ_criterion 결과를 결정점에 선택지로 *제시*"**(autofetch arm이 e2e 전환 검정).
- **정렬 모호 해소**: 72=ⓑ mis-ground(gold 양 op 다 #W5270061·L0는 정렬 artifact·dump 확인) · 102=⋈(주소출처)+L1.

## 4. 분류 절차 (harness)
1. gold write-actions vs 궤적 write-calls 정렬. **★L0/L1 divergence가 정렬키를 깬다(리뷰#4·순환)**: operator 틀리면 operator로·order 틀리면 order_id로 정렬 불가 → **폴백 = gold action-index 순서 우선, 동률이면 best-field-overlap**(어느 traj-call이 어느 gold-action 대응인지 측정대상에 비의존). 정렬 모호 케이스는 카탈로그에 플래그.
2. **first-divergence 층** 판정(L0→L1→L2→L3→OVER 순).
3. 그 층의 candidate collection 로드(L1=orders·L2=order items[anchor_source]·L3=variants[candidate_source]).
4. 그 층 faithful 술어(큐레이션) → σ(결정점 state) → |σ|.
5. tiebreaker 유무 판정(§3.1 규칙) → ⓐ-ask vs ⓐ-resolve(ordinal/semantic).
6. 궤적선택 대조 → 최종 라벨 + impasse 타입 + §3.5 레버.
7. (Arm-II·★스코프 리뷰#5) select-probe = **L1/L2/L3 entity 선택에만**(후보집합 떠먹이고 32B가 고르나). **L0=변형**("유효 operator 집합서 선택"=status로 결정되는 eligibility를 줄 수 있나) · **OVER=probe 대상 아님**(stop 판단=결정론·후보선택 형태 아님).

## 5. 출력 (rev2 §5 확장·★정성/비율 분리 리뷰#5)
1. **층별 실패 분포**(L0/L1/L2/L3/OVER) — gap의 어디서 깨지나.
2. 층별 **ⓐ-ask / ⓐ-resolve(ord/sem) / ⓑ** split + §3.5 레버 집계(결정론 vs 학습 vs capability).
3. **escape-catchable = Σ ⓐ-ask** = 진짜 escape 너비.
4. ⓑ·ⓐ-resolve·L1-3 케이스 select-probe pass율(grounding-됨=학습여지 vs capability-bound).
5. impasse×층×gap-class 교차표.
- **★n=15는 비율 아님**: 15 task × 5층 × 라벨 ≈ 셀당 0-2 → Stage-1 분포·비율 통계 무의미. **Stage-1=정성 카탈로그(방향: escape narrow? 어느 층 지배? 결정론 vs 학습 어디로?)·케이스별 근거**, **비율은 S4(retail 전체 실패)**.

## 6. GO/NO-GO 영향 (rev2 §6 정밀화·★방향+케이스, 비율 아님)
- **헤드라인 = 비율이 아니라 *방향+케이스***(n=15): escape narrow한가? gap이 결정론게이트로 기우나 학습으로 기우나?
- preview 조짐: 다수=gold-order-picked→하류 / tiebreaker 있는 tie=ⓐ-resolve / 틀린주문=σ=1 ⓑ → **ⓐ-ask(진짜 escape) 작고, 본체가 결정론(L0/OVER/B2)+formalize(ⓑ)로 기움.**
- ⓐ-ask 작음 → **abstain-ASK 커리큘럼은 좁은 레버.** 진짜 본체 = **(i) 결정론 게이트(L0 eligibility·OVER stop·B2 ordinal=[[06]] G1-G4 영역) + (ii) faithful-formalize 학습(ⓑ) + (iii) B1-semantic 학습**. ([[10]] B2=결정론·B1=학습 정정 반영.)
- NO-GO(b 강화): ⓑ·B1 케이스 select-probe서 **후보 줘도 32B 틀림** 압도 → capability-bound(escalate). 후보 주면 맞힘 → 학습여지(부분 GO).
- **★전략 함의(리뷰#2·표류방지 못박기)**: **escape narrow ⇒ 학습 잔여 전체가 좁음(ⓐ-ask+ⓑ-fixable+B1) ⇒ 헤드라인은 epistemic-abstain이 *아니라* 결정론게이트(offload된 decidable)+TCO/전이([[06]] 북극성).** epistemic-abstain은 *잔여 보조 기여*. = thesis §4가 예고한 정밀화. 나중에 "escape가 헤드라인인 줄"로 표류 금지.
- **★★레버-종류 분할 ≠ 해결 분할 (대칭 함정 차단·정적진단 범위)**: 이 진단은 정적이라 **"어느 레버 *종류*가 적용표면인가"만 측정**하지 **"그 레버가 task를 pass로 *전환*하나"는 측정 못 함.** 결정론 다리(L0/OVER)엔 불리한 전례: [[06]] **G5(eligibility-steer) 인과효과=0·G1-G4 닫아도 7B capability 벽**(eligibility 클래스=닫혀도 미전환 전례). L0를 "결정론 G-gate"로 분류하는 건 *deny/enforce면 전환되리란 가정*인데 **미입증**(G5는 steer였고 inert)·Arm-II는 L1-L3 전환만 봄(L0/OVER 비대상·§4.7). ⇒ **진단 산출 = *레버-종류* 분할이지 *해결* 분할 아님. L0/OVER 적용→pass 전환은 별도(FLOW_DISCIPLINE류 실런이 답할 몫·이 정적진단 밖).** abstain 과대평가를 막은 자리서 *결정론게이트 과대평가*도 똑같이 금지.
- ★단 이것이 SOAR 반박은 아님 → §9.

## 6.5 ★사전등록 재설계 트리거 (선언 아니라 진단의 귀결·리뷰 수용·[[03]] 표류방지)
결정론 수렴은 현재 *3중 신호*(경험 [[06]] G1-G4=레버·G5/SFT inert · 진단예측 L0/OVER/B2=결정론 · 선행 SOAR/CoALA control=결정론). **단 3 모두 *예측/선행*이지 *이 진단 데이터* 아님** → 지금 갈아엎으면 [[03]] 표류. 그래서 *사전등록*:
- **트리거 충족** = 진단이 (i)escape-narrow(ⓐ-ask 작음) **AND** (ii)결정론-층(L0/OVER/B2-ordinal) 지배 **AND** (iii)Arm-II서 ⓑ가 후보 줘도 안 닫힘(번역 capability-bound) 중 *방향이 확증*되면:
  → **중심실험 재설계**: learn-first(abstain-SFT) → **"SOAR-최소 결정론 impasse-엔진 + LLM=boundary translator(§2) + 측정된 learn 잔여(faithful-formalize·B1)"**. escape=external-ask 슬라이스로 *종속*(폐기 아님).
- **반대 결과** = Arm-II서 ⓑ가 *후보 주면 닫힘*(=free-formalize만 실패·번역 학습가능) → **learn 잔여가 본체로 생존**·재설계 보류·learn-first 유지.
- **★[[05]] 가드(결정론 재설계의 최대 함정)**: 재설계가 "failure-type마다 retail 게이트 더 박기"=도메인특화 scaffold 순증=반복위반. **척추는 SOAR-최소 *일반* 엔진(kind/decision-cycle)+A2(어느 status·tiebreaker)** 여야지 게이트 증식 아님.
- **공정성 가드(rig 방지)**: "결정론으로 기움"이 결정론 결론으로 쉽게 가려는 편향 안 되게 — **Arm-II select-probe가 adjudicator**(ⓑ=capability-bound→결정론/escalate vs 학습여지→learn). 반드시 유지.

## 7. 구현 (rev2 §8에 추가)
- S1b: `escape_scope_diag.py`에 **layer_decompose(gold, traj·§4 폴백정렬)** + L2/L3 σ(grounding.json anchor/candidate_source 재사용·`t2_resolve_patch._ground` 참조) + **§3.1 tiebreaker 규칙 검출**(ordinal/semantic 구분) + §3.5 레버 태깅 + select-probe(L1-3·L0 변형).
- `escape_predicates.json`을 **층별 술어**로 확장(task별 L1/L2/L3 filter + tiebreaker{type:ordinal/semantic/none}).
- S2 대면검증 항목: (e) first-divergence 층 판정 (f) **tiebreaker 유무+종류(ord/sem)** 판정 (= ⓐ-ask/resolve·B1/B2 갈림이 여기 달림).
- S3 정성 카탈로그 = 15 task 층별·레버별 분류. S4 비율 = retail 전체 실패 층화.

## 8. 불변
- 정적·tau2 학습0·A2 σ(grounding.json 재사용)·도메인분기0·gpt-4.1 불요(select-probe=로컬 32B)([[05]][[11]]).
- **S2 (a)~(f) 대면검증 전 무인 전수 금지.** tiebreaker/층 오판=헤드라인(escape 너비·결정론vs학습) 오도.

## 9. ★SOAR 정합 재검토 — "결정론으로 기움"은 SOAR 반박 아님 (엄밀)
이 진단의 결론이 결정론게이트로 기울 때 SOAR 선행과 *다른 결론*인가? **아니다 — SOAR 자신의 예측이고, 우리 delta를 날카롭게 한다.**
- **(1) 크기**: SOAR도 *유능한* 행동의 대부분 = recognition(직접 발화)이고 **impasse는 지식 프런티어에만**·chunking이 프런티어를 뒤로 밈 → 숙련될수록 결정론 결정절차 지배·impasse(학습)는 잔여. ⇒ "결정론 우세·학습 잔여"는 SOAR와 **정합**.
- **(2) 혼동의 정체**: SOAR에서 그 결정론 게이트(eligibility/precond/stop)는 **chunking이 학습으로 컴파일한 산물**(결정화된 학습·학습의 대안 아님). 우리는 decidable을 **손작성 도메인-일반 scaffold + A2-swap으로 offload**([[10]] decidable→offload·[[05]]). ⇒ **같은 종착점, 다른 획득경로**: SOAR=runtime chunk·도메인마다 재학습 / 우리=design-time author·relation-swap. **이게 delta(+TCO 효율)이지 반박 아님.** ★**단 조건부(정직)**: 이 획득경로 delta(재학습0·TCO 우위)는 **우리 A2-swap 전이가 실재할 때만 성립** — 전이는 아직 미입증(증명=도메인 내·cross-domain scaffold는 부분, [[06]]·[[11]]·RELWORK §4.5). 전이 깨지면 "SOAR chunking은 *자동* 학습+전이인데 손작성+A2가 왜 우위?"에 무방비 → **delta는 "A2-swap 전이 GO"를 선결로 단다.**
- **(3) ⓐ-resolve vs ⓐ-ask = SOAR 결정절차 그 자체**: SOAR "선호 충분→impasse 아님"(=ⓐ-resolve) vs "선호 부족 tie→subgoal"(=ⓐ-ask·외부 subgoal=ASK). 우리 split이 SOAR 결정절차를 문자 그대로 인스턴스화.
- **(4) 유일 divergence = 아키텍처 아닌 LLM-capability**: SOAR의 유일 낙관=프런티어가 *학습가능*(chunking 신뢰). 우리 잔여(formalize/B1)가 LLM에서 capability-bound(§6 NO-GO-b)면 그게 *다른* 결론 — 단 LLM 사실이지 아키텍처 아님. 게다가 SOAR도 **chunking 과일반화/mis-chunk**(=우리 ⓑ 동형) 알려진 실패모드 → SOAR조차 프런티어 학습 공짜라 안 함. **정면충돌 아님.**
- **(5) [[06]]과 비중복**: [[06]] "learn inert"는 *왜*(decidable? G5 틀린타깃? capability?)를 구분 못 함. 층화 진단은 셋 분리→**진짜 학습 잔여(ⓐ-ask+ⓑ-fixable+B1) 격리** = abstain-SFT 조준 표면 제공.
- **한 줄**: 결정론으로 기움 = SOAR와 *결론* 차이 아니라 *획득경로* 차이(우리=offload·SOAR=chunk). thesis §3 SOAR 블록 delta-ⓐ/ⓑ와 정합·강화.
