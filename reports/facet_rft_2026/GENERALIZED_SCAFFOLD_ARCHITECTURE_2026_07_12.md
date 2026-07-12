# 일반화 Scaffold 아키텍처 — LOCK (2026-07-12)

> ★★이 문서 = 2026-07-12 설계토론의 박제·LOCK. **앞으로 retail-B/scaffold/상용/특허 작업은 전부 여기서 파생.** 표류 시 즉시 복귀.
> 상위 = `RESEARCH_MASTER §3`(C73) · 파생 = `TRIVIAL_REGRESSION_ABLATION`·`SCAFFOLD_STATE_ROUTER`·`INTERVENTION_LEVER_CONDITIONALIZATION`.
> 불변: [[05]] 엔진 도메인일반·A2만 · [[10]] 선택기=결정론·생성기=LLM · [[11]] learn=도메인일반·A2-swap 전이 · Δspurious≤0(모트) · gold-independence · [[09]] 무료우선.

## 0. 한 문장 원리 (LOCK)
**도메인-일반 검증 엔진(가드) + 거의-무료 A2(도구 스키마) + 일반 GET→FIND→INFER→ASK 루프 + learn(의미). 도메인 차이는 A2로만. 개입레버의 "엔진이 의도 추측"은 전부 폐기 — 대신 (구조=A2/엔진) + (의미=learn/ASK)로 분해.**

## 1. 가드 vs 개입 — 확정된 선 (C73 실증)
- **가드**(gate/prov/calc/coverage) = "정책·데이터·산술 확인" → **답이 spec 안에** → 도메인일반·Δspurious=0·상용 엔진 코어.
- **개입**(present/disamb/ground/principle/eplan as-override) = "의도 추측" → **답이 대화-순간 의도에** → 엔진 결정 시 트릭(전이깨짐) or 부작용(C73 trivial 회귀 실증). **as-override 폐기.**
- 실증(C73·task106 양방향 절단): full(개입 always-override)이 COMP-robust 6 trivial 회귀·advise가 5/6 회복(부작용원=override 기전)·106=intent 잔여.

## 2. 5개 개입레버의 일반화 분해 (LOCK·설계토론 확정)
각 레버 = **구조(A2/엔진 가드) + 의미(learn/ASK)**, **트릭 휴리스틱은 폐기**:

| 레버 | 구조 (A2/엔진·도메인일반) | 의미 (learn/ASK) | 폐기 트릭 |
|---|---|---|---|
| **disamb** | operand valid-set 탐지(스키마) + **GET→FIND→INFER→ASK 루프** | INFER 정확도·INFER→ASK 보정 | 엔진이 골라줌(most-recent/first) |
| **ground** | **미grounded 거부**(write값 ∈ tool출력∪사용자발화·보편규칙) | — (순수 가드) | 엔진이 교정/치환 |
| **principle_default** | 정책-강제 채움(=gate) | ASK(사용자 선택) | 통계 default(C58 트릭) |
| **eplan** | A2 scope 커버리지(요청타입→관련집합) | 요청-관련성 이해 | 무조건 검토(§2d 기각) |
| **present** | (스키마와 중복·불필요) | — | 후보 떠먹임(spoon-feed·C59) |

## 3. operator/operand A2 스키마 = 거의 무료 (LOCK)
모든 tool-use는 도구를 호출하려면 **이미 스키마를 가짐**(function schema/JSON Schema/OpenAPI/MCP):
| A2 조각 | 출처 | 비용 |
|---|---|---|
| operator/operand (validity·type·enum) | **도구 정의**(이미 있음) | **0** |
| provenance (미grounded 거부) | **보편 규칙**(operand-무관) | **0** |
| getter-availability 플래그 | 스키마 파생(그 필드 반환 getter 있으면) | ~0 |
| gate/정책 preconditions | **기존 정책문서** 추출 | 추출 싸고·**검증 1회/도메인** |
| coverage scope | 요청타입→scope | 소액 |
- 실증: banking A2-swap = "GB1 게이트 몇 개 + applies_when"·**엔진 리터럴 0**([[05]]). A2가 작음.
- ⇒ **가드 절반은 비용 0**·유일 실비용 = 정책-게이트 1회 검증.

## 4. GET→FIND→INFER→ASK 루프 = 출처-해소 엔진 (LOCK·기구현 C44-C48/C67)
operand 값의 출처는 고정 아님(사용자/getter/추론). A2는 **getter-available 플래그**만, 일반 루프가 해소:
```
1. GET   : A2에 getter-available → getter 먼저 (결정론)
2. FIND  : 없으면 → 사용자 발화 검색 (체크가능)
3. INFER : 없으면 → 문맥 추론 (semantic·모델)
4. ASK   : 그래도 없으면 → 질의 (일반 채널)
```
- **기구현·격리 실증**: C44-C48(4도메인·over-block 67%→0%·spurious 0)·C67(loop 0.77·**base ASK남발 21→loop ASK 0**=과질문 자동제거).
- 효과: GET/FIND=결정론 해소 → **semantic 부담=INFER 하나로 축소**·실패=ASK로 경계(날조 대신).
- task106 렌즈: GET(변형집합)→FIND("black"·사이즈X)→**INFER**("smaller"→XL)→ASK. 실패=INFER confident-wrong(black-S)+ASK 미발동.

## 5. Proven / Open 경계 (LOCK·정직)
| 조각 | 상태 |
|---|---|
| 가드(gate/prov/calc) 도메인일반 | **실증**(banking A2-swap·엔진리터럴0) |
| operand/operand A2 스키마 거의무료 | **실증**(도구정의=기존) |
| GET/FIND/ASK 루프 구조 | **격리 실증**(C67·C44-48) |
| **in-vivo 멀티턴 e2e 루프** | **미확정**(C-stage 판정) |
| **INFER 정확도 + INFER→ASK 보정** | **learn/scale·미확립**(C38/C42 SFT 퇴화·confident-wrong이 ASK 안 함) = **유일 잔여·make-or-break** |
| valid-but-wrong intent 잔여 | learn/ASK/fleet(voting=settled-neg·fleet≠voting) |

## 6. ★실험 계획 — B 78+36 (일반화 스택·다음)
> 목표: 일반화 원칙 하에서 **(a) 78 개선폭 (b) 36 무회귀(COMP-only·S0) (c) 잔여=INFER→ASK semantic만인지** 확인.
- **스택 정의 "COMP+D-v2(generalized)"**: COMP(gate+prov+calc) + ground=미grounded거부 + disamb=GET/FIND/INFER/ASK루프 + principle=gate+ASK + eplan=A2-coverage. **present 제거·개입-override 전부 제거.**
- **대조**: COMP(base) vs COMP+D-v2(generalized) vs (참고)full-override(C73 회귀본).
- **태스크셋**: 78-hard + 36-trivial. **nt=1 누적**(§0b·nt4-한방 금지) or nt≥2 rate(비결정성 커버·[[09]] 승인).
- **판독**:
  - 36: 일반화 스택서 회귀 0(=COMP·S0 무발화)이어야 함 — 개입-override 뺐으니 C73 회귀 사라질 것 예측.
  - 78: GET/FIND/coverage로 결정가능분 개선·잔여 = INFER confident-wrong(intent).
  - 잔여 per-case = **INFER→ASK 미보정**만 남는지 = learn/scale 타깃 크기(P3).
- **인프라 주의**: full-override는 subcall+gate로 sim 20-58분 pathology(C73 §8b) → 일반화 스택은 루프가 가벼워야·per-sim 타임아웃 가드·concurrency 제한.

## 7. ★Anti-drift 규칙 (LOCK·위반 시 즉시 중단)
1. **개입레버를 as-override로 재도입 금지** — 반드시 (A2-가드) + (learn/ASK)로만.
2. **도메인-특화 휴리스틱 hand-code 금지**(most-recent·first·통계 default·무조건 열거) = 트릭.
3. **present(후보 떠먹임) 재도입 금지** — 스키마와 중복·spoon-feed.
4. **"default" 개념 금지** — 정책-강제(gate) or ASK로만.
5. **엔진에 도메인 리터럴 0** — 도메인은 A2(스키마+정책)로만.
6. **pass 수치를 헤드라인화 금지**(리뷰어 공격면) — 헤드라인=가드-일반화·개입-실패·intent-경계.
7. **잔여의 정답 = ASK/learn** — scaffold로 valid-but-wrong 닫으려 하지 말 것.
8. 실험 전 이 문서 §6 + [[05]]/[[09]] 점검.

## 8. ★도구셋 완전성 — 결정론 DB-도구 도입 (2026-07-12 사용자·실무 원칙)
- **원리**: "가장 최근·가장 오래된·날짜순·내용-매칭·최고가/최저가" 같은 **well-defined 쿼리는 DB에 답이 결정론적으로 존재**한다. 도메인은 이를 조회하는 **결정론 DB-getter 도구를 노출**해야 하고, **LLM이 INFER(추측)하게 두면 안 된다** → GET/FIND 루프의 GET 단계가 결정론으로 해소.
- **재분류(잔여 축소)**: "가장 최근"(t71·t102)은 intent-undecidable이 **아니라** tool-미완(retail 출력에 날짜필드 0)일 뿐 → **날짜 getter 추가 시 GET+정렬로 결정가능**. argmax(t20)=CALC-EXT·내용매칭(t102)=FIND도 동일하게 결정가능.
- **⇒ 진짜 irreducible 잔여 = 표현-애매성(t106형·"one size smaller"=XL?S?·사용자 말이 답을 미결정)뿐.** 나머지는 완전한 도구셋 + 가드/루프면 닫힘.
- **정당성 선 (중요)**: 일반 DB-쿼리 primitive(sort/date/filter/min-max) 도입 = **데이터 접근 제공 = 정당**(답을 *계산할* 접근). vs 태스크-정답-특정 도구("사용자가 뜻한 주문 반환") = **cheating/spoon-feed = 금지**. present(후보 떠먹임)와 구분.
- **아키텍처 정합**: 도구셋 = **도메인 제공물**(A2가 스키마 기술)·엔진(gate/prov/loop) 무수정. 새 DB-도구는 A2/환경 확장이지 엔진 변경 아님.
- **벤치 vs 상용 (정직)**: 상용=도구셋 통제→날짜/정렬 getter 노출→"최근"류 해결. tau2=도구 고정→이 fail 지속=**벤치 아티팩트**(능력한계 아님·논문 명시). tau2에 도구 추가는 frontier 대조 공정성 깨니 벤치는 고정·원리만 기록.
- **미구현·후속**: (선택) tau2 retail에 `get_orders_sorted_by_date`류 추가해 t71/t102가 GET로 닫히는지 격리 probe(단 frontier 비대칭 주의). 상용 특허=이 도구셋-완전성이 "거의 다 결정가능"의 근거.
- **★"가장 최근" 해소 = 결정론(list-order)·frontier 방식은 비-uniform (2026-07-12 전수 궤적 검증·4모델)**: 
  - **정답원천 = list-order 관례(단 ★미문서화)**: 주문 리스트가 **최신순 정렬** → "최근"=`order_list[0]`(검증 n=3: t40/36/71 gold=list[0]). **★그러나 (2026-07-12 정책확인) `retail/policy.md`에 순서 명시 0·db에 날짜필드 0** → **"list[0]=최근"은 undocumented db 관례**(정책-도출 아님). ⇒ **정당한 A2 표기 불가**(정책에 없음)·frontier positional 성공=관례 exploitation(tau2 작동·원리보장 아님)·**원리적 해소=ASK**(에이전트가 "최근"을 정당히 알 수 없음). **실배포=정책에 "주문=최신순" 문서화 or 날짜필드 추가 → A2 → 결정론**(사용자 원안은 배포서 맞음·tau2가 문서화 안 함). = 순수 결정론 주장 철회·"문서화 조건부 결정론".
  - **frontier 방식 비-uniform(3전략 혼재)**: ① **positional**(리스트 첫 주문 silent 선택·지배적·t36/t40) ② **describe-confirm/ASK-list**(후보 서술·확인·t71) ③ **content-match/FIND**(내용제약 필터·t102 "시계2개"). 태스크·모델별로 섞임(claude=ASK 성향·gpt/o4=positional 성향).
  - **우리 32B 실패 = 리스트순서 오용**(t71서 `list[3]`=가장오래된 선택) + fallback(ASK/FIND) 부족.
  - **닫는 법(도메인-일반)**: 엔진 결정론 규칙 **"최근/오래된 = list[0]/[-1]"**(entity-리스트 순서·A2가 "recency-sorted" 표기) + 내용제약=FIND + 잔여애매=ASK-confirm. ⇒ t71/t102 전부 결정가능. 진짜 잔여=t106형 표현-애매성만.
  - 데이터: 궤적 `tau2-bench/data/tau2/results/final/{claude-3-7,gpt-4.1,gpt-4.1-mini,o4-mini}_retail_*4trials.json`·우리 `retail_gpt41_nogate`([[47]]). caveat: db 순서=최신순 관례를 전체 확인 필요(n=3 검증)·t71 user-sim이 gold 모를 수 있음.
  - **★신형 top 모델 검증 (S3 submissions·2026-05·user-sim=gpt-5.2)**: **qwen3.5-397b(retail 84.4·top)=71/102 둘다 4/4·*일관되게 describe-confirm/ASK*(positional 안 씀)** · gpt-5-2(81.6)=3/4·주로 ASK·일부 positional · opus-4-5(79.6)=3/4~4/4·**trial마다 방식 다름**(describe/positional/content 혼재). ⇒ **방식 비-uniform(모델간·*trial간*)**·**최고모델=일관 ASK-confirm=robust**·positional=약한 shortcut. **미문서화(정책확인) 상황선 ASK-confirm이 원리적 정답**=top 모델이 그것. ⇒ 처방=**describe-and-confirm 스킬**(애매-엔티티 write 전 서술-확인·learn/scaffold-force)·"date 도구 추가"는 배포-only 대안. 신형 retail traj=S3 `submissions/<model>/trajectories/*_retail_gpt-5.2_4trials.json`(opus-4-6/4-7=retail 미제출·banking_knowledge만).
- **★pass 이중-회계 규칙 (2026-07-12 사용자·LOCK)**: (1) **벤치/논문 = 기존-도구만** pass → frontier 공정대조(우리 도입 DB-도구 *제외*·최근류 fail 그대로 계상=정직). (2) **특허 = 도입 DB-도구를 별도 기술** + **enhanced pass**. (3) **pass를 분리 보고**: `pass_existing`(기존 도구로 닫힌 것) vs `pass_added`(우리 도입 DB-getter로 추가 닫힌 것) → "기존 X% + 도입도구 +Y%" 기여 명시. 벤치 수치엔 pass_added 혼입 금지.
