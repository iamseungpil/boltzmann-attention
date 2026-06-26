# Assembled-Deterministic 한-런 설계 (2026-06-25) — convergence checkpoint·learn 질문 종결

> 근거 = `NESTED_ARM_FAILURE_CENSUS` §4.5 정본 레버 지도(robust·stable-20·gold-diff). disentangle가 "40% capability-under-load" 기각(=user-sim 노이즈)→ learn 질문 재오픈·**operand 9건(45%)으로 환원**. 사용자 지시(2026-06-25): (a)calc_NL·(b)operand census를 *직렬 말고* **한 assembled-deterministic 런**으로 병합(gold-diff가 이미 de-confound·직렬 불요). GPU 1회로 모든 깨끗한 결정론을 max한 뒤 잔여=진짜 learn-or-capability 격리.

## 0. 목표 (왜 한 런)
정본 잔여(stable-20): operand 45%·calc_NL 20%·no-write 20%·over 15%. 직렬(a→b) 대신 **present-개선 + calc_NL-compute + 기존게이트(+disjoint)를 한 스택**으로 retail 재실행 → gold-diff 잔여 census. 이 한 수 = operand를 결정론 present로 공격(b 첫 substep·Probe-B 가설 라이브 검증) + calc_NL bank(a 동승) + over 흡수 → **남는 잔여 = 모든 결정론 max 후 = 오염 없는 make-or-break(learn-or-capability).**

## 0.5 ⓐ 소스 확인 결과 (2026-06-25·GPU-free·[[05]]+enumerate-not-resolve)
- **present 구현 = 이미 enumerate-not-resolve**: `_present_candidates`(L150)·`candidate_summary`(L290)·`nested_candidate_summary`(L323) 전부 `shown={f:rec.get(f) for f in fields}` = **as-is 필드 dump·criterion 필터 0·join resolve 0**(`len(ids)>1` 게이트). 코드상 rig 위험 부재.
- **★⋈ present = 새 resolve 불요·공정**: G6 `present_fields=["status","address","items"]` → 각 주문 주소 *전부 열거* → 모델이 "어느 주문 주소" 스스로 판단(=측정 대상 operand-formalize가 모델에 잔류). **공정 ⋈ = 기존 order-enumerate-with-address**·"다른 주문 주소 직접 제시(join-resolve)" *추가 금지*가 유일 경계.
- **calc = general aggregate op 신규**(count-where·sum)·경로 순수 A2-path(`variants[].available`·`items[].price`) = [[05]]-clean 소량 확장.

## 1. 스택 구성
`present-개선 + calc_NL-compute + 전체게이트(auth·confirm·ownership·notice·preconditions) + new≠old(disjoint)`, 32B+14B, retail, **multi-trial(≥3·robust + user-sim 노이즈 평균화)**, gpt-4.1 user-sim, replay-safe.

## 2. ★Mechanism 1 — present-개선 (Probe-B 품질·operand 45% 공격)
- **현 nested present의 한계**: 필드 dump이나 (i)결정점 정렬 약함 (ii)⋈(cross-entity·다른주문 주소) 미포함 (iii)raw 완전성 부족 → present-arm이 operand "약간만" 닫은 이유(task71류 무시).
- **개선 = Probe-B 7/7 됐던 *그 형식* 복제**: 결정점(write 직전)에 **raw 후보를 명시 choice-set**으로 — 중첩 dict 그대로. 단순 fetch 아님(=무시됨·task71).
- **★★enumerate-not-resolve 경계 (#1·make-or-break 타당성의 뿌리)**: present는 **raw 엔티티 집합을 *열거*만**(네 주문들+각 주소·각 item들·변형들). **criterion으로 미리 거르거나 join을 미리 풀면 금지**(=scaffold가 operand-formalize 대행=측정 rig·[[05]] Q3 yes).
  - 공정(σ): "네 주문들+각 상태/주소[raw]" → 모델이 'DC'·'delivered 주문 주소'를 *스스로* 매칭(=Probe-B 공정).
  - **불공정(금지)**: "다른 주문의 주소"를 *직접* 제시 = scaffold가 ⋈ join 수행 → "present-closed=결정론승리"가 동어반복·operand-formalize 영영 미측정.
- 구현: ⓐ 확인대로 **새 resolve 불요** — 기존 enumerate present(as-is dump)가 결정점에 raw·완전하게 발화하도록만(⋈ 주소는 G6 present_fields에 이미 열거). **join-resolver 추가 절대 금지.**
- **가설**: Probe-B(격리 7/7)가 라이브서도 되면 → operand 다수가 **present-형식약함=결정론-fixable**(learn 아님). 안 되면 → capability/learn.

## 3. ★Mechanism 2 — calc_NL-compute (계산/집계 offload·20% bank)
- calc_NL 실패 = {산술(환불총액)·집계(가용 변형 수)·조회(tracking#)→보고}. 모델이 available 필터 안 해 오산.
- **offload = 엔진이 파생사실 결정론 계산·주입**(read 응답 증강): 예 get_order_details→"items 총액=$X"·get_product_details→"available 변형=N개"·order→tracking#. **계산은 결정론·보고는 모델.**
- [[05]]: 엔진=일반 aggregate 연산{sum·count-where·lookup}·A2=retail이 어느 필드/조건. (별도 소-스펙 `calc_specs` in A2.)
- **★report-conversion 측정(필수·[[06]] lever≠resolution)**: 주입해도 모델이 *말 안* 할 수 있음(present order-pick과 동일 리스크). 주입↔보고 전환율 별도 측정.

## 4. 측정 ([[06]]/[[08]])
- **pass^all(robust=fail-all-3) 1차·pass^1 단독 금지**(노이즈 0.19). 결정론 action-census(escape_det_census) 병행.
- **★per-task 이중확증 (#4·작은-n 귀속)**: "present가 task X 닫음" = **baseline robust-fail → assembled robust-pass *AND* 그 write가 gold-correct(action-census write-correctness)**로 이중. pass만으론 불충(1-2건 노이즈 스윙이 헤드라인 흔듦). 전후 per-task 명시.
- crash/infra/too_many 배제(--clean)·종료분포 먼저. user-sim 노이즈=multi-trial·불일치류(경로분산 7) 분리.
- over-deny 체크(disjoint·present가 양성 막나).
- baseline = present+nest+g15(현 정본 `*_presentnest_g15_retail_t3`).

## 5. ★잔여 태깅 (make-or-break 종결·gold-diff)
런 후 reward=0(robust) 잔여를 배정:
| 잔여 | 의미 | 레버 |
|---|---|---|
| **present-closed operand** | present-개선이 닫음 | **결정론 승리**(learn 아님) |
| **present-but-wrong operand** | raw choice-set 줘도 틀림 | **capability-or-learn = priority-4 SFT 유일 타깃** |
| **calc_NL (★2-way 분리·리뷰 2026-06-26)** | calc 범위 = available-count(count_where)·order-total(sum)만 | ↓ |
| ── 계산 주입했으나 미보고 | report-conversion 실패 | prompt/scaffold |
| ── 애초에 calc 범위 밖 | **subset-refund($918.43=반품품목만 합·t104/t28류)·tracking#(lookup spec 無)** | **calc 확장 필요**(report-conv 실패 아님) |
| no-write/orchestration | 도달 실패 | recovery/auth·user-sim |
- **★공정 present → copy축 소거 (#2·종착 질문 단순화)**: present가 *값을 열거*하면(주소·item_id·변형 화면에 있음) **copy-fabrication이 구조적으로 제거**(날조할 게 없음). ⇒ **present-but-wrong = 순수 selection/criterion-formalize = non-C4 (by construction).** 즉 §5의 "C4-copy vs criterion-formalize" 갈림이 present가 값 열거하는 한 *자동 해소* → present-but-wrong = **learn-able criterion-formalize 한 종류 = learn-GO 후보**(C4 전이음성 아님). *단 #1 경계(enumerate-not-resolve) 지킬 때만.*
- **present-but-wrong operand = 이 arc 전체의 종착 질문**·위 #2로 learn-GO 후보(criterion-formalize)로 깔끔히 떨어짐(C4 dead-end 아님).

## 6. NO-GO / 분기 (종결 조건·3-way)
- **분기1 — present가 operand 대부분 닫음** → **결정론 천장↑·learn NO-GO**·헤드라인=결정론+TCO([[06]]).
- **분기2 — present-but-wrong이 *측정가능하게* 크고 (값열거라 non-C4)** → **learn GO**(priority-4 SFT·`A2_RULE_USE_SFT_PREP`·이 criterion-formalize 잔여만 타깃).
- **★분기3 — present-but-wrong이 *존재하나 너무 작음*(~3-5 task·#3)** → learn-GO도 NO-GO도 아닌 **"잔여 too-small-to-measure(노이즈 0.19+user-sim 경로분산서 검출불가) → broader-eval 필요"**. thesis 학습주장=도메인-일반 전이라 어차피 eval은 tau2 ~5건 아니라 **벤치-횡단**이어야 → 이 분기가 **priority-4 eval을 tau2-단독서 벤치-횡단으로 설계 전환**으로 연결(SFT 효과는 SOP/TB/Synth held-out + 다도메인 A2-swap서 측정·tau2는 전이확인 1점).

## 7. 선행 (구현 전) — ⓐ로 대부분 확정
- ✅ **⋈ present**: 새 resolve 불요·기존 enumerate(G6 present_fields에 address)로 공정·**join-resolver 추가 금지**가 유일 경계(§2·§0.5).
- ✅ **calc aggregate**: count-where/sum=도메인-일반 op·경로(`variants[].available`·`items[].price`)=순수 A2-path([[05]]-clean). 엔진 소량 확장(2 op)+A2 `calc_specs`.
- GPU = priority-2 종료 후(현 present-nest 런 점유). **구현=GPU-free 가능 지금**(엔진 calc op·A2 calc_specs·present 결정점-발화 확인·드라이버).
- (이 doc은 `ASSEMBLED_STACK_CENSUS_DESIGN`의 Phase1+3을 구체화·병합 = 그 doc의 operative 버전.)
