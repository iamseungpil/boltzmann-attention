# τ² 전이 풀체인 수정 설계 (통합·리뷰용) — 2-stage gate 병목 + 프로토타입-우선

> 상태 = **리뷰용 통합 설계**(승인 후 구현). 통합 대상 = `V9_ANTIFABRICATION_DESIGN_2026_06_15.md`(상류 P2b)·`SYNTHESIS_IMPL_SPEC_2026_06_15.md`(P6/P7 합성)·`R1B_PROVENANCE_DESIGN_2026_06_14.md`(검증기). 진입점 = `HANDOFF_2026_06_15.md`.
> 불변 = `feedback-thesis-tbox-transfer-direction`(SOPBench/TaskBench 학습·τ² held-out)·`feedback-selector-verifier-deterministic`(검증기=결정론·LLM=생성기).
> ★방법론 = **프로토타입-우선**: 학습은 오래 걸림 → *무재학습 결정론 가드레일*로 "모든 원인 동시 차단 시 pass 이동?"을 먼저 확인 → 양성이면 본 학습.

## 1. 통합 근본원인 — τ² 실패는 2-stage gate 체인 (전수 census 확정)
체인-census(`tau2_chain_census.py`·n=20)가 단계별 병목을 분리 확정:

| 단계 | BASE | L1(anti-fab) | 병목 원인 |
|---|---|---|---|
| auth | 13 | 17 | **P2b 날조**(스키마-example `#W0000000`·email) |
| gather(get_user_details) | 8 | 12 | ↑(상류 P2b가 막던 것) |
| real_order 추출 | 8 | 12 | (추출은 100% 생산적·P4 문제 없음) |
| write 도달 | 7 | 10 | ↑ |
| **write 성공(PASS)** | 1 | **1** | **★벽** |

**→ 병목이 *단계 분리*돼 있다:**
- **Stage A (상류 auth/order)** = **P2b fetchable-값 날조**(모델이 τ² 스키마 example 복사). anti-fab(L1)이 이 단계 통과율을 끌어올림(실증) — 하지만 거기서 끝.
- **Stage B (write)** = **P6 confirm 미수행 + P7 recovery 실패**. dump 확정:
  - task3: `G2_CONFIRM_WRITE blocked`(confirm 필요) → confirm 없이 **동일 write 6연타** → too_many_errors.
  - task0: `Non-delivered cannot be exchanged`(P5 정책) → **동일 호출 9연타**.
  - = 게이트-블록 후 **P7 retry-loop가 write 단계 지배**(상류선 P7 작동했으나 write-게이트엔 무력).

**확정 결론**: anti-fab 단독은 **상류만 뚫고 write 벽서 막힘** → write_ok 1/20 불변. **Stage A(P2b) + Stage B(P6+P7) 동시 수정해야 pass 이동.** (gather/추출/P4는 멀쩡 = 처방 불요.)

## 2. 통합 처방 = 결정론 검증기/게이트가 3 실패를 동시 차단
모든 실패는 **결정론으로 검출 가능** → 검증기 하나가 가드(프로토타입)·라벨러(DPO)·보상(RLVR) through-line.

| # | 실패 | 결정론 검출 | 처방(런타임 가드 = 학습 타깃) |
|---|---|---|---|
| **G-fab** | fetchable 값 날조(스키마-example) | arg값 ∉ {user∪tool 출력} | 내부 재생성→gather 유도(L1 bad_words + L2 검증기) |
| **G-confirm** | 비가역 write 전 confirm 미수행 | write 호출인데 직전 user "yes" 부재 | "먼저 user에 확인 요청" 안내 → 모델이 ask → user 승인 → write |
| **G-loop** | 게이트/에러 후 동일 호출 반복 | tool_call == 직전 *실패* 호출 | 차단 + "이미 실패함; 반복 금지·전략전환(confirm/re-gather/대안)" |

- **G-confirm·G-loop은 *실제 대화*서 작동**(user 승인이 필요하니 내부-재생성 불가) — orchestrator 게이트가 안내 surface, **G-loop이 연타를 끊어** 모델이 안내를 *읽고* 행동하게 강제.
- **G-fab은 내부-재생성**(user 불요·gather로 전환).
- 핵심 = **G-loop**: task3/task0이 연타로 budget 소진해 죽었음 → 연타만 끊으면 모델이 게이트 안내(confirm/정책)에 반응할 *기회*가 생김(task6: 막히면 ask/전환 *할 줄 안다*).

## 3. ★Phase P — 통합 프로토타입 (무재학습·결정론 가드레일)
**목적**: "3 실패 동시 차단 시 pass 이동?" → C/D/E 학습투자 게이트.
- 구현 = `t2_gate_patch.py` 확장: 기존 G1-G4(작동중) + provenance-regen(G-fab·구현됨) + **G-loop 추가**(orchestrator: 직전 실패와 동일 (name,args) 호출 시 실행 차단·강한 redirect) + G-confirm 안내 강화(이미 G2 surface·G-loop이 반응 강제).
- **측정**: `tau2_chain_census.py` 단계별 통과율 + pass^1 + 게이트별 준수율(confirm-before-write·no-loop·no-fab).
- **arm**: BASE / +G-fab / +G-fab+G-loop / +G-fab+G-loop+G-confirm(full). 단조 개선 + 어느 게이트가 어느 단계를 푸는지 분해.
- **★판정 게이트**: full 가드레일서 **write_ok 1→다수·pass 0.05→유의(예: 0.3+)** = 원인 확정·학습 정당. 미이동 = 더 깊은 원인(재진단). **caveat: 프로토타입은 천장 추정(런타임 가드의 상한)** — 학습이 그만큼 내재화하는지는 별도.
- **비용**: GPU 1개·~30분/arm·무재학습. (학습 1회 = 수 시간 + eval → 프로토타입이 10배 싸다.)

## 4. Phase T — 본 학습 (Phase P 양성 후에만) — 가드를 weight에 내재화
각 런타임 가드를 학습으로 내재화(검증기 = 라벨러/보상):

### Stage A 내재화 (P2b) — v9
- **C. 확장 randomization**: 전 fetchable 값(order_id·payment_method_id·item_id·address)을 format-보존 랜덤화 + tool 출력에만 등장 → **fetch-first 구조 강제**(맞추려면 getter 먼저).
- **D. DPO**: chosen(fetch-then-copy) vs rejected(스키마-example 날조) 합성쌍·검증기 라벨·양방향.
- **E. RLVR**(양성 시): on-policy·보상=task성공∧무날조∧fetch-first = gate-in-loop(Track B).

### Stage B 내재화 (P6+P7) — v8 + recovery
- **P6 confirm**: `fc_confirm_augment.py`(구현됨·v8 학습중) — confirm-then-write SFT(pos)+neg(no→미실행). 반환시그니처 분류.
- **P7 recovery**: `fc_recovery_augment.py`(스펙) — error-injection SFT(게이트-블록→재시도 금지·전략전환) + **gate-in-loop RL**(원형). **G-loop이 직격하는 "동일행동 반복 금지"가 학습 타깃.**

### 통합 레시피(잠정)
v-final = sft_v7(P2b소스) + 확장-randomize SOPBench(C) + sop_confirm(P6·v8) + sop_recovery(P7) + DPO쌍(D) → (양성) RLVR(E). 전부 SOPBench/TaskBench서·τ² held-out.

## 5. 평가 / 사전등록
- **헤드라인**: τ² pass^1(held-out·키 source 필수).
- **기제(사전등록·`tau2_chain_census.py`)**: 단계별 통과율(auth/gather/추출/write도달/write성공) + 게이트준수율(no-fab·confirm-before-write·no-loop). **단계별 개선이 어느 처방에 귀속되는지 분해**(C→상류·P6/P7→write).
- **예측**: Phase P full → write_ok↑·pass↑(상류+write 동시 해소). Phase T가 그 천장을 weight로 재현하면 전이 성립.
- ablation: 각 가드/처방 leave-one(어느 단계가 어느 처방에 의존).

## ★★9. Randomization 전수 재검토 = 표면 등방화 (원리 반영·2026-06-15 사용자 지시)
> 원리(마스터 §0 ★★★★ + `ALGEBRAIC_DERIVATION_CLOSURE` §5.10/5.11/5.14): **전이 = 모델이 표면-불변(저차원 추상) 부분공간에 안착.** randomization = 표면군 G 등방화 → 불변(추상) 강제. **§5.10 LODO 교훈(정리): *덮인 표면 차원만* 전이 → train↔test서 *변하는 모든 표면 dim*을 randomize해야** (미커버 dim = 과적합 = 전이실패). Olver 실험 = 등방화 작동 확증(var-side robust). **⇒ randomization 설계 = "값 더 섞기"가 아니라 *미커버 표면 dim 전수 식별·커버*.**

### 9.1 표면-차원 감사 (train SOPBench ↔ test τ²)
| # | 표면 dim | 표면군 | train↔test 갭 | 현 status | 처방 | primitive | enforce=offload? |
|---|---|---|---|---|---|---|---|
| 1 | tool/field **이름** | naming(치환) | τ² 다른 이름 | alias 부분(native-FC R1) | 전역 alias-mask 강화 | P1 | 학습 |
| 2 | **fetchable 값**(order_id·payment·item) | value(reformat) | τ² placeholder | ✅ **v6 `fc_randomize_fetchable`**(identity+fetchable) | 커버됨·τ² 값-형식 span 확인 | P2b | 학습(+검증 offload) |
| 3 | **스키마-example 값**(`#W0000000`) | (artifact·**pretraining prior**) | τ² tool 스키마 example | ✗ 미커버 | **★randomize로 불가**(pretrain prior) → **DPO negative + runtime blocklist**(bad_words) | P1 | runtime gate offload |
| 4 | **출력/직렬 format** | format | LODO resource↔temporal | ✗ **미randomize=전이실패 실증** | **format-uniform/randomize**(신규) | R4 표면 | 학습 |
| 5 | **게이트 phrasing**(confirm Q·error/deny msg·정책 NL) | (speech-act surface) | τ² 다른 wording | ◐ sop_confirm 4템플릿(스펙) | template+LLM paraphrase randomize | P6/P7 | enforce=offload·phrasing=학습 |

### 9.2 ★핵심 통찰 (왜 v6 값-randomize했는데 τ² 전이 실패했나)
- **dim 2(값)는 이미 대부분 커버** — but 전이 실패. 원리가 정확히 진단: **미커버 dim 3·4·5가 전이 안 됨**(§5.10).
  - **dim 3(스키마-example)**: 날조의 *진짜* 소스(이번 세션 root-cause). **pretraining prior라 randomize로 못 지움** → DPO negative(스키마-example=rejected) + runtime bad_words. ← v6 값-randomize가 *놓친* 축.
  - **dim 4(format)**: LODO가 *직접 실패*로 입증(resource↔temporal 직렬화 미randomize→전이실패). **format randomization 신규 필수.**
  - **dim 5(gate phrasing)**: P6/P7 전이의 표면 — confirm/error/정책 NL을 randomize해야 *추상 speech-act*(confirm·recover) 학습(특정 문구 과적합 아님).
- **offload 분리(원리 반영)**: 각 dim의 *enforcement*(게이트 집행·값 검증)=결정론 offload / *abstract interaction*(언제 confirm·어떻게 gather·deny후 recover)=LLM 학습. **randomization은 후자의 표면만 등방화**(전자는 결정론이라 randomize 무관).

### 9.3 v8 (P6) 재검토 — 원리 반영
- P6 = 층B policy primitive. 게이트(G2 confirm)=결정론 offload(됨)·모델은 *추상 confirm speech-act* 학습.
- **전이 위해 sop_confirm이 전 randomization 상속 필요**: 값(dim2)·이름(dim1)=sop_rand2 상속 ✅ / **confirm phrasing(dim5)=template+LLM paraphrase 강화**(현 4템플릿→다양화). → 모델이 특정 문구 아닌 추상 "비가역 write 전 확인" 학습.
- **현 v8 gap**: sop_confirm이 sop_rand2(값/이름 randomized) 기반이나 **confirm Q 문구 다양성 부족**(dim5 미흡) → 재빌드 시 paraphrase 확대.

### 9.4 v9 (anti-fab) 재검토 — 원리 반영 (Stage A 재정의)
- ~~"값 더 randomize"~~ → **미커버 dim 추가가 핵심**:
  - dim 2(값): 이미 v6 커버 — 유지.
  - **dim 3(스키마-example)**: DPO negative 쌍(chosen=fetch·rejected=스키마-example 날조) + runtime bad_words. ← Stage A의 *진짜 신규*.
  - **dim 4(format)**: 학습데이터 출력-format randomize(JSON 직렬·key순·separator) = LODO 처방을 등방화 framing이 예측(§5.10).
- ⇒ **v9 Stage A = dim3(DPO/blocklist) + dim4(format-rand) 신규** (dim2는 유지). RLVR 보상 = 전-체인 task성공(§4).

### 9.5 Olver 진단 확장 (randomization coverage 사전탐지)
- Olver 실험(`olver_dimension_experiment.py`)을 **dim별 coverage 진단**으로: 각 표면 dim을 augment群에 넣고 그 dim이 isotropize되는지(var↑·inv↓) 측정 → **미isotropize dim = 전이 위험 사전탐지**(학습 전 zero-cost). 특히 dim4(format)·dim5(phrasing)가 모델 표현서 분리 등방화되는지 확인.

## 6. scope / caveat (정직)
- **프로토타입=천장 추정·가드는 결정론(프로덕션 가드로도 유효)·학습=내재화**. 둘 다 보고.
- **write 천장 다인자**: P6(confirm)·P5(정책 위반 적응)·P7(루프). G-loop이 공통 분모(연타 차단)지만 P5 정책-적응(대안 행동)은 별 능력일 수 있음 → Phase P서 분해.
- **L1L2 auth false-positive**(census: 17→9) = 검증기 오판(legit name/email 차단) → **고정밀화 선결**(스키마-example literal blocklist·context-subtraction 정확도).
- **transfer 미보장**: v7(CFB)도 randomize였으나 전이 실패 → 처방이 *런타임선 작동(Phase P)*해도 *학습 전이*는 별 검증(Phase T). 프로토타입-우선이 이 리스크를 싸게 분리.

## 7. ★열린 질문 (리뷰 훅)
1. **G-loop 정의**: "직전 실패와 *완전 동일*" vs "유사(같은 tool·다른 변형 arg)"? 너무 엄격하면 정당 재시도 차단·너무 느슨하면 false-positive.
2. **G-confirm 프로토타입**: user 승인이 필요 → 내부-재생성 불가. G-loop이 연타 끊으면 모델이 *스스로* confirm-ask하나? 아니면 명시 "ask for confirmation" 안내 필요? (task6: 막히면 ask 할 줄 앎 = 낙관적.)
3. **P5 정책-적응**: non-delivered 같은 정책 위반은 "반복 금지"(G-loop) 넘어 *대안 행동*(다른 주문·user 설명)이 필요 — 이게 학습가능 단일 스킬인가 별 능력인가.
4. **프로토타입→학습 천장 갭**: Phase P가 pass 0.3 만들어도 학습이 0.1만 내재화하면? (런타임 가드 유지가 프로덕션 답일 수도 — 결정론 가드는 thesis 정합.)
5. **검증기 false-positive 비용**: 학습 보상으로 쓸 때 오판이 모델 오학습 → 고정밀 선결.
6. **순서**: Phase P 어느 arm까지 확인 후 학습 착수? (full 양성만? 아니면 G-fab+G-loop만 양성이어도 v9+P7 착수?)

## 8. 마일스톤
- **M0 ✅**: 3-arm provenance prototype → 상류 P2b 레버 확정·**write가 진짜 벽**(체인-census). 본 문서의 출발.
- **M1 (다음·프로토타입-우선)**: **G-loop 구현**(`t2_gate_patch` orchestrator·동일-실패 차단) → **full 가드레일 4-arm 프로토타입**(BASE/+fab/+loop/+confirm) → pass 이동 판정. **이게 본 학습 게이트.**
- **M2 (양성 시)**: Stage B 학습 — v8(P6) 완주 eval + **sop_confirm 재빌드(dim5 confirm-phrasing 다양화·§9.3)** + `fc_recovery_augment`(P7·dim5 error/deny phrasing randomize) 구현·학습.
- **M3**: Stage A 학습(v9) — **§9.4 재정의**: dim2(값·유지) + **dim3 DPO negative(스키마-example)+bad_words** + **dim4 format-randomize(신규·LODO 처방)**. ~~"값 더 randomize"~~ 아님.
- **M3.5 (zero-GPU·선행)**: **Olver dim별 coverage 진단(§9.5)** — dim4(format)·dim5(phrasing)가 모델 표현서 등방화되는지 학습 전 측정 → 미커버 dim 사전탐지.
- **M4**: 통합 v-final + RLVR(E) + 전이 eval(chain-census 단계별).
- **M5**: 논문/특허 — 결정론 검증기 through-line(가드/라벨/보상)·2-stage 전이·**원리(유한-학습+offload+표면등방화)** 헤드라인.
