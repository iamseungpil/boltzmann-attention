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
- **M2 (양성 시)**: Stage B 학습 — v8(P6·진행중) 완주 eval + `fc_recovery_augment`(P7) 구현·학습.
- **M3**: Stage A 학습 — 확장 randomize + DPO(v9).
- **M4**: 통합 v-final + RLVR(E) + 전이 eval(chain-census 단계별).
- **M5**: 논문/특허 — 결정론 검증기 through-line(가드/라벨/보상)·2-stage 전이 헤드라인.
