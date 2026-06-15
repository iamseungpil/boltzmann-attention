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
- **★판정 게이트**: full 가드레일서 **write_ok 1→다수·pass 0.05→유의(예: 0.3+)** = 원인 확정·학습 정당. 미이동 = 더 깊은 원인(재진단).
- **★★가드 2분류 (리뷰 비판4 — 천장 gap이 아니라 scaffold 분담)**: Phase P가 측정하는 건 **{모델+scaffold} 천장**. 가드를 명시 구분:
  - **(a) 남는 것 = 결정론 soundness leg** (G_term·G1-G4 게이트 집행·값 *검증*) — **내재화 대상 아님**. 결정론으로 영구 담당이 *정답*(thesis 정합·§5.12 구체/게이트=A2·scaffold).
  - **(b) 학습 타깃 = 추상 interaction** (P2b gather·P6 *timing*[언제 confirm]·P7 *전략전환*[deny후 무엇]) — 모델이 내재화할 저차원 추상.
  - ⇒ "Phase P 천장 0.3인데 학습이 0.1만 내재화"는 **gap 아니라 "0.2는 scaffold가 영구 분담"** (§5.12 추상-전이+구체-A2·scaffold 정합). **= §7-Q4 답.** ⚠️단 가드 일부(G-fab 재생성)는 env가 값 알아야 가능 → 그 부분은 (a)로 분류(모델이 fetch 없이 못 함=결정론 보강이 옳음).
- **caveat**: 프로토타입 천장 = (a)+(b) 합. *전이 헤드라인*은 (b) 내재화분만 + (a) scaffold 상시동반 명시.
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

### 9.1b ★감사 완전성 = load-bearing (리뷰 비판2 — G_term 맹점 재발 방지)
- **위험**: 원리("train↔test 변하는 *모든* dim randomize") → **미커버 dim 하나가 전이를 죽임**(§5.10). ⇒ 감사 *완전성*이 load-bearing인데 dim 1-5는 **육안 나열** = τ²-경험이 G1-G4 주고 **G_term 놓친 그 맹점과 동형**. dim 6 부재 보장 없음.
- **★처방 (육안→도출)**: dim 목록을 **표면 군 구조서 도출** — "SOPBench 궤적 → τ² 궤적 *presentation*으로 보내는 변환들의 군 G"의 **생성자가 곧 dim**(육안 아님). G = (이름치환 × 값reformat × 직렬format × 게이트-phrasing × ...) → 생성자 전수가 dim 전수.
- **★자기-검증 (Olver-per-dim 완전성 체크·§9.5)**: dim 1-5 *전부 커버 후* **잔여 전이가능-분산이 0인가** 측정 → **>0이면 dim 6 존재**(미식별 표면). 이래야 감사가 자기-검증(육안 신뢰 아님). = G1-G4 맹점의 구조적 재발방지.

### 9.2 ★핵심 통찰 (왜 v6 값-randomize했는데 τ² 전이 실패했나)
- **dim 2(값)는 이미 대부분 커버** — but 전이 실패. 원리가 정확히 진단: **미커버 dim 3·4·5가 전이 안 됨**(§5.10).
  - **dim 3(스키마-example) — ★재프레임(리뷰 비판3): 새 dim 아니라 dim-2를 이긴 prior.** v6가 dim2(fetchable 값)를 randomize *했는데도* 모델이 `#W0000000` emit = **미커버 dim 아니라 dim-2 커버리지가 pretraining prior에 *패배***. ⇒ dim3 = "이미 커버한 dim에서 pretrain prior > 훈련신호인 실패모드". **함의(핵심)**: 레버 = ~~"dim 더 커버"~~ 만이 아니라 **"이미 커버한 dim에서 훈련신호가 prior를 *압도*해야"** → 답 = 더 많은 randomization 아니라 **더 강한 신호(DPO-negative)** + runtime bad_words(prior 직접차단). randomization(등방화)은 *훈련서 랜덤화 가능한 dim*에만 작동·prior-collapsed dim은 다른 도구(대조신호) 필요 = isotropization 원리의 *경계*.
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

### 9.5 Olver 진단 확장 (randomization coverage 사전탐지 + 완전성 자기검증)
- Olver 실험(`olver_dimension_experiment.py`)을 **dim별 coverage 진단**으로: 각 표면 dim을 augment群에 넣고 그 dim이 isotropize되는지 측정 → **미isotropize dim = 전이 위험 사전탐지**(학습 전 zero-cost).
- **★완전성 자기검증(§9.1b)**: dim 1-5 *전부* augment群에 넣고 **잔여 between-input 분산(불변부)이 task-구조만 남기는가** 측정 → 잔여에 *아직 표면-상관* 성분 있으면(probe로 surface 예측 가능) **dim 6 존재** = 감사 미완. = G_term 맹점의 구조적 탐지.
- **★just-so 회피(비판1 준용)**: 이 진단도 **inv-측(불변 붕괴)으로 판정**(var-측은 거의 동어반복). dim 커버 시 *inv_dim이 그 dim 방향만큼 감소*하는지 — 사전등록. var만 보면 무정보.
- caveat: §9.5 진단은 **base/trained 둘 다** (전제-스크린은 base·전이주장은 trained). 현 Olver 1차=base=전제-스크린(§5.14 M-Olver 강등 참조).

### 9.6 ★★LLM 강/약 기준 역적용 — dim 분류 + "절차의 로마→아라비아" (`ALGEBRAIC_DERIVATION_CLOSURE` §5.15, 2026-06-15)
> §5.15 3단계 기준(유한생성×표현노출×깊이외부화)을 §9.1 감사에 *역적용*. **핵심 결과: dim 1·2·4·5 = step-2 표면(등방화로 노출=절차판 아라비아숫자) / dim 3 = step-2 아님(가중치-내 prior).** 절차는 step-1(BJ 유한)·step-3(궤적이 루프 외부화·공짜) 통과 → **step-2 하나만 문제 → 절차가 산술보다 구제 쉬움**(§5.15 예측).

| dim | 가린 불변량 | §5.15 분류 | fix 원리 | 측정(약→강) |
|---|---|---|---|---|
| 1 이름 | 역할(의존구조) | **step-2 데이터-표면** | 등방화(alias) → 역할 노출 | alias on/off LODO |
| 2 fetchable 값 | provenance(tool출력) | **step-2 데이터-표면** | 등방화(rand) → fetch 강제 | ✅v6 |
| **3 스키마-example** | (없음·이미 노출) | **★step-2 아님 = 가중치-내 prior** | **재표현 불가**(데이터 randomize는 *데이터* 표면만 등방화·*가중치* prior 못 건드림) → DPO-neg+blocklist(prior 압도/차단) | DPO A/B |
| 4 format | 추상 시퀀스 | **step-2 데이터-표면(★clean)** | 등방화(format-rand) = 로마→아라비아 정본 | **LODO=without arm·format-uniform=with arm** |
| 5 phrasing | speech-act(confirm/recover) | **step-2 데이터-표면** | 등방화(paraphrase) → act 노출 | phrasing on/off |

- **★dim 3 재확정(비판3 정리화)**: 데이터-등방화는 *데이터 표면*만 등방화. dim3(`#W0000000`)은 **pretraining이 가중치에 박은 고정 비등방 방향** → 데이터 randomize 도달불가(=v6 값-randomize했는데 날조한 이유). step-2 표면 dim이 *아니라* "가중치-prior가 이미 노출된 불변량을 *이김*". fix = 재표현 아닌 **prior 압도(DPO-neg 강신호)+runtime 차단**. ⇒ 감사 taxonomy = {데이터-표면 1·2·4·5: 등방화} ⊎ {가중치-prior 3: 압도/차단}.
- **★측정 = 절차판 산술-토큰화 clean experiment(§5.15 step-2 실증 차용)**: dim4가 가장 깨끗 — **without-등방화(LODO·단일 format 학습)=약 vs with-등방화(format-uniform)=강**. 산술의 Nogueira/McLeish("표현 교체→약→강")의 *절차 인스턴스*. ⚠§5.15 잔여④: 원리는 차용·절차서 약→강은 *이 측정으로만* 확정(편의데이터 격상 금지).
- **★비자명 예측(§5.15)**: 절차는 step-3(궤적) 공짜라 **step-2 fix(등방화) 하나로 전이 복원** — 산술처럼 CoT(step-3) 추가 불요. ⇒ format-uniform 단독이 LODO 전이실패를 뒤집으면 = "절차가 산술보다 구제 쉬움" 실증 + step-2 단독충분 확인.

### 9.7 ★★orbit-consistency = 명시적 불변-손실 (데이터-노출 → 불변 *강제*·`ALGEBRAIC_DERIVATION_CLOSURE` §5.11/5.12, 2026-06-15)
> 사용자 통찰: 저차원 불변 전이의 실제 구현. **전이 메커니즘 불변(동결 LoRA+ABox-swap=재학습0)** — orbit-consistency는 *학습-시점* LoRA를 불변 부분공간에 정렬시키는 손실. = §9 데이터-등방화의 "노출"을 "강제"로 격상.

**기제.** 한 task의 *궤도(orbit)* = 표면군 G 변형들(alias·format·value-rand)의 집합. 손실 `L_inv = ‖h(x) − h(g·x)‖²` (g∈G) → 변형 표현을 *같게 강제* → 변형 간 *불변(추상)* 살아남고 *변이(표면)* 페널티. = **Reynolds 투영 P_G의 *학습판***(§5.11)·**대조학습(SimCLR/BYOL invariance)과 동원리**(augment 군=우리 표면군).
- **노출(§9) vs 강제(9.7)**: 데이터-randomize=표면 *노출*(무시 *희망*) / orbit-consistency=표면-민감도 *직접 페널티*(불변 *강제*). 더 강한 신호.
- **전이 불변**: LoRA가 학습 중 불변하게 변할 뿐, 전이=동결 LoRA+ABox-swap 그대로(웨이트 변경 0). "TBox 웨이트 변경?"=학습 중 YES·전이 시 NO.

**★결정적 caveat 2(설계 제약)**:
1. **변환 G = *구성상 의미보존*만**: order_id가 표면-placeholder 아닌 *실제 fetch대상*이면 randomize 금지(consistency가 *주의할 값 무시* 오학습). 이름·포맷·직렬화=OK / 실제 결정값=NO(=dim3 prior와 별·§9.6). Locatello/coverage 핵.
2. **★계획-수준만·실행 grounding 표면-가독 유지(binding 긴장)**: 전체 불변화하면 `get_user_details`를 *읽어 호출* 못함. **계획구조(어느 의존)=불변·실행 grounding(지금 어느 도구)=표면가독** 둘 다 필요 → orbit-consistency = **soft 정규화·계획-수준 표현 한정**(blanket 하드 제약 금지). = §5.10/5.12 binding 잔여의 구체판.

**★probe 게이트(선행)**: invariant-probe(`olver_definitive.py`·`0fe14a6`)가 **불변 부분공간이 실제 task 전이하나** 먼저 측정 → 양성(caveat2 안깨짐)이면 soft orbit-consistency를 §9에 얹음·음성이면 불변≠task라 헛수고. **진단→양성 시 구현.**

**위치(§5.16 원장)**: orbit-consistency = **C8(전이) 음성→양성 후보 메커니즘**(데이터-등방화보다 강한 레버)·아직 결과 아님=가설/방법. invariant-probe 결과로 판정.

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

### 7.1 ★Q-판정 (리뷰 2026-06-15 — 답 박제)
- **Q1 G-loop 정의** → **exact-same 먼저**(정당 재시도 false-positive 0)·그 다음 데이터서 "유사-arg 루프"가 지배적이면 완화. 보수적 시작.
- **Q2 G-confirm 자가-ask?** → Phase P에 **loop-only vs loop+confirm-guidance 두 arm** 넣어 실측(task6=낙관적이나 측정으로).
- **Q3 P5 정책-적응(대안행동)** → **가장 깊은 잔여**. "반복 금지"(G-loop)는 유한하나 "**대안 찾기**"(re-plan)는 열린 search 스킬 = **P7 recovery의 어려운 절반**(§5.13: 게이트-상호작용은 닫히나 *어느 대안*인지는 search). **별 능력으로 플래그**(닫힘 주장서 분리).
- **Q4 프로토타입 천장 gap** → **§3 가드 2분류로 해소**: (a)남는 게이트=결정론 soundness·(b)학습타깃=추상 interaction. 천장=(a)+(b)·내재화는 (b)만·(a)는 scaffold 영구분담 = §5.12 정합(gap 아님).
- **Q5 검증기 false-positive** → §6 L1L2 auth FP(17→9·legit name/email 차단)는 **학습 보상으로 쓰기 *전* 선결**(오판=오학습). 고정밀 blocklist(스키마-example literal)·context-subtraction 정확도 선행.
- **Q6 순서** → **chain-census가 단계 귀속을 깨끗이 하면 G-fab+G-loop 양성만으로 Stage A(v9) 착수 가능**(full 양성 안 기다려도). 단 write-단계(P6/P7)는 G-confirm arm 양성 확인 후 Stage B.

## 8. 마일스톤
- **M0 ✅**: 3-arm provenance prototype → 상류 P2b 레버 확정·**write가 진짜 벽**(체인-census). 본 문서의 출발.
- **M1 (다음·프로토타입-우선)**: **G-loop 구현**(`t2_gate_patch` orchestrator·동일-실패 차단) → **full 가드레일 4-arm 프로토타입**(BASE/+fab/+loop/+confirm) → pass 이동 판정. **이게 본 학습 게이트.**
- **M2 (양성 시)**: Stage B 학습 — v8(P6) 완주 eval + **sop_confirm 재빌드(dim5 confirm-phrasing 다양화·§9.3)** + `fc_recovery_augment`(P7·dim5 error/deny phrasing randomize) 구현·학습.
- **M3**: Stage A 학습(v9) — **§9.4 재정의**: dim2(값·유지) + **dim3 DPO negative(스키마-example)+bad_words** + **dim4 format-randomize(신규·LODO 처방)**. ~~"값 더 randomize"~~ 아님.
- **M3.5 (선행·이론검증)**: ①**Olver inv-측 재측정**(비판1·사전등록) — trained adapter(v7)·O(d) 연속회전·중간층 pooling·깊은층 둔감을 *예측-후-측정* → inv_dim=n−s 수vs수 통과 여부 확정(현=consistent-with). ②**Olver dim별 coverage 진단(§9.5)** — dim1-5 전부 augment 후 잔여 표면-상관=0인가(dim6 탐지·완전성). ③ 둘 다 inv-측 판정(var=동어반복).
- **M3.6 (★절차판 clean experiment·§9.6)**: dim4(format)로 **without-등방화(단일 format)=약 vs with-등방화(format-uniform)=강** A/B = 산술-토큰화(Nogueira/McLeish)의 절차 인스턴스. 사전등록 예측: **format-uniform 단독이 LODO 전이실패를 뒤집음**(step-2 단독충분·절차>산술 구제용이 §5.15). 음성=step-2 외 잔여(dim6 or prior) 재진단.
- **M3.7 (★orbit-consistency·§9.7·probe 게이트後)**: invariant-probe(olver_definitive) 양성 시 → **soft orbit-consistency 손실**(`L_inv=‖h(x)−h(g·x)‖²`·의미보존 G·계획-수준 한정)을 v-final LoRA 학습에 추가 → 전이 eval. 전이 메커니즘 불변(동결+ABox-swap). 데이터-등방화(M3.6) 대비 *강제* 레버.
- **M4**: 통합 v-final + RLVR(E) + 전이 eval(chain-census 단계별).
- **M5**: 논문/특허 — 결정론 검증기 through-line(가드/라벨/보상)·2-stage 전이·**원리(유한-학습+offload+표면등방화+LLM강약기준 §5.15)** 헤드라인.
