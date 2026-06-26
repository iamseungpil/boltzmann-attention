# ORCHESTRATION-UNDER-LOAD = 능력 공개 기술 + plan/execute 분리 + 3-레버 통제 비교 (설계서)

> **상태**: 설계(리뷰#1 반영·2026-06-26). 진입=`06-NOW`·`HANDOFF_2026_06_26`·`MAKEORBREAK_VERDICT_2026_06_26`.
> **한 줄**: operand make-or-break NO-GO 이후 32B의 *진짜 미해결 잔여*=orchestration-under-load. 이를 단일 능력이 아닌 **5기능으로 공개 기술**하고, 각 기능을 **{LLM 짧은 번역 / 결정론 controller}**로 귀속한 뒤, 닫는 레버를 **{결정론-orchestration-scaffold / learn-rules}** 2분류로 **통제 비교**한다. 비용=gpt-4.1 0(로컬 우선).
>
> **★리뷰#1 반영 요지(2026-06-26)**: (1) **빌드 범위=Phase-0 `plan_probe.py` 단 1개로 시작**(나머지 phase는 결과가 요구할 때만·미리 짓지 말 것). (2) **유료 경로는 무료 probe 뒤로 게이팅**: H1은 plan_probe(무료)+이미측정된 atomic-isolation(GIVEN-SPEC 100%)으로 답 가능→end-to-end(유료/비싼 user-sim)는 결론 후 승인후 1회 확인만. (3) **plan_probe는 "plan-correct-in-isolation"을 잼**(planning이 블로커냐 실행이냐)·**(i)SELECT/(ii)GENERATE 구분은 출력만으론 불가→부차로 강등**. (4) **n=5는 rate 아님→C3 트리거=케이스 기반**("격리서도 plan-틀림 genuine 케이스 ≥1 생존?")·% 폐기. (5) 레버=깨끗한 query-vs-scaffold A/B 아님→**{baseline / 결정론-orchestration / learn} 3분류**(C1/C2=내부 진단증분). (6) **순환 주의**: 5-셋=orchestration-fail로 정의됨→"닫힘"이 "orchestration 일반 닫힘" 주장 못 함(일반화=후행).

---

## §0. 동기 (왜 지금 이 실험인가)

- **operand make-or-break 종결**(`MAKEORBREAK_VERDICT`): 32B operand 실행능력=GIVEN-SPEC 100%(88/88). faithful-formalize SFT=NO-GO. 잔여=criterion해석/⋈/user-sim 아티팩트로 *분류*되었으나, **"orchestration-under-load" 자체는 능력으로 기술·측정·레버비교된 적 없음.**
- **fail-all 포렌식(2026-06-26·`sim_results/cu_batch3_failall_8of10`)**: fail-all 10 task를 충실 Claude-user full-flow로 재구동·전수 궤적 분류:
  | task | 진짜 원인 | 분류 |
  |---|---|---|
  | t17, t92 | — | **flip**(user-sim 노이즈였음) |
  | t20 | 품목 하나씩 수정→non-pending 에러→망가진 toolcall | orchestration(배치 위반) |
  | t36 | 16388>16384 컨텍스트 초과 | orchestration(long-horizon·scaffold READ가 악화) |
  | t37 | 조건 분기 중 new≠old 위반 | orchestration(시퀀싱) |
  | t99 | 2주문 중 1개 누락 + `NEW_BIKE_ITEM_ID` 날조 | orchestration(multi-order+날조) |
  | t71 | ⋈·operand 정확, backpack 가용불가 변형 over-modify | criterion/present |
  | t105 | 액션 실행 전 턴 소진(MODEL=[]) | 측정 아티팩트(고정턴 경직) |
  | t109(관측) | 20분+ degenerate 생성 루프(GPU 99%·결국 context 초과 추정) | orchestration(루프) |
- **결론**: clean flip 2/10(천장 하한 ≈0.402+2/112≈0.420), 잔여 **지배 원인=orchestration-under-load**(t20·t36·t37·t99·t109 = ≥5/10). operand 아님. **이게 scale이 사는 능력의 직접 증거**(frontier는 native로 버팀)이고, **현 scaffold(present/calc/gate)는 이걸 offload 안 함**(t36은 calc/present READ가 컨텍스트를 부풀려 *오히려 악화*).
- **미해결 핵심**: 이 orchestration을 (a)학습이냐 (b)질의방식이냐 (c)scaffold냐 — **make-or-break는 operand-formalize만 닫았지 이걸 안 닫음.** 따라서 별도 통제 비교 필요(사용자 directive).

---

## §1. 능력 공개 기술 — orchestration-under-load = 5기능 분해

"orchestration"은 단일 능력이 아니다. 궤적이 드러낸 **5개 독립 기능**으로 분해하고 각각을 귀속한다:

| # | 기능 | 정의 | 실패 예 | 본질 | 1차 귀속 |
|---|---|---|---|---|---|
| **A** | **decomposition / plan 생성** | open NL 목표 → 실행가능 서브스텝 시퀀스(=plan-spec) | t20 "전 품목 업그레이드"를 스텝화 못 함 | NL→plan 번역(**boundary translation 한 층 위**) | **LLM** (짧은 번역) |
| **B** | **state-tracking** | 어느 주문·뭘 이미 했나·누적합·진행 위치 | t99 2주문 중 1개 누락 | 순수 부기 | **결정론 controller** |
| **C** | **sequencing/batching 제약** | "수정은 한 호출로(이후 non-pending)" 등 순서·집합 규칙 | t20 하나씩→에러 | 도메인 규칙(decidable) | **gate/controller**(A2 spec) |
| **D** | **conditional 실행** | "안되면 X, 그래도 안되면 Y" 의사결정 트리 | t37 분기 붕괴 | 트리 *실행*=결정론; 트리 *형식화*=LLM | **controller**(실행) + LLM(형식화) |
| **E** | **fabrication-avoidance** | 값 없으면 lookup/ASK, placeholder 금지 | t99 `NEW_BIKE_ITEM_ID` 날조 | provenance | **gate** ([[43]] epistemic-A2) |

### 핵심 원칙 — **분해(LLM) vs 실행(controller) 분리**
- LLM은 **분해를 하되 실행과 인터리브하지 않는다**: 전용 짧은 호출 1회로 plan-spec를 내고 끝. 부하 실패(t20·t99·t109)는 모델이 *plan+execute를 동시에* 하다 무너진 것.
- controller는 plan-spec를 **결정론으로 walk**(B 루프·state, C 배치강제, D 트리실행, E provenance), 각 atomic leaf에서만 LLM 짧은 번역 재호출.
- = boundary-translator를 한 층 위로: `NL→gate_spec` 번역과 동형인 **`NL→plan-spec` 번역** + 결정론 인터프리터.

### A의 갈림 — plan이 SELECT냐 GENERATE냐 (측정 가능한 미지)
| | plan의 성격 | 난이도 | 레버 |
|---|---|---|---|
| **(i) SELECT+PARAM** | plan = *authored 절차 라이브러리*(A2 control-flow)에서 어느 절차+파라미터 | 짧은 번역 = **GIVEN-SPEC 100% 계열·해결 추정** | scaffold(절차 A2 author·[[05]]/[[11]]) |
| **(ii) GENERATE** | plan = 알려진 템플릿 없는 novel 구조 생성 | 진짜 planning = **미지** | learn 후보(SOPBench control-flow→A2-swap·[[02]] P-primitives) |

tau2는 트랜잭션 closure 유한([[02]])→**대부분 (i) 추정**. 그러나 **미검증.** §3 Phase 0 probe가 (i):(ii) 비율을 숫자로 답한다 — 이게 본 설계의 *crux 측정*.

---

## §2. 가설 (반증가능 형태)

- **H1 (load-cause)**: t20·t99·t109 실패는 능력부족이 아니라 *plan+execute 동시수행 부하*다. → **plan/execute를 분리하면**(C1) atomic 정확도가 GIVEN-SPEC(100%) 수준으로 회복된다. *반증*: 분리해도 atomic leaf가 틀림.
- **H2 (select-not-generate)**: 잔여 plan은 대부분 authored 절차 SELECT(i)다. → **plan-spec 단독 채점이 높다(≥~85%)**. *반증*: plan-spec 채점이 낮음 → (ii) GENERATE가 실재 → learn 후보.
- **H3 (controller-general)**: B/C/D-실행/E를 닫는 controller는 도메인-일반으로 author 가능(제어흐름 IR + A2 spec). *반증*: retail 하드코딩 없이 못 짬([[05]] 위반=설계 실패).
- **H4 (lever-order)**: 흡수우선순위([[13]] scale→learn→scaffold)의 control 버전 = **query-method/scaffold(결정론 control) 먼저, learn 최후.** C1·C2가 잔여를 닫으면 C3(SFT) NO-GO.

---

## §3. 측정 도구 + task셋

### 도구 (대부분 재사용·gpt-4.1 0)
- **`plan_probe.py` (신규·핵심·`operand_controlled.py` 패턴 재사용)**: task의 open NL 목표(instruction)만 주고 **plan-spec만** 산출(실행 0). plan-spec=정규화된 (action, args-skeleton) 시퀀스. gold `evaluation_criteria.actions`와 대조 채점: {정확 / 순서만틀림 / 누락 / 과잉 / 잘못된action}. → H2의 (i):(ii) 분해. **단독 격리=load 0.**
- **`atomic_exec_probe.py` (신규·`operand_controlled.py` GIVEN-SPEC 확장)**: gold plan을 주고 각 atomic leaf만 단문으로 질의→정확도. H1의 leaf-회복 측정.
- **`claude_user_batch.py`/`gen_turns.py` (기존)**: full-flow robust 회수(C0/C1 end-to-end). **단 고정턴 경직 confound** → 복잡 task는 적응형 turn 필요(아래 리스크).
- **`escape_det_census.py --clean` (기존)**: 결정론 census·레버 발화율 전수확인([[30]] 천장주장 전 필수).

### task셋
- **orchestration-잔여 셋(1차)**: t20, t36, t37, t99, t109 (+ 06-NOW의 multi-item/multi-order family: t36/37/38, t111). fail-all∧포렌식=orchestration.
- **대조군**: t17·t92(flip=단순)·t71(criterion non-orchestration) → 레버가 orchestration에만 듣고 단순엔 무해한지(over-action 유발 안 하는지) 확인.
- 전이검증(후행): airline/bank A2-swap 동일셋(스택 도메인-일반성).

---

## §4. 실험 조건 — 3-레버 통제 비교

같은 task셋에 레버만 바꿔 lift 측정:

| 조건 | 레버 | 무엇을 바꾸나 | 빌드 | 입증/반증 |
|---|---|---|---|---|
| **C0 baseline** | — | 현재(full 컨텍스트·모델이 plan+execute 인라인) | 0(있음=0.402) | 기준선 |
| **C1 query-method** | 질의 방식 | LLM=plan-spec 1회 + atomic leaf 짧은 재질의. orchestrator가 **compact state만** 주입(raw transcript 아님)·결과 결정론 조합. **재학습 0** | 중(plan-execute 하네스) | H1: 회복하면 load-cause 확정·learn 불요 |
| **C2 scaffold-controller** | scaffold | C1 + B/C/D-실행/E를 결정론 controller로(batching 강제·multi-order loop·conditional-tree executor·provenance gate). 제어흐름 IR + A2 spec | 중상(controller 엔진) | H3: 도메인-일반으로 닫히나 |
| **C3 learn-rules** | 학습 | plan-generation(A-ii) SFT — **SOPBench control-flow서 학습→tau2 A2-swap**([[11]] tau2 학습0). `build_tbox_planner_sft.py` 재사용 | 상(GPU·[[13]] 최후) | H2: C1/C2 후 *plan-generate* 잔여 실재 시만 GO |

**순서 = Phase 0(plan-probe) → C1 → C2 → C3(조건부).** C1·C2는 다수 중복(분해=scaffold기능)이라 실제론 "input 단문화(C1) + state/실행/조합(C2)"가 한 묶음으로 진화 가능. C3는 Phase 0/C1/C2가 *plan-generate 잔여*를 남길 때만.

---

## §5. 지표 (엄수)

- **robust pass^all / compliant-pass(F4)** 만. **pass^1 점추정 금지**([[06]] user-sim 노이즈 ~0.11).
- **plan-spec 정확도**(Phase 0): 정확/순서/누락/과잉/오action 5분류 비율 → (i):(ii).
- **atomic-leaf 정확도**(C1): leaf별 정답률(GIVEN-SPEC 회복 검증).
- **레버별 lift**: ΔpassAll(C1−C0, C2−C1, C3−C2). 대조군 무해성(over-action 증가 0).
- **레버 발화율 전수확인**([[30]]): plan 호출/controller 분기 실발화 검증(단위테스트≠라이브·calc 31/342 교훈).
- **per-task 이중확증**: robust-fail→robust-pass ∧ write gold-correct.

---

## §6. 단계 (구현 순서·각 단계가 다음의 게이트)

- **Phase 0 — plan-probe (가장 싼·먼저·~30분·gpt-4.1 0)**: orchestration-셋 + 대조군에 `plan_probe.py`. 산출=plan-spec 정확도 → **(i):(ii) 비율.**
  - (i) 지배 → 분해는 SELECT=해결됨 → 실패원=실행부하 → **C1/C2가 고침·C3 불요 가설 강화.**
  - (ii) 비무시 → plan-GENERATE 잔여 실재 → **C3(SOPBench learn) 후보 점화.**
- **Phase 1 — C1 plan-execute 하네스**: plan/execute 분리 + compact-state 주입. 회수=robust + atomic-leaf 정확도. H1 판정.
- **Phase 2 — C2 controller 엔진**: B/C/D/E 결정론화(도메인-일반 IR + A2 spec). [[05]] 게이트(retail 필드0) 단위테스트 강제. H3 판정.
- **Phase 3 — C3 (조건부·GPU)**: Phase 0/1/2 후 *plan-generate* 잔여 실재 시만. SOPBench SFT→A2-swap. [[09]] full-run 승인+1회.

---

## §7. 비용규율 ([[09]]·엄수·★리뷰#1 교정)

- **user-sim 무료/유료 정명**(혼동의 뿌리였음):
  | 모드 | 비용 | 정체 |
  |---|---|---|
  | **scripted-replay** (`claude_user_batch.py`·고정턴 JSON) | **무료** | 로컬 32B 전용·user 턴=정적 문자열·Claude/gpt-4.1 API 0 |
  | live user-sim — gpt-4.1 | 유료(최저) | OpenRouter 예산 |
  | live user-sim — Claude API | 유료(**최악 15-30x**·[[30]] COST GUARD) | 절대 기본 아님 |
  - 즉 지금까지 돌린 배치=이미 무료. "Claude-user-sim"이 {무료 스크립트 / 유료 live}를 한 이름으로 덮어 위험했음→분리 명명.
- **유료 게이팅(엄수)**: Phase-0 `plan_probe`=로컬 32B 무료. atomic-isolation=이미 측정(GIVEN-SPEC 100%). → **H1/H2 무료로 답** → live user-sim end-to-end는 **결론 선 뒤·승인후·1회 확인만**(탐색목적 금지).
- **결과 즉시 영속화**([[30]]): gzip→`reports/facet_rft_2026/sim_results/`→`git add -f`+push. 재런=distinct tag.
- **full-run 전 SMOKE**(레버 실발화·크래시0). 천장주장 전 발화율 전수.

---

## §8. 제약 정합 (행동 전 점검)

- **[[05]] A2만 도메인특화**: controller(C2)=제어흐름 IR=도메인-일반. 절차/제약=A2 spec(retail 인스턴스). 엔진 retail 필드0=단위테스트 강제. plan-spec 포맷=일반. **게이트 증식 금지**(척추=일반).
- **[[11]] tau2 학습 0**: C3는 SOPBench서 학습·전이=A2-swap. tau2 fit 금지.
- **[[13]] 흡수우선순위(control판)**: query-method/scaffold(결정론 control) 먼저·learn 최후.
- **[[02]] generator-algebra**: orchestration=control-flow 축=P-primitives. C3 학습원=SOPBench(control-flow bench). 재발명 금지.
- **[[03]]/[[08]]**: 예측으로 갈아엎기 금지(probe=adjudicator)·집계→결론 직행 금지·전수포렌식.

---

## §9. 판정 매트릭스 (결과→결론)

| Phase 0 (i):(ii) | C1 회복 | C2 추가 | 결론 |
|---|---|---|---|
| (i) 지배 | 회복 | — | **orchestration=query-method/scaffold로 닫힘·learn NO-GO.** 헤드라인=결정론 controller+base translator. thesis §2 강화. |
| (i) 지배 | 부분 | 닫음 | controller가 B/C/D/E 흡수=결정론 승리. learn NO-GO. |
| (ii) 비무시 | 회복(실행만) | — | 실행=결정론, **plan-generate=learn GO**(SOPBench→A2-swap). thesis 두 날개 carry. |
| (i) 지배 | 미회복 | 미회복 | capability-bound(부하무관)→scale 필요. learn/scaffold 둘 다 한계. (예상밖) |

→ **어느 경로든 "orchestration을 학습할지"가 숫자로 종결.** make-or-break(operand)와 함께 thesis의 learn-wing 운명을 tau2서 확정.

---

## §10. 리스크 / 함정

1. **고정턴 경직 confound(★)**: 사전스크립트 턴은 복잡 조건부 task엔 충실 user-sim 아님(t105 MODEL=[]). C1/C0 end-to-end 비교 시 **적응형 Claude-user-sim**(turn을 라이브 생성) 필요 — 그래야 "분리효과 vs 턴경직"이 안 섞임. Phase 0 plan-probe는 turn 무관(격리)이라 이 함정 면역=먼저 하는 또 다른 이유.
2. **plan-spec 포맷 누설**: plan-probe가 gold action 스키마를 과하게 주면 SELECT를 인위적으로 쉽게 만듦([[05]] make-or-break rig 전례=ID-given 아티팩트). instruction만 주고 도구목록은 일반 제공·gold 누설 0.
3. **C2 controller의 [[05]] 위반 유혹**: "non-pending이면 배치" 같은 규칙을 엔진에 하드코딩하면 retail 누설. 반드시 A2 spec(gate_spec류)으로·엔진은 일반 인터프리터.
4. **C3 재유도 위험([[20]] settled 음성)**: plan-generate SFT가 C4/M-σ 계열($ref copy) 되면 전이음성. SOPBench control-flow(구조 학습)≠operand-$ref. prep §5 게이트(잔여 실재·non-C4·probe격리) 통과 시만.
5. **degenerate 생성 루프**(t109): 일부 task는 모델이 무한 생성→context 초과. C1 plan-execute가 이걸 구조적으로 차단(짧은 호출)하는지도 측정 대상.

---

## §11. 자산

- **재사용**: `operand_controlled.py`(probe 패턴)·`claude_user_batch.py`·`gen_turns.py`·`escape_det_census.py`·`gate_interpreter.py`·`t2_gate_patch.py`·`a2/retail.gate.json`·`build_tbox_planner_sft.py`(C3)·SOPBench 클론(`/home/woori/scratch/SOPBench`).
- **신규**: `plan_probe.py`(Phase 0)·`atomic_exec_probe.py`(C1 leaf)·`plan_execute_orch.py`(C1 하네스)·`controller_engine.py`(C2·제어흐름 IR 인터프리터)·A2 절차/제약 spec 확장.
- **정본 doc**: 본 설계서. 결과=`sim_results/`. CAPABILITY_LEVER 매트릭스에 **새 능력행 C(orchestration-under-load)** + 3레버 칼럼 추가(별도 정렬편집).

---

## §12. 리뷰#1 합의 (확정)

1. **범위**: 1차=orchestration-잔여 5+family+대조군(case-level·무료). retail 전task 확대=레버 유망+rate 필요 시 승인후 후행.
2. **빌드 순서**: C1/C2 하네스·adaptive user-sim **미리 짓지 말 것.** **Phase-0 `plan_probe` 1개만** 먼저→그 결과가 C1/C2 필요여부 결정.
3. **adaptive user-sim**: 이번에 안 만듦(유료 end-to-end 전제). Phase-0=turn-confound 면역·무료→1차 판정 거기서.
4. **C3 트리거**: **%가 아니라 케이스 기반** — C1/C2 후에도 "격리서 plan-틀림"인 genuine 케이스 ≥1 생존 시만 별도 GO. % 폐기([[06]] n=5).
