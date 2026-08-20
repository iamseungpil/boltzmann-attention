# 레버 정본 명부 + 통합안 (2026-08-19)

> 사용자 지시: *"레버들을 단순화하고 효과를 명확하게 측정해서 장단점을 찾아야 한다. 지금 있는 레버들을 모두 다시 점검하고 유효한 레버들을 확정하라."*
>
> **유효의 정의 = reward 하나뿐**([[69]] §0). gold 일치율·action_match·행동 지표(호출했나·발화했나)는 성적이 아니다.
> `VALID` = 같은 태스크·같은 시드의 짝 A/B 에서 reward 이득 방향 + 부정통제([[57]]) + 단일 변수 + 차가 잡음 바닥(팔당 ±4/20·C483) 초과.
>
> **단순화 = 통합이지 끄기가 아니다**([[60]]). 이 문서에 "끄자"는 권고는 없다.

선행 산출물(재조사 금지·인용만): `OPERAND_LEVER_AUDIT_2026_08_19.md` · `OPERAND_LEVER_ISOLATION_PLAN_2026_08_19.md` · `LEVER_CONSOLIDATION_2026_08_19.md` · `CALC_LEVER_PASS_PROVENANCE_2026_08_19.md` · `OPERAND_FORMALIZE_RETROSPECTIVE_2026_08_19.md`

---

## §1 요약표

### 1-1 판정별 개수 (이번 배치 155건)

| 판정 | 개수 | 비율 | 뜻 |
|---|---:|---:|---|
| **VALID** | **0** | 0.0% | reward 짝 A/B + 부정통제 + 단일변수 + 잡음 바닥 초과 |
| **HARMFUL** | **22** | 14.2% | 손해 방향이거나, 정답을 지우거나 잘못 거부한 실측 사례 존재 |
| **DARK** | **38** | 24.5% | 두 신호(stderr 태그 · 궤적 효과 문자열) 모두 0 = 발화한 적 없음 |
| **UNMEASURED** | **82** | 52.9% | 주장은 있으나 조건을 갖춘 대조가 없음 |
| **NOT_A_LEVER** | **13** | 8.4% | 인쇄 문자열 · 파라미터 · 계기 · 경로 상수 |
| 합계 | 155 | 100% | |

### 1-2 기능군 × 판정

| 군 | 플래그 | VALID | HARMFUL | DARK | UNMEASURED | NOT_A_LEVER |
|---|---:|---:|---:|---:|---:|---:|
| L1 쓰기-근거(인자 접지) | 27 | 0 | 7 | 10 | 10 | 0 |
| L2 이름 원장 | 15 | 0 | 0 | 2 | 13 | 0 |
| L3 서술 출처 | 18 | 0 | 4 | 8 | 6 | 0 |
| L4 회수·집합 차 | 9 | 0 | 0 | 4 | 5 | 0 |
| **L5 선행 강제(요건 그래프)** | **18** | — | — | — | — | — |
| **L6 완결·후속(사임 시점)** | **12** | — | — | — | — | — |
| L7 말-행동 등가 | 10 | 0 | 0 | 0 | 10 | 0 |
| L8 부재 종결 | 9 | 0 | 0 | 2 | 7 | 0 |
| L9 국면 배치 | 15 | 0 | 2 | 3 | 9 | 1 |
| L10 형식화·되묻기 | 13 | 0 | 5 | 2 | 4 | 2 |
| L11 계산 이관·원장 대조 | 14 | 0 | 3 | 3 | 8 | 0 |
| L12 재료 배달·부하 축소 | 5 | 0 | 0 | 1 | 4 | 0 |
| L13 출구 거버넌스 | 6 | 0 | 0 | 1 | 5 | 0 |
| L14 형식·채널 | 14 | 0 | 1 | 2 | 1 | 10 |
| **판정 완료 소계** | **155** | **0** | **22** | **38** | **82** | **13** |
| L5+L6 미판정 | 30 | — | — | — | — | — |
| **14군 총계** | **185** | | | | | |

**L5·L6 (30종) 는 이번 배치의 판정 대상이 아니었다.** 기능군 분류만 확정돼 있고 판정은 미착수다 — §5·§6 에 그 사실을 그대로 남긴다.

### 1-3 실효 레버 분모

| 층 | 개수 | 근거 |
|---|---:|---|
| 코드가 실제로 읽는 `T2_*` 전수 | **331** | 정본 스캐너 `scan_flags.py` (`os.environ.get`/`os.getenv`/`os.environ[]`) |
| − 프로브/테스트 전용 | −61 | 엔진이 한 번도 안 읽음 (`x*.py`·`test_*.py` 만) |
| = 런타임 | 270 | |
| − 레버 아님(런타임) | −85 | 인쇄 문자열 10 · 파라미터 61 · 계기 12(중복 1 조정) · 경로 상수 2 · arm 노브 10 |
| = **14군 레버 명부** | **185** | |
| − 본 배치에서 NOT_A_LEVER 로 재분류 | −13 | `T2_DECLFIRST_ENFORCE`·`T2_DISAMB_MODE`·`T2_L4_MODE` + L14 형식·채널 10종 |
| = **★실효 레버 분모** | **172** | 그중 **판정 완료 142** · **미판정 30(L5·L6)** |

**분모 부풀림의 원인 2종을 못박는다.**
1. 인쇄 문자열이 레버로 이중계수됐다 — `T2_ARGSCHEMA`·`T2_WRITEPROV`·`T2_SG_ARGS`(태그 문자열) · `T2_ARG_REPEAT`·`T2_WRITE_DEDUP`(구현 부재) · 신규 확인 `T2_COMPUTE`·`T2_OVERFLOW_GUARD`·`T2_PROC_ABSENT_CAP`·`T2_DD_FB`·`T2_FOLLOWUP_CAP`.
2. **`T2_COMPUTE` 는 존재하지 않는 이름이다** — `grep environ.*T2_COMPUTE` 0건. 실제 게이트는 `T2_CALC`(t2_gate_patch.py:678). 원장·감사가 `T2_COMPUTE` 로 인용해 온 판정은 **내용은 유효, 이름만 `T2_CALC` 로 정정**해야 한다.

### 1-4 한 줄 요약

> **185 레버 중 유효(VALID) 0건.** 인용 가능한 실측은 HARMFUL 22 · DARK 38 뿐이고, **나머지 82(+미판정 30) 가 한 번도 reward 로 재어지지 않았다.**
> 발화량 1위 `T2_DISCOVERY_STEP2`(2,216회)·2위 `T2_SG_ISOLATE`(7,849회)·최다 `T2_RESOLVE`(13,766회) 가 전부 대조 팔 0이다. **가장 많이 말하는 레버들이 가장 안 재어졌다.**

---

## §2 ★VALID 목록

### **0건.**

정직하게 쓴다 — 155 판정 중 네 조건(① reward 종점 짝 A/B ② 부정통제 ③ 단일변수 ④ 이득 방향이면서 잡음 바닥 초과)을 **동시에** 만족한 레버는 **하나도 없다**.

### 2-1 가장 근접한 것들 (조건 3/4 이상 충족)

| 레버 | 런 태그 (A ↔ B) | reward | 단일변수 | 부정통제 | 잡음 바닥 초과 | 빠진 조건 |
|---|---|---|:---:|:---:|:---:|---|
| **T2_ACT_DEMAND** | `bank_t7297_{ctl,treat}_20260815q` | **8/20 ↔ 9/20** | ✔ (`$DEMAND` 하나) | ✔ 강함 (098 5/5↔5/5 불변) | ✘ (차 1 < 5) | **이득 없음 + over-action 2→8 (차 6 ≥ 5)** ⇒ HARMFUL |
| **T2_HANDOFF_PREDICATE** | `bank_t7308_{ctl,treat}_20260818c` | **2/12 ↔ 2/12** | ✔ (`$HP` 하나·26 플래그 PIN 동일) | ✔ 이중 (ctl 발화 0 + x370 `D_NEG`) | ✘ (차 0) | **null + 지연 1.90× · CWE 13↔0** ⇒ HARMFUL |
| **T2_ACTION_SUB** | `bank_asubON_20260810` ↔ `bank_isoOFF_20260810` | **9/12 ↔ 7/12** | ✔ 플래그 기준 (단 **다른 런처·다른 GPU·순차**) | ✔ P5 Δspurious 양팔 0 | ✘ (차 2 < ~3) | 동시 실행 짝 아님 · n 부족. **이 배치 최근접** |
| **T2_DECISION_ISOLATE** | `bank_isoON_20260810` ↔ `bank_isoOFF_20260810` | **6/12 ↔ 7/12** | ✔ | ✔ 부분 (010 회귀 감시칸·팔 오염 0) | ✘ | 손해 방향 + **배제 근거 삭제 실측** ⇒ HARMFUL |
| **T2_UNLOCK_QUIET** | `bank_uq_20260811`(R8c) ↔ asubON ↔ isoOFF | **3/6 ↔ 3/6 ↔ 3/6** | ✔ (R8c↔asubON) | ✔ 동일조건 팔 2개 | ✘ (차 0) | 결과 파일 미영속(원장 인용) |
| **T2_DISCOVERY_REQUIRED** | `bank_ctl2_nt20_20260718` ↔ `bank_dreq2_nt20_20260718` | **9/20 ↔ 11/20** | ✔ (ctl 발화 0 검산) | ✔ 배선 오염 0 | ✘ (p≈0.34) | **삭제 편향**(`run_with_retry` 가 실패 궤적 버림) = Δ와 같은 방향 상향 |
| **T2_DELIVER_PRECOMMIT** | `bank_t7303_{ctl,treat}_20260816h` | **5/12 ↔ 6/12** | ✔ (`$PRE`) | ✔ 098 3/4↔3/4 | ✘ | 1차 종점이 처치 배정의 재인쇄(C502⒝) · CWE 5↔0 · 지연 1.38× |
| **T2_MATERIAL_RESERVE** | `bank_t7299_{ctl,treat}_20260816b` | **3/8 ↔ 4/8** | ✔ (`$RESERVE`) | ✔ 098 | ✘ | **처치가 무동작**(억제할 2회째 배달이 0건) |
| **T2_PROV_MODE** | `compabl_noP_eplan` ↔ `..._rescue` | **29/50 ↔ 31/50** | ✔ (`$PROVM` 하나) | ✘ | ✘ (+4pp) | 부정통제 없음 |
| **T2_PRESENT_NESTED** | `comp_retail_t4` ↔ `compabl_noP`(공유 25 태스크) | **72/100 ↔ 26/50 = +20pp** | 설계상 ✔ (COMP 런처 부재) | ✘ | ✔ (여유 2배) | **부정통제 없음 · nt 4↔2 불일치 · 비동시 실행** ⇒ 최대 미청구 효과 |
| **T2_RETRY_CONTROLLER** | `c8_gate` ↔ `c8_gate_retry` | 소형 풀링 **127 ↔ 84 · p=0.0037** | ✔ (플래그 1개) | ✔ 대조군 발화 0 | ✔ (유일하게 유의) | **부호가 손해** ⇒ HARMFUL. 32B 는 null(p=0.656) |
| **T2_KEEP_DENY_BODY** | `bank_uq_20260811` ↔ `bank_kb_20260811` | **3/6 ↔ 2/6** | ✔ (런처가 발사 거부로 강제) | 격리에만 (x246 `C_NONE` 0/8) | ✘ | 접힘 노출이 11↔3 으로 팔 간 불일치 |

**읽는 법.** 이 표에서 조건 ④(잡음 바닥 초과)를 넘긴 것은 `T2_RETRY_CONTROLLER`(손해 방향)와 `T2_PRESENT_NESTED`(부정통제·nt 결손) **둘뿐**이다. 즉 **유효 레버를 못 찾은 것이 아니라, 유효를 판정할 실험을 아직 안 돌린 것이다.**

---

## §3 HARMFUL (22종) — 무엇을 어떻게 깼는가

### 3-1 정답을 직접 지운 것 (가장 강한 부류)

| 레버 | 실측 축자 | 종점 피해 |
|---|---|---|
| **T2_OPERATOR_PINPOINT** | 격리 `A_REF` **24/24 → `B_PINPOINT` 0/24**. 부정통제 `E_NEG`(가짜 도구명 지목) **0/24** = *존재하지 않는 이름에도 순응*. 라이브 t7292 073 t0: 이미 성공한 write 에 지목 5회 → 재-unlock·재호출 → **같은 계좌에 9.50 을 두 번**(5200→5209.50→5219.00) | RESEARCH_MASTER:420 축자 *"못 해서 실패한 게 아니라 **한 번 더 해서** 실패했다"* — reward 0 |
| **T2_GROUND** | 치환 385건 중 **301건(78%)이 `arg=agent_tool_name` → 고객 이름**. 축자 `substituted arg=agent_tool_name val=cancel_debit_card_7823 -> LIANG JINHAI`. task_077 은 `Unknown agent tool 'LIANG JINHAI'` **79회 반복** | 궤적 생존 110건/11 sim = **11/11 전부 reward 0.0** |
| **T2_DISAMB** | C61 축자 *"+27 살리고 **−37 부숨**"* — **write-소멸 39건**(DISAMB 분기가 tool_calls 없는 재확인 응답을 무조건 수락). t46 **4/4 → 0/4** | pass^1 260/456 ↔ 263/456(−0.7pp·CI[−6.14,+4.61]) |
| **T2_RESOLVE** | 라이브 단일변수 `bankar_rec_g5` **3/5 ↔ `bankar_rec_gr5` 0/5**(양의 칸 0·음 3). gold 오차단 **24**(operator-fab 12 + operator-scope 12). 독립 확인: `task_050` 5자리 거부 대상이 전부 gold 요구 이름 | Fisher p≈0.167 (방향 균일·수치 약함) |
| **T2_L4** | **엔진이 스스로 자기 기능을 껐다** — t2_gate_patch.py:9750 축자 *"치환 성적 2/2 오답(t58 정답파괴 · t20 제약절단) → `T2_L4_MODE` 기본 \"keep\"(관측·audit only·치환 없음)"* | 양의 칸 0 |
| **T2_REF_ISO** | 축자 `[T2_REF_ISO] switched param=transaction_id txn_adea68821a1d->txn_9a72b84326d1` — **앞이 gold**, 뒤는 손님이 언급조차 안 한 값. 3회 메모이즈 | 그 sim `action_match False` · reward 0.0 |
| **T2_WRITE_EVIDENCE** | t7326 deny 25 중 **gold 이름 표적 22 · 미회복 5**. 축자 *"094 t1 turn71 의 gold write 를 차단·미회복시킨 것 [S]"*. task_040 t0 turn 39/43/45/47 동일 이름 4연타 후 영영 미실행 | 미회복 5건 **전부 reward 0.0**. 런 전체 gold 오차단 56 중 22 = **39% 단일 최대** |
| **T2_SG_GROUND** | task_046/t0 `outstanding_balance=0.00`(**참인 값**) 드롭 → 11턴 루프 → 잔여 gold 소실 | 조건부 — t7326 드롭은 거짓 드롭 0 |
| **T2_SG_BYREF** | 70런 census: byref 시도 380 → deny 314 중 **거짓 deny 49건/11 sim**(옳은 참조를 잘못 거부). 원인 = `_resolve_ref_output` 이 래퍼의 `agent_tool_name` 미색인 | 직접 반례 존재 — t7297 ctl 073 은 거짓 deny 5회를 맞고도 **reward 1.0**(자력 복구) |
| **T2_TOOL_SIGNATURE** | 정본 계기 `x392_block_join.py` t7326: `gold✓ 4 · 이후실행 3/4`. 미회복 1건 축자 — `task_017 tr0 turn 53 · submit_cash_back_dispute_0589 · reward 0.0` | 공동 귀속(같은 표적을 RESOLVE·PROCEDURE 도 막음). C267 축자 *"이득이 실증된 적 없고 … 승격 금지"* |

### 3-2 [[25]] 위반 — 우리 층이 근거원을 오염시킨 것

| 레버 | 실측 축자 |
|---|---|
| **T2_MATCH_COUNT** | t7326 '전부 표시' 102건 중 **19건이 반증 가능하게 거짓**(주장 31·23·23·20·20·15·14 vs 실제 반환 10). 원인 = `shown_in()`(t2_match_count.py:88-91)의 `^\s*(\d+)\.\s` 가 회수 문서 **본문 안 번호 목록**까지 셈(task_055 실제 10 ↔ 보고 34). **계기 결함이 곧 모델 입력 결함** |
| **T2_CLAIM_PROV** | C341 축자 — `log_verification` 이 **양쪽 sim 에서 실행됐는데** turn 28/30 에 *"the conversation ledger shows NO such event"* 가 나갔다. *"우리 출력은 그 대화의 **유일한 근거원**이라 거짓은 그 자체로 오염이고, 모델이 고칠 수도 없다"* |
| **T2_VALUE_ACQUIRE** | `gate.json:2386` 이 사실 아닌 전제를 단정 — *"they are NOT stored in the account records … file the dispute with that value"*. 전달된 넛지 **80/80 이 표적 밖 태스크**, 그중 20건 접촉 **20/20 reward 0.0** |
| **T2_ARG_AXIS** | 축자 `[T2_ARG_AXIS] deny got=checking want=['business_checking','savings']` — **055 gold 는 `Purple/checking` + `Silver Plus/savings`** ⇒ gold 축 거부 9건/15. want 집합 **자기모순 11/26 = 42%** |

### 3-3 pass 를 사되 **측정 대상 자체를 지운 것** ([[62]] 정면)

| 레버 | 실측 축자 |
|---|---|
| **T2_PRESENT_READS** | 4셀 요인 실측: pass^1 floor 0.547 → present **0.576**(+2.9pp). 그러나 **order 조회/sim 2.62 → 1.29**(present+g15 **0.48** = 5.5× 억제) · **미조회 날조 5.6% → 6.9%**(present+g15 10.4% = 1.9× 증가). 축자 *"present 는 **frontier 격차의 83%를 차지하는 실패(C29)를 스스로 제조한다**"*. C34 [M]: 엔진이 대신 `detail_producer` 호출 = **규칙 0 위반** |
| **T2_AUTOFETCH** | `no_gather` **36 → 24** · `schema_copy` **36 → 23**. 축자 *"**autofetch 도 present 도 gather 를 가르치지 않는다. 더 가까운 복사 대상을 문맥에 놓아줄 뿐이다** … 두 꼼수가 켜진 채로는 **gather 를 측정조차 할 수 없다**"*. reward 는 23/112 ↔ 25/109 = 잡음 |
| **T2_PROV_REGEN** | 456 sim 짝 McNemar: 이득 83 · 손해 74 · **p≈0.52 = 합 null**. 그러나 **pass^4 0.333 → 0.281(−5.3pp)** · robust 4/4 상실 16 vs 획득 10. ★t17 0/4→4/4 · ★t61 **4/4→0/4**(전 trial 파손). *p1 을 사고 p4 를 판다* |

### 3-4 과-행동을 산 것

| 레버 | 실측 |
|---|---|
| **T2_ACT_DEMAND** | over-action(`ONLY-PRED`) **2 → 8**(차 6 ≥ 잡음 바닥 5). 새로 생긴 것이 `050 ONLY-PRED:user_discoverable_tools 4건`(ctl 0)이고 **그 050 이 pass 1→0** = 시키지 않은 도구 지급. C492⒡ 축자 *"라이브에선 성적을 못 사고 **과행동만 샀다**"* |
| **T2_SOURCE_QUALIFY** | t2_source.py:299-303 축자 — *"102 는 직전 구성에서 db_match 2/2 였는데 이 갈래를 켠 arm 에서 **0/2**로 떨어졌고 제출도 1·1 → **5·3**으로 늘었다"*. 단 소거법 귀속 · 런 태그 미기록 |
| **T2_HANDOFF_PREDICATE** | reward null 인데 **CWE 13건 ↔ ctl 0**(2 sim 이 context_window_exceeded 종료) · **지연 1.90×**(67,949s → 129,179s) · msg/sim 87.8→97.4 · 절차 종점 **반전**(`formalized_target` ctl 14 → treat 4) |
| **T2_DECISION_ISOLATE** | C403⒟ 축자 — ON 의 100 t2 실패는 `World Blue` 제출인데 *"우리가 억제한 문장이 바로 `not reachable yet — World Blue needs 90` 이다 ⇒ 억제가 **경쟁 이름만이 아니라 배제 근거까지** 뺐다"* |
| **T2_RETRY_CONTROLLER** | 소형(7B+14B) 4 arm-pair 풀링 **127 vs 84 · p=0.0037**(4/4 전부 음). ★scale 별 부호가 갈린다 — 32B **null**(p=0.656). 합으로 뭉치면 신호가 지워진다 |

---

## §4 DARK (38종) — 발화 0 · `T2_FN_ISOLATE` 형 사고 후보

두 신호(stderr 태그 · 궤적 효과 문자열) **모두 0**. 원인이 셋으로 갈린다.

### 4-1 (D-a) 런처가 한 번도 안 켰다 — 배선 사고 후보 12종

| 레버 | 축자 근거 |
|---|---|
| `T2_ASK_UNKNOWN_BOOL` | `T2_ASK_UNKNOWN_BOOL=` 설정 .sh **0개**(go_stack 에도 없음). `_ubeat`(:7085) 배선은 정상 |
| `T2_HANDOFF_ARG_GROUND` | 설정 .sh 0개. 게다가 `if not wd and wag_specs` 라 **WAG 315회가 전부 상류에서 소비** |
| `T2_TRANSFER_PREREQ` | 설정 .sh 0개. 모듈 완성 · 자기 헤더가 표적 9건/9 sim 을 세어 놓음 |
| `T2_GROUND_DROP_NAVKEYS` | 3중 도달 불가 — 설정 0개 + `--resolve` 0건(`t2_resolve_patch.apply()` 미호출) + 형제 스펙 파일 부재 |
| `T2_PROV_ORIGIN` | 어느 런처에도 export 0건. 코드는 배선(:6312) |
| `T2_NLNUM_PROV` | 런처 export 0건. 배선 생존(:9828) |
| `T2_CLAIM_BLOCK` | 어느 런처에도 없음 = **[[60]] 진짜 표적**. 사전 인구조사 완료(표적 28건/23 sim · 과차단 후보 1) |
| `T2_UNKNOWN_UNVERIFIED` | 1로 세우는 런처 0개. 코드 축자 *"기본 OFF"* |
| `T2_TOOLERR` | 런처에 나오는 자리가 `unset` 목록 하나뿐 |
| `T2_READ_NEARDUP` | 레포 전 .sh 히트 0. go_stack:126 축자 *"OFF 유지(승격 조건: **오탐 계측 후**)"* — 그 계측이 없다 |
| `T2_REPEAT_GOV` | .sh 히트 0. 축자 *"死배선복구 · 런처 0(레거시 경로 사용 중)"* |
| `T2_SALVAGE` | 전 .sh grep 0건. 기본 OFF |
| `T2_MAIN_ANSWERS_ONLY` | .sh 히트가 **주석 한 줄뿐** — *"라이브 미측정 · 구성이 바뀌므로 별도 팔([[65]])"* |
| `T2_FEXEC` | .sh 4회 등장이 **전부 echo/감사 grep 줄**, `export` 0건 |
| `T2_MATERIAL_BYPASS` | 어떤 .sh 에도 0회. C498⒡ 가 스스로 철회 — *"앞선 수리는 표적을 빗나갔다"* |
| `T2_PENDING_DISCOVERED` | 15개 .sh 전부 **=0**. 게다가 켰더라도 0이었다(코드 축자: `_ts` NameError 를 `except` 가 삼켜 *"켜는 순간 처음부터 죽은 레버"*·수리 후 미점등) |
| `T2_DECLFIRST_GUIDE_FIX` | 4개 런처 전부 =0 · 유일 설계 A/B(t7324) 미착지. 축자 *"unified 경로에서 가이드는 **한 번도 주입된 적이 없다**"* |

### 4-2 (D-b) 켰는데 술어가 참이 된 적이 없다 — 표적 부재 후보

| 레버 | 확정 원인 |
|---|---|
| `T2_HAVE_VALUE` | 관측 전용 분기 `would-fire but suppressed by=`(:7138-7155)도 **0** ⇒ 배타 체인에 밀린 것이 아니라 **술어가 참이 된 적이 없다**. 형제 VALUE_ACQUIRE 1,780회로 배선 생존 증명 |
| `T2_COV` · `T2_COV_MIDDRIVE` | `_cov_formalize_M` 이 `len(recs)>=2` 를 요구하는데 recs 는 tool 출력 **dict 최상위 키**에 `entity_key` 가 있어야 쌓인다. banking `entity_key="transaction_id"` 인데 t7326 40 sim 전수에서 최상위 `transaction_id` **0건** ⇒ M 은 항상 `[]` |
| `T2_SCALAR_ARRAY` | ON 이었던 유일 런(ax32) **1,047 호출** · t7326 **908 호출** 에 정본 술어 통과 → 발화했을 자리 **0건** |
| `T2_SUPPRESS_AUTH` | banking A2 L3 `suppression_authority` 가 두 호출부 **모두** 유효 선언 보유 ⇒ `may_suppress` 항상 True ⇒ **OFF 와 거동 동일**. C320 축자 *"라이브 영향 0"* |
| `T2_WRITE_CAP` | 2겹 — `gated` 미설치(C261) + `_confirm_write_tools` 가 banking 에서 **∅**(C263) |
| `T2_WITHDRAWN_ROW` | 노브 ON 인데 발화 0. a2 `withdrawn_row_check` 매칭 tool 출력 인구 미측정 |
| `T2_PRINCIPLE_DEFAULT` | 켠 런(t5c·trivial36·abl106)의 로그에서도 0 ⇒ **치환한 적이 없다** |

### 4-3 (D-c) ★배선 결함이 확정된 것 — 즉시 수리 대상

| 레버 | 결함 (파일:줄 축자) |
|---|---|
| **`T2_CHOICE_GROUND`** | `_args_dict`(t2_gate_patch.py:387-397)가 바깥 dict 만 반환하는데 이 환경은 `call_discoverable_agent_tool(arguments=…)` 로 디스패치 ⇒ `_ar_cg.get(...)` 이 **항상 None** → `continue`(:11045). **같은 파일에 이미 있는 `_parse_nested_args`(t2_resolve.py:775)를 이 자리에서 안 쓴다** — 한 줄 수리 |
| **`T2_CALLABLE_HINT`** | go_stack:247 의 근거 *"[READ-FIRST] 44발화/18sim"* 은 **형제 `T2_SG_REQREADS` 의 것**이다 — 420 결과파일 전수에서 `[READ-FIRST]` 195건 중 **접미사 실명 포함 0건**. 원인 후보 = `registry(orch)` 빈 집합(모듈 자신이 적어 둔 함정) |
| **`T2_CONSISTENCY` / `T2_CONS_NOOP`** | 인쇄 태그가 `[T2_CONSISTENCY]` 가 아니라 **`[T2_CONS]`** — 그것으로도 414 로그 전수 0. 켠 .sh 3개(generalized_stack_v4/5/6)뿐이고 go_stack 에 없음. 술어가 `a2["eplan"]` + `_confirm_write_tools(a2)` 둘 다 요구 |
| **`T2_QUOTE_HINT`** | 표적 부재가 **반증됐다** — `bank_smoke8b_pin_20260805` 061 t0 의 실제 출력으로 `t2_quote_hint.hint('0', outs)` 를 돌리면 199자 힌트 + beat 가 난다. 원인은 상류(원격 플래그 적용·모듈 배포·A2 병합) |
| **`T2_PROV_GROUND`** | **원리상 불가** — t2_run_gated.py:222-223 축자 `raise SystemExit("[t2_run] T2_PROV_GROUND is not supported in unified mode (E-COMP scope). Use T2_GROUND=1.")`. 현행 정본 스택에서 켜면 런이 **기동조차 못 한다** |
| **`T2_PROV_ADDR_FULL`** | 발화 조건이 `prov_mode == "rescue"` 전용인데 go_stack:31 은 `T2_PROV_MODE=full` ⇒ **켜도 조건이 원리상 거짓**. `_FULL` 접미사라 `audit_unset` 에 지워져 감사에서도 안 보임 |
| **`T2_FIT_DIFF`** | 자기 태그 없음(`[T2_AXIS]` 4레버 공용). ON 이었던 유일 런에서 fit 표적 15건인데 그 런의 `[T2_AXIS]` 6회는 **전부 `call_/unlock_discoverable_agent_tool`** = fit 도구 부착 **0건** |
| **`T2_READALL`** | 상류 원장은 살아 있다(`T2_EPLAN_WALK` 331회) ⇒ 침묵 원인은 **플래그가 banking 런처에 없는 것** |

### 4-4 (D-d) 결과 파일이 영속되지 않아 판정 불가

`T2_PROVENANCE`(c3_* 태그 0건) · `T2_PROV_BADWORDS`(prov_eval 산출물 0건) · `T2_CLAIM_VERIFY`(run_one.sh `cverify` 팔 설계돼 있으나 실행 태그 0건).

**[주의]** DARK 는 "쓸모없다"가 아니라 **"발화한 적이 없다"** 이다. 4-3 의 8종은 배선 결함이 확정됐으므로 [[55]] 0단계 대상이다 — *모델의 결손으로 귀속하기 전에 우리 배관부터*.

---

## §5 UNMEASURED (82종) — 기능군별 · `to_measure` 한 줄

### L1 쓰기-근거 (10종)

| 레버 | 발화 | to_measure (한 줄) |
|---|---:|---|
| `T2_WRITE_ARG_GROUND` | 315/80런 | `=1↔=0` 짝에서 **gold 이름 deny 수 × 그 deny 후 회복 여부** |
| `T2_ARG_EMPTY` | 26/13런 | 발화 26건이 **gold 변이 집합 소속 도구**였는가 (현재 81%가 read 자리 = 점수를 만들 수 없는 자리) |
| `T2_REF_VERIFY` | 41/7런 | 현행 6인자 판본 replay 재인증 + deny 9건의 교정 성공률 |
| `T2_HAVE_VALUE_FORCE` | 신호 0 | 먼저 인쇄를 심어라 — `force_required=True` 로 간 턴 수와 그 턴에 죽은 `_gen_action_sub` 수 |
| `T2_WRITE_ARG_ENUM` | 192/21런 | 동봉 채널 `It answers: X` 의 **gold 일치율** + 동봉 제거 팔과의 reward 짝 |
| `T2_GROUND_HDR` | 고유 계기 0 | 설계서가 사전등록한 수 — **scalar 회복률 사전/사후**(현 38/58 · 사후 수치 repo 0건) |
| `T2_QUOTE_PIN` | 10+15 | `paired` 모드를 실제로 돌려라. 종점 = **pin_kind 라우팅이 판정을 바꾼 발화 수**(현재 0) 대 rate 드롭 손실 행 수 |
| `T2_PROD_BIND` | 44/23런 | **전-행 오강등률** — 강등 44행 중 실제로 producer 출력에 있던 행 비율 |
| `T2_SG_TRUTH` | 592/113런 | 되돌린 592건에서 **우리 도구의 답이 맞았는가** — 이 한 표가 부호를 정한다 |
| `T2_PARAM_CAP` | 18/15런 | ★먼저 분류 수리(`t2_levers.py:411` 이 META 로 오등재·실제는 A2 스펙을 읽어 deny). 그 다음 deny 18건의 서브 fail-open skip 비율 |

### L2 이름 원장 (13종)

| 레버 | 발화 | to_measure |
|---|---:|---|
| `T2_UNLOCK_NAME` | 1,461/128런 | deny 1,461건의 **회복률**(deny → 결국 옳은 이름으로 unlock 했는가) |
| `T2_UNLOCK_QUIET` | 13/2런 | 010 +1 · 099 −1 이 잡음인지 태스크-국소 실물인지 (nt≥8 재런) |
| `T2_UNLOCK_PROV` | 310/75런 | deny 310건 중 **레지스트리 실재 이름 차단 건수** |
| `T2_UNKNOWN_NAME_BL` | 255/34런 | 255건에서 env 가 Unknown 이라 한 이름이 `__discoverable__` 에 있었는가 — 하나라도 있으면 HARMFUL 로 이동 |
| `T2_UNKNOWN_REPEAT_GUARD` | 175/42런 (태그 `[T2_UNKNOWN_REPEAT]`) | 175건 중 반려 이름이 실제로는 레지스트리에 있던 건수 + cap 2 도달 후 통과분 결과 |
| `T2_FAB_STRIP` | 222/60런 | strip 222건의 제거 대상이 레지스트리에 있었는가 — **이 계기 하나면 L2 7종이 동시에 판정된다** |
| `T2_DISCOVERY_NAMES` | 1,849/84런 | 병기 이름 중 **다음 턴 실제 호출 비율** + 문맥 길이/CWE |
| `T2_DISCOVERY_REQUIRED` | 17/3런 | C112 지정 — **DISCREQ + (a1 env 문구 정정) 조합** + 삭제 편향 없는 하네스에서 20쌍 재현 |
| `T2_DISCOVERY_DISPATCH` | 1/1런 | 트리거 인구조사 — `ep_spec` 실재 턴 수 |
| **`T2_DISCOVERY_STEP2`** | **2,216/73런 (1위)** | **발화 2,216회 × 다음 턴 step2 실행 비율 + `=0` 팔 reward 짝. 표본은 이미 충분하고 없는 것은 대조 팔뿐** |
| `T2_UNCALLED_UNLOCK` | 362/85런 | *이름 발화 후 give 0* 비율 62%(x368) ↔ 10%(C529⒝) 간극이 이 가족 때문인지 |
| `T2_TOOLLIST` | 56/26런 | 캡 도달 sim 수 × 도달 이후 목록 밖 호출 수 (부호가 캡 값에 달려 있는데 그 곡선 미측정) |
| `T2_SELF_DECLARATION` | 4,830 서브콜 → 개입 **15** | **적중률 0.31%** — 4,830 서브콜의 지연 총량과 15회 개입 각각의 sim reward. (`t2_levers.py:471` NOT_LAUNCHED 등재는 stale·수정 필요) |

### L3 서술 출처 (6종)

| 레버 | 발화 | to_measure |
|---|---:|---|
| `T2_PROV_OURS` | 자기 마커 없음 | `1↔0` 에서 `[T2_UNLOCK_PROV] deny` + operator-fab deny 건수 · **gold 이름 오차단** 건수 · reward. 자기차단 44/56 이 줄어드는지가 생사 |
| `T2_PROV_MODE` | REGEN 에 포함 | 독립 판정보다 위 짝(29/50↔31/50)을 n≥200 으로 키워 **rescue 의 주소-날조 통과분과 full 의 과개입분을 동시에** |
| `T2_SOURCE` | 인쇄 1,639 : 실전달 343 (4:1) | `1↔0` 20태스크×nt2. 2급 종점 = **합병 메시지 길이**와 그 턴의 gold 호출율 |
| `T2_WRITE_PROV` | 마크 12,181 : 실발화 **3** (524:1) | ★먼저 계기 수리(`window hit` 을 발화 카운터에서 분리). 그 다음 창 12,038 ↔ regen 141 의 격차 원인 |
| `T2_GIVE_QUOTE` | 734/96런 · 마크:전달 1:1 | 철회된 188건 중 **gold 가 실제로 그 give 를 요구한 건수** |
| `T2_UNAVAIL_PROMISE` | 1,121/152런 | 발화를 `unavailable`↔`locked` 로 가르고 각 갈래 뒤 **다음 턴 unlock 여부** |

### L4 회수·집합 차 (5종) — ★표적 최대 · 발화 최소

t7326 실측: COV 0 · COV_MIDDRIVE 0 · READALL 0 · DISPATCH_LEDGER 2 · COVERAGE_FU 2 = **합 4발화**인데 다중요구가 gold 결말의 **150/289(52%)**.

| 레버 | 발화 | to_measure |
|---|---:|---|
| `T2_LEDGER` | 1,925/101런 | ★상류 정정 먼저 — `ep_led` 를 세우는 것은 **`T2_EPLAN` + `a2['eplan']`**(:6002-6008)이지 T2_LEDGER 가 아니다. 잴 것 = `1↔0` reward + 뷰 큐잉이 밀어낸 메시지 수 |
| `T2_DISPATCH_LEDGER` | 등재 208 ↔ 표면화 **0**(t7326) | 등재↔표면화 격차 원인(`_dispatch_ledger_check` 가 왜 None 인가). 그 뒤 reward + **여분 제출 건수** |
| `T2_COVERAGE_FOLLOWUP` | 153/70런 (태그 `[T2_COVERAGE_FU]`) | 표적이 2도구뿐 — 그 태스크만 모아 `1↔0`. 종점 = 미판정 행이 그 뒤 실제로 판정됐는가 |
| `T2_TRANSFER_LEAVES_STEPS` | 148/35런 (표면화는 **17%**) | 표면화 14건에서 **이관 취소 후 남은 단계가 실행됐는가** |
| `T2_SG_DEDUP` | 197/49런 | 스텁 197건 각각에서 **그 사이에 write 가 있었는가**(있었다면 이전 결과는 낡았다) |

### L7 말-행동 등가 (10종 전원)

| 레버 | 발화 | to_measure |
|---|---:|---|
| `T2_FORCE_ACTION` | 5,490/261런 | `action-required` 분기가 걸리는 태스크만 골라 짝 A/B — reward + **강제된 그 턴이 낸 도구**(표적 ↔ 검색). C330: `tool_choice=required` 는 *아무 도구나* 를 사고 모델이 그것을 검색에 쓴다 |
| `T2_DECIDE_ANY` | 13/9런 | 결정점이 열리는 태스크(070/071/075)에서 reward + **099/100 거동 불변(Δspurious 0)** — 코드가 스스로 적어 둔 의무인데 OFF 팔이 없다 |
| `T2_DISPATCH_ROLE` | 1,571/123런 | deny 나는 태스크(003·017·063)에서 reward + **막힌 give 중 gold 요구 건수** |
| `T2_DISPATCH_ROLE_ENVSET` | 부모와 태그 공용(귀속 불가) | ★먼저 계기 — print 에 분기명(`user_held`/`agent_tool`/`unknown`)을 찍어 부모와 분리 |
| `T2_TOOL_CHANNEL` | 699+beat 541/101런 | 이 레버는 **자리가 둘**(:1038 출력-부착 ↔ :10996 호출-前 regen). 두 자리를 각각 갈라 reward + **붙인 노트 총 문자수** |
| `T2_USER_TOOL_NOTE` | 845/136런 | give 태스크(018·019·040)에서 reward + **give 후 손님이 실제 실행한 sim 수** |
| `T2_UNINSTRUCTABLE` | 54/31런 | 012 계열에서 reward + **존재하지 않는 도구명·앱 경로 날조 건수**(기준선 0) |
| `T2_GIVE_EXEC_NUDGE` | 528/99런 | 표적 모집단이 62%→10% 로 떨어졌다(C529⒝) — 그 비율이 살아 있는 태스크군을 먼저 특정 |
| `T2_GIVE_RELEVANCE_NUDGE` | 10/7런 (현행 스택 0) | *한 sim 에서 give 2회 이상* 태스크가 실재하는지(현행 술어 전제) + **gold give 넛지 0** |
| `T2_GUIDED` | beat **3,169/181런** | 스키마밖 이름 방출이 실제로 나는 라이브 태스크에서 reward + 스키마밖 호출 건수 (격리는 대조군도 0/2 라 원리상 불가) |

### L8 부재 종결 (7종)

| 레버 | 발화 | to_measure |
|---|---:|---|
| `T2_SEARCH_EXHAUST_NUDGE` | 605/122런 | 012·032·033 에서 reward + **넛지 이후 새 문서 id 회수 여부** |
| `T2_KB_NOHIT_SURFACE` | 28/12런 (현행 0) | ★먼저 **왜 현행 스택에서 0인가** — alltools 전환으로 `Score:` 행 채널이 남았는지 |
| `T2_PROC_ABSENT` | 274/54런 | 050·051·048 에서 reward + **지목된 잔여 단계 도구 호출 비율**. x86 K-sweep 이 이미 write 0 을 냈으므로 **사전 기대치 null** |
| `T2_SG_WINDOW_ABSTAIN` | 57/7런 (현행 0) | `_expected_groups` 선언 도구가 현행 로스터에서 호출되는지 + **정당한 0 을 abstain 으로 접은 건수** |
| `T2_ABSTAIN_FIELDS` | 37 + 궤적 113 | 019·020·026 에서 reward + **지목 후 재호출에서 결핍 필드가 채워진 비율** |
| `T2_NOREC_BRANCH` | 궤적 103/97 sim (stderr 0) | v1↔v2 **문면만** 갈라(엔진 분기 순증 0) reward + 같은 인자 재조회 횟수 + **조기 종결 건수** |
| `T2_RETURN_EMPTY` | 궤적 2/1 sim (near-dark) | `get_reward_discrepancies` 빈 결과 sim 에서 문면만 갈라 reward + **빈 결과 위 여분 dispute 건수** |

### L9 국면 배치 (9종)

| 레버 | 발화 | to_measure |
|---|---:|---|
| `T2_DECIDE_BEFORE_WRITE` | 27/14태그 (수리 후 미대조) | 수리된 가드로 ON/OFF 재실행. 1차 종점 = **유예된 write 의 인자가 서브 답으로 바뀌었는가** + Δspurious |
| `T2_DECISION_CARRY` | 2,305/70태그 | 이것만 0↔1 + **슬롯 클로버(`[T2_CP2_CLOBBER]`) 팔-대칭 계수** — 안 그러면 이득과 그것이 지운 배달이 분리 안 됨 |
| `T2_DECLFIRST` | 3,802/13태그 | t7324 착지 · **가이드 주입 한 축만** 가르고 종점 = reward + Δspurious |
| `T2_DECLFIRST_GUIDE` | 주입 발화 0 | go_stack 명시 등재(DEFAULT_ON 사각 제거) 후 `_GUIDE` 만 0↔1 · **가이드 문면 도달 훅**을 배선 게이트로 |
| `T2_DELIVER_PRECOMMIT` | 12회 (실도달 8/12) | 슬롯 덮어쓰기 수리 → **팔-대칭 실제-부착 지표** → 부피 통제 팔 추가로 '시점'과 '부피' 분리 |
| `T2_SEARCH_ON_PROCEED` | 440/26태그 | `T2_NOW_SELFCALL` 양팔 ON 고정 후 이것만 0↔1. 1차 종점 = **결정 직전 생성에 재료가 있었던 sim 비율** |
| `T2_PHASE_OWNER` | 441/7태그 (현행 0) | `phase_of` 를 무료로 돌려 **`verify` 단계에 들어가는 sim 수**부터 |
| `T2_NOW_SELFCALL` | 515/29태그 | SEARCH_ON_PROCEED 고정 후 단독. 배선 게이트 = `now 미확정 침묵` 46→0 |
| `T2_TERMINAL_TURN` | 계기 결함(발화 판정 불가) | ★`terminal_turn_note` 에 자기 태그를 붙여 mention_note 와 분리(계기 수리)부터 |

### L10 형식화·되묻기 (4종)

| 레버 | 발화 | to_measure |
|---|---:|---|
| `T2_DISAMB_ORDER` | 미확인(태그 없음) | ★먼저 [[59]] 위반 확정 — :4524 가 엔진에 `order`/`order_id` **도메인 리터럴**을 박고 있다. A2 `disamb_args` 이설 후에야 측정 가치 |
| `T2_SG_ISOLATE` | **7,849/151태그 (2위)** | go_stack 에서 이것만 0 으로 내린 귀속 arm 하나([[19]] 허용). 종점 = reward + **격리 실패 메인 폴백 비율** + 서브콜 지연. **7,849회 발화하면서 순이득 미측정 = 이 군 최대 공백** |
| `T2_SG_ISOFB` | 궤적 170/32태그 (자체 태그 없음) | `T2_TOOLLIST` 고정 후 단독. 1차 종점 = **ground 실패 operand 가 다음 라운드에 회복된 비율** |
| `T2_PRESCRIPTION` | 34/21태그 (태그당 1~7) | 배타 체인에서 이 레버까지 내려오는 턴 수(분모) 확인. 분모가 얇으면 **통합 대상** |

### L11 계산 이관·원장 대조 (8종)

| 레버 | 발화 | to_measure |
|---|---:|---|
| `T2_CALC` | 궤적 10,074/41런 (retail) | retail 로스터에서 nt 동일·동시 실행으로 `T2_CALC` 만 뒤집고 **발화 0 부정통제 팔**과 함께 |
| `T2_TRANSCRIBE` | 마크 118 · 효과(deny) **18** | 표적(017·019·020·022)만 모은 로스터에서 ON/OFF reward + **deny 가 막은 호출 중 gold 소속 비율** |
| `T2_BRANCH_REGROUND` | 100/23런 | close 표적을 가진 043 계열 nt≥4 짝(현재 근거는 n=1·Δ0) |
| `T2_READ_DEDUP` | 3,015 + 궤적 2,276 | ★`exec_augment` 설치 조건을 별 플래그로 **분리**한 뒤 스텁 주입만 0/1 (분리 전 A/B 는 216줄 교락) |
| `T2_NO_DIGEST_REEXEC` | 궤적 17/4런 | ax32 스택 고정 후 단독. reward + **재유입 문자수** |
| `T2_STALE_STRIP` | 183/48런 | 단독 짝 reward + **strip 된 호출 중 gold 이름 비율**(오차단 직접 계측) |
| **`T2_PRESENT_NESTED`** | 궤적 **5,331/25런** | **같은 로스터·같은 nt·동시 실행 + 발화 0 부정통제 팔. 이 하나면 이 배치 첫 VALID 가 나올 수 있다** |
| `T2_DUP_REPRESENT` | 158/47런 + 궤적 170 | `T2_SG_DEDUP` 이 반복을 잡는 로스터에서 0/1 + **재제시 후 동일 호출 재발률** |

### L12 재료 배달 (4종) — ★격리는 다 재놨고 라이브 짝만 비었다

| 레버 | 발화 | to_measure |
|---|---:|---|
| `T2_SEARCH_AGENT` | 4,141/76런 · t7326 415 | 0/1 짝. **1차 종점은 성적이 아니라** 코드가 사전등록한 것 — *"결정 직전 생성에 재료가 있었는가"* |
| `T2_ACTION_INDEX` | 266/36런 | 0/1 짝(n≥8) reward + **gold 도구 이름 호출률**(격리 x319 10/24→24/24 를 라이브에서) |
| **`T2_WRITE_SUB`** | **9,993/42런 (최다)** | **같은 시드·같은 스택·같은 런처에서 `=0` vs `=3` n≥8. 이 배치 최우선** |
| `T2_ACTION_SUB` | 453/37런 | **098 표적**으로 같은 런처·동시 실행·nt≥8. C406 의 +2 칸 재현 여부 |

### L13 출구 거버넌스 (5종) — ★다른 모든 군의 A/B 를 교락시킨다

| 레버 | 발화 | to_measure |
|---|---:|---|
| `T2_ARBITRATE` | 3,994/91런 | 나머지 넷 고정 후 단독 0/1, `_reqs` 비지 않는 태스크 팔당 ≥20 sim. **`identical demand suppressed`(:7846)는 이 플래그의 형제이지 자식이 아니므로 계정에서 빼라** |
| `T2_SURFACE_BUS` | beat 87/40 sim (stderr 0) | reward 이전에 무료로 — 런당 `budget\|skipped_replay\|honesty_reject\|failclosed` beat 수. t7326 은 87 중 **2** ⇒ 사실상 통과 장치 |
| `T2_KEEP_DENY_BODY` | 1,514/79런 | 010·099 팔당 nt≥10, **접힘 사건 수를 공변량으로 층화** + 사전등록 Δspurious |
| `T2_STACK_WINDOW` | 6,334/106런 | 자기 비상구(`=0`)를 쓴 팔 하나. sim 당 접힘 수를 반드시 공변량으로 |
| `T2_WINDOW` | **10,571/106런 (최다 발화군)** | 단독 A/B + 사전등록 Δspurious + `stop=resolve_cap` 계수. go_stack:347 이 *"Δspurious 필수 계측"* 을 적어 놓고 **한 번도 보고 안 됐다** |

### L14 형식·채널 (1종)

| 레버 | 발화 | to_measure |
|---|---:|---|
| `T2_CALL_FORM` | 778/26런 | ARBITRATE·WINDOW·ARG_EMPTY 양팔 고정 후 단독. 099 계열 팔당 ≥20 sim + **`T2_ROUTE_TRACE` 의 `lost_to`** 로 억제/체인탈락/미생성 3분해 |

### L5·L6 (30종) — 미판정

분류만 확정. **L5 선행 강제**(18): EPLAN·EPLAN_WALK·EPLAN_REPLAN·EPLAN_READS_ONLY·EPLAN_EXAMINED_SAFE·SCAFFOLD_GET·SG_REQREADS·PREKB·PIN_READ·PIN_READ_STEPS·PROCEDURE·TOOLGATE·REQUIRE_DOC·PROCEED_DOCBODY·DOCS_AT_WRITE·ARG_PRODUCERS·SPEAK_PROHIBIT·ELIG_LINE. 인용 가능한 선행 판정 2건 — `T2_ARG_PRODUCERS` = HARMFUL(넛지 705회 전부 검색/read 지목 · gold 변이 도구 0건 · 술어가 도구 결과 산문 substring = [[59]] 위반) · `T2_DOCS_AT_WRITE` = DARK(t7304 단일변수 055 0/8↔0/8 · 본체 양팔 0발화).
**L6 완결·후속**(12): FOLLOWUP_REQUIRED·FOLLOWUP_FORCE·FOLLOWUP_READLOOP·UNVERIFIED_FOLLOWUP·VERDICT_SURFACE·VERDICT_CARRY·VERDICT_GATE·TERM_GRANT·TERM_GRANT_USERDEMAND·NOTICE_REPEAT·TRANSFER_TIER·SUB_REQUIREMENT.

---

## §6 ★★통합안 — "N개로 줄인다 = 통합"

원칙: **끄지 않는다**([[60]]). 같은 결손을 겨냥하고 **같은 자리에서 발화하는** 레버들을 하나로 접되, 접으면서 잃는 것을 반드시 적는다.

### L1 쓰기-근거 → **P1_ARGBASIS 하나**

- **합칠 후보(6)**: `T2_WRITE_EVIDENCE` · `T2_WRITE_ARG_GROUND` · `T2_ARG_EMPTY` · `T2_REF_VERIFY` · `T2_ASK_UNKNOWN_BOOL` · `T2_HANDOFF_ARG_GROUND`.
  **통합 실체는 이미 하나다** — spec 로딩이 여섯 줄 연속(t2_gate_patch.py:6017·6020·6024·6028·6032·6035)이고, 판정은 :7049-7105 의 **단일 `wd` 폴스루**가 순서대로 시도한다. 남은 것은 이름 6개뿐.
- **별건 통합(2)**: `T2_WRITE_ARG_ENUM`(:8684) + `T2_ARG_AXIS`(:8716) — wd 체인이 아니라 이름-deny 이웃에서 발화하고 **`T2_WRITE_ARG_ENUM_CAP` 을 나눠 쓴다**.
- **합치면 잃는 것**:
  1. **지금도 귀속이 안 된다** — :7110 `_lbeat("T2_WRITE_EVIDENCE", …)` 가 `_wtag` 와 무관하게 하드코딩돼 6종 발화가 전부 WEV 로 계수된다. **선결 = `_wtag` 를 beat 에 넘기는 한 줄.** 그 전에는 어느 조각이 이득/손해인지 원리상 못 가른다.
  2. ENUM+AXIS 를 합치면 캡 공유로 인한 상류 억제는 해소되지만, 두 술어(집합 밖 이름 ↔ 축 종류)가 한 캡을 쓰던 현행 거동이 바뀐다.
- **합친 뒤 재는 법**: 단일 `P1_ARGBASIS` 를 0↔1 로 가르고 종점 = reward + **gold 이름 deny 후 미회복 건수**(현행 22건이 gold 오차단 56 중 39%). 조각별 기여는 `_wtag` 층화로만 읽는다.

### L2 이름 원장 → **P3_NAME 7비트 튜플 하나**

- **합칠 후보(15 전원)**: 술어가 전부 **같은 세 집합의 소속**이다 — env 레지스트리 ∩ 이미 받은 도구 출력 텍스트 ∩ 호출 원장. 축자(:7379) *"집합의 출처는 env 레지스트리 ∩ 이미 받은 도구 출력 텍스트뿐이다(닫힌 집합·정규식 0·도메인 어휘 0). 고르는 일은 LLM."*
- **통합의 실질**: 7비트(실재/회수/해제/전달/호출/반려/우리가말함) 하나. 그것이 서면 **자기지목–자기차단이 정의상 성립 불가**가 된다 — t7326 deny 173 중 56(32%)이 자기차단이고 gold 오차단 56 중 44 가 여기서 나왔다.
- **합치면 잃는 것**: 지금 문면이 다르다 — 차단은 이름을 대고 표면화는 목록을 준다. 한 튜플로 접으면 **[[64]] 요구("무엇이 틀렸나 + 무엇을 하면 풀리나")를 문면 층에서 다시 세워야 한다.**
- **합친 뒤 재는 법**: 튜플 하나에 대해 0↔1 + **오차단률 단일 계기**(strip/deny 대상이 레지스트리에 있었는가). 그 계기 하나로 현행 7종이 동시에 판정된다.

### L3 서술 출처 → **regen 채널 하나**

- **합칠 후보(18)**: `T2_PROVENANCE`(구세대) · `PROV_REGEN`(후계) · `PROV_BADWORDS` · `PROV_GROUND` · `PROV_ORIGIN` · `PROV_ADDR_FULL` · `PROV_MODE` — t2_run_gated.py:201-208 이 **연속 4줄**에서 읽어 같은 regen 호출로 넘기고 재시도 예산(:225·:240)까지 공유한다.
- **근거**: 라이브 사이드카 2,153행 중 `unified_regen` 이 **1,412행(66%)**, 개별 레버 문면은 대부분 한 자릿수 도달(writeprov 3 · givequote 4) ⇒ **개별 이름을 줄여도 노출은 안 변하고, 바꿀 수 있는 것은 채널이다.**
- **합치면 잃는 것**: `PROV_MODE`(full|rescue)와 `PROV_GROUND` 는 이미 상호 배타로 강제돼 있다 — `raise SystemExit("[t2_run] T2_PROV_GROUND is not supported in unified mode …")`. **합치기 전에 unified 모드가 무엇을 쓰는지 확정해야 한다.**
- **합친 뒤 재는 법**: 채널 자체(regen 1회)를 0↔1 로 가르되 **무내용 재시도 부정통제 팔**을 반드시 동반([[57]]) — 지금 PROV_REGEN 의 +2.0pp 가 문구 효과인지 재발화 효과인지 원리상 안 갈린다.

### L4 회수·집합 차 → **P5_SETGAP 하나** (단, 통합보다 결손 측정이 먼저)

- **합칠 후보(9)**: LISTED∖EXAMINED · TARGETS∖SUBMITTED · GIVEN∖RAN 을 원소 하나로. **상류 단일점이 이미 하나다** — `ep_led`(COV :6120 · COV_MIDDRIVE :6150 · READALL :6487 이 동일 가드).
- **[금지] 순서 역전**: 이 군은 '레버가 많아서' 문제가 아니라 **표적이 최대(52%)인데 발화가 4회**인 자리다. `entity_key` 형상 불일치로 `_cov_formalize_M` 이 항상 `[]` 를 돌려준다. **[[62]]① 대로 격리로 결손부터 재고, 그 다음에 통합·점등한다.**
- **합치면 잃는 것**: 대상별 문면(어느 집합의 차인가)이 사라지면 [[64]] 재료가 없어진다 ⇒ **층 태그를 인자로 남긴다.**
- **합친 뒤 재는 법**: `M≥2 ∧ remaining≠∅` 이 서는 턴 수(표적 크기)를 먼저 세고, 0 이 아니면 P5_SETGAP 0↔1 로 reward + **미판정 행이 그 뒤 판정됐는가**.

### L5 선행 강제 → **P9_REQUIRE 하나** (판정 미착수)

- **합칠 후보(18)**: 같은 A2 구조(`gates[]`·`require_tool_before`·`requires_reads`·`eplan`·`procedures`)를 읽고 같은 형태의 deny/directive 를 낸다. 배선도 한 줄기(t2_run_gated.py:252→261→273 순차 apply).
- **비용 0**: `requirements_for()`/`merged_text()` 가 이미 라이브 배선(:7537/7589/7952)이고 새 A2 키 0.
- **합치면 잃는 것**: 지금은 **문면이 사슬의 끝을 명령**하는 판본과 **다음 한 수를 명령**하는 판본이 섞여 있다. 합치면 후자로 통일되는데 그것은 거동 변화 ⇒ [[62]] 4문을 먼저 답해야 한다.

### L6 완결·후속 → **이름 통합이 아니라 `_resign` 출구 병합** (이 배치 최대 소득)

- **사실**: 단일 술어 `_resign`(:9850) 하나가 **11자리**를 연다(:10272·10296·10317·10345·10378·10419·10495·10656·10767·10941·10959). 그리고 그 자리는 **배타 `elif` 체인이라 턴당 하나만 나간다.**
- 코드 축자(:9222-9226): *"위 `elif` 는 같은 tool_call 에 대해 **하나만** 내보낸다 … 오프라인 32/32 인 문장이 라이브에서 3/6 만 닿았고, 원인의 절반이 이 배타성인 것을 오늘에야 코드 추적으로 알았다 — **계수가 없어서 몰랐다.**"*
- **함의**: 이 창을 두고 L2(UNCALLED_UNLOCK) · L3(WRITE_PROV·CLAIM_PROV) · L7(GIVE_EXEC_NUDGE) · L8(SEARCH_EXHAUST_NUDGE) 이 **군 경계를 넘어 경합**한다 ⇒ **군별 A/B 가 rank 교락을 안고 있다.**
- **통합안**: 이름을 12→1 로 줄이는 것이 아니라 **출구를 병합**한다(명령 하나 + 사실 합집합·L13 ARBITRATE 가 이미 그 형태). 그러면 *"어느 레버가 좋은가"* 라는 질문 자체가 rank 교락에서 벗어난다.
- **합치면 잃는 것**: 병합하면 문면이 길어진다 — **[[65]](메인에는 답만)와 정면 충돌** ⇒ 길이 상한을 함께 정해야 한다.

### L7 말-행동 등가 → **P6_SPEECH 하나** (본문 ∩ 레지스트리 − CALLED)

- **합칠 후보(10)**: 자리가 두 지점뿐 — tool_calls 빈 턴(:8305 FORCE_ACTION · :11067 UNINSTRUCTABLE) 또는 디스패처 호출(:8398 DISPATCH_ROLE · :8439/8460 ENVSET · :11095 USER_TOOL_NOTE).
- **합치면 잃는 것**: `DISPATCH_ROLE_ENVSET` 의 **3갈래 문면 분기**(손님이 이미 가진 도구 / 우리 에이전트 도구 / 미상·:8443-8459)를 하나로 접으면 [[25]] 가 잡은 사고(*"같은 명제에 두 진리값"*)가 역방향으로 되돌아온다 ⇒ **문면 3분기는 유지한다.**
- **합친 뒤 재는 법**: **사전 기대치를 null 로 두고** 설계한다 — 인접 축 실측이 이미 null 이다(C492 8/20↔9/20 · C529 2/12↔2/12). 종점을 reward 로만 두지 말고 **over-action 을 팔-대칭으로 함께** 센다.

### L8 부재 종결 → **원소 하나 + 층 태그 인자**

- **합칠 후보(9)**: 술어가 전부 *집합이 비었는가* 하나이고 출력이 전부 *없다 + 찾은 범위*다(KB 검색 :11164 · stub 누적 :10419 · 절차 :6702 · 판정 상세 t2_scaffold_get.py:1860 · 서브 창 t2_compute.py:226 · :343 · :675).
- **합치면 잃는 것**: 각 조각이 **서로 다른 층에서** 빈 집합을 본다. 접으면 *어느 층이 비었는지*가 문면에서 사라지고 그것이 [[64]] 의 "무엇을 하면 풀리나" 재료다 ⇒ **층 태그를 인자로 남긴다.**
- **합친 뒤 재는 법**: 층 태그별 층화 후 reward + *부재 선언 이후 같은 검색 반복 횟수*.

### L9 국면 배치 → **합치지 말고 예산 총량 고정·사용처 이동**

- **이 군만 다르다**: 레버를 더하지 않고 **배치만 바꾼다** ⇒ [[63]](빼기) 형태의 처치를 만들 수 있는 유일한 자리.
- 코드 축자(:8180-8186): *"전부 써 버렸고(DELIVER 3·3·2·3), 정작 상품을 고르는 turn 14+ 에는 재료가 문맥에 **없었다** … 격리 24/24 ↔ 라이브 0/4 의 기전이 이것이다. ⇒ 일반 자리 배달을 **1회로 묶고** 나머지 예산을 결정 자리에 남긴다. **총량은 그대로 3**이고 새 판단 기구도 없다 — 같은 예산의 **사용처만** 옮긴다."*
- **정리 대상**: `T2_DECLFIRST_ENFORCE` 는 NOT_A_LEVER(소비자 0) → 레버 목록에서 내린다. `T2_DECLFIRST_GUIDE_FIX` 는 `T2_DECLFIRST` 의 배선 수리이지 별 레버가 아니다 ⇒ **한 이름으로 합치되 t7324 착지 전에는 판정 금지.**
- **[금지]**: 국면 자기평가('아직 결정 중인가')는 LLM 이 못 하는 부류로 확정([[66]]·judge6 −6) ⇒ 이 군의 술어는 **인용-근거(가리키기 + substring 검산)** 로만 세운다.
- **★재는 법의 핵심**: 이 군의 A/B 는 **합으로 읽으면 안 된다**(§7 참조).

### L10 형식화·되묻기 → **통합 방향이 반대: 분리한다**

- 다른 13군과 갈리는 결정적 차이 — **여기서만 엔진이 값을 고를 수 있다.** 그래서 [[62]]·[[10]] 금지선이 정확히 이 군에 걸리고, 실측이 그것을 지지한다(RESOLVE 3/5→0/5 · OPERATOR_PINPOINT 24/24→0/24 · L4 스스로 치환 OFF).
- **분리 후보**: `T2_RESOLVE` 를 **채널 3분해** — `action-required`(행동 촉구) / `operator-scope`(범위 표면화) / `reference-filter`(참조 차단). 지금은 13,766회 발화가 한 이름 뒤에 숨어 어느 채널이 파는지 못 가른다.
- **분리하면 잃는 것**: RESOLVE 가 지금 나르는 **재료(요건 문면)** 까지 같이 사라진다 — 그 회귀가 실측됐다(`REGRESSION_2026_08_18_CAP_LATCH.md:29-36` · 098 14/15→0/4) ⇒ **재료 배달을 L12 로 먼저 이관한 뒤** 지목을 끊는다.
- **파라미터 정리**: `T2_DISAMB_MODE`·`T2_L4_MODE` 는 NOT_A_LEVER(부모의 모드). 단 `T2_DISAMB_MODE=enumerate` 라벨이 붙은 과거 런들은 **코드에 `enumerate` 분기가 없어 조용히 dialog 로 떨어졌다** ⇒ 그 라벨로 낸 결론은 철회 대상.

### L11 계산 이관·원장 대조 → **P2_EVENT 하나 (`_call_key` 동일성)**

- **합칠 후보(7)**: `READ_DEDUP`(:5025) · `READ_NEARDUP`(:5175) · `NO_DIGEST_REEXEC`(:5052) · `WRITE_CAP`(:702) · `STALE_STRIP`(:9575) · `RETRY_CONTROLLER`(:683) · `REPEAT_GOV`(:5088) — 전부 **`_call_key`(name+args) 동일성** 하나로 판정한다.
- **선결(구조)**: `T2_READ_DEDUP` 은 지금 **채널이자 레버**다 — `exec_augment` 본문 216줄을 감싸므로 끄면 arm 이 조용히 능력을 잃는다(축자: *"오늘 READ_DEDUP 하나로 런 6회를 태웠다"*). **설치 조건을 별 플래그로 분리**하지 않으면 어떤 A/B 도 216줄 교락이다.
- **합치면 잃는 것**: 이 군은 [[59]] 위반 위험 최대 — `t2_compute.py:301-521` `catalog_filter` 에 **banking 카드 필드 42 리터럴 · ~220줄 전부 판정**. 통합하면서 도메인 리터럴을 들고 가면 전이가 성립하지 않는다.
- **이름 정정**: `T2_COMPUTE` 는 유령 — 정본은 **`T2_CALC`**.

### L12 재료 배달 → **합칠 것이 아니라 먼저 잴 것**

- **이 군만 새 결정론을 0개 추가한다** — 축자(:3436) *"새 결정론 0 — 우리가 **덜 올릴 뿐**이고, 고르는 일은 그대로 모델이다"*. [[65]]·[[63]] 이 직접 지지하는 유일한 군.
- 격리 실측이 가장 강하다: `WRITE_SUB` x307 **0/8 ↔ 7/8** · x308 7~8/8 · x309 8/8 · 부정통제 `D_NOBASIS` 0/8 (차 7~8 ≥ 잡음 바닥 5). `ACTION_INDEX` x319 **10/24 → 24/24**(블록 8·8·8).
- **[주의] [[62]]① 규율**: 둘 다 **결손 측정이지 레버 효과가 아니다.** 라이브 귀속은 재현 실패 — t7313 ctl 073 1.0 ↔ treat 0.0 인데 **양팔 `T2_WRITE_SUB=3` 동일**, t7326 073 [0.0, 0.0].
- **결론**: 이 군은 통합 대상이 아니라 **측정 대상 1순위**다.

### L13 출구 거버넌스 → **합치는 대신 rank 를 노출한다**

- 이들은 무엇을 말할지 정하지 않는다 — 다른 13군이 만든 문면들이 **어떻게 한 턴에서 하나로 나가는지**를 정한다.
- **이 군이 다른 모든 군의 A/B 를 교락시킨다**: 억제 총량 t7326 **606회**를 네 기구가 나눠 집행(지문창 150 · deny 본문 접힘 116 · 지침 drop 84 · ARBITRATE 72 · PHASE_PRECEDE 184) · `route ≠ chose` **76턴** · **개입 강도 sim별 4~118(30배)** ⇒ **층화 또는 공변량 없이 어떤 레버 A/B 도 해석 불가.**
- **통합이 아니라 계기 통합**: `T2_ROUTE_TRACE` 의 `lost_to` 를 상설 공변량으로 승격해 모든 A/B 표에 *(억제 / 체인 탈락 / 미생성)* 3열을 붙인다.
- **부호가 갈리는 유일한 조각**: `T2_KEEP_DENY_BODY` 는 억제가 아니라 그 반대 방향([[64]]) ⇒ 이 군 안에서 별도 취급한다.
- **계정 정정**: `identical demand suppressed`(:7846)는 `T2_ARBITRATE`(:7528)의 **형제**이지 자식이 아니다(들여쓰기 36 동일) ⇒ C330·C429⒞ 가 잰 손해와 `LEVER_CONSOLIDATION:44` 의 "ARBITRATE 72" 는 **ARBITRATE 의 부작용이 아니다.**

### L14 형식·채널 → **줄이면 안 된다**

- 가르는 술어(t2_levers.py:436-444 축자): *"`finish_reason == \"length\"` → **채널 사실**. 도메인에 대해 아무 주장도 하지 않는다 → 하네스 / \"이 카드가 자격이 되는가\" → **도메인 판정** → 레버"*.
- 축자(:459): 끄면 능력이 조용히 사라지는 게 아니라 **"측정 자체가 무효가 된다"**(2026-08-07 실측 3회).
- **단 두 개는 레버 논쟁이 열려 있다**:
  - `T2_VIEW_COMPACT` — **이 군 최대 미측정 위험.** t7326 **34/40 sim** 에서 원문 대신 요약을 보고 결정했다. 소화된 행에 gold 값이 있었는지 **한 번도 대조하지 않았다**. 하나라도 있으면 하네스가 아니라 레버다.
  - `T2_ENVELOPE_GUARD` ↔ `T2_SALVAGE` — **술어가 바이트 수준으로 같고 응답만 다르다**(`not am.tool_calls and _envtag in content` ↔ *"tool_calls 가 비어 있고 ∧ 본문에 `<tool_call>` 이 있다"*). ⇒ SALVAGE 의 표적 인구 = ENVELOPE_GUARD 가 이미 **223회 발화한 그 자리**. **재야 할 것 = 그 223 자리에서 회수(첫 블록만) vs 재생성 중 어느 쪽이 reward 를 더 사는가**, 그때 잃는 것은 회수 쪽의 '나머지 블록'.

---

## §7 ★태스크별 부호 문제 — 합으로 null 인데 특정 태스크에서 양

### 7-1 실물 목록

| 레버 | 합 | ★양의 칸 | 음의 칸 | 런 태그 |
|---|---|---|---|---|
| **T2_ACT_DEMAND** | 8/20 ↔ 9/20 (null) | **073 2/5 → 4/5 (+2)** | 050 1/5 → 0/5 | `bank_t7297_{ctl,treat}_20260815q` |
| **T2_HANDOFF_PREDICATE** | 2/12 ↔ 2/12 (null) | **028 0/2 → 2/2 (+2)** | 019 −1 · 029 −1 | `bank_t7308_{ctl,treat}_20260818c` |
| **T2_ACTION_SUB** | 9/12 ↔ 7/12 (Δ2 < 바닥) | **098 3/3 ↔ 1/3 (+2)** · 010 +1 | 099 −1 | `bank_asubON` ↔ `bank_isoOFF_20260810` |
| **T2_UNLOCK_QUIET** | 3/6 ↔ 3/6 ↔ 3/6 (null) | **010 2/3 (↑ 1/3·0/3)** | 099 1/3 (↓ 2/3·3/3) | `bank_uq_20260811` (C408) |
| **T2_DECISION_ISOLATE** | 6/12 ↔ 7/12 (−1) | 010 1/3 ↔ 0/3 | 099 −1 · 100 −1 | `bank_iso{ON,OFF}_20260810` |
| **T2_PROV_MODE** | 29/50 ↔ 31/50 (+4pp) | 9↑ (101 0→2 · 32·38·43·46·69·7·77·83) | 6↓ (110·23 2→0 · 33·52·84·94 2→0) | `compabl_noP_eplan{,_rescue}` |
| **T2_PROV_REGEN** | McNemar p≈0.52 (null) | 33↑ · **t17 0/4 → 4/4** | 35↓ · **t61 4/4 → 0/4** | `fl32b_floor_retail_t4` ↔ `prov_e2e_retail_t4` |
| **T2_DISAMB** | 260/456 ↔ 263/456 (−3) | 8↑ · **t61 0/4 → 4/4** · t16/t18 2→4 | 6↓ · **t46 4/4 → 0/4** · t47·t95 4→1 | `routerv1_retail_t4` ↔ `prov_e2e_retail_t4` |
| **T2_MATCH_COUNT** | 24/64 ↔ 24/64 (null) | 5↑ (012·018·019·021·034) | 5↓ (004·008·017·020·035) | `bank_ax33n_*` ↔ `bank_b4_*` |
| **T2_DELIVER_PRECOMMIT** | 5/12 ↔ 6/12 (+1) | **024 2/4 → 3/4** | — (055·098 불변) | `bank_t7303_{ctl,treat}_20260816h` |
| **T2_PRESENT_NESTED** | +20pp | 13↑ (7·23 각 0/2→4/4 · 42·52·92) | 7↓ (전부 ON 저조 칸) | `comp_retail_t4` ↔ `compabl_noP` |
| **ax32 축-레버 묶음**(TOOL_CHANNEL·TERMINAL_TURN·NO_DIGEST_REEXEC 등 10~11종) | 18/62~63 ↔ 23/62~63 (Δ+5·바닥 안) | 007 0/2→2/2 · 008 0/2→2/2 · 022 · 017 · 021 · 019 · 035 | 020 2/2→0/2 · 001 · 018 · 027 | `bank_qp32p*` ↔ `bank_ax32p*` |
| **T2_DISCOVERY_REQUIRED** | 9/20 ↔ 11/20 (p≈0.34) | 시드 4승 | 2패 (14 동률) | `bank_ctl2_nt20` ↔ `bank_dreq2_nt20_20260718` |
| **T2_RETRY_CONTROLLER** | ★**scale 별로 갈린다** | 32B **null**(p=0.656) | 소형 4/4 arm-pair 음(p=0.0037) | `c8_gate` ↔ `c8_gate_retry` |

### 7-2 현재 방식이 체계적으로 버리는 것

**현재 방식 = "전 태스크 합 하나로 스위치를 정한다".** 이 방식이 버리는 것은 셋이다.

1. **상쇄로 지워지는 태스크-국소 이득.** 위 14건 중 **11건이 합 null 인데 태스크 단위로는 ±2 가 나 있다.** 합으로만 읽으면 `T2_ACT_DEMAND`(073 +2) · `T2_HANDOFF_PREDICATE`(028 +2) · `T2_ACTION_SUB`(098 +2) 가 전부 "효과 없음"으로 폐기된다. **폐기되는 것은 레버가 아니라 그 레버가 유일하게 통하는 태스크다.**

2. **부호가 갈리는 축을 통째로 못 본다.** `T2_RETRY_CONTROLLER` 는 소형에서 유의하게 해롭고 32B 에서 null 인데, C262 가 자기교정한 축자대로 **전 scale 풀링(198 vs 162·p=0.065)에서는 "근거 없음"** 이 된다. 같은 함정이 태스크 축에도 있다.

3. **결손이 다른 태스크를 한 스위치가 지배한다.** L4 는 다중요구가 gold 결말의 **52%** 인데 발화 4회, L12 는 격리에서 0/8→7/8 인데 라이브 짝 0. 합 하나로 켜고 끄는 한, 표적이 있는 태스크와 없는 태스크가 같은 값을 받는다 — 표적 없는 태스크의 부작용(지연·CWE·over-action)이 표적 있는 태스크의 이득을 상쇄한다. `T2_HANDOFF_PREDICATE` 가 그 실물이다: **028 +2 를 사면서 CWE 13건·지연 1.90× 를 전 태스크에 지불했다.**

### 7-3 처방 (설계 규칙으로 승격)

- **[규칙 A] A/B 결과표는 합과 태스크별 부호를 **항상 함께** 적는다.** 합만 적은 원장 항목은 [D] 로 강등한다.
- **[규칙 B] 판정선을 두 개로 둔다.** ① 합 기준 차 ≥5(팔당 ±4/20·C483) ② **표적 태스크 기준 차 ≥5**(그 태스크만 nt≥8~20). ②만 넘으면 `VALID_LOCAL`(태스크-국소 유효) 로 등재하고 스위치는 태스크-조건부로 남긴다.
- **[규칙 C] 표적이 없는 태스크에서의 부작용(지연·CWE·over-action)을 **팔-대칭 지표로 같은 표에** 적는다.** 그것이 [[등대 §1]] 모트 — *하나를 사면 하나를 판다* 를 표로 만드는 유일한 방법이다.
- **[금지]** "전 태스크 합이 null 이므로 폐기" 라는 문장 — 태스크별 부호를 보지 않은 채로 쓰지 않는다.

---

## §8 측정 큐 — 정보량 순 다음 A/B 5개

### 0단계 (유료 런 전 · 무료 선결)

| # | 할 일 | 이유 |
|---|---|---|
| G1 | `_wtag` 를 `_lbeat` 에 넘기는 **한 줄**(t2_gate_patch.py:7110) | L1 6종 발화가 전부 WEV 로 오귀속 — 고치기 전 L1 A/B 는 귀속 불가 |
| G2 | `t2_match_count.shown_in()` 정규식 수리(:88-91) | **계기 결함이 곧 모델 입력 결함**(거짓 완결 인증 19/102). 고치기 전 A/B 무효 |
| G3 | `T2_ROUTE_TRACE` `lost_to` 를 **상설 공변량**으로 승격 | L13 개입 강도 sim별 4~118(30배) — 층화 없이 어떤 A/B 도 해석 불가 |
| G4 | L4 `entity_key` 형상 계약 수리 후 **M≥2 서는 sim 수** census (무료) | 표적 52%인데 발화 4회. [[62]]① — 안 재고 켜는 것은 금지 |
| G5 | 인증 회귀 3건 복구 — `test_ref_verify.py:67`·`test_ref_verify_replay.py:85`(TypeError·6인자 미반영) · `test_c207_envelope.py:145`(ValueError·`_fu_cap` 소실) | 인용되는 8/8·25/25 가 **출고본을 검정하지 못한다** |

### Q1 — `T2_WRITE_SUB` (L12) ★최우선

| 항목 | 값 |
|---|---|
| **변수** | `T2_WRITE_SUB` **0 ↔ 3** (단일변수. 나머지 PIN 전량 양팔 바이트 동일) |
| **표적** | write operand 결정이 있는 태스크 — **073 · 075** 포함(+072·074 는 분모 0이므로 제외) |
| **표본수** | 팔당 **n≥20 sim**(태스크×시드 짝지음). 최소 n≥8 |
| **종점** | 1차 **reward** · 2차 서브콜 수(사전등록 문턱 ≤20) · 3차 사이드카 실도달 |
| **부정통제** | 격리에 이미 있다 — `D_NOBASIS`(근거 제거) **0/8** = 날조 안 함 |
| **판정선** | 차 **≥5**(C483). 넘으면 **이 배치 첫 VALID** |
| **왜 1순위** | 격리 사슬 x307~x310 이 **0/8 ↔ 7/8**(차 7~8 ≥ 바닥 5)로 다 재어져 있고 **라이브 짝 하나만 비어 있다.** 그리고 이 군은 **새 결정론 0**([[63]]·[[65]] 지지) |
| **선결 주의** | 라이브 귀속 재현 실패가 이미 있다 — t7313 ctl 073 1.0 ↔ treat 0.0 인데 **양팔 `=3` 동일**. 같은 런처·동시 실행이 아니면 또 시드 분산을 잰다 |

### Q2 — 태스크-국소 부호 재현 (L9·L12) ★방법론 payoff 최대

| 항목 | 값 |
|---|---|
| **변수** | 두 짝을 각각 단일변수로 — ⒜ `T2_ACT_DEMAND` 0↔1 · ⒝ `T2_HANDOFF_PREDICATE` 0↔1 |
| **표적** | ⒜ **task_073 단독**(합 null 에서 +2 났던 칸) · ⒝ **task_028 단독**(0/2→2/2) |
| **표본수** | 태스크당 팔당 **nt≥20**(같은 시드 집합·동시 실행·같은 런처) |
| **종점** | 1차 **reward** · 2차 **over-action 팔-대칭**(`ONLY-PRED` 계수) · 3차 지연·CWE |
| **부정통제** | ⒜ task_098 거동 불변 칸(양팔 5/5 실적) · ⒝ ctl 팔 `named-but-not-given` 0회 검산 |
| **판정선** | **표적 태스크에서 차 ≥5** → `VALID_LOCAL` 등재. 안 넘으면 원장 C492·C529 폐기 판정이 최종 |
| **왜 2순위** | 이 하나가 **§7 규칙 A~C 의 채택 여부**를 정한다. 재현되면 "합 null 이므로 폐기" 로 버려진 11건을 전부 다시 봐야 하고, 안 되면 합 판정이 옳았음이 확정된다 — **어느 쪽이든 판정 방식 자체가 결정된다** |

### Q3 — `T2_PRESENT_NESTED` (L11) ★최대 미청구 효과

| 항목 | 값 |
|---|---|
| **변수** | `T2_PRESENT_NESTED` **1 ↔ 0** (calc·gate·prov 양팔 고정) |
| **표적** | retail 로스터 — `comp_retail_t4` ↔ `compabl_noP` 의 **공유 25 태스크** |
| **표본수** | **같은 nt**(4↔4) · **동시 실행** · 팔당 25태스크 × nt4 = 100 sim |
| **종점** | **reward**(pass^1) + pass^4 · 부수로 `[OPERAND DISAMBIGUATION]` 부착 수 |
| **부정통제** | ★**필수 신설** — 발화 0 팔(같은 자리·같은 길이·재제시 없음). 지금 없는 유일한 조건 |
| **판정선** | 차 ≥5/100. 현행 관측 **+20pp**(72/100 ↔ 26/50)는 여유 2배 |
| **왜 3순위** | 이 배치에서 **잡음 바닥을 확실히 넘은 유일한 양의 효과**. 규칙 0 위반이 아니다(에이전트 자신의 응답 레코드 재제시). 단 nt 4↔2 불일치·비동시 실행·부정통제 부재 셋을 고치면 **첫 VALID 후보** |
| **주의** | 이득이 [[62]] 결손을 지우는 형태인지 함께 본다 — 형제 `T2_PRESENT_READS` 는 조회 5.5× 억제·날조 1.9× 증가로 폐기됐다 |

### Q4 — `T2_ACTION_SUB` @ task_098 (L12)

| 항목 | 값 |
|---|---|
| **변수** | `T2_ACTION_SUB` 0↔1 |
| **표적** | **task_098 단독**(C406 에서 3/3 ↔ 1/3 = +2 났고 기전까지 특정됨) |
| **표본수** | 팔당 **nt≥20** · **같은 런처·같은 GPU·동시 실행**(C406 의 결함이 정확히 이것) |
| **종점** | 1차 **reward** · 2차 **넘긴 인자가 gold 값이었는가**(C406⒞: OFF 실패는 `Dark Green Account`·`Light Blue Account`, ON 성공은 전부 gold `Blue Account`) |
| **부정통제** | P5 Δspurious — gold 밖 write · 게이트 거부(현행 양팔 0) |
| **판정선** | 차 ≥5 |
| **왜 4순위** | §2 표에서 **가장 VALID 에 가깝다**(4조건 중 3.5). 산 것이 "넘김"이 아니라 **"넘길 때 실린 값"** 이라는 기전이 이미 특정돼 있어 종점이 명확하다 |

### Q5 — `T2_DISCOVERY_STEP2` (L2)

| 항목 | 값 |
|---|---|
| **변수** | `T2_DISCOVERY_STEP2` 1↔0 (57개 .sh 에 상수로 박혀 있는 것을 팔 변수로 승격) |
| **표적** | 발견형 도구가 나오는 전 로스터(발화가 73런에 고루 분포) |
| **표본수** | 팔당 **n≥20 sim** — **표본은 이미 충분하고 없는 것은 대조 팔뿐** |
| **종점** | 1차 **reward** · 2차 발화 2,216회 × **다음 턴 step2 실제 실행 비율** |
| **부정통제** | ctl 팔 `[T2_DISCOVERY_STEP2]` 발화 0 검산(배선 오염 확인) |
| **판정선** | 차 ≥5 |
| **왜 5순위** | **이 배치 발화량 1위(2,216회 / 73런)인데 reward 대조가 한 번도 없다.** 발화가 가장 많은 레버가 한 번도 안 재어진 것이 이 감사의 요약이고, 이득이든 손해든 이 규모면 신호가 나야 정상이다 |

### 대기열 (Q6 이후 · 정보량 순)

`T2_SG_ISOLATE`(7,849 발화·순이득 미측정·귀속 arm 하나) → `T2_WINDOW`(10,571 발화·사전등록 Δspurious 미보고) → `T2_RESOLVE` 채널 3분해(13,766 발화·HARMFUL 확정이나 어느 채널이 파는지 미분리) → L4 `P5_SETGAP`(G4 통과 후) → `T2_VIEW_COMPACT`(34/40 sim 에서 요약이 원문을 대체·gold 행 대조 0회).

---

## §9 계기 한계 — 이 감사가 신뢰할 수 없는 것들

### 9-1 파일 명명이 **세 가지**다

`<tag>.results.json.gz` / `<tag>_results.json.gz`(밑줄) / `<tag>.json.gz`.
`glob("*.results.json.gz")` 만 쓰면 t7273~t7299 를 통째로 놓친다(250 ↔ 실제 420). **세 번째 형태(`bank_isoON_20260810.json.gz`)는 정본 `t2_forensic.all_result_files()` 도 못 잡는다** — 직접 로드해야 보인다. 정본 진입점:

```python
import sys; sys.path.insert(0, r"C:/workspace/ba-frft/scripts/distill/tau2")
import t2_forensic as F
F.all_result_files(); F.iter_all_sims({"task_073"})
```

### 9-2 태그 ≠ 플래그명 (태그 오조회 함정)

| 플래그 | 실제 인쇄 태그 |
|---|---|
| `T2_UNKNOWN_REPEAT_GUARD` | `[T2_UNKNOWN_REPEAT]` |
| `T2_CONSISTENCY` / `T2_CONS_NOOP` | `[T2_CONS]` |
| `T2_DECISION_ISOLATE` | `[T2_R8B]` |
| `T2_COVERAGE_FOLLOWUP` | `[T2_COVERAGE_FU]` |
| `T2_SG_ISOFB` | `[T2_SG_ISOLATE] … ground-피드백` (자체 태그 없음) |
| `T2_DECIDE_BEFORE_WRITE` | `[T2_DECIDE_BEFORE_WRITE] write 1턴 유예` (한글 본문) |
| `T2_CLAIM_PROV` | `[T2_CLAIMPROV]` · `T2_WRITE_PROV` → `[T2_WRITEPROV]` · `T2_ARG_SCHEMA` → `[T2_ARGSCHEMA]` |

### 9-3 **마크 ≠ 전달** (인쇄를 발화로 세면 최대 524배 과대)

| 레버 | 마크 | 실제 전달 | 비율 |
|---|---:|---:|---:|
| `T2_WRITE_PROV` | 12,181 (`window hit` 1,570) | **3** | **524 : 1** |
| `T2_SELF_DECLARATION` | 4,830 (선언 서브콜) | **15** | **322 : 1 (적중률 0.31%)** |
| `T2_CLAIM_PROV` | 12,622 | 1,206 | 10.5 : 1 |
| `T2_TOOL_SIGNATURE` | 44 (t7326) | 5 | 8.8 : 1 |
| `T2_SOURCE` | 1,639 | 343 | 4.8 : 1 |
| `T2_GIVE_QUOTE` | 87 | 87 | 1 : 1 (예외 — 인쇄=발화) |

### 9-4 공유 태그로 귀속 불가

- `[T2_AXIS]`(4,148~4,453회) — **4레버 공용**(TOOL_CHANNEL · TERMINAL_TURN · FIT_DIFF · SCALAR_ARRAY). x44:74 축자 *"발화>0은 존재 증명 수준"* 이라 **네 레버가 같은 6건을 각자 자기 것으로 셌다**.
- `[T2_DISPATCH_ROLE]` — 부모/자식(`_ENVSET`) 단일 print 공용 ⇒ 어느 분기가 발화했는지 사후 불가.
- `[T2_WRITE_EVIDENCE]` beat — `_lbeat` 하드코딩으로 L1 6종 오귀속(**단 stderr `_wtag` 는 조각별로 정상 인쇄된다** — 오귀속은 beat 채널에서만).
- `T2_GROUND_HDR` — 자기 태그가 없고 하는 일이 `[GROUNDING WARNING]` **꼬리 문자열 삭제**뿐 ⇒ SG_GROUND 444건과 **원리상 구분 불가**.

### 9-5 stderr 만 보면 오독하는 형태 4종

1. **정상 시 침묵형** — `T2_MATCH_COUNT`(stderr 는 오류 때만·궤적에 151건 부착) · `T2_PAIRCHECK`(불변식 깨질 때만·0 = 정상이지 死배선 아님).
2. **beat-only 형** — `T2_GUIDED` beat **3,169/181런** · `T2_SURFACE_BUS` beat 87/40 sim (stderr 태그는 실패 경로 전용).
3. **문면 교체형** — `T2_NOREC_BRANCH` 는 stderr 0인데 궤적 축자 *"Use a DIFFERENT identifier each time"* **103회/97 sim/11런**.
4. **비커밋 뷰형** — `T2_SEARCH_AGENT`·`T2_WRITE_SUB`·`T2_MAIN_ANSWERS_ONLY` 는 배달물이 궤적에 안 남는다. **코퍼스 문자열 스캔 0 을 死배선으로 읽지 말 것.**

### 9-6 死배선 오독 3건 정정 (본 조사 확인)

`LEVER_CONSOLIDATION §2-b` 의 死배선 C군 분류와 `OPERAND_LEVER_AUDIT:198` 의 *"t7326 stderr 미영속"* 은 **틀렸다** — t7326/t7328 4런의 `.log.gz` 는 sim_results 에 실재하고 직독된다. 그 로그에서 `T2_BRANCH_REGROUND`(t7295 6+4회) · `T2_GROUND`(t7326 halfB 13 · t7328 halfB 18) · `T2_DECIDE_BEFORE_WRITE`(1·2·4)가 확인된다. **셋 다 死배선이 아니고 로스터에 표적이 없을 뿐이다.**

### 9-7 결과 미영속으로 판정 불가한 런 (설계는 있으나 데이터가 없다)

`qp32on`/`qp32off`(paired 모드 미실행) · `bank_cf_20260811`(CALL_FORM) · `bank_uq/kb_20260811`(로컬 재현 불가·원장 인용) · `abl106s_*`(leave-one-out 6팔) · `c3_*`·`c4_*`(PROVENANCE·AUTOFETCH) · `prov_eval`·`ground_eval` 산출물 · `bank_t7324_*`(DECLFIRST A/B 미착지) · `bank_reg043dd_base` · `bank_r095h_isofb`.

### 9-8 레지스트리(`t2_levers.py`) stale 4건

| 항목 | 오등재 | 실제 |
|---|---|---|
| `T2_SELF_DECLARATION` | `NOT_LAUNCHED`(:471) | go_stack:341 이 켜고 **101런에서 발화** |
| `T2_DISCOVERY_REQUIRED` | `NOT_LAUNCHED` | 현행 스택 기준으로만 맞다 — 과거 3런은 라이브 |
| `T2_PARAM_CAP` | `META`(:411 *"레버가 아니라 레버에 대한 규칙"*) | **A2 `param_cap_check` 를 읽어 실제 deny 하는 레버**(:1104·:7194). 진짜 파라미터는 `T2_PARAM_CAP_CAP` |
| `T2_COMPUTE` | 셀 등재(:340) | **유령** — 정본은 `T2_CALC` |

### 9-9 통계 신뢰를 깨는 두 편향

1. **삭제 편향** — `run_with_retry` 가 실패 궤적을 **버리고 재추첨**하고 `save_fn` 은 성공분에만 걸린다 ⇒ 버려진 궤적은 저장조차 안 된다. C112 축자: dreq2 가 나쁜 궤적을 더 많이 버렸다 = **Δ와 같은 방향 상향 편향**. **모든 소차(Δ≤2) 결과가 이 편향 안에 있다.**
2. **층화 붕괴자** — C262 축자: *"'레버 실발화 sim만 층화'는 retry 를 파멸적으로 해로움으로 보이게 했다(p=3.5e-9). **이것은 인과가 아니라 붕괴자다**"* ⇒ 그 수치 인용 금지. 같은 함정이 본 문서의 모든 *"발화 sim vs 무발화 sim"* 대비에 있다(L7·L8 서술 관측 전부 **선택 교락**이지 인과가 아니다).

### 9-10 개입 강도가 sim 마다 30배 다르다

t7326 실측 개입 강도 **4~118**. 억제 총량 606회를 네 기구가 나눠 집행하고 `route ≠ chose` 76턴. ⇒ **층화 또는 공변량 없이 어떤 레버 A/B 도 해석 불가.** G3(`lost_to` 상설 공변량)이 유료 런 전 선결인 이유가 이것이다.

---

*정본 판정 155건 + 기능군 14 + 비-레버 census 331. 갱신 시 `RESEARCH_MASTER.md` §증거원장에 등급과 함께 반영할 것 — 이 문서의 수치는 provenance 를 갖고 있으나 scratchpad-only 인용은 없다.*
