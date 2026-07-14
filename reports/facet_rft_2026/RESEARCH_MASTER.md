# RESEARCH MASTER — 등대 문서 (single source of truth)

> **이 문서가 최상위다.** 모든 실험·설계·특허·논문은 여기서 파생되고 여기로 되돌아온다.
> 표류 = 이 문서를 안 읽고 지엽에 들어가는 것. **작업 시작 전 §0·§1을 읽고, 작업 종료 시 §3 원장을 갱신한다.**
> 확정: 2026-07-08 (프레임 LOCK·논문 4분할). **정렬: 2026-07-09**(C43~C48·출처선언 레버·§0·§1.4·§1.7·§4·§5 갱신).
> **§1 프레임은 LOCKED — 새 측정의 반증 없이 재론 금지([[03]]).**

---

## 0. 한 줄 (논문 헤드라인 · 변하지 않음)
**Scale은 horizon을 사지, guarantee도 semantic reference도 사지 못한다.** guarantee는 결정론 scaffold가 pass-비용 0으로 산다.
symbolic 추론은 test-time compute가 싸게 산다 — **단 그것은 persistence를 판다**. 그러므로 레버는 독립 배분이 아니라
**측정된 상쇄에 의한 합성(composition)**으로 배치한다. **부작용 없는 레버는 없다.**
잔여(semantic reference)는 우리가 시험한 어떤 레버도 열지 못한 **경계**다.
> ★**2026-07-09 정련(C43~C48)**: frontier 격차의 큰 조각이던 **operand 날조는 능력 결손이 아니라 *정박 치환***(문맥 인접 id의
> 변형·WM 아님·C43)이고, **"출처를 안 대도 되는 인터페이스"의 산물**이다. **출처 선언 4지선다 + provenance 검증이 pass비용
> 0로 닫는다**(C45·32B 날조 67→0%·over-block 0·Δspurious 0·present 없이). ⇒ **닫은 뒤 남는 유일 경계 = ⋈**(C46·후보 2+개서
> 옳은 값 *선택*). learn 축(gather 학습)은 미확립 — cfbsynth가 결손을 재현 못 해 시험된 적 없음(C38).
> ⚠️ **C9 단서(2026-07-08)**: tau2-retail서 **horizon 복리 감쇠는 관측되지 않았고**($p_{step}pprox1.0$ 전 모델),
> frontier와의 격차는 **F2 symbolic operand**에 집중된다(⋈ 경계는 frontier와 **공유**). ⇒ 이 벤치서 F6은 잔여가 아니며,
> **격차는 닫히는 축**에 있다. "scale은 horizon을 산다"는 [S-lit](DR#2)이지 우리 벤치의 binding constraint가 아니다.

## 1. 🔒 LOCKED 프레임 (재론 금지)
### 1.1 기능 분해 (에이전트의 논리적 작업)
| # | 기능 | 실패 형태 |
|---|---|---|
| F1 | compliance/guarantee | 정책 위반 |
| F2 | symbolic operand (비교·계산·기준) | 잘못된 값/변형 |
| F3 | semantic operand (⋈ 참조매칭·의도) | 틀린 대상 |
| F4 | coverage/completion (all/both/every) | 미완 |
| F5 | persistence/escalation | 조기 포기 |
| F6 | horizon (복리 $p^H$) | 누적 붕괴 |

### 1.2 기능 → 최저비용 레버
| 기능 | scale | thinking | scaffold | 최저비용 레버 |
|---|---|---|---|---|
| F1 | invariant | 직교 | **위반 0** | **scaffold** |
| F2 | 미약 | **✅ 싸다**(단 전-궤적 적용 시 F5 매도·**결정점 격리하면 채널 폐쇄**) | calc/present(토큰0) | **격리 thinking sub-call + 결정론 실행** |
| F3 | flat | ✗ | ✗ | **없음 = 경계** |
| F4 | invariant | **✗ 악화** | 완결 게이트 | **scaffold** |
| F5 | ? | **✗ 악화** | persistence 게이트 | **scaffold** |
| F6 | **✅ 산다** | 부분 | 분해 | **scale (싸게=fleet)** |

### 1.3 ★제1원리 — 역효과와 합성 (모트)
레버는 **하나를 사면 하나를 판다**. 이기는 배치는 **부작용이 서로 상쇄되는 합성**이다.
| 레버 | 산다(+) | 판다(−) | 순 | 상쇄 파트너 |
|---|---|---|---|---|
| thinking | F2 결정정확도 | F4·F5 완결/persistence | ≈0 | 완결게이트 [D] |
| present | F3 grounding | over-action | ≈0 | g15 status-lock |
| present+g15 | — | — | **+12.3pp** | (합성이 이득) |
| **완결게이트 자신** | F4·F5 | **scope 규율(over-action↑)** | 미측정 | — (자기-역효과) |
| retry·투표 | — | 해로움 / +0% | 음성 | 죽은 레버 |
> **법칙은 우리 처방에도 적용된다.** 합성은 무한후퇴가 아니라 **측정된 상쇄**여야 한다 → 모든 게이트에 `Δspurious ≤ 0` 같은
> 반대편 계측을 필수로 단다.

### 1.4 🔒 부하 vs 능력 — 기능별 진단·처방 (측정 기반)
> **부하(load) 정의**: $\text{load}(s) = p_{iso}(s) - p_{traj}(s) > 0$ — 격리하면 푸는데 궤적서 못 푼다.
> ★**측정 규율**: $p_{iso}$는 **에이전트가 그 지점에 실제로 갖고 있던 정보와 맞춰야** 한다. 정보가 더 빈약한 프로브로
> 재면 "부하 없음"이 정보량 차이일 뿐이다(역도 성립). **정보-맞춘 격리 replay만이 부하를 잰다.**

| 기능 | 실패 양상 | 진단 | 측정 근거 | 처방 | 등급 |
|---|---|---|---|---|---|
| **F1 compliance** | 확인 없이 write | **결정가능·미집행** | g2 rate scale-flat(.103/.070/.075) | **결정론 게이트** → 위반 0·pass비용 0 | [S] |
| **reach / plan-structure** | 주문 누락·배칭 오류 | **★부하(load)** | `PLAN_PROBE §1` t99: *격리 계획선 2주문 다 정답, 실제 런선 1주문 누락+날조* | **plan/execute 분리 + 결정론 controller** | [M] |
| **F2 변형 선택** | 실재하나 틀린 변형 | **★부하 아님 = 능력** | $p_{traj}$**0.762** > $p_{iso}$0.727 (궤적이 정보 더 많음). frontier $p_{traj}$**0.908~0.919** = **15pp 능력격차** | thinking(격리천장 .864<frontier) / scale·fleet(미검) / learn(미검) | **[M]**·정보-맞춤 재검 필요 |
| **F2b 계산형 기준**(예산·최저가) | compound criterion | **결정가능** | CoT .538→**.538**(thinking 무효) | **형식화(LLM)→결정론 실행(argmin/filter)** | [P] |
| **F3 ⋈ 참조매칭** | 틀린 주문 | **경계**(+탐색 절반) | 격리천장 ~.44 (scale·budget·CoT·RL 전부) · E3: wrong-pick의 43~52%는 gold **이미 조회**했는데 틀림 · frontier도 동일(14 vs 12) | **없음**(map) · 나머지 절반은 탐색=reach | [S]부분 |
| **F4 coverage** | all/both 미완 | **불변**(scale·thinking 무효·thinking은 악화) | 17≈16 | 완결 게이트 — **단 write 강제는 금지**(§1.5) | [S]/[D] |
| **F5 persistence** | 성급한 escalate | **불변**·thinking 악화 | QwQ transfer 24% vs base 13% | persistence 게이트 — **단 결정론 판별 상한 존재**(C3) | [M]/[D] |
| **F6 horizon** | 복리 붕괴 | **본 벤치서 미발현** | $p_{step}\approx1.0$ 전 모델 | (해당 없음) | [M] |
| **operand 날조** | 없는 id 발명 | **정박 치환**(WM 아님·C43) — 문맥 인접 id의 edit≤2 변형 70% | 32B 미조회날조 70%가 근접변형·read 0-3회 27%→6+회 2.1% | **★출처 선언+provenance 검증**(C45·67→0%·Δspurious 0) — 차단(C12)은 환경이 이미 함 | **[M]** |
| **over-action** | 안 시킨 write | LLM scope 잔여 + 정책 precondition(C25 8/12·**단 DB-state 아니라 대화 semantic**·C50) | passing-spurious QwQ 0 vs base 47 | scope=**게이트 금지**([[06]]) / precondition **DB-게이트 NO-GO**(C50)→대화-controller/ASK | [S]/[M] |

### 1.4b 🔒 frontier 격차 전수 분해 (C15·성공-실행 호출 기준·456 sim)
| 원인 | ours | o4-mini | Δ | gpt-4.1 | Δ | 처방 축 |
|---|---|---|---|---|---|---|
| **NO-WRITE: 모든 시도 ERROR** | 16 | 1 | **+15** | 1 | **+15** | **repair**(유효 후보 제공·차단 아님) |
| ~~NL/communication~~ | 23 | 12 | ~~+11~~ | 16 | ~~+7~~ | ❌**철회(C19: 채점기준 불일치)** |
| **F3 ⋈ 틀린 주문** | 37 | 30 | +7 | 16 | **+21** | 경계(map) + 탐색(reach) |
| **F2 wrong variant** | 32 | 26 | +6 | 24 | +8 | thinking/능력 (E1′가 판정) |
| OVER-ACTION | 5 | 0 | +5 | 2 | +3 | 게이트 금지축 |
| other operand(address1) | 6 | 2 | +4 | 3 | +3 | operand |
| MISSED-THIS-WRITE | 21 | 22 | −1 | 21 | 0 | — |
| other operand(payment) | 3 | 9 | **−6** | 5 | −2 | (우리 우세) |
| **NO-WRITE: 시도조차 안 함** | 8 | 19 | **−11** | 12 | −4 | (우리 우세·frontier는 기권) |
| **합계** | | | **+34 = 7.5pp** | | **+46 = 10.1pp** | |
- **★"F2가 유일한 격차"는 철회.** F2는 34 중 +6(18%).
- ~~**최대 조각 = "모든 write 시도가 ERROR"(16 vs 1)**~~ → **❌ 철회(C27)**: o4-mini의 `never-attempted 19`를 뺀 비교였다.
  공통 버킷(성공 write 0)으로 재면 **24 vs 20 = +4**. **repair는 최대 조각이 아니다.**
- **★gpt-4.1 대비 ⋈이 +21(46%)** — "⋈은 frontier와 공유"는 H3-4 부분집합 착시였다(전 구간 37 vs 16 = 2.3×).
- **★A4 실증**: o4-mini는 *기권*(never-attempted 19)해서, 우리는 *틀리게 행동*(all-errored 16·over-action 5)해서 실패한다.
- caveat: sim당 **근인 1개**만 귀속(다중 원인 가능) · NL 버킷은 judge 의존 · frontier 파일은 공식 하네스.

### 1.4c 🔒 DB-only 정본 분해 (C22~C28·456 sim·infra 0·최소-diff 매칭)
> **§1.4b는 gold action_checks 혼입(C20) + index-pairing 아티팩트를 안고 있다. 아래가 정본.**

| 조각 (vs o4-mini) | Δ | 처방 축 |
|---|---|---|
| **over-action (MORE+EXTRA)** | **+9** | ★**정책-precondition 게이트**(C25 — scope residual 아님) |
| **⋈ order_id** | **+8** | 경계(map) |
| item 집합 혼합 | +6 | write-scope |
| **주소 free-text 날조** | **+5** | ★**E9′ provenance repair**(C24·Δspurious 0) |
| ZERO write (repair 대상) | +4 | repair (**최대 조각 아님**·C27) |
| op 불일치 | +3 | 대부분 상류 날조·user-sim 이탈의 결과(doc §6·§10) |
| FEWER (미완) | **0** | — (coverage 격차 없음) |
| **F2 변형선택** | **−4** | (성공-write 한정·선택편향·C23) |
| **payment_method** | **−11** | (우리 우세) |

- **잔여의 이름은 `precision`이 아니다**(C18 부분정정). 정확히 그 값(F2 변형·payment)은 우리가 앞선다.
  남는 것은 **⋈ 참조 · 집합 범위 · 미조회 날조**.
- **reason enum·variant-leak 버그는 레버 아님**(C26·C28).

### 1.5 🔒 원인 → 해결 결정절차 (순서대로·먼저 걸리는 데서 멈춤)
```
Q1. 술어가 decidable한가?
    ├─ yes → Q1b. 그 술어가 이미 집행되고 있나? (환경·기존 scaffold)
    │        ├─ yes → 레버 아님 (C12: decidable ≠ 유용).  예) operand 날조
    │        └─ no  → ★결정론 scaffold (부작용 없음).      예) F1 compliance, F2b 계산
    └─ no  → Q2

Q2. 부하인가?  (정보-맞춘 격리서 p_iso > p_traj)
    ├─ yes → ★결정론 분리/controller (plan↔execute 분리·context 격리).  예) reach/plan-structure
    │         ※ thinking·learn 아님. 스킬은 이미 있다.
    └─ no  → 능력이다 → Q3

Q3. 어떤 종류의 능력인가?
    ├─ symbolic (비교·계산·기준) → thinking이 산다
    │     └─ ★scope 주의: 전-궤적 thinking은 F5를 판다 → 결정점 격리 시도(단 F2선 정보손실로 역효과 가능)
    ├─ semantic (참조·의도) → thinking 무효 → Q4
    └─ compound/계산형 → Q1으로(결정론 실행)

Q4. scale이 사는가? (측정된 scale-민감도)
    ├─ yes → scale, 싸게 사려면 fleet(위임)
    │     ※ 위임 조건 3: (i)큰 tier가 *측정상* 더 잘함 (ii)격리 sub-call이 토큰-싸다 (iii)이산 결정점
    └─ no  → ★경계(boundary): map·수용. 또는 learn(미검증·C7).   예) F3 ⋈

Q5. 실패가 "틀림"이 아니라 "안 함"인가? (coverage·persistence)
    └─ 완결/persistence 게이트 — ★단 **읽기만 강제, 쓰기는 절대 강제 금지**
       (abstain→forced act 전환은 pass +p / 피해 +(1−p); ⋈서 p≈0.44 <0.5 ⇒ 기대-유해)
```

### 1.6 🔒 레버별 사정거리 (무엇을 살 수 있고 무엇을 못 사나)
| 레버 | 산다 | 못 산다 | 부작용 | 적용 조건 |
|---|---|---|---|---|
| **결정론 scaffold(게이트)** | decidable·미집행 술어 (F1) | 의미 판단(F3·F5 판별) | **없음**(단 write 강제 시 over-action) | Q1 통과 |
| **결정론 controller(분리)** | **부하**(reach/plan) | 능력 결손 | 턴 예산 | Q2 yes |
| **thinking** | symbolic 능력(F2 부분·격리천장 .864) | semantic(F3)·compound(.538 flat) | **전-궤적 적용 시 F5 매도** | Q3 symbolic |
| **learn** | 미검증(C7) | — | 망각·역전이(C4/M-σ) | Q4 no & 경계 |
| **scale / fleet(위임)** | 측정된 scale-민감 기능 | scale-flat(F3)·invariant(F1·F4) | 비용 R·on-prem 이탈 | Q4 yes & 위임 3조건 |

### 1.7 ▶ 다음 실험 (2026-07-09 정렬 · C43~C48 반영)
> **오늘 밤 방향 전환**: 날조는 능력이 아니라 *정박 치환*(C43)이고, **출처 선언+provenance 검증이 pass비용 0로 닫는다**
> (C45·날조 67→0%·over-block 0·Δspurious 0). 남은 유일 경계 = **⋈**(C46). learn 축은 *데이터 실패*로 미확립(C38).

1. **★★prov e2e 다중턴 (🔄 실행중·유료승인·make-or-break)** — 단일턴 날조 억제(67→0%·C45)가 **실제 다중턴 pass**로
   이어지는가. arm=floor+prov(`T2_PROV_REGEN=1`·present/autofetch OFF·규칙0 준수) vs floor(0.547·재사용).
   **GO = pass↑ ∧ too_many_errors 폭증 없음**(재발화 예산이 pass를 깎을 수 있음·C38서 SFT가 이걸로 죽음). ⇒ **출처선언 레버의 생사.**
2. **E6′ v3 (learn·데이터 재설계 先)** — C38: cfbsynth가 결손을 *재현 못함*(결손 큐 100% 제공·base 0.98). **D7 필수**:
   근접-오답 id를 창에 배치(C43) + 음성사례(C38 SFT 퇴화) + on-policy rejected(C38 DPO off-policy) + 발명형 rejected(C39).
   **타당성 게이트**(base가 tau2 수준 날조) 통과 전 학습 착수 금지.
3. **ASK 위계 (C48)** — R1(호출가능성 위계 FIND→GET→DISAMBIGUATE→ASK) 검정. retail/airline선 R0≈R1(저빈도) →
   **clarification 벤치(ToolDial·τ²-airline)서만 갈림**. DISAMBIGUATE(⋈·retail 45.8%)가 실질 부담.
4. ~~E10 정책-precondition 게이트~~ → **❌ NO-GO(C50)**: over-action(+9)의 불가능성은 DB-state 아니라 대화(policy+intent)=semantic. DB-게이트 P1 over-block>TP·P2 환경 이미 집행. 남는 후보=대화-precondition controller/ASK(게이트 아님).
5. ~~E1′ Phase A~~ → **하향(C23: DB-기준 성공-write서 F2 −4·payoff 작음).** C13(15pp)과 긴장은 남으나 우선순위 낮음.
6. ~~E9′ free-text provenance~~ → **E11에 흡수(C24·C44).**

## 2. 🔒 불변 규율 (모든 작업에 적용)
- **[[05]]** scaffold 도메인-일반·A2만 변경·게이트 증식 금지 · **[[13]]** scaffold 최소·scale/learn 최후
- **[[08]]** 집계→결론 금지·per-case 포렌식·**실측 前 banking 금지** · **[[03]]#9** 대칭크레딧(이득에 증거 요구하면 피해에도)
- **[[09]]** 무료 검증 先·유료 full-run은 승인+최소scope · **[[10]]** verifier/selector=결정론·LLM=formalize
- **[[06]]** over-action = **게이트 금지 축**(선례) · **[[42]]** prompt-only 레버 무효
- **영속 규칙**: 인용하는 모든 수치는 **doc에 provenance와 함께 영속**. scratchpad-only 수치 인용 금지.

## 3. 증거 원장 (LIVING — 실험 후 반드시 갱신)
등급: **[S]**settled(다-trial·per-case) · **[M]**measured(단일 clean run) · **[P]**promise(isolated/nt=1) · **[D]**design · **[?]**미실행
| # | 주장 | 등급 | 정본 doc |
|---|---|---|---|
| C1 | compliance scale-invariant·게이트만 위반0 | **[S]** | `what_scale_buys` §5.3 · `_WRITING_BRIEF §3` |
| C2 | compliant-pass crossover 14B+scaffold .336 > 32B bare .300 (전 k) | **[S]** | `_WRITING_BRIEF §3` |
| C3a | 애매모호성 = symbolic vs semantic 이분 | **[S]** | `RELWORK_LOAD_COT_2026_07_05` |
| C3b | semantic(⋈) = 경계 — **정보-present 상태서도 실패**(43~52%)·base≈QwQ 불변 | **[S] 부분** | **`E3_E1A_RESULTS_2026_07_08` §E3** (단 agentic ⋈ 실패의 나머지 ~절반은 *탐색* 실패이지 경계 아님 → 경계 슬라이스 축소) |
| C4a | present+g15 합성이 이득(+12.3pp) | **[S]** | `PRESENT_G15_DET_CENSUS_2026_06_25` |
| C4b | thinking: +F2 / −F4·F5 → 순 0 | **[M]** | `QWQ_AGENTIC_FAILURE_FORENSIC §7` |
| C4c | **합성(thinking+게이트)이 순이득으로 전환** | **[D] · Phase A CONDITIONAL GO** | `E3_E1A_RESULTS` §E1A: recall 3/6·**파손 0**(per-case)·closed 판정은 **Phase B** |
| C4d | **게이트 자신이 over-action 역효과** | **[M]** | `QWQ_..._FORENSIC §7c` (QwQ passing-spurious 0 vs base 47) |
| C5 | scale이 사는 유일 축 = horizon | **[S-lit]** | `RELWORK_SCALE_LOAD`(DR#2) |
| C6 | fleet = horizon 전용·저-ROI | **[M]** | `FLEET_FUNCTION_DELEGATION_DESIGN` §4b(rev) |
| C7 | learn-wing이 F3/mis-formalize를 여는가 | **[?]** | 미실행 (E6) |
| C8 | TCO ~23× | **[EST]** | `TCO_TABLE_DESIGN` |
| **C9** | frontier 격차 = **horizon 아님**($p_{step}pprox1.0$·H8+선 우리가 gpt-4.1 초과). ⚠️**"⋈은 frontier와 공유"·"F2가 유일"은 둘 다 철회**(H3-4 부분집합 착시) → **C15가 정본** | **[M]·부분철회** | `HORIZON_GAP_DECOMPOSITION` → **정정 C15** |
| **★C15** | **frontier 격차 전수 분해**(성공-실행 호출 기준·456 sim). vs o4-mini +34(7.5pp): **all-attempts-ERRORED +15** · NL/communication +11 · ⋈ +7 · F2-variant +6 · over-action +5 / **우리가 우세**: never-attempted −11 · payment −6. vs gpt-4.1 +46: **⋈ +21** 최대. ⇒ **F2는 18%에 불과·격차는 최소 5조각**. o4-mini는 *기권*해서, 우리는 *틀리게 행동*해서 실패(A4) | **[M]** | `gapdecomp2` · 본 doc §1.4b |
| **C11** | operand 날조율 우리 5.9% vs gpt-4.1·claude-3.7 **0.0%** = **유효한 진단 표지**. ★**단 레버 아님**: 환경이 32/32 거부·12/15 복구 후에도 실패 ⇒ **근인 아님**(상한 +3.3pp **철회**) | **[M]** | `E9_..._DESIGN` §1·**§4b NO-GO** |
| **C13** | **F2 변형선택에 위치부하 없음** — $p_{traj}$ .762 > $p_{iso}$ .727 · frontier $p_{traj}$ .908~.919 = **15pp 능력격차**. ★단 $p_{iso}$ 프로브가 정보-빈약 → **정보-맞춘 격리(E1′ PhA)가 확정** | **[M]** | `RESEARCH_MASTER §1.4` · `load_measure` |
| **C14** | **부하는 reach/plan-structure에 실재** — 격리 계획선 정답·실제 런선 주문 누락(t99) · 단 도달률 격차는 frontier 대비 **3pp**뿐 | **[M]** | `PLAN_PROBE_PHASE0_VERDICT §1` |
| **C16** | **RBW(읽고쓰기) 격차는 scaffold 아티팩트** — ours+scaffold 21.8% vs **32B floor 95.5%** vs frontier ~100%. present가 주문정보를 주입해 `get_order_details` 호출이 사라짐(우리 sim 66%가 0회·floor 4%). **도구호출 부재 ≠ 정보 부재**. within-arm 상관 반대(안읽고쓴 실패율 33% < 읽고쓴 43%) ⇒ **읽기-부족 인과 기각** | **[M]** | `HARDCORE_STEP_FORENSIC §0` |
| **C17** | **HARD CORE 10 task(양 frontier ≥3/4 ∧ ours ≤1/4) = 8개 서로 다른 근인.** 단일 상류원인 없음. **역방향 0 task**(우리가 두 frontier를 모두 robust하게 이기는 task 없음). 신규 근인 **N1 값 충실도**(약어)·**N2 write-scope**(item 과포함)·**N3 payment 선택**·**N4 도구 거부**·**N5 present의 읽기 억제**[P] | **[M] 소표본** | `HARDCORE_STEP_FORENSIC §1-2` |
| **C18** | **frontier의 우위 이름 = precision**(정확히 그 값·그 범위·그 사실). planning도 reading도 아님. 조건부 NL 실패율 ours 7.3% vs frontier 3.6~4.5% | **[M]** | 같은 doc §3 |
| **★C19** | **채점기준 불일치 발견**: 우리 런 `reward_basis=['DB','NL_ASSERTION']` vs frontier 공식파일 `['DB','COMMUNICATE']`. ⇒ **C15의 "NL/communication +11" 철회**(비교불가). **DB는 공통 기준** | **[M]** | `goldvalid` |
| **★C20** | **gold action_checks는 보상 아님**(약한 프록시): 통과했는데 write-action 불일치 5~7% · 실패했는데 전부 일치 9~22%(gpt-4.1). ⇒ gold-기반 원인표는 **5~10% 노이즈**(gpt-4.1 22%) | **[M]** | 같은 |
| **★C21** | **DB-only 재분해(진짜 기준)**: 실패 형태가 o4-mini와 **거의 동일**(50/25/17/5/4 vs 51/30/17/3/0) = *다른 종류가 아니라 같은 종류를 더 자주*. vs o4-mini DB격차 +23: **operand 정밀도 +10 · over-action(MORE+EXTRA) +9(39%·frontier EXTRA=0) · zero-write +4 · 미완 0(철회)**. vs gpt-4.1 +39: 미완 +17·zero +11·operand +14·과잉 **−6** ⇒ **frontier마다 구성이 다름**(o4-mini는 기권형, gpt-4.1은 실행형) | **[M]** | 같은 |
| **C12** | **decidable ≠ 유용.** decidability는 *부작용 없음*의 필요조건이지 *이득*의 충분조건이 아니다(환경이 이미 집행 중일 수 있음) | **[M]** | 같은 doc §4b |
| **C10** | **레버 부작용은 scope에서 온다** — 전-궤적 thinking=persistence 매도 / 결정점 격리=채널 폐쇄 | **[D]** | 같은 doc §4 · **E1′가 검정** |
| **★C22** | **DB-only hard-core = 7 task**(t17·37·57·63·86·91·111). 기존 reward기준 10 중 **t40·t68은 DB 4/4**(순수 NL=C19 구역)·t105는 2/4 ⇒ 탈락. **역방향 0은 DB 기준으로도 유지.** infra 0·전 arm 456/456 `user_stop` | **[M]** | `DB_ONLY_HARDCORE_FORENSIC_2026_07_08` §0.0·§1 |
| **★C23** | **"operand 정밀도 +10"은 단일 축이 아니다.** SAME 버킷 write쌍 Δ(vs o4-mini): **⋈ +8 · item집합혼합 +6 · 주소 +5 · op +3** / **F2 변형선택 −4 · payment −11**. ★단 성공-write만 세므로 zero-write 24(우리)·20(o4)의 변형오류는 비가시 = **선택편향**. "F2 우세" **banking 금지** | **[M]·편향주의** | 같은 doc §3·§5 |
| **★C24** | **free-text 날조는 환경이 못 잡는다.** 문맥에 없던 `address1` write: **ours 5 / o4-mini 0 / gpt-4.1 0**, 5/5 db_fail, **passing-spurious 0**. t17은 조회 0회 상태서 주소 전체 생성(4/4 동일). ⇒ **C11/C12 정련**: id 날조는 환경이 32/32 거부하나 free-text는 타입상 불가 ⇒ **E9 NO-GO는 free-text로 확장되지 않음**. 상한 5 sim=1.1pp | **[M]** | 같은 doc §7 |
| **★C25** | **over-action(+9)의 실체 = 정책-precondition 위반.** 12건 per-case: 정책-불가능/철회 요청 수행 **8** · ⋈·대상불일치 2 · degenerate write 1(`item_ids:[]`) · **순수 unrequested scope 1**(t111). ⇒ [[06]] "over-action 게이트 금지" 선례가 겨눈 대상(scope residual)과 **다른 술어** | **[M]** | 같은 doc §6 |
| **★C26** | **tau2-bench 버그**: `retail/tools.py:531-537` `variant` 루프 누수 → 수정된 모든 item이 *마지막* 변형의 price/options 획득 ⇒ **db_hash가 `item_ids` 나열 순서에 의존**. 의미동일 write가 fail. 실측 피해 ours 1 / frontier 0 (**격차 설명력 없음**) | **[M]** | 같은 doc §4 |
| **★C27** | **`§1.7 #4` repair 최우선 근거 붕괴**(자기교정 #10). "all-errored 16 vs 1=+15"는 o4-mini의 `never-attempted 19`를 제외한 프레이밍. **공통 버킷(성공 write 0) = 24 vs 20 = +4.** DB격차 +23의 최대 조각은 **over-action +9**와 **⋈ +8** | **[M]** | 같은 doc §9 |
| **★C29** | **★gather-before-act = DB격차의 최대 축.** `new_item_id` 중 **변형목록 미조회 상태의 날조**: ours **63/439(14.4%)** vs o4-mini 1/341 vs gpt-4.1 **0/416**. reads/sim 3.59 vs 5.32/5.92. sim 단위 **28/456**(db_fail **19**·db_pass 9) ⇒ **상한 4.2pp = o4-mini 격차(5.0pp)의 83%**. ★차단은 이미 환경이 함(날조 93/93 ERR·C12) ⇒ 게이트는 **공급(repair)** 이어야 | **[M]** | `DB_ONLY_HARDCORE_FORENSIC` §10.3-10.4 |
| **★C30** | **프롬프트 천장 실증(우리 데이터)**: `retail/policy.md:18`이 이미 *"do not make up information"* 을 명령하는데 순수 날조 ours **91** vs o4-mini 1 vs gpt-4.1 3. ⇒ **[[42]] scale-emergent prior-override** · prompt-only 레버 무효 확정 | **[M]** | 같은 doc §10.1 |
| **★C48** | **★ASK는 금지가 아니라 최후위 위계**(자기교정 #19: C44 검증기 `producer 존재→ASK 금지`는 too strong). 교정 = **호출가능성 위계** FIND → GET(producer가 *지금 호출 가능*) → DISAMBIGUATE(후보 열거) → ASK. **실측: R0와 R1이 두 벤치서 거의 같은 답** — 조회가능 인자(id·producer 有)와 물어야 할 인자(선호·producer 無)가 분리돼 R0가 우연히 안전. R0 정당ASK 차단: retail 32B **0**·14B **7**(t48 auth실패)·airline **0**(cabin/flight_type=producer없음). 원리결함 실재하나 저빈도 → **clarification 벤치(ToolDial)서만 갈림**. ★**더 큰 발견: DISAMBIGUATE 모집단=retail write인자 45.8%**(후보2+·전부 gold=FIND)=C46 ⋈이 사는 곳=**날조 닫은 뒤 유일 잔여** | **[M]** | `C49_ASK_HIERARCHY_NOT_BAN_2026_07_09` |
| **★C45+선행** | **★출처선언 레버 선행 지형 확정**(딥리서치 `wb07r5hi7`·3-vote): **4지선다 완전체·INFER 분기·"producer 있으면 ASK 금지" 규칙 = 정식화한 선행 0 = 신규**. 지지 확립사실: 날조=default(When2Call) · 결정이 **scale로 안 열림**(tau-bench pass^8<25%·retail) · over-asking 실증(단 모델의존). ★**명시 ask-프롬프트가 ask정확도 0.52→0.90 but 최종 call 0.48→0.58만**(Learning-to-Ask)=**우리 D″가 날조 닫되 ⋈ 못 닫음(C46)과 정확히 정합**. ★"tool-field 매칭 병목"(C40)은 선행 미지지→**자체 ablation 필요** | **[M]+[S-lit]** | `C44_..._2026_07_09` §5b |
| **★C45** | **★출처 선언 레버가 날조를 닫는다** (C44-C48·32B·무료). 결정점서 operand 출처를 4지선다{GET·FIND·INFER·ASK}로 **선언 강제** + 갈래별 결정론 검증기(FIND=문맥실재·GET=producer매칭·**ASK=producer없어야**) + 재발화 + GET폴백. **날조 67% → 0%** (DB주입0·도구대신호출0·학습0). **over-block=0/2650**(전수·tau2 retail은 write인자 ASK 0.0%). **Δspurious=0**(파손3건 전부 원본도 틀린 ⋈지점·자기교정#18). ⇒ **GO 3조건 충족.** [[05]]: A2=`{인자→producer}` 매핑뿐·present/autofetch와 달리 DB 안읽음 | **[M] 소표본(n=60)** | `C44_SOURCE_DECLARATION_LEVER_2026_07_09` |
| **★C46** | **★provenance 검증은 *날조*를 닫고 *⋈ 오선택*을 격리한다** — 둘은 다른 실패. 날조(정박치환·engineering)=67→0% / FIND-wrong(문맥의 틀린 값 선택)=안닫힘(3/30·FAB·CLEAN 공통). 후자는 **문맥에 후보 2+개일 때** 발생 = **F3 ⋈ 경계(C3b)**. ⇒ "operand 정밀도"를 날조 vs ⋈로 분리(**C23 정련**)·남은 잔여는 오직 ⋈(레버 아님) | **[M]** | 같은 doc §4 |
| **★C47** | **예시 태그+금지문은 무효**(C45 arm C): `<EXAMPLE>` 씌우고 "쓰지마라" 지시해도 예시복사 9→8. **예시 제거는** 9→0(arm B) but 원천①(47%)만 없앰·근접변형(②·과제내재)은 남음. **4지선다+검증(D″)가 ①②③ 전부를 검증가능한 주장으로 변환** | **[M]** | 같은 doc §1 |
| **★C43+선행** | **★선행연구가 C43을 지지**(딥리서치 `wy3wbu6o9`·3-vote): 정박치환 기전=**contextual entrainment**(문맥 아무 토큰이나 logit↑·관련성 독립·`2505.09338`)+induction head(`2209.11895`). **WM 가설=반증 foil**(`2506.08184`은 문맥↑서 오류↑ 예측=우리와 정반대). distractor 증거 지지(Context Rot·present 2배와 일치). **off-policy DPO 실패=likelihood displacement**(valid vs fab id=CHES 高·`2410.08847`)=C38 정확히 설명. ★**scale 형태변화·provenance 검증기 tool-arg 효과=선행 없음=원본/whitespace**(C45가 점유) | **[M]+[S-lit]** | `C43_..._2026_07_09` §6 |
| **★C43** | **★날조 = WM 문제가 아니라 *정박 치환*(anchored substitution).** 전수 4-arm: (a) **H-forget ≈ 0**(32B 71건 중 **1**·읽은 걸 잊지 않음) (b) **외생 부하와 역상관**(사용자 턴 Q1 9.5% → Q4 5.4% · 날조 시점 문맥 median 6,355자 < 정상 9,421자) (c) distractor 밀도는 read 층화 후 무효 (d) **지배 변수 = "아직 안 읽음"**(read 0–3 27.0% → 6–7 **1.0%**) (e) **기전: 32B 미조회 날조 new_item_id의 70%가 문맥 내 기존 id의 edit≤2 변형**(`4127323219→4127323220`). 14B는 placeholder 70%(scale이 양식을 바꿈·C36 정련). **present가 id를 뿌려 재료를 공급 ⇒ 날조 30→64**(C34/C35의 기전). ⇒ 외부 메모장·짧은 문맥은 처방이 아니다(E12 short−full = **+0.03**). 처방 = **창에 근접-오답 금지 · provenance 검증 · gather 강제** | **[M]** | `C43_ANCHORED_SUBSTITUTION_NOT_WM_2026_07_09` |
| **C42** | 짧은 합성서 base 7B가 4지선다(GET/FIND/INFER/ASK)를 **완벽히** 푼다(1.00/1.00/0.95/1.00·fabricate **0.00**) ⇒ cfbsynth v1·v2 **타당성 게이트 FAIL**(학습 gradient 0). ~~"⇒ 결손은 긴 문맥 = load"~~ → **❌ 철회(C43)**: 합성엔 **정박할 id가 하나도 없어서** 날조가 안 나온 것. **D7 필수**: 근접-오답 id를 창에 배치해야 실패가 재현된다 | **[M]·부분철회** | `C42_SHORT_CONTEXT_SOLVES_FOURWAY_2026_07_09` |
| **C41** | 짧은 문맥선 **시스템 프롬프트가 작동**(2×2: 규칙有·큐無 gather **0.87** vs 규칙無·큐無 **0.20**). tau2선 같은 규칙(`policy.md:18`) 무효(C30). 규칙 없으면 모델은 날조가 아니라 **묻는다**(ask 0.65) — tau2선 escape-ask 0/15 | **[M]** | 같은 doc §1 |
| **★C34** | **★규칙 0 위반 scaffold 색출**: `candidate_summary`(`T2_PRESENT_READS`)는 에이전트가 부르지 않은 `get_order_details`를 **엔진이 주문마다 대신 호출**해 주입 · `_autofetch_text`(`T2_AUTOFETCH`)는 provenance-deny 시 **실 레코드를 먹임**. 둘 다 *"DB 내용은 도구로만"* 을 에이전트 대신 우회 ⇒ **폐기**. `nested`·`calc`·gate-kinds·`REGEN_FEEDBACK`은 위반 아님(에이전트 자신이 가져온 내용 위에서 동작) | **[M]** (코드 정독) | `SCAFFOLD_AUDIT_RULE0_2026_07_08` §1 |
| **★C35** | **★supply 꼼수는 gather를 가르치지 않고 induction head를 *이용*한다** — 더 가까운 **복사 대상**을 문맥에 놓을 뿐. 7B retail: `no_gather` base 36 → **deny 36(무변화)** → **autofetch 24**. 프롬프트 전부 무효(fetchfirst 23 = base 23). ⇒ **present/autofetch가 켜진 채로는 gather를 측정조차 못 한다** | **[M]** | 같은 doc §2b · `c4_dpo_eval_retail.log` |
| **★C36** | **scale은 날조의 *형태*를 바꾼다**: 7B는 날조 36/36이 **스키마-예시 복사**. 32B는 93건 중 예시복사 18 · 그럴듯한 10자리 id **발명 48** · 조합형 placeholder 16. **복사 → 발명**. 미조회 날조율 7B 38.8% → 14B 7.0% → 32B 6.7%(floor) → frontier **0.0%**(불연속) | **[M]** | 같은 doc §2b · `DB_ONLY..._FORENSIC §10.3` |
| ~~C37~~ | ~~"gather 학습은 시도된 적이 없다"~~ → **❌ 철회(자기교정 #14)**. `cfbsynth_dpo_pairs.py`에 **`gather` 페어 존재**(prompt=id없음 → chosen=getter 호출 / rejected=예시값 consumer) **+ `copy` 페어**(값 있으면 쓰기) = 조건부 양쪽 모두 학습됨 | — | `cfbsynth_dpo_pairs.py:9-10` |
| **★C37′** | **★"gather 학습 NO-GO"는 미확립.** (자기교정 #15: 나도 처음엔 "역전이"로 단정) **C4의 pass^1은 해석 불가** — *동일* 7B base가 두 런서 **21 vs 32/114**(nt=1·gpt-4.1 user-sim 편차)로 학습 효과 전체를 삼킨다. 기전 *비율*로 보면: **DPO는 자기가 벌준 것을 눌렀다** `schema_copy` **.439→.376**·`no_gather` .439→.398·도구호출 무손상(tme 12→13). **SFT는 진짜 해로움**(A_notfound .31→.41·`learn-pure` .49·tme 13→**25**). autofetch(꼼수) `no_gather` **.286**. ⇒ **DPO 경로는 살아 있고, 표본이 부족했을 뿐** | **[M]** | `c4_dpo_eval_retail.log` · `c4_ff_eval_retail.log` · 종료사유 전수 |
| **★C39** | **★DPO는 `rejected`에 넣은 것만 배운다.** 벌점 대상 = 스키마-예시값 ⇒ `schema_copy`만 −6.3pp 감소. **그런데 32B의 실패 양식은 예시복사가 아니라 *발명*** (93건 중 예시복사 18 · 발명 48 · 조합형 16·C36). ⇒ **7B용 rejected 집합은 32B의 실패를 벌하지 않는다.** E6′는 rejected를 {예시값, 발명형 id, 조합형 placeholder}로 확장해야 | **[M]** | `E6PRIME_GATHER_LEARN_DESIGN §2 T3` |
| **★C38** | ✅ **측정 완료** → **"학습 NO-GO"는 *데이터*의 실패였다.** held-out 합성(seed 7): **base** gather 규칙有 **0.98** / 규칙除 **0.40** · **sft** 규칙除 **1.00**(완벽 학습) · **dpo** 규칙除 **0.33**(base보다 낮음) + copy **0.77→0.63**(피해만). ⇒ (a) cfbsynth 사용자 발화가 *"I don't have the id"* 큐를 **150/150(100%)** 제공(tau2는 120 sim 중 **1건**) + 시스템에 규칙 명시 ⇒ **base가 이미 0.98 = 학습 여지 없음** (b) SFT 궤적 **2000/2000이 lookup으로 시작**·직접-act **0** ⇒ **무조건 조회(퇴화 정책)** 학습 → tau2서 `too_many_errors` 13→**25** (c) DPO chosen/rejected 둘 다 **off-policy**(모델은 산문을 냄) ⇒ 마진이 지지집합 밖. **learn 축은 진짜 결손 위에서 시험된 적이 없다** | **[M]** | `C38_INDIST_GATHER_RESULT_2026_07_09` |
| **★C40** | **E11-a(정보주입 0 provenance 피드백)는 32B서 작동한다** — 격리 단일턴·짝지음·**충실도 50/50(ARM env ≡ ORIG)**. 다음 행동이 getter: env 8/50(16%) → **e11a 28/50(56%)**, 살림 22·망침 2. ★단 **getter ≠ 올바른 getter**: 정합 판정서 23건 중 12건만 올바른 원천. **주소는 getter가 오답**(gold 주소는 사용자 발화에 있어야 함) · E11 주소 발화 21건 중 **수리 가능 7건**(나머지는 user-sim이 주소 미제공 5 · gold에 그 write 부재 7) | **[M]** | `e11a_isolated_probe.py` · `E11_..._DESIGN §6` |
| **★C31** | **present는 학습 신호를 파괴한다**: 주입이 궤적에서 `read→act`를 지움(reads 3.59 vs frontier 5.9). ⇒ learn-wing(P4·C7)이 배워야 할 감독 신호가 사라짐. **주입보다 강제 조회가 학습-정합적** | **[D]** | 같은 doc §10.5 |
| **★C32** | **E1 Phase B는 게이트 이득을 보이지 않는다(짝맞춤 후)**: arm-off 10 중 **2 infra**(t12·t68) ⇒ 공통 8 task 짝비교 **off .625 = on .625, Δ=0.000**(flip: t46 +1 / **t57 −1**). 비-짝 `.625→.667`은 **task 구성 아티팩트** | **[M] 소표본** | `e1pb_{off,on}/results.json` |
| **★C33** | **QwQ arm은 비대칭 결측을 만든다**: QwQ가 `finish_reason=length`로 `content=None ∧ tool_calls=[]`(reasoning만) → tau2 `AssistantMessage` 검증 실패 → 3회 재시도 → `infrastructure_error`. Step3 **8/359(2.2%)**(task 1·16·27·113) vs base arm **infra 0**. ⇒ "생각하다 행동 못 함"을 인프라로 버림 = **thinking에 유리한 표본 삭감**. C4b/E2는 **infra-as-failure 민감도분석 없이는 [S] 승격 불가** | **[M]** | 같은 · `reexp_qwq_rparser_nt4.log` |
| **★C28** | **reason enum 불일치 = arm 공통 하네스 노이즈**(ours 8/93 · o4 5/84 · g41 4/97). 절반은 **user-sim이 gold와 모순**(t38 사용자 *"Ordered by mistake"* ↔ gold `no longer needed`; o4-mini는 5/5가 이 경우). **레버 아님** | **[M]** | 같은 doc §8 |
| **★C54** | **★능력 재조합(BC0-7)·G→BC는 도메인별·전이가설 진단지지** — G1-9는 관찰층·**BC(원인+구제)는 처방층**·§1.5 라우팅 명시화. **G→BC split은 도메인별 per-case 필수**(Phase 0): 같은 **G6 OPERAND가 retail=BC4(변형매칭·의미) vs airline=BC0/scope(baggage 필드의미·과삽입)**. ★**정정(자기교정·per-case 전수)**: airline baggage는 처음 휴리스틱이 "BC3 계산"이라 했으나 **전수 스텝정독서 gold 9/9=0**(사용자 짐 안부침·모델이 허용량/기본값 삽입)=계산 아니라 필드의미+scope. **numeric-arg 휴리스틱이 오분류→per-case 필수성 강화**([[08]] 실증). telecom G4=**BC6 조기포기**(fix 미시도 17/30·per-case) · banking G2=**BC2 부하**(KB검색함 119/125·조립실패·능력 아님). ★**3도메인 지배 splitter가 전부 도메인-일반 결정론 구제(calc·persistence게이트·controller)로 라우팅** = A2 전이가설을 *진단층*서 지지(처방 실측=Phase 3). **status·계획 정본**=`DOMAIN_TRANSFER_STATUS_AND_PLAN_2026_07_09`·`phase0_bc_split.py`·`fine/aggregate` | **[M]진단·[D]처방** | 같은 doc §3.4 |
| **★C53** | **★출처선언 레버 다중턴 e2e = GO(make-or-break 통과)** — floor+prov(`T2_PROV_REGEN=1`·present OFF·gpt-4.1 user-sim·nt=4·443/456). **reward 0.580 > floor 0.547(+3.3pp)·db_pass 0.624·tme 1/443(0.2%)**=C38 재발화-예산 폭증 우려 미발현·infra 0. **per-case([[08]])**: t17(정본 날조 태스크·floor 4/4 db_fail) → prov **4/4 db_pass**(날조 교정 실증)·t39 미교정(일부 잔존). ⇒ **단일턴 날조 억제(67→0·C45)가 다중턴 pass로 전이.** ★**456/456 확정(2026-07-10)**: reward **0.5768(+3.0pp)**·db_pass 0.621(282/454)·user_stop 454·tme 1·max_steps 1·t17 4/4 유지·t39 0/4 유지·**GO 불변**·gz persist 완료. ★**보강(정본지표·같은 날)**: 공식 pass^k 0.577/0.420/0.331/**0.281** — **p1을 사고 p4를 판다**(짝 114task: floor p4 0.333→0.281 −5.3pp·robust 4/4 상실 16 vs 획득 10·12건은 3/4로=분산·**t61 4/4→0/4 체계파손**=payment 원결제→gift_card 전 trial·사용자 미지정=원리디폴트 위반 방향·기전 [P] regen-유발 재선택). compliant-full p4 0.237(g2=37 잔존·게이트 없음). ⇒ 제1원리 실증 추가·E-COMP 판정에 p4 회귀 1급 축 승격 | **[M] 456/456 확정·p4 비용 [M]** | `C44 §4.2` · **`RETAIL_PASS_COMPOSITION_DESIGN §3c`** · `sim_results/prov_e2e_retail_t4.results.json.gz` | 
| **★C52** | **★교차-도메인 기능 불변성 = 도메인-일반 규칙 실증**(retail vs banking_knowledge·top frontier·gpt-5.2 sim·per-case 2건 검증). **도메인-불변 실패기능 = {F1 verify · reach/절차조립 · F4 coverage}** 양 도메인 재현 → 결정론 scaffold(게이트+controller) 타깃·**TBox 도메인-일반 규칙 후보**(같은 추상규칙·ABox만 교체). **도메인-의존 *능력* stress는 다름**: retail=F2 operand(→thinking·operand argdiff 지배) / banking=**reach(MISS_P_reach 24~48%)+coverage(MISS write 20~26%)+F6 horizon(gold 절차 median 8단계)**·operand argdiff **0**. ⇒ 단일 "그 잔여" 없음·**scaffold 기능은 불변·binding 능력축만 도메인마다 다름**. banking이 retail 못 본 F6/reach 실측 → F1-F6 프레임 도메인-일반 교차검증. ★**fine 9-기능 taxonomy(§3.2d·4도메인·reward통일기준·telecom=ENV_ASSERTION)**: G1 COVERAGE(전 도메인 40~57%·**최강 불변**)·G2 REACH·G3 VERIFY·G4 PERSISTENCE·G5 SCOPE = **도메인-불변→도메인-일반 결정론 구제(게이트/controller)·ABox만 교체** / G6 OPERAND(thinking·retail27%)·G7 REFERENCE(경계)·G8 HORIZON(banking 9단계·scale) = 능력축·stress 도메인의존 / G9 GUIDANCE=telecom 특이. **구제방법 일반화 지도.** **특허 "도메인-일반 게이트·계획정책" 실측 지지** | **[M]·per-case** | `TAU2_FRONTIER_..._MASTER §3.2c·§3.2d` · `fine_function_decomp.py`·`opus45/gpt52/gpt55 × 4도메인` |
| **★C51** | **frontier 자신의 잔여 = F2 변형(F3 아님)·신형까지 측정·per-case 검증** — per-arm db_fail 구성(456 sim·db_match). **① baseline(gpt-4.1 sim)**: F2변형 평탄(ours 3.9·o4 4.4·g41 4.2·claude-3.7 3.9=scale-불변)·F3⋈ 감소(3.1→3.1→1.5→**0.9**=scale가 삼). **② 신형 top 8(gpt-5.2 sim·S3 재다운로드)**: 거의 전 모델 F2변형이 최대 잔여클래스(F2 9~16건)·F3⋈ 전부 <1.8%. **③ Qwen3.5-397B-think(챔피언 0.855)만 F2=0.4%** = F2 symbolic은 **thinking이 산다**(C13·명제 확증). **④ per-case 정독(Opus4.5 5건)**: exec new_item_id가 gold와 함께 **둘 다 문맥 카탈로그에 실재** = 진짜 변형-오선택(날조 C9·leak C26 아님). ⇒ **C3b("F3 scale-flat 경계")는 구성된 격리프로브·agentic선 작음**·경계=order-⋈ 아니라 fine 속성매칭(F2 변형). reason_enum=노이즈(C28)·payment=우리우세(C23). **표류종식**=[[47]] | **[M]·per-case 검증** | `TAU2_FRONTIER_..._MASTER_2026_07_09` §3.2·**§3.2b** · `frontier_function_decomp.py`·`f3_probe1/2` |
| **★C60** | **★T5-B 라우터 e2e 완주 = 조건부 GO — DISAMB는 "robust-축 환매 레버"**(routerv1=floor+prov+T2_DISAMB·456/456·infra 0·**tme 0**·공식채점·gz 영속). pass^k **.570/.434/.357/.298** vs prov .577/.420/.331/.281: **p1 −0.7(잡음권·사전 상한 +2.9pp 미달 예상대로)·p2 +1.5·p3 +2.6·★p4 +1.8pp = prov의 robust 매도(−5.3pp) 부분 환매**(완전 환매 아님: floor p4 −3.5pp 잔존). DISAMB 발화 1,274(2.8/sim)·switched 26(2%). **flip(vs prov): 상승 8**(★**t61 0/4→4/4 완전 복구**·체계핵 t16/t18 2→4·t17 유지) vs 하락 6(t46 4→0·t47/t95 4→1). **per-case(t46)**: switch-오답 아니라 **무-write 탈선**(write 호출 소멸·기전 [P]·t47/t95 후속 정독 필요). ⇒ ① 간섭-보상 합성(B09 5-1)의 실증: prov(+p1/−p4) ⊕ DISAMB(+p4/−p1미세) — 미완 상쇄 ② **"격리 이득≠e2e 이득" 정량 2호**(c51 +31pp→e2e p1 0·이득은 robust 축 = 열거는 정보를 안 늘리고 **해독 분산을 줄인다**·DPI 정합) ③ 완전 상쇄 조합 탐색=E-COMP 합류 | **[M] 456/456·per-case 부분** | `E_AMB_..._PLAN §7h` · `sim_results/routerv1_retail_t4.results.json.gz` |
| **★C61** | **★격리→e2e 희석의 per-step 전수 포렌식 = 회계 닫힘**(eamb7·3-arm×456·step-이벤트+짝 교차표). **① prov 희석 = H-D 지배**: floor 날조 70/70 **전부 env-차단**(수락 0)·29% 자연회복 — 격리 67→0%는 "환경 없는 세계"의 조건부 확률·e2e 부가가치=에러-루프 미회복분 회복뿐. 표적국소 이득 실재(표적 21task +7.1pp·표적실패 25 중 13 pass 전환)·상쇄(살림83−죽임74=+9시행=+2.0pp). **② DISAMB e2e-0의 정체 = "+27 살리고 −37 부숨"**: 표적(⋈오선택-실패 74) 중 27 pass 전환(robust 상승의 원천)·33 재확인-무효(**진짜 경계 7.2%**·c51 잔여 .34 일치) — 부순 37 = **write-소멸 39건·기전 코드-확정**: DISAMB 분기가 tool_calls 없는 재확인 응답을 무조건 수락→write 유실(**경계 아니라 구현 결함**). 수정 시 상한 ≈+27시행 → **T5-C(수정 재런) 후보**. ③ 교훈 일반화: **격리 이득 = P(교정\|표적) — e2e 이득 = P(표적)×P(교정)×P(타결함 없음)−부작용**·env-중복 검사([[C12]])는 격리 설계의 필수 전처리 | **[M]·456×3·per-case 부분** | `E_AMB_..._PLAN §7i` · `eamb7_dilution_census.py` |
| **★C55** | **★애매모호성 이론 T1·T2 = 카디널리티 단조성 실증**(무료 재분석·fl32b floor 456 sim·infra 0). **T1**: 결정론 후보 열거기 유효(gold∉C 3.7%<10%)·**전수 write-인자 census $\|C\|\ge2$ = 75.5%**. ⚠️**C48의 45.8%는 repo에 산정 스크립트·로그 부재(provenance 미비)** — 모집단 규약 변형 전부 71~78%로 재현 불가·c50 `candidate_count`(order_id=0 처리·모집단=날조지점)일 개연성 → 리모트 검증 항목·**"45.8%" 인용 시 모집단 명시 필수**. **T2**: 실패율 $\|C\|$ 단조 — **env-수락∧gold∈C 정본 슬라이스: $\|C\|$=1 실패 0/351 → 2: .047 → 3+: .093 (CA z=6.17)**·**4 trial 각각 독립 재현**($\|C\|$=1 전 trial 0·전 trial 단조)·within-task MH OR 2.27·공변량 역방향·per-case 2궤적 정독(t11=정책-불가 시도 오염 발견→env-수락 한정 확정·t16=진짜 order-⋈). **★발견: 1차 분할=gold∈X(측정 축·gold∉C면 70~100% $\|C\|$-무관)·2차=$\|C\|$ 단조(선택 축)** = 이론 형식화/참조 분리의 데이터 실증. P2b/c(prov arm)=리모트 보류 | **[M]·per-case** | `E_AMB_MEASUREMENT_PLAN_2026_07_10 §7` · `eamb1_census.py` |
| **★C56** | **★T3: 동-scale thinking은 $\|C\|\ge2$ 선택을 못 산다 + 체계성 2성분**(base vs QwQ 32B·456 sim each·infra 0). ① **T2 제2-arm 재현**: QwQ 정본 슬라이스 0/293→.071→.106 = 단조성 arm-불변 ② **E-AMB-3 예측(a) 반증**: 변형선택 실패 base .145 = QwQ **.143(동일)**·item .047→.107 악화 — C51③ 챔피언 효과는 397B+think로 **scale과 분리 불가**·동-scale에선 F2b(CoT flat)와 정합 ⇒ 기준-형식화형 레버 = thinking 아니라 **formalize→결정론 직렬화** ③ thinking은 측정 축(gold∉C 4.7→6.6%)도 무개선 ④ **체계성 2성분**: $\|C\|\ge2$ 실패점 51 = 부분실패 45(경로·user-sim 변동 지배) + **전-trial 동일-오답 체계핵 6**(t8 변형 4/4·t71 주문 4/4·t82 결제 3/3 — 대화가 달라도 같은 오답). **체계핵 정독(t71 전문·순환매칭 감사 통과)**: 순수 ⋈(t8·t82)와 **형식화-가능 기준 오적용**(t71 "최근 주문"=argmax날짜·4/4 오적용·user-sim 오확인 고착·나머지 선택 전부 정답)의 **혼합** ⇒ 표적 레버 = DISAMBIGUATE + **calc 직렬화**. "분산 0 체계성"(§3c)은 고정-문맥·체계핵 스코프로 한정 | **[M]·per-case** | `E_AMB_MEASUREMENT_PLAN_2026_07_10 §7b` |
| **★C57** | **★T4 앙상블-불일치 + P2b/c 완결 + gold-버전 불일치 발견**(리모트·2026-07-10). ① **P2b/c**: prov arm서 payment $\|C\|$=1 **0/319**(base 17/376)=provenance가 placeholder-칸을 닫음 · $\|C\|\ge2$ 잔존(payment .22 불변)=⋈ 못 닫음(C46 재확인) · 단조성 **제3-arm 재현**(z=4.28·MH OR 5.43) ② **T4**(해독기 8종·502 gold-일치 슬롯): frontier-공통실패 7개가 고-불일치 집중(H 0.95 vs 0.39·2.8% vs 1.0%)=P4a 지지[소표본] · **scatter(미결정형) 5 > same-wrong(공유-prior) 2**=반증조건 미발동 · scatter 최상위=t20(우리 체계핵과 동일)=미결정 섹터는 실재하되 **좁다**(1.4%) ③ ⚠️**하네스 간 gold-버전 불일치 발견**(t18·t91·t107 — 공식 frontier파일과 우리 하네스의 action_checks가 다름·t107 정독으로 발견) = **C19 계열 신규**: frontier 대조 채점 시 반드시 각자-gold로. 신형 17모델 앙상블은 raw 소실로 제외([[47]]) | **[M]·per-case** | `E_AMB_..._PLAN §7c·§7d` · `eamb4_ensemble.py` |
| **★C58** | **★T6 slot-filling + 디폴트 불변성 기각**(fl32b·counterfactual·무료). ① slot-대조기: order/payment FILL .94~1.0·ENUM gold-포함 1.0 — **variant만 .667(반증조건 발동)** = 패러프레이즈 잔여 실측(무증거 구간 28%) ② **★T6h: 디폴트 불변성 기각** — 같은 오버라이드 기계 위에서 rand .065 vs **통계(LOTO) .185(+12pp)** vs 원리(기존유지) **.009**(default-잔여 108pts). ⇒ (a) **통계 디폴트 = 벤치-분포의 숨은 일 = 전이불가·트릭 비판 정당**(쓰려면 ablation 분리 필수) (b) **★디폴트 개념 폐기(T6h-2 층화)**: payment .940은 혼합물 — **정책-강제 슬롯(return·cancel) 164/164=1.000**(=정책-도출 결정론 규칙·트릭 아님) vs **사용자-선택 슬롯(exchange 차액) .840**(=분포 base-rate·디폴트 깔면 트릭). 남기는 것 = 정책-도출 규칙(provenance=policy.md)/단언 OVERRIDE(문맥-실재 검증)/ASK 셋뿐 (c) 잔여의 주인 = LLM 의미해석({KEEP/OVERRIDE/ASK} 분류) 또는 ASK ③ T6e: 독립근사 오차 0.67비트(얽힘 보정 중간·닫힌세계=정확셈) | **[M]** | `E_AMB_..._PLAN §7f` · `eamb6_slotfill.py` |
| **★C59** | **★T5-A: 후보 열거가 ⋈를 연다 — C46 "열거 무효" 대규모 기각**(c51·n=400·32B·격리). full-자유생성 .345 → **열거 .657(+31pp)**·new_item .116→.658·item .079→.545·ncand5+ .191→.640 / payment·order·address는 불변(이미 높음). 잔여 ~.34=진짜 의미/미결정. T3와 정합: **연산(thinking)은 못 열고 구조(명시 열거)는 연다**. T5-B DISAMB arm의 Phase A **GO**. caveat: 격리 프로토콜·절대수준≠e2e | **[M] 격리** | `E_AMB_..._PLAN §7g` · `c51_results.jsonl`(scratch→영속 필요) |
| **★C61** | **★E-ISO 정보-맞춘 3단 격리 replay 완료(72 결정점·무료)** — semantic 잔여의 3분해. **①오염/경로 몫 ≈20%**(오염-전 전체문맥 A가 정답인데 실런 실패: REF 32%·ITEMS 10%·PAY 12% — t61형 고착 포함) → CP1/CP2/pin 좌석 실증. **②형식화-부하 실재**: ITEMS A .10→B .23→**C(열거) .44**·PAYMENT .12→.12→**.38** — 단 PAYMENT는 열거+정책 포함에도 .38 = **원리-디폴트는 프롬프트로 안 닫힘→결정론 P2**(C58 KEEP .940 정합). **③order-⋈는 열거가 역효과**(A=B .32·**C .21** — id-형 열거는 판별정보 없음·C59의 열거-GO는 내용-매칭 열거였음=열거 설계 의존) ⇒ **⋈=능력/경계 지배 재확인**(C3b/C51 정합·오염-전에도 68% 오답). **④PREINFO 6**(gold 미조회 시점 결정=gather-선행)→E-PLAN. 격리 서브콜(B−A) 좌석=ITEMS만 +13pp 소폭. caveat: C 판정가능 n 감소(ITEMS 18/30)·프로브≠e2e·per-case 정독 6건(t61/71/82/109/110/2) | **[M]** | `eiso_full.jsonl`·`ecomp_iso_probe.py`·`SCAFFOLD_ENDGAME §L0` |
| **★C60** | **★"user-sim 노이즈" 재해부 — flip의 사인은 agent paraphrase-brittleness** (prov-lost 15태스크·pass/fail 17쌍 per-step 전수). ① 최초 분기 주체 **17/17=user 발화**(assistant 동일-prefix 결정론적=vLLM/seed 기각) ② **사실-토큰 동일 5/17서도 결과 반전**("ordered→bought"에 EXTRA return 3발 등)·diff 12/17도 상당수 계시-순서(agent 질문이 상류) ③ gpt-4.1은 같은 표현-분산서 12/15를 4/4 유지 ⇒ **"flip 95%=user-sim seed"(2026-07-07)는 방아쇠 귀속이지 사인 귀속 아님** — flaky 질량=노이즈 천장이 아니라 **주소가능 표면**(원리디폴트·DISAMB·plan-walk·[[12]] 다양성 축). robust 격차(gpt41 4/4 64 vs floor 42)의 주성분=같은 태스크의 불안정 해결(§3d) | **[M]·per-case** | `RETAIL_PASS_COMPOSITION_DESIGN §3d·§3f` · `ecomp_divergence_census.py` |
| **★C62** | **★"regen 손상 = replay-채점 아티팩트" 기각 + 손상 episode-분해 + "열거는 무해" 지지**(무료·3-arm 456×3·2026-07-11). ① 채점=커밋 히스토리 replay(`evaluator_env.py` set_state·mutating만 재실행·기록응답 비교)+DB hash+NL judge — 개입 마커 히스토리 오염 **0/33,730 msgs**(routerv1+prov 전수)·infra 0 ⇒ **손상은 실행동**(write가 live서 파기됨). ② episode-분해: no-write 실패 floor 12→prov **22(+10)**→router 25·write 총량 860→833→809 단조감소 — prov 대화-교란 실재. 단 **[P]·기전 미확정**: 정독 3건(t92/69/40)은 루프-삼킴 아니라 escalation/오결론 형상=대화-발산 하류. ③ DISAMB 손상 지배형=WRONG_WRITE(부분 소멸 포함)·switch 26/1,274 ⇒ **열거 *정보*는 무해·*전달 기전*(턴-파기 재생성)이 유해** — silent repair(제자리 치환·격리 서브콜·대화 불변·replay-clean) 설계 근거. 부수: db_match=True∧reward=0(NL축) router 11<prov 19. ④ **T5-C fix #1 적용**(커밋 `07337a3`·양 분기·banking arm 포함 발사) | **[M]·②기전은 [P]** | `T5C_SILENT_REPAIR_DESIGN_2026_07_11` |
| **★C62** | **★E-COMP arm1(COMP) = 합성 GO + p4 회복 실증** (게이트6종+prov+nested/calc 단일 unified·456 sim·infra 0·짝 114task). **reward pass^1..4 = 0.634/0.480/0.382/0.316** vs prov 0.577/../0.281 vs floor 0.557/../0.333. **① 합성 GO**: COMP p1 prov 대비 **+5.7pp**(db +4.0)·floor +7.7pp. **② ★p4 회복**(설계 1급축): prov가 floor 대비 팔았던 p4(−5.3pp)를 COMP가 회복 = reward 0.316·**db 0.375=floor 동일** → 게이트 결정론성이 prov 분산 상쇄(§3d/pin 가설 실증). **③ 위반0**(compliance g1~g4=0·compliant=bench) = C1/C2 모트 재현·frontier 낙폭 대비 우위. **④ Δspurious 후보 1건**: t95 floor 3/4→COMP 0/4=**constraints(equal_len) 게이트 over-steering**(gold=2주문 각1item→게이트가 1주문 중복 유도)·per-case 확정=constraints kind 부작용(격리 재검 대상). robust: vs prov 획득16/상실12(대부분 3/4 분산). COMP+D 실행중 | **[M]** | `RETAIL_PASS_COMPOSITION_DESIGN §3c` · `ecomp_checkpoint.py` · `sim_results/comp_retail_t4.results.json.gz` |
| **★C63** | **★게이트-deny↔transfer 100% 상관 = impasse 표지·게이트 부작용 아님**(COMP 456 전수 + 전문 정독 3/3·2026-07-11). 집계(deny 31 sims 전원 transfer·19 fail)는 "게이트-유발 포기"로 읽혔으나 **정독이 반증**: t93.0=⋈오선택→"수동 status 변경" 날조-escape를 게이트가 옳게 차단 · t8.0=**사용자가 transfer 요청**(결제 impasse·deny 부수) · t95.0=discovery 실패→중복-id **env 3회 거부**(게이트 아님)→"manual adjustment" 날조-escape 차단. ⇒ 게이트=compliance 정상작동(위반0·12/31 pass 유지)·**cooldown/문구 처방 철회**·진짜 레버=상류(discovery/lookup/DISAMB). 신규 서명: **impasse-시 발명형 escape-write**(C36 행동판)·deny 노트=무료 검출 마커. [[08]] 집계-오도 교정 사례 추가 | **[M]·정독 3/3** | `RETAIL_FULL_FAIL_CENSUS_2026_07_11` §4 |
| **★C64** | **★COMP 전수 실패 census — 실패는 78 task/167 sims·6표적은 부분집합**(per-case SYS+MOST 26 전수 정독). SYSTEMIC 15(신규 13: 20,34,36,37,41,57,71,76,79,99,100,102,111)·MOSTLY 11·FLAKY 52. 클래스: **A coverage/discovery ≈8**(E-PLAN·t81/t95형) · **B 대화-조건 over-action ≈5**(C25/C50 재확인·t99="내가 직접 하겠다" 수행) · **C compound-criterion ≈6**(t20 argmax·t37 예산·t79 attr매칭→calc/formalize) · **D GET-chain lookup ≈5**(t86/t102/t109 타-주문 주소 오복사·t39 빈값→결정론 filter-lookup+DISAMB) · **E DISAMB-도달 63 sims**(COMP+D-v2) · **F 값충실도**(t17 St≠Street 4/4→GROUND 원문-치환) · **G NL_ONLY 13**(count/총액=calc-NL) · **H 게이트-포기**(C63). v25e: t0·t61 4/4(**P2 GO**)·t47 3/4·t17/t40/t95 잔존 원인 확정 | **[M]·per-case** | `RETAIL_FULL_FAIL_CENSUS_2026_07_11` 전체 |
| **★C65** | **★GROUND-VERBATIM NO-GO + t17 재진단 = prov rescue 입도 구멍**(V0 실측+로그+코드 정독·2026-07-11). ① V0 census: fuzzy 치환 표적 **양 arm 0건**(COMP·v25e fix 0)·empty-치환 break 실재(t59 gold=빈값) ⇒ **레버 폐기**. ② t17 "축약-복사"는 오진 — v25e 4/4+v25d 전수 **read 0회·정답 원문 문맥 부재** = 미조회 자유텍스트 날조(실주소 근사=오염 의심). ③ **기전 코드-확정**: 첫 fab=order_id `#W8665881`(`#`-접두 거짓양성) → rescue 분기(`t2_gate_patch.py:881-888`) **per-call `break`** → address1 fab 미검사. ⇒ 처방=**PROV-RESCUE-PERARG**(per-arg 스킵+`#`정규화·무료) — C24 free-text 봉합·C29 정합. 리뷰의 substring-가설은 기각 | **[M]·코드+전수** | `CENSUS_LEVERS_DESIGN §1` |
| **★C66** | **★E-REF scale 사다리: 참조-바인딩 1.5B emergent·정적소음 강건**(밤샘 6모델·무료·결정론gold). clean(P1/P2 bind): 0.5B **0.04**→1.5B **0.89**→3B 0.99→32B 1.00(parse 0.92라 파싱문제 아님). 부하(20k)·distractor(10) 어느 scale서도 안 부숨(능력有면 1.00). ⇒ **tool-use 추상능력=초소형서 삶**([[00]] 증거)·범인=동적오염(정적 아님·lost-in-middle 기각). 동적축(near-miss/paraphrase) 실행중 | **[M]** | `OVERNIGHT_RESULTS_2026_07_12` §1 |
| **★C67** | **★4지선다 출처선언 처방-비교**(base/prompt/loop·n=60·격리단일턴·per-case검증). 출처선언 정확도 base **0.38**→prompt **0.75**→loop **0.77**. base=ASK남발(21/60·t17형 시스템적·GET 1)·loop=ASK **0**(문맥길이 불변·[[45]]). find_exact 60/60·find_wrong 20/20 검증. ★격리(C42 regime·prompt 작동)이라 [[42]] 모순 아님 — **in-vivo(멀티턴) prompt붕괴 여부는 C-stage 판정**·loop이 이길 것 예측. GET 최종값 미검(출처선언 정확도). learn arm=격리서 여지작음·오염데이터로 필요성 先실증 | **[M]·per-case** | `OVERNIGHT_RESULTS_2026_07_12` §2 · `c51_fourway_prescriptions.jsonl` |
| **★C68** | **★fexec 형식화 실행-채점 = 0.79 (기존 EM 0.00은 채점버그·폐기)**. hand-gold 폐기·실행기가 gold item_id 내는가로 채점(exec-correct). target(t20/t37/t79) **0.88**(t20 4/4·t37 3/3·t79만 ⋈오류)·전체 변형선택 87건 **0.79**. 기존 "constraints 0%·EM 0.00"=태스크당 단일 hand-gold가 멀티품목 무시한 **채점버그**(3문서 deprecation). ⇒ **FORMALIZE-EXEC 미편입 판정 붕괴→재편입**(B-max②)·형식화는 실제 작동·잔여=⋈경계 | **[M]** | `E_REF_BOUNDARY_DESIGN §0a` · `fexec_exec_probe.py` |
| **★C73** | **★trivial-회귀 오염원=개입레버 spurious·"양" 아님 + advise>override(decidable) + intent-ambiguity 잔여**(task106 양방향절단·prompt-rx nt4·per-case·`abl_sysprompt_106*`·`prompt_rx`). ① **full(COMP+D-v2)이 COMP-robust 6 trivial 회귀**(nt1 30/36·COMP 16/16 robust=flaky 아님). ② **"양" 아님**: sys-prompt COMP=b78c **바이트동일**(1674tok)=C66 정합(양 안 늘어남)·기전=런타임 엔진인터셉트(프롬프트 아님). ③ **오염원=개입레버**(operand/값/discovery 바꿈): 양방향절단 DISAMB 주범(COMP+DISAMB 0/2)·EPLAN/GROUND/PRINCIPLE 중복상호작용(단일제거로 안 고쳐짐)·**가드(PROV-rescue·cap) 무죄**=Δspurious 정밀실증(등대 "레버는 하나 사면 하나 판다"). ④ **advise(prompt-rx)가 decidable 5/6 회복**(1,6,7,75→1.00)=부작용원=**override 기전**(단 advise=(a)상주채널·이득 아닌 부작용부재). ⑤ **106=0/4 잔여=intent-ambiguity**(XXL "one size smaller"=XL gold vs 대화 "small"→agent black-S·override/advise 둘다 경로분기 오답)=**valid-but-wrong-for-user=scaffold 불가=ASK/E7**(voting settled-neg·fleet≠voting). ⑥ full **runtime 폭발**(20-58min·subcall+gate루프). caveat: 절단 nt2 배칭비결정·B(guard nt4 실task) pathology-kill 미완. ⇒ 처방=**결핍-게이트 라우터**(난이도 아닌 실패에 게이트·36/78 창발)+router-adv. | **[M]·per-case·양방향** | `TRIVIAL_REGRESSION_ABLATION` · `SCAFFOLD_STATE_ROUTER_DESIGN` · `INTERVENTION_LEVER_CONDITIONALIZATION` |
| **★C69** | **★동적오염이 참조-바인딩을 정박치환으로 부순다**(GPU0·32B GPTQ·infra 0·per-case 전수). C66(정적강건·1.00) 대비: **near-miss(축B same-dim distractor)가 바인딩(**thinking-off**) clean 1.00→lv1 0.72→lv2 0.75→lv4 0.47 (★단 **thinking-ON이 닫음**=C72 F3분기②·8B서 lv4 1.00·near-miss=decidable)**(오염밀도 단조·`op`연산선택 1.00 불변=제약 *값*만 오염). **per-case B lv4 19/36 fail 전건 = gold 대신 near-miss anchor값 포획**("large"→"small"·"M"→"S"·"2L"→"1.5L")=랜덤 아닌 **정박치환**(C43 동형). paraphrase(축P)는 약함(bind 생존·cons 1.00→0.75). fexec_all 87건 exec_correct_avail 0.770(제약형 0.74<무제약 0.93). ⇒ **tool-use 격차=정적소음 아니라 *동적* 간섭**·E-REF 완성·[[00]] 직접증거·2509.09677 self-cond-scale불변과 수렴 | **[M]·per-case** | `OVERNIGHT_RESULTS_2026_07_12 §결과4` · `eref_gpu0_*.jsonl`(persist 539b36d) |
| **★C72** | **★E-HORIZON: verify(싼 결정론)가 per-step을 scale보다 급격히 산다·기전=상태-발산 절단**(무료·GPU1·Qwen2.5 사다리 0.5-32B·직접-증분 running-sum·결정론gold·per-run 정독). base(scale) per-step 0.5B→14B **0.006→0.322**(완만) **→32B 0.748**(단순task 포화·헤드라인=banking로·§FE1) vs **verify 0.911→32B 0.958**(pass-cost0·scale-불변). **detect≈base**=탐지 아닌 *교정*이 핵심. **★기전=상태-발산(지속6·self-conditioning 아님)**: **inject arm 통제 오류주입 후 자기일관(selfcons_postinj) ≈ clean base **전 scale 불변**(★회수 2026-07-12: 7B 0.806/base 0.731 · 14B 0.931/base 0.915 · 32B 0.969/clean 0.960)**=주입오류가 산술 무손상=순수 sd 인과(F9 자연 0.917 승격). 2509.09677 self-conditioning은 distinct mode·인용. ⇒ verify가 horizon(=per-step^H)도 scaffold로 삼. **통합이론(§9)**: horizon=동적오염 시간축→오염-방화벽. **명명=근본기능6 지속([[48]]·M-코드 로컬)**. caveat: synth(2509.09677 동형)·thinking arm=near-miss **닫음 확정·F3분기②**(★E-THINK2 회수 2026-07-12·jsonl직접: lv4 bind thinking-on 1.7B 0.69→4B 0.97→**8B 1.00**→**14B 1.00** vs off 0.00/0.28/0.33·32B-GPTQ 0.47 = near-miss는 decidable via thinking·8B서 완결·1차 "thinking≠외부오염"은 파싱 아티팩트 확정)·in-vivo=sd/sc proxy [M] | **[M]·per-run·inject 인과** | `THINKING_HORIZON §8·§9` · `UNIFIED_TAXONOMY 근본기능6` · `ehoriz_*.jsonl` |
| **★C71** | **★banking = "scale로는 horizon을 *충분히·싸게* 못 산다"의 실-도메인 실증**(기존 §3.2f 재해석·[[47]] 재런 0·per-step 엄밀도출). ★**정정**: "scale이 horizon *못* 산다"는 틀린 강한형 — **개선하되 느리고·불충분·비쌈**. frontier banking pass 0.098~0.384(17모델·GPT-5.5 최강)·gold median H=8. pass=p_step^H: per-step은 scale로 **개선되나**(gemini2.5pro 0.748→GPT-5.5 **0.887**·err 25%→11%) **상용 임계(pass0.7@H=8엔 p=0.956·err 4.4%)에 2.6× 미달**→0.887⁸=0.373 갇힘. 정체=**구조적 per-step**(reach/unlock 발견체인 universal-fail 28/28 + 결정가능-인자 ≈40%·census 모델기울기=미실행은 사되 인자오류 잔여)=**scaffold가 p→1·pass-cost0·scale-불변으로 만드는 것**(controller·calc·provenance·coverage). ⇒ **주장=scale로 상용 못 사니 scaffold로 더 싸게 산다**(2509.09677 합성=계산오류=scale삼처럼 보임 / banking 실tool-use=구조=scaffold가 더 쌈). 정직: per-step 전무개선 아님·all-or-nothing 실재(완주후불일치 45%=진짜 인자오류 per-case 3건). **통제-인과 짝=E-HORIZON(실행중)·미측정=scaffold 실붙임 E-XFER-bank(유료)** | **[M]·per-case(§3.2f)** | `THINKING_HORIZON_LEVER_SURVIVAL_DESIGN §7` · `TAU2_FRONTIER..._MASTER §3.2f` |
| **★C70** | **★선행지형: "scale=동적오염내성"은 축별로 갈림·강한형 반박·whitespace 미선점**(죽은 딥리서치 wf 수동종합·claim179+verdict36). **[지지·verdict]**: scale이 사는 것=**horizon/실행길이**(2509.09677 Sinha/Geiping ICLR2026=우리 **F6**·C5 수렴)·self-conditioning=scale-불변 잔여·thinking(비-scale)이 닫음 / snowballing 존재(2305.13534 ACL2024·capable-but-fails 67/87% 자기인식). **[반박·verdict]**: 2606.07937(multi-agent "smaller GPT-5.3"=날조·인용금지)·2511.12869 강한 architectural 주장(RULER 2404.06654 반증)·2505.17656 self-cond 오분류(i.i.d.≠궤적). **[미검→검증중]**: 축 b sycophancy(Perez 2212.09251·Sharma 2310.13548)·c entrainment(2505.09338·2604.13275 scale-비균일)·d 멀티턴(Laban 2505.06120 39%붕괴)=verdict 미실행 → **표적 딥리서치 재발사**(`wf_42f15797-d8d`·진행중). whitespace(agentic궤적×오염축별 통제주입×scale사다리)=미선점·2509.09677 인접이나 단일합성task·축a단독=구분 | **[M-lit]·축bcd [?]검증중** | `SCALE_DYNAMIC_CONTAMINATION_PRIORWORK_2026_07_12` |
| **★C64** | **★COMP 456 전수 실패 census — 실패 78task/167sim·6표적은 부분집합**(per-case SYS+MOST 26 정독). ⋈경계·coverage미완·compound·derailment·조건체인. 신스택 재확정(`NEWSTACK §G`): ⋈경계9+coverage미완6 지배·신규부작용 t17(PERARG 과교정=정밀부족·loop해결)·t102(⋈루프)·**Δspurious 음성**(3.8%<5.7%·over-action 3건 전부 기존 B-class per-case) | **[M]·per-case** | `RETAIL_FULL_FAIL_CENSUS_2026_07_11` · `NEWSTACK_GAIN_SIDEEFFECT_2026_07_11` |
| **★C63** | **게이트-deny↔transfer 100% 상관 = impasse 표지·게이트 부작용 아님**(전문정독 3/3 반증). t93=날조-escape 차단·t8=사용자 transfer요청·t95=env거부. cooldown 처방 철회 | **[M]** | `RETAIL_FULL_FAIL_CENSUS §4` |
| **★C65** | **GROUND-VERBATIM NO-GO + t17 재진단 = prov rescue 입도구멍**(V0·코드정독). fuzzy치환 표적 0·t17="123 Elm St"=미조회 자유텍스트날조(축약복사 아님)·'#'접두 거짓양성이 rescue break 선점→PERARG per-arg 봉합 | **[M]·코드** | `CENSUS_LEVERS_DESIGN §1` |
| **★C50** | **E10 정책-precondition 게이트 = NO-GO**(무료 격리검증·arm asmregen32b·db.json). over-action(+9·C25)의 "precondition"은 **decidable하나 DB-state 아니라 대화(policy+intent)** 에 산다: P1(refund-target) over-block 6 > TP 5·비판별(t99 pass/fail 동일 PM) · P2(status) 실행 write ineligible **0/602**=환경 이미 집행(C12 redundant) · per-case 5 task 전부 환불 PM=주문 pm_orig 일치·status-eligible. ⇒ **C25 "decidable ∧ 미집행" 정련**: policy-decidable ≠ DB-decidable · [[06]] "over-action 게이트 금지"가 DB-게이트에 한해 재확인 · Lever A(2026-06-27) 동형. 남는 후보=대화-precondition controller/ASK(게이트 아님) | **[M]** | `E10_PRECONDITION_GATE_DESIGN_2026_07_09` §5.1 · `e10_precond_probe.py` |
| **★C73** | **★retail scaffold 천장 재확정 = ~0.64(0.5x는 측정오인)·base=COMP(present 폐기)**(2026-07-12·`scratchpad/hist.sh`·decomp·comp_reg.sh·per-case). ★**세션 내내 "천장 0.5x·대부분 경계" 비관은 측정-전제 6회 오인**(subset↔full·잘못된 base·db↔reward·nt1↔nt4·파서버그·집계라벨). 정본: **full pass^1 COMP 0.634·assembled 0.640**(0.5x는 **78-하드셋**·36 trivial=COMP 1.000 제외). **assembled 폐기**(present=DB주입 트릭·부작용 C16/C34)·**base=COMP**(gate+prov+nested+calc·present 없음). **b78c(COMP+D-v2)는 COMP 대비 개선**: 하드78 db 0.506→**0.551**·**robust 퇴행 0**(COMP 4/4→b78c fail=0)·회복17/퇴행11(flaky)·full 추정 **0.693 db**. 제 "퇴행 6⋈"은 assembled(present) 대비였음·COMP엔 없음. **35 db-fail 완전분해**: 아티팩트~5(C26 sameopt+price·C28 reason=실패아님)+addressable~22+경계~5-8 ⇒ "78중 ~30 회수" 지지. **다음=B(78) COMP-base regression-safe 개선→C-full(36 회귀 확인)** | **[M]·per-case·정정** | `B78C_FORENSIC_AND_S1_REDESIGN_2026_07_12 §6` · `_cdp_private_local/HANDOFF_2026_07_12_NIGHT` |
| **★C74** | **★banking "act-vs-advise 36%" = 아티팩트·action-required 레버 오정합**(2026-07-13 LATE-2·offline 유도+15 sim 스모크·[[08]] 포렌식). action-required 리마인더 채널은 **라이브 배선 정상**(순수-조언 회피=UserMessage 재생성·offline 14/14·live 발화 9/5sim). **그러나 banking에 오정합**: (1) **gold action 대다수가 `requestor='user'`**(에이전트 호출 불가·user-실행): apply_for_credit_card 19·submit_referral 7·submit_transaction 1 / agent-실행은 `call/unlock_discoverable` 428+275(=reach/discovery). (2) 표적 스모크(task_001/003/007=apply) 전부 **user가 실행**(msg role=user)·에이전트는 KB+`give_discoverable_user_tool`로 발견·제안 → **행동 일어남·실패=⋈ 오선택**(t003 gold `Silver Rewards`인데 user가 `Business Platinum` 신청·db_match=F). (3) 오프라인 36% = "gold action_tool 미호출∧마지막 asst 텍스트"인데 user-실행이라 **미호출 항상 참**+마무리 인사를 회피로 오분류=**분류오류**. ⇒ 핸드오프 REACH-재진단 **재철회**(초기 REACH는 KB-death 오염·act-vs-advise는 user-실행 구조 놓침). **40-태스크 유료 probe NO-GO**(스퓨리어스 확인·예산). banking binding=reach/coverage/horizon+⋈(C52 재확인)·scaffold 후보=discovery controller·coverage(action-required 아님). ★**재채점(사용자 지시·`action_checks.action_match`정확채점·floor 198 sim)**: pass **6.1%**·실패 gold-action=**(A)필요도구 미호출 580(REACH/give-up)+(B)operator/operand ⋈오선택 509+(C)정확도달·타인자 152**. per-case: t003=operand⋈(카드)·t023=검증F1+조기escalationF5·t035=operator GET실패(KB discovery). ⇒ **banking=retail과 동일 {operand+operator} 루프·ABox만 다름**(operator_resolution=discoverable=KB-GET 추가). 레버=operator/operand FIND(formalize_intent_tool 재사용)+discovery controller(eplan 동형)+verify/persistence 게이트. **frontier 궤적 소실([[47]])=aggregate만**(opus4.5 24.7·gpt5.5 37.4 vs 6.1%) | **[M]·소표본·구조증거(requestor split·재채점)=[S]급** | `BANK_ACTIONREQ_PROBE_FORENSIC_2026_07_13` §6b · `bank_rescore_pathdiff.py` · `sim_results/bankar_smoke5c.results.json.gz` |

| **★★C75** | **★표준 user-sim(gpt-5.2) 반전 = banking 레버는 gpt-4.1 아티팩트를 겨냥·표준서 순수 손해**(2026-07-13 LATE-3·G vs GR·apply 5태스크·nt1·유료·per-case). leaderboard 표준 user-sim=**gpt-5.2**(`docs/leaderboard-submission.md`·frontier 전부 사용·gpt-4.1보다 저렴). 표준 채택 후: **G(레버無) 3/5(60%) vs GR(레버有) 0/5**. 기전(궤적): 레버 regen이 깨끗한 gpt-5.2 흐름 탈선(t003 오추천 유발·t023 transfer 포기). **★t016(submit_transaction)·t003(오추천) 둘 다 G서 pass** = gpt-5.2 user-sim이 거래 실제실행·자연 정답 ⇒ **§6b 재채점(6.1%)·오추천/미추천/reach/user-sim아티팩트 포렌식은 상당부분 gpt-4.1 user-sim 아티팩트**. 우리 banking floor(gpt-4.1·openai_embeddings)=비표준·실패 인플레. ⇒ **banking 레버 L0-4 표준서 유익하지 않음**·진짜 격차는 표준 gpt-5.2 floor 재측정 필요. 등대 "하나 사면 하나 판다"의 극단(사는것 없이 팔기만). caveat n=5 소표본·기전 구조적 | **[M]·소표본·per-case·기전 명확** | `BANK_ACTIONREQ_PROBE_FORENSIC §6d` · `sim_results/bankar_rec_g5·gr5` |

| **★★C76** | **★banking frontier-irreducible 격차 = 정책-구동 파라미터 계산/판정(경계 아님·decidable)**(2026-07-13·17모델 전수 per-step·`C:/tmp/traj/*_banking.json`·user-sim gpt-5.2·[[08]]). frontier pass 11-37%(gpt55 최강·openai_emb=우리 retrieval은 11% 아티팩트). **지배 실패=`call_discoverable_agent_tool`**(발견 뱅킹 오퍼레이션 실행): 틀린 도구명 4384(operator-⋈)·**도구맞음·파라미터틀림 4311**. ★**틀린 nested 파라미터=정책-계산/판정**: `customer_max_liability_amount`602·`eligible_for_provisional_credit`439·`amount(_difference)/expected_apy`289/283/295/243·`pin_compromised`206 등. **per-case(t085·전 frontier 0/66)**: gold 책임상한=50 vs gpt55=100(전액)=**정책(즉시신고→$50 cap) 미적용**. **hard core 45/97 태스크 전 frontier ≤10%.** ⇒ **격차=F2b계산/F1정책(decidable)=결정론 scaffold 사정거리**(gpt55조차 못 여는 걸 scaffold가 열 여지=thesis 지지 후보). ★**내 레버 L0-4(reach/verify/추천)=엉뚱한 표적**(gpt-4.1 아티팩트·C75). 진짜 레버=정책-파라미터 formalize→compute(fexec 동형·정책규칙=A2). caveat: nested 스키마 도구별 상이→A2 인코딩 부담·실현가능성 별도판단 | **[M]·17모델×388·per-case** | `BANK_FRONTIER_PERSTEP_FORENSIC_2026_07_13` |

| **★★C77** | **★banking 강레버 = reference-filter(참조-⋈ ~83% 결정론 필터가능)·keystone 엔진 실증**(2026-07-13·frontier 17모델 hard-core 축 정량·[[08]]·리뷰 R1-R5). ★**자기교정**: 초판 "F3경계 지배·필터 31.8%·슬라이스 8%"는 **파서 오염**(타 모델 record 미추출 None==None 허위동일·"57개 동일" 부조리로 발각). **파싱신뢰만(field≥70%·n=798·15모델)**: transaction_id ⋈ **결정론필터 유일식별 83%**(gemini/gpt/grok/opus/sonnet 전부 80-88% 일관)·진짜중복 17%·ASK가능 0. ⇒ **참조-⋈=경계 아니라 대부분 필터가능**(user가 date/amount/merchant 식별정보 제공→formalize→수집record filter→id·retail fexec 일반화). 참조실패가 hard-core 지배버킷 × ⋈93% × 필터83% ≈ **참조실패의 ~77%**=고사정권(compute 7.5% 압도·사용자 참조/reach 재설계가 정답). **compute 7.5%=소슬라이스 병기**. keystone 엔진(t2_compute 일반op·A2선언·전이=A2만)=아키텍처 실증(liability 95% gold재현·unit 14/14). caveat: 83%=파싱신뢰 26%부분표본+필터천장(formalize×filter<83%)·per-case 1건([[08]] 잠정) | **⚠C80서 *prevalence* 철회(⋈≠지배·사정권853→222)·[M]** | `BANK_COMPUTE_OP_KEYSTONE_DESIGN §5c-5d·8-0` · `bank_filter_repro`·`bank_reference_scope` |

| **★★C78** | **★reference-filter keystone REPLAY 확정 = 비-중복 참조-⋈ 결정론 filter로 100% 교정가능(전체 81.9%)·0오답**(2026-07-14·`bank_keystone_replay.py`·17모델 hard-core ⋈ n=853·무료 offline·[[08]]). C77 "83% 필터*가능*"을 **실제 A2 `reference_filter` 규칙**(gate json 102-115)을 `t2_compute.apply_op(op=filter)` 엔진으로 replay해 **교정률 실측**으로 확정(criteria=gold서 파생=perfect-formalize 천장). **수치**: (a) date+type 640(75.0%·결정가능부 91.6%) / (b) +merchant토큰 646(75.7%) / (c) **+amount 699(81.9%·결정가능부 100%)**·**전 variant 0 오답**. **진짜중복 154=18.1%**(全식별필드 동일·`on_ambiguous=none` 정당abstain·벤치 인디터미너시)=결정가능부 699. **★핵심**: 비-중복 ⋈ 전부 date+type+amount 결정론 filter로 gold 도달·merchant(어려운 NER) 거의 무관(date+type만으로 91.6%)→천장이 저모호 formalize에 기댐. **★캐비엇([[08]])**: ①**0오답=구조적**(gold파생 criteria→gold 항상 매칭셋→유일매칭이면 필연 gold)=**reach 천장**만 잼·**Δspurious 미측정**(별도게이트 §8-2)·"0오답=Δspurious0" 오독금지. ②perfect-formalize 천장·실제=formalize정확도×천장·formalize half 미측정(§0 선택유료). ③미해결 비-중복53=merchant토큰 조잡(다중매칭)·amount로 해소·근본한계 아님. ④파싱신뢰 미필터→하한 성격·merchant_phrase(65%)=비연속substring 아티팩트 무효. | **[D]filter천장·[S]진짜중복·[?]라이브교정률** | `BANK_COMPUTE_OP_KEYSTONE_DESIGN §13` · `bank_keystone_replay.py` |

| **★★C79** | **★formalize half 실측 = 추출은 좋고(정당대상 69.5%·랜덤오류 1.5%) binding은 다중-dispute 앵커링**(2026-07-14·`bank_keystone_formalize.py`·리모트 Qwen2.5-32B·localhost:8140·API비용0·n=853·[[08]] 재분류). **구조적**: banking hard-core ⋈ = **852/853 다중-dispute**(한 대화 여러 거래 동시 dispute). 실측(user발화→32B formalize→결정론 filter, 전 dispute셋 대조): 교정(this gold) 28(3.3%)·**오답→다른 *정당* dispute대상 565(66.2%)=mis-pairing(앵커링)**·none 247(29%·진짜중복18.1% 포함)·**오답→dispute셋밖 진짜오류 13(1.5%)**. ⇒ **★formalize 필드추출은 병목 아님**(정당 dispute대상 69.5% 도달·랜덤 1.5%뿐·per-case "FitLife/11-10/89.99"→정확매칭). **binding = 전역 formalize가 "지금 이 dispute=어느 거래" 못 가림**(참조/완결성·[[45]] reach·C52 banking binding 정합). 천장(C78) 대비 gap = 필드값 아니라 **앵커(어느 거래)**. **캐비엇**: per-action 페어링이 라이브보다 엄격(라이브=각 호출시점 교정)→**라이브 formalize교정률 ∈ [3.3% 하한, 69.5% 상한]**·dispute순서 정렬 의존·오프라인 앵커없이 확정불가. Δspurious 안전(랜덤 1.5%). **레버 함의**: 다중-dispute 유효화엔 per-dispute 앵커 필요(정련 방향). | **[M]추출·[M]앵커binding·[?]라이브교정률** | `BANK_COMPUTE_OP_KEYSTONE_DESIGN §14` · `bank_keystone_formalize.py`·`bank_xmatch_formalize.results` |

| **★★★C80** | **★[[08]] 포렌식 방향전환 = "transaction_id ⋈ 지배" 오염 발각·진짜 지배는 COVERAGE**(2026-07-14·forensic-guard 촉발·`bank_xmatch_forensic.py`·궤적 전수·[S]). **발각**: [3.3,69.5] 좁히려 per-dispute 앵커 진단 중 task_086 정독→16 "⋈ case"가 **전부 동일 chosen·distinct chosen=1**·gold5·user *"한도로 다 못 냄"* = **⋈ 아니라 COVERAGE 미제출**. 추출이 `same[0]`(첫 호출)을 미제출 gold마다 페어링해 853으로 오분류. **궤적 재정량(agent 실제 제출 id 집합 vs gold 집합·`bank_xmatch_forensic.py`·disputes credit+debit 결합·936 sim·gold 3904)**: 제출·맞음 **2904(74.4%)**·**⋈wrong 159(4.1%)**·**missed=coverage 1000(25.6%)←지배**. **missed 분해: A.0제출(REACH/DISCOVERY) 804(80%)·B.한도 110(11%)·C.부분미완 86(9%)**. ⇒ **transaction_id 선택 정확·진짜 ⋈=4.1% 소수·지배=COVERAGE(그중 80%=dispute 미착수=reach/discovery)**. **C52/C71(reach/coverage)·C76(compute param)·handoff§1(32B dispute 도달0)과 3중 수렴**. (⚠초판 222/1121/27%=전-discoverable 스코프 오염·이게 dispute-한정 정본.) **철회**: C77 "⋈ hard-core 지배·reference-filter 큰레버·82% filterable"의 *prevalence*(사정권 853→진짜 222). **생존**: C78 filter 유일식별 능력(결정가능부 100%)=데이터사실 유효·단 사정권 222. **재정렬**: banking 큰레버 = **coverage/completion**([[14]] E-PLAN·§1.4 F4·write강제금지→미제출감지 read/plan 게이트), reference-filter 아님. | **[S]·궤적전수·결정론 집합비교** | `BANK_COMPUTE_OP_KEYSTONE_DESIGN §14.5` · `bank_xmatch_forensic.py` |

| **★★★C81** | **★COMPUTE slice 실측 = banking decidable lever 정답·id-correct dispute의 22.4%(전체 gold 16.7%)·verified**(2026-07-14·`bank_compute_slice.py`·궤적전수·[S]·forensic 종착). C80(⋈ 비지배)·§14.7(act-gate refuted·gold dispute=13-25필드 gather+compute) 뒤, **진짜 decidable lever=dispute 인자의 computed 필드**를 실측. id-correct dispute 2904 필드-대조: **liability 51.1% 오답**(최대 단일)·provisional_credit 22/15%·partial_refund 14%·card_action 3%(verified 실오류: agent None/0 vs gold50·keep vs cancel·True vs False). **compute-closability: pass 47.4%·compute만오답=엔진이닫음 651(22.4%)·혼합13%·noncompute만17%**. ⇒ **★compute slice=651(id-correct 22.4%·전체gold 16.7%)=키스톤 compute(§7 엔진 有·liability lookup 95%실증)의 실측 사정권**·⋈(159·4%)의 4배·frontier-irreducible. noncompute 잔여=card_last_4(369·도출후보)·pin(gather). **캐비엇**: 651=gold재현 가정 천장·실효=651×§8-1 gold-blind 재현율(재저작 필요)·id/gather/reach 선결(compound). **종착 우선순위: compute > ⋈ ≫ act-gate(0). 첫 설계(C76/§7)가 옳고 ⋈우회(C77-79)=오염 곁길.** | **[S]slice·[M]실효상한·[S]우선순위** | `BANK_COMPUTE_OP_KEYSTONE_DESIGN §14.8` · `bank_compute_slice.py` |

| **★★C82** | **★horizon=논리곱·"거래고정→파생" 아키텍처 검증 = slot-fill 순이득~0·가치는 Stage1(⋈)에 집중**(2026-07-14·사용자 설계·`BANK_COMPUTE_OP_KEYSTONE_DESIGN §15`·[S]). banking pass=거대 논리곱(∧ 55-100필드)·N커지면 곱붕괴(0.97²⁰≈54%/dispute×4.2=7%)·상관(필드가 거래이해서 뭉침)이라 scale이 다발 통째 올림. 사용자 처방: ①거래(root) 결정론 고정→②slot-fill/link/compute 파생→③user필드만. **검증(per-레코드 일치)**: transaction_date/account_id **100%**·disputed_amount 68%·transaction_type 30%·card_last_4 링크messy. **★잔인한 정렬**: 파생 깨끗한 필드=agent 이미정답(date 0.2%·account 0% 오답)·agent 오답필드(type 11%·card_last_4 19.5%)=파생불가. ⇒ **slot-fill 레버 순이득~0**(고칠 곳은 파생불가·파생가능은 이미정답=§8-3 moat 재현). **∴ Stage2 redundant·미구현([[13]])·가치는 Stage1(거래고정=⋈)=이미 구현된 reference-filter**. 사용자 아키텍처=reference-filter가 왜 핵심인지 논리곱 재증명. **banking pass 큰상승=여전히 scale영역**(horizon collapse 부분적·§0/C71 재확정). | **[S]slot-fill일치·[S]정렬·[M]Stage2redundant** | `BANK_COMPUTE_OP_KEYSTONE_DESIGN §15` |

| **★★★C83** | **★horizon 재반전 = verify-or-ASK로 곱→1·H_min≈15bits(작음)·gather는 scale아닌 효율문제**(2026-07-14·사용자·§16·[S]). 사용자 통찰: horizon실패=곱≠1이지 "많음"이 아님. **verify-or-ASK**(decidable→결정론verify→1·non-decidable→ASK→user가1)로 silent p^N 붕괴 해결. **H_min 방법 확정**(τ_derive 0.95→DERIVE·τ_default 0.9→DEFAULT·나머지 ASK의 joint엔트로피·amortize). **banking 측정(per dispute-type joint)**: DEBIT **4.27bits**(category+pin+contacted+discovery)·CREDIT **2.60bits**(reason+contacted)·대화 ≈**11-18bits≈질문 5-7개**. vs naive 26필드=수십~수백bit(~5×↓). DERIVE~18필드(정보0)·DEFAULT~5(police_report 100%false·written 100%true 등). **★재해석**: agent gather실패=정보큼 아니라 DERIVE안하고 재질문·DEFAULT안쓰고 다물어 대화부풀림→user-STOP. ⇒ **DERIVE+DEFAULT+VOI로 N_eff→H_min(~7질문) 접으면 완주=non-scale 레버.** "gather=scale영역"(§8-5/C82) **부분철회**·irreducible=~15bit(작음). verify-or-ASK controller(①거래고정+파생 verify ②default ③VOI-ASK)가 처방. caveat: 엔트로피=gold기준 하한·DERIVE는 기전작동 가정(card_last_4 messy면 H_min↑). | **[S]H_min측정·[D]방법·[M]gather=효율** | `BANK_COMPUTE_OP_KEYSTONE_DESIGN §16` |

## 4. 실험 큐 (우선순위·상태 · 2026-07-09 정렬)
| ID | 실험 | 닫는 것 | 비용 | 상태 |
|---|---|---|---|---|
| **★E-XFER-bank** | **Phase 3 도메인 전이 실측 — banking**(32B floor + 게이트 arm·97태스크) | A2 전이가설(처방 전이)·덱 결과⑩ banking 칸·특허 도메인-일반 청구 | 유료(승인 2026-07-10) | ⏸ **재시퀀싱(사용자 2026-07-11): retail 스택 확정 후 최종 스택으로 gate arm 재실측**. 근거=C52(banking binding=reach/coverage/horizon인데 구스택은 F1/날조/⋈ 표적=이중지출 방지·[[09]]). ✅확보분: **floor nt2 n=192 mean 0.050 pass 9 infra 13**(`bankxfer_floor_bank_t2.gz`) + 구스택 gate arm partial **n=13 pass 1 infra 0**(`..._t1_partial_oldstack.gz`·진단표본) + 스모크 UNI_OK(banking A2·레버 라이브 검증됨). 재개 조건 = retail 종결(census 레버+E-PLAN 편입·0.66~0.70) | 
| ~~(구) E-XFER-bank gate arm nt1~~ | 구스택(게이트+prov+DISAMB+calc) 전이 arm | — | — | ❌ **13/97서 중단**(사용자 결정)·partial 영속. 예측-확인 가치는 partial+스모크로 갈음 |
| **★E11-e2e** | **출처선언 레버 다중턴 e2e** (floor+prov·`T2_PROV_REGEN=1`·present OFF) | **make-or-break**: 단일턴 날조 67→0(C45)이 pass로 이어지나 | 유료(승인) | ✅ **GO 확정**(C53·**456/456**·reward 0.5768>floor 0.547 +3.0pp·db_pass 0.621·**tme 1/456**·t17 날조 0/4→4/4 교정·per-case·gz persist) |
| **★E11** | **출처선언 레버**(4지선다+provenance 검증+재발화·GET폴백) | 날조 억제 | 무료 | ✅ **GO**(C45·67→0%·over-block 0·Δspurious 0·단일턴 n=60) |
| **★E-COMP** | **retail 검증기 합성 arm**(게이트+prov+nested/calc 단일 생성-레벨 통합 → COMP·+DISAMB → COMP+D) | GO 레버 최초 합성(이중패치 해소)·C59 DISAMB e2e 승격·retail pass 상향 | 구현 무료·full 2런(912 sims·승인) | 🔄 **리뷰 GO(블로킹2 반영)·구현+24테스트 PASS·스모크 실행중** — census: disamb-도달 77/193·NL_ONLY 16/19=calc 사정거리·기대 0.63±0.03. `RETAIL_PASS_COMPOSITION_DESIGN §2·§3b`. full은 E-XFER-bank 후 |
| **★T5-C** | **silent repair → COMP+D-v2**(사용자 "턴 버리지 말고 조용히 개선"·2026-07-11): fix #1 적용(07337a3) + **P-A** GROUND unified 이식(\|C\|=1 제자리 치환) + **P-B** DISAMB→격리 서브콜 후 인자만 치환(대화 불변·replay-clean) + **P-C** prov 구조대 모드. ★**rev3 리뷰 반영**: 새 실험=**COMP+D-v2**(전체 unified 스택+silent·기준선 C62 COMP .634/.316·routerv2=격리 ablation 강등)·**constraints kind 제외**(t95 over-steer·R-β)·unified 분기 이식=이번 범위. ★**단계 A(v25e·6표적 nt=4) 완료**: t0·t61 4/4(**P2 GO**)·t47 3/4 / t17=값충실도(GROUND 확장 표적)·t40=NL·t95=discovery+NL(C64 §5). ★**단계 B 표적 재구성(C64)**: 기존 13 + 신규 SYSTEMIC 13 = **26 task**·census 레버 5종(GROUND-VERBATIM·CALC-EXT·EXCLUSIVITY·NOTICE-REFUND·DISAMB-ADDR = **설계 완료** `CENSUS_LEVERS_DESIGN_2026_07_11`·각 V0 무료 census 先) 편입 후 사이클 | C61 −37 환매·C53 p4 환매·나비효과 채널 제거·**COMP 위 DISAMB 순증 검정** | 구현·V0·V1 무료·V3 유료(**승인 대기**) | **단계 A [M] 완료** — `T5C_SILENT_REPAIR_DESIGN §6·§10` · `RETAIL_FULL_FAIL_CENSUS` · `CENSUS_LEVERS_DESIGN` |
| **E-ENDGAME** | **scaffold 레버 전량 소진 프로그램**(R1 무료빌드: 대화-precond controller·retry이식·feasibility·E-PLAN live / R2: COMP2+E-PLAN arm) | over-action(+9)·coverage 47sims·zero-att 13·feasibility 10 — C21 잔여 조각 합성 | R1 무료·R2 ~912 sims(승인) | **[D]** — `SCAFFOLD_ENDGAME_PLAN_2026_07_10`(타 세션 리뷰 우선순위 반영·T5-B/E6′ 소유권 경계 명기). 도달목표 retail 32B 0.66~0.70 |
| **★E-ISO** | **정보-맞춘 3단 격리 replay**(A 궤적재현/B 격리원문/C 격리형식화 — census 실패 결정점 77+C60 flip 쌍) | **§1.5 Q2 재실행**: semantic 잔여의 부하 몫 확정(C13=정보-빈약 프로브·E1′ 미실행 공백). learn 표적의 사전 필터·B≈A∧C≫B 예상(C59/C60 정합) | 무료(로컬 32B) | ✅ **완료(C61)**: 오염20%·형식화-부하 실재(ITEMS C .44)·PAYMENT=결정론 P2 必·**⋈=경계 재확인(열거 역효과)**·PREINFO 6→E-PLAN. horizon은 별도: E-PLAN walk/기록(C43 메모장-무효는 날조 한정) |
| **★E-SPEC** | **전문 에이전트 × 결정론 오케스트레이터 재설계**(결정유형별 {agent, 검증기, A2슬라이스} 3중쌍 + pin ledger·context-종합=코드) | 오류-전파 구조 차단(오염 t61·간섭 C53 p4·표면형 C60)·thesis 완전형·7/7 멀티에이전트 제안 계승 | Phase A 무료(E-ISO 질량 게이트)·B 프로토(③operand+pin) | **[D]·Phase B 보류(사용자 2026-07-10 저녁: retail pass 완료 우선)** — `SPECIALIST_AGENTS_CP_ORCH_DESIGN`. ③=T5-B 좌석 공유·②④⑤=ENDGAME 재배치 |
| **★E6′** | **gather 학습 — 데이터 v3 재설계 先** (D7 근접오답+음성사례+on-policy+발명형 rejected · **★+C60 신규 표적: paraphrase-invariance** — 등가-표현 변주에 동일 결정 감독. C38과 달리 **재현 가능**: 실 궤적 user-발화 패러프레이즈 증강 + 4-trial pass/fail 분기=자연 대조쌍. [[00-thesis]] 도메인-일반 스킬 전형·P4 실증 축) | C7 · **C38: cfbsynth가 결손 재현 못함**(미확립) · C60 표현-민감 잔여 | 큼 | **▶ 데이터 게이트 통과 前 착수 금지** |
| ~~E10~~ | 정책-precondition 게이트 | C25 over-action +9 | — | ❌ **NO-GO**(C50·무료 격리검증: P1 over-block>TP·P2 환경집행 redundant·불가능성=대화 semantic·게이트 아님) |
| **E-ASK** | **ASK 위계 R1**(호출가능성·GET-chain·DISAMBIGUATE) | C48 · clarification 벤치서만 R0≠R1 | 무료 | ▶ ToolDial/airline 필요 |
| **E-XGRAMMAR** | **디코드-시점 제약 채널(f)**(xgrammar guided decoding·현 스택 미포함=전 개입이 생성-후) — 1차 표적: banking 도구명 날조 61 sims(유한집합 스키마 제약·ASK 불필요=위험 최소·`BANKING_FLOOR_LEVER_FIT`) / 값-수준(문맥-실재 열거)은 **ASK/null 분기 필수**(§1.5 abstain 봉쇄 위험·C48 위계·C58 통계-디폴트 동형 경계) 설계 후. 배선=tau2→litellm per-call extra_body | 채널-분화 제3실증(판정 동일·채널 상이)·특허 실시예 축(taxonomy 부록 X) | 무료 구현 | **[D] 등재만(2026-07-11 사용자 지시)·스택 동결 — S4/S5 후 검토** |
| ~~E1′~~ | 격리 formalize 서브콜 | C23: F2 −4·payoff 작음 | — | **하향**(우선순위 낮음) |
| ~~E1~~ | 완결/persistence 게이트 | C32: 짝맞춤 Δ=0 | — | **[M] 게이트 이득 미확인**(소표본) |
| **E3** | F3 경계 full-agent | C3b **[S]부분** | 무료 | ✅ 완료 |
| **E8** | frontier 격차 분해 | C9 | 무료 | ✅ 완료 |
| ~~E9 / E9′~~ | operand grounding 게이트 / free-text | — | — | ❌ E9 NO-GO(환경이 거부) · E9′ **E11에 흡수** |
| ~~present / autofetch~~ | — | **규칙 0 위반**(C34) | — | ❌ **영구 폐기** |
| E2 | QwQ+rparser nt=4 | C4b [M]→[S] | 유료 | ✅ 완료(clean 0.547·infra=fail 0.537·C33 해소) |
| E5 | 7B assembled | C2 사다리 | 소액 | 대기 |
| E6 | learn-wing four-bench→τ² swap | C7 | 큼 | **E6′로 대체**(τ² 전 데이터 타당성) |
| E7 | fleet | C6 | 보류 | big-tier 시 |
| **E-AMB** | 애매모호성 이론(고전 정식화) 검증 T1~T6 | 이론 [D]→[M] 승격·제2발명 배분의 단일-원리 유도 | T1~T4·T6 무료·T5-B 유료(승인됨) | ✅ **T1~T4·T5-A·T6 완료**(C55~C59): 단조성 3-arm 재현·P2b/c(prov가 \|C\|=1만 닫음)·앙상블 P4a 지지·**열거가 ⋈ 엶(.345→.657·C59)**·slot-filling+디폴트 불변성 기각(C58·T6h-2: FORCED 1.000 vs CHOICE .840→디폴트 개념 폐기) ✅ **T5-B 완주(C60·조건부 GO)**: DISAMB=robust-축 환매 레버(p4 +1.8pp vs prov·t61 완전복구·p1 중립·t46 무-write 탈선 잔여) ⇒ **E-AMB 전 항목 완결**(T4-확장 17모델·t46류 정독·완전상쇄 조합=E-COMP 합류만 잔여) — `E_AMB_..._PLAN §7~§7h` |
| **E-P1~3** | 21벤치 tier2 분해(P_OC §4·몫 안정성): E-P1 BFCL miss_param(P10 검증)·E-P2 NESTful·E-P3 CFB/WorkBench/ToolEmu | 새 non-empty C-류 0 수렴 가설 | 무료(로컬 생성·8141) | ▶ **사용자 승인(2026-07-10)·진행 중**: 공개 궤적 부재 확인(전부 데이터만)→로컬 생성 경로. **E-P2 실행 중**(NESTful 200샘플·QwQ 8141)·E-P1 스테이징 완료(gorilla clone·bfcl_eval import OK·데이터 v4 확보·생성은 T5-B 후 8140)·CFB/ToolEmu clone 완료 |
**GO 조건(공통)**: per-case 복구 ∧ over-block=0 ∧ **Δspurious ≤ 0** ∧ turn-예산 초과 0. pass 비교는 nt=4 후 *부차*.

## 5. 논문 4분할 (승인됨 2026-07-08 · 2026-07-09 특허 정련)
| 논문 | 담는 주장 | 상태 | 게이팅 | 특허 |
|---|---|---|---|---|
| **P1** *What Scale Buys* | C1·C2·C3a·C5·C8 | **[S] 즉시 출고** | (E5 강화) | **A**(게이트·calc) + **B**(배분·knee) |
| **P2** *Levers Interfere* (**모트**) | C4a·C4b·C4c·**C4d** | C4c=[D] | **E1** | ✅**B에 흡수(2026-07-10)**: 간섭-보상 합성 = B04 §6.10 실시예 + B09 종속항 5-1(우선일 확보). 독립 특허 C는 **분할출원 옵션**으로 유보(E1급 실측 확정 시 승격) |
| **P3** *The Semantic Boundary* | C3a·C3b·**C46**(날조 닫은 뒤 ⋈만 남음) | C3b=[M] | **E3** | B(배분 경계) |
| **P4** *Learned TBox Transfer* | C7 | [?] | **E6′**(데이터 v3 先·C38) | **A**(재학습0 전이) 실증 |
| **★P?** *Source Declaration* (신규 후보) | **C43(정박치환)·C45(출처선언 레버)·C48(ASK 위계)** | E11-e2e 게이팅 | **prov e2e** | ★**특허 D 후보**(출처선언+provenance 집행·DB 안읽음=present 차별) |
| P5 | A2 frontend | 범위 밖 | — | 후속 |
> ★**present/autofetch는 특허 A에서 제거**(C34 규칙0 위반). 출처선언 레버(C45)가 그 자리를 대체하되 **DB를 대신 읽지 않는다**는
> 점이 신규성(present 선행과 차별·[[46]]). **E11-e2e가 GO면 특허 D 우선 출원 후보.**

### 5.1 🔒 시퀀싱 제약 (특허 우선) — 2026-07-10 갱신(특허 C의 B 흡수)
특허 A/B 명세: **"논문 공개 전 출원 필수(신규성)"**.
⇒ **특허 A·B(간섭-보상 합성 포함: B04 §6.10 + B09 종속항 5-1) 출원 → P1 공개 → P2 공개.**
- 구 "특허 C 별도 출원" 단계 **폐지**: B 공개(~18개월) 후의 별도 C 신규 출원은 B 자체가 선행문헌이 되므로 "B 패밀리 안 또는 B 공개 전" 양자택일 — 현 합성 실측(+12.3pp [S] 1건·C4c=[D])은 독립 발명 기둥으로 얇고 종속항 재료로 적정 → 흡수가 우선일·비용 양면에서 우월.
- **분할출원 옵션 유지**: E1급 합성 실측 확정 시 B 출원 계속 중 분할로 독립 청구 승격 판단(변리사 확인 필요: 분할 시한·기재요건). **E1 > P2 집필** 제약은 "분할 승격 판단의 뒷받침"으로 유지.

## 6. 문서 지도 (마스터 → 하위)
- ★**명명 권위본(통일)**: `UNIFIED_TAXONOMY_2026_07_09.md` — **F1-F6·G1-G9·BC0-7·N1-N4 코드 폐기·서술형 이름만**(3축: 관찰·근본기능 11개·해결레버·4도메인). C51~C54의 F2/F3/BC 표현은 이 통일명으로 읽어라([[48]]). 상세근거=`CAPABILITY_BC_LEVER_TRADEOFF`·`DOMAIN_TRANSFER_STATUS_AND_PLAN`·`TAU2_FRONTIER_..._MASTER`.
- **★이론 정본(2026-07-10)**: `THEORY_AMBIGUITY_CLASSICAL_2026_07_10.md` — 애매모호성의 고전 정보이론 정식화(세-수준 분해 H(gold|φ(X))/H(gold|X)·카디널리티 삼분법·DPI/Fano·2축). 3계층·레버 배분을 연역으로 유도, 기존 [S]/[M] 실측의 retrodiction. **등급 [D] — 실측처럼 인용 금지.** 검증 설계 = `E_AMB_MEASUREMENT_PLAN_2026_07_10.md`(T1~T4 무료 재분석·T5 유료 승인). 덱 반영 = `_cdp_private_local/PATENT_BRIEF_DECK_2026_07_10_rev2_theory.pptx` 부록 C(103~110).
- **프레임 상세**: `MASTER_FRAME_LEVER_COMPOSITION_2026_07_08.md`
- **포트폴리오/실험맵**: `PORTFOLIO_ROADMAP_2026_07_08.md`
- **★출처선언 레버·날조 정본(2026-07-09)**: `C43_ANCHORED_SUBSTITUTION_NOT_WM_2026_07_09.md`(정박치환·WM 반증) ·
  `C44_SOURCE_DECLARATION_LEVER_2026_07_09.md`(4지선다+provenance·67→0%·Δspurious 0) ·
  `C49_ASK_HIERARCHY_NOT_BAN_2026_07_09.md`(ASK 위계) · `C42_SHORT_CONTEXT_SOLVES_FOURWAY_2026_07_09.md` ·
  `C38_INDIST_GATHER_RESULT_2026_07_09.md`(learn 데이터 실패) · `SCAFFOLD_AUDIT_RULE0_2026_07_08.md`(present/autofetch 폐기) ·
  `E11_GATHER_BEFORE_ACT_DESIGN` · `E6PRIME_GATHER_LEARN_DESIGN`. 스크립트: `c47_dprime.py`·`c48_dprime_full.py`·`e11a_isolated_probe.py`.
- **포렌식(정본)**: **`DB_ONLY_HARDCORE_FORENSIC_2026_07_08.md`(★DB-only 정본 분해·C22~C28·재현 `scripts/distill/tau2/dbonly_forensic.py`)** ·
  `HARDCORE_STEP_FORENSIC_2026_07_08.md`(**N1·N4 철회됨 — C24/C22 참조**) ·
  `QWQ_AGENTIC_FAILURE_FORENSIC_2026_07_08.md`(thinking 약점·§7c 게이트 실측) ·
  `CLEAN_NT4_FAILURE_FORENSIC_2026_07_07.md` · `PRESENT_G15_DET_CENSUS_2026_06_25.md` · `ASSEMBLED_FAILURE_FORENSIC_2026_06_27.md`
- **설계**: `THINKING_PERSISTENCE_SCAFFOLD_DESIGN_2026_07_08.md`(rev2·E1) · `LOAD_REDUCTION_ARCH_DESIGN_2026_07_07.md`(E1/E2 원형) ·
  `FLEET_FUNCTION_DELEGATION_DESIGN_2026_07_07.md`(E7) · `LEARNED_WING_MECHANISM_DESIGN_2026_07_07.md`(E6) ·
  `TESTTIME_COMPUTE_LEVER_DESIGN_2026_07_07.md`(thinking) · `NEXT_DET_LEVERS_DESIGN_2026_06_27.md`(over-action 게이트 금지 선례)
- **DR(선행)**: `RELWORK_LOAD_COT_2026_07_05`(symbolic/semantic) · `RELWORK_AGENTIC_HORIZON_2026_07_07`(조기종료·external verifier) ·
  `RELWORK_SCALE_LOAD_2026_07_07`(DR#2 horizon) · `40/41/42/43/44/45/46`(메모리 인용 정본)
- **수치 정본**: `_cdp_private_local/_WRITING_BRIEF §3` · `sim_results/*.compliance.json` · `sim_results/*.results.json.gz`
- **논문**: `papers/paper1_capability_scale_lever/what_scale_buys.md`
- **특허(로컬 전용·[[32]])**: `_cdp_private_local/sections/A*.md`(제1발명) · `B*.md`(제2발명)

## 7. 갱신 프로토콜 (표류 방지)
1. **작업 시작**: 이 문서 §0·§1·§2 읽기 → §4 큐에서 최우선 항목 선택. 큐 밖 작업은 **금지**(사용자 승인 시만).
2. **실험 종료**: (a) 수치를 해당 정본 doc에 **provenance와 함께 영속** (b) 이 문서 §3 원장 등급 갱신 (c) §4 상태 갱신.
3. **주장 인용**: 반드시 §3 등급을 달고. **[D]/[?]를 [M]처럼 쓰지 말 것.**
4. **프레임 변경**: §1은 LOCKED. 반증 측정이 있을 때만, 그 측정을 §3에 올린 뒤 개정.
5. **특허/논문**: §5 매핑과 시퀀싱 제약을 따른다. 새 결과는 먼저 §3 → 그 다음 특허/논문 반영.
6. **리뷰**: 외부/타 세션 리뷰도 [[08]]에 건다 — 인용된 선례·수치를 **직접 검증** 후 반영.
