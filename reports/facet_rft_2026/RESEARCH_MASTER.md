# RESEARCH MASTER — 등대 문서 (single source of truth)

> **이 문서가 최상위다.** 모든 실험·설계·특허·논문은 여기서 파생되고 여기로 되돌아온다.
> 표류 = 이 문서를 안 읽고 지엽에 들어가는 것. **작업 시작 전 §0·§1을 읽고, 작업 종료 시 §3 원장을 갱신한다.**
> 확정: 2026-07-08 (프레임 LOCK·논문 4분할 사용자 승인). **§1은 LOCKED — 새 측정의 반증 없이 재론 금지([[03]]).**

---

## 0. 한 줄 (논문 헤드라인 · 변하지 않음)
**Scale은 horizon을 사지, guarantee도 semantic reference도 사지 못한다.** guarantee는 결정론 scaffold가 pass-비용 0으로 산다.
symbolic 추론은 test-time compute가 싸게 산다 — **단 그것은 persistence를 판다**. 그러므로 레버는 독립 배분이 아니라
**측정된 상쇄에 의한 합성(composition)**으로 배치한다. **부작용 없는 레버는 없다.**
잔여(semantic reference)는 우리가 시험한 어떤 레버도 열지 못한 **경계**다.
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
| **operand 날조** | 없는 id 발명 | **결정가능·이미 집행중** | 32/32 도구가 거부 · 12/15 복구 후에도 실패 | **레버 아님**(증상 표지) — C12 | [M] |
| **over-action** | 안 시킨 write | LLM scope 잔여 | passing-spurious QwQ 0 vs base 47 | **게이트 금지**([[06]] 선례) · 반대편 계측만 | [S] |

### 1.4b 🔒 frontier 격차 전수 분해 (C15·성공-실행 호출 기준·456 sim)
| 원인 | ours | o4-mini | Δ | gpt-4.1 | Δ | 처방 축 |
|---|---|---|---|---|---|---|
| **NO-WRITE: 모든 시도 ERROR** | 16 | 1 | **+15** | 1 | **+15** | **repair**(유효 후보 제공·차단 아님) |
| **NL/communication**(db ok·reward 0) | 23 | 12 | **+11** | 16 | +7 | calc_NL/보고 (일부 벤치 아티팩트) |
| **F3 ⋈ 틀린 주문** | 37 | 30 | +7 | 16 | **+21** | 경계(map) + 탐색(reach) |
| **F2 wrong variant** | 32 | 26 | +6 | 24 | +8 | thinking/능력 (E1′가 판정) |
| OVER-ACTION | 5 | 0 | +5 | 2 | +3 | 게이트 금지축 |
| other operand(address1) | 6 | 2 | +4 | 3 | +3 | operand |
| MISSED-THIS-WRITE | 21 | 22 | −1 | 21 | 0 | — |
| other operand(payment) | 3 | 9 | **−6** | 5 | −2 | (우리 우세) |
| **NO-WRITE: 시도조차 안 함** | 8 | 19 | **−11** | 12 | −4 | (우리 우세·frontier는 기권) |
| **합계** | | | **+34 = 7.5pp** | | **+46 = 10.1pp** | |
- **★"F2가 유일한 격차"는 철회.** F2는 34 중 +6(18%).
- **★최대 조각 = "모든 write 시도가 ERROR"(16 vs 1)** — 날조를 *막는 것*은 무의미(환경이 이미 막음·C12), **유효 후보를 주어 write를
  *성사*시키는 repair**가 처방. **차단 ≠ 수리.**
- **★gpt-4.1 대비 ⋈이 +21(46%)** — "⋈은 frontier와 공유"는 H3-4 부분집합 착시였다(전 구간 37 vs 16 = 2.3×).
- **★A4 실증**: o4-mini는 *기권*(never-attempted 19)해서, 우리는 *틀리게 행동*(all-errored 16·over-action 5)해서 실패한다.
- caveat: sim당 **근인 1개**만 귀속(다중 원인 가능) · NL 버킷은 judge 의존 · frontier 파일은 공식 하네스.

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

### 1.7 ▶ 다음 실험 (이 표가 지시하는 것)
1. **★E1′ Phase A(무료·단일 결정 실험)** — **정보-맞춘 격리 replay**: 결정 지점에서 에이전트가 *실제로 갖고 있던* 대화·후보를
   짧은 clean 컨텍스트로 재구성해 재질문. **$>0.762$면 F2=부하(격리 회복)·$\approx0.762$면 F2=능력(15pp 격차·scaffold 무효)**.
   ⇒ **F2 행의 [M] 진단을 확정하는 유일 실험. E1′의 생사.**
2. E1 Phase B(게이트·실행중) — F4/F5의 closed 판정 + Δspurious.
3. F2가 능력으로 확정되면: thinking 격리천장(.864) vs frontier(.908) 간극 → fleet 재-scope or learn(C7).
   ※ 우리 scale 곡선(14B .732→32B .762 = +3pp/step) 외삽 시 72B≈.79 ⇒ open big-tier로는 못 메움(추정·[EST]).
4. **★C15 신설로 최우선 재지정**: 격차의 최대 조각 = **"모든 write 시도가 ERROR"(16 vs 1)**. 처방은 **차단 아니라 수리**
   (present/autofetch로 유효 후보를 주어 유효한 write를 *성사*시킴). E9(차단)는 죽었으나 **repair 레버는 미검증**.

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
| **C12** | **decidable ≠ 유용.** decidability는 *부작용 없음*의 필요조건이지 *이득*의 충분조건이 아니다(환경이 이미 집행 중일 수 있음) | **[M]** | 같은 doc §4b |
| **C10** | **레버 부작용은 scope에서 온다** — 전-궤적 thinking=persistence 매도 / 결정점 격리=채널 폐쇄 | **[D]** | 같은 doc §4 · **E1′가 검정** |

## 4. 실험 큐 (우선순위·상태)
| ID | 실험 | 닫는 것 | 비용 | 상태 |
|---|---|---|---|---|
| ~~E9~~ | operand grounding 게이트 | — | 무료 | ❌ **Phase A NO-GO**(환경이 이미 거부·근인 아님·passing 12 발화) |
| **E1′** | **격리 formalize 서브콜** — grounding된 후보 위 선택(날조+wrong-real-variant = 같은 결손의 두 얼굴) | **C9 잔여 전체** + C10 검정 | 무료→소액 | **▶ 최우선 복귀**(E9 사망으로) |
| **E1** | 완결/persistence 게이트 A→B→C (F4/F5) | C4c (더 작은 잔여·자기-역효과 보유) | 무료→소액 | ✅A(CONDITIONAL GO) · 🔄B 실행중 |
| **E3** | F3 경계 full-agent 확인 | C3b → **[S] 부분** | 무료 | ✅ 완료 (`E3_E1A_RESULTS`) |
| **E8** | frontier 격차 분해(horizon vs 기능) | **C9** | 무료 | ✅ 완료 (`HORIZON_GAP_DECOMPOSITION`) |
| E2 | QwQ+rparser nt=4 | C4b [M]→[S] | 유료(실행중) | 🔄 |
| E4 | base + 게이트 회귀(Δspurious 필수) | 게이트 일반성 | 소액 | E1 후 |
| E5 | 7B assembled | C2 사다리 | 소액 | 대기 |
| E6 | learn-wing four-bench→τ² swap | C7 | 큼 | E3 후 |
| E7 | fleet | C6 | 보류 | big-tier 시 |
**GO 조건(공통)**: per-case 복구 ∧ over-block=0 ∧ **Δspurious ≤ 0** ∧ turn-예산 초과 0. pass 비교는 nt=4 후 *부차*.

## 5. 논문 4분할 (승인됨 2026-07-08) · 특허 매핑
| 논문 | 담는 주장 | 상태 | 게이팅 | 특허 |
|---|---|---|---|---|
| **P1** *What Scale Buys* | C1·C2·C3a·C5·C8 | **[S] 즉시 출고** | (E5 강화) | **A**(게이트·present/calc) + **B**(배분·knee) |
| **P2** *Levers Interfere* (**모트**) | C4a·C4b·C4c·**C4d** | C4c=[D] | **E1** | **B 확장 → 특허 C 후보**(간섭-보상 배분) |
| **P3** *The Semantic Boundary* | C3a·C3b | C3b=[M] | **E3** | B(배분 경계) |
| **P4** *Learned TBox Transfer* | C7 | [?] | E6 | **A**(재학습0 전이) 실증 |
| P5 | A2 frontend | 범위 밖 | — | 후속 |

### 5.1 🔒 시퀀싱 제약 (특허 우선)
특허 A/B 명세: **"논문 공개 전 출원 필수(신규성)"**.
⇒ **특허 A·B 출원 → P1 공개 → (합성이 신규청구면) 특허 C 출원 → P2 공개.**
**E1 결과가 특허 C의 뒷받침이므로 E1 > P2 집필.**

## 6. 문서 지도 (마스터 → 하위)
- **프레임 상세**: `MASTER_FRAME_LEVER_COMPOSITION_2026_07_08.md`
- **포트폴리오/실험맵**: `PORTFOLIO_ROADMAP_2026_07_08.md`
- **포렌식(정본)**: `QWQ_AGENTIC_FAILURE_FORENSIC_2026_07_08.md`(thinking 약점·§7c 게이트 실측) ·
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
