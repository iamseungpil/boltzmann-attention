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
| **C9** | frontier 격차 = **horizon 아님**($p_{step}pprox1.0$) · **⋈ 경계 아님**(frontier와 공유 14 vs 12) · H3-4 write-arg 결손에 집중. ★**기전 라벨 정정**: criterion 아니라 **operand 날조**가 지배(7 vs 5) | **[M]** | `HORIZON_GAP_DECOMPOSITION` + **정정** `E9_OPERAND_GROUNDING_DESIGN` §0 |
| **C11** | **operand 날조 = frontier-분리 결손** — 우리 5.9% vs gpt-4.1·claude-3.7 **0.0%**(o4-mini 0.2%) · **scaffold도 scale도 미커버** · 술어가 **decidable** | **[M]** | `E9_OPERAND_GROUNDING_DESIGN` §1 |
| **C10** | **레버 부작용은 scope에서 온다** — 전-궤적 thinking=persistence 매도 / 결정점 격리=채널 폐쇄 | **[D]** | 같은 doc §4 · **E1′가 검정** |

## 4. 실험 큐 (우선순위·상태)
| ID | 실험 | 닫는 것 | 비용 | 상태 |
|---|---|---|---|---|
| **E9** | **operand grounding 게이트** — write id 인자의 레지스트리 멤버십(decidable·오탐 구조적 0) | **C11** = frontier-분리 결손(상한 +3.3pp[EST]) | 무료→소액 | **▶ 최우선 신설** ([[05]] 3문 통과) |
| **E1′** | 격리 formalize 서브콜 (criterion 조각·C10 검정) | 초과 3 (작음) | 무료→소액 | 강등·후순위 |
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
