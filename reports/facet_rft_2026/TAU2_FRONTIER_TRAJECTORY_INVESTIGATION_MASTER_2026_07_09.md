# tau / tau2 궤적 원인 전수조사 — 정본 기록 (raw → 과정 → 결과)

> **이 문서 = frontier·자체 모델 궤적 조사의 단일 진실원. 다시 재유도·재다운로드·"궤적 없다" 금지.**
> 작성 2026-07-09 (표류 종식용·사용자 지시 "raw부터 과정·결과까지 모두 기록·다시는 잊지 않게"). 등대 §3 C-원장·§1.4b/1.4c와 상호참조.
> ★**표류 원인**: 이 기록이 없어 "frontier 잔여가 F2인가 F3인가"·"신형 모델 compliance"를 *이미 했는데* 리더보드/궤적을 재탐색했다. **이 문서를 먼저 읽으면 반복 안 함.**

---

## 0. tau vs tau2 (혼동 방지 — 표류의 뿌리)
| | **tau-bench** (2406.12045·Yao·2024) | **tau2-bench** (2506.07982·Barres·2025) |
|---|---|---|
| control | single (에이전트만 행동·유저=수동) | **dual-control**(유저도 도구 행동) |
| 도메인 | retail·airline | retail·airline(이월) + **telecom**(신규·dual-control 무대) + banking_knowledge |
| 우리 벤치 | — | **tau2-retail** (사실상 single-control·유저 DB 안 바꿈) |
- ★**리더보드 2종을 섞지 말 것**: llm-stats "tau-bench retail"(원본·Sonnet 4.5 **0.862**) ≠ 우리 tau2-retail. 같은 "retail"이라도 **하네스 세대 다름**(재구현·더 어려운 user-sim). claude-3.7 tau 0.812 vs tau2 0.787.

---

## 1. RAW 데이터 인벤토리 (어디 있고·무엇이 소실됐나)

### 1.1 frontier baseline 4종 (gpt-4.1 user-sim) — ✅ 영구 보존
`external tau2-bench` 레포 `data/tau2/results/final/<model>_<domain>_default_gpt-4.1-2025-04-14_4trials.json`.
- 모델: **claude-3-7-sonnet-20250219 · gpt-4.1-2025-04-14 · gpt-4.1-mini-2025-04-14 · o4-mini-2025-04-16** × {retail·airline·telecom(+variants)}.
- 리모트 경로: `/home/woori/workspace_common/boltzmann-attention/external/tau2-bench/data/tau2/results/final/` (= `/home/woori/scratch/tau2-bench/.../final/`).
- 공개 출처: `github.com/sierra-research/tau2-bench` 동일 경로. **재현 가능.**

### 1.2 우리 런 (Qwen 2.5 agent) — ✅ 영속(git -f)
`boltzmann-attention-pi/reports/facet_rft_2026/sim_results/*.results.json.gz`(+`.compliance.json`):
- **asmregen32b_regen_retail_t4** / **asmregen14b_regen_retail_t4** (32B/14B + scaffold·replay-safe gate·clean nt=4·gpt-4.1 sim) — **정본 arm**.
- **fl32b_floor_retail_t4 / fl14b_floor_retail_t4** (floor base).
- **asmscale_32b_gpt52sim_retail_t4** / **...16k_contaminated** / **...32k_retry** (우리 32B·**gpt-5.2 user-sim**·유료·clean 병합 pass^4=0.333).
- ★scratch(`tau2-bench/data/simulations/`)는 **gitignored → 즉시 위 sim_results로 gzip+`git add -f` 영속**([[30]] 철칙).

### 1.3 ★신형 frontier (gpt-5.2 user-sim) — ⚠️ **raw 소실·수치만 생존**
- **누가**: Opus 4.5 · Opus 4.6/4.7 · Sonnet 4.5 · GPT-5/5.2/5.4/5.5 · Gemini-3-Pro/3.1 · GLM-5/5-2 · Grok-4.x · Qwen3-Max/3.5-397B · **ToolOrchestra** 등 (tau2 리더보드 제출 50+).
- **어디서**: `sierra-research/tau2-bench` 레포 `web/leaderboard/public/submissions/<model>/submission.json`. 각 submission은 `results.{retail,airline,telecom,banking_knowledge}.pass_1..4` + `trajectory_files`(파일명) + `trajectories_available:true`. **user_simulator = gpt-5.2**.
- **★raw 궤적 파일 자체는 레포에 없다**(submission.json=2185B·점수만). 실제 4trials 궤적은 taubench.com/S3 별도 호스팅. **한 번 다운로드→compliant^4 계산→수치추출 후 raw 미영속(소실)**. 리모트·로컬 전수검색 결과 **없음**. = [[30]] 데이터소실 재발.
- **생존한 것**: (a) **pass^1..4** = submission.json에서 **언제든 재취득**(무료·`raw.githubusercontent.com/sierra-research/tau2-bench/main/web/leaderboard/public/submissions/<dir>/submission.json`). (b) **compliant^4** = `_cdp_private_local/make_figs_results.py` `fr52` dict에만.
- **소실된 것**: 신형 모델의 **compliant^1..3**·**궤적 원인(F2/F3) 분해** — raw 없으면 재다운로드해야 가능.

---

## 2. 과정 (스크립트·방법)
| 스크립트 (repo `scripts/distill/tau2/`) | 무엇 | 기준 |
|---|---|---|
| **dbonly_forensic.py** | DB-only 원인 분해(core·buckets·classes·fabricate·reason) | `reward_info.db_check.db_match`(C19 arm-공통 축)·min-diff write 짝짓기 |
| **e10_precond_probe.py** | over-action 정책-precondition 격리검증(NO-GO) | 위 + db.json P1/P2 술어 |
| **f3_probe1/2** (본 조사·`/c/tmp/`·아래 §3.2) | per-arm 잔여 F2/F3 구성 + scale 추세 | 위 |
| **t2_compliance.py** | pass→compliant(정책위반 g1-g4 검사) | 궤적 필요 |
| **frontier_solve_kit.py**·`driver_frontier_ceiling/f4b.sh` | frontier 공개궤적 처리 | — |
| **make_figs_results.py** (`_cdp_private_local`) | pass^1..4 + compliant^1..4 그림(수치 하드코딩·No new runs) | BENCH/COMPL/fr52 |
- ★**dbonly_forensic는 리모트 미푸시**(로컬 ba-frft만). 리모트 실행 시 헬퍼 인라인(자기완결) 또는 push 필요.

---

## 3. 결과

### 3.1 frontier 격차 = 우리가 *어디서* 뒤지나 (DB-only 정본·C15·C21~C29·456 sim)
정본 doc `DB_ONLY_HARDCORE_FORENSIC_2026_07_08.md`. **vs o4-mini DB격차 +23**:
| 조각 | Δ | 처방축 |
|---|---|---|
| over-action(MORE+EXTRA) | +9 | 대화-precondition(C50 게이트 NO-GO)·[[06]] |
| ⋈ order_id | +8 | F3 경계 |
| item 집합 혼합 | +6 | write-scope |
| 주소 free-text 날조 | +5 | provenance(C24) |
| zero-write | +4 | repair |
| **F2 변형선택** | **−4** | (우리 우세) |
| payment_method | **−11** | (우리 우세) |
- vs gpt-4.1 +39: 미완 +17·zero +11·operand +14·과잉 −6 (frontier마다 구성 다름·o4-mini=기권형·gpt-4.1=실행형·A4).
- caveat: gold action_checks 노이즈 5~10%(C20)·근인 1개 귀속.

### 3.2 ★frontier 자신의 잔여 = F2인가 F3인가 (2026-07-09 본 조사·f3_probe1/2)
**질문**: F3가 scale-flat 경계라면 frontier도 F3에 막혀야 한다 — 사실인가?
**소스**: §1.1 baseline 4trials + §1.2 우리 arm · 기준 db_match · 456 sim.

**(a) best frontier(claude-3.7·db 0.80) db_fail=91 구성** (SAME-count wrong-content):
`F2 변형(new_item_ids) 18(최대) · op불일치 10 · item+new 8 · payment 6 · reason 5 · F3 ⋈ 5 · 주소 2`.
gpt-4.1 db_fail=102: `F2변형 19(최대)·op불일치 17·F3⋈ 10`. ours 32B db_fail=141: `F3⋈ 23(최대)·F2변형 17`.

**(b) 전체 456 대비 % + scale 추세**:
| 클래스 | ours 32B | o4-mini | gpt-4.1 | **claude-3.7** | 추세 |
|---|---|---|---|---|---|
| **F2 변형(new_item_ids)** | 3.9% | 4.4% | 4.2% | **3.9%** | **평탄=scale-불변 잔여** |
| **F3 ⋈(order_id)** | 3.1% | 3.1% | 1.5% | **0.9%** | **감소=scale가 산다** |

**★규정**: **frontier가 못 푼 잔여 = F2 변형(속성-변형 매칭)이지 F3가 아니다.** F3 order-⋈는 scale이 산다(3.1%→0.9%). composition이 scale로 뒤집힘(우리=F3지배·frontier=F2지배).
- **C3b("F3 ~.44 scale-flat 경계")와 긴장**: C3b는 *구성된 격리프로브*(후보 2+개 억지제시)·그 슬라이스는 flat하되 **agentic선 작다**. new_item_ids=속성→변형 매칭=**fine-grained ⋈**(order-level 아님) → 경계는 *coarse 참조*(scale가 삼)가 아니라 *fine 속성매칭*.
- ⚠️ **reach/boundary 프록시 오염**: gold order_id가 `get_user_details` 목록에 항상 등장 → reach=0으로 나옴(느슨). E3 정밀기준(주문 *상세* 조회)이라야 43~52% 경계.
- ~~⚠️ 미측정: 신형 F2/F3 분해 미실행~~ → **✅ 측정 완료(§3.2b·2026-07-09 밤)**: raw를 S3서 재다운로드해 8개 top frontier 전수 분해. 외삽 아니라 측정.

### 3.2b ★신형 frontier 전수 F2/F3 분해 (측정·gpt-5.2 user-sim·2026-07-09)
**raw 재취득**: S3 `https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com/submissions/<dir>/trajectories/<trajfile>`(§1.3 URL 확정). **manifest**=`.../submissions/manifest.json`. 스크립트=`scripts/distill/tau2/frontier_function_decomp.py`(로컬·CPU-only). raw 40MB×8=`/c/tmp/traj/`(transient·재다운로드 가능).

전 20 standard submission 중 retail 궤적 보유 top 8 (GLM-5.2 85.7·RAFT 82.5는 S3 궤적 404=제외):
| model (gpt-5.2 sim) | db-pass | db_fail | **F2변형%** | **F3⋈%** | top 잔여 클래스 |
|---|---|---|---|---|---|
| **Qwen3.5-397B-*think*** | 0.855 | 66 | **0.4** | 0.9 | reason_enum 6·op 4·F3 4 |
| GPT-5.2 | 0.825 | 80 | 3.3 | 0.7 | **F2 15**·payment 9·reason 8 |
| Gemini-3-Pro | 0.821 | 78 | 2.4 | 1.8 | **F2 11**·F3 8·payment 8 |
| Claude Opus 4.5 | 0.816 | 84 | 2.6 | 0.7 | **F2 12**·reason 5·F3 3 |
| Gemini-3-Flash | 0.787 | 97 | 2.0 | 1.1 | **F2 9**·payment 8·reason 5 |
| GPT-5.2-none | 0.770 | 93(**infra 51**) | 3.5 | 0.4 | **F2 16**·op 9·reason 8 |
| GLM-5 | 0.752 | 112 | 3.3 | 0.2 | **F2 15**·reason 6·payment 6 |
| Claude Sonnet 4.5 | 0.741 | 118 | 2.2 | 0.9 | **F2 10**·reason 9·F3 4 |

**★규정 (측정·확정)**:
1. **거의 모든 top frontier에서 F2 변형(new_item_ids)이 최대 잔여 클래스**(F2 9~16건)·**F3 ⋈는 전부 <1.8%**(대개 <1%). ⇒ **신형에서도 frontier가 못 푼 잔여 = F2 변형, F3 아님.** §3.2(gpt-4.1 sim baseline)와 다른 user-sim 체제서도 **동일 결론**(체제-불변).
2. **★Qwen3.5-397B-think(챔피언·0.855)만 F2=0.4%**(유일하게 F2 거의 닫음). = **F2 symbolic operand은 test-time compute(thinking)가 산다**는 명제(§1.2 F2 레버=thinking·C13)의 **실측 확증**. 비-think/약한 think 모델은 F2 2~3.5% 잔존.
3. F3 ⋈는 규모·reach로 대부분 해소(전 top <1.8%) = §3.2 "F3는 scale가 삼" 재확인. 경계는 order-⋈ 아니라 fine 속성매칭(F2 변형).

**caveats([[08]])**: (a) **user-sim 2체제**(baseline=gpt-4.1·신형=gpt-5.2) → 절대 pass 병치 금지·F2/F3 *구성*만 비교. (b) **reason_enum**(전 모델 5~9건)=arm-공통 하네스 노이즈(C28·user-sim이 gold와 모순·레버 아님). **payment**=우리 우세(C23). **op_mismatch**=상류 날조/user-sim 이탈(C21). ⇒ 진짜 능력 잔여=**F2 변형**. (c) GPT-5.2-none infra 51(reasoning=none이 크래시多)·gemini3pro infra 20 → 그만큼 F2/F3 분모 축소(경미). (d) GLM-5.2(85.7)·RAFT(82.5) 궤적 미확보(S3 404).

> ★**신형 retail 재확인 대상 없음(재드리프트 방지)**: retail/airline/telecom은 **2026-02-26 라운드**(위 top-8이 전부)·**2026-05-05 라운드(Opus 4.6/4.7·GPT-5.4/5.5·Grok 4.x·Gemini 2.5/3.1·Distyl)는 신규 `banking_knowledge` 도메인 *하나만* 제출**(retail·airline 미제출). ⇒ **retail에서 Opus 4.5(0.816)·Qwen3.5-397B(0.855)보다 새 frontier는 존재하지 않는다.** banking_knowledge=τ³ 지식-검색 축(frontier 12~37%·GPT-5.5 37.4 최고)·우리 F1-F6(operand tool-use) 프레임 밖. "신형 모델로 retail F2/F3 재확인" = **불가·불필요**(제출 자체가 없음).

### 3.2c ★교차-도메인 기능 불변성 → 도메인-일반 규칙 (banking_knowledge·2026-07-09)
**목적**: TBox=도메인-불변 규칙을 찾으려면 *다른 도메인*서 같은 실패 기능이 재현되는지 봐야 한다. banking(신규 τ³ 도메인·KB검색+도구발견/unlock+신원검증+shell·gpt-5.2 sim)을 기능-사상 분해(`opus45/gpt55_banking.json`).

**banking 실패 구성** (db_fail·per-sim divergence·gold 절차 median **8단계**):
| divergence | Opus 4.5 (db 0.245) | GPT-5.5 (best 0.384) |
|---|---|---|
| **MISS_P_reach** (도구 발견/unlock 실패) | **48.5%** | 23.8% |
| **MISS_operand_write** (coverage·필수 write 누락) | 26.3% | 20.1% |
| EXTRA_read (KB 과탐색·비인과) | 98% | 100% |
| MISS_F1_verify (신원검증 누락) | 6.5% | 1.3% |
| **OPERAND arg 불일치** (retail의 F2) | **0** | **0** |
- per-case 검증: t032=필요 도구 0 unlock하고 escalate · t043="카드 해지" gold 14단계인데 3/7 도구만 unlock·6/14 단계만 = **장기 절차 부분완수**. ⇒ MISS_P_reach 진짜.

**★교차-도메인 기능 표 (도메인-일반 규칙 도출)**:
| 기능 | retail top frontier | banking top frontier | **도메인-불변?** → 규칙 |
|---|---|---|---|
| **F1 verify/compliance** | confirm/auth 게이트(g2) | log_verification(1~6%) | **✅ 불변** → *확인-후-실행* 게이트 |
| **reach/절차 조립** | 소(C14 ~3pp gather-before-act) | **지배(24~48%·8단계)** | **✅ 불변** → *선행자원 조립/gather* controller |
| **F4 coverage/완결** | 일부(FEWER) | **무거움(20~26%)** | **✅ 불변** → *전-단계 완결* 게이트 |
| **F6 horizon** | ~0(p_step 1.0·1~2 write) | **무거움(8단계 절차)** | 도메인-의존 stress → scale/분해 |
| **F2 symbolic operand** | **지배(변형)** | **~0** | 도메인-의존 stress → thinking |
| F3 semantic ⋈ | 소 | KB grounding(별도) | 양쪽·형태 다름 |

**★규정 (명제 핵심)**:
1. **도메인-불변 실패기능 = {F1 verify · reach/절차 조립 · F4 coverage}** — retail·banking **양쪽서 재현**. 이 셋이 정확히 **결정론 scaffold(게이트+controller) 타깃**이고 **도메인-일반 TBox 규칙 후보**(같은 추상 규칙·ABox만 교체). ⇒ 특허 "도메인-일반 게이트·계획 정책" 실측 지지.
2. **도메인-의존 *능력* stress는 다르다**: retail=F2 operand(→thinking)·banking=reach+coverage+horizon(→controller/분해/scale). **단일 "그" 잔여 없음** — 도메인마다 binding 축이 다르나 **scaffold 기능은 불변**.
3. banking은 **retail이 못 보여준 F6 horizon(8단계)·reach를 실측** — 우리 프레임 F1-F6이 도메인-일반임을 교차검증. operand precision(F2)은 retail 특이 stress.

**caveats([[08]])**: banking 액션모델 이색적(discoverable tool·shell·KB)·기능사상 근사(단 per-case 2건 확인). EXTRA_read ~100%=검색탐색·비인과(down-weight). db_fail 62~75%=신규 난도(db-only 엄격·장기절차). GLM-5.2/RAFT banking 미취득.

### 3.2d ★★일반 능력 기능 fine 분해 + 구제방법 일반화 (4도메인 전수·2026-07-09)
**동기(사용자)**: F1-F6은 거칠다. 구제방법을 *일반화*하려면 실패를 leaf 서브기능까지 결정론 분해하고 4도메인서 재현 확인해야 한다. 스크립트 `fine_function_decomp.py`(도메인-불가지·액션 기능클래스 패턴사상 + 발산arg 타입분류 + 메타신호). ★**도메인마다 reward 온톨로지 다름**(retail/airline=DB·banking=DB+절차·**telecom=ENV_ASSERTION**=기기상태·dual-control) → **reward 통일기준** 사용.

**★전수 sweep** (retail 8·airline 8·telecom 8·banking **17** 모델·pooled·% of domain fails·`aggregate_fine.py`):
| leaf 기능 | retail(n=858) | airline(338) | telecom(477) | banking(4955) |
|---|---|---|---|---|
| **G1 COVERAGE (전 단계/항목/결함 완수)** | **52%** | **52%** | **128%**(다중결함) | **22%** |
| **G2 REACH (gather/discover/plan)** | 18% | 2% | 6% | **38%** |
| **G6 OPERAND-symbolic (변형/값)** | **30%** | **28%** | 0 | 0 |
| **G7 REFERENCE ⋈ (엔티티 결속)** | 9% | **23%** | 0 | 0 |
| **G4 PERSISTENCE/ESCALATE** | 10% | **28%** | **59%**(결함남고 사람연결) | 12% |
| **G5 SCOPE/over-action** | 5% | 21% | *(미포착)* | ~0 |
| **G3 VERIFY** | **0**(인증-skip 0) | **0** | — | 5%(명시 log) |
| **HORIZON (gold-act med)** | 4~5 | 4.5~5 | 다단 | **9** |
- 2모델 예비값은 full-sweep에 포함·대체. per-case 검증(retail C51·banking t043·telecom MMS/refuel).
- **★VERIFY 정밀화(보완)**: frontier는 **write 전 인증을 항상 수행**(retail 48/48·airline 22/22·인증無 **0** 실측) ⇒ *인증-skip*은 frontier 실패축 아님. 준수 잔여=*더 미묘한 confirm-before-write*(§3.3 compliance-drop −5pp·별도 계측)·banking만 log_verification 명시 gold(5%).
- **★telecom 정밀화(보완)**: COVERAGE만 아니라 **PERSISTENCE 59%**(결함 남은 채 조기 transfer_to_human)·REACH 6% 포착.
- **★불변 확정**: **G1 COVERAGE + G4 PERSISTENCE = 전 4도메인 불변**(52/52/128/22 · 10/28/59/12%). 둘 다 **결정론 게이트로 닫는 도메인-일반 구제**.

**★일반 능력 기능 → 구제방법 매핑 (일반화)**:
| G | leaf 기능 | 구제방법 (일반화) | 도메인-불변? |
|---|---|---|---|
| **G1** | **COVERAGE** — 전 단계/항목/결함 완수 | **완결 게이트**(읽기강제·write금지) | **✅✅ 전 도메인(최강)** |
| **G2** | **REACH** — 선행자원 gather/도구 discover·unlock/plan | **결정론 controller**(gather강제·도구발견·plan↔exec 분리) | ✅ (banking 강) |
| **G3** | **VERIFY/COMPLIANCE** — auth·확인·신원검증 | **결정론 verify 게이트** | ✅ |
| **G4** | **PERSISTENCE/ESCALATE** — 조기 포기/과잉 escalate | **persistence 게이트** | ✅ |
| **G5** | **SCOPE/over-action** — 비요청·불가 행동 | **precondition 게이트/경계** | ✅ |
| **G6** | **OPERAND-symbolic** — 변형/값/계산 선택 | **thinking + calc offload** | 도메인-의존(transactional) |
| **G7** | **REFERENCE ⋈** — 올바른 엔티티 결속 | **경계/learn**(hard residual) | 도메인-의존 |
| **G8** | **HORIZON** — 장기 절차 지속 | **분해+상태 controller+scale** | 도메인-의존(banking/telecom) |
| **G9** | **GUIDANCE** — 사용자에게 행동 지시(dual-control) | user-instruction scaffold | telecom-only |

**★규정 (구제 일반화의 핵심)**:
1. **G1-G5(COVERAGE·REACH·VERIFY·PERSISTENCE·SCOPE) = 도메인-불변** → **각각 *하나의* 도메인-일반 결정론 구제(게이트/controller)·ABox만 교체.** = TBox/scaffold. 4도메인 교차 재현이 이를 실증.
2. **G6-G8(OPERAND·REFERENCE·HORIZON) = 능력축·도메인마다 stress 다름** → 구제 유형은 고정(thinking/offload/scale/경계)이나 *어디에 거나*가 도메인 의존. retail/airline=OPERAND, banking=REACH+HORIZON, telecom=COVERAGE(다중결함).
3. **★COVERAGE(G1)가 4도메인 전부의 최강 도메인-불변 잔여** — 종전 프레임의 F2/F3 강조를 넘어 **완결 게이트가 가장 일반화 가능한 구제**임이 교차도메인서 드러남.
4. G9 dual-control guidance = telecom 특이(신규 축).

**caveats([[08]])**: (a) **capture 한계(0=부재 아님)**: telecom 분해는 env-assertion(=COVERAGE)만 포착 → G2-G7 "0"은 미포착이지 부재 아님(telecom도 reach/guidance/escalate 有). retail/airline **VERIFY(G3)=0**은 확인이 *게이트*로 일어나 gold action_checks에 없어서(banking만 log_verification이 gold) → 미포착. (b) **MISS_write(coverage)**는 errored-write와 진짜 coverage-miss 혼재(heuristic)·단 retail OPERAND(F2)는 C51 per-case 검증. (c) reward 통일기준(NL/env 포함)이라 dbonly보다 관대. (d) glm52·distyl 궤적 S3 404. **표본**: retail/airline/telecom 각 8모델·banking 17모델·pooled(fail 가중).
**gpt-4.1 user-sim** (BENCH=pass / COMPL=compliant):
| arm | pass^1 | ^2 | ^3 | ^4 | compl^1 | ^2 | ^3 | ^4 |
|---|---|---|---|---|---|---|---|---|
| Claude-3.7 (fr) | .787 | .693 | .634 | .597 | .737 | .611 | .537 | .500 |
| GPT-4.1 (fr) | .741 | .642 | .579 | .526 | .715 | .602 | .533 | .482 |
| o4-mini (fr) | .715 | .594 | .517 | .456 | .693 | .572 | .500 | .439 |
| GPT-4.1-mini (fr) | .660 | .529 | .443 | .386 | .634 | .498 | .414 | .360 |
| **32B+scaffold (ours)** | .640 | .504 | .423 | .360 | **=bench (drop 0)** ||||
| **14B+scaffold (ours)** | .588 | .430 | .336 | .272 | **=bench (drop 0)** ||||
| 32B floor | .557 | .411 | .357 | .333 | .509 | .360 | .300 | .263 |
| 14B floor | .425 | .273 | .193 | .149 | .327 | .203 | .145 | .114 |
- **★모트 = compliance-drop**: 모든 frontier가 준수 낙폭(Claude-3.7 −5.0pp@pass^1 등)·**우리 게이트만 drop 0**. bench-pass는 frontier가 ~9pt 위(우리 pass 미달)지만 **우위는 pass 아니라 준수보장**. crossover(준수-pass): 14B+scaffold 0.336 > 32B floor-compliant 0.300(전 k).

**gpt-5.2 user-sim** (더 어려움·우리 32B+scaffold pass^1..4 = .591/.465/.393/.367·clean-merge pass^4 **0.333**):
- 신형 frontier **pass^4 → compliant^4** (`fr52`·공개 S3 궤적·raw 소실): **GPT-5.2 .518→.500 · Opus-4.5 .518→.500 · Gemini-3-Pro .474→.447 · Sonnet-4.5 .395→.386 · GPT-5.2-none .290** · GLM-5 .439 · Qwen3.5-397B .597.
- ★**신형 compliant^1..3은 미보유**(raw 소실). pass^1..4는 submission.json서 재취득 가능(예 아래).

**신형 pass^1..4 (submission.json·gpt-5.2 sim·재취득본)** — Claude Opus 4.5:
- retail **79.61 / 67.40 / 58.77 / 51.75** · airline 84.0/77.67/73.5/70.0 · telecom 92.32/86.11/81.36/78.07.
- (다른 신형은 동일 경로 submission.json에 pass_1..4 존재. 필요시 그 파일만 취득.)

### 3.4 리더보드 (권위·2026-07)
- **tau2-bench overall**(codesota): Opus 4.5 ~0.79 > GPT-5.2 ~0.73 > Sonnet 4.5 ~0.63. **챔피언=Claude 계열**(GPT/GLM 아님). retail 분리값은 submission.json이 권위본.
- **tau-bench retail**(llm-stats·원본): Sonnet 4.5 **0.862** > Opus 4.1 0.824 > claude-3.7 0.812 > o4-mini 0.718 > GPT-4.1 0.680.
- ⚠️ **버린 값(sanity 실패·[[40]])**: pricepertoken "GLM-5.2 99.1%(tau2)"·benchlm "telecom Opus 4.6 0.993"·"Mythos 5" = 비신뢰·인용금지. taubench.com=권위본이나 JS렌더로 스크랩 불가.

### 3.2e ★★모델별 전수 분해 (per-model·not pooled·2026-07-10)
**동기(사용자)**: pooled(§3.2d)를 넘어 **모든 모델의 실패 분해를 다 비교**하라. → 스크립트 `permodel_fine.py`(=`fine_function_decomp.decomp()` 모델별 호출·G1-G7 %of-fails). 산출 `_cdp_private_local/permodel_fine.json` · 그림 `fig_permodel_decomp.png`(4도메인 히트맵·모델 행×기능 열).

**표본**: retail 8 · airline 8 · telecom 8 · banking 17 모델(pooled 아님·모델 개별). 값 = 각 모델 실패의 % (leaf→G 매핑 §3.2d와 동일).

**★핵심(그림이 말하는 것)**:
1. **COVERAGE(완결)·PERSISTENCE(중단판단) = 모델-불변 AND 도메인-불변**: 모든 모델·모든 도메인서 두 열이 뜬다(retail COVERAGE 32~68 / airline 30~73 / telecom 10~174 / banking 10~26 · PERSIST 전 도메인 유). ⇒ **단일 도메인-일반 게이트**로 닫을 잔여가 규모·모델 무관 실재.
2. **OPERAND·REFERENCE = retail/airline 전용**(telecom/banking 0=capture-limited). airline엔 **모델별 스파이크**(gemini3pro OPERAND 80·REF 56 · qwen35 REF 103) = 도메인-특이 stress가 모델마다도 다름.
3. **REACH = banking 지배**(10~62·전 모델) · telecom = COVERAGE(다중결함)+PERSIST 지배.
4. **VERIFY ≈ 0**(banking만 명시 gold 1~14%) — §3.2d 정밀화(인증은 게이트로 일어나 gold action_checks에 없음) 재확인.

**caveats([[08]])**: telecom cell >100%=다중결함/케이스 · telecom·banking의 SCOPE/OPERAND/REFERENCE "0"=reward 온톨로지(env-assertion·discoverable-tool) capture-limit이지 부재 증명 아님. reward 통일기준(관대). glm52·distyl 미취득. **결론 방향은 pooled(§3.2d)와 일치·모델별로도 재현**.

### 3.2f ★banking 저-pass 근본원인 전수 포렌식 (17모델·2026-07-10·[[08]] 완주)
**동기(사용자)**: banking은 frontier도 pass 0.098~0.374로 매우 낮다 — 왜인가. 스크립트 `banking_forensic.py`.

**(1) 종료사유 [M]**: crash 아님 — user_stop 5923(92%)·too_many_errors 405(glm5 197·qwen397B 208에 집중)·max_steps 122·infra 65. ⇒ 대다수는 **정상 대화 종료 후 채점 실패**(agent가 끝났다고 믿음). glm5·qwen35의 최저치는 하네스 오류 성분 포함(아티팩트 down-weight).

**(2) 태스크-레벨 교차표 [M]**: 97태스크 중 **28개(29%) = 17모델×4trial 전패(universal-fail)**·≤2모델 통과까지 36%·17/17 통과 4개뿐. universal-fail 28/28 전부 `unlock_discoverable_agent_tool→call` **발견 체인** 요구·horizon med 12.5(전체 8.0).

**(3) 부하 기울기 [M]**: pass는 gold 길이에 단조 급락 — 1-3act **0.442** → 4-6 0.187 → 7-9 0.133 → **10+ 0.079**(최대 버킷 n=2546). unlock/discover 요구 유 0.132 vs 무 0.468(3.5×). basis=['DB'] all-or-nothing.

**(4) 실패 2형 분해 (universal-fail 실패궤적 n=1005·gold-체인 이름 커버리지) [M]**:
- **완주-후-불일치형 45%**(커버리지≥0.8·Q3=1.00 — 4분의 1은 체인 100% 실행): gold 액션을 이름 수준으로 다 하고도 최종 DB 불일치 = 인자 수준 오류(다계좌 중 오선택·금액)·추가/누락 write·순서. per-case: gpt55 task_049 = gold 19중 대부분 실행·reward 0.0.
- **발견/조기중단형 31%**(커버리지<0.5): unlock 체인 미발견(조립)·조기 transfer(중단판단).
- 나머지 24% 중간.

**★규정**: banking 저-pass = **3중 부하의 곱** — 긴 horizon(med 8~12.5) × 발견 체인(80% 태스크) × all-or-nothing DB 채점 ⇒ per-step p<1의 지수 붕괴(p^H·이론 '지속' 축의 실측 극단). 원인 기능 = 지속×조립 + 인자 정밀(계산/참조-기준형·실행규율) + 완결(자기검증: user_stop으로 '다 했다' 종료). 레버 사상 = gather/unlock controller + 완결게이트[체크리스트] + calc/provenance(인자) + persistence 게이트. **아티팩트 성분 정직 표기 [?]**: all-or-nothing 채점·4-trial 변동으로 벤치 설계 자체가 가혹 — 커버리지 1.00 실패의 일부는 gold 모호 가능성(2~3건 DB-diff 정독으로 확정 필요·미실행).

---

## 4. 확정·미측정 정리
- **[측정·확정]** frontier 잔여=**F2 변형**(§3.2 baseline gpt-4.1 sim + **§3.2b 신형 8개 gpt-5.2 sim·둘 다**)·F2=thinking이 삼(Qwen-think 0.4%)·F3 ⋈는 scale가 삼(<1.8%)·compliance drop 모트(§3.3)·격차 분해(§3.1)·챔피언 Claude/Qwen 상위(§3.4). **전부 재현 가능**(baseline 공개·우리 arm sim_results·**신형 S3 재다운로드**).
- **[★raw 재취득 = 해결·[[30]] 소실 재발 방지]** 신형 궤적은 S3 상주: base `https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com/submissions` · `manifest.json`으로 dir 목록 · 각 `submission.json.trajectory_files.retail` · `<base>/<dir>/trajectories/<trajfile>`. 스크립트 `frontier_function_decomp.py`. **더는 "소실"이 아니다 — URL이 이 문서에 영속됨.**
- **[잔여 미취득]** GLM-5.2(85.7)·RAFT(82.5) 궤적 S3 404 · 신형 **compliant^1..3**(pass^4→compliant^4만 fr52에·pass^1..4는 submission.json).
- **[하지 말 것]** "궤적 없다·유료런 필요·리더보드 재검색" 재단정 금지 — **재탐색 전 이 문서 §1·§3.2b 먼저.** ([[47]])

## 5. 재현 커맨드
- frontier 잔여 F2/F3: `f3_probe1/2` 로직(§3.2·본 doc)·헬퍼 인라인·`seka_env python` on remote.
- DB-only 격차: `dbonly_forensic.py --stage {core,buckets,classes,fabricate,reason}`.
- compliance 그림: `_cdp_private_local/make_figs_results.py`(수치 하드코딩·No new runs).
- 신형 pass^1..4: `curl raw.githubusercontent.com/.../submissions/<dir>/submission.json` → `.results.<domain>.pass_1..4`.
