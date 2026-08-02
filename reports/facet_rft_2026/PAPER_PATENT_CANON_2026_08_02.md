# 논문·특허 정본 지도 — 최신-우선 정렬·주장 추림·폐기 명부 (2026-08-02)

> **목적(사용자 지시)**: 특허·논문 입장에서 설계서·실험결과 문서를 정리하고 최신 문서 위주로 정렬한다.
> *"이전 문서들 때문에 폐기된 내용을 가지고 헷갈리고 있다. 최신 결과로 단순화하고, 주장할 부분만 추려라."*
> **지위 = 권위 위계 3층**(위계 정본 = 등대 헤더 🔒표·2026-08-02 고정): 등대(1층) > 주제별 정본 설계서(2층·§6 표) >
> **이 문서(3층·실물 지도)** > 로컬 기밀(4층) > 메모리(5층·포인터만). 충돌 시 등대 §5·§1이 위다.
> **★갱신 라인 결정(2026-08-02 사용자 승인)**: 논문 = **3편 통합**(what_scale_buys·declfirst[restate·feedback 흡수]·interp) /
> 특허 = **A·B 출원 라인 확정**(구 C=B 흡수 유지). ⚠**우산 처리(별도 vs A·B 흡수: ask-트리거→A·경계도출+triage→B·
> 분할 옵션) = 결정 대기(08-02 오후)**. §3·§4의 개별 행보다 이 결정이 우선한다.
> **명명 규율([[48]] 동형)**: 논문을 "P1/PAPER1" 같은 **번호로 부르지 말 것** — 세 체계가 같은 번호로 다른
> 논문을 가리킨다(§4 충돌표). **경로-이름**(declfirst·restate·…)과 특허 **문자**(A/B/우산/D후보)로만 부른다.

---

## §1. 지금 주장 가능한 것 (추림 · 증거등급 · 어디에 쓰나)

**전역 규율**: 런-간 pass 점수 비교는 전부 [D](C215/C217·Y2-C 검정력 실측 — n=32 짝으로는 0.5 효과에 짝 ~530 필요).
⇒ **주장은 per-case 인과사슬·기전 소멸·통제 대조 [S]/[M]만**. 아래 표가 전부이고, 표 밖 옛 수치는 §2를 먼저 보라.

| # | 주장(한 줄) | 등급 | 출처 | 쓰는 곳 |
|---|---|---|---|---|
| 1 | **선언-우선 봉투율 96%**: two-pass 선언이 A 32%<B 47%<C 78%<D 96%(드롭 보정 구간 비중첩)·대본 소비 100%·지평 초과 0 | [S] | C250(X13 144런) | 특허 A 실시예·우산·declfirst |
| 2 | **선언의 대가**(모트 실시예): 봉투를 사면 행동량을 판다 — 호출/런 10.5→6.1·multi-act 45→3 | [S] | C250 | 특허 B(간섭-보상)·declfirst |
| 3 | **QUOTE_PIN 인과사슬**: 유일-변경 arm에서 5런-전패 태스크 통과·도구층→db_match 전 구간 인과 확인 | [S] | C282 | 특허 A(접지 게이트) 실시예 |
| 4 | **3중층이 false-apply를 라이브 포획**(서브가 핀을 반대쪽에서 복사→1층이 차단) | [S] | C289 | 특허 A 실시예 |
| 5 | **게이트 자신의 역효과 실증**: 우리 quote-ground 가드의 false-abstain이 회수 손실을 만듦(022 ba8b) | [S] | C275 | 제1원리(§1.3)·특허 B 배분 근거 |
| 6 | **파서가 "미실행"을 만든다**: 정지 실패→완결 호출까지 텍스트 강등(42건/16sim)·살리기 회수 38/38·오탐 0 | [S] | C248 | 엔진 위생·restate 배경 |
| 7 | **채점 진단층 아티팩트**: 4런 gold-miss의 24.5%가 문자열-리터럴 비교의 가짜(판정은 무결) | [S] | C274 | 측정 위생(덱 정직 섹션) |
| 8 | **측정 위기 자체가 결과**: user-sim이 시나리오 재생성(유사도 0.77·20/32 변경)·pass 검정 불가 실측 | [S] | C215·C272·C249 | 덱·모든 논문의 방법론 절 |
| 9 | **검증-불가능성 축**: 결정가능 케이스 오류 0/98 — 문제는 오류율이 아니라 env로 검증 불가한 수치의 비중 | [M] | C225(X2 v1·감사 2회) | restate (v2 재계산 선결) |
| 10 | **피드백 등급은 클래스-의존 + 단조-상세의 반례**(enum: g1 6/6, g2 0/6 퇴행) | [M] | X5 r2-3 | feedback |
| 11 | **프록시 술어 3종 열림 — A2 데이터도 열림**("A2로 옮기면 닫힘/도메인-일반"의 반증) | [M] | C225(X4·Wilson LB .65~.78) | AXIS·닫힘 배터리·특허 A ADDENDUM ① |
| 12 | **공용 primitive 흡수율 100%·callback 0**·엔진 도메인 리터럴 0(banking A2-swap) | [M] | C239(+C231~C237) | 특허 B(capex/opex 분리)·what_scale_buys |
| 13 | **A(u) 정확도 스케일 불변**(32B도 1/3) — scale-invariant 잔여 축 실측 보강 | [M] | C247 | what_scale_buys |
| 14 | **frontier 잔여의 compliance-drop 모트**·챔피언 Claude 계열([[47]] 정본) | [S] | TAU2_FRONTIER…MASTER_2026_07_09 | what_scale_buys·덱 |
| 15 | **지시-결함 5분류(D1~D5)+env 1종**: 폭주 114/9,957 per-step 전수에서 우리 지시 몫을 셈 | [S](포렌식) | C290 | 엔진 위생·덱 |
| 16 | 정책 OR 닫기=**EGCWA**(비용은 K=2~3에서 ~0)·**EGCWA=집행 규칙 아닌 질문 트리거**·잔여=UNA뿐 | 종합(선행 위) | C285·C287([[51]][[52]]) | interp 논문·게이트 의미론 |
| 17 | 엔진-이관 3조건은 낱개 전부 선점(Reiter 동형)·**모트는 "측정+할당 결정" 합성**·(i)="finitely foldable"로 수정 | 종합(선행 위) | C281·C283([[50]]) | interp·AXIS·특허 관할 판정 절차 |
| 18 | 매핑 화이트스페이스 **PARTIALLY TRUE** — ①+②+③ 결합 sweep 부재·foil 5편 명시 인용 의무 | [M-lit] | C277([[53]]) | what_scale_buys relwork |

**대기(주장 금지·판정 후 승격)**: wrap 격리+문맥 76%↓로 022 PASS(마이크로 n=1)·레버 11종 축 묶음 효과 —
**ax32 판정 후**(오늘 라이브·`HANDOFF_2026_08_02 §2` 프레임). QP 승격·C291 구현도 같은 게이트에 묶임.
**미실측 키스톤(주장 금지·실험만이 채움)**: **crossover**(7B 학습 vs 32B scaffold·pass^all-compliance) — what_scale_buys의
모트([[46]]). 설계 의무: **per-error-type continuous rate 병행 보고**(Schaeffer 방어·사후 추가 불가).

## §2. 주장 금지·폐기 명부 (헷갈림의 원천 — 이것부터 확인)

**측정·주장 축**
1. ☠**런-간 pass 점수 귀속 전부 [D]**(C217 소급 강등) — "day6 11→day9c 5 하락", "레버로 +N" 류 옛 문장 인용 금지.
2. ☠**"재진술 오류율" 축 폐기** — X2 감사로 **검증-불가능성**으로 전환(결정가능 케이스 오류 0). "스케일 단조감소"는 **철회**(C225).
3. ☠**022를 enum-coverage 결손([[49]]) 근거로 쓰지 말 것** — C275가 전복(우리 가드의 false-abstain이 원인).
4. ☠"A2로 옮기면 도메인-일반/닫힘" — X4 반증(A2 데이터도 열림).
5. ☠"유한성 증명" 헤드라인 — X6-a 실측이 부정(C231) → **capex/opex 목적함수 정렬**로 재정식화(C233~C237). "새 프레이밍" 아님(C237 — 원 특허 §3.4에 이미 있음).
6. ☠발화-층화 분석(C262=붕괴자)·wrong-pick=경계(C265/266 기각)·V7 표적성(C268 방향 미상·C273 붕괴) — 인용 금지.
7. ☠X5 **v1** 수치(타당성 FAIL·전 등급 천장) — v2만. 살아남은 명제는 "parroting은 g3부터"뿐.

**특허·논문 구판**
8. ☠repo `paper/patent/{OISA_v5, TBOX_ABOX_v1}` = **구판**(배너 있음) — 정본은 로컬 [[32]] `PATENT_{A,B}_*` + ADDENDUM(07-30) + 덱 rev4.
9. ☠특허 C 별도 출원 단계 **폐지**(07-10·B에 흡수·분할출원 옵션만) / present·autofetch는 특허 A에서 **제거**(C34).
10. ☠`paper/archive_neurips2026_withdrawn/` 전체·`UNIFIED_OFFLOAD_DESIGN`(폐기 배너)·CFB 직접 사용([[01]]·cfbsynth가 대체).
11. ☠코드식 분류명 F1-F6·G1-G9·BC0-7·N1-N4 폐기([[48]]) — 서술형 이름만.

**relwork 헤드라인 금지(선점·오귀속)**
12. "소형>대형"(ToolOrchestra 선점 → "대형에 버금가는")·"knee 존재"·slope-측정 헤드라인([[46]]).
13. "새 기준"(C281)·"no published work does X"(C277 → "no work combines ①+②+③").
14. 오귀속 금지(C285): Baral 2003 포섭 서술·Lifschitz=CWA 연결·GCWA↔DL closed predicate 잇기. Lutz 2015는 **인용 의무**(C283).
15. FinAgent-RAG 인용 금지(C221·실존 검증 실패).

## §3. 특허 — 최신 상태 (정본=로컬 [[32]]·여기는 포인터만)

| 자산 | 정본(로컬 `_cdp_private_local/`) | 상태 | 다음 |
|---|---|---|---|
| **A**(TBox/ABox·게이트·calc) | `PATENT_A_PRIORITY_…_2026_07_05.md` + ADDENDUM(07-30 ①~⑤) | 본문+부가 완 | §1 신규 [S] 실시예 반입(행 1~4)·변리 |
| **B**(배분·knee·간섭-보상 흡수) | `PATENT_B_CAPABILITY_…_2026_07_05.md` + ADDENDUM(07-30 ①~④) | 본문+부가 완 | 행 2·5·12 반입·비용표(미작성) |
| **우산**(formalize-경계 도출+3분기 triage) | 스케치=repo `PATENT_SKETCH_FORMALIZE_BOUNDARY_TRIAGE` (C220) | **정식 명세화 미착수** | 청구 문구·변리 병행 |
| **D 후보**(출처선언·DB 안 읽음) | 등대 §5 P? 행 | E11-e2e 게이팅 | GO 시 우선 출원 |
| 덱 | `PATENT_BRIEF_DECK_2026_07_30_rev4_declfirst.pptx`(138면) | rev4 완 | **QA(COM PNG 육안) 미완**·신규 [S] 6장 후보 |
| 정렬 로그 | `PATENT_ALIGNMENT_2026_07_30.md` → **`PATENT_PORTFOLIO_STATUS_2026_08_02.md`(오늘·최신)** | — | — |

**시퀀싱 LOCK(등대 §5.1)**: **A·B 출원 → what_scale_buys 공개 → 모트 논문 공개.** 출원이 공개의 선결 — 논문 출고 순서는 특허가 정한다.

## §4. 논문 — 최신-우선 정렬 + 이름 충돌 해소

**충돌표(같은 번호≠같은 논문 — 번호 인용 금지의 이유)**

| 번호 | 등대 §5(07-08 LOCK) | `papers/paperN`(06-26) | `paper/*`(07-30) |
|---|---|---|---|
| "1" | What Scale Buys | what_scale_buys(=1+4 병합) | **declfirst**(선언-우선) |
| "2" | Levers Interfere(모트) | a2_generation | **restate**(재진술) |
| "3" | Semantic Boundary | path_selection | **feedback**(누출-등급) |
| "4" | Learned TBox Transfer | system_cost(→1에 병합) | — |

**활성(초안 실물·최신순)**

| 경로-이름 | 초안 | 주장 축(§1 행) | 선결 |
|---|---|---|---|
| `papers/paper5_interpretation_boundary/` | v0.1 (08-01) | 해석/이론 분담·행 16·17 (C287) | 프레임워크 논문 — 기전 미측정 명시(§10) 유지 |
| `paper/declfirst/` | v0.1 (07-30) | 행 1·2·3·11 — W5 우산 OPEN(C221) | X3 32B-장문·Y2급 e2e(측정 고정 후) |
| `paper/feedback/` | v0.4 (07-30) → **declfirst §로 흡수(08-02 결정)** | 행 10 | X5 v2(흡수 후 declfirst 실험) |
| `paper/restate/` | v0.3 (07-30) → **declfirst §동기로 흡수(08-02 결정·X2 v2 강하면 분리 재출)** | 행 6·9 | X2 v2 재계산 |
| `papers/paper1_capability_scale_lever/` (what_scale_buys) | md/tex/pdf (07-12) | 행 12·13·14·18 + 등대 §5 [S] 묶음 | **출원 선결** + crossover 키스톤([[46]]·Schaeffer 병행 보고) |

**휴면(증거 동결·07-19/24 아크·활성 5 뒤로 정렬)**: `PAPER_TRACKA`(Same-Rule Interference)·`PAPER_TRACKB`(Degenerate-Cue)·
`PAPER_TRACKC`(Isolation Replay·C121~C157) — 재개 시 §2 규율(pass=[D]) 아래 증거 재감사 필수.

**proposal만(실측 대기)**: `papers/paper2_a2_generation`(=등대 P5 계열)·`papers/paper3_path_selection`(CDP 세부=로컬 전용).
**등대-게이팅 대기**: 모트 논문(E1)·Semantic Boundary(E3)·TBox Transfer(E6′)·Source Declaration(E11).
**아카이브**: `papers/paper4_system_cost`(병합 소스)·`paper/archive_neurips2026_withdrawn/`.

### §4b. 실패-축 → 논문 매핑 재정 (2026-08-02 사용자 질문·박제)

- **축별 논문 금지**: qp32p1의 관찰 6축은 한 런의 포렌식 클러스터이고, §0.2 계보상 절반이 **기지-미구현
  재발**(C208)이며, pass=[D] 위기로 축별 효과 정량이 불가하다. 6축 재료는 **declfirst(레버 지도)와
  interp(닫힘/열림 경계)의 실증 절로 흡수**한다 — 새 논문을 만들지 않는다.
- **논문·특허 프레임은 인과 3(+1)축으로 말한다**(관찰 6축은 런 포렌식용으로만): ①**닫힌 성분**(채널·턴·
  반복·접지·diff=엔진 강제) ②**열린 성분**(해석·선택·UNA=LLM+표면화·ASK — **over↔under 켤레가 사는 곳**)
  ③**부하=증폭기**(격리·문맥 경제 — 독립 원인 아님·A5·[[18]]) (+④측정·벤치 결함=분모 제외·위생).
  근거: C287(해석/이론)+LEVER_CAUSE_MAP(A5=증폭기·A6=레버 금지)+[[18]] — 재론 아님, 접기만.
- **over-action vs under-action = 같은 문제의 켤레 방향**(등대 §1.3 제1원리). 실측 3건: C250(선언↔행동량)·
  C275(가드의 false-abstain)·fit 상쇄(intent-가드↔판별력). 모든 레버에 Δspurious≤0 계측 의무의 근거.
- 학습은 별도 축이 아니라 **축을 가로지르는 잔여 흡수기**([[13]]·enum·INFER-cal). 멀티에이전트 격리는
  축 해결책이 아니라 **증폭기 제거**(022 마이크로 n=1·ax32 판정 대기 — 주장 금지 등급).

## §5. 딥리서치 — 정본 3세대 (최신이 정본·구세대는 인용 전 재확인)

| 세대 | 문서 | 권위 범위 | 메모리 |
|---|---|---|---|
| **3(현행 정본)** | `PRIOR_OFFLOAD_CRITERION_2026_08_01`(C281·C283) · `PRIOR_GCWA_DISJUNCTION_2026_08_01`(C285) · `PRIOR_MAPPING_BOUNDARY_SCALE_2026_08_01`(C277) · `INTERPRETATION_THEORY_SPLIT_2026_08_01`(C287) · `DR_DECLFIRST_DR2_2026_07_30`(C221) · `DR_BANLIST5_PRIOR_WORK_2026_07_30`(C218) | 경계·게이트 의미론·매핑·선언-우선 선점 지형 — **경계/스케일/OR/분담 주장 전 필독** | [[50]][[51]][[53]][[52]] |
| 2(주제 캡) | `PRIORWORK_SYNTHESIS_4AXIS_2026_07_14` · `RELWORK_{LOAD_COT,AGENTIC_HORIZON,SCALE_LOAD,FINITE_TOOLSET_SELECTION,ISOLATION_ATTRIBUTION}` | 부하·horizon·도구선택·귀속 축 | [[41]]~[[45]][[47]][[49]] |
| 1(부품 정독·아카이브급) | `relwork_{selector,diversity,nlformalize,metrics,determinism,arch}_2026_06_14` | 옛 부품 계보 — 인용 전 재확인 | — |

주의: CMU 2604.15579는 [[50]]의 "정독 최우선"이 **C283에서 완료**됨(양보 확정·τ² 사용 경쟁자) — 미독으로 오인 금지.
메모리 링크 수리(오늘): `[[49]]` 이중지시 해소 — mapping-boundary는 **[[53]]으로 개번**·[[49]]=coverage(2404.09593)로 단일화.

## §6. 설계서·실험결과 — 주제별 정본 (특허·논문이 인용할 곳)

| 주제 | 정본(최신) | 비고 |
|---|---|---|
| 프레임·원장·실험큐 | `RESEARCH_MASTER.md` §1·§3·§4 | 최상위 |
| 선언-우선 아키텍처 | `DECLARATION_FIRST_REDESIGN_2026_07_29`(rev1b·C219 LOCK) | LLM=formalize만 |
| 결정론/학습 경계 | `AXIS_DECISION_…_2026_07_29`(rev1) + `STACK_PREDICATE_AUDIT_2026_07_29` | 코어 6층·닫힘 배터리 |
| 스캐폴드 일반화 | `GENERALIZED_SCAFFOLD_ARCHITECTURE_2026_07_12`([[16]]) | GET/FIND/INFER/ASK |
| 레버 통합(사다리·거버너·버스) | `LEVER_CONSOLIDATION_DESIGN_2026_08_02`(승인) | 교체는 x45 게이트 |
| 레버-원인 전도·규율 | `LEVER_CAUSE_MAP_2026_08_02` | 메타결함 4 |
| 폭주(캡 정본) | `RUNAWAY_AXIS_REDESIGN_2026_08_02` | 3기전 분리 |
| 실패 축(비폭주 5축) | `FAILURE_AXES_REDESIGN_2026_08_02` | §0.2 기지/신규 계보 |
| 지시-결함 | `INSTRUCTION_DEFECT_REDESIGN_2026_08_01`(r3·C290) | D1~D5 |
| 인용-접지·핀 | `QUOTE_GROUND_PINKIND_REDESIGN_2026_08_01`(rev2·§8b/§8c) | R5 철회 주의 |
| 멀티에이전트 | `FUNCTION_AGENT_ARCH_REV2_2026_08_02`(승인) | 구현은 통합 후 |
| 실험 계획(특허·논문용) | `EXPERIMENT_PLAN_PATENT_PAPERS_2026_07_30`(rev1) | X1~X8·Y1~Y3 |
| 명명 권위 | `UNIFIED_TAXONOMY_2026_07_09`([[48]]) | 코드 폐기 |
| frontier 궤적 | `TAU2_FRONTIER_TRAJECTORY_INVESTIGATION_MASTER_2026_07_09`([[47]]) | 재단정 금지 |
| 결과 원본 | `sim_results/`(영속)·`QP32P1_FAILURE_TRAJECTORIES_2026_08_02`·`X_FREE_TRACK_RESULTS_2026_07_30` | 수치는 여기서만 |

## §7. 히스토리 (압축 연표 — 이 아래는 전부 위 정본들로 대체됨)

06-14 부품 relwork 6편 → 06-26 papers 포트폴리오(1+4 병합·3편 체제) → 07-05 특허 A/B/C/D+덱 → **07-08 등대 §5
4분할 LOCK** → 07-09 특허 정련(present 제거·D후보) → **07-10 C→B 흡수**·덱 rev2 → 07-11 rev3(레버) → 07-12 덱
인용 QA(132면) → 07-19/24 Track A/B/C 드래프트(휴면) → **07-29/30 선언-우선 아크**(C218~C225: LOCK·DR1/2·
논문 신규 3편·A/B ADDENDUM·덱 rev4 138면·실험계획) → 07-31 Y1 완주(측정 임계 28%) → **08-01 선행 3부작+
interp 논문+QP 인과사슬([S] 행 3·4)** → **08-02 지금**: 6축 포렌식·레버 11종·AXIS-32 라이브(판정 대기).

## §8. 링크 수리 로그 (2026-08-02)

- 메모리 `[[49]]` 이중지시 해소(mapping→[[53]] 개번·46/50/51/52/MEMORY.md 갱신).
- `papers/README.md`에 이 문서 포인터 배너(07-30 3편+interp 미반영 상태였음).
- `RESEARCH_MASTER §5`에 실물-지도 포인터 1줄(추가만·재론 아님).
- 구판 특허 2종 배너 = 07-30에 완료돼 있음(확인만).
