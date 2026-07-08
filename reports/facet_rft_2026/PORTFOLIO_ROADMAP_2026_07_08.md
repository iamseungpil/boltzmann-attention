# 포트폴리오 로드맵 — 프레임 확정 · 증거 원장 · 실험 맵 · 논문/특허 분할 (2026-07-08)

> **목적**: 표류 종식. **프레임 확정 → 확정/미확정 구분 → 실험으로 맵 채우기 → 논문 분할**을 하나의 문서로.
> 상위 = `MASTER_FRAME_LEVER_COMPOSITION_2026_07_08`(전체 그림). 특허 = `_cdp_private_local/`(로컬·[[32]]).
> 증거등급 **[S]**settled(다-trial·per-case) · **[M]**measured(단일 clean run) · **[P]**promise(isolated/nt=1) · **[D]**design · **[?]**미실행.

---

## 1. 확정된 프레임 (LOCK — 재론 금지)
에이전트의 논리적 작업을 **6기능**으로 분해하고, 각 기능의 **부하 반응**으로 **최저비용 레버**를 정한다. 단
**레버는 역효과를 낳으므로 독립 배분이 아니라 합성**한다.

| 기능 | scale | thinking | scaffold | 최저비용 레버 |
|---|---|---|---|---|
| F1 compliance/guarantee | **invariant** | 직교 | **게이트=위반0** | **scaffold** |
| F2 symbolic operand | 미약 | **✅ 싸다** | calc/present(토큰0) | **thinking / 결정론 compute** |
| F3 semantic operand (⋈) | **flat** | **✗** | ✗ | **없음 = 경계** |
| F4 coverage/completion | invariant | **✗ 악화** | **완결 게이트** | **scaffold** |
| F5 persistence/escalation | ? | **✗ 악화** | **persistence 게이트** | **scaffold** |
| F6 horizon (복리 $p^H$) | **✅ 산다** | 부분 | 분해 | **scale (싸게=fleet)** |

**★제1원리(모트)**: 레버는 하나를 사면 하나를 판다. **thinking**: +F2 / −F4·F5 → 순 0. **present**: +F3grounding / −over-action → 순 0.
**present+g15**(status-lock이 over-action 상쇄) → **+12.3pp**. ⇒ **설계 = 부작용이 상쇄되는 합성**.

## 2. 증거 원장 (무엇이 확정, 무엇이 아닌가)
| # | 주장 | 등급 | 근거 | 미결 |
|---|---|---|---|---|
| C1 | compliance는 scale-invariant·게이트만 위반0 | **[S]** | g2 per-write rate 7B .103/14B .070/32B .075(CI중첩)·게이트 전 scale 0·frontier 낙폭 −2.2~−5.0pp | — |
| C2 | **compliant-pass crossover**: 14B+scaffold .336 > 32B bare .300 (전 k) | **[S]** | clean nt=4 replay-safe·pass^1 .588>.509 | 7B arm 미실행 |
| C3a | 애매모호성 = symbolic vs semantic 이분 | **[S]** | CoT: symbolic +17/+35% vs semantic +4%≈0 · 투표 8/8 동일오답 | — |
| C3b | **semantic(⋈)은 경계** — scale·budget·CoT·reasoning-RL 다 실패 | **[M]** | ⋈ 격리: 32B .41 / CoT900 .42 / big8000 .40 / QwQ native .40 / 14B+CoT .43 → 실링 ~.44 | **isolated under-spec** → full-agent 확인 필요(E3) |
| C4a | present(+grounding, −over-action)·present+g15 합성이 이득 | **[S]** | det census·passAll +12.3pp | — |
| C4b | **thinking: +F2 / −F4·F5 → 순 0** | **[M]** | QwQ+rparser 0.526 vs base 0.557 · **12승=12패** · 손실 8/12=orchestration | nt=4(E2 진행중)·QwQ≠Qwen2.5 교란 |
| C4c | **합성(thinking+완결게이트)이 순이득으로 전환** | **[D]** | 설계만(`THINKING_PERSISTENCE_SCAFFOLD_DESIGN`) | **미실행 = 모트의 빈칸(E1)** |
| C5 | scale이 사는 유일 축 = horizon | **[S-lit]/[M-ours]** | DR#2(복리 $p^H$) + 우리 F1/F3/F4 non-scaling 실측 | 우리 horizon 직접측정 없음 |
| C6 | fleet = horizon 전용·저-ROI(현 벤치) | **[M]** | f realistic .065·1.15~2.3×(R 지배·미측정)·잔여 scale-flat | benefit 미측정(big-tier 미로컬) |
| C7 | learn-wing이 F3 경계/mis-formalize를 여는가 | **[?]** | tau2 operand SFT=NO-GO 확정 / four-bench→τ² swap 미실행 | **전면 미실행** |
| C8 | TCO ~23× | **[EST]** | $0.0019 vs $0.044 (배포환경 의존) | 정밀화 |

**요약**: C1·C2·C3a·C4a = **확정(논문화 가능)**. C3b·C4b = 측정됐으나 caveat. **C4c(모트의 심장) = 미실행.** C7 = 미개척.

## 3. 실험 맵 (미결을 닫는 최소 집합·우선순위)
| ID | 실험 | 닫는 것 | 비용 | 의존 |
|---|---|---|---|---|
| **E1** | **완결/persistence 게이트** Phase A(offline·무료) → B(smoke·무료 user-sim) → C(유료 확인 1회) | **C4c [D]→[M]** = 모트 | 무료→소액 | naive 프록시 발화 19(확실파손 5)·정밀 over-block **미측정**·**Δspurious≤0이 GO 조건** |
| E2 | QwQ+rparser **nt=4**(진행중) | C4b [M]→[S] | 유료(실행중) | — |
| E3 | F3 경계 **full-agent 확인**(agentic wrong-exec order_id 지배 분석 + 맥락-정합 격리 probe) | C3b [M]→[S] | **무료** | 기존 궤적 |
| E4 | **base + 완결게이트**(일반성 회귀) | 게이트가 QwQ-특이 아님 → C1/C4 강화 | 무료/소액 | E1 후 |
| E5 | 7B assembled arm | C2 사다리 완성([[46]] keystone) | 소액 | — |
| E6 | **learn-wing**: four-bench → τ² ABox-swap | **C7 [?]→판정** | 큼(GPU+유료) | E3 결과(경계면 learn만 남음) |
| E7 | fleet | C6 | 보류 | big-tier 확보 or F6 지배 입증 시 |

**우선순위**: **E1(모트) > E2(진행중) > E3(무료) > E4 > E5 > E6 > E7.**

## 4. 논문 분할 (필요한 만큼 · 각 논문의 완결 조건)
| 논문 | 주제 | 담는 주장 | 증거 상태 | 게이팅 실험 | 특허 대응 |
|---|---|---|---|---|---|
| **P1** *What Scale Buys…* (기 작성·near-complete) | 능력×규모×레버×비용 **맵**; guarantee는 규모로 못 산다 | C1·C2·C3a·C5·C8 | **전부 [S]/[EST]** | 없음(즉시 가능·E5로 강화) | **특허 A**(게이트·present/calc) + **B**(배분·knee) |
| **P2** *Levers Interfere: Composition, not Allocation* (신설·**모트**) | 레버 역효과와 합성 법칙 | C4a·C4b·**C4c** | C4a [S]·C4b [M]·**C4c [D]** | **E1**(+E2·E4) | **특허 B 확장** → *간섭-보상 배분* = **신규 청구 후보(특허 C?)** |
| **P3** *The Semantic Boundary* | symbolic/semantic 이분과 **semantic 벽** | C3a·**C3b** | C3b [M] | **E3**(무료)·(선택)E6 | 특허 B(무엇을 레버로 못 사는지 = 배분 경계) |
| **P4** *Learned TBox Transfer* (미개척) | four-bench TBox → τ² ABox-swap 전이 | C7 | **[?]** | **E6** | **특허 A**(TBox/ABox·재학습0 전이)의 실증 |
| P5 | *A2 frontend*(NL→A2 생성기) | — | 범위 밖(2026-06-25 결정) | — | 후속 |

**분할 근거**: P1은 이미 [S]만으로 완결 → **E1을 기다리지 않고 출고**(de-risk). P2는 모트지만 **E1 하나에 달림**. P3는 **무료 E3**로 완결 가능(가성비 최고). P4는 thesis의 나머지 날개지만 미개척 → 최후.
**병합 옵션**: P3를 P1 §5 잔여-특성화로 흡수 가능(논문 수 축소). 권장 = **독립**(⋈ 4-레버 실링은 단독 기여).

## 5. ★시퀀싱 제약 — 특허 우선 출원
특허 명세(A/B front matter)에 **"논문(what_scale_buys 등) 공개 전 출원 필수(신규성)"** 명시.
⇒ **순서 강제**: (1) 특허 **A·B 출원** → (2) P1 공개 → (3) 역효과/합성이 신규 청구면 **특허 C(or B 보정) 출원** → (4) P2 공개.
**즉 E1(모트 실험) 결과가 특허 C의 뒷받침이 되므로, E1을 P2 집필보다 먼저.**

## 6. 실행 순서 (표류 방지·확정)
1. **[지금] 프레임 LOCK**(§1) — 재론 금지.
2. **E3(무료)** → C3b 확정 → **P3 집필 가능**.
3. **E1 Phase A(무료)** → 정밀 트리거로 over-block 0 나오는지 → GO면 Phase B(무료 smoke).
4. **E2 회수**(진행중) → C4b [S].
5. **특허 A·B 출원 준비** 병행(P1 공개 차단 해제).
6. E1 GO → **특허 C(간섭-보상 배분) 검토** → **P2 집필**.
7. E5(7B) → P1 강화. E6(learn) → P4 판정. E7(fleet) 보류.

## 7. 이 문서의 규율
- §1 프레임은 **LOCK**: 새 측정이 반증하지 않는 한 재론 금지([[03]]).
- 모든 주장은 §2 원장의 등급을 달고만 인용. **[D]/[?]를 [M]처럼 쓰지 말 것**([[08]]·대칭크레딧).
- 실험은 §3 우선순위대로. 지엽 최적화 금지 — **E1이 모트의 심장**.
