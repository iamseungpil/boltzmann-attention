# 전체 그림 — 기능 × 부하 × 레버 × 역효과(합성) 마스터 프레임 (2026-07-08)

> **목적**: 흩어진 DR·실험 전부를 하나의 스파인으로. Paper1(`what_scale_buys`)의 프레임을 **갱신**(재발명 아님·[[03]]).
> **사용자 이론 진술(2026-07-08) 채점 결과 반영**: 3개 확증·2개 교정·1개 축 신설(역효과/합성).
> 증거등급: **[S]**=settled(다-trial·per-case forensic) · **[M]**=measured(단일 clean run) · **[P]**=promise(isolated·nt=1) · **[D]**=design.

---

## 0. 한 줄 (논문 헤드라인)
**Scale은 horizon을 사지, guarantee도 semantic reference도 사지 못한다.** guarantee는 결정론 scaffold가 **pass-비용 0으로** 산다.
symbolic 추론은 test-time compute가 싸게 산다 — **단 test-time compute는 persistence를 *판다***. 그래서 레버는 독립 배분이 아니라
**부작용이 상쇄되는 합성(composition)**으로 배치해야 한다. 잔여(semantic reference)는 우리가 시험한 어떤 레버도 열지 못한 **경계**다.

## 1. 축 1 — 기능 분해 (에이전트가 해야 하는 논리적 작업)
| # | 기능 | 실패 형태 |
|---|---|---|
| F1 | **compliance/guarantee** (정책·상태변경 전 확인) | 위반 |
| F2 | **symbolic operand** (비교·계산·기준선택) | 잘못된 변형/값 |
| F3 | **semantic operand** (⋈ 참조매칭·의도) | 틀린 주문/대상 |
| F4 | **coverage/completion** (all/both/every 전수) | 미완 |
| F5 | **persistence/escalation** (성급 포기 금지) | 조기 transfer |
| F6 | **horizon** (다단계 신뢰도 복리 $p^H$) | 누적 붕괴 |

## 2. 축 2 — 부하 반응 (무엇이 그 기능을 움직이나) · 축 3 — 최저비용 레버
| 기능 | scale | thinking | scaffold | 최저비용 레버 | 증거 |
|---|---|---|---|---|---|
| F1 compliance | **invariant** (per-write g2 rate 7B .103·14B .070·32B .075·CI중첩) | 직교 | **✅ 게이트=위반 0** | **결정론 게이트** | [S] |
| F2 symbolic operand | 미약(+5pp·CI 0 포함) | **✅ CoT +7~14pp(싸다)** | calc/present offload(토큰0) | **thinking or 결정론 compute** | [S]/[P] |
| F3 semantic operand (⋈) | **flat** (14B+CoT .43≈32B .41≈QwQ .40) | **✗ 무효**(native reasoning=base 직답) | present 열거로도 안 됨 | **없음 = 경계** | [M] |
| F4 coverage | **invariant** (17≈16) | **✗ 악화**(give-up) | **✅ 완결 게이트(E2)** | **결정론 게이트** | [S]/[D] |
| F5 persistence | ? | **✗ 악화**(조기 escalate) | **✅ persistence 게이트** | **결정론 게이트** | [M]/[D] |
| F6 horizon | **✅ scale이 삼**(DR#2·복리) | 부분 | 분해/E1 | **scale(→싸게=fleet)** | [S] |

**★F3 경계 실증(2026-07-07 밤)**: ⋈ 격리정확도 = 32B base .41 · prompted-CoT(900) .42 · big-budget(8000) .40 ·
**QwQ native reasoning(1639 tok) .40** · 14B base .22→+CoT .43. **scale·budget·CoT·reasoning-RL 넷 다 ~.44 실링 못 뚫음.**
⇒ semantic ambiguity는 "큰 모델에 위임(fleet)"으로 안 풀린다. **map + 수용 (or learn=미검증)**.

## 3. ★축 4 — 역효과(antagonism)와 합성 (신설·본 논문의 모트)
**단일 레버의 이득은 그 레버가 유발한 회귀로 상쇄된다. 이기는 구성은 *부작용이 서로 상쇄되는 합성*이다.** 측정 4건:
| 레버 | 사는 것(+) | 파는 것(−) | 순효과 | 상쇄 파트너 |
|---|---|---|---|---|
| **thinking** | F2 결정정확도 (criterion .727→.864) | **F4/F5 완결·persistence**(조기포기) | **≈0** (QwQ 12승=12패·parity) | **완결/persistence 게이트** [D] |
| **present** (scaffold) | F3 order-pick +.063 | **over-action +11pp** | ≈0 | **g15 status-lock** |
| present+g15 | — | — | **+12.3pp passAll** | (합성이 이득) |
| g15 단독 | over-action↓ | L0↑ | 음성 | — |
| retry / 투표(self-consistency) | — | 해로움 / +0%(8/8 동일오답) | 음성 | — (죽은 레버) |
- **함의(설계원리)**: 기능별 최적 레버를 독립 선택하면 실패한다. **레버 배분 = 교차-기능 간섭을 측정하고, 부작용을 서로
  상쇄하도록 합성하는 문제.** (prior work 부재·[[46]] 노벨티 지형과 정합.)
- **DR 정합**: self-reflection(내부 피드백 없음)=무효[2310.01798]·long-horizon 조기종료 1%→25% ⇒ **외부 결정론 verifier**가
  thinking의 부작용을 상쇄하는 정당한 파트너(prompt/self-critique 아님).

## 4. 축 5 — 비용·크로스오버 (배포 결론)
- **compliant-pass crossover [S]**: 14B+scaffold **0.336** > 32B bare-compliant **0.300** (pass^1 .588>.509) — **전 k 성립**.
  = 합성(작은 모델+게이트)이 규모를 이긴다, **guarantee 축에서**.
- **frontier도 준수 낙폭**(claude-3.7 −5.0·gpt-4.1 −2.6·o4-mini −2.2·gpt-4.1-mini −2.6pp)·**우리 게이트만 0** [S].
- bench-pass는 frontier ~9pt 위(우리 미달) — **우위는 pass가 아니라 guarantee** [S].
- **TCO ~23×** ($0.0019 vs $0.044).
- **fleet [M]**: 정당 영역=F6 horizon뿐. 비용 realistic 1.15~2.3×(R 지배·미측정 가정). 우리 잔여는 scale-flat/invariant라
  **저-ROI·보류**. (semantic→fleet은 F3 경계로 **반증**.)

## 5. 논문 스파인 (Paper1 갱신 지도)
1. **주장 1 [S]**: scale은 guarantee를 못 산다(F1 scale-invariant·강한형). 결정론 게이트가 pass-비용 0으로 산다. frontier도 낙폭.
2. **주장 2 [S]**: 따라서 **compliant-pass crossover** — 작은 모델+scaffold > 큰 모델 bare(전 k). 비용 23×.
3. **주장 3 [S/P]**: 애매모호성은 **symbolic**(CoT/결정론 compute로 싸게 닫힘)과 **semantic**(⋈)으로 갈리며, **semantic은 scale·
   budget·CoT·reasoning-RL 어느 것도 못 여는 경계**다.
4. **★주장 4 [M/D·모트]**: 레버는 **역효과**를 낳는다 — thinking은 F2를 사고 F4/F5를 판다(순 0). 이기는 배치는 **합성**
   (thinking + 완결/persistence 게이트 · present + status-lock). ⇒ **cost-optimal lever allocation은 분리불가·간섭을 측정해야.**
5. **주장 5 [S]**: scale이 실제로 사는 유일 축은 **horizon**(복리 $p^H$) → 그것만 싸게 사려면 fleet. 그 외 fleet은 저-ROI.

## 6. 무엇이 아직 안 닫혔나 (정직)
- **F3 경계**: isolated under-spec 하한 — full-agent 맥락서도 경계인지 미확정(agentic wrong-exec의 order_id 지배는 정합적 증거).
- **주장 4의 [D] 절반**: thinking+완결게이트가 실제로 순이득을 내는지 = `THINKING_PERSISTENCE_SCAFFOLD_DESIGN` Phase A/B 미실행.
  (offline naive 게이트는 over-block 19 → 정밀 조건 필요.)
- **QwQ 결과 [M]**: nt=1(0.526 vs base 0.557)·Step3 nt=4 진행중. QwQ≠Qwen2.5(RL/템플릿) 교란.
- **learn-wing**: F3 경계·F-mis-formalize를 학습이 여는지 미검증(four-bench→τ² swap).

## 7. 다음 (프레임 확정 후)
1. 본 프레임을 Paper1에 반영: §3 프레임에 **축4(역효과/합성)** 신설·§5에 thinking/⋈경계/fleet 교정 편입.
2. 주장 4를 [D]→[M]으로: 완결/persistence 게이트 Phase A(offline)→B(smoke).
3. Step3 nt=4 회수 → QwQ 정본 pass^1..4.
