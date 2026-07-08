# Phase-0 plan_probe 판정 — orchestration 잔여 = 학습이냐 실행부하냐 (2026-07-06)

> **입력**: `sim_results/plan_probe_phase0_2026_06_26.txt.gz` (2026-06-26 실행·미판정) + 설계 `ORCHESTRATION_CAPABILITY_LEVER_DESIGN_2026_06_26.md §9` + robust run-side `ASSEMBLED_FAILURE_FORENSIC_2026_06_27.md`.
> **왜 지금**: probe는 돌았으나 **전수 per-case 판정 없이** 한 줄 집계로만 요약됨(`HANDOFF_2026_06_26_PM` L10: "절반 닫힘·절반 planning miss→Paper 3"). [[08]] = 집계→결론 금지·전수 궤적 판정. 이 문서가 그 판정.
> **비용**: gpt-4.1 0 (기존 데이터 재분석·신규 런 0).

---

## 1. 측정 (plan-in-isolation·구조채점·temp=0·1샷)
open NL 목표 + reads-done 주문맥락만 주고 **plan-spec(실행 0)** 요청 → gold write-구조와 대조. concrete operand(변형)는 안 봄(GIVEN-SPEC 이미 100%). 부하 0·turn-confound 면역.

- **core_ok(필수구조·batching 정확) = 6/10 · 완전 STRUCT_OK = 3/10.**

| task | 라벨 | per-case 판정 (전수 정독) | 진짜 축 |
|---|---|---|---|
| **t20** | CORE_OK+EXTRA | 필수 modify_items(W9911714·4품목 배칭) **정답**. EXTRA=투기적 exchange+payment | 실행부하(H1)+over-reach |
| **t36** | CORE_OK+EXTRA | 필수 modify_items(W9348897) **정답**(조건분기 다계획). EXTRA=cancel+payment | 실행부하+over-reach |
| **t99** | CORE_OK+EXTRA | **2주문 exchange 둘 다 정답**(W4689314·W3916020). EXTRA=제3주문 over-reach. ★실제 런선 "1주문 누락+날조"였는데 **격리 계획선 둘 다 맞음** | **H1 직접 증거**(실행부하) |
| t37 | MISSING/EXTRA | t36과 **동일 주문·동일 gold**인데 modify_items 누락(payment→cancel만). t36은 CORE_OK → 조건분기 형식화 **변이(noise)**·안정적 gap 아님 | 변이 |
| t109 | MISSING/EXTRA | pending 주문에 `exchange_delivered`(잘못된 action-type) 사용 = **status→action 결정가능** | 게이트/status |
| t111 | MISSING/EXTRA | ⋈ **틀린 주문**(W3964602 vs gold W9810810) = operand-binding·규모로 은퇴(14B23→32B14) | scale |
| t71 | BATCH_SPLIT/EXTRA | 한 주문 품목변경을 **2콜로 분할**(gold 1콜) = 배칭 제약(결정가능) + EXTRA payment | A2/controller |
| t17·t92·t105 | STRUCT_OK | 단순(flip)·고정턴 아티팩트 | — |

---

## 2. 판정 (설계 §9 매트릭스)

**(i) SELECT 지배 · plan-GENERATE 학습 = NO-GO.**

세 가지가 수렴한다:

1. **필수 구조 planning은 블로커가 아니다 (core_ok 6/10, 그리고 non-core 4건은 plan-generation 아님).** 실제 런에서 "orchestration 실패"로 분류된 멀티주문/멀티품목 케이스(t20·t36·**t99**)가 **격리 계획선에서는 필수 구조를 정확히 산출**한다. robust run-side(ASSEMBLED_FORENSIC: t99가 다-trial 전부 fail)와 대조하면 = **실패는 planning 능력이 아니라 plan+execute 동시수행 부하(H1)**. ⇒ 레버 = **C1/C2 결정론(plan/execute 분리 + controller)**, 학습 아님.

2. **살아남은 "MISSING/BATCH"는 novel-plan-GENERATE가 아니다.** 전수 판정 결과: t37=단일샷 변이(t36 동일과제 CORE_OK), t109=status-decidable action-type, t111=⋈ 틀린주문(규모 은퇴), t71=배칭 제약(결정가능). **"planning을 새로 생성 못 함"인 케이스 = 0.** ⇒ 한 줄 요약의 "절반 planning miss→Paper 3"는 **집계 착시**였고, 전수 정독하면 그 miss들은 전부 {변이·결정가능·⋈·배칭}으로 환원 = **학습 레버 근거 아님**(설계 §12.4 트리거 "격리서 plan-틀림 genuine ≥1"= **0건**).

3. **지배 잔여 = over-reach(EXTRA)** — 복잡 과제 **전건**에 gold에 없는 쓰기(투기적 payment 변경·cancel·제3주문)를 계획한다. 이는 OVER-ACTION(scope 추론) 잔여로, ASSEMBLED_FORENSIC서 **규모로 악화(32B 19%↑·파괴적)**·LLM-scope(게이트 금지·[[06]] lever-type≠해결)로 확정된 그것과 동일. ⇒ 규모가 사는 능력(scope-discipline).

**결론 = make-or-break(operand NO-GO)와 정합·강화**: tau2-retail의 남은 잔여는 {실행부하→결정론 C1/C2, ⋈·over-action→scale, 아티팩트}이며 **plan-generation 학습(C3)의 근거는 없다.** thesis learn-wing은 tau2서 carry 안 함(§ MAKEORBREAK_VERDICT 정합). 헤드라인 = **결정론 controller + base translator + scale가 사는 scope/binding**.

---

## 3. 정직한 단서 (over-claim 방지)
- **단일샷**(temp=0·task당 1 plan). t36/t37 발산이 단일샷 노이즈를 실증. non-core 개별건은 robust 아님.
- **완화**: (a) run-side는 robust 다-trial(ASSEMBLED_FORENSIC)이라 H1의 실패측은 견고. (b) core_ok 6/10 + non-core 4건 전수 판정이 일관되게 "plan-generation 아님". 두 축이 같은 결론.
- **robust 확증(권장·저비용·무료)**: `plan_probe`를 **k=4 다-trial**로 재실행(10과제×4=40 경량콜) → MISSING/BATCH의 robust 여부(t37류 flip 분리). **단 in-flight 유료 gpt-5.2 재측정(32B 공유)과 GPU 경합 회피 위해 PERSISTED 후 실행**([[09]]).

---

## 4. 성능개선 함의 (Track 2 다음)
- **결정론 개선 남은 여지 = C1 plan/execute 분리 하네스**(§6 Phase 1): 부하-유발 CORE_OK-but-run-fail 집합(t20·t99·t36…)을 짧은 plan-spec 1회 + atomic leaf 재질의 + compact-state 주입으로 회복하는지 측정. 배칭(t71)·status(t109)는 C2 controller에 흡수. **이것이 유일하게 headroom 있는 결정론 레버.**
- **over-action·⋈·criterion = scale/LLM-resident** → 결정론 레버 아님(헤드라인으로 흡수).
- **판정**: Phase 1(C1) = GPU-free 하네스 빌드 가능·로컬 robust 평가. **단 end-to-end 유료 확인은 결론 후 1회**([[09]]). C3(SFT) = **NO-GO 유지**(genuine plan-generate 잔여 0).

## 5. 불변 정합
- [[08]] 전수 per-case 판정 완료·robust run-side 교차·단일샷 caveat 명시. [[03]] probe=adjudicator(예측으로 결론 안 함).
- [[13]] 결정론(C1/C2) 먼저·학습 최후·여기선 학습 NO-GO. [[05]] C2 controller=도메인-일반 IR·retail 하드코딩 금지(빌드 시 게이트).
- [[09]] robust 재확증·end-to-end = 유료 in-flight 후·승인·최소.
