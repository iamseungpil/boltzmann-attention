# RUNG1 T1c — grounded-permitted @ source=1 (treeval@s1) + 결정 레버: 정밀 설계서

> 2026-06-04. 진입점 체인: `EXPERIMENT_DESIGN.md §2` → `RUNG1_SOURCE_LADDER_DESIGN.md §11-12` → **이 문서**.
> 직전 결과: `Exp-4-rung1-upperbound`(권위본) — source=1(C) BOTH 3 = A, **전수조사 = gathered_then_REFUSE C 29/A 24**(완전게더 후 permitted 콜드붕괴).
> ★실수 방지용 정밀 설계. **다른 세션에서 ①리뷰 → ②(거의 무구현, 플래그 조합) → ③실험.**

## 0. 한 줄
upper-bound 전수조사 = 병목은 **"완전 게더 → permitted 콜드붕괴 → 거부"(29/48)**. 처방 = permitted을 *cold should_succeed 라벨 예측*에서 **게더한 leaf-truth의 grounded AND/OR 평가**로 바꿈. **핵심 발견: 이건 새 메커니즘이 아니라 *이미 구현·디버깅된 treeval grounded-gate를 source=1에서 돌리는 것*** — v3 treeval 실패는 fabrication(source=3) 때문이었고, source=1은 구조를 줘서 fabrication을 원천 차단한다.

## 0.5 ★PRE-CHECK 결과 (2026-06-04, 학습 전 필수 게이트 — PASSED) + 사전등록 threshold
**🔴 build-time 천장 검사 (리뷰 BLOCKING)**: bank s1 treeval teacher SFT_TRACE → should_T 48 중 **grounded-ACT(agree=True, `gate=...=true;ACT`)=34 / fallback(cold permitted)=14**. fallback 14 = **transfer_funds 8·cancel_credit_card 4·pay_bill_with_credit_card 2**(dependency-undefined leaf=카드/목적지 미정의 → tv=None → 13% fallback, **T1c 못 고침**).
- **→ T1c 천장 ≈ 34** (not ~10). dirgraph-충족 34 == grounded 34 → **gathered_then_REFUSE 29는 grounded 집합 내 → 전환 대상.** 4h 투자 정당.
- **★사전등록 판정(n=48, ±2 noise, 모든 선행 BOTH≤5, 천장 34)**: **성공 = BOTH ≥ 12**(29의 ≥1/3 전환) · 강성공 ≥ 20 · 부분 6–11 · **null ≤ 5**. (천장 34로 cap.)
- **🟡 보조 검사(s3 no-fab subset)**: treeval@s3 RLLOG small(≤4op,non-fab) gate = true 50/false 50(콜드붕괴로 false 쏠림 아님=20/80 아님) → scaffold가 true도 생성함(약-고무적). ⚠️should_T 라벨 부재로 inconclusive지만 "non-fab서 콜드붕괴 지배" 반증 못 함 = train 막을 신호 아님.
- **→ 결론: 🔴 통과(천장34) + 🟡 약-고무적 → train 진행.**

## 1. 동기 — 전수조사가 정확히 가리키는 것 (권위본 `Exp-4-rung1-upperbound`)
- C(s1) should_T 48 버킷: **gathered_then_REFUSE 29**(dirgraph 충족·goal 미호출·exit) · premature 10 · BOTH 3 · dg+goal부정확 2.
- dirgraph-충족 34 중 **29 거부**(goal 호출 5뿐). RLLOG: C `permitted=false;STOP 53`·ACT 38(콜드거부 planner수준 절반↓ vs A 114, but should_T 완전게더분은 여전히 거부).
- **C-none teacher 타깃 분포(검증)**: `permitted=true;ACT 301 / permitted=false;STOP 542` = teacher가 permitted=**should_succeed(cold label)** 가르침. 모델은 inference서 should_T를 false로 과소예측(prior 1.9× false 편향).
- **확정**: 병목 = permitted이 *게더 결과와 분리된 콜드 라벨 예측* → 게더 다 해놓고도 false. source=1은 게더만 늘려(dirgraph 29→34) 거부 대상만 늘림.

## 2. ★핵심 메커니즘 — 왜 T1c가 C-none과 다른가 (정밀, 역설 해소)
**역설**: "permitted 값이 결국 should_succeed와 (87%) 같으면 C-none과 무슨 차이?"
**해소**: 차이는 *값*이 아니라 **출력 시퀀스의 scaffold**다.
- **C-none**: `ready=true; preconds_verified=true; permitted=<should_succeed>; ACT/STOP` — permitted을 **단일 토큰으로 콜드 예측**(추상 라벨). 게더한 truth가 출력에 안 들어옴 → 모델이 게더결과를 결정에 *연결 안 함* → false로 추측.
- **T1c(treeval@s1)**: `ready=true; gate = AND(cond_a=true, OR(cond_b=true, cond_c=false), ...) = <val>; ACT/STOP` — **게더한 leaf-truth를 먼저 emit하고 그 AND/OR로 gate 도출**. 방금 "cond_a=true, cond_b=true…"를 쓴 직후라 **local consistency로 `=true`를 쓰게 됨**(콜드 false 추측이 아니라 *방금 쓴 값들의 fold*). = Kim&Suzuki 중간단계-supervision 효과(최종값 같아도 *학습가능성*이 바뀜).
- **★결정적 (리뷰 강화): s1·s3 treeval teacher 타깃은 BYTE-IDENTICAL**. 둘 다 `task["constraints"]`서 생성(L341, 모델생성 아님)=correct-size. **fabrication은 *inference-time* 아티팩트**: s3선 모델이 *안 보여준* 트리를 emit하도록 학습돼 발명(pay_bill 10op); **s1은 트리를 프롬프트가 공급 → 발명할 게 없음**. → **train-time 유일 변수 = 프롬프트(구조 given vs not), 타깃 동일 = source의 가장 깨끗한 isolation**. 3중 차단: ①source=1 구조 given ②leaf=관측치 ③트리 얕음(1-2 level).
- **요지**: T1c = "permitted을 *emit된 관측 truth의 grounded fold*로 scaffold" → 게더결과가 결정을 끌어올림. source=1이 fabrication 3중 차단.

## 3. teacher 타깃 포맷 (정확히 — 기존 treeval 코드)
gather 스텝(변경 없음): `ready=false; <다음 required 도구>` (required 전부 executed까지 = **complete-gather readiness 이미 내장**, premature 차단).
terminal(treeval grounded, `build_tbox_planner_sft.py` L382-406 기존):
- **grounded emit (val==should_succeed, ~87%)**: `ready=true; gate = <treeval_expr(constraints, observed)> = <true|false>; <ACT|STOP>`. expr = 주어진 constraint 트리를 관측 truth로 채운 AND/OR (source=1이라 모델이 *주어진* 구조 재현).
- **fallback (val≠should_succeed, ~13%, 구조-불충분)**: `ready=true; preconds_verified=true; permitted=<should_succeed>; <target>` (콜드, 잔여). = treeval 기존 agreement-gate(L401-406) 그대로. consistency 보장(emit한 gate val == 결정).
- 결정(ACT/STOP) = should_succeed (권위 GT; 벤치 evaluator 기준 — should_F를 ACT하면 안 되므로 결정은 GT 고정).

⚠️**T1c = `--treeval --source 1` (신규 코드 0)**. v3는 `--treeval` + alias_s3(=source 3)였음. **유일한 변경 = source 3→1**. teacval_reduce(inductive)는 OFF(단일식 grounded gate, inductive는 NULL).

## 4. 구현 (거의 무구현 — 플래그 조합 + 가드)
- **primary: `build_tbox_planner_sft.py --alias --source 1 --treeval`** (inductive 미사용). 기존·디버깅된 코드. 파일 태그 = `_alias_gate_scratch_treeval`(source=1→ _s 없음).
- eval: `SOPBENCH_SOURCE=1 SOPBENCH_ALIAS=1 SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_PLAN_MAXTOK=1024`, 버그수정 client(no-400 적용됨).
- 드라이버: `rung1_t1c.sh`(신규, rung1_agent2_upperbound.sh 복제 후 C-arm을 `--treeval` 추가로). control = C-none(이미 학습됨 `qwen7b_tbox_ub_s1_lodo_bank`, 재사용 → eval만) vs T1c(신규 `--treeval --source 1`).
- **변경 코드 없음**(treeval·source·버그수정 전부 기존). → 신규 버그 위험 최소.

## 5. 실험 설계
| arm | teacher | 비고 |
|---|---|---|
| **C-none** (control) | `--alias --source 1 --scratchpad` (기학습 `ub_s1`) | permitted=should_succeed(cold). eval 재사용 or 재eval |
| **T1c** (treeval@s1) | `--alias --source 1 --treeval` (신규 학습) | permitted=grounded AND/OR, source=1 fabrication 차단 |
| **treeval@s3** (★required 2×2) | 기존 어댑터 `alias_s3_treeval` | **버그수정 client서 re-eval 필수**(기존 BOTH4는 tool_choice 버그·n_T=45=비교불가). s1 vs s3 = fabrication-attribution 헤드라인 |
- **★2×2 (리뷰)**: {none, treeval(grounded)} × {s3, s1}. **T1c만 학습(4h)**; 나머지 3셀 eval-only: **C-none(`ub_s1`)·A(`ub_s3`) 재사용**(이미 fixed-client eval, n_T=48 — **재eval 금지**) + **treeval@s3 re-eval(~25분, idle GPU1)**. **interaction(grounding이 s1서 더 도움)이 헤드라인.**
- **★control 동일성 확인**: `ub_s1`(C-none)을 control로 재사용 — `rung1_agent2_upperbound.sh`의 COMMON(ep3·r16·seqlen2048·skip-overlong·val0.05·동일 LODO·n_train) 그대로라 T1c와 일치(터미널 포맷만 다름, step수 동일). 학습 전 재확인.
- 공통: alias=ON, LODO=bank, ep3, r16, SOLO, maxtok=1024, **버그수정 client(no-400)**. GPU0=T1c 학습, GPU1=treeval@s3 re-eval.

## 6. 지표·판정
- **헤드라인 BOTH(48)** + gathered_then_REFUSE(↓ 기대 29→소), premature, over-refuse, dirgraph, should_F STOP(비회귀 기준 nt 42%).
- **★fabrication 가드(필수)**: RLLOG terminal의 distinct-op vs 실제 #leaf — **s1서 과대생성 0 확인**(s3선 pay_bill 10op). ≈#leaf면 차단 성공.
- **over-gather 가드**: should_T 게더 step median (s3 treeval=10=cap; s1서 ≤6 기대).
- **terminal-reach%**, 조건수별 BOTH.
- **★format-mixing(리뷰 🟢)**: inference terminal 중 grounded `gate=AND(..)` vs fallback `permitted=` 비율; **grounded 중 최종 fold val ↔ emit된 ACT/STOP 모순율**(이게 높으면 "grounded인 척 콜드붕괴"=조용한 실패).
- **★should_F 대칭 회복(리뷰 🟢)**: should_F는 s1서 **false leaf→fold false→STOP**로 회복해야(s3선 fabricated all-true→STOP 20%로 붕괴). should_F STOP을 **grounded-false vs fallback**로 분해. nt 42% 쪽으로 회복 안 하면 scaffold가 대칭으로 작동 안 하는 것 = 헤드라인 BOTH만으론 안 보이는 진단.
- **판정**:
  - **T1c BOTH ≫ C-none(3)** → grounded-permitted가 답, gathered_then_REFUSE 29 전환 성공. → 다음 = 전이(§8 of source-ladder)+Agent1.
  - **T1c ≈ C-none(3)** but fabrication 0 → grounding-scaffold도 콜드붕괴 못 고침 → DPO(over-refuse dispreferred) 또는 §9 trace 증류.
  - **T1c < C-none** & fabrication↑ → s1도 fabrication 못 막음(예상 외) → `--gather_complete_gate`(flat, §7-옵션2) 구현.

## 7. ★리스크·함정 (실수 방지 체크리스트)
1. **fabrication 재발(s1서도)?** — 가설: source=1이 막음. **반드시 빌드후 타깃 + eval RLLOG서 distinct-op 측정**. s1 teacher 타깃의 gate expr가 주어진 #leaf와 일치하는지 육안+카운트.
2. **consistency**: grounded val ≠ should_succeed인 13%는 fallback(cold permitted) — treeval 기존 가드(L401-406)가 처리. 빌드후 "gate=X=true … STOP" 같은 모순 0 확인(treeval 단일식은 이미 검증됨).
3. **"값 같으면 차이 없다" 역설**: §2 — 차이는 scaffold(emit된 관측 truth)이지 최종값 아님. T1c≈C-none이면 scaffold도 부족 → DPO.
4. **over-gather**: treeval@s3는 게더 cap까지 갔음(fabricate 트리 채우려). s1은 구조 given이라 게더 명확 → over-gather 없어야. step median 모니터.
5. **decision=should_succeed 고정**: should_F를 grounded=true로 잘못 ACT하면 안 됨 → 결정은 GT, gate token만 grounded. (fallback이 이 일관성 보장.)
6. **inductive OFF**: `--treeval`만(treeval_reduce 미사용). inductive는 NULL.
7. **eval source 일치**: 학습 source=1 ↔ eval `SOPBENCH_SOURCE=1`. 불일치 = train/test mismatch(치명).
8. **버그수정 client 사용**: `_resolve` no-400 (HEAD ≥ 434c515). 드롭0·레이스0 확인(upperbound서 검증됨).

## 8. 선결 검증 (학습 전 게이트)
1. SYNTAX OK + 빌드 성공.
2. **타깃 육안**: should_T 완전게더 터미널 = `ready=true; gate = AND(<주어진 leaf들>=관측값) = true; ACT`. **gate의 op 개수 == 그 goal의 실제 #leaf**(과대생성 0). grounded% ≈ 87, fallback ≈ 13.
3. consistency: gate val == 끝 ACT/STOP 일치 0 위반.
4. overlong(>2048) 비율 < 5%.
통과 시 학습 launch.

## 9. 코드/파일 맵
| 파일 | 역할 |
|---|---|
| `build_tbox_planner_sft.py` | `--treeval --source 1`(기존, 변경 없음). treeval_expr grounded gate + agreement fallback |
| `two_stage_client.py` | source=1 렌더·`SOPBENCH_PLAN_MAXTOK`·`_resolve` no-400(기존) |
| `rung1_t1c.sh` | 신규 드라이버(upperbound 복제, C-arm에 `--treeval`) |
| `rung1_agent2_upperbound.sh` | 직전(복제원, C-none baseline 보유) |
| `RUNG1_SOURCE_LADDER_DESIGN.md` | 상위 설계(2-agent·source 축) |

## 10. 다음 세션 실행 순서
1. 이 설계서 리뷰(특히 §2 메커니즘·§3 포맷·§7 함정). 적대검증.
2. `rung1_t1c.sh` 작성(upperbound 복제 + `--treeval`). 변경 코드 없음 확인.
3. **빌드 + §8 선결 검증**(타깃 육안: gate op수==#leaf, fabrication 0). 통과해야만 학습.
4. 학습(C T1c, GPU0 SOLO, ~4h) → eval s1 maxtok=1024 → 전수조사(gathered_then_REFUSE 전환·fabrication·over-gather).
5. **§6 판정** → 권위본 `Exp-4-rung1-T1c` 기록.
6. T1c 성공 시 → DPO 잔여 + 전이(LODO 도구변경) + Agent1(NL→구조) 2-LoRA(§11 of source-ladder).
