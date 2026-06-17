> **★방향 갱신 (2026-06-17) = [`../LIE_ABSTRACTION_THEORY_2026_06_17.md`](../LIE_ABSTRACTION_THEORY_2026_06_17.md) §7f**: 전이(C8) 재정식화 = "내재화 *가능한 추상*(절차-타입 분류·생성원)을 *고립 학습*". ⇒ 학습타깃 = concrete/정적-criteria 아니라 **절차-타입(연산 명명)**. 정적 $select=해로움 실측·**연산-IR(LLM 명명+엔진 실행)이 7B서 극복**(rank 0.17→1.00). M-σ 등방화 골격 재사용·**타깃만 절차-분류로 교체**. C8 = 절차-분류가 held-out 도메인 전이하나.

# M-σ v4 (리뷰용) — 타깃-레벨 통일 코퍼스(SOP+TB+CFB 재추출 + Synth) + 재추출-진단 lead → held-out τ² 전이 — 2026-06-16

> ★설계 전환(v3→v4): v3는 **순수-synth factorial**(추상 substrate만·축 격리)이었다. v4 = **타깃-레벨로 통일된 union 코퍼스** + **재추출-진단(실험0)을 lead로**. 3라운드 리뷰 합의 박제.
> 상위 = `M_SIGMA_V3_TRANSFER_FACTORIAL_DESIGN.md`(factorial 상세)·`M_SIGMA_V2_SYNTH_DESIGN.md`(substrate §1-3)·`THESIS_STATEMENT_2026_06_16.md`. 불변 = [[feedback-thesis-tbox-transfer-direction]](τ² 참조/학습 금지)·[[feedback-nl-formalize-llm-selection-deterministic]](LLM=formalize·concrete=결정론)·[[feedback-selector-verifier-deterministic]](검증기 결정론).

## 0. 한 줄 + ★통일 원리
**코퍼스를 *도메인*이 아니라 *타깃 레벨*로 통일한다: 전 gold = provenance-typed spec(literal/$ref/$select), concrete 일절 없음. 실벤치(SOPBench/TaskBench/CFB)는 *재추출*로 자기가 가진 primitive(P-gate/thread/fetch)를 typed로 공급하고, Synth는 실벤치에 *없는* P-select(변형선택)를 배타 공급. 학습 전, 같은 데이터·같은 파이프로 {concrete-target vs typed-target} matched 쌍을 돌려 "타깃-레벨이 binder인가"를 0-synth로 진단(실험0)하고, 그 per-provenance split이 synth 배타 territory를 *치수*한다.**

- **★v4-v7 = negative control (데이터독성 아니라 타깃-레벨).** 세 autopsy 공통분모 = concrete-emit·selection 부재: 날조(order_id 지어냄=$ref 해야 할 걸 concrete로)·full-catalog 붕괴([[project-nativefc-fullcatalog-collapse]]·tool-selection 미학습)·write-벽([[project-tau2-write-failure-rootcause]]·new_item_ids만 틀림=variant-selection 미학습). ⇒ 실패는 *소스 데이터*가 아니라 *학습 타깃이 selection-레벨 아래(concrete)*였던 것. 재추출로 typed화하면 같은 궤적이 유효 신호.
- **★negative control은 *matched 쌍*이지 *역사적 v4-v7*이 아니다.** 역사 run은 ISO부재·다른 mix·v9 anti-fab DPO로 교란 → "실패=타깃레벨 탓" 단독귀속 불가. **같은 데이터·같은 재추출 파이프·차이는 {concrete vs typed} 하나뿐**인 쌍이 진짜 통제군(실험0).
- **thesis 정합**: 통일원리는 도메인-union보다 선명 — "TBox=벤치-횡단 도메인-일반 스킬"(thesis 문자)을 *타깃-레벨 불변*으로 조작화. 분업 = `DECOMPOSITION_OPTIMALITY.md`와 동형.

## 1. step 0 — subtract-coverage 맵 (zero-cost·진단 전제·[[feedback-zero-cost-diagnosis-strongest-case]])
**무엇을 박나**: ① 4족(P-gate/thread/fetch/select)을 기존 `PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md`의 **P1-P9 위에 매핑** ② **scope-partition**(아래) ③ τ²의 NL→formal 결정을 **arg별 provenance 라벨**(literal / passive-$ref / proactive-gather / $select) — 실험0 판독의 전제.

- **★scope-partition (수렴 아님·collapse 금지)**: 이 실험이 *측정*하는 primitive = **P-select(단발 formalize 전이)**. *유보*하는 primitive = **P6(confirm)·P7(recovery)** = multi-turn control-flow. v3 §5 단발 eval은 P7을 *측정조차 못 한다*([[project-v9-dpo-antifab-result]]서 복구 2→8 *독립* 이동). "P6/P7→P-select 수렴"은 구별되는 primitive를 단일 갭-스토리로 collapse = 세탁의 역방향 → **금지**. 매트릭스에 측정/유보 경계를 명시(유보는 별 트랙·§7 bridge서 정산).
- **subtract**: τ²-요구 NL→formal 추상 − (SOP∪TB∪CFB 커버) = **잔차**. 잔차가 *τ²-특정*이 아니라 *일반 primitive*(어떤 selection 벤치도 change/keep/fallback 안 가르침)로 기술되는지가 anti-targeting 1차 합격선(진짜 증명=§8 multi-target).
- **분업표**:

| primitive | 소유 벤치 | typed 공급 | 비고 |
|---|---|---|---|
| **P-gate**(정책·precondition) | SOPBench | literal/gate | |
| **P-thread**(tool-DAG 인덱싱) | TaskBench | $ref(passive) | |
| **P-fetch**(grounded 값 obtain) | CFB | $ref + **proactive-gather** | CFB native-FC subset-tool confound 별표([[project-nativefc-fullcatalog-collapse]]) |
| **P-select**(변형선택·change/keep/fallback) | **Synth(배타)** | $select | 실벤치 *재추출 불가*(에피소드 부재) |

## 2. ★실험 0 (lead·첫 GPU·진단) — 재추출 matched 쌍 → per-provenance split
**설계**: 동일 실벤치 데이터(SOP/TB/CFB) → 동일 재추출 파이프 → **두 arm만 차이 {concrete-target} vs {typed-target}** → held-out τ² eval.
**판독 = per-provenance split (binary 금지·3-way)**: 소스벤치에 selection 에피소드가 없으므로 결과는 *부분 사전결정*. 사전등록 예측:

| provenance 버킷 | 재추출 가능? | 예측 | 함의 |
|---|---|---|---|
| **passive-$ref**(obs에 이미 있음→ref) | ✅ | **개선**(날조↓) | 타깃-레벨이 grounding의 binder 입증 |
| **proactive-gather**(getter 능동호출 후 ref·P2b/R4) | ❌(control-flow) | **평탄 가능** | residual #1(synth/RL territory·v7 미해결층) |
| **$select**(new_item_ids 변형선택) | ❌(에피소드 부재) | **평탄** | residual #2(P-select=Synth 배타·치수됨) |

- **★split이 박는 것**: (i) 타깃-레벨이 *추출-가능 부분*($ref grounding)의 binder다 (ii) P-select·proactive-gather가 *추출-불가*임을 **치수**(두 residual territory 동시 사이즈). **단 입증되는 건 *음성·치수*까지** — "synth가 그걸 *고친다*"는 §7 factorial 몫(여기서 미끄러지지 말 것).
- **오독 가드**: 2-way("$ref개선/$select평탄")로 읽으면 proactive-gather가 $ref에 섞여 "타깃-레벨 부분무효"로 오독 → **반드시 3-way**.
- **n**: per-provenance split이 n을 쪼갬 → 집계 n≥50이면 $select 버킷 ~15-20. **관심 최소버킷($select)에 맞춰 n** (집계 n 아님), 안 그러면 split이 "noise/애매".
- **"0-synth ≠ 0-infra"**: 실험0은 resolver($ref)·재추출기·고친 harness 필요(synth *생성*만 0). 그 infra는 전 하류 공유 → 첫 빌드로 정확(synth 없이 더 싼 kill-switch).

## 3. 코퍼스 (실험0 양성 후) — 타깃-레벨 통일 union
- 실벤치 gold → **typed 재추출**(concrete 아님): SOP→P-gate, TB→P-thread($ref), CFB→P-fetch($ref + proactive-gather). 
- **Synth → P-select 배타**(substrate = v2 §1-3·추상 selection-by-criteria·랜덤 스키마·changes/keep/fallback·provenance-typed). 
- **혼합비 ablation 동반**(raw union 금지·§5 ISO 함께).

## 4. Synth substrate (P-select·v2 §1-3 승계)
- 각 예제: 랜덤 추상 스키마(K속성·값-vocab)→카탈로그(M item+item_id+available)→current→**NL 요청**(패러프레이즈·"X 바꾸고 유지·없으면 Y 완화")→**gold = {current⊕changes} 매칭 available item_id**(미가용시 fallback).
- 학습타깃 = `$select{from,by,fallback}` / `$ref` / `literal`(concrete 아님·resolver 해결). root-cause 결박(v2 §1a·`M_A_RESULTS §3`): write-벽=**구조적 변경-오계산**, 어휘 아님 → 어휘 grounding은 ABox 몫.

## 5. ★ISO = per-primitive (표면만·관계 보존)
- **규칙**: ISO는 각 primitive의 *입력 표면*을 randomize, *정의 관계*는 보존.
  - **P-fetch**: tool/field명만 randomize, **value-provenance(must-fetch) 보존** — 값 randomize는 P-fetch 신호를 *지움*(v1 isotropize가 도구명만 바꾼 게 이 점서 옳았음).
  - **P-select**: 속성명/값-vocab randomize, **change/keep/fallback 관계 보존**.
  - **P-gate**: 게이트 술어 표면 randomize, **precondition 논리 보존**.
- **귀결**: per-primitive면 factorial의 ISO 축은 *비원자*(이종 처치 묶음) → "ISO main effect"는 이종 평균 → 헤드라인이 ISO면 비원자성 명시(v3 ISO-confound 연장).

## 6. union-ablation (applied 헤드라인)
- **{real(표면-ISO 재추출)} vs {real + Synth}** → held-out τ² per-provenance. **혼합비 ablation 동반**.
- primary = τ² **new_item_ids selection 정확률**(base 0.41·cfb-Mσ 0.34 퇴화·`M_A_RESULTS §11`). secondary = all-arg(base 0.41·Mσ 0.03)·over-$ref율.
- bar(사전등록·post-hoc 금지): new_item_ids **≥ base+2σ**(≥+12pp 잠정·n은 §2 최소버킷 기준) **∧ over-$ref↓** **∧ all-arg ≥ base 회복**(cfb-Mσ식 0.41→0.03 퇴화 안 냄).

## 7. factorial (mechanism·union과 독립·서로 비-게이트)
- v3 2³(ISO×NL×PROV·순수 synth substrate) 그대로 = 어느 축이 *왜* 전이? **union 부호와 무관하게 실행**(순수-synth FULL arm 자체가 τ² 독립 전이신호).
- **★격상 시험(공짜 산출)**: **synth-FULL vs union-FULL** 대조 = "실벤치 재추출이 synth-only 위에 *무엇을 보태나*". 동률이면 union의 격상-주장 붕괴(순수-synth 회귀). 보강 #4(병행)의 진짜 배당.
- 검정력: interaction은 n≥50서 미검출 가능(정직)·헤드라인=main effect·**tier1(M0+3단일+FULL=5arm)** 먼저·tier2(조합 3셀)는 단일 모호 ∧ n 상향 시만.

## 8. 2차 타깃 전이 (P-select generality·진짜 anti-targeting 증명)
- subtract-맵 서술-합격선은 *세탁 가능* → **2차 held-out selection 타깃**(WebShop·τ²-airline 등)으로 전이해야 일반 primitive 입증.
- **blind 설계**: 2차 타깃·synth 분포를 *타깃 specifics 보기 전* 사전등록(leave-one-bench-out). 봤다 튜닝하면 multi-target도 세탁.
- **generality는 graded**: "N개 독립 held-out 분포 전이"가 조작적 정의·천장 주장=테스트 집합으로 한정.
- **★후보 검증 필수**(이름으로 믿지 말 것·[[reference_sopbench_bench_defects_settled]] 규율): WebShop=속성-selection(P-select 가능)·AppWorld=orchestration(P-thread 공산)·BFCL multi-turn(P-fetch/thread 공산). **후보별 mini subtract-map으로 "정말 P-select 행사하나" 확인** 후 커밋. 의존: 딥리서치 완료(field-orphan 정당화·후보 수확) — **이 step만 게이트**(lead는 의존 0).

## 9. ★τ² pass@1 bridge (컴포넌트→task·scope-partition 정산처)
- M-σ 전부 = *단발 formalize 전이*. multi-target P-select 전이 완벽도 "서브스킬 전이"지 "τ² task pass" 아님 → **최선 arm → 풀 τ² agentic harness → pass@1 vs base**가 thesis 헤드라인 변환.
- **★per-primitive 분해 동반(필수)**: pass@1은 multi-turn → 유보한 P6/P7을 도로 entangle. **pass@1 평탄 ∧ new_item_ids↑ = "P-select 전이됨·P6/P7이 task를 cap"** = 유보 primitive 트랙(P7 RL) 청구 지점 = §1 scope-partition이 정산되는 자리. 분해 없으면 평탄 pass@1을 "P-select 실패"로 오독.

## 10. 위험 (정직·리뷰 훅)
1. **주장 분리**: 실험0=음성·치수까지만(synth 작동 아님·§2 R2). union-ablation 음성=커버리지fill 무효 vs synth recipe 틀림 모호 → factorial 독립실행이 mechanism 분리(§7).
2. **검정력**: per-provenance split·interaction 모두 버킷 n에 민감 → 최소버킷 기준 n·interaction은 방향만.
3. **proactive-gather residual**: 실험0 날조-fix 부분적이면 원인이 P2b/R4(능동 getter 호출)일 수 있음 → "타깃-레벨 무효"로 오독 말 것(별 residual·§2).
4. **혼합 간섭**: 실벤치 재추출에도 per-primitive ISO 적용 안 하면 표면 과적합 재교육(§5)·혼합비는 설계변수.
5. **딥리서치 의존**: §8 2차 타깃·field-orphan 정당화는 미산출 의존(손에 든 것만 주장·[[feedback-arxiv-citation-discipline]]). lead(step0+실험0)는 의존 0·즉시 실행.
6. **결정론 분담 유지**: concrete는 절대 학습타깃 아님(resolver). 새면 v4-v7 재현.

## 11. 순서 + infra build order + 비용 tier
0. **step 0**(zero-cost): subtract-map(P1-P9×4족·scope-partition·provenance 라벨).
1. **infra**(하류 공유): 재추출기·resolver($ref+$select)·고친 harness(payment=값·$select 채점·n≥최소버킷·over-$ref율·구조/어휘 autopsy 라벨).
2. **실험 0**(첫 GPU·진단): 재추출 {concrete vs typed} matched 쌍 → per-provenance 3-way split. = 타깃-레벨 입증 + 두 residual territory 치수.
3. **Synth 생성**(P-select substrate·per-primitive ISO).
4. **union-ablation**(§6·표면-ISO real·혼합비) = applied 헤드라인.
5. **factorial**(§7·synth-FULL vs union-FULL=격상시험) = mechanism·독립.
6. **2차 타깃 blind 전이**(§8·후보 검증 후) = P-select generality.
7. **τ² pass@1 bridge**(§9·per-primitive 분해) = 컴포넌트→thesis.
- 결과 박제(`M_A_RESULTS.md §12`·split표/격상대조/bridge 분해).

## 12. 한 줄
**코퍼스를 도메인 아니라 *타깃 레벨*(provenance-typed everywhere)로 통일 — 실벤치 재추출(P-gate/thread/fetch) + Synth(P-select 배타). lead = 0-synth 재추출 matched 쌍의 per-provenance 3-way split($ref개선·proactive평탄·$select평탄)로 "타깃-레벨이 binder"를 진단하고 두 residual territory를 치수. 이어 union-ablation(applied)·factorial(mechanism·격상시험)·multi-target blind(generality)·τ² pass@1+per-primitive(thesis 다리). v4-v7=selection-less 타깃의 negative control(데이터독성 아님)·matched 쌍이 진짜 통제군.**
