# 능력×레버 배정 가이드라인 — ★논문 목표 (2026-06-21·사용자 reframe)

> **사용자 지시**: "기능을 더 세분화하고, 비용-효과 관점에서 각 기능을 효율적으로 *조합할 가이드라인을 설계*하는 것이 논문 목표. + 큰 모델이 하는 일을 작은 모델에서 *더 적은 구조/학습으로* 따라잡는 방법을 제시."
> 상위 = `THESIS_STATEMENT_2026_06_16`(분담 명제·이 doc가 §3 라우팅표를 *완전 가이드라인*으로 확장)·`A2_MINIMIZATION_FRONTIER_DESIGN`·`M_A_RESULTS §35`. 불변 = [[00-thesis]][[05-fixed-vs-variable]][[13-absorption-priority]].

## 0. 논문 목표 (두 기둥)
1. **배정 가이드라인**: agentic tool-use를 *세분 능력*으로 분해 → 각 능력을 *비용-효과 최소 레버*(prompt / 결정론 scaffold / A2-config / 최소학습 / scale)에 배정하는 측정-보정 의사결정 프레임워크.
2. **★스케일-대체 방법집(less structure/learning)**: *스케일이 사는* 능력 각각을, 작은 모델 + *최소* 개입(엔진 offload·최소 전이학습·구조변경)으로 따라잡는 구체 방법 — **"scale은 단일체 아니라 *분해가능 번들*이고, 각 조각이 scale보다 싸게 얻어진다"** 를 실증.

## 1. 신규성 (rival 회피)
**"작은>큰"(ToolOrchestra가 다툼) 아님.** 기여 = (a) 측정-보정 *배정 가이드라인* + (b) *스케일-대체 방법집* + (c) 명제 "scale 능력은 decidable/behavior/irreducible로 분해되고 대부분 scale보다 싸게 설치된다 — *어디까지 가능하고 어디서 막히는지*(genuinely scale-bound) 경계를 측정." = systematization + cheap-replication + honest boundary.

## 2. ★세분 능력 분류 (atomic·정의속성·스케일거동·증거)
| # | atomic 능력 | 정의 속성 | 스케일 거동(실측) | 레버 |
|---|---|---|---|---|
| C1 | provenance-check (arg값∈{user,tool}?) | **decidable 술어** | — | **scaffold** |
| C2 | producer-식별 (값 X의 생산도구) | **decidable**(I/O 스키마) | — | **scaffold/A2** |
| C3 | fetch-first **default**(날조 대신 가져옴) | scale-emergent *behavior* | A 76→23→3 | **scaffold-autofetch** or 최소학습 (←scale 대체) |
| C4 | dependency/sequencing 순서 | **decidable**(data-dep DAG) | — | **scaffold** |
| C5 | identity-before-scoped(인증선행) | **decidable**(policy precond) | — | **A2(gate_spec)+scaffold** |
| C6 | precondition/policy gate | **decidable**(policy replay) | — | **A2(gate_spec)+scaffold** |
| C7 | confirm-before-write | **decidable**(gate) | — | **scaffold** |
| C8 | error-recovery(진단→다르게 재시도) | scale-emergent *behavior* | too_many_err 7B36→32B0 | ★방법집(§5)·scale 대체 후보 |
| C9 | selection-resolution(기준→item) | **decidable**(resolve 엔진) | — | **scaffold** |
| C10 | **operand/value-formalize**(NL서 옳은 값/속성) | **irreducible NL→formalize** | ✗plateau 83→62 | **최소 전이학습**(유일 잔여) |
| C11 | flow-rule following(일반 규칙 적용) | promptable@small | D 7B −4 | **prompt** |
| C12 | NL communication | scale-emergent | INFO 47→7 | scale or 방법집(§5) |

## 3. 레버 비용 (3축·단일순서 금지)
| 레버 | build(1회) | 도메인당 TCO | 효과(크기 의존) |
|---|---|---|---|
| prompt | 최저 | 도메인-일반=~0 | **작은모델 무효/역효과**·32B만 |
| scaffold(엔진) | 중(1회) | **~0**(고정·grep if-domain=0) | 전 크기(decidable) |
| A2/ABox | — | **recurring tail**($50-150k/yr) | — |
| 학습(LoRA) | 고(GPU) | 전이·무망각이면 ~0 / 도메인타깃이면 高 | irreducible에 |
| scale | 추론비↑영구 | 0 | 큰 능력 일괄 |
⇒ **결정규칙 = "능력별, 효과 있는 *최소비용* 레버"** (build×TCO×효과 3렌즈).

## 4. ★배정 가이드라인 = decidability-first 의사결정 절차
```
각 능력 C에 대해 (비용 오름차순 단락평가):
 1. C가 decidable 술어/계산(스키마+policy+state서 도출가능)인가?  → SCAFFOLD(엔진). [scale-불변·도메인당~0]
 2. 아니면 C가 도메인-특정 사실(카탈로그/정책/vocab)인가?         → A2/ABox.       [환원불가 도메인비용]
 3. 아니면 C가 배포 크기에서 promptable인가?                      → PROMPT.        [효과있으면 최저]
 4. 아니면 C가 scale-emergent behavior이고 크기 감당되나?         → SCALE or §5 방법집.
 5. 그 외(irreducible NL→formalize·promptable 아님·scale-bound)  → 최소 전이학습(LoRA·무망각). [최후·잔여]
```

## 5. ★★스케일-대체 방법집 (둘째 기둥·"더 적은 구조/학습으로 scale 따라잡기")
스케일이 사는 능력(C3 fetch-first·C8 recovery·C12 NL)을 *최소* 개입으로:
- **C3 fetch-first default**: scale 없이 **엔진 autofetch**(provenance-deny→producer 결정론 호출→실값 주입). = decidable offload·학습 0·구조 최소. **실증중**(S-min arm1a: A 33→15·arm1b 진행). ⇒ scale의 최대선물(A)을 *엔진 한 조각*으로 대체.
- **C8 error-recovery**: 후보 = (a)결정론 retry-controller(같은 실패호출 차단·diff 강제) (b)reflection scaffold (c)최소 LoRA(recovery-default). **딥리서치(w23rp33zx)가 어느 게 싸고 무붕괴인지 조사중.**
- **C12 NL-comm**: 일부 genuinely scale-bound 가능(정직 경계) — 딥리서치 판정.
- **C10 operand**: 최소-rank LoRA + replay + 무망각(딥리서치 §3) — 유일 학습 잔여.
- ★공통 제약(사용자): **less structure / less learning** = 최소개입·붕괴망각 0·일반화 손상 0. 딥리서치가 PEFT/steering/distill/구조변경 중 최소비용·무붕괴 방법 확정.

## 6. 실증 = 능력×레버×스케일 매트릭스 (보정)
- 축: 능력(C1-C12) × 레버(5) × 크기(7/14/32B·frontier).
- 측정됨: 스케일 분해(§M_A_RESULTS §35)·프롬프트 크기의존·provenance gate(A 33→15)·operand plateau.
- 진행: arm1b(autofetch=C3 대체)·딥리서치(C8/C10 방법·무붕괴).
- 펜딩: C8 retry-controller 실험·operand 최소LoRA·전이(airline ABox-swap)·각 셀 GO/NO-GO.

## 7. GO/NO-GO (가이드라인 검증)
- **GO**: 각 scale-bound 능력이 §5 방법으로 *작은모델+최소개입 = 큰모델*에 도달(무붕괴·전이) → 가이드라인 실증·"scale 분해가능" 입증.
- **NO-GO(정직 경계)**: 어떤 능력이 §5 어느 방법으로도 작은모델서 회복 불가 → **그게 genuinely scale-bound** = 논문의 정직한 한계선(이것도 기여=경계 측정).

## 8. 정직 한계 (bitter-lesson 대비)
일부 능력(C12? 복합추론?)은 scale-bound로 남을 수 있음. 논문 주장 = "*전부* 싸게 대체"가 아니라 **"어느 능력이 어느 레버로 얼마에 회복되고, 어디가 진짜 scale 벽인지의 *측정된 지도*."** = 과주장 금지·경계가 곧 기여.
