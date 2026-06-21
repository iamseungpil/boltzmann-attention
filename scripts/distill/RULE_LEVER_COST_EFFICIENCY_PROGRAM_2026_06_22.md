# RULE × LEVER COST-EFFICIENCY PROGRAM — 규칙별 비용-효율 곡선 완성 (2026-06-22)

> **권위 프로그램 설계서**(사용자 지시 2026-06-22). 논문1 헤드라인 = `EXPERIMENT_DESIGN §0★★ eval#1(배정 정당성)`의 실행.
> spine = `CAPABILITY_LEVER_ALLOCATION_DESIGN_2026_06_21`(a-priori 배정표·이 프로그램이 *측정*으로 완성).
> 통제 프레임 = `LLM_CONTROL_EXPERIMENT_REDESIGN_2026_06_22`(통제점×강제강도). 이론 = `PRIMITIVE_COVERAGE_MATRIX`(P1-P9 closure).
> 첫 셀(worked example) = `C4_LEARN_FETCHFIRST_CROSSOVER_DESIGN_2026_06_22`(= 규칙 C3/P8 × 레버 전부).

---

## §0. 착수 전 [[05]] 결정질문 (이 프로그램이 정의하는 전 실험 cell)

1. **도메인-특화 순증?** — ❌ 모든 레버를 **도메인-일반 전이**로만 측정: learn=**canonical learn 벤치 = SOPBench + TaskBench + Synth**(★CFB 직접 폐기·그 P2b/P4=Synth의 cfbsynth 추상합성 stratum이 대체·`CFBSYNTH_P2B_P4_DESIGN`·`INTEGRATED_TBOX v2`·확인 2026-06-22) 학습→ABox-swap·hook=grep `if domain`/도구명 0·prompt/skill=도메인-일반 텍스트(특정 도구명/정책 금지). **tau2(retail/airline)=held-out 전이 eval *only*·학습 금지([[11]]).**
2. **유동성 동결?** — ⚠️ **측정대상.** flexibility-loss = 모든 셀의 비용항(§4). enforced 레버가 정당 유동성 죽이면 비용으로 계상.
3. **scaffold가 도메인 행동 수행?** — hook-perform(autofetch류)=레버 사다리 *최강*·결정질문3 yes ⇒ 측정으로만 정당화. 기본=더 약한 레버.

---

## §1. 사용자 지시 reframe

**fetch-first는 규칙 하나일 뿐.** 통제점×강제강도 프레임을 **모든 일반 규칙**(P1-P9·R1-R8·content-op·= 실험형 C1-C12)에 적용한다.
- 각 규칙을 **prompt / skill / hook / learn** *다(多)레버*로 비용-효율 측정(+scale 기준선).
- ★**규칙마다 비용-효율 곡선이 다르다** — 어떤 건 *얕아서*(prompt/skill로 포화) 싸고, 어떤 건 *깊어서*(hook/learn) 비싸다.
- **결과물 = 모든 규칙의 비용-효율 곡선 *완성* → 규칙별 *최소비용 레버* 지도**(measured 배정 가이드라인·assert 아님).

= "작은>큰" 아님. **규칙별 최소-레버 배정 가이드라인 + 깊이 지도**(systematization). [[13]] 흡수우선(scale→learn→scaffold)을 규칙별로 실측.

---

## §2. 규칙 행(行) — 정본 목록 (P1-P9 ↔ C1-C12 화해)

이론 closure(P)와 실험형 분해(C)를 한 표로. **측정 행 = C1-C12**(scale 거동·a-priori 레버 기보유)·**P/R-매핑**으로 "모든 일반 규칙" 커버 확인.

| 측정행 C | 규칙 | P/R | 정의속성 | scale거동(§35) | a-priori 레버(가설) | 예상 깊이 |
|---|---|---|---|---|---|---|
| C1 | provenance-check | P8/P1·R1b | decidable 술어 | — | hook(deny) | 얕음(decidable) |
| C2 | producer-식별 | P2b부분 | decidable I/O스키마 | — | hook/A2 | 얕음 |
| **C3** | **fetch-first default** | **P2b/P8** | scale-emergent behavior | A 76→3 | ★3분해: SHAPE=learn(cfbsynth)·의존맵=A2·집행=hook (`CFBSYNTH §9`) | **깊음(=worked ex)** |
| C4 | dependency/sequencing | P3·R4/R6 | decidable DAG | — | hook | 얕음 |
| C5 | identity-before-scoped | P8·G1/G3 | decidable precond | — | A2-gate+hook | 얕음 |
| C6 | precondition/policy gate | P5·G4 | decidable policy-replay | — | A2-gate+hook | 얕음(A2=난제) |
| C7 | confirm-before-write | P6·G2 | decidable gate | — | hook | 얕음 |
| C8 | error-recovery | P7 | scale-emergent·A×B경계 | too_many_err 36→0 | 방법집(retry/reflect/learn) | **깊음(RL/learn?)** |
| C9 | selection-resolution | P4·content-op | decidable resolve | — | hook(resolve) | 중(operand 얽힘) |
| C10 | **operand/value-formalize** | content-op·NL | irreducible NL→formalize | ✗plateau 83→62 | learn(최소LoRA) | **최깊음(유일 잔여)** |
| C11 | flow-rule following | P일반적용 | promptable@small | D 7B−4 | prompt | 얕음(promptable) |
| C12 | NL communication | (closure 밖) | scale-emergent | INFO 47→7 | scale or 방법집 | **깊음(scale-bound?)** |
| (C9') | content-op 8 | filter/argmax/argmin/rank/comparative/substitute/create/project | resolve+operand | rank 0.17→1.00(gloss) | hook(resolve)+prompt(gloss) | 혼합 |

- **P9 parallelism** = 성능최적화축(정확도 아님)·별도(저우선).
- **★예상 깊이 = 가설**(CAPABILITY_LEVER a-priori + §35 scale분해). 이 프로그램이 *반증가능 측정*으로 확정. "얕다 예상했는데 깊음"·"깊다 예상했는데 prompt로 포화" = 발견.

---

## §3. 레버 열(列) — 사다리 (싼→비쌈·도메인-일반 operationalize)

사용자 4레버 + scale 기준선. **enforcement·생애주기비용 오름차순.**

| 레버 | = 통제점 | 도메인-일반 구현 | 강제성 | 핵심비용 |
|---|---|---|---|---|
| **(scale)** | 기준선 | 큰 base(14/32/72B) as-is | soft | OpEx 영구최고(대체대상) |
| **prompt** | C0 | 도메인-일반 시스템 규칙/few-shot(도구명·정책 무) | soft(게임가능) | build 최저·OpEx 토큰·**작은모델 무효 가능** |
| **skill** | C0+ | **온-디맨드 절차모듈**(사용자 확정 2026-06-22): 규칙 발동 시 retrieval/router로 *invoke*되는 구조화 절차+few-shot exemplar. 컨텍스트 비용=발동시만. soft(모델이 따를지 선택)·단 prompt(always-on)보다 타깃·구조화 | soft·타깃 | build 중(모듈+라우터)·OpEx 발동시만·소형서 prompt보다↑? |
| **hook** | C1/C2 | gate 엔진(grep if-domain=0·A2-gated)·**deny<substitute<perform** 하위사다리 | **enforced** | build 중·도구왕복 OpEx·**flexibility-loss·A2-growth**(perform) |
| **learn** | C4 | primitive-벤치 학습 LoRA(small-rank+replay·무붕괴)→ABox-swap 전이 | **internalized** | build 최고(데이터+학습)·OpEx 0·**무망각 필요·flexibility-loss 0** |

- **hook 내부 강제강도 사다리**(`LLM_CONTROL §2`): deny(거부·유동성보존) < require(선행강제) < substitute(인자교정) < perform(엔진이 행동수행=autofetch). ★최소행동 원칙.
- **skill vs prompt 구분(핵심)**: prompt=always-on 정적 / skill=on-demand 구조화 절차(발동시만 컨텍스트)·composable. 비용·신뢰 다름 → 곡선상 별점.
- **레버 결합 가능**(곡선=envelope): 예 hook-deny + learn(=fetch-first crossover A1/A2 조합).

---

## §4. 비용-효율 메트릭 + 곡선 정의 (셀 균일)

**셀 = (규칙 C, 레버 L) → 측정:**

```
efficiency(C,L) = reliability_gain(C,L)
                / ( build + inference_OpEx + flexibility_loss
                    + A2_growth + learn_cost + maintenance )
```

| 측정량 | 정의 | 도구 |
|---|---|---|
| reliability_gain | 그 규칙의 failure 닫힘(Δpass·Δfailure-census) | `t2_failcensus`(규칙별 실패코드) |
| flexibility_loss | false-block·over-constrain rate(enforced만) | validate over-deny·held-out 정상경로 차단 |
| A2_growth | 그 레버가 요구하는 A2 필드 수 | gate.json diff |
| OpEx | 추론 토큰·도구왕복·레이턴시 | 런 telemetry |
| no_collapse | held-out 일반능력 불변(learn) | tbnfc eval |
| transfer | airline held-out(ABox-swap만) 유지 | 동일 LoRA/엔진·A2-swap |

**★곡선 = 규칙 C마다: x축=레버(생애주기비용 오름차순) · y축=reliability.**
- **knee = reliability 포화하는 최소비용 레버 = 그 규칙의 배정.**
- **얕은 규칙** = knee가 prompt/skill(왼쪽·싸다). **깊은 규칙** = knee가 hook/learn(오른쪽·비싸다). **decidable 규칙** = hook-deny가 싸게 포화(왕복비용만).
- 곡선에 flexibility-loss·A2-growth 주석(같은 reliability면 이 둘 작은 레버 승).
- **scale 기준선 = 수평선**(큰모델 도달치)·각 레버가 작은모델로 그 선 도달하나(=cheap-replication·[[13]] 둘째기둥).

---

## §5. a-priori 깊이 분류 (검증할 가설·§2 표)

§35 scale분해 + CAPABILITY_LEVER가 예측:
- **얕음(decidable→hook-deny 싸게·또는 promptable)**: C1·C2·C4·C5·C6·C7·C11. 곡선 knee 왼쪽 예상.
- **깊음(scale-emergent·learn/방법집)**: C3(fetch-first)·C8(recovery)·C10(operand)·C12(NL). 곡선 knee 오른쪽 예상.
- **혼합**: C9(selection·operand 얽힘)·content-op(rank=gloss로 얕아짐·comparative=명명 깊음).

**★헤드라인 가설 = 곡선 모양이 규칙마다 다르다**(얕음≠깊음). 깊은 규칙에서만 learn 정당·얕은 규칙은 hook/prompt로 충분 = **레버 낭비 회피 지도.** 반증 = 다 똑같은 곡선이면 프레임 무효.

---

## §6. 도메인-일반 전이 (모든 레버 공통 게이트)

각 레버가 **도메인 특화 없이** 규칙을 전이시키나가 핵심:
- **learn**: primitive 벤치 학습만(tau2 금지·jsonl tau2-도구명 grep=0)·retail+airline 둘 다 held-out 전이.
- **hook**: 엔진 grep `if domain`/도구명=0·A2-swap만으로 airline 작동(키스톤 검증됨).
- **prompt/skill**: 도메인-일반 텍스트(특정 도구명/카탈로그 금지)·retail+airline 동일 텍스트.
- **transfer 미달=그 (규칙,레버) 셀 음성**(도메인-특화로만 되면 [[05]] 위반·배정 불가).

---

## §7. 시퀀싱 (한 번에 불가·곡선 모양 다양 먼저 입증)

전 매트릭스(12규칙×5레버×전이×scale) = 큰 프로그램. **곡선-모양-다양 가설을 먼저 입증할 spread 선정:**

1. **C3 fetch-first(깊음 예상)** = `C4_LEARN_FETCHFIRST_CROSSOVER`(이미 설계·4레버로 확장) — knee 오른쪽?
2. **C11 flow-rule following(얕음·promptable 예상)** = prompt만으로 포화? knee 왼쪽 대조.
3. **C1 provenance-check(decidable·hook-deny 싸게 예상)** = hook 한 점 포화 대조.
4. → 3 규칙이 (오른·왼·hook) 서로 다른 곡선 = 가설 입증 → 나머지 규칙으로 확장.
5. **C10 operand**·**C8 recovery** = 최깊음·둘째기둥 핵심(별 설계서 상속).

각 규칙 = design→review→build→eval(GPU 리뷰 후). 진행률 가시([[30]]).

---

## §8. 결과물·GO/NO-GO

- **결과물 = 규칙별 비용-효율 곡선 아틀라스 + 최소-레버 배정 가이드라인**(measured). = `EXPERIMENT_DESIGN eval#1` 충족.
- **GO**: 곡선이 규칙마다 다르고(얕음/깊음 분리)·각 규칙 최소-레버가 측정으로 정해지고·전이 도메인-일반·scale-bound 규칙은 정직 식별 ⇒ 논문1 배정 가이드라인 성립.
- **NO-GO**: 곡선 구분 안 됨(다 prompt로 되거나 다 learn 필요) → 프레임 무효(정직 보고).
- **경계지도**: genuinely scale-bound(어느 레버로도 작은모델서 안 닫힘·C12 후보) = 정직 한계 기여.

---

## §9. 빌드 순서

1. ✅ 이 프로그램 설계서.
2. **사용자 리뷰** ← 멈춤(행/열/메트릭/시퀀싱 확정).
3. C3 fetch-first 4레버 확장(`C4_..CROSSOVER` 업데이트)·C11·C1 = spread 첫 3곡선.
4. eval → 곡선 모양 다양 입증 → 나머지 규칙.

---

**불변 정합**: [[05]](결정질문·tau2학습금지·minimize-A2)·[[07]](통제점×강제강도)·[[11]](전이=primitive학습)·[[12]](다양성)·[[13]](흡수우선 scale→learn→scaffold 규칙별 실측). 상위 = `EXPERIMENT_DESIGN §0★★`·`CAPABILITY_LEVER_ALLOCATION`.
