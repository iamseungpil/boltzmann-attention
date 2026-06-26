# RULE × LEVER COST-EFFICIENCY PROGRAM — 규칙별 비용-효율 곡선 완성 (2026-06-22)

> **권위 프로그램 설계서**(사용자 지시 2026-06-22). 논문1 헤드라인 = `EXPERIMENT_DESIGN §0★★ eval#1(배정 정당성)`의 실행.
> spine = `CAPABILITY_LEVER_ALLOCATION_DESIGN_2026_06_21`(a-priori 배정표·이 프로그램이 *측정*으로 완성).
> 통제 프레임 = `LLM_CONTROL_EXPERIMENT_REDESIGN_2026_06_22`(통제점×강제강도). 이론 = `PRIMITIVE_COVERAGE_MATRIX`(P1-P9 closure).
> 첫 셀(worked example) = `C4_LEARN_FETCHFIRST_CROSSOVER_DESIGN_2026_06_22`(= 규칙 C3/P8 × 레버 전부).

---

## §0. 착수 전 [[05]] 결정질문 (이 프로그램이 정의하는 전 실험 cell)

1. **도메인-특화 순증?** — ❌ 모든 레버를 **도메인-일반 전이**로만 측정: learn=**canonical learn 벤치 = SOPBench + TaskBench + Synth**(★CFB 직접 폐기·그 P2b/P4=Synth의 cfbsynth 추상합성 stratum이 대체·`CFBSYNTH_P2B_P4_DESIGN`·`INTEGRATED_TBOX v2`·확인 2026-06-22) 학습→ABox-swap·scaffold=grep `if domain`/도구명 0·prompt/skill=도메인-일반 텍스트(특정 도구명/정책 금지). **tau2(retail/airline)=held-out 전이 eval *only*·학습 금지([[11]]).**
2. **유동성 동결?** — ⚠️ **측정대상.** flexibility-loss = 모든 셀의 비용항(§4). enforced 레버가 정당 유동성 죽이면 비용으로 계상.
3. **scaffold가 도메인 행동 수행?** — scaffold-perform(autofetch류)=레버 사다리 *최강*·결정질문3 yes ⇒ 측정으로만 정당화. 기본=더 약한 레버.

---

## §1. 사용자 지시 reframe

**fetch-first는 규칙 하나일 뿐.** 통제점×강제강도 프레임을 **모든 일반 규칙**(P1-P9·R1-R8·content-op·= 실험형 C1-C12)에 적용한다.
- 각 규칙을 **prompt / skill / grammar / scaffold / learn** *다(多)레버*로 비용-효율 측정(+scale 기준선).
- ★**규칙마다 비용-효율 곡선이 다르다** — 어떤 건 *얕아서*(prompt/skill로 포화) 싸고, 어떤 건 *깊어서*(scaffold/learn) 비싸다.
- **결과물 = 모든 규칙의 비용-효율 곡선 *완성* → 규칙별 *최소비용 레버* 지도**(measured 배정 가이드라인·assert 아님).

= "작은>큰" 아님. **규칙별 최소-레버 배정 가이드라인 + 깊이 지도**(systematization). [[13]] 흡수우선(scale→learn→scaffold)을 규칙별로 실측.

---

## §2. 규칙 행(行) — 정본 목록 (P1-P9 ↔ C1-C12 화해)

이론 closure(P)와 실험형 분해(C)를 한 표로. **측정 행 = C1-C12**(scale 거동·a-priori 레버 기보유)·**P/R-매핑**으로 "모든 일반 규칙" 커버 확인.

| 측정행 C | 규칙 | P/R | 정의속성 | scale거동(§35) | a-priori 레버(가설) | 예상 깊이 |
|---|---|---|---|---|---|---|
| C1 | provenance-check | P8/P1·R1b | decidable 술어 | — | scaffold(deny) | 얕음(decidable) |
| C2 | producer-식별 | P2b부분 | decidable I/O스키마 | — | scaffold/A2 | 얕음 |
| **C3** | **fetch-first default** | **P2b/P8** | scale-emergent behavior | A 76→3 | ★3분해: SHAPE=learn(cfbsynth)·의존맵=A2·집행=scaffold (`CFBSYNTH §9`) | **깊음(=worked ex)** |
| C4 | dependency/sequencing | P3·R4/R6 | decidable DAG | — | scaffold | 얕음 |
| C5 | identity-before-scoped | P8·G1/G3 | decidable precond | — | A2-gate+scaffold | 얕음 |
| C6 | precondition/policy gate | P5·G4 | decidable policy-replay | — | A2-gate+scaffold | 얕음(A2=난제) |
| C7 | confirm-before-write | P6·G2 | decidable gate | — | scaffold | 얕음 |
| C8 | error-recovery | P7 | scale-emergent·A×B경계 | too_many_err 36→0 | 방법집(retry/reflect/learn) | **깊음(RL/learn?)** |
| C9 | selection-resolution | P4·content-op | decidable resolve | — | scaffold(resolve) | 중(operand 얽힘) |
| C10 | **operand/value-formalize** | content-op·NL | irreducible NL→formalize | ✗plateau 83→62 | learn(최소LoRA) | **최깊음(유일 잔여)** |
| C11 | flow-rule following | P일반적용 | promptable@small | D 7B−4 | prompt | 얕음(promptable) |
| C12 | NL communication | (closure 밖) | scale-emergent | INFO 47→7 | scale or 방법집 | **깊음(scale-bound?)** |
| (C9') | content-op 8 | filter/argmax/argmin/rank/comparative/substitute/create/project | resolve+operand | rank 0.17→1.00(gloss) | scaffold(resolve)+prompt(gloss) | 혼합 |

- **P9 parallelism** = 성능최적화축(정확도 아님)·별도(저우선).
- **★예상 깊이 = 가설**(CAPABILITY_LEVER a-priori + §35 scale분해). 이 프로그램이 *반증가능 측정*으로 확정. "얕다 예상했는데 깊음"·"깊다 예상했는데 prompt로 포화" = 발견.

---

## §3. 레버 열(列) (★용어 정정 2026-06-22·사용자: "hook"→established **scaffold**·xgrammar=별개 층)

**용어 원칙**: "hook"은 새 용어라 폐기 — established = **scaffold**(결정론 엔진·[[05]] §3 "군-실행+gate 집행+resolve+per-step verify"). self-hook(`scaffold_guard`=PreToolUse)은 *나에게 건 scaffold 유비*일 뿐 레버 이름 아님. **xgrammar(output TYPE 강제)는 scaffold-gate와 다른 메커니즘**(decode-time config·[[05]] 4-way: TYPE=xgrammar/CONTENT=LLM/concrete=결정기/변환=scaffold) → 별 레버 `grammar/config`.

| 레버 | 통제점·시점 | 도메인-일반 구현 | 강제성 | 핵심비용 |
|---|---|---|---|---|
| **(scale)** | 기준선 | 큰 base(14/32/72B) as-is | soft | OpEx 영구최고(대체대상) |
| **prompt** | C0·생성전 | 도메인-일반 시스템 규칙/few-shot(도구명·정책 무) | soft(게임가능) | build 최저·OpEx 토큰·**작은모델 무효 가능** |
| **skill** | C0+·온디맨드 | router/retrieval로 *invoke*되는 구조화 절차+exemplar(발동시만 컨텍스트·사용자 확정) | soft·타깃 | build 중(모듈+라우터)·OpEx 발동시만 |
| **grammar/config** | **decode-time** | xgrammar `guided_json`이 출력 **TYPE/enum/format** 강제(ABox A1 config·rigid concrete 강제는 폐기=validity≠correctness) | **enforced(decode)** | build 저·OpEx≈0·**적용가능=TYPE/format-표현 규칙만**(behavioral 규칙엔 N/A) |
| **scaffold** | C1 pre-call·C2 post-call | gate/resolve/provenance 엔진(grep if-domain=0·A2-gated)·**deny<substitute<perform** 하위사다리 (구 "hook") | **enforced(post-emit)** | build 중·도구왕복 OpEx·**flexibility-loss·A2-growth**(perform) |
| **learn** | C4·오프라인 | primitive-벤치 학습 LoRA(small-rank+replay·무붕괴)→ABox-swap 전이 | **internalized** | build 최고(데이터+학습)·OpEx 0·**무망각 필요·flexibility-loss 0** |

- **★두 날개(§4a)**: **내재화 날개**=prompt/skill/learn(모델이 규칙 갖춤·flex/A2=0) / **offload 날개**=grammar/config(decode 제약)+scaffold(gate/resolve·flex/A2>0). 한 x축 아님.
- **scaffold 내부 강제강도 사다리**(`LLM_CONTROL §2`): deny(거부·유동성보존) < require(선행강제) < substitute(인자교정) < perform(엔진이 행동수행=autofetch). ★최소행동 원칙.
- **grammar/config vs scaffold(둘 다 offload·다른 메커니즘)**: grammar=*decode 시 출력공간 제약*(형식 위반 불가능화·TYPE만)·scaffold=*emit 후 툴콜 gate/resolve*(행동·값 검증). 같은 "hook"으로 묶으면 category error(사용자 지적).
- **레버 결합 = 분해 책임**(envelope 아님): 예 fetch-first = learn(SHAPE)+A2(의존맵)+scaffold(provenance 가드) 3분해(`CFBSYNTH §9`).

---

## §4. 메트릭 — ★Pareto + 두-날개 (단일 스칼라 efficiency·단일 knee 폐기·리뷰 #1·#2)

### §4a. ⛔ 폐기: 단일 cost선 위 knee (리뷰 #1)
**prompt<skill<scaffold<learn을 *한 x축*에 놓고 단일 knee를 읽는 건 잘못된 객체.** prompt/skill/learn = **모델이 그 능력을 갖추나(내재화 날개)** · scaffold = **능력을 불필요하게 만드는 결정론 offload 날개**([[00-thesis]] 두 날개). 종류 선택이지 단조곡선 한 점 아님. 한 축에 섞으면 "얕음/깊음"이 *"작은 모델이 학습가능한가"* 와 *"결정론으로 떼낼 수 있는가"* 를 뒤섞음. **6항 이종단위·가중치 미지정 스칼라로 collapse하면 레버 순위 안 나옴.**

### §4b. ✅ 규칙당 산출 = Pareto 집합 + 어느 날개가 이기나
각 (규칙 C, 레버 L) → **벡터** `(reliability, flexibility_loss, A2_growth, lifecycle_cost)` 측정 → 규칙당:
1. **Pareto 집합**(지배되는 레버 버림·가중치 부여 거부=정직).
2. **두 날개 판정**: ⓐ**내재화 날개**(prompt→skill→learn) = 모델이 규칙 갖추나·flex_loss·A2_growth=0. ⓑ**offload 날개**(scaffold-deny→substitute→perform) = 결정론이 규칙 떼내나·flex_loss·A2_growth>0. **규칙당 "어느 날개가 Pareto-지배하나".**
3. **곡선/knee는 *날개 내부에서만***: 내재화 날개의 soft 사다리(prompt→skill→learn·"작은모델이 scale선 도달하나")에 knee 의미 有. scaffold 진입 = **별 분기**(곡선 위 한 점 아님·"decidable해서 떼냄").

### §4c. ★y축 = 규칙-격리 failure-census Δ (global pass 아님·리뷰 #2)
control §5 denoise 교훈: pass^1 천장은 B(operand)가 결정·autofetch는 A_notfound만 닫음. **단일규칙 곡선 y에 global pass 쓰면 규칙 confound→곡선 노이즈.** ⇒ **y = 그 규칙의 *격리된 실패코드* Δ**(fetch-first=ΔA_notfound·gate=Δgate-violation 등·`t2_failcensus` 규칙별 코드). **global pass^1 = 보조 지표로 강등**(전체효과 참고만).

| 측정량 | 정의 | 도구 |
|---|---|---|
| **reliability(규칙격리)** | 그 규칙의 *격리 실패코드* Δ(global pass 아님) | `t2_failcensus` 규칙별 코드 |
| flexibility_loss | held-out 정상경로 false-block rate(§fetch-first doc §4 라벨셋·enforced만) | held-out 정상경로 라벨셋 |
| A2_growth | 그 레버가 요구하는 A2 필드 수(delta·**키스톤 정리 후 baseline**) | gate.json diff |
| lifecycle_cost | build+OpEx(토큰·왕복·레이턴시)+maintenance | telemetry |
| no_collapse | held-out 일반능력 불변(learn 날개) | tbnfc eval |
| transfer | airline held-out(ABox-swap만) 유지 | 동일 LoRA/엔진·A2-swap |
| (보조) global pass^1 | e2e DB-match·confound 주의 | `t2_failcensus` |

- **scale 기준선 = 수평선**(큰모델 격리코드 도달치)·내재화 날개가 작은모델로 그 선 도달하나(=cheap-replication·[[13]]).
- **레버 결합 가능**(예 scaffold-deny + learn): 날개 혼합 셀은 *분해 책임*으로 표기(fetch-first 3분해=learn SHAPE + A2 의존맵 + scaffold 가드).

---

## §5. a-priori 분류 + ★진짜 헤드라인 (리뷰 #1 꼬리·헤드라인 재배치)

§35 scale분해 + CAPABILITY_LEVER 예측(검증할 가설·§2 표):
- **offload 날개로 떼냄(decidable)**: C1·C2·C4·C5·C6·C7·C9(filter). scaffold-deny가 싸게 닫음 예상.
- **내재화 날개 필요(scale-emergent·learn)**: C3(fetch-first SHAPE)·C8(recovery)·C10(operand)·C12(NL). promptable=C11.
- **혼합/분해**: C3(SHAPE=learn·의존맵=A2·가드=scaffold)·content-op(rank=gloss 얕음·comparative 깊음).

### ★헤드라인 재배치 ("곡선 다양"은 약한 가설 — 폐기)
"곡선이 규칙마다 다르다"는 **거의 자명**(decidable provenance < NL operand = a priori 참·NO-GO 사실상 불가). **진짜 헤드라인 둘:**
- **(a) ★thesis-load-bearing**: *깊은 규칙*에서 **learn(내재화)이 scaffold-perform(offload)을 flex_loss=0·A2_growth=0으로 *동률* 닫나** = 두 날개 crossover. = 바로 fetch-first crossover(§첫 곡선). GO=동률→learn 우위(offload의 flex/A2 비용 없이)·NO-GO=learn 못 따라옴→offload 정당.
- **(b) 예측 오류**: a-priori 깊이 분류가 *틀리는* 규칙(얕다 했는데 깊음·decidable 예상인데 학습 필요 등). = 가이드라인의 *비자명* 정보가.
- 부산물 = "곡선 다양"은 결과로 따라옴(헤드라인 아님).

---

## §6. 도메인-일반 전이 (모든 레버 공통 게이트)

각 레버가 **도메인 특화 없이** 규칙을 전이시키나가 핵심:
- **learn**: primitive 벤치 학습만(tau2 금지·jsonl tau2-도구명 grep=0)·retail+airline 둘 다 held-out 전이.
- **scaffold**: 엔진 grep `if domain`/도구명=0·A2-swap만으로 airline 작동(키스톤 검증됨).
- **prompt/skill**: 도메인-일반 텍스트(특정 도구명/카탈로그 금지)·retail+airline 동일 텍스트.
- **transfer 미달=그 (규칙,레버) 셀 음성**(도메인-특화로만 되면 [[05]] 위반·배정 불가).

---

## §7. 시퀀싱 (한 번에 불가·헤드라인 (a)/(b) 먼저 시험)

전 매트릭스(12규칙×레버×전이×scale) = 큰 프로그램. **헤드라인 (a)thesis-crossover + (b)예측오류를 싸게 시험할 spread 선정:**

1. **C3 fetch-first** = `C4_LEARN_FETCHFIRST_CROSSOVER`(전 레버) — ★헤드라인 (a) 직접: learn이 scaffold-perform을 flex/A2=0으로 동률 닫나. 가장 정보가 높음.
2. **C11 flow-rule following(내재화 날개·promptable 예상)** = prompt로 닫히나(scale선 도달) 대조.
3. **C1 provenance-check(offload 날개·scaffold-deny 예상)** = decidable로 떼냄 대조.
4. → 3 규칙이 (crossover·내재화·offload) 서로 다른 날개-판정 = 프레임 판별력 + (b)예측오류 노출 → 나머지 규칙 확장.
5. **C10 operand**·**C8 recovery** = 최깊음·둘째기둥 핵심(별 설계서 상속).

각 규칙 = design→review→build→eval(GPU 리뷰 후). 진행률 가시([[30]]).

---

## §8. 결과물·GO/NO-GO (헤드라인 재배치)

- **결과물 = 규칙별 Pareto/두-날개 판정 아틀라스 + 최소-레버 배정 가이드라인**(measured). = `EXPERIMENT_DESIGN eval#1`.
- **GO (a)**: 깊은 규칙(C3 등)서 learn이 scaffold-perform을 flex_loss=0·A2_growth=0으로 동률 닫음 ⇒ 두-날개 thesis(내재화가 offload의 숨은비용 없이 동등) 성립.
- **GO (b)**: a-priori 깊이 분류의 *비자명 오류*(예측↔측정 불일치) 식별 = 가이드라인 정보가.
- **NO-GO**: 깊은 규칙서 learn이 scaffold-perform 못 따라옴 → offload 정당(정직)·또는 모든 규칙이 한 날개로 collapse → 프레임 약함(정직 보고).
- **경계지도**: genuinely scale-bound(어느 내재화 레버로도 작은모델 안 닫힘·C12 후보) = 정직 한계.
- ⚠️ "곡선 다양"은 자명 부산물(헤드라인 아님·리뷰 #1).

---

## §9. 빌드 순서

1. ✅ 이 프로그램 설계서 + 리뷰 픽스(#1 Pareto/두날개·#2 격리 y).
2. **#3 결정**(키스톤 A2 정리 선결 vs 병행) ← 사용자 확인.
3. fetch-first 설계서에 #2(격리 y)·#4(flex-loss 라벨셋) 박기(완료) → C3 fetch-first 빌드(spread 1).
4. C11·C1 빌드(spread 2·3) → 날개-판정 다양 + (b)예측오류 → 나머지 규칙.

---

**불변 정합**: [[05]](결정질문·tau2학습금지·minimize-A2)·[[07]](통제점×강제강도)·[[11]](전이=primitive학습)·[[12]](다양성)·[[13]](흡수우선 scale→learn→scaffold 규칙별 실측). 상위 = `EXPERIMENT_DESIGN §0★★`·`CAPABILITY_LEVER_ALLOCATION`.
