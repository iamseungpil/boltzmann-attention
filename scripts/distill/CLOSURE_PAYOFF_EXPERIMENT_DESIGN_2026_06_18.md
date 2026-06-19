# Closure-Payoff 실험 설계 (2026-06-18) — "닫힌 유한 규칙-기저"가 ICLR 기여인지 가르는 실험

> **자립 문서**(다른 세션 리뷰용·컨텍스트 불요). 상위 = `RELWORK_AND_DIRECTION_2026_06_18.md`(§1.5 핵심주장·차별)·`GENERATOR_ALGEBRA_DESIGN_2026_06_17.md`(closure 논증)·`ma/M_A_RESULTS.md`(§21·§23A 씨앗). 메모리 = `00-thesis`·`41-relwork-rivals-whitespace`·`06-NOW`.

## 0. 왜 이 실험인가 (한 문단)
정독 종료(2026-06-18) 판정: 우리 유일한 깨끗한 신규 = **"tool-use 계획이 *닫힌(closure-justified) 유한 생성원 기저*로 환원되고, 그 *규칙 추상화*를 작은 모델이 학습·전이"**. 단 **"전이" 자체는 신규 아님**(schema-guided DST·ToolLLM 선점)·**closure 수학도 재적용**(관계대수[Codd 1970/72]+집계·순서 확장[SQL-92급]·Böhm–Jacopini 1966). ⇒ ICLR-worthiness는 **"닫힘(완전성)이 *측정 가능한 전이 이득*의 원인"임을 통제 실증**하느냐에 달림(`RELWORK §6`). 명제·프레이밍만으론 taxonomy 논문 = 부족. **이 실험이 그 payoff를 만들거나 죽인다.**

**★반순환(anti-circularity) 전제 — 전면화**: 최강 반박 = "기저를 *정의*하고 held-out이 거기 맞음을 보임 = 자기충족." 방어 = 기저는 **BJ+관계대수서 *벤치 보기 전* a priori 도출**(벤치서 귀납 아님) + held-out은 **다른 패러다임·다른 출처 벤치**(§5). 정리는 *기저가 닫혀 있음(완전·유한)*을 보장하는 **발판**이지 *학습·전이 명제*가 아님 — 후자는 §2-5의 경험적 기여. 이 선험-도출을 논문 전면에 두지 않으면 closure는 순환으로 읽힌다.

**용어 충돌 경고**: 여기 "closure"=**생성원-기저 닫힘**(BJ+관계대수, *이론 기여*). [[41-relwork-rivals-whitespace]]의 "closure leg(gate/recovery/deferral)"=**결정론 offload 엔지니어링**(*기여 아님·재사용*). 이름만 같은 **두 개** — offload를 "closure"로 부르지 말 것(헤드라인 자기모순 방지).

## 1. 가설 (falsifiable) + 반증조건
**H (주가설)**: 작은 모델을 **닫힌 완전 생성원 기저**(flow P1-P9 + content ops)로 학습하면, **(a) open-set 모방**(ToolLLM류·기저 없음)·**(b) 비폐포/불완전 기저**(STAR류·생성원 누락) 대비 **cross-domain·cross-bench 전이·커버리지·sample-efficiency가 우월**하고, 그 이득은 **closure(완전성)에 귀속**된다.

**반증(이 중 하나면 H 기각)**:
- 불완전 기저가 완전 기저만큼 전이함 → closure 무관(완전성이 이득 아님).
- open-set 모방이 cross-bench서 closed와 동급 → 기저 구조화 무가치.
- 이득이 diversity/data/compute 교란으로 설명됨(closure 귀속 실패).

## 2. 두 비교 (상보적·★역할 분리)
**closure는 두 다리다 — 섞지 말 것**: ① **필요성(necessity)** = "각 생성원이 비잉여"(기저 *최소*) → 비교1 ablation. ② **충분성(sufficiency=닫힘)** = "기저 밖이 0"(기저 *완전*) → §3 orphan율=0(held-out 벤치). **충분성이 더 신규·더 위험** — orphan율을 비교1과 동격 헤드라인으로.

**비교 1 — 완전성 ablation (★closure *인과*의 가장 깨끗한 격리)**: 모든 것 고정(모델·**인터페이스·op 어휘**·offload·데이터량·diversity), **기저 완전성만** 변화 = `A_closed` vs `A_incomplete`(같은 resolve_selection 인터페이스·같은 weight-내재화·생성원만 제거해 닫힘 깸).
- 씨앗 = `M_A_RESULTS §23A`: 5-op(불완전·substitute 누락)→0.03 / 7-op(완전)→0.44. **단 이는 오프라인 op-eval 씨앗 = 신뢰불가([[03-anti-drift]]·PM2 §3) → 일반화는 실 e2e로만.** harness 검증 용도로만 재현.
- 설계: 완전 기저서 생성원 *체계적 제거*(flow P_k·content op) → 그 생성원 요구 held-out 도메인서 전이 붕괴하나. **예측: 제거된 생성원 쓰는 도메인서만 국소 붕괴**(closure가 그 생성원을 *필요*로 함=필요성 증명). ⚠️ ablation은 **필요성**만 증명한다 — "가장 깨끗한 *귀속*"은 전이이득→closure 인과를 뜻할 때만이고, 그건 *동일 인터페이스* A_closed vs A_incomplete라서 깨끗(어휘 교란 없음).

**비교 2 — regime 비교 (vs rival 패러다임·★포지셔닝이지 closure 인과 아님)**: 동일 base·동일 예산·**offload 세 arm 모두 동일 제공**(§4).
- **A_closed**(우리): 닫힌 기저 라우팅(flow native tool_calls + content `resolve_selection`) + 결정론 offload + ABox.
- **B_open**(ToolLLM류·`2307.16789` 충실 재현): 기저 추상화 *없이* 벤치별 raw native tool-call 모방·도구 schema in-context. resolve_selection·op 어휘 없음.
- **C_nonclosed**(STAR류·`2010.11853` 충실 재현): 절차를 *데이터로 제공*(per-task 순서도/schema)+따라가기·이탈은 학습 fallback. (= 규칙이 weight 아닌 data.)
- **예측**: cross-bench 전이·sample-eff에서 A_closed > {B_open, C_nonclosed}.
- ⚠️ **비교2에 closure 인과를 싣지 말 것**: A>B는 "닫힘" 또는 "구조화 op 어휘를 *가졌다는 것 자체*"(어휘 교란)일 수 있고, C는 *닫힘+data/weight* 두 축이 동시에 다름. **순수 closure 격리=비교1**(완전 vs 불완전, 동일 인터페이스). 비교2 = "rival 이긴다"(capability 포지셔닝).

## 3. 지표 (전부 결정론·LLM-judge 금지·[[10-roles-deterministic]])
- **★전이 보존율 = 벤치-*내* 비율** = held-out / in-domain (**같은 벤치 자기 지표**). **비율은 무차원 → 벤치 간 *비율 패턴* 비교는 집계 아님**([[20-proven-results]] 집계금지 준수). 보고 = per-cell 매트릭스 + **셀별 부호 일관성(sign test)**(예: closed가 τ² 0.85·SOP 0.80·TB 0.78 / open 0.40·0.35·0.30 → 전 셀 closed↑가 증거, **평균 금지**). 벤치 native: τ² pass^1·SOPBench official success·TaskBench graph-F1·Synth round-trip.
- **★orphan / 신-생성원 출현율 (=충분성=닫힘 직접시험·헤드라인 동격)** = held-out *벤치*서 필요 연산 중 기저 *밖*인 비율. **closure 예측 = ~0**(census가 τ²서 orphan=0 이미 확인·`PRIMITIVE_COVERAGE_MATRIX`). held-out *벤치*(다른 패러다임)로 확장 = closure의 가장 직접적 시험·반순환 방어(a priori 기저가 미견 벤치를 덮나).
- **sample-efficiency** = 성능 vs 학습데이터량 곡선. 예측: closed가 *적은 데이터로* 전이 도달(완전 기저=압축).
- **커버리지** = 기저 재조합만으로 풀리는 task 비율.
- **비용** = 토큰·USD(F1 장부·`EXPERIMENT_DESIGN §1.6`).

## 4. closure-귀속 통제 (이득이 교란 아님 보장)
- **동일 base 모델·동일 총 compute·arm 간 데이터량 matched.**
- **결정론 offload = 비교2 *세 arm 모두* 동일 제공(기본·미결 아님)** → closed>open이 "우린 offload 있고 쟨 없다"(=[[41]] 재사용 엔지니어링)로 격하되지 않게. **비교1에서만 offload 별도 ablate.** closure *인과* 귀속은 비교1(완전성 ablation·동일 인터페이스)이 담당.
- **diversity matched**(표현/구조·[[12-diversity-required]]) → 전이 차이가 surface-mapping 아티팩트(§17-18) 아니게.
- **per-domain/bench 분기 0**(grep `if domain` = 0·CI).
- **contamination 0**: held-out 벤치·도메인은 학습서 제외.
- **★scaffold vs weight 분리 (헤드라인 폭 결정·매트릭스보다 *먼저*)**: flow 전이가 결정론 scaffold가 나르는지(전수본 adapter held-out≈0·`SOP:583`) vs 학습 weight인지 *분리 측정*(adapter-only arm). **flow가 scaffold면 "*학습된* 규칙추상화 전이"는 content-op(resolve_selection)로 좁혀진다** → 헤드라인을 미리 줄여 정직하게: "학습=content-op 추상화(관계대수-확장 selection 명명), flow=closure-정당화 결정론 scaffold." 이 분리를 §5 매트릭스 전에 해야 무엇을 주장할지 정해짐(위험: content-op의 *학습된* 전이조차 실 e2e 미입증 — §6 S0).

## 4.7 ★추상화 = *다층 최소 구조* — 무엇을 전이하나 (관건 정의·2026-06-19)
**관건(연구질문)**: 도메인/벤치 특화가 아닌 **도구계획에 필요한 *비잉여·충분한 다층 추상화 구조*를 학습해 전이하나.** 한 규칙(verify-before-advance)도 전체추론도 아님 — *구조화된 stratum 객체*의 전이.

> **★용어 고정(충돌 방지)**: 본 문서의 **"층/stratum/tier" = *추상화* 구조의 단계**(축·층A/B·일반성tier·A1-A7). **트랜스포머 네트워크 깊이는 "decoder layer(네트워크 층)"로만** 부른다(예: LoRA를 mid decoder layer 8-19에만). 둘은 *다른 referent* — 라이브 mid arm(네트워크 layer subset 실험)과 이 §의 추상화 stratum을 혼동 금지. (단 mid arm = "네트워크 층이 추상화 stratum을 담나"의 구조적 탐침이라 *연결될 수 있음* → 그래서 더더욱 명칭 분리.)

**다층 구조(flat 아님·= 추상화 stratum)**:
- **축**: flow 생성원(P1-P9) ⊥ content 생성원(8-op).
- **flow 내부**: 층A(control×data·*구성적* 닫힘) / 층B(policy overlay·*상대* 닫힘).
- **횡단**: 인식/구조화(*학습* 추상화) / 평가·집행(*결정론* 게이트).
- **일반성 tier**: 보편 규율(provenance·grounding·모든 단계)→구조(의존·순서)→인식(어느 primitive).

**★"최소(minimal)" = 두 뜻 고정**: (1) **경험적 비잉여+충분**(ablation=각 stratum 비잉여 ∘ orphan=0=충분)이지 **minimality *정리* 아님**(Kozen-Tseng 자제·과주장 금지). ⚠️ ablation은 *테스트된 벤치서* 비잉여만 → **전역 최소 아님**. "**the** 최소 구조"라 쓰지 말 것(자제한 정리로 회귀) → **"비잉여·충분한 *한* 기저"**로 일관. (2) **"최소=저차원"이 소형-학습·전이의 *근거***(§0 Olver n−s·scale=암기지 추상화 아님). ⇒ 최소는 economy 아니라 *왜 소형으로 되나*의 메커니즘.
- **★sense-2를 *실측가능*하게 (Olver 유추 → 측정)**: 추상화의 **내재차원 프록시 = 그 facet을 설치하는 최소 LoRA rank + 학습 step 수.** 우리 라이브 실험이 직접 측정: `step5`(델타 극소·opt 5)가 이미 resolve 호출 = **매우 저차원** 직접 증거. → "각 facet의 install-rank/steps"를 보고하면 sense-2가 유추 아닌 **실측**(rank 스윕 arm: r4/r8/r16서 스킬 발화점). low-dim일수록 소형 학습·전이 용이의 *경험적* 근거.
  - **⚠️ rank=차원 *상한*이지 차원 아님(교란 차감 필수)**: install-rank = 내재차원 + (간섭·`solo_sts 망각`=차원 아닌 간섭) + (데이터량) + (base가 이미 아는 양). "step5가 resolve 호출=저차원"이 실은 "base 반쯤 앎/간섭 적음"일 수 있음. → **격리 단일-facet LoRA · 데이터 고정 · base-능력 차감** 조건 하의 rank만 차원 상한으로 인용. (이번 세션 solo_sts 0/80 간섭이 직접 사례.)

**정정(2026-06-19)**: 앞 §4 "flow=scaffold"는 *인식/구조(학습)*와 *평가/집행(결정론)* 혼동. `PRIMITIVE_COVERAGE_MATRIX:29` = **모델=coverage(학습)/게이트=soundness(집행)**. flow에 *학습가능 추상화* 실재 → "학습-전이=content-op뿐"(위험4) *부분* 완화. 아래 A1-A7 = *일반성 tier × flow층*의 인스턴스(flat 목록 아님).

**일반화 flow-TBox (도메인-일반 *규율*·학습 / 구체 술어·평가=ABox+게이트·벤치 횡단 구성)**:
| # | 추상 규칙(TBox·학습) | P-prim·벤치 | ABox/scaffold(도메인특정) | 전이 실태 |
|---|---|---|---|---|
| **A1** | provenance: 인자값∈{user,상류출력}·날조금지 | P1·전벤치 | 어느 tool이 어느 값 생산 | ✅ **전이 검증(단독)** |
| **A2** | gather-before-use: 없는 값→생산 getter 먼저 bind | P2a/b·CFB/τ² | 어느 getter·필드 | ❌ v7 전이실패(R4 의미층) |
| **A3** | identity-before-scoped: user-scoped는 신원 선행 | P8·SOP/τ² | 어느 auth·scope | ◐ in-dist |
| **A4** | dependency-order: produce→consume 순서·독립 병렬 | P3/P9·TaskBench | 구체 의존그래프 | ◐ in-dist(graph-F1) |
| **A5** | confirm-before-irreversible: write 전 confirm | P6·τ²/SOP | 어느 action 비가역 | ◐ task17 전이실패 |
| **A6** | precond-recognition: 정책-선행 action은 선행값 gather 후 호출 | P5·SOP/τ² | 정책 술어(gate_spec) | ◐ in-dist |
| **A7** | recovery-on-deny: deny/error→re-gather/ask·무한루프 금지 | P7·전벤치 | — (반응형) | ✗ static불가·RL |

- **★A6이 핵심 예시**: 모델=정책선행 *인식+gather*(추상 SHAPE·학습) / 게이트=그 선행 *충족 평가*(결정론). 앞서 "gate=LOCK 학습불가"는 *평가*에 한함·인식/구조는 coverage=학습. ⇒ "flow=scaffold" 오류 정정.
- **★전이 패턴 가설(sharp·falsifiable)**: 추상규칙이 *순수 구조적*이면 전이(A1=provenance·도메인무관)·*도메인-의미 인식*을 요하면 미전이(A2=어느 getter가 필요값 생산=R4 의미). ⇒ **"계획-규칙 추상화 전이"를 *순수구조* 부분으로 좁히되 *살림*** — flow에 학습-전이 추상화 실재(A1+)이므로 "content-op뿐" 아님. 정직한 주장 = **"순수구조 flow 규율은 전이·의미인식은 offload/미해결"**, *어느 A_k가 전이하는지 = 측정 대상*.

## 4.8 ★무엇이 학습되나 — COPY / COMPUTE / ABox 3분 (A2≠offload 교정·2026-06-19)
앞서 "인식 tier(A2)=offload하는 부분"은 **오류**(메타-리뷰): offload = **COMPUTE-selection**(resolve argmax/rank·235B도 0.02). A2(어느 getter가 값 생산)는 offload 아님. 정확한 3분 ([[CFBSYNTH_P2B_P4_DESIGN_2026_06_19]] §3 COPY/COMPUTE 경계):
- **COPY** (P2b·P4-filter): 관찰된 값 복사·얕음 → **학습**(또는 자명·결정론 가드 backstop).
- **COMPUTE** (P4-argmax/rank·content 8-op): 카탈로그 위 깊은 선택 → **offload**(엔진).
- **A2 (어느 getter)**: ★**graded by 스키마 모호도** — 타입-*명확*(생산자 유일)이면 **compute-from-schema=offload**(planner가 도출·ABox 사실도 아님)·v7 "A2 실패"=PROVIDE-누락(카탈로그 미노출)일 수 있음 / 타입-*모호*(여러 getter 동타입)면 **ABox 사실 또는 학습**.

**★비자명성 바 (γ 재정의)**: 층-지도가 ICLR 기여이려면 — (1) **확인적이어도 *예측적*이면 기여**(held-out 벤치서 어느 stratum 전이할지 예측→검증, post-hoc 서술 아님). 밴드 축 = 확인/발견이 아니라 **post-hoc vs 예측력**. (2) **진짜 discovery 후보 = A2의 *스키마-모호* 부분집합이 학습-전이**(명확측은 전부 planner-offload). (3) 전이하는 게 A1-provenance(자명·모든 FC 모델)뿐이면 taxonomy 회귀. + rival(비교2)이 "prior 못 하는 전이" 축 담당. **명확측이 대부분이면 = "학습 잔여 거의 없다"가 그 자체로 thesis 결과**(더 많은 게 offloadable).

## 5. 다벤치 전이 매트릭스 (cross-bench = schema-guided DST와 차별점·★stratum-분해 전이 지도)
| 학습 | held-out(도메인) | held-out(★벤치) |
|---|---|---|
| SOPBench(일부 도메인) + TaskBench + Synth | SOPBench 잔여 도메인 | **τ²(retail·airline)·SOP-Bench(Amazon)** |
- **cross-domain 셀** = schema-guided DST도 함(차별 약). **cross-bench 셀**(다른 벤치·포맷·task 패러다임) = DST는 한 포맷 내라 *안 함* → **여기가 우리 고유**. 매트릭스가 "닫힌 기저는 패러다임 횡단도 전이"를 보이면 ①이 capability 기여로 섬.
- ABox-swap: A_closed는 unchanged·catalog/gate_spec(ABox)만 교체. 재학습 0.
- **★*stratum-분해* 전이 지도 (단일 숫자·flat 목록 둘 다 금지·§4.7 다층)**: 각 셀의 전이 보존율을 **추상화 stratum으로 분해**: 축(flow/content) → flow stratum(A 구성/B 정책) → 일반성 tier(보편 규율/구조/인식) → 인스턴스(A1-A7·content-op). 보고 = **"어느 *stratum*이 전이하나"의 지도**, 단일 평균 아님.
- **★sharp 핵심 = *tier-단조성*(binary보다 강한 falsifiable)**: "A1 전이/A2 미전이"는 §4.7 표 재확인이라 신규성 약함. 더 강한 예측 = **전이율이 일반성 tier를 따라 *단조 감소*(보편 규율 > 구조 > 인식).** 여러 벤치서 단조성 성립 = *예측 법칙*(헤드라인 falsifiable). 반증 = 비단조(인식 tier가 구조 tier보다 더 전이 등).
  - **★★식별가능성 사전점검 (complexity 통제로 불충분·헤드라인 위협)**: tier-단조성이 (a) 일반성 때문인지 (b) **단순성**(일반=대개 단순)인지 (c) **DAG-depth attrition**(gated DAG라 하류 stratum이 상류 감쇠로 체계적 저측정·§5 ε)인지 안 갈림. tier가 DAG-depth와 *완전 공선*(보편규율=얕고 도처·인식=깊고 특정)이면 **off-diagonal 셀이 없어 통제할 잔차변동 자체가 없음 = 식별불가**. → 출구 2개: **①off-diagonal(tier⊥depth) 케이스 구성/발굴**(高tier-deep·低tier-shallow) → depth 고정서 tier 분리 / **②없으면 §86을 *재명명***: "전이는 *dataflow-depth*로 감쇠"(참·검정가능·덜 웅장) ≠ "일반성 법칙". **단조성은 complexity *및* DAG-depth 통제(=off-diagonal 존재) 후 잔존 시에만 법칙.**
  - **★γ↔③ 커플링 (자기정합)**: §4.8 γ-예측 법칙의 *내용*을 ③이 결정 — **식별가능(off-diagonal 존재)→"일반성-tier 법칙"(헤드라인 강) / 불가→"전이는 dataflow-depth로 감쇠"(여전히 예측적·검정가능·기여, 웅장도↓).** 둘 다 예측적 기여이나 **헤드라인 격은 ③이 가른다 → ③ 해소 전 헤드라인 격(일반성 법칙) 주장 금지.**
- **stratum-분해 패턴 가설**: 보편·순수구조 tier(A1 provenance) 전이 / 도메인-의미 인식 tier(A2 getter·R4) 미전이 / content축(§21) 전이 / 층B(정책 평가)=결정론(애초 학습 아님). ⇒ **정직한 결과 형태 = "비잉여 다층 구조가 *통째* 전이가 아니라 *stratum별 부분* 전이"** = [[00-thesis]] 학습/결정론 경계를 *측정으로 그린 것*(부분전이=실패 아니라 경계 특성화=기여).
- **⚠️ 셀 검정력 사전지정 (분해 입도 vs 통계력)**: 4직교 cut(축×flow-stratum×tier×인스턴스)→셀 폭증·벤치 유한→셀당 n 과소·빈 셀. **검정력 있는 셀만 사전지정**(빈 셀 노이즈 보고 금지)·*성긴* 해상도가 정직.
- **★★ε: 층-지도 = 평면 N셀 아니라 *gated DAG* (독립 측정 불가·측정가능성 게이팅)**: dataflow는 의존순서(P2b→P4→content-op)가 있어 **상류 stratum 실패 시 하류는 *측정불능(NA)*이지 "미전이(0)"가 아님.** 직접 실측: 이번 세션 resolve_selection 19회 *전부* grounding 실패 = 에이전트가 P2b(fetch)서 죽어 content-op에 *도달 못함* → "content-op 전이하나"를 **잴 수 없었음**(미전이 아님). ⇒ §88 검정력의 *진짜 형태* = 통계력 아니라 **측정가능성**:
  - **NA≠0**: 상류 미작동 셀은 하류를 mask·NA로 표기(0과 구분). 거짓-음성 금지.
  - **조건부율 + base율 동반**: 하류 셀 = "상류 성공 조건부"로만 측정 → `P(하류 전이 | 상류 성공)` + `P(상류 성공)` 둘 다 보고.
  - **★낙관 편향 플래그(부호)**: 상류 성공 궤적 = *쉬운 케이스* → 그 위 하류 전이율은 **체계적 *과대*(상한 추정).** "조건부율은 상한"이라 명시.
- **순서 의존**: 이 지도는 **S0 하류** — 깨끗+스킬있는 checkpoint(§6 S0) 없이는 측정 불가(현 mid/cons/cfb arm이 S0 생산 중). "지금 측정가능"으로 읽지 말 것. scaffold/weight 분리(§4)를 *stratum별로* 실시.
- **▷데이터-대기 표시**: ε(NA≠0·게이팅)·①(A2 모호도 분할)·③(식별가능성)의 *수치*는 라이브 **cfbsynth 실험**이 채운다(P2b 살면 resolve가 *측정가능*해짐=ε 확증·A2 잔여 노출·하류 depth-감쇠 첫 데이터). 구조는 robust(지금)·수치는 데이터-정합.

## 6. 빌드 단계 (증분·각 단계 실측·기존 자산 재사용)
- **★S0 (전제 관문·실 e2e — 미통과면 S1+ 전부 모래)**: *학습된* content-op(resolve_selection)가 **실 retail user-sim e2e서 발화 + base 대비 도움**인가. 판정 = (a) resolve_selection assistant 호출 ≥ 유의 횟수 (b) base(pass^1 0.205) 대비 Δ≥3-4 pass (분산 ±2-3·multi-trial). **오프라인 op-eval 금지([[03-anti-drift]]).** 진행: `qwen7b_solo_sts`(lr2e-4·r64·loss→0) = **0/80·NO-GO**. **★S0 NO-GO = 3-way 진단(δ: 명제만이 아니라 *아키텍처*도 의심)**:
  - **(a) forgetting ✅해결**: solo_sts 0/80 = 익명툴 망각+bleed+환각 → `solo_lite`(lr↓r↓) = 깨끗(step5 bleed0·canon0) 단 pass≈base. `mid`(decoder layer 8-19)·`cons`(no_alias)도 망각 억제. = 튜닝/구조로 해결.
  - **(b) primitive-누락 (진짜 병목)**: lite도 pass 안 오름 → 전수 진단: **retail 연속홉 fetch=P2b+P4, 권위본상 CFB만 공급·solo가 CFB 제외** → order_id 날조 405회·resolve 100% grounding 실패(ε: P2b서 죽어 도달못함). 처방=`solo_cfb`(CFB-참조 추상 synth·[[CFBSYNTH_P2B_P4_DESIGN_2026_06_19]]) **학습/검증 중**.
  - **(c) 명제**: (a)(b) 다 처리 후에도 안 되면 명제 재구성. **(b) 미해결이면 closure-payoff 이전에 데이터/아키텍처(intercept·ABox 의존맵)로.**
- **S1 (완전성 ablation·비교1·closure 인과)**: 전 기저서 생성원 1개씩 제거 arm → 그 생성원 요구 held-out 도메인 *국소* 붕괴 + 그 외 보존. **offload·diversity·인터페이스·데이터 고정.** 모두 **실 e2e**.
- **★GO/NO-GO 관문 (S1 직후·사전등록)**: 국소 붕괴(필요성) **또는** orphan율≈0(충분성)이 **안 나오면 closure 주장 사망 → rival 재현(S2) 들어가기 전 정지.** (rival 충실 재현 = 수 주 비용·교란 폭발점 → 이 관문 전 착수 금지.)
- **S2 (baseline 구현·비교2)**: B_open(ToolLLM류)·C_nonclosed(STAR류) 충실 재현(논문 방법 그대로·`RELWORK 버킷 가/마` 인용). 공정 통제(§4·offload 전 arm 동일).
- **S3 (전이 매트릭스)**: 세 arm × 다벤치 매트릭스 → 전이 보존-*비율*(부호검정)·orphan율·sample-eff.
- **S4 (귀속·autopsy)**: 이득이 closure에 귀속되나(diversity/data/offload 통제 후 잔존?)·scaffold vs weight 분리(§4, 단 매트릭스 전 선행).
- 자산: 기존 LoRA(SOP·TB·Synth)·`synth_to_nativefc`·resolve wiring·e2e harness(`real_e2e_base.sh`·`real_e2e_solo.sh`·`eval_solo_ckpts.sh`)·census(`tau2_primitive_census`).

## 7. ★자가심사 (리뷰 안건)
- **thesis-정합**: 학습=닫힌 규칙-기저(도메인일반)·offload=decidable·ABox=swap·e2e=학습 TBox. ✅
- **치팅 방어**: orphan율·완전성 ablation은 결정론 census·per-domain 분기 0. 이득 귀속은 통제(§4)로. real 도구 미대체(offload=인자계산·`INTEGRATED_TBOX §3b`).
- **선행 재사용**: open/non-closed baseline = ToolLLM/STAR 방법 *그대로*(재발명 아님·[[41]] directive).
- **정직 scope**: closure = transactional slice 한정·층B 상대닫힘(`GENERATOR_ALGEBRA`·`ALGEBRAIC_DERIVATION_CLOSURE`). "무제한 닫힘" 주장 금지. **층B(policy-gate) = ABox(gate_spec)·GateInterpreter 결정론 집행 = 학습 안 됨 → 닫힘-*학습*-전이 주장서 명시적 제외**(ABox-swap/offload에 속함). 확정 문장: "학습-기저 전이 = flow+content 생성원 한정, policy 집행은 결정론 ABox-구동."
- **★수학 인용 정확화 (DB 리뷰어 즉사 방지)**: `argmax/argmin/rank/comparative` = 집계·순서 연산 → **Codd 원본 관계대수에 없음**(SQL이 GROUP BY/ORDER BY로 추가) → "관계대수 + 집계·순서 확장"으로 인용. `substitute/create` = 쓰기·부수효과 → Codd(read 닫힘) 아님 → "(관계대수로 operand 해소) ∘ (flow 생성원이 효과 적용)"으로 분리. `GENERATOR_ALGEBRA_DESIGN`서 이 확장·분리가 명시됐는지 확인.
- **무엇이 H를 죽이나 명시(§1)** = 정직.

## 8. 리뷰 받을 미결 질문 (1·3·4·6 = 리뷰 직답으로 *해소*·2·5 잔존)
1. **[해소] C_nonclosed 조작화**: 통합 금지 — *서로 다른 축*. C_nonclosed(STAR·절차=data+follow) = "weight 내재화 > in-context 절차"(**학습/내재화** 축, closure와 직교). 비교1 불완전기저 = **closure** 축. **둘 다 유지·라벨 분리.**
2. **[잔존] 공정 예산**: open-set(ToolLLM)은 본래 대량 API 데이터로 큼 — 동일 데이터량 통제가 공정한가, 아니면 동일 compute? matched 축 확정. (잠정: 동일 compute + 동일 in-domain 데이터량 둘 다 보고.)
3. **[해소] offload를 baseline에도**: 비교2 세 arm 모두 동일 제공(기본)·비교1만 ablate(§4). 미결 아님.
4. **[해소] cross-bench 지표 정합**: 벤치-*내* 보존-비율(무차원)+셀별 부호검정·평균 금지(§3). 원점수 벤치-간 비교 안 함.
5. **[잔존] sample-eff 곡선 범위**: 어디까지 data-scaling 할지·소형 조건부(§22 width=scale 교훈) 반영.
6. **[해소·헤드라인 폭]** scaffold/weight 분리가 ①을 약화: **예 — 정직의 핵심.** flow=scaffold면 "*학습된* 규칙추상화 전이"는 content-op로 좁혀짐 → 헤드라인 미리 축소(§4). content-op의 *학습된* 전이가 실 e2e서 성립함을 S0가 먼저 입증해야 함. 좁혀도 ICLR 가능하나 범위 정직 필수.

## 9. 성공/실패 판정 (사전등록·★단계 관문 명시)
- **관문 S0 (전제)**: 학습된 content-op가 실 e2e서 발화+도움(Δ≥3-4 pass) — **음성이면 "학습된 전이" 미성립 → 명제 재구성**(closure-payoff 이전 문제).
- **관문 S1 (closure 사망판정·rival 착수 전)**: 완전성 ablation 국소붕괴(필요성) **또는** orphan율≈0(충분성) — **둘 다 음성이면 closure 무가치 → 정지·rival 재현 금지.**
- **성공(ICLR 간다)**: S0·S1 통과 + 완전성 ablation 국소 붕괴(closure 필요) + orphan율≈0(닫힘) + A_closed가 cross-bench 보존-비율서 B_open·C_nonclosed 전 셀 초과(부호검정) + 이득이 통제 후 closure 귀속.
- **부분(워크숍/전문 venue)**: 이득 있으나 modest·cross-domain만(cross-bench 약)·일부 confound 잔존·또는 학습-전이가 content-op로 좁혀짐(flow=scaffold).
- **실패(taxonomy 회귀)**: 불완전≈완전 or open≈closed → closure 무가치 → 명제 재구성 필요.

> 권위 = `RELWORK_AND_DIRECTION_2026_06_18.md`·`GENERATOR_ALGEBRA_DESIGN_2026_06_17.md`·`ma/M_A_RESULTS.md §21·§23A`·`PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md`(census·orphan=0). 불변 = [[03-anti-drift]]·[[10-roles-deterministic]]·[[12-diversity-required]].
