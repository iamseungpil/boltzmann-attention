# E-REF — 참조-해결 부담(deictic binding)은 scale/learn으로만 풀리는가: 형식화·실험 사다리·1차 프로브 (2026-07-11)

> **상태 = [D] 설계서 v1 + §2b 1차 프로브 [M] 완료(32B·무료·짝지은 108케이스·per-case 포렌식 포함). 커밋 전(세션 지시 "커밋 금지").**
> **★한 줄 결과**: P0/P1/P2 전 셀 **1.00**(정규화 채점·bind/answer/EM) · EM_strict **0.00**(전 원인=`options.` 접두 108/108)
> ⇒ ① 오늘 fexec V0 "cons 0%"=채점-규약 아티팩트 지배 ② clean deictic binding은 32B가 공짜로 풂(hop-낙폭 0)
> ③ in-vivo 잔여의 정체=오염/부하 하 유지(C42 교훈 재연) → V2 경화 설계가 다음 단(§2.4·§2.5 1차 답).
> 질문: "같은 사이즈"류 **deictic constraint**(발화의 지시 표현 → 문맥 레코드의 필드값)의 해소가
> **결정론 scaffold로는 원리적으로 불가능**하고 **scale/learn 축만 여는** 잔여인가 — 를 엄밀 실험+이론으로.
> 이론 기계 재사용: `THEORY_AMBIGUITY_CLASSICAL_2026_07_10.md`(**타 세션 소유 [D] — 인용·확장점 명시만·수정 금지**).
> 원장 앵커: C51(397B+think만 변형-잔여 0.4%=scale이 삼)·C55(카디널리티 단조성)·C56(동-scale thinking 무효·체계핵)·
> C59(열거 +31pp)·C61 E-ISO(형식화-부하·ITEMS C .44)·C43+선행(induction head=copy·binding 아님·2209.11895).
> 메모리 정본: [[42]](prompt-only 레버 무효·prior-override=scale-emergent) · [[45]](load=scale-invariant 축과 capability 축의 분리) ·
> [[41]](선행 재사용·실험은 whitespace 집중). 규율: [[05]]/[[08]]/[[09]]/[[10]]·수치는 본 doc에 provenance와 영속.

---

## §0. 출발점 — 오늘 V0가 남긴 문제 (NEXT_LEVER_GEN 부록 Z)

FORMALIZE-EXEC V0(`fexec_iso_probe.py`·실 궤적 결정점): **full-EM 0.00 (op/field 0.68 · cons 0.00)** → 게이트 불통과·미편입.
그러나 이 수치는 세 성분이 교락돼 있다:
1. **채점-규약 아티팩트**: cons 채점이 field 문자열 exact-set(`options.size`≢`size`)·availability 제약을 오답 처리 — 부록 Z 자신이 "cons 채점 규약 재검 여지"를 명기.
2. **궤적 잡음**: 실 궤적 결정점은 발화가 산만하고 gold-기준 자체가 근사(GOLD 표는 task-수준 요약).
3. **진짜 결손**: 기준의 참조-바인딩(deixis→값) 실패.

E-REF는 이 셋을 분리한다: **(1)은 strict/normalized 이중 채점**으로 정량화, **(2)는 합성 clean 결정점**으로 제거,
**(3)은 hop 수 통제(P0/P1/P2)**로 격리한다. 오늘 V0의 "cons 0%"가 규약+잡음이면 P0(정규화 후)는 높아야 하고,
바인딩이 진짜 결손이면 P0→P1/P2 낙폭으로 나타난다. → **§2b 실측이 판정.**

## §1. 문제의 형식화 (이론 골자 · 논문 §2 후보)

### §1.1 분해 q = f(g(X))

에이전트의 기준-형식화형 결정점(THEORY §2 (a)갈래)의 질의를 두 합성으로 분해한다:

- **g = 참조-바인딩** (semantic): 발화의 deixis("같은 사이즈"·"내가 산 것"·"지금 것과 같게")를 문맥 $X$의
  레코드 필드값으로 해소해 **형식 제약** $(k, v)$와 랭킹 기준을 산출. 출력 = 도메인-일반 spec
  `{op, field, constraints:[(k,v),…]}`.
- **f = 결정론 실행** (decidable): spec을 후보 레코드 위에서 집행(argmax/filter — op-library).
  **f는 기성이다**: `gate_interpreter.compute_facts`의 `argmax_where`/`argmin_where`·
  `t2_formalize_exec.execute_formalized`. 오류 0(THEORY §3(a)의 ψ-조건부 그대로 — 상류 g가 틀리면 f는 그 오류를 *확신을 갖고* 집행).

오늘 V0(부록 Z)가 이 분해를 시사한다: **f 쪽(op/field·랭킹 문법)은 서고(0.68), g 쪽(constraints=바인딩)만 무너진다(0.00)** — 단 §0의 교락 3성분 분리는 §2b가 담당.
따라서 "compound-criterion을 결정론으로 닫는다"(FORMALIZE-EXEC)의 성패는 전적으로 g에 달려 있고,
질문은 "**g는 어떤 레버가 여는가**"로 환원된다.

### §1.2 정리-스케치 — g는 결정론 엔진의 함수가 아니다 (결정론 불가능성)

**결정론 엔진의 가정** (우리 scaffold 실물의 정직한 공리화·THEORY §1의 $\varphi$와 동일 대상):
- (i) **유한 op-library**: 엔진이 아는 연산은 유한 상수 집합(argmax/argmin/filter·비교자 6종).
- (ii) **타입-필드 exact-match 조회**: $\varphi(X)$가 꺼내는 특징은 정확 일치·열거·하드 제약 평가뿐
  (key-token 매칭·문자열 포함 판정 — `_grounded_candidates`·`_field_lookup`이 실물).
- (iii) **의미 등가 오라클 없음**: "same"≈"동일한"≈"안 바꾸고"를 묶어 주는 판정기는 가정에 없다.

이 가정하에 **g가 표면형의 함수가 아님**을 보이는 두 경로:

**(a) 정보-이론 경로 (Fano 하계·THEORY §4b 기계 재사용).** gold를 바인딩 $b=(k,v)$로 특수화하고
$\varphi$를 (i)–(iii)의 추출로 두면, 패러프레이즈 앙상블 $\{u_i\}$ — 같은 gold-바인딩 $b$·표면형 분산
(예: "same size" / "the size I already have" / "don't change the fit" — key 이름 토큰이 표면에 없는 변주 포함) —
에서 $H(b \mid \varphi(X)) > 0$이 **측정 가능**하다(표면-패턴 클래스별 gold 분포의 조건부 엔트로피).
그러면 Fano에 의해 $\varphi(X)$-가측인 **임의의** 결정론 함수 $d$의 오류율은
$$P_e \ \ge\ \frac{H(b\mid\varphi(X)) - 1}{\log_2 |V|}$$
꼴의 모델-무관 하계를 갖는다($V$=값 치역). 이는 THEORY §4의 DPI/Fano 논증을 **미결정 섹터가 아니라
"결정론-가측 섹터의 바깥"에 적용**한 것이다 — 정보는 $X$ 안에 있으므로(의미 섹터·$H(b|X)=0$) LLM 해독은
가능하지만, $\varphi$-가측 함수에게는 하계가 걸린다. **미측정 [D]**: $H(b|\varphi(X))$의 실측 = V2(패러프레이즈 arm·§2.4).

**(b) 구성적 경로 (결정론화 시도의 필연적 형태).** (a)의 하계를 피하려면 $\varphi$를 NL-패턴으로 증강해야
한다("same X as mine" → anchor-따라가기 규칙). 그런 증강의 실물은 **표면-패턴 사전**이고, 사전은
도메인·언어·분포-특화다 — 우리 원장에 반례-실증이 이미 있다:
- **C58 [M] (트릭 경계)**: 통계(LOTO) 디폴트 +12pp — "벤치-분포의 숨은 일 = 전이불가·트릭"으로 스스로 판정.
  패턴-사전 g는 정확히 이 동형이다.
- **나열-힌트 사례 ([M]·`APRIME_REGRESSION_FORENSIC §t81`·`E_PLAN_LIVE_WIRING_DESIGN 부록 Y`)**: 품목-나열
  센서(`_enum_items`)가 t81을 직격 회복하지만, 부록 Y가 스스로 **C58-동형으로 강등** — "측정된 recall은
  gpt-4.1 user-sim의 정돈된 영어 산문에 결합된 값·쉼표-부재/타 구분자/턴-분산/비영어서 무력·전이 주장에
  계상 금지". 사전의 한 표제어 = task 하나·한 분포 — 열거 뒤에는 다음 표제어가 기다린다(비폐포·[[41]] STAR-형 반대축).
- **C41 [M]**: 같은 규칙 문구가 짧은 합성선 0.87·tau2선 무효 — 패턴의 유효 범위가 분포에 붙어 있다.
⇒ 결정론화된 g = 도메인-특화 scaffold의 재발명 = **[[05]] 위반의 구성적 증명**. (이것이 "g를 엔진에 넣지
말라"의 규범적 내용이다 — 못 넣는 게 아니라, 넣는 순간 전이가 죽는다.)

**정직 표기**: "증명"이 아니라 3각 측량이다 — (a)의 **경험적 Fano 하계**(V2 측정 대상) + (b)의 **구성적
논증**(반례-실증 3건) + §1.3의 **선행 이론 인용**. 각각 단독으로는 뚫리지만(예: (a)는 앙상블 설계 의존·
(b)는 귀납) 셋이 같은 곳을 가리킨다.

### §1.3 왜 "깊은" 추론인가 — 수학적 방향과 인용 지도

g의 계산 구조는 **2-hop 관계 합성 = 변수-바인딩**이다:
$$g = \text{apply} \circ \text{extract} \circ \text{identify}: \quad \text{anchor 식별} \to \text{필드 추출}(k \mapsto v^\*) \to \text{제약 적용}(k=v^\*)$$
hop마다 중간 결과가 **변수로 유지**돼야 하고(anchor의 정체·추출된 값), 이것이 copy와 갈라지는 지점이다.

**인용 지도** (repo-앵커 [S-lit] vs 착수-시-정독 필요 [?] 구분 — 과대주장 금지):
| 주장 | 출처 | 지위 |
|---|---|---|
| induction head = **copy**·바인딩 아님 | `2209.11895` (C43+선행 §6서 정독·채택) | **[S-lit·repo]** — 날조=정박치환의 기전이 곧 "바인딩 없는 copy"의 행동판 |
| 얕은-병렬 한계 TC⁰·CoT=직렬화로만 초과 | `2207.00729`·`2305.15408` (THEORY §5) | **[S-lit·repo]** — 단 이는 **연산-깊이** 축(f쪽 계산)이지 바인딩 자체는 아님 |
| contextual entrainment(문맥 토큰 logit↑·관련성 독립) | `2505.09338` (C43) | **[S-lit·repo]** — g 실패 시 무엇이 대신 나오는가(인접-값 치환)의 기전 |
| transformer의 function-composition/multi-hop 깊이 한계 (양적) | Peng et al.류 합성-한계·multi-hop grokking(`2405.15071`류)·depth-통신복잡도(Sanford et al.류) | **[?·정독 필요]** — 논문 인용 전 딥리서치 1회 필수([[40]] 규율). 알려진 결과는 전부 **합성 QA/형식언어** 세팅 |
| prior-override=scale-emergent·프롬프트로 못 닫음 | [[42]] (`2303.03846` 외) | **[S-lit·repo]** — "g를 프롬프트로 열기"가 죽은 채널인 이유 |

**본 실험이 채우는 whitespace** ([[41]] directive 정합 — 선행은 재사용·실험은 빈칸에):
선행 합성-한계 문헌은 hop-깊이를 형식언어/QA에서 다룬다. **agentic tool-argument 형식화에서
binding-hop 수를 통제변수로 한 scale-사다리 실측은 없다** — P0(0-hop)/P1(1-hop)/P2(2-hop) × 모델-사다리가
그 칸이다. C51(agentic ⋈을 scale이 삼)·C59(열거가 변형-⋈을 엶)·C61(order-⋈은 열거 역효과)은 hop을
통제하지 않은 채 잔여의 *존재*만 쟀다 — E-REF는 잔여의 *구조*(어느 hop이 비싼가)를 잰다.

### §1.4 예측 (반증 가능하게)

| # | 예측 | 근거 | 반증되면 |
|---|---|---|---|
| H1 | P0(정규화 후)은 높다(≥.8) — f-문법·상수-제약 형식화는 32B가 이미 푼다 | C42(짧은 합성 완벽)·C59 열거 | 오늘 V0의 cons 0%가 규약이 아니라 형식화 전반 결손 |
| H2 | P0→P1 낙폭 > 0 — 1-hop 바인딩이 유의미한 몫 | C61 ITEMS C .44·C56 체계핵 | 바인딩은 clean 세팅선 공짜 → 잔여는 전부 궤적-오염/경로(C61 ①) |
| H3 | P1→P2 낙폭 > 0 — anchor-식별 hop이 추가 비용 | 2-hop 합성-한계 [?] | hop-깊이 무관 → "깊은 추론" 프레임 철회·단일-hop 의미 해독으로 재서술 |
| H4 | (사다리·후속) P1/P2는 scale-민감(7B≪32B<frontier) | C51 ③·C36(복사→발명 형태변화) | scale-flat이면 Q4-no → learn/ASK만 잔여 |

## §2. 실험 사다리 (무료-먼저)

### §2.1 셀 설계 — hop 수만 통제

동일 과제("매칭 variants 중 최고가 선택")·동일 후보·동일 f·동일 채점. **셀 간 짝지음**
(같은 (product, key, value) 시나리오 36개가 세 셀에 반복·표면만 상이):

| 셀 | 발화 | 문맥 제공 | g의 hop |
|---|---|---|---|
| **P0 상수-제약** | "whose {key} is **{value}**, the most expensive" — 값이 발화에 명시 | 상품 variant 덤프 | **0** (전사→spec 전사만) |
| **P1 1-hop** | "the same {key} **as mine**" | anchor 레코드 1개(현재 아이템·tool 출력) + 덤프 | **1** (anchor 주어짐→필드 추출) |
| **P2 2-hop** | "the same {key} as the {product} **in my orders**" | 주문-더미(anchor + 타-상품 4개·≥2개는 같은 key의 다른 값) + 덤프 | **2** (anchor 식별→필드 추출) |

- 데이터 = comp gz(`sim_results/comp_retail_t4.results.json.gz`)의 **실 product/variants 36종**(합성 아님·
  시나리오 315개 중 결정론 필터 통과분). gold = 결정론 계산(자가-채점): 유일 price-argmax·available·
  availability-유무 무관 동일답이 되도록 시나리오를 필터(available 관용과 정합).
- **H(gold|X)=0 by construction** — 미결정 섹터가 아니라 **의미 섹터의 해독 측정**이다(THEORY §1.1).
  P2의 anchor 식별 큐(상품명)는 유일-매칭으로 고정 — 애매성이 아니라 hop이 변수다.
- 한계 정직: ① 셀당 템플릿 3종 회전뿐 — 패러프레이즈 앙상블(=Fano 하계 실측)은 V2 ② 합성 전사이지
  정보-맞춘 궤적 replay(E-ISO)가 아님 — agentic 전이는 별도 ③ 프롬프트/파서 = 라이브
  `t2_formalize_exec` FORMALIZE_SYS/parse_formalize 동형(자족 사본).

### §2.2 채점 — 정규화 규약 (오늘 아티팩트 교정·strict 병기)

- **field 정규화**: lowercase·strip·`options.` 접두 제거·공백 정규화·alias({cost→price}).
- **available 관용**: {available, availability, in stock, …} 제약은 EM·실행 양쪽서 무시(시나리오가 availability-무관 동일답이라 안전).
- **값 유형 정규화**: 숫자 동치(9≡"9"≡9.0)·bool 관용("true"/"yes")·case-insensitive.
- 지표: `op`/`field`/`cons`/`full-EM`(형식화 EM) · **`bind`**(표적 (k,v) 제약 재현 = **g 그 자체**) ·
  **`answer`**(기성 f를 모델 spec에 적용한 최종답 == gold item_id) · `EM_strict`(구-규약) · 셀 간 짝 flip.
- 구현 = `scripts/distill/tau2/eref_probe.py` (stdlib-only·리모트 단독 실행·`--dry` 케이스 덤프).

### §2b ★1차 프로브 실측 [M] — 32B·짝지은 108케이스 (2026-07-11)

> arm: Qwen2.5-32B-Instruct-GPTQ-Int8·temp 0·리모트 8140. 케이스 36/셀·시나리오 짝지음.
> 원자료: 리모트 `/home/woori/scratch/eref/eref_v1_32b.{log,jsonl}` (영속화 = 커밋 승인 후 sim_results 이관 예정).

| cell | n | parse | op | field | cons | **full-EM** | **bind(g)** | **answer(f∘g)** | EM_strict(구-규약) |
|---|---|---|---|---|---|---|---|---|---|
| **P0** (0-hop) | 36 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | **1.00** | **1.00** | 0.00 |
| **P1** (1-hop) | 36 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | **1.00** | **1.00** | 0.00 |
| **P2** (2-hop) | 36 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | **1.00** | **1.00** | 0.00 |

- 36/36의 Wilson 95% CI 하한 ≈ **0.90**. SERVER_ERR 0·parse-fail 0·exec_status ok 108/108. 짝 교차표: bind/answer/em 전부 (1,1,1)×36 — **셀 간 flip 0**.
- **[[08]] 포렌식 (per-case 정독 9건 + 전수 census)**: ① spec은 자명-참 채점이 아니라 실질 정답 — P1/P2의 제약값
  ("XXL"·"hardshell"·"white"·"large"·"automatic")은 발화에 없고 **anchor 레코드에만** 있는 값을 모델이 해소해 기입
  (P2는 4개 distractor 주문 — 같은 key·다른 값 ≥2 — 사이에서 anchor를 식별). ② **EM_strict 0.00의 전 원인 census**:
  field 표기 `options.<key>` **108/108**(dotted-prefix 단독으로 strict 전멸) + availability 제약 추가 8/108.
  constraints 개수 {1개: 100, 2개: 8} — 과잉-제약 없음. ③ underscore/hyphen 표기 변주 0건(재채점 norm-v2 동일).

**판독 (H1–H4 판정)**:
- **H1 ✅ 확증(초과)**: P0 형식화는 정규화 후 완벽 — **오늘 fexec V0의 "cons 0.00"은 능력 결손이 아니라
  채점-규약 아티팩트(`options.` 접두 일관 표기)가 지배**한다는 것이 clean 세팅서 완전 정량화됐다.
  ⇒ **fexec V0 게이트 판정은 정규화-채점으로 재채점 필요**(무료·기존 케이스 파일 재사용 — 부록 Z "cons 채점 규약 재검"의 확증).
- **H2·H3 ❌ 반증(32B·clean 한정)**: 1-hop도 2-hop도 낙폭 0 — **clean 의미 섹터의 deictic binding(≤2-hop·유일 식별 큐)은
  32B에게 공짜다.** "hop-깊이가 32B의 벽"이라는 프레임은 이 세팅에선 성립하지 않는다.
- **⇒ C42 교훈의 재연 (설계의 다음 단이 확정됨)**: cfbsynth가 날조 결손을 재현 못 했듯(C42: "정박할 id가 없어서"),
  **clean 합성은 in-vivo 바인딩 결손(C61 ITEMS C .44·C56 체계핵 t8/t82)을 재현하지 못한다.** in-vivo 잔여의 정체는
  hop-능력이 아니라 **오염/부하 하의 바인딩 유지**(C43 정박치환·C61 오염 20%·C60 paraphrase-brittleness)다.
  V2(§2.4)는 D7-동형 경화가 필수: 근접-오답 anchor(같은 상품 타-주문·값이 1토큰 차이)·key-토큰 부재 패러프레이즈
  ("don't change the fit")·문맥 길이/distractor 밀도 축·정보-맞춘 실궤적 replay(E-ISO C 좌석 재사용).
- **H4(사다리)의 재배치**: clean 케이스로는 7B/14B 사다리도 천장에 붙을 개연성(C42 선례: 7B도 짧은 합성 완벽) —
  **사다리는 V2 경화 케이스 위에서만 정보량이 있다.** 유료 frontier 신규런 불요 재확인.

### §2.3 모델 사다리·learn-축 (후속·설계 명기)

- **로컬 무료 사다리**: 7B(`Qwen2.5-7B-Instruct`)·14B·32B — 서빙 계획: woori GPU 2×A6000, 현행 32B 서버
  유지 + 빈 슬롯에 7B/14B 순차 기동(vllm·기존 `tau2_vllm_env`·다른 실험과 GPU 충돌 금지 확인 후).
  같은 케이스 파일 재사용(짝지음 유지) → **H4 scale-기울기**.
- **frontier = 기존 데이터 재사용·유료 신규런 0**: C51/C57 앙상블(신형 top8·해독기 8종)의 per-case 산물은
  *궤적* 기준이라 P0/P1/P2 셀에 직접 사상되지 않는다 — 재사용 가능한 것은 (i) T4 앙상블-불일치 슬롯의
  hop-후행 분류(우리 케이스와 동형인 변형-선택 결정점의 frontier 정오) (ii) C51 ③ 챔피언 F2 0.4% =
  "frontier P2 천장 ≈ 높음"의 간접 상한. **frontier로 P0/P1/P2를 직접 재려면 유료** → 승인 항목으로 유보([[09]]).
- **learn-축 검정 설계 (E6′ 연계)**: P1/P2형 합성 데이터(본 프로브 생성기 재사용·패러프레이즈 증강
  [[12]] 다양성)로 SFT/DPO. **데이터 게이트(C38 교훈) 명기**: base 7B가 P1/P2를 이미 풀면 학습 무효 —
  **7B 실측이 게이트 통과 여부를 판정**(P1/P2 낮고 P0 높아야 "바인딩만의" 학습 표적이 성립).
  rejected 구성은 C39 승계(관찰된 오답-바인딩 = anchor-오식별·인접-값 치환을 on-policy로 수확).

### §2.4 V2 — 경화(hardening) + 패러프레이즈 앙상블 (Fano 하계의 경험판·무료)

**V1 실측(§2b)이 V2의 필요성을 확정했다**: clean 케이스는 32B서 천장 — 결손을 재현하려면 C42/D7 교훈 그대로
난이도 축을 넣어야 한다. V2 = 같은 짝지음 골격 위에 4개 경화 축(요인 설계·각 축 on/off):
1. **근접-오답 anchor** (D7-동형·C43): 같은 상품의 타-주문 아이템(값 1토큰 차)·비슷한 상품명("Running Shoes" vs
   "Hiking Boots")을 더미에 배치 — anchor 식별을 exact-match서 의미 판별로.
2. **key-토큰 부재 패러프레이즈**: "don't change the fit"(size)·"keep it the way I have it"(전-속성) —
   발화-표면에 key 이름이 없는 변주 ≥8종/시나리오. (= Fano 앙상블 겸용.)
3. **문맥 부하**: distractor 주문 4→12+·무관 tool 출력 삽입(길이 축) — C61 오염-몫의 격리 재현.
4. **정보-맞춘 실궤적 replay**: E-ISO C 좌석의 결정점에 본 채점기를 접속(합성이 아니라 in-vivo 결정점·[[08]] 정합).

Fano 대조(§1.2(a)의 완성): 앙상블 위에서 ① 결정론 baseline d(표면-패턴 사전: key-토큰 매칭+anchor-따라가기
규칙 — 우리가 직접 최선을 다해 구현)의 오류율 vs $H(b|\varphi(X))$ 추정치의 Fano 하계 ② 같은 앙상블의 32B —
**LLM이 하계 아래로 내려가면** $\varphi$-가측 아님의 행동 증거. C60(paraphrase-brittleness·"ordered→bought"
반전)의 격리 재현이기도 하다. **모델 사다리(§2.3 H4)는 V2 케이스 위에서 실행**(V1 clean은 천장이라 무정보).

### §2.5 판정 기준 (질문 "scale/learn만인가"에 대한 결정 절차)

| 신호 | 판독 |
|---|---|
| **P0↔P1 격차** | 바인딩 몫의 격리 — f·형식화-문법과 분리된 순수 g 비용 |
| **P1↔P2 격차** | hop-깊이 비용 — "깊은 추론" 프레임의 성립 여부(H3) |
| **P2의 scale-기울기** (7B→14B→32B→frontier) | 가파르면 **Q4-yes = scale이 삼**(C51 ③ 정합·fleet/위임 후보) / flat이면 learn 또는 ASK만 잔여 |
| **frontier P2 천장** | 천장<1.0이면 그 잔여가 진짜 경계(미결정 아님·의미 섹터 내 해독 한계) — P3 헤드라인 후보 |
| V2 결정론 baseline vs Fano 하계 | d가 하계에 붙으면 §1.2(a) 실증 완성·scaffold-불가 확정 |

**★1차 답 (2026-07-11·V1 [M] 기준)** — "참조-해결 부담은 scale/learn으로만 풀리는가":
1. **결정론 scaffold로는 못 푼다** — 는 방향은 유지(§1.2 3각: 구성적 반례 3건 [M] + Fano 경로 [D·V2 측정] +
   선행 [S-lit]). 단 그 반대편이 놀랍다:
2. **clean 의미 섹터의 deictic binding(≤2-hop·유일 큐)은 scale/learn을 *기다릴 필요조차 없다* — 32B가 이미
   1.00으로 푼다**(§2b·bind=answer=1.00·hop-낙폭 0). THEORY §1.1 의미 섹터 예측("정보가 X 안이면 LLM 해독이
   꺼낸다")의 직접 실증.
3. ⇒ 질문이 이동한다: 진짜 잔여는 "binding 능력"이 아니라 **"오염·부하·표면형 변주 하의 binding 유지"**
   (C61 ITEMS .44·C56 체계핵·C60 brittleness)이고, **그 축에서 scale(C51: agentic ⋈을 scale이 삼)·
   learn(paraphrase-invariance·E6′)·scaffold(격리 서브콜+내용-매칭 열거·C59/C61-C)가 경합**한다 — V2 요인
   설계가 이 셋을 같은 케이스 위에서 분리 측정한다. "scale/learn만"이라고 답하기엔 열거-scaffold가 이미
   +31pp(C59)를 실증했고, "scaffold로 충분"이라고 답하기엔 열거는 정보를 안 늘리고 분산만 줄이며(C60 ②·DPI
   정합) order-⋈엔 역효과였다(C61 ③). **현 증거의 최선 요약 = "clean-바인딩은 공짜·오염-바인딩은 합성
   (scaffold 격리 + scale/learn 잔여)"** — 배분은 V2 실측이 확정.

## §3. 논문 매핑·분업·특허 연관

### §3.1 P3 §로 들어갈지 독립 논문인지

**기본 권고 = P3 *The Semantic Boundary*의 §2(이론)+§4(통제 실험)로 편입.**
- P3의 현재 골격(C3a·C3b·C46)은 "날조를 닫으면 ⋈만 남는다"까지고, C51이 C3b를 격리프로브 산물로 부분
  강등한 상태 — **P3에 지금 없는 것이 정확히 "잔여의 구조"다.** E-REF의 hop-사다리 + q=f(g(X)) 분해가
  그 §를 채운다(P0/P1/P2 = 경계의 *해부도*·"boundary"를 카디널리티(C55)와 hop-깊이의 2좌표로).
- **독립 논문 승격 조건**(둘 다 충족 시): ① V2 Fano-대조가 깨끗(결정론 baseline이 하계에 붙고 LLM이 아래로)
  ② scale-사다리서 crisp crossover(예: P1은 14B가 닫고 P2는 frontier도 잔여). 그 전에는 부품이 얇다 —
  [[46]] 교훈(부품·전제는 전부 선점→인용·양보) 그대로: hop-통제 실측 1개로 논문을 세우지 말 것.

### §3.2 T-AMB(타 세션)와의 분업 제안

| 소유 | T-AMB (THEORY doc·타 세션) | E-REF (본 설계) |
|---|---|---|
| 대상 | **정보 축**: $H(gold\|X)$·카디널리티 $\|C\|$·미결정 섹터·ASK | **계산/바인딩 축**: 의미 섹터 *내부*의 해독 구조(hop)·g/f 분해 |
| 기계 | 세-수준 분해·DPI/Fano·삼분법 라우터 | 그 기계의 **특수화 재사용**: gold:=바인딩 b·φ:=(i)–(iii) 공리화 |
| 확장점 제안(수정 아님) | ① §1 φ의 공리화 (i)–(iii)를 명시 부록으로 ② §4b Fano를 "미결정 섹터" 밖 "φ-가측 섹터 경계"에도 적용하는 계 ③ §2 (a)갈래에 "형식화 성공의 전제=g"를 각주로 | — (본 doc §1이 그 확장의 초안·T-AMB 세션이 채택 여부 결정) |
| 겹침 방지 | T2(카디널리티 단조성)는 $\|C\|$ 축 | E-REF는 $\|C\|$ 고정·hop만 변주 — **직교 설계** |

### §3.3 특허 B(배분·경계) 연관

- 특허 B의 라우터 청구(THEORY §7)는 "|C|≥2 순수-참조형 → LLM/ASK"로 **경계를 그어** 청구한다.
  §1.2의 불가능성 3각(특히 (b): 결정론화=도메인-특화 사전=전이 파괴)은 그 경계-청구의 **명세서 근거**
  ("왜 결정론 수단으로는 해당 섹터를 청구 범위에 넣을 수 없는가 = 발명이 경계를 옳게 그었다는 효과 절") —
  [D]/[M] 등급 명기 의무(THEORY 규율 승계).
- FORMALIZE-EXEC(레버2·미편입)의 재상정 조건도 여기서 나온다: **f는 기성·병목=g**인데, §2b가 보이듯
  **clean g는 32B서 1.00** — V0 불통과(full-EM 0.00)는 채점-규약(`options.` 접두)+궤적 요인이 지배했을
  개연성이 높다. ⇒ **재상정 절차**: ① fexec V0 케이스를 본 doc §2.2 정규화 규약으로 재채점(무료·즉시)
  ② 통과 시 §2.6 V0 게이트 재판정 ③ 라이브 편입 판단은 여전히 P-B 좌석 shadowing(부록 Z·variants dict-key)
  해소와 Δspurious 계측이 선행 조건.

## §4. provenance

- 오늘 V0 수치(full-EM 0.00·op/field 0.68·cons 0) = `NEXT_LEVER_GEN_DESIGN_2026_07_11.md` 부록 Z ·
  `scripts/distill/tau2/fexec_iso_probe.py`.
- §2b 실측 = `scripts/distill/tau2/eref_probe.py` (본 세션 작성·커밋 전·workers=4 병렬) · 리모트
  `/home/woori/scratch/eref/eref_v1_32b.{log,jsonl}` · **로컬 사본(검증됨·108행) =
  `reports/facet_rft_2026/sim_results/eref_v1_32b.jsonl`(미커밋)** · 케이스 원천 =
  `reports/facet_rft_2026/sim_results/comp_retail_t4.results.json.gz`(실 product 36종·시나리오 315 중 36 짝지음).
  arm = Qwen2.5-32B-Instruct-GPTQ-Int8·temp 0·8140. per-case 포렌식 = 본 doc §2b(정독 9건·strict-원인 census 전수).
- 원장 인용 전부 `RESEARCH_MASTER.md §3` 등급 표기 그대로. 이론 인용 = `THEORY_AMBIGUITY_CLASSICAL_2026_07_10.md`([D]·타 세션 소유).
