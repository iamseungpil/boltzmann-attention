# scale-reach 행렬 설계서 — 기전 × 난이도 × 크기 × 도메인 (2026-07-12)

> 상위 = `EXPERIMENT_DESIGN_RIGOR_2026_07_12`(E1-E6 셀-상세·FE1/R1-R7) · `THINKING_HORIZON §8-§10` · 원장 C56/C66/C69/C71/C72.
> 규율: [[05]] A2만·엔진 도메인-일반 · [[08]] 기전·per-case · [[09]] 무료 우선 · [[40]] 등급.
> **상태**: 설계(리뷰 후 구현). 사용자 지적(2026-07-12): "synth만 아니라 synth/retail/banking을 난이도별 기전-분해해 *scale이 어느 문제를 어디까지 푸는지* 측정하라."

## 0. 목표 = scale의 도달 지도
**질문**: 모델 크기(0.5B→32B→frontier)가 **어느 기전을·어느 난이도까지 scale로 푸는가**, 어디서 포화(scale가 해결)하고 어디서 flat(잔여=레버 필요)인가.
**현 상태(정직)**: 지금까지 대부분 **synth(우리 사다리)** + retail/banking은 frontier 집계·proxy. **retail/banking을 *우리 사다리*로 기전×난이도 분해한 것 거의 없음.** 이 설계서가 그 공백을 격리-프로브(무료)로 채운다.

## 1. 행렬 (셀 = {우리 사다리 0.5-32B + frontier 앵커} 성능곡선 → 포화/flat + 닫는 레버)
| 기전(근본기능) | 격리 프로브 | 난이도 축 | 도메인 | 현황 | 예상 scale 거동 |
|---|---|---|---|---|---|
| **M1 상태추적/sd** | E-HORIZON(순수-산술·FE1-b) | 값범위·K·H | synth | ✅ 일부(C72·32B) | 쉬우면 32B 포화·어려우면 flat |
| **M2 참조/near-miss** | E-REF/E-THINK | decoy 밀도(L1-L4) | retail(상품) | ✅ C66/C69 + E-THINK2 진행 | thinking/prov 닫음(§ E-THINK2 ②) |
| **M3 ⋈ 의미모호** | ⋈-probe(NEW) | 유효후보 수 | retail(주문) | ❌ NEW(C56/C59 부분) | **전 scale flat**(C56) |
| **M4 reach/발견** | banking-reach probe(NEW) | 발견체인 깊이 | banking(도구) | ❌ NEW(frontier=§3.2f) | banking 지배 잔여 |
| **M5 coverage/완결** | coverage-probe(NEW) | 항목 수 | retail/telecom | ❌ NEW(C52 frontier) | 전-도메인 flat |
| **M6 compute** | calc-probe | 자릿수·연산 | synth/retail | ✅ C-B 부분 | 전 scale flat→offload |
| **M7 compliance** | 게이트 census | — | 전 도메인 | ✅ C1 [S] | scale-불변 |

**핵심 산출**: 각 (기전, 난이도)서 성능이 크기와 함께 **포화하는 임계 크기**(scale가 푸는 지점) vs **flat**(어느 크기도 못 품=레버 영토). = capability×scale×lever 지도(논문 코어).

**★M-코드 = 본 doc 로컬 프로브 인덱스 (새 분류 아님·[[48]] 준수·리뷰 신규발견1)**: 정본 명명 = `UNIFIED_TAXONOMY §축2` 서술형 근본기능 11개. M1-M7은 그중 6개를 *격리 측정*하는 프로브 이름일 뿐:
| M | → 정본 근본기능([[48]]) | 3계층 |
|---|---|---|
| M1 상태추적/sd | **6 지속**(장기 per-step 복리 붕괴) | 레버(scale/thinking/분해) |
| M2 near-miss | **5 참조·기준형**(판별기준 형식화 가능) | 오프로드가능(직렬화·C56) |
| M3 ⋈ | **5 참조·순수⋈**(의도 의미매칭) | **환원불가 경계=모트** |
| M4 reach | **3 조립**(gather·발견·계획) | 오프로드(controller) |
| M5 coverage | **2 완결** | 오프로드(게이트) |
| M6 compute | **4 계산** | 레버(calc/thinking) |
| M7 compliance | **1 준수** | 오프로드(게이트) |
⇒ 행렬 = 정본 근본기능을 scale×난이도로 재는 것·**M-코드로 논문/원장 표기 금지**(서술형만). M2/M3=참조 2모드는 taxonomy가 *이미* 구분(§37).

**★앵커 구성물 컬럼(리뷰 §8.2/8.4)**: frontier "앵커"는 셀마다 *구성물 상이* — 곡선 점 금지·별도 "기전-실재 배지": {M1=**없음**(2509.09677=문헌앵커)·M2/M3/M5=**pass1 역산**(H가정)·M4=**e2e 궤적 포렌식**(§3.2f·프로브 형식 아님)}. 비대칭을 컬럼으로 노출=정직 표기.

## 2. 프로브 스펙 (격리·무료·결정론gold·[[05]] 클린)
각 프로브 = **한 기전만 격리**(다른 기전 gold-제공)·user-sim 0·우리 사다리 전수 + frontier 앵커.

### M1 상태추적/sd — E-HORIZON (기존·FE1-b 반영)
- 순수-산술 running-sum(lookup 금지)·난이도=값범위/K/H. arms{base,verify,detect,inject}. 기전=sd(inject 자기일관·F9). 난이도 보정=FE1-a 사전등록(32B base∈[0.75,0.88]).

### M2 참조/near-miss — E-REF/E-THINK (기존·확장·정본=근본기능5 참조-기준형)
- retail 상품 바인딩·decoy 밀도 L1-L4(C69). arms{base, +thinking, +prov-constraint(E2)}. **E-THINK2 예비**: thinking이 near-miss 닫음(4b L4 0.28→0.97·**parse율 수정후**). ⇒ **M2=decidable 단정은 [진행·조건부]**(리뷰 신규발견2): parse-게이트(≥0.95) 통과 + **8b/14b 확정 후에만** [M]. F8 교훈(2모델·아티팩트 재발 방지).
- **★교차-문서 갱신 등재(8b/14b 확정 시)**: M2 confirmed = **F3 분기②**(thinking이 축c near-miss 닫음) → `INTEGRATION §1`·`THINKING_HORIZON` 방화벽 표의 **축c행·thinking행을 비용-우위 프레임으로 *동시* 갱신**("thinking=축a만·외부오염 무력" 폐기·thinking 닫되 inference-scaling 비용 vs scaffold pass-cost≈0). 행렬 doc만 갱신·서사 doc 방치 = 문서분열 금지.

### M3 ⋈ 의미모호 — ⋈-probe (NEW)
- **설계**: retail 주문서 기준("my recent order")이 **복수 유효후보** 매칭(전부 owned·near-miss 아님)·의도로만 결정. 후보 수 난이도. arms{base, +thinking, +enumerate(C59)}.
- **★chance-라인 사전등록(리뷰 §8.3)**: ⋈는 구성상 모호 → "flat"은 **우연수준(1/유효후보수) 대비**로만 의미. **flat-at-chance = 진짜 undecidable(ASK 영토) / above-chance = 부분 결정가능(재분류)**. chance 라인 없인 결론 과대·과소 모두 가능.
- **판정(사전등록)**: 전 scale·thinking이 **chance 수준서 flat** → M3=진짜 잔여([[40]] "scale/learn/ASK 영토"). above-chance flat → 부분 결정가능(레버 탐색). enumerate 부분(C59 +31pp)이나 잔여~.34.
- **gold-independence**: 기준-매칭은 결정론(gold=유일 의도)이나 *의도 자체*가 모호=경계. [[05]]: A2 0(측정).

### M4 reach/발견 — banking-reach probe (NEW·최고가치·정본=근본기능3 조립)
- **동기**: banking 지배기전(조립/reach·§3.2f 발견체인 universal-fail 28/28)을 **우리 사다리로 처음 분해**(현재 frontier만).
- **설계**: banking 도메인(도구+KB+정책)·목표 제시·**필요 도구가 unlock 선행조건 뒤**(발견체인·도구은닉→발견→unlock→call 3단 그대로). 지표=올바른 도구 발견+unlock+호출. 난이도=체인 깊이=unlock hop 수(1/3/6+).
- **★(a) 격리 정직(리뷰 §8.1a)**: §3.2f-237 핵심규정 = banking 실패 = **3중 부하의 곱**(horizon × 발견체인 × all-or-nothing DB). M4는 **발견체인 1개 인수만 격리** → **"M4 닫힘 ⇒ banking 닫힘" 금지**. 판정문구 = "controller가 *reach-인수*를 닫음"으로 한정.
- **★(b) controller 2-variant 필수(리뷰 §8.1b·가장 중요)**: §3.2f-143 caveat = **EXTRA_read ~100%=탐색량 비인과**(frontier도 많이 읽고 실패). ∴ "읽기 더 강제"(generic gather)는 비-원인 공격 위험. 분해: **C-gather**(도메인-일반 읽기/열거 강제·topology 무지) vs **C-topo**(A2 unlock-topology 정적 그래프 순회). **판정**: C-topo만 닫음 → "reach=A2-지식 영토" / 둘 다 닫음 → generic 충분 / **둘 다 못 닫음 → reach는 controller 밖(정직 보고)**. = [[05]] 경계질문의 실측 답.
- **★(c) [[05]] 판정(리뷰 §8.1c)**: A2 unlock-topology = **정당(배포-시점 인지 가능 도메인 사실·도구문서/정책=ABox)·단 정적 그래프만**. **per-task gold 체인 주입 = 금지(gold-leak)** — A2=정적 위상 / gold=이 task 정답경로.
- **판정(사전등록)**: 체인깊이↑서 base 급락(banking 곡선 *reach-인수* 재현) ∧ (C-topo or C-gather) 회복 → 위 2-variant 분기. 격리(user-sim 0·gold=§3.2f 발견체인 채점용).

### M5 coverage/완결 — coverage-probe (NEW·정본=근본기능2 완결)
- **설계**: "모든 Y에 X" task·항목 수 난이도. 지표=완결율. arms{base, +완결게이트(읽기강제·C52 G1)}.
- **★N-원천 명기(리뷰 §8.3·gold-independence)**: 완결게이트의 항목 수 N = **대화/열거 출력서 파생**(E-PLAN 규칙0 동형·E6-i와 동형)·**gold 항목목록 주입 금지**(안 그러면 gold-peek).
- **판정**: 항목수↑서 base 하락(전-도메인 flat·C52) ∧ 게이트 완결(N=관측-파생) → coverage=게이트 영토.

### M6 compute — calc-probe (기존·C-B)
- 자릿수/연산 난이도. arms{base, +calc-offload}. 예상 전 scale flat→offload 1.00(C-B 대후보선택 0.02 vs 결정론 1.00).

### M7 compliance — 게이트 census (기존·C1)
- scale-불변 [S]·per-write rate flat. 난이도축 없음(위반=이분).

## 3. 측정 프로토콜 (포화 탐지 → 레버 귀속)
1. **각 셀**: 우리 사다리 0.5/1.5/3/7/14/32B(+frontier 앵커=submission.json 무료) × 난이도 × arms. runs≥30·**R1 CI**.
2. **포화 탐지(사전등록)**: base(scale) 곡선서 *포화 임계 크기*(Δ<ε 되는 지점) 또는 *flat*(전 사다리 저조). 난이도별로 임계 이동.
3. **레버 귀속(R4·per-case)**: 잔여(flat 부분)를 닫는 레버 측정(arm 대비)·**기전 확인**(F9식 self-consistency/per-case·집계 직행 금지).
4. **frontier 앵커**: 상단(0/18 banking·retail 챔피언)은 submission.json 역산(단 M4=§3.2f 직접·M1 synth엔 frontier 없음·주의).

## 4. 엄격성 (F1-F9 + FE1 코드화)
- **R1 CI**·**R2 대조군(detect/arm)**·**R3 gold-independence**(verify=재계산·prov=출처제약·controller=recency·전부 gold 아님)·**R4 기전 not 집계**·**R5 사전등록 양분기**·**R6 [[05]] 감사(셀별)**·**R7 confound 배제(parse율·termination)**.
- **★도메인-교차 축 금지(리뷰 F-banking)**: banking p_step=pass1 역산(H가정)·E-HORIZON=직접측정 → **같은 곡선/축 금지**. 각 도메인 자기-축·병렬 제시(외적타당도 vs 내적타당도).
- **★기전 단일성(FE1-b)**: 각 프로브는 한 기전만·난이도 상향이 다른 기전 유입 안 함(synth=산술만·lookup 금지).

## 5. 무료 / 기보유 / 유료
- **무료(격리 프로브·우리 사다리)**: M1-M6 전부. user-sim 0·결정론gold.
- **기보유**: M2(C66/C69)·M6(C-B)·M7(C1)·M4-frontier(§3.2f)·M3-부분(C56/C59).
- **NEW 구현**: M4 banking-reach(최우선)·M3 ⋈-probe·M5 coverage-probe.
- **유료 없음**: e2e(E-XFER-bank류)는 결론-후-확인 대기열([[09]]).

## 6. 산출 = 도달 지도 → 논문 주장
| 주장 | 셀 근거 | 등급목표 |
|---|---|---|
| scale는 상태추적을 난이도-임계까지 산다 | M1(E-HORIZON) | [M] |
| near-miss=decidable(thinking/prov) | M2(E-THINK2·E2) | [M] |
| ⋈=전 scale flat=잔여 | M3 | [M] |
| reach=controller 영토(banking) | M4 | [미검]→[M] |
| coverage=게이트 영토 | M5 | [미검]→[M] |
| compute=offload | M6 | [M] |
| compliance=scale-불변 | M7 | [S] |
⇒ **"scale는 M1(임계까지)·M6(못)·M7(못)... 각 기전마다 도달이 다르다"** = 논문 capability×scale×lever 지도 완성.

## 7. 실행 순서 (전 무료·리뷰 후)
1. **진행 중**: E-THINK2 8b/14b(M2 near-miss·F3 판정) · E3 inject 7B/14B(M1 sd).
2. **최우선 NEW**: **M4 banking-reach probe**(우리 사다리로 banking 지배기전 첫 분해) · M1 E-HORIZON FE1 파일럿(난이도 보정) · E2 prov-constraint(M2 레버).
3. **다음**: M3 ⋈-probe · M5 coverage-probe.
4. 완료마다: R1-R7·per-case·원장 C-신규·[[40]] 등급·행렬 셀 채움.

## 8. 리뷰 포인트
1. M4 banking-reach probe = banking 액션모델(discoverable/unlock/shell) 격리가 충실한가·gold=§3.2f 발견체인 사용 타당?
2. 도메인-교차 "같은 축 금지"(R7·§4)로 3-도메인을 *병렬*(외적 vs 내적 타당도)로만 — 통합 곡선 유혹 차단 충분?
3. NEW 프로브 3종(M4/M3/M5)의 [[05]]·gold-independence 사전 검증 항목 충분?
4. frontier 앵커: M1(synth)엔 frontier 없음·M4(banking)만 §3.2f 직접 — 앵커 비대칭 어떻게 정직 표기?

## 9. 리뷰 판정 (2026-07-12·§3.2f 정본 대조·리뷰어 세션) — **구현 착수 승인·조건 6건**

**§8.1 (M4 충실성) — 조건부 타당·3건 반영 필수.**
- gold=§3.2f 발견체인을 *채점*에 쓰는 것 = 타당([M]·per-case 확립·측정용).
- **(a) 격리의 정직 명기**: banking 실패 = **3중 부하의 곱**(horizon × 발견체인 × all-or-nothing DB·§3.2f-핵심규정). M4는 발견체인 *하나만* 격리 → **"M4 닫힘 ⇒ banking 닫힘" 주장 금지**(곱의 1개 인수). 판정문구를 "reach-인수를 controller가 닫음"으로 한정.
- **(b) ★controller 2-variant 필수**: §3.2c/:178 caveat — **EXTRA_read ~100% = 탐색량 비인과**. 즉 "읽기 더 강제"(generic gather)는 정본과 상충 가능 — frontier도 이미 많이 읽고 실패한다. 분해: **C-gather**(도메인-일반 읽기/열거 강제·topology 무지) vs **C-topo**(A2 unlock-topology 정적 그래프 순회). C-topo만 닫으면 "reach=A2-지식 영토"·둘 다 닫으면 generic 충분·**둘 다 못 닫으면 reach는 controller 밖(정직 보고)**. 이 분해가 [[05]] 경계 질문의 실측 답도 됨.
- **(c) [[05]] 판정: A2 unlock-topology = 정당·단 정적만**: unlock 선행조건 = 배포-시점 인지 가능한 도메인 사실(도구 문서·정책 = ABox)·**per-task gold 체인 주입은 금지**(A2=정적 그래프 / gold=이 task의 정답 경로 — 혼동 시 gold-leak). 액션모델 재현 충실성: 도구 은닉(shell)→발견→unlock→call 3단을 그대로·체인 깊이=unlock hop 수.

**§8.2+§8.4 (교차 축·앵커 비대칭) — 도메인-교차 금지는 채택·단 불충분: cell 내 cross-construct가 남음.**
- frontier 앵커는 *같은 셀 안에서도* 구성물 상이: M2/M3/M5 앵커=pass1 역산·M4 앵커=e2e 궤적 포렌식(§3.2f — **프로브 형식 아님**)·M1=앵커 없음. ⇒ **앵커=곡선 점 금지·별도 마커 + "기전-실재 배지" 프레임**(frontier서 그 기전이 실재함의 확인이지 곡선 연장 아님).
- **행렬에 '앵커 구성물' 컬럼 추가**: {없음(M1·대신 2509.09677=문헌-앵커)·역산(pass1^(1/H))·e2e-포렌식(M4)}. 이게 §8.4의 정직 표기 그 자체 — 비대칭을 숨기지 않고 컬럼으로 노출.

**§8.3 (gold-independence 항목) — 2건 누락·추가 필수.**
- **M3: chance-수준 비교선 부재** — ⋈는 구성상 모호라 "전 scale flat"은 **우연수준(1/유효후보수) 대비**로만 의미. 사전등록 보강: flat-at-chance = 진짜 undecidable(ASK 영토) / above-chance = 부분 결정가능(재분류). chance 라인 없인 M3 결론이 과대·과소 모두 가능.
- **M5: N-원천 명기** — 완결게이트의 항목 수 N은 **대화/열거 출력에서 파생**(E-PLAN 규칙0 동일)·gold 항목목록 주입 금지. 명문 없으면 gold-peek 의심(E6-i와 동형).

**신규 발견 (설계서 밖·리뷰어 몫) 2건:**
- **★[중대·[[48]]] M1-M7 = 평행 taxonomy 위험.** [[48]] 정본: 새 실패-코드 금지(F1-F6·G1-G9 폐기 이력)·명명은 UNIFIED_TAXONOMY 서술형 근본기능. M-코드가 그 11개 근본기능과 어떻게 대응하는지 **명시 매핑 1행씩** 추가 + "M-코드=본 doc 로컬 인덱스·정본 명명=서술형" 명기. 없으면 정확히 [[48]]이 경고한 표류(경쟁 분류 체계).
- **[중간·교차-문서 정합 alert] M2 예비(4b L4 0.28→0.97)가 사실이면 F3-분기②가 축c에서 현실화** — INTEGRATION/THINKING_HORIZON의 "thinking=축a만·외부오염 무력" 서사와 정면 충돌 예정. 8b/14b 확정 시 **방화벽 표 축c행·thinking행을 F3-②(비용-우위 프레임)로 동시 갱신**(행렬 doc만 갱신하고 서사 doc 방치 = 문서 간 분열). §2 M2의 "⇒ M2=decidable" 단정도 8b/14b 前엔 **[진행] 조건부**로 표기(parse-게이트 F8 교훈).

**종합**: 행렬 프레임(포화 임계 vs flat = scale의 도달 지도) = 논문 코어에 정확·전부 무료·[[09]] 정합. **구현 착수 승인** — 순서는 §7 유지(M4 최우선·단 위 (a)(b)(c) 반영 후). M3/M5는 chance-라인·N-원천 반영 후.

## 10. 리뷰 재평가 + 반영 확정 (2026-07-12·본 세션·공정 평가)
리뷰 §9 전건 **정본 대조로 재평가 → 전부 타당·과잉/오류 0·반영 완료**:
- **§8.1a/b/c (M4)**: §3.2f-237("3중 부하의 곱")·§3.2f-143("EXTRA_read ~100% 비인과") 정본 확증 → (a)격리정직·(b)**C-gather/C-topo 2-variant**(generic gather는 비-원인 공격 위험·가장 중요)·(c)정적-topology만·gold-leak 금지 → §M4 반영 ✅
- **§8.2/8.4 (앵커 비대칭)**: 셀별 앵커 구성물 상이(M1 없음·M2/3/5 역산·M4 e2e-포렌식) → **앵커=배지·구성물 컬럼 노출** → §1 반영 ✅
- **§8.3 (gold-independence)**: M3 chance-라인·M5 N-관측파생 → §M3/M5 반영 ✅
- **★신규1 [[48]] 평행-taxonomy(최고가치)**: 11 근본기능(`UNIFIED_TAXONOMY §축2`) 정본 대조 → **내가 M1-M7 새 코드를 또 만든 게 맞음**(F/G/BC/N 폐기 이력 반복). **M-코드→근본기능 매핑표 + 로컬-인덱스 명기** → §1 반영 ✅ (M2/M3=참조 2모드는 taxonomy가 이미 §37서 구분·내 분할이 우연히 정합).
- **★신규2 교차-문서**: M2 "decidable" 단정 [진행]-조건부화 + 8b/14b 확정 시 방화벽표 동시갱신 등재 → §M2 반영 ✅
- **공정 재평가 결론**: 반려 0. **[[48]] 평행-taxonomy catch가 최고가치**(내가 문서화된 표류를 반복·리뷰어가 정본으로 색출). (a)(b)(c)·chance·N 반영 후 M4→M3→M5 순 **구현 착수**.
