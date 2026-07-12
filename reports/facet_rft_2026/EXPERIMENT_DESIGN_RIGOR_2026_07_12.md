# 실험 설계서 — sd/sc 확정·A2-provenance 실증·32B·논문 엄격성 보강 (2026-07-12)

> 상위 = `THINKING_HORIZON_LEVER_SURVIVAL_DESIGN §8-§10` · 원장 C69/C71/C72 · 리뷰 `_cdp_private_local/INTEGRATION_PLAN_REVIEW §7-§9`(F1-F9).
> 규율: [[05]] A2만 변경·엔진 도메인-일반 · [[08]] 집계→결론 前 per-case·기전 · [[09]] 무료 우선 · [[40]] 등급([S]/[M]/[미검]/[D]) 정직.
> **상태**: 본 문서 = **설계**(리뷰 후 구현). 일부 예비결과 = 본 세션 실행분(동기·보정 근거).

## 0. 확립 / 갭 / 이 설계서가 닫는 것
**확립([M])**: E-HORIZON verify>scale(C72) · 기전=상태-발산(sd·F9 자연 0.917 + **inject 인과 0.969@32B**) · sd/sc proxy(frontier retail+banking 100% sd-form) · A2-provenance 논리(sd 결정가능분=A2-blockable).

**엄격성 갭(reviewer 표적)**:
| # | 갭 | 닫는 실험 |
|---|---|---|
| G1 | **synth 난이도 미보정** — 32B가 running-sum 거의 풂(base 0.322→**0.748**)·verify 대비 약화 | E1 난이도-스케일링 |
| G2 | **A2가 C69 near-miss 실제로 닫나 [미검]** (논리만) | E2 provenance-constraint arm |
| G3 | **silent-sc 미배제** (proxy=sd-form만·decision-quality 미측정) | E3 inject(synth·완료) + E4 tool-use decision-quality |
| G4 | **32B 미포함** (사다리 14B 절단) | E1/E5 32B 확장 |
| G5 | **축 d(상충지시) [미검]** (F2 지적) | E6 축-d controlled probe |
| G6 | **variance/CI 부재** (n=12-16·1.5b>3b 잡음) | R1 전 실험 |
| G7 | **thinking×near-miss 미확정** (1차 아티팩트·재실행중) | E5 |
| G8 | **synth→in-vivo 갭** (E-HORIZON=synth) | E4 |

## 1. 실험 큐 (각: 가설·arms·지표·판정규칙[사전등록]·비용·[[05]])

### E1. banking-보정 난이도 × 32B (사용자 제안 2026-07-12·재설계)
- **★역할 분담(정직·핵심)**: **"scale 느림/상용미달" 주장 = banking(실·구조적 per-step)이 짊어진다**(C71: 전 18 frontier per-step 0.748~0.884·0/18 상용). synth는 그 주장의 근거 *아님*. **synth의 고유 가치 = (a) 기전 sd/sc 인과(inject) (b) verify 개입 실증** — 실 banking선 싸게 불가(유료 e2e).
- **동기**: 단순 running-sum(K=2)은 per-step=산술이라 32B 포화(base 14B 0.322→32B **0.748**·selfcons 0.96). banking per-step=구조적(도구발견+바인딩+검증)이라 frontier도 0.75~0.88. ⇒ synth를 **banking 난이도로 보정**하면 "scale 느린 그 지점서 verify가 산다"를 통제로 실증.
- **설계(★FE1-b 반영·기전 단일성 보존)**: 난이도 상향은 **순수-산술 부하만** — 값범위↑(큰수·음수 [-999,999])·K↑(스텝당 정수 개수). **lookup 재도입 금지**(원 `eref_horizon.py:31` dict-free 격리 유지 = E1·E3가 *동일 태스크* 위·기전과 verify 주장이 같은 지반). 그 보정 난이도서 전 Qwen2.5 사다리 0.5B~32B × arms{base,verify,detect,inject}.
- **★FE1-a 사전등록**: 보정 목표 **32B base per-step ∈ [0.75,0.88]을 verify 효과 보기 前 고정**. 파일럿(base만)으로 K·값범위 튜닝→목표 도달 확인→그 난이도 잠금→*그 다음* verify/inject 측정. 보정 멈춘 지점이 우호적 verify 결과 지점이면 안 됨.
- **★FE1-b 컨틴전시**: 순수-산술로 32B가 [0.75,0.88] 못 내려가면(포화) → (a) 도달가능 최고난이도 수용(보고) 또는 (b) 불가피 lookup 시 **inject/self-consistency를 그 lookup-난이도서도 재실행**(sd 기전이 유지되나 확인·FE1-c: verify 재계산은 문맥-내 테이블만·gold-independent).
- **지표**: 보정 확인 · scale 곡선 · verify−base gap(비포화) · **R1 CI 병기**.
- **판정(사전등록·양분기)**: (i) 보정 성공 ∧ 32B서도 verify≫base → "scale 느린 난이도서 verify 우위" [M]. (ii) 보정해도 verify 이득 소멸 → verify 이득=쉬운-난이도 아티팩트 → verify 주장 하향.
- **★정직 명기**: banking=실-앵커(외적타당도) / synth=기전+개입 통제(내적타당도) **2-leg 병렬**. per-step "매칭"=*난이도-수준*(둘 다 p~0.8)이지 기전 동일 아님(banking=reach/unlock·synth=상태추적). **banking p_step은 pass1 역산(H=8 가정)이라 E-HORIZON 직접측정과 동일 곡선·축 금지**(역산이 pass=p^H 가정→horizon 주장에 순환). synth→banking 일반화 문장 금지([[40]]).
- **비용**: 무료(로컬). **[[05]]**: 무관(측정).

### E2. near-miss provenance-constraint arm — **A2가 C69를 닫나 ([미검]→[M])**
- **가설**: sd의 *결정가능-provenance* 부분은 A2 resolver-제약으로 gold 없이 닫힌다.
- **설계**: C69 near-miss task(eref_probe --v2 B). **+prov-constraint arm**: 엔진이 A2 spec `{arg:key, source_constraint: status∈{delivered,owned}}` 읽어, 모델이 formalize한 뒤 **바운드 값이 *제약 만족 레코드*(현소유)서 왔는지 검증**·아니면 지시형 regen("현소유 레코드서 바인딩"). base(1.00→0.47@L4) vs +prov-constraint.
- **★E2a(단일 소유레코드)**: 제약이 값을 유일결정 → **예측 →~1.0**(A2가 sd 닫음).
- **★E2b(복수 소유레코드=⋈)**: 제약 통과하나 여럿 → **예측 안 닫힘**(의도 필요·A2 밖 잔여). = **경계 실증**(A2가 어디까지·어디부터 못).
- **지표**: bind율 base vs prov-constraint · E2a vs E2b 대조.
- **판정(사전등록)**: E2a prov→≥0.95 ∧ E2b prov≈base → "A2가 결정가능-sd 닫고 ⋈는 못 닫음" [M] 확정. E2a가 안 닫히면 → resolver 설계결함(디버그) or near-miss가 provenance-결정가능 아님(재분류).
- **★E2a/b 쌍 필수(§5.2 확정)**: E2a 단독="A2가 sd 닫음"은 F8형 과대(⋈ 잔여 은폐). **E2b(⋈ 안 닫힘)가 [[05]]-정직한 A2 경계이자 모트의 정직성** — 반드시 함께 보고.
- **★gold-independence + 구현 게이트(§5.2)**: 제약=status 필드(문맥 실재)이지 gold 아님. **★코드-감사 게이트: 제약검증이 *에이전트-기조회 레코드만* 읽고 env/DB 직접조회 0임을 코드로 확인**(C34 autofetch 선 안 밟음 = 유일 구현 위험). 엔진은 값을 *읽어* 검증(문구/regen만). **트릭 아님 명시.**
- **비용**: 무료(격리·user-sim 0). **[[05]]**: source_constraint = retail 정책사실(A2)·엔진 도메인-일반(제약 만족 레코드 조회).

### E3. inject arm — synth sd/sc 인과 (✅ 본 세션 완료·dose-response 보강)
- **완료**: 32B inject 후 자기일관 **0.969**≈clean 0.960 = **순수 sd 인과 확증**. 7B/14B 진행중(드라이버).
- **보강(사전등록)**: dose-response — inject 오류율 r∈{0,25,50%}(2509.09677 동형)·전 scale. self-consistency(r) 평탄 → sd(주입량 무관 산술 무손상)·하락 → sc. **완결성**: 전 scale·r 격자.
- **비용**: 무료.

### E4. tool-use decision-quality probe — **in-vivo sd/sc (silent-sc 배제·G8 교량)**
- **동기**: proxy는 sd-*form*만(붕괴 배제)·decision-quality 미측정. 이 세션 사용자 지적: "침묵 sc" 배제엔 통제 필요.
- **설계**: 격리 tool-use 바인딩 결정점(eref 계열)에 **prior-error 주입 대조** — 조건A(정답 prior 문맥) vs 조건B(그럴듯한 *틀린* prior 액션/값 주입 문맥)서 *현재 스텝* 바인딩 정확도. A≈B → sd(prior 오류가 현결정 무손상)·A≫B → sc.
- **지표**: 현-스텝 정확도 A vs B · 전 scale.
- **판정(사전등록)**: B≈A(±잡음) → in-vivo sd 확인([M]·proxy를 mechanism으로 승격)·"침묵 sc" 배제. B≪A → sc 실재(서사 수정·비용-우위 프레임).
- **★일반화 처리(§5.3 확정)**: E4가 in-vivo면 tool-use 주장은 **E4를 직접 인용** → "synth→tool-use 일반화" 문장은 *허용*이 아니라 **불필요**해진다. [[40]] 금지 유지·**각 주장이 자기-도메인 증거 인용**(synth=E3·in-vivo=E4). prior-error 주입값은 문맥서 구성(gold-유래 아님).
- **비용**: 무료(격리 프로브). **[[05]]**: 무관(측정).

### E5. E-THINK 완료 + F3 분기 (thinking×near-miss·32B·G7)
- **진행중**: Qwen3 {1.7/4/8/14B} think=on 재실행(파싱수정·max_tokens 12000). **선결 게이트: parse율 ≥0.95**(아티팩트 재발 방지·F8 교훈).
- **32B 보강**: QwQ-32B(near-miss·thinking·교차계열 caveat) or Qwen3-32B 다운로드. non-think 32B=C69.
- **판정(F3 사전등록·3분기)**: thinking이 near-miss ①못닫음→"thinking=축a만" 확정 / ②닫음→비용-우위 서사 전환(레버유효·배타성 하향) / ③부분→"불완전·고비용" 정직. **parse 게이트 통과 후에만 판정.**
- **비용**: 무료.

### E6. 축-d controlled probe — 상충지시/멀티턴 ([미검]→측정·G5)
- **동기**: 서사의 축 d(상충지시 누적)가 [미검/D]·F2가 [M]처럼 쓰지 말라 지적. 측정으로 등급 확보.
- **설계**: 멀티턴 대화에 **통제된 상충/수정 지시** 주입(턴 k서 "아니 X 말고 Y")·후속 결정이 최신 지시 따르나 vs 이전에 고착되나. canonical-state controller arm(정본상태 유지) 대조.
- **지표**: 상충-후 정확도 base vs controller · 오염 밀도별.
- **★E6-i gold-independence(구현 前 해소)**: controller "정본상태" = **최신-대화-지시가 이전을 supersede하는 결정론적 recency**(대화순서서 파생·gold 아님). 명시 안 하면 gold-peek 의심. canonical = 최신-지시(구조적 recency).
- **★E6-ii 스코프 caveat(C50 경계·구현 前 해소)**: E6가 재는 것 = **"상충이 *탐지된* 조건에서 controller가 회복시키나"**이지 "상충을 탐지할 수 있나"가 아님. synth는 템플릿이라 탐지 구조적이나 **in-vivo선 상충-탐지 자체가 semantic 경계**(C50·CENSUS §5 B-잔여와 동일 한계). 이 caveat 없이 [M] 주장 시 in-vivo 전이 과대.
- **판정(사전등록)**: (탐지된 조건 하) 상충 밀도↑서 base 하락 ∧ controller 회복 → 축 d 실재·controller 처방 [M·탐지-조건부]. 무효과 → 축 d 서사서 격하/철회.
- **비용**: 무료(격리·synth 대화). **[[05]]**: controller = 도메인-일반 상태유지(recency).

## 2. 엄격성 요구 (전 실험 공통·F1-F9 교훈)
- **R1 variance/CI**: runs↑(≥30)·bootstrap 95% CI 병기·점추정 단독 금지(F-잡음). 사다리 비단조(1.5b>3b)는 CI로 해소.
- **R2 대조군**: verify의 이득이 trivial-calc-offload 아님을 **detect arm**(플래그만→base 회귀)이 증명(교정≠탐지). 전 실험 대조군 명시.
- **R3 gold-independence(★핵심 rigor)**: verify=**입력서 결정론 재계산**(running-sum: 주어진 증분 합)·provenance=**출처-제약**(status 필드)·**둘 다 gold 아님·추론시점 가능**. 모든 arm에 "gold-peeking 아님" 논증 명기(reviewer 정면 반론 봉쇄).
- **R4 기전 not 집계(F8/F9)**: 집계 붕괴를 기전으로 귀속 前 **self-consistency/per-case**로 sd/sc 판별(F9식). 깨끗한 숫자(1.000류)는 도출법 검증 후 채택(acc_pre tautology 재발 방지).
- **R5 사전등록(F3)**: 모든 판정규칙 **양분기 사전등록**·결과 무엇이든 반영경로 존재(motivated-reasoning 방지).
- **R6 [[05]] 준수**: A2 변경만·엔진 도메인 리터럴 0·arm별 [[05]] 감사.
- **R7 termination/confound 배제([[08]])**: infra/crash/parse-fail 배제 후 집계·parse율 게이트(E5).

## 3. 주장 → 실험 매핑 (반영 시 등급)
| 논문/특허 주장 | 근거 실험 | 목표 등급 |
|---|---|---|
| verify가 per-step을 scale보다 싸게 (난이도 조건부) | E1(난이도×32B)·C72 | [M] |
| 기전=상태-발산(≠self-cond) | E3 inject(✅0.969)·F9 | [M] 인과 |
| A2가 sd 결정가능분 닫고 ⋈ 못 닫음 | E2a/E2b | [미검]→[M] |
| in-vivo sd(침묵 sc 배제) | E4 | [미검]→[M] |
| thinking 경계(near-miss) | E5(parse 게이트 후) | [진행]→[M] or 3분기 |
| 축 d(상충지시)+controller | E6 | [미검]→[M] or 철회 |
| banking=scale 상용미달(실-앵커) | C71(✅) | [M] |

## 4. 실행 순서·비용 (전부 무료·[[09]])
1. **진행중**: ET2(E5 thinking 재실행)+inject 7B/14B(E3) · 32B E-HORIZON+inject(E1 부분·E3·✅완료).
2. **다음(리뷰 후 구현)**: E1 난이도격자(K=4,8) · **E2 provenance-constraint(A2 실증·최우선)** · E4 tool-use decision-quality · E6 축-d.
3. **32B thinking(E5)**: QwQ-32B or Qwen3-32B 다운로드 결정.
4. 완료마다: R1-R7 준수·원장 C-신규·[[40]] 등급·리뷰 문서 §4 갱신.
- **유료 없음**: 전 실험 격리/로컬/결정론gold. in-vivo e2e(E-XFER-bank류·유료)는 결론-후-확인 대기열([[09]]).

## 5. 리뷰 포인트 (다른 세션/리뷰어 판단 요청)
1. ✅**확정(사용자 2026-07-12)**: "scale 느림" 주장=**banking(실)**이 짊어짐 / **synth=banking-난이도 보정**해 그 지점서 verify/inject 작동 통제실증(E1 재설계). synth→banking 일반화 금지·역할 분담 명기.
2. E2b(⋈ 경계) 없이 E2a만으로 "A2가 sd 닫음" 주장 시 과대 위험 — E2a/b 쌍 필수?
3. E4 in-vivo decision-quality가 [M]이면 synth→tool-use 일반화 문장 허용? (현재 [[40]]로 금지)

## 6. 리뷰 판정 (2026-07-12·코드/데이터 대조·리뷰어 세션)
> banking per-step 원파일(`sim_results/banking_perstep_frontier_2026_07_12.txt`) + `eref_horizon.py` 대조. 설계 전반 견고·사전등록/gold-independence 규율 모범. E1 재설계 방향 **승인**·단 재설계가 들인 신규 리스크 3건 + §5.2/5.3 판정 + E6 갭.

**§5.1 (E1 banking-보정) — 승인.** banking이 크기, synth가 기전+개입을 나누고 일반화를 금지한 게 정확. 근거 확증: banking per-step은 **리더보드 pass1 역산**(p_step=pass1^(1/H)·H=8 가정)이라 E-HORIZON 직접측정과 **동일 곡선 금지**가 맞고(역산이 pass=p_step^H를 이미 가정), banking 실패기전=reach/unlock 발견체인(파일 명시)≠running-sum이라 "같은 태스크 어렵게"가 아닌 것도 맞다. **단 재설계 신규 리스크 3건(사전등록 필수)**:
- **★FE1-b (중대·confound 재유입)**: 원 E-HORIZON은 lookup을 *의도적으로 제거*해 실패를 순수 상태추적으로 격리(`eref_horizon.py:32` "dict-free…self-conditioning으로 격리"). 난이도 보정에 **lookup 재도입 시 그 격리가 깨진다** — 보정-태스크의 실패 = lookup-오류 ∨ 상태-발산. ⇒ **기전 주장(E3 inject=순수 running-sum)과 verify 주장(E1=lookup-보정)이 서로 다른 태스크 위에 앉는다.** 처방: inject/self-consistency를 **보정 난이도에서도** 실행(sd 기전이 그 난이도서 유지되나) — 안 하면 "쉬운 데서 기전, 어려운 데서 verify" 분리를 리뷰어가 정면 타격. 대안: 난이도 상향을 lookup 아닌 **순수-산술 부하(큰수/음수/K↑)만**으로 국한해 기전 단일 유지(권장).
- **FE1-a (post-hoc 방지)**: 보정 목표 [0.75~0.88]을 **verify 효과 보기 前 사전등록**. "0.8서 verify가 산다"는 그 0.8이 결과-무관하게 고정됐을 때만 통제 — 보정을 멈춘 지점이 우호적 결과가 나온 지점이면 안 됨.
- **FE1-c (gold-independence at 보정)**: lookup 재도입 시 verify 재계산이 **문맥-내 제공 테이블에서만** 재계산해야 gold-independent 유지. 숨은 store 조회면 R3 위반.

**§5.2 (E2a/b 쌍 필수?) — 예·강제. 이미 반영됨·강화 권고.** E2a 단독으로 "A2가 sd 닫음" = F8형 과대주장(⋈ 잔여를 은폐). E2b(⋈ 안 닫힘)가 바로 [[05]]-정직한 경계(A2는 결정가능분만·의도 필요분=잔여)이자 모트의 정직성 그 자체. **쌍 필수 확정.** + gold-independence 구현 게이트: E2 제약검증이 **에이전트-기조회 레코드만** 읽고 env/DB 직접조회 0임을 코드로 확인(autofetch 선 안 밟음·C34). 설계 문구(§40)는 옳으나 구현 시 이 지점이 유일 위험.

**§5.3 (E4[M]→일반화 문장 허용?) — 아니오. E4는 일반화를 *허용*하는 게 아니라 *불필요화*한다.** E4가 in-vivo면 tool-use 주장은 E4를 *직접* 인용하면 됨 — "synth→tool-use 일반화" 문장은 여전히 금지([[40]] 유지)·대신 "in-vivo(E4)가 synth가 격리한 동일 sd 기전을 확인"으로 각 주장이 *자기-도메인 증거*를 인용. 일반화 문장은 어디에도 안 씀. (E4 실패 시 서사 수정은 사전등록대로.)

**E6 [[05]]/gold-independence 감사 — 갭 2건(구현 前 해소).**
- **E6-i (gold-independence 미명시)**: controller의 "정본상태"를 무엇으로 아나? verify는 입력서 재계산(gold-무관)인데, 상충지시에서 정본 = **"최신 지시가 이전을 supersede"의 결정론적 recency**(대화순서서 파생·gold 아님)여야 함. 이걸 명시 안 하면 controller가 gold-peek로 의심받음. **명시 필수: canonical = 최신-대화-지시(구조적 recency)·gold 아님.**
- **E6-ii (스코프 caveat·C50 경계)**: synth 상충은 템플릿이라 상충-*탐지*가 구조적(턴순서)이나, in-vivo선 상충 탐지 자체가 semantic(C50 대화-semantic·게이트 불가 경계). ⇒ **E6가 재는 것 = "상충이 탐지된 조건에서 canonical-state가 회복시키나"이지 "상충을 탐지할 수 있나"가 아님.** 이 caveat 없이 [M] 주장하면 in-vivo 전이 과대(CENSUS §5 B-잔여 semantic과 동일 한계). 판정규칙에 명기.

**E3/E4/기타 [[05]] — 클린 확인.** E3 inject=F9 무료-재분석의 정확한 실현(self-consistency로 sd/sc 판별·✅). E4=순수측정(A2 0·prior-error 주입이 gold-유래 아니면 클린·주입값 구성만 주의). R1-R7 = F1-F9 교훈의 정직한 코드화.

**종합**: E1 방향 승인+FE1-a/b/c 사전등록 / E2a/b 필수 확정+구현 게이트 / E4는 일반화 불필요화(문장 금지 유지) / E6 갭2 해소 후 구현. **구현 착수 승인** — 단 FE1-b(lookup 확산)는 "순수-산술 부하만"으로 국한하는 쪽을 강권(기전 단일성 보존 = 논문 방어력 최대).

## 7. 반영 확정 (2026-07-12·리뷰 재평가 후·본 세션)
리뷰 §6 전건 **코드/데이터로 재평가 → 전부 타당·반영 완료**:
- **FE1-b 코드 확증**: `eref_horizon.py:31`("dict-free·순수 누적 격리") = 내가 lookup을 의도 제거한 게 맞음 → 재도입은 격리 파괴. **채택: E1 난이도=순수-산술 부하만(값범위·K↑)·lookup 금지**(강권 채택·기전 단일성). 컨틴전시(포화 시 lookup+inject 재실행) 명기. → §E1 반영 ✅
- **FE1-a**: 보정 목표 [0.75,0.88] verify 前 사전등록·파일럿-잠금 → §E1 ✅
- **§5.2**: E2a/b 쌍 필수 + **autofetch 코드-감사 게이트**(env/DB 직접조회 0·C34) → §E2 ✅
- **§5.3**: E4 in-vivo면 일반화 문장 *불필요*(허용 아님)·자기-도메인 인용·[[40]] 유지 → §E4 ✅
- **E6-i/ii**: controller=구조적 recency(gold 아님)·**탐지-조건부 [M]**(탐지 자체=semantic 경계·C50) → §E6 ✅
- **공정 재평가 결론**: 리뷰 과잉/오류 0. FE1-b가 최고가치(내 재설계가 들인 confound를 코드-대조로 색출). 반려 없음·전건 채택. **구현 착수(순수-산술 E1·E2 최우선).**
