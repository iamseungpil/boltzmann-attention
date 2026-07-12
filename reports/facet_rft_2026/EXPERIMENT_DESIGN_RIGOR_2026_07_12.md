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
- **설계**: per-step 난이도 상향(lookup 재도입·큰수/음수·K↑)으로 **32B base per-step을 banking frontier 범위 [0.75~0.88]로 보정**(현 0.96→목표 ~0.8). 그 보정 난이도서 전 Qwen2.5 사다리 0.5B~32B × arms{base,verify,detect,inject}.
- **지표**: 보정 확인(32B base∈[0.75,0.88]) · scale 곡선 · verify−base gap(비포화 확인).
- **판정(사전등록)**: (i) 보정 성공 ∧ 32B서도 verify≫base 유지 → "scale 느린 난이도서 verify 우위" [M](banking-충실 통제). (ii) 보정해도 verify 이득 소멸 → verify의 이득이 *쉬운 난이도 아티팩트*였음 → 서사 수정(verify 주장 하향). 어느 쪽이든 반영.
- **★정직 명기**: per-step "매칭"=*난이도-수준* 매칭(둘 다 p~0.8)이지 *기전 동일 아님*(banking=구조·synth=산술). synth는 "그 난이도에서 verify 작동"의 통제 증명·banking은 실-앵커. **synth→banking 일반화 문장 금지([[40]]).**
- **비용**: 무료(로컬). **[[05]]**: 무관(측정).

### E2. near-miss provenance-constraint arm — **A2가 C69를 닫나 ([미검]→[M])**
- **가설**: sd의 *결정가능-provenance* 부분은 A2 resolver-제약으로 gold 없이 닫힌다.
- **설계**: C69 near-miss task(eref_probe --v2 B). **+prov-constraint arm**: 엔진이 A2 spec `{arg:key, source_constraint: status∈{delivered,owned}}` 읽어, 모델이 formalize한 뒤 **바운드 값이 *제약 만족 레코드*(현소유)서 왔는지 검증**·아니면 지시형 regen("현소유 레코드서 바인딩"). base(1.00→0.47@L4) vs +prov-constraint.
- **★E2a(단일 소유레코드)**: 제약이 값을 유일결정 → **예측 →~1.0**(A2가 sd 닫음).
- **★E2b(복수 소유레코드=⋈)**: 제약 통과하나 여럿 → **예측 안 닫힘**(의도 필요·A2 밖 잔여). = **경계 실증**(A2가 어디까지·어디부터 못).
- **지표**: bind율 base vs prov-constraint · E2a vs E2b 대조.
- **판정(사전등록)**: E2a prov→≥0.95 ∧ E2b prov≈base → "A2가 결정가능-sd 닫고 ⋈는 못 닫음" [M] 확정. E2a가 안 닫히면 → resolver 설계결함(디버그) or near-miss가 provenance-결정가능 아님(재분류).
- **★gold-independence(F-이 세션)**: 제약=status 필드(문맥 실재)이지 gold 아님·엔진은 값을 *읽어* 검증(autofetch 아님=문구/regen만·[[05]] 클린). **트릭 아님 명시.**
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
- **판정(사전등록)**: 상충 밀도↑서 base 하락 ∧ controller 회복 → 축 d 실재·controller 처방 [M]. 무효과 → 축 d 서사서 격하/철회.
- **비용**: 무료(격리·synth 대화). **[[05]]**: controller = 도메인-일반 상태유지.

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
