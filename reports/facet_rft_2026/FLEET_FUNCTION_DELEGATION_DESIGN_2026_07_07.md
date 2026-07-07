# fleet 설계 — function-level 정적 위임 by 측정된 scale-민감도 (2026-07-07)

> **위치**: 부하-감축 lever 지도의 "fleet(안 닫고 위임)" 축을 정식화. 사용자 통찰(2026-07-07)=**task-level cascade가
> 아니라 기능-level 정적 위임**로 재프레임 → silent-leak 라우터 함정 해소.
> **불변**: [[05]] 라우터/트리거 도메인-일반·[[13]] scale은 *측정된 scale-sensitive 기능에만*·[[03]] 분류는 측정先·
> build後·[[09]] e-추정·분류=무료·위임 target 유료/미로컬 주의·[[08]] per-case.
> **foil**: ToolOrchestra [2511.21689]=학습 RL(GRPO) 라우터·frontier-as-tool·**클라우드(온프레미스 실격)**. 우리=**결정론
> 기능-분류 라우터·온프레미스 tier**.

---

## 0. 한 줄
task를 결정론 scaffold가 기능으로 분해하고, **각 기능을 측정된 scale-민감도로 정적 분류**해, **scale이 실제로 사는
기능(예 ⋈)의 sub-call만 큰 tier로 위임**한다. 라우터=난이도-예측기가 아니라 **기능-분류표**(=우리가 지금 실측하는 지도).
난이도-신호·실패-탐지·silent-leak이 원리적으로 불필요.
> **★f-실측 판정(2026-07-07·§4b·2026-07-07 저녁 리뷰-교정)**: 아키텍처는 유효하나 **tau2-retail 잔여엔 저-ROI·보류**.
> 근거(교정본): (a)**비용은 미측정 가정에 좌우**(prefix-caching·tier비율 R·big-thinking) → realistic 1.15–2.3×(R 지배·
> 전체 가정공간 1.02–5.0×). 비용은 판정 하중 못 받음(§4b 옛 "0.167·비쌈"은 분모 오류·철회). (b)**benefit이 작고 미측정**
> (⋈ 실패 ~7-10 task·isolated +9pp는 full-run 전이·present-흡수 미검증·72B +8pp는 순수 외삽·big-tier 미로컬).
> (c)**하중을 받는 유일 assumption-free 근거 = 잔여-지배**: 잔여 지배축(criterion·coverage)은 scale로 안 닫히거나
> (coverage invariant) 더 싼 lever가 닫음(criterion=thinking·Phase A +8pp near-free) → fleet 무익. **⇒ 우선순위=
> thinking/coverage-controller/learn·fleet=보류**(잔여가 진짜 scale-sensitive·⋈-지배인 배포서만 고가치·big-tier 확보 시
> 재개). [[09]] 무료추정이 과투자 사전차단·[[08]] 분류는 CI 미분리(◐·§2).

## 1. 재프레임 — 왜 task-level cascade가 아니라 function-level인가
- **task-level cascade(폐기)**: "이 쿼리 어렵나?"를 판정해 위임 → 난이도-예측기 필요. **confidence=무작위**(τ²-retail AUROC
  0.52–0.58) → 실패-탐지 라우터. **★e-실측 함정**: 결정론 compliance 게이트는 위반을 0으로 만드나 실패의 대부분이
  **silent 능력실패**(32B 0.360·14B 0.406·operand 오답인데 게이트 통과)라 **게이트가 못 봄** → gate-라우터 cascade는
  그만큼 싼-오답 누수. detectable(loop/giveup)만 ~0.7%.
- **function-level 위임(채택)**: 실패를 *탐지*하지 않는다. scaffold가 이미 task를 기능(present·calc·gate·select·
  formalize·⋈-resolve)으로 분해하므로, **각 기능을 *정적으로* scale-sensitive/invariant로 분류**해, sensitive 기능의
  sub-call만 큰 tier로 보낸다. **silent-leak 소멸**(실패탐지 불요·애초에 sensitive 기능만 큰 모델로).

## 2. ★라우터 = 측정된 scale-민감도 분류표 (지금 실측 중)
operand_probe·CLEAN_NT4 forensic·Phase A/B·DR#2가 정확히 이 지도를 만든다:
| 기능 | 측정 scale-민감도 (14B→32B·isolated operand_probe) | 위임 판정 |
|---|---|---|
| **⋈ reference-matching** | 40%→**49%** (+9pp·n=219·95%CI≈[−0.3,+18]pp) = scale-sensitive**?** | **◐ 위임 candidate**(효과 있으나 CI 미확정·paired McNemar 필요) |
| criterion/variant-formalize | 67%→72% (+5pp·n=88·CI≈[−8.5,+18.5]pp·**0과 미분리**) | ❌ 위임 아님(flat이라서 아니라 thinking이 더 쌈·Phase A +8pp) |
| coverage (for-all 전수) | 17≈16 (**scale-invariant**) | ❌ 위임 무익 → coverage-controller(결정론) |
| compliance (정책 준수) | scale-invariant(위반율 CI 중첩) | ❌ → 게이트(small) |
| calc (집계·총액) | decidable | ❌ → calc scaffold(결정론·토큰0) |
| horizon/per-step reliability | scale 삼(DR#2·복리 $p^H$) | ◐ trajectory-level(§5) |
- **★[[08]] 단서(2026-07-07 교정)**: +9pp(⋈)와 +5pp(criterion)의 **CI가 대량 중첩** → "⋈=sensitive vs criterion=flat"은
  **측정으로 분리 안 됨**. n은 충분(219/88)이나 **효과가 작아** unpaired CI가 0·서로와 미분리. 이 분류가 문서 pivot이므로
  **✅→◐**·paired 검정(McNemar·discordant count)·big-tier 실측 전엔 "시사적"으로만 취급.
- **핵심(교정)**: 위임의 유일 후보축 ⋈조차 효과가 modest·CI 미확정. flat/invariant 기능(criterion·coverage·compliance)은
  큰 모델로 보내도 소용없거나(coverage) **더 싼 lever(thinking·scaffold)가 닫음**(criterion) → fleet 위임집합은 좁고
  저-가치. 나머지=scaffold/learn/thinking. 이 분류가 곧 **결정론 라우터**(학습·confidence 불요)이나 **분류 자체가 측정 중**.

## 3. 아키텍처 — E1(context-격리 sub-call)과 통합
- **트리거 = 결정점 탐지(E1과 공유·도메인-일반)**: write-tool + 선택인자(⋈ order_id·변형 new_item) = scale-sensitive
  결정점. scaffold pipeline이 그 지점 감지 → **격리된 sub-call**을 **기능-분류에 따라 tier 라우팅**:
  - scale-sensitive(⋈) → **큰 tier**(72B/235B/frontier) sub-call
  - scale-flat-reasoning(criterion) → thinking(Phase A/B) 또는 learn
  - decidable(calc·coverage) → 결정론 scaffold(위임 아님)
- **입도 = sub-decision**(ToolOrchestra per-tool-call과 동일)·**라우터 = 결정론 기능-분류**(ToolOrchestra 학습RL과 차별).
- 대부분 궤적(present·calc·gate·orchestration)은 small tier·**토큰-희소한 sensitive sub-call만 큰 tier** → 비용 low
  (★교정: ⋈은 **task-빈번**[90%]하나 궤적당 **토큰-희소**[753/≈11.5k≈6.5%]라 둘은 모순 아님·§4b 옛 "비쌈"이 오류 노드) →
  realistic 1.15~2.3× small(§4b 표·R 지배).
- 큰 tier에도 **게이트 부착**(준수는 scale-직교·frontier도 낙폭 −2.2~−5.0pp 실측).

## 4. 비용 모델
- $\text{cost} \approx (1-f)\cdot c_{small} + f\cdot c_{big}$, $f$=**scale-sensitive sub-call의 토큰/호출 비중**(task 비중 아님).
- $f$는 작다: ⋈-resolution은 multi-order task당 1.92회·전체 **서빙-비용 토큰**의 소수. ★**분모 주의(2026-07-07 교정)**:
  분모=trajectory 서빙비용=**prefix-cached prefill(≈peak) + decode**(≈11.5k 실측·decode mean≈1.1k라 ≈peak)이지, 단일 turn
  snapshot(§4b 옛 3755=오류)도 naive throughput(101k·재처리 가정·과대)도 아님. 올바른 분모서 $f≈753/11.5k≈0.065$ → **§4의
  "f 작다" 직관은 대략 옳고, §4b가 분모를 과소로 잡아 f·비용을 과대했던 것**(아래 교정). $c_{big}=20\times$·$f{=}0.065$면
  cost≈2.3× small = 순수 big(20×) 대비 ~9× 쌈. (배수=배포환경(caching·**R**·big-thinking) 의존·미측정·[[09]].)
- ★task-level cascade(전 task escalate·e≈0.36)보다 훨씬 쌈: 위임이 *기능 sub-call*이지 *전 task*가 아니라서.
- **무료 $f$/이득 추정**: 기존 궤적서 (a)scale-sensitive 결정점 수·(b)그 sub-call 토큰비중·(c)⋈ 14B→32B→(외삽)72B 개선
  → cost-품질 스케치. 신규 런 0.

## 4b. ★f-실측 결과·판정 (2026-07-07·기존 데이터·신규 런 0 / ★2026-07-07 저녁 리뷰-교정: 분모 반증·근거 재배치)
- **★분모 재측정 (asmregen32b nt=4·n=456·per-msg `usage.prompt_tokens` 실측)**: peak single-context **median 9,971 /
  mean 10,396**·summed throughput(prefill+decode) **median 88,815 / mean 100,981**·mean per-turn context **7,707**·
  agent turns/sim **median 12**. ⋈ 결정 **1.92/task·90% task**(task-빈번은 맞음). ⋈ sub-call ≈392 tok·delegated=392×1.92≈**753 tok**.
- **🔴 옛 §4b 분모 3755 = 오류(철회)**: peak의 1/2.8·throughput의 1/27·mean per-turn(7707)의 1/2 — 어떤 궤적-레벨 기준으로도
  anomalous(추정 早-⋈ turn snapshot). 이 과소분모가 $f≈0.167$·"fleet 1.2~4.2×·비쌈·near-small 아님"을 만들었음 = **철회**.
- **비용(교정) = fleet/small = 1 + R·753/denom·분모/caching/R에 좌우**:

  | 분모 (mean) | R=2.25 | R=20 |
  |---|---|---|
  | 옛 3755 (과소·철회) | 1.45× | 5.0× |
  | **realistic 11.5k (prefix-cached prefill≈peak + decode·기본 vLLM)** | **1.15×** | **2.31×** |
  | throughput 101k (재처리 가정·과대) | 1.02× | 1.15× |
  | peak 10.4k | 1.16× | 2.45× |

  → realistic **1.15~2.3×**(decode mean≈1.1k라 realistic≈peak·전체 가정공간 1.02~5.0×). **비용은 미측정 가정(caching·**특히
  tier비율 R**·big-thinking decode)이 지배 → 판정 하중 못 받음**(옛 "비쌈"도, "near-small(1.02–1.15×)"도 single-endpoint;
  realistic은 R=2.25[72B]서 1.15×·R=20[frontier]서 2.31× — big tier 선택이 지배). pure-big(20×)보단 항상 훨씬 쌈은 유지.
- **🟡 benefit = 작고·미측정(진짜 보류 사유)**: ⋈ scale-lift 49%(32B)→~57%(72B **순수 외삽**)·isolated +9pp는 **full-run 전이
  미검증**(present-scaffold가 MAKEORBREAK 7/13 이미 흡수 가능·순증분<+9pp)·⋈ 실패 ~7-10 task뿐 → **task-pass +1~3pp=점추정**
  (상·하한 아님). **big-tier 미로컬 → benefit 사실상 미측정.**
- **🟢 잔여-지배(assumption-free·하중 근거)**: 잔여 지배 = **criterion·coverage**. coverage=scale-invariant(위임 무익)·
  criterion=**+5pp이나 CI가 +9pp와 미분리**(flat 단정 불가)·그러나 **thinking이 더 쌈**(Phase A: criterion-simple prompted-CoT
  **+8pp·near-free**) → 위임 대신 thinking/scaffold이 닫음. 이 논증은 R·caching·전이 미측정에 **무관**하게 성립.
- **판정(불변·근거 교정)**: fleet=**조건부 유효**(잔여가 진짜 scale-sensitive·⋈-지배·big-tier 확보 배포서만)·**tau2-retail
  저-ROI·보류**. 보류 사유 = **(비용이 비싸서 아니라) benefit이 작고·미측정 + 잔여-지배축은 더 싼 lever가 닫음**.
  [[09]] 무료추정이 build 전 과투자 차단·[[08]] 분류 CI 미분리(◐)·[[13]] scale-최소 정합.

## 5. 두 층 (정직한 뉘앙스·범위)
1. **이산 기능(⋈·criterion 결정점)** = 깨끗이 분리·sub-call 위임 가능(§3). = 본 설계 핵심 범위.
2. **trajectory horizon 부하**(scale가 per-step reliability로 삼·DR#2) = 단일 기능으로 안 쪼개짐. → scaffold가 부하 축소
   (E1/E2)하거나, 필요시 whole-trajectory를 상위 tier로(task-level·별도). **본 설계는 (1) 이산-기능 위임에 한정**·(2)는
   범위 밖(scaffold/E2가 담당).

## 6. novelty / positioning (foil=ToolOrchestra)
| | ToolOrchestra | 본 fleet |
|---|---|---|
| 라우터 | 학습 RL(GRPO)·outcome+cost 보상 | **결정론 기능-분류**(측정된 scale-민감도·무학습) |
| 위임 신호 | 학습 정책(난이도 암묵) | **정적 기능-타입**(⋈=sensitive) |
| tier | frontier-as-tool(API) | **온프레미스 tier**(14B→32B→72B)·큰 tier도 게이트 |
| 배포 | 클라우드(egress 필수·온프레미스 실격) | **온프레미스·무-egress** |
| silent-leak | 학습라우터가 일부 커버 | **원리적 무**(실패탐지 아닌 기능-정적 라우팅) |
- ★whitespace: **"측정된 기능별 scale-민감도로 정적 위임하는 결정론 라우터"**=선행 부재(confidence/학습 라우터만). open-methods
  DR(`w5m5cyfss`)이 라우터-신호 증거로 확증 예정.

## 7. 측정 상태 (분류=라우터가 이미 실측 중)
- ◐ ⋈ +9pp(n=219)·criterion +5pp(n=88)·coverage invariant = operand_probe·forensic. **★[[08]]: +9pp/+5pp CI 중첩·
  criterion CI가 0 포함 → sensitive/flat 라벨 미분리**(측정 아니라 시사)·paired McNemar·big-tier 실측 전엔 ◐(✅ 아님).
- 🔄 Phase A/B(thinking): criterion/⋈이 thinking으로 닫히나(위임 대안) — 닫히면 위임 불요·안 닫히고 scale-sensitive면 위임.
- 🔄 DR(`w5m5cyfss`): cascade 라우터-신호 선행(confidence 死 vs verifier/정적).
- ⚠️ **위임 target(72B/235B) 미로컬**(handoff high-scale 포기) → ⋈의 72B 개선은 **외삽**(14B40→32B49→72B?)·또는 frontier를
  top tier로(단 유료·on-prem 아님). **∴ 본 설계=분류(측정)+위임 메커니즘(설계)·실제 big-tier 실측은 모델 확보 시.**

## 8. 규율 자가감사
- [[05]]: 결정점-탐지·기능-분류·게이트 = 도메인-일반(⋈/criterion=기능타입·retail 아님). `grep if domain=0`.
- [[13]]: scale(위임)을 **측정된 scale-sensitive 기능에만**(⋈)·flat/invariant엔 안 씀 = scale 최소·정확 사용.
- [[03]]: 분류=**측정先**(어느 기능 sensitive인지 assume 금지·operand_probe/Phase가 확정)·build後. task-level cascade(내
  초기 오류)는 e-실측이 반증→폐기. #8 forensic-정합(위임집합=측정 버킷).
- [[09]]: $f$/이득 추정=무료(기존 궤적)·big-tier 실측=모델확보·유료 주의·배수 추정 명시.
- **thesis 정합**: fleet=cost-optimal 레버맵의 한 축(scale-sensitive→위임)·compliance moat(게이트)는 전 tier 불변·직교.

## 9. 시퀀싱 (★f-실측 반영·재정렬)
1. ✅ **무료 $f$ 추정 완료·리뷰-교정**(§4b): 분모 재측정(peak 10.4k·throughput 101k·realistic 11.5k)로 옛 3755·$f≈0.167$·
   "1.2~4.2× 비쌈" **철회** → realistic **1.15~2.3×**(R 지배·비용은 미측정 가정 지배·하중 못 받음). **보류 사유=benefit 작고·미측정
   + 잔여-지배축(criterion/coverage)은 더 싼 lever가 닫음**(비용 아님). → fleet **보류**(판정 불변·근거 교정).
2. **★우선순위 재정렬(fleet→후순위)**: 우리 잔여 지배가 scale-flat(criterion)·scale-invariant(coverage)라 fleet 무익
   → **(a) thinking(Phase A/B·진행중) (b) coverage-controller(결정론) (c) learn(criterion-formalize)** 가 최저비용 닫개.
3. Phase A/B·open-methods DR 회수 → 기능별 닫개 확정(criterion이 thinking/learn으로 닫히나·coverage가 controller로).
4. **fleet 재개 조건**: 잔여가 진짜 scale-sensitive-지배로 판명되거나(⋈-지배 배포)·big-tier(72B/235B) 확보 시에만.
   그때 ⋈ sub-call 위임 smoke(격리 결정점·per-case). 지금은 build 안 함.
5. 특허 B/덱: fleet=cost-knee의 "**조건부**" 축으로 기재(무조건 이점 아님·저-ROI 실측 caveat 병기).
