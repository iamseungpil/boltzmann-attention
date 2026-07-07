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
| 기능 | 측정 scale-민감도 (14B→32B) | 위임 판정 |
|---|---|---|
| **⋈ reference-matching** | 40% → **49% (+9pp)** = scale-sensitive | **✅ 위임 candidate**(큰 tier가 실제 더 함) |
| criterion/variant-formalize | 67% → 72% (**+5pp·거의 flat**) | ❌ 위임 무익 → thinking/learn(Phase A/B) |
| coverage (for-all 전수) | 17≈16 (**scale-invariant**) | ❌ 위임 무익 → coverage-controller(결정론) |
| compliance (정책 준수) | scale-invariant(위반율 CI 중첩) | ❌ → 게이트(small) |
| calc (집계·총액) | decidable | ❌ → calc scaffold(결정론·토큰0) |
| horizon/per-step reliability | scale 삼(DR#2·복리 $p^H$) | ◐ trajectory-level(§5) |
- **핵심**: 위임은 **scale이 *실제로* 사는 기능(⋈)에만** 가치. flat/invariant 기능(criterion·coverage·compliance)은
  큰 모델로 보내도 소용없음 → 다른 lever가 닫음. ⇒ **fleet 위임집합 = 좁은 {scale-sensitive 기능}**·나머지=scaffold/
  learn/thinking. 이 분류가 곧 **결정론 라우터**(학습·confidence 불요).

## 3. 아키텍처 — E1(context-격리 sub-call)과 통합
- **트리거 = 결정점 탐지(E1과 공유·도메인-일반)**: write-tool + 선택인자(⋈ order_id·변형 new_item) = scale-sensitive
  결정점. scaffold pipeline이 그 지점 감지 → **격리된 sub-call**을 **기능-분류에 따라 tier 라우팅**:
  - scale-sensitive(⋈) → **큰 tier**(72B/235B/frontier) sub-call
  - scale-flat-reasoning(criterion) → thinking(Phase A/B) 또는 learn
  - decidable(calc·coverage) → 결정론 scaffold(위임 아님)
- **입도 = sub-decision**(ToolOrchestra per-tool-call과 동일)·**라우터 = 결정론 기능-분류**(ToolOrchestra 학습RL과 차별).
- 대부분 궤적(present·calc·gate·orchestration)은 small tier·**희소한 sensitive sub-call만 큰 tier** → 비용 small 근처.
- 큰 tier에도 **게이트 부착**(준수는 scale-직교·frontier도 낙폭 −2.2~−5.0pp 실측).

## 4. 비용 모델
- $\text{cost} \approx (1-f)\cdot c_{small} + f\cdot c_{big}$, $f$=**scale-sensitive sub-call의 토큰/호출 비중**(task 비중 아님).
- $f$는 작다: ⋈-resolution은 multi-order task당 1회·전체 토큰의 소수 → $c_{big}=20\times c_{small}$여도 $f{=}0.05$면
  cost $\approx 0.95{+}1.0 = 1.95\times$ small = 순수 big(20×) 대비 ~10× 쌈. (배수=배포환경 의존·추정·[[09]].)
- ★task-level cascade(전 task escalate·e≈0.36)보다 훨씬 쌈: 위임이 *기능 sub-call*이지 *전 task*가 아니라서.
- **무료 $f$/이득 추정**: 기존 궤적서 (a)scale-sensitive 결정점 수·(b)그 sub-call 토큰비중·(c)⋈ 14B→32B→(외삽)72B 개선
  → cost-품질 스케치. 신규 런 0.

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
- ✅ ⋈ scale-sensitive(+9pp)·criterion flat(+5pp)·coverage invariant = operand_probe·forensic(측정됨).
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

## 9. 시퀀싱
1. **무료 $f$ 추정**(즉시·기존 데이터): scale-sensitive 결정점 수·sub-call 토큰비중·⋈ 개선 외삽 → cost-품질 스케치.
2. Phase A/B·open-methods DR 회수 → 기능별 최저비용 닫개 확정(위임 vs thinking vs learn vs scaffold).
3. big-tier 확보 시(72B/235B or frontier-top-tier) → ⋈ sub-call 위임 smoke(격리 결정점·per-case).
4. 확정 후 특허 B(cost-knee·레버배분)·덱에 "function-level 결정론 위임" 편입.
