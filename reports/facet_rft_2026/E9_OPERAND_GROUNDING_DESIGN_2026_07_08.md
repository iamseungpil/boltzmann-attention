# E9 (operand grounding) · E1′ (격리 formalize) 설계 — F2 격차의 두 조각 · 2026-07-08

> 상위 = `RESEARCH_MASTER.md`. **★본 doc은 C9의 *기전 라벨 정정*을 담는다.** 최초 판정("F2 = criterion 선택")은
> per-case 정독으로 **반증**됐다(포렌식 가드 발동·[[08]]). 실제 지배 기전 = **operand 날조(fabrication)**.
> **불변**: [[05]] 도메인-일반·join-resolver 금지 · [[10]] LLM=formalize·verifier=결정론 · [[08]] per-case ·
> [[09]] 무료先 · [[03]]#9 대칭크레딧 · [[03]] settled 음성 재유도 금지(§3서 정면 처리).

---

## 0. C9 재판정 (기전 라벨 정정)
`HORIZON_GAP_DECOMPOSITION`은 frontier 격차를 **H3-4의 item/variant 불일치(우리 13 vs o4-mini 2)** 로 국소화했고, 이를
**"F2 symbolic operand = criterion 선택"** 이라 라벨했다. **per-case 정독이 이를 반증한다.**

**H3-4 · order_id 정확 · item 불일치의 실제 구성:**
| | **ours** | **o4-mini** |
|---|---|---|
| **★날조된 id (DB에 부재)** | **7** | **0** |
| 실재 변형·틀린 선택 (진짜 criterion) | 5 | 2 |
| set-equal 아티팩트 | 1 | 0 |
- 실례: `new_item_ids=['6117189162',...]`(원 id **+1 증분**) · `['CHEAPEST_OPTION_6117189161',...]`(**placeholder 문자열**) ·
  `['1008292230','1008292231','1008292232']`(**연번**).
- ⇒ **우리 초과분 11 중 7 = 날조.** thinking으로 안 고쳐진다. **provenance 문제.** (메모리 R1B "값 없으면 ASK 대신 placeholder".)

## 1. 전 구간 날조 census [M] (`fabcensus`·전 sim·write-call의 id 인자가 DB에 실재하는가)
| model | 날조 sim | 날조율 | 날조∧실패 |
|---|---|---|---|
| **ours 32B+scaffold** | 27 | **5.9%** | 15 |
| 14B+scaffold | 27 | 5.9% | 18 |
| 32B floor | 23 | 5.0% | 18 |
| gpt-4.1-mini | 6 | 1.3% | 3 |
| o4-mini | 1 | 0.2% | 0 |
| **gpt-4.1** | 0 | **0.0%** | 0 |
| **claude-3.7** | 0 | **0.0%** | 0 |

- **★operand 날조는 frontier를 가르는 현상**: 중·상위 frontier = **문자 그대로 0**. 우리 = 5~6%.
- **우리 scaffold가 못 줄임**(floor 5.0% → +scaffold 5.9%) · **scale도 못 줄임**(14B ≈ 32B). ⇒ 기존 레버로 미커버.
- **상한 [EST·인용금지]**: 날조∧실패 15/456 = **최대 +3.3pp**. o4-mini와의 compliant 격차(5.3pp)의 절반 이상.
- **"id ∈ DB"는 완전 decidable** — 의미 판단 0 ⇒ 우리 decidability 기준상 **부작용 없이 살 수 있는 기능**.

## 2. E9 — operand grounding gate (신설·최우선)
### 2.1 술어 (결정론·false-positive 구조적 0)
write-tool 호출의 **id-타입 인자**(A2 `id_args`)의 각 값이 **도메인 엔티티 레지스트리에 실재**하는가.
- 실재 = A2 `registry` 경로(retail: products.*.variants 키 / orders 키)에서의 **멤버십**.
- **정당한 id는 정의상 레지스트리에 있다 ⇒ over-block 채널이 구조적으로 닫힌다.**(옛 prov의 harm 기전과 결정적 차이·§3)
- deny 시 **regen 피드백**: "id X는 존재하지 않는다. 실제로 조회한 레코드에서 값을 취하라." **답 미주입**(어느 id인지 안 알려줌).

### 2.2 [[05]] 자가감사
| 질문 | 답 |
|---|---|
| (1) 도메인-특화 순증? | 소 — A2에 `id_args`·`registry` **경로만** 선언(값 아님). 도메인 지식 0. |
| (2) 유동 판단을 결정론에 동결? | **No** — "어느 id인가"는 여전히 모델. 게이트는 **존재하지 않는 값**만 거부. |
| (3) scaffold가 도메인 행동을 수행? | **No** — 멤버십 검사만. |
⇒ 세 질문 모두 no/소 ⇒ **[[05]] 관문 통과**(E1 완결게이트와 대조: 그건 Q2=YES였다).

### 2.3 ★[[03]] settled 음성 정면 처리
- **기록된 판정**: `ASSEMBLED_FAILURE_FORENSIC §retry/provenance` — *"retry_controller·provenance = 해로움
  (treat 0.077 < control 0.154)"*. 그리고 같은 문단이 *"raw로 재확인 필요(미검증)"*.
- **왜 이것이 재론이 아닌가 (3 근거)**:
  1. **다른 체제**: 그 런은 **replay-safe 이전의 deny 방식**. 당시 deny는 히스토리를 오염시켜 replay-divergence 크래시를
     냈고(그래서 `REPLAY_SAFE_GATE_DESIGN`을 만들었다), 크래시는 실패로 계상됐다. **prov는 regen 체제서 한 번도 측정된 적이
     없다**(코드상 `T2_GATE_REGEN`과 **상호배타**·`t2_run_gated:115`).
  2. **다른 술어**: 옛 `_provenance_deny`는 *"인자가 이전 tool 출력에 등장하는가"* = **휴리스틱**(정당한 id도 차단 → over-block이
     harm 기전으로 유력). E9는 **레지스트리 멤버십** = false-positive 구조적 0.
  3. **새 증거**: §1 census(그때는 없었다) — 날조가 우리 write-arg 결손의 **frontier-분리 요인**이고 scaffold·scale 둘 다 미커버.
- **그럼에도 harm 선례는 살아있다** ⇒ **Δspurious·transfer율·turn 예산을 GO 조건에 강제**하고, **NO-GO면 즉시 폐기**.

### 2.4 반증 가능한 예측
- **P10**: E9는 날조 write를 **0**으로 만든다(술어가 결정론이므로 자명) — *측정 대상은 그 다음이다*.
- **P11 (핵심)**: deny+regen 후 모델은 **(a) 올바른 id로 복구** 하거나 **(b) 포기/과행동** 한다.
  **P11a**: pass 개선 > 0 (상한 +3.3pp). **P11b**: `Δspurious ≤ 0` ∧ `transfer율 불변`.
  **옛 prov harm이 재현되면 P11b가 깨진다** → 그때 harm 기전이 술어(over-block)가 아니라 **deny 자체**임이 밝혀진다.
- **P12**: over-block(정당 id 차단) = **0**(레지스트리 멤버십이므로 구조적).

### 2.5 측정 (무료先)
- **Phase A (offline·무료)**: 기존 궤적의 날조 write-call 32건 전수 → (a) 술어가 정확히 그 32건만 잡나(오탐 0) (b) 그 sim들의
  실패 원인이 정말 날조인지 **per-case 정독**(다른 이유로도 실패했으면 상한이 깎임).
- **Phase B (live-smoke·무료 로컬 user-sim)**: E9 배선(regen 체제) · 짝 arm(off/on) · fail-set(날조 sim) + control(passing sim).
  측정: pass 복구 · **Δspurious** · **transfer율** · turn 예산 · over-block.
- **Phase C (유료 1회·승인)**: nt=4 → compliant pass 갱신.

### 2.6 GO / NO-GO
| Phase | GO | 실패 시 |
|---|---|---|
| A | 술어 오탐 0 ∧ per-case서 날조가 실패의 **근인** | 상한 하향·재설계 |
| **B** | 복구 > 0 ∧ **Δspurious ≤ 0** ∧ **transfer율 불변** ∧ over-block 0 | **옛 harm 재현 → 즉시 폐기**([[03]] 존중) |
| C | compliant pass 개선(확인용) | — |

## 3. E1′ — 격리 formalize 서브콜 (강등·criterion 조각)
날조를 뺀 **진짜 criterion 잔여 = 우리 5 vs o4-mini 2 (초과 3)** — E9보다 훨씬 작다. 설계는 유지하되 **후순위**.
- **동기**: C10(부작용은 scope에서 온다) 검정 + criterion 잔여.
- **메커니즘**: write-tool의 선택 인자 결정점서 **격리 짧은 컨텍스트 서브콜**(후보 C = 전 변형·필터 금지, 목표 G = user turns)
  → 모델이 formalize+select. **메인 에이전트는 사고하지 않음** ⇒ persistence 부작용 채널 폐쇄(P8).
- **두 갈래 강제(측정 근거)**: variant **simple** .760→.840(CoT가 삼) / variant **compound·budget** .538→**.538**(CoT가 못 삼)
  ⇒ compound는 **형식화(LLM) → 결정론 실행(argmin/filter)** 필요([[10]] 정합).
- **범위 제외**: ⋈(order_id) — [[05]] join-resolver 금지 · C9: frontier도 동일 실패(경쟁 이득 0).
- **[[05]] Q3**: 결정론 실행 arm은 **YES** ⇒ 기본 NO·측정으로만 정당화.

## 4. 원장 갱신
| | 변경 |
|---|---|
| **C9** | **기전 라벨 정정**: frontier 격차의 지배 조각 = **operand 날조**(criterion 아님). [M] |
| **신규 C11** | **operand 날조 = frontier-분리 결손**(우리 5.9% vs gpt-4.1·claude-3.7 **0.0%**)·scaffold·scale 둘 다 미커버·**decidable** | **[M]** |
| **큐** | **E9(operand grounding) 신설·최우선** · E1′ 강등(criterion 조각) |

## 5. 방법 교훈 ([[08]] 3회차)
집계 교차표(item/variant 13 vs 2)에서 **기전을 추정**해 설계를 세웠다. **per-case 정독이 그 기전을 반증**했다
(날조 7 / criterion 5). 포렌식 가드가 커밋 직전에 잡았다. **분류표는 기전이 아니다 — 궤적을 읽어야 기전이다.**
