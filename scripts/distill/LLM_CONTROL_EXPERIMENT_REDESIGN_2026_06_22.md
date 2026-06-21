# LLM CONTROL EXPERIMENT REDESIGN — 통제점 × 강제강도 (2026-06-22)

> **권위 재설계 doc.** 진입 = 메모리 `06-NOW` + `07-control-not-prompt` + `HANDOFF_2026_06_22` §1-2.
> 상위 정합 = `EXPERIMENT_DESIGN §0★★`(목적함수·평가8) · spine `CAPABILITY_LEVER_ALLOCATION_DESIGN_2026_06_21` · 불변 [[05-fixed-vs-variable]].
> 이 doc = 논문1(per-step 신뢰·싸게)을 **"어느 통제점·어느 강제강도로 통제해야 효율적인가"** 로 reframe.

---

## §0. 착수 전 [[05]] 결정질문 (이 doc가 *정의*하는 실험 cell들에 대해)

이 문서는 메타-프레임이라 자체로는 scaffold/A2를 안 건드린다. 단 *정의하는 실험 cell*이 결정론을 늘릴 수 있으므로 프레임 차원에서 답한다:

1. **도메인-특화 순증?** — ❌ 통제점(C0..C4)·강제강도는 전부 도메인-일반 메커니즘. cell이 도메인 리터럴을 scaffold/A2에 넣으면 그 cell은 [[05]] 위반으로 *기각*(grep `if domain`=0 불변).
2. **유동적 판단을 결정론에 동결?** — ⚠️ **이게 이 프레임의 핵심 측정대상.** 모든 enforced 통제(C1/C2)는 flexibility-loss를 *비용으로 계상*하고 측정한다(§4-②). null = "강제가 공짜" 기각.
3. **scaffold가 도메인 행동 수행?** — ⚠️ autofetch(=C1-perform)는 scaffold가 fetch를 *수행* = 결정질문3 yes ⇒ **기본 아님**. 강제강도 사다리에서 *최강·최후* 옵션이며 측정(§4-①)으로만 정당화. 기본 = C1-deny(사실 게이트) + 모델 fetch-first(C0 또는 C4).

---

## §1. 최상위 reframe — 증명된 전제

**soft control(prompt+memory+skill)은 hard-constraint엔 불충분 — 두 실증:**
- (a) **frontier-나의 autofetch 드리프트**: ★★★★★ 메모리로도 [[05]]를 우회 = prompt-following이 *크기 무관* hard-constraint엔 약함(`07-control-not-prompt`).
- (b) **§35b/C3 sweep**(2026-06-22 회수): gate deny-only(=prompt "실값 obtain")가 7B서 stall. 아래 §5 표가 실측.

⇒ **hard-constraint = enforced(★레버명=scaffold·결정론 gate/resolve·구 "hook") 또는 learn(weights) 필요.** (xgrammar=별개 메커니즘=decode-time TYPE 강제 config·scaffold-gate 아님·`PROGRAM §3`.) 논문1의 질문이 "무엇을 offload하나"에서 **"어느 통제점에서 어느 강제강도로 거나"** 로 한 단계 올라감.

**★우리 scaffold = 모델 툴콜에 건 hook**(`t2_gate_patch._execute_tool_calls` 몽키패치 = PreToolUse on the model). self-hook(`scaffold_guard.py`)이 *나*에게 같은 걸 건 게 동형 검증.

---

## §2. 통제점 × 강제강도 매트릭스

### 통제점 (control point = tool-loop 어디서 개입)
| | 통제점 | 시점 | 메커니즘 |
|---|---|---|---|
| **C0** | prompt | 생성 전 | 시스템/few-shot 지시 (soft·기각된 채 출발) |
| **C1** | pre-call gate | 툴콜 직전 | 인자/권한 검사 후 deny·require·substitute·perform |
| **C2** | post-call | 툴결과 후 | ground·retry·abstain |
| **C3** | plan-structure | 다단계 경로 | 경로 제약·탐색 (논문2 영역) |
| **C4** | learn | 오프라인 | weights에 내재화 (LoRA·무붕괴) |

### 강제강도 (enforcement strength = C1 내부 사다리·★최소행동 원칙)
C1 한 통제점 안에서도 행동량이 다르며 **block < substitute < perform** 순으로 flexibility를 뺏는다:
- **deny/block**: 위반 툴콜 거부, 모델이 스스로 고치게 (최소·유동성 보존)
- **require**: 선행 사실/필드 강제, 모델이 채우게
- **substitute**: 엔진이 인자를 *교정*
- **perform**: 엔진이 *대신 행동 수행* (autofetch = 이 끝·결정질문3 yes·최후수단)

**★원칙(autofetch 교훈): 효과가 같으면 항상 더 약한 강제를 택한다.** null hypothesis = "C0 prompt suffices" 는 *기각된 채* 출발(soft 불충분 실증). 그러나 거꾸로 "강할수록 좋다"도 아님 — 강제강도↑ = flexibility-loss↑ + A2-growth↑ = 비용. **답 = reliability를 사는 *최소* 강제강도(crossover).**

---

## §3. 기존 실험 cell 매핑

| cell | 통제점·강도 | 결과·위치 | 판정 |
|---|---|---|---|
| **keystone** | C1 엔진(gate engine·A2-구동) | `gate_interpreter.py`·validate PassA/B=0·parity A_notfound=14 | ✅ C1 메커니즘 도메인-일반화 완료(grep if-domain=0·airline A2-swap) |
| **C3 sweep** | C0(prompt) vs C1(autofetch) | §5 표·`c3_prompt_sweep.log` | ✅ C0 단독 grounding 못 닫음·C1 닫음(soft 불충분 실증) |
| **C8** | C2-retry (post-call recovery) | `M_A_RESULTS §35c`·`c8_crux.py` | ⚠️ retry=잘못된 레버(지배실패=인증후 producer 미호출·placeholder 날조지 복구실패 아님) |
| **C10** | C4-learn (operand) | `REST/C10 설계`·미빌드 | 🔜 operand=scale-불변 학습잔여(§35 B-plateau) |

**해석**: 메커니즘 축은 keystone으로 닫혔다. 남은 건 *각 failure를 어느 통제점·강도로 거는 게 최소비용인가* 의 **배정 실험**.

---

## §4. 신규 실험 cell (3개)

### ① 강제강도 crossover (per-failure 최소강도 측정)
- **질문**: A_notfound(entity-id grounding) failure를 닫는 *최소* 강제강도는? C0(prompt fetch-first) → C1-deny → C1-require → C1-perform(autofetch) → C4(learn) 사다리에서 각 강도의 (reliability, flexibility-loss, A2-growth, cost) 측정.
- **예측**(§5 부분실측): C0 전부 못 닫음(A 17~28). C1-perform(autofetch) 닫음(A 12). **빈칸 = C1-deny/require 와 C4-learn** — 더 약한 enforced 또는 learn이 perform만큼 닫나? 닫으면 autofetch(최강·A2성장·동결)는 *불필요*.
- **GO/NO-GO**: 더 약한 강도가 동등 reliability면 autofetch 강등. 오직 perform만 닫으면 → 그 비용(flexibility-loss·A2성장)을 measured-justified로 명시.

### ② flexibility-loss 측정 (강제의 숨은 비용)
- **질문**: enforced 통제가 모델의 *정당한* 유동성을 얼마나 죽이나. = false-block rate(엔진이 옳은 툴콜을 막은 비율) + over-deny(§validate already 0) + held-out task에서 강제가 가로막은 정상경로.
- **메트릭**: per-cell { reliability gain Δpass, flexibility-loss(false-block/over-constrain), A2-growth(필드 수), build/maintenance }. 목적함수(§6)의 분모.
- **왜 신규**: 지금까지 reliability만 봄. 강제강도 사다리의 트레이드오프를 *수치화*해야 [[05]] 결정질문2가 측정으로 닫힘.

### ③ C3 plan-structure (다단계 경로 통제·논문2 접점)
- **질문**: 단일-스텝 통제(C1/C2)로 못 닫는 *다단계* failure(조합폭발·경로선택)를 plan-structure 제약으로 통제하면? = 논문2(`PATH_SELECTION_AXIS_DESIGN`)와의 직교 접점.
- **scope**: 이 doc(논문1)에선 *경계 표시*만 — C1/C2가 닫는 한계 너머가 C3 영역임을 census로 식별. 본 실험은 논문2.

---

## §5. C3 sweep 실측 (2026-06-22 회수·§35b 정정)

retail n=114·base 7B·`c3_prompt_sweep.log`. **A_notfound = entity-id grounding failure(핵심신호)·B = operand/write-args.**

| arm | 통제 | pass1 | A_notfound | B_wrong_write |
|---|---|---|---|---|
| base | (없음) | 23/114 | 25 (0.27) | 32 (0.35) |
| nofab | C0 prompt | 27/114 | 26 (0.30) | 35 (0.40) |
| fetchfirst | C0 prompt | 23/114 | 28 (0.31) | 26 (0.29) |
| fewshot | C0 prompt | 26/114 | **17 (0.19)** | 30 (0.34) |
| structured | C0 prompt | 26/113 | 26 (0.30) | 29 (0.33) |
| **engine** | **C1 perform(autofetch)** | 27/105 | **12 (0.15)** | 21 (0.27) |

**판독**:
- **C0(prompt) 전 변종이 grounding(A)을 못 닫는다**: 최선 fewshot A=17이 base 25보다 낮으나 pass는 안 오름(26). 나머지 C0는 A를 *악화*(26~28). = **soft control이 grounding hard-constraint엔 불충분 — §1(b) 실증.**
- **C1-perform(autofetch)만 A를 12(0.15)로 닫는다**(최저). 단 pass=27/105(분모 105·일부 태스크 탈락) = grounding 닫아도 pass는 B(operand)·기타가 막음.
- **§35b "autofetch pass 2×" 정정**: 이번 denoised sweep에선 pass 게인이 작다(27 vs base 23). autofetch의 진짜 효과는 *A_notfound 닫기*에 국한·pass 천장은 B/operand가 결정. ⇒ **autofetch는 grounding-only 레버지 만능 아님.**
- **빠진 3번째 축 = C4(모델 fetch-first 학습)**: §4-① crossover의 핵심 빈칸. C0(prompt)로 안 되는 fetch-first를 C4(learn)로 내재화하면 autofetch(C1-perform·A2동결) 없이 닫히나? = autofetch 원칙 확정(§7-3)의 결정실험.

---

## §6. 효율 목적함수 (배정 규칙)

각 failure → **최소비용 enforced 통제** 배정:

```
efficiency(cell) = reliability_gain
                 / ( build + inference_OpEx + flexibility_loss
                     + A2_growth + learn_cost + maintenance )
```

- **배정 사다리**(약→강·동등 reliability면 약한 쪽): C0 prompt → C1-deny → C1-require → C1-substitute → C1-perform → C4-learn.
- **null 기각**: "C0 prompt로 충분" 은 실증으로 기각된 채 출발.
- **enforced 최소행동**: enforced 안에서도 block > substitute > perform (autofetch 교훈).
- **공통 적 = A2-growth·flexibility-loss**(둘 다 [[05]] 위반축). 공통 친구 = C4-learn(무붕괴면 A2 안 키우고 flexibility 안 죽임) + C1-deny(최소 enforced).

---

## §7. 다음 행동 (HANDOFF §2 순서)

1. ✅ **이 doc 작성** = §2.1 완료.
2. ✅ **C3 sweep 회수** = §5 (완주·GPU free).
3. 🔜 **autofetch 원칙 확정** = §4-① crossover의 C4-learn 빈칸을 채워라 — 모델 fetch-first 학습(C4)이 autofetch(C1-perform) 없이 A를 닫나. 닫으면 autofetch 강등→gate-only(C1-deny)+C4 기본. = C3/C8/C10 전부 reframe.
4. 🔜 **C10(operand) = C4-learn leg**: §5 B-plateau(scale-불변)가 C4의 정당 타깃. "C0 안 되면 C4" 결정실험. 설계=`C10_OPERAND_LORA_DESIGN`(키스톤 위).
5. 🔜 **키스톤 A2 과성장 정리**: `identifying_arg_types`·`placeholders`(도메인-일반) → scaffold 기본값으로 되돌려 minimize-A2.

**GO/NO-GO 헤드라인**: 각 failure가 measured-minimal 통제점·강도에 배정되고(가이드라인), C0-불가가 C1-deny 또는 C4로 닫히며(autofetch 불필요 입증 시도), flexibility-loss가 수치화되면 = 논문1 "per-step 신뢰를 *최소 강제*로" 성립.

---

**불변 정합**: [[05-fixed-vs-variable]](결정질문3·minimize-A2)·[[07-control-not-prompt]](soft 불충분)·[[03-anti-drift]]·[[13-absorption-priority]](흡수 우선순위 scale→learn→scaffold). 상위 = `EXPERIMENT_DESIGN §0★★`·`CAPABILITY_LEVER_ALLOCATION_DESIGN`.
