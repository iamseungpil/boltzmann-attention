# 통합 Scaffold 구현 설계 (2026-06-18) — step-orchestrator + 결합 + resolve wiring + GateInterpreter

> **범위**: `INTEGRATED_TBOX_DESIGN_2026_06_18.md §9[잔여]`가 남긴 구현 미정 4개를 *구체화*한다. **아키텍처 재설계 아님** — §5 분해 멀티-스페셜리스트를 코드로 실현하는 설계. 상위 권위 = `INTEGRATED_TBOX_DESIGN §5`·`FIXED_VS_VARIABLE.md`·`GATE_INTERPRETER_UNIFICATION_DESIGN_2026_06_18.md`. 메모리 = `06-NOW`·`05-fixed-vs-variable`·`03-anti-drift`.
> **불변 재확인**: 단일 merged LoRA 없음(각 스페셜리스트 독립 호출). 고정={TBox weights + Scaffold 엔진}/변경={ABox}만. per-bench 분기 0. 측정=실 τ² user-sim e2e만.

---

## 0. 메우는 4개 (§9 잔여)
| # | 잔여 | 본 문서 절 |
|---|---|---|
| A | GateInterpreter(gate_spec) 통일 — t2_gate_patch 구체 마이그레이션 | §2 |
| B | resolve_selection 도구의 t2 레지스트리 wiring | §3 |
| C | step-orchestrator scaffold (스텝별 어느 스페셜리스트 LoRA·얽힘 분해) | §4 |
| D | 결합 알고리즘 명세 + decidable-비율 측정 | §5 |

---

## 1. ★주입점 = 기존 monkeypatch 2곳 (발명 아님·코드 실측)
tau2 에이전트 루프는 두 지점만 패치하면 전부 제어된다 (`t2_gate_patch.py` 실측):

```
LLMAgent._generate_next_message(self, message, state)   ← [생성 레벨]
    └─ la.generate(model=self.llm, tools=self.tools, messages=...)   # model= 을 스텝마다 swap 가능
            ↓ assistant 턴(tool_calls) 산출
BaseOrchestrator._execute_tool_calls(self, tool_calls)  ← [실행 레벨]
    └─ gate.check → (allow) orig 실행 / (deny) ToolMessage(error)
```

- **[생성 레벨]** = **step-orchestrator(C)** + 스페셜리스트 라우팅. 기존 `apply_provenance_regen`이 이미 *작업본(work=list(state.messages))에서 generate 재호출·state 무오염*하는 패턴 확립 → 그대로 확장.
- **[실행 레벨]** = **GateInterpreter(A)** + **resolve_selection 실행(B)** + **결합 offload(D)**. 기존 `gated()`의 `RetailGate.check`/provenance/ground 자리를 일반화.

**= 두 패치 함수를 `apply_integrated()` 하나로 묶는다** (현 `apply()`/`apply_provenance_regen()` 대체). 벤치 측정 인터페이스(turn·error·user-sim·보상) 불변.

---

## 2. (A) GateInterpreter 통일 — t2_gate_patch 마이그레이션
설계·스키마·kind closure = **`GATE_INTERPRETER_UNIFICATION_DESIGN_2026_06_18.md` 그대로**(중복 금지). 본 문서는 *t2_gate_patch 결선*만 구체화:

1. `t2_gate.py`: module-global `AUTH_TOOLS`/`WRITE_TOOLS`/`USER_SCOPED` → `GATE_SPEC[*]["applies_to"]` 데이터로 흡수 + 각 항목에 `"kind"` 필드 추가(auth/confirm/ownership/notice). `RetailGate` 클래스 폐기 → `GateInterpreter(gate_spec, resolvers)` (설계서 §3 코드).
2. `gate_spec`는 파일서 로드: `load_gate_spec(domain)` → `tau2_adapter/<domain>_gate_spec.json` (ABox A2). retail = 현 GATE_SPEC 덤프(`t2_gate.py --export-spec` 이미 존재). airline = 같은 스키마·값만(write=update_reservation_*·cabin attr).
3. `t2_gate_patch.gated()`: `RetailGate(db=env.tools.db)` → `GateInterpreter(load_gate_spec(env.domain_name), resolvers=_engine_resolvers(env))`. `GATE_DOMAINS` 하드셋 폐기 → gate_spec 존재 여부로 판정(spec 없으면 ungated = airline 현 동작 보존하되 spec 주면 자동 게이팅).
4. ownership G3의 `db.orders` 직참조 → `resolver_path`(order_id→get_order_details→user_id)를 `resolvers`(엔진 제공 결정론 lookup)로. = per-domain 코드 0.
5. **검증**: `grep -E "if.*(retail|airline|domain|bench)" GateInterpreter` = 0. retail+airline 같은 인터프리터 unchanged 작동. 빈 spec→전부 allow(ablation).

---

## 3. (B) resolve_selection wiring
설계 = `INTEGRATED_TBOX_DESIGN §3a/§3b` 그대로. 결선만:

- **카탈로그 노출(ABox A1)**: `resolve_selection(op, attr?, among?, dir?, k?, set?) -> item_id` 를 도메인 도구 목록에 추가(tau2 `environment.get_tools()` 패치 — A1 데이터). **`anchor_id`는 모델-가시 인자서 제외**(`tau2_op_resolver.py:74-77`·order_id 날조 재수입 차단). 스키마=기존 op-IR(`{op,attr?,among?,dir?,k?,set?}`).
- **실행**: `_execute_tool_calls`에서 `tc.name == "resolve_selection"` → `ma/tau2_op_resolver.py`의 엔진(`resolve_op_tau2`/`resolve_operation`) 호출. anchor(수정대상 item)는 **offload 층이 직전 fetch 결과(context anchor)서 grounding**해 엔진에 주입(모델은 op·delta만 명명). 반환 item_id → ToolMessage.
- **★operand = LLM delta 명명 + 엔진 조립 (별도 전용 LoRA만 금지·§23D)**:
  - **NL→formalize(delta 명명) = LLM(facet3 스페셜리스트)이 함** — resolve_selection의 `set`/`among` 인자로 **intensional delta**("그 item color를 silver로"·변경 attr만) emit. **이게 operand의 NL→formalize.**
  - **엔진(offload) = 명명 안 한 부분만**: keep-rest copy(anchor 나머지 attr 유지·`resolver:82`) + concrete `item_id` 해소(extensional·모델은 모름·날조 차단) + multi-attr 조립.
  - multi-attr 과소추출(§20-B 천장 0.44) = **per-attr 분해 명명(여전히 LLM)** → 엔진 조립으로 회피(§22 0.51→0.87) = "decomposition-offload"(분해=명명은 LLM·조립=엔진).
  - ⚠️ operand를 *별도 facet4 전용 LoRA로 학습*하면 라우팅 퇴행(§23D: synth SET_EXACT 1.00이나 τ² 0.44→0.30) — **delta 명명은 facet3가 op와 같은 콜에서**(별도 LoRA 안 만듦). operand formalize를 LLM이 안 한다는 뜻 *아님*.
- real action 도구(exchange_…) 대체 아님 = 인자 *계산* 보조(계산기 동형·§3b). 벤치 성공기준(DB state) 불변.
- ⚠️ **선결 의존**: 모델이 resolve_selection을 *native tool_call로 emit*하도록 학습된 synth content-op 스페셜리스트 필요(§6·facet3). 도구만 노출하고 스페셜리스트 없으면 base는 안 부름 → §8 단계순서 준수.

---

## 4. (C) ★step-orchestrator scaffold — 핵심 신규
**문제**: 한 assistant 턴이 여러 facet(게이트+grounding+threading+content-op)을 얽어 요구. 단일 merged LoRA 금지 → 각 스페셜리스트를 *독립 호출*해 typed 스텝으로 분해해야.

**메커니즘 = 턴-내부 typed 파이프라인** (벤치 비가시·기존 regen "작업본" 패턴 동형):
`_generate_next_message` 안에서, state 오염 없이 작업본 위에 스텝을 돌리고 *최종 native tool_call 하나*만 반환.

```
generate_next(message, state):
  work = list(state.messages)
  steps = route(state)                       # ① 결정론: 이 시점에 필요한 facet **집합** (단일 선택 아님)
  verdicts = {}
  for s in steps:                            #   전제조건 facet=순차 / 종합필요 facet=병렬 수집
      spec = SPECIALIST[s.facet]             # ② facet→스페셜리스트 LoRA (model 이름)
      cand = la.generate(model=spec.lora, tools=visible_tools(s), messages=work)
      verdicts[s.facet] = facet_check(s, cand, state)   # ③ 그 facet 결정론 per-step verify + 명확성 신호
  return combine(steps, verdicts, state)     # ④ §5 결합·메타-판정 (emit / consensus / defer)
```

- **① route(state)** (scaffold·고정·per-bench 분기 0): 대화상태 → *필요 facet **집합***. **단일 선택 아님** — 얽힌 결정엔 여러 facet을 모은다(병렬). route를 "순서 강제만"으로 좁히면 = `04 §22 "잔여=오케스트레이션뿐"` 오류 재발(사용자 교정 2026-06-18).
  - **전제조건(순차·결정론으로 닫힘)**: 미인증+user-scoped → **facet1(SOP/gate·auth)** / 미해소 구체값 → **결정론 GROUND(offload·학습 아님)** / 출력→입력 바인딩 → **facet2(TaskBench threading)**.
  - **내용(스페셜리스트가 emit)**: 선택/정렬/치환 의도 → **facet3(Synth content-op)** → resolve_selection(op + **intensional delta**) emit = op·operand-formalize 둘 다 LLM. **operand의 *조립*(keep-rest·multi-attr·concrete `item_id`) = 엔진(§3) — *명명*은 LLM·*조립*만 엔진.**
  - 신호 = 카탈로그 타입(A1) + gate_spec 상태 + 직전 tool 출력 유무. **도메인 문자열 0.**
- **★route 경계 = procedure-offload 방어 (③·`마스터 §1`)**: route는 **절차를 *발명*하지 않는다.** dispatch는 오직 (a)범용 data-dependency(미fetch 값 참조 불가) + (b)gate_spec(ABox) policy 집행에서만. "어느 op/도구"의 NL 내용은 **스페셜리스트가 emit**(route 아님). = DGGATE 선례(policy를 ABox서 *읽지* 절차 발명 안 함·`GATE_INTERPRETER §1`). ⚠️ **per-bench 분기 0은 필요조건일 뿐** — 도메인-일반 결정론이라도 절차를 발명하면 L0(NL→dirgraph를 외부가 대신 = 전이주장 붕괴).
- **② SPECIALIST 맵** = §6. multi-LoRA 서빙(`vllm serve BASE --enable-lora --lora-modules f1=… …`·per-request `model=<name>`).
- **③ facet_check** = 그 facet 한정 결정론 검증 (gate.check / provenance / threading-id / resolve 가능성) **+ 명확성 신호**(단일 결정 vs 복수 후보) — combine 메타-판정 입력.
- **④ combine** = §5 (verdict 튜플 종합·메타-판정).

**얽힘 = 순차 분해 + 병렬 종합 (둘 다)**: 순서로 닫히는 전제조건(gate→ground→thread)은 순차 진행. **순서로 안 닫히고 여러 facet을 *종합*해야 하는 결정은 병렬 수집 후 combine**(§5). 이 "복합 종합"이 사용자가 지적한, route만으로 해결 안 되는 결정.

**Fallback(점진)**: 1차 = 턴당 필요 facet만(대개 1-2개)·결정론 combine. 측정 후 비-decidable 잔여 크면 consensus LoRA. = §8 C1→C2.

---

## 5. (D) 결합(combine) + 메타-판정 + decidable-비율 (★thesis 핵심 측정)
combine = facet-verdict 튜플 → **emit / consensus / defer**. **순서로 안 닫히는 "복합 종합 결정"의 자리**(`04 §22` offload 경계 ②).

```
combine(steps, verdicts, state):
  if any(not v.ok for v in verdicts.values()): return regen_feedback(...)  # facet 미충족 → 재생성
  decidable = META_DECIDE(verdicts, state)    # ★메타-판정 (1차 결정론·verdict 구조 속성)
  if decidable: return emit(resolve_tool_call(verdicts))   # 규칙으로 닫힘 → 결정론 emit
  else:         return consensus(verdicts, state)          # 모호/충돌 → consensus LoRA (잔여만)
```

- **META_DECIDE = "이 결합이 decidable인가" 판정 (1차 결정론)**: 전 facet verdict가 *명확(단일)·무충돌*이면 decidable·하나라도 모호/충돌이면 비-decidable. = verdict 집합의 구조적 속성(선택기=결정론·`10-roles-deterministic`).
- **decidable → 결정론 종합**(offload): gate∧grounded∧threaded∧resolved → emit.
- **비-decidable → consensus LoRA**(그 결합 결정만·잔여 학습·다른 facet 미접촉). **1차엔 consensus 학습 0** — 비-decidable은 defer/로깅.
- **★측정 = 실 e2e autopsy 3-카운트**:
  1. **decidable-emit** — 결정론이 닫고 *맞음*.
  2. **ambiguous→consensus** — 결정론이 모호 판정(학습이 풀 자리).
  3. **wrongly-decided** — META_DECIDE가 "명확"이라 했으나 *실제 틀린 emit* = **메타-판정 결정론의 오류율**.
  - **★#3 검출법(⑤·근사 귀속)**: 실 e2e는 *최종 task 보상만* 주고 per-combine GT를 안 줌 → #3은 휴리스틱 귀속으로 잡는다: emit이 (i)직후 gate-deny 유발 / (ii)downstream tool 실패 / (iii)최종 DB-state 불일치 → 그 결합 결정에 귀속. **이 귀속 없으면 #3 미검출 → decidable-비율이 #1/(#1+#2)로 퇴화해 메타오류를 못 잡음.**
  - decidable-비율 = #1/(#1+#2+#3). 대부분 decidable = offload 지배 = 분담선 실측(`#2`·thesis 핵심). **실 e2e 로그서만**(오프라인 프록시 금지).
- **★메타-판정 학습 승격 규칙 (사용자 가설 2026-06-18)**: **#3(wrongly-decided)이 높으면 = 결정론 META_DECIDE가 모호성을 못 잡음 = "decidable 여부 판정"도 LoRA 학습 필요.** **단 *증거(높은 #3) 후* 도입** — 처음부터 메타-판정을 학습하면 *측정하려는 분담선(decidable-비율)을 블랙박스화*해 thesis 목적 자기파괴(`03-anti-drift 6`·`10-roles` 선택기=결정론). = "결정론 1차 → 측정 → 증거 시 학습 승격".

---

## 6. 스페셜리스트 맵 — 학습 3 + operand offload (머지 안 함·CFB 폐기)
**학습 스페셜리스트 = SOP·TaskBench·Synth(content-op) 3개.** operand·concrete·gate enforcement = 전부 결정론(학습 아님).

| 역할 | 담당 | 후보/구현 | emit/실행 | 상태 |
|---|---|---|---|---|
| facet1 flow/gate | SOPBench LoRA | `phase4/qwen7b_tbox_*`(LODO) | 도메인 tool_calls | ⚠️ **e2e 기여=측정 대상**(held-out adapter 기여≈0·scaffold 우위·SOP:583) |
| facet2 threading | TaskBench LoRA | `qwen7b_tb_*`·`qwen7b_nfc_lodo_*` | 멀티 tool_calls·threading | ⚠️ **TaskBench→τ² 0건 미실행=측정 대상** |
| facet3 content-op | Synth LoRA | `synth_to_nativefc.py` 재생성 | **resolve_selection(op + intensional delta)** | 🔄 ★C0 keystone(native τ² 전이=§21 동급?) |
| operand 조립 | **엔진(offload)** | `tau2_op_resolver` + GROUND + per-attr decomp | keep-rest·multi-attr·concrete 해소 | = 결정론·**학습 아님**(§23D 퇴행) |
| gate/concrete | **엔진(offload)** | GateInterpreter + provenance/GROUND | enforce·grounding | = 결정론·ABox-도출 |

★**CFB 폐기 + operand 전용학습 금지 (사용자 결정 2026-06-18·리뷰 ①)**: operand의 **delta 명명(NL→formalize)은 facet3 스페셜리스트(LLM)가** op와 *같은 콜*에서 함. **별도 facet4 전용 LoRA는 안 만든다**(전용 학습=§23D 라우팅 퇴행·multi-attr은 per-attr 분해 명명으로 회피). CFB의 concrete grounding + keep-rest + 조립 = **엔진(offload)**. ⇒ facet3가 op+args 한 콜이라 별도 LoRA 쪼개기(C2 압력)가 사라짐(리뷰 ②). (operand-formalize를 LLM이 안 한다는 뜻 아님 — 전용 LoRA만 안 둠.)
⚠️ **검증 선결**(§8 C0): 각 학습 어댑터가 (a)자기 facet native-FC emit (b)base Qwen2.5-7B 위 LoRA 로드 sanity. SOP/TaskBench의 e2e 학습 기여는 *측정 대상*(자산 단정 금지).

---

## 7. 3-way e2e 하버스 + ABox-swap (real_e2e 확장)
`real_e2e_base.sh` 골격 재사용·arm만 추가:
- **arm0 base** = 현 `real_e2e_base.sh` (floor·pass^1≈0.17 검증).
- **arm1 스페셜리스트-only** = step-orchestrator(C) ON·결합/offload OFF(gate·resolve 끔). 라우팅만. ⚠️ facet3 콜은 resolve 엔진 OFF면 **no-op = bare 협업붕괴를 보이는 게 arm1의 *의도된 목적***(헤드라인 "offload 없으면 낮다"·arm2와 대비).
- **arm2 full** = C + GateInterpreter(A) + resolve(B) + 결합 offload(D).
- **전이 매트릭스** = arm2 unchanged·`load_gate_spec`/카탈로그(ABox)만 swap → retail·airline·(SOP-Bench). 재학습 0. †타깃별 행사 facet 다름(airline=facet3 content-op 포함·SOP-Bench=주로 facet1 flow+gate·content-op 거의 없음) — "같은 시스템·ABox-swap"이 타깃마다 다른 facet 행사 = 정상·각 facet 일반성은 별도 입증.
- 보상 = 결정론(τ² DB∧NL∧comm·compliant-pass)·user-sim=gpt-4.1(COST GUARD).
- **헤드라인 = arm2 > arm1 > arm0 상대 Pareto-지배**(절대수 약속 금지·§5b Risk B).

---

## 8. ★단계 빌드 (각 단계 실측·작은 증분)
- **C0 ★GO/NO-GO keystone (sanity 아님·리뷰 ④·동전던지기·임계경로)**: native facet3가 **retail+airline τ² ABox-swap 전이를 op-IR §21(0.44)과 동급 재현하나.**
  - 배경: §28 = native synth held-out **1.00**(op-naming 보존·신뢰·*synth-level만*) / §29 = facet3_native_ep1 τ² **역전이 0.19<base 0.34**(단 **오프라인 op-eval=신뢰불가·철회**·HANDOFF §3). ⇒ **C0 = §29를 실 user-sim e2e로 재시험**(오프라인 프록시 폐기).
  - §33 진단(역전이 3원인): (a)`synth_to_nativefc` 데이터가 MD_route보다 좁음(다양성) (b)native 포맷 근본(§23E) (c)operand 과적합(facet3가 `set`까지 gold 학습→§23D 퇴행).
  - **GO** = §21 동급 → 진행.
  - **★NO-GO Plan B (갈림은 autopsy로)**: ① op 맞힘(recognition↑)·operand만 틀림 → **데이터 다양성↑ 재생성 + operand SFT-가중 축소**(`set`을 loss서 약화·routing 보존) ② native emit 자체 실패(op도 틀림) → 학습데이터/converter 포맷 문제 ③ 둘 다 OK인데 전이 degrade → **§23E 근본 = resolve_selection 접근 재설계**. (막지 말고 *가장 먼저·작게* 던질 것.)
  - 병행: SOP·TaskBench native-FC emit·로드 확인(기여는 측정 대상).
- **A GateInterpreter 통일** + retail/airline grep-clean 검증 (keystone·먼저).
- **B resolve wiring** + facet3 스페셜리스트가 resolve_selection emit→엔진(operand 조립 포함) 실행 e2e 1태스크 smoke.
- **C1 라우터(턴당 필요 facet)** + multi-LoRA 서빙 → arm1 측정.
- **D 결합 offload + 메타-판정** → arm2 측정 + decidable-비율·wrongly-decided 로깅.
- **C2(필요시)** 턴-내부 분해 → 재측정. (operand offload라 C2 압력↓·리뷰 ②.)
- **전이**: ABox swap → airline·SOP-Bench.

---

## 9. 자가심사 (리뷰 안건·치팅 방어)
- **thesis-정합**: 학습=도메인일반(facet1·2·3 = 3 스페셜리스트)·offload=decidable(gate/resolve/operand-decomp/combine)·ABox=swap·e2e=학습 TBox(base 아님). ✅ operand는 학습 아님(§23D).
- **per-bench 분기 0**: route()·GateInterpreter·resolver 전부 카탈로그/spec-도출. grep `if domain/bench` = 0 (CI 체크).
- **★route ≠ procedure-offload (③·분기0보다 강한 체크)**: route가 *절차를 발명하나* vs *gate_spec(ABox) policy + 범용 data-dep만 집행하나* 감사. 발명이면 L0(전이 붕괴)·DGGATE 선 유지. (분기0은 통과해도 절차 발명은 따로 친다.)
- **resolve=치팅 아님**: 도메인-일반 decidable 도구(ABox config 도출)·real action 대체 아님(§3b)·전이가 일반성 증명.
- **contamination 0**: τ²·SOP-Bench held-out.
- **decidable-비율 = 정직 측정**: 실 e2e 로그서·오프라인 프록시 금지.

## 10. 미결 / 결정 로그
- ~~#1 CFB 폐기 + facet4 처리~~ **해소(2026-06-18·리뷰 ①)**: CFB 폐기. concrete grounding = 결정론 GROUND offload. **operand-formalize = decomposition-offload (학습 아님)** — 별도 LoRA 두면 §23D 라우팅 퇴행 재발(retail 0.44→0.47·airline 0.44→0.30·MA §23D·§26·메모리 04/20). 학습 스페셜리스트 = SOP·TaskBench·Synth(content-op) **3개**.
- ~~② facet3/4 합성 갭~~ **자동 소멸(리뷰 ②)**: operand offload로 synth=facet3 단일 콜 emit·엔진이 args 조립 → C2 강제 사라짐.
- ~~#4 synth-native 확정~~ **해소**: `synth_to_nativefc.py` 재생성. anchor_id 제외 = `_emit_args` line 67-68 + resolver:74-77로 이미 보장. (단 실재 파일·`_emit_args` 확인 완료 ✓.)
- ~~#5 META_DECIDE~~ **해소(리뷰 ⑤+#5)**: 결정론 1차 + wrongly-decided(#3·근사 귀속 검출) 측정 → 증거 시 학습 승격 (§5).
- **잔존 #2 route() 결정론 vs 학습 (리뷰 #2)**: 전제조건 순서 = data-dep + gate_spec policy로 결정론 닫힘(DGGATE 선례). "어느 op/도구" NL 의도만 스페셜리스트. **route 자신의 decidable-비율도 §5처럼 측정**(모호 dispatch 비율) → 증거 시 학습 승격(META_DECIDE 동일 규율).
- **잔존 #3 턴-내부 분해(C2)**: C1로 충분한지 측정 후 — 선구현 금지(operand offload로 압력↓).
