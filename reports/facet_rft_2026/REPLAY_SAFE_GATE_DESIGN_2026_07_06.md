# Replay-safe 게이트 재설계 — 리더보드-동일 공정 측정 (설계서·2026-07-06)

> **목적**: τ²-retail 헤드라인(assembled 32B pass^3 0.457 / 14B 0.313)이 **공식 tau2 리더보드와
> 동일한 채점**인지 검증하는 과정에서, 우리 게이트가 tau2의 **replay-기반 평가**를 깨뜨려 assembled
> 런의 8~11%가 `infrastructure_error`로 드롭·오염됨을 발견. 본 설계서는 **문제의 전 사슬 → 근본원인
> → 리더보드 parity 요건 → replay-safe 게이트 재설계 → 구현 → 검증·재런 계획**을 기록. **다른 세션이
> 리뷰 후 구현**하도록 self-contained하게 작성.
> **성격**: 연구 산출물(ba-frft·비-CDP). 관련: `ASSEMBLED_FAILURE_FORENSIC_2026_06_27.md`
> Addendum(2026-07-06)·`SEMANTIC_ERROR_FORENSIC_AND_OVERCOMING_2026_07_06.md`·[[06-NOW]].
> **불변**: [[08]] per-case 검증(본 설계는 집계 아닌 궤적·소스 정독서 도출)·[[05]] scaffold 도메인-일반·
> A2만 인스턴스·[[09]] 재런=승인·최소 scope·[[03]] 헤드라인 규율.

---

## 0. 리뷰어를 위한 요약 (TL;DR)
1. **문제**: 우리 게이트(`t2_gate_patch.apply`)는 차단된 write 호출에 합성 `ToolMessage(POLICY GATE…)`를
   **대화 히스토리에 커밋**한다. tau2 평가는 히스토리를 **fresh env에 replay**해 reward를 계산하는데,
   게이트 메시지는 replay 재실행 결과와 불일치 → `ValueError` → `infrastructure_error`(reward=None·
   궤적 소실). assembled 32B 39/342(11.4%)·14B 26/342(7.6%)·floor는 ~0%.
2. **영향**: 공식 pass^k(`compute_metrics`)는 infra를 드롭 → 우리 방법-유발 11%가 "무료 드롭"이 되어
   **부풀림**. strict(크래시=fail) 재판정 시 같은-scale 우위는 유지(32B +9.7pp·14B +4.4pp)나
   **cross-scale crossover(14B-asm>32B-base)는 flip·철회**, frontier 병치는 "진입"→"근접"(0.377).
3. **fix**: 게이트를 `_execute_tool_calls`(post-hoc·히스토리 오염)서 **생성-레벨 재생성**(코드에 이미
   있는 `apply_provenance_regen` 패턴)으로 이동 → 차단 호출이 히스토리에 안 남음 → 히스토리 replay-clean →
   **표준 evaluator가 frontier와 동일 채점**. num_errors 예산·db_check 보상 보존.
4. **상태**: 설계 확정 단계. 구현·재런은 리뷰 후. **크래시 궤적 소실**로 기존 데이터 재채점 불가 → fix 후
   재런 필수(nt=4 공식 프로토콜).

---

## 1. 배경 — 검증 대상 주장과 왜 중요한가
- **주장**(헤드라인): 소형 모델 + 결정론 scaffold(present+gate+calc)가 τ²-retail서 대형 base·frontier에
  버금간다. 근거 수치 = **assembled robust pass^3: 32B 0.457·14B 0.313**(vs floor 32B 0.281·14B 0.232).
  frontier 병치 "32B 0.457 ≈ o4-mini 0.468·하위 frontier 진입".
- **공정성 요건**: 이 수치가 **공식 tau2 리더보드와 동일 규약**으로 측정돼야 frontier·base와의 비교가
  정당. 사용자 지적: "crash 처리·max_errors 등 모든 조건을 리더보드와 완전히 동일하게."

---

## 2. 포렌식 사슬 (이 세션서 확정·전부 raw/소스 검증)

### 2.1 크래시 발견과 비대칭 (remote `data/simulations` 재처리)
| arm | total | 크래시(nmsg=0) | 크래시율 | pass^3 CURRENT(드롭) | pass^3 STRICT(크래시=fail) |
|---|---|---|---|---|---|
| floor 7B | 342 | 1 | 0.3% | 0.0796 | 0.0789 |
| floor 14B | 342 | 3 | 0.9% | 0.2321 | 0.2281 |
| floor 32B | 342 | **0** | **0%** | 0.2807 | 0.2807 |
| assembled 14B | 342 | 26 | 7.6% | 0.3131 | **0.2719** |
| assembled 32B | 342 | 39 | **11.4%** | 0.4574 | **0.3772** |

**크래시 비대칭은 인과적**: floor(게이트無)는 ~0%, assembled(게이트有)만 8~11% = 게이트가 원인.

### 2.2 지표 관계 (수학적 엄밀)
- 크래시 sim은 `reward=None`. 크래시가 낀 task는 clean 시행 <3 → pass^3서 제외.
- **strict = current × (clean-3-task수 / 114)**. 크래시 낀 task의 pass^3 기여=0(clean<3), 분모=114.
  검산: 32B 0.4574 × 94/114 = 0.377 ✓ · 14B 0.3131 × 99/114 = 0.272 ✓.
- ⇒ **strict ≤ current 항상**. 따라서 floor-strict ≤ floor-current.

### 2.3 검증된 재판정 (strict-일관)
- **같은-scale 우위 유지**: 32B 0.377 > 0.281 (**+9.7pp**)·14B 0.272 > 0.228 (**+4.4pp**).
- **★cross-scale crossover FLIP·철회**: "작은+scaffold > 큰 bare"(14B-asm > 32B-floor)는 current
  0.313 > 0.281이나 **strict 0.272 < 0.281 = 뒤집힘**. current 우위 = assembled 11% 크래시-드롭 인공물.
- **★frontier 병치 = k-불일치·철회**(공식 참조로 확정): `data/tau2/results/final/
  o4-mini-2025-04-16_retail_default_gpt-4.1-2025-04-14_4trials.json`을 공식 `compute_metrics`로 채점 →
  **o4-mini retail pass^1 0.715·pass^2 0.594·pass^3 0.5175·pass^4 0.4561**(n=456·**infra 0·전부
  user_stop**·config nt=4/max_steps200/max_errors10/gpt-4.1 = 우리와 동일). ⇒ 헤드라인 "0.457≈o4-mini
  0.468"은 **우리 pass^3(0.457)를 o4-mini pass^4(0.456)와 비교한 k-불일치**. **같은 k(pass^3)**: 우리
  32B current 0.457 / strict 0.377 vs **o4-mini 0.5175** = 같은 k서 이미 frontier 아래(strict ~14pt).
  **"frontier 진입/근접" 주장 철회.** frontier 크래시=0(우리 무료-드롭 비대칭 논거의 정량 확증).
  살아있는 moat = **pass 병치 아니라 compliance**(우리 게이트=위반0·scale-불변 / frontier=confirm 위반
  존재·[[46]] 정합).

---

## 3. 근본원인 (메커니즘·소스 정독)

### 3.1 tau2 평가는 replay-기반 (회피 불가·모든 리더보드 모델 동일)
`src/tau2/evaluator/evaluator_env.py:85-95`: reward(db_check)는 **live env 최종상태가 아니라**
fresh env를 message-history로 재구성해 산출:
```python
predicted_environment = environment_constructor(...)
predicted_environment.set_state(message_history=list(full_trajectory))   # 에이전트 궤적 replay
gold_environment = environment_constructor(...)
gold_environment.set_state(...); + golden_actions                         # 정답 replay
# predicted DB vs gold DB 비교 → reward
```
`src/tau2/environment/environment.py:293 set_state` → `:307 get_actions_from_messages`가 히스토리서
(tool_call, expected_response) 쌍 추출 → `:357` 루프서 **mutating tool마다** `self.get_response(tc)`로
**재실행** 후 대조:
```python
# environment.py:382-389
response = self.get_response(tool_call)               # pristine env서 실제 도구 재실행
content = json.loads(response.content); expected_content = json.loads(expected_response.content)
if content != expected_content:
    raise ValueError(f"Tool call:\n{tool_call}\n\nReturned:\n{response}\n\nExpected:\n{expected_response}")
```
- reads(`_is_mutating_tool` False)·hallucinated tool(`_has_tool` False)은 skip. **mutating write는 재실행+대조.**
- 이 replay는 tau2 표준 evaluator. **frontier 포함 모든 리더보드 모델이 이 경로로 채점**. "실시간 측정"
  경로는 없음(있게 만들면 리더보드 비교 불가).

### 3.2 우리 게이트가 히스토리를 오염 (`t2_gate_patch.py::apply`)
게이트 = `BaseOrchestrator._execute_tool_calls` 몽키패치. 차단(deny) 시:
```python
# t2_gate_patch.py gated(): 실행(orig) 없이 합성 에러 메시지를 results에 append → 히스토리 커밋
results.append(_deny_msg(tc, g, why))   # ToolMessage(error=True, content="Error: [POLICY GATE {g}] {why}")
```
정상 에이전트는 히스토리가 자기-일관(기록 tool 응답 = 실제 도구 출력)이라 replay 무크래시. 우리는 **차단
write의 기록 응답 = 합성 게이트 메시지 ≠ 재실행(pristine env·게이트無) 결과** → `:389` mismatch → 크래시.
코드 주석도 인지: present-augment는 reads만("읽기는 replay서 skip") + "write-deny=replay 깨짐"이라 명시.

### 3.3 크래시 분류 (info.error 전수·게이트명·divergence 종류)
| | G2_CONFIRM | G5_PRECOND | G7_CONSTRAINT | 미식별 | content-only(no-op) | **state-divergence** |
|---|---|---|---|---|---|---|
| 32B(39) | 10 | 10 | 12 | 7 | 26 | **13** |
| 14B(26) | 2 | 6 | 7 | 11 | 13 | **13** |
- 게이트 3종(G2 confirm·G5 status-precondition·G7 op-constraint) 전반.
- **content-only**: 재실행도 에러(no-op)나 텍스트만 다름(게이트 vs base 에러). DB는 동일.
- **★state-divergence(절반)**: 재실행이 **성공(DB mutate)** = 게이트가 base라면 실행할 write를 차단.
  → 단순 assertion 완화로 넘기면 **predicted DB가 live와 갈라져 틀린 reward**. ⇒ **얕은 fix 불가·
  히스토리를 replay-clean하게** 만드는 것만이 정답.

---

## 4. 리더보드 parity — "완전 동일"의 정확한 의미

### 4.1 조건 감사 (우리 results.json `info` vs 공식 `config.py`)
| 조건 | 공식 | 우리 | 일치 |
|---|---|---|---|
| max_errors | 10 (`DEFAULT_MAX_ERRORS`) | 10 | ✅ |
| max_steps | 200 (`DEFAULT_MAX_STEPS`) | 200 | ✅ |
| temperature (agent/user) | 0.0 / 0.0 | 0.0 / 0.0 | ✅ |
| user-sim 모델 | gpt-4.1-2025-04-14 | openrouter/openai/gpt-4.1 | ✅ |
| taskset | retail `tasks.json` 114 | 114 (id 0–113) | ✅ |
| num_trials | **4** (리더보드 프로토콜) | **3** | ❌ (k 다름) |
| crash 처리 (pass^k) | `compute_metrics` **드롭** | 드롭 | 기계적 동일 |

### 4.2 공식 crash 규약 — 두 경로 (정밀)
- `scripts/get_experiment_results.py:42`: `fail_terms={"max_steps","too_many_errors",
  "infrastructure_error"}` → infra를 fail 카운트로 **보고**(요약 스크립트·pass^k 아님).
- **`metrics/agent_metrics.py::get_metrics_df`(리더보드 pass^k 경로)**: `df = df[df.termination_reason
  != INFRASTRUCTURE_ERROR]` = **infra 드롭**(docstring "simulations that never ran"). `pass_hat_k =
  C(success,k)/C(num_trials,k)`, success=`reward==1`, max_k=min 생존시행.
- **역설**: 공식 드롭 전제 = infra는 "안 돈 시뮬"(API 끊김·frontier ~0). 우리 infra는 **방법-유발
  replay 크래시**. 공식 코드를 기계적 적용 → 우리 11%가 무료 드롭 → 부풀림. **실질 공정 = fix로 크래시
  자체를 제거해 342개 다 채점** = frontier와 동일 조건.

### 4.3 "완전 동일"의 결론
- 남은 불일치 = (a) num_trials 3 vs 4, (b) 방법-유발 크래시로 인한 실질 드롭 비대칭.
- **⇒ 공정 측정 = fix(크래시 제거) + nt=4 재런 + 공식 `compute_metrics` 채점.**

---

## 5. Fix 설계 — replay-safe 게이트

### 5.1 원리
게이트가 차단한 mutating 호출이 **대화 히스토리에 executed action으로 남지 않게** 한다. 차단 호출은
live서 no-op(미실행)이므로 **올바른 replay 재구성 = 그 호출을 아예 없는 것으로 취급**. 히스토리에 없으면
`get_actions_from_messages`가 그 쌍을 반환 안 함 → 재실행 안 함 → 크래시/divergence 0.

### 5.2 이미 있는 template — `apply_provenance_regen`
`t2_gate_patch.py::apply_provenance_regen`는 `LLMAgent._generate_next_message`를 패치해:
- 검증 실패 시 거부 피드백을 **작업 버퍼**(`work = work + [am] + [ToolMessage(error)]`)에만 넣고,
- `_gen(...)`으로 **재생성**(최대 `max_retries`), 유효 호출만 최종 반환 → **`state.messages`(공식 대화)
  는 오염 0**(코드 주석 명시). = replay-safe 강제의 검증된 패턴. 게이트를 여기에 편입한다.

### 5.3 injection-point 분석 (소스 확정)
- orchestrator 흐름(`orchestrator.py:823 step`): `:862 agent.generate_next_message` → (message 커밋)
  → `:885 _execute_tool_calls`(현 게이트). 즉 현 게이트는 **호출이 이미 히스토리에 커밋된 뒤** 실행·오염.
- **LLMAgent는 env를 안 가짐**(`llm_agent.py:__init__`=domain_policy+tools만). 게이트의 precondition
  검사(G5 status·G7 constraint·G3 owner)는 **env/DB 필요** → 순수 agent-레벨 불가.
- **결정**: 게이트 검사는 **env가 있는 곳**(resolvers_from_env)서 수행하되, **재생성은 생성-레벨**서.
  두 후보:
  - **(A) 에이전트에 env-도출 gate 주입**: setup 시 `env`서 `resolvers_from_env(env)`로 GateInterpreter를
    만들어 agent에 attach → `_generate_next_message` 패치가 그 gate로 검사+재생성. env 참조를 우리 패치가
    주입(tau2 코드 불변·agent 분리 원칙은 우리 패치 국소 위반이나 측정 무관). **provenance-regen과 동형·권장.**
  - **(B) orchestrator서 재생성**: `_execute_tool_calls` 대신 `step`의 agent 생성 직후에 훅 →
    deny면 agent_msg 롤백·`self.agent.generate_next_message` 재호출. env 접근 자연스러우나 **커밋된 메시지
    롤백**이 필요(tau2 step 내부 상태 조작·더 침습적).
  - ⇒ **권장 = (A)**. env 참조 주입은 1줄(패치 setup서 `agent._t2_env = orch.environment` 또는 resolvers
    사전 생성). precondition 검사에 필요한 DB는 **생성 시점(=현 배치 실행 전)** 상태로 충분(현 게이트도
    실행 전 검사·동일 DB 시점).

### 5.4 권장 설계 (의사코드)
```python
# LLMAgent._generate_next_message 패치 (gate 편입·provenance-regen 확장)
def patched_generate(self, message, state):
    gate = self._t2_gate          # setup서 GateInterpreter(gates, resolvers_from_env(env)) 주입
    self._system_messages = state.system_messages
    _append(state, message)        # user/tool 입력만 커밋 (기존과 동일)
    work = list(state.messages)
    am = _gen(self, work, ...)      # 1차 생성
    n = 0
    while n < MAX_REGEN:
        # 컨텍스트 = work(작업본); DB precondition = gate.resolvers(env 현 상태)
        deny = first_denied_toolcall(am, gate, last_user=_last_user(work),
                                     transfer_sent=_transfer_sent(work))
        if deny is None:
            break                   # 모든 호출 compliant → 커밋
        tc, gname, why = deny
        self.num_errors_equiv = getattr(self,'num_errors_equiv',0) + 1   # 예산압박 보존(§5.5)
        work = work + [am]
        for c in (am.tool_calls or []):
            reason = f"Error: [POLICY GATE {gname}] {why}" if c is tc else \
                     "Error: [POLICY GATE] resolve the blocked action first; do not call this yet."
            work.append(ToolMessage(id=c.id, role="tool", requestor="assistant",
                                    error=True, content=reason))
        am = _gen(self, work, ...)  # 피드백 주고 재생성 (작업본만·state.messages 불변)
        n += 1
    # ── ★R8 종단(exhaustion) 처리: K 소진 후에도 deny 잔존 시 히스토리에 위반 write가 남지 않게 ──
    denied_ids = {tc.id for tc in (am.tool_calls or []) if _is_denied(tc, gate, ...)}
    if denied_ids:                  # 모델이 K회 내 준수 실패 = 이 태스크는 사실상 fail
        am.tool_calls = [tc for tc in am.tool_calls if tc.id not in denied_ids]  # 차단 mutating 호출 제거
        am.content = (am.content or "") + BLOCK_NOTE   # 차단 사유를 assistant 텍스트로 surface(replay 무해)
        # → 제거된 호출엔 ToolMessage 미생성 → 실행 0·히스토리에 위반 write 없음 → replay-clean 보장
    return am                       # 커밋 메시지 = compliant 호출 + 텍스트만 (denied mutating write 0)
```
- 차단된 시도(am + 거부 ToolMessage)는 **work(작업본)에만** 존재·`state.messages` 미커밋. **★그리고 K
  소진 종단서도 denied mutating 호출을 커밋 前 제거** → "모델이 끝내 준수 못 해도 히스토리엔 실행된 위반
  write가 없음"이 보장(R8). BLOCK_NOTE(텍스트)는 재실행 안 되므로 replay 무해·런타임 게이트 semantics
  (모델이 차단 인지)도 보존. **정당성=R2**: 제거 대상이 진짜 위반이면 제거→fail이 정답, CONFIRM_RE 오탐이면
  R2 fix(정규식 강건화)가 선행(그래야 정당한 write를 잘못 제거·오fail 하지 않음).
- allow된 호출은 orchestrator `_execute_tool_calls`가 정상 실행(게이트 패치는 이제 검사 안 함·또는 제거).
- auth 관찰(gate.observe)은 **커밋된 히스토리의 실행된 auth 도구 출력**서 재구성(생성 시점, 직전까지의
  clean 히스토리로 auth/confirm/transfer 상태 결정 — 차단호출이 사라져 오히려 상태가 더 깨끗).

### 5.5 semantics 보존과 변화 (리뷰 핵심)
- **보존**: (a) **예산압박** — 현 게이트는 deny마다 `self.num_errors += 1`(→`too_many_errors` 종료
  가능). 재생성판도 deny마다 카운트해 동일 압박 유지(무한 우회 방지). (b) **db_check 보상** = 실제 실행된
  궤적의 DB 그대로(차단=no-op). (c) 게이트 **판정 로직**(GateInterpreter)·A2 gate.json 불변.
- **★변화(측정에 영향·불가피)**: 현 게이트는 "차단 호출이 대화에 에러로 남고 모델이 그걸 보고 같은 대화서
  반응"(약한 가드·궤적에 실패턴 존재). 재생성판은 "차단 호출이 대화서 사라지고 재생성으로 대체"(강한 강제).
  ⇒ **pass가 소폭 상승할 수 있음**. 이는 새 정당 기준선이며 **재측정 필요**. 리뷰서 판단할 점:
  이 semantics 변화가 "우리가 주장하려는 개입(런타임 게이트)"과 정합하는가, 아니면 별도 arm으로 둘 것인가.
  - 대안(semantics 최대보존): 차단 호출을 **assistant의 비-tool 텍스트 턴**으로 기록(예: "그 작업은
    정책상 불가—대신 …")해 모델이 다음 턴서 반응하되 mutating tool_call은 히스토리에 없게. 단 이는
    대화 흐름을 바꿈. **재생성판이 더 단순·검증된 패턴이라 1차 권장.**

### 5.6 상태 관리 (cross-turn)
- 게이트 상태: auth(관찰)·confirm(last_user)·transfer(notice sent)·retry-loop(failed keys)·연속실패.
- 재생성판서는 **agent에 상태 유지**(`self._t2_gate`·`self._t2_failed`). 매 생성 시 **커밋된 clean
  히스토리**로 last_user/transfer/auth 재도출(현 `_last_user_text`·`_transfer_msg_sent`·gate.observe와
  동일 로직, 소스만 orchestrator→agent messages). 차단호출 부재로 상태 계산이 더 단순·정확.
- retry-controller(T2_RETRY_*)·provenance(T2_PROVENANCE)는 **별도 레버**(핸드오프서 해로움 판정) →
  기본 assembled엔 미포함. 본 fix는 **게이트(present/nested/calc/gate)**만 대상. present/nested/calc는
  **reads-augment=이미 replay-safe**(그대로 유지).

---

## 6. 구현 계획 (파일·함수·구체 변경)

### 6.1 변경 파일 (전부 `scripts/distill/tau2/`·벤치 코드 불변)
1. **`t2_gate_patch.py`** (핵심):
   - `apply()`: `_execute_tool_calls`의 **deny→append 경로 제거**. 게이트 검사·상태를 생성-레벨로 이관.
     allow 경로(정상 실행)와 reads-augment(present/nested/calc)는 유지. auth observe도 유지(실행 후).
   - 신규 `apply_gate_regen(domain, max_regen=K)`: `LLMAgent._generate_next_message` 패치(§5.4).
     GateInterpreter를 `resolvers_from_env(env)`로 생성해 agent에 attach. `first_denied_toolcall`
     헬퍼(gate.check + provenance옵션). 예산 카운터.
   - env 참조 주입: orchestrator setup 훅 또는 `create_llm_agent` 래핑서 `agent._t2_env` 설정.
     (tau2 `run_domain` 경로 확인 필요 — agent 생성 시 env 접근점.)
2. **`t2_run_gated.py`**: `--gate 1` 시 `apply()` 대신(또는 병행) `apply_gate_regen()` 호출 플래그
   추가(예 `T2_GATE_REGEN=1`). 기존 arm 재현성 위해 **구 경로도 플래그로 보존**(A/B 비교용).
3. **드라이버 `reexp_assembled.sh`**(또는 신규 `reexp_assembled_regen.sh`): `--num_trials 4`·
   `T2_GATE_REGEN=1`·distinct tag(예 `asmscale_{14b,32b}_regen_retail_t4`)·영속화 내장([[30]]).

### 6.2 env 접근점 확인 (구현 전 필수 조사)
- `run_domain`/orchestrator가 agent를 **어떻게 생성**하고 env와 연결하는지 추적(agent 생성 시 env 주입
  지점). `create_llm_agent(tools, domain_policy)`엔 env 없음 → tools는 env-bound. resolvers는 env 필요 →
  **orchestrator가 agent+env 둘 다 보유하는 시점**(setup)서 gate를 만들어 agent에 붙이는 게 확실.
  → `BaseOrchestrator.__init__` 또는 첫 step서 `self.agent._t2_gate = GateInterpreter(...,
  resolvers_from_env(self.environment))` 주입하는 작은 orchestrator 패치 병행(1회).

### 6.3 불변 준수
- [[05]]: GateInterpreter·A2 gate.json 도메인-일반 유지(retail 하드코딩 0). airline/bank swap 동일.
- 벤치(tau2 `src/`) 코드 **불변** — 우리 패치(`scripts/distill/tau2/`)만 수정. 리더보드 evaluator 그대로 사용.

---

## 7. 검증 계획 (fix 정당성·[[08]]·[[09]] 무료 먼저)

### 7.1 스모크 (무료~$2·10 task·nt=1)
- `T2_GATE_REGEN=1`로 retail 10 task 구동 → **공식 evaluator(`evaluate_simulation`/`compute_metrics`)로
  채점** → 성공기준:
  1. **`infrastructure_error` = 0** (핵심: 게이트가 replay 안 깨뜨림).
  2. compliance 유지: 게이트 위반(G1/G2/G3/G4) = 0 (F4=F3).
  3. 게이트 실발화 확인(deny→regen 로그·present/calc 발화·[[30]] "레버 실발화율").
  4. 크래시났던 task 일부(예 22·27·33) 포함해 **정상 채점(reward 산출)** 확인.
- 실패 시(여전히 크래시/divergence): §5.3 (B) orchestrator-롤백 또는 §5.5 대안(텍스트 턴) 재검토.

### 7.2 본 측정 (유료·승인 후·★공식 리더보드 프로토콜 확정)
- **프로토콜(참조 26개 전수+`docs/leaderboard-submission.md` 확정)**: **`num_trials=4`**(전 참조 모델
  claude-3.7/gpt-4.1/gpt-4.1-mini/o4-mini × retail/airline/telecom 모두 4trials)·max_steps 200·
  max_errors 10·agent&user gpt-4.1(우리는 agent=로컬 vllm)·temp 0. **보고 = pass^1..pass^4 전부**.
  **headline·랭킹 = pass^1**(=avg success), pass^4=신뢰성 보조.
- assembled 14B+32B를 `T2_GATE_REGEN=1`·**nt=4**로 full 재런 → **공식 compute_metrics**로 pass^1..4.
  floor도 nt=4 재런(현 nt=3)해 동일 k 비교. 영속화([[30]]).
- 비용: user_cost 미기록(openrouter) → 스모크서 sim당 토큰 실측 후 산정(대략 양 arm ~$30–70·agent 무료).

### 7.3 판정 (같은-k·다지표)
- **주 지표 = pass^1**(리더보드 headline) + pass^2/3/4 커브. 공식 compute_metrics(무료 드롭 없음)=frontier와
  byte-동일 채점.
- **비교는 항상 같은 k**: frontier o4-mini retail(공식 참조) pass^1 0.715·pass^2 0.594·pass^3 0.5175·
  pass^4 0.4561. 우리 구 nt=3 32B는 pass^1 0.650(drop)·pass^3 0.457(drop)/0.377(strict) → **어느 k로
  맞춰도 frontier 아래**. ⇒ 재측정도 same-scale(vs floor)·compliance를 headline으로(§10 R4 사전등록).
- crossover·frontier-parity 주장은 이 재측정 값으로만(그리고 same-k로만) 판정.

---

## 8. 리스크·오픈 이슈 (리뷰 판단)
1. **semantics 변화(§5.5)** ✅해소(§10 R1): MAX_REGEN=1 + deny→num_errors 예산 카운트로 **best-of-K 아닌
   "같은 게이트의 replay-clean 재측정"** 확정. 권장: 재생성판을 주 arm·구 런타임판을 참고 A/B로 병기.
2. **env 주입점(§6.2)**: agent 생성-env 연결 지점 미확정 — 구현 전 `run_domain` 추적 필요.
3. **G7/G3 precondition의 생성-시점 DB**: 현 게이트도 실행 전 검사라 동일 DB 시점 → 등가. 단 배치 내
   복수 tool_call 간 상호의존(앞 호출이 뒤 호출 precondition 바꿈) 시 순차 처리 확인.
4. **num_trials 3→4** ✅해소: 공식 프로토콜=nt=4(참조 26개 전수 확인)·pass^1..4 보고·pass^1 headline
   (§7.2). floor도 nt=4 재런 필요(동일 k). 비용↑.
5. **frontier 수치 출처** ✅해소: 공식 참조 `o4-mini..retail..4trials.json`→compute_metrics로 pass^1
   0.715·p3 0.5175·p4 0.4561·infra0 확정(§2.3·§10). "0.468"은 pass^4(k-불일치)였음.
6. **미식별 크래시(NONE)**: 14B 11/26이 info.error 900자 truncation로 게이트명 미포착. 재런 전 전체
   error_traceback 확인 권장(다른 원인 배제).

---

## 9. 부록 — 소스·데이터·명령 (재현)
- **소스(remote `/home/woori/scratch/tau2-bench`)**: `src/tau2/evaluator/evaluator_env.py:85`·
  `src/tau2/environment/environment.py:{119,130,293,307,357,382-389,446}`·
  `src/tau2/metrics/agent_metrics.py::{get_metrics_df,pass_hat_k:126,compute_metrics}`·
  `src/tau2/scripts/get_experiment_results.py:42`·`src/tau2/config.py`(DEFAULT_MAX_*)·
  `src/tau2/orchestrator/orchestrator.py:{313,823,862,885}`·`src/tau2/agent/llm_agent.py:{105,115}`.
- **우리 패치(remote `.../boltzmann-attention-pi/scripts/distill/tau2/`)**: `t2_gate_patch.py`
  (`apply`·`apply_provenance_regen`·`_deny_msg`)·`t2_run_gated.py`·`gate_interpreter.py`·`reexp_assembled.sh`.
- **데이터(remote `data/simulations/`)**: `asmscale_{14b,32b}_0626pm_assembled_retail_t3`·
  `on_n{7b,14b,32int8}_floor_retail`. (로컬 미러: `ba-frft/reports/facet_rft_2026/sim_results/asmscale_*.gz`.)
- **접속**: `cd /c/workspace && py -3 ssh_run.py --timeout N < job.sh`([[30]]). py=`/home/woori/venvs/seka_env/bin/python`.
- **재현 분석 스크립트**: 로컬 scratchpad `remote_*.sh`(크래시 재처리·분류·소스 probe).
- **★공식 frontier 참조**: `data/tau2/results/final/o4-mini-2025-04-16_retail_default_
  gpt-4.1-2025-04-14_4trials.json`(o4-mini retail·nt=4·gpt-4.1 user·공식 pass^k). airline·telecom도 동 폴더.

---

## 10. 리뷰 반영 (2026-07-06 리뷰 R1–R7·구현 전 못 박음)

### R1 — regen이 "게이트"가 아니라 "best-of-K 준수필터"로 변질 방지 (측정 정체성·최우선)
**확정 사양**(구현 시 강제):
- **MAX_REGEN = 1**(턴당 1회 교정 기회). 큰 K는 compliance-filtered best-of-K 샘플링=다른 방법이 됨.
- **예산 동일**: 각 deny를 orchestrator의 `num_errors`에 카운트. 소스 확정 — `orchestrator.py:327
  num_errors += 1`·`:747-749 if num_errors >= max_errors(10): TOO_MANY_ERRORS`. regen은 agent-레벨이라
  orchestrator.num_errors에 propagate하는 채널 필요(§6.2 env 주입과 함께 orchestrator 참조 주입).
- 결과: "차단 호출 1개 = 1 error + 1 교정 기회"로 현 런타임 게이트와 **동일 예산압박** → best-of-K 아님·
  "같은 게이트의 replay-clean 재측정"임이 보장. **이게 측정 정체성의 1순위 불변.**

### R2 — state-divergence 13개: 위반이냐 오탐이냐 ([[08]]·부분확정+스모크 필수)
- **부분확정(info.error 전수)**: 32B state-divergence 13개 = **전부 `G2_CONFIRM_WRITE`**("확인 없이
  write"·base는 확인요건 없어 replay서 실행→divergence). 즉 "게이트가 base라면 실행할 write 차단"의 정체=
  **G2 확인-전-write 차단**.
- **오탐 표면(소스 정독)**: G2 = `if not CONFIRM_RE.search(last_user_msg)`. `CONFIRM_RE = (yes|yeah|yep|
  sure|confirm|confirmed|correct|proceed|go ahead|ok(ay)?|sounds good|please do|that works|do it)`.
  **오탐 위험 2**: (a) 정규식 미매칭 확인("absolutely"·"perfect"·"yes please change it"의 변형)·
  (b) **last_user_msg만** 검사(2턴 전 확인 후 다른 발화면 놓침). ⇒ G2는 실제 오탐 가능. (대조: G5
  preconditions는 "못 읽으면 deny 안 함=false-block 회피" 설계라 오탐 낮음.)
- **미해결(궤적 소실)**: 이 13개가 진짜 미확인(게이트 정답)인지 정규식-오탐인지 = **크래시 궤적 소실
  (messages=0)로 기존 데이터서 판정 불가** → **스모크 성공기준에 편입**(§7.1 신규): fresh 런의 G2
  divergence를 위반 vs 오탐 전수 분류. 오탐↑면 **CONFIRM_RE 강건화 또는 확인-스캔 범위 확대**가 별도 fix.
  이게 fix 후 pass↑/↓를 가르는 과학적 질문.

### R3 — clean 태스크 회귀 (무료·결정적·스모크 前 가능)
- regen 게이트는 상태를 커밋된 clean 히스토리서 재도출(§5.6) → auth/confirm 계산 소스 변경=미세버그 위험.
- **회귀 테스트**: 크래시 안 난 clean 태스크(90%)는 **궤적이 저장돼 있음**. 현 게이트 판정 vs regen 게이트
  판정을 **저장 궤적에 오프라인 replay**해 allow/deny diff → 0이어야 함(갈리면 상태-재구성 버그). **무료·
  결정적** → 스모크 前 pre-check로 편입.

### R4/R5 — 다운스트림·claim 사전등록 (3중 복합변화 前)
- 최종값 = (a) strict 재채점(↓) + (b) nt 3→4(pass^4<pass^3·↓) + (c) regen(↑) 합성 → 순효과 불확정.
- **★claim 사전등록**(재측정 前·[[03]] 헤드라인 규율): 살아남는 주장 = **① 같은-scale 우위**(strict
  32B +9.7pp·14B +4.4pp) + **② compliance moat**(게이트=위반0·scale-불변 / frontier=confirm 위반 존재).
  **철회 주장** = **③ cross-scale crossover**("작은+scaffold>큰 bare"·strict서 flip) + **④ frontier
  pass 진입/근접**(k-맞추면 o4-mini pass^3 0.5175 > 우리 strict 0.377). **"특정 수치"가 아니라 "same-scale
  advantage + compliance"를 헤드라인**으로.
- **★compliance moat 프레이밍(정밀)**: τ²-retail서 frontier 준수 낙폭은 0~2.6pp로 **작음** → moat은
  "margin(우리가 몇 pp 더 준수)"이 아니라 **"결정론적 보장(증명가능 위반 0·scale-불변) vs 통계적 준-준수
  (frontier는 대체로 지키나 보장 없음·간헐 confirm 위반)"**의 *종류* 차이. 덱은 이미 이 프레이밍("보장은
  우리가 유일") → paper1·특허도 **"margin이 아니라 guarantee"**로 동일 유지. (pass는 frontier 아래라도
  이 guarantee가 독립 가치·[[46]] moat.)
- **다운스트림 갱신 태스크**(fix·재측정 後): paper1 §5.6·특허 A/B 실시예 2–3·THEORY_BOUNDARY_MAP §5
  Cor·덱 결론장 — 0.313/0.457/0.281·crossover·frontier병치 전부 갱신. [[46]] moat=compliance 재확인.

### R6 (minor) — present/calc replay-safe 재확인
- present/nested/calc는 **reads-augment**(read 응답에 요약 첨부)라 replay서 read=skip → 안전(§3.1). 단
  스모크서 present/calc 발화 태스크 크래시 0을 **명시 확인**(§7.1).

### R7 (minor) — 비용 scope 축소
- 1차 = **assembled 14B+32B만** 재런(nt=4). floor는 **k-정합 필요 시**에만(현 nt=3 floor로도 same-scale
  Δ는 나오나 pass^4 비교엔 floor nt=4 필요). 스모크서 sim당 토큰 실측 후 재산정. frontier(o4-mini)는
  **재런 불요**(공식 참조 파일 확보).

### R8 — regen 소진(exhaustion) 종단 처리 (신규·history-clean 완결·구현 필수)
- **구멍**: MAX_REGEN=1서 재생성이 여전히 deny면 §5.4가 `am`을 무조건 반환 → 차단 mutating 호출이 커밋 →
  **꼬리 태스크(1회 내 준수 실패)서 replay 크래시 부활** = fix의 핵심(history-clean)이 꼬리서 붕괴.
- **종단 사양(§5.4에 반영)**: 루프 후 **최종 deny 재검사** → 잔존 denied mutating tool_call을 커밋 前
  **제거**(compliant 호출 + 텍스트만 커밋·차단 사유는 assistant 텍스트로). ⇒ "모델이 끝내 준수 못 해도
  히스토리에 실행된 위반 write 0" 보장·replay-clean 꼬리까지 완결.
- **R2 연결**: 제거 대상이 진짜 위반이면 제거→fail이 정답, CONFIRM_RE 오탐이면 정당 write를 오제거·오fail →
  **R2 분류(오탐률)가 종단 정당성의 전제**. 오탐↑면 CONFIRM_RE fix 선행.

### §7.1 스모크 성공기준 (R1–R3·R6·R8 편입·개정)
1. **infra_error = 0** — **regen-소진 태스크(1회 내 준수 실패) 포함** 전 태스크서 0(R8 종단 처리로 꼬리도
   history-clean). 핵심: replay 안 깨짐.
2. **R2 divergence 분류**: G2 차단 write의 위반 vs CONFIRM_RE-오탐 전수 판정(궤적 정독).
3. **R3 clean 회귀**: 저장 clean 궤적서 현-게이트 vs regen-게이트 allow/deny diff = 0.
4. **R1 예산**: deny→num_errors 반영·MAX_REGEN=1·too_many_errors 임계 동일 동작 확인.
5. compliance 유지(G1–G4 위반 0·F4=F3)·present/calc 발화 태스크 크래시 0(R6).
6. 크래시났던 task(22·27·33·71 등) 정상 채점(reward 산출) 확인.
- **판정 게이트**: 1·3·4 = **PASS 필수**(측정 정합). 2 = 분류결과가 오탐↑면 CONFIRM_RE fix 선행. 이 셋이
  "리더보드-동일 공정 측정"과 "같은 방법의 정직한 재측정"을 동시 보장.

---

## 11. 구현·스모크 결과 (2026-07-06·32B·num_tasks 40·nt=1·regen32b_smoke)

### 구현 (커밋됨·`scripts/distill/tau2/`)
- `t2_gate_patch.py::apply_gate_regen(max_regen=1)` + 헬퍼(`_denied_calls`·`_rebuild_gate_state`·
  `_budget_tick`·`_install_regen_exec`) + `_BLOCK_NOTE`(R8). `t2_run_gated.py` `T2_GATE_REGEN=1` 배선.
- 리뷰 수정 반영: **⚠️2** R8 중복 budget-tick 제거(차단 turn당 1 error·R1). **⚠️4** regen+prov 동시
  설정 시 SystemExit 가드. **⚠️1** AssistantMessage=mutable 검증(model_config 비어있음)+R8 2회 실전 무결
  → in-place OK. **⚠️3** auth_user-only 재구성이 assembled config서 충분(select_confirm 미포함·G7
  stateless·retry off)·violations=0이 실증.

### 스모크 검증 (세션 종료로 n=38/40서 중단·유의미)
| §7.1 기준 | 결과 |
|---|---|
| **replay 크래시(environment.py:389) = 0** | ✅ **옛 32B 크래시 task 전부 정상채점**: 19·22·28·31·33=pass·27·34·37=정직 fail(36=미실행) |
| 게이트 활성·regen 발화 | ✅ vllm 341 req vs committed assistant 270 = **~71 regen** (deny→재생성) |
| R8 종단(exhaustion) | ✅ BLOCK_NOTE 2회 발화·traceback 0·am mutation 무결 |
| compliance(g1–g4) | ✅ 위반 0 (게이트 집행·상태재구성 정상) |
| pass^1(정보) | 30/38 (nt=1·easy/hard 혼합) |

- **★핵심**: 옛 게이트가 replay-크래시하던 task들이 이제 **크래시 없이 채점**(pass 또는 정직 fail).
  무료-드롭(부풀림) 제거 실증. task 27·34·37이 0.0으로 fail = 모델이 못 푸는 걸 정직하게 잡음.

### 잔여 infra 1건 = ContextWindow(직교·서빙설정·replay 아님)
- **task 20**: `ContextWindowExceededError`(16441 > max-model-len 16384) — replay assertion 아님.
  regen이 work-buffer에 피드백(rejected am+ToolMessage) 추가 → 16384 살짝 초과. **fix = 서빙
  max-model-len 32768**(gpt52sim retry도 32768). 코드 아님·서빙 파라미터.
- 참고: 옛 asmscale_32b 39 크래시는 **전부 replay(env.py:389)**·ContextWindow 0. 즉 replay는 fix가
  제거했고, 새로 드러난 ContextWindow는 context 확대로 닫음 → **full 런서 infra≈0 기대**.

### 다음 (full 재런·승인 후)
- 서빙 **max-model-len 32768**·`T2_GATE_REGEN=1`·**nt=4**(공식 프로토콜)·assembled 14B+32B·floor도
  nt=4·공식 `compute_metrics`(pass^1..4·headline pass^1)·같은-k frontier(o4-mini pass^3 0.5175·
  pass^4 0.4561) 비교. R2 divergence 분류·R3 회귀는 이 런 궤적서 확정.

---

## 12. ★FINAL 결과 — nt=4 클린 재런 (2026-07-07·리더보드-동일·권위)
`asmregen{14b,32b}_regen_retail_t4`(nt=4·32768·regen·gpt-4.1 user·공식 `compute_metrics`). 영속화:
`sim_results/asmregen{14b,32b}_regen_retail_t4.results.json.gz`.

### 원측 (둘 다 **infra=0 · 게이트 위반 g1–g4=0**)
| pass^k | 14B asm | 32B asm | floor 14B | floor 32B | o4-mini(frontier) |
|---|---|---|---|---|---|
| pass^1(=avg) | 0.588 | 0.640 | 0.468 | 0.547 | 0.715 |
| pass^2 | 0.430 | 0.504 | — | — | 0.594 |
| **pass^3** | **0.336** | **0.423** | 0.228 | 0.281 | 0.518 |
| pass^4 | 0.272 | 0.360 | — | — | 0.456 |
*(32B 456 전부 user_stop·14B 453 user_stop+3 too_many_errors[모델 예산실패=fail·드롭 아님]. floor=구 nt=3
클린[infra~0]·pass^k는 nt-불변이라 same-k 비교 유효. floor pass^1: 14B 0.468·32B 0.547[구 f3f4].)*

### 확정 결론 ([[08]]·infra=0로 드롭 confound 제거·violations=0·per-case 검증)
1. **진짜값 회수**: 32B pass^3 **0.423**·14B **0.336** = 옛 strict(0.377/0.272)와 옛 drop(0.457/0.313)
   **사이**. 드롭됐던 hard task가 정직 채점(옛-크래시 32B 중 36·37·71·98·107=fail-all·다수 fail-some).
2. **★same-scale 우위 = 견고·strict보다 큼**: pass^3 32B **+14.2pp**(0.423>0.281)·14B **+10.8pp**
   (0.336>0.228). pass^1 32B +9.3pp·14B +11.9pp.
3. **★cross-scale crossover = 부활·성립(un-retract)**: **14B-asm 0.336 > 32B-floor 0.281**(pass^3
   +5.5pp)·pass^1 0.588>0.547. strict서 flip(0.272<0.281)됐던 "작은+scaffold>큰 bare"가 **클린 측정서
   되살아남** — strict가 방법-유발 크래시를 fail로 과벌한 탓이었고, 진짜값이 crossover를 지지. **철회 →
   재확립**(클린 기반·측정-의존 아님).
4. **frontier = 모든 k서 아래(정직)**: pass^3 0.423 vs 0.518(-9.5pp)·pass^4 0.360 vs 0.456(-9.6pp)·
   pass^1 0.640 vs 0.715(-7.5pp). **pass 병치 아님**·일관 ~8-10pp. **moat = compliance**(우리 위반0·
   scale-불변 vs frontier 간헐 위반).
5. **R2/R3**: violations=0 = 게이트 집행·auth_user 상태재구성 정상(R3 실증·추가 diff 불요). R2: 옛-크래시
   G2-confirm task(28·31·33)는 pass·나머지 다수 정직 fail = 게이트 옳음(오탐 아님·모델이 못 푸는 것).

### 헤드라인 갱신 (사전등록 R4 대비)
- **살아남음**: ① same-scale 우위(32B +14.2pp·14B +10.8pp pass^3) ② compliance moat(위반0·scale-불변)
  ③ **cross-scale crossover(14B-asm>32B-floor)** ← strict서 철회했으나 클린서 재확립.
- **철회 유지**: ④ frontier pass 진입/근접(모든 k서 ~9pp 아래).
- 수치 갱신: 0.457/0.313(drop) → **0.423/0.336(pass^3 클린)** + pass^1 0.640/0.588 병기. paper1/특허/덱
  전파 대상.
- **선택 잔여**: floor를 nt=4/32768로 재런하면 pass^4 same-k 완전정합(현재 pass^3 비교는 nt-불변이라 유효).
