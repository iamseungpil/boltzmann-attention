# FIXED vs VARIABLE — 무엇을 고정하고 무엇만 바꾸는가 (단일 권위·2026-06-18·재정의 금지)

> 계속된 표류의 뿌리 = 이 경계가 단일 권위로 박제된 적 없어서. **이 문서가 그 단일 권위.** 모든 설계문서(EXPERIMENT_DESIGN·INTEGRATED_TBOX·ISOLATION_EXPERIMENTS·CROSS_BENCH_TRANSFER_PLAN)는 이걸 *참조*하고 재정의하지 않는다. 메모리 `05-fixed-vs-variable`이 요약.
> 근거 = `NL_PROCEDURE_OFFLOAD_THEORY §6-7`·`DECOMPOSITION_OPTIMALITY §A`·`CROSS_BENCH_TRANSFER_PLAN §3`.

## 0. 이론적 근거 (𝔤 / G / 구체 삼분)
모든 연산 O = 유한차원 **대수 𝔤**(생성원·추상화·도메인-불변·저차원) + 그것이 exp으로 생성하는 **군 G**(궤도·실행·무한). (이론 §1)
- LLM은 **𝔤(생성원 *명명*)에 강하고** 군-실행(궤도 적분)에 약하다 → 𝔤 = 학습·전이.
- 결정론은 **군 G를 실행**(exp을 돌린다·d-불변) → 엔진 = 고정.
- 구체(이 카탈로그·이 값·이 정책) = 군의 *구체 좌표* → ABox = 변경.

## 1. ★고정 (FIXED) — 도메인·벤치 무관·절대 미수정
| | 정체 | 이론 | 무엇을 담나 |
|---|---|---|---|
| **TBox (LLM weights)** | 𝔤-식별 = **NL→formalize**(생성원 명명·4 facet) + 환원불가 추론 | §6 "학습 대상=𝔤-식별·도메인불변·전이가능" | flow-타입/순서·threading·content op-명명·operand formalize (전부 *intensional*) |
| **Scaffold (결정론 엔진 코드)** | 군-실행 = gate 집행·resolve·per-step verify·step-orchestrator | §7 "결정론은 군 실행·엔진 d-불변·직렬깊이 흡수" | GATE_SPEC 해석기·provenance 검증기·resolve 엔진·typed-step 루프 |

- TBox = {SOPBench+TaskBench+CFB+Synth} **1회 학습 후 FROZEN**.
- Scaffold = **ONE 구현·per-bench 분기 0**(grep `if bench`/`if domain` = 0)·절대 per-bench 미수정.

## 2. ★변경 (VARIABLE) — swap되는 *유일한* 것 (데이터지 코드 아님)
| | 정체 | 구성 |
|---|---|---|
| **ABox** | 구체(이 카탈로그·값·정책) | A1(도구 카탈로그·기계)·**A2(정책 NL→GATE_SPEC·유일 난제)**·A5(출력 문법)·vocabulary |

## 3. ★전이의 정의
**전이 ≡ ABox만 swap, TBox·Scaffold는 unchanged (재학습0·코드수정0).**
TBox나 Scaffold를 per-bench 수정 = **bench-베이킹 = thesis 실패** (="login 특별취급 금지"·per-domain 분기와 동일 죄).

## 4. ★현 위반 (= 표류의 정체·고칠 것)
**`tau2/t2_gate.py`의 `RetailGate`가 retail을 코드에 하드코딩 → 군-실행 엔진에 *구체(ABox)*를 박은 위반:**
```python
# ❌ 위반 (현재) — retail 도구·정책이 엔진 코드에 박힘
AUTH_TOOLS  = {"find_user_id_by_name_zip", ...}            # retail 구체
WRITE_TOOLS = {"cancel_pending_order", "exchange_...", ...} # retail 구체
class RetailGate:
    def check(self, tool_name, ...):
        if tool_name in WRITE_TOOLS and not confirmed: deny()   # 집행은 일반이나 멤버십이 코드에
# + SOPBench는 또 다른 별도 gate = 또 위반
```
```python
# ✅ thesis-정합 — 엔진 1개·구체는 ABox(gate_spec)서 읽음
class GateInterpreter:                  # FIXED·벤치 무관·절대 안 고침
    def check(self, call, gate_spec, state):
        if call.name in gate_spec["auth_tools"]  and not state.authed:     deny(...)
        if call.name in gate_spec["write_tools"] and not state.confirmed:  deny(...)
# ABox (VARIABLE·데이터): retail_spec/airline_spec/sop_<domain>_spec = {"auth_tools":[...],"write_tools":[...]}
```
→ `RetailGate`·SOP gate를 **하나의 `GateInterpreter(gate_spec)`로 통일**·도구멤버십/정책은 ABox서 읽고 집행로직만 고정.

## 5. ★우선순위 정정
**scaffold 통일이 처음부터 핵심 keystone** — facet TBox 전이(②)보다 *선행*. scaffold가 bench-baked면 "fix scaffold, swap ABox"라는 전이 주장 *자체가 성립 안 함*. (`ISOLATION_EXPERIMENTS §S` = 이 통일·조건 ③.)

## 6. 정렬 체크리스트 (전 설계문서)
- [ ] TBox=학습·고정 / Scaffold=결정론·고정 / ABox=유일 변경 — 이 경계로 기술.
- [ ] "scaffold 전이"라는 표현 금지(scaffold는 *전이*하는 게 아니라 *불변*·ABox가 전이).
- [ ] per-bench 분기 0 (grep `if bench`) — 위반 시 표기·수정.
- [ ] facet 전이 실험은 *고정 scaffold + swap ABox* 전제 하에서만 의미.
