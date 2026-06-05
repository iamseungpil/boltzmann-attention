# Fix-3 설계서 — STOP-after-success (goal-call looping 차단)

> **상태: DRAFT — 리뷰 대기. 리뷰 통과 후 구현·A/B.** 마스터 = `EXPERIMENT_DESIGN.md`. 진단 근거 = [`RESIDUAL_PREMATURE_DIAGNOSIS_2026_06_05.md`](RESIDUAL_PREMATURE_DIAGNOSIS_2026_06_05.md) §10, [`LEADERBOARD_METRIC_GROUNDING_2026_06_05.md`](LEADERBOARD_METRIC_GROUNDING_2026_06_05.md).
> 메타규칙: 강한 주장은 reliable test 후 박제 · GPU 전 zero-cost 사전검증 · dead-end 변종 금지 · 회귀·should_F 코드강제. flag = `SOPBENCH_STOPSUCCESS` (off by default).

---

## §0. 요약
- **문제**: 공식 `success`(리더보드 지표)에서, BOTH(dg∧acc)는 통과하나 **cnv/dbm/ntce가 실패**하는 12 태스크 전부 **goal 액션을 5–9회 반복 호출**(step cap까지). 모델이 첫 성공 후 **종료(STOP)하지 않고** goal을 재호출 → DB 오염·제약위반·에러.
- **처방**: gate가 **goal이 한 번 성공적으로 호출된 직후 STOP**(→ `exit_conversation`)을 반환해 루프를 끊는다.
- **기대**: should_T full_success **+최대 12** (현 logincall 28 → 최대 40). 공식 pass rate(134) 40.30% → 최대 ~49%. (실측은 A/B로 확정; 12 전부 flip 보장 아님.)

## §1. 문제 (진단 근거, 실측)
- `diag_loop_check.py` (L1C=logincall run): **BOTH-but-not-full 12개 전부** goal_calls 5–9, total_calls 10(step cap). 예:
  - get_loan ×4: goal 6–7회, cnv=False dbm=False (대출 6–7번 적용 → DB 오염).
  - pay_bill ×2 / pay_loan ×2 / transfer ×1: goal 5–7회, cnv=False dbm=False.
  - close_account / set_account_information: goal 8회, cnv=True dbm=True지만 **ntce=False**(닫힌/변경된 상태 재호출 → tool error).
- `evaluator.py`: `success = no_tool_call_error ∧ constraint_not_violated ∧ database_match ∧ action_called_correctly ∧ dirgraph_satisfied`. 반복 goal-call이 앞 3개를 깬다(strict replay가 2번째+ 호출을 거부/불일치, 비-멱등 DB 변경 누적, 닫힌 객체 재호출 에러).
- ✅ **선행 reliable 근거**: `diag_fix_validate.py`의 ideal trace(올바른 순서 + **goal 1회**)는 11/11 full success(cnv∧dg∧acc∧dbm)였다 → **goal을 1회로 줄이면 full success로 flip**이 이미 권위 evaluator로 입증됨(단 이는 truncate 시뮬레이션; live는 A/B로 확정).
- **왜 지금까지 안 고쳤나**: 프로젝트 헤드라인이 BOTH(dg∧acc)였고 looping은 dg∧acc를 깨지 않음 → 가시화 안 됨. 공식 success로 전환하니 지배 블로커로 드러남.

## §2. 처방 (메커니즘)
**gate(`_plan_v2`)가 매 turn 결정 시, "이 태스크의 goal 액션이 이미 성공적으로 호출됨"을 감지하면 즉시 `"STOP"` 반환.** `STOP` → `inference`가 `exit_conversation` 툴콜로 종료(`two_stage_client.py:444` 확인: `if chosen_action in ("STOP","exit_conversation","") → exit_conversation`).
- "성공적 호출" 정의 = messages의 tool 응답 중 `tool_name == self._goal_name` 이고 그 content가 success-truthy(`True` 또는 `(True, ...)`; `"Error"` 미포함).
- 결과: 첫 성공 goal-call 다음 turn에 STOP → goal이 정확히 1회 → cnv/dbm/ntce 회복.

## §3. 구현 위치 + 코드 스케치 (리뷰용)
`scripts/distill/sopbench/two_stage_client.py`, `_plan_v2` 진입 직후(offload 분기 평가 전, 또는 ACT 분기 최상단).
```python
# Fix-3 STOPSUCCESS: once the goal action has been SUCCESSFULLY called, terminate (STOP) so the
# model does not re-call it. Repeated goal calls break cnv/dbm/ntce (official success). The first
# successful call already satisfies dg∧acc; further calls only corrupt state.
if self._stopsuccess and self._goal_name:
    for m in messages:
        if not isinstance(m, dict): continue
        if m.get("tool_name") == self._goal_name:
            c = m.get("content")
            ok = (c is True) or (isinstance(c, (list, tuple)) and len(c) and c[0] is True) \
                 or (isinstance(c, str) and c.strip() in ("True", "(True", "[True"))
            if ok and "Error" not in str(c):
                return "STOP"
```
- `self._stopsuccess = bool(os.environ.get("SOPBENCH_STOPSUCCESS"))` (in `__init__`, off by default).
- 위치: offload·gate·일반 경로 모두 적용되도록 `_plan_v2` 최상단(슬롯/플랜 계산 전). LOGINFIRST/LOGINCALL/ARGFIX 등과 독립(직교).
- content 파싱은 released eval JSON 포맷(tool 응답이 `True`/`(True, 400.0)` 등 raw)·live 포맷 둘 다 커버하도록 보수적으로(타입 우선, 문자열 fallback).

## §4. BLOCKING 가드 (사전등록 — 하나라도 실패 시 미발동/롤백)
1. **B-1 (success-gated only)**: goal이 **성공 반환**한 경우에만 STOP. goal 미호출·실패호출(False/Error) 시 STOP 금지 → would-be success를 non-call로 바꾸지 않음. (코드: `ok and "Error" not in c` 전건.)
2. **B-2 (should_F 무회귀)**: Fix-3는 goal 성공 호출 *후*에만 작동. should_F(거부 태스크)는 goal을 성공 호출하면 안 되는 케이스 → 정상 trajectory에선 트리거 안 됨. **A/B에서 should_F 공식 success 카운트 불변(±0) 확인**. 만약 should_F에서 goal-성공 후 STOP이 should_F success를 바꾸면(=원래 거부해야 할 걸 호출) 그건 Fix-3 이전부터 오답 → Fix-3가 악화 안 시킴을 카운트로 확인.
3. **B-3 (zero-cost 사전검증, GPU 전 필수)**: L1C run의 12 BOTH-but-not-full에 대해 **"첫 성공 goal-call까지 truncate" 후 strict-replay-DB로 evaluator 재실행**(diag_fix_validate 방식, final_database=truncated 시퀀스의 strict gt) → full success로 flip하는 개수 측정. **flip ≥ (예측 다수)면 진행, 0이면 폐기.** 이게 Fix-3 상한을 GPU 전에 박제.
4. **B-4 (현 28 무회귀)**: 현재 full_success 28개는 Fix-3로 떨어지면 안 됨(이미 성공=goal 1회거나 looping이 무해했던 케이스). A/B에서 full_success 28→감소 0 확인. STOP은 성공 *후*만이라 성공 케이스엔 무영향이어야 함.

## §5. zero-cost 사전검증 (B-3 상세, 구현 전 실행)
신규 스크립트 `diag_fix3_offline.py`:
- 입력 = L1C eval JSON. 대상 = should_T 중 BOTH∧¬full.
- 각 태스크: func_calls를 **첫 goal-call(포함)까지** 자르고, strict 도메인시스템에 그 시퀀스를 replay해 `final_database` 산출 → `evaluator_function_directed_graph`로 재채점.
- 출력: 각 태스크 full_success flip 여부 + 전체 flip 수 = **Fix-3 예상 상한**.
- **판정**: flip 수가 충분(예: ≥8/12)하면 구현·A/B. flip 0이면 looping이 원인 아님 → 폐기(메타규칙).

## §6. A/B 실행 + 사전등록 판정
- 드라이버 `offload_stopsuccess.sh` (augment OFF): **S0** = 현 스택(LOGINFIRST+LOGINCALL, STOPSUCCESS off) = L1C 재현(공식 success 28). **S1** = + `SOPBENCH_STOPSUCCESS=1`.
- 한 서버 순차 sim+eval, fresh OUT/OFFLOG, task_sig·augment-invariant identity 조인(`diag_ab2.py` full_success + flip/regression).
- **사전등록 성공 기준**: ① 공식 full_success S0→S1 **증가**(예측 +≈(B-3 flip 수)) ② **회귀 0**(B-4) ③ should_F success 불변(B-2) ④ goal_calls 분포가 looping 케이스서 1로 수렴(`diag_loop_check`).
- 실패(증가 0 or 회귀>0)면 롤백·원인 재진단.

## §7. 리스크 / 엣지케이스
- **R1 (조기 STOP)**: 성공 감지 오탐(False를 True로) → 미성공인데 STOP. B-1 파서가 type-우선이라 위험 낮음. 단 goal 응답 포맷이 도메인마다 다를 수 있음(bank는 bool/(bool,val)) → bank 한정 검증, cross-domain은 포맷 확인 후.
- **R2 (멀티-goal 태스크?)**: bank goal은 단일 액션. goal이 1회 성공이면 태스크 완료가 정상. 복수 성공이 필요한 태스크는 bank엔 없음(확인됨).
- **R3 (dirgraph 미충족 상태의 goal 성공)**: 드물게 goal이 성공했으나 dg 미충족이면 STOP해도 success 아님 — 하지만 그건 Fix-3 무관(어차피 실패), 악화 없음.
- **R4 (should_F에서 goal 성공)**: should_F는 goal을 호출하면 안 됨. 만약 모델이 should_F에서 goal을 성공 호출하면 이미 오답; STOP이 그 뒤 종료해도 결과(오답) 불변. B-2로 카운트 확인.
- **R5 (LOGINCALL과 상호작용)**: cred-absent에서 login 실패→goal 성공(qwen 기전)인 경우, 1회 호출+STOP이면 cnv/dbm가 strict와 일치하는지 = B-3에서 직접 측정(추론 금지).

## §8. 리뷰 체크리스트 (리뷰어 확인 항목)
- [ ] B-3 zero-cost 사전검증을 GPU 전에 돌리고 flip 수를 박제했는가?
- [ ] STOP 트리거가 goal **성공** 후에만(B-1) — 미호출/실패에 트리거 안 함?
- [ ] should_F 무회귀(B-2)·현 28 무회귀(B-4)를 A/B 카운트로 강제하는가?
- [ ] 성공 감지 파서가 live(vllm) tool-응답 포맷과 eval JSON 포맷 둘 다 커버?
- [ ] 다른 fix(LOGINFIRST/LOGINCALL/ARGFIX/VALFIX/DGGATE)와 직교(독립 flag)인가?
- [ ] 공식 success(134, tool_full)로만 보고(BOTH 헤드라인 금지)?

## §9. 구현 산출물 (리뷰 통과 후)
1. `two_stage_client.py`: `_stopsuccess` + `_plan_v2` 최상단 STOP 블록.
2. `diag_fix3_offline.py`: B-3 사전검증.
3. `offload_stopsuccess.sh`: S0/S1 A/B 드라이버(augment OFF).
4. 결과 → `RESIDUAL_PREMATURE_DIAGNOSIS_2026_06_05.md` §11 + 마스터 §2 갱신.
