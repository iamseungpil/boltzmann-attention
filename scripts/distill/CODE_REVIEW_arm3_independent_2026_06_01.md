# 코드리뷰 (독립 2차) — arm-3 TwoStage 구현

> 대상: `sopbench/{two_stage_client.py, apply_two_stage_patch.py, run_arm3_sweep.sh}`
> 1차 코드리뷰(`CODE_REVIEW_arm3_2026_06_01.md`, HEAD=42e5a3b)와 **독립**으로 재독 후 대조.
> 단계: 코드리뷰 (설계리뷰 → **코드리뷰** → 실험). 작성 2026-06-01.
> 목적: arm-3v2/arm-4a가 이 코드 위에 올라가므로, 다음 구현 디리스킹.

## 0. 결론
1차 리뷰의 골격 판정(합성 ChatCompletion 호환·표준 평가 정합·5패치 멱등·endpoint 패치 안전)과
**P0/P1 발견(C1·C2·C6·C7·C3·S2) 전부를 코드 라인 단위로 확증**한다. 0%는 측정 버그가 아니라
진짜 결과라는 점, 그리고 "L0 먼저"가 0% 근본원인 확정의 유일하게 깨끗한 방법이라는 권고에 동의한다.
**추가로 1차가 다루지 않은 7건(N1–N7)을 발견했으며, 특히 N1은 0% 메커니즘에 직결되고 설계리뷰 A3를
코드 레벨에서 확증한다.**

---

## 1. 1차 발견 확인/정정 (라인 대조)

| ID | 1차 주장 | 코드 라인 | 판정 |
|---|---|---|---|
| C1 | planner가 `split()[0]` 첫토큰만 파싱 → 프리앰블 시 불일치, 빈출력 IndexError, 폴백 무로깅 | `two_stage_client.py:141-142` | **확인(정정 1건↓)** |
| C2 | planner 입력에 구조적 slot_state 없음 → precondition 상태 못 봄 (A1 코드확증) | `:130-136` (프롬프트), `:108-128` | **확인** |
| C3 | det 경로가 slot 원형값 무검증 주입 → 오호출 | `:165` | **확인** |
| C6 | coverage 진단이 `--two_stage` 경로에서 저장 안 됨 | `coverage():212-216` 존재하나 patch가 미호출 (`apply_*.py` Edit B/C에 save 없음) | **확인** |
| C7 | retry 시 planner는 temp=0.0 하드코딩, resolver만 다양화 | `:140` (planner 고정) vs `:172` (resolver는 create_params temp 사용) | **확인** |
| S2 | 드라이버가 arm-1 실패 시 NA만, 이유는 simlog에만 | `run_arm3_sweep.sh:73,75-77` | **확인(보강 N6)** |

**C1 정정 (사소하나 정확히):** 진짜 빈 문자열 `""`은 삼항 가드(`if ...content else ""`, :141-142)가
잡아 `chosen=""`로 간다. IndexError는 **공백만 있는 출력**(`" "`, `"\n"`)에서 난다 — `" ".strip()=""`,
`"".split()=[]`, `[][0]` → crash. 즉 "빈출력 크래시"가 아니라 "공백-only 크래시"다. 위험은 동일하게 실재.

---

## 2. 추가 발견 (1차 미포함)

### N1 — resolver의 강제 tool_choice가 거부·종료를 구조적으로 차단 (P0, 강함)
- `_resolve`가 항상 `tool_choice={"type":"function","function":{"name":action}}`로 **선택 도구를
  강제 호출**한다 (`:170-173`). 모델이 거부하거나 자연어로 답하거나 멈출 경로가 resolver에 없다.
- 비-도구 경로는 `inference`의 `if not tools` 분기(`:92-96`)뿐인데, SOPBench는 매 어시스턴트 턴
  tools를 넘기므로 사실상 발동 안 됨.
- **귀결**: arm-3는 planner가 매 턴 도구를 고르고 resolver가 강제 호출 → **`action_should_succeed=false`
  (거부 정확도, §9.3 hard axis) 태스크를 구조적으로 통과 불가.** planner는 slot_state도 못 보므로(C2/A1)
  "제약 충족 불가 → 거부"를 판단할 입력조차 없다. 0% 중 거부-축 부분의 직접 메커니즘.
- **설계리뷰 A3(refuse 종단) 및 §10.5 #6(종료 정책 부재)을 코드에서 확증.** 종료는 planner가
  `exit_conversation`을 *고를 때만* 가능(그 도구가 tools에 있을 때) — 능동적 거부와는 다름.
- **검증 권고**: SOPBench가 거부-축을 어떻게 채점하는지(금지 액션 미호출로 충분한가 vs 명시적 거부
  필요)를 oracle 코드에서 확인. 전자라면 강제 tool_choice가 "안전한 무해 도구" 무한호출로도 위반 누적.
- **수정**: arm-3v2 planner에 `refuse`/`terminate`를 1급 선택지로(설계리뷰 A3), resolver에서 그 두
  선택은 tool_choice 강제를 우회해 빈 도구턴/종료로 매핑.

### N2 — first-tool 폴백이 "첫 도구"가 아니라 임의(해시순) (P1, C1 정밀화)
- `:143` `tool_names = {... for t in tools}`는 **set**, `:145` `next(iter(tool_names))`는 **set 반복
  순서** = str 해시순. `PYTHONHASHSEED` 미고정 시 **프로세스마다 폴백 대상이 달라짐.**
- 즉 1차 C1의 "순서의존 폴백"보다 나쁨 — 선언 순서의 첫 도구가 아니라 **재현 불가능한 임의 도구**.
  실패 모드조차 재현이 안 돼 디버깅을 방해.
- **수정**: 폴백을 `tools[0]`의 이름(리스트 순서)으로 바꾸고, **폴백 발생을 카운터로 로깅**(C1과 합쳐
  "틀린 선택 vs 파싱 실패" 분해 가능하게).

### N3 — slot_state가 도구-스코프 없는 평탄 네임스페이스 → 키 충돌 오염 (P1, C3/C6 악화)
- `_update_slots`(`:179-195`)가 모든 `role==tool` 결과 dict를 `self._slot_state.update(r)`로 **블라인드
  병합.** 서로 다른 도구가 같은 필드명(`status`,`id`,`result`,`success`,`error`)을 반환하면 **나중 값이
  조용히 덮어씀.** 도구별 스코프가 없다.
- 귀결: (a) det 커버리지 판정(`:159` `all(r in slot_state)`)이 오염된 값으로 충족 판정 → C3 오호출 확대,
  (b) coverage 진단(C6 살릴 경우)도 부정확, (c) 실패 도구가 `{"success":false,"error":...}`를 반환하면
  `error`/`success` 키가 slot에 박혀 동명 required param을 오충족.
- 부수: 매 턴 **전체 메시지 재스캔**(`:181` 루프가 messages 전체) → 대화 길이에 O(n²), 사소하나 불필요.
- **수정**: slot 키를 `tool_name.field`로 스코프하거나, required 매칭 시 arg-source(설계 `arg(param←slot)`)
  관계로만 채움. arm-4a의 copy-grounding이 정확한 slot 출처를 요구하므로 지금 정리가 이득.

### N4 — planner max_tokens=32가 C1과 상호작용 (P1)
- `:140` planner `max_tokens=32`. 모델이 추론 프리앰블을 먼저 내면 32토큰이 **전부 프리앰블**이라 이름에
  도달 못 함 → C1 파싱을 고쳐 전체 토큰을 스캔해도 **이름이 잘려 없을 수 있음** → 폴백 확정.
- **수정**: arm-3v2의 constrained decoding(제공 operator명으로 제약)이 근본 해결. 임시로는 "이름만
  출력" 강제 + 토큰 여유.

### N5 — slot mining과 planner의 절단 비대칭 (P2)
- slot mining은 도구 결과를 **전체 파싱**(`:184-186`)하나, planner history는 같은 결과를 **80자 절단**
  (`:120`)하고 `CALLED:`는 **인자 없이 이름만**(`:122-125`). planner는 "무엇을 어떤 인자로 이미 했는지"를
  못 봄 → C2/A1 악화. (어느 쪽이 옳다기보다, planner가 보는 상태가 mining보다 빈약하다는 비대칭.)

### N6 — 드라이버 견고성: pipefail 부재 + eval stderr 억제 (P1, S2 보강)
- `run_arm3_sweep.sh:26`은 `set -u`만. `set -o pipefail`/`set -e` 없음 → sim 실패해도 진행하고,
  `:75-76`의 grep 파이프 실패가 마스킹됨.
- `:73` eval을 `2>/dev/null`로 호출 → **eval이 크래시해도 NA만 뜨고 진단 0.** S2(arm-1 실패 혼동)의
  근본: 성공/실패/0%가 전부 NA로 수렴.
- **수정**: eval stderr를 `${odir}/${d}.evallog`로 보존, `pipefail` 추가, sim/eval 종료코드를 SUMMARY에
  컬럼으로 기록(coworker가 "0%인가 실패인가" 즉시 판별).

### N7 — description 120자 절단은 원칙적 abstract/concrete 분리가 아님 (P1, arm-4a 전이 직결)
- `:110` operator 설명을 `[:120]` 절단. 설계 §9.1 가드는 "param schema 비노출"이나, SOPBench 설명문은
  **앞부분에 인자명·전제조건·도메인 고유어**를 담는 경우가 많음 → 120자가 **누수 + 정보손실**을 동시에
  유발하고, 컷이 단어 중간을 자름.
- arm-4a는 이 텍스트가 학습 입력이 되므로 **전이 오염(§9.1)** 위험으로 승격. 설계 §10.4의 "truncation/
  leak via descriptions" 우려를 코드에서 확인.
- **수정**: arm-3v2에서 affordance를 자유 텍스트 desc가 아니라 **구조화된 ABox 관계**(precondition pred·
  produces slots·achieves=produces∩output)로 주입(설계 §11.3) → 절단 휴리스틱 폐기.

---

## 3. 안전성 재확인 (1차 동의, 이견 없음)
- 합성 ChatCompletion(`:197-210`): openai 2.x에서 `tool_calls[0].function.name/.arguments`+`.id` 산출 →
  core.py·run_evaluation 둘 다와 정합. ✅
- 5패치 멱등(`apply_*.py` `_patch`, `:46-59`): marker 가드 + anchor `assert n==1` fail-fast + `.bak` +
  `py_compile`. 견고. ✅ (사소: anchor가 upstream 공백에 민감하나 fail-fast라 안전쪽 실패.)
- endpoint 패치(`:128-140`): `SOPBENCH_VLLM_BASE_URL` 있으면 pre-served 사용, 없으면 원경로. 안전. ✅

---

## 4. 심각도 요약 + 조치

| ID | 심각도 | 한줄 | 1차연계 |
|---|---|---|---|
| **N1** 강제 tool_choice→거부 차단 | **P0** | arm-3가 거부-축 구조적 통과불가; A3 코드확증 | 신규 |
| **C2/A1** slot_state 미주입 | **P0** | planner가 precondition 못 봄 | C2 |
| **C1+N2+N4** 파싱·임의폴백·토큰컷 | **P0(합산)** | 0%를 룰 vs 파싱으로 분해 불가, 폴백 비재현 | C1 |
| **N7** desc 절단 누수 | P1 | arm-4a 전이오염 위험 | §10.4 |
| **N3** 평탄 slot 충돌 | P1 | det/coverage 오염 | C3·C6 |
| **C6/C7** coverage 미저장·planner retry 미다양화 | P1 | 진단손실·회복불가 | C6·C7 |
| **N6/S2** 드라이버 NA 수렴 | P1 | 성공/실패/0% 구분불가 | S2 |
| **C3** det 무검증 주입 | P1 | det 모드 오호출 | C3 |

## 5. 권고 (1차와 합치)
1. **L0 먼저** (LLM·파싱·tool_choice 교란 0, GT ABox). means-ends 룰(설계 A1)을 means-ends vs greedy
   둘 다 GT ABox로 돌려 깨끗이 검증/반증 — 0% 근본원인 확정의 유일·최저비용 경로. **N1/N2/N4가 L0엔
   전혀 없으므로 L0가 유일한 무교란 중재자임을 코드가 재확인.**
2. **C1·C2는 arm-3v2 planner 경로 재작성으로 동시 수정** (constrained decoding + ABox/slot_state 주입).
   별도 패치보다 효율적이라는 1차 권고 지지. **N1·N3·N7도 같은 재작성에 흡수**(refuse/terminate 1급
   선택지·slot 스코프·구조화 affordance) — 즉 arm-3v2는 단순 프롬프트 추가가 아니라 planner I/O 계약의
   재설계여야 한다.
3. **N6는 즉시 적용 가능**(드라이버만 수정, 실험 신뢰성 즉효): pipefail·evallog 보존·종료코드 컬럼.

## 6. 미결 (사용자 판단)
- N1 검증: SOPBench oracle의 거부-축 채점 방식 확인 후, arm-3v2에서 거부를 빈-도구턴으로 낼지
  전용 `refuse` operator로 낼지. (설계 A3와 연동.)
