# 코드리뷰 — arm-3 구현 (TwoStageClient + 배포/드라이버)

> 리뷰 대상: `scripts/distill/sopbench/{two_stage_client.py, apply_two_stage_patch.py, run_arm3_sweep.sh}`
> 단계: **코드리뷰** (설계리뷰 `DESIGN_REVIEW_s11_tbox_abox_2026_06_01.md` 완료 → 이 문서 → 실험).
> 작성 2026-06-01 (Track A). 코드 정독 + 라이브 클론 대조 기반.

## 0. 한 줄 결론
파이프라인 골격(합성 ChatCompletion 호환·표준 run_evaluation 정합·5패치 멱등 배포)은 **이미 검증·건전**.
그러나 **planner의 출력 파싱(C1)과 무손실 IO(C8)이 arm-3 bank 0%를 설계리뷰의 룰 결함(A1)과 함께 공동 교란**한다
→ 0%를 "룰 vs 파싱실패 vs 상태 비가시성"으로 분해 불가. **⇒ L0(LLM 無, GT ABox)가 유일한 깨끗한 중재자**
(설계리뷰 C-2 강화). 코드는 arm-3v2/arm-4a가 위에 올라가기 전 **C1·C6·C7 수정 권장**.

---

## A. `two_stage_client.py`

### C1. Planner 출력 파싱이 취약 → 무로깅·순서의존 first-tool 폴백 (P0, 측정 타당성)
- 141행: `chosen = resp...content.strip().split()[0].strip(".,:")`. 모델이 프리앰블을 붙이면("The best action
  is apply_credit_card") `split()[0]`="The" → tool_names 불일치 → 144–145행 **`next(iter(tool_names))`=리스트
  첫 도구로 조용히 폴백**. 문제 둘:
  1. **순서 의존·무가시 폴백**: 어느 도구가 첫째냐(도메인별 dict 순서)에 따라 편향. 폴백이 얼마나 자주 터지는지
     **로깅이 없어** arm-3 0%가 "planner가 틀린 선택"인지 "파싱 실패→첫 도구"인지 **구분 불가**.
  2. **빈 출력 크래시**: content가 공백뿐이면(truthy) `"".split()[0]` → **IndexError** → run_simulation 예외→retry 소진.
- **영향**: arm-3 naive 0%의 일부가 룰이 아니라 이 파싱일 수 있음 → 설계리뷰 A1(greedy 룰)과 **교란**.
- **조치**: (a) 전체 출력에서 operator명 **containment 매칭**(첫 토큰 아님), (b) **폴백률 카운터+로깅**,
  (c) 빈 split 가드. ★근본수정 = arm-3v2의 **copy-grounded constrained decoding**(제공 operator명으로 디코딩 제약,
  §11.9-4) — 이걸 넣으면 C1 소멸.

### C2. Planner 입력이 무손실 → precondition 상태를 못 봄 (P0, 설계 A1 코드 확증)
- planner 컨텍스트(108–146행) = goal **첫 400자**(113행) + operator **이름+desc[:120]**(110행) + **last-6 history,
  tool_result는 80자 절단**(120행). **구조적 slot_state(무엇이 검증/충족됐는지)가 입력에 전혀 없다.**
- ⇒ planner는 "선행 제약이 충족됐는지"를 알 수 없어 SOP 순서를 못 따름 = arm-3 실패의 제약위반 90%와 정합.
  설계리뷰 A1(후방 회귀 부재)을 **코드 레벨에서 확증**(룰뿐 아니라 IO도 상태를 안 줌).
- **조치**: arm-3v2 = planner에 **ABox(precondition/produces) + 현 slot_state** 주입(§11.3 means-ends 입력).
  이미 설계에 있음 — 코드가 그 필요를 재확인.

### C6. Coverage 진단이 --two_stage 경로에서 저장 안 됨 (P1)
- `coverage()`(212행)는 cov_turns/cov_deterministic을 누적하나, **run_simulation.py는 `client.coverage()`를
  호출/저장하지 않음**(구 run_two_stage만 했음). ⇒ "would-be deterministic %" 진단이 산출되나 **폐기**.
- **조치**: 진단이 필요하면 run_simulation 패치에 per-task `client.coverage()` 캡처 추가, 아니면 docstring에서
  "현재 미수집" 명시. (pass@1엔 영향 없음 — 진단축일 뿐.)

### C7. 재시도 시 planner는 temperature 고정 → 회복 불가 (P1)
- run_simulation은 retry에서 `assistant_agent.temperature=0.7`로 올림. resolver는 `create_params["temperature"]`
  (=0.7) 사용(172행)하나, **planner는 140행에서 `temperature=0.0` 하드코딩** → retry가 **resolver만 다양화**.
- ⇒ 실패점이 planner면 retry로 못 살림(헛 retry 소진). 
- **조치**: planner도 `temperature`(create_params 또는 인자) 반영, 또는 의도면 주석 명시.

### C3 / C5. 결정론 경로·slot 마이닝 위험 (P1, use_deterministic_shortcut=True일 때만)
- C3: 165–166행 `args = {r: slot_state[r] ...}` = **원형 slot 값(타입/의미 미검)** → 동명 슬롯 충돌 시 오호출.
- C5: `_update_slots`의 blind `dict.update`(186행)는 후행 tool-result가 선행/사용자 슬롯을 **덮어씀**.
- 기본(off)에선 pass@1 무관(진단만). **`--two_stage_det` 켤 때 수치 신뢰 전 반드시 검토.**

### C4. `_update_slots` 매턴 전체 history 스캔 (P2, 성능만)
- 181행 매 턴 전체 messages 재스캔·재파싱 = O(turns²). 정확성 무관, 긴 대화서 느림.

### C2-guard. `parameters`가 None이면 크래시 가능 (P2)
- 156행 `fn.get("parameters", {})`는 키 **부재** 시만 {} — 값이 `None`이면 `None.get("required")` AttributeError.
  SOPBench 스펙은 항상 parameters를 주나, 방어적으로 `(fn.get("parameters") or {})` 권장.

**OK (유지)**: `_make_tool_call_completion` 합성객체(openai 2.38.0 검증), `inference` 인터페이스, reset() 슬롯
초기화, `_try_parse`(json→ast, run_evaluation try_eval 정합).

---

## B. `apply_two_stage_patch.py`

- **P4 패치 안전 확인 ✅**: endpoint early-return이 `self.VLLM_PORT` 미설정하나, `self.process=None`은 `__init__`
  (llm_handler:96)에서 설정되고 `kill_process`(252)가 `if self.process:`로 가드, `VLLM_PORT`는 spawn 경로(180/232)
  에서만 참조(endpoint 시 스킵). 코드+overnight 실증 모두 무크래시.
- **멱등·앵커·컴파일 ✅**: fresh-originals 클론서 5패치 적용+컴파일+재실행 skip 검증완. assert(앵커 1회) 안전망 있음.
- **주의(문서화됨, 코드결함 아님)**: arm-1 fc는 `--served-model-name`=짧은 표준id가 constants 등록명과 일치해야
  통과. 불일치 시 OpenAIHandler 예외→해당 도메인 empty results→eval에서 **조용히 NA**(아래 S2).

---

## C. `run_arm3_sweep.sh`

- **S2 (P1)**: arm-1이 모델명 불일치/예외로 실패하면 summary에 **NA만 뜨고 이유는 `<domain>.simlog`에만**. 드라이버가
  실패를 표면화하지 않음 → coworker가 "0%인가 실패인가" 혼동 가능. **조치**: eval 파싱 실패 시 simlog 마지막
  에러줄을 summary에 같이 출력, 또는 sim 비정상종료 감지.
- **S3 (P2)**: 직렬 14런(7도메인×{arm1,arm3}). 32B/72B서 느림(정상). 도메인 병렬은 단일 endpoint 부하로 보류 타당.
- **OK**: arm별 odir 분리(파일 충돌 0), `mkdir -p` 수정됨, "Mean Pass Rate" 파싱 end-to-end 검증.

---

## D. 우선순위 + arm-3v2/arm-4a 영향

| 우선 | 막는 것 | 조치 | 항목 |
|---|---|---|---|
| **P0** | arm-3 0%를 룰/파싱/IO로 분해 불가 | **L0(LLM無) 먼저** = 코드 교란까지 제거하는 유일 중재자 | C1·C2 + 설계 C-2 |
| **P0** | planner 출력 무가시 폴백·빈크래시 | containment 매칭+폴백률 로깅+빈가드; arm-3v2 copy-grounded decoding이 근본수정 | C1 |
| **P0** | planner가 precondition 상태 못 봄 | arm-3v2: ABox(precondition/produces)+slot_state 주입(§11.3) | C2 |
| **P1** | coverage 진단 폐기 / retry 헛소모 | run_simulation서 coverage 캡처(선택); planner temperature 반영 | C6·C7 |
| **P1** | det 경로 오호출 / arm-1 실패 은폐 | det 켤 때 타입검·슬롯충돌 검토; 드라이버가 sim 실패 표면화 | C3·C5·S2 |
| **P2** | 방어/성능 | parameters None 가드; slot 스캔 캐싱 | C2-guard·C4 |

**결론**: 현 arm-3 코드는 *측정 파이프라인*으로는 건전하나(수치 신뢰 가능, 0%는 진짜 결과), **planner 정책 자체가
파싱·IO·룰 세 층에서 동시에 약하다.** arm-3v2가 도입할 **copy-grounded constrained decoding + ABox/slot_state
주입**이 C1·C2를 동시에 고치므로, 별도 패치보다 **arm-3v2 구현에서 planner 경로를 재작성**하는 것이 효율적.
단 그 전에 **L0를 먼저** 돌려(LLM·파싱 교란 0) means-ends 룰(설계 A1)을 깨끗이 검증/반증할 것 — 이것이
0%의 근본원인을 확정하는 가장 싸고 유일한 방법.
