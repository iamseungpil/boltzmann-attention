# 설계리뷰 — §11 TBox/ABox 완전 분리 학습 (19-domain, entanglement-free)

> 리뷰 대상: `WORKFLOW_ONTOLOGY_DESIGN.md` §11 (2026-06-01, 사용자 지시본)
> 리뷰 단계: **설계리뷰** (다음 = 코드리뷰 → 실험). 작성 2026-06-01.
> 요청 검토 항목: (a) §11.3 means-ends 룰 정식화, (b) §11.4 copy-grounding+셔플/alias 분리 강도,
> (c) arm-3v2(무학습)를 arm-4a(학습) 앞에 두는 순서.

## 0. 한 줄 결론
분리 명제와 실험 골격(copy-target SFT → LODO 전이 → empty/wrong-ABox 음성대조)은 **건전**하고,
**(c) 순서는 맞다(유지)**. 다만 **(a)와 (b)는 "의도는 맞으나 현재 정식화로는 부족"**하다.
특히 **(a) §11.3 룰의 후방 회귀 부재**가 naive arm-3의 pass@1 0%를 만든 가장 유력한 근본 원인이므로,
arm-3v2를 돌리기 전에(또는 L0로 먼저 반증하고) 고쳐야 한다.

---

## (a) §11.3 means-ends 룰 정식화 — 의도와 맞는가

**판정: 의도 일치 ✅ / 정식화 불충분 ❌.**
의도(도메인-불변 선택 정책 = 적용가능·목표관련 operator 선택, 나머지 gate, 완료 시 종료)는 분리 명제와
정확히 일치한다. 그러나 §11.3 774행에 적힌 룰은 means-ends가 아니라 **greedy forward 휴리스틱**이다.

### A1. 후방 회귀(backward regression)가 없다 — 핵심 결함 (P0)
- 적힌 룰(774): "precondition이 현 slot_state로 충족 ∧ 미충족 goal/subgoal slot을 produces하는 operator 선택."
- 진짜 means-ends(GPS/HTN)는 **목표에서 역방향**: 미충족 goal slot → 그것을 produces하는 operator X →
  X의 미충족 precondition을 **subgoal로 push** → 재귀. 그런데 룰엔 **subgoal을 생성하는 회귀 단계가 없다.**
  "goal/subgoal slot"이라 쓰지만 subgoal을 만드는 메커니즘이 정의돼 있지 않다.
- 귀결(실패 시나리오):
  > goal이 X를 요구 → X의 precondition이 slot s 요구 → s는 operator Y가 produces. 하지만 Y는 *goal* slot이
  > 아니라 *subgoal* slot만 produces → **현재 룰은 Y를 절대 선택 못 함** → precondition 체인에서 정지.
- 정황 증거: SOPBench bank 태스크 대부분이 verify_identity → check_constraint → act 형태의 **2–3단
  precondition 체인**이고, arm-3 naive 실패의 ~90%가 "제약/dirgraph 위반". 이는 precondition 순서를 못 맞추는
  증상 = greedy forward의 불완전성과 일치한다.
- 문서는 "= PDDL/HTN means-ends"라 표기하나, 적용가능성+관련성 필터만 있고 회귀가 없어 **PDDL/HTN보다
  증명 가능하게 약하다.**
- **조치**: goal-slot 스택 + `precondition → subgoal push` 재귀를 명시. (L0에서 결정론적으로 구현, L1/L2는
  같은 구조를 컨텍스트/가중치가 근사.)

### A2. `achieves` 관계 불일치 (3곳 충돌) (P0)
- §11.2 표: 8 관계 = realizes/precondition/produces/arg/next/scenario/terminate/output — **achieves 없음.**
- §11.3 입력·룰: `achieves(fn→goal_slot)` 사용.
- §11.9 `induce_ontology_zekun.py`: 8 관계만 induce → achieves는 induce되지 않음.
- 즉 planner 룰이 induce되지도, 8에 포함되지도 않은 9번째 관계에 의존한다.
- **조치(택1)**: (i) achieves를 induction에 추가, 또는 **(ii) achieves 제거하고 `produces ∩ output.required`로
  재정의** — `goal_slots := output.required`, "operator가 goal slot을 achieve" ⟺ `produces(fn) ∩ output.required ≠ ∅`.
  → **(ii) 권장** (새 관계 불필요, 8관계 닫힘 유지).

### A3. 거부(refusal) 종단이 룰에 없다 (P0)
- §9.3 hard axis = `action_should_succeed=false`(거부 정확도). 그러나 룰의 종단은 "select" 또는
  "goal 다 차면 terminate" 둘뿐. "어떤 operator의 precondition도 (전이적으로) 충족 불가 → 거부"가 없다.
- 진짜 means-ends는 이를 자연히 산출(goal 스택 비지 않았는데 적용가능 operator도, 그걸 만들 operator도 없음
  → fail). 단, 이는 A1의 후방 탐색이 있어야 탐지 가능.
- **조치**: 명시적 `refuse` 종단 추가 (A1에 의존).

### A4. L0 tie-break 미정의 (P1)
- 조건을 만족하는 operator가 여럿일 때 선택 미정의. L1/L2는 LLM/가중치가 끊으나, **L0(arm-2, 결정론)는
  tie-break 미정의면 재현 불가**이고 §9.4(c) "L0<L1<L2 gap" 주장이 흔들린다.
- **조치**: `next` 토폴로지 순 또는 "잔여 precondition 최소" 등 결정론 tie-break 명시.

**→ (a) 종합**: A1(후방 subgoal 회귀), A2(achieves 통일=produces∩output), A3(refuse 종단), A4(L0 tie-break)를
추가. A1이 0%의 가장 유력한 근본 원인 → arm-3v2 실행 전 수정 권장.

---

## (b) §11.4 copy-grounding + 셔플/alias — 충분히 강한 분리 장치인가

**판정: 백본 ✅ / 현재 명세 ❌.**
백본(copy-target + selection-span만 supervise + cross-domain 배치)은 옳고 §11.0 entanglement를 직접 고친다.
그러나 현재 명세(alias가 "(선택)", 붕괴 임계 미정량)로는 부족하다.

### B1. alias는 "(선택)"이 아니라 load-bearing이다 (P1)
- 셔플은 **위치** 암기만 막는다. **어휘** 암기는 못 막는다.
- operator 이름이 의미를 담아(verify_identity, check_balance…) 7B는 6개 학습 도메인에서 "goal에 'transfer' →
  먼저 verify_identity emit"를 **이름 공기(co-occurrence)만으로** 학습 가능 = ABox 내용이 가중치로 누수
  = 막으려던 그 실패(§11.0).
- LODO held-out이 부분적으로만 잡는다: 도메인 간 공유/유사 이름(verify_*, check_*, get_*)이 어휘 prior를
  전이시켜 **룰 학습 없이도 held-out 숫자를 부풀린다.**
- **조치**: alias를 **필수 + per-epoch 랜덤화**(같은 trace가 epoch마다 다른 alias). 구조 기반 정책 강제.

### B2. copy-grounding은 출력만 묶지 입력 reading을 보장 않는다 (P1)
- "이름이 context에 verbatim → copy"는 target이 pointer임을 보장하나, 선택이 precondition/produces로
  계산됐는지는 보장 못 함.
- copy는 분리의 **필요조건**이고, 충분하게 만드는 건 alias(이름 의미 절단) + empty/wrong-ABox 붕괴(reading 증명).
- **결론**: (b)가 "충분히 강함"은 **alias 필수 + (ii)(iii) 붕괴를 큰 효과크기로 입증**할 때만 참.

### B3. 붕괴 임계치 사전 정량화 (P1)
- §11.7 "붕괴"가 눈대중이면 약하다. 사전 등록 예: `wrong-ABox pass ≤ 1.2× empty-ABox` AND
  `≤ 0.3× correct-ABox`. 안 그러면 "약하게 낮아진 것도 붕괴냐" 반론.

### B4. ablation (v) alias-불변 추가 (P1)
- 현 ablation (iv)=operator-shuffle 불변에 더해 **(v) alias-불변**(이름을 무작위 치환해도 선택 동일)을 추가.
  alias 없이 통과한 LODO는 해석 모호(어휘 전이 가능성).

### B5. slot 이름 잔여 누수 (P2, 스트레치)
- operator를 alias해도 goal/slot_state 키는 도메인명(account_id, loan_balance). produces-matching 자체는
  의도된 동작이라 OK지만, slot명 어휘 과적합 여지 잔존.
- **조치(스트레치)**: operator명+slot명 둘 다 alias한 ablation 하나(관계 그래프만 남김) — Phase 1엔 과할 수
  있으나 가장 깨끗한 분리 증명.

**→ (b) 종합**: alias 필수+per-epoch, (v) alias-불변 ablation 추가, 붕괴 임계 사전등록, (스트레치) slot명 alias.
이를 반영하면 충분히 강하다.

---

## (c) arm-3v2(무학습) → arm-4a(학습) 순서 — 맞는가

**판정: 맞다. 유지.** ABox 주입 효과를 격리하는 정확한 방법.

- arm-3-naive(0%) → arm-3v2는 변수 **하나**(ABox precondition/produces 주입 + gate + exit, 학습 0)만 바꾼다.
  arm-3v2 ≫ 0%면 "ABox 주입+gate가 활성 성분"을 GPU 없이 증명. 여전히 ~0%면 SFT로도 못 살릴 확률이 높으니
  학습 사이클 절약하고 planner/gate/means-ends(a)부터 디버그.
- 정직한 분해가 이 순서라야 나온다: Δ(naive→3v2)="구조+ABox in-context" 효과, Δ(3v2→4a)="학습된 정책" 효과.
  중간 arm-3v2 없으면 0%→학습숫자 점프가 "그래프 줬다"와 "학습했다"를 뒤섞고, reviewer가 정확히 이를 묻는다.
- cheapest-first 리스크 정렬로도 맞다: arm-3v2는 `two_stage_client.py` 프롬프트/디코딩 변경(학습·induce 품질
  의존 적음)으로 §11 전제를 가장 빨리 반증.

### C-보강1. induce는 arm-3v2의 병렬이 아니라 선행 의존 (P0)
- arm-3v2가 주입할 precondition/produces ABox가 induce 산출물 → 임계경로 =
  **induce → induced↔GT `directed_action_graph` 대조 검증(§9.4d) → arm-3v2.**
- 이 검증을 **명시적 gate**로 격상. 안 그러면 arm-3v2 저조 시 "룰/gate 오류"와 "induce 오류"를 구분 불가.
  (현 build order가 이미 induce→3v2 순서이니, 사이 검증을 사후가 아닌 관문으로 올리기만 하면 됨.)

### C-보강2. 최소 L0(arm-2)를 arm-3v2 앞으로 (P0)
- L0는 LLM 없이(cheap) "operator만으로 워크플로가 결정되는가"의 가장 깨끗한 테스트.
- GT ABox 위 L0가 bank를 못 풀면 means-ends 룰(a)이 LLM과 무관하게 깨진 것 = (a)에 대한 가장 빠른 신호이자
  gate 로직을 공유하는 arm-3v2의 디리스킹.

**권장 실행 순서:**
```
induce
  → (induced ↔ GT dirgraph 대조 = gate)
  → L0 / arm-2  (GT ABox로 룰+ABox 검증, LLM 無)
  → arm-3v2     (LLM tie-break 추가, 무학습 L1)
  → arm-4a      (cross-domain copy-grounded SFT, 학습)
```

---

## 요약 — 코딩 전 조치 (우선순위)

| 우선 | 막는 것 | 조치 | 항목 |
|---|---|---|---|
| **P0** | §11.3 룰이 0%의 원인일 수 있음 | 후방 subgoal 회귀 + achieves 통일(produces∩output) + refuse 종단 | A1·A2·A3 |
| **P0** | induce 오류와 룰 오류 혼동 | induce↔GT dirgraph 대조를 명시 gate로; L0를 arm-3v2 앞에 | C-보강1·2 |
| **P1** | §11.4 어휘 누수로 LODO 해석 모호 | alias 필수+per-epoch, (v) alias-불변 ablation, 붕괴 임계 사전등록 | B1·B3·B4 |
| **P1** | L0 재현성 / gap 주장 | L0 결정론 tie-break 명시 | A4 |
| **P2** | slot명 잔여 누수 | slot명 alias ablation(스트레치) | B5 |

순서 자체(arm-3v2 → arm-4a)는 그대로 유지. **가장 싼 검증은 L0를 먼저 당겨 means-ends 룰(a)을 반증/검증하는 것.**

## 미결 결정 (사용자 판단 필요)
1. **§11.3 P0를 설계에 먼저 반영하고 갈지** vs **현 greedy forward 룰 그대로 arm-3v2를 먼저 돌려
   "greedy로도 0%를 벗어나는가"를 데이터로 보고 회귀 추가를 결정할지.** (리뷰어 권고: L0 우선.)
2. achieves 처리: (i) induction에 추가 vs **(ii) produces∩output 재정의(권장).**
