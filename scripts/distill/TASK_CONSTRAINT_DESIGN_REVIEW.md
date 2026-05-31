# 리뷰 — `TASK_CONSTRAINT_DESIGN.md` (should_T 병목 설계서)

> 상태: **리뷰 완료 (구현 게이트 제안)**. 작성 2026-06-02.
> 리뷰 방법: 통독이 아니라 설계서가 인용한 코드를 직접 검증(load-bearing 주장 trace).
> 대상: `scripts/distill/TASK_CONSTRAINT_DESIGN.md`.
> 검증 파일: `sopbench/build_tbox_planner_sft.py`, `sopbench/two_stage_client.py`.

---

## 0. 한 줄 결론

§2.2의 "코드-증명"(프롬프트=무거운 default / 타깃=가벼운 task 제약 → 모순쌍)은 **코드와 일치하며 진단 위치는 맞다.**
그러나 그 위에 세운 **인과 메커니즘·표본 근거·거부축 영향**이 미검증이라, 현재 상태로 재학습에 착수하면 **순손해 위험**이 있다.
재학습 전 3개 선결 진단(P1·P2·P3)을 "구현 착수 게이트"로 둘 것을 권고.

---

## 1. 검증된 부분 (토대는 견고)

| 설계서 주장 | 코드 근거 | 판정 |
|---|---|---|
| GT 타깃 시퀀스는 task 제약 기반 | `build_tbox_planner_sft.py:86` `task_dep[goal] = task["constraints"]`; `de.process(goal,…)`가 이 dep로 평가(L122-124) | ✅ 성립 |
| 프롬프트 goal status는 induced(무거운 default) precond 기반 | `two_stage_client.py:88` `_render_precond_mod(op.get("precondition"),…)`, `op = abox["operators"][nm]` (=induced) | ✅ 성립 |
| → 모순쌍 발생 | goal=`BLOCKED — first call: login_user`로 렌더(L99-100)되나 타깃 시퀀스엔 login 부재(체크→goal) | ✅ 성립 |

진단의 **위치**(프롬프트 status가 무거운 default에서 옴)는 정확하다. 이하 문제는 그 위의 추론·실험설계.

---

## 2. 문제점 (심각도순)

### 🔴 P1 — 인과 메커니즘이 자기모순적. A 단독이 듣지 않을 수 있음 (최우선)

§2.2는 "7B가 혼란 정책을 학습 → 환각 login 호출"이라 **단언**한다. 그러나 SFT 데이터는 모든 transfer_funds 예제에서
**일관되게** `(BLOCKED-login 프롬프트 → login 안 부르는 타깃)`을 보여준다. 일관되게 모순적인 데이터는 환각을 유발하는 게 아니라
**"BLOCKED라도 login 무시하고 goal 호출"이라는 override 규칙을 가르친다.** 추론 시 login을 환각한다면 원인은 둘 중 하나:

- **(a)** 7B가 override를 **학습 못 함**(용량/데이터 부족) → A(프롬프트 정합화)가 도움.
- **(b)** 추론 상태가 학습과 **다름**(fact가 observed=True가 아니라 goal이 실제 BLOCKED/VERIFY로 보임 / login이 READY로 떠 유혹) → 진단 위치가 어긋남. 이 경우 A는 헛발질.

설계서는 (a)를 기정사실로 쓰나 **미검증**. → **재학습 전**, 추론 궤적에서 *login 호출이 프롬프트 BLOCKED-status와 상관하는지* 측정 필요.

### 🔴 P2 — 진단 표본 n≈1. 분포 없이 도메인 전체 SFT 재생성

§2 증거는 거의 전부 **task 111 단일**. (직전 "레버1 인자바인딩"도 n≈1로 세웠다가 반증된 동일 함정.)
40개 should_T 중 **A가 돕는 모수**(induced-login 요구 ∧ task-constraint-login 불요)와 **A가 망가뜨리는 모수**(login이 genuinely innate)의
히스토그램이 없으면, 재학습이 순손해일 수 있다. → 분포 조사 선행.

### 🔴 P3 — should_F(거부축)의 over-gating이 load-bearing일 위험 (P2의 쌍대)

설계서가 제거하려는 "무거운 게이팅"이 **현재 should_F 31/86 성공의 일부를 떠받칠** 수 있다.
즉 일부 거부는 "fact false → STOP"이 아니라 **"login을 못 세워 goal에 영영 도달 못 함"이라는 우연한 거부**일 수 있다.
A가 그 게이팅을 걷어내면 이들이 goal 호출로 뒤집혀 should_F 회귀. §9.2는 "유지 확인 필수"라고만 적고
**현재 31건의 거부 경로 분해가 없다.** → 재학습 전 분해 필수.

### 🟠 P4 — 공정성: 출처1은 사실상 oracle 재구성인데 "full"로 보고될 위험

§4(B)가 스스로 "프루닝 셋 = 사실상 oracle 셋(directed_action_graph)"이라 인정. 그런데 1차안(출처1)은 정책 텍스트가 아니라
**`task["constraints"]`(oracle이 쓰는 동일 구조체)를 직접** 주입한다. "full-mode 7B가 천장 근접"이라는 헤드라인을 oracle-등가 정보로
도구를 프루닝해 달성하면, 리뷰어는 "oracle을 재구성하고 full이라 불렀다"고 본다. → **E-AB는 별도(semi-oracle) 조건으로 보고**,
헤드라인 주장은 출처2/3(정책에서 유도)으로만. §8에 이 보고 분리가 없음.

### 🟠 P5 — innate-dep(Q2)은 사후 체크가 아니라 선결 — 마스크 정의를 결정

§9.4는 "login이 innate인 goal에서 마스크가 login을 빠뜨리면 안 됨, 검증 필요"라고 적었으나, 이는 **빌드 후 체크가 아니라
마스크 정의(`task_constraint` 단독 vs `∪ innate_dep`)를 정하는 선결 조건**. P2와 동일 데이터(분포)로 먼저 풀어야 SFT를 한 번에 옳게 생성.

### 🟠 P6 — 분모 40 vs 34의 새로운 혼선 — "정정 문서"가 또 다른 모호성 생성

- §8: "분모=40 (정직; 결함8 제외)"
- §9.5: "현실 상한 ~34"(극難 6 제외)
- §8 성공기준 "should_T ≥ 7/48≈7/40": 7/48=14.6% vs 7/40=17.5%로 **다른 값**인데 카운트 7을 분모만 바꿔 동일시.

이 문서의 존재 이유가 "천장 24→40 오정정"인데 40/34/32와 카운트 7을 정리 안 하면 **같은 실수 반복**.
→ 단일 reconciled 표(48 = 8결함 + 6극難 + 2경계 + 32해결가능)로 못박고 사전등록 분모 하나로 고정.

### 🟡 P7 — 비싼 재학습 전 무료(zero-train) 진단을 건너뜀

§7은 추론-전용 band-aid를 "train/test 불일치"로 비추천하나 이는 **최종 방법**으로서의 평가.
**방향성 진단**으로는 추론 시 프롬프트 status만 task 제약으로 덮어써(재학습 0) login 호출률 변화를 보는 게 P1을 싸게 검증.
원격 GPU 비용·과거 낭비 사이클 감안 시 이 순서가 합리적.

### 🟡 P8 — plumbing 가정 미검증

§6.3 "task 객체 접근 가능 지점 존재"·`client.reset(task_constraints=…)`는 **주장일 뿐 확인 안 됨**.
추론 시 client가 task 객체에 정당히 접근 못 하고 env만 가진다면 출처1은 agent/env 경계 누수가 되고 가용성도 불확실.
`reset` 시그니처 포함 확인 필요(가벼움).

---

## 3. 권고 — 재학습 전 게이트

| # | 선결 진단 | 막는 위험 | 비용 |
|---|---|---|---|
| 1 | **분포 조사**: 40 should_T를 (induced-login 요구 ∧ task-constraint-login 불요) vs (login innate)로 분류 → 마스크 정의 확정 | P2·P5 | 정적 분석, 무료 |
| 2 | **should_F 거부경로 분해**: 31건이 FACT-STOP 거부인지 login-미수립 우연 거부인지 | P3 | 기존 궤적 재분석, 무료 |
| 3 | **zero-train 프롬프트 패치 진단**: login 호출률이 BLOCKED-status와 상관하는지, 정합 status에서 떨어지는지 | P1·P7 | eval 1회, 재학습 0 |

1~3이 모두 긍정일 때만 재학습 1회 착수. 실험은 **full vs policy-pruned(semi-oracle) 분리 보고**(P4), 분모는 reconciled 표 하나로(P6).

---

## 4. 리뷰어 결정 질문(설계서 §10)에 대한 답

- **Q1(마스크 출처)**: 헤드라인 주장엔 출처1 부적합(P4) — 출처1은 *진단 상한*으로만, 보고는 출처2/3.
- **Q2(마스크 정의)**: 게이트#1(분포 조사) 결과로 결정. 미해결 상태로 빌드 금지(P5).
- **Q3(범위)**: A 단독 성공이 P1 때문에 불확실 → A 단독을 "병목2 격리"로 쓰려면 게이트#3 선행 필수.
- **Q4(프루닝 강도)**: P4(공정성)와 직결 — 정확히 task 제약 함수만 프루닝하면 oracle-등가, 1-hop 여유를 둬야 회복경로 보존 + oracle-격차 유지.

---

## 부록 — 정정 누적(메모리/결과본 연계)

설계서 §1 정정("천장 40, 24 아님")은 본 리뷰 P6의 reconciled 표가 확정될 때 결과본(`SOPBENCH_EXPERIMENT_RESULTS.md`)·메모리에
일괄 반영하는 것이 자연스럽다(같은 표를 공유). 본 리뷰는 정정 자체엔 손대지 않았다(리뷰 범위 한정).

---

## 5. 저자 응답 + 게이트 실측 (2026-06-02) — 리뷰 수용, 일부 정밀화

리뷰의 3-게이트를 **구현 게이트로 채택**. 무료 게이트(P2·P3·P6·P8)를 즉시 실측, zero-train(게이트3)은 실행 중. 결과:

- **P1 정밀화 (framing 양측 정정)**: SFT 데이터 실측 = "일관된 모순"(리뷰)도 아니고 "일관 모순쌍"(설계서 §2.2 초안)도 아님.
  (BLOCKED-login)→login **76** vs →GOAL(skip) **24** = **비단사 프롬프트**(같은 status, 다른 정답). 리뷰의 "override 학습"·"A 헛발질"
  가설은 약화 — A(status를 단사로)가 **오히려 강화**됨. 단 "검증 후 진행" 결론은 유지.
- **P2 (분포) ✅ 실측**: A_HELPS=**14**(default-login∧task-light), A-무손해=31(task가 login 필요→A가 올바르게 무겁게), neutral=3.
  → A addressable=14(n≈1 아님), **should_T에서 A 순손해 위험 낮음**(A는 31엔 손대지 않음).
- **P3 (should_F 회귀) ✅ 실측 — 리뷰가 옳음**: 통과 31 = PRINCIPLED(fact-False)16 + STOP/기타9 + **ACCIDENTAL(auth=F 우연거부)5[위험]** + 1.
  **5/31 fragile**(set_safety_box). 16 principled는 A 무관(fact 게이팅 보존). → **배포 전 이 5건 회귀 모니터를 성공기준에 포함.**
- **P4 (공정성) 수용+분리**: A=공정(정책 등가)·헤드라인 가능 / B=semi-oracle·별도 조건 보고. 설계서 §9.1·§4 반영.
- **P5 (마스크 정의)**: P2의 "31 task-login-필요"가 곧 innate/필요 모수 → 마스크는 **`task_constraint`만으로 충분**(login 필요 task는 task 제약 자체에 login 포함). `∪ innate_dep`까지 필요한지는 zero-train 결과로 최종 확인.
- **P6 (분모) 확정**: 48 = 8결함+6극難+2경계+32통상. 등록 분모 = **/48 주 + /40 보조**(설계서 §2.4·§8).
- **P7 (zero-train 먼저) 채택**: env `SOPBENCH_LIGHTEN`으로 추론 시 goal status만 task 제약으로 렌더(재학습 0). `output_v4a_v2_lighten` eval 실행 중 → login 호출률·should_T(/40)·**should_F 5 fragile 회귀** 측정.
- **P8 (plumbing) ✅ 확인**: 클론 `run_simulation.py:151` per-task `client.reset()`가 `task` 스코프 내 → `reset(task_constraints, goal)`로 주입 가능(패치 완료, compile OK).

**게이트 통과 기준(사전등록)**: zero-train에서 (i) login 과잉호출 ≥50%↓ AND (ii) should_T(/40) 상승 AND (iii) should_F fragile 5건 회귀 ≤2 → 재학습 1회 착수. 미달 시 메커니즘 재검토(A가 status에 둔감 등).
