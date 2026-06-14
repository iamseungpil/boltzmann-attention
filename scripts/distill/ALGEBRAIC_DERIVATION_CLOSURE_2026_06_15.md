# 대수적 도출 닫힘 — 형식 companion (2026-06-15)

> **도출 본체 = 매트릭스 `PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md` §1.5(b)** (Böhm–Jacopini control × provenance-완전분할 data × 게이트-타입 policy·닫힘 판정표·부산물 4·정직잔여 3). 이 문서는 그 도출을 *대체하지 않고* 형식적으로 보강하는 companion — net-new 3건만 담는다. 본체와 충돌 시 **본체가 권위**.

## 0. companion이 더하는 것 (net-new 3)
1. **흡수 메커니즘 + 두 개별화 기준의 통합** — 왜 순수 branch·유계 loop가 primitive가 *아닌가*를 명시하고, 그로써 §1.5(a)(separable-learnability)와 §1.5(b)(대수)가 **한 파이프라인의 두 단계**임을 박제(§1).
2. **두 후보 seam의 화해 + 1차 seam 확정** — 내 1차 패스는 "계산된 인자(data-transform)"를 유일 seam으로 봤으나, **§1.5(b)가 더 날카롭다: 1차 live seam = 층B 게이트-타입 유한성(G5→P10)**. data-transform은 scope로 닫힌 *2차* seam(census 조건부). 이 화해를 박제(§2).
3. **P8 ↔ P1 provenance 중복 = merge-후보 flag** — §1.5(b)가 함의하나 명시 안 한 분해(§3).

---

## 1. 흡수 메커니즘 — 대수와 separable-learnability의 통합 (net-new)
§1.5(b)는 control 생성기를 {순차·선택·반복·병렬}로 둔다. 그런데 **"선택"과 "반복"은 그 자체로 primitive가 아니다** — 이게 §1.5(b)가 P2a/P5(선택)·P7(반복)으로 *분산*시킨 이유를 설명한다:

- **순수 선택(branch)**: 술어가 grounded되면 control fork는 자명. 분리 학습가능성 0 ⇒ **흡수**: 술어소스가 데이터면 P2a, 정책이면 P5. (= §1.5(b) "선택의 조건소스 두 맛"의 메커니즘적 근거.)
- **유계 loop(컬렉션 map)**: = P4(원소선택) × (P3|P9). 멈춤·전략 판단 없음 = 분리 학습가능성 0 ⇒ **흡수**. *남는* loop = 무계 retry 하나뿐 → 층B verdict와 곱일 때만 = **P7**. (= §1.5(b) "유일 유의미 루프=에러/deny 재시도"의 근거.)

> **통합 주장**: 대수(§1.5b)가 **후보 생성기**를 낳고, separable-learnability(§1.5a-ii)가 그중 **학습불가 생성기(branch·bounded-loop)를 가지친다**. 두 개별화 기준은 독립 임시방편이 아니라 **생성→가지치기 한 파이프라인의 두 단계**. ⇒ "왜 정확히 이 10셀/9패밀리인가"가 두 기준의 *합성*으로 답해짐(어느 하나만으로는 불완전: 대수만 두면 branch/loop가 남고, learnability만 두면 후보집합이 비원리적).

---

## 2. 두 후보 seam의 화해 — 1차 = 층B (net-new 화해)
닫힘이 **정리로 잠기지 않은** 곳 = 정확히 둘. §1.5(b) 판정표를 seam-언어로 재진술:

| 축 | 닫힘 근거 | seam 지위 |
|---|---|---|
| 층A control | Böhm–Jacopini + par | **잠김**(정리) — seam 아님 |
| 층A data | provenance 완전분할 — **단 transform 제외(제약 S) 하에서** | **2차 seam**: S는 scope-자인이지 정리가 아님. 실타깃이 *계산된 인자*(obs→fn→arg)를 요구하면 P10 발화. **census 조건부.** |
| 층B policy | 게이트-타입당 1 primitive — **단 타입이 유한이라는 경험적 가정** | **★1차 live seam**: G5 게이트타입 출현 시 P10. 적대 벤치탐색(§4 #6)의 정조준 대상. |

- **내 1차 패스 정정**: 나는 data-transform(S)을 "유일 seam"이라 했으나 **틀림**. §1.5(b)가 옳다 — 층A-data는 transform을 scope로 제외하면 *완전분할*로 강하게 닫히고(S는 명시 scope 결정), **남는 진짜 soft spot = 층B 게이트-타입 유한성**(우리가 본 G1-G4 밖 G5가 없다는 보장 없음). 적대 포화는 **층B를 친다**(층A는 정리로 잠겨 있으니).
- **그러나 S도 공짜 아님**: 층A-data의 "완전분할 강함"은 *S 상대*다 — 즉 층B의 "게이트타입 유한 상대"와 **같은 종류의 상대성**(하나는 scope-제외, 하나는 경험-미관측). 차이는 S는 *우리가 명시적으로 그은 경계*(§5 무거운 짐), G5는 *아직 안 본 것*. ⇒ **정직 진술**: 정리로 잠긴 축은 control 하나, data·policy는 각각 S·게이트유한에 *상대적으로* 닫힘.
- **그래서 census가 둘 다 시험**: §6.2 τ²/CDP census는 (1)*계산된 인자* 요구 task 수(=S 시험) (2)G1-G4 밖 게이트 유형(=층B 시험) **둘 다** 세야 한다. 0/0이면 닫힘 경험확정, >0이면 해당 축서 P10 발화·매트릭스 1행 추가(여전히 유한).

---

## 3. P8 ↔ P1 provenance 중복 — merge-후보 flag (net-new)
§1.5(b)는 P8 = G1(auth)+G3(single-user·값-출처∈{user,tool}), P1 = grounding(복사·날조금지)로 둔다. 그런데 **"값-출처∈{user,tool}"(G3 provenance) = P1 무날조의 음의 공간과 동일 명제**. ⇒ P8은 두 성분:
- (i) **provenance**(값∈{user,tool}) — **P1과 중복**.
- (ii) **auth-gate**(인증 선행·단일 활성유저) — P8의 *분리가능* 본체.

> **flag**: P8의 행이 (i)+(ii)를 합쳐 표기 중 = 약간의 과대. **leave-one-out ablation(§1.5a) 정조준**: P1 커버 시 P8-provenance가 전이되면 merge(분리 아님)·auth-gate는 별도 전이테스트. 본체 §1.5(b) "정직잔여 (ii) 분리학습가능성은 leave-one-out 필요"의 **구체 첫 타깃** = (P1, P8-provenance) 쌍.

---

## 3b. ★실증 census — seam β 해소 + P7 구조부재 확인 (net-new·zero-cost)
`tau2_primitive_census.py`(retail n=114·정적: gold action + `@is_tool(ToolType.WRITE)` 분류 + scenario). 모델 실행 0.
- **orphan 도구 = 0 / 114** — 전 gold 도구가 P1-P9로 매핑. **분류 밖 연산(P10) 0**(전수). §2 두 seam의 *경험* 시험 = 통과(이 도메인서).
- **요구 분포**: P1 112·P5 104·**P6 104(91%)**·**P2b 97(85%)**·P3 92·P8 66·P2a 52·P4 28·**P7 0(gold)**·**P9 0**.
- **★seam β 해소(τ²)**: 유일 변환도구 `calculate`(13/114)도 *tool-call* → 모델이 변환을 환경에 offload·결과를 **P2b로 소비**. ⇒ in-model 변환 primitive 불요·**β는 "변환 도구 부재 시에만 열림"**. 잘 설계된 벤치(τ²)는 변환을 도구로 제공 → β = scope 경계선이지 유한성 위협 아님. **§2 결론 강화: live seam = α(층B) 하나.**
- **★P7 구조부재 확인**: gold P7=0·잠재(unknown_info fallback) 89/114. §1 흡수(iter→무계 retry만 P7)·"성공-gold에 deny 없음"을 **census가 독립 확증** → P7 SFT-소싱 불가·gate-in-loop RL(리뷰#5) 재확인.
- **gap 재발(리뷰#4a/#4b)**: P6 91%·P2b 85% = task17 아티팩트 아닌 **전수 지배** → "남은 gap=P6(+P7)" 동어반복 탈출. ("정확히 N" 1차 방어 = orphan 0 + 요구집합 ⊆ {P1-P9}.)

## 3c. ★교차층 구조 — P7·P8 = A×B (net-new 부산물)
§3의 "P8 = provenance(=P1) ⊕ auth-gate"는 P8이 **층 A·B 교차**임을 함의. P7도 iter(A)×verdict(B) 교차. ⇒ 10셀 정연 분해:
| 분류 | primitive | 수 |
|---|---|---|
| 순수 층 A | P1·P2a·P2b·P3·P4·P9 | 6 |
| 순수 층 B | P5·P6 | 2 |
| **교차 A×B** | **P7**(iter×verdict)·**P8**(provenance×auth-gate) | **2** |
- **예측 적중(post-hoc 아님)**: 교차층 2개(P7·P8)가 정확히 *가장 어려운* primitive — P7=RL 필요(리뷰#5)·P8=fab/auth 실패 클러스터(autopsy 지배·census P8 66/114). **난이도가 구조(교차층)서 도출**됨. ⇒ 학습 우선순위 = 교차층(P7·P8) 최후·최난(현 gap과 정합).

## 4. 정직 경계 (본체 §7과 동일·재확인)
- 닫힘 = **구조적·상대적**(control=정리 잠김 / data=S 상대 / policy=게이트유한 상대). 학습가능성 증명 아님 — 전이는 ✓→✓! 경험 측정으로만.
- 도출의 효과 = 포화를 **"증명→확인"으로 강등**할 자격. 실증(적대 포화 §4#6 + held-out 전이)을 *대체*하지 않음.
- 유한 생성 = control/data-flow 슬라이스 한정(P1-P9). 계산·장기계획·GUI·세션메모리 = 도출 밖(§5 무거운 짐).
