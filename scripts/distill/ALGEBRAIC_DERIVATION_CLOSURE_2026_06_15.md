# 대수적 도출: control 정리-잠김 + data/policy scope-상대 닫힘 — 형식 companion (2026-06-15)
> ⚠️**제목 정직화(리뷰 #1)**: "닫힘"은 3층 중 **control만 정리-급**(Böhm–Jacopini 표현완전성). data=제약 S 상대·policy=게이트유한 상대(§2·§4). 무조건 "닫힘"은 과대 — 이 caveat가 제목·매트릭스 §1.5(b) 헤드라인에 동행해야 함.

> **도출 본체 = 매트릭스 `PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md` §1.5(b)** (Böhm–Jacopini control × provenance-완전분할 data × 게이트-타입 policy·닫힘 판정표·부산물 4·정직잔여 3). 이 문서는 그 도출을 *대체하지 않고* 형식적으로 보강하는 companion — net-new 3건만 담는다. 본체와 충돌 시 **본체가 권위**.

## 0′. ★전면화 (리뷰 2026-06-15 — 진짜 상 + 개수 잠정성)
- **★진짜 성과 = control 층 정리-급 닫힘(전면 배치)**: Böhm–Jacopini는 "임의 제어흐름 = {순차·선택·반복} 표현가능"(표현 완전성). primitive가 (가지치기 후) 이 basis를 덮으면 **새 control-flow primitive는 원리상 불가**(control 축 P10 불가능) = 경험적 포화 아닌 **정리**. 1/3 층에서 "whack-a-mole 끝" 증명. ⇒ §2 표 한 칸에 묻지 말고 헤드라인.
- **★유한성(robust) vs 개수(잠정) 분리(필독)**: whack-a-mole 반박 = *유한성*이지 *9*가 아님. 유한성 = control 정리 + provenance 완전분할로 **robust**. 정확한 개수 = leave-one-out 후 **~7–9 잠정**(merge 후보 = P1↔P8-provenance §3·**P3↔P9** §2-#2). merge가 풀려도 thesis 무손상 — 오히려 더 깨끗(개수↓=더 강한 압축). **개수-논쟁이 thesis를 흔든다는 인상 차단.**

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

> **★#2 (리뷰): par는 공리·BJ 아님 + P9 merge-후보.** Böhm–Jacopini = {순차·선택·반복} 3개뿐, **병렬은 정리에 없음** → "BJ+par"의 par는 bolt-on 공리(아래 표 "정리 잠김"은 par 성분엔 약간 과대). 더: 도구호출 병렬 = "데이터 의존 없는 호출집합" = **순서 제약의 부재** = P3(시퀀싱)의 뒷면. ⇒ P9의 내용 = "독립성 인식해 *가짜 순서 안 매김*" → **(P3,P9)를 (P1,P8-prov)와 같은 merge-scrutiny**(census P9=0[τ²]도 P9=default/null 시사). 개수 ~7-9 잠정(§0′).

| 축 | 닫힘 근거 | seam 지위 |
|---|---|---|
| 층A control | Böhm–Jacopini(순차·선택·반복=정리) **+ par(공리 bolt-on)** | **잠김**(정리, par 제외) — seam 아님 |
| 층A data | provenance 완전분할 — **단 transform 제외(제약 S) 하에서** | **2차 seam**: S는 scope-자인이지 정리가 아님. 실타깃이 *계산된 인자*(obs→fn→arg)를 요구하면 P10 발화. **census 조건부.** |
| 층B policy | 게이트-타입당 1 primitive — **단 타입이 유한이라는 경험적 가정** | **★1차 live seam**: G5 게이트타입 출현 시 P10. 적대 벤치탐색(§4 #6)의 정조준 대상. |

- **내 1차 패스 정정**: 나는 data-transform(S)을 "유일 seam"이라 했으나 **틀림**. §1.5(b)가 옳다 — 층A-data는 transform을 scope로 제외하면 *완전분할*로 강하게 닫히고(S는 명시 scope 결정), **남는 진짜 soft spot = 층B 게이트-타입 유한성**(우리가 본 G1-G4 밖 G5가 없다는 보장 없음). 적대 포화는 **층B를 친다**(층A는 정리로 잠겨 있으니).
- **그러나 S도 공짜 아님**: 층A-data의 "완전분할 강함"은 *S 상대*다 — 즉 층B의 "게이트타입 유한 상대"와 **같은 종류의 상대성**(하나는 scope-제외, 하나는 경험-미관측). 차이는 S는 *우리가 명시적으로 그은 경계*(§5 무거운 짐), G5는 *아직 안 본 것*. ⇒ **정직 진술**: 정리로 잠긴 축은 control 하나, data·policy는 각각 S·게이트유한에 *상대적으로* 닫힘.
- **그래서 census가 둘 다 시험**: §6.2 τ²/CDP census는 (1)*계산된 인자* 요구 task 수(=S 시험) (2)G1-G4 밖 게이트 유형(=층B 시험) **둘 다** 세야 한다. 0/0이면 닫힘 경험확정, >0이면 해당 축서 P10 발화·매트릭스 1행 추가(여전히 유한).

### 2.1 ★★두 seam의 진짜 해소 = ABox/A2 재배치 (사용자 통찰 2026-06-15 — §2 재진술, 위 "경험으로 메울 구멍" 프레임 격상)
> **핵심 전환**: 두 열린 seam(data-transform·policy-게이트타입)은 **"모델 스킬-basis의 구멍"이 아니라 "ABox/A2로 올바로 격리된 도메인 *내용*"이다.** 무한은 사라지지 않고 *전이 TBox가 아닌 swap ABox*로 이동 — 도메인 지식이 있어야 할 정확한 자리.

- **공통 성질**: 두 seam 모두 도메인 *내용*(content)이지 전이가능 *스킬*이 아니다.
  - data `computed`(임의 변환·Turing-완전) = `is_authenticated` 파싱·금액 산술·날짜 정규화 → **per-domain authored `compute` 노드**(§14 bounded·audited pure-fn·§14.6 "사전 authoring 결정론 자산").
  - policy 게이트-타입(HRU 무한) = G5("예산초과→승인자2") → **그 도메인 A2/GATE_SPEC**에 인코딩.
- **★구성상 유한 논증(경험 포화보다 강함)**: 모델 쪽 *상호작용 어휘*가 유한이라 **새 변환·새 게이트가 새 *모델* primitive(P10)를 강제할 수 없다** — 강제 가능한 건 새 *A2 항목*뿐.
  - 임의 compute여도 모델은 *호출+출력 threading* = **P2b**(census 실증: τ² `calculate`→도구offload→P2b 소비).
  - 새 게이트(G5)여도 모델은 *제안→deny 관찰→복구(승인자 fetch/ask)* = **P5+P7+P2a**. 게이트의 *내용*(precond X)은 임의(A2)나 모델의 *충족 수단*(fetch/ask/선행행동/confirm)은 유한.
  - ⇒ **"G5→P10" 질문 해소**: G5는 P10이 아니라 A2 항목 하나를 요구. live seam(§2 표)이 "경험적 위협"에서 "A2 표현가능성 질문"으로 강등.
- **★단 하나의 구분(섞으면 과대) — A2 *소비* vs *생산***:
  - (a) A2 **소비**(주어진 GATE_SPEC/카탈로그 안에서 제안·복구) = **P1+P5/P6/P7 = 이미 유한집합 안. 새 R 불요.** ← 닫힘 성립 영역.
  - (b) A2 **생산**(정책 NL→GATE_SPEC 컴파일) = **NL→구조 front-end = thesis core 미해결**(§17.9·E1). **이 닫힘 논증이 덮지 않음.**
  - = **2-prong**: prong②(OISA·A2 authored→모델은 소비만→유한 primitive 충분) / prong③(학습 front-end가 A2 자동생산→hard 열림). "A2 쓰는 R이면 충분"은 **(a)에서 정확히 참**.
- **CDP census 재정의(정련)**: "모델 basis 완전한가"는 이 논증으로 닫힘 → census가 실제 시험하는 것 = ①**A2가 CDP 게이트·compute를 표현가능한가**(prong② authoring-feasibility) ②**front-end가 CDP NL서 A2 생산가능한가**(prong③ 진짜 hard). 더 정직·정확한 시험.
- **deterministic 불변 정합**: 게이트 집행(A2+scaffold)·compute 모두 결정론 사전자산. 모델은 생성기 역할만(게이트로직 생성 0) = `feedback-selector-verifier-deterministic` 부합.

---

## 3. P8 ↔ P1 provenance 중복 — merge-후보 flag (net-new)
§1.5(b)는 P8 = G1(auth)+G3(single-user·값-출처∈{user,tool}), P1 = grounding(복사·날조금지)로 둔다. 그런데 **"값-출처∈{user,tool}"(G3 provenance) = P1 무날조의 음의 공간과 동일 명제**. ⇒ P8은 두 성분:
- (i) **provenance**(값∈{user,tool}) — **P1과 중복**.
- (ii) **auth-gate**(인증 선행·단일 활성유저) — P8의 *분리가능* 본체.

> **flag**: P8의 행이 (i)+(ii)를 합쳐 표기 중 = 약간의 과대. **leave-one-out ablation(§1.5a) 정조준**: P1 커버 시 P8-provenance가 전이되면 merge(분리 아님)·auth-gate는 별도 전이테스트. 본체 §1.5(b) "정직잔여 (ii) 분리학습가능성은 leave-one-out 필요"의 **구체 첫 타깃** = (P1, P8-provenance) 쌍.

---

## 3b. ★실증 census — seam β 해소 + P7 구조부재 확인 (net-new·zero-cost)
`tau2_primitive_census.py`(retail n=114·정적: gold action + `@is_tool(ToolType.WRITE)` 분류 + scenario). 모델 실행 0.
- **★orphan 도구 = 0 — 전 τ² 도메인 전수**: retail(114)·airline(50)·telecom(2285·dual-control)·mock 모든 gold 도구(~2450 task)가 P1-P9로 매핑. **분류 밖 연산(P10) 0**. 도구 분류=각 도메인 `tools.py`+`user_tools.py`의 `@is_tool(ToolType.WRITE)` 동적 파싱(반환시그니처 원칙). §2 두 seam의 *경험* 시험 = 통과(전 도메인).
  - ⚠️**#4 (리뷰) 주장범위 정밀화**: orphan=0은 **도구-분류 커버리지**(개별 도구∈P1-P9) — **스킬-조합 커버리지**(task 요구 *패턴* call→observe→branch→confirm ∈ P1-P9)는 한 단계 강한 주장. 요구분포(P1 112…)가 스킬-수준에 근접하나, **그 task별 primitive 태깅이 수동검증인지 도구-타입 자동도출인지 명시 필요**(후자면 도구-수준 한계 상속). 현 census = 도구-커버리지 확정·스킬-조합은 요구분포로 *근사*.
  - ★**#6 (리뷰) telecom device-actuation 경계**: `toggle_*`·`reboot_device` write도 P5/P6 매핑. **in/out 기준 = 구조화 tool-call(=in) vs 픽셀/DOM 조작(=out)** — device-actuation은 API 호출이라 **명확히 in**(GUI-grounding 제외축과 구분됨). "GUI-인접" 모호표현 폐기·이 기준으로 대체.
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
- **구조-난이도 상관 = 일치(N=2, 적중 아님 — #5 리뷰 정정)**: 교차층 2개(P7·P8)가 가장 어려운 primitive — P7=RL 필요·P8=fab/auth 실패 클러스터(autopsy 지배·census P8 66/114). 구조적 정의(P7=iter×verdict·P8=provenance×auth)는 난이도와 **독립**이라 genuine prediction 자격은 있음. ⚠️**단 교차층 셀=2개뿐·둘 다 어려움 → 반증 불가**(쉬운 교차층 셀 부재로 "교차층→어려움" falsify 불가). ⇒ "예측 적중·post-hoc 아님" → **"구조-난이도 상관과 일치(N=2)"로 강등**. 학습 우선순위(교차층 최후·최난)는 현 gap과 정합하나 N=2 근거.

## 4. 정직 경계 (본체 §7과 동일·재확인 — ★§2.1 재배치 + 딥리서치 반영 2026-06-15)
- ⚠️**구판("data=S 상대 / policy=게이트유한 상대")은 §2.1로 격상**: 열림은 *모델 스킬-basis*가 아니라 *A2 내용*에 있음. **모델 basis는 닫힘**(상호작용 어휘 구성상 유한). 단 A2-생산(NL→구조 front-end·prong③)은 별개 미해결.
- 도출의 효과 = 포화를 **"증명→확인"으로 강등**할 자격. 실증(적대 포화 §4#6 + held-out 전이)을 *대체*하지 않음.
- 유한 생성 = control/data-flow 슬라이스 한정(P1-P9). 계산·장기계획·GUI·세션메모리 = 도출 밖(§5 무거운 짐).

### 4.1 ★딥리서치 — data/policy 정리-닫힘 선행연구 (2026-06-15, 3클러스터 1차소스. ⚠️일부 verbatim PDF 미추출=secondary 교차확인·박제 전 원문 재검)
- **data 축 = 부분 정리-닫힘 + 원리적 잔여**:
  - **출처·결합·추출·배선 = 정리-닫힘**: Green–Karvounarakis–Tannen *provenance semirings*(PODS'07) — 다항식반환 **ℕ[X] universal**(모든 provenance 모델이 factor through), +/×=대안/결합 유도 닫힌 대수(threading=×). **Codd 정리**(rel-algebra=rel-calculus=FO·유한 연산자 완전) = data축 Böhm–Jacopini 최근접 유사물. **SSA** 참조투명성=배선 정규형. **Denning 격자**(1976)=출처라벨 join 닫힘(`fc_d5` 라벨전파와 일치).
  - **`computed` 분할 = 원리적 열림**: 임의 변환=Turing-완전 → **Church–Turing/Rice로 유한 기본형 닫힘 불가**. ⇒ **§2 제약 S가 "임의 scope결정"에서 "Turing-완전성이 강제하는 경계"로 격상**(훨씬 방어적). ⚠️"computed=Turing이라 닫힘불가"는 *우리 도출*(특정논문 정리 아님·인용없는 우리주장 표기).
- **policy 축 = 쪼개진 정리(당신 구분 정확)**:
  - **집행가능성=정리-닫힘**: Schneider *Enforceable Security Policies*(TISSEC'00) — 런타임 모니터(EM) 집행가능 = **정확히 safety property**. G1-G4 전부 prefix-closed safety → **집행가능 증명**. (edit-automata[Ligatti]·BLS monitorability는 safety 넘어 확장=P7 recovery 자리.)
  - **게이트-타입 유한성=정리 불가(부정결과)**: **HRU(1976)** 일반 access-matrix safety **결정불가능** → 일반 정책공간 정리-닫힘 불가 = "게이트유한"이 보편정리일 수 없는 *이유*(우리 실패 아닌 근본장벽).
  - **탈출구=타입제약**: **Sandhu TAM**(1992) typed·monotonic·acyclic → 결정가능(ternary 다항시간) = **우리 "타입→인자타입→선행조건→replay" 결정론 검증기와 동형**. ⇒ G5부재 = 보편정리 아니라 **TAM-제약 하 산물**(§2.1 ABox-재배치와 정합: 게이트내용=A2, 모델상호작용=유한).
- **빈 칸 + 위험 인용**: "control·data·policy 도출 + 벤치횡단 전이 완전성검증"은 **무주공산**(BFCL/τ-bench=경험 카테고리·완전성 무주장). 차용=component-synthesis 상대완전성[Jha-Seshia]·Böhm–Jacopini[IPARC `2506.13820` 인용만]. ⚠️**위험 인용 = `2510.06002` "Deterministic Legal Agents: Canonical Primitive API"**(deterministic+primitive+provenance 키워드 3개 동시·단 법률도메인 손나열·도출/전이 0) → **FIELD_GAP §5.5에 명시 차별 문장 필수**(리뷰어 "이미 했다" 오해 차단).
