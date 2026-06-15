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

---

## 5. ★★소진 보조정리 형식화 — "카탈로그 차용자"에서 "조건부 closure 정리 보유자"로 (2026-06-15)
> 동기 = `2510.06002` 정독(법률 카탈로그·정리 0건) 대비 우리 형식성 위치 확정 + 메모리 "대수적도출 2층 closure=최고가치미완"의 **모델-쪽 절반을 §2.1로 증명**. 이 §은 *준-형식* 증명(워크숍 rigor; Coq 미검증). 차용정리/우리-신정리/경험을 **엄격 구분**.

### 5.1 프레임 (정의)
- **궤적** τ = (m₀,o₀,m₁,o₁,…): mᵢ=에이전트 move(tool-call f(a₁..aₖ) | ask-user | terminal), oᵢ=관측(tool 출력 또는 user 발화).
- **가용집합** Aᵢ at step i = (역할=user인 o_{<i}) ∪ (tool 출력 o_{j<i}) ∪ (스키마 상수 어휘 A1). *유한*.
- **primitive(competence)** = 도메인-독립 ∧ 분리학습가능(§1.5a-ii)한 *다음-move 산출 하위결정*.
- **아키텍처 전제(우리 설계 = prong②·결정론 불변, 명시)**: (i) 계산은 결정론 도구/A2로 offload(compute-as-tool) (ii) 게이트는 A2-인코딩 spec을 결정론 scaffold가 replay 집행 (iii) 행동 = 유한 스키마 A1 위 tool-call/ask/terminal emit.

### 5.2 ★보조정리 1 (Provenance 완전성) — **우리 신정리·완전증명**
**진술.** step i의 tool-call 인자값 v에 대해, 우선순위 U≻T≻K≻C로 다음이 *상호배타·전수망라*:
- (U) v가 user 입력 o_{<i}에 **verbatim** 등장
- (T) v가 선행 tool 출력 o_{j<i}에 verbatim 등장
- (K) v가 스키마 상수(A1)
- (C) 위 어디에도 없음(=모델-내부: 변환 또는 날조)

**증명.** v는 유한 문자열. Aᵢ는 유한. "v ∈ verbatim-가용(Aᵢ)"의 배중률: 참이면 최고우선 출처(U≻T≻K)에 배정, 거짓이면 정의상 C. 배타는 우선순위로, 망라는 배중률로. ∎

**따름.** {U,T,K}=유한 primitive {P1 ground, P2a/P2b from-obs, P4 select}가 처리. C는 둘로:
- **C-computed** (v=g(Aᵢ), g=도메인 변환): 전제(i)로 **compute-도구 호출 후 T로 재진입** → 모델은 C-computed를 *직접 emit 안 함*. (정규화 "tomorrow"→date도 C=offload.)
- **C-fabricated** (provenance 0): = P1 위반 = **금지 대상**.
⇒ **모델이 emit해야 할 인자값은 {U,T,K}(유한 primitive) ∪ offload된 T뿐; 5번째 data-competence 없음.** Turing-잔여(`computed`)는 전제(i)로 A2/도구에 격리. □

### 5.3 ★보조정리 2 (게이트-상호작용 닫힘) — **우리 신정리·구조적 귀납**
**진술.** 게이트 g=(선행조건 술어 X, 충족자집합 S_X). 모델의 g-상호작용 competence는 X 내용과 *무관하게* {P5 respect-deny, P6 confirm, P7 recover} ∪ (이미 계상된 establishment primitive)로 닫힌다 — 어떤 새 게이트타입 G5도 새 모델 primitive 불요.

**증명(구조적 귀납).** 모델의 게이트 상호작용 = "move 제안 → deny 관측 → X 확립 → 재시도". X 확립 = 세계상태를 X-참으로 변경 또는 결손정보 제공. 세계상태 변경 수단 = 유한 도구집합 A1의 tool-call(=서브궤적). 정보제공 = ask/fetch. ⇒ "X 확립"은 **같은 basis 위 서브궤적**으로 재귀 → 구조적 귀납으로 basis 내 닫힘(바닥=단일 fetch/ask/prereq-action/confirm). 게이트의 *내용*(X,S_X)=도메인 데이터(A2·HRU-wild), 모델의 *상호작용*=관측-deny-확립(유한). ∎

**따름.** G5는 A2 항목(X,S_X) 하나를 요구·**P10 불요**(§2.1 "G5→P10 해소"의 형식 근거). Schneider EM=safety로 g 집행가능성은 차용-정리로 닫힘; 본 보조정리는 그 위 *모델-쪽* 닫힘. □

### 5.4 ★control 소진 (차용 BJ + 흡수)
Böhm–Jacopini(차용): 임의 제어흐름 = {순차·선택·반복} 구조적 합성. 흡수(§1·정의적): 선택=술어 grounded면 자명(→P2a/P5 흡수)·유계반복=P4×(P3|P9) 흡수·무계retry만 잔존(→P7). par=독립성(공리). ⇒ control-competence ⊆ {P3,P9, P7-제어부}. *차용정리 + 흡수(분리학습가능성 기준 상대)*.

### 5.5 ★정리 (Tool-Use Skill-Basis 소진, 조건부) — **우리 정리**
**전제** 5.1-(i)(ii)(iii) 하에서, 올바른 에이전트 move 산출에 필요한 도메인-독립·분리학습가능 competence 집합은 **유한**이며 다음의 닫힘과 같다:
- control {P3,P9} (BJ+흡수) · data {P1,P2a,P2b,P4} (보조정리1+offload) · policy {P5,P6,P7} (보조정리2) · 교차 P7=iter×verdict·P8=prov×auth.
**어떤 도메인 내용도 이 닫힘 밖 competence를 강제하지 않는다**(새 변환·새 게이트타입 = A2 항목만 요구).
**증명개요.** "다음 move 산출"의 하위결정 = control(5.4 BJ 망라)·data(보조정리1 망라)·policy(보조정리2 망라) 3분할이 전수; 전제(i)(ii)가 유일 비유한 잔여(Turing-computed·HRU-wild 게이트타입)를 모델-쪽서 A2로 제거. ∎

### 5.6 ★엄격 3분 — 무엇이 정리이고 무엇이 열렸나 (정직)
| 층위 | 내용 | 지위 |
|---|---|---|
| **차용 정리** | BJ(control)·provenance-semiring ℕ[X]·Codd·Schneider EM=safety·HRU·Sandhu TAM | ✅ 증명된 수학(차용) |
| **우리 신정리(준-형식)** | **보조정리1 Provenance 완전성**(완전증명)·**보조정리2 게이트-상호작용 닫힘**(구조적 귀납)·정리 5.5(조건부 소진) | ◐→✅ **모델-쪽 closure = 본 §이 닫음**(메모리 "2층 closure 미완"의 절반 해소). Coq 미검증=워크숍 rigor |
| **정의적/상대** | 흡수(§5.4)·분리학습가능성 개별화 기준 | leave-one-out 경험 확인 필요(개수 P1/P8·P3/P9 merge — *유한성 불변·개수만*) |
| **아키텍처 전제** | (i)compute-as-tool (ii)gate-as-A2(scaffold) (iii)유한 A1 | 우리 *설계*(prong②·결정론 불변)·증명 아닌 명시 가정 → 정리는 *조건부* |
| **순수 경험** | 벤치횡단 전이 완전성(basis 학습→전이) | ✗ 정리 아님·반증가능. 현 τ² 음성(0.0–0.105<base0.17)=미지지 → "basis 닫혔으나 전이학습 미흡" vs "basis 불완전" 구분이 v7/P6·P7 |

### 5.7 ⇒ 위치 확정 (vs 2510.06002)
- 2510.06002 = 손-나열 카탈로그·정리 0·완전성 무주장(v3 "representative core subset"·"does not claim every interaction deterministic").
- **우리 = 차용정리 위 *연역*을 넘어 모델-쪽 *closure 정리*(보조정리1·2 + 정리5.5) 보유.** "증명했다" 선 = **차용정리 + 보조정리1·2 + 조건부 정리5.5(모델 skill-basis 유한)**. *열린 것* = (a)개수(merge·유한성 불변) (b)아키텍처 전제의 경험 타당성(=CDP census: A2가 compute·gate 표현가능?) (c)전이 완전성(경험·현 음성).
- **상승 판정**: "카탈로그 차용자"→**"조건부 closure 정리 보유자"**. 미완 = 전제(i)(ii)를 *경험 사실*로 승격(census)하고 전이를 *양성*으로(P6/P7·front-end). 이 둘은 정리 아니라 측정 — 정직 보존.

### 5.8 ★★전제 (i)(ii)의 격상 = 구성가능성 정리 (사용자 통찰 2026-06-15) — census→정리
> 핵심(사용자): 전제 (i)(ii)를 도메인마다 *관찰*(census)할 게 아니라 *구성*할 수 있다. "∀ 도메인에 그런 A2가 존재한다"는 **경험이 아니라 수학(존재 정리)**. ⇒ §5.5의 "아키텍처 *가정*"을 "도메인-범위 조건 하 *구성 정리*"로 격상.

**보조정리 3 (compute-offload 구성가능성) — 우리 신정리·구성적.**
- **진술.** 도메인 D의 요구 변환집합 G_D가 유한이면, 모든 변환을 offload하는 유한 compute-도구 카탈로그가 존재 → 전제 (i) 성립.
- **증명.** 임의 계산가능 g에 도구 T_g(x)=g(x) 정의(함수→도구 wrap은 무장애). G_D 유한 → {T_g : g∈G_D} 유한 카탈로그가 전 변환 offload. 복잡도는 T_g(=A2)로, 모델 스킬은 P2b(호출+threading) 하나로 유지. ∎
- **주의(유한성 소재)**: 유한해야 할 건 *함수*이지 *입력* 아님("정렬"은 입력 무한해도 함수 하나). **G_D 유한 = 정책/SOP가 유한문서 = "도메인"의 정의**(무한·근거없는 spec으로 도메인 운영 불가) → census 아닌 정의에서 닫힘.

**보조정리 4 (gate 표현가능성 구성) — 차용(Schneider) + 구성.**
- **진술.** D의 선행조건이 safety property(또는 의무→선행조건 변환 가능)면, 전부 집행하는 A2 게이트-spec 집합이 존재 → 전제 (ii) 성립.
- **증명.** 각 safety X에 Schneider EM-집행 모니터 존재(차용) → 게이트-spec 인코딩; 의무형은 Ligatti edit-automata 변환. G1-G4 전부 safety. ∎
- **★증명된 잔여**: 진짜 liveness("언젠가 환불")는 유한시점 deny 불가 = **Schneider가 증명한 제외**(경험 구멍 아닌 정리 경계 — 무엇이 왜 빠지는지 *앎*). 실무 의무는 대부분 마감有(bounded→safety)라 드묾.

**★정리 5.8 (구성가능성) — 우리 정리.**
> 도메인 D가 **(a) 유한 정책**(유한 G_D + 유한 선행조건) **∧ (b) safety-표현가능**이면, (i)(ii)를 성립시키는 유한 A2(D)가 **존재**(보조정리3·4). ⇒ **따름: 범위 (a)+(b) 안 임의 도메인서 모델 skill-basis 유한** — 정리5.5의 *아키텍처 조건이 소거*되고 조건은 "(a)유한정책 + (b)safety"로 줄며, **둘 다 census 아닌 정리(유한성·Schneider)로 특성화.**

**효과 (census 강등).** census는 "전제 참인가"(경험)에서 **"이 도메인이 정리 범위인가=유한정책+safety인가"(분류·near-정의적)**로 강등. (i)는 "유한정책=도메인 정의", (ii)는 "Schneider safety"에서 닫힘.

**★부산물: prong② 보편가능 증명.** "A2(D) 존재"=손-authoring(OISA·prong②)이 **임의 in-범위 도메인서 보편 가능함이 증명됨**(기존엔 가정).

**★엄격 정직 — 5.8이 *안* 닫는 것 (섞으면 과대)**:
- (1) **전이는 여전히 경험.** 구성가능성 = "basis 유한 ∧ A2 만들 수 있다"; "그 basis 학습 모델이 전이된다"는 미포함 = v7/P6·P7 측정(현 음성). **finiteness 닫고 learning 안 닫음.**
- (2) **A2 *자동생산*(NL→spec)은 여전히 prong③.** 정리는 A2 *존재*(손-도달 보편가능)를 줄 뿐, NL서 *자동 컴파일*하는 front-end는 별개 학습문제. ⇒ 깔끔 분리: 손-구성=보편가능(증명)·자동구성=열린 학습.
- (3) **liveness 잔여** = 증명된 제외(강점이나 범위 한정).
⇒ **무조건화 잔벽 = 이제 "유한성"이 아니라 "전이(학습)" 하나.** 종이 아닌 v7이 답함.

**3분 갱신(§5.6 보강).** 우리 신정리 = 보조정리1(provenance 완전·증명)·2(게이트-상호작용·귀납)·**3(compute-offload 구성·증명)**·정리5.5(조건부 소진)·**정리5.8(구성가능성=전제 (i)(ii)를 도메인-범위 조건으로 격상)**. 차용 = 보조정리4의 Schneider/Ligatti. 잔여(경험) = **전이 하나로 축소**(+개수 merge·front-end prong③).

### 5.10 ★★전이-축 분해 — σ는 증명·γ는 등방화로 구성·잔여는 coverage (사용자 통찰 2026-06-15)
> 메모리 "무조건화 잔벽=전이"를 정밀 위치확정. 핵심: 전이 ≠ "두 DAG라 자동". 분해 후 *증명 가능 부분*과 *구성 가능 부분*과 *환원불가 핵*을 분리.

**분해.** 최적정책 π* = **γ ∘ σ**: σ=추상 solver(의존그래프 D+상태→다음행동, 도메인-무관 고정 알고리즘)·γ=grounding(도메인 *표면*[도구명·NL·값형식]→추상 D+역할 매핑, 도메인 의존).
- **σ 전이 = 수학(증명됨)**: 두 도메인이 유한 DAG면 추상 D 형태 동일·σ 동일 → 새 R 불요·도메인2 정답 같은 basis로 표현(§5.5/5.8 realizability). 사용자 "두 DAG면 R이 푼다"는 **이 추상 수준서 정확**.
- **★갭(학습된 모델)**: DAG-동형은 *추상 D*에서지만 모델은 *표면*서 동작·γ로 D 복원. 학습모델 전이 = **γ가 표면-불변인가**에 의존 — DAG-동형이 표면 갭을 메우지 않음.

**★반례(우리 데이터) = realizability ≠ generalization.** TaskBench LODO: HF·MM·daily 전부 동일 추상구조(tool-DAG)인데 전이 **실패**. 원인(census 진단) = **출력-포맷 간섭**(resource vs temporal 직렬화) = *표면 과적합*이지 빠진 primitive 아님. ⇒ "추상-동형 ∧ 표면-상이 → 전이 실패" 실증. no-free-lunch: 분포-이동 가정 없이 학습기 일반화 선험 증명 불가.

**★표면-불변성은 구성 가능 = 등방화 (사용자: 다양성/무작위화).** 두 경로가 같은 불변에 수렴:
- 데이터-적응 = 다중 실제 도메인 학습(표면 변동 제공) · 데이터-무관 = 무작위 surface-augmentation(alias/value/format-randomization).
- **기제**: 표면 feature가 도메인 간 변하면 신뢰 예측자 못 됨 → 불변(추상)만 학습가능.
- **1차 수학 (직접 인용 — 2차 응용물 미인용)**:
  - **무작위 회전→등방 Gaussian marginal**: Poincaré–Maxwell–Borel; 엄밀 정량판 = **Diaconis & Freedman 1984, Ann. Stat. 12(3):793–815** (√d·S^{d-1} 균등분포의 처음 k좌표 → N(0,I_k), TV거리 O(k/d)). = "random rotation이 좌표 등방화"의 직접 수학.
  - **거리 등방보존**: **Johnson–Lindenstrauss 1984, Contemp. Math. 26:189–206**(무작위 직교투영, (1±ε)‖·‖² 보존).
  - **측도집중**: **Lévy–Milman**(Milman–Schechtman 1986, LNM 1200): 고차원 구면 1-Lipschitz 함수 집중 σ(|f−M_f|>ε)≤2exp(−(n−1)ε²/2) = "표면변동 평균화"의 기저원리.
  - **다환경→불변 feature**: **IRM (Arjovsky–Bottou–Gulrajani–Lopez-Paz 2019, arXiv:1907.02893)**(모든 환경서 동시최적인 표현=불변→OOD 일반화) · **ICP (Peters–Bühlmann–Meinshausen 2016, JRSS-B 78(5):947–1012)**(다환경 예측-불변=인과 식별).
- **⚠️★엄격 선긋기 (인용규율)**: 위 정리들은 **원입력 벡터좌표의 등방화·거리보존·집중**(①②③)과 **환경 간 예측/변수 불변성**(④)을 증명. **"신경망 *은닉표현*의 표면-feature를 augmentation으로 등방화→추상 불변 학습"의 연쇄는 어느 논문도 증명 안 함 = 우리 *유추적 동기*(analogy)**. 표현은 균등구면분포 아니고 학습으로 결정·IRM도 불변을 *강제*할 뿐 무작위화가 자동 *유도*함을 안 보임. 박제·논문서 §1–4=직접정리, 등방화→전이 연쇄=명시적 analogy로 표기.

**★조건 — 등방화는 *덮인 방향*에서만(LODO 실패의 정체).** 다양성은 학습도메인 *사이 변하는* 표면만 벗김; *모든 학습도메인 공유∧테스트서 다른* feature는 안 벗겨지고 강화. LODO: 학습 둘 다 resource→"resource 포맷"이 학습 내 불변→안 벗겨짐→temporal 테스트 실패. (유한표본 단일회전=잔여 비등방성; 등방성은 점근.) ⇒ **"다양성 늘리면 등방"은 학습 다양성이 테스트 표면-변이를 *덮을(span)* 때만.**

**⇒ 위치확정(전이 잔벽의 해부).**
| 부분 | 지위 |
|---|---|
| 추상 σ | **전이=수학**(증명·realizability) |
| 표면 γ, 덮인 방향 | 등방화로 전이(다양성/random-aug·1차수학 유추) |
| 표면 γ, 안 덮인 방향 | 잔여 비전이(LODO 포맷=미덮음) |
| 환원불가 핵 | "학습 다양성이 임의 unseen 표면 덮나"=no-free-lunch·선험 보장 불가 |
- **공학가능 처방**: 안 덮인 *알려진* 차원(포맷·naming·값형식)을 합성회전으로 덮음 = format-uniform + alias-mask + value-rand → 잔여를 *진짜-신규* 표면차원(드묾)으로 축소. **LODO 처방(uniform-schema)을 등방화 framing이 예측**.
- **결론**: 전이 = "두 DAG라 자동" 아니라 **"표면 벗기면 자동(덮인 방향)·벗기는 건 우리가 구성(합성회전)·임의 unseen 표면만 환원불가 경험"**. 무조건화 잔벽 = 이제 "전이 일반"이 아니라 **"테스트 표면-차원의 학습-coverage" 하나로 정밀 축소**.
