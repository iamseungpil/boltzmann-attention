# Content 생성원 대수 — 연역적 도메인-독립 연산자 집합 + 다중벤치 닫힘 (설계) — 2026-06-17

> 상위 = `PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md`(flow 축 P1-P9·닫힘 도출됨) · `ALGEBRAIC_DERIVATION_CLOSURE` · `EXPRESSION_DIVERSITY_TRANSFER_DESIGN`(𝔥 표면축). 동기 = 사용자(2026-06-17): 생성원을 *벤치 귀납 아닌 연역*으로 규정·도메인독립·거의 모든 벤치 닫힘. + substitute "retail 특화" 공격 방어.

## 0. ★두 생성원 축 (핵심 프레임)
tool-use 절차 = 두 직교 축의 생성원으로 분해:
- **Flow 생성원 (P1-P9·`PRIMITIVE_COVERAGE_MATRIX`)**: *언제·어떻게* 도구를 엮나(sequence/branch/loop/par + provenance + gate). **대수적 닫힘 도출됨**(Böhm–Jacopini + provenance 완전분할 + 유한 게이트타입).
- **Content 생성원 (이 문서·op-IR)**: 데이터에 *무슨 연산*(select/aggregate/relative/transform). `PRIMITIVE_MATRIX §5`가 **명시적 scope-out**("transform 연산자 없음·data축=copy/select만"). = 우리 op-IR 작업이 채우는 누락 축.
- **τ² substitution 실패 = content 축**(flow는 닫혔으나 content 미도출). LLM이 content op를 *명명*·결정론 엔진이 실행(offload).

## 1. Content 생성원 연역 (벤치 아닌 연산 taxonomy에서)
출처 = 관계대수(Codd) + 함수형 변환 + 순서대수:
| 생성원 | 연산 | 연역 출처 |
|---|---|---|
| **filter** | 조건으로 항목 pin | σ select |
| **argmax/argmin** | 극값 | γ aggregate-extremum |
| **rank-k** | k번째 정렬 | sort∘select |
| **comparative** | anchor 상대 (just-above/below) | order relation |
| **substitute** | 기존 객체 일부 필드 override·나머지 유지 | record update |
| **create** | 빈 base에 필드 설정 (신규 객체) | record construct (substitute의 base=∅) |
| **project** | 출력서 필드 추출 | π project |
- **command-trivial**(toggle/invoke·인자 없음) = content-identity = **flow 축**(P-something), content 생성원 아님.
- **scope-out(정직)**: 수치/기호 *계산*(sum·format·산술)은 제외 — tool-call로 환원(`PRIMITIVE_MATRIX` seam β) 또는 별 축.

## 2. 도메인 독립성
- 생성원 = 순수 연산(item_id·zoom·mode 같은 도메인 어휘 0). **ABox config가 도메인 채움**(attr 타입·값 vocabulary).
- 학습 = 등방화 합성으로 생성원 *라우팅*(NL→어떤 content op). 도메인 사실 X. 전이 = ABox swap.

## 3. ★tau2 5도메인 실증 (substitute 도메인-일반 = 공격 반박)
| content op | retail(114) | airline(50) | telecom(2285) | banking(97) | mock(10) |
|---|---|---|---|---|---|
| selection(filter/argmax/comparative) | ✓✓(sup 62) | ◐(sup 10) | enum | ✗ | ✗ |
| **substitute** | ✓✓(exchange/modify·subst-kw 82) | **✓✓(update_reservation_*·subst-kw 18)** | ✓(set_mode=X single-field) | ✗ | ✓(update_task) |
| **create** | ✗ | ✓(book_reservation) | ✗ | ✓(apply_credit_card) | ✓(create_task) |
| command-trivial(flow) | ✗ | ◐ | ✓✓(toggle/reboot/grant) | ✓✓(call/unlock/verify) | ◐ |
- **★substitute는 retail·airline 둘 다 지배**(airline `update_reservation_flights/baggages/passengers` 전부 substitution)·telecom degenerate(single-field set)·mock도. = **트랜잭션 도메인 일반**(retail 특화 *아님*) = 사용자 공격 직접 반박.
- **content-rich(retail·airline) vs content-light(telecom·banking)**: telecom/banking = command/auth 지배(content-trivial) → flow 축(P1-P9)이 답. op-IR은 content-rich 벤치용.

## 4. 닫힘 + create 분리 검증
- **content 생성원 ~7개**(filter·argmax·argmin·rank·comparative·substitute·project·+create) 가 tau2 5도메인 content 연산을 *닫는 신호*(P10-content 부재·5도메인 전수).
- **create vs substitute 분리(leave-one-out)**: create=base∅ override·substitute=existing override. 별 생성원인지 = create 미커버 학습 → substitute서 전이되면 병합·안 되면 분리(§1.5a flow와 동일 기준).
- **닫힘 = 경험적**(선험 아님): 벤치 추가 시 새 content-op 수 → 0 곡선. SOPBench/TaskBench/CFB content 연산 매핑 추가 필요(다음).

## 5. ★공격 방어 (연역·유한·닫힘)
> "substitute는 τ² retail 특화 아니냐" → **"아니다. (1)substitute=관계대수 record-update, 벤치 보기 전 연산 taxonomy서 연역. (2)airline update_reservation·telecom set·mock update가 전부 substitute=도메인 일반 실증. (3)~7 content 생성원이 5도메인 content를 닫음(새 op 0)."**
- 핵심 = 생성원을 *벤치 귀납*(특화·공격O) 아닌 *taxonomy 연역*(일반·공격무효)으로.
- **위험 신호(정직)**: content-rich 벤치 추가마다 새 content-op면 무한→thesis 위험. 닫힘은 검증 대상.

## 6. 학습/검증 계획
1. content 생성원 집합 확정(§1) + ABox attr-타입 config(ordinal/categorical·comparative는 ordinal만).
2. 등방화 합성으로 **content-routing 학습**(NL→content op·여러 op 균형·substitute/create 포함).
3. **다도메인 전이**: 합성 학습 → retail·airline 동시(같은 라우팅·config swap). 닫히면 도메인-일반 입증.
4. **flow 축(P1-P9)과 합성**: content-rich 벤치는 content+flow 둘 다(op-IR + sequencing/gate). telecom/banking은 flow만.
5. **𝔥 표면 다양성(K-sweep)은 그 위**: content 생성원(𝔤) 완전 후 표면 다양화.

## 7. 정직 경계
- content 닫힘 = **가해(solvable) 연산 구간**(이론 §4)·무계 탐색·복잡 계산은 밖.
- tau2 5도메인은 *서비스-API 트랜잭션*에 편중 — 적대적 벤치(다른 content 패턴)로 닫힘 반증 시도 필요(`PRIMITIVE_MATRIX` §4 리뷰#6 content판).
- command-trivial=flow 분류는 경계 사례(set_mode=X는 substitute-single vs command 모호) — leave-one-out로 판정.
- SOPBench/TaskBench/CFB/BFCL content 매핑 미완(문서 기반·실데이터 분석 다음).

## 8. ★다중벤치 닫힘 검증 = 적대적 딥리서치 (2026-06-17·3 agent·arxiv 검증)
> "모든 도구계획 벤치가 예외없이 생성원으로 닫히나"를 *반증 시도*로 검증(self-fulfilling 회피). FC 11벤치·agentic 14벤치·taxonomy 이론.

### 8.1 판정 = "예외없이"는 FALSE·단 scope 안에선 닫힘 (정직)
**3층 결과:**
1. **FLOW(P1-P9) = 닫힘** (transactional tool agents): FC 11벤치(BFCL·API-Bank·ToolBench·Seal-Tools·NESTful·CFB·τ²·RestBench·API-BLEND·ToolAlpaca) 전부 FLOW 닫힘·agentic 6벤치(ToolEmu·TaskBench·WorkBench·SOPBench·MetaTool·AgentBoard-tool) 닫힘. **P1·P3·P4 = irreducible core(전 벤치)**. **P10 elicit/abstain**(BFCL Miss-Parameter/Function·"호출 안 함" 결정)=추가 후보(여전히 유한).
2. **CONTENT = Codd6 + aggregate확장 + functional-transform**: **문헌이 도출을 *지지***(Agent C) — Codd6(σπ×∪−ρ)는 filter/project만·**argmax/rank/aggregate는 증명상 밖**(Libkin-Wong JCSS 1997·aggregate=relational algebra 확장 필요)·create/substitute=functional-transform 층. ⇒ 우리 "관계대수+함수형" 도출 *정확·필요*. **단 value-derivation**(CFB relative-date "tomorrow"→concrete·unit/currency normalize)=filter/argmax/…/project 밖 = **`derive/normalize` 누락**(또는 offload·NESTful 선례).
3. **scope 밖 4축 = 이미 자인한 제외 축**(`PRIMITIVE_MATRIX §5`)·딥리서치가 *어느 벤치인지 확정*:
   - **G_loop**(stateful fold·AppWorld Python for-loop+aggregate)=코드실행. **G_csp**(TravelPlanner budget 결합제약·GPT-4 0.6%)=combinatorial solve. **G_ground**(WebArena·Mind2Web·OSWorld·VisualWebArena DOM/pixel)=GUI-grounding. **G_plan**(OSWorld·AgentBoard·Aquawar 부분관측 search)=장기계획.
   - **★NESTful math = 예외 아님**(argmax/divide를 *tool-call*로 emit·엔진 계산=offload) = **thesis 직접 지지**(벤치가 스스로 computation 외부화).

### 8.2 ★이론 정당화 + caveat (Agent C·문헌)
- **FLOW 닫힘 = Böhm–Jacopini**(CACM 1966·sequence/selection/iteration로 모든 제어 생성). **★단 "auxiliary state 허용·*minimal 아님*"**(Kozen-Tseng MPC 2008 propositional 반례·Kosaraju 위계). **⇒ "finiteness·closure" 주장 OK·"unique minimal basis" 주장 금지**(반증가능). **parallelism(P9)은 Böhm–Jacopini 밖**(sequential 정리)·structured-concurrency로 별도 정당화 필요.
- **CONTENT = Codd 정리(완전성)+Libkin-Wong(aggregate 필요성)** 가 2-part 도출 지지(반박 아님·강점).
- **우리 2축 finite closure-justified 집합 = 문헌 whitespace**(아무도 안 함). rival = Voyager(2305.16291·open-ended skill·단 *library층* 무한이지 *operator층* 아님)·BFCL(flow-only enumeration)·ReAct(2210.03629). 반례(tool-use 무한생성 증명) = **없음**.

### 8.3 ⇒ thesis 재정식 (경계 강화)
- **헤드라인 정정**: "모든 벤치 닫힘" 아님 → **"policy-governed transactional tool-orchestration(FLOW+CONTENT)서 유한 생성·닫힘"** + scope 밖 4축(loop/code·CSP·grounding·planning)·value-derivation은 **명시 out-of-scope or offload**. = 위배 아닌 *경계 정밀화*([[reference-repo-energy-lie-prior-work]] 정직 라벨 규율).
- **arxiv 검증**: 핵심 id 검증됨(2407.18901 AppWorld·2402.01622 TravelPlanner·2406.12045 τ²·2501.10132 CFB·2409.03797 NESTful·2305.16291 Voyager·2210.03629 ReAct·Codd1972·Libkin-Wong JCSS1997·Böhm-Jacopini CACM1966·Kozen-Tseng MPC2008). snippet-only(2602.*·2603.*·2508.*) = 인용 전 재검증([[feedback-arxiv-citation-discipline]]).

## 9. ★value-derivation 결정 = NESTful 식 offload = P2b 환원 (2026-06-17·사용자)
딥리서치가 드러낸 유일 CONTENT 누락(value-derivation: relative-date·unit/currency normalize)을 **content 생성원에 추가하지 않고 flow P2b(gather-for-arg)로 환원**:
- NESTful이 `divide($a,$b)`를 *tool-call*로 emit·엔진이 계산하듯, `"tomorrow"→date`=`date_resolver(relative)` 호출→출력 인자(**P2b**)·`100USD→EUR`=`currency_convert(...)`→출력(P2b).
- = `PRIMITIVE_MATRIX §3` "calculate→tool-call→P2b 환원(seam β 닫힘)" 확증(이미 도출됨·딥리서치 재확인).
- **⇒ CONTENT 생성원 8개 *불변***(filter·argmax·argmin·rank·comparative·substitute·create·project)·value-derivation=flow P2b·**양 축 닫힘 강화**.
- **처방**: 도메인-일반 derivation 도구(date-resolver·unit-convert·calculator)를 **scaffold/엔진 제공**→모델은 *계산 안 함*·P2b 호출만(offload thesis 정합·LLM=명명/결정론=실행).
- **CFB value_error 진단**: CFB는 derivation 도구 *미제공*→모델 직접계산 강요→실패. 처방=도구 scaffold 추가시 P2b로 풀림.
- anti-targeting 안전: calculator/date-resolver=도메인-일반(retail 특화 아님).
