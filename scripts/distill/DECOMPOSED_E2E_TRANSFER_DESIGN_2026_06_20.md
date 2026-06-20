# 분해-협업 e2e 전이 설계 v2 (2026-06-20) — 미빌드 step-scaffold로 벤치-TBox를 tau2에 전이

> **v2 = 사용자 리뷰 4위험 반영**(2026-06-20). v1 치명결함: grounding을 CFB 스페셜리스트에 둠 → (a) 직전 A2_GROUNDING 엔진작업 무효화 (b) v7 R4 전이실패 반복. v2 교정 = **grounding=엔진·스페셜리스트=flow/threading만**.
> 진입 = 06-NOW(드리프트 교정) + 이 설계. **tau2 학습 0**(도메인-타겟 금지·[[11]]). 상위 = `A2_GROUNDING_WIRING_DESIGN`·`ABSTENTION_AS_DECIDABLE`·`INTEGRATED_TBOX_DESIGN §5/§7`·`CROSS_BENCH_TRANSFER_PLAN`·`FIXED_VS_VARIABLE`. 불변 = [[03-anti-drift]][[05-fixed-vs-variable]][[00-thesis]].

## 0. 왜 재개인가 (리뷰 검증·재론 아님)
- 프론티어 실측(학습0): base 7B 0.24 / 32B 0.60 / frontier ~0.81. **벤치-TBox 단일-LoRA 전이 전부 base 미만**(solo_cfb_mid 0.16·sts 0.00·fact_* 0.02) = 음성.
- §5 분해 아키텍처는 2026-06-18 폐기됐으나 *본체(step-scaffold) 미빌드*하고 단독LoRA로 피벗(그것도 실패). 폐기사유=facet3 단독 어댑터가 full맥락서 0회 호출=**맨 어댑터 문제**(격리 scaffold 미빌드).
- 격리-작동 증거: forced-selection 0.14→0.62. ⇒ 두 우회(맨어댑터·단독LoRA) 실패·분해 본체 미측정 → 빌드=anti-drift 정합.
- ⚠️ **6번째 아키텍처 피벗 — 이번엔 S-min 빌드·측정하고 데이터 전 재피벗 금지**(§6 S-min 생사판정이 그 규율).

## 1. ★분담 (리뷰 위험1 교정·정합의 핵심)
**세 버킷([[00-thesis]] 1.5)을 tau2 e2e에 정확히 매핑:**
- **ENGINE (결정론·offload·학습 0·이미 빌드됨)** = **grounding 전체**: fetch-first 라우팅 + provenance 검증(R1b·날조거부) + A2-decidable producer 선택(어느 getter=스키마서 결정론 도출) + concrete-resolution(resolve_selection 엔진) + gate(GateInterpreter·gate_spec). = `A2_GROUNDING_WIRING`+`ABSTENTION_AS_DECIDABLE`+`GATE_INTERPRETER`.
- **SPECIALIST (학습·도메인-일반·격리 호출·전이 대상)** = **flow-discipline(SOP: gather-first·gate-존중·write-confirm 순서) + data-flow threading(TaskBench: 출력→입력 바인딩)**. *이것만* 학습/스페셜리스트.
- **ABox (swap·도메인사실)** = 카탈로그·gate_spec·producer-맵·vocab.

**★grounding은 스페셜리스트 아님 (리뷰 위험1·치명)**: 
- 직전 세션이 grounding을 *엔진 일*로 확정(A2_GROUNDING). grounding을 재학습 스페셜리스트로 두면 그 작업 무효화.
- v7 실측 박제(`PRIMITIVE_COVERAGE_MATRIX:92`): "grounded 2-hop CFB가 τ² 전이 *실패*·기제 = CFB linear-observe ≠ τ² proactive producer-selection(R4 의미층 미학습)." ⇒ **CFB-스페셜리스트 grounding = v7 알려진 실패 반복**. 게다가 R4 실패는 **맥락-격리로 안 고쳐짐**(0×호출 문제 아니라 의미-전이 문제). ⇒ grounding은 엔진(A2-decidable)이 함.

## 2. ★타깃 = 실병목 (flow/grounding/recovery·selection 아님)
- 진짜 병목(06-NOW census) = action-flow(DB-state)·order_id grounding 날조·P7 recovery. selection=소수경로(5%).
- **★root-cause census 박제(`PRIMITIVE_COVERAGE_MATRIX:92`)**: order_id 격차 = **"fetchable 값 날조-FIRST" 기본행동(17/20·스키마 example `#W0000000` 복사)·능력부족 아님**. 모델은 2-hop gather·P7 복구 *할 줄 안다*(task6 dump). 처방 = **R1b provenance(arg∈{user,tool}·example 거부) + fetch-first = 엔진**. ⇒ grounding은 *capability gap이 아니라 default-behavior gap* → **엔진이 결정론으로 닫는다**(학습 아님).
- 이제 분담상: **grounding 날조 = 엔진이 결정론 차단/해소**(A2_GROUNDING). **flow 신뢰성(언제 gather/auth/confirm·복구) = SOP 스페셜리스트 격리 호출**. 두 병목을 각 버킷이.
- **★S-min 사전등록 예측**: order_id는 default-behavior 문제라 **"엔진만" arm(R1b provenance+fetch-first·학습0)이 이미 대부분 닫을 공산** → grounding decidable-비율 높음(offload 지배). 그렇다면 flow-스페셜리스트의 *증분*은 multi-step 신뢰성/복구에서만 — **S-min 3-way가 이 분담선을 직접 실측**(엔진 몫 vs 학습 몫). (이게 thesis 핵심 측정: grounding이 offload면 학습가치는 flow/전이경제로 이동.)
- 목표 = 분해 > base 7B(0.24)·상대 Pareto. 헤드라인 = ABox-swap airline 전이 + 변경-흡수 경제(§7).

## 3. 아키텍처 = 결정론 step-router + 엔진 + flow/threading 스페셜리스트
주입점(코드확인): `LLMAgent.generate_next_message`(emit) + `BaseOrchestrator._execute_tool_calls`(t2_resolve_patch 기존hook).
- **step-router(신규·결정론)**: 턴마다 *구조 신호*로 다음 sub-결정 종류 판정 → 디스패치.
- **엔진 디스패치**(grounding/gate/resolve): §1 ENGINE. 이미 빌드.
- **스페셜리스트 디스패치**(flow/threading): SOP/TaskBench LoRA를 **native 포맷 격리맥락**으로 호출.

### 3a. ★router 분류신호 = 순수 구조/ABox (리뷰 위험3 확정·make-or-break)
router는 **절차를 발명하면 안 됨**(procedure-offload=thesis 무효·[[03]] L0). 허용 신호 = **구조/데이터만**:
- gate_spec 미충족 전제(어느 precondition 안 됨) — ABox 데이터.
- 미충족 required 인자가 *producer 출력서 와야 함* — 도구 스키마 + 직전 출력(구조).
- data-dependency(출력→입력 미바인딩) — 구조.
**금지**: "먼저 gather 다음 act" 같은 flow *순서*를 router가 인코딩. flow 순서 = **SOP 스페셜리스트가 학습**(도메인-일반 discipline) 또는 **gate_spec precondition**(ABox)이 집행. router는 *지금 어느 버킷*만 구조로 판정.
- **테스트(필요+충분)**: grep `if domain`=0 (필요) **+ router 코드에 절차 dirgraph 0**(충분·리뷰 위험3: 분기0은 procedure-offload 아님의 충분조건 아님). + airline-unchanged 작동.

### 3b. ★격리맥락 = 스페셜리스트 native 분포 재현 (리뷰 위험4)
"프롬프트 축소"(τ²-reduced) ≠ 격리. 각 스페셜리스트는 자기 벤치 *native 포맷*으로 학습 → 격리맥락은 그 **native 입력 분포를 재현**해야(다른 입력=분포이동=실패). 0.62 증거는 synth-native 포맷이었을 공산. ⇒ flow-스페셜리스트 호출 시 SOP-native 포맷(도구 부분집합·지시·상태)으로 변환·검증(포맷 일치 측정).

## 4. ★사전등록 (리뷰 위험2·결론 전 박제)
- **격리-작동 증거(0.62)는 selection뿐**. flow/threading에서 격리가 작동한다는 증거 **없음**. selection→flow 일반화 **금지**.
- **S-min은 두 가지를 *분리* 측정**: (a) **격리**=스페셜리스트가 자기 facet 호출되나(0×호출 해소) / (b) **전이**=그 facet이 τ²로 일반화해 *돕나*. v7 R4처럼 (a) 풀려도 (b) 실패 가능. 두 숫자 합치지 말 것.
- **Risk B**(`INTEGRATED_TBOX §5b`): 절대천장 = real-NL facet 인식. 헤드라인=상대 Pareto지 절대수.

## 5. 측정 (`INTEGRATED_TBOX §7.3` 부활·tau2 학습0)
- **3-way**: base 7B / 엔진만(grounding/gate·스페셜리스트 0) / 엔진+격리 flow·threading 스페셜리스트. 헤드라인 = > 둘 다.
- autopsy: 스페셜리스트 호출되나(격리)·order_id 날조 소멸(엔진)·flow 신뢰/복구 개선(스페셜리스트).
- **decidable-비율**: flow/threading 결정 중 엔진(gate_spec)이 닫는 비율 vs 스페셜리스트 필요. (너무 높으면=학습 불필요·정직 보고.)
- 전이매트릭스: 같은 시스템 ABox만 swap → retail·airline. 보상=결정론(DB-match). 32B궤적348=facet 오라클(대조·학습아님).

## 6. 단계 (최소→확장·재피벗 금지)
- **S-min** = order_id task에서 **엔진-grounding(A2_GROUNDING) + 격리 SOP flow-스페셜리스트(native 포맷)**. 측정: (a)격리 호출됨? (b)base 0.24 초과? **생사판정·데이터 전 재피벗 금지.**
- **S1**: +threading(TaskBench)·decidable-비율.
- **S2**: ABox-swap airline 전이(헤드라인).
- 각 GO/NO-GO. NO-GO(분해도 base 못 넘음 or 격리해도 flow 전이실패=v7 R4류) = thesis 음성→비용결론 정직 후퇴(§7).

## 7. 빌드 가치 정렬 (리뷰 전 세션 위험①·정직)
- 학습-TBox 가치 = 프롬프트-following 깨지는 작은크기서만. 32B는 프롬프트로 0.64(학습0). ⇒ **lift 자체는 저가치**(사용자).
- **유일 방어선 = 변경-흡수/전이 경제학**: 32B+scaffold는 도메인마다 프롬프트-엔지니어링 / 학습-TBox는 ABox-swap 공짜 전이. S2(airline 전이)가 이걸 실측. **S-min 통과 후에도 이 프레임으로 평가**(절대 lift 아님).

## 8. 리뷰 잔여 확인 (구현 전)
- §3a router 신호 구현 = 순수 구조 함수(절차 0) — 코드 리뷰로 검증.
- §3b 격리 = SOP-native 포맷 변환기 — 0.62 재현 조건 대조.
- S-min facet = order_id(엔진-grounding) + flow(SOP) 분담 맞나.
