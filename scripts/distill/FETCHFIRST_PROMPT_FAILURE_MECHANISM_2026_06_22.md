# FETCH-FIRST 프롬프트 실패 메커니즘 — 7B는 왜 안 되나 (엄밀 원인 확정·2026-06-22)

> 사용자 지시 2026-06-22: 7B/14B/32B fetch-first 프롬프트 효과 **궤적 전수 조사**로 7B 실패 *이유 확정*·결과만 보지 말고 **프롬프트 분할로 원인 격리·다른 원인 배제**·필요시 **SFT 아닌 DPO/RLVR(verifier penalty)**.
> 상위 = `RULE_LEVER_COST_EFFICIENCY_PROGRAM`(prompt 레버의 7B 실패 메커니즘 = promptable@size vs learn 결정 근거). 관련 = `C4_LEARN_FETCHFIRST_CROSSOVER`·`PRIMITIVE_COVERAGE_MATRIX:92`(날조-first prior 가설)·§35(프롬프트=32B만+).

---

## §0. 질문 (확정 대상)
**7B가 fetch-first *프롬프트*를 안 따르는 원인은 무엇이고, 대안 원인은 배제되나?** 후보:
- **M1 non-attend (instruction-following 크기게이트)**: 프롬프트를 *주의/적용 못 함* — 규칙 없는 듯 행동. 큰 모델은 같은 프롬프트 따름(§35 32B+).
- **M4 prior-override (날조-first 사전행동)**: 규칙을 *알면서도* 날조 prior가 이김(`PRIMITIVE_MATRIX:92` 가설). 규칙 복창해도 날조 → SFT 양성예시론 못 죽임 → **DPO/RLVR penalty 필요**.
- **M2 capability (2-hop 불가)**: gather *시도*하나 threading 실패. (§35는 7B가 gather/복구 *할 줄 안다* → M2 약함·단 검증 대상.)
- **M3 partial / M5 success.**

핵심: 결과(A_notfound 17~28)만으론 M1/M4/M2 구분 불가 → **궤적 전수 + 프롬프트 분할**로 격리.

## §1. 스케일 축 (M1 시험 = 크기-의존 instruction-following인가)
동일 `C4_FETCHFIRST_DG` 프롬프트 × **7B / 14B / 32B**. 기존: 7B prompt sweep 有(`c3_*`)·14B/32B는 floor+rules(일반)만 → **14B/32B fetch-first 프롬프트 신규 실행**.
- M1 예측: A_notfound이 크기↑로 단조↓(프롬프트가 큰 모델만 먹힘). 7B만 안 됨 = 크기-게이트.
- M4 예측: 큰 모델도 *날조 prior* 잔존(A_notfound 크게 안 줄거나·gather-없이-날조율 크기 무관).

## §2. 프롬프트 분할 (어느 sub-rule서 깨지나·원인 격리)
fetch-first = 4 sub-rule. 단독·조합 arm으로 격리(7B 중심·핵심은 14/32B도):
| arm | 내용 | 격리 목적 |
|---|---|---|
| **P0** | 무프롬프트(base) | 기준 |
| **P-neg** | "절대 날조/추측/예시복사 금지"만(negative constraint) | 날조 억제*만*으로 닫히나(=prior 약화) |
| **P-fetch** | "값 없으면 생산도구 *먼저* 호출"만(proactive) | proactive gather*만*으로 닫히나 |
| **P-full** | neg+fetch+copy+ask (=C4_FETCHFIRST_DG) | 완전 프롬프트 |
| **P-restate** | P-full + *매 도구호출 전 규칙을 한 줄 복창하고 각 인자 출처 명시* | ★M1 vs M4 결정자: 복창=주의강제 |
| **P-cot** | P-full + "각 인자 출처를 단계적으로 추론 후 호출" | reasoning-depth/B-budget 시험 |

★**M1 vs M4 결정**: P-restate가 7B A_notfound을 닫으면 → **M1(주의/instruction-following)**(강제주의로 해결). 복창은 정확한데 *여전히 날조*하면 → **M4(prior-override)**(앎≠행동·penalty 필요). P-cot은 추론깊이 기여 분리.

## §3. 궤적 전수 메커니즘 census (도구 신규 `c4_prompt_mechanism.py`)
각 *실패* 태스크 궤적을 읽어 분류(sim-수준 A_notfound 너머):
- **gather-attempt**: 날조 *전에* 생산도구(producer/getter·auth 포함)를 호출했나? (yes/no)
- **rule-echo**(P-restate arm): 모델이 규칙을 복창했나? 복창 후 행동이 일치했나?
- 분류:
  - **M1**: gather-attempt=no·날조 즉시(규칙 무시·복창 안 함) → 주의/적용 실패.
  - **M4**: gather-attempt=no지만 (P-restate서) 복창은 정확 → 앎에도 날조 = prior-override. + **schema-example 복사**(#W0000000류) 비율 = prior 강도 직접 proxy.
  - **M2**: gather-attempt=yes나 wrong-id(threading 실패) → capability.
  - **M3**: 일부 gather·일부 날조. **M5**: 성공.
- **출력**: 각 arm×크기별 {M1,M2,M3,M4,M5} 분포 + 예시 궤적 5개/카테고리(전수 근거).

## §4. 대안 원인 배제 (엄밀성)
- **포맷/하버스 오류 아님**: raw tool_call transport 정상 확인(`SFT_COLLAPSE_AUTOPSY` 식 raw 점검).
- **user-sim 교란 아님**: 동일 user_llm(gpt-4.1·temp0)·동일 태스크셋 paired.
- **프롬프트 위치 아님**: system vs 첫-user 위치 변주 1셀(docstring 약했던 전례·§118).
- **토큰 truncation 아님**: max-model-len 충분·프롬프트 길이 census.
- **태스크 난이도 confound 아님**: 크기 arm 동일 태스크·격리 A_notfound(global pass 아님).

## §5. 결정 → learn 방법 (SFT vs DPO/RLVR)
- **M1(크기-게이트 주의)** → prompt 레버는 7B서 무효·learn 필요. SFT(cfbsynth SHAPE)로 *충분할 수 있음*(주의 못하는 걸 weights에 박음).
- **M4(prior-override)** → ★**SFT 양성예시 불충분**(prior 잔존)·**DPO/RLVR + penalty reward 필요**: verifier=provenance gate(날조 id=reward<0·gather-후-copy=reward≥1). DPO pair=(fab traj, grounded traj)·RLVR=gate-검증 reward. **현 cfbsynth SFT는 비교 baseline**(M4면 SFT<DPO 예측).
- **M2(capability)** → 2-hop threading 학습(cfbsynth가 정확히 그것·SHAPE).
- ⇒ 이 진단이 cfbsynth-SFT가 맞는 방법인지/DPO-penalty로 가야 하는지 *확정*.

## §6. 실행 순서
1. ✅ 메커니즘 census 도구(`c4_prompt_mechanism.py`) → **기존 7B `c3_fetchfirst`·`c3_nofab` 즉시 census**(무비용·첫 신호).
2. 14B/32B fetch-first 프롬프트 실행(GPU1·SFT는 GPU0) → 스케일 census.
3. 프롬프트 분할 arm(P-neg/P-fetch/P-restate/P-cot) × 7B(+14/32B 핵심) → M1 vs M4 결정.
4. §4 배제 체크.
5. 결정 → DPO/RLVR 설계서(M4 시) 또는 SFT 확정(M1/M2).

## ★§RESULTS (2026-06-22·확정) — M4 schema-example-copy prior

**도구 `c4_prompt_mechanism.py`(grounding 버그=토큰-마침표글루 수정 후·궤적 전수). 핵심신호=schema_copy(스키마 예시값 #W0000001 emit).**

**(1) 7B 메커니즘 분포 (실패 nfail 정규화·retail)**
| arm | nfail | no_gather | schema_copy | gather_wrong(M2) | grounded_other(B) |
|---|---|---|---|---|---|
| base | 91 | 44 | **43(.47)** | 5 | 42 |
| nofab | 87 | 39 | 38 | 8 | 40 |
| fetchfirst | 91 | 28 | **28(.31)** | 9 | 54 |
| fewshot | 88 | **0** | **0** | 20 | 68 |
| structured | 87 | 28 | 27 | 8 | 51 |
- ★**no_gather ≈ schema_copy 전 arm 일치** ⇒ 7B의 날조는 *거의 전부 스키마 예시값 복사*(랜덤 invention 아님).

**(2) 스케일 축 (schema_copy 율)**: 7B 0.47 → **14B 0.045 → 32B 0.006**. = 강한 크기-의존·7B-특이 prior.

**(3) 궤적 확정**(task2·fetchfirst arm): 모델이 get_user_details로 실제 order_id(#W2378156…) 관측 후에도 첫 get_order_details=**#W0000001**(스키마 예시). = gathered-then-ignored.

### ★확정 = M4 (schema-example-copy prior)·대안 배제
- **non-attend(M1) 배제**: 프롬프트가 "#W0000000 ALWAYS fail"로 *그 토큰을 명시*함에도 28/91 여전히 emit·동일 프롬프트가 큰 모델선 작동 → 주의실패 아님.
- **capability(M2) 배제**: fewshot서 no_gather→0(gather 능력 有)·gather_wrong 낮음.
- **format/harness·user-sim·task-confound 배제**: raw 궤적 실값 확인·paired 동일 태스크/user_llm.
- ⇒ **7B fetch-first 프롬프트 실패 원인 = 스키마 예시값 복사 prior**(크기로 획득되나 7B엔 부재)·**NL 지시로 override 불가**(예시값 명시해도)·**시연(fewshot)/scale만 억제**.

### ★learn 방법 함의 (SFT vs DPO/RLVR)
- prior가 **시연-민감**(fewshot→0) ⇒ **SFT(cfbsynth=gradient 시연)이 억제할 공산**(내재화된 fewshot). 현 cfbsynth SFT가 정확히 이 시험.
- prior가 **특정 나쁜 행동**(예시값 emit) ⇒ **DPO/RLVR penalty가 더 표적적**(verifier=provenance gate·#W-placeholder=reward<0·grounded-copy=reward≥1). SFT 잔여 schema_copy 시 escalate.
- ★**예측 검증법**: 학습 후 이 census 재실행 → schema_copy 율이 fewshot처럼 ~0이면 성공. (SFT<DPO 예측은 잔여로 판정.)
- ⇒ cfbsynth-SFT 정당화됨(시연이 시연-민감 prior 죽임)·DPO-penalty=fallback. 진행 중 SFT가 1차 답·잔여시 DPO 설계.

### 남은 (선택·belt-and-suspenders)
- P-restate arm(복창 후 행동): 현 증거(예시값 명시-저항+scale+시연대조)로 M4 충분 — P-restate는 추가확인용·우선순위 낮음.
- 14B/32B에 *동일 C4_FETCHFIRST 프롬프트* 직접 실행(현재는 floor/rules만): prompt가 14B서 prior 더 누르나(현 floor만으로도 prior 거의 없음=무관할 수 있음).

## §7. GO/NO-GO (원인 확정)
- **확정 = M-분포가 한 메커니즘 지배 + 대안 배제 + P-restate 결정자 통과.** 예: "7B 실패의 X%가 M4(복창정확+날조)·P-restate로 안 닫힘·schema-copy Y% = prior-override 확정 → DPO-penalty."
- 모호(분포 혼재)면 추가 분할·표본↑.

---
**불변**: [[05]]([[03]] 설계먼저·궤적 전수)·[[13]](흡수우선)·[[12]]. 상위=`RULE_LEVER_COST_EFFICIENCY_PROGRAM`·`C4_LEARN_FETCHFIRST_CROSSOVER`.
