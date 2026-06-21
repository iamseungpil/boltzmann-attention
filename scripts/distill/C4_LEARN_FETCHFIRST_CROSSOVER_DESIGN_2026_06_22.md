# FETCH-FIRST 규칙 × 전 레버 비용-효율 곡선 (구 C4-LEARN CROSSOVER·2026-06-22)

> ★**확장(사용자 지시 2026-06-22)**: fetch-first를 prompt/skill/hook/learn *전 레버*로 측정 = `RULE_LEVER_COST_EFFICIENCY_PROGRAM`의 첫 곡선(worked example). autofetch 원칙 결정 = 그 곡선의 한 비교(learn vs hook-perform).
> **설계서(빌드 전·GPU 0).** 상위 = `RULE_LEVER_COST_EFFICIENCY_PROGRAM_2026_06_22` + `LLM_CONTROL_EXPERIMENT_REDESIGN_2026_06_22 §4-①`.
> 진입 = 메모리 `06-NOW`·[[05-fixed-vs-variable]]·[[07-control-not-prompt]]·[[11-transfer-direction]].
> 관련 = `R1B_PROVENANCE_DESIGN`·`V7_PROACTIVE_GATHER_DESIGN`·`CROSS_BENCH_TRANSFER_PLAN`·`CFBSYNTH_P2B_P4_DESIGN`·`REST_INTERNALIZE_DESIGN`(데이터-기근 교훈).

---

## §0. 착수 전 [[05]] 결정질문 (이 실험 + C4 학습)

1. **도메인-특화 순증?** — ❌ **C4 학습 = canonical learn 벤치 = SOPBench + TaskBench + Synth서만**(★CFB 폐기·`INTEGRATED_TBOX v2`). fetch-first는 SOP(gather R2)+TaskBench(threading P2b). tau2(retail/airline) = **held-out eval *only***. ⛔ **tau2 학습 절대 금지**(= 2026-06-20 ReST-on-tau2 드리프트·killed·[[11]] 정면위반). 전이 = ABox-swap·재학습0.
2. **유동성 동결?** — ⚠️ **이 실험이 그 측정 자체.** C4-learn(weights·유동성 보존)이 C1-perform(autofetch·결정론 동결)을 *대체*하는지를 flexibility-loss 축(§4)으로 비교. null = "autofetch가 공짜" 기각.
3. **scaffold가 도메인 행동 수행?** — C4-learn arm = scaffold 무증설(모델이 fetch-first 수행). C1-perform(autofetch) arm = 비교군(=현 [[05]] 위반축·이 실험이 강등 정당성 판정).

---

## §1. 결정할 질문 (autofetch 원칙)

§35 분해의 잠정 배정 = "**A(grounding)=엔진(autofetch)·B(operand)=학습**". 그러나 2026-06-22 [[05]] 정정: autofetch(C1-perform)는 **scaffold가 fetch 수행 + A2 성장 + 유동성 동결** = 결정질문 3개 다 yes ⇒ **기본 아님.** [[05]] 재프레임: **fetch-first = 모델 스킬(C4-learn/scale)**·autofetch는 측정으로만 정당화되는 비용옵션.

**∴ 결정질문**: A(entity-id grounding) failure를, autofetch(C1-perform) **없이**, 더 약한 통제로 닫을 수 있나?
- **C4-learn**(fetch-first를 도메인-일반 학습 → ABox-swap 전이)이 A를 autofetch만큼 닫으면 → **autofetch 강등**(gate-only C1-deny + C4 기본).
- 오직 autofetch만 닫으면 → 그 비용(A2-growth·flexibility-loss)을 **measured-justified**로 명시(정직 경계).

C3 sweep(§35b·`LLM_CONTROL §5`) 부분실측: C0(prompt) 전부 A 못 닫음(17~28) / C1-perform(autofetch) A=12 닫음. **빈칸 = C1-deny·C4-learn.** 이 실험이 채운다.

---

## §2. Arms = fetch-first 규칙의 *전 레버 곡선* (사용자 지시 2026-06-22·`RULE_LEVER_COST_EFFICIENCY_PROGRAM` 첫 곡선)

★fetch-first(C3/P8)를 **prompt/skill/hook/learn 4레버 전부**로 측정 = 이 규칙의 비용-효율 곡선 1개 완성. retail+airline τ²·base=Qwen2.5-7B·평가=pass^1 + 실패 census(`t2_failcensus`). **동일 ABox·동일 eval 코드.**

| arm | 레버(통제점·강도) | 구현 | 곡선 x위치(생애비용) | [[05]] |
|---|---|---|---|---|
| **A0 base** | (없음·기준선) | base 7B·gate0·autofetch off | 0 | 기준 |
| **A0' scale** | scale 기준선 | 14/32B base as-is | (수평선) | 대체대상 |
| **Ap prompt** | C0 prompt | 도메인-일반 fetch-first 지시(도구명 무·=C3 sweep fetchfirst/nofab) | 최저 | soft·동결0 |
| **As skill** | C0+ skill | 온-디맨드 절차모듈(확정): 미해결 entity-ref 감지(router)→"생산도구 먼저 호출해 실값 복사" 절차+exemplar invoke·발동시만 컨텍스트 | 저-중 | soft·동결0 |
| **Ah-deny** | hook C1-deny | gate1(provenance 사실게이트)·autofetch off·날조 id deny | 중 | 최소 enforced·동결0 |
| **Ah-perform** | hook C1-perform | gate1 + autofetch on(현 engine arm·§5 A=12) | 중-고 | ★위반축(perform·A2성장) |
| **Al learn** | C4 learn | base + fetch-first LoRA(§3·도메인-일반)→ABox-swap·autofetch off | 고(build)·OpEx0 | ★원칙 후보·동결0 |

**곡선 판독(예측·반증가능)**:
- Ap(prompt) = C3 sweep 실측 A 17~28(**못 닫음**) → 곡선 왼쪽 낮음(soft 불충분).
- Ah-deny = §35b "7B stall" 예측(A≈base) → enforced여도 *복구 능력* 없으면 deny만으론 부족.
- **Ah-perform = A=12 닫음**(실측) → 단 perform=A2성장·동결(비용 높은 x).
- **As(skill)·Al(learn) = 빈칸 = 이 실험 핵심**: skill(온-디맨드 절차)이 prompt보다 닫나? learn(내재화)이 perform 없이 닫나(동결·A2성장 0)?
- **knee = reliability 포화 최소비용 레버 = fetch-first 배정.** learn이 perform만큼 닫으면 → autofetch 강등(§5).

---

## §3. C4-learn 데이터 (★도메인-일반·tau2 아님)

**fetch-first discipline = "값이 없으면 *생산 도구를 먼저 호출*해 실값을 복사·날조 금지"**(R1b/P8 provenance + V7 proactive gather). 이건 도메인-불변 규율 → **canonical learn 벤치 = SOPBench + TaskBench + Synth**서 학습(★CFB 폐기됨·`INTEGRATED_TBOX v2`·2026-06-18 사용자결정·사용자 확인 2026-06-22).

- **소스**(CFB 아님): **SOPBench(gather-first R2·P2a)** + **TaskBench(data-flow threading = 2-hop arg-binding·P2b/P3·id-grounding·R1)** 궤적. = CFB가 담던 grounded 2-hop을 TaskBench threading + SOP gather가 커버. 기존 자산 = SOP/TaskBench native-FC LoRA·`R1B_PROVENANCE_DESIGN`·`CROSS_BENCH_TRANSFER_PLAN`.
- **★설계 함의**: CFB 폐기 논리(grounding=결정론 게이트=hook·not learn·`INTEGRATED_TBOX:73`) ⇒ fetch-first의 **learn arm이 설계상 약할 수 있음**(P8 provenance는 hook로·P2b threading만 learn). 이게 곡선의 *발견*(knee가 hook쪽?). §35 "A=엔진·B=learn"과 정합.
- **⛔ 금지**: tau2 retail/airline 궤적 학습(`sft_rest_s0_retail.jsonl` 등 = 드리프트·미사용). tau2는 **전이 측정 타깃**.
- **무붕괴**: replay 1:1(일반 tool-use·tbnfc/tb_all) 혼합·small-rank LoRA(r16류). held-out 일반능력 불변 확인.
- **데이터-기근 위험(REST 교훈·`REST §4.2`)**: base-7B 자기생성 = HOLE. 단 *여기 타깃은 도메인-일반 fetch-first 규율*(narrow·SOP gather + TaskBench threading서 풍부)이지 tau2 task 커버가 아니므로 데이터-기근 양상 다름. 그래도 **coverage 진단 내장**(fetch-first 패턴이 학습셋에 충분히 대표되나).
- **다양성**([[12]]): 표현/구조 다양 필수(단일템플릿 SFT=표면매핑 역전이). SOP+TaskBench 두 소스·표현 다양.

---

## §4. 메트릭 (목적함수 분모 포함)

| 메트릭 | 정의 | 왜 |
|---|---|---|
| **A_notfound** | entity-id grounding 실패율(`t2_failcensus`) | ★핵심신호(닫혔나) |
| **pass^1** | e2e DB-match | 전체효과 |
| **flexibility-loss** | false-block rate(옳은 툴콜 차단)·over-deny(validate) | ★C1 enforced의 숨은비용·A2-동결 측정 |
| **A2-growth** | arm이 요구하는 A2 필드 수(autofetch producer-map·placeholders) | ★[[05]] minimize-A2 |
| **no-collapse** | held-out 일반 tool-use(tbnfc 등) 불변 | C4 무붕괴(§3 replay) |
| **transfer** | **airline held-out**(동일 LoRA·ABox-swap만) A_notfound·pass | ★[[11]] 도메인-일반 입증(retail 학습 아님→자동 충족·SOP+TaskBench 학습이라) |

**핵심: A2 arm(C4-learn)은 A2-growth=0·flexibility-loss=0(weights·gate-deny만)·전이=SOP+TaskBench 학습이라 retail·airline 둘 다 held-out.** A3(autofetch)는 A2-growth>0(producer-map)·동결.

---

## §5. GO/NO-GO (autofetch 원칙 결정)

- **GO(autofetch 강등)**: A2(C4-learn) A_notfound ≈ A3(autofetch) **그리고** A2-growth=0·flexibility-loss≈0·무붕괴·airline 전이. ⇒ **기본 = gate-only(C1-deny) + C4-learn fetch-first.** autofetch 폐기 또는 측정-비용옵션으로만.
- **NO-GO(autofetch 정당)**: 오직 A3만 A 닫고 A2(C4-learn)는 stall. ⇒ autofetch를 **measured-justified** 비용옵션으로 명시(정직 경계)·단 A2-growth/동결 비용 계상.
- **중간**: A2가 부분 닫음(A1<A2<A3) → C4+잔여 deny 조합·crossover 점 보고.

= `LLM_CONTROL §4-①` crossover의 A-leg 채움. autofetch 원칙(§2.3)·C10(B-leg) 동시 정보.

---

## §6. 위험·함정

- **tau2 학습 유혹**(반복 드리프트 1순위): C4 데이터는 SOP+TaskBench(+Synth)만·CFB 폐기. 코드리뷰 게이트 = 학습 jsonl에 tau2 도구명 grep=0.
- **deny-only stall**(§35b): A1이 stall이면 정상(가설). 단 A2(C4-learn)도 stall이면 → fetch-first가 7B서 학습불가(=genuinely scale-bound or operand-entangled) → 경계지도 기여.
- **single-facet→full-agent mismatch**([[05]]·2026-06-20 죄): fetch-first LoRA를 *full-agent*로 평가 = OK(이건 규율 내재화지 narrow 선택기 아님). 단 데이터=full-agent concrete-arg 궤적이어야(추상 단일도구 금지·`SFT_COLLAPSE_AUTOPSY` 교훈).
- **A2-growth 잠입**: A2(C4-learn) arm이 placeholders/producer-map 안 쓰는지 확인(autofetch off). 키스톤 A2 과성장 정리(§2.5)와 동반.
- **진행률 가시**([[30]]): 학습 잡 step/loss/ETA flush·save_steps·`| tail` 금지.

---

## §7. 둘째 leg = C10 operand(B) — 동일 방법

- B(operand/write-args) = §35 scale-불변 학습잔여(plateau)·C4-learn의 *원래* 타깃(`HANDOFF §2.4 C10_OPERAND_LORA_DESIGN`).
- 동일 crossover 프레임: base / C0-prompt / C4-learn(operand-formalize·도메인-일반 Synth/TaskBench) / (B엔 C1-perform 부적용=operand는 결정론 offload 불가·NL→formalize).
- **본 doc = A-leg(fetch-first) 우선**(autofetch 원칙 결정이 C3/C8/C10 전부 reframe). B-leg = 별도 `C10_OPERAND_LORA_DESIGN`(이 프레임 상속).

---

## §8. 빌드 순서 (GPU = 리뷰 후)

1. ✅ 이 설계서 (GPU 0).
2. **사용자 리뷰** ← 여기서 멈춤.
3. C4 데이터 빌드(SOP gather + TaskBench threading fetch-first·coverage 진단·tau2-grep0 게이트).
4. C4 학습(small-rank LoRA+replay·진행률 가시·무붕괴 check).
5. 4-arm eval(retail+airline·A_notfound·pass·flexibility-loss·A2-growth) → GO/NO-GO §5.

---

**불변 정합**: [[05]](결정질문3·minimize-A2·tau2학습금지)·[[07]](soft 불충분→C4)·[[11]](전이방향=primitive학습·tau2 held-out)·[[12]](다양성)·[[13]](흡수우선 scale→learn→scaffold). 상위 = `LLM_CONTROL_EXPERIMENT_REDESIGN_2026_06_22`.
