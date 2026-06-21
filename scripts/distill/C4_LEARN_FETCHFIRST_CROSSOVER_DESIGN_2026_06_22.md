# FETCH-FIRST 규칙 × 전 레버 비용-효율 곡선 (구 C4-LEARN CROSSOVER·2026-06-22)

> ★**확장(사용자 지시 2026-06-22)**: fetch-first를 prompt/skill/hook/learn *전 레버*로 측정 = `RULE_LEVER_COST_EFFICIENCY_PROGRAM`의 첫 곡선(worked example). autofetch 원칙 결정 = 그 곡선의 한 비교(learn vs hook-perform).
> **설계서(빌드 전·GPU 0).** 상위 = `RULE_LEVER_COST_EFFICIENCY_PROGRAM_2026_06_22` + `LLM_CONTROL_EXPERIMENT_REDESIGN_2026_06_22 §4-①`.
> 진입 = 메모리 `06-NOW`·[[05-fixed-vs-variable]]·[[07-control-not-prompt]]·[[11-transfer-direction]].
> 관련 = `R1B_PROVENANCE_DESIGN`·`V7_PROACTIVE_GATHER_DESIGN`·`CROSS_BENCH_TRANSFER_PLAN`·`CFBSYNTH_P2B_P4_DESIGN`·`REST_INTERNALIZE_DESIGN`(데이터-기근 교훈).

---

## §0. 착수 전 [[05]] 결정질문 (이 실험 + C4 학습)

1. **도메인-특화 순증?** — ❌ **C4 학습 = canonical learn 벤치 SOPBench + TaskBench + Synth서만·fetch-first는 Synth의 cfbsynth stratum(추상 synth·P2b/P4)**(★CFB 직접 폐기·`CFBSYNTH_P2B_P4_DESIGN`·`INTEGRATED_TBOX v2`). tau2(retail/airline) = **held-out eval *only***. ⛔ **tau2 학습 절대 금지**(= 2026-06-20 ReST-on-tau2 드리프트·killed·[[11]] 정면위반). 전이 = ABox-swap·재학습0.
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

**fetch-first discipline = "값이 없으면 *생산 도구를 먼저 호출*해 그 출력서 실값을 *복사*·리스트면 매칭 선택·날조 금지"**(P2b gather-for-arg + P4 filter + P1 copy). = COPY stratum(얕음·in-head 가능·`CFBSYNTH §3`).

- **★소스 = 추상 synth(cfbsynth)** = Synth 벤치의 한 stratum (`CFBSYNTH_P2B_P4_DESIGN_2026_06_19`·사용자 결정·확인 2026-06-22). **CFB 직접 아님**(표면매핑 역전이 위험·[[12]])·**SOP/TaskBench 자연발생도 아님.** = CFB의 P2b/P4 *구조만* 추상합성: per-traj **랜덤화 id**(gold id가 오직 getter 출력에만→복사강제·암기불가)·**익명 툴/필드명**(lexical 단축 차단)·hops∈{2,3}·list_n∈{1..5}·표현 다양. 자산 = `ma/synth_fetch_nativefc.py`·`tau2/build_solo_data_cfb.sh`·`build_solo_train_cfb.sh`(이름은 cfb지만 *추상 synth* 파이프라인).
- **★fetch-first는 단일레버 아님 = 3분해**(`CFBSYNTH §9`): **①구조 SHAPE(copy-no-fabricate) = learn(cfbsynth/Synth)** · **②의미 의존(어느 getter가 order_id 생산) = A2(ABox value-type→producer map)·학습불가** · **③집행(상류/user에 없는 id 거부) = hook(provenance 가드)**. ⇒ 이 규칙의 곡선은 *레버 결합*(learn SHAPE + A2 의존맵 + hook 가드)·순수 단일레버 비교가 아니라 *분해된 책임*의 비용배분.
- **★COPY vs COMPUTE 경계**(`CFBSYNTH §3`·resolve_selection과 무충돌): cfbsynth=P4 *filter*(매칭·복사·모델이 id emit)·resolve_selection=P4 *argmax/rank*(계산·offload·엔진 grounds). 모델은 "관찰됐으면 복사, 계산필요면 op명명" 구분 학습.
- **⛔ 금지**: tau2 retail/airline 궤적 학습(`sft_rest_s0_retail.jsonl` 등 = 드리프트·미사용). tau2는 **전이 측정 타깃**.
- **무붕괴**: replay 1:1(일반 tool-use·tbnfc/tb_all) 혼합·small-rank LoRA(r16류). held-out 일반능력 불변 확인.
- **데이터-기근 위험 회피 = 합성**: cfbsynth는 *생성기*라 데이터-기근 없음(REST HOLE 문제 무관·원하는 만큼 P2b/P4 합성). 단 **과대표현 감시**(`CFBSYNTH §8`·6000=synth/taskbench 비등·다른 primitive 희석 곡선서 감시).
- **다양성**([[12]]·치팅면 `CFBSYNTH §5`): per-traj 랜덤 id(복사강제·암기불가)·익명 툴/필드(lexical 단축 차단)·hops/list_n/포맷 변주. ⚠️ *충분한가*는 실 e2e 전이가 판정(C8식 역전이 재발 위험·미확정).

---

## §4. 메트릭 (★y=격리코드·리뷰 #2·#4)

| 메트릭 | 정의 | 왜 |
|---|---|---|
| **A_notfound** (★y축) | entity-id grounding *격리* 실패율(`t2_failcensus` A코드) | ★핵심신호·**곡선 y(global pass 아님)** |
| (보조) pass^1 | e2e DB-match | 전체효과·⚠️B(operand)가 천장 결정→confound 주의·보조만 |
| **flexibility-loss** | held-out 정상경로 **false-block rate**(§4a 라벨셋·enforced arm만) | ★offload 날개의 숨은비용·동결 측정 |
| **A2-growth** | arm이 요구하는 A2 필드 수(delta·**키스톤 정리 후 baseline**) | ★[[05]] minimize-A2 |
| **no-collapse** | held-out 일반 tool-use(tbnfc 등) 불변 | learn 무붕괴(§3 replay) |
| **transfer** | **airline held-out**(동일 LoRA·ABox-swap만) A_notfound | ★[[11]]·cfbsynth 추상합성이라 도메인 무관(retail·airline 둘 다 held-out) |

### §4a. ★flexibility-loss 조작적 정의 (리뷰 #4·빌드 선결·비-GPU)
"false-block rate"는 **held-out 정상경로 라벨셋** 필요(validate over-deny=0은 학습/검증셋이지 held-out 아님·이 메트릭이 offload-날개 비용 기둥 전체를 떠받침).
- **라벨셋 = held-out τ² 태스크 중 *게이트가 발동할 수 있으나 정당한* 툴콜 시퀀스**: ⓐ user가 *이미 실값 제공*(provenance OK인데 gate가 의심?) ⓑ producer 이미 호출됨(중복 fetch 강요?) ⓒ 정상 multi-step write. 각 시퀀스에 "정당(block돼선 안 됨)" gold 라벨.
- **메트릭 = block된 정당콜 / 전체 정당콜**(hook arm: deny/perform이 정상흐름 가로챈 비율). learn/prompt/skill arm = 0(enforced 아님·구조적).
- **소스**: retail+airline held-out split서 정상경로 추출(`t2_failcensus` 정상-종료 sim의 툴콜 시퀀스)·수동 라벨 소량. **빌드 전 이 라벨셋 파일 박기**(`flex_loss_labels.json`·grep tau2-도구명 OK=eval셋이라).

**핵심: learn arm(SHAPE)은 flex-loss=0(weights·구조적)·전이=cfbsynth라 retail·airline held-out.** 단 fetch-first 완전동작엔 ②A2 의존맵·③hook 가드 동반(3분해·§3). A3(autofetch)=A2-growth>0·동결로 ②③ 엔진 대행→flex-loss 측정대상.

---

## §5. GO/NO-GO (autofetch 원칙 결정)

★이건 **두 날개 crossover**(내재화 learn vs offload hook-perform·`PROGRAM §4a/§5(a)`)·단일곡선 knee 아님. y=A_notfound(격리). A2-growth는 **키스톤 정리 후 baseline** delta로(리뷰 #3·오염 회피).
- **GO(autofetch 강등)**: Al(learn) A_notfound ≈ Ah-perform(autofetch) **그리고** A2-growth=0·flex-loss≈0·무붕괴·airline 전이. ⇒ **기본 = hook-deny(가드)+A2(의존맵)+learn(SHAPE)·autofetch(perform) 불필요.** = thesis 핵심(내재화가 offload의 flex/A2 비용 없이 동등).
- **NO-GO(autofetch 정당)**: 오직 Ah-perform만 닫고 Al(learn)·As(skill) stall. ⇒ autofetch를 **measured-justified** 비용옵션 명시(정직 경계)·A2-growth/동결 비용 계상.
- **중간**: 부분 닫음(deny<skill<learn<perform) → 날개별 Pareto 점 보고(가중합 금지).

= `LLM_CONTROL §4-①` A-leg + `PROGRAM` 헤드라인 (a) 첫 시험. autofetch 원칙(§2.3)·C10(B-leg) 동시 정보.

---

## §6. 위험·함정

- **tau2 학습 유혹**(반복 드리프트 1순위): C4 데이터 = cfbsynth(추상 synth·익명 툴)만. 코드리뷰 게이트 = 학습 jsonl에 tau2 도구명 grep=0(애초 익명이라 0).
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

1. ✅ 이 설계서 + 리뷰 픽스(#2 격리 y·#4 flex-loss 라벨셋·#1 두날개 판독).
2. **⚠️ #3 게이팅 결정**(사용자): 키스톤 A2 과성장 정리(placeholders·arg-types→scaffold 기본값)를 **빌드 전 선결**(권장·A2-growth/flex-loss baseline 비오염) vs **병행+주석**. → Ah-perform A2-growth 숫자가 이에 좌우.
3. **flex-loss 라벨셋 박기**(`flex_loss_labels.json`·§4a·비-GPU).
4. C4 데이터 = cfbsynth 생성(`ma/synth_fetch_nativefc.py`·랜덤id·익명툴·hops/list_n 변주·`CFBSYNTH §2`)·과대표현 감시.
5. C4 학습(small-rank LoRA+replay·진행률 가시·무붕괴 check).
6. 7-arm eval(retail+airline·**y=A_notfound 격리**·flex-loss·A2-growth·보조 pass) → 두날개 GO/NO-GO §5.

---

**불변 정합**: [[05]](결정질문3·minimize-A2·tau2학습금지)·[[07]](soft 불충분→C4)·[[11]](전이방향=primitive학습·tau2 held-out)·[[12]](다양성)·[[13]](흡수우선 scale→learn→scaffold). 상위 = `LLM_CONTROL_EXPERIMENT_REDESIGN_2026_06_22`.
