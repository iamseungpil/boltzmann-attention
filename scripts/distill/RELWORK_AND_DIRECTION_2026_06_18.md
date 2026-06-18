# 선행연구 리뷰 + 방향 명시 (2026-06-18) — 무엇이 이미 확정됐고, 우리 고유 기여는 무엇인가

> **목적**: 통합 TBox/Scaffold 설계(`INTEGRATED_TBOX_DESIGN_2026_06_18.md`·`INTEGRATED_SCAFFOLD_IMPL_DESIGN_2026_06_18.md`)를 *구현 착수 전*에, 딥리서치(52 검증 claim·~20 출처) + 핵심 4편 primary 재검증으로 정렬한다. **coworker 공유용 자립 문서** — 아래 §1만 읽으면 thesis 프레임이 선다.
> **한 줄 결론**: 우리 시스템의 *결정론 leg*(게이트·복구·소형+symbolic·deferral)는 **전부 이미 발표됨**(특히 [2603.20449]가 우리 게이트를 τ-bench에서 선점). 남은 고유 기여 = **(1) 학습된 content-op 라우팅의 ABox-swap 전이 (2) 사전고정 술어로 decidable-비율 *측정* (3) 멀티-facet verdict 결합** 셋뿐. ⇒ 실험을 이 셋에 집중.

---

## 1. thesis 한 문단 (coworker용 프레임)
자연어 멀티턴 요청을 도메인 온톨로지(ABox)로 재해석해 native function-calling 시퀀스를 추론·실행하는 agentic planner를, **작은 모델 weight(TBox)에 학습**시키고, 본 적 없는 도메인은 **ABox 교체만으로 재학습 0 전이**한다. 분담 = **LEARN**(LLM·NL→formalize·도메인일반·전이) / **PROVIDE**(ABox·도메인특정·swap) / **DETERMINISTIC**(decidable: gate·resolve·verify·고정). 벤치 = τ²-bench·SOPBench·TaskBench·ComplexFuncBench. 측정 = 실 τ² user-sim e2e + 전이 매트릭스.

---

## 2. 검증된 선행연구 (신뢰 티어 명시)

**티어 표기**: 🟢 primary 정독 검증(WebFetch) · 🔵 cutoff(2026-01) 이전·확립 · 🟡 딥리서치 surface·primary 미검증(인용 전 확인 요).

| arxiv | 제목 (약칭) | 날짜 | 티어 | 우리와의 관계 |
|---|---|---|---|---|
| **2603.20449** | Solver-Aided Verification of Policy Compliance (Winston·Winston·Just) | 2026-03 | 🟢 | ★**gate-leg 최근접 rival** — NL 정책→SMT-LIB→Z3 런타임 게이트·**τ-bench**·위반 차단 |
| **2510.16381** | ATA: Neuro-Symbolic Autonomous Trustworthy Agents (Peer·Stabinger) | 2025-10 | 🟢 | ★**thesis-framing 최근접 rival** — LLM=NL→형식 KB·symbolic engine 결정·소형>대형·결정론·injection 면역 |
| **2604.07036** | ReDAct: Uncertainty-Aware Deferral for LLM Agents | 2026-04 | 🟢 | calibrated-threshold defer(소형→대형)·ALFWorld/MiniGrid |
| **2606.01416** | Self-Healing Agentic Orchestrators | 2026-05 | 🟢 | monitor→diagnose→recover→verify 결정론 루프·fault-injection 98.8% |
| **2511.21689** | ToolOrchestra | 2025-11 | 🟢 | ★**가장 위험한 rival** — 8B RL이 **τ² 80.2%@10.3¢ > GPT-5 77.7%@31.3¢**·HLE 37.1%>35.1%·미관측 도구=*학습* 일반화 |
| **2510.16381** | ATA Neuro-Symbolic (Peer·Stabinger) | 2025-10 | 🟢 | LLM=NL→형식 KB·symbolic 결정·**자동 72.94<gemini-pro 76.50·human-KB 87.17만 추월**·단발 보험 reasoning |
| **2603.20449** | Solver-Aided Policy Compliance | 2026-03 | 🟢 | NL→SMT-LIB→Z3 게이트·**airline만**·위반 50%→29%·**human-번역**·**게이트 only** |
| 2402.01817 | LLM-Modulo (Kambhampati, ICML'24) | 2024 | 🟢 | LLM=근사 지식원 + 외부 sound verifier 루프(LLM 자기검증 불가)·"생성+외부검증" 정전 이름 |
| 2407.01032 | Overcoming Common Flaws in Selective Classification Eval (Traub) | 2024 | 🟢 | ★측정 규율 — **AUGRC**(generalized risk-coverage 곡선)·단일점 게이밍 |
| 2502.17216 | Intermediate Languages Matter (neurosymbolic) | 2025 | 🟢 | NL→형식 IR 선택이 1차 결정변수·context-aware 인코딩만 효과·**단발 reasoning(ProntoQA/ProofWriter)** |
| 2509.25370 | AgentDebug / Where LLM Agents Fail | 2025 | 🟢 | AgentErrorTaxonomy·귀인+repair·+24%/+17%/최대 26%·**ReAct(ALFWorld/GAIA/WebShop)=별 벤치족** |
| 1902.06349 | Learning to Infer Program Sketches (SketchAdapt, Nye) | 2019 | 🔵 | 학습 sketch + symbolic hole-fill·경계 *학습*(우리 델타축)·⚠️PDF 렌더 실패·기존 thesis로 신뢰 |
| 2107.11277 | Machine Learning with a Reject Option (survey) | 2021 | 🔵 | (h,r) 예측기+거부기·3분류(separated/dependent/integrated) |
| 2410.10347 | Unified Routing and Cascading for LLMs | 2024 | 🟢 | "cascade routing"·**학습 quality estimator가 핵심 인자**·model-selection deferral(verdict 아님) |

⚠️ **인용 규율(메모리 `40-settled-cite-only`·`feedback-arxiv-citation-discipline`)**: 본 표 🟢 = 본 세션 primary 정독 완료(2026-06-18). 딥리서치가 surface했으나 본 문서 미사용 post-cutoff snippet(2603.04474·기타 2602/2603류)은 사용 시 primary 검증. SketchAdapt(🔵)만 PDF 렌더 실패 — 2019 확립 논문이라 기존 thesis(경계 학습)로 신뢰하되 인용 시 재확인.

---

## 3. 버킷 A — 이미 *확정*됨 (재증명 금지·인용으로 대체)

0. **★소형 모델이 우리 *바로 그* τ²-bench서 GPT-5를 추월(80.2% vs 77.7%)·~1/3 비용** = **ToolOrchestra 2511.21689** (monolithic 8B RL). ⇒ ★**"소형이 대형 tool-use 성능 도달"은 우리 헤드라인이 될 수 없다 — monolithic-learned로 이미 됨, 그것도 우리 벤치에서.** 우리 차별은 *결과*가 아니라 *방식*(아래 §6).
1. **NL 정책 → 형식제약 → 런타임 결정론 게이트가 위반 차단·정확도 유지 (τ-bench airline)** = **2603.20449**. ⇒ 우리 **GateInterpreter의 gate-compliance leg는 novelty 아님 = 재현.** "게이트가 작동함" 단독 실험 불요.
2. **LLM=NL→형식 + 결정론 symbolic engine + verify/swap 형식 KB + 결정론 + injection 면역** = **ATA 2510.16381**. ⇒ 이 *framing*은 이미 실증됨(단 단발 보험 reasoning·자동은 대형 *못* 넘고 human-KB만 넘음·도메인마다 KB 재구축 = 우리 델타 여지 §4).
3. **calibrated-threshold deferral이 대형 품질을 일부 비용에 달성** = **ReDAct 2604.07036**. ⇒ "불확실하면 미루기·calibration"은 확정.
4. **결정론 monitor→diagnose→recover→verify 루프가 cascade 오류 대부분 복구** = **Self-Healing 2606.01416** + **AgentDebug 2509.25370**. ⇒ 우리 facet_check→regen·오차교정 결합은 *패턴으로 확정*.
5. **selective-classification 게이밍 + 사전 threshold + 다중-threshold 곡선(AUGRC) 규율** = **Traub 2407.01032**. ⇒ 측정 방법론 확정 = 채택만(발명 아님). 우리 decidable-비율 = AUGRC식 곡선으로 보고.
6. **모듈식 에이전트의 cascade 오류는 실재·정량** = 2503.13657·AgentDebug. 확정.

---

## 4. 버킷 B — 유사하나 실제 델타 있음 (논문서 델타 *명시* 필수)

**vs 2603.20449 (gate-leg 최근접·검증)** — 같은 NL→형식→런타임 게이트·같은 τ-bench. 단 **airline만·게이트 compliance만·정책 human-번역·전이 없음·decidable-비율 미측정**.
- 우리 델타: (a) 그들은 **게이트 compliance only** — 우리는 + content-op resolve(argmax/substitute) + threading + 학습 라우팅. (b) **유한 typed-gate kind 폐포**(P5/P6/P8) vs 임의 per-policy SMT. (c) **ABox-swap 무재학습 전이** vs 정책마다 human SMT 번역(= 우리가 없애려는 A2 비용·그들 한계로 명시됨).
- ⚠️ 정직: gate-leg 자체("게이트가 위반 줄임 on τ-bench")는 그들 것 = 인용. 우리는 "게이트는 한 facet일 뿐 + 더 일반·전이·content-op 포함"으로 포지셔닝.

**vs ATA 2510.16381 (framing 최근접·검증)** — 같은 "LLM→형식 + 결정론 엔진 + verify/swap KB + 결정론." 단 **단발 보험청구 reasoning(tool-use 아님)·자동은 대형 못넘음(72.94<76.50)·human-KB만 추월·도메인마다 KB 재구축·decidable-비율 미측정**.
- 우리 델타: (a) **학습된 content-op 라우팅이 *전이*함**(§21) — ATA는 순수 encode-then-solve. (b) **멀티턴 tool-use 벤치**(ATA=단발 claims). (c) **decidable-비율 *측정***. (d) **ABox-swap 무재학습**(ATA=KB 재구축·human 보정). (e) **완전 자동서 우위**(ATA 자동은 대형 못넘음).
- ⚠️ (a)가 실재 델타이려면 *학습 라우팅이 순수 encode 대비 가치 더함*을 보여야 → C0 실험.

**vs ReDAct 2604.07036** — 둘 다 calibrated-threshold defer.
- 우리 델타: 우리는 **결정론 엔진으로 defer(decidability)**·그들은 **대형 모델로 defer(cost)**. META_DECIDE의 defer 대상이 모델이 아니라 엔진.

---

## 4.5 ★전이(transfer) 전용 — 남이 한 것 vs 우리 것 (논문 핵심이라 별도)
"전이"는 세 질문으로 쪼개야 명확: **(1) 무엇이 전이하나** (학습 스킬? 프롬프트?) · **(2) 새 도메인엔 뭘 바꾸나** (가중치 재학습? KB 재구축? 데이터만?) · **(3) 얼마나 멀리** (도메인 내? 벤치 횡단?).

| | 무엇이 전이 | 새 도메인엔 뭘 바꾸나 | 학습 스킬 전이? | 완성? |
|---|---|---|---|---|
| **ToolOrchestra** 2511.21689 | 미관측 *도구* | **아무것도 안 바꿈**(모델이 도구설명 읽음·학습일반화) | ✅ 단 monolithic·black-box RL | ✅**완성**(τ² 발표) |
| **ATA** 2510.16381 | 보험 도메인 간 | **symbolic KB 재구축**(human 검증·= A2 비용) | ❌ 학습스킬 없음(프롬프트 encoder) | ✅**완성**(발표) |
| **2603.20449** | (전이 안 함) | 정책마다 **human SMT 번역** | ❌ | airline만·전이 없음 |
| **우리** | 도메인 **+ 벤치 횡단** | **ABox만 교체**(선언적 catalog+gate_spec) | ✅ 학습된 TBox 라우팅 | ❌**미완성**(C0 동전던지기) |

**🟰 이미 남이 한 것(우리 novelty 아님)**: "가중치 재학습 없이 새 도메인 적응" 자체. ToolOrchestra(학습일반화)·ATA(KB 재구축)가 이미 함. ⇒ "무재학습 전이"라는 말만으론 우리 것이 아님.

**🆚 우리만 다른 것(3축)**: ① 바꾸는 단위 = **선언적 ABox 교체**(ToolOrchestra=아무것도 안 바꿈·먼 도메인이면 재학습 필요 / ATA=KB 재구축=A2 비용). ② 전이 범위 = **벤치 횡단**(남들=도메인 내). ③ 전이 주체 = **학습된 라우팅 스킬**(ATA=프롬프트·스킬 없음).

**★불편한 진실**: 위 차별 3축이 정확히 *아직 미증명* 부분이다. 이미 증명된 우리 전이(synth `§21` retail+airline 0.44)는 (a) op-IR 포맷(`§23E`로 native서 깨짐→축①미확보) (b) 도메인 *내*(retail/airline 둘 다 τ²→축②미확보) (c) 전수본상 cross-domain은 **결정론 scaffold가 나르고 학습 adapter는 held-out≈0**(`SOP:583`→축③약함). ⇒ **증명된 전이는 rival과 덜 구별되고, 우리만의 전이(벤치횡단·ABox-swap·학습스킬)는 아직 결과 없음.** C0(native 라우팅 전이)+벤치횡단 매트릭스 = **논문 존립 실험**(양성이면 3축 동시 확보·음성이면 rival과 구별 안 됨).

## 5. 버킷 C — 우리 고유 whitespace (선행 없음 = 실험이 *세워야* 함)

1. **사전고정 verdict-튜플 술어로 decidable-비율을 *측정***. 확인: 2603.20449·ATA·Self-Healing 누구도 측정 안 함; SketchAdapt(1902.06349)는 경계를 *학습*함. = **가장 깨끗한 단독 novelty.**
2. **학습된 부분의 ABox-swap 무재학습 전이**. rival = 학습 일반화(ToolOrchestra) 또는 도메인마다 KB 재구축(ATA·2603.20449 human 번역). 우리 = 학습 facet(§21 routing) + 결정론 결합이 **gate_spec/catalog swap만**으로 전이.
3. **typed *멀티-facet* verdict 튜플**(gate×ground×thread×content) 고정 술어 결합. 선행 reject-option은 단일예측/2분류뿐.

---

## 6. ★방향 = 무게중심 재정렬 (이 문서의 핵심 결론)

**(1) 헤드라인을 *결과*에서 *방식*으로 피벗하라.** ToolOrchestra(검증)가 우리 바로 그 τ²서 8B로 GPT-5를 추월(80.2>77.7)·1/3 비용·미관측 도구 일반화까지 했다. ⇒ **"소형이 대형 tool-use 도달"은 더 이상 novelty가 아니다(monolithic으로 됨).** 우리가 주장할 수 있는 건 *어떻게* 도달하느냐의 차별뿐:
- **무재학습 ABox-swap 전이** (그들=RL 재학습 + 학습된 도구표현 / 우리=config swap·0 재학습).
- **측정된 decidable-offload 분담선** (그들=black-box RL policy / 우리=투명한 verdict-튜플 + AUGRC 곡선).
- **결정론 compliance 보장** (그들=형식 보장 없음 / 우리=게이트 구조적 0-위반).

**(2) GateInterpreter는 keystone이 아니다** — 2603.20449가 gate-leg를 이미 τ-bench에 발표. e2e에 *필요한 엔지니어링*이지 기여가 아니다.

**진짜 keystone = (C0) 학습 라우팅 전이 + (측정) decidable-비율 + (대비) ToolOrchestra 차별.** C0가 실패하면 학습-leg novelty 0 → 남는 건 전부 버킷 A(이미 발표된 결정론) → 논문 무붕괴.

---

## 7. 실험 목록 (run / don't-run / cite)

**❌ 돌리지 말 것 (버킷 A·인용 대체)**: "GateInterpreter가 위반 차단" 단독·"소형+결정론이 대형과 경쟁" 일반주장·"recover 루프가 cascade 복구"·"calibrated defer 작동".

**✅ 반드시 돌릴 것 (버킷 C·여기에만 novelty)**:
1. **★C0 keystone (이중 load-bearing)**: native facet3(content-op 라우팅)가 retail→airline 전이를 *native 포맷*으로 §21 동급 재현하나. (배경: `M_A_RESULTS §21`=op-IR 포맷서 0.44·held-out 1.00 / `§23E`=op-IR을 native로 옮기면 깨짐 pass^1 0.075<base.) **실패 시 학습-novelty 0 → 논문 무붕괴 → 멈춤·재검토.**
2. **decidable-비율 측정**: risk-coverage 곡선 + AURC·META_DECIDE 술어 **사전등록**(Traub 2407.01032 규율). 단일 숫자 금지(게이밍).
3. **ABox-swap 전이 매트릭스**: arm2 unchanged·gate_spec/catalog만 swap → retail·airline·SOP-Bench. ATA(KB 재구축)·2603.20449(정책마다 번역)와 대비.
4. **★head-to-head vs ToolOrchestra(검증된 기준선: τ² 80.2%@10.3¢)**: monolithic-learned(8B GRPO) vs 우리 decomposed-deterministic. *정확도로* 이길 필요 없음 — 우리 축 = **무재학습 ABox-swap 전이**(그들=학습 일반화)·**투명한 decidable-분담 측정**(그들=black-box)·**결정론 compliance 보장**(그들=무보장). 이 비교·이 세 축이 빠지면 "ToolOrchestra가 더 단순·강한데?"에 무방비.

---

## 8. 검증 상태 / 잔여 TODO (2026-06-18 정독 완료분)
- [x] **ToolOrchestra 2511.21689** ✅정독: τ² 80.2%@10.3¢ > GPT-5 77.7%@31.3¢·HLE 37.1%>35.1%·벤치=HLE/FRAMES/τ²·GRPO RL(outcome+efficiency+preference)·미관측 도구=trajectory-기반 학습 표현. **확정 = 가장 위험한 rival(우리 결과 헤드라인 선점).**
- [x] **ATA 2510.16381** ✅정독: std 1.07 vs 4.89·Travel 자동 72.94(±0)/human-KB 87.17·gpt-5 68.75·gemini-pro 76.50. **자동은 대형 못넘음·human-KB만 추월·단발 보험 reasoning·KB 도메인마다 재구축.**
- [x] **2603.20449** ✅정독: airline만·위반 50%→29%·human-guided SMT 번역(4 설계·자동 실패)·게이트 only·전이/decidable 미측정.
- [x] **Traub 2407.01032** ✅: 지표=AUGRC(다중-threshold 곡선)·단일 working point 게이밍.
- [ ] **SketchAdapt 1902.06349** — PDF 렌더 실패. 2019 확립 논문·"경계 학습"이 알려진 thesis라 신뢰하되, 인용 시 html/semantic-scholar로 재확인.
- [ ] 잔여 post-cutoff snippet(2603.04474 등) — 사용 시 primary 검증(현재 미사용).

---

## 9. 권위본 포인터
- 설계: `INTEGRATED_TBOX_DESIGN_2026_06_18.md`(§5 분해 아키텍처)·`INTEGRATED_SCAFFOLD_IMPL_DESIGN_2026_06_18.md`(구현·§5 decidable-비율·§9 방어).
- 결과 권위본: `ma/M_A_RESULTS.md`(§21 라우팅 전이·§23D operand 퇴행·§23E native 붕괴)·`reports/facet_rft_2026/{SOPBENCH,TASKBENCH}_EXPERIMENT_RESULTS.md`.
- thesis·경계: `THESIS_STATEMENT_2026_06_16.md`·`DECOMPOSITION_OPTIMALITY.md`·마스터 `EXPERIMENT_DESIGN.md §1`(fact-offload OK·procedure-offload 금지 경계).
- 딥리서치 원본: 세션 워크플로 `wf_30a790e2-566`(52 claim·journal.jsonl).
