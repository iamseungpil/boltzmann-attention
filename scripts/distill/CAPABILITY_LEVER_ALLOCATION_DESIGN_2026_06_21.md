# 능력×레버 배정 가이드라인 — ★논문 목표 (2026-06-21·사용자 reframe)

> ★**마스터 진입점 = [`EXPERIMENT_DESIGN.md §0★★`](EXPERIMENT_DESIGN.md)** (이 문서가 그 spine·평가항목·마스터 실험·문서정렬은 거기). 메모리 `06-NOW`.

> **사용자 지시**: "기능을 더 세분화하고, 비용-효과 관점에서 각 기능을 효율적으로 *조합할 가이드라인을 설계*하는 것이 논문 목표. + 큰 모델이 하는 일을 작은 모델에서 *더 적은 구조/학습으로* 따라잡는 방법을 제시."
> 상위 = `THESIS_STATEMENT_2026_06_16`(분담 명제·이 doc가 §3 라우팅표를 *완전 가이드라인*으로 확장)·`A2_MINIMIZATION_FRONTIER_DESIGN`·`M_A_RESULTS §35`. 불변 = [[00-thesis]][[05-fixed-vs-variable]][[13-absorption-priority]].

## 0'. ★Payoff — 왜 하는가 (사용자 2026-06-21·연구 동기의 경제적 핵심)
**scale의 진짜 비용 = ① 추론 OpEx(매요청 GPU·$·영구) + ② 하드웨어 인프라 CapEx(GPU *구매*·VRAM 등급·전력·냉각·랙 — on-prem은 선불·직접부담).** 모델 2배=대략 VRAM·GPU·전력 2배+. ⇒ **소형 모델 + 구조화(scaffold/최소학습)로 큰모델과 *똑같이* 하면 → ①②를 동시에 절감**(싼·적은 GPU·낮은 전력으로 동일 성능). = on-prem 배포의 경제적 본체이자 이 연구의 payoff. **가이드라인·방법집의 가치 = "더 작은 하드웨어로 같은 능력" = 인프라 CapEx+OpEx 절감.**

## 0''. ★목적함수 = 4비용 *동시* 최소화 (사용자 2026-06-21·통합)
**minimize { ① GPU 등급/대수(싼 GPU) + ② VRAM(적은 메모리·양자화 포함) + ③ 유지보수(변경당 재구성) + ④ 인간 전문가 } 동시에.**

**★수렴 통찰 (네 최소화가 같은 답을 가리킴):**
- ①② (싼·적은 HW) → **소형 모델**(+ int8/int4 양자화).
- ③④ (저유지보수·저전문가) → **고정 도메인-일반 부분 극대화**(scaffold + 전이·무망각 학습 = 1회 ML엔지·변경흡수) + **per-domain 가변 부분(A2) 극소화** + **도메인-타깃 재학습 금지**.
- ⇒ **공통 적 = A2**(③ recurring 유지보수 + ④ 도메인전문가·현장·도메인당 둘 다 끌어올림) → **A2 최소화가 ③④를 동시에 푼다.** 공통 친구 = **고정 scaffold + 전이 TBox**(①②의 소형화가 요구하는 구조를, ③④ 싸게 충족=1회·무변경).

**★긴장·knee (정직)**: 모델을 더 줄이면(①②↓) 같은 능력에 *더 많은 구조/학습 필요* → 단 그게 **고정·전이형**이면 ③④ 안 오름(1회), **A2/도메인-타깃형**이면 ③④ 폭증. ⇒ 답 = "가장 작은 모델"이 아니라 **총비용(①+②+③+④) 최소 *크기*** = 06-NOW 곡선의 knee. 구조 추가가 ③④를 안 올리게(=고정·전이형으로만) 하는 게 knee를 *작은 쪽*으로 미는 핵심.
- **양자화**(int8/int4)=② 직접 절감 레버(같은 모델 더 싼 VRAM)·단 능력 손실 측정 필요(매트릭스 축에 포함).

## 0'''. ★일반화 비용 = 5번째 축 + 다양-벤치 확장 목표 (사용자 2026-06-21)
실무 다분야 사용(금융→의료)의 핵심: **고정부(모델·scaffold·학습 TBox)는 재사용, 변경부는 *한정·쉽게 swap*.** 분야 바뀔 때 모델/규칙/scaffold 다 바꾸면 = 실패.
- **일반화 비용 = 새 *field* 1개 추가의 한계비용.** 이상 = **A2-swap만**(모델·scaffold·TBox·재학습 0). 측정: N개 *구조적으로 다른* field서 ⓐ고정부 진짜 불변(grep `if field`=0·재학습0) ⓑA2-swap 노력 ⓒ성능 유지.
- **★retail↔airline은 약한 증거**(둘 다 고객서비스·유사구조). **진짜 일반화 = 구조 다른 field 횡단**: 고객서비스(tau2 retail/airline/telecom) × 금융(banking) × 의료/행정(SOPBench healthcare/DMV/bank) × …
- **⇒ 다양-벤치 확장이 연구 목표**(이 일반화 비용을 *실측*하는 instrument). 학습 = 도메인-일반 primitive 벤치(SOP+TB+Synth+CFB·[[01-four-bench-tbox]])·전이 타깃 = 다양 field 벤치. 고정부가 field 횡단 재사용되면 = 일반화 비용 ≈ A2-swap = thesis 최종 증명.
- **목적함수 갱신**: min { ①GPU ②VRAM ③유지보수 ④인간전문가 **⑤일반화비용(새 field당 = A2-swap만)** } 동시. ⑤도 A2가 적·고정부 클수록 작아짐 = ③④와 같은 방향(공통 적=A2·공통 친구=고정 scaffold+전이 TBox 재확인·field 횡단으로).

## 0. 논문 목표 (두 기둥)
1. **배정 가이드라인**: agentic tool-use를 *세분 능력*으로 분해 → 각 능력을 *비용-효과 최소 레버*(prompt / 결정론 scaffold / A2-config / 최소학습 / scale)에 배정하는 측정-보정 의사결정 프레임워크.
2. **★스케일-대체 방법집(less structure/learning)**: *스케일이 사는* 능력 각각을, 작은 모델 + *최소* 개입(엔진 offload·최소 전이학습·구조변경)으로 따라잡는 구체 방법 — **"scale은 단일체 아니라 *분해가능 번들*이고, 각 조각이 scale보다 싸게 얻어진다"** 를 실증.

## 1. 신규성 (rival 회피)
**"작은>큰"(ToolOrchestra가 다툼) 아님.** 기여 = (a) 측정-보정 *배정 가이드라인* + (b) *스케일-대체 방법집* + (c) 명제 "scale 능력은 decidable/behavior/irreducible로 분해되고 대부분 scale보다 싸게 설치된다 — *어디까지 가능하고 어디서 막히는지*(genuinely scale-bound) 경계를 측정." = systematization + cheap-replication + honest boundary.

### 1a. ★ToolOrchestra(2511.21689) 정독 대조 — 우리 whitespace 확인 (2026-06-21·본문 정독)
ToolOrchestra 본문 실측: action space = "GPT-5, GPT-5-mini, Qwen2.5-Coder-32B, Qwen2.5-Math-72B, search, code"·"calling a large model (GPT-5) in only ∼40% of the steps"·"**No explicit capability matching**...Tool selection **emerges from RL**."
| 축 | ToolOrchestra | 우리 |
|---|---|---|
| 오케스트레이션 단위 | **per-step**(ReAct 루프·매 스텝 도구/모델) | **per-function**(C1-C12 세분 능력) |
| 선택 대상 | **모델+도구**(frontier 포함) | **방법**(scaffold/prompt/소형학습/escalate) |
| 결정 | **학습 RL·black-box·emergent** | **명시·측정된 cost-method** |
| 결정론 scaffold(=모델 0) | ❌(항상 모델 추론) | ✅(decidable→엔진) |
| frontier | ✅ GPT-5 40% step | ❌ on-prem only |
⇒ **선행 = 문제/스텝별 *모델* 라우팅(frontier·emergent). 없음 = 기능 세분 + 각 기능의 *가장 싼 방법*(결정론 offload 포함) *명시* 배정 + on-prem fleet.** = 우리 기여의 핵심 whitespace(확인됨).

## 2. ★세분 능력 분류 (atomic·정의속성·스케일거동·증거)
| # | atomic 능력 | 정의 속성 | 스케일 거동(실측) | 레버 |
|---|---|---|---|---|
| C1 | provenance-check (arg값∈{user,tool}?) | **decidable 술어** | — | **scaffold** |
| C2 | producer-식별 (값 X의 생산도구) | **decidable**(I/O 스키마) | — | **scaffold/A2** |
| C3 | fetch-first **default**(날조 대신 가져옴) | scale-emergent *behavior* | A 76→23→3 | **scaffold-autofetch** or 최소학습 (←scale 대체) |
| C4 | dependency/sequencing 순서 | **decidable**(data-dep DAG) | — | **scaffold** |
| C5 | identity-before-scoped(인증선행) | **decidable**(policy precond) | — | **A2(gate_spec)+scaffold** |
| C6 | precondition/policy gate | **decidable**(policy replay) | — | **A2(gate_spec)+scaffold** |
| C7 | confirm-before-write | **decidable**(gate) | — | **scaffold** |
| C8 | error-recovery(진단→다르게 재시도) | scale-emergent *behavior* | too_many_err 7B36→32B0 | ★방법집(§5)·scale 대체 후보 |
| C9 | selection-resolution(기준→item) | **decidable**(resolve 엔진) | — | **scaffold** |
| C10 | **operand/value-formalize**(NL서 옳은 값/속성) | **irreducible NL→formalize** | ✗plateau 83→62 | **최소 전이학습**(유일 잔여) |
| C11 | flow-rule following(일반 규칙 적용) | promptable@small | D 7B −4 | **prompt** |
| C12 | NL communication | scale-emergent | INFO 47→7 | scale or 방법집(§5) |

## 3. 레버 비용 = ★다축 모델 (단일순서 금지·사용자 2026-06-21: on-prem·인간전문가·변경범위 추가)
| 레버 | build(1회) | 도메인당 TCO | **인간-전문가** | **on-prem 가능성** | 효과(크기) |
|---|---|---|---|---|---|
| prompt | 최저 | 도메인-일반~0 | 프롬프트엔지(저) | ✓(데이터반출 무) | **작은모델 무효**·32B만 |
| scaffold(엔진) | 중(1회) | **~0**(고정) | SW/ML엔지 **1회**(상각) | ✓✓(도메인-일반·오프라인 구축·반입) | 전 크기(decidable) |
| A2/ABox | — | **recurring tail** | ★**도메인 전문가·도메인당·현장** | ⚠️**최악**(데이터 반출불가→frontier 생성기 못씀→현장 수작업) | — |
| 학습(LoRA) | 고(GPU) | 전이·무망각=~0 / 도메인타깃=高 | ML엔지 **1회**(전이시) | ✓(벤치서 오프라인 학습·반입·고객데이터 학습0) / ✗(도메인타깃 재학습은 현장GPU+ML) | irreducible에 |
| scale | 추론비↑영구 | 0 | ~0 | ⚠️(현장 GPU 예산·하드웨어 상한) | 큰 능력 일괄 |

**★on-prem이 비용순위를 바꿈**: A2가 build축에선 중간이나 **on-prem TCO에선 *최악*** — ①데이터 반출불가→frontier A2-생성기(ATA-like) 사용불가 ②도메인 전문가가 *현장*서 수작업 ③변경마다 재저작. ⇒ **A2 최소화가 on-prem 핵심 가치**(`A2_MINIMIZATION_FRONTIER` 정합). 반대로 scaffold·전이학습은 **오프라인 구축→반입·고객데이터 불요**라 on-prem 친화.

⇒ **결정규칙 = "능력별, 효과 있는 *최소비용* 레버" (build × TCO × 인간전문가 × on-prem가능성 × 변경범위 × 효과)**. on-prem 배포선 = scaffold·전이학습 선호·A2 최소화·prompt(크기되면)·scale(하드웨어 되면).

## 3b. ★변경-영향 매트릭스 (유지보수 비용 = 변경 시 무엇을 재구성하나·사용자 핵심)
| 변경 이벤트 | scaffold | 전이학습(LoRA) | A2/ABox | prompt |
|---|---|---|---|---|
| 새 도메인(retail→airline) | **무변경** | **무변경**(ABox-swap) | ★재저작(도메인 A2 전체) | 재튜닝(도메인특정시) |
| 정책 변경(동도메인) | 무변경 | 무변경 | gate_spec 일부수정(부분) | 규칙텍스트 수정(부분) |
| 새 도구 추가 | 무변경(스키마구동) | 무변경 | 카탈로그 추가(부분) | 일부 |
| 카탈로그 변경(신상품) | 무변경 | 무변경 | 데이터 갱신(부분·자동가능) | 무변경 |
| 새 능력 필요 | 엔진 확장(1회) | ★재학습(전체·고가) | — | 규칙 추가 |
- **★핵심**: scaffold·전이학습은 도메인/정책/도구/카탈로그 변경을 **무변경 or 부분**으로 흡수 = 유지보수 최저. **A2가 변경 비용 대부분을 떠안음**(부분이나 recurring)·**재학습은 "새 능력"에서만 전체비용**(드묾). = [[05-fixed-vs-variable]] 고정/변경 경계를 *유지보수 매트릭스*로 정량화. ⇒ 가이드라인은 *변경이 부분-재구성으로 닫히게* 레버 배정(전체 재빌드 회피).

## 3c. ★운영·실무 비용 (OpEx/operational·사용자 2026-06-21: 실무·운영비 포함)
build(CapEx)와 *별개*. 배포 후 매 요청·매일 드는 비용 = 소형-on-prem 가치의 핵심.
| 레버 | 추론 OpEx(매요청) | 레이턴시 | 신뢰성/감사가능 | 통합 노력 |
|---|---|---|---|---|
| prompt | +토큰(컨텍스트↑·recurring) | +소 | 확률적 | 사소 |
| scaffold | 결정론 계산 싸나 **+도구 왕복**(autofetch=producer 1호출/grounding) | +왕복당 | ★**결정론·감사가능**(규제 on-prem 필수) | 엔지니어링 |
| A2/ABox | ~0(config 로드) | 0 | config-구동·추적가능 | 로딩 |
| 학습(LoRA) | base+LoRA(vLLM 오버헤드 미미) | 미미 | 확률적이나 설치됨 | 서빙 인프라 |
| **scale** | ★**GPU/$ 영구 최고**(2배+ 모델=2배+ 추론비) | ★**최고**(큰모델 느림) | 확률적 | — |

**★결정적 — scale은 build엔 싸도(그냥 큰모델) *운영비 영구 최고***: 32B는 7B 대비 추론비·레이턴시·GPU상한 영구 부담. **소형+scaffold+최소LoRA = OpEx 최소**(amortized build·싼 추론) = on-prem TCO 승리의 본체. ⇒ 가이드라인은 **build 싼 lever가 아니라 *전 생애주기(CapEx+OpEx+유지보수) 최소* lever**를 배정. scale은 "build 공짜"로 보이나 OpEx에서 패배(소형이 동능력 도달하면).
- **실무 trade-off 명시**: scaffold autofetch는 도구 왕복=레이턴시 추가(공짜 아님)·단 결정론·감사가능이 규제 on-prem서 가치. agentic 멀티턴은 레이턴시 민감 → scaffold 개입수 최소화도 비용축.

## 3d. 전 생애주기 비용 종합 (레버 선택의 실제 목적함수)
**총비용 = HW인프라 CapEx(GPU구매·VRAM·전력·냉각·랙) + build(인간전문가) + OpEx(추론·레이턴시·매요청) + 유지보수(변경당 재구성범위) + on-prem 페널티(데이터반출·현장전문가).**
- **★HW인프라 CapEx = scale의 숨은 최대비용**(on-prem 선불): 7B는 단일 소형GPU·낮은 VRAM/전력 / 32B·72B는 다중·고급 GPU·고전력·냉각. **소형+구조화가 큰모델 능력 도달 = 더 *작고 싼 하드웨어*로 동일 = CapEx+OpEx 동시 절감(§0' payoff).** scale은 이 축에서 영구 패배(소형 대체가능 능력 한정).
- A2 = build 중 + **유지보수·on-prem 페널티 최악** → 최소화.
- scaffold = build 중 + **OpEx 저·유지보수 최저·on-prem 친화** → 선호(단 레이턴시 왕복 주의).
- 전이학습 = build 고 + **OpEx 저·유지보수 저(전이)·on-prem 친화** → irreducible에 선호.
- scale = build 무 + **OpEx 영구 최고** → 소형 대체가능하면 패배.
- prompt = 전부 저 + **효과가 크기-게이트** → 크기 되면 1순위.

## 4. ★배정 가이드라인 = decidability-first 의사결정 절차
```
각 능력 C에 대해 (비용 오름차순 단락평가):
 1. C가 decidable 술어/계산(스키마+policy+state서 도출가능)인가?  → SCAFFOLD(엔진). [scale-불변·도메인당~0]
 2. 아니면 C가 도메인-특정 사실(카탈로그/정책/vocab)인가?         → A2/ABox.       [환원불가 도메인비용]
 3. 아니면 C가 배포 크기에서 promptable인가?                      → PROMPT.        [효과있으면 최저]
 4. 아니면 C가 scale-emergent behavior이고 크기 감당되나?         → SCALE or §5 방법집.
 5. 그 외(irreducible NL→formalize·promptable 아님·scale-bound)  → 최소 전이학습(LoRA·무망각). [최후·잔여]
```

## 5. ★★스케일-대체 방법집 (둘째 기둥·"더 적은 구조/학습으로 scale 따라잡기")
스케일이 사는 능력(C3 fetch-first·C8 recovery·C12 NL)을 *최소* 개입으로:
- **C3 fetch-first default**: scale 없이 **엔진 autofetch**(provenance-deny→producer 결정론 호출→실값 주입). = decidable offload·학습 0·구조 최소. **실증중**(S-min arm1a: A 33→15·arm1b 진행). ⇒ scale의 최대선물(A)을 *엔진 한 조각*으로 대체.
- **C8 error-recovery**: 후보 = (a)결정론 retry-controller(같은 실패호출 차단·diff 강제) (b)reflection scaffold (c)최소 LoRA(recovery-default). **딥리서치(w23rp33zx)가 어느 게 싸고 무붕괴인지 조사중.**
- **C12 NL-comm**: 일부 genuinely scale-bound 가능(정직 경계) — 딥리서치 판정.
- **C10 operand**: 최소-rank LoRA + replay + 무망각(딥리서치 §3) — 유일 학습 잔여.
- ★공통 제약(사용자): **less structure / less learning** = 최소개입·붕괴망각 0·일반화 손상 0. 딥리서치가 PEFT/steering/distill/구조변경 중 최소비용·무붕괴 방법 확정.

## 6. 실증 = 능력×레버×스케일×**field** 매트릭스 (보정)
- 축: 능력(C1-C12) × 레버(5) × 크기(7/14/32B·frontier·**양자화 int8/int4**) × **field(구조 다른 분야)**.
- **field 다양성**(일반화 비용 §0''' 검증): 고객서비스(tau2 retail/airline/telecom) → 금융(banking) → 의료/행정(SOPBench healthcare/DMV/bank) → … 각 field서 고정부 불변·A2-swap만 확인.
- 측정됨: 스케일 분해(§M_A_RESULTS §35)·프롬프트 크기의존·provenance gate(A 33→15)·operand plateau·C3 autofetch(§35b).
- 진행: 딥리서치(C8/C10 방법·무붕괴).
- 펜딩: C8 retry-controller·operand 최소LoRA·**field-횡단 전이(retail→banking→healthcare ABox-swap·일반화비용 실측)**·양자화 능력손실·각 셀 GO/NO-GO.

## 7. GO/NO-GO (가이드라인 검증)
- **GO**: 각 scale-bound 능력이 §5 방법으로 *작은모델+최소개입 = 큰모델*에 도달(무붕괴·전이) → 가이드라인 실증·"scale 분해가능" 입증.
- **NO-GO(정직 경계)**: 어떤 능력이 §5 어느 방법으로도 작은모델서 회복 불가 → **그게 genuinely scale-bound** = 논문의 정직한 한계선(이것도 기여=경계 측정).

## 8. 정직 한계 (bitter-lesson 대비)
일부 능력(C12? 복합추론?)은 scale-bound로 남을 수 있음. 논문 주장 = "*전부* 싸게 대체"가 아니라 **"어느 능력이 어느 레버로 얼마에 회복되고, 어디가 진짜 scale 벽인지의 *측정된 지도*."** = 과주장 금지·경계가 곧 기여.

## 9. ★응용 타깃·포지셔닝 = Palantir 레퍼런스 (사용자 2026-06-21·딥리서치 wo3v3kmw3)
**비전**: **Palantir 결정론(Foundry Ontology) + frontier의 agentic customer AI를, *최소 비용*으로 달성** (소형 on-prem + 고정 scaffold + 최소 A2 + 전이 TBox).
- **매핑**: Palantir Ontology(객체·속성·링크·action·function·governance) ≈ 우리 **결정론 scaffold + per-domain A2** / AIP-LLM(ontology-grounded agent) ≈ 우리가 **소형+TBox**로 싸게 얻으려는 agentic 능력.
- **차별(가설·딥리서치 검증)**: Palantir = frontier-LLM + *전문가-집약 ontology 저작*(forward-deployed engineer·고객당 구축). 우리 = **소형 모델(싼 HW) + 학습된 도메인-일반 TBox(전이) + 최소 A2(field-swap)** → ① HW CapEx/OpEx↓ ② 전문가 노력↓(고정부 재사용·A2만) ③ field 횡단 일반화(config-only) ④ 데이터 반출 0. = "Palantir-급 결정론·거버넌스 + frontier-급 agentic AI를 최소 비용에."
- **응용 타깃 = 금융 마케팅·의료 운영 시스템**: 고정부(모델·scaffold·TBox)는 그대로, **최소 수정(A2 config)** 으로 배치. 사람 노력↓·운영비↓·변경비↓ = §0''/§0''' 목적함수 그대로.
- **딥리서치 wo3v3kmw3** = Palantir Ontology/AIP/배포·비용·전문가구조 정독 → 우리 레버 매핑·whitespace(어디서 더 싼가)·Palantir가 진짜 강한 곳 확정. 회수 후 이 §9 채움.
- ⚠️ 정직: Palantir governance/scale/통합 성숙도는 강함 — 우리 주장 = *비용*(소형·최소전문가·field 일반화)이지 기능 우위 아님. 경계 측정.

## 10. ★Cost-optimal heterogeneous fleet — L-vs-E 크로스오버·난이도 사이징·싼 라우터 (사용자 2026-06-21)
model-size를 **per-capability 레버**로 추가 → 능력별 *효과 있는 최소비용 tier*. ToolOrchestra(문제/스텝별 emergent 모델라우팅·frontier)와 구별 = **기능별 명시 cost-method + on-prem fleet + 결정론 offload 우선**(§1a).

### 10.1 L-vs-E 크로스오버 (사용자 둘째 통찰·핵심)
능력 X마다:
```
cost_L(작은모델에 학습 내재화) = 학습 CapEx / 빈도 + 작은모델 OpEx
cost_E(적정 큰모델에 escalate) = escalation OpEx × 빈도 + 큰모델 CapEx 분담
→ min(scaffold, prompt, cost_L, cost_E) 선택  (scaffold=decidable이면 0순위)
```
- **빈번+싸게 학습가능 → L** (학습 상각). **희소+학습 비쌈 → E** (작은 몫 위해 큰 학습 CapEx 쓰지 마라·그냥 라우팅). ← 작은 몫은 큰 모델에.
- ⚠️ **"큰 모델 특화 학습"은 보통 불필요** — 큰 모델은 크기로 zero-shot 처리 → 작은 몫이면 *as-is 라우팅*이 최저(추가 학습 CapEx 0). 특화 학습은 그 몫이 충분 빈번할 때만.

### 10.2 난이도-사이징 fleet (ToolOrchestra와 차별)
- frontier 0·**on-prem fleet**(예 10×7B + 4×32B + 1×72B)을 *난이도 분포에 맞춰* 사이징. 대부분 쉬움→7B 多·측정된 5-10% 하드→72B 소수.
- **동질 all-72B보다 쌈**(용량을 수요 분포에 매칭). "CapEx 함정"은 *불필요한* 큰 tier 추가 때만 — 측정된 하드 fraction에 사이징하면 정당.
- ToolOrchestra(40% frontier)와 달리 **5-10%만·on-prem·measured residual에만.**

### 10.3 성립 조건 (make-or-break)
1. **라우터 싸고 정확**: "어느 tier 필요"를 *작은* 판정기(decidable confidence/난이도 신호)로. 큰 모델이 라우팅 결정 금지(비용 그대로). 오라우팅(쉬운걸 큰모델·하드를 작은모델)이 절감 잠식. ← 연구 sub-problem.
2. **하드 fraction 진짜 작아야**(측정)·큰 tier를 그 fraction에 사이징.
3. fleet 운영복잡도(서빙·라우팅·유지보수·§3c/④) bounded.

### 10.4 우리 실험이 크로스오버를 *실측*
- C3 grounding = 7B+엔진(L 승·tier↓ 실증 §35b). C8 recovery = 측정중(c8_eval). operand = L(최소LoRA) vs E. path-selection = H_abstract(작게) vs H_offload(외부탐색)·둘다 235B 상시추론 아님.
- ⇒ **fleet 믹스 = 정적 가정 아니라 능력별 크로스오버 *데이터*로 결정.** = 06-NOW "②총비용-최적 크기"를 "**총비용-최적 fleet 믹스**"로 일반화.

### 10.5 thesis 진화 (정직·더 방어가능)
"작은 모델이 *전부*"(데이터상 거짓·operand/path 일부 scale-bound) → **"작은모델+scaffold가 *대부분*, 측정된 환원불가 소수만 적정-크기 on-prem 모델에, 총비용 최소"**. 큰 모델을 *쓰는 게* 기여가 아니라 *안 쓰게(최소화) 만드는* 게 기여.
