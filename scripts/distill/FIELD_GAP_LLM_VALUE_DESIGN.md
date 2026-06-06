# FIELD-GAP / LLM-VALUE 설계서 — "결정론 프로그램 대비 LLM의 효용을 어디서·어떻게 증명하나"

> **신설 2026-06-06.** 동기 = ICLR 리뷰어의 **"이건 그냥 결정론 프로그램 아니냐"** 공격에 대한 정면 방어 + 그 방어를 *정량 실험*으로 박는 설계.
> **상태**: 리뷰 1회 반영 (2026-06-06). 반영: ①§1 메트릭 정정(37=scripted-*oracle* 상한, honest 천장~32, 우리 33=천장 도달 → "결정론이 이김" 철회) ②정면반박(E0 gather-grounding 격리·E5 파일럿)을 *조기 신호*로 격상 ③amortization regime을 비용 헤드라인으로(정적 LOC 격하)+Δ-LOC-per-change ④E1 found/inherited 분리를 선행 게이트로 ⑤τ²→SOPBench 통제 멀티턴 래퍼 primary, τ-bench 외부타당성(dual-control 오염 회피) ⑥실행순서 리스크-조정(E2 안전승리→E0/E5 조기신호→E1→E3). **마스터 = `EXPERIMENT_DESIGN.md`**(목표·순서·지표 권위본; 이 문서는 그 §1/§2의 detail). 관련 = `CROSS_DOMAIN_TRANSFER_DESIGN.md`(Exp-5=A축 transfer) · `RUNG1_SOURCE_LADDER_DESIGN.md`(LOCK·2-agent §11) · `RUNG1_REDESIGN_2026_06_04.md`(decision-axis A/B/C §9). 결과 권위본 = `../../reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md`.
> **메타규칙(승계)**: 강한 주장은 reliable test 후 박제 · GPU 전 zero-cost 진단 · 공식 success로만 보고 · scaffold 도메인-분기 금지 · **offline forced-ACT 헤드룸 측정 금지(§8 RETRACTED 교훈)**.

---

## §0. 한 줄
**SOPBench-기본 regime(구조화 제약을 떠먹임)에서는 결정론 프로그램이 LLM을 대체/추월한다 — 이건 벤치의 입력 제시 방식 탓이지 우리 방법 결함이 아니다. LLM의 대체불가 효용은 "지저분하고·변하고·자연어인 현장 입력 → 결정론 엔진이 먹을 구조"로 변환하는 데 있고, 그 효용을 입력-regime을 현장 쪽으로 돌린 실험(NL-only·변경·멀티턴·벤치횡단)으로 정량 증명한다.**

## §1. 위협 정식화 (the deterministic-program threat) — 정직하게 인정부터
리뷰어 steelman: *"Guard-2가 증명했듯 너희 게이트(DGGATE)는 `task["constraints_original"]`+도메인규칙에서 evaluator의 directed_action_graph를 결정론적으로 정확 재구성한다(OVER=0/UNDER=0). 즉 '어떤 precond·순서'(절차)는 학습이 아니라 결정론 알고리즘이 계산한다. 게다가 너희 ablation이 `adapter-only≈0 → stack 75–95%, scaffold가 전부`라 적었다. 그러면 학습 7B는 ~0 기여, 성능 전부가 도메인-일반 결정론 프로그램이다. scaffold+scripted-gather / scaffold+base-7B 베이스라인은?"*

**내부 증거가 이 공격을 떠받친다(인정)**:
1. **Guard-2 = 절차 결정론 재구성** = 마스터 §1이 금지한 "절차 offload(=답지=L0)"에 근접. `dfsgather_invfunccalldirgraph(constraints_original,...)`는 per-domain 분기 0의 도메인-일반 그래프 알고리즘 = **구조화 입력만 있으면 결정론이 plan을 푼다.**
2. **Exp-5 cross-domain(77.3% 등)은 전부 decision-axis A(결정론 offload)** = 즉 현 헤드라인 transfer 수치는 *결정론 프로그램의 성능*에 가깝다. "scaffold가 전이된다"는 결정론 알고리즘의 도메인-일반성을 확인한 것 = 공격 강화.

**⚠️주의 — "결정론이 정확도서 우리를 이긴다"는 과장(메트릭 정정, 리뷰)**: `run_scripted` **37/48은 명시적으로 scripted-*oracle*(완전 게더)** = oracle-gather 상한이지 *배포 가능한* 결정론 프로그램이 아니다(권위본 `SOPBENCH_EXPERIMENT_RESULTS.md:374`). **정직 천장 = ~32/48**(8 PartA결함+8 자격부재 제거, :388)이고 우리 stack honest **33/48 = 이미 그 천장 도달**. 37은 (a) oracle-gather (b) cred-injection(에이전트 불가, :384)으로 부풀린 비실현 수치다. ⇒ **올바른 문장**: "oracle-gather를 가진 결정론만 37에 닿고, **gather를 스스로 풀어야 하는 *배포 가능* 결정론은 그만큼 못 간다 — 그 gather가 바로 학습되어 전이된 우리 기여(0→43%)다**"(§4·E0와 직결). 즉 메트릭은 공격이 아니라 방어를 떠받친다.

⇒ **결정론과 "정확도 누가 높냐"로 싸우지 말되**(구조화-입력 천장에선 oracle-결정론이 동률권), **진짜 전장 = 배포 가능 결정론이 gather/NL/변경/대화를 못 한다는 것.** 방어는 그 축들에서.

### §1.1 ★L0 비용 감사 결과 (2026-06-06, `diag_cost_audit.py`, zero-GPU) — §1 정량 반박 + 정직 분류
| 측정 | 수치 | 함의 |
|---|---|---|
| **배포 결정론 작성 (assistant.py=순수 SOP 인코딩)** | **~968 LOC/도메인** (7도메인 합 6,777; 백엔드 포함 시 1,546/도메인·합 10,825) | 결정론 SOP-executor의 per-domain 사람 작성 비용 = floor |
| **변경당 Δ-LOC** (operator/조건 1개당, =assist LOC/#actions) | **47–105 LOC** (bank 47·univ 105) | 정책 1조건 add/도구 rename = 결정론 ~50–105줄 수정 = §9 변경축 탄환 |
| **우리 scaffold per-domain 코드** | **0 `if domain==` 분기** (`two_stage_client.py` grep 빈 결과) | **B-1 PASS** — scaffold는 진짜 도메인-일반(0줄/도메인) |
| **우리 ontology_<d>.json 출처** | **INHERITED** (induce가 `domain_assistant_keys`+task `directed_action_graph`+`dep_full` 읽음, **NL 정책 아님**) | "우리 0 작성"은 벤치 NL→구조 노동 상속 ⇒ E1 NL-only는 BLOCKED(NL-source induce 부재) |

- **정직한 1급 반박(메모리 메트릭-규율 준수)**: "우리는 NL서 0줄 작성"은 **거짓**(induce=inherited). 정확한 주장 = **"우리 도메인-일반 scaffold는 per-domain 코드가 *문자 그대로 0*(B-1 검증); 배포 결정론 SOP-executor는 ~968 LOC/도메인 + 변경당 ~50–105 LOC가 든다. LLM이 그 구조를 *NL서* 복원할 수 있는가(현재는 벤치 구조 상속)가 정확히 E1이고, found/inherited 분리 게이트가 선결이다."**
- **남은 감사 gap(minor)**: `domain_assistant_keys`가 module이라 #constraints 분리 카운트 실패(#actions로 LOC/unit 대용). constraint별 Δ-LOC 정밀화는 후속(E2 harness서).

## §2. 진단 — 왜 SOPBench-기본이 LLM을 underplay하나
SOPBench(2503.08669)는 **"구조가 주어졌을 때 SOP를 따르는가"를 테스트하려고** 설계됐다(NL→구조 컴파일 테스트가 아님). 기본 regime이 어려운 부분을 미리 제거한다:
- **(D1) 구조화 제약 떠먹임**: task마다 `constraints_original`(파싱된 정책조건) + 의존규칙 + getter_map 제공 → 결정론 재구성 가능.
- **(D2) dummy-user**: `user_known`을 구조화 덤프로 선제공(leaderboard 표준) → NL 대화 이해 불요. ARGFIX가 인자를 user_known서 결정론 추출.
- **(D3) 고정 도구·고정 정책**: 변경 없음 → 하드코딩이 안 깨짐.

이 셋이 LLM의 대체불가 역할(NL→구조, 대화이해, 변경적응)을 전부 우회시킨다. **효용은 D1–D3를 현장 쪽으로 되돌릴 때만 드러난다.**

## §2.5. ★Capability-source 분해 — 모델-크기 격차가 *어느 하위기술*서 나오나 (E1 동기의 뿌리)
> **핵심 주장**: "Opus 4.8(기업) vs {7B,32B,72B}+우리방법"의 SOPBench 격차는 균일하지 않다. 격차가 가장 큰 두 하위기술이 *정반대 성질* — 하나는 scaffold가 완전히 지우고(=비판이 옳은 영역), 하나는 scaffold가 전혀 못 건드린다(=LLM load-bearing·E1 타깃). **이 분해가 "왜 구조화-regime 격차는 의도적으로 양보하고 NL-only(E1)로 옮기나"를 자명하게 만든다.**

SOPBench success = 6 하위기술의 곱. 각 기술의 (scale 민감도 × scaffold-외부화 가능성)이 다르다:

| 하위기술 | scale 민감도 | scaffold 결정론 외부화? | 7B 단독 | Opus 단독 | 내부 증거 |
|---|---|---|---|---|---|
| **결정 계산** (AND/OR 정책트리 충실평가) | **가장 가파름** | **예**(DGGATE/offload) | fabricate·1-false-leaf 붕괴(Track C NULL) | 거의 충실 | offload→BOTH 6→15→29/34 |
| **순서**(precond establish 순서) | 중간 | **예**(DGGATE) | 부분 | 자연수행 | DGGATE +3 |
| **arg/slot 바인딩**(transfer dual-username) | 중간-가파름 | **예**(ARGFIX 결정론 resolve) | 오염 | 자연수행 | ARGFIX +6 |
| **gather 완전성**(dirgraph 순회) | **완만** | 부분(active-gather) | **학습됨 0→43%·전이** | 자연수행 | LODO 전이 |
| **거부 보정**(should_F, over-refuse 없이) | 중간 | 부분 | over-refuse/over-act | 보정됨 | should_F 회귀모니터 |
| **NL→구조 컴파일**(D1 제거 시) | **가장 가파름** | **아니오** | 미지(아마 못함) | 네이티브 | E1 미측정 |

**두 갈래 결론**:
1. **결정·순서·arg 축 = scaffold가 곡선을 평탄화** → 작은 모델도 Opus 근처(honest 천장 34 도달, cross-domain 75–95%). 즉 **이 격차는 *지식*이 아니라 *신뢰가능 계산* 격차**(Guard-2 OVER=0/UNDER=0가 결정론 계산가능 증명). = **"그냥 결정론 프로그램" 비판이 옳은 영역.** compositionality gap(2210.03350)·self-verify 한계(2402.08115)의 발현.
2. **NL→구조·멀티턴 축 = scaffold가 못 평탄화** → 게이팅할 구조화 입력이 없으니 외부화 불가 → 격차가 **재개방**, Opus ≫ 7B가 **환원불가능**. = **LLM load-bearing·E1 타깃·진짜 헤드라인 숫자.**

**기업 배포 함의**: 구조(dirgraph) 생산 경로는 셋뿐이고 **집행(scaffold)은 셋 모두에 동일** — ① frontier in-context(Opus가 NL→구조+집행 한번에) ② deterministic per-domain authoring(**우리 특허 OISA/CDP**가 이 prong, 변경마다 재작성=§1.1 ~968 LOC/도메인) ③ **학습된 소형 front-end(우리 Agent1/gather)**. ⇒ **Opus-인-기업 vs 소형방법 차이 = 구조를 *누가 만드나*의 차이지 집행능력 차이가 아니다.** ★②③ 모두 우리 IP = **2-prong**(특허=결정론 authoring, thesis=학습 front-end가 ②의 per-domain 재구축 비용 제거). thesis의 정당한 자리 = ③, OISA와 *보완*.

**★규제-기업 반전 (assurance 축)**: bank substrate가 우연이 아님 — 규제산업선 stochastic 모델의 자기-집행을 준수근거로 못 쓸 수 있다. Opus가 네이티브로 잘해도 감사가능·결정론·증명가능 집행이 요구되면 결정론 게이트가 **독립적으로 필요**(LLM-Modulo soundness). ⇒ scaffold+소형은 Opus와 *비용*뿐 아니라 *assurance*서도 경쟁. Opus조차 "확률적으로 따른다"≠"보증한다". **§9 비용축에 assurance(감사가능 집행 유무)를 4번째 차원으로 추가.**

**측정계획 함의(scaling curve, R1)**: 0.5→72B 곡선을 하위기술별로 분리해 읽는다 — scaffold가 *평탄화하는* 곡선(결정/순서/arg) = 비판 옳은 부분 / scaffold가 *못 평탄화하는* 곡선(NL→구조, E1) = LLM load-bearing. **두 곡선 분리가 §8 양면 ablation의 진짜 목적.** NL→구조가 7B/32B/72B 중 어디서 emerge하나 = thesis 헤드라인.

## §3. ★이전 Track A/B/C·LOCK과의 정합 (사용자 요구 — 면밀 평가, 죽은 길 재답습 금지)
> 이 §은 "모델이 NL→구조를 해야 한다"는 본 설계의 핵심 처방이 **이미 7B에서 NULL인 Track C(decision-emission)를 되살리는 게 아님**을 못박는다.

**Track C(=decision-axis C, self-emit) = 7B NULL — 무엇이 죽었나 (LOCK 정밀범위, `RUNG1_SOURCE_LADDER_DESIGN.md` §8-10)**:
- 죽은 것 = **단일 agent가 terminal에 결정(permitted AND-진리도출)을 execution과 얽어서 emit**하는 SFT 스캐폴드(treeval 단일식 → inductive → T1c grounded-permitted). 실패 = 구조 fabrication(실제보다 큰 트리 환각) + AND가 1 false-leaf로 붕괴 + permitted 콜드붕괴. over-refuse/over-call/early-act = MODEL 회귀 = SFT-positive로 불가.
- **★Track C는 *구조화-입력* regime에서 테스트됐다**: emit한 "구조"는 *이미 주어진 leaf*에 대한 tree-eval 결정도출이지 **NL→구조 컴파일이 아니었다.** 즉 NL-only regime(§6 E1)은 Track C의 시험대가 **아니었다.**

**LOCK이 명시적으로 살려둔 것 (over-prune 금지)**:
- **gather-grounding (0→43%, LODO 전이됨)** = 어떤 precond/검증도구를 establish할지 = *부분 구조 생산* — 이미 학습·전이.
- **2-agent Agent1 (구조분리)** = NL→dirgraph 전담 LoRA(GT constraints 검증), Agent2/결정론 실행과 분리 → fabrication 근본원인(얽힘) 구조적 제거. `RUNG1_SOURCE_LADDER_DESIGN.md` §11 = **유일 헤드라인·미-killed**.
- **DPO/RFT (decision-axis B)** = 모델 판단과 결정론 verifier의 *차이*를 음성신호로 교정 = LOCK escape(emission 재답습 아님).

**⇒ 본 설계의 "NL→구조"는 살아있는 두 경로로만 한다**: ① gather-style(alive) ② 2-agent Agent1(분리, 미-killed). **decision-emission terminal(C)은 금지.** 검증은 Guard-2식 독립검증(emit된 구조 vs GT, OVER/UNDER). SFT-positive가 구조생산에 저항하면 → **Track B(DPO/RFT)로 강등**(C 변종 금지).

**decision-axis 재좌표 (이 설계의 위치)**:
| axis | 정의 | 상태 | 이 설계서에서 |
|---|---|---|---|
| **A** 결정론 offload | 시스템이 기록된 raw 결과로 permitted 계산 | LIVE(=Exp-5 stack) | **결정론 baseline = 공격 대상**. LLM-Modulo 인용. 단독 novelty 약 |
| **B** verifier-DPO/RFT 내재화 | 모델 판단−verifier 차이를 음성신호로 weight 학습 | 미시험 | **adapter-only>0 만드는 길 = "학습이 결정론을 내재화·전이"**(§6 E5) |
| **C** self-emit | terminal에 진리도출 emit | **NULL(7B)** | **금지(재답습 안 함)** |
| **2-agent** | Agent1 NL→구조 + Agent2 실행 | 미시험(헤드라인) | **NL→구조 load-bearing의 정당 구현(§6 E1)** |

## §4. ★연구 동기 — LLM만의 대체불가 장점 (결정론이 원리상 못하는 것)
현장은 SOPBench-기본과 D1–D3에서 다르고, 각 차이마다 결정론은 *사람 노동으로만* 대응한다:
1. **NL→구조 컴파일**: 현장 정책은 NL 산문(규정·10k줄 SOP)으로 온다. 결정론은 사람이 절차를 코드로 작성해야. **LLM = NL→dirgraph 자동 컴파일.** (= Logic-LM/PAL의 semantic-parsing 역할.)
2. **지저분한 멀티턴 NL 요청**: 모호·불완전·턴 분산·동의어/패러프레이즈. 결정론 인자추출이 깨짐. **LLM = NL 이해 + 명료화 대화.**
3. **변경 robustness**: 정책 조건 add/remove, 도구 rename/add/remove가 상시. 결정론 = 변경마다 코드 경로 찾아 재작성. **LLM = 바뀐 NL/도구리스트만 교체(코드 0).**
4. **미지·롱테일 도메인**: 모든 도메인에 프로그램 미리 못 씀. **LLM = ABox-swap 재학습0 전이.**
5. **정책 규모**: 10k줄을 per-decision context에 못 넣음. **LLM = 압축 컴파일·내재화.**

**핵심 프레이밍**: LLM의 가치 = "지저분/변화/자연어 현실 → 결정론 엔진이 먹을 구조" 변환. **결정론과 정확도로 싸우지 말고, 노력·변경·자연어 축으로 싸운다** — 거기선 결정론이 원리상 못 따라온다.

## §5. ★기존 연구 동향 비교 + 우리 비판 극복법
> **검증 상태 (deep-research 2026-06-06, `wr7eoeakr`, 25주장 3-vote 적대검증·0 killed)**: 아래 ✅행 = 1차소스 verbatim 확증(vote 표기). ⚠️행 = **미검증**(검증셋 부재; §10-6 정직플래그 + 2차 리서치 대기). **확증된 field-wide 합의**: NL→구조 선행연구는 *전부* 프롬프트(in-context, 비-weight) + 결정론 엔진 offload (ACL'25 survey 2503.18971). ⇒ 우리 novelty는 그 위가 아니라 "구조 *생성*을 소형 weight에 박고 전이"에 있음.

| 계열 | 주장 | 우리와의 관계 / 극복 |
|---|---|---|
| ✅**LLM-Modulo** (Kambhampati ICML'24, `2402.01817`) (3-0) | autoregressive LLM은 plan·self-verify 못함; soundness는 외부 critic서 상속, LLM=후보생성기, weight에 구조스킬 **무귀속** | **A축 인용 권위**. 단 LLM-Modulo는 *LLM이 (프롬프트로) 생성*·weight학습/전이 무귀속. 우리=NL→구조를 weight에 학습·전이, 결정론은 *집행*만(생성 아님). ⚠️**범위한정 필수**: o1/LRM이 "LLM plan불가"를 부분완화(Kambhampati 본인 `2409.13373`=o1 PlanBench "quantum improvement") → 동기를 **"autoregressive+self-critique"로 scope** |
| ✅**Logic-LM / PAL / SatLM / PoT / LINC** (`2305.12295`/`2211.10435`/SatLM `2305.09656`/`2211.12588`/LINC EMNLP'23) (8× 3-0) | LLM=NL→형식스펙(FOL/SMT/Python) 파싱, 솔버=건전 실행; faithfulness=**solver정확도**; **정적추론** 데이터셋(FOLIO/ProofWriter/math) | **★핵심 템플릿(양면 ablation).** **우리 delta**: 이들=프롬프트(GPT)+솔버·정적추론·재학습0전이 無, 우리=agentic tool-use 집행용 dirgraph·*작은모델 weight 학습*·cross-domain/벤치 전이·구조 exact-match faithfulness |
| ✅**LLM+P / Guan'23 / Oswald'24 / L2P** (`2304.11477`/`2305.14909`/`2405.06650`/L2P PLAN-FM'25) (다수 3-0) | 프롬프트로 PDDL action-model(precond/effect) emit → sound planner가 *solve때* 순서 합성; 정확성=planner보증/validator루프 | **최인접(절차구조).** **차별**: PDDL action-model은 planner가 순서를 *런타임 합성*; 우리는 instance-level 순서(login→balance) **직접 컴파일·weight학습**. 전부 프롬프트·재학습0전이 無. ⚠️인용수정: Oswald'24 저자에 **"Wang" 없음**(Oswald/Srinivas/Kokel/Lee/Katz/Sohrabi) |
| ✅**ACL'25 survey "LLMs as Planning Formalizers"** (`2503.18971`) (3-0) | 지배 패러다임=LLM-as-formalizer + 결정론 off-the-shelf planner; LLM은 long-horizon 직접플래닝 취약 | **related-work 앵커**: surveyed norm=프롬프트 formalizer→결정론 planner, **weight-baked 전이가능 컴파일러는 norm 아님** = gap주장 문서적 근거 |
| ⚠️**RoG** (Luo ICLR'24, `2310.01061`) **(미검증)** | 7B가 relation-path *구조 생성* + GT 그래프 distill | 방법론 최인접 후보. **주장 차별**(미검증): RoG=기존 KG 스키마서 path-finding / 우리=NL서 절차 graph-construction; RoG=KG마다 재instruction-tune / 우리=재학습0 전이. **★2차 리서치서 construct-vs-pathfind·재tune 확인 필수(§10-6)** |
| 🔶**OISA / CDP** (★**우리 개발중 특허 시스템** — 기존연구/극복대상 *아님*) | 도메인-특화 파이프라인(코드 AST induce)으로 결정론 구조 생산, 도구변경 시 재구축 | **우리 deterministic-authoring prong** = §2.5 경로② = §8 human-author/det baseline의 구체 인스턴스. thesis(학습 소형 front-end ③)는 OISA를 *극복*이 아니라 **보완** — OISA의 per-domain 재구축 비용(변경마다 재작성, §1.1 ~968 LOC/도메인)을 NL→구조 학습으로 제거. = **2-prong 연구 프로그램**(특허=결정론 authoring·집행 / thesis=학습 front-end). §6 E2 비용대조의 *우리쪽 결정론 arm*(외부 대조군 아님) |
| **process-supervision / verifier-RFT** (PRM 등) | 중간단계 보상으로 추론 교정 | **B축 positioning**: 우리 B = verifier 차이를 음성신호 내재화. process-supervision은 통계우위 불명[Jia ICML'25] → 우리는 *전이*로 정당화 |
| ✅**Self-Ask / compositionality gap** (`2210.03350`) + **LLM self-verify limits** (`2402.08115`) (보강확증) | 서브Q는 답하나 *조합* 실패; LLM self-verify 불가 | 우리 Track C NULL의 *이론적 예언*. → 결정 자기-emit 포기·offload/2-agent 정당화 |
| ⚠️**[미검증 위험축] 프로세스-그래프 추출·event-schema induction** (van der Aa·Friedrich·`2104.06344`) | text→business-process/precedence graph·event schema | **★가장 위험한 "이미 했다" 후보(검증셋 전무).** 학습된 directed-precedence graph면 novelty 약화 → 2차 리서치 필수(§10-6) |
| ⚠️**[미검증 위험축] agent 도구-의존그래프·SOP agent** (FlowBench·AgentOrca·ToolChain·TaskBench) | tool/API 의존그래프 + SOP-following | **★SOPBench-substrate novel 여부를 직접 결정**: NL서 *구성* vs 제공스키마 *path-find*? 프롬프트 vs 학습? 2차 리서치 필수(§10-6) |
| ⚠️**[미검증 핵심] distill된 NL→구조 zero-shot 전이** (`2305.19472`/`2505.17612`/`2510.19429` 등 후보) | 소형모델에 구조생성 distill·전이 | **★우리 핵심 novelty = 현재 "배제에 의한 gap"일 뿐 positively 미확인 = highest-risk.** 2차 리서치가 닫아야 정당주장(§10-6) |

**faithfulness eval (✅ 3-0)**: 검증된 priors의 구조평가는 전부 **behavioral**(plan-set 동치·solver정확도·validator+human 루프; Oswald'24 `2405.06650`·Guan'23). **OVER=0/UNDER=0 edge-level exact-match 구현한 prior 없음** → 우리 Guard-2식 구조 exact-match = 검증셋에 부재한 **더 강한 faithfulness 기준** = 별도 기여축.

**극복 요지**: (a) 결정론-프로그램 공격 = Logic-LM 양면 ablation으로(§8); (b) "LLM 효용 없음" = §4 장점을 §6 regime서 측정; (c) LLM-Modulo 역전 비판 = 2-agent Agent1로 LLM을 생성 자리로 복귀; (d) "프롬프트면 충분(학습 불요)" = 작은모델 weight 내재화 + 전이 delta(RoG·Logic-LM 대비). **⚠️(e) novelty 확정 선결 = §10-6 미검증 위험축(프로세스-그래프·agent-의존그래프·distill전이) 2차 리서치 통과.**

## §6. ★실험 — 현장↔SOPBench 차이를 regime으로 (상세)
각 실험: regime 정의 → 결정론 baseline → 지표 → 사전등록 성공기준 → Track/LOCK 매핑.

### E0. gather-grounding 격리 — ★조기·최저위험 정면증거 (이미 측정된 양성) (리뷰 격상)
- **동기**: 리뷰어 핵심 공격은 "7B=0 기여". gather-grounding(어떤 getter/precond를 establish할지 결정)은 **이미 측정된, 결정론 아닌, 학습되어 LODO 전이되는 양성**(0→43%, LOCK alive). = "7B가 비결정론적인 뭔가를 배워 전이한다"의 *현존하는 가장 깨끗한 증거* — E5(DPO 도박)보다 빠르고 NULL 위험 없음.
- **regime**: 동일 scaffold(A 게이트 고정) 위에서 gather 소스만 교체 — ① 학습 모델 gather ② scripted/random gather ③ base-7B gather. A 게이트는 gather된 결과로만 permit하므로 success Δ = gather 품질 기여.
- **지표**: 공식 success + dirgraph_satisfied(gather 1차지표) + **LODO held-out 전이**(in-domain vs held-out gather 기여 유지율).
- **성공기준(사전등록)**: 학습-gather success ≫ scripted/base-gather, ∧ held-out서 격차 유지. = "결정론 게이트를 고정해도 *학습된 gather*가 성능을 만든다."
- **Track/LOCK**: gather-grounding(alive). C 아님. **E1·E5 전에 1급 증거로 박제** — thesis de-risk.

### E1. NL-only 입력 (D1 제거) — ★핵심 실험 (= 우리 thesis의 정직한 버전)
> **근거 = §2.5**: E1이 핵심인 이유 = NL→구조가 scaffold로 *평탄화 불가능한* 유일 하위기술축(나머지 결정/순서/arg는 결정론 외부화됨). 구조화-regime 격차를 의도적으로 양보하고 여기로 옮기는 정당성 전체가 §2.5 분해에 있다.
- **★(a) NL-정책 정찰 결과 (2026-06-06, 리모트 코드정독) — E1 SOPBench 실현가능(조건부)**:
  - **NL 정책 존재 O·위치 확정**: `policy` = **시스템 메시지 prose**(benchmark가 `bank_assistant.py:instructions`+`action_descriptions`+`action_returns`로 조립; `get_action_full_description`=desc+return). 우리 `_plan_v2`(`two_stage_client.py:561`)가 이미 이를 `policy`로 뽑아 **source=3에 투입** → source=3 = "NL+도구설명만, 모델이 구조 추론"(=E1 메커니즘 일부 이미 존재; gather가 그 추론=alive·학습됨 0→43%).
  - ⚠️**2 결함**: ① **`policy`가 `[:600]` 절단**(`:561`) → 모델이 정책을 거의 못 봄(메모리 "잘라 안 봄"). ② **성격 = render(structured)**: NL이 구조화 spec(설명·instructions)에서 생성 → NL→구조 = "렌더된 정책 파싱" = 정당하나 *낮은 바*(약한 순환, 독립-authored prose 아님).
  - **남은 inherited = GATE만**: 제약 자체는 구조화 predicate 튜플(`action_customizable_dependencies`)이고 source=3는 이를 모델에 안 줌(gather가 추론). 그러나 **DGGATE 게이트가 `constraints_original`(GT 구조) 소비** = 여기가 inherited 잔존.
  - **⇒ E1 SOPBench 실현 = 2 변경**: (i) policy 절단 `[:600]`→full(모델이 전체 NL 봐야 구조 복원) (ii) 게이트가 GT `constraints_original` 대신 **모델 NL-추론 구조(2-agent Agent1)** 소비 → 완전 NL-sourced. Agent1 구조 vs GT = Guard-2식 독립검증. **(Agent1=분리·검증 = 죽은 Track C 아님.)**
  - **판정**: E1은 **SOPBench서 정직 가능**(위 2변경 후 + 약한순환 caveat 명시) + **SOP-Bench(Amazon 독립 free-text NL)가 강한 바** → cross-bench가 E1 강화의 필수.
- **★선행 게이트 (E1 돌리기 전 BLOCKING, 리뷰)**: found vs inherited 분리 절차 확정 必. `induce`(ontology)·`autoderive`(getter_map)가 **벤치의 구조화 산출물(`<domain>_assistant.py`의 constraint 정의·directed_action_graph)에서 추출하면 "우리 0줄"도 벤치 scaffold 상속** = Guard-2 공격이 E1서 재발(구조화 입력 읽기) + 결정론 "0줄" 대조가 조작. ⇒ **기계적 분리**: induce 입력이 (i) NL 정책 텍스트·도구 시그니처(=found, 정당) vs (ii) constraints_original·directed_action_graph 같은 *파싱된 구조*(=inherited, E1서 금지)인지 코드로 판정. inherited 의존이 있으면 그 경로를 NL-only로 대체하거나 E1 범위서 표기. **이 분리가 깨끗하지 않으면 E1 cost 주장 = Guard-2 재현 → 돌리지 말 것.**
- **regime**: `constraints_original`(구조화 제약) **제거**, 입력 = **NL 정책 + 도구 API 설명만**. 구조는 누군가 생산해야.
- **arms**: ① 결정론(NL→구조를 *사람이 작성*) = per-domain authoring 필요 ② 우리(모델이 NL→구조 emit: gather-style 또는 2-agent Agent1, Guard-2식 독립검증 후 결정론 게이트(A)에 투입).
- **지표**: 공식 success + **도메인당 사람-작성 LOC**(결정론) vs **0**(우리, NL은 *발견*) + 모델이 구조화-입력 천장의 회복률(%).
- **성공기준(사전등록)**: 우리 NL-only success ≥ (구조화-입력 stack의 **X%**, X 사전등록 예: ≥60%) ∧ 결정론은 사람작성 0줄에서 ≈0%. = "LLM이 NL만으로 결정론이 사람 없이 못 만드는 구조를 회복."
- **Track/LOCK**: 살아있는 gather/Agent1 경로(§3). **decision-emission(C) 금지.** SFT-positive 구조생산 저항 시 → E5(B축)로.
- **⚠️정직 caveat**: NL→구조가 7B에 어려울 수 있음(gather는 학습됐으나 full NL→dirgraph는 더 큼). 그래서 *정확도 지배*가 아니라 **비용/효용 대조**(사람작성 vs 0)가 1급 주장. 천장 못 채워도 "결정론은 0(사람 없이)"이 결론을 만든다.

### E2. 변경 robustness (D3 제거) — ★가장 방어적
- **regime**: K개 perturbation 주입 — 정책 조건 add/remove, 도구 rename/add/remove(distractor 추가).
- **지표 (2D Pareto)**: x = 적응에 든 **사람 edit(Δ-LOC)**, y = 적응 후 success. 결정론 = (N edit, 성공) / 우리 = (0 edit = NL·리스트만 교체, 성공).
- **성공기준**: 우리 perturbation-후 success가 무편집으로 비-perturbation의 **≥Y%** 유지(Y 사전등록) ∧ 결정론은 편집 없이는 깨짐.
- **Track/LOCK**: A(offload)는 도구리스트만 갱신; gather/Agent1은 NL/desc 재매칭. = coworker plan "도구변경 robust 전이" 축(v1.36/v1.38) 정량화.

### E3. 멀티턴 messy user (D2 제거)
- **regime**: dummy-user → **user_sim**(`--user_model`) + paraphrase/모호/정보누락→명료화 요구. (벤치: τ²-bench가 기질적으로 이 축.)
- **지표**: success + 명료화-질문 정확도(필요시 cred 요청) + 패러프레이즈 강건성. 결정론 인자추출 vs LLM 대조.
- **성공기준**: 멀티턴 success가 결정론 인자추출 baseline 대비 유의 우위.
- **Track/LOCK**: NL 이해 = LLM 고유, 결정론 미해당. credential-binding(alive) 활용.

### E4. 정책 규모 / 미지 task (선택, stress)
- 대형 NL 정책(연결 또는 실 엔터프라이즈 SOP, 10k줄급) + 구조 스펙에 없는 unseen 조건(NL엔 있음). per-decision context 초과 → 컴파일 필수. 결정론은 코드경로 없음.

### E5. ★Track B — verifier-corrected DPO/RFT (adapter-only>0 만들기 = "결정론을 학습이 내재화")
- **regime**: 현 scaffold(A) 행동을 **음성신호(DPO: 결정론 게이트가 deny한 궤적 dispreferred / RFT: 게이트 차이 보상)** 로 weight 학습 → scaffold OFF에서도(adapter-only) 행동 보존되나?
- **지표**: **adapter-only success ↑**(현 ~0 → ?). adapter-only가 오르면 "학습 모델이 scaffold 행동을 내재화" = 결정론-프로그램 공격 직접 무력화 + novelty A↔B 전이.
- **성공기준(사전등록)**: adapter-only success가 base 대비 유의↑ ∧ 전이(held-out)서 유지. NULL이면 = "offload 필수"(LLM-Modulo 강화) — **양 결과 게재가능**.
- **Track/LOCK**: B축(음성신호=LOCK escape, emission 아님). C 아님.

## §7. ★Cross-benchmark 유용성 증명 (SOPBench + SOP-Bench + 제3벤치)
도메인-일반을 넘어 **벤치-일반**으로 = "학습된 NL→구조 스킬이 벤치 경계를 넘는다" + 각 벤치가 다른 현장 축을 커버.
| 벤치 | 성격 | 본 설계서 역할 |
|---|---|---|
| **SOPBench** (Zekun Li, `2503.08669`) | 7도메인, **구조화 제약** 제공, rule evaluator | 메커니즘 증명 substrate(학습원). E1에서 구조 제거해 NL-only로 변환 |
| **SOP-Bench** (Amazon, `2506.08119`) | 12도메인, **free-text NL SOP**(input/decision/outcome 메타·tool I/O·복잡도 7–8/10) | **NL-native 기질** = E1(NL→구조)의 자연 시험대 + **벤치횡단 전이 타깃**(SOPBench 학습→SOP-Bench ABox-swap 재학습0). 8관계 표현적합성 검증완(마스터 §1) |
| **τ-bench** (Sierra, `2406.12045`) | 멀티턴 NL 정책 대화, 실 고객서비스 | **E3 외부타당성 검증만**(부차). dual-control 아님이라 τ²보다 깨끗 |
| (선택) **AppWorld** (ACL'24) | 복잡 agentic 도구사용·실앱 | E4 scale/복잡도 stress |

- **횡단 전이 매트릭스**: 학습=SOPBench(LODO/train-1) → 전이 평가=SOP-Bench ABox-swap **재학습0**. 헤드라인 = "한 벤치서 배운 스킬이 다른 벤치(다른 형식·다른 도메인)서 작동" = LODO보다 강한 주장.
- **각 벤치가 다른 축 증명**: SOPBench=메커니즘·구조화, SOP-Bench=NL→구조·규모, (멀티턴=아래 통제 래퍼). 셋이 §4 장점 1·2·5를 분담 커버.
- **★E3 멀티턴 substrate = τ²가 아니라 SOPBench 위 *통제 멀티턴 래퍼*가 primary (리뷰)**: τ²는 **dual-control**(user_sim이 코어 실행 일부 담당)이라 "역할 한정"으로 상표만 바꿔도 E3 멀티턴 success 지표가 *여전히 오염*된다(마스터 §0.0 피벗-아웃 사유). ⇒ **E3 primary = SOPBench에 우리가 통제하는 멀티턴 래퍼**(paraphrase + 정보 턴-분산 + 명료화-요구; user_sim을 *우리가* 통제 → 에이전트를 깨끗이 측정). **τ-bench/τ²는 외부타당성 *부차* 인용으로만**(dual-control이 문제 핵심이므로 τ-bench가 외부대조로 더 깨끗).

## §8. 베이스라인 (사전등록) — 양면 ablation
결정론-프로그램 공격은 **양면 ablation 표**로만 닫힌다(Logic-LM 템플릿):
| arm | 구성 | 기대(우리 가설) |
|---|---|---|
| **det-scripted** | scaffold(A) + **scripted-gather**(LLM 없음) | 구조화-입력서 강(천장); **NL-only(E1)·변경(E2)·멀티턴(E3)서 실패** |
| **det+base** | scaffold(A) + **base-7B**(SFT 없음) | SFT 기여 격리 |
| **adapter-only** | SFT 모델 + scaffold OFF | 모델 단독(현 ~0; E5로 올리는 게 목표) |
| **stack(ours)** | SFT + scaffold(A) | 구조화-입력 강(=Exp-5); **현장 regime서도 유지** |
| **human-author** | 결정론 프로그램 사람작성 | per-domain LOC/시간 = 비용 baseline(§9) |
- **닫힘 조건**: det-scripted ≫ ours (구조화-입력) 이지만 **det-scripted ≪ ours (NL-only/변경/멀티턴)**, 그리고 **adapter-only ≪ stack** 이되 E5가 adapter-only↑. = 두 컴포넌트(LLM·결정론) 각자 어떤 regime서 load-bearing인지 분리.

## §9. 지표 (4축)
1. **성능**: 공식 success(`evaluator.py:277`, tool_full, BOTH 금지, honest). 도메인·regime별.
2. **★비용/노력 = amortization regime이 헤드라인 (정적 LOC는 약함, 리뷰)**: "0줄 결정론=0%"는 trivially true라 리뷰어가 *"사람 하루 주고 도메인당 한 번 짜면 그 뒤 수백만 콜서 7B보다 빠르고 정확 — amortize해라"*로 한 방에 받아친다. ⇒ **진짜 전장 = `도메인 수 × 변경 빈도 × 콜당 authoring비`의 amortization.** **LLM이 이기는 regime을 전경화**: 롱테일 다도메인 · 고빈도 정책/도구 변경 · 저~중 콜량(authoring 고정비를 못 amortize하는 영역). 우리 = 1회 고정비(scaffold+SFT) + 한계비용≈0; 결정론 = 작성 선형 + **변경마다 재작성**. **핵심 수치 = 변경당 Δ-LOC**(조건 1개 add/도구 1개 rename당 결정론이 몇 줄 바뀌나 vs 우리 0) — 정적 "도메인당 N줄"보다 이게 변경축의 진짜 탄환(E2 Pareto에 직접 투입).
3. **robustness**: perturbation Pareto(누적 Δ-edit vs success).
4. **★assurance (감사가능·결정론 집행 유무) — 규제-기업 축 (§2.5 반전)**: bank substrate가 우연 아님 — 규제산업선 stochastic 모델 자기-집행을 준수근거로 못 쓸 수 있다. Opus가 네이티브로 잘해도 **증명가능·감사가능·결정론 집행이 요구되면 결정론 게이트가 독립적으로 필요**(LLM-Modulo soundness). 이진 지표: {LLM 자기-집행=확률적·감사불가} vs {scaffold 게이트=결정론·감사가능·증명가능}. ⇒ scaffold+소형은 frontier와 *비용*뿐 아니라 *assurance*서도 경쟁; "Opus가 in-context로 하면 되잖아"의 직접 반박(Opus조차 "확률적으로 따른다"≠"보증한다").
- ⚠️ **성능 < 결정론이면 "지배" 아님 = trade-off 프레이밍**(0 작성·무재학습·변경robust 대가로 정확도 X% 양보) — Pareto frontier로 정직 보고. 단 amortization·변경·assurance regime서는 *지배* 주장 가능.

## §10. 정직 범위 / threats
1. **found vs authored vs benchmark-inherited 분류 필수**: 우리 per-domain 산출물(ontology=induced, getter_map=auto-derived, 정책=found)을 "작성 0"이라 쓰려면 각각이 NL/API서 자동도출인지 vs 벤치 구조 상속인지 명시(induce는 벤치 구조 추출이라 "0"이 일부 상속). 안 하면 "너희 0도 벤치 덕" 반박.
2. **offline 측정 함정**(§8 RETRACTED 교훈): forced-ACT full-success는 constraint/database 게이트서 artifact → **실제 rollout eval로만**. E1–E5 전부 live rollout.
3. **검정력**: should_T effective n 작음 → seed·도메인-mix 고정·사전등록.
4. **NL→구조 난이도 미지**: gather는 학습됐으나 full NL→dirgraph는 더 큼 → E1은 *비용 대조*가 1급(정확도 지배 아님), 저항 시 E5(B).
5. **벤치 induce 품질**: SOP-Bench 8관계 적합성(마스터 §1)·τ² 역할한정(§7).
6. **★novelty가 아직 "배제에 의한 gap" = positively 미확인 (deep-research `wr7eoeakr` 정직플래그)**: 1차 리서치(2026-06-06)는 sub-area 1·2·7(NL→logic·PDDL·faithfulness)만 3-vote 확증; **가장 위험한 priors는 검증셋 부재** — ⓐ프로세스-그래프 추출·event-schema induction(van der Aa·Friedrich·`2104.06344`)이 *학습된 directed-precedence graph*인가, ⓑFlowBench·AgentOrca·ToolChain·TaskBench가 NL서 *구성* vs 제공스키마 *path-find*(=SOPBench-substrate novel 여부 직결), ⓒRoG가 KG마다 재instruction-tune·path-find인가, ⓓ**핵심**: distill된 NL→구조가 cross-domain zero-shot 전이한 prior 존재하나(=우리 헤드라인 주장). **이 4축이 닫히기 전엔 §5 ⚠️행·"gap" 주장을 확정 금지.** → **2차 타깃 리서치 진행 중(launch 2026-06-06)**. 결과 통과 시 §5 ⚠️→✅ 승격, 반례 발견 시 novelty 재좌표.

## §11. 실행 순서 (리스크-조정, 리뷰 — 정면반박을 조기 신호로)
> 원칙: ①GPU 불요 먼저 ②안전한 승리로 thesis de-risk ③**정면반박(학습 기여)을 맨 끝 아님 조기에** — 죽으면 thesis 골격을 바꿔야 하므로 일찍 알아야 함.
1. **✅DONE (2026-06-06, `diag_cost_audit.py`) 비용 감사**: 결정론 ~968 LOC/도메인(assist) + 변경당 Δ-LOC 47–105 + B-1 PASS(0 분기) + **ontology=INHERITED 확정**(§1.1). ⚠️induce가 NL 아닌 구조 상속 → **E1 선결게이트 현재 FAIL**(NL-source induce 경로 신설 필요). constraint별 Δ-LOC 정밀화만 E2 harness서 후속.
2. **(L0) E0/E2/E1 harness 설계**: gather 소스 교체(E0) + constraints 제거 토글(E1 NL-only) + perturbation 주입기 + scripted-gather baseline. apply_two_stage_patch에 통합. **E1은 found/inherited 선행게이트(§6 E1) 통과 후만.**
3. **(GPU, 큐 후) E2 perturbation = 가장 안전한 승리 먼저** (무재학습 A축 toggle, 결정론이 변경서 논란없이 깨짐, Pareto 정직) → thesis de-risk.
4. **E0 gather-grounding 격리** (이미 측정된 양성의 1급 격상, NULL 위험 없음) = 정면증거.
5. **E5 Track B 싼 파일럿** (bank 1도메인·소규모 DPO) — **adapter-only가 0서 움직이나 *조기* 확인**. ⚠️Track C가 7B NULL이었으므로 E5 NULL 사전확률 비낮음 → 이게 죽으면 정확도축 ML 기여 약화·thesis 재골격 → **맨 끝 아님 여기서 신호**. (양성이면 full E5.)
6. **E1 NL-only** (야심찬 핵심, 최고위험) on bank → 양성 시 SOP-Bench로 확장.
7. **E3 멀티턴** (SOPBench 통제 래퍼 primary, τ-bench 외부).
8. **E4 scale** (선택).
> 1–2 GPU 불요(큐 병행). 3–8 큐 완료 후. **순서 핵심: E2(안전)→E0/E5파일럿(조기 정면신호)→E1(야심)→E3.**

## §12. 리뷰 훅 (박제 전 확인)
- [x] §1 메트릭 정정(37=oracle-gather 상한·honest 천장~32·우리33 도달; "결정론이 이김" 철회) 반영 (리뷰 1회)
- [ ] §2.5 capability-source 분해가 "scaffold-평탄화 가능(결정/순서/arg=비판 옳음) vs 평탄화 불가(NL→구조=E1=LLM load-bearing)" 두 갈래를 명확히 분리하나? E1·§8 양면ablation·assurance축(§9-4)이 거기서 도출되나?
- [ ] 정면반박(E0 gather-grounding·E5 파일럿)이 *조기* 신호로 배치됐나(맨 끝 아님)? E5 NULL이면 thesis 재골격 인지?
- [ ] amortization regime(도메인수×변경빈도×콜당비)이 비용 헤드라인이고 정적 LOC가 격하됐나? 변경당 Δ-LOC 카운트하나?
- [ ] E1 found/inherited 선행게이트 통과 전엔 E1 안 돌리나(Guard-2 재현 차단)?
- [ ] E3가 τ²(dual-control 오염) 아니라 SOPBench 통제 멀티턴 래퍼 primary인가?
- [ ] E1이 죽은 Track C(decision-emission)를 안 되살리나? (gather/Agent1·독립검증·decision-emit 금지 확인)
- [ ] 양면 ablation(§8)이 det-scripted·adapter-only 양쪽을 *실패*시켜 두 컴포넌트 load-bearing 분리하나?
- [ ] 비용축(§9)이 found/authored/inherited(§10-1)를 정직 분류했나?
- [ ] 성능<결정론 시 trade-off로 정직 보고하나(지배 주장 아님)?
- [ ] τ² 역할이 E3 전용으로 한정됐나(헤드라인 substrate 아님)?
- [ ] 벤치횡단 전이가 재학습0(ABox-swap)으로 정의됐나?

## §13. ★SOPBench 인용 8편 정밀독해 (2026-06-06, 8 병렬 에이전트·원문 인용) + Track B 레시피 + 논문 분할
> Semantic Scholar 인용 = ~8편. **실제 SOPBench서 실험 = FM SO.P 1편뿐**; 나머지 7편은 인용+자기벤치. 각 원문(HTML/PDF) 정독. 이 §이 §5/§10-6의 여러 ⚠️ 미검증 플래그를 **확정**한다.

### §13.1 8편 관련성 맵 (검증 완료)
| 논문 | 분류 | 우리 축 접점 | 위협 / 활용 (확정) |
|---|---|---|---|
| **FM SO.P** (2602.09336) | **경쟁(유일 SOPBench-user)** | A6(학습)·부분 A1(DAG in-context) | **in-domain only·전이 주장 0**(제목의 "cross-domain"=7도메인 동시학습·동일테스트). 순수학습 7B **34.3%**/32B 48.3%(=Qwen72B 동급). **위협**: 학습 7B 34% ≫ 우리 adapter-only ~0. **활용**: 레시피 채택(§13.2). **우리 delta=held-out 전이(그들 없음) CONFIRMED.** |
| **CAP-CPT** (2510.11588) | **보완/baseline(A6 foil)** | A6(정책→weight CPT) | **"정책을 weight 내재화"가 이미 published**(but 32B만·in-domain·변경하면 *더 어려움* "substitution lags in-context"). **위협**: 맨 "내재화" novelty 소멸. **우리 delta(확정)**: 재학습0 전이(그들은 추가학습 필요)+소형모델+변경-trivial. tau-bench+CC-Gen, A1/A3/A5 전무. |
| **SOP-Maze** (2510.08942) | **보완/후보벤치** | A1 동기(route blindness)·A2/A6·messy-NL | NL-native 장문 SOP(5,040 tok)·tool-free 결정벤치·397/3422/23(中文, Meituan). **단 비-인터랙티브**(정적 transcript)→A5 멀티턴 대체 불가. 공개 repo. |
| **SAGE** (2604.09285) | **보완+baseline** | A3 검증(결정론 rule-engine)·A5(멀티턴 적대)·A1/A7 baseline | **그래프를 에이전트에 *제공*("receives the SOP graph described by text")·채점만**=우리 A1(NL서 *생성*)과 반대. **위협**: "Execution Gap"(intent 맞고 action 실패=우리 진단) 먼저 명명·"deterministic graph 채점" published→인용. **우리 delta**: NL서 컴파일+weight 내재화+in-loop 게이트. |
| **ODCV** (2512.20798) | **보완(동기)** | A3(그들 verifier=LLM-judge+게임가능 스크립트=우리 foil)·should_F | **refusal-fragility 먼저 정량**("refuses-mandate-but-violates")→"거부 약하다" novelty 금지. **활용**: 우리 *건전·결정론 verifier*+dual should_T/should_F가 그들이 *없는* 답. (그들 "deterministic checker는 게임당하는 loophole"=우리 동기 강화.) |
| **TOD-ProcBench** (2511.15976) | **보완/A5 후보벤치** | A5(멀티턴 TOD)·A1(condition-action·JSON-tuple 렌더) | A5 deterministic per-turn 라벨(Task1/2). **LOCK 보강**: "포맷 재구성은 소형모델 개선 못함"(우리 결론 외부확증). Amazon, 데이터 링크 미공개. |
| **LogicIFEval** (2508.09125) | **보완(동기)** | A1/A3(faithful 로직실행 천장)·**LOCK/Track C 외부확증** | **가장 강한 외부증거**: faithful in-text AND/OR·loop 실행 = 능력천장(gpt-5 85%·open <70B **<10%**). decision-faithfulness 실패(CFM/STE)·"output 맞아도 logic 틀림"=우리 offload 논거. ⚠️**"unlearnable 아님"**(FT 안함)→LOCK이 이 논문에 기대지 말 것. state-tracker faithfulness 감사 차용. |
| **AgentSandbox/보안** (2505.24019) | **보완(A3 프레이밍)** | A3(Saltzer-Schroeder complete-mediation=reference monitor) | A3 보안-원리 어휘·"no formal guarantee→외부 static backstop" 인용권위. **단** 그들 결정론층=schema만·정책로직은 여전히 LLM-tuned(stochastic)→**우리 delta=full dirgraph 결정론 집행**(schema 아님). 규제-기업 assurance 갭 그들이 남김. |

### §13.2 ★FM SO.P 레시피 → Track B 채택안 (LOCK-호환 확인)
**FM SO.P 레시피 = 대조(contrastive InfoNCE, τ=1) SFT + scoring head, 3-stage 누적혼합(α=⅓)**: ①개념변별(유사어 치환 음성, 1:4) ②행동시퀀스(error-injection reorder/omit/insert 음성, 1:4) ③그래프추론(음성그래프 cycle/precond-removal/edge±, 1:8; DAG in-context). AdamW lr1e-4·batch256·12ep. **DPO/RL 없음**(순수 대조).
- **★LOCK-호환**: FM SO.P는 *대조(정답≻오염)*=음성신호 regime = LOCK이 명시 허용한 길(DPO/RFT). **decision-emission(Track C, dead) 아님.** ⇒ Track B로 *정당* 채택 가능.
- **채택 2모드**:
  - **B-DPO(권장·생성정책)**: gold SOPBench 궤적 vs **FM SO.P 3-연산자**(용어치환·시퀀스 error-injection·그래프 corruption)로 만든 dispreferred → DPO로 *생성 어댑터*가 순서/precond/제약 선호 내재화 → **adapter-only ↑**(현 ~0). 누적 커리큘럼(용어→시퀀스→그래프, α=⅓), 음성비 1:4/1:4/1:8 출발.
  - **B'-critic(LLM-Modulo 학습critic)**: scoring head를 학습해 후보 next-action/궤적 rerank(생성기 위 학습 verifier). A(결정론 offload)와 B' 비교 = "학습 critic이 룰엔진 대체/보완하나".
- **재현 gap(PDF 필요)**: per-stage 데이터 크기·scoring-head 정의. **B-DPO는 scoring-head 불요(선호쌍만)** → gap 회피하며 즉시 채택 가능.
- **차별 유지**: 채택 후 (i) **held-out 전이**(FM SO.P 없음·우리 Exp-5) (ii) **A3 결정론 offload 결합**(그들 없음). FM SO.P 7B 34%(in-domain·DAG in-context) vs 우리 목표(adapter-only 전이).
- **⚠️경쟁 경보**: 학습축에서 FM SO.P(34%)가 현재 우리 adapter-only(~0)를 압도 → **B-DPO가 P2(학습 논문)의 생존선**. E5 파일럿(§11-5)이 곧 이것.

### §13.3 ★논문 분할안 (논점이 2개의 *비충돌* 코어로 갈림)
8편 대조 결과 우리 기여가 **위협도·준비도가 다른 두 축**으로 깨끗이 갈린다 → 2(+1) 논문 분할 권장:

- **P1 — [ANCHOR, 준비됨, 저위협] 재학습0 cross-domain/bench 전이 + 결정론 감사가능 SOP 집행** (Systems/Agents)
  - 코어 = **A2(held-out 전이, Exp-5 데이터 존재)** + **A3(full-dirgraph 결정론 집행=assurance)** + A4(변경robust) + A7(amortization).
  - 全 8편 대비 *un-threatened*: FM SO.P/CAP-CPT 전이 없음·SAGE/AgentSandbox 결정론은 schema/제공그래프뿐·ODCV verifier는 게임가능. 
  - 벤치: SOPBench(학습) → SOP-Bench/SAGE/SOP-Maze 횡단 전이.
  - **가장 빨리·강하게 게재가능.**
- **P2 — [FRONTIER, 고위험, 조건부] 소형모델에 *전이가능* SOP-컴파일 내재화** (Learning/ML)
  - 코어 = **A6(adapter-only>0, Track B=§13.2 B-DPO)** + **A1(NL→구조 컴파일, E1)**.
  - CAP-CPT(32B·in-domain·변경難) 대비 delta = 소형+전이+변경-trivial; FM SO.P(in-domain) 대비 delta = held-out.
  - 위험: E5/E1 NULL 가능(LogicIFEval 천장·Track C 전례). **P1과 분리**해야 P1이 P2 리스크에 안 묶임.
- **P3 — [선택, 포지션/벤치] field-gap 측정** (NL-only·perturbation·멀티턴·cost/assurance)
  - 본 설계서 §6·§9 전체. P1의 평가장(章)으로 흡수 가능, 또는 독립 position/benchmark 논문.

**분할 논리**: P1은 "누가 구조를 만드나 무관하게, *전이*와 *감사가능 집행*이 우리 것" = 8편 누구도 안 함 = 안전. P2는 "구조를 *소형모델이 NL서 학습*" = CAP-CPT/FM SO.P와 정면 경쟁축 = 위험. 섞으면 P1이 P2의 NULL 위험에 인질. **권고: P1 먼저 확정·집필, P2는 E0/E5/E1 신호 본 뒤.**

### §13.4 자기-정정 (마스터 §0 "FM SO.P/CAP-CPT weight-baking 격파")
- ✅**확정 유효**: "전이가 차별점" — FM SO.P=in-domain(전이0), CAP-CPT=새 정책엔 재학습 필요. 
- ⚠️**라벨 부정확**: FM SO.P은 "weight-baking"이라기보다 *대조학습+DAG in-context*; CAP-CPT가 진짜 CPT weight-baking. 마스터 §0 문구를 "전이 부재"로 정밀화 권장.
