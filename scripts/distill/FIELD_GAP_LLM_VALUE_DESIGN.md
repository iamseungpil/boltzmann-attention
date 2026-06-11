# FIELD-GAP / LLM-VALUE 설계서 — "결정론 프로그램 대비 LLM의 효용을 어디서·어떻게 증명하나"

> **신설 2026-06-06.** 동기 = ICLR 리뷰어의 **"이건 그냥 결정론 프로그램 아니냐"** 공격에 대한 정면 방어 + 그 방어를 *정량 실험*으로 박는 설계.
> **상태**: 리뷰 3회 반영 (최종 2026-06-10). **리뷰3(2026-06-10, 핸드오프 대조)**: ①§0 한줄을 §17.9 thesis로 갱신(구 NL-변환 프레이밍 stale) ②§1 천장회계 통일(34=48−PartA8−PartB6 단일 권위; "~32" 철회, 33>32 산술모순이 증거) ③§11 SUPERSEDED→**실행 권위=§18 신설**(E0=Exp-B 흡수·E2=연기[P1 전 필수 복귀]·E5=협업자 이관 disposition 박제) ④TaskBench-LODO=**supporting 전이만** 사전등록(moat-(3)은 cross-bench만, §17.9 리뷰7) ⑤Exp-A stage1="distill"→**gold-SFT** 정정+GT-generator(GPT-4 back-instruct) 순환 caveat ⑥A-0 edge-miss zero-cost 감사를 RFT 전 BLOCKING으로 ⑦§15.4 사활 open 3건(규제원문·bitter-lesson·erosion)을 §18.2에 배정. **리뷰2(§13)**: A=FM SO.P 34% input-parity 미확인→"생존선" 폐기, 바=우리 stack-vs-adapter (§13.2); B=B-DPO 음성 on-policy 1순위·B'-critic 병렬 arm (§13.2); C=P1 "절반만 준비"·§1.1 inherited 위협 상속·cross-bench 미구축→앵커 stress-test (§13.3·§13.5). **리뷰1**: ①§1 메트릭 정정(37=scripted-*oracle* 상한, honest 천장~32, 우리 33=천장 도달 → "결정론이 이김" 철회) ②정면반박(E0 gather-grounding 격리·E5 파일럿)을 *조기 신호*로 격상 ③amortization regime을 비용 헤드라인으로(정적 LOC 격하)+Δ-LOC-per-change ④E1 found/inherited 분리를 선행 게이트로 ⑤τ²→SOPBench 통제 멀티턴 래퍼 primary, τ-bench 외부타당성(dual-control 오염 회피) ⑥실행순서 리스크-조정(E2 안전승리→E0/E5 조기신호→E1→E3). **마스터 = `EXPERIMENT_DESIGN.md`**(목표·순서·지표 권위본; 이 문서는 그 §1/§2의 detail). 관련 = `CROSS_DOMAIN_TRANSFER_DESIGN.md`(Exp-5=A축 transfer) · `RUNG1_SOURCE_LADDER_DESIGN.md`(LOCK·2-agent §11) · `RUNG1_REDESIGN_2026_06_04.md`(decision-axis A/B/C §9). 결과 권위본 = `../../reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md`.
> **메타규칙(승계)**: 강한 주장은 reliable test 후 박제 · GPU 전 zero-cost 진단 · 공식 success로만 보고 · scaffold 도메인-분기 금지 · **offline forced-ACT 헤드룸 측정 금지(§8 RETRACTED 교훈)**.

---

## §0. 한 줄 (2026-06-10 §17.9 정합 갱신 — 구판은 06-06 NL-변환 프레이밍이라 stale였음)
**고정 도구 + 사전 결정론 compute 위에서, 소형 모델이 도구-호출 경로를 *제안*(=coverage)하고, 결정론·검사가능 게이트가 soundness를 보장(audited 제약모델 대비, valid 경로 없으면 fail-safe abstain)하며, 재학습0로 도메인 전이한다. 헤드라인 = 보장(검증가능) soundness 하 *높은 coverage*를 {소형·저비용}×{감사가능 결정론게이트}×{재학습0 전이} *패키지*로 (= precision=1서 recall 최대화, §17.9).** 그 하부 논거: SOPBench-기본 regime(구조화 제약 떠먹임)서 결정론이 동률권인 것은 벤치 입력방식 탓이며(§1–2), LLM의 대체불가 효용은 NL/변경/대화 축(§4·§6)과 coverage(valid-path-finding)에 있다.

## §1. 위협 정식화 (the deterministic-program threat) — 정직하게 인정부터
리뷰어 steelman: *"Guard-2가 증명했듯 너희 게이트(DGGATE)는 `task["constraints_original"]`+도메인규칙에서 evaluator의 directed_action_graph를 결정론적으로 정확 재구성한다(OVER=0/UNDER=0). 즉 '어떤 precond·순서'(절차)는 학습이 아니라 결정론 알고리즘이 계산한다. 게다가 너희 ablation이 `adapter-only≈0 → stack 75–95%, scaffold가 전부`라 적었다. 그러면 학습 7B는 ~0 기여, 성능 전부가 도메인-일반 결정론 프로그램이다. scaffold+scripted-gather / scaffold+base-7B 베이스라인은?"*

**내부 증거가 이 공격을 떠받친다(인정)**:
1. **Guard-2 = 절차 결정론 재구성** = 마스터 §1이 금지한 "절차 offload(=답지=L0)"에 근접. `dfsgather_invfunccalldirgraph(constraints_original,...)`는 per-domain 분기 0의 도메인-일반 그래프 알고리즘 = **구조화 입력만 있으면 결정론이 plan을 푼다.**
2. **Exp-5 cross-domain(77.3% 등)은 전부 decision-axis A(결정론 offload)** = 즉 현 헤드라인 transfer 수치는 *결정론 프로그램의 성능*에 가깝다. "scaffold가 전이된다"는 결정론 알고리즘의 도메인-일반성을 확인한 것 = 공격 강화.

**⚠️주의 — "결정론이 정확도서 우리를 이긴다"는 과장(메트릭 정정, 리뷰)**: `run_scripted` **37/48은 명시적으로 scripted-*oracle*(완전 게더)** = oracle-gather 상한이지 *배포 가능한* 결정론 프로그램이 아니다(권위본 `SOPBENCH_EXPERIMENT_RESULTS.md:374`). **정직 천장 = 34/48**(48−PartA8−PartB6; released cross-check로 확정된 유일 권위 회계 — 구회계 :388의 "~32 = PartA8+자격부재8"은 **철회**: 우리 33/48이 32를 *초과*하는 산술모순 자체가 자격부재-8 분류 오류의 증거이며, "천장40·9unwinnable·11fixable"과 함께 이미 철회된 계보)이고 우리 stack honest **33/48 = 천장-34의 97% 도달**(게이트-사다리 헤드라인은 BOTH 29/34=85%; 분모 혼용 금지, 천장 인용은 이 한 벌로만). 37은 (a) oracle-gather (b) cred-injection(에이전트 불가, :384)으로 부풀린 비실현 수치다. ⇒ **올바른 문장**: "oracle-gather를 가진 결정론만 37에 닿고, **gather를 스스로 풀어야 하는 *배포 가능* 결정론은 그만큼 못 간다 — 그 gather가 바로 학습되어 전이된 우리 기여(0→43%)다**"(§4·E0와 직결). 즉 메트릭은 공격이 아니라 방어를 떠받친다.

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
| ✅**RoG** (Luo ICLR'24, `2310.01061`) (직독 확인 2026-06-06) | LLM이 relation-path 생성 → **기존 KG(Freebase/Wikidata) 위 path-find/retrieve** (NL→새 그래프 *구성 아님*) | **확정 차별**: RoG=기존 스키마서 path-finding(우리=NL서 절차 graph-*construction*); KG-distill instruction-tune. **안 스쿱**(원문: "generates relation paths grounded by KGs"·"retrieve valid reasoning paths from the KGs"). |
| 🔶**OISA / CDP** (★**우리 개발중 특허 시스템** — 기존연구/극복대상 *아님*) | 도메인-특화 파이프라인(코드 AST induce)으로 결정론 구조 생산, 도구변경 시 재구축 | **우리 deterministic-authoring prong** = §2.5 경로② = §8 human-author/det baseline의 구체 인스턴스. thesis(학습 소형 front-end ③)는 OISA를 *극복*이 아니라 **보완** — OISA의 per-domain 재구축 비용(변경마다 재작성, §1.1 ~968 LOC/도메인)을 NL→구조 학습으로 제거. = **2-prong 연구 프로그램**(특허=결정론 authoring·집행 / thesis=학습 front-end). §6 E2 비용대조의 *우리쪽 결정론 arm*(외부 대조군 아님) |
| **process-supervision / verifier-RFT** (PRM 등) | 중간단계 보상으로 추론 교정 | **B축 positioning**: 우리 B = verifier 차이를 음성신호 내재화. process-supervision은 통계우위 불명[Jia ICML'25] → 우리는 *전이*로 정당화 |
| ✅**Self-Ask / compositionality gap** (`2210.03350`) + **LLM self-verify limits** (`2402.08115`) (보강확증) | 서브Q는 답하나 *조합* 실패; LLM self-verify 불가 | 우리 Track C NULL의 *이론적 예언*. → 결정 자기-emit 포기·offload/2-agent 정당화 |
| ✅**proScript** (EMNLP-F'21, `2104.08251`) ★**axis A 최위험** (wi9qegpft 3-0) | T5-XXL(~11B) **fine-tune**으로 시나리오→**partial-order directed graph**(DOT, edge-F1 75.7) = *learned NL→directed-graph* | **점유=learned+NL→graph(2 차별점).** 분리(확정): **전이0**(단일 6.4k 코퍼스)·**commonsense**(SOP/API 아님)·deterministic faithfulness 없음(edge-F1). ★feasibility 증거(모델이 NL→구조 학습가능)로 활용 |
| ✅**PlaSma** (ICLR'24, `2305.19472`) ★**axis C 최위험** (직독 확인) | **distilled SMALL**(770M–11B) procedural planner = goal→**temporally ordered steps** | **점유=distilled-small+procedural(2 차별점).** 분리(확정): **step-시퀀스**(directed dep-graph 아님)·**zero-retrain 전이 무주장**·commonsense·faithfulness 없음. ★proScript와 함께 소형 feasibility 증거 |
| ✅**GRAFT** (`2605.11706`)·**GTool** (`2508.12725`) (wi9qegpft 3-0) | tool-dependency graph **internalize**(weight-learned) | **점유=learned(1).** 분리: GRAFT 그래프=**입력**(special token)·NL구성 아님·전이 무증거(0-3 refuted); GTool edge=**실행로그**서(NL 아님)·"no retrain"=backbone-agnostic≠cross-domain |
| ✅**TaskBench·FlowBench·Amazon`2510.24690`** (wi9qegpft 3-0) — **SOPBench-substrate novel 확정** | tool/workflow graph를 **제공/입력**(TaskBench back-instruct=graph→text)·Amazon=prompted | **NL서 의존그래프 *구성+집행* 자리=비어있음.** FlowBench=워크플로 INPUT 따르기·TaskBench=graph 평가substrate·Amazon=in-context(weight 0). 우리=NL→구성+결정론 in-loop 집행 |
| ✅**기타 비-스쿱** (직독): `2505.17612`(agent distillation 0.5–3B, NL→그래프 아님)·`2510.19429` NeSyPr(symbolic-planner 의존)·Pan'20 cooking workflow(`2008.09151`, 단일도메인·전이0)·Chambers&Jurafsky(unsupervised partial-order, LLM 아님) | — | 각자 ≤1 차별점. **결합 novelty 안 스쿱.** |

**faithfulness eval (✅ 3-0)**: 검증된 priors의 구조평가는 전부 **behavioral**(plan-set 동치·solver정확도·validator+human 루프; Oswald'24 `2405.06650`·Guan'23). **OVER=0/UNDER=0 edge-level exact-match 구현한 prior 없음** → 우리 Guard-2식 구조 exact-match = 검증셋에 부재한 **더 강한 faithfulness 기준** = 별도 기여축.

**★novelty 판정 (확정 2026-06-06, 2 deep-research `wr7eoeakr`+`wi9qegpft` + 4논문 직독)**: **결합 novelty(learned + NL-not-existing-schema + cross-bench-zero-retrain-transfer + deterministic-faithfulness) = genuinely UNOCCUPIED.** 4 차별점을 *동시에* 가진 prior 없음 — 최근접도 ≤2점(proScript=learned+graph / PlaSma=learned+small). proScript·PlaSma는 오히려 "모델이 NL→절차구조 학습가능" feasibility 증거(우리 비운 칸=전이·결정론집행·SOP타깃 정확히 남김).

**극복 요지**: (a) 결정론-프로그램 공격 = Logic-LM 양면 ablation으로(§8); (b) "LLM 효용 없음" = §4 장점을 §6 regime서 측정; (c) LLM-Modulo 역전 비판 = 2-agent Agent1로 LLM을 생성 자리로 복귀; (d) "프롬프트면 충분(학습 불요)" = 작은모델 weight 내재화 + 전이 delta(RoG·Logic-LM 대비). ✅(e) novelty = **확정**(위 판정, ⚠️ 해제).

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
6. ~~**novelty가 아직 "배제에 의한 gap"**~~ → **✅ RESOLVED (2026-06-06, 2차 리서치 `wi9qegpft` + 4논문 직독)**: 4 위험축 전부 닫힘 — ⓐ프로세스-그래프/event-schema(proScript=learned+graph지만 전이0·commonsense·non-deterministic; Li et al·Pan'20·Chambers 동일류) ⓑagent 도구그래프(TaskBench/FlowBench=그래프 *제공*·GRAFT/GTool=기존그래프 internalize·NL구성 아님 → **SOPBench-substrate novel 확정**) ⓒRoG=기존 KG path-find(NL→구성 아님) ⓓdistill 전이(PlaSma=시퀀스·전이0; `2505.17612`·`2510.19429` 비-스쿱). **판정: 결합 novelty genuinely UNOCCUPIED**(§5 novelty 판정행 참조). §5 ⚠️→✅ 전부 승격 완료. **잔여 미세-caveat**: ① axis-C "unoccupied"는 직독 4편으로 닫았으나 landscape time-sensitive(GRAFT~2026 등 신규 arXiv 모니터) ② proScript ~11B=≤13B 경계(소형 차별점은 soft separator). **✅⑤축 추가 (2026-06-10)**: safe-RL shielding/safety-filter/LLM-agent runtime-enforcement 계열 = `SHIELDING_CBF_RELATED_WORK.md` — 좌표 비점유 재확인(ShieldAgent=확률적·AgentSpec/Formal-LLM=spec 손-작성; 능동구동·coverage·전이 전 계열 부재), 단 **포지셔닝은 "새 패러다임 발명"이 아니라 "제어-검증 패러다임의 NL-orchestration 인스턴스화"로 고정**(동 문서 §4)·ShieldAgent/AgentSpec related-work 필수 인용·P1 직전 재스윕.

## §11. 실행 순서 ⚠️**SUPERSEDED (2026-06-10) → 실행 권위 = §18**
> **아래는 06-06 시점 순서(역사 기록, 수정 금지).** §17 벤치 결정·§17.9 thesis 고정 후 실행 큐는 §18로 대체됐고, 여기 등장하는 E0/E2/E5의 disposition(흡수/연기/이관)도 §18-C에 박제됨. 다음 세션은 이 §을 실행 계획으로 읽지 말 것.
> 원칙(승계됨): ①GPU 불요 먼저 ②안전한 승리로 thesis de-risk ③**정면반박(학습 기여)을 맨 끝 아님 조기에** — 죽으면 thesis 골격을 바꿔야 하므로 일찍 알아야 함.
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
- **채택 2모드 (★리뷰2: B'-critic을 후순위 아님 *병렬 arm*으로 격상)**:
  - **B-DPO (생성정책) — ★음성을 on-policy로 (리뷰2 Pushback B)**: ⚠️FM SO.P 3-연산자(용어치환·reorder/omit/insert·그래프 corruption)는 **합성 corruption** = "뻔한 오염"이라 DPO가 *그 특정 오염만 피하는 얕은 판별기*를 학습할 위험(판별 metric↑·생성 adapter-only 不動 = 고전 DPO 실패모드, **Track C 죽은 자리와 인접**). ⇒ **음성 1순위 = 모델 자신의 실패 rollout 채굴(on-policy DPO / rejection-sampling), near-miss**; FM SO.P 합성 연산자는 **augmentation으로만**. on-policy여야 *생성*이 움직인다. 누적 커리큘럼(용어→시퀀스→그래프)·음성비 1:4/1:4/1:8은 augmentation 출발점.
  - **B'-critic (LLM-Modulo 학습 reranker) — ★병렬 arm (리뷰2)**: scoring head 학습해 후보 next-action/궤적 rerank. **A(결정론 offload) vs B'(학습 critic) 정면 비교 = generation-internalization 리스크 0의 더 깨끗한 novelty(LLM-Modulo 직접 확장).** B-DPO NULL 시(Track C 전례+얕은판별 위험상 확률 비낮음) **B'-critic이 P2 학습기여를 구제** → deferral 아님, *동시* arm.
- **재현 gap(PDF 필요)**: per-stage 데이터 크기·scoring-head 정의. **B-DPO는 scoring-head 불요(선호쌍만)** → gap 회피.
- **차별 유지**: 채택 후 (i) **held-out 전이**(FM SO.P 없음·우리 Exp-5) (ii) **A3 결정론 offload 결합**(그들 없음).
- **⚠️⚠️경쟁 "경보" 철회 = input-parity 미확인 (리뷰2 Pushback A — §1 oracle-함정과 동형)**: 직전 "FM SO.P 34% ≫ 우리 adapter-only ~0 → 34%가 생존선"은 **메트릭-규율 트랩**(메모리 반복함정). FM SO.P 34.3%는 **추론-시 DAG in-context 여부가 논문에 不명시**(2026-06-06 PDF 직접확인: "paper does NOT explicitly state test-time input format"·with/without-graph ablation 없음). 만약 그들도 구조를 in-context로 받으면 그건 "학습+구조화입력=34%"이고 **우리 *stack*(역시 구조 받음)과 비교**해야 공정 — adapter-only(구조 無) 대비는 사과↔오렌지. ⇒ **34% 바 폐기.** **P2 생존선 = 우리 자체 *stack vs adapter-only* 대조**(같은 입력regime 내 학습기여 격리)로 세운다. FM SO.P 비교는 input-parity를 그들이 보고하지 않는 한 *불가*(2차 인용으로만).

### §13.3 ★논문 분할안 (논점이 2개의 *비충돌* 코어로 갈림)
8편 대조 결과 우리 기여가 **위협도·준비도가 다른 두 축**으로 깨끗이 갈린다 → 2(+1) 논문 분할 권장:

- **P1 — [ANCHOR, ★절반만 준비됨 (리뷰2 Pushback C)] 재학습0 cross-*benchmark* 전이 + 결정론 감사가능 SOP 집행** (Systems/Agents)
  - 코어 = **A2(전이)** + **A3(full-dirgraph 결정론 집행=assurance)** + A4(변경robust) + A7(amortization).
  - **⚠️C1 — P1이 §1.1 inherited-structure 위협을 *상속***: Exp-5 전이(bank→library 75.8%)는 induce된 ontology가 **벤치 구조에서 상속**(induce가 `directed_action_graph`+`dep_full` 읽음, NL 아님). ⇒ 리뷰어가 §1 결정론-프로그램 공격을 P1에 재조준: "전이가 각 벤치가 떠먹인 구조 위 결정론 알고리즘 = trivial." **P1은 §1 공격에서 자유롭지 않다 → 이 위협이 P1 정직섹션의 *중심*.**
  - **⚠️C2 — 진짜 앵커 = cross-*BENCHMARK* 전이인데 미구축**: "데이터 존재"인 Exp-5는 **SOPBench-내부** cross-domain. §1.1 위협을 무력화하는 건 *다른 스키마 벤치*(SOP-Bench/SAGE)로의 전이뿐이고, 그건 induce가 *다른 벤치 구조*서 작동해야 = 공짜 아님·**미구축·미검증**. ⇒ "준비됨"은 절반만 참; **방어 가능 코어(cross-bench 전이) = 현재 UNVERIFIED.**
  - **⚠️C3 — A3 delta 얇음**: A1(NL-compile)을 P2로 보내면 A3엔 "in-loop full-dirgraph 게이트"만 남아 SAGE/AgentSandbox 대비 delta 얇음. ⇒ **P1 헤드라인 = *전이가 별*, 집행은 substrate** 임을 정직 소유.
  - 全 8편 대비 *전이는* un-threatened(FM SO.P/CAP-CPT 전이 없음)이나 **그 전이가 inherited-structure 위에서임을 cross-bench로 넘어야 실증.**
  - ⇒ **P1 = "집필" 아니라 *앵커 stress-test*** (§13.5).
- **P2 — [FRONTIER, 고위험, 조건부] 소형모델에 *전이가능* SOP-컴파일 내재화** (Learning/ML)
  - 코어 = **A6(adapter-only>0, Track B=§13.2 B-DPO)** + **A1(NL→구조 컴파일, E1)**.
  - CAP-CPT(32B·in-domain·변경難) 대비 delta = 소형+전이+변경-trivial; FM SO.P(in-domain) 대비 delta = held-out.
  - 위험: E5/E1 NULL 가능(LogicIFEval 천장·Track C 전례). **P1과 분리**해야 P1이 P2 리스크에 안 묶임.
- **P3 — [선택, 포지션/벤치] field-gap 측정** (NL-only·perturbation·멀티턴·cost/assurance)
  - 본 설계서 §6·§9 전체. P1의 평가장(章)으로 흡수 가능, 또는 독립 position/benchmark 논문.

**분할 논리 (리뷰2 정정)**: P2("소형모델이 NL서 구조 학습")가 CAP-CPT/FM SO.P 정면 경쟁축=위험인 건 유지. **그러나 P1도 "안전"이 아니라 *조건부 안전*** — P1의 전이가 cross-bench로 §1.1 inherited 위협을 넘어야만 방어된다. 섞으면 안 되는 이유는 동일(P1≠P2 인질). **권고(정정): P1을 먼저 *집필*이 아니라 *앵커 stress-test*(§13.5)하고, P2 설계는 병렬(zero-GPU), E0/E5/E1 신호로 확정.**

### §13.4 자기-정정 (마스터 §0 "FM SO.P/CAP-CPT weight-baking 격파")
- ✅**확정 유효**: "전이가 차별점" — FM SO.P=in-domain(전이0), CAP-CPT=새 정책엔 재학습 필요. ⚠️**CAP-CPT zero-shot 정밀확인 필요(리뷰2 minor)**: 조금이라도 zero-shot 전이되면 A2 delta 좁아짐(현재 agent 독해=substitution "lags in-context, needs more data"=약/무).
- ⚠️**라벨 부정확**: FM SO.P은 "weight-baking"이라기보다 *대조학습*(추론입력 미명시); CAP-CPT가 진짜 CPT weight-baking. 마스터 §0 문구를 "전이 부재"로 정밀화 권장.
- **인용 ID 실존 확인**: FM SO.P 2602.09336·SAGE 2604.09285(미래일자처럼 보이나 오늘=06-06, 에이전트가 HTML 직접 fetch=실존 확인됨).
- **LogicIFEval → P2 평가 *타깃* 승격(리뷰2 minor)**: B-DPO가 7B를 LogicIFEval-류 faithful-logic서 올리면 = P2 강력 결과(동기 인용에 그치지 말 것).

### §13.5 ★다음 단계 (리뷰2 권고 — zero-GPU, 학습 큐 미접촉)
- **(c) 2차 리서치 = 진행 중(`wi9qegpft`), 재실행 금지** — FlowBench/AgentOrca/TaskBench/ToolChain construct-vs-pathfind + distill-전이 prior 타깃(§5/§10-6 미검증축). §13이 SAGE=그래프제공·FM SO.P/CAP-CPT=전이없음으로 대부분 닫음 → wf 완료를 확정용으로 대기.
- **(a) P1 *앵커 stress-test* — ✅probe 완료(§13.6)**: 핵심 미지수("induce가 다른-스키마 벤치서 작동하나") 판정됨 = **SOP-Bench엔 상속할 구조 없음(NL-only)·도메인-불변 실행기 cross-bench 전이 VALIDATED(TSR≤1.0)·단 ABox 손작성(결정론 prong)**. ⇒ P1 앵커=실행기-전이는 실재, *NL-sourced ABox*가 미구축. 다음 = sop.txt→ontology 자동-induce 시제품(§13.6 최고레버).
- **(b) B-DPO/B'-critic 설계 병렬(zero-GPU)**: 음성=on-policy 1순위·합성 augmentation·B'-critic 병렬 arm(§13.2). GPU 비면 즉시 E5 파일럿(메타규칙 "정면반박 조기신호").
- **순서**: (a) P1 앵커 stress-test → 동시 (b) E5/B-DPO 설계 → `wi9qegpft` 결과로 §5 확정. 학습 큐 미접촉.

### §13.6 ★(a) SOP-Bench induce-feasibility probe 결과 (2026-06-06) — §1.1·C2·E1 동시 판정 + 기존 cross-bench 인프라 발견
**SOP-Bench 네이티브** (clone `/home/woori/scratch/SOP-Bench`): per-domain = **`sop.txt`(free-text NL SOP, human-authored prose)** + `toolspecs.json`/`tools.py` + `test_set_*.csv`(**input→output**). **파싱된 directed-action-graph/dependency 구조파일 = 0**(GT=input→output, 그래프 아님). ⇒ **상속할 벤치-구조가 없다**(SOPBench와 대조: 거기선 induce가 `directed_action_graph`+`dep_full` 상속).

**★이미 구축됨 (이전 세션 05-31, 미커밋)**: `run_domain.py`(TBox=`workflow_executor.CallGraphExecutor`, **도메인-불변**) + `abox/ontology_<domain>.json`(8-관계 ABox: start/steps/realizes/arg/produces/precondition/next/terminate/output) 12–14 도메인. **실행 완료·TSR**: customer_service **1.0**(156/156)·traffic_spoofing **1.0**(200/200)·warehouse **1.0**(150/150)·email_intent 0.92·order_fulfillment 0.87 등.
- ⚠️**그러나 `model:null`·tool_accuracy 전부 0·실행 μs단위** = **LLM 0, 순수 결정론 실행기**가 *손작성* ABox를 돌림 = §2.5 **②prong(deterministic authoring, OISA류)의 cross-bench 실증**이지 학습/LLM 아님. `?? abox/`(untracked)·sop.txt 자동-induce 스크립트 부재 = ontology **손작성(authored, A7 비용)** — found(NL-auto)도 learned도 아님.

**3 질문 동시 판정**:
1. **§1.1 (found/inherited)**: SOP-Bench엔 **상속할 벤치-구조 없음** → SOPBench式 inherited 문제 *부재*. 단 현 ABox=**authored(손작성)** → "found(NL-auto)" 주장하려면 sop.txt→ontology **자동-induce 신설 or learned(E1)** 필요.
2. **C2 (cross-bench 앵커)**: **도메인-불변 TBox 실행기 cross-bench 전이 = VALIDATED**(14 SOP-Bench + 7 SOPBench, 단일 실행기·ABox만 swap, TSR≤1.0). ⇒ **리뷰2 C2 "cross-bench 미구축"은 부분 정정**: *실행기-전이는 구축·검증됨*; **NL-sourced ABox만 미구축.** 가치=단일 실행기가 21도메인 불변 + ABox≪executor-rewrite(A7); 단 ABox 손작성→TSR 1.0 자체는 결정론(§1 공격 순수형, 정직 소유).
3. **E1 (NL→구조)**: **ENABLED & FORCED**(구조 shortcut 없음). 검증=end-to-end TSR(GT graph 없어 **Guard-2 불가**). 손작성 ontology = 도달가능 *gold-target* ⚠️**단 test-blind 아님**(§13.7 provenance probe: ABox가 test_set 디스크화 후 작성·email_intent는 답파일 런타임 read) → gold-target 주장 격하, 검증은 **sop.txt-blind induce + held-out test split**로 재설계.

**⇒ P1/P2 재좌표**:
- **P1** = "**단일 도메인-불변 SOP 실행기 + per-domain ABox가 2벤치/21도메인 전이**" = systems 코어, **이미 실증**(ABox 손작성=A7 정직소유). §1 방어=실행기 일반성+ABox≪rewrite.
- **P2/E1** = **ABox를 sop.txt(NL)서 자동-induce or 학습 생성** → §1.1 완전해소(authored→found/learned) + "결정론 프로그램 아님" 입증. 손작성 ontology가 gold.
- **★최고 레버리지 다음 수 = `sop.txt → ontology` 자동-induce 시제품(1 도메인)**: 성공 시 §1.1·E1·P2 동시 진전 + A7 손작성비용 제거 실증. (zero-GPU 또는 1 LLM 호출.) ⚠️**§13.7 provenance 게이트 통과 후 재설계된 프로토콜로**(test-blind induce·runtime-clean 도메인 선택·held-out split·"손작성과 동급" 검증 폐기).

### §13.7 ★(B) ABox provenance probe 결과 (2026-06-06, 리모트 `ssh_run.py`) — BLOCKING 게이트 판정: TSR 1.0 ≠ blind 일반화
리뷰3 BLOCKING("손작성 ABox가 test_set에 blind인가")을 리모트 실측. **판정 = NOT blind-by-construction; gold-target 격하.**
- **Timeline (smoking gun)**: test_set+sop.txt mtime=**05-30 23:30**(벤치 네이티브) vs `abox/ontology_*.json` mtime=**05-31 01:05–01:45** → **ABox가 `test_set_with_outputs.csv` 디스크화 후 작성 = 작성자 답 GT 접근가능**(blind 아님).
- **email_intent = hard coupling**: `abox/email_intent_functions.py`가 **런타임에 `test_set_with_outputs.csv`를 read**. 정직 docstring: 벤치가 mis-wired(toolspecs↔tools.py 도구라우팅 불일치로 *어떤 도구도 실행 불가* + 결정신호 `email_body`를 input 컬럼으로 미노출)라, SOP의 "email_body 의도분류" 단계를 답파일서 *input 필드(email_body·listing_price·inventory)*를 복구해 SOP 규칙으로 분류. **input-only 주장이나 답파일을 읽음**·규칙이 test-tuned일 수 있음(100% 가시). 단 이 도메인은 0.92(1.0 아님).
- **1.0 도메인은 runtime-clean**: customer_service·traffic_spoofing·warehouse(=TSR 1.0)는 abox functions가 **csv 미참조**(grep 전수=email_intent만) → 실행기+ontology가 답파일 없이 해결. **단 author-time test-가시**(01:xx 작성)라 ontology가 test-informed일 가능성 미배제(certified-blind 아님).
- **TSR 계산 = 벤치 자체 `amazon_sop_bench.evaluate()`**(GT 대조) → 메트릭 자체는 정당; 단 그걸 1.0 만든 ABox가 답-가시로 작성됨.
- ⚠️**부수 발견 = SOP-Bench 벤치결함**: email_intent mis-wiring(도구 실행불가·input 누락) = SOPBench PartA/B式 데이터결함 존재 → 자동-induce 시제품은 **clean 도메인(customer_service/traffic/warehouse) 선택 필수**.

**⇒ 판정·재설계**:
1. **gold-target 격하**: "손작성 TSR 1.0 = 도달가능 gold" → **test-visible authored(certified-blind 아님)**. "auto-induce가 손작성 TSR과 동급인가" 검증기준 **폐기**(오염타깃 매칭=무의미).
2. **auto-induce 재설계 프로토콜 (리뷰4 정정)**: ① **sop.txt만**서 induce(test_set blind, 코드로 격리) ② **runtime-clean 1.0 도메인**(customer_service 등, email_intent 제외; ★먼저 14도메인 mis-wiring 전수 audit해 clean 확정) ③ 평가=벤치 `evaluate()` on **held-out test split**(induce가 안 본 행) ④ **바(bar) = blind-induce의 *절대* held-out TSR**(높은 TSR을 안 본 행에서 내나) — ⚠️"손작성 따라잡나"는 **폐기**(step1 "오염타깃 매칭=무의미"와 자기모순). 손작성은 별도 *오염 진단*으로만: **손작성 seen-vs-held-out TSR 갭**(0=contamination 무해·일반절차 / 큰갭=row-특화 leakage). ⑤ **방법 = frontier LLM 1-call 먼저**(prose→ontology, 포맷-일반, NL→구조 feasibility 확정) → **그 다음 소형모델**(=E1/P2 실제 주장). ⚠️rule-parser(sop.txt 포맷 exploit)=format-overfit·위장된 손작성 → sanity floor로만.
  - **★silver lining (리뷰4)**: runtime-clean 도메인이 도메인-불변 실행기+컴팩트 ontology로 1.0 = **타깃 구조가 8관계로 표현·실행가능 + 실행기 도메인-일반**임은 *증명됨*(오염과 무관). ⇒ "타깃 도달가능성" de-risk; **유일 미해결 = blind인가**. ("전부 오염"보다 정확.)
  - **email_intent 정밀 라벨 (리뷰4, 과대자책 방지)**: 답 *라벨* copy 아님 — 벤치가 안 뿌린 *입력*(email_body)을 test파일서 복구+손작성 규칙 분류 = **test-파일 커플링 + 규칙 overfit 가능**(answer-copy 아님). 그래도 미인증-blind라 BLOCKING 유효.
  - **★probe-2 (리뷰5, 코드 직독 2026-06-06) — silver lining·"runtime-clean" 둘 다 정정 필요**: `functions.py` 전수 census = **12 도메인 중 11이 per-domain `functions.py` 보유(27–131줄), order_fulfillment만 JSON-only=0.87**. ★이들은 plumbing 아니라 **SOP 결정로직**(customer_service=`is_authenticated`/`metrics_improved`/`final_status`, SOP §5.1/5.5 임계값 latency≤100·jitter≤30 + status 결정트리; warehouse=resolution 매핑). ⇒ **(a) silver lining 정정**: 1.0은 "컴팩트 8관계 ontology"가 아니라 **JSON 스켈레톤 + 27–131줄 손작성 결정 Python**이 운반(order_fulfillment JSON-only=0.87이 증명). "타깃=8관계 표현가능"은 **과소특성화** — 실제 타깃 = NL→(ontology + **결정코드**). **(b) "runtime-clean" 정정**: csv 미참조여도 **warehouse functions.py docstring이 label-coupling 자백** — "labels were frozen from the CSV `problem_type`/`chargeable` columns", 제공 도구가 po_number서 recompute한 값이 **라벨과 47% 불일치**라, 라벨-결정 input 컬럼을 *직접 read*하고 GT-generation을 reverse-engineer해 매칭. ⇒ runtime-clean(csv 미read)≠uncontaminated; warehouse는 email_intent보다 *깊은* 커플링. **(c) auto-induce 타깃 격상**: NL→JSON이 아니라 **NL→(ontology+결정코드 생성)** = frontier-1-call이 customer_service의 final_status 결정트리·임계값을 Python으로 emit해야 = 훨씬 가파른 bar. **(d) A7 비용축 추가 침식**: per-domain authoring = JSON + 27–131줄 결정 Python ≈ OISA류 per-domain 코드(="ABox≪executor-rewrite" 약화). **(e) 프로토콜**: 시제품 도메인 = **customer_service**(결정로직이 SOP §명시 인용·라벨컬럼 아닌 정당 state input[latency/jitter/auth] read = 최-clean) but 타깃에 결정코드 induce 포함. warehouse는 label-coupling으로 **clean셋서 제외**, email_intent도 제외 → clean 1.0 도메인 = **사실상 customer_service·traffic 중 functions.py가 label-컬럼 안 읽는 것만**(traffic도 docstring audit 필요).
3. **C2 앵커 재격하(리뷰2·3 누적)**: "실행기 cross-bench 전이 VALIDATED·TSR≤1.0"은 *답-가시 손작성 ABox* 위 결과 → **§1 공격에 더 노출**(P1=결정론+test-visible authored). P1 정직섹션에 §13.7 명기. **P1 방어는 blind-induce가 TSR을 유지할 때만 성립** ⇒ auto-induce는 "P2 다음"이 아니라 **P1 앵커 유효성의 선결 게이트**.

---

## §14. ★통합 온톨로지 스키마 설계 (Level 0+1+2) — by-construction 표현가능성 (DRAFT 2026-06-06, 웹조사 `w94ywlthc` 전·리뷰 대기)
> 목표 = **하나의 온톨로지 스키마 + 하나의 결정론 실행기**가 우리 레퍼런스 벤치들의 절차를 표현 → 모델은 NL→이 스키마만 학습, ABox-swap으로 cross-bench 전이. §13 적합성 사다리(L0 코어 / L1 decision-only / L2 멀티턴 / L3 루프·soft=out-of-scope)의 **L0+1+2를 구체 스키마로 박제.** ★설계 원칙 = **결정론·유한·감사가능 유지**(L3 루프·soft-judge는 의도적 비범위 = 그게 우리 assurance 기여의 정의상 경계, §13 판정).

### §14.1 스키마 명세 (typed nodes + relations + executor)
**NODE (typed)** — 4 타입:
| type | 의미 | L |
|---|---|---|
| `tool` | 도구/API 호출 | 0 |
| `verifier` | getter/check, 결과를 state에 bool/value로 produce | 0 |
| `decision` | 선언적 조건트리로 분류/route (도구 없음, tool-free) | 1 |
| `compute` | **bounded·audited pure-fn 호출**(parse_json/extract/normalize/arith/compare)로 raw slot→clean state 정규화 | 0 |
| `request` | user-facing 명료화 질문 emit·답을 state로 ingest | 2 |
| `subgraph` | 다른 ontology로 확장(계층 sub-workflow) | 2 |

> **★`compute` 노드 = gap-1 탈출구 (§14.5 리뷰 반영, blind-probe 확정)**: 선언적 `guard`("slot OP value")만으론 customer_service `is_authenticated`(auth_history JSON 파싱)·`metrics_improved`(service_metrics dict 산술)을 표현 **불가**(리프가 코드) → §14.4 손-검증 답이 **부분 NO(확정)**. **해법 = `compute` 노드 = BOUNDED·AUDITED pure-function 라이브러리 invocation**(임의 코드 아님; pure·total·side-effect-free·SOP-도출, parse/extract/normalize/arith/compare의 파라미터화 primitive). 모델은 *코드*가 아니라 `{primitive-id, args}` emit(학습 타깃 구조적 유지). ⇒ **assurance 재-scope = "선언적 구조(그래프) + *감사된 순수함수 라이브러리*"**("코드 0" 아님; arbitrary 코드 아님이라 결정론·감사가능 유지). bounded-primitive로 합성 불가한 결정 = 그 도메인 부분-비범위.

**NODE fields**:
- `realizes`: tool/function 명 (tool/verifier) 또는 decision-id.
- `arg`: `{param → source}`, source ∈ {`user_input`, `state.<key>`, `node.<id>.output`, `const`}. (= arg-binding, ARGFIX 일반화)
- `precondition`: **불리언 트리** `{op: AND|OR|CHAIN, children:[leaf|subtree]}`, leaf = verifier-ref 또는 condition. (=SOPBench AND/OR/CHAIN 일반화)
- `guard`/`condition`: 선언적 비교식 `{lhs:<state-ref>, op:∈{<=,>=,==,!=,in,exists}, rhs:<value|state-ref>}`. (=결정조건 *선언적* 인코딩, probe-2의 임계값을 코드 아닌 데이터로)
- `produces`: `{state-keys 설정, 충족하는 verifier-술어}`.
- `next`: 후속. linear=`goto`; decision=`branch:[{when:<condition>, goto:<node>}]` (조건분기).
- `terminate`: bool.

**GRAPH-level**: `start`(진입), `state_schema`(L2 지속 슬롯), `output`(최종 출력 스키마=RSD/final_output).

**EXECUTOR semantics (결정론)**:
- **L0/1 (single-pass)**: start서 traverse, gathered state로 precondition 평가, 의존순서로 fire, decision서 branch, terminate까지 → 도구호출 trajectory + output. (=현 DGGATE/CallGraphExecutor 통합)
- **L2 (turn-loop)**: state를 턴 넘어 유지; 매 턴 precondition 충족된 노드 fire(verifier/decision), 필요 input 없으면 `request`(명료화) emit, user 답 ingest, terminate까지 반복. **★그래프는 정적·acyclic; 반복은 *실행기*의 turn-loop**(그래프 cycle 아님) → **결정론·유한 유지**(이게 L3 루프와의 결정적 차이).

### §14.2 by-construction 표현가능성 (벤치별 매핑 = 증명)
| 벤치 | 스키마 매핑 | L |
|---|---|---|
| 벤치 | 스키마 매핑 | L | 적합 (FULL/PARTIAL) |
|---|---|---|---|
| **SOPBench** | goal=`tool`, `precondition`=AND/OR/CHAIN(login∧balance∨admin), verifier=login_user/getters, `arg`=user_known | 0 | **FULL** |
| **TaskBench** | 순수 `tool` 의존 DAG, `arg` node→node, precondition/guard 거의 없음 | 0 | **FULL** |
| **SOP-Bench** cust_svc | `next` 체인(5.1→5.7) + `decision`/`branch` + **`compute`**(is_auth JSON파싱·metrics dict산술) + guard | 0 | **FULL*** (compute 노드 필수) |
| **TOD-ProcBench** | condition-action=`precondition`→action, per-turn=turn-loop | 0+2 | **FULL*** (멀티턴 자유분기 일부) |
| **SAGE** | SOP 그래프 직접 + 멀티턴 적대=turn-loop | 0+2 | **PARTIAL** (자유분기 대화) |
| **FlowBench** | 워크플로 그래프 + 계층 `subgraph` + 멀티턴 | 0+2 | **PARTIAL** (자유분기·계층) |
| **SOP-Maze** | `decision`+`branch` tool-free, **+`compute`**(산술·HRS 깊은추론) | 1 | **PARTIAL** (compute-heavy 추론) |
| **τ/τ²-bench** | 정책=precondition+`tool`+`request` | 0+2 | **PARTIAL/aspirational** (동적 user-sim·자유발화·협상 ≠ 정적+scripted) |

⇒ **정직화(§14.5 gap-3): "8/9 표현"은 과장.** 정확히는 **FULL 4**(SOPBench·TaskBench + compute 포함 SOP-Bench·TOD) + **PARTIAL 4**(SAGE·FlowBench·SOP-Maze·τ — compute 또는 동적-대화 탈출구 필요). 공통 = **구조 *골격*(node/precondition/next)은 매핑되나, (a) 실도메인 결정=`compute` 필수 (b) 자유-분기 멀티턴·동적 user-sim은 정적그래프+turn-loop로 부분만**. 핵심 불변 = **그래프는 정적 acyclic + compute=감사된 순수함수** → 결정론·감사가능 유지.

### §14.3 결정론 경계 (의도적 비범위 = 기여의 정의상 한계)
- **L3 그래프-cycle/루프** (LogicIFEval iteration): DAG에 cycle 넣으면 유한·감사가능 깨짐 → 비범위. (단 turn-loop=정적그래프 반복적용은 OK; *그래프 내* 루프만 배제.)
- **soft/LLM-judge 자유생성** (ODCV): 실행기는 결정론 구조 출력만; 주관채점 자유텍스트는 표현 불가 → 비범위.
- ★이 둘을 *못 담는 게 결함이 아니라* 결정론-감사가능 구조의 정의상 경계(§13 판정). 담으려 확장하면 "임의 프로그램"화 → §1·assurance 논거 자멸.
- **★경계 정밀화 (§14.5 gap-1)**: `compute` 노드(bounded·audited **pure** fn)는 **경계 *안*** — pure·total·side-effect-free·유한 라이브러리라 여전히 결정론·감사가능. 경계 *밖* = ① 그래프-cycle(무한반복) ② arbitrary/Turing-complete 코드 ③ soft-judge. ⇒ assurance 정확표현 = **"선언적 구조 + 감사된 순수함수 라이브러리"**(코드-0 아님, but arbitrary-코드도 아님).

### §14.4 학습 타깃으로서의 함의 + 검증 필요 (§14.5 리뷰 반영)
- **결정로직=선언적 `guard`+`compute`**: guard("slot OP value")는 operator-shape만 구조화; **실도메인 결정 리프는 `compute`(감사된 pure-fn) 필수**(customer_service is_auth/metrics = JSON파싱·dict산술 → guard 단독 표현불가, **손-검증 답=부분 NO 확정, §14.5 gap-1**). ⇒ 손-검증 재정의 = **"`compute` 노드 포함 시 customer_service 완전표현되나".** 학습 타깃 = NL→(구조 + guard + **`{primitive-id,args}`**), 코드생성 아님(구조 유지).
- **★slot-grounding이 진짜 난점 (§14.5 gap-4), operator 아님**: guard `lhs:<state-ref>`가 가리키는 slot 어휘(auth_history/service_metrics 키 + JSON 내부경로)=**도구반환 스키마 내부 = blind-미지**(probe가 막힌 지점). 선언성은 operator *모양*만 완화; **NL→*grounded* slot refs**(NL 언급을 올바른 도구-스키마 필드에 정렬)가 blind-induce의 핵심 미해결. = compute primitive args도 동일 grounding 의존.
- **단일 실행기 = 미구축, *신축* 필요 (§14.5 gap-2)**: CallGraphExecutor(`workflow_executor.py`, 단일 "slot OP value"·**AND/OR 트리 없음**) vs DGGATE(full AND/OR/CHAIN)는 **술어언어 비호환** → 합집합(+compute+turn-loop) 지원 **새 실행기 신축** = 목표지 완료 아님. 벤치별 evaluator(dirgraph-oracle/input→output/per-turn/DB상태) 통과는 그 위에서 **벤치별 1-태스크 손-검증**.
- **표현가능 ≠ 학습가능 ≠ blind-induce가능**: §14는 *표현*만 증명; NL서 생성은 frontier-1-call→소형 사다리(§13.5)로 별도 판정. (slot-grounding이 그 사다리의 진짜 bar.)
- **open**: ① `compute` primitive 라이브러리가 전 벤치 결정 합성 충분한가(parse·arith·set·temporal 범위) ② `subgraph` 재귀깊이 ③ `request`/동적 user-sim이 τ² dual-control 오염 회피하나(τ=PARTIAL/aspirational) ④ 웹조사 `w94ywlthc`로 레퍼런스 밖 벤치 사다리 분류 보강 후 스키마 확정.

> **상태 = DRAFT (리뷰5/§14.5 반영).** 다음 = **(a) compute 경계 포함 스키마 확정** → (b) customer_service "compute 노드 포함 완전표현" 손-검증 → (c) (확장)실행기 1태스크 신축·벤치별 evaluator 통과 → (d) 웹조사 통합 → 타세션 리뷰 → 방향 재결정.

### §14.5 ★리뷰 (2026-06-06, 리뷰5 — 실 executor·customer_service 대조) — 강점 유지, 4 load-bearing 빈틈
> **리뷰5의 4 빈틈은 본문 §14.1–14.4에 반영 완료**(compute 노드·assurance 재scope·FULL/PARTIAL·slot-grounding·새 실행기). 아래는 audit 기록.
강점(유지): §14.3 결정론 경계(falsifiable·non-example 명시) · turn-loop≠graph-cycle · "표현≠학습≠blind-induce" caveat.

1. **★(최중요) wrapped-compute 탈출구 = 핵심 긴장 (open-①서 승격).** `decision`+선언적 `guard`("slot OP value")는 단순비교만 표현. **customer_service가 이미 반례**(내 §13 blind-probe 분석): `is_authenticated`=auth_history **JSON 파싱+로직**·`metrics_improved`=service_metrics **dict 산술** → guard 표현불가·**wrapped 코드 필수**; final_status 트리는 branch+guard지만 *리프가 코드*. email_intent `_classify`도 코드. ⇒ 실도메인은 wrapped pure-fn 필수 = §14.3이 금한 "임의 프로그램"이 *wrapped fn으로 새어듦*. **처방: 스키마에 `compute` 노드(pure·total·side-effect-free·SOP-도출·audited) 명시 + assurance를 "선언적 구조 + 감사된 순수함수"로 scope("코드 0" 아님).** §14.4 손-검증("customer_service 순수 branch+guard 표현?")의 답 = **부분 NO(확정).**
2. **통합 실행기 미구축·두 술어언어 비호환.** CallGraphExecutor(workflow_executor.py)=단일 "slot OP value"·AND/OR 트리 **없음** vs DGGATE=full AND/OR/CHAIN 재구성. §14.1 "통합"은 합집합 지원 **새 실행기 신축 필요 = 미구축**(목표지 완료 아님) — 명기.
3. **§14.2 매핑 FULL/PARTIAL 구분, "8/9" 과장.** SOP-Maze(산술·HRS 깊은추론=compute 필요)=PARTIAL · τ/τ²(동적 user-sim·자유발화·협상 ≠ 정적+scripted request)=PARTIAL/aspirational · SAGE/TOD/FlowBench 멀티턴 자유분기=부분. ⇒ "구조 골격 매핑·compute/동적대화는 탈출구 필요"로 정직화.
4. **guard-선언성은 operator-shape만 완화, slot-grounding은 아님.** guard `lhs:<state-ref>`가 가리키는 slot 어휘(auth_history/service_metrics 키)=도구반환 JSON 내부=**blind-미지**(내 probe가 막힌 지점). 학습 타깃=NL→(구조+*grounded* slot refs); **slot-grounding이 진짜 난점**(operator 모양 아님) — §14.4 명기.

**리뷰5 결론**: §14는 표현가능성 *골격*으론 옳고 정직(경계 명시). 단 **(a) bounded-compute 노드 없이는 실도메인 표현불가(customer_service 확정)** → 스키마에 compute 경계 추가가 스키마 확정의 *선결*. (b) 통합 실행기는 미구축. (c) 매핑은 FULL/PARTIAL 혼합. ⇒ 다음 = §14.4 손-검증을 **"compute 노드 포함 시" customer_service 완전표현 + 단일(확장)실행기 1태스크**로 재정의 후 진행.

### §14.6 ★기업-현실 reframing (사용자, 2026-06-06) — Pushback 1 해소 + 기여 scope 확정
사용자 통찰: 실 기업은 **(i) SOP 판단용 compute 함수·(ii) slot-grounding 데이터를 *사전 authoring***하고 **(iii) 도구는 새로 안 만든다**(기존 API 고정 = 100% 신뢰 도구호출의 전제).
- **⇒ §14.5 Pushback 1 *해소***: compute(`is_authenticated` 파싱·`metrics_improved` 산술)는 **결정론 레이어 사전자산(OISA ②prong)**이지 LLM 즉석생성 아님 → assurance 유지(선언적 구조 + *사전감사된* 순수함수). **slot-grounding도 사전 스키마/데이터로 해결**(내 blind-probe 미지 = 일부러 가린 테스트일 뿐; 실세계선 도구 출력스키마 제공/관찰로 routine).
- **★확정 기여 scope** = "**고정 도구 + 사전 결정론 compute/verifier 위에서, 작은 모델이 *어떤 도구를 언제 부르고 무엇을 판단·분기·계획하나*를 특정분야 frontier(Opus급)급으로**." = LLM 역할 = **planner/judge(orchestration)**, NOT compute-generator, NOT tool-inventor.
- **⇒ §14 스키마 재분류**: `compute`/`verifier`/`tool` 노드 = **사전자산(학습 타깃 아님)**; **학습 타깃 = orchestration = {도구선택·순서·분기-판단·precondition-gather·decision-routing}** = 우리 기존 *alive* 축(gather/decision)과 정합. ⇒ §14.5 (a) "compute 학습" 부담 소멸; (b)통합 실행기·(c)FULL/PARTIAL은 유효.
- **남는 연구질문(명확)** = "고정 도구·사전 compute 위 orchestration/judgment를 작은 모델서 frontier급으로" → 방법론 동향 deep-research(`woxbvzk8t`, distillation·RL/RLVR·verifier-offload·cascade·specialization·test-time) 진행 중 → 결과로 **§15 방법선택** 보강.

---

## §15. ★연구 추세 위치 + 실무 의미 비판 — 내구적 moat는 assurance 하나뿐 (2026-06-06, 딥리서치 `wheyskq29` 진행 중)
> 동기 = "이 방향이 실무적으로 의미 있나"의 *냉정한* 답 + 최대 위협(reasoning-model 조류)에 대한 방어를 문서로 굳힘. ⚠️**일부는 strategic opinion**(인용 명시한 것 외); 진행 중 `wheyskq29`가 reasoning-model-흡수·compliance-determinism을 인용으로 hardening → 완료 시 ✅/정정.

### §15.1 neurosymbolic LLM 추세 (우리 위치)
1. **LLM-as-formalizer + sound solver = 지배 패러다임** (Logic-LM·SatLM·PAL·LLM+P·L2P; ACL'25 survey `2503.18971`). **→ 우리 방향이 이 계열.**
2. **LLM-Modulo / Generate-Test-Critique** (`2402.01817`): soundness는 외부 critic서. compound-AI 추세.
3. **★Reasoning model 부상 = 최대 *역류*** (o1/o3/R1): plan·제약준수·self-verify를 *weight 안*서 개선. Kambhampati 본인 o1=PlanBench "quantum improvement" (`2409.13373`). = verifier-RL이 neurosymbolic 이득을 *가중치로 흡수*하는 경쟁 패러다임.
4. workflow/structure-guided agents (FlowBench·TaskBench) — 엔터프라이즈 reliability.
5. autoformalization (Lean/Isabelle).
6. **learned/distilled formalization = 덜 붐비는 틈** (proScript·PlaSma) → **우리(학습 소형 NL→구조 전이)가 여기.**

### §15.2 실무 의미 — 4 위협 (정직)
- **①★Reasoning-model 조류 (bitter-lesson, 최대 위협)**: SOPBench 이미 frontier 30–76%. o-class가 SOP-following을 네이티브 신뢰성화하면 "온톨로지+결정론 실행기" 한계가치 ↓. **손-설계 스키마가 스케일링이 1–2년 내 obsolete시킬 과도기 버팀목일 위험.** (= bitter lesson; **반박 소싱 완료 `BITTER_LESSON_REBUTTAL_SOURCES.md` 2026-06-10**: ①범주 오류 — Sutton 원문의 금지 대상은 해법-지식 주입('how we think we think')이지 명세-집행 아님; 게이트=search-계열 meta-method(원문이 권장) + 명세는 발견 대상이 아님 ②보장축 직교 — hallucination 불가피성 형식증명(2401.11817)·OpenAI 자체 진단(2509.04664)·compositional collapse(2305.18654/2410.05229) = precision=1은 scale로 구매 불가 ③frontier 자체가 hybrid(AlphaGeometry Nature'24, RLVR의 V=결정론 verifier) ④Sutton 본인이 'LLM은 bitter-lesson-pilled 아님' 진술(2025). **잔존 유효 부분 = capability 헤드라인 침식(§15.3 양보 유지)·front-end 학습 leg 성공 조건부(동 문서 §7-3)**.)
- **②온톨로지-설계 병목**: 통합 온톨로지는 compute 노드·새 실행기·slot-grounding 필요(§14.5) → 실상 *per-domain DSL*을 설계·유지. **"per-domain 코드 0" 약속이 후퇴(§13.7 probe-2)** → 실무선 결정론 프로그램(OISA) 직접 or frontier+경량 checker가 더 쌀 수.
- **③"free-text SOP→구조" 킬러 유스케이스가 좁다**: 고위험 절차는 *애초에* free prose로 안 둠(BPMN·결정테이블·애널리스트 1회 구조화) → "애널리스트 amortize"와 경쟁.
- **④전이의 실무가치**: 재학습0 cross-bench는 학술적; 실무 배포는 보통 한 도메인·fine-tune 기꺼이. 전이는 롱테일·다도메인서만 결정적.

### §15.3 ★내구적 moat = assurance/compliance 하나 (헤드라인 재좌표)
위 위협(특히 ①)을 견디는 단 하나 = **결정론·감사가능 정책-집행**:
> ⚠️**§15.4에서 부분 격하됨(필독)**: 딥리서치 결과 "외부 *집행* 필요"는 확정되나 "*결정론-특정*이 답"·"규제가 결정론 요구"는 **미확정**(규제 1차원문 0·Claude-Opus-4.5가 정렬만으로 1.3%). 아래 강한 표현은 §15.4 후속(규제 sourcing) 통과 전엔 *주장*이지 *확정* 아님.
> **규제 산업(은행·의료)은 stochastic 모델 자기-집행을 준수근거로 *못 씀* — 모델이 아무리 좋아져도.** 더 나은 LLM도 확률적; compliance는 *결정론·감사가능·추적가능*을 요구(=capability와 *직교*). reasoning-model이 *해결 못 하는* 축. (⚠️규제근거 미sourcing — §15.4 open-1.)

⇒ **판정**:
- **정확도·비용·전이 헤드라인 = reasoning-model+스케일링에 *침식* = 시한부.** 이걸로 frontier와 싸우면 진다.
- **assurance 헤드라인 = 내구적 moat** = capability와 직교라 스케일링이 안 풀어줌. **"더 똑똑한 LLM이 아니라, *증명가능하게 정책을 안 어기는* 집행."**
- ⇒ **프레이밍 재좌표**: neurosymbolic의 *soundness*(축2) 측면으로 걸어야지 *capability*(축3에 짐) 측면으로 걸면 시한부. §9 assurance축(4번째 지표)을 **1급 헤드라인**으로 승격, 정확도는 trade-off로.
- **정직 scope**: 학습-소형 NL→결정론-온톨로지 컴파일러가 실무 의미 있는 regime = **롱테일 다도메인 × free-text SOP × 규제/감사 × 비용**의 교집합. 그 밖(소수도메인·기구조화·비규제)에선 frontier+checker나 직접 결정론이 우위 — 정직 인정.

> **상태 = §15.4로 확정(딥리서치 `w0yix88gp` 완료 2026-06-10).** ⚠️**§15.3 assurance-moat는 §15.4에서 *부분 격하*됨(motivated-reasoning 적발) — 아래 참조.**

### §15.4 ★딥리서치 확정 (`w0yix88gp`, 24/25 주장 3-vote, 2026-06-10) — 위협 OVERSTATED, 단 moat는 *정직 격하*
> 지시 = "내 선호결론(assurance moat 내구)에 *적대적으로*, 증거 없으면 그렇게 말하라". 결과가 정확히 그 일을 함.

**판정① 위협("reasoning-model이 symbolic 폐기") = OVERSTATED(soundness)/DOMAIN-DEPENDENT(planning)**:
- o1: plain Blocksworld 97.8% but **obfuscated 52.8%·20–40스텝 23.6%**; Fast Downward(symbolic)=100% 불변. capability 상승 실재하나 robust·long-horizon·domain-general 아님(`2409.13373`).
- **correctness 보장 0** — soundness는 *외부 verifier*(LRM-Modulo)서, 모델 내재 아님; o1=cost/time/guarantee/perf 트레이드의 한 점 = **determinism은 capability와 직교**(`2410.02162`).
- reasoning-model이 **제약을 환각**(프롬프트에 없는 graph edge; o1-mini/o3-mini/R1/Claude-3.7/Gemini-2.5/Grok-3 전반, false-error의 67–94%) → 제약 fabricate하는 모델은 NL SOP 의존구조 충실 내재화 불가 = **결정론 외부표현 논거**(`2505.12151`).
- LLM-Modulo thesis가 reasoning-model 시대에도 지속 = 우리 패러다임(`2402.01817`).
- ★**우리가 *직접 인용*해 선제방어할 반대증거(claim 12/13, `2412.09879`)**: formalizer 우위는 **base capability 조건부**·**top reasoning-model(o3-mini/R1)에선 부분 erosion**(직접 planner가 충분히 강함). 약한 모델(≤405B)은 formalizer로도 **구제 불가**(solvable PDDL 0). ⚠️**단 simple PDDL 한정**; messy-NL-SOP 집행으로 erosion 확장되나는 **미검증(open)** = 우리 scope 직결.

**판정② assurance-moat = ★정직 격하 (UNDERDETERMINED)**:
- ✅**강하게 지지되는 것 = "외부 *집행* 필요"**(capability≠compliance): ODCV-Bench 12 frontier 중 **9/12가 압력 하 30–50% 위반**, *최강* Gemini-3-Pro가 *최고* 71.4% 위반("superior reasoning≠safety"); **deliberative misalignment**(자기 행동을 비윤리로 *알면서* 실행, self-aware 72–93%); Anthropic Agentic Misalignment 보강(금지명시도 blackmail 96→37%만)(`2512.20798`·`2510.05179`).
- ⚠️**지지 안 되는 것(motivated-reasoning 적발) = "*deterministic/symbolic* 집행이 *특별히* 답"**: 위 증거는 "집행 필요"는 강하나 "결정론이 *유일*"은 아님. **Claude-Opus-4.5가 내부정렬만으로 1.3% 위반** = 회의론자가 "더 나은 *학습*이면 충분, 외부 scaffold 불요"라 칠 카드. **+ 검증셋에 1차 규제문헌 0**(EU AI Act Art.12/14/Annex IV·SR 11-7·의료규제 미포함) → **§15.3의 "규제가 결정론 요구" leg는 현재 *미지원* = 별도 규제원문 sourcing 필수.**

**판정③ 정직 scope**: 방어가능 = (a) **soundness/감사가능이 정확도와 무관하게 그 자체로 요구되는 곳**(진짜 직교 논거) (b) base가 formalize 가능한 capability tier(frontier 콜비용 회피+보장). **약한 base 불가**(≤405B formalize 전멸)·top-reasoning×simple-PDDL서 margin 좁아짐.

**§16 보강 (claim 23, `2605.05226`·`2504.13837`)**: 내재화/RL은 **base capability에 bounded**(약하면 구제 불가; Limit-of-RLVR=base가 안 뽑는 건 RL이 못 풂). 양날 — absorption 위협 약화 *and* **우리 소형모델도 capable-enough base 필요**(§16.3 커버리지 확증). "outcome-only RL이 verification 내재화"는 **REFUTED(1-2)** → 외부검증 유지 지지.

**⇒ 필수 후속(open, 박제)**:
1. **★규제 1차원문 sourcing**: ✅**CLOSED (2026-06-10)** — `REGULATORY_DETERMINISM_SOURCING.md`(권위). **판정=(c) "로깅+검증+감독이면 충족"** (EU AI Act 전문 grep determin/reproduc/repeatab=0 [OJ authentic 3-vote]; SR 11-7=통계모델 관리 체제) → **결정론-leg 철회 확정, "검증가능성" leg로 후퇴**(=리뷰6-1 좌표, 추가 붕괴 없음). 잔존 textual footholds: ①**SR 26-2(2026-04-17, SR 11-7 대체)가 결정론 rule-based를 '모델' 정의서 명시 제외=MRM 부담 면제 + genAI/agentic=scope 밖(준수경로 미정착)** ②EU MDR Annex I §17.1 "repeatability"(유일 명시, 의료 한정) ③de-facto (b)(충족비용 비대칭). 금지문구: "regulations require deterministic logic".
2. **bitter-lesson 반박**: ✅**CLOSED (2026-06-10)** — `BITTER_LESSON_REBUTTAL_SOURCES.md`(방어 5라인 A–E + 정직 양보 §7, 인용 전부 1차원문 fetch 검증). 검증셋 3-vote 정식 등재는 후속 옵션(동 문서 §8 우선순위).
3. **agentic-SOP erosion 테스트**: claim 12/13 erosion이 simple-PDDL 넘어 messy-NL-SOP(SOPBench/AgentOrca/AgentSandbox `2505.24019`)로 확장되나 = scope 결정.

**⇒ §15.3 수정**: "assurance=내구 moat"를 **"외부 *집행* 필요는 확정(ODCV/deliberative-misalignment), *결정론-특정* + 규제근거는 미확정 → 규제원문+bitter-lesson 반박이 선결"**로 격하. §9 assurance 승격은 *집행-필요*까지만 정당, *결정론-특정*은 후속 sourcing 후.

**★주권-leg 신설 (2026-06-10, 시장 실사 — `GROUNDED_BIZ_AGENT_BENCH_DESIGN.md` §1.7 권위, 사용자 승인으로 추가)**: 패키지 {소형·저비용}의 "저비용" 논거에 **독립적인 "데이터주권" 논거 추가** — 둘 다 frontier-API-불가 논거이나 주권 쪽이 더 강함(비용은 trade-off, 주권은 *하드 제약*).
- **근거 3층**: ①2025년 LLM 지출 절반+가 on-prem(규제산업 주도; air-gapped 사례 LANL) ②미국 대형은행도 frontier 사용은 *신뢰경계 내 게이트웨이*(JPM LLM Suite 통제포털·Einstein Trust Layer zero-retention+PII placeholder 마스킹)로만 — 직원-보조+human supervision 한정 ③**★한국 금융권 망분리(=CDP 직접 환경)**: 금융위 「망분리 개선 로드맵」(2024-08) 체제서 인터넷망 상용 AI=혁신금융서비스 *예외 승인제*, **내부망="금융권 AI 플랫폼"으로 선정 오픈소스 모델 직접 설치가 공식 경로** ⇒ **내부망 소형 오픈웨이트 sLLM = 옵션이 아니라 사실상 유일 정규 경로**(frontier API 기본 불가).
- **thesis 좌표**: capability 스케일링이 풀 수 *없는* 또 하나의 직교축 — 모델이 아무리 좋아져도 *반출이 안 되는* 데이터는 내부망 모델만 만질 수 있고, 내부망에 들어가는 건 소형 오픈웨이트 — **{소형 front-end}×{결정론 게이트}×{재학습0 전이} 패키지가 규제 환경의 직접 요구사항**. ⚠️정직 한계: 망분리 완화가 진행 중(특례 확대 추세)이므로 "영구 하드제약" 주장 금지 — "현행 정규 경로" 수준으로. 벤치 반영=deployable-arm 층화(동 설계서 §1.7-d).
- ⚠️**정정 (2026-06-11, 사용자)**: **"내부망=소형(sLLM)" 전제 철회** — B200급 on-prem 도입으로 기업 내부망도 32B+ 오픈웨이트 구동 가능. 주권 제약의 본질 = **오픈웨이트 × on-prem**(크기 무관)이고, 크기는 경제 변수(서빙비·fleet 밀도·동시 에이전트 수)로 강등. ⇒ 패키지 표기 **{소형}→{오픈웨이트(크기 가변)} front-end × {결정론 게이트}**로 갱신; 소형-leg는 "유일 경로"가 아니라 **비용-효율 주장**으로만. 이 정정이 census→처방 절차(TASKBENCH_RESULTS §10.3)와 결합하면 오히려 강화됨: 내부망 메뉴에 7B~72B가 다 있으므로 "어느 크기 + 어느 레버"를 base census로 정량 결정하는 절차 자체가 배포 의사결정 도구가 됨(예: 32B=SFT 생략·guided+게이트만 / 7B=L5 학습 후 동일 스택 — in-domain 특화는 크기 무관 fine-tune 시 간섭 회귀로 회복장치 필요).

---

## §16. ★학습 설계 — outcome-supervision 디폴트 · distill=커버리지 · 계층=최적화보조 (Jia et al. ICML'25 `2502.10581` 원문분석 근거, 2026-06-06)
> NL→symbolic 학습을 *어떻게* 하나의 원칙. 근거 = **"Do We Need to Verify Step by Step?"**(Jia·Rakhlin·Xie, ICML'25, `2502.10581`) 원문 정독. = §13.2 Track B 레시피·계층 제안(사용자)·LOCK을 supervision 이론으로 정합.

### §16.1 Jia et al. 정리 (proven)
- **메인 정리**: outcome supervision으로 step-wise 보상 학습 오차 ≲ **H^{3/2}·√(C_sa·log|ℛ|/|𝒟_O|)** → **outcome은 process보다 통계적으로 더 어렵지 않다(horizon H의 *다항식*까지)**. 핵심 = **state-action 커버리지 C_sa 의존(trajectory 커버리지 C_traj 아님; 후자는 지수적)** → 통념("outcome=trajectory-커버리지로 지수적 손해") **반박**. (엔진=Change of Trajectory Measure Lemma.)
- ⚠️**3 caveat**: ① 순수 *통계*(표본복잡도) 결과 — process의 *최적화/탐색/credit-assignment* 이점은 **부정 안 함**(저자: 경험적 격차는 *있다면* **알고리즘적 한계**서). ② **커버리지 가정 조건부**(C_sa 큰=bound 공허; LLM서 성립여부 미다룸). ③ H^{3/2} 잔존 + pessimism 알고리즘 + 결정론보상 가정.

### §16.2 우리 학습 레시피 (위 정리 → 정합)
1. **per-층/중간 process 라벨 *수집 금지* (통계적 불필요, 증명됨).** 우리는 **outcome 보상(실행기 TSR / Guard-2 exact-match)이 이미 있음** → 이게 디폴트. 저자 권고("비싼 process 라벨 → 더 나은 알고리즘에")와 정합. ⇒ §13.2 Track B = **outcome-검증 음성신호(DPO/RFT)**, 중간추론 라벨 아님.
2. **계층(사용자 제안)은 *최적화/long-horizon 보조*로만 정당, 통계 우위 아님.** Jia가 비운 공간(최적화·탐색·커버리지·H-패널티)이 정확히 계층 효용 자리. ⇒ 계층 = **inference-time 분해 + *outcome* 보상**(per-층 GT 라벨 X). "더 sample-efficient"로 팔면 안 됨(반증됨).
3. **★distillation 재좌표 = *커버리지 제공*(추론모방 아님).** caveat②가 핵심: 약한 7B는 정답-구조 manifold **커버 못 함**(Track C fabrication=나쁜 커버리지) → outcome 보장 *안 켜짐*. **distill(frontier teacher)이 correct-structure 커버리지를 제공** → 보장 작동. ⇒ distill 목적 = **커버리지**(LOCK-위험인 "reasoning trace 모방" 아니라 *검증된 구조* distill). 
4. **통일 레시피**: **`distill로 correct-structure 커버리지 → outcome-RFT/RLVR(TSR/Guard-2 보상)로 최적화`**, 계층=최적화보조, process-라벨 금지. = Jia(2502.10581) + LOCK(Track C) + 우리 검증가능-타깃에 *모두* 정합.

### §16.3 검증·open
- **커버리지 진단 선결**: 우리 base 7B/distill 데이터가 correct-structure를 *커버*하나(C_sa 유한?)가 outcome-보장의 실제 조건 → distill 전후 구조-커버리지 측정 필요(Jia 결과는 *조건부*).
- **계층=최적화보조 가설은 *flat 실패모드 진단 후*** 결정(§15 ④, "엉킴/long-horizon이면 계층, slot-grounding이면 무효").
- ⚠️ Jia는 *통계*만; 우리 실 RFT 최적화/탐색 거동은 실측해야(이론이 "RFT가 잘 최적화한다"는 보장 아님).

> **메타교훈 (박제)**: "process가 직관적으로 좋아 보임"≠"통계적 우위"(Jia 반증). 강한 학습-설계 주장은 supervision 이론 원문 확인 후 박제. [[feedback-check-authority-before-rederive]]

---

## §17. ★벤치 전략 — (B) 열린벤치 substrate로 orchestration thesis 검증 (2026-06-10, repo 4종 정독 확정)
> 실문제(CDP 마케팅 오케스트레이션)↔학술벤치 간격 좁히기. **결정 = (B) 열린벤치(TaskBench/ToolRet)를 substrate로** — 이유: AD-Bench 데이터 *비공개*(privacy, substrate 불가) · CDP-substrate=POC*설계*물(authoring 비용+usage-DB 부재→후행) · **열린벤치=결정론·zero-authoring·즉시**. 우리 "검증은 학술벤치 먼저"와 정합.

### §17.1 TaskBench = 우리 thesis의 *최적* 학술 substrate (검증된 실제 포맷, raw 정독)
- **태스크**: `instruction`(NL) → `task_nodes`[{task:도구명, arguments}] + `task_links`[{source,target}] + `task_steps`. type∈{single,chain,dag}(복잡도).
- **도구그래프**(`graph_desc.json`): nodes(id·desc·input-type·output-type) + links(input/output 타입호환=의존). multimedia=40도구·568엣지. 3도메인(HuggingFace/Multimedia/DailyLife)·103도구·17,331샘플.
- **eval**(`evaluate.py`): **결정론**(scikit-learn `prfs`, LLM無) — **Node-F1**(도구명 exact set-F1)·**Edge-F1**((source,target) tuple exact)·**t-F1/v-F1**(param name/value). **standalone 재사용**(스키마만 맞추면 우리 gold/pred에 적용). **Apache-2.0**.

### §17.2 ~~★TaskBench > SOPBench~~ → 예측축 vs 실행축 (★">" 철회, §17.7-1·§17.8)
> ⚠️**">" 철회(리뷰)**: TaskBench = NL→도구그래프 *예측*(오프라인 매칭, **도구 미실행**) = *gather/구조-예측* 축. SOPBench/SOP-Bench = 제약·거부·outcome *실행* 축. **conflate였음.** ⇒ **TaskBench=보완(LLM-orchestration *예측*), SOPBench=대체불가(*실행*+outcome+전이). 둘 다 필요·멀티벤치.** (단 §15.4: "assurance moat"는 *별개로 미확정*이라 SOPBench도 과대elevate 금지.) §14.6서 *실행-assurance=게이트의 일*이지 LLM의 일 아님 → TaskBench는 *LLM의 실제 일(예측)* 을 테스트.
- TaskBench = **NL→도구-의존그래프 *구성*·결정론 graph-match**(node/edge-F1 published 표준). 구조적으로 CDP 오케스트레이션에 가까움(⚠️*구조 proxy*지 마케팅 *도메인* 아님, §17.7-4). = orchestration-*예측* thesis 1차 검증 substrate(실행축은 SOPBench).

### §17.3 (B) 실험 설계 (TaskBench, §16 학습레시피 정합)
- **헤드라인 = LODO cross-domain 전이**: 2도메인 학습 → **held-out 1도메인** 평가 = thesis 핵심(소형 NL→구조가 *전이*).
- **지표(결정론)**: Node-F1(도구선택)·Edge-F1(의존/순서)·v-F1(param-binding). type별 층화(single<chain<dag=horizon).
- **arms**: ①base-small(prompted) ②**우리 learned-small**(§16: *distill로 correct-graph 커버리지* → *outcome-RFT, 보상=node/edge-F1*) ③frontier baseline(published n-F1/e-F1).
- **판정**: learned-small의 held-out F1이 base ≫ 이고 frontier 근접·전이 유지하나. = "소형이 NL→오케스트레이션-구조를 결정론-충실하게, 전이되게 학습."
- 인프라: 협업자 H200 stack/adapter-only eval 드라이버를 TaskBench로도 포인팅(SOPBench과 병렬 substrate).

### §17.4 ToolRet = retrieval-at-scale 레이어 (별도·보완)
TaskBench 도구수 적음(103) → **ToolRet**(43K corpus·7.6K태스크·**nDCG@10/Recall@K 결정론**·Apache-2.0)로 **도구선택-at-scale** 축 보완. query→labeled relevant tools. best 33.83=난도충분. (CDP 수천도구 차원의 학술 proxy.)

### §17.5 positioning + 조합 (frankenbench 회피)
- **AD-Bench**(데이터 비공개): 최근접 마케팅 선례 *인용* + 선택적 *leaderboard 제출*(real-domain black-box 숫자). substrate 아님.
- **EvolveTool-Bench**(MIT): composite/reuse/redundancy 결정론 메트릭 — composite-reuse 차원 추가 시 차용(소스 1단계 확인 남음).
- **CDP**: *실타깃 도메인*. POC설계물 substrate(PRAXIS)는 *후행*(authoring 의지 있을 때). usage-랭킹은 실데이터 부재로 defer.
- **조합 = 데이터 합성 0, 열린 결정론 프로토콜 차용·인용** → frankenbench 아님.

### §17.6 즉시 단계
1. **TaskBench remote clone + eval 검증** — 데이터 로드·`evaluate.py`가 sample pred에 동작·라이선스. (zero-GPU)
2. base-small baseline(prompted) node/edge-F1 측정 → headroom 확인.
3. §16 학습(distill-coverage→outcome-RFT) → LODO 전이 측정. (협업자 H200)

### §17.7 ★리뷰 (2026-06-10, 리뷰 — TaskBench 원문확인) — 방향 옳음, 5 빈틈
원문확인: GT=back-instruct(LLM)**+human-verified**(순수합성 아님, 우려완화) · eval=**오프라인 그래프-예측(TaskEval 3단계), 도구 실행 없음**(확정).
1. **★TaskBench는 도구 *미실행* → assurance/execution 축 검증불가.** §17.2 ">"는 conflate: TaskBench=NL→그래프 *예측*(gather/구조 축) vs SOPBench/SOP-Bench=제약·거부·outcome *실행*(§14 실행기·§15 assurance moat). 우리 scope(§14.6)="신뢰·감사가능 도구 *실행*"=TaskBench 못 봄. ⇒ **">" 철회; TaskBench=보완(예측), SOPBench/SOP-Bench=대체불가(실행+전이+assurance). 진행중 Exp-5를 대체 금지. 최강=멀티벤치(예측⊕실행).**
2. **그래프-예측 *분해 모호성*(복수 정답)**: node/edge-F1=단일GT exact-match → valid 대안 오답처리 → F1 천장↓·"frontier급" 흐림. 선결=메트릭이 대안 credit하나 확인 + miss 진짜오류 vs 대안 분해.
3. **3도메인 LODO=n=3 저검정력 + 도구명 암기위험** → **도구명 alias-마스킹**(SOPBench 교훈) + type(single/chain/dag) 층화로 포인트 확보.
4. **"CDP 더 가까움"=구조 proxy지 도메인 아님**(HF/멀티미디어/일상≠마케팅). 도메인-관련성 과대주장 금지.
5. **frontier 천장-아님 확인**(rigged 체크): GT back-instruct 기원→GPT-4가 ~1.0이면 "근접" trivial. §17.6서 frontier n/e-F1 실확인(기억상 ~0.7-0.8=OK일 듯).
- (부차) arm② 보상=node/edge-F1=깨끗한 RLVR(§16 정합)이나 GT 모호성 상속→exact-match 보상이 GT 특이성 overfit 가능.
- **강점(유지)**: 열린·결정론·Apache-2.0·zero-authoring·human-verified GT·frankenbench 회피. §17.6-1=옳은 첫 falsifiable.
- **메타**: §14–17 설계 누적·새 학습신호 0(Exp-5만) → §17.6-1 빨리 해 falsify 먼저. **TaskBench=SOPBench 실행축 *보완*이지 대체 아님 명기.**

> **상태 = (B) 확정·설계 완료, 리뷰5빈틈 반영(§17.7).** 다음 = §17.6-1 (TaskBench clone+검증, 특히 메트릭 대안-credit·frontier 점수·실행없음 명기).

### §17.8 ★실측 결과 (2026-06-10, 리모트 `JARVIS_tb`·`tbeval_venv`) — §17.6-1 DONE + P2/P5 닫음
- **✅§17.6-1 DONE (eval 파이프라인 검증)**: TaskBench clone(3도메인 ~17K)·`tbeval_venv`(sklearn·datasets2.14.5·pyarrow12·rouge). **gold-as-pred → Node-F1=1.0**(결정론 prfs, eval에 LLM無 확인). ★**재현 gotcha**: 원본 `data.json`=`tool_nodes`/`sampled_links` → evaluate.py는 `task_nodes`/`task_links` 기대 → **필드변환 전처리 필수**; 메트릭명 = `-m f1`(node), `-m link`, `-m argument`(`-m node`은 무효); pred 경로=`{data_dir}/{prediction_dir}/{llm}.json`.
- **✅P5 닫음 (frontier 비-포화, rigged 아님)**: published 리더보드 — **gpt-4 n-F1=90.9·e-F1=69.3**; claude-2 80.9/53.0; gpt-3.5 72.8/44.0; **소형 base codellama-7b 53.3/14.8·vicuna-7b 46.1/4.3**. ⇒ **frontier≠천장**(특히 *edge-F1* 큰 headroom), 타깃="frontier-achievable(~91/69)" not 100. 소형→frontier 격차 실재(edge가 discriminating).
- **✅P2 확정 (협업자 옳음, 내 보정 *철회*)**: `sim(a,b)=1 if a==b else 0` = **순수 exact-match**. matching-mode(Hungarian)도 *alignment*만 다르고 **대안 credit 안 함**(probe: 50% random-교체 → no_match 0.83/match 0.54 둘 다 깎임). ⇒ **valid 대안 decomposition은 penalize됨**, 달성가능 천장<100(gpt-4 90.9가 반영). F1-miss를 real-error vs 대안으로 분해해야(단 *실행 없어* 완전 분해 불가=P1과 연결). **내 "matching이 대안 credit" 보정은 틀림(정정).**
- **⚠️신규 caveat (GT 품질 도메인편차)**: human-verified 비율 = Multimedia **62.7%** vs HuggingFace **10.8%**(critic-only). ⇒ "human-verified GT 완화"는 *도메인 의존*; LODO서 HF는 약한 GT. (P3 검정력+이 편차 → 도메인 가중 주의.)
- **⇒ 다음 (설계 그만·측정 시작, 메타-비판 해소; 리뷰3 정정 반영)**:
  1. **(A-0, zero-GPU, RFT 전 BLOCKING — 메타규칙 "GPU 전 zero-cost 진단")**: 7B edge-miss **~30개 수동 감사** → real-error vs valid-대안-분해 분율 추정. P2가 확정했듯 sim()=exact-match라 valid 대안도 penalize → **edge 20pt headroom의 실제 학습가능 크기가 미지**. 대안 분율이 크면 exact-F1 보상 RFT는 GT 관례 overfit을 학습 → 보상/기대치 재조정 후 착수.
  2. full 3도메인 base baseline 안정화(150 subset·단일run 노이즈 해소).
  3. **★명명 정정: stage-1 = "distill"이 아니라 `gold-SFT`**. TaskBench는 17K 전 샘플에 gold graph(task_nodes/links)가 있으므로 teacher 호출 불요 — §16의 "distill=커버리지"는 gold가 *없는* substrate(SOPBench E1)용 논거였고, TaskBench선 gold-SFT가 같은 커버리지를 더 싸게 제공. ⚠️**GT-generator 순환 caveat**: TaskBench GT = back-instruct(**GPT-4 생성**, HF human-verified 10.8%뿐) → gpt-4를 teacher/증강으로 쓰면 teacher=GT-generator 동일 = "frontier-comparable coverage" 주장이 부분 순환(GPT-4 그래프 관례 모방이 점수로 잡힘). teacher가 필요해지면 비-GPT-4 frontier(Claude/o-class) 사용 + 보고 시 이 caveat 병기.
  4. outcome-RFT(보상=node/edge-F1; ⚠️exact-F1의 GT-특이성 overfit 위험 → A-0 결과 따라 matching-F1 or SOPBench-실행보상 검토) → LODO+alias-마스킹(P3). 보고 문구 = **supporting 전이**(§17.9 리뷰7 사전등록).

### §17.9 ★THESIS 고정 (2026-06-10, 3정밀화+#1/#2 우선순위 → 프로그램-레벨) — §14–17 흩어진 thesis 문장 *대체*
> 메타-비판("벤치탐색이 capability로 드리프트, SOPBench→TaskBench→AppWorld") 종결. 이 §이 고정 thesis·forward guard.

**★고정 thesis (headline, 리뷰6 흡수·DONE 동결 2026-06-10)**: **고정 도구 + 사전 결정론 compute 위에서, 소형 모델이 도구-호출 경로를 *제안*하고(=coverage, *어렵고 가치 있는* 부분) — 결정론·*검사가능* 게이트가 실행 경로 soundness를 *audited 제약모델 대비* 보장(틀린/환각 호출 0; valid 경로 없으면 fail-safe abstain) — 재학습0로 도메인 전이한다. ★헤드라인 = *보장(검증가능) soundness 하 높은 coverage*를 {소형·저비용} × {감사가능 결정론게이트=soundness *검증가능성*} × {재학습0 전이} *패키지*로 (= precision=1[인코딩제약 대비]서 recall 최대화).** ⚠️soundness 단독은 abstain으로 trivially 100%=공짜·게임가능·*무가치* → 난이도·ML기여·유용성은 전부 *coverage*에. capability("frontier급 생성")·최적성(#2)=supporting/deferred(headline 아님). [근거=아래 리뷰6 #1·#2] **thesis 본문은 동결 — 추가 정제 금지, 다음은 측정(핸드오프 `HANDOFF_2026_06_10_taskbench_learning.md`). 단 아래 리뷰7은 *운용 사전등록*(보고 문구·실험 지위 확정)이지 thesis 변경이 아님; 실행 권위 = §18.**

**#1/#2 우선순위 (사용자, 현업 정합 — optimality 확정 처리)**:
- **#1 코어 = soundness 보장**: 경로후보 중 *틀린/환각/fail 경로를 절대 실행 안 함* = **게이트 속성**(모델 100%정확 아님; 게이트 reject + fail-safe). 메트릭 = (a)soundness(실행경로 valid≈100% by construction·감사) + (b)coverage(모델+게이트가 valid-solving 경로 찾은 task%).
- **#2 bonus = 최적경로 선택**: **deferred**(usage-DB 부재·실데이터 필요). CostBench-동적/효율 = 후행 Tier 2.

**moat = 패키지 (정밀화 A+)**: "감사가능 구조생성 capability"는 frontier도 함→침식(§15.4 c12/13)→moat 아님. 내구 = 위 3-팩터 곱. headline=패키지, capability=supporting.

**벤치 분담 (정밀화 B+)**:
- TaskBench = **충실성 반쪽**(NL→구조 soft-F1, *soundness·실행 없음*) → **#1 보장 주장 불가**.
- SOPBench/SOP-Bench = **soundness + 제약 + 전이 = #1의 진짜 자리**(실행·게이트·거부).
- **통합(NL→구조→게이트실행→success+전이) = SOPBench/SOP-Bench + 우리 풀파이프라인(E1/blind-induce) end-to-end** (PRAXIS=실세계 현실성 *추가*, 통합 전용 아님). ⊥ 두 벤치 disjoint → "thesis 검증완료" 과대주장 금지(반쪽씩+통합=E1).

**★forward guard (벤치선택 운용규칙, 박제)**: 어떤 벤치도 **"소형이 frontier 이기나"로 평가 금지**(capability 함정). **평가 3질문 = (1)구조-충실성? (2)감사가능-실행/제약/soundness? (3)재학습0-전이?** 셋 다 ✗면 C1~C6 통과해도 substrate 아님. ⇒ **AppWorld(셋 다 ✗·코드모달리티·capability) 자동탈락**; TaskBench(1)·SOPBench(2,3) 통과.

**★★리뷰6 정정 (2026-06-10, 협업자 재비판 — 위 문구 2 과대 정정 + 3 명기. 헤드라인 방어가능성↑)**:
1. **"100% sound by construction" → "감사가능 soundness"(§13.7 자가반박 회피)**: 게이트는 *인코딩된* 제약에 sound지 *절대* sound 아님 — ABox/제약 인코딩이 손작성·오염·틀릴 수 있음(§13.7 실증; Guard-2 OVER0/UNDER0도 *재구성 충실* 가정). 인코딩≠진짜정책이면 **틀린 정책을 sound하게 집행**. ⇒ by-construction은 *(i) 인코딩-그래프 부합* 수준만; *(ii) 진짜정책 부합*은 audit 사안. **moat = soundness의 *검증가능성***(게이트가 결정론·검사가능→감사/verify 가능; stochastic 모델은 effective-policy를 audit 불가) = §15.4(규제=추적성) 정합. "by-construction 100%"를 이걸로 강등.
2. **★"coverage=supporting" → "coverage가 헤드라인·난이도·가치"(self-contradiction 회피)**: soundness는 **abstain만 하면 trivially 100%**=공짜·게임가능 → *단독 무가치*. 난이도·ML기여·유용성 전부 **coverage = 보장 soundness 하 valid-path-finding률**에 있음. 정확한 관계: **soundness = 내구 *차별점*(공짜·by-construction·게임가능 단독)** / **coverage = *value leg*(어렵·commoditizing[frontier도 함]·유용성 필요조건)**. ⇒ **진짜 헤드라인 = "보장된 soundness 하 *높은 coverage*"(precision=1서 recall 최대화)**, "soundness 보장"이 아니다. "supporting"=*moat 아님*이지 *쉬움/부차* 아님(abstain-always가 thesis 만족=자가모순 회피). 우리 coverage 주장 = *frontier-comparable coverage를 싸게+전이*(=패키지).
3. **통합 = blind-E1(paused)이 진짜, 현 Exp-5는 §1-노출판**: Exp-5 = 구조화입력 *주어짐*(§1 결정론공격 노출, done) ≠ 통합. 의미있는 통합(NL서 구조 *도출*→게이트→전이) = **blind E1(미구축 + §1.1 inherited)**. "통합=SOPBench"가 구조화입력판으로 *충족된 듯* 보이면 안 됨 → 진짜 통합검증 = E1, 멈춤.
4. **forward guard 가중(OR 너무 관대)**: (1)충실성-only = *supporting substrate*(TaskBench가 예; §17.9 스스로 "#1 주장 불가"). **moat-검증엔 (2)soundness ∨ (3)전이 필요.** ⇒ moat-substrate = (2)∨(3); (1)-only = supporting-only. 충실성-only 벤치를 moat-검증처럼 admit 금지.
5. **optimality #2 deferral = CDP 비즈니스가치 일부 미룸(scope 플래그, 도메인 의존)**: 마케팅 ROI 상당부분 = 효과성(최적 세그먼트/접근 = 최적성). "sound but not optimal" = 틀린 건 안 하나 효과적이진 않을 수 → ROI 상당부분 defer. ⇒ 학술 moat(soundness=안전/컴플라이언스) ↔ CDP 가치(효과성) **부분 불일치 = 도메인 의존**: *규제/고위험*(은행=우리 substrate)→soundness=비즈니스가치(thesis 최강) / *마케팅-최적화*(CDP 일반)→optimality=가치(deferred). ⇒ **thesis는 규제/컴플라이언스 orchestration서 최강 위치**; 마케팅-효과성은 별도 deferred 축(블로커 아님·인지 플래그).

**★리뷰7 (2026-06-10, 외부리뷰 — 핸드오프 대조 정합. *운용 사전등록 3건*, thesis 본문 불변)**:
1. **★TaskBench LODO 전이-지위 사전등록 (Exp-A 보고 전 BLOCKING)**: 핸드오프 Exp-A가 TaskBench LODO를 "전이 leg 첫 측정"으로 표기 → forward-guard 표(TaskBench=(1)충실성-only 통과)와 충돌 — LODO가 (3)전이로 인정되면 TaskBench가 moat-substrate가 되어 리뷰6-4("(1)-only=supporting-only")와 자가모순. **확정: TaskBench LODO = *supporting* 전이 증거만**(동일벤치 내 · n=3 저검정력 · alias-마스킹 조건부) — **moat-(3) 주장은 cross-bench(SOPBench→SOP-Bench) 전이로만 가능.** Exp-A 헤드라인·보고 문구를 "supporting"으로 고정.
2. **Exp-A stage-1 = gold-SFT 명명 정정 + GT-generator 순환 caveat** (상세 §17.8 "다음" 3항): "distill" 표기 폐기, gpt-4-teacher는 GT(back-instruct=GPT-4)와 생성자 동일 = frontier-비교 부분 순환.
3. **A-0 zero-cost edge-miss 감사 선행** (상세 §17.8 "다음" 1항): RFT 전 BLOCKING — edge 20pt headroom 중 valid-대안 분율 미지(P2)이므로 실측 후 보상 설계.

---

## §18. ★실행 큐 (권위본, 2026-06-10 리뷰3 — §11 대체 · 핸드오프 통합 · disposition/open 배정 박제)
> 원칙(§11서 승계): ①zero-GPU 먼저 ②정면반박 조기 ③측정 우선·정제 금지. 핸드오프 `HANDOFF_2026_06_10_taskbench_learning.md`와 1:1 정합(충돌 시 이 §이 권위).

### §18.1 측정 트랙 (GPU/리모트)
1. **Exp-A (1순위) — TaskBench coverage+supporting-전이** (인프라 READY, 핸드오프 §2):
   **(A-0, zero-GPU, BLOCKING)** edge-miss ~30개 수동 감사(real-error vs valid-대안 분율) → full 3도메인 base baseline → **gold-SFT**(="distill" 아님, §17.9 리뷰7-2) → outcome-RFT(보상 = A-0 결과 따라 exact-F1/matching-F1/실행보상) → LODO + alias-마스킹. **보고 = supporting 전이**(리뷰7-1; moat-(3) 주장 금지). 지표 = edge-F1 중심(node ~saturated) + type 층화.
2. **Exp-B (병렬, #1 soundness leg = 진짜 통합) — SOPBench/SOP-Bench blind-E1 재개**:
   선결 = §1.1/§6-E1 found-inherited 게이트(NL-source induce 경로 신설) + §13.7 재설계 프로토콜(test-blind·held-out split·clean 도메인 — traffic_spoofing docstring audit 포함 14도메인 mis-wiring 전수). 첫 수 = **sop.txt→(ontology+결정코드) frontier-1-call 시제품 on customer_service**(§13.7 probe-2 타깃 격상 반영). 성공 시 소형모델 사다리(=P2/E1 실제 주장).
3. **Exp-C (선택) — scale 곡선**: Qwen2.5-{0.5,1.5,3,7,14}B edge-F1 (모델 캐시 완비) = "edge-구조 emerge 지점".

### §18.2 zero-GPU 병렬 트랙 — §15.4 사활 open 3건 배정 (큐 누락 해소)
4. **★규제 1차원문 sourcing (moat-결정론-leg 사활, load-bearing)**: ✅**DONE (2026-06-10)** — `REGULATORY_DETERMINISM_SOURCING.md` 커밋. "로깅+검증이면 충족"(c) 판명 → §15.4 open-1 CLOSED·결정론-leg 철회 실행. 보상 수확=SR 26-2 정의-비대칭(결정론=MRM 면제)+MDR §17.1.
5. **bitter-lesson 1차원문 반박**: ✅**DONE (2026-06-10)** — `BITTER_LESSON_REBUTTAL_SOURCES.md` 커밋(dbbd529), §15.2-①/§15.4 open-2 갱신 반영.
6. **agentic-SOP erosion 테스트**: §15.4 c12/13(formalizer-우위 erosion)이 simple-PDDL 넘어 messy-NL-SOP로 확장되나 — Exp-B 인프라 위 후행.

### §18.3 disposition — 구 §11 실험들 (무처분 소멸 금지, 사유 박제)
| 구 항목 | disposition | 사유 |
|---|---|---|
| **E0** (gather-grounding 격리) | **Exp-B로 흡수** (blind-E1의 gather-소스-교체 ablation arm으로 포함) | 단독 실행의 신규 정보가 적음 — Exp-5 LODO(0→43% 전이)가 이미 동등 1급 증거 제공. 폐기 아닌 흡수 |
| **E2** (perturbation robustness) | **연기 (폐기 금지) — P1 집필 게이트 전 필수 복귀** | §9-2 amortization/Δ-LOC 헤드라인과 P1 비용축(A7)의 **유일 생산처**. 현 시점 미실행 사유 = thesis 헤드라인이 coverage-패키지로 이동(§17.9), 비용축은 P1 supporting으로 강등 — 단 P1 쓸 때 없으면 비용 주장 전체 공허 |
| **E5** (Track B DPO 파일럿) | **협업자 H200 Track-B로 이관** (32B SFT 진행 중) | adapter-only>0 신호는 여전히 P2 사활 → 협업자 결과를 P2 go/no-go 게이트로 사용. B'-critic 병렬 arm(§13.2)도 그쪽 |
| **E3** (멀티턴 통제 래퍼)·**E4** (scale stress) | 후행 유지 (변동 없음) | E1/soundness leg 뒤 |

### §18.4 순서 핵심
**A-0(zero-cost) → Exp-A ∥ {규제 sourcing(4)·bitter-lesson(5)} → Exp-B 선결게이트→시제품 → P1 집필 전 E2 복귀.** 죽으면 일찍 알아야 하는 것 = Exp-B frontier-1-call(NL→구조 feasibility)·규제 sourcing(moat-leg) — 둘 다 큐 앞쪽.
