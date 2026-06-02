# ★ EXPERIMENT DESIGN — MASTER (단일 권위본, 이것부터 읽을 것)

> 2026-06-01 신설. **설계서 난립 방지용 단일 진입점.** 목표·현재 실험 순서·헤드라인 지표를 여기서 고정한다.
> 세부는 §7 문서지도의 detail 문서로. **방향이 흔들리면 이 문서 §0~§3만 다시 읽는다.**

---

## §0. 목표 (한 문장, 변하지 않음)
**자연어 멀티턴 요청을, 도메인별 구조화 온톨로지(ABox)로 재해석해 내부적으로 절차(dirgraph)를 추론·실행하는 agentic planner를, 작은 모델 weight(TBox)에 학습시키고, 본 적 없는 도메인은 ABox 교체만으로 재학습 0 전이한다.**
- **TBox(weight, 학습·전이)** = "NL 요청 + ABox 어휘 → dirgraph(절차) 도출 + 실행" 스킬. **★TBox는 NL 정책도 dirgraph도 *아니다* — 둘 사이의 *컴파일 스킬*(도메인-일반)**. NL 정책 = ABox(도메인-특수 *입력*, swap) / dirgraph = **모델 *출력***(컨닝 아님, 도메인-특수). 정책을 weight에 구우면=FM weight-baking(전이 불가); 정책은 ABox에 두고 *컴파일 스킬*만 weight = 새 도메인은 정책 교체로 전이. L0(결정론)는 NL→dirgraph 불가(난이도 주장, §1에서 정량 검증) → 이 매핑이 비자명·대체불가 기여.
- **ABox(in-context swap, 후속 xattn)** = 도메인 도구 affordance + NL 정책. goal precondition '정답 구조'는 안 떠먹임.
- injection/steering(구 라인)은 agentic서 null로 폐기됐다 재정의(§3 Rung3). FM SO.P/CAP-CPT의 weight-baking과 달리 **TBox/ABox 분리 → 전이**가 핵심 차별.

## §1. 벤치마크 · 평가
- **주 = SOPBench(Zekun Li, 2503.08669)**, 7도메인. **파일럿 = bank(N=134: should_T 48 / should_F 86)**. rule oracle `env/evaluator.py`(LLM judge無).
- **헤드라인 지표**:
  1. **LODO 전이** (6도메인 학습 → held-out bank, ABox swap, **재학습 0**) — 1급 결과.
  2. **should_T = dirgraph+ ∩ goal+ 동시충족** (BOTH). ⚠️**총 Mean Pass Rate는 거부로 부풀려져 헤드라인 금지**(s1 0.605=거부 degenerate 함정).
  3. **should_F gross** gain/loss(net 금지).
- **평가 세팅**: 현재 정적-user(`default_response` 덤프, leaderboard 정합). **멀티턴 user_sim**(`--user_model`)은 더 어려운 robustness 축(천장 불변 — user_sim도 user_known만 앎; PartB cred-부재 해소 안 됨).
- **ablation(매 rung)**: 빈/틀린 ABox→붕괴(온톨로지 실사용) · L0 vs L1(in-context) vs L2(학습) · **alias on/off**.
- **★★TBox/ABox 분리: 강제 메커니즘 + 증명 (transfer 주장의 근간)**: "TBox만 학습·ABox 제외"는 *하드 보장*이 아니라 **3기둥으로 강제 + 전이/ablation으로 증명**.
  - **강제 3기둥**: ① **loss=assistant-only**(ABox=정책·도구affordance·요청은 *마스킹된 프롬프트*에, dirgraph 스텝 target만 supervise=labels -100 except assistant) ② **alias**(도구명 별칭화→lexical 암기 차단·NL설명↔도구 의미매칭 강제; ⚠️s1=실도구명 타깃 노출→누수 가능→**헤드라인은 alias_s3/alias-Δ**) ③ **다중도메인 LODO 혼합**(6 도메인 동시 → 암기로 못 풀어 *공통 불변량=절차 스킬*만 추출; 단일도메인 ABox 과적합 불가).
  - **증명**: held-out bank 전이(재학습0) + **ABox-ablation**(빈/틀린 ABox→붕괴 = weight에 안 구워짐) + **alias on/off Δ**.
  - **★혼합이 강화하는 *축* (실측 분해)**: 6도메인 혼합 = **게더 TBox 강화·전이 성공**(dirgraph 36-45/48 held-out) **but 순서 게이트 미해결**(BOTH 0-2) → 혼합은 *공통구조서 배울 수 있는 축*(게더)만 강화; 순서 경합은 사다리(①②③) 필요. "혼합이 더 강하다"는 **게더 축에 한해 사실**.
  - **★개수 효과 = 미측정 → 실험**: **도메인-수 스케일링(2/4/6 → bank 전이 BOTH·게더)** 우상향이면 "혼합→TBox 강화" 정량 입증 + **그래프충실도 vs 구조유사도**(보간 vs 일반화 판별; bank가 학습도메인 near-dup면 전이=보간으로 주장 약화). 다양성 > 개수, negative transfer 주의.
- **★★벤치 횡단 전이: SOPBench(학습) → SOP-Bench(Amazon, 2506.08119) ABox-swap 재학습0 (2026-06-01, 표현 적합성 검증 완료)**: 8관계 스키마 공유 → SOPBench 6도메인 학습 TBox를 **벤치 경계 넘어** SOP-Bench로 ABox swap만 전이 = 도메인-일반을 넘은 **벤치-일반** 증명(LODO보다 훨씬 강한 주장).
  - **표현 적합성 = 강함 ✓ (전문 정독)**: SOP-Bench(12도메인·2411 task/13 SOP·9 tools/SOP·1394토큰) = free-text NL SOP + 메타(input/decision/outcome) + tool I/O + executable code. 매핑: 순차 step→`realizes`+`next`, tool I/O→`realizes`/`arg`/`produces`, prereq→`precondition`, decision point→`next`(조건), 서브절차→`scenario_*`, 조기종료→`terminate`, 최종→`output`. **scoring/threshold(점수≥0.4·decision table)=함수 캡슐화**(우리 설계와 정확히 일치). **루프/병렬은 SOP-Bench에 부재** → 8관계로 충분(있었으면 미적합). eval=**최종상태 매칭**(ECR/C-TSR/TSR)→SOPBench rule evaluator와 호환.
  - **단(정직)**: SOP-Bench는 훨씬 길고 복잡(28 step·12 decision points·복잡도 7–8/10 vs bank 1–3 check) → **순서/conjunction의 스케일 테스트**(BOTH 낮을 수 있음; scratchpad `all_verified`가 여기서 핵심) + SOP-Bench ABox를 induce 파이프라인(8관계 call-graph)으로 추출 필요. 향후 루프/병렬 필요한 도메인은 8관계 미적합(스키마 경계 명시).
  - **무학습으론?** 스키마(8관계)만으론 못 풂 — 7B in-context 무학습은 SOPBench 자기 자신도 실패(arm-3 0%·arm-3v2 게이팅무시); 대형 모델 in-context는 부분 가능. **관계=표현이지 NL→dirgraph *스킬*(TBox) 아님** → 횡단 전이의 핵심은 *학습된 TBox*가 벤치 경계를 넘느냐.
- **★★결정론 도구 offload 경계 (검증-타당성 북극성; "어디까지 도구로 빼도 되나" 흔들리면 이것부터)**: 작은 모델이 확률적으로 할루시네이션하는 부분은 결정론 도구로 빼는 게 옳다(ReAct·verifier·PAL 표준). **단 경계 = 사실(fact) vs 절차(procedure)**.
  - ✅ **사실 offload (권장·표준)**: "precond X가 *실제로 충족됐나*"의 검증 = 결정론 도구(할루시네이션 0). 사실-노이즈 제거.
  - ❌ **절차 offload (= 답지 = 기여 자체, 금지)**: "이 goal엔 *어떤* precond가 필요한가 + 순서"(=dirgraph/TBox)를 함수가 쥐면 = NL→dirgraph 추론을 외부가 대신함 = **L0**. 모델은 도구호출기로 전락, 전이(ABox swap 무재학습) 주장 붕괴.
  - **★결정론 전부-처리 버전은 이미 존재 = `run_scripted`(오라클 천장 37/48)**: 방법이 아니라 *상한*. 전이 불가(도메인마다 절차 손-인코딩 = 격파 대상 FM weight-baking) → method면 전이 주장 소멸. (구 "ACT 전제조건 가드"를 supersede한 이유.)
  - **깨끗한 분해(직관 살림+기여 보존)**: 모델(학습·전이) = NL정책+history → **required-set + 순서 emit**(=dirgraph) / 도구(offload) = 각 precond *사실* 결정론 검증 / **readiness = 모델이 *주장한* set에 대한 사실검증 AND** → ACT 게이트. 모델이 required-set을 틀리게 주장(누락/과잉)할 수 있고 **그게 측정하는 학습 타깃** — 도구는 사실 할루시네이션만 없애지 절차 추론은 못 빼냄. 노트 도구는 *raw 관찰*만 담아야지 "ready 판정"을 담으면 안 됨.
  - **논문 처리**: 사실검증 도구는 설계로 명시 + **ablation**(with/without): 절차 정확도(required-set·순서)가 학습 신호임을 분리 입증. 범위만 못 박으면 novelty 강화(깨끗한 factoring).
- **★★그래프-충실도 지표 + NL→graph 실현가능 천장 (BOTH 보조, 절차축 분리)**: 현 헤드라인 BOTH는 *절차추론+실행+순서*를 뭉뚱그림 → **모델이 emit한 절차 그래프 vs GT `directed_action_graph`**를 **노드·엣지 P/R + graph-edit-distance**로 비교 = **절차추론 축만 분리 측정**(어디서 깨지나: 추론 vs 실행).
  - **NL→graph 실현가능 천장 (벤치 타당성·thesis 전제 검증, 선행 필수)**: 강한 컴파일러가 NL 정책*만*으로 GT dirgraph를 복원 가능한가? **세 컴파일러 대조군 = 결정론 파서(L0) / 큰 LLM(천장) / 우리 작은 모델(method)**, 기준=GT dirgraph. 복원 가능→학습 신호 존재(정당); **복원 불가→dirgraph가 NL에 없는 정보 포함=배울 수 없음=벤치 타당성 문제**("L0 불가"는 *난이도*여야지 *불가능*이면 안 됨). ⚠️ 컴파일 결과를 모델에 *떠먹이면* 절차 offload=L0(§1 경계) — 비교는 "누가 얼마나 잘 컴파일하나" 진단이지 입력 주입 아님.
- **★구조적 vs 공간적(RAG) baseline + 10k라인 동기**: 정책 처리 2패러다임 — **공간적/RAG**(GraphRAG·Graphify·LLM-Wiki: 텍스트 유지, 청크 retrieve해 프롬프트 부분처리; "그래프"=retrieval 인덱스; 전역 의존 누락 위험) vs **구조적**(정책→명시 절차그래프 컴파일, 전역 위상순 추론; "그래프"=실행가능 절차; **모델이 emit=내재화·전이**). **baseline = RAG-over-policy**(스텝마다 청크 retrieve, 명시 절차그래프 無) → 구조적 > RAG면 절차 컴파일이 retrieval 넘어 기여. **동기 = 10k라인 정책**(매 결정 context 불가·비용/손실; RAG는 먼 절 cross-ref precond 누락=globality; 컴파일=goal당 노드 몇 개로 압축+스킬 내재화로 전이). ⚠️ SOPBench 실정책은 1만 라인보다 작음 → 1만은 실세계 엔터프라이즈 SOP 타깃(2단계: SOPBench 메커니즘 증명→대형 실정책 stress-test).
  - **선행 — 공간적/그래프 RAG 계열 (전부 검증 2026-06-01, 우리와 구분 명시)**:
    - **RAG** [Lewis et al. 2020, `2005.11401`] = parametric+non-parametric 메모리·retrieve-then-generate 원형.
    - **GraphRAG** [Edge et al. MS 2024, `2404.16130`] = LLM이 엔티티/관계/claim 추출→그래프 *인덱스*+community detection 요약. 그래프 = retrieval 인덱스(사실), 절차 아님.
    - **HippoRAG** [Gutiérrez et al. 2024, `2405.14831`] = KG+Personalized PageRank(해마 인덱싱)로 대량 신규지식 retrieve.
    - **Think-on-Graph** [Sun et al. 2023, `2307.07697`] = LLM이 KG 위 beam search로 추론경로 탐색(plug-and-play·무학습).
    - **★★Reasoning on Graphs (RoG) — 정밀분석(전문 정독, 2026-06-01)** [Luo et al. ICLR'24, `2310.01061`]: **방법론적으로 우리와 가장 인접**. planning-retrieval-reasoning. **planning 모듈이 relation path `z={r1..rl}`를 토큰으로 *생성***(`<PATH> r1 <SEP> .. </PATH>`)=구조 emit. **distill 학습**: `L_plan=−Σ log P_θ(z|q)`, Z*=question→answer 최단 KG경로(GT 구조 증류) + `L_reason=log P_θ(a|q,paths,G)`. LLaMA2-Chat-7B, WebQSP/CWQ+Freebase(88M ent·20k rel) 3ep. retrieval=z 따라 constrained BFS로 reasoning path 인스턴스. 결과 WebQSP 85.7 Hits@1·CWQ 62.6.
      - **공통(=RoG가 선례)**: 7B가 *구조(path)를 생성*·**GT 그래프로 distill**·plan-then-ground(faithful)·특수 path 토큰 → "7B를 GT dirgraph emit하도록 SFT"의 방법론적 선례.
      - **★우리 차별(4)**: ① **NL→구조 컴파일**: RoG는 path를 *기존 KG 스키마(20k relation)에서* 생성(스키마 주어짐·BFS로 grounding) / 우리는 **NL 정책 prose에서 절차 구조를 *추론·구성***(스키마 미제공) = path-finding vs graph-construction. ② **사실KG vs 실행 절차**: RoG 그래프=사실 relation(QA "누가 X") / 우리=실행 절차(login→check→act, 의존·순서 의미). ③ **전이**: RoG는 **KG/데이터셋마다 재instruction-tune 필요**(논문 명시 약점, App A.7.1) / 우리 헤드라인=**LODO ABox-swap 재학습0**(RoG 약점 축이 우리 중심 주장). ④ **순서 경합 부재**: RoG=retrieval chain(act-vs-stop gate·조기행동 문제 없음) / 우리 병목=gather-act 순서.
    - **Unifying LLMs & KGs: Roadmap** [Pan et al. 2023, `2306.08302`] = survey 우산.
    - **엔지니어링 도구(arXiv 아님, 명시)**: **LLM Wiki**(`nashsu/llm_wiki`·Karpathy 개념) = RAG의 매번-재검색 대신 *persistent 위키로 사전컴파일*(부분적 구조화 but *지식* 위키지 실행 *절차* 아님·학습/전이 아님). **Graphiti**(Zep/Neo4j) = temporal KG 에이전트 메모리. **Graphify**(`safishamsi/graphify`) = 구조적 지식 도구.
    - **★우리와 구분(한 줄)**: 이들은 *사실/엔티티 KG*를 build·retrieve(그래프=검색 인덱스). 우리는 **NL 정책→실행가능 *절차* 그래프(dirgraph)를 *모델이 emit*** (그래프=실행 절차·weight 내재화·ABox swap 전이). RoG(구조 emit)·LLM Wiki(사전컴파일)가 인접하나 **절차 컴파일 + 학습 전이**가 우리 고유.

## §2. 현재 진단 (어디까지 왔나)
- **arm-1(LLM-alone) bank 5.2% / arm-3 naive 0% / arm-3v2(in-context 구조, 무학습) gating 무시 / arm-4a(학습 L2) 26.1%(should_T 4/48)**.
- **★run_scripted(결정론 직접제어): 완전게더 A+B+C = 37/48(oracle), 프롬프트-LLM = 4/48.** → **should_T binding = 도구 선택(=게더: 어떤 검증/establish 도구를 부르는가)**. "직접제어 ≫ 프롬프트" 실증.
- **★현 SFT 진단(전수조사)**: **게더는 학습됨**(dirgraph 36-45/48, LODO 전이) **but 터미널 게이트(게더 후 act vs STOP) 못 박음** → **BOTH 0-2**. s1(STATUS)=게더후 STOP / s3(NL)=안게더 act. gate-token(Exp-4d)=부분(act 3→13, BOTH 0→2). **= 병목이 게더에서 터미널 게이트로 이동.**
- **결함 제외 천장**: should_T 정직천장 ~34/48(8 PartA 코드결함 + 6 PartB cred-부재 제외; run_scripted oracle 37).

## §3. ★실험 순서 = 학습 사다리. 핵심 = **게더-act 경합(순서) 해소: "게더 미완이면 ACT 금지"를 *학습*** (가드 아님)

> 병목(전수조사) = 게더는 학습됨(dirgraph 36-45) **but 게더 AND act 공존 못함(BOTH 0-2)**: 게더후 STOP(s1)·안게더 act(s3) = **순서 경합**. bank는 체크 1-3개라 conjunction-*계산*은 안 어려움 → 병목 = **act/STOP 결정·순서**(s1은 READY 보여줘도 over-STOP). 목표 = **"ready(required 전부 게더)면만 ACT"를 weight에 학습**(결정론 가드 offload 아님). 세 레버 = **①SFT readiness-게이트 → ②DPO preference → ③RFT 페널티**. (순서 정당화: process supervision은 통계적 우위 없음[Jia ICML25] → ①② 먼저, ③로 경계.)

### Phase 1 = Rung 1 = ① SFT: per-step readiness 게이트 토큰 〔현재〕
- **목표**: "게더 미완이면 ACT 금지"를 SFT로 학습(순서 경합 해소).
- **방법(deep-research 2026-06-01 확정, 적대검증 20/25)**: 매 스텝 타깃 = **`ready=<true|false>; <행동>`** (educated 중간 게이트):
  - **게더 미완** → `ready=false; <다음 게더 도구>` — ★학습 데이터에 **`ready=false` 뒤 ACT가 한 번도 안 나옴 → "미완→ACT 금지"를 구조적으로 학습**(positive-only SFT로도 순서 강제).
  - **게더 완료** → `ready=true; all_verified=<T/F>; <ACT|STOP>` (all_verified = 게더된 체크 truths의 **AND**; true→ACT/false→STOP).
  - ready 판정 = required 다 게더됐나(prompt observed/history서 셈) = act결정보다 **단순 sub-task** → 명시 supervise면 학습 [educated 중간타깃 Abbe NeurIPS24; parity+CoT 분리 Kim&Suzuki ICML25]. **★educated 필수**(그냥 "생각공간"은 globality barrier 못 깸).
  - **scratchpad(all_verified)는 conjunction-*계산*용**(체크 많은 복잡 도메인 일반화); **bank엔 readiness 게이트가 핵심**(작은 conjunction). 같은 토큰 메커니즘.
- **한계(SFT positive-only)**: "조기 ACT는 *나쁨*" 음성 신호 부재 → ready 오판시 조기 act 잔존 → **② DPO**.
- **근거 논문**: Nye scratchpad(2112.00114)·Teaching Arithmetic(2307.03381)·**Kim&Suzuki parity+CoT 증명(2410.08633)**·**Abbe globality+inductive scratchpad(2406.06467, NeurIPS'24)**·Feng CoT expressivity(2402.12875)·least-to-most(2205.10625).
- **소스코드 재활용 판정**: 학습코드(`aryol/inductive-scratchpad`, `lee-ny/teaching_arithmetic`)는 **from-scratch 합성과제(parity/arithmetic)라 우리 LoRA-SFT엔 직접 재활용 불가**. **재활용 = 데이터포맷 레시피**(inductive carry-update + educated 중간토큰)를 `build_tbox_planner_sft.py` teacher에 적용. 트레이너는 `lora_train_chat_toolcall.py` 유지.
- **⚠️근거 한계(정직)**: 증거 대부분 parity/arithmetic/automata(parity=XOR≠AND, 증명은 1-layer) → 7B tool-use 전이는 **analogical**. 과대주장 5 sub-claim은 검증서 killed(SRL 등). 일반 원리(educated 중간토큰 supervision)는 강함, gate 분해는 동기부여된 instantiation.
- **현황**: Exp-4d(gate-token)=부분(act 3→13·BOTH 0→2). Exp-4e=절차분해(inductive scratchpad). **목표 BOTH 0-2 → 다수.**
- **★★Rung1 v2 실측 + grounding 비대칭 발견 (2026-06-02, raw RLLOG 전수)**: 2-토큰(`preconds_verified`+`permitted`, ACT iff 둘 다) 재학습. **클린 측정**(수렴 정상=134/134 터미널 도달; confound 제거): bank LODO **s1 BOTH 0·alias_s3 BOTH 1**, **클린 over-refusal = (should_T ∩ login=True)서 ACT 0/30(s1)·0/23(alias_s3)**=불균형(2:1)로 설명 안 되는 **다수클래스 붕괴**. raw 터미널 92/98 = `preconds_verified=true; permitted=false; STOP`(포맷 0-malformed=분기·파서 정상; **값만 false 쏠림**).
  - **★메커니즘=grounding 비대칭**(불균형/holistic/일반화 나열보다 정확): `preconds_verified`=gather가 history에 박은 부울 **룩업**(성공) vs `permitted`=정책을 요청에 적용하는 **콜드 추론**(대응 gather 없음→붕괴). 같은 AND인데 *근거 종류*가 다름. (alias_s3 ACT 4개 전부 noLogin=조건 적은 태스크 → ungrounded 조건 쌓일수록 거부 = 비대칭 확증.) 불균형=증폭기, confound=측정 차단기(선결), 둘 다 근본 아님.
  - **⚠️[priority-lock 후속 정합, 2026-06-02 PM]**: 아래 v3 처방은 **두 부분으로 분해됨** — **load-bearing = "required-set에 policy조건 포함"**, 이것의 구현이 곧 **Track A(condition→getter 맵 auto-derive)**(위 priority-lock에서 1순위로 잠금). **`wait` 3-상태 토큰 = 그 위 *선택적* scaffold**(caveat① A/B ablation 대상; 조건폭 2-4라 이득 잉여일 수 있음). 즉 v3=설계, **Track A=v3 load-bearing의 실행본**. 아래는 원 v3 설계(참조).
  - **★★처방 = Rung1 v3 = `wait` 3-상태 grounded-permitted (gather-until-resolved)**: permitted을 콜드추론→룩업으로 강등. 각 정책조건 ∈ {true,false,**wait**}; **하나라도 wait → `ready=false; <그 조건 gather 도구>`(판단 보류)**; 전부 resolved시만 `ready=true; login=T; credit=T; ...; permitted=AND; ACT|STOP`. = **readiness-gate를 permitted 조건까지 일반화**(preconds∪policy를 단일 gather-until-resolved 루프로 통합). wait판정=부재탐지=룩업·wait→gather=기존 스텝·전부resolved→AND=trivial → **전부 grounded=SFT학습가능**(콜드추론 소멸). 보너스: 불균형 무효(값=게더로 결정)·confound→wait→사용자에 자격증명 요청(멀티턴, 거부 아님)·유한루프(조건당 1회). **실현**: bank 임계조건 게더도구 존재(`internal_get_credit_score` 등) — teacher required-set이 policy조건 누락이 버그. caveat: 도구없는 순수 정책룰은 여전히 콜드(도메인 점검). 구현=`build_tbox_planner_sft` required-set에 policy조건 추가+터미널 wait-aware.
  - **★선행연구 (전부 arXiv 검증, 2026-06-02)**: **Self-Ask/Compositionality Gap** [Press et al. 2022, `2210.03350`] = ★*이론적* 최인접: "compositionality gap"(서브Q 답함 but *조합* 실패, 모델 커져도 안 줆) = 우리 grounding 비대칭(preconds 룩업됨/permitted 조합 붕괴) 그 자체; 명시 follow-up 서브Q 분해 후 답 = 조건별 게더 후 AND. **Slot-filling DST** [예 Ye et al. WWW'21, `2101.09374`] = ★*고전적* 최인접: required 슬롯 tri-state(filled/**requested**/empty) 다 채우고 API 실행 = 우리 wait-until-resolved. **ReAct** [Yao et al. ICLR'23, `2210.03629`] = gather(act)+reason 교차, 결정 전 정보수집. **R-Tuning** [Zhang et al. 2023, `2311.09677`] = SFT로 unknown 거부(refusal-aware IT; 단 *이진 abstain*이지 게더-해소 아님). **GATE/Eliciting Preferences** [Li et al. ICLR'25, `2310.11589`] = LM이 빠진 정보를 질문으로 elicit = wait→사용자 자격증명 요청. (+개념) PDDL/STRIPS operator precondition 충족=symbolic AND-gate.
  - **★우리 차별**: 위 어느 것도 결합 안 함 — **구조적 절차 게이트(permitted=정책조건 AND)를 agentic SOP에서, tri-state가 *도구*-게더를 구동(슬롯=user발화 아닌 정책조건 도구확인), 생성형 gate-then-act를 *weight에 학습*해 ABox-swap 재학습0 전이.** Self-Ask=프롬프트 QA(무학습·행동 아님)/DST=user발화 슬롯 분류/R-Tuning=이진 거부(게더 없음)/GATE=선호 elicit. = "도구-grounded 조건게더 + 학습된 gate-act + 전이"는 우리 고유.
  - **★★"모든 조건 충족까지 대기" 추론 계보 (심층, 2026-06-02 검증) — 왜 ReAct엔 없고 어디 사는가**: greedy 추론(ReAct·CoT·Self-Ask)은 좌→우 토큰결정이라 **전-조건 게이트가 없음**→조기결론(ToT 논문 명시: "LM은 토큰레벨 좌→우에 갇혀 초기결정 결정적 과제서 실패"). = 우리 permitted 조기결정. **sound 추론의 "전 조건 후 결론" 게이트가 사는 3계층**:
    - **(a) 고전 기호주의(뿌리)**: **production system/forward-chaining/Rete** [Forgy 1982, 고전] = 규칙은 좌변 조건 *전부* 매치시만 발화 = 문자 그대로 wait-until-all + PDDL/STRIPS operator precondition. **우리 wait-3상태 = 이 "전조건-발화"의 신경망 재구현.**
    - **(b) LLM 검증/계획 루프**: **★LLM-Modulo** [Kambhampati ICML'24, `2402.01817`] = "auto-regressive LLM은 혼자 plan·self-verify 못 함→외부 검증기가 모든 제약 체크 generate-test" = **우리 실측(permitted 콜드추론 붕괴)의 이론적 진단**, 처방(외부 검증기)=우리(각 조건 도구-ground). **CoVe** [Dhuliawala 2023, `2309.11495`]=검증질문 각각 답한 뒤에야 확정. **ToT** [Yao NeurIPS'23, `2305.10601`]·**RAP** [Hao 2023, `2305.14992`]=상태 탐색+평가 후 commit.
    - **(c) LLM 분해**: Least-to-Most [`2205.10625`]·Decomposed [`2210.02406`]·Self-Refine [`2303.17651`] = 모든 서브 해결/만족 후 조합.
    - **★종합 포지셔닝**: LLM-Modulo "self-verify 불가" + Self-Ask "compositionality gap" = 우리 grounding 비대칭의 *예언*. **단 위 전부 inference-time(프롬프트 루프·외부 검증기·탐색)** / 우리는 **그 전조건-발화 게이트를 weight에 학습(SFT)**=모델 스스로 wait→gather→AND emit, 검증기=스왑 ABox 도구(전이). = "Rete식 전조건-발화를 *학습된·전이되는* neural 게이트로".
  - **★★v3 = 가설 (비판 반영, 2026-06-02) — 발견은 확증·처방은 미검증**: grounding 비대칭 *발견*=확증(s1+alias_s3 **양쪽 raw**로 permitted-collapse: terminal 전부 `permitted=false`; 클린 측정 수렴정상·(should_T∩login=T) ACT 0/30·0/23). **그러나 v3 *처방*은 가설**:
    - **(1) load-bearing vs scaffold 분리·ablation 필수**: *진짜* 수정 = **required-set에 policy조건(income·credit) 포함**(→기존 이진 `ready=false;<gather>`가 이미 끝까지 게더, wait 토큰 불요). per-condition **wait 토큰=그 위 선택적 scaffold**. **ablation: (A) required-set 확장+이진ready / (B) A+tri-state wait.** 조건폭 2-4(parity 약)라 B 이득이 잉여일 수 있음 — 측정 전 "wait가 처방" 단정 금지.
    - **(2) ★콜드추론은 제거 아니라 *이전*(진짜 크럭스, caveat 아님)**: "어디에도 콜드추론 없음"은 과장. **"이 goal에 어떤 조건 required인가" 식별이 여전히 추론**이고 신 부담. preconds=존재게이팅(login) vs policy=**값 비교(credit>600)** = 난이도 다름 → grounding 비대칭을 *조건-식별*로 옮긴 것이고 **그 전이는 미증명**.
    - **(3) ★완전성 census 필수(v2 `preconds=false;ACT==0`의 쌍대)**: 학습 전수에서 **AND(teacher 모델링 조건) == should_succeed** 검증. 불일치 = 누락조건 or 게터없는 순수 정책룰("US 거주자") = **wait로 못 고치는 집합 = v3 천장**. 규모 모르면 천장 모름 → v3 teacher에 census 내장.
    - **(4) confound = 인프라 의존(미해결로 표기)**: cred-부재 19개 login=wait→사용자요청은 **멀티턴 user_sim + 요청시 cred 제공** eval 전제(미해결). 단일턴이면 wait=실패 지연. 선결: 19개 should_succeed=True가 ask-for-cred 정답 vs 오라클 cred-주입 라벨(evidence_a_probe) — 후자면 evaluator가 ask 미보상→wait 무효.
    - **(5) ★novelty 재프레이밍(중요)**: "wait 3-상태"는 slot-filling DST서 수십년 = *약함*. 방어가능 기여 = **(a) required-set(무엇 채울지)을 *학습으로 추론*하는 일반 planner + (b) held-out ABox(bank LODO) 전이**(내재화/전이). (Self-Ask는 *해법모양*만 맞고 *진단*엔 부정확—우리=게더 안 하고 조기종료 ≠ compositionality gap; "최인접"은 slot-filling.)
    - **(6) 빠진 고전 정확본 = sensing-action planning**: **PKS** [Petrick & Bacchus 2002/2004, "Knowledge-Based Approach to Planning with Incomplete Information and Sensing"] = 지식DB(Kf/Kw/Kv/Kx)+**sensing action**으로 미지 해소까지 보류하는 forward-chaining contingent planner. **wait=sensing action** = STRIPS precondition보다 진짜 본가(least-commitment/value-of-information). conformant(감지無 plan) vs contingent(감지결과별 plan) 구분.
    - **★잠금 전 게이트(순서)**: (i) alias_s3 raw permitted-collapse ✅(106+28/134) → (ii) 완전성 census(천장 규모) → (iii) A/B ablation. (ii)(iii) 닫은 *후* v3 teacher 구현. 그 전까진 **가설**.

### Rung 1 진단·우선순위 확정 (2026-06-02 PM) — condition→getter 맵 결함 = 근본 / Track A(getter-map auto-derive) 선행 / 합성(§3.0b)은 fallback
> **2명 리뷰 + 코드 교차검증으로 우선순위 재정렬.** v3 "grounding 비대칭"의 *하위 근본원인*을 코드로 확정하고, "합성 먼저냐 데이터-fix 먼저냐"를 **데이터-fix(Track A) 선행**으로 잠금.

- **★근본원인 (코드 확정, 가설 아님)**: permitted-collapse의 기계적 원인 = **condition predicate가 getter 도구에 링크 안 됨**.
  - `induce_ontology_zekun.py` L.114-123: 인덕션은 **설계상** action이 state를 flip하는 `establishable`만 `by`를 링크하고(L.118-120), 값-비교 `condition`은 **의도적으로 `by=None`**(L.122-123). 즉 `by=null`은 버그가 아니라 인덕션의 의도된 한계. (실측 bank cond by=null 16/18·hc 18/19; establishable login/auth만 by-linked 2/2 → **그것만 게더됨** = preconds_verified=true/permitted=false의 정확한 원인.)
  - condition→getter 링크는 인덕션이 아니라 `build_tbox_planner_sft.py` L.76-88의 **손-유지 `GETTER_BY_DOMAIN`** dict에 삽. teacher required-set(L.152-159)은 `pred in GETTER and GETTER[pred] in tool_names`일 때만 condition 게더 → **map에 없으면 누락→ungrounded→붕괴.**
  - 커버리지: **bank만 9개 매핑("bank verified") · 나머지 6개 도메인 GETTER 전부 비어있음.** 코멘트 L.75: *"TODO: auto-derive per domain."* → **LODO 전이(non-bank 학습→bank)면 학습 도메인들의 permitted가 전부 붕괴 상태.**
  - ★도구는 존재(induction-fix ≠ 도구 추가): bank tool_names에 `internal_get_credit_score`·`internal_check_credit_card_exist`·`internal_check_foreign_currency_available` 등 실존. **링크만 결여.**

- **★우선순위 = Track A(데이터-fix) 선행, 합성(§3.0b)은 fallback** (리뷰어 2인 합의 + 위 코드):
  - **Track A (PRIMARY)** = condition→getter 맵을 **auto-derive**(손-라벨 금지). 근거: 도메인마다 손으로 9-매핑 짜면 **"ABox-swap 재학습0 전이" thesis를 스스로 깸**(per-domain 손-라벨=전이 아님). 따라서 auto-derive(lever_decomp directed-action-graph 공기 도출, L.74 코멘트 경로)는 **선택이 아니라 thesis 필수조건.** → teacher 재생성(판별조건 게더 포함)→재학습. root을 *실데이터에서 직접* 고치는 가장 싼 길. 되면 합성 우회 불요.
  - **§3.0b 합성 (FALLBACK only)**: Track A 후에도 (clean grounding인데) 학습 안 되면 → "절차 자체가 학습되나" 격리(순수 GO/NO-GO) + many-conditions 일반화(bank conjunction 작음 2-4) + 전이-thesis vehicle. **지금 1순위 아님**(합성은 induction-data 문제를 우회만 하지 안 고침 = "데이터 문제면 틀린 처방").
  - 명명 교정: 이 작업은 "induction-fix(by/produces 복구)"가 **아님** — 인덕션은 condition을 의도적 by=null. 정확히는 **"condition→getter 맵 auto-derive(teacher-side)".**

- **★결정적 pre-check (구현 전 1순위 행동, 이게 합성-vs-fix를 경험적으로 종결)**: 7개 도메인 **전 판별조건 → 존재하는 `internal_*`/getter 도구** 전수 매칭. 산출: (i) groundable 비율, (ii) auto-derive 규칙 검증 — bank 손-map(9개)을 ground-truth로 precision/recall + **⚠️6개 도메인 per-domain spot-check**(bank-only 검증은 타 도메인 일반화 미보장; single-generic-token 매칭=저신뢰 플래그 수동 점검), (iii) **getter 없는 순수 정책룰 잔여집합 = 합성/대안의 진짜 범위(=진짜 ungroundable 천장; 크면 Track A·합성 둘 다 천장 제한).** ⚠️로컬 불가(bank tool_names=리모트 SOPBench env) → 리모트 실행. `pre-check 도구=scripts/distill/sopbench/precheck_getter_groundability.py`. `income_proof_enough`·`within_enrollment_period`는 현 GETTER 미커버 → 도구 존재 여부가 핵심 미지수.

- **5개 결정 (확정값)**: ① **토큰=2-토큰 유지**(단일화 반대) — precond(=action이 state flip)와 condition(=read-only getter sense, state 불변)은 **기계적 2-범주**이고 SOPBench evaluator가 precondition-violation/constraint-violation을 **다르게 채점**; grounding 비대칭은 *증상*이지 2-범주는 *내재적*. 단 **게더 루프는 통일**(gather-until-all-resolved 단일 루프 + 터미널 2-토큰 emit; *루프 통일 ≠ 토큰 통일*). 최종 포맷은 **재생성 데이터에서 permitted≡AND 성립 확인 후** 확정(지금 단정 금지). ② **지표 위계 (리뷰 교정 — over-refusal 붕괴 가시화)**: 1차=**ACT-recall | 충분게더**(양성클래스: 게더 완료·정답=ACT인 태스크에서 실제 ACT한 비율) — 우리가 관측한 실패(항상 STOP=over-refusal 붕괴)는 ACT-recall=0으로 직접 잡힘. + **STOP-recall 분리 보고**(should_F/STOP 클래스). ★붕괴는 **비대칭**(STOP-recall=1·ACT-recall=0)이라 합친 BOTH/총점이 ACT-recall=0을 **가림** → 반드시 분리. ~~ordering-violation 1차~~ = **3차 가드레일로 강등**(전부 게더 전 조기 ACT = *반대* 실패=과소게더 검출; 붕괴 모델은 ACT를 안 해 ordering-violation=0으로 만점 → 1차로 쓰면 붕괴를 건강으로 오판). BOTH=헤드라인 산출이되 (ACT-recall|게더, STOP-recall, ordering-violation)로 분해. ③ bank=Step2(전이 증명). ④ 레버 K∈{1,2,3,5}·AND+chain, **OR 조기 포함**(ablation 아님 — OR은 "전부 게더 후 AND" 불변식을 깸=short-circuit; AND만 학습→OR도메인 전이 시 과잉게더)·distractor 2-3. ⑤ ~~Track A 병행~~ → **Track A 선행**(위).

- **잠금 전 게이트(순서)**: (i) 리모트 pre-check(groundable 비율·ungroundable 잔여) → (ii) auto-derive getter-map 구현 + teacher 재생성 → (iii) 재학습→bank/LODO 재측정(**ACT-recall|게더 + STOP-recall 분리 + BOTH**; ordering-violation은 가드레일). (iii)서 clean grounding인데도 학습 실패 시 *그때* §3.0b 합성 발동. 그 전까지 합성은 **대기.**

### §3.0b 합성 TBox-절차 격리 (clean-getter, Rung1 v4) — ⚠️FALLBACK (Track A 실패 시에만 발동)
> **상태: 대기.** 위 priority-lock으로 **Track A(condition→getter auto-derive) 선행**이 확정됨. 이 §3.0b는 *Track A 후에도 clean grounding인데 절차가 안 학습될 때*만 발동하는 격리 실험. (이전 mojibake 커밋 재작성·token=priority-lock 2-토큰 결정에 정합.)

- **동기 (코드 census)**: in-dist에서도 permitted-collapse → 합성으로 "절차 자체가 학습 가능한가"를 induction-data 결함과 **분리**. healthcare in-dist: should_T=45·goal=0·terminal `permitted=false` 다수 = teacher should_T 45/45가 클린 ACT인데도 학습 실패 → **데이터(induced ABox 링크) 결함이지 절차 아님**을 의심. 합성은 grounding을 보장해 이 의심을 격리 검증.
- **목표**: "주어진 constraint 트리 → 잎별 getter 게더 → AND → ACT/STOP" 절차를 weight에. 증명 = synth held-out 전이(재학습0), 종착 = 실 bank 전이.
- **합성 생성기 스펙** (`synth_tbox_gen.py`, 미구현):
  - constraint 트리: 가변 K∈{1,2,3,5}, DSL 미러(`["and",[...]]`·`["chain",[...]]`·`["single",pred,args]`)+**OR 조기포함**.
  - 각 잎 = predicate + **clean getter**(잎마다 산출 도구 보장 → grounding 비대칭 0).
  - **NL 정책 패러프레이즈**: 순수 비트열 금지 → NL 문장에서 의미매칭 강제(도구암기 차단).
  - **alias**: predicate/getter/도구명 per-task 별칭(lexical 암기 차단).
  - **균형**: ACT:STOP 50:50 + 잎별 truth 균형(AND 결과가 한쪽 쏠림 금지).
  - **distractor**: goal당 무관 조건/getter 2-3개("덫 도구") 섞음.
- **포맷/토큰 (priority-lock 정합 = 2-토큰 유지)**: gather `ready=false; <미충족 잎 getter>` / terminal `ready=true; preconds_verified=<AND>; permitted=<AND>; ACT|STOP`. 합성에선 모든 조건이 ground돼 두 토큰 다 룩업-결정 = **2-토큰이 real bank(precond=action-flip / condition=getter-sense 2범주)와 동형 전이 가능.** (단일 토큰 단순화는 priority-lock에서 기각 — 재생성 데이터 검증 후 재고.)
- **레버(ablation)**: K(1/2/3/5=AND폭) · 트리(AND vs AND/OR/chain) · distractor(0 vs 2-3) · alias on/off · NL 패러프레이즈 on/off.
- **staging + 게이트**:
  - **Step 1 = synth-only 학습 → synth held-out eval**: GO/NO-GO "grounding 깨끗하면 절차 학습되나". 1차 지표=**ACT-recall|충분게더 + STOP-recall 분리**(합성 50:50라 다수클래스 없음 → recall 분리가 핵심; over-refusal 붕괴=ACT-recall 0으로 검출). ordering-violation=3차 가드레일, BOTH는 분해 산출. (confound·sim-to-real 둘 다 합성서 제거.)
  - **Step 2 = synth 학습 → 실 bank eval**(Track A induction-fix 후): 이게 thesis 증명(전이, 재학습0). 게이트: bank BOTH > L0 baseline + **synth→bank 음성전이**(synth 높은데 bank 0)는 명시 실패조건.
  - **Step 3 (조건부) = synth+real co-train**: sim-to-real 보강.
- **리스크/caveat**: ① sim-to-real(NL 패러프레이즈+distractor+co-train로 완화) · ② degenerate 표면템플릿(distractor·alias·NL강제로 차단) · ③ synth→synth 전이는 thesis 증거로 **약함**(vocab만 swap) → thesis는 **Step 2**가 짊 · ④ novelty 무증가(slot-filling DST) → 합성은 *용량 프로브*이지 기여 아님(기여=학습된 required-set+전이).
- **선행근거**(§3.9 SFT 참조): Abbe inductive-scratchpad·Kim&Suzuki parity·Teaching-Arithmetic(전부 from-scratch 합성과제 = 합성 정당성) + slot-filling DST(gather-until-resolved)·PKS sensing-action·LLM-Modulo(self-verify 불가).

### Phase 1.5 = Rung 1.5 = ② DPO: 순서 preference (SFT 음성-부재 보강)
- **GT 쌍**(reward model 불요): **chosen** = 게더→올바른 분기(완전 궤적) / **rejected** = (a) **조기 act**(게더 미완 상태서 ACT/goal 호출), (b) should_T인데 게더→STOP(과잉거부), (c) should_F인데 act(과소거부).
- **DPO loss** → 조기 act·과/소거부를 **명시 dispreferred**. init = Rung1 SFT, KL to SFT(ref=frozen). = SFT positive-only가 못 주는 **음성 신호**를 GT-유도 대조로 직접 주입.
- **✅구현·단위검증**: `build_dpo_pairs.py`(readiness-gate SFT jsonl → {prompt,chosen,rejected}; rejected=조기ACT/과·소거부) + `dpo_train.py`(수동 DPO, policy=SFT-init LoRA·ref=frozen SFT·assistant-token logp DPO loss, trl-free). Exp-4e-dpo.

### Phase 2 = Rung 2 = ③ RFT (GRPO): 조기 ACT 패널티 + BOTH 보상
- **init = Rung1/1.5 어댑터**. 수동 GRPO 루프(`grpo_reward.py` 검증, trl 회피, KL to SFT).
- **reward = SOPBench rule evaluator**(결정론·judge無·무료): **+w_pass·BOTH(dirgraph∧goal 성공)** + w_proc·dirgraph진행(dense, sparse cold-start 구제) **− w_early·(조기 ACT 호출수)**(=게더 미완 상태 ACT/goal 직접 페널티), should_F면 **+올바른 STOP**(dual-axis gross).
- 조기 act 롤아웃→음성 advantage→억제(순서 위반 직접 페널티). outcome-only도 조기act=dirgraph실패→0(암묵 페널티); 명시 process 패널티는 dense·빠름.
- **✅구현**: `sopbench_reward.py`(reward=BOTH+dirgraph진행−early-ACT, dual-axis, GT-grounded·judge無; 자체테스트: gather-act 1.3>refuse 0.2>premature −0.4) + `grpo_train_sopbench.py`(수동 GRPO: group-norm advantage + −adv·logp + KL **update core ✅**; rollout assembly) + `_plan_v2` `SOPBENCH_RLLOG` 훅(planner step 로깅)·temp 샘플링. ⚠️**rollout-serving 오케스트레이션은 Rung2 진입시 원격검증**(task 격리). Exp-4f.
- **SFT(Rung1) vs RFT 역할(연구 확정)**: process supervision은 outcome 대비 **통계적 우위 없음**(알고리즘적 credit-assignment만) [Jia ICML'25, 2502.10581] → **Rung1 SFT 중간토큰 supervision이 primary**(rare-gate도 dense 신호); RFT는 보조(outcome-only RL은 게이트가 드물게 샘플돼 신호0일 때 실패 → process/SRL식 dense reward로 보완). 즉 **게이트는 SFT 분해로 먼저, RFT로 경계 다듬기**.

### Phase 3 = Rung 3 — 내재화: xattn ABox-memory + steering (직접 주입)
- 원 ★novelty(B5*). ABox를 프롬프트(토큰) 아닌 **xattn 메모리로 직접 주입**. TBox=xattn weights/ABox=메모리 M 스왑(토큰0·전이).
- **근거**: MetaTool basis-specificity(직접개입 specific, random/프롬프트 재현불가) + run_scripted(37≫4). **진입조건 = Rung1-2가 in-context 천장 근접 후**. Exp-4g.
- **★Rung3 변종 = gated/conditional 직접제어 (2026-06-01 추가)**: 작은 모델(7B)은 prompt로 강제해도 critical step서 계속 틀린 행동 교정 불가(prompt=간접). always-on steering은 유창성·일반능력 저하(**우리 Phase 2a steering 전부 null = 그 증거**). → **개입을 결정 지점에만 켜고 직후 끈다**(강점만 취함). (i) **게이팅 트리거**: Rung1 `ready=` 토큰을 syntactic anchor로(위치 기반=(a)순환성 회피) 또는 학습된 **condition vector**(CAST)로 "act-gather 경합" 탐지. (ii) **주입 신호**: 정적 steering vector는 generic "더 게더해"만(약함) → 강한 조건충족 신호는 **xattn으로 ABox 메모리 read**(내용 의존, B5*). (iii) **disable**: 추론시 additive 개입을 끄면 base weight 무손상·가역 — 단 KV-cache 잔존 = "**의도된 올바른 결정만 잔존**"으로 정직 표현(완전 0 아님; 필요시 logit-레벨 개입으로 완화).
- **선행 (전부 arXiv 검증, 2026-06-01)**: **ITI** [Li et al. NeurIPS'23, `2306.03341`] = 소수 attention head만 inference-time 개입(minimally invasive·strength로 truthful↔helpful tradeoff 조절 = 우리 부작용 우려와 동형). **★CAST 조건부 activation steering** [Lee&Padhi et al. 2024, `2409.05907`] = **condition vector 유사도를 스위치로 "조건 만족 시에만" steering 적용** = 우리 gated 아이디어의 **직접 선례**(트리거 (a) 학습형 해법). **CAA** [Rimsky et al. ACL'24, `2312.06681`] = 대조쌍 steering vector를 **모든 위치 가산**(=우리가 개선할 always-on 베이스라인)·finetune/prompt 위에서도 작동·능력 손실 최소. **ActAdd** [Turner et al. 2023, `2308.10248`] = 최적화 없는 정적 steering(off-target 보존=약한 정적 버전). **Function Vectors** [Todd et al. ICLR'24, `2310.15213`] = 소수 head가 task를 vector로 운반·robust 트리거 = "readiness/gather를 주입 vector로 인코딩" 가능성 근거. **RepE** [Zou et al. 2023, `2310.01405`] = 표현 읽기/제어 상위 틀.
- **★★CAST 정밀 분석 (전문 정독, 2026-06-01)** — 우리 gated의 가장 가까운 선례, 단 4가지가 갈림.
  - **메커니즘**: 두 벡터 모두 PCA·training-free. **behavior vector `v_l`**=대조쌍(refuse"Sorry I can't" vs comply"Sure!") 응답 **suffix 토큰** 활성 첫 주성분(레이어별). **condition vector `c_l`**=동일하나 프롬프트 **전체 토큰 평균** 활성(=입력 *부류* 표현). 규칙 `h' ← h + f(sim(h, proj_c h))·α·v`, `proj_c h=(c⊗c/c·c)h`, `f=1 if cos(h,proj_c h)>θ else 0`(레이어 따라 < 뒤집힘, tanh 안정화). **조건은 첫 forward(프롬프트) 1회 검사→발화 시 이후 *모든* 생성토큰에 지속**. θ·layer·방향=F1 grid-search, α 수동. 논리결합 OR/NOT(여집합)/동시 add-sub. 결과=유해거부 83-91%/무해 0-6%(무조건 91-96% 대비), 7-8모델, ~1h.
  - **우리와 갈리는 4점(=novelty 축)**: ① **입도**: CAST=프롬프트 1회(주제 분류) / 우리=**스텝별 재평가**(궤적 내 precond-충족 상태가 스텝마다 변함; `ready=` 위치서 매번). ② **시간 패턴**: CAST=onset 후 **지속** / 우리=**transient(단일 결정 스텝 후 disable)**. ③ **신호**: CAST=고정 1방향("refuse") / 우리 강버전=**내용 의존(ABox-xattn read, 어느 precond 미충족)**. ④ **결합**: 우리는 **ABox-swap 전이**와 묶음.
  - **★위험(CAST가 직접 경고)**: CAST는 "**의미적으로 유사한 범주 구분은 약함**" 명시 — 그런데 우리 "게더함 vs 못함"은 같은 도메인·도구, precond 개수만 다른 **극도로 유사한 상태**=정확히 CAST 약점 영역. 선형 condition vector가 conjunction 상태를 못 가를 위험. → **완화 = `ready=` 토큰 *syntactic 앵커*를 주 트리거(위치 기반, 선형분리성 불요), condition vector는 보조/주입내용**.
  - **★싼 1차 실험(xattn 투자 전)**: CAST 레시피 그대로(대조쌍→mean-center→레이어별 PCA, grid-search θ/layer/방향, few-hundred·~1h·training-free) → **readiness condition vector**(required 게더 vs 미게더 궤적) + **gather-don't-act behavior vector**(조기-act vs 적정-게더 연속) 추출, `ready=` 위치 게이트. **"gated > always-on" 가설을 값싸게 선검증** → 통과 시 xattn(강버전) 진입.
- **사전등록 ablation (효과는 실측으로만, 가능성≠효과)**: ① **ON-at-decision vs always-on vs OFF** → gated > always-on(Phase2a null 재현) > base 면 "강점만 취함" 입증. ② **disable 후 회귀 0** (개입 끈 스텝 perplexity·should_F base 동일). ③ **ABox 메모리 swap 전이** (M 교체만으로 held-out BOTH 유지 = TBox/ABox 분리 HT2). ④ **앵커 의존성** (ready 토큰 제거 시 게이팅 붕괴 = Rung1이 Rung3 enabler).

> **공정성/anti-cheat (전 rung 적용)**: **도구명 ALIAS 마스킹**(그래프 전체 일관) — 이름 암기 차단·NL↔설명 의미매칭 강제 = LODO 전이 타당성 게이트. **source3**(STATUS 정답지 미렌더, NL 정책만). 진짜 anti-cheat = alias ON + source3.

### §3.9 선행 연구 — 단계별 학습 사다리 (staged SFT→preference→RL) · 커리큘럼 〔2026-06-01 추가, 전부 arXiv 검증〕
> §3 기존 인용(Nye·Abbe·Kim&Suzuki·Feng·least-to-most)은 "readiness-게이트 sub-task가 *학습 가능한가*"(analogical, parity/arithmetic 증명)를 다룬다. **아래는 사다리 *구조 자체*(왜 SFT → preference → RL 순서로 쌓나)의 직접 LLM-post-training 선례** — analogical 아님.
- **SFT→선호→RL 정준 파이프라인(우리 ①②③의 직계 선례)**:
  - **InstructGPT** [Ouyang et al. 2022, `2203.02155`, NeurIPS'35]: SFT → reward model → PPO RL의 3단계 사후학습 원형. 우리 사다리의 골격.
  - **DPO** [Rafailov et al. 2023, `2305.18290`]: reward model·in-loop RL 없이 선호를 policy/ref log-ratio로 재매개화한 BCE 손실. = ② `dpo_train.py`가 구현한 방법의 원전.
  - **★Tülu 3** [Lambert et al. 2024, `2411.15124`]: open recipe **SFT → DPO → RLVR(RL with Verifiable Rewards)**. 우리 사다리 **①SFT → ②DPO → ③RFT(SOPBench rule reward)와 거의 동형** — RLVR의 "검증가능 보상"이 우리의 결정론 rule evaluator와 정확히 대응. 가장 직접적인 선례.
  - **★DeepSeek-R1** [DeepSeek-AI 2025, `2501.12948`]: 순수 RL(R1-Zero)은 가독성·언어혼합 붕괴 → **cold-start SFT + multi-stage 후 RL**로 해소. **"RL 전에 SFT 발판이 필요"**의 강한 증거 = 우리 "①② 먼저, ③로 경계 다듬기"를 직접 지지.
- **process vs outcome 감독(③ reward 설계 근거, 양면 제시·정직)**:
  - **Lightman et al. "Let's Verify Step by Step"** [2023, `2305.20050`, PRM800K]: MATH에서 **process 감독 > outcome 감독**(중간 step별 보상). 우리 dense dirgraph-progress·early-ACT 패널티(③)의 경험적 지지.
  - ⚠️**대척점(이미 §3에 인용)**: Jia ICML'25 [`2502.10581`]: process는 outcome 대비 **통계적 우위 없음**(알고리즘적 credit-assignment 이점만) → **①② SFT/DPO를 primary로, ③ RFT는 보조**(드문 게이트 신호0 구제)라는 우리 순서의 정당화. 두 결과를 함께 둠 = process가 *언제* 도움되는지(hard 추론·dense step 보상)는 경험적, *통계적 일반우위*는 아님.
- **커리큘럼 (easy→hard, 게이트 sub-task 우선)**:
  - **Bengio et al. "Curriculum Learning"** [2009, ICML]: 쉬운→복잡 순 제시 = continuation method(매끄러운 목표부터). **readiness 판정(required 다 게더?)은 act-결정보다 단순 sub-task** → 먼저 명시 supervise(①)하고 어려운 순서-결정을 나중(②③)에 다듬는 우리 분해의 근거.
- **반복 자기개선 사다리 (③ RFT가 자기 데이터로 부트스트랩, Rung2 옵션)**:
  - **STaR** [Zelikman et al. 2022, `2203.14465`]: 정답 rationale만 모아 FT→반복. **ReST-EM "Beyond Human Data"** [Singh et al. 2023, `2312.06585`; 원 ReST `2308.08998`]: generate→**검증가능 피드백으로 필터**→FT→반복(EM). = ③ GRPO 롤아웃을 rule evaluator로 필터해 재학습하는 우리 루프의 선례(reward hacking 없는 GT-grounded 자기학습).
- **정직 단서**: 이 선례들은 대형 모델(수십B~)·math/instruction 도메인 결과 → 7B LoRA + SOPBench tool-use 전이는 **레시피 차용**(단계 순서·검증가능 보상)이지 성능 보장 아님. 헤드라인은 우리 LODO BOTH 실측으로만 주장.

## §4. 성공 게이트 (사전등록)
- **G-SFT(Rung1, ① readiness-게이트)**: BOTH(dirgraph∩goal) 현 0-2 → **≥ 다수**(예 ≥15/48) + **조기 ACT율↓**(ready=false서 ACT 거의 0). 못 넘으면 ② DPO.
- **G-DPO(Rung1.5, ②)**: DPO가 SFT 대비 **조기 act율 추가 감소 + BOTH 상승**. 음성 신호 효과 확인.
- **G-RFT(Rung2, ③)**: RFT가 SFT/DPO BOTH 대비 **유의 상승** + early-ACT 페널티로 순서위반↓ + should_F gross 회귀 ≤기준.
- **G-전이(헤드라인)**: held-out bank LODO가 in-domain의 **≥70%** 회수, **재학습 0**.
- **G-ablation**: 빈/틀린 ABox → **붕괴**(온톨로지 실사용 증명).
- **G-xattn(Rung3 진입)**: Rung1-2 in-context 천장 확인 후, 직접주입이 프롬프트 잔여천장 상회 + ABox-ablation 붕괴.

## §5. 트랙 분업
- **Track A (우리, 7B/14B)**: Rung1-2 구현·파일럿·진단. 현재 bank LODO.
- **Coworker (32B/72B)**: 대형 모델 arm 매트릭스·alias·user_sim (별도 `COWORKER_EXPERIMENT_PLAN.md`). **32B 바닐라는 leaderboard 인용(재측정 금지).**
- **특허**: 별 트랙(중복 도구 SELECTION 내재화; MetaTool/ToolBench). 본 라인=계획·전이.

## §6. 인프라 불변식 (반복 사고 방지)
- 로컬 python=Store 스텁 → **모든 측정 리모트(rr.ps1)**, RC·scanned 확인 후만 "실측" 인용. **fabrication 금지**.
- `rr.ps1 pkill -f`가 자기 셸 self-match→SSH drop → **문자열 split**(`"x""y"`)로 회피. vLLM kill-9 잔여 `/dev/shm/vllm*` → GPU wedge → **shm 정리+GPU별 PID kill**. 2잡/48GB=OOM위험(solo 권장). eval=GPU별 PID-kill 격리.
- 배포=git pull(SFTP 금지). seka_env(py3.12, peft 0.19 있음=학습 가능). 어댑터명 `tbox_v2`=FC통과.

## §7. 문서 지도 (이 문서가 마스터; 나머지는 detail)
| 문서 | 역할 | 상태 |
|---|---|---|
| **이 문서** `EXPERIMENT_DESIGN.md` | **목표·순서·지표 권위본** | ★마스터 |
| `WORKFLOW_ONTOLOGY_DESIGN.md` | TBox/ABox 전체 스펙·planner L0/L1/L2·§9 LLM-in-loop·prior art | detail (개념 원본) |
| `TASK_CONSTRAINT_DESIGN.md` | should_T 병목 진단·게이트·§8.6 전수진단·§8.7 사다리 상세 | detail (Rung1-2 상세) |
| `GRPO_REWARD_DESIGN.md` | RFT reward 함수·GRPO 루프(Rung2 상세) | detail |
| `SOPBENCH_EXPERIMENT_RESULTS.md` | 모든 실측 결과(Exp-1~4) 누적 | 결과 권위본 |
| `COWORKER_EXPERIMENT_PLAN.md` | 32B/72B 분업 | detail (Track B) |
| `TASK_CONSTRAINT_{DESIGN_REVIEW,IMPL_REVIEW}.md` | 리뷰 라운드 | 참조 |
| `EXPERIMENT_DESIGN_v1_7_facet_rft.md`, `steering_paper/*` | 구 facet-rft/steering 라인 | ⚠️superseded(개념 참조만) |

**규칙: 목표·실험 순서 변경은 이 문서 §0~§4에서만. detail 문서는 구현 세부만.**
