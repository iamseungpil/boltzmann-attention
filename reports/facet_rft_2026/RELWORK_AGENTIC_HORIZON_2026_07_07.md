# Related work — 멀티스텝 에이전트 horizon 부하 (2026-07-07·DR#1 `wf_22bb5647`)

> RELWORK_LOAD_COT(2026-07-05·단일스텝 CoT/WM/self-consistency)의 **멀티스텝 horizon 확장**. 22소스·101클레임→
> 25검증(20확증·5기각). ★대부분 2025-26 preprint·다수 post-cutoff·미재현 → 수치=single-source 데이터점(상수 취급 금지).
> 유일 peer-reviewed=Press 2210.03350. 목적=부하-감축 아키텍처(LOAD_REDUCTION_ARCH_DESIGN) 선행근거 + Paper1 relwork.

---

## 0. 한 줄
2025-26 문헌이 **멀티스텝 실패 = execution-under-load(추론/지식과 별개)**를 직접 지지하고, **부하의 일부가
scale-invariant**(self-conditioning·compositionality gap)임을 실증. 최선 개입 = **결정론 decomposition(minimal-context
+ 외부 상태)** — 단 합성과제(Hanoi)서만 증명. **"소형+scaffold ≈ 대형-bare on 실제 tool-use" = 미확립 = 우리 whitespace.**

## 1. 확증된 핵심 (thesis 지지)

### 1.1 ★keystone — self-conditioning (2509.09677 Sinha/Geiping "Illusion of Diminishing Returns"·ICLR2026)
- **execution 격리**: 지식(in-context dict)+명시 plan 줘도 **per-step 정확도가 step 수 늘수록 하락** → 실패=추론 아닌 **실행**.
- **self-conditioning = scale-invariant**: "context에 자기 과거오류 있으면 더 틀림·**모델 크기 키워도 안 줄어듦**." 200B+도 degrade.
  단 **test-time thinking은 완화**. ⇒ **parameter-scale 축(무효) vs test-time-compute 축(유효) 분리** = "일부 부하 scale-invariant"의 정본.
- **단 scale은 horizon 길이를 삼**(clean isolated-execution서 marginal per-step↑가 지수적으로 긴 과제 완수로 compound). p^H 수치:
  99% per-step→90.4%@10·36.6%@100·0.004%@1000. **∴ scale=isolated-execution regime의 horizon만·오류진입後엔 test-time-compute.**
- ★우리 정합: **이게 "부하 두 원천"의 문헌판** — scale이 삼는 축(isolated per-step) vs 안 삼는 축(self-conditioning drift).

### 1.2 ★compositionality gap (Press 2210.03350·EMNLP2023·**peer-reviewed·유일**)
- **정의(우리 competence-performance 갭의 정본 名)**: "모든 하위문제는 격리서 맞히나 전체 합성은 실패하는 비율." GPT-3 family
  전반 **~40% flat = scale-invariant**(single-hop이 multi-hop보다 빨리 개선돼서). Yang 2402.16837 확증(hop-1만 scaling).
- **Self-Ask scaffold가 갭 축소 = 구조가 닫지 scale 아님** → 우리 thesis 직접 지지.
- caveat: GPT-3세대·2-hop QA(멀티툴 궤적 아님)·pre-CoT. o1급 미검. = 우리가 실제 tool-use로 확장.

### 1.3 super-linear 붕괴·capability≠reliability (2603.29231·post-cutoff)
- 오류 **양의 상관** → long-horizon pass@k가 **i.i.d. p^H보다 빨리** 붕괴("confused agent stays confused"·Ω(ε·e^{ρT})).
- **capability rank ≠ reliability rank**(GLM-4.5 Air short 94.9%→very-long 66.7%·1위→4위). coverage/조기종료 1%→25%.
- ⇒ **naive p^H는 하한**·실제는 더 나쁨(self-conditioning 정합). pass^1 무효·robust 지표 정당([[08]]).

### 1.4 gradual drift on REAL tool-use (2602.19008 "Capable but Unreliable"·Toolathlon·post-cutoff)
- **within-unit 자연실험**(같은 모델·같은 task): 성공 run이 실패 run보다 canonical path 준수 ↑(+0.060 Jaccard·p<0.0001·n=488)
  = 실패=**reliability/execution**이지 capability 아님(실제 벤치서 격리).
- **점진적 self-reinforcing drift**(단일 초기실수 아님): off-canonical 호출이 다음 off-canonical을 **+22.7pp**↑(전반부 50%는 성공·실패 구분무). = 우리 M3 loop·M4 drift의 기전.
- "**agent reliability는 capability scaling만으론 개선 불가**·lever=실행중 canonical-path로 drift 제약(결정론 개입)." = 우리 접근의 가장 근접한 출판 진술. (단 그 논문 자체 monitor+8.8pp는 검증서 기각.)

### 1.5 taxonomy (2604.11978 HORIZON·Dawn Song 외·3100궤적·post-cutoff)
- 7범주: Planning·**Catastrophic Forgetting**(제약이 context엔 있으나 미주목=우리 M5/M2)·**History Error Accumulation**(M3 loop)·
  Memory Limitation·Environment·Instruction·False-Assumption. = **우리 6기전과 가장 직접 대응하는 출판 taxonomy.**
- "model scaling alone unlikely to resolve dominant failures"·memory+forgetting=27.5%·breaking-region서 모델간 격차 좁혀짐.
  단 within-family scale ladder·scaffold ablation 無 → scale-invariance는 추론(우리 14B/32B가 그 공백 메움).

## 2. 개입 증거

### 2.1 ★MAKER (2511.09030 Meyerson/Cognizant·Nov2025) = 사용자 멀티에이전트 제안의 문헌판
- **maximal decomposition**: step당 **minimal-context microagent 1개 + 강제 state-output** → **per-step 오류가 horizon 늘어도 FLAT**
  (백만 step). 기전: "clear-instruction 실행"을 "insight 필요"에서 격리·minimal context가 attention dilution 제거·각 agent가
  next state 산출=외부화 상태가 tracking loss 방지. = **정확히 LOAD_REDUCTION_ARCH E1(격리)+E2(외부상태 dispatch).**
- ★★**STRONG caveat**: **Hanoi·곱셈(합성·동질 step·상태 자명 외부화)만**. **heterogeneous 실제 tool-use(tau2) 미증명.**
  flat per-step는 부분적으로 by-construction. ⇒ **우리가 실제 벤치로 옮기면 그게 novel.**

### 2.2 planner-executor·reflection
- COPE(2506.11578): small/large plan 교대. self-correction(2310.01798): **외부 피드백 없는 self-reflection은 개선 안 함**.
- ★self-conditioning 함의: **오류 context 위 naive reflection은 오히려 악화 가능** → 우리 개입은 반드시 **외부-결정론 피드백**
  (gate/verifier)이지 self-reflection 아님. 우리 gate=외부-결정론 ⇒ 정합.

## 3. ★★WHITESPACE = 우리 기여 (openQuestions·전부 미충족)
1. **"소형+결정론 scaffold ≈ 대형-bare on 실제 agentic multi-step tool-use" = 미확립.** MAKER=합성만·AppWorld scaffold
   (2604.11465)=통계 비유의(N=168)·memory-scaffold=0/10 개선. **tau2/Toolathlon급 controlled 비교 공백** = 우리 헤드라인.
2. **어느 기전이 scale-reducible/invariant인지 within-family scale ladder + 매칭 scaffold ablation으로 분리한 연구 無**
   → **우리 14B/32B same-scaffold**가 정확히 그 공백.
3. **isolated ~100% vs in-trajectory 정확도를 sub-skill별·history길이 함수로 정량한 연구 無**(Press=2-hop QA뿐) → **우리 operand
   probe(격리) vs full-run(궤적)**이 그 측정.
4. **coverage "do X for all Y"는 서술적 命名뿐**(early-failure·budget exhaustion)·**결정론 set-iteration이 model-frozen으로 닫나
   증거 無** → **우리 E2**가 그 open question 직격.
5. self-correction/critic 수렴조건 미해결(언제 drift 줄이고 언제 증폭하나).

## 4. 기각(support로 인용 금지·검증서 죽음)
- "memory scaffold가 reliability 개선한 적 없음"(과장·0-3) · "Mistral12B>Llama400B"(1-2) · "MAKER=scale이 per-step 무효"
  (overreach·0-3) · "mid-trajectory monitor +8.8pp"(0-3) · "Qwen3-8B scaffold≈33B"(비유의·N=168·1-2).

## 5. Paper1 편입 (신규 cite)
- §2 relwork "agentic horizon" 문단 신설: **2509.09677**(self-conditioning=scale-invariant execution·keystone)·
  **2210.03350**(compositionality gap=競/성 갭 名·peer-reviewed)·**2511.09030**(MAKER=decomposition)·**2604.11978**(HORIZON taxonomy)·
  **2602.19008**(Capable-but-Unreliable=실벤치 drift·"scaling만으론 불가").
- §3/§4 framing: 부하 두 원천(isolated-execution[scale가 삼] vs self-conditioning/composition[scale-invariant·구조가 삼]).
  우리 scaffold=MAKER의 실제-tool-use판 + coverage set-iteration(open Q4).
- 헤드라인 강화: whitespace #1·#2가 **정확히 우리 iso-scaffold×cross-scale on tau2** = moat 재확인([[46]]).
- ★규율: post-cutoff·single-preprint 다수 → 수치 인용 시 caveat·상수 취급 금지. peer-reviewed(Press)만 강하게.

---

## 6. ★2026-08-30 갱신 — AgenticQwen (arXiv:2604.21590) 정독 · **whitespace #1 부분 붕괴**

> 계기 = 사용자 지시 *"AgenticQwen 정독하라"*. 원문 전체(§1–5 + Limitations + Appendix A·B) 정독.
> Appendix C(배포 도구)·D(프롬프트 전문)는 훑기만 함. 텍스트 사본 = scratchpad `agq.txt`(9,615 words).

### 6.1 서지·자원
**arXiv:2604.21590v1 [cs.CL] · 2026-04-23 · Alibaba Group(Yuanjie Lyu·Chengyu Wang 교신 외).**
⚠**Qwen 팀이 아니라 Alibaba PAI 조직**이고, **Qwen3.8 계열과 무관한 별도 모델**이다(아래 §6.5).
- 가중치 공개: `alibaba-pai/AgenticQwen-8B`(8B) · `alibaba-pai/AgenticQwen-30B-A3B`(31B) · `AgenticQwen-Data`
- 데이터 합성 + RL 학습 코드 공개: `github.com/haruhi-sudo/data_synth_and_rl` · EasyDistill 통합

### 6.2 방법 (요지)
베이스 = Qwen3-8B / Qwen3-30B-A3B. **GRPO 다라운드 RL**, 총 학습데이터 **약 100K**.
Qwen3-235B가 **합성기·유저시뮬·툴시뮬·보상심판을 전부 겸함**(외부 API 의존 0 = 완전 로컬 파이프라인).
**에이전트 플라이휠 4단계**: ①선형 초기화(SynthAgent) → ②**행동트리 확장**(환경상태로 갈리는 분기 주입) →
③**분기→태스크 역변환**(각 분기를 필수 경로로 만들고 **agent instruction=SOP를 함께 갱신** — 축자:
*"The SOP is initially empty, but it expands as the behavior tree and task complexity grow"*) →
④**적대적 mock-user**(함정 분기를 골라 사용자 발화가 틀린 행동을 유도하도록 재작성).

### 6.3 수치 (TAU-2 = **Avg@4**·최종 환경상태 Exact Match / BFCL-V4 멀티턴)
| 모델 | Airline | Telecom | Retail | BFCL Base | MissFunc | MissParam | LongCtx | **Avg** |
|---|---|---|---|---|---|---|---|---|
| Qwen3-235B-A22B-Inst | 47.5 | 53.2 | 68.0 | 58.5 | 47.5 | 35.0 | 54.0 | **52.0** |
| Qwen3-30B-A3B-Inst | 32.0 | 31.6 | 55.3 | 47.0 | 14.0 | 28.0 | 45.5 | 36.2 |
| **Qwen3-32B** | **22.5** | **27.6** | **44.7** | 50.5 | 43.0 | 30.5 | 33.0 | 36.0 |
| Qwen3-8B | 14.5 | 7.9 | 31.6 | 35.5 | 35.0 | 20.5 | 21.5 | 23.8 |
| **AgenticQwen-8B** | 40.5 | **53.5** | 60.3 | 56.0 | 47.5 | 33.5 | 40.5 | **47.4** |
| **AgenticQwen-30B-A3B** | 42.0 | 52.6 | 60.5 | 60.0 | 52.0 | 29.0 | 55.5 | **50.2** |

산업 검색: WebWalker 45.0→52.5 · XBench 30.0→**47.0(+17.0)** · GAIA 37.3→41.7 (검색 데이터 **<10K**만 투입).
BFCL Memory **48.4로 235B(47.1) 추월**. 추론시간 355.6s→**344.1s** 단축.
**3라운드에서 포화** — 축자: *"performance already approaches that of the strong model used for synthetic data generation."*

### 6.4 ★§3 whitespace 에 대한 영향 — **#1 을 좁혀야 한다**
§3-#1 은 *"소형+결정론 scaffold ≈ 대형-bare on 실제 agentic multi-step tool-use = 미확립"* 이었다.
AgenticQwen 은 **scaffold 없이 학습만으로** τ²에서 그것을 사실상 보였다(8B 23.8→**47.4** = 235B 의 91%).
⇒ **"소형이 대형에 근접" 명제는 더 이상 우리 것이 아니다**([[41]] ToolOrchestra 에 이은 두 번째 선점·경로가 다름:
ToolOrchestra=frontier 위임 / AgenticQwen=**위임 없는 순수 학습**).

**#1 을 다음으로 대체한다**:
> **가중치를 바꾸지 않고**(전이 = ABox swap) **교사 상한 없이** **pass^k 신뢰성**을 사는 경로는 미확립이다.

### 6.5 생존한 화이트스페이스 (근거 있는 것만)
1. **★pass^k 부재.** 전 논문이 `Avg@4` 만 쓴다 — `pass^`·`pass@`·일관성·분산 grep 결과 **0건**.
   Avg@4 는 pass^1급이지 신뢰성 축이 아니다. **[[46]] pass^all-compliance crossover 온전히 생존.**
2. **교사 상한.** 3라운드에서 합성기(Qwen3-235B) 수준에 수렴 — **이 방법은 교사를 못 넘는다.** scaffold 는 그 상한이 없다.
3. **전이 = 재학습.** 새 도메인마다 100K 합성 + 3라운드 RL. 우리 [[05]] ABox-swap 은 **가중치 불변**. 비용 축 생존.
4. **가중치 고정 배포에서 무력** — 고객이 모델을 못 바꾸는 온프레미스에서 이 방법은 적용 불가.
5. **보상이 LLM 심판**(235B rubric)이다. 우리 [[69]] reward = DB 해시가 더 엄격.
6. **모델-계열 편향 자인** — 축자: *"we use Qwen models as the synthesizer, simulator, and evaluator… This may introduce model-family bias."*
7. **40K 컨텍스트가 그들 실패의 주원인**인데 **Qwen3.8-27B 는 262k** ⇒ 이 한계는 우리 base 에서 이미 없다.

### 6.6 ⚠우리 레버와 **동형**인 것 (선점·인용 의무)
- **[[21]] user-sim 허위주장 강건성** — 축자: *"The mock user may deliberately claim 'I should get cash compensation',
  even if they are a standard member. The agent must therefore verify membership status through tool queries."*
- **[[64]] 거부가 합법 작업을 삼키면 안 된다** — Appendix B.8 성공기준이 **양면**이다:
  ✓Pass = bio 갱신 성공 **AND** `submit_nomination()` 미호출 / ✗Fail = 위반 호출 **OR** bio 갱신 미실행.
  §B.9 축자: *"…**regardless of user claims about external verification**. **Legal profile updates (bio, honors) must
  still be completed as they are independent operations** not subject to nomination eligibility criteria."*
  ⇒ 우리가 C413/C414 에서 발견한 접힘-부작용을 이들은 **보상함수로** 설치했다.
- **[[49]] 열거 불완전** — 행동트리 분기 확장이 그 처방.
- **[[72]] A2 저작** — SOP 를 행동트리에서 **자동 생성**. 우리는 오프라인 수작업.

### 6.7 우리가 쓸 수 있는 것
- **Qwen3-32B 참조점**: airline 22.5 / retail 44.7. handoff §10.4b MUA-RL 앵커(airline 23.5 / retail 50.2)와
  **독립 2편 교차검증** — airline 일치(22.5↔23.5)·retail 5.5pp 차(하네스·메트릭 차이 의심).
- **AgenticQwen-30B-A3B 체크포인트 공개** ⇒ "학습 팔 vs 우리 scaffold 팔" A/B 를 **실물로** 돌릴 수 있다.
- `data_synth_and_rl` 의 행동트리→SOP 추출 = 우리 A2/A3 저작 자동화 후보. **경쟁자이자 도구.**

### 6.8 판정
**[[41]] 등급 = 부분 선점 · 명제 전환 필요.** [[00]] 의 *"작아도 된다"* 절반은 학습 경로로 선점됐다.
우리가 팔 수 있는 것은 **⒜가중치 불변 전이 ⒝교사 상한 없음 ⒞pass^k 신뢰성** 셋으로 좁혀진다.
셋 다 이 논문이 건드리지 않았고, **⒞는 논문에 축 자체가 없다**.
